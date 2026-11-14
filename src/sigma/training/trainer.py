"""The SIGMA optimisation procedure (Appendix A.2) and its single-view / paired controls.

One optimiser step

1. samples a minibatch of ``B`` original instances uniformly with replacement,
2. pairs every instance with an edited guidance view: a sampled binding swap on odd steps and
   the representation edit (reversal) on even steps (steps are numbered from 1), with the
   identity replacement for empty edit sets and invalid constructions,
3. scores the complete candidate inventory under both views, reusing the original score
   vector for an identity pair,
4. averages Eq. (5) over the minibatch and updates the trainable (LoRA) parameters.

Checkpoints are evaluated on a fixed development pool at five approximately equally spaced
even-numbered steps including the last one; the best checkpoint (earliest on ties) is kept.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np
import torch

from sigma import losses
from sigma.config import OptimConfig, ScoringConfig
from sigma.data.fewshot import Example
from sigma.guidance.edits import AppliedEdit, BindingSwap, IdentityEdit, apply_edit, scheduled_edit
from sigma.guidance.schema import TaskRecord
from sigma.methods import MethodSpec
from sigma.models import load_trainable_state, trainable_parameters, trainable_state
from sigma.scoring.scorer import CandidateScorer
from sigma.seeding import stream

logger = logging.getLogger(__name__)


def num_training_steps(num_examples: int, batch_size: int, passes: float = 10.0) -> int:
    """``T = 2 * ceil(passes * n / (2 B))``: about ``passes`` passes and an even step count."""
    return 2 * math.ceil(passes * num_examples / (2 * batch_size))


def checkpoint_steps(total_steps: int, num_checkpoints: int = 5) -> list[int]:
    """Approximately equally spaced even-numbered steps that include the final step."""
    steps = {
        min(total_steps, 2 * max(1, round(k * total_steps / (2 * num_checkpoints))))
        for k in range(1, num_checkpoints + 1)
    }
    return sorted(steps | {total_steps})


def learning_rate_factor(step_index: int, total_steps: int, warmup_ratio: float) -> float:
    """Linear warm-up over ``ceil(warmup_ratio * T)`` steps followed by cosine decay to zero.

    ``step_index`` is zero-based: the factor multiplies the learning rate of step
    ``step_index + 1``.
    """
    warmup = math.ceil(warmup_ratio * total_steps)
    if step_index < warmup:
        return (step_index + 1) / warmup
    progress = (step_index - warmup) / max(1, total_steps - warmup)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


@dataclass
class StepPlan:
    """The views and targets of one minibatch.

    ``prompts`` holds each distinct view once; ``base_view[i]`` and ``edit_view[i]`` index
    into it.  For an identity pair both indices are equal, so the same score vector (and its
    computation graph) is used in both loss positions.
    """

    prompts: list[list[int]]
    base_view: list[int]
    edit_view: list[int]
    permutation: torch.Tensor  # (B, m)
    target_base: torch.Tensor  # (B,)
    target_edit: torch.Tensor  # (B,)
    edits: list[AppliedEdit]


@dataclass
class InterventionCounts:
    """Intervention coverage counters (Appendix A.1, "Intervention coverage").

    The empirical target-change rate is ``changed / attempted``; it is also tracked for
    binding steps alone.  Identity replacements share the same denominator.
    """

    attempted: int = 0
    changed: int = 0
    binding_attempted: int = 0
    binding_changed: int = 0
    identity_pairs: int = 0
    identity_replacements: int = 0

    def update(self, plan: StepPlan) -> None:
        for edit, y, y_edit in zip(plan.edits, plan.target_base, plan.target_edit, strict=True):
            changed = int(y_edit != y)
            binding = isinstance(edit.edit, BindingSwap | IdentityEdit)
            self.attempted += 1
            self.changed += changed
            self.binding_attempted += int(binding)
            self.binding_changed += int(binding and changed)
            self.identity_pairs += int(edit.is_identity_pair)
            self.identity_replacements += int(edit.replaced_by_identity)


@dataclass
class TrainResult:
    """Outcome of one training run."""

    total_steps: int
    best_step: int
    best_dev_accuracy: float
    dev_curve: list[tuple[int, float]]
    history: list[dict[str, float]]
    counts: InterventionCounts
    training_tokens: int
    best_state: dict[str, torch.Tensor] = field(repr=False, default_factory=dict)


class Trainer:
    """Trains one method of :mod:`sigma.methods` on a fixed labeled subset.

    Parameters
    ----------
    model:
        Language model whose trainable parameters are optimised (LoRA adapters in the paper).
    scorer:
        Candidate scorer bound to ``model``.
    task:
        Task record that defines ``g0`` and the candidate inventory seen during training.
    examples:
        The labeled training subset (``n = b * m`` original instances).
    method:
        Training recipe (views, consistency term, coefficient, edit schedule).
    dev_accuracy:
        Callable returning the development-pool accuracy of the current parameters; ``None``
        disables checkpoint selection (the final parameters are kept).
    include_outputs:
        Render output bindings into the guidance (False for the SIGMA-F classifier).
    """

    def __init__(
        self,
        model: torch.nn.Module,
        scorer: CandidateScorer,
        task: TaskRecord,
        examples: Sequence[Example],
        method: MethodSpec,
        optim: OptimConfig,
        scoring: ScoringConfig,
        seed: int,
        dev_accuracy: Callable[[], float] | None = None,
        num_checkpoints: int = 5,
        include_outputs: bool = True,
        log_every: int = 50,
    ) -> None:
        if not method.trainable:
            raise ValueError(f"method {method.name!r} has nothing to train")
        self.model, self.scorer, self.task = model, scorer, task
        self.examples = list(examples)
        self.method, self.optim, self.scoring = method, optim, scoring
        self.dev_accuracy = dev_accuracy
        self.include_outputs = include_outputs
        self.log_every = log_every
        self._base = task.base_guidance()
        self._candidates = scorer.encode_candidates(task.candidates)
        self._targets = [self._base.target_index(e.label) for e in self.examples]
        self._batch_rng = stream(seed, "minibatches")
        self._edit_rng = stream(seed, "train-edits")
        self._prompt_cache: dict[tuple[int, tuple], list[int]] = {}
        self.total_steps = self._resolve_total_steps()
        self._checkpoints = set(checkpoint_steps(self.total_steps, num_checkpoints))
        self.parameters = trainable_parameters(model)
        self.optimizer = torch.optim.AdamW(
            self.parameters,
            lr=optim.lr,
            betas=(optim.betas[0], optim.betas[1]),
            weight_decay=optim.weight_decay,
        )

    # ------------------------------------------------------------------ schedule
    def _resolve_total_steps(self) -> int:
        optim = self.optim
        if optim.max_steps is not None:
            return 2 * math.ceil(optim.max_steps / 2)
        if self.method.token_matched:
            if optim.token_budget is None:
                raise ValueError("token-matched SFT needs optim.token_budget (Table 11)")
            passes = 2 if self.scoring.gradient_mode == "two_pass" else 1
            lengths = [len(self._prompt(i, None)) for i in range(len(self.examples))]
            block = sum(len(ids) - 1 for ids in self._candidates.token_ids)
            per_step = optim.batch_size * passes * (float(np.mean(lengths)) + block)
            return 2 * math.ceil(optim.token_budget / (2 * per_step))
        return num_training_steps(len(self.examples), optim.batch_size, optim.passes)

    # ------------------------------------------------------------------ views
    def _prompt(self, index: int, edit: AppliedEdit | None) -> list[int]:
        """Token ids of ``Render(x_i, C, g)``; base and reversed views are cached."""
        guidance = self._base if edit is None else edit.guidance
        cacheable = edit is None or not isinstance(edit.edit, BindingSwap)
        key = (index, guidance.key())
        if cacheable and key in self._prompt_cache:
            return self._prompt_cache[key]
        prompt = self.scorer.encode_view(guidance, self.examples[index].text, self.include_outputs)
        if cacheable:
            self._prompt_cache[key] = prompt
        return prompt

    def plan_step(self, step: int) -> StepPlan:
        """Sample the minibatch of optimiser step ``step`` (numbered from 1) and its views."""
        m = self.task.num_candidates
        indices = self._batch_rng.integers(len(self.examples), size=self.optim.batch_size)
        prompts: list[list[int]] = []
        base_view, edit_view, perms, y_base, y_edit, edits = [], [], [], [], [], []
        for index in indices.tolist():
            if self.method.views == "single":
                applied = apply_edit(self._base, IdentityEdit())
            else:
                edit = scheduled_edit(self.task, step, self._edit_rng, self.method.edit_schedule)
                applied = apply_edit(self._base, edit)
            target = self._targets[index]
            prompts.append(self._prompt(index, None))
            base_view.append(len(prompts) - 1)
            if applied.is_identity_pair:
                edit_view.append(base_view[-1])
            else:
                prompts.append(self._prompt(index, applied))
                edit_view.append(len(prompts) - 1)
            perms.append(applied.permutation)
            y_base.append(target)
            y_edit.append(applied.permutation[target])
            edits.append(applied)
        device = self.scorer.device
        return StepPlan(
            prompts=prompts,
            base_view=base_view,
            edit_view=edit_view,
            permutation=torch.tensor(perms, dtype=torch.long, device=device).view(-1, m),
            target_base=torch.tensor(y_base, dtype=torch.long, device=device),
            target_edit=torch.tensor(y_edit, dtype=torch.long, device=device),
            edits=edits,
        )

    # ------------------------------------------------------------------ objective
    def loss_from_scores(self, scores: torch.Tensor, plan: StepPlan) -> losses.PairedLossOutput:
        """Minibatch objective as a function of the score vectors of all distinct views."""
        scores_base = scores[plan.base_view]
        if self.method.views == "single":
            loss = losses.single_view_loss(scores_base, plan.target_base)
            zero = torch.zeros((), device=scores.device)
            return losses.PairedLossOutput(loss, loss.detach(), loss.detach(), zero)
        return losses.paired_loss(
            scores_base,
            scores[plan.edit_view],
            plan.permutation,
            plan.target_base,
            plan.target_edit,
            coefficient=self.method.coefficient,
            consistency=self.method.consistency,
        )

    def backward(self, plan: StepPlan) -> losses.PairedLossOutput:
        """Accumulate the gradient of the minibatch objective in the trainable parameters.

        ``direct`` back-propagates through all views at once.  ``two_pass`` is the
        memory-bounded procedure of Appendix C.5: all scores are first computed without
        graphs, the loss is differentiated with respect to the scores, and the views (and
        candidate microbatches) are then recomputed one at a time while accumulating their
        score-weighted gradients.  With fixed parameters and dropout disabled both modes give
        the same gradient.
        """
        if self.scoring.gradient_mode == "direct":
            output = self.loss_from_scores(self.scorer.score(plan.prompts, self._candidates), plan)
            output.loss.backward()
            return output
        if self.scoring.gradient_mode != "two_pass":
            raise ValueError(f"unknown gradient mode {self.scoring.gradient_mode!r}")
        with torch.no_grad():
            detached = self.scorer.score(plan.prompts, self._candidates)
        leaf = detached.clone().requires_grad_(True)
        output = self.loss_from_scores(leaf, plan)
        (score_grad,) = torch.autograd.grad(output.loss, leaf)
        views = self.scorer.view_batch_size
        chunk = self.scorer.candidate_chunk_size or len(self._candidates)
        for start in range(0, len(plan.prompts), views):
            for first in range(0, len(self._candidates), chunk):
                columns = list(range(first, min(first + chunk, len(self._candidates))))
                weights = score_grad[start : start + views][:, columns]
                if not torch.any(weights != 0):
                    continue
                scores = self.scorer.score(
                    plan.prompts[start : start + views], self._candidates, columns
                )
                (scores * weights).sum().backward()
        return output

    # ------------------------------------------------------------------ loop
    def fit(self) -> TrainResult:
        """Run the full schedule and restore the best development checkpoint."""
        self.model.eval()  # dropout stays disabled in the paired scorer (Appendix A.2)
        counts = InterventionCounts()
        history: list[dict[str, float]] = []
        dev_curve: list[tuple[int, float]] = []
        best_accuracy, best_step, best_state = -1.0, 0, {}
        tokens_before = self.scorer.stats.processed_tokens
        training_tokens = 0
        for step in range(1, self.total_steps + 1):
            factor = learning_rate_factor(step - 1, self.total_steps, self.optim.warmup_ratio)
            for group in self.optimizer.param_groups:
                group["lr"] = self.optim.lr * factor
            plan = self.plan_step(step)
            if self.method.views == "paired":
                counts.update(plan)
            self.optimizer.zero_grad(set_to_none=True)
            output = self.backward(plan)
            if self.optim.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.parameters, self.optim.grad_clip)
            self.optimizer.step()
            record = {
                "step": step,
                "loss": float(output.loss.detach()),
                "ce_base": float(output.ce_base.detach()),
                "ce_edit": float(output.ce_edit.detach()),
                "consistency": float(output.consistency.detach()),
                "lr": self.optim.lr * factor,
            }
            history.append(record)
            if step % self.log_every == 0 or step == self.total_steps:
                logger.info(
                    "step %d/%d loss %.4f ce %.4f/%.4f consistency %.5f",
                    step,
                    self.total_steps,
                    record["loss"],
                    record["ce_base"],
                    record["ce_edit"],
                    record["consistency"],
                )
            if self.dev_accuracy is not None and step in self._checkpoints:
                training_tokens += self.scorer.stats.processed_tokens - tokens_before
                accuracy = float(self.dev_accuracy())
                tokens_before = self.scorer.stats.processed_tokens
                dev_curve.append((step, accuracy))
                logger.info("checkpoint at step %d: development accuracy %.4f", step, accuracy)
                if accuracy > best_accuracy:  # strict: ties keep the earlier checkpoint
                    best_accuracy, best_step = accuracy, step
                    best_state = trainable_state(self.model)
        training_tokens += self.scorer.stats.processed_tokens - tokens_before
        if self.dev_accuracy is None:
            best_step, best_accuracy = self.total_steps, float("nan")
            best_state = trainable_state(self.model)
        else:
            load_trainable_state(self.model, best_state)
        return TrainResult(
            total_steps=self.total_steps,
            best_step=best_step,
            best_dev_accuracy=best_accuracy,
            dev_curve=dev_curve,
            history=history,
            counts=counts,
            training_tokens=training_tokens,
            best_state=best_state,
        )
