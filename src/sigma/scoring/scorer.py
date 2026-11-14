"""Candidate scoring with a causal language model (Eq. (3), Appendix A.2).

Every candidate ``c = (c_1, ..., c_|c|)``, including its end-of-answer token, is scored by its
mean token log-probability under teacher forcing,

    s(c | x, g) = 1/|c| * sum_t log p(c_t | prompt(x, C, g), c_<t),

and all candidates of one view share the same prompt.  Two exact evaluation strategies are
provided:

``packed`` (default)
    The prompt is processed once per view.  The answer tokens of all candidates are appended
    to the same sequence; a block attention mask lets every candidate attend to the prompt and
    to its own earlier tokens only, and position ids restart at the answer boundary for every
    candidate.  This is numerically equivalent to scoring the candidates one by one, at the
    cost of ``|prompt| + sum_c (|c| - 1)`` token positions per view.
``rows``
    One row ``prompt + candidate`` per candidate with ordinary causal masking and right
    padding.  It is the reference implementation used to test the packed strategy and works
    with any causal language model.

Gradients flow through whatever the caller enables (``torch.no_grad`` for inference).  The
scorer never switches dropout on: the caller keeps the model in evaluation mode, as required
for the paired scorer (Appendix A.2).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import torch
from torch import Tensor

from sigma.guidance.render import render_prompt
from sigma.guidance.schema import Guidance
from sigma.scoring.encoders import PromptEncoder


class ContextLimitError(ValueError):
    """Raised when a prompt plus its longest candidate exceeds the context limit."""


@dataclass(frozen=True)
class CandidateTokens:
    """Tokenised candidate inventory in the fixed candidate order.

    ``token_ids[j]`` holds the answer tokens of candidate ``j`` *including* the final
    end-of-answer token, so every sequence has length at least two.
    """

    token_ids: tuple[tuple[int, ...], ...]

    def __post_init__(self) -> None:
        if any(len(ids) < 2 for ids in self.token_ids):
            raise ValueError("every candidate needs at least one token before end-of-answer")
        if len(set(self.token_ids)) != len(self.token_ids):
            raise ValueError("candidates must have distinct token sequences (Appendix A.1)")

    def __len__(self) -> int:
        return len(self.token_ids)

    @property
    def max_length(self) -> int:
        return max(len(ids) for ids in self.token_ids)


@dataclass
class ScorerStats:
    """Running cost counters (Appendix A.2 "Cost accounting", Appendix C.5)."""

    processed_tokens: int = 0
    forward_calls: int = 0
    scored_views: int = 0

    def reset(self) -> None:
        self.processed_tokens = self.forward_calls = self.scored_views = 0


@dataclass
class CandidateScorer:
    """Scores the complete candidate inventory under one or more prompts.

    Parameters
    ----------
    model:
        A ``transformers`` causal language model, optionally wrapped by PEFT.
    encoder:
        Prompt encoder defining the answer boundary and the end-of-answer token.
    strategy:
        ``"packed"`` or ``"rows"`` (see the module docstring).
    view_batch_size:
        Number of views (prompts) per forward pass in the packed strategy.
    candidate_chunk_size:
        Candidate microbatch size; ``None`` scores all candidates of a view together.
    max_context_tokens:
        Context limit (8,192 tokens in the paper, Appendix C.5); ``None`` disables the check.
    """

    model: torch.nn.Module
    encoder: PromptEncoder
    strategy: str = "packed"
    view_batch_size: int = 1
    candidate_chunk_size: int | None = None
    max_context_tokens: int | None = None
    stats: ScorerStats = field(default_factory=ScorerStats)

    def __post_init__(self) -> None:
        if self.strategy not in ("packed", "rows"):
            raise ValueError(f"unknown scoring strategy {self.strategy!r}")
        core = self.model.get_base_model() if hasattr(self.model, "get_base_model") else self.model
        self._backbone = getattr(core, core.base_model_prefix)
        self._head = core.get_output_embeddings()

    # ------------------------------------------------------------------ encoding
    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    def encode_candidates(self, candidates: Sequence[str]) -> CandidateTokens:
        """Tokenise every candidate at the answer boundary, with its end-of-answer token."""
        return CandidateTokens(tuple(tuple(self.encoder.encode_answer(c)) for c in candidates))

    def encode_view(
        self,
        guidance: Guidance,
        input_text: str,
        include_outputs: bool = True,
        candidates: tuple[str, ...] | None = None,
    ) -> list[int]:
        """Token ids of ``Render(x, C, g)`` up to the answer boundary."""
        fields = render_prompt(guidance, input_text, include_outputs, candidates)
        return self.encoder.encode_prompt(fields)

    # ------------------------------------------------------------------ scoring
    def score(
        self,
        prompts: Sequence[Sequence[int]],
        candidates: CandidateTokens,
        candidate_indices: Sequence[int] | None = None,
    ) -> Tensor:
        """Return ``s_g`` for every prompt: a tensor of shape ``(len(prompts), n_selected)``.

        ``candidate_indices`` restricts scoring to a subset of the inventory (used for
        candidate microbatching); by default all candidates are scored, in candidate order.
        """
        selected = (
            list(range(len(candidates))) if candidate_indices is None else list(candidate_indices)
        )
        tokens = [candidates.token_ids[j] for j in selected]
        longest = max(len(t) for t in tokens)
        for prompt in prompts:
            # The last input position of a candidate is |prompt| + |c| - 2.
            if self.max_context_tokens is not None and len(prompt) + longest - 1 > (
                self.max_context_tokens
            ):
                raise ContextLimitError(
                    f"prompt of {len(prompt)} tokens plus a {longest}-token answer exceeds the "
                    f"context limit of {self.max_context_tokens}; shorten the label descriptions"
                )
        chunk = self.candidate_chunk_size or len(tokens)
        columns: list[Tensor] = []
        for start in range(0, len(tokens), chunk):
            part = tokens[start : start + chunk]
            if self.strategy == "packed":
                rows = [
                    self._score_packed(prompts[b : b + self.view_batch_size], part)
                    for b in range(0, len(prompts), self.view_batch_size)
                ]
                columns.append(torch.cat(rows, dim=0))
            else:
                columns.append(torch.stack([self._score_rows(p, part) for p in prompts]))
        self.stats.scored_views += len(prompts)
        return torch.cat(columns, dim=1)

    def _hidden(self, input_ids: Tensor, attention_mask: Tensor, position_ids: Tensor) -> Tensor:
        self.stats.forward_calls += 1
        output = self._backbone(
            input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids
        )
        return output.last_hidden_state

    def _log_probs(self, hidden: Tensor) -> Tensor:
        """Token log-probabilities, computed in float32 for the selected positions only."""
        return torch.log_softmax(self._head(hidden).float(), dim=-1)

    def _score_packed(
        self, prompts: Sequence[Sequence[int]], tokens: Sequence[Sequence[int]]
    ) -> Tensor:
        device = self.device
        batch = len(prompts)
        # Candidate block shared by all views: inputs c_1..c_{L-1}, targets c_2..c_L.
        block_inputs = [t for ids in tokens for t in ids[:-1]]
        block_targets = [t for ids in tokens for t in ids[1:]]
        block_owner = [j for j, ids in enumerate(tokens) for _ in ids[:-1]]
        block_offset = [k for ids in tokens for k in range(len(ids) - 1)]
        block = len(block_inputs)
        lengths = torch.tensor([len(p) for p in prompts], device=device)
        total = int(lengths.max()) + block
        input_ids = torch.full((batch, total), self.encoder.pad_token_id, device=device)
        position_ids = torch.zeros((batch, total), dtype=torch.long, device=device)
        segment = torch.full((batch, total), -1, dtype=torch.long, device=device)
        owner = torch.tensor(block_owner, device=device)
        offset = torch.tensor(block_offset, device=device)
        block_ids = torch.tensor(block_inputs, device=device)
        for b, prompt in enumerate(prompts):
            p = len(prompt)
            input_ids[b, :p] = torch.tensor(list(prompt), device=device)
            input_ids[b, p : p + block] = block_ids
            position_ids[b, :p] = torch.arange(p, device=device)
            position_ids[b, p : p + block] = p + offset
            segment[b, :p] = 0
            segment[b, p : p + block] = owner + 1
        # Position i may attend to position k <= i if k is a prompt token or belongs to the
        # same candidate; padding positions only attend to themselves.
        causal = torch.ones((total, total), dtype=torch.bool, device=device).tril()
        same = segment[:, :, None] == segment[:, None, :]
        visible = (segment == 0)[:, None, :] | same
        allowed = causal[None] & visible & (segment >= 0)[:, None, :]
        allowed |= torch.eye(total, dtype=torch.bool, device=device)[None]
        dtype = next(self._backbone.parameters()).dtype
        mask = torch.zeros((batch, 1, total, total), dtype=dtype, device=device)
        mask.masked_fill_(~allowed[:, None], torch.finfo(dtype).min)
        self.stats.processed_tokens += int(lengths.sum()) + batch * block

        hidden = self._hidden(input_ids, mask, position_ids)
        rows = torch.arange(batch, device=device)
        first_tokens = torch.tensor([ids[0] for ids in tokens], device=device)
        first = self._log_probs(hidden[rows, lengths - 1])[:, first_tokens]  # (batch, n_cand)
        block_index = lengths[:, None] + torch.arange(block, device=device)[None]
        block_hidden = hidden[rows[:, None], block_index]  # (batch, block, hidden)
        targets = torch.tensor(block_targets, device=device).expand(batch, block)
        rest = self._log_probs(block_hidden).gather(-1, targets[..., None]).squeeze(-1)
        sums = first.index_add(1, owner, rest)
        answer_lengths = torch.tensor([len(ids) for ids in tokens], device=device)
        return sums / answer_lengths

    def _score_rows(self, prompt: Sequence[int], tokens: Sequence[Sequence[int]]) -> Tensor:
        device = self.device
        p, n_cand = len(prompt), len(tokens)
        width = p + max(len(ids) for ids in tokens) - 1
        input_ids = torch.full((n_cand, width), self.encoder.pad_token_id, device=device)
        attention = torch.zeros((n_cand, width), dtype=torch.long, device=device)
        position_ids = torch.arange(width, device=device).expand(n_cand, width)
        prompt_ids = torch.tensor(list(prompt), device=device)
        for j, ids in enumerate(tokens):
            input_ids[j, :p] = prompt_ids
            input_ids[j, p : p + len(ids) - 1] = torch.tensor(ids[:-1], device=device)
            attention[j, : p + len(ids) - 1] = 1
        self.stats.processed_tokens += int(attention.sum())
        hidden = self._hidden(input_ids, attention, position_ids)
        log_probs = self._log_probs(hidden[:, p - 1 :])  # predictions for answer positions
        scores = []
        for j, ids in enumerate(tokens):
            steps = torch.arange(len(ids), device=device)
            scores.append(log_probs[j, steps, torch.tensor(ids, device=device)].mean())
        return torch.stack(scores)
