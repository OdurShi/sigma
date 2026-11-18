"""Code of the supplementary experiments: broader edits (E01), path residuals (E02), training
time and time caps (E03, E04, E06), the PBM-style control (E07) and global rebinding (E08)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from sigma import losses
from sigma.config import OptimConfig, ScoringConfig
from sigma.data import Example
from sigma.evaluation.aggregate import summarize
from sigma.evaluation.diagnostics import path_residual_summary, trajectory_diagnostic
from sigma.evaluation.evaluate import evaluate
from sigma.evaluation.protocol import CROSS_GROUP, RANDOM_ORDER, EvalPlan, ViewPlanner
from sigma.guidance import (
    CrossGroupSwap,
    GlobalRebinding,
    RandomOrder,
    apply_edit,
    sample_cross_group_swap,
    scheduled_edit,
)
from sigma.guidance.schema import TaskRecordError, with_descriptions
from sigma.methods import METHODS
from sigma.training import Trainer


class LookupPredictor:
    """Reads the current binding from the guidance (exactly equivariant)."""

    tie_by_label = False

    def __init__(self, task, belief):
        self.task, self.belief = task, belief

    def score_views(self, views):
        scores = np.zeros((len(views), self.task.num_candidates))
        for row, (text, guidance) in enumerate(views):
            scores[row, guidance.target_index(self.belief[text])] = 4.0
        return scores


class MemorizingPredictor(LookupPredictor):
    """Ignores the current guidance and always answers with the base binding."""

    def score_views(self, views):
        base = self.task.base_guidance()
        return super().score_views([(text, base) for text, _ in views])


EXAMPLES = [Example(f"utterance {i}", label) for i, label in enumerate(["refund", "login"] * 4)]
PLAN = EvalPlan(2, (2,), 4, transfer=(RANDOM_ORDER, CROSS_GROUP))


def test_cross_group_swap_exchanges_outputs_across_parents(toy_task):
    base = toy_task.base_guidance()
    applied = apply_edit(base, CrossGroupSwap("refund", "login"))
    assert not applied.replaced_by_identity
    assert applied.guidance.binding["refund"] == base.binding["login"]
    assert applied.guidance.binding["login"] == base.binding["refund"]
    assert applied.guidance.child_order == base.child_order
    target = base.target_index("refund")
    assert applied.permutation[target] == base.target_index("login")
    # Siblings are not a cross-group pair: the construction falls back to the identity.
    assert apply_edit(base, CrossGroupSwap("refund", "invoice")).replaced_by_identity
    rng = np.random.default_rng(0)
    for _ in range(20):
        edit = sample_cross_group_swap(toy_task, rng)
        assert toy_task.parent_of(edit.leaf_a) != toy_task.parent_of(edit.leaf_b)


def test_random_order_keeps_bindings_and_is_deterministic(toy_task):
    base = toy_task.base_guidance()
    applied = apply_edit(base, RandomOrder(3))
    assert not applied.replaced_by_identity
    assert applied.permutation == tuple(range(toy_task.num_candidates))
    assert applied.guidance.binding == base.binding
    assert {v: sorted(k) for v, k in applied.guidance.child_order.items()} == {
        v: sorted(k) for v, k in base.child_order.items()
    }
    assert apply_edit(base, RandomOrder(3)).guidance.key() == applied.guidance.key()
    orders = {apply_edit(base, RandomOrder(seed)).guidance.key() for seed in range(20)}
    assert len(orders) > 2


def test_global_rebinding_is_a_label_independent_bijection(toy_task):
    base = toy_task.base_guidance()
    applied = apply_edit(base, GlobalRebinding(11))
    assert not applied.replaced_by_identity
    assert sorted(applied.guidance.binding.values()) == list(toy_task.candidates)
    for label in toy_task.labels:
        image = applied.permutation[base.target_index(label)]
        assert applied.guidance.target_index(label) == image
    edit = scheduled_edit(toy_task, 1, np.random.default_rng(0), "global")
    assert isinstance(edit, GlobalRebinding)


def test_paraphrased_descriptions_change_only_the_text(toy_task):
    paraphrased = with_descriptions(toy_task, {"refund": "asks for a reimbursement"})
    assert paraphrased.nodes["refund"].description == "asks for a reimbursement"
    assert paraphrased.nodes["login"].description == toy_task.nodes["login"].description
    assert paraphrased.binding == toy_task.binding and paraphrased.labels == toy_task.labels
    with pytest.raises(TaskRecordError):
        with_descriptions(toy_task, {"unknown": "x"})


def test_transfer_conditions_extend_the_plan_without_touching_comp(toy_task):
    plain = EvalPlan(2, (2,), 4)
    assert PLAN.columns()[: len(plain.columns())] == plain.columns()
    assert [c for c, _ in PLAN.columns()[len(plain.columns()) :]] == ["order:k2"] * 4 + [
        "cross:k2"
    ] * 4
    with pytest.raises(ValueError):
        EvalPlan(transfer=("unknown",))
    views = ViewPlanner(toy_task, PLAN, seed=1).views(0)
    standard = ViewPlanner(toy_task, plain, seed=1).views(0)
    assert [v.guidance.key() for v in views[: len(standard)]] == [
        v.guidance.key() for v in standard
    ]

    belief = {e.text: e.label for e in EXAMPLES}
    exact = summarize(evaluate(LookupPredictor(toy_task, belief), toy_task, EXAMPLES, PLAN, 1, 0))
    reference = summarize(
        evaluate(LookupPredictor(toy_task, belief), toy_task, EXAMPLES, plain, 1, 0)
    )
    assert exact["composition"] == reference["composition"]
    for family in (RANDOM_ORDER, CROSS_GROUP):
        assert exact["transfer"][family]["accuracy"] == 100.0
    memorised = summarize(
        evaluate(MemorizingPredictor(toy_task, belief), toy_task, EXAMPLES, PLAN, 1, 0)
    )
    assert memorised["transfer"][CROSS_GROUP]["num_changed"] > 0
    assert memorised["transfer"][CROSS_GROUP]["changed_accuracy"] == 0.0
    assert "transfer" not in reference


def test_path_residual_summary_groups_by_depth():
    identity = (0, 1, 2)
    flat = np.array([[2.0, 0.0, -2.0]] * 3)  # no residual, correct
    drift = np.array([[2.0, 0.0, -2.0], [1.0, 1.0, -2.0]])  # one step, residual sqrt(2)
    summary = path_residual_summary(
        [
            trajectory_diagnostic(flat, [identity, identity], 0),
            trajectory_diagnostic(drift, [identity], 2),
        ]
    )
    assert summary[2] == {"count": 1, "median_residual": 0.0, "accuracy": 100.0}
    assert summary[1]["median_residual"] == pytest.approx(np.sqrt(2.0))
    assert summary[1]["accuracy"] == 0.0


def test_pbm_consistency_vanishes_only_for_transported_point_masses():
    base = torch.tensor([[3.0, 1.0, -1.0]])
    perm = torch.tensor([[1, 0, 2]])
    transported = losses.transport(base, perm)
    aligned = losses.pbm_consistency(base, transported, perm)
    entropy = -(torch.softmax(base, -1) * torch.log_softmax(base, -1)).sum(-1)
    assert torch.allclose(aligned, entropy)  # zero dispersion: only the entropy term is left
    # A common offset does not change log-probabilities, a different shape does.
    assert torch.allclose(losses.pbm_consistency(base, transported + 5.0, perm), entropy)
    assert losses.pbm_consistency(base, base, perm).item() > aligned.item()
    output = losses.paired_loss(
        base, transported, perm, torch.tensor([0]), torch.tensor([1]), 1.0, "pbm"
    )
    assert torch.allclose(output.consistency, aligned.mean())


def _trainer(scorer, task, method, **optim):
    for parameter in scorer.model.parameters():
        parameter.requires_grad_(True)
    examples = [Example("please give my money back now", "refund"), Example("log in", "login")]
    settings = {"batch_size": 2, "max_steps": 6, "lr": 1e-3} | optim
    return Trainer(
        scorer.model,
        scorer,
        task,
        examples,
        METHODS[method],
        OptimConfig(**settings),
        ScoringConfig(gradient_mode="direct", max_context_tokens=None),
        seed=0,
    )


def test_training_time_is_recorded_and_the_cap_stops_the_run(tiny_scorer, toy_task):
    full = _trainer(tiny_scorer, toy_task, "sigma").fit()
    assert full.steps_run == full.total_steps == 6
    assert full.training_seconds > 0
    assert full.gpu_hours == pytest.approx(full.training_seconds / 3600.0)

    calls = []
    capped = _trainer(tiny_scorer, toy_task, "sigma", max_gpu_hours=1e-12, num_gpus=4)
    capped.dev_accuracy = lambda: calls.append(1) or 0.5
    result = capped.fit()
    assert result.steps_run == 1 and len(result.history) == 1
    assert result.dev_curve == [(1, 0.5)] and result.best_step == 1  # evaluated when stopped
    assert result.gpu_hours == pytest.approx(4 * result.training_seconds / 3600.0)


@pytest.mark.parametrize("method", ["pbm_adapted", "global_aug_paired", "global_aug_sigma"])
def test_supplementary_methods_train(tiny_scorer, toy_task, method):
    trainer = _trainer(tiny_scorer, toy_task, method, max_steps=2)
    plan = trainer.plan_step(1)
    if METHODS[method].edit_schedule == "global":
        assert all(isinstance(applied.edit, GlobalRebinding) for applied in plan.edits)
    result = trainer.fit()
    assert all(np.isfinite(h["loss"]) for h in result.history)
