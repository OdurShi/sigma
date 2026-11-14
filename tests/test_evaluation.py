"""Evaluation protocol and aggregation (Appendix C.2, C.3, C.6) with model-free predictors."""

from __future__ import annotations

import numpy as np
import pytest

from sigma.data import Example
from sigma.evaluation.aggregate import per_instance_scores, summarize, view_table
from sigma.evaluation.evaluate import build_dev_pool, evaluate, predict, score_unique
from sigma.evaluation.protocol import BASE, BINDING, REVERSAL, EvalPlan, ViewPlanner
from sigma.evaluation.records import EvalRecords
from sigma.guidance import BindingSwap, Reversal, apply_edit

PLAN = EvalPlan(binding_draws=4, composition_depths=(2, 4), composition_draws=5)


class LookupPredictor:
    """Reads the binding from the guidance: an exactly equivariant predictor.

    ``accuracy_map`` maps an input text to the semantic label the predictor believes in.
    """

    tie_by_label = False

    def __init__(self, task, belief):
        self.task, self.belief, self.calls = task, belief, 0

    def score_views(self, views):
        self.calls += len(views)
        scores = np.zeros((len(views), self.task.num_candidates))
        for row, (text, guidance) in enumerate(views):
            for c, label_index in enumerate(guidance.label_of_candidate()):
                label = self.task.labels[label_index]
                scores[row, c] = 4.0 if label == self.belief[text] else -0.1 * label_index
        return scores


class MemorizingPredictor(LookupPredictor):
    """Ignores the current guidance and always answers with the base binding."""

    def score_views(self, views):
        base = self.task.base_guidance()
        return super().score_views([(text, base) for text, _ in views])


@pytest.fixture
def examples(toy_task):
    labels = ["refund", "invoice", "charge", "login", "closure", "other"] * 4
    return [Example(f"utterance {i}", label) for i, label in enumerate(labels)]


@pytest.fixture
def belief(examples):
    truth = {e.text: e.label for e in examples}
    truth[examples[0].text] = "invoice"  # one semantic mistake (gold: refund)
    return truth


def test_plan_columns_follow_the_paper_protocol(toy_task):
    default = EvalPlan()
    columns = default.columns()
    assert len(columns) == 1 + 1 + 10 + 6 * 10 == 72
    assert default.composition_conditions() == ["k2", "k2+R", "k4", "k4+R", "k8", "k8+R"]
    assert [c for c, _ in columns[:2]] == [BASE, REVERSAL]
    assert sum(c == BINDING for c, _ in columns) == 10


def test_views_are_label_independent_and_shared_across_methods(toy_task):
    first = ViewPlanner(toy_task, PLAN, seed=13).views(7)
    again = ViewPlanner(toy_task, PLAN, seed=13).views(7)
    assert [v.guidance.key() for v in first] == [v.guidance.key() for v in again]
    other_instance = ViewPlanner(toy_task, PLAN, seed=13).views(8)
    other_seed = ViewPlanner(toy_task, PLAN, seed=21).views(7)
    assert [v.guidance.key() for v in first] != [v.guidance.key() for v in other_instance]
    assert [v.guidance.key() for v in first] != [v.guidance.key() for v in other_seed]
    assert [(v.condition, v.draw) for v in first] == PLAN.columns()


def test_compositions_reuse_swaps_with_and_without_reversal_and_reject_g_train(toy_task):
    planner = ViewPlanner(toy_task, PLAN, seed=3)
    rejected = 0
    for instance in range(40):
        views = {(v.condition, v.draw): v for v in planner.views(instance)}
        for depth in PLAN.composition_depths:
            for draw in range(PLAN.composition_draws):
                plain, with_reversal = views[(f"k{depth}", draw)], views[(f"k{depth}+R", draw)]
                assert plain.permutation == with_reversal.permutation
                assert with_reversal.guidance.key() == Reversal().apply(plain.guidance)[0].key()
                assert plain.accepted == with_reversal.accepted
                assert plain.accepted == (not planner.in_train_support(plain.guidance))
                # An even number of swaps never yields a single swap: rejection means a
                # returning composition, i.e. the identity permutation.
                if not plain.accepted:
                    rejected += 1
                    assert plain.permutation == tuple(range(toy_task.num_candidates))
        assert all(views[(name, 0)].accepted for name in (BASE, REVERSAL))
    assert rejected > 0  # with four possible swaps, cancelling pairs do occur


def test_equivariant_predictor_has_no_composition_drop(toy_task, examples, belief):
    predictor = LookupPredictor(toy_task, belief)
    records = evaluate(predictor, toy_task, examples, PLAN, seed=5, log_every=0)
    summary = summarize(records)
    assert summary["base"]["accuracy"] == pytest.approx(100 * 23 / 24)
    assert summary["reversal"]["invariance"] == 100.0
    assert summary["binding"]["equiv_accuracy"] == 100.0
    assert summary["composition"]["equiv_accuracy"] == 100.0
    assert summary["composition"]["tv"] == pytest.approx(0.0, abs=1e-12)
    assert summary["composition"]["paired_drop"] == pytest.approx(0.0, abs=1e-9)
    assert summary["composition"]["accuracy"] == pytest.approx(
        summary["composition"]["matched_base_accuracy"]
    )
    # Base and composition correctness coincide instance-wise for an exact transport.
    per_instance = per_instance_scores(records)
    assert np.array_equal(per_instance["base_correct"], per_instance["composition_accuracy"])


def test_memorizing_predictor_fails_exactly_on_changed_targets(toy_task, examples, belief):
    records = evaluate(MemorizingPredictor(toy_task, belief), toy_task, examples, PLAN, 5, 0)
    summary = summarize(records)
    comp = summary["composition"]
    assert summary["base"]["accuracy"] == pytest.approx(100 * 23 / 24)
    assert summary["reversal"]["invariance"] == 100.0  # stable, but ...
    assert comp["changed"]["count"] > 0 and comp["unchanged"]["count"] > 0
    table = view_table(records)
    assert comp["changed"]["pair_accuracy"] == 0.0
    # ... an outdated answer is only "correct" when the believed class was wrong and the
    # swap happens to move the gold output onto it.
    changed_correct = table.correct & table.changed
    assert set(np.nonzero(changed_correct)[0].tolist()) <= {0}
    assert comp["accuracy"] < comp["matched_base_accuracy"]
    assert comp["paired_drop"] > 0
    assert comp["equiv_accuracy"] < 100.0 and comp["tv"] > 0.0
    assert comp["pair_accuracy"] <= comp["pooled_accuracy"]  # same pairs, same weights


def test_aggregation_matches_a_hand_computed_example(toy_task):
    """Comp. averages views within an instance, then instances, then conditions."""
    conditions = [BASE, REVERSAL, "k2", "k2", "k2+R", "k2+R"]
    m = toy_task.num_candidates
    identity = np.arange(m)
    swap = identity.copy()
    swap[[0, 1]] = [1, 0]
    perms = np.tile(identity, (2, 6, 1))
    perms[0, 2] = perms[0, 4] = swap  # instance 0: first draw changes the target (gold = 0)
    gold = np.array([0, 3])
    scores = np.full((2, 6, m), -5.0)

    def answer(i, v, c):
        scores[i, v, c] = 1.0

    for v in range(6):
        answer(1, v, 3)  # instance 1 is always right
    answer(0, 0, 0), answer(0, 1, 0)  # instance 0: base and reversal right
    answer(0, 2, 0)  # k2 draw 0: keeps the outdated answer (target is now 1) -> wrong
    answer(0, 3, 0)  # k2 draw 1: unchanged target -> right
    answer(0, 4, 1)  # k2+R draw 0: follows the swap -> right
    answer(0, 5, 2)  # k2+R draw 1: wrong although the target is unchanged
    accepted = np.ones((2, 6), dtype=bool)
    accepted[1, 3] = False  # one rejected composition for instance 1
    records = EvalRecords(conditions, scores, perms, accepted, gold, np.array([0, 3]), np.arange(m))
    summary = summarize(records)
    comp = summary["composition"]
    # k2: instance means (0.5, 1.0) -> 75; k2+R: (0.5, 1.0) -> 75.
    assert comp["by_condition"] == {"k2": 75.0, "k2+R": 75.0}
    assert comp["accuracy"] == 75.0 and comp["matched_base_accuracy"] == 100.0
    assert comp["paired_drop"] == 25.0
    # Accepted pairs: instance 0 has four, instance 1 has three (all correct).
    assert comp["pair_accuracy"] == pytest.approx(100 * 5 / 7)
    assert comp["changed"] == {"accuracy": 50.0, "pair_accuracy": 50.0, "count": 2}
    assert comp["unchanged"]["count"] == 5
    assert comp["unchanged"]["accuracy"] == pytest.approx(100 * 4 / 5)
    assert comp["equiv_accuracy"] == pytest.approx(100 * 5 / 7)
    assert summary["acceptance"]["k2"] == {
        "accepted": 3,
        "attempted": 4,
        "rate": 75.0,
        "instance_coverage": 100.0,
        "changed": 1,
        "unchanged": 2,
        "labels_changed": 1,
        "labels_supported": 2,
    }
    # Reciprocal ranks: the missed target is ranked second (fixed tie order) in both conditions.
    assert comp["mrr"] == pytest.approx(np.mean([(0.5 + 1) / 2, 1.0]))
    assert comp["pooled_accuracy"] == pytest.approx(100 * 5 / 7)
    assert comp["hit@4"] == 100.0


def test_records_round_trip(tmp_path, toy_task, examples, belief):
    records = evaluate(LookupPredictor(toy_task, belief), toy_task, examples[:5], PLAN, 1, 0)
    records.save(tmp_path / "records.npz")
    restored = EvalRecords.load(tmp_path / "records.npz")
    assert restored.conditions == records.conditions
    assert np.allclose(restored.scores, records.scores, atol=1e-6)
    assert np.array_equal(restored.permutations, records.permutations)
    assert (
        summarize(restored)["composition"]["accuracy"]
        == summarize(records)["composition"]["accuracy"]
    )


def test_identical_views_are_scored_once(toy_task, belief, examples):
    predictor = LookupPredictor(toy_task, belief)
    base = toy_task.base_guidance()
    views = [(examples[0].text, base), (examples[0].text, base), (examples[1].text, base)]
    scores = score_unique(predictor, views)
    assert predictor.calls == 2 and np.array_equal(scores[0], scores[1])
    assert predict(predictor, views).shape == (3,)


def test_dev_pool_has_one_base_view_and_one_edit_per_instance(toy_task, examples, belief):
    pool = build_dev_pool(toy_task, examples, seed=13)
    assert len(pool.views) == 2 * len(examples) == len(pool.targets)
    base = toy_task.base_guidance()
    train_keys = {base.key(), apply_edit(base, Reversal()).guidance.key()}
    for (text, first), (same_text, second) in zip(pool.views[0::2], pool.views[1::2], strict=True):
        assert text == same_text and first.key() == base.key()
        # Development views stay inside the single-edit support G_train.
        assert (
            second.key() in train_keys
            or sum(first.binding[z] != second.binding[z] for z in toy_task.labels) == 2
        )
    again = build_dev_pool(toy_task, examples, seed=13)
    assert [g.key() for _, g in again.views] == [g.key() for _, g in pool.views]
    assert pool.accuracy(LookupPredictor(toy_task, belief)) == pytest.approx(1 - 2 / 48)
    assert pool.accuracy(MemorizingPredictor(toy_task, belief)) < 1 - 2 / 48


def test_swap_view_targets_follow_the_permutation(toy_task):
    base = toy_task.base_guidance()
    applied = apply_edit(base, BindingSwap("account", "closure", "login"))
    target = base.target_index("login")
    assert toy_task.candidates[applied.permutation[target]] == "team beta"
