"""Metrics of Appendix C.3."""

from __future__ import annotations

import numpy as np
import pytest

from sigma.evaluation import metrics


def test_top1_breaks_ties_by_fixed_candidate_order():
    scores = np.array([[1.0, 3.0, 3.0, 0.0], [2.0, 2.0, 2.0, 2.0]])
    assert metrics.top1(scores).tolist() == [1, 0]
    # A different fixed tie order (e.g. semantic-label order for SIGMA-F).
    tie_rank = np.array([[3, 2, 0, 1], [3, 2, 0, 1]])
    assert metrics.top1(scores, tie_rank).tolist() == [2, 2]


def test_top_k_returns_the_ranked_candidates_and_agrees_with_ranks():
    scores = np.array([[0.1, 0.9, 0.5, 0.9, 0.5]])
    assert metrics.top_k(scores, 5).tolist() == [[1, 3, 2, 4, 0]]
    assert metrics.top_k(scores, 2).tolist() == [[1, 3]]
    assert metrics.top_k(scores, 1)[..., 0].tolist() == metrics.top1(scores).tolist()
    tie_rank = np.array([[4, 3, 2, 1, 0]])
    assert metrics.top_k(scores, 3, tie_rank).tolist() == [[3, 1, 4]]
    rng = np.random.default_rng(0)
    random_scores = rng.integers(0, 4, size=(6, 9)).astype(float)  # many ties
    order = metrics.top_k(random_scores, 9)
    for gold in range(9):
        ranks = metrics.gold_ranks(random_scores, np.full(6, gold))
        assert np.array_equal(np.argmax(order == gold, axis=1) + 1, ranks)
    with pytest.raises(ValueError):
        metrics.top_k(scores, 0)


def test_gold_ranks_with_ties():
    scores = np.array([[0.1, 0.9, 0.5, 0.9]])
    assert metrics.gold_ranks(scores, np.array([1])).tolist() == [1]
    assert metrics.gold_ranks(scores, np.array([3])).tolist() == [2]  # tie lost to index 1
    assert metrics.gold_ranks(scores, np.array([2])).tolist() == [3]
    assert metrics.gold_ranks(scores, np.array([0])).tolist() == [4]
    tie_rank = np.array([[0, 3, 1, 2]])
    assert metrics.gold_ranks(scores, np.array([3]), tie_rank).tolist() == [1]


def test_hit_at_k_and_mrr():
    ranks = np.array([1, 2, 5, 9])
    assert metrics.hit_at_k(ranks, 1).mean() == pytest.approx(0.25)  # equals accuracy
    assert metrics.hit_at_k(ranks, 4).mean() == pytest.approx(0.5)
    assert metrics.hit_at_k(ranks, 8).mean() == pytest.approx(0.75)
    assert metrics.reciprocal_rank(ranks).mean() == pytest.approx((1 + 1 / 2 + 1 / 5 + 1 / 9) / 4)


def test_macro_f1_uses_classes_with_test_support():
    gold = np.array([0, 0, 1, 1, 1, 2])
    predicted = np.array([0, 1, 1, 1, 3, 3])  # class 3 is predicted but has no support
    # class 0: P=1, R=1/2 -> 2/3; class 1: P=2/3, R=2/3 -> 2/3; class 2: 0.
    assert metrics.macro_f1(predicted, gold) == pytest.approx((2 / 3 + 2 / 3 + 0) / 3)
    weights = np.array([1, 1, 1, 1, 1, 0.0])  # zero weight removes class 2 from the support
    assert metrics.macro_f1(predicted, gold, weights) == pytest.approx(2 / 3)
    assert metrics.macro_f1(gold, gold) == pytest.approx(1.0)


def test_pair_and_equivariant_accuracy():
    perm = np.array([[1, 0, 2], [1, 0, 2], [0, 1, 2]])
    pred_base = np.array([0, 2, 1])
    gold_base = np.array([0, 0, 1])
    pred_edit = np.array([1, 2, 0])
    gold_edit = np.array([1, 1, 1])
    assert metrics.pair_correct(pred_base, gold_base, pred_edit, gold_edit).tolist() == [
        True,
        False,
        False,
    ]
    # Instance 2 is consistent (pi(2) = 2) although both answers are wrong.
    assert metrics.equivariant_prediction(pred_base, pred_edit, perm).tolist() == [
        True,
        True,
        False,
    ]


def test_tv_equivariance_is_zero_for_transported_and_one_for_disjoint():
    scores = np.array([[2.0, 0.0, -1.0]])
    perm = np.array([[2, 0, 1]])
    transported = metrics.transport(scores, perm) + 3.0  # shifts do not matter
    assert metrics.tv_equivariance(scores, transported, perm)[0] == pytest.approx(0.0, abs=1e-12)
    peaked_a = np.array([[50.0, 0.0, 0.0]])
    peaked_b = np.array([[0.0, 50.0, 0.0]])
    identity = np.array([[0, 1, 2]])
    assert metrics.tv_equivariance(peaked_a, peaked_b, identity)[0] == pytest.approx(1.0, abs=1e-9)
    # Hand-computed: q_g = (0.5, 0.5), q_g' = (0.8, 0.2), identity edit -> TV = 0.3.
    base = np.log(np.array([[0.5, 0.5]]))
    edit = np.log(np.array([[0.8, 0.2]]))
    assert metrics.tv_equivariance(base, edit, np.array([[0, 1]]))[0] == pytest.approx(0.3)
    assert metrics.tv_equivariance(base, edit, np.array([[1, 0]]))[0] == pytest.approx(0.3)


def test_aulc_matches_table_5_of_the_paper():
    shots = [1, 2, 4, 8, 16]
    # LoRA-SFT Base accuracies on BANKING77 (Table 5); the paper reports AULC 83.29.
    assert metrics.aulc(shots, [72.84, 79.65, 84.72, 87.63, 89.46]) == pytest.approx(
        83.29, abs=0.005
    )
    assert metrics.aulc(shots, [50.0] * 5) == pytest.approx(50.0)
    with pytest.raises(ValueError):
        metrics.aulc([1, 1, 2], [1.0, 2.0, 3.0])


def test_paired_bootstrap_interval():
    rng = np.random.default_rng(0)
    differences = 0.03 + 0.2 * rng.standard_normal((5, 400))
    mean, lower, upper = metrics.paired_bootstrap_interval(differences, 2000, seed=1)
    assert mean == pytest.approx(differences.mean())
    assert lower < mean < upper
    assert upper - lower < 0.2
    again = metrics.paired_bootstrap_interval(differences, 2000, seed=1)
    assert again == (mean, lower, upper)  # deterministic given the seed
    constant = metrics.paired_bootstrap_interval(np.full((3, 10), 0.5), 200)
    assert constant == (0.5, 0.5, 0.5)
    with pytest.raises(ValueError):
        metrics.paired_bootstrap_interval(np.zeros(4))
