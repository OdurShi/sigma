"""Metrics reported in the paper (Appendix C.3).  Pure NumPy, independent of any model.

Score arrays hold candidate scores ``s_g`` in the fixed candidate order along the last axis.
Rankings sort scores in descending order and break ties by the fixed candidate order
(Section 4.3); an optional ``tie_rank`` array supplies a different fixed tie order, which the
factorised variant uses to resolve ties in semantic-label order (Appendix A.3).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def softmax(scores: np.ndarray) -> np.ndarray:
    """Candidate distribution ``q_g = softmax(s_g)`` (unit temperature, log-sum-exp)."""
    shifted = scores - scores.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


def transport(vector: np.ndarray, permutation: np.ndarray) -> np.ndarray:
    """``[P_e v]_c = v_{pi_e^{-1}(c)}`` for ``permutation[..., c] = pi_e(c)``."""
    vector = np.broadcast_to(vector, permutation.shape)
    out = np.empty(permutation.shape, dtype=vector.dtype)
    np.put_along_axis(out, permutation, vector, axis=-1)
    return out


def top1(scores: np.ndarray, tie_rank: np.ndarray | None = None) -> np.ndarray:
    """Top-ranked candidate index; ties go to the smallest candidate index (or ``tie_rank``)."""
    if tie_rank is None:
        return scores.argmax(axis=-1)
    order = np.argsort(tie_rank, axis=-1, kind="stable")
    best = np.take_along_axis(scores, order, axis=-1).argmax(axis=-1)
    return np.take_along_axis(order, best[..., None], axis=-1)[..., 0]


def top_k(scores: np.ndarray, k: int, tie_rank: np.ndarray | None = None) -> np.ndarray:
    """The ``K_top`` best candidate indices in rank order, ``1 <= K_top <= m`` (Section 4.3).

    Scores are sorted in descending order with ties broken by the fixed candidate order (or
    by ``tie_rank``); no output permutation is applied.
    """
    if not 1 <= k <= scores.shape[-1]:
        raise ValueError("K_top must lie between 1 and the number of candidates")
    if tie_rank is None:
        tie_rank = np.broadcast_to(np.arange(scores.shape[-1]), scores.shape)
    # Stable two-key sort: by tie order first, then by descending score.
    by_tie = np.argsort(tie_rank, axis=-1, kind="stable")
    by_score = np.argsort(-np.take_along_axis(scores, by_tie, axis=-1), axis=-1, kind="stable")
    return np.take_along_axis(by_tie, by_score, axis=-1)[..., :k]


def gold_ranks(
    scores: np.ndarray, gold: np.ndarray, tie_rank: np.ndarray | None = None
) -> np.ndarray:
    """1-based rank ``r_i`` of the gold candidate under descending score, fixed tie order."""
    gold_score = np.take_along_axis(scores, gold[..., None], axis=-1)
    if tie_rank is None:
        tie_rank = np.broadcast_to(np.arange(scores.shape[-1]), scores.shape)
    gold_tie = np.take_along_axis(tie_rank, gold[..., None], axis=-1)
    ahead = (scores > gold_score) | ((scores == gold_score) & (tie_rank < gold_tie))
    return 1 + ahead.sum(axis=-1)


def weighted_mean(values: np.ndarray, weights: np.ndarray | None = None) -> float:
    """Mean of ``values`` with optional non-negative weights; ``nan`` if the weight is zero."""
    values = np.asarray(values, dtype=np.float64)
    if weights is None:
        return float(values.mean()) if values.size else float("nan")
    total = float(np.sum(weights))
    return float(np.sum(values * weights) / total) if total > 0 else float("nan")


def macro_f1(predicted: np.ndarray, gold: np.ndarray, weights: np.ndarray | None = None) -> float:
    """Macro-F1 over the semantic classes that have support in ``gold`` (Appendix C.3).

    ``predicted`` and ``gold`` hold semantic-label indices: predictions made under edited
    guidance are first mapped back with ``a_g^{-1}`` so that views can be pooled.
    """
    predicted, gold = np.ravel(predicted), np.ravel(gold)
    weights = np.ones(gold.shape, dtype=np.float64) if weights is None else np.ravel(weights)
    scores = []
    for label in np.unique(gold[weights > 0]):
        tp = weights[(predicted == label) & (gold == label)].sum()
        fp = weights[(predicted == label) & (gold != label)].sum()
        fn = weights[(predicted != label) & (gold == label)].sum()
        denominator = 2 * tp + fp + fn
        scores.append(2 * tp / denominator if denominator > 0 else 0.0)
    return float(np.mean(scores)) if scores else float("nan")


def hit_at_k(ranks: np.ndarray, k: int) -> np.ndarray:
    """Indicator ``1[r_i <= K_top]``; its mean is Hit@K (Hit@1 equals accuracy)."""
    return (ranks <= k).astype(np.float64)


def reciprocal_rank(ranks: np.ndarray) -> np.ndarray:
    """``1 / r_i``; its mean is MRR."""
    return 1.0 / ranks


def pair_correct(
    pred_base: np.ndarray, gold_base: np.ndarray, pred_edit: np.ndarray, gold_edit: np.ndarray
) -> np.ndarray:
    """Indicator of Eq. (C.1): both the base and the edited answer are correct."""
    return (pred_base == gold_base) & (pred_edit == gold_edit)


def equivariant_prediction(
    pred_base: np.ndarray, pred_edit: np.ndarray, permutation: np.ndarray
) -> np.ndarray:
    """Indicator of Eq. (C.2): ``yhat_{g'} = pi_e(yhat_g)`` (invariance under reversal)."""
    transported = np.take_along_axis(permutation, pred_base[..., None], axis=-1)[..., 0]
    return pred_edit == transported


def tv_equivariance(
    scores_base: np.ndarray, scores_edit: np.ndarray, permutation: np.ndarray
) -> np.ndarray:
    """``1/2 || q_{g'} - P_e q_g ||_1`` per pair (Eq. (C.3)); lies in ``[0, 1]``."""
    aligned = transport(softmax(scores_base), permutation)
    return 0.5 * np.abs(softmax(scores_edit) - aligned).sum(axis=-1)


def aulc(shots: Sequence[int], accuracies: Sequence[float]) -> float:
    """Normalised area under the label-efficiency curve (Appendix C.3).

    Trapezoidal area of accuracy against ``log2 b`` divided by the log-budget range; for
    ``b in {1, 2, 4, 8, 16}`` the divisor is four, as in the paper.
    """
    x = np.log2(np.asarray(shots, dtype=np.float64))
    y = np.asarray(accuracies, dtype=np.float64)
    if x.ndim != 1 or x.shape != y.shape or len(x) < 2 or np.any(np.diff(x) <= 0):
        raise ValueError("shots must be strictly increasing and match accuracies")
    area = float(np.sum((y[1:] + y[:-1]) * np.diff(x) / 2.0))
    return area / float(x[-1] - x[0])


def paired_bootstrap_interval(
    differences: np.ndarray, num_resamples: int = 10_000, level: float = 0.95, seed: int = 0
) -> tuple[float, float, float]:
    """Paired percentile bootstrap over seeds and original instances (Appendix C.3).

    ``differences[s, i]`` is the per-instance metric of method A minus method B for seed
    ``s`` and test instance ``i`` (all views and budgets of an instance already combined, so
    they are resampled together).  Seeds and instances are resampled jointly with
    replacement.  Returns ``(mean difference, lower, upper)``.
    """
    differences = np.asarray(differences, dtype=np.float64)
    if differences.ndim != 2:
        raise ValueError("differences must have shape (seeds, instances)")
    num_seeds, num_instances = differences.shape
    rng = np.random.default_rng(seed)
    estimates = np.empty(num_resamples)
    for r in range(num_resamples):
        seeds = rng.integers(num_seeds, size=num_seeds)
        instances = rng.integers(num_instances, size=num_instances)
        estimates[r] = differences[seeds][:, instances].mean()
    alpha = (1.0 - level) / 2.0
    lower, upper = np.quantile(estimates, [alpha, 1.0 - alpha])
    return float(differences.mean()), float(lower), float(upper)
