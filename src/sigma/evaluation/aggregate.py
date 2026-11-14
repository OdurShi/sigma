"""Aggregation of evaluation records into the quantities of the paper's tables.

Units follow the paper: accuracy, macro-F1, PairAcc, EquivAcc and Hit@K are percentages;
``TV_eq`` and MRR lie in ``[0, 1]``.

Aggregation rules (Appendix C.3, C.6)
-------------------------------------
* **Base** uses ``g0``; binding and reversal results use the original test population.
* **Comp.** averages the held-out composition conditions (``k in {2, 4, 8}`` swaps, with and
  without reversal).  Within a condition the accepted views of an instance are averaged
  first, then instances with at least one accepted view are averaged, then conditions.
  Conditions without any accepted view are excluded.
* **Matched Base** (Table 11) re-evaluates base correctness on exactly the instances and
  weights of each composition condition; the paired drop is Matched Base minus Comp.
* Pair-level diagnostics on compositions (PairAcc, EquivAcc, ``TV_eq``, the changed and
  unchanged strata) pool all accepted composition pairs with equal weight.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sigma.evaluation import metrics
from sigma.evaluation.protocol import BASE, BINDING, REVERSAL
from sigma.evaluation.records import EvalRecords


@dataclass
class ViewTable:
    """Per-(instance, view) quantities derived from the raw scores; all of shape ``(N, V)``."""

    predicted: np.ndarray  # top-ranked candidate index
    target: np.ndarray  # candidate index of y' = pi(y)
    correct: np.ndarray
    rank: np.ndarray  # 1-based rank of the gold output
    changed: np.ndarray  # y' != y
    equivariant: np.ndarray  # predicted == pi(base prediction), Eq. (C.2)
    pair: np.ndarray  # base and edited answers both correct, Eq. (C.1)
    tv: np.ndarray  # Eq. (C.3) against the base view
    predicted_label: np.ndarray  # a_g^{-1}(predicted) as a semantic-label index


def view_table(records: EvalRecords) -> ViewTable:
    """Compute predictions, ranks and pairwise diagnostics for every view."""
    scores, perms = records.scores, records.permutations
    m = scores.shape[-1]
    inverse = np.empty_like(perms)
    np.put_along_axis(inverse, perms, np.broadcast_to(np.arange(m), perms.shape), axis=-1)
    label_of_candidate = records.base_label_of_candidate[inverse]  # a_g^{-1}(c) per view
    tie_rank = label_of_candidate if records.tie_by_label else None
    predicted = metrics.top1(scores, tie_rank)
    target = np.take_along_axis(perms, records.gold[:, None, None], axis=-1)[..., 0]
    base_column = int(records.columns(BASE)[0])
    base_predicted = predicted[:, base_column : base_column + 1]
    base_scores = scores[:, base_column : base_column + 1]
    correct = predicted == target
    return ViewTable(
        predicted=predicted,
        target=target,
        correct=correct,
        rank=metrics.gold_ranks(scores, target, tie_rank),
        changed=target != records.gold[:, None],
        equivariant=metrics.equivariant_prediction(base_predicted, predicted, perms),
        pair=correct[:, base_column : base_column + 1] & correct,
        tv=metrics.tv_equivariance(base_scores, scores, perms),
        predicted_label=np.take_along_axis(label_of_candidate, predicted[..., None], axis=-1)[
            ..., 0
        ],
    )


def _percent(value: float) -> float:
    return 100.0 * value


def _instance_means(values: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mean of ``values`` over the masked views of each instance, and which instances have any."""
    counts = mask.sum(axis=1)
    sums = (values * mask).sum(axis=1)
    has_view = counts > 0
    return np.divide(sums, counts, out=np.zeros(len(counts)), where=has_view), has_view


def _pooled(values: np.ndarray, mask: np.ndarray) -> float:
    return float(values[mask].mean()) if mask.any() else float("nan")


def composition_conditions(records: EvalRecords) -> list[str]:
    """Composition condition names present in the records, in column order."""
    seen: list[str] = []
    for name in records.conditions:
        if name not in (BASE, REVERSAL, BINDING) and name not in seen:
            seen.append(name)
    return seen


def per_instance_composition(
    records: EvalRecords, values: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """``(C, N)`` per-condition instance means of ``values`` and the matching coverage mask."""
    means, masks = [], []
    for condition in composition_conditions(records):
        columns = records.columns(condition)
        mean, has_view = _instance_means(
            values[:, columns].astype(np.float64), records.accepted[:, columns]
        )
        means.append(mean)
        masks.append(has_view)
    return np.array(means), np.array(masks)


def composition_average(records: EvalRecords, values: np.ndarray) -> tuple[float, dict[str, float]]:
    """Comp. aggregate of a per-view quantity and its per-condition values (Appendix C.6)."""
    means, masks = per_instance_composition(records, values)
    by_condition: dict[str, float] = {}
    for condition, mean, mask in zip(composition_conditions(records), means, masks, strict=True):
        if mask.any():
            by_condition[condition] = float(mean[mask].mean())
    overall = float(np.mean(list(by_condition.values()))) if by_condition else float("nan")
    return overall, by_condition


def matched_base(records: EvalRecords, base_values: np.ndarray) -> float:
    """Base quantity re-weighted to the accepted instances of every composition condition."""
    _, masks = per_instance_composition(records, np.zeros(records.accepted.shape))
    per_condition = [float(base_values[mask].mean()) for mask in masks if mask.any()]
    return float(np.mean(per_condition)) if per_condition else float("nan")


def _view_weights(records: EvalRecords, columns: np.ndarray) -> np.ndarray:
    """Weights that give every instance total weight one within one condition."""
    mask = records.accepted[:, columns].astype(np.float64)
    counts = mask.sum(axis=1, keepdims=True)
    return np.divide(mask, counts, out=np.zeros_like(mask), where=counts > 0)


def _macro_f1(records: EvalRecords, table: ViewTable, conditions: list[str]) -> float:
    predicted, gold, weights = [], [], []
    for condition in conditions:
        columns = records.columns(condition)
        predicted.append(table.predicted_label[:, columns])
        gold.append(np.repeat(records.gold_label[:, None], len(columns), axis=1))
        weights.append(_view_weights(records, columns))
    return _percent(
        metrics.macro_f1(
            np.concatenate(predicted, 1), np.concatenate(gold, 1), np.concatenate(weights, 1)
        )
    )


def acceptance_table(records: EvalRecords, table: ViewTable) -> dict[str, dict[str, float]]:
    """Held-out composition acceptance and coverage per condition (Table 14)."""
    supported = np.unique(records.gold_label)
    rows: dict[str, dict[str, float]] = {}
    for condition in composition_conditions(records):
        columns = records.columns(condition)
        accepted = records.accepted[:, columns]
        changed = table.changed[:, columns] & accepted
        labels = np.repeat(records.gold_label[:, None], len(columns), axis=1)
        rows[condition] = {
            "accepted": int(accepted.sum()),
            "attempted": int(accepted.size),
            "rate": _percent(float(accepted.mean())),
            "instance_coverage": _percent(float(accepted.any(axis=1).mean())),
            "changed": int(changed.sum()),
            "unchanged": int(accepted.sum() - changed.sum()),
            "labels_changed": int(len(np.unique(labels[changed]))),
            "labels_supported": int(len(supported)),
        }
    return rows


def summarize(records: EvalRecords) -> dict:
    """All reported metrics of one evaluated model (Tables 1, 2, 7, 11, 12, 14)."""
    table = view_table(records)
    base = int(records.columns(BASE)[0])
    reversal = int(records.columns(REVERSAL)[0])
    binding = records.columns(BINDING)
    comp_names = composition_conditions(records)
    comp_columns = (
        np.concatenate([records.columns(c) for c in comp_names])
        if comp_names
        else np.array([], int)
    )
    comp_mask = np.zeros(records.accepted.shape, dtype=bool)
    comp_mask[:, comp_columns] = records.accepted[:, comp_columns]
    bind_mask = np.zeros(records.accepted.shape, dtype=bool)
    bind_mask[:, binding] = True

    base_correct = table.correct[:, base].astype(np.float64)
    comp_accuracy, by_condition = composition_average(records, table.correct)
    matched = matched_base(records, base_correct)
    summary: dict = {
        "num_instances": records.num_instances,
        "num_candidates": int(records.scores.shape[-1]),
        "base": {
            "accuracy": _percent(float(base_correct.mean())),
            "macro_f1": _macro_f1(records, table, [BASE]),
        },
        "reversal": {
            "accuracy": _percent(float(table.correct[:, reversal].mean())),
            "invariance": _percent(float(table.equivariant[:, reversal].mean())),
            "macro_f1": _macro_f1(records, table, [REVERSAL]),
            "tv": float(table.tv[:, reversal].mean()),
        },
    }
    if len(binding):
        summary["binding"] = {
            "accuracy": _percent(_pooled(table.correct, bind_mask)),
            "changed_accuracy": _percent(_pooled(table.correct, bind_mask & table.changed)),
            "pair_accuracy": _percent(_pooled(table.pair, bind_mask)),
            "equiv_accuracy": _percent(_pooled(table.equivariant, bind_mask)),
            "tv": _pooled(table.tv, bind_mask),
            "macro_f1": _macro_f1(records, table, [BINDING]),
            "num_changed": int((bind_mask & table.changed).sum()),
            "num_pairs": int(bind_mask.sum()),
        }
    if comp_names:
        changed, unchanged = comp_mask & table.changed, comp_mask & ~table.changed
        summary["composition"] = {
            "accuracy": _percent(comp_accuracy),
            "by_condition": {k: _percent(v) for k, v in by_condition.items()},
            "matched_base_accuracy": _percent(matched),
            "paired_drop": _percent(matched - comp_accuracy),
            "pooled_accuracy": _percent(_pooled(table.correct, comp_mask)),
            "changed_accuracy": _percent(_pooled(table.correct, changed)),
            "pair_accuracy": _percent(_pooled(table.pair, comp_mask)),
            "equiv_accuracy": _percent(_pooled(table.equivariant, comp_mask)),
            "tv": _pooled(table.tv, comp_mask),
            "changed": {
                "accuracy": _percent(_pooled(table.correct, changed)),
                "pair_accuracy": _percent(_pooled(table.pair, changed)),
                "count": int(changed.sum()),
            },
            "unchanged": {
                "accuracy": _percent(_pooled(table.correct, unchanged)),
                "pair_accuracy": _percent(_pooled(table.pair, unchanged)),
                "count": int(unchanged.sum()),
            },
            "macro_f1": _macro_f1(records, table, comp_names),
            "hit@4": _percent(composition_average(records, metrics.hit_at_k(table.rank, 4))[0]),
            "hit@8": _percent(composition_average(records, metrics.hit_at_k(table.rank, 8))[0]),
            "mrr": composition_average(records, metrics.reciprocal_rank(table.rank))[0],
        }
        summary["acceptance"] = acceptance_table(records, table)
    return summary


def per_instance_scores(records: EvalRecords) -> dict[str, np.ndarray]:
    """Per-instance base correctness and Comp. accuracy, the inputs of the paired bootstrap."""
    table = view_table(records)
    means, masks = per_instance_composition(records, table.correct)
    weights = masks.astype(np.float64)
    total = weights.sum(axis=0)
    comp = np.divide(
        (means * weights).sum(axis=0), total, out=np.full(len(total), np.nan), where=total > 0
    )
    return {
        "base_correct": table.correct[:, int(records.columns(BASE)[0])].astype(np.float64),
        "composition_accuracy": comp,
    }
