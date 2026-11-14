"""Running a predictor over the development pool and the test-guidance protocol."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from sigma.data.fewshot import Example
from sigma.evaluation import metrics
from sigma.evaluation.protocol import EvalPlan, ViewPlanner
from sigma.evaluation.records import EvalRecords
from sigma.guidance.edits import apply_edit, sample_mixture_edit
from sigma.guidance.schema import Guidance, TaskRecord
from sigma.predictors import Predictor
from sigma.seeding import stream

logger = logging.getLogger(__name__)


def score_unique(predictor: Predictor, views: Sequence[tuple[str, Guidance]]) -> np.ndarray:
    """Score ``views``; identical ``(input, document)`` pairs are scored once."""
    index: dict[tuple, int] = {}
    unique: list[tuple[str, Guidance]] = []
    positions = []
    for text, guidance in views:
        key = (text, guidance.key())
        if key not in index:
            index[key] = len(unique)
            unique.append((text, guidance))
        positions.append(index[key])
    return predictor.score_views(unique)[positions]


def predict(predictor: Predictor, views: Sequence[tuple[str, Guidance]]) -> np.ndarray:
    """Top-ranked candidate index for every view (ties by the predictor's fixed order)."""
    scores = score_unique(predictor, views)
    tie_rank = None
    if predictor.tie_by_label:
        tie_rank = np.array([guidance.label_of_candidate() for _, guidance in views])
    return metrics.top1(scores, tie_rank)


@dataclass(frozen=True)
class DevPool:
    """Fixed development pool: one base view and one edit from ``rho_T`` per instance.

    The pool depends only on the task, the development instances and the seed, so it is
    identical across the settings that are compared (Appendix A.2, step 7).
    """

    views: list[tuple[str, Guidance]]
    targets: np.ndarray

    def accuracy(self, predictor: Predictor) -> float:
        """Mean top-1 accuracy across all views of the pool."""
        return float((predict(predictor, self.views) == self.targets).mean())


def build_dev_pool(task: TaskRecord, examples: Sequence[Example], seed: int) -> DevPool:
    """Construct the development pool; edited views stay in the development split."""
    base = task.base_guidance()
    views: list[tuple[str, Guidance]] = []
    targets: list[int] = []
    for index, example in enumerate(examples):
        target = base.target_index(example.label)
        applied = apply_edit(base, sample_mixture_edit(task, stream(seed, "dev-pool", index)))
        views += [(example.text, base), (example.text, applied.guidance)]
        targets += [target, applied.permutation[target]]
    return DevPool(views, np.array(targets))


def evaluate(
    predictor: Predictor,
    task: TaskRecord,
    examples: Sequence[Example],
    plan: EvalPlan,
    seed: int,
    log_every: int = 200,
) -> EvalRecords:
    """Score every test instance under all planned guidance views (Appendix C.2)."""
    planner = ViewPlanner(task, plan, seed)
    columns = plan.columns()
    n, v, m = len(examples), len(columns), task.num_candidates
    scores = np.zeros((n, v, m))
    permutations = np.zeros((n, v, m), dtype=np.int64)
    accepted = np.ones((n, v), dtype=bool)
    base = task.base_guidance()
    for i, example in enumerate(examples):
        views = planner.views(i)
        scores[i] = score_unique(predictor, [(example.text, view.guidance) for view in views])
        permutations[i] = np.array([view.permutation for view in views])
        accepted[i] = [view.accepted for view in views]
        if log_every and (i + 1) % log_every == 0:
            logger.info("evaluated %d/%d test instances", i + 1, n)
    return EvalRecords(
        conditions=[name for name, _ in columns],
        scores=scores,
        permutations=permutations,
        accepted=accepted,
        gold=np.array([base.target_index(e.label) for e in examples]),
        gold_label=np.array([task.label_index[e.label] for e in examples]),
        base_label_of_candidate=np.array(base.label_of_candidate()),
        tie_by_label=predictor.tie_by_label,
    )
