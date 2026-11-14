"""Model-dependent diagnostics: margin-residual trajectories (Section 5.4, Appendix D.7),
spectral residual contraction (Appendix D.8), LoRA knockouts (Appendix D.9) and the
preliminary study (Section 3, Appendix C.7).

Each function scores guidance views with a predictor and reduces the scores with the pure
functions of :mod:`sigma.evaluation.diagnostics`.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from sigma.data.fewshot import Example
from sigma.evaluation import diagnostics as dg
from sigma.evaluation.aggregate import summarize
from sigma.evaluation.evaluate import evaluate, score_unique
from sigma.evaluation.metrics import top1
from sigma.evaluation.protocol import BASE, BINDING, EvalPlan
from sigma.evaluation.records import EvalRecords
from sigma.guidance.edits import BindingSwap, Reversal, apply_edit, sample_binding_edit
from sigma.guidance.schema import TaskRecord, grouped_task
from sigma.models import lora_knockout, lora_updates
from sigma.predictors import Predictor
from sigma.seeding import stream

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- Section 5.4
def diagnostic_subtask(task: TaskRecord, num_candidates: int = 12, seed: int = 0) -> TaskRecord:
    """A restricted-inventory sub-task with ``num_candidates`` leaves (Appendix D.7).

    Eligible sibling groups are visited in a seeded random order and their leaves are taken
    (in identifier order) until the inventory is full; a group is only used if at least two
    of its leaves still fit, so that every kept group remains eligible for swaps.
    """
    rng = stream(seed, "diagnostic-subtask")
    parents = list(task.eligible_parents())
    groups: dict[str, tuple[str, dict[str, str]]] = {}
    remaining = num_candidates
    for index in rng.permutation(len(parents)):
        parent = parents[int(index)]
        leaves = task.leaf_children(parent)[:remaining]
        if len(leaves) < 2:
            continue
        groups[parent] = (
            task.nodes[parent].description,
            {z: task.nodes[z].description for z in leaves},
        )
        remaining -= len(leaves)
        if remaining < 2:
            break
    binding = {z: task.binding[z] for _, members in groups.values() for z in members}
    return grouped_task(
        f"{task.name}-diagnostic",
        groups,
        binding=binding,
        root_description=task.nodes[task.root].description,
        instruction=task.instruction,
    )


def margin_residual_study(
    predictor_for: Callable[[TaskRecord], Predictor],
    task: TaskRecord,
    examples: Sequence[Example],
    depths: Sequence[int] = (2, 4, 8),
    num_candidates: int = 12,
    seed: int = 0,
) -> list[dg.TrajectoryDiagnostic]:
    """Margin-residual diagnostics of swap trajectories on a restricted inventory.

    ``predictor_for(subtask)`` must return a predictor for the restricted task.  One
    trajectory is drawn per in-subtask test instance, with its depth cycling through
    ``depths``; only trajectories whose initial target is uniquely top-ranked are kept, so all
    summaries are conditional on initial correctness (Appendix D.7).
    """
    subtask = diagnostic_subtask(task, num_candidates, seed)
    predictor = predictor_for(subtask)
    base = subtask.base_guidance()
    selected = [e for e in examples if e.label in subtask.label_index]
    diagnostics: list[dg.TrajectoryDiagnostic] = []
    for index, example in enumerate(selected):
        depth = depths[index % len(depths)]
        rng = stream(seed, "margin-residual", index)
        guidances, permutations = [base], []
        for _ in range(depth):
            applied = apply_edit(guidances[-1], sample_binding_edit(subtask, rng))
            guidances.append(applied.guidance)
            permutations.append(applied.permutation)
        scores = score_unique(predictor, [(example.text, g) for g in guidances])
        diagnostic = dg.trajectory_diagnostic(
            scores, permutations, base.target_index(example.label)
        )
        if diagnostic.initial_margin > 0:
            diagnostics.append(diagnostic)
    return diagnostics


# --------------------------------------------------------------------------- Appendix D.8
def binding_residuals(records: EvalRecords) -> np.ndarray:
    """Aligned centred-score residuals of all held-out single-swap views, ``(pairs, m)``."""
    base = records.scores[:, int(records.columns(BASE)[0])]
    columns = records.columns(BINDING)
    residuals = dg.aligned_residuals(
        base[:, None], records.scores[:, columns], records.permutations[:, columns]
    )
    return residuals.reshape(-1, residuals.shape[-1])


def spectral_contraction(
    task: TaskRecord, method_records: EvalRecords, reference_records: EvalRecords
) -> dict[str, np.ndarray]:
    """Directional log energy ratios of one method against a reference (Eq. (D.5)).

    Both record sets must come from the same seed, so that they share the evaluation pairs.
    Returns the intervention eigenvalues ``kappa_j``, the two directional energies and
    ``R_j = log2(E_j^method / E_j^reference)`` for all ``m - 1`` centred directions.
    """
    basis = dg.spectral_basis(task.num_candidates, dg.sibling_index_groups(task))
    energy = dg.directional_energy(binding_residuals(method_records), basis.directions)
    reference = dg.directional_energy(binding_residuals(reference_records), basis.directions)
    return {
        "eigenvalues": basis.eigenvalues,
        "energy": energy,
        "reference_energy": reference,
        "log_ratio": dg.log_energy_ratio(energy, reference),
    }


# --------------------------------------------------------------------------- Appendix D.9
@dataclass(frozen=True)
class KnockoutRecord:
    """Effect of removing one LoRA update from an otherwise intact checkpoint."""

    layer: int
    projection: str
    base_accuracy: float
    composition_accuracy: float
    tv: float
    guide_effect: float  # I_guide, percentage points
    equivariance_effect: float  # I_eq, absolute TV


def knockout_study(
    model,
    predictor_factory: Callable[[], Predictor],
    task: TaskRecord,
    examples: Sequence[Example],
    plan: EvalPlan,
    seed: int,
) -> list[KnockoutRecord]:
    """Single-module knockouts over every adapted projection (Appendix D.9).

    The intact and knockout models are evaluated on the same Base inputs and held-out
    guidance compositions; nothing is retrained.
    """

    def measure() -> tuple[float, float, float]:
        summary = summarize(evaluate(predictor_factory(), task, examples, plan, seed, log_every=0))
        comp = summary["composition"]
        return summary["base"]["accuracy"], comp["accuracy"], comp["tv"]

    full_base, full_comp, full_tv = measure()
    records = []
    for layer, projection in sorted(lora_updates(model)):
        with lora_knockout(model, layer, projection):
            base, comp, tv = measure()
        guide, equivariance = dg.knockout_effects(full_base, full_comp, base, comp, full_tv, tv)
        records.append(KnockoutRecord(layer, projection, base, comp, tv, guide, equivariance))
        logger.info("knockout layer %d %s: I_guide %.3f pp", layer, projection, guide)
    return records


# --------------------------------------------------------------------------- Section 3
@dataclass(frozen=True)
class RevisionScores:
    """Scores of each diagnostic instance under ``g0``, a binding revision and an order
    revision, with the instance-specific transposition of the binding revision."""

    indices: np.ndarray  # positions of the kept instances in the evaluated list
    gold: np.ndarray  # candidate index of the gold output under g0
    base: np.ndarray  # (N, m)
    binding: np.ndarray  # (N, m)
    order: np.ndarray  # (N, m)
    permutations: np.ndarray  # (N, m)

    @property
    def correct(self) -> np.ndarray:
        return top1(self.base) == self.gold

    def response_error(self) -> np.ndarray:
        """Union response error of Eq. (C.9) for every kept instance."""
        return dg.response_error(
            top1(self.base), top1(self.binding), self.permutations, top1(self.order)
        )


def revision_scores(
    predictor: Predictor, task: TaskRecord, examples: Sequence[Example], seed: int = 0
) -> RevisionScores:
    """Score the diagnostic revisions of the preliminary study (Appendix C.7.1).

    The binding revision exchanges the instance's gold output with a uniformly drawn sibling
    partner, so the required answer always changes; the order revision is the reversal.
    Instances whose label has no eligible sibling are skipped.  The draws depend only on
    ``seed`` and the instance index, so they are shared across budgets and runs.
    """
    base = task.base_guidance()
    order = apply_edit(base, Reversal()).guidance
    kept, gold, rows, perms = [], [], [], []
    for index, example in enumerate(examples):
        parent = task.parent_of(example.label)
        if parent not in task.eligible_parents():
            continue
        partners = [z for z in task.leaf_children(parent) if z != example.label]
        partner = partners[int(stream(seed, "revision", index).integers(len(partners)))]
        applied = apply_edit(base, BindingSwap(parent, *sorted((example.label, partner))))
        views = [(example.text, base), (example.text, applied.guidance), (example.text, order)]
        rows.append(score_unique(predictor, views))
        perms.append(applied.permutation)
        gold.append(base.target_index(example.label))
        kept.append(index)
    scores = np.array(rows)
    return RevisionScores(
        np.array(kept), np.array(gold), scores[:, 0], scores[:, 1], scores[:, 2], np.array(perms)
    )
