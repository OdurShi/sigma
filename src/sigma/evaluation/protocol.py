"""Test-guidance protocol (Appendix C.2, "Test guidance"; Appendix A.2 for ``G_train``).

Every test instance is evaluated under

* the base guidance ``g0`` and its reversal,
* ``binding_draws`` label-independent single binding swaps,
* for every depth ``k`` in ``composition_depths``, ``composition_draws`` sequences of ``k``
  sibling swaps, whose final guidance is evaluated both without and with the reversal.

Optionally (``EvalPlan.transfer``, supplementary experiment E01) each composition is also
followed by a random child order or by a cross-group binding swap, two edits that training
never uses; these views are reported separately and never enter "Comp.".

Transfer ("Comp.") evaluation keeps only final documents outside ``G_train`` (the base
document, its reversal and all single-swap documents); returning or cancelling compositions
are rejected and counted.  All draws depend on the seed and the instance index only, never on
the label or on a model, so they are shared across methods and budgets.
"""

from __future__ import annotations

from dataclasses import dataclass

from sigma.guidance.edits import (
    Permutation,
    RandomOrder,
    Reversal,
    apply_edit,
    apply_sequence,
    identity_permutation,
    sample_binding_edit,
    sample_composition,
    sample_cross_group_swap,
    train_support_keys,
)
from sigma.guidance.schema import Guidance, TaskRecord
from sigma.seeding import stream

BASE, REVERSAL, BINDING = "base", "reversal", "binding"
#: Evaluation-only edit families of supplementary experiment E01, appended to a composition.
RANDOM_ORDER, CROSS_GROUP = "order", "cross"
TRANSFER_FAMILIES = (RANDOM_ORDER, CROSS_GROUP)


def composition_condition(depth: int, with_reversal: bool) -> str:
    """Name of a held-out composition condition, e.g. ``k4`` or ``k4+R``."""
    return f"k{depth}+R" if with_reversal else f"k{depth}"


def transfer_condition(family: str, depth: int) -> str:
    """Name of a broader-edit condition, e.g. ``order:k4`` or ``cross:k4``."""
    return f"{family}:k{depth}"


@dataclass(frozen=True)
class EvalPlan:
    """Numbers of guidance draws per test instance; the defaults are the paper's."""

    binding_draws: int = 10
    composition_depths: tuple[int, ...] = (2, 4, 8)
    composition_draws: int = 10
    transfer: tuple[str, ...] = ()  # subset of TRANSFER_FAMILIES (off in the paper's protocol)

    def __post_init__(self) -> None:
        unknown = set(self.transfer) - set(TRANSFER_FAMILIES)
        if unknown:
            raise ValueError(f"unknown transfer families {sorted(unknown)}")

    def columns(self) -> list[tuple[str, int]]:
        """``(condition, draw)`` for every view of an instance, in a fixed order."""
        columns = [(BASE, 0), (REVERSAL, 0)]
        columns += [(BINDING, d) for d in range(self.binding_draws)]
        for depth in self.composition_depths:
            for with_reversal in (False, True):
                name = composition_condition(depth, with_reversal)
                columns += [(name, d) for d in range(self.composition_draws)]
        for family in self.transfer:
            for depth in self.composition_depths:
                name = transfer_condition(family, depth)
                columns += [(name, d) for d in range(self.composition_draws)]
        return columns

    def composition_conditions(self) -> list[str]:
        """The held-out composition conditions (six in the paper) averaged into "Comp."."""
        return [
            composition_condition(depth, with_reversal)
            for depth in self.composition_depths
            for with_reversal in (False, True)
        ]


@dataclass(frozen=True)
class PlannedView:
    """One guidance view of a test instance.

    ``permutation`` maps base outputs to the outputs required under ``guidance`` (the ordered
    product of the applied edits), and ``accepted`` is False for composed documents that fall
    back into ``G_train``.
    """

    condition: str
    draw: int
    guidance: Guidance
    permutation: Permutation
    accepted: bool = True


class ViewPlanner:
    """Generates the label-independent guidance views of each test instance."""

    def __init__(self, task: TaskRecord, plan: EvalPlan, seed: int) -> None:
        self.task = task
        self.plan = plan
        self.seed = seed
        self._base = task.base_guidance()
        self._reversed = apply_edit(self._base, Reversal()).guidance
        self._train_keys = train_support_keys(task)

    def in_train_support(self, guidance: Guidance) -> bool:
        """Whether a document belongs to ``G_train``."""
        return guidance.key() in self._train_keys

    def views(self, instance_index: int) -> list[PlannedView]:
        """All views of one instance, aligned with :meth:`EvalPlan.columns`."""
        task, plan = self.task, self.plan
        identity = identity_permutation(task.num_candidates)
        views = [
            PlannedView(BASE, 0, self._base, identity),
            PlannedView(REVERSAL, 0, self._reversed, identity),
        ]
        for draw in range(plan.binding_draws):
            rng = stream(self.seed, "eval-binding", instance_index, draw)
            applied = apply_edit(self._base, sample_binding_edit(task, rng))
            views.append(PlannedView(BINDING, draw, applied.guidance, applied.permutation))
        for depth in plan.composition_depths:
            sequences = [
                sample_composition(
                    task, depth, stream(self.seed, "eval-comp", instance_index, depth, d)
                )
                for d in range(plan.composition_draws)
            ]
            for with_reversal in (False, True):
                name = composition_condition(depth, with_reversal)
                for draw, swaps in enumerate(sequences):
                    edits = (*swaps, Reversal()) if with_reversal else swaps
                    composed = apply_sequence(self._base, edits)
                    views.append(
                        PlannedView(
                            name,
                            draw,
                            composed.guidance,
                            composed.permutation,
                            accepted=not self.in_train_support(composed.guidance),
                        )
                    )
        for family in plan.transfer:
            for depth in plan.composition_depths:
                for draw in range(plan.composition_draws):
                    # The swaps are those of the standard composition with the same draw.
                    swaps = sample_composition(
                        task, depth, stream(self.seed, "eval-comp", instance_index, depth, draw)
                    )
                    rng = stream(self.seed, f"eval-{family}", instance_index, depth, draw)
                    if family == RANDOM_ORDER:
                        extra = RandomOrder(int(rng.integers(2**31)))
                    else:
                        extra = sample_cross_group_swap(task, rng)
                    composed = apply_sequence(self._base, (*swaps, extra))
                    views.append(
                        PlannedView(
                            transfer_condition(family, depth),
                            draw,
                            composed.guidance,
                            composed.permutation,
                            accepted=not self.in_train_support(composed.guidance),
                        )
                    )
        return views
