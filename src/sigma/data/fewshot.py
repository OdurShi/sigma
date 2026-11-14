"""Few-shot sampling with nested budgets (Appendix C.1, "Few-shot sampling").

Each seed defines a random ordering within each training class; prefixes of that ordering
give nested ``b``-shot subsets with ``n = b * m`` distinct training annotations.  Methods
compared at the same budget and seed therefore share the same subsets.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

from sigma.seeding import stream

#: Budgets of the label-efficiency sweep (Table 5, Figure 5).
SWEEP_SHOTS: tuple[int, ...] = (1, 2, 4, 8, 16)


@dataclass(frozen=True)
class Example:
    """One labeled input ``(x, z)``: a text and its semantic label."""

    text: str
    label: str


def class_orderings(examples: Sequence[Example], seed: int) -> dict[str, list[int]]:
    """For every class, a seed-defined random ordering of its example indices.

    The ordering of one class depends only on the seed and on that class's own examples, so
    it is unaffected by the presence of other classes.
    """
    by_class: dict[str, list[int]] = defaultdict(list)
    for index, example in enumerate(examples):
        by_class[example.label].append(index)
    orderings: dict[str, list[int]] = {}
    for label in sorted(by_class):
        indices = by_class[label]
        permutation = stream(seed, "fewshot", label).permutation(len(indices))
        orderings[label] = [indices[i] for i in permutation]
    return orderings


def nested_fewshot_subset(examples: Sequence[Example], seed: int, shots: int) -> list[Example]:
    """The ``shots``-shot subset: the first ``min(shots, N_z)`` examples of every class.

    Subsets for increasing ``shots`` are nested for a fixed seed.  Classes with fewer than
    ``shots`` examples contribute all of them (the capped budgets ``n_z = min(b, N_z)`` used
    for MASSIVE-EN in Appendix C.1).
    """
    if shots < 1:
        raise ValueError("shots must be positive")
    orderings = class_orderings(examples, seed)
    return [examples[i] for label in sorted(orderings) for i in orderings[label][:shots]]
