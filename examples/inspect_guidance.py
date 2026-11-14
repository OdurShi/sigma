"""Print a guidance document, a binding swap, the reversal and a composed edit.

No model is involved; this shows the structured-guidance interface on the synthetic task::

    python examples/inspect_guidance.py
"""

from __future__ import annotations

import numpy as np

from sigma.data.synthetic import make_synthetic_task
from sigma.guidance import (
    Reversal,
    apply_edit,
    apply_sequence,
    render_guidance,
    sample_binding_edit,
    sample_composition,
    train_support_keys,
)


def main() -> None:
    task = make_synthetic_task(num_groups=2, classes_per_group=3)
    base = task.base_guidance()
    rng = np.random.default_rng(0)
    print("Base guidance g0\n" + render_guidance(base))

    swap = sample_binding_edit(task, rng)
    applied = apply_edit(base, swap)
    print(f"\nBinding swap {swap}\n" + render_guidance(applied.guidance))
    label = swap.leaf_a
    y, y_edit = base.target_index(label), applied.guidance.target_index(label)
    print(f"required answer for class '{label}': {task.candidates[y]} -> {task.candidates[y_edit]}")
    print(f"permutation of candidate indices pi_e = {applied.permutation}")

    reversed_ = apply_edit(base, Reversal())
    print("\nRepresentation edit (reversal; every binding is preserved)")
    print(render_guidance(reversed_.guidance))

    edits = sample_composition(task, depth=4, rng=rng, with_reversal=True)
    composed = apply_sequence(base, edits)
    held_out = composed.guidance.key() not in train_support_keys(task)
    print(f"\nComposition of 4 swaps + reversal (outside the training support: {held_out})")
    print(render_guidance(composed.guidance))


if __name__ == "__main__":
    main()
