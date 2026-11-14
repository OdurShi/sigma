"""Guidance edit operators, their sampling laws and validation (Section 4.1, Appendix A.1).

Two families of edits are used for training:

* **Local binding swaps** (decision-changing): pick an eligible non-root parent ``v`` and an
  unordered pair of its leaf children, and exchange the two output strings bound to them.  The
  induced transposition ``pi_e`` of the candidate set satisfies Eq. (1):
  ``a_{g'} = pi_e o a_g`` and ``y' = pi_e(y)``.
* **Representation edit** (meaning-preserving): reverse the child list of every internal
  node; ``pi_e`` is the identity.

Evaluation composes several swaps, optionally followed by the reversal, and keeps only final
documents outside the single-edit training support ``G_train`` (Appendix A.2, C.2).

Permutation convention
----------------------
A permutation ``pi`` of the candidate set is stored as a tuple ``perm`` of candidate indices
with ``perm[c] = pi(c)``.  The permutation matrix of the paper acts as
``[P_e v]_c = v_{pi_e^{-1}(c)}``, i.e. ``(P_e v)[perm[c]] = v[c]``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import comb

import numpy as np

from sigma.guidance.render import (
    GuidanceParseError,
    parse_guidance,
    render_guidance,
    same_structure,
)
from sigma.guidance.schema import Guidance, TaskRecord

Permutation = tuple[int, ...]


# --------------------------------------------------------------------------- permutations
def identity_permutation(m: int) -> Permutation:
    """``id_C`` on ``m`` candidates."""
    return tuple(range(m))


def compose_permutations(second: Permutation, first: Permutation) -> Permutation:
    """``second o first``: apply ``first``, then ``second``."""
    return tuple(second[first[c]] for c in range(len(first)))


def invert_permutation(perm: Permutation) -> Permutation:
    """``pi^{-1}``."""
    inverse = [0] * len(perm)
    for c, image in enumerate(perm):
        inverse[image] = c
    return tuple(inverse)


def is_identity(perm: Permutation) -> bool:
    return all(c == image for c, image in enumerate(perm))


# --------------------------------------------------------------------------- edit operators
@dataclass(frozen=True)
class IdentityEdit:
    """The identity edit ``(g', pi_e) = (g, id_C)``."""

    def apply(self, guidance: Guidance) -> tuple[Guidance, Permutation]:
        return guidance, identity_permutation(guidance.task.num_candidates)


@dataclass(frozen=True)
class BindingSwap:
    """Exchange the outputs bound to two leaf children of one non-root parent (Eq. (1))."""

    parent: str
    leaf_a: str
    leaf_b: str

    def apply(self, guidance: Guidance) -> tuple[Guidance, Permutation]:
        task = guidance.task
        binding = dict(guidance.binding)
        out_a, out_b = binding[self.leaf_a], binding[self.leaf_b]
        binding[self.leaf_a], binding[self.leaf_b] = out_b, out_a
        perm = list(range(task.num_candidates))
        index_a, index_b = task.candidate_index[out_a], task.candidate_index[out_b]
        perm[index_a], perm[index_b] = index_b, index_a
        edited = Guidance(task=task, child_order=guidance.child_order, binding=binding)
        return edited, tuple(perm)


@dataclass(frozen=True)
class Reversal:
    """Representation edit: reverse the child list at every internal node (``pi_e = id``)."""

    def apply(self, guidance: Guidance) -> tuple[Guidance, Permutation]:
        order = {v: tuple(reversed(kids)) for v, kids in guidance.child_order.items()}
        edited = Guidance(task=guidance.task, child_order=order, binding=guidance.binding)
        return edited, identity_permutation(guidance.task.num_candidates)


Edit = IdentityEdit | BindingSwap | Reversal


@dataclass(frozen=True)
class AppliedEdit:
    """Result of applying one edit, after validation and identity replacement.

    Attributes
    ----------
    guidance:
        The edited guidance ``g'`` (equal to the input guidance for an identity pair).
    permutation:
        ``pi_e`` as candidate indices (``perm[c] = pi_e(c)``).
    edit:
        The edit that was requested.
    replaced_by_identity:
        True when the constructed document failed validation and the pair was replaced by
        ``(g, id_C)`` (Appendix A.1).
    """

    guidance: Guidance
    permutation: Permutation
    edit: Edit
    replaced_by_identity: bool = False

    @property
    def is_identity_pair(self) -> bool:
        """True when ``g' = g``; the trainer then reuses the original score vector."""
        return isinstance(self.edit, IdentityEdit) or self.replaced_by_identity


def _well_formed(edit: Edit, guidance: Guidance) -> bool:
    """Structural preconditions of an edit (swaps act on leaf siblings of a non-root parent)."""
    if not isinstance(edit, BindingSwap):
        return True
    task = guidance.task
    if edit.parent not in task.nodes or edit.parent == task.root or edit.leaf_a == edit.leaf_b:
        return False
    leaves = task.leaf_children(edit.parent)
    return edit.leaf_a in leaves and edit.leaf_b in leaves


def validate_edit(
    original: Guidance, edited: Guidance, permutation: Permutation, edit: Edit, document: str
) -> bool:
    """Validate a constructed guidance document (Appendix A.1, "Validation").

    The rendered ``document`` is parsed back and must have the same semantic tree and
    descriptions, the same candidate inventory, and satisfy ``a_{g'}(z) = pi_e(a_g(z))`` for
    every label.  A binding edit must fix all unselected bindings; a representation edit must
    fix every binding.
    """
    task = original.task
    try:
        parsed = parse_guidance(document)
    except GuidanceParseError:
        return False
    if not same_structure(parsed, task):
        return False
    if sorted(o for o in parsed.binding.values() if o is not None) != list(task.candidates):
        return False
    if sorted(permutation) != list(range(task.num_candidates)):
        return False
    for label in task.labels:
        expected = task.candidates[permutation[task.candidate_index[original.binding[label]]]]
        if parsed.binding[label] != expected or edited.binding[label] != expected:
            return False
    for node, kids in edited.child_order.items():
        if parsed.child_order.get(node) != tuple(kids):
            return False
    selected = {edit.leaf_a, edit.leaf_b} if isinstance(edit, BindingSwap) else set()
    for label in task.labels:
        if label not in selected and parsed.binding[label] != original.binding[label]:
            return False
    # A binding edit leaves the serialisation order untouched.
    return not (isinstance(edit, BindingSwap) and parsed.child_order != dict(original.child_order))


def apply_edit(guidance: Guidance, edit: Edit, renderer=render_guidance) -> AppliedEdit:
    """Apply ``edit`` to ``guidance``; on a failed construction fall back to the identity.

    A failed construction is replaced by ``(g', pi_e) = (g, id_C)`` and stays in the
    minibatch; there is no rejection resampling (Appendix A.1).  ``renderer`` is the function
    that serialises the edited guidance; it is a parameter so that the fallback can be tested.
    """
    identity = identity_permutation(guidance.task.num_candidates)
    if isinstance(edit, IdentityEdit):
        return AppliedEdit(guidance, identity, edit)
    if not _well_formed(edit, guidance):
        return AppliedEdit(guidance, identity, edit, replaced_by_identity=True)
    edited, permutation = edit.apply(guidance)
    if not validate_edit(guidance, edited, permutation, edit, renderer(edited)):
        return AppliedEdit(guidance, identity, edit, replaced_by_identity=True)
    return AppliedEdit(edited, permutation, edit)


# --------------------------------------------------------------------------- sampling laws
def binding_edit_support(task: TaskRecord) -> list[tuple[BindingSwap, float]]:
    """All binding swaps with their probabilities under ``rho_bind`` (Eq. (A.1)).

    ``rho_bind(v, {z_a, z_b}) = 1 / (|V| * C(d_v, 2))``.  The list is empty when ``V`` is
    empty, in which case ``rho_bind`` is a point mass on the identity edit.
    """
    parents = task.eligible_parents()
    support: list[tuple[BindingSwap, float]] = []
    for parent in parents:
        leaves = task.leaf_children(parent)
        weight = 1.0 / (len(parents) * comb(len(leaves), 2))
        for i, leaf_a in enumerate(leaves):
            for leaf_b in leaves[i + 1 :]:
                support.append((BindingSwap(parent, leaf_a, leaf_b), weight))
    return support


def sample_binding_edit(task: TaskRecord, rng: np.random.Generator) -> Edit:
    """Draw one edit from ``rho_bind``: uniform parent in ``V``, then uniform leaf pair.

    Both stages are independent of instance labels and model predictions.  Returns the
    identity edit when ``V`` is empty (e.g. for a flat tree).
    """
    parents = task.eligible_parents()
    if not parents:
        return IdentityEdit()
    parent = parents[int(rng.integers(len(parents)))]
    leaves = task.leaf_children(parent)
    first, second = sorted(rng.choice(len(leaves), size=2, replace=False).tolist())
    return BindingSwap(parent, leaves[first], leaves[second])


def sample_mixture_edit(task: TaskRecord, rng: np.random.Generator) -> Edit:
    """Draw one edit from ``rho_T = 1/2 rho_bind + 1/2 rho_repr`` (Appendix A.1)."""
    if rng.random() < 0.5:
        return sample_binding_edit(task, rng)
    return Reversal()


def scheduled_edit(task: TaskRecord, step: int, rng: np.random.Generator, schedule: str) -> Edit:
    """Edit for one minibatch instance at optimiser step ``step`` (numbered from 1).

    ``schedule="alternate"`` is the paper's schedule: a sampled binding edit on odd steps and
    the representation edit on even steps.  ``"binding"`` and ``"reversal"`` are the
    binding-only and reversal-only ablations of Table 2, which keep the two-view budget.
    """
    if schedule == "alternate":
        return sample_binding_edit(task, rng) if step % 2 == 1 else Reversal()
    if schedule == "binding":
        return sample_binding_edit(task, rng)
    if schedule == "reversal":
        return Reversal()
    raise ValueError(f"unknown edit schedule {schedule!r}")


def target_change_probability(task: TaskRecord, label: str) -> float:
    """``Pr[pi_e(a_g(z)) != a_g(z) | e ~ rho_bind] = 2 / (|V| d_{v_z})`` (Eq. (A.2)).

    Zero for labels whose parent is not eligible.  Under the equal mixture ``rho_T`` the
    probability is halved.
    """
    parents = task.eligible_parents()
    parent = task.parent_of(label)
    if parent not in parents:
        return 0.0
    return 2.0 / (len(parents) * len(task.leaf_children(parent)))


# --------------------------------------------------------------------------- compositions
def train_support_keys(task: TaskRecord) -> set[tuple]:
    """Keys of ``G_train``: ``g0``, its reversal and every single-swap document from ``g0``."""
    base = task.base_guidance()
    keys = {base.key(), Reversal().apply(base)[0].key()}
    for swap, _ in binding_edit_support(task):
        keys.add(swap.apply(base)[0].key())
    return keys


@dataclass(frozen=True)
class ComposedEdit:
    """An ordered composition ``g_j = T_{e_j}(g_{j-1})``, ``j = 1..k`` (Section 4.3).

    Attributes
    ----------
    steps:
        The applied edits with their intermediate documents.
    guidance:
        Final document ``g_k``.
    permutation:
        Ordered product ``pi_{e_k} o ... o pi_{e_1}``.
    """

    steps: tuple[AppliedEdit, ...]
    guidance: Guidance
    permutation: Permutation


def apply_sequence(guidance: Guidance, edits: Sequence[Edit]) -> ComposedEdit:
    """Apply ``edits`` in order, composing their candidate permutations."""
    permutation = identity_permutation(guidance.task.num_candidates)
    steps: list[AppliedEdit] = []
    current = guidance
    for edit in edits:
        applied = apply_edit(current, edit)
        steps.append(applied)
        current = applied.guidance
        permutation = compose_permutations(applied.permutation, permutation)
    return ComposedEdit(tuple(steps), current, permutation)


def sample_composition(
    task: TaskRecord, depth: int, rng: np.random.Generator, with_reversal: bool = False
) -> tuple[Edit, ...]:
    """Draw ``depth`` sibling swaps from ``rho_bind``, optionally followed by the reversal."""
    edits: list[Edit] = [sample_binding_edit(task, rng) for _ in range(depth)]
    if with_reversal:
        edits.append(Reversal())
    return tuple(edits)
