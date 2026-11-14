"""Edit operators, sampling laws, validation and compositions (Section 4.1, App. A.1, A.2)."""

from __future__ import annotations

from collections import Counter
from math import comb

import numpy as np
import pytest

from sigma.guidance import (
    BindingSwap,
    IdentityEdit,
    Reversal,
    apply_edit,
    apply_sequence,
    binding_edit_support,
    compose_permutations,
    flat_task,
    identity_permutation,
    invert_permutation,
    render_guidance,
    sample_binding_edit,
    sample_composition,
    sample_mixture_edit,
    scheduled_edit,
    target_change_probability,
    train_support_keys,
)
from sigma.guidance.edits import validate_edit


def test_binding_swap_satisfies_equation_1(toy_task):
    base = toy_task.base_guidance()
    applied = apply_edit(base, BindingSwap("billing", "refund", "invoice"))
    edited, perm = applied.guidance, applied.permutation
    assert not applied.replaced_by_identity and not applied.is_identity_pair
    # a_{g'} = pi_e o a_g for every label, and y' = pi_e(y).
    for label in toy_task.labels:
        expected = toy_task.candidates[perm[toy_task.candidate_index[base.binding[label]]]]
        assert edited.binding[label] == expected
        assert edited.target_index(label) == perm[base.target_index(label)]
    assert edited.binding["refund"] == "desk blue" and edited.binding["invoice"] == "desk red"
    unchanged = set(toy_task.labels) - {"refund", "invoice"}
    assert all(edited.binding[z] == base.binding[z] for z in unchanged)
    # pi_e is the transposition of the two exchanged candidates.
    moved = [c for c in range(toy_task.num_candidates) if perm[c] != c]
    assert sorted(toy_task.candidates[c] for c in moved) == ["desk blue", "desk red"]
    assert compose_permutations(perm, perm) == identity_permutation(toy_task.num_candidates)
    # The tree, the descriptions, the candidates and their order are preserved.
    assert edited.child_order == base.child_order
    assert edited.task is toy_task


def test_swap_only_changes_output_references_in_the_document(toy_task):
    base = toy_task.base_guidance()
    edited = apply_edit(base, BindingSwap("account", "closure", "login")).guidance
    before, after = render_guidance(base).split("\n"), render_guidance(edited).split("\n")
    changed = [(b, a) for b, a in zip(before, after, strict=True) if a != b]
    assert len(changed) == 2
    for old, new in changed:
        assert old.split(" => ")[0] == new.split(" => ")[0]  # descriptions stay attached


def test_reversal_is_an_involution_that_fixes_every_binding(toy_task):
    base = toy_task.base_guidance()
    applied = apply_edit(base, Reversal())
    assert applied.permutation == identity_permutation(toy_task.num_candidates)
    assert applied.guidance.binding == base.binding
    for node, children in base.child_order.items():
        assert applied.guidance.child_order[node] == tuple(reversed(children))
    assert render_guidance(applied.guidance) != render_guidance(base)
    twice = apply_edit(applied.guidance, Reversal()).guidance
    assert render_guidance(twice) == render_guidance(base)


def test_binding_law_matches_equation_a1(toy_task):
    support = binding_edit_support(toy_task)
    # |V| = 2: "account" has C(2,2) = 1 pair, "billing" has C(3,2) = 3 pairs.
    assert len(support) == 4
    assert sum(weight for _, weight in support) == pytest.approx(1.0)
    for swap, weight in support:
        d_v = len(toy_task.leaf_children(swap.parent))
        assert weight == pytest.approx(1.0 / (2 * comb(d_v, 2)))
    rng = np.random.default_rng(0)
    draws = Counter(sample_binding_edit(toy_task, rng) for _ in range(12000))
    for swap, weight in support:
        assert draws[swap] / 12000 == pytest.approx(weight, abs=0.015)
    assert set(draws) == {swap for swap, _ in support}


def test_target_change_probability_matches_equation_a2(toy_task):
    support = binding_edit_support(toy_task)
    for label in toy_task.labels:
        direct = sum(w for swap, w in support if label in (swap.leaf_a, swap.leaf_b))
        assert target_change_probability(toy_task, label) == pytest.approx(direct)
    assert target_change_probability(toy_task, "refund") == pytest.approx(2 / (2 * 3))
    assert target_change_probability(toy_task, "login") == pytest.approx(2 / (2 * 2))
    assert target_change_probability(toy_task, "other") == 0.0


def test_mixture_law_is_an_equal_mix(toy_task):
    rng = np.random.default_rng(1)
    draws = [sample_mixture_edit(toy_task, rng) for _ in range(4000)]
    share = sum(isinstance(e, Reversal) for e in draws) / len(draws)
    assert share == pytest.approx(0.5, abs=0.03)


def test_schedule_alternates_binding_and_representation_steps(toy_task):
    rng = np.random.default_rng(2)
    for step in range(1, 9):
        edit = scheduled_edit(toy_task, step, rng, "alternate")
        assert isinstance(edit, BindingSwap if step % 2 == 1 else Reversal)
        assert isinstance(scheduled_edit(toy_task, step, rng, "binding"), BindingSwap)
        assert isinstance(scheduled_edit(toy_task, step, rng, "reversal"), Reversal)
    with pytest.raises(ValueError):
        scheduled_edit(toy_task, 1, rng, "unknown")


def test_flat_tree_uses_identity_on_binding_steps():
    flat = flat_task("flat", {"a": "first", "b": "second", "c": "third"})
    rng = np.random.default_rng(3)
    edit = scheduled_edit(flat, 1, rng, "alternate")
    assert isinstance(edit, IdentityEdit)
    applied = apply_edit(flat.base_guidance(), edit)
    assert applied.is_identity_pair and not applied.replaced_by_identity
    assert applied.guidance is flat.base_guidance() or applied.guidance == flat.base_guidance()
    assert binding_edit_support(flat) == []


@pytest.mark.parametrize(
    "swap",
    [
        BindingSwap("billing", "refund", "login"),  # not siblings
        BindingSwap("root", "billing", "account"),  # root parent, non-leaf children
        BindingSwap("billing", "refund", "refund"),  # not a pair
        BindingSwap("nowhere", "refund", "invoice"),  # unknown parent
    ],
)
def test_invalid_swaps_are_replaced_by_the_identity(toy_task, swap):
    base = toy_task.base_guidance()
    applied = apply_edit(base, swap)
    assert applied.replaced_by_identity and applied.is_identity_pair
    assert applied.guidance == base
    assert applied.permutation == identity_permutation(toy_task.num_candidates)


def test_failed_document_construction_falls_back_to_identity(toy_task):
    base = toy_task.base_guidance()
    swap = BindingSwap("billing", "refund", "invoice")

    def truncating_renderer(guidance):
        return render_guidance(guidance).rsplit("\n", 1)[0]  # drops the last record

    def stale_renderer(_):
        return render_guidance(base)  # does not reflect the edit

    for renderer in (truncating_renderer, stale_renderer):
        applied = apply_edit(base, swap, renderer=renderer)
        assert applied.replaced_by_identity
        assert applied.guidance == base
        assert applied.permutation == identity_permutation(toy_task.num_candidates)
    assert not apply_edit(base, swap).replaced_by_identity


def test_validation_requires_the_stated_permutation(toy_task):
    base = toy_task.base_guidance()
    swap = BindingSwap("billing", "refund", "invoice")
    edited, perm = swap.apply(base)
    document = render_guidance(edited)
    assert validate_edit(base, edited, perm, swap, document)
    identity = identity_permutation(toy_task.num_candidates)
    assert not validate_edit(base, edited, identity, swap, document)
    # A representation edit must fix every binding.
    assert not validate_edit(base, edited, perm, Reversal(), document)


def test_ordered_composition_of_non_commuting_swaps(toy_task):
    base = toy_task.base_guidance()
    first = BindingSwap("billing", "refund", "invoice")
    second = BindingSwap("billing", "invoice", "charge")
    forward = apply_sequence(base, [first, second])
    backward = apply_sequence(base, [second, first])
    assert forward.permutation != backward.permutation
    for composed in (forward, backward):
        for label in toy_task.labels:
            assert (
                composed.guidance.target_index(label)
                == composed.permutation[base.target_index(label)]
            )
    single = [step.permutation for step in forward.steps]
    assert forward.permutation == compose_permutations(single[1], single[0])
    inverse = invert_permutation(forward.permutation)
    assert compose_permutations(inverse, forward.permutation) == identity_permutation(6)
    # A three-cycle of outputs: refund -> blue -> ... -> green.
    assert forward.guidance.binding["refund"] == "desk blue"
    assert forward.guidance.binding["invoice"] == "desk green"
    assert forward.guidance.binding["charge"] == "desk red"


def test_train_support_contains_exactly_the_single_edit_documents(toy_task):
    base = toy_task.base_guidance()
    keys = train_support_keys(toy_task)
    assert len(keys) == 2 + len(binding_edit_support(toy_task))
    swap = BindingSwap("billing", "refund", "invoice")
    other = BindingSwap("account", "closure", "login")

    def key(*edits):
        return apply_sequence(base, edits).guidance.key()

    assert key() in keys and key(Reversal()) in keys and key(swap) in keys
    assert key(swap, swap) in keys  # a cancelling composition returns to g0
    assert key(swap, swap, Reversal()) in keys
    assert key(swap, Reversal()) not in keys
    assert key(swap, other) not in keys
    assert key(swap, BindingSwap("billing", "invoice", "charge")) not in keys


def test_sampled_compositions_are_label_independent_and_reproducible(toy_task):
    first = sample_composition(toy_task, 4, np.random.default_rng(5), with_reversal=True)
    again = sample_composition(toy_task, 4, np.random.default_rng(5), with_reversal=True)
    assert first == again and len(first) == 5
    assert isinstance(first[-1], Reversal)
    assert all(isinstance(e, BindingSwap) for e in first[:-1])
