"""Margin-residual, spectral, knockout and preliminary-study diagnostics."""

from __future__ import annotations

import numpy as np
import pytest

from sigma.evaluation import diagnostics as dg
from sigma.evaluation.metrics import transport


def _random_trajectory(rng, m, k, noise):
    """Scores along k transpositions: transported scores plus a perturbation at every step."""
    scores = [rng.standard_normal(m) * 2.0]
    perms = []
    for _ in range(k):
        perm = np.arange(m)
        a, b = rng.choice(m, size=2, replace=False)
        perm[a], perm[b] = b, a
        perms.append(perm)
        scores.append(transport(scores[-1], perm) + noise * rng.standard_normal(m) + rng.normal())
    return np.array(scores), perms


def test_trajectory_quantities_match_their_definitions():
    scores = np.array([[3.0, 0.0, 0.0], [0.0, 3.0, 0.5], [0.0, 3.0, 0.5]])
    perms = [[1, 0, 2], [0, 1, 2]]
    d = dg.trajectory_diagnostic(scores, perms, initial_target=0)
    # u0 = (2,-1,-1) -> P u0 = (-1,2,-1); u1 = (-7/6, 11/6, -4/6); delta = (-1/6,-1/6,2/6).
    assert d.residuals[0] == pytest.approx(np.sqrt(6) / 6)
    assert d.residuals[1] == pytest.approx(0.0)
    assert d.local_error == pytest.approx(np.sqrt(6) / 6)
    assert d.initial_margin == pytest.approx(3.0)
    assert d.ratio == pytest.approx(np.sqrt(2) * (np.sqrt(6) / 6) / 3.0)
    assert d.targets == (0, 1, 1)
    assert d.certified and d.final_correct and d.bound_holds()
    assert d.prefix_margins[-1] == pytest.approx(2.5)


def test_accumulation_bound_and_margin_certificate_hold_on_random_trajectories():
    """Eq. (7), Proposition B.3 and Corollary B.4."""
    rng = np.random.default_rng(0)
    diagnostics = []
    for _ in range(300):
        scores, perms = _random_trajectory(rng, m=12, k=int(rng.choice([2, 4, 8])), noise=0.15)
        target = int(scores[0].argmax())
        diagnostics.append(dg.trajectory_diagnostic(scores, perms, target))
    assert all(d.bound_holds() for d in diagnostics)
    certified = [d for d in diagnostics if d.certified]
    assert 20 < len(certified) < 300  # the certificate is neither vacuous nor universal here
    assert all(d.final_correct for d in certified)
    assert all(np.all(d.prefix_margins > 0) for d in certified)  # unique top-1 at every prefix
    summary = dg.summarize_trajectories(diagnostics)
    assert summary["certified_errors"] == 0 and summary["bound_violations"] == 0
    assert summary["certified_count"] == len(certified)


def test_uncertified_trajectories_can_still_fail():
    scores = np.array([[1.0, 0.9, 0.0], [0.0, 1.0, 0.0]])  # the answer flips without an edit
    d = dg.trajectory_diagnostic(scores, [[0, 1, 2]], initial_target=0)
    assert not d.certified and not d.final_correct and d.bound_holds()


def test_contrast_operator_equals_the_edit_average_d3():
    groups = [[0, 1, 2, 3], [5, 6], [7, 8, 9]]
    m = 11  # candidates 4 and 10 are outside the eligible sibling groups
    operator = dg.contrast_operator(m, groups)
    assert np.allclose(operator, dg.edit_averaged_contrast(m, groups))
    values = np.sort(np.linalg.eigvalsh(operator))[::-1]
    expected = sorted(
        [1 / (2 * 3 * 3)] * 3 + [1 / (2 * 3 * 1)] * 1 + [1 / (2 * 3 * 2)] * 2, reverse=True
    )
    assert np.allclose(values[:6], expected)
    assert np.allclose(values[6:], 0.0)
    assert np.all(values <= 0.5 + 1e-12)  # 0 < c_v <= 1/2


def test_spectral_basis_is_orthonormal_and_diagonalises_the_operator():
    groups = [[0, 1, 2, 3], [5, 6], [7, 8, 9]]
    m = 11
    basis = dg.spectral_basis(m, groups)
    directions = basis.directions
    assert directions.shape == (m - 1, m)
    assert np.allclose(directions @ directions.T, np.eye(m - 1), atol=1e-12)
    assert np.allclose(directions.sum(axis=1), 0.0, atol=1e-12)  # all directions are centred
    operator = dg.contrast_operator(m, groups)
    assert np.allclose(operator @ directions.T, directions.T * basis.eigenvalues, atol=1e-12)
    assert basis.exposed.sum() == 6 and (~basis.exposed).sum() == 4
    # First Helmert contrast of the first group: (e_0 - e_1) / sqrt(2)  (Eq. (D.4)).
    expected = np.zeros(m)
    expected[0], expected[1] = 1 / np.sqrt(2), -1 / np.sqrt(2)
    assert np.allclose(directions[0], expected)


def test_direction_counts_of_a_clinc_like_hierarchy():
    """Ten domains of fifteen intents: 140 exposed and 9 null directions, kappa = 1/280."""
    groups = [list(range(15 * g, 15 * (g + 1))) for g in range(10)]
    basis = dg.spectral_basis(150, groups)
    assert basis.exposed.sum() == 140 and (~basis.exposed).sum() == 9
    assert np.allclose(basis.eigenvalues[basis.exposed], 1 / 280)
    assert round(float(basis.eigenvalues.max()), 6) == 0.003571  # value quoted in Appendix D.8


def test_directional_energy_and_log_ratio():
    groups = [[0, 1], [2, 3]]
    basis = dg.spectral_basis(4, groups)
    rng = np.random.default_rng(1)
    coefficients = rng.standard_normal((500, 3)) * np.array([2.0, 1.0, 0.5])
    residuals = coefficients @ basis.directions
    energy = dg.directional_energy(residuals, basis.directions)
    assert np.allclose(energy, (coefficients**2).mean(axis=0))
    # Total energy over a complete basis of the centred space equals the mean squared norm.
    assert energy.sum() == pytest.approx((residuals**2).sum(axis=1).mean())
    halved = dg.directional_energy(residuals / np.sqrt(2), basis.directions)
    assert np.allclose(dg.log_energy_ratio(halved, energy), -1.0)
    scores_base = rng.standard_normal((5, 4))
    perms = np.array([[1, 0, 2, 3]] * 5)
    exact = transport(scores_base, perms) + 2.0
    assert np.allclose(dg.aligned_residuals(scores_base, exact, perms), 0.0)


def test_knockout_effects_d7_d8():
    # Layer-22 Down example of Appendix D.9: Base drop 0.378, composition drop 1.340.
    i_guide, i_eq = dg.knockout_effects(90.0, 88.0, 90.0 - 0.378, 88.0 - 1.340, 0.0189, 0.0284)
    assert i_guide == pytest.approx(0.962, abs=1e-9)
    assert i_eq == pytest.approx(0.0095)


def test_rank_transitions_are_diagonal_for_an_equivariant_model():
    rng = np.random.default_rng(2)
    base = rng.standard_normal((40, 15))
    perms = np.array([rng.permutation(15) for _ in range(40)])
    edited = transport(base, perms)  # exactly transported scores
    transitions = dg.aligned_rank_transitions(base, edited, perms, top=10)
    assert transitions.shape == (10, 11)
    assert np.allclose(transitions[:, :10], np.eye(10))
    assert dg.rank_retention(transitions) == (1.0, 1.0)


def test_rank_transitions_count_overflow_and_normalise_rows():
    base = np.array([[5.0, 4.0, 3.0, 2.0, 1.0]])
    edited = np.array([[4.0, 5.0, 1.0, 2.0, 3.0]])  # ranks: 2, 1, 5, 4, 3
    identity = np.arange(5)[None]
    transitions = dg.aligned_rank_transitions(base, edited, identity, top=3)
    assert np.allclose(transitions.sum(axis=1), 1.0)
    assert transitions[0, 1] == 1.0 and transitions[1, 0] == 1.0
    assert transitions[2, 3] == 1.0  # original rank 3 moved beyond the top-3: overflow bin
    exact, local = dg.rank_retention(transitions)
    assert exact == 0.0 and local == pytest.approx(2 / 3)


def test_reference_uncertainty_percentiles_and_bins():
    uniform = np.zeros((1, 150))
    assert dg.normalized_entropy(uniform)[0] == pytest.approx(1.0)
    peaked = np.array([[100.0] + [0.0] * 149])
    assert dg.normalized_entropy(peaked)[0] == pytest.approx(0.0, abs=1e-9)
    values = np.array([0.1, 0.2, 0.2, 0.4])
    assert dg.upper_percentiles(values).tolist() == [25.0, 75.0, 75.0, 100.0]  # ties go up
    percentiles = np.array([0.0, 8.0, 8.3334, 99.9, 100.0])
    assert dg.uncertainty_bins(percentiles).tolist() == [0, 0, 1, 11, 11]
    assert dg.uncertainty_bins(np.array([50.0])).tolist() == [6]  # boundary enters the right bin


def test_union_response_error_and_binned_rates():
    perm = np.array([[1, 0, 2]] * 4)
    pred_base = np.array([0, 0, 0, 2])
    pred_binding = np.array([1, 0, 1, 2])  # instance 1 keeps the outdated answer
    pred_order = np.array([0, 0, 2, 2])  # instance 2 changes its answer unnecessarily
    errors = dg.response_error(pred_base, pred_binding, perm, pred_order)
    assert errors.tolist() == [False, True, True, False]
    counts, rates = dg.binned_error_rates(errors, np.array([0, 0, 1, 1]), num_bins=3)
    assert counts.tolist() == [2, 2, 0]
    assert rates[0] == 0.5 and rates[1] == 0.5 and np.isnan(rates[2])
