"""Numerical checks of the fixed-feature proxy (Section 4.4, Appendix B.3 - B.6)."""

from __future__ import annotations

import numpy as np
import pytest

from sigma import theory
from sigma.evaluation.diagnostics import contrast_operator


def _random_design(rng, n=7, h=3, d=5, edits=4):
    base = rng.standard_normal((n, h, d))
    aligned = base[:, None] + 0.6 * rng.standard_normal((n, edits, h, d))
    law = rng.dirichlet(np.ones(edits))
    return theory.ProxyDesign(base, aligned, law)


def test_closed_form_estimator_minimises_the_proxy_objective():
    """Lemma B.5: w_lambda = (S + 4 lambda C)^{-1} Mbar^T t / n minimises Eq. (B.6)."""
    rng = np.random.default_rng(0)
    design = _random_design(rng)
    targets = rng.standard_normal((7, 3))
    for lam in (0.0, 0.3, 5.0):
        estimate = theory.proxy_estimator(design, targets, lam)
        value = theory.proxy_objective(design, estimate, targets, lam)
        for _ in range(20):
            perturbed = estimate + 1e-3 * rng.standard_normal(estimate.shape)
            assert theory.proxy_objective(design, perturbed, targets, lam) > value


def test_spectral_range_of_the_whitened_contrast():
    rng = np.random.default_rng(1)
    gram, contrast = _random_design(rng).gram_matrices()
    assert np.all(np.linalg.eigvalsh(gram) > 0)
    whitener = np.linalg.inv(np.linalg.cholesky(gram))
    kappa = np.linalg.eigvalsh(whitener @ contrast @ whitener.T)
    assert kappa.min() > -1e-10 and kappa.max() < 1 + 1e-10  # 0 <= K <= I


def test_exact_risk_identity_with_reused_labels():
    """Theorem B.6 against a direct bias/variance computation of the linear estimator."""
    rng = np.random.default_rng(2)
    n, h, d = 7, 3, 5
    design = _random_design(rng, n, h, d)
    means = rng.standard_normal((n, h))
    roots = rng.standard_normal((n, h, h))
    covariances = roots @ roots.transpose(0, 2, 1)
    gram, contrast = design.gram_matrices()
    averaged = design.averaged_mean_view.reshape(n * h, d)
    oracle = np.linalg.solve(gram, averaged.T @ means.reshape(-1) / n)
    block = np.zeros((n * h, n * h))
    for i in range(n):
        block[i * h : (i + 1) * h, i * h : (i + 1) * h] = covariances[i]
    for lam in (0.0, 0.2, 3.0):
        linear_map = np.linalg.solve(gram + 4 * lam * contrast, averaged.T) / n
        bias = linear_map @ means.reshape(-1) - oracle
        direct = bias @ gram @ bias + np.trace(gram @ linear_map @ block @ linear_map.T)
        assert theory.exact_risk(design, means, covariances, lam) == pytest.approx(direct, rel=1e-9)


def test_theorem_4_1_samplewise_contraction_under_exact_equivariance():
    """If C w_dagger = 0 the regularised estimate is never further from w_dagger in S-norm."""
    rng = np.random.default_rng(3)
    n, m, p = 12, 6, 2
    groups = [[0, 1, 2], [3, 4]]
    features = rng.standard_normal((n, p))
    design = theory.hierarchy_design(features, m, groups)
    gram, contrast = design.gram_matrices()
    h = m - 1
    # An equivariant oracle: arbitrary semantic weights, zero guidance-independent offset.
    oracle = np.concatenate([rng.standard_normal(p * h), np.zeros(h)])
    assert np.allclose(contrast @ oracle, 0.0)
    means = np.einsum("nhd,d->nh", design.base, oracle)
    for _ in range(25):
        targets = means + rng.standard_normal((n, h))
        errors = []
        for lam in (0.0, 0.1, 1.0, 10.0):
            gap = theory.proxy_estimator(design, targets, lam) - oracle
            errors.append(gap @ gram @ gap)
        assert all(
            later <= earlier + 1e-10 for earlier, later in zip(errors, errors[1:], strict=False)
        )


def test_hierarchy_model_geometry_and_closed_form_risk():
    """Proposition B.10: S = I, K = 0 + C_out, and Eq. (B.38) for isotropic noise."""
    rng = np.random.default_rng(4)
    n, m, p = 9, 7, 2
    groups = [[0, 1, 2], [4, 5]]  # candidates 3 and 6 are not covered by any eligible group
    raw = rng.standard_normal((n, p))
    raw -= raw.mean(axis=0)
    q, _ = np.linalg.qr(raw)
    features = q * np.sqrt(n)  # centred and whitened features, Eq. (B.32)
    assert np.allclose(features.mean(axis=0), 0) and np.allclose(
        features.T @ features / n, np.eye(p)
    )
    design = theory.hierarchy_design(features, m, groups)
    gram, contrast = design.gram_matrices()
    h = m - 1
    d = h * (p + 1)
    assert np.allclose(gram, np.eye(d), atol=1e-10)
    basis = theory.centered_basis(m)
    expected = np.zeros((d, d))
    expected[p * h :, p * h :] = basis.T @ contrast_operator(m, groups) @ basis
    assert np.allclose(contrast, expected, atol=1e-10)
    sigma2 = 0.7
    covariances = np.broadcast_to(sigma2 * np.eye(h), (n, h, h))
    means = np.zeros((n, h))
    for lam in (0.0, 0.5, 4.0):
        general = theory.exact_risk(design, means, covariances, lam)
        closed = theory.hierarchy_risk(d, [3, 2], lam, sigma2, n)
        assert general == pytest.approx(closed, rel=1e-9)
    # Baseline and limiting risks of Eq. (B.39).
    baseline = sigma2 / n * (d - 1 + sum(1 / (4 * 2**2 * (b - 1)) for b in (3, 2)))
    assert theory.hierarchy_risk(d, [3, 2], 0.0, sigma2, n) == pytest.approx(baseline)
    limit = sigma2 / n * (d - 3)
    assert theory.hierarchy_risk(d, [3, 2], 1e9, sigma2, n) == pytest.approx(limit, rel=1e-6)
    risks = [theory.hierarchy_risk(d, [3, 2], lam, sigma2, n) for lam in (0, 0.1, 1, 10)]
    assert all(b < a for a, b in zip(risks, risks[1:], strict=False))  # every lambda > 0 helps
