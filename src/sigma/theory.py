"""The fixed-feature squared-loss proxy of Section 4.4 (Appendix B.3 - B.7), in NumPy.

The proxy replaces each cross-entropy of Eq. (5) by a candidate-normalised squared loss and
keeps the discrepancy coefficient ``lambda / (2m)``; after multiplying by ``m`` the objective
is Eq. (B.6).  This module implements its closed-form minimiser (Lemma B.5), the exact risk
with reused labels (Theorem B.6) and the explicit hierarchy model of Appendix B.6, so that
the statements of the paper can be checked numerically (see ``tests/test_theory.py``).

It is an analysis tool and is not used by the training pipeline.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ProxyDesign:
    """Fixed designs and edit law of the linear proxy (Appendix B.3).

    Attributes
    ----------
    base:
        ``A_i``, shape ``(n, h, d)``.
    aligned:
        ``B_ie = R_e^T A'_ie``, shape ``(n, E, h, d)``: edited designs transported back to
        base coordinates.
    law:
        Edit probabilities ``rho(e)``, shape ``(E,)`` (the same finite law for every example).
    """

    base: np.ndarray
    aligned: np.ndarray
    law: np.ndarray

    def __post_init__(self) -> None:
        n, h, d = self.base.shape
        if self.aligned.shape[0] != n or self.aligned.shape[2:] != (h, d):
            raise ValueError("aligned designs must have shape (n, E, h, d)")
        if self.law.shape != (self.aligned.shape[1],) or not np.isclose(self.law.sum(), 1.0):
            raise ValueError("the edit law must be a probability vector over the E edits")

    @property
    def mean_view(self) -> np.ndarray:
        """``M_ie = (A_i + B_ie) / 2``, shape ``(n, E, h, d)``."""
        return 0.5 * (self.base[:, None] + self.aligned)

    @property
    def contrast(self) -> np.ndarray:
        """``D_ie = (B_ie - A_i) / 2``, shape ``(n, E, h, d)``."""
        return 0.5 * (self.aligned - self.base[:, None])

    @property
    def averaged_mean_view(self) -> np.ndarray:
        """``Mbar_i = E_e M_ie``, shape ``(n, h, d)``."""
        return np.einsum("e,nehd->nhd", self.law, self.mean_view)

    def gram_matrices(self) -> tuple[np.ndarray, np.ndarray]:
        """Paired Gram matrix ``S`` and aligned contrast Gram matrix ``C`` (Eq. (B.7))."""
        n = self.base.shape[0]
        contrast = np.einsum("e,nehd,nehf->df", self.law, self.contrast, self.contrast) / n
        mean = np.einsum("e,nehd,nehf->df", self.law, self.mean_view, self.mean_view) / n
        return mean + contrast, contrast


def proxy_objective(design: ProxyDesign, w: np.ndarray, targets: np.ndarray, lam: float) -> float:
    """``L_lambda(w)`` of Eq. (B.6) for responses ``targets`` of shape ``(n, h)``."""
    n = design.base.shape[0]
    base_error = np.einsum("nhd,d->nh", design.base, w) - targets
    edit_error = np.einsum("nehd,d->neh", design.aligned, w) - targets[:, None]
    gap = np.einsum("nehd,d->neh", design.aligned - design.base[:, None], w)
    supervised = (base_error**2).sum() + np.einsum("e,neh->", design.law, edit_error**2)
    penalty = np.einsum("e,neh->", design.law, gap**2)
    return float(supervised / (4 * n) + lam * penalty / (2 * n))


def proxy_estimator(design: ProxyDesign, targets: np.ndarray, lam: float) -> np.ndarray:
    """``w_lambda = (S + tau C)^{-1} Mbar^T t / n`` with ``tau = 4 lambda`` (Lemma B.5)."""
    gram, contrast = design.gram_matrices()
    n = design.base.shape[0]
    rhs = np.einsum("nhd,nh->d", design.averaged_mean_view, targets) / n
    return np.linalg.solve(gram + 4.0 * lam * contrast, rhs)


def _inverse_sqrt(matrix: np.ndarray) -> np.ndarray:
    values, vectors = np.linalg.eigh(matrix)
    return (vectors / np.sqrt(values)) @ vectors.T


def exact_risk(
    design: ProxyDesign, means: np.ndarray, covariances: np.ndarray, lam: float
) -> float:
    """Exact excess risk ``R_lambda = E ||w_lambda - w_dagger||_S^2`` (Theorem B.6).

    ``R_lambda = sum_j [(tau kappa_j beta_j)^2 + omega_j] / (1 + tau kappa_j)^2`` in an
    eigenbasis of ``K = S^{-1/2} C S^{-1/2}``, where ``omega`` is the whitened covariance of
    the original-label noise *after* edit averaging (labels are reused across edits).

    Parameters
    ----------
    means:
        Response means ``mu_i``, shape ``(n, h)``.
    covariances:
        Noise covariances ``Sigma_i``, shape ``(n, h, h)``.
    """
    gram, contrast = design.gram_matrices()
    n = design.base.shape[0]
    whitener = _inverse_sqrt(gram)
    kappa, basis = np.linalg.eigh(whitener @ contrast @ whitener)
    kappa = np.clip(kappa, 0.0, None)
    averaged = design.averaged_mean_view
    oracle = np.linalg.solve(gram, np.einsum("nhd,nh->d", averaged, means) / n)
    beta = basis.T @ (np.linalg.inv(whitener) @ oracle)
    noise = np.einsum("nhd,nhk,nkf->df", averaged, covariances, averaged) / n**2
    omega = np.einsum("dj,df,fj->j", basis, whitener @ noise @ whitener, basis)
    tau = 4.0 * lam
    return float(np.sum(((tau * kappa * beta) ** 2 + omega) / (1.0 + tau * kappa) ** 2))


def hierarchy_risk(
    dimension: int, group_sizes: Sequence[int], lam: float, noise_variance: float, n: int
) -> float:
    """Closed-form risk of the hierarchy model under SIGMA's edit law (Eq. (B.38)).

    ``R_lambda = sigma^2 / n * [d - r + sum_v (b_v - 1) (1 - c_v)^2 / (1 + 4 lambda c_v)^2]``
    with ``c_v = 1 / (2 J (b_v - 1))`` and ``r = sum_v (b_v - 1)``.
    """
    sizes = np.asarray(group_sizes, dtype=np.float64)
    c = 1.0 / (2.0 * len(sizes) * (sizes - 1.0))
    r = float(np.sum(sizes - 1.0))
    shrunk = np.sum((sizes - 1.0) * (1.0 - c) ** 2 / (1.0 + 4.0 * lam * c) ** 2)
    return float(noise_variance / n * (dimension - r + shrunk))


def centered_basis(m: int) -> np.ndarray:
    """``Q`` in ``R^{m x (m-1)}`` with orthonormal columns spanning ``1^perp`` (Appendix B.3)."""
    q, _ = np.linalg.qr(np.eye(m) - np.full((m, m), 1.0 / m))
    return q[:, : m - 1]


def hierarchy_design(features: np.ndarray, m: int, groups: Sequence[Sequence[int]]) -> ProxyDesign:
    """Design of the model ``f_{W,a}(x, g) = P_g W phi(x) + a`` under the sibling-edit law.

    Parameters are ``(vec(W), a)`` in the orthonormal coordinates of ``1^perp``; the law puts
    probability 1/2 on the representation edit (``P_e = I``) and
    ``[2 J C(b_v, 2)]^{-1}`` on every transposition within group ``v`` (Appendix B.6).
    """
    n, p = features.shape
    h = m - 1
    q = centered_basis(m)
    permutations = [np.eye(m)]
    law = [0.5]
    for group in groups:
        pairs = [(a, b) for i, a in enumerate(group) for b in list(group)[i + 1 :]]
        for a, b in pairs:
            matrix = np.eye(m)
            matrix[[a, b]] = matrix[[b, a]]
            permutations.append(matrix)
            law.append(0.5 / (len(groups) * len(pairs)))
    semantic = np.einsum("np,hk->nhpk", features, np.eye(h)).reshape(n, h, p * h)
    base = np.concatenate([semantic, np.broadcast_to(np.eye(h), (n, h, h))], axis=-1)
    aligned = np.stack(
        [
            np.concatenate([semantic, np.broadcast_to((q.T @ perm @ q).T, (n, h, h))], axis=-1)
            for perm in permutations
        ],
        axis=1,
    )
    return ProxyDesign(base, aligned, np.array(law))
