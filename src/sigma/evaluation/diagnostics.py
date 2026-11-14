"""Diagnostics of the paper beyond accuracy: margin-residual certificates (Section 5.4,
Appendix D.7), spectral residual contraction (Appendix D.8), LoRA knockouts (Appendix D.9)
and the preliminary study (Section 3, Appendix C.7).  Pure NumPy.

Score vectors are in the fixed candidate order; ``permutation[c] = pi_e(c)``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from sigma.evaluation.metrics import softmax, top1, transport
from sigma.guidance.schema import TaskRecord


def center(scores: np.ndarray) -> np.ndarray:
    """``u = s - mean(s) 1`` (Eq. (4))."""
    return scores - scores.mean(axis=-1, keepdims=True)


# --------------------------------------------------------------------------- Section 5.4 / D.7
def margin(centered: np.ndarray, target: int) -> float:
    """``gamma = u[y] - max_{c != y} u[c]``."""
    others = np.delete(centered, target)
    return float(centered[target] - others.max())


@dataclass(frozen=True)
class TrajectoryDiagnostic:
    """Margin-residual quantities of one edit trajectory ``g0 -> g1 -> ... -> gk``.

    Attributes
    ----------
    residuals:
        ``eps_j = ||u_j - P_j u_{j-1}||_2`` for ``j = 1..k`` (Eq. (D.1)).
    local_error:
        ``E_local = sum_j eps_j``.
    initial_margin:
        ``gamma_0`` of the correct initial output (Eq. (D.2)).
    ratio:
        ``R_mr = sqrt(2) E_local / gamma_0`` (``inf`` if ``gamma_0 <= 0``).
    certified:
        ``R_mr < 1``: Corollary B.4 then guarantees a unique correct top-1 at every prefix.
    targets:
        Transported targets ``y_0, ..., y_k``.
    prefix_margins:
        ``gamma_j = u_j[y_j] - max_{c != y_j} u_j[c]`` for ``j = 0..k``.
    final_correct:
        Whether ``y_k`` is top-ranked under ``g_k``.
    accumulated:
        ``||u_k - P_k ... P_1 u_0||_2``, the left-hand side of Eq. (7).
    """

    residuals: np.ndarray
    local_error: float
    initial_margin: float
    ratio: float
    certified: bool
    targets: tuple[int, ...]
    prefix_margins: np.ndarray
    final_correct: bool
    accumulated: float

    def bound_holds(self, tolerance: float = 1e-8) -> bool:
        """Check Eq. (7) and the prefix margin bound of Corollary B.4 for this trajectory."""
        prefix = np.concatenate([[0.0], np.cumsum(self.residuals)])
        margins_ok = np.all(
            self.prefix_margins >= self.initial_margin - np.sqrt(2.0) * prefix - tolerance
        )
        return bool(self.accumulated <= self.local_error + tolerance and margins_ok)


def trajectory_diagnostic(
    scores: np.ndarray, permutations: Sequence[Sequence[int]], initial_target: int
) -> TrajectoryDiagnostic:
    """Reconstruct the margin-residual diagnostic from saved scores and swaps.

    Parameters
    ----------
    scores:
        ``(k + 1, m)`` candidate scores under ``g_0, ..., g_k``.
    permutations:
        ``k`` single-step permutations ``pi_{e_j}``.
    initial_target:
        Candidate index of the correct output ``y_0`` under ``g_0``.
    """
    centered = center(np.asarray(scores, dtype=np.float64))
    perms = [np.asarray(p, dtype=np.int64) for p in permutations]
    if len(perms) != len(centered) - 1:
        raise ValueError("need exactly one permutation per step")
    residuals = np.array(
        [np.linalg.norm(centered[j + 1] - transport(centered[j], p)) for j, p in enumerate(perms)]
    )
    targets = [int(initial_target)]
    transported = centered[0]
    for p in perms:
        targets.append(int(p[targets[-1]]))
        transported = transport(transported, p)
    margins = np.array([margin(u, y) for u, y in zip(centered, targets, strict=True)])
    local_error = float(residuals.sum())
    gamma0 = float(margins[0])
    ratio = float(np.sqrt(2.0) * local_error / gamma0) if gamma0 > 0 else float("inf")
    return TrajectoryDiagnostic(
        residuals=residuals,
        local_error=local_error,
        initial_margin=gamma0,
        ratio=ratio,
        certified=ratio < 1.0,
        targets=tuple(targets),
        prefix_margins=margins,
        final_correct=int(top1(centered[-1])) == targets[-1],
        accumulated=float(np.linalg.norm(centered[-1] - transported)),
    )


def summarize_trajectories(diagnostics: Sequence[TrajectoryDiagnostic]) -> dict[str, float]:
    """Certified coverage, pooled final accuracy and errors inside the certified region."""
    certified = np.array([d.certified for d in diagnostics])
    correct = np.array([d.final_correct for d in diagnostics])
    outside = ~certified
    return {
        "count": len(diagnostics),
        "accuracy": 100.0 * float(correct.mean()),
        "certified_coverage": 100.0 * float(certified.mean()),
        "certified_count": int(certified.sum()),
        "certified_errors": int((certified & ~correct).sum()),
        "accuracy_outside": 100.0 * float(correct[outside].mean())
        if outside.any()
        else float("nan"),
        "bound_violations": int(sum(not d.bound_holds() for d in diagnostics)),
    }


# --------------------------------------------------------------------------- Appendix D.8
def sibling_index_groups(task: TaskRecord) -> list[list[int]]:
    """Candidate indices of the outputs bound (under ``g0``) to each eligible sibling group."""
    return [
        sorted(task.candidate_index[task.binding[z]] for z in group)
        for group in task.sibling_groups()
    ]


def contrast_operator(m: int, groups: Sequence[Sequence[int]]) -> np.ndarray:
    """``C_out = sum_v c_v Pi_v`` with ``c_v = 1 / (2 J (b_v - 1))`` (Eq. (B.33), (D.3)).

    ``Pi_v`` projects onto the zero-sum contrasts within group ``G_v``.  The factor 1/2 in
    ``c_v`` comes from the equal mixture of binding and representation edits.
    """
    operator = np.zeros((m, m))
    for group in groups:
        b = len(group)
        indicator = np.zeros(m)
        indicator[list(group)] = 1.0
        projector = np.diag(indicator) - np.outer(indicator, indicator) / b
        operator += projector / (2.0 * len(groups) * (b - 1))
    return operator


def edit_averaged_contrast(m: int, groups: Sequence[Sequence[int]]) -> np.ndarray:
    """``E_{e ~ rho_T}[1/4 (P_e - I)^T (P_e - I)]`` by explicit enumeration of the edit law.

    Equals :func:`contrast_operator`; kept as an independent computation of Eq. (D.3).
    """
    total = np.zeros((m, m))
    for group in groups:
        pairs = [(a, b) for i, a in enumerate(group) for b in list(group)[i + 1 :]]
        for a, b in pairs:
            difference = np.zeros((m, m))  # P_e - I for the transposition (a b)
            difference[a, a] = difference[b, b] = -1.0
            difference[a, b] = difference[b, a] = 1.0
            total += 0.5 * difference.T @ difference / (4.0 * len(groups) * len(pairs))
    return total  # the representation edit contributes zero


@dataclass(frozen=True)
class SpectralBasis:
    """Orthonormal basis of the centred score space with intervention eigenvalues.

    ``directions`` has shape ``(m - 1, m)``; ``eigenvalues[j]`` is ``kappa_j`` (``c_v`` for
    the Helmert contrasts of group ``v``, zero for the null space of ``C_out``).
    """

    directions: np.ndarray
    eigenvalues: np.ndarray

    @property
    def exposed(self) -> np.ndarray:
        return self.eigenvalues > 0


def helmert_contrasts(m: int, group: Sequence[int]) -> np.ndarray:
    """Helmert contrasts of one group in its fixed candidate order (Eq. (D.4)).

    ``v_r = (sum_{l <= r} e_{g_l} - r e_{g_{r+1}}) / sqrt(r (r + 1))`` for ``r = 1..b - 1``.
    """
    ordered = sorted(group)
    rows = np.zeros((len(ordered) - 1, m))
    for r in range(1, len(ordered)):
        rows[r - 1, ordered[:r]] = 1.0
        rows[r - 1, ordered[r]] = -float(r)
        rows[r - 1] /= np.sqrt(r * (r + 1.0))
    return rows


def spectral_basis(m: int, groups: Sequence[Sequence[int]]) -> SpectralBasis:
    """Helmert contrasts of every sibling group plus a basis of ``ker(C_out)`` within ``1^perp``.

    The null-space basis consists of Helmert-type contrasts between the blocks on which
    null vectors are constant (the sibling groups, then every uncovered candidate), taken in
    order of their smallest candidate index.  Within a repeated eigenspace the directional
    energies depend on this choice; subspace totals do not.
    """
    exposed = [helmert_contrasts(m, group) for group in groups]
    eigen = [np.full(len(g) - 1, 1.0 / (2.0 * len(groups) * (len(g) - 1))) for g in groups]
    covered = {c for group in groups for c in group}
    blocks = [sorted(group) for group in groups] + [[c] for c in range(m) if c not in covered]
    blocks.sort(key=lambda block: block[0])
    null = np.zeros((len(blocks) - 1, m))
    for r in range(1, len(blocks)):
        left = [c for block in blocks[:r] for c in block]
        right = blocks[r]
        null[r - 1, left] = 1.0 / len(left)
        null[r - 1, right] = -1.0 / len(right)
        null[r - 1] /= np.linalg.norm(null[r - 1])
    directions = np.concatenate([*exposed, null]) if exposed else null
    eigenvalues = np.concatenate([*eigen, np.zeros(len(null))]) if eigen else np.zeros(len(null))
    return SpectralBasis(directions, eigenvalues)


def aligned_residuals(
    scores_base: np.ndarray, scores_edit: np.ndarray, permutations: np.ndarray
) -> np.ndarray:
    """``delta = u_{g'} - P_e u_g`` for a batch of evaluation pairs, shape ``(pairs, m)``."""
    return center(scores_edit) - transport(center(scores_base), permutations)


def directional_energy(residuals: np.ndarray, directions: np.ndarray) -> np.ndarray:
    """``E_j = mean over pairs of (v_j^T delta)^2`` (Eq. (D.5))."""
    return ((residuals @ directions.T) ** 2).mean(axis=0)


def log_energy_ratio(
    energy: np.ndarray, reference: np.ndarray, epsilon: float = 1e-12
) -> np.ndarray:
    """``R_j = log2((E_j + eps) / (E_j^ref + eps))``; negative values mean lower energy."""
    return np.log2((energy + epsilon) / (reference + epsilon))


# --------------------------------------------------------------------------- Appendix D.9
def knockout_effects(
    full_base: float,
    full_comp: float,
    knockout_base: float,
    knockout_comp: float,
    full_tv: float,
    knockout_tv: float,
) -> tuple[float, float]:
    """Differential knockout effects ``(I_guide, I_eq)`` (Eq. (D.7), (D.8)).

    ``I_guide = (A_comp^full - A_comp^ko) - (A_base^full - A_base^ko)`` in percentage points;
    ``I_eq = TV_eq^ko - TV_eq^full`` on the absolute TV scale.
    """
    return (full_comp - knockout_comp) - (full_base - knockout_base), knockout_tv - full_tv


# --------------------------------------------------------------------------- Appendix C.7
def _ranks(scores: np.ndarray) -> np.ndarray:
    """1-based rank of every candidate (descending score, ties by candidate index)."""
    order = np.argsort(-scores, axis=-1, kind="stable")
    ranks = np.empty_like(order)
    np.put_along_axis(
        ranks, order, np.broadcast_to(np.arange(1, scores.shape[-1] + 1), order.shape), -1
    )
    return ranks


def aligned_rank_transitions(
    scores_base: np.ndarray, scores_edit: np.ndarray, permutations: np.ndarray, top: int = 10
) -> np.ndarray:
    """Row-normalised candidate-rank transition matrix of one run (Eq. (C.5), (C.6)).

    The edited scores are first aligned to original coordinates,
    ``q~(c) = q_{g'}(pi(c))``.  Entry ``[r - 1, s - 1]`` is the frequency with which a
    candidate of original rank ``r <= top`` has aligned rank ``s``; the last column collects
    destination ranks greater than ``top``.
    """
    aligned = np.take_along_axis(scores_edit, permutations, axis=-1)
    source, destination = _ranks(scores_base), _ranks(aligned)
    counts = np.zeros((top, top + 1))
    for r in range(1, top + 1):
        ranks = np.minimum(destination[source == r], top + 1)
        counts[r - 1] = np.bincount(ranks - 1, minlength=top + 1)
    return counts / counts.sum(axis=1, keepdims=True)


def rank_retention(transitions: np.ndarray) -> tuple[float, float]:
    """``(R_exact, R_{+-1, <=top})`` of Eq. (C.7), as fractions."""
    top = transitions.shape[0]
    exact = float(np.trace(transitions[:, :top]) / top)
    local = sum(transitions[r, s] for r in range(top) for s in range(top) if abs(s - r) <= 1)
    return exact, float(local / top)


def normalized_entropy(scores: np.ndarray) -> np.ndarray:
    """``H_i = -(1 / log m) sum_c q(c) log q(c)`` of the candidate distribution (Eq. (C.8))."""
    q = softmax(scores)
    return -(q * np.log(np.clip(q, 1e-300, None))).sum(axis=-1) / np.log(scores.shape[-1])


def upper_percentiles(values: np.ndarray) -> np.ndarray:
    """``u_i = (100 / N) sum_l 1[H_l <= H_i]``: ties receive their upper empirical percentile."""
    ordered = np.sort(values)
    return 100.0 * np.searchsorted(ordered, values, side="right") / len(values)


def uncertainty_bins(percentiles: np.ndarray, num_bins: int = 12) -> np.ndarray:
    """Equal-width bins on ``[0, 100]``; internal boundaries go right, 100 goes to the last bin."""
    bins = np.floor(percentiles * num_bins / 100.0).astype(int)
    return np.clip(bins, 0, num_bins - 1)


def response_error(
    pred_base: np.ndarray,
    pred_binding: np.ndarray,
    permutation: np.ndarray,
    pred_order: np.ndarray,
) -> np.ndarray:
    """Union response error of Eq. (C.9): the binding swap is not followed, or the answer
    changes under the order-only revision."""
    followed = np.take_along_axis(permutation, pred_base[:, None], axis=-1)[:, 0]
    return (pred_binding != followed) | (pred_order != pred_base)


def binned_error_rates(
    errors: np.ndarray, bins: np.ndarray, num_bins: int = 12
) -> tuple[np.ndarray, np.ndarray]:
    """Counts ``N_h`` and pooled error rates ``p_h`` per uncertainty bin (Eq. (C.10))."""
    counts = np.bincount(bins, minlength=num_bins).astype(np.float64)
    failures = np.bincount(bins, weights=errors.astype(np.float64), minlength=num_bins)
    rates = np.divide(failures, counts, out=np.full(num_bins, np.nan), where=counts > 0)
    return counts.astype(int), rates
