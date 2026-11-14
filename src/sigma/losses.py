"""The SIGMA objective and its controls (Section 4.2, Appendix C.4).

All functions operate on batches of candidate-score vectors in the fixed candidate order:
``scores_base`` and ``scores_edit`` have shape ``(batch, m)`` and hold ``s_g`` and ``s_{g'}``
of Eq. (3); ``permutation`` has shape ``(batch, m)`` with ``permutation[i, c] = pi_e(c)``.

For an identity pair the caller passes the *same* score tensor in both positions together
with the identity permutation.  The paired loss then reduces to exactly ``-log q(y | x, g)``
with zero equivariance penalty (Section 4.2, Appendix A.1).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor


def center_scores(scores: Tensor) -> Tensor:
    """``u_g = s_g - (1^T s_g / m) 1`` (Eq. (4)): remove the softmax shift freedom."""
    return scores - scores.mean(dim=-1, keepdim=True)


def transport(vector: Tensor, permutation: Tensor) -> Tensor:
    """Apply the permutation matrix of Eq. (2): ``[P_e v]_c = v_{pi_e^{-1}(c)}``.

    Implemented by candidate-index scattering, ``(P_e v)[pi_e(c)] = v[c]``, in ``O(m)``
    without building a dense matrix (Appendix A.2, "Cost accounting").
    """
    permutation = permutation.expand_as(vector)
    return torch.empty_like(vector).scatter(-1, permutation, vector)


def candidate_log_probs(scores: Tensor) -> Tensor:
    """``log q_g = log softmax(s_g)`` with unit temperature and log-sum-exp normalisation."""
    return F.log_softmax(scores, dim=-1)


def equivariance_residual(scores_base: Tensor, scores_edit: Tensor, permutation: Tensor) -> Tensor:
    """``delta = u_{g'} - P_e u_g`` per instance, shape ``(batch, m)``."""
    return center_scores(scores_edit) - transport(center_scores(scores_base), permutation)


def equivariance_penalty(scores_base: Tensor, scores_edit: Tensor, permutation: Tensor) -> Tensor:
    """``||u_{g'} - P_e u_g||_2^2 / m`` per instance (the alignment term of Eq. (5) without
    its coefficient ``lambda / 2``)."""
    delta = equivariance_residual(scores_base, scores_edit, permutation)
    return delta.pow(2).sum(dim=-1) / scores_base.shape[-1]


def transported_kl(scores_base: Tensor, scores_edit: Tensor, permutation: Tensor) -> Tensor:
    """``KL(P_e q_g || q_{g'}) + KL(q_{g'} || P_e q_g)`` per instance (Appendix C.4).

    This is the symmetric distribution-matching control adapted from R-Drop; it replaces the
    centred-score penalty and is multiplied by ``beta / 2`` in :func:`paired_loss`.
    """
    log_p = transport(candidate_log_probs(scores_base), permutation)
    log_q = candidate_log_probs(scores_edit)
    forward = (log_p.exp() * (log_p - log_q)).sum(dim=-1)
    backward = (log_q.exp() * (log_q - log_p)).sum(dim=-1)
    return forward + backward


@dataclass(frozen=True)
class PairedLossOutput:
    """Batch-averaged paired loss and its components (all scalars)."""

    loss: Tensor
    ce_base: Tensor
    ce_edit: Tensor
    consistency: Tensor


def paired_loss(
    scores_base: Tensor,
    scores_edit: Tensor,
    permutation: Tensor,
    target_base: Tensor,
    target_edit: Tensor,
    coefficient: float = 1.0,
    consistency: str = "centered_l2",
) -> PairedLossOutput:
    """Minibatch average of the paired objective, Eq. (5).

    ``centered_l2`` (SIGMA)::

        l(i, e) = -1/2 log q(y_i | x_i, g_i) - 1/2 log q(pi_e(y_i) | x_i, g'_i)
                  + lambda / (2 m) * || u_{g'_i} - P_e u_{g_i} ||_2^2

    ``transported_kl`` replaces the last term by
    ``beta / 2 * {KL(P_e q_g || q_g') + KL(q_g' || P_e q_g)}`` (Appendix C.4), and ``none`` or
    ``coefficient == 0`` gives paired supervised fine-tuning on the same views.

    Parameters
    ----------
    scores_base, scores_edit:
        ``s_g`` and ``s_{g'}``, shape ``(batch, m)``.  Both receive gradients.
    permutation:
        ``pi_e`` as candidate indices, shape ``(batch, m)``.
    target_base, target_edit:
        Candidate indices of ``y_i`` and ``y'_i = pi_e(y_i)``, shape ``(batch,)``.
    coefficient:
        ``lambda`` (or ``beta`` for the transported-KL control); must be non-negative.
    consistency:
        ``"centered_l2"``, ``"transported_kl"`` or ``"none"``.
    """
    if coefficient < 0:
        raise ValueError("the loss coefficient must be non-negative")
    ce_base = F.cross_entropy(scores_base, target_base, reduction="none")
    ce_edit = F.cross_entropy(scores_edit, target_edit, reduction="none")
    if consistency == "centered_l2":
        term = equivariance_penalty(scores_base, scores_edit, permutation)
    elif consistency == "transported_kl":
        term = transported_kl(scores_base, scores_edit, permutation)
    elif consistency == "none":
        term = torch.zeros_like(ce_base)
    else:
        raise ValueError(f"unknown consistency term {consistency!r}")
    per_instance = 0.5 * ce_base + 0.5 * ce_edit + 0.5 * coefficient * term
    return PairedLossOutput(
        loss=per_instance.mean(),
        ce_base=ce_base.mean(),
        ce_edit=ce_edit.mean(),
        consistency=term.mean(),
    )


def single_view_loss(scores: Tensor, target: Tensor) -> Tensor:
    """Candidate-ranking cross-entropy under one guidance view (single-view LoRA-SFT)."""
    return F.cross_entropy(scores, target)
