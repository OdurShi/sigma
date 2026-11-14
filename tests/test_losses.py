"""Every term of the SIGMA objective (Section 4.2, Appendix B.1, Appendix C.4)."""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from sigma import losses

torch.manual_seed(0)


def _random_permutations(batch: int, m: int, generator: torch.Generator) -> torch.Tensor:
    return torch.stack([torch.randperm(m, generator=generator) for _ in range(batch)])


def _permutation_matrix(perm: torch.Tensor) -> torch.Tensor:
    """Dense P_e with [P_e v]_c = v_{pi^{-1}(c)}, i.e. P[pi(c), c] = 1."""
    m = len(perm)
    matrix = torch.zeros(m, m, dtype=torch.float64)
    matrix[perm, torch.arange(m)] = 1.0
    return matrix


def test_centering_removes_the_common_shift():
    scores = torch.tensor([[1.0, 2.0, 6.0], [0.5, 0.5, 0.5]])
    centered = losses.center_scores(scores)
    assert torch.allclose(centered, torch.tensor([[-2.0, -1.0, 3.0], [0.0, 0.0, 0.0]]))
    assert torch.allclose(losses.center_scores(scores + 7.3), centered, atol=1e-6)


def test_transport_matches_the_permutation_matrix_of_equation_2():
    generator = torch.Generator().manual_seed(1)
    perm = torch.randperm(7, generator=generator)
    vector = torch.randn(7, dtype=torch.float64, generator=generator)
    transported = losses.transport(vector[None], perm[None])[0]
    assert torch.allclose(transported, _permutation_matrix(perm) @ vector)
    # [P_e v]_c = v_{pi_e^{-1}(c)}  <=>  [P_e v]_{pi_e(c)} = v_c.
    for c in range(7):
        assert transported[perm[c]] == vector[c]


def test_hand_computed_two_candidate_case():
    # s_g = (2, 0), s_g' = (0.5, 1.5), pi swaps the two candidates, y = 0, y' = 1, lambda = 1.
    # CE terms: -1/2 log sigmoid(2) - 1/2 log sigmoid(1).
    # u_g = (1, -1), P u_g = (-1, 1), u_g' = (-0.5, 0.5): ||delta||^2 = 0.5, m = 2,
    # so the penalty is lambda / (2 m) * 0.5 = 0.125.
    out = losses.paired_loss(
        torch.tensor([[2.0, 0.0]]),
        torch.tensor([[0.5, 1.5]]),
        torch.tensor([[1, 0]]),
        torch.tensor([0]),
        torch.tensor([1]),
        coefficient=1.0,
    )
    ce = -0.5 * math.log(1 / (1 + math.exp(-2))) - 0.5 * math.log(1 / (1 + math.exp(-1)))
    assert out.loss.item() == pytest.approx(ce + 0.125, abs=1e-6)
    assert out.consistency.item() == pytest.approx(0.25, abs=1e-6)  # ||delta||^2 / m
    assert out.ce_base.item() == pytest.approx(-math.log(1 / (1 + math.exp(-2))), abs=1e-6)


def test_hand_computed_three_candidate_case():
    # pi: 0 -> 1, 1 -> 2, 2 -> 0 (a three-cycle), so P u_g = (u_g[2], u_g[0], u_g[1]).
    s_g = torch.tensor([[3.0, 0.0, 0.0]], dtype=torch.float64)  # u_g = (2, -1, -1)
    s_e = torch.tensor([[0.0, 1.0, 2.0]], dtype=torch.float64)  # u_g' = (-1, 0, 1)
    perm = torch.tensor([[1, 2, 0]])
    # P u_g = (-1, 2, -1); delta = (0, -2, 2); ||delta||^2 = 8; lambda/(2m) * 8 = 0.5 * 8 / 6.
    lam = 0.5
    out = losses.paired_loss(s_g, s_e, perm, torch.tensor([0]), torch.tensor([1]), lam)
    ce_base = math.log(math.exp(3) + 2) - 3
    ce_edit = math.log(1 + math.e + math.e**2) - 1
    expected = 0.5 * ce_base + 0.5 * ce_edit + lam / (2 * 3) * 8.0
    assert out.loss.item() == pytest.approx(expected, abs=1e-12)
    delta = losses.equivariance_residual(s_g, s_e, perm)
    assert torch.allclose(delta, torch.tensor([[0.0, -2.0, 2.0]], dtype=torch.float64))


def test_matches_the_dense_matrix_formula_on_random_batches():
    generator = torch.Generator().manual_seed(2)
    batch, m, lam = 5, 9, 0.7
    s_g = torch.randn(batch, m, dtype=torch.float64, generator=generator)
    s_e = torch.randn(batch, m, dtype=torch.float64, generator=generator)
    perms = _random_permutations(batch, m, generator)
    y = torch.randint(m, (batch,), generator=generator)
    y_edit = perms[torch.arange(batch), y]
    out = losses.paired_loss(s_g, s_e, perms, y, y_edit, lam)
    total = 0.0
    centering = torch.eye(m, dtype=torch.float64) - torch.full((m, m), 1.0 / m, dtype=torch.float64)
    for i in range(batch):
        q_g, q_e = torch.softmax(s_g[i], 0), torch.softmax(s_e[i], 0)
        residual = centering @ s_e[i] - _permutation_matrix(perms[i]) @ (centering @ s_g[i])
        total += (
            -0.5 * torch.log(q_g[y[i]])
            - 0.5 * torch.log(q_e[y_edit[i]])
            + lam / (2 * m) * residual.pow(2).sum()
        )
    assert out.loss.item() == pytest.approx((total / batch).item(), abs=1e-12)


@pytest.mark.parametrize("consistency", ["centered_l2", "transported_kl", "none"])
@pytest.mark.parametrize("coefficient", [0.0, 1.0, 10.0])
def test_identity_pair_reduces_to_one_cross_entropy(consistency, coefficient):
    """Identical views: zero residual and exactly -log q(y | x, g) (Section 4.2, App. A.1)."""
    generator = torch.Generator().manual_seed(3)
    scores = torch.randn(4, 6, dtype=torch.float64, generator=generator)
    identity = torch.arange(6).expand(4, 6)
    y = torch.randint(6, (4,), generator=generator)
    out = losses.paired_loss(scores, scores, identity, y, y, coefficient, consistency)
    assert out.consistency.item() == pytest.approx(0.0, abs=1e-12)
    assert out.loss.item() == pytest.approx(F.cross_entropy(scores, y).item(), abs=1e-12)


def test_zero_coefficient_is_paired_supervised_fine_tuning():
    generator = torch.Generator().manual_seed(4)
    s_g = torch.randn(3, 5, generator=generator)
    s_e = torch.randn(3, 5, generator=generator)
    perms = _random_permutations(3, 5, generator)
    y = torch.tensor([0, 3, 4])
    y_edit = perms[torch.arange(3), y]
    expected = 0.5 * F.cross_entropy(s_g, y) + 0.5 * F.cross_entropy(s_e, y_edit)
    for consistency in ("centered_l2", "transported_kl", "none"):
        out = losses.paired_loss(s_g, s_e, perms, y, y_edit, 0.0, consistency)
        assert out.loss.item() == pytest.approx(expected.item(), abs=1e-6)
    with pytest.raises(ValueError):
        losses.paired_loss(s_g, s_e, perms, y, y_edit, -1.0)
    with pytest.raises(ValueError):
        losses.paired_loss(s_g, s_e, perms, y, y_edit, 1.0, "unknown")


def test_loss_is_invariant_to_score_shifts_in_either_view():
    generator = torch.Generator().manual_seed(5)
    s_g = torch.randn(4, 7, dtype=torch.float64, generator=generator)
    s_e = torch.randn(4, 7, dtype=torch.float64, generator=generator)
    perms = _random_permutations(4, 7, generator)
    y = torch.randint(7, (4,), generator=generator)
    y_edit = perms[torch.arange(4), y]
    shifts_g = torch.randn(4, 1, dtype=torch.float64, generator=generator) * 5
    shifts_e = torch.randn(4, 1, dtype=torch.float64, generator=generator) * 5
    for consistency in ("centered_l2", "transported_kl"):
        reference = losses.paired_loss(s_g, s_e, perms, y, y_edit, 2.0, consistency).loss
        shifted = losses.paired_loss(
            s_g + shifts_g, s_e + shifts_e, perms, y, y_edit, 2.0, consistency
        ).loss
        assert shifted.item() == pytest.approx(reference.item(), abs=1e-10)


def test_penalty_vanishes_exactly_for_equivariant_predictions():
    """q_g' = P_e q_g  <=>  u_g' = P_e u_g (Eq. (4), Proposition B.1)."""
    generator = torch.Generator().manual_seed(6)
    s_g = torch.randn(3, 8, dtype=torch.float64, generator=generator)
    perms = _random_permutations(3, 8, generator)
    s_e = losses.transport(s_g, perms) + torch.tensor([[1.5], [-2.0], [0.3]], dtype=torch.float64)
    assert torch.allclose(
        losses.equivariance_penalty(s_g, s_e, perms),
        torch.zeros(3, dtype=torch.float64),
        atol=1e-12,
    )
    aligned = losses.transport(torch.softmax(s_g, -1), perms)
    assert torch.allclose(torch.softmax(s_e, -1), aligned, atol=1e-12)
    assert torch.allclose(
        losses.transported_kl(s_g, s_e, perms), torch.zeros(3, dtype=torch.float64), atol=1e-12
    )
    # Any non-constant perturbation gives a strictly positive penalty.
    perturbed = s_e.clone()
    perturbed[:, 0] += 0.5
    assert torch.all(losses.equivariance_penalty(s_g, perturbed, perms) > 0)
    assert torch.all(losses.transported_kl(s_g, perturbed, perms) > 0)


def test_pairwise_log_odds_identity_b2():
    """sum_{c<d} Delta_cd^2 = m ||delta||^2, so the penalty is lambda/(2 m^2) sum Delta^2."""
    generator = torch.Generator().manual_seed(7)
    m = 6
    s_g = torch.randn(1, m, dtype=torch.float64, generator=generator)
    s_e = torch.randn(1, m, dtype=torch.float64, generator=generator)
    perm = torch.randperm(m, generator=generator)[None]
    delta = losses.equivariance_residual(s_g, s_e, perm)[0]
    log_q_g, log_q_e = torch.log_softmax(s_g[0], 0), torch.log_softmax(s_e[0], 0)
    inverse = torch.argsort(perm[0])  # pi^{-1}
    total = 0.0
    for c in range(m):
        for d in range(c + 1, m):
            odds_edit = log_q_e[c] - log_q_e[d]
            odds_base = log_q_g[inverse[c]] - log_q_g[inverse[d]]
            aligned = odds_edit - odds_base
            assert aligned.item() == pytest.approx((delta[c] - delta[d]).item(), abs=1e-12)
            total += aligned**2
    assert total.item() == pytest.approx(m * delta.pow(2).sum().item(), abs=1e-10)
    penalty = losses.equivariance_penalty(s_g, s_e, perm)[0]  # ||delta||^2 / m
    assert (0.5 * penalty).item() == pytest.approx(total.item() / (2 * m**2), abs=1e-12)


def test_distance_to_the_aligned_subspace_b3():
    """||(I - Pi_e) r||^2 = 1/2 ||u_g' - P_e u_g||^2 for r = (u_g, u_g')."""
    generator = torch.Generator().manual_seed(8)
    m = 5
    perm = torch.randperm(m, generator=generator)
    matrix = _permutation_matrix(perm)
    u_g = losses.center_scores(torch.randn(m, dtype=torch.float64, generator=generator))
    u_e = losses.center_scores(torch.randn(m, dtype=torch.float64, generator=generator))
    eye = torch.eye(m, dtype=torch.float64)
    projector = 0.5 * torch.cat([torch.cat([eye, matrix.T], 1), torch.cat([matrix, eye], 1)], 0)
    assert torch.allclose(projector @ projector, projector)
    stacked = torch.cat([u_g, u_e])
    residual = stacked - projector @ stacked
    delta = u_e - matrix @ u_g
    assert residual.pow(2).sum().item() == pytest.approx(0.5 * delta.pow(2).sum().item())


def test_loss_is_invariant_to_relabelling_the_candidate_order():
    generator = torch.Generator().manual_seed(9)
    m = 6
    s_g = torch.randn(2, m, dtype=torch.float64, generator=generator)
    s_e = torch.randn(2, m, dtype=torch.float64, generator=generator)
    perms = _random_permutations(2, m, generator)
    y = torch.tensor([1, 4])
    y_edit = perms[torch.arange(2), y]
    reference = losses.paired_loss(s_g, s_e, perms, y, y_edit, 1.3).loss
    sigma = torch.randperm(m, generator=generator)  # new index of old candidate c is sigma[c]
    inverse = torch.argsort(sigma)
    relabelled = losses.paired_loss(
        s_g[:, inverse],
        s_e[:, inverse],
        sigma[perms[:, inverse]],
        sigma[y],
        sigma[y_edit],
        1.3,
    ).loss
    assert relabelled.item() == pytest.approx(reference.item(), abs=1e-12)


def test_both_branches_receive_gradients_through_centering():
    generator = torch.Generator().manual_seed(10)
    s_g = torch.randn(2, 4, dtype=torch.float64, generator=generator, requires_grad=True)
    s_e = torch.randn(2, 4, dtype=torch.float64, generator=generator, requires_grad=True)
    perms = _random_permutations(2, 4, generator)
    y = torch.tensor([0, 2])
    out = losses.paired_loss(s_g, s_e, perms, y, perms[torch.arange(2), y], 1.0)
    out.loss.backward()
    assert s_g.grad.abs().sum() > 0 and s_e.grad.abs().sum() > 0
    # The gradient of the penalty alone is zero-sum per instance (centering) and satisfies
    # d/du_g = -P^T d/du_g'.
    s_g2 = s_g.detach().clone().requires_grad_(True)
    s_e2 = s_e.detach().clone().requires_grad_(True)
    losses.equivariance_penalty(s_g2, s_e2, perms).sum().backward()
    assert torch.allclose(s_g2.grad.sum(-1), torch.zeros(2, dtype=torch.float64), atol=1e-12)
    assert torch.allclose(losses.transport(s_g2.grad, perms), -s_e2.grad, atol=1e-12)


def test_identity_pair_gradient_equals_single_cross_entropy_gradient():
    scores = torch.tensor([[0.3, -1.2, 2.0, 0.1]], dtype=torch.float64, requires_grad=True)
    identity = torch.arange(4)[None]
    y = torch.tensor([2])
    losses.paired_loss(scores, scores, identity, y, y, 5.0).loss.backward()
    reference = scores.detach().clone().requires_grad_(True)
    F.cross_entropy(reference, y).backward()
    assert torch.allclose(scores.grad, reference.grad, atol=1e-12)


def test_transported_kl_is_the_symmetric_divergence():
    generator = torch.Generator().manual_seed(11)
    s_g = torch.randn(3, 5, dtype=torch.float64, generator=generator)
    s_e = torch.randn(3, 5, dtype=torch.float64, generator=generator)
    perms = _random_permutations(3, 5, generator)
    value = losses.transported_kl(s_g, s_e, perms)
    for i in range(3):
        p = _permutation_matrix(perms[i]) @ torch.softmax(s_g[i], 0)
        q = torch.softmax(s_e[i], 0)
        expected = (p * (p / q).log()).sum() + (q * (q / p).log()).sum()
        assert value[i].item() == pytest.approx(expected.item(), abs=1e-12)
    out = losses.paired_loss(
        s_g, s_e, perms, torch.zeros(3, dtype=torch.long), perms[:, 0], 0.4, "transported_kl"
    )
    ce = 0.5 * F.cross_entropy(s_g, torch.zeros(3, dtype=torch.long)) + 0.5 * F.cross_entropy(
        s_e, perms[:, 0]
    )
    assert out.loss.item() == pytest.approx((ce + 0.4 / 2 * value.mean()).item(), abs=1e-12)
