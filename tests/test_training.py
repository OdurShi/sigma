"""The optimisation procedure of Appendix A.2 / C.5 on the tiny model."""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from sigma.config import OptimConfig, ScoringConfig
from sigma.data import Example
from sigma.guidance import BindingSwap, IdentityEdit, Reversal, flat_task
from sigma.methods import EXTERNAL_BASELINES, METHODS, ExternalBaselineError, resolve_method
from sigma.models import (
    LoraSettings,
    apply_lora,
    build_tiny_backbone,
    count_trainable_parameters,
    load_trainable_state,
    lora_knockout,
    lora_updates,
    make_all_trainable,
    tiny_encoder,
    trainable_state,
)
from sigma.scoring import CandidateScorer
from sigma.training import (
    Trainer,
    checkpoint_steps,
    learning_rate_factor,
    num_training_steps,
)

EXAMPLES = [
    Example("please give my money back now", "refund"),
    Example("i cannot log in today", "login"),
    Example("money back please", "refund"),
    Example("give my money", "charge"),
    Example("log in now", "closure"),
    Example("i cannot give money", "invoice"),
]


def _trainer(scorer, task, method="sigma", mode="direct", seed=0, **optim):
    for parameter in scorer.model.parameters():
        parameter.requires_grad_(True)
    settings = {"batch_size": 4, "max_steps": 4, "lr": 1e-3} | optim
    return Trainer(
        scorer.model,
        scorer,
        task,
        EXAMPLES,
        METHODS[method],
        OptimConfig(**settings),
        ScoringConfig(gradient_mode=mode, max_context_tokens=None),
        seed=seed,
    )


def test_step_budget_matches_appendix_c5():
    # BANKING77 at 10 shots: n = 770, B = 16 -> T = 2 * ceil(7700 / 32) = 482 (about ten passes).
    assert num_training_steps(770, 16) == 482
    assert num_training_steps(1500, 16) == 938
    assert num_training_steps(64, 16) == 40
    for n in (77, 150, 640, 1232, 2400):
        steps = num_training_steps(n, 16)
        assert steps % 2 == 0 and 10 * n <= steps * 16 < 10 * n + 32


def test_checkpoints_are_five_even_steps_including_the_last():
    assert checkpoint_steps(482) == [96, 192, 290, 386, 482]
    assert checkpoint_steps(40) == [8, 16, 24, 32, 40]
    assert checkpoint_steps(4) == [2, 4]
    for total in (2, 10, 38, 482, 938):
        steps = checkpoint_steps(total)
        assert steps[-1] == total and all(s % 2 == 0 for s in steps) and len(steps) <= 5


def test_learning_rate_schedule_has_three_percent_warmup_and_cosine_decay():
    total = 482
    warmup = math.ceil(0.03 * total)  # 15 steps
    factors = [learning_rate_factor(i, total, 0.03) for i in range(total)]
    assert factors[0] == pytest.approx(1 / warmup)
    assert factors[warmup - 1] == pytest.approx(1.0)
    assert factors[warmup] == pytest.approx(1.0)
    assert all(b <= a for a, b in zip(factors[warmup:], factors[warmup + 1 :], strict=False))
    midpoint = warmup + (total - warmup) // 2
    assert factors[midpoint] == pytest.approx(0.5, abs=0.01)
    assert 0 <= factors[-1] < 1e-3


def test_odd_steps_use_binding_swaps_and_even_steps_the_reversal(tiny_scorer, toy_task):
    trainer = _trainer(tiny_scorer, toy_task)
    odd, even = trainer.plan_step(1), trainer.plan_step(2)
    assert all(isinstance(e.edit, BindingSwap) for e in odd.edits)
    assert all(isinstance(e.edit, Reversal) for e in even.edits)
    m = toy_task.num_candidates
    for plan in (odd, even):
        assert plan.permutation.shape == (4, m)
        rows = torch.arange(4)
        assert torch.equal(plan.target_edit, plan.permutation[rows, plan.target_base])  # y' = pi(y)
        assert len(plan.prompts) == 8 and plan.base_view == [0, 2, 4, 6]
    assert torch.equal(even.permutation, torch.arange(m).expand(4, m))
    assert torch.equal(even.target_edit, even.target_base)
    binding_only = _trainer(tiny_scorer, toy_task, "sigma_binding_only")
    reversal_only = _trainer(tiny_scorer, toy_task, "sigma_reversal_only")
    assert all(isinstance(e.edit, BindingSwap) for e in binding_only.plan_step(2).edits)
    assert all(isinstance(e.edit, Reversal) for e in reversal_only.plan_step(1).edits)


def test_compared_methods_share_minibatches_and_edits(tiny_scorer, toy_task):
    plans = {
        name: [_trainer(tiny_scorer, toy_task, name, seed=7).plan_step(s) for s in (1,)]
        for name in ("sft", "paired_sft", "sigma", "transported_kl")
    }
    reference = plans["sigma"][0]
    for name in ("paired_sft", "transported_kl"):
        assert plans[name][0].prompts == reference.prompts  # same instances, same edits
        assert torch.equal(plans[name][0].permutation, reference.permutation)
    single = plans["sft"][0]
    assert single.prompts == reference.prompts[0::2]  # the same original instances
    assert single.edit_view == single.base_view
    other_seed = _trainer(tiny_scorer, toy_task, "sigma", seed=8).plan_step(1)
    assert other_seed.prompts != reference.prompts


def test_identity_pairs_reuse_the_original_score_vector(tiny_scorer):
    flat = flat_task(
        "flat",
        {"refund": "money back", "login": "cannot log in", "charge": "give money"},
        {"refund": "desk red", "login": "team alpha", "charge": "desk green"},
    )
    examples = [e for e in EXAMPLES if e.label in flat.labels]
    for parameter in tiny_scorer.model.parameters():
        parameter.requires_grad_(True)
    trainer = Trainer(
        tiny_scorer.model,
        tiny_scorer,
        flat,
        examples,
        METHODS["sigma"],
        OptimConfig(batch_size=3, max_steps=2),
        ScoringConfig(gradient_mode="direct", max_context_tokens=None),
        seed=0,
    )
    plan = trainer.plan_step(1)  # binding step on a flat tree: V is empty
    assert all(isinstance(e.edit, IdentityEdit) and e.is_identity_pair for e in plan.edits)
    assert plan.edit_view == plan.base_view and len(plan.prompts) == 3
    scores = tiny_scorer.score(plan.prompts, trainer._candidates)
    output = trainer.loss_from_scores(scores, plan)
    # Exactly -log q(y | x, g) with zero penalty; the minibatch denominator is unchanged.
    assert output.loss.item() == pytest.approx(F.cross_entropy(scores, plan.target_base).item())
    assert output.consistency.item() == 0.0


@pytest.mark.parametrize("method", ["sigma", "transported_kl", "sft"])
def test_two_pass_gradient_equals_direct_gradient(tiny_scorer, toy_task, method):
    """The memory-bounded recomputation of Appendix C.5 preserves the full objective."""
    gradients = {}
    for mode in ("direct", "two_pass"):
        trainer = _trainer(tiny_scorer, toy_task, method, mode)
        if mode == "two_pass":
            tiny_scorer.view_batch_size, tiny_scorer.candidate_chunk_size = 3, 4
        plan = trainer.plan_step(1)
        tiny_scorer.model.zero_grad()
        output = trainer.backward(plan)
        gradients[mode] = (
            output.loss.item(),
            torch.cat([p.grad.flatten() for p in trainer.parameters if p.grad is not None]),
        )
    assert gradients["direct"][0] == pytest.approx(gradients["two_pass"][0], abs=1e-5)
    assert torch.allclose(gradients["direct"][1], gradients["two_pass"][1], atol=1e-5)
    assert gradients["direct"][1].abs().sum() > 0


def test_fit_runs_an_even_number_of_steps_and_counts_interventions(tiny_scorer, toy_task):
    calls = []

    def dev_accuracy() -> float:
        calls.append(len(calls))
        return [0.2, 0.6, 0.6][len(calls) - 1]

    trainer = _trainer(tiny_scorer, toy_task, max_steps=5)  # rounded up to 6
    trainer.dev_accuracy = dev_accuracy
    trainer._checkpoints = {2, 4, 6}
    result = trainer.fit()
    assert result.total_steps == 6 and len(result.history) == 6
    assert result.dev_curve == [(2, 0.2), (4, 0.6), (6, 0.6)]
    assert result.best_step == 4  # ties keep the earlier checkpoint
    counts = result.counts
    assert counts.attempted == 24 and counts.binding_attempted == 12  # half binding steps
    assert counts.changed == counts.binding_changed <= 12  # reversal never changes a target
    assert counts.identity_replacements == 0
    assert result.training_tokens > 0
    assert all(math.isfinite(h["loss"]) for h in result.history)
    assert result.history[0]["lr"] == pytest.approx(1e-3)  # one warm-up step out of six


def test_training_reduces_the_paired_loss(tiny_scorer, toy_task):
    trainer = _trainer(tiny_scorer, toy_task, max_steps=40, lr=3e-3)
    history = trainer.fit().history
    early = sum(h["loss"] for h in history[:6]) / 6
    late = sum(h["loss"] for h in history[-6:]) / 6
    assert late < early


def test_lora_trains_only_adapter_parameters_and_knockout_restores(toy_task):
    encoder = tiny_encoder([toy_task.instruction, *toy_task.candidates])
    base = build_tiny_backbone(len(encoder.tokenizer), seed=1)
    frozen = {n: p.detach().clone() for n, p in base.named_parameters()}
    for parameter in base.parameters():
        parameter.requires_grad_(False)
    model = apply_lora(base, LoraSettings(), seed=1)
    updates = lora_updates(model)
    assert len(updates) == 2 * 7  # seven projections in each of the two layers
    assert {name for _, name in updates} == set(LoraSettings().target_modules)
    hidden, inner = 96, 192
    per_layer = 16 * (4 * 2 * hidden + 3 * (hidden + inner))
    assert count_trainable_parameters(model) == 2 * per_layer
    state = trainable_state(model)
    assert all("lora_" in name for name in state)
    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.requires_grad:
                parameter.add_(0.05)
    for name, parameter in model.named_parameters():
        original = name.replace("base_model.model.", "", 1)
        if original in frozen:
            assert torch.equal(parameter, frozen[original])  # pretrained weights untouched
    scorer = CandidateScorer(model, encoder)
    prompt = [1, 5, 6, 2]
    candidates = scorer.encode_candidates(toy_task.candidates)
    with torch.no_grad():
        intact = scorer.score([prompt], candidates)
        with lora_knockout(model, layer=1, projection="down_proj"):
            knocked = scorer.score([prompt], candidates)
            assert torch.count_nonzero(lora_updates(model)[(1, "down_proj")]) == 0
        restored = scorer.score([prompt], candidates)
    assert not torch.allclose(intact, knocked) and torch.equal(intact, restored)
    load_trainable_state(model, state)
    assert all(torch.equal(v, state[k]) for k, v in trainable_state(model).items())


def test_tiny_backbone_is_deterministic_and_fully_trainable_on_request():
    first = build_tiny_backbone(50, seed=3)
    second = build_tiny_backbone(50, seed=3)
    other = build_tiny_backbone(50, seed=4)
    a, b, c = (torch.cat([p.flatten() for p in m.parameters()]) for m in (first, second, other))
    assert torch.equal(a, b) and not torch.equal(a, c)
    make_all_trainable(first)
    assert count_trainable_parameters(first) == a.numel()
    assert not first.training  # dropout stays disabled


def test_method_registry_covers_the_trainable_rows_of_table_1_and_2():
    assert set(METHODS) == {
        "frozen",
        "sft",
        "sft_token_matched",
        "paired_sft",
        "transported_kl",
        "sigma",
        "sigma_binding_only",
        "sigma_reversal_only",
        "sigma_f",
    }
    sigma, paired = METHODS["sigma"], METHODS["paired_sft"]
    assert (
        sigma.consistency == "centered_l2" and sigma.coefficient == 1.0 and sigma.tunes_coefficient
    )
    assert (
        paired.consistency == "none" and paired.coefficient == 0.0 and not paired.tunes_coefficient
    )
    assert METHODS["sft"].views == "single" and not METHODS["frozen"].trainable
    assert resolve_method("sigma").with_coefficient(0.1).coefficient == 0.1
    for name in EXTERNAL_BASELINES:
        with pytest.raises(ExternalBaselineError):
            resolve_method(name)
    with pytest.raises(KeyError):
        resolve_method("does-not-exist")
