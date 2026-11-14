"""End-to-end runs of the pipeline on the synthetic task with the tiny random model."""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from sigma import analysis
from sigma.config import load_config
from sigma.demo import QUICKSTART_OVERRIDES, format_results, quickstart_config
from sigma.evaluation import report
from sigma.evaluation.diagnostics import summarize_trajectories
from sigma.evaluation.protocol import EvalPlan
from sigma.pipeline import Trial, load_records, load_run, run, select_trial
from sigma.predictors import DirectPredictor, FactorizedPredictor
from sigma.training.trainer import InterventionCounts, TrainResult

TINY = [
    "model.name=tiny-random",
    "model.gradient_checkpointing=false",
    "model.tiny_hidden_size=48",
    "data.name=synthetic",
    "data.shots=4",
    "data.synthetic_groups=2",
    "data.synthetic_classes_per_group=2",
    "optim.batch_size=4",
    "optim.lr=2e-3",
    "scoring.view_batch_size=32",
    "scoring.max_context_tokens=null",
    "eval.binding_draws=2",
    "eval.composition_depths=[2,4]",
    "eval.composition_draws=2",
    "eval.save_scores=true",
    "data.max_test_instances=12",
    "data.max_dev_instances=8",
]


def _config(tmp_path, method, *extra):
    return load_config(None, [*TINY, f"method={method}", f"output_dir={tmp_path}/{method}", *extra])


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    """SIGMA and paired SFT with LoRA adapters, trained for a few steps on the same seed."""
    root = tmp_path_factory.mktemp("runs")
    results = {}
    for method in ("sigma", "paired_sft"):
        config = _config(root, method, "optim.max_steps=6", "scoring.gradient_mode=two_pass")
        results[method] = (config, run(config))
    return root, results


def test_sigma_runs_end_to_end_with_lora(runs):
    root, results = runs
    config, result = results["sigma"]
    selected = result["selection"]["selected"]
    assert selected["total_steps"] == 6 and len(selected["dev_curve"]) == 3
    assert selected["intervention_counts"]["attempted"] == 6 * 4
    assert selected["intervention_counts"]["binding_attempted"] == 3 * 4
    # Only LoRA parameters are trained: rank 16 on seven projections in two layers.
    hidden = 48
    assert result["trainable_parameters"] == 2 * 16 * (4 * 2 * hidden + 3 * 3 * hidden)
    metrics = result["metrics"]
    for block in ("base", "reversal", "binding", "composition", "acceptance"):
        assert block in metrics
    assert 0 <= metrics["composition"]["accuracy"] <= 100
    assert 0 <= metrics["composition"]["tv"] <= 1
    directory = root / "sigma"
    for name in (
        "config.yaml",
        "results.json",
        "train_log.json",
        "adapter_state.pt",
        "records.npz",
        "per_instance.npz",
    ):
        assert (directory / name).exists()
    history = json.loads((directory / "train_log.json").read_text())
    assert len(history) == 6 and all(np.isfinite(h["loss"]) for h in history)
    assert any(h["consistency"] > 0 for h in history)  # the alignment term is exercised


def test_paired_controls_share_views_with_sigma(runs):
    _, results = runs
    sigma, paired = results["sigma"][1], results["paired_sft"][1]
    assert sigma["num_training_labels"] == paired["num_training_labels"] == 4 * 4
    a = sigma["selection"]["selected"]["intervention_counts"]
    b = paired["selection"]["selected"]["intervention_counts"]
    assert a == b  # identical minibatches and edits; only the penalty differs
    assert sigma["metrics"]["acceptance"] == paired["metrics"]["acceptance"]


def test_saved_run_can_be_reloaded_and_reproduces_its_scores(runs):
    root, results = runs
    experiment = load_run(root / "sigma")
    records = load_records(root / "sigma")
    predictor = experiment.predictor()
    base = experiment.task.base_guidance()
    scores = predictor.score_views([(experiment.test[0].text, base)])
    assert np.allclose(scores[0], records.scores[0, 0], atol=1e-4)


def test_diagnostics_run_on_saved_runs(runs):
    root, results = runs
    experiment = load_run(root / "sigma")
    spectral = analysis.spectral_contraction(
        experiment.task, load_records(root / "sigma"), load_records(root / "paired_sft")
    )
    m = experiment.task.num_candidates
    assert spectral["log_ratio"].shape == (m - 1,)
    assert (spectral["eigenvalues"] > 0).sum() == 2  # two groups of two leaves
    trajectories = analysis.margin_residual_study(
        lambda subtask: DirectPredictor(experiment.scorer, subtask),
        experiment.task,
        experiment.test,
        depths=(2, 4),
        num_candidates=4,
    )
    summary = (
        summarize_trajectories(trajectories)
        if trajectories
        else {"bound_violations": 0, "certified_errors": 0}
    )
    assert summary["bound_violations"] == 0 and summary["certified_errors"] == 0
    knockouts = analysis.knockout_study(
        experiment.model,
        experiment.predictor,
        experiment.task,
        experiment.test[:4],
        EvalPlan(1, (2,), 1),
        seed=0,
    )
    assert len(knockouts) == 14
    revisions = analysis.revision_scores(experiment.predictor(), experiment.task, experiment.test)
    assert revisions.base.shape == revisions.binding.shape == revisions.order.shape
    assert revisions.response_error().shape == revisions.gold.shape
    # Every stored binding revision changes the required answer.
    changed = np.take_along_axis(revisions.permutations, revisions.gold[:, None], axis=1)[:, 0]
    assert np.all(changed != revisions.gold)


def test_tables_are_produced_from_run_directories(runs):
    root, _ = runs
    collected = report.collect_runs(root)
    assert {r["method"] for r in collected} == {"sigma", "paired_sft"}
    for table in ("1", "2", "4", "7", "11", "12", "14"):
        text = report.TABLES[table](collected)
        assert "SIGMA" in text or table == "14"
    paired = report.table_13(collected, num_resamples=200)
    assert "Paired SFT" in paired and "[" in paired


def test_single_view_training_learns_the_base_task(tmp_path):
    config = _config(
        tmp_path,
        "sft",
        "model.adaptation=full",
        "model.tiny_hidden_size=96",
        "data.shots=16",
        "data.max_test_instances=null",
        "data.max_dev_instances=null",
        "optim.batch_size=8",
        "optim.max_steps=200",
        "scoring.gradient_mode=direct",
    )
    result = run(config)
    history = json.loads((tmp_path / "sft" / "train_log.json").read_text())
    assert np.mean([h["loss"] for h in history[-10:]]) < 0.2 * np.mean(
        [h["loss"] for h in history[:10]]
    )
    assert result["metrics"]["base"]["accuracy"] >= 75.0  # chance is 25% with four classes
    assert result["selection"]["selected"]["intervention_counts"]["attempted"] == 0


def test_factorized_variant_is_exactly_equivariant(tmp_path):
    result = run(_config(tmp_path, "sigma_f", "optim.max_steps=4", "scoring.gradient_mode=direct"))
    metrics = result["metrics"]
    assert metrics["reversal"]["invariance"] == 100.0
    assert metrics["binding"]["equiv_accuracy"] == 100.0
    assert metrics["composition"]["equiv_accuracy"] == 100.0
    assert metrics["composition"]["tv"] == pytest.approx(0.0, abs=1e-12)
    assert metrics["composition"]["paired_drop"] == pytest.approx(0.0, abs=1e-9)
    experiment = load_run(tmp_path / "sigma_f")
    assert isinstance(experiment.predictor(), FactorizedPredictor)


def test_frozen_prompting_needs_no_training(tmp_path):
    result = run(_config(tmp_path, "frozen"))
    assert "selection" not in result and result["trainable_parameters"] == 0
    assert not (tmp_path / "frozen" / "adapter_state.pt").exists()


def test_token_matched_sft_extends_training_to_the_budget(tmp_path):
    reference = run(_config(tmp_path, "sft", "optim.max_steps=2", "scoring.gradient_mode=direct"))
    per_step = reference["selection"]["selected"]["training_tokens"] / 2
    config = _config(
        tmp_path,
        "sft_token_matched",
        f"optim.token_budget={int(5 * per_step)}",
        "scoring.gradient_mode=direct",
    )
    matched = run(config)["selection"]["selected"]
    assert matched["total_steps"] == 6  # ceil(5 / 2) * 2 steps reach the budget
    assert matched["training_tokens"] >= 5 * per_step * 0.9


def test_reduced_evaluation_subsets_are_fixed_and_spread_over_the_split():
    from sigma.data import Example
    from sigma.pipeline import fixed_subset

    examples = [Example(f"text {i}", f"label_{i // 10}") for i in range(100)]  # label-sorted
    subset = fixed_subset(examples, 20, "test")
    assert subset == fixed_subset(examples, 20, "test") and len(subset) == 20
    assert len({e.label for e in subset}) > 5  # not just the first classes of a sorted file
    assert [examples.index(e) for e in subset] == sorted(examples.index(e) for e in subset)
    assert fixed_subset(examples, None, "test") == examples
    assert fixed_subset(examples, 500, "test") == examples
    assert subset != fixed_subset(examples, 20, "dev")


def test_selection_rule_breaks_ties_as_in_appendix_c5():
    def trial(lr, coefficient, step, accuracy):
        result = TrainResult(10, step, accuracy, [], [], InterventionCounts(), 0)
        return Trial(lr, coefficient, result)

    best = select_trial(
        [
            trial(1e-4, 1.0, 4, 0.9),
            trial(5e-5, 0.1, 8, 0.9),
            trial(1e-5, 0.1, 6, 0.9),
            trial(1e-4, 0.1, 6, 0.9),
            trial(1e-5, 0.0, 2, 0.8),
        ]
    )
    # Highest accuracy, then smaller coefficient, earlier checkpoint, smaller learning rate.
    assert (best.lr, best.coefficient, best.result.best_step) == (1e-5, 0.1, 6)


def test_development_search_runs_the_grid_from_one_initialisation(tmp_path):
    config = _config(
        tmp_path,
        "sigma",
        "optim.max_steps=2",
        "scoring.gradient_mode=direct",
        "selection.search=true",
        "selection.lr_grid=[1e-3, 2e-3]",
        "selection.coefficient_grid=[0.0, 1.0]",
    )
    result = run(config)
    trials = result["selection"]["trials"]
    assert [(t["lr"], t["coefficient"]) for t in trials] == [
        (1e-3, 0.0),
        (1e-3, 1.0),
        (2e-3, 0.0),
        (2e-3, 1.0),
    ]
    assert result["selection"]["selected"] in trials


def test_quickstart_configuration_matches_the_yaml_file():
    from pathlib import Path

    yaml_path = Path(__file__).resolve().parents[1] / "configs" / "quickstart.yaml"
    from_file = load_config(yaml_path)
    from_code = quickstart_config("sigma", 13, "runs/quickstart")
    assert from_file == from_code
    assert "model.name=tiny-random" in QUICKSTART_OVERRIDES
    fake = {
        "sigma": {
            "metrics": {
                k: {
                    "accuracy": 1.0,
                    "invariance": 1.0,
                    "changed_accuracy": 1.0,
                    "pair_accuracy": 1.0,
                    "equiv_accuracy": 1.0,
                    "tv": 0.0,
                }
                for k in ("base", "reversal", "binding", "composition")
            }
        }
    }
    assert "SIGMA" in format_results(fake)


def test_scores_are_deterministic_across_processes_given_the_seed(tmp_path):
    first = run(
        _config(tmp_path / "a", "sigma", "optim.max_steps=2", "scoring.gradient_mode=direct")
    )
    second = run(
        _config(tmp_path / "b", "sigma", "optim.max_steps=2", "scoring.gradient_mode=direct")
    )
    assert first["metrics"] == second["metrics"]
    assert torch.equal(
        torch.load(tmp_path / "a" / "sigma" / "adapter_state.pt")[
            "base_model.model.model.layers.0.self_attn.q_proj.lora_B.default.weight"
        ],
        torch.load(tmp_path / "b" / "sigma" / "adapter_state.pt")[
            "base_model.model.model.layers.0.self_attn.q_proj.lora_B.default.weight"
        ],
    )
