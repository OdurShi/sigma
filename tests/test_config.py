"""Configuration files and experiment grids."""

from __future__ import annotations

from pathlib import Path

import pytest

from sigma.cli import build_parser
from sigma.config import ConfigError, RunConfig, load_config
from sigma.experiments import expand_grid
from sigma.methods import METHODS
from sigma.models import PAPER_BACKBONES, PAPER_TARGET_MODULES
from sigma.seeding import PAPER_SEEDS, PRELIMINARY_SEEDS, stream

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def test_base_configuration_holds_the_paper_hyperparameters():
    config = load_config(CONFIGS / "base.yaml")
    assert config.model.name == "Qwen/Qwen2.5-7B-Instruct"
    assert (config.model.lora.rank, config.model.lora.alpha, config.model.lora.dropout) == (
        16,
        32.0,
        0.0,
    )
    assert tuple(config.model.lora.target_modules) == PAPER_TARGET_MODULES
    optim = config.optim
    assert (optim.lr, optim.betas, optim.weight_decay, optim.grad_clip) == (
        5e-5,
        [0.9, 0.95],
        0.01,
        1.0,
    )
    assert (optim.warmup_ratio, optim.batch_size, optim.passes) == (0.03, 16, 10.0)
    assert config.selection.lr_grid == [1e-5, 5e-5, 1e-4]
    assert config.selection.coefficient_grid == [0.0, 0.01, 0.1, 1.0, 10.0]
    assert len(config.selection.lr_grid) * len(config.selection.coefficient_grid) == 15
    assert config.selection.num_checkpoints == 5
    assert config.scoring.max_context_tokens == 8192
    assert (
        config.eval.binding_draws,
        config.eval.composition_depths,
        config.eval.composition_draws,
    ) == (10, [2, 4, 8], 10)
    assert config.data.shots == 10 and METHODS[config.method].coefficient == 1.0


def test_python_defaults_agree_with_the_base_file():
    from_file = load_config(CONFIGS / "base.yaml")
    defaults = RunConfig()
    assert defaults.optim.lr == from_file.optim.lr and defaults.optim.betas == from_file.optim.betas
    assert defaults.model.lora == from_file.model.lora
    assert defaults.eval.composition_depths == from_file.eval.composition_depths


def test_overrides_and_unknown_keys(tmp_path):
    config = load_config(
        CONFIGS / "base.yaml", ["optim.lr=1e-4", "data.shots=4", "method=paired_sft"]
    )
    assert config.optim.lr == 1e-4 and config.data.shots == 4 and config.method == "paired_sft"
    bad = tmp_path / "bad.yaml"
    bad.write_text("optim:\n  learning_rate: 1\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="learning_rate"):
        load_config(bad)
    with pytest.raises(ConfigError):
        load_config(CONFIGS / "base.yaml", ["no-equals-sign"])


@pytest.mark.parametrize(
    ("name", "count"),
    [
        ("table1_main", 3 * 6 * 5),
        ("table2_ablations", 3 * 7 * 5),
        ("table5_label_efficiency", 3 * 5 * 5 * 5),
        ("supp_e01_broader_edits", 3 * 4 * 5),
        ("supp_e04_time_caps", 5 * 2 * 5),
        ("supp_e07_pbm_adapted", 3 * 3 * 5),
        ("supp_e08_global_binding", 3 * 2 * 5),
        ("figure4_backbones", 3 * 3 * 3 * 2 * 5),
        ("table10_opaque_outputs", 3 * 5 * 5),
        ("figure2_preliminary", 5 * 5),
    ],
)
def test_experiment_grids_expand_to_the_paper_settings(name, count):
    runs = expand_grid(CONFIGS / "experiments" / f"{name}.yaml")
    assert len(runs) == count
    assert len({r.config.output_dir for r in runs}) == count
    assert all(r.config.method in METHODS for r in runs)
    seeds = {r.config.seed for r in runs}
    assert seeds == set(PRELIMINARY_SEEDS if name == "figure2_preliminary" else PAPER_SEEDS)


def test_grid_details():
    backbones = expand_grid(CONFIGS / "experiments" / "figure4_backbones.yaml")
    assert {r.config.model.name for r in backbones} == set(PAPER_BACKBONES.values())
    qwen3 = next(r for r in backbones if r.config.model.name == "Qwen/Qwen3-8B")
    assert qwen3.config.model.template_kwargs == {"enable_thinking": False}
    assert {r.config.data.shots for r in backbones} == {5, 10}
    sweep = expand_grid(CONFIGS / "experiments" / "table5_label_efficiency.yaml")
    assert {r.config.data.shots for r in sweep} == {1, 2, 4, 8, 16}
    opaque = expand_grid(CONFIGS / "experiments" / "table10_opaque_outputs.yaml")
    assert {r.config.data.outputs for r in opaque} == {"opaque"}
    only = expand_grid(
        CONFIGS / "experiments" / "table1_main.yaml",
        overrides=["eval.binding_draws=2"],
        only={"dataset": "hwu64", "method": "sigma", "seed": "42"},
    )
    assert len(only) == 1 and only[0].config.eval.binding_draws == 2
    assert only[0].config.output_dir == "runs/table1_main/hwu64/sigma/seed42"


def test_named_streams_are_independent_and_reproducible():
    a = stream(13, "fewshot", "label").integers(1 << 30, size=4)
    assert (a == stream(13, "fewshot", "label").integers(1 << 30, size=4)).all()
    assert (a != stream(13, "fewshot", "other").integers(1 << 30, size=4)).any()
    assert (a != stream(21, "fewshot", "label").integers(1 << 30, size=4)).any()
    assert (stream(13, "x", 1).random(3) != stream(13, "x", 2).random(3)).any()


def test_cli_parses_every_command():
    parser = build_parser()
    assert parser.parse_args(["demo", "--methods", "sft", "sigma"]).methods == ["sft", "sigma"]
    assert parser.parse_args(["train", "--config", "c.yaml", "--set", "seed=21"]).set == ["seed=21"]
    args = parser.parse_args(
        ["experiment", "--experiment", "e.yaml", "--only", "method=sigma", "--dry-run"]
    )
    assert args.only == ["method=sigma"] and args.dry_run
    assert parser.parse_args(["tables", "--runs", "runs", "--table", "5"]).table == "5"
    assert parser.parse_args(["prepare-data", "--dataset", "all"]).dataset == "all"
    assert parser.parse_args(["prepare-data", "--dataset", "synthetic"]).out_dir == "data"
    with_data = parser.parse_args(["train", "--config", "c.yaml", "--data-dir", "/tmp/elsewhere"])
    assert with_data.data_dir == "/tmp/elsewhere"
