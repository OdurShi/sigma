"""Quick start: train SIGMA end to end on CPU with a tiny, randomly initialised model.

The demo builds a synthetic intent-routing task and a two-layer Qwen2-style language model in
code (nothing is downloaded), trains single-view fine-tuning and SIGMA on the same labeled
subset, and evaluates both under the test-guidance protocol.  Because the tiny model has no
pretrained knowledge, all of its weights are trained (``model.adaptation: full``); with a
pretrained backbone the same pipeline trains LoRA adapters only.

The printed table describes this synthetic task and tiny model (the configuration in
``configs/quickstart.yaml``).
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from sigma.config import RunConfig, load_config
from sigma.methods import METHODS
from sigma.pipeline import run

logger = logging.getLogger(__name__)

#: Overrides that turn the paper's default configuration into the CPU quick start.  The same
#: values are written out in ``configs/quickstart.yaml``.
QUICKSTART_OVERRIDES: tuple[str, ...] = (
    "model.name=tiny-random",
    "model.adaptation=full",
    "model.gradient_checkpointing=false",
    "data.name=synthetic",
    "data.shots=16",
    "data.synthetic_groups=2",
    "data.synthetic_classes_per_group=3",
    "optim.lr=1e-3",
    "scoring.gradient_mode=direct",
    "scoring.view_batch_size=64",
    "scoring.max_context_tokens=null",
    "eval.binding_draws=4",
    "eval.composition_draws=3",
)

#: Optimiser steps per method in the quick start (single-view training converges quickly).
QUICKSTART_STEPS: dict[str, int] = {"sft": 300, "paired_sft": 1600, "sigma": 1600}

COLUMNS: tuple[tuple[str, str], ...] = (
    ("Base", "base.accuracy"),
    ("Rev.", "reversal.accuracy"),
    ("Rev.Inv", "reversal.invariance"),
    ("Bind.Chg", "binding.changed_accuracy"),
    ("Comp.", "composition.accuracy"),
    ("Comp.Chg", "composition.changed_accuracy"),
    ("PairAcc", "composition.pair_accuracy"),
    ("EquivAcc", "composition.equiv_accuracy"),
    ("TV_eq", "composition.tv"),
)


def quickstart_config(
    method: str, seed: int, output_dir: str, steps: int | None = None
) -> RunConfig:
    """Run configuration of one quick-start method."""
    steps = steps if steps is not None else QUICKSTART_STEPS.get(method, 1600)
    overrides = [
        *QUICKSTART_OVERRIDES,
        f"method={method}",
        f"seed={seed}",
        f"optim.max_steps={steps}",
        f"output_dir={output_dir}/{method}",
    ]
    return load_config(None, overrides)


def _metric(results: dict, dotted: str) -> float:
    value = results["metrics"]
    for key in dotted.split("."):
        value = value[key]
    return float("nan") if value is None else float(value)


def format_results(results: dict[str, dict]) -> str:
    """Plain-text comparison table of the quick-start runs."""
    header = f"{'method':<24}" + "".join(f"{title:>10}" for title, _ in COLUMNS)
    lines = [header, "-" * len(header)]
    for method, result in results.items():
        cells = "".join(
            f"{_metric(result, dotted):>10.3f}"
            if dotted.endswith("tv")
            else f"{_metric(result, dotted):>10.1f}"
            for _, dotted in COLUMNS
        )
        lines.append(f"{METHODS[method].paper_name:<24}{cells}")
    return "\n".join(lines)


def run_demo(
    methods: tuple[str, ...] = ("sft", "sigma"),
    seed: int = 13,
    output_dir: str = "runs/quickstart",
    steps: int | None = None,
) -> dict[str, dict]:
    """Train and evaluate the requested methods on the synthetic task; print a summary."""
    results: dict[str, dict] = {}
    for method in methods:
        start = time.time()
        config = quickstart_config(method, seed, output_dir, steps)
        results[method] = run(config)
        selected = results[method]["selection"]["selected"]
        history = json.loads((Path(config.output_dir) / "train_log.json").read_text())
        first, last = history[0], history[-1]
        print(
            f"[{METHODS[method].paper_name}] {selected['total_steps']} steps in "
            f"{time.time() - start:.0f}s | loss {first['loss']:.3f} -> {last['loss']:.3f} | "
            f"consistency term {first['consistency']:.4f} -> {last['consistency']:.4f} | "
            f"selected step {selected['best_step']} (dev acc {selected['best_dev_accuracy']:.3f})"
        )
        counts = selected["intervention_counts"]
        if counts["attempted"]:
            print(
                f"    paired views: {counts['attempted']} pairs, "
                f"{counts['changed']} target-changing, {counts['identity_replacements']} identity "
                "replacements"
            )
    print()
    print(format_results(results))
    print(
        "\nAccuracies in percent, TV_eq in [0, 1]. Comp. = unseen compositions of 2/4/8 binding "
        "swaps with and without reversal.\nSetting: synthetic routing task, tiny randomly "
        "initialised model (configs/quickstart.yaml)."
    )
    return results
