"""Figures of the paper from finished runs and diagnostic outputs (needs the ``plots`` extra).

Usage::

    python scripts/make_figures.py figure5 --input runs/table5_label_efficiency --output fig5.pdf
    python scripts/make_figures.py figure4 --input runs/figure4_backbones --output fig4.pdf
    python scripts/make_figures.py figure6 --input margin_residual.json --output fig6.pdf
    python scripts/make_figures.py figure7 --input spectral_*.json --output fig7.pdf
    python scripts/make_figures.py figure8 --input knockout.json --output fig8.pdf
    python scripts/make_figures.py figure2 --input prelim.json --output fig2.pdf

The figures plot the reported quantities; the purely decorative elements of the paper's
figures (smoothed colour fields, density contours, display jitter) are not reproduced.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from sigma.evaluation.report import collect_runs, label_efficiency, seed_statistics  # noqa: E402
from sigma.methods import METHODS  # noqa: E402
from sigma.models import PAPER_TARGET_MODULES  # noqa: E402


def figure5(inputs: list[str], output: str) -> None:
    """Annotation-generalisation curves: Comp. accuracy against unique training labels."""
    runs = collect_runs(inputs[0])
    curves = label_efficiency(runs, "composition.accuracy")
    labels_per_shot = {
        r["dataset"]: r["num_training_labels"] / r["shots"] for r in runs if r["shots"]
    }
    datasets = sorted({d for d, _ in curves})
    figure, axes = plt.subplots(1, len(datasets), figsize=(4.2 * len(datasets), 3.4), squeeze=False)
    for axis, dataset in zip(axes[0], datasets, strict=True):
        points = []
        for (name, method), entry in sorted(curves.items()):
            if name != dataset:
                continue
            x = np.array(entry["shots"]) * labels_per_shot[dataset]
            mean, sd = np.array(entry["mean"]), np.array(entry["sd"])
            axis.plot(x, mean, marker="o", label=METHODS[method].paper_name)
            axis.fill_between(x, mean - sd, mean + sd, alpha=0.15)
            points += list(zip(x, mean, strict=True))
        frontier, best = [], -np.inf
        for x, y in sorted(points):  # empirical Pareto frontier over methods and budgets
            if y > best:
                frontier.append((x, y))
                best = y
        axis.plot(*zip(*frontier, strict=True), color="black", linewidth=2.5, alpha=0.5, zorder=0)
        axis.set_xscale("log")
        ticks = sorted({int(x) for x, _ in points})
        axis.set_xticks(ticks)
        axis.set_xticklabels([f"{t:,}" for t in ticks], fontsize=7)
        axis.minorticks_off()
        axis.set_title(dataset)
        axis.set_xlabel("Unique training labels, N = b m")
    axes[0][0].set_ylabel("Comp. accuracy (%)")
    axes[0][0].legend(fontsize=7)
    figure.tight_layout()
    figure.savefig(output)


def figure4(inputs: list[str], output: str) -> None:
    """Backbone comparison: Base (open squares) and Comp. (filled circles) per method."""
    runs = collect_runs(inputs[0])
    groups: dict[tuple, list] = defaultdict(list)
    for run in runs:
        groups[(run["dataset"], run["backbone"], run["shots"], run["method"])].append(run)
    datasets = sorted({k[0] for k in groups})
    settings = sorted({(k[1], k[2]) for k in groups})
    methods = [m for m in METHODS if any(k[3] == m for k in groups)]
    figure, axes = plt.subplots(1, len(datasets), figsize=(4.6 * len(datasets), 3.4), squeeze=False)
    for axis, dataset in zip(axes[0], datasets, strict=True):
        for j, method in enumerate(methods):
            for i, (backbone, shots) in enumerate(settings):
                group = groups.get((dataset, backbone, shots, method))
                if not group:
                    continue
                base = seed_statistics(group, "base.accuracy")[0]
                comp = seed_statistics(group, "composition.accuracy")[0]
                x = i + (j - (len(methods) - 1) / 2) * 0.2
                axis.plot([x, x], [comp, base], color=f"C{j}")
                axis.plot(x, base, marker="s", markerfacecolor="white", color=f"C{j}")
                axis.plot(x, comp, marker="o", color=f"C{j}", label=METHODS[method].paper_name)
        axis.set_xticks(range(len(settings)))
        axis.set_xticklabels([f"{b.split('/')[-1]}\n{s} shots" for b, s in settings], fontsize=6)
        axis.set_title(dataset)
    handles, labels = axes[0][0].get_legend_handles_labels()
    unique = dict(zip(labels, handles, strict=True))
    axes[0][0].legend(unique.values(), unique.keys(), fontsize=7)
    axes[0][0].set_ylabel("Accuracy (%)")
    figure.tight_layout()
    figure.savefig(output)


def figure6(inputs: list[str], output: str) -> None:
    """Margin-residual diagnostic: initial margin against cumulative local residual."""
    report = json.loads(Path(inputs[0]).read_text(encoding="utf-8"))
    figure, axes = plt.subplots(1, len(report), figsize=(3.8 * len(report), 3.4), squeeze=False)
    for axis, entry in zip(axes[0], report, strict=True):
        x = np.array([t["local_error"] for t in entry["trajectories"]])
        y = np.array([t["initial_margin"] for t in entry["trajectories"]])
        correct = np.array([t["final_correct"] for t in entry["trajectories"]], dtype=bool)
        axis.scatter(x[correct], y[correct], s=6, label="correct")
        axis.scatter(x[~correct], y[~correct], s=6, label="incorrect")
        limit = float(x.max()) if len(x) else 1.0
        axis.plot([0, limit], [0, np.sqrt(2) * limit], "k--", label="R_mr = 1")
        summary = entry["summary"]
        axis.set_title(
            f"{entry['method']}: N={summary['count']}, acc {summary['accuracy']:.1f}%, "
            f"certified {summary['certified_coverage']:.1f}%",
            fontsize=8,
        )
        axis.set_xlabel("Cumulative local residual E_local")
    axes[0][0].set_ylabel("Initial margin")
    axes[0][0].legend(fontsize=7)
    figure.tight_layout()
    figure.savefig(output)


def figure7(inputs: list[str], output: str) -> None:
    """Directional residual contraction against the intervention eigenvalue."""
    kappa, ratio = [], []
    for path in inputs:
        result = json.loads(Path(path).read_text(encoding="utf-8"))
        kappa += result["eigenvalues"]
        ratio += result["log_ratio"]
    kappa, ratio = np.array(kappa), np.array(ratio)
    figure, axis = plt.subplots(figsize=(6, 3.6))
    axis.scatter(kappa, ratio, s=6, alpha=0.5)
    exposed = kappa > 0
    if exposed.any():
        edges = np.linspace(kappa[exposed].min(), kappa[exposed].max(), 11)
        bins = np.clip(np.digitize(kappa[exposed], edges) - 1, 0, 9)
        centres = [(edges[b] + edges[b + 1]) / 2 for b in range(10) if np.any(bins == b)]
        medians = [np.median(ratio[exposed][bins == b]) for b in range(10) if np.any(bins == b)]
        axis.plot(centres, medians, color="black", label="median trend")
        axis.legend(fontsize=7)
    axis.axhline(0.0, linestyle="--", color="gray")
    axis.set_xlabel("Intervention eigenvalue")
    axis.set_ylabel("log2 residual-energy ratio")
    figure.tight_layout()
    figure.savefig(output)


def figure8(inputs: list[str], output: str) -> None:
    """LoRA knockout map: differential accuracy loss per layer and projection."""
    cells: dict[tuple[int, str], list[float]] = defaultdict(list)
    for path in inputs:
        for record in json.loads(Path(path).read_text(encoding="utf-8")):
            cells[(record["layer"], record["projection"])].append(record["guide_effect"])
    layers = sorted({layer for layer, _ in cells})
    projections = [p for p in PAPER_TARGET_MODULES if any(k[1] == p for k in cells)]
    grid = np.array([[np.mean(cells[(layer, p)]) for layer in layers] for p in projections])
    figure, axis = plt.subplots(figsize=(0.35 * len(layers) + 2.5, 3.2))
    image = axis.imshow(grid, aspect="auto", cmap="viridis")
    axis.set_yticks(range(len(projections)))
    axis.set_yticklabels(projections)
    axis.set_xticks(range(len(layers)))
    axis.set_xticklabels(layers, fontsize=6)
    axis.set_xlabel("Transformer layer (zero-based)")
    figure.colorbar(image, label="I_guide (pp)")
    figure.tight_layout()
    figure.savefig(output)


def figure2(inputs: list[str], output: str) -> None:
    """Preliminary study: rank transitions and response error by uncertainty and budget."""
    report = json.loads(Path(inputs[0]).read_text(encoding="utf-8"))
    figure, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    for axis, name in zip(axes[:2], ("binding", "order"), strict=True):
        axis.imshow(np.array(report["retention"][name]["matrix"]), vmin=0, vmax=1, cmap="magma")
        axis.set_title(
            f"{name} revision: exact retention {report['retention'][name]['exact']:.1f}%"
        )
        axis.set_xlabel("Rank after edit (last column: beyond top)")
        axis.set_ylabel("Original candidate rank")
    budgets = report["budgets"]
    rates = np.array([report["response_error"][str(b)]["bin_rates"] for b in budgets], dtype=float)
    image = axes[2].imshow(rates, aspect="auto", vmin=0, vmax=1, origin="lower")
    axes[2].set_yticks(range(len(budgets)))
    axes[2].set_yticklabels(budgets)
    axes[2].set_xlabel("Reference-uncertainty bin")
    axes[2].set_ylabel("Labeled examples per intent")
    figure.colorbar(image, ax=axes[2], label="Response-error rate")
    figure.tight_layout()
    figure.savefig(output)


FIGURES = {
    "figure2": figure2,
    "figure4": figure4,
    "figure5": figure5,
    "figure6": figure6,
    "figure7": figure7,
    "figure8": figure8,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("figure", choices=sorted(FIGURES))
    parser.add_argument("--input", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    FIGURES[args.figure](args.input, args.output)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
