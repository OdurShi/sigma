#!/usr/bin/env python
"""Draw the result figures of the README from the reported tables.

    python scripts/plot_results.py                 # all figures -> assets/results/
    python scripts/plot_results.py --only main e05 # a subset
    python scripts/plot_results.py --list

Every figure is drawn from the CSV files in ``results/paper`` and ``results/supplementary``
through ``sigma.supplementary``; no number is written in this script, so the figures follow
the tables when the numbers are updated.  Requires matplotlib (``pip install -e ".[plots]"``).
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path

from sigma.supplementary import DATASETS, SHOTS, Index, load_index, load_paper_tables

REPOSITORY = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPOSITORY / "assets" / "results"

# Chart surface and ink.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
# One colour per method in every figure; controls that are context stay grey.
BLUE, ORANGE, AQUA, GREY, GREY_LIGHT = "#2a78d6", "#eb6834", "#1baf7a", "#898781", "#c3c2b7"
METHOD_COLOURS = {
    "SIGMA": BLUE,
    "SIGMA-F": ORANGE,
    "Transported KL": AQUA,
    "Paired SFT": GREY,
    "LoRA-SFT": GREY_LIGHT,
}


def _style(plt) -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
            "font.size": 8.5,
            "text.color": INK,
            "axes.labelcolor": INK_SECONDARY,
            "axes.edgecolor": AXIS,
            "axes.linewidth": 0.8,
            "axes.titlesize": 9,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "xtick.color": INK_SECONDARY,
            "ytick.color": INK,
            "legend.frameon": False,
            "legend.fontsize": 8,
        }
    )


def _frame(ax, value_axis: str = "x") -> None:
    """Recessive chrome: hairline grid on the value axis, no box, no tick marks."""
    for side in ("top", "right", "left" if value_axis == "x" else "bottom"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=0)
    ax.grid(axis=value_axis, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def _dot(ax, x, y, colour, size: float = 7.0, **kwargs) -> None:
    """A filled marker with a ring in the surface colour."""
    ax.plot(x, y, "o", ms=size, mfc=colour, mec=SURFACE, mew=1.2, zorder=3, **kwargs)


def _titles(fig, title: str, subtitle: str) -> None:
    """Title and subtitle in a header band of fixed height, then lay out the panels below."""
    height = fig.get_figheight()
    fig.suptitle(
        title, x=0.01, y=1 - 0.07 / height, ha="left", va="top", fontsize=10.5, fontweight="bold"
    )
    fig.text(
        0.01, 1 - 0.33 / height, subtitle, ha="left", va="top", fontsize=8.5, color=INK_SECONDARY
    )
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.58 / height))


def _legend(fig, plt, entries, ncols: int | None = None) -> None:
    handles = [
        plt.Line2D([], [], color=colour, marker="o", mec=SURFACE, ls="", ms=7, label=label)
        for label, colour in entries
    ]
    fig.legend(
        handles=handles, loc="upper right", ncols=ncols or len(handles), bbox_to_anchor=(0.995, 1.0)
    )


def plot_main(index: Index, paper: Index, plt):
    """Table 1: Base and Comp. accuracy of every method on the three datasets."""
    table = paper["T01"]
    fig, axes = plt.subplots(1, len(DATASETS), figsize=(9.6, 3.9), sharey=True)
    methods = [row["method"] for row in table.rows]
    for ax, dataset in zip(axes, DATASETS, strict=True):
        for position, row in enumerate(table.rows):
            base = table.number(row, f"{dataset.lower()}_base")
            comp = table.number(row, f"{dataset.lower()}_comp")
            colour = METHOD_COLOURS.get(row["method"], GREY)
            emphasised = row["method"] in ("SIGMA", "SIGMA-F")
            colour = colour if emphasised else GREY
            ax.hlines(position, comp, base, color=colour, linewidth=2, alpha=0.45, zorder=2)
            ax.plot(base, position, "s", ms=6, mfc=SURFACE, mec=colour, mew=1.4, zorder=3)
            _dot(ax, comp, position, colour)
            if emphasised:
                ax.annotate(
                    f"{comp:.2f}",
                    (comp, position),
                    xytext=(-8, 0),
                    textcoords="offset points",
                    ha="right",
                    va="center",
                    fontsize=8,
                )
        ax.set_yticks(range(len(methods)), methods)
        ax.set_title(dataset)
        ax.set_xlabel("Accuracy (%)")
        _frame(ax)
    axes[0].invert_yaxis()
    handles = [
        plt.Line2D([], [], color=GREY, marker="o", mec=SURFACE, ls="", ms=7, label="Comp."),
        plt.Line2D([], [], color=GREY, marker="s", mfc=SURFACE, ls="", ms=6, label="Base"),
    ]
    fig.legend(handles=handles, loc="upper right", ncols=2, bbox_to_anchor=(0.995, 1.0))
    _titles(
        fig,
        "Accuracy under the original guidance (Base) and under unseen compositions (Comp.)",
        "Table 1 of the paper: Qwen2.5-7B-Instruct, 10 shots per intent, five-seed means; "
        "the segment is the Base-to-Comp. drop",
    )
    return fig


def plot_e05(index: Index, paper: Index, plt):
    """Table 5 and E05: Comp. accuracy against the label budget, including SIGMA-F."""
    sweep, table5 = index["E05"], paper["T05"]
    series = [
        ("LoRA-SFT", table5, {"view": "Comp."}),
        ("Transported KL", table5, {"view": "Comp."}),
        ("SIGMA", sweep, {}),
        ("SIGMA-F", sweep, {}),
    ]
    fig, axes = plt.subplots(1, len(DATASETS), figsize=(9.6, 3.5))
    for ax, dataset in zip(axes, DATASETS, strict=True):
        for method, table, where in series:
            row = table.row(dataset=dataset, method=method, **where)
            means = [table.number(row, f"shots_{s}_mean") for s in SHOTS]
            sds = [table.number(row, f"shots_{s}_sd") for s in SHOTS]
            colour = METHOD_COLOURS[method]
            ax.fill_between(
                range(len(SHOTS)),
                [m - s for m, s in zip(means, sds, strict=True)],
                [m + s for m, s in zip(means, sds, strict=True)],
                color=colour,
                alpha=0.10,
                linewidth=0,
            )
            ax.plot(range(len(SHOTS)), means, color=colour, linewidth=2, solid_capstyle="round")
            _dot(ax, range(len(SHOTS)), means, colour, size=6)
        ax.set_xticks(range(len(SHOTS)), [str(s) for s in SHOTS])
        ax.set_xlabel("Labeled examples per intent")
        ax.set_title(dataset)
        _frame(ax, value_axis="y")
    axes[0].set_ylabel("Comp. accuracy (%)")
    _legend(fig, plt, [(method, METHOD_COLOURS[method]) for method, _, _ in series])
    _titles(
        fig,
        "Label efficiency on unseen guidance compositions (Table 5, E05)",
        "Mean ± SD over five seeds; nested training subsets of 1 to 16 examples per intent",
    )
    return fig


def plot_e01(index: Index, paper: Index, plt):
    """E01: changed-target accuracy of frozen models under three broader guidance edits."""
    table = index["E01"]
    methods = (
        ("Paired SFT", "paired_sft"),
        ("Transported KL", "transported_kl"),
        ("SIGMA", "sigma"),
        ("SIGMA-F", "sigma_f"),
    )
    conditions = list(dict.fromkeys(row["condition"] for row in table.rows))
    labels = [c.replace(" + compositions", "").replace(" compositions", "") for c in conditions]
    fig, axes = plt.subplots(1, len(DATASETS), figsize=(9.6, 2.7), sharey=True)
    for ax, dataset in zip(axes, DATASETS, strict=True):
        for position, condition in enumerate(conditions):
            row = table.row(dataset=dataset, condition=condition)
            values = [table.number(row, f"{prefix}_changed") for _, prefix in methods]
            ax.hlines(position, min(values), max(values), color=AXIS, linewidth=2, zorder=1)
            for (method, _), value in zip(methods, values, strict=True):
                _dot(ax, value, position, METHOD_COLOURS[method])
        ax.set_yticks(range(len(conditions)), labels)
        ax.set_ylim(len(conditions) - 0.5, -0.5)
        ax.set_title(dataset)
        ax.set_xlabel("Changed-target accuracy (%)")
        _frame(ax)
    _legend(fig, plt, [(method, METHOD_COLOURS[method]) for method, _ in methods])
    _titles(
        fig,
        "Frozen transfer to broader guidance edits (E01)",
        "Accuracy on test instances whose required output changes; each edit is combined with "
        "held-out compositions",
    )
    return fig


def plot_e02(index: Index, paper: Index, plt):
    """E02: median local residual along unseen binding paths of depth 8 and 16."""
    table = index["E02"]
    methods = (
        ("Paired SFT", "paired_sft"),
        ("Transported KL", "transported_kl"),
        ("SIGMA", "sigma"),
    )
    depths = list(dict.fromkeys(row["depth"] for row in table.rows))
    fig, axes = plt.subplots(1, len(DATASETS), figsize=(9.6, 2.9), sharey=True)
    for ax, dataset in zip(axes, DATASETS, strict=True):
        for method, prefix in methods:
            rows = [table.row(dataset=dataset, depth=depth) for depth in depths]
            residuals = [table.number(row, f"{prefix}_median_residual") for row in rows]
            colour = METHOD_COLOURS[method]
            ax.plot(range(len(depths)), residuals, color=colour, linewidth=2)
            _dot(ax, range(len(depths)), residuals, colour)
            ax.annotate(
                f"{rows[-1][f'{prefix}_comp']}%",
                (len(depths) - 1, residuals[-1]),
                xytext=(9, 0),
                textcoords="offset points",
                va="center",
                fontsize=8,
            )
        ax.set_xticks(range(len(depths)), [f"depth {depth}" for depth in depths])
        ax.set_xlim(-0.3, len(depths) - 0.45)
        ax.set_title(dataset)
        _frame(ax, value_axis="y")
    axes[0].set_ylabel("Median local residual")
    _legend(fig, plt, [(method, METHOD_COLOURS[method]) for method, _ in methods])
    _titles(
        fig,
        "Local residuals along unseen binding paths (E02)",
        "All candidates, frozen 10-shot models; lower is better; labels give the all-path "
        "accuracy at depth 16",
    )
    return fig


def plot_cost(index: Index, paper: Index, plt):
    """E03 and E06: Comp. accuracy against training GPU-hours for three operating points."""
    e03, e06 = index["E03"], index["E06"]
    points = (
        ("LoRA-SFT, 16 shots", e03, "sft_16", METHOD_COLOURS["LoRA-SFT"]),
        ("SIGMA, 4 shots", e03, "sigma_4", BLUE),
        ("SIGMA-F, 4 shots", e06, "sigma_f_4", ORANGE),
    )
    fig, axes = plt.subplots(1, len(DATASETS), figsize=(9.6, 2.5), sharey=True)
    for ax, dataset in zip(axes, DATASETS, strict=True):
        hours = []
        for position, (_, table, prefix, colour) in enumerate(points):
            row = table.row(dataset=dataset)
            x = table.number(row, f"{prefix}_gpu_hours_approx")
            hours.append(x)
            ax.barh(position, x, height=0.42, color=colour, zorder=2)
            ax.annotate(
                f"{row[f'{prefix}_gpu_hours_approx']} GPU-h, Comp. {row[f'{prefix}_comp']}%",
                (x, position),
                xytext=(6, 0),
                textcoords="offset points",
                va="center",
                fontsize=8,
            )
        ax.set_xlim(0, max(hours) * 1.9)
        ax.set_yticks(range(len(points)), [label for label, _, _, _ in points])
        ax.set_ylim(len(points) - 0.5, -0.5)
        ax.set_title(dataset)
        ax.set_xlabel("Training GPU-hours")
        _frame(ax)
    _titles(
        fig,
        "Fewer labels and less training time (E03, E06)",
        "SIGMA with 4 examples per intent against LoRA-SFT with 16; GPU-hours cover training only",
    )
    return fig


def plot_e09(index: Index, paper: Index, plt):
    """E09: Comp. gains of SIGMA over paired SFT and transported KL with 95% intervals."""
    table = index["E09"]
    comparators = (
        ("over paired SFT", "gain_vs_paired_sft", BLUE, -0.16),
        ("over transported KL", "gain_vs_transported_kl", "#86b6ef", 0.16),
    )
    fig, ax = plt.subplots(figsize=(6.6, 2.6))
    for position, row in enumerate(table.rows):
        for _, name, colour, offset in comparators:
            y = position + offset
            low, high = table.number(row, f"{name}_ci_low"), table.number(row, f"{name}_ci_high")
            ax.hlines(y, low, high, color=colour, linewidth=2, capstyle="round", zorder=2)
            _dot(ax, table.number(row, name), y, colour)
    ax.axvline(0, color=AXIS, linewidth=1, zorder=1)
    ax.set_yticks(range(len(table.rows)), [row["dataset"] for row in table.rows])
    ax.set_ylim(len(table.rows) - 0.5, -0.5)
    ax.set_xlabel("Gain of SIGMA in Comp. accuracy (percentage points)")
    _frame(ax)
    _legend(fig, plt, [(f"SIGMA {label}", colour) for label, _, colour, _ in comparators])
    _titles(
        fig,
        "Paired gains with 95% intervals (E09)",
        "10 shots; paired bootstrap over seeds and test instances",
    )
    return fig


FIGURES: dict[str, tuple[str, Callable]] = {
    "main": ("main_results", plot_main),
    "e05": ("label_efficiency", plot_e05),
    "e01": ("e01_broader_edits", plot_e01),
    "e02": ("e02_depth_residuals", plot_e02),
    "cost": ("e03_e06_labels_and_cost", plot_cost),
    "e09": ("e09_paired_gains", plot_e09),
}


def draw(
    output_dir: Path = OUTPUT_DIR,
    only: list[str] | None = None,
    dpi: int = 200,
    suffix: str = "png",
) -> list[Path]:
    """Draw the selected figures and return the written paths."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _style(plt)
    index, paper = load_index(), load_paper_tables()
    output_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for key in only or list(FIGURES):
        name, function = FIGURES[key.lower()]
        figure = function(index, paper, plt)
        path = output_dir / f"{name}.{suffix}"
        figure.savefig(path, dpi=dpi)
        plt.close(figure)
        written.append(path)
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--only", nargs="+", choices=sorted(FIGURES), help="figures to draw")
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--format", default="png", choices=("png", "pdf", "svg"))
    parser.add_argument("--list", action="store_true", help="list the figures and exit")
    args = parser.parse_args(argv)
    if args.list:
        for key, (name, function) in FIGURES.items():
            print(f"{key}  {name}  {function.__doc__.splitlines()[0]}")
        return 0
    for path in draw(args.output_dir, args.only, args.dpi, args.format):
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
