"""Tables of the paper from finished runs (``results.json`` files written by the pipeline).

Every table is a seed mean (and standard deviation where the paper reports one) of
quantities stored under ``results["metrics"]``; multi-dataset tables use equal dataset
weights after computing each dataset's metric (Appendix C.6).
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np

from sigma.evaluation.metrics import aulc, paired_bootstrap_interval
from sigma.methods import METHODS

Run = dict


def collect_runs(root: str | Path) -> list[Run]:
    """Load every ``results.json`` below ``root``; each run remembers its directory."""
    runs = []
    for path in sorted(Path(root).rglob("results.json")):
        run = json.loads(path.read_text(encoding="utf-8"))
        run["directory"] = str(path.parent)
        runs.append(run)
    return runs


def lookup(run: Run, dotted: str) -> float:
    """Follow a ``a.b.c`` path into a run's metric dictionary."""
    value = run["metrics"]
    for key in dotted.split("."):
        value = value[key]
    return float("nan") if value is None else float(value)


def _group(runs: Sequence[Run], keys: Sequence[str]) -> dict[tuple, list[Run]]:
    groups: dict[tuple, list[Run]] = defaultdict(list)
    for run in runs:
        groups[tuple(run[k] for k in keys)].append(run)
    return groups


def seed_statistics(runs: Sequence[Run], dotted: str) -> tuple[float, float]:
    """Mean and sample standard deviation over the seeds of one setting."""
    values = np.array([lookup(r, dotted) for r in runs])
    return float(values.mean()), float(values.std(ddof=1)) if len(values) > 1 else 0.0


def method_order(names: Sequence[str]) -> list[str]:
    """Methods in the order of the paper's tables."""
    order = list(METHODS)
    return sorted(names, key=lambda n: order.index(n) if n in order else len(order))


def _markdown(header: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


def per_dataset_table(
    runs: Sequence[Run],
    columns: Sequence[tuple[str, str]],
    with_sd: bool = False,
    digits: int = 2,
    matched_base_for: Sequence[str] = (),
) -> str:
    """Rows = methods, column groups = datasets (the layout of Tables 1, 7, 10, 11, 12).

    For the methods in ``matched_base_for`` the Base accuracy is taken on the composition-
    matched support, as the paper does for SIGMA-F in Table 1.
    """
    by_setting = _group(runs, ["dataset", "method"])
    datasets = sorted({d for d, _ in by_setting})
    methods = method_order({m for _, m in by_setting})
    header = ["Method"] + [f"{d}: {title}" for d in datasets for title, _ in columns]
    rows = []
    for method in methods:
        row = [METHODS[method].paper_name if method in METHODS else method]
        for dataset in datasets:
            group = by_setting.get((dataset, method), [])
            for _, dotted in columns:
                if not group:
                    row.append("-")
                    continue
                if method in matched_base_for and dotted == "base.accuracy":
                    dotted = "composition.matched_base_accuracy"
                mean, sd = seed_statistics(group, dotted)
                row.append(
                    f"{mean:.{digits}f} ± {sd:.{digits}f}" if with_sd else f"{mean:.{digits}f}"
                )
        rows.append(row)
    return _markdown(header, rows)


def dataset_average_table(runs: Sequence[Run], columns: Sequence[tuple[str, str, int]]) -> str:
    """Equal-weight dataset means of seed means (the layout of Tables 2 and 4)."""
    by_setting = _group(runs, ["dataset", "method"])
    datasets = sorted({d for d, _ in by_setting})
    methods = method_order({m for _, m in by_setting})
    rows = []
    for method in methods:
        row = [METHODS[method].paper_name if method in METHODS else method]
        for _, dotted, digits in columns:
            values = [
                seed_statistics(by_setting[(d, method)], dotted)[0]
                for d in datasets
                if (d, method) in by_setting
            ]
            row.append(f"{np.mean(values):.{digits}f}" if values else "-")
        rows.append(row)
    return _markdown(["Method"] + [title for title, _, _ in columns], rows)


TABLE_1 = [("Base", "base.accuracy"), ("Comp.", "composition.accuracy")]
TABLE_2 = [
    ("Bind. Chg. Acc.", "binding.changed_accuracy", 2),
    ("Bind. PairAcc", "binding.pair_accuracy", 2),
    ("Comp. Chg. Acc.", "composition.changed_accuracy", 2),
    ("Comp. PairAcc", "composition.pair_accuracy", 2),
    ("EquivAcc", "composition.equiv_accuracy", 2),
    ("TV_eq", "composition.tv", 4),
]
TABLE_7 = [
    ("F1 Base", "base.macro_f1"),
    ("F1 Bind.", "binding.macro_f1"),
    ("F1 Rev.", "reversal.macro_f1"),
    ("F1 Comp.", "composition.macro_f1"),
    ("Hit@4", "composition.hit@4"),
    ("Hit@8", "composition.hit@8"),
    ("MRR", "composition.mrr"),
]
TABLE_11 = [
    ("Matched Base", "composition.matched_base_accuracy"),
    ("Comp.", "composition.accuracy"),
    ("Paired drop", "composition.paired_drop"),
]
TABLE_12 = [
    ("Chg. Acc.", "composition.changed.accuracy"),
    ("Chg. PairAcc", "composition.changed.pair_accuracy"),
    ("Unch. Acc.", "composition.unchanged.accuracy"),
    ("Unch. PairAcc", "composition.unchanged.pair_accuracy"),
    ("Rev. Acc.", "reversal.accuracy"),
    ("Rev. Inv.", "reversal.invariance"),
]


def table_4(runs: Sequence[Run]) -> str:
    """Average accuracy and the Base-to-Comp. drop (means computed before rounding)."""
    by_setting = _group(runs, ["dataset", "method"])
    datasets = sorted({d for d, _ in by_setting})
    rows = []
    for method in method_order({m for _, m in by_setting}):
        base = np.mean(
            [seed_statistics(by_setting[(d, method)], "base.accuracy")[0] for d in datasets]
        )
        comp = np.mean(
            [seed_statistics(by_setting[(d, method)], "composition.accuracy")[0] for d in datasets]
        )
        rows.append(
            [METHODS[method].paper_name, f"{base:.2f}", f"{comp:.2f}", f"{base - comp:.2f}"]
        )
    return _markdown(["Method", "Avg. Base", "Avg. Comp.", "Base-to-Comp. drop"], rows)


def label_efficiency(runs: Sequence[Run], dotted: str) -> dict[tuple[str, str], dict]:
    """Per (dataset, method): seed mean/sd for every budget and the AULC (Table 5)."""
    out: dict[tuple[str, str], dict] = {}
    for (dataset, method), group in _group(runs, ["dataset", "method"]).items():
        by_shots = _group(group, ["shots"])
        shots = sorted(s for (s,) in by_shots)
        stats = {s: seed_statistics(by_shots[(s,)], dotted) for s in shots}
        entry: dict = {
            "shots": shots,
            "mean": [stats[s][0] for s in shots],
            "sd": [stats[s][1] for s in shots],
        }
        seeds = sorted({r["seed"] for r in group})
        curves = []
        for seed in seeds:
            values = {r["shots"]: lookup(r, dotted) for r in group if r["seed"] == seed}
            if len(shots) >= 2 and all(s in values for s in shots):
                curves.append(aulc(shots, [values[s] for s in shots]))
        if curves:
            entry["aulc"] = (
                float(np.mean(curves)),
                float(np.std(curves, ddof=1)) if len(curves) > 1 else 0.0,
            )
        out[(dataset, method)] = entry
    return out


def table_5(runs: Sequence[Run]) -> str:
    """Label efficiency across nested budgets with the normalised log-budget AULC."""
    views = [("Base", "base.accuracy"), ("Comp.", "composition.accuracy")]
    tables = {title: label_efficiency(runs, dotted) for title, dotted in views}
    settings = sorted(tables["Base"], key=lambda k: (k[0], list(METHODS).index(k[1])))
    shots = sorted({s for entry in tables["Base"].values() for s in entry["shots"]})
    rows = []
    for dataset, method in settings:
        for title, _ in views:
            entry = tables[title][(dataset, method)]
            cells = dict(
                zip(entry["shots"], zip(entry["mean"], entry["sd"], strict=True), strict=True)
            )
            row = [dataset, METHODS[method].paper_name, title]
            row += [f"{cells[s][0]:.2f} ± {cells[s][1]:.2f}" if s in cells else "-" for s in shots]
            row.append("{:.2f} ± {:.2f}".format(*entry["aulc"]) if "aulc" in entry else "-")
            rows.append(row)
    return _markdown(["Dataset", "Method", "View", *[f"b = {s}" for s in shots], "AULC"], rows)


def _per_instance(run: Run) -> np.ndarray:
    with np.load(Path(run["directory"]) / "per_instance.npz") as data:
        return data["composition_accuracy"]


def paired_difference(
    runs_a: Sequence[Run], runs_b: Sequence[Run], num_resamples: int = 10_000, seed: int = 0
) -> tuple[float, float, float]:
    """Paired bootstrap of Comp. accuracy, method A minus method B (percentage points).

    Runs are matched by seed; instances without an accepted composition view in either run
    are dropped.  Returns ``(difference, lower, upper)`` of the pointwise 95% interval.
    """
    by_seed_b = {r["seed"]: r for r in runs_b}
    rows = [
        100.0 * (_per_instance(a) - _per_instance(by_seed_b[a["seed"]]))
        for a in sorted(runs_a, key=lambda r: r["seed"])
        if a["seed"] in by_seed_b
    ]
    differences = np.array(rows)
    differences = differences[:, ~np.isnan(differences).any(axis=0)]
    return paired_bootstrap_interval(differences, num_resamples, seed=seed)


def table_13(runs: Sequence[Run], reference: str = "sigma", num_resamples: int = 10_000) -> str:
    """Paired differences ``reference`` minus each comparator at equal budgets (Table 13)."""
    by_setting = _group(runs, ["dataset", "method", "shots"])
    rows = []
    for (dataset, method, shots), group in sorted(by_setting.items()):
        if method == reference or (dataset, reference, shots) not in by_setting:
            continue
        diff, low, high = paired_difference(
            by_setting[(dataset, reference, shots)], group, num_resamples
        )
        rows.append(
            [
                METHODS[method].paper_name,
                dataset,
                str(shots),
                f"{diff:+.2f} [{low:.2f}, {high:.2f}]",
            ]
        )
    return _markdown(["Comparator", "Dataset", "Shots", "Comp. difference [95% CI]"], rows)


def table_14(runs: Sequence[Run]) -> str:
    """Held-out composition acceptance and coverage, summed over seeds (Table 14)."""
    rows = []
    for (dataset,), group in sorted(_group(runs, ["dataset"]).items()):
        # The draws and the acceptance mask are shared across methods and budgets, so the
        # runs of one (method, budget) setting are enough; counts are summed over its seeds.
        method, shots = sorted({(r["method"], r["shots"]) for r in group})[0]
        selected = [r for r in group if (r["method"], r["shots"]) == (method, shots)]
        for condition in selected[0]["metrics"]["acceptance"]:
            cells = [r["metrics"]["acceptance"][condition] for r in selected]
            accepted, attempted = (
                sum(c["accepted"] for c in cells),
                sum(c["attempted"] for c in cells),
            )
            rows.append(
                [
                    dataset,
                    condition,
                    f"{100 * accepted / attempted:.2f} ({accepted:,} / {attempted:,})",
                    f"{np.mean([c['instance_coverage'] for c in cells]):.2f}",
                    f"{sum(c['changed'] for c in cells):,}",
                    f"{sum(c['unchanged'] for c in cells):,}",
                    f"{max(c['labels_changed'] for c in cells)} / {cells[0]['labels_supported']}",
                ]
            )
    return _markdown(
        ["Dataset", "Condition", "Accept (%)", "Inst. (%)", "Changed", "Unchanged", "Labels"], rows
    )


TABLES: dict[str, Callable[[Sequence[Run]], str]] = {
    "1": lambda runs: per_dataset_table(runs, TABLE_1, matched_base_for=("sigma_f",)),
    "2": lambda runs: dataset_average_table(runs, TABLE_2),
    "4": table_4,
    "5": table_5,
    "7": lambda runs: per_dataset_table(runs, TABLE_7, digits=3),
    "10": lambda runs: per_dataset_table(runs, TABLE_11[:2], with_sd=True),
    "11": lambda runs: per_dataset_table(runs, TABLE_11, with_sd=True),
    "12": lambda runs: per_dataset_table(runs, TABLE_12, with_sd=True),
    "13": table_13,
    "14": table_14,
}
