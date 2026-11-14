"""Preliminary study of single-view SFT (Section 3, Figure 2, Table 9, Appendix C.7).

Usage::

    python scripts/preliminary_study.py --runs runs/figure2_preliminary --output prelim.json

``--runs`` is the root of the ``figure2_preliminary`` grid: single-view SFT runs for every
budget and seed.  The reference uncertainty is computed with the unadapted backbone and
shared across budgets and runs.  Panel (a) conditions on correctness at ``--panel-budget``
(8 in the paper); panel (b) uses the cohort that is correct at every budget.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

import numpy as np

from sigma.analysis import revision_scores
from sigma.evaluation import diagnostics as dg
from sigma.evaluation.evaluate import score_unique
from sigma.evaluation.report import collect_runs
from sigma.pipeline import build_experiment, load_run
from sigma.predictors import DirectPredictor


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--runs", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--panel-budget", type=int, default=8)
    parser.add_argument("--num-bins", type=int, default=12)
    parser.add_argument("--top", type=int, default=10)
    args = parser.parse_args()
    runs = [r for r in collect_runs(args.runs) if r["method"] == "sft"]
    by_seed: dict[int, dict[int, dict]] = defaultdict(dict)
    for run in runs:
        by_seed[run["seed"]][run["shots"]] = run
    budgets = sorted({r["shots"] for r in runs})

    scores: dict[int, dict[int, object]] = defaultdict(dict)
    reference = None
    for seed, per_budget in sorted(by_seed.items()):
        for budget, run in sorted(per_budget.items()):
            experiment = load_run(run["directory"])
            if reference is None:  # reference distribution: the backbone before SFT, under g0
                frozen = build_experiment(replace(experiment.config, method="frozen"))
                predictor = DirectPredictor(frozen.scorer, frozen.task)
                base = frozen.task.base_guidance()
                reference = score_unique(predictor, [(e.text, base) for e in frozen.test])
            scores[seed][budget] = revision_scores(
                experiment.predictor(), experiment.task, experiment.test, seed=0
            )
    percentiles = dg.upper_percentiles(dg.normalized_entropy(reference))

    transitions = {"binding": [], "order": []}
    errors = {b: [] for b in budgets}
    bins: list[np.ndarray] = []
    for seed in sorted(scores):
        panel = scores[seed][args.panel_budget]
        cohort_a = panel.correct
        transitions["binding"].append(
            dg.aligned_rank_transitions(
                panel.base[cohort_a],
                panel.binding[cohort_a],
                panel.permutations[cohort_a],
                args.top,
            )
        )
        identity = np.broadcast_to(np.arange(panel.base.shape[1]), panel.base.shape)
        transitions["order"].append(
            dg.aligned_rank_transitions(
                panel.base[cohort_a], panel.order[cohort_a], identity[cohort_a], args.top
            )
        )
        fixed = np.all([scores[seed][b].correct for b in budgets], axis=0)
        bins.append(dg.uncertainty_bins(percentiles[panel.indices][fixed], args.num_bins))
        for budget in budgets:
            errors[budget].append(scores[seed][budget].response_error()[fixed])
    pooled_bins = np.concatenate(bins)
    report: dict = {"budgets": budgets, "retention": {}, "response_error": {}}
    for name, matrices in transitions.items():
        mean = np.mean(matrices, axis=0)  # equal weight per run, Eq. (C.6)
        exact, local = dg.rank_retention(mean)
        report["retention"][name] = {
            "exact": 100 * exact,
            "local": 100 * local,
            "top1": 100 * mean[0, 0],
            "matrix": mean.tolist(),
        }
        print(
            f"{name}: exact {100 * exact:.2f}%, within one rank {100 * local:.2f}%, "
            f"top-1 {100 * mean[0, 0]:.2f}%"
        )
    for budget in budgets:
        pooled = np.concatenate(errors[budget])
        counts, rates = dg.binned_error_rates(pooled, pooled_bins, args.num_bins)
        report["response_error"][budget] = {
            "overall": 100 * float(pooled.mean()),
            "low": 100 * float(np.mean(rates[:2])),
            "high": 100 * float(np.mean(rates[-2:])),
            "bin_counts": counts.tolist(),
            "bin_rates": rates.tolist(),
        }
        print(f"budget {budget}: overall {100 * pooled.mean():.2f}%")
    Path(args.output).write_text(json.dumps(report), encoding="utf-8")


if __name__ == "__main__":
    main()
