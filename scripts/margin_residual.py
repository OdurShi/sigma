"""Margin-residual score diagnostic (Section 5.4, Figure 6, Appendix D.7).

Usage::

    python scripts/margin_residual.py --runs RUN_DIR [RUN_DIR ...] --output margin_residual.json

Every run directory must contain ``config.yaml`` and ``adapter_state.pt`` (written by the
pipeline).  For each run, swap trajectories of depth 2, 4 and 8 are scored on a restricted
12-candidate inventory and reduced to ``E_local``, ``gamma_0`` and ``R_mr``.

Supplementary experiment E02 (all candidates, depths 8 and 16, every sampled path)::

    python scripts/margin_residual.py --runs RUN_DIR [...] --output e02.json \
        --depths 8 16 --num-candidates all --all-paths
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from sigma.analysis import margin_residual_study
from sigma.evaluation.diagnostics import path_residual_summary, summarize_trajectories
from sigma.pipeline import load_run
from sigma.predictors import DirectPredictor


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--num-candidates", default="12", help="inventory size, or 'all'")
    parser.add_argument("--all-paths", action="store_true", help="keep initially wrong paths")
    parser.add_argument("--depths", type=int, nargs="+", default=[2, 4, 8])
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    num_candidates = None if args.num_candidates == "all" else int(args.num_candidates)
    report = []
    for directory in args.runs:
        experiment = load_run(directory)
        diagnostics = margin_residual_study(
            lambda subtask, e=experiment: DirectPredictor(e.scorer, subtask),
            experiment.task,
            experiment.test,
            args.depths,
            num_candidates,
            experiment.config.seed,
            initially_correct_only=not args.all_paths,
        )
        report.append(
            {
                "run": directory,
                "method": experiment.method.name,
                "summary": summarize_trajectories(diagnostics),
                "by_depth": path_residual_summary(diagnostics),
                "trajectories": [
                    {
                        "local_error": d.local_error,
                        "initial_margin": d.initial_margin,
                        "ratio": d.ratio,
                        "final_correct": d.final_correct,
                        "depth": len(d.residuals),
                    }
                    for d in diagnostics
                ],
            }
        )
        print(directory, json.dumps(report[-1]["summary"]))
    Path(args.output).write_text(json.dumps(report), encoding="utf-8")


if __name__ == "__main__":
    main()
