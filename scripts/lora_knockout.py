"""Single-module LoRA knockouts (Appendix D.9, Figure 8).

Usage::

    python scripts/lora_knockout.py --run RUN_DIR --output knockout.json

Removes one low-rank update at a time from the selected SIGMA checkpoint (no retraining) and
reports the differential accuracy loss ``I_guide`` and the equivariance increase ``I_eq``.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict
from pathlib import Path

from sigma.analysis import knockout_study
from sigma.evaluation.protocol import EvalPlan
from sigma.pipeline import load_run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--run", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    experiment = load_run(args.run, args.set)
    config = experiment.config
    plan = EvalPlan(
        config.eval.binding_draws,
        tuple(config.eval.composition_depths),
        config.eval.composition_draws,
    )
    records = knockout_study(
        experiment.model, experiment.predictor, experiment.task, experiment.test, plan, config.seed
    )
    Path(args.output).write_text(json.dumps([asdict(r) for r in records]), encoding="utf-8")
    worst = max(records, key=lambda r: r.guide_effect)
    print(f"largest I_guide: layer {worst.layer} {worst.projection} = {worst.guide_effect:.3f} pp")


if __name__ == "__main__":
    main()
