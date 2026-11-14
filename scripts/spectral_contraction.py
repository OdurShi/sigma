"""Directional residual contraction (Appendix D.8, Figure 7).

Usage::

    python scripts/spectral_contraction.py --method RUN_DIR --reference RUN_DIR --output out.json

Both runs must share dataset, seed and evaluation protocol and must have been evaluated with
``eval.save_scores: true``.  The held-out single binding swaps of the test protocol provide
the label-independent input/edit pairs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from sigma.analysis import spectral_contraction
from sigma.config import load_config
from sigma.pipeline import load_records, load_task_and_splits


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--method", required=True, help="run directory of SIGMA")
    parser.add_argument("--reference", required=True, help="run directory of paired SFT")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    task, _ = load_task_and_splits(load_config(Path(args.method) / "config.yaml"))
    result = spectral_contraction(task, load_records(args.method), load_records(args.reference))
    exposed = result["eigenvalues"] > 0
    ratio = result["log_ratio"]
    print(f"directions: {int(exposed.sum())} exposed, {int((~exposed).sum())} null")
    print(
        f"median log2 energy ratio: exposed {np.median(ratio[exposed]):.3f}, "
        f"null {np.median(ratio[~exposed]):.3f}"
    )
    Path(args.output).write_text(
        json.dumps({key: value.tolist() for key, value in result.items()}), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
