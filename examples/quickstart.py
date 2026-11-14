"""Quick start as a script: SIGMA versus single-view fine-tuning on a synthetic task.

Equivalent to ``sigma demo``.  Everything (data, tokenizer, model) is built in code and the
whole script runs on a laptop CPU in a few minutes::

    python examples/quickstart.py

The printed table describes the synthetic task and the tiny randomly initialised model of
``configs/quickstart.yaml``.
"""

from __future__ import annotations

import logging

from sigma.demo import run_demo

if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    run_demo(methods=("sft", "sigma"), seed=13, output_dir="runs/quickstart")
