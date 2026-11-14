"""Run the grid of one paper experiment (one YAML per table/figure in configs/experiments).

Usage::

    python scripts/run_experiment.py --experiment configs/experiments/table1_main.yaml --dry-run
    python scripts/run_experiment.py --experiment configs/experiments/table1_main.yaml \
        --only dataset=banking77 method=sigma seed=13 --skip-existing

``--set key=value`` applies configuration overrides to every run of the grid.
"""

from sigma.cli import main

if __name__ == "__main__":
    import sys

    main(["experiment", *sys.argv[1:]])
