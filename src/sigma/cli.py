"""Command-line interface: ``sigma <command>``.

Commands
--------
``demo``          CPU quick start on a synthetic task with a tiny random model.
``train``         Train and evaluate one run from a YAML configuration.
``experiment``    Run (or list) the grid of a paper experiment.
``tables``        Print a paper table from finished runs.
``prepare-data``  Rebuild the files under ``data/`` (downloads the upstream release if needed).
``describe``      Print the hierarchy statistics of a prepared dataset.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path


def _demo(args: argparse.Namespace) -> None:
    from sigma.demo import run_demo

    run_demo(tuple(args.methods), args.seed, args.output_dir, args.steps)


def _overrides(args: argparse.Namespace) -> list[str]:
    """Configuration overrides of a command, including the ``--data-dir`` shortcut."""
    overrides = list(args.set)
    if args.data_dir:
        overrides.append(f"data.directory={args.data_dir}")
    return overrides


def _train(args: argparse.Namespace) -> None:
    from sigma.config import load_config
    from sigma.pipeline import run

    results = run(load_config(args.config, _overrides(args)))
    print(json.dumps(results["metrics"], indent=2))


def _experiment(args: argparse.Namespace) -> None:
    from sigma.experiments import expand_grid, token_budget_from
    from sigma.pipeline import run

    only = dict(item.split("=", 1) for item in args.only)
    runs = expand_grid(args.experiment, _overrides(args), only, args.output_root)
    for grid_run in runs:
        config = grid_run.config
        target = Path(config.output_dir)
        if args.dry_run:
            print(f"{grid_run.name:<60} -> {target}")
            continue
        if args.skip_existing and (target / "results.json").exists():
            continue
        if config.method == "sft_token_matched" and config.optim.token_budget is None:
            # Table 11: repeat original views up to the measured token budget of the SIGMA
            # run with the same dataset, budget and seed.
            sigma_dir = Path(str(target).replace("sft_token_matched", "sigma"))
            config.optim.token_budget = token_budget_from(sigma_dir)
        run(config)
    if args.dry_run:
        print(f"{len(runs)} runs")


def _tables(args: argparse.Namespace) -> None:
    from sigma.evaluation.report import TABLES, collect_runs

    runs = collect_runs(args.runs)
    if not runs:
        sys.exit(f"no results.json found below {args.runs}")
    print(TABLES[args.table](runs))


def _prepare(args: argparse.Namespace) -> None:
    from sigma.data.datasets import DATASETS, prepare_dataset, prepare_synthetic, write_checksums

    out_dir = Path(args.out_dir)
    raw_dir = Path(args.raw_dir) if args.raw_dir else out_dir / "raw"
    names = [*DATASETS, "synthetic"] if args.dataset == "all" else [args.dataset]
    for name in names:
        if name == "synthetic":
            print(f"prepared {prepare_synthetic(out_dir)}")
        else:
            print(f"prepared {prepare_dataset(name, raw_dir, out_dir)}")
    print(f"wrote {write_checksums(out_dir)}")


def _describe(args: argparse.Namespace) -> None:
    from sigma.data.datasets import load_prepared
    from sigma.evaluation.diagnostics import sibling_index_groups, spectral_basis
    from sigma.guidance.edits import binding_edit_support

    task, splits = load_prepared(Path(args.directory))
    groups = sibling_index_groups(task)
    basis = spectral_basis(task.num_candidates, groups)
    covered = sum(len(g) for g in groups)
    print(f"task: {task.name}")
    print(
        f"labels m = {task.num_candidates}; splits: "
        + ", ".join(f"{k}={len(v)}" for k, v in splits.items())
    )
    print(f"eligible sibling groups J = {len(groups)}; sizes = {sorted(len(g) for g in groups)}")
    print(f"labels outside eligible groups = {task.num_candidates - covered}")
    print(f"single binding swaps = {len(binding_edit_support(task))}")
    exposed = int(basis.exposed.sum())
    print(
        f"edit-exposed directions = {exposed}; null directions = {len(basis.eigenvalues) - exposed}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sigma", description=__doc__.split("\n")[0])
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--quiet", action="store_true", help="only log warnings")
    commands = parser.add_subparsers(dest="command", required=True)

    demo = commands.add_parser(
        "demo", parents=[common], help="CPU quick start (synthetic task, tiny random model)"
    )
    demo.add_argument("--methods", nargs="+", default=["sft", "sigma"])
    demo.add_argument("--seed", type=int, default=13)
    demo.add_argument("--steps", type=int, default=None, help="override the optimiser steps")
    demo.add_argument("--output-dir", default="runs/quickstart")
    demo.set_defaults(handler=_demo)

    train = commands.add_parser("train", parents=[common], help="train and evaluate one run")
    train.add_argument("--config", required=True)
    train.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    train.add_argument("--data-dir", default=None, help="directory holding the datasets")
    train.set_defaults(handler=_train)

    experiment = commands.add_parser(
        "experiment", parents=[common], help="run the grid of a paper experiment"
    )
    experiment.add_argument("--experiment", required=True)
    experiment.add_argument("--only", nargs="*", default=[], metavar="KEY=VALUE")
    experiment.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    experiment.add_argument("--data-dir", default=None, help="directory holding the datasets")
    experiment.add_argument("--output-root", default="runs")
    experiment.add_argument("--dry-run", action="store_true")
    experiment.add_argument("--skip-existing", action="store_true")
    experiment.set_defaults(handler=_experiment)

    tables = commands.add_parser(
        "tables", parents=[common], help="print a paper table from finished runs"
    )
    tables.add_argument("--runs", required=True)
    tables.add_argument(
        "--table", required=True, choices=["1", "2", "4", "5", "7", "10", "11", "12", "13", "14"]
    )
    tables.set_defaults(handler=_tables)

    prepare = commands.add_parser(
        "prepare-data", parents=[common], help="rebuild the shipped data files"
    )
    prepare.add_argument(
        "--dataset", required=True, choices=["banking77", "clinc150", "hwu64", "synthetic", "all"]
    )
    prepare.add_argument("--raw-dir", default=None, help="download cache (default: <out-dir>/raw)")
    prepare.add_argument("--out-dir", default="data")
    prepare.set_defaults(handler=_prepare)

    describe = commands.add_parser(
        "describe", parents=[common], help="hierarchy statistics of a prepared dataset"
    )
    describe.add_argument("directory")
    describe.set_defaults(handler=_describe)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(asctime)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    for noisy in ("httpx", "httpx2", "httpcore", "huggingface_hub", "urllib3", "filelock"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    args.handler(args)


if __name__ == "__main__":
    main()
