"""Experiment grids: one YAML file per paper table or figure (``configs/experiments``).

An experiment file has the keys of a run configuration (usually inherited from
``configs/base.yaml`` through ``defaults``) plus a ``grid`` mapping.  Grid keys are dotted
configuration keys (``data.shots``, ``seed``, ``method`` ...) or one of two include keys:
``dataset`` merges ``configs/datasets/<value>.yaml`` and ``backbone`` merges
``configs/backbones/<value>.yaml``.  The Cartesian product of the grid defines the runs.
"""

from __future__ import annotations

import itertools
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sigma.config import RunConfig, _build, _merge, _read, apply_overrides

INCLUDE_KEYS = {"dataset": "datasets", "backbone": "backbones"}


@dataclass(frozen=True)
class GridRun:
    """One cell of an experiment grid."""

    name: str  # path-like run name, unique within the experiment
    settings: dict[str, Any]  # the grid values of this cell
    config: RunConfig


def _run_name(settings: dict[str, Any]) -> str:
    parts = []
    for key, value in settings.items():
        short = key.split(".")[-1]
        text = str(value).split("/")[-1]
        parts.append(text if short in ("dataset", "method", "backbone") else f"{short}{text}")
    return "/".join(parts)


def expand_grid(
    path: str | Path,
    overrides: list[str] | None = None,
    only: dict[str, str] | None = None,
    output_root: str | Path = "runs",
) -> list[GridRun]:
    """Expand an experiment file into its runs.

    ``overrides`` are applied to every run (e.g. a smaller evaluation for a smoke test) and
    ``only`` keeps the grid cells whose values match (compared as strings).
    """
    path = Path(path)
    payload = _read(path)
    payload.pop("description", None)
    grid: dict[str, list[Any]] = payload.pop("grid", {})
    config_root = path.resolve().parent.parent
    experiment = path.stem
    runs: list[GridRun] = []
    for values in itertools.product(*grid.values()):
        settings = dict(zip(grid.keys(), values, strict=True))
        if only and any(str(settings.get(k)) != v for k, v in only.items()):
            continue
        merged = dict(payload)
        dotted: list[str] = []
        for key, value in settings.items():
            if key in INCLUDE_KEYS:
                merged = _merge(merged, _read(config_root / INCLUDE_KEYS[key] / f"{value}.yaml"))
            else:
                dotted.append(f"{key}={json.dumps(value)}")
        merged.pop("description", None)
        name = _run_name(settings)
        merged = apply_overrides(merged, dotted + (overrides or []))
        merged["output_dir"] = str(Path(output_root) / experiment / name)
        runs.append(GridRun(name, settings, _build(RunConfig, merged, "")))
    return runs


def token_budget_from(run_directory: str | Path) -> int:
    """Measured training-token count of a finished run (for token-matched SFT, Table 11)."""
    results = json.loads((Path(run_directory) / "results.json").read_text(encoding="utf-8"))
    return int(results["selection"]["selected"]["training_tokens"])
