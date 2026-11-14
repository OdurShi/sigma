"""Run configuration: dataclasses loaded from YAML (values of Appendix C.5 as defaults)."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, get_args, get_origin, get_type_hints

import yaml

from sigma.models import PAPER_TARGET_MODULES, TINY_BACKBONE


@dataclass
class DataConfig:
    """Dataset, budget and output inventory."""

    name: str = "synthetic"  # "synthetic" | "banking77" | "clinc150" | "hwu64"
    directory: str = "data"  # prepared datasets live in <directory>/<name>
    shots: int = 10  # labeled examples per intent (b); n = b * m
    outputs: str = "original"  # "original" intent strings or "opaque" identifiers (Table 10)
    max_test_instances: int | None = None  # None = full test split
    max_dev_instances: int | None = None  # None = full development split
    label_subset: list[str] | None = None  # restrict to these groups (reduced-scale runs)
    synthetic_groups: int = 3
    synthetic_classes_per_group: int = 4


@dataclass
class LoraConfig:
    """Appendix C.5: rank 16, scaling 32, zero dropout on Q/K/V/O and gate/up/down."""

    rank: int = 16
    alpha: float = 32.0
    dropout: float = 0.0
    target_modules: list[str] = field(default_factory=lambda: list(PAPER_TARGET_MODULES))


@dataclass
class ModelConfig:
    name: str = "Qwen/Qwen2.5-7B-Instruct"  # or "tiny-random" (built in code)
    dtype: str = "bfloat16"
    device: str | None = None  # None = keep on CPU; "cuda", "mps"
    gradient_checkpointing: bool = True
    adaptation: str = "lora"  # "lora" (paper) or "full" (from-scratch tiny model only)
    lora: LoraConfig = field(default_factory=LoraConfig)
    end_of_answer: str = "auto"
    template_kwargs: dict[str, Any] = field(default_factory=dict)
    tiny_hidden_size: int = 96
    tiny_layers: int = 2
    tiny_heads: int = 4


@dataclass
class OptimConfig:
    """Appendix C.5 optimiser settings."""

    lr: float = 5e-5
    betas: list[float] = field(default_factory=lambda: [0.9, 0.95])
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    warmup_ratio: float = 0.03
    batch_size: int = 16  # B original instances per optimiser step
    passes: float = 10.0  # T = 2 * ceil(passes * n / (2 B))
    max_steps: int | None = None  # override of T (rounded up to an even number)
    token_budget: int | None = None  # processed-token budget for token-matched SFT


@dataclass
class ScoringConfig:
    strategy: str = "packed"  # "packed" | "rows"
    view_batch_size: int = 1
    candidate_chunk_size: int | None = None
    max_context_tokens: int | None = 8192
    gradient_mode: str = "two_pass"  # "two_pass" (Appendix C.5) | "direct"


@dataclass
class SelectionConfig:
    """Model selection on the fixed development pool (Appendix A.2 step 7, C.5)."""

    num_checkpoints: int = 5
    search: bool = False  # True runs the lr x coefficient grid below
    lr_grid: list[float] = field(default_factory=lambda: [1e-5, 5e-5, 1e-4])
    coefficient_grid: list[float] = field(default_factory=lambda: [0.0, 0.01, 0.1, 1.0, 10.0])


@dataclass
class EvalConfig:
    """Test-guidance protocol (Appendix C.2)."""

    binding_draws: int = 10
    composition_depths: list[int] = field(default_factory=lambda: [2, 4, 8])
    composition_draws: int = 10
    save_scores: bool = False


@dataclass
class RunConfig:
    method: str = "sigma"
    coefficient: float | None = None  # None = the method preset's nominal value
    seed: int = 13
    output_dir: str = "runs/default"
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    scoring: ScoringConfig = field(default_factory=ScoringConfig)
    selection: SelectionConfig = field(default_factory=SelectionConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)

    @property
    def is_tiny(self) -> bool:
        return self.model.name == TINY_BACKBONE


class ConfigError(ValueError):
    """Raised for unknown keys or malformed configuration files."""


def _build(cls: type, payload: dict[str, Any], path: str) -> Any:
    hints = get_type_hints(cls)
    known = {f.name for f in dataclasses.fields(cls)}
    unknown = set(payload) - known
    if unknown:
        raise ConfigError(
            f"unknown configuration key(s) at {path or 'top level'}: {sorted(unknown)}"
        )
    kwargs: dict[str, Any] = {}
    for name, value in payload.items():
        hint = hints[name]
        if dataclasses.is_dataclass(hint):
            if not isinstance(value, dict):
                raise ConfigError(f"{path}{name} must be a mapping")
            kwargs[name] = _build(hint, value, f"{path}{name}.")
        else:
            origin = get_origin(hint)
            if origin is list and value is not None:
                # PyYAML reads "5e-5" as a string, so numeric lists are converted explicitly.
                value = [float(v) for v in value] if get_args(hint) == (float,) else list(value)
            elif float in get_args(hint) or hint is float:
                value = float(value) if value is not None else None
            kwargs[name] = value
    return cls(**kwargs)


def _merge(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _read(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ConfigError(f"{path} must contain a mapping")
    merged: dict[str, Any] = {}
    for parent in payload.pop("defaults", []) or []:
        merged = _merge(merged, _read((path.parent / parent).resolve()))
    return _merge(merged, payload)


def apply_overrides(payload: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    """Apply ``key.subkey=value`` overrides (values are parsed as YAML)."""
    result = dict(payload)
    for item in overrides:
        if "=" not in item:
            raise ConfigError(f"override {item!r} is not of the form key=value")
        dotted, raw = item.split("=", 1)
        update: Any = yaml.safe_load(raw)
        for key in reversed(dotted.split(".")):
            update = {key: update}
        result = _merge(result, update)
    return result


def load_config(path: str | Path | None = None, overrides: list[str] | None = None) -> RunConfig:
    """Load a run configuration from YAML (with ``defaults:`` inheritance) plus overrides."""
    payload = _read(Path(path)) if path is not None else {}
    payload.pop("description", None)
    return _build(RunConfig, apply_overrides(payload, overrides or []), "")


def config_to_dict(config: RunConfig) -> dict[str, Any]:
    return dataclasses.asdict(config)
