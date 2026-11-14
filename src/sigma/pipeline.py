"""End-to-end experiment: data, model, training with model selection, evaluation.

This module wires the components together exactly as in Appendix A.2: the task record is
validated, the labeled subset is drawn before any guidance view is constructed, the adapter
is trained with the chosen method, the checkpoint (and optionally the learning rate and loss
coefficient) is selected on the fixed development pool, and the frozen model is evaluated
under the test-guidance protocol.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import torch
import yaml

from sigma.config import RunConfig, config_to_dict
from sigma.data.datasets import load_dataset, load_fewshot_subset, resolve_data_dir
from sigma.data.fewshot import Example, nested_fewshot_subset
from sigma.data.hierarchy import restrict_task, with_opaque_outputs
from sigma.data.synthetic import make_synthetic_data
from sigma.evaluation.aggregate import per_instance_scores, summarize
from sigma.evaluation.evaluate import DevPool, build_dev_pool, evaluate
from sigma.evaluation.protocol import EvalPlan
from sigma.evaluation.records import EvalRecords
from sigma.guidance.render import render_prompt
from sigma.guidance.schema import TaskRecord
from sigma.methods import MethodSpec, resolve_method
from sigma.models import (
    LoraSettings,
    TinyBackboneSettings,
    apply_lora,
    build_tiny_backbone,
    count_trainable_parameters,
    load_backbone,
    load_trainable_state,
    make_all_trainable,
    tiny_encoder,
    trainable_state,
)
from sigma.predictors import DirectPredictor, FactorizedPredictor, Predictor, semantic_task
from sigma.scoring.scorer import CandidateScorer
from sigma.seeding import seed_everything, stream
from sigma.training.trainer import Trainer, TrainResult

logger = logging.getLogger(__name__)


@dataclass
class Experiment:
    """Everything a run needs: task, splits, model and scorer."""

    config: RunConfig
    method: MethodSpec
    task: TaskRecord
    train: list[Example]
    dev: list[Example]
    test: list[Example]
    model: torch.nn.Module
    scorer: CandidateScorer

    def predictor(self) -> Predictor:
        """A fresh predictor for the current parameters (no cached scores)."""
        if self.method.factorized:
            return FactorizedPredictor(self.scorer, self.task)
        return DirectPredictor(self.scorer, self.task)


def load_task_and_splits(config: RunConfig) -> tuple[TaskRecord, dict[str, list[Example]]]:
    """Load the task record and the train pool / development / test splits.

    Benchmarks are read from the data directory (``data/`` of the repository by default, see
    :func:`sigma.data.datasets.resolve_data_dir`); the synthetic task is generated in code.
    """
    data = config.data
    if data.name == "synthetic":
        synthetic = make_synthetic_data(
            config.seed, data.synthetic_groups, data.synthetic_classes_per_group
        )
        task = synthetic.task
        splits = {"train": synthetic.train_pool, "valid": synthetic.dev, "test": synthetic.test}
    else:
        task, splits = load_dataset(data.name, resolve_data_dir(data.directory))
    if data.label_subset:
        task = restrict_task(task, data.label_subset)
        keep = set(task.labels)
        splits = {k: [e for e in v if e.label in keep] for k, v in splits.items()}
    if data.outputs == "opaque":
        task = with_opaque_outputs(task)
    elif data.outputs != "original":
        raise ValueError(f"unknown output inventory {data.outputs!r}")
    return task, splits


def training_subset(config: RunConfig, task: TaskRecord, pool: list[Example]) -> list[Example]:
    """The ``shots``-shot labeled subset of the run seed (nested across budgets).

    For the benchmarks the shipped orderings ``data/<name>/fewshot/seed<seed>.json`` are used
    when present; they are identical to what the seeded sampler computes.
    """
    data = config.data
    if data.name == "synthetic":
        return nested_fewshot_subset(pool, config.seed, data.shots)
    directory = resolve_data_dir(data.directory) / data.name
    _, full = load_dataset(data.name, resolve_data_dir(data.directory))
    subset = load_fewshot_subset(directory, full["train"], config.seed, data.shots)
    keep = set(task.labels)
    return [example for example in subset if example.label in keep]


def fixed_subset(examples: list[Example], size: int | None, name: str) -> list[Example]:
    """A fixed random subset of ``size`` instances (all of them if ``size`` is ``None``).

    The subset depends only on the split name and its length, never on the run seed or the
    method, so reduced evaluations stay comparable across runs.  Original order is kept.
    """
    if size is None or size >= len(examples):
        return list(examples)
    chosen = stream(0, "fixed-subset", name).permutation(len(examples))[:size]
    return [examples[i] for i in sorted(chosen.tolist())]


def _vocabulary_texts(task: TaskRecord, splits: dict[str, list[Example]]) -> list[str]:
    texts = [render_prompt(task.base_guidance(), "").as_text(), *task.candidates, *task.labels]
    return texts + [e.text for split in splits.values() for e in split]


def build_experiment(config: RunConfig) -> Experiment:
    """Validate the task, draw the labeled subset and build the (adapted) model."""
    seed_everything(config.seed)
    method = resolve_method(config.method)
    if config.coefficient is not None:
        method = method.with_coefficient(config.coefficient)
    task, splits = load_task_and_splits(config)
    # The labeled subset is fixed before any guidance view is constructed (Appendix A.2).
    train = training_subset(config, task, splits["train"])
    dev = fixed_subset(splits["valid"], config.data.max_dev_instances, "dev")
    test = fixed_subset(splits["test"], config.data.max_test_instances, "test")
    cfg = config.model
    if config.is_tiny:
        encoder = tiny_encoder(_vocabulary_texts(task, splits))
        settings = TinyBackboneSettings(
            hidden_size=cfg.tiny_hidden_size,
            intermediate_size=2 * cfg.tiny_hidden_size,
            num_layers=cfg.tiny_layers,
            num_heads=cfg.tiny_heads,
        )
        model = build_tiny_backbone(len(encoder.tokenizer), config.seed, settings)
        if cfg.device:
            model = model.to(cfg.device)
    else:
        model, encoder = load_backbone(
            cfg.name,
            cfg.dtype,
            cfg.device,
            cfg.gradient_checkpointing,
            cfg.end_of_answer,
            cfg.template_kwargs,
        )
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if method.trainable:
        if cfg.adaptation == "lora":
            lora = LoraSettings(
                cfg.lora.rank, cfg.lora.alpha, cfg.lora.dropout, tuple(cfg.lora.target_modules)
            )
            model = apply_lora(model, lora, config.seed)
        elif cfg.adaptation == "full":
            model = make_all_trainable(model)
        else:
            raise ValueError(f"unknown adaptation {cfg.adaptation!r}")
    scoring = config.scoring
    scorer = CandidateScorer(
        model,
        encoder,
        strategy=scoring.strategy,
        view_batch_size=scoring.view_batch_size,
        candidate_chunk_size=scoring.candidate_chunk_size,
        max_context_tokens=scoring.max_context_tokens,
    )
    return Experiment(config, method, task, train, dev, test, model, scorer)


@dataclass
class Trial:
    """One configuration trial of the development search (Appendix C.5)."""

    lr: float
    coefficient: float
    result: TrainResult

    def sort_key(self) -> tuple[float, float, int, float]:
        """Higher accuracy first; ties favour the smaller coefficient, the earlier
        checkpoint and then the smaller learning rate."""
        return (-self.result.best_dev_accuracy, self.coefficient, self.result.best_step, self.lr)


def select_trial(trials: list[Trial]) -> Trial:
    """Apply the selection rule of Appendix C.5 to a list of finished trials."""
    return min(trials, key=Trial.sort_key)


def train_with_selection(experiment: Experiment, dev_pool: DevPool) -> tuple[Trial, list[Trial]]:
    """Train the method, optionally over the learning-rate x coefficient grid.

    Every trial starts from the same adapter initialisation and uses the same minibatches,
    edits and development pool; at most ``3 x 5 = 15`` trials are run, as in the paper.
    """
    config, method = experiment.config, experiment.method
    selection = config.selection
    if selection.search:
        coefficients = (
            selection.coefficient_grid if method.tunes_coefficient else [method.coefficient]
        )
        grid = [(lr, c) for lr in selection.lr_grid for c in coefficients]
    else:
        grid = [(config.optim.lr, method.coefficient)]
    initial = trainable_state(experiment.model)
    train_task = semantic_task(experiment.task) if method.factorized else experiment.task
    trials: list[Trial] = []
    for lr, coefficient in grid:
        load_trainable_state(experiment.model, initial)
        trainer = Trainer(
            experiment.model,
            experiment.scorer,
            train_task,
            experiment.train,
            method.with_coefficient(coefficient),
            replace(config.optim, lr=lr),
            config.scoring,
            config.seed,
            dev_accuracy=lambda: dev_pool.accuracy(experiment.predictor()),
            num_checkpoints=selection.num_checkpoints,
            include_outputs=not method.factorized,
        )
        logger.info("trial lr=%g coefficient=%g: %d steps", lr, coefficient, trainer.total_steps)
        trials.append(Trial(lr, coefficient, trainer.fit()))
    best = select_trial(trials)
    load_trainable_state(experiment.model, best.result.best_state)
    return best, trials


def _trial_summary(trial: Trial) -> dict:
    result = trial.result
    return {
        "lr": trial.lr,
        "coefficient": trial.coefficient,
        "total_steps": result.total_steps,
        "best_step": result.best_step,
        "best_dev_accuracy": result.best_dev_accuracy,
        "dev_curve": result.dev_curve,
        "training_tokens": result.training_tokens,
        "intervention_counts": asdict(result.counts),
    }


def _finite(value):
    """Replace undefined (non-finite) metric values by ``None`` so that the JSON is strict."""
    if isinstance(value, dict):
        return {key: _finite(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_finite(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def run(config: RunConfig) -> dict:
    """Train (if the method is trainable), select, evaluate and write the run directory.

    Returns the result dictionary that is also written to ``results.json``.
    """
    experiment = build_experiment(config)
    output = Path(config.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.yaml").write_text(yaml.safe_dump(config_to_dict(config)), encoding="utf-8")
    task = experiment.task
    logger.info(
        "task %s: m=%d candidates, %d eligible sibling groups, n=%d training labels",
        task.name,
        task.num_candidates,
        len(task.eligible_parents()),
        len(experiment.train),
    )
    results: dict = {
        "method": experiment.method.name,
        "seed": config.seed,
        "dataset": config.data.name,
        "shots": config.data.shots,
        "backbone": config.model.name,
        "num_training_labels": len(experiment.train),
        "trainable_parameters": count_trainable_parameters(experiment.model),
    }
    if experiment.method.trainable:
        dev_pool = build_dev_pool(task, experiment.dev, config.seed)
        best, trials = train_with_selection(experiment, dev_pool)
        results["selection"] = {
            "selected": _trial_summary(best),
            "trials": [_trial_summary(t) for t in trials],
        }
        (output / "train_log.json").write_text(json.dumps(best.result.history), encoding="utf-8")
        torch.save(best.result.best_state, output / "adapter_state.pt")
    plan = EvalPlan(
        config.eval.binding_draws,
        tuple(config.eval.composition_depths),
        config.eval.composition_draws,
    )
    experiment.scorer.stats.reset()
    records = evaluate(experiment.predictor(), task, experiment.test, plan, config.seed)
    results["metrics"] = summarize(records)
    results["evaluation_tokens"] = experiment.scorer.stats.processed_tokens
    np.savez_compressed(output / "per_instance.npz", **per_instance_scores(records))
    if config.eval.save_scores:
        records.save(output / "records.npz")
    results = _finite(results)
    (output / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results


def load_records(run_directory: str | Path) -> EvalRecords:
    """Load the raw evaluation records of a run saved with ``eval.save_scores: true``."""
    return EvalRecords.load(Path(run_directory) / "records.npz")


def load_run(run_directory: str | Path, overrides: list[str] | None = None) -> Experiment:
    """Rebuild the experiment of a finished run and restore its selected adapter weights."""
    from sigma.config import load_config

    directory = Path(run_directory)
    experiment = build_experiment(load_config(directory / "config.yaml", overrides))
    state_path = directory / "adapter_state.pt"
    if state_path.exists():
        load_trainable_state(experiment.model, torch.load(state_path, map_location="cpu"))
    return experiment
