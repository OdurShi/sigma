"""The datasets and derived artefacts shipped under ``data/`` (Appendix C.1, C.2)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from sigma.config import load_config
from sigma.data import datasets as ds
from sigma.data import nested_fewshot_subset
from sigma.data.fewshot import class_orderings
from sigma.data.synthetic import make_synthetic_data
from sigma.guidance import parse_guidance, render_guidance
from sigma.guidance.schema import task_from_dict
from sigma.pipeline import build_experiment, load_task_and_splits, training_subset
from sigma.seeding import PAPER_SEEDS

DATA = Path(__file__).resolve().parents[1] / "data"
NAMES = ("banking77", "clinc150", "hwu64")


@pytest.fixture(scope="module")
def loaded():
    return {name: ds.load_prepared(DATA / name) for name in NAMES}


@pytest.mark.parametrize("name", NAMES)
def test_shipped_splits_have_the_sizes_of_table_3(loaded, name):
    task, splits = loaded[name]
    ds.verify_split_sizes(name, splits)
    assert len(task.labels) == ds.DATASETS[name].num_labels == task.num_candidates
    assert set(task.labels) == {e.label for e in splits["train"]}
    assert task.binding == {z: z for z in task.labels}  # original intent strings under g0
    manifest = json.loads((DATA / name / "manifest.json").read_text())
    assert manifest["source_revision"] == ds.RELEASE_REVISION
    assert tuple(manifest["split_sizes"].values()) == ds.DATASETS[name].sizes


def test_checksums_match_the_shipped_files_and_are_listed_in_the_readme():
    listed = {}
    for line in (DATA / "SHA256SUMS").read_text().splitlines():
        digest, path = line.split("  ", 1)
        listed[path] = digest
    files = {p.relative_to(DATA).as_posix() for p in ds.shipped_files(DATA)}
    assert files == set(listed)
    for path, digest in listed.items():
        assert hashlib.sha256((DATA / path).read_bytes()).hexdigest() == digest, path
    readme = (DATA / "README.md").read_text()
    assert all(digest in readme for digest in listed.values())


@pytest.mark.parametrize("name", NAMES)
def test_shipped_fewshot_orderings_equal_the_seeded_sampler(loaded, name):
    _, splits = loaded[name]
    train = splits["train"]
    for seed in PAPER_SEEDS:
        record = json.loads((DATA / name / "fewshot" / f"seed{seed}.json").read_text())
        assert record["paper_shots"] == list(ds.PAPER_SHOTS) and record["max_shots"] == 16
        computed = class_orderings(train, seed)
        assert record["ordering"] == {z: order[:16] for z, order in computed.items()}
        previous: set = set()
        for shots in ds.PAPER_SHOTS:
            subset = ds.load_fewshot_subset(DATA / name, train, seed, shots)
            assert len(subset) == shots * ds.DATASETS[name].num_labels  # n = b * m
            assert previous <= set(subset)  # nested budgets
            previous = set(subset)
        assert ds.load_fewshot_subset(DATA / name, train, seed, 10) == nested_fewshot_subset(
            train, seed, 10
        )
    # Seeds or budgets without a shipped ordering fall back to the sampler.
    assert ds.load_fewshot_subset(DATA / name, train, 7, 3) == nested_fewshot_subset(train, 7, 3)
    assert len(ds.load_fewshot_subset(DATA / name, train, 13, 20)) == 20 * len(
        {e.label for e in train}
    )


@pytest.mark.parametrize("name", NAMES)
def test_hierarchy_task_record_and_guidance_document_are_consistent(loaded, name):
    task, _ = loaded[name]
    hierarchy = json.loads((DATA / name / "hierarchy.json").read_text())
    assert hierarchy["source"] == ds.DATASETS[name].hierarchy
    groups = hierarchy["groups"]
    assert sorted(z for members in groups.values() for z in members) == list(task.labels)
    rebuilt = ds.build_task(name, groups)
    assert render_guidance(rebuilt.base_guidance()) == render_guidance(task.base_guidance())
    document = (DATA / name / "guidance.txt").read_text().rstrip("\n")
    assert document == render_guidance(task.base_guidance())
    parsed = parse_guidance(document)
    assert parsed.binding == dict(task.binding)


def test_sibling_group_structure_of_the_shipped_hierarchies(loaded):
    sizes = {name: sorted(len(g) for g in loaded[name][0].sibling_groups()) for name in NAMES}
    assert sizes["clinc150"] == [15] * 10  # ten published domains of fifteen intents
    assert sizes["hwu64"] == [2, 2, 2, 3, 3, 3, 3, 3, 3, 4, 4, 5, 5, 9, 10]  # scenario prefixes
    assert sizes["banking77"] == [2, 3, 4, 5, 18, 42]  # average-linkage label clusters
    clusters = json.loads((DATA / "banking77" / "hierarchy.json").read_text())
    assert clusters["method"]["num_groups"] == len(clusters["groups"]) == 9  # ceil(sqrt(77))


def test_shipped_synthetic_task_is_the_generator_output():
    directory = DATA / "synthetic" / "quickstart"
    data = make_synthetic_data(**ds.SYNTHETIC_QUICKSTART)
    task = task_from_dict(json.loads((directory / "task.json").read_text()))
    assert render_guidance(task.base_guidance()) == render_guidance(data.task.base_guidance())
    for split, examples in (("train", data.train_pool), ("valid", data.dev), ("test", data.test)):
        assert ds._read_jsonl(directory / f"{split}.jsonl") == examples
    config = load_config(Path(__file__).resolve().parents[1] / "configs" / "quickstart.yaml")
    settings = (config.seed, config.data.synthetic_groups, config.data.synthetic_classes_per_group)
    assert settings == tuple(ds.SYNTHETIC_QUICKSTART.values())


def test_loaders_use_the_shipped_files_without_downloading(monkeypatch, tmp_path):
    def forbidden(*args, **kwargs):
        raise AssertionError("the loader tried to download a shipped dataset")

    monkeypatch.setattr(ds, "download_release", forbidden)
    monkeypatch.setattr(ds, "_fetch", forbidden)
    monkeypatch.chdir(tmp_path)  # the default "data" is resolved relative to the checkout
    assert ds.resolve_data_dir("data") == DATA
    config = load_config(None, ["data.name=hwu64", "data.shots=4", "seed=21"])
    task, splits = load_task_and_splits(config)
    assert task.num_candidates == 64 and len(splits["test"]) == 1076
    subset = training_subset(config, task, splits["train"])
    assert subset == nested_fewshot_subset(splits["train"], 21, 4) and len(subset) == 256
    # A label subset keeps the same per-class orderings.
    reduced = load_config(
        None, ["data.name=hwu64", "data.shots=4", "seed=21", "data.label_subset=[alarm, transport]"]
    )
    small_task, small_splits = load_task_and_splits(reduced)
    small = training_subset(reduced, small_task, small_splits["train"])
    assert small == [e for e in subset if e.label in small_task.labels] and len(small) == 28
    monkeypatch.setenv(ds.DATA_DIR_ENV, str(tmp_path / "elsewhere"))
    assert ds.resolve_data_dir("data") == tmp_path / "elsewhere"


def test_pipeline_builds_an_experiment_from_shipped_data_with_the_tiny_model():
    config = load_config(
        None,
        [
            "data.name=hwu64",
            "data.shots=1",
            "data.label_subset=[alarm, audio]",
            "data.max_test_instances=5",
            "model.name=tiny-random",
            "method=frozen",
        ],
    )
    experiment = build_experiment(config)
    assert experiment.task.num_candidates == 6 and len(experiment.train) == 6
    assert len(experiment.test) == 5 and len(experiment.task.eligible_parents()) == 2
