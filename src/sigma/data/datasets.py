"""BANKING77, CLINC150 and HWU64 (Appendix C.1) and the files shipped under ``data/``.

The paper uses the public few-shot intent-detection release that is also used by CPFT and
QAID, including its BANKING77 validation split.  That release stores every split as two
aligned text files, ``seq.in`` (one utterance per line) and ``label`` (one intent per line);
its split sizes are those of Table 3.

The repository ships the converted splits together with the artefacts derived from them
(hierarchy, task record, rendered base guidance, nested few-shot orderings) in
``data/<name>/``.  Loaders read those files by default and only download the upstream
release when a dataset directory is absent.  The parsing, verification and tree
construction in this module need no network.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from sigma.data.fewshot import Example, class_orderings, nested_fewshot_subset
from sigma.data.hierarchy import (
    LABEL_EMBEDDING_MODEL,
    embed_labels,
    humanize,
    label_cluster_groups,
    prefix_groups,
    task_from_groups,
)
from sigma.data.synthetic import make_synthetic_data
from sigma.guidance.render import render_guidance
from sigma.guidance.schema import TaskRecord, task_from_dict, task_to_dict
from sigma.seeding import PAPER_SEEDS

logger = logging.getLogger(__name__)

#: Upstream sources, pinned to the revisions the shipped files were converted from.
RELEASE_REPOSITORY = "https://github.com/jianguoz/Few-Shot-Intent-Detection"
RELEASE_REVISION = "fa9553312b1dba54b1be0aae2d30b13171f898e5"
RELEASE_URL = (
    "https://raw.githubusercontent.com/jianguoz/Few-Shot-Intent-Detection/"
    f"{RELEASE_REVISION}/Datasets"
)
CLINC_REPOSITORY = "https://github.com/clinc/oos-eval"
CLINC_REVISION = "828f8093932c8fe6ca7936c3d2e52903b1c523de"
CLINC_DOMAINS_URL = (
    f"https://raw.githubusercontent.com/clinc/oos-eval/{CLINC_REVISION}/data/domains.json"
)
SPLITS: tuple[str, ...] = ("train", "valid", "test")

#: Shot counts used in the paper: the sweep (1, 2, 4, 8, 16), the primary comparison (10) and
#: the backbone panel (5, 10).  Shipped few-shot orderings cover the largest of them.
PAPER_SHOTS: tuple[int, ...] = (1, 2, 4, 5, 8, 10, 16)

#: Environment variable that overrides the data directory.
DATA_DIR_ENV = "SIGMA_DATA_DIR"

#: Quick-start variant of the synthetic task shipped under ``data/synthetic/quickstart``.
SYNTHETIC_QUICKSTART = {"seed": 13, "num_groups": 2, "classes_per_group": 3}


@dataclass(frozen=True)
class DatasetSpec:
    """Static facts about one benchmark (Table 3)."""

    release_dir: str
    num_labels: int
    sizes: tuple[int, int, int]  # training pool, development, test
    hierarchy: str  # "label_clusters" | "clinc_domains" | "label_prefix"


DATASETS: dict[str, DatasetSpec] = {
    "banking77": DatasetSpec("BANKING77", 77, (8622, 1540, 3080), "label_clusters"),
    "clinc150": DatasetSpec("CLINC150", 150, (15000, 3000, 4500), "clinc_domains"),
    "hwu64": DatasetSpec("HWU64", 64, (8954, 1076, 1076), "label_prefix"),
}


def read_release_split(directory: Path) -> list[Example]:
    """Read one split directory holding aligned ``seq.in`` and ``label`` files."""
    texts = (directory / "seq.in").read_text(encoding="utf-8").splitlines()
    labels = (directory / "label").read_text(encoding="utf-8").splitlines()
    if len(texts) != len(labels):
        raise ValueError(f"{directory}: {len(texts)} utterances but {len(labels)} labels")
    return [Example(text.strip(), label.strip()) for text, label in zip(texts, labels, strict=True)]


def verify_split_sizes(name: str, splits: dict[str, list[Example]]) -> None:
    """Check the split sizes and label count against Table 3."""
    spec = DATASETS[name]
    sizes = tuple(len(splits[s]) for s in SPLITS)
    if sizes != spec.sizes:
        raise ValueError(f"{name}: split sizes {sizes} differ from Table 3 {spec.sizes}")
    labels = {e.label for e in splits["train"]}
    if len(labels) != spec.num_labels:
        raise ValueError(f"{name}: {len(labels)} training labels, expected {spec.num_labels}")
    for split in ("valid", "test"):
        unknown = {e.label for e in splits[split]} - labels
        if unknown:
            raise ValueError(f"{name}: {split} labels missing from training: {sorted(unknown)}")


def _fetch(url: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "sigma-data-preparation"})
    with urllib.request.urlopen(request, timeout=60) as response:  # noqa: S310 - fixed https URL
        target.write_bytes(response.read())


def download_release(name: str, raw_dir: Path) -> Path:
    """Download the release files of one dataset into ``raw_dir/<RELEASE_DIR>``."""
    spec = DATASETS[name]
    root = raw_dir / spec.release_dir
    for split in SPLITS:
        for filename in ("seq.in", "label"):
            target = root / split / filename
            if not target.exists():
                logger.info("downloading %s/%s/%s", spec.release_dir, split, filename)
                _fetch(f"{RELEASE_URL}/{spec.release_dir}/{split}/{filename}", target)
    if spec.hierarchy == "clinc_domains" and not (root / "domains.json").exists():
        _fetch(CLINC_DOMAINS_URL, root / "domains.json")
    return root


def build_groups(
    name: str, labels: list[str], release_root: Path, existing: Path | None = None
) -> dict[str, list[str]]:
    """Group the labels of one benchmark according to its hierarchy source (Appendix C.2).

    If ``existing`` points to a ``hierarchy.json`` that covers exactly ``labels`` its groups
    are reused, so the label clustering of BANKING77 is not recomputed.
    """
    if existing is not None and existing.exists():
        groups = json.loads(existing.read_text(encoding="utf-8"))["groups"]
        if sorted(z for members in groups.values() for z in members) == sorted(labels):
            return {group: sorted(members) for group, members in groups.items()}
    kind = DATASETS[name].hierarchy
    if kind == "label_prefix":
        return prefix_groups(labels)
    if kind == "clinc_domains":
        domains = json.loads((release_root / "domains.json").read_text(encoding="utf-8"))
        groups = {domain: sorted(members) for domain, members in domains.items()}
        if sorted(z for members in groups.values() for z in members) != sorted(labels):
            raise ValueError("CLINC150 domain metadata does not cover the in-scope labels")
        return groups
    return label_cluster_groups(labels, embed_labels(labels))


def build_task(name: str, groups: dict[str, list[str]]) -> TaskRecord:
    """Root -> group -> intent task with the original intent strings as outputs."""
    clustered = DATASETS[name].hierarchy == "label_clusters"
    descriptions = None if clustered else {g: f"{humanize(g)} intents" for g in groups}
    return task_from_groups(name, groups, group_descriptions=descriptions)


def _hierarchy_record(name: str, groups: dict[str, list[str]]) -> dict:
    """Provenance of a hierarchy together with its groups (written to ``hierarchy.json``)."""
    kind = DATASETS[name].hierarchy
    if kind == "label_clusters":
        method = {
            "embedding_model": LABEL_EMBEDDING_MODEL,
            "embedded_text": "humanised label names",
            "distance": "cosine",
            "linkage": "average",
            "num_groups": len(groups),
            "tie_breaking": "sorted label identifiers",
        }
    elif kind == "clinc_domains":
        method = {"metadata": CLINC_DOMAINS_URL}
    else:
        method = {"rule": "scenario prefix of the intent name (text before the first underscore)"}
    return {"dataset": name, "source": kind, "method": method, "groups": groups}


def fewshot_record(name: str, examples: list[Example], seed: int) -> dict:
    """Nested few-shot orderings of one seed: per label, the first training indices.

    The ``b``-shot subset consists of the first ``b`` indices of every label, so the subsets
    for the budgets in :data:`PAPER_SHOTS` are nested prefixes of one ordering.
    """
    limit = max(PAPER_SHOTS)
    ordering = {label: order[:limit] for label, order in class_orderings(examples, seed).items()}
    return {
        "dataset": name,
        "seed": seed,
        "max_shots": limit,
        "paper_shots": list(PAPER_SHOTS),
        "train_file": "train.jsonl",
        "ordering": ordering,
    }


def _dump_fewshot(record: dict) -> str:
    """JSON with one label per line (keeps the files small and diff-friendly)."""
    head = {key: value for key, value in record.items() if key != "ordering"}
    lines = [json.dumps(head)[:-1] + ', "ordering": {']
    items = list(record["ordering"].items())
    for position, (label, indices) in enumerate(items):
        comma = "," if position < len(items) - 1 else ""
        lines.append(f" {json.dumps(label)}: {json.dumps(indices)}{comma}")
    lines.append("}}")
    return "\n".join(lines) + "\n"


def _write_jsonl(path: Path, examples: list[Example]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(json.dumps({"text": example.text, "label": example.label}) + "\n")


def _read_jsonl(path: Path) -> list[Example]:
    with path.open(encoding="utf-8") as handle:
        return [Example(**json.loads(line)) for line in handle if line.strip()]


def prepare_dataset(name: str, raw_dir: Path, out_dir: Path) -> Path:
    """Download (if needed), verify and convert one benchmark; returns its directory.

    Writes into ``out_dir/<name>``:

    * ``{train,valid,test}.jsonl``: the splits of Table 3, one ``{"text", "label"}`` per line;
    * ``hierarchy.json``: the sibling groups and how they were obtained;
    * ``task.json``: tree, descriptions and base binding (the task record);
    * ``guidance.txt``: the rendered base guidance document ``g0``;
    * ``fewshot/seed<seed>.json``: nested few-shot orderings for the paper's seeds;
    * ``manifest.json``: upstream source, revision and SHA-256 of the upstream files.
    """
    release_root = download_release(name, raw_dir)
    splits = {split: read_release_split(release_root / split) for split in SPLITS}
    verify_split_sizes(name, splits)
    labels = sorted({e.label for e in splits["train"]})
    target = out_dir / name
    target.mkdir(parents=True, exist_ok=True)
    groups = build_groups(name, labels, release_root, existing=target / "hierarchy.json")
    task = build_task(name, groups)
    for split, examples in splits.items():
        _write_jsonl(target / f"{split}.jsonl", examples)
    (target / "hierarchy.json").write_text(
        json.dumps(_hierarchy_record(name, groups), indent=1) + "\n", encoding="utf-8"
    )
    (target / "task.json").write_text(
        json.dumps(task_to_dict(task), indent=1) + "\n", encoding="utf-8"
    )
    (target / "guidance.txt").write_text(
        render_guidance(task.base_guidance()) + "\n", encoding="utf-8"
    )
    (target / "fewshot").mkdir(exist_ok=True)
    for seed in PAPER_SEEDS:
        (target / "fewshot" / f"seed{seed}.json").write_text(
            _dump_fewshot(fewshot_record(name, splits["train"], seed)), encoding="utf-8"
        )
    manifest = {
        "dataset": name,
        "source_repository": RELEASE_REPOSITORY,
        "source_revision": RELEASE_REVISION,
        "source_path": f"Datasets/{DATASETS[name].release_dir}",
        "upstream_sha256": {
            f"{split}/{filename}": hashlib.sha256(
                (release_root / split / filename).read_bytes()
            ).hexdigest()
            for split in SPLITS
            for filename in ("seq.in", "label")
        },
        "num_labels": len(labels),
        "split_sizes": {split: len(examples) for split, examples in splits.items()},
        "eligible_sibling_groups": len(task.eligible_parents()),
    }
    (target / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    return target


def prepare_synthetic(out_dir: Path) -> Path:
    """Write the quick-start variant of the synthetic task to ``out_dir/synthetic/quickstart``.

    The files are the output of :func:`sigma.data.synthetic.make_synthetic_data` for the
    settings in :data:`SYNTHETIC_QUICKSTART`; the pipeline itself regenerates the data in
    code, so they serve as a reference copy.
    """
    data = make_synthetic_data(**SYNTHETIC_QUICKSTART)
    target = out_dir / "synthetic" / "quickstart"
    target.mkdir(parents=True, exist_ok=True)
    splits = {"train": data.train_pool, "valid": data.dev, "test": data.test}
    for split, examples in splits.items():
        _write_jsonl(target / f"{split}.jsonl", examples)
    (target / "task.json").write_text(
        json.dumps(task_to_dict(data.task), indent=1) + "\n", encoding="utf-8"
    )
    (target / "guidance.txt").write_text(
        render_guidance(data.task.base_guidance()) + "\n", encoding="utf-8"
    )
    manifest = {"generator": "sigma.data.synthetic.make_synthetic_data", **SYNTHETIC_QUICKSTART}
    (target / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    return target


def shipped_files(data_dir: Path) -> list[Path]:
    """All data files below ``data_dir`` that are part of the repository (sorted)."""
    skipped = {"README.md", "SHA256SUMS", ".DS_Store"}
    return sorted(
        path
        for path in data_dir.rglob("*")
        if path.is_file()
        and path.name not in skipped
        and "raw" not in path.relative_to(data_dir).parts
    )


def write_checksums(data_dir: Path) -> Path:
    """Write ``SHA256SUMS`` (``<sha256>  <relative path>``) for the shipped data files."""
    lines = [
        f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(data_dir).as_posix()}"
        for path in shipped_files(data_dir)
    ]
    target = data_dir / "SHA256SUMS"
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return target


# --------------------------------------------------------------------------- loading
def resolve_data_dir(directory: str | Path = "data") -> Path:
    """Locate the data directory.

    Order: the ``SIGMA_DATA_DIR`` environment variable; ``directory`` if it is absolute or
    exists relative to the working directory; the ``data/`` folder of the source checkout.
    """
    override = os.environ.get(DATA_DIR_ENV)
    if override:
        return Path(override)
    path = Path(directory)
    if path.is_absolute() or path.exists():
        return path
    checkout = Path(__file__).resolve().parents[3] / path
    return checkout if checkout.exists() else path


def load_prepared(directory: Path) -> tuple[TaskRecord, dict[str, list[Example]]]:
    """Load a dataset directory written by :func:`prepare_dataset`."""
    task = task_from_dict(json.loads((directory / "task.json").read_text(encoding="utf-8")))
    return task, {split: _read_jsonl(directory / f"{split}.jsonl") for split in SPLITS}


def load_dataset(name: str, data_dir: Path) -> tuple[TaskRecord, dict[str, list[Example]]]:
    """Load a benchmark from ``data_dir/<name>``; download and convert it only if absent."""
    directory = data_dir / name
    if not (directory / "task.json").exists():
        logger.info("%s not found in %s; downloading the upstream release", name, data_dir)
        prepare_dataset(name, data_dir / "raw", data_dir)
    return load_prepared(directory)


def load_fewshot_subset(
    directory: Path, train: list[Example], seed: int, shots: int
) -> list[Example]:
    """The ``shots``-shot training subset of ``seed``.

    Uses the shipped ordering ``directory/fewshot/seed<seed>.json`` when it exists and covers
    the requested budget, and otherwise computes the same ordering with
    :func:`sigma.data.fewshot.nested_fewshot_subset`.  ``train`` must be the complete training
    pool of the dataset, in file order.
    """
    path = directory / "fewshot" / f"seed{seed}.json"
    if path.exists():
        record = json.loads(path.read_text(encoding="utf-8"))
        if shots <= record["max_shots"]:
            ordering = record["ordering"]
            return [train[i] for label in sorted(ordering) for i in ordering[label][:shots]]
    return nested_fewshot_subset(train, seed, shots)
