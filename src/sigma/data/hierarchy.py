"""Task hierarchies for the intent benchmarks (Appendix C.2, "Schema construction").

* CLINC150 and HWU64 use root -> domain -> intent trees built from published domain /
  scenario metadata.
* BANKING77 has no published grouping.  Its label-only groups come from frozen sentence
  embeddings of the label names, cosine distance and average-linkage clustering cut into
  ``ceil(sqrt(m))`` groups; ties are resolved by sorted label identifiers and group
  descriptions list their member intent names.

Only label metadata is used; evaluation inputs and test annotations never influence a tree.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import numpy as np

from sigma.guidance.schema import TaskRecord, grouped_task, with_binding
from sigma.seeding import stream

LABEL_EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"


def humanize(identifier: str) -> str:
    """``card_arrival`` -> ``card arrival`` (default semantic description of a label)."""
    return " ".join(identifier.replace("_", " ").split()).lower()


def average_linkage_groups(
    labels: Sequence[str], embeddings: np.ndarray, num_groups: int | None = None
) -> list[list[str]]:
    """Cluster labels by average linkage on cosine distance and cut into ``num_groups``.

    ``num_groups`` defaults to ``ceil(sqrt(m))``.  Labels are processed in sorted order and
    ties between equally distant cluster pairs are resolved by the smallest member labels,
    which makes the result independent of the input order.

    Returns the groups as sorted label lists, ordered by their first label.
    """
    if len(labels) != len(embeddings):
        raise ValueError("one embedding per label is required")
    order = sorted(range(len(labels)), key=lambda i: labels[i])
    names = [labels[i] for i in order]
    vectors = np.asarray(embeddings, dtype=np.float64)[order]
    vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    distance = 1.0 - vectors @ vectors.T
    m = len(names)
    target = math.ceil(math.sqrt(m)) if num_groups is None else num_groups
    if not 1 <= target <= m:
        raise ValueError("num_groups must lie between 1 and the number of labels")
    clusters: dict[int, list[int]] = {i: [i] for i in range(m)}
    # dist[a][b] for a < b holds the average pairwise distance between clusters a and b.
    dist = {(a, b): float(distance[a, b]) for a in range(m) for b in range(a + 1, m)}
    while len(clusters) > target:
        # Cluster ids equal their smallest member index, so the lexicographic minimum over
        # (distance, a, b) resolves ties by sorted label identifiers.
        _, a, b = min((d, a, b) for (a, b), d in dist.items())
        size_a, size_b = len(clusters[a]), len(clusters[b])
        for c in clusters:
            if c in (a, b):
                continue
            d_ac = dist.pop((min(a, c), max(a, c)))
            d_bc = dist.pop((min(b, c), max(b, c)))
            dist[(min(a, c), max(a, c))] = (size_a * d_ac + size_b * d_bc) / (size_a + size_b)
        del dist[(a, b)]
        clusters[a] = sorted(clusters[a] + clusters.pop(b))
    return [[names[i] for i in members] for _, members in sorted(clusters.items())]


def task_from_groups(
    name: str,
    groups: Mapping[str, Sequence[str]],
    label_descriptions: Mapping[str, str] | None = None,
    group_descriptions: Mapping[str, str] | None = None,
) -> TaskRecord:
    """Root -> group -> intent tree with identity bindings (original intent strings).

    Label descriptions default to the humanised label name.  Group descriptions default to a
    list of the member intent names (Appendix C.2).
    """
    all_labels = {z for members in groups.values() for z in members}
    spec: dict[str, tuple[str, dict[str, str]]] = {}
    for group, members in groups.items():
        identifier = group if group not in all_labels and group != "root" else f"{group}_group"
        description = (group_descriptions or {}).get(group) or "Intents: " + ", ".join(
            sorted(members)
        )
        spec[identifier] = (
            description,
            {z: (label_descriptions or {}).get(z) or humanize(z) for z in members},
        )
    return grouped_task(name, spec, root_description=f"All intents of {name}.")


def label_cluster_groups(labels: Sequence[str], embeddings: np.ndarray) -> dict[str, list[str]]:
    """Label-only groups ``group_01 ...`` for a dataset without published domains."""
    clusters = average_linkage_groups(labels, embeddings)
    width = len(str(len(clusters)))
    return {f"group_{i + 1:0{width}d}": members for i, members in enumerate(clusters)}


def prefix_groups(labels: Sequence[str], separator: str = "_") -> dict[str, list[str]]:
    """Group labels by the scenario prefix of their name (HWU64: ``alarm_set`` -> ``alarm``)."""
    groups: dict[str, list[str]] = {}
    for label in sorted(labels):
        groups.setdefault(label.split(separator, 1)[0], []).append(label)
    return groups


def embed_labels(labels: Sequence[str], model_name: str = LABEL_EMBEDDING_MODEL) -> np.ndarray:
    """Embed humanised label names with a frozen sentence encoder (needs the ``data`` extra)."""
    from sentence_transformers import SentenceTransformer

    encoder = SentenceTransformer(model_name)
    return np.asarray(encoder.encode([humanize(z) for z in labels], normalize_embeddings=True))


def with_opaque_outputs(task: TaskRecord, seed: int = 0) -> TaskRecord:
    """Replace output strings by unique opaque identifiers (verbaliser control, Table 10).

    Identifiers ``out_000 ...`` are assigned to the sorted labels by a fixed,
    label-independent permutation that depends only on ``seed``; semantic descriptions, tree
    and edit definitions are unchanged.
    """
    m = len(task.labels)
    width = max(3, len(str(m - 1)))
    permutation = stream(seed, "opaque-outputs").permutation(m)
    binding = {label: f"out_{int(permutation[i]):0{width}d}" for i, label in enumerate(task.labels)}
    return with_binding(task, binding, name=f"{task.name}-opaque")


def restrict_task(task: TaskRecord, groups: Sequence[str]) -> TaskRecord:
    """Sub-task containing only the given groups of a root -> group -> intent tree.

    Used for reduced-scale runs; the candidate inventory shrinks to the outputs of the kept
    intents, so results are not comparable with full-inventory numbers.
    """
    unknown = [g for g in groups if g not in task.nodes or task.nodes[g].is_leaf]
    if unknown:
        raise ValueError(f"unknown groups: {unknown}")
    spec = {
        group: (
            task.nodes[group].description,
            {z: task.nodes[z].description for z in task.nodes[group].children},
        )
        for group in groups
    }
    binding = {z: task.binding[z] for _, members in spec.values() for z in members}
    return grouped_task(
        f"{task.name}-subset",
        spec,
        binding=binding,
        root_description=task.nodes[task.root].description,
        instruction=task.instruction,
    )
