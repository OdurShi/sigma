"""Few-shot sampling, hierarchies, dataset parsing and the synthetic task (Appendix C.1, C.2)."""

from __future__ import annotations

import math
from collections import Counter

import numpy as np
import pytest

from sigma.data import Example, class_orderings, make_synthetic_data, nested_fewshot_subset
from sigma.data.datasets import DATASETS, build_task, read_release_split, verify_split_sizes
from sigma.data.hierarchy import (
    average_linkage_groups,
    humanize,
    label_cluster_groups,
    prefix_groups,
    restrict_task,
    task_from_groups,
    with_opaque_outputs,
)
from sigma.guidance import binding_edit_support


def _pool(num_classes=5, per_class=20):
    return [
        Example(f"utterance {c}-{i}", f"class_{c}")
        for i in range(per_class)
        for c in range(num_classes)
    ]


def test_fewshot_subsets_are_balanced_and_nested():
    pool = _pool()
    subsets = {b: nested_fewshot_subset(pool, seed=13, shots=b) for b in (1, 2, 4, 8, 16)}
    for shots, subset in subsets.items():
        assert len(subset) == shots * 5  # n = b * m
        assert set(Counter(e.label for e in subset).values()) == {shots}
        assert len({e.text for e in subset}) == len(subset)  # distinct annotations
    for small, large in zip((1, 2, 4, 8), (2, 4, 8, 16), strict=True):
        assert set(subsets[small]) <= set(subsets[large])  # prefixes of one ordering


def test_fewshot_subsets_depend_on_the_seed_only():
    pool = _pool()
    assert nested_fewshot_subset(pool, 13, 4) == nested_fewshot_subset(pool, 13, 4)
    assert nested_fewshot_subset(pool, 13, 4) != nested_fewshot_subset(pool, 21, 4)
    # The ordering of a class does not depend on which other classes are present.
    reduced = [e for e in pool if e.label != "class_0"]
    full_order = class_orderings(pool, 42)["class_3"]
    reduced_order = class_orderings(reduced, 42)["class_3"]
    assert [pool[i] for i in full_order] == [reduced[i] for i in reduced_order]


def test_fewshot_budget_is_capped_by_class_size():
    pool = [*_pool(2, 3), Example("rare", "class_rare")]
    subset = nested_fewshot_subset(pool, 0, 10)
    assert Counter(e.label for e in subset) == {"class_0": 3, "class_1": 3, "class_rare": 1}
    with pytest.raises(ValueError):
        nested_fewshot_subset(pool, 0, 0)


def test_average_linkage_recovers_well_separated_groups():
    rng = np.random.default_rng(0)
    centres = np.eye(3)
    labels = [f"label_{i:02d}" for i in range(9)]
    embeddings = np.array([centres[i % 3] + 0.05 * rng.standard_normal(3) for i in range(9)])
    groups = average_linkage_groups(labels, embeddings)  # ceil(sqrt(9)) = 3 groups
    assert sorted(map(tuple, groups)) == [
        ("label_00", "label_03", "label_06"),
        ("label_01", "label_04", "label_07"),
        ("label_02", "label_05", "label_08"),
    ]
    shuffled = rng.permutation(9)
    again = average_linkage_groups([labels[i] for i in shuffled], embeddings[shuffled])
    assert again == groups  # independent of the input order


def test_average_linkage_cuts_into_ceil_sqrt_m_groups_and_breaks_ties_by_label():
    labels = [f"l{i}" for i in range(7)]
    groups = average_linkage_groups(labels, np.random.default_rng(1).standard_normal((7, 4)))
    assert len(groups) == math.ceil(math.sqrt(7)) == 3
    assert sorted(z for g in groups for z in g) == labels
    # Four equidistant points: the tie is resolved in sorted label order.
    equidistant = np.eye(4)
    assert average_linkage_groups(["d", "c", "b", "a"], equidistant, 3) == [
        ["a", "b"],
        ["c"],
        ["d"],
    ]
    clustered = label_cluster_groups(["d", "c", "b", "a"], equidistant)
    assert list(clustered) == ["group_1", "group_2"]


def test_prefix_groups_follow_scenario_names():
    labels = ["alarm_set", "alarm_query", "iot_hue_lighton", "iot_coffee", "weather_query"]
    groups = prefix_groups(labels)
    assert groups == {
        "alarm": ["alarm_query", "alarm_set"],
        "iot": ["iot_coffee", "iot_hue_lighton"],
        "weather": ["weather_query"],
    }
    task = build_task("hwu64", groups)
    assert task.eligible_parents() == ("alarm", "iot")  # the singleton scenario is not eligible
    assert task.binding["alarm_set"] == "alarm_set"  # original intent strings, identity binding
    assert task.nodes["alarm_set"].description == humanize("alarm_set") == "alarm set"


def test_group_descriptions_list_member_intents_and_ids_never_collide():
    task = task_from_groups(
        "demo", {"group_1": ["b_intent", "a_intent"], "travel": ["travel", "x"]}
    )
    assert task.nodes["group_1"].description == "Intents: a_intent, b_intent"
    assert "travel_group" in task.nodes and task.nodes["travel"].is_leaf
    assert task.candidates == ("a_intent", "b_intent", "travel", "x")


def test_opaque_outputs_are_a_fixed_label_independent_bijection(toy_task):
    opaque = with_opaque_outputs(toy_task)
    assert len(set(opaque.binding.values())) == toy_task.num_candidates
    assert all(c.startswith("out_") for c in opaque.candidates)
    assert opaque.binding == with_opaque_outputs(toy_task).binding  # deterministic
    assert opaque.labels == toy_task.labels
    assert {z: n.description for z, n in opaque.nodes.items()} == {
        z: n.description for z, n in toy_task.nodes.items()
    }
    assert len(binding_edit_support(opaque)) == len(binding_edit_support(toy_task))
    # The assignment is not the sorted (label-order) one.
    assert [opaque.binding[z] for z in opaque.labels] != sorted(opaque.binding.values())


def test_restrict_task_keeps_selected_groups(toy_task):
    subset = restrict_task(toy_task, ["billing", "misc"])
    assert subset.labels == ("charge", "invoice", "other", "refund")
    assert subset.binding["refund"] == "desk red"
    assert subset.eligible_parents() == ("billing",)
    with pytest.raises(ValueError):
        restrict_task(toy_task, ["refund"])


def test_release_split_parsing_and_size_verification(tmp_path):
    split = tmp_path / "train"
    split.mkdir()
    (split / "seq.in").write_text("first utterance\nsecond utterance \n", encoding="utf-8")
    (split / "label").write_text("alarm_set\nalarm_query\n", encoding="utf-8")
    examples = read_release_split(split)
    assert examples == [
        Example("first utterance", "alarm_set"),
        Example("second utterance", "alarm_query"),
    ]
    (split / "label").write_text("alarm_set\n", encoding="utf-8")
    with pytest.raises(ValueError, match="utterances"):
        read_release_split(split)
    with pytest.raises(ValueError, match="Table 3"):
        verify_split_sizes("hwu64", {"train": examples, "valid": examples, "test": examples})


def test_dataset_specs_match_table_3():
    assert (
        DATASETS["banking77"].sizes == (8622, 1540, 3080) and DATASETS["banking77"].num_labels == 77
    )
    assert (
        DATASETS["clinc150"].sizes == (15000, 3000, 4500) and DATASETS["clinc150"].num_labels == 150
    )
    assert DATASETS["hwu64"].sizes == (8954, 1076, 1076) and DATASETS["hwu64"].num_labels == 64


def test_synthetic_data_is_deterministic_and_keyword_based():
    data = make_synthetic_data(seed=3, num_groups=2, classes_per_group=3)
    again = make_synthetic_data(seed=3, num_groups=2, classes_per_group=3)
    assert data.train_pool == again.train_pool and data.test == again.test
    assert data.task.num_candidates == 6 and len(data.task.eligible_parents()) == 2
    assert Counter(e.label for e in data.train_pool) == dict.fromkeys(data.task.labels, 16)
    for example in data.test:
        keywords = set(data.task.nodes[example.label].description.split())
        assert len(keywords & set(example.text.split())) == 2
    assert not set(data.task.candidates) & set(data.task.labels)  # outputs are arbitrary words
