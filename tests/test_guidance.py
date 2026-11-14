"""Task records, guidance rendering and parsing (Section 4.1, Appendix A.1)."""

from __future__ import annotations

import pytest

from sigma.guidance import Node, TaskRecord, TaskRecordError, flat_task, grouped_task
from sigma.guidance.render import (
    GuidanceParseError,
    parse_guidance,
    render_guidance,
    render_prompt,
    same_structure,
)
from sigma.guidance.schema import task_from_dict, task_to_dict


def _nodes(**overrides):
    nodes = {
        "root": Node("root", "all", ("g",)),
        "g": Node("g", "group", ("a", "b")),
        "a": Node("a", "first"),
        "b": Node("b", "second"),
    }
    nodes.update(overrides)
    return nodes


def test_fixed_orders_are_lexicographic(toy_task):
    assert toy_task.labels == ("charge", "closure", "invoice", "login", "other", "refund")
    assert toy_task.candidates == tuple(sorted(toy_task.binding.values()))
    assert toy_task.num_candidates == 6
    base = toy_task.base_guidance()
    assert base.child_order["root"] == ("account", "billing", "misc")
    assert base.child_order["billing"] == ("charge", "invoice", "refund")


def test_eligible_parents_exclude_root_and_small_groups(toy_task):
    # V: non-root nodes with at least two leaf children; "misc" has a single leaf.
    assert toy_task.eligible_parents() == ("account", "billing")
    assert toy_task.sibling_groups() == (("closure", "login"), ("charge", "invoice", "refund"))
    flat = flat_task("flat", {"a": "first", "b": "second", "c": "third"})
    assert flat.eligible_parents() == ()


def test_binding_lookup_and_inverse(toy_task):
    base = toy_task.base_guidance()
    for label in toy_task.labels:
        output = base.output_of(label)
        assert base.label_of(output) == label
        assert toy_task.candidates[base.target_index(label)] == output
    lookup = base.label_of_candidate()
    assert [toy_task.labels[i] for i in lookup] == [base.label_of(c) for c in toy_task.candidates]


@pytest.mark.parametrize(
    ("nodes", "binding", "message"),
    [
        (_nodes(), {"a": "x", "b": "x"}, "bijection"),
        (_nodes(), {"a": "x"}, "exactly on the leaves"),
        (_nodes(), {"a": "x", "b": " "}, "non-empty"),
        (_nodes(a=Node("a", "")), {"a": "x", "b": "y"}, "description"),
        (_nodes(g=Node("g", "group", ("a", "b", "zzz"))), {"a": "x", "b": "y"}, "unknown child"),
        (
            _nodes(root=Node("root", "all", ("g", "a"))),
            {"a": "x", "b": "y"},
            "more than one parent",
        ),
        (_nodes(c=Node("c", "orphan")), {"a": "x", "b": "y", "c": "z"}, "not reachable"),
        (
            {"root": Node("root", "all", ("a",)), "a": Node("a", "only")},
            {"a": "x"},
            "at least two",
        ),
    ],
)
def test_malformed_task_records_are_rejected(nodes, binding, message):
    with pytest.raises(TaskRecordError, match=message):
        TaskRecord("bad", "root", nodes, binding)


def test_render_parse_round_trip(toy_task):
    base = toy_task.base_guidance()
    document = render_guidance(base)
    parsed = parse_guidance(document)
    assert same_structure(parsed, toy_task)
    assert parsed.binding == dict(base.binding)
    assert parsed.child_order == dict(base.child_order)
    lines = document.split("\n")
    assert lines[0] == '- group "root": "All classes of the task."'
    assert '    - class "refund": "wants money back" => "desk red"' in lines
    assert len(lines) == len(toy_task.nodes)


def test_records_are_unambiguously_delimited():
    tricky = grouped_task(
        "tricky",
        {
            "g": (
                'a "quoted" group: with => arrows',
                {
                    "a": 'says "x" => "y"\nsecond line \\ backslash',
                    "b": "unicode café ✓",
                },
            )
        },
        binding={"a": 'out "1"', "b": "out => 2"},
    )
    base = tricky.base_guidance()
    parsed = parse_guidance(render_guidance(base))
    assert same_structure(parsed, tricky)
    assert parsed.binding == dict(base.binding)


@pytest.mark.parametrize(
    "document",
    [
        "",
        "not a record",
        '- class "a": "leaf root" => "x"',
        '- group "root": "r"',
        '- group "root": "r"\n    - class "a": "skips a level" => "x"',
        '- group "root": "r"\n  - class "a": "d" => "x"\n  - class "a": "d" => "y"',
        '- group "root": "r"\n  - class "a": "d" => "x"\n- group "second": "root"',
    ],
)
def test_parser_rejects_malformed_documents(document):
    with pytest.raises(GuidanceParseError):
        parse_guidance(document)


def test_prompt_has_four_fields_in_fixed_order_and_no_label(toy_task):
    base = toy_task.base_guidance()
    fields = render_prompt(base, "please give my money back")
    text = fields.as_text()
    positions = [
        text.index(toy_task.instruction),
        text.index("Guidance:"),
        text.index("Input:"),
        text.index("Candidate outputs:"),
    ]
    assert positions == sorted(positions)
    assert fields.candidates == toy_task.candidates
    # The candidate list is the full inventory in fixed order, whatever the guidance version.
    listing = text.split("Candidate outputs:\n")[1].split("\n")
    assert listing == [f"- {c}" for c in toy_task.candidates]


def test_canonical_rendering_hides_bindings(toy_task):
    document = render_guidance(toy_task.base_guidance(), include_outputs=False)
    assert "=>" not in document
    assert all(output not in document for output in toy_task.candidates)


def test_task_serialisation_round_trip(toy_task):
    restored = task_from_dict(task_to_dict(toy_task))
    assert restored.labels == toy_task.labels
    assert restored.candidates == toy_task.candidates
    assert render_guidance(restored.base_guidance()) == render_guidance(toy_task.base_guidance())
