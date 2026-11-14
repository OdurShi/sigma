"""Task records and structured guidance (Section 4.1, Appendix A.1).

A *task record* stores a rooted tree whose leaves are the semantic labels ``Z`` and whose
internal nodes are semantic groups, one immutable identifier and one semantic description per
node, and a bijection ``a : Z -> C`` that binds every label to an output string.  *Guidance*
``g`` is one concrete version of that specification: the same tree and descriptions together
with a child order for every internal node and a (possibly edited) binding map ``a_g``.

Two orders are fixed once per task and never change (Appendix A.1): the *candidate order* (the
lexicographic order of the output strings, used for every score vector) and the *semantic-label
order* (the lexicographic order of the leaf identifiers).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from functools import cached_property

DEFAULT_INSTRUCTION = (
    "You are given structured guidance: a tree of groups and semantic classes. Every class "
    "has a description and is bound to exactly one output string. Read the input, decide "
    "which semantic class it belongs to, and answer with the output string that the guidance "
    "currently binds to that class. Follow the bindings written in the guidance even if they "
    "differ from what the class name suggests. The order in which sibling entries are listed "
    "does not denote priority. Answer with exactly one of the candidate outputs."
)


class TaskRecordError(ValueError):
    """Raised when a task record is malformed (Appendix A.1, "Validation")."""


@dataclass(frozen=True)
class Node:
    """A node of the task tree: a semantic group (internal) or a semantic label (leaf)."""

    identifier: str
    description: str
    children: tuple[str, ...] = ()

    @property
    def is_leaf(self) -> bool:
        return len(self.children) == 0


@dataclass(frozen=True)
class TaskRecord:
    """Immutable task schema: tree, descriptions and base binding map ``a_{g0}``.

    Parameters
    ----------
    name:
        Task name (used for logging only).
    root:
        Identifier of the root node.
    nodes:
        All nodes by identifier.  ``Node.children`` lists child identifiers; their stored
        order is irrelevant, the base guidance always uses ascending identifier order.
    binding:
        Base binding ``a_{g0}``: leaf identifier -> output string (a bijection onto ``C``).
    instruction:
        Task instruction placed at the start of every prompt.
    """

    name: str
    root: str
    nodes: Mapping[str, Node]
    binding: Mapping[str, str]
    instruction: str = DEFAULT_INSTRUCTION
    _parents: dict[str, str] = field(default_factory=dict, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        self._parents.update(_validate(self))

    # ------------------------------------------------------------------ fixed orders
    @cached_property
    def labels(self) -> tuple[str, ...]:
        """Semantic labels ``Z`` (leaf identifiers) in the fixed semantic-label order."""
        return tuple(sorted(n.identifier for n in self.nodes.values() if n.is_leaf))

    @cached_property
    def candidates(self) -> tuple[str, ...]:
        """Output inventory ``C`` in the fixed (lexicographic) candidate order."""
        return tuple(sorted(self.binding.values()))

    @cached_property
    def candidate_index(self) -> dict[str, int]:
        return {c: i for i, c in enumerate(self.candidates)}

    @cached_property
    def label_index(self) -> dict[str, int]:
        return {z: i for i, z in enumerate(self.labels)}

    @property
    def num_candidates(self) -> int:
        """``m = |C|``."""
        return len(self.candidates)

    # ------------------------------------------------------------------ tree helpers
    def parent_of(self, identifier: str) -> str | None:
        """Parent identifier, or ``None`` for the root."""
        return self._parents.get(identifier)

    def internal_nodes(self) -> tuple[str, ...]:
        """Identifiers of all internal nodes in ascending order."""
        return tuple(sorted(n.identifier for n in self.nodes.values() if not n.is_leaf))

    def leaf_children(self, identifier: str) -> tuple[str, ...]:
        """``L(v)``: the leaf children of node ``v`` in ascending identifier order."""
        node = self.nodes[identifier]
        return tuple(sorted(c for c in node.children if self.nodes[c].is_leaf))

    def eligible_parents(self) -> tuple[str, ...]:
        """``V``: non-root nodes with at least two leaf children (Section 4.1), ascending."""
        return tuple(
            v for v in self.internal_nodes() if v != self.root and len(self.leaf_children(v)) >= 2
        )

    def sibling_groups(self) -> tuple[tuple[str, ...], ...]:
        """Leaf sets ``L(v)`` of the eligible parents, in the order of :meth:`eligible_parents`."""
        return tuple(self.leaf_children(v) for v in self.eligible_parents())

    def base_guidance(self) -> Guidance:
        """Base guidance ``g0``: ascending child-identifier order and the base binding."""
        order = {v: tuple(sorted(self.nodes[v].children)) for v in self.internal_nodes()}
        return Guidance(task=self, child_order=order, binding=dict(self.binding))


def _validate(task: TaskRecord) -> dict[str, str]:
    """Check tree shape, descriptions and binding; return the child -> parent map."""
    nodes = task.nodes
    if task.root not in nodes:
        raise TaskRecordError(f"root {task.root!r} is not a node")
    for key, node in nodes.items():
        if key != node.identifier:
            raise TaskRecordError(f"node key {key!r} does not match identifier {node.identifier!r}")
        if not node.identifier:
            raise TaskRecordError("empty node identifier")
        if not node.description or not node.description.strip():
            raise TaskRecordError(f"node {key!r} has no semantic description")
        if len(set(node.children)) != len(node.children):
            raise TaskRecordError(f"node {key!r} lists a child twice")
    parents: dict[str, str] = {}
    for node in nodes.values():
        for child in node.children:
            if child not in nodes:
                raise TaskRecordError(f"node {node.identifier!r} has unknown child {child!r}")
            if child == task.root:
                raise TaskRecordError("the root cannot be a child")
            if child in parents:
                raise TaskRecordError(f"node {child!r} has more than one parent")
            parents[child] = node.identifier
    # Every non-root node must be reachable from the root (rules out cycles and orphans).
    seen: set[str] = set()
    stack = [task.root]
    while stack:
        current = stack.pop()
        if current in seen:
            raise TaskRecordError("the node graph contains a cycle")
        seen.add(current)
        stack.extend(nodes[current].children)
    if seen != set(nodes):
        raise TaskRecordError(f"nodes not reachable from the root: {sorted(set(nodes) - seen)}")
    if nodes[task.root].is_leaf:
        raise TaskRecordError("the root must be an internal node")
    leaves = {k for k, n in nodes.items() if n.is_leaf}
    if set(task.binding) != leaves:
        raise TaskRecordError("the binding map must be defined exactly on the leaves")
    outputs = list(task.binding.values())
    if any((not isinstance(c, str)) or not c.strip() for c in outputs):
        raise TaskRecordError("output strings must be non-empty")
    if len(set(outputs)) != len(outputs):
        raise TaskRecordError("the binding map is not a bijection (duplicate output strings)")
    if len(outputs) < 2:
        raise TaskRecordError("a task needs at least two candidate outputs (m >= 2)")
    return parents


@dataclass(frozen=True)
class Guidance:
    """One version of the structured guidance: child order and binding map ``a_g``.

    The tree, identifiers and descriptions are those of ``task``; only the serialisation order
    of siblings and the outputs bound to the leaves can differ between versions.
    """

    task: TaskRecord
    child_order: Mapping[str, tuple[str, ...]]
    binding: Mapping[str, str]

    def output_of(self, label: str) -> str:
        """``a_g(z)``: the output string currently bound to semantic label ``z``."""
        return self.binding[label]

    def label_of(self, output: str) -> str:
        """``a_g^{-1}(c)``: the semantic label currently bound to output ``c``."""
        return self._inverse[output]

    @cached_property
    def _inverse(self) -> dict[str, str]:
        return {c: z for z, c in self.binding.items()}

    def target_index(self, label: str) -> int:
        """Index of ``a_g(z)`` in the fixed candidate order."""
        return self.task.candidate_index[self.binding[label]]

    def label_of_candidate(self) -> tuple[int, ...]:
        """For every candidate index ``c``, the semantic-label index of ``a_g^{-1}(c)``."""
        task = self.task
        return tuple(task.label_index[self._inverse[c]] for c in task.candidates)

    def key(self) -> tuple:
        """Hashable identity of the document (child orders and bindings)."""
        order = tuple((v, tuple(self.child_order[v])) for v in sorted(self.child_order))
        binding = tuple(self.binding[z] for z in self.task.labels)
        return order, binding


def flat_task(
    name: str,
    label_descriptions: Mapping[str, str],
    binding: Mapping[str, str] | None = None,
    root_description: str = "All classes of the task.",
) -> TaskRecord:
    """Build a flat task (root with leaf children only); binding steps then use the identity."""
    nodes = {z: Node(z, d) for z, d in label_descriptions.items()}
    nodes["root"] = Node("root", root_description, tuple(sorted(label_descriptions)))
    bind = dict(binding) if binding is not None else {z: z for z in label_descriptions}
    return TaskRecord(name=name, root="root", nodes=nodes, binding=bind)


def grouped_task(
    name: str,
    groups: Mapping[str, tuple[str, Mapping[str, str]]],
    binding: Mapping[str, str] | None = None,
    root_description: str = "All classes of the task.",
    instruction: str = DEFAULT_INSTRUCTION,
) -> TaskRecord:
    """Build a root -> group -> label tree (the shape used for all benchmarks, Appendix C.2).

    ``groups`` maps a group identifier to ``(group description, {label: label description})``.
    Group identifiers must not collide with label identifiers.
    """
    nodes: dict[str, Node] = {}
    for _, members in groups.values():
        for label, description in members.items():
            if label in nodes:
                raise TaskRecordError(f"label {label!r} appears in more than one group")
            nodes[label] = Node(label, description)
    for group_id, (group_description, members) in groups.items():
        if group_id in nodes or group_id == "root":
            raise TaskRecordError(f"group identifier {group_id!r} collides with another node")
        nodes[group_id] = Node(group_id, group_description, tuple(sorted(members)))
    nodes["root"] = Node("root", root_description, tuple(sorted(groups)))
    leaves = [z for z, n in nodes.items() if n.is_leaf]
    bind = dict(binding) if binding is not None else {z: z for z in leaves}
    return TaskRecord(name=name, root="root", nodes=nodes, binding=bind, instruction=instruction)


def task_to_dict(task: TaskRecord) -> dict:
    """JSON-serialisable form of a task record (used by the data-preparation scripts)."""
    return {
        "name": task.name,
        "root": task.root,
        "instruction": task.instruction,
        "nodes": [
            {
                "identifier": node.identifier,
                "description": node.description,
                "children": sorted(node.children),
            }
            for node in sorted(task.nodes.values(), key=lambda n: n.identifier)
        ],
        "binding": {label: task.binding[label] for label in task.labels},
    }


def task_from_dict(payload: Mapping) -> TaskRecord:
    """Inverse of :func:`task_to_dict`; malformed records are rejected."""
    nodes = {
        item["identifier"]: Node(item["identifier"], item["description"], tuple(item["children"]))
        for item in payload["nodes"]
    }
    return TaskRecord(
        name=payload["name"],
        root=payload["root"],
        nodes=nodes,
        binding=dict(payload["binding"]),
        instruction=payload.get("instruction", DEFAULT_INSTRUCTION),
    )


def with_binding(
    task: TaskRecord, binding: Mapping[str, str], name: str | None = None
) -> TaskRecord:
    """Copy of ``task`` with a different base binding (e.g. opaque output identifiers)."""
    return TaskRecord(
        name=name or task.name,
        root=task.root,
        nodes=task.nodes,
        binding=dict(binding),
        instruction=task.instruction,
    )
