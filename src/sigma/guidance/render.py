"""Serialisation of guidance documents and prompts (Appendix A.1, "Task record and rendering").

Guidance is serialised as a nested list.  Every internal-node record carries its identifier,
description and child records; every leaf record carries its identifier, semantic description
and designated output string.  Fields are written as JSON string literals, which makes the
records unambiguously delimited, so that every constructed document can be parsed back into its
tree, descriptions and binding map for validation (Appendix A.1, "Validation").

Example::

    - group "root": "All classes of the task."
      - group "billing": "Payments and refunds."
        - class "refund": "The customer wants money back." => "Refund"
        - class "invoice": "The customer asks for an invoice." => "Billing"
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from sigma.guidance.schema import Guidance, TaskRecord

INDENT = "  "
_STRING = r'"(?:[^"\\]|\\.)*"'
_LINE = re.compile(
    rf"^(?P<indent>(?:{INDENT})*)- (?P<kind>group|class) (?P<id>{_STRING}): "
    rf"(?P<description>{_STRING})(?: => (?P<output>{_STRING}))?$"
)


class GuidanceParseError(ValueError):
    """Raised when a guidance document cannot be parsed back into a task structure."""


@dataclass(frozen=True)
class ParsedGuidance:
    """Structure recovered from a rendered guidance document."""

    root: str
    descriptions: dict[str, str]
    child_order: dict[str, tuple[str, ...]]
    binding: dict[str, str | None]


def _quote(text: str) -> str:
    return json.dumps(text, ensure_ascii=False)


def render_guidance(guidance: Guidance, include_outputs: bool = True) -> str:
    """Serialise ``guidance`` as a nested list (one record per line, two spaces per level).

    ``include_outputs=False`` renders the *canonical* representation used by the factorised
    variant SIGMA-F (Appendix A.3), which shows identifiers and descriptions but no bindings.
    """
    task = guidance.task
    lines: list[str] = []

    def visit(identifier: str, depth: int) -> None:
        node = task.nodes[identifier]
        head = f"{INDENT * depth}- {'class' if node.is_leaf else 'group'} "
        head += f"{_quote(identifier)}: {_quote(node.description)}"
        if node.is_leaf:
            if include_outputs:
                head += f" => {_quote(guidance.binding[identifier])}"
            lines.append(head)
            return
        lines.append(head)
        for child in guidance.child_order[identifier]:
            visit(child, depth + 1)

    visit(task.root, 0)
    return "\n".join(lines)


def parse_guidance(document: str) -> ParsedGuidance:
    """Parse a document produced by :func:`render_guidance` back into its structure."""
    descriptions: dict[str, str] = {}
    children: dict[str, list[str]] = {}
    binding: dict[str, str | None] = {}
    stack: list[str] = []  # identifiers of the open internal nodes, one per depth
    root: str | None = None
    for number, line in enumerate(document.split("\n"), start=1):
        match = _LINE.match(line)
        if match is None:
            raise GuidanceParseError(f"line {number} is not a guidance record: {line!r}")
        depth = len(match["indent"]) // len(INDENT)
        identifier = json.loads(match["id"])
        if identifier in descriptions:
            raise GuidanceParseError(f"duplicate identifier {identifier!r} (line {number})")
        if depth == 0:
            if root is not None:
                raise GuidanceParseError("more than one root record")
            root = identifier
        elif depth > len(stack):
            raise GuidanceParseError(f"line {number} is indented below a leaf or skips a level")
        del stack[depth:]
        descriptions[identifier] = json.loads(match["description"])
        if depth > 0:
            children[stack[-1]].append(identifier)
        if match["kind"] == "group":
            if match["output"] is not None:
                raise GuidanceParseError(f"group {identifier!r} carries an output (line {number})")
            children[identifier] = []
            stack.append(identifier)
        else:
            binding[identifier] = None if match["output"] is None else json.loads(match["output"])
    if root is None:
        raise GuidanceParseError("empty guidance document")
    if root in binding:
        raise GuidanceParseError("the root record must be a group")
    for identifier, kids in children.items():
        if not kids:
            raise GuidanceParseError(f"group {identifier!r} has no child records")
    return ParsedGuidance(
        root=root,
        descriptions=descriptions,
        child_order={k: tuple(v) for k, v in children.items()},
        binding=binding,
    )


def same_structure(parsed: ParsedGuidance, task: TaskRecord) -> bool:
    """True iff ``parsed`` has the semantic tree and descriptions of ``task``.

    Sibling order is ignored (it is a property of the document, not of the task).
    """
    if parsed.root != task.root or set(parsed.descriptions) != set(task.nodes):
        return False
    for identifier, node in task.nodes.items():
        if parsed.descriptions[identifier] != node.description:
            return False
        if node.is_leaf:
            if identifier not in parsed.binding:
                return False
        elif set(parsed.child_order.get(identifier, ())) != set(node.children):
            return False
    return True


@dataclass(frozen=True)
class PromptFields:
    """The four prompt fields, in the fixed order of Appendix A.1.

    The instance label is never part of a prompt.
    """

    instruction: str
    guidance: str
    input_text: str
    candidates: tuple[str, ...]

    def as_text(self) -> str:
        """Concatenate instruction, guidance, input and candidate list into one string."""
        listing = "\n".join(f"- {c}" for c in self.candidates)
        return (
            f"{self.instruction}\n\n"
            f"Guidance:\n{self.guidance}\n\n"
            f"Input:\n{self.input_text}\n\n"
            f"Candidate outputs:\n{listing}"
        )


def render_prompt(
    guidance: Guidance,
    input_text: str,
    include_outputs: bool = True,
    candidates: tuple[str, ...] | None = None,
) -> PromptFields:
    """``Render(x, C, g)`` of Eq. (3): instruction, complete guidance, input, candidate list.

    The candidate list is always the full inventory in the fixed candidate order; it does not
    depend on the guidance version.
    """
    task = guidance.task
    return PromptFields(
        instruction=task.instruction,
        guidance=render_guidance(guidance, include_outputs=include_outputs),
        input_text=input_text,
        candidates=task.candidates if candidates is None else candidates,
    )
