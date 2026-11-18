"""Reported result tables: the paper's tables and the supplementary experiments (E01-E09).

Two directories hold numbers reported by the authors, one CSV file per table:

* ``results/paper``          Tables 1, 2, 5, 8, 11, 12 and 13 of the paper (ids ``T01`` ...);
* ``results/supplementary``  the supplementary experiments ``E01`` ... ``E09``.

Each directory has an ``index.yaml`` that lists its tables and says how the CSV columns are
displayed (``mean ± sd``, ``value [low, high]``, ``a / b``).  This module

* loads an index and its CSV files (:func:`load_index`, :func:`load_paper_tables`, :func:`load`);
* renders tables as Markdown (:meth:`Table.to_markdown`, :func:`render_markdown`,
  :func:`embed_tables`), which is how ``docs/supplementary_results.md`` and the tables of the
  README are produced;
* checks that tables reporting the same quantity agree (:func:`check_consistency`), e.g. that
  the SIGMA rows of E05 equal Table 5 and that every interval contains its point estimate.

The numbers are never stored in code: edit a CSV file and re-run
``python scripts/render_supplementary.py``.

A directory is resolved in this order: an explicit ``root`` argument, the environment variable
(``SIGMA_SUPPLEMENTARY_DIR`` / ``SIGMA_PAPER_TABLES_DIR``), the folder of a source checkout.
"""

from __future__ import annotations

import csv
import os
import re
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path

import yaml

SUPPLEMENTARY, PAPER = "supplementary", "paper"
ENV_VARS = {SUPPLEMENTARY: "SIGMA_SUPPLEMENTARY_DIR", PAPER: "SIGMA_PAPER_TABLES_DIR"}
INDEX_NAME = "index.yaml"
_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]

MINUS = "−"
DATASETS = ("BANKING77", "CLINC150", "HWU64")
SHOTS = (1, 2, 4, 8, 16)

# Values are printed with two decimals.  The same quantity in two tables must agree to the
# printed precision; a difference of two rounded values can be off by three half-units.
SAME_VALUE_TOLERANCE = 0.005
DERIVED_TOLERANCE = 0.015


def tables_dir(kind: str = SUPPLEMENTARY, root: str | Path | None = None) -> Path:
    """Directory that holds ``index.yaml`` and the CSV files of ``kind``."""
    if root is not None:
        return Path(root)
    if os.environ.get(ENV_VARS[kind]):
        return Path(os.environ[ENV_VARS[kind]])
    return _REPOSITORY_ROOT / "results" / kind


def parse_number(cell: str | None) -> float | None:
    """Numeric value of a CSV cell (``'+1.8'`` -> 1.8); ``None`` for blank or text cells."""
    if cell is None:
        return None
    text = cell.strip().replace(MINUS, "-")
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _typeset(cell: str | None) -> str:
    """Display form of one cell: ``-`` for blanks, a typographic minus for negative numbers."""
    if cell is None or not cell.strip():
        return "-"
    text = cell.strip()
    return MINUS + text[1:] if re.fullmatch(r"-\d[\d.]*", text) else text


@dataclass(frozen=True)
class Column:
    """One displayed column, built from one to three CSV fields.

    ``kind`` is ``col`` (one field as written), ``mean_sd`` (``mean ± sd``), ``interval``
    (``value [low, high]``) or ``pair`` (``a / b``).  ``prefixes`` are put before the parts
    (``≈`` for approximate values) and ``best`` (``max`` / ``min``) marks the column whose best
    entry is set in bold.
    """

    header: str
    kind: str
    fields: tuple[str, ...]
    align: str = "right"
    prefixes: tuple[str, ...] = ()
    best: str | None = None

    @classmethod
    def from_spec(cls, spec: Mapping[str, object]) -> Column:
        header = str(spec["header"])
        align = str(spec.get("align", "right"))
        if align not in ("left", "right"):
            raise ValueError(f"column {header!r}: align must be 'left' or 'right'")
        best = spec.get("best")
        if best not in (None, "max", "min"):
            raise ValueError(f"column {header!r}: best must be 'max' or 'min'")
        if "col" in spec:
            kind, fields = "col", (str(spec["col"]),)
        elif "mean" in spec:
            kind, fields = "mean_sd", (str(spec["mean"]), str(spec["sd"]))
        elif "value" in spec:
            kind, fields = "interval", (str(spec["value"]), str(spec["low"]), str(spec["high"]))
        elif "pair" in spec:
            first, second = spec["pair"]  # type: ignore[misc]
            kind, fields = "pair", (str(first), str(second))
        else:
            raise ValueError(
                f"column {header!r}: expected one of col, mean/sd, value/low/high, pair"
            )
        prefixes = tuple(str(p) for p in spec.get("prefix", ()))  # type: ignore[union-attr]
        if prefixes and len(prefixes) != len(fields):
            raise ValueError(f"column {header!r}: one prefix per part is required")
        return cls(header, kind, fields, align, prefixes, best)  # type: ignore[arg-type]

    def render(self, row: Mapping[str, str]) -> str:
        cells = [_typeset(row.get(name)) for name in self.fields]
        if all(cell == "-" for cell in cells):
            return "-"
        if self.prefixes:
            cells = [prefix + cell for prefix, cell in zip(self.prefixes, cells, strict=True)]
        if self.kind == "col":
            return cells[0]
        if self.kind == "mean_sd":
            return f"{cells[0]} ± {cells[1]}"
        if self.kind == "interval":
            return f"{cells[0]} [{cells[1]}, {cells[2]}]"
        return f"{cells[0]} / {cells[1]}"


@dataclass(frozen=True)
class Group:
    """A folder of related experiments."""

    dir: str
    title: str
    summary: str = ""


@dataclass(frozen=True)
class Table:
    """One reported table: its description and the rows of its CSV file."""

    id: str
    title: str
    group: str
    relative_path: str
    path: Path
    columns: tuple[Column, ...]
    fieldnames: tuple[str, ...]
    rows: tuple[dict[str, str], ...]
    label: str | None = None
    extends: str | None = None
    question: str | None = None
    setup: tuple[str, ...] = ()
    findings: tuple[str, ...] = ()

    def number(self, row: Mapping[str, str], name: str) -> float | None:
        """Numeric value of ``name`` in ``row`` (``None`` if blank or not a number)."""
        if name not in self.fieldnames:
            raise KeyError(f"{self.id}: no column {name!r}")
        return parse_number(row.get(name))

    def values(self, name: str) -> list[float | None]:
        """The column ``name`` as numbers, in file order."""
        return [self.number(row, name) for row in self.rows]

    def find(self, **where: str) -> dict[str, str] | None:
        """The row whose text fields equal ``where``, or ``None`` if there is not exactly one."""
        matches = [
            row for row in self.rows if all(row.get(key) == value for key, value in where.items())
        ]
        return matches[0] if len(matches) == 1 else None

    def row(self, **where: str) -> dict[str, str]:
        """Like :meth:`find`, but raise ``KeyError`` when the row is missing or ambiguous."""
        found = self.find(**where)
        if found is None:
            raise KeyError(f"{self.id}: expected exactly one row with {where}")
        return found

    def _best_rows(self, column: Column) -> set[int]:
        values = self.values(column.fields[0])
        numbers = [value for value in values if value is not None]
        if column.best is None or not numbers:
            return set()
        target = max(numbers) if column.best == "max" else min(numbers)
        return {index for index, value in enumerate(values) if value == target}

    def to_markdown(self) -> str:
        """The table as GitHub-flavoured Markdown."""
        header = "| " + " | ".join(column.header for column in self.columns) + " |"
        rule = "|" + "|".join("---" if c.align == "left" else "---:" for c in self.columns) + "|"
        best = [self._best_rows(column) for column in self.columns]
        body = []
        for index, row in enumerate(self.rows):
            cells = [
                f"**{column.render(row)}**" if index in marked else column.render(row)
                for column, marked in zip(self.columns, best, strict=True)
            ]
            body.append("| " + " | ".join(cells) + " |")
        return "\n".join([header, rule, *body])


@dataclass(frozen=True)
class Index:
    """All tables of one directory together with the shared conventions."""

    root: Path
    conventions: str
    groups: tuple[Group, ...]
    tables: dict[str, Table]

    def __getitem__(self, table_id: str) -> Table:
        return self.tables[table_id]

    def __contains__(self, table_id: object) -> bool:
        return table_id in self.tables

    def __iter__(self) -> Iterator[Table]:
        return iter(self.tables.values())

    def __len__(self) -> int:
        return len(self.tables)

    def in_group(self, group: str) -> list[Table]:
        return [table for table in self if table.group == group]


def _read_csv(path: Path) -> tuple[tuple[str, ...], tuple[dict[str, str], ...]]:
    with open(path, encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = tuple(reader.fieldnames or ())
        rows = tuple({key: (value or "") for key, value in row.items()} for row in reader)
    return fieldnames, rows


def _squash(text: object) -> str:
    return " ".join(str(text).split())


def _as_tuple(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (_squash(value),)
    return tuple(_squash(item) for item in value)  # type: ignore[union-attr]


def load_index(root: str | Path | None = None, kind: str = SUPPLEMENTARY) -> Index:
    """Load ``index.yaml`` and every CSV file it lists.

    Raises ``ValueError`` if a table id is repeated, a listed file is missing, a display column
    refers to a field the CSV does not have, a CSV field is not displayed by any column, or a
    CSV file in the directory is not listed in the index.
    """
    directory = tables_dir(kind, root)
    spec = yaml.safe_load((directory / INDEX_NAME).read_text(encoding="utf-8"))
    groups = tuple(
        Group(str(g["dir"]), str(g["title"]), _squash(g.get("summary", "")))
        for g in spec.get("groups", [])
    )
    group_dirs = {group.dir for group in groups}

    tables: dict[str, Table] = {}
    for entry in spec["experiments"]:
        table_id = str(entry["id"])
        if table_id in tables:
            raise ValueError(f"{INDEX_NAME}: duplicate id {table_id}")
        relative = str(entry["file"])
        path = directory / relative
        if not path.is_file():
            raise ValueError(f"{table_id}: missing data file {relative}")
        parts = Path(relative).parts
        group = parts[0] if len(parts) > 1 else ""
        if group and group not in group_dirs:
            raise ValueError(f"{table_id}: folder {group!r} is not a listed group")
        fieldnames, rows = _read_csv(path)
        columns = tuple(Column.from_spec(column) for column in entry["columns"])
        displayed = {name for column in columns for name in column.fields}
        unknown = sorted(displayed - set(fieldnames))
        if unknown:
            raise ValueError(f"{table_id}: index refers to missing CSV fields {unknown}")
        hidden = sorted(set(fieldnames) - displayed)
        if hidden:
            raise ValueError(f"{table_id}: CSV fields not displayed by any column {hidden}")
        tables[table_id] = Table(
            id=table_id,
            title=_squash(entry["title"]),
            group=group,
            relative_path=relative,
            path=path,
            columns=columns,
            fieldnames=fieldnames,
            rows=rows,
            label=str(entry["label"]) if entry.get("label") else None,
            extends=str(entry["extends"]) if entry.get("extends") else None,
            question=_squash(entry["question"]) if entry.get("question") else None,
            setup=_as_tuple(entry.get("setup")),
            findings=_as_tuple(entry.get("findings")),
        )

    listed = {table.relative_path for table in tables.values()}
    on_disk = {path.relative_to(directory).as_posix() for path in directory.rglob("*.csv")}
    unlisted = sorted(on_disk - listed)
    if unlisted:
        raise ValueError(f"CSV files not listed in {INDEX_NAME}: {unlisted}")
    return Index(directory, _squash(spec["conventions"]), groups, tables)


def load_paper_tables(root: str | Path | None = None) -> Index:
    """The tables of the paper (``results/paper``)."""
    return load_index(root, PAPER)


def load(table_id: str, root: str | Path | None = None) -> Table:
    """Load one table by id: ``'E07'`` (supplementary) or ``'T05'`` (Table 5 of the paper)."""
    kind = PAPER if table_id.upper().startswith("T") else SUPPLEMENTARY
    return load_index(root, kind)[table_id.upper()]


# --------------------------------------------------------------------------- rendering
def render_markdown(
    index: Index | None = None, link_prefix: str = "../results/supplementary/"
) -> str:
    """The supplementary experiments as one Markdown document.

    ``link_prefix`` is the path from the document to the data directory.
    """
    index = index or load_index()
    first, last = next(iter(index)).id, list(index)[-1].id
    lines: list[str] = [
        f"# Supplementary experiments ({first}-{last})",
        "",
        "<!-- Generated by scripts/render_supplementary.py from results/supplementary/.",
        "     Edit the CSV files or index.yaml there, not this file. -->",
        "",
        "These tables are supplementary results reported by the authors of the paper. They",
        "extend the paper's tables with broader guidance edits, deeper compositions, cost and",
        "annotation accounting, a label sweep for the factorised variant and related-method",
        "controls. The numbers are stored as CSV files in",
        f"[`results/supplementary/`]({link_prefix}) and loaded with",
        "`sigma.supplementary`; the figures are in",
        "[`assets/results/`](../assets/results/README.md) and the commands and",
        "configurations for each experiment are listed in the",
        "[README](../README.md#supplementary-experiments).",
        "",
        index.conventions,
        "",
        "## Index",
        "",
        "| ID | Experiment | Extends | Data |",
        "|---|---|---|---|",
    ]
    for table in index:
        name = Path(table.relative_path).name
        lines.append(
            f"| [{table.id}](#{table.id.lower()}) | {table.title} "
            f"| {table.extends or '-'} | [`{name}`]({link_prefix}{table.relative_path}) |"
        )
    for group in index.groups:
        lines += ["", f"## {group.title}", ""]
        if group.summary:
            lines += [group.summary, ""]
        for table in index.in_group(group.dir):
            lines += [f'<a id="{table.id.lower()}"></a>', "", f"### {table.id}. {table.title}", ""]
            if table.extends:
                lines += [f"*Extends {table.extends} of the paper.*", ""]
            if table.question:
                lines += [f"**Question.** {table.question}", ""]
            if table.setup:
                lines += ["**Setup.**", "", *[f"- {item}" for item in table.setup], ""]
            lines += [table.to_markdown(), ""]
            if table.findings:
                lines += ["**Key findings.**", "", *[f"- {item}" for item in table.findings], ""]
            lines.append(f"Data: [`{table.relative_path}`]({link_prefix}{table.relative_path})")
    return "\n".join(lines).rstrip() + "\n"


EMBED = re.compile(r"(<!-- table:([ET]\d\d) -->\n)(?:.*?\n)??(<!-- /table:\2 -->)", re.DOTALL)


def embed_tables(text: str, *indexes: Index) -> str:
    """Refresh the tables a document embeds between ``<!-- table:E05 -->`` markers.

    Everything between ``<!-- table:E05 -->`` and ``<!-- /table:E05 -->`` is replaced by the
    current table with that id, so a README can show tables without keeping a second copy of
    the numbers.  Ids are looked up in ``indexes`` (default: supplementary and paper tables).
    """
    indexes = indexes or (load_index(), load_paper_tables())

    def table(match: re.Match[str]) -> str:
        table_id = match.group(2)
        for index in indexes:
            if table_id in index:
                return match.group(1) + index[table_id].to_markdown() + "\n" + match.group(3)
        raise KeyError(f"no table {table_id}")

    return EMBED.sub(table, text)


# --------------------------------------------------------------------------- consistency
class _Checker:
    """Collects human-readable descriptions of disagreements between tables."""

    def __init__(self, index: Index, paper: Index) -> None:
        self.index, self.paper = index, paper
        self.problems: list[str] = []

    def table(self, table_id: str) -> Table:
        return (self.paper if table_id.startswith("T") else self.index)[table_id]

    def value(self, table_id: str, name: str, **where: str) -> float | None:
        """Number in column ``name`` of the row selected by ``where`` (problems are recorded)."""
        table = self.table(table_id)
        row = table.find(**where)
        if row is None:
            self.problems.append(f"{table_id}: expected exactly one row with {where}")
            return None
        number = table.number(row, name)
        if number is None:
            self.problems.append(f"{table_id}: column {name} is blank or not a number")
        return number

    def close(self, left: float | None, right: float | None, tolerance: float, label: str) -> None:
        if left is None or right is None:
            return
        if abs(left - right) > tolerance + 1e-9:
            self.problems.append(f"{label}: {left:g} vs {right:g} (tolerance {tolerance:g})")

    def same(self, left: float | None, right: float | None, label: str) -> None:
        self.close(left, right, SAME_VALUE_TOLERANCE, label)


def _check_intervals(check: _Checker) -> None:
    """Every interval contains its point estimate and every SD is non-negative."""
    for table in (*check.index, *check.paper):
        for number, row in enumerate(table.rows, start=2):
            for column in table.columns:
                values = [parse_number(row.get(name)) for name in column.fields]
                where = f"{table.id} line {number}, {column.header}"
                if column.kind == "interval" and all(v is not None for v in values):
                    value, low, high = values
                    if not low <= value <= high:  # type: ignore[operator]
                        check.problems.append(f"{where}: {value} is outside [{low}, {high}]")
                elif column.kind == "interval" and any(v is not None for v in values):
                    check.problems.append(f"{where}: interval is partly blank")
                elif column.kind == "mean_sd" and values[1] is not None and values[1] < 0:
                    check.problems.append(f"{where}: negative standard deviation")


def _check_transfer(check: _Checker) -> None:
    """E01: SIGMA-F is unaffected by serialisation and binding edits, so it repeats Table 1."""
    for dataset in DATASETS:
        reference = check.value("T01", f"{dataset.lower()}_comp", method="SIGMA-F")
        for condition in (
            "Random serialization + compositions",
            "Cross-group binding compositions",
        ):
            check.same(
                check.value("E01", "sigma_f_comp", dataset=dataset, condition=condition),
                reference,
                f"E01 {dataset} SIGMA-F Comp. ({condition}) vs Table 1",
            )
        check.same(
            check.value(
                "E01",
                "sigma_f_changed",
                dataset=dataset,
                condition="Random serialization + compositions",
            ),
            check.value("T12", "changed_acc_mean", dataset=dataset, method="SIGMA-F"),
            f"E01 {dataset} SIGMA-F changed-target accuracy (random serialization) vs Table 12",
        )


def _check_cost(check: _Checker) -> None:
    """E03, E04 and E06 against each other and against Tables 1, 5, 8 and 11."""
    for dataset in DATASETS:
        label = f"E03 {dataset}"
        four = check.value("E03", "training_labels_4shot", dataset=dataset)
        sixteen = check.value("E03", "training_labels_16shot", dataset=dataset)
        if four is not None and sixteen is not None:
            check.close(sixteen, 4 * four, 0.0, f"{label}: 16-shot vs four times the 4-shot labels")
            check.close(
                check.value("E03", "train_dev_labels_4shot", dataset=dataset) - four,  # type: ignore[operator]
                check.value("E03", "train_dev_labels_16shot", dataset=dataset) - sixteen,  # type: ignore[operator]
                0.0,
                f"{label}: development labels at 4 and 16 shots",
            )
        sigma4 = check.value("E03", "sigma_4_comp", dataset=dataset)
        check.same(
            sigma4,
            check.value("E05", "shots_4_mean", dataset=dataset, method="SIGMA"),
            f"{label}: SIGMA-4 Comp. vs E05",
        )
        check.same(
            sigma4, check.value("E06", "sigma_4_comp", dataset=dataset), f"{label}: SIGMA-4 vs E06"
        )
        check.same(
            check.value("E03", "sigma_4_gpu_hours_approx", dataset=dataset),
            check.value("E06", "sigma_4_gpu_hours_approx", dataset=dataset),
            f"{label}: SIGMA-4 GPU-hours vs E06",
        )
        check.same(
            check.value("E03", "sft_16_comp", dataset=dataset),
            check.value("T05", "shots_16_mean", dataset=dataset, method="LoRA-SFT", view="Comp."),
            f"{label}: SFT-16 Comp. vs Table 5",
        )
        check.same(
            check.value("E06", "sigma_f_4_comp", dataset=dataset),
            check.value("E05", "shots_4_mean", dataset=dataset, method="SIGMA-F"),
            f"E06 {dataset}: SIGMA-F-4 Comp. vs E05",
        )

    # E04 (BANKING77): the 13.73 GPU-h cap is SIGMA's own cost, so finished runs repeat
    # Tables 1 and 8; its LoRA-SFT row is the token-matched control of Tables 8 and 11.
    table1 = {"Paired SFT": "Paired SFT (λ = 0)"}
    for method in ("Paired SFT", "Transported KL", "SIGMA", "SIGMA-F"):
        check.same(
            check.value("E04", "cap_13p73_comp", method=method),
            check.value("T01", "banking77_comp", method=table1.get(method, method)),
            f"E04 {method}: Comp. at the 13.73 GPU-h cap vs Table 1",
        )
        check.same(
            check.value("E04", "cap_13p73_actual_gpu_hours", method=method),
            check.value("T08", "gpu_hours", method=method),
            f"E04 {method}: actual GPU-hours vs Table 8",
        )
    check.same(
        check.value("E04", "cap_13p73_comp", method="LoRA-SFT"),
        check.value("T11", "comp_mean", dataset="BANKING77", method="SFT (token-matched)"),
        "E04 LoRA-SFT: Comp. at the 13.73 GPU-h cap vs token-matched SFT of Table 11",
    )
    check.same(
        check.value("E04", "cap_13p73_actual_gpu_hours", method="LoRA-SFT"),
        check.value("T08", "gpu_hours", method="SFT (token-matched)"),
        "E04 LoRA-SFT: actual GPU-hours vs token-matched SFT of Table 8",
    )
    check.same(
        check.value("E04", "cap_7p84_comp", method="LoRA-SFT"),
        check.value("T01", "banking77_comp", method="LoRA-SFT"),
        "E04 LoRA-SFT: Comp. at the 7.84 GPU-h cap (its own cost) vs Table 1",
    )


def _check_label_sweep(check: _Checker) -> None:
    """E05: the SIGMA rows are the Comp. rows of Table 5."""
    for dataset in DATASETS:
        for shots in SHOTS:
            for part in ("mean", "sd"):
                name = f"shots_{shots}_{part}"
                check.same(
                    check.value("E05", name, dataset=dataset, method="SIGMA"),
                    check.value("T05", name, dataset=dataset, method="SIGMA", view="Comp."),
                    f"E05 {dataset} SIGMA {name} vs Table 5",
                )


def _check_related_methods(check: _Checker) -> None:
    """E07-E09 against Tables 1, 12 and 13, and gains against the values they are computed from."""
    for dataset in DATASETS:
        key = dataset.lower()
        sigma = check.value("T01", f"{key}_comp", method="SIGMA")
        kl = check.value("T01", f"{key}_comp", method="Transported KL")
        paired = check.value("T01", f"{key}_comp", method="Paired SFT (λ = 0)")
        sigma_changed = check.value("T12", "changed_acc_mean", dataset=dataset, method="SIGMA")
        kl_changed = check.value(
            "T12", "changed_acc_mean", dataset=dataset, method="Transported KL"
        )

        check.same(
            check.value("E07", "kl_comp", dataset=dataset), kl, f"E07 {dataset} KL vs Table 1"
        )
        check.same(
            check.value("E07", "sigma_comp", dataset=dataset),
            sigma,
            f"E07 {dataset} SIGMA vs Table 1",
        )
        check.same(
            check.value("E07", "sigma_changed", dataset=dataset),
            sigma_changed,
            f"E07 {dataset} SIGMA changed-target accuracy vs Table 12",
        )

        with_alignment = check.value("E08", "aug_sigma", dataset=dataset)
        without = check.value("E08", "aug_paired_ce", dataset=dataset)
        if with_alignment is not None and without is not None:
            check.close(
                check.value("E08", "gain_pp", dataset=dataset),
                with_alignment - without,
                DERIVED_TOLERANCE,
                f"E08 {dataset}: gain vs difference of the two columns",
            )

        for name, comparator, reference in (
            ("gain_vs_paired_sft", "Paired SFT", paired),
            ("gain_vs_transported_kl", "Transported KL", kl),
        ):
            gain = check.value("E09", name, dataset=dataset)
            where = {"comparator": comparator, "quantity": "Comp., 10 vs. 10"}
            check.same(
                gain,
                check.value("T13", f"{key}_diff", **where),
                f"E09 {dataset} {name} vs Table 13",
            )
            for bound in ("ci_low", "ci_high"):
                check.same(
                    check.value("E09", f"{name}_{bound}", dataset=dataset),
                    check.value("T13", f"{key}_{bound}", **where),
                    f"E09 {dataset} {name}_{bound} vs Table 13",
                )
            if sigma is not None and reference is not None:
                check.close(
                    gain, sigma - reference, DERIVED_TOLERANCE, f"E09 {dataset} {name} vs Table 1"
                )
        if sigma_changed is not None and kl_changed is not None:
            check.close(
                check.value("E09", "gain_vs_kl_changed_pp", dataset=dataset),
                sigma_changed - kl_changed,
                DERIVED_TOLERANCE,
                f"E09 {dataset}: changed-target gain vs Table 12",
            )


CHECKS: tuple[Callable[[_Checker], None], ...] = (
    _check_intervals,
    _check_transfer,
    _check_cost,
    _check_label_sweep,
    _check_related_methods,
)


def check_consistency(index: Index | None = None, paper: Index | None = None) -> list[str]:
    """Disagreements between the tables, one description per problem (empty if none).

    The checks cover quantities that a supplementary table repeats from another one or from
    the paper (Tables 1, 5, 8, 11, 12, 13), gains that follow from two reported values, label
    counts, and the containment of every point estimate in its confidence interval.
    """
    check = _Checker(index or load_index(), paper or load_paper_tables())
    for function in CHECKS:
        try:
            function(check)
        except KeyError as error:
            check.problems.append(f"{function.__name__}: missing table or column {error}")
    return check.problems
