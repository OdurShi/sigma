"""The reported tables: loading, rendering, cross-table consistency and the README.

The tests never compare against literal results.  They check the structure of the two
collections, that tables reporting the same quantity agree, that the rendered documents are
current, and that the checks notice an edited value.
"""

from __future__ import annotations

import csv
import importlib.util
import re
import shutil
from pathlib import Path

import pytest

from sigma import supplementary
from sigma.data.datasets import DATASETS as DATASET_SPECS
from sigma.supplementary import (
    DATASETS,
    Column,
    check_consistency,
    embed_tables,
    load,
    load_index,
    load_paper_tables,
    parse_number,
    render_markdown,
    tables_dir,
)

REPOSITORY = Path(__file__).resolve().parents[1]
DOCUMENT = REPOSITORY / "docs" / "supplementary_results.md"
README = REPOSITORY / "README.md"
FORBIDDEN = ("openreview", "rebuttal", "reviewer", "submission")


@pytest.fixture(scope="module")
def index():
    return load_index()


@pytest.fixture(scope="module")
def paper():
    return load_paper_tables()


@pytest.fixture
def copy(tmp_path):
    """An editable copy of the supplementary data directory."""
    target = tmp_path / "supplementary"
    shutil.copytree(tables_dir(), target)
    return target


def _rewrite(path: Path, edit) -> None:
    """Apply ``edit(rows)`` to the rows of a CSV file and write it back."""
    with open(path, encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames, rows = list(reader.fieldnames), list(reader)
    edit(rows)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, REPOSITORY / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_indexes_list_the_tables_in_order(index, paper):
    assert list(index.tables) == [f"E{n:02d}" for n in range(1, 10)]
    assert [group.dir for group in index.groups] == sorted(group.dir for group in index.groups)
    for table in index:
        assert table.rows and table.title and table.setup and table.findings, table.id
        assert table.relative_path.startswith(table.group + "/")
    assert list(paper.tables) == ["T01", "T02", "T05", "T08", "T11", "T12", "T13"]
    assert all(table.label == f"Table {int(table.id[1:])}" for table in paper)


def test_cells_are_numbers_or_text_labels(index, paper):
    for table in (*index, *paper):
        displayed = [name for column in table.columns for name in column.fields]
        assert sorted(displayed) == sorted(table.fieldnames), table.id
        for row in table.rows:
            for column in table.columns:
                if column.kind != "col":
                    assert all(parse_number(row[name]) is not None for name in column.fields)


def test_tables_cover_the_three_datasets(index):
    for table in index:
        if "dataset" in table.fieldnames:
            assert set(row["dataset"] for row in table.rows) == set(DATASETS), table.id


def test_label_counts_follow_the_shipped_datasets(index):
    """E03: training labels are shots x intents, development labels the development split."""
    table = index["E03"]
    for name, spec in DATASET_SPECS.items():
        row = table.row(dataset=name.upper())
        assert table.number(row, "training_labels_4shot") == 4 * spec.num_labels
        assert table.number(row, "training_labels_16shot") == 16 * spec.num_labels
        development = table.number(row, "train_dev_labels_4shot") - 4 * spec.num_labels
        assert development == spec.sizes[1]


def test_tables_agree_with_each_other_and_with_the_paper(index, paper):
    assert check_consistency(index, paper) == []


def test_documents_are_current(index, paper):
    assert DOCUMENT.read_text(encoding="utf-8") == render_markdown(index)
    assert _load_script("render_supplementary").main(["--check"]) == 0
    text = README.read_text(encoding="utf-8")
    embedded = re.findall(r"<!-- table:([ET]\d\d) -->", text)
    assert {"T01", "E05"} <= set(embedded)
    assert embed_tables(text, index, paper) == text


def test_figures_are_drawn_from_the_tables(tmp_path):
    pytest.importorskip("matplotlib")
    module = _load_script("plot_results")
    written = module.draw(tmp_path, dpi=50)
    assert [path.stem for path in written] == [name for name, _ in module.FIGURES.values()]
    assert all(path.stat().st_size > 2_000 for path in written)
    assert all((REPOSITORY / "assets" / "results" / path.name).is_file() for path in written)


def test_edited_values_are_reported(copy, paper):
    def sweep(rows):
        rows[0]["shots_4_mean"] = f"{float(rows[0]['shots_4_mean']) + 1:.2f}"

    _rewrite(copy / "03_sigma_vs_sigma_f" / "E05_sigma_vs_sigma_f_label_sweep.csv", sweep)
    problems = check_consistency(load_index(copy), paper)
    assert any("vs Table 5" in problem for problem in problems)
    assert any("SIGMA-4 Comp. vs E05" in problem for problem in problems)


def test_interval_that_misses_its_estimate_is_reported(copy, paper):
    def edit(rows):
        rows[0]["gain_vs_paired_sft_ci_high"] = rows[0]["gain_vs_paired_sft_ci_low"]

    _rewrite(copy / "04_related_method_baselines" / "E09_paired_gains_with_ci.csv", edit)
    problems = check_consistency(load_index(copy), paper)
    assert any(problem.startswith("E09 line 2") and "outside" in problem for problem in problems)


def test_malformed_collections_are_rejected(copy):
    extra = copy / "04_related_method_baselines" / "E99_extra.csv"
    extra.write_text("a,b\n1,2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="not listed"):
        load_index(copy)
    extra.unlink()
    path = copy / "04_related_method_baselines" / "E08_symbol_tuning_style_augmentation.csv"
    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text("\n".join([lines[0] + ",extra"] + [line + ",1" for line in lines[1:]]) + "\n")
    with pytest.raises(ValueError, match="not displayed"):
        load_index(copy)


def test_environment_variable_selects_the_directory(copy, monkeypatch):
    monkeypatch.setenv(supplementary.ENV_VARS[supplementary.SUPPLEMENTARY], str(copy))
    assert tables_dir() == copy
    assert load("E07").path.is_relative_to(copy)
    assert load("T01").label == "Table 1"


@pytest.mark.parametrize(
    ("spec", "row", "expected"),
    [
        ({"header": "h", "col": "a"}, {"a": ""}, "-"),
        ({"header": "h", "mean": "m", "sd": "s"}, {"m": "1.0", "s": "0.25"}, "1.0 ± 0.25"),
        (
            {"header": "h", "value": "v", "low": "l", "high": "u"},
            {"v": "-0.4", "l": "-1.2", "u": "+0.4"},
            "−0.4 [−1.2, +0.4]",
        ),
        (
            {"header": "h", "pair": ["a", "b"], "prefix": ["", "≈"]},
            {"a": "1.5", "b": "2.5"},
            "1.5 / ≈2.5",
        ),
    ],
)
def test_column_rendering(spec, row, expected):
    assert Column.from_spec(spec).render(row) == expected


def test_best_entries_are_bold(paper):
    lines = paper["T02"].to_markdown().splitlines()
    assert "**0.0000**" in lines[-1] and "**100.00**" in lines[-1]
    assert "**" not in lines[2]


def test_embedded_tables_are_refreshed(index, paper):
    table = index["E08"].to_markdown()
    text = "before\n<!-- table:E08 -->\nstale\n<!-- /table:E08 -->\nafter\n"
    refreshed = embed_tables(text, index, paper)
    assert refreshed == text.replace("stale", table)
    assert embed_tables(refreshed, index, paper) == refreshed


def _documents():
    yield README
    yield from (REPOSITORY / "docs").glob("*.md")
    for folder in ("results/paper", "results/supplementary", "assets/results"):
        yield REPOSITORY / folder / "README.md"


@pytest.mark.parametrize(
    "path", list(_documents()), ids=lambda p: p.relative_to(REPOSITORY).as_posix()
)
def test_documents_link_to_existing_files_and_stay_anonymous(path):
    text = path.read_text(encoding="utf-8")
    targets = re.findall(r"\]\(([^)#\s]+)(?:#[^)]*)?\)", text) + re.findall(r'src="([^"]+)"', text)
    for target in targets:
        if not target.startswith(("http://", "https://")):
            assert (path.parent / target).exists(), f"{path.name}: broken link {target}"
    lowered = text.lower()
    assert not [word for word in FORBIDDEN if word in lowered]


def test_readme_highlights_quote_the_reported_tables(index, paper):
    """Every number of the Highlights and Main results sections is a value of a shipped table."""
    text = README.read_text(encoding="utf-8")
    section = text[text.index("## Highlights") : text.index("## Method")]
    section = re.sub(r"<!-- table:.*?<!-- /table:\w+ -->", "", section, flags=re.DOTALL)
    reported = {
        f"{abs(value):.2f}"
        for table in (*index, *paper)
        for row in table.rows
        for value in (parse_number(cell) for cell in row.values())
        if value is not None
    }
    quoted = re.findall(r"\d+\.\d\d", section)
    assert quoted and not [number for number in quoted if number not in reported]
