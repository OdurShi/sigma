#!/usr/bin/env python
"""Render the reported tables (paper tables and supplementary experiments) as Markdown.

The numbers live in ``results/paper/*.csv`` and ``results/supplementary/*/*.csv``; their
layout is described by the ``index.yaml`` next to them.  This script writes
``docs/supplementary_results.md`` and refreshes the tables that ``README.md`` embeds between
``<!-- table:E05 -->`` markers.

    python scripts/render_supplementary.py            # rewrite the documents
    python scripts/render_supplementary.py --check    # verify they are current and consistent
    python scripts/render_supplementary.py --show E07 # print one table (T01 = Table 1)

``--check`` exits with status 1 if a document differs from what the CSV files produce or if
tables that report the same quantity disagree (see ``sigma.supplementary``).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from sigma.supplementary import (
    check_consistency,
    embed_tables,
    load_index,
    load_paper_tables,
    render_markdown,
)

REPOSITORY = Path(__file__).resolve().parents[1]
DOCUMENT = REPOSITORY / "docs" / "supplementary_results.md"
README = REPOSITORY / "README.md"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, default=DOCUMENT, help="Markdown file to write")
    parser.add_argument("--check", action="store_true", help="verify instead of writing")
    parser.add_argument("--show", metavar="ID", help="print one table, e.g. E07 or T05")
    args = parser.parse_args(argv)

    index, paper = load_index(), load_paper_tables()
    if args.show:
        table_id = args.show.upper()
        table = (paper if table_id in paper else index)[table_id]
        print(f"{table.label or table.id}. {table.title}\n")
        print(table.to_markdown())
        return 0

    problems = check_consistency(index, paper)
    for problem in problems:
        print(f"inconsistent: {problem}", file=sys.stderr)

    documents = {args.output: render_markdown(index)}
    if README.is_file():
        documents[README] = embed_tables(README.read_text(encoding="utf-8"), index, paper)
    stale = [
        path
        for path, text in documents.items()
        if not path.is_file() or path.read_text(encoding="utf-8") != text
    ]
    if args.check:
        for path in stale:
            print(f"{path} is out of date; run {Path(__file__).name}", file=sys.stderr)
        return 1 if stale or problems else 0

    for path in stale:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(documents[path], encoding="utf-8")
        print(f"wrote {path}")
    print(f"{len(index)} experiments, {len(paper)} paper tables, {len(problems)} inconsistencies")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
