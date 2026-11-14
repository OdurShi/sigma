"""Extract the numbered figures of the paper from its PDF into ``assets/paper_figures``.

Usage::

    python scripts/extract_paper_figures.py /path/to/paper.pdf [--output assets/paper_figures]

Requires PyMuPDF (``pip install pymupdf``).  The PDF is only read.  Every figure is rendered
from a fixed crop rectangle (PDF points, origin at the top-left corner of the page) that
contains the figure without its caption, the running page header or the margin line numbers.
The rectangles below were set for the submitted manuscript and checked by eye; adjust them if
the pagination of the PDF changes.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class FigureCrop:
    """One figure: output name, 1-based page number and crop rectangle ``(x0, y0, x1, y1)``."""

    name: str
    page: int
    box: tuple[float, float, float, float]


CROPS: tuple[FigureCrop, ...] = (
    FigureCrop("fig01", 2, (104.0, 78.0, 508.0, 208.0)),
    FigureCrop("fig02", 3, (104.0, 78.0, 508.0, 206.0)),
    FigureCrop("fig03", 4, (104.0, 180.0, 514.0, 402.0)),
    FigureCrop("fig04", 8, (104.0, 312.0, 508.0, 468.0)),
    FigureCrop("fig05", 9, (104.0, 78.0, 508.0, 212.0)),
    FigureCrop("fig06", 9, (104.0, 302.0, 508.0, 454.0)),
    FigureCrop("fig07", 49, (104.0, 384.0, 508.0, 598.0)),
    FigureCrop("fig08", 51, (100.0, 316.0, 510.0, 484.0)),
)


def extract(pdf: Path, output: Path, dpi: int = 200) -> list[Path]:
    """Render every crop of :data:`CROPS` to ``<output>/<name>.png`` and return the paths."""
    import fitz  # PyMuPDF

    output.mkdir(parents=True, exist_ok=True)
    written = []
    with fitz.open(pdf) as document:
        for crop in CROPS:
            page = document[crop.page - 1]
            pixmap = page.get_pixmap(clip=fitz.Rect(*crop.box), dpi=dpi)
            target = output / f"{crop.name}.png"
            pixmap.save(target)
            written.append(target)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("pdf", type=Path, help="path to the paper PDF (read-only)")
    parser.add_argument("--output", type=Path, default=Path("assets/paper_figures"))
    parser.add_argument("--dpi", type=int, default=200)
    args = parser.parse_args()
    for path in extract(args.pdf, args.output, args.dpi):
        print(f"wrote {path} ({path.stat().st_size / 1024:.0f} kB)")


if __name__ == "__main__":
    main()
