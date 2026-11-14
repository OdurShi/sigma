"""Rebuild the files shipped under ``data/`` (Appendix C.1, C.2).

Usage::

    python scripts/prepare_data.py --dataset all --out-dir data

For BANKING77, CLINC150 and HWU64 the upstream few-shot intent-detection release is
downloaded into ``data/raw`` (only if it is not there yet), its split sizes are checked
against Table 3, and the converted splits, the hierarchy, the task record, the rendered base
guidance and the nested few-shot orderings are written.  An existing ``hierarchy.json`` is
reused; delete it to recompute the BANKING77 label clusters (needs the ``data`` extra for
the all-mpnet-base-v2 label embeddings).  ``--dataset synthetic`` writes the quick-start
variant of the synthetic task.  ``data/SHA256SUMS`` is refreshed at the end.
"""

from sigma.cli import main

if __name__ == "__main__":
    import sys

    main(["prepare-data", *sys.argv[1:]])
