"""Print a table of the paper (Markdown) from finished runs.

Usage::

    python scripts/make_tables.py --runs runs/table1_main --table 1

Tables: 1 (Base/Comp.), 2 (diagnostics and ablations), 4 (averages and gap), 5 (label
budgets and AULC), 7 (macro-F1 and ranking), 10 (opaque outputs), 11 (matched support),
12 (changed/unchanged targets and reversal), 13 (paired bootstrap), 14 (acceptance).
"""

from sigma.cli import main

if __name__ == "__main__":
    import sys

    main(["tables", *sys.argv[1:]])
