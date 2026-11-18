# Result figures

Drawn by [`scripts/plot_results.py`](../../scripts/plot_results.py) from the reported tables
in [`results/paper/`](../../results/paper/README.md) and
[`results/supplementary/`](../../results/supplementary/README.md).

| File | Content | Tables |
|---|---|---|
| `main_results.png` | Base and Comp. accuracy of every method on the three datasets | Table 1 |
| `label_efficiency.png` | Comp. accuracy against the label budget, including SIGMA-F | Table 5, E05 |
| `e01_broader_edits.png` | Changed-target accuracy of frozen models under three broader guidance edits | E01 |
| `e02_depth_residuals.png` | Median local residual on unseen binding paths of depth 8 and 16 | E02 |
| `e03_e06_labels_and_cost.png` | Training GPU-hours and Comp. accuracy of SFT-16, SIGMA-4 and SIGMA-F-4 | E03, E06 |
| `e09_paired_gains.png` | Comp. gains of SIGMA over paired SFT and transported KL with 95% intervals | E09 |

To redraw after editing a table: `python scripts/plot_results.py`.
