# Tables of the paper

Tables 1, 2, 5, 8, 11, 12 and 13 of the paper as CSV files, one file per table, with the
column layout described in `index.yaml`. They are the source of the tables and figures in
the [README](../../README.md) and the reference for the consistency checks of the
[supplementary experiments](../supplementary/README.md).

| File | Table |
|---|---|
| `table01_composition_generalization.csv` | Table 1: Base and Comp. accuracy at 10 shots |
| `table02_guidance_use_diagnostics.csv` | Table 2: guidance-use diagnostics and ablations |
| `table05_label_efficiency.csv` | Table 5: label efficiency over 1/2/4/8/16 shots, with AULC |
| `table08_cost_and_latency.csv` | Table 8: computational cost and inference latency |
| `table11_matched_support.csv` | Table 11: accuracy with matched evaluation support |
| `table12_guidance_edits.csv` | Table 12: changed / unchanged targets and reversal |
| `table13_paired_differences.csv` | Table 13: paired differences with 95% intervals |

```python
from sigma.supplementary import load

table = load("T05")  # Table 5
row = table.row(dataset="HWU64", method="SIGMA", view="Comp.")
print(table.number(row, "shots_4_mean"))
```

```bash
python scripts/render_supplementary.py --show T01
```
