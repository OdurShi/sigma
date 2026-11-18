# Supplementary experiment tables (E01-E09)

Supplementary results reported by the authors of the paper: transfer to broader guidance
edits and deeper compositions, annotation and training-cost accounting, a label sweep for the
factorised variant SIGMA-F, and related-method controls. The rendered tables, with the setup
and the key findings of each experiment, are in
[docs/supplementary_results.md](../../docs/supplementary_results.md); the figures are in
[assets/results/](../../assets/results/README.md).

## Layout

| Folder | Experiments |
|---|---|
| `01_transfer_and_composition/` | E01 paraphrased descriptions, random serialisation and cross-group binding swaps; E02 local residuals and accuracy on binding paths of depth 8 and 16 |
| `02_annotation_efficiency_and_cost/` | E03 SIGMA-4 vs. SFT-16: labels and training GPU-hours; E04 equal training-time caps |
| `03_sigma_vs_sigma_f/` | E05 SIGMA vs. SIGMA-F over 1/2/4/8/16 shots; E06 their 4-shot training cost |
| `04_related_method_baselines/` | E07 PBM-LoRA-style control; E08 Symbol-Tuning-style binding augmentation; E09 paired gains with 95% intervals |

`index.yaml` lists the experiments (title, the tables of the paper they extend, setup, key
findings) and says how the CSV columns are displayed.

## File format

One CSV file per experiment, UTF-8, one header row. Composite cells are split into one
column per number:

| Displayed as | Columns |
|---|---|
| `mean ± SD` | `<name>_mean`, `<name>_sd` |
| `value [low, high]` (95% CI) | `<name>`, `<name>_ci_low`, `<name>_ci_high` |
| `a / b` | two columns, e.g. `sigma_comp`, `sigma_changed` |
| `≈value` | `<name>_approx` |

Accuracy is in percent, gains in percentage points, cost in training GPU-hours.

## Reading the tables

```python
from sigma.supplementary import load, load_index

table = load("E05")  # one experiment
print(table.to_markdown())
row = table.row(dataset="BANKING77", method="SIGMA-F")
print(table.number(row, "shots_4_mean"))

for experiment in load_index():  # all of them
    print(experiment.id, experiment.title, len(experiment.rows))
```

## Updating the numbers

1. Edit the CSV file (keep the header). For a new experiment add a CSV file and an entry in
   `index.yaml`.
2. `python scripts/render_supplementary.py` rewrites `docs/supplementary_results.md` and the
   README tables, and prints every disagreement between tables that report the same quantity
   (including the tables of the paper in [`results/paper/`](../paper/README.md)).
3. `python scripts/plot_results.py` redraws the figures.
4. `pytest tests/test_supplementary_results.py` runs the same checks.
