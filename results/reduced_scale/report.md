# Reduced-scale run: generated report

Configuration: `configs/reduced_scale/hwu64_subset_qwen2.5-0.5b.yaml` (Qwen2.5-0.5B-Instruct with LoRA, 7-intent subset of HWU64, 4 shots, 12 optimiser steps, 30 test instances, seed 13; one run per method).

Runs: `sigma train --config configs/reduced_scale/hwu64_subset_qwen2.5-0.5b.yaml --set method=<method> output_dir=runs/reduced_scale/<method>` for `sft`, `paired_sft`, `sigma`.

Tables generated with `python scripts/make_tables.py --runs results/reduced_scale --table <n>` from the `results.json` files in this folder. With a single seed the standard deviations are zero.

## Base and composition accuracy (layout of Table 1)

| Method | hwu64: Base | hwu64: Comp. |
|---|---|---|
| LoRA-SFT | 70.00 | 41.47 |
| Paired SFT (lambda = 0) | 76.67 | 41.45 |
| SIGMA | 66.67 | 35.11 |

## Guidance-use diagnostics (layout of Table 2)

| Method | Bind. Chg. Acc. | Bind. PairAcc | Comp. Chg. Acc. | Comp. PairAcc | EquivAcc | TV_eq |
|---|---|---|---|---|---|---|
| LoRA-SFT | 16.67 | 50.00 | 12.50 | 33.33 | 46.55 | 0.3522 |
| Paired SFT (lambda = 0) | 0.00 | 61.67 | 14.58 | 34.48 | 42.53 | 0.3926 |
| SIGMA | 16.67 | 53.33 | 11.46 | 25.29 | 34.48 | 0.3934 |

## Matched evaluation support (layout of Table 11)

| Method | hwu64: Matched Base | hwu64: Comp. | hwu64: Paired drop |
|---|---|---|---|
| LoRA-SFT | 68.94 ± 0.00 | 41.47 ± 0.00 | 27.47 ± 0.00 |
| Paired SFT (lambda = 0) | 75.84 ± 0.00 | 41.45 ± 0.00 | 34.39 ± 0.00 |
| SIGMA | 65.49 ± 0.00 | 35.11 ± 0.00 | 30.38 ± 0.00 |

## Changed and unchanged targets, reversal (layout of Table 12)

| Method | hwu64: Chg. Acc. | hwu64: Chg. PairAcc | hwu64: Unch. Acc. | hwu64: Unch. PairAcc | hwu64: Rev. Acc. | hwu64: Rev. Inv. |
|---|---|---|---|---|---|---|
| LoRA-SFT | 12.50 ± 0.00 | 3.12 ± 0.00 | 76.92 ± 0.00 | 70.51 ± 0.00 | 73.33 ± 0.00 | 86.67 ± 0.00 |
| Paired SFT (lambda = 0) | 14.58 ± 0.00 | 5.21 ± 0.00 | 74.36 ± 0.00 | 70.51 ± 0.00 | 73.33 ± 0.00 | 83.33 ± 0.00 |
| SIGMA | 11.46 ± 0.00 | 1.04 ± 0.00 | 64.10 ± 0.00 | 55.13 ± 0.00 | 80.00 ± 0.00 | 83.33 ± 0.00 |

## Composition acceptance and coverage (layout of Table 14)

| Dataset | Condition | Accept (%) | Inst. (%) | Changed | Unchanged | Labels |
|---|---|---|---|---|---|---|
| hwu64 | k2 | 96.67 (29 / 30) | 96.67 | 14 | 15 | 7 / 7 |
| hwu64 | k2+R | 96.67 (29 / 30) | 96.67 | 14 | 15 | 7 / 7 |
| hwu64 | k4 | 93.33 (28 / 30) | 93.33 | 16 | 12 | 7 / 7 |
| hwu64 | k4+R | 93.33 (28 / 30) | 93.33 | 16 | 12 | 7 / 7 |
| hwu64 | k8 | 100.00 (30 / 30) | 100.00 | 18 | 12 | 5 / 7 |
| hwu64 | k8+R | 100.00 (30 / 30) | 100.00 | 18 | 12 | 5 / 7 |
