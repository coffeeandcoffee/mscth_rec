# EEG Pipeline — Build Complete

## What Was Built

**32 files** in `05-12_data_analysis_and_results/scripts/`:

| File | Purpose |
|---|---|
| `config.py` | Constants, participant→session mapping, DSP utilities |
| `run.py` | Orchestrator with `--resume` and `--from-run` support |
| `step01_preprocess.py` | Raw CSV → baseline-normalised relative power (nonotch + notch) |
| `step02_artifact_flag.py` | Blink (AF7/AF8 PTP>150µV) + EMG (TP9/TP10 broadband) flagging |
| `step03_label_windows.py` | SKIP/STAY window extraction, burst-skip flagging, 4 sensitivity manifests |
| `step04_balance_split.py` | Temporal blocked 5-fold CV, 3s gap, undersample to 50/50, 5 seeds |
| `step05_feature_engineering.py` | 112 features (full), 56 frontal, 56 temporal, EI scalar — nonotch + notch |
| `step06_grid_search.py` | Nested CV grid search, locks `best_params.json` per participant |
| `step07_train_intra.py` | **PRIMARY RESULT** — RF training, temporal blocked + random-split, Wilcoxon |
| `step08_train_ei.py` | EI logistic regression with identical CV for apples-to-apples comparison |
| `step09_significance.py` | Wilcoxon, binomial CIs, seed stability, electrode ablation, EI vs RF |
| `step10_ablation_notch.py` | Notch vs nonotch paired Wilcoxon |
| `step11_feature_importance.py` | Gini importance, Friedman band test, SFEI construction |
| `step12_robustness_subgroups.py` | TikTok usage / sex / paid-unpaid Mann-Whitney comparisons |
| `step13_sensitivity_manifests.py` | Artifact-excluded, burst-excluded sensitivity runs |
| `step14_logo_cv.py` | Leave-one-group-out CV with confusion matrices |
| `step15_compile_report.py` | Master results CSV, LaTeX tables, `numbers_to_update.txt` |
| `viz01–viz15` | 15 visualization scripts (one per step) |

## How to Run

```bash
cd "05-12_data_analysis_and_results/scripts"
../venv/bin/python3 run.py
```

> [!IMPORTANT]
> The pipeline will pause after writing `parameters.json` and ask for approval before running.

### Resume a failed/interrupted run:
```bash
../venv/bin/python3 run.py --resume 20260513_1930
```

### Pre-fill parameters from a prior run:
```bash
../venv/bin/python3 run.py --from-run 20260513_1930 --auto-approve
```

## Output Structure

All outputs go into a timestamped run folder: `runs/run_YYYYMMDD_HHMMSS/`

```
runs/run_20260513_1930/
├── parameters.json
├── dropout_log.csv
├── artifact_summary.csv
├── balanced_counts.csv
├── step07_primary_result.json    ← THE primary number
├── ei_summary.csv
├── stats_summary.json
├── notch_ablation.csv
├── sfei_result.json
├── subgroup_comparisons.csv
├── sensitivity_comparisons.csv
├── logo_results.csv
├── master_results.csv            ← ALL comparisons in one table
├── numbers_to_update.txt
├── processed/{nonotch,notch}/P*.pkl
├── windows/{primary,artifact_exclude,...}/P*.pkl
├── splits/P*_splits.pkl
├── features/P*.pkl
├── features_notch/P*.pkl
├── grid_search/P*_best_params.json
├── results/intra/P*_results.pkl
├── results/ei/P*_ei.pkl
├── tables/*.tex
├── viz/viz01–viz15.png
└── step{01–15}.done
```

## Validation Status

- ✅ 25 participants mapped (P4–P31, excluding P16, P19, P29)
- ✅ All 32 files pass syntax check
- ✅ All 15 step modules import successfully with `run()` entry point
- ✅ `config.py` validates: data dir exists, survey CSV exists, 112 features confirmed
- ✅ Virtual environment created at `05-12_data_analysis_and_results/venv/`

## Key Design Decisions

1. **One run = all comparisons.** No cross-run symlinks or compare scripts. Everything internal.
2. **STAY recall is primary metric.** Accuracy reported as secondary confirmation only.
3. **Decision chain:** step06 locks hyperparams → step07 locks primary result → everything else is observational.
4. **4 sensitivity manifests** created in step03, tested in step13 against the locked primary.
5. **Balanced N from step04** (not step03 raw counts) used for binomial CIs in step09.
