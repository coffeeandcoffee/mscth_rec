# Thesis Build Plan: Step 1 (Pipeline Spec) + Step 2 (Full Task List)

---

## STEP 1 — Exact Pipeline Specification for Results Tables and Figures

### What we know from the run data

All four runs share the same 25 participants (PIDs 4–26, one missing: PID 16 absent in sample listing), same window count per participant, same EI values. The distinguishing axes are:

- **Run 1** (`run_20260525_115853`): skip_window=±3.0 s, window=3.0 s, stride=0.6 s (80% overlap), grid-search RF (depth≤5, n≤300), standard features (no Hilbert, no ERP, min/max active)
- **Run 2** (`run_20260526_113959`): skip_window=±0.5 s, window=1.0 s, stride=1.0 s (0% overlap), grid-search RF (depth≤5, n≤300), same standard features
- **Run 3** (`run_20260526_133703`): identical to Run 2 but RF capped: depth=3, n=100, grid search bypassed
- **Run 4** (`run_20260526_150106`): identical to Run 3 but +Hilbert envelope, +raw ERP features, min/max ablated

Primary model in all runs: `Intra|NoNotch|NoArt|NoBurst|RF`  
Source for primary model per-participant data: `parallel_universe_metrics.csv` (columns: pid, test_recall, train_recall, test_f1, train_f1, test_accuracy, train_accuracy) filtered to `combination == "Intra|NoNotch|NoArt|NoBurst|RF"`  
Source for EI per-participant data: same file filtered to `combination == "Intra|NoNotch|NoArt|NoBurst|EI"`  
Source for LOGO-CV per-participant data: `logo_results.csv` (columns: test_pid, accuracy, recall)  
Source for feature importance: `feature_ranking_112.csv` (columns: rank, feature, mean_importance, std_importance)  
Source for electrode-band heatmap: `electrode_band_heatmap.json` (28 keys: electrode_band → float)  
Source for notch ablation per-participant: `notch_ablation.csv` (columns: pid, recall_nonotch, recall_notch, delta)  
Source for per-participant significance: `per_participant_significance.csv` (columns: pid, recall, n_total, n_correct, ci_low, ci_high, above_chance, significantly_above_chance)

---

### TABLE 1 — Per-Participant Results Across All Four Runs

**Purpose:** The thesis equivalent of Sold's Table 6.1. Each row is one participant. Shows individual variation, which participants drive aggregate, and the trajectory from overfitting baseline to final pipeline. This is the single most important table in the results chapter.

**Source files:**  
- `parallel_universe_metrics.csv` from each run, filtered to `Intra|NoNotch|NoArt|NoBurst|RF`  
- `per_participant_significance.csv` from Run 4 (for CI columns)

**Exact columns (left to right):**

| Column | Source | Notes |
|---|---|---|
| PID | pid | Displayed as P04, P05 … P26 |
| n_skip | n_skip_raw from balanced_counts.csv | Number of SKIP windows (= number of swipe events used) |
| R1 Test Acc | test_accuracy, R1 | Run 1 primary model |
| R1 Train Acc | train_accuracy, R1 | |
| R1 ΔAcc | train_accuracy − test_accuracy, R1 | Generalisation gap; positive = train better than test |
| R2 Test Acc | test_accuracy, R2 | |
| R2 Train Acc | train_accuracy, R2 | |
| R2 ΔAcc | delta R2 | |
| R3 Test Acc | test_accuracy, R3 | |
| R3 Train Acc | train_accuracy, R3 | |
| R3 ΔAcc | delta R3 | |
| R4 Test Acc | test_accuracy, R4 | |
| R4 Train Acc | train_accuracy, R4 | |
| R4 ΔAcc | delta R4 | Lowest gap = best generalisation |
| R4 Test Recall | test_recall, R4 | STAY class recall, primary metric |
| R4 95% CI | [ci_low, ci_high], per_participant_significance.csv R4 | Binomial CI on R4 recall |
| R4 Sig.↑ | significantly_above_chance, R4 | ✓ or — |

**Footer row:** Mean ± SD across all 25 participants for each numeric column.

**Placement:** Main results chapter, immediately before the aggregate progression narrative.

**Rendered format:** Full-width landscape table. In LaTeX: `\begin{table*}` with `\small` font, `\caption` referencing all four runs.

---

### TABLE 2 — Aggregate Pipeline Progression Summary

**Purpose:** The four-run aggregate view in a compact form. One row per run. This is the table referenced constantly in the narrative.

**Source files:** `stats_summary.json`, `notch_ablation_result.json`, `logo_confusion.json`, `parallel_universe_metrics.csv` (mean over 25 PIDs for primary model), `parallel_universe_comparisons.csv`

**Exact columns:**

| Column | Exact source field | Notes |
|---|---|---|
| Run | — | R1 Baseline, R2 Non-Overlap, R3 Regularised RF, R4 Full Signal |
| Skip window | parameters.json step01.skip_window_s | ±3.0 s (R1) vs ±0.5 s (R2–R4) |
| Window / stride | parameters.json step03 | 3.0 s / 0.6 s vs 1.0 s / 1.0 s |
| RF config | grid_search results / fixed | "Grid (depth≤5)" vs "Fixed depth=3, n=100" |
| Mean Train Acc | mean of train_accuracy over 25 PIDs, primary model | |
| Mean Test Acc | mean of test_accuracy over 25 PIDs, primary model | |
| ΔAcc (train−test) | difference of above two | |
| Mean Test Recall | mean of test_recall over 25 PIDs, primary model | Primary metric throughout |
| Mean EI Recall | mean of test_recall, EI combination | Baseline comparison |
| Wilcoxon vs chance: W, p, r | stats_summary.json → primary_wilcoxon | One-sample Wilcoxon, per-participant recall vs 0.50 |
| RF vs EI: p | parallel_universe_comparisons.csv, row where Compared_To contains "EI", Wilcoxon_p | Paired Wilcoxon |
| NoNotch vs Notch: Δrecall, W, p, r | notch_ablation_result.json | mean_nn − mean_nt, W, p, r |
| LOGO-CV Acc | logo_confusion.json → mean_accuracy | |
| LOGO-CV Recall | logo_confusion.json → mean_recall | |

**Note on Run 1:** `stats_summary.json` for Run 1 does not exist in the files provided (Run 1 = `run_20260525_115853`). The W=102, p=0.107 and r=0.372 values come from methods_and_results.md. Cross-check against `parallel_universe_comparisons.csv` row "vs 50% chance baseline" in that run folder.

---

### TABLE 3 — Statistical Tests Summary

**Purpose:** A dedicated table of all Wilcoxon tests reported in the thesis, as a single reference. Sold has no direct equivalent but a professor will want all statistical claims findable in one place.

**Source files:** `master_results.csv`, `stats_summary.json`, `notch_ablation_result.json`, `sensitivity_comparisons.csv`, `electrode_ablation.csv`, `parallel_universe_comparisons.csv`

**Exact columns:** Test description | Run | W (or U) | p-value | Effect r | Significant | Direction

**Exact rows (one per test):**

| Test description | Run | Source row |
|---|---|---|
| RF (primary) vs 50% chance | R1 | stats_summary.json primary_wilcoxon / master_results.csv row 1 |
| RF (primary) vs 50% chance | R2 | same |
| RF (primary) vs 50% chance | R3 | same |
| RF (primary) vs 50% chance | R4 | same |
| RF vs EI (paired) | R2 | parallel_universe_comparisons.csv, Compared_To = EI |
| RF vs EI (paired) | R3 | same |
| RF vs EI (paired) | R4 | same |
| NoNotch vs Notch (paired) | R1 | notch_ablation_result.json |
| NoNotch vs Notch (paired) | R2 | same |
| NoNotch vs Notch (paired) | R3 | same |
| NoNotch vs Notch (paired) | R4 | same |
| Full vs frontal-only electrode set | R2 | electrode_ablation.csv row full_vs_frontal |
| Full vs temporal-only electrode set | R2 | electrode_ablation.csv row full_vs_temporal |
| Full vs frontal-only electrode set | R4 | electrode_ablation.csv R4 |
| Full vs temporal-only electrode set | R4 | electrode_ablation.csv R4 |
| Artifact exclusion effect | R2 | sensitivity_comparisons.csv row artifact_exclude |
| Burst exclusion effect | R2 | sensitivity_comparisons.csv row burst_exclude |
| Artifact exclusion effect | R3 | same file R3 (n.s.) |
| Burst exclusion effect | R3 | same file R3 (n.s.) |
| Paid vs unpaid subgroup | R1 | subgroup_comparisons.csv paid_vs_unpaid |
| Paid vs unpaid subgroup | R2 | same (resolved) |

**This table goes in the appendix** (Appendix B), with all tests cited by their table row number in the main text.

---

### TABLE 4 — LOGO-CV Per-Participant Results, Final Run (R4)

**Purpose:** Shows cross-subject generalisation at the individual level. Analogous to knowing which "fold" produced which result. Shows the variance around the 65.9% mean.

**Source file:** `logo_results.csv` from `run_20260526_150106 3 True`

**Exact columns:**

| Column | Source field | Notes |
|---|---|---|
| PID | test_pid | Formatted as P04 … P26 |
| n_test | n_test | Number of balanced test windows for this participant as the held-out fold |
| n_train | n_train | Total training windows (all other 24 participants combined) |
| LOGO Accuracy | accuracy | |
| LOGO Recall (STAY) | recall | |
| TN / FP / FN / TP | cm_tn, cm_fp, cm_fn, cm_tp | 4 cells of the confusion matrix |

**Footer row:** Mean ± SD for accuracy and recall. Note aggregate accuracy separately: sum(TP+TN) / sum(all) across all folds.

**Companion table (same format, R3):** `logo_results.csv` from Run 3. Placed adjacent for comparison because the narrative compares R3 LOGO (65.3%) with R4 LOGO (65.9%).

---

### TABLE 5 — Feature Importance: Top 20 Features, Final Run (R4)

**Purpose:** Full ranked feature list, showing feature name, mean Gini importance across 25 participants, and SD. Placed in Appendix A. Top 10 referenced in main text.

**Source file:** `feature_ranking_112.csv` from `run_20260526_150106 3 True`

**Exact columns:** Rank | Feature name | Mean Gini importance | SD | Band | Electrode

The Band and Electrode columns are derived by parsing the feature name string (e.g., `TP9_delta_macrofreq` → Electrode: TP9, Band: delta, Feature type: macrofreq). These do not exist in the file and must be split by the pipeline script.

**Companion tables:** Same table for R2 and R3 (placed in Appendix A), showing the top-20 feature shift across runs. The three tables placed side-by-side (R2 | R3 | R4) make the theta→delta/low-gamma topology shift visually legible.

---

### FIGURE 1 — Intra-Subject Pipeline Progression (3-panel)

**Purpose:** The visual anchor of the results chapter. Shows the four-run trajectory in the three metrics that matter most: test accuracy, test recall, and generalisation gap.

**Source:** Mean values from `parallel_universe_metrics.csv` (primary model), SD from same. ΔAcc computed as train_accuracy − test_accuracy per participant, then mean/SD over 25.

**Layout:** Three vertically stacked panels sharing the same x-axis (R1, R2, R3, R4).

**Panel A — Mean Test Accuracy:**  
- Y-axis: 0.45 to 0.75, labelled "Mean Test Accuracy"  
- X-ticks: "R1\nBaseline", "R2\nNon-Overlap", "R3\nRegularised RF", "R4\nFull Signal"  
- Error bars: ±1 SD across 25 participants  
- Horizontal dashed line at 0.50 (chance), labelled "Chance"  
- Significance markers above each bar: n.s. (R1), *** (R2, p<0.001), *** (R3), **** (R4, p<0.0001)  
- Exact values: R1=0.542, R2=0.648, R3=0.631, R4=0.658

**Panel B — Mean Test Recall (STAY class):**  
- Y-axis: 0.45 to 0.85  
- Same x-axis, same error bars  
- Horizontal dashed line at 0.50  
- Significance markers same as Panel A  
- Exact values to read from `parallel_universe_metrics.csv` mean over 25 PIDs: R1 = mean test_recall primary model R1; R2=0.694; R3=0.668; R4=0.752  
- *(Note: R1 mean recall from the md is not stated as aggregate mean — read from the R1 parallel_universe_metrics.csv directly. The W=102 test was on per-participant recalls so those values exist in that file.)*

**Panel C — Mean Generalisation Gap (ΔAcc = Train Acc − Test Acc):**  
- Y-axis: −0.60 to 0.35, labelled "Train − Test Accuracy"  
- Horizontal dashed line at 0.0, labelled "No gap"  
- Red shading for negative values (R1 = −0.389 approx from mean train 0.931 − mean test 0.542; exact from file), blue for positive  
- Exact values: R1 = 0.931−0.542 = +0.389 (train domination), R2 = 0.915−0.648 = +0.267, R3 = 0.829−0.631 = +0.198, R4 = 0.801−0.658 = +0.143  
- *(Note: the md stated ΔF1 not ΔAcc — use accuracy-based gap here for consistency with the table. Both are computed in the script.)*

**Caption:** "Figure X. Intra-subject classification performance of the primary model (Intra|NoNotch|NoArt|NoBurst|RF) across the four pipeline iterations (n=25 participants). (A) Mean test accuracy. (B) Mean STAY-class test recall. (C) Mean generalisation gap (train accuracy minus test accuracy). Error bars indicate ±1 SD. Dashed lines mark chance level (0.50) in A–B and zero gap in C. Significance of one-sample Wilcoxon test against 0.50 per recall: n.s., ***, ***, **** respectively (see Table [stats table ref])."

---

### FIGURE 2 — Inter-Subject (LOGO-CV) Pipeline Progression (2-panel)

**Purpose:** Shows cross-subject generalisation trajectory across runs. Directly paired with Figure 1 to show the intra/inter contrast.

**Source:** `logo_confusion.json` from each run: `mean_accuracy`, `mean_recall`, `std_accuracy`, `std_recall`

**Exact values from the data files:**  
- R1: mean_accuracy=0.6627 (from logo_confusion.json Run 1 = run_20260525_115853), mean_recall=0.9440  
- R2: mean_accuracy=0.6627 (same file structure, Run 2 rerun), mean_recall=0.9440  
- R3: mean_accuracy=0.6527, mean_recall=0.8873  
- R4: mean_accuracy=0.6590, mean_recall=0.9107

*(Note: Runs 1 and 2 have identical logo_confusion.json values — the LOGO-CV in the rerun used the same feature set as the original ±0.5 s run and produced identical results. Verify this is not a copy/paste artifact in the data files before finalising.)*

**Layout:** Two panels side by side.

**Panel A — LOGO-CV Mean Accuracy:**  
- Y-axis: 0.50 to 0.75  
- X-axis: R1, R2, R3, R4  
- Error bars: ±1 SD (std_accuracy from logo_confusion.json)  
- Horizontal dashed line at 0.50

**Panel B — LOGO-CV Mean STAY Recall:**  
- Y-axis: 0.50 to 1.00  
- Same structure  
- Note in caption that the high recall with modest accuracy indicates systematic false-positive bias: the model tends to predict STAY for most windows, which inflates recall but depresses accuracy for non-STAY windows. This is stated explicitly in the caption.

**Caption:** "Figure X. Cross-subject generalisation performance (LOGO-CV, n=25 leave-one-participant-out folds) across the four pipeline iterations. (A) Mean test accuracy. (B) Mean STAY-class test recall. Error bars indicate ±1 SD. Dashed lines mark chance (0.50). Note that the systematically elevated recall in Panel B relative to accuracy in Panel A reflects a prediction bias toward the majority (STAY) class under cross-subject conditions, consistent with the idiosyncratic nature of the neural engagement signature (see Section X.X)."

---

### FIGURE 3 — Per-Participant Test Accuracy: R1 vs R4 (Paired Scatter)

**Purpose:** The most compelling single figure. Shows that the pipeline improvement is consistent across nearly every individual participant, not driven by a few outliers.

**Source:** `parallel_universe_metrics.csv`, R1 and R4, filtered to primary model, columns `pid` and `test_accuracy`

**Layout:** Single scatter plot.  
- X-axis: R1 Test Accuracy (0.40–0.80), labelled "Run 1 (Baseline) Test Accuracy"  
- Y-axis: R4 Test Accuracy (0.40–0.80), labelled "Run 4 (Final) Test Accuracy"  
- Diagonal reference line (x=y), labelled "No change"  
- Each point = one participant  
- Colour: points where R4 > R4 chance (>0.50) AND R4 significantly above chance (from per_participant_significance.csv) → filled blue circle; above chance but not significant → open blue circle; not above chance → grey circle  
- Annotate two or three outlier PIDs if they lie far from the diagonal (identify from data)

**Caption:** "Figure X. Per-participant test accuracy for the baseline pipeline (R1) and the final pipeline (R4) (n=25). Each point represents one participant. Points above the diagonal indicate improved performance in R4. Colour coding reflects R4 binomial significance status: filled blue = significantly above chance (binomial 95% CI, corrected), open blue = above chance but not significant, grey = at or below chance."

---

### FIGURE 4 — Feature Importance Heatmap (4-panel, electrode × band)

**Purpose:** Shows the feature topology shift across runs. The 4×7 electrode-band importance matrix is the most informative single summary of what the model is doing differently in each run.

**Source:** `electrode_band_heatmap.json` from each of the four runs. This file contains 28 keys (4 electrodes × 7 bands) with their mean Gini importance value aggregated across all features in that cell and all 25 participants.

**Layout:** Four panels arranged in a 2×2 grid (R1 top-left, R2 top-right, R3 bottom-left, R4 bottom-right).

Each panel is a heatmap:  
- Rows: TP9, AF7, AF8, TP10 (top to bottom)  
- Columns: δ, θ, α, β, Lγ, Hγ, VH (left to right)  
- Cell value: the importance value from electrode_band_heatmap.json for that electrode_band key  
- Colour scale: sequential (e.g. white → dark blue), normalised within each panel to [0, 1] so that intra-panel contrast is maximised; panels are NOT cross-normalised (each panel's colour scale is independent, because the absolute importance values are comparable across runs but the pattern is what matters)  
- Each cell labelled with its numeric value (3 decimal places)  
- Panel title: "R1 — Baseline (±3.0 s, 80% overlap)", "R2 — Non-Overlap (±0.5 s)", "R3 — Regularised RF", "R4 — Full Signal (+Hilbert, +ERP)"

**What to expect:** R1 should show AF7/AF8 high-gamma and very-high as darkest cells. R2 and R3 should show TP9/TP10 theta as darkest. R4 should show TP9 delta and low-gamma as darkest, with broader distribution.

**Caption:** "Figure X. Mean Gini feature importance aggregated by electrode and frequency band across the four pipeline iterations (n=25). Each cell represents the normalised sum of mean importance values for all features derived from that electrode–band combination. Colour scale is normalised independently within each panel. The progressive shift from frontal high-gamma dominance (R1) to temporal theta (R2–R3) to distributed delta and low-gamma (R4) reflects increasing physiological coherence as the pipeline is refined."

---

### FIGURE 5 — Per-Participant Notch Ablation, Final Run (R4)

**Purpose:** Shows the NoNotch advantage is consistent across participants, not driven by a subset. Directly supports the high-gamma/cognitive variance interpretation.

**Source:** `notch_ablation.csv` from Run 4 (columns: pid, recall_nonotch, recall_notch, delta)

**Layout:** Two panels.

**Panel A — Paired dot plot:**  
- Each participant shown as two connected dots: one for NoNotch recall, one for Notch recall  
- Participants sorted by delta (descending)  
- Y-axis: 0.30–1.00 STAY recall  
- Connecting line colour: blue if NoNotch > Notch, grey if Notch ≥ NoNotch

**Panel B — Delta distribution:**  
- Histogram of (recall_nonotch − recall_notch) across 25 participants  
- Vertical dashed line at 0  
- Annotate: mean delta = +X.XX pp, W=7, p<10⁻⁶

**Caption:** "Figure X. Per-participant effect of 50 Hz notch filter removal on STAY-class test recall for Run 4 (n=25). (A) Paired recall for each participant under NoNotch and Notch conditions. Connecting lines are blue where NoNotch outperforms Notch. (B) Distribution of per-participant recall differences (NoNotch minus Notch). The mean advantage of NoNotch (+[X] pp) is highly significant (paired Wilcoxon: W=[W], p=[p], r=[r]), consistent with the hypothesis that the 40–60 Hz band contains recoverable cognitive variance rather than solely power-line artefact."

---

### APPENDIX FIGURES — Feature Importance Full Rankings

**Appendix A, Figures A.1–A.3:**

Three bar charts, one per run (R2, R3, R4), showing the top 20 features by mean Gini importance with error bars (±1 SD across 25 participants). Features labelled on the y-axis. Source: `feature_ranking_112.csv` from each run.

Why only R2–R4: R1's feature ranking is dominated by high-gamma artefacts that are explicitly discarded in later runs; it is shown in the main text heatmap (Figure 4) and described in prose, but the full bar chart is not needed in the appendix.

---

### The Pipeline Step Itself (Code Specification)

The above outputs must be produced by a single Python script: `step17_thesis_figures_and_tables.py`. It reads directly from the four run directories, requires no manual data entry, and produces all tables as both `.csv` (for inspection) and `.tex` (for direct inclusion), and all figures as `.pdf` (vector, for LaTeX) and `.png` (for preview).

**Required inputs (paths hardcoded or passed as arguments):**
```
RUN_DIRS = {
    "R1": "path/to/run_20260525_115853",
    "R2": "path/to/run_20260526_113959",
    "R3": "path/to/run_20260526_133703",
    "R4": "path/to/run_20260526_150106"
}
```

**Required outputs (all written to `thesis_outputs/`):**
```
tables/
  table1_per_participant_all_runs.csv
  table1_per_participant_all_runs.tex
  table2_aggregate_progression.csv
  table2_aggregate_progression.tex
  table3_statistical_tests.csv
  table3_statistical_tests.tex
  table4_logo_cv_r4.csv
  table4_logo_cv_r4.tex
  table4_logo_cv_r3.csv
  table4_logo_cv_r3.tex
  table5_feature_importance_r4_top20.csv
  table5_feature_importance_r4_top20.tex
  appendix_a_feature_importance_r2_full.csv
  appendix_a_feature_importance_r3_full.csv
  appendix_a_feature_importance_r4_full.csv

figures/
  fig1_intra_progression.pdf / .png
  fig2_logo_progression.pdf / .png
  fig3_r1_vs_r4_scatter.pdf / .png
  fig4_electrode_band_heatmap.pdf / .png
  fig5_notch_ablation_r4.pdf / .png
  appendix_a_fig_a1_feature_ranking_r2.pdf / .png
  appendix_a_fig_a2_feature_ranking_r3.pdf / .png
  appendix_a_fig_a3_feature_ranking_r4.pdf / .png
```

**Key implementation notes:**
- Load `parallel_universe_metrics.csv` once per run; filter by combination string.
- PIDs must be sorted numerically and formatted as `P{pid:02d}` throughout.
- All significance stars follow convention: n.s. = p≥0.05, * = p<0.05, ** = p<0.01, *** = p<0.001, **** = p<0.0001.
- LaTeX tables use `\toprule / \midrule / \bottomrule` (booktabs), `\small` font, `\caption{}` and `\label{}` on every table.
- All figures use a consistent style: serif axis labels (matching LaTeX document font), no grid lines on scatter/paired plots, light grid on bar/line charts, figure size 3.5 in × 2.5 in per panel (matches A4 thesis column width).
- The heatmap colour scale: use `matplotlib` `Blues` sequential cmap, `vmin=0`, `vmax=max value in that panel` (per-panel normalisation).
- The scatter figure (Figure 3): use `matplotlib` `scatter()` with `zorder` to ensure annotated points are visible; add `annotate()` for the 2–3 most extreme movers.
- All numeric outputs in tables: 4 decimal places for importance values, 3 decimal places for recall/accuracy, scientific notation for p-values below 0.001.

---

---

## STEP 2 — Full Task List in Priority Order

### Block 1 — Results (code first, everything else depends on having the numbers)

1. **Write and run `step17_thesis_figures_and_tables.py`** as specified above. This produces all tables and figures from the raw run directories. No thesis chapter can be finalised until this step is complete and the outputs are reviewed for correctness. This is the single highest-priority task.

2. **Produce one raw results reference document** (`results_reference.md` or `results_reference.tex`): a flat, unnested document containing every number in the thesis — every mean, every SD, every W, every p, every r — with its exact source file and field name cited next to it. This document is used as the sole reference when writing result sentences. No number enters the thesis text without appearing in this document first. This prevents the abstract/body contradiction that currently exists.

3. **Produce the full appendix (Appendix A) with per-feature bar charts** as specified (Figures A.1–A.3 from `feature_ranking_112.csv`, top 20 features per run with SD error bars). Appendix B contains Table 3 (all statistical tests). Both produced by `step17`.

4. **GitHub cleanup and upload**: archive old pipeline runs (do not delete), tag the final run directory (`run_20260526_150106`) as canonical, add `step17_thesis_figures_and_tables.py` to the repository, and push a clean commit with a tag (`v_thesis_final_data`). This must happen before writing begins so that the LaTeX document references stable file paths.

---

### Block 2 — Methods chapter (write after Block 1 numbers are confirmed)

5. **Formal notation and definition framework**: define every symbol used in the methods chapter before it appears. At minimum: $n$ (number of participants), $W_i$ (analysis window), $\Delta t_{offset}$ (clock offset), $t_{LSL}^{key}$ (LSL-mapped keypress timestamp), $\tau_{sync}$ (synchronisation tolerance), $P_{env}(t)$ (amplitude envelope), $z(t)$ (z-score normalised signal), $f_s$ (sampling rate), $K$ (number of CV folds), $t_{split,k}$ (split marker), $\Delta_{gap}$ (temporal firewall). These are all used in the md but never collected into a notation section. Add a "Notation" subsection at the start of the Methods chapter.

6. **Add Participants subsection**: move the participant description from `thesis.tex` into the `methods_and_results.md`, update it with final numbers ($n=25$, age $M=24.6$, $SD=2.9$, range 20–30, 3 exclusion criteria stated), add compensation split (n=14 paid, n=11 unpaid), add ethics statement, add informed consent statement.

7. **Add Experimental Protocol subsection**: move the experimental session description from `thesis.tex` (cabin, phone, Guided Access, 25-minute limit, B/A keystroke mapping, post-session debrief) into the `methods_and_results.md`, rewrite in academic register, add the protocol flowchart figure placeholder with proper caption.

8. **Justify the 1.0-second window choice in the Methods** (Section 1.4 Window Extraction), not in the Results progression. The justification exists in the md's run descriptions (transient motor-readiness hypothesis, Libet readiness potential ~1 s pre-action). Move that argument into Methods, with the Libet and Shakeel citations, before the window extraction section. Delete the same text from the Results narrative.

9. **Convert literature bullet-lists to continuous academic prose**: remove all search query annotations (`\textit{Search Query:}...`), remove all `\begin{itemize}` structure from the literature chapter, and rewrite each section as flowing paragraphs where each claim leads to the next. The argumentative structure must be: (a) the platform behaviour creates an attention problem → (b) retrospective self-report cannot capture it → (c) EEG can, specifically via readiness potentials → (d) consumer-grade EEG has been validated for this purpose → (e) the 1.0-second pre-swipe window is theoretically grounded → therefore the present approach is justified. Every step must cite evidence; every transition must be argued, not just asserted.

10. **Fill all `[Insert citation]` placeholders**: minimum citations needed — Butterworth filtfilt (scipy docs or standard DSP text), LSL protocol (Kothe & Jung 2009), Hilbert transform (standard signal processing text), Hjorth parameters (Hjorth 1970), the balanced class prior (cite a standard ML reference), non-overlapping epoch extraction (cite one BCI paper using the same approach). Cross-reference the existing `\bibliography{library}` file; add any missing entries.

---

### Block 3 — Discussion, Conclusion, Limitations (write after Block 2)

11. **Write the full Discussion chapter**: current version is 3 pages; target 8–10 pages. Structure:
    - Section X.1: Revisiting the research question (what was asked, what was found, properly scoped)
    - Section X.2: The four-run progression as evidence of methodological maturation (connect each run's finding to its design choice, citing the appropriate BCI/EEG literature for each claim)
    - Section X.3: Feature topology interpretation (what the theta-peakfreq dominance in R2/R3 and the delta/low-gamma shift in R4 mean neuroscientifically; cite Libet readiness potential, theta motor planning literature, delta involvement in sustained attention)
    - Section X.4: The notch filter reversal and its implications (what it means for consumer-grade EEG practice)
    - Section X.5: Cross-subject generalisation and its theoretical expectation (Hasson et al. cortical idiosyncrasy argument, already partially written — expand with citations to Apicella and Bellos)
    - Section X.6: Limitations (see item 12 below)
    - Section X.7: Practical implications and future work (markerless detection concept; what would need to change for deployment)

12. **Write the formal Limitations section** (Section X.6 within Discussion, and a brief summary within the Methods conclusion subsection):
    - **L1 — No artefact rejection**: AF7/AF8 are maximally contaminated by blink and EMG; no ICA or threshold rejection was applied. The artefact_summary.csv shows ~12–26% of samples flagged per participant. The model learns features from these contaminated windows. Cite standard EEG preprocessing literature.
    - **L2 — Motor artefact confound**: the ±0.5 s window straddles the keypress. The model may be classifying pre-movement EMG (thumb/arm preparation) rather than cognitive disengagement. The STAY-class windows contain no such motor artefact by design. This is structurally confounded and cannot be resolved without a no-button control condition. This is the single most important limitation and must be stated prominently.
    - **L3 — Active video-watching baseline vs. resting state**: the 90-second baseline is a YouTube nature video, not true rest. The z-score normalisation therefore normalises against low-level visual attention, not eyes-closed resting. This inflates the apparent deviation of TikTok browsing from baseline.
    - **L4 — Keypress synchronisation jitter**: manual simultaneous keypress introduces ~200–500 ms behavioural latency. The ±0.5 s window may therefore not be precisely centred on the neural decision point. Cite Libet (1983) for the observation that conscious motor intention precedes action by ~550 ms.
    - **L5 — EI vs RF evaluation protocol mismatch**: the EI was evaluated on imbalanced data with 5-fold CV across the full dataset; the RF was evaluated on balanced data with stratified temporal CV per participant. The comparison is therefore not strictly controlled. State this explicitly and note that the direction of any bias (if the imbalanced EI evaluation inflates or deflates EI performance) should be examined.
    - **L6 — Homogeneous convenience sample**: n=25, age 20–30, all HPI affiliates. The results cannot generalise to older adults, non-academic populations, or participants with different hair/scalp properties affecting dry-electrode contact. State this explicitly once in Methods (participants section) and once in the Limitations section.
    - **L7 — Paid vs unpaid cohort confound**: subgroup_comparisons.csv shows this resolved in R2–R4 (p=0.396) but was significant in R1 (p=0.035, r=0.506). The initial recording session (unpaid, different room) introduced a confound that was masked by the overlapping geometry and resolved only when the pipeline was corrected. This must be noted as a historical artefact and its resolution confirmed.

13. **Write the Conclusion chapter** (separate from Discussion, ~3 pages): restate research question, state the three defensible claims in proper academic register, identify the two key methodological contributions (STAY paradigm definition; temporal firewall CV design that reveals the data-leakage inflation), and enumerate five specific directions for future work with citations.

---

### Block 4 — Document cleanup and formal structure

14. **Rewrite the Abstract**: current version references the old 3.0-second pipeline and 53.8% number. The final pipeline (R4) produces 75.2% mean recall and 65.9% LOGO-CV accuracy. The abstract must state: the final pipeline configuration (±0.5 s, 1.0 s non-overlapping windows, depth-3 RF with Hilbert + ERP features), the primary result (mean STAY recall 75.2%, Wilcoxon W=1, p<10⁻⁶, r=0.994), the cross-subject result (LOGO-CV accuracy 65.9%), the feature finding (distributed delta and low-gamma topology), and the methodological contribution (discovery of ~8 pp inflation from overlapping-window evaluation). All numbers from the `results_reference.md` produced in step 2.

15. **Add Section 1.1 — Thesis Overview**: one paragraph per chapter, written from the committee member's perspective. "Chapter 2 establishes the theoretical basis... Chapter 3 describes the data acquisition and pipeline... Chapter 4 presents the four-stage results progression... Chapter 5 situates the findings in the literature and enumerates limitations... The thesis concludes in Chapter 6 with..."

16. **Add Section 1.2 — Contributions**: explicit bullet list. Minimum entries:
    - The STAY paradigm: a labelling strategy that targets the neural state of sustained cognitive engagement rather than the motor act of swiping, operationalised as all viewing periods outside ±0.5 s of a keypress event
    - The SKIP isolation window: empirical identification of 1.0 s as the operative temporal boundary for pre-swipe neural classification, grounded in motor-readiness potential literature
    - The temporal firewall cross-validation architecture: a stratified temporal CV that eliminates autocorrelation leakage from overlapping windows, with empirical demonstration that this leakage inflates reported accuracy by ~8 pp
    - Discovery that 50 Hz notch filtering actively degrades classification recall under non-overlapping 1.0 s windows, with statistically significant reversal of the effect direction compared to overlapping windows
    - The four-stage pipeline progression: systematic methodological optimisation showing that feature engineering (Hilbert envelope, ERP extraction) contributes more to generalisation improvement than architectural changes alone (ΔAcc gap: 0.389 → 0.143)

17. **Remove all internal TODO/NOTE/INTERNAL comments**: search for `\textit{Note:}`, `% TODO`, `% NOTE`, `% INTERNAL`, `\textbf{The claims you are not allowed}`, `\textbf{Why the proof-of-concept}`, and every block of epistemic guardrail text. These must not exist in the submitted document.

18. **Remove all `[Insert...]` placeholders**: replace each with either the actual content (figures, numbers) or a proper LaTeX `\todo{}` tag that is stripped before submission.

19. **Write proper Acknowledgements**: 2–3 paragraphs. Name supervisors individually, state their specific contributions (methodological guidance, equipment access, statistical review). Acknowledge the HPI Digital Health Lab for compute access (logo_confusion.json came from runs on that infrastructure). Acknowledge participants. One sentence for any personal acknowledgements.

20. **Format and complete the bibliography**: verify every `\cite{}` key in the document has a corresponding entry in `library.bib`. Add missing entries. Ensure every entry has: author, title, venue/journal, year, and (where applicable) DOI or URL. Use `unsrt` or `natbib` consistently as already declared. Cross-check Libet 1983, Shakeel 2015, Hasson 2004, Apicella 2022, Bellos 2025, Pattisapu 2021, Knierim (2026 — check year), Baughan 2022, Anderson 2023, deBettencourt 2021, Pfurtscheller 1996, Pope 1995, Miller 2001, Fries 2015, Tong 2020, Krigolson 2017, Moontaha 2023 — all cited in the current thesis.tex but some without confirmed bib entries.
