# Thesis Build Plan

---

## STEP 1 — Pipeline step `step17_thesis_figures_and_tables.py`

**Inputs:** 4 run directories → **Outputs:** `thesis_outputs/tables/` + `thesis_outputs/figures/`

### Tables

**Table 1 — Per-participant, all 4 runs** (main results table, = Sold's Table 6.1)  
Source: `parallel_universe_metrics.csv` filtered to `Intra|NoNotch|NoArt|NoBurst|RF`  
Columns: PID | n_skip | R1–R4 Test Acc | R1–R4 Train Acc | R1–R4 ΔAcc | R4 Test Recall | R4 95% CI | R4 Sig↑  
Footer: mean ± SD

**Table 2 — Aggregate progression** (one row per run)  
Source: `stats_summary.json`, `notch_ablation_result.json`, `logo_confusion.json`, `parallel_universe_metrics.csv`  
Columns: Run | Window config | RF config | Mean Train Acc | Mean Test Acc | ΔAcc | Mean Test Recall | Mean EI Recall | Wilcoxon vs chance (W, p, r) | RF vs EI (p) | NoNotch vs Notch (Δ, p) | LOGO Acc | LOGO Recall

**Table 3 — All statistical tests** (appendix B)  
Source: `master_results.csv`, `notch_ablation_result.json`, `sensitivity_comparisons.csv`, `electrode_ablation.csv`  
Columns: Test | Run | W | p | r | Sig | Direction

**Table 4 — LOGO-CV per-participant, R3 + R4**  
Source: `logo_results.csv` from R3 and R4  
Columns: PID | n_test | LOGO Acc | LOGO Recall | TN/FP/FN/TP

**Table 5 — Feature importance top-20, R4** (main text); full ranking R2/R3/R4 in Appendix A  
Source: `feature_ranking_112.csv`  
Columns: Rank | Feature | Mean Gini | SD | Electrode | Band

### Figures

**Fig 1 — Intra-subject progression, 3 panels** (Test Acc / Test Recall / ΔAcc, x=R1–R4, error bars ±1SD, chance line, significance stars)

**Fig 2 — LOGO-CV progression, 2 panels** (Acc / Recall, x=R1–R4, note recall bias in caption)

**Fig 3 — R1 vs R4 per-participant scatter** (x=R1 test acc, y=R4 test acc, diagonal=no change, colour=significance status from `per_participant_significance.csv`)

**Fig 4 — Electrode × band heatmap, 4 panels** (4×7 grid per run, source: `electrode_band_heatmap.json`, per-panel normalised)

**Fig 5 — Notch ablation R4, 2 panels** (paired dot plot + delta histogram, source: `notch_ablation.csv` R4)

**Appendix A Figs A.1–A.3** — Top-20 feature bar charts for R2, R3, R4

---

## STEP 2 — Remaining thesis tasks (verbatim, priority order)

1. Full results tables and figures + appendix (feature_importance_full_table.csv etc.)
2. Raw results reference doc — single source of truth for all numbers
3. GitHub cleanup and upload
4. Formal notation/definition framework through methods
5. Participants + experimental protocol into methods_and_results.md
6. Convert literature to continuous prose, remove search queries — justify every parameter (e.g. 1.0 s window belongs in Methods, not Results)
7. Fill all citations
8. Full discussion + conclusion chapter (9+ pages, per-component limitations embedded)
9. Formal limitations section (L1 no artefact rejection, L2 motor artefact confound, L3 active baseline, L4 keypress jitter, L5 EI vs RF protocol mismatch, L6 homogeneous sample, L7 paid/unpaid confound — embedded per chapter + final chapter)
10. Full formatted bibliography
11. Remove all TODO/NOTE/INTERNAL comments and [Insert...] placeholders
12. Add Section 1.1 Thesis Overview + Section 1.2 Contributions (STAY paradigm, temporal firewall CV, data leakage finding, notch reversal, 4-stage progression)
13. Rewrite abstract — headline numbers from R4: mean recall 75.2%, LOGO-CV 65.9%
14. Write proper acknowledgements
