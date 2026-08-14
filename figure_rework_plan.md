# Figure Rework — Audit & Plan

> **STATUS: COMPLETE.** Both phases implemented and verified. Figures regenerated in
> `run_20260611_220844`; `thesis.tex` recompiled cleanly. Re-run at any time with
> `python 05-12_data_analysis_and_results/regen_thesis_figures.py`.
>
> Verification: all 21 generated `.tex` files are byte-identical to the hand-edited
> backup (zero regressions), all 24 target figures regenerate, and the only files
> changed in the run folder are PNGs.

Scope: the 19 `% ADD:` markers in `latex_thesis_pdf/thesis.tex`, covering 24 figure files.

Backup of everything a re-run can overwrite:
`05-12_data_analysis_and_results/runs/run_20260611_220844_BACKUP_20260813_142706/`
(contains `viz/`, `results/`, and all root-level artifacts — 23 MB. The 8.3 GB
`processed/`, `windows/`, `features*/`, `splits/` are untouched by any step in this plan.)

---

## PHASE 1 — Manual text edits to port back into the pipeline

### Method

The generated `.tex` files that `thesis.tex` `\input`s live inside the run folder and are
tracked in git. Manual edits are identifiable as commits that touch a `.tex` file **without**
touching its sibling `.png` (a real pipeline re-run would change both).

Two such commits exist: `6d0e1fd` and `9615e30`.

Ground truth was then established empirically: step 17 was re-run into a scratch run
directory (symlinked to the real data, real `viz/`) and every output diffed against the
committed files.

### Result

Of the three classes of manual edit found in git history, **two are already in the code**:

| Manual edit | Status |
|---|---|
| `\begin{table}[ht]` → `\begin{table}[H]` | Already in code — `viz17_diagnostics.py:227,344`, `viz17_5_top_vs_ei.py`, `:596` |
| `\textcolor{green}{}` removed where the value is significant but *worse* than the comparison | Already in code — `viz17_diagnostics.py:258` gates green on `mean_val > np.mean(comparison)` |
| `-subject` → `-Subject`, and `(Intra)`/`(Inter)` → `(Intra-Subject)`/`(Inter-Subject)` | **NOT in code — must be ported** |

So exactly **8 caption strings** would regress on a re-run, from **2 code sites**:

1. `scripts/viz17_diagnostics.py:279`
   `Step 1 ({scale.capitalize()}-subject)` → `Step 1 ({scale.capitalize()}-Subject)`
   affects `viz17_1_intra.tex`, `viz17_1_inter.tex`

2. `scripts/viz17_5_top_vs_ei.py:246`
   `Step 5 ({scale})` → `Step 5 ({scale}-Subject)`
   affects `viz17_5_{intra,inter}_feat{1,2,3}.tex`

Everything else — `[H]`, all green highlighting, every number — the current code reproduces
byte-identically. Verified: only these 8 lines differed across all 21 regenerated `.tex` files.

`viz03/burst_stats.tex` regenerates byte-identical; no manual edits there.

---

## Reproducibility check

All 24 target figures were regenerated into a scratch directory and compared byte-wise
against the committed originals. **All 24 are byte-identical.** The pipeline is deterministic
for everything in scope.

Five non-target files differ run-to-run (unseeded jitter/bootstrap):
`viz02_artifact_flag.png`, `viz05c_exploration_band_power.png`, `viz05d_exploration_stats.png`,
`viz17_{intra,inter}_final.png`, and `viz17_0.1.4_cohend_interactions.png`.

`viz17_0.1.4` **is** a target figure (Fig. `cohend_interactions`) — its bootstrapped 95% CI
error bars are drawn by seaborn without a fixed seed, so the whiskers shift slightly on every
run. Recommend seeding it so the figure is reproducible.

---

## Regeneration mechanism — the `--resume` plan does not work

`run.py --resume` after deleting `step16.done`/`step17.done` only reaches **7 of the 24**
figures. The mapping:

| Source | Figures | Reachable by resuming step 17? |
|---|---|---|
| `viz17_feature_ranking.py` | `viz17_0.1.1`, `0.1.2`, `0.1.3`, `0.1.4` | yes |
| `viz17_4_identifying_top.py` | `viz17_4_identifying_top` | yes |
| `viz17_diagnostics.py` | `viz17_2_inter_significance` | yes |
| `viz17_window_exploration_line.py` | `viz17_window_exploration_line` | **no — disabled, see below** |
| `viz01_preprocess_viz.py` | `viz01.3` | no — step 01 |
| `viz02_artifact_flag_viz.py` | `viz02.1` | no — step 02 |
| `viz03_label_windows_viz.py` | `viz03.3`, `viz03.3.2_3s`, `viz03.3.3_3s`, `viz03.3.3_5s` | no — step 03 |
| `viz04_balance_split_viz.py` | `viz04.1`, `viz04.2_before/after_balancing`, `viz04.4`, `viz04.5` | no — step 04 |
| `viz05_feature_engineering_viz.py` | `viz05a.A2`, `viz05a.A3` | no — step 05 |
| **no generator in repo** | `viz03_exploration_7`, `_9`, `_12` | **no — see below** |

Deleting `step01.done`…`step05.done` would re-run preprocessing, windowing, balancing and
feature extraction — hours of compute, and it rewrites `processed/`, `windows/`, `splits/`,
`features/`, i.e. the data every number in the thesis rests on. Not worth the risk.

**Instead**: a single standalone regeneration script that calls only the viz modules against
the existing run directory. This is the pattern already used by `scripts/regen_viz03.py` and
`fix_gen.py`. Proven above: every module runs standalone and reproduces its figures exactly.

---

## Two blockers needing a decision

### A. `viz03_exploration_7/9/12` have no generator in the repo

These three carry the first three `% ADD` markers. They are crops of
`viz/viz03_label_windows_exploration.png`, produced by `split_image.py`, which only *reads*
that composite. **No script in the repo (or anywhere in git history) generates it.**

The underlying data does still exist, so a faithful generator can be written:
`exploration/univ_*/windows/primary/P4.pkl` holds `windows` (`start_time`, `end_time`,
`label`) and `a_press_times`. Panel→universe mapping confirmed from the titles:

- `_7` → `univ_a3.0_g2.0_w0.3` (ws = 0.3 s) — thesis "reduced window size"
- `_9` → `univ_a3.0_g2.0_w1.0` (defaults) — thesis "default parameters"
- `_12` → `univ_a4.0_g0.5_w1.0` (gs = 0.5 s, isp = 4 s) — thesis "reduced gap, extended ISP"

### B. `viz17_window_exploration_line.png` is disabled and points at an archived copy

`viz17_window_exploration_line.py` returns immediately because
`config.ENABLE_WINDOW_EXPLORATION` is **not defined anywhere in `config.py`**
(`getattr(..., False)`). So the figure is never written to `viz/`, and the thesis points at
`viz/run_before_ei_fix/viz17_window_exploration_line.png` instead.

Recomputed from the current `exploration/` data, the values match the archived figure
**exactly** (gap sweep 0.1581/0.1642/0.1820/0.1991 …; window sweep 0.1911/0.1927/0.1958/
0.2101). The EI correction genuinely did not affect this sweep.

Consequence: if the flag is enabled and the figure regenerated into `viz/`, the
`\includegraphics` path must change, and the caption sentence "This figure originates from an
earlier pipeline run predating a correction to the Engagement Index computation…" becomes
false and should be deleted.

---

## PHASE 2 — Visual changes requested per figure

Verbatim from the `% ADD:` markers, grouped by generating script.

**`viz03_label_windows_viz.py`**
- `viz03.3` (L575) — all text 2×, consistent
- `viz03.3.2_3s` (L860) — all text 3×, consistent
- `viz03.3.3_3s` + `_5s` (L868) — all text 2×; stack the two figures vertically, full width

**`viz01_preprocess_viz.py`**
- `viz01.3` (L524) — X+Y axis labels, Y "in %"; all text 2×

**`viz02_artifact_flag_viz.py`**
- `viz02.1` (L557) — all text 2×

**`viz04_balance_split_viz.py`**
- `viz04.2_before/after` (L686) — all text 2×; X axis "Participant"; stack vertically, full width
- `viz04.5` (L706) — all text 2×
- `viz04.4` + `viz04.1` (L718) — all text 2×; stack vertically, full width; render the second
  as a line of colours rather than colour blocks (timeline-style)

**`viz05_feature_engineering_viz.py`**
- `viz05a.A2` (L592) — all text 2×; X axis "Statistical Moment"; lines at value instead of
  bars, keep the value printed above
- `viz05a.A3` (L609) — all text 2×; X axis "Band Powers"; same bars→lines change

**`viz17_feature_ranking.py`**
- `viz17_0.1.1` (L900) — all text 2×
- `viz17_0.1.2` (L916) — all text 2×; bars purple, not red (the EI line is already red)
- `viz17_0.1.3` (L930) — all text 2×; X axis "Statistical Moment" instead of "Stat"
- `viz17_0.1.4` (L954) — all text 2×; move legend inside the axes for compactness

**`viz17_4_identifying_top.py`**
- `viz17_4_identifying_top` (L966) — all text 2×; drop black bar borders; grey bars; move the
  right-hand annotation inside the bar, line-broken so it never overflows
  (e.g. "Top Band" / "Top Electrode" / "Overall Top Feature #2")

**`viz17_diagnostics.py`**
- `viz17_2_inter_significance` (L1001) — all text 2×; X axis "Participant", Y axis
  "Performance Metric"; consider transposing so metrics are columns and participants are rows

**new generator (blocker A)**
- `viz03_exploration_9` (L420) — X+Y axis labels
- `viz03_exploration_7` (L440) — X+Y axis labels; all text 2×
- `viz03_exploration_12` (L448) — X+Y axis labels; all text 3×

**`viz17_window_exploration_line.py`** (blocker B)
- (L456) — all text 3×; stack the two panels vertically, not side by side
