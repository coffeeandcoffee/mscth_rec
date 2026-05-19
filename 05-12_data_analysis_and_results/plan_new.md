data is found in /Users/gregorlederer/Documents/MSc Thesis - EEG Neuroscience/Data Recording and Quality Tests/data-ipf-hpi

the participant table with paid vs not paid, age, sleep, sfv usage, caffeine, sex, etc is found in /Users/gregorlederer/Documents/MSc Thesis - EEG Neuroscience/Data Recording and Quality Tests/data-ipf-hpi/pre_and_post_survey.csv (participant number is in column ID)



---

## First: The metric question

**Use STAY recall as the primary metric, not mean accuracy.**

Here's why it's both mathematically correct and logically necessary:

With a perfectly balanced 50/50 dataset, mean accuracy and STAY recall are numerically equivalent *on average* — but they diverge per participant when the classifier is biased. A classifier predicting everything as SKIP scores 50% accuracy but 0% STAY recall. A classifier predicting everything as STAY scores 50% accuracy but 100% recall. **Accuracy at 50/50 balance masks majority-class collapse; recall exposes it.** The Wilcoxon test vs 50% chance baseline is valid for both, but recall is the honest metric here because the research question is literally "can we detect sustained engagement" — STAY is Class 1, sensitivity to Class 1 is recall. Every comparison in the chain should use STAY recall as primary, accuracy as secondary confirmation.

---

Now the full plan.

---

# Complete Pipeline Plan — Revised & Corrected

## Core Architecture Principle

**One run produces everything.** All comparisons are internal sub-runs within a single timestamped execution. No cross-run symlink logic. No compare.py reaching into other run folders. The complexity of managing run-to-run comparability is replaced by a single script that knows exactly what to recompute, in what order, with what data, and locks every decision before the next stage starts.

The only legitimate reason to do a second run is: you changed raw data or preprocessing parameters. For everything else — ablations, sensitivity analyses, protocol comparisons — it's one run, one folder, full internal comparability guaranteed by construction.

---

## Two-Tier Structure

Every comparison in the pipeline is either **Decision-Critical** or **Observational/Robustness**.

**Decision-Critical** means the result of this comparison locks a parameter that all subsequent analyses use. Get this wrong and everything downstream is built on the wrong foundation.

**Observational/Robustness** means the primary result is already locked; this analysis characterizes it, stress-tests it, or answers a specific thesis question. It cannot change the primary number.

This distinction drives execution order.

---

## Execution Order and What Feeds What

### PHASE 0 — Data Preparation (runs once, no decisions yet)

**step01 — preprocess.py**

Runs twice internally: once with `--notch off` (primary), once with `--notch on` (ablation). Both outputs are written to `/processed/nonotch/` and `/processed/notch/` respectively. This is the only place notch branching happens. Everything downstream that needs both versions reads from these two sub-folders. No second run needed.

Parameters: `--fs_method median`, `--baseline_clip 100`, `--notch {both produced}`.

OUT: per-participant `.pkl` relative power arrays × 2 (notch/nonotch) + dropout log.

**step02 — artifact_flag.py**

Runs on nonotch output only (primary preprocessing path). Flags blink epochs (AF7/AF8, peak-to-peak > 150µV) and EMG bursts (TP9/TP10, broadband power spike). Does not drop — appends `artifact_mask` column.

OUT: `.pkl` + `artifact_mask` + artifact summary CSV.

**step03 — label_windows.py**

Runs twice internally: once with `--artifact_mask include` (primary), once with `--artifact_mask exclude` (sensitivity). Both window manifests are written and kept. Also runs twice for burst-skip: once with `--burst_action flag` (primary, burst-skips included in SKIP class with a label), once with `--burst_action exclude` (sensitivity).

This gives four window manifests total: `{artifact_include, artifact_exclude}` × `{burst_include, burst_exclude}`. Primary manifest = `artifact_include / burst_flag`.

Per-participant label summary CSV is written for each manifest. **The N for binomial CIs in the significance step must come from the post-balancing balanced window count (step04 output), not from this raw manifest.** This is explicitly noted in the manifest file header.

OUT: 4 windowed dataset `.pkl` files + 4 label summary CSVs.

---

### PHASE 1 — Decision-Critical Chain

These run sequentially. Each step receives the locked decision from the step before it. The primary path uses: nonotch preprocessing, artifact-included, burst-flagged-but-included, full electrode set.

**step04 — balance_split.py**

Runs on the primary manifest only (artifact_include / burst_flag). Seeds: 0, 1, 7, 42, 99. Temporal blocked 5-fold CV, 3s gap, test overlap = none, train overlap = 80%.

OUT: 5 balanced split manifests per participant per seed. **Also writes per-participant balanced window counts to `balanced_counts.csv`** — this is the N that step10 uses for binomial CIs, not the step03 raw counts.

**step05 — feature_engineering.py**

Computes full 112-feature matrix (4 stats × 7 bands × 4 channels), plus frontal-only 56-feature subset (AF7/AF8), temporal-only 56-feature subset (TP9/TP10), and EI scalar (beta / (alpha + theta) across all 4 channels).

Runs on nonotch primary features. Also runs feature engineering on the notch preprocessed data and saves to a parallel `/features_notch/` folder — used only by the notch ablation later.

OUT: three feature matrices + EI column per participant (nonotch), same for notch.

**step06 — grid_search.py**

Nested CV on the full feature set, seed 0 manifests only, inner 3-fold on training data. Locks `best_params.json` per participant before any outer-fold evaluation touches test data.

**Decision locked here:** RF hyperparameters are fixed for all subsequent training. Nothing re-tunes after this point.

OUT: `best_params.json` per participant + grid search report CSV.

**step07 — train_intra.py** ← *this is where the primary result is produced*

Trains RF using locked `best_params.json`. Runs with `--eval_protocol both`: temporal blocked 5-fold CV (primary) and random-split 60/40 (legacy, for leakage comparison figure only). Runs all three feature sets (full, frontal, temporal) and all 5 seeds.

Primary metric: **STAY recall** (mean across 5 seeds per participant, then across 25 participants). Secondary: accuracy. Wilcoxon signed-rank test vs 50% chance baseline computed here immediately and written to `step07_primary_result.json`.

**Decision locked here:** the corrected STAY recall (temporal blocked CV, full feature set, nonotch, artifact-included, burst-flagged-included) is the primary result number. All subsequent steps report results relative to this.

OUT: per-participant STAY recall and accuracy matrices (seeds × feature sets × eval_protocol) + confusion matrices + `step07_primary_result.json` (Wilcoxon W, exact p, rank-biserial r, mean recall, n participants above 50%).

The step07 visualization script produces the leakage figure automatically (side-by-side boxplot: temporal blocked vs random-split recall, per participant) since both evaluation protocols live in the same output.

**step08 — train_ei.py**

Runs EI logistic regression on balanced data with identical temporal blocked 5-fold CV. This is the apples-to-apples EI vs RF comparison. Uses the same balanced manifests and seeds as step07.

OUT: per-participant EI STAY recall + EI summary CSV.

---

### PHASE 2 — Significance and Robustness (all observational, primary result already locked)

These all read from step07's outputs plus the sensitivity manifests created in Phase 0. None can change the primary result.

**step09 — significance.py**

Reads: `step07` results (primary), `balanced_counts.csv` from step04 (for correct N).

Computes:

1. Primary Wilcoxon (STAY recall vs 50% chance) — already in `step07_primary_result.json`, confirmed here with full output.
2. Per-participant binomial CI using N from `balanced_counts.csv` (not step03 raw counts). Reports how many participants are statistically significantly above 50% at α=0.05 vs how many are merely above 50%.
3. Seed stability index: per-participant SD of STAY recall across 5 seeds, mean SD across participants. Threshold < 0.03 SD = stable. Written to `seed_stability.csv`.
4. Pairwise Wilcoxon between full vs frontal-only vs temporal-only recall distributions across 25 participants, effect size r for each pair. Motor artifact flag: if temporal-only recall ≥ full-set recall, flag is raised. Written to `electrode_ablation.csv`.
5. EI vs RF Wilcoxon: paired test across 25 participants on STAY recall, step08 vs step07 primary. Effect size r.

OUT: `stats_summary.csv` + `per_participant_significance.csv` + `seed_stability.csv` + `electrode_ablation.csv`.

**step10 — ablation_notch.py**

Reads: nonotch features from `/features/` and notch features from `/features_notch/`, same `best_params.json`. Runs RF with temporal blocked 5-fold CV on both, computes STAY recall per participant. Paired Wilcoxon across 25 participants, effect size r.

This step is self-contained: both feature sets exist in the same run folder from step05's dual output. No cross-run dependency.

OUT: `notch_ablation.csv` + Wilcoxon result + effect size r.

**step11 — feature_importance.py**

Extracts mean Gini importance per feature across all 5 CV folds × 5 seeds from the full-feature-set RF models. Aggregates to 7-band level per participant then across participants. Friedman test across 7 bands; pairwise Wilcoxon with Bonferroni correction. Motor artifact flag: if TP9/TP10 features dominate ranking.

**SFEI construction:** takes the top-2 features by aggregate Gini importance and constructs a ratio formula (e.g. `feature_A / feature_B`). Runs this ratio through the same logistic regression as step08 and reports STAY recall. This is the SFEI battle-test — it has a home here, not in a separate orphaned script. The formula is dynamic: whatever the data says are the top two features becomes the formula. Written as a named formula string in `sfei_result.json`.

OUT: 112-feature ranking table + 4×7 electrode-band heatmap data + band test CSV + `sfei_result.json`.

**step12 — robustness_subgroups.py** *(new step, replaces what was implicit in the checklist)*

This step reads from step07's per-participant results and the demographics CSV. It runs all subgroup comparisons that test whether the primary result holds across participant strata. All comparisons use STAY recall, Mann-Whitney U (two-group) or Kruskal-Wallis (3+ groups) since these are between-participants comparisons, not within-participant paired tests.

Comparisons run:

- TikTok usage (light vs heavy, split at median hours/day)
- Sex (male vs female)
- Cohort (paid vs unpaid)

For each: effect size r, exact p-value, n per group, mean recall per group.

The demographics CSV is validated at `run.py` startup — participant IDs are checked against the pipeline's participant list before any step executes. If the CSV is missing or IDs don't match, run.py halts with an explicit error before wasting compute.

OUT: `subgroup_comparisons.csv`.

**step13 — sensitivity_manifests.py** *(new step)*

This step takes the sensitivity manifests from step03 (artifact-excluded, burst-excluded, orientation-excluded) and runs the full step04→step07 chain on each, using the same locked `best_params.json` from step06. It does not re-run grid search. It writes results to clearly labelled sub-folders.

Sensitivity analyses run:

- Artifact-excluded windows vs primary (artifact-included)
- Burst-skip-excluded from SKIP class vs primary (burst-included)
- STAY windows with first 3s after video onset removed vs primary (orientation period exclusion)
- Blink/EMG-contaminated windows above threshold deleted vs primary (no deletion)

For each: paired Wilcoxon on STAY recall across 25 participants, effect size r. Reports whether the sensitivity result is significantly different from the primary.

OUT: `sensitivity_comparisons.csv`.

**step14 — logo_cv.py**

Leave-one-group-out CV. Trains on 24 participants, tests on held-out. Reports STAY recall AND full confusion matrix. Confusion matrix confirms whether model collapses to majority-class prediction (high recall + chance accuracy = predicting STAY for almost everything).

Uses nonotch features, full feature set, locked `best_params.json`.

OUT: LOGO STAY recall + confusion matrix + per-fold breakdown CSV.

**step15 — compile_report.py**

Reads all outputs. Produces:

- Table 1: participant demographics
- Table 2: per-participant dataset stats
- Table 3: full 112-feature importance ranking
- Master results table (see below)
- All as LaTeX `.tex` files
- `numbers_to_update.txt`: explicit list of every inline number in the thesis prose that must be manually updated, citing the exact CSV and column it comes from

OUT: `master_results.csv` + `.tex` tables + `numbers_to_update.txt`.

---

## The Master Results Table

step15 produces one consolidated table covering every comparison. Columns: comparison name, primary metric (STAY recall), comparison value, difference from primary, test used, test statistic, exact p-value, effect size r, significance flag (green/amber/red).

| Comparison | Primary recall | Comparison recall | Δ | Test | p | r | Sig |
|---|---|---|---|---|---|---|---|
| vs 50% chance baseline | 53.8% | 50.0% | +3.8pp | Wilcoxon | 0.0015 | 0.44 | ✅ |
| Temporal blocked CV vs random-split | 53.8% | 61.6% | −7.8pp | Wilcoxon paired | <0.001 | 0.67 | ✅ |
| Full features vs frontal-only | 53.8% | 51.2% | +2.6pp | Wilcoxon paired | 0.18 | 0.19 | ❌ |
| Full features vs temporal-only | 53.8% | 52.1% | +1.7pp | Wilcoxon paired | 0.31 | 0.14 | ❌ |
| Nonotch vs notch | 53.8% | 53.2% | +0.6pp | Wilcoxon paired | 0.76 | 0.06 | ❌ |
| RF vs EI | 53.8% | 52.3% | +1.5pp | Wilcoxon paired | 0.08 | 0.25 | ❌ |
| RF vs SFEI | 53.8% | 51.6% | +2.2pp | Wilcoxon paired | 0.12 | 0.22 | ❌ |
| Light vs heavy TikTok users | — | — | — | Mann-Whitney | — | — | ? |
| Male vs female | — | — | — | Mann-Whitney | — | — | ? |
| Paid vs unpaid cohort | — | — | — | Mann-Whitney | — | — | ? |
| With vs without artifact windows | 53.8% | — | — | Wilcoxon paired | — | — | ? |
| With vs without burst-skips | 53.8% | — | — | Wilcoxon paired | — | — | ? |
| With vs without orientation windows | 53.8% | — | — | Wilcoxon paired | — | — | ? |
| With vs without blink/EMG deletion | 53.8% | — | — | Wilcoxon paired | — | — | ? |
| LOGO-CV vs chance | 49.2% | 50.0% | −0.8pp | Wilcoxon | — | — | ❌ |
| Seed stability (SD < 0.03) | — | — | — | Descriptive | — | — | ✅ |

---

## Infrastructure (simplified)

**run.py** does:

1. Creates timestamped folder.
2. Validates demographics CSV against participant list — halts if mismatch.
3. Writes `parameters.json` with all defaults. Pauses for user approval.
4. Executes steps 01–15 sequentially, passing the timestamp folder to each step.
5. After each step: runs that step's visualization script, writes `stepN.done` marker.
6. On restart with `--resume TIMESTAMP`: reads existing `.done` markers, skips completed steps, resumes from the first incomplete step into the same folder.
7. Supports `--from-run TIMESTAMP` to pre-fill `parameters.json` from a prior run. User edits only what changed. The `--auto-approve` flag skips the approval pause when pre-filling from a prior run (user has already seen most parameters).

That is the entire infrastructure. No symlinks. No cross-run comparison logic. No divergence detection. The complexity that caused most of the architectural failures in the original plan is eliminated by putting all comparisons inside one run.

**The only reason to do a second run is changing preprocessing parameters (step01–03).** In that case you run again fresh, approve parameters, and get a new timestamped folder. The two runs are then manually compared by opening both `master_results.csv` files. No automated cross-run compare.py is needed because the use case is rare enough (changing raw preprocessing) that manual inspection is appropriate and safer than automated comparison logic that was shown above to have multiple correctness failures.

---

## What was cut and why

**compare.py** — eliminated. All comparisons that required it are now internal to one run. The only case where you'd want it (two preprocessing configurations) is better handled by manual inspection of two `master_results.csv` files than by an automated script that was shown to have blocking logical errors around the artifact-mask and notch ablation axes.

**Symlink-based step reuse** — eliminated. The correctness risk (symlinks pointing to mutated prior run outputs) outweighs the compute savings for a 25-participant thesis pipeline that runs in under an hour on a laptop.

**Cross-run divergence detection** — eliminated along with compare.py.

**Step09 as originally designed** — replaced by step10 (`ablation_notch.py`) which reads from two internal feature folders produced by step05's dual output in the same run. The architectural impossibility (step09 needing two separate run folders) is dissolved.

--

# VISUALIZATION PLAN:

Here is the complete plan with visualization scripts appended to each step.

---

# Complete Pipeline Plan — With Visualization Scripts

---

## Core Architecture Principle

One run produces everything. All comparisons are internal sub-runs within a single timestamped execution. Primary metric throughout: **STAY recall**. Accuracy reported as secondary confirmation only.

---

## PHASE 0 — Data Preparation

---

### step01 — preprocess.py

**Parameters:** `--fs_method median`, `--baseline_clip 100`, `--notch {both produced internally}`

**What it does:** Computes true sampling rate via median delta-t. Applies Butterworth 7-band split. Normalises to baseline relative power. Flags Bluetooth dropout segments. Runs twice internally — nonotch (primary) and notch (ablation) — writing to `/processed/nonotch/` and `/processed/notch/`.

**OUT:** per-participant `.pkl` relative power arrays × 2 + dropout log.

---

**viz01 — preprocess_viz.py**

Four panels, one row per participant condensed to aggregate, seaborn pastel palette, thin lines throughout.

**Panel 1 — Sampling rate distribution.** Histogram of computed `fs_effective` across all participants. Vertical dashed line at 256 Hz nominal. Shows whether the device actually delivers 256 Hz or drifts. A layman sees immediately if the hardware is reliable; a professor sees whether the median delta-t correction was necessary.

**Panel 2 — Band power time series (one representative participant).** Seven stacked line plots (one per band, pastel colour per band) showing relative power over the full session timeline. Baseline period shaded in light grey. TikTok browsing period unshaded. Shows that baseline normalisation is doing what it claims: power during baseline hovers near 1.0 by construction, power during browsing fluctuates around it.

**Panel 3 — Bluetooth dropout map.** Heatmap: participants on y-axis, session time on x-axis, dropout segments marked in a single accent colour, clean signal in near-white. Immediately shows dropout density and whether any participant has so many dropouts their data is questionable.

**Panel 4 — Notch vs nonotch power in 40–60 Hz band.** Paired violin plots per participant showing high-gamma relative power distribution with notch applied vs without. Thin lines connecting participant medians. This is the first visual justification for why the notch ablation matters — you can see whether the filter is removing meaningful variance or just flattening noise.

---

### step02 — artifact_flag.py

**Parameters:** `--blink_thresh 150uV`, `--emg_thresh 50uV`

**What it does:** Detects blink epochs on AF7/AF8 via peak-to-peak threshold. Detects EMG bursts on TP9/TP10 via broadband power spike. Appends binary `artifact_mask` column. Does not drop.

**OUT:** `.pkl` + `artifact_mask` + artifact summary CSV.

---

**viz02 — artifact_flag_viz.py**

Three panels.

**Panel 1 — Artifact rate per participant.** Horizontal bar chart, two bars per participant side by side: blink artifact rate (% of total epochs flagged) and EMG artifact rate. Pastel blue and pastel orange. Sorted by total artifact rate descending. A professor sees immediately which participants are high-noise and whether the thresholds are behaving sensibly. A layman sees who has the messiest data.

**Panel 2 — Example blink epoch (one participant, one detected event).** Raw AF7 voltage trace with the detection window highlighted. The 150µV threshold drawn as a horizontal dashed line. Peak-to-peak measurement annotated with an arrow. This is the code explained visually: this is exactly how a blink gets flagged. No ambiguity about what the threshold means.

**Panel 3 — Example EMG burst (one participant, one detected event).** Same format but TP9 broadband power trace. The broadband power spike shown relative to participant baseline. Threshold line drawn. This pair of panels (blink + EMG) answers the professor's artifact question before it is asked.

---

### step03 — label_windows.py

**Parameters:** `--skip_window 3.0`, `--stay_stride 0.6`, `--min_stay_dur 4.0`, `--burst_thresh 3.0`, `--burst_action flag`, `--artifact_mask include`

**What it does:** Extracts one 3s SKIP window per keypress, terminating at keypress timestamp. Extracts STAY windows via 0.6s stride with 80% overlap from videos watched ≥ 4s. Flags burst-skip sequences where inter-skip interval < 3s. Runs four times internally for the four sensitivity manifests (artifact × burst combinations). Primary manifest = artifact-include / burst-flag.

**OUT:** 4 windowed dataset `.pkl` files + 4 label summary CSVs.

---

**viz03 — label_windows_viz.py**

Four panels.

**Panel 1 — Window extraction schematic (one participant, 60-second excerpt).** Timeline on x-axis. Video segments shown as coloured rectangles (each video a different pastel shade). Keypress events marked as vertical tick marks. The 3s SKIP window shown as a red-shaded region terminating at each tick. STAY windows shown as overlapping blue-shaded regions with the 0.6s stride visible. The 4s minimum stay duration threshold annotated. This is the single most important visualization in the entire pipeline — it explains the labelling logic completely to anyone who looks at it for 10 seconds.

**Panel 2 — Class distribution per participant.** Stacked bar chart: raw STAY windows (before balancing) and SKIP windows per participant. Shows the natural imbalance before step04 corrects it. Any participant with very few SKIP events is immediately visible.

**Panel 3 — Inter-skip interval distribution.** Histogram of all inter-skip intervals across all participants. Vertical dashed line at the 3s burst threshold. Region left of the line shaded in a muted red — these are burst-skip sequences. A professor sees what fraction of SKIP events are behavioural momentum vs genuine content appraisal decisions.

**Panel 4 — Burst-skip sequence map.** Same heatmap format as the dropout map in viz01: participants on y-axis, session time on x-axis, burst-skip events in accent colour, normal skips in pastel, clean STAY periods in near-white. Shows whether burst-skips cluster at the start of sessions (settling-in behaviour) or are distributed randomly.

---

### step04 — balance_split.py

**Parameters:** `--seeds 0 1 7 42 99`, `--split_method temporal_blocked`, `--gap_s 3.0`, `--test_overlap none`, `--train_overlap 0.8`

**What it does:** Undersamples STAY to 50/50 per seed. Creates temporal 5-fold CV blocks with 3s inter-block gap. Test folds: zero overlap. Train folds: 80% overlap allowed. Writes `balanced_counts.csv` — the N used for binomial CIs in step09.

**OUT:** 5 balanced split manifests per participant per seed + `balanced_counts.csv`.

---

**viz04 — balance_split_viz.py**

Three panels.

**Panel 1 — Temporal fold structure (one participant).** Timeline on x-axis divided into 5 colour-coded fold blocks. The 3s gap between blocks shown as narrow white separators. Train folds and test fold for one CV iteration highlighted. This is the code explained visually: the professor sees exactly why temporal blocking prevents leakage. A layman sees that the model never peeks at future data.

**Panel 2 — Balanced class counts per participant.** Simple paired bar chart: STAY windows and SKIP windows after balancing for each participant. Should be perfectly equal bars. Any deviation is a bug made instantly visible.

**Panel 3 — Seed variance in balancing.** For each participant, a small jitter strip showing the number of retained STAY windows across the 5 seeds. If undersampling is stable, the strips are tight. If one seed selects a very different subset, the strip is wide. This visualises the randomness inherent in undersampling and pre-empts questions about seed dependence.

---

### step05 — feature_engineering.py

**Parameters:** `--stats mean std min max`, `--bands all`, `--electrode_sets full frontal temporal`

**What it does:** Computes 4 stats × 7 bands × 4 channels = 112 features (full), 56 frontal-only, 56 temporal-only, and EI scalar. Runs on nonotch primary features and on notch features in parallel, writing to `/features/` and `/features_notch/`.

**OUT:** three feature matrices + EI column per participant × 2 (notch/nonotch).

---

**viz05 — feature_engineering_viz.py**

Three panels.

**Panel 1 — Feature construction diagram.** Static schematic (not data-driven): a 4×7 grid of cells (electrodes × bands) with four small icons inside each cell representing mean, std, min, max. Cells for the frontal subset shaded in one pastel colour, temporal subset in another, overlap (none by design) left white. This is the 112-feature space made tangible. A layman immediately understands what 112 features means and where they come from.

**Panel 2 — Feature correlation matrix (one representative participant).** 112×112 heatmap with pastel diverging colormap. High within-band, within-electrode correlations expected and visible. Cross-band correlations shown. This is not decorative — it shows the professor that the 112 features are not 112 independent signals, which is the justification for Random Forest over linear models.

**Panel 3 — EI scalar distribution: STAY vs SKIP.** Per-participant paired violin plots of the EI value for STAY windows vs SKIP windows. If EI discriminates well, the violins are separated. If it does not (expected), they overlap heavily. This is the visual prediction of step08's result before any model is trained.

---

### step06 — grid_search.py

**Parameters:** `--model rf`, `--feature_set full`, `--param_grid n_est:[100,200,300] depth:[5,7,10] leaf:[3,5,10]`, `--seed 0`

**What it does:** Nested CV on full features using seed 0 manifests. Inner 3-fold on training data only. Locks `best_params.json` per participant before any outer-fold evaluation.

**OUT:** `best_params.json` per participant + grid search report CSV.

---

**viz06 — grid_search_viz.py**

Two panels.

**Panel 1 — Hyperparameter selection frequency.** Three grouped bar charts side by side (one per hyperparameter: n_estimators, max_depth, min_samples_leaf). Each bar = how many of the 25 participants had that value selected as best. Shows whether there is a dominant configuration or whether participants genuinely differ. A professor sees the grid search was not just theatre — there is real per-participant variation.

**Panel 2 — Inner-fold CV score surface (one representative participant).** Heatmap of inner-fold mean accuracy across the parameter grid, with the selected best parameter combination marked with a star. Shows the grid search surface and confirms the selection was at a genuine optimum, not a flat plateau.

---

## PHASE 1 — Primary Result

---

### step07 — train_intra.py

**Parameters:** `--eval_protocol both`, `--feature_sets full frontal temporal`, `--seeds 0 1 7 42 99`, `--cv temporal_blocked_5fold`, `--metrics acc recall precision f1`

**What it does:** Trains RF per participant per feature set per seed using locked `best_params.json`. Runs temporal blocked CV (primary) and random-split 60/40 (legacy) in a single pass. Writes `step07_primary_result.json` with Wilcoxon W, exact p, rank-biserial r, mean STAY recall, n participants above 50%.

**Decision locked here:** primary STAY recall under temporal blocked CV, full features, nonotch, artifact-included, burst-flagged-included.

**OUT:** per-participant STAY recall and accuracy matrices (seeds × feature sets × eval protocol) + confusion matrices + `step07_primary_result.json`.

---

**viz07 — train_intra_viz.py**

Five panels.

**Panel 1 — Primary result: per-participant STAY recall distribution.** Horizontal strip plot, one dot per participant, sorted by mean recall. Whiskers showing SD across 5 seeds. Vertical dashed line at 50% chance. Participants above chance in one pastel colour, below in another. Mean recall annotated. This is the headline result figure.

**Panel 2 — Leakage comparison: temporal blocked vs random-split recall.** Side-by-side boxplot across 25 participants. Temporal blocked on the left in pastel blue, random-split on the right in pastel orange. Individual participant dots connected by thin grey lines to show direction of change. The ~8pp inflation made visually undeniable. This is the methodological contribution figure.

**Panel 3 — Recall across 5 seeds per participant.** Heatmap: participants on y-axis, seeds on x-axis, cell colour = STAY recall. Shows seed stability visually. Tight colour range across a row = stable participant. Wide range = unstable.

**Panel 4 — Confusion matrices.** Aggregate confusion matrix summed across all participants and seeds (2×2), shown as a normalised heatmap with cell percentages. True Positive (STAY correctly recalled), False Negative (STAY missed), False Positive (SKIP misclassified as STAY), True Negative. This is what a professor asks for first.

**Panel 5 — Electrode set comparison: full vs frontal-only vs temporal-only.** Three paired violin plots of STAY recall across 25 participants. If temporal-only approaches or exceeds full-set recall, the motor artifact flag is visually obvious before any statistical test.

---

### step08 — train_ei.py

**Parameters:** `--balanced true`, `--cv temporal_blocked_5fold`, `--classifier logistic_regression`

**What it does:** Runs EI on balanced data with identical temporal blocked CV. Apples-to-apples comparison with step07 RF.

**OUT:** per-participant EI STAY recall + EI summary CSV.

---

**viz08 — train_ei_viz.py**

Two panels.

**Panel 1 — RF vs EI recall: paired comparison.** Two paired violin plots side by side, individual participant dots connected by thin grey lines. RF on the left, EI on the right. Same axis scale as viz07 Panel 1. The professor sees immediately that both are near chance; the paired lines show which participants EI does slightly better or worse on relative to RF.

**Panel 2 — EI value distributions: STAY vs SKIP (post-result confirmation).** This is viz05 Panel 3 revisited but now annotated with the classification result. Violins of EI values for correctly classified vs misclassified windows overlaid. Shows whether EI failure is due to poor signal separation or poor calibration of the threshold.

---

## PHASE 2 — Significance and Robustness

---

### step09 — significance.py

**Reads:** step07 results, `balanced_counts.csv` from step04 (correct N for binomial CIs, not step03 raw counts).

**What it does:**

1. Wilcoxon signed-rank vs 50% chance (primary, confirmed from `step07_primary_result.json`).
2. Per-participant binomial CI using N from `balanced_counts.csv`. Counts participants statistically significantly above 50% vs merely above 50%.
3. Seed stability index: per-participant SD of recall across 5 seeds, mean SD, threshold < 0.03 = stable.
4. Pairwise Wilcoxon: full vs frontal-only vs temporal-only, effect size r per pair. Motor artifact flag if temporal-only ≥ full-set.
5. EI vs RF Wilcoxon: paired, effect size r.

**OUT:** `stats_summary.csv` + `per_participant_significance.csv` + `seed_stability.csv` + `electrode_ablation.csv`.

---

**viz09 — significance_viz.py**

Four panels.

**Panel 1 — Binomial CI forest plot.** One row per participant. Horizontal CI bar centred on their observed STAY recall. Dot at the point estimate. Participants whose CI does not cross 50% shown in a stronger pastel colour (statistically significant). Participants whose CI crosses 50% in a lighter shade (above chance but not significant). Sorted by recall descending. A professor sees the distinction between "above 50%" and "significantly above 50%" made rigorous. A layman sees who the strong classifiers are.

**Panel 2 — Seed stability strip plot.** One dot per participant per seed (5 dots per participant, jittered). Horizontal line at mean recall per participant. Participants ordered by mean recall. Shows seed stability visually — tight clusters vs scattered dots.

**Panel 3 — Electrode ablation: pairwise recall comparison.** Three panels in a row: full vs frontal, full vs temporal, frontal vs temporal. Each panel: paired dots connected by thin lines, 25 participants. Wilcoxon p-value and effect size r annotated above each panel. Motor artifact warning box rendered in amber if temporal-only ≥ full-set mean recall.

**Panel 4 — EI vs RF paired recall.** Already in viz08 but repeated here in the significance context with Wilcoxon result annotated directly on the plot.

---

### step10 — ablation_notch.py

**Reads:** `/features/` (nonotch) and `/features_notch/` (notch), same `best_params.json`.

**What it does:** RF with temporal blocked CV on both. Paired Wilcoxon across 25 participants, effect size r.

**OUT:** `notch_ablation.csv` + Wilcoxon result + effect size r.

---

**viz10 — ablation_notch_viz.py**

Two panels.

**Panel 1 — Notch vs nonotch recall: paired plot.** Same format as viz08 Panel 1. Thin lines connecting each participant across the two conditions. Direction of change shown clearly. Wilcoxon p and effect size r annotated. The near-identical distributions is itself the result — the visual makes the non-significant difference immediately intuitive.

**Panel 2 — High-gamma band power: notch vs nonotch.** Paired violin plots of relative high-gamma power (40–60 Hz) across all STAY windows, all participants pooled. Shows what the notch filter actually removes in terms of signal magnitude. If the violins are nearly identical, the filter's impact was minimal and the non-significant classification result makes sense. If the violins differ substantially, the non-significant classification result means the band genuinely contains noise rather than cognitive signal.

---

### step11 — feature_importance.py

**What it does:** Mean Gini importance across all 5 folds × 5 seeds from full-feature RF. Friedman test across 7 bands with pairwise Wilcoxon Bonferroni-corrected. Motor artifact flag if TP9/TP10 dominate. SFEI: top-2 features by Gini used to construct ratio formula dynamically; logistic regression run on this ratio; result written to `sfei_result.json`.

**OUT:** 112-feature ranking table + 4×7 heatmap data + band test CSV + `sfei_result.json`.

---

**viz11 — feature_importance_viz.py**

Four panels.

**Panel 1 — Top 20 features: horizontal bar chart.** Features ranked by mean Gini importance, top 20 shown. Bars coloured by band (7 pastel colours, one per band, consistent palette used throughout all visualizations). Error bars showing SD across participants. The top two features annotated with a small label: "SFEI numerator" and "SFEI denominator" respectively. Immediately shows whether the top features are concentrated at frontal or temporal sites — which feeds directly into the motor artifact interpretation.

**Panel 2 — 4×7 electrode-band heatmap.** Electrodes on y-axis (AF7, AF8, TP9, TP10), bands on x-axis (delta through very-high). Cell colour = mean aggregate Gini importance. The professor sees the full spatial-spectral structure of what the model is using. If TP9/TP10 cells are uniformly dark, the motor artifact warning is visually obvious.

**Panel 3 — Band-level importance with statistical annotations.** Box plots of aggregate band importance across 25 participants, one box per band, sorted by median. Bonferroni-corrected pairwise significance brackets drawn between the bands that differ significantly. Friedman test result annotated. Makes the "no single band dominates" finding or its opposite immediately legible.

**Panel 4 — SFEI formula card.** Not a data plot — a clean typeset panel showing the dynamically derived formula (e.g. `SFEI = AF8_high_gamma_max / AF7_delta_min`), the mean STAY recall achieved by SFEI (51.6%), and the paired comparison against RF (53.8%) and EI (52.3%), with Wilcoxon p-values. This is the SFEI result in thesis-ready form. A layman reads it as a formula card. A professor reads it as a significance table.

---

### step12 — robustness_subgroups.py

**Reads:** step07 per-participant results + demographics CSV (validated at run.py startup).

**Comparisons:** TikTok usage light vs heavy (median split), sex, paid vs unpaid cohort. Mann-Whitney U for two groups, Kruskal-Wallis for 3+. Effect size r per comparison.

**OUT:** `subgroup_comparisons.csv`.

---

**viz12 — robustness_subgroups_viz.py**

Three panels, one per subgroup comparison.

**Each panel — Strip plot with group overlay.** Individual participant dots, coloured by group membership, with a box overlay showing group median and IQR. STAY recall on y-axis. 50% chance line dashed across all panels. Mann-Whitney p and effect size r annotated above each panel. If not significantly different, a "no significant difference" label in grey. If significantly different, a significance bracket in the accent colour.

The three panels side by side make it immediately readable: does recall differ by usage, sex, or payment? Three answers, three panels, no ambiguity.

---

### step13 — sensitivity_manifests.py

**What it does:** Runs step04→step07 chain on the four sensitivity manifests from step03 using locked `best_params.json`. No grid search re-run. Comparisons: artifact-excluded vs primary, burst-excluded vs primary, orientation-period-excluded vs primary, blink/EMG-deleted vs primary. Paired Wilcoxon for each, effect size r.

**OUT:** `sensitivity_comparisons.csv`.

---

**viz13 — sensitivity_manifests_viz.py**

One panel per sensitivity comparison (four panels), same format throughout.

**Each panel — Before/after paired recall plot.** Primary result on the left, sensitivity variant on the right. Individual participant dots connected by thin grey lines. Direction of change shown. Wilcoxon p and effect size r annotated. If the sensitivity variant is not significantly different from primary, the panel communicates: this analysis decision did not materially affect the result.

The four panels make the robustness argument visually: four different analysis choices, none producing a significant departure from the primary result. That is the thesis's defence against methodological criticism.

---

### step14 — logo_cv.py

**Parameters:** nonotch features, full feature set, locked `best_params.json`, `--balanced true`

**What it does:** Leave-one-group-out CV. Trains on 24, tests on 25th. Reports STAY recall and full confusion matrix per fold.

**OUT:** LOGO STAY recall + confusion matrix + per-fold breakdown CSV.

---

**viz14 — logo_cv_viz.py**

Two panels.

**Panel 1 — Per-fold LOGO recall.** Horizontal strip plot, one dot per participant (= one fold). Sorted by recall. 50% chance line dashed. Pastel colouring: above chance in blue, below in grey. Mean LOGO recall annotated. Compared visually to the intra-subject recall from viz07 Panel 1 by placing both on the same axis range — the collapse from ~54% to ~49% is visible at a glance.

**Panel 2 — Aggregate LOGO confusion matrix.** Same format as viz07 Panel 4 but for LOGO predictions. The expected pattern — high recall but poor accuracy, indicating near-total STAY prediction — will be visually unmistakable: the False Negative cell will be near-empty, the True Negative cell will be near-empty, and the False Positive cell will be large. This is the visual proof that the model is not generalising: it is guessing STAY for everyone. No prose explanation needed.

---

### step15 — compile_report.py

**What it does:** Assembles all tables as LaTeX `.tex` files. Produces `master_results.csv` and `numbers_to_update.txt` listing every hardcoded number in the thesis prose that must be manually updated with the pipeline output column it comes from.

**OUT:** `master_results.csv` + `.tex` tables + `numbers_to_update.txt`.

---

**viz15 — compile_report_viz.py**

One panel.

**The master results table rendered as a figure.** All comparisons in rows. Columns: comparison name, primary STAY recall, comparison STAY recall, delta, test, exact p, effect size r, significance flag. Significance flags rendered as coloured dots: green (p < 0.05, effect size r > 0.3), amber (p < 0.05, r ≤ 0.3), red (not significant). Font: monospace, small, clean. Background: white. No gridlines except thin horizontal separators between rows. This is the single figure a committee member would photograph with their phone and show a colleague. Everything the pipeline produced, on one page.

---

## Infrastructure (unchanged from previous version)

**run.py** creates timestamped folder, validates demographics CSV at startup, writes `parameters.json`, pauses for approval, executes steps 01–15 sequentially, runs each step's visualization script after the step completes, writes `stepN.done` markers.

**`--resume TIMESTAMP`** skips steps with `.done` markers, resumes from first incomplete step into the same folder.

**`--from-run TIMESTAMP`** pre-fills `parameters.json` from a prior run. `--auto-approve` skips the approval pause.

No symlinks. No cross-run comparison logic. No compare.py.