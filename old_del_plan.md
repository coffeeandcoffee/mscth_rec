This pipeline has 12 stages. Given the strict single-column flowchart requirement and the density of information per step, I'll split this into two diagrams — preprocessing through to feature engineering, then modelling through to outputs — with a brief connector between them.Steps 1–6 cover everything up to the point where you touch a model. Grid search is deliberately the last thing before training — no hyperparameters are touched after seeing any test data. Now the modelling and evaluation half:A few things worth calling out about the sequencing logic:

`step02_artifact_flag.py` flags but does not drop. The masking decision is applied at step03 window extraction time via `--artifact_mask {include|exclude}` — not at training time. This means the sensitivity analysis (artifact-included vs. artifact-excluded) produces two genuinely separate window manifests, and all downstream steps from step04 onward operate on whichever population step03 produced. Two runs differing only in this parameter are fully comparable.

`step06_grid_search.py` runs before any model training and uses only inner folds of the training data. This is the nested CV structure that prevents hyperparameter leakage. The locked `best_params.json` output is then consumed read-only by all subsequent training scripts — nothing re-tunes after this point.

`step08_train_ei.py` is its own script rather than a flag in `train_intra.py` because it uses a different classifier (LogisticRegression) and previously used a different protocol. Making it an explicit separate step with `--balanced true --cv temporal_blocked_5fold` hard-coded forces the protocol parity that makes the EI vs RF comparison valid. Note: this corrected protocol will produce different EI accuracy numbers than the 52.3% currently reported in the thesis (which used imbalanced data + standard 5-fold CV). The thesis EI result must be updated to whatever step08 outputs before submission.

Default: nonotch. Note: existing thesis reports 49.2% from a notch run — this run will produce different numbers; thesis text must be updated to step12 output before submission.

`step11_feature_importance.py` has the motor artifact warning flag built in — if the top-ranked features cluster on TP9/TP10 band dimensions, the script prints a warning to the report. This is Arnrich's question answered automatically rather than as a post-hoc narrative.

Note: the plan presents determinism as a blanket guarantee without flagging that parameter changes to early steps invalidate all downstream comparisons. Actually, the pipeline is deterministic within a single run once seeds are fixed at step04. Changing any parameter in steps 01–05 between runs invalidates downstream comparisons — use compare.py only between runs where the differing parameter is the intended variable. install this as a safety mechanism, and also use the following solution for comparing anything that forcefully requires changing parameters in the early steps: run.py determines the first step where parameters diverge from --from-run TIMESTAMP. All steps before that point are satisfied by symlinking to the prior run's outputs rather than re-executing - found as symlinks in the new output folders respectively for efficiency - careful to not make mistakes here ;). All steps from the divergence point onward execute fresh into the new timestamped folder. compare.py reads both runs' locked parameters.json files, identifies the divergence step automatically, and refuses to produce a comparison if the differing parameter is upstream of the step being compared — printing an explicit error: "Runs diverge at stepXX which is upstream of the requested comparison. Results are not comparable on this axis."

--

we will build the pipeline step by step.

before that, let me explain HOW it will be operated, which is the infrastructure around we first need to create to ensure multiple runs with parameter tracking and cleanly seperated folders for each run including all respective outputs and results and visualizations for that specific run:
1. there will be a run.py script that starts the pipeline
2. it creates a unique folder inside /Users/gregorlederer/Documents/MSc Thesis - EEG Neuroscience/Data Recording and Quality Tests/05-12_data_analysis_and_results/outputs/TIMESTEMP -> it uses the timestemp YYYYMMDD_HHMMSSat script start.
3. on run.py start after folder creation, inside that folder will be created a parameters.json filled with all possible parameters for each step that are by default all activated. the user can then manually adjust (terminal says "script paused, adjust parameters as needed in parameter.json and approve) on approving it changes the runtime parameter in that json "paramters_approved_by_user" from default FALSE to TRUE
4. then the first step gets executed and called from the run script, the run script monitors execution and also notices when that first step is done, and of course applies the parameters from the json that were predefined.
5. once that step is done, the timestemp id is passed on, and the step 1 's visualization script (this is a seperate script, each script has a visualization script that gets run afterwards, that visually describes what was done in that step not with words but with visualization, this must be kept super simple of course) gets executed saving the outputs inside the step 1 folder within results folder within the timestep id folder of this run. then step 2 starts automatically with respective parameters read from the json predefined and also gets again the timestemp id from previous run and so on so it always saves into correct folder of timestemp id.
6. it continutes like this chain until done.

this way, every new run of run.py is seperate and clearly defined by parameters. all results and visualizations get saved into that folder.

furthermore:
- in run.py we need the concept of "diff from previous run" or "copy last run's parameters." we will want a --from-run TIMESTAMP flag that pre-fills the JSON from a prior run's locked parameters, so the user only touches what changed.
- we need a compare.py script that takes two or more run timestamps as arguments and produces a paired comparison output - with numerical results and after a visualization run from those quantitative results. This is not a minor convenience — it's required for example for step09_ablation_notch.py, step10_significance.py's Wilcoxon inputs, and the thesis's central methodological contribution (the 8pp leakage finding). this comparison results need to also include the significance test and present an specific exact result if significantly different. main metric to compare by is the recall within each subject. (the ability and hitrate to predict STAY class correctly)
- run.py has no concept of a checkpoint or resume flag. If step07 crashes at participant 18 of 25, does the user re-run from step07 with the same timestamp? Or start a new run? This needs explicit handling.


--

exact flow chart of exact pipeline with exact functionality:

```mermaid
flowchart TD
    A([Raw data ingestion]) --> B

    B["**step01 — preprocess.py**
    --data_dir /raw  --out_dir /processed
    --fs_method median  --notch {on|off}  --baseline_clip 100
    Computes true fs via median delta-t
    Butterworth 7-band split: delta/theta/alpha/beta/low-gamma/high-gamma/very-high
    Active-baseline normalisation: P_rel = P_abs / mu_baseline
    Flags and logs Bluetooth dropout segments
    OUT: per-participant .pkl of relative power arrays + dropout log"]

    B --> C

    C["**step02 — artifact_flag.py**
    --in_dir /processed  --out_dir /clean
    --blink_thresh 150uV  --emg_thresh 50uV  --reject {flag|drop}
    Detects blink epochs on AF7/AF8 via peak-to-peak threshold
    Detects EMG bursts on TP9/TP10 via broadband power spike
    Does NOT remove epochs — appends binary artifact_mask column for downstream window-level filtering
    OUT: same .pkl + artifact_mask + artifact summary CSV"]

    C --> D

    D["**step03 — label_windows.py**
    --in_dir /clean  --out_dir /labelled
    --skip_window 3.0  --stay_stride 0.6  --min_stay_dur 4.0
    --burst_thresh 3.0  --burst_action {flag|exclude}
    --artifact_mask {include|exclude}
    If exclude: skips windows overlapping flagged artifact epochs before writing manifest
    Extracts 3s SKIP windows: 1 per keypress, terminates at keypress timestamp
    Extracts STAY windows only from videos watched >= min_stay_dur seconds
    Flags burst-skip epochs where inter-skip interval < burst_thresh
    OUT: windowed dataset .pkl + per-participant label summary CSV"]

    D --> E

    E["**step04 — balance_split.py**
    --in_dir /labelled  --out_dir /balanced
    --seeds 0 1 7 42 99  --split_method temporal_blocked
    --gap_s 3.0  --test_overlap none  --train_overlap 0.8
    Undersamples STAY to 50/50 for each seed independently
    Creates temporal 5-fold CV blocks with 3s inter-block gap
    Test folds: zero overlap. Train folds: 80% overlap allowed
    OUT: 5 balanced split manifests per participant per seed"]

    E --> F

    F["**step05 — feature_engineering.py**
    --in_dir /balanced  --out_dir /features
    --stats mean std min max  --bands all
    --electrode_sets full frontal temporal
    Computes 4 stats x 7 bands x 4 channels = 112 features (full set)
    Computes AF7/AF8-only subset: 56 features (frontal ablation)
    Computes TP9/TP10-only subset: 56 features (temporal ablation)
    Computes EI scalar: beta / (alpha + theta) averaged across all 4 channels
    OUT: three feature matrices + EI column per participant"]

    F --> G

    G["**step06 — grid_search.py**
    --in_dir /features  --out_dir /hparams
    --model rf  --feature_set full
    --param_grid n_est:[100,200,300] depth:[5,7,10] leaf:[3,5,10]
    --seed 0
    Nested CV: outer = temporal 5-fold, inner = 3-fold on training data only
    Uses seed 0 manifests only — hyperparams are selected seed-agnostically
    Selects best hyperparams per participant on inner folds only
    Locks params before any outer-fold evaluation: no leakage
    OUT: best_params.json per participant (seed-agnostic, applied to all 5 seeds in step07) + grid search report CSV]

    G --> H([continues in part 2])
```

---

```mermaid
flowchart TD
    A([from part 1 — grid search complete]) --> B

    B["**step07 — train_intra.py**
    --in_dir /features  --hparams_dir /hparams  --out_dir /results_intra
    --feature_sets full frontal temporal  --seeds 0 1 7 42 99
    --cv temporal_blocked_5fold  --metrics acc recall precision f1
    --artifact_mask {include|exclude}
    --eval_protocol {temporal_blocked|random_split|both}
    Default is temporal_blocked. When both is passed, step07 runs the random-split evaluation in addition to the blocked CV evaluation and writes both result sets into /results_intra with clearly labelled filenames (rf_summary_temporal_blocked.csv and rf_summary_random_split_legacy.csv).
    If --eval_protocol both: also runs random-split 60/40 evaluation and outputs labelled legacy results for leakage comparison figure; these numbers must never be used as primary results
    Trains RF per participant, per feature set, per seed
    Uses locked best_params.json from grid_search.py: no re-tuning
    Frontal-only and temporal-only runs provide electrode ablation
    If temporal features dominate accuracy: motor artifact flag raised
    OUT: per-participant accuracy matrix across seeds + confusion matrices"]
    --
    The visualization script for step07 then produces the side-by-side boxplot automatically when both files are present in the output folder — no separate step needed, and compare.py is not required for this particular figure since both evaluations live in the same run.

    B --> C

    C["**step08 — train_ei.py**
    --in_dir /features  --out_dir /results_ei
    --balanced true  --cv temporal_blocked_5fold
    --classifier logistic_regression
    Runs EI (beta / alpha+theta) on balanced data
    Uses identical CV protocol as RF: valid apples-to-apples comparison
    This is the methodological fix vs the original pipeline
    OUT: per-participant EI accuracy + EI summary CSV"]

    C --> D

    D["**step09 — ablation_notch.py**
    --in_dir /features  --hparams_dir /hparams  --out_dir /results_notch
    --notch_versions notch nonotch  --cv temporal_blocked_5fold
    Reruns full RF on both notch and no-notch preprocessed feature sets
    Both versions use blocked CV only: corrected evaluation throughout
    Paired Wilcoxon signed-rank across 25 participants: notch vs no-notch
    Reports effect size r alongside p-value
    OUT: ablation_comparison CSV (includes eval_protocol: temporal_blocked_5fold column on every row to prevent confusion with old random-split numbers) + Wilcoxon result + effect size r"]

    D --> E

    E["**step10 — significance.py**
    --in_dir /results_intra  --out_dir /stats
    --alpha 0.05  --correction bonferroni
    --trial_counts_dir /labelled (reads per-participant window counts from step03 output to supply N for binomial CI)
    --ablation_compare true
    Computes per-participant SD of recall across 5 seeds; reports mean SD across participants as seed stability index; threshold < 0.03 SD = stable (reported as confirmed stable in thesis language)
    Binomial CI uses per-participant N from label summary CSV produced by step03
    Pairwise Wilcoxon between full vs. frontal-only vs. temporal-only accuracy distributions across 25 participants; reports effect size r for each pair; if temporal-only >= full-set accuracy: appends motor artifact interpretation flag
    Wilcoxon signed-rank across 25 participants vs 50% chance baseline
    Reports W-statistic, exact p-value, rank-biserial r as effect size
    Per-participant binomial CI: one-sample test vs p=0.5 per trial count
    Separates 'above 50%' from 'statistically significantly above chance'
    OUT: stats_summary CSV + per-participant significance table"] + electrode_ablation_comparison CSV (3 pairwise tests + effect sizes + motor artifact flag if triggered) + seed_stability CSV (per-participant SD across seeds + mean SD + stability verdict)

    E --> F

    F["**step11 — feature_importance.py**
    --in_dir /results_intra  --out_dir /importance
    --cv temporal_blocked_5fold
    Extracts mean Gini importance per feature across all 5 CV folds × 5 seeds from the full feature set model only (feature_set=full, all seeds averaged)
    Aggregates to 7-band level per participant then across participants
    Friedman test across 7 bands; pairwise Wilcoxon with Bonferroni correction
    If TP9/TP10 features dominate ranking: appends motor artifact warning
    OUT: full 112-feature ranking table + 4x7 electrode-band heatmap data + band test CSV"]

    F --> G

    G["**step12 — logo_cv.py**
    --in_dir /features  --hparams_dir /hparams  --out_dir /results_logo
    --notch_version {notch|nonotch}  --balanced true
    Leave-one-group-out: train on 24 participants, test on held-out participant
    Reports accuracy AND full confusion matrix: not just recall
    Confusion matrix confirms whether model collapses to majority-class prediction
    OUT: LOGO accuracy + confusion matrix + per-fold breakdown CSV"]

    G --> H

    H["**step13 — compile_report.py**
    --results_dirs /results_intra /results_ei /results_notch
    /stats /importance /results_logo
    --demographics_csv /path/to/demographics.csv  (manually curated from survey data, lives outside run folder — age, sex, cohort, TikTok usage, dominant hand)
    Assembles Table 1: participant demographics (read from --demographics_csv, not from pipeline outputs)
    Assembles Table 2: per-participant dataset stats (skips, duration, windows)
    Assembles Table 3: full 112-feature importance ranking
    OUT: master_results.csv + all thesis tables as LaTeX-ready .tex files"]

    H --> I([pipeline complete])
```