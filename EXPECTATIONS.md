# EXPECTATIONS.md — Scientific Rigor Requirements

> Derived from deep structural analysis of the two reference papers and mapped against the current thesis state.
>
> **Reference Paper 1 (RP1):** _"Online Learning for Wearable EEG-based Emotion Classification"_ — consumer-grade EEG (Muse S + Neurosity Crown), emotion classification, online learning, AMIGOS validation
> **Reference Paper 2 (RP2):** _"Brain activity forecasts video engagement in an internet attention market"_ (Tong et al. 2020, PNAS) — fMRI, video engagement, neuroforecasting, individual + aggregate prediction

---

## 1. Structural Comparison — How the Reference Papers Are Built

### 1.1 RP1 Structure (EEG Emotion Classification)

| Section | Content | Depth | Our Equivalent |
|---------|---------|-------|----------------|
| **Introduction** | Literature review of EEG emotion recognition → gap (no wearable real-time) → contribution statement | 3 pages, ~60 citations | ❌ Not yet written |
| **2.1 Existing Dataset** | AMIGOS: 40 participants, 16 videos, Emotiv EPOC 14ch, 128Hz, SAM ratings, preprocessing described | Full reproduction spec | ✅ Partial (survey demographics + exclusion mask) |
| **2.2.1 Participants** | N, sex, age (µ, σ), exclusion criteria, informed consent | 4 sentences, precise stats | ✅ Available from Steps 2,3 |
| **2.2.2 Data Acquisition** | Hardware specs (2 devices), channel positions (10-20), sampling rate, streaming protocol (Bluetooth→OSC), timestamp handling | ½ page, very technical | ⚠️ [GAP] — No dedicated analysis doc |
| **2.2.3 Stimuli Selection** | 16 clips from DECAF/MAHNOB-HCI, duration stats (µ, σ), valence-arousal quadrant mapping, why these specific clips | ½ page with figure | ⚠️ [GAP] — Naturalistic stimuli not formally documented |
| **2.2.4 Experimental Protocol** | PsychoPy-guided flow, instruction screenshots, questionnaire structure, self-paced navigation | ½ page with figure | ⚠️ [GAP] — Protocol not formally documented |
| **2.3 Preprocessing** | Band-pass 4-45Hz, re-referencing options tested (CAR, REST), artifact channels noted | ½ page | ✅ Partial (Step 1,4) |
| **2.4 Feature Extraction** | PSD via Welch's method, 5 frequency bands, window sizes tested (1-5s), features per window, formulas | 1 page with equations | ⚠️ [GAP] — Feature engineering not separately documented |
| **2.5 Classification** | 3 online classifiers (ARF, SRP, LR), hyperparameters, river library, progressive validation | 1 page | ✅ Steps 6,7,8,11 |
| **2.5.1 Evaluation Metrics** | Accuracy + F1 defined with TP/TN/FP/FN formulas, t-test for significance (α=0.05) | ½ page | ✅ Steps 9,12 |
| **Results** | Immediate label → Window size → Delayed label → Cross-device → Per-subject distributions | 4 pages, many tables | ✅ Steps 6-17 |
| **Discussion** | Summary → Literature comparison → Limitations → Future work | 2 pages | ❌ Not yet written |

### 1.2 RP2 Structure (Tong et al. 2020, PNAS)

| Section | Content | Depth | Our Equivalent |
|---------|---------|-------|----------------|
| **Significance Statement** | 2-sentence public impact summary | 2 sentences | ❌ Not yet written |
| **Introduction** | Theoretical framework (AIM: Affect-Integration-Motivation), specific hypotheses, neuroforecasting paradigm | 1.5 pages, 30 citations | ❌ Not yet written |
| **Results: Individual Prediction** | ROI analysis (NAcc, AIns, MPFC, PCC), onset/average/offset temporal windows, β coefficients + p-values + effect sizes | 1 page, 1 table | ✅ Steps 11,14,15 |
| **Results: Aggregate Forecasting** | Internet metadata extraction, view frequency + view percentage, neural vs behavioral vs combined models, R², AIC, CV RMSE | 1.5 pages, 1 table | ⚠️ Partial — no aggregate real-world validation |
| **Discussion** | AIM framework validation, scaling from individual→aggregate, practical implications, limitations | 1 page | ❌ Not yet written |
| **Materials and Methods: Participants** | N=40, demographics, exclusion criteria, IRB | 3 sentences | ✅ Steps 2,3 |
| **Materials and Methods: Task Design** | Video choice task + Video viewing task, trial structure (figure), rating scales (4-point Likert × 4 dimensions), counterbalancing, progress indicators omitted | ½ page + figure | ⚠️ [GAP] |
| **Materials and Methods: Behavioral Summary** | Trial completion rates, skip rates (74%), view duration (48%), missing data rates (1.32%), familiarity check | ½ page | ⚠️ [GAP] — No formal behavioral summary doc |
| **Materials and Methods: Neuroimaging** | Acquisition + preprocessing cited to prior work, data availability (OSF + Neurovault) | 2 sentences | ⚠️ [GAP] — Data acquisition undocumented |

---

## 2. Deep Characterization of Expectations — All Levels

### 2.1 Participant Documentation Expectations

**What the papers do:**
- RP1: Reports N, sex distribution, age range with µ and σ, exclusion criteria listed explicitly, informed consent statement
- RP2: Reports N=40, mentions 39 completed all trials / 1 had technical interruption, 38 watched ≥1 video to completion, 2 skipped every video

**What we have:** Steps 2 (survey_demographics) + Step 3 (exclusion_mask) cover demographics and exclusions.

**What is still needed for scientific rigor:**
- [ ] **Formal demographics table** with exact cell counts: age (M, SD, range), sex (N_F, N_M), handedness, education level — as a publication-ready table image
- [ ] **Exclusion flowchart** (CONSORT-style): Recruited → Screened → Excluded (with reasons branching) → Analyzed — this is the gold standard for reporting participant flow
- [ ] **Informed consent statement** — even a single sentence confirming ethical approval / consent procedure
- [x] Exclusion reasons documented (Step 3: P16, P19 keypress errors; P29 neurological disorder)

### 2.2 Data Acquisition Documentation Expectations

**What the papers do:**
- RP1: Exact device name + generation, electrode positions mapped to 10-20 system (with figure), sampling rate, bit resolution, streaming protocol (Bluetooth→Mind Monitor→OSC→Python), timestamp handling described
- RP2: References prior neuroforecasting studies for acquisition pipeline; provides data availability links (OSF, Neurovault)

**What we have:** Described in `THESIS_STRUCTURE.md` §3.2 but marked as `[GAP - No analysis documentation folder currently assigned]`.

**What is still needed for scientific rigor:**
- [ ] **Dedicated analysis documentation folder** for data acquisition with:
  - [ ] Exact hardware specification (Muse S Gen 2, 4 dry electrodes, Fpz reference, 256Hz)
  - [ ] Streaming stack diagram: Muse → Bluetooth → muselsl → LSL → Python recording script
  - [ ] Recording script version documentation (v4 features: auto-reconnect, real-time frequency monitoring)
  - [ ] Electrode placement figure (10-20 positions: AF7, AF8, TP9, TP10) — can reference RP1 Figure 1b directly since same device
  - [ ] Bluetooth disconnection handling: auto-reconnect mechanism, split CSV merging pipeline
  - [ ] Environment description: recording location, electrical shielding (or lack thereof), potential 50Hz exposure

### 2.3 Experimental Protocol Documentation Expectations

**What the papers do:**
- RP1: PsychoPy software, guided instructions, screenshots of experiment flow, questionnaire integration, self-paced navigation, stimuli presentation order
- RP2: Trial structure figure (Fig. 1), exact timing (4-8s viewing, 4s rating × 4 scales), counterbalancing of rating anchors, progress indicators omitted, forced minimum viewing (4s)

**What we have:** Described in `THESIS_STRUCTURE.md` §3.3 but marked as `[GAP - No analysis documentation folder currently assigned]`.

**What is still needed for scientific rigor:**
- [ ] **Dedicated analysis documentation folder** for experimental protocol with:
  - [ ] Trial structure timeline figure: Baseline_1 → TikTok Free Browsing → Baseline_2
  - [ ] Keypress marker protocol: A = video transition, B = baseline marker
  - [ ] Participant instruction script (what were they told?)
  - [ ] Stimulus description: "For You" feed (algorithmically personalized, uncontrolled content)
  - [ ] Session duration specification (target vs actual: ~30 min nominal)
  - [ ] Ecological validity justification vs controlled stimuli tradeoff (this is a key methodological contribution)
  - [ ] Pre-experiment survey instrument description

### 2.4 Labeling Strategy Documentation Expectations

**What the papers do:**
- RP1: SAM (Self-Assessment Manikin) 1-9 scale, threshold at 0.5 for binary high/low, valence and arousal dimensions independently
- RP2: Binary choice (watch/skip), continuous view percentage, 4-point Likert ratings (engaging-self, engaging-others, valence, arousal), mean-deviated within participant, projected onto positive/negative arousal axes

**What we have:** Described in `THESIS_STRUCTURE.md` §3.4 but marked as `[GAP - No analysis documentation folder currently assigned]`.

**What is still needed for scientific rigor:**
- [ ] **Dedicated analysis documentation folder** for labeling strategy with:
  - [ ] Formal definition of the 3-second pre-skip window with diagram
  - [ ] Justification citing motor preparation / readiness potential literature (Libet et al.)
  - [ ] Temporal alignment: exactly when does the 3s window start/end relative to keypress
  - [ ] Edge case handling: what happens when videos are < 3 seconds (rapid skipping)?
  - [ ] Class boundary diagram: visual showing where `about_to_skip` vs `not_about_to_skip` vs `baseline` labels fall on a timeline
  - [ ] Ground truth validity discussion: keypress ≠ neural decision moment (delay between intention and action)

### 2.5 Preprocessing Documentation Expectations

**What the papers do:**
- RP1: Explicit filter specifications (band-pass 4-45Hz), re-referencing tested (CAR, REST), mentions artifact handling, downsampling to 128Hz
- RP2: References prior work (ref 26) for neuroimaging preprocessing

**What we have:** Steps 1 (basic_quality_check) + Step 4 (recording_session_summary) cover quality but not preprocessing decisions.

**What is still needed for scientific rigor:**
- [x] Quality metrics per participant (Step 1)
- [x] Pass/fail criteria (Step 4: BT loss <20%, duration ≥20min, 100s baseline)
- [ ] **Formal preprocessing pipeline documentation**:
  - [ ] Butterworth filter specifications: order (4th), type (bandpass), per-band cutoff frequencies
  - [ ] Why no artifact rejection was applied (consumer-grade limitation — no EOG/EMG channels)
  - [ ] Rationale for NOT applying 50Hz notch filter (covered empirically in Step 11 vs notch, but needs formal documentation as a preprocessing decision)
  - [ ] Sampling rate verification across all recordings (mentioned in Step 1 but should be formalized)
  - [ ] Bluetooth dropout handling impact on signal continuity

### 2.6 Feature Engineering Documentation Expectations

**What the papers do:**
- RP1: Welch's PSD for each frequency band, explicit formula (Equation 1), window size experiments (1-5s), feature count per window stated
- RP2: ROI extraction from fMRI volumes, temporal segmentation (onset/average/offset), β-series correlation

**What we have:** Described in `THESIS_STRUCTURE.md` §3.6 but marked as `[GAP - No analysis documentation folder currently assigned]`.

**What is still needed for scientific rigor:**
- [ ] **Dedicated analysis documentation folder** for feature engineering with:
  - [ ] Formal definition of the 112-feature vector: 7 bands × 4 channels × 4 statistics
  - [ ] Why these 4 statistics (mean, std, min, max) — what does each capture neurophysiologically?
  - [ ] Frequency band definitions table with neurological correlates and citations
  - [ ] Windowing parameters: 3s window, 80% overlap (stride=0.6s), interpolation to 768 timesteps (256Hz × 3s)
  - [ ] Baseline normalization formula: how raw band power is converted to relative power against 100s resting state
  - [ ] Feature distribution visualization: are features approximately normal? Any clipping/saturation?

### 2.7 Model Architecture & Training Documentation Expectations

**What the papers do:**
- RP1: 3 classifiers (ARF, SRP, LR), all hyperparameters listed, library specified (river), progressive validation explained, training paradigm (online learning, mini-batch size=1)
- RP2: GLM with standardized coefficients, multivariate regression, model comparison (AIC, CV RMSE), separate models for behavior / ratings / brain / combined

**What we have:** Steps 6-8, 11 cover RF training; `THESIS_STRUCTURE.md` §3.8 lists DT/RF/Transformer but marked as `[GAP]`.

**What is still needed for scientific rigor:**
- [ ] **Dedicated analysis documentation folder** for model architecture with:
  - [ ] All three model architectures formally specified with hyperparameter tables
  - [ ] Decision Tree: max_depth=5, min_samples_leaf=10, class_weight=balanced
  - [ ] Random Forest: n_estimators=200, max_depth=7, min_samples_leaf=5, class_weight=balanced, random_state=42
  - [ ] Transformer: number of layers, heads, d_model, positional encoding type, optimizer, learning rate schedule
  - [ ] Data splitting protocol: 60/40 per-pool split, shuffling strategy, seed
  - [ ] Class balancing method: random undersampling of majority class
  - [ ] Training compute specifications (hardware: CPU/GPU/MPS, training time per participant)

### 2.8 Evaluation Strategy Documentation Expectations

**What the papers do:**
- RP1: Accuracy + F1 with TP/TN/FP/FN formulas written out, two-sided t-test (α=0.05), progressive validation, per-subject distributions shown (boxplots)
- RP2: Standardized β coefficients, SEs, t-statistics, p-values, adjusted R², AIC, CV RMSE, paired comparisons, supplementary analyses for robustness

**What we have:** Described in `THESIS_STRUCTURE.md` §3.9 but marked as `[GAP]`. Statistical tests done in Steps 9,12.

**What is still needed for scientific rigor:**
- [ ] **Dedicated analysis documentation folder** for evaluation strategy with:
  - [ ] Formal metric definitions (Accuracy, Precision, Recall, F1) with formulas
  - [ ] Why Recall is prioritized for BCI applications (formal justification)
  - [ ] Statistical test specifications: one-sample t-test vs 50% baseline, Cohen's d formula, 95% CI calculation
  - [ ] LOGO-CV procedure: formal algorithm description, what constitutes a "fold"
  - [ ] Multiple comparisons consideration: if testing 25 individual participants, is Bonferroni needed?
  - [ ] Effect reporting standards: exact p-values, confidence intervals, effect sizes — not just significance stars

### 2.9 Results Presentation Expectations

**What the papers do:**
- RP1: Per-subject F1 distributions (figure), comparison table across classifiers, per-device comparison, window size optimization curve, delayed label experiment, ablation of labeling strategies
- RP2: Bivariate correlations (scatter plots with r, ρ), multivariate regression table (Table 1 with β, SE, t, P for all predictors across 4 model types × 2 outcomes), supplementary robustness checks

**What we have:** Steps 6-17 cover results comprehensively. `THESIS_STRUCTURE.md` §4.1-4.4 structure the narrative.

**What is still needed for scientific rigor:**
- [ ] **Paired model comparison table** (DT vs RF vs Transformer) — currently DT and Transformer results exist individually but not in a single comparative table for n=25 with baseline normalization
- [ ] **Effect size for model comparison**: is RF significantly better than Transformer? Wilcoxon signed-rank test on paired per-participant accuracies
- [ ] **Confidence interval plots** for primary results (bootrap CI or t-distribution CI)
- [ ] Supplementary robustness:
  - [ ] Sensitivity analysis excluding flagged subgroups (ADHD n=?, medication n=?)
  - [ ] Effect of dataset size on performance (learning curve analysis)

### 2.10 Discussion Expectations

**What the papers do:**
- RP1: Explicit comparison to AMIGOS baseline (Table 3), device comparison (Muse vs Crown), limitations (data quality, small N for own experiment, online vs offline), future work (larger N, more devices, real-world deployment)
- RP2: Theoretical framework validation (AIM), scaling argument (individual → aggregate), practical implications (video marketing, content curation), limitations (sample demographics, video selection, ecological validity), data availability

**What we have:** `THESIS_STRUCTURE.md` §5 has a skeleton only.

**What is still needed for scientific rigor:**
- [ ] **Direct literature comparison table**: Our results vs at least 3-5 published EEG-BCI classification studies using consumer-grade devices
- [ ] **RP1 comparison**: Their Muse S results (F1 ~82% for emotion binary) vs our Muse S results (66% for skip binary) — discuss why different (controlled stimuli vs naturalistic, emotion vs micro-decision, online learning vs batch)
- [ ] **High-gamma controversy**: explicit engagement with literature on gamma artifacts vs genuine neural gamma in consumer EEG (cite Muthukumaraswamy 2013, Yuval-Greenberg 2008)
- [ ] **Ecological validity tradeoff**: position contribution relative to controlled-stimulus papers and explain what is gained/lost
- [ ] **Per-participant calibration implication**: position relative to zero-shot / transfer learning BCI literature

---

## 3. Identified Missing Analysis Documentation Steps

The following are analysis documentation steps that are scientifically expected but **do not yet have dedicated analysis_and_documentation folders**:

### 3.1 Steps Still Needed — Methods Documentation

| # | Proposed Folder Name | Purpose | Priority | THESIS_STRUCTURE Section |
|---|---------------------|---------|----------|--------------------------|
| 18 | `YYYYMMDD_HHMMSS_data_acquisition_protocol` | Formal documentation of hardware, streaming stack, recording script features, electrode placement, environment | HIGH | §3.2, §3.3 |
| 19 | `YYYYMMDD_HHMMSS_labeling_strategy` | Formal definition, timeline diagram, edge case handling, ground truth validity | HIGH | §3.4 |
| 20 | `YYYYMMDD_HHMMSS_feature_engineering_specification` | 112-feature vector formal definition, frequency band table, windowing math, baseline normalization formula | HIGH | §3.6 |
| 21 | `YYYYMMDD_HHMMSS_model_architecture_specification` | All three model families formally specified with hyperparameter tables | MEDIUM | §3.8 |
| 22 | `YYYYMMDD_HHMMSS_evaluation_protocol` | Metric definitions, statistical test specifications, why Recall, LOGO-CV algorithm | MEDIUM | §3.9 |

### 3.2 Steps Still Needed — Results / Ablation

| # | Proposed Folder Name | Purpose | Priority | THESIS_STRUCTURE Section |
|---|---------------------|---------|----------|--------------------------|
| 23 | `YYYYMMDD_HHMMSS_50hz_notch_filter_investigation` | Re-run baseline normalized RF **with** 50Hz notch filter on n=25. Currently only exists for n=3 feasibility data. | HIGH | §4.4.1 |
| 24 | `YYYYMMDD_HHMMSS_sensitivity_analysis` | Exclude flagged subgroups (ADHD, medication, caffeine) and compare aggregate results | MEDIUM | §3.1, §5 |
| 25 | `YYYYMMDD_HHMMSS_literature_comparison` | Direct numerical comparison to published EEG-BCI studies (RP1, others) | LOW | §5 |

### 3.3 Steps Still Needed — Supplementary

| # | Proposed Folder Name | Purpose | Priority |
|---|---------------------|---------|----------|
| 26 | `YYYYMMDD_HHMMSS_participant_flow_consort` | CONSORT-style participant flow diagram | MEDIUM |
| 27 | `YYYYMMDD_HHMMSS_behavioral_summary` | Pre-analysis behavioral overview: session durations, skip rates, video view durations — like RP2's "Behavioral Summary" | MEDIUM |

---

## 4. Paper-Derived Expectations: What a Reviewer Would Demand

### 4.1 From RP1 (EEG Emotion Classification)

| Expectation | What They Did | What We Must Do | Status |
|-------------|---------------|-----------------|--------|
| **Cross-dataset validation** | Validated on AMIGOS (40 participants) first, then own data (11 participants) | We use only our own dataset — must justify with statistical significance test | ✅ (Steps 9,12: p=6.29×10⁻⁹) |
| **Multiple device comparison** | Compared Muse S vs Neurosity Crown (4 vs 8 channels) | Not applicable (single device) — but should discuss 4-channel limitation | ❌ Needs documentation |
| **Window size ablation** | Tested 1s, 2s, 3s, 4s, 5s windows | We use 3s only — must justify (Libet / readiness potential literature) | ❌ Needs formal justification |
| **Per-subject F1 distributions** | Boxplots of per-subject performance | ✅ Have boxplots (Step 11) | ✅ |
| **Classifier comparison on same data** | ARF vs SRP vs LR on identical data pipeline | DT vs RF vs Transformer — need formal comparison table for n=25 | ⚠️ Partially done |
| **Feature extraction formulas** | Welch's PSD with equation | Must show Butterworth + aggregation formulas | ❌ Needs documentation |
| **Progressive / streaming evaluation** | Online learning with progressive validation | Not applicable (batch training) — but should discuss tradeoff | ❌ Needs documentation |

### 4.2 From RP2 (Tong et al. 2020, PNAS)

| Expectation | What They Did | What We Must Do | Status |
|-------------|---------------|-----------------|--------|
| **Theoretical framework** | AIM (Affect-Integration-Motivation) with specific hypotheses | Need theoretical grounding: attention allocation theory, readiness potential, media engagement models | ❌ Not documented |
| **Temporal specificity** | Tested onset vs average vs offset neural activity separately | We use fixed 3s pre-skip window — should discuss temporal resolution limits of consumer EEG | ❌ Needs documentation |
| **Individual → aggregate scaling** | Individual brain activity forecasts aggregate internet engagement | We do individual only — should discuss why aggregate is out of scope (content is personalized/uncontrolled) | ❌ Needs documentation |
| **Combined models** | Tested brain + behavior + ratings models, compared R², AIC, CV RMSE | We compare feature sets (112 vs EI-5) on same model — adequate for thesis scope | ✅ (Step 14) |
| **Behavioral summary statistics** | Trial completion %, skip rate (74%), view duration (48%), missing data rate (1.32%) | Must report: total skip events, mean session duration, BT dropout rates, inclusion rate | ⚠️ Partial (Steps 1,4,5) |
| **Data/code availability** | OSF + Neurovault links | Should document code availability (GitHub already mentioned in README) | ⚠️ Mentioned but not formalized |
| **Supplementary robustness** | 6+ supplementary tables (controlling for prior choices, familiarity, rank-ordered metrics) | Need at minimum: sensitivity analysis for subgroup exclusion, temporal autocorrelation check | ❌ Steps 24-25 above |

---

## 5. Critical Scientific Rigor Gaps — The Non-Negotiables

These are items that would cause a thesis examiner or reviewer to flag the work as insufficiently rigorous:

### 🔴 Must-Fix (Examination Risk)

1. **No formal preprocessing pipeline documentation** — Every EEG paper must specify exactly what signal processing was applied. Currently embedded in analysis scripts but not documented as a methods section.
2. **No window size justification** — Why 3 seconds? The readiness potential provides theoretical support but this is not formally documented with citations.
3. **No temporal autocorrelation analysis** — 80% overlap creates 0.6s stride between adjacent samples. While we shuffle after extraction, a reviewer will ask: "Did you verify there's no information leakage from overlapping windows ending up in both train and val sets?" This is partially addressed by the per-pool split strategy, but needs explicit documentation.
4. **No CONSORT flow diagram** — Standard for any study reporting participant exclusions.
5. **50Hz notch ablation only on n=3 feasibility data** — The thesis claims notch filter hurts, but this hasn't been demonstrated on the n=25 baseline-normalized pipeline.

### 🟡 Should-Fix (Strengthening)

6. **No formal paired statistical test between models** — RF vs Transformer comparison is descriptive ("similar accuracy") but lacks Wilcoxon signed-rank test on paired per-participant metrics.
7. **No sensitivity analysis** — Discussed in THESIS_STRUCTURE but never executed.
8. **No learning curve analysis** — How does performance change with dataset size per participant? This would inform the minimum calibration time for consumer BCI.
9. **No discussion of Type I error inflation** — 25 individual models tested × 4 metrics = 100 tests. Even with the aggregate t-test (which is valid), individual participant significance claims need care.

### 🟢 Would-Be-Nice (Excellence)

10. **No cross-validation within participant** — Currently single 60/40 split. Adding 5-fold CV within each participant would eliminate split-luck concerns.
11. **No effect of recording order** — Do participants perform differently in first vs last sub-recording (fatigue, habituation)?
12. **No feature stability analysis** — Are the top features consistent across participants, or does each participant use a fundamentally different feature subset?

---

## 6. Summary of Expectations by Thesis Section

| Section | Documentation Status | Missing Documentation Steps |
|---------|---------------------|---------------------------|
| 3.1 Participants | ✅ Mostly complete | CONSORT diagram, formal demographics table |
| 3.2 Data Acquisition | ❌ GAP | Step 18: hardware, streaming, environment |
| 3.3 Experimental Protocol | ❌ GAP | Step 18: protocol, instructions, timeline |
| 3.4 Labeling Strategy | ❌ GAP | Step 19: definitions, diagrams, justification |
| 3.5 Signal Preprocessing | ⚠️ Partial | Formal preprocessing pipeline spec |
| 3.6 Feature Engineering | ❌ GAP | Step 20: feature vector, formulas |
| 3.7 Windowing | ⚠️ Partial (Step 5) | Temporal autocorrelation documentation |
| 3.8 Classification Models | ❌ GAP | Step 21: formal architecture specs |
| 3.9 Evaluation Strategy | ❌ GAP | Step 22: metric definitions, LOGO-CV spec |
| 4.1 Model Comparison | ⚠️ Partial | Paired statistical test (Wilcoxon), unified table |
| 4.2 LOGO-CV | ✅ Complete (Step 13) | — |
| 4.3 Explainability | ✅ Complete (Step 15) | Feature stability across participants |
| 4.4.1 Notch Filter | ⚠️ Need n=25 | Step 23: re-run with notch on n=25 |
| 4.4.2 EI Analysis | ✅ Complete (Step 14) | — |
| 4.4.3 Skip Behavior | ✅ Complete (Step 16) | — |
| 4.4.4 Survey Correlation | ✅ Complete (Step 17) | — |
| 5. Discussion | ❌ Not started | Step 25: literature comparison |
