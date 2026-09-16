# Literature memory — 2026-09-16 12:26 CEST

Verified verbatim evidence for 15 sources cited in `../thesis.tex`.
Written for a later processor that will read *other* sources and needs to fit
them into the existing argument without re-reading these PDFs.

**Provenance.** Every quote below was extracted from the source PDF in
`~/Zotero/storage/`, located via the `file` field of `../library.bib`, text
layer read with `pypdf`. Quotes are verbatim, including the source's own typos
and OCR artefacts where noted. Nothing here is paraphrase unless labelled as such.

**How to read an entry.** `SUPPORTS` = a thesis claim the quote carries in full.
`DOES NOT SUPPORT` = a claim the source was, or could be, asked to carry and
cannot. `LIMIT` = scope facts a later argument must respect (domain, N, task).
These distinctions are the point of the file — a quote reused outside its
`LIMIT` will produce an overclaim.

**Status of the thesis as of this file.** 138 pages, compiles clean, 0 undefined
refs/citations. All `precedent` wording removed; 22 `% CITE-NEEDED Pnn:` markers
sit in the source at the sites this evidence belongs to (`grep -n "CITE-NEEDED"
../thesis.tex`). Companion files: `../word_analysis/precedent_rewrite.md`,
`../word_analysis/source_audit.md`.

---

## 1. `rahman_detection_2025` — the closest published prior work

**Rahman, Mahmudur; Mahi, Atqiya Munawara; Sultana, Sharmin; Churpek, Matthew M; Alam, Mohammad Arif Ul.** *Detection of Short-Form Video Addiction with Wearable Sensors Via Temporally-Coherent Domain Adaptation.* 2025. Zotero `JSV3GIZR`.

**Cited in:** Lit. §Adjacent studies · §Libraries · §Delta-Band Function · Methods §Data Acquisition · Discussion §Neuroscience (5×)

> "We employed 10 student volunteers from a diverse demographic background as subjects and collected data while playing short-form videos from different platforms."

> "We used 3 most popular short-form video platforms, TikTok, YouTube Shorts, and Instagram Reels as the source of content. The volunteers were free to use their smartphones to visit the short-form video apps which ensures the media contents of their choice."

> "For every subject, we collected data from multiple sessions of continuous consumption of short-form videos in the segment of 15 minutes sessions. To establish the ground truth, we also collected data sessions in a 1:1 ratio of other daily life activities not involving any smartphone media consumption. We labeled short-form video consumption session as 1 and other activities as 0."

> "In the case of the Muse S sensor, we used the Mind Monitor app on the Android platform to collect and record the EEG data."

> "The Muse S gen 2 sensor has 4 channels of EEG sensing capability with 2 amplified auxiliary channels. All of the channels sampled data with a sampling rate of 256 Hz."

> "EEG signal complexes have a total of five major frequency bands spanning over the low to high-frequency spectrum respectively, Delta (1-4Hz), Theta (4-8Hz), Alpha (7.5-13Hz), Beta (13-30Hz) and Gamma (30-44Hz)."

> "To remove any power line noise, we applied a bandstop filter around 50Hz and its harmonics."

> "We used a k-fold cross validation scheme to choose 80% of the available data for pre-training and training, and then used 20% data for testing."

> "we found that, the Beta and the Gamma frequency bands contributed to high accuracy score of more than 80% among all of the five frequency bands … the Beta and Gamma frequency bands are more related to the brain stimulation caused by Short-form Video Addiction."

**SUPPORTS** — same hardware (Muse S, TP9/AF7/AF8/TP10, 256 Hz); naturalistic own-feed short-form video incl. TikTok; n=10; 1:1 block design; 80/20 k-fold; β and γ most informative >80%; band edges terminate at 44 Hz.
**DOES NOT SUPPORT** — anything about swipe-level or within-session labelling. Their contrast is *consuming short-form video at all* vs *not*, at session level.
**LIMIT** — trait-like, block-level, multi-modal (EEG + Empatica E4 EDA) domain-adaptation contribution. Their gamma stops at 44 Hz, i.e. *below* the thesis's high-gamma 40–60 Hz band, which is why the two sets of band findings are convergent but not comparable. Notch applied at 50 Hz and harmonics — the opposite of the thesis's un-notched choice.
**UNVERIFIED** — "ages 20–35" is in their Table 1, which did not extract from the PDF text layer.

---

## 2. `moontaha_online_2023` — the methods backbone

**Moontaha, Sidratul; et al.** *Online Learning for Wearable EEG-Based Emotion Classification.* 2023. Zotero `7G4MNK82`.

**Cited in:** Intro §Research Gap · Lit. §Parameter Choice (2) · §Sampling Boundaries (2) · §Sensitivity to Parameter Choice (2) · §Default Labeling Parameter Selection · Methods §Data Acquisition · §Operational Definition of Labeling Parameters · §Baseline Exclusion · §Parameter Sensitivity Analysis (2) · §Windowing and Class Labeling (14×, the most-cited methods source)

> "Since EEG data is considered stationary only over short time intervals, the preprocessing and the feature extraction were performed in tumbling windows with a fixed size and no overlap."

> "Different window length, l ϵ [1s, 2s, 3s, 4s, 5s] were tested on the AMIGOS dataset and the dataset from Experiment I to find the optimal one for the classification pipeline."

> "the best predictive performance was achieved with a window length of 1 second irrespective of the affect dimensions, classifiers and devices. Moreover, **in most cases** the classification performance is decreasing with increasing window sizes emphasizing the need of more data points."

> "The analysis on window length shows a clear trend of increasing performance scores with decreasing window length; therefore, a window length of 1 second is chosen for further analysis."

> "16 short clips (51-150 s long, µ = 86.7 s, σ = 27.8 s) depicting scenes from 15 different movies were used in the experiments for emotion elicitation."

> "the mean F1-Score across all participants achieved 63% for arousal and did not reach chance level for the valence classification."

> "ARF and SRP showed better performance with the mean F1-Score of more than 82% with no statistically significant difference (p > 0.05) in between." *(Experiment I, immediate-label setting)*

**SUPPORTS** — tumbling non-overlapping windows with whole-interval label assignment; boundary-straddling windows excluded; 1–5 s sweep run one-at-a-time; 1 s optimal; label delay of 86 s drops arousal F1 from >0.82 to 0.637 (≈20 pp) and puts valence below chance.
**DOES NOT SUPPORT** — the word **"monotonic"**. It appears 0 times in the paper. "In most cases" and "clear trend" both admit exceptions. *The thesis previously said "monotonic" in two places; corrected 2026-09-16.*
**LIMIT** — affect (valence/arousal) recognition from externally cued video stimuli, not self-paced action. 15 participants, two consumer devices (Muse, Neurosity Crown). The abstract's headline "87% arousal" is a different figure from the Experiment I mean of >82% — use the latter when quoting the delay drop.

---

## 3. `tzimourta_eeg_2019` — window length increases separability

**Tzimourta, Katerina D.; et al.** *EEG Window Length Evaluation for the Detection of Alzheimer's Disease over Different Brain Regions.* Brain Sciences 9(4):81, 2019. Zotero `4AXHMPU6`.

**Cited in:** Lit. §Parameter Choice · Methods §Parameter Sensitivity Analysis (3×)

> "The EEG signals are initially segmented in nonoverlapping epochs of different lengths ranging from 5 s to 12 s."

> "With regard to the window length, the results showed a high classification accuracy as the length of the window was gradually increasing, and the best classification results were obtained for epochs of 12 s."

Their Table 3 (accuracy % at 5,6,7,8,9,10,11,12 s), all six rows rising without reversal:
```
CN/AD             86.98 88.04 89.15 89.93 90.37 91.09 91.66 91.80
CN/mild           86.60 87.65 88.81 89.50 90.09 90.81 91.43 91.77
CN/moderate       94.68 95.13 95.64 95.99 96.18 96.46 96.56 96.76
CN-mild/moderate  92.59 93.27 93.78 94.06 94.29 94.70 94.88 94.99
mild/moderate     87.63 88.70 89.52 90.25 90.69 91.19 91.38 91.71
CN/mild/moderate  82.34 83.73 85.23 86.10 86.93 87.72 88.47 88.79
```

**SUPPORTS** — monotonic, reversal-free accuracy gain with window length, across **six classification problems**, whole-brain.
**DOES NOT SUPPORT** — any per-region window sweep:
> "Τhe rest of the analysis is conducted **solely for the 12-s window length**, which is the best classification window length according to the analysis."

The five brain regions were analysed at one window length only. *The thesis previously claimed the sweep held "across … five brain regions"; corrected 2026-09-16.*
**LIMIT** — clinical AD detection from resting recordings, subject-level trait label, no event locking. Shares only the signal-processing logic (more samples per estimate → more separable features).

---

## 4. `henao_isaza_comprehensive_2026` — Cohen's *d* for EEG feature ranking

**Henao Isaza, V.; et al.** *Comprehensive methodology for sample enrichment in EEG biomarker studies for Alzheimer's risk classification.* PLOS One, 11 Mar 2026, doi:10.1371/journal.pone.0343722. Zotero `64YEXHKY`.

**Cited in:** Lit. §Class Separation Metric · Methods §Class Separation Metric (2×)

> "Cohen's d values were utilized to quantify the magnitude of differences **between groups ACr and HC** for each specific metric."
*(ACr = asymptomatic PSEN1-E280A mutation carriers; HC = healthy controls)*

> "From nearly 967 initial features, the model first removed those with the highest correlations. **The decision tree algorithm then identified the 100 most important features for inclusion (Model Selection).** Cohen's d values were utilized to quantify the magnitude of differences between groups ACr and HC for each specific metric."

> "The pipeline generates spectral, connectivity, and entropy-based features across multiple frequency bands and independent components."

> "Feature selection was based on model performance **and** effect sizes (Cohen's d)." *(abstract)*

> "In the 2:1 ratio, relative power in components C7 and C10 within the Beta3 band showed very large effect sizes (Cohen's d = 1.22 and 1.12, respectively; p < 0.0001)."

**SUPPORTS** — Cohen's *d* used to rank EEG-derived spectral/connectivity/entropy features by group separation; licenses the phrase "group- and class-separability analysis".
**DOES NOT SUPPORT** — *d* as a selection criterion applied **prior to** model training. The methods order is: correlation pruning → decision-tree importance ranking → *d* characterising what was already selected. *The thesis previously said "prior to model/classifier training" in three places; corrected 2026-09-16.*
**CONFLICT INSIDE THE SOURCE** — the abstract ("selection was based on model performance and effect sizes") is looser than the methods. Treat the methods section as authoritative.
**LIMIT** — between-subject clinical group comparison with a diagnosis-fixed label. Not within-subject, not event-locked.

---

## 5. `kiarashi_sleep-derived_2025` — *d* responds to methodological choice

**Kiarashi, Yashar; Giannotto, Emily L.; Motie-Shirazi, Mohsen; Rodriguez, Amy D.; Levey, Allan I.; Clifford, Gari D.** *Sleep-Derived Features From Multi-Night In-ear EEG Identify Patterns Linked To Mild Cognitive Impairment.* 2025. Zotero `95X8663J`.

**Cited in:** Lit. §Class Separation Metric · Methods §Class Separation Metric (2×)

> "To assess how well each model distinguished **between MCI and CN groups**, we evaluated group separability using predicted probabilities from **leave-one-subject-out** cross-validation. Cohen's 𝑑 effect size measured the standardized difference between MCI and CN probability distributions, while Mann-Whitney tests (two-sided) assessed statistical significance."

> "Multi-night aggregation improved AUROC from 0.61 to 0.77, reduced within-subject variability 1.6-fold, and enhanced group separability based on Cohen's d improving from 0.65 to 0.96."

> "Multi-night averaging improved group separability across all methods. TinySleepNet showed Cohen's 𝑑 increase from 0.65 to 0.96, with Mann-Whitney test 𝑝-value improving from 0.208 to 0.0023. SleepTransformer demonstrated Cohen's 𝑑 increase from 0.56 to 1.08, with 𝑝-value improving from 0.080 to 0.0013. U-Time showed Cohen's 𝑑 increase from 0.18 to 0.53, with 𝑝-value improving from 0.599 to 0.021."

**SUPPORTS** — *d* as a **sensitivity metric**: a processing decision (multi-night averaging) moves *d* systematically while the underlying signal is unchanged, replicated across three embedding methods. This is the transferable structure, and the reason *d* is usable for the thesis's labelling-parameter sweep and not only for feature ranking.
**LIMIT** — sleep, in-ear EEG, MCI-vs-CN clinical contrast, LOSO. None of the domain transfers; only the metric behaviour does.

---

## 6. `cohen_statistical_1988` — the definition of *d*

**Cohen, Jacob.** *Statistical power analysis for the behavioral sciences.* Psychology Press, 1988. 579 pp. Zotero `THL2HAQM`.

**Cited in:** Lit. §Class Separation Metric · Methods §Class Separation Metric (2×)

> "Since both numerator and denominator are expressed in scale units, these 'cancel out,' and d is a pure number (here a ratio), freed of dependence upon any specific unit of measurement."

> "t is a 'pure' (dimensionless) number, one free of raw unit, as are also, for example, correlation coefficients or proportions of variance. Thus … the ES index for differences between population means is standardized by division by the common within-population standard deviation (σ)."

**SUPPORTS** — scale-invariance and dimensionlessness of *d*; comparability across heterogeneous measurement units. The thesis's "pure, dimensionless index" wording is near-verbatim Cohen.
**LIMIT** — this is the statistical definition only. It says nothing about EEG, feature ranking, or class separability in a classifier; those uses come from entries 4 and 5.

---

## 7. `hjorth_eeg_1970` — Mobility as a time-domain frequency estimate

**Hjorth, Bo.** *EEG analysis based on time domain properties.* 1970. Zotero `LNZ9MREI`.

**Cited in:** Lit. §Peak Frequency as a Time-Domain Frequency Estimator · Methods §Peak Frequency (2×)

> "1. Activity, giving a measure of the squared standard deviation of the amplitude, sometimes referred to as the variance or mean power. 2. **Mobility**, giving a measure of the standard deviation of the slope with reference to the standard deviation of the amplitude. It is expressed as a ratio per time unit and **may be conceived also as a mean frequency**. 3. Complexity, giving a measure of excessive details with reference to the 'softest' possible curve shape, the sine wave."

**SUPPORTS** — frequency content estimable directly in the time domain without a Fourier transform; Mobility is an analytically grounded mean-frequency estimate. All three Hjorth parameters as the thesis defines them.

---

## 8. `khambampati_frequency_2023` — zero-crossing rate as a frequency proxy

**Khambampati, et al.** *Frequency and Time Domain EEG Analysis for Prognostication of Postanoxic Comatose Patients.* 2023. Zotero `HT8V4I8V`.

**Cited in:** Lit. §Peak Frequency · Methods §Peak Frequency (2×)

> "Zero Crossing Rate (ZCR): ZCR is a measure of how often a signal changes its sign within a given frame. **This feature provides information about the frequency of rapid changes in the EEG signal. It quantifies the rate at which the signal crosses zero.** We calculate the number of zero-crossings and normalize it by dividing it by twice the length of the signal."

**SUPPORTS** — ZCR is explicitly defined as a frequency descriptor, in an EEG classification pipeline. Second independent line of precedent for time-domain frequency estimation.
**DOES NOT SUPPORT** — **speech**. The paper is EEG-only. No speech-processing source exists anywhere in `library.bib` (searched: zero-crossing, speech, mean frequency, time-domain → only Hjorth, Khambampati, `hajarian_gamification_2024`, the last of which never mentions ZCR). *The thesis previously said "in speech and EEG classification pipelines" in two places; "speech and" deleted 2026-09-16.*
**OPEN** — if a later processor wants the speech claim back, it needs a new source (e.g. Rabiner & Schafer, or an MFCC/ZCR speech-feature reference). Nothing in the current library can carry it.

---

## 9. `hasson_intersubject_2004` — free viewing works; higher-order cortex does not synchronise

**Hasson, Uri; Nir, Yuval; Levy, Ifat; Fuhrmann, Galit; Malach, Rafael.** *Intersubject Synchronization of Cortical Activity During Natural Vision.* Science, 2004. Zotero `4PXJ8CMB`.

**Cited in:** Lit. §Defining Engagement · Methods §Experimental Protocol · Discussion §Psychology · §What the Results Teach Us (4×)

> "We implemented this approach in the study of the functional organization of human cortex under free viewing of a long (30 min) uninterrupted segment taken from an original audiovisual feature film. Subjects were instructed to freely view the movie segment and report its plot at the end of the experiment."

> "Despite the free viewing and complex nature of the movie, we found an **extensive and highly significant correlation** across individuals watching the same movie. Thus, on average over 29% ± 10 SD of the cortical surface showed a highly significant intersubject correlation during the movie."

> "In addition to the highly synchronized cortex, we also found a pattern of areas which consistently failed to show intersubject coherence. These areas included the supramarginal gyrus, angular gyrus, and prefrontal areas. Thus, the 'collective' coherence effect naturally divides the cortex into a system of areas that manifest an across-subject, stereotypical response to external world stimuli versus regions that are **linked to unique, individual variations**."

**SUPPORTS** — (a) unconstrained free viewing is an established naturalistic-neuroimaging design; (b) higher-order association cortex incl. prefrontal fails to synchronise across individuals, attributed to individual variation — the load-bearing precedent for the thesis's cross-participant model collapse being *expected* rather than anomalous.
**NOTE ON WORDING** — the source's own term is *intersubject correlation*, not "cortical responses". *Thesis wording matched to source 2026-09-16.*
**LIMIT** — fMRI, passive film viewing, n=5 for the pairwise analysis. No EEG, no self-paced action.

---

## 10. `plucinska_impact_2022` — non-adjacent band peaks

**Plucińska, Renata; Jędrzejewski, Konrad; Waligóra, Marek; Malinowska, Urszula; Rogala, Jacek.** *Impact of EEG Frequency Bands and Data Separation on the Performance of Person Verification Employing Neural Networks.* Sensors 22(15):5529, 2022, doi:10.3390/s22155529. Zotero `RK98WJPW`.

**Cited in:** Discussion §Feature Importance and Neurophysiological Correlates (1×)

> "Among single frequency bands, the best outcomes were obtained for the beta frequency band (mean accuracy of 91 and 89% for the first and second scenarios, respectively)."

> "The β frequency band significantly differs only from the θ and δ frequency bands. **The second-best performance was found for the α and γ frequency bands.** Compared to the first scenario, the δ band performed worse. Again, the worst results were obtained for the θ band."

> "delta δ (1–4 Hz), theta θ (4–8 Hz), alpha α (8–12 Hz), beta β (12–30 Hz), and gamma γ (30–45 Hz). The upper border of the γ frequency band was set to 45 Hz to avoid signal contamination by the 50 Hz power line interferences."

**SUPPORTS** — band-wise performance peaks in **non-adjacent** ranges (β best; α *and* γ tied second; θ worst), i.e. not a smooth function of frequency. Precedent for the thesis's up–down–up separability profile.
**LIMIT — important.** The task is **person verification** (biometric identity), not cognitive-state decoding. *Thesis now names the task explicitly, 2026-09-16.* Also note their γ stops at 45 Hz for exactly the line-noise reason the thesis declines to notch — a useful contrast if a later argument revisits the notch decision.

---

## 11. `saltelli_how_2010` — ⚠ only a 3-page conference extract exists

**Saltelli, Andrea; Annoni, Paola; D'Hombres, Beatrice.** *How to avoid a perfunctory sensitivity analysis.* Procedia — Social and Behavioral Sciences 2(6):7592–7594, 2010, doi:10.1016/j.sbspro.2010.05.133. Zotero `N7WRMLJC`.

**Cited in:** Lit. §Parameter Choice · §Sensitivity to Parameter Choice (2×)

> "The most popular SA practice seen in the literature is that of 'one-factor-at-a-time' (OAT). This consists of analyzing the effect of varying one model input factor at a time while keeping all other fixed. While the shortcomings of OAT are known from the statistical literature, its widespread use among modellers raises concern on the quality of the associated sensitivity analyses. We introduce a novel geometric proof of the inefficiency of OAT … Alternatives to OAT are indicated which are based on statistical theory, drawing from experimental design, regression analysis and sensitivity analysis proper."

**CRITICAL PROVENANCE NOTE.** This PDF is **not truncated** — it is complete at 3 pages, because it is the conference-presentation extract from the *Sixth International Conference on Sensitivity Analysis of Model Output*. Its entire body is: the abstract, that same abstract repeated verbatim under the heading "1. Main text", one paragraph about the Stern Review, and the reference list. Its own bibliography points forward to the journal version:

> "Saltelli, A., Annoni Paola, 2010 How to avoid a perfunctory sensitivity analysis, **Revised for Environmental Modelling and Software**."

That journal article (Saltelli & Annoni, two authors, *Environmental Modelling & Software*) contains the geometric proof. **It is not obtainable** — the user confirmed every route leads back to this same 3-page PDF.

**SUPPORTS** — the definition of OAT; that OAT is statistically inefficient; that alternatives come from statistical experimental theory.
**DOES NOT SUPPORT** — the geometric argument itself ("axis-aligned lines through a single fixed operating point") or the claim that interactions are unidentifiable. *Thesis paraphrase removed 2026-09-16; the interaction point now stands on design logic with no citation, which is correct — if no two parameters are ever varied together, their interaction is not estimable by construction.*

---

## 12. `saltelli_global_2008` — OAT sampling, and locality of attribution

**Saltelli, Andrea; et al.** *Global sensitivity analysis: the primer.* John Wiley, 2008, doi:10.1002/9780470725184. 305 pp. Zotero `B4E2YMJQ`.

**Cited in:** Lit. §Sensitivity to Parameter Choice · Methods §Parameter Sensitivity Analysis (3×, one as `\cite[\S2.4.2]{...}`)

§2.4.2 "One-at-a-time (OAT) Sampling", pp. 66–70:

> "One way of simplifying X_Nk is to use a 'one-at-a-time' (OAT) design, where only one parameter changes values between consecutive simulations."

> "This equation demonstrates that if there is any change in value between y_i and y_i+1, it can only be attributed to a change in parameter x_i … The quantity Δy_i = y_i+1 − y_i is an estimate of the effect on y of changing X_i from 0 to 1. **It is applicable everywhere if the linear model is appropriate, and for some region around the current sample point otherwise.**"

**SUPPORTS** — the definition of OAT; and, via the italicised clause, the thesis's claim that **attribution is local to the operating point** at which the remaining parameters were fixed.
**DOES NOT SUPPORT** — interaction effects. §2.4.2 does not discuss them (checked: 0 hits for "interact" / "additive" in that section). It is **not** a valid re-home for the interaction claim removed from entry 11.

---

## 13. `rimbert_impact_2023` — distance from the event increases separability

**Rimbert, Sébastien; Trocellier, David; Lotte, Fabien.** *Impact of the baseline temporal selection on the ERD/ERS analysis for Motor Imagery-based BCI.* EMBC 2023, 45th Annual Int. Conf. IEEE EMBS, Sydney. HAL `hal-04077693`. Zotero `RV7Z63PY`.

**Cited in:** Lit. §Parameter Choice · Methods §Parameter Sensitivity Analysis (3×)

> "if two trials are separated by a few seconds, taking a baseline close to the end of the previous trial could result in an over-estimation of the ERD, while taking a baseline too close to the upcoming trial could result in an under-estimation of the ERD."

> "the closer the selected baseline/rest time window to the preceding trial, i.e. to the end of the previous MI, **the more global the ERD is (more electrodes show an ERD) and the stronger its amplitude** (p < 0.01)."

> "the closer the selected baseline/rest time window to the preceding trial … the higher the BCI performance. In contrast, if the selected resting time is close to the MI cue, the BCI performance is worse. BCI performance is significantly better for Baseline 1 compared to Baseline 3 (+4%, p < 0.05) or Baseline 4 (+8%, p < 0.001) for both right-hand MI and left-hand MI tasks."

Four baseline windows tested: `[-5;-2]s, [-4;-1]s, [-3;-0]s, [-2;+1]s`. n=71.

**SUPPORTS** — moving a labelled window further from a to-be-predicted motor event yields stronger, more spatially global ERD and higher classification accuracy. Structurally the same question as the thesis's gap-length and imminent-skip-period parameters. Verified exact — the thesis's "significantly stronger, more spatially global" is the source's own wording.
**LIMIT** — cued motor imagery, not self-paced action; the window in question is a *baseline*, not the class window.

---

## 14. `mi_exploring_2026` — the optimum is subject-specific

**Mi, Jian-Xun; Li, Rong-Feng; Liu, Ke; Li, Weisheng.** *Exploring multi-scale time group for common spatial pattern feature based motor imagery EEG classification.* Biomedical Signal Processing and Control 112 (Feb 2026), 108591, doi:10.1016/j.bspc.2025.108591. Zotero `D7MLEK94`.

**Cited in:** Lit. §Parameter Choice · Methods §Parameter Sensitivity Analysis (3×)

> "for subject A01T, when the time window is set to 0.5 s, the classification accuracy is only about 81%, and when the time window is set to 2.5 s, the classification accuracy can reach 94%. **The maximum gap between different time window sizes can reach about 13% for subject A09T.** More importantly, in the case of time window size constraints, for different subjects, the optimal classification accuracy time window size is not the same for different subjects when considering time window size constraints."

Datasets: BCI Competition IV-2a (subjects A01T–A09T) and IV-2b (nine subjects).

**SUPPORTS** — the accuracy-optimal window size differs between subjects; up to a 13% accuracy gap between window choices within a single participant; window-length effects are therefore not a universal law. This is the counterweight to entry 3 (Tzimourta).
**DOES NOT SUPPORT** — non-monotonicity *within* a single subject. Not asserted in the text; their worked example (A01T 0.5 s → 81%, 2.5 s → 94%) rises. It may be visible in their Fig. 6 but would have to be cited to the figure. *Thesis claim dropped 2026-09-16 and replaced with a description of their per-participant method plus an explicit statement that the present sweep is cohort-level and was not resolved per participant.*
**LIMIT** — motor imagery, CSP features, cued paradigm.

---

## 15. `frenay_classification_2014` — label noise concentrates at boundaries

**Frénay, Benoît; Verleysen, Michel.** *Classification in the Presence of Label Noise: A Survey.* IEEE Trans. Neural Networks and Learning Systems 25(5):845–869, May 2014, doi:10.1109/TNNLS.2013.2292894. 26 pp. Zotero `Y2FSGPLL`.

**Cited in:** Lit. §Sampling Boundaries · Methods §Baseline Exclusion (2×)

> "Statistical taxonomy of label noise …: (a) noisy completely at random (NCAR), (b) noisy at random (NAR) and (c) noisy not at random (NNAR)."

> "In Fig. 1(c), E depends on both variables X and Y, i.e. mislabelling is more probable for certain classes and in certain regions of the X space. This noisy not at random (NNAR) model is the most general case of label noise. For example, **mislabelling near the classification boundary or in low density regions can only be modelled in terms of NNAR label noise.**"

> "B. Consequences on Learning Requirements and Model Complexity. **Label noise can affect learning requirements (e.g. number of necessary instances)** or the complexity of learned models."

> "the consequences of label noise are important and diverse: decrease in classification performances, changes in learning requirements, increase in the complexity of learned models, distortion of observed frequencies, difficulties to identify relevant features, etc."

**SUPPORTS** — mislabelling is NNAR, concentrating near the classification boundary; label noise affects the number of instances required, not only accuracy. Together these justify excluding boundary-adjacent epochs *despite* the sample-count loss — the thesis's core reason for its safety gaps.
**WATCH THE STRENGTH** — the source says label noise **"can affect"** learning requirements; the thesis says it **"increases the number of training samples required to reach a given level of reliability"**. Directionally consistent with the whole section, but firmer than the sentence quoted. The follow-on inference ("a smaller but purer set … can outperform a larger set containing boundary-adjacent noise") is the thesis's own reasoning, appropriately hedged with "meaning that".
**LIMIT** — general machine-learning survey, not EEG-specific. Its authority is about label noise as such.

---

## ⛔ Do not mine the `\iffalse ... \fi` blocks

`../thesis.tex` contains ~235 lines inside `\iffalse ... \fi`. They do not
compile and **must stay that way**. They are not earlier drafts of the current
argument — they are **a different, superseded study**: a Random-Forest pipeline
whose results were replaced entirely by the present Cohen's *d* / logistic-regression
/ Engagement-Index benchmark.

Proof, by token count in `thesis.tex`:

| token | live thesis | inside `\iffalse` |
|---|---|---|
| `RF-112` | 0 | 4 |
| `53.8` (intra accuracy) | 0 | 4 |
| `61.4` | 0 | 3 |
| `49.2` (LOGO-CV) | 0 | 5 |
| `SFEI` | 0 | 8 |
| "Random Forest" | 0 | 8 |

No number from those blocks appears in the live thesis. Importing any of it would
contradict the reported results.

The blocks also contain the author's own unresolved working notes, e.g. *"There is
a logical error possibly: firstly we need to be clear about what our two classes
cover cause i think that there is a mismapping to what they mean. also, posting?
why posting? thats weird and out of context."* Text carrying that flag is not
evidence of anything.

**Rule for any later processor.** Treat `\iffalse` content as out of scope. If a
claim in there looks useful, do not lift it: re-derive it from a source verified in
this file, check it against the *live* text first (the live Discussion frequently
already makes the point, better hedged), and write it fresh. The scripts in
`../word_analysis/` already exclude these blocks by default; `--include-iffalse`
exists for auditing only.

**Worked example of the trap.** On 2026-09-16 the assistant proposed surfacing a
SKIP-heterogeneity paragraph from `\iffalse` to answer a reviewer objection. Wrong
on two counts: the paragraph rested on the superseded pipeline, and the live
Discussion §Behaviour already made the same point correctly — citing
`anderson_social_2023` and then explicitly bounding it ("That evidence concerns
posting, not swiping, and comes from a different behavioral paradigm … treated as
an open question rather than assumed"), which is exactly the correction the legacy
note had been asking for.

---

## Cross-cutting notes for a later processor

**Where the argument is thin and a new source would do most work.**
1. `fries_rhythms_2015` (gamma synchronisation → attentional selection) is cited **once** yet carries the whole mechanistic case for the AF7 high-gamma finding, one of three headline features. Same for `miller_integrative_2001` (prefrontal top-down control) and `mcmenamin_electromyogenic_2011` (EMG confound). These are the highest-value targets for additional support.
2. `pope_biocybernetic_1995-3` is cited 5× but is the origin of the Engagement Index, the instrument the entire benchmark is built against. Low citation density for the load it bears.
3. The **speech** half of the ZCR claim (entry 8) has no source in the library at all.

**Three claims in the thesis that are deliberately uncited**, because they assert an absence and an absence cannot be quoted. All three now read "to our knowledge": the labelling parameters having no established value for a self-paced swiping paradigm (Intro §Approach, Lit. §Default Labeling Parameter Selection, Conclusion). Do not "fix" these by attaching a citation.

**One claim deliberately uncited on logic, not evidence:** OAT cannot identify interaction effects, because no two parameters are ever varied together. See entries 11 and 12 for why neither Saltelli source can carry it.

**Domain-transfer ledger** — what each source may and may not be generalised to:
| source | domain | transfers as |
|---|---|---|
| rahman_detection_2025 | short-form video, session-level | hardware + setting precedent only |
| moontaha_online_2023 | cued affect video | windowing + label-timing method |
| tzimourta_eeg_2019 | clinical AD, resting | signal-processing logic only |
| henao_isaza / kiarashi | clinical between-subject | behaviour of Cohen's *d* only |
| cohen_statistical_1988 | statistics | definition of *d* only |
| hjorth / khambampati | EEG time-domain features | estimator precedent |
| hasson_intersubject_2004 | fMRI, passive film | free-viewing design + idiosyncrasy of higher-order cortex |
| plucinska_impact_2022 | EEG biometrics | band-profile shape only |
| saltelli ×2 | general sensitivity analysis | OAT definition + locality |
| rimbert / mi | cued motor imagery | window-placement effects |
| frenay_classification_2014 | general ML | label-noise structure |

**Provenance of this file.** All quotes machine-extracted and read in session on 2026-09-16 while removing the word "precedent" from the thesis and auditing the 13 sources it appeared with; entries 6 and 15 added when the duplicate-paragraph decision required reading Cohen 1988 and Frénay & Verleysen in full. Ten claim-level corrections were applied to `thesis.tex` as a result; see `../word_analysis/source_audit.md` for the before/after of each.
