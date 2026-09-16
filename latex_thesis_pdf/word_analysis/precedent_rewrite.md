# "precedent" rewrite — handover list

22 occurrences in `thesis.tex`, all rewritten. Every site carries an invisible
LaTeX comment marker `% CITE-NEEDED Pnn: …` on the line above its paragraph.
Grep them with:

```bash
grep -n "CITE-NEEDED" thesis.tex
```

Backup of the pre-rewrite file: `thesis.tex.bak-preprecedent`.
Compiled clean afterwards (140 pages, biber + 3× pdflatex, exit 0). Verified in
the PDF text layer: `CITE-NEEDED` 0 hits, `precedent` 0 hits, `prior work` 17 hits.

**22 sites, 17 unique** — P03/P06, P11/P17, P12/P18 and P13/P19 are the same
sentences duplicated across two sections (see "Duplication" at the bottom).
The table lists each unique site once, naming its twin.

Quotes in column 4 are copied verbatim from the source PDFs in
`~/Zotero/storage/`, located via the `file` field of `library.bib`.

---

## P01 — Introduction §Approach (no twin)

| | |
|---|---|
| **Old** | "Because none of these parameters **had established precedent** for a self-paced, naturalistic swiping paradigm specifically, their influence on downstream class separability was additionally characterized through a one-at-a-time sensitivity analysis…" |
| **New** | "Because **no prior work had established these parameters** for a self-paced, naturalistic swiping paradigm specifically, …" |
| **Reference** | *none — and none is possible* |
| **Quote** | — |

✅ **RESOLVED** — hedged to "Because, to our knowledge, no prior work had established…".

## P02 — §Adjacent studies

| | |
|---|---|
| **Old** | "The closest published **precedent** to the present work is Rahman et al.'s study on the detection of Short-form Video Addiction (SVA) from wearable sensors." |
| **New** | "The closest published **prior work** is Rahman et al.'s study on the detection of Short-form Video Addiction (SVA) from wearable sensors." |
| **Reference** | `rahman_detection_2025` ✔ already cited |
| **Quote** | "We used 3 most popular short-form video platforms, TikTok, YouTube Shorts, and Instagram Reels as the source of content. The volunteers were free to use their smartphones to visit the short-form video apps which ensures the media contents of their choice." |

Supports the adjacency claim (same platform class, own-feed, own phone). Cite is already in place.

## P03 — §Sampling Boundaries (twin: P06)

| | |
|---|---|
| **Old** | "For the present work, this **precedent** motivates treating boundary overlap between STAY and SKIP not as an edge case to be smoothed over…" |
| **New** | "For the present work, this **prior work** motivates treating boundary overlap between STAY and SKIP not as an edge case…" |
| **Reference** | `moontaha_online_2023` ✔ already cited |
| **Quote** | "Since EEG data is considered stationary only over short time intervals, the preprocessing and the feature extraction were performed in tumbling windows with a fixed size and no overlap." |

## P04 — §Parameter Choice in EEG-classification

| | |
|---|---|
| **Old** | "This provides an independent, monotonic **precedent** for the intuition that longer windows — carrying more samples per estimate — yield more separable features…" |
| **New** | "This provides independent, monotonic **support** for the intuition that longer windows…" |
| **Reference** | `tzimourta_eeg_2019` ✔ already cited |
| **Quote** | "With regard to the window length, the results showed a high classification accuracy as the length of the window was gradually increasing, and the best classification results were obtained for epochs of 12 s." |

Monotonicity is confirmed by their Table 3 (CN/AD: 86.98 → 91.80% across 5→12 s, no reversals).

## P05 — §Sampling Boundaries (2nd copy of section)

| | |
|---|---|
| **Old** | "…this **precedent** motivates excluding boundary-adjacent epochs outright in the present swipe-event paradigm as well…" |
| **New** | "…this **prior work** motivates excluding boundary-adjacent epochs outright…" |
| **Reference** | `moontaha_online_2023` ✔ already cited |
| **Quote** | same as P03 — "…performed in tumbling windows with a fixed size and no overlap." |

## P06 — duplicate of P03. Same old, new, reference and quote.

## P07 — §Sensitivity to Parameter Choice

| | |
|---|---|
| **Old** | "Nonetheless, OAT sweeps of individual temporal parameters **have established precedent** in EEG decoding: window length has been varied in isolation across a 1–5 s range…" |
| **New** | "Nonetheless, OAT sweeps of individual temporal parameters **are established** in EEG decoding: …" |
| **Reference** | `moontaha_online_2023` ✔ already cited |
| **Quote** | **Sweep:** "Different window length, l ϵ [1s, 2s, 3s, 4s, 5s] were tested on the AMIGOS dataset and the dataset from Experiment I to find the optimal one for the classification pipeline." **Result:** "the best predictive performance was achieved with a window length of 1 second irrespective of the affect dimensions, classifiers and devices. Moreover, in most cases the classification performance is decreasing with increasing window sizes emphasizing the need of more data points." **Restated in Discussion:** "The analysis on window length shows a clear trend of increasing performance scores with decreasing window length; therefore, a window length of 1 second is chosen for further analysis." |

✅ **RESOLVED** — my earlier flag was wrong. The sweep is stated in running text
(lines 323–325 of the source), not only in a table. Both the sweep and the
monotonic-decline result are directly quotable.

## P08 — §Default Labeling Parameter Selection

| | |
|---|---|
| **Old** | "…while the third **has no comparable precedent**." |
| **New** | "…while the third **has no comparable prior work**." |
| **Reference** | *none — and none is possible* |
| **Quote** | — |

✅ **RESOLVED** — hedged to "…while the third has, to our knowledge, no comparable prior work."

## P09 — §Class Separation Metric (Ch. 2, first version)

| | |
|---|---|
| **Old** | "Cohen's *d* **has established precedent** specifically in EEG-based group- and class-separability analysis:" |
| **New** | "Cohen's *d* **is established** specifically in EEG-based group- and class-separability analysis:" |
| **Reference** | `henao_isaza_comprehensive_2026` ✔ already cited |
| **Quote** | "Cohen's d values were utilized to quantify the magnitude of differences **between groups ACr and HC** for each specific metric." (ACr = asymptomatic PSEN1-E280A mutation carriers; HC = healthy controls) |

This is the quote that licenses the phrase "group- and class-separability
analysis": *group* = ACr vs HC, *class* = the same contrast fed to the classifier.

## P10 — §Class Separation Metric (Ch. 2)

| | |
|---|---|
| **Old** | "This **precedent** motivates treating Cohen's *d*, rather than a raw or model-performance-based criterion, as the primary descriptive currency for separability…" |
| **New** | "This **prior work** motivates treating Cohen's *d*, rather than …" |
| **Reference** | `henao_isaza_comprehensive_2026` ✔ already cited |
| **Quote** | "From nearly 967 initial features, the model first removed those with the highest correlations. The decision tree algorithm then identified the 100 most important features for inclusion (Model Selection). **Cohen's d values were utilized to quantify the magnitude of differences between groups ACr and HC for each specific metric.**" — and, in the abstract: "**Feature selection was based on model performance and effect sizes (Cohen's d).**" |

⚠ **Partial.** The paper uses Cohen's *d* *alongside* model performance, not
*rather than* it. The thesis's "rather than a raw or model-performance-based
criterion" overstates the contrast. **Recommend softening to "alongside".**

## P11 — §Class Separation Metric (twin: P17)

| | |
|---|---|
| **Old** | "The metric **has established precedent** specifically in EEG-based group and class separability analysis:" |
| **New** | "The metric **is established** specifically in EEG-based group and class separability analysis:" |
| **Reference** | `henao_isaza_comprehensive_2026` ✔ already cited |
| **Quote** | same as P09 |

## P12 — §Class Separation Metric (twin: P18)

| | |
|---|---|
| **Old** | "Taken together, this **precedent** establishes two points relevant to the present work…" |
| **New** | "Taken together, this **prior work** establishes two points relevant to the present work…" |
| **Reference** | `kiarashi_sleep-derived_2025` ✔ already cited |
| **Quote** | "Multi-night aggregation improved AUROC from 0.61 to 0.77, reduced within-subject variability 1.6-fold, and enhanced group separability based on Cohen's d improving from 0.65 to 0.96." |

**Transferability:** the shared structure is *a methodological choice changing d
without changing the underlying signal*. There it is multi-night averaging
(0.65→0.96, p 0.208→0.0023, replicated across three embedding methods:
SleepTransformer 0.56→1.08, U-Time 0.18→0.53). Here it is the labelling-parameter
sweep. In both cases *d* tracks the processing decision, which is why it works as
a sensitivity metric and not only as a ranking metric. That is the transferable
claim — not the sleep domain, the classifier, or the clinical contrast.

## P13 — §Class Separation Metric (twin: P19)

| | |
|---|---|
| **Old** | "…extending this **precedent** to that specific setting is a contribution of the present work." |
| **New** | "…extending this **prior work** to that specific setting is a contribution of the present work." |
| **Reference** | *none in sentence* — inherits `henao_isaza_comprehensive_2026` + `kiarashi_sleep-derived_2025` from the two preceding sentences |
| **Quote** | Kiarashi: "To assess how well each model distinguished **between MCI and CN groups**, we evaluated group separability using predicted probabilities from **leave-one-subject-out** cross-validation." Henao Isaza: "Cohen's d values were utilized to quantify the magnitude of differences **between groups ACr and HC**." |

**What "that specific setting" means.** In both sources, the two things being
separated are **two groups of different people** — patients vs. controls — and the
label comes from a clinical diagnosis that is fixed for the whole recording. In
this thesis, the two things being separated are **two moments inside one person's
own recording**, and the label comes from a behavioural event that person
generated seconds earlier. Same statistic, different object: between-subject and
diagnosis-given vs. within-subject and event-locked. Extending *d* to that is the
claimed contribution. The quotes above prove the contrast by showing what the
sources actually compared.

## P14 — §Peak Frequency as a Time-Domain Frequency Estimator

| | |
|---|---|
| **Old** | "Estimating a signal's frequency content directly in the time domain, without an intervening Fourier transform, **has established precedent** along two lines." |
| **New** | "…**is established** along two lines." |
| **Reference** | `hjorth_eeg_1970` ✔ already cited |
| **Quote** | "Mobility, giving a measure of the standard deviation of the slope with reference to the standard deviation of the amplitude. It is expressed as a ratio per time unit and **may be conceived also as a mean frequency**." |

Strong match — this is exactly the claim the thesis makes about Mobility.

## P15 — §Peak Frequency

| | |
|---|---|
| **Old** | "A second, independent line of **precedent** is the zero-crossing rate, which counts sign changes per unit time…" |
| **New** | "A second, independent line of **prior work** is the zero-crossing rate…" |
| **Reference** | `khambampati_frequency_2023` ✔ already cited |
| **Quote** | "Zero Crossing Rate (ZCR): ZCR is a measure of how often a signal changes its sign within a given frame. **This feature provides information about the frequency of rapid changes in the EEG signal. It quantifies the rate at which the signal crosses zero.** We calculate the number of zero-crossings and normalize it by dividing it by twice the length of the signal." |

✅ **Correction to my earlier flag.** The paper *does* define ZCR as a frequency
descriptor, in §2.3 Time domain features. The EEG half of the claim is fully
carried. What is **not** carried is the word **"speech"** — there is no speech
source anywhere in `library.bib` (searched: zero-crossing, speech, mean
frequency, time-domain → only `hjorth_eeg_1970`, `khambampati_frequency_2023`,
`hajarian_gamification_2024`, and the last does not mention ZCR).
**Recommend: delete "speech and", keep the sentence otherwise unchanged.**

## P16 — §Experimental Protocol

| | |
|---|---|
| **Old** | "This unconstrained free-viewing design **follows established precedent** in naturalistic neuroimaging, where subjects freely viewing continuous audiovisual material… nonetheless produced extensive, highly significant cortical responses." |
| **New** | "This unconstrained free-viewing design **follows standard practice** in naturalistic neuroimaging, …" |
| **Reference** | `hasson_intersubject_2004` ✔ already cited |
| **Quote (design half)** | "We implemented this approach in the study of the functional organization of human cortex under free viewing of a long (30 min) uninterrupted segment taken from an original audiovisual feature film. Subjects were instructed to freely view the movie segment and report its plot at the end of the experiment." |
| **Quote (result half — "extensive, highly significant cortical responses")** | "Despite the free viewing and complex nature of the movie, we found an **extensive and highly significant correlation** across individuals watching the same movie. Thus, on average over 29% ± 10 SD of the cortical surface showed a highly significant intersubject correlation during the movie." |

Note the thesis paraphrases this as "cortical responses"; the source says
"correlation across individuals" (intersubject correlation), which is a stronger
and more specific claim. **Optional: match the source wording.**

## P17, P18, P19 — duplicates of P11, P12, P13. Same old, new, reference and quote.

## P20 — §Results, Feature Importance

| | |
|---|---|
| **Old** | "Non-monotonic, band-specific separability profiles of this kind **are not without precedent** in EEG classification more broadly; band-wise decoding performance has been shown to peak in non-adjacent frequency ranges rather than scaling smoothly with frequency." |
| **New** | "Non-monotonic, band-specific separability profiles of this kind **have been reported** in EEG classification more broadly; …" |
| **Reference** | `plucinska_impact_2022` ✔ already cited |
| **Quote** | "The β frequency band significantly differs only from the θ and δ frequency bands. **The second-best performance was found for the α and γ frequency bands.** Compared to the first scenario, the δ band performed worse. Again, the worst results were obtained for the θ band." |

Genuinely supports the claim: peak at β, second place shared by α *and* γ
(non-adjacent), worst at θ. Not a smooth function of frequency.

## P21 — §Discussion, Psychology

| | |
|---|---|
| **Old** | "This pattern **has precedent** on both the neural and the methodological side." |
| **New** | "This pattern **is supported by prior work** on both the neural and the methodological side." |
| **Reference** | `hasson_intersubject_2004` ✔ already cited |
| **Quote** | "In addition to the highly synchronized cortex, we also found a pattern of areas which consistently failed to show intersubject coherence. These areas included the supramarginal gyrus, angular gyrus, and prefrontal areas. Thus, the 'collective' coherence effect naturally divides the cortex into a system of areas that manifest an across-subject, stereotypical response to external world stimuli versus regions that are **linked to unique, individual variations**." |

Strong match — this is the exact sentence the thesis paraphrases.

## P22 — §Conclusion

| | |
|---|---|
| **Old** | "…conditional on three temporal parameters that **had no established precedent** for a self-paced behavioral event and were therefore fixed by argument rather than by evidence." |
| **New** | "…conditional on three temporal parameters that **no prior work had established** for a self-paced behavioral event and were therefore fixed by argument rather than by evidence." |
| **Reference** | *none — and none is possible* |
| **Quote** | — |

✅ **RESOLVED** — hedged to "…that, to our knowledge, no prior work had established…".

---

## Summary of what still needs attention

| site | status |
|---|---|
| P01, P08, P22 | ✅ **done** — hedged to "to our knowledge" |
| P07 | ✅ **done** — sweep and decline both quotable in running text; my earlier flag was wrong |
| P09 | ✅ **done** — correct quote is the ACr-vs-HC one, not the abstract line |
| P15 | ⚠ EEG half is fully supported. **"speech" is not** — no speech source exists in the bib. Delete "speech and" |
| P10 | ⚠ "rather than a model-performance-based criterion" overstates it; the source uses *d* **alongside** model performance. Soften to "alongside" |
| P16 | ⚠ Source says "correlation across individuals", thesis says "cortical responses". Optional wording match |
| P04 | ⚠ Same non-overlapping-epoch unit as ours, but epoch = whole-recording segment for a **subject-level trait**, not an event-locked label. Support is for the monotonicity direction only |
| P13, P19 | No cite in sentence; inherits from the two preceding sentences. Fine as prose |

Everything else (P02, P03, P05, P06, P11, P12, P14, P17, P18, P20, P21) has a
correctly placed citation and a verbatim quote that carries the claim.

## Duplication — separate issue, not fixed

Two passages exist twice in `thesis.tex`, which is why 22 sites collapse to 17:

- **Cohen's *d* paragraph** — `thesis.tex:480` and `thesis.tex:652`.
  §2 "Class Separation Metric" and §3.x "Class Separation Metric" are near-identical.
  Accounts for P11/P17, P12/P18, P13/P19.
- **Boundary-handling paragraph** — `thesis.tex:459` and `thesis.tex:469`.
  Two sections both titled "Sampling Boundaries in EEG…", differing only in capitalisation.
  Accounts for P03/P06.

Not touched — deleting a duplicate is a structural edit, not a word swap.
