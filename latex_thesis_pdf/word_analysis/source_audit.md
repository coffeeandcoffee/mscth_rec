# Source audit — every use of the ten "precedent" sources

Scope: all 44 citing sentences in the live thesis for `rahman_detection_2025`,
`moontaha_online_2023`, `tzimourta_eeg_2019`, `henao_isaza_comprehensive_2026`,
`kiarashi_sleep-derived_2025`, `hjorth_eeg_1970`, `khambampati_frequency_2023`,
`hasson_intersubject_2004`, `plucinska_impact_2022`, `saltelli_how_2010`,
`saltelli_global_2008`, `rimbert_impact_2023`, `mi_exploring_2026`.

Every claim was checked against the source PDF text. Flagged aggressively:
anything below marked 🔴 or 🟠 is worth your attention even where the thesis is
arguably defensible.

---

## ✅ A1 (FIXED) — Tzimourta: the per-region window sweep was never run

**Thesis (two places):**
> "…found that classification accuracy increased step-wise with window length
> **across all six classification problems and five brain regions tested**, with
> no reversals"
> "…increased step-wise from 5 s to 12 s windows **across six classification
> problems and five brain regions** with no exceptions"

**What the paper actually did.** The window-length sweep (Table 3, Figure 2)
covers the six classification problems at **whole-brain level only**. Then:

> "Τhe rest of the analysis is conducted **solely for the 12-s window length**,
> which is the best classification window length according to the analysis."

The five brain regions were analysed at a single window length. There is no
region × window-length grid anywhere in the paper.

**Verdict.** The six-problems half is solid — Table 3 rises monotonically in all
six rows (CN/AD 86.98→91.80, CN/mild 86.60→91.77, CN/moderate 94.68→96.76,
CN-mild/moderate 92.59→94.99, mild/moderate 87.63→91.71, CN/mild/moderate
82.34→88.79, no reversals). The five-regions half attributes a sweep that does
not exist.

**Fix:** delete "and five brain regions tested" / "and five brain regions" from
both sentences. The claim stays true and still supports the argument.

## ✅ A2 (FIXED) — Moontaha: "monotonic" is stronger than the source

**Thesis (three places):**
> "a general, **monotonic** decline in F1-score as window length increased"
> "accuracy **declined monotonically** as window length increased"
> "swept window sizes from 1 to 5 s … found a general, **monotonic** decline"

**What the paper says.** The word "monotonic" does not appear anywhere in it
(0 hits). What it says is:

> "the best predictive performance was achieved with a window length of 1 second
> irrespective of the affect dimensions, classifiers and devices. Moreover, **in
> most cases** the classification performance is decreasing with increasing
> window sizes"

and

> "The analysis on window length shows a **clear trend** of increasing performance
> scores with decreasing window length"

**Verdict.** "In most cases" and "clear trend" explicitly allow exceptions.
"Monotonic" forbids them. This is a real strengthening of the source.

**Fix:** "monotonic" → "general" or "consistent", in all three places. Note that
your own §Parameter Sensitivity already cites `mi_exploring_2026` precisely to
say window-length effects are *not* always monotonic — so the thesis currently
contradicts itself on this point.

## ✅ A3 (FIXED) — Henao Isaza: Cohen's *d* came after selection, not before

**Thesis (three places):**
> "…use it to rank and select EEG features … **prior to model training**"
> "…treating it as a principled feature-selection criterion **prior to classifier
> training**" (×2)

**What the paper's methods say:**
> "From nearly 967 initial features, the model first removed those with the
> highest correlations. **The decision tree algorithm then identified the 100 most
> important features for inclusion (Model Selection).** Cohen's d values were
> utilized to quantify the magnitude of differences between groups ACr and HC for
> each specific metric."

Order: correlation pruning → **decision-tree importance ranking** → Cohen's *d*
characterising what was selected. The *d* is downstream of a trained model, not
upstream of one.

**Mitigating.** Their own abstract says "Feature selection was based on model
performance **and** effect sizes (Cohen's d)", which is genuinely ambiguous. So
the source contradicts itself; the methods section is the more reliable text.

**Fix:** drop "prior to model training" / "prior to classifier training". "Use
Cohen's *d* to rank EEG features by their ability to separate Alzheimer's-risk
carriers from healthy controls" is fully supported on its own.

## ✅ A4 (FIXED) — Mi: "not monotonic within a single subject" is not stated

**Thesis:**
> "found that the accuracy-optimal window size differed from subject to subject
> and **was not monotonic within a single subject**, with differences of up to 13%"

**What the paper says.** Confirmed: nine subjects ✔, 13% ✔, subject-specific
optimum ✔ —
> "The maximum gap between different time window sizes can reach about 13% for
> subject A09T. More importantly … the optimal classification accuracy time window
> size is not the same for different subjects"

Its worked example runs the other way (A01T: 0.5 s → 81%, 2.5 s → 94%, rising).
Within-subject non-monotonicity may be visible in their Fig. 6, but it is not
asserted in the text.

**Fix:** either drop "and was not monotonic within a single subject", or attribute
it to the figure explicitly. The between-subject claim alone carries your argument.

## ✅ A5 (FIXED) — Plucińska is biometrics, not cognitive-state decoding

**Thesis:**
> "band-wise **decoding performance** has been shown to peak in non-adjacent
> frequency ranges rather than scaling smoothly with frequency"

The band ordering is real and well supported (β best; "The second-best
performance was found for the α and γ frequency bands"; "the worst results were
obtained for the θ band"). But the task is **person verification** — identifying
*who* someone is from their EEG — not classifying a cognitive state. Nothing in
the sentence is false, and "in EEG classification more broadly" is honest
hedging, but a reader will likely assume a cognitive-decoding result.

**Fix (optional):** name the task — "in EEG-based person verification".

## ✅ A6 (FIXED) — Saltelli 2010: the PDF on disk is abstract-only

The file resolves and opens, but extracts to **7.7 KB** — front matter plus
abstract, not the full article. The geometric proof is not in it.

Supported from what is there: "OAT … consists of analyzing the effect of varying
one model input factor at a time while keeping all other fixed"; "a novel
geometric proof of the **inefficiency** of OAT".

**Not verifiable from this copy:** the thesis's own restatement, "any two OAT
sweeps only ever probe the space along axis-aligned lines through a single fixed
operating point", and "cannot identify interaction effects". Both are standard
and almost certainly correct, but they are your paraphrase of an argument that
is not in the file you hold.

**Fix:** obtain the full text before relying on the geometric wording, or
attribute it more loosely.

---

## ✅ Verified clean

**`rahman_detection_2025` — 5 uses, every number checked and correct.**
n=10 ("We employed 10 student volunteers"); Mind Monitor ✔; 256 Hz ✔; bands
"Delta (1-4Hz), Theta (4-8Hz), Alpha (7.5-13Hz), Beta (13-30Hz) and Gamma
(30-44Hz)" — matches your text exactly including the unusual 7.5 lower bound;
80/20 k-fold ✔; "the Beta and the Gamma frequency bands contributed to high
accuracy score of more than 80%" ✔; 1:1 ratio with video=1, non-smartphone
activity=0 ✔; TikTok/YouTube Shorts/Instagram Reels on own phones ✔.
*Only unverified:* "ages 20–35" sits in their Table 1, which did not extract.

**`rimbert_impact_2023` — 3 uses, exact.**
> "the closer the selected baseline/rest time window to the preceding trial …
> the higher the BCI performance"
> "the more **global** the ERD is (more electrodes show an ERD) and the **stronger
> its amplitude** (p < 0.01)"
> "significantly better for Baseline 1 compared to Baseline 3 (+4%, p < 0.05) or
> Baseline 4 (+8%, p < 0.001)"

Your "significantly stronger, more spatially global event-related
desynchronization and higher classification accuracy" is a faithful reading.

**`hasson_intersubject_2004` — 4 uses, all supported.** Free-viewing design ✔;
"an extensive and highly significant correlation across individuals … over 29% ±
10 SD of the cortical surface" ✔; "areas which consistently failed to show
intersubject coherence … supramarginal gyrus, angular gyrus, and prefrontal
areas … linked to unique, individual variations" ✔.

**`hjorth_eeg_1970` — 2 uses.** "Mobility … may be conceived also as a mean
frequency" ✔.

**`khambampati_frequency_2023` — 2 uses.** "This feature provides information
about the frequency of rapid changes in the EEG signal" ✔ (after removing the
unsupported "speech").

**`kiarashi_sleep-derived_2025` — 3 uses.** 0.65→0.96, p 0.208→0.0023 ✔;
"group separability … Cohen's d effect size measured the standardized difference
between MCI and CN probability distributions" ✔.

**`moontaha_online_2023` label-latency figures — correct.** "µ = 86.7 s" stimulus
length and "a label delay of 86 s" ✔; "more than 82%" is the Experiment I mean,
so "from above 0.82 to 0.637" is right (the abstract's 87% is a different,
headline figure) ✔; "did not reach chance level for the valence classification" ✔.

**`saltelli_global_2008`, tumbling-window and boundary-exclusion claims —
supported.**

---

## Changes already applied in this pass

| | change | status |
|---|---|---|
| P01, P08, P22 | absence claims hedged with "to our knowledge" | ✅ applied |
| P10 | deleted "rather than a raw or model-performance-based criterion" — the source uses *d* **alongside** model performance, so the contrast was unsupported. ("alongside" could not be swapped in directly: it contradicts "as the primary … currency" in the same clause.) | ✅ applied |
| P15 ×2 | deleted "speech and" — literature chapter **and** methods chapter both carried it; no speech source exists in `library.bib` | ✅ applied |
| P16 | "cortical responses" → "intersubject correlation", matching Hasson's own wording | ✅ applied |

Recompiled: 140 pages, exit 0. PDF text layer: `CITE-NEEDED` 0, `precedent` 0,
`in speech and EEG` 0, `cortical responses` 0, `to our knowledge` 6.

| A1 | deleted "and five brain regions tested" / "and five brain regions" (2 sentences) | ✅ applied |
| A2 | "a general, monotonic decline" → "a general decline"; "accuracy declined monotonically" → "accuracy generally declined" (2 sentences). Only 2 sites, not 3 — the third already read "a general decline". Tzimourta's own "monotonic" uses are correct and were left alone (9 remain in the PDF) | ✅ applied |
| A3 | deleted "prior to model training" / "prior to classifier training" (3 sentences) | ✅ applied |

Recompiled after A1–A3: 140 pages, exit 0. PDF text layer: `five brain regions` 0,
`monotonic decline` 0, `declined monotonically` 0, `prior to model training` 0,
`prior to classifier training` 0.

Backup before this pass: `thesis.tex.bak-preA1A3`.

| A4 | dropped "was not monotonic within a single subject"; rewritten to say what Mi et al. actually did (per-participant sweep, nine subjects, up to 13% within-participant gap) and to state plainly that the present sweep is cohort-level and was **not** resolved per participant, leaving that to future work | ✅ applied |
| A5 | "band-wise decoding performance" → "band-wise performance in EEG-based person verification" | ✅ applied |
| A6 | cut "since any two OAT sweeps only ever probe the space along axis-aligned lines through a single fixed operating point"; the interaction limitation now stands on design logic ("no two parameters are ever varied together") with no citation, in both places. `\cite{saltelli_how_2010}` retained only for statistical inefficiency, which the extract states verbatim | ✅ applied |

Recompiled after A4–A6: 140 pages, exit 0. Backup: `thesis.tex.bak-preA4A6`.

**A6 — correction to the original finding.** The Procedia PDF is *not* truncated;
it is complete at 3 pages (pp. 7592–7594), because it is the conference-presentation
extract from the Sixth International Conference on Sensitivity Analysis of Model
Output. Its whole body is the abstract, that abstract repeated verbatim under
"1. Main text", one paragraph on the Stern Review, and the reference list. Its own
bibliography points at the journal version: *"Saltelli, A., Annoni Paola, 2010 How
to avoid a perfunctory sensitivity analysis, Revised for Environmental Modelling
and Software."* That journal article is not obtainable, so the thesis now claims
only what the extract supports.

**Also checked:** `saltelli_global_2008` §2.4.2 ("One-at-a-time (OAT) Sampling",
pp. 66–70) does not discuss interactions either, so it was not a valid re-home for
that claim. It *does* support the thesis's other claim — attribution being local to
the operating point — via "It is applicable everywhere if the linear model is
appropriate, and **for some region around the current sample point** otherwise."
That citation was left in place.

**All six audit items resolved. Nothing outstanding.**
