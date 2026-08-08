# Thesis-Wide De-Risking Edit Plan

**Principle applied throughout:** every claim identified as over-scoped gets either (a) a short, explicit, self-aware qualification framed as observation / limitation / future work, propagated consistently to every place the claim recurs, or (b) — only where (a) genuinely can't rescue the sentence — a small excision (a phrase or weak example dropped), never a full-paragraph rewrite or silent deletion of the underlying point. Nothing is hidden; everything flagged is named out loud, once, in the right register, and then referenced consistently everywhere it resurfaces so no two sections contradict each other's confidence level.

Each entry: **Anchor** (where) → **Issue** → **Proposed edit** (near-final wording, minimally invasive) → **Propagate to** (other spots that must match).

---

## 1. Literature and State of the Art

### 1.1 `\section{EMG Artifacts in EEG}`
**Issue:** Current framing scopes the myogenic-contamination risk to "the raw broadband signal or a temporal-site (TP9/TP10) feature." Goncharova et al.'s reported EMG beta-peak ranges (13–38 Hz at frontal and temporal sites) sit inside conventional alpha/beta, not just high-gamma/broadband — so the risk as currently scoped is narrower than the cited evidence actually supports.

**Edit:** After the existing Goncharova/Whitham/Yilmaz sentence, add:
> "Because EMG's beta-range peak power has been reported as low as 13–38~Hz at frontal and temporal sites specifically \cite{goncharova_emg_2003}, this contamination risk is not confined to the raw signal or high-gamma alone; it plausibly extends to alpha- and beta-band features at AF7, AF8, TP9, and TP10 as well."

**Propagate to:** 2.3 (Methods Artifact Analysis), 3.3 (Discussion Neuroscience), 4.2 (Applications limits paragraph).

### 1.2 `\subsection{Notch Filtering and the High-Gamma Band}`
**Issue:** Frames consumer-hardware gamma-band distortion as one-directional (added noise). Pattisapu & Ray also report the consumer device can *underestimate* true gamma amplitude relative to research-grade — a different failure mode than "contamination," and citing only the alarming direction is imprecise, not conservative.

**Edit:** Add one sentence:
> "It is worth noting this risk is not necessarily one-directional: comparisons between consumer-grade and research-grade amplifiers under matched conditions have also found the consumer device to *underestimate* true gamma-band amplitude relative to the research-grade reference \cite{pattisapu_stimulus-induced_2021}, indicating hardware-related gamma distortion in this class of device is not uniformly an inflationary artifact."

**Propagate to:** 3.3 (Discussion Neuroscience, notch-filter paragraph).

### 1.3 `\section{Engagement and Attention...}` — de Carvalho citation (first occurrence)
**Issue:** "Engagement-sensitive markers... failed to discriminate" is stated as a clean result. The paper's own authors attribute the FMT null result partly to a baseline confound (post-effort baseline possibly saturated with residual fatigue) and note the modest sample (n=15).

**Edit:** Append clause:
> "...though the authors themselves attribute this null result partly to a baseline confound (a post-effort baseline potentially saturated with residual fatigue) and a modest sample (n=15), rather than presenting it as an unqualified failure of FMT as an engagement marker."

**Propagate to:** 2.2 (Methods §op_stay_skip, second occurrence of the same citation — must use matching wording).

### 1.4 `\subsection{Delta-Band Function in Awake Cognition}` — Xu / Rahman precision
**Issue:** "Mixed... rather than confirmatory" is accurate in direction but understates how weak and how differently-scoped both supporting studies are (Xu: adolescents, mean age 12.3, small effect F=2.184/p=.040/R²=.115; Rahman: n=10, binary phone-use classification, not a skip decision, different gamma-band boundaries).

**Edit:** Tighten the existing sentence to name these specifics — this actually *strengthens* your novelty claim, since neither prior study addresses anything resembling STAY/SKIP:
> "...Xu et al., studying adolescents (mean age 12.3) with a small, borderline-significant effect (F=2.184, p=.040, R²=.115)... Rahman et al., with n=10 and a coarser binary phone-use classification (not a skip decision) using different gamma-band boundaries (30–44~Hz)... Neither paradigm resembles the self-paced STAY/SKIP decision studied here, which is itself evidence that the present question has not previously been addressed rather than that prior evidence on it is merely mixed."

**Propagate to:** 3.3 (Discussion Neuroscience, delta paragraph — same tightening).

---

## 2. Materials and Methods

### 2.1 `\section{Participants}` (§method_participants)
**Issue:** The sample's academic skew is only stated later, in Results §results_exp_learnings, as an incidental observation about recruitment channels — not flagged as a generalizability limitation where the sample itself is defined.

**Edit:** Add one sentence at the end of the Participants section:
> "Because recruitment relied substantially on university mailing lists, the sample skews toward an academically-oriented population; the implications of this for generalizability are addressed in Section~\ref{sec:results_interpretation} and Future Research."

**Propagate to:** 4.6/4.7 (Applications, Public Health and Message Design subsections — wherever population-wide applicability is implied).

### 2.2 `\subsection{Operational Definition of STAY and SKIP}` (§op_stay_skip) — de Carvalho (second occurrence)
Same fix as 1.3, same wording, so the hedge doesn't read differently depending on which chapter someone opens.

### 2.3 `\subsection{Artifact Analysis}` (Methods)
Same additions as 1.1 and 1.2 — the broadened EMG scope and the Pattisapu bidirectionality note belong here too, phrased identically, since this is the section a methods-focused examiner will scrutinize hardest.

### 2.4 `\section{Metrics and Significance Testing}` (§method_metrics_and_significance) — **new item**
**Issue:** Not previously flagged, but it's the same category of gap: many Wilcoxon tests are run (per feature × per metric × per class × per intra/inter setting) at uncorrected α=0.05. A statistically literate examiner will very likely ask about this, and right now the thesis gives them nothing to point to.

**Edit:** Add one sentence at the end of the section:
> "Because performance comparisons were repeated across a number of feature–baseline pairs, metrics, and evaluation settings, the reported \(p\)-values are not corrected for multiple comparisons; individual comparisons are therefore best read as exploratory screening criteria rather than confirmatory hypothesis tests, and family-wise error control is identified as a priority for future replication (Section~\ref{sec:future_research})."

**Propagate to:** Future Research (tighten existing "validation and replication" paragraph to name this specifically), and lightly to §results_alternative_index where individual comparisons are reported feature-by-feature (a one-clause reminder is enough — don't repeat the full sentence).

---

## 3. Results and Discussion

### 3.1 `\section{Engagement Index Model Performance...}` / `\subsection{Alternative Index Exploration}` (§results_model_performance, §results_alternative_index)
**Issue:** This is the load-bearing fix. Numbers like "F1-test = 44.3%" are reported without reminding the reader that ≈0.5 is the stated chance-level reference for balanced binary classification (established earlier in §method_metrics_and_significance). Left implicit, this is exactly the gap an examiner will use to puncture the Applications section's confidence.

**Edit:** Add, once, at the first point where an inter-subject SKIP F1 below 50% is reported:
> "It is worth stating explicitly: for the \texttt{SKIP} class in the inter-subject setting, even the best-performing individual feature (TP10 raw standard deviation, F1-test = 44.3\%) remains below the approximate 0.5 chance-level reference for balanced binary classification (Section~\ref{sec:method_metrics_and_significance}). The reported gains are therefore properly read as improvements over a weak Engagement Index baseline, not as evidence of exceeding chance-level discrimination in the cross-participant setting."

**Propagate to:** §results_interpretation (Behaviour subsection, where "the signal is informative but not deterministic" is already stated — add a one-clause anchor back to the 0.5 reference so the two statements visibly agree), and Applications §Application para 3 + Public Health Rec 2 (soften per 4.1/4.7 below).

### 3.2 `\section{Behavioral Swipe-Burst Phenomenon}` (§results_burst_analysis)
**Issue:** Currently relies entirely on `\input{burst_stats.tex}` with no prose fallback — and now that the numbers exist, they should be stated in the text itself, not only in an auto-inserted table.

**Edit:** Add prose alongside the input:
> "Specifically, under the default \(\pm3.0\)s window, 67.4\% of swipe events are isolated singletons (participant-level mean 69.1\%~\(\pm\)~16.2\%), while widening to \(\pm5.0\)s reduces this to 48.9\% (participant-level mean 46.9\%~\(\pm\)~20.8\%). The large between-participant standard deviation is itself informative: burst tendency appears to be a strongly individual trait rather than a fixed population parameter, foreshadowing the precision-public-health argument developed in Section~\ref{sec:results_interpretation}."

**Propagate to:** Applications Message Design (iv) — this is what lets that recommendation cite real numbers instead of asserting the pattern.

### 3.3 `\subsection{Neuroscience}` (§results_interpretation)
Carries forward 1.1 (broadened EMG scope), 1.2 (Pattisapu bidirectionality), and 1.4 (Xu/Rahman precision) — same wording each time, so a reader who jumps straight to Discussion sees the identical level of care as the Lit Review reader.

---

## 4. Application, Policy, Public Health, Message Design, Societal Scope

### 4.1 `\subsection{Application}`, paragraph 3
**Current:** "...could be adapted to identify when a user is likely to continue engaging versus when they are likely to skip."
**Edit:** append — "...though at present this discrimination remains modest and, in the cross-participant setting, does not yet exceed the chance-equivalent F1 reference established in Section~\ref{sec:results_model_performance}."

### 4.2 `\subsection{Application}`, limits paragraph
Already the best-hedged paragraph in the section — extend it with the broadened EMG scope from 1.1 (one clause: "...including in alpha and beta bands, not only high-gamma...") so it matches the widened Discussion/Methods hedge exactly.

### 4.3 Policy Rec 1 ("watch time is the wrong regulatory quantity")
**Edit:** Add one sentence distinguishing source of evidence:
> "This recommendation follows from the cited dissociation and flow literature \cite{baughan_i_2022, knierim_flow_2026} rather than from the present EEG results directly, which characterize a candidate physiological signature of the pre-swipe state but do not themselves measure dissociation, flow, or recall."

### 4.4 Policy Rec 2 ("friction is an evidence-backed lever")
- Add, after the citation: "(a single study, on a Twitter client rather than short-form video; replication on TikTok-like platforms is needed before generalizing)."
- **Excision, not caveat:** drop "time-limit prompts" from the parenthetical example list — that specific feature's evidence in Baughan et al. is more ambiguous (its use was *positively* associated with prior dissociation, i.e. plausibly reflects people already dissociated reaching for the exit, not the dialog causing lower dissociation). Keep "custom lists and reading-history labels," which are more cleanly supported.

### 4.5 Policy Rec 3 ("auditability of engagement optimization")
**Current:** "It is reasonable to assume that platforms... model the same decision boundary with far greater precision."
**Edit:** Reframe as a named hypothesis rather than an asserted premise:
> "Platforms plausibly have access to behavioral, and possibly physiological, signals at a scale and precision this thesis cannot match; if so, the same auditability logic applies with greater force. This thesis has no access to platform-internal data, and this claim should be read as a reasonable extrapolation rather than a demonstrated fact."

### 4.6 Public Health Rec 1 ("economically feasible... for the first time")
**Excision:** drop "for the first time" (unfalsifiable, uncited).
**Edit:** add the internal-consistency note against your own Results §results_pipeline:
> "...makes small-\(n\) physiological pre-testing more accessible to public institutions than fully custom or clinical-grade EEG pipelines would be — though, as Section~\ref{sec:results_pipeline} notes, substantial engineering investment was still required to reach this point (reflected in the pipeline's own multi-month development and code consolidation), and this should be read as a reduction in barrier-to-entry rather than a turnkey solution."

### 4.7 Public Health Rec 2 ("operationalizes directly")
**Edit:** "...a quantity the present labeling framework provides an initial, currently modest-power operationalization of (Section~\ref{sec:results_model_performance})" — replacing "operationalizes directly."

### 4.8 Message Design (i) — the biggest rewrite, split into two properly-scoped claims
**Current:** "...implying that by the time a viewer swipes, the disengagement decision is already neurally underway; health messages must therefore place their hook within the first seconds and cannot rely on delayed payoffs."

**Edit:**
> "The imminent-skip window examined here (2–5 seconds before the physical swipe) was adopted as a literature-motivated heuristic starting point rather than an independently validated decision boundary (Section~\ref{sec:op_labeling_params}), and class separability within it, while statistically distinguishable from the Engagement Index baseline, remains modest in absolute terms (Section~\ref{sec:results_model_performance}). With that scoping in mind, the general pattern — that some pre-swipe neural signature precedes the physical swipe action — is broadly consistent with content needing to establish interest early rather than relying on a delayed payoff. Testing where within a video's opening seconds this signature is strongest, and whether it maps onto specific content features, is a natural extension of the present labeling framework rather than a claim established by it here."

### 4.9 Message Design (ii)
**Current:** "...known to occur episodically within the dissociative stream \cite{baughan_i_2022}."
**Edit:**
> "...because such states are not reliably self-interrupted — Baughan et al. found re-engagement is often only recognized retrospectively, if at all — message formats should include pattern interrupts... intended to manufacture re-engagement points rather than rely on spontaneous recovery of attention."

### 4.10 Message Design (iv)
**Edit**, using the numbers from 3.2:
> "Swipes cluster into rapid bursts for a substantial share of events, though the exact proportion is parameter-dependent: 67.4\% of swipes are isolated singletons under the default \(\pm3\)s window, falling to 48.9\% under a wider \(\pm5\)s window (Section~\ref{sec:results_burst_analysis}). Burst tendency also varies sharply between participants (SD of 16–21 percentage points around these means), consistent with the individual heterogeneity noted in Section~\ref{sec:results_interpretation}. This favors repeated, distributed placement over single high-cost placements as a population-level default, while flagging burst-aware, per-user delivery timing as a natural extension once outcome-linked validation is available — the present work establishes that swipes cluster, not that burst membership itself predicts weaker message encoding or evaluation."

### 4.11 Societal Scope — "documented population-level trends"
**Issue:** "documented" asserts settled fact for genuinely contested trends, with no citation attached.
**Edit — needs your input:** either supply a citation for youth-socialization/loneliness trends (e.g., something you'd stand behind), or soften the word:
> "...widely discussed, though contested, population-level trends in youth socialization, loneliness, or, further downstream, family formation..."
I'd default to the softened wording unless you have a specific citation in mind.

---

## 5. Future Research

Tighten (don't expand) the existing generic paragraphs to name the specific gaps now surfaced, so they read as pre-identified rather than examiner-discovered:
- "Validation and replication" paragraph → add multiple-comparisons correction as a named next step (ties to 2.4).
- "Mechanistic clarification" paragraph → widen the artifact-control language from "electromyographic control" to explicitly include alpha/beta-band EMG overlap, not only high-frequency/broadband (ties to 1.1).
- "Generalizability across contexts" paragraph → add the academically-skewed sample explicitly as a named limitation to be addressed by broader recruitment (ties to 2.1), rather than leaving it only as content-domain generalizability.
- New one-sentence addition: outcome-linked validation of burst membership (does a burst predict weaker encoding/recall, not just cluster in time) as a named open question (ties to 4.10).

---

## 6. Global consistency checklist (grep before submission)

- Every occurrence of `de_carvalho_selection_2026` → same hedge clause (1.3/2.2).
- Every occurrence of `baughan_i_2022` → Twitter-not-TikTok reminder present at least once per chapter it's newly invoked in (SOTA already has it; Policy/Message-Design need it added).
- Every occurrence of TP10/AF7 as a "top feature" in prose (Results, Discussion, Applications) → confirm the EMG-confound clause used is the *broadened* alpha/beta version (1.1), not the older high-gamma-only version, everywhere.
- Every occurrence of an inter-subject SKIP F1 number → confirm the 0.5 chance-reference reminder (3.1) is either present or was already established earlier in the same section (don't repeat verbatim more than once per chapter — one clear statement per chapter is enough, over-repeating reads as anxious rather than rigorous).
- "for the first time" / "directly" / "reliably" / "clearly" — scan Applications specifically for these words; each one is a place worth checking whether the underlying number supports the adverb.

## 7. Items needing your decision before I draft final text

1. Societal Scope §4.11 — supply a citation for the youth-socialization/loneliness claim, or confirm the softened "widely discussed, though contested" wording.
2. Whether you want the multiple-comparisons item (2.4) added at all — it's a genuinely new observation (not from our earlier pass), so flagging it separately in case you'd rather address it a different way (e.g., an FDR correction re-run) than a stated limitation.
3. Confirm you're fine with the two small excisions (Policy Rec 2's "time-limit prompts" example; Public Health Rec 1's "for the first time") — these are the only two places I'm proposing removal rather than reframing.
