# Chapter: Methods and Results

## 1. Materials and Methods

We built a data pipeline to turn continuous, unstructured EEG data—recorded while people scrolled through TikTok—into a clear format that a machine learning model could learn from. The following sections explain how we isolated, processed, and evaluated the brain signals related to staying engaged (`STAY`) versus deciding to skip a video (`SKIP`).

### 1.1 Data Acquisition and Asynchronous Synchronization
The raw EEG stream was recorded using the Muse S (Gen 2) headband, utilizing four dry electrodes (TP9, AF7, AF8, TP10) positioned according to the International 10-20 system [Insert citation]. Data was transmitted via Bluetooth using the Lab Streaming Layer (LSL) protocol [Insert citation]. 

To capture naturalistic browsing behavior while ensuring sub-millisecond precision, a custom Python architecture (`recording_script_v4.py`) was developed to asynchronously manage the physiological data stream and the behavioral action markers (keystrokes). 

**Asynchronous Clock Synchronization:** A fundamental challenge in consumer-grade BCI research is aligning asynchronous system events (a keyboard press) with a continuous physiological data stream. The LSL protocol utilizes its own high-resolution monotonic clock ($t_{LSL}$), whereas the `pynput` keyboard listener operates on the standard system clock ($t_{sys}$). 

To solve this, our script computes a hardware transmission offset upon the arrival of the first EEG sample:
$$ \Delta t_{offset} = t_{sys}^{sample} - t_{LSL}^{sample} $$

When a participant executes a mechanical swipe, the simultaneous keyboard press is logged in system time ($t_{sys}^{key}$). To map this behavioral event directly onto the physiological data stream, its LSL timestamp is retroactively derived:
$$ t_{LSL}^{key} = t_{sys}^{key} - \Delta t_{offset} $$

During the sample-pulling loop, the script flags an individual EEG sample as an explicit event marker if its native LSL timestamp falls within a predefined chronological tolerance $\tau_{sync}$ (empirically set to $100$ ms) of the derived keypress timestamp:
$$ | t_{LSL}^{sample} - t_{LSL}^{key} | \leq \tau_{sync} $$
This rigorous mathematical mapping ensures that the physical action of swiping is anchored to the exact corresponding brainwave signature, independent of Bluetooth transmission delays.

**Hardware Fault Tolerance:** Because consumer Bluetooth devices are highly susceptible to packet drops during continuous recording, an explicit connection-loss detection algorithm was implemented. The system continuously evaluates the delta between the current system time and the last received sample: $\Delta t_{drop} = t_{sys}^{now} - t_{sys}^{last\_sample}$. If $\Delta t_{drop} > 5.0s$, a hardware disconnection is formally diagnosed. The script automatically kills the dormant LSL subprocess and instantiates a new recording segment, preserving the structural continuity of the session without requiring manual intervention.

**Uniform Interpolation:** Following raw data acquisition, the continuous signal was mapped to a uniform $256$ Hz temporal grid using linear interpolation for the continuous EEG voltage values and nearest-neighbor interpolation for the discrete behavioral class markers. This normalization is mathematically required to conduct valid frequency-domain transformations [Insert citation].

**The $\pm$0.5s Isolation Paradigm:** Our fundamental objective was to isolate the neurological state immediately preceding the mechanical swipe action. Given the hypothesis that the decision to disengage manifests as a transient motor-readiness or event-related potential (ERP) [Insert citation], the `SKIP` class (Class 0) was strictly bounded to a 1.0-second window centered exactly on the synchronized LSL timestamp of the keystroke:
$$ t \in [t_{LSL}^{key} - 0.5s, t_{LSL}^{key} + 0.5s] $$
Any periods of video viewing that fell completely outside these boundaries were structurally labeled as sustained engagement (`STAY`, Class 1).

### 1.2 Frequency Decomposition and Baseline Normalization
The continuous, uniformly interpolated signal was decomposed into 7 canonical neurophysiological frequency bands ($\delta$: 1-4 Hz, $\theta$: 4-8 Hz, $\alpha$: 8-13 Hz, $\beta$: 13-30 Hz, Low $\gamma$: 30-40 Hz, High $\gamma$: 40-60 Hz, Very High: 60-100 Hz) utilizing a 4th-order Butterworth bandpass filter [Insert citation]. 

**Zero-Phase Digital Filtering:** To prevent non-linear phase distortion—which destroys the temporal morphology of transient event-related potentials—the Butterworth transfer function $H(s) = \frac{1}{1 + (s/\omega_c)^{2n}}$ (where $n=4$) was applied symmetrically in both the forward and reverse temporal directions (`scipy.signal.filtfilt`). This effectively doubles the filter order to 8 while perfectly preserving phase alignment [Insert citation].

**Analytic Amplitude Envelope:** To prevent statistical models from artificially collapsing to zero-variance when high-frequency oscillations rapidly cross the zero-voltage line, instantaneous power was mathematically extracted via the analytic signal rather than simple squared amplitude. The amplitude envelope $P_{env}(t)$ tracks the true morphological boundary of the wave using the Hilbert transform $\mathcal{H}$ [Insert citation]:
$$ P_{env}(t) = \left| x(t) + j \cdot \mathcal{H}\{x(t)\} \right| $$

**Z-Score Baseline Normalization:** To rigorously correct for idiosyncratic physiological noise (e.g., individual skull impedance, baseline cortical arousal), every data point was standardized against the participant's own resting baseline. A dedicated 90-second continuous baseline epoch ($W_{base}$) was extracted exactly 10 seconds following a designated relaxation marker ($t_{B}$), defining the window $t \in [t_{B} + 10s, t_{B} + 100s]$.

For each specific channel and frequency band, the absolute baseline mean ($\mu_{base}$) and standard deviation ($\sigma_{base}$) were computed from this epoch. Every subsequent experimental sample was then mathematically z-score normalized to represent relative physiological deviation rather than absolute microvolts [Insert citation]:
$$ z(t) = \frac{P_{env}(t) - \mu_{base}}{\sigma_{base}} $$

### 1.3 Feature Engineering and Dimensionality
Extracting generalized, deep structural patterns from noisy biological signals required compressing the high-resolution temporal arrays ($256$ Hz) into discrete statistical and geometric vectors. For every uniform 1.0-second analysis window $W$ consisting of $N = 256$ samples, a comprehensive feature matrix was extracted per channel and per frequency band [Insert citation].

**Statistical Compression:** The primary statistical distributions were computed mathematically:
*   **Mean Power:** $\mu_W = \mathbb{E}[W]$
*   **Variance/Standard Deviation:** $\sigma_W = \sqrt{\mathbb{V}[W]}$
*   \textit{Note: The absolute minimum ($\min$) and maximum ($\max$) bounds were mathematically ablated from the final dataset. Variance analysis proved that within tightly constrained 1.0s windows, these extremes often collapsed to zero-variance constants due to the amplitude envelope, unnecessarily inflating the spatial dimensionality.}

**Hjorth Parameters:** To capture the time-domain morphological complexity of the signal independent of standard frequency transforms, Hjorth parameters were computed using the first ($x'$) and second ($x''$) discrete differences of the signal [Insert citation]:
*   **Activity:** $\text{Act} = \mathbb{V}[x(t)]$ (Total signal variance)
*   **Mobility:** $\text{Mob} = \sqrt{\frac{\mathbb{V}[x'(t)]}{\mathbb{V}[x(t)]}}$ (Estimate of the mean frequency)
*   **Complexity:** $\text{Comp} = \frac{\text{Mob}(x'(t))}{\text{Mob}(x(t))} = \frac{\sqrt{\mathbb{V}[x''(t)]/\mathbb{V}[x'(t)]}}{\text{Mob}}$ (Estimate of the bandwidth and deviation from a pure sine wave)

**Relative Band Power:** The normalized spectral contribution of a specific band $b$ within a given channel $c$, proportional to the total active power across all 7 bands for that channel:
$$ P_{rel,b,c} = \frac{\mathbb{E}[P_b]}{\sum_{i=1}^{7} \mathbb{E}[P_{i,c}]} $$

**Dynamic Macro Frequency:** To mathematically quantify sustained oscillatory bursts, a custom thresholding algorithm was implemented. The algorithm identifies contiguous blocks where the signal exceeds the local window mean ($\bar{x}_W$). To prevent sub-millisecond signal drops from falsely dividing a continuous cognitive burst, a dynamic gap threshold $\tau_{gap} = \mathbb{E}[\text{length}(x < \bar{x}_W)]$ is computed. Gaps where $\Delta t \leq \tau_{gap}$ are logically bridged, enabling robust extraction of true macro-oscillations.

**Explicit ERP Extraction:** To explicitly allow the machine learning algorithms to search for raw transient electrical shifts (Event-Related Potentials) [Insert citation], unfiltered raw voltage features were extracted directly from the primary time domain:
*   Raw Mean ($\mu_{raw}$) and Standard Deviation ($\sigma_{raw}$)
*   **Linear Slope:** Calculated via a first-degree polynomial fit over the 1.0s epoch, scaled to the sampling rate $f_s$: $\Delta_{slope} = \text{polyfit}(t, x_{raw}, 1)_0 \times f_s$.

This comprehensive engineering pipeline successfully compressed the raw high-dimensional temporal signal into a dense vector of mathematically rigorous features, enabling classical machine learning pattern recognition.

### 1.4 Window Extraction and Algorithmic Evaluation
**Extraction Strategy:** Epochs were extracted from the labeled regions utilizing a fixed duration $w = 1.0s$ and a strict non-overlapping stride $s = 1.0s$. A window $W_i$ was only appended to the dataset if it fit entirely within the predefined boundary of a contiguous `SKIP` or `STAY` region, ensuring absolute class purity without boundary overlap. 

**Burst Detection:** Rapid, consecutive sequential skips heavily confound cognitive state interpretation. A contiguous `SKIP` block was flagged as a "Burst" if the number of actual synchronized keypress events within the structural block boundary satisfied $n_{presses} \geq 2$.

**Smart Stratified Temporal Split (Cross-Validation):** To rigorously evaluate the predictive capability of the models while physically preventing temporal data leakage, a specialized Cross-Validation architecture was deployed. Standard random $K$-Fold CV is invalid for time-series, and simple chronological block CV fails on highly imbalanced behavioral datasets. 

Our algorithm stratified the chronological blocks based strictly on the cumulative temporal density of the minority class (`SKIP`). Given the ordered set of all `SKIP` timestamps $T_{skip}$, the algorithm placed $K-1$ temporal split markers precisely at the behavior-driven percentiles:
$$ t_{split, k} = T_{skip}\left[ \left\lfloor |T_{skip}| \times \frac{k}{K} \right\rfloor \right] $$
This guaranteed that every test fold contained exactly 20\% of the actual physical swipe events.

**The Temporal Firewall:** To definitively prevent the machine learning models from artificially memorizing continuous background physiological noise spanning across the train/test boundaries, an absolute deletion zone ($\Delta_{gap} = 3.0s$) was enforced at every split marker. Any window $W_i$ falling within the boundary $t_i \in [t_{split, k} - \frac{\Delta_{gap}}{2}, t_{split, k} + \frac{\Delta_{gap}}{2}]$ was permanently purged from the array. This constraint guarantees a strict mathematical separation:
$$ \min(|t_{train} - t_{test}|) \geq 3.0s $$
Finally, the remaining `STAY` windows within the training sets were randomly undersampled using randomized seeds to enforce a strict $1:1$ prior class probability. This formally prevents the Random Forest classifier from defaulting to base-rate topological bias [Insert citation].

---

## 2. Results: The Experimental Progression

We ran four main experiments to test and improve our pipeline. This progression shows how we identified mistakes, adapted our approach, and slowly improved the model's modest, yet meaningful, predictive ability.

### 2.1 The Baseline Paradigm & Algorithmic Overfitting (`run_20260524_220329 ofit`)

This initial analytical run was engineered using conventions from traditional Event-Related Potential (ERP) literature. It was hypothesized that the cognitive decision to swipe was a macro-scale process. Therefore, the physiological boundaries were set to a wide $\pm 3.0s$ around the keypress, isolating a 6.0-second behavioral region (`skip_window_s = 3.0`). To test classical spatial continuity, massive 3.0-second classification windows (`window_s = 3.0`) with a 0.6-second sliding stride (`stride_s = 0.6`) were extracted, enforcing an $80\%$ temporal overlap (`train_overlap = 0.8`) [Insert citation]. 
The 50Hz electrical `notch` filter was evaluated as a toggle, and the bounding statistics (`min`, `max`) were left active. It is important to note that the advanced experimental parameters (Hilbert Amplitude Envelope, Raw ERP Extraction, Min/Max Ablation) were **not yet activated** in this foundational run, serving as a strict conventional baseline.

**Step 06: Hyperparameter Optimization Matrix**
To maximize algorithmic capability, a grid search rigorously evaluated 27 structural permutations of the Random Forest across an internal 3-fold CV. The optimization frequently selected deep, heavily populated forests, with `max_depth = 5`, `min_samples_leaf = 3`, and `n_estimators = 300` consistently yielding the highest inner-recall matrices ($\mu_{inner\_recall} \approx 0.75 - 0.85$). This early structural preference for maximum depth was an initial mathematical indicator of potential over-parameterization.

**Step 07: Algorithmic Overfitting & Generalization Gap**
The performance matrix of the primary model (`Intra | NoNotch | NoArt | NoBurst | RF`) revealed a substantial generalization gap. During training, the deep topological branches adapted closely to the $80\%$ overlapping background noise, achieving a Train F1 of $0.930$ (Train Accuracy: $0.931$). However, when evaluated across the strictly enforced $3.0s$ temporal firewall, the performance declined. The Test F1 fell to $0.438$ (Test Accuracy: $0.542$), representing a generalization gap of $\Delta_{F1} = -0.493$. 

**Step 08 & 09: Significance and the Classical Formula**
A Wilcoxon Signed-Rank Test indicated that the model failed to consistently beat a $50\%$ random behavioral baseline ($W = 102.0$, $p = 0.107$, Rank-Biserial $r = 0.372$). Only $36\%$ ($9/25$) of participants demonstrated predictive markers significantly above chance. Simultaneously, the classical Engagement Index (EI = $\beta / (\alpha + \theta)$) [Insert citation] yielded a Mean Test Recall of $0.482$. A paired non-parametric test verified that the Random Forest was statistically indistinguishable from the EI baseline ($W = 145.0$, $p = 0.653$).

**Step 10: Hardware Filter Ablation**
Evaluating the 50Hz Notch ablation revealed that heavily filtered data slightly outperformed the raw `NoNotch` baseline in this specific scenario ($\Delta_{recall} = +2.2\%$, $W = 85.0$, $p = 0.036$). While statistically significant, both pipelines remained functionally below chance, indicating that filter tuning alone was insufficient to rescue the overlapping spatial geometry.

**Step 11 & 12: Feature Matrices & Algorithmic Robustness**
The Gini-importance matrix was primarily dominated by Frontal High-Gamma complexity (e.g., `AF7_high_gamma_std`, `AF8_high_gamma_max`), suggesting the model may have been utilizing localized high-frequency tension rather than generalized slow-wave cognition.
Algorithmic robustness across demographic subgroups demonstrated no significant deviation across Gender ($p = 0.567$) or TikTok Consumption habits ($p = 0.798$). However, a statistically significant discrepancy arose between Paid and Unpaid participants ($p = 0.035$, Effect $r = 0.506$), highlighting dataset instability under these overlapping conditions.
Furthermore, the temporal exclusion analyses indicated that purging physiological artifact windows ($\Delta_{recall} = +2.7\%$, $p = 0.034$) and logically deleting consecutive "burst-skips" ($\Delta_{recall} = +12.2\%$, $p < 0.001$) significantly stabilized the predictive matrix.

**Step 14: Cross-Participant Generalization (LOGO-CV)**
Testing the algorithm's capability to generalize across unseen participants (Leave-One-Group-Out CV) yielded a modest, non-significant (+10.2%) deviation above chance ($0.602$ vs $0.500$). It should be noted that the corresponding training metrics for the LOGO-CV evaluation were not explicitly tracked in this baseline run, preventing the calculation of a direct generalization delta for this specific step. However, the limited test performance suggested that overlapping intra-participant noise may have negatively impacted universal structural generalization.

**Conclusions & The Structural Pivot**
The scientific conclusions drawn from this initial exploration were incremental: standard long-window continuous EEG methodology, in this specific experimental setup, may not be optimal for rapid micro-behavioral tasks like short-form video consumption. The $80\%$ temporal overlap created a dimensionally dense dataset that resulted in algorithmic overfitting ($\Delta_{F1} = -0.493$). The dominance of high-frequency features alongside the failure of the classic EI ratio suggested that the target cognitive signature may not be a slow 6.0-second physiological shift, but rather a more rapid, transient micro-state.

This methodological evaluation mathematically justified the structural changes enacted in the subsequent pipeline: stripping the overlapping arrays entirely and radically restricting the temporal boundaries.

### 2.2 The 1.0-Second Non-Overlapping Architecture (`run_20260526_113959 +-0.5s`)

To systematically address the algorithmic overfitting observed in the baseline evaluation, two structural parameters were adapted. It was hypothesized that the target cognitive signature was a rapid, transient micro-state rather than a slow macro-shift. Consequently, the physiological boundaries were restricted to a narrow $\pm 0.5s$ window ($window\_s = 1.0$). Furthermore, to eliminate the dimensional density that induced memorization, the spatial overlap was entirely removed, enforcing a strictly contiguous extraction ($stride\_s = 1.0$).

**Step 06: Hyperparameter Optimization Matrix**
The grid search continued to evaluate the 27 structural permutations. While deep forests (`max_depth = 5`) were still selected for some participants, a notable shift toward shallower trees (`max_depth = 2` or `3`) was observed across multiple folds. This reflected a natural mathematical regularization effect driven by the reduced, non-overlapping dataset volume.

**Step 07: Algorithmic Overfitting & Generalization Gap**
The performance matrix of the primary model (`Intra | NoNotch | NoArt | NoBurst | RF`) demonstrated marked improvement. The Train F1 stabilized at $0.921$ (Train Accuracy: $0.915$), while the Test F1 increased significantly to $0.641$ (Test Accuracy: $0.648$). This resulted in a substantially narrowed generalization gap of $\Delta_{F1} = 0.280$ (compared to the $\Delta_{F1} = -0.493$ gap in the baseline), indicating that the removal of overlapping data in combination with reduced window size and a reduced SKIP window significantly reduced background noise memorization.

**Step 08 & 09: Significance and the Classical Formula**
Unlike the baseline, a Wilcoxon Signed-Rank Test confirmed that the primary model (Mean Test Recall: $0.694$) performed significantly above the $50\%$ random chance baseline ($W = 9.0$, $p = 2 \times 10^{-6}$, Effect $r = 0.945$). Furthermore, the algorithmic approach statistically outperformed the classical Engagement Index formula ($0.483$ Test Recall), yielding a highly significant difference ($W = 44.0$, $p = 0.0008$, Effect $r = 0.729$).

**Step 10: Hardware Filter Ablation**
Evaluating the 50Hz Notch filter ablation revealed a complete reversal from the baseline. The raw, unfiltered `NoNotch` pipeline significantly outperformed the `Notch` filtered data ($\Delta_{recall} = +15.3\%$, $W = 15.0$, $p = 8 \times 10^{-6}$, Effect $r = 0.908$). This suggested that standard electrical filtering may inadvertently erase critical high-frequency cognitive markers when working with narrow 1.0-second non-overlapping windows.

**Step 11 & 12: Feature Matrices & Algorithmic Robustness**
The Gini-importance matrix exhibited a profound shift. Rather than Frontal High-Gamma complexity, the top predictive features transitioned to Temporal and Frontal Theta Peak Frequencies (e.g., `TP9_theta_peakfreq`, `TP10_theta_peakfreq`, `AF8_theta_peakfreq`). This shift aligns more closely with established literature regarding motor-cognitive initiation and rapid attention switching.
Importantly, algorithmic robustness stabilized across all evaluated demographic subgroups. The concerning discrepancy between Paid and Unpaid participants observed in the baseline was completely resolved ($p = 0.396$), indicating that the dataset instability was largely a byproduct of the overlapping geometry in combination with narrow 1.0-second windows and the reduced SKIP durations. 

**Step 14: Cross-Participant Generalization (LOGO-CV)**
Descriptive evaluation of the Leave-One-Group-Out CV yielded a highly elevated test recall ($0.944$) compared to chance ($0.500$). While the corresponding training metrics were not logged for this specific fold, the stabilization of the intra-participant metrics suggested an improved capacity for generalized structural extraction.

**Conclusions & The Regularization Pivot**
The adaptation to a strictly contiguous 1.0-second window successfully halted the extreme memorization seen in the baseline run. The algorithm performed significantly above chance, and the feature reliance shifted toward physiologically plausible Theta peak frequencies. However, despite the improved stability, a notable $\Delta_{F1} = 0.280$ generalization gap persisted. It was hypothesized that projecting the high dimensionality of 224 statistical features onto a restricted 1.0-second window continued to allow minor algorithmic memorization. This observation mathematically justified the next structural progression: enforcing strict algorithmic regularization to artificially cap model complexity.

### 2.3 Algorithmic Regularization & The Research Question (`run_20260526_133703 smaller RF`)

Although the 1.0-second non-overlapping adaptation successfully halted extreme memorization, a notable $\Delta_{F1} = 0.280$ generalization gap persisted. It was hypothesized that the high dimensionality of 224 statistical features still permitted minor algorithmic overfitting. To counteract this, strict structural regularization was manually enforced. The grid search was bypassed, and the Random Forest was strictly capped at `max_depth = 3` and `n_estimators = 100`, forcing a mathematically constrained topology.

**Step 07: Algorithmic Overfitting & Generalization Gap**
This strict regularization successfully compressed the generalization gap. The Train F1 fell to $0.835$ (Train Accuracy: $0.829$), while the Test F1 stabilized at $0.619$ (Test Accuracy: $0.631$). This yielded a heavily reduced gap of $\Delta_{F1} = 0.216$, confirming that the algorithm was now less capable of deep memorization.

**Step 08 & 09: Significance and the Classical Formula**
A Wilcoxon Signed-Rank Test confirmed that the regularized model maintained performance significantly above the 50% chance baseline ($W = 23.0$, $p = 3.8 \times 10^{-5}$, Effect $r = 0.858$), validating that the extracted patterns were legitimate despite the severe architectural restrictions. It also continued to significantly outperform the EI formula ($p = 0.003$).

**Step 10: Hardware Filter Ablation**
The raw `NoNotch` pipeline maintained its superiority over the filtered data ($\Delta_{recall} = +12.8\%$, $W = 34.0$, $p = 0.0002$, Effect $r = 0.791$), confirming the earlier observation from Section 2.2 that 50Hz hardware filtering may actively erase critical cognitive markers in this specific non-overlapping temporal setup in the attempt to predict micro-decisions during naturalistic TikTok consumption.

**Step 11 & 12: Feature Matrices & Algorithmic Robustness**
The feature importance matrix remained stable compared to the previous run, with Temporal and Frontal Theta Peak Frequencies (`TP10_theta_peakfreq`, `TP9_theta_peakfreq`, `AF8_theta_peakfreq`) continuing to dominate the predictive topology. Notably, the impact of temporal exclusions (artifact windows and burst-skips) dropped to non-significance ($p = 0.126$ and $p = 0.262$ respectively), suggesting the regularized, non-overlapping windows were inherently robust against localized noise bursts.

**Addressing the Research Question & LOGO-CV**
At this methodological stage, the primary overarching research hypothesis—whether a 4-channel consumer-grade EEG can reliably classify sustained cognitive engagement during naturalistic TikTok browsing at the intra-subject level—was answered affirmatively. Specifically for this homogeneous sample and dataset, using this constrained Random Forest methodology, it was concluded that the pipeline predicted engagement significantly above chance ($p = 3.8 \times 10^{-5}$), albeit with a modest, individually variable effect size. 
Addressing the secondary research question, the methodology yielded evidence of partial cross-individual consistency. Cross-subject generalization via Leave-One-Group-Out Cross-Validation (LOGO-CV) yielded a mean accuracy of $65.3\%$. While superior to random chance, this performance drop compared to intra-subject metrics underscores that the neural signature of swipe-initiation remains highly individualized.

**Justification for the Final Progression**
With the algorithmic architecture fully regularized and mathematical overfitting minimized, the final experimental progression was justified. The structural parameters were stabilized; therefore, the subsequent step focused entirely on maximizing the physiological signal-to-noise ratio by activating the three advanced feature-extraction toggles (Hilbert Envelope, Raw ERP, and Min/Max Ablation).

### 2.4 The Experimental Signal Pipeline (`run_20260526_150106 3 True`)

With the algorithmic architecture strictly regularized against overfitting (`max_depth = 3`), the focus shifted entirely to maximizing the physiological signal-to-noise ratio. Based on theoretical vulnerabilities identified in the earlier phases, three explicit signal modifications were enacted:
1.  **Hilbert Amplitude Envelope:** Computed to track macroscopic wave morphology rather than volatile high-frequency oscillations.
2.  **Raw ERP Extraction:** The unfiltered voltage timeseries was preserved and appended to the spectral matrices to capture classic Event-Related slow waves.
3.  **Min/Max Ablation:** The extreme bounding statistics were explicitly ablated, as they were theorized to introduce erratic mathematical instability.

**Step 07: Algorithmic Overfitting & Generalization Gap**
These signal modifications yielded the most robust intra-participant matrix of the entire progression. While the Train F1 decreased slightly to $0.819$ (Train Accuracy: $0.801$), the generalized Test F1 reached its peak at $0.672$ (Test Accuracy: $0.658$). Consequently, the generalization gap collapsed to $\Delta_{F1} = 0.147$, establishing that the enhanced physiological features allowed the constrained model to extract significantly more generalized patterns without reverting to memorization.

**Step 08 & 09: Significance and the Classical Formula**
The primary model's Mean Test Recall reached $0.752$. A Wilcoxon Signed-Rank Test confirmed this performance was profoundly above the 50% chance baseline ($W = 1.0$, $p < 10^{-6}$, Effect $r = 0.994$). Furthermore, the Random Forest definitively outperformed the classical Engagement Index formula ($0.529$ Test Recall) with high statistical significance ($p = 0.000188$, Effect $r = 0.797$), validating the necessity of machine-learning-derived features over simple historic ratios for rapid continuous tasks.

**Step 11 & 12: Feature Matrices & Algorithmic Robustness**
The feature topography diversified significantly following the introduction of the experimental toggles. Rather than a single dominant frequency band, predictive importance was broadly distributed across the full spectral structure, with Delta, Theta, and Low-Gamma macro-frequencies (e.g., `TP9_delta_macrofreq`, `TP9_low_gamma_peakfreq`, `TP10_delta_macrofreq`) driving the top ranks. This indicated the algorithm was no longer exploiting localized tension artifacts, but rather synthesizing a complex, multi-band cognitive signature. Furthermore, algorithmic robustness remained perfectly stable across all demographic subgroups.

**LOGO-CV, The Research Question, & Future Implications**
Evaluating the Leave-One-Group-Out CV (LOGO-CV) provided the definitive answer to the study's overarching hypotheses. 

First, regarding the primary research question: specifically for this homogeneous demographic sample and this specific consumer-grade hardware setup, it was concluded that the methodology can indeed classify sustained cognitive engagement at the intra-subject level significantly better than random chance ($p < 10^{-6}$), albeit with a modest effect size.

Addressing the secondary hypothesis exploring cross-individual consistency, the final model achieved its highest generalization metrics. The LOGO-CV mean cross-subject test accuracy reached $65.9\%$ (Aggregate Accuracy: $66.7\%$). 

Neuroscientifically, this yields a nuanced conclusion: the algorithm successfully extracted genuine cognitive markers within individual brains, and these markers do share a partial baseline structure across the demographic sample (allowing the 66% cross-subject generalization). However, the performance gap between individualized training and generalized testing confirms that the neural signature of continuous video engagement retains highly idiosyncratic components. The cognitive manifestation of "swiping" does not natively translate across different brains with high fidelity without subject-specific calibration. This inherent biological variability means that while intra-subject BCI applications are feasible, future investigations must explore advanced personalized transfer-learning paradigms or zero-shot foundation models to bridge the inter-subject divide.

---

## 3. Discussion

The results of this exploratory study provide encouraging preliminary support for a simple idea: that a basic, consumer-grade EEG headband might be able to detect the brief neural shifts that happen right before someone decides to skip a short-form video. 

### 3.1 Resolving the Research Goal
Our data suggests that during rapid TikTok browsing, "engagement" is very hard to measure as a slow, continuous mood over many seconds. Traditional metrics struggle here. However, when we zoom in on the single second surrounding the swipe, a more consistent pattern emerges. The model appears to be detecting a brief, transient signal—likely the brain's "readiness" to move the thumb, or a quick spark of disinterest. Because our simple, restricted model performed similarly on unseen participants, it suggests this brief signal might share common traits across different people. 

**What We Cannot Claim:** It is crucial to note that we cannot definitively prove the model is measuring "pure cognitive boredom." It is highly likely that the model is heavily relying on the "Motor Readiness Potential"—the brain's signal preparing to move the thumb to swipe. Furthermore, our sample size of 25 university-aged individuals is very small and homogeneous. We absolutely cannot claim these specific algorithms would generalize to older adults, young children, or broader, more diverse populations. 

### 3.2 Practical Applications: Markerless Detection
Despite these limitations, this 1-second predictive approach presents an interesting concept for future research: **markerless skip-intent detection**. 
If a simple headband can guess that a user wants to skip a video fractions of a second *before* their thumb actually moves, we could potentially study attention spans without making people physically swipe at all. In future public health or media studies, researchers could observe viewers watching content completely hands-free, using the EEG to map exactly when their attention breaks, independent of their physical reaction time.

### 3.3 Future Work and Limitations
This study is a humble first step, a proof-of-concept showing that data pipeline architecture matters deeply when using noisy consumer hardware. To build on this, future research must address several clear limitations:
1.  **Feature Distillation:** Even our best model uses over 200 features, which is far too many for such noisy data. Future studies should try to strip away 90% of these features to find the exact 5 or 10 specific brainwave shapes that actually matter. This will make the models much faster and less prone to errors.
2.  **Tighter Stimulus Locking:** To prove whether the signal is purely a motor-planning reflex, future pipelines need to look at an even narrower slice of time—perhaps just the 200 to 600 milliseconds before the swipe happens.
3.  **Representative Studies:** This entire pipeline must be re-tested on a much larger, demographically diverse group of people. Consumer EEG is notoriously sensitive to different hair types, head shapes, and skin variations. Until this is validated across a wider population, these results remain an interesting, but isolated, case study.
