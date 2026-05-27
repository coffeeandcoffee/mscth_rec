# Chapter: Methods and Results

## 1. Materials and Methods

We built a data pipeline to turn continuous, unstructured EEG data—recorded while people scrolled through TikTok—into a clear format that a machine learning model could learn from. The following sections explain how we isolated, processed, and evaluated the brain signals related to staying engaged (`STAY`) versus deciding to skip a video (`SKIP`).

### 1.1 Signal Preprocessing and Labeling
The raw EEG stream was recorded using the Muse S (Gen 2) headband with four dry electrodes (TP9, AF7, AF8, TP10). Because consumer Bluetooth devices often have slight timing irregularities, we first cleaned the timeline.

**Uniform Interpolation:** We aligned the continuous brainwave data to a steady 256 Hz timeline. This ensures all data points are evenly spaced, which is a requirement for accurate frequency analysis.

**The $\pm$0.5s Isolation Paradigm:** Our main goal was to isolate the brain's state right before the physical thumb swipe happened. We hypothesized that the decision to skip might show up as a brief "readiness" signal in the brain. Therefore, we defined the `SKIP` class (Class 0) as a strict 1.0-second window centered exactly on the moment the button was pressed:
$$ t \in [t_{press} - 0.5s, t_{press} + 0.5s] $$
Any periods of video watching that fell completely outside these `SKIP` windows were labeled as sustained engagement (`STAY`, Class 1).

### 1.2 Filtering and Baseline Normalization
We separated the continuous brainwaves into 7 standard frequency bands (Delta, Theta, Alpha, Beta, Low Gamma, High Gamma, Very High) using a standard 4th-order Butterworth bandpass filter. 

**Zero-Phase Filtering:** We applied the filter equation $H(s) = \frac{1}{1 + (s/\omega_c)^{2n}}$ both forwards and backwards. This is a standard technique to ensure the shape of the brainwaves isn't distorted in time.

**Amplitude Envelope Calculation:** High-frequency waves cross the zero-line very quickly, which can confuse statistical models. To fix this, we calculated the "amplitude envelope" using the Hilbert transform. This tracks the overall outer shape of the wave's power rather than its rapid up-and-down swings:
$$ P_{env}(t) = \left| x(t) + j \cdot \mathcal{H}\{x(t)\} \right| $$

**Relative Power Normalization:** Every person's skull thickness and skin conductivity is slightly different. To make the data comparable across participants, we divided each active brainwave reading by that person's own resting baseline (recorded during a 100-second relaxation period):
$$ P_{rel}(t) = \frac{P_{env}(t)}{\mu_{baseline}} $$

### 1.3 Feature Engineering and Dimensionality
To help the machine learning model find patterns, we condensed the complex, high-speed brainwaves into simple summary statistics for every 1-second window.

**Statistical Compression:** For each 1.0-second window $W$, we calculated basic descriptive metrics:
*   **Mean:** $\mu_W = \mathbb{E}[W]$
*   **Standard Deviation:** $\sigma_W = \sqrt{\mathbb{V}[W]}$
*   \textit{Note: We initially calculated the minimum and maximum values as well. However, in such short 1-second windows, these often hit zero and provided no useful variance, so we removed them to keep the model clean.}

**Hjorth Parameters:** These parameters help describe how complex or "jagged" the brainwave shape is, without just looking at raw power.
*   **Activity:** $\text{Act} = \mathbb{V}[x(t)]$ (how much the signal varies)
*   **Mobility:** $\text{Mob} = \sqrt{\frac{\mathbb{V}[x'(t)]}{\mathbb{V}[x(t)]}}$ (an estimate of the average frequency)
*   **Complexity:** $\text{Comp} = \frac{\text{Mob}(x'(t))}{\text{Mob}(x(t))}$ (how much the frequency changes)

**Relative Band Power:** We calculated what percentage of a channel's total power was occupied by each specific frequency band.

**Dynamic Macro Frequency:** To count sustained bursts of brainwaves, we used a peak-detection algorithm. Because real brainwaves flutter rapidly, we used a dynamic threshold $\tau$ to bridge tiny gaps, allowing us to count true, sustained wave oscillations rather than just noise.

**Explicit ERP Extraction:** To see if the model could detect raw electrical shifts (Event-Related Potentials, or ERPs) related to moving the thumb, we also looked at the unfiltered raw voltage. We extracted the raw mean ($\mu_{raw}$), raw standard deviation ($\sigma_{raw}$), and the overall slope or trend of the wave over that 1 second.

In total, this process gave the model **224 simple numbers (features)** to look at for every single second of data.

### 1.4 Window Extraction and Cross-Validation
**Extraction Strategy:** We extracted these 1.0-second windows with no overlapping ($stride\_s = 1.0s$). If a participant rapidly skipped multiple times within a single second, we flagged it as a "Burst" to handle it carefully.

**Smart Stratified Temporal Split:** To fairly test if our model actually learned anything—and wasn't just memorizing data—we built a strict chronological firewall. We split the data over time, but we always forced an absolute 3.0-second dead zone ($\Delta_{gap}$) between any data the model learned from and data it was tested on:
$$ \min(|t_{train} - t_{test}|) \geq 3.0s $$
Finally, to prevent the model from just guessing `STAY` because it happens more often, we randomly discarded extra `STAY` windows to create a perfectly balanced 50/50 dataset.

---

## 2. Results: The Experimental Progression

We ran four main experiments to test and improve our pipeline. This progression shows how we identified mistakes, adapted our approach, and slowly improved the model's modest, yet meaningful, predictive ability.

### 2.1 The Baseline & Overfitting (`run_20260524_220329 ofit`)
Initially, we used a very wide 6.0-second window ($\pm$3.0s) and let the windows overlap by 80%. 
This approach failed. The Random Forest models showed severe "overfitting"—they scored incredibly high on the training data but failed on unseen test data. The deep decision trees were just memorizing the overlapping background noise instead of learning genuine brain patterns. We also tested the classic "Engagement Index" formula here, and found it performed no better than a coin flip (around 50% accuracy), suggesting that simple formulas are not enough for rapid TikTok scrolling.

### 2.2 The 1-Second Breakthrough (`run_20260526_113959 +-0.5s`)
To stop the model from getting confused by 6 seconds of mixed thoughts, we shrank the focus to just $\pm$0.5s (1.0s total) around the swipe, and stopped the windows from overlapping.
*   **A Clearer Picture:** This simple change stopped the model from collapsing. While the scores were lower overall (Test F1 = 0.564), they were much more honest, and the model was finally performing slightly above random chance.
*   **The Hardware Filter Issue:** Interestingly, using the raw, unfiltered data ("NoNotch") performed notably better than using data passed through a standard 50Hz electrical noise filter. This suggests that the 50Hz filter might accidentally be erasing real, high-frequency brain activity (like Gamma waves) that is useful for prediction.

### 2.3 Strict Regularization (`run_20260526_133703 smaller RF`)
Even with 1-second windows, having 224 features meant the model was still trying to memorize noise. To fix this, we strictly limited how complex the Random Forest model could get by capping its depth (`max_depth = 3`) and the number of trees (`n_estimators = 100`).
By forcing the model to be simple, it could no longer memorize the data. The gap between training scores and testing scores shrank significantly. Most importantly, when we tested the model on entirely new participants it had never seen before (LOGO-CV), it maintained a stable performance (Test F1: 0.723). This gave us preliminary hope that the pattern it found was somewhat shared across people.

### 2.4 The Experimental Toggles (`run_20260526_150106 3 True`)
In our final attempt, we applied three specific changes based on our earlier observations:
1.  Using the Hilbert Amplitude Envelope to track wave shapes more smoothly.
2.  Adding the raw, unfiltered voltage features (ERP extraction).
3.  Removing the minimum/maximum features that were causing math errors.

These three modest changes resulted in our best outcome. For individual participants, the Test F1 score improved by +5.2\% (to 0.671) and Test Recall improved by +8.4\% (to 0.752). 
When testing across different people (LOGO-CV), the difference between the training score and the testing score dropped to less than 1\%. While the overall accuracy remains modest, this tiny gap means we successfully stopped the model from "cheating" by memorizing noise. 

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
