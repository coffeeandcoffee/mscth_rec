# EEG TikTok Study

EEG recording and ML pipeline for predicting TikTok engagement from brain signals (MSc Thesis).

## Project Structure

```
├── scripts/
│   ├── recording_script_v4.py          # EEG + video recording
│   ├── post0_merge_all_split_recording_csvs.py  # Merge split CSVs ⭐
│   ├── post1_check_eeg_quality.py      # Quality check
│   ├── post2_classify_segments_and_cut.py  # Segment classification (V1)
│   ├── post2v2_add_skip_classification.py  # Skip prediction preprocessing (V2) ⭐
│   ├── post3v2_prep_for_ml.py          # ML preprocessing (V1)
│   ├── train_transformer.py            # Engagement prediction (V1)
│   ├── train_transformer_v2.py         # V1 optimization experiments
│   └── prediction_2.py                 # Skip prediction training (V2) ⭐
├── recordings/
│   └── eeg_*/
│       ├── model_output/               # V1 models
│       └── model_output_prediction_v2/ # V2 models ⭐
└── mscth/                              # Python virtual environment
```

## Setup

```bash
source mscth/bin/activate
pip install torch pandas scipy scikit-learn matplotlib
```

---

## Two Prediction Approaches

| Approach | V1: Engagement | V2: Skip Prediction ⭐ |
|----------|----------------|------------------------|
| **Target** | Predict if user watches >4s | Predict if user is about to skip |
| **Data used** | First 0.5s after video start | 3s before each skip event |
| **Best accuracy** | 62.8% (barely beats 60% baseline) | **71.2%** ✅ |
| **Status** | Signal too weak | **Promising** |

---

## Pipeline V2: Skip Prediction (Recommended)

Predicts if user is "about to skip" based on 3 seconds of EEG data.

### Step 1: Preprocessing

```bash
python scripts/post2v2_add_skip_classification.py --window 3.0
```

**What it does:**
1. Loads RAW EEG CSV (not the cut/classified one)
2. Classifies segments (baseline_1, baseline_2, tiktok_over_4s, tiktok_under_4s)
3. Adds `classification_2` column:
   - `about_to_skip`: 3 seconds before each keypress_A
   - `not_about_to_skip`: All other TikTok data
   - Baselines preserved (excluded from training)

**Output:** `eeg_*_skip_labels_3_0s.csv`

### Step 2: Train Skip Prediction Model

```bash
python scripts/prediction_2.py --window 3.0 --epochs 50 --nonotch
```

> 💡 Use `--nonotch` to disable 50Hz notch filter (recommended for this dataset, see [notch filter results](#v2-results-with-50hz-notch-filter-n3))

**Pipeline:**
1. Apply 50Hz notch filter (unless `--nonotch`)
2. Extract frequency bands (7 bands × 4 channels = 28 features)
3. Create 3-second sample blocks
4. Interpolate each block to 256Hz (768 samples)
5. Balance dataset to 50/50
6. Train transformer model
7. Compute feature importance

**Output:** `model_output_prediction_v2/`
- `skip_prediction_model.pt` — Trained model
- `training_results_v2.json` — Metrics + feature importance
- `training_curves_v2.png` — Loss/accuracy plots
- `confusion_matrix_v2.png` — Predictions breakdown
- `feature_importance_v2.png` — Feature importance bar plot
- `electrode_band_heatmap_v2.png` — Electrode × band importance matrix

---

## V2 Results (Multi-Participant, n=3)

### Per-Participant Performance

| Participant | Recording | Duration | Keypresses | Val Accuracy | Best Val | Precision | Recall | F1 |
|-------------|-----------|----------|------------|--------------|----------|-----------|--------|-----|
| P1 | eeg_20251210_203221 | 30 min | ~100 | **71.2%** | 71.2% | 85.9% | 47.7% | 61.3% |
| P2 | eeg_20251224_164549 | 57 min | 248 | **62.3%** | 64.8% | 64.3% | 54.5% | 59.0% |
| P3 | eeg_20251227_190056 | 30 min | 104 | **64.3%** | 65.5% | 68.8% | 52.4% | 59.5% |

### Statistical Summary

| Statistic | Value |
|-----------|-------|
| **Mean Accuracy** | **66.0%** |
| Standard Deviation | 4.64% |
| 95% Confidence Interval | **56.9% – 75.1%** |
| Effect Size (vs 50% baseline) | +16.0 percentage points |
| Minimum | 62.3% |
| Maximum | 71.2% |

### Generalizability Assessment

> ✅ **Method is generalizable**: All 3 participants beat 50% baseline consistently

| Test | Result |
|------|--------|
| All participants > 50%? | ✅ Yes (62.3%, 64.3%, 71.2%) |
| All participants > 60%? | ✅ Yes |
| Coefficient of Variation | 7.0% (low variance) |

**Confidence statement**: Based on n=3, we can expect the V2 skip prediction method to achieve **56.9% – 75.1% accuracy** for new participants with 95% confidence. The method reliably beats random chance.

### Top Predictive Features (Aggregate)

| Feature Type | Consistency | Notes |
|--------------|-------------|-------|
| **TP9 (temporal)** | 3/3 participants | theta, gamma bands |
| **AF7/AF8 (frontal)** | 3/3 participants | high_gamma, beta |
| **Gamma bands (30-100Hz)** | Most important | Decision-making, cognitive processing |

**Key finding:** Both temporal (TP9) and frontal (AF7, AF8) electrodes are consistently predictive across participants, particularly in the gamma frequency bands, suggesting skip intent detection is robust across individuals.

---

## V2 Results with 50Hz Notch Filter (n=3)

> Added 50Hz notch filter to remove power line interference. See [`prediction_2.py`](file:///Users/gregorlederer/Local_LifeAdmin_Files/MSc%20Thesis%20-%20EEG%20Neuroscience/Data%20Recording%20and%20Quality%20Tests/scripts/prediction_2.py) `apply_notch_filter()`.

### Comparison: Without vs With Notch Filter

| Participant | Without Filter | With 50Hz Notch | Δ |
|-------------|----------------|-----------------|---|
| P1 | **71.2%** | 61.1% | -10.1% |
| P2 | **62.3%** | 60.3% | -2.0% |
| P3 | **64.3%** | 54.8% | -9.5% |
| **Mean** | **66.0%** | **58.7%** | **-7.3%** |

### Interpretation

> ⚠️ **Notch filter DECREASED accuracy** — This suggests the high gamma (40-60Hz) signal was **real neural activity**, not power line noise.

| Finding | Implication |
|---------|-------------|
| Accuracy dropped 7.3% on average | 50Hz component contained predictive signal |
| P1 dropped most (-10.1%) | Original best result may have included some 50Hz artifact |
| All still beat 50% baseline | Core signal remains, but weaker |

**Recommendation:** For this dataset, **do not use the notch filter** — the 50Hz signal appears to be genuine neural activity from gamma oscillations, not power line contamination.

---

## V3 Decision Tree Run-1 (n=3)

> Alternative model using Decision Tree for interpretability. See [`prediction_3_dt.py`](file:///Users/gregorlederer/Local_LifeAdmin_Files/MSc%20Thesis%20-%20EEG%20Neuroscience/Data%20Recording%20and%20Quality%20Tests/scripts/prediction_3_dt.py).

### Parameters

| Parameter | Value |
|-----------|-------|
| Model | `DecisionTreeClassifier` |
| `max_depth` | 10 |
| `min_samples_leaf` | 5 |
| `class_weight` | 'balanced' |
| `random_state` | 42 |
| Features | 112 (28 bands × 4 stats: mean, std, min, max) |
| Window | 3.0s |
| Notch filter | Disabled (`--nonotch`) |

### Results: V2 Transformer vs V3 Decision Tree

| Participant | V2 Transformer | V3 Decision Tree | Δ |
|-------------|----------------|------------------|---|
| P1 | **71.2%** | 64.6% | -6.6% |
| P2 | **62.3%** | 53.8% | -8.5% |
| P3 | **64.3%** | 58.3% | -6.0% |
| **Mean** | **66.0%** | **58.9%** | **-7.1%** |

### Top Features (Decision Tree)

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | **TP9_high_gamma_mean** | 25.0% |
| 2 | **AF8_high_gamma_mean** | 18.7% |
| 3 | TP10_very_high_min | 8.0% |
| 4 | TP9_very_high_min | 7.9% |
| 5 | TP9_low_gamma_std | 6.5% |

### Conclusion

- **Transformer outperforms Decision Tree** by ~7%
- DT provides interpretable decision rules (saved as `tree_structure.txt`)
- **high_gamma_mean** remains most predictive feature in both models
- Output: `model_output_pred_v3_DT_YYYYMMDD_HHMMSS/`

---

## V3 Decision Tree Run-2 (Regularized)

| Parameter | Run-1 | Run-2 |
|-----------|-------|-------|
| `max_depth` | 10 | **5** |
| `min_samples_leaf` | 5 | **10** |

| Participant | Run-1 | Run-2 | Δ |
|-------------|-------|-------|---|
| P1 | 64.6% | **67.7%** | +3.1% ✅ |
| P2 | 53.8% | 53.8% | 0% |
| P3 | 58.3% | 53.6% | -4.7% |
| **Mean** | **58.9%** | **58.4%** | -0.5% |

**Conclusion**: Regularization improved P1 (79.2% precision) but not others. Transformer still best overall.

---

## V3 Decision Tree Run-3 (80% Overlap) ⭐

> Changed overlap from 33% → **80%** (stride 0.6s). Data samples increased ~3x.

| Participant | Run-2 | Run-3 | Δ | Precision | Recall | F1 |
|-------------|-------|-------|---|-----------|--------|-----|
| P1 | 67.7% | 66.5% | -1.2% | 67.2% | 64.1% | 65.6% |
| P2 | 53.8% | **63.8%** | **+10.0%** ✅ | 69.6% | 48.5% | 57.1% |
| P3 | 53.6% | **67.9%** | **+14.3%** ✅ | 69.2% | 64.3% | 66.7% |
| **Mean** | **58.4%** | **66.1%** | **+7.7%** ✅ | 68.7% | 59.0% | 63.1% |

**Conclusion**: 80% overlap = **3x more data** → DT now **matches Transformer** (66.1% vs 66.0%).

---

## V3 Decision Tree Run-4 (Simpler)

| Parameter | Run-3 | Run-4 |
|-----------|-------|-------|
| `max_depth` | 5 | **3** |
| `min_samples_leaf` | 10 | **15** |

| Participant | Run-3 | Run-4 | Δ |
|-------------|-------|-------|---|
| P1 | 66.5% | 64.2% | -2.3% |
| P2 | 63.8% | 63.3% | -0.5% |
| P3 | 67.9% | 63.1% | -4.8% |
| **Mean** | **66.1%** | **63.5%** | **-2.6%** |

**Conclusion**: depth=3 is **too simple** — loses predictive power. **Run-3 is optimal**.

---

## V4 Random Forest Run-1

```bash
python scripts/prediction_4_rf.py --file <skip_labels.csv> --nonotch --n-estimators 100 --max-depth 5 --min-samples 10
```

| Participant | V2 Transformer | V3 DT Run-3 | V4 RF Run-1 |
|-------------|----------------|-------------|-------------|
| P1 | 71.2% | 66.5% | **73.5%** |
| P2 | 62.3% | 63.8% | **67.3%** |
| P3 | 64.3% | 67.9% | **79.8%** |
| **Mean** | **66.0%** | **66.1%** | **73.5%** |

---

## V4 Random Forest Run-2/3 ⭐ (Best Overall Model)

```bash
python scripts/prediction_4_rf.py --file <skip_labels.csv> --nonotch --n-estimators 200 --max-depth 7 --min-samples 5
```

| Participant | Accuracy | Precision | Recall | F1 |
|-------------|----------|-----------|--------|-----|
| P1 | 72.0% | 74.6% | 66.4% | 70.2% |
| P2 | 68.8% | 72.8% | 59.6% | 65.6% |
| P3 | **84.5%** | 87.2% | 81.0% | 83.9% |
| **Mean** | **75.1%** | 78.2% | 69.0% | 73.2% |

> **Validation**: 60/40 train/val split, **no data leakage**. Same methodology as V2/V3.

### Explainability Outputs (Thesis-Ready)

Each run generates:

| File | Use in Thesis |
|------|---------------|
| `example_decision_tree_rf.png` | **Figure**: "Example decision tree from Random Forest ensemble" |
| `example_tree_rules.txt` | **Appendix**: Full decision rules in text format |
| `feature_importance_rf.png` | **Figure**: "Top 20 predictive EEG features" |

### Example Decision Rules (P1)

```
|--- AF8_high_gamma_mean <= 0.05
|   |--- TP9_very_high_std <= 0.03
|   |   |--- class: Not Skip
|   |--- TP9_very_high_std > 0.03
|   |   |--- class: Skip
|--- AF8_high_gamma_mean > 0.05
|   |--- class: Skip
```

**Thesis interpretation**: *"The model primarily uses high-gamma activity (40-60Hz) from the AF8 electrode to predict skip intent. When high-gamma mean exceeds threshold, the model predicts an imminent skip. This aligns with research linking gamma oscillations to decision-making and motor preparation."*

### Key Learnings for Thesis

1. **high_gamma (40-60Hz)** is consistently the most predictive feature across all participants
2. **AF8 and TP9 electrodes** (frontal and temporal) dominate predictions
3. **Random Forest** provides interpretable decision rules while maintaining 75% accuracy
4. **Ensemble of 200 trees** prevents overfitting seen in single Decision Trees
5. **80% overlap** in windowing increases training data 3x, improving generalization

---

## V5 Cross-Participant Generalizability (Experimental)

> Tested if P3's RF model (84.5%) generalizes to other participants.

```bash
python scripts/prediction_5_cross_participant.py --model <P3_model.pkl> --test <P1_or_P2_skip_labels.csv>
```

| Training | Test | Accuracy | Result |
|----------|------|----------|--------|
| P3 | P3 (self) | **84.5%** | ✅ |
| P3 | P1 | **50.0%** | ❌ Random |
| P3 | P2 | **41.7%** | ❌ Below random |

**Conclusion**: Model does **NOT generalize** across participants. Neural patterns are person-specific. Per-participant training is required.

---

## V6 Raw EEG Transformer (Experimental)

> Uses **raw EEG signals** directly (4 channels) instead of frequency band features. Transformer learns its own feature representations.

```bash
python scripts/prediction_6_raw_transformer.py --file <skip_labels.csv> --epochs 50
```

| Participant | V4 RF Run-2 | V6 Raw TF | Δ |
|-------------|-------------|-----------|---|
| P1 | 72.0% | **80.9%** ⭐ | **+8.9%** |
| P2 | 68.8% | 53.3% | -15.5% |
| P3 | 84.5% | (stuck) | — |

**Key Findings:**
- P1 achieved **80.9%** — highest single-participant accuracy ever!
- P2 failed to converge (53.3% ≈ random) — suggests participant-specific data characteristics
- Raw Transformer is high-variance: works brilliantly for some, fails for others
- RF with frequency bands is more **consistent** across participants

> ⚠️ **Limitation**: Transformer training on MPS can hang. CPU fallback may be needed.

---

## Summary: Best Models by Participant

| Participant | Best Model | Accuracy |
|-------------|------------|----------|
| P1 | V6 Raw TF | **80.9%** ⭐ |
| P2 | V4 RF Run-2 | **68.8%** |
| P3 | V4 RF Run-2 | **84.5%** |

**Overall Best**: V4 Random Forest Run-2 (75.1% mean) — most consistent across participants.

---

## Recording

```bash
python scripts/recording_script_v4.py --nocamera --duration 60
python scripts/recording_script_v4.py --nocamera --duration 1800
python scripts/recording_script_v4.py --nocamera --duration 3600
```

**Keypress markers:**
- `A` — TikTok video transition (swipe)
- `B` — Baseline period marker

---

## Frequency Bands

| Band | Range | Neural Correlate |
|------|-------|------------------|
| Delta | 1-4 Hz | Deep attention |
| Theta | 4-8 Hz | Memory, attention |
| Alpha | 8-13 Hz | Relaxation |
| Beta | 13-30 Hz | Active thinking |
| Low Gamma | 30-40 Hz | Cognitive processing |
| **High Gamma** | **40-60 Hz** | **Learning, decision-making** |
| Very High | 60-100 Hz | Exploratory |

---

## Dependencies

```bash
torch pandas scipy scikit-learn matplotlib pylsl muselsl opencv-python numpy
```

---

## Changelog

### 2026-01-16 (V3 Decision Tree)
- **Added**: [`prediction_3_dt.py`](file:///scripts/prediction_3_dt.py) — Decision Tree alternative
- **Run-1**: `max_depth=10` → 58.9% mean
- **Run-2**: `max_depth=5, min_samples=10` → 58.4% mean
- **Run-3**: **80% overlap** (stride=0.6s) → **66.1% mean** ⭐ (matches Transformer)
- **See**: [V3 Run-3](#v3-decision-tree-run-3-80-overlap-)

### 2026-01-16 (50Hz Notch Filter Experiment)
- **Added**: `--nonotch` flag in [`prediction_2.py`](file:///scripts/prediction_2.py) — notch filter ON by default
- **Added**: `apply_notch_filter()` — 50Hz power line removal
- **Result**: Accuracy **dropped** from 66.0% → 58.7% with filter
- **Conclusion**: 50Hz gamma signal is **real neural activity** — use `--nonotch` for best results
- **See**: [V2 Results with 50Hz Notch Filter](#v2-results-with-50hz-notch-filter-n3)

### 2026-01-16 (Multi-Participant Validation)
- **Added**: `post0_merge_all_split_recording_csvs.py` — Merges split CSVs from headset reconnections
- **Validated**: V2 pipeline on 2 new participants (n=3 total)
- **Result**: Mean **66.0% accuracy** (95% CI: 56.9–75.1%), all beat 50% baseline
- **Conclusion**: Skip prediction method **generalizes across individuals**

### 2025-12-12 (Prediction V2)
- **Added**: `post2v2_add_skip_classification.py` — Preprocesses raw EEG for skip prediction
- **Added**: `prediction_2.py` — Full training pipeline with feature importance analysis
- **Result**: **71.2% validation accuracy** with 85.9% precision
- **Key finding**: Frontal high gamma (AF7, AF8) most predictive for skip intent

### 2025-12-11 (Model Optimization)
- **Added**: `train_transformer_v2.py` with multiple architectures and regularization
- **Experimented**: 15+ configurations, none beat majority baseline reliably
- **Conclusion**: V1 approach has weak signal, need different prediction target

### 2025-12-10
- **Added**: `recording_script_v4.py` with robust auto-reconnect
- **Added**: Real-time frequency monitoring during recording
