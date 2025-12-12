# EEG TikTok Study

EEG recording and ML pipeline for predicting TikTok engagement from brain signals (MSc Thesis).

## Project Structure

```
├── scripts/
│   ├── recording_script_v4.py          # EEG + video recording
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
| **Best accuracy** | 62.8% (barely beats 60% baseline) | **69.65%** ✅ |
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
python scripts/prediction_2.py --window 3.0 --epochs 50
```

**Pipeline:**
1. Extract frequency bands (7 bands × 4 channels = 28 features)
2. Create 3-second sample blocks:
   - `about_to_skip`: 3s ending at each keypress_A (all 321 blocks)
   - `not_about_to_skip`: Random windows with ~2s stride (<33% overlap)
3. Interpolate each block to 256Hz (768 samples)
4. Balance dataset to 50/50
5. Train transformer model

**Output:** `model_output_prediction_v2/`
- `skip_prediction_model.pt` — Trained model
- `training_results_v2.json` — Metrics
- `training_curves_v2.png` — Loss/accuracy plots
- `confusion_matrix_v2.png` — Predictions breakdown

---

## V2 Results

| Metric | Value |
|--------|-------|
| **Validation Accuracy** | **69.65%** |
| Precision | 81.7% |
| Recall | 45.3% |
| F1 Score | 58.3% |
| Training Samples | 385 |
| Validation Samples | 257 |
| Data Loss | 0.0% |

### Interpretation

- **High precision (82%)**: When model predicts "about to skip", it's correct 82% of the time
- **Lower recall (45%)**: Model is conservative, catches ~half of actual skip events
- **Beats baseline**: 70% accuracy vs 50% random chance

### Why V2 Works Better

1. **Different target**: Predicts "about to skip" intent vs engagement after the fact
2. **More relevant data**: Uses 3s before skip (when brain is deciding) vs 0.5s after video start
3. **No data loss**: Uses raw EEG without cutting, preserving continuity

---

## Pipeline V1: Engagement Prediction (Archived)

> ⚠️ This approach had weak signal (~60% accuracy). See V2 instead.

```bash
# V1 Pipeline (for reference)
python scripts/post2_classify_segments_and_cut.py
python scripts/post3v2_prep_for_ml.py --duration 0.5
python scripts/train_transformer.py --balanced --epochs 100
```

**V1 Conclusions:**
- 62.8% best accuracy (only 2.7% above majority baseline)
- Multiple architectures and regularization techniques tried
- Signal appears too weak for this prediction target

---

## Recording

```bash
python scripts/recording_script_v4.py --nocamera --duration 1800
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
| High Gamma | 40-60 Hz | Learning |
| Very High | 60-100 Hz | Exploratory |

---

## Dependencies

```bash
torch pandas scipy scikit-learn matplotlib pylsl muselsl opencv-python numpy
```

---

## Changelog

### 2025-12-12 (Prediction V2)
- **Added**: `post2v2_add_skip_classification.py` — Preprocesses raw EEG for skip prediction
- **Added**: `prediction_2.py` — Full skip prediction training pipeline
- **Result**: **69.65% validation accuracy** (significant improvement over V1)
- **Key insight**: Predicting "about to skip" from 3s before skip event works better than predicting engagement from first 0.5s of viewing

### 2025-12-11 (Model Optimization)
- **Added**: `train_transformer_v2.py` with multiple architectures and regularization
- **Experimented**: 15+ configurations, none beat majority baseline reliably
- **Conclusion**: V1 approach has weak signal, need different prediction target

### 2025-12-10
- **Added**: `recording_script_v4.py` with robust auto-reconnect
- **Added**: Real-time frequency monitoring during recording
