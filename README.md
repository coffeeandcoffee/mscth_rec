# EEG TikTok Study

EEG recording and ML pipeline for predicting TikTok engagement from brain signals (MSc Thesis).

## Project Structure

```
├── scripts/                          # Processing scripts
│   ├── recording_script_v4.py          # EEG + video recording (with auto-reconnect)
│   ├── post1_check_eeg_quality.py      # Quality check
│   ├── post2_classify_segments_and_cut.py  # Segment classification
│   ├── post3_preprocess_eeg.py         # Signal preprocessing
│   ├── post3v2_prep_for_ml.py          # ML preprocessing
│   ├── post4_preprocess_eeg_v2.py      # Feature extraction
│   ├── train_transformer.py            # Transformer training (V1)
│   └── train_transformer_v2.py         # Model optimization experiments (V2)
├── recordings/                       # EEG data (videos gitignored, data synced)
│   └── eeg_*/model_output/             # Trained models & visualizations
├── mscth/                            # Python virtual environment
└── old_delete/                       # Deprecated files
```

## Setup

```bash
source mscth/bin/activate
pip install torch pandas scipy scikit-learn matplotlib
```

---

## Complete Pipeline

### 1. Recording

```bash
python scripts/recording_script_v4.py --nocamera --duration 1800
```

**Keypress markers:**
- `A` — TikTok video transition (swipe)
- `B` — Baseline period marker

### 2. Post-processing

```bash
# Quality check
python scripts/post1_check_eeg_quality.py

# Classify segments (baseline vs TikTok, engaged vs skipped)
python scripts/post2_classify_segments_and_cut.py
```

### 3. ML Preprocessing

```bash
python scripts/post3v2_prep_for_ml.py --duration 0.5 --verbose
```

**What it does:**
1. Baseline normalization (z-score using baseline_1)
2. Frequency band extraction (7 bands × 4 channels = 28 features)
3. Timestamp-based segment extraction (first 0.5s after each video start)
4. Interpolation to uniform 256 Hz
5. Output: `*_ml_ready.npz` with shape `(n_videos, 128, 28)`

**Frequency bands:**
| Band | Range | Neural Correlate |
|------|-------|------------------|
| Delta | 1-4 Hz | Deep attention |
| Theta | 4-8 Hz | Memory, attention |
| Alpha | 8-13 Hz | Relaxation |
| Beta | 13-30 Hz | Active thinking |
| Low Gamma | 30-40 Hz | Cognitive processing |
| High Gamma | 40-60 Hz | Learning |
| Very High | 60-100 Hz | Exploratory |

### 4. Transformer Training

```bash
python scripts/train_transformer.py --balanced --epochs 100 --lr 0.003
```

**Architecture (per ML specialist):**
- 28-dimensional input (4 channels × 7 bands)
- 3 attention heads, 1 transformer layer
- Fully connected → sigmoid output
- Binary classification: engaged (>4s) vs skipped (<4s)

**Features:**
- Mac M1 MPS GPU acceleration
- Class balancing via weighted sampling
- 60/40 train/validation split
- Feature importance analysis
- Comprehensive visualizations

---

## Results & Model Optimization Experiments

### Initial Results (V1 Transformer)

| Metric | Value |
|--------|-------|
| Dataset | 321 videos (128 engaged, 193 skipped) |
| Best Val Accuracy | **62.8%** (but 79% train = overfitting) |
| Majority Baseline | 60.1% (just predicting "skipped") |

**Problem identified:** Model overfits severely with limited data. Val accuracy barely beats majority class baseline.

---

### Optimization Experiments (V2)

Created `train_transformer_v2.py` with multiple regularization and architecture options.

#### Techniques Tried

| Technique | What It Does | Result |
|-----------|--------------|--------|
| **Data Augmentation** | Noise, time shift, channel dropout, time masking | ❌ No improvement, sometimes worse |
| **Label Smoothing** | Soft labels (0.05-0.1) to prevent overconfidence | ❌ Model collapsed to majority class |
| **Higher Dropout** | 0.15-0.3 dropout throughout | ❌ Underfitting, stopped learning |
| **Weight Decay** | L2 regularization (0.02-0.05) | ❌ No improvement |
| **Early Stopping** | Stop when val accuracy stops improving | ⚠️ Stopped at ~60% (baseline) |
| **Focal Loss** | Focus on hard examples for class imbalance | ❌ Still overfit or collapsed |
| **Mixup** | Blend samples and labels | ❌ Worse performance |
| **K-Fold CV** | 5-fold cross-validation | ⚠️ Confirmed ~60% across all folds |

#### Architectures Tried

| Model | Parameters | Val Accuracy | Notes |
|-------|------------|--------------|-------|
| Transformer | 48K | ~60-62% | Overfits to 80%+ train |
| CNN | 23K | ~60% | Overfit to 100% train quickly |
| LSTM | ~5K | ~57% | Couldn't converge well |
| Hybrid CNN-LSTM | 19K | ~57% | Same issues |

#### Sklearn Baselines (Sanity Check)

```
Model                5-Fold CV Accuracy
LogisticRegression   0.495 ± 0.042
RandomForest         0.554 ± 0.017
GradientBoosting     0.561 ± 0.051  
SVM                  0.492 ± 0.029
```

**Critical finding:** Even simple sklearn models with hand-crafted features perform at ~50-56%, barely above random chance.

---

### Conclusions

1. **The signal is very weak (or not present)** — No model architecture or regularization technique could reliably beat the 60% majority class baseline
2. **Not an overfitting problem alone** — Even when preventing overfitting, models couldn't learn meaningful patterns
3. **321 samples is too few** — But even with proper regularization, the underlying signal appears too weak
4. **The 4-second threshold may not be optimal** — Binary engaged/skipped classification might not capture what EEG can actually predict

### What We Learned

- Temporal lobe theta/delta (TP9) showed highest feature importance in V1, but feature importance ≠ predictive power
- First 0.5s of video viewing may not contain enough discriminative signal
- The task (predicting if user will watch >4s) may not have a strong neural correlate detectable with consumer EEG

---

## Recommended Next Steps

### 1. Record More Data (Preferred)
- Target: 1000+ video segments minimum
- Consider longer recording sessions or multiple sessions
- More data allows models to find subtle patterns

### 2. Try Different Prediction Targets

Instead of predicting engagement from the **first 0.5s of viewing**, try:

| Alternative Target | Description | Why It Might Work |
|--------------------|-------------|-------------------|
| **Pre-skip brain state** | Use 2-4s EEG *before* the skip happens | Captures "thinking about skipping" neural state |
| **Skip vs No-skip intent** | Binary: user is about to skip within 2s | More direct neural correlate |
| **Continuous engagement** | Regression: predict watch duration | Avoids arbitrary threshold |
| **Boredom detection** | Detect when attention drops (alpha increase) | Well-established EEG pattern |

**Recommended experiment:**
```python
# Instead of: first 0.5s after video start → engaged/skipped
# Try: 4s before skip event → "about to skip" vs "random baseline"
```

### 3. Preprocessing Experiments
- Try longer segment durations (1s, 2s, 4s)
- Different frequency band combinations
- Raw EEG without frequency decomposition
- Other engagement thresholds (2s, 6s, 10s)

### 4. Feature Engineering
- Add temporal derivatives (rate of change)
- Asymmetry features (left vs right hemisphere)
- Coherence between electrodes
- Time-frequency spectrograms

---

## Top Predictive Features (for reference)

| Rank | Feature | Importance | Interpretation |
|------|---------|------------|----------------|
| 1 | **TP9_theta** | 100% | Temporal lobe, 4-8 Hz |
| 2 | **TP9_delta** | 99.3% | Temporal lobe, 1-4 Hz |
| 3 | AF8_alpha | 75.6% | Right frontal, 8-13 Hz |
| 4 | AF8_theta | 67.2% | Right frontal, 4-8 Hz |
| 5 | TP9_alpha | 64.0% | Temporal lobe, 8-13 Hz |

*Note: High feature importance doesn't guarantee predictive power with current dataset size.*

### Output Files

Generated in `recordings/eeg_*/model_output/`:
- `eeg_transformer.pt` / `eeg_transformer_v2.pt` — Trained models
- `training_results.json` / `training_results_v2.json` — Metrics + config
- `training_curves.png` — Loss/accuracy over epochs
- `electrode_band_heatmap.png` — Electrode × band importance matrix
- `confusion_matrix.png` — Prediction breakdown

---

## Dependencies

- `torch` — Transformer model (MPS for M1 Mac)
- `pylsl`, `muselsl` — Muse S streaming
- `opencv-python` — Video recording
- `scipy`, `numpy`, `pandas` — Signal processing
- `scikit-learn`, `matplotlib` — ML & visualization

---

## Changelog

### 2025-12-11 (Model Optimization)
- **Added**: `train_transformer_v2.py` with:
  - Multiple architectures: Transformer, CNN, LSTM, Hybrid
  - Regularization: dropout, weight decay, label smoothing
  - Focal Loss for class imbalance
  - Data augmentation (noise, time shift, masking)
  - Early stopping on validation accuracy
  - K-fold cross-validation
  - Model selection via `--model cnn/lstm/hybrid/transformer`
- **Experimented**: 15+ configurations, none beat majority baseline reliably
- **Conclusion**: Signal too weak with current setup, need more data or different prediction target

### 2025-12-11 (Earlier)
- **Updated**: `.gitignore` now excludes only video files in `recordings/`
- **Added**: `post3v2_prep_for_ml.py` — ML preprocessing pipeline
- **Added**: `train_transformer.py` — Original transformer training
- **First results**: 62.8% validation accuracy (overfitting)

### 2025-12-10
- **Added**: `recording_script_v4.py` with robust auto-reconnect
- **Added**: Real-time frequency monitoring during recording
- **Fixed**: All post-processing scripts use absolute paths

