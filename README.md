# EEG TikTok Study

EEG recording and ML pipeline for predicting TikTok engagement from brain signals (MSc Thesis).

## Project Structure

```
├── scripts/                          # Processing scripts
│   ├── recording_script_v4.py          # EEG + video recording (with auto-reconnect)
│   ├── post1_check_eeg_quality.py      # Quality check
│   ├── post2_classify_segments_and_cut.py  # Segment classification
│   ├── post3_preprocess_eeg.py         # Signal preprocessing
│   ├── post3v2_prep_for_ml.py          # ML preprocessing (NEW)
│   ├── post4_preprocess_eeg_v2.py      # Feature extraction
│   └── train_transformer.py            # Transformer training (NEW)
├── recordings/                       # EEG data (videos gitignored, .csv/.png/.pt/.npz synced -- large file warning of 1H csv recording. for now ok.)
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

## Results (First Training Run)

| Metric | Value |
|--------|-------|
| Dataset | 321 videos (128 engaged, 193 skipped) |
| Best Val Accuracy | **62.8%** |
| Train Accuracy | ~79% |

### Top Predictive Features

| Rank | Feature | Importance | Interpretation |
|------|---------|------------|----------------|
| 1 | **TP9_theta** | 100% | Temporal lobe, 4-8 Hz |
| 2 | **TP9_delta** | 99.3% | Temporal lobe, 1-4 Hz |
| 3 | AF8_alpha | 75.6% | Right frontal, 8-13 Hz |
| 4 | AF8_theta | 67.2% | Right frontal, 4-8 Hz |
| 5 | TP9_alpha | 64.0% | Temporal lobe, 8-13 Hz |

**Key finding:** Temporal lobe theta/delta activity (TP9) is most predictive of engagement!

### Output Files

Generated in `recordings/eeg_*/model_output/`:
- `eeg_transformer.pt` — Trained model
- `training_results.json` — Metrics + feature importance
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

### 2025-12-11
- **Updated**: `.gitignore` now excludes only video files in `recordings/`; data files (.csv, .png, .pt, .npz, .txt, .json) are synced to git

- **Added**: `post3v2_prep_for_ml.py` — ML preprocessing pipeline:
  - Baseline normalization using baseline_1
  - 7 frequency band extraction (delta through 100Hz)
  - Timestamp-based segment slicing (not sample-based)
  - 256 Hz interpolation
  - Output: numpy arrays ready for transformer
- **Added**: `train_transformer.py` — Transformer training script:
  - 3 heads, 1 layer, d_model=66
  - Mac M1 MPS GPU support
  - Class balancing via weighted sampling
  - Feature importance via gradient attribution
  - Visualizations: training curves, boxplots, heatmaps, confusion matrix
- **First results**: 62.8% validation accuracy, TP9_theta most predictive

### 2025-12-10
- **Added**: `recording_script_v4.py` with robust auto-reconnect
- **Added**: Real-time frequency monitoring during recording
- **Fixed**: All post-processing scripts use absolute paths

---

## Possible Next Steps if nothing else stated by user:

1. **More data**: Collect additional recordings to improve model generalization (currently overfitting with 321 samples)
2. **Hyperparameter tuning**: Experiment with different segment durations (0.25s, 1s, 2s)
3. **Cross-validation**: Implement k-fold CV for more robust evaluation
4. **Model variants**: Try different architectures (CNN, LSTM, larger transformers)
5. **Real-time prediction**: Deploy model for live engagement prediction
