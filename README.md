# EEG TikTok Engagement Study

MSc Thesis research project for recording and analyzing EEG brain signals during TikTok video consumption. The goal is to classify neural patterns associated with user engagement based on viewing duration.

## 📁 Project Structure

```
Data Recording and Quality Tests/
├── scripts/                      # Main processing scripts
│   ├── recording_script_v2.py      # Live EEG + video recording
│   ├── post1_check_eeg_quality.py  # Quality assessment
│   ├── post2_classify_segments_and_cut.py  # Segment classification
│   ├── post3_preprocess_eeg.py     # Signal preprocessing
│   ├── post4_preprocess_eeg_v2.py  # Feature extraction & statistics
│   ├── how_to_activate_venv.txt    # Virtual environment activation
│   └── 00_rec_steps_learnings.txt  # Research notes
│
├── recordings/                   # Raw and processed EEG data
│   ├── eeg_YYYYMMDD_HHMMSS.csv     # Raw EEG recordings
│   ├── video_YYYYMMDD_HHMMSS.mp4   # Synchronized video recordings
│   ├── *_classified.csv            # Segments labeled by behavior
│   ├── *_preprocessed.csv          # Cleaned EEG signals
│   ├── *_bands.csv                 # Frequency band features
│   └── *.png / *.jpg               # Visualizations (t-SNE, channel plots)
│
├── data/                         # Additional experimental data
│   └── p2_stfn.csv                 # Participant data
│
├── mscth/                        # Python virtual environment
│   └── bin/activate                # Activate with: source mscth/bin/activate
│
└── old_delete/                   # Deprecated scripts (archived)
```

---

## 🔬 Research Methodology

### Study Design
1. **Participant** wears Muse S EEG headband (4 channels: TP9, AF7, AF8, TP10)
2. **Baseline recording** captured at start and end of session (B keypress markers)
3. **TikTok viewing** with A keypress marking each video transition
4. **Classification** based on viewing duration:
   - `tiktok_over_4s_watched` — user engaged with video
   - `tiktok_under_4s_watched` — user skipped quickly

### Frequency Bands Analyzed
| Band | Frequency Range | Associated With |
|------|-----------------|-----------------|
| Delta | 1-4 Hz | Deep sleep, unconscious |
| Theta | 4-8 Hz | Drowsiness, memory |
| Alpha | 8-13 Hz | Relaxation, calm focus |
| Beta | 13-30 Hz | Active thinking, alertness |
| Gamma | 30-40 Hz | Higher cognitive processing |

---

## 🚀 Pipeline Overview

### 1. Recording (`recording_script_v2.py`)
Simultaneous capture of EEG and video with synchronized timestamps.

```bash
# Start a 30-minute recording session
python scripts/recording_script_v2.py --duration 1800 --output_dir ./recordings
```

**Key Features:**
- Muse S connection via LSL (Lab Streaming Layer)
- Keypress logging (A = video transition, B = baseline markers)
- Real-time video preview with timestamp overlay
- Outputs: `eeg_*.csv` + `video_*.mp4`

**Hardware Requirements:**
- Muse S headband (paired via Bluetooth)
- Webcam for face video
- muselsl running: `muselsl stream --name MuseS`

---

### 2. Quality Check (`post1_check_eeg_quality.py`)
Quick assessment of recording quality before further processing.

```bash
# Analyze most recent recording
python scripts/post1_check_eeg_quality.py

# Analyze specific file with graphs
python scripts/post1_check_eeg_quality.py --file recordings/eeg_*.csv --graph
```

**Output:**
- Sampling rate verification (~256 Hz expected)
- Per-channel statistics (coverage, median, mean)
- Optional: Time-series plots per channel

---

### 3. Classification & Cutting (`post2_classify_segments_and_cut.py`)
Labels EEG segments based on keypress patterns and trims TikTok segments.

```bash
# Default: use latest file, keep first 0.5s after each A keypress
python scripts/post2_classify_segments_and_cut.py

# Custom cut duration (first 3 seconds)
python scripts/post2_classify_segments_and_cut.py --cut-duration 3.0
```

**Classification Logic:**
| Label | Definition |
|-------|------------|
| `baseline_1` | Between 1st and 2nd B keypress |
| `baseline_2` | Between 2nd-to-last and last B keypress |
| `tiktok_over_4s_watched` | Between A keypresses >4s apart |
| `tiktok_under_4s_watched` | Between A keypresses ≤4s apart |

**Output:** `*_classified.csv` with added `class` column

---

### 4. Preprocessing (`post3_preprocess_eeg.py`)
Evidence-based signal cleaning with documented decisions.

```bash
# Basic preprocessing
python scripts/post3_preprocess_eeg.py

# With t-SNE visualization
python scripts/post3_preprocess_eeg.py --tsne
```

**Processing Steps:**
1. **DC offset removal** — Centers signal at zero
2. **Bandpass filter (1-40 Hz)** — Removes drift and high-frequency noise
3. **Baseline normalization** — Z-score using baseline periods

**Output:** `*_preprocessed.csv`

---

### 5. Feature Extraction (`post4_preprocess_eeg_v2.py`)
Computes frequency band powers using Welch's PSD method.

```bash
# Basic feature extraction
python scripts/post4_preprocess_eeg_v2.py

# With t-SNE and statistical comparison
python scripts/post4_preprocess_eeg_v2.py --tsne --stats
```

**Processing Steps:**
1. **Segment extraction** — One segment per TikTok video or baseline block
2. **Welch PSD** — Power spectral density estimation
3. **Band power integration** — Delta, Theta, Alpha, Beta, Gamma
4. **Baseline drift correction** — Linear interpolation between baselines
5. **Z-score normalization** — Using baseline statistics

**Statistical Analysis:**
- Mann-Whitney U tests (non-parametric)
- Bonferroni correction for multiple comparisons
- Effect sizes: Rank-biserial correlation, Cohen's d

**Outputs:**
- `*_bands.csv` — ML-ready feature matrix
- `*_stats.png` — Box plots with significance annotations
- `*_statistics.csv` — Detailed test results
- `*_summary.txt` — Human-readable interpretation

---

## ⚙️ Setup Instructions

### 1. Activate Virtual Environment
```bash
cd "Data Recording and Quality Tests"
source mscth/bin/activate
```

### 2. Install Dependencies
```bash
pip install numpy pandas scipy scikit-learn matplotlib seaborn
pip install pylsl muselsl opencv-python pynput
```

### 3. Connect Muse S Headband
```bash
# Scan for devices
muselsl list

# Start LSL stream (keep terminal open)
muselsl stream --name MuseS
```

---

## 📊 Output File Naming Convention

Files follow a timestamped pipeline progression:

```
eeg_20251126_161954.csv                           # Raw recording
    └── eeg_20251126_161954_classified.csv        # + class labels
            └── *_classified_preprocessed.csv     # + signal cleaning
                    └── *_preprocessed_bands.csv  # + frequency features
```

---

## 🧪 Typical Workflow

```bash
# 1. Run recording session
python scripts/recording_script_v2.py --duration 1800

# 2. Check quality (verify good signal)
python scripts/post1_check_eeg_quality.py --graph

# 3. Classify and cut segments
python scripts/post2_classify_segments_and_cut.py --cut-duration 0.5

# 4. Preprocess signals
python scripts/post3_preprocess_eeg.py

# 5. Extract features and analyze
python scripts/post4_preprocess_eeg_v2.py --tsne --stats
```

---

## 📝 Notes & Learnings

- **TP electrodes** (temporal-parietal) may need 3+ minutes to stabilize after putting on headband
- **Keypress timing**: Press 'A' exactly when swiping to next TikTok, press 'B' for baseline start/end
- **Minimum segment duration**: 100ms required for valid frequency analysis
- **Sampling rate**: Muse S provides ~256 Hz for EEG channels

---

## 📚 Dependencies

| Package | Purpose |
|---------|---------|
| `pylsl` | Lab Streaming Layer for Muse data |
| `muselsl` | Muse headband streaming |
| `opencv-python` | Video recording |
| `pynput` | Keypress detection |
| `scipy` | Signal processing (filters, Welch PSD) |
| `scikit-learn` | t-SNE, StandardScaler |
| `matplotlib` / `seaborn` | Visualizations |
| `pandas` | Data handling |
| `numpy` | Numerical operations |

---

## 👤 Author

MSc Thesis project — EEG Neuroscience
