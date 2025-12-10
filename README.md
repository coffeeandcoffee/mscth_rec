# EEG TikTok Study

EEG recording and processing pipeline for MSc Thesis research.

## Project Structure

```
├── scripts/                      # Processing scripts
│   ├── recording_script_v3.py      # EEG + video recording (latest)
│   ├── recording_script_v2.py      # EEG + video recording
│   ├── post1_check_eeg_quality.py  # Quality check
│   ├── post2_classify_segments_and_cut.py  # Segment classification
│   ├── post3_preprocess_eeg.py     # Signal preprocessing
│   └── post4_preprocess_eeg_v2.py  # Feature extraction
├── recordings/                   # EEG/video data (gitignored)
├── data/                         # Additional data (gitignored)
├── mscth/                        # Python virtual environment
└── old_delete/                   # Deprecated files
```

## Setup

```bash
# Activate virtual environment
source mscth/bin/activate
```

## Scripts

### Recording (`recording_script_v3.py`)

Records EEG from Muse S headband via LSL, optionally with synchronized video.

```bash
# EEG + video
python scripts/recording_script_v3.py --duration 1800

# EEG only (no video, saves storage)
python scripts/recording_script_v3.py --nocamera --duration 1800
```

**Keypress markers:**
- `A` — marks TikTok video transition
- `B` — marks baseline periods

---

### Post-processing Pipeline

Run in order after recording:

```bash
# 1. Check recording quality
python scripts/post1_check_eeg_quality.py

# 2. Classify segments and cut
python scripts/post2_classify_segments_and_cut.py

# 3. Preprocess EEG signals
python scripts/post3_preprocess_eeg.py

# 4. Extract frequency band features
python scripts/post4_preprocess_eeg_v2.py --tsne --stats
```

**Classification labels:**
- `baseline_1` / `baseline_2` — between B keypresses
- `tiktok_over_4s_watched` — A keypresses >4s apart
- `tiktok_under_4s_watched` — A keypresses ≤4s apart

## Dependencies

- `pylsl`, `muselsl` — Muse S streaming
- `opencv-python` — Video recording
- `pynput` — Keypress detection
- `scipy`, `numpy`, `pandas` — Signal processing
- `scikit-learn`, `matplotlib`, `seaborn` — Analysis & visualization

## Changelog

### 2025-12-10
- **Fixed**: Recording output directory now correctly saves to `recordings/` folder regardless of where the script is run from (previously created `scripts/recordings/` when run from scripts folder)
