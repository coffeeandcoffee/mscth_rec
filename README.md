# EEG TikTok Study

EEG recording and processing pipeline for MSc Thesis research.

## Project Structure

```
├── scripts/                      # Processing scripts
│   ├── recording_script_v4.py      # EEG + video recording (latest, with auto-reconnect)
│   ├── recording_script_v3.py      # EEG + video recording
│   ├── recording_script_v2.py      # EEG + video recording (legacy)
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

### Recording (`recording_script_v4.py`)

Records EEG from Muse S headband via LSL, optionally with synchronized video. **v4 adds auto-reconnect on connection loss.**

```bash
# EEG + video
python scripts/recording_script_v4.py --duration 1800

# EEG only (no video, saves storage)
python scripts/recording_script_v4.py --nocamera --duration 1800
```

**v4 Features:**
- Real-time frequency monitoring with ✓/✗ indicator
- Connection loss detection with "⚠️ NO DATA" warning
- Auto-reconnect: scans for any available Muse device and restarts stream
- Creates new CSV files on reconnect (`_2.csv`, `_3.csv`, etc.)
- All keypress markers preserved across reconnects

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
- **Added**: `recording_script_v4.py` with robust auto-reconnect:
  - Scans for any available Muse device on connection loss
  - Restarts muselsl stream automatically
  - Creates new CSV file (`_2.csv`, `_3.csv`) and continues recording
  - Shows "⚠️ NO DATA" warning and "reconnect.. - scanning for devices..." status
- **Added**: Real-time frequency monitoring during recording (green ✓ / red ✗ indicator)
- **Fixed**: Recording output directory now correctly saves to `recordings/` folder regardless of CWD
