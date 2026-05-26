#!/usr/bin/env python3
"""
config.py — Shared constants, participant mapping, and DSP utilities.

Every pipeline step imports from here. No step defines its own constants.
"""

import numpy as np
from scipy import signal
from pathlib import Path

# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data-ipf-hpi"
SURVEY_CSV = DATA_DIR / "pre_and_post_survey.csv"
PIPELINE_DIR = Path(__file__).resolve().parent.parent  # 05-12_data_analysis_and_results

# ──────────────────────────────────────────────
# EEG constants
# ──────────────────────────────────────────────
EEG_CHANNELS = ['TP9', 'AF7', 'AF8', 'TP10']
FRONTAL_CHANNELS = ['AF7', 'AF8']
TEMPORAL_CHANNELS = ['TP9', 'TP10']

FREQUENCY_BANDS = [
    ('delta',     1,   4),
    ('theta',     4,   8),
    ('alpha',     8,  13),
    ('beta',     13,  30),
    ('low_gamma', 30,  40),
    ('high_gamma',40,  60),
    ('very_high', 60, 100),
]

N_BANDS = len(FREQUENCY_BANDS)
N_CHANNELS = len(EEG_CHANNELS)
N_STATS = 10  # mean, std, min, max, peakfreq, macrofreq, rel_power, act, mob, comp
N_FEATURES_FULL = N_CHANNELS * N_BANDS * N_STATS       # 280
N_FEATURES_SUBSET = 2 * N_BANDS * N_STATS              # 140

# ──────────────────────────────────────────────
# Participants
# ──────────────────────────────────────────────
EXCLUDED_PARTICIPANTS = [16, 19, 29]
ALL_PARTICIPANTS = list(range(4, 32))
INCLUDED_PARTICIPANTS = [p for p in ALL_PARTICIPANTS if p not in EXCLUDED_PARTICIPANTS]

# Explicit mapping: PID → session folder relative to DATA_DIR
# Order within each day is chronological = participant order
SESSION_MAP = {
    # Day 1 — Dresden (P4–P8)
    4:  "day 1 - 24-02-26 - P4-8/eeg_20260224_140907",
    5:  "day 1 - 24-02-26 - P4-8/eeg_20260224_145244",
    6:  "day 1 - 24-02-26 - P4-8/eeg_20260224_153850",
    7:  "day 1 - 24-02-26 - P4-8/eeg_20260224_162033",
    8:  "day 1 - 24-02-26 - P4-8/eeg_20260224_171333",
    # Day 2 — Dresden (P9–P14)
    9:  "day 2 - 25-02-26 - P9-14/eeg_20260225_132601",
    10: "day 2 - 25-02-26 - P9-14/eeg_20260225_141033",
    11: "day 2 - 25-02-26 - P9-14/eeg_20260225_145130",
    12: "day 2 - 25-02-26 - P9-14/eeg_20260225_153605",
    13: "day 2 - 25-02-26 - P9-14/eeg_20260225_162137",
    14: "day 2 - 25-02-26 - P9-14/eeg_20260225_170526",
    # Day 3 — HPI Potsdam (P15–P19)
    15: "day 3 - HPI - 26-02-26 - P15-19/eeg_20260226_135615",
    16: "day 3 - HPI - 26-02-26 - P15-19/eeg_20260226_144146",
    17: "day 3 - HPI - 26-02-26 - P15-19/eeg_20260226_153334",
    18: "day 3 - HPI - 26-02-26 - P15-19/eeg_20260226_162208",
    19: "day 3 - HPI - 26-02-26 - P15-19/eeg_20260226_165932",
    # Day 4 — HPI Potsdam (P20–P26)
    20: "day 4 - HPI - 27-02-26 - P20-26/eeg_20260227_135530",
    21: "day 4 - HPI - 27-02-26 - P20-26/eeg_20260227_144620",
    22: "day 4 - HPI - 27-02-26 - P20-26/eeg_20260227_152804",
    23: "day 4 - HPI - 27-02-26 - P20-26/eeg_20260227_160630",
    24: "day 4 - HPI - 27-02-26 - P20-26/eeg_20260227_164106",
    25: "day 4 - HPI - 27-02-26 - P20-26/eeg_20260227_174609",
    26: "day 4 - HPI - 27-02-26 - P20-26/eeg_20260227_184143",
    # Day 5 — HPI Potsdam (P27–P31)
    27: "day 5 - HPI - 28-02-26 - P27-31/eeg_20260228_103417",
    28: "day 5 - HPI - 28-02-26 - P27-31/eeg_20260228_114813",
    29: "day 5 - HPI - 28-02-26 - P27-31/eeg_20260228_132226",
    30: "day 5 - HPI - 28-02-26 - P27-31/eeg_20260228_135931",
    31: "day 5 - HPI - 28-02-26 - P27-31/eeg_20260228_145017",
}

# Paid participants (HPI sessions, days 3-5)
PAID_PARTICIPANTS = [p for p in range(15, 32) if p not in EXCLUDED_PARTICIPANTS]
UNPAID_PARTICIPANTS = [p for p in range(4, 15) if p not in EXCLUDED_PARTICIPANTS]

# ──────────────────────────────────────────────
# Default parameters (written to parameters.json)
# ──────────────────────────────────────────────
DEFAULT_PARAMS = {
    "step01": {
        "target_fs": 256.0,
        "baseline_offset_s": 10.0,
        "baseline_duration_s": 90.0,
        "skip_window_s": 0.5,      # Default: 3.0 (±0.5s instead of ±3s)
        "butterworth_order": 4,
    },
    "step02": {
        "blink_thresh_uv": 800,
        "emg_thresh_uv": 800,
        "blink_channels": ["AF7", "AF8"],
        "emg_channels": ["TP9", "TP10"],
    },
    "step03": {
        "half_window_s": 0.5,      # Default: 3.0 (matches step01's ±0.5s)
        "window_s": 1.0,           # Default: 3.0 (1s window duration)
        "stride_s": 1.0,           # Default: 0.6 (1.0s stride means 0% overlap)
        "burst_thresh_s": 3.0,     # A-presses < 3s apart → burst-skip flag
    },
    "step04": {
        "seeds": [0, 1, 7, 42, 99],
        "n_folds": 5,
        "gap_s": 3.0,
        "test_overlap": 0.0,
        "train_overlap": 0.8,
    },
    "step05": {
        "stats": ["mean", "std", "min", "max", "peakfreq", "macrofreq", "rel_power", "hjorth_act", "hjorth_mob", "hjorth_comp"],
    },
    "step06": {
        "param_grid": {
            'n_estimators': [30, 60, 100], # before 26/05: [100, 200, 300],
            'max_depth': [1, 2, 3], # before 26/05: [2, 3, 5],
            'min_samples_leaf': [10, 20, 30], # before 26/05: [3, 5, 10],
        },
        "inner_cv_folds": 3,
        "seed": 0,
    },
    "step07": {
        "eval_protocols": ["temporal_blocked", "temporal_blocked_no_gap", "random_split"],
        "feature_sets": ["full", "frontal", "temporal"],
        "seeds": [0, 1, 7, 42, 99],
        "random_split_ratio": 0.6,
    },
    "step08": {
        "classifier": "logistic_regression",
    },
    "step14": {
        "balanced": True,
    },
    "experimental": {
        "remove_min_max": True,       # X1: Ablate zero-variance min/max features
        "use_hilbert_envelope": True, # X2: Use Hilbert amplitude envelope instead of squared instantaneous power
        "extract_erp_features": True, # X4: Extract time-domain ERP features (raw voltage, slope)
    },
}

# ──────────────────────────────────────────────
# DSP Utility Functions
# ──────────────────────────────────────────────

def apply_notch_filter(data, fs, notch_freq=50.0, quality_factor=30.0):
    """Apply notch filter to remove power line interference."""
    if len(data) < 20:
        return data
    try:
        b, a = signal.iirnotch(notch_freq, quality_factor, fs)
        return signal.filtfilt(b, a, data)
    except Exception as e:
        print(f"NOTCH FAILED: {e}")
        return data


def get_bandpass_filter(fs, low_freq, high_freq, order=4):
    """Return Butterworth bandpass filter coefficients."""
    nyquist = fs / 2
    low = max(low_freq / nyquist, 0.01)
    high = min(high_freq / nyquist, 0.99)
    if low >= high:
        return None, None
    return signal.butter(order, [low, high], btype='band')


def extract_band_amplitude(data, fs, low_freq, high_freq, order=4):
    """Bandpass filter returning amplitude time series."""
    b, a = get_bandpass_filter(fs, low_freq, high_freq, order)
    if b is None or len(data) < 20:
        return np.zeros_like(data)
    try:
        return signal.filtfilt(b, a, data)
    except Exception:
        return np.zeros_like(data)


def compute_fs(timestamps):
    """Compute effective sampling rate from timestamps via median delta-t."""
    diffs = np.diff(timestamps)
    diffs = diffs[diffs > 0]
    if len(diffs) == 0:
        return 256.0
    return 1.0 / np.median(diffs)


def load_participant_csvs(pid):
    """Load all CSV parts for a participant as a list of separate DataFrames.
    
    Returns:
        list of pd.DataFrame — one per CSV file, NOT merged, in chronological order.
    Raises:
        FileNotFoundError if no CSVs found.
    """
    import pandas as pd
    import re

    session_rel = SESSION_MAP[pid]
    session_dir = DATA_DIR / session_rel
    session_name = session_dir.name

    csvs = []
    base_csv = session_dir / f"{session_name}.csv"
    if base_csv.exists():
        csvs.append((1, base_csv))

    for f in sorted(session_dir.glob(f"{session_name}_*.csv")):
        match = re.search(r'_(\d+)\.csv$', f.name)
        if match:
            csvs.append((int(match.group(1)), f))

    csvs.sort(key=lambda x: x[0])
    if not csvs:
        raise FileNotFoundError(f"No CSV files found for P{pid} in {session_dir}")

    return [pd.read_csv(f) for _, f in csvs]


def build_feature_names(channels=None):
    """Build ordered list of band feature names for given channels."""
    if channels is None:
        channels = EEG_CHANNELS
    names = []
    for ch in channels:
        for band_name, _, _ in FREQUENCY_BANDS:
            names.append(f"{ch}_{band_name}")
    return names


def build_agg_feature_names(channels=None):
    """Build ordered list of aggregated feature names."""
    base = build_feature_names(channels)
    agg = []
    
    # Check experimental toggles
    p_exp = DEFAULT_PARAMS.get('experimental', {})
    remove_min_max = p_exp.get('remove_min_max', False)
    extract_erp_features = p_exp.get('extract_erp_features', False)
    
    stats = ["mean", "std", "min", "max", "peakfreq", "macrofreq", "rel_power", "hjorth_act", "hjorth_mob", "hjorth_comp"]
    if remove_min_max:
        stats = [s for s in stats if s not in ('min', 'max')]
        
    for fname in base:
        for stat in stats:
            agg.append(f"{fname}_{stat}")
            
    if extract_erp_features:
        target_chs = channels if channels is not None else EEG_CHANNELS
        for ch in target_chs:
            for stat in ['raw_mean', 'raw_std', 'raw_slope']:
                agg.append(f"{ch}_{stat}")
                
    return agg


def pprint_step(step_num, title):
    """Print a formatted step header."""
    print(f"\n{'='*70}")
    print(f"  STEP {step_num:02d} — {title}")
    print(f"{'='*70}")
