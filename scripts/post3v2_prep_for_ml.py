#!/usr/bin/env python3
"""
EEG ML Preprocessing Pipeline - Post3v2

Transforms classified EEG data into ML-ready format with frequency band features.
Designed for transformer-based binary classification (over vs under 4s watched).

Pipeline:
1. Load classified CSV
2. Baseline normalization (using baseline_1)
3. Frequency band extraction (7 bands × 4 channels = 28 features)
4. Interpolation to uniform 256 Hz
5. Segment extraction (first X seconds after each video start)
6. Output as numpy arrays

Usage:
    python post3v2_prep_for_ml.py                    # Use latest classified.csv
    python post3v2_prep_for_ml.py --duration 0.5     # First 0.5s per video
    python post3v2_prep_for_ml.py --verbose          # Detailed output
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import signal
from scipy.interpolate import interp1d
import warnings
warnings.filterwarnings('ignore')


# Constants
EEG_CHANNELS = ['TP9', 'AF7', 'AF8', 'TP10']
TARGET_FS = 256  # Target sampling rate (Hz)

# Frequency bands: (name, low_freq, high_freq)
FREQUENCY_BANDS = [
    ('delta', 1, 4),
    ('theta', 4, 8),
    ('alpha', 8, 13),
    ('beta', 13, 30),
    ('low_gamma', 30, 40),
    ('high_gamma', 40, 60),
    ('very_high', 60, 100),
]


def find_latest_classified_file(recordings_dir=None):
    """Find the most recently created classified CSV file."""
    if recordings_dir is None:
        recordings_dir = Path(__file__).resolve().parent.parent / "recordings"
    recordings_path = Path(recordings_dir)
    
    if not recordings_path.exists():
        raise FileNotFoundError(f"Recordings directory '{recordings_dir}' not found")
    
    # Search for classified files (but not already preprocessed ones)
    classified_files = [f for f in recordings_path.rglob("*_classified.csv") 
                       if '_ml_ready' not in str(f) and '_preprocessed' not in str(f)]
    
    if not classified_files:
        raise FileNotFoundError(f"No classified CSV files found in '{recordings_dir}'")
    
    return max(classified_files, key=lambda f: f.stat().st_mtime)


def extract_band_power(data, fs, low_freq, high_freq):
    """
    Extract power in a specific frequency band using bandpass filter.
    
    Args:
        data: 1D array of EEG values
        fs: sampling frequency
        low_freq: lower bound of frequency band
        high_freq: upper bound of frequency band
    
    Returns:
        Filtered signal (bandpass)
    """
    nyquist = fs / 2
    
    # Ensure frequencies are within valid range
    low = max(low_freq / nyquist, 0.01)
    high = min(high_freq / nyquist, 0.99)
    
    if low >= high:
        return np.zeros_like(data)
    
    # Design bandpass filter
    try:
        b, a = signal.butter(4, [low, high], btype='band')
        filtered = signal.filtfilt(b, a, data)
        return filtered
    except Exception:
        return np.zeros_like(data)


def baseline_normalize(df, verbose=False):
    """
    Normalize EEG channels using baseline_1 statistics.
    
    Args:
        df: DataFrame with EEG data
        verbose: Print normalization stats
    
    Returns:
        DataFrame with normalized EEG channels
    """
    df_norm = df.copy()
    
    # Get baseline_1 data
    baseline_data = df[df['class'] == 'baseline_1']
    
    if len(baseline_data) == 0:
        print("WARNING: No baseline_1 data found, using global normalization")
        baseline_data = df
    
    if verbose:
        print(f"\nBaseline normalization using {len(baseline_data)} baseline_1 samples:")
    
    for ch in EEG_CHANNELS:
        mean = baseline_data[ch].mean()
        std = baseline_data[ch].std()
        
        if std == 0:
            std = 1  # Prevent division by zero
        
        df_norm[ch] = (df[ch] - mean) / std
        
        if verbose:
            print(f"  {ch}: μ={mean:.2f}, σ={std:.2f}")
    
    return df_norm


def interpolate_to_uniform(timestamps, data, target_fs=256):
    """
    Interpolate data to uniform sampling rate using timestamps.
    
    Args:
        timestamps: Original timestamps (non-uniform)
        data: Data array (n_samples, n_features)
        target_fs: Target sampling frequency
    
    Returns:
        uniform_timestamps, uniform_data
    """
    t_start = timestamps[0]
    t_end = timestamps[-1]
    duration = t_end - t_start
    
    # Create uniform timestamp grid
    n_samples = int(duration * target_fs) + 1
    uniform_t = np.linspace(t_start, t_end, n_samples)
    
    # Interpolate each feature
    n_features = data.shape[1] if len(data.shape) > 1 else 1
    
    if n_features == 1:
        f = interp1d(timestamps, data, kind='linear', fill_value='extrapolate')
        return uniform_t, f(uniform_t)
    
    uniform_data = np.zeros((len(uniform_t), n_features))
    for i in range(n_features):
        f = interp1d(timestamps, data[:, i], kind='linear', fill_value='extrapolate')
        uniform_data[:, i] = f(uniform_t)
    
    return uniform_t, uniform_data


def extract_frequency_features(df, verbose=False):
    """
    Extract frequency band features from EEG data.
    
    Args:
        df: DataFrame with normalized EEG data
        verbose: Print extraction progress
    
    Returns:
        DataFrame with band power features (original columns + band features)
    """
    # Estimate actual sampling rate from data
    time_diffs = np.diff(df['lsl_timestamp'].values)
    actual_fs = 1.0 / np.median(time_diffs)
    
    if verbose:
        print(f"\nEstimated sampling rate: {actual_fs:.1f} Hz")
        print(f"Extracting {len(FREQUENCY_BANDS)} frequency bands × {len(EEG_CHANNELS)} channels = {len(FREQUENCY_BANDS) * len(EEG_CHANNELS)} features")
    
    # Create new columns for band power
    band_features = {}
    
    for ch in EEG_CHANNELS:
        channel_data = df[ch].values
        
        for band_name, low_freq, high_freq in FREQUENCY_BANDS:
            feature_name = f"{ch}_{band_name}"
            band_features[feature_name] = extract_band_power(
                channel_data, actual_fs, low_freq, high_freq
            )
    
    # Create new DataFrame with band features
    df_bands = df.copy()
    for feature_name, feature_data in band_features.items():
        df_bands[feature_name] = feature_data
    
    return df_bands


def extract_video_segments(df, duration_seconds=0.5, verbose=False):
    """
    Extract uniform segments after each video start (keypress_A=1).
    Uses timestamps for slicing, not sample counts.
    
    Args:
        df: DataFrame with band features
        duration_seconds: Duration to extract after each video start
        verbose: Print extraction progress
    
    Returns:
        X: array of shape (n_videos, n_timepoints, n_features)
        y: array of binary labels
        metadata: dict with video info
    """
    # Get feature column names (all band features)
    feature_cols = [f"{ch}_{band[0]}" for ch in EEG_CHANNELS for band in FREQUENCY_BANDS]
    
    # Find video start indices (keypress_A == 1)
    video_starts = df[df['keypress_A'] == 1].index.tolist()
    
    if verbose:
        print(f"\nFound {len(video_starts)} video starts")
    
    segments = []
    labels = []
    video_info = []
    skipped = 0
    
    for i, start_idx in enumerate(video_starts):
        # Get start timestamp
        start_time = df.loc[start_idx, 'lsl_timestamp']
        end_time = start_time + duration_seconds
        
        # Get class label
        video_class = df.loc[start_idx, 'class']
        
        # Skip if not a TikTok class
        if video_class not in ['tiktok_over_4s_watched', 'tiktok_under_4s_watched']:
            skipped += 1
            continue
        
        # Binary label: over_4s = 1, under_4s = 0
        label = 1 if video_class == 'tiktok_over_4s_watched' else 0
        
        # Get data within time window
        mask = (df['lsl_timestamp'] >= start_time) & (df['lsl_timestamp'] < end_time)
        segment_df = df[mask]
        
        if len(segment_df) < 10:  # Too few samples
            skipped += 1
            continue
        
        # Extract timestamps and features
        timestamps = segment_df['lsl_timestamp'].values
        features = segment_df[feature_cols].values
        
        # Interpolate to uniform 256 Hz
        n_target_samples = int(duration_seconds * TARGET_FS)
        uniform_t = np.linspace(start_time, end_time, n_target_samples, endpoint=False)
        
        # Interpolate each feature
        uniform_features = np.zeros((n_target_samples, len(feature_cols)))
        for j, col in enumerate(feature_cols):
            f = interp1d(timestamps, features[:, j], kind='linear', 
                        bounds_error=False, fill_value='extrapolate')
            uniform_features[:, j] = f(uniform_t)
        
        segments.append(uniform_features)
        labels.append(label)
        video_info.append({
            'start_time': start_time,
            'original_class': video_class,
            'original_samples': len(segment_df)
        })
    
    if verbose:
        print(f"  Extracted: {len(segments)} valid segments")
        print(f"  Skipped: {skipped} (non-TikTok or insufficient data)")
        print(f"  Labels: {sum(labels)} engaged (over 4s), {len(labels) - sum(labels)} skipped (under 4s)")
    
    X = np.array(segments)  # (n_videos, n_timepoints, n_features)
    y = np.array(labels)    # (n_videos,)
    
    return X, y, video_info, feature_cols


def main():
    parser = argparse.ArgumentParser(description='EEG ML Preprocessing Pipeline')
    parser.add_argument('--file', '-f', type=str, help='Specific classified CSV file')
    parser.add_argument('--duration', '-d', type=float, default=0.5, 
                       help='Duration (seconds) after video start to extract (default: 0.5)')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    print("="*60)
    print("EEG ML PREPROCESSING PIPELINE")
    print("="*60)
    
    try:
        # Step 1: Load data
        if args.file:
            input_file = Path(args.file)
        else:
            input_file = find_latest_classified_file()
        
        print(f"\n1. LOADING DATA")
        print(f"   File: {input_file}")
        
        df = pd.read_csv(input_file)
        print(f"   Samples: {len(df)}")
        print(f"   Classes: {df['class'].value_counts().to_dict()}")
        
        # Step 2: Baseline normalization
        print(f"\n2. BASELINE NORMALIZATION")
        df_norm = baseline_normalize(df, verbose=args.verbose)
        print("   ✓ Normalized using baseline_1 statistics")
        
        # Step 3: Frequency band extraction
        print(f"\n3. FREQUENCY BAND EXTRACTION")
        df_bands = extract_frequency_features(df_norm, verbose=args.verbose)
        print(f"   ✓ Extracted 7 bands × 4 channels = 28 features")
        
        # Step 4 & 5: Segment extraction with interpolation
        print(f"\n4. SEGMENT EXTRACTION")
        print(f"   Duration: {args.duration}s after each video start")
        print(f"   Target samples: {int(args.duration * TARGET_FS)} @ {TARGET_FS}Hz")
        
        X, y, video_info, feature_names = extract_video_segments(
            df_bands, 
            duration_seconds=args.duration,
            verbose=args.verbose
        )
        
        print(f"\n5. OUTPUT SUMMARY")
        print(f"   X shape: {X.shape} (videos × timepoints × features)")
        print(f"   y shape: {y.shape}")
        print(f"   Features: {feature_names}")
        print(f"   Label balance: {np.mean(y)*100:.1f}% engaged (over 4s)")
        
        # Step 6: Save output
        output_file = input_file.parent / f"{input_file.stem}_ml_ready.npz"
        
        np.savez(output_file,
                 X=X,
                 y=y,
                 feature_names=feature_names,
                 channels=EEG_CHANNELS,
                 bands=[b[0] for b in FREQUENCY_BANDS],
                 duration_seconds=args.duration,
                 target_fs=TARGET_FS,
                 n_videos=len(y),
                 n_timepoints=X.shape[1] if len(X) > 0 else 0,
                 n_features=X.shape[2] if len(X) > 0 else 28)
        
        print(f"\n   ✓ Saved: {output_file}")
        
        # Also save as CSV for inspection
        csv_output = input_file.parent / f"{input_file.stem}_ml_ready_summary.csv"
        summary_df = pd.DataFrame({
            'video_idx': range(len(y)),
            'label': y,
            'label_name': ['over_4s' if l == 1 else 'under_4s' for l in y],
            'start_time': [v['start_time'] for v in video_info],
            'original_samples': [v['original_samples'] for v in video_info]
        })
        summary_df.to_csv(csv_output, index=False)
        print(f"   ✓ Summary: {csv_output}")
        
        print("\n" + "="*60)
        print("PREPROCESSING COMPLETE")
        print("="*60)
        print(f"\nReady for ML training with:")
        print(f"  - {len(y)} video samples")
        print(f"  - {X.shape[1]} timepoints per sample ({args.duration}s @ {TARGET_FS}Hz)")
        print(f"  - {X.shape[2]} features (4 channels × 7 bands)")
        
        return 0
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
