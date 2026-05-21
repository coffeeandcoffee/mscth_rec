#!/usr/bin/env python3
"""
step05_feature_engineering.py — Aggregate feature computation.

Computes 4 stats × 7 bands × 4 channels = 112 features (full),
56 frontal-only (AF7/AF8), 56 temporal-only (TP9/TP10), and EI scalar.
Runs on both nonotch and notch preprocessed data.

OUT: feature matrices per participant × 2 (nonotch/notch)
"""

import numpy as np
import pickle
from pathlib import Path

import config


def compute_window_features(window_data, feature_names, channels=None):
    """Compute mean/std/min/max per band feature for a single window.

    window_data: shape (n_samples, n_features) where n_features = len(feature_names)
    Returns: 1D feature vector
    """
    if channels is None:
        # Use all features
        col_mask = list(range(len(feature_names)))
    else:
        col_mask = [i for i, fn in enumerate(feature_names)
                    if any(fn.startswith(ch + '_') for ch in channels)]

    features = []
    for col_idx in col_mask:
        col = window_data[:, col_idx]
        features.extend([
            np.mean(col),
            np.std(col),
            np.min(col),
            np.max(col),
        ])

    return np.array(features, dtype=np.float32)


def compute_ei(window_data, feature_names):
    """Compute Engagement Index = beta / (alpha + theta) across all 4 channels."""
    alpha_powers = []
    beta_powers = []
    theta_powers = []

    for ch in config.EEG_CHANNELS:
        for i, fn in enumerate(feature_names):
            if fn == f"{ch}_alpha":
                alpha_powers.append(np.mean(window_data[:, i]))
            elif fn == f"{ch}_beta":
                beta_powers.append(np.mean(window_data[:, i]))
            elif fn == f"{ch}_theta":
                theta_powers.append(np.mean(window_data[:, i]))

    alpha = np.mean(alpha_powers) if alpha_powers else 1e-12
    theta = np.mean(theta_powers) if theta_powers else 1e-12
    beta = np.mean(beta_powers) if beta_powers else 0.0

    denom = alpha + theta
    if denom < 1e-12:
        return 0.0
    return beta / denom


def process_windows(windows, feature_names):
    """Compute all feature sets for a list of windows."""
    n = len(windows)
    if n == 0:
        return None

    agg_names_full = config.build_agg_feature_names(config.EEG_CHANNELS)
    agg_names_frontal = config.build_agg_feature_names(config.FRONTAL_CHANNELS)
    agg_names_temporal = config.build_agg_feature_names(config.TEMPORAL_CHANNELS)

    features_full = np.zeros((n, len(agg_names_full)), dtype=np.float32)
    features_frontal = np.zeros((n, len(agg_names_frontal)), dtype=np.float32)
    features_temporal = np.zeros((n, len(agg_names_temporal)), dtype=np.float32)
    ei_values = np.zeros(n, dtype=np.float32)
    labels = np.zeros(n, dtype=np.int32)
    window_ids = np.zeros(n, dtype=np.int32)

    for i, w in enumerate(windows):
        data = w['data']
        if data.shape[0] == 0:
            continue

        features_full[i] = compute_window_features(data, feature_names, None)
        features_frontal[i] = compute_window_features(data, feature_names, config.FRONTAL_CHANNELS)
        features_temporal[i] = compute_window_features(data, feature_names, config.TEMPORAL_CHANNELS)
        ei_values[i] = compute_ei(data, feature_names)
        labels[i] = w['label']
        window_ids[i] = w['window_id']

    return {
        'features_full': features_full,
        'features_frontal': features_frontal,
        'features_temporal': features_temporal,
        'ei_values': ei_values,
        'labels': labels,
        'window_ids': window_ids,
        'agg_names_full': agg_names_full,
        'agg_names_frontal': agg_names_frontal,
        'agg_names_temporal': agg_names_temporal,
    }


def run(run_dir, params):
    """Entry point called by run.py."""
    config.pprint_step(5, "FEATURE ENGINEERING")

    windows_dir = run_dir / "windows" / "primary"
    features_dir = run_dir / "features"
    features_notch_dir = run_dir / "features_notch"
    features_dir.mkdir(parents=True, exist_ok=True)
    features_notch_dir.mkdir(parents=True, exist_ok=True)

    for pid in config.INCLUDED_PARTICIPANTS:
        # ── Nonotch features (primary) ──
        with open(windows_dir / f"P{pid}.pkl", 'rb') as f:
            win_data = pickle.load(f)

        windows = win_data['windows']
        feature_names = win_data['feature_names']

        result = process_windows(windows, feature_names)
        if result is None:
            print(f"  P{pid}: ⚠ No windows, skipping.")
            continue

        with open(features_dir / f"P{pid}.pkl", 'wb') as f:
            pickle.dump({'pid': pid, **result}, f)

        # ── Notch features (ablation) ──
        # Re-extract windows from notch-processed data
        notch_processed_dir = run_dir / "processed" / "notch"
        notch_pkl = notch_processed_dir / f"P{pid}.pkl"

        if notch_pkl.exists():
            with open(notch_pkl, 'rb') as f:
                notch_data = pickle.load(f)

            # [D3] pkl stores list of DataFrames under 'dfs', not single 'df'
            import pandas as pd
            notch_dfs = notch_data['dfs']
            notch_df = pd.concat(notch_dfs, ignore_index=True)
            notch_fnames = notch_data['feature_names']

            # Re-compute features using the same window time ranges but notch data
            notch_windows = []
            for w in windows:
                ts = notch_df['lsl_timestamp'].values
                mask = (ts >= w['start_time']) & (ts < w['end_time'])
                indices = np.where(mask)[0]
                if len(indices) >= 5:
                    notch_win_data = notch_df.iloc[indices][notch_fnames].values.astype(np.float32)
                    notch_windows.append({
                        'window_id': w['window_id'],
                        'label': w['label'],
                        'data': notch_win_data,
                    })

            if notch_windows:
                notch_result = process_windows(notch_windows, notch_fnames)
                if notch_result:
                    with open(features_notch_dir / f"P{pid}.pkl", 'wb') as f:
                        pickle.dump({'pid': pid, **notch_result}, f)

        print(f"  P{pid}: {result['features_full'].shape[0]} windows → "
              f"{result['features_full'].shape[1]} features (full)")

    print(f"\n  ✓ Feature matrices saved to features/ and features_notch/")


if __name__ == "__main__":
    print("Use run.py to execute the pipeline.")
