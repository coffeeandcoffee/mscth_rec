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
from scipy.signal import find_peaks

import config


def compute_window_features(window_data, feature_names, channels=None, remove_min_max=False, extract_erp_features=False):
    """Compute mean/std/min/max per band feature for a single window.

    window_data: shape (n_samples, n_features) where n_features = len(feature_names)
    Returns: 1D feature vector
    """
    band_names = [fn for fn in feature_names if '_' in fn]
    raw_names = [fn for fn in feature_names if '_' not in fn]

    if channels is None:
        # Use all features
        col_mask = [i for i, fn in enumerate(feature_names) if fn in band_names]
        target_raw = raw_names
    else:
        col_mask = [i for i, fn in enumerate(feature_names)
                    if fn in band_names and any(fn.startswith(ch + '_') for ch in channels)]
        target_raw = [ch for ch in raw_names if ch in channels]

    # Pre-calculate total power per channel for Relative Power
    total_powers = {}
    for ch in config.EEG_CHANNELS:
        ch_indices = [i for i, fn in enumerate(feature_names) if fn.startswith(f"{ch}_")]
        if ch_indices:
            total_powers[ch] = sum(np.mean(window_data[:, i]) for i in ch_indices)

    features = []
    for col_idx in col_mask:
        col = window_data[:, col_idx]
        col_name = feature_names[col_idx]
        ch_name = col_name.split('_')[0]
        
        # Calculate Peak Frequency (Raw, Jagged)
        peaks, _ = find_peaks(col)
        duration_s = len(col) / config.DEFAULT_PARAMS['step01']['target_fs']
        peakfreq = len(peaks) / duration_s if duration_s > 0 else 0.0

        # Calculate Macro Frequency (User's custom contiguous-block algorithm with dynamic gap bridging)
        if len(peaks) > 0:
            mean_peak_amp = np.mean(col[peaks])
            is_above = col >= mean_peak_amp
            
            # Calculate dynamic gap threshold
            false_lengths = []
            curr_len = 0
            for val in is_above:
                if not val:
                    curr_len += 1
                elif curr_len > 0:
                    false_lengths.append(curr_len)
                    curr_len = 0
            if curr_len > 0:
                false_lengths.append(curr_len)
            gap_threshold = np.mean(false_lengths) if len(false_lengths) > 0 else 0
            
            # Bridge short gaps (<= gap_threshold)
            is_above_smoothed = is_above.copy()
            last_true_idx = -1
            for i in range(len(is_above)):
                if is_above[i]:
                    if last_true_idx != -1:
                        gap = i - last_true_idx - 1
                        if 0 < gap <= gap_threshold:
                            is_above_smoothed[last_true_idx+1:i] = True
                    last_true_idx = i
                    
            starts_high = 1 if is_above_smoothed[0] else 0
            macro_count = starts_high + np.sum((is_above_smoothed[:-1] == False) & (is_above_smoothed[1:] == True))
        else:
            macro_count = 0
            
        macrofreq = macro_count / duration_s if duration_s > 0 else 0.0

        # Relative Power
        mean_power = np.mean(col)
        rel_power = mean_power / total_powers[ch_name] if total_powers.get(ch_name, 0) > 0 else 0.0

        # Hjorth Parameters
        var_zero = np.var(col)
        diff1 = np.diff(col)
        var_d1 = np.var(diff1)
        diff2 = np.diff(diff1)
        var_d2 = np.var(diff2)

        hjorth_act = float(var_zero)
        hjorth_mob = float(np.sqrt(var_d1 / var_zero)) if var_zero > 0 else 0.0
        mob_d1 = float(np.sqrt(var_d2 / var_d1)) if var_d1 > 0 else 0.0
        hjorth_comp = float(mob_d1 / hjorth_mob) if hjorth_mob > 0 else 0.0

        band_feats = [
            mean_power,
            np.std(col),
        ]
        
        if not remove_min_max:
            band_feats.extend([np.min(col), np.max(col)])
            
        band_feats.extend([
            float(peakfreq),
            float(macrofreq),
            rel_power,
            hjorth_act,
            hjorth_mob,
            hjorth_comp,
        ])
        features.extend(band_feats)

    # Extract ERP features on raw channels
    if extract_erp_features:
        for ch in target_raw:
            raw_idx = feature_names.index(ch)
            raw_col = window_data[:, raw_idx]
            
            r_mean = np.mean(raw_col)
            r_std = np.std(raw_col)
            
            # Simple slope via linear fit
            x = np.arange(len(raw_col))
            r_slope = np.polyfit(x, raw_col, 1)[0] * config.DEFAULT_PARAMS['step01']['target_fs']
            
            features.extend([float(r_mean), float(r_std), float(r_slope)])

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
        
    p_exp = config.DEFAULT_PARAMS.get('experimental', {})
    remove_min_max = p_exp.get('remove_min_max', False)
    extract_erp_features = p_exp.get('extract_erp_features', False)

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

        features_full[i] = compute_window_features(data, feature_names, None, remove_min_max, extract_erp_features)
        features_frontal[i] = compute_window_features(data, feature_names, config.FRONTAL_CHANNELS, remove_min_max, extract_erp_features)
        features_temporal[i] = compute_window_features(data, feature_names, config.TEMPORAL_CHANNELS, remove_min_max, extract_erp_features)
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
    features_notch_art_dir = run_dir / "features_notch_artifact"
    features_notch_burst_dir = run_dir / "features_notch_burst"
    features_notch_ab_dir = run_dir / "features_notch_artifact_burst"
    
    for d in [features_dir, features_notch_dir, features_notch_art_dir, features_notch_burst_dir, features_notch_ab_dir]:
        d.mkdir(parents=True, exist_ok=True)

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
                        'artifact_frac': w.get('artifact_frac', 0),
                        'is_burst_skip': w.get('is_burst_skip', False),
                    })

            if notch_windows:
                notch_result = process_windows(notch_windows, notch_fnames)
                if notch_result:
                    with open(features_notch_dir / f"P{pid}.pkl", 'wb') as f:
                        pickle.dump({'pid': pid, **notch_result}, f)
                
                # Filter 1: Artifact Rejection (fraction <= 0.5)
                art_windows = [w for w in notch_windows if w.get('artifact_frac', 0) <= 0.5]
                if art_windows:
                    art_result = process_windows(art_windows, notch_fnames)
                    if art_result:
                        with open(features_notch_art_dir / f"P{pid}.pkl", 'wb') as f:
                            pickle.dump({'pid': pid, **art_result}, f)

                # Filter 2: Burst Rejection (no burst skips)
                burst_windows = [w for w in notch_windows if not w.get('is_burst_skip', False)]
                if burst_windows:
                    burst_result = process_windows(burst_windows, notch_fnames)
                    if burst_result:
                        with open(features_notch_burst_dir / f"P{pid}.pkl", 'wb') as f:
                            pickle.dump({'pid': pid, **burst_result}, f)

                # Filter 3: Artifact + Burst Rejection
                ab_windows = [w for w in notch_windows if w.get('artifact_frac', 0) <= 0.5 and not w.get('is_burst_skip', False)]
                if ab_windows:
                    ab_result = process_windows(ab_windows, notch_fnames)
                    if ab_result:
                        with open(features_notch_ab_dir / f"P{pid}.pkl", 'wb') as f:
                            pickle.dump({'pid': pid, **ab_result}, f)

        print(f"  P{pid}: {result['features_full'].shape[0]} windows (full) → "
              f"{len(notch_windows) if notch_windows else 0} notch, "
              f"{len(art_windows) if art_windows else 0} art, "
              f"{len(burst_windows) if burst_windows else 0} burst, "
              f"{len(ab_windows) if ab_windows else 0} art+burst")

    print(f"\n  ✓ Feature matrices saved for all 5 universes.")


if __name__ == "__main__":
    print("Use run.py to execute the pipeline.")
