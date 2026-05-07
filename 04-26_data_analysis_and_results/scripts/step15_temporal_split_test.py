#!/usr/bin/env python3
"""
Step 15: Temporal Block Split — Data Leakage Safety Test

Re-evaluates intra-subject RF accuracy using a TEMPORAL split instead of
a random split. Train on the first 60% of each participant's session
chronologically, test on the last 40%.

This eliminates the data leakage concern from overlapping STAY windows
landing in both train and test via random splitting.

If accuracy still exceeds chance → the thesis finding is real.
If accuracy drops to ~50%    → the original result was a leakage artifact.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import json
from datetime import datetime

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from scipy import stats

import utils

import warnings
warnings.filterwarnings('ignore')


def create_aggregated_samples_with_time(df, feature_names, window_seconds=3.0):
    """
    Same as step2 logic but also returns the center timestamp of each window
    so we can do a temporal split.
    """
    timestamps = df['lsl_timestamp'].values
    engagement_state = df['engagement_state'].values
    keypress_A = df['keypress_A'].values

    samples = []
    labels = []
    center_times = []

    # SKIP (Label 0): 3s ending at each keypress_A
    keypress_indices = np.where(keypress_A == 1)[0]
    for kp_idx in keypress_indices:
        kp_time = timestamps[kp_idx]
        window_start = kp_time - window_seconds
        mask = (timestamps >= window_start) & (timestamps < kp_time)
        block_indices = np.where(mask)[0]
        if len(block_indices) < 10:
            continue
        block_data = df.iloc[block_indices][feature_names].values
        agg_features = []
        for j in range(block_data.shape[1]):
            col = block_data[:, j]
            agg_features.extend([np.mean(col), np.std(col), np.min(col), np.max(col)])
        samples.append(agg_features)
        labels.append(0)
        center_times.append(kp_time - window_seconds / 2)

    # STAY (Label 1): Sliding windows with 80% overlap
    stay_mask = engagement_state == 1
    stay_indices = np.where(stay_mask)[0]
    if len(stay_indices) > 0:
        breaks = np.where(np.diff(stay_indices) > 1)[0] + 1
        regions = np.split(stay_indices, breaks)
        for region in regions:
            if len(region) < 10:
                continue
            region_timestamps = timestamps[region]
            if (region_timestamps[-1] - region_timestamps[0]) < window_seconds:
                continue
            effective_stride = window_seconds * 0.2
            t_start = region_timestamps[0]
            t_end = region_timestamps[-1] - window_seconds
            current_t = t_start
            while current_t <= t_end:
                window_end = current_t + window_seconds
                mask = (timestamps >= current_t) & (timestamps < window_end) & stay_mask
                block_indices = np.where(mask)[0]
                if len(block_indices) < 10:
                    current_t += effective_stride
                    continue
                block_data = df.iloc[block_indices][feature_names].values
                agg_features = []
                for j in range(block_data.shape[1]):
                    col = block_data[:, j]
                    agg_features.extend([np.mean(col), np.std(col), np.min(col), np.max(col)])
                samples.append(agg_features)
                labels.append(1)
                center_times.append(current_t + window_seconds / 2)
                current_t += effective_stride

    return np.array(samples), np.array(labels), np.array(center_times)


def rebalance_dataset(X, y, times, seed=42):
    """Rebalance AND keep time arrays aligned."""
    np.random.seed(seed)
    n_class_0 = np.sum(y == 0)
    n_class_1 = np.sum(y == 1)
    if n_class_0 == n_class_1:
        return X, y, times
    if n_class_0 > n_class_1:
        minority_n = n_class_1
        majority_indices = np.where(y == 0)[0]
        minority_indices = np.where(y == 1)[0]
    else:
        minority_n = n_class_0
        majority_indices = np.where(y == 1)[0]
        minority_indices = np.where(y == 0)[0]
    selected = np.random.choice(majority_indices, size=minority_n, replace=False)
    all_idx = np.concatenate([minority_indices, selected])
    np.random.shuffle(all_idx)
    return X[all_idx], y[all_idx], times[all_idx]


def main():
    project_root = Path(__file__).resolve().parent.parent.parent
    processed_dir = project_root / "04-26_data_analysis_and_results" / "processed_data"
    outputs_dir = project_root / "04-26_data_analysis_and_results" / "outputs"

    out_dir = outputs_dir / f"temporal_split_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("STEP 15: TEMPORAL BLOCK SPLIT — DATA LEAKAGE SAFETY TEST")
    print("=" * 70)
    print("If accuracy still > 50% with temporal split → finding is REAL")
    print("If accuracy drops to ~50% → original result was leakage artifact")
    print("=" * 70)

    input_files = sorted(list(processed_dir.glob("P*_labeled.csv")))
    if not input_files:
        print("ERROR: No labeled data found.")
        return 1

    results_temporal = []
    results_random = []  # Also re-run random split for direct comparison

    for f_path in input_files:
        pid = f_path.stem.split('_')[0]
        print(f"\nProcessing {pid}...")

        df_raw = pd.read_csv(f_path)
        baseline_powers, fs = utils.compute_100s_baseline_power(df_raw, notch_freq=None)
        df_scaled, feature_names = utils.build_normalized_features(df_raw, baseline_powers, notch_freq=None)

        X, y, times = create_aggregated_samples_with_time(df_scaled, feature_names)

        if len(np.unique(y)) < 2:
            print(f"   ⚠ Single class. Skipping.")
            continue

        X_bal, y_bal, times_bal = rebalance_dataset(X, y, times)
        n_total = len(y_bal)

        # ── METHOD A: TEMPORAL SPLIT (leakage-safe) ──
        # Sort by time, train on first 60%, test on last 40%
        time_order = np.argsort(times_bal)
        X_sorted = X_bal[time_order]
        y_sorted = y_bal[time_order]

        split_idx = int(0.6 * n_total)
        X_train_t = X_sorted[:split_idx]
        y_train_t = y_sorted[:split_idx]
        X_test_t = X_sorted[split_idx:]
        y_test_t = y_sorted[split_idx:]

        if len(np.unique(y_train_t)) < 2 or len(np.unique(y_test_t)) < 2:
            print(f"   ⚠ Temporal split produced single class. Skipping.")
            continue

        clf_t = RandomForestClassifier(n_estimators=200, max_depth=7,
                                        min_samples_leaf=5, random_state=42,
                                        n_jobs=-1)
        clf_t.fit(X_train_t, y_train_t)
        y_pred_t = clf_t.predict(X_test_t)

        acc_t = accuracy_score(y_test_t, y_pred_t)
        rec_t = recall_score(y_test_t, y_pred_t, pos_label=1, zero_division=0)

        # ── METHOD B: RANDOM SPLIT (original method, for comparison) ──
        from sklearn.model_selection import train_test_split
        X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
            X_bal, y_bal, test_size=0.4, random_state=42, stratify=y_bal)

        clf_r = RandomForestClassifier(n_estimators=200, max_depth=7,
                                        min_samples_leaf=5, random_state=42,
                                        n_jobs=-1)
        clf_r.fit(X_train_r, y_train_r)
        y_pred_r = clf_r.predict(X_test_r)

        acc_r = accuracy_score(y_test_r, y_pred_r)
        rec_r = recall_score(y_test_r, y_pred_r, pos_label=1, zero_division=0)

        print(f"   Temporal: acc={acc_t:.3f} rec={rec_t:.3f}  |  "
              f"Random: acc={acc_r:.3f} rec={rec_r:.3f}  |  "
              f"Δ={acc_t - acc_r:+.3f}")

        results_temporal.append({
            'participant': pid, 'accuracy': acc_t, 'recall': rec_t,
            'n_train': len(y_train_t), 'n_test': len(y_test_t)
        })
        results_random.append({
            'participant': pid, 'accuracy': acc_r, 'recall': rec_r,
        })

    if not results_temporal:
        print("No valid results.")
        return 1

    df_t = pd.DataFrame(results_temporal)
    df_r = pd.DataFrame(results_random)

    mean_acc_t = df_t['accuracy'].mean()
    mean_rec_t = df_t['recall'].mean()
    mean_acc_r = df_r['accuracy'].mean()
    mean_rec_r = df_r['recall'].mean()

    above_chance_t = (df_t['accuracy'] > 0.5).sum()
    above_chance_r = (df_r['accuracy'] > 0.5).sum()

    print(f"\n{'=' * 70}")
    print("VERDICT: TEMPORAL vs RANDOM SPLIT COMPARISON")
    print(f"{'=' * 70}")
    print(f"\n  TEMPORAL SPLIT (leakage-safe):")
    print(f"    Mean Accuracy: {mean_acc_t:.4f} ± {df_t['accuracy'].std():.4f}")
    print(f"    Mean Recall:   {mean_rec_t:.4f} ± {df_t['recall'].std():.4f}")
    print(f"    Above chance:  {above_chance_t}/{len(df_t)}")
    print(f"\n  RANDOM SPLIT (original method):")
    print(f"    Mean Accuracy: {mean_acc_r:.4f} ± {df_r['accuracy'].std():.4f}")
    print(f"    Mean Recall:   {mean_rec_r:.4f} ± {df_r['recall'].std():.4f}")
    print(f"    Above chance:  {above_chance_r}/{len(df_r)}")

    # Paired comparison
    deltas = df_t['accuracy'].values - df_r['accuracy'].values
    try:
        w_stat, w_p = stats.wilcoxon(deltas)
        print(f"\n  Wilcoxon paired test (temporal vs random accuracy):")
        print(f"    W={w_stat:.1f}, p={w_p:.4f}")
        print(f"    Mean Δ accuracy: {np.mean(deltas):+.4f}")
    except Exception as e:
        print(f"\n  Wilcoxon test failed: {e}")

    # One-sample Wilcoxon: is temporal accuracy > 0.5?
    try:
        w_stat2, w_p2 = stats.wilcoxon(df_t['accuracy'].values - 0.5)
        print(f"\n  One-sample Wilcoxon: temporal accuracy vs 0.50 chance:")
        print(f"    W={w_stat2:.1f}, p={w_p2:.6f}")
        if w_p2 < 0.05:
            print(f"    → SIGNIFICANT: Temporal split accuracy is above chance (p<0.05)")
            print(f"    → THE THESIS FINDING SURVIVES THE LEAKAGE TEST ✓")
        else:
            print(f"    → NOT significant: Cannot confirm accuracy > chance")
            print(f"    → THE ORIGINAL RESULT MAY HAVE BEEN A LEAKAGE ARTIFACT ✗")
    except Exception as e:
        print(f"  Test failed: {e}")

    # Save
    summary = {
        'temporal_split': {
            'mean_accuracy': mean_acc_t,
            'std_accuracy': float(df_t['accuracy'].std()),
            'mean_recall': mean_rec_t,
            'above_chance': int(above_chance_t),
            'total': len(df_t),
        },
        'random_split': {
            'mean_accuracy': mean_acc_r,
            'std_accuracy': float(df_r['accuracy'].std()),
            'mean_recall': mean_rec_r,
            'above_chance': int(above_chance_r),
            'total': len(df_r),
        },
        'per_participant': []
    }
    for i in range(len(df_t)):
        summary['per_participant'].append({
            'participant': df_t.iloc[i]['participant'],
            'temporal_accuracy': float(df_t.iloc[i]['accuracy']),
            'temporal_recall': float(df_t.iloc[i]['recall']),
            'random_accuracy': float(df_r.iloc[i]['accuracy']),
            'random_recall': float(df_r.iloc[i]['recall']),
            'delta_accuracy': float(deltas[i]),
        })

    with open(out_dir / 'temporal_split_results.json', 'w') as f:
        json.dump(summary, f, indent=2)

    df_t.to_csv(out_dir / 'temporal_split_per_participant.csv', index=False)
    print(f"\n{'=' * 70}")
    print(f"Outputs written to: {out_dir.name}")
    return 0


if __name__ == "__main__":
    main()
