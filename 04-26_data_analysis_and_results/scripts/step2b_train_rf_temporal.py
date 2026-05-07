#!/usr/bin/env python3
"""
Step 2b: Leakage-Corrected RF — Blocked Temporal 5-Fold CV

Splits each participant's session into 5 equal temporal blocks with a
3-second gap between each block (= 1 window width, prevents any window
from spanning a block boundary).

Standard 5-fold: hold out 1 block as test, train on remaining 4.
  - Training blocks: 80% overlapping windows (data augmentation)
  - Test block: 0% overlapping windows (strict independence)
  - Each set rebalanced independently

This solves:
  1. Data leakage (gaps + no overlap between blocks)
  2. Non-stationarity confound (each fold tests a different time period)
  3. Tiny test sets (5 folds × test samples)
  4. Evaluation stability (mean ± SD across 5 folds)

Output format identical to step2 (rf_summary.json + P*_rf_model.pkl).
"""

import numpy as np
import pandas as pd
from pathlib import Path
import json
from datetime import datetime
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from scipy import stats as sp_stats

import utils
import warnings
warnings.filterwarnings('ignore')

K_FOLDS = 5
GAP_SECONDS = 3.0  # gap between blocks = 1 window, prevents overlap


def extract_windows(df, feature_names, window_seconds, time_start, time_end, overlap_frac):
    """Extract aggregated windows from a time range. overlap_frac: 0.8 or 0.0."""
    timestamps = df['lsl_timestamp'].values
    engagement_state = df['engagement_state'].values
    keypress_A = df['keypress_A'].values
    stride = window_seconds * (1.0 - overlap_frac)

    samples, labels = [], []

    # SKIP windows (3s ending at keypress)
    for kp_idx in np.where(keypress_A == 1)[0]:
        kp_time = timestamps[kp_idx]
        win_start = kp_time - window_seconds
        if win_start < time_start or kp_time > time_end:
            continue
        mask = (timestamps >= win_start) & (timestamps < kp_time)
        bidx = np.where(mask)[0]
        if len(bidx) < 10:
            continue
        block = df.iloc[bidx][feature_names].values
        agg = []
        for j in range(block.shape[1]):
            c = block[:, j]
            agg.extend([np.mean(c), np.std(c), np.min(c), np.max(c)])
        samples.append(agg)
        labels.append(0)

    # STAY windows (sliding)
    stay_mask = engagement_state == 1
    stay_idx = np.where(stay_mask)[0]
    if len(stay_idx) > 0:
        breaks = np.where(np.diff(stay_idx) > 1)[0] + 1
        for region in np.split(stay_idx, breaks):
            if len(region) < 10:
                continue
            rts = timestamps[region]
            rs = max(rts[0], time_start)
            re = min(rts[-1], time_end)
            if (re - rs) < window_seconds:
                continue
            t = rs
            while t + window_seconds <= re:
                mask = (timestamps >= t) & (timestamps < t + window_seconds) & stay_mask
                bidx = np.where(mask)[0]
                if len(bidx) >= 10:
                    block = df.iloc[bidx][feature_names].values
                    agg = []
                    for j in range(block.shape[1]):
                        c = block[:, j]
                        agg.extend([np.mean(c), np.std(c), np.min(c), np.max(c)])
                    samples.append(agg)
                    labels.append(1)
                t += stride

    return (np.array(samples) if samples else np.empty((0, len(feature_names)*4)),
            np.array(labels) if labels else np.empty(0, dtype=int))


def rebalance(X, y, seed=42):
    np.random.seed(seed)
    n0, n1 = np.sum(y == 0), np.sum(y == 1)
    if n0 == n1 or n0 == 0 or n1 == 0:
        return X, y
    if n0 > n1:
        maj, mi = np.where(y == 0)[0], np.where(y == 1)[0]
    else:
        maj, mi = np.where(y == 1)[0], np.where(y == 0)[0]
    sel = np.random.choice(maj, size=len(mi), replace=False)
    idx = np.concatenate([mi, sel])
    np.random.shuffle(idx)
    return X[idx], y[idx]


def main():
    project_root = Path(__file__).resolve().parent.parent.parent
    processed_dir = project_root / "04-26_data_analysis_and_results" / "processed_data"
    outputs_dir = project_root / "04-26_data_analysis_and_results" / "outputs"

    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = outputs_dir / f"rf_run_temporal_nonotch_{run_ts}"
    output_dir.mkdir(parents=True, exist_ok=True)

    input_files = sorted(list(processed_dir.glob("P*_labeled.csv")))

    print("=" * 70)
    print(f"STEP 2b: BLOCKED TEMPORAL {K_FOLDS}-FOLD CV (LEAKAGE-CORRECTED)")
    print(f"  Gap between blocks: {GAP_SECONDS}s | Train overlap: 80% | Test overlap: 0%")
    print("=" * 70)

    global_metrics = []

    for f_path in input_files:
        pid = f_path.stem.split('_')[0]
        print(f"\n{'─'*50}\n{pid}")

        df_raw = pd.read_csv(f_path)
        baseline_powers, fs = utils.compute_100s_baseline_power(df_raw, notch_freq=None)
        df_scaled, feature_names = utils.build_normalized_features(df_raw, baseline_powers, notch_freq=None)

        # Find viewing range
        viewing_mask = df_scaled['engagement_state'] != -1
        vt = df_scaled.loc[viewing_mask, 'lsl_timestamp'].values
        if len(vt) < 100:
            print("   ⚠ Insufficient data. Skipping.")
            continue

        session_start, session_end = vt[0], vt[-1]
        session_dur = session_end - session_start

        # Create K blocks with gaps
        block_dur = (session_dur - GAP_SECONDS * (K_FOLDS - 1)) / K_FOLDS
        if block_dur < 10:
            print("   ⚠ Session too short for 5 blocks. Skipping.")
            continue

        blocks = []
        for k in range(K_FOLDS):
            b_start = session_start + k * (block_dur + GAP_SECONDS)
            b_end = b_start + block_dur
            blocks.append((b_start, b_end))

        # Run k-fold
        fold_accs, fold_recs, fold_f1s = [], [], []
        best_clf = None
        best_acc = -1

        for fold in range(K_FOLDS):
            # Test: current block (non-overlapping)
            X_test, y_test = extract_windows(df_scaled, feature_names, 3.0,
                                              blocks[fold][0], blocks[fold][1], 0.0)
            # Train: all other blocks (overlapping)
            X_train_parts, y_train_parts = [], []
            for k in range(K_FOLDS):
                if k == fold:
                    continue
                Xk, yk = extract_windows(df_scaled, feature_names, 3.0,
                                          blocks[k][0], blocks[k][1], 0.8)
                if len(yk) > 0:
                    X_train_parts.append(Xk)
                    y_train_parts.append(yk)

            if not X_train_parts or len(y_test) < 4:
                continue

            X_train = np.concatenate(X_train_parts)
            y_train = np.concatenate(y_train_parts)

            if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
                continue

            X_train, y_train = rebalance(X_train, y_train, seed=42+fold)
            X_test, y_test = rebalance(X_test, y_test, seed=42+fold)

            clf = RandomForestClassifier(n_estimators=200, max_depth=7,
                                          min_samples_leaf=5, random_state=42, n_jobs=-1)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)

            acc = accuracy_score(y_test, y_pred)
            rec = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
            f1 = f1_score(y_test, y_pred, pos_label=1, zero_division=0)
            fold_accs.append(acc)
            fold_recs.append(rec)
            fold_f1s.append(f1)

            if acc > best_acc:
                best_acc = acc
                best_clf = clf

            print(f"   Fold {fold}: train={len(y_train)} test={len(y_test)} "
                  f"acc={acc:.3f} rec={rec:.3f}")

        if not fold_accs:
            print("   ⚠ No valid folds. Skipping.")
            continue

        mean_acc = np.mean(fold_accs)
        mean_rec = np.mean(fold_recs)
        mean_f1 = np.mean(fold_f1s)

        print(f"   → MEAN: acc={mean_acc:.3f}±{np.std(fold_accs):.3f}  "
              f"rec={mean_rec:.3f}±{np.std(fold_recs):.3f}")

        # Save best model
        if best_clf is not None:
            joblib.dump(best_clf, output_dir / f"{pid}_rf_model.pkl")

        global_metrics.append({
            'participant': pid,
            'accuracy': mean_acc,
            'recall': mean_rec,
            'f1': mean_f1,
            'std_accuracy': float(np.std(fold_accs)),
            'std_recall': float(np.std(fold_recs)),
            'n_folds': len(fold_accs),
            'fold_accuracies': fold_accs,
            'fold_recalls': fold_recs,
        })

    if not global_metrics:
        print("No results.")
        return 1

    accs = [m['accuracy'] for m in global_metrics]
    recs = [m['recall'] for m in global_metrics]
    above_chance = sum(1 for a in accs if a > 0.5)

    try:
        w_stat, w_p = sp_stats.wilcoxon(np.array(accs) - 0.5)
    except:
        w_stat, w_p = 0, 1.0

    print(f"\n{'='*70}")
    print(f"BLOCKED TEMPORAL {K_FOLDS}-FOLD CV — AGGREGATE")
    print(f"  N = {len(global_metrics)} participants")
    print(f"  Mean Accuracy: {np.mean(accs):.4f} ± {np.std(accs):.4f}")
    print(f"  Mean Recall:   {np.mean(recs):.4f} ± {np.std(recs):.4f}")
    print(f"  Above 50%:     {above_chance}/{len(global_metrics)}")
    print(f"  Wilcoxon vs chance: W={w_stat:.1f}, p={w_p:.6f}")
    if w_p < 0.05:
        print(f"  → SIGNIFICANT above chance ✓")
    else:
        print(f"  → NOT significant above chance ✗")
    print(f"{'='*70}")

    summary = {
        'configuration': {
            'method': f'blocked_temporal_{K_FOLDS}fold_cv',
            'gap_seconds': GAP_SECONDS,
            'train_overlap': 0.8,
            'test_overlap': 0.0,
            'n_folds': K_FOLDS,
            'notch': None,
        },
        'aggregate': {
            'mean_accuracy': float(np.mean(accs)),
            'std_accuracy': float(np.std(accs)),
            'mean_recall': float(np.mean(recs)),
            'above_chance': above_chance,
            'wilcoxon_p': float(w_p),
        },
        'individuals': global_metrics,
    }
    with open(output_dir / "rf_summary.json", 'w') as f:
        json.dump(summary, f, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else x)

    print(f"\nOutputs: {output_dir.name}")
    return 0


if __name__ == "__main__":
    main()
