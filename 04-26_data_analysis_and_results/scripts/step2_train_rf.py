#!/usr/bin/env python3
"""
Step 2: Core ML Pipeline - Random Forest (Intra-Subject)

Evaluates the primary Baseline Normalized RF (200 estimators) natively per participant.
Designed rigidly against the STAY (Case B) narrative.
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import json
from datetime import datetime
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import utils

# Suppress sklearn warnings gracefully
import warnings
warnings.filterwarnings('ignore')

def create_aggregated_samples(df, feature_names, window_seconds=3.0):
    """
    Exactly matches legacy logic, but pulling from normalized arrays.
    Target metric correctly mapping to STAY=1 and SKIP=0.
    """
    timestamps = df['lsl_timestamp'].values
    engagement_state = df['engagement_state'].values
    keypress_A = df['keypress_A'].values
    
    samples = []
    labels = []
    
    # --- SKIP (Label 0): 3s ending at each keypress_A ---
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
        labels.append(0)  # SKIP
        
    # --- STAY (Label 1): Sliding windows with exactly 80% overlap ---
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
                
            effective_stride = window_seconds * 0.2  # 80% overlap -> stride 0.6s
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
                labels.append(1)  # STAY
                
                current_t += effective_stride
                
    X = np.array(samples)
    y = np.array(labels)
    
    # 112 Aggregated feature names mapping natively
    agg_feature_names = []
    for fname in feature_names:
        agg_feature_names.extend([f"{fname}_mean", f"{fname}_std", f"{fname}_min", f"{fname}_max"])
        
    return X, y, agg_feature_names

def rebalance_dataset(X, y, seed=42):
    """Randomly undersample majority class rigidly to achieve exactly 50/50 balance."""
    np.random.seed(seed)
    
    n_class_0 = np.sum(y == 0)
    n_class_1 = np.sum(y == 1)
    
    if n_class_0 == n_class_1:
        return X, y
        
    if n_class_0 > n_class_1:
        minority_n = n_class_1
        majority_indices = np.where(y == 0)[0]
        minority_indices = np.where(y == 1)[0]
    else:
        minority_n = n_class_0
        majority_indices = np.where(y == 1)[0]
        minority_indices = np.where(y == 0)[0]
        
    selected_majority = np.random.choice(majority_indices, size=minority_n, replace=False)
    all_indices = np.concatenate([minority_indices, selected_majority])
    np.random.shuffle(all_indices)
    
    return X[all_indices], y[all_indices]

def train_and_eval_rf(X_train, y_train, X_val, y_val, n_estimators=200, max_depth=7, min_samples_leaf=5):
    """Deploy exactly legacy mapped RF evaluating against Case B."""
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=42,
        class_weight='balanced',
        n_jobs=-1
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_val)
    
    # POS_LABEL IS EXPLICITLY "1" BECAUSE "STAY" IS 1.
    return clf, {
        'accuracy': accuracy_score(y_val, y_pred),
        'precision': precision_score(y_val, y_pred, zero_division=0, pos_label=1),
        'recall': recall_score(y_val, y_pred, zero_division=0, pos_label=1),
        'f1': f1_score(y_val, y_pred, zero_division=0, pos_label=1)
    }

def main():
    parser = argparse.ArgumentParser(description='Step 2: RF Training Pipeline (Intra-Subject)')
    parser.add_argument('--window', type=float, default=3.0, help='Window duration limit in seconds')
    parser.add_argument('--nonotch', action='store_true', help='Disable 50Hz notch filter')
    parser.add_argument('--nonormalize', action='store_true', help='Disable explicit baseline normalization computation')
    args = parser.parse_args()
    
    project_root = Path(__file__).resolve().parent.parent.parent
    processed_dir = project_root / "04-26_data_analysis_and_results" / "processed_data"
    
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    notch_freq = None if args.nonotch else 50.0
    normalize = not args.nonormalize
    
    suffix = ""
    if args.nonotch: suffix += "_nonotch"
    if args.nonormalize: suffix += "_nonorm"
    
    output_dir = project_root / "04-26_data_analysis_and_results" / "outputs" / f"rf_run{suffix}_{run_timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find active files strictly generated by Step 1
    input_files = list(processed_dir.glob("P*_labeled.csv"))
    if not input_files:
        print("ERROR: No labeled data found in processed_data dir.")
        return 1
        
    print("=" * 60)
    print("STEP 2: RF PIPELINE (STAY vs SKIP)")
    print(f"Notch Filter: {'DISABLED' if args.nonotch else '50.0Hz'}")
    print(f"Norm Baseline: {'DISABLED' if args.nonormalize else 'ENABLED'}")
    print("=" * 60)
    
    global_metrics = []
    
    for f_path in input_files:
        pid = f_path.stem.split('_')[0]
        print(f"\nProcessing {pid}...")
        
        df_raw = pd.read_csv(f_path)
        
        if normalize:
            baseline_powers, fs = utils.compute_100s_baseline_power(df_raw, notch_freq=notch_freq)
            if not baseline_powers:
                print(f"   ⚠ Failsafe: {pid} had defective baseline regions. Applying default amplitudes.")
        else:
            baseline_powers = None
            
        df_scaled, feature_names = utils.build_normalized_features(df_raw, baseline_powers, notch_freq=notch_freq)
        print(f"   ✓ Extracted {len(feature_names)} scaled features natively.")
        
        X, y, agg_names = create_aggregated_samples(df_scaled, feature_names, window_seconds=args.window)
        print(f"   ✓ Isolated {np.sum(y == 1)} STAY and {np.sum(y == 0)} SKIP blocks")
        
        if len(np.unique(y)) < 2:
            print("   ⚠ Single class detected. Skipping ML stage.")
            continue
            
        X_bal, y_bal = rebalance_dataset(X, y)
        print(f"   ✓ Rebalanced effectively to N={len(y_bal)} ({len(y_bal)//2} STAY / {len(y_bal)//2} SKIP)")
        
        X_train, X_val, y_train, y_val = train_test_split(X_bal, y_bal, test_size=0.4, random_state=42, stratify=y_bal)
        
        clf, results = train_and_eval_rf(X_train, y_train, X_val, y_val)
        
        print(f"   - Val Accuracy : {results['accuracy']:.3f}")
        print(f"   - Val Recall   : {results['recall']:.3f} (Sustained Engagement Sensitivity)")
        
        # Save exact model
        model_out = output_dir / f"{pid}_rf_model.pkl"
        joblib.dump(clf, model_out)
        
        results['participant'] = pid
        results['model_path'] = str(model_out.name)
        global_metrics.append(results)
        
    if global_metrics:
        mean_acc = np.mean([m['accuracy'] for m in global_metrics])
        mean_rec = np.mean([m['recall'] for m in global_metrics])
        
        print(f"\n{'='*60}")
        print("AGGREGATE PIPELINE PERFORMANCE")
        print(f"Total participants validated: {len(global_metrics)}")
        print(f"OVERALL MEAN ACCURACY: {mean_acc:.4f} ± {np.std([m['accuracy'] for m in global_metrics]):.4f}")
        print(f"OVERALL MEAN RECALL (Case B): {mean_rec:.4f} ± {np.std([m['recall'] for m in global_metrics]):.4f}")
        
        # Output summary json
        summary_dump = {
            'configuration': {
                'normalize': normalize,
                'notch': notch_freq
            },
            'aggregate': {
                'mean_accuracy': mean_acc,
                'mean_recall': mean_rec
            },
            'individuals': global_metrics
        }
        with open(output_dir / "rf_summary.json", 'w') as f:
            json.dump(summary_dump, f, indent=2)
            
    print(f"\nFinished. Outputs written to {output_dir.name}")
    return 0

if __name__ == "__main__":
    main()
