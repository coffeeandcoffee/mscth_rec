#!/usr/bin/env python3
"""
Step 3: Leave-One-Group-Out Cross Validation (LOGO-CV) Random Forest

Tests cross-participant generalizability by iteratively training on n-1 
participants and testing on 1 held-out participant. 
Explicitly tracks Case B mapping (STAY=1) and Baseline Normalization natively.
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import json
from datetime import datetime

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import utils
import step2_train_rf

# Suppress sklearn warnings gracefully
import warnings
warnings.filterwarnings('ignore')

def main():
    parser = argparse.ArgumentParser(description='Step 3: LOGO-CV RF Pipeline')
    parser.add_argument('--window', type=float, default=3.0, help='Window duration limit in seconds')
    parser.add_argument('--nonotch', action='store_true', help='Disable 50Hz notch filter')
    parser.add_argument('--nonormalize', action='store_true', help='Disable explicit baseline normalization')
    args = parser.parse_args()
    
    project_root = Path(__file__).resolve().parent.parent.parent
    processed_dir = project_root / "04-26_data_analysis_and_results" / "processed_data"
    
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    notch_freq = None if args.nonotch else 50.0
    normalize = not args.nonormalize
    
    suffix = ""
    if args.nonotch: suffix += "_nonotch"
    if args.nonormalize: suffix += "_nonorm"
    
    output_dir = project_root / "04-26_data_analysis_and_results" / "outputs" / f"logo_cv_run{suffix}_{run_timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    input_files = sorted(list(processed_dir.glob("P*_labeled.csv")))
    if not input_files:
        print("ERROR: No labeled data found.")
        return 1
        
    print("=" * 60)
    print("STEP 3: LOGO-CV RF PIPELINE (STAY vs SKIP)")
    print(f"Total Participants: {len(input_files)}")
    print("=" * 60)
    
    # Preload and extract all data natively matching perfectly logic from Step 2
    participant_data = {}
    
    for f_path in input_files:
        pid = f_path.stem.split('_')[0]
        df_raw = pd.read_csv(f_path)
        
        if normalize:
            baseline_powers, fs = utils.compute_100s_baseline_power(df_raw, notch_freq=notch_freq)
        else:
            baseline_powers = None
            
        df_scaled, feature_names = utils.build_normalized_features(df_raw, baseline_powers, notch_freq=notch_freq)
        X, y, _ = step2_train_rf.create_aggregated_samples(df_scaled, feature_names, window_seconds=args.window)
        
        # We rebalance at the very end directly on the aggregated pools, but wait...
        # If we concatenate unbalanced arrays, the overall majority class might swamp. 
        # But for LOGO-CV, balancing individual participant chunks first guarantees uniform density scaling.
        # Legacy pipeline: "The model was trained... on the 25 included participants."
        # We will conservatively balance the test set natively, and simply balance the aggregated train set.
        
        if len(np.unique(y)) < 2:
            print(f"Skipping {pid} - Single class detected.")
            continue
            
        participant_data[pid] = {'X': X, 'y': y}
        
    if not participant_data:
        print("No valid extracted data to process.")
        return 1
        
    # Execute LOGO-CV
    logo_metrics = []
    
    for test_pid in participant_data.keys():
        print(f"\nEvaluating fold: Held-Out Test = {test_pid}")
        
        # Separate Test
        X_test_unbal = participant_data[test_pid]['X']
        y_test_unbal = participant_data[test_pid]['y']
        X_test, y_test = step2_train_rf.rebalance_dataset(X_test_unbal, y_test_unbal)
        
        # Aggregate Traing subset identically 
        X_train_list = []
        y_train_list = []
        for train_pid, data_dict in participant_data.items():
            if train_pid != test_pid:
                X_train_list.append(data_dict['X'])
                y_train_list.append(data_dict['y'])
                
        X_train_unbal = np.concatenate(X_train_list, axis=0)
        y_train_unbal = np.concatenate(y_train_list, axis=0)
        X_train, y_train = step2_train_rf.rebalance_dataset(X_train_unbal, y_train_unbal)
        
        print(f"   Train subset: N={len(y_train)} | Test subset: N={len(y_test)}")
        
        clf, results = step2_train_rf.train_and_eval_rf(X_train, y_train, X_test, y_test)
        
        print(f"   - Val Accuracy: {results['accuracy']:.3f}")
        print(f"   - Val Recall  : {results['recall']:.3f}")
        
        results['test_participant'] = test_pid
        logo_metrics.append(results)
        
    mean_acc = np.mean([m['accuracy'] for m in logo_metrics])
    mean_rec = np.mean([m['recall'] for m in logo_metrics])
    
    print(f"\n{'='*60}")
    print("LOGO-CV AGGREGATE PERFORMANCE (Generalizability)")
    print(f"Mean Accuracy: {mean_acc:.4f} ± {np.std([m['accuracy'] for m in logo_metrics]):.4f}")
    print(f"Mean Recall  : {mean_rec:.4f} ± {np.std([m['recall'] for m in logo_metrics]):.4f}")
    
    summary_dump = {
        'aggregate': {
            'mean_accuracy': mean_acc,
            'mean_recall': mean_rec
        },
        'folds': logo_metrics
    }
    with open(output_dir / "logo_cv_summary.json", 'w') as f:
        json.dump(summary_dump, f, indent=2)
        
    print(f"\nFinished. Outputs written to {output_dir.name}")
    return 0

if __name__ == "__main__":
    main()
