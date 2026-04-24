#!/usr/bin/env python3
"""
Step 10: SFEI Battle-test

Evaluates the newly proposed Short-Form Engagement Index (SFEI):
    SFEI = AF8_high_gamma_mean / AF7_delta_min
against the traditional Engagement Index (EI).
Runs identical k-fold Logistic Regression classification for fair comparison.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import json
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, recall_score

import utils
import step2_train_rf

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

def main():
    project_root = Path(__file__).resolve().parent.parent.parent
    processed_dir = project_root / "04-26_data_analysis_and_results" / "processed_data"
    outputs_dir = project_root / "04-26_data_analysis_and_results" / "outputs"
    
    out_dir = outputs_dir / f"sfei_battletest_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("STEP 10: SFEI BATTLE-TEST (SFEI vs Traditional EI)")
    print("Proposed Formula: SFEI = AF8_high_gamma_mean / AF7_delta_min")
    print("=" * 60)
    
    input_files = list(processed_dir.glob("P*_labeled.csv"))
    if not input_files:
        print("No labeled data found.")
        return 1
        
    sfei_metrics = []
    
    for f_path in input_files:
        pid = f_path.stem.split('_')[0]
        df_raw = pd.read_csv(f_path)
        
        # 1. Normalize data (NoNotch)
        baseline_powers, fs = utils.compute_100s_baseline_power(df_raw, notch_freq=None)
        df_scaled, feature_names = utils.build_normalized_features(df_raw, baseline_powers, notch_freq=None)
        
        # 2. Extract block samples
        X, y, agg_names = step2_train_rf.create_aggregated_samples(df_scaled, feature_names, window_seconds=3.0)
        
        if len(np.unique(y)) < 2:
            continue
            
        # 3. Calculate SFEI for each block
        idx_num = agg_names.index('AF8_high_gamma_mean')
        idx_den = agg_names.index('AF7_delta_min')
        
        sfei_values = X[:, idx_num] / (X[:, idx_den] + 1e-8)
        
        X_sfei = sfei_values.reshape(-1, 1)
        
        # 4. K-fold CV identical to EI
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        accs, recs = [], []
        
        for train_i, test_i in skf.split(X_sfei, y):
            if len(np.unique(y[train_i])) < 2: continue
            
            lr = LogisticRegression(class_weight='balanced')
            # Normalize internal scaler to avoid extreme ratio values crashing LR
            from sklearn.preprocessing import RobustScaler
            scaler = RobustScaler()
            X_tr = scaler.fit_transform(X_sfei[train_i])
            X_te = scaler.transform(X_sfei[test_i])
            
            lr.fit(X_tr, y[train_i])
            preds = lr.predict(X_te)
            
            accs.append(accuracy_score(y[test_i], preds))
            recs.append(recall_score(y[test_i], preds, pos_label=1, zero_division=0))
            
        if accs:
            sfei_metrics.append({
                'participant': pid,
                'accuracy': float(np.mean(accs)),
                'recall': float(np.mean(recs))
            })
            
    # Calculate Aggregate
    mean_sfei_acc = np.mean([m['accuracy'] for m in sfei_metrics])
    mean_sfei_rec = np.mean([m['recall'] for m in sfei_metrics])
    
    print("\n--- RESULTS ---")
    print(f"Traditional EI Mean Accuracy: ~53.6%")
    print(f"SFEI Mean Accuracy: {mean_sfei_acc*100:.1f}%")
    print(f"Traditional EI Mean Recall: ~54.0%")
    print(f"SFEI Mean Recall: {mean_sfei_rec*100:.1f}%")
    
    with open(out_dir / "sfei_metrics.json", "w") as f:
        json.dump({'aggregate': {'accuracy': mean_sfei_acc, 'recall': mean_sfei_rec}, 'participants': sfei_metrics}, f, indent=2)
        
    print(f"\nOutputs written to {out_dir.name}")
    return 0

if __name__ == "__main__":
    main()
