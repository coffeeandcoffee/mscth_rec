#!/usr/bin/env python3
"""
Step 4: Engagement Index (EI) Comparison

Computes the standard Muse Engagement Index:
    EI = beta / (alpha + theta)
Provides baseline diagnostic proof using Case B (STAY) logic.
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import json
from scipy import stats
from datetime import datetime
import matplotlib.pyplot as plt

import utils
import warnings
warnings.filterwarnings('ignore')

def compute_ei_per_window(df, notch_freq=None, window_seconds=3.0):
    """
    Computes Standard EI for STAY and SKIP arrays explicitly tracing raw bounds.
    """
    fs = 1.0 / np.median(np.diff(df['lsl_timestamp'].values))
    timestamps = df['lsl_timestamp'].values
    engagement_state = df['engagement_state'].values
    keypress_A = df['keypress_A'].values
    
    df_sig = df.copy()
    if notch_freq:
        for ch in utils.EEG_CHANNELS:
            df_sig[ch] = utils.apply_notch_filter(df_sig[ch].values, fs, notch_freq)
            
    ei_by_class = {'STAY': [], 'SKIP': []}
    
    def _get_ei(block_df):
        alpha_powers = []
        beta_powers = []
        theta_powers = []
        
        for ch in utils.EEG_CHANNELS:
            ch_data = block_df[ch].values
            alpha_powers.append(np.mean(utils.extract_band_power(ch_data, fs, 8, 13) ** 2))
            beta_powers.append(np.mean(utils.extract_band_power(ch_data, fs, 13, 30) ** 2))
            theta_powers.append(np.mean(utils.extract_band_power(ch_data, fs, 4, 8) ** 2))
            
        a, t, b = np.mean(alpha_powers), np.mean(theta_powers), np.mean(beta_powers)
        denom = a + t
        if denom < 1e-12:
            return None
        return b / denom
        
    # --- SKIP (0) windows ---
    kp_indices = np.where(keypress_A == 1)[0]
    for idx in kp_indices:
        t_end = timestamps[idx]
        t_start = t_end - window_seconds
        mask = (timestamps >= t_start) & (timestamps < t_end)
        block = df_sig[mask]
        if len(block) >= 10:
            ei = _get_ei(block)
            if ei is not None:
                ei_by_class['SKIP'].append(ei)
                
    # --- STAY (1) windows ---
    stay_indices = np.where(engagement_state == 1)[0]
    if len(stay_indices) > 0:
        breaks = np.where(np.diff(stay_indices) > 1)[0] + 1
        regions = np.split(stay_indices, breaks)
        for r in regions:
            if len(r) < 10: continue
            r_ts = timestamps[r]
            if (r_ts[-1] - r_ts[0]) < window_seconds: continue
            
            t = r_ts[0]
            stride = window_seconds * 0.2
            while t <= (r_ts[-1] - window_seconds):
                mask = (timestamps >= t) & (timestamps < t + window_seconds) & (engagement_state == 1)
                block = df_sig[mask]
                if len(block) >= 10:
                    ei = _get_ei(block)
                    if ei is not None:
                        ei_by_class['STAY'].append(ei)
                t += stride
                
    return ei_by_class

def run_stats(ei_stay, ei_skip):
    results = {
        'stay_mean': float(np.mean(ei_stay)),
        'skip_mean': float(np.mean(ei_skip)),
        'stay_n': len(ei_stay),
        'skip_n': len(ei_skip)
    }
    
    # Mann Whitney U
    u, p = stats.mannwhitneyu(ei_stay, ei_skip, alternative='two-sided')
    results['mannwhitney_p'] = float(p)
    
    # Cohen's d (STAY vs SKIP)
    pooled_std = np.sqrt((np.std(ei_stay)**2 + np.std(ei_skip)**2) / 2)
    delta = results['stay_mean'] - results['skip_mean']
    results['cohens_d'] = float(delta / pooled_std) if pooled_std > 1e-12 else 0.0
    return results

def main():
    parser = argparse.ArgumentParser(description='Step 4: Engagement Index Evaluator')
    parser.add_argument('--nonotch', action='store_true')
    args = parser.parse_args()
    
    out_dir = Path(__file__).resolve().parent.parent.parent / "04-26_data_analysis_and_results" / "outputs" / f"ei_stats_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    data_dir = Path(__file__).resolve().parent.parent.parent / "04-26_data_analysis_and_results" / "processed_data"
    files = list(data_dir.glob("P*_labeled.csv"))
    
    all_results = {}
    
    print("=" * 60)
    print("STEP 4: ENGAGEMENT INDEX (STAY vs SKIP)")
    print("=" * 60)
    
    for f in files:
        pid = f.stem.split('_')[0]
        df = pd.read_csv(f)
        
        notch = None if args.nonotch else 50.0
        ei_dict = compute_ei_per_window(df, notch_freq=notch)
        
        if len(ei_dict['STAY']) < 3 or len(ei_dict['SKIP']) < 3:
            continue
            
        st = run_stats(ei_dict['STAY'], ei_dict['SKIP'])
        
        # Intra-EI ML calculation to compare fairly against Intra-RF
        X_ei = np.array(ei_dict['STAY'] + ei_dict['SKIP']).reshape(-1, 1)
        y_ei = np.array([1] * len(ei_dict['STAY']) + [0] * len(ei_dict['SKIP']))
        
        # Simple balanced logistic classification directly evaluating diagnostic sensitivity
        if len(np.unique(y_ei)) > 1:
            from sklearn.linear_model import LogisticRegression
            from sklearn.model_selection import StratifiedKFold
            from sklearn.metrics import accuracy_score, recall_score, precision_score
            
            # Since N can be small, use 5-fold CV to get a stable estimate
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            accs, recs, precs = [], [], []
            for train_i, test_i in skf.split(X_ei, y_ei):
                lr = LogisticRegression(class_weight='balanced')
                # If only one class in split failsafe (some pids have very few samples)
                if len(np.unique(y_ei[train_i])) < 2: continue
                lr.fit(X_ei[train_i], y_ei[train_i])
                preds = lr.predict(X_ei[test_i])
                accs.append(accuracy_score(y_ei[test_i], preds))
                recs.append(recall_score(y_ei[test_i], preds, pos_label=1, zero_division=0))
                precs.append(precision_score(y_ei[test_i], preds, pos_label=1, zero_division=0))
                
            if accs:
                st['intra_ei_ml'] = {
                    'accuracy': float(np.mean(accs)),
                    'recall': float(np.mean(recs)),
                    'precision': float(np.mean(precs)),
                    'f1': 0.0 # Placeholder
                }
        
        all_results[pid] = st
        print(f"P{pid} -> STAY Mean: {st['stay_mean']:.3f} | SKIP Mean: {st['skip_mean']:.3f} | p={st['mannwhitney_p']:.4f}")
        
    with open(out_dir/'ei_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"Outputs saved to {out_dir.name}")

if __name__ == "__main__":
    main()
