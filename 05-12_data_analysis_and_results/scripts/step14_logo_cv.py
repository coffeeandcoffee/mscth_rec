#!/usr/bin/env python3
"""
step14_logo_cv.py — Leave-One-Group-Out Cross-Validation.

Trains on 24 participants, tests on held-out 1. Reports STAY recall and
full confusion matrix per fold. Uses locked best_params.json.

OUT: logo_results.csv + logo_confusion.json
"""

import numpy as np
import pickle
import json
from pathlib import Path
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, recall_score, confusion_matrix)
import config
import warnings
warnings.filterwarnings('ignore')


def run(run_dir, params):
    config.pprint_step(14, "LOGO-CV")

    features_dir = run_dir / "features"
    splits_dir = run_dir / "splits"
    gs_dir = run_dir / "grid_search"

    # Load all participant feature data and Step 04 validated splits
    all_data = {}
    for pid in config.INCLUDED_PARTICIPANTS:
        fp = features_dir / f"P{pid}.pkl"
        sp = splits_dir / f"P{pid}_splits.pkl"
        if not fp.exists() or not sp.exists():
            continue
            
        with open(fp, 'rb') as f:
            fd = pickle.load(f)
        with open(sp, 'rb') as f:
            sd = pickle.load(f)
            
        # We strictly use the union of Step 04 mathematically balanced test_ids (Seed 0)
        # to form the completely balanced, validated subset of windows for this participant.
        splits_seed0 = sd['splits'].get(0, [])
        valid_test_ids = []
        for fold in splits_seed0:
            valid_test_ids.extend(fold['test_ids'])
            
        # Map these window IDs to feature indices
        wid_to_idx = {int(wid): i for i, wid in enumerate(fd['window_ids'])}
        valid_indices = [wid_to_idx[wid] for wid in valid_test_ids if wid in wid_to_idx]
        
        all_data[pid] = {
            'X_bal': fd['features_full'][valid_indices],
            'y_bal': fd['labels'][valid_indices]
        }

    if len(all_data) < 5:
        print("  ⚠ Too few participants for LOGO-CV.")
        return

    fold_rows = []
    agg_cm = np.zeros((2, 2), dtype=int)

    for test_pid in sorted(all_data.keys()):
        # Train data: concatenate validated Step 04 balanced data for all EXCEPT test_pid
        X_train_parts, y_train_parts = [], []
        for pid, data in all_data.items():
            if pid == test_pid:
                continue
            X_train_parts.append(data['X_bal'])
            y_train_parts.append(data['y_bal'])

        X_train_bal = np.concatenate(X_train_parts)
        y_train_bal = np.concatenate(y_train_parts)

        # Test data: validated Step 04 balanced data FOR test_pid
        X_test_bal = all_data[test_pid]['X_bal']
        y_test_bal = all_data[test_pid]['y_bal']

        if len(np.unique(y_train_bal)) < 2 or len(np.unique(y_test_bal)) < 2:
            continue

        # Use median hyperparams across all participants
        bp_path = gs_dir / f"P{test_pid}_best_params.json"
        bp = json.load(open(bp_path)) if bp_path.exists() else {}

        clf = RandomForestClassifier(
            n_estimators=bp.get('n_estimators', 200),
            max_depth=bp.get('max_depth', 7),
            min_samples_leaf=bp.get('min_samples_leaf', 5),
            random_state=42, n_jobs=-1)
        clf.fit(X_train_bal, y_train_bal)
        y_pred = clf.predict(X_test_bal)

        acc = accuracy_score(y_test_bal, y_pred)
        rec = recall_score(y_test_bal, y_pred, pos_label=1, zero_division=0)
        cm = confusion_matrix(y_test_bal, y_pred, labels=[0, 1])
        agg_cm += cm

        fold_rows.append({
            'test_pid': test_pid,
            'n_train': len(y_train_bal),
            'n_test': len(y_test_bal),
            'accuracy': round(acc, 4),
            'recall': round(rec, 4),
            'cm_tn': int(cm[0, 0]),
            'cm_fp': int(cm[0, 1]),
            'cm_fn': int(cm[1, 0]),
            'cm_tp': int(cm[1, 1]),
        })
        print(f"  Held-out P{test_pid}: acc={acc:.3f}, recall={rec:.3f}")

    df = pd.DataFrame(fold_rows)
    df.to_csv(run_dir / "logo_results.csv", index=False)

    # Save aggregate confusion matrix
    logo_result = {
        'mean_accuracy': float(df['accuracy'].mean()),
        'mean_recall': float(df['recall'].mean()),
        'std_accuracy': float(df['accuracy'].std()),
        'std_recall': float(df['recall'].std()),
        'n_folds': len(fold_rows),
        'aggregate_confusion_matrix': agg_cm.tolist(),
    }
    with open(run_dir / "logo_confusion.json", 'w') as f:
        json.dump(logo_result, f, indent=2)

    print(f"\n  LOGO-CV: mean acc={logo_result['mean_accuracy']:.3f}, "
          f"mean recall={logo_result['mean_recall']:.3f}")
    print(f"  Aggregate CM: {agg_cm.tolist()}")
    print(f"  ✓ LOGO results saved.")
