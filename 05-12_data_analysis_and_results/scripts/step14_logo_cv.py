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
    gs_dir = run_dir / "grid_search"

    # Load all participant feature data
    all_data = {}
    for pid in config.INCLUDED_PARTICIPANTS:
        fp = features_dir / f"P{pid}.pkl"
        if not fp.exists():
            continue
        with open(fp, 'rb') as f:
            fd = pickle.load(f)
        all_data[pid] = fd

    if len(all_data) < 5:
        print("  ⚠ Too few participants for LOGO-CV.")
        return

    fold_rows = []
    agg_cm = np.zeros((2, 2), dtype=int)

    for test_pid in sorted(all_data.keys()):
        # Train data: all except test_pid
        X_train_parts, y_train_parts = [], []
        for pid, fd in all_data.items():
            if pid == test_pid:
                continue
            X_train_parts.append(fd['features_full'])
            y_train_parts.append(fd['labels'])

        X_train = np.concatenate(X_train_parts)
        y_train = np.concatenate(y_train_parts)

        # Balance training data
        rng = np.random.RandomState(42)
        n0 = np.sum(y_train == 0)
        n1 = np.sum(y_train == 1)
        n_min = min(n0, n1)
        if n_min == 0:
            continue

        idx0 = np.where(y_train == 0)[0]
        idx1 = np.where(y_train == 1)[0]
        idx0_sel = rng.choice(idx0, size=n_min, replace=False)
        idx1_sel = rng.choice(idx1, size=n_min, replace=False)
        bal_idx = np.concatenate([idx0_sel, idx1_sel])
        rng.shuffle(bal_idx)
        X_train_bal = X_train[bal_idx]
        y_train_bal = y_train[bal_idx]

        # Test data (balanced)
        X_test = all_data[test_pid]['features_full']
        y_test = all_data[test_pid]['labels']
        n0t = np.sum(y_test == 0)
        n1t = np.sum(y_test == 1)
        n_min_t = min(n0t, n1t)
        if n_min_t == 0:
            continue
        idx0t = np.where(y_test == 0)[0]
        idx1t = np.where(y_test == 1)[0]
        idx0t_sel = rng.choice(idx0t, size=n_min_t, replace=False)
        idx1t_sel = rng.choice(idx1t, size=n_min_t, replace=False)
        bal_idx_t = np.concatenate([idx0t_sel, idx1t_sel])
        X_test_bal = X_test[bal_idx_t]
        y_test_bal = y_test[bal_idx_t]

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
