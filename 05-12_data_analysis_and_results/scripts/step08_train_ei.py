#!/usr/bin/env python3
"""
step08_train_ei.py — Engagement Index logistic regression comparison.

Runs EI on balanced data with identical temporal blocked 5-fold CV.
Apples-to-apples comparison with step07 RF.

OUT: per-participant EI STAY recall + ei_summary.csv
"""

import numpy as np
import pickle
import json
from pathlib import Path
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, recall_score

import config

import warnings
warnings.filterwarnings('ignore')


def run(run_dir, params):
    """Entry point called by run.py."""
    config.pprint_step(8, "ENGAGEMENT INDEX COMPARISON")

    features_dir = run_dir / "features"
    splits_dir = run_dir / "splits"
    results_dir = run_dir / "results" / "ei"
    results_dir.mkdir(parents=True, exist_ok=True)

    p = params.get('step07', {})
    seeds = p.get('seeds', [0, 1, 7, 42, 99])

    all_results = []

    for pid in config.INCLUDED_PARTICIPANTS:
        feat_path = features_dir / f"P{pid}.pkl"
        split_path = splits_dir / f"P{pid}_splits.pkl"

        if not feat_path.exists() or not split_path.exists():
            continue

        with open(feat_path, 'rb') as f:
            feat_data = pickle.load(f)
        with open(split_path, 'rb') as f:
            split_data = pickle.load(f)

        ei_values = feat_data['ei_values']
        labels = feat_data['labels']
        window_ids = feat_data['window_ids']
        wid_to_idx = {int(wid): i for i, wid in enumerate(window_ids)}

        seed_recalls = []

        for seed in seeds:
            splits = split_data['splits'].get(seed, [])
            fold_recalls = []

            for fold_info in splits:
                train_indices = [wid_to_idx[wid] for wid in fold_info['train_ids']
                                 if wid in wid_to_idx]
                test_indices = [wid_to_idx[wid] for wid in fold_info['test_ids']
                                if wid in wid_to_idx]

                if len(train_indices) < 4 or len(test_indices) < 4:
                    continue

                X_train = ei_values[train_indices].reshape(-1, 1)
                y_train = labels[train_indices]
                X_test = ei_values[test_indices].reshape(-1, 1)
                y_test = labels[test_indices]

                if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
                    continue

                scaler = StandardScaler()
                X_train_s = scaler.fit_transform(X_train)
                X_test_s = scaler.transform(X_test)

                clf = LogisticRegression(class_weight='balanced', random_state=seed)
                clf.fit(X_train_s, y_train)
                y_pred = clf.predict(X_test_s)

                rec = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
                fold_recalls.append(rec)

            if fold_recalls:
                seed_recalls.append(np.mean(fold_recalls))

        if seed_recalls:
            mean_recall = float(np.mean(seed_recalls))
        else:
            mean_recall = 0.5

        result = {
            'pid': pid,
            'ei_recall': mean_recall,
            'ei_recall_std': float(np.std(seed_recalls)) if seed_recalls else 0.0,
            'n_seeds': len(seed_recalls),
        }
        all_results.append(result)

        # Save per-participant
        with open(results_dir / f"P{pid}_ei.pkl", 'wb') as f:
            pickle.dump(result, f)

        print(f"  P{pid}: EI STAY recall = {mean_recall:.3f}")

    # Summary CSV
    df_summary = pd.DataFrame(all_results)
    df_summary.to_csv(run_dir / "ei_summary.csv", index=False)

    mean_ei = df_summary['ei_recall'].mean()
    print(f"\n  ✓ Mean EI STAY recall: {mean_ei:.4f} across {len(all_results)} participants.")


if __name__ == "__main__":
    print("Use run.py to execute the pipeline.")
