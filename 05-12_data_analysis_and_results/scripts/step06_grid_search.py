#!/usr/bin/env python3
"""
step06_grid_search.py — Nested CV hyperparameter tuning.

Nested CV on full feature set, seed 0 manifests only, inner 3-fold on
training data. Locks best_params.json per participant before any
outer-fold evaluation touches test data.

OUT: best_params.json per participant + grid_search_report.csv
"""

import numpy as np
import pickle
import json
from pathlib import Path
from itertools import product
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import recall_score

import config

import warnings
warnings.filterwarnings('ignore')


def run_inner_cv(X_train, y_train, param_combo, inner_folds=3, seed=0):
    """Run inner CV to evaluate a single hyperparameter combination."""
    tscv = TimeSeriesSplit(n_splits=inner_folds)
    recalls = []

    for train_idx, val_idx in tscv.split(X_train):
        if len(np.unique(y_train[train_idx])) < 2:
            continue
        if len(np.unique(y_train[val_idx])) < 2:
            continue

        clf = RandomForestClassifier(
            n_estimators=param_combo['n_estimators'],
            max_depth=param_combo['max_depth'],
            min_samples_leaf=param_combo['min_samples_leaf'],
            random_state=seed,
            n_jobs=-1,
        )
        clf.fit(X_train[train_idx], y_train[train_idx])
        y_pred = clf.predict(X_train[val_idx])
        rec = recall_score(y_train[val_idx], y_pred, pos_label=1, zero_division=0)
        recalls.append(rec)

    return np.mean(recalls) if recalls else 0.0


def grid_search_participant(pid, features_dir, splits_dir, params):
    """Run grid search for a single participant using seed 0."""
    p = params.get('step06', {})
    grid = p.get('param_grid', {
        'n_estimators': [100, 200, 300],
        'max_depth': [5, 7, 10],
        'min_samples_leaf': [3, 5, 10],
    })
    inner_folds = p.get('inner_cv_folds', 3)
    search_seed = p.get('seed', 0)

    # Load features
    with open(features_dir / f"P{pid}.pkl", 'rb') as f:
        feat_data = pickle.load(f)

    X_full = feat_data['features_full']
    labels = feat_data['labels']
    window_ids = feat_data['window_ids']

    # Load splits (seed 0)
    with open(splits_dir / f"P{pid}_splits.pkl", 'rb') as f:
        split_data = pickle.load(f)

    splits = split_data['splits'][search_seed]

    # Build window_id → index mapping
    wid_to_idx = {int(wid): i for i, wid in enumerate(window_ids)}

    # Aggregate all training data across outer folds for grid search
    # (we use inner CV within each outer fold's training set)
    param_combos = []
    keys = sorted(grid.keys())
    for vals in product(*[grid[k] for k in keys]):
        param_combos.append(dict(zip(keys, vals)))

    best_score = -1
    best_params = param_combos[0]
    report_rows = []

    for combo in param_combos:
        fold_scores = []

        for fold_info in splits:
            train_ids = fold_info['train_ids']
            train_indices = [wid_to_idx[wid] for wid in train_ids if wid in wid_to_idx]

            if len(train_indices) < 10:
                continue

            X_train = X_full[train_indices]
            y_train = labels[train_indices]

            if len(np.unique(y_train)) < 2:
                continue

            score = run_inner_cv(X_train, y_train, combo, inner_folds, search_seed)
            fold_scores.append(score)

        mean_score = np.mean(fold_scores) if fold_scores else 0.0
        report_rows.append({**combo, 'mean_inner_recall': round(mean_score, 4)})

        if mean_score > best_score:
            best_score = mean_score
            best_params = combo

    return best_params, best_score, report_rows


def run(run_dir, params):
    """Entry point called by run.py."""
    config.pprint_step(6, "GRID SEARCH")

    features_dir = run_dir / "features"
    splits_dir = run_dir / "splits"
    gs_dir = run_dir / "grid_search"
    gs_dir.mkdir(parents=True, exist_ok=True)

    all_report = []

    for pid in config.INCLUDED_PARTICIPANTS:
        feat_path = features_dir / f"P{pid}.pkl"
        split_path = splits_dir / f"P{pid}_splits.pkl"

        if not feat_path.exists() or not split_path.exists():
            print(f"  P{pid}: ⚠ Missing data, skipping.")
            continue

        best_params, best_score, report = grid_search_participant(
            pid, features_dir, splits_dir, params
        )

        # Save best params
        with open(gs_dir / f"P{pid}_best_params.json", 'w') as f:
            json.dump(best_params, f, indent=2)

        for row in report:
            row['pid'] = pid
        all_report.extend(report)

        print(f"  P{pid}: best={best_params}, inner_recall={best_score:.3f}")

    # Save report
    df_report = pd.DataFrame(all_report)
    df_report.to_csv(gs_dir / "grid_search_report.csv", index=False)

    print(f"\n  ✓ Grid search complete. Best params locked per participant.")
    print(f"    Decision locked: RF hyperparameters fixed for all subsequent training.")


if __name__ == "__main__":
    print("Use run.py to execute the pipeline.")
