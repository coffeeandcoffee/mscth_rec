#!/usr/bin/env python3
"""
step07_train_intra.py — Primary intra-subject RF training.

Trains RF per participant per feature set per seed using locked best_params.json.
Runs temporal blocked CV (primary) and random-split 60/40 (legacy leakage comparison).
Produces step07_primary_result.json with Wilcoxon test vs 50% chance.

This is where the primary result number is produced.

OUT: per-participant result matrices + confusion matrices + step07_primary_result.json
"""

import numpy as np
import pickle
import json
from pathlib import Path
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, recall_score, precision_score,
                             f1_score, confusion_matrix)
from scipy import stats as sp_stats

import config

import warnings
warnings.filterwarnings('ignore')


def load_best_params(gs_dir, pid):
    """Load locked hyperparameters for a participant."""
    path = gs_dir / f"P{pid}_best_params.json"
    if path.exists():
        with open(path, 'r') as f:
            return json.load(f)
    # Fallback defaults
    return {'n_estimators': 200, 'max_depth': 7, 'min_samples_leaf': 5}


def train_temporal_blocked(X, labels, window_ids, splits, best_params, seed):
    """Train using temporal blocked CV for a single seed."""
    wid_to_idx = {int(wid): i for i, wid in enumerate(window_ids)}
    fold_results = []

    for fold_info in splits:
        train_indices = [wid_to_idx[wid] for wid in fold_info['train_ids'] if wid in wid_to_idx]
        test_indices = [wid_to_idx[wid] for wid in fold_info['test_ids'] if wid in wid_to_idx]

        if len(train_indices) < 4 or len(test_indices) < 4:
            continue

        X_train, y_train = X[train_indices], labels[train_indices]
        X_test, y_test = X[test_indices], labels[test_indices]

        if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
            continue

        clf = RandomForestClassifier(
            n_estimators=best_params.get('n_estimators', 200),
            max_depth=best_params.get('max_depth', 7),
            min_samples_leaf=best_params.get('min_samples_leaf', 5),
            random_state=seed,
            n_jobs=-1,
        )
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        fold_results.append({
            'accuracy': accuracy_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred, pos_label=1, zero_division=0),
            'precision': precision_score(y_test, y_pred, pos_label=1, zero_division=0),
            'f1': f1_score(y_test, y_pred, pos_label=1, zero_division=0),
            'cm': confusion_matrix(y_test, y_pred, labels=[0, 1]),
            'model': clf,
        })

    return fold_results


def train_random_split(X, labels, best_params, seed, split_ratio=0.6):
    """Train using random 60/40 split (legacy, for leakage comparison)."""
    rng = np.random.RandomState(seed)
    n = len(labels)
    indices = rng.permutation(n)
    split_point = int(n * split_ratio)

    train_idx = indices[:split_point]
    test_idx = indices[split_point:]

    if len(train_idx) < 4 or len(test_idx) < 4:
        return []

    X_train, y_train = X[train_idx], labels[train_idx]
    X_test, y_test = X[test_idx], labels[test_idx]

    if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
        return []

    clf = RandomForestClassifier(
        n_estimators=best_params.get('n_estimators', 200),
        max_depth=best_params.get('max_depth', 7),
        min_samples_leaf=best_params.get('min_samples_leaf', 5),
        random_state=seed,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    return [{
        'accuracy': accuracy_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred, pos_label=1, zero_division=0),
        'precision': precision_score(y_test, y_pred, pos_label=1, zero_division=0),
        'f1': f1_score(y_test, y_pred, pos_label=1, zero_division=0),
        'cm': confusion_matrix(y_test, y_pred, labels=[0, 1]),
    }]


def run(run_dir, params):
    """Entry point called by run.py."""
    config.pprint_step(7, "INTRA-SUBJECT TRAINING (PRIMARY RESULT)")

    features_dir = run_dir / "features"
    splits_dir = run_dir / "splits"
    gs_dir = run_dir / "grid_search"
    results_dir = run_dir / "results" / "intra"
    results_dir.mkdir(parents=True, exist_ok=True)

    p = params.get('step07', {})
    feature_sets = p.get('feature_sets', ['full', 'frontal', 'temporal'])
    seeds = p.get('seeds', [0, 1, 7, 42, 99])
    eval_protocols = p.get('eval_protocols', ['temporal_blocked', 'random_split'])

    all_participant_results = []

    for pid in config.INCLUDED_PARTICIPANTS:
        feat_path = features_dir / f"P{pid}.pkl"
        split_path = splits_dir / f"P{pid}_splits.pkl"

        if not feat_path.exists() or not split_path.exists():
            print(f"  P{pid}: ⚠ Missing data, skipping.")
            continue

        with open(feat_path, 'rb') as f:
            feat_data = pickle.load(f)
        with open(split_path, 'rb') as f:
            split_data = pickle.load(f)

        best_params = load_best_params(gs_dir, pid)
        window_ids = feat_data['window_ids']
        labels = feat_data['labels']

        pid_results = {
            'pid': pid,
            'best_params': best_params,
            'results': {},  # protocol → feature_set → seed → metrics
        }

        for protocol in eval_protocols:
            pid_results['results'][protocol] = {}

            for fset in feature_sets:
                X = feat_data[f'features_{fset}']
                pid_results['results'][protocol][fset] = {}

                for seed in seeds:
                    if protocol == 'temporal_blocked':
                        splits = split_data['splits'].get(seed, [])
                        fold_results = train_temporal_blocked(
                            X, labels, window_ids, splits, best_params, seed)
                    else:
                        fold_results = train_random_split(
                            X, labels, best_params, seed)

                    if fold_results:
                        mean_metrics = {
                            'accuracy': float(np.mean([r['accuracy'] for r in fold_results])),
                            'recall': float(np.mean([r['recall'] for r in fold_results])),
                            'precision': float(np.mean([r['precision'] for r in fold_results])),
                            'f1': float(np.mean([r['f1'] for r in fold_results])),
                        }
                        # Aggregate confusion matrix
                        agg_cm = sum(r['cm'] for r in fold_results)
                        mean_metrics['confusion_matrix'] = agg_cm.tolist()
                    else:
                        mean_metrics = {
                            'accuracy': 0.5, 'recall': 0.5,
                            'precision': 0.5, 'f1': 0.5,
                            'confusion_matrix': [[0, 0], [0, 0]],
                        }

                    pid_results['results'][protocol][fset][seed] = mean_metrics

        # Save per-participant results
        with open(results_dir / f"P{pid}_results.pkl", 'wb') as f:
            pickle.dump(pid_results, f)

        # Extract primary number for display
        primary_recalls = []
        for seed in seeds:
            r = pid_results['results'].get('temporal_blocked', {}).get('full', {}).get(seed, {})
            primary_recalls.append(r.get('recall', 0.5))
        mean_recall = np.mean(primary_recalls)

        all_participant_results.append({
            'pid': pid,
            'mean_recall_temporal': mean_recall,
        })

        print(f"  P{pid}: primary STAY recall = {mean_recall:.3f} "
              f"(mean across {len(seeds)} seeds)")

    # ── Compute primary result: Wilcoxon vs 50% chance ──
    recalls = np.array([r['mean_recall_temporal'] for r in all_participant_results])
    n_above = int(np.sum(recalls > 0.5))

    try:
        w_stat, w_p = sp_stats.wilcoxon(recalls - 0.5)
        # Rank-biserial r
        n = len(recalls)
        r_effect = 1 - (2 * w_stat) / (n * (n + 1) / 2)
    except Exception:
        w_stat, w_p, r_effect = 0.0, 1.0, 0.0

    primary_result = {
        'metric': 'STAY_recall',
        'protocol': 'temporal_blocked_5fold_cv',
        'feature_set': 'full',
        'preprocessing': 'nonotch',
        'mean_recall': float(np.mean(recalls)),
        'std_recall': float(np.std(recalls)),
        'median_recall': float(np.median(recalls)),
        'n_participants': len(recalls),
        'n_above_chance': n_above,
        'wilcoxon_W': float(w_stat),
        'wilcoxon_p': float(w_p),
        'rank_biserial_r': float(r_effect),
        'seeds': seeds,
        'per_participant': [
            {'pid': r['pid'], 'recall': float(r['mean_recall_temporal'])}
            for r in all_participant_results
        ],
    }

    with open(run_dir / "step07_primary_result.json", 'w') as f:
        json.dump(primary_result, f, indent=2)

    print(f"\n  {'='*60}")
    print(f"  PRIMARY RESULT LOCKED:")
    print(f"    Mean STAY recall: {primary_result['mean_recall']:.4f} "
          f"± {primary_result['std_recall']:.4f}")
    print(f"    Above chance: {n_above}/{len(recalls)}")
    print(f"    Wilcoxon vs 50%: W={w_stat:.1f}, p={w_p:.6f}, r={r_effect:.3f}")
    sig = "✓ SIGNIFICANT" if w_p < 0.05 else "✗ NOT significant"
    print(f"    → {sig}")
    print(f"  {'='*60}")


if __name__ == "__main__":
    print("Use run.py to execute the pipeline.")
