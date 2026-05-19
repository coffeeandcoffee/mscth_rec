#!/usr/bin/env python3
"""
step13_sensitivity_manifests.py — Sensitivity analyses.

Runs the step04→step07 chain on sensitivity manifests from step03 using locked
best_params.json. No grid search re-run. Compares each variant to primary.

OUT: sensitivity_comparisons.csv
"""

import numpy as np
import pickle
import json
from pathlib import Path
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score
from scipy import stats as sp_stats
import config
import warnings
warnings.filterwarnings('ignore')

from step04_balance_split import build_cv_splits
from step05_feature_engineering import process_windows


def run_sensitivity_variant(run_dir, manifest_name, params):
    """Run balance→features→train chain on a sensitivity manifest."""
    windows_dir = run_dir / "windows" / manifest_name
    gs_dir = run_dir / "grid_search"
    seeds = params.get('step07', {}).get('seeds', [0, 1, 7, 42, 99])

    recalls_per_pid = {}

    for pid in config.INCLUDED_PARTICIPANTS:
        win_path = windows_dir / f"P{pid}.pkl"
        bp_path = gs_dir / f"P{pid}_best_params.json"
        if not win_path.exists():
            continue

        with open(win_path, 'rb') as f:
            win_data = pickle.load(f)

        windows = win_data['windows']
        feature_names = win_data['feature_names']
        bp = json.load(open(bp_path)) if bp_path.exists() else {}

        if len(windows) < 10:
            continue

        # Feature engineering
        feat = process_windows(windows, feature_names)
        if feat is None:
            continue

        X = feat['features_full']
        labels = feat['labels']
        wids = feat['window_ids']

        # Balance and split
        splits = build_cv_splits(windows, params)
        wid_map = {int(w): i for i, w in enumerate(wids)}

        seed_recalls = []
        for seed in seeds:
            fold_recalls = []
            for fi in splits.get(seed, []):
                tri = [wid_map[w] for w in fi['train_ids'] if w in wid_map]
                tei = [wid_map[w] for w in fi['test_ids'] if w in wid_map]
                if len(tri) < 4 or len(tei) < 4:
                    continue
                Xtr, ytr = X[tri], labels[tri]
                Xte, yte = X[tei], labels[tei]
                if len(np.unique(ytr)) < 2 or len(np.unique(yte)) < 2:
                    continue
                clf = RandomForestClassifier(
                    n_estimators=bp.get('n_estimators', 200),
                    max_depth=bp.get('max_depth', 7),
                    min_samples_leaf=bp.get('min_samples_leaf', 5),
                    random_state=seed, n_jobs=-1)
                clf.fit(Xtr, ytr)
                fold_recalls.append(
                    recall_score(yte, clf.predict(Xte), pos_label=1, zero_division=0))
            if fold_recalls:
                seed_recalls.append(np.mean(fold_recalls))

        if seed_recalls:
            recalls_per_pid[pid] = float(np.mean(seed_recalls))

    return recalls_per_pid


def run(run_dir, params):
    config.pprint_step(13, "SENSITIVITY ANALYSES")

    # Load primary recalls
    with open(run_dir / "step07_primary_result.json", 'r') as f:
        primary = json.load(f)
    primary_map = {r['pid']: r['recall'] for r in primary['per_participant']}

    variants = [
        ('artifact_exclude', 'Artifact-excluded windows'),
        ('burst_exclude', 'Burst-skip-excluded from SKIP'),
        ('artifact_exclude_burst_exclude', 'Both excluded'),
    ]

    rows = []
    for manifest_name, description in variants:
        print(f"\n  Running: {description}...")
        var_recalls = run_sensitivity_variant(run_dir, manifest_name, params)

        if not var_recalls:
            print(f"    ⚠ No results for {manifest_name}")
            continue

        # Match participants
        matched_primary = []
        matched_var = []
        for pid in sorted(var_recalls.keys()):
            if pid in primary_map:
                matched_primary.append(primary_map[pid])
                matched_var.append(var_recalls[pid])

        p_arr = np.array(matched_primary)
        v_arr = np.array(matched_var)

        try:
            w, p = sp_stats.wilcoxon(p_arr - v_arr)
            n = len(p_arr)
            r = 1 - (2 * w) / (n * (n + 1) / 2)
        except Exception:
            w, p, r = 0.0, 1.0, 0.0

        rows.append({
            'comparison': manifest_name,
            'description': description,
            'n_participants': len(matched_primary),
            'mean_primary': round(float(np.mean(p_arr)), 4),
            'mean_variant': round(float(np.mean(v_arr)), 4),
            'delta': round(float(np.mean(p_arr) - np.mean(v_arr)), 4),
            'wilcoxon_W': round(float(w), 2),
            'p_value': float(p),
            'effect_r': round(float(r), 4),
            'significant': p < 0.05,
        })
        sig = "SIG" if p < 0.05 else "n.s."
        print(f"    {description}: Δ={np.mean(p_arr)-np.mean(v_arr):+.3f}, "
              f"p={p:.4f} ({sig})")

    df = pd.DataFrame(rows)
    df.to_csv(run_dir / "sensitivity_comparisons.csv", index=False)
    print(f"\n  ✓ Sensitivity comparisons saved.")
