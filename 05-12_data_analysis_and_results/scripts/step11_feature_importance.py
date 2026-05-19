#!/usr/bin/env python3
"""
step11_feature_importance.py — Gini importance + SFEI construction.

Extracts mean Gini importance across all folds × seeds from full-feature RF.
Friedman test across 7 bands. SFEI: top-2 features → ratio → logistic regression.

OUT: 112-feature ranking + band test CSV + sfei_result.json
"""

import numpy as np
import pickle
import json
from pathlib import Path
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import recall_score
from scipy import stats as sp_stats
import config
import warnings
warnings.filterwarnings('ignore')


def collect_importances(run_dir, params):
    """Train RF models and collect Gini importances across all folds×seeds."""
    features_dir = run_dir / "features"
    splits_dir = run_dir / "splits"
    gs_dir = run_dir / "grid_search"
    seeds = params.get('step07', {}).get('seeds', [0, 1, 7, 42, 99])

    per_participant_imp = {}

    for pid in config.INCLUDED_PARTICIPANTS:
        fp = features_dir / f"P{pid}.pkl"
        sp = splits_dir / f"P{pid}_splits.pkl"
        bp_p = gs_dir / f"P{pid}_best_params.json"
        if not fp.exists() or not sp.exists():
            continue

        with open(fp, 'rb') as f:
            fd = pickle.load(f)
        with open(sp, 'rb') as f:
            sd = pickle.load(f)
        bp = json.load(open(bp_p)) if bp_p.exists() else {}

        X = fd['features_full']
        labels = fd['labels']
        wids = fd['window_ids']
        wid_map = {int(w): i for i, w in enumerate(wids)}
        imps = []

        for seed in seeds:
            for fi in sd['splits'].get(seed, []):
                tri = [wid_map[w] for w in fi['train_ids'] if w in wid_map]
                if len(tri) < 10:
                    continue
                Xtr, ytr = X[tri], labels[tri]
                if len(np.unique(ytr)) < 2:
                    continue
                clf = RandomForestClassifier(
                    n_estimators=bp.get('n_estimators', 200),
                    max_depth=bp.get('max_depth', 7),
                    min_samples_leaf=bp.get('min_samples_leaf', 5),
                    random_state=seed, n_jobs=-1)
                clf.fit(Xtr, ytr)
                imps.append(clf.feature_importances_)

        if imps:
            per_participant_imp[pid] = np.mean(imps, axis=0)

    return per_participant_imp


def run(run_dir, params):
    config.pprint_step(11, "FEATURE IMPORTANCE & SFEI")

    agg_names = config.build_agg_feature_names(config.EEG_CHANNELS)
    per_pid_imp = collect_importances(run_dir, params)

    if not per_pid_imp:
        print("  ⚠ No importance data collected.")
        return

    # Aggregate across participants
    all_imp = np.array(list(per_pid_imp.values()))
    mean_imp = np.mean(all_imp, axis=0)
    std_imp = np.std(all_imp, axis=0)

    # 112-feature ranking table
    ranking = sorted(range(len(agg_names)),
                     key=lambda i: mean_imp[i], reverse=True)
    rank_rows = []
    for rank, idx in enumerate(ranking):
        rank_rows.append({
            'rank': rank + 1,
            'feature': agg_names[idx],
            'mean_importance': round(float(mean_imp[idx]), 6),
            'std_importance': round(float(std_imp[idx]), 6),
        })
    df_rank = pd.DataFrame(rank_rows)
    df_rank.to_csv(run_dir / "feature_ranking_112.csv", index=False)

    # 4×7 electrode-band heatmap data
    heatmap = {}
    for ch in config.EEG_CHANNELS:
        for band_name, _, _ in config.FREQUENCY_BANDS:
            prefix = f"{ch}_{band_name}_"
            band_imp = sum(mean_imp[i] for i, n in enumerate(agg_names)
                          if n.startswith(prefix))
            heatmap[f"{ch}_{band_name}"] = round(float(band_imp), 6)

    with open(run_dir / "electrode_band_heatmap.json", 'w') as f:
        json.dump(heatmap, f, indent=2)

    # Band-level Friedman test
    band_arrays = {}
    for band_name, _, _ in config.FREQUENCY_BANDS:
        band_arrays[band_name] = []
        for pid, imp in per_pid_imp.items():
            band_total = sum(imp[i] for i, n in enumerate(agg_names)
                           if f"_{band_name}_" in n)
            band_arrays[band_name].append(band_total)

    band_data = [np.array(v) for v in band_arrays.values()]
    try:
        friedman_stat, friedman_p = sp_stats.friedmanchisquare(*band_data)
    except Exception:
        friedman_stat, friedman_p = 0.0, 1.0

    band_rows = []
    band_names = list(band_arrays.keys())
    for i, b1 in enumerate(band_names):
        for j, b2 in enumerate(band_names):
            if j <= i:
                continue
            try:
                w, p = sp_stats.wilcoxon(
                    np.array(band_arrays[b1]) - np.array(band_arrays[b2]))
            except Exception:
                w, p = 0.0, 1.0
            # Bonferroni correction
            p_corr = min(p * 21, 1.0)  # 7 choose 2 = 21
            band_rows.append({
                'band_a': b1, 'band_b': b2,
                'p_raw': round(float(p), 6),
                'p_bonferroni': round(float(p_corr), 6),
                'significant': p_corr < 0.05,
            })

    df_band = pd.DataFrame(band_rows)
    df_band.to_csv(run_dir / "band_test.csv", index=False)

    # Motor artifact flag
    temporal_imp = sum(mean_imp[i] for i, n in enumerate(agg_names)
                      if n.startswith('TP9_') or n.startswith('TP10_'))
    frontal_imp = sum(mean_imp[i] for i, n in enumerate(agg_names)
                     if n.startswith('AF7_') or n.startswith('AF8_'))
    motor_flag = bool(temporal_imp > frontal_imp)

    # ── SFEI Construction ──
    top2_idx = ranking[:2]
    top2_names = [agg_names[i] for i in top2_idx]
    sfei_formula = f"{top2_names[0]} / {top2_names[1]}"

    features_dir = run_dir / "features"
    splits_dir = run_dir / "splits"
    seeds = params.get('step07', {}).get('seeds', [0, 1, 7, 42, 99])

    sfei_recalls = []
    for pid in config.INCLUDED_PARTICIPANTS:
        fp = features_dir / f"P{pid}.pkl"
        sp_path = splits_dir / f"P{pid}_splits.pkl"
        if not fp.exists() or not sp_path.exists():
            continue
        with open(fp, 'rb') as f:
            fd = pickle.load(f)
        with open(sp_path, 'rb') as f:
            sd = pickle.load(f)

        X = fd['features_full']
        labels = fd['labels']
        wids = fd['window_ids']
        wid_map = {int(w): i for i, w in enumerate(wids)}

        num_col = X[:, top2_idx[0]]
        den_col = X[:, top2_idx[1]]
        sfei_vals = num_col / (den_col + 1e-8)

        pid_recalls = []
        for seed in seeds:
            for fi in sd['splits'].get(seed, []):
                tri = [wid_map[w] for w in fi['train_ids'] if w in wid_map]
                tei = [wid_map[w] for w in fi['test_ids'] if w in wid_map]
                if len(tri) < 4 or len(tei) < 4:
                    continue
                Xtr = sfei_vals[tri].reshape(-1, 1)
                ytr = labels[tri]
                Xte = sfei_vals[tei].reshape(-1, 1)
                yte = labels[tei]
                if len(np.unique(ytr)) < 2 or len(np.unique(yte)) < 2:
                    continue
                sc = StandardScaler()
                lr = LogisticRegression(class_weight='balanced', random_state=seed)
                lr.fit(sc.fit_transform(Xtr), ytr)
                pid_recalls.append(recall_score(yte, lr.predict(sc.transform(Xte)),
                                                pos_label=1, zero_division=0))
        if pid_recalls:
            sfei_recalls.append(np.mean(pid_recalls))

    sfei_mean = float(np.mean(sfei_recalls)) if sfei_recalls else 0.5

    sfei_result = {
        'formula': sfei_formula,
        'numerator': top2_names[0],
        'denominator': top2_names[1],
        'mean_recall': sfei_mean,
        'n_participants': len(sfei_recalls),
        'friedman_stat': float(friedman_stat),
        'friedman_p': float(friedman_p),
        'motor_artifact_flag': motor_flag,
        'temporal_importance': float(temporal_imp),
        'frontal_importance': float(frontal_imp),
    }
    with open(run_dir / "sfei_result.json", 'w') as f:
        json.dump(sfei_result, f, indent=2)

    print(f"  Top features: {top2_names[0]}, {top2_names[1]}")
    print(f"  SFEI formula: {sfei_formula}")
    print(f"  SFEI recall: {sfei_mean:.3f}")
    print(f"  Friedman: χ²={friedman_stat:.2f}, p={friedman_p:.4f}")
    if motor_flag:
        print(f"  ⚠ MOTOR ARTIFACT FLAG: temporal > frontal importance")
    print(f"  ✓ Feature importance + SFEI saved.")
