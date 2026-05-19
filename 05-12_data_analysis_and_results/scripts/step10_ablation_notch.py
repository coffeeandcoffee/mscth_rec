#!/usr/bin/env python3
"""
step10_ablation_notch.py — Notch filter ablation comparison.

Reads nonotch features from /features/ and notch features from /features_notch/.
Runs RF with temporal blocked 5-fold CV on both. Paired Wilcoxon across participants.

OUT: notch_ablation.csv + notch_ablation_result.json
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


def train_eval(X, labels, wids, splits, bp, seeds):
    wid_map = {int(w): i for i, w in enumerate(wids)}
    sr = []
    for seed in seeds:
        folds = splits.get(seed, [])
        fr = []
        for fi in folds:
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
            fr.append(recall_score(yte, clf.predict(Xte), pos_label=1, zero_division=0))
        if fr:
            sr.append(np.mean(fr))
    return float(np.mean(sr)) if sr else 0.5


def run(run_dir, params):
    config.pprint_step(10, "NOTCH ABLATION")
    seeds = params.get('step07', {}).get('seeds', [0, 1, 7, 42, 99])
    rows = []
    for pid in config.INCLUDED_PARTICIPANTS:
        nn_p = run_dir / "features" / f"P{pid}.pkl"
        n_p = run_dir / "features_notch" / f"P{pid}.pkl"
        s_p = run_dir / "splits" / f"P{pid}_splits.pkl"
        bp_p = run_dir / "grid_search" / f"P{pid}_best_params.json"
        if not all(p.exists() for p in [nn_p, n_p, s_p]):
            continue
        with open(nn_p, 'rb') as f: nn = pickle.load(f)
        with open(n_p, 'rb') as f: nt = pickle.load(f)
        with open(s_p, 'rb') as f: sp = pickle.load(f)
        bp = json.load(open(bp_p)) if bp_p.exists() else {}
        r_nn = train_eval(nn['features_full'], nn['labels'], nn['window_ids'], sp['splits'], bp, seeds)
        r_nt = train_eval(nt['features_full'], nt['labels'], nt['window_ids'], sp['splits'], bp, seeds)
        rows.append({'pid': pid, 'recall_nonotch': round(r_nn, 4), 'recall_notch': round(r_nt, 4), 'delta': round(r_nn - r_nt, 4)})
        print(f"  P{pid}: nonotch={r_nn:.3f}, notch={r_nt:.3f}, d={r_nn-r_nt:+.3f}")
    df = pd.DataFrame(rows)
    nn_arr = df['recall_nonotch'].values
    nt_arr = df['recall_notch'].values
    try:
        w, p = sp_stats.wilcoxon(nn_arr - nt_arr)
        r = 1 - (2*w)/(len(nn_arr)*(len(nn_arr)+1)/2)
    except Exception:
        w, p, r = 0.0, 1.0, 0.0
    df.to_csv(run_dir / "notch_ablation.csv", index=False)
    json.dump({'W': float(w), 'p': float(p), 'r': float(r), 'mean_nn': float(np.mean(nn_arr)), 'mean_nt': float(np.mean(nt_arr))},
              open(run_dir / "notch_ablation_result.json", 'w'), indent=2)
    print(f"\n  Notch ablation: W={w:.1f}, p={p:.4f}, r={r:.3f}")
    print(f"  ✓ Saved.")
