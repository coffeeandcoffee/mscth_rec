#!/usr/bin/env python3
"""viz08 — EI comparison visualization. 2 panels."""

import numpy as np, pickle, json, pandas as pd
from pathlib import Path
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import config

PALETTE = sns.color_palette("pastel")

def run(run_dir, params):
    viz_dir = run_dir / "viz"; viz_dir.mkdir(exist_ok=True)
    ei_csv = run_dir / "ei_summary.csv"
    primary_json = run_dir / "step07_primary_result.json"
    if not ei_csv.exists() or not primary_json.exists(): return

    df_ei = pd.read_csv(ei_csv)
    with open(primary_json, 'r') as f: primary = json.load(f)
    rf_map = {r['pid']: r['recall'] for r in primary['per_participant']}

    # Match participants
    matched = []
    for _, row in df_ei.iterrows():
        pid = row['pid']
        if pid in rf_map:
            matched.append({'pid': pid, 'rf': rf_map[pid], 'ei': row['ei_recall']})

    if not matched: return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Step 08 — RF vs EI Comparison", fontsize=16, fontweight='bold')

    # Panel 1 — Paired comparison
    ax = axes[0]
    rf_v = [m['rf'] for m in matched]
    ei_v = [m['ei'] for m in matched]
    bp1 = ax.boxplot([rf_v], positions=[1], widths=0.35, patch_artist=True)
    bp2 = ax.boxplot([ei_v], positions=[2], widths=0.35, patch_artist=True)
    bp1['boxes'][0].set_facecolor(PALETTE[0]); bp2['boxes'][0].set_facecolor(PALETTE[1])
    for m in matched:
        ax.plot([1, 2], [m['rf'], m['ei']], color='gray', alpha=0.3, linewidth=0.5)
    ax.set_xticks([1, 2]); ax.set_xticklabels(['RF (112 features)', 'EI (scalar)'])
    ax.axhline(0.5, color='gray', linestyle='--', linewidth=0.5)
    ax.set_ylabel('STAY Recall'); ax.set_title('RF vs EI: Paired Recall')

    # Panel 2 — EI distributions by class
    ax = axes[1]
    features_dir = run_dir / "features"
    ei_stay, ei_skip = [], []
    for pid in config.INCLUDED_PARTICIPANTS:
        fp = features_dir / f"P{pid}.pkl"
        if not fp.exists(): continue
        with open(fp, 'rb') as f: fd = pickle.load(f)
        ei = fd['ei_values']; labels = fd['labels']
        ei_stay.extend(ei[labels == 1].tolist())
        ei_skip.extend(ei[labels == 0].tolist())
    if ei_stay and ei_skip:
        parts = ax.violinplot([ei_stay, ei_skip], positions=[1, 2], showmeans=True)
        for i, pc in enumerate(parts['bodies']):
            pc.set_facecolor(PALETTE[0] if i == 0 else PALETTE[3]); pc.set_alpha(0.7)
        ax.set_xticks([1, 2]); ax.set_xticklabels(['STAY', 'SKIP'])
        ax.set_ylabel('EI Value'); ax.set_title('EI Value Distributions')

    plt.tight_layout()
    plt.savefig(viz_dir / "viz08_train_ei.png", dpi=200); plt.close()
