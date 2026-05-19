#!/usr/bin/env python3
"""viz12 — Subgroup robustness visualization. 3 panels."""

import numpy as np, pandas as pd, json
from pathlib import Path
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import config

PALETTE = sns.color_palette("pastel")

def run(run_dir, params):
    viz_dir = run_dir / "viz"; viz_dir.mkdir(exist_ok=True)
    sub_csv = run_dir / "subgroup_comparisons.csv"
    if not sub_csv.exists(): return
    df = pd.read_csv(sub_csv)
    primary_json = run_dir / "step07_primary_result.json"
    if not primary_json.exists(): return
    with open(primary_json) as f: primary = json.load(f)
    recall_map = {r['pid']: r['recall'] for r in primary['per_participant']}

    n_panels = len(df)
    fig, axes = plt.subplots(1, max(n_panels, 1), figsize=(6 * max(n_panels, 1), 6))
    if n_panels == 1: axes = [axes]
    fig.suptitle("Step 12 — Subgroup Robustness", fontsize=16, fontweight='bold')

    for idx, (_, row) in enumerate(df.iterrows()):
        ax = axes[idx]
        # Split participants into groups
        a_name, b_name = row['group_a'], row['group_b']
        comp = row['comparison']

        # Get group membership
        if comp == 'paid_vs_unpaid':
            a_pids = [p for p in config.PAID_PARTICIPANTS if p in recall_map]
            b_pids = [p for p in config.UNPAID_PARTICIPANTS if p in recall_map]
        else:
            a_pids = list(recall_map.keys())[:int(row['n_a'])]
            b_pids = list(recall_map.keys())[int(row['n_a']):int(row['n_a'])+int(row['n_b'])]

        a_vals = [recall_map[p] for p in a_pids if p in recall_map]
        b_vals = [recall_map[p] for p in b_pids if p in recall_map]

        # Strip plot with box overlay
        for i, v in enumerate(a_vals):
            ax.scatter(0 + np.random.uniform(-0.1, 0.1), v, color=PALETTE[0], alpha=0.6, s=30)
        for i, v in enumerate(b_vals):
            ax.scatter(1 + np.random.uniform(-0.1, 0.1), v, color=PALETTE[1], alpha=0.6, s=30)

        if a_vals:
            ax.boxplot([a_vals], positions=[0], widths=0.4, patch_artist=True,
                      boxprops=dict(facecolor=PALETTE[0], alpha=0.3))
        if b_vals:
            ax.boxplot([b_vals], positions=[1], widths=0.4, patch_artist=True,
                      boxprops=dict(facecolor=PALETTE[1], alpha=0.3))

        ax.axhline(0.5, color='gray', linestyle='--', linewidth=0.5)
        ax.set_xticks([0, 1]); ax.set_xticklabels([a_name, b_name])
        ax.set_ylabel('STAY Recall')
        sig_text = f"p={row['p_value']:.3f}, r={row['effect_r']:.3f}"
        if not row['significant']:
            sig_text += " (n.s.)"
        ax.set_title(f"{comp}\n{sig_text}", fontsize=10)

    plt.tight_layout()
    plt.savefig(viz_dir / "viz12_subgroups.png", dpi=200); plt.close()
