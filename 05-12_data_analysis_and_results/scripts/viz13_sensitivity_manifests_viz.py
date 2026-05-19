#!/usr/bin/env python3
"""viz13 — Sensitivity manifests visualization. One panel per comparison."""

import numpy as np, pandas as pd, json
from pathlib import Path
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import config

PALETTE = sns.color_palette("pastel")

def run(run_dir, params):
    viz_dir = run_dir / "viz"; viz_dir.mkdir(exist_ok=True)
    sens_csv = run_dir / "sensitivity_comparisons.csv"
    if not sens_csv.exists(): return
    df = pd.read_csv(sens_csv)
    if len(df) == 0: return

    n = len(df)
    fig, axes = plt.subplots(1, max(n, 1), figsize=(6 * max(n, 1), 6))
    if n == 1: axes = [axes]
    fig.suptitle("Step 13 — Sensitivity Analyses", fontsize=16, fontweight='bold')

    primary_json = run_dir / "step07_primary_result.json"
    if not primary_json.exists(): return
    with open(primary_json) as f: primary = json.load(f)
    recall_map = {r['pid']: r['recall'] for r in primary['per_participant']}

    for idx, (_, row) in enumerate(df.iterrows()):
        ax = axes[idx]
        mean_p = row['mean_primary']
        mean_v = row['mean_variant']

        # Bar comparison
        bars = ax.bar([0, 1], [mean_p, mean_v],
                      color=[PALETTE[0], PALETTE[2]], alpha=0.8, width=0.5)
        ax.axhline(0.5, color='gray', linestyle='--', linewidth=0.5)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Primary', row['comparison']], fontsize=8, rotation=15)
        ax.set_ylabel('Mean STAY Recall')

        sig = "SIG" if row['significant'] else "n.s."
        ax.set_title(f"{row['description']}\nΔ={row['delta']:+.3f}, "
                     f"p={row['p_value']:.3f} ({sig})", fontsize=9)

        # Annotate delta arrow
        ax.annotate('', xy=(1, mean_v), xytext=(0, mean_p),
                    arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))

    plt.tight_layout()
    plt.savefig(viz_dir / "viz13_sensitivity.png", dpi=200); plt.close()
