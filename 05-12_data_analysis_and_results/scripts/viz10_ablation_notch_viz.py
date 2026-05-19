#!/usr/bin/env python3
"""viz10 — Notch ablation visualization. 2 panels."""

import numpy as np, pandas as pd, json
from pathlib import Path
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import config

PALETTE = sns.color_palette("pastel")

def run(run_dir, params):
    viz_dir = run_dir / "viz"; viz_dir.mkdir(exist_ok=True)
    notch_csv = run_dir / "notch_ablation.csv"
    if not notch_csv.exists(): return
    df = pd.read_csv(notch_csv)
    df = df[df['pid'] != 'WILCOXON']

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Step 10 — Notch Ablation", fontsize=16, fontweight='bold')

    # Panel 1 — Paired recall
    ax = axes[0]
    nn = df['recall_nonotch'].values.astype(float)
    nt = df['recall_notch'].values.astype(float)
    bp1 = ax.boxplot([nn], positions=[1], widths=0.35, patch_artist=True)
    bp2 = ax.boxplot([nt], positions=[2], widths=0.35, patch_artist=True)
    bp1['boxes'][0].set_facecolor(PALETTE[0]); bp2['boxes'][0].set_facecolor(PALETTE[1])
    for i in range(len(nn)):
        ax.plot([1, 2], [nn[i], nt[i]], color='gray', alpha=0.3, linewidth=0.5)
    ax.set_xticks([1, 2]); ax.set_xticklabels(['Nonotch', 'Notch'])
    ax.axhline(0.5, color='gray', linestyle='--', linewidth=0.5)
    ax.set_ylabel('STAY Recall'); ax.set_title('Nonotch vs Notch Recall')
    res = json.load(open(run_dir / "notch_ablation_result.json")) if (run_dir / "notch_ablation_result.json").exists() else {}
    if res:
        ax.text(0.5, 0.02, f"p={res.get('p', 1):.4f}, r={res.get('r', 0):.3f}",
                transform=ax.transAxes, ha='center', fontsize=9)

    # Panel 2 — Delta distribution
    ax = axes[1]
    deltas = df['delta'].values.astype(float)
    ax.hist(deltas, bins=15, color=PALETTE[2], edgecolor='white', alpha=0.8)
    ax.axvline(0, color='gray', linestyle='--', linewidth=0.8)
    ax.axvline(np.mean(deltas), color='blue', linestyle=':', label=f'Mean Δ={np.mean(deltas):.3f}')
    ax.set_xlabel('Δ Recall (nonotch − notch)'); ax.set_ylabel('Count')
    ax.set_title('Per-Participant Recall Difference'); ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(viz_dir / "viz10_ablation_notch.png", dpi=200); plt.close()
