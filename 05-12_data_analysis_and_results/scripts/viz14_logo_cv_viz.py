#!/usr/bin/env python3
"""viz14 — LOGO-CV visualization. 2 panels."""

import numpy as np, pandas as pd, json
from pathlib import Path
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import config

PALETTE = sns.color_palette("pastel")

def run(run_dir, params):
    viz_dir = run_dir / "viz"; viz_dir.mkdir(exist_ok=True)
    logo_csv = run_dir / "logo_results.csv"
    logo_json = run_dir / "logo_confusion.json"
    if not logo_csv.exists(): return

    df = pd.read_csv(logo_csv).sort_values('recall', ascending=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Step 14 — LOGO-CV", fontsize=16, fontweight='bold')

    # Panel 1 — Per-fold LOGO recall
    ax = axes[0]
    for i, (_, row) in enumerate(df.iterrows()):
        color = PALETTE[0] if row['recall'] > 0.5 else PALETTE[4]
        ax.scatter(row['recall'], i, color=color, s=40, zorder=3)
    ax.axvline(0.5, color='gray', linestyle='--', linewidth=0.8, label='50% chance')
    mean_r = df['recall'].mean()
    ax.axvline(mean_r, color='blue', linestyle=':', linewidth=0.8,
               label=f'Mean={mean_r:.3f}')
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels([f"P{int(r['test_pid'])}" for _, r in df.iterrows()], fontsize=7)
    ax.set_xlabel('STAY Recall')
    ax.set_title('Per-Fold LOGO Recall')
    ax.legend(fontsize=8)
    ax.set_xlim(0, 1)

    # Panel 2 — Aggregate confusion matrix
    ax = axes[1]
    if logo_json.exists():
        with open(logo_json) as f: logo = json.load(f)
        cm = np.array(logo.get('aggregate_confusion_matrix', [[0,0],[0,0]]))
        total = cm.sum()
        if total > 0:
            cm_norm = cm / total
        else:
            cm_norm = cm.astype(float)

        im = ax.imshow(cm_norm, cmap='Blues', vmin=0, vmax=0.5)
        for i in range(2):
            for j in range(2):
                val = cm_norm[i, j]
                count = cm[i, j]
                ax.text(j, i, f'{val:.1%}\n(n={count})',
                        ha='center', va='center', fontsize=11,
                        color='white' if val > 0.3 else 'black')
        ax.set_xticks([0, 1]); ax.set_xticklabels(['Pred SKIP', 'Pred STAY'])
        ax.set_yticks([0, 1]); ax.set_yticklabels(['True SKIP', 'True STAY'])
        ax.set_title('Aggregate LOGO Confusion Matrix')
        plt.colorbar(im, ax=ax, shrink=0.7)

    plt.tight_layout()
    plt.savefig(viz_dir / "viz14_logo_cv.png", dpi=200); plt.close()
