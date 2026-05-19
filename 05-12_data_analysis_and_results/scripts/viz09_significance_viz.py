#!/usr/bin/env python3
"""viz09 — Significance testing visualization. 4 panels."""

import numpy as np, pandas as pd, json
from pathlib import Path
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import config

PALETTE = sns.color_palette("pastel")

def run(run_dir, params):
    viz_dir = run_dir / "viz"; viz_dir.mkdir(exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Step 09 — Significance Testing", fontsize=16, fontweight='bold')

    # Panel 1 — Binomial CI forest plot
    ax = axes[0, 0]
    sig_csv = run_dir / "per_participant_significance.csv"
    if sig_csv.exists():
        df = pd.read_csv(sig_csv).sort_values('recall', ascending=True)
        for i, (_, row) in enumerate(df.iterrows()):
            color = PALETTE[0] if row['significantly_above_chance'] else PALETTE[4]
            ax.errorbar(row['recall'], i, xerr=[[row['recall'] - row['ci_low']],
                        [row['ci_high'] - row['recall']]], fmt='o', color=color,
                        markersize=5, capsize=3)
        ax.axvline(0.5, color='gray', linestyle='--', linewidth=0.8)
        ax.set_yticks(range(len(df)))
        ax.set_yticklabels([f"P{int(r['pid'])}" for _, r in df.iterrows()], fontsize=6)
        ax.set_xlabel('STAY Recall'); ax.set_title('Binomial CI Forest Plot')

    # Panel 2 — Seed stability strip
    ax = axes[0, 1]
    stab_csv = run_dir / "seed_stability.csv"
    if stab_csv.exists():
        df_s = pd.read_csv(stab_csv).sort_values('seed_sd')
        colors = [PALETTE[0] if r['stable'] else PALETTE[3] for _, r in df_s.iterrows()]
        ax.barh(range(len(df_s)), df_s['seed_sd'], color=colors)
        ax.axvline(0.03, color='red', linestyle='--', linewidth=0.8, label='SD=0.03 threshold')
        ax.set_yticks(range(len(df_s)))
        ax.set_yticklabels([f"P{int(r['pid'])}" for _, r in df_s.iterrows()], fontsize=6)
        ax.set_xlabel('SD across 5 seeds'); ax.set_title('Seed Stability'); ax.legend(fontsize=8)

    # Panel 3 — Electrode ablation pairwise
    ax = axes[1, 0]
    abl_csv = run_dir / "electrode_ablation.csv"
    if abl_csv.exists():
        df_a = pd.read_csv(abl_csv)
        df_a = df_a[df_a['comparison'] != 'MOTOR_ARTIFACT_FLAG']
        if len(df_a) > 0:
            x = range(len(df_a))
            ax.bar(x, df_a['delta'], color=[PALETTE[0] if s else PALETTE[3]
                   for s in df_a['significant']], alpha=0.8)
            ax.set_xticks(x)
            ax.set_xticklabels(df_a['comparison'], fontsize=7, rotation=30, ha='right')
            ax.axhline(0, color='gray', linewidth=0.5)
            ax.set_ylabel('Δ Recall'); ax.set_title('Electrode Ablation Deltas')
            for i, (_, r) in enumerate(df_a.iterrows()):
                ax.text(i, r['delta'], f"p={r['p_value']:.3f}", ha='center', va='bottom', fontsize=7)

    # Panel 4 — EI vs RF (from stats)
    ax = axes[1, 1]
    stats_path = run_dir / "stats_summary.json"
    if stats_path.exists():
        with open(stats_path) as f: stats = json.load(f)
        ei_comp = stats.get('ei_vs_rf', {})
        if ei_comp:
            vals = [ei_comp.get('mean_rf', 0.5), ei_comp.get('mean_ei', 0.5)]
            bars = ax.bar([0, 1], vals, color=[PALETTE[0], PALETTE[1]], alpha=0.8)
            ax.set_xticks([0, 1]); ax.set_xticklabels(['RF', 'EI'])
            ax.axhline(0.5, color='gray', linestyle='--', linewidth=0.5)
            ax.set_ylabel('Mean STAY Recall')
            p_val = ei_comp.get('p_value', 1)
            ax.set_title(f'EI vs RF (p={p_val:.4f})')
        else:
            ax.text(0.5, 0.5, 'No EI data', ha='center', transform=ax.transAxes)

    plt.tight_layout()
    plt.savefig(viz_dir / "viz09_significance.png", dpi=200); plt.close()
