#!/usr/bin/env python3
"""viz11 — Feature importance visualization. 4 panels."""

import numpy as np, pandas as pd, json
from pathlib import Path
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import config

PALETTE = sns.color_palette("pastel", 7)
BAND_COLORS = {b[0]: PALETTE[i] for i, b in enumerate(config.FREQUENCY_BANDS)}

def run(run_dir, params):
    viz_dir = run_dir / "viz"; viz_dir.mkdir(exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Step 11 — Feature Importance & SFEI", fontsize=16, fontweight='bold')

    # Panel 1 — Top 20 features
    ax = axes[0, 0]
    rank_csv = run_dir / "feature_ranking_112.csv"
    if rank_csv.exists():
        df = pd.read_csv(rank_csv).head(20)
        colors = []
        for fname in df['feature']:
            found = False
            for bname in BAND_COLORS:
                if f"_{bname}_" in fname:
                    colors.append(BAND_COLORS[bname]); found = True; break
            if not found: colors.append('gray')
        ax.barh(range(len(df)-1, -1, -1), df['mean_importance'], color=colors,
                xerr=df['std_importance'], capsize=2)
        ax.set_yticks(range(len(df)-1, -1, -1))
        ax.set_yticklabels(df['feature'], fontsize=6)
        ax.set_xlabel('Mean Gini Importance'); ax.set_title('Top 20 Features')

    # Panel 2 — 4×7 electrode-band heatmap
    ax = axes[0, 1]
    hm_json = run_dir / "electrode_band_heatmap.json"
    if hm_json.exists():
        with open(hm_json) as f: hm_data = json.load(f)
        channels = config.EEG_CHANNELS
        bands = [b[0] for b in config.FREQUENCY_BANDS]
        hm = np.zeros((len(channels), len(bands)))
        for i, ch in enumerate(channels):
            for j, bn in enumerate(bands):
                hm[i, j] = hm_data.get(f"{ch}_{bn}", 0)
        im = ax.imshow(hm, cmap='YlOrRd', aspect='auto')
        ax.set_xticks(range(len(bands))); ax.set_xticklabels(bands, fontsize=7, rotation=45)
        ax.set_yticks(range(len(channels))); ax.set_yticklabels(channels)
        ax.set_title('Electrode × Band Importance'); plt.colorbar(im, ax=ax, shrink=0.7)

    # Panel 3 — Band-level importance boxplots
    ax = axes[1, 0]
    band_csv = run_dir / "band_test.csv"
    sfei_json = run_dir / "sfei_result.json"
    if sfei_json.exists():
        with open(sfei_json) as f: sfei = json.load(f)
        ax.text(0.5, 0.9, f"Friedman χ²={sfei.get('friedman_stat', 0):.2f}, "
                f"p={sfei.get('friedman_p', 1):.4f}", transform=ax.transAxes,
                ha='center', fontsize=9)
    # Simple bar chart of band importance from heatmap
    if hm_json.exists():
        band_totals = {}
        for bn in bands:
            band_totals[bn] = sum(hm_data.get(f"{ch}_{bn}", 0) for ch in channels)
        ax.bar(range(len(bands)), [band_totals[b] for b in bands],
               color=[PALETTE[i] for i in range(len(bands))], alpha=0.8)
        ax.set_xticks(range(len(bands))); ax.set_xticklabels(bands, fontsize=8, rotation=45)
        ax.set_ylabel('Aggregate Importance'); ax.set_title('Band-Level Importance')

    # Panel 4 — SFEI formula card
    ax = axes[1, 1]
    ax.axis('off')
    if sfei_json.exists():
        with open(sfei_json) as f: sfei = json.load(f)
        text = (f"SFEI Formula:\n\n"
                f"  {sfei.get('formula', '?')}\n\n"
                f"SFEI Recall: {sfei.get('mean_recall', 0):.3f}\n"
                f"Motor Artifact Flag: {'⚠ YES' if sfei.get('motor_artifact_flag') else '✓ NO'}\n"
                f"Frontal importance: {sfei.get('frontal_importance', 0):.4f}\n"
                f"Temporal importance: {sfei.get('temporal_importance', 0):.4f}")
        ax.text(0.1, 0.5, text, transform=ax.transAxes, fontsize=11,
                family='monospace', verticalalignment='center',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout()
    plt.savefig(viz_dir / "viz11_feature_importance.png", dpi=200); plt.close()
