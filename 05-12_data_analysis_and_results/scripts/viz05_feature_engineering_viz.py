#!/usr/bin/env python3
"""viz05 — Feature engineering visualization. 3 panels."""

import numpy as np, pickle
from pathlib import Path
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import config

PALETTE = sns.color_palette("pastel", 7)

def run(run_dir, params):
    viz_dir = run_dir / "viz"; viz_dir.mkdir(exist_ok=True)
    features_dir = run_dir / "features"

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle("Step 05 — Feature Engineering", fontsize=16, fontweight='bold')

    # Panel 1 — Feature construction diagram (static schematic)
    ax = axes[0]
    channels = config.EEG_CHANNELS
    bands = [b[0] for b in config.FREQUENCY_BANDS]
    grid = np.ones((len(channels), len(bands)))
    im = ax.imshow(grid, cmap='Blues', alpha=0.3, aspect='auto')
    # Color frontal vs temporal
    for i, ch in enumerate(channels):
        for j in range(len(bands)):
            color = PALETTE[0] if ch in config.FRONTAL_CHANNELS else PALETTE[1]
            ax.add_patch(plt.Rectangle((j-0.5, i-0.5), 1, 1, fill=True,
                                        facecolor=color, alpha=0.3, edgecolor='gray'))
            ax.text(j, i, '4', ha='center', va='center', fontsize=8, color='gray')
    ax.set_xticks(range(len(bands))); ax.set_xticklabels(bands, fontsize=7, rotation=45)
    ax.set_yticks(range(len(channels))); ax.set_yticklabels(channels, fontsize=9)
    ax.set_title(f'Feature Space: {config.N_FEATURES_FULL} features\n(4 stats × 7 bands × 4 ch)')

    # Panel 2 — Feature correlation matrix (one representative participant)
    ax = axes[1]
    rep_pid = config.INCLUDED_PARTICIPANTS[0]
    fp = features_dir / f"P{rep_pid}.pkl"
    if fp.exists():
        with open(fp, 'rb') as f: fd = pickle.load(f)
        X = fd['features_full']
        if X.shape[0] > 5:
            corr = np.corrcoef(X.T)
            corr = np.nan_to_num(corr)
            im = ax.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
            ax.set_title(f'Feature Correlation — P{rep_pid}')
            plt.colorbar(im, ax=ax, shrink=0.7)
        else:
            ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center', transform=ax.transAxes)
    ax.set_xlabel('Feature Index'); ax.set_ylabel('Feature Index')

    # Panel 3 — EI scalar distribution: STAY vs SKIP
    ax = axes[2]
    ei_stay_all, ei_skip_all = [], []
    for pid in config.INCLUDED_PARTICIPANTS:
        fp = features_dir / f"P{pid}.pkl"
        if not fp.exists(): continue
        with open(fp, 'rb') as f: fd = pickle.load(f)
        ei = fd['ei_values']; labels = fd['labels']
        ei_stay_all.extend(ei[labels == 1].tolist())
        ei_skip_all.extend(ei[labels == 0].tolist())
    if ei_stay_all and ei_skip_all:
        parts = ax.violinplot([ei_stay_all, ei_skip_all], positions=[1, 2], showmeans=True)
        for i, pc in enumerate(parts['bodies']):
            pc.set_facecolor(PALETTE[0] if i == 0 else PALETTE[3])
            pc.set_alpha(0.7)
        ax.set_xticks([1, 2]); ax.set_xticklabels(['STAY', 'SKIP'])
        ax.set_ylabel('EI Value'); ax.set_title('EI Distribution: STAY vs SKIP')

    plt.tight_layout()
    plt.savefig(viz_dir / "viz05_feature_engineering.png", dpi=200); plt.close()
