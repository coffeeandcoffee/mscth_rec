#!/usr/bin/env python3
"""viz05 — Feature engineering visualization. 4 panels."""

import numpy as np, pickle
from pathlib import Path
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import config

PALETTE = sns.color_palette("pastel", 7)

def run(run_dir, params):
    viz_dir = run_dir / "viz"; viz_dir.mkdir(exist_ok=True)
    features_dir = run_dir / "features"

    fig = plt.figure(figsize=(24, 6))
    fig.suptitle("Step 05 — Feature Engineering", fontsize=16, fontweight='bold')

    # Panel 1 — Feature construction diagram (static schematic)
    ax = fig.add_subplot(1, 4, 1)
    channels = config.EEG_CHANNELS
    bands = [b[0] for b in config.FREQUENCY_BANDS]
    stats = params.get('step05', {}).get('stats', ['mean', 'std', 'min', 'max'])
    n_stats = len(stats)
    n_feats = len(channels) * len(bands) * n_stats
    grid = np.ones((len(channels), len(bands)))
    ax.imshow(grid, cmap='Blues', alpha=0.3, aspect='auto')
    for i, ch in enumerate(channels):
        for j in range(len(bands)):
            color = PALETTE[0] if ch in config.FRONTAL_CHANNELS else PALETTE[1]
            ax.add_patch(plt.Rectangle((j-0.5, i-0.5), 1, 1, fill=True,
                                        facecolor=color, alpha=0.3, edgecolor='gray'))
            ax.text(j, i, str(n_stats), ha='center', va='center', fontsize=8, color='gray')
    ax.set_xticks(range(len(bands))); ax.set_xticklabels(bands, fontsize=7, rotation=45)
    ax.set_yticks(range(len(channels))); ax.set_yticklabels(channels, fontsize=9)
    ax.set_title(f'Feature Space: {n_feats} features\n({n_stats} stats × {len(bands)} bands × {len(channels)} ch)')

    # Panel 2 — Feature correlation heatmap (one representative participant)
    ax = fig.add_subplot(1, 4, 2)
    rep_pid = config.INCLUDED_PARTICIPANTS[0]
    fp = features_dir / f"P{rep_pid}.pkl"
    corr = None
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

    # Panel 3 — 3D correlation topography surface
    ax3d = fig.add_subplot(1, 4, 3, projection='3d')
    if corr is not None:
        n = corr.shape[0]
        # Downsample for visual clarity if large
        step = max(1, n // 40)
        corr_ds = corr[::step, ::step]
        n_ds = corr_ds.shape[0]
        X_grid, Y_grid = np.meshgrid(np.arange(n_ds), np.arange(n_ds))

        surf = ax3d.plot_surface(X_grid, Y_grid, corr_ds,
                                 cmap='RdBu_r', vmin=-1, vmax=1,
                                 alpha=0.9, edgecolor='none',
                                 antialiased=True, rstride=1, cstride=1)
        ax3d.set_zlim(-1, 1)
        ax3d.set_xlabel('Feature', fontsize=7, labelpad=1)
        ax3d.set_ylabel('Feature', fontsize=7, labelpad=1)
        ax3d.set_zlabel('r', fontsize=8, labelpad=1)
        ax3d.set_title('Correlation Topography', fontsize=10, pad=8)
        ax3d.view_init(elev=35, azim=-60)
        ax3d.tick_params(axis='both', which='major', labelsize=5, pad=0)
        ax3d.tick_params(axis='z', which='major', labelsize=6)
    else:
        ax3d.text2D(0.5, 0.5, 'No data', ha='center', va='center',
                    transform=ax3d.transAxes)

    # Panel 4 — EI scalar distribution: STAY vs SKIP
    ax = fig.add_subplot(1, 4, 4)
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
