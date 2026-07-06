#!/usr/bin/env python3
"""
viz01_preprocess_viz.py — Preprocessing visualizations.

Panel 1: Sampling rate distribution
Panel 2: Band power time series (one representative participant)
Panel 3: Bluetooth dropout heatmap
Panel 4: Notch vs nonotch high-gamma violin
"""

import numpy as np
import pickle
import pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import config


PALETTE = sns.color_palette("pastel", n_colors=7)
BAND_COLORS = {b[0]: PALETTE[i] for i, b in enumerate(config.FREQUENCY_BANDS)}


def run(run_dir, params):
    viz_dir = run_dir / "viz"
    viz_dir.mkdir(parents=True, exist_ok=True)

    nonotch_dir = run_dir / "processed" / "nonotch"
    notch_dir = run_dir / "processed" / "notch"

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    title_str = "Step 01 — Preprocessing Overview"
    if params.get('experimental', config.DEFAULT_PARAMS.get('experimental', {})).get('use_hilbert_envelope', False):
        title_str += "\n[X2: Hilbert Envelope: ON]"
    fig.suptitle(title_str, fontsize=16, fontweight='bold')

    # Collect data across participants
    fs_values = []
    dropout_data = {}
    rep_pid = None
    rep_data = None

    for pid in config.INCLUDED_PARTICIPANTS:
        pkl = nonotch_dir / f"P{pid}.pkl"
        if not pkl.exists():
            continue
        with open(pkl, 'rb') as f:
            data = pickle.load(f)
        fs_values.append(data['fs'])
        dropout_data[pid] = data.get('dropouts', [])
        if rep_pid is None:
            rep_pid = pid
            rep_data = data

    # Panel 1 — Sampling rate distribution
    ax = axes[0, 0]
    ax.hist(fs_values, bins=20, color=PALETTE[0], edgecolor='white', alpha=0.8)
    ax.axvline(256, color='red', linestyle='--', linewidth=1, label='256 Hz nominal')
    ax.set_xlabel('Effective Sampling Rate (Hz)')
    ax.set_ylabel('Count')
    ax.set_title('Sampling Rate Distribution')
    ax.legend()

    # Panel 2 — Band power time series (representative participant)
    ax = axes[0, 1]
    if rep_data is not None:
        df = pd.concat(rep_data['dfs'], ignore_index=True)
        ts = df['lsl_timestamp'].values
        t_rel = ts - ts[0]
        # Downsample for plotting
        step = max(1, len(t_rel) // 2000)
        for i, (bname, _, _) in enumerate(config.FREQUENCY_BANDS):
            fname = f"AF7_{bname}"
            if fname in df.columns:
                ax.plot(t_rel[::step], df[fname].values[::step],
                        color=PALETTE[i], alpha=0.7, linewidth=0.5, label=bname)
        # Shade SKIP regions
        skip_mask = (df['class'].values == 'SKIP')
        if skip_mask.any():
            in_skip = False
            for j in range(0, len(skip_mask), step):
                if skip_mask[j] and not in_skip:
                    skip_start = t_rel[j]
                    in_skip = True
                elif not skip_mask[j] and in_skip:
                    ax.axvspan(skip_start, t_rel[j], alpha=0.08, color='red')
                    in_skip = False
            if in_skip:
                ax.axvspan(skip_start, t_rel[-1], alpha=0.08, color='red')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Z-score Normalised Power')
        ax.set_title(f'Band Power — P{rep_pid} (AF7)')
        ax.legend(fontsize=7, ncol=4)

    # Panel 3 — Dropout heatmap
    ax = axes[1, 0]
    pids = sorted(dropout_data.keys())
    max_dur = max(
        (max((d['end_time'] for d in drops), default=0) -
         min((d['start_time'] for d in drops), default=0))
        if drops else 0
        for drops in dropout_data.values()
    )
    if max_dur < 1:
        max_dur = 1
    n_bins = 100
    heatmap = np.zeros((len(pids), n_bins))
    for row, pid in enumerate(pids):
        drops = dropout_data[pid]
        if not drops:
            continue
        t0 = min(d['start_time'] for d in drops) if drops else 0
        for d in drops:
            start_bin = int(((d['start_time'] - t0) / max_dur) * n_bins)
            end_bin = int(((d['end_time'] - t0) / max_dur) * n_bins)
            start_bin = max(0, min(start_bin, n_bins - 1))
            end_bin = max(0, min(end_bin, n_bins - 1))
            heatmap[row, start_bin:end_bin + 1] = 1

    ax.imshow(heatmap, aspect='auto', cmap='YlOrRd', interpolation='nearest')
    ax.set_yticks(range(len(pids)))
    ax.set_yticklabels([f'P{p}' for p in pids], fontsize=6)
    ax.set_xlabel('Session Time (normalised)')
    ax.set_title('Bluetooth Dropout Map')

    # Panel 4 — Notch vs nonotch high-gamma
    ax = axes[1, 1]
    hg_nonotch = []
    hg_notch = []
    for pid in config.INCLUDED_PARTICIPANTS:
        nn_pkl = nonotch_dir / f"P{pid}.pkl"
        n_pkl = notch_dir / f"P{pid}.pkl"
        if not nn_pkl.exists() or not n_pkl.exists():
            continue
        with open(nn_pkl, 'rb') as f:
            nn = pickle.load(f)
        with open(n_pkl, 'rb') as f:
            nt = pickle.load(f)
        nn_df = pd.concat(nn['dfs'], ignore_index=True)
        nt_df = pd.concat(nt['dfs'], ignore_index=True)
        if 'AF7_high_gamma' in nn_df.columns:
            hg_nonotch.append(nn_df['AF7_high_gamma'].median())
            hg_notch.append(nt_df['AF7_high_gamma'].median())

    if hg_nonotch:
        positions = [1, 2]
        parts1 = ax.violinplot([hg_nonotch], positions=[1], showmeans=True)
        parts2 = ax.violinplot([hg_notch], positions=[2], showmeans=True)
        for pc in parts1['bodies']:
            pc.set_facecolor(PALETTE[0])
        for pc in parts2['bodies']:
            pc.set_facecolor(PALETTE[1])
        ax.set_xticks([1, 2])
        ax.set_xticklabels(['Nonotch', 'Notch'])
        ax.set_ylabel('Median High-Gamma Relative Power')
        ax.set_title('Notch vs Nonotch — High Gamma (AF7)')

    plt.tight_layout()
    plt.savefig(viz_dir / "viz01_preprocess.png", dpi=200)
    plt.close()

    # --- Individual Panel 1 (viz01.1) ---
    fig_ind, ax_ind = plt.subplots(figsize=(8, 6))
    ax_ind.hist(fs_values, bins=20, color=PALETTE[0], edgecolor='white', alpha=0.8)
    ax_ind.axvline(256, color='red', linestyle='--', linewidth=1, label='256 Hz nominal')
    ax_ind.set_xlabel('Effective Sampling Rate (Hz)')
    ax_ind.set_ylabel('Count')
    ax_ind.set_title('Sampling Rate Distribution')
    ax_ind.legend()
    plt.tight_layout()
    plt.savefig(viz_dir / "viz01.1.png", dpi=200)
    plt.close()

    # --- Individual Panel 3 (viz01.3) ---
    fig_ind, ax_ind = plt.subplots(figsize=(8, 6))
    ax_ind.imshow(heatmap, aspect='auto', cmap='YlOrRd', interpolation='nearest')
    ax_ind.set_yticks(range(len(pids)))
    ax_ind.set_yticklabels([f'P{p}' for p in pids], fontsize=6)
    ax_ind.set_xlabel('Session Time (normalised)')
    ax_ind.set_title('Bluetooth Dropout Map')
    plt.tight_layout()
    plt.savefig(viz_dir / "viz01.3.png", dpi=200)
    plt.close()
