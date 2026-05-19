#!/usr/bin/env python3
"""viz02 — Artifact flagging visualization. 3 panels.

Plots the SAME notched + bandpassed signal the detector saw, with threshold
lines at ±thresh_uv/2 so the visual span between lines equals the ptp threshold.
Thresholds are read from params — nothing is hardcoded.

Note: artifact flags live in the NONOTCH pkl (where downstream reads them),
but were computed from the NOTCH signal. We read flag positions from nonotch
and the displayed signal from notch — both have identical row alignment.
"""

import numpy as np, pickle, pandas as pd
from pathlib import Path
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import config

PALETTE = sns.color_palette("pastel")
WINDOW_S = 120  # 2 minutes centered on artifact


def _plot_artifact_panel(ax, df_flags, df_signal, fs, art_col, channel,
                         low_hz, high_hz, thresh_uv, color, label):
    """Plot bandpassed notched signal around the median flagged sample.

    df_flags: nonotch df with artifact_* columns (for finding event index)
    df_signal: notch df with raw channel (for displaying)
    """
    if art_col not in df_flags.columns:
        ax.set_title(f'{label} — column missing')
        return

    art_idx = df_flags[df_flags[art_col]].index
    if len(art_idx) == 0:
        ax.set_title(f'{label} — no events flagged')
        return

    center = art_idx[len(art_idx) // 2]
    win = int(fs * WINDOW_S / 2)
    s = max(0, center - win)
    e = min(len(df_signal), center + win)

    # Bandpass the FULL notched signal first then slice — avoids edge artifacts
    full_filtered = config.extract_band_amplitude(
        df_signal[channel].values.astype(float), fs, low_hz, high_hz)
    filtered = full_filtered[s:e]

    t = (df_signal['lsl_timestamp'].values[s:e]
         - df_signal['lsl_timestamp'].values[s]) / 60.0  # minutes

    ax.plot(t, filtered, color=color, linewidth=0.5)
    ax.axhline(y=thresh_uv / 2, color='red', linestyle='--', linewidth=0.8,
               label=f'±{thresh_uv/2:.0f}µV (ptp = {thresh_uv}µV)')
    ax.axhline(y=-thresh_uv / 2, color='red', linestyle='--', linewidth=0.8)
    ax.set_xlabel('Time (min)')
    ax.set_ylabel('µV')
    ax.legend(fontsize=8, loc='upper right')


def run(run_dir, params):
    viz_dir = run_dir / "viz"
    viz_dir.mkdir(exist_ok=True)
    art_csv = run_dir / "artifact_summary.csv"
    if not art_csv.exists():
        return
    df = pd.read_csv(art_csv)
    nonotch_dir = run_dir / "processed" / "nonotch"
    notch_dir = run_dir / "processed" / "notch"

    p = params.get('step02', {})
    blink_thresh = p.get('blink_thresh_uv', 800)
    emg_thresh = p.get('emg_thresh_uv', 500)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Step 02 — Artifact Flagging (detection on notched signal)",
                 fontsize=16, fontweight='bold')

    # Panel 1 — Artifact rate per participant
    ax = axes[0]
    df_sorted = df.sort_values('any_rate', ascending=True)
    y_pos = range(len(df_sorted))
    ax.barh(y_pos, df_sorted['blink_rate'], color=PALETTE[0],
            label='Blink', height=0.4, align='edge')
    ax.barh([y + 0.4 for y in y_pos], df_sorted['emg_rate'], color=PALETTE[1],
            label='EMG', height=0.4, align='edge')
    ax.set_yticks([y + 0.4 for y in y_pos])
    ax.set_yticklabels([f"P{p_}" for p_ in df_sorted['pid']], fontsize=7)
    ax.set_xlabel('Artifact Rate')
    ax.set_title('Artifact Rate per Participant')
    ax.legend(fontsize=8)

    # Panel 2 — Example blink epoch (AF7, notched + 1-40Hz bandpassed)
    ax = axes[1]
    rep_pid = int(df.iloc[df['blink_rate'].argmax()]['pid'])
    nn_pkl = nonotch_dir / f"P{rep_pid}.pkl"
    n_pkl = notch_dir / f"P{rep_pid}.pkl"
    if nn_pkl.exists() and n_pkl.exists():
        with open(nn_pkl, 'rb') as f:
            nn_data = pickle.load(f)
        with open(n_pkl, 'rb') as f:
            n_data = pickle.load(f)
        df_flags = pd.concat(nn_data['dfs'], ignore_index=True)
        df_signal = pd.concat(n_data['dfs'], ignore_index=True)
        fs = n_data['fs']
        _plot_artifact_panel(ax, df_flags, df_signal, fs, 'artifact_blink',
                             channel='AF7', low_hz=1.0, high_hz=40.0,
                             thresh_uv=blink_thresh, color=PALETTE[0],
                             label=f'Blink — P{rep_pid}')
        ax.set_title(f'Example Blink — P{rep_pid} (AF7, notched + 1–40Hz)')

    # Panel 3 — Example EMG burst (TP9, notched + 30-100Hz bandpassed)
    ax = axes[2]
    rep_pid2 = int(df.iloc[df['emg_rate'].argmax()]['pid'])
    nn_pkl2 = nonotch_dir / f"P{rep_pid2}.pkl"
    n_pkl2 = notch_dir / f"P{rep_pid2}.pkl"
    if nn_pkl2.exists() and n_pkl2.exists():
        with open(nn_pkl2, 'rb') as f:
            nn_data2 = pickle.load(f)
        with open(n_pkl2, 'rb') as f:
            n_data2 = pickle.load(f)
        df_flags2 = pd.concat(nn_data2['dfs'], ignore_index=True)
        df_signal2 = pd.concat(n_data2['dfs'], ignore_index=True)
        fs2 = n_data2['fs']
        _plot_artifact_panel(ax, df_flags2, df_signal2, fs2, 'artifact_emg',
                             channel='TP9', low_hz=30.0, high_hz=100.0,
                             thresh_uv=emg_thresh, color=PALETTE[1],
                             label=f'EMG — P{rep_pid2}')
        ax.set_title(f'Example EMG Burst — P{rep_pid2} (TP9, notched + 30–100Hz)')

    plt.tight_layout()
    plt.savefig(viz_dir / "viz02_artifact_flag.png", dpi=200)
    plt.close()