#!/usr/bin/env python3
"""viz03 — Window labelling visualization. 4 panels."""

import numpy as np, pickle, pandas as pd
from pathlib import Path
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import config

PALETTE = sns.color_palette("pastel")

def run(run_dir, params):
    viz_dir = run_dir / "viz"; viz_dir.mkdir(exist_ok=True)
    windows_dir = run_dir / "windows" / "primary"
    label_csv = windows_dir / "label_summary.csv"
    if not label_csv.exists(): return
    df_labels = pd.read_csv(label_csv)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Step 03 — Window Labelling", fontsize=16, fontweight='bold')

    # Panel 1 — Window extraction schematic (one participant, 60s excerpt)
    # A-keypress positions derived from SKIP window end_times (end_time = A-press moment)
    ax = axes[0, 0]
    rep_pid = config.INCLUDED_PARTICIPANTS[0]
    pkl = windows_dir / f"P{rep_pid}.pkl"
    if pkl.exists():
        with open(pkl, 'rb') as f: data = pickle.load(f)
        wins = data['windows']
        if wins:
            t_min = min(w['start_time'] for w in wins)
            excerpt = [w for w in wins if w['start_time'] < t_min + 60]

            # Draw windows as horizontal bars
            for w in excerpt:
                color = PALETTE[0] if w['label'] == 1 else PALETTE[3]
                alpha = 0.6 if w['label'] == 1 else 0.8
                ax.barh(w['label'], w['end_time'] - w['start_time'],
                        left=w['start_time'] - t_min, height=0.3,
                        color=color, alpha=alpha)

            # Draw A-keypress vertical lines at SKIP window midpoints (= keypress time)
            skip_in_excerpt = [w for w in excerpt if w['label'] == 0]
            for i, w in enumerate(skip_in_excerpt):
                t_rel = (w['start_time'] + w['end_time']) / 2 - t_min
                ax.axvline(x=t_rel, color='crimson', linewidth=1.0,
                           alpha=0.85,
                           label='A-keypress' if i == 0 else None)

            ax.set_yticks([0, 1])
            ax.set_yticklabels(['SKIP', 'STAY'])
            ax.set_xlabel('Time (s)')
            ax.set_title(f'Window Extraction — P{rep_pid} (60s excerpt)')
            ax.legend(fontsize=8, loc='upper right')

    # Panel 2 — Class distribution per participant
    ax = axes[0, 1]
    pids = df_labels['pid'].values
    x = range(len(pids))
    ax.bar(x, df_labels['n_stay'], color=PALETTE[0], label='STAY', alpha=0.8)
    ax.bar(x, df_labels['n_skip'], bottom=df_labels['n_stay'],
           color=PALETTE[3], label='SKIP', alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f'P{p}' for p in pids], rotation=90, fontsize=6)
    ax.set_ylabel('Window Count')
    ax.set_title('Class Distribution (Raw)')
    ax.legend(fontsize=8)

    # Panel 3 — Inter-skip interval distribution
    ax = axes[1, 0]
    all_intervals = []
    for pid in config.INCLUDED_PARTICIPANTS:
        pkl = windows_dir / f"P{pid}.pkl"
        if not pkl.exists(): continue
        with open(pkl, 'rb') as f: data = pickle.load(f)
        skip_wins = sorted([w for w in data['windows'] if w['label'] == 0],
                           key=lambda w: w['end_time'])
        for i in range(1, len(skip_wins)):
            interval = skip_wins[i]['end_time'] - skip_wins[i-1]['end_time']
            if interval < 60:
                all_intervals.append(interval)
    if all_intervals:
        burst_thresh = params.get('step03', {}).get('burst_thresh_s', 3.0)
        ax.hist(all_intervals, bins=50, color=PALETTE[2], edgecolor='white', alpha=0.8)
        ax.axvline(burst_thresh, color='red', linestyle='--',
                   label=f'{burst_thresh}s burst threshold')
        n_burst = sum(1 for i in all_intervals if i < burst_thresh)
        ylim = ax.get_ylim()
        ax.fill_betweenx([0, ylim[1] if ylim[1] > 0 else 10],
                         0, burst_thresh, alpha=0.1, color='red')
        ax.set_xlabel('Inter-Skip Interval (s)')
        ax.set_ylabel('Count')
        ax.set_title(f'Inter-Skip Intervals (burst region: {n_burst})')
        ax.legend(fontsize=8)

    # Panel 4 — Burst-skip map
    ax = axes[1, 1]
    burst_data = []
    for pid in config.INCLUDED_PARTICIPANTS:
        pkl = windows_dir / f"P{pid}.pkl"
        if not pkl.exists(): continue
        with open(pkl, 'rb') as f: data = pickle.load(f)
        n_burst = sum(1 for w in data['windows'] if w.get('is_burst_skip', False))
        n_total = sum(1 for w in data['windows'] if w['label'] == 0)
        burst_data.append({'pid': pid, 'burst': n_burst,
                           'normal_skip': n_total - n_burst})
    if burst_data:
        df_b = pd.DataFrame(burst_data)
        x = range(len(df_b))
        ax.bar(x, df_b['normal_skip'], color=PALETTE[4], label='Normal skip')
        ax.bar(x, df_b['burst'], bottom=df_b['normal_skip'],
               color=PALETTE[3], label='Burst skip')
        ax.set_xticks(x)
        ax.set_xticklabels([f"P{r['pid']}" for _, r in df_b.iterrows()],
                           rotation=90, fontsize=6)
        ax.set_ylabel('Count')
        ax.set_title('Burst vs Normal Skips')
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(viz_dir / "viz03_label_windows.png", dpi=200)
    plt.close()