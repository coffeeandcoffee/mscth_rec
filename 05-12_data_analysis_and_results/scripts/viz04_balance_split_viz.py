#!/usr/bin/env python3
"""viz04 — Balance & split visualization. 3 panels."""

import numpy as np, pickle, pandas as pd
from pathlib import Path
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import config

PALETTE = sns.color_palette("pastel")

def run(run_dir, params):
    viz_dir = run_dir / "viz"; viz_dir.mkdir(exist_ok=True)
    splits_dir = run_dir / "splits"
    bc_csv = run_dir / "balanced_counts.csv"
    if not bc_csv.exists(): return
    df_bc = pd.read_csv(bc_csv)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Step 04 — Balance & Temporal Split", fontsize=16, fontweight='bold')

    # Panel 1 — Temporal fold structure (one participant)
    ax = axes[0]
    rep_pid = config.INCLUDED_PARTICIPANTS[0]
    sp = splits_dir / f"P{rep_pid}_splits.pkl"
    win_pkl = run_dir / "windows" / "primary" / f"P{rep_pid}.pkl"
    if sp.exists() and win_pkl.exists():
        with open(win_pkl, 'rb') as f: wd = pickle.load(f)
        windows = wd['windows']
        if windows:
            t_min = min(w['start_time'] for w in windows)
            t_max = max(w['end_time'] for w in windows)
            gap_s = params.get('step04', {}).get('gap_s', 3.0)
            n_folds = params.get('step04', {}).get('n_folds', 5)
            total_dur = t_max - t_min
            block_dur = (total_dur - gap_s * (n_folds - 1)) / n_folds
            for k in range(n_folds):
                b_start = k * (block_dur + gap_s)
                b_end = b_start + block_dur
                color = PALETTE[k % len(PALETTE)]
                label = f'Fold {k+1} (test)' if k == 0 else f'Fold {k+1}'
                ax.barh(0, block_dur, left=b_start, height=0.5, color=color, alpha=0.7, label=label)
                if k < n_folds - 1:
                    ax.barh(0, gap_s, left=b_end, height=0.5, color='white', edgecolor='gray',
                            linewidth=0.5, alpha=0.3)
            ax.set_yticks([]); ax.set_xlabel('Time (s)')
            ax.set_title(f'Temporal Fold Structure — P{rep_pid}'); ax.legend(fontsize=7)

    # Panel 2 — Balanced class counts
    ax = axes[1]
    pids = df_bc['pid'].values
    x = range(len(pids))
    half = df_bc['n_balanced_test_total'] / 2  # approx since 50/50
    ax.bar(x, half, color=PALETTE[0], label='STAY (balanced)', alpha=0.8)
    ax.bar(x, half, bottom=half, color=PALETTE[3], label='SKIP (balanced)', alpha=0.8)
    ax.set_xticks(x); ax.set_xticklabels([f'P{p}' for p in pids], rotation=90, fontsize=6)
    ax.set_ylabel('Windows'); ax.set_title('Balanced Counts'); ax.legend(fontsize=8)

    # Panel 3 — Seed variance in balancing
    ax = axes[2]
    seeds = params.get('step04', {}).get('seeds', [0, 1, 7, 42, 99])
    seed_data = []
    for pid in config.INCLUDED_PARTICIPANTS:
        sp_path = splits_dir / f"P{pid}_splits.pkl"
        if not sp_path.exists(): continue
        with open(sp_path, 'rb') as f: sd = pickle.load(f)
        for seed in seeds:
            folds = sd['splits'].get(seed, [])
            n_test = sum(len(f['test_ids']) for f in folds)
            seed_data.append({'pid': pid, 'seed': seed, 'n_test': n_test})
    if seed_data:
        df_seed = pd.DataFrame(seed_data)
        for i, pid in enumerate(sorted(df_seed['pid'].unique())):
            sub = df_seed[df_seed['pid'] == pid]
            ax.scatter([i] * len(sub), sub['n_test'], color=PALETTE[0], alpha=0.6, s=15)
            ax.hlines(sub['n_test'].mean(), i - 0.3, i + 0.3, colors='gray', linewidth=0.5)
        ax.set_xticks(range(len(df_seed['pid'].unique())))
        ax.set_xticklabels([f'P{p}' for p in sorted(df_seed['pid'].unique())],
                           rotation=90, fontsize=6)
        ax.set_ylabel('Test Windows'); ax.set_title('Seed Variance in Balancing')

    plt.tight_layout()
    plt.savefig(viz_dir / "viz04_balance_split.png", dpi=200); plt.close()
