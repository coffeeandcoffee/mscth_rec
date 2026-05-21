#!/usr/bin/env python3
"""viz04 — Balance & split visualization. 5 panels."""

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
    if not splits_dir.exists(): return

    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(2, 6)
    ax1 = fig.add_subplot(gs[0, 0:2])
    ax2 = fig.add_subplot(gs[0, 2:4])
    ax3 = fig.add_subplot(gs[0, 4:6])
    ax4 = fig.add_subplot(gs[1, 0:3])
    ax5 = fig.add_subplot(gs[1, 3:6])
    
    fig.suptitle("Step 04 — Balance & Temporal Split", fontsize=16, fontweight='bold')
    gap_s = params.get('step04', {}).get('gap_s', 3.0)
    n_folds = params.get('step04', {}).get('n_folds', 5)
    seeds = params.get('step04', {}).get('seeds', [0, 1, 7, 42, 99])
    ref_seed = seeds[0]

    # --- Panel 1 — Temporal fold structure (one participant) ---
    rep_pid = config.INCLUDED_PARTICIPANTS[0]
    sp = splits_dir / f"P{rep_pid}_splits.pkl"
    win_pkl = run_dir / "windows" / "primary" / f"P{rep_pid}.pkl"
    if sp.exists() and win_pkl.exists():
        with open(win_pkl, 'rb') as f: wd = pickle.load(f)
        with open(sp, 'rb') as f: sd = pickle.load(f)
        windows = wd['windows']
        ref_folds = sd['splits'].get(ref_seed, [])
        if windows and ref_folds:
            t_min = min(w['start_time'] for w in windows)
            for k, fold in enumerate(ref_folds):
                fold_ids = fold['test_ids']
                if not fold_ids:
                    continue
                fold_starts = [windows[i]['start_time'] - t_min for i in fold_ids]
                fold_ends = [windows[i]['end_time'] - t_min for i in fold_ids]
                b_start = min(fold_starts)
                b_end = max(fold_ends)
                color = PALETTE[k % len(PALETTE)]
                ax1.barh(0, b_end - b_start, left=b_start, height=0.5,
                        color=color, alpha=0.7, label=f'Fold {k+1}')
            ax1.set_yticks([]); ax1.set_xlabel('Time (s)')
            ax1.set_title(f'Temporal Fold Structure — P{rep_pid}'); ax1.legend(fontsize=7)

    # --- Panel 2 — Balanced class counts (actual per-class from splits) ---
    pids_list, stay_counts, skip_counts = [], [], []
    for pid in config.INCLUDED_PARTICIPANTS:
        sp_path = splits_dir / f"P{pid}_splits.pkl"
        if not sp_path.exists(): continue
        with open(sp_path, 'rb') as f: sd = pickle.load(f)
        folds = sd['splits'].get(ref_seed, [])
        pids_list.append(pid)
        stay_counts.append(sum(f['test_n_stay'] for f in folds))
        skip_counts.append(sum(f['test_n_skip'] for f in folds))
    x = range(len(pids_list))
    if x:
        ax2.bar(x, stay_counts, color=PALETTE[0], label='STAY (balanced)', alpha=0.8)
        ax2.bar(x, skip_counts, bottom=stay_counts, color=PALETTE[3], label='SKIP (balanced)', alpha=0.8)
        ax2.set_xticks(x); ax2.set_xticklabels([f'P{p}' for p in pids_list], rotation=90, fontsize=6)
        ax2.set_ylabel('Windows'); ax2.set_title('Balanced Counts'); ax2.legend(fontsize=8)

    # --- Panel 3 — Seed variance in balancing ---
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
            ax3.scatter([i] * len(sub), sub['n_test'], color=PALETTE[0], alpha=0.6, s=15)
            ax3.hlines(sub['n_test'].mean(), i - 0.3, i + 0.3, colors='gray', linewidth=0.5)
        ax3.set_xticks(range(len(df_seed['pid'].unique())))
        ax3.set_xticklabels([f'P{p}' for p in sorted(df_seed['pid'].unique())],
                           rotation=90, fontsize=6)
        ax3.set_ylabel('Test Windows'); ax3.set_title('Seed Variance in Balancing')

    # --- Panel 4 — Temporal Firewall Validation ---
    # Visual proof of the 3s gap
    ax4.axvline(0, color='red', linestyle='-', alpha=0.8, linewidth=2, label='Edge of Train')
    ax4.axvline(gap_s, color='red', linestyle='--', alpha=0.8, linewidth=2, label=f'Minimum Gap ({gap_s}s)')
    
    y_ticks_p4 = []
    y_labels_p4 = []
    for idx, pid in enumerate(config.INCLUDED_PARTICIPANTS):
        sp_path = splits_dir / f"P{pid}_splits.pkl"
        win_pkl = run_dir / "windows" / "primary" / f"P{pid}.pkl"
        if not sp_path.exists() or not win_pkl.exists(): continue
        with open(win_pkl, 'rb') as f: wd = pickle.load(f)
        with open(sp_path, 'rb') as f: sd = pickle.load(f)
        windows = wd['windows']
        folds = sd['splits'].get(ref_seed, [])
        
        # Calculate min distances for all folds
        for fold in folds:
            train_ids = fold['train_ids']
            test_ids = fold['test_ids']
            if not train_ids or not test_ids: continue
            
            # Efficient min distance computation
            train_times = np.sort(np.array([windows[i]['start_time'] for i in train_ids]))
            test_times = np.sort(np.array([windows[i]['start_time'] for i in test_ids]))
            
            min_dist = float('inf')
            i, j = 0, 0
            while i < len(train_times) and j < len(test_times):
                dist = abs(train_times[i] - test_times[j])
                if dist < min_dist:
                    min_dist = dist
                if train_times[i] < test_times[j]: i += 1
                else: j += 1
                
            # Plot the pair (0, min_dist)
            # Add slight jitter to y to separate folds visually
            jitter = (fold['fold'] - n_folds/2) * 0.1
            ax4.scatter(0, idx + jitter, color='black', s=10)
            ax4.scatter(min_dist, idx + jitter, color=PALETTE[3], s=20)
            ax4.plot([0, min_dist], [idx+jitter, idx+jitter], color='gray', alpha=0.3, linewidth=1)
            
        y_ticks_p4.append(idx)
        y_labels_p4.append(f"P{pid}")
            
    ax4.set_yticks(y_ticks_p4)
    ax4.set_yticklabels(y_labels_p4, fontsize=6)
    ax4.set_xlabel('Temporal Distance (s)')
    ax4.set_title('Temporal Firewall Validation')
    ax4.legend(fontsize=8)

    # --- Panel 5 — Dynamic Split visualization ---
    # Shows where the stratified cuts are made based on SKIP density
    y_ticks_p5 = []
    y_labels_p5 = []
    for idx, pid in enumerate(config.INCLUDED_PARTICIPANTS):
        win_pkl = run_dir / "windows" / "primary" / f"P{pid}.pkl"
        if not win_pkl.exists(): continue
        with open(win_pkl, 'rb') as f: wd = pickle.load(f)
        windows = wd['windows']
        if not windows: continue
        
        t_min = min(w['start_time'] for w in windows)
        skip_times = [w['start_time'] - t_min for w in windows if w['label'] == 0]
        
        # Plot SKIP phases
        if skip_times:
            ax5.scatter(skip_times, [idx] * len(skip_times), color=PALETTE[3], s=2, alpha=0.5, marker='|')
            
        # Calculate splits natively as done in step04
        splits = []
        if len(skip_times) >= n_folds:
            for k in range(1, n_folds):
                splits.append(skip_times[int(len(skip_times) * k / n_folds)])
        else:
            t_max = max(w['start_time'] for w in windows) - t_min
            splits = [t_max * k / n_folds for k in range(1, n_folds)]
            
        # Draw split markers
        for s in splits:
            ax5.axvline(s, ymin=(idx-0.4)/len(config.INCLUDED_PARTICIPANTS), 
                        ymax=(idx+0.4)/len(config.INCLUDED_PARTICIPANTS), 
                        color='black', linewidth=1)
            
        y_ticks_p5.append(idx)
        y_labels_p5.append(f"P{pid}")
        
    ax5.set_yticks(y_ticks_p5)
    ax5.set_yticklabels(y_labels_p5, fontsize=6)
    ax5.set_xlabel('Time (s)')
    ax5.set_title('Dynamic Split Allocation (SKIP density driven)')

    plt.tight_layout()
    plt.savefig(viz_dir / "viz04_balance_split.png", dpi=200); plt.close()

if __name__ == "__main__":
    print("Use run.py to execute the pipeline.")
