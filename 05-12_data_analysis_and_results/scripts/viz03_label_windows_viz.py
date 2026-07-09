#!/usr/bin/env python3
"""viz03 — Window labelling visualization. 4 panels.

Panel 1: Window extraction schematic showing SKIP/STAY regions, extracted
         overlapping windows in a staircase pattern, and actual A-keypress lines.
Panel 2: Class distribution per participant (stacked bar).
Panel 3: Inter-A-press interval histogram with merge threshold.
Panel 4: Burst vs normal skip windows per participant.
"""

import numpy as np, pickle, pandas as pd
from pathlib import Path
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import config

PALETTE = sns.color_palette("pastel")


def run(run_dir, params):
    viz_dir = run_dir / "viz"; viz_dir.mkdir(exist_ok=True)
    windows_dir = run_dir / "windows" / "primary"
    processed_dir = run_dir / "processed" / "nonotch"
    label_csv = windows_dir / "label_summary.csv"
    if not label_csv.exists(): return
    df_labels = pd.read_csv(label_csv)

    p = params.get('step03', {})
    half_window_s = p.get('half_window_s', 3.0)
    window_s = p.get('window_s', 3.0)
    stride_s = p.get('stride_s', 0.6)
    burst_thresh = p.get('burst_thresh_s', 3.0)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Step 03 — Window Labelling", fontsize=16, fontweight='bold')

    # ── Panel 1 — Window extraction schematic ──
    ax = axes[0, 0]
    rep_pid = config.INCLUDED_PARTICIPANTS[0]
    pkl = windows_dir / f"P{rep_pid}.pkl"
    proc_pkl = processed_dir / f"P{rep_pid}.pkl"

    if pkl.exists() and proc_pkl.exists():
        with open(pkl, 'rb') as f: wdata = pickle.load(f)
        with open(proc_pkl, 'rb') as f: pdata = pickle.load(f)

        wins = wdata['windows']
        a_press_times = wdata.get('a_press_times', [])
        dfs = pdata['dfs']
        fs = pdata['fs']

        if wins and dfs:
            t_min = min(w['start_time'] for w in wins)
            t_max = t_min + 70
            excerpt = [w for w in wins if w['start_time'] < t_max and w['end_time'] > t_min]

            # Draw background SKIP regions from first CSV
            df0 = dfs[0]
            ts_all = df0['lsl_timestamp'].values
            classes = df0['class'].values
            skip_mask = np.array([c == 'SKIP' for c in classes])
            skip_idx = np.where(skip_mask)[0]
            if len(skip_idx) > 0:
                breaks = np.where(np.diff(skip_idx) > 1)[0] + 1
                blocks = np.split(skip_idx, breaks)
                for block in blocks:
                    if len(block) < 2: continue
                    bs = ts_all[block[0]] - t_min
                    be = ts_all[block[-1]] - t_min
                    if be < 0 or bs > 70: continue
                    ax.axvspan(bs, be, alpha=0.08, color='red', zorder=0)

            # ── Draw windows as staircase pattern ──
            # Within each contiguous region, windows step up one notch per stride,
            # then reset for the next region. This makes the overlap visible.

            n_steps = max(int(round(window_s / stride_s)), 1)  # windows per full cycle
            bar_h = 0.04
            region_spread = n_steps * bar_h * 1.2  # total vertical spread for one region

            # Group windows by label and then by region (contiguous in time)
            for label_int, y_base, color in [(0, -0.3, PALETTE[3]), (1, 0.7, PALETTE[0])]:
                label_wins = sorted([w for w in excerpt if w['label'] == label_int],
                                    key=lambda w: w['start_time'])
                if not label_wins:
                    continue

                # Group into contiguous regions (windows from same region cluster together)
                regions = []
                current_region = [label_wins[0]]
                for w in label_wins[1:]:
                    # If this window starts within one stride of the previous, same region
                    if w['start_time'] - current_region[-1]['start_time'] <= stride_s * 1.1:
                        current_region.append(w)
                    else:
                        regions.append(current_region)
                        current_region = [w]
                regions.append(current_region)

                for region_wins in regions:
                    for i, w in enumerate(region_wins):
                        step = i % n_steps  # cycle through positions
                        y = y_base + step * bar_h * 1.3
                        ax.barh(y, w['end_time'] - w['start_time'],
                                left=w['start_time'] - t_min, height=bar_h,
                                color=color, alpha=0.65, edgecolor='white',
                                linewidth=0.3, zorder=2)

            # Draw actual A-keypress vertical lines
            drawn_label = False
            for t in a_press_times:
                t_rel = t - t_min
                if 0 <= t_rel <= 70:
                    ax.axvline(x=t_rel, color='crimson', linewidth=1.2,
                               alpha=0.85, zorder=3,
                               label='A-keypress' if not drawn_label else None)
                    drawn_label = True

            # Clean up axes
            ax.set_yticks([0, 1])
            ax.set_yticklabels(['SKIP', 'STAY'])
            ax.set_ylim(-0.5, 1.5)
            ax.set_xlabel('Time (s)')
            ax.set_xlim(-1, 71)
            ax.set_title(f'Window Extraction — P{rep_pid} (70s excerpt)\n'
                         f'Window={window_s}s, stride={stride_s}s, '
                         f'SKIP region=±{half_window_s}s')
            ax.legend(fontsize=8, loc='upper right')

    # ── Panel 2 — Class distribution per participant ──
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

    # ── Panel 3 — Inter-A-press interval distribution ──
    ax = axes[1, 0]
    all_intervals = []
    for pid in config.INCLUDED_PARTICIPANTS:
        pkl_path = windows_dir / f"P{pid}.pkl"
        if not pkl_path.exists(): continue
        with open(pkl_path, 'rb') as f: data = pickle.load(f)
        a_times = sorted(data.get('a_press_times', []))
        for i in range(1, len(a_times)):
            interval = a_times[i] - a_times[i-1]
            if interval < 120:
                all_intervals.append(interval)

    if all_intervals:
        ax.hist(all_intervals, bins=60, color=PALETTE[2], edgecolor='white', alpha=0.8)
        ax.axvline(2 * half_window_s, color='red', linestyle='--', linewidth=1.5,
                   label=f'{2*half_window_s}s merge threshold\n'
                         f'(A-presses closer → merged SKIP region)')
        n_merged = sum(1 for i in all_intervals if i < 2 * half_window_s)
        ylim = ax.get_ylim()
        ax.fill_betweenx([0, ylim[1] if ylim[1] > 0 else 10],
                         0, 2 * half_window_s, alpha=0.1, color='red')
        ax.set_xlabel('Inter-A-press Interval (s)')
        ax.set_ylabel('Count')
        ax.set_title(f'Inter-A-press Intervals ({n_merged} below merge threshold)')
        ax.legend(fontsize=7)

    # ── Panel 4 — Burst vs normal skip windows per participant ──
    ax = axes[1, 1]
    burst_data = []
    for pid in config.INCLUDED_PARTICIPANTS:
        pkl_path = windows_dir / f"P{pid}.pkl"
        if not pkl_path.exists(): continue
        with open(pkl_path, 'rb') as f: data = pickle.load(f)
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

    # --- Individual pics in viz03/ ---
    viz03_dir = viz_dir / "viz03"
    viz03_dir.mkdir(exist_ok=True)

    # Panel 1
    if pkl.exists() and proc_pkl.exists() and wins and dfs:
        fig_ind, ax_ind = plt.subplots(figsize=(10, 6))
        if len(skip_idx) > 0:
            for block in blocks:
                if len(block) < 2: continue
                bs = ts_all[block[0]] - t_min
                be = ts_all[block[-1]] - t_min
                if be < 0 or bs > 70: continue
                ax_ind.axvspan(bs, be, alpha=0.08, color='red', zorder=0)

        for label_int, y_base, color in [(0, -0.3, PALETTE[3]), (1, 0.7, PALETTE[0])]:
            label_wins = sorted([w for w in excerpt if w['label'] == label_int],
                                key=lambda w: w['start_time'])
            if not label_wins: continue
            regions = []
            current_region = [label_wins[0]]
            for w in label_wins[1:]:
                if w['start_time'] - current_region[-1]['start_time'] <= stride_s * 1.1:
                    current_region.append(w)
                else:
                    regions.append(current_region)
                    current_region = [w]
            regions.append(current_region)
            for region_wins in regions:
                for i, w in enumerate(region_wins):
                    step = i % n_steps
                    y = y_base + step * bar_h * 1.3
                    ax_ind.barh(y, w['end_time'] - w['start_time'],
                            left=w['start_time'] - t_min, height=bar_h,
                            color=color, alpha=0.65, edgecolor='white',
                            linewidth=0.3, zorder=2)
        drawn_label = False
        for t in a_press_times:
            t_rel = t - t_min
            if 0 <= t_rel <= 70:
                ax_ind.axvline(x=t_rel, color='crimson', linewidth=1.2,
                           alpha=0.85, zorder=3,
                           label='A-keypress' if not drawn_label else None)
                drawn_label = True
        ax_ind.set_yticks([0, 1])
        ax_ind.set_yticklabels(['SKIP', 'STAY'])
        ax_ind.set_ylim(-0.5, 1.5)
        ax_ind.set_xlabel('Time (s)')
        ax_ind.set_xlim(-1, 71)
        ax_ind.set_title(f'Window Extraction — P{rep_pid} (70s excerpt)\n'
                     f'Window={window_s}s, stride={stride_s}s, '
                     f'SKIP region=±{half_window_s}s')
        ax_ind.legend(fontsize=8, loc='upper right')
        plt.tight_layout()
        plt.savefig(viz03_dir / "viz03.1.png", dpi=200)
        plt.close()

    # Panel 2
    fig_ind, ax_ind = plt.subplots(figsize=(8, 6))
    ax_ind.bar(x, df_labels['n_stay'], color=PALETTE[0], label='STAY', alpha=0.8)
    ax_ind.bar(x, df_labels['n_skip'], bottom=df_labels['n_stay'],
           color=PALETTE[3], label='SKIP', alpha=0.8)
    ax_ind.set_xticks(x)
    ax_ind.set_xticklabels([f'P{p}' for p in pids], rotation=90, fontsize=6)
    ax_ind.set_ylabel('Window Count')
    ax_ind.set_title('Class Distribution (Raw)')
    ax_ind.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(viz03_dir / "viz03.2.png", dpi=200)
    plt.close()

    # Panel 3
    if all_intervals:
        fig_ind, ax_ind = plt.subplots(figsize=(8, 6))
        ax_ind.hist(all_intervals, bins=60, color=PALETTE[2], edgecolor='white', alpha=0.8)
        ax_ind.axvline(2 * half_window_s, color='red', linestyle='--', linewidth=1.5,
                   label=f'{2*half_window_s}s merge threshold\n'
                         f'(A-presses closer → merged SKIP region)')
        ylim = ax_ind.get_ylim()
        ax_ind.fill_betweenx([0, ylim[1] if ylim[1] > 0 else 10],
                         0, 2 * half_window_s, alpha=0.1, color='red')
        ax_ind.set_xlabel('Inter-A-press Interval (s)')
        ax_ind.set_ylabel('Count')
        ax_ind.set_title(f'Inter-A-press Intervals ({n_merged} below merge threshold)')
        ax_ind.legend(fontsize=7)
        plt.tight_layout()
        plt.savefig(viz03_dir / "viz03.3.png", dpi=200)
        plt.close()

    # Panel 4
    if burst_data:
        fig_ind, ax_ind = plt.subplots(figsize=(8, 6))
        ax_ind.bar(x, df_b['normal_skip'], color=PALETTE[4], label='Normal skip')
        ax_ind.bar(x, df_b['burst'], bottom=df_b['normal_skip'],
               color=PALETTE[3], label='Burst skip')
        ax_ind.set_xticks(x)
        ax_ind.set_xticklabels([f"P{r['pid']}" for _, r in df_b.iterrows()],
                           rotation=90, fontsize=6)
        ax_ind.set_ylabel('Count')
        ax_ind.set_title('Burst vs Normal Skips')
        ax_ind.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(viz03_dir / "viz03.4.png", dpi=200)
        plt.close()

    # ── Panel 3.3.2 and 3.3.3: Burst Group Sizes (Comparison of ±3s and ±5s) ──
    from matplotlib.ticker import MaxNLocator
    stats_data = {}
    for hw in [3.0, 5.0]:
        all_group_sizes = []
        pid_group_sizes = []
        for pid in config.INCLUDED_PARTICIPANTS:
            pkl_path = windows_dir / f"P{pid}.pkl"
            if not pkl_path.exists(): continue
            with open(pkl_path, 'rb') as f: data = pickle.load(f)
            a_times = sorted(data.get('a_press_times', []))
            
            groups = []
            current_group = []
            for t in a_times:
                if not current_group:
                    current_group.append(t)
                else:
                    if (t - current_group[-1]) <= 2 * hw:
                        current_group.append(t)
                    else:
                        groups.append(current_group)
                        current_group = [t]
            if current_group:
                groups.append(current_group)
                
            group_sizes = [len(g) for g in groups]
            all_group_sizes.extend(group_sizes)
            for gs in group_sizes:
                pid_group_sizes.append({'pid': f"P{pid}", 'group_size': gs})
                
        df_groups = pd.DataFrame(pid_group_sizes)
        
        if not df_groups.empty:
            # viz03.3.2.png - Count plot per participant
            fig_ind, ax_ind = plt.subplots(figsize=(14, 6))
            sns.countplot(data=df_groups, x='pid', hue='group_size', ax=ax_ind, palette="pastel")
            ax_ind.set_xlabel('Participant')
            ax_ind.set_ylabel('Count of Burst Groups')
            ax_ind.set_title(f'Distribution of Burst Group Sizes per Participant (±{hw}s)')
            ax_ind.set_xticklabels(ax_ind.get_xticklabels(), rotation=90, fontsize=8)
            ax_ind.yaxis.set_major_locator(MaxNLocator(integer=True))
            plt.tight_layout()
            plt.savefig(viz03_dir / f"viz03.3.2_{int(hw)}s.png", dpi=200)
            plt.close()
            
            # viz03.3.3.png - Count plot all participants
            fig_ind, ax_ind = plt.subplots(figsize=(8, 6))
            sns.countplot(x=all_group_sizes, ax=ax_ind, palette="pastel")
            ax_ind.set_xlabel('Burst Group Size (consecutive swipes)')
            ax_ind.set_ylabel('Total Count')
            ax_ind.set_title(f'Distribution of Burst Group Sizes Across All Participants (±{hw}s)')
            for container in ax_ind.containers:
                ax_ind.bar_label(container)
            ax_ind.yaxis.set_major_locator(MaxNLocator(integer=True))
            plt.tight_layout()
            plt.savefig(viz03_dir / f"viz03.3.3_{int(hw)}s.png", dpi=200)
            plt.close()

            # Calculate stats
            total_groups = len(all_group_sizes)
            total_n1 = all_group_sizes.count(1)
            overall_pct = (total_n1 / total_groups * 100) if total_groups > 0 else 0
            
            participant_pcts = []
            for pid in config.INCLUDED_PARTICIPANTS:
                pid_gs = [d['group_size'] for d in pid_group_sizes if d['pid'] == f"P{pid}"]
                if pid_gs:
                    pct = (pid_gs.count(1) / len(pid_gs)) * 100
                    participant_pcts.append(pct)
            
            mean_pct = np.mean(participant_pcts) if participant_pcts else 0
            std_pct = np.std(participant_pcts) if participant_pcts else 0
            
            stats_data[hw] = {
                'overall_pct': overall_pct,
                'mean_pct': mean_pct,
                'std_pct': std_pct
            }

    # Write stats to tex file
    if 3.0 in stats_data and 5.0 in stats_data:
        tex_path = viz03_dir / "burst_stats.tex"
        with open(tex_path, "w") as f:
            f.write(f"Specifically, under the \\(\\pm3.0\\)s window paradigm, {stats_data[3.0]['overall_pct']:.1f}\\% of all swipe events are isolated \\(n=1\\) sequences (participant-level average: {stats_data[3.0]['mean_pct']:.1f}\\% \\(\\pm\\) {stats_data[3.0]['std_pct']:.1f}\\%). ")
            f.write(f"In contrast, widening the window to \\(\\pm5.0\\)s reduces the overall proportion of isolated swipes to {stats_data[5.0]['overall_pct']:.1f}\\% (participant-level average: {stats_data[5.0]['mean_pct']:.1f}\\% \\(\\pm\\) {stats_data[5.0]['std_pct']:.1f}\\%).\n")