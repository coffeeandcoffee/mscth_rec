#!/usr/bin/env python3
"""
step00_window_exploration.py — Window Parameter Exploration (Side Mission)

Sweeps through a grid of SKIP window sizes and gaps before SKIP, extracting
all features by dynamically creating isolated pipeline universes.
"""

import numpy as np
import pandas as pd
import shutil
from pathlib import Path
import copy
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

import config
import step01_preprocess
import step02_artifact_flag
import step03_label_windows
import step04_balance_split
import step05_feature_engineering


def group_by_stat_and_find_best(features_df):
    """Computes effect sizes for STAY vs SKIP and groups by STAT."""
    stay_df = features_df[features_df['Class'] == 'STAY']
    skip_df = features_df[features_df['Class'] == 'SKIP']
    
    if len(stay_df) == 0 or len(skip_df) == 0:
        return "", 0.0
        
    bands = [b[0] for b in config.FREQUENCY_BANDS]
    stat_means = {}
    
    # Identify feature columns (exclude PID, Window_ID, Class, Session, etc.)
    exclude_cols = ['PID', 'Window_ID', 'Class', 'Session', 'Target']
    feat_cols = [c for c in features_df.columns if c not in exclude_cols]
    
    # Compute global Cohen's d for each feature using the entire DataFrame
    mean_stay = stay_df[feat_cols].mean()
    mean_skip = skip_df[feat_cols].mean()
    var_stay = stay_df[feat_cols].var()
    var_skip = skip_df[feat_cols].var()
    
    pooled_std = np.sqrt((var_stay + var_skip) / 2)
    pooled_std = pooled_std.replace(0, 1e-9)
    
    d_vals = np.abs((mean_stay - mean_skip) / pooled_std)
    
    for c in feat_cols:
        st = None
        for band in bands:
            if f"_{band}_" in c:
                parts = c.split(f"_{band}_")
                if len(parts) == 2:
                    st = parts[1]
                break
        if not st and '_raw_' in c:
            st = 'Raw ' + c.split('_raw_')[1].capitalize()
            
        if not st: continue
        st = st.capitalize() if not st.startswith('Raw') else st
        
        if st not in stat_means:
            stat_means[st] = []
        stat_means[st].append(d_vals[c])
        
    best_stat = ""
    best_val = -1.0
    for st, vals in stat_means.items():
        m_val = np.mean(vals)
        if m_val > best_val:
            best_val = m_val
            best_stat = st
            
    return best_stat, best_val


def run_universe(base_params, area, gap, win_size, parent_run_dir):
    univ_name = f"univ_a{area}_g{gap}_w{win_size}"
    univ_dir = parent_run_dir / "exploration" / univ_name
    
    # If the universe already completed successfully, skip it
    features_pkl = univ_dir / "ml_data" / "features.pkl"
    if features_pkl.exists():
        df = pd.read_pickle(features_pkl)
        return group_by_stat_and_find_best(df)
        
    univ_dir.mkdir(parents=True, exist_ok=True)
    
    p = copy.deepcopy(base_params)
    p['step01']['pre_skip_window_s'] = float(area)
    p['step01']['gap_s'] = float(gap)
    p['step03']['window_size_s'] = float(win_size)
    
    print(f"\n  [UNIVERSE {univ_name}] Starting Pipeline")
    
    # Redirect prints to a log file to avoid spamming the console
    import sys
    original_stdout = sys.stdout
    class Unbuffered:
        def __init__(self, stream): self.stream = stream
        def write(self, data):
            self.stream.write(data)
            self.stream.flush()
        def __getattr__(self, attr): return getattr(self.stream, attr)
        
    with open(univ_dir / "pipeline.log", "w") as f_log:
        sys.stdout = Unbuffered(f_log)
        try:
            def check_done(step_num): return (univ_dir / f"step{step_num:02d}.done").exists()
            def mark_done(step_num): (univ_dir / f"step{step_num:02d}.done").touch()
            
            if not check_done(1):
                step01_preprocess.run(univ_dir, p)
                mark_done(1)
            if not check_done(2):
                step02_artifact_flag.run(univ_dir, p)
                mark_done(2)
            if not check_done(3):
                step03_label_windows.run(univ_dir, p)
                mark_done(3)
            if not check_done(4):
                step04_balance_split.run(univ_dir, p)
                mark_done(4)
            if not check_done(5):
                step05_feature_engineering.run(univ_dir, p)
                mark_done(5)
        except Exception as e:
            sys.stdout = original_stdout
            print(f"  [UNIVERSE {univ_name}] FAILED: {e}")
            return "", 0.0
        finally:
            sys.stdout = original_stdout
            
    # Cleanup large intermediate files to save disk space
    if (univ_dir / "processed").exists():
        shutil.rmtree(univ_dir / "processed")
    if (univ_dir / "windows").exists():
        shutil.rmtree(univ_dir / "windows")
        
    print(f"  [UNIVERSE {univ_name}] Done. Extracted features.")
    
    if features_pkl.exists():
        df = pd.read_pickle(features_pkl)
        return group_by_stat_and_find_best(df)
        
    return "", 0.0


def run(run_dir, params):
    if not getattr(config, 'ENABLE_WINDOW_EXPLORATION', False):
        print("  Skipping Window Exploration (ENABLE_WINDOW_EXPLORATION=False)")
        return
        
    config.pprint_step(0, "WINDOW PARAMETER EXPLORATION")
    print("  Creating multiple Universes using the full standard pipeline...\n")
    
    total_areas = [6.0, 4.0, 3.0, 2.0]
    gaps = [3.0, 2.0, 1.0, 0.5]
    
    matrix_4x4_vals = np.zeros((4, 4))
    matrix_4x4_stats = [["" for _ in range(4)] for _ in range(4)]
    
    print("  Evaluating 4x4 Matrix (Total Area vs Gap)...")
    for i, gap in enumerate(gaps):
        for j, area in enumerate(total_areas):
            best_stat, best_val = run_universe(params, area, gap, win_size=1.0, parent_run_dir=run_dir)
            matrix_4x4_vals[i, j] = best_val
            matrix_4x4_stats[i][j] = best_stat
            
    # 1x4 Matrix
    window_sizes = [0.3, 0.7, 1.0, 2.0]
    matrix_1x4_vals = np.zeros((1, 4))
    matrix_1x4_stats = [["" for _ in range(4)]]
    
    print("\n  Evaluating 1x4 Matrix (Window Size)...")
    for j, ws in enumerate(window_sizes):
        best_stat, best_val = run_universe(params, area=3.0, gap=2.0, win_size=ws, parent_run_dir=run_dir)
        matrix_1x4_vals[0, j] = best_val
        matrix_1x4_stats[0][j] = best_stat
        
    # Plotting
    viz_dir = run_dir / "viz"
    viz_dir.mkdir(parents=True, exist_ok=True)
    
    # Map each unique STAT to a Set2 color consistently
    all_unique_stats = set()
    for row in matrix_4x4_stats: all_unique_stats.update(row)
    for row in matrix_1x4_stats: all_unique_stats.update(row)
    all_unique_stats.discard("")
    unique_stats_sorted = sorted(list(all_unique_stats))
    
    set2_palette = sns.color_palette("Set2", n_colors=max(8, len(unique_stats_sorted)))
    stat_colors = {st: set2_palette[i % len(set2_palette)] for i, st in enumerate(unique_stats_sorted)}
    stat_colors[""] = (1.0, 1.0, 1.0) # White for empty
    
    def plot_categorical_matrix(ax, vals, stats_grid, xticklabels, yticklabels, title):
        rows, cols = vals.shape
        color_matrix = np.zeros((rows, cols, 3))
        for r in range(rows):
            for c in range(cols):
                color_matrix[r, c] = stat_colors[stats_grid[r][c]]
                
        ax.imshow(color_matrix, aspect='auto')
        for r in range(rows):
            for c in range(cols):
                txt = f"{stats_grid[r][c]}\n{vals[r, c]:.2f}"
                ax.text(c, r, txt, ha='center', va='center', color='black', fontsize=10, fontweight='bold')
                
        ax.set_xticks(range(cols))
        ax.set_yticks(range(rows))
        ax.set_xticklabels(xticklabels)
        ax.set_yticklabels(yticklabels)
        ax.set_title(title, pad=15)
        
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), gridspec_kw={'width_ratios': [2, 1]})
    
    plot_categorical_matrix(axes[0], matrix_4x4_vals, matrix_4x4_stats, 
                            [f"{x}s" for x in total_areas], [f"{y}s" for y in gaps], 
                            "Best STAT Separation (|d|)\nSweep: Skip Area vs Gap")
    axes[0].set_xlabel("Total Skip Area Length")
    axes[0].set_ylabel("Gap Before Skip")
    
    plot_categorical_matrix(axes[1], matrix_1x4_vals, matrix_1x4_stats, 
                            [f"{x}s" for x in window_sizes], ["3s Area\n2s Gap"], 
                            "Best STAT Separation (|d|)\nSweep: Window Size")
    axes[1].set_xlabel("Window Duration")
    
    # Add legend
    import matplotlib.patches as mpatches
    patches = [mpatches.Patch(color=stat_colors[st], label=st) for st in unique_stats_sorted]
    fig.legend(handles=patches, loc='center left', bbox_to_anchor=(0.95, 0.5), title="Top Stat")
    
    plt.tight_layout(rect=[0, 0, 0.95, 1])
    plt.savefig(viz_dir / "viz00_window_exploration.png", dpi=200, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    print("Use run.py to execute.")
