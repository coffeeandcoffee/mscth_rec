#!/usr/bin/env python3
import json
import numpy as np
from pathlib import Path
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import config

def run(run_dir, params):
    if not getattr(config, 'ENABLE_WINDOW_EXPLORATION', False):
        return
        
    exploration_dir = run_dir / "exploration"
    if not exploration_dir.exists():
        return
        
    viz_dir = run_dir / "viz"
    viz_dir.mkdir(parents=True, exist_ok=True)
    
    total_areas = [6.0, 4.0, 3.0, 2.0]
    gaps = [3.0, 2.0, 1.0, 0.5]
    window_sizes = [0.3, 0.7, 1.0, 2.0]
    
    matrix_4x4_vals = np.zeros((4, 4))
    matrix_4x4_stats = [["" for _ in range(4)] for _ in range(4)]
    
    matrix_1x4_vals = np.zeros((1, 4))
    matrix_1x4_stats = [["" for _ in range(4)]]
    
    def get_best(univ_name):
        json_file = exploration_dir / univ_name / "features" / "best_features_ranking.json"
        if not json_file.exists(): return "", 0.0
        with open(json_file, 'r') as f:
            ranking = json.load(f)
        if not ranking: return "", 0.0
        
        from collections import defaultdict
        stat_max = defaultdict(float)
        for r in ranking:
            name_parts = r['name'].split('_')
            if len(name_parts) >= 3:
                stat = "_".join(name_parts[2:])
                if r['d'] > stat_max[stat]:
                    stat_max[stat] = r['d']
        if not stat_max: return "", 0.0
        best_stat = max(stat_max.items(), key=lambda x: x[1])
        return best_stat[0], best_stat[1]
        
    for i, gap in enumerate(gaps):
        for j, area in enumerate(total_areas):
            univ_name = f"univ_a{area}_g{gap}_w1.0"
            s, v = get_best(univ_name)
            matrix_4x4_stats[i][j] = s
            matrix_4x4_vals[i, j] = v
            
    for j, ws in enumerate(window_sizes):
        univ_name = f"univ_a3.0_g2.0_w{ws}"
        s, v = get_best(univ_name)
        matrix_1x4_stats[0][j] = s
        matrix_1x4_vals[0, j] = v
        
    all_unique_stats = set()
    for row in matrix_4x4_stats: all_unique_stats.update(row)
    for row in matrix_1x4_stats: all_unique_stats.update(row)
    all_unique_stats.discard("")
    unique_stats_sorted = sorted(list(all_unique_stats))
    
    set2_palette = sns.color_palette("Set2", n_colors=max(8, len(unique_stats_sorted)))
    stat_colors = {st: set2_palette[k % len(set2_palette)] for k, st in enumerate(unique_stats_sorted)}
    stat_colors[""] = (1.0, 1.0, 1.0)
    
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
    
    import matplotlib.patches as mpatches
    patches = [mpatches.Patch(color=stat_colors[st], label=st) for st in unique_stats_sorted]
    fig.legend(handles=patches, loc='center left', bbox_to_anchor=(0.95, 0.5), title="Top Stat")
    
    plt.tight_layout(rect=[0, 0, 0.95, 1])
    plt.savefig(viz_dir / "viz05b_window_exploration.png", dpi=200, bbox_inches='tight')
    plt.close()
    
if __name__ == "__main__":
    print("Use run.py to execute.")
