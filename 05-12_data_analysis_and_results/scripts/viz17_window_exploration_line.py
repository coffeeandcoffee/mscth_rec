#!/usr/bin/env python3
import json
import numpy as np
from pathlib import Path
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import config
import viz_style

def run(run_dir, params):
    if not getattr(config, 'ENABLE_WINDOW_EXPLORATION', False):
        return
        
    exploration_dir = run_dir / "exploration"
    if not exploration_dir.exists():
        return
        
    viz_dir = run_dir / "viz"
    viz_dir.mkdir(parents=True, exist_ok=True)
    
    total_areas = [2.0, 3.0, 4.0, 6.0] # Sorted for x-axis
    gaps = [3.0, 2.0, 1.0, 0.5]
    window_sizes = [0.3, 0.7, 1.0, 2.0]
    
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
        
    # Stacked vertically so each panel spans the full figure width.
    fig, axes = plt.subplots(2, 1, figsize=(16, 18),
                             gridspec_kw={'hspace': 0.45})
    fig.suptitle("Window Exploration: Maximizing Feature Separation", fontweight='bold')
    viz_style.style_suptitle(fig, viz_style.FONT_3X)
    
    # ---- Graph 1: Gap vs Skip-Area (Window fixed to 1.0s) ----
    ax1 = axes[0]
    palette1 = sns.color_palette("Set1", n_colors=len(total_areas))
    
    for a_idx, area in enumerate(total_areas):
        x_gaps = []
        y_vals = []
        for gap in gaps:
            univ_name = f"univ_a{area}_g{gap}_w1.0"
            s, v = get_best(univ_name)
            if (exploration_dir / univ_name).exists():
                x_gaps.append(gap)
                y_vals.append(v)
                
        if len(x_gaps) > 0:
            ax1.plot(x_gaps, y_vals, marker='o', label=f'Skip-Area={area}s', color=palette1[a_idx], linewidth=2, markersize=8)
            
    ax1.set_xticks(gaps)
    ax1.grid(True, linestyle='--', alpha=0.7)
    # All four series rise to the top right, so no corner is free at the
    # default limits; add headroom and place the legend above the data.
    ax1.margins(y=0.30)
    ax1.legend(title="Total Skip-Area", loc="upper left", ncol=2)
    viz_style.style_axes(
        ax1, viz_style.FONT_3X,
        title="Sweep: GAP and SKIP-AREA (Fixed Window=1.0s)",
        xlabel="GAP (s)",
        ylabel="Max Separation (|d|)",
    )
    
    # ---- Graph 2: Window Length (Area fixed to 3.0s, Gap fixed to 2.0s) ----
    ax2 = axes[1]
    
    x_ws = []
    y_vals_ws = []
    for ws in window_sizes:
        univ_name = f"univ_a3.0_g2.0_w{ws}"
        s, v = get_best(univ_name)
        if (exploration_dir / univ_name).exists():
            x_ws.append(ws)
            y_vals_ws.append(v)
            
    if len(x_ws) > 0:
        ax2.plot(x_ws, y_vals_ws, marker='o', color='purple', linewidth=2, markersize=8, label="Gap=2.0s, Area=3.0s")
        
    ax2.set_xticks(window_sizes)
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend(loc="upper left")
    viz_style.style_axes(
        ax2, viz_style.FONT_3X,
        title="Sweep: Window Duration",
        xlabel="Window Length (s)",
        ylabel="Max Separation (|d|)",
    )
            
    plt.savefig(viz_dir / "viz17_window_exploration_line.png", dpi=200, bbox_inches='tight')
    plt.close()
    
if __name__ == "__main__":
    print("Use run.py to execute.")
