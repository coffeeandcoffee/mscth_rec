#!/usr/bin/env python3
import json
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

def run(run_dir, params):
    json_file = run_dir / "features" / "best_features_ranking.json"
    if not json_file.exists():
        print(f"  viz17_4: {json_file.name} not found.")
        return
        
    with open(json_file, 'r') as f:
        ranking = json.load(f)
        
    if not ranking:
        return
        
    # Parse full ranking
    parsed = []
    for r in ranking:
        name = r['name']
        d = abs(r['d'])
        parts = name.split('_')
        elec = parts[0]
        
        if 'high_gamma' in name:
            band = 'high_gamma'
            stat = name.split('high_gamma_')[1] if len(name.split('high_gamma_')) > 1 else 'unknown'
        elif 'low_gamma' in name:
            band = 'low_gamma'
            stat = name.split('low_gamma_')[1] if len(name.split('low_gamma_')) > 1 else 'unknown'
        elif 'very_high' in name:
            band = 'very_high'
            stat = name.split('very_high_')[1] if len(name.split('very_high_')) > 1 else 'unknown'
        elif 'raw' in name:
            band = 'raw'
            stat = name.split('raw_')[1] if len(name.split('raw_')) > 1 else 'unknown'
        else:
            band = parts[1] if len(parts) > 1 else 'unknown'
            stat = '_'.join(parts[2:]) if len(parts) > 2 else 'unknown'
            
        parsed.append({'Name': name, 'Electrode': elec, 'Band': band, 'Stat': stat, 'd': d})
        
    df = pd.DataFrame(parsed)
    
    # 1. Identify Top Parameters based on Mean d
    top_band = df.groupby('Band')['d'].mean().idxmax()
    top_elec = df.groupby('Electrode')['d'].mean().idxmax()
    top_stat = df.groupby('Stat')['d'].mean().idxmax()
    
    # 2. Find the strongest representative feature for each
    rep_band = df[df['Band'] == top_band].sort_values('d', ascending=False).iloc[0]
    rep_elec = df[df['Electrode'] == top_elec].sort_values('d', ascending=False).iloc[0]
    rep_stat = df[df['Stat'] == top_stat].sort_values('d', ascending=False).iloc[0]
    
    # 3. Overall Top 2 Features
    top_overall = df.sort_values('d', ascending=False).head(2)
    rep_top1 = top_overall.iloc[0]
    rep_top2 = top_overall.iloc[1]
    
    # Map them unique
    selected = {}
    
    def add_feature(rep, category_label):
        name = rep['Name']
        if name not in selected:
            selected[name] = {'d': rep['d'], 'labels': [category_label]}
        else:
            selected[name]['labels'].append(category_label)
            
    add_feature(rep_band, f"Top Band ({top_band})")
    add_feature(rep_elec, f"Top Electrode ({top_elec})")
    add_feature(rep_stat, f"Top Statistic ({top_stat})")
    add_feature(rep_top1, "Overall Top Feature #1")
    add_feature(rep_top2, "Overall Top Feature #2")
    
    # Save mapping for step 5
    mapping_file = run_dir / "viz" / "viz17_4_selected_features.json"
    with open(mapping_file, 'w') as f:
        json.dump(selected, f, indent=4)
        
    # --- Visualization ---
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Sort for plotting (lowest d at bottom)
    plot_data = sorted([(k, v['d'], " & ".join(v['labels'])) for k,v in selected.items()], key=lambda x: x[1])
    y_labels = [p[0].replace('_', ' ') for p in plot_data]
    d_vals = [p[1] for p in plot_data]
    annotations = [p[2] for p in plot_data]
    
    y_pos = np.arange(len(plot_data))
    bars = ax.barh(y_pos, d_vals, color='#9b59b6', alpha=0.8, edgecolor='black', linewidth=1.5)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(y_labels, fontweight='bold')
    ax.set_xlabel("|Cohen's d|", fontweight='bold')
    ax.set_title("Identified Representative Top Features for Logistic Regression", pad=20, fontweight='bold', fontsize=14)
    
    # Annotate bars
    for i, bar in enumerate(bars):
        width = bar.get_width()
        label = annotations[i]
        # Text inside or outside
        ax.text(width + 0.005, bar.get_y() + bar.get_height()/2, label, 
                va='center', ha='left', fontsize=11, fontstyle='italic', 
                bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', boxstyle='round,pad=0.2'))
        
    # Expand xlim to fit text
    max_d = max(d_vals)
    ax.set_xlim(0, max_d * 1.5)
    
    ax.grid(True, axis='x', linestyle='--', alpha=0.6)
    
    # Write description to tex file instead of plot
    viz_dir = run_dir / "viz"
    tex_text = (
        "\\textbf{Methodology:}\\\\\n"
        "To ensure a 1-to-1 Apples-to-Apples comparison with the 1D Engagement Index, "
        "each of the macro 'Top Parameters' (Band, Electrode, Stat) is represented "
        "by its strongest single constituent feature."
    )
    with open(viz_dir / "viz17_4_description.tex", "w") as f:
        f.write(tex_text)
            
    plt.tight_layout()
    viz_dir = run_dir / "viz"
    plt.savefig(viz_dir / "viz17_4_identifying_top.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  viz17_4: Generated top features mapping and visualization.")

if __name__ == "__main__":
    import sys
    run(Path(sys.argv[1]), {})
