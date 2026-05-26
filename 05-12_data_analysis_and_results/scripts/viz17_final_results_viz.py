#!/usr/bin/env python3
"""
viz17_final_results_viz.py — Final clean thesis plots.

Creates two clean plots (one for Intra, one for Inter LOGO-CV).
Each plot compares the Best RF vs the Best EI across all participants for Recall and F1.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import json
import config

PALETTE = sns.color_palette("pastel")

def run(run_dir, params):
    viz_dir = run_dir / "viz"
    viz_dir.mkdir(parents=True, exist_ok=True)
    
    csv_path = run_dir / "parallel_universe_metrics.csv"
    best_path = run_dir / "best_combinations.json"
    
    if not csv_path.exists() or not best_path.exists():
        return
        
    df = pd.read_csv(csv_path)
    with open(best_path, 'r') as f:
        best_configs = json.load(f)
        
    p_exp = params.get('experimental', config.DEFAULT_PARAMS.get('experimental', {}))
    toggles = []
    if p_exp.get('remove_min_max', False): toggles.append("No Min/Max")
    if p_exp.get('use_hilbert_envelope', False): toggles.append("Hilbert Env")
    if p_exp.get('extract_erp_features', False): toggles.append("ERP ON")
    toggle_str = f" [{', '.join(toggles)}]" if toggles else ""
        
    for scale in ['Intra', 'Inter']:
        rf_comb = best_configs.get(f'{scale}_RF')
        ei_comb = best_configs.get(f'{scale}_EI')
        
        if not rf_comb or not ei_comb:
            continue
            
        df_rf = df[df['combination'] == rf_comb].copy()
        df_ei = df[df['combination'] == ei_comb].copy()
        
        # Format labels
        rf_label = f"Best RF\n({rf_comb.replace(scale+'|', '')})"
        ei_label = f"Best EI\n({ei_comb.replace(scale+'|', '')})"
        
        df_rf['ModelLabel'] = rf_label
        df_ei['ModelLabel'] = ei_label
        
        df_plot = pd.concat([df_rf, df_ei])
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 7))
        fig.suptitle(f"Final Performance Results ({scale} Scale){toggle_str}", fontsize=18, fontweight='bold')
        
        # Plot Recall
        sns.boxplot(data=df_plot, x='ModelLabel', y='test_recall', ax=axes[0], 
                    palette=[PALETTE[2], PALETTE[0]], width=0.5)
        sns.stripplot(data=df_plot, x='ModelLabel', y='test_recall', ax=axes[0], 
                      color='black', alpha=0.6, jitter=True)
        axes[0].set_title(f"Test Recall", fontsize=14)
        axes[0].set_ylabel("Recall")
        axes[0].set_xlabel("")
        axes[0].set_ylim(0, 1.05)
        axes[0].axhline(0.5, color='gray', linestyle='--', alpha=0.5, label='Random Chance')
        axes[0].legend()
        
        # Plot F1
        sns.boxplot(data=df_plot, x='ModelLabel', y='test_f1', ax=axes[1], 
                    palette=[PALETTE[2], PALETTE[0]], width=0.5)
        sns.stripplot(data=df_plot, x='ModelLabel', y='test_f1', ax=axes[1], 
                      color='black', alpha=0.6, jitter=True)
        axes[1].set_title(f"Test F1 Score", fontsize=14)
        axes[1].set_ylabel("F1 Score")
        axes[1].set_xlabel("")
        axes[1].set_ylim(0, 1.05)
        axes[1].axhline(0.5, color='gray', linestyle='--', alpha=0.5, label='Random Chance')
        axes[1].legend()
        
        plt.tight_layout(rect=[0, 0, 1, 0.93])
        plt.savefig(viz_dir / f"viz17_{scale.lower()}_final.png", dpi=200)
        plt.close()

if __name__ == "__main__":
    print("Use run.py to execute the pipeline.")
