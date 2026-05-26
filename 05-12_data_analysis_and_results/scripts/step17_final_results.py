#!/usr/bin/env python3
"""
step17_final_results.py — Determine best RF and EI combinations and output them for visualization.

Reads parallel_universe_metrics.csv, finds the best RF configuration (max Test F1) for Intra, 
and the best EI for Intra. Does the same for Inter.
Trigger viz17 for plotting.
"""

import pandas as pd
from pathlib import Path
import config
import importlib

def run(run_dir, params):
    config.pprint_step(17, "FINAL RESULTS SUMMARY")
    
    csv_path = run_dir / "parallel_universe_metrics.csv"
    if not csv_path.exists():
        print(f"  {csv_path.name} not found. Ensure step 16 has been run.")
        return
        
    df = pd.read_csv(csv_path)
    
    # We want to identify the best RF and best EI for both Intra and Inter
    best_configs = {}
    
    for scale in ['Intra', 'Inter']:
        df_scale = df[df['scale'] == scale]
        
        # Best RF
        df_rf = df_scale[df_scale['model'] == 'RF']
        if not df_rf.empty:
            mean_metrics = df_rf.groupby('combination')['test_f1'].mean().reset_index()
            best_rf_comb = mean_metrics.loc[mean_metrics['test_f1'].idxmax()]['combination']
            best_configs[f'{scale}_RF'] = best_rf_comb
            
        # Best EI
        df_ei = df_scale[df_scale['model'] == 'EI']
        if not df_ei.empty:
            mean_metrics = df_ei.groupby('combination')['test_f1'].mean().reset_index()
            best_ei_comb = mean_metrics.loc[mean_metrics['test_f1'].idxmax()]['combination']
            best_configs[f'{scale}_EI'] = best_ei_comb

    print("  Identified Best Combinations (by highest average Test F1):")
    for k, v in best_configs.items():
        print(f"    {k}: {v}")
        
    # Save best configs to a small json
    import json
    with open(run_dir / "best_combinations.json", 'w') as f:
        json.dump(best_configs, f, indent=4)
        
    print("  ✓ Triggering visualization...")
    try:
        viz_module = importlib.import_module("viz17_final_results_viz")
        viz_module.run(run_dir, params)
    except Exception as e:
        print(f"  ⚠ Failed to run viz17_final_results_viz: {e}")

if __name__ == "__main__":
    print("Use run.py to execute the pipeline.")
