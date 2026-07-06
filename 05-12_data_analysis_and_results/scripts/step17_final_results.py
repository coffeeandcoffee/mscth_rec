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
            
        if getattr(config, 'ENABLE_TOP_FEATURES_EVAL', False):
            for top_model in ['TP10_raw_std_LR', 'All_raw_std_LR', 'AF7_high_gamma_mean_LR', 'All_high_gamma_mean_LR']:
                df_top = df_scale[df_scale['model'] == top_model]
                if not df_top.empty:
                    mean_metrics = df_top.groupby('combination')['test_f1'].mean().reset_index()
                    best_comb = mean_metrics.loc[mean_metrics['test_f1'].idxmax()]['combination']
                    best_configs[f'{scale}_{top_model}'] = best_comb

    print("  Identified Best Combinations (by highest average Test F1):")
    for k, v in best_configs.items():
        print(f"    {k}: {v}")
        
    # Save best configs to a small json
    import json
    with open(run_dir / "best_combinations.json", 'w') as f:
        json.dump(best_configs, f, indent=4)
        
    print("  ✓ Triggering visualization...")
    try:
        import viz17_diagnostics
        for scale in ['Intra', 'Inter']:
            viz17_diagnostics.generate_diagnostics(scale, df, best_configs, run_dir / "viz")
        try:
            viz_module = importlib.import_module("viz17_final_results_viz")
            viz_line_module = importlib.import_module("viz17_window_exploration_line")
            viz_feat_module = importlib.import_module("viz17_feature_ranking")
            viz_top_id = importlib.import_module("viz17_4_identifying_top")
            viz_top_eval = importlib.import_module("viz17_5_top_vs_ei")
            
            print("  -> Generating visual diagnostics...")
            viz_module.run(run_dir, params)
            print("  -> Generating line plots for exploration...")
            viz_line_module.run(run_dir, params)
            print("  -> Generating feature rankings...")
            viz_feat_module.run(run_dir, params)
            print("  -> Generating viz17_4 identifying top features...")
            viz_top_id.run(run_dir, params)
            print("  -> Generating viz17_5 top vs ei evaluation...")
            viz_top_eval.run(run_dir, params)
        except Exception as e:
            print(f"  ✗ Failed to generate some visuals: {e}")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"  ⚠ Failed to run visualizations: {e}")

if __name__ == "__main__":
    print("Use run.py to execute the pipeline.")
