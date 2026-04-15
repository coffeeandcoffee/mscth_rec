#!/usr/bin/env python3
"""
Step 6: Visual Representation of Results

Generates scientific box plots from the compiled Step 2 (Intra-Subject) 
and Step 3 (LOGO-CV) outputs, visualizing structural variance 
across the participant pool metrics mapping to Case B.
"""

import json
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt

def find_latest_run_dir(base_dir, prefix):
    dirs = [d for d in base_dir.glob(f"{prefix}*") if d.is_dir()]
    if not dirs:
        return None
    return max(dirs, key=lambda d: d.stat().st_mtime)

def plot_box_metrics(rf_data, logo_data, ei_data, output_dir):
    """
    Plots a side-by-side Box & Whisker plot comparing Key Performance Metrics
    Intra-Subject vs Cross-Participant vs Intra-Subject EI.
    """
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    metric_labels = ['Accuracy', 'Precision (STAY)', 'Recall / Sensitivity (STAY)', 'F1 Score']
    
    # Extract
    if rf_data:
        rf_lists = {m: [ind[m] for ind in rf_data.get('individuals', [])] for m in metrics}
    if logo_data:
        logo_lists = {m: [fold[m] for fold in logo_data.get('folds', [])] for m in metrics}
    if ei_data:
        ei_lists = {m: [] for m in metrics}
        for pid, stats in ei_data.items():
            if 'intra_ei_ml' in stats:
                ei_lists['accuracy'].append(stats['intra_ei_ml']['accuracy'])
                ei_lists['precision'].append(stats['intra_ei_ml']['precision'])
                ei_lists['recall'].append(stats['intra_ei_ml']['recall'])
                ei_lists['f1'].append(stats['intra_ei_ml']['f1'])
                
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Case B (STAY Paradigm) Result Distributions', fontsize=16, fontweight='bold', y=0.98)
    
    axes = axes.flatten()
    
    for i, m in enumerate(metrics):
        ax = axes[i]
        
        data_to_plot = []
        labels = []
        
        if rf_data and len(rf_lists[m]) > 0:
            data_to_plot.append(rf_lists[m])
            labels.append('Intra-Subject RF\n(Step 2)')
            
        if logo_data and len(logo_lists[m]) > 0:
            data_to_plot.append(logo_lists[m])
            labels.append('LOGO-CV RF\n(Step 3)')
            
        if ei_data and len(ei_lists[m]) > 0:
            data_to_plot.append(ei_lists[m])
            labels.append('Intra-Subject EI\n(Step 4)')
            
        if data_to_plot:
            box = ax.boxplot(data_to_plot, patch_artist=True, labels=labels,
                             boxprops=dict(facecolor='lightblue', color='black'),
                             capprops=dict(color='black'),
                             whiskerprops=dict(color='black'),
                             flierprops=dict(color='black', markeredgecolor='black'),
                             medianprops=dict(color='firebrick', linewidth=2))
                             
            ax.set_title(metric_labels[i], fontsize=13)
            ax.set_ylim(0.0, 1.0)
            ax.set_ylabel('Score')
            
            # Baseline explicitly mapped assuming perfectly balanced arrays
            ax.axhline(0.5, color='gray', linestyle='--', alpha=0.7)
            ax.text(len(labels)+0.3, 0.51, 'Random Baseline (0.5)', color='gray', fontsize=9)
            
            # Print exact mean values mapping
            for idx, plot_data in enumerate(data_to_plot):
                mean_val = sum(plot_data) / len(plot_data)
                ax.text(idx + 1, 0.05, f'Mean: {mean_val:.3f}', ha='center', fontsize=10, 
                        bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
                
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    out_file = output_dir / 'rf_metrics_boxplot.png'
    plt.savefig(out_file, dpi=300)
    plt.close()
    return out_file

def main():
    project_root = Path(__file__).resolve().parent.parent.parent
    outputs_dir = project_root / "04-26_data_analysis_and_results" / "outputs"
    
    vis_dir = outputs_dir / f"visualizations_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    vis_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("STEP 6: VISUAL REPRESENTATION GENERATOR")
    print("=" * 60)
    
    # 1. Grab Latest Results
    rf_dir = find_latest_run_dir(outputs_dir, "rf_run_")
    logo_dir = find_latest_run_dir(outputs_dir, "logo_cv_run_")
    ei_dir = find_latest_run_dir(outputs_dir, "ei_stats_")
    
    rf_data = None
    if rf_dir and (rf_dir / "rf_summary.json").exists():
        with open(rf_dir / "rf_summary.json", 'r') as f:
            rf_data = json.load(f)
        print(f"✓ Loaded Intra-Subject metrics: {rf_dir.name}")
    else:
        print("⚠ Could not evaluate rf_summary.json")

    logo_data = None
    if logo_dir and (logo_dir / "logo_cv_summary.json").exists():
        with open(logo_dir / "logo_cv_summary.json", 'r') as f:
            logo_data = json.load(f)
        print(f"✓ Loaded LOGO-CV metrics: {logo_dir.name}")
    else:
        print("⚠ Could not evaluate logo_cv_summary.json")
        
    ei_data = None
    if ei_dir and (ei_dir / "ei_results.json").exists():
        with open(ei_dir / "ei_results.json", 'r') as f:
            ei_data = json.load(f)
        print(f"✓ Loaded Engagement Index metrics: {ei_dir.name}")
    else:
        print("⚠ Could not evaluate ei_results.json")

    if not rf_data and not logo_data and not ei_data:
        print("Fatal error: No summary matrices located. Run ML loops first.")
        return 1

    # Plot
    plot_file = plot_box_metrics(rf_data, logo_data, ei_data, vis_dir)
    print(f"\n✓ Generated Visual Models => {plot_file.name}")
    print(f"Output Directory => {vis_dir}")

if __name__ == "__main__":
    main()
