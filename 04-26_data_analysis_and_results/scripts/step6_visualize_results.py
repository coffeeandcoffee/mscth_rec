#!/usr/bin/env python3
"""
Step 6: Master Narrative Visual Representation

Generates the 4-Pillar Scientific Boxplot demonstrating the thesis logic arc:
1. Clinical Baseline (EI)
2. Structural Expansion (RF Notch)
3. High Gamma Proof (RF Nonotch)
4. Generalizability Limit (LOGO-CV)
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

def plot_master_box_metrics(ei_data, rf_notch_data, rf_nonotch_data, logo_data, output_dir):
    """
    Plots a side-by-side Box & Whisker plot comparing Key Performance Metrics
    across the four distinct narrative steps.
    """
    metrics = ['accuracy', 'recall']
    metric_labels = ['Validation Accuracy', 'STAY Recall (Sensitivity)']
    
    # Extract data arrays
    plot_data_dict = {'accuracy': [], 'recall': []}
    labels = []
    
    # 1. EI
    ei_acc, ei_rec = [], []
    if ei_data:
        for pid, stats in ei_data.items():
            if 'intra_ei_ml' in stats:
                ei_acc.append(stats['intra_ei_ml']['accuracy'])
                ei_rec.append(stats['intra_ei_ml']['recall'])
        if ei_acc:
            plot_data_dict['accuracy'].append(ei_acc)
            plot_data_dict['recall'].append(ei_rec)
            labels.append('1. Clinical EI\n(Baseline)')

    # 2. RF Notch
    if rf_notch_data:
        acc = [ind['accuracy'] for ind in rf_notch_data.get('individuals', [])]
        rec = [ind['recall'] for ind in rf_notch_data.get('individuals', [])]
        plot_data_dict['accuracy'].append(acc)
        plot_data_dict['recall'].append(rec)
        labels.append('2. RF-112\n(With 50Hz Notch)')
        
    # 3. RF NoNotch
    if rf_nonotch_data:
        acc = [ind['accuracy'] for ind in rf_nonotch_data.get('individuals', [])]
        rec = [ind['recall'] for ind in rf_nonotch_data.get('individuals', [])]
        plot_data_dict['accuracy'].append(acc)
        plot_data_dict['recall'].append(rec)
        labels.append('3. RF-112\n(No Notch / Gamma)')
        
    # 4. LOGO
    if logo_data:
        acc = [fold['accuracy'] for fold in logo_data.get('folds', [])]
        rec = [fold['recall'] for fold in logo_data.get('folds', [])]
        plot_data_dict['accuracy'].append(acc)
        plot_data_dict['recall'].append(rec)
        labels.append('4. LOGO-CV\n(Generalizability)')

    # Plotting
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle('The STILL Paradigm: 4-Pillar Narrative Progression', fontsize=18, fontweight='bold', y=0.98)
    
    colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']
    
    for i, m in enumerate(metrics):
        ax = axes[i]
        data_to_plot = plot_data_dict[m]
        
        if data_to_plot:
            bplot = ax.boxplot(data_to_plot, patch_artist=True, labels=labels,
                               boxprops=dict(facecolor='lightblue', color='black'),
                               capprops=dict(color='black'),
                               whiskerprops=dict(color='black'),
                               flierprops=dict(color='black', markeredgecolor='black'),
                               medianprops=dict(color='firebrick', linewidth=2.5))
            
            for patch, color in zip(bplot['boxes'], colors[:len(data_to_plot)]):
                patch.set_facecolor(color)
                               
            ax.set_title(metric_labels[i], fontsize=15, pad=15)
            ax.set_ylim(0.2, 1.0)
            ax.set_ylabel('Score', fontsize=12)
            ax.tick_params(axis='x', labelsize=11)
            
            # Baseline explicitly mapped
            ax.axhline(0.5, color='gray', linestyle='--', alpha=0.8, linewidth=2)
            ax.text(0.6, 0.51, 'Random Chance (0.5)', color='gray', fontsize=10, fontweight='bold')
            
            # Print exact mean values mapping
            for idx, plot_data in enumerate(data_to_plot):
                mean_val = sum(plot_data) / len(plot_data)
                ax.text(idx + 1, 0.25, f'Mean:\n{mean_val:.3f}', ha='center', fontsize=11, 
                        bbox=dict(facecolor='white', alpha=0.9, edgecolor='gray', boxstyle='round,pad=0.3'))
                
    plt.tight_layout(rect=[0, 0.03, 1, 0.93])
    
    out_file = output_dir / 'master_narrative_boxplot.png'
    plt.savefig(out_file, dpi=300, bbox_inches='tight')
    plt.close()
    return out_file

def main():
    project_root = Path(__file__).resolve().parent.parent.parent
    outputs_dir = project_root / "04-26_data_analysis_and_results" / "outputs"
    
    vis_dir = outputs_dir / f"visualizations_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    vis_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("STEP 6: MASTER NARRATIVE VISUAL GENERATOR")
    print("=" * 60)
    
    # 1. Grab Latest Results
    # Carefully parse distinct prefixes
    ei_dir = find_latest_run_dir(outputs_dir, "ei_stats_")
    rf_notch_dir = find_latest_run_dir(outputs_dir, "rf_run_2") # Matches 2026...
    rf_nonotch_dir = find_latest_run_dir(outputs_dir, "rf_run_nonotch_")
    logo_dir = find_latest_run_dir(outputs_dir, "logo_cv_run_")
    
    def load_json(dir_path, filename):
        if dir_path and (dir_path / filename).exists():
            with open(dir_path / filename, 'r') as f:
                return json.load(f)
        return None

    ei_data = load_json(ei_dir, "ei_results.json")
    rf_notch_data = load_json(rf_notch_dir, "rf_summary.json")
    rf_nonotch_data = load_json(rf_nonotch_dir, "rf_summary.json")
    logo_data = load_json(logo_dir, "logo_cv_summary.json")

    print(f"✓ Loaded EI Baseline: {'YES' if ei_data else 'NO'}")
    print(f"✓ Loaded RF Notch: {'YES' if rf_notch_data else 'NO'}")
    print(f"✓ Loaded RF NoNotch: {'YES' if rf_nonotch_data else 'NO'}")
    print(f"✓ Loaded LOGO-CV: {'YES' if logo_data else 'NO'}")

    if not any([ei_data, rf_notch_data, rf_nonotch_data, logo_data]):
        print("Fatal error: No summary matrices located. Run ML loops first.")
        return 1

    # Plot
    plot_file = plot_master_box_metrics(ei_data, rf_notch_data, rf_nonotch_data, logo_data, vis_dir)
    print(f"\n✓ Generated Master Boxplot => {plot_file.name}")
    print(f"Output Directory => {vis_dir}")

if __name__ == "__main__":
    main()
