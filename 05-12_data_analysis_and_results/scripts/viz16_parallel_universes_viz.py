#!/usr/bin/env python3
"""
viz16_parallel_universes_viz.py — Plot the 20-way evaluation matrix.

Generates three figures:
1. viz16a_parallel_universes_recall.png: Test Recall
2. viz16b_parallel_universes_f1.png: Test F1
3. viz16c_parallel_universes_overfitting.png: Train vs Test F1 (Grouped)
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import config
import seaborn as sns

PALETTE = sns.color_palette("deep")

def plot_matrix(df_metrics, metric_col, out_filename, title, ylabel, is_grouped=False):
    # Universes in order
    universes = [
        'NoNotch|NoArt|NoBurst',
        'Notch|NoArt|NoBurst',
        'Notch|Art|NoBurst',
        'Notch|NoArt|Burst',
        'Notch|Art|Burst'
    ]
    labels = [
        'NoNotch\n(Raw)',
        'Notch\n(Baseline)',
        '+ Artifact\nRejection',
        '+ Burst\nRejection',
        '+ Art & Burst\nRejection'
    ]

    fig, axes = plt.subplots(2, 2, figsize=(16, 12), sharey=True)
    fig.suptitle(title, fontsize=16, fontweight='bold', y=0.95)

    scales = ['Intra', 'Inter']
    models = ['EI', 'RF']

    for i, scale in enumerate(scales):
        for j, model in enumerate(models):
            ax = axes[i, j]
            
            # Prepare data
            valid_positions = []
            valid_data_test = []
            valid_data_train = []
            
            for k, u in enumerate(universes):
                comb_name = f"{scale}|{u}|{model}"
                subset = df_metrics[df_metrics['combination'] == comb_name]
                if not subset.empty:
                    valid_positions.append(k + 1)
                    if is_grouped:
                        # Assuming metric_col is 'f1', we want 'test_f1' and 'train_f1'
                        valid_data_test.append(subset[f'test_{metric_col}'].values)
                        valid_data_train.append(subset[f'train_{metric_col}'].values)
                    else:
                        valid_data_test.append(subset[metric_col].values)
                        
            if valid_data_test:
                if is_grouped:
                    # Plot grouped boxplots
                    # Offset positions for train and test
                    pos_train = [p - 0.15 for p in valid_positions]
                    pos_test = [p + 0.15 for p in valid_positions]
                    
                    bp1 = ax.boxplot(valid_data_train, positions=pos_train, widths=0.25, patch_artist=True)
                    bp2 = ax.boxplot(valid_data_test, positions=pos_test, widths=0.25, patch_artist=True)
                    
                    for patch in bp1['boxes']:
                        patch.set_facecolor(PALETTE[0])
                        patch.set_alpha(0.4)
                    for patch in bp2['boxes']:
                        patch.set_facecolor(PALETTE[0])
                        patch.set_alpha(0.9)
                        
                    # Custom legend for grouped
                    from matplotlib.lines import Line2D
                    custom_lines = [Line2D([0], [0], color=PALETTE[0], lw=4, alpha=0.4),
                                    Line2D([0], [0], color=PALETTE[0], lw=4, alpha=0.9)]
                    ax.legend(custom_lines, ['Train', 'Test'], loc='lower right', fontsize=8)
                else:
                    bp = ax.boxplot(valid_data_test, positions=valid_positions, widths=0.5, patch_artist=True)
                    for k, patch in enumerate(bp['boxes']):
                        is_baseline = (valid_positions[k] == 2)
                        color = PALETTE[0] if is_baseline else PALETTE[1]
                        patch.set_facecolor(color)
                        patch.set_alpha(0.7)

            ax.set_xticks(range(1, 6))
            ax.set_xticklabels(labels, fontsize=9)
            ax.set_title(f"{scale}-Subject Evaluation — {model} Model", fontsize=12, fontweight='bold')
            ax.axhline(0.5, color='gray', linestyle='--', linewidth=1)
            if j == 0:
                ax.set_ylabel(ylabel)
            ax.grid(axis='y', linestyle=':', alpha=0.6)

    plt.tight_layout(rect=[0, 0.03, 1, 0.93])
    plt.savefig(out_filename, dpi=300, bbox_inches='tight')
    plt.close()

def run(run_dir, params):
    config.pprint_step(16, "VISUALIZING PARALLEL UNIVERSES")

    viz_dir = run_dir / "viz"
    viz_dir.mkdir(parents=True, exist_ok=True)

    csv_path = run_dir / "parallel_universe_metrics.csv"
    if not csv_path.exists():
        print("  ⚠ parallel_universe_metrics.csv not found. Run step 16 first.")
        return

    df = pd.read_csv(csv_path)

    # 1. Recall
    plot_matrix(df, 'test_recall', viz_dir / "viz16a_parallel_universes_recall.png", 
                "Master 20-Way Parallel Universe Evaluation (Test Recall)", "STAY Recall")
    print(f"  ✓ Saved to viz/viz16a_parallel_universes_recall.png")
    print("\n  [How to interpret viz16a - Recall]")
    print("  Recall measures how many actual STAY windows were successfully detected.")
    print("  However, a model that simply predicts STAY for everything will score 100% Recall.")
    print("  Use this to see if the model misses positive cases, but verify with F1.\n")

    # 2. F1
    plot_matrix(df, 'test_f1', viz_dir / "viz16b_parallel_universes_f1.png", 
                "Master 20-Way Parallel Universe Evaluation (Test F1-Score)", "STAY F1-Score")
    print(f"  ✓ Saved to viz/viz16b_parallel_universes_f1.png")
    print("\n  [How to interpret viz16b - F1-Score]")
    print("  F1 harmonically balances Precision and Recall. Because the test set is exactly 50/50 balanced,")
    print("  a naive 'always STAY' classifier scores exactly 0.66 (1.0 recall, 0.5 precision).")
    print("  True performance above chance will sit between 0.67 and 1.0.\n")

    # 3. Overfitting (Train vs Test F1)
    plot_matrix(df, 'f1', viz_dir / "viz16c_parallel_universes_overfitting.png", 
                "Master 20-Way Parallel Universe Evaluation (Train vs Test F1)", "F1-Score", is_grouped=True)
    print(f"  ✓ Saved to viz/viz16c_parallel_universes_overfitting.png")
    print("\n  [How to interpret viz16c - Overfitting]")
    print("  Lighter bars are Train F1, darker bars are Test F1.")
    print("  A large gap between Train and Test indicates severe overfitting (memorization).")
    print("  Closely matched bars indicate the model is generalizing well to strictly unseen data.\n")

if __name__ == "__main__":
    print("Use run.py to execute the pipeline.")
