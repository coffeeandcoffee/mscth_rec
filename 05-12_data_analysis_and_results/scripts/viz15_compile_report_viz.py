#!/usr/bin/env python3
"""viz15 — Master results table rendered as figure."""

import numpy as np, pandas as pd
from pathlib import Path
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import config


def run(run_dir, params):
    viz_dir = run_dir / "viz"; viz_dir.mkdir(exist_ok=True)
    master_csv = run_dir / "master_results.csv"
    if not master_csv.exists(): return

    df = pd.read_csv(master_csv)

    # Render table as a figure
    n_rows = len(df)
    fig_height = max(4, 0.5 * n_rows + 2)
    fig, ax = plt.subplots(figsize=(18, fig_height))
    ax.axis('off')
    fig.suptitle("Master Results — All Pipeline Comparisons",
                 fontsize=14, fontweight='bold', y=0.98)

    # Select display columns
    display_cols = ['comparison', 'primary_recall', 'comparison_recall',
                    'delta_pp', 'test', 'p_value', 'effect_r', 'significant']
    cols_present = [c for c in display_cols if c in df.columns]
    df_display = df[cols_present].copy()

    # Rename for display
    rename_map = {
        'comparison': 'Comparison',
        'primary_recall': 'Primary',
        'comparison_recall': 'Comp.',
        'delta_pp': 'Δ (pp)',
        'test': 'Test',
        'p_value': 'p-value',
        'effect_r': 'Effect r',
        'significant': 'Sig',
    }
    df_display = df_display.rename(columns=rename_map)

    table = ax.table(
        cellText=df_display.values,
        colLabels=df_display.columns,
        cellLoc='center',
        loc='center',
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.scale(1, 1.4)

    # Style header
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor('#4472C4')
            cell.set_text_props(color='white', fontweight='bold', fontsize=8)
        else:
            # Color significance column
            if col == len(cols_present) - 1:
                val = str(df_display.iloc[row - 1].iloc[-1])
                if '✅' in val:
                    cell.set_facecolor('#C6EFCE')
                elif '❌' in val:
                    cell.set_facecolor('#FFC7CE')
            # Alternate row colors
            if row % 2 == 0:
                cell.set_facecolor('#F2F2F2')

        cell.set_edgecolor('#D9D9D9')

    plt.tight_layout()
    plt.savefig(viz_dir / "viz15_master_results.png", dpi=200, bbox_inches='tight')
    plt.close()
