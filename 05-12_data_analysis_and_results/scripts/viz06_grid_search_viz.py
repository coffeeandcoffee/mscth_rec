#!/usr/bin/env python3
"""viz06 — Grid search visualization. 2 panels."""

import numpy as np, json, pandas as pd
from pathlib import Path
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import config

PALETTE = sns.color_palette("pastel")

def run(run_dir, params):
    viz_dir = run_dir / "viz"; viz_dir.mkdir(exist_ok=True)
    gs_dir = run_dir / "grid_search"
    report_csv = gs_dir / "grid_search_report.csv"
    if not report_csv.exists(): return

    df = pd.read_csv(report_csv)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Step 06 — Grid Search", fontsize=16, fontweight='bold')

    # Panel 1 — Hyperparameter selection frequency
    ax = axes[0]
    # Load best params per participant
    best_ne, best_md, best_ml = [], [], []
    for pid in config.INCLUDED_PARTICIPANTS:
        bp_path = gs_dir / f"P{pid}_best_params.json"
        if bp_path.exists():
            bp = json.load(open(bp_path))
            best_ne.append(bp.get('n_estimators', 200))
            best_md.append(bp.get('max_depth', 7))
            best_ml.append(bp.get('min_samples_leaf', 5))

    if best_ne:
        grid = params.get('step06', {}).get('param_grid', {})
        ne_vals = grid.get('n_estimators', [100, 200, 300])
        md_vals = grid.get('max_depth', [5, 7, 10])
        ml_vals = grid.get('min_samples_leaf', [3, 5, 10])

        width = 0.25
        x_ne = np.arange(len(ne_vals))
        x_md = np.arange(len(md_vals)) + len(ne_vals) + 1
        x_ml = np.arange(len(ml_vals)) + len(ne_vals) + len(md_vals) + 2

        for vals, x, data, label, color in [
            (ne_vals, x_ne, best_ne, 'n_estimators', PALETTE[0]),
            (md_vals, x_md, best_md, 'max_depth', PALETTE[1]),
            (ml_vals, x_ml, best_ml, 'min_samples_leaf', PALETTE[2]),
        ]:
            counts = [sum(1 for d in data if d == v) for v in vals]
            ax.bar(x, counts, color=color, alpha=0.8, label=label)
            ax.set_xticks(list(x_ne) + list(x_md) + list(x_ml))
            ax.set_xticklabels([str(v) for v in ne_vals + md_vals + ml_vals],
                               fontsize=7, rotation=45)
        ax.set_ylabel('Count (participants)'); ax.set_title('HP Selection Frequency')
        ax.legend(fontsize=8)

    # Panel 2 — Inner CV score surface (one participant)
    ax = axes[1]
    rep_pid = config.INCLUDED_PARTICIPANTS[0]
    pid_rows = df[df['pid'] == rep_pid]
    if not pid_rows.empty and 'max_depth' in pid_rows.columns and 'n_estimators' in pid_rows.columns:
        grid_p = params.get('step06', {}).get('param_grid', {})
        md_v = sorted(pid_rows['max_depth'].unique())
        ne_v = sorted(pid_rows['n_estimators'].unique())
        # Average over min_samples_leaf for heatmap
        heatmap = np.zeros((len(md_v), len(ne_v)))
        for i, md in enumerate(md_v):
            for j, ne in enumerate(ne_v):
                sub = pid_rows[(pid_rows['max_depth'] == md) & (pid_rows['n_estimators'] == ne)]
                heatmap[i, j] = sub['mean_inner_recall'].mean() if len(sub) > 0 else 0.5
        im = ax.imshow(heatmap, cmap='YlGnBu', aspect='auto')
        ax.set_xticks(range(len(ne_v))); ax.set_xticklabels(ne_v)
        ax.set_yticks(range(len(md_v))); ax.set_yticklabels(md_v)
        ax.set_xlabel('n_estimators'); ax.set_ylabel('max_depth')
        ax.set_title(f'Inner CV Score — P{rep_pid}')
        # Mark best
        bp_path = gs_dir / f"P{rep_pid}_best_params.json"
        if bp_path.exists():
            bp = json.load(open(bp_path))
            bi = md_v.index(bp['max_depth']) if bp['max_depth'] in md_v else 0
            bj = ne_v.index(bp['n_estimators']) if bp['n_estimators'] in ne_v else 0
            ax.plot(bj, bi, '*', color='red', markersize=15)
        plt.colorbar(im, ax=ax, shrink=0.7)

    plt.tight_layout()
    plt.savefig(viz_dir / "viz06_grid_search.png", dpi=200); plt.close()
