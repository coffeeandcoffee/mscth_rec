#!/usr/bin/env python3
"""viz07 — Intra-subject training visualization. 5 panels."""

import numpy as np, pickle, json
from pathlib import Path
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import config

PALETTE = sns.color_palette("pastel")

def run(run_dir, params):
    viz_dir = run_dir / "viz"; viz_dir.mkdir(exist_ok=True)
    results_dir = run_dir / "results" / "intra"
    seeds = params.get('step07', {}).get('seeds', [0, 1, 7, 42, 99])

    # Collect per-participant data
    pdata = []
    for pid in config.INCLUDED_PARTICIPANTS:
        pkl = results_dir / f"P{pid}_results.pkl"
        if not pkl.exists(): continue
        with open(pkl, 'rb') as f: d = pickle.load(f)
        rec_tb = [d['results'].get('temporal_blocked', {}).get('full', {}).get(s, {}).get('recall', 0.5) for s in seeds]
        rec_ng = [d['results'].get('temporal_blocked_no_gap', {}).get('full', {}).get(s, {}).get('recall', 0.5) for s in seeds]
        rec_rs = [d['results'].get('random_split', {}).get('full', {}).get(s, {}).get('recall', 0.5) for s in seeds]
        rec_fr = [d['results'].get('temporal_blocked', {}).get('frontal', {}).get(s, {}).get('recall', 0.5) for s in seeds]
        rec_te = [d['results'].get('temporal_blocked', {}).get('temporal', {}).get(s, {}).get('recall', 0.5) for s in seeds]
        pdata.append({'pid': pid, 'tb': np.mean(rec_tb), 'ng': np.mean(rec_ng), 'rs': np.mean(rec_rs),
                       'fr': np.mean(rec_fr), 'te': np.mean(rec_te),
                       'tb_sd': np.std(rec_tb), 'seeds_tb': rec_tb})

    if not pdata: return
    pdata.sort(key=lambda x: x['tb'])

    fig, axes = plt.subplots(2, 3, figsize=(20, 15))
    fig.suptitle("Step 07 — Intra-Subject Training (PRIMARY RESULT)", fontsize=16, fontweight='bold')

    # Panel 1 — Primary result: per-participant STAY recall
    ax = axes[0, 0]
    for i, d in enumerate(pdata):
        color = PALETTE[0] if d['tb'] > 0.5 else PALETTE[3]
        ax.errorbar(d['tb'], i, xerr=d['tb_sd'], fmt='o', color=color, markersize=5, capsize=2)
    ax.axvline(0.5, color='gray', linestyle='--', linewidth=0.8, label='50% chance')
    ax.set_yticks(range(len(pdata)))
    ax.set_yticklabels([f"P{d['pid']}" for d in pdata], fontsize=7)
    ax.set_xlabel('STAY Recall'); ax.set_title('Primary: Per-Participant STAY Recall')
    mean_r = np.mean([d['tb'] for d in pdata])
    ax.axvline(mean_r, color='blue', linestyle=':', linewidth=0.8, label=f'Mean={mean_r:.3f}')
    ax.legend(fontsize=7)
    
    # Add explanation
    ax.text(0.5, -0.20, "HOW TO INTERPRET:\nEach dot is one participant's ability to classify STAY vs SKIP.\nThe line extending from the dot shows the variation across different Random Seeds.\nIf a dot is right of the dotted 50% line, the model learned something real.\nIf it's left of the line, the model completely failed for that person.", 
            ha='center', va='top', transform=ax.transAxes, fontsize=8, bbox=dict(facecolor='lightyellow', alpha=0.8, edgecolor='orange', boxstyle='round,pad=0.5'))

    # Panel 2 — Leakage comparison: temporal vs random-split
    ax = axes[0, 1]
    tb_vals = [d['tb'] for d in pdata]
    ng_vals = [d['ng'] for d in pdata]
    rs_vals = [d['rs'] for d in pdata]
    bp1 = ax.boxplot([tb_vals], positions=[1], widths=0.25, patch_artist=True)
    bp2 = ax.boxplot([ng_vals], positions=[2], widths=0.25, patch_artist=True)
    bp3 = ax.boxplot([rs_vals], positions=[3], widths=0.25, patch_artist=True)
    bp1['boxes'][0].set_facecolor(PALETTE[0])
    bp2['boxes'][0].set_facecolor(PALETTE[2])
    bp3['boxes'][0].set_facecolor(PALETTE[1])
    for d in pdata:
        ax.plot([1, 2, 3], [d['tb'], d['ng'], d['rs']], color='gray', alpha=0.3, linewidth=0.5)
    ax.set_xticks([1, 2, 3]); ax.set_xticklabels(['Temporal\nBlocked\n(3s Gap)', 'Temporal\nBlocked\n(0s Gap)', 'Random\nSplit\n(Shuffled)'])
    ax.set_ylabel('STAY Recall'); ax.set_title('Leakage Comparison')
    ax.axhline(0.5, color='gray', linestyle='--', linewidth=0.5)

    # Add explanation
    ax.text(0.5, -0.20, "HOW TO INTERPRET:\nCompares strict evaluation (3s firewall) vs two flawed evaluations.\n'0s Gap' allows overlapping adjacent sliding windows to leak data.\n'Random Split' ignores time entirely and leaks everything.", 
            ha='center', va='top', transform=ax.transAxes, fontsize=8, bbox=dict(facecolor='lightyellow', alpha=0.8, edgecolor='orange', boxstyle='round,pad=0.5'))

    # Panel 3 — Recall across seeds (heatmap)
    ax = axes[0, 2]
    hm = np.array([d['seeds_tb'] for d in pdata])
    im = ax.imshow(hm, cmap='YlGnBu', aspect='auto', vmin=0.3, vmax=0.7)
    ax.set_xticks(range(len(seeds))); ax.set_xticklabels(seeds, fontsize=8)
    ax.set_yticks(range(len(pdata))); ax.set_yticklabels([f"P{d['pid']}" for d in pdata], fontsize=6)
    ax.set_xlabel('Seed'); ax.set_title('Recall × Seed')
    plt.colorbar(im, ax=ax, shrink=0.7)
    
    # Add explanation
    ax.text(0.5, -0.20, "HOW TO INTERPRET:\nShows how much the 'random seed' (starting luck) affects the model.\nDark blue means the model scored well on that seed. Yellow means it scored poorly.\nIf a row is highly variable, the model is unstable. If a row is mostly blue, it is robust.", 
            ha='center', va='top', transform=ax.transAxes, fontsize=8, bbox=dict(facecolor='lightyellow', alpha=0.8, edgecolor='orange', boxstyle='round,pad=0.5'))

    # Panel 4 — Aggregate confusion matrix
    ax = axes[1, 0]
    agg_cm = np.zeros((2, 2))
    for pid in config.INCLUDED_PARTICIPANTS:
        pkl = results_dir / f"P{pid}_results.pkl"
        if not pkl.exists(): continue
        with open(pkl, 'rb') as f: d = pickle.load(f)
        for s in seeds:
            cm = d['results'].get('temporal_blocked', {}).get('full', {}).get(s, {}).get('confusion_matrix')
            if cm: agg_cm += np.array(cm)
    if agg_cm.sum() > 0:
        cm_norm = agg_cm / agg_cm.sum()
        im = ax.imshow(cm_norm, cmap='Blues', vmin=0, vmax=0.5)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f'{cm_norm[i,j]:.2%}', ha='center', va='center', fontsize=12)
        ax.set_xticks([0,1]); ax.set_xticklabels(['Pred SKIP', 'Pred STAY'])
        ax.set_yticks([0,1]); ax.set_yticklabels(['True SKIP', 'True STAY'])
        ax.set_title('Aggregate Confusion Matrix')

    # Add explanation
    ax.text(0.5, -0.20, "HOW TO INTERPRET:\nA confusion matrix showing where the model made its guesses across everyone.\nTop Left: Correctly guessed SKIP. Bottom Right: Correctly guessed STAY.\nOff-diagonals are mistakes. We want the highest numbers on the diagonal.", 
            ha='center', va='top', transform=ax.transAxes, fontsize=8, bbox=dict(facecolor='lightyellow', alpha=0.8, edgecolor='orange', boxstyle='round,pad=0.5'))

    # Panel 5 — Electrode set comparison
    ax = axes[1, 1]
    full_v = [d['tb'] for d in pdata]
    front_v = [d['fr'] for d in pdata]
    temp_v = [d['te'] for d in pdata]
    parts = ax.violinplot([full_v, front_v, temp_v], positions=[1, 2, 3], showmeans=True)
    colors = [PALETTE[0], PALETTE[1], PALETTE[2]]
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colors[i]); pc.set_alpha(0.7)
    ax.set_xticks([1, 2, 3]); ax.set_xticklabels([f'Full\n({config.N_FEATURES_FULL})', f'Frontal\n({config.N_FEATURES_SUBSET})', f'Temporal\n({config.N_FEATURES_SUBSET})'])
    ax.axhline(0.5, color='gray', linestyle='--', linewidth=0.5)
    ax.set_ylabel('STAY Recall'); ax.set_title('Electrode Set Comparison')

    # Add explanation
    ax.text(0.5, -0.20, "HOW TO INTERPRET:\nCompares performance when the model only sees data from Frontal electrodes vs Temporal vs Full.\nHelps answer: 'Which part of the brain actually contains the useful signal?'\nIf Frontal is just as good as Full, we don't need Temporal.", 
            ha='center', va='top', transform=ax.transAxes, fontsize=8, bbox=dict(facecolor='lightyellow', alpha=0.8, edgecolor='orange', boxstyle='round,pad=0.5'))

    axes[1, 2].axis('off')  # Empty panel

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.subplots_adjust(hspace=0.4)
    plt.savefig(viz_dir / "viz07_train_intra.png", dpi=200); plt.close()
