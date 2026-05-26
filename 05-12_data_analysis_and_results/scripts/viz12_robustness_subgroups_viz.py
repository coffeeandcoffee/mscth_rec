#!/usr/bin/env python3
"""viz12 — Subgroup robustness visualization. 3 panels."""

import numpy as np, pandas as pd, json
from pathlib import Path
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import config

PALETTE = sns.color_palette("pastel")

def run(run_dir, params):
    viz_dir = run_dir / "viz"; viz_dir.mkdir(exist_ok=True)
    sub_csv = run_dir / "subgroup_comparisons.csv"
    if not sub_csv.exists(): return
    df = pd.read_csv(sub_csv)
    primary_json = run_dir / "step07_primary_result.json"
    if not primary_json.exists(): return
    with open(primary_json) as f: primary = json.load(f)
    recall_map = {r['pid']: r['recall'] for r in primary['per_participant']}

    # Load demographics
    df_demo = pd.read_csv(config.SURVEY_CSV)
    demo_map = {}
    for _, row in df_demo.iterrows():
        pid_str = str(row['ID']).strip()
        if pid_str.startswith('P'):
            try:
                pid = int(pid_str[1:])
                demo_map[pid] = row
            except ValueError:
                pass

    def parse_tiktok_usage(usage_str):
        s = str(usage_str).strip().lower()
        if '1-30 min' in s or '1-30' in s: return 15
        elif '30–60' in s or '30-60' in s: return 45
        elif '1–2 hour' in s or '1-2 hour' in s: return 90
        elif '2–3 hour' in s or '2-3 hour' in s: return 150
        elif 'more than 3' in s or '3+' in s: return 210
        return 45

    n_panels = len(df)
    fig, axes = plt.subplots(1, max(n_panels, 1), figsize=(6 * max(n_panels, 1), 6))
    if n_panels == 1: axes = [axes]
    fig.suptitle("Step 12 — Subgroup Robustness", fontsize=16, fontweight='bold')

    for idx, (_, row) in enumerate(df.iterrows()):
        ax = axes[idx]
        a_name, b_name = row['group_a'], row['group_b']
        comp = row['comparison']

        a_vals = []
        b_vals = []

        if comp == 'paid_vs_unpaid':
            a_vals = [recall_map[p] for p in config.PAID_PARTICIPANTS if p in recall_map]
            b_vals = [recall_map[p] for p in config.UNPAID_PARTICIPANTS if p in recall_map]
        elif comp == 'tiktok_light_vs_heavy':
            usage_col = 'How much time per day do you spend watching short-form videos?'
            usage_data = []
            for pid in recall_map:
                if pid in demo_map and usage_col in demo_map[pid].index:
                    usage_data.append({'pid': pid, 'recall': recall_map[pid], 'mins': parse_tiktok_usage(demo_map[pid][usage_col])})
            if usage_data:
                df_u = pd.DataFrame(usage_data)
                med = df_u['mins'].median()
                a_vals = df_u[df_u['mins'] <= med]['recall'].tolist()
                b_vals = df_u[df_u['mins'] > med]['recall'].tolist()
        elif comp == 'sex_male_vs_female':
            for pid in recall_map:
                if pid in demo_map and 'sex' in demo_map[pid].index:
                    sex = str(demo_map[pid]['sex']).strip().lower()
                    if sex == 'm': a_vals.append(recall_map[pid])
                    elif sex == 'f': b_vals.append(recall_map[pid])

        # Strip plot with box overlay
        for i, v in enumerate(a_vals):
            ax.scatter(0 + np.random.uniform(-0.1, 0.1), v, color=PALETTE[0], alpha=0.6, s=30)
        for i, v in enumerate(b_vals):
            ax.scatter(1 + np.random.uniform(-0.1, 0.1), v, color=PALETTE[1], alpha=0.6, s=30)

        if a_vals:
            ax.boxplot([a_vals], positions=[0], widths=0.4, patch_artist=True,
                      boxprops=dict(facecolor=PALETTE[0], alpha=0.3))
        if b_vals:
            ax.boxplot([b_vals], positions=[1], widths=0.4, patch_artist=True,
                      boxprops=dict(facecolor=PALETTE[1], alpha=0.3))

        ax.axhline(0.5, color='gray', linestyle='--', linewidth=0.5)
        ax.set_xticks([0, 1]); ax.set_xticklabels([a_name, b_name])
        ax.set_ylabel('STAY Recall')
        sig_text = f"p={row['p_value']:.3f}, r={row['effect_r']:.3f}"
        if not row['significant']:
            sig_text += " (n.s.)"
        ax.set_title(f"{comp}\n{sig_text}", fontsize=10)

    plt.tight_layout()
    plt.savefig(viz_dir / "viz12_subgroups.png", dpi=200); plt.close()
