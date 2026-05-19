#!/usr/bin/env python3
"""
step12_robustness_subgroups.py — Demographic subgroup comparisons.

Reads step07 per-participant results and demographics CSV.
Runs Mann-Whitney U for two-group comparisons: TikTok usage, sex, paid vs unpaid.

OUT: subgroup_comparisons.csv
"""

import numpy as np
import pickle
import json
from pathlib import Path
import pandas as pd
from scipy import stats as sp_stats
import config
import warnings
warnings.filterwarnings('ignore')


def parse_tiktok_usage(usage_str):
    """Convert TikTok usage string to minutes for median split."""
    s = str(usage_str).strip().lower()
    if '1-30 min' in s or '1-30' in s:
        return 15
    elif '30–60' in s or '30-60' in s:
        return 45
    elif '1–2 hour' in s or '1-2 hour' in s:
        return 90
    elif '2–3 hour' in s or '2-3 hour' in s:
        return 150
    elif 'more than 3' in s or '3+' in s:
        return 210
    return 45  # default


def mann_whitney_r(group_a, group_b):
    """Mann-Whitney U with rank-biserial effect size r."""
    if len(group_a) < 2 or len(group_b) < 2:
        return 0.0, 1.0, 0.0
    try:
        u, p = sp_stats.mannwhitneyu(group_a, group_b, alternative='two-sided')
        n1, n2 = len(group_a), len(group_b)
        r = 1 - (2 * u) / (n1 * n2)
        return float(u), float(p), float(r)
    except Exception:
        return 0.0, 1.0, 0.0


def run(run_dir, params):
    config.pprint_step(12, "SUBGROUP ROBUSTNESS")

    # Load primary result
    with open(run_dir / "step07_primary_result.json", 'r') as f:
        primary = json.load(f)

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

    rows = []

    # ── 1. TikTok usage: light vs heavy (median split) ──
    usage_col = 'How much time per day do you spend watching short-form videos?'
    usage_data = []
    for pid in recall_map:
        if pid in demo_map and usage_col in demo_map[pid].index:
            mins = parse_tiktok_usage(demo_map[pid][usage_col])
            usage_data.append({'pid': pid, 'recall': recall_map[pid], 'minutes': mins})

    if usage_data:
        df_u = pd.DataFrame(usage_data)
        median_usage = df_u['minutes'].median()
        light = df_u[df_u['minutes'] <= median_usage]['recall'].values
        heavy = df_u[df_u['minutes'] > median_usage]['recall'].values

        u, p, r = mann_whitney_r(light, heavy)
        rows.append({
            'comparison': 'tiktok_light_vs_heavy',
            'group_a': 'light', 'group_b': 'heavy',
            'n_a': len(light), 'n_b': len(heavy),
            'mean_a': round(float(np.mean(light)), 4) if len(light) > 0 else 0,
            'mean_b': round(float(np.mean(heavy)), 4) if len(heavy) > 0 else 0,
            'U': round(u, 2), 'p_value': float(p), 'effect_r': round(r, 4),
            'significant': p < 0.05,
        })
        print(f"  TikTok: light(n={len(light)}) vs heavy(n={len(heavy)}), p={p:.4f}")

    # ── 2. Sex ──
    sex_data = []
    for pid in recall_map:
        if pid in demo_map and 'sex' in demo_map[pid].index:
            sex = str(demo_map[pid]['sex']).strip().lower()
            if sex in ('m', 'f'):
                sex_data.append({'pid': pid, 'recall': recall_map[pid], 'sex': sex})

    if sex_data:
        df_s = pd.DataFrame(sex_data)
        male = df_s[df_s['sex'] == 'm']['recall'].values
        female = df_s[df_s['sex'] == 'f']['recall'].values

        u, p, r = mann_whitney_r(male, female)
        rows.append({
            'comparison': 'sex_male_vs_female',
            'group_a': 'male', 'group_b': 'female',
            'n_a': len(male), 'n_b': len(female),
            'mean_a': round(float(np.mean(male)), 4) if len(male) > 0 else 0,
            'mean_b': round(float(np.mean(female)), 4) if len(female) > 0 else 0,
            'U': round(u, 2), 'p_value': float(p), 'effect_r': round(r, 4),
            'significant': p < 0.05,
        })
        print(f"  Sex: male(n={len(male)}) vs female(n={len(female)}), p={p:.4f}")

    # ── 3. Paid vs unpaid cohort ──
    paid = [recall_map[pid] for pid in config.PAID_PARTICIPANTS if pid in recall_map]
    unpaid = [recall_map[pid] for pid in config.UNPAID_PARTICIPANTS if pid in recall_map]

    u, p, r = mann_whitney_r(np.array(paid), np.array(unpaid))
    rows.append({
        'comparison': 'paid_vs_unpaid',
        'group_a': 'paid', 'group_b': 'unpaid',
        'n_a': len(paid), 'n_b': len(unpaid),
        'mean_a': round(float(np.mean(paid)), 4) if paid else 0,
        'mean_b': round(float(np.mean(unpaid)), 4) if unpaid else 0,
        'U': round(u, 2), 'p_value': float(p), 'effect_r': round(r, 4),
        'significant': p < 0.05,
    })
    print(f"  Cohort: paid(n={len(paid)}) vs unpaid(n={len(unpaid)}), p={p:.4f}")

    df = pd.DataFrame(rows)
    df.to_csv(run_dir / "subgroup_comparisons.csv", index=False)
    print(f"\n  ✓ Subgroup comparisons saved.")
