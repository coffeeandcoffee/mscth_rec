#!/usr/bin/env python3
"""
step09_significance.py — Statistical significance testing.

Reads step07 results and balanced_counts.csv from step04.
Computes: Wilcoxon vs chance, per-participant binomial CI, seed stability,
electrode ablation pairwise tests, EI vs RF comparison.

OUT: stats_summary.csv + per_participant_significance.csv + seed_stability.csv + electrode_ablation.csv
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


def binomial_ci(n_correct, n_total, alpha=0.05):
    """Wilson score interval for binomial proportion."""
    if n_total == 0:
        return 0.0, 0.0, 1.0
    p_hat = n_correct / n_total
    z = sp_stats.norm.ppf(1 - alpha / 2)
    denom = 1 + z**2 / n_total
    centre = (p_hat + z**2 / (2 * n_total)) / denom
    margin = z * np.sqrt((p_hat * (1 - p_hat) + z**2 / (4 * n_total)) / n_total) / denom
    return max(0, centre - margin), min(1, centre + margin), p_hat


def rank_biserial_r(w_stat, n):
    """Compute rank-biserial effect size r from Wilcoxon W statistic."""
    if n == 0:
        return 0.0
    return 1 - (2 * w_stat) / (n * (n + 1) / 2)


def run(run_dir, params):
    """Entry point called by run.py."""
    config.pprint_step(9, "SIGNIFICANCE TESTING")

    results_dir = run_dir / "results" / "intra"
    seeds = params.get('step07', {}).get('seeds', [0, 1, 7, 42, 99])

    # Load balanced counts for correct N
    bc_path = run_dir / "balanced_counts.csv"
    balanced_counts = pd.read_csv(bc_path) if bc_path.exists() else None

    # Load primary result
    with open(run_dir / "step07_primary_result.json", 'r') as f:
        primary = json.load(f)

    # ── Collect per-participant data ──
    participant_data = []
    for pid in config.INCLUDED_PARTICIPANTS:
        pkl_path = results_dir / f"P{pid}_results.pkl"
        if not pkl_path.exists():
            continue
        with open(pkl_path, 'rb') as f:
            pdata = pickle.load(f)

        # Extract recalls per seed for full/frontal/temporal under temporal_blocked
        recalls_full = []
        recalls_frontal = []
        recalls_temporal = []
        for seed in seeds:
            r = pdata['results'].get('temporal_blocked', {})
            recalls_full.append(r.get('full', {}).get(seed, {}).get('recall', 0.5))
            recalls_frontal.append(r.get('frontal', {}).get(seed, {}).get('recall', 0.5))
            recalls_temporal.append(r.get('temporal', {}).get(seed, {}).get('recall', 0.5))

        participant_data.append({
            'pid': pid,
            'recall_full': np.mean(recalls_full),
            'recall_frontal': np.mean(recalls_frontal),
            'recall_temporal': np.mean(recalls_temporal),
            'recalls_per_seed': recalls_full,
            'seed_sd': np.std(recalls_full),
        })

    pids = [d['pid'] for d in participant_data]
    recalls_full = np.array([d['recall_full'] for d in participant_data])
    recalls_frontal = np.array([d['recall_frontal'] for d in participant_data])
    recalls_temporal = np.array([d['recall_temporal'] for d in participant_data])

    # ── 1. Primary Wilcoxon (confirmation) ──
    try:
        w, p_val = sp_stats.wilcoxon(recalls_full - 0.5)
        r_eff = rank_biserial_r(w, len(recalls_full))
    except Exception:
        w, p_val, r_eff = 0.0, 1.0, 0.0

    # ── 2. Per-participant binomial CI ──
    sig_rows = []
    for d in participant_data:
        pid = d['pid']
        if balanced_counts is not None:
            bc_row = balanced_counts[balanced_counts['pid'] == pid]
            n_total = int(bc_row['n_balanced_test_total'].values[0]) if len(bc_row) > 0 else 100
        else:
            n_total = 100

        recall = d['recall_full']
        n_correct = int(round(recall * n_total))
        ci_low, ci_high, p_hat = binomial_ci(n_correct, n_total)
        sig = ci_low > 0.5

        sig_rows.append({
            'pid': pid, 'recall': round(recall, 4),
            'n_total': n_total, 'n_correct': n_correct,
            'ci_low': round(ci_low, 4), 'ci_high': round(ci_high, 4),
            'above_chance': recall > 0.5,
            'significantly_above_chance': sig,
        })

    df_sig = pd.DataFrame(sig_rows)
    df_sig.to_csv(run_dir / "per_participant_significance.csv", index=False)
    n_sig = df_sig['significantly_above_chance'].sum()
    n_above = df_sig['above_chance'].sum()

    # ── 3. Seed stability ──
    stability_rows = []
    for d in participant_data:
        stability_rows.append({
            'pid': d['pid'],
            'seed_sd': round(d['seed_sd'], 4),
            'stable': d['seed_sd'] < 0.03,
        })

    df_stab = pd.DataFrame(stability_rows)
    df_stab.to_csv(run_dir / "seed_stability.csv", index=False)
    mean_sd = df_stab['seed_sd'].mean()
    n_stable = df_stab['stable'].sum()

    # ── 4. Electrode ablation pairwise tests ──
    ablation_rows = []
    pairs = [
        ('full_vs_frontal', recalls_full, recalls_frontal),
        ('full_vs_temporal', recalls_full, recalls_temporal),
        ('frontal_vs_temporal', recalls_frontal, recalls_temporal),
    ]
    for name, a, b in pairs:
        try:
            w_ab, p_ab = sp_stats.wilcoxon(a - b)
            r_ab = rank_biserial_r(w_ab, len(a))
        except Exception:
            w_ab, p_ab, r_ab = 0.0, 1.0, 0.0

        ablation_rows.append({
            'comparison': name,
            'mean_a': round(float(np.mean(a)), 4),
            'mean_b': round(float(np.mean(b)), 4),
            'delta': round(float(np.mean(a) - np.mean(b)), 4),
            'wilcoxon_W': round(float(w_ab), 2),
            'p_value': float(p_ab),
            'effect_r': round(float(r_ab), 4),
            'significant': p_ab < 0.05,
        })

    motor_flag = float(np.mean(recalls_temporal)) >= float(np.mean(recalls_full))
    ablation_rows.append({
        'comparison': 'MOTOR_ARTIFACT_FLAG',
        'mean_a': float(np.mean(recalls_temporal)),
        'mean_b': float(np.mean(recalls_full)),
        'delta': 0, 'wilcoxon_W': 0, 'p_value': 1.0, 'effect_r': 0,
        'significant': motor_flag,
    })

    df_abl = pd.DataFrame(ablation_rows)
    df_abl.to_csv(run_dir / "electrode_ablation.csv", index=False)

    # ── 5. EI vs RF ──
    ei_csv = run_dir / "ei_summary.csv"
    ei_comparison = {}
    if ei_csv.exists():
        df_ei = pd.read_csv(ei_csv)
        ei_recalls = []
        rf_recalls_matched = []
        for _, row in df_ei.iterrows():
            pid = row['pid']
            match = [d for d in participant_data if d['pid'] == pid]
            if match:
                ei_recalls.append(row['ei_recall'])
                rf_recalls_matched.append(match[0]['recall_full'])

        if len(ei_recalls) > 5:
            ei_arr = np.array(ei_recalls)
            rf_arr = np.array(rf_recalls_matched)
            try:
                w_ei, p_ei = sp_stats.wilcoxon(rf_arr - ei_arr)
                r_ei = rank_biserial_r(w_ei, len(ei_arr))
            except Exception:
                w_ei, p_ei, r_ei = 0.0, 1.0, 0.0
            ei_comparison = {
                'mean_rf': float(np.mean(rf_arr)),
                'mean_ei': float(np.mean(ei_arr)),
                'wilcoxon_W': float(w_ei),
                'p_value': float(p_ei),
                'effect_r': float(r_ei),
            }

    # ── Write stats summary ──
    summary = {
        'primary_wilcoxon': {'W': float(w), 'p': float(p_val), 'r': float(r_eff),
                             'mean_recall': float(np.mean(recalls_full)),
                             'significant': bool(p_val < 0.05)},
        'binomial_ci': {'n_above_chance': int(n_above),
                        'n_significantly_above': int(n_sig),
                        'n_total': len(participant_data)},
        'seed_stability': {'mean_sd': float(mean_sd),
                           'n_stable': int(n_stable),
                           'threshold': 0.03},
        'ei_vs_rf': ei_comparison,
    }

    with open(run_dir / "stats_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

    # Also CSV version
    stats_csv_rows = [
        {'test': 'Wilcoxon_vs_chance', 'statistic': w, 'p_value': p_val,
         'effect_r': float(r_eff), 'significant': bool(p_val < 0.05)},
    ]
    pd.DataFrame(stats_csv_rows).to_csv(run_dir / "stats_summary.csv", index=False)

    print(f"\n  Results:")
    print(f"    Wilcoxon vs 50%: W={w:.1f}, p={p_val:.6f}, r={r_eff:.3f}")
    print(f"    Above chance: {n_above}/{len(participant_data)}")
    print(f"    Significantly above (binomial CI): {n_sig}/{len(participant_data)}")
    print(f"    Seed stability: mean SD={mean_sd:.4f}, {n_stable}/{len(participant_data)} stable")
    if motor_flag:
        print(f"    ⚠ MOTOR ARTIFACT FLAG: temporal ≥ full recall")
    print(f"  ✓ All significance outputs saved.")


if __name__ == "__main__":
    print("Use run.py to execute the pipeline.")
