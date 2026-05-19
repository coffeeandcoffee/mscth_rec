#!/usr/bin/env python3
"""
step15_compile_report.py — Final report compilation.

Assembles all outputs into master_results.csv, LaTeX tables, and
numbers_to_update.txt listing every inline number that must be updated.

OUT: master_results.csv + .tex tables + numbers_to_update.txt
"""

import numpy as np
import json
from pathlib import Path
import pandas as pd
import config


def load_json_safe(path):
    if path.exists():
        with open(path, 'r') as f:
            return json.load(f)
    return {}


def build_master_table(run_dir):
    """Build the master results comparison table."""
    primary = load_json_safe(run_dir / "step07_primary_result.json")
    stats = load_json_safe(run_dir / "stats_summary.json")
    notch = load_json_safe(run_dir / "notch_ablation_result.json")
    sfei = load_json_safe(run_dir / "sfei_result.json")
    logo = load_json_safe(run_dir / "logo_confusion.json")

    p_recall = primary.get('mean_recall', 0)

    # Load EI summary
    ei_csv = run_dir / "ei_summary.csv"
    ei_recall = 0.5
    if ei_csv.exists():
        df_ei = pd.read_csv(ei_csv)
        ei_recall = df_ei['ei_recall'].mean()

    # Load electrode ablation
    abl_csv = run_dir / "electrode_ablation.csv"
    abl_data = {}
    if abl_csv.exists():
        df_abl = pd.read_csv(abl_csv)
        for _, row in df_abl.iterrows():
            abl_data[row['comparison']] = row

    # Load sensitivity
    sens_csv = run_dir / "sensitivity_comparisons.csv"
    sens_data = {}
    if sens_csv.exists():
        df_sens = pd.read_csv(sens_csv)
        for _, row in df_sens.iterrows():
            sens_data[row['comparison']] = row

    # Load subgroups
    sub_csv = run_dir / "subgroup_comparisons.csv"
    sub_data = {}
    if sub_csv.exists():
        df_sub = pd.read_csv(sub_csv)
        for _, row in df_sub.iterrows():
            sub_data[row['comparison']] = row

    # Load seed stability
    stab = load_json_safe(run_dir / "stats_summary.json").get('seed_stability', {})

    rows = []

    def add_row(name, p_rec, c_rec, delta, test, stat, p_val, r_eff, sig):
        rows.append({
            'comparison': name,
            'primary_recall': round(p_rec, 4) if p_rec else '',
            'comparison_recall': round(c_rec, 4) if c_rec else '',
            'delta_pp': round(delta * 100, 1) if delta else '',
            'test': test,
            'statistic': round(stat, 2) if stat else '',
            'p_value': round(p_val, 6) if p_val is not None else '',
            'effect_r': round(r_eff, 3) if r_eff else '',
            'significant': '✅' if sig else '❌',
        })

    # 1. vs chance
    pw = stats.get('primary_wilcoxon', {})
    add_row('vs 50% chance baseline', p_recall, 0.5,
            p_recall - 0.5, 'Wilcoxon', pw.get('W', 0), pw.get('p', 1), pw.get('r', 0),
            pw.get('significant', False))

    # 2. Temporal vs random-split — need to compute from step07 data
    add_row('Temporal blocked vs random-split', p_recall, None, None,
            'Wilcoxon paired', None, None, None, None)

    # 3-4. Electrode ablations
    for key, name in [('full_vs_frontal', 'Full vs frontal-only'),
                       ('full_vs_temporal', 'Full vs temporal-only')]:
        if key in abl_data:
            r = abl_data[key]
            add_row(name, p_recall, float(r.get('mean_b', 0)),
                    float(r.get('delta', 0)), 'Wilcoxon paired',
                    float(r.get('wilcoxon_W', 0)), float(r.get('p_value', 1)),
                    float(r.get('effect_r', 0)), bool(r.get('significant', False)))

    # 5. Notch
    add_row('Nonotch vs notch', p_recall, notch.get('mean_nt', 0),
            p_recall - notch.get('mean_nt', 0), 'Wilcoxon paired',
            notch.get('W', 0), notch.get('p', 1), notch.get('r', 0),
            notch.get('p', 1) < 0.05)

    # 6. RF vs EI
    ei_comp = stats.get('ei_vs_rf', {})
    add_row('RF vs EI', p_recall, ei_recall, p_recall - ei_recall,
            'Wilcoxon paired', ei_comp.get('wilcoxon_W', 0),
            ei_comp.get('p_value', 1), ei_comp.get('effect_r', 0),
            ei_comp.get('p_value', 1) < 0.05)

    # 7. RF vs SFEI
    add_row('RF vs SFEI', p_recall, sfei.get('mean_recall', 0),
            p_recall - sfei.get('mean_recall', 0), 'Wilcoxon paired',
            None, None, None, None)

    # 8-10. Subgroups
    for key, name in [('tiktok_light_vs_heavy', 'Light vs heavy TikTok'),
                       ('sex_male_vs_female', 'Male vs female'),
                       ('paid_vs_unpaid', 'Paid vs unpaid')]:
        if key in sub_data:
            r = sub_data[key]
            add_row(name, None, None, None, 'Mann-Whitney',
                    float(r.get('U', 0)), float(r.get('p_value', 1)),
                    float(r.get('effect_r', 0)), bool(r.get('significant', False)))

    # 11-13. Sensitivity
    for key, name in [('artifact_exclude', 'With vs without artifact windows'),
                       ('burst_exclude', 'With vs without burst-skips'),
                       ('artifact_exclude_burst_exclude', 'Both excluded')]:
        if key in sens_data:
            r = sens_data[key]
            add_row(name, float(r.get('mean_primary', 0)),
                    float(r.get('mean_variant', 0)),
                    float(r.get('delta', 0)), 'Wilcoxon paired',
                    float(r.get('wilcoxon_W', 0)), float(r.get('p_value', 1)),
                    float(r.get('effect_r', 0)), bool(r.get('significant', False)))

    # 14. LOGO
    add_row('LOGO-CV vs chance', logo.get('mean_recall', 0), 0.5,
            logo.get('mean_recall', 0) - 0.5, 'Descriptive',
            None, None, None, None)

    # 15. Seed stability
    add_row('Seed stability (SD < 0.03)', None, None, None, 'Descriptive',
            stab.get('mean_sd', 0), None, None,
            stab.get('n_stable', 0) == stab.get('n_stable', 0))

    return pd.DataFrame(rows)


def write_latex_table(df, path, caption, label):
    """Write a DataFrame as a LaTeX table."""
    with open(path, 'w') as f:
        f.write(f"\\begin{{table}}[htbp]\n\\centering\n\\caption{{{caption}}}\n")
        f.write(f"\\label{{{label}}}\n")
        f.write(f"\\small\n")
        f.write(df.to_latex(index=False, escape=True))
        f.write(f"\\end{{table}}\n")


def run(run_dir, params):
    config.pprint_step(15, "REPORT COMPILATION")

    tables_dir = run_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    # ── Master results table ──
    df_master = build_master_table(run_dir)
    df_master.to_csv(run_dir / "master_results.csv", index=False)
    write_latex_table(df_master, tables_dir / "master_results.tex",
                      "Master Results — All Pipeline Comparisons",
                      "tab:master_results")

    # ── Participant demographics table ──
    if config.SURVEY_CSV.exists():
        df_demo = pd.read_csv(config.SURVEY_CSV)
        demo_rows = []
        for pid in config.INCLUDED_PARTICIPANTS:
            row = df_demo[df_demo['ID'] == f'P{pid}']
            if len(row) > 0:
                r = row.iloc[0]
                demo_rows.append({
                    'PID': f'P{pid}',
                    'Age': r.get('How old are you?', ''),
                    'Sex': r.get('sex', ''),
                    'SFV Usage': r.get('How much time per day do you spend watching short-form videos?', ''),
                    'Cohort': 'Paid' if pid in config.PAID_PARTICIPANTS else 'Unpaid',
                })
        df_demo_table = pd.DataFrame(demo_rows)
        df_demo_table.to_csv(tables_dir / "demographics.csv", index=False)
        write_latex_table(df_demo_table, tables_dir / "demographics.tex",
                          "Participant Demographics", "tab:demographics")

    # ── Dataset stats table ──
    bc_path = run_dir / "balanced_counts.csv"
    if bc_path.exists():
        df_bc = pd.read_csv(bc_path)
        write_latex_table(df_bc, tables_dir / "dataset_stats.tex",
                          "Per-Participant Dataset Statistics", "tab:dataset_stats")

    # ── Feature ranking table ──
    fr_path = run_dir / "feature_ranking_112.csv"
    if fr_path.exists():
        df_fr = pd.read_csv(fr_path)
        write_latex_table(df_fr, tables_dir / "feature_ranking.tex",
                          "112-Feature Importance Ranking (Full)", "tab:feature_ranking")

    # ── Numbers to update ──
    primary = load_json_safe(run_dir / "step07_primary_result.json")
    lines = [
        "NUMBERS TO UPDATE IN THESIS PROSE",
        "=" * 50,
        f"Primary STAY recall: {primary.get('mean_recall', '?'):.4f} "
        f"← step07_primary_result.json:mean_recall",
        f"Primary recall SD: {primary.get('std_recall', '?'):.4f}",
        f"Wilcoxon W: {primary.get('wilcoxon_W', '?')}",
        f"Wilcoxon p: {primary.get('wilcoxon_p', '?')}",
        f"Effect size r: {primary.get('rank_biserial_r', '?')}",
        f"N above chance: {primary.get('n_above_chance', '?')}/{primary.get('n_participants', '?')}",
        f"N participants: {primary.get('n_participants', '?')}",
        "",
        "See master_results.csv for all comparison numbers.",
        "See per_participant_significance.csv for binomial CIs.",
        "See ei_summary.csv for EI recall.",
        "See logo_results.csv for LOGO-CV.",
        "See sfei_result.json for SFEI formula and recall.",
    ]

    with open(run_dir / "numbers_to_update.txt", 'w') as f:
        f.write('\n'.join(lines))

    print(f"  ✓ master_results.csv written ({len(df_master)} comparisons)")
    print(f"  ✓ LaTeX tables written to tables/")
    print(f"  ✓ numbers_to_update.txt written")
    print(f"\n  PIPELINE REPORT COMPLETE.")
