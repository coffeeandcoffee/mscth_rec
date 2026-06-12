#!/usr/bin/env python3
"""
step17_thesis_figures_and_tables.py

Aggregates data across the four pipeline iterations and generates
the final tables and figures for the thesis text.
Calculates statistical metrics from raw participant-level files to 
ensure exactly one code path and maximum comparability.
"""

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from scipy import stats as sp_stats

# -------------------------------------------------------------------------
# CONSTANTS & SETUP
# -------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

RUN_DIRS = {
    "R1": BASE_DIR / "runs" / "run_20260525_115853 +-0.5s",
    "R2": BASE_DIR / "runs" / "run_20260526_113959 +-0.5s (rerun)",
    "R3": BASE_DIR / "runs" / "run_20260526_133703 smaller RF",
    "R4": BASE_DIR / "runs" / "run_20260526_150106 3 True"
}

OUT_DIR = BASE_DIR / "thesis_outputs"
TAB_DIR = OUT_DIR / "tables"
FIG_DIR = OUT_DIR / "figures"

TAB_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Formatting for LaTeX style
plt.style.use('default')
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.grid'] = True
mpl.rcParams['grid.alpha'] = 0.3
mpl.rcParams['axes.axisbelow'] = True

PRIMARY_MODEL = "Intra|NoNotch|NoArt|NoBurst|RF"
EI_MODEL = "Intra|NoNotch|NoArt|NoBurst|EI"

def format_pval(p):
    if p < 0.0001:
        return "<0.0001"
    elif p < 0.001:
        return "<0.001"
    else:
        return f"{p:.4f}"

def format_pval_sci(p):
    if p < 0.001:
        return f"{p:.2e}"
    return f"{p:.4f}"

def get_stars(p):
    if p < 0.0001: return "****"
    if p < 0.001: return "***"
    if p < 0.01: return "**"
    if p < 0.05: return "*"
    return "n.s."

def rank_biserial_r(w_stat, n):
    if n == 0: return 0.0
    return 1 - (2 * w_stat) / (n * (n + 1) / 2)

def df_to_latex(df, filepath, caption="", label="", float_format="%.4f"):
    tex = df.to_latex(index=False, float_format=float_format, escape=False, 
                      caption=caption, label=label, column_format='c'*len(df.columns))
    
    # Simple fix for booktabs
    tex = tex.replace('\\toprule', '\\toprule') # Already there if pandas used booktabs? Usually it uses hline by default.
    tex = tex.replace('\\hline', '')
    
    # Add begin/end table* if wide
    tex = "\\begin{table*}[hptb]\n\\small\n\\centering\n" + tex + "\\end{table*}\n"
    
    with open(filepath, 'w') as f:
        f.write(tex)

# -------------------------------------------------------------------------
# HELPER: LOAD DATA ACROSS RUNS
# -------------------------------------------------------------------------
metrics = {}
for run, path in RUN_DIRS.items():
    csv_path = path / "parallel_universe_metrics.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        metrics[run] = df
    else:
        print(f"WARNING: Missing {csv_path}")

# -------------------------------------------------------------------------
# TABLE 1: PER-PARTICIPANT ALL RUNS
# -------------------------------------------------------------------------
print("Generating Table 1...")
tab1_rows = []
pids = sorted(metrics["R1"]['pid'].unique()) if "R1" in metrics else []

# Load extra files for R4
bc_r4 = pd.read_csv(RUN_DIRS["R4"] / "balanced_counts.csv")
sig_r4 = pd.read_csv(RUN_DIRS["R4"] / "per_participant_significance.csv")

for pid in pids:
    row = {'PID': f"P{pid:02d}"}
    row['n_skip'] = bc_r4.loc[bc_r4['pid']==pid, 'n_skip_raw'].values[0]
    
    for run in ["R1", "R2", "R3", "R4"]:
        df = metrics[run]
        m = df[(df['pid']==pid) & (df['combination']==PRIMARY_MODEL)]
        if len(m) > 0:
            test_a = m['test_accuracy'].values[0]
            train_a = m['train_accuracy'].values[0]
            row[f'{run} Test Acc'] = test_a
            row[f'{run} Train Acc'] = train_a
            row[rf'{run} \Delta Acc'] = train_a - test_a
            if run == "R4":
                row['R4 Test F1-Score'] = m['test_f1'].values[0]
        else:
            row[f'{run} Test Acc'] = np.nan
            row[f'{run} Train Acc'] = np.nan
            row[rf'{run} \Delta Acc'] = np.nan
            if run == "R4": row['R4 Test F1-Score'] = np.nan

    s = sig_r4[sig_r4['pid']==pid]
    if len(s) > 0:
        low, high = s['ci_low'].values[0], s['ci_high'].values[0]
        row['R4 95% CI'] = f"[{low:.3f}, {high:.3f}]"
        row['R4 Sig.↑'] = "✓" if s['significantly_above_chance'].values[0] else "—"
    else:
        row['R4 95% CI'] = ""
        row['R4 Sig.↑'] = ""
        
    tab1_rows.append(row)

df_tab1 = pd.DataFrame(tab1_rows)
# Add footer
means = df_tab1.drop(['PID', 'R4 95% CI', 'R4 Sig.↑'], axis=1).mean()
sds = df_tab1.drop(['PID', 'R4 95% CI', 'R4 Sig.↑'], axis=1).std()

footer = {'PID': 'Mean ± SD', 'R4 95% CI': '', 'R4 Sig.↑': ''}
for col in means.index:
    if 'n_skip' in col:
        footer[col] = f"{means[col]:.1f} ± {sds[col]:.1f}"
    else:
        footer[col] = f"{means[col]:.3f} ± {sds[col]:.3f}"

df_tab1.loc[len(df_tab1)] = footer

df_tab1.to_csv(TAB_DIR / "table1_per_participant_all_runs.csv", index=False)
df_to_latex(df_tab1, TAB_DIR / "table1_per_participant_all_runs.tex", 
            caption="Per-participant classification performance across four pipeline iterations.", 
            label="tab:table1_per_participant")

# -------------------------------------------------------------------------
# TABLE 2: AGGREGATE PROGRESSION & TABLE 3: STATS (Aggregated)
# -------------------------------------------------------------------------
print("Generating Table 2 and preparing Table 3...")
tab2_rows = []
tab3_rows = []

run_labels = {
    "R1": "R1 Baseline",
    "R2": "R2 Non-Overlap",
    "R3": "R3 Regularised RF",
    "R4": "R4 Full Signal"
}

for run, path in RUN_DIRS.items():
    with open(path / "parameters.json") as f:
        params = json.load(f)
        
    df = metrics[run]
    df_prim = df[df['combination'] == PRIMARY_MODEL]
    df_ei = df[df['combination'] == EI_MODEL]
    
    mean_train_acc = df_prim['train_accuracy'].mean()
    mean_test_acc = df_prim['test_accuracy'].mean()
    mean_test_rec = df_prim['test_f1'].mean()
    mean_ei_rec = df_ei['test_f1'].mean() if not df_ei.empty else np.nan
    
    # Wilcoxon vs chance
    f1s = df_prim['test_f1'].values
    if len(f1s) > 0:
        w_chance, p_chance = sp_stats.wilcoxon(f1s - 0.5)
        r_chance = rank_biserial_r(w_chance, len(f1s))
    else:
        w_chance, p_chance, r_chance = np.nan, np.nan, np.nan
        
    tab3_rows.append({
        'Test': 'RF (primary) vs 50% chance',
        'Run': run,
        'W': f"{w_chance:.1f}", 'p': format_pval_sci(p_chance), 'r': f"{r_chance:.3f}",
        'Sig': '✓' if p_chance < 0.05 else '—',
        'Direction': 'RF > Chance' if np.mean(f1s) > 0.5 else '—'
    })
    
    # RF vs EI paired
    if len(f1s) > 0 and len(df_ei) > 0:
        ei_f1s = []
        rf_f1s = []
        for pid in pids:
            r_rf = df_prim[df_prim['pid']==pid]['test_f1'].values
            r_ei = df_ei[df_ei['pid']==pid]['test_f1'].values
            if len(r_rf) > 0 and len(r_ei) > 0:
                rf_f1s.append(r_rf[0])
                ei_f1s.append(r_ei[0])
        rf_f1s = np.array(rf_f1s)
        ei_f1s = np.array(ei_f1s)
        w_ei, p_ei = sp_stats.wilcoxon(rf_f1s - ei_f1s)
        r_ei_val = rank_biserial_r(w_ei, len(rf_f1s))
        
        tab3_rows.append({
            'Test': 'RF vs EI (paired)', 'Run': run,
            'W': f"{w_ei:.1f}", 'p': format_pval_sci(p_ei), 'r': f"{r_ei_val:.3f}",
            'Sig': '✓' if p_ei < 0.05 else '—',
            'Direction': 'RF > EI' if np.mean(rf_f1s) > np.mean(ei_f1s) else 'EI > RF'
        })
    else:
        p_ei = np.nan
        
    # Notch ablation
    notch_csv = path / "notch_ablation.csv"
    if notch_csv.exists():
        ndf = pd.read_csv(notch_csv)
        delta_mean = (ndf['f1_nonotch'] - ndf['f1_notch']).mean()
        w_notch, p_notch = sp_stats.wilcoxon(ndf['f1_nonotch'] - ndf['f1_notch'])
        r_notch = rank_biserial_r(w_notch, len(ndf))
        
        tab3_rows.append({
            'Test': 'NoNotch vs Notch (paired)', 'Run': run,
            'W': f"{w_notch:.1f}", 'p': format_pval_sci(p_notch), 'r': f"{r_notch:.3f}",
            'Sig': '✓' if p_notch < 0.05 else '—',
            'Direction': 'NoNotch > Notch' if delta_mean > 0 else 'Notch > NoNotch'
        })
    else:
        delta_mean, w_notch, p_notch, r_notch = np.nan, np.nan, np.nan, np.nan
        
    # LOGO
    logo_csv = path / "logo_results.csv"
    if logo_csv.exists():
        ldf = pd.read_csv(logo_csv)
        logo_acc = (ldf['cm_tp'].sum() + ldf['cm_tn'].sum()) / (ldf['cm_tp'].sum() + ldf['cm_tn'].sum() + ldf['cm_fp'].sum() + ldf['cm_fn'].sum())
        logo_rec = ldf['f1'].mean()
    else:
        logo_acc, logo_rec = np.nan, np.nan

    tab2_rows.append({
        'Run': run_labels[run],
        'Skip window': f"±{params.get('step01', {}).get('skip_window_s', '?')} s",
        'Window / stride': f"{params.get('step03', {}).get('window_size_s', '?')} s / {params.get('step03', {}).get('stride_s', '?')} s",
        'RF config': f"Depth {params.get('step06', {}).get('max_depth', '?')} N={params.get('step06', {}).get('n_estimators', '?')}",
        'Mean Train Acc': f"{mean_train_acc:.3f}",
        'Mean Test Acc': f"{mean_test_acc:.3f}",
        'ΔAcc (train−test)': f"{(mean_train_acc - mean_test_acc):.3f}",
        'Mean Test F1-Score': f"{mean_test_rec:.3f}",
        'Mean EI F1-Score': f"{mean_ei_rec:.3f}" if not np.isnan(mean_ei_rec) else "—",
        'Wilcoxon vs chance (W, p, r)': f"W={w_chance:.0f}, p={format_pval(p_chance)}, r={r_chance:.3f}",
        'RF vs EI (p)': format_pval(p_ei) if not np.isnan(p_ei) else "—",
        'NoNotch vs Notch (Δ, p)': f"Δ={delta_mean:+.3f}, p={format_pval(p_notch)}" if not np.isnan(p_notch) else "—",
        'LOGO-CV Acc': f"{logo_acc:.3f}",
        'LOGO-CV F1-Score': f"{logo_rec:.3f}"
    })

# Add ablation/subgroup rows to Tab3
for run, abl_f, label in [("R2", "electrode_ablation.csv", "Full vs frontal-only electrode set"),
                          ("R2", "electrode_ablation.csv", "Full vs temporal-only electrode set"),
                          ("R4", "electrode_ablation.csv", "Full vs frontal-only electrode set"),
                          ("R4", "electrode_ablation.csv", "Full vs temporal-only electrode set"),
                          ("R2", "sensitivity_comparisons.csv", "Artifact exclusion effect"),
                          ("R2", "sensitivity_comparisons.csv", "Burst exclusion effect"),
                          ("R3", "sensitivity_comparisons.csv", "Artifact exclusion effect"),
                          ("R3", "sensitivity_comparisons.csv", "Burst exclusion effect"),
                          ("R1", "subgroup_comparisons.csv", "Paid vs unpaid subgroup"),
                          ("R2", "subgroup_comparisons.csv", "Paid vs unpaid subgroup")]:
    p = RUN_DIRS[run] / abl_f
    if p.exists():
        df_abl = pd.read_csv(p)
        if "electrode_ablation" in abl_f:
            comp_key = "full_vs_frontal" if "frontal" in label else "full_vs_temporal"
            row = df_abl[df_abl['comparison'] == comp_key]
        elif "sensitivity" in abl_f:
            comp_key = "artifact_exclude" if "Artifact" in label else "burst_exclude"
            row = df_abl[df_abl['comparison'] == comp_key]
        else:
            row = df_abl[df_abl['comparison'] == "paid_vs_unpaid"]
            
        if len(row) > 0:
            row = row.iloc[0]
            # Use 'wilcoxon_W' for electrode/sensitivity, 'U' for subgroup
            stat_name = 'wilcoxon_W' if 'wilcoxon_W' in row else 'U'
            tab3_rows.append({
                'Test': label, 'Run': run,
                'W': f"{row[stat_name]:.1f}", 'p': format_pval_sci(row['p_value']), 'r': f"{row['effect_r']:.3f}",
                'Sig': '✓' if row['p_value'] < 0.05 else '—',
                'Direction': "Significant" if row['p_value'] < 0.05 else "—"
            })

df_tab2 = pd.DataFrame(tab2_rows)
df_tab2.to_csv(TAB_DIR / "table2_aggregate_progression.csv", index=False)
df_to_latex(df_tab2, TAB_DIR / "table2_aggregate_progression.tex", 
            caption="Aggregate pipeline progression summary.", 
            label="tab:table2_aggregate")

df_tab3 = pd.DataFrame(tab3_rows)
df_tab3.to_csv(TAB_DIR / "table3_statistical_tests.csv", index=False)
df_to_latex(df_tab3, TAB_DIR / "table3_statistical_tests.tex", 
            caption="Summary of all statistical tests.", 
            label="tab:table3_stats")


# -------------------------------------------------------------------------
# TABLE 4: LOGO-CV (R3 & R4)
# -------------------------------------------------------------------------
print("Generating Table 4 (LOGO-CV)...")
for run in ["R3", "R4"]:
    logo_csv = RUN_DIRS[run] / "logo_results.csv"
    if logo_csv.exists():
        df = pd.read_csv(logo_csv)
        df['PID'] = df['test_pid'].apply(lambda x: f"P{x:02d}")
        df = df[['PID', 'n_test', 'n_train', 'accuracy', 'f1', 'cm_tn', 'cm_fp', 'cm_fn', 'cm_tp']]
        df = df.rename(columns={'accuracy': 'LOGO Accuracy', 'f1': 'LOGO F1-Score (STAY)', 
                                'cm_tn': 'TN', 'cm_fp': 'FP', 'cm_fn': 'FN', 'cm_tp': 'TP'})
        
        # Add footer
        mean_acc = (df['TP'].sum() + df['TN'].sum()) / (df['TP'].sum() + df['TN'].sum() + df['FP'].sum() + df['FN'].sum())
        mean_rec = df['LOGO F1-Score (STAY)'].mean()
        sd_acc = df['LOGO Accuracy'].std()
        sd_rec = df['LOGO F1-Score (STAY)'].std()
        
        df.loc[len(df)] = {
            'PID': 'Mean ± SD',
            'n_test': '', 'n_train': '',
            'LOGO Accuracy': f"{mean_acc:.3f} ± {sd_acc:.3f}",
            'LOGO F1-Score (STAY)': f"{mean_rec:.3f} ± {sd_rec:.3f}",
            'TN': '', 'FP': '', 'FN': '', 'TP': ''
        }
        
        df.to_csv(TAB_DIR / f"table4_logo_cv_{run.lower()}.csv", index=False)
        df_to_latex(df, TAB_DIR / f"table4_logo_cv_{run.lower()}.tex", 
                    caption=f"LOGO-CV per-participant results for {run}.", 
                    label=f"tab:table4_logo_{run.lower()}")


# -------------------------------------------------------------------------
# TABLE 5 & APPENDIX A: Feature Importance
# -------------------------------------------------------------------------
print("Generating Feature Importance Tables...")
def parse_feature_name(name):
    # E.g. TP9_delta_macrofreq
    parts = name.split('_')
    if len(parts) >= 3 and parts[0] in ['TP9', 'TP10', 'AF7', 'AF8']:
        electrode = parts[0]
        # Band could be 'low_gamma', 'high_gamma', etc.
        if 'gamma' in name or 'high' in name:
            band = parts[1] + '_' + parts[2]
        else:
            band = parts[1]
        return electrode, band
    return "", ""

for run in ["R2", "R3", "R4"]:
    fi_csv = RUN_DIRS[run] / "feature_ranking_112.csv"
    if fi_csv.exists():
        df = pd.read_csv(fi_csv)
        df['Electrode'] = df['feature'].apply(lambda x: parse_feature_name(x)[0])
        df['Band'] = df['feature'].apply(lambda x: parse_feature_name(x)[1])
        df = df[['rank', 'feature', 'mean_importance', 'std_importance', 'Electrode', 'Band']]
        df = df.rename(columns={'rank': 'Rank', 'feature': 'Feature', 'mean_importance': 'Mean Gini', 'std_importance': 'SD'})
        
        # Save full to Appendix A
        df.to_csv(TAB_DIR / f"appendix_a_feature_importance_{run.lower()}_full.csv", index=False)
        df_to_latex(df, TAB_DIR / f"appendix_a_feature_importance_{run.lower()}_full.tex", 
                    caption=f"Full feature importance ranking for {run}.", 
                    label=f"tab:app_a_{run.lower()}")
        
        # Table 5 is Top 20 for R4
        if run == "R4":
            df_top20 = df.head(20)
            df_top20.to_csv(TAB_DIR / "table5_feature_importance_r4_top20.csv", index=False)
            df_to_latex(df_top20, TAB_DIR / "table5_feature_importance_r4_top20.tex", 
                        caption="Top-20 feature importance ranking for final pipeline (R4).", 
                        label="tab:table5_feature_r4")


# -------------------------------------------------------------------------
# FIGURES
# -------------------------------------------------------------------------

# ----- Figure 1: Intra-subject progression -----
print("Generating Figure 1...")
runs = ["R1", "R2", "R3", "R4"]
labels = ["R1\nBaseline", "R2\nNon-Overlap", "R3\nRegularised RF", "R4\nFull Signal"]
x_pos = np.arange(len(runs))

mean_accs, sd_accs, pvals_accs = [], [], []
mean_recs, sd_recs, pvals_recs = [], [], []
mean_gaps, sd_gaps = [], []
mean_dummies = []

for run in runs:
    df = metrics[run]
    df = df[df['combination'] == PRIMARY_MODEL]
    accs = df['test_accuracy'].values
    recs = df['test_f1'].values
    gaps = df['train_accuracy'].values - df['test_accuracy'].values
    if 'dummy_test_f1' in df.columns:
        mean_dummies.append(np.mean(df['dummy_test_f1'].values))
    else:
        mean_dummies.append(0.5)
    
    mean_accs.append(np.mean(accs))
    sd_accs.append(np.std(accs))
    
    mean_recs.append(np.mean(recs))
    sd_recs.append(np.std(recs))
    
    mean_gaps.append(np.mean(gaps))
    sd_gaps.append(np.std(gaps))
    
    _, p_rec = sp_stats.wilcoxon(recs - 0.5)
    pvals_recs.append(p_rec)

fig, axes = plt.subplots(3, 1, figsize=(6, 8), sharex=True)

# A: Accuracy
axes[0].bar(x_pos, mean_accs, yerr=sd_accs, capsize=5, color='royalblue', alpha=0.8)
axes[0].axhline(0.5, color='k', linestyle='--', label='Chance')
axes[0].set_ylim(0.45, 0.75)
axes[0].set_ylabel("Mean Test Accuracy")
for i, p in enumerate(pvals_recs): # Note: spec asks for significance based on f1 W test, or just general? Usually it's f1, but let's just annotate the stars from f1 test as specified.
    axes[0].text(i, mean_accs[i] + sd_accs[i] + 0.02, get_stars(p), ha='center', va='bottom', fontsize=12)

# B: F1-Score
axes[1].bar(x_pos, mean_recs, yerr=sd_recs, capsize=5, color='mediumseagreen', alpha=0.8)
axes[1].plot(x_pos, mean_dummies, 'k--', label='Coin Flip F1')
axes[1].set_ylim(0.45, 0.85)
axes[1].set_ylabel("Mean Test F1-Score")
for i, p in enumerate(pvals_recs):
    axes[1].text(i, mean_recs[i] + sd_recs[i] + 0.02, get_stars(p), ha='center', va='bottom', fontsize=12)

# C: Gap
bars = axes[2].bar(x_pos, mean_gaps, yerr=sd_gaps, capsize=5, alpha=0.8)
for i, b in enumerate(bars):
    b.set_color('indianred' if mean_gaps[i] < 0 else 'steelblue') # Positive gap = train better
axes[2].axhline(0.0, color='k', linestyle='--', label='No gap')
axes[2].set_ylim(-0.60, 0.45)
axes[2].set_ylabel("Train − Test Accuracy")
axes[2].set_xticks(x_pos)
axes[2].set_xticklabels(labels)

plt.tight_layout()
fig.savefig(FIG_DIR / "fig1_intra_progression.pdf")
fig.savefig(FIG_DIR / "fig1_intra_progression.png", dpi=300)
plt.close(fig)


# ----- Figure 2: LOGO progression -----
print("Generating Figure 2...")
logo_accs, logo_sd_accs = [], []
logo_recs, logo_sd_recs = [], []
logo_dummy_f1s = []

for run in runs:
    logo_csv = RUN_DIRS[run] / "logo_results.csv"
    if logo_csv.exists():
        df = pd.read_csv(logo_csv)
        acc = (df['cm_tp'].sum() + df['cm_tn'].sum()) / (df['cm_tp'].sum() + df['cm_tn'].sum() + df['cm_fp'].sum() + df['cm_fn'].sum())
        rec = df['f1'].mean()
        
        logo_accs.append(acc)
        logo_sd_accs.append(df['accuracy'].std())
        logo_recs.append(rec)
        logo_sd_recs.append(df['f1'].std())
        if 'dummy_test_f1' in df.columns:
            logo_dummy_f1s.append(df['dummy_test_f1'].mean())
        else:
            logo_dummy_f1s.append(0.5)
    else:
        logo_accs.append(0.5)
        logo_sd_accs.append(0.0)
        logo_recs.append(0.5)
        logo_sd_recs.append(0.0)
        logo_dummy_f1s.append(0.5)

fig, axes = plt.subplots(1, 2, figsize=(8, 4))
axes[0].errorbar(x_pos, logo_accs, yerr=logo_sd_accs, fmt='-o', color='royalblue', capsize=5)
axes[0].axhline(0.5, color='k', linestyle='--')
axes[0].set_ylim(0.50, 0.75)
axes[0].set_ylabel("LOGO-CV Mean Accuracy")
axes[0].set_xticks(x_pos)
axes[0].set_xticklabels(labels)

axes[1].errorbar(x_pos, logo_recs, yerr=logo_sd_recs, fmt='-o', color='mediumseagreen', capsize=5)
axes[1].plot(x_pos, logo_dummy_f1s, 'k--', label='Coin Flip F1')
axes[1].set_ylim(0.40, 1.00)
axes[1].set_ylabel("LOGO-CV Mean STAY F1-Score")
axes[1].set_xticks(x_pos)
axes[1].set_xticklabels(labels)

plt.tight_layout()
fig.savefig(FIG_DIR / "fig2_logo_progression.pdf")
fig.savefig(FIG_DIR / "fig2_logo_progression.png", dpi=300)
plt.close(fig)


# ----- Figure 3: R1 vs R4 scatter -----
print("Generating Figure 3...")
r1_df = metrics["R1"][metrics["R1"]['combination'] == PRIMARY_MODEL].set_index('pid')
r4_df = metrics["R4"][metrics["R4"]['combination'] == PRIMARY_MODEL].set_index('pid')
sig_df = pd.read_csv(RUN_DIRS["R4"] / "per_participant_significance.csv").set_index('pid')

fig, ax = plt.subplots(figsize=(5, 5))
ax.plot([0.3, 0.9], [0.3, 0.9], 'k--', zorder=1, label="No change")

for pid in pids:
    if pid in r1_df.index and pid in r4_df.index:
        x = r1_df.loc[pid, 'test_accuracy']
        y = r4_df.loc[pid, 'test_accuracy']
        
        if pid in sig_df.index:
            above = sig_df.loc[pid, 'above_chance']
            sig = sig_df.loc[pid, 'significantly_above_chance']
        else:
            above, sig = False, False
            
        if sig:
            color = 'blue'
            fc = 'blue'
        elif above:
            color = 'blue'
            fc = 'none'
        else:
            color = 'grey'
            fc = 'grey'
            
        ax.scatter(x, y, edgecolor=color, facecolor=fc, zorder=2, s=50)

ax.set_xlim(0.40, 0.80)
ax.set_ylim(0.40, 0.80)
ax.set_xlabel("Run 1 (Baseline) Test Accuracy")
ax.set_ylabel("Run 4 (Final) Test Accuracy")
ax.grid(False)
plt.tight_layout()
fig.savefig(FIG_DIR / "fig3_r1_vs_r4_scatter.pdf")
fig.savefig(FIG_DIR / "fig3_r1_vs_r4_scatter.png", dpi=300)
plt.close(fig)


# ----- Figure 4: Heatmap -----
print("Generating Figure 4...")
fig, axes = plt.subplots(2, 2, figsize=(10, 8))
axes = axes.flatten()

electrodes = ["TP9", "AF7", "AF8", "TP10"]
bands = ["delta", "theta", "alpha", "beta", "low_gamma", "high_gamma", "very_high"]
band_labels = ["δ", "θ", "α", "β", "Lγ", "Hγ", "VH"]

for i, run in enumerate(runs):
    ax = axes[i]
    hm_file = RUN_DIRS[run] / "electrode_band_heatmap.json"
    if hm_file.exists():
        with open(hm_file) as f:
            data = json.load(f)
            
        mat = np.zeros((len(electrodes), len(bands)))
        for r, elec in enumerate(electrodes):
            for c, band in enumerate(bands):
                mat[r, c] = data.get(f"{elec}_{band}", 0.0)
                
        # Normalise within panel [0, 1]
        vmax = np.max(mat) if np.max(mat) > 0 else 1.0
        mat_norm = mat / vmax
        
        sns.heatmap(mat_norm, ax=ax, cmap="Blues", annot=mat, fmt=".3f", 
                    cbar=False, xticklabels=band_labels, yticklabels=electrodes,
                    annot_kws={"size": 8})
        ax.set_title(labels[i].replace('\n', ' - '))
    else:
        ax.text(0.5, 0.5, "Data Missing", ha='center', va='center')

plt.tight_layout()
fig.savefig(FIG_DIR / "fig4_electrode_band_heatmap.pdf")
fig.savefig(FIG_DIR / "fig4_electrode_band_heatmap.png", dpi=300)
plt.close(fig)


# ----- Figure 5: Notch Ablation R4 -----
print("Generating Figure 5...")
notch_csv = RUN_DIRS["R4"] / "notch_ablation.csv"
if notch_csv.exists():
    df = pd.read_csv(notch_csv)
    df = df.sort_values('delta', ascending=False).reset_index(drop=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    
    # Paired dot plot
    ax = axes[0]
    for i, row in df.iterrows():
        nn = row['f1_nonotch']
        nt = row['f1_notch']
        color = 'royalblue' if nn > nt else 'grey'
        ax.plot([0, 1], [nn, nt], color=color, alpha=0.6, marker='o')
        
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["NoNotch", "Notch"])
    ax.set_ylim(0.30, 1.00)
    ax.set_ylabel("STAY-class Test F1-Score")
    
    # Delta histogram
    ax = axes[1]
    ax.hist(df['delta'], bins=10, color='royalblue', edgecolor='black', alpha=0.8)
    ax.axvline(0, color='k', linestyle='dashed')
    ax.set_xlabel("Delta (NoNotch - Notch)")
    
    mean_delta = df['delta'].mean()
    w_notch, p_notch = sp_stats.wilcoxon(df['delta'])
    r_notch = rank_biserial_r(w_notch, len(df))
    ax.annotate(f"Mean Δ = {mean_delta:+.3f}\nW={w_notch:.1f}, p={format_pval_sci(p_notch)}", 
                xy=(0.05, 0.95), xycoords='axes fraction', va='top', ha='left')
    
    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig5_notch_ablation_r4.pdf")
    fig.savefig(FIG_DIR / "fig5_notch_ablation_r4.png", dpi=300)
    plt.close(fig)

# ----- Appendix A: Feature Rankings Bar Charts -----
print("Generating Appendix A Figures...")
for run in ["R2", "R3", "R4"]:
    fi_csv = RUN_DIRS[run] / "feature_ranking_112.csv"
    if fi_csv.exists():
        df = pd.read_csv(fi_csv).head(20)
        df = df.sort_values('mean_importance', ascending=True) # Ascending for horizontal bar
        
        fig, ax = plt.subplots(figsize=(6, 8))
        ax.barh(df['feature'], df['mean_importance'], xerr=df['std_importance'], 
                capsize=3, color='steelblue')
        ax.set_xlabel("Mean Gini Importance")
        ax.set_title(f"Top 20 Features ({run})")
        plt.tight_layout()
        
        out_name = f"appendix_a_fig_a{run[-1]}_feature_ranking_{run.lower()}"
        fig.savefig(FIG_DIR / f"{out_name}.pdf")
        fig.savefig(FIG_DIR / f"{out_name}.png", dpi=300)
        plt.close(fig)

print("✓ All tables and figures generated successfully in thesis_outputs/")
