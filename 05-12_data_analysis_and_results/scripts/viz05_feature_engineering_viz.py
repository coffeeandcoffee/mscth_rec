#!/usr/bin/env python3
"""viz05 — Feature engineering visualization. 5 detailed explorations."""

import numpy as np
import pickle
import pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

import config

PALETTE = sns.color_palette("pastel")
STAY_COLOR = PALETTE[0]
SKIP_COLOR = PALETTE[3]

def run(run_dir, params):
    viz_dir = run_dir / "viz"
    viz_dir.mkdir(exist_ok=True)
    features_dir = run_dir / "features"
    windows_dir = run_dir / "windows" / "primary"

    # Gather data for explorations
    all_features = []
    all_ei = []
    all_labels = []
    all_pids = []
    
    rep_pid = config.INCLUDED_PARTICIPANTS[0]
    rep_X = None
    rep_names = []
    
    for pid in config.INCLUDED_PARTICIPANTS:
        fp = features_dir / f"P{pid}.pkl"
        if not fp.exists(): continue
        with open(fp, 'rb') as f: fd = pickle.load(f)
        
        X = fd['features_full']
        ei = fd['ei_values']
        y = fd['labels']
        
        all_features.append(X)
        all_ei.append(ei)
        all_labels.append(y)
        all_pids.extend([f"P{pid}"] * len(y))
        
        if pid == rep_pid:
            rep_X = X
            rep_names = fd['agg_names_full']

    if not all_features:
        print("No feature data found for viz05.")
        return

    X_all = np.vstack(all_features)
    ei_all = np.concatenate(all_ei)
    y_all = np.concatenate(all_labels)
    pids_all = np.array(all_pids)
    feature_names = rep_names
    
    df_all = pd.DataFrame(X_all, columns=feature_names)
    df_all['EI'] = ei_all
    df_all['Label'] = y_all
    df_all['Class'] = df_all['Label'].map({1: 'STAY', 0: 'SKIP'})
    df_all['PID'] = pids_all

    # --- viz05a: Reality Check (Parametric Trace & Macro View) ---
    _plot_reality_check(viz_dir, rep_pid, windows_dir, features_dir)
    
    # --- viz05b: Engagement Index ---
    _plot_engagement_index(viz_dir, df_all)
    
    # --- viz05c: Band Power Means ---
    _plot_band_powers(viz_dir, df_all, feature_names)
    
    # --- viz05d: Statistical Moments ---
    _plot_statistical_moments(viz_dir, df_all, feature_names)

    # --- viz05e: Correlations ---
    _plot_correlations(viz_dir, rep_X, rep_pid)


def _plot_reality_check(viz_dir, rep_pid, windows_dir, features_dir):
    fig = plt.figure(figsize=(20, 12))
    fig.suptitle("Step 05a — Feature Engineering Reality Check", fontsize=18, fontweight='bold')
    
    win_pkl = windows_dir / f"P{rep_pid}.pkl"
    if not win_pkl.exists():
        plt.close()
        return
        
    with open(win_pkl, 'rb') as f: wd = pickle.load(f)
    windows = wd['windows']
    if not windows:
        plt.close()
        return

    # Panel A: Single Window Trace
    stay_idx = next((i for i, w in enumerate(windows) if w['label'] == 1), 0)
    w = windows[stay_idx]
    data = w['data']
    f_names = wd['feature_names']
    
    ax1 = fig.add_subplot(2, 3, 1)
    ax2 = fig.add_subplot(2, 3, 2)
    ax3 = fig.add_subplot(2, 3, 3)
    
    time_axis = np.arange(data.shape[0]) / 256.0
    
    # Find indices safely
    try:
        ch_alpha_idx = f_names.index(f"{config.EEG_CHANNELS[0]}_alpha")
        ch_beta_idx = f_names.index(f"{config.EEG_CHANNELS[0]}_beta")
        ch_theta_idx = f_names.index(f"{config.EEG_CHANNELS[0]}_theta")
        
        alpha_ts = data[:, ch_alpha_idx]
        beta_ts = data[:, ch_beta_idx]
        theta_ts = data[:, ch_theta_idx]

        ax1.plot(time_axis, alpha_ts, label='Alpha', color='green', alpha=0.7)
        ax1.plot(time_axis, beta_ts, label='Beta', color='red', alpha=0.7)
        ax1.plot(time_axis, theta_ts, label='Theta', color='purple', alpha=0.7)
        ax1.set_xlabel("Time (s) within window")
        ax1.set_ylabel("Band Power")
        ax1.set_title(f"Panel A1: Sample-Level Band Powers\n(Window {stay_idx}, {config.EEG_CHANNELS[0]})", fontsize=10)
        ax1.legend(fontsize=8)
        
        # Panel A2: Extracted Stats for Alpha
        mean_a, std_a, min_a, max_a = np.mean(alpha_ts), np.std(alpha_ts), np.min(alpha_ts), np.max(alpha_ts)
        ax2.bar(['Mean', 'Std', 'Min', 'Max'], [mean_a, std_a, min_a, max_a], color='green', alpha=0.6)
        ax2.set_title("Panel A2: Extracted Statistical Moments\n(Alpha Band)", fontsize=10)
        ax2.set_ylabel("Power")
        for i, v in enumerate([mean_a, std_a, min_a, max_a]):
            ax2.text(i, v, f"{v:.2f}", ha='center', va='bottom', fontsize=8)

        # Panel A3: EI Calculation
        alpha_means = [np.mean(data[:, f_names.index(f"{ch}_alpha")]) for ch in config.EEG_CHANNELS]
        beta_means = [np.mean(data[:, f_names.index(f"{ch}_beta")]) for ch in config.EEG_CHANNELS]
        theta_means = [np.mean(data[:, f_names.index(f"{ch}_theta")]) for ch in config.EEG_CHANNELS]
        
        avg_alpha = np.mean(alpha_means)
        avg_beta = np.mean(beta_means)
        avg_theta = np.mean(theta_means)
        ei_val = avg_beta / (avg_alpha + avg_theta) if (avg_alpha + avg_theta) > 0 else 0
        
        ax3.bar(['Avg Beta\n(Numerator)', 'Avg Alpha\n(Denom)', 'Avg Theta\n(Denom)'], 
                [avg_beta, avg_alpha, avg_theta], color=['red', 'green', 'purple'], alpha=0.6)
        ax3.set_title(f"Panel A3: Engagement Index Math\nEI = {avg_beta:.2f} / ({avg_alpha:.2f} + {avg_theta:.2f}) = {ei_val:.3f}", fontsize=10)
    except ValueError:
        ax1.text(0.5, 0.5, 'Band columns not found', ha='center', va='center')

    # Panel B: Continuous Macro View
    ax4 = fig.add_subplot(2, 1, 2)
    fp = features_dir / f"P{rep_pid}.pkl"
    if fp.exists():
        with open(fp, 'rb') as f: fd = pickle.load(f)
        ei_vals = fd['ei_values']
        labels = fd['labels']
        
        n_plot = min(150, len(ei_vals))
        x_plot = np.arange(n_plot)
        ax4.plot(x_plot, ei_vals[:n_plot], color='black', linewidth=1.5, marker='o', markersize=3)
        
        for i in range(n_plot):
            color = STAY_COLOR if labels[i] == 1 else SKIP_COLOR
            ax4.axvspan(i-0.5, i+0.5, color=color, alpha=0.3, lw=0)
            
        ax4.set_xlim(-0.5, n_plot - 0.5)
        ax4.set_xlabel("Window Index (Consecutive Time)")
        ax4.set_ylabel("Engagement Index (EI)")
        ax4.set_title("Panel B: Macro Trajectory of Features\nBackground color denotes class (Blue=STAY, Orange=SKIP). Shows how the feature evolves over time across phase changes.", fontsize=12)

        import matplotlib.patches as mpatches
        stay_patch = mpatches.Patch(color=STAY_COLOR, alpha=0.3, label='STAY Phase')
        skip_patch = mpatches.Patch(color=SKIP_COLOR, alpha=0.3, label='SKIP Phase')
        ax4.legend(handles=[stay_patch, skip_patch], loc='upper right')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(viz_dir / "viz05a_reality_check.png", dpi=200)
    plt.close()


def _plot_engagement_index(viz_dir, df):
    fig = plt.figure(figsize=(16, 8))
    fig.suptitle("Step 05b — Exploration: Engagement Index (EI)", fontsize=18, fontweight='bold')
    
    ax1 = fig.add_subplot(1, 2, 1)
    sns.violinplot(data=df, x='Class', y='EI', order=['STAY', 'SKIP'], 
                   palette=[STAY_COLOR, SKIP_COLOR], ax=ax1, inner="quartile")
    ax1.set_title("Global EI Distribution (All Participants)\nDoes EI separate classes on a population level?", fontsize=12)
    
    ax2 = fig.add_subplot(1, 2, 2)
    sns.boxplot(data=df, x='PID', y='EI', hue='Class', hue_order=['STAY', 'SKIP'],
                palette=[STAY_COLOR, SKIP_COLOR], ax=ax2, fliersize=1)
    ax2.set_title("Per-Participant EI Distribution\nDoes EI consistently separate classes for individuals?", fontsize=12)
    ax2.tick_params(axis='x', rotation=45)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(viz_dir / "viz05b_exploration_ei.png", dpi=200)
    plt.close()


def _plot_band_powers(viz_dir, df, feature_names):
    bands = [b[0] for b in config.FREQUENCY_BANDS]
    
    band_cols = {}
    for band in bands:
        band_mean_cols = [c for c in feature_names if f"_{band}_mean" in c]
        if band_mean_cols:
            df[f"{band}_avg_power"] = df[band_mean_cols].mean(axis=1)
            band_cols[band] = f"{band}_avg_power"
            
    if not band_cols:
        return

    fig = plt.figure(figsize=(18, 10))
    fig.suptitle("Step 05c — Exploration: Average Band Powers", fontsize=18, fontweight='bold')
    
    ax1 = fig.add_subplot(2, 1, 1)
    melted_global = df.melt(id_vars=['Class'], value_vars=list(band_cols.values()), var_name='Band', value_name='Power')
    melted_global['Band'] = melted_global['Band'].str.replace('_avg_power', '').str.capitalize()
    
    sns.barplot(data=melted_global, x='Band', y='Power', hue='Class', 
                hue_order=['STAY', 'SKIP'], palette=[STAY_COLOR, SKIP_COLOR], ax=ax1, errorbar='ci')
    ax1.set_title("Global Mean Power by Frequency Band\nWhich bands show the largest overall difference?", fontsize=12)
    
    ax2 = fig.add_subplot(2, 1, 2)
    diff_matrix = pd.DataFrame(index=df['PID'].unique(), columns=bands)
    for pid in df['PID'].unique():
        sub_stay = df[(df['PID'] == pid) & (df['Class'] == 'STAY')]
        sub_skip = df[(df['PID'] == pid) & (df['Class'] == 'SKIP')]
        for band in bands:
            col = band_cols.get(band)
            if col:
                diff = sub_stay[col].mean() - sub_skip[col].mean()
                pooled_std = np.sqrt((sub_stay[col].std()**2 + sub_skip[col].std()**2) / 2)
                diff_matrix.loc[pid, band] = diff / pooled_std if pooled_std > 0 else 0

    diff_matrix = diff_matrix.astype(float)
    sns.heatmap(diff_matrix.T, cmap='RdBu_r', center=0, ax=ax2, annot=True, fmt=".2f", 
                cbar_kws={'label': "Effect Size (Cohen's d)\nPositive = Higher in STAY"})
    ax2.set_title("Per-Participant Band Power Effect Sizes\nRows=Bands, Cols=Participants. Consistent red/blue rows indicate robust biomarkers.", fontsize=12)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(viz_dir / "viz05c_exploration_band_power.png", dpi=200)
    plt.close()


def _plot_statistical_moments(viz_dir, df, feature_names):
    stats = ['mean', 'std', 'min', 'max']
    results = []
    
    for c in feature_names:
        st = c.split('_')[-1]
        if st not in stats: continue
        
        stay_vals = df[df['Class'] == 'STAY'][c]
        skip_vals = df[df['Class'] == 'SKIP'][c]
        
        diff = stay_vals.mean() - skip_vals.mean()
        pooled_std = np.sqrt((stay_vals.std()**2 + skip_vals.std()**2) / 2)
        effect_size = diff / pooled_std if pooled_std > 0 else 0
        
        results.append({'Feature': c, 'Stat': st.capitalize(), 'EffectSize': abs(effect_size)})
        
    if not results: return
    df_res = pd.DataFrame(results)

    fig = plt.figure(figsize=(12, 8))
    fig.suptitle("Step 05d — Exploration: Statistical Moments", fontsize=18, fontweight='bold')
    
    ax1 = fig.add_subplot(1, 1, 1)
    sns.boxplot(data=df_res, x='Stat', y='EffectSize', ax=ax1, palette="Set2")
    sns.stripplot(data=df_res, x='Stat', y='EffectSize', ax=ax1, color=".3", size=3, alpha=0.5)
    
    ax1.set_title("Global Absolute Effect Sizes by Statistical Moment\nDo variations (Std, Min, Max) separate classes better than simple Means?", fontsize=12)
    ax1.set_ylabel("Absolute Effect Size |d| (STAY vs SKIP)")
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(viz_dir / "viz05d_exploration_stats.png", dpi=200)
    plt.close()


def _plot_correlations(viz_dir, rep_X, rep_pid):
    if rep_X is None: return
    
    fig = plt.figure(figsize=(24, 8))
    fig.suptitle("Step 05e — Feature Correlations & Space", fontsize=18, fontweight='bold')
    
    # Panel 1
    ax = fig.add_subplot(1, 3, 1)
    channels = config.EEG_CHANNELS
    bands = [b[0] for b in config.FREQUENCY_BANDS]
    stats = ['mean', 'std', 'min', 'max']
    n_stats = len(stats)
    n_feats = len(channels) * len(bands) * n_stats
    grid = np.ones((len(channels), len(bands)))
    ax.imshow(grid, cmap='Blues', alpha=0.3, aspect='auto')
    for i, ch in enumerate(channels):
        for j in range(len(bands)):
            color = PALETTE[0] if ch in config.FRONTAL_CHANNELS else PALETTE[1]
            ax.add_patch(plt.Rectangle((j-0.5, i-0.5), 1, 1, fill=True,
                                        facecolor=color, alpha=0.3, edgecolor='gray'))
            ax.text(j, i, str(n_stats), ha='center', va='center', fontsize=8, color='gray')
    ax.set_xticks(range(len(bands))); ax.set_xticklabels(bands, fontsize=7, rotation=45)
    ax.set_yticks(range(len(channels))); ax.set_yticklabels(channels, fontsize=9)
    ax.set_title(f'Feature Space Schematic\n{n_feats} features ({n_stats} stats × {len(bands)} bands × {len(channels)} ch)')
    
    # Panel 2
    ax2 = fig.add_subplot(1, 3, 2)
    if rep_X.shape[0] > 5:
        corr = np.corrcoef(rep_X.T)
        corr = np.nan_to_num(corr)
        im = ax2.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
        ax2.set_title(f'Feature Correlation Heatmap — P{rep_pid}\nReveals feature redundancy (dark red/blue = highly correlated)')
        plt.colorbar(im, ax=ax2, shrink=0.7)
    
        # Panel 3
        ax3 = fig.add_subplot(1, 3, 3, projection='3d')
        n = corr.shape[0]
        step = max(1, n // 40)
        corr_ds = corr[::step, ::step]
        n_ds = corr_ds.shape[0]
        X_grid, Y_grid = np.meshgrid(np.arange(n_ds), np.arange(n_ds))

        surf = ax3.plot_surface(X_grid, Y_grid, corr_ds,
                                 cmap='RdBu_r', vmin=-1, vmax=1,
                                 alpha=0.9, edgecolor='none',
                                 antialiased=True, rstride=1, cstride=1)
        ax3.set_zlim(-1, 1)
        ax3.set_title('Correlation Topography (3D View)', pad=8)
        ax3.view_init(elev=35, azim=-60)
        
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(viz_dir / "viz05e_correlations.png", dpi=200)
    plt.close()

if __name__ == "__main__":
    print("Use run.py to execute the pipeline.")
