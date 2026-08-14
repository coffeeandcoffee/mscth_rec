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
import viz_style

PALETTE = sns.color_palette("pastel")
STAY_COLOR = PALETTE[0]
SKIP_COLOR = PALETTE[3]

def run(run_dir, params):
    viz_dir = run_dir / "viz"
    viz_dir.mkdir(exist_ok=True)
    features_dir = run_dir / "features"
    features_notch_dir = run_dir / "features_notch"
    windows_dir = run_dir / "windows" / "primary"

    all_features = []
    all_features_notch = []
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
        
        fp_notch = features_notch_dir / f"P{pid}.pkl"
        X_notch = X
        if fp_notch.exists():
            with open(fp_notch, 'rb') as f: fdn = pickle.load(f)
            X_notch = fdn['features_full']
            
        all_features.append(X)
        all_features_notch.append(X_notch)
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
    X_notch_all = np.vstack(all_features_notch)
    ei_all = np.concatenate(all_ei)
    y_all = np.concatenate(all_labels)
    pids_all = np.array(all_pids)
    feature_names = rep_names
    
    df_all = pd.DataFrame(X_all, columns=feature_names)
    df_all['EI'] = ei_all
    df_all['Label'] = y_all
    df_all['Class'] = df_all['Label'].map({1: 'STAY', 0: 'SKIP'})
    df_all['PID'] = pids_all

    df_notch = pd.DataFrame(X_notch_all, columns=feature_names)
    df_notch['Class'] = df_all['Class']
    df_notch['PID'] = df_all['PID']

    # --- viz05a: Reality Check (Parametric Trace & Macro View) ---
    _plot_reality_check(viz_dir, rep_pid, windows_dir, features_dir)
    
    # --- viz05b: Engagement Index ---
    _plot_engagement_index(viz_dir, df_all)
    
    # --- viz05c: Band Power Means ---
    _plot_band_powers(viz_dir, df_all, df_notch, feature_names)
    
    # --- viz05d: Statistical Moments ---
    _plot_statistical_moments(viz_dir, df_all, feature_names, params)

    # --- viz05e: Correlations ---
    _plot_correlations(viz_dir, rep_X, rep_pid)


def _plot_reality_check(viz_dir, rep_pid, windows_dir, features_dir):
    from scipy.signal import find_peaks
    fig = plt.figure(figsize=(20, 18))
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

    stay_idx = next((i for i, w in enumerate(windows) if w['label'] == 1), 0)
    w = windows[stay_idx]
    data = w['data']
    f_names = wd['feature_names']
    
    ax1 = fig.add_subplot(3, 3, 1)
    ax2 = fig.add_subplot(3, 3, 2)
    ax3 = fig.add_subplot(3, 3, 3)
    
    time_axis = np.arange(data.shape[0]) / 256.0
    
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
        
        # Draw the interrupted horizontal line for the custom threshold
        def plot_macro_threshold(ts, color, ax, t_axis):
            peaks, _ = find_peaks(ts)
            if len(peaks) == 0: return
            mean_amp = np.mean(ts[peaks])
            is_above = ts >= mean_amp
            
            false_lengths = []
            curr_len = 0
            for val in is_above:
                if not val: curr_len += 1
                elif curr_len > 0:
                    false_lengths.append(curr_len)
                    curr_len = 0
            if curr_len > 0: false_lengths.append(curr_len)
            gap_threshold = np.mean(false_lengths) if len(false_lengths) > 0 else 0
            
            is_above_smoothed = is_above.copy()
            last_true_idx = -1
            for i in range(len(is_above)):
                if is_above[i]:
                    if last_true_idx != -1:
                        gap = i - last_true_idx - 1
                        if 0 < gap <= gap_threshold:
                            is_above_smoothed[last_true_idx+1:i] = True
                    last_true_idx = i
            
            thresh_line = np.full_like(ts, np.nan)
            thresh_line[is_above_smoothed] = mean_amp
            ax.plot(t_axis, thresh_line, color=color, linestyle='--', linewidth=2, alpha=0.9)

        plot_macro_threshold(alpha_ts, 'green', ax1, time_axis)
        plot_macro_threshold(beta_ts, 'red', ax1, time_axis)
        plot_macro_threshold(theta_ts, 'purple', ax1, time_axis)

        ax1.set_xlabel("Time (s) within window")
        ax1.set_ylabel("Band Power")
        ax1.set_title(f"Panel A1: Sample-Level Band Powers\n(Window {stay_idx}, {config.EEG_CHANNELS[0]})", fontsize=10)
        ax1.legend(fontsize=8)
        
        ax1.text(0.5, -0.2, "Note: The features we extract capture the amplitude (y-axis) of these waves.", 
                 ha='center', va='top', transform=ax1.transAxes, fontsize=8, style='italic', bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))

        # Peak freq box for A1
        duration = len(alpha_ts) / 256.0
        
        def calc_macro(ts):
            peaks, _ = find_peaks(ts)
            if len(peaks) == 0: return len(peaks), 0
            mean_amp = np.mean(ts[peaks])
            is_above = ts >= mean_amp
            
            false_lengths = []
            curr_len = 0
            for val in is_above:
                if not val: curr_len += 1
                elif curr_len > 0:
                    false_lengths.append(curr_len)
                    curr_len = 0
            if curr_len > 0: false_lengths.append(curr_len)
            gap_threshold = np.mean(false_lengths) if len(false_lengths) > 0 else 0
            
            is_above_smoothed = is_above.copy()
            last_true_idx = -1
            for i in range(len(is_above)):
                if is_above[i]:
                    if last_true_idx != -1:
                        gap = i - last_true_idx - 1
                        if 0 < gap <= gap_threshold:
                            is_above_smoothed[last_true_idx+1:i] = True
                    last_true_idx = i
                    
            starts_high = 1 if is_above_smoothed[0] else 0
            return len(peaks), starts_high + np.sum((is_above_smoothed[:-1] == False) & (is_above_smoothed[1:] == True))

        raw_a, macro_a = calc_macro(alpha_ts)
        raw_b, macro_b = calc_macro(beta_ts)
        raw_t, macro_t = calc_macro(theta_ts)
        
        box_str = (f"Frequencies (Raw vs Above-Mean Blocks)\n"
                   f"Alpha: {raw_a} raw / {macro_a} macro peaks → {macro_a/duration:.1f} Hz\n"
                   f"Beta: {raw_b} raw / {macro_b} macro peaks → {macro_b/duration:.1f} Hz\n"
                   f"Theta: {raw_t} raw / {macro_t} macro peaks → {macro_t/duration:.1f} Hz")
        ax1.text(0.5, -0.35, box_str, ha='center', va='top', transform=ax1.transAxes, fontsize=8, bbox=dict(facecolor='lightyellow', alpha=0.8, edgecolor='orange'))
        
        # Export Beta sample map to text file
        import textwrap
        peaks_b_raw, _ = find_peaks(beta_ts)
        if len(peaks_b_raw) > 0:
            mean_amp_b = np.mean(beta_ts[peaks_b_raw])
            is_above_b = beta_ts >= mean_amp_b
            
            false_lengths = []
            curr_len = 0
            for val in is_above_b:
                if not val: curr_len += 1
                elif curr_len > 0:
                    false_lengths.append(curr_len)
                    curr_len = 0
            if curr_len > 0: false_lengths.append(curr_len)
            gap_threshold = np.mean(false_lengths) if len(false_lengths) > 0 else 0
            
            is_above_b_smoothed = is_above_b.copy()
            last_true_idx = -1
            for i in range(len(is_above_b)):
                if is_above_b[i]:
                    if last_true_idx != -1:
                        gap = i - last_true_idx - 1
                        if 0 < gap <= gap_threshold:
                            is_above_b_smoothed[last_true_idx+1:i] = True
                    last_true_idx = i
            
            beta_map_str = "".join(["i" if val else "x" for val in is_above_b_smoothed])
            beta_wrapped = "\n".join(textwrap.wrap(beta_map_str, width=120))
            box2_str = f"Beta Sample Map (x=under, i=over threshold. Total i-blocks = {macro_b}):\n\n{beta_wrapped}"
            
            with open(viz_dir / "viz05_beta_sample_map.txt", "w") as map_f:
                map_f.write(box2_str + "\n")

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

    # Panel B1: Continuous Macro View (EI)
    ax4 = fig.add_subplot(3, 1, 2)
    fp = features_dir / f"P{rep_pid}.pkl"
    if fp.exists():
        with open(fp, 'rb') as f: fd = pickle.load(f)
        ei_vals = fd['ei_values']
        labels = fd['labels']
        
        # Find first transition from SKIP to STAY
        stay_idx = next((i for i, l in enumerate(labels) if l == 1), 0)
        
        start_idx = max(0, stay_idx - 50)
        end_idx = min(len(ei_vals), stay_idx + 100)
        
        plot_ei = ei_vals[start_idx:end_idx]
        plot_labels = labels[start_idx:end_idx]
        x_plot = np.arange(start_idx, end_idx)
        
        ax4.plot(x_plot, plot_ei, color='black', linewidth=1.5, marker='o', markersize=3)
        
        for i in range(len(x_plot)):
            color = STAY_COLOR if plot_labels[i] == 1 else SKIP_COLOR
            ax4.axvspan(x_plot[i]-0.5, x_plot[i]+0.5, color=color, alpha=0.3, lw=0)
            
        ax4.set_xlim(start_idx - 0.5, end_idx - 0.5)
        ax4.set_xlabel("Window Index (Consecutive Time)")
        ax4.set_ylabel("Engagement Index (EI)")
        ax4.set_title(f"Panel B1: Macro Trajectory of EI across a Class Transition (Windows {start_idx} to {end_idx})\nBackground color denotes class (Blue=STAY, Orange=SKIP).", fontsize=12)

        import matplotlib.patches as mpatches
        stay_patch = mpatches.Patch(color=STAY_COLOR, alpha=0.3, label='STAY Phase')
        skip_patch = mpatches.Patch(color=SKIP_COLOR, alpha=0.3, label='SKIP Phase')
        ax4.legend(handles=[stay_patch, skip_patch], loc='upper right')

        # Panel B2: Macro Peak Frequency Trajectory
        ax5 = fig.add_subplot(3, 1, 3)
        features_all = fd['features_full']
        feature_names = fd['agg_names_full']
        
        try:
            ch = config.EEG_CHANNELS[0]
            pf_alpha_idx = feature_names.index(f"{ch}_alpha_macrofreq")
            pf_beta_idx = feature_names.index(f"{ch}_beta_macrofreq")
            pf_theta_idx = feature_names.index(f"{ch}_theta_macrofreq")
            
            pf_alpha = features_all[start_idx:end_idx, pf_alpha_idx]
            pf_beta = features_all[start_idx:end_idx, pf_beta_idx]
            pf_theta = features_all[start_idx:end_idx, pf_theta_idx]
            
            ax5.plot(x_plot, pf_alpha, color='green', linewidth=1.5, marker='o', markersize=3, label='Alpha Macro Freq')
            ax5.plot(x_plot, pf_beta, color='red', linewidth=1.5, marker='o', markersize=3, label='Beta Macro Freq')
            ax5.plot(x_plot, pf_theta, color='purple', linewidth=1.5, marker='o', markersize=3, label='Theta Macro Freq')
            
            for i in range(len(x_plot)):
                color = STAY_COLOR if plot_labels[i] == 1 else SKIP_COLOR
                ax5.axvspan(x_plot[i]-0.5, x_plot[i]+0.5, color=color, alpha=0.3, lw=0)
                
            ax5.set_xlim(start_idx - 0.5, end_idx - 0.5)
            ax5.set_xlabel("Window Index (Consecutive Time)")
            ax5.set_ylabel("Macro Frequency (Hz)")
            ax5.set_title(f"Panel B2: Macro Trajectory of Macro Frequencies across the same transition", fontsize=12)
            ax5.legend(loc='upper right')
        except ValueError:
            ax5.text(0.5, 0.5, "Macrofreq features missing. Please re-run step05.", ha='center', va='center')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(viz_dir / "viz05a_reality_check.png", dpi=200)
    plt.close()

    # --- Individual Panel A2 ---
    # Each quantity is a single scalar, so it is marked with a rule at its value
    # rather than a bar; a bar would imply an area that carries no meaning.
    a2_labels = ['Mean', 'Std', 'Min', 'Max']
    a2_values = [mean_a, std_a, min_a, max_a]
    fig_ind, ax_ind = plt.subplots(figsize=(9, 6))
    for i, v in enumerate(a2_values):
        ax_ind.hlines(v, i - 0.3, i + 0.3, colors='green', linewidth=5)
        ax_ind.text(i, v, f"{v:.2f}", ha='center', va='bottom',
                    fontsize=viz_style.FONT_2X)
    ax_ind.axhline(0, color='gray', linewidth=1, zorder=0)
    ax_ind.set_xticks(range(len(a2_labels)))
    ax_ind.set_xticklabels(a2_labels)
    ax_ind.set_xlim(-0.6, len(a2_labels) - 0.4)
    ax_ind.margins(y=0.15)
    viz_style.style_axes(
        ax_ind, viz_style.FONT_2X,
        title="Panel A2: Extracted Statistical Moments\n(Alpha Band)",
        xlabel="Statistical Moment",
        ylabel="Power",
    )
    plt.tight_layout()
    plt.savefig(viz_dir / "viz05a.A2.png", dpi=200)
    plt.close()

    # --- Individual Panel A3 ---
    a3_labels = ['Avg Beta\n(Numerator)', 'Avg Alpha\n(Denom)', 'Avg Theta\n(Denom)']
    a3_values = [avg_beta, avg_alpha, avg_theta]
    a3_colors = ['red', 'green', 'purple']
    fig_ind, ax_ind = plt.subplots(figsize=(9, 6))
    for i, (v, c) in enumerate(zip(a3_values, a3_colors)):
        ax_ind.hlines(v, i - 0.3, i + 0.3, colors=c, linewidth=5)
        ax_ind.text(i, v, f"{v:.2f}", ha='center', va='bottom',
                    fontsize=viz_style.FONT_2X)
    ax_ind.axhline(0, color='gray', linewidth=1, zorder=0)
    ax_ind.set_xticks(range(len(a3_labels)))
    ax_ind.set_xticklabels(a3_labels)
    ax_ind.set_xlim(-0.6, len(a3_labels) - 0.4)
    ax_ind.margins(y=0.15)
    viz_style.style_axes(
        ax_ind, viz_style.FONT_2X,
        title=f"Panel A3: Engagement Index Math\nEI = {avg_beta:.2f} / ({avg_alpha:.2f} + {avg_theta:.2f}) = {ei_val:.3f}",
        xlabel="Band Powers",
    )
    plt.tight_layout()
    plt.savefig(viz_dir / "viz05a.A3.png", dpi=200)
    plt.close()



def _plot_engagement_index(viz_dir, df):
    fig = plt.figure(figsize=(16, 16))
    fig.suptitle("Step 05b — Exploration: Engagement Index (EI)", fontsize=18, fontweight='bold')
    
    # Top Row: RAW (Including massive outliers)
    ax1 = fig.add_subplot(2, 2, 1)
    sns.violinplot(data=df, x='Class', y='EI', order=['STAY', 'SKIP'], 
                   hue='Class', palette=[STAY_COLOR, SKIP_COLOR], ax=ax1, inner="quartile", legend=False)
    ax1.set_title("Global EI Distribution [RAW]\n(Extreme outliers due to near-zero Alpha+Theta)", fontsize=12)
    
    ax2 = fig.add_subplot(2, 2, 2)
    sns.boxplot(data=df, x='PID', y='EI', hue='Class', hue_order=['STAY', 'SKIP'],
                palette=[STAY_COLOR, SKIP_COLOR], ax=ax2, fliersize=1)
    ax2.set_title("Per-Participant EI Distribution [RAW]", fontsize=12)
    ax2.tick_params(axis='x', rotation=45)

    # Bottom Row: ZOOMED IN (95th percentile)
    p95 = np.percentile(df['EI'].dropna(), 95)
    
    ax3 = fig.add_subplot(2, 2, 3)
    sns.violinplot(data=df, x='Class', y='EI', order=['STAY', 'SKIP'], 
                   palette=[STAY_COLOR, SKIP_COLOR], ax=ax3, inner="quartile")
    ax3.set_ylim(0, p95)
    ax3.set_title("Global EI Distribution [ZOOMED IN]\n(Y-axis capped at 95th percentile)", fontsize=12)

    ax4 = fig.add_subplot(2, 2, 4)
    sns.boxplot(data=df, x='PID', y='EI', hue='Class', hue_order=['STAY', 'SKIP'],
                palette=[STAY_COLOR, SKIP_COLOR], ax=ax4, fliersize=1)
    ax4.set_ylim(0, p95)
    ax4.set_title("Per-Participant EI Distribution [ZOOMED IN]", fontsize=12)
    ax4.tick_params(axis='x', rotation=45)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(viz_dir / "viz05b_exploration_ei.png", dpi=200)
    plt.close()


def _plot_band_powers(viz_dir, df, df_notch, feature_names):
    bands = [b[0] for b in config.FREQUENCY_BANDS]
    
    band_cols = {}
    for band in bands:
        band_mean_cols = [c for c in feature_names if f"_{band}_mean" in c]
        if band_mean_cols:
            df[f"{band}_avg_power"] = df[band_mean_cols].mean(axis=1)
            df_notch[f"{band}_avg_power"] = df_notch[band_mean_cols].mean(axis=1)
            band_cols[band] = f"{band}_avg_power"
            
    if not band_cols:
        return

    fig = plt.figure(figsize=(18, 15))
    fig.suptitle("Step 05c — Exploration: Average Band Powers\n(Regular vs Notched Data)", fontsize=18, fontweight='bold')
    
    # 1. Global (Nonotch)
    ax1 = fig.add_subplot(3, 1, 1)
    melted_global = df.melt(id_vars=['Class'], value_vars=list(band_cols.values()), var_name='Band', value_name='Power')
    melted_global['Band'] = melted_global['Band'].str.replace('_avg_power', '').str.capitalize()
    sns.barplot(data=melted_global, x='Band', y='Power', hue='Class', 
                hue_order=['STAY', 'SKIP'], palette=[STAY_COLOR, SKIP_COLOR], ax=ax1, errorbar='ci')
    ax1.set_title("Global Mean Power by Frequency Band (Un-Notched Data)", fontsize=12)

    # 2. Global (Notch)
    ax2 = fig.add_subplot(3, 1, 2)
    melted_notch = df_notch.melt(id_vars=['Class'], value_vars=list(band_cols.values()), var_name='Band', value_name='Power')
    melted_notch['Band'] = melted_notch['Band'].str.replace('_avg_power', '').str.capitalize()
    sns.barplot(data=melted_notch, x='Band', y='Power', hue='Class', 
                hue_order=['STAY', 'SKIP'], palette=[STAY_COLOR, SKIP_COLOR], ax=ax2, errorbar='ci')
    ax2.set_title("Global Mean Power by Frequency Band (Notched Data)\nNotice the huge drop in High Gamma and Very High power!", fontsize=12)

    # 3. Effect sizes
    ax3 = fig.add_subplot(3, 1, 3)
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
    sns.heatmap(diff_matrix.T, cmap='RdBu_r', center=0, ax=ax3, annot=True, fmt=".2f", 
                cbar_kws={'label': "Effect Size (Cohen's d)\nPositive = Higher in STAY"})
    ax3.set_title("Per-Participant Band Power Effect Sizes (Un-Notched)", fontsize=12)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(viz_dir / "viz05c_exploration_band_power.png", dpi=200)
    plt.close()


def _plot_statistical_moments(viz_dir, df, feature_names, params):
    bands = [b[0] for b in config.FREQUENCY_BANDS]
    results = []
    
    for c in feature_names:
        st = None
        for band in bands:
            if f"_{band}_" in c:
                parts = c.split(f"_{band}_")
                if len(parts) == 2:
                    st = parts[1]
                break
        
        # If not matched to a band, maybe it's a raw ERP feature
        if not st and '_raw_' in c:
            st = 'Raw ' + c.split('_raw_')[1].capitalize()
            
        if not st: 
            continue
        
        stay_vals = df[df['Class'] == 'STAY'][c]
        skip_vals = df[df['Class'] == 'SKIP'][c]
        
        diff = stay_vals.mean() - skip_vals.mean()
        pooled_std = np.sqrt((stay_vals.std()**2 + skip_vals.std()**2) / 2)
        effect_size = diff / pooled_std if pooled_std > 0 else 0
        
        results.append({'Feature': c, 'Stat': st.capitalize() if not st.startswith('Raw') else st, 'EffectSize': abs(effect_size)})
        
    # Add EI manually as the "first" block concept
    if 'EI' in df.columns:
        stay_vals = df[df['Class'] == 'STAY']['EI']
        skip_vals = df[df['Class'] == 'SKIP']['EI']
        diff = stay_vals.mean() - skip_vals.mean()
        pooled_std = np.sqrt((stay_vals.std()**2 + skip_vals.std()**2) / 2)
        effect_size = diff / pooled_std if pooled_std > 0 else 0
        results.append({'Feature': 'Engagement Index', 'Stat': 'Engagement Index\n(Classic Ratio)', 'EffectSize': abs(effect_size)})

    if not results: return
    df_res = pd.DataFrame(results)

    fig = plt.figure(figsize=(16, 8))
    
    title_str = "Step 05d — Conclusion: Comparing the separation power of all statistical features"
    p_exp = params.get('experimental', config.DEFAULT_PARAMS.get('experimental', {}))
    toggles = []
    if p_exp.get('remove_min_max', False): toggles.append("Min/Max Removed")
    if p_exp.get('extract_erp_features', False): toggles.append("ERP Features ON")
    if toggles: title_str += f"\n[{' | '.join(toggles)}]"
    
    fig.suptitle(title_str, fontsize=18, fontweight='bold')
    
    ax1 = fig.add_subplot(1, 1, 1)
    sns.boxplot(data=df_res, x='Stat', y='EffectSize', ax=ax1, hue='Stat', palette="Set2", legend=False)
    sns.stripplot(data=df_res, x='Stat', y='EffectSize', ax=ax1, color=".3", size=3, alpha=0.5)
    ax1.tick_params(axis='x', rotation=45)
    
    ax1.set_title("Global Absolute Effect Sizes by Statistical Moment", fontsize=12)
    ax1.set_ylabel("Absolute Effect Size |d| (STAY vs SKIP)")
    
    # Interpretation text box
    text_str = (
        "How to Read This Graph:\n\n"
        "• The Y-axis is Effect Size (|d|). It measures how well a feature separates the STAY and SKIP classes.\n"
        "• 0.2 = Weak separation | 0.5 = Moderate separation | 0.8 = Strong separation.\n"
        "• Each dot is a specific feature (e.g., 'Alpha Mean', 'Beta Hjorth_act').\n\n"
        "Takeaway:\n"
        "This compares all extracted feature families (Mean, Peak Frequency, Hjorth parameters, etc.)\n"
        "to see which mathematical transformation provides the strongest class separation."
    )
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax1.text(0.02, 0.95, text_str, transform=ax1.transAxes, fontsize=10,
             verticalalignment='top', bbox=props)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(viz_dir / "viz05d_exploration_stats.png", dpi=200)
    plt.close()


def _plot_correlations(viz_dir, rep_X, rep_pid):
    if rep_X is None: return
    
    fig = plt.figure(figsize=(20, 16))
    fig.suptitle("Step 05e — Feature Correlations & Redundancy", fontsize=18, fontweight='bold')
    
    # Calculate Correlation
    corr = np.corrcoef(rep_X.T)
    corr = np.nan_to_num(corr)
    n = corr.shape[0]

    # Panel 1: Full Heatmap
    ax1 = fig.add_subplot(2, 2, 1)
    channels = config.EEG_CHANNELS
    bands = [b[0] for b in config.FREQUENCY_BANDS]
    stats = ['mean', 'std', 'min', 'max', 'peakfreq', 'macrofreq']
    n_stats = len(stats)
    im1 = ax1.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    ax1.set_title(f'Full Feature Correlation Heatmap — P{rep_pid}\n(Dark red/blue = highly redundant)')
    plt.colorbar(im1, ax=ax1, shrink=0.7)

    # Panel 2: Masked Heatmap (Remove trivial correlations)
    # Trivial = same channel, or same band. Let's just create a block mask.
    # We have features grouped by ch -> band -> stat.
    mask = np.ones_like(corr)
    # Just an approximation: zero out the diagonal and near-diagonal blocks
    block_size = len(bands) * n_stats
    for i in range(len(channels)):
        mask[i*block_size:(i+1)*block_size, i*block_size:(i+1)*block_size] = 0
    
    corr_masked = corr.copy()
    corr_masked[mask == 0] = 0
    
    ax2 = fig.add_subplot(2, 2, 2)
    im2 = ax2.imshow(corr_masked, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    ax2.set_title('Masked Heatmap (Cross-Channel Correlations Only)\nHighlights "Surprising" Long-Range Dependencies')
    plt.colorbar(im2, ax=ax2, shrink=0.7)

    # Panel 3: Interpretation Box
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.axis('off')
    text_str = (
        "Why Do We Care About Correlation?\n\n"
        "1. Redundancy: The dark red diagonal line in the top-left plot is 'trivial' because\n"
        "   every feature is 100% correlated with itself. The thick red blocks around the diagonal\n"
        "   show that features from the same electrode (like AF7 Alpha Mean vs AF7 Alpha Std) are\n"
        "   highly redundant. They contain the exact same underlying information.\n\n"
        "2. Dimensionality Reduction: If 80% of our 112 features are highly correlated, we can safely\n"
        "   drop them later using techniques like PCA or Feature Selection. This makes the Machine\n"
        "   Learning model faster and less prone to overfitting, without losing actual insights.\n\n"
        "3. Surprising Links: The top-right plot hides intra-channel correlations to expose\n"
        "   'long-range' correlations—for example, if Frontal Beta strongly synchronizes with Temporal Theta."
    )
    ax3.text(0.1, 0.5, text_str, fontsize=12, va='center', ha='left',
             bbox=dict(boxstyle='round', facecolor='white', alpha=1.0, edgecolor='gray'))

    # Panel 4: 3D Topography
    ax4 = fig.add_subplot(2, 2, 4, projection='3d')
    step = max(1, n // 40)
    corr_ds = corr[::step, ::step]
    n_ds = corr_ds.shape[0]
    X_grid, Y_grid = np.meshgrid(np.arange(n_ds), np.arange(n_ds))

    surf = ax4.plot_surface(X_grid, Y_grid, corr_ds,
                             cmap='RdBu_r', vmin=-1, vmax=1,
                             alpha=0.9, edgecolor='none',
                             antialiased=True, rstride=1, cstride=1)
    ax4.set_zlim(-1, 1)
    ax4.set_title('Correlation Topography (3D View)', pad=8)
    ax4.view_init(elev=35, azim=-60)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(viz_dir / "viz05e_correlations.png", dpi=200)
    plt.close()

if __name__ == "__main__":
    print("Use run.py to execute the pipeline.")
