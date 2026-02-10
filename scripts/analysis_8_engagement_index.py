#!/usr/bin/env python3
"""
Analysis V8: Engagement Index (EI) Comparison

Computes the standard Muse Engagement Index per class:
    EI = beta / (alpha + theta)

Averaged across all 4 electrodes for each participant, then compared
between about_to_skip vs not_about_to_skip using statistical tests.

Usage:
    python scripts/analysis_8_engagement_index.py --nonotch
    python scripts/analysis_8_engagement_index.py --file <specific_skip_labels.csv>
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import signal, stats
import json
from datetime import datetime
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

# Constants
EEG_CHANNELS = ['TP9', 'AF7', 'AF8', 'TP10']
FREQUENCY_BANDS = {
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta':  (13, 30),
}

PARTICIPANT_LABELS = {
    'eeg_20251210_203221': 'P1',
    'eeg_20251224_164549': 'P2',
    'eeg_20251227_190056': 'P3',
}


# ============================================================================
# SIGNAL PROCESSING
# ============================================================================

def apply_notch_filter(data, fs, notch_freq=50.0, quality_factor=30.0):
    """Apply notch filter to remove power line interference."""
    if len(data) < 20:
        return data
    try:
        b, a = signal.iirnotch(notch_freq, quality_factor, fs)
        return signal.filtfilt(b, a, data)
    except Exception:
        return data


def extract_band_power(data, fs, low_freq, high_freq):
    """Extract power in a specific frequency band using bandpass filter."""
    nyquist = fs / 2
    low = max(low_freq / nyquist, 0.01)
    high = min(high_freq / nyquist, 0.99)

    if low >= high or len(data) < 20:
        return np.zeros_like(data)

    try:
        b, a = signal.butter(4, [low, high], btype='band')
        filtered = signal.filtfilt(b, a, data)
        return filtered ** 2  # power = amplitude squared
    except Exception:
        return np.zeros_like(data)


# ============================================================================
# ENGAGEMENT INDEX COMPUTATION
# ============================================================================

def compute_ei_per_window(df, fs, window_seconds=3.0, notch_freq=None):
    """
    Compute Engagement Index for each windowed sample per class.

    For each window:
      1. Compute band power (theta, alpha, beta) per electrode
      2. Average across 4 electrodes
      3. EI = beta / (alpha + theta)

    Returns dict: { 'about_to_skip': [ei_values], 'not_about_to_skip': [ei_values] }
    """
    # Optionally apply notch filter
    if notch_freq is not None:
        print(f"   Applying {notch_freq}Hz notch filter...")
        df_f = df.copy()
        for ch in EEG_CHANNELS:
            df_f[ch] = apply_notch_filter(df[ch].values, fs, notch_freq)
    else:
        print(f"   ⏭ Notch filter DISABLED")
        df_f = df

    timestamps = df_f['lsl_timestamp'].values
    classification = df_f['classification_2'].values
    keypress_A = df_f['keypress_A'].values

    ei_by_class = {'about_to_skip': [], 'not_about_to_skip': []}

    def _compute_ei_for_block(block_df):
        """Compute EI for a single time window, averaged across electrodes."""
        band_powers = {band: [] for band in FREQUENCY_BANDS}

        for ch in EEG_CHANNELS:
            ch_data = block_df[ch].values
            for band_name, (lo, hi) in FREQUENCY_BANDS.items():
                power = extract_band_power(ch_data, fs, lo, hi)
                band_powers[band_name].append(np.mean(power))

        # Average across 4 electrodes
        theta_avg = np.mean(band_powers['theta'])
        alpha_avg = np.mean(band_powers['alpha'])
        beta_avg  = np.mean(band_powers['beta'])

        denom = alpha_avg + theta_avg
        if denom < 1e-12:
            return None
        return beta_avg / denom

    # --- about_to_skip: 3s window ending at each keypress_A ---
    keypress_indices = np.where(keypress_A == 1)[0]
    for kp_idx in keypress_indices:
        kp_time = timestamps[kp_idx]
        window_start = kp_time - window_seconds
        mask = (timestamps >= window_start) & (timestamps < kp_time)
        block_indices = np.where(mask)[0]
        if len(block_indices) < 10:
            continue
        ei = _compute_ei_for_block(df_f.iloc[block_indices])
        if ei is not None:
            ei_by_class['about_to_skip'].append(ei)

    # --- not_about_to_skip: sliding windows with 80% overlap ---
    not_skip_mask = classification == 'not_about_to_skip'
    not_skip_indices = np.where(not_skip_mask)[0]

    if len(not_skip_indices) > 0:
        breaks = np.where(np.diff(not_skip_indices) > 1)[0] + 1
        regions = np.split(not_skip_indices, breaks)

        for region in regions:
            if len(region) < 10:
                continue
            region_ts = timestamps[region]
            if (region_ts[-1] - region_ts[0]) < window_seconds:
                continue

            stride = window_seconds * 0.2  # 80% overlap
            t = region_ts[0]
            t_end = region_ts[-1] - window_seconds

            while t <= t_end:
                mask = (timestamps >= t) & (timestamps < t + window_seconds) & not_skip_mask
                block_indices = np.where(mask)[0]
                if len(block_indices) >= 10:
                    ei = _compute_ei_for_block(df_f.iloc[block_indices])
                    if ei is not None:
                        ei_by_class['not_about_to_skip'].append(ei)
                t += stride

    return ei_by_class


# ============================================================================
# STATISTICAL TESTING
# ============================================================================

def run_stats(ei_skip, ei_no_skip):
    """Run Mann-Whitney U and independent t-test."""
    results = {}

    # Descriptive
    results['skip_mean'] = float(np.mean(ei_skip))
    results['skip_std'] = float(np.std(ei_skip))
    results['skip_median'] = float(np.median(ei_skip))
    results['skip_n'] = len(ei_skip)

    results['no_skip_mean'] = float(np.mean(ei_no_skip))
    results['no_skip_std'] = float(np.std(ei_no_skip))
    results['no_skip_median'] = float(np.median(ei_no_skip))
    results['no_skip_n'] = len(ei_no_skip)

    results['delta_mean'] = results['skip_mean'] - results['no_skip_mean']

    # Mann-Whitney U (non-parametric, no normality assumption)
    u_stat, u_p = stats.mannwhitneyu(ei_skip, ei_no_skip, alternative='two-sided')
    results['mannwhitney_U'] = float(u_stat)
    results['mannwhitney_p'] = float(u_p)

    # Independent t-test (parametric)
    t_stat, t_p = stats.ttest_ind(ei_skip, ei_no_skip, equal_var=False)
    results['ttest_t'] = float(t_stat)
    results['ttest_p'] = float(t_p)

    # Cohen's d effect size
    pooled_std = np.sqrt((np.std(ei_skip)**2 + np.std(ei_no_skip)**2) / 2)
    if pooled_std > 1e-12:
        results['cohens_d'] = float(results['delta_mean'] / pooled_std)
    else:
        results['cohens_d'] = 0.0

    return results


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_ei_comparison(all_results, output_dir):
    """Bar chart: EI per class per participant + aggregate."""
    participants = list(all_results.keys())
    n = len(participants)

    fig, axes = plt.subplots(1, n + 1, figsize=(4 * (n + 1), 5), sharey=True)
    if n + 1 == 1:
        axes = [axes]

    colors = ['#e74c3c', '#2ecc71']  # skip = red, no-skip = green
    labels = ['About to Skip', 'Not About to Skip']

    for i, pid in enumerate(participants):
        r = all_results[pid]['stats']
        means = [r['skip_mean'], r['no_skip_mean']]
        stds = [r['skip_std'], r['no_skip_std']]
        p_val = r['mannwhitney_p']

        bars = axes[i].bar(labels, means, yerr=stds, color=colors, alpha=0.85,
                           capsize=5, edgecolor='black', linewidth=0.5)
        axes[i].set_title(f"{pid}\n(p={p_val:.4f})", fontsize=11)
        axes[i].set_ylabel('Engagement Index' if i == 0 else '')

        # Significance star
        if p_val < 0.001:
            sig = '***'
        elif p_val < 0.01:
            sig = '**'
        elif p_val < 0.05:
            sig = '*'
        else:
            sig = 'n.s.'

        y_max = max(means) + max(stds) * 1.2
        axes[i].text(0.5, y_max, sig, ha='center', fontsize=14, fontweight='bold',
                     transform=axes[i].get_xaxis_transform())

    # Aggregate
    all_skip = []
    all_no_skip = []
    for pid in participants:
        all_skip.extend(all_results[pid]['ei']['about_to_skip'])
        all_no_skip.extend(all_results[pid]['ei']['not_about_to_skip'])

    agg_stats = run_stats(all_skip, all_no_skip)
    means = [agg_stats['skip_mean'], agg_stats['no_skip_mean']]
    stds = [agg_stats['skip_std'], agg_stats['no_skip_std']]
    p_val = agg_stats['mannwhitney_p']

    bars = axes[-1].bar(labels, means, yerr=stds, color=colors, alpha=0.85,
                        capsize=5, edgecolor='black', linewidth=0.5)
    axes[-1].set_title(f"ALL (n={len(participants)})\n(p={p_val:.4f})", fontsize=11)

    if p_val < 0.001:
        sig = '***'
    elif p_val < 0.01:
        sig = '**'
    elif p_val < 0.05:
        sig = '*'
    else:
        sig = 'n.s.'

    y_max = max(means) + max(stds) * 1.2
    axes[-1].text(0.5, y_max, sig, ha='center', fontsize=14, fontweight='bold',
                  transform=axes[-1].get_xaxis_transform())

    plt.suptitle('Engagement Index: β / (α + θ)', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / 'engagement_index_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   ✓ Comparison plot saved")

    return agg_stats


def plot_ei_distributions(all_results, output_dir):
    """Histogram/boxplot of EI distributions per class."""
    participants = list(all_results.keys())
    n = len(participants)

    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]

    for i, pid in enumerate(participants):
        ei_skip = all_results[pid]['ei']['about_to_skip']
        ei_no_skip = all_results[pid]['ei']['not_about_to_skip']

        bp = axes[i].boxplot([ei_skip, ei_no_skip], labels=['Skip', 'No Skip'],
                             patch_artist=True)
        bp['boxes'][0].set_facecolor('#e74c3c')
        bp['boxes'][0].set_alpha(0.6)
        bp['boxes'][1].set_facecolor('#2ecc71')
        bp['boxes'][1].set_alpha(0.6)

        axes[i].set_title(pid, fontsize=11)
        axes[i].set_ylabel('EI' if i == 0 else '')

    plt.suptitle('EI Distribution by Class', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'engagement_index_boxplot.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   ✓ Boxplot saved")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Analysis V8: Engagement Index Comparison')
    parser.add_argument('--file', '-f', type=str, nargs='*',
                        help='Specific skip labels file(s). If omitted, auto-finds all.')
    parser.add_argument('--window', '-w', type=float, default=3.0,
                        help='Window duration in seconds (default: 3.0)')
    parser.add_argument('--nonotch', action='store_true',
                        help='Disable 50Hz notch filter')
    args = parser.parse_args()

    print("=" * 60)
    print("ANALYSIS V8: ENGAGEMENT INDEX (EI)")
    print("  EI = beta / (alpha + theta)  [Muse standard]")
    print("=" * 60)

    # Find files
    if args.file:
        input_files = [Path(f) for f in args.file]
    else:
        recordings_dir = Path(__file__).resolve().parent.parent / "recordings"
        input_files = sorted(recordings_dir.rglob("*_skip_labels_3_0s.csv"))
        # Exclude the classified variant to avoid doubles
        input_files = [f for f in input_files if 'classified_skip_labels' not in f.name]

    if not input_files:
        print("ERROR: No skip labels files found!")
        return 1

    print(f"\n   Found {len(input_files)} participant file(s)")
    for f in input_files:
        print(f"     • {f.name}")

    notch_freq = None if args.nonotch else 50.0
    all_results = {}

    for input_file in input_files:
        # Determine participant label
        rec_dir = input_file.parent.name
        pid = PARTICIPANT_LABELS.get(rec_dir, rec_dir)

        print(f"\n{'='*60}")
        print(f"  {pid}: {input_file.name}")
        print(f"{'='*60}")

        # Load
        df = pd.read_csv(input_file)
        required = ['lsl_timestamp', 'keypress_A', 'classification_2'] + EEG_CHANNELS
        missing = [c for c in required if c not in df.columns]
        if missing:
            print(f"   ⚠ Missing columns: {missing} — skipping")
            continue
        print(f"   ✓ Loaded {len(df)} rows")

        # Compute sample rate
        time_diffs = np.diff(df['lsl_timestamp'].values)
        fs = 1.0 / np.median(time_diffs)
        print(f"   ✓ Sample rate: {fs:.1f} Hz")

        # Compute EI
        ei_by_class = compute_ei_per_window(df, fs, window_seconds=args.window,
                                            notch_freq=notch_freq)

        n_skip = len(ei_by_class['about_to_skip'])
        n_no_skip = len(ei_by_class['not_about_to_skip'])
        print(f"   ✓ EI computed: {n_skip} skip windows, {n_no_skip} no-skip windows")

        if n_skip < 3 or n_no_skip < 3:
            print(f"   ⚠ Too few samples for stats — skipping")
            continue

        # Stats
        stat_results = run_stats(ei_by_class['about_to_skip'],
                                 ei_by_class['not_about_to_skip'])

        print(f"\n   Results:")
        print(f"   ┌─────────────────────┬────────────┬────────────────┐")
        print(f"   │                     │ About Skip │ Not About Skip │")
        print(f"   ├─────────────────────┼────────────┼────────────────┤")
        print(f"   │ EI Mean             │ {stat_results['skip_mean']:10.4f} │ {stat_results['no_skip_mean']:14.4f} │")
        print(f"   │ EI Std              │ {stat_results['skip_std']:10.4f} │ {stat_results['no_skip_std']:14.4f} │")
        print(f"   │ EI Median           │ {stat_results['skip_median']:10.4f} │ {stat_results['no_skip_median']:14.4f} │")
        print(f"   │ N windows           │ {stat_results['skip_n']:10d} │ {stat_results['no_skip_n']:14d} │")
        print(f"   └─────────────────────┴────────────┴────────────────┘")
        print(f"   Δ Mean EI:       {stat_results['delta_mean']:+.4f}")
        print(f"   Cohen's d:       {stat_results['cohens_d']:.3f}")
        print(f"   Mann-Whitney U:  p={stat_results['mannwhitney_p']:.6f}")
        print(f"   t-test:          p={stat_results['ttest_p']:.6f}")

        sig = '***' if stat_results['mannwhitney_p'] < 0.001 else \
              '**'  if stat_results['mannwhitney_p'] < 0.01  else \
              '*'   if stat_results['mannwhitney_p'] < 0.05  else 'n.s.'
        print(f"   Significance:    {sig}")

        all_results[pid] = {
            'file': str(input_file),
            'stats': stat_results,
            'ei': ei_by_class
        }

    if not all_results:
        print("\nERROR: No valid results computed.")
        return 1

    # ---- Save outputs ----
    print(f"\n{'='*60}")
    print("SAVING OUTPUTS")
    print("=" * 60)

    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(__file__).resolve().parent.parent / "recordings" / f"analysis_v8_EI_{timestamp_str}"
    output_dir.mkdir(exist_ok=True)

    # Plots
    agg_stats = plot_ei_comparison(all_results, output_dir)
    plot_ei_distributions(all_results, output_dir)

    # Summary JSON
    save_data = {}
    for pid, data in all_results.items():
        save_data[pid] = data['stats']
        save_data[pid]['file'] = data['file']
    save_data['AGGREGATE'] = agg_stats
    save_data['parameters'] = {
        'window_seconds': args.window,
        'notch_filter': notch_freq is not None,
        'formula': 'EI = beta_power / (alpha_power + theta_power)',
        'electrodes': EEG_CHANNELS,
    }

    with open(output_dir / 'engagement_index_results.json', 'w') as f:
        json.dump(save_data, f, indent=2)
    print(f"   ✓ Results JSON saved")

    # Summary table
    print(f"\n{'='*60}")
    print("SUMMARY TABLE")
    print("=" * 60)
    print(f"\n{'Participant':<14} {'EI(skip)':<12} {'EI(no-skip)':<14} {'Δ':<10} {'p (MW)':<12} {'Cohen d':<10} {'Sig'}")
    print("-" * 82)
    for pid, data in all_results.items():
        r = data['stats']
        sig = '***' if r['mannwhitney_p'] < 0.001 else \
              '**'  if r['mannwhitney_p'] < 0.01  else \
              '*'   if r['mannwhitney_p'] < 0.05  else 'n.s.'
        print(f"{pid:<14} {r['skip_mean']:<12.4f} {r['no_skip_mean']:<14.4f} {r['delta_mean']:<+10.4f} {r['mannwhitney_p']:<12.6f} {r['cohens_d']:<10.3f} {sig}")

    # Aggregate
    sig = '***' if agg_stats['mannwhitney_p'] < 0.001 else \
          '**'  if agg_stats['mannwhitney_p'] < 0.01  else \
          '*'   if agg_stats['mannwhitney_p'] < 0.05  else 'n.s.'
    print("-" * 82)
    print(f"{'AGGREGATE':<14} {agg_stats['skip_mean']:<12.4f} {agg_stats['no_skip_mean']:<14.4f} {agg_stats['delta_mean']:<+10.4f} {agg_stats['mannwhitney_p']:<12.6f} {agg_stats['cohens_d']:<10.3f} {sig}")

    print(f"\n   Outputs: {output_dir}")
    print(f"\n{'='*60}")
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    exit(main())
