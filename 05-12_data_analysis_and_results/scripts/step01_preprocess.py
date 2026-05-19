#!/usr/bin/env python3
"""
step01_preprocess.py — Baseline-normalised relative power computation.

Runs twice internally: nonotch (primary) and notch (ablation).
Computes true sampling rate via median delta-t, interpolates each CSV to
256 Hz, applies Butterworth 7-band split, z-score normalises via 90s baseline
window, labels each sample as STAY or SKIP via ±3s A-press windows.

OUT: per-participant .pkl files + dropout_log.csv

# ── DEPENDENCY NOTES FOR DOWNSTREAM STEPS ──────────────────────────────────
# [D1] df['class'] values are now 'STAY' / 'SKIP' (was 'tiktok_over_4s' /
#      'tiktok_under_4s') → step03 must match these exact strings.
# [D2] Band power columns are z-score normalised: z = (value - baseline_mean)
#      / baseline_std (was ratio: value / baseline_mean) → viz01, step05
#      feature interpretation, and any threshold-based logic must be updated.
# [D3] pkl now stores a list of DataFrames (one per CSV) under key 'dfs',
#      NOT a single merged 'df' → step03 must iterate over 'dfs' instead of
#      reading 'df'.
# [D4] baseline_stats in pkl is now a dict with two sub-dicts: 'mean' and
#      'std', keyed by feature name (was baseline_means dict of means only)
#      → step05 and viz01 must read baseline_stats['mean'] and
#      baseline_stats['std'] instead of baseline_means[fname].
# [D5] Dropouts are now per-CSV-boundary gaps stored in 'dropouts' list,
#      each entry has keys: start_idx, end_idx, start_time, end_time, gap_s.
#      Semantics unchanged, count now equals number_of_csvs - 1.
# ────────────────────────────────────────────────────────────────────────────
"""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from scipy.interpolate import interp1d

import config

# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────


# ──────────────────────────────────────────────
# Baseline extraction
# ──────────────────────────────────────────────

def extract_baseline_stats(csv_list, baseline_offset_s, baseline_duration_s, target_fs, notch_freq=None):
    """Extract baseline_stats dict with 'mean' and 'std' sub-dicts per
    channel×band, computed from the interpolated+filtered 90s baseline window.

    Searches all CSVs by timestamp for the first B press. Collects raw rows
    from t_start to t_end across CSV boundaries if needed (e.g. B press near
    end of CSV 1, window continues into CSV 2).
    """
    # ── Find first B press across all CSVs ──
    first_b_time = None
    for df in csv_list:
        if 'keypress_B' not in df.columns:
            continue
        b_rows = df[df['keypress_B'] == 1]
        if len(b_rows) == 0:
            continue
        first_b_time = df.loc[b_rows.index[0], 'lsl_timestamp']
        break

    if first_b_time is None:
        raise ValueError("No B keypress found in any CSV — cannot extract baseline.")

    t_start = first_b_time + baseline_offset_s
    t_end   = first_b_time + baseline_offset_s + baseline_duration_s

    # ── Collect raw rows within [t_start, t_end] across all CSVs ──
    chunks = []
    for df in csv_list:
        mask = (df['lsl_timestamp'] >= t_start) & (df['lsl_timestamp'] <= t_end)
        chunk = df[mask]
        if len(chunk) > 0:
            chunks.append(chunk)

    if not chunks:
        raise ValueError(
            f"Baseline window [{t_start:.2f}, {t_end:.2f}] found no rows "
            f"in any CSV. Check B press timing."
        )

    df_base_raw = pd.concat(chunks, ignore_index=True).sort_values('lsl_timestamp')

    if len(df_base_raw) < 50:
        raise ValueError(
            f"Baseline window too short: only {len(df_base_raw)} raw samples found "
            f"across {len(chunks)} CSV(s)."
        )

    # ── Interpolate to target_fs ──
    t_raw = df_base_raw['lsl_timestamp'].values
    t_uniform = np.arange(t_raw[0], t_raw[-1], 1.0 / target_fs)

    baseline_stats = {'mean': {}, 'std': {}}

    for ch in config.EEG_CHANNELS:
        raw_ch = df_base_raw[ch].values.astype(float)
        interp_fn = interp1d(t_raw, raw_ch, kind='linear', fill_value='extrapolate')
        ch_uniform = interp_fn(t_uniform)

        if notch_freq:
            ch_uniform = config.apply_notch_filter(ch_uniform, target_fs, notch_freq)

        for band_name, lo, hi in config.FREQUENCY_BANDS:
            fname = f"{ch}_{band_name}"
            amp = config.extract_band_amplitude(ch_uniform, target_fs, lo, hi)
            power = amp ** 2
            m = float(np.mean(power))
            s = float(np.std(power))
            if m < 1e-12:
                m = 1e-12
            if s < 1e-12:
                s = 1e-12
            baseline_stats['mean'][fname] = m
            baseline_stats['std'][fname] = s

    return baseline_stats

# ──────────────────────────────────────────────
# STAY / SKIP labeling
# ──────────────────────────────────────────────

def label_stay_skip(df, skip_window_s):
    """Label every row as STAY or SKIP.

    Rule:
        All rows start as STAY.
        For each A press, all rows within ±SKIP_WINDOW_S of that press
        (checked via lsl_timestamp, not row count) are relabeled SKIP.
        Overlapping SKIP windows from adjacent A presses merge naturally.

    Returns df with added 'class' column ('STAY' or 'SKIP').
    """
    df = df.copy()
    df['class'] = 'STAY'

    if 'keypress_A' not in df.columns:
        return df

    timestamps = df['lsl_timestamp'].values
    a_press_rows = df[df['keypress_A'] == 1]

    for row_idx in a_press_rows.index:
        t_press = df.loc[row_idx, 'lsl_timestamp']
        t_lo = t_press - skip_window_s
        t_hi = t_press + skip_window_s
        mask = (timestamps >= t_lo) & (timestamps <= t_hi)
        df.loc[mask, 'class'] = 'SKIP'

    return df


def trim_to_a_presses(df):
    """Remove rows before the first A press and after the last A press.

    Returns None if the CSV has zero A presses (caller must skip it).
    """
    if 'keypress_A' not in df.columns:
        return None
    a_rows = df[df['keypress_A'] == 1]
    if len(a_rows) == 0:
        return None
    first_a_time = df.loc[a_rows.index[0], 'lsl_timestamp']
    last_a_time  = df.loc[a_rows.index[-1], 'lsl_timestamp']
    skip_window_s = 3.0  # must include the full ±3s around first and last press
    mask = (
        (df['lsl_timestamp'] >= first_a_time - skip_window_s) &
        (df['lsl_timestamp'] <= last_a_time  + skip_window_s)
    )
    return df[mask].copy()


# ──────────────────────────────────────────────
# Interpolation + band power
# ──────────────────────────────────────────────

def interpolate_to_target_fs(df, target_fs):
    """Interpolate all EEG channels in df to TARGET_FS uniform grid.

    Returns a new DataFrame with uniform timestamps and interpolated channel
    values. All other columns (keypress, class) are dropped — they are no
    longer needed after labeling and trimming.
    """
    t_raw = df['lsl_timestamp'].values
    if len(t_raw) < 4:
        return None

    t_uniform = np.arange(t_raw[0], t_raw[-1], 1.0 / target_fs)
    if len(t_uniform) < 4:
        return None

    out = {'lsl_timestamp': t_uniform, 'class': None}

    # Interpolate class label (nearest neighbour — preserve STAY/SKIP)
    class_numeric = (df['class'].values == 'SKIP').astype(float)
    interp_class = interp1d(t_raw, class_numeric, kind='nearest',
                            fill_value='extrapolate')
    out['class'] = np.where(interp_class(t_uniform) > 0.5, 'SKIP', 'STAY')

    for ch in config.EEG_CHANNELS:
        raw_ch = df[ch].values.astype(float)
        interp_fn = interp1d(t_raw, raw_ch, kind='linear',
                             fill_value='extrapolate')
        ch_interp = interp_fn(t_uniform)
        out[ch] = ch_interp - np.mean(ch_interp)  # remove DC offset

    return pd.DataFrame(out)


def build_relative_power(df_uniform, baseline_stats, target_fs, notch_freq=None):
    """Compute z-score normalised band power for all 28 channel×band features.

    z = (instantaneous_power - baseline_mean) / baseline_std

    Returns df_uniform with 28 new feature columns added, plus feature_names.
    [D2] Values are z-scores, not ratios. Downstream must not assume > 0.
    """
    df_out = df_uniform.copy()
    feature_names = []

    for ch in config.EEG_CHANNELS:
        raw = df_uniform[ch].values.astype(float)
        if notch_freq:
            raw = config.apply_notch_filter(raw, target_fs, notch_freq)

        for band_name, lo, hi in config.FREQUENCY_BANDS:
            fname = f"{ch}_{band_name}"
            feature_names.append(fname)
            amp = config.extract_band_amplitude(raw, target_fs, lo, hi)
            power = amp ** 2

            b_mean = baseline_stats['mean'].get(fname, 1e-12)
            b_std = baseline_stats['std'].get(fname, 1e-12)
            df_out[fname] = (power - b_mean) / b_std

    return df_out, feature_names


# ──────────────────────────────────────────────
# Per-participant processing
# ──────────────────────────────────────────────

def process_participant(pid, out_dir_nonotch, out_dir_notch, params):
    """Process a single participant across all their CSVs."""
    print(f"  P{pid}...", end="", flush=True)

    p = params['step01']
    target_fs        = float(p['target_fs'])
    baseline_offset_s   = float(p['baseline_offset_s'])
    baseline_duration_s = float(p['baseline_duration_s'])
    skip_window_s    = float(p['skip_window_s'])

    csv_list = config.load_participant_csvs(pid)

    # ── Validate baseline (hard prerequisite) ──
    try:
        baseline_stats = extract_baseline_stats(
            csv_list, baseline_offset_s, baseline_duration_s, target_fs, notch_freq=None
        )
        baseline_stats_notch = extract_baseline_stats(
            csv_list, baseline_offset_s, baseline_duration_s, target_fs, notch_freq=50.0
        )
    except ValueError as e:
        raise RuntimeError(f"P{pid} baseline validation failed: {e}")

    # ── Compute dropout gaps (between consecutive CSVs) ──
    dropouts = []
    cumulative_len = 0
    for i in range(len(csv_list) - 1):
        df_a = csv_list[i]
        df_b = csv_list[i + 1]
        t_end = df_a['lsl_timestamp'].values[-1]
        t_start = df_b['lsl_timestamp'].values[0]
        boundary_idx = cumulative_len + len(df_a) - 1
        dropouts.append({
            'start_idx': int(boundary_idx),
            'end_idx': int(boundary_idx + 1),
            'start_time': float(t_end),
            'end_time': float(t_start),
            'gap_s': float(t_start - t_end),
        })
        cumulative_len += len(df_a)

    # ── Label, trim, interpolate, and compute band power per CSV ──
    # Run twice: nonotch and notch
    processed_dfs_nonotch = []
    processed_dfs_notch = []
    n_stay_total = 0
    n_skip_total = 0

    for csv_df in csv_list:
        # Label STAY/SKIP
        labeled = label_stay_skip(csv_df, skip_window_s)

        # Trim to first/last A press — skip CSV if no A presses
        trimmed = trim_to_a_presses(labeled)
        if trimmed is None:
            continue

        # Interpolate to target_fs
        uniform = interpolate_to_target_fs(trimmed, target_fs)
        if uniform is None:
            continue

        n_stay_total += (uniform['class'] == 'STAY').sum()
        n_skip_total += (uniform['class'] == 'SKIP').sum()

        # Count A-presses in this CSV before interpolation drops keypress col
        n_a_presses = int((trimmed['keypress_A'] == 1).sum()) \
            if 'keypress_A' in trimmed.columns else 0

        # Nonotch path
        df_nn, feature_names = build_relative_power(
            uniform, baseline_stats, target_fs, notch_freq=None
        )
        df_nn.attrs['n_a_presses'] = n_a_presses
        processed_dfs_nonotch.append(df_nn)

        # Notch path
        df_n, _ = build_relative_power(
            uniform, baseline_stats_notch, target_fs, notch_freq=50.0
        )
        df_n.attrs['n_a_presses'] = n_a_presses
        processed_dfs_notch.append(df_n)

    if not processed_dfs_nonotch:
        raise RuntimeError(f"P{pid}: no valid CSVs with A presses found.")

    # ── Build pkl payload ──
    # [D3] 'dfs' is a list of DataFrames, one per CSV — NOT a single merged df
    # [D4] 'baseline_stats' has sub-dicts 'mean' and 'std'
    data_nonotch = {
        'pid': pid,
        'fs': target_fs,
        'dfs': processed_dfs_nonotch,       # [D3]
        'feature_names': feature_names,
        'baseline_stats': baseline_stats,   # [D4]
        'dropouts': dropouts,               # [D5]
    }
    with open(out_dir_nonotch / f"P{pid}.pkl", 'wb') as f:
        pickle.dump(data_nonotch, f)

    data_notch = {
        'pid': pid,
        'fs': target_fs,
        'dfs': processed_dfs_notch,              # [D3]
        'feature_names': feature_names,
        'baseline_stats': baseline_stats_notch,  # [D4]
        'dropouts': dropouts,                    # [D5]
    }
    with open(out_dir_notch / f"P{pid}.pkl", 'wb') as f:
        pickle.dump(data_notch, f)

    print(f" fs={target_fs:.1f}Hz, dropouts={len(dropouts)}, "
          f"stay_samples={n_stay_total}, skip_samples={n_skip_total}")

    return {
        'pid': pid,
        'fs': target_fs,
        'n_dropouts': len(dropouts),
        'total_dropout_s': round(sum(d['gap_s'] for d in dropouts), 2),
        'n_csvs': len(csv_list),
        'stay_samples': int(n_stay_total),
        'skip_samples': int(n_skip_total),
    }


def run(run_dir, params):
    """Entry point called by run.py."""
    config.pprint_step(1, "PREPROCESSING")

    out_dir_nonotch = run_dir / "processed" / "nonotch"
    out_dir_notch = run_dir / "processed" / "notch"
    out_dir_nonotch.mkdir(parents=True, exist_ok=True)
    out_dir_notch.mkdir(parents=True, exist_ok=True)

    dropout_rows = []
    for pid in config.INCLUDED_PARTICIPANTS:
        info = process_participant(pid, out_dir_nonotch, out_dir_notch, params)
        dropout_rows.append(info)

    df_log = pd.DataFrame(dropout_rows)
    df_log.to_csv(run_dir / "dropout_log.csv", index=False)
    print(f"\n  ✓ Dropout log saved. {len(dropout_rows)} participants processed.")

    # ── Detailed STAY/SKIP summary table ──
    p = params['step01']
    target_fs = float(p['target_fs'])
    skip_window_s = float(p['skip_window_s'])

    print(f"\n{'─'*78}")
    print(f"  STEP 01 — STAY/SKIP CLASS SUMMARY")
    print(f"  Labelling: ±{skip_window_s}s around each A-press → SKIP, rest → STAY")
    print(f"  Interpolated to {target_fs:.0f} Hz")
    print(f"{'─'*78}")
    print(f"  {'PID':>4}  {'CSVs':>4}  {'STAY samp':>10}  {'SKIP samp':>10}"
          f"  {'STAY%':>6}  {'SKIP%':>6}  {'STAY dur':>9}  {'SKIP dur':>9}"
          f"  {'Drops':>5}  {'Gap(s)':>7}")
    print(f"  {'─'*4}  {'─'*4}  {'─'*10}  {'─'*10}"
          f"  {'─'*6}  {'─'*6}  {'─'*9}  {'─'*9}"
          f"  {'─'*5}  {'─'*7}")

    tot_stay = 0
    tot_skip = 0
    warnings = []

    for r in dropout_rows:
        n_stay = r['stay_samples']
        n_skip = r['skip_samples']
        n_total = n_stay + n_skip
        pct_stay = (n_stay / n_total * 100) if n_total > 0 else 0
        pct_skip = (n_skip / n_total * 100) if n_total > 0 else 0
        dur_stay = n_stay / target_fs
        dur_skip = n_skip / target_fs
        tot_stay += n_stay
        tot_skip += n_skip

        flag = ""
        if pct_skip > 80:
            flag = " ⚠ heavy SKIP"
            warnings.append(f"P{r['pid']}: {pct_skip:.0f}% SKIP — mostly skipping")
        elif pct_stay > 90:
            flag = " ⚠ heavy STAY"
            warnings.append(f"P{r['pid']}: {pct_stay:.0f}% STAY — rarely skipped")
        elif n_skip < 100:
            flag = " ⚠ few SKIP"
            warnings.append(f"P{r['pid']}: only {n_skip} SKIP samples")

        print(f"  P{r['pid']:>3}  {r['n_csvs']:>4}  {n_stay:>10,}  {n_skip:>10,}"
              f"  {pct_stay:>5.1f}%  {pct_skip:>5.1f}%  {dur_stay:>8.1f}s  {dur_skip:>8.1f}s"
              f"  {r['n_dropouts']:>5}  {r['total_dropout_s']:>6.1f}s{flag}")

    # Totals row
    grand_total = tot_stay + tot_skip
    print(f"  {'─'*4}  {'─'*4}  {'─'*10}  {'─'*10}"
          f"  {'─'*6}  {'─'*6}  {'─'*9}  {'─'*9}"
          f"  {'─'*5}  {'─'*7}")
    print(f"  {'TOTAL':>4}  {sum(r['n_csvs'] for r in dropout_rows):>4}"
          f"  {tot_stay:>10,}  {tot_skip:>10,}"
          f"  {tot_stay/grand_total*100:>5.1f}%  {tot_skip/grand_total*100:>5.1f}%"
          f"  {tot_stay/target_fs:>8.1f}s  {tot_skip/target_fs:>8.1f}s"
          f"  {sum(r['n_dropouts'] for r in dropout_rows):>5}"
          f"  {sum(r['total_dropout_s'] for r in dropout_rows):>6.1f}s")

    if warnings:
        print(f"\n  ⚠ Edge cases ({len(warnings)}):")
        for w in warnings:
            print(f"    • {w}")


if __name__ == "__main__":
    print("Use run.py to execute the pipeline.")