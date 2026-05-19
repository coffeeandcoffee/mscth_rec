#!/usr/bin/env python3
"""
step03_label_windows.py — Window extraction and labelling.

Extracts SKIP windows (3s ending at keypress) and STAY windows (0.6s stride,
80% overlap, from videos watched ≥ 4s). Flags burst-skip sequences.
Runs four times internally for sensitivity manifests:
  {artifact_include, artifact_exclude} × {burst_include, burst_exclude}

OUT: 4 windowed dataset .pkl files + 4 label summary CSVs
"""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path

import config


def extract_skip_windows(df, fs, half_window_s=3.0, window_s=3.0):
    """Extract SKIP windows via sliding window from SKIP-labelled regions.

    SKIP regions are ±half_window_s wide (labelled in step01).
    Windows of window_s are extracted with the same stride as STAY.
    Multiple windows per SKIP region if region is wide enough.
    """
    timestamps = df['lsl_timestamp'].values
    classes = df['class'].values
    windows = []

    skip_mask = np.array([c == 'SKIP' if isinstance(c, str) else False
                          for c in classes])
    skip_indices = np.where(skip_mask)[0]
    if len(skip_indices) == 0:
        return windows

    breaks = np.where(np.diff(skip_indices) > 1)[0] + 1
    blocks = np.split(skip_indices, breaks)

    for block in blocks:
        region_ts = timestamps[block]
        region_dur = region_ts[-1] - region_ts[0]
        if region_dur < window_s:
            continue

        # Find keypress = center of block (for kp_idx metadata only)
        center_idx = block[len(block) // 2]

        t = region_ts[0]
        region_end = region_ts[-1] + (1.0 / fs)  # include last sample
        while t + window_s <= region_end:
            mask = (timestamps >= t) & (timestamps < t + window_s) & skip_mask
            indices = np.where(mask)[0]
            if len(indices) >= int(fs * 0.5):
                windows.append({
                    'start_time': float(t),
                    'end_time': float(t + window_s),
                    'label': 0,
                    'indices': indices,
                    'kp_idx': int(center_idx),
                })
            t += window_s  # no overlap for SKIP — one non-overlapping pass

    return windows

def extract_stay_windows(df, fs, stride_s=0.6, window_s=6.0):
    """Extract STAY windows via sliding window from STAY-labelled regions.

    Window size matches SKIP window (2 × half_window_s = 6s).
    No minimum region duration — every STAY region ≥ window_s is used.
    """
    timestamps = df['lsl_timestamp'].values
    classes = df['class'].values
    windows = []

    stay_mask = np.array([c == 'STAY' if isinstance(c, str) else False
                          for c in classes])
    stay_indices = np.where(stay_mask)[0]

    if len(stay_indices) == 0:
        return windows

    breaks = np.where(np.diff(stay_indices) > 1)[0] + 1
    regions = np.split(stay_indices, breaks)

    for region in regions:
        region_ts = timestamps[region]
        region_dur = region_ts[-1] - region_ts[0]
        if region_dur < window_s:
            continue

        t = region_ts[0]
        while t + window_s <= region_ts[-1]:
            mask = (timestamps >= t) & (timestamps < t + window_s) & stay_mask
            indices = np.where(mask)[0]
            if len(indices) >= int(fs * 0.5):
                windows.append({
                    'start_time': t,
                    'end_time': t + window_s,
                    'label': 1,
                    'indices': indices,
                })
            t += stride_s

    return windows


def flag_burst_skips(skip_windows, burst_thresh_s=3.0):
    """Flag burst-skip sequences where inter-skip interval < threshold."""
    if len(skip_windows) < 2:
        return

    # Sort by end time
    sorted_wins = sorted(skip_windows, key=lambda w: w['end_time'])

    for i in range(len(sorted_wins)):
        sorted_wins[i]['is_burst_skip'] = False

    for i in range(1, len(sorted_wins)):
        interval = sorted_wins[i]['end_time'] - sorted_wins[i - 1]['end_time']
        if interval < burst_thresh_s:
            sorted_wins[i]['is_burst_skip'] = True
            sorted_wins[i - 1]['is_burst_skip'] = True


def compute_artifact_fraction(df, indices):
    """Compute fraction of samples in window that have any artifact flag."""
    if 'artifact_any' not in df.columns:
        return 0.0
    return float(df.iloc[indices]['artifact_any'].mean())


def extract_window_data(df, feature_names, indices):
    """Extract the relative power data array for a window."""
    return df.iloc[indices][feature_names].values.astype(np.float32)


def build_manifest(pid, df, fs, feature_names, params,
                   include_artifacts=True, include_bursts=True):
    """Build a complete window manifest for one participant under given settings."""
    p = params.get('step03', {})
    half_window_s = p.get('half_window_s', 3.0)
    window_s = p.get('window_s', 3.0)
    stride_s = p.get('stay_stride_s', 0.6)
    burst_thresh = p.get('burst_thresh_s', 3.0)
    print(f"    DEBUG build_manifest: window_s={window_s}, half_window_s={half_window_s}")

    skip_windows = extract_skip_windows(df, fs, half_window_s, window_s)
    stay_windows = extract_stay_windows(df, fs, stride_s, window_s)

    # Flag burst-skips
    flag_burst_skips(skip_windows, burst_thresh)

    # Set default burst flag for STAY windows
    for w in stay_windows:
        w['is_burst_skip'] = False

    all_windows = skip_windows + stay_windows

    # Apply filters
    filtered = []
    for w in all_windows:
        art_frac = compute_artifact_fraction(df, w['indices'])

        if not include_artifacts and art_frac > 0.5:
            continue
        if not include_bursts and w.get('is_burst_skip', False):
            continue

        data = extract_window_data(df, feature_names, w['indices'])
        filtered.append({
            'window_id': len(filtered),
            'start_time': w['start_time'],
            'end_time': w['end_time'],
            'label': w['label'],
            'is_burst_skip': w.get('is_burst_skip', False),
            'artifact_frac': art_frac,
            'n_samples': len(w['indices']),
            'data': data,
        })

    return filtered


MANIFEST_CONFIGS = [
    ('primary',                    True,  True),   # artifact_include, burst_flag
    ('artifact_exclude',           False, True),   # artifact_exclude, burst_flag
    ('burst_exclude',              True,  False),  # artifact_include, burst_exclude
    ('artifact_exclude_burst_exclude', False, False),
]


def run(run_dir, params):
    """Entry point called by run.py."""
    config.pprint_step(3, "WINDOW LABELLING")

    processed_dir = run_dir / "processed" / "nonotch"

    for manifest_name, _, _ in MANIFEST_CONFIGS:
        (run_dir / "windows" / manifest_name).mkdir(parents=True, exist_ok=True)

    summary_rows = {name: [] for name, _, _ in MANIFEST_CONFIGS}

    for pid in config.INCLUDED_PARTICIPANTS:
        pkl_path = processed_dir / f"P{pid}.pkl"
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)

        # [D3] pkl stores list of DataFrames under 'dfs', not single 'df'
        dfs = data['dfs']
        fs = data['fs']
        feature_names = data['feature_names']

        print(f"  P{pid}:", end="", flush=True)

        for manifest_name, incl_art, incl_burst in MANIFEST_CONFIGS:
            # Collect windows across all CSVs for this participant
            all_windows = []
            for df in dfs:
                windows = build_manifest(pid, df, fs, feature_names, params,
                                         include_artifacts=incl_art,
                                         include_bursts=incl_burst)
                all_windows.extend(windows)

            # Re-index window_ids across all CSVs
            for i, w in enumerate(all_windows):
                w['window_id'] = i

            out_path = run_dir / "windows" / manifest_name / f"P{pid}.pkl"
            with open(out_path, 'wb') as f:
                pickle.dump({
                    'pid': pid,
                    'windows': all_windows,
                    'feature_names': feature_names,
                    'fs': fs,
                }, f)

            n_stay = sum(1 for w in all_windows if w['label'] == 1)
            n_skip = sum(1 for w in all_windows if w['label'] == 0)
            n_burst = sum(1 for w in all_windows if w.get('is_burst_skip', False))

            summary_rows[manifest_name].append({
                'pid': pid,
                'n_stay': n_stay,
                'n_skip': n_skip,
                'n_burst_skip': n_burst,
                'n_total': len(all_windows),
                'imbalance_ratio': round(n_stay / max(n_skip, 1), 2),
            })

            if manifest_name == 'primary':
                print(f" STAY={n_stay} SKIP={n_skip} burst={n_burst}", end="")

        print()

    # Write label summaries
    for manifest_name, _, _ in MANIFEST_CONFIGS:
        df_s = pd.DataFrame(summary_rows[manifest_name])
        csv_path = run_dir / "windows" / manifest_name / "label_summary.csv"
        df_s.to_csv(csv_path, index=False)

    primary = summary_rows['primary']
    total_stay = sum(r['n_stay'] for r in primary)
    total_skip = sum(r['n_skip'] for r in primary)
    print(f"\n  ✓ Primary manifest: {total_stay} STAY, {total_skip} SKIP windows "
          f"across {len(primary)} participants.")

    # ── Detailed windowing diagnostics ──
    p = params.get('step03', {})
    window_s = p.get('skip_window_s', 3.0)
    stride_s = p.get('stay_stride_s', 0.6)
    min_stay_s = p.get('min_stay_dur_s', 4.0)
    burst_thresh = p.get('burst_thresh_s', 3.0)

    print(f"\n{'─'*92}")
    print(f"  STEP 03 — WINDOW EXTRACTION DIAGNOSTICS")
    print(f"  Window: {window_s}s | STAY stride: {stride_s}s ({(1-stride_s/window_s)*100:.0f}% overlap)"
          f" | Min STAY region: {min_stay_s}s | Burst: <{burst_thresh}s")
    print(f"{'─'*92}")

    # Per-participant window details from the primary manifest
    print(f"\n  {'PID':>4}  {'SKIP blk':>8}  {'SKIP win':>8}  {'STAY reg':>8}"
          f"  {'STAY win':>8}  {'Burst':>5}  {'Ratio':>6}"
          f"  │  {'noArt':>5}  {'noBurst':>7}  {'noA+noB':>7}"
          f"  │  {'A/blk':>6}")
    print(f"  {'─'*4}  {'─'*8}  {'─'*8}  {'─'*8}"
          f"  {'─'*8}  {'─'*5}  {'─'*6}"
          f"  │  {'─'*5}  {'─'*7}  {'─'*7}"
          f"  │  {'─'*6}")

    warnings = []
    tot_skip_blk = 0
    tot_stay_reg = 0

    for pid in config.INCLUDED_PARTICIPANTS:
        pkl_path = processed_dir / f"P{pid}.pkl"
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)

        dfs = data['dfs']
        fs = data['fs']

        # Count raw SKIP blocks and STAY regions across all CSVs
        n_skip_blocks = 0
        n_stay_regions = 0
        for df in dfs:
            classes = df['class'].values
            skip_mask = np.array([c == 'SKIP' for c in classes])
            stay_mask = np.array([c == 'STAY' for c in classes])

            # SKIP blocks
            skip_idx = np.where(skip_mask)[0]
            if len(skip_idx) > 0:
                breaks = np.where(np.diff(skip_idx) > 1)[0] + 1
                n_skip_blocks += len(np.split(skip_idx, breaks))

            # STAY regions ≥ min_stay_s
            stay_idx = np.where(stay_mask)[0]
            if len(stay_idx) > 0:
                breaks = np.where(np.diff(stay_idx) > 1)[0] + 1
                for region in np.split(stay_idx, breaks):
                    if len(region) > 0:
                        dur = (df['lsl_timestamp'].values[region[-1]]
                               - df['lsl_timestamp'].values[region[0]])
                        if dur >= min_stay_s:
                            n_stay_regions += 1

        tot_skip_blk += n_skip_blocks
        tot_stay_reg += n_stay_regions

        # Window counts from each manifest
        pri = next(r for r in summary_rows['primary'] if r['pid'] == pid)
        ae = next(r for r in summary_rows['artifact_exclude'] if r['pid'] == pid)
        be = next(r for r in summary_rows['burst_exclude'] if r['pid'] == pid)
        aebe = next(r for r in summary_rows['artifact_exclude_burst_exclude'] if r['pid'] == pid)

        ratio = pri['n_stay'] / max(pri['n_skip'], 1)

        flag = ""
        if pri['n_skip'] == 0:
            flag = " ⚠ 0 SKIP"
            warnings.append(f"P{pid}: zero SKIP windows — no skipping behaviour captured")
        elif pri['n_stay'] == 0:
            flag = " ⚠ 0 STAY"
            warnings.append(f"P{pid}: zero STAY windows — no sustained engagement captured")
        elif ratio > 20:
            flag = " ⚠ extreme"
            warnings.append(f"P{pid}: STAY:SKIP = {ratio:.0f}:1 — extreme imbalance")

        # Skip momentum: avg A-presses per SKIP block
        total_a = sum(df.attrs.get('n_a_presses', 0) for df in dfs)
        a_per_block = total_a / max(n_skip_blocks, 1)

        print(f"  P{pid:>3}  {n_skip_blocks:>8}  {pri['n_skip']:>8}  {n_stay_regions:>8}"
              f"  {pri['n_stay']:>8}  {pri['n_burst_skip']:>5}  {ratio:>5.1f}x"
              f"  │  {ae['n_total']:>5}  {be['n_total']:>7}  {aebe['n_total']:>7}"
              f"  │  {a_per_block:>5.1f}x{flag}")

    # Totals
    print(f"  {'─'*4}  {'─'*8}  {'─'*8}  {'─'*8}"
          f"  {'─'*8}  {'─'*5}  {'─'*6}"
          f"  │  {'─'*5}  {'─'*7}  {'─'*7}")
    tot_pri_stay = sum(r['n_stay'] for r in summary_rows['primary'])
    tot_pri_skip = sum(r['n_skip'] for r in summary_rows['primary'])
    tot_pri_burst = sum(r['n_burst_skip'] for r in summary_rows['primary'])
    tot_ae = sum(r['n_total'] for r in summary_rows['artifact_exclude'])
    tot_be = sum(r['n_total'] for r in summary_rows['burst_exclude'])
    tot_aebe = sum(r['n_total'] for r in summary_rows['artifact_exclude_burst_exclude'])
    print(f"  {'TOT':>4}  {tot_skip_blk:>8}  {tot_pri_skip:>8}  {tot_stay_reg:>8}"
          f"  {tot_pri_stay:>8}  {tot_pri_burst:>5}  {tot_pri_stay/max(tot_pri_skip,1):>5.1f}x"
          f"  │  {tot_ae:>5}  {tot_be:>7}  {tot_aebe:>7}"
          f"  │  {'':>6}")

    # Manifest comparison
    print(f"\n  Manifest comparison (total windows across all participants):")
    for mname, _, _ in MANIFEST_CONFIGS:
        rows_m = summary_rows[mname]
        ts = sum(r['n_stay'] for r in rows_m)
        tk = sum(r['n_skip'] for r in rows_m)
        tb = sum(r['n_burst_skip'] for r in rows_m)
        print(f"    {mname:>35}: {ts+tk:>6} windows  "
              f"(STAY={ts}, SKIP={tk}, burst={tb})")

    if warnings:
        print(f"\n  ⚠ Edge cases ({len(warnings)}):")
        for w in warnings:
            print(f"    • {w}")
    else:
        print(f"\n  ✓ No edge cases detected.")

    print(f"\n    NOTE: N for binomial CIs comes from step04 balanced counts, "
          f"NOT these raw counts.")


if __name__ == "__main__":
    print("Use run.py to execute the pipeline.")
