#!/usr/bin/env python3
"""
step03_label_windows.py — Window extraction and labelling.

Reads the STAY/SKIP labels from step01 and extracts fixed-size windows
from contiguous SKIP and STAY regions using a sliding window with stride.

Logic:
  1. Find contiguous SKIP regions (labelled ±half_window_s around each
     A-press in step01, so each isolated region is ~2×half_window_s = 6s).
  2. Find contiguous STAY regions (everything between SKIP regions).
  3. From each region, extract windows of `window_s` seconds using
     `stride_s` stride.  A window is only emitted if it fits entirely
     within the region (no crossing label boundaries).
  4. Burst detection: when a SKIP region is longer than a single A-press
     region (~2×half_window_s), it means multiple A-presses merged.
     Count actual A-presses per block using a_press_times from step01.
     Any block with ≥2 A-presses → all its windows are burst-flagged.

Runs four times internally for sensitivity manifests:
  {artifact_include, artifact_exclude} × {burst_include, burst_exclude}

OUT: 4 windowed dataset .pkl files + 4 label summary CSVs

Parameters (from config.DEFAULT_PARAMS['step03']):
  half_window_s  — step01 labels ±this around each A-press (→ 2× = SKIP region)
  window_s       — extracted window duration (both SKIP and STAY)
  stride_s       — sliding window stride (both SKIP and STAY)
  burst_thresh_s — unused directly; burst detected by merged regions + A-press count
"""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from collections import defaultdict

import config


def extract_windows_from_regions(df, fs, label, window_s, stride_s):
    """Extract windows from contiguous regions of a given label.

    For each contiguous block of `label` (SKIP or STAY), slide a window
    of `window_s` seconds with `stride_s` stride from the start of the
    region.  A window is only emitted if it fits entirely within the
    region (no crossing label boundaries).

    Returns a list of window dicts:
        start_time, end_time, label (0=SKIP, 1=STAY), indices, region_idx
    """
    timestamps = df['lsl_timestamp'].values
    classes = df['class'].values
    windows = []

    target_mask = np.array([c == label if isinstance(c, str) else False
                            for c in classes])
    target_indices = np.where(target_mask)[0]
    if len(target_indices) == 0:
        return windows

    # Split into contiguous blocks
    breaks = np.where(np.diff(target_indices) > 1)[0] + 1
    blocks = np.split(target_indices, breaks)

    label_int = 0 if label == 'SKIP' else 1

    for region_idx, block in enumerate(blocks):
        if len(block) < 2:
            continue
        region_start = timestamps[block[0]]
        region_end = timestamps[block[-1]]

        # Region span includes the last sample's duration (1/fs)
        region_span = (region_end - region_start) + (1.0 / fs)

        if region_span < window_s - (1.0 / fs):
            continue

        # Deterministic window count: round() absorbs ±1-sample jitter
        # from 256 Hz interpolation (a "6s" region may be 5.996–6.004s)
        n_windows = int(round((region_span - window_s) / stride_s)) + 1
        n_windows = max(1, n_windows)

        for wi in range(n_windows):
            # Compute t from integer multiple to avoid float accumulation
            t = region_start + wi * stride_s

            # Find samples within [t, t + window_s)
            mask = (timestamps >= t) & (timestamps < t + window_s) & target_mask
            indices = np.where(mask)[0]

            if len(indices) >= int(fs * window_s * 0.8):  # ≥80% of expected samples
                windows.append({
                    'start_time': float(t),
                    'end_time': float(t + window_s),
                    'label': label_int,
                    'indices': indices,
                    'region_idx': region_idx,
                })

    return windows


def get_skip_blocks(df, fs):
    """Return list of (block_start_time, block_end_time) for contiguous SKIP regions."""
    timestamps = df['lsl_timestamp'].values
    classes = df['class'].values
    skip_mask = np.array([c == 'SKIP' if isinstance(c, str) else False for c in classes])
    skip_idx = np.where(skip_mask)[0]
    if len(skip_idx) == 0:
        return []

    breaks = np.where(np.diff(skip_idx) > 1)[0] + 1
    blocks = np.split(skip_idx, breaks)

    result = []
    for block in blocks:
        if len(block) < 2:
            continue
        result.append((timestamps[block[0]], timestamps[block[-1]]))
    return result


def flag_burst_skips(df, skip_windows, a_press_times, half_window_s):
    """Flag burst-skip windows using actual A-press timestamps.

    Burst detection:
    1. Find contiguous SKIP blocks in the data.
    2. For each block, count how many A-presses (from a_press_times)
       fall within it.
    3. If a block contains ≥2 A-presses, it's a merged/burst block.
    4. All windows from burst blocks are flagged is_burst_skip=True.

    Returns:
        n_flagged: total number of windows flagged
        block_stats: list of dicts with per-block info for viz/diagnostics
    """
    # Default: no burst
    for w in skip_windows:
        w['is_burst_skip'] = False

    if not skip_windows or not a_press_times:
        return 0, []

    fs = 256.0  # approximate, only used for block detection
    timestamps = df['lsl_timestamp'].values
    classes = df['class'].values
    skip_mask = np.array([c == 'SKIP' if isinstance(c, str) else False for c in classes])
    skip_idx = np.where(skip_mask)[0]
    if len(skip_idx) == 0:
        return 0, []

    breaks = np.where(np.diff(skip_idx) > 1)[0] + 1
    blocks = np.split(skip_idx, breaks)

    block_stats = []
    n_flagged = 0

    for region_idx, block in enumerate(blocks):
        if len(block) < 2:
            block_stats.append({
                'region_idx': region_idx,
                'start_time': float(timestamps[block[0]]),
                'end_time': float(timestamps[block[-1]]),
                'duration_s': 0.0,
                'n_a_presses': 0,
                'is_burst': False,
            })
            continue

        block_start = timestamps[block[0]]
        block_end = timestamps[block[-1]]
        block_dur = block_end - block_start

        # Count A-presses within this block's time range
        # An A-press is "in" a block if it falls within [block_start, block_end]
        # (with small tolerance for the ±half_window labelling)
        a_in_block = [t for t in a_press_times
                      if block_start - 0.1 <= t <= block_end + 0.1]
        n_a = len(a_in_block)

        expected_single = 2 * half_window_s
        is_burst = (n_a >= 2) or (block_dur > expected_single + 1.0)

        block_stats.append({
            'region_idx': region_idx,
            'start_time': float(block_start),
            'end_time': float(block_end),
            'duration_s': float(block_dur),
            'n_a_presses': n_a,
            'is_burst': is_burst,
        })

        if is_burst:
            # Flag all windows from this region
            for w in skip_windows:
                if w['region_idx'] == region_idx:
                    w['is_burst_skip'] = True
                    n_flagged += 1

    return n_flagged, block_stats


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
    """Build a complete window manifest for one participant under given settings.

    Returns:
        filtered: list of window dicts
        block_stats: list of per-SKIP-block burst statistics
    """
    p = params.get('step03', {})
    window_s = p.get('window_s', 3.0)
    stride_s = p.get('stride_s', 0.6)
    half_window_s = p.get('half_window_s', 3.0)

    # Extract windows from SKIP and STAY regions using same window & stride
    skip_windows = extract_windows_from_regions(df, fs, 'SKIP', window_s, stride_s)
    stay_windows = extract_windows_from_regions(df, fs, 'STAY', window_s, stride_s)

    # Flag burst-skips using actual A-press timestamps from step01
    a_press_times = df.attrs.get('a_press_times', [])
    _, block_stats = flag_burst_skips(df, skip_windows, a_press_times, half_window_s)

    # STAY windows are never bursts
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

    return filtered, block_stats


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

    # Read params for display
    p = params.get('step03', {})
    window_s = p.get('window_s', 3.0)
    stride_s = p.get('stride_s', 0.6)
    half_window_s = p.get('half_window_s', 3.0)
    burst_thresh = p.get('burst_thresh_s', 3.0)

    print(f"\n  Parameters:")
    print(f"    half_window_s  = {half_window_s}  (step01 labels ±{half_window_s}s around A-press → {2*half_window_s}s SKIP region)")
    print(f"    window_s       = {window_s}  (extracted window duration)")
    print(f"    stride_s       = {stride_s}  (stride for both SKIP and STAY)")
    print(f"    burst_thresh_s = {burst_thresh}  (A-presses < {burst_thresh}s apart → merged region → burst)")
    print()

    # Collect all block stats across participants for viz03
    all_block_stats = {}

    for pid in config.INCLUDED_PARTICIPANTS:
        pkl_path = processed_dir / f"P{pid}.pkl"
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)

        # [D3] pkl stores list of DataFrames under 'dfs', not single 'df'
        dfs = data['dfs']
        fs = data['fs']
        feature_names = data['feature_names']

        print(f"  P{pid}:", end="", flush=True)

        pid_block_stats = []

        for manifest_name, incl_art, incl_burst in MANIFEST_CONFIGS:
            # Collect windows across all CSVs for this participant
            all_windows = []
            for df in dfs:
                windows, block_stats = build_manifest(
                    pid, df, fs, feature_names, params,
                    include_artifacts=incl_art,
                    include_bursts=incl_burst)
                all_windows.extend(windows)

                # Collect block stats only for primary manifest
                if manifest_name == 'primary':
                    pid_block_stats.extend(block_stats)

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
                    'a_press_times': [t for df in dfs
                                      for t in df.attrs.get('a_press_times', [])],
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

        all_block_stats[pid] = pid_block_stats
        print()

    # Write label summaries
    for manifest_name, _, _ in MANIFEST_CONFIGS:
        df_s = pd.DataFrame(summary_rows[manifest_name])
        csv_path = run_dir / "windows" / manifest_name / "label_summary.csv"
        df_s.to_csv(csv_path, index=False)

    # Write block stats for viz03
    all_block_rows = []
    for pid, stats in all_block_stats.items():
        for bs in stats:
            bs['pid'] = pid
            all_block_rows.append(bs)
    if all_block_rows:
        df_blocks = pd.DataFrame(all_block_rows)
        df_blocks.to_csv(run_dir / "windows" / "primary" / "block_stats.csv", index=False)

    primary = summary_rows['primary']
    total_stay = sum(r['n_stay'] for r in primary)
    total_skip = sum(r['n_skip'] for r in primary)
    total_burst = sum(r['n_burst_skip'] for r in primary)
    print(f"\n  ✓ Primary manifest: {total_stay} STAY, {total_skip} SKIP windows "
          f"({total_burst} burst-flagged) across {len(primary)} participants.")

    # ── Detailed windowing diagnostics ──
    print(f"\n{'─'*92}")
    print(f"  STEP 03 — WINDOW EXTRACTION DIAGNOSTICS")
    print(f"  SKIP region: ±{half_window_s}s around A-press = {2*half_window_s}s per single A-press")
    print(f"  Window: {window_s}s | Stride: {stride_s}s ({(1-stride_s/window_s)*100:.0f}% overlap)")
    print(f"  Burst: SKIP blocks with ≥2 A-presses (merged regions)")
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

            # STAY regions ≥ window_s
            stay_idx = np.where(stay_mask)[0]
            if len(stay_idx) > 0:
                breaks = np.where(np.diff(stay_idx) > 1)[0] + 1
                for region in np.split(stay_idx, breaks):
                    if len(region) > 0:
                        dur = (df['lsl_timestamp'].values[region[-1]]
                               - df['lsl_timestamp'].values[region[0]])
                        if dur >= window_s:
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

    # Burst block statistics
    burst_blocks = [bs for stats in all_block_stats.values() for bs in stats if bs['is_burst']]
    normal_blocks = [bs for stats in all_block_stats.values() for bs in stats if not bs['is_burst']]
    print(f"\n  Burst block statistics:")
    print(f"    Normal blocks (1 A-press, ~{2*half_window_s}s): {len(normal_blocks)}")
    print(f"    Burst blocks  (≥2 A-presses, merged):  {len(burst_blocks)}")
    if burst_blocks:
        a_counts = [bs['n_a_presses'] for bs in burst_blocks]
        durs = [bs['duration_s'] for bs in burst_blocks]
        print(f"    Burst A-press counts: min={min(a_counts)}, max={max(a_counts)}, "
              f"mean={np.mean(a_counts):.1f}")
        print(f"    Burst durations: min={min(durs):.1f}s, max={max(durs):.1f}s, "
              f"mean={np.mean(durs):.1f}s")

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
