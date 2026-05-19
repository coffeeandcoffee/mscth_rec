#!/usr/bin/env python3
"""
Exact SKIP window validation.
Goes back to raw CSVs to get precise A-press times, then verifies
step01 labelling matches exactly ±3s around each press, including
correct merged block boundaries for overlapping presses.
"""
import pickle
import numpy as np
import pandas as pd
import sys
sys.path.insert(0, '.')
import config

RUN = '../runs/run_20260518_171659'  # update if needed
PID = 4
CSV_IDX = 0  # which CSV to inspect
N_BLOCKS = 5  # how many blocks to print

# ── Load raw CSV ──
csv_list = config.load_participant_csvs(PID)
raw_df = csv_list[CSV_IDX]

# ── Get exact A-press timestamps from raw CSV ──
a_press_times = raw_df[raw_df['keypress_A'] == 1]['lsl_timestamp'].values
print(f"P{PID} CSV{CSV_IDX} — {len(a_press_times)} A-presses found in raw CSV")
print(f"  Times (relative to CSV start):")
t0_raw = raw_df['lsl_timestamp'].values[0]
for i, t in enumerate(a_press_times):
    print(f"    press {i}: t={t-t0_raw:.3f}s (absolute={t:.3f})")

# ── Compute EXACT expected SKIP intervals from A-press times ──
# Each press creates [t-3, t+3]. Overlapping intervals merge.
skip_window_s = 3.0
intervals = []
for t in a_press_times:
    intervals.append((t - skip_window_s, t + skip_window_s))

# Merge overlapping intervals
intervals.sort()
merged = [intervals[0]]
for lo, hi in intervals[1:]:
    if lo <= merged[-1][1]:
        merged[-1] = (merged[-1][0], max(merged[-1][1], hi))
    else:
        merged.append((lo, hi))

print(f"\n  Expected merged SKIP intervals ({len(merged)} total):")
# Trim to CSV time range
csv_t_start = raw_df['lsl_timestamp'].values[0]
csv_t_end   = raw_df['lsl_timestamp'].values[-1]
for i, (lo, hi) in enumerate(merged[:N_BLOCKS]):
    lo_rel = lo - t0_raw
    hi_rel = hi - t0_raw
    dur = hi - lo
    # Which presses contributed?
    contributors = [j for j, t in enumerate(a_press_times)
                    if t - skip_window_s <= hi and t + skip_window_s >= lo]
    print(f"    interval {i}: [{lo_rel:.3f}s, {hi_rel:.3f}s]  "
          f"dur={dur:.3f}s  from press(es): {contributors}")

# ── Load step01 labelled pkl and compare ──
with open(f'{RUN}/processed/nonotch/P{PID}.pkl', 'rb') as f:
    data = pickle.load(f)

df = data['dfs'][CSV_IDX]
ts = df['lsl_timestamp'].values
classes = df['class'].values
t0_pkl = ts[0]

skip_mask = classes == 'SKIP'
skip_idx = np.where(skip_mask)[0]
breaks = np.where(np.diff(skip_idx) > 1)[0] + 1
actual_blocks = np.split(skip_idx, breaks)

print(f"\n  Actual SKIP blocks in step01 pkl ({len(actual_blocks)} total):")
for i, block in enumerate(actual_blocks[:N_BLOCKS]):
    t_start = ts[block[0]]
    t_end   = ts[block[-1]]
    dur = t_end - t_start
    t_start_rel = t_start - t0_pkl
    t_end_rel   = t_end - t0_pkl
    print(f"    block {i}: [{t_start_rel:.3f}s, {t_end_rel:.3f}s]  "
          f"dur={dur:.3f}s  n_samples={len(block)}")

# ── Direct comparison: expected vs actual ──
print(f"\n  COMPARISON (expected vs actual, first {N_BLOCKS}):")
print(f"  {'#':>3}  {'exp_start':>10}  {'act_start':>10}  {'Δstart':>8}"
      f"  {'exp_end':>10}  {'act_end':>10}  {'Δend':>8}  {'exp_dur':>8}  {'act_dur':>8}  OK?")

# Note: pkl timestamps start at first A-press (trimmed), raw CSV starts earlier
# Align by using the first A-press as anchor
first_a_raw = a_press_times[0]
trim_offset = first_a_raw - skip_window_s  # step01 trims to first A-press

n_compare = min(len(merged), len(actual_blocks), N_BLOCKS)
all_ok = True
for i in range(n_compare):
    exp_lo, exp_hi = merged[i]
    # Adjust expected to pkl time reference (trimmed at first A-press boundary)
    exp_lo_rel = exp_lo - trim_offset
    exp_hi_rel = exp_hi - trim_offset
    exp_dur = exp_hi - exp_lo

    act_lo = ts[actual_blocks[i][0]] - t0_pkl
    act_hi = ts[actual_blocks[i][-1]] - t0_pkl
    act_dur = act_hi - act_lo

    delta_start = act_lo - exp_lo_rel
    delta_end   = act_hi - exp_hi_rel
    ok = abs(delta_start) < 0.05 and abs(delta_end) < 0.05  # within 50ms = 1-2 samples

    if not ok:
        all_ok = False

    print(f"  {i:>3}  {exp_lo_rel:>10.3f}  {act_lo:>10.3f}  {delta_start:>+8.3f}"
          f"  {exp_hi_rel:>10.3f}  {act_hi:>10.3f}  {delta_end:>+8.3f}"
          f"  {exp_dur:>8.3f}  {act_dur:>8.3f}  {'✓' if ok else '✗ MISMATCH'}")

if all_ok:
    print(f"\n  ✓ All compared blocks match expected ±3s labelling exactly.")
else:
    print(f"\n  ✗ Mismatches found — step01 labelling does not match expected ±3s windows.")