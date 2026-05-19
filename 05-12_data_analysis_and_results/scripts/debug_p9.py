#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config

csv_list = config.load_participant_csvs(9)

print(f"Number of CSVs: {len(csv_list)}")
for i, df in enumerate(csv_list):
    t0 = df['lsl_timestamp'].values[0]
    t1 = df['lsl_timestamp'].values[-1]
    n_rows = len(df)
    has_b = 'keypress_B' in df.columns
    b_presses = df[df['keypress_B'] == 1] if has_b else []
    b_times = [df.loc[idx, 'lsl_timestamp'] for idx in b_presses.index] if has_b and len(b_presses) > 0 else []
    a_presses = df[df['keypress_A'] == 1] if 'keypress_A' in df.columns else []
    print(f"\n  CSV {i+1}:")
    print(f"    rows={n_rows}, t_start={t0:.2f}, t_end={t1:.2f}, duration={t1-t0:.1f}s")
    print(f"    B presses: {len(b_presses) if has_b else 'col missing'} at times {[round(t,2) for t in b_times]}")
    print(f"    A presses: {len(a_presses)}")
    if b_times:
        window_start = b_times[0] + 10.0
        window_end   = b_times[0] + 100.0
        in_window = df[(df['lsl_timestamp'] >= window_start) & (df['lsl_timestamp'] <= window_end)]
        print(f"    Baseline window: {window_start:.2f} → {window_end:.2f}")
        print(f"    Rows in baseline window: {len(in_window)}")