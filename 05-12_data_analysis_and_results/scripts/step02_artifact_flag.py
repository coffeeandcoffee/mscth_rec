#!/usr/bin/env python3
"""
step02_artifact_flag.py — Blink and EMG artifact detection.

Detection runs on NOTCHED signal (50Hz mains removed) so EMG detection sees
actual muscle bursts, not line noise. Artifact flags are written to the
NONOTCH pkl where downstream steps expect them.

  - Blink: peak-to-peak of 1-40Hz bandpassed AF7/AF8 > blink_thresh_uv
  - EMG:   peak-to-peak of 30-100Hz bandpassed TP9/TP10 > emg_thresh_uv

No adaptive SD floors, no z-scores. The threshold from params is THE threshold.
Does NOT drop — appends artifact_blink, artifact_emg, artifact_any boolean columns.

OUT: updated nonotch .pkl files + artifact_summary.csv
"""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path

import config


def _flag_ptp_percentile(df, fs, channels, low_hz, high_hz, window_s=0.5):
    """Generic peak-to-peak artifact flagger on a bandpassed signal using 95th percentile.

    A window is flagged if ptp of the bandpassed signal exceeds the 95th percentile
    of all windows for that channel. Sliding window with 50% overlap.
    """
    window_samples = max(int(fs * window_s), 20)
    n = len(df)
    mask = np.zeros(n, dtype=bool)

    for ch in channels:
        raw = df[ch].values.astype(float)
        filtered = config.extract_band_amplitude(raw, fs, low_hz, high_hz)

        # Collect all PTP values for this channel
        ptps = []
        for start in range(0, n - window_samples, window_samples // 2):
            end = start + window_samples
            ptps.append(np.ptp(filtered[start:end]))
            
        if not ptps:
            continue
            
        # Compute 95th percentile threshold
        thresh_uv = np.percentile(ptps, 95)

        # Apply threshold
        idx = 0
        for start in range(0, n - window_samples, window_samples // 2):
            end = start + window_samples
            if ptps[idx] > thresh_uv:
                mask[start:end] = True
            idx += 1

    return mask


def flag_blinks(df, fs):
    """Blink: ptp of 1-40Hz bandpassed AF7/AF8 > 95th percentile."""
    return _flag_ptp_percentile(df, fs, config.FRONTAL_CHANNELS,
                                low_hz=1.0, high_hz=40.0,
                                window_s=0.5)


def flag_emg(df, fs):
    """EMG: ptp of 30-100Hz bandpassed TP9/TP10 > 95th percentile."""
    return _flag_ptp_percentile(df, fs, config.TEMPORAL_CHANNELS,
                                low_hz=30.0, high_hz=100.0,
                                window_s=0.25)


def process_participant(pid, nonotch_dir, notch_dir, params):
    """Flag artifacts using notched signal for detection, write to nonotch pkl.

    The notch pkl is used only to derive artifact masks (clean of 50Hz mains).
    The masks are written back to the nonotch pkl so downstream steps
    (which read nonotch) see them as before.
    """
    # Read NOTCHED data for detection
    notch_pkl = notch_dir / f"P{pid}.pkl"
    with open(notch_pkl, 'rb') as f:
        notch_data = pickle.load(f)
    notch_dfs = notch_data['dfs']
    fs = notch_data['fs']

    # Read NONOTCH for writing flags back
    nonotch_pkl = nonotch_dir / f"P{pid}.pkl"
    with open(nonotch_pkl, 'rb') as f:
        nonotch_data = pickle.load(f)
    nonotch_dfs = nonotch_data['dfs']

    if len(notch_dfs) != len(nonotch_dfs):
        raise RuntimeError(
            f"P{pid}: notch ({len(notch_dfs)}) and nonotch "
            f"({len(nonotch_dfs)}) CSV counts differ — pipeline state corrupt."
        )

    flagged_dfs = []
    total_blink = 0
    total_emg = 0
    total_n = 0

    for notch_df, nonotch_df in zip(notch_dfs, nonotch_dfs):
        if len(notch_df) != len(nonotch_df):
            raise RuntimeError(
                f"P{pid}: notch/nonotch row count mismatch within CSV."
            )

        # Detect on notched signal using dynamic 95th percentile thresholds
        blink_mask = flag_blinks(notch_df, fs)
        emg_mask = flag_emg(notch_df, fs)

        # Write flags to nonotch df (which is what downstream reads)
        out_df = nonotch_df.copy()
        out_df['artifact_blink'] = blink_mask
        out_df['artifact_emg'] = emg_mask
        out_df['artifact_any'] = blink_mask | emg_mask

        flagged_dfs.append(out_df)
        total_blink += int(blink_mask.sum())
        total_emg += int(emg_mask.sum())
        total_n += len(out_df)

    nonotch_data['dfs'] = flagged_dfs
    with open(nonotch_pkl, 'wb') as f:
        pickle.dump(nonotch_data, f)

    blink_rate = total_blink / total_n if total_n > 0 else 0
    emg_rate = total_emg / total_n if total_n > 0 else 0
    any_rate = min((total_blink + total_emg) / total_n, 1.0) if total_n > 0 else 0

    return {
        'pid': pid,
        'n_samples': total_n,
        'blink_flagged': total_blink,
        'emg_flagged': total_emg,
        'any_flagged': min(total_blink + total_emg, total_n),
        'blink_rate': round(blink_rate, 4),
        'emg_rate': round(emg_rate, 4),
        'any_rate': round(any_rate, 4),
    }


def run(run_dir, params):
    """Entry point called by run.py."""
    config.pprint_step(2, "ARTIFACT FLAGGING")

    print(f"  Detection on NOTCHED signal (50Hz removed).")
    print(f"  Thresholds: Outer 5% rejection (95th percentile) for each participant's channels.")

    nonotch_dir = run_dir / "processed" / "nonotch"
    notch_dir = run_dir / "processed" / "notch"
    rows = []

    for pid in config.INCLUDED_PARTICIPANTS:
        info = process_participant(pid, nonotch_dir, notch_dir, params)
        print(f"  P{pid}: blink={info['blink_rate']:.1%}, "
              f"emg={info['emg_rate']:.1%}, total={info['any_rate']:.1%}")
        rows.append(info)

    df_summary = pd.DataFrame(rows)
    df_summary.to_csv(run_dir / "artifact_summary.csv", index=False)
    print(f"\n  ✓ Artifact summary saved. Mean artifact rate: "
          f"{df_summary['any_rate'].mean():.1%}")

    # ── Detailed artifact × class breakdown ──
    print(f"\n{'─'*90}")
    print(f"  STEP 02 — ARTIFACT × CLASS BREAKDOWN")
    print(f"  Purpose: Verify artifacts don't systematically remove one class")
    print(f"{'─'*90}")
    print(f"  {'PID':>4}  {'Samples':>8}  {'Blink%':>7}  {'EMG%':>6}"
          f"  {'Total%':>7}  │  {'STAY samp':>10}  {'STAY art%':>9}"
          f"  {'SKIP samp':>10}  {'SKIP art%':>9}  {'Δ art%':>7}")
    print(f"  {'─'*4}  {'─'*8}  {'─'*7}  {'─'*6}"
          f"  {'─'*7}  │  {'─'*10}  {'─'*9}"
          f"  {'─'*10}  {'─'*9}  {'─'*7}")

    warnings = []
    for pid in config.INCLUDED_PARTICIPANTS:
        pkl_path = nonotch_dir / f"P{pid}.pkl"
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)

        dfs = data['dfs']
        all_classes = []
        all_art = []
        all_blink = []
        all_emg = []
        for df in dfs:
            all_classes.extend(df['class'].values)
            all_art.extend(df['artifact_any'].values)
            all_blink.extend(df['artifact_blink'].values)
            all_emg.extend(df['artifact_emg'].values)

        classes = np.array(all_classes)
        art = np.array(all_art, dtype=bool)
        blink = np.array(all_blink, dtype=bool)
        emg = np.array(all_emg, dtype=bool)
        n = len(classes)

        stay_mask = classes == 'STAY'
        skip_mask = classes == 'SKIP'
        n_stay = int(stay_mask.sum())
        n_skip = int(skip_mask.sum())

        stay_art_pct = (art[stay_mask].sum() / n_stay * 100) if n_stay > 0 else 0
        skip_art_pct = (art[skip_mask].sum() / n_skip * 100) if n_skip > 0 else 0
        delta = skip_art_pct - stay_art_pct

        flag = ""
        if abs(delta) > 15:
            flag = " ⚠"
            warnings.append(
                f"P{pid}: Δ={delta:+.1f}% — artifacts hit "
                f"{'SKIP' if delta > 0 else 'STAY'} harder"
            )

        print(f"  P{pid:>3}  {n:>8,}  {blink.sum()/n*100:>6.1f}%  {emg.sum()/n*100:>5.1f}%"
              f"  {art.sum()/n*100:>6.1f}%  │  {n_stay:>10,}  {stay_art_pct:>8.1f}%"
              f"  {n_skip:>10,}  {skip_art_pct:>8.1f}%  {delta:>+6.1f}%{flag}")

    if warnings:
        print(f"\n  ⚠ Class-imbalanced artifact flags ({len(warnings)}):")
        for w in warnings:
            print(f"    • {w}")
    else:
        print(f"\n  ✓ No major class-imbalanced artifact patterns detected.")


if __name__ == "__main__":
    print("Use run.py to execute the pipeline.")