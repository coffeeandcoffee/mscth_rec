#!/usr/bin/env python3
"""
Step 14: Dataset Summary Statistics

Reads all 25 processed CSV files and produces a comprehensive per-participant
dataset summary including: number of keypresses (skips), STAY/SKIP window counts,
total viewing time, skip rate, etc.

Outputs:
  - dataset_summary.csv     (per-participant + aggregate)
  - dataset_summary.txt     (human-readable table)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime


def main():
    project_root = Path(__file__).resolve().parent.parent.parent
    processed_dir = project_root / "04-26_data_analysis_and_results" / "processed_data"
    outputs_dir = project_root / "04-26_data_analysis_and_results" / "outputs"

    out_dir = outputs_dir / f"dataset_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("STEP 14: DATASET SUMMARY STATISTICS")
    print("=" * 70)

    input_files = sorted(processed_dir.glob("P*_labeled.csv"))
    if not input_files:
        print("ERROR: No labeled data files found.")
        return 1

    rows = []
    for f_path in input_files:
        pid = f_path.stem.split('_')[0]
        print(f"  Processing {pid}...", end="")

        df = pd.read_csv(f_path)

        timestamps = df['lsl_timestamp'].values
        total_duration_s = timestamps[-1] - timestamps[0]

        # Count keypresses
        n_skips = int(df['keypress_A'].sum()) if 'keypress_A' in df.columns else 0

        # Count samples per class
        if 'engagement_state' in df.columns:
            n_stay_samples = int((df['engagement_state'] == 1).sum())
            n_skip_samples = int((df['engagement_state'] == 0).sum())
        elif 'class' in df.columns:
            n_stay_samples = int(df['class'].isin(['stay']).sum())
            n_skip_samples = int(df['class'].isin(['skip']).sum())
        else:
            n_stay_samples = 0
            n_skip_samples = 0

        # Effective sampling rate
        time_diffs = np.diff(timestamps)
        fs = 1.0 / np.median(time_diffs)

        # Baseline duration
        if 'class' in df.columns:
            baseline_samples = df['class'].isin(['baseline_1', 'baseline_2']).sum()
            baseline_duration_s = baseline_samples / fs
        else:
            baseline_duration_s = 0

        total_samples = len(df)

        rows.append({
            'participant': pid,
            'total_samples': total_samples,
            'effective_fs_hz': round(fs, 1),
            'total_duration_min': round(total_duration_s / 60, 1),
            'baseline_duration_s': round(baseline_duration_s, 1),
            'n_keypresses_skips': n_skips,
            'n_stay_samples': n_stay_samples,
            'n_skip_samples': n_skip_samples,
            'skip_rate_per_min': round(n_skips / (total_duration_s / 60), 2) if total_duration_s > 0 else 0,
            'mean_viewing_per_video_s': round(total_duration_s / max(n_skips, 1), 1),
        })
        print(f" {n_skips} skips, {total_duration_s/60:.1f} min")

    df_summary = pd.DataFrame(rows)

    # Add aggregate row
    agg = {
        'participant': 'AGGREGATE (mean±sd)',
        'total_samples': f"{df_summary['total_samples'].mean():.0f}±{df_summary['total_samples'].std():.0f}",
        'effective_fs_hz': f"{df_summary['effective_fs_hz'].mean():.1f}",
        'total_duration_min': f"{df_summary['total_duration_min'].mean():.1f}±{df_summary['total_duration_min'].std():.1f}",
        'baseline_duration_s': f"{df_summary['baseline_duration_s'].mean():.1f}±{df_summary['baseline_duration_s'].std():.1f}",
        'n_keypresses_skips': f"{df_summary['n_keypresses_skips'].mean():.1f}±{df_summary['n_keypresses_skips'].std():.1f}",
        'n_stay_samples': f"{df_summary['n_stay_samples'].mean():.0f}",
        'n_skip_samples': f"{df_summary['n_skip_samples'].mean():.0f}",
        'skip_rate_per_min': f"{df_summary['skip_rate_per_min'].mean():.2f}±{df_summary['skip_rate_per_min'].std():.2f}",
        'mean_viewing_per_video_s': f"{df_summary['mean_viewing_per_video_s'].mean():.1f}±{df_summary['mean_viewing_per_video_s'].std():.1f}",
    }

    df_summary.to_csv(out_dir / 'dataset_summary.csv', index=False)

    # Human-readable output
    lines = []
    lines.append("DATASET SUMMARY STATISTICS")
    lines.append("=" * 90)
    lines.append(f"{'PID':<5} {'Samples':>8} {'fs(Hz)':>7} {'Dur(min)':>9} "
                 f"{'Baseline':>9} {'Skips':>6} {'Rate/min':>9} {'Avg View(s)':>12}")
    lines.append("-" * 90)

    for _, r in df_summary.iterrows():
        lines.append(f"{r['participant']:<5} {r['total_samples']:>8} {r['effective_fs_hz']:>7.1f} "
                     f"{r['total_duration_min']:>9.1f} {r['baseline_duration_s']:>9.1f} "
                     f"{r['n_keypresses_skips']:>6} {r['skip_rate_per_min']:>9.2f} "
                     f"{r['mean_viewing_per_video_s']:>12.1f}")

    lines.append("-" * 90)
    lines.append(f"MEAN   {df_summary['total_samples'].mean():>8.0f} "
                 f"{df_summary['effective_fs_hz'].mean():>7.1f} "
                 f"{df_summary['total_duration_min'].mean():>9.1f} "
                 f"{df_summary['baseline_duration_s'].mean():>9.1f} "
                 f"{df_summary['n_keypresses_skips'].mean():>6.1f} "
                 f"{df_summary['skip_rate_per_min'].mean():>9.2f} "
                 f"{df_summary['mean_viewing_per_video_s'].mean():>12.1f}")
    lines.append(f"SD     {df_summary['total_samples'].std():>8.0f} "
                 f"{df_summary['effective_fs_hz'].std():>7.1f} "
                 f"{df_summary['total_duration_min'].std():>9.1f} "
                 f"{df_summary['baseline_duration_s'].std():>9.1f} "
                 f"{df_summary['n_keypresses_skips'].std():>6.1f} "
                 f"{df_summary['skip_rate_per_min'].std():>9.2f} "
                 f"{df_summary['mean_viewing_per_video_s'].std():>12.1f}")

    text_output = "\n".join(lines)
    with open(out_dir / 'dataset_summary.txt', 'w') as f:
        f.write(text_output)

    print(f"\n{text_output}")
    print(f"\n{'=' * 70}")
    print(f"Outputs written to: {out_dir.name}")
    return 0


if __name__ == "__main__":
    main()
