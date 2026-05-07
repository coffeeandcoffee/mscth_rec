#!/usr/bin/env python3
"""
Step 12: Notch Ablation Statistical Comparison

Loads both RF summary JSONs (notch vs nonotch) and runs paired
Wilcoxon signed-rank tests to determine whether the notch removal
effect is statistically significant or just noise.

Outputs:
  - ablation_comparison.csv       (per-participant deltas)
  - ablation_statistical_test.txt (Wilcoxon + effect sizes)
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from scipy import stats


def find_latest_run_dir(base_dir, prefix):
    dirs = [d for d in base_dir.glob(f"{prefix}*") if d.is_dir()]
    if not dirs:
        return None
    return max(dirs, key=lambda d: d.stat().st_mtime)


def main():
    project_root = Path(__file__).resolve().parent.parent.parent
    outputs_dir = project_root / "04-26_data_analysis_and_results" / "outputs"

    # Find both runs
    notch_dir = find_latest_run_dir(outputs_dir, "rf_run_2026")
    nonotch_dir = find_latest_run_dir(outputs_dir, "rf_run_nonotch_")

    if not notch_dir or not nonotch_dir:
        print("ERROR: Cannot find both notch and nonotch RF run directories.")
        return 1

    out_dir = outputs_dir / f"ablation_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("STEP 12: NOTCH ABLATION STATISTICAL COMPARISON")
    print(f"  Notch run:    {notch_dir.name}")
    print(f"  NoNotch run:  {nonotch_dir.name}")
    print("=" * 70)

    with open(notch_dir / "rf_summary.json") as f:
        notch_data = json.load(f)
    with open(nonotch_dir / "rf_summary.json") as f:
        nonotch_data = json.load(f)

    # Build lookup by participant
    notch_by_pid = {r['participant']: r for r in notch_data['individuals']}
    nonotch_by_pid = {r['participant']: r for r in nonotch_data['individuals']}

    common_pids = sorted(set(notch_by_pid.keys()) & set(nonotch_by_pid.keys()))

    rows = []
    for pid in common_pids:
        n = notch_by_pid[pid]
        nn = nonotch_by_pid[pid]
        rows.append({
            'participant': pid,
            'notch_accuracy': n['accuracy'],
            'nonotch_accuracy': nn['accuracy'],
            'delta_accuracy': nn['accuracy'] - n['accuracy'],
            'notch_recall': n['recall'],
            'nonotch_recall': nn['recall'],
            'delta_recall': nn['recall'] - n['recall'],
        })

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / 'ablation_comparison.csv', index=False)

    # Print table
    print(f"\n{'PID':<6} {'Notch Acc':>10} {'NoNotch Acc':>12} {'Δ Acc':>8} "
          f"{'Notch Rec':>10} {'NoNotch Rec':>12} {'Δ Rec':>8}")
    print("-" * 70)
    for _, r in df.iterrows():
        sign_a = "+" if r['delta_accuracy'] >= 0 else ""
        sign_r = "+" if r['delta_recall'] >= 0 else ""
        print(f"{r['participant']:<6} {r['notch_accuracy']:>10.3f} {r['nonotch_accuracy']:>12.3f} "
              f"{sign_a}{r['delta_accuracy']:>7.3f} "
              f"{r['notch_recall']:>10.3f} {r['nonotch_recall']:>12.3f} "
              f"{sign_r}{r['delta_recall']:>7.3f}")

    improved_acc = (df['delta_accuracy'] > 0).sum()
    worsened_acc = (df['delta_accuracy'] < 0).sum()
    tied_acc = (df['delta_accuracy'] == 0).sum()

    lines = []
    lines.append("NOTCH ABLATION STATISTICAL COMPARISON")
    lines.append("=" * 70)
    lines.append(f"\nParticipants compared: {len(common_pids)}")
    lines.append(f"Notch aggregate:   acc={notch_data['aggregate']['mean_accuracy']:.4f}, "
                 f"rec={notch_data['aggregate']['mean_recall']:.4f}")
    lines.append(f"NoNotch aggregate: acc={nonotch_data['aggregate']['mean_accuracy']:.4f}, "
                 f"rec={nonotch_data['aggregate']['mean_recall']:.4f}")

    lines.append(f"\nAccuracy: {improved_acc} improved, {worsened_acc} worsened, {tied_acc} tied")
    lines.append(f"Mean Δ accuracy:  {df['delta_accuracy'].mean():+.4f}")
    lines.append(f"Mean Δ recall:    {df['delta_recall'].mean():+.4f}")

    # Wilcoxon signed-rank tests
    lines.append("\n" + "-" * 70)
    lines.append("Wilcoxon Signed-Rank Tests (paired, two-sided)")
    lines.append("-" * 70)

    for metric in ['accuracy', 'recall']:
        notch_vals = df[f'notch_{metric}'].values
        nonotch_vals = df[f'nonotch_{metric}'].values
        deltas = nonotch_vals - notch_vals

        try:
            w_stat, w_p = stats.wilcoxon(deltas)
            # Effect size r = Z / sqrt(N)
            z_score = stats.norm.ppf(w_p / 2)
            effect_r = abs(z_score) / np.sqrt(len(deltas))
            lines.append(f"\n  {metric.upper()}:")
            lines.append(f"    W = {w_stat:.1f}, p = {w_p:.4f}")
            lines.append(f"    Effect size r = {effect_r:.4f}")
            if w_p < 0.05:
                lines.append(f"    → SIGNIFICANT at α=0.05")
            else:
                lines.append(f"    → NOT significant at α=0.05 (the difference is noise)")
        except Exception as e:
            lines.append(f"\n  {metric.upper()}: Test failed — {e}")

    output_text = "\n".join(lines)
    with open(out_dir / 'ablation_statistical_test.txt', 'w') as f:
        f.write(output_text)

    print(f"\n{output_text}")
    print(f"\n{'=' * 70}")
    print(f"Outputs written to: {out_dir.name}")
    return 0


if __name__ == "__main__":
    main()
