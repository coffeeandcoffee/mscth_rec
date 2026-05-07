#!/usr/bin/env python3
"""
Step 13: EI Aggregate Computation & F1 Bug Investigation

Reads the existing ei_results.json, computes missing aggregate stats,
investigates the F1=0.0 bug, and outputs a corrected summary.

Outputs:
  - ei_aggregate_summary.csv   (per-participant + aggregate row)
  - ei_bug_report.txt          (F1 bug investigation)
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime


def find_latest_run_dir(base_dir, prefix):
    dirs = [d for d in base_dir.glob(f"{prefix}*") if d.is_dir()]
    if not dirs:
        return None
    return max(dirs, key=lambda d: d.stat().st_mtime)


def main():
    project_root = Path(__file__).resolve().parent.parent.parent
    outputs_dir = project_root / "04-26_data_analysis_and_results" / "outputs"

    ei_dir = find_latest_run_dir(outputs_dir, "ei_stats_")
    if not ei_dir:
        print("ERROR: No EI stats directory found.")
        return 1

    out_dir = outputs_dir / f"ei_corrected_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("STEP 13: EI AGGREGATE COMPUTATION & F1 BUG INVESTIGATION")
    print(f"Source: {ei_dir.name}")
    print("=" * 70)

    with open(ei_dir / "ei_results.json") as f:
        ei_data = json.load(f)

    rows = []
    bug_lines = []
    bug_lines.append("EI F1 SCORE BUG INVESTIGATION")
    bug_lines.append("=" * 70)

    for pid, data in ei_data.items():
        ml = data.get('intra_ei_ml', {})
        rows.append({
            'participant': pid,
            'stay_mean_ei': data['stay_mean'],
            'skip_mean_ei': data['skip_mean'],
            'ei_difference': data['stay_mean'] - data['skip_mean'],
            'mannwhitney_p': data['mannwhitney_p'],
            'cohens_d': data['cohens_d'],
            'significant_p05': data['mannwhitney_p'] < 0.05,
            'ml_accuracy': ml.get('accuracy', None),
            'ml_recall': ml.get('recall', None),
            'ml_precision': ml.get('precision', None),
            'ml_f1': ml.get('f1', None),
            'n_stay': data['stay_n'],
            'n_skip': data['skip_n'],
        })

        # F1 bug check
        if ml.get('f1', None) == 0.0:
            bug_lines.append(f"\n{pid}: F1=0.0 | acc={ml['accuracy']:.4f} | "
                             f"rec={ml['recall']:.4f} | prec={ml['precision']:.4f}")
            # F1 = 2 * (prec * rec) / (prec + rec)
            expected_f1 = 2 * ml['precision'] * ml['recall'] / (ml['precision'] + ml['recall']) \
                if (ml['precision'] + ml['recall']) > 0 else 0
            bug_lines.append(f"  Expected F1 = 2*{ml['precision']:.4f}*{ml['recall']:.4f} / "
                             f"({ml['precision']:.4f}+{ml['recall']:.4f}) = {expected_f1:.4f}")

    df = pd.DataFrame(rows)

    # Compute aggregates
    mean_acc = df['ml_accuracy'].mean()
    std_acc = df['ml_accuracy'].std()
    mean_rec = df['ml_recall'].mean()
    std_rec = df['ml_recall'].std()
    n_significant = df['significant_p05'].sum()

    # Add aggregate row
    agg_row = {
        'participant': 'AGGREGATE',
        'ml_accuracy': mean_acc,
        'ml_recall': mean_rec,
        'ml_precision': df['ml_precision'].mean(),
        'ml_f1': df['ml_f1'].mean(),
        'mannwhitney_p': None,
        'cohens_d': df['cohens_d'].mean(),
    }
    df_out = pd.concat([df, pd.DataFrame([agg_row])], ignore_index=True)
    df_out.to_csv(out_dir / 'ei_aggregate_summary.csv', index=False)

    print(f"\n{'─' * 70}")
    print("ENGAGEMENT INDEX AGGREGATE RESULTS")
    print(f"{'─' * 70}")
    print(f"  Mean EI ML Accuracy:  {mean_acc:.4f} ± {std_acc:.4f}")
    print(f"  Mean EI ML Recall:    {mean_rec:.4f} ± {std_rec:.4f}")
    print(f"  Participants with significant EI difference (p<0.05): {n_significant}/25")
    print(f"\n  Per-participant EI direction:")

    for _, r in df.iterrows():
        direction = "STAY > SKIP" if r['ei_difference'] > 0 else "SKIP > STAY"
        sig = "*" if r['significant_p05'] else " "
        print(f"    {r['participant']:<5} {direction}  d={r['cohens_d']:+.3f}  "
              f"p={r['mannwhitney_p']:.4f} {sig}  acc={r['ml_accuracy']:.3f}")

    # Bug report
    bug_lines.append(f"\n{'─' * 70}")
    bug_lines.append("DIAGNOSIS:")
    bug_lines.append("The F1 scores are all 0.0 in the original JSON despite non-zero")
    bug_lines.append("precision and recall values. This is a BUG in step4_engagement_index.py.")
    bug_lines.append("The F1 was likely hardcoded to 0.0 or computed incorrectly.")
    bug_lines.append(f"\nCorrected aggregate F1 would be computed from the per-participant")
    bug_lines.append(f"precision and recall values shown above.")

    # Compute corrected F1 for each participant
    corrected_f1s = []
    for _, r in df.iterrows():
        p, rec = r['ml_precision'], r['ml_recall']
        f1 = 2 * p * rec / (p + rec) if (p + rec) > 0 else 0
        corrected_f1s.append(f1)
    bug_lines.append(f"\nCorrected mean F1: {np.mean(corrected_f1s):.4f} ± {np.std(corrected_f1s):.4f}")

    bug_text = "\n".join(bug_lines)
    with open(out_dir / 'ei_bug_report.txt', 'w') as f:
        f.write(bug_text)

    print(f"\n{bug_text}")
    print(f"\n{'=' * 70}")
    print(f"Outputs written to: {out_dir.name}")
    return 0


if __name__ == "__main__":
    main()
