#!/usr/bin/env python3
"""
Step 11: Deep Feature Importance Analysis (Numerical Evidence)

Loads all 25 trained RF .pkl models from the nonotch run and produces
comprehensive numerical data files backing the feature importance claims.

Outputs:
  - feature_importance_full_table.csv   (112 features ranked)
  - feature_importance_per_participant.csv (25 rows × 112 cols)
  - band_importance_summary.csv         (7 bands aggregated)
  - electrode_importance_summary.csv    (4 electrodes aggregated)
  - electrode_band_matrix.csv           (4×7 = 28 cells)
  - statistical_tests.txt              (Friedman + pairwise Wilcoxon)
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from datetime import datetime
from scipy import stats
from itertools import combinations

import utils


def find_latest_run_dir(base_dir, prefix):
    dirs = [d for d in base_dir.glob(f"{prefix}*") if d.is_dir()]
    if not dirs:
        return None
    return max(dirs, key=lambda d: d.stat().st_mtime)


def recreate_feature_names():
    """Reproduce the exact 112 aggregated feature names from the pipeline."""
    base_features = []
    for ch in utils.EEG_CHANNELS:
        for band_name, _, _ in utils.FREQUENCY_BANDS:
            base_features.append(f"{ch}_{band_name}")

    agg_names = []
    for fname in base_features:
        agg_names.extend([f"{fname}_mean", f"{fname}_std",
                          f"{fname}_min", f"{fname}_max"])
    return agg_names


def parse_feature_name(feat_name):
    """Split 'AF7_high_gamma_std' into electrode, band, stat_type."""
    # Feature names are like: TP9_delta_mean, AF7_high_gamma_std, etc.
    parts = feat_name.split('_')
    electrode = parts[0]
    stat_type = parts[-1]  # mean, std, min, max
    band = '_'.join(parts[1:-1])  # handles 'high_gamma', 'low_gamma', 'very_high'
    return electrode, band, stat_type


def main():
    project_root = Path(__file__).resolve().parent.parent.parent
    outputs_dir = project_root / "04-26_data_analysis_and_results" / "outputs"

    # Find the nonotch RF run (contains the .pkl models)
    run_dir = find_latest_run_dir(outputs_dir, "rf_run_nonotch_")
    if not run_dir:
        print("ERROR: No nonotch RF run found.")
        return 1

    out_dir = outputs_dir / f"deep_importance_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("STEP 11: DEEP FEATURE IMPORTANCE ANALYSIS")
    print(f"Source: {run_dir.name}")
    print(f"Output: {out_dir.name}")
    print("=" * 70)

    # Load all models
    model_files = sorted(run_dir.glob("P*_rf_model.pkl"))
    feature_names = recreate_feature_names()
    n_features = len(feature_names)

    participant_ids = []
    all_importances = []

    for pkl_file in model_files:
        pid = pkl_file.stem.replace('_rf_model', '')
        clf = joblib.load(pkl_file)
        if hasattr(clf, 'feature_importances_'):
            participant_ids.append(pid)
            all_importances.append(clf.feature_importances_)

    all_importances = np.array(all_importances)  # shape: (n_participants, 112)
    n_participants = len(participant_ids)
    print(f"\n✓ Loaded {n_participants} models with {n_features} features each.\n")

    # =========================================================================
    # OUTPUT 1: Full Feature Table (112 features ranked by mean importance)
    # =========================================================================
    mean_imp = np.mean(all_importances, axis=0)
    std_imp = np.std(all_importances, axis=0)
    median_imp = np.median(all_importances, axis=0)
    min_imp = np.min(all_importances, axis=0)
    max_imp = np.max(all_importances, axis=0)

    electrodes, bands, stat_types = [], [], []
    for fn in feature_names:
        e, b, s = parse_feature_name(fn)
        electrodes.append(e)
        bands.append(b)
        stat_types.append(s)

    df_full = pd.DataFrame({
        'feature_name': feature_names,
        'electrode': electrodes,
        'band': bands,
        'stat_type': stat_types,
        'mean_importance': mean_imp,
        'std_importance': std_imp,
        'median_importance': median_imp,
        'min_importance': min_imp,
        'max_importance': max_imp,
    })
    df_full = df_full.sort_values('mean_importance', ascending=False).reset_index(drop=True)
    df_full.index += 1  # 1-indexed rank
    df_full.index.name = 'rank'
    df_full.to_csv(out_dir / 'feature_importance_full_table.csv')

    print("─" * 70)
    print("TOP 20 FEATURES BY MEAN GINI IMPORTANCE")
    print("─" * 70)
    for i, row in df_full.head(20).iterrows():
        print(f"  {i:>2}. {row['feature_name']:<35s}  "
              f"mean={row['mean_importance']:.5f}  std={row['std_importance']:.5f}  "
              f"[{row['electrode']}] [{row['band']}] [{row['stat_type']}]")

    # =========================================================================
    # OUTPUT 2: Per-Participant Feature Importance Matrix
    # =========================================================================
    df_per_participant = pd.DataFrame(all_importances,
                                       index=participant_ids,
                                       columns=feature_names)
    df_per_participant.index.name = 'participant'
    df_per_participant.to_csv(out_dir / 'feature_importance_per_participant.csv')
    print(f"\n✓ Per-participant matrix saved ({n_participants} × {n_features})")

    # =========================================================================
    # OUTPUT 3: Band Importance Summary
    # =========================================================================
    band_names = [b[0] for b in utils.FREQUENCY_BANDS]
    band_participant_importance = {b: np.zeros(n_participants) for b in band_names}

    for idx, fn in enumerate(feature_names):
        _, band, _ = parse_feature_name(fn)
        for p in range(n_participants):
            band_participant_importance[band][p] += all_importances[p, idx]

    total_per_participant = np.sum(all_importances, axis=1)

    band_rows = []
    for band in band_names:
        raw_vals = band_participant_importance[band]
        pct_vals = raw_vals / total_per_participant * 100
        band_rows.append({
            'band': band,
            'mean_aggregate_importance': np.mean(raw_vals),
            'std_aggregate_importance': np.std(raw_vals),
            'mean_pct_of_total': np.mean(pct_vals),
            'std_pct_of_total': np.std(pct_vals),
            'median_pct_of_total': np.median(pct_vals),
        })

    df_bands = pd.DataFrame(band_rows).sort_values('mean_pct_of_total', ascending=False)
    df_bands.to_csv(out_dir / 'band_importance_summary.csv', index=False)

    print("\n" + "─" * 70)
    print("IMPORTANCE BY FREQUENCY BAND (% of total)")
    print("─" * 70)
    for _, row in df_bands.iterrows():
        bar = "█" * int(row['mean_pct_of_total'] / 2)
        print(f"  {row['band']:<14s}  {row['mean_pct_of_total']:>5.1f}% ± {row['std_pct_of_total']:.1f}%  {bar}")

    # =========================================================================
    # OUTPUT 4: Electrode Importance Summary
    # =========================================================================
    electrode_participant_importance = {ch: np.zeros(n_participants) for ch in utils.EEG_CHANNELS}

    for idx, fn in enumerate(feature_names):
        electrode, _, _ = parse_feature_name(fn)
        for p in range(n_participants):
            electrode_participant_importance[electrode][p] += all_importances[p, idx]

    elec_rows = []
    for ch in utils.EEG_CHANNELS:
        raw_vals = electrode_participant_importance[ch]
        pct_vals = raw_vals / total_per_participant * 100
        elec_rows.append({
            'electrode': ch,
            'mean_aggregate_importance': np.mean(raw_vals),
            'std_aggregate_importance': np.std(raw_vals),
            'mean_pct_of_total': np.mean(pct_vals),
            'std_pct_of_total': np.std(pct_vals),
        })

    df_elec = pd.DataFrame(elec_rows).sort_values('mean_pct_of_total', ascending=False)
    df_elec.to_csv(out_dir / 'electrode_importance_summary.csv', index=False)

    print("\n" + "─" * 70)
    print("IMPORTANCE BY ELECTRODE (% of total)")
    print("─" * 70)
    for _, row in df_elec.iterrows():
        bar = "█" * int(row['mean_pct_of_total'])
        print(f"  {row['electrode']:<6s}  {row['mean_pct_of_total']:>5.1f}% ± {row['std_pct_of_total']:.1f}%  {bar}")

    # =========================================================================
    # OUTPUT 5: Electrode × Band Matrix (28 cells)
    # =========================================================================
    matrix_data = {}
    for ch in utils.EEG_CHANNELS:
        for band in band_names:
            key = f"{ch}_{band}"
            matrix_data[key] = np.zeros(n_participants)

    for idx, fn in enumerate(feature_names):
        electrode, band, _ = parse_feature_name(fn)
        key = f"{electrode}_{band}"
        for p in range(n_participants):
            matrix_data[key][p] += all_importances[p, idx]

    matrix_rows = []
    for ch in utils.EEG_CHANNELS:
        row = {'electrode': ch}
        for band in band_names:
            key = f"{ch}_{band}"
            pct_vals = matrix_data[key] / total_per_participant * 100
            row[band] = f"{np.mean(pct_vals):.2f}"
        matrix_rows.append(row)

    df_matrix = pd.DataFrame(matrix_rows)
    df_matrix.to_csv(out_dir / 'electrode_band_matrix.csv', index=False)

    print("\n" + "─" * 70)
    print("ELECTRODE × BAND IMPORTANCE MATRIX (mean % of total)")
    print("─" * 70)
    header = f"{'':>6s}" + "".join(f"{b:>12s}" for b in band_names)
    print(header)
    for _, row in df_matrix.iterrows():
        line = f"{row['electrode']:>6s}"
        for band in band_names:
            line += f"{float(row[band]):>12.2f}"
        print(line)

    # =========================================================================
    # OUTPUT 6: Statistical Tests
    # =========================================================================
    test_lines = []
    test_lines.append("STATISTICAL TESTS FOR BAND IMPORTANCE DIFFERENCES")
    test_lines.append("=" * 70)

    # Friedman test (non-parametric repeated measures across 7 bands)
    band_arrays = [band_participant_importance[b] / total_per_participant for b in band_names]
    try:
        friedman_stat, friedman_p = stats.friedmanchisquare(*band_arrays)
        test_lines.append(f"\nFriedman Test (7 bands × {n_participants} participants):")
        test_lines.append(f"  χ² = {friedman_stat:.4f}, p = {friedman_p:.2e}")
        test_lines.append(f"  Conclusion: {'Significant' if friedman_p < 0.05 else 'NOT significant'} "
                          f"difference across bands (α=0.05)")
    except Exception as e:
        test_lines.append(f"\nFriedman Test failed: {e}")

    # Pairwise Wilcoxon: high_gamma vs every other band
    test_lines.append(f"\nPairwise Wilcoxon Signed-Rank: high_gamma vs. each other band")
    test_lines.append("-" * 70)
    hg_vals = band_participant_importance['high_gamma'] / total_per_participant

    for band in band_names:
        if band == 'high_gamma':
            continue
        other_vals = band_participant_importance[band] / total_per_participant
        try:
            w_stat, w_p = stats.wilcoxon(hg_vals, other_vals)
            direction = "higher" if np.mean(hg_vals) > np.mean(other_vals) else "LOWER"
            effect_r = w_stat / (n_participants * (n_participants + 1) / 2)
            test_lines.append(
                f"  high_gamma vs {band:<14s}: W={w_stat:>6.1f}, p={w_p:.4e}, "
                f"high_gamma is {direction} (r={effect_r:.3f})")
        except Exception as e:
            test_lines.append(f"  high_gamma vs {band:<14s}: FAILED ({e})")

    # Also test: frontal vs temporal
    test_lines.append(f"\nWilcoxon Signed-Rank: Frontal (AF7+AF8) vs Temporal (TP9+TP10)")
    test_lines.append("-" * 70)
    frontal = (electrode_participant_importance['AF7'] +
               electrode_participant_importance['AF8']) / total_per_participant
    temporal = (electrode_participant_importance['TP9'] +
                electrode_participant_importance['TP10']) / total_per_participant
    try:
        w_stat, w_p = stats.wilcoxon(frontal, temporal)
        direction = "higher" if np.mean(frontal) > np.mean(temporal) else "LOWER"
        test_lines.append(f"  W={w_stat:.1f}, p={w_p:.4e}, Frontal is {direction}")
    except Exception as e:
        test_lines.append(f"  FAILED: {e}")

    test_output = "\n".join(test_lines)
    with open(out_dir / 'statistical_tests.txt', 'w') as f:
        f.write(test_output)

    print("\n" + "─" * 70)
    print(test_output)

    print(f"\n{'=' * 70}")
    print(f"All outputs written to: {out_dir.name}")
    print(f"{'=' * 70}")
    return 0


if __name__ == "__main__":
    main()
