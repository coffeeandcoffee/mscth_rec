#!/usr/bin/env python3
"""
Baseline Normalized RF Training — Step 11
===========================================
Trains V4 Random Forest (200 trees, depth=7, no notch) individually on each
of the 25 included participants, but WITH a fundamental pre-processing step:
Baseline Normalization.

For each sub-recording:
  1. Locates the 100-second baseline (starting 10s after first 'B' press).
  2. Extracts the 28 frequency bands for this 100s pure-rest period.
  3. Calculates the 'Baseline Mean' for each of the 28 bands.
  4. Extracts 3-second task samples (skip/noskip) identically as before.
  5. NORMALIZES the task features against the Baseline Means before training.

Produces:
  - results.json                    — per-participant train + val metrics
  - images/boxplot_train_vs_val.png — publication-ready box plots
  - images/per_participant_table.png— summary table
  - description.txt                 — findings summary

Run:  cd <this folder> && python analysis.py
Debug: cd <this folder> && python analysis.py --debug-one P4
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal as sig
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent.parent / "data"
MASK_JSON = SCRIPT_DIR.parent / "20260311_145900_exclusion_mask" / "exclusion_mask.json"
SAMPLE_JSON = SCRIPT_DIR.parent / "20260311_150100_sample_classification" / "sample_classification.json"
IMAGES_DIR = SCRIPT_DIR / "images"
IMAGES_DIR.mkdir(exist_ok=True)

# Baseline Config
BASELINE_REQUIRED_S = 100
BASELINE_OFFSET_S = 10

# RF hyper-parameters (V4 Run-2)
N_ESTIMATORS = 200
MAX_DEPTH = 7
MIN_SAMPLES_LEAF = 5
WINDOW_S = 3.0
OVERLAP = 0.80
STRIDE_S = WINDOW_S * (1 - OVERLAP)
TARGET_SAMPLES = int(256 * WINDOW_S)
SEED = 42

EEG_CHANNELS = ["TP9", "AF7", "AF8", "TP10"]
FREQUENCY_BANDS = [
    ("delta", 1, 4),
    ("theta", 4, 8),
    ("alpha", 8, 13),
    ("beta", 13, 30),
    ("low_gamma", 30, 40),
    ("high_gamma", 40, 60),
    ("very_high", 60, 100),
]


# ---------------------------------------------------------------------------
# 1. Frequency band extraction
# ---------------------------------------------------------------------------
def extract_band_power(data, fs, lo, hi):
    ny = fs / 2
    lo_n = max(lo / ny, 0.01)
    hi_n = min(hi / ny, 0.99)
    if lo_n >= hi_n or len(data) < 20:
        return np.zeros_like(data)
    try:
        b, a = sig.butter(4, [lo_n, hi_n], btype="band")
        return sig.filtfilt(b, a, data)
    except Exception:
        return np.zeros_like(data)


def add_frequency_bands(df: pd.DataFrame):
    """Add 28 frequency band columns (7 bands × 4 channels) to DataFrame."""
    td = np.diff(df["lsl_timestamp"].values)
    fs = 1.0 / np.median(td)
    feat_names = []
    for ch in EEG_CHANNELS:
        cd = df[ch].values
        for bname, lo, hi in FREQUENCY_BANDS:
            col_name = f"{ch}_{bname}"
            df[col_name] = extract_band_power(cd, fs, lo, hi)
            feat_names.append(col_name)
    return df, feat_names, fs


# ---------------------------------------------------------------------------
# 2. Baseline Calculation
# ---------------------------------------------------------------------------
def calculate_baseline_means(df: pd.DataFrame, feat_names: list[str]) -> dict[str, float] | None:
    """
    Locates the 100s baseline block (10s after first B press) 
    and returns the mean amplitude for each of the 28 frequency bands.
    Returns None if no keypress_B is found in this dataframe.
    """
    if "keypress_B" not in df.columns:
        return None
        
    b_presses = df[df["keypress_B"] == 1]
    if b_presses.empty:
        return None
        
    first_b_ts = float(b_presses.iloc[0]["lsl_timestamp"])
    base_start = first_b_ts + BASELINE_OFFSET_S
            
    base_end = base_start + BASELINE_REQUIRED_S
    
    # Extract the baseline chunk
    ts = df["lsl_timestamp"].values
    mask = (ts >= base_start) & (ts < base_end)
    indices = np.where(mask)[0]
    
    baseline_means = {}
    if len(indices) < 256 * 10: # If we have less than 10 seconds of baseline
        print(f"      ❌ Baseline segment too short! (Found {len(indices)} samples).")
        return None

    # Calculate the mean for each band over the 100s resting state
    for fn in feat_names:
        baseline_means[fn] = float(np.mean(df[fn].values[indices]))
        
        # Prevent zero-division in normalization
        if baseline_means[fn] == 0:
            baseline_means[fn] = 1e-6
            
    return baseline_means


# ---------------------------------------------------------------------------
# 3. Sample extraction WITH NORMALIZATION
# ---------------------------------------------------------------------------
def interpolate_block(ts, data_cols, target_n=TARGET_SAMPLES):
    t_start, t_end = ts[0], ts[-1]
    uniform_t = np.linspace(t_start, t_end, target_n)
    result = {}
    for col_name, vals in data_cols.items():
        result[col_name] = np.interp(uniform_t, ts, vals)
    return result


def extract_normalized_samples(df, block, feat_names, baseline_means):
    """
    Extract sliding-window samples from a single class block.
    Normalizes the 112 extracted features using the 28 baseline means.
    """
    start_t = block["start_t"]
    end_t = block["end_t"]
    duration = block["duration_s"]
    expected_n = block["n_samples"]

    if expected_n == 0 or duration < WINDOW_S:
        return []

    ts = df["lsl_timestamp"].values
    samples = []

    n_windows = int(math.floor((duration - WINDOW_S) / STRIDE_S)) + 1
    for w in range(n_windows):
        w_start = start_t + w * STRIDE_S
        w_end = w_start + WINDOW_S

        mask = (ts >= w_start) & (ts < w_end)
        indices = np.where(mask)[0]

        if len(indices) < 10:
            continue

        block_ts = ts[indices]
        block_data = {fn: df[fn].values[indices] for fn in feat_names}
        interp_data = interpolate_block(block_ts, block_data, TARGET_SAMPLES)

        # Aggregate and NORMALIZE
        agg = []
        for fn in feat_names:
            col = interp_data[fn]
            
            # 1. Calculate raw stats for this 3s window
            raw_mean = np.mean(col)
            raw_std = np.std(col)
            raw_min = np.min(col)
            raw_max = np.max(col)
            
            # 2. The participant's resting state for this specific frequency band
            b_mu = baseline_means[fn]
            
            # 3. Apply Relative Power Normalization (Percentage Change vs Rest)
            norm_mean = (raw_mean - b_mu) / abs(b_mu)
            norm_min = (raw_min - b_mu) / abs(b_mu)
            norm_max = (raw_max - b_mu) / abs(b_mu)
            
            # Std is scaled by the baseline mean (Coefficient of Variation) 
            # to reflect variance relative to the resting state magnitude.
            norm_std = raw_std / abs(b_mu)
            
            agg.extend([norm_mean, norm_std, norm_min, norm_max])
            
        samples.append(agg)

    return samples


# ---------------------------------------------------------------------------
# 4. Rebalance & Data Prep
# ---------------------------------------------------------------------------
def rebalance_pools(skip_pool, noskip_pool, seed=SEED):
    rng = np.random.RandomState(seed)
    n_skip = len(skip_pool)
    n_noskip = len(noskip_pool)
    if n_skip == n_noskip: return skip_pool, noskip_pool
    if n_skip > n_noskip:
        sel = rng.choice(n_skip, size=n_noskip, replace=False)
        return [skip_pool[i] for i in sel], noskip_pool
    else:
        sel = rng.choice(n_noskip, size=n_skip, replace=False)
        return skip_pool, [noskip_pool[i] for i in sel]


def build_train_val(skip_pool, noskip_pool, seed=SEED):
    rng = np.random.RandomState(seed)
    
    skip_idx = np.arange(len(skip_pool))
    noskip_idx = np.arange(len(noskip_pool))
    rng.shuffle(skip_idx)
    rng.shuffle(noskip_idx)
    skip_pool = [skip_pool[i] for i in skip_idx]
    noskip_pool = [noskip_pool[i] for i in noskip_idx]

    n_skip = len(skip_pool)
    n_noskip = len(noskip_pool)
    split_skip = int(round(n_skip * 0.6))
    split_noskip = int(round(n_noskip * 0.6))

    skip_train, skip_val = skip_pool[:split_skip], skip_pool[split_skip:]
    noskip_train, noskip_val = noskip_pool[:split_noskip], noskip_pool[split_noskip:]

    X_train = np.array(skip_train + noskip_train)
    y_train = np.array([1] * len(skip_train) + [0] * len(noskip_train))
    X_val = np.array(skip_val + noskip_val)
    y_val = np.array([1] * len(skip_val) + [0] * len(noskip_val))

    tr_idx = np.arange(len(y_train))
    va_idx = np.arange(len(y_val))
    rng.shuffle(tr_idx)
    rng.shuffle(va_idx)
    X_train, y_train = X_train[tr_idx], y_train[tr_idx]
    X_val, y_val = X_val[va_idx], y_val[va_idx]

    return X_train, y_train, X_val, y_val


def train_and_evaluate(X_tr, y_tr, X_va, y_va, seed=SEED):
    clf = RandomForestClassifier(
        n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH,
        min_samples_leaf=MIN_SAMPLES_LEAF, random_state=seed,
        class_weight="balanced", n_jobs=-1,
    )
    clf.fit(X_tr, y_tr)

    def metrics(yt, yp):
        return {
            "accuracy": round(accuracy_score(yt, yp), 4),
            "precision": round(precision_score(yt, yp, zero_division=0), 4),
            "recall": round(recall_score(yt, yp, zero_division=0), 4),
            "f1": round(f1_score(yt, yp, zero_division=0), 4),
        }

    train_m = metrics(y_tr, clf.predict(X_tr))
    val_m = metrics(y_va, clf.predict(X_va))
    return {
        "train": train_m, "val": val_m,
        "n_train": len(y_tr), "n_val": len(y_va),
        "n_total_balanced": len(y_tr) + len(y_va),
    }


# ---------------------------------------------------------------------------
# 5. Pipeline Execution per Participant
# ---------------------------------------------------------------------------
def process_participant(pid, sub_recordings, debug_dir=None):
    skip_pool = []
    noskip_pool = []
    agg_names = None
    baseline_means = None
    
    # 1. First Pass: Find the global Baseline Means for this Participant
    for sr in sub_recordings:
        csv_name = sr["file"]
        csv_path = DATA_DIR / csv_name
        if not csv_path.exists(): continue
        
        df = pd.read_csv(csv_path)
        if "lsl_timestamp" not in df.columns: continue
        
        df, feat_names, fs = add_frequency_bands(df)
        b_means = calculate_baseline_means(df, feat_names)
        
        if b_means is not None:
            baseline_means = b_means
            print(f"  ✅ Extracted 100s Baseline uniquely from {csv_name}")
            break
            
    if baseline_means is None:
        print(f"  ❌ CRITICAL: Could not locate a valid 100s keypress_B baseline in any of {pid}'s recordings. Skipping participant.")
        return None

    # 2. Second Pass: Extract features and normalize using the global baseline
    for sr in sub_recordings:
        csv_name = sr["file"]
        blocks = sr["blocks"]
        csv_path = DATA_DIR / csv_name

        if not csv_path.exists():
            print(f"  ⚠️ File not found: {csv_name} — skipping sub-recording")
            continue

        df = pd.read_csv(csv_path)
        print(f"  {csv_name}: {len(df):,} rows, {len(blocks)} blocks")

        required = ["lsl_timestamp"] + EEG_CHANNELS
        if any(c not in df.columns for c in required):
            print(f"    ❌ Missing required columns — skipping")
            continue

        # Extract 28 frequency bands
        df, feat_names, fs = add_frequency_bands(df)
        if agg_names is None:
            agg_names = []
            for fn in feat_names:
                agg_names.extend([f"{fn}_mean_norm", f"{fn}_std_scale", f"{fn}_min_norm", f"{fn}_max_norm"])

        # Extract normalized samples using the global baseline
        sub_skip = 0
        sub_noskip = 0
        for blk in blocks:
            samples = extract_normalized_samples(df, blk, feat_names, baseline_means)
            if blk["label"] == "about_to_skip":
                skip_pool.extend(samples)
                sub_skip += len(samples)
            else:
                noskip_pool.extend(samples)
                sub_noskip += len(samples)

        print(f"    Extracted (Normalized): skip={sub_skip}, noskip={sub_noskip}")

    if len(skip_pool) < 5:
        print(f"  ❌ Too few skip samples ({len(skip_pool)}) — SKIPPING")
        return None

    # Rebalance & Split
    skip_pool, noskip_pool = rebalance_pools(skip_pool, noskip_pool, SEED)
    print(f"  Balanced: {len(skip_pool)} per class")

    X_tr, y_tr, X_va, y_va = build_train_val(skip_pool, noskip_pool, SEED)
    
    # Train
    res = train_and_evaluate(X_tr, y_tr, X_va, y_va, SEED)
    
    # Save the pools to temp JSON for Explainability (Step 15)
    pool_data = {
        "skip_pool": skip_pool,
        "noskip_pool": noskip_pool
    }
    out_dir = Path(__file__).resolve().parent
    with open(out_dir / f"{pid}_pools_{out_dir.name}.json.tmp", "w") as f:
        json.dump(pool_data, f)
        
    print(f"  Val acc={res['val']['accuracy']:.1%}  Val Rec={res['val']['recall']:.1%}")
    return res


# ---------------------------------------------------------------------------
# 6. Presentation
# ---------------------------------------------------------------------------
def plot_boxplots(results: dict):
    metric_names = ["accuracy", "precision", "recall", "f1"]
    labels_pretty = ["Accuracy", "Precision", "Recall", "F1-Score"]

    fig, axes = plt.subplots(1, 4, figsize=(16, 5))
    fig.suptitle(
        f"Baseline-Normalized RF Performance (Intra-Subject, n={len(results)})",
        fontsize=13, fontweight="bold",
    )

    for ax, mkey, mlabel in zip(axes, metric_names, labels_pretty):
        train_vals = [r["train"][mkey] for r in results.values()]
        val_vals = [r["val"][mkey] for r in results.values()]

        bp = ax.boxplot(
            [train_vals, val_vals],
            labels=["Train", "Validation"],
            patch_artist=True,
            widths=0.5,
            medianprops=dict(color="black", linewidth=1.5),
        )
        bp["boxes"][0].set_facecolor("#7fbf7f")
        bp["boxes"][1].set_facecolor("#7f9fbf")

        for i, vals in enumerate([train_vals, val_vals], 1):
            jitter = np.random.default_rng(42).uniform(-0.08, 0.08, len(vals))
            ax.scatter(
                np.full(len(vals), i) + jitter, vals,
                alpha=0.5, s=18, c="black", zorder=3,
            )

        ax.axhline(0.5, ls="--", c="red", alpha=0.4, lw=1)
        ax.set_ylabel(mlabel)
        ax.set_title(mlabel)
        ax.set_ylim(0.3, 1.05)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    out = IMAGES_DIR / "boxplot_train_vs_val.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"✅ Saved: {out}")


def plot_table(results: dict):
    pids = sorted(results.keys(), key=lambda p: int(p[1:]))
    col_labels = [
        "P", "Samples",
        "Train\nAcc", "Train\nPrec", "Train\nRec", "Train\nF1",
        "Val\nAcc", "Val\nPrec", "Val\nRec", "Val\nF1",
    ]
    rows = []
    colors = []
    for pid in pids:
        r = results[pid]
        t, v = r["train"], r["val"]
        row = [
            pid, str(r["n_total_balanced"]),
            f"{t['accuracy']:.1%}", f"{t['precision']:.1%}", f"{t['recall']:.1%}", f"{t['f1']:.1%}",
            f"{v['accuracy']:.1%}", f"{v['precision']:.1%}", f"{v['recall']:.1%}", f"{v['f1']:.1%}",
        ]
        rows.append(row)
        bg = "#d4edda" if v["accuracy"] >= 0.60 else (
            "#fff3cd" if v["accuracy"] >= 0.50 else "#f8d7da"
        )
        colors.append([bg] * len(col_labels))

    # Aggregate row
    t_accs = [results[p]["train"]["accuracy"] for p in pids]
    v_accs = [results[p]["val"]["accuracy"] for p in pids]
    t_f1s = [results[p]["train"]["f1"] for p in pids]
    v_f1s = [results[p]["val"]["f1"] for p in pids]
    rows.append([
        "MEAN", "—",
        f"{np.mean(t_accs):.1%}", "—", "—", f"{np.mean(t_f1s):.1%}",
        f"{np.mean(v_accs):.1%}", "—", "—", f"{np.mean(v_f1s):.1%}",
    ])
    colors.append(["#e8f0fe"] * len(col_labels))

    n_rows = len(rows)
    fig_h = max(6, 0.40 * n_rows + 2)
    fig, ax = plt.subplots(figsize=(16, fig_h))
    ax.axis("off")
    ax.set_title(
        f"Baseline-Normalized Per-Participant RF Results (n={len(results)})\n"
        f"[{N_ESTIMATORS} trees, depth={MAX_DEPTH}]",
        fontsize=12, fontweight="bold", pad=20,
    )
    table = ax.table(
        cellText=rows, colLabels=col_labels, cellColours=colors,
        colColours=["#4472c4"] * len(col_labels),
        loc="center", cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.4)
    for ci in range(len(col_labels)):
        cell = table[0, ci]
        cell.set_text_props(color="white", fontweight="bold", fontsize=8)
        cell.set_edgecolor("white")
    fig.tight_layout()
    out = IMAGES_DIR / "per_participant_table.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"✅ Saved: {out}")


def write_description(results: dict):
    pids = sorted(results.keys(), key=lambda p: int(p[1:]))
    v_accs = [results[p]["val"]["accuracy"] for p in pids]
    v_recs = [results[p]["val"]["recall"] for p in pids]

    lines = [
        "=" * 70,
        "BASELINE NORMALIZED PER-PARTICIPANT RF (Intra-Subject)",
        "=" * 70, "",
        "OBJECTIVE", "-" * 9,
        "To test if applying a 100-sec Baseline Normalization per sub-recording",
        "improves or degrades the intra-subject predictive power. This normalizes",
        "the raw absolute EEG amplitude into a 'relative power' percentage change",
        "vs the individual's resting state.", "",
        "METHODOLOGY", "-" * 11,
        "  1. Isolate 100s baseline (10s after first B press).",
        "  2. Calculate the 'Resting Mean' for each of the 28 freq bands.",
        "  3. Extract 3-second task blocks (skip/noskip) + calculate stats.",
        "  4. Normalization Formula:",
        "     Normalized = (Task_Stat - Resting_Mean) / |Resting_Mean|",
        "     Std Devs = Task_Std / |Resting_Mean|",
        "  5. Train Intra-subject Custom RF models on these 112 relative features.", "",
        "AGGREGATE RESULTS", "-" * 17,
        f"  Participants:     {len(results)}",
        f"  Val Accuracy:     {np.mean(v_accs):.1%} ± {np.std(v_accs):.1%}",
        f"  Val Recall:       {np.mean(v_recs):.1%} ± {np.std(v_recs):.1%}",
        f"  Beat 50% baseline:{sum(1 for a in v_accs if a > 0.5)}/{len(v_accs)}", "",
        "PER-PARTICIPANT", "-" * 15,
    ]
    for pid in pids:
        r = results[pid]
        lines.append(
            f"  {pid}: val_acc={r['val']['accuracy']:.1%}  "
            f"val_rec={r['val']['recall']:.1%}  "
            f"n={r['n_total_balanced']}"
        )
    lines += [
        "", "=" * 70,
        "Generated by analysis.py (Step 11: baseline_normalized_rf)",
        "=" * 70,
    ]
    out = SCRIPT_DIR / "description.txt"
    out.write_text("\n".join(lines))
    print(f"✅ Saved: {out}")


# ---------------------------------------------------------------------------
# Main Router
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug-one", type=str, default=None)
    args = parser.parse_args()

    print("=" * 60)
    print("STEP 11: BASELINE NORMALIZED RF (Intra-Subject, n=25)")
    print("=" * 60)

    mask = json.loads(MASK_JSON.read_text())
    included = set(mask["included"])

    sample_cls = json.loads(SAMPLE_JSON.read_text())
    participants_data = sample_cls["participants"]

    if args.debug_one:
        sorted_pids = [args.debug_one.upper()]
    else:
        sorted_pids = sorted([p for p in participants_data if p in included], key=lambda p: int(p[1:]))

    all_results = {}
    for i, pid in enumerate(sorted_pids, 1):
        print(f"\n{'='*60}\n[{i}/{len(sorted_pids)}] {pid}\n{'='*60}")
        sub_recs = participants_data[pid]["sub_recordings"]
        res = process_participant(pid, sub_recs)
        if res is not None:
            all_results[pid] = res

    if not args.debug_one:
        json_out = SCRIPT_DIR / "results.json"
        json_out.write_text(json.dumps({
            "config": {
                "model": "RandomForestClassifier",
                "normalization": "Baseline (100s)",
                "n_estimators": N_ESTIMATORS,
                "max_depth": MAX_DEPTH,
                "min_samples_leaf": MIN_SAMPLES_LEAF,
                "window_s": WINDOW_S,
                "overlap": OVERLAP,
                "stride_s": STRIDE_S,
                "interpolation_samples": TARGET_SAMPLES,
                "notch_filter": False,
                "train_val_split": "60/40 per pool",
                "seed": SEED
            },
            "participants": all_results,
        }, indent=2))
        plot_boxplots(all_results)
        plot_table(all_results)
        write_description(all_results)
        print("\n✅ Baseline Normalized Pipeline Complete!")

if __name__ == "__main__":
    main()
