#!/usr/bin/env python3
"""
Per-Participant RF Training — Step 6 (corrected pipeline)
=========================================================
Trains V4 Random Forest (200 trees, depth=7, no notch) individually on each
of the 25 included participants. Produces:
  - results.json              — per-participant train + val metrics
  - images/boxplot_train_vs_val.png — publication-ready box plots
  - images/per_participant_table.png — summary table
  - description.txt           — findings summary

Run:  cd <this folder> && python analysis.py
Debug: cd <this folder> && python analysis.py --debug-one P7
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

# RF hyper-parameters (V4 Run-2 — best from feasibility)
N_ESTIMATORS = 200
MAX_DEPTH = 7
MIN_SAMPLES_LEAF = 5
WINDOW_S = 3.0
OVERLAP = 0.80
STRIDE_S = WINDOW_S * (1 - OVERLAP)  # 0.6s
TARGET_SAMPLES = int(256 * WINDOW_S)  # 768 timesteps per 3s block
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
# 1. Frequency band extraction (per sub-recording)
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
# 2. Sample extraction from blocks (using sample_classification.json)
# ---------------------------------------------------------------------------
def interpolate_block(ts, data_cols, target_n=TARGET_SAMPLES):
    """
    Interpolate a 3s block from irregular timestamps to a uniform grid.
    ts: array of timestamps for this block
    data_cols: dict of {col_name: array} for this block
    Returns dict of {col_name: interpolated_array} with exactly target_n points.
    """
    t_start, t_end = ts[0], ts[-1]
    uniform_t = np.linspace(t_start, t_end, target_n)
    result = {}
    for col_name, vals in data_cols.items():
        result[col_name] = np.interp(uniform_t, ts, vals)
    return result


def extract_samples_from_block(df, block, feat_names):
    """
    Extract sliding-window samples from a single class block.
    Uses the block boundaries from sample_classification.json.
    Returns list of (aggregated_features_vector,) for this block.
    """
    start_t = block["start_t"]
    end_t = block["end_t"]
    duration = block["duration_s"]
    expected_n = block["n_samples"]

    if expected_n == 0 or duration < WINDOW_S:
        return []

    ts = df["lsl_timestamp"].values
    samples = []

    # Slide 3s windows with 0.6s stride
    n_windows = int(math.floor((duration - WINDOW_S) / STRIDE_S)) + 1
    for w in range(n_windows):
        w_start = start_t + w * STRIDE_S
        w_end = w_start + WINDOW_S

        # Find rows in this window
        mask = (ts >= w_start) & (ts < w_end)
        indices = np.where(mask)[0]

        if len(indices) < 10:
            continue

        # Extract raw band data for this window
        block_ts = ts[indices]
        block_data = {fn: df[fn].values[indices] for fn in feat_names}

        # Interpolate to uniform 768 timesteps
        interp_data = interpolate_block(block_ts, block_data, TARGET_SAMPLES)

        # Aggregate: mean, std, min, max per band feature → 112 features
        agg = []
        for fn in feat_names:
            col = interp_data[fn]
            agg.extend([np.mean(col), np.std(col), np.min(col), np.max(col)])
        samples.append(agg)

    return samples


# ---------------------------------------------------------------------------
# 3. Rebalance
# ---------------------------------------------------------------------------
def rebalance_pools(skip_pool, noskip_pool, seed=SEED):
    """Undersample majority pool to match minority pool size."""
    rng = np.random.RandomState(seed)
    n_skip = len(skip_pool)
    n_noskip = len(noskip_pool)

    if n_skip == n_noskip:
        return skip_pool, noskip_pool

    if n_skip > n_noskip:
        sel = rng.choice(n_skip, size=n_noskip, replace=False)
        return [skip_pool[i] for i in sel], noskip_pool
    else:
        sel = rng.choice(n_noskip, size=n_skip, replace=False)
        return skip_pool, [noskip_pool[i] for i in sel]


# ---------------------------------------------------------------------------
# 4. Train RF + collect train & val metrics
# ---------------------------------------------------------------------------
def build_train_val(skip_pool, noskip_pool, seed=SEED):
    """
    Shuffle within each pool, then split each 60/40.
    Combine to form balanced train and val sets.
    """
    rng = np.random.RandomState(seed)

    # Shuffle within each pool (breaks temporal adjacency from 80% overlap)
    skip_idx = np.arange(len(skip_pool))
    noskip_idx = np.arange(len(noskip_pool))
    rng.shuffle(skip_idx)
    rng.shuffle(noskip_idx)
    skip_pool = [skip_pool[i] for i in skip_idx]
    noskip_pool = [noskip_pool[i] for i in noskip_idx]

    # 60/40 split per pool
    n_skip = len(skip_pool)
    n_noskip = len(noskip_pool)
    split_skip = int(round(n_skip * 0.6))
    split_noskip = int(round(n_noskip * 0.6))

    skip_train, skip_val = skip_pool[:split_skip], skip_pool[split_skip:]
    noskip_train, noskip_val = noskip_pool[:split_noskip], noskip_pool[split_noskip:]

    # Combine and label
    X_train = np.array(skip_train + noskip_train)
    y_train = np.array([1] * len(skip_train) + [0] * len(noskip_train))
    X_val = np.array(skip_val + noskip_val)
    y_val = np.array([1] * len(skip_val) + [0] * len(noskip_val))

    # Shuffle combined sets
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
# 5. Visualization
# ---------------------------------------------------------------------------
def plot_boxplots(results: dict):
    metric_names = ["accuracy", "precision", "recall", "f1"]
    labels_pretty = ["Accuracy", "Precision", "Recall", "F1-Score"]

    fig, axes = plt.subplots(1, 4, figsize=(16, 5))
    fig.suptitle(
        f"Per-Participant RF Performance (n={len(results)})  —  "
        f"V4 RF: {N_ESTIMATORS} trees, depth={MAX_DEPTH}, no notch",
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
        "P", "Samples\n(balanced)",
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
        f"Per-Participant Random Forest Results (n={len(results)})\n"
        f"[{N_ESTIMATORS} trees, depth={MAX_DEPTH}, window={WINDOW_S}s, "
        f"overlap={OVERLAP*100:.0f}%, no notch, 60/40 split]",
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


# ---------------------------------------------------------------------------
# 6. Description
# ---------------------------------------------------------------------------
def write_description(results: dict):
    pids = sorted(results.keys(), key=lambda p: int(p[1:]))
    v_accs = [results[p]["val"]["accuracy"] for p in pids]
    v_f1s = [results[p]["val"]["f1"] for p in pids]

    lines = [
        "=" * 70,
        "PER-PARTICIPANT RANDOM FOREST TRAINING — EEG TikTok Study",
        "=" * 70, "",
        "OBJECTIVE", "-" * 9,
        "Train the best model from feasibility (V4 RF Run-2) individually on",
        "each of the 25 included participants and report train + validation",
        "performance metrics.", "",
        "MODEL CONFIG", "-" * 12,
        f"  Model:            RandomForestClassifier",
        f"  n_estimators:     {N_ESTIMATORS}",
        f"  max_depth:        {MAX_DEPTH}",
        f"  min_samples_leaf: {MIN_SAMPLES_LEAF}",
        f"  class_weight:     balanced",
        f"  Window:           {WINDOW_S}s",
        f"  Overlap:          {OVERLAP*100:.0f}%",
        f"  Stride:           {STRIDE_S}s",
        f"  Interpolation:    {TARGET_SAMPLES} timesteps (256Hz × 3s)",
        f"  Notch filter:     OFF",
        f"  Train/Val split:  60/40 (per pool)",
        f"  Random seed:      {SEED}", "",
        "PIPELINE STEPS", "-" * 14,
        "  1. Load block boundaries from sample_classification.json",
        "  2. Per sub-recording: extract 7×4 frequency bands from raw EEG",
        "  3. Slide 3s windows (0.6s stride) through each block",
        "  4. Interpolate each window to 768 uniform timesteps",
        "  5. Aggregate: mean/std/min/max per band → 112 features",
        "  6. Collect into skip_pool and noskip_pool",
        "  7. Rebalance: undersample majority pool to 50/50",
        "  8. Shuffle within each pool (break overlap adjacency)",
        "  9. Split each pool 60/40 → combined train/val sets",
        "  10. Train RF, evaluate on train and val", "",
        "AGGREGATE RESULTS", "-" * 17,
        f"  Participants:     {len(results)}",
        f"  Val Accuracy:     {np.mean(v_accs):.1%} ± {np.std(v_accs):.1%}",
        f"  Val F1:           {np.mean(v_f1s):.1%} ± {np.std(v_f1s):.1%}",
        f"  Val Acc range:    {np.min(v_accs):.1%} – {np.max(v_accs):.1%}",
        f"  > 50% baseline:   {sum(1 for a in v_accs if a > 0.5)}/{len(v_accs)}", "",
        "PER-PARTICIPANT", "-" * 15,
    ]
    for pid in pids:
        r = results[pid]
        lines.append(
            f"  {pid}: val_acc={r['val']['accuracy']:.1%}  "
            f"val_f1={r['val']['f1']:.1%}  "
            f"n={r['n_total_balanced']}"
        )
    lines += [
        "", "=" * 70,
        "Generated by analysis.py (Step 6: per_participant_rf)",
        "=" * 70,
    ]
    out = SCRIPT_DIR / "description.txt"
    out.write_text("\n".join(lines))
    print(f"✅ Saved: {out}")


# ---------------------------------------------------------------------------
# 7. Process one participant (with optional debug output)
# ---------------------------------------------------------------------------
def process_participant(pid, sub_recordings, debug_dir=None):
    """
    Run the corrected pipeline for one participant.
    sub_recordings: list of {file, blocks} from sample_classification.json
    Returns results dict or None if skipped.
    """
    skip_pool = []     # list of 112-dim feature vectors
    noskip_pool = []

    agg_names = None  # set on first sub-recording

    for sr in sub_recordings:
        csv_name = sr["file"]
        blocks = sr["blocks"]
        csv_path = DATA_DIR / csv_name

        if not csv_path.exists():
            print(f"  ⚠️ File not found: {csv_name} — skipping sub-recording")
            continue

        # Load raw EEG
        df = pd.read_csv(csv_path)
        print(f"  {csv_name}: {len(df):,} rows, {len(blocks)} blocks")

        required = ["lsl_timestamp"] + EEG_CHANNELS
        missing = [c for c in required if c not in df.columns]
        if missing:
            print(f"    ❌ Missing columns: {missing} — skipping")
            continue

        # Step 2: Extract frequency bands
        df, feat_names, fs = add_frequency_bands(df)

        # Build aggregated feature names (same for all sub-recordings)
        if agg_names is None:
            agg_names = []
            for fn in feat_names:
                agg_names.extend([f"{fn}_mean", f"{fn}_std", f"{fn}_min", f"{fn}_max"])

        if debug_dir:
            out = debug_dir / f"{pid}_{csv_name.replace('.csv', '')}_bands.csv"
            cols = ["lsl_timestamp"] + feat_names
            df[cols].head(1000).to_csv(out, index=False)
            print(f"    🔍 DEBUG: {out.name} (first 1000 rows, {len(cols)} cols)")

        # Step 3–5: Extract samples from each block
        sub_skip = 0
        sub_noskip = 0
        for blk in blocks:
            samples = extract_samples_from_block(df, blk, feat_names)
            if blk["label"] == "about_to_skip":
                skip_pool.extend(samples)
                sub_skip += len(samples)
            else:
                noskip_pool.extend(samples)
                sub_noskip += len(samples)

        print(f"    Extracted: skip={sub_skip}, noskip={sub_noskip}")

    total_skip = len(skip_pool)
    total_noskip = len(noskip_pool)
    print(f"  Pools: skip={total_skip}, noskip={total_noskip}")

    if total_skip < 5:
        print(f"  ❌ Too few skip samples ({total_skip}) — SKIPPING")
        return None

    if debug_dir:
        out = debug_dir / f"{pid}_step5_pools.json"
        pool_debug = {
            "info": f"Pre-balance pools: each sample is a {len(agg_names)}-dim "
                    f"aggregated feature vector from one interpolated {TARGET_SAMPLES}-timestep window",
            "feature_names": agg_names,
            "skip_pool": {"count": total_skip, "samples": [s for s in skip_pool]},
            "noskip_pool": {"count": total_noskip, "samples": [s for s in noskip_pool]},
        }
        Path(out).write_text(json.dumps(pool_debug, indent=2))
        print(f"  🔍 DEBUG: {out.name} — skip={total_skip}, noskip={total_noskip}")

    # Step 6: Rebalance
    skip_pool, noskip_pool = rebalance_pools(skip_pool, noskip_pool, SEED)
    print(f"  Balanced: {len(skip_pool)} per class")

    if debug_dir:
        out = debug_dir / f"{pid}_step6_balanced.json"
        bal_debug = {
            "info": f"After rebalancing: {len(skip_pool)} samples per class",
            "skip_pool": {"count": len(skip_pool), "samples": [s for s in skip_pool]},
            "noskip_pool": {"count": len(noskip_pool), "samples": [s for s in noskip_pool]},
        }
        Path(out).write_text(json.dumps(bal_debug, indent=2))
        print(f"  🔍 DEBUG: {out.name} — {len(skip_pool)} per class")

    # Step 7-8: Shuffle per pool + split 60/40 per pool
    X_tr, y_tr, X_va, y_va = build_train_val(skip_pool, noskip_pool, SEED)
    print(f"  Split: train={len(y_tr)} (s={np.sum(y_tr==1)}/n={np.sum(y_tr==0)}), "
          f"val={len(y_va)} (s={np.sum(y_va==1)}/n={np.sum(y_va==0)})")

    if debug_dir:
        out = debug_dir / f"{pid}_step7_train_val.json"
        split_debug = {
            "info": "After per-pool shuffle + 60/40 split, then combined",
            "train": {
                "count": len(y_tr),
                "skip": int(np.sum(y_tr == 1)),
                "noskip": int(np.sum(y_tr == 0)),
                "samples": [{"features": X_tr[i].tolist(), "label": int(y_tr[i])}
                            for i in range(len(y_tr))],
            },
            "val": {
                "count": len(y_va),
                "skip": int(np.sum(y_va == 1)),
                "noskip": int(np.sum(y_va == 0)),
                "samples": [{"features": X_va[i].tolist(), "label": int(y_va[i])}
                            for i in range(len(y_va))],
            },
        }
        Path(out).write_text(json.dumps(split_debug, indent=2))
        print(f"  🔍 DEBUG: {out.name} — train={len(y_tr)}, val={len(y_va)}")

    # Step 9-10: Train RF + evaluate
    res = train_and_evaluate(X_tr, y_tr, X_va, y_va, SEED)
    print(f"  Train acc={res['train']['accuracy']:.1%}  "
          f"Val acc={res['val']['accuracy']:.1%}  "
          f"Val F1={res['val']['f1']:.1%}")
    return res


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Step 6: Per-Participant RF Training")
    parser.add_argument("--debug-one", type=str, default=None, metavar="PID",
                        help="Run only one participant and save intermediate CSVs "
                             "to debug_output/ (e.g. --debug-one P7)")
    args = parser.parse_args()

    debug_dir = None
    if args.debug_one:
        debug_dir = SCRIPT_DIR / "debug_output"
        debug_dir.mkdir(exist_ok=True)
        print(f"🔍 DEBUG MODE: processing only {args.debug_one}")
        print(f"   Intermediate CSVs → {debug_dir}/\n")

    print("=" * 60)
    print("STEP 6: PER-PARTICIPANT RF TRAINING (n=25)")
    print("=" * 60)
    print(f"  Config: {N_ESTIMATORS} trees, depth={MAX_DEPTH}, "
          f"window={WINDOW_S}s, overlap={OVERLAP*100:.0f}%, no notch")
    print(f"  Interpolation: each 3s block → {TARGET_SAMPLES} uniform timesteps\n")

    # Load exclusion mask
    mask = json.loads(MASK_JSON.read_text())
    included = set(mask["included"])
    print(f"  Included participants: {len(included)}")

    # Load sample classification (pre-computed block boundaries)
    sample_cls = json.loads(SAMPLE_JSON.read_text())
    participants_data = sample_cls["participants"]
    print(f"  Block data loaded from sample_classification.json\n")

    # Filter to debug target if needed
    if args.debug_one:
        target = args.debug_one.upper()
        if target not in participants_data:
            print(f"❌ {target} not found in sample_classification.json")
            sys.exit(1)
        sorted_pids = [target]
    else:
        sorted_pids = sorted(
            [p for p in participants_data if p in included],
            key=lambda p: int(p[1:])
        )

    # Process each participant
    all_results = {}
    for i, pid in enumerate(sorted_pids, 1):
        p_data = participants_data[pid]
        sub_recs = p_data["sub_recordings"]

        print(f"\n{'='*60}")
        print(f"[{i}/{len(sorted_pids)}] {pid} ({len(sub_recs)} sub-recording(s))")
        print("=" * 60)

        res = process_participant(pid, sub_recs, debug_dir=debug_dir)
        if res is not None:
            all_results[pid] = res

    if args.debug_one:
        print(f"\n🔍 DEBUG complete for {args.debug_one}")
        print(f"   Check CSVs in: {debug_dir}/")
        print(f"   Files saved:")
        for f in sorted(debug_dir.glob(f"{args.debug_one.upper()}_*.csv")):
            print(f"     {f.name} ({f.stat().st_size / 1024:.0f} KB)")
        return

    # --- Save results ---
    print(f"\n{'='*60}")
    print(f"RESULTS: {len(all_results)} participants processed")
    print("=" * 60)

    json_out = SCRIPT_DIR / "results.json"
    json_out.write_text(json.dumps({
        "config": {
            "model": "RandomForestClassifier",
            "n_estimators": N_ESTIMATORS,
            "max_depth": MAX_DEPTH,
            "min_samples_leaf": MIN_SAMPLES_LEAF,
            "window_s": WINDOW_S,
            "overlap": OVERLAP,
            "stride_s": STRIDE_S,
            "interpolation_samples": TARGET_SAMPLES,
            "notch_filter": False,
            "train_val_split": "60/40 per pool",
            "seed": SEED,
        },
        "participants": all_results,
    }, indent=2))
    print(f"✅ Saved: {json_out}")

    # --- Plots ---
    plot_boxplots(all_results)
    plot_table(all_results)
    write_description(all_results)

    print(f"\n✅ Done! Check results.json, images/, and description.txt")


if __name__ == "__main__":
    main()
