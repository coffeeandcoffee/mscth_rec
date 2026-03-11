#!/usr/bin/env python3
"""
Leave-One-Group-Out (LOGO) Cross-Validation — Step 10
=====================================================
Evaluates the V4 Random Forest (112 features) using LOGO-CV to test
clinical generalizability across the 25 independent participants. 
For each of the 25 participants, it:
  1. Trains the RF on the combined data of the *other* 24 participants.
  2. Evaluates the RF on the completely unseen 1 held-out participant.

Produces:
  - results.json              — LOGO-CV train + val metrics per held-out participant
  - images/boxplot_train_vs_val.png — publication-ready box plots
  - images/per_participant_table.png — summary table
  - description.txt           — findings summary

Run:  cd <this folder> && python analysis.py
"""

from __future__ import annotations

import argparse
import json
import math
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
# 4. Extract participant data
# ---------------------------------------------------------------------------
def extract_participant_data(pid, sub_recordings):
    """
    Extract perfectly balanced skip and noskip pools for one participant.
    """
    skip_pool = []
    noskip_pool = []

    agg_names = None

    for sr in sub_recordings:
        csv_name = sr["file"]
        blocks = sr["blocks"]
        csv_path = DATA_DIR / csv_name

        if not csv_path.exists():
            print(f"  ⚠️ File not found: {csv_name} — skipping sub-recording")
            continue

        df = pd.read_csv(csv_path)
        required = ["lsl_timestamp"] + EEG_CHANNELS
        if any(c not in df.columns for c in required):
            continue

        df, feat_names, fs = add_frequency_bands(df)

        for blk in blocks:
            samples = extract_samples_from_block(df, blk, feat_names)
            if blk["label"] == "about_to_skip":
                skip_pool.extend(samples)
            else:
                noskip_pool.extend(samples)

    if len(skip_pool) < 5:
        return None

    # Rebalance to exactly 50/50 per participant before pooling!
    # This ensures no single participant dominates the training dataset 
    # and the validation set is 50/50.
    skip_pool, noskip_pool = rebalance_pools(skip_pool, noskip_pool, SEED)
    return {"skip": skip_pool, "noskip": noskip_pool}


# ---------------------------------------------------------------------------
# 5. Train & Evaluate LOGO RF
# ---------------------------------------------------------------------------
def run_logo_evaluation(target_pid, data_dict, seed=SEED):
    """
    Train on all PIDs except target_pid. Test on target_pid.
    """
    X_tr_list, y_tr_list = [], []
    X_va_list, y_va_list = [], []

    for pid, pools in data_dict.items():
        skips = pools["skip"]
        noskips = pools["noskip"]
        
        X_curr = skips + noskips
        y_curr = [1] * len(skips) + [0] * len(noskips)

        if pid == target_pid:
            X_va_list.extend(X_curr)
            y_va_list.extend(y_curr)
        else:
            X_tr_list.extend(X_curr)
            y_tr_list.extend(y_curr)

    X_tr, y_tr = np.array(X_tr_list), np.array(y_tr_list)
    X_va, y_va = np.array(X_va_list), np.array(y_va_list)

    # Shuffle training set 
    rng = np.random.RandomState(seed)
    tr_idx = np.arange(len(y_tr))
    rng.shuffle(tr_idx)
    X_tr, y_tr = X_tr[tr_idx], y_tr[tr_idx]

    # Model definition
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
        "n_total_balanced": len(y_va) # For tables, reference the size of the test participant
    }


# ---------------------------------------------------------------------------
# 6. Visualization
# ---------------------------------------------------------------------------
def plot_boxplots(results: dict):
    metric_names = ["accuracy", "precision", "recall", "f1"]
    labels_pretty = ["Accuracy", "Precision", "Recall", "F1-Score"]

    fig, axes = plt.subplots(1, 4, figsize=(16, 5))
    fig.suptitle(
        f"Leave-One-Group-Out CV Results (n={len(results)})\n"
        f"Each point represents the Held-Out Participant (Train n=24, Val n=1)",
        fontsize=13, fontweight="bold",
    )

    for ax, mkey, mlabel in zip(axes, metric_names, labels_pretty):
        train_vals = [r["train"][mkey] for r in results.values()]
        val_vals = [r["val"][mkey] for r in results.values()]

        bp = ax.boxplot(
            [train_vals, val_vals],
            labels=["Train (n-1)", "LOGO Val (Hold-out)"],
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
        ax.set_ylim(0.2, 1.05)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    out = IMAGES_DIR / "boxplot_train_vs_val.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"✅ Saved: {out}")


def plot_table(results: dict):
    pids = sorted(results.keys(), key=lambda p: int(p[1:]))
    col_labels = [
        "Held-Out\nParticipant", "Test\nSamples",
        "Train\nAcc (n-1)", "Train\nRec (n-1)",
        "LOGO Val\nAcc", "LOGO Val\nPrec", "LOGO Val\nRec", "LOGO Val\nF1",
    ]
    rows = []
    colors = []
    for pid in pids:
        r = results[pid]
        t, v = r["train"], r["val"]
        row = [
            pid, str(r["n_val"]),
            f"{t['accuracy']:.1%}", f"{t['recall']:.1%}",
            f"{v['accuracy']:.1%}", f"{v['precision']:.1%}", f"{v['recall']:.1%}", f"{v['f1']:.1%}",
        ]
        rows.append(row)
        bg = "#d4edda" if v["accuracy"] >= 0.60 else (
            "#fff3cd" if v["accuracy"] >= 0.50 else "#f8d7da"
        )
        colors.append([bg] * len(col_labels))

    # Aggregate row
    t_accs = [results[p]["train"]["accuracy"] for p in pids]
    t_recs = [results[p]["train"]["recall"] for p in pids]
    
    v_accs = [results[p]["val"]["accuracy"] for p in pids]
    v_precs = [results[p]["val"]["precision"] for p in pids]
    v_recs = [results[p]["val"]["recall"] for p in pids]
    v_f1s = [results[p]["val"]["f1"] for p in pids]
    
    rows.append([
        "MEAN", "—",
        f"{np.mean(t_accs):.1%}", f"{np.mean(t_recs):.1%}",
        f"{np.mean(v_accs):.1%}", f"{np.mean(v_precs):.1%}", f"{np.mean(v_recs):.1%}", f"{np.mean(v_f1s):.1%}",
    ])
    colors.append(["#e8f0fe"] * len(col_labels))

    n_rows = len(rows)
    fig_h = max(6, 0.40 * n_rows + 2)
    fig, ax = plt.subplots(figsize=(12, fig_h))
    ax.axis("off")
    ax.set_title(
        f"Leave-One-Group-Out CV Results (n={len(results)})\n"
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


# ---------------------------------------------------------------------------
# 7. Description
# ---------------------------------------------------------------------------
def write_description(results: dict):
    pids = sorted(results.keys(), key=lambda p: int(p[1:]))
    v_accs = [results[p]["val"]["accuracy"] for p in pids]
    v_recs = [results[p]["val"]["recall"] for p in pids]
    v_f1s = [results[p]["val"]["f1"] for p in pids]

    lines = [
        "=" * 70,
        "LEAVE-ONE-GROUP-OUT (LOGO) CROSS VALIDATION",
        "=" * 70, "",
        "OBJECTIVE", "-" * 9,
        "To test the clinical generalizability of the neurological signature.",
        "Instead of intra-subject validation (training and testing on the same",
        "person), LOGO-CV enforces training on completely separate individuals",
        "and evaluating on a strictly unseen held-out participant. This verifies",
        "whether the learned state maps across different humans.", "",
        "MODEL CONFIG", "-" * 12,
        f"  Model:            RandomForestClassifier",
        f"  n_estimators:     {N_ESTIMATORS}",
        f"  max_depth:        {MAX_DEPTH}",
        f"  Features:         112 Aggregated Stats",
        f"  Train:            {len(results)-1} participants",
        f"  Val:              1 held-out participant",
        f"  Random seed:      {SEED}", "",
        "AGGREGATE LOGO-CV RESULTS", "-" * 25,
        f"  Target Cohort:    {len(results)} Participants",
        f"  LOGO Val Recall:  {np.mean(v_recs):.1%} ± {np.std(v_recs):.1%}",
        f"  LOGO Val Acc:     {np.mean(v_accs):.1%} ± {np.std(v_accs):.1%}",
        f"  LOGO Val F1:      {np.mean(v_f1s):.1%} ± {np.std(v_f1s):.1%}",
        f"  Beat 50% baseline:{sum(1 for a in v_accs if a > 0.5)}/{len(v_accs)}", "",
        "HELD-OUT PERFORMANCE", "-" * 20,
    ]
    for pid in pids:
        r = results[pid]
        lines.append(
            f"  {pid}: logoval_acc={r['val']['accuracy']:.1%}  "
            f"logoval_rec={r['val']['recall']:.1%}  "
            f"(train n={r['n_train']}, val n={r['n_val']})"
        )
    lines += [
        "", "=" * 70,
        "Generated by analysis.py (Step 10: Generalizability LOGO-CV RF)",
        "=" * 70,
    ]
    out = SCRIPT_DIR / "description.txt"
    out.write_text("\n".join(lines))
    print(f"✅ Saved: {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("STEP 10: LOGO-CV GENERALIZABILITY (n=25)")
    print("=" * 60)

    # 1. Load exclusion mask
    mask = json.loads(MASK_JSON.read_text())
    included = set(mask["included"])

    # 2. Load block boundaries
    sample_cls = json.loads(SAMPLE_JSON.read_text())
    participants_data = sample_cls["participants"]
    
    sorted_pids = sorted(
        [p for p in participants_data if p in included],
        key=lambda p: int(p[1:])
    )

    print("Phase 1: Extracting and balancing feature pools per participant...")
    all_data = {}
    for i, pid in enumerate(sorted_pids, 1):
        p_data = participants_data[pid]
        sub_recs = p_data["sub_recordings"]
        
        pools = extract_participant_data(pid, sub_recs)
        if pools is not None:
            all_data[pid] = pools
            print(f"  [{i}/{len(sorted_pids)}] {pid}: {len(pools['skip'])} samples per class")

    print("\nPhase 2: Executing LOGO-CV training...")
    valid_pids = sorted(list(all_data.keys()), key=lambda p: int(p[1:]))
    all_results = {}
    
    for i, target_pid in enumerate(valid_pids, 1):
        print(f"  [{i}/{len(valid_pids)}] Holding out {target_pid} for Validation...")
        res = run_logo_evaluation(target_pid, all_data, SEED)
        all_results[target_pid] = res
        print(f"      Train Acc: {res['train']['accuracy']:.1%} | Val Acc: {res['val']['accuracy']:.1%} | Val Recall: {res['val']['recall']:.1%}")

    print("\nPhase 3: Saving results and generating plots...")
    
    json_out = SCRIPT_DIR / "results.json"
    json_out.write_text(json.dumps({
        "config": {
            "model": "RandomForestClassifier",
            "evaluation": "LOGO-CV",
            "n_estimators": N_ESTIMATORS,
            "max_depth": MAX_DEPTH,
        },
        "participants": all_results,
    }, indent=2))
    print(f"✅ Saved: {json_out}")

    plot_boxplots(all_results)
    plot_table(all_results)
    write_description(all_results)

    print(f"\n✅ LOGO-CV Complete!")


if __name__ == "__main__":
    main()
