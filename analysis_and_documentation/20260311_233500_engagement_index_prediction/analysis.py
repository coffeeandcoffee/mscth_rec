#!/usr/bin/env python3
"""
Engagement Index Two-Class Prediction — Step 14
===============================================
Attempts standard two-class skip prediction using ONLY the traditional 
Engagement Index formula (EI = beta / (alpha + theta)).

Trains the V4 Random Forest architecture specifically on the 5 EI features
(1 per channel + 1 global mean) per 3-second sample window. This serves as 
the baseline control to justify the necessity of the complex 112-feature approach.

Produces:
  - results.json                    — per-participant train + val metrics
  - images/boxplot_train_vs_val.png — publication-ready box plots
  - images/per_participant_table.png— summary table
  - description.txt                 — findings summary
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
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

# RF hyper-parameters (Keep consistent with V4 Run-2 to isolate only the feature change)
N_ESTIMATORS = 200
MAX_DEPTH = 7
MIN_SAMPLES_LEAF = 5
WINDOW_S = 3.0
OVERLAP = 0.80
STRIDE_S = WINDOW_S * (1 - OVERLAP)  # 0.6s
TARGET_SAMPLES = int(256 * WINDOW_S)  # 768 timesteps per 3s block
SEED = 42

EEG_CHANNELS = ["TP9", "AF7", "AF8", "TP10"]
# Only the bands required for EI
FREQUENCY_BANDS = [
    ("theta", 4, 8),
    ("alpha", 8, 13),
    ("beta", 13, 30),
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
        filtered = sig.filtfilt(b, a, data)
        return filtered ** 2  # power = amplitude squared
    except Exception:
        return np.zeros_like(data)


def add_frequency_bands(df: pd.DataFrame):
    """Add theta, alpha, beta power columns to DataFrame."""
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
# 2. Sample extraction + EI calculation
# ---------------------------------------------------------------------------
def extract_ei_samples_from_block(df, block, feat_names):
    """
    Extract sliding-window samples from a single class block.
    For each window, calculates the EI for each channel locally:
        EI = mean(beta) / (mean(alpha) + mean(theta))
    Returns list of 5-dim [EI_TP9, EI_AF7, EI_AF8, EI_TP10, EI_Mean] vectors.
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

        agg_ei_vector = []
        for ch in EEG_CHANNELS:
            # Get the exact segment for this window
            b_pow = df[f"{ch}_beta"].values[indices]
            a_pow = df[f"{ch}_alpha"].values[indices]
            t_pow = df[f"{ch}_theta"].values[indices]
            
            b_mean = np.mean(b_pow)
            a_mean = np.mean(a_pow)
            t_mean = np.mean(t_pow)
            
            denom = a_mean + t_mean
            ei = (b_mean / denom) if denom > 1e-12 else 0.0
            agg_ei_vector.append(ei)
            
        # The 5th feature is the global mean across the 4 electrodes    
        ei_mean_global = np.mean(agg_ei_vector)
        agg_ei_vector.append(ei_mean_global)
        
        samples.append(agg_ei_vector)

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

    # Shuffle within each pool
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
# 5. Visualization (identically styled)
# ---------------------------------------------------------------------------
def plot_boxplots(results: dict):
    metric_names = ["accuracy", "precision", "recall", "f1"]
    labels_pretty = ["Accuracy", "Precision", "Recall", "F1-Score"]

    fig, axes = plt.subplots(1, 4, figsize=(16, 5))
    fig.suptitle(
        f"Engagement Index vs RF-112: EI-Only Model (n={len(results)})  —  "
        f"V4 RF: {N_ESTIMATORS} trees, depth={MAX_DEPTH}",
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
        ax.set_ylim(0.0, 1.05)

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
        f"Engagement Index: Two-Class Prediction (n={len(results)})\n"
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
# 6. EI vs RF-112 Comparison (Seaborn)
# ---------------------------------------------------------------------------
RF112_RESULTS = SCRIPT_DIR.parent / "20260311_220000_baseline_normalized_rf" / "results.json"


def plot_ei_vs_rf112_recall(ei_results: dict):
    """Paired per-participant val recall comparison: Conservative EI vs RF-112."""
    if not RF112_RESULTS.exists():
        print(f"⚠️  RF-112 results not found at {RF112_RESULTS}, skipping comparison plot.")
        return

    rf_data = json.loads(RF112_RESULTS.read_text())
    rf_participants = rf_data["participants"]

    # Build paired data for participants present in BOTH result sets
    rows = []
    for pid in sorted(ei_results.keys(), key=lambda p: int(p[1:])):
        if pid not in rf_participants:
            continue
        rows.append({"Participant": pid, "Model": "Conservative EI\n(5 features)",
                     "Val Recall": ei_results[pid]["val"]["recall"]})
        rows.append({"Participant": pid, "Model": "RF-112\n(112 features)",
                     "Val Recall": rf_participants[pid]["val"]["recall"]})

    if not rows:
        print("⚠️  No overlapping participants between EI and RF-112 results.")
        return

    df = pd.DataFrame(rows)
    n_paired = len(df) // 2
    ei_vals = df.loc[df["Model"].str.startswith("Conservative"), "Val Recall"].values
    rf_vals = df.loc[df["Model"].str.startswith("RF-112"), "Val Recall"].values

    # ---- Seaborn figure ----
    sns.set_theme(style="whitegrid", font_scale=1.05)
    fig, ax = plt.subplots(figsize=(7, 6.5))

    palette = {"Conservative EI\n(5 features)": "#e07b54",
               "RF-112\n(112 features)": "#4a90d9"}

    # Box + strip
    sns.boxplot(data=df, x="Model", y="Val Recall", palette=palette,
                width=0.45, linewidth=1.3, fliersize=0,
                boxprops=dict(alpha=0.35), ax=ax)
    sns.stripplot(data=df, x="Model", y="Val Recall", palette=palette,
                  size=7, alpha=0.75, jitter=0.12, ax=ax)

    # Paired lines
    pids_sorted = sorted(
        [p for p in ei_results if p in rf_participants],
        key=lambda p: int(p[1:])
    )
    for pid in pids_sorted:
        ei_r = ei_results[pid]["val"]["recall"]
        rf_r = rf_participants[pid]["val"]["recall"]
        ax.plot([0, 1], [ei_r, rf_r], color="grey", alpha=0.25, linewidth=0.8, zorder=1)

    # 50% chance line
    ax.axhline(0.50, ls="--", color="red", alpha=0.5, lw=1, label="Chance (50%)")

    # Annotate means
    ei_mean, rf_mean = np.mean(ei_vals), np.mean(rf_vals)
    ei_std, rf_std = np.std(ei_vals), np.std(rf_vals)
    ax.text(0, ei_mean + 0.025, f"µ = {ei_mean:.1%}", ha="center", fontsize=10,
            fontweight="bold", color="#b8532e")
    ax.text(1, rf_mean + 0.025, f"µ = {rf_mean:.1%}", ha="center", fontsize=10,
            fontweight="bold", color="#2e5d99")

    ax.set_ylabel("Validation Recall (skip class)", fontsize=12)
    ax.set_xlabel("")
    ax.set_ylim(0.25, 1.05)
    ax.set_title(
        f"Conservative EI vs RF-112 — Per-Participant Validation Recall\n"
        f"(n = {n_paired}, same RF architecture, same data pipeline)",
        fontsize=12, fontweight="bold", pad=12,
    )
    ax.legend(loc="lower right", fontsize=9)

    plt.tight_layout()
    out = IMAGES_DIR / "ei_vs_rf112_val_recall.png"
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
        "ENGAGEMENT INDEX TWO-CLASS PREDICTION (Intra-Subject)",
        "=" * 70, "",
        "OBJECTIVE", "-" * 9,
        "To test if the traditional Engagement Index formula (EI = beta / (alpha + theta))",
        "can reliably discriminate skip vs no-skip states when trained within the exact",
        "same Random Forest evaluation pipeline. This serves as the control to justify",
        "the advanced 112-feature pipeline.", "",
        "METHODOLOGY", "-" * 11,
        "Instead of 112 extracted features, the algorithm extracts ONLY 5 features",
        "per 3-second sample window:",
        "  1-4. EI values for each of the 4 electrodes (TP9, AF7, AF8, TP10).",
        "  5. Global average EI across the 4 electrodes.", "",
        "MODEL CONFIG", "-" * 12,
        f"  Model:            RandomForestClassifier",
        f"  Features:         5 (EI only)",
        f"  n_estimators:     {N_ESTIMATORS}",
        f"  max_depth:        {MAX_DEPTH}",
        f"  min_samples_leaf: {MIN_SAMPLES_LEAF}",
        f"  class_weight:     balanced",
        f"  Window:           {WINDOW_S}s",
        f"  Overlap:          {OVERLAP*100:.0f}%",
        f"  Notch filter:     OFF",
        f"  Train/Val split:  60/40 (per pool)",
        f"  Random seed:      {SEED}", "",
        "AGGREGATE RESULTS", "-" * 17,
        f"  Participants:     {len(results)}",
        f"  Val Accuracy:     {np.mean(v_accs):.1%} ± {np.std(v_accs):.1%}",
        f"  Val Recall:       {np.mean(v_recs):.1%} ± {np.std(v_recs):.1%}",
        f"  Val F1:           {np.mean(v_f1s):.1%} ± {np.std(v_f1s):.1%}",
        f"  Beat 50% baseline:{sum(1 for a in v_accs if a > 0.50)}/{len(v_accs)}", "",
        "PER-PARTICIPANT", "-" * 15,
    ]
    for pid in pids:
        r = results[pid]
        lines.append(
            f"  {pid}: val_acc={r['val']['accuracy']:.1%}  "
            f"val_rec={r['val']['recall']:.1%}  "
            f"val_f1={r['val']['f1']:.1%}  "
            f"n={r['n_total_balanced']}"
        )
    lines += [
        "", "=" * 70,
        "Generated by analysis.py (Step 14: engagement_index_prediction)",
        "=" * 70,
    ]
    out = SCRIPT_DIR / "description.txt"
    out.write_text("\n".join(lines))
    print(f"✅ Saved: {out}")


# ---------------------------------------------------------------------------
# 7. Process one participant
# ---------------------------------------------------------------------------
def process_participant(pid, sub_recordings):
    skip_pool = []     
    noskip_pool = []

    for sr in sub_recordings:
        csv_name = sr["file"]
        blocks = sr["blocks"]
        csv_path = DATA_DIR / csv_name

        if not csv_path.exists():
            continue

        df = pd.read_csv(csv_path)
        required = ["lsl_timestamp"] + EEG_CHANNELS
        missing = [c for c in required if c not in df.columns]
        if missing:
            continue

        # Step 1: Extract alpha, beta, theta powers globally
        df, feat_names, fs = add_frequency_bands(df)

        # Step 2: Extract EI samples locally from blocks
        for blk in blocks:
            samples = extract_ei_samples_from_block(df, blk, feat_names)
            if blk["label"] == "about_to_skip":
                skip_pool.extend(samples)
            else:
                noskip_pool.extend(samples)

    total_skip = len(skip_pool)
    total_noskip = len(noskip_pool)

    if total_skip < 5:
        return None

    # Step 3: Rebalance
    skip_pool, noskip_pool = rebalance_pools(skip_pool, noskip_pool, SEED)

    # Step 4: Split 60/40 and Train
    X_tr, y_tr, X_va, y_va = build_train_val(skip_pool, noskip_pool, SEED)
    res = train_and_evaluate(X_tr, y_tr, X_va, y_va, SEED)
    
    print(f"  EI Features! Train acc={res['train']['accuracy']:.1%}  "
          f"Val acc={res['val']['accuracy']:.1%}  "
          f"Val Rec={res['val']['recall']:.1%}  "
          f"Val F1={res['val']['f1']:.1%}")
    return res


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("STEP 14: ENGAGEMENT INDEX TWO-CLASS PREDICTION (n=25)")
    print("=" * 60)

    # Load exclusion mask
    mask = json.loads(MASK_JSON.read_text())
    included = set(mask["included"])

    # Load sample classification (pre-computed block boundaries)
    sample_cls = json.loads(SAMPLE_JSON.read_text())
    participants_data = sample_cls["participants"]
    
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

        res = process_participant(pid, sub_recs)
        if res is not None:
            all_results[pid] = res

    # --- Save results ---
    print(f"\n{'='*60}")
    print(f"RESULTS: {len(all_results)} participants processed")
    print("=" * 60)

    json_out = SCRIPT_DIR / "results.json"
    json_out.write_text(json.dumps({
        "config": {
            "model": "RandomForestClassifier",
            "features": "Engagement Index ONLY (beta/(alpha+theta))",
            "n_estimators": N_ESTIMATORS,
            "max_depth": MAX_DEPTH,
            "min_samples_leaf": MIN_SAMPLES_LEAF,
            "window_s": WINDOW_S,
            "overlap": OVERLAP,
            "stride_s": STRIDE_S,
            "train_val_split": "60/40 per pool",
            "seed": SEED,
        },
        "participants": all_results,
    }, indent=2))
    print(f"✅ Saved: {json_out}")

    # --- Plots ---
    plot_boxplots(all_results)
    plot_table(all_results)
    plot_ei_vs_rf112_recall(all_results)
    write_description(all_results)

    print(f"\n✅ Done! Check results.json, images/, and description.txt")


if __name__ == "__main__":
    main()
