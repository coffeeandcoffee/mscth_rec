#!/usr/bin/env python3
"""
Per-Participant RF Training — Step 6
=====================================
Trains V4 Random Forest (200 trees, depth=7, no notch) individually on each
of the 25 included participants. Produces:
  - results.json              — per-participant train + val metrics
  - images/boxplot_train_vs_val.png — publication-ready box plots
  - images/per_participant_table.png — summary table
  - description.txt           — findings summary

Run:  cd <this folder> && python analysis.py
"""

from __future__ import annotations

import json
import re
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
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent.parent / "data"
MASK_JSON = SCRIPT_DIR.parent / "20260311_145900_exclusion_mask" / "exclusion_mask.json"
IMAGES_DIR = SCRIPT_DIR / "images"
IMAGES_DIR.mkdir(exist_ok=True)

# RF hyper-parameters (V4 Run-2 — best from feasibility)
N_ESTIMATORS = 200
MAX_DEPTH = 7
MIN_SAMPLES_LEAF = 5
WINDOW_S = 3.0
OVERLAP = 0.80
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
# 1. Segment classification (from post2v2)
# ---------------------------------------------------------------------------
def classify_segments(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["class"] = ""
    a_presses = df[df["keypress_A"] == 1].index.tolist()
    b_presses = df[df["keypress_B"] == 1].index.tolist()

    if len(b_presses) >= 2:
        df.loc[b_presses[0]:b_presses[1], "class"] = "baseline_1"
        if len(b_presses) >= 3:
            df.loc[b_presses[-2]:b_presses[-1], "class"] = "baseline_2"

    if len(a_presses) >= 2:
        for i in range(len(a_presses) - 1):
            s, e = a_presses[i], a_presses[i + 1]
            td = df.loc[e, "lsl_timestamp"] - df.loc[s, "lsl_timestamp"]
            df.loc[s:e, "class"] = "tiktok_over_4s_watched" if td > 4.0 else "tiktok_under_4s_watched"
    return df


def add_skip_labels(df: pd.DataFrame, window_s: float = WINDOW_S) -> pd.DataFrame:
    df = df.copy()
    ts = df["lsl_timestamp"].values
    kp = df["keypress_A"].values
    cls = df["class"].values
    c2 = np.array([
        cls[i] if cls[i] in ("baseline_1", "baseline_2") else "not_about_to_skip"
        for i in range(len(df))
    ], dtype=object)

    for i in range(len(df)):
        if kp[i] == 1:
            t0 = ts[i] - window_s
            for j in range(i - 1, -1, -1):
                if ts[j] < t0:
                    break
                if c2[j] not in ("baseline_1", "baseline_2"):
                    c2[j] = "about_to_skip"
    df["classification_2"] = c2
    return df


# ---------------------------------------------------------------------------
# 2. Frequency feature extraction (from prediction_4_rf)
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


def extract_frequency_features(df: pd.DataFrame):
    td = np.diff(df["lsl_timestamp"].values)
    fs = 1.0 / np.median(td)
    band_feats = {}
    for ch in EEG_CHANNELS:
        cd = df[ch].values
        for bname, lo, hi in FREQUENCY_BANDS:
            band_feats[f"{ch}_{bname}"] = extract_band_power(cd, fs, lo, hi)
    df_b = df.copy()
    for k, v in band_feats.items():
        df_b[k] = v
    return df_b, list(band_feats.keys()), fs


# ---------------------------------------------------------------------------
# 3. Aggregated sample creation (from prediction_4_rf)
# ---------------------------------------------------------------------------
def create_aggregated_samples(df, feat_names, window_s=WINDOW_S):
    ts = df["lsl_timestamp"].values
    cls = df["classification_2"].values
    kp = df["keypress_A"].values
    samples, labels = [], []

    # about_to_skip: 3s ending at each keypress_A
    for idx in np.where(kp == 1)[0]:
        t_end = ts[idx]
        t_start = t_end - window_s
        mask = (ts >= t_start) & (ts < t_end)
        bi = np.where(mask)[0]
        if len(bi) < 10:
            continue
        block = df.iloc[bi][feat_names].values
        agg = []
        for j in range(block.shape[1]):
            c = block[:, j]
            agg.extend([np.mean(c), np.std(c), np.min(c), np.max(c)])
        samples.append(agg)
        labels.append(1)

    # not_about_to_skip: sliding windows with 80% overlap
    ns_mask = cls == "not_about_to_skip"
    ns_idx = np.where(ns_mask)[0]
    if len(ns_idx) > 0:
        breaks = np.where(np.diff(ns_idx) > 1)[0] + 1
        for region in np.split(ns_idx, breaks):
            if len(region) < 10:
                continue
            rt = ts[region]
            if rt[-1] - rt[0] < window_s:
                continue
            stride = window_s * (1 - OVERLAP)
            ct = rt[0]
            while ct <= rt[-1] - window_s:
                m = (ts >= ct) & (ts < ct + window_s) & ns_mask
                bi = np.where(m)[0]
                if len(bi) < 10:
                    ct += stride
                    continue
                block = df.iloc[bi][feat_names].values
                agg = []
                for j in range(block.shape[1]):
                    c = block[:, j]
                    agg.extend([np.mean(c), np.std(c), np.min(c), np.max(c)])
                samples.append(agg)
                labels.append(0)
                ct += stride

    agg_names = []
    for fn in feat_names:
        agg_names.extend([f"{fn}_mean", f"{fn}_std", f"{fn}_min", f"{fn}_max"])
    return np.array(samples), np.array(labels), agg_names


# ---------------------------------------------------------------------------
# 4. Rebalance
# ---------------------------------------------------------------------------
def rebalance(X, y, seed=SEED):
    rng = np.random.RandomState(seed)
    n0, n1 = np.sum(y == 0), np.sum(y == 1)
    if n0 == n1:
        return X, y
    if n0 > n1:
        maj, minn = np.where(y == 0)[0], np.where(y == 1)[0]
    else:
        maj, minn = np.where(y == 1)[0], np.where(y == 0)[0]
    sel = rng.choice(maj, size=len(minn), replace=False)
    idx = np.concatenate([minn, sel])
    rng.shuffle(idx)
    return X[idx], y[idx]


# ---------------------------------------------------------------------------
# 5. Train RF + collect train & val metrics
# ---------------------------------------------------------------------------
def train_and_evaluate(X, y, seed=SEED):
    X_tr, X_va, y_tr, y_va = train_test_split(
        X, y, test_size=0.4, random_state=seed, stratify=y
    )
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
        "n_total_balanced": len(y),
    }


# ---------------------------------------------------------------------------
# 6. Visualization
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

        # Overlay individual data points
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
# 7. Description
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
        f"  Notch filter:     OFF",
        f"  Train/Val split:  60/40",
        f"  Random seed:      {SEED}", "",
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
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("STEP 6: PER-PARTICIPANT RF TRAINING (n=25)")
    print("=" * 60)
    print(f"  Config: {N_ESTIMATORS} trees, depth={MAX_DEPTH}, "
          f"window={WINDOW_S}s, overlap={OVERLAP*100:.0f}%, no notch\n")

    # Load exclusion mask
    mask = json.loads(MASK_JSON.read_text())
    included = set(mask["included"])
    print(f"  Included participants: {len(included)}\n")

    # Collect CSV files per participant
    files = sorted(DATA_DIR.glob("P*_*.csv"), key=lambda p: (
        int(re.search(r"P(\d+)", p.name).group(1)),
        int(re.search(r"_(\d+)\.", p.name).group(1)),
    ))
    pid_files: dict[str, list[Path]] = defaultdict(list)
    for f in files:
        m = re.match(r"(P\d+)_", f.name, re.IGNORECASE)
        if m:
            pid = m.group(1).upper()
            if pid in included:
                pid_files[pid].append(f)

    sorted_pids = sorted(pid_files.keys(), key=lambda p: int(p[1:]))

    # Process each participant
    all_results = {}
    for i, pid in enumerate(sorted_pids, 1):
        print(f"\n{'='*60}")
        print(f"[{i}/{len(sorted_pids)}] {pid} ({len(pid_files[pid])} file(s))")
        print("=" * 60)

        # Concatenate all sub-recordings
        dfs = []
        for csv_path in pid_files[pid]:
            df = pd.read_csv(csv_path)
            print(f"  Loaded {csv_path.name}: {len(df):,} rows")
            dfs.append(df)
        df = pd.concat(dfs, ignore_index=True)
        print(f"  Total: {len(df):,} rows")

        # Check required columns
        required = ["lsl_timestamp", "keypress_A", "keypress_B"] + EEG_CHANNELS
        missing = [c for c in required if c not in df.columns]
        if missing:
            print(f"  ❌ Missing columns: {missing} — SKIPPING")
            continue

        # Step 1: Classify segments
        df = classify_segments(df)

        # Step 2: Add skip labels
        df = add_skip_labels(df, WINDOW_S)
        c2 = df["classification_2"].value_counts()
        n_skip = c2.get("about_to_skip", 0)
        n_noskip = c2.get("not_about_to_skip", 0)
        print(f"  Labels: skip={n_skip:,}  noskip={n_noskip:,}")

        if n_skip < 10:
            print(f"  ❌ Too few skip samples — SKIPPING")
            continue

        # Step 3: Extract frequency features
        df_b, feat_names, fs = extract_frequency_features(df)
        print(f"  Fs={fs:.1f}Hz, {len(feat_names)} band features")

        # Step 4: Create aggregated samples
        X, y, agg_names = create_aggregated_samples(df_b, feat_names, WINDOW_S)
        print(f"  Samples: skip={np.sum(y==1)}  noskip={np.sum(y==0)}")

        if np.sum(y == 1) < 5:
            print(f"  ❌ Too few samples after aggregation — SKIPPING")
            continue

        # Step 5: Rebalance
        X, y = rebalance(X, y, SEED)
        print(f"  Balanced: {len(y)} total ({np.sum(y==1)} per class)")

        # Step 6: Train + evaluate
        res = train_and_evaluate(X, y, SEED)
        print(f"  Train acc={res['train']['accuracy']:.1%}  "
              f"Val acc={res['val']['accuracy']:.1%}  "
              f"Val F1={res['val']['f1']:.1%}")

        all_results[pid] = res

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
            "notch_filter": False,
            "train_val_split": "60/40",
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
