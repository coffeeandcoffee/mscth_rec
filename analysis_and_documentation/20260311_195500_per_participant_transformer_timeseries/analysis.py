#!/usr/bin/env python3
"""
Per-Participant Transformer — Full Time Series Features (Experimental)
======================================================================
Same data pipeline as Step 7 (per_participant_rf_timeseries) but uses
a Transformer model instead of Random Forest.

Input: 768 timesteps × 28 band signals (as a 768×28 sequence, NOT flattened)
Model: EEGTransformer (same architecture as prediction_2.py / V2)
       1 layer, 3 heads, d_model=66, GELU activation, learnable positional encoding

Everything else is identical to the RF time series experiment:
  - Same block extraction from sample_classification.json
  - Same frequency band extraction, interpolation, rebalancing
  - Same 60/40 train/val split per pool
  - Same seed (42)

Run:  cd <this folder> && python analysis.py
Debug: cd <this folder> && python analysis.py --debug-one P7
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
import pandas as pd
from scipy import signal as sig
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

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

# Transformer hyper-parameters (same as prediction_2.py / V2)
N_HEADS = 3
N_LAYERS = 1
D_MODEL = 66
DROPOUT = 0.1
EPOCHS = 50
BATCH_SIZE = 16
LR = 0.001

# Data pipeline parameters (identical to RF time series experiment)
WINDOW_S = 3.0
OVERLAP = 0.80
STRIDE_S = WINDOW_S * (1 - OVERLAP)  # 0.6s
TARGET_SAMPLES = int(256 * WINDOW_S)  # 768 timesteps per 3s block
SEED = 42

N_BANDS = 7
N_CHANNELS = 4
N_BAND_FEATURES = N_BANDS * N_CHANNELS  # 28

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
# 1. Frequency band extraction (same as Step 6/7)
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
# 2. Sample extraction — FULL TIME SERIES as 2D arrays (768 × 28)
# ---------------------------------------------------------------------------
def interpolate_block(ts, data_cols, target_n=TARGET_SAMPLES):
    t_start, t_end = ts[0], ts[-1]
    uniform_t = np.linspace(t_start, t_end, target_n)
    result = {}
    for col_name, vals in data_cols.items():
        result[col_name] = np.interp(uniform_t, ts, vals)
    return result


def extract_samples_from_block(df, block, feat_names):
    """
    Extract sliding-window samples from a single class block.
    Returns list of 2D arrays (768 × 28) — one per window.
    """
    start_t = block["start_t"]
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

        # Interpolate to uniform 768 timesteps
        interp_data = interpolate_block(block_ts, block_data, TARGET_SAMPLES)

        # Stack as 2D array: (768, 28) — preserving sequence structure for Transformer
        sample_2d = np.column_stack([interp_data[fn] for fn in feat_names])
        samples.append(sample_2d)

    return samples


# ---------------------------------------------------------------------------
# 3. Rebalance (same as Step 6/7)
# ---------------------------------------------------------------------------
def rebalance_pools(skip_pool, noskip_pool, seed=SEED):
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
# 4. Build train/val sets (same split logic as Step 6/7)
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# 5. Transformer model (same architecture as prediction_2.py)
# ---------------------------------------------------------------------------
class EEGTransformer(nn.Module):
    """Transformer for EEG classification (same as V2 prediction_2.py)."""

    def __init__(self, n_features=28, seq_len=768, n_heads=N_HEADS,
                 n_layers=N_LAYERS, d_model=D_MODEL, dropout=DROPOUT):
        super().__init__()
        self.input_projection = nn.Linear(n_features, d_model)
        self.pos_encoder = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.1)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=dropout, activation='gelu', batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.classifier = nn.Sequential(
            nn.Linear(d_model, 32), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(32, 1), nn.Sigmoid()
        )

    def forward(self, x):
        x = self.input_projection(x)
        x = x + self.pos_encoder[:, :x.size(1), :]
        x = self.transformer(x)
        x = x.mean(dim=1)
        return self.classifier(x).squeeze(-1)


# ---------------------------------------------------------------------------
# 6. Training + evaluation
# ---------------------------------------------------------------------------
def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def train_and_evaluate(X_tr, y_tr, X_va, y_va, seed=SEED):
    """Train Transformer and return metrics dict (matching RF output format)."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = get_device()

    # DataLoaders
    train_ds = TensorDataset(
        torch.tensor(X_tr, dtype=torch.float32),
        torch.tensor(y_tr, dtype=torch.float32),
    )
    val_ds = TensorDataset(
        torch.tensor(X_va, dtype=torch.float32),
        torch.tensor(y_va, dtype=torch.float32),
    )
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    # Model
    model = EEGTransformer(
        n_features=N_BAND_FEATURES, seq_len=TARGET_SAMPLES,
    ).to(device)

    criterion = nn.BCELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_val_acc = 0.0
    best_state = None

    for epoch in range(EPOCHS):
        # Train
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
        scheduler.step()

        # Val accuracy (check for best)
        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                out = model(xb)
                val_preds.extend((out > 0.5).float().cpu().numpy())
                val_labels.extend(yb.numpy())
        val_acc = accuracy_score(val_labels, val_preds)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # Load best model and compute final metrics
    model.load_state_dict(best_state)
    model.to(device)
    model.eval()

    def compute_metrics(loader):
        all_preds, all_labels = [], []
        with torch.no_grad():
            for xb, yb in loader:
                xb = xb.to(device)
                out = model(xb)
                all_preds.extend((out > 0.5).float().cpu().numpy())
                all_labels.extend(yb.numpy())
        return {
            "accuracy": round(accuracy_score(all_labels, all_preds), 4),
            "precision": round(precision_score(all_labels, all_preds, zero_division=0), 4),
            "recall": round(recall_score(all_labels, all_preds, zero_division=0), 4),
            "f1": round(f1_score(all_labels, all_preds, zero_division=0), 4),
        }

    train_m = compute_metrics(train_loader)
    val_m = compute_metrics(val_loader)

    return {
        "train": train_m, "val": val_m,
        "n_train": len(y_tr), "n_val": len(y_va),
        "n_total_balanced": len(y_tr) + len(y_va),
    }


# ---------------------------------------------------------------------------
# 7. Visualization
# ---------------------------------------------------------------------------
def plot_boxplots(results: dict):
    metric_names = ["accuracy", "precision", "recall", "f1"]
    labels_pretty = ["Accuracy", "Precision", "Recall", "F1-Score"]

    fig, axes = plt.subplots(1, 4, figsize=(16, 5))
    fig.suptitle(
        f"Per-Participant Transformer — Full Time Series (n={len(results)})  —  "
        f"{N_LAYERS}L, {N_HEADS}H, d={D_MODEL}, {EPOCHS} epochs",
        fontsize=13, fontweight="bold",
    )

    for ax, mkey, mlabel in zip(axes, metric_names, labels_pretty):
        train_vals = [r["train"][mkey] for r in results.values()]
        val_vals = [r["val"][mkey] for r in results.values()]

        bp = ax.boxplot(
            [train_vals, val_vals],
            labels=["Train", "Validation"],
            patch_artist=True, widths=0.5,
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
        f"Per-Participant Transformer — Full Time Series (n={len(results)})\n"
        f"[{N_LAYERS}L, {N_HEADS}H, d={D_MODEL}, {EPOCHS} epochs, "
        f"window={WINDOW_S}s, overlap={OVERLAP*100:.0f}%, no notch, 60/40 split]",
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
# 8. Description
# ---------------------------------------------------------------------------
def write_description(results: dict):
    pids = sorted(results.keys(), key=lambda p: int(p[1:]))
    v_accs = [results[p]["val"]["accuracy"] for p in pids]
    v_f1s = [results[p]["val"]["f1"] for p in pids]

    lines = [
        "=" * 70,
        "PER-PARTICIPANT TRANSFORMER — FULL TIME SERIES (Experimental)",
        "=" * 70, "",
        "OBJECTIVE", "-" * 9,
        "Experimental investigation: use the FULL time series (768×28) with",
        "a Transformer model instead of Random Forest. The Transformer can",
        "natively process temporal sequences — unlike RF which requires",
        "flattening and loses all temporal structure.", "",
        "COMPARISON", "-" * 10,
        "  Step 6:  RF + 112 aggregated features     → 65.7% ± 6.3%",
        "  Step 7:  RF + 21,504 flattened features    → 57.6% ± 8.3%",
        "  This:    Transformer + 768×28 sequence      → see below", "",
        "MODEL CONFIG", "-" * 12,
        f"  Model:            EEGTransformer",
        f"  n_layers:         {N_LAYERS}",
        f"  n_heads:          {N_HEADS}",
        f"  d_model:          {D_MODEL}",
        f"  dropout:          {DROPOUT}",
        f"  epochs:           {EPOCHS}",
        f"  batch_size:       {BATCH_SIZE}",
        f"  learning_rate:    {LR}",
        f"  optimizer:        AdamW (weight_decay=0.01)",
        f"  scheduler:        CosineAnnealingLR",
        f"  Window:           {WINDOW_S}s",
        f"  Overlap:          {OVERLAP*100:.0f}%",
        f"  Stride:           {STRIDE_S:.1f}s",
        f"  Interpolation:    {TARGET_SAMPLES} timesteps (256Hz × 3s)",
        f"  Input shape:      ({TARGET_SAMPLES}, {N_BAND_FEATURES}) per sample",
        f"  Notch filter:     OFF",
        f"  Train/Val split:  60/40 (per pool)",
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
        "Generated by analysis.py (Experimental: Transformer + full time series)",
        "=" * 70,
    ]
    out = SCRIPT_DIR / "description.txt"
    out.write_text("\n".join(lines))
    print(f"✅ Saved: {out}")


# ---------------------------------------------------------------------------
# 9. Process one participant
# ---------------------------------------------------------------------------
def process_participant(pid, sub_recordings):
    skip_pool = []
    noskip_pool = []

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
        missing = [c for c in required if c not in df.columns]
        if missing:
            print(f"    ❌ Missing columns: {missing} — skipping")
            continue

        df, feat_names, fs = add_frequency_bands(df)

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

    skip_pool, noskip_pool = rebalance_pools(skip_pool, noskip_pool, SEED)
    print(f"  Balanced: {len(skip_pool)} per class")

    X_tr, y_tr, X_va, y_va = build_train_val(skip_pool, noskip_pool, SEED)
    print(f"  Split: train={len(y_tr)}, val={len(y_va)}")
    print(f"  Input shape: {X_tr.shape}")

    res = train_and_evaluate(X_tr, y_tr, X_va, y_va, SEED)
    print(f"  Train acc={res['train']['accuracy']:.1%}  "
          f"Val acc={res['val']['accuracy']:.1%}  "
          f"Val F1={res['val']['f1']:.1%}")
    return res


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Experimental: Per-Participant Transformer with Full Time Series"
    )
    parser.add_argument("--debug-one", type=str, default=None, metavar="PID",
                        help="Run only one participant (e.g. --debug-one P7)")
    args = parser.parse_args()

    device = get_device()

    print("=" * 60)
    print("EXPERIMENTAL: PER-PARTICIPANT TRANSFORMER — FULL TIME SERIES")
    print("=" * 60)
    print(f"  Device: {device}")
    print(f"  Model:  EEGTransformer ({N_LAYERS}L, {N_HEADS}H, d={D_MODEL})")
    print(f"  Config: window={WINDOW_S}s, overlap={OVERLAP*100:.0f}%, no notch")
    print(f"  Input:  ({TARGET_SAMPLES}, {N_BAND_FEATURES}) per sample")
    print(f"  Train:  {EPOCHS} epochs, batch={BATCH_SIZE}, lr={LR}\n")

    mask = json.loads(MASK_JSON.read_text())
    included = set(mask["included"])
    print(f"  Included participants: {len(included)}")

    sample_cls = json.loads(SAMPLE_JSON.read_text())
    participants_data = sample_cls["participants"]
    print(f"  Block data loaded from sample_classification.json\n")

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

    if args.debug_one:
        print(f"\n🔍 DEBUG complete for {args.debug_one}")
        if all_results:
            r = list(all_results.values())[0]
            print(f"   Val acc: {r['val']['accuracy']:.1%}")
        return

    print(f"\n{'='*60}")
    print(f"RESULTS: {len(all_results)} participants processed")
    print("=" * 60)

    json_out = SCRIPT_DIR / "results.json"
    json_out.write_text(json.dumps({
        "config": {
            "model": "EEGTransformer",
            "feature_representation": "full_time_series_2d",
            "input_shape": f"({TARGET_SAMPLES}, {N_BAND_FEATURES})",
            "n_layers": N_LAYERS,
            "n_heads": N_HEADS,
            "d_model": D_MODEL,
            "dropout": DROPOUT,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LR,
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

    plot_boxplots(all_results)
    plot_table(all_results)
    write_description(all_results)

    print(f"\n✅ Done! Check results.json, images/, and description.txt")


if __name__ == "__main__":
    main()
