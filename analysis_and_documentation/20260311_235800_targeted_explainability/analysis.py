#!/usr/bin/env python3
"""
Targeted Explainability (Feature Importance) — Step 15
======================================================
Extracts and plots Gini feature importances from the 25 
participant-specific Baseline Normalized RF models.

Outputs:
- Top 20 predictive features overall
- Importance by frequency band
- Importance by electrode region
"""

import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"
ANALYSIS_DIR = BASE_DIR / "analysis_and_documentation"
PREV_STEP_DIR = ANALYSIS_DIR / "20260311_220000_baseline_normalized_rf"
OUT_DIR = Path(__file__).resolve().parent
IMAGES_DIR = OUT_DIR / "images"

# BANDS and ELECTRODES map
BANDS = ["delta", "theta", "alpha", "beta", "low_gamma", "high_gamma", "very_high"]
CHANNELS = ["TP9", "AF7", "AF8", "TP10"]
STATS = ["mean", "std", "min", "max"]

FEATURE_NAMES = []
for band_nm in BANDS:
    for stat_nm in STATS:
        for ch_nm in CHANNELS:
            FEATURE_NAMES.append(f"{ch_nm}_{band_nm}_{stat_nm}")

np.random.seed(42)

def load_data(pid: str):
    """Loads pools from previous step data to reconstruct models. We need the data to fit the trees."""
    pool_data_path = PREV_STEP_DIR / f"{pid}_pools_{PREV_STEP_DIR.name}.json.tmp"
    if not pool_data_path.exists():
        print(f"Missing file: {pool_data_path}")
        return None, None
    with open(pool_data_path, "r") as f:
        data = json.load(f)
    return np.array(data["skip_pool"]), np.array(data["noskip_pool"])

def balance_pools(skip_pool, noskip_pool, seed):
    n_min = min(len(skip_pool), len(noskip_pool))
    rng = np.random.default_rng(seed)
    idx_skip = rng.choice(len(skip_pool), n_min, replace=False)
    idx_noskip = rng.choice(len(noskip_pool), n_min, replace=False)
    
    # Shuffle within pools
    shuf_skip = rng.permutation(skip_pool[idx_skip])
    shuf_noskip = rng.permutation(noskip_pool[idx_noskip])
    return shuf_skip, shuf_noskip

def build_train_val(skip_pool, noskip_pool, seed):
    n = len(skip_pool)
    n_train = int(n * 0.60)
    
    X_tr = np.vstack([skip_pool[:n_train], noskip_pool[:n_train]])
    y_tr = np.array([1]*n_train + [0]*n_train)
    
    X_va = np.vstack([skip_pool[n_train:], noskip_pool[n_train:]])
    y_va = np.array([1]*(n - n_train) + [0]*(n - n_train))
    
    # shuffle train
    rng = np.random.default_rng(seed)
    p_tr = rng.permutation(len(X_tr))
    X_tr, y_tr = X_tr[p_tr], y_tr[p_tr]
    
    p_va = rng.permutation(len(X_va))
    X_va, y_va = X_va[p_va], y_va[p_va]
    
    return X_tr, y_tr, X_va, y_va

def get_importances_for_participant(pid: str):
    skip_pool, noskip_pool = load_data(pid)
    if skip_pool is None:
        return None
        
    shuf_s, shuf_ns = balance_pools(skip_pool, noskip_pool, 42)
    X_tr, y_tr, X_va, y_va = build_train_val(shuf_s, shuf_ns, 42)
    
    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=7,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
    clf.fit(X_tr, y_tr)
    return clf.feature_importances_


def run():
    print(f"============================================================")
    print(f"STEP 15: TARGETED EXPLAINABILITY (FEATURE IMPORTANCE)")
    print(f"============================================================")
    
    with open(PREV_STEP_DIR / "results.json", "r") as f:
        prev_res = json.load(f)["participants"]
    
    pids = sorted([k for k in prev_res.keys() if k.startswith("P")], key=lambda x: int(x[1:]))
    all_importances = []
    
    for i, pid in enumerate(pids, 1):
        print(f"Processing [{i}/{len(pids)}] {pid}...")
        imps = get_importances_for_participant(pid)
        if imps is not None:
            all_importances.append(imps)
            
    if not all_importances:
        print("ERROR: Could not load pool data from previous step. Run aborted.")
        sys.exit(1)
        
    all_importances = np.array(all_importances) # shape: (25, 112)
    mean_importances = np.mean(all_importances, axis=0)
    std_importances = np.std(all_importances, axis=0)
    
    # 1. Top 20 Features
    df_im = pd.DataFrame({
        "Feature": FEATURE_NAMES,
        "Importance": mean_importances,
        "Std": std_importances
    }).sort_values("Importance", ascending=False)
    
    top20 = df_im.head(20)
    
    plt.figure(figsize=(12, 8))
    sns.barplot(
        data=top20,
        x="Importance",
        y="Feature",
        xerr=top20["Std"],
        capsize=0.1,
        color="#7f9fbf",
        errorbar=None
    )
    plt.title("Top 20 Predictive Features (Mean Gini Importance across 25 Participants)", fontsize=14, fontweight="bold")
    plt.xlabel("Mean Importance (Standard Deviation shown as error bars)")
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / "top20_features.png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    
    # 2. Add Band and Channel metadata
    df_im["Channel"] = df_im["Feature"].apply(lambda x: x.split("_")[0])
    df_im["Band"] = df_im["Feature"].apply(lambda x: "_".join(x.split("_")[1:-1]))
    df_im["Region"] = df_im["Channel"].map({"AF7": "Frontal", "AF8": "Frontal", "TP9": "Temporal", "TP10": "Temporal"})
    
    # Importance by Band
    band_agg = df_im.groupby("Band")["Importance"].sum().reset_index()
    # Sort logically
    band_order = ["delta", "theta", "alpha", "beta", "low_gamma", "high_gamma", "very_high"]
    band_agg["Band"] = pd.Categorical(band_agg["Band"], categories=band_order, ordered=True)
    band_agg = band_agg.sort_values("Band")
    
    plt.figure(figsize=(10, 6))
    colors = ["#ccebc5" if b != "high_gamma" else "#e41a1c" for b in band_agg["Band"]]
    sns.barplot(data=band_agg, x="Band", y="Importance", palette=colors)
    plt.title("Total Feature Importance by Frequency Band", fontsize=14, fontweight="bold")
    plt.ylabel("Sum of Importances")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / "importance_by_band.png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    
    # 3. Heatmap
    heatmap_data = df_im.groupby(["Channel", "Band"])["Importance"].sum().unstack()
    heatmap_data = heatmap_data[band_order] # Reorder columns
    heatmap_data = heatmap_data.loc[["TP9", "TP10", "AF7", "AF8"]] # Reorder rows
    
    plt.figure(figsize=(10, 4))
    sns.heatmap(heatmap_data, annot=True, cmap="YlOrRd", fmt=".3f", linewidths=.5)
    plt.title("Feature Importance Heatmap (Channel x Band)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / "electrode_band_heatmap.png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    
    # Construct Results JSON
    out_dict = {
        "n_participants": len(pids),
        "top_features": top20.to_dict(orient="records"),
        "band_importance": band_agg.to_dict(orient="records"),
        "channel_band_heatmap": heatmap_data.to_dict()
    }
    
    with open(OUT_DIR / "results.json", "w") as f:
        json.dump(out_dict, f, indent=2)
        
    # Write description.txt
    lines = [
        "TARGETED EXPLAINABILITY: FEATURE IMPORTANCE",
        "===========================================",
        "Extracted Mean Gini Feature Importances across 25 Intra-Subject RF Models.",
        "",
        "TOP 10 FEATURES OVERALL:",
        "------------------------",
    ]
    for idx, row in top20.head(10).iterrows():
        lines.append(f"  {row['Feature']:_>25}: {row['Importance']:.4f} ± {row['Std']:.4f}")
        
    lines.extend([
        "",
        "IMPORTANCE BY FREQUENCY BAND:",
        "-----------------------------",
    ])
    for idx, row in band_agg.iterrows():
        lines.append(f"  {row['Band']:_>15}: {row['Importance']:.4f}")
        
    with open(OUT_DIR / "description.txt", "w") as f:
        f.write("\n".join(lines) + "\n")
        
    print(f"\n✅ Created images/top20_features.png")
    print(f"✅ Created images/importance_by_band.png")
    print(f"✅ Created images/electrode_band_heatmap.png")
    print(f"✅ Output description.txt and results.json")
    print("DONE.")


if __name__ == "__main__":
    run()
