#!/usr/bin/env python3
"""
Step 7: Explainability and Feature Importance Extraction

Demystifies the Random Forest 'black box' by aggregating the Gini Impurity
(feature_importances_) across all 25 participant models.
Groups structural relevance by Frequency Band and Brain Region to map
mathematical bounds back to established human cognitive neuroscience.
"""

import json
from pathlib import Path
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import joblib

import utils

def find_latest_run_dir(base_dir, prefix):
    dirs = [d for d in base_dir.glob(f"{prefix}*") if d.is_dir()]
    if not dirs:
        return None
    return max(dirs, key=lambda d: d.stat().st_mtime)

def recreate_feature_names():
    base_features = []
    for ch in utils.EEG_CHANNELS:
        for band_name, _, _ in utils.FREQUENCY_BANDS:
            base_features.append(f"{ch}_{band_name}")
            
    agg_names = []
    for fname in base_features:
        agg_names.extend([f"{fname}_mean", f"{fname}_std", f"{fname}_min", f"{fname}_max"])
    return agg_names

def plot_top_features(importances, feature_names, out_dir):
    # Sort
    indices = np.argsort(importances)[::-1]
    top_n = 15
    top_indices = indices[:top_n]
    
    top_names = [feature_names[i] for i in top_indices]
    top_vals = [importances[i] for i in top_indices]
    
    plt.figure(figsize=(10, 8))
    plt.title('Top 15 Structural Drivers of Sustained Engagement', fontsize=14, pad=15)
    plt.barh(range(top_n), top_vals[::-1], color='darkred', align='center')
    plt.yticks(range(top_n), top_names[::-1], fontsize=11)
    plt.xlabel('Mean Gini Importance across Participants', fontsize=12)
    plt.tight_layout()
    plt.savefig(out_dir / 'top_15_features.png', dpi=300)
    plt.close()

def plot_grouped_importances(importances, feature_names, out_dir):
    band_importance = {band[0]: 0.0 for band in utils.FREQUENCY_BANDS}
    region_importance = {'Frontal (AF7/AF8)': 0.0, 'Temporal (TP9/TP10)': 0.0}
    
    for idx, name in enumerate(feature_names):
        val = importances[idx]
        
        # Band grouping
        for band_name, _, _ in utils.FREQUENCY_BANDS:
            if f"_{band_name}_" in name:
                band_importance[band_name] += val
                break
                
        # Region grouping
        if "AF7" in name or "AF8" in name:
            region_importance['Frontal (AF7/AF8)'] += val
        elif "TP9" in name or "TP10" in name:
            region_importance['Temporal (TP9/TP10)'] += val

    # Normalize roughly so sum = 1
    total = sum(importances)
    band_importance = {k: v/total for k, v in band_importance.items()}
    region_importance = {k: v/total for k, v in region_importance.items()}
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Ax1: Bands
    bands = list(band_importance.keys())
    b_vals = list(band_importance.values())
    ax1.bar(bands, b_vals, color='steelblue')
    ax1.set_title('Importance by Frequency Band', fontsize=14)
    ax1.set_ylabel('Aggregated Global Importance')
    ax1.tick_params(axis='x', rotation=45)
    
    # Ax2: Regions
    regions = list(region_importance.keys())
    r_vals = list(region_importance.values())
    ax2.pie(r_vals, labels=regions, autopct='%1.1f%%', startangle=90, colors=['salmon', 'lightgray'])
    ax2.set_title('Importance by Brain Region', fontsize=14)
    
    plt.tight_layout()
    plt.savefig(out_dir / 'grouped_physiological_importance.png', dpi=300)
    plt.close()

def main():
    project_root = Path(__file__).resolve().parent.parent.parent
    outputs_dir = project_root / "04-26_data_analysis_and_results" / "outputs"
    
    # Target the NoNotch run (most complete neuro-data)
    run_dir = find_latest_run_dir(outputs_dir, "rf_run_nonotch_")
    if not run_dir:
        print("ERROR: No nonotch RF run found to extract biology from.")
        return 1
        
    out_dir = outputs_dir / f"explainability_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("STEP 7: EXTRACTING NEUROLOGICAL EXPLAINABILITY")
    print(f"Targeting Run: {run_dir.name}")
    print("=" * 60)
    
    model_files = list(run_dir.glob("P*_rf_model.pkl"))
    if not model_files:
        print("No .pkl files found in directory.")
        return 1
        
    feature_names = recreate_feature_names()
    all_importances = []
    
    for pkl_file in model_files:
        clf = joblib.load(pkl_file)
        if hasattr(clf, 'feature_importances_'):
            all_importances.append(clf.feature_importances_)
            
    if not all_importances:
        print("Models do not contain feature_importances_.")
        return 1
        
    mean_importances = np.mean(all_importances, axis=0)
    
    print(f"✓ Aggregated feature importances across {len(all_importances)} participants.")
    
    plot_top_features(mean_importances, feature_names, out_dir)
    print(f"✓ Generated Top 15 Global Features Graph")
    
    plot_grouped_importances(mean_importances, feature_names, out_dir)
    print(f"✓ Generated Physiological Grouping Graphs")
    
    print(f"\nOutputs written to: {out_dir.name}")

if __name__ == "__main__":
    main()
