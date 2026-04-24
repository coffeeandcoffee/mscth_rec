#!/usr/bin/env python3
"""
Step 9: Universal SFEI Extraction

Extracts the root decision logic (Decision Stump) for all 25 participants
to find the consensus dimensions. Uses this consensus to mathematically propose
a universal Short-Form Engagement Index (SFEI).
"""

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from collections import Counter
from sklearn.tree import DecisionTreeClassifier

import utils
import step2_train_rf

def main():
    project_root = Path(__file__).resolve().parent.parent.parent
    processed_dir = project_root / "04-26_data_analysis_and_results" / "processed_data"
    outputs_dir = project_root / "04-26_data_analysis_and_results" / "outputs"
    
    out_dir = outputs_dir / f"sfei_extraction_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("STEP 9: UNIVERSAL SFEI RULE EXTRACTION")
    print("=" * 60)
    
    input_files = list(processed_dir.glob("P*_labeled.csv"))
    if not input_files:
        print("No labeled data found.")
        return 1
        
    root_features = []
    second_level_features = []
    
    for f_path in input_files:
        pid = f_path.stem.split('_')[0]
        df_raw = pd.read_csv(f_path)
        
        baseline_powers, fs = utils.compute_100s_baseline_power(df_raw, notch_freq=None)
        df_scaled, feature_names = utils.build_normalized_features(df_raw, baseline_powers, notch_freq=None)
        
        X, y, agg_names = step2_train_rf.create_aggregated_samples(df_scaled, feature_names, window_seconds=3.0)
        
        if len(np.unique(y)) < 2:
            continue
            
        X_bal, y_bal = step2_train_rf.rebalance_dataset(X, y)
        
        dt = DecisionTreeClassifier(max_depth=2, random_state=42, class_weight='balanced')
        dt.fit(X_bal, y_bal)
        
        tree_ = dt.tree_
        if tree_.node_count > 1:
            root_feature = agg_names[tree_.feature[0]]
            root_features.append(root_feature)
            
            left_node = tree_.children_left[0]
            if tree_.children_left[left_node] != tree_.children_right[left_node]:
                second_level_features.append(agg_names[tree_.feature[left_node]])
                
            right_node = tree_.children_right[0]
            if tree_.children_left[right_node] != tree_.children_right[right_node]:
                second_level_features.append(agg_names[tree_.feature[right_node]])

    root_counts = Counter(root_features)
    second_counts = Counter(second_level_features)
    
    print("\n--- Top Root Decision Dimensions (Primary Driver) ---")
    for feat, count in root_counts.most_common(5):
        print(f" - {feat}: {count} participants ({count/len(input_files)*100:.1f}%)")
        
    print("\n--- Top Second-Level Dimensions (Secondary Driver) ---")
    for feat, count in second_counts.most_common(5):
        print(f" - {feat}: {count} occurrences")
        
    # Proposing the SFEI Formula
    # We will grab the most common High-Frequency and Low-Frequency drivers
    all_common = [f[0] for f in (root_counts + second_counts).most_common()]
    
    high_freq_candidates = [f for f in all_common if 'gamma' in f or 'beta' in f or 'alpha' in f]
    low_freq_candidates = [f for f in all_common if 'theta' in f or 'delta' in f]
    
    top_hf = high_freq_candidates[0] if high_freq_candidates else "High_Freq_Feature"
    top_lf = low_freq_candidates[0] if low_freq_candidates else "Low_Freq_Feature"
    
    print(f"\nPROPOSED SFEI FORMULA:")
    print(f"SFEI = {top_hf} / {top_lf}")
    
    with open(out_dir / 'sfei_consensus.txt', 'w') as f:
        f.write("Root Features Consensus:\n")
        for feat, count in root_counts.most_common():
            f.write(f"{feat}: {count}\n")
        f.write("\nProposed Formula:\n")
        f.write(f"SFEI = {top_hf} / {top_lf}\n")
        
    print(f"\nOutputs written to {out_dir.name}")
    return 0

if __name__ == "__main__":
    main()
