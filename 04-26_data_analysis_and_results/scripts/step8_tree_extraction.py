#!/usr/bin/env python3
"""
Step 8: Surrogate Mathematical Rule Extraction

Extracts a highly interpretable, single Decision Tree (max_depth=3) acting as a surrogate
for the best performing Random Forest models (e.g. P12).
Generates the tree visualization and mathematically formalizes the splitting rule
to serve as a proposed "Next-Gen Engagement Index" specifically for algorithmic short-form media.
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text

import utils
import step2_train_rf

def get_surrogate_rule(tree, feature_names):
    """Extracts a simple mathematical formula string from the top splits of the tree."""
    tree_ = tree.tree_
    if tree_.node_count <= 1:
        return "No rules extracted."
    
    # Root node
    feature_0 = feature_names[tree_.feature[0]]
    threshold_0 = tree_.threshold[0]
    
    # Left child of root
    left_node = tree_.children_left[0]
    if tree_.children_left[left_node] != tree_.children_right[left_node]: # Not a leaf
        feature_1_left = feature_names[tree_.feature[left_node]]
        threshold_1_left = tree_.threshold[left_node]
    else:
        feature_1_left = "Leaf"
        threshold_1_left = 0
        
    # Right child of root
    right_node = tree_.children_right[0]
    if tree_.children_left[right_node] != tree_.children_right[right_node]: # Not a leaf
        feature_1_right = feature_names[tree_.feature[right_node]]
        threshold_1_right = tree_.threshold[right_node]
    else:
        feature_1_right = "Leaf"
        threshold_1_right = 0
        
    rule = f"""
New Short-Form Engagement Index (SFEI) Logical Approximation:
IF ({feature_0} <= {threshold_0:.3f}):
    THEN IF ({feature_1_left} <= {threshold_1_left:.3f}): [Predict Class {np.argmax(tree_.value[tree_.children_left[left_node]][0])}]
    ELSE: [Predict Class {np.argmax(tree_.value[tree_.children_right[left_node]][0])}]
ELSE ({feature_0} > {threshold_0:.3f}):
    THEN IF ({feature_1_right} <= {threshold_1_right:.3f}): [Predict Class {np.argmax(tree_.value[tree_.children_left[right_node]][0])}]
    ELSE: [Predict Class {np.argmax(tree_.value[tree_.children_right[right_node]][0])}]
    """
    return rule

def main():
    project_root = Path(__file__).resolve().parent.parent.parent
    processed_dir = project_root / "04-26_data_analysis_and_results" / "processed_data"
    outputs_dir = project_root / "04-26_data_analysis_and_results" / "outputs"
    
    out_dir = outputs_dir / f"tree_explainability_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("STEP 8: DECISION TREE SURROGATE & RULE EXTRACTION")
    print("=" * 60)
    
    # We choose P12 as it had the highest accuracy (81.8%) and P7 (76.7%) in the NoNotch run.
    target_pid = "P12"
    f_path = processed_dir / f"{target_pid}_labeled.csv"
    
    if not f_path.exists():
        print(f"File {f_path} not found.")
        return 1
        
    df_raw = pd.read_csv(f_path)
    
    # Re-extract data
    baseline_powers, fs = utils.compute_100s_baseline_power(df_raw, notch_freq=None)
    df_scaled, feature_names = utils.build_normalized_features(df_raw, baseline_powers, notch_freq=None)
    X, y, agg_names = step2_train_rf.create_aggregated_samples(df_scaled, feature_names, window_seconds=3.0)
    X_bal, y_bal = step2_train_rf.rebalance_dataset(X, y)
    
    # Train shallow surrogate DT for high explainability
    # max_depth=3 to keep it readable
    dt = DecisionTreeClassifier(max_depth=3, random_state=42, class_weight='balanced')
    dt.fit(X_bal, y_bal)
    
    acc = dt.score(X_bal, y_bal)
    print(f"Surrogate Tree Training Accuracy on {target_pid}: {acc:.3f}")
    
    # 1. Plot the Tree
    plt.figure(figsize=(20, 10))
    plot_tree(dt, feature_names=agg_names, class_names=['SKIP (0)', 'STAY (1)'], filled=True, rounded=True, fontsize=10)
    plt.title(f"Surrogate Decision Tree (Max Depth 3) for {target_pid} (Acc: {acc:.3f})", fontsize=16)
    plt.tight_layout()
    plt.savefig(out_dir / f'{target_pid}_surrogate_tree.png', dpi=300)
    plt.close()
    
    # 2. Extract Mathematical Rule
    rule_str = get_surrogate_rule(dt, agg_names)
    print("\nExtracted Rule:")
    print(rule_str)
    
    # Also print the raw text export from sklearn
    text_tree = export_text(dt, feature_names=agg_names)
    print("\nFull Text Tree:")
    print(text_tree)
    
    with open(out_dir / 'extracted_mathematical_rule.txt', 'w') as f:
        f.write(f"Surrogate Accuracy: {acc:.3f}\n")
        f.write(rule_str)
        f.write("\n\nFull Sklearn Text Tree:\n")
        f.write(text_tree)
        
    print(f"\nOutputs written to: {out_dir.name}")
    return 0

if __name__ == "__main__":
    main()
