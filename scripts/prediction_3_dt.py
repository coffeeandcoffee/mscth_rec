#!/usr/bin/env python3
"""
Prediction V3: Skip Prediction with Decision Tree

Predicts if user is about to skip (within next 3s) from EEG data using Decision Tree.

Pipeline:
1. Load skip-labeled data
2. Extract frequency bands from raw EEG
3. Create 3-second sample blocks
4. Aggregate features (mean, std, min, max per band)
5. Rebalance to 50/50
6. Train Decision Tree classifier

Usage:
    python prediction_3_dt.py                        # Default 3s window
    python prediction_3_dt.py --window 3.0 --nonotch
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import signal
import json
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import joblib

import warnings
warnings.filterwarnings('ignore')

# Constants
EEG_CHANNELS = ['TP9', 'AF7', 'AF8', 'TP10']
FREQUENCY_BANDS = [
    ('delta', 1, 4),
    ('theta', 4, 8),
    ('alpha', 8, 13),
    ('beta', 13, 30),
    ('low_gamma', 30, 40),
    ('high_gamma', 40, 60),
    ('very_high', 60, 100),
]


# ============================================================================
# STEP 1: LOAD DATA
# ============================================================================

def find_latest_skip_labels(recordings_dir=None):
    """Find the most recently created skip labels CSV file."""
    if recordings_dir is None:
        recordings_dir = Path(__file__).resolve().parent.parent / "recordings"
    recordings_path = Path(recordings_dir)
    
    skip_files = list(recordings_path.rglob("*_skip_labels_*.csv"))
    if not skip_files:
        raise FileNotFoundError(f"No skip labels files found in '{recordings_dir}'")
    
    return max(skip_files, key=lambda f: f.stat().st_mtime)


def load_and_verify_data(file_path):
    """Load data and verify structure."""
    df = pd.read_csv(file_path)
    
    required = ['lsl_timestamp', 'keypress_A', 'classification_2'] + EEG_CHANNELS
    missing = [c for c in required if c not in df.columns]
    assert len(missing) == 0, f"Missing columns: {missing}"
    
    valid_classes = {'baseline_1', 'baseline_2', 'about_to_skip', 'not_about_to_skip'}
    actual_classes = set(df['classification_2'].unique())
    assert actual_classes.issubset(valid_classes), f"Invalid classes: {actual_classes - valid_classes}"
    
    print(f"   ✓ All {len(required)} required columns present")
    print(f"   ✓ Classes valid: {actual_classes}")
    
    return df


# ============================================================================
# STEP 2: EXTRACT FREQUENCY BANDS
# ============================================================================

def apply_notch_filter(data, fs, notch_freq=50.0, quality_factor=30.0):
    """Apply notch filter to remove power line interference."""
    if len(data) < 20:
        return data
    try:
        b, a = signal.iirnotch(notch_freq, quality_factor, fs)
        return signal.filtfilt(b, a, data)
    except Exception:
        return data


def extract_band_power(data, fs, low_freq, high_freq):
    """Extract power in a specific frequency band using bandpass filter."""
    nyquist = fs / 2
    low = max(low_freq / nyquist, 0.01)
    high = min(high_freq / nyquist, 0.99)
    
    if low >= high or len(data) < 20:
        return np.zeros_like(data)
    
    try:
        b, a = signal.butter(4, [low, high], btype='band')
        return signal.filtfilt(b, a, data)
    except Exception:
        return np.zeros_like(data)


def extract_frequency_features(df, notch_freq=50.0):
    """Extract frequency band features from EEG data."""
    time_diffs = np.diff(df['lsl_timestamp'].values)
    actual_fs = 1.0 / np.median(time_diffs)
    
    # Apply notch filter (if enabled)
    if notch_freq is not None:
        print(f"   Applying {notch_freq}Hz notch filter...")
        df_filtered = df.copy()
        for ch in EEG_CHANNELS:
            df_filtered[ch] = apply_notch_filter(df[ch].values, actual_fs, notch_freq)
        print(f"   ✓ Notch filter applied")
    else:
        print(f"   ⏭ Notch filter DISABLED (--nonotch flag)")
        df_filtered = df.copy()
    
    # Extract bands
    band_features = {}
    for ch in EEG_CHANNELS:
        channel_data = df_filtered[ch].values
        for band_name, low_freq, high_freq in FREQUENCY_BANDS:
            feature_name = f"{ch}_{band_name}"
            band_features[feature_name] = extract_band_power(channel_data, actual_fs, low_freq, high_freq)
    
    df_bands = df_filtered.copy()
    for name, data in band_features.items():
        df_bands[name] = data
    
    feature_names = list(band_features.keys())
    print(f"   ✓ Extracted {len(feature_names)} band features")
    
    return df_bands, feature_names, actual_fs


# ============================================================================
# STEP 3: CREATE SAMPLE BLOCKS WITH AGGREGATED FEATURES
# ============================================================================

def create_aggregated_samples(df, feature_names, window_seconds=3.0, stride_seconds=1.0):
    """
    Create samples with aggregated features (mean, std, min, max) per block.
    Returns X (n_samples, n_aggregated_features), y (n_samples,)
    """
    timestamps = df['lsl_timestamp'].values
    classification = df['classification_2'].values
    keypress_A = df['keypress_A'].values
    
    samples = []
    labels = []
    
    # --- ABOUT_TO_SKIP: 3s ending at each keypress_A ---
    keypress_indices = np.where(keypress_A == 1)[0]
    
    for kp_idx in keypress_indices:
        kp_time = timestamps[kp_idx]
        window_start = kp_time - window_seconds
        
        mask = (timestamps >= window_start) & (timestamps < kp_time)
        block_indices = np.where(mask)[0]
        
        if len(block_indices) < 10:
            continue
        
        block_data = df.iloc[block_indices][feature_names].values
        
        # Aggregate: mean, std, min, max for each feature
        agg_features = []
        for j in range(block_data.shape[1]):
            col = block_data[:, j]
            agg_features.extend([np.mean(col), np.std(col), np.min(col), np.max(col)])
        
        samples.append(agg_features)
        labels.append(1)  # about_to_skip
    
    # --- NOT_ABOUT_TO_SKIP: Random windows ---
    not_skip_mask = classification == 'not_about_to_skip'
    not_skip_indices = np.where(not_skip_mask)[0]
    
    if len(not_skip_indices) > 0:
        breaks = np.where(np.diff(not_skip_indices) > 1)[0] + 1
        regions = np.split(not_skip_indices, breaks)
        
        for region in regions:
            if len(region) < 10:
                continue
            
            region_timestamps = timestamps[region]
            region_duration = region_timestamps[-1] - region_timestamps[0]
            
            if region_duration < window_seconds:
                continue
            
            effective_stride = max(stride_seconds, window_seconds * 0.67)
            t_start = region_timestamps[0]
            t_end = region_timestamps[-1] - window_seconds
            
            current_t = t_start
            while current_t <= t_end:
                window_end = current_t + window_seconds
                
                mask = (timestamps >= current_t) & (timestamps < window_end) & not_skip_mask
                block_indices = np.where(mask)[0]
                
                if len(block_indices) < 10:
                    current_t += effective_stride
                    continue
                
                block_data = df.iloc[block_indices][feature_names].values
                
                # Aggregate features
                agg_features = []
                for j in range(block_data.shape[1]):
                    col = block_data[:, j]
                    agg_features.extend([np.mean(col), np.std(col), np.min(col), np.max(col)])
                
                samples.append(agg_features)
                labels.append(0)  # not_about_to_skip
                
                current_t += effective_stride
    
    X = np.array(samples)
    y = np.array(labels)
    
    # Create aggregated feature names
    agg_feature_names = []
    for fname in feature_names:
        agg_feature_names.extend([f"{fname}_mean", f"{fname}_std", f"{fname}_min", f"{fname}_max"])
    
    print(f"   ✓ Created {np.sum(y == 1)} about_to_skip samples")
    print(f"   ✓ Created {np.sum(y == 0)} not_about_to_skip samples")
    
    return X, y, agg_feature_names


# ============================================================================
# STEP 4: REBALANCE DATASET
# ============================================================================

def rebalance_dataset(X, y, seed=42):
    """Randomly undersample majority class to achieve 50/50 balance."""
    np.random.seed(seed)
    
    n_class_0 = np.sum(y == 0)
    n_class_1 = np.sum(y == 1)
    
    print(f"   Before: class_0={n_class_0}, class_1={n_class_1}")
    
    if n_class_0 == n_class_1:
        return X, y
    
    if n_class_0 > n_class_1:
        minority_n = n_class_1
        majority_indices = np.where(y == 0)[0]
        minority_indices = np.where(y == 1)[0]
    else:
        minority_n = n_class_0
        majority_indices = np.where(y == 1)[0]
        minority_indices = np.where(y == 0)[0]
    
    selected_majority = np.random.choice(majority_indices, size=minority_n, replace=False)
    all_indices = np.concatenate([minority_indices, selected_majority])
    np.random.shuffle(all_indices)
    
    X_balanced = X[all_indices]
    y_balanced = y[all_indices]
    
    print(f"   After: class_0={np.sum(y_balanced == 0)}, class_1={np.sum(y_balanced == 1)}")
    print(f"   ✓ Exactly 50/50 balanced")
    
    return X_balanced, y_balanced


# ============================================================================
# STEP 5: TRAIN DECISION TREE
# ============================================================================

def train_decision_tree(X_train, y_train, X_val, y_val, max_depth=10, min_samples_leaf=5):
    """Train Decision Tree classifier."""
    clf = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=42,
        class_weight='balanced'
    )
    
    clf.fit(X_train, y_train)
    
    # Evaluate
    y_pred = clf.predict(X_val)
    
    results = {
        'accuracy': accuracy_score(y_val, y_pred),
        'precision': precision_score(y_val, y_pred, zero_division=0),
        'recall': recall_score(y_val, y_pred, zero_division=0),
        'f1': f1_score(y_val, y_pred, zero_division=0),
        'predictions': y_pred,
        'labels': y_val
    }
    
    return clf, results


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_confusion_matrix(labels, preds, output_dir):
    """Plot confusion matrix."""
    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(6, 5))
    
    im = ax.imshow(cm, cmap='Blues')
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1]) 
    ax.set_xticklabels(['Not Skip', 'About to Skip'])
    ax.set_yticklabels(['Not Skip', 'About to Skip'])
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('Confusion Matrix (Decision Tree)')
    
    for i in range(2):
        for j in range(2):
            color = 'white' if cm[i, j] > cm.max()/2 else 'black'
            ax.text(j, i, str(cm[i, j]), ha='center', va='center', color=color, fontsize=16)
    
    plt.colorbar(im)
    plt.tight_layout()
    plt.savefig(output_dir / 'confusion_matrix_dt.png', dpi=150)
    plt.close()


def plot_feature_importance(clf, feature_names, output_dir):
    """Plot feature importance from Decision Tree."""
    importances = clf.feature_importances_
    
    # Sort by importance
    indices = np.argsort(importances)[::-1][:20]  # Top 20
    top_names = [feature_names[i] for i in indices]
    top_values = [importances[i] * 100 for i in indices]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    y_pos = np.arange(len(top_names))
    ax.barh(y_pos, top_values, alpha=0.8, color='steelblue')
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_names)
    ax.invert_yaxis()
    ax.set_xlabel('Feature Importance (%)')
    ax.set_title('Top 20 Features: Decision Tree Skip Prediction')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'feature_importance_dt.png', dpi=150)
    plt.close()
    
    return {name: float(val) for name, val in zip(top_names, top_values)}


def plot_tree_visualization(clf, feature_names, output_dir):
    """Plot tree structure visualization."""
    fig, ax = plt.subplots(figsize=(20, 12))
    plot_tree(clf, feature_names=feature_names, class_names=['Not Skip', 'Skip'],
              filled=True, rounded=True, ax=ax, max_depth=4, fontsize=8)
    ax.set_title('Decision Tree Structure (max_depth=4 shown)')
    plt.tight_layout()
    plt.savefig(output_dir / 'tree_visualization_dt.png', dpi=150)
    plt.close()


def plot_class_distribution(y_train, y_val, output_dir):
    """Plot class distribution in train/val sets."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    
    # Train
    train_counts = [np.sum(y_train == 0), np.sum(y_train == 1)]
    axes[0].bar(['Not Skip', 'Skip'], train_counts, color=['steelblue', 'coral'])
    axes[0].set_title(f'Training Set (n={len(y_train)})')
    axes[0].set_ylabel('Count')
    for i, v in enumerate(train_counts):
        axes[0].text(i, v + 1, str(v), ha='center')
    
    # Val
    val_counts = [np.sum(y_val == 0), np.sum(y_val == 1)]
    axes[1].bar(['Not Skip', 'Skip'], val_counts, color=['steelblue', 'coral'])
    axes[1].set_title(f'Validation Set (n={len(y_val)})')
    for i, v in enumerate(val_counts):
        axes[1].text(i, v + 1, str(v), ha='center')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'class_distribution_dt.png', dpi=150)
    plt.close()


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Skip Prediction V3: Decision Tree')
    parser.add_argument('--file', '-f', type=str, help='Specific skip labels file')
    parser.add_argument('--window', '-w', type=float, default=3.0, help='Window duration (default: 3.0s)')
    parser.add_argument('--max-depth', type=int, default=10, help='Max tree depth (default: 10)')
    parser.add_argument('--min-samples', type=int, default=5, help='Min samples per leaf (default: 5)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--nonotch', action='store_true', help='Disable 50Hz notch filter')
    
    args = parser.parse_args()
    
    np.random.seed(args.seed)
    
    print("=" * 60)
    print("PREDICTION V3: SKIP PREDICTION (DECISION TREE)")
    print("=" * 60)
    
    try:
        # STEP 1: Load data
        print(f"\n{'='*60}")
        print("STEP 1: LOAD DATA")
        print("=" * 60)
        
        if args.file:
            input_file = Path(args.file)
        else:
            input_file = find_latest_skip_labels()
        
        print(f"   File: {input_file.name}")
        df = load_and_verify_data(input_file)
        print(f"   Total rows: {len(df)}")
        
        # STEP 2: Extract frequency bands
        print(f"\n{'='*60}")
        print("STEP 2: EXTRACT FREQUENCY BANDS")
        print("=" * 60)
        
        notch_freq = None if args.nonotch else 50.0
        df_bands, feature_names, actual_fs = extract_frequency_features(df, notch_freq=notch_freq)
        
        # STEP 3: Create aggregated samples
        print(f"\n{'='*60}")
        print("STEP 3: CREATE AGGREGATED SAMPLES")
        print("=" * 60)
        
        X, y, agg_feature_names = create_aggregated_samples(
            df_bands, feature_names, window_seconds=args.window
        )
        
        print(f"   Feature vector size: {X.shape[1]} (28 bands × 4 stats)")
        
        # STEP 4: Rebalance
        print(f"\n{'='*60}")
        print("STEP 4: REBALANCE DATASET")
        print("=" * 60)
        
        X, y = rebalance_dataset(X, y, seed=args.seed)
        
        # STEP 5: Train
        print(f"\n{'='*60}")
        print("STEP 5: TRAIN DECISION TREE")
        print("=" * 60)
        
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.4, random_state=args.seed, stratify=y
        )
        
        print(f"   Train: {len(y_train)} samples")
        print(f"   Val: {len(y_val)} samples")
        print(f"   Max depth: {args.max_depth}")
        print(f"   Min samples/leaf: {args.min_samples}")
        
        clf, results = train_decision_tree(
            X_train, y_train, X_val, y_val,
            max_depth=args.max_depth,
            min_samples_leaf=args.min_samples
        )
        
        print(f"\n   Final Results:")
        print(f"   ✓ Accuracy:  {results['accuracy']:.4f}")
        print(f"   ✓ Precision: {results['precision']:.4f}")
        print(f"   ✓ Recall:    {results['recall']:.4f}")
        print(f"   ✓ F1:        {results['f1']:.4f}")
        
        if results['accuracy'] > 0.55:
            print(f"   ✓ Model beats random baseline (50%)")
        else:
            print(f"   ⚠ Model close to random baseline")
        
        # STEP 6: Save results
        print(f"\n{'='*60}")
        print("STEP 6: SAVE RESULTS")
        print("=" * 60)
        
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = input_file.parent / f"model_output_pred_v3_DT_{timestamp_str}"
        output_dir.mkdir(exist_ok=True)
        
        # Plot confusion matrix
        plot_confusion_matrix(results['labels'], results['predictions'], output_dir)
        print(f"   ✓ Confusion matrix saved")
        
        # Plot feature importance
        importance_dict = plot_feature_importance(clf, agg_feature_names, output_dir)
        print(f"   ✓ Feature importance plot saved")
        
        # Print top features
        print(f"\n   Top 10 most predictive features:")
        for i, (name, val) in enumerate(list(importance_dict.items())[:10], 1):
            print(f"   {i:2}. {name}: {val:.1f}%")
        
        # Save results JSON
        results_data = {
            'model': 'DecisionTree',
            'timestamp': datetime.now().isoformat(),
            'input_file': str(input_file),
            'window_seconds': args.window,
            'max_depth': args.max_depth,
            'min_samples_leaf': args.min_samples,
            'notch_filter': notch_freq is not None,
            'train_samples': len(y_train),
            'val_samples': len(y_val),
            'accuracy': results['accuracy'],
            'precision': results['precision'],
            'recall': results['recall'],
            'f1': results['f1'],
            'feature_importance': importance_dict
        }
        
        with open(output_dir / 'training_results_dt.json', 'w') as f:
            json.dump(results_data, f, indent=2)
        
        # Save tree structure (text)
        tree_text = export_text(clf, feature_names=agg_feature_names, max_depth=5)
        with open(output_dir / 'tree_structure.txt', 'w') as f:
            f.write(tree_text)
        print(f"   ✓ Tree structure saved")
        
        # Plot tree visualization
        plot_tree_visualization(clf, agg_feature_names, output_dir)
        print(f"   ✓ Tree visualization saved")
        
        # Plot class distribution
        plot_class_distribution(y_train, y_val, output_dir)
        print(f"   ✓ Class distribution plot saved")
        
        # Save trained model
        joblib.dump(clf, output_dir / 'decision_tree_model.pkl')
        print(f"   ✓ Trained model saved (decision_tree_model.pkl)")
        
        # Save feature names for inference
        with open(output_dir / 'feature_names.json', 'w') as f:
            json.dump(agg_feature_names, f, indent=2)
        print(f"   ✓ Feature names saved")
        
        print(f"\n{'='*60}")
        print("TRAINING COMPLETE")
        print("=" * 60)
        print(f"Outputs saved to: {output_dir}")
        
        return 0
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
