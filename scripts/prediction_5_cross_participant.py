#!/usr/bin/env python3
"""
Prediction V5: Cross-Participant Generalizability Test

Tests if a model trained on one participant generalizes to others.
This is a highly experimental script for cross-participant validation.

Usage:
    python prediction_5_cross_participant.py --model <model.pkl> --test <skip_labels.csv>
    
Example:
    # Train on P1, test on P2
    python prediction_5_cross_participant.py \
        --model recordings/eeg_20251210_203221/model_output_pred_v4_RF_*/random_forest_model.pkl \
        --test recordings/eeg_20251224_164549/eeg_20251224_164549_skip_labels_3_0s.csv
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import signal
import json
from datetime import datetime
import joblib

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

# Constants (must match training script)
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


def load_and_verify_data(file_path):
    """Load data and verify structure."""
    df = pd.read_csv(file_path)
    
    required = ['lsl_timestamp', 'keypress_A', 'classification_2'] + EEG_CHANNELS
    missing = [c for c in required if c not in df.columns]
    assert len(missing) == 0, f"Missing columns: {missing}"
    
    return df


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


def extract_frequency_features(df):
    """Extract frequency band features from EEG data."""
    time_diffs = np.diff(df['lsl_timestamp'].values)
    actual_fs = 1.0 / np.median(time_diffs)
    
    # Extract bands (no notch filter, matching training)
    band_features = {}
    for ch in EEG_CHANNELS:
        channel_data = df[ch].values
        for band_name, low_freq, high_freq in FREQUENCY_BANDS:
            feature_name = f"{ch}_{band_name}"
            band_features[feature_name] = extract_band_power(channel_data, actual_fs, low_freq, high_freq)
    
    df_bands = df.copy()
    for name, data in band_features.items():
        df_bands[name] = data
    
    feature_names = list(band_features.keys())
    return df_bands, feature_names, actual_fs


def create_aggregated_samples(df, feature_names, window_seconds=3.0):
    """
    Create samples with aggregated features (mean, std, min, max) per block.
    Uses 80% overlap as in training.
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
        
        agg_features = []
        for j in range(block_data.shape[1]):
            col = block_data[:, j]
            agg_features.extend([np.mean(col), np.std(col), np.min(col), np.max(col)])
        
        samples.append(agg_features)
        labels.append(1)
    
    # --- NOT_ABOUT_TO_SKIP: Sliding windows with 80% overlap ---
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
            
            effective_stride = window_seconds * 0.2  # 80% overlap
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
                
                agg_features = []
                for j in range(block_data.shape[1]):
                    col = block_data[:, j]
                    agg_features.extend([np.mean(col), np.std(col), np.min(col), np.max(col)])
                
                samples.append(agg_features)
                labels.append(0)
                
                current_t += effective_stride
    
    X = np.array(samples)
    y = np.array(labels)
    
    return X, y


def rebalance_dataset(X, y, seed=42):
    """Randomly undersample majority class to achieve 50/50 balance."""
    np.random.seed(seed)
    
    n_class_0 = np.sum(y == 0)
    n_class_1 = np.sum(y == 1)
    
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
    
    return X[all_indices], y[all_indices]


def plot_confusion_matrix(labels, preds, output_path):
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
    ax.set_title('Cross-Participant Confusion Matrix')
    
    for i in range(2):
        for j in range(2):
            color = 'white' if cm[i, j] > cm.max()/2 else 'black'
            ax.text(j, i, str(cm[i, j]), ha='center', va='center', color=color, fontsize=16)
    
    plt.colorbar(im)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='V5: Cross-Participant Generalizability Test')
    parser.add_argument('--model', '-m', type=str, required=True, help='Path to trained model (.pkl)')
    parser.add_argument('--test', '-t', type=str, required=True, help='Path to test participant skip labels CSV')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    np.random.seed(args.seed)
    
    print("=" * 60)
    print("V5: CROSS-PARTICIPANT GENERALIZABILITY TEST")
    print("=" * 60)
    
    try:
        # Load model
        print(f"\n{'='*60}")
        print("STEP 1: LOAD MODEL")
        print("=" * 60)
        
        model_path = Path(args.model)
        clf = joblib.load(model_path)
        print(f"   Model: {model_path.name}")
        print(f"   From: {model_path.parent.name}")
        
        # Load test data
        print(f"\n{'='*60}")
        print("STEP 2: LOAD TEST DATA")
        print("=" * 60)
        
        test_file = Path(args.test)
        df = load_and_verify_data(test_file)
        print(f"   Test file: {test_file.name}")
        print(f"   Total rows: {len(df)}")
        
        # Extract features
        print(f"\n{'='*60}")
        print("STEP 3: EXTRACT FEATURES")
        print("=" * 60)
        
        df_bands, feature_names, actual_fs = extract_frequency_features(df)
        print(f"   ✓ Extracted {len(feature_names)} band features")
        
        # Create samples
        print(f"\n{'='*60}")
        print("STEP 4: CREATE TEST SAMPLES")
        print("=" * 60)
        
        X_test, y_test = create_aggregated_samples(df_bands, feature_names)
        print(f"   About_to_skip samples: {np.sum(y_test == 1)}")
        print(f"   Not_about_to_skip samples: {np.sum(y_test == 0)}")
        
        # Rebalance
        X_test, y_test = rebalance_dataset(X_test, y_test, seed=args.seed)
        print(f"   Balanced test set: {len(y_test)} samples")
        
        # Evaluate
        print(f"\n{'='*60}")
        print("STEP 5: EVALUATE ON TEST PARTICIPANT")
        print("=" * 60)
        
        y_pred = clf.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        print(f"\n   Cross-Participant Results:")
        print(f"   ✓ Accuracy:  {accuracy:.4f}")
        print(f"   ✓ Precision: {precision:.4f}")
        print(f"   ✓ Recall:    {recall:.4f}")
        print(f"   ✓ F1:        {f1:.4f}")
        
        if accuracy > 0.55:
            print(f"   ✓ Model generalizes to new participant!")
        else:
            print(f"   ⚠ Model does NOT generalize well")
        
        # Save results
        print(f"\n{'='*60}")
        print("STEP 6: SAVE RESULTS")
        print("=" * 60)
        
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = test_file.parent / f"cross_participant_test_{timestamp_str}"
        output_dir.mkdir(exist_ok=True)
        
        # Confusion matrix
        plot_confusion_matrix(y_test, y_pred, output_dir / 'confusion_matrix_cross.png')
        print(f"   ✓ Confusion matrix saved")
        
        # Results JSON
        results = {
            'experiment': 'cross_participant_generalizability',
            'timestamp': datetime.now().isoformat(),
            'model_path': str(model_path),
            'model_trained_on': model_path.parent.parent.name,
            'test_file': str(test_file),
            'test_participant': test_file.parent.name,
            'test_samples': len(y_test),
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
        
        with open(output_dir / 'cross_participant_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        print(f"   ✓ Results saved")
        
        print(f"\n{'='*60}")
        print("CROSS-PARTICIPANT TEST COMPLETE")
        print("=" * 60)
        print(f"Results: {output_dir}")
        
        return 0
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
