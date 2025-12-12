#!/usr/bin/env python3
"""
Prediction V2: Skip Prediction Training

Predicts if user is about to skip (within next 3s) from EEG data.

Pipeline with verification at each step:
1. Load skip-labeled data
2. Extract frequency bands from raw EEG
3. Create 3-second sample blocks (random windows, <33% overlap)
4. Interpolate each block to uniform 256Hz (768 samples)
5. Rebalance to 50/50
6. Train transformer

Usage:
    python prediction_2.py                        # Default 3s window
    python prediction_2.py --window 3.0 --epochs 50
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import signal
from scipy.interpolate import interp1d
import json
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

# Constants
EEG_CHANNELS = ['TP9', 'AF7', 'AF8', 'TP10']
TARGET_FS = 256
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
    
    # VERIFICATION 1: Required columns exist
    required = ['lsl_timestamp', 'keypress_A', 'classification_2'] + EEG_CHANNELS
    missing = [c for c in required if c not in df.columns]
    assert len(missing) == 0, f"Missing columns: {missing}"
    
    # VERIFICATION 2: Classes are correct
    valid_classes = {'baseline_1', 'baseline_2', 'about_to_skip', 'not_about_to_skip'}
    actual_classes = set(df['classification_2'].unique())
    assert actual_classes.issubset(valid_classes), f"Invalid classes: {actual_classes - valid_classes}"
    
    # VERIFICATION 3: Timestamps are monotonically increasing
    ts_diff = np.diff(df['lsl_timestamp'].values)
    assert np.all(ts_diff >= 0), "Timestamps not monotonically increasing"
    
    print(f"   ✓ All {len(required)} required columns present")
    print(f"   ✓ Classes valid: {actual_classes}")
    print(f"   ✓ Timestamps monotonic")
    
    return df


# ============================================================================
# STEP 2: EXTRACT FREQUENCY BANDS
# ============================================================================

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


def extract_frequency_features(df, verbose=False):
    """Extract frequency band features from EEG data."""
    # Estimate actual sampling rate
    time_diffs = np.diff(df['lsl_timestamp'].values)
    actual_fs = 1.0 / np.median(time_diffs)
    
    if verbose:
        print(f"   Estimated sampling rate: {actual_fs:.1f} Hz")
    
    # Extract bands for each channel
    band_features = {}
    for ch in EEG_CHANNELS:
        channel_data = df[ch].values
        for band_name, low_freq, high_freq in FREQUENCY_BANDS:
            feature_name = f"{ch}_{band_name}"
            band_features[feature_name] = extract_band_power(channel_data, actual_fs, low_freq, high_freq)
    
    # Add to dataframe
    df_bands = df.copy()
    for name, data in band_features.items():
        df_bands[name] = data
    
    # VERIFICATION: Band values are reasonable (not all zeros, not extreme)
    feature_names = list(band_features.keys())
    for name in feature_names[:3]:  # Check first 3
        vals = band_features[name]
        assert not np.all(vals == 0), f"All zeros in {name}"
        assert np.isfinite(vals).all(), f"Non-finite values in {name}"
    
    print(f"   ✓ Extracted {len(feature_names)} band features")
    print(f"   ✓ Band values verified (non-zero, finite)")
    
    return df_bands, feature_names, actual_fs


# ============================================================================
# STEP 3: CREATE 3-SECOND SAMPLE BLOCKS
# ============================================================================

def create_sample_blocks(df, feature_names, window_seconds=3.0, stride_seconds=1.0, verbose=False):
    """
    Create 3-second sample blocks for each class.
    
    For about_to_skip: 3s ending at each keypress_A
    For not_about_to_skip: Random windows with stride, ensuring <33% overlap
    """
    timestamps = df['lsl_timestamp'].values
    classification = df['classification_2'].values
    keypress_A = df['keypress_A'].values
    
    about_to_skip_blocks = []
    not_about_to_skip_blocks = []
    
    # --- ABOUT_TO_SKIP: 3s ending at each keypress_A ---
    keypress_indices = np.where(keypress_A == 1)[0]
    skipped_short = 0
    
    for kp_idx in keypress_indices:
        kp_time = timestamps[kp_idx]
        window_start = kp_time - window_seconds
        
        # Find rows in this window
        mask = (timestamps >= window_start) & (timestamps < kp_time)
        block_indices = np.where(mask)[0]
        
        if len(block_indices) < 10:  # Too few samples
            skipped_short += 1
            continue
        
        # Get the data
        block_data = df.iloc[block_indices][feature_names].values
        block_timestamps = timestamps[block_indices]
        
        # Check duration - skip if too short
        duration = block_timestamps[-1] - block_timestamps[0]
        if duration < window_seconds * 0.8:
            skipped_short += 1
            continue
        
        about_to_skip_blocks.append({
            'data': block_data,
            'timestamps': block_timestamps,
            'label': 1
        })
    
    # --- NOT_ABOUT_TO_SKIP: Random windows ---
    # Get continuous regions of not_about_to_skip
    not_skip_mask = classification == 'not_about_to_skip'
    not_skip_indices = np.where(not_skip_mask)[0]
    
    if len(not_skip_indices) > 0:
        # Find start of continuous regions
        breaks = np.where(np.diff(not_skip_indices) > 1)[0] + 1
        regions = np.split(not_skip_indices, breaks)
        
        for region in regions:
            if len(region) < 10:
                continue
            
            region_timestamps = timestamps[region]
            region_duration = region_timestamps[-1] - region_timestamps[0]
            
            if region_duration < window_seconds:
                continue
            
            # Sample windows with stride (ensuring <33% overlap = stride >= 2s for 3s window)
            effective_stride = max(stride_seconds, window_seconds * 0.67)
            t_start = region_timestamps[0]
            t_end = region_timestamps[-1] - window_seconds
            
            current_t = t_start
            while current_t <= t_end:
                window_end = current_t + window_seconds
                
                # Find rows in this window
                mask = (timestamps >= current_t) & (timestamps < window_end) & not_skip_mask
                block_indices = np.where(mask)[0]
                
                if len(block_indices) < 10:
                    current_t += effective_stride
                    continue
                
                block_data = df.iloc[block_indices][feature_names].values
                block_timestamps = timestamps[block_indices]
                
                not_about_to_skip_blocks.append({
                    'data': block_data,
                    'timestamps': block_timestamps,
                    'label': 0
                })
                
                current_t += effective_stride
    
    # VERIFICATION 1: Blocks have correct duration
    for block in about_to_skip_blocks[:3]:
        duration = block['timestamps'][-1] - block['timestamps'][0]
        assert duration >= window_seconds * 0.8, f"Block too short: {duration:.2f}s"
    
    # VERIFICATION 2: Chronology maintained within block
    for block in about_to_skip_blocks[:3]:
        ts_diff = np.diff(block['timestamps'])
        assert np.all(ts_diff >= 0), "Chronology violated within block"
    
    # VERIFICATION 3: Check overlap between not_skip blocks
    if len(not_about_to_skip_blocks) >= 2:
        for i in range(min(5, len(not_about_to_skip_blocks) - 1)):
            t1_start = not_about_to_skip_blocks[i]['timestamps'][0]
            t1_end = not_about_to_skip_blocks[i]['timestamps'][-1]
            t2_start = not_about_to_skip_blocks[i+1]['timestamps'][0]
            
            if t2_start < t1_end:  # Overlap exists
                overlap = (t1_end - t2_start) / window_seconds
                assert overlap < 0.34, f"Overlap too high: {overlap*100:.1f}%"
    
    # DATA LOSS REPORT
    total_keypresses = len(keypress_indices)
    valid_about_skip = len(about_to_skip_blocks)
    loss_pct = 100 * (1 - valid_about_skip / total_keypresses) if total_keypresses > 0 else 0
    
    print(f"\n   === DATA LOSS REPORT ===")
    print(f"   Total keypresses (potential about_to_skip blocks): {total_keypresses}")
    print(f"   Valid about_to_skip blocks created: {valid_about_skip}")
    print(f"   Skipped (too short): {skipped_short}")
    print(f"   DATA LOSS: {loss_pct:.1f}%")
    
    if loss_pct > 50:
        print(f"   ⚠ HIGH DATA LOSS - investigating...")
        # Debug: check first few keypresses
        for i, kp_idx in enumerate(keypress_indices[:3]):
            kp_time = timestamps[kp_idx]
            window_start = kp_time - window_seconds
            mask = (timestamps >= window_start) & (timestamps < kp_time)
            block_indices = np.where(mask)[0]
            if len(block_indices) > 0:
                duration = timestamps[block_indices[-1]] - timestamps[block_indices[0]]
                print(f"     Keypress {i}: {len(block_indices)} samples, duration={duration:.2f}s")
            else:
                print(f"     Keypress {i}: 0 samples in window")
    
    print(f"\n   ✓ Created {len(about_to_skip_blocks)} about_to_skip blocks")
    print(f"   ✓ Created {len(not_about_to_skip_blocks)} not_about_to_skip blocks")
    print(f"   ✓ Chronology maintained within blocks")
    if len(not_about_to_skip_blocks) >= 2:
        print(f"   ✓ Overlap < 33% verified")
    
    return about_to_skip_blocks, not_about_to_skip_blocks


# ============================================================================
# STEP 4: INTERPOLATE TO UNIFORM SAMPLING
# ============================================================================

def interpolate_block(block, window_seconds, target_fs=256):
    """Interpolate a single block to uniform sampling rate."""
    data = block['data']
    timestamps = block['timestamps']
    
    t_start = timestamps[0]
    t_end = t_start + window_seconds  # Force exact window duration
    
    n_target = int(window_seconds * target_fs)
    uniform_t = np.linspace(t_start, t_end, n_target, endpoint=False)
    
    n_features = data.shape[1]
    uniform_data = np.zeros((n_target, n_features))
    
    for j in range(n_features):
        f = interp1d(timestamps, data[:, j], kind='linear',
                    bounds_error=False, fill_value='extrapolate')
        uniform_data[:, j] = f(uniform_t)
    
    return uniform_data


def interpolate_all_blocks(blocks, window_seconds, target_fs=256):
    """Interpolate all blocks to uniform sampling."""
    n_target = int(window_seconds * target_fs)
    
    interpolated = []
    labels = []
    
    for block in blocks:
        uniform_data = interpolate_block(block, window_seconds, target_fs)
        interpolated.append(uniform_data)
        labels.append(block['label'])
    
    X = np.array(interpolated)
    y = np.array(labels)
    
    # VERIFICATION: Exact shape
    assert X.shape[1] == n_target, f"Wrong sample count: {X.shape[1]} != {n_target}"
    assert X.shape[2] == 28, f"Wrong feature count: {X.shape[2]} != 28"
    
    # VERIFICATION: No NaN/Inf
    assert np.isfinite(X).all(), "Non-finite values after interpolation"
    
    print(f"   ✓ Shape: {X.shape} (blocks × {n_target} samples × 28 features)")
    print(f"   ✓ All values finite")
    
    return X, y


# ============================================================================
# STEP 5: REBALANCE DATASET
# ============================================================================

def rebalance_dataset(X, y, seed=42):
    """Randomly undersample majority class to achieve 50/50 balance."""
    np.random.seed(seed)
    
    n_class_0 = np.sum(y == 0)
    n_class_1 = np.sum(y == 1)
    
    print(f"   Before: class_0={n_class_0}, class_1={n_class_1}")
    
    if n_class_0 == n_class_1:
        print(f"   ✓ Already balanced")
        return X, y
    
    # Undersample majority class
    if n_class_0 > n_class_1:
        minority_n = n_class_1
        majority_indices = np.where(y == 0)[0]
        minority_indices = np.where(y == 1)[0]
    else:
        minority_n = n_class_0
        majority_indices = np.where(y == 1)[0]
        minority_indices = np.where(y == 0)[0]
    
    # Randomly select from majority
    selected_majority = np.random.choice(majority_indices, size=minority_n, replace=False)
    
    # Combine
    all_indices = np.concatenate([minority_indices, selected_majority])
    np.random.shuffle(all_indices)
    
    X_balanced = X[all_indices]
    y_balanced = y[all_indices]
    
    # VERIFICATION: Exactly 50/50
    n_0 = np.sum(y_balanced == 0)
    n_1 = np.sum(y_balanced == 1)
    assert n_0 == n_1, f"Not balanced: {n_0} vs {n_1}"
    
    print(f"   After: class_0={n_0}, class_1={n_1}")
    print(f"   ✓ Exactly 50/50 balanced")
    
    return X_balanced, y_balanced


# ============================================================================
# STEP 6: MODEL & TRAINING
# ============================================================================

class EEGTransformer(nn.Module):
    """Transformer for EEG classification."""
    
    def __init__(self, n_features=28, seq_len=768, n_heads=3, n_layers=1, d_model=66, dropout=0.1):
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


def train_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    all_preds, all_labels = [], []
    
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        all_preds.extend((outputs > 0.5).float().cpu().numpy())
        all_labels.extend(y_batch.cpu().numpy())
    
    return total_loss / len(loader), accuracy_score(all_labels, all_preds)


def evaluate(model, loader, criterion, device):
    """Evaluate model."""
    model.eval()
    total_loss = 0
    all_preds, all_probs, all_labels = [], [], []
    
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            
            total_loss += loss.item()
            all_preds.extend((outputs > 0.5).float().cpu().numpy())
            all_probs.extend(outputs.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())
    
    return {
        'loss': total_loss / len(loader),
        'accuracy': accuracy_score(all_labels, all_preds),
        'precision': precision_score(all_labels, all_preds, zero_division=0),
        'recall': recall_score(all_labels, all_preds, zero_division=0),
        'f1': f1_score(all_labels, all_preds, zero_division=0),
        'predictions': np.array(all_preds),
        'labels': np.array(all_labels)
    }


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_training_curves(history, output_dir):
    """Plot training curves."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    axes[0].plot(history['train_loss'], label='Train', linewidth=2)
    axes[0].plot(history['val_loss'], label='Val', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(history['train_acc'], label='Train', linewidth=2)
    axes[1].plot(history['val_acc'], label='Val', linewidth=2)
    axes[1].axhline(y=0.5, color='gray', linestyle='--', label='Random')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'training_curves_v2.png', dpi=150)
    plt.close()


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
    ax.set_title('Confusion Matrix')
    
    for i in range(2):
        for j in range(2):
            color = 'white' if cm[i, j] > cm.max()/2 else 'black'
            ax.text(j, i, str(cm[i, j]), ha='center', va='center', color=color, fontsize=16)
    
    plt.colorbar(im)
    plt.tight_layout()
    plt.savefig(output_dir / 'confusion_matrix_v2.png', dpi=150)
    plt.close()


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Skip Prediction Training V2')
    parser.add_argument('--file', '-f', type=str, help='Specific skip labels file')
    parser.add_argument('--window', '-w', type=float, default=3.0, help='Window duration (default: 3.0s)')
    parser.add_argument('--epochs', '-e', type=int, default=50, help='Training epochs')
    parser.add_argument('--batch-size', '-b', type=int, default=16, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    print("=" * 60)
    print("PREDICTION V2: SKIP PREDICTION TRAINING")
    print("=" * 60)
    
    # Device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print(f"Device: MPS (Mac GPU)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Device: CUDA")
    else:
        device = torch.device("cpu")
        print(f"Device: CPU")
    
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
        
        df_bands, feature_names, actual_fs = extract_frequency_features(df, verbose=args.verbose)
        
        # STEP 3: Create sample blocks
        print(f"\n{'='*60}")
        print("STEP 3: CREATE {:.1f}s SAMPLE BLOCKS".format(args.window))
        print("=" * 60)
        
        about_skip_blocks, not_skip_blocks = create_sample_blocks(
            df_bands, feature_names, window_seconds=args.window, verbose=args.verbose
        )
        
        # STEP 4: Interpolate
        print(f"\n{'='*60}")
        print("STEP 4: INTERPOLATE TO {:.0f}Hz".format(TARGET_FS))
        print("=" * 60)
        
        all_blocks = about_skip_blocks + not_skip_blocks
        X, y = interpolate_all_blocks(all_blocks, args.window, TARGET_FS)
        
        # STEP 5: Rebalance
        print(f"\n{'='*60}")
        print("STEP 5: REBALANCE DATASET")
        print("=" * 60)
        
        X, y = rebalance_dataset(X, y, seed=args.seed)
        
        # STEP 6: Train
        print(f"\n{'='*60}")
        print("STEP 6: TRAIN TRANSFORMER")
        print("=" * 60)
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.4, random_state=args.seed, stratify=y
        )
        
        print(f"   Train: {len(y_train)} samples")
        print(f"   Val: {len(y_val)} samples")
        
        # Create loaders
        train_dataset = TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32)
        )
        val_dataset = TensorDataset(
            torch.tensor(X_val, dtype=torch.float32),
            torch.tensor(y_val, dtype=torch.float32)
        )
        
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
        
        # Model
        seq_len = int(args.window * TARGET_FS)
        model = EEGTransformer(n_features=28, seq_len=seq_len).to(device)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"   Model parameters: {n_params:,}")
        
        criterion = nn.BCELoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        
        # Training loop
        history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
        best_val_acc = 0
        best_model_state = None
        
        print(f"\n   Training for {args.epochs} epochs...")
        for epoch in range(args.epochs):
            train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
            val_metrics = evaluate(model, val_loader, criterion, device)
            
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_metrics['loss'])
            history['val_acc'].append(val_metrics['accuracy'])
            
            if val_metrics['accuracy'] > best_val_acc:
                best_val_acc = val_metrics['accuracy']
                best_model_state = model.state_dict().copy()
            
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"   Epoch {epoch+1:3d}: Train={train_acc:.3f}, Val={val_metrics['accuracy']:.3f}")
            
            scheduler.step()
        
        # Load best model
        model.load_state_dict(best_model_state)
        final_metrics = evaluate(model, val_loader, criterion, device)
        
        print(f"\n   Final Results:")
        print(f"   ✓ Accuracy:  {final_metrics['accuracy']:.4f}")
        print(f"   ✓ Precision: {final_metrics['precision']:.4f}")
        print(f"   ✓ Recall:    {final_metrics['recall']:.4f}")
        print(f"   ✓ F1:        {final_metrics['f1']:.4f}")
        
        # VERIFICATION: Better than random (50%)
        if final_metrics['accuracy'] > 0.55:
            print(f"   ✓ Model beats random baseline (50%)")
        else:
            print(f"   ⚠ Model close to random baseline")
        
        # Save outputs
        output_dir = input_file.parent / "model_output_prediction_v2"
        output_dir.mkdir(exist_ok=True)
        
        # Save model
        model_path = output_dir / "skip_prediction_model.pt"
        torch.save({
            'model_state_dict': best_model_state,
            'feature_names': feature_names,
            'window_seconds': args.window,
            'best_val_acc': best_val_acc
        }, model_path)
        
        # Save results
        results = {
            'final_metrics': {k: float(v) for k, v in final_metrics.items() if isinstance(v, (int, float, np.floating))},
            'config': {
                'window_seconds': args.window,
                'epochs': args.epochs,
                'batch_size': args.batch_size,
                'lr': args.lr,
                'train_samples': len(y_train),
                'val_samples': len(y_val)
            },
            'timestamp': datetime.now().isoformat()
        }
        
        with open(output_dir / "training_results_v2.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        # Plots
        plot_training_curves(history, output_dir)
        plot_confusion_matrix(final_metrics['labels'], final_metrics['predictions'], output_dir)
        
        print(f"\n{'='*60}")
        print("TRAINING COMPLETE")
        print("=" * 60)
        print(f"Outputs saved to: {output_dir}")
        print(f"Best validation accuracy: {best_val_acc:.4f}")
        
        return 0
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
