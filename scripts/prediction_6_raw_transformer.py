#!/usr/bin/env python3
"""
Prediction V6: Skip Prediction with Raw EEG Transformer

Uses RAW EEG signals (4 channels) directly instead of frequency band features.
The Transformer learns its own feature representations from raw time-series data.

Pipeline:
1. Load skip-labeled data with raw EEG
2. Create 3-second sample blocks (raw EEG values)
3. Normalize per-block (z-score)
4. Rebalance to 50/50
5. Train Transformer end-to-end

Usage:
    python prediction_6_raw_transformer.py                        # Default 3s window
    python prediction_6_raw_transformer.py --window 3.0 --epochs 50
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import json
from datetime import datetime
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import joblib

import warnings
warnings.filterwarnings('ignore')

# Constants
EEG_CHANNELS = ['TP9', 'AF7', 'AF8', 'TP10']
TARGET_FS = 256  # Target sampling rate for interpolation


# ============================================================================
# TRANSFORMER MODEL
# ============================================================================

class PositionalEncoding(nn.Module):
    """Positional encoding for transformer."""
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class RawEEGTransformer(nn.Module):
    """
    Transformer for raw EEG time-series classification.
    Input: (batch, seq_len, n_channels=4)
    Output: (batch, 2) for binary classification
    """
    def __init__(self, n_channels=4, seq_len=768, d_model=64, nhead=4, 
                 num_layers=3, dim_feedforward=128, dropout=0.1):
        super().__init__()
        
        self.n_channels = n_channels
        self.seq_len = seq_len
        
        # Project raw EEG to d_model dimensions
        self.input_projection = nn.Linear(n_channels, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_len=seq_len)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 2)
        )
        
        # Global average pooling will be applied before classifier
    
    def forward(self, x):
        # x: (batch, seq_len, n_channels)
        
        # Project to d_model
        x = self.input_projection(x)  # (batch, seq_len, d_model)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Transformer encoding
        x = self.transformer(x)  # (batch, seq_len, d_model)
        
        # Global average pooling over time
        x = x.mean(dim=1)  # (batch, d_model)
        
        # Classification
        x = self.classifier(x)  # (batch, 2)
        
        return x


# ============================================================================
# DATA LOADING
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
# SAMPLE CREATION
# ============================================================================

def create_raw_eeg_samples(df, window_seconds=3.0):
    """
    Create samples of raw EEG data (4 channels).
    Returns X (n_samples, seq_len, n_channels), y (n_samples,)
    """
    timestamps = df['lsl_timestamp'].values
    classification = df['classification_2'].values
    keypress_A = df['keypress_A'].values
    eeg_data = df[EEG_CHANNELS].values  # (n_rows, 4)
    
    samples = []
    labels = []
    
    # Estimate sampling rate
    time_diffs = np.diff(timestamps)
    actual_fs = 1.0 / np.median(time_diffs)
    print(f"   Actual sampling rate: {actual_fs:.1f} Hz")
    
    # --- ABOUT_TO_SKIP: 3s ending at each keypress_A ---
    keypress_indices = np.where(keypress_A == 1)[0]
    
    for kp_idx in keypress_indices:
        kp_time = timestamps[kp_idx]
        window_start = kp_time - window_seconds
        
        mask = (timestamps >= window_start) & (timestamps < kp_time)
        block_indices = np.where(mask)[0]
        
        if len(block_indices) < 50:  # Need enough samples
            continue
        
        block_data = eeg_data[block_indices]  # (n_samples_in_block, 4)
        samples.append(block_data)
        labels.append(1)  # about_to_skip
    
    # --- NOT_ABOUT_TO_SKIP: Sliding windows with 80% overlap ---
    not_skip_mask = classification == 'not_about_to_skip'
    not_skip_indices = np.where(not_skip_mask)[0]
    
    if len(not_skip_indices) > 0:
        breaks = np.where(np.diff(not_skip_indices) > 1)[0] + 1
        regions = np.split(not_skip_indices, breaks)
        
        for region in regions:
            if len(region) < 50:
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
                
                if len(block_indices) < 50:
                    current_t += effective_stride
                    continue
                
                block_data = eeg_data[block_indices]
                samples.append(block_data)
                labels.append(0)  # not_about_to_skip
                
                current_t += effective_stride
    
    print(f"   ✓ Created {sum(labels)} about_to_skip samples")
    print(f"   ✓ Created {len(labels) - sum(labels)} not_about_to_skip samples")
    
    return samples, np.array(labels), actual_fs


def interpolate_blocks(samples, window_seconds, target_fs):
    """
    Interpolate all blocks to uniform length.
    Returns X (n_samples, target_len, n_channels)
    """
    target_len = int(window_seconds * target_fs)
    n_channels = samples[0].shape[1]
    
    X = np.zeros((len(samples), target_len, n_channels), dtype=np.float32)
    
    for i, block in enumerate(samples):
        current_len = block.shape[0]
        
        if current_len == target_len:
            X[i] = block
        else:
            # Interpolate each channel
            x_old = np.linspace(0, 1, current_len)
            x_new = np.linspace(0, 1, target_len)
            for ch in range(n_channels):
                X[i, :, ch] = np.interp(x_new, x_old, block[:, ch])
    
    return X


def normalize_blocks(X):
    """Z-score normalize each block independently."""
    for i in range(X.shape[0]):
        block = X[i]
        mean = block.mean()
        std = block.std() + 1e-8  # Avoid division by zero
        X[i] = (block - mean) / std
    return X


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
# TRAINING
# ============================================================================

def train_transformer(model, train_loader, val_loader, epochs, lr, device, verbose=True):
    """Train the transformer model."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    best_val_acc = 0
    best_state = None
    history = {'train_loss': [], 'val_acc': []}
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
        
        scheduler.step()
        
        # Validation
        model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                outputs = model(X_batch)
                preds = outputs.argmax(dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(y_batch.numpy())
        
        val_acc = accuracy_score(all_labels, all_preds)
        history['train_loss'].append(total_loss / len(train_loader))
        history['val_acc'].append(val_acc)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = model.state_dict().copy()
        
        if verbose and (epoch + 1) % 5 == 0:
            print(f"   Epoch {epoch+1}/{epochs}: loss={total_loss/len(train_loader):.4f}, val_acc={val_acc:.4f}")
    
    # Load best weights
    if best_state is not None:
        model.load_state_dict(best_state)
    
    return model, history, best_val_acc


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_training_history(history, output_dir):
    """Plot training curves."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    axes[0].plot(history['train_loss'])
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Loss')
    
    axes[1].plot(history['val_acc'])
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Validation Accuracy')
    axes[1].axhline(y=0.5, color='r', linestyle='--', label='Random baseline')
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / 'training_history_v6.png', dpi=150)
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
    ax.set_title('Confusion Matrix (Raw EEG Transformer)')
    
    for i in range(2):
        for j in range(2):
            color = 'white' if cm[i, j] > cm.max()/2 else 'black'
            ax.text(j, i, str(cm[i, j]), ha='center', va='center', color=color, fontsize=16)
    
    plt.colorbar(im)
    plt.tight_layout()
    plt.savefig(output_dir / 'confusion_matrix_v6.png', dpi=150)
    plt.close()


def plot_class_distribution(y_train, y_val, output_dir):
    """Plot class distribution."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    
    train_counts = [np.sum(y_train == 0), np.sum(y_train == 1)]
    axes[0].bar(['Not Skip', 'Skip'], train_counts, color=['steelblue', 'coral'])
    axes[0].set_title(f'Training Set (n={len(y_train)})')
    axes[0].set_ylabel('Count')
    
    val_counts = [np.sum(y_val == 0), np.sum(y_val == 1)]
    axes[1].bar(['Not Skip', 'Skip'], val_counts, color=['steelblue', 'coral'])
    axes[1].set_title(f'Validation Set (n={len(y_val)})')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'class_distribution_v6.png', dpi=150)
    plt.close()


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='V6: Raw EEG Transformer')
    parser.add_argument('--file', '-f', type=str, help='Specific skip labels file')
    parser.add_argument('--window', '-w', type=float, default=3.0, help='Window duration (default: 3.0s)')
    parser.add_argument('--epochs', '-e', type=int, default=50, help='Training epochs (default: 50)')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size (default: 32)')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate (default: 0.001)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    print("=" * 60)
    print("V6: RAW EEG TRANSFORMER")
    print("=" * 60)
    print(f"   Device: {device}")
    
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
        
        # STEP 2: Create raw EEG samples
        print(f"\n{'='*60}")
        print("STEP 2: CREATE RAW EEG SAMPLES")
        print("=" * 60)
        
        samples, y, actual_fs = create_raw_eeg_samples(df, window_seconds=args.window)
        
        # STEP 3: Interpolate to uniform length
        print(f"\n{'='*60}")
        print("STEP 3: INTERPOLATE & NORMALIZE")
        print("=" * 60)
        
        X = interpolate_blocks(samples, args.window, TARGET_FS)
        X = normalize_blocks(X)
        seq_len = X.shape[1]
        print(f"   ✓ Interpolated to {seq_len} samples @ {TARGET_FS}Hz")
        print(f"   ✓ Shape: ({X.shape[0]}, {X.shape[1]}, {X.shape[2]})")
        
        # STEP 4: Rebalance
        print(f"\n{'='*60}")
        print("STEP 4: REBALANCE DATASET")
        print("=" * 60)
        
        X, y = rebalance_dataset(X, y, seed=args.seed)
        
        # STEP 5: Train/Val split
        print(f"\n{'='*60}")
        print("STEP 5: TRAIN TRANSFORMER")
        print("=" * 60)
        
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.4, random_state=args.seed, stratify=y
        )
        
        print(f"   Train: {len(y_train)} samples")
        print(f"   Val: {len(y_val)} samples")
        print(f"   Epochs: {args.epochs}")
        print(f"   Batch size: {args.batch_size}")
        print(f"   Learning rate: {args.lr}")
        
        # Create dataloaders
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train), 
            torch.LongTensor(y_train)
        )
        val_dataset = TensorDataset(
            torch.FloatTensor(X_val), 
            torch.LongTensor(y_val)
        )
        
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
        
        # Create model
        model = RawEEGTransformer(
            n_channels=4,
            seq_len=seq_len,
            d_model=64,
            nhead=4,
            num_layers=3,
            dim_feedforward=128,
            dropout=0.1
        ).to(device)
        
        print(f"   Model params: {sum(p.numel() for p in model.parameters()):,}")
        
        # Train
        model, history, best_val_acc = train_transformer(
            model, train_loader, val_loader, 
            epochs=args.epochs, lr=args.lr, device=device, verbose=True
        )
        
        # Final evaluation
        model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                outputs = model(X_batch)
                preds = outputs.argmax(dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(y_batch.numpy())
        
        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, zero_division=0)
        recall = recall_score(all_labels, all_preds, zero_division=0)
        f1 = f1_score(all_labels, all_preds, zero_division=0)
        
        print(f"\n   Final Results:")
        print(f"   ✓ Accuracy:  {accuracy:.4f}")
        print(f"   ✓ Precision: {precision:.4f}")
        print(f"   ✓ Recall:    {recall:.4f}")
        print(f"   ✓ F1:        {f1:.4f}")
        
        if accuracy > 0.55:
            print(f"   ✓ Model beats random baseline (50%)")
        else:
            print(f"   ⚠ Model close to random baseline")
        
        # STEP 6: Save results
        print(f"\n{'='*60}")
        print("STEP 6: SAVE RESULTS")
        print("=" * 60)
        
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = input_file.parent / f"model_output_pred_v6_RawTF_{timestamp_str}"
        output_dir.mkdir(exist_ok=True)
        
        # Plots
        plot_training_history(history, output_dir)
        print(f"   ✓ Training history saved")
        
        plot_confusion_matrix(all_labels, all_preds, output_dir)
        print(f"   ✓ Confusion matrix saved")
        
        plot_class_distribution(y_train, y_val, output_dir)
        print(f"   ✓ Class distribution saved")
        
        # Save model
        torch.save(model.state_dict(), output_dir / 'raw_transformer_model.pt')
        print(f"   ✓ Model saved (raw_transformer_model.pt)")
        
        # Save results JSON
        results_data = {
            'model': 'RawEEGTransformer',
            'timestamp': datetime.now().isoformat(),
            'input_file': str(input_file),
            'window_seconds': args.window,
            'target_fs': TARGET_FS,
            'seq_len': seq_len,
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'lr': args.lr,
            'd_model': 64,
            'nhead': 4,
            'num_layers': 3,
            'train_samples': len(y_train),
            'val_samples': len(y_val),
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'best_val_acc': best_val_acc
        }
        
        with open(output_dir / 'training_results_v6.json', 'w') as f:
            json.dump(results_data, f, indent=2)
        print(f"   ✓ Results JSON saved")
        
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
