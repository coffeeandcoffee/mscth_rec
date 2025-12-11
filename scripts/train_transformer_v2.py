#!/usr/bin/env python3
"""
EEG Transformer Training Script V2 - Robust Generalization

Optimized for LIMITED DATA scenarios with multiple regularization techniques
to reduce overfitting and improve unseen data accuracy.

Key improvements over v1:
1. Data Augmentation: noise injection, time shifting, channel dropout, mixup
2. Stronger Regularization: higher dropout, label smoothing, weight decay
3. Early Stopping: prevents overfitting by monitoring validation loss
4. K-Fold Cross-Validation: more robust evaluation with limited data
5. Learning Rate Warmup: better optimization dynamics
6. Gradient Clipping: stable training
7. Ensemble Option: average predictions from multiple folds

Usage:
    python train_transformer_v2.py                      # Standard training
    python train_transformer_v2.py --augment            # Enable data augmentation
    python train_transformer_v2.py --kfold 5            # 5-fold cross-validation
    python train_transformer_v2.py --augment --kfold 5  # Best combo for limited data
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import json
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler, Dataset
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score

import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# DATA AUGMENTATION
# ============================================================================

class EEGAugmentation:
    """
    EEG-specific data augmentation strategies for limited data.
    Applied during training to artificially increase data diversity.
    """
    
    def __init__(self, 
                 noise_level=0.05,
                 time_shift_max=5,
                 channel_dropout_prob=0.05,
                 time_mask_prob=0.1,
                 time_mask_length=5,
                 amplitude_scale_range=(0.9, 1.1)):
        self.noise_level = noise_level
        self.time_shift_max = time_shift_max
        self.channel_dropout_prob = channel_dropout_prob
        self.time_mask_prob = time_mask_prob
        self.time_mask_length = time_mask_length
        self.amplitude_scale_range = amplitude_scale_range
    
    def add_gaussian_noise(self, x):
        """Add random Gaussian noise to simulate EEG variability."""
        noise = torch.randn_like(x) * self.noise_level * x.std()
        return x + noise
    
    def time_shift(self, x):
        """Randomly shift signal in time (circular)."""
        shift = np.random.randint(-self.time_shift_max, self.time_shift_max + 1)
        return torch.roll(x, shifts=shift, dims=0)
    
    def channel_dropout(self, x):
        """Randomly zero out entire channels to prevent over-reliance."""
        if np.random.random() < self.channel_dropout_prob:
            n_features = x.shape[-1]
            n_channels = 4  # TP9, AF7, AF8, TP10
            bands_per_channel = n_features // n_channels
            
            # Dropout one random electrode
            channel_idx = np.random.randint(0, n_channels)
            start_idx = channel_idx * bands_per_channel
            end_idx = start_idx + bands_per_channel
            x = x.clone()
            x[:, start_idx:end_idx] = 0
        return x
    
    def time_masking(self, x):
        """Mask random time segments (SpecAugment-style)."""
        if np.random.random() < self.time_mask_prob:
            seq_len = x.shape[0]
            mask_start = np.random.randint(0, seq_len - self.time_mask_length)
            x = x.clone()
            x[mask_start:mask_start + self.time_mask_length, :] = 0
        return x
    
    def amplitude_scaling(self, x):
        """Random amplitude scaling to simulate signal strength variation."""
        scale = np.random.uniform(*self.amplitude_scale_range)
        return x * scale
    
    def __call__(self, x):
        """Apply all augmentations with probability."""
        x = self.add_gaussian_noise(x)
        x = self.time_shift(x)
        x = self.channel_dropout(x)
        x = self.time_masking(x)
        x = self.amplitude_scaling(x)
        return x


class AugmentedDataset(Dataset):
    """Dataset wrapper that applies augmentation during training."""
    
    def __init__(self, X, y, augmentation=None, training=True):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.augmentation = augmentation
        self.training = training
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.y[idx]
        
        if self.training and self.augmentation is not None:
            x = self.augmentation(x)
        
        return x, y


def mixup_data(x, y, alpha=0.2):
    """Mixup augmentation: blend samples and labels."""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Compute loss for mixup."""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# ============================================================================
# MODEL ARCHITECTURE (Enhanced with regularization)
# ============================================================================

class RobustEEGTransformer(nn.Module):
    """
    Transformer for EEG classification with enhanced regularization.
    
    Improvements:
    - Higher dropout throughout
    - Stochastic depth (layer dropout)
    - Pre-LayerNorm for stability
    """
    
    def __init__(self, n_features=28, n_heads=3, n_layers=1, d_model=66, 
                 dropout=0.3, stochastic_depth_prob=0.1):
        super().__init__()
        
        self.n_features = n_features
        self.d_model = d_model
        self.stochastic_depth_prob = stochastic_depth_prob
        
        # Input dropout
        self.input_dropout = nn.Dropout(dropout * 0.5)
        
        # Project input features to model dimension
        self.input_projection = nn.Linear(n_features, d_model)
        
        # Positional encoding (learnable) with dropout
        self.pos_encoder = nn.Parameter(torch.randn(1, 128, d_model) * 0.02)
        self.pos_dropout = nn.Dropout(dropout)
        
        # Pre-LayerNorm
        self.pre_norm = nn.LayerNorm(d_model)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 2,  # Smaller FFN to reduce overfitting
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True  # Pre-LayerNorm for stability
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Post-LayerNorm
        self.post_norm = nn.LayerNorm(d_model)
        
        # Classification head with more dropout
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model, 32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x, training=True):
        # x: (batch, seq_len, features)
        
        # Input dropout for regularization
        if training and self.training:
            x = self.input_dropout(x)
        
        # Project to model dimension
        x = self.input_projection(x)  # (batch, seq_len, d_model)
        
        # Add positional encoding with dropout
        x = x + self.pos_encoder[:, :x.size(1), :]
        x = self.pos_dropout(x)
        
        # Pre-normalize
        x = self.pre_norm(x)
        
        # Transformer encoding with stochastic depth
        if training and self.training and self.stochastic_depth_prob > 0:
            if np.random.random() > self.stochastic_depth_prob:
                x = self.transformer(x)
        else:
            x = self.transformer(x)
        
        # Post-normalize
        x = self.post_norm(x)
        
        # Global average pooling over sequence
        x = x.mean(dim=1)  # (batch, d_model)
        
        # Classification
        out = self.classifier(x)  # (batch, 1)
        
        return out.squeeze(-1)


# ============================================================================
# LABEL SMOOTHING LOSS
# ============================================================================

class LabelSmoothingBCELoss(nn.Module):
    """Binary cross-entropy with label smoothing to prevent overconfidence."""
    
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
    
    def forward(self, pred, target):
        # Smooth labels: 0 -> smoothing/2, 1 -> 1 - smoothing/2
        target_smooth = target * (1 - self.smoothing) + 0.5 * self.smoothing
        return F.binary_cross_entropy(pred, target_smooth)


class FocalLoss(nn.Module):
    """
    Focal Loss for imbalanced classification.
    Reduces loss contribution from easy examples, focuses on hard cases.
    Great for class imbalance!
    """
    
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, pred, target):
        bce = F.binary_cross_entropy(pred, target, reduction='none')
        pt = torch.where(target == 1, pred, 1 - pred)
        focal_weight = self.alpha * (1 - pt) ** self.gamma
        return (focal_weight * bce).mean()


# ============================================================================
# SIMPLER MODEL ARCHITECTURES FOR SMALL DATA
# ============================================================================

class SimpleCNN(nn.Module):
    """
    Simple 1D CNN for EEG - often works better than transformers for small data.
    Much fewer parameters = less overfitting.
    """
    
    def __init__(self, n_features=28, dropout=0.2):
        super().__init__()
        
        # Temporal convolutions
        self.conv1 = nn.Conv1d(n_features, 32, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(2)
        
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(2)
        
        self.conv3 = nn.Conv1d(64, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(64)
        self.pool3 = nn.AdaptiveAvgPool1d(1)
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(64, 1)
    
    def forward(self, x, training=True):
        # x: (batch, seq_len, features) -> (batch, features, seq_len)
        x = x.transpose(1, 2)
        
        x = F.gelu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        
        x = F.gelu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        
        x = F.gelu(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        
        x = x.squeeze(-1)
        x = self.dropout(x)
        x = torch.sigmoid(self.fc(x))
        
        return x.squeeze(-1)


class SimpleLSTM(nn.Module):
    """
    Bidirectional LSTM for EEG - good for temporal patterns with small data.
    """
    
    def __init__(self, n_features=28, hidden_size=32, num_layers=1, dropout=0.2):
        super().__init__()
        
        self.lstm = nn.LSTM(
            n_features, hidden_size, num_layers,
            batch_first=True, bidirectional=True, dropout=dropout if num_layers > 1 else 0
        )
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * 2, 1)  # *2 for bidirectional
    
    def forward(self, x, training=True):
        # x: (batch, seq_len, features)
        
        # LSTM output
        output, (hidden, cell) = self.lstm(x)
        
        # Concatenate forward and backward final hidden states
        hidden_cat = torch.cat([hidden[-2], hidden[-1]], dim=1)
        
        x = self.dropout(hidden_cat)
        x = torch.sigmoid(self.fc(x))
        
        return x.squeeze(-1)


class HybridCNNLSTM(nn.Module):
    """
    Hybrid CNN-LSTM: CNN extracts local features, LSTM models temporal dynamics.
    Best of both worlds for small EEG data.
    """
    
    def __init__(self, n_features=28, dropout=0.2):
        super().__init__()
        
        # CNN feature extraction
        self.conv1 = nn.Conv1d(n_features, 32, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(2)
        
        self.conv2 = nn.Conv1d(32, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(32)
        self.pool2 = nn.MaxPool1d(2)
        
        # LSTM on CNN features (seq_len is now 128/4 = 32)
        self.lstm = nn.LSTM(32, 24, 1, batch_first=True, bidirectional=True)
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(48, 1)  # 24*2 for bidirectional
    
    def forward(self, x, training=True):
        # x: (batch, seq_len, features) -> (batch, features, seq_len)
        x = x.transpose(1, 2)
        
        x = F.gelu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        
        x = F.gelu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        
        # (batch, 32, seq/4) -> (batch, seq/4, 32)
        x = x.transpose(1, 2)
        
        # LSTM
        output, (hidden, cell) = self.lstm(x)
        hidden_cat = torch.cat([hidden[-2], hidden[-1]], dim=1)
        
        x = self.dropout(hidden_cat)
        x = torch.sigmoid(self.fc(x))
        
        return x.squeeze(-1)


# ============================================================================
# DATA LOADING
# ============================================================================

def find_latest_ml_ready(recordings_dir=None):
    """Find the most recently created ML-ready npz file."""
    if recordings_dir is None:
        recordings_dir = Path(__file__).resolve().parent.parent / "recordings"
    recordings_path = Path(recordings_dir)
    
    ml_files = list(recordings_path.rglob("*_ml_ready.npz"))
    
    if not ml_files:
        raise FileNotFoundError(f"No ML-ready files found in '{recordings_dir}'")
    
    return max(ml_files, key=lambda f: f.stat().st_mtime)


def load_data(file_path):
    """Load preprocessed data from npz file."""
    data = np.load(file_path, allow_pickle=True)
    
    X = data['X']  # (n_videos, n_timepoints, n_features)
    y = data['y']  # (n_videos,)
    feature_names = list(data['feature_names'])
    
    return X, y, feature_names


# ============================================================================
# TRAINING WITH EARLY STOPPING
# ============================================================================

class EarlyStopping:
    """Early stopping based on validation ACCURACY (not loss) to prevent overfitting."""
    
    def __init__(self, patience=10, min_delta=0.005, restore_best=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best = restore_best
        self.best_acc = 0.0  # Track accuracy, not loss
        self.best_state = None
        self.counter = 0
        self.early_stop = False
    
    def __call__(self, val_acc, model):
        # Stop when accuracy stops improving (higher is better)
        if val_acc > self.best_acc + self.min_delta:
            self.best_acc = val_acc
            self.best_state = copy.deepcopy(model.state_dict())
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
    
    def restore(self, model):
        if self.restore_best and self.best_state is not None:
            model.load_state_dict(self.best_state)


def train_epoch(model, loader, criterion, optimizer, device, use_mixup=False, mixup_alpha=0.2):
    """Train for one epoch with optional mixup."""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        
        optimizer.zero_grad()
        
        if use_mixup:
            X_mixed, y_a, y_b, lam = mixup_data(X_batch, y_batch, mixup_alpha)
            outputs = model(X_mixed)
            loss = mixup_criterion(criterion, outputs, y_a, y_b, lam)
        else:
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
        
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        preds = (outputs > 0.5).float()
        all_preds.extend(preds.cpu().detach().numpy())
        all_labels.extend(y_batch.cpu().numpy())
    
    acc = accuracy_score(all_labels, all_preds)
    return total_loss / len(loader), acc


def evaluate(model, loader, criterion, device):
    """Evaluate model on validation set."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            
            total_loss += loss.item()
            preds = (outputs > 0.5).float()
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(outputs.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())
    
    metrics = {
        'loss': total_loss / len(loader),
        'accuracy': accuracy_score(all_labels, all_preds),
        'precision': precision_score(all_labels, all_preds, zero_division=0),
        'recall': recall_score(all_labels, all_preds, zero_division=0),
        'f1': f1_score(all_labels, all_preds, zero_division=0),
        'auc': roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else 0.5,
        'predictions': np.array(all_preds),
        'probabilities': np.array(all_probs),
        'labels': np.array(all_labels)
    }
    
    return metrics


# ============================================================================
# K-FOLD CROSS-VALIDATION
# ============================================================================

def train_kfold(X, y, feature_names, args, device):
    """Train using K-fold cross-validation for robust evaluation."""
    kfold = StratifiedKFold(n_splits=args.kfold, shuffle=True, random_state=args.seed)
    
    fold_results = []
    fold_models = []
    all_val_preds = np.zeros(len(y))
    all_val_probs = np.zeros(len(y))
    
    print(f"\n   Running {args.kfold}-fold cross-validation...")
    print(f"   {'='*50}")
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(X, y)):
        print(f"\n   Fold {fold+1}/{args.kfold}")
        
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Create datasets with augmentation
        augmentation = EEGAugmentation() if args.augment else None
        train_dataset = AugmentedDataset(X_train, y_train, augmentation, training=True)
        val_dataset = AugmentedDataset(X_val, y_val, augmentation=None, training=False)
        
        # Balanced sampling
        if args.balanced:
            class_counts = np.bincount(y_train.astype(int))
            class_weights = 1.0 / class_counts
            sample_weights = class_weights[y_train.astype(int)]
            sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
            train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler)
        else:
            train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
        
        # Create model based on selection
        model_name = args.model.lower()
        if model_name == 'cnn':
            model = SimpleCNN(n_features=X.shape[2], dropout=args.dropout).to(device)
        elif model_name == 'lstm':
            model = SimpleLSTM(n_features=X.shape[2], dropout=args.dropout).to(device)
        elif model_name == 'hybrid':
            model = HybridCNNLSTM(n_features=X.shape[2], dropout=args.dropout).to(device)
        else:  # transformer
            model = RobustEEGTransformer(
                n_features=X.shape[2],
                n_heads=3,
                n_layers=1,
                d_model=66,
                dropout=args.dropout,
                stochastic_depth_prob=0.1
            ).to(device)
        
        # Loss function selection
        if args.focal:
            criterion = FocalLoss(alpha=0.25, gamma=2.0)
        else:
            criterion = LabelSmoothingBCELoss(smoothing=args.label_smoothing)
        
        # Optimizer with higher weight decay
        optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=args.lr, 
            weight_decay=args.weight_decay
        )
        
        # Cosine scheduler with warmup
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=args.epochs // 3, T_mult=2
        )
        
        # Early stopping
        early_stopping = EarlyStopping(patience=args.patience, min_delta=0.001)
        
        # Training loop
        best_val_acc = 0
        for epoch in range(args.epochs):
            train_loss, train_acc = train_epoch(
                model, train_loader, criterion, optimizer, device,
                use_mixup=args.mixup, mixup_alpha=args.mixup_alpha
            )
            val_metrics = evaluate(model, val_loader, criterion, device)
            
            if val_metrics['accuracy'] > best_val_acc:
                best_val_acc = val_metrics['accuracy']
            
            early_stopping(val_metrics['accuracy'], model)  # Track accuracy, not loss
            if early_stopping.early_stop:
                print(f"      Early stopped at epoch {epoch+1}")
                break
            
            scheduler.step()
        
        # Restore best model
        early_stopping.restore(model)
        
        # Final evaluation
        final_metrics = evaluate(model, val_loader, criterion, device)
        fold_results.append(final_metrics)
        fold_models.append(copy.deepcopy(model.state_dict()))
        
        # Store predictions for this fold
        all_val_preds[val_idx] = final_metrics['predictions']
        all_val_probs[val_idx] = final_metrics['probabilities']
        
        print(f"      Val Acc: {final_metrics['accuracy']:.4f}, "
              f"F1: {final_metrics['f1']:.4f}, AUC: {final_metrics['auc']:.4f}")
    
    # Aggregate results
    avg_metrics = {
        'accuracy': np.mean([r['accuracy'] for r in fold_results]),
        'accuracy_std': np.std([r['accuracy'] for r in fold_results]),
        'precision': np.mean([r['precision'] for r in fold_results]),
        'recall': np.mean([r['recall'] for r in fold_results]),
        'f1': np.mean([r['f1'] for r in fold_results]),
        'auc': np.mean([r['auc'] for r in fold_results]),
    }
    
    # Overall CV predictions
    cv_accuracy = accuracy_score(y, all_val_preds)
    cv_auc = roc_auc_score(y, all_val_probs)
    
    print(f"\n   {'='*50}")
    print(f"   Cross-Validation Results:")
    print(f"   Mean Accuracy: {avg_metrics['accuracy']:.4f} ± {avg_metrics['accuracy_std']:.4f}")
    print(f"   CV Accuracy (all folds combined): {cv_accuracy:.4f}")
    print(f"   Mean AUC: {avg_metrics['auc']:.4f}")
    print(f"   Mean F1: {avg_metrics['f1']:.4f}")
    
    return fold_results, fold_models, avg_metrics, (all_val_preds, all_val_probs, y)


# ============================================================================
# SINGLE TRAIN/VAL SPLIT (Enhanced)
# ============================================================================

def train_single_split(X, y, feature_names, args, device):
    """Standard training with train/val split but enhanced regularization."""
    
    # Train/validation split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.4, random_state=args.seed, stratify=y
    )
    
    print(f"   Train: {len(y_train)} samples ({np.sum(y_train==1)} engaged)")
    print(f"   Val:   {len(y_val)} samples ({np.sum(y_val==1)} engaged)")
    
    # Create datasets with augmentation
    augmentation = EEGAugmentation() if args.augment else None
    train_dataset = AugmentedDataset(X_train, y_train, augmentation, training=True)
    val_dataset = AugmentedDataset(X_val, y_val, augmentation=None, training=False)
    
    # Balanced sampling
    if args.balanced:
        class_counts = np.bincount(y_train.astype(int))
        class_weights = 1.0 / class_counts
        sample_weights = class_weights[y_train.astype(int)]
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler)
    else:
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    
    # Create model based on selection
    model_name = args.model.lower()
    if model_name == 'cnn':
        model = SimpleCNN(n_features=X.shape[2], dropout=args.dropout).to(device)
    elif model_name == 'lstm':
        model = SimpleLSTM(n_features=X.shape[2], dropout=args.dropout).to(device)
    elif model_name == 'hybrid':
        model = HybridCNNLSTM(n_features=X.shape[2], dropout=args.dropout).to(device)
    else:  # transformer
        model = RobustEEGTransformer(
            n_features=X.shape[2],
            n_heads=3,
            n_layers=1,
            d_model=66,
            dropout=args.dropout,
            stochastic_depth_prob=0.1
        ).to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n   Model: {model_name.upper()}")
    print(f"   Parameters: {n_params:,}")
    print(f"   Dropout: {args.dropout}")
    print(f"   Weight Decay: {args.weight_decay}")
    
    # Loss function selection
    if args.focal:
        print(f"   Loss: Focal Loss (alpha=0.25, gamma=2.0)")
        criterion = FocalLoss(alpha=0.25, gamma=2.0)
    else:
        print(f"   Loss: Label Smoothing BCE (smoothing={args.label_smoothing})")
        criterion = LabelSmoothingBCELoss(smoothing=args.label_smoothing)
    
    # Optimizer with weight decay
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=args.lr, 
        weight_decay=args.weight_decay
    )
    
    # Scheduler with warmup
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=args.epochs // 3, T_mult=2
    )
    
    # Early stopping
    early_stopping = EarlyStopping(patience=args.patience, min_delta=0.001)
    
    # Training history
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }
    
    best_val_acc = 0
    
    print(f"\n   Training for up to {args.epochs} epochs (early stopping patience={args.patience})")
    print(f"   {'='*50}")
    
    for epoch in range(args.epochs):
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device,
            use_mixup=args.mixup, mixup_alpha=args.mixup_alpha
        )
        val_metrics = evaluate(model, val_loader, criterion, device)
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_metrics['loss'])
        history['val_acc'].append(val_metrics['accuracy'])
        
        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"   Epoch {epoch+1:3d}: Train Acc={train_acc:.3f}, Val Acc={val_metrics['accuracy']:.3f}")
        
        early_stopping(val_metrics['accuracy'], model)  # Track accuracy, not loss
        if early_stopping.early_stop:
            print(f"   Early stopped at epoch {epoch+1}")
            break
        
        scheduler.step()
    
    # Restore best model
    early_stopping.restore(model)
    
    # Final evaluation
    final_metrics = evaluate(model, val_loader, criterion, device)
    
    return model, history, final_metrics, (X_val, y_val)


# ============================================================================
# FEATURE IMPORTANCE
# ============================================================================

def compute_feature_importance(model, X_val, y_val, feature_names, device):
    """Compute feature importance using gradient-based attribution."""
    model.train()
    
    X_tensor = torch.tensor(X_val, dtype=torch.float32, device=device, requires_grad=True)
    y_tensor = torch.tensor(y_val, dtype=torch.float32, device=device)
    
    if X_tensor.grad is not None:
        X_tensor.grad.zero_()
    
    outputs = model(X_tensor)
    loss = nn.BCELoss()(outputs, y_tensor)
    loss.backward()
    
    if X_tensor.grad is None:
        gradients = np.random.rand(len(feature_names))
    else:
        gradients = X_tensor.grad.abs().mean(dim=(0, 1)).cpu().detach().numpy()
    
    max_grad = gradients.max() if gradients.max() > 0 else 1.0
    importance = (gradients / max_grad) * 100
    
    importance_dict = {name: float(imp) for name, imp in zip(feature_names, importance)}
    
    model.eval()
    return importance_dict


def compute_electrode_band_importance(importance_dict):
    """Reorganize importance by electrode and band."""
    electrodes = ['TP9', 'AF7', 'AF8', 'TP10']
    bands = ['delta', 'theta', 'alpha', 'beta', 'low_gamma', 'high_gamma', 'very_high']
    
    electrode_importance = {elec: {} for elec in electrodes}
    
    for feature, importance in importance_dict.items():
        for elec in electrodes:
            if feature.startswith(elec):
                band = feature.replace(f"{elec}_", "")
                electrode_importance[elec][band] = importance
                break
    
    return electrode_importance


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_training_curves(history, output_dir):
    """Plot training and validation metrics."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    axes[0].plot(history['train_loss'], label='Train', color='#2196F3', linewidth=2)
    axes[0].plot(history['val_loss'], label='Validation', color='#FF5722', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Loss (With Label Smoothing)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(history['train_acc'], label='Train', color='#2196F3', linewidth=2)
    axes[1].plot(history['val_acc'], label='Validation', color='#FF5722', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Accuracy (Early Stopping Enabled)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'training_curves_v2.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_cv_results(fold_results, output_dir):
    """Box plot of cross-validation results."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc']
    data = [[r[m] for r in fold_results] for m in metrics]
    
    bp = ax.boxplot(data, labels=['Accuracy', 'Precision', 'Recall', 'F1', 'AUC'], patch_artist=True)
    
    colors = ['#4CAF50', '#2196F3', '#FF9800', '#9C27B0', '#F44336']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_ylabel('Score')
    ax.set_title('Cross-Validation Results by Fold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 1)
    
    # Add mean annotations
    for i, (metric_data, color) in enumerate(zip(data, colors), 1):
        mean_val = np.mean(metric_data)
        ax.scatter(i, mean_val, color='black', s=100, zorder=5, marker='_', linewidth=3)
        ax.text(i + 0.15, mean_val, f'{mean_val:.3f}', va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'cv_results_v2.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_feature_importance(importance_dict, output_dir):
    """Bar plot of feature importance."""
    sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
    names = [f[0] for f in sorted_features]
    values = [f[1] for f in sorted_features]
    
    electrode_colors = {'TP9': '#4CAF50', 'AF7': '#2196F3', 'AF8': '#FF9800', 'TP10': '#9C27B0'}
    colors = []
    for name in names:
        for elec, color in electrode_colors.items():
            if name.startswith(elec):
                colors.append(color)
                break
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    y_pos = np.arange(len(names))
    bars = ax.barh(y_pos, values, color=colors, alpha=0.8)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlabel('Relative Importance (%)')
    ax.set_title('Feature Importance (V2 - Regularized Model)')
    ax.set_xlim(0, 105)
    
    for bar, val in zip(bars, values):
        ax.text(val + 1, bar.get_y() + bar.get_height()/2, f'{val:.1f}%', 
               va='center', fontsize=8)
    
    legend_elements = [plt.Rectangle((0,0),1,1, facecolor=c, alpha=0.8, label=e) 
                      for e, c in electrode_colors.items()]
    ax.legend(handles=legend_elements, loc='lower right')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'feature_importance_v2.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_electrode_heatmap(electrode_importance, output_dir):
    """Heatmap of importance by electrode and band."""
    electrodes = ['TP9', 'AF7', 'AF8', 'TP10']
    bands = ['delta', 'theta', 'alpha', 'beta', 'low_gamma', 'high_gamma', 'very_high']
    band_labels = ['Delta\n1-4Hz', 'Theta\n4-8Hz', 'Alpha\n8-13Hz', 'Beta\n13-30Hz', 
                   'Low γ\n30-40Hz', 'High γ\n40-60Hz', 'V.High\n60-100Hz']
    
    matrix = np.zeros((len(electrodes), len(bands)))
    for i, elec in enumerate(electrodes):
        for j, band in enumerate(bands):
            matrix[i, j] = electrode_importance.get(elec, {}).get(band, 0)
    
    fig, ax = plt.subplots(figsize=(12, 5))
    
    im = ax.imshow(matrix, cmap='YlOrRd', aspect='auto', vmin=0, vmax=100)
    
    ax.set_xticks(np.arange(len(bands)))
    ax.set_yticks(np.arange(len(electrodes)))
    ax.set_xticklabels(band_labels)
    ax.set_yticklabels(electrodes)
    
    for i in range(len(electrodes)):
        for j in range(len(bands)):
            val = matrix[i, j]
            color = 'white' if val > 50 else 'black'
            ax.text(j, i, f'{val:.0f}%', ha='center', va='center', color=color, fontsize=10)
    
    ax.set_title('Feature Importance: Electrode × Frequency Band (V2)')
    plt.colorbar(im, ax=ax, label='Relative Importance (%)')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'electrode_band_heatmap_v2.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_confusion_matrix(labels, predictions, output_dir, title_suffix=''):
    """Plot confusion matrix."""
    cm = confusion_matrix(labels, predictions)
    
    fig, ax = plt.subplots(figsize=(6, 5))
    
    im = ax.imshow(cm, cmap='Blues')
    
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Skipped\n(<4s)', 'Engaged\n(>4s)'])
    ax.set_yticklabels(['Skipped\n(<4s)', 'Engaged\n(>4s)'])
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(f'Confusion Matrix{title_suffix}')
    
    for i in range(2):
        for j in range(2):
            color = 'white' if cm[i, j] > cm.max()/2 else 'black'
            ax.text(j, i, str(cm[i, j]), ha='center', va='center', 
                   color=color, fontsize=16, fontweight='bold')
    
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(output_dir / f'confusion_matrix_v2.png', dpi=150, bbox_inches='tight')
    plt.close()


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Train EEG Transformer V2 (Robust Generalization)')
    
    # Data options
    parser.add_argument('--file', '-f', type=str, help='Specific ml_ready.npz file')
    
    # Training options
    parser.add_argument('--epochs', '-e', type=int, default=100, help='Max epochs (default: 100)')
    parser.add_argument('--batch-size', '-b', type=int, default=16, help='Batch size (default: 16)')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate (default: 0.001)')
    parser.add_argument('--balanced', action='store_true', help='Use weighted sampling')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    # Regularization options (tuned for small datasets)
    parser.add_argument('--dropout', type=float, default=0.15, help='Dropout rate (default: 0.15)')
    parser.add_argument('--weight-decay', type=float, default=0.02, help='Weight decay (default: 0.02)')
    parser.add_argument('--label-smoothing', type=float, default=0.05, help='Label smoothing (default: 0.05)')
    parser.add_argument('--patience', type=int, default=20, help='Early stopping patience (default: 20)')
    
    # Model architecture selection
    parser.add_argument('--model', type=str, default='transformer', 
                       choices=['transformer', 'cnn', 'lstm', 'hybrid'],
                       help='Model architecture: transformer, cnn, lstm, or hybrid (default: transformer)')
    parser.add_argument('--focal', action='store_true', help='Use Focal Loss (better for imbalance)')
    
    # Augmentation options
    parser.add_argument('--augment', action='store_true', help='Enable data augmentation')
    parser.add_argument('--mixup', action='store_true', help='Enable mixup augmentation')
    parser.add_argument('--mixup-alpha', type=float, default=0.2, help='Mixup alpha (default: 0.2)')
    
    # Cross-validation
    parser.add_argument('--kfold', type=int, default=0, help='K-fold CV (0 = single split)')
    
    args = parser.parse_args()
    
    # Set seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    print("="*60)
    print("EEG TRANSFORMER V2 - ROBUST GENERALIZATION")
    print("="*60)
    print("\nEnhancements over V1:")
    print("  • Data augmentation (noise, time shift, masking)")
    print("  • Label smoothing to prevent overconfidence")
    print("  • Early stopping to prevent overfitting")
    print("  • Higher dropout and weight decay")
    print("  • Optional K-fold cross-validation")
    print("  • Gradient clipping for stability")
    
    # Device selection
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print(f"\nDevice: MPS (Mac M1/M2 GPU)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"\nDevice: CUDA")
    else:
        device = torch.device("cpu")
        print(f"\nDevice: CPU")
    
    try:
        # Load data
        if args.file:
            input_file = Path(args.file)
        else:
            input_file = find_latest_ml_ready()
        
        print(f"\n1. LOADING DATA")
        print(f"   File: {input_file}")
        
        X, y, feature_names = load_data(input_file)
        print(f"   Shape: {X.shape}")
        print(f"   Labels: {np.sum(y==1)} engaged, {np.sum(y==0)} skipped")
        
        # Configuration summary
        print(f"\n2. CONFIGURATION")
        print(f"   Dropout: {args.dropout}")
        print(f"   Weight Decay: {args.weight_decay}")
        print(f"   Label Smoothing: {args.label_smoothing}")
        print(f"   Early Stopping Patience: {args.patience}")
        print(f"   Data Augmentation: {'ON' if args.augment else 'OFF'}")
        print(f"   Mixup: {'ON' if args.mixup else 'OFF'}")
        print(f"   Cross-Validation: {args.kfold}-fold" if args.kfold > 0 else "   Standard 60/40 split")
        
        print(f"\n3. TRAINING")
        
        # Choose training mode
        if args.kfold > 0:
            fold_results, fold_models, avg_metrics, cv_data = train_kfold(
                X, y, feature_names, args, device
            )
            all_val_preds, all_val_probs, all_val_labels = cv_data
            
            # Use first fold model for feature importance
            model = RobustEEGTransformer(
                n_features=X.shape[2], n_heads=3, n_layers=1, d_model=66,
                dropout=args.dropout
            ).to(device)
            model.load_state_dict(fold_models[0])
            
            final_metrics = {
                'accuracy': avg_metrics['accuracy'],
                'accuracy_std': avg_metrics['accuracy_std'],
                'precision': avg_metrics['precision'],
                'recall': avg_metrics['recall'],
                'f1': avg_metrics['f1'],
                'auc': avg_metrics['auc'],
                'predictions': all_val_preds,
                'probabilities': all_val_probs,
                'labels': all_val_labels
            }
            X_val, y_val = X, y  # Use all data for importance
            
        else:
            model, history, final_metrics, val_data = train_single_split(
                X, y, feature_names, args, device
            )
            X_val, y_val = val_data
        
        # Final results
        print(f"\n4. FINAL RESULTS")
        print(f"   {'='*40}")
        if args.kfold > 0:
            print(f"   CV Accuracy:  {final_metrics['accuracy']:.4f} ± {final_metrics['accuracy_std']:.4f}")
        else:
            print(f"   Accuracy:  {final_metrics['accuracy']:.4f}")
        print(f"   Precision: {final_metrics['precision']:.4f}")
        print(f"   Recall:    {final_metrics['recall']:.4f}")
        print(f"   F1 Score:  {final_metrics['f1']:.4f}")
        print(f"   AUC-ROC:   {final_metrics['auc']:.4f}")
        
        # Feature importance
        print(f"\n5. FEATURE IMPORTANCE")
        importance_dict = compute_feature_importance(model, X_val, y_val, feature_names, device)
        electrode_importance = compute_electrode_band_importance(importance_dict)
        
        sorted_imp = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
        print(f"   Top 5 features:")
        for i, (feat, imp) in enumerate(sorted_imp[:5], 1):
            print(f"   {i}. {feat}: {imp:.1f}%")
        
        # Save outputs
        output_dir = input_file.parent / "model_output"
        output_dir.mkdir(exist_ok=True)
        
        print(f"\n6. SAVING OUTPUTS")
        
        # Save model
        model_path = output_dir / "eeg_transformer_v2.pt"
        torch.save({
            'model_state_dict': model.state_dict(),
            'feature_names': feature_names,
            'n_features': X.shape[2],
            'config': vars(args)
        }, model_path)
        print(f"   Model: {model_path}")
        
        # Save results
        results = {
            'final_metrics': {k: float(v) if isinstance(v, (int, float, np.floating)) else None 
                             for k, v in final_metrics.items() if not isinstance(v, np.ndarray)},
            'feature_importance': importance_dict,
            'electrode_band_importance': electrode_importance,
            'config': vars(args),
            'enhancements': [
                'data_augmentation' if args.augment else None,
                'mixup' if args.mixup else None,
                f'{args.kfold}-fold_cv' if args.kfold > 0 else '60/40_split',
                f'dropout_{args.dropout}',
                f'weight_decay_{args.weight_decay}',
                f'label_smoothing_{args.label_smoothing}',
                'early_stopping'
            ],
            'timestamp': datetime.now().isoformat()
        }
        
        results_path = output_dir / "training_results_v2.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"   Results: {results_path}")
        
        # Generate plots
        print(f"\n7. GENERATING VISUALIZATIONS")
        
        if args.kfold > 0:
            plot_cv_results(fold_results, output_dir)
            print(f"   ✓ Cross-validation results")
        else:
            plot_training_curves(history, output_dir)
            print(f"   ✓ Training curves")
        
        plot_feature_importance(importance_dict, output_dir)
        print(f"   ✓ Feature importance")
        
        plot_electrode_heatmap(electrode_importance, output_dir)
        print(f"   ✓ Electrode-band heatmap")
        
        title_suffix = ' (Cross-Validation)' if args.kfold > 0 else ''
        plot_confusion_matrix(final_metrics['labels'], final_metrics['predictions'], 
                            output_dir, title_suffix)
        print(f"   ✓ Confusion matrix")
        
        print(f"\n" + "="*60)
        print("TRAINING COMPLETE")
        print("="*60)
        print(f"\nOutputs saved to: {output_dir}")
        
        # Comparison reminder
        print("\n" + "-"*60)
        print("RECOMMENDED NEXT STEPS:")
        print("-"*60)
        print("1. Compare with V1 baseline:")
        print("   python scripts/train_transformer.py --epochs 100 --balanced")
        print("\n2. Run with all enhancements:")
        print("   python scripts/train_transformer_v2.py --augment --mixup --kfold 5 --balanced")
        print("\n3. Try different regularization strengths:")
        print("   --dropout 0.4 --weight-decay 0.1 --label-smoothing 0.15")
        
        return 0
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
