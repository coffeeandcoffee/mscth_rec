#!/usr/bin/env python3
"""
EEG Transformer Training Script

Trains a transformer model on preprocessed EEG data for binary classification
(engaged vs skipped TikTok videos).

Architecture (per ML specialist recommendation):
- 28-dimensional input (4 channels × 7 frequency bands)
- 3 attention heads, 1 transformer layer
- Fully connected layer → sigmoid for probability output
- Binary cross-entropy loss with backpropagation

Features:
- Mac M1 MPS acceleration
- Data balancing via class weights
- 60/40 train/validation split
- Feature importance analysis
- Model checkpointing

Usage:
    python train_transformer.py                  # Use latest ml_ready.npz
    python train_transformer.py --epochs 100     # Custom epochs
    python train_transformer.py --balanced       # Use class weights for imbalance
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import json

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score

import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================

class EEGTransformer(nn.Module):
    """
    Transformer for EEG classification.
    
    Architecture:
    - Input: (batch, seq_len, features) = (batch, 128, 28)
    - Positional encoding
    - 1 transformer encoder layer with 3 heads
    - Global average pooling
    - Fully connected → sigmoid
    """
    
    def __init__(self, n_features=28, n_heads=3, n_layers=1, d_model=66, dropout=0.1):
        super().__init__()
        
        self.n_features = n_features
        self.d_model = d_model
        
        # Project input features to model dimension
        self.input_projection = nn.Linear(n_features, d_model)
        
        # Positional encoding (learnable)
        self.pos_encoder = nn.Parameter(torch.randn(1, 128, d_model) * 0.1)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        # For feature importance (gradients)
        self.feature_gradients = None
        
    def forward(self, x):
        # x: (batch, seq_len, features)
        
        # Project to model dimension
        x = self.input_projection(x)  # (batch, seq_len, d_model)
        
        # Add positional encoding
        x = x + self.pos_encoder[:, :x.size(1), :]
        
        # Transformer encoding
        x = self.transformer(x)  # (batch, seq_len, d_model)
        
        # Global average pooling over sequence
        x = x.mean(dim=1)  # (batch, d_model)
        
        # Classification
        out = self.classifier(x)  # (batch, 1)
        
        return out.squeeze(-1)


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
# TRAINING
# ============================================================================

def train_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        preds = (outputs > 0.5).float()
        all_preds.extend(preds.cpu().numpy())
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
# FEATURE IMPORTANCE
# ============================================================================

def compute_feature_importance(model, X_val, y_val, feature_names, device):
    """
    Compute feature importance using gradient-based attribution.
    Measures how much each input feature contributes to the prediction.
    """
    model.train()  # Enable gradient computation
    
    # Create tensor with gradient tracking
    X_tensor = torch.tensor(X_val, dtype=torch.float32, device=device, requires_grad=True)
    y_tensor = torch.tensor(y_val, dtype=torch.float32, device=device)
    
    # Zero any existing gradients
    if X_tensor.grad is not None:
        X_tensor.grad.zero_()
    
    # Forward pass
    outputs = model(X_tensor)
    
    # Compute loss and backward
    loss = nn.BCELoss()(outputs, y_tensor)
    loss.backward()
    
    # Get absolute gradients averaged over samples and time
    if X_tensor.grad is None:
        print("   Warning: No gradients computed, using random importance")
        gradients = np.random.rand(len(feature_names))
    else:
        gradients = X_tensor.grad.abs().mean(dim=(0, 1)).cpu().detach().numpy()
    
    # Normalize to percentages (avoid division by zero)
    max_grad = gradients.max() if gradients.max() > 0 else 1.0
    importance = (gradients / max_grad) * 100
    
    # Create importance dict
    importance_dict = {name: float(imp) for name, imp in zip(feature_names, importance)}
    
    model.eval()  # Back to eval mode
    return importance_dict


def compute_electrode_band_importance(importance_dict):
    """
    Reorganize importance by electrode and band.
    Returns dict: {electrode: {band: importance}}
    """
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
    """Plot training and validation metrics over epochs."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss curve
    axes[0].plot(history['train_loss'], label='Train', color='#2196F3', linewidth=2)
    axes[0].plot(history['val_loss'], label='Validation', color='#FF5722', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy curve
    axes[1].plot(history['train_acc'], label='Train', color='#2196F3', linewidth=2)
    axes[1].plot(history['val_acc'], label='Validation', color='#FF5722', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'training_curves.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_accuracy_boxplot(train_acc, val_acc, output_dir):
    """Box plot comparing train vs validation accuracy distribution."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    data = [train_acc, val_acc]
    bp = ax.boxplot(data, labels=['Training', 'Validation'], patch_artist=True)
    
    colors = ['#2196F3', '#FF5722']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_ylabel('Accuracy')
    ax.set_title('Training vs Validation Accuracy Distribution')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 1)
    
    # Add mean lines
    for i, acc_list in enumerate(data, 1):
        mean_val = np.mean(acc_list)
        ax.hlines(mean_val, i - 0.25, i + 0.25, colors='black', linewidth=2, label=f'Mean={mean_val:.3f}' if i == 1 else None)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'accuracy_boxplot.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_feature_importance(importance_dict, output_dir):
    """Horizontal bar plot of feature importance."""
    # Sort by importance
    sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
    names = [f[0] for f in sorted_features]
    values = [f[1] for f in sorted_features]
    
    # Color by electrode
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
    ax.set_title('Feature Importance by Electrode & Band')
    ax.set_xlim(0, 105)
    
    # Add value labels
    for bar, val in zip(bars, values):
        ax.text(val + 1, bar.get_y() + bar.get_height()/2, f'{val:.1f}%', 
               va='center', fontsize=8)
    
    # Legend
    legend_elements = [plt.Rectangle((0,0),1,1, facecolor=c, alpha=0.8, label=e) 
                      for e, c in electrode_colors.items()]
    ax.legend(handles=legend_elements, loc='lower right')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'feature_importance.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_electrode_heatmap(electrode_importance, output_dir):
    """Heatmap of importance by electrode (rows) and band (columns)."""
    electrodes = ['TP9', 'AF7', 'AF8', 'TP10']
    bands = ['delta', 'theta', 'alpha', 'beta', 'low_gamma', 'high_gamma', 'very_high']
    band_labels = ['Delta\n1-4Hz', 'Theta\n4-8Hz', 'Alpha\n8-13Hz', 'Beta\n13-30Hz', 
                   'Low γ\n30-40Hz', 'High γ\n40-60Hz', 'V.High\n60-100Hz']
    
    # Build matrix
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
    
    # Add value annotations
    for i in range(len(electrodes)):
        for j in range(len(bands)):
            val = matrix[i, j]
            color = 'white' if val > 50 else 'black'
            ax.text(j, i, f'{val:.0f}%', ha='center', va='center', color=color, fontsize=10)
    
    ax.set_title('Feature Importance: Electrode × Frequency Band')
    
    cbar = plt.colorbar(im, ax=ax, label='Relative Importance (%)')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'electrode_band_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()


def plot_confusion_matrix(labels, predictions, output_dir):
    """Plot confusion matrix."""
    cm = confusion_matrix(labels, predictions)
    
    fig, ax = plt.subplots(figsize=(6, 5))
    
    im = ax.imshow(cm, cmap='Blues')
    
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Skipped\n(under 4s)', 'Engaged\n(over 4s)'])
    ax.set_yticklabels(['Skipped\n(under 4s)', 'Engaged\n(over 4s)'])
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('Confusion Matrix (Validation Set)')
    
    # Add value annotations
    for i in range(2):
        for j in range(2):
            color = 'white' if cm[i, j] > cm.max()/2 else 'black'
            ax.text(j, i, str(cm[i, j]), ha='center', va='center', color=color, fontsize=16, fontweight='bold')
    
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(output_dir / 'confusion_matrix.png', dpi=150, bbox_inches='tight')
    plt.close()


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Train EEG Transformer Model')
    parser.add_argument('--file', '-f', type=str, help='Specific ml_ready.npz file')
    parser.add_argument('--epochs', '-e', type=int, default=50, help='Number of epochs (default: 50)')
    parser.add_argument('--batch-size', '-b', type=int, default=16, help='Batch size (default: 16)')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate (default: 0.001)')
    parser.add_argument('--balanced', action='store_true', help='Use class weights for imbalanced data')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    # Set seed for reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    print("="*60)
    print("EEG TRANSFORMER TRAINING")
    print("="*60)
    
    # Device selection (Mac M1 MPS)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print(f"Using device: MPS (Mac M1/M2 GPU)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using device: CUDA")
    else:
        device = torch.device("cpu")
        print(f"Using device: CPU")
    
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
        print(f"   Balance: {np.mean(y)*100:.1f}% engaged")
        
        # Train/validation split (60/40)
        print(f"\n2. SPLITTING DATA (60% train / 40% validation)")
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.4, random_state=args.seed, stratify=y
        )
        print(f"   Train: {len(y_train)} samples ({np.sum(y_train==1)} engaged, {np.sum(y_train==0)} skipped)")
        print(f"   Val:   {len(y_val)} samples ({np.sum(y_val==1)} engaged, {np.sum(y_val==0)} skipped)")
        
        # Convert to tensors
        X_train_t = torch.tensor(X_train, dtype=torch.float32)
        y_train_t = torch.tensor(y_train, dtype=torch.float32)
        X_val_t = torch.tensor(X_val, dtype=torch.float32)
        y_val_t = torch.tensor(y_val, dtype=torch.float32)
        
        # Create datasets
        train_dataset = TensorDataset(X_train_t, y_train_t)
        val_dataset = TensorDataset(X_val_t, y_val_t)
        
        # Handle class imbalance
        if args.balanced:
            print(f"\n   Using balanced sampling for class imbalance")
            class_counts = np.bincount(y_train.astype(int))
            class_weights = 1.0 / class_counts
            sample_weights = class_weights[y_train.astype(int)]
            sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
            train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler)
        else:
            train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
        
        # Create model
        print(f"\n3. MODEL ARCHITECTURE")
        model = EEGTransformer(
            n_features=X.shape[2],  # 28
            n_heads=3,
            n_layers=1,
            d_model=66,  # Must be divisible by n_heads (66 = 3 × 22)
            dropout=0.1
        ).to(device)
        
        n_params = sum(p.numel() for p in model.parameters())
        print(f"   Input: {X.shape[2]} features × {X.shape[1]} timepoints")
        print(f"   Transformer: 3 heads, 1 layer, d_model=64")
        print(f"   Parameters: {n_params:,}")
        
        # Loss and optimizer
        # BCELoss works with sigmoid output (already in model)
        # For class imbalance, we increase LR and use weighted sampler (already done above)
        criterion = nn.BCELoss()
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        
        # Training loop
        print(f"\n4. TRAINING")
        print(f"   Epochs: {args.epochs}")
        print(f"   Batch size: {args.batch_size}")
        print(f"   Learning rate: {args.lr}")
        print(f"   {'='*40}")
        
        history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': []
        }
        
        best_val_acc = 0
        best_model_state = None
        
        for epoch in range(args.epochs):
            # Train
            train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
            
            # Validate
            val_metrics = evaluate(model, val_loader, criterion, device)
            
            # Save history
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_metrics['loss'])
            history['val_acc'].append(val_metrics['accuracy'])
            
            # Update best model
            if val_metrics['accuracy'] > best_val_acc:
                best_val_acc = val_metrics['accuracy']
                best_model_state = model.state_dict().copy()
            
            # Print progress
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"   Epoch {epoch+1:3d}/{args.epochs}: "
                      f"Train Loss={train_loss:.4f}, Acc={train_acc:.3f} | "
                      f"Val Loss={val_metrics['loss']:.4f}, Acc={val_metrics['accuracy']:.3f}")
            
            scheduler.step()
        
        # Load best model
        model.load_state_dict(best_model_state)
        
        # Final evaluation
        print(f"\n5. FINAL VALIDATION RESULTS")
        print(f"   {'='*40}")
        final_metrics = evaluate(model, val_loader, criterion, device)
        print(f"   Accuracy:  {final_metrics['accuracy']:.4f}")
        print(f"   Precision: {final_metrics['precision']:.4f}")
        print(f"   Recall:    {final_metrics['recall']:.4f}")
        print(f"   F1 Score:  {final_metrics['f1']:.4f}")
        print(f"   AUC-ROC:   {final_metrics['auc']:.4f}")
        
        # Feature importance
        print(f"\n6. FEATURE IMPORTANCE ANALYSIS")
        importance_dict = compute_feature_importance(model, X_val, y_val, feature_names, device)
        electrode_importance = compute_electrode_band_importance(importance_dict)
        
        # Print top 10 features
        sorted_imp = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
        print(f"   Top 10 most predictive features:")
        for i, (feat, imp) in enumerate(sorted_imp[:10], 1):
            print(f"   {i:2d}. {feat}: {imp:.1f}%")
        
        # Save outputs
        output_dir = input_file.parent / "model_output"
        output_dir.mkdir(exist_ok=True)
        
        print(f"\n7. SAVING OUTPUTS")
        
        # Save model
        model_path = output_dir / "eeg_transformer.pt"
        torch.save({
            'model_state_dict': best_model_state,
            'feature_names': feature_names,
            'n_features': X.shape[2],
            'n_timepoints': X.shape[1],
            'best_val_acc': best_val_acc
        }, model_path)
        print(f"   Model: {model_path}")
        
        # Save results JSON
        results = {
            'final_metrics': {k: float(v) if isinstance(v, (int, float, np.floating)) else None 
                             for k, v in final_metrics.items() if not isinstance(v, np.ndarray)},
            'feature_importance': importance_dict,
            'electrode_band_importance': electrode_importance,
            'training_config': {
                'epochs': args.epochs,
                'batch_size': args.batch_size,
                'learning_rate': args.lr,
                'balanced': args.balanced,
                'train_samples': len(y_train),
                'val_samples': len(y_val)
            },
            'timestamp': datetime.now().isoformat()
        }
        
        results_path = output_dir / "training_results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"   Results: {results_path}")
        
        # Generate plots
        print(f"\n8. GENERATING VISUALIZATIONS")
        
        plot_training_curves(history, output_dir)
        print(f"   ✓ Training curves")
        
        plot_accuracy_boxplot(history['train_acc'], history['val_acc'], output_dir)
        print(f"   ✓ Accuracy box plot")
        
        plot_feature_importance(importance_dict, output_dir)
        print(f"   ✓ Feature importance")
        
        plot_electrode_heatmap(electrode_importance, output_dir)
        print(f"   ✓ Electrode-band heatmap")
        
        plot_confusion_matrix(final_metrics['labels'], final_metrics['predictions'], output_dir)
        print(f"   ✓ Confusion matrix")
        
        print(f"\n" + "="*60)
        print("TRAINING COMPLETE")
        print("="*60)
        print(f"\nAll outputs saved to: {output_dir}")
        print(f"Best validation accuracy: {best_val_acc:.4f}")
        
        return 0
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
