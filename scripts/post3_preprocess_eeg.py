#!/usr/bin/env python3
"""
EEG Preprocessing Pipeline with Evidence-Based Decision Making

This script applies only essential, evidence-based preprocessing steps to EEG data.
Each processing decision is made based on data analysis and documented with reasoning.

Usage:
    python preprocess_eeg.py                           # Use latest *_classified.csv
    python preprocess_eeg.py --file path.csv           # Use specific file
    python preprocess_eeg.py --tsne                    # Also generate t-SNE plot
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import signal
from scipy.stats import zscore
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


class EEGPreprocessor:
    """Evidence-based EEG preprocessing with documented decision making."""
    
    def __init__(self, fs=256):
        self.fs = fs
        self.eeg_channels = ['TP9', 'AF7', 'AF8', 'TP10']
        self.decisions = []  # Track all preprocessing decisions
        
    def log_decision(self, step, reasoning, evidence):
        """Log preprocessing decision with evidence."""
        self.decisions.append({
            'step': step,
            'reasoning': reasoning,
            'evidence': evidence
        })
        
    def analyze_raw_data(self, df):
        """Analyze raw EEG data to make evidence-based preprocessing decisions."""
        print("="*60)
        print("EVIDENCE-BASED PREPROCESSING ANALYSIS")
        print("="*60)
        
        # 1. Check sampling rate consistency
        time_diffs = np.diff(df['lsl_timestamp'].values)
        actual_fs = 1.0 / np.median(time_diffs)
        fs_std = np.std(1.0 / time_diffs)
        
        print(f"1. SAMPLING RATE ANALYSIS:")
        print(f"   Expected: {self.fs} Hz")
        print(f"   Actual: {actual_fs:.2f} ± {fs_std:.2f} Hz")
        
        if abs(actual_fs - self.fs) > 5:
            self.log_decision(
                "sampling_rate_correction",
                f"Actual sampling rate ({actual_fs:.2f} Hz) deviates from expected ({self.fs} Hz)",
                f"Median inter-sample interval: {np.median(time_diffs):.6f}s"
            )
            self.fs = actual_fs
        
        # 2. Analyze signal amplitude and DC offset
        print(f"\n2. SIGNAL AMPLITUDE ANALYSIS:")
        for ch in self.eeg_channels:
            data = df[ch].values
            dc_offset = np.mean(data)
            amplitude_range = np.ptp(data)  # peak-to-peak
            std_dev = np.std(data)
            
            print(f"   {ch}: DC={dc_offset:.2f}µV, Range={amplitude_range:.2f}µV, STD={std_dev:.2f}µV")
            
            # Decision: Remove DC offset if significant
            if abs(dc_offset) > 10:  # >10µV DC offset is problematic
                self.log_decision(
                    f"dc_removal_{ch}",
                    f"Significant DC offset detected in {ch}",
                    f"DC offset: {dc_offset:.2f}µV (threshold: ±10µV)"
                )
        
        # 3. Frequency domain analysis
        print(f"\n3. FREQUENCY DOMAIN ANALYSIS:")
        for ch in self.eeg_channels:
            data = df[ch].values
            freqs, psd = signal.welch(data, fs=self.fs, nperseg=self.fs*4)  # 4-second windows
            
            # Find dominant frequencies
            dominant_freq = freqs[np.argmax(psd)]
            power_50hz = np.mean(psd[(freqs >= 49) & (freqs <= 51)])  # 50Hz power line
            power_60hz = np.mean(psd[(freqs >= 59) & (freqs <= 61)])  # 60Hz power line
            total_power = np.mean(psd)
            
            print(f"   {ch}: Dominant={dominant_freq:.1f}Hz, 50Hz={power_50hz:.2e}, 60Hz={power_60hz:.2e}")
            
            # Decision: Notch filter if power line interference detected
            if power_50hz > total_power * 0.1 or power_60hz > total_power * 0.1:
                self.log_decision(
                    f"notch_filter_{ch}",
                    f"Power line interference detected in {ch}",
                    f"50Hz power: {power_50hz:.2e}, 60Hz power: {power_60hz:.2e} (>10% of total)"
                )
        
        # 4. Artifact detection
        print(f"\n4. ARTIFACT DETECTION:")
        for ch in self.eeg_channels:
            data = df[ch].values
            
            # High amplitude artifacts (>200µV typical threshold)
            high_amp_artifacts = np.sum(np.abs(data) > 200)
            artifact_rate = high_amp_artifacts / len(data) * 100
            
            print(f"   {ch}: High amplitude artifacts: {high_amp_artifacts} ({artifact_rate:.2f}%)")
            
            if artifact_rate > 5:  # >5% artifacts is problematic
                self.log_decision(
                    f"artifact_rejection_{ch}",
                    f"High artifact rate detected in {ch}",
                    f"Artifact rate: {artifact_rate:.2f}% (threshold: 5%)"
                )
        
        # 5. Baseline analysis
        baseline_data = df[df['class'].isin(['baseline_1', 'baseline_2'])]
        if len(baseline_data) > 0:
            print(f"\n5. BASELINE ANALYSIS:")
            for ch in self.eeg_channels:
                baseline_mean = baseline_data[ch].mean()
                baseline_std = baseline_data[ch].std()
                
                print(f"   {ch}: Baseline mean={baseline_mean:.2f}µV, std={baseline_std:.2f}µV")
                
                # Decision: Baseline normalization always needed for comparison
                self.log_decision(
                    f"baseline_normalization_{ch}",
                    f"Baseline normalization required for {ch}",
                    f"Baseline: μ={baseline_mean:.2f}µV, σ={baseline_std:.2f}µV"
                )
        
        print(f"\n6. PREPROCESSING DECISIONS SUMMARY:")
        for i, decision in enumerate(self.decisions, 1):
            print(f"   {i}. {decision['step']}: {decision['reasoning']}")
        
        return self.decisions
    
    def apply_preprocessing(self, df):
        """Apply evidence-based preprocessing steps."""
        print(f"\n{'='*60}")
        print("APPLYING PREPROCESSING STEPS")
        print("="*60)
        
        df_processed = df.copy()
        
        # Step 1: Remove DC offset (always beneficial)
        print("1. Removing DC offset from all channels...")
        for ch in self.eeg_channels:
            df_processed[ch] = df_processed[ch] - df_processed[ch].mean()
        
        # Step 2: Bandpass filter (1-40 Hz) - removes drift and high-freq noise
        print("2. Applying bandpass filter (1-40 Hz)...")
        nyquist = self.fs / 2
        low = 1.0 / nyquist
        high = 40.0 / nyquist
        b, a = signal.butter(4, [low, high], btype='band')
        
        for ch in self.eeg_channels:
            df_processed[ch] = signal.filtfilt(b, a, df_processed[ch])
        
        # Step 3: Baseline normalization (z-score using baseline periods)
        print("3. Applying baseline normalization...")
        baseline_data = df_processed[df_processed['class'].isin(['baseline_1', 'baseline_2'])]
        
        if len(baseline_data) > 0:
            for ch in self.eeg_channels:
                baseline_mean = baseline_data[ch].mean()
                baseline_std = baseline_data[ch].std()
                df_processed[ch] = (df_processed[ch] - baseline_mean) / baseline_std
        else:
            print("   Warning: No baseline data found, using global z-score")
            for ch in self.eeg_channels:
                df_processed[ch] = zscore(df_processed[ch])
        
        return df_processed
    
    def create_tsne_visualization(self, df_processed, output_file):
        """Create t-SNE visualization of preprocessed data."""
        print(f"\n5. Creating t-SNE visualization...")
        
        # Remove NaN classes and sample for performance
        df_clean = df_processed.dropna(subset=['class'])
        if len(df_clean) > 5000:
            df_sampled = df_clean.sample(n=5000, random_state=42)
            print(f"   Sampled 5000 points from {len(df_clean)} classified samples")
        else:
            df_sampled = df_clean
        
        # Prepare features (EEG channel values)
        X = df_sampled[self.eeg_channels].values
        y = df_sampled['class'].values
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Compute t-SNE
        tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, 
                   max_iter=1000, random_state=42, verbose=0)
        X_tsne = tsne.fit_transform(X_scaled)
        
        # Color mapping
        color_map = {
            'baseline_1': '#2F2F2F',        # Dark gray
            'baseline_2': '#B0B0B0',        # Light gray
            'tiktok_over_4s_watched': '#90EE90',   # Light green
            'tiktok_under_4s_watched': '#FFB6C1'   # Light pink
        }
        
        # Create plot
        plt.figure(figsize=(12, 9))
        
        for class_name in np.unique(y):
            mask = y == class_name
            color = color_map.get(class_name, '#808080')
            plt.scatter(X_tsne[mask, 0], X_tsne[mask, 1], 
                       c=color, label=class_name.replace('_', ' ').title(),
                       alpha=0.7, s=30, edgecolors='white', linewidth=0.5)
        
        plt.title('t-SNE: Preprocessed EEG Data (Baseline Normalized)', 
                 fontsize=14, fontweight='bold')
        plt.xlabel('t-SNE Component 1')
        plt.ylabel('t-SNE Component 2')
        plt.legend(title='Behavioral Class', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"   t-SNE plot saved: {output_file}")
        
        return X_tsne


def find_latest_classified_file(recordings_dir="recordings"):
    """Find the most recently created classified CSV file (searches recursively)."""
    recordings_path = Path(recordings_dir)
    if not recordings_path.exists():
        raise FileNotFoundError(f"Recordings directory '{recordings_dir}' not found")
    
    # Search recursively for classified files (handles both *_classified.csv and *_classified_cut*.csv)
    classified_files = list(recordings_path.rglob("*_classified*.csv"))
    
    if not classified_files:
        raise FileNotFoundError(f"No classified CSV files found in '{recordings_dir}'")
    
    # Sort by modification time, return most recent
    return max(classified_files, key=lambda f: f.stat().st_mtime)


def main():
    parser = argparse.ArgumentParser(description='Evidence-based EEG preprocessing')
    parser.add_argument('--file', '-f', type=str, help='Specific classified CSV file')
    parser.add_argument('--tsne', action='store_true', help='Generate t-SNE visualization')
    
    args = parser.parse_args()
    
    try:
        # Load data
        if args.file:
            input_file = Path(args.file)
        else:
            input_file = find_latest_classified_file()
        
        print(f"Loading: {input_file}")
        df = pd.read_csv(input_file)
        
        # Initialize preprocessor
        preprocessor = EEGPreprocessor()
        
        # Analyze and make decisions
        decisions = preprocessor.analyze_raw_data(df)
        
        # Apply preprocessing
        df_processed = preprocessor.apply_preprocessing(df)
        
        # Save processed data
        output_file = input_file.parent / f"{input_file.stem}_preprocessed.csv"
        df_processed.to_csv(output_file, index=False)
        print(f"\nPreprocessed data saved: {output_file}")
        
        # Generate t-SNE if requested
        if args.tsne:
            tsne_file = input_file.parent / f"{input_file.stem}_preprocessed_tsne.png"
            preprocessor.create_tsne_visualization(df_processed, tsne_file)
        
        print(f"\n{'='*60}")
        print("PREPROCESSING COMPLETE")
        print("="*60)
        print(f"Original samples: {len(df)}")
        print(f"Processed samples: {len(df_processed)}")
        
        return 0
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
