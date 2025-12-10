#!/usr/bin/env python3
"""
EEG Frequency Band Feature Extraction (Welch Method)

Extracts canonical frequency band powers from preprocessed EEG data using Welch's PSD method.
Applies baseline drift correction and creates ML-ready feature matrix.

Usage:
    python preprocess_eeg_v2.py                           # Use latest *_preprocessed.csv
    python preprocess_eeg_v2.py --file path.csv           # Use specific file
    python preprocess_eeg_v2.py --tsne                    # Also generate t-SNE plot
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import signal
from scipy.stats import zscore, mannwhitneyu, normaltest, levene
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


class EEGBandExtractor:
    """Extract frequency band features using Welch's method."""
    
    def __init__(self, fs=256):
        self.fs = fs
        self.eeg_channels = ['TP9', 'AF7', 'AF8', 'TP10']
        
        # Canonical EEG frequency bands
        self.bands = {
            'delta': (1, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 40)
        }
        
        self.feature_names = []
        for ch in self.eeg_channels:
            for band in self.bands.keys():
                self.feature_names.append(f'{ch}_{band}')
    
    def extract_segments(self, df):
        """Extract one segment per TikTok block (already cut to specified duration in post2)."""
        print("="*60)
        print("EXTRACTING SEGMENTS FOR BAND ANALYSIS")
        print("="*60)
        
        segments = []
        min_segment_size = int(0.1 * self.fs)  # Minimum 100ms for valid segment
        
        # For TikTok classes, use A keypresses to identify segment boundaries
        # For baseline classes, use continuous blocks
        df_clean = df.dropna(subset=['class'])
        
        # Find all A keypresses for TikTok segments
        a_presses = df_clean[df_clean['keypress_A'] == 1].index.tolist()
        
        for class_name in df_clean['class'].unique():
            class_data = df_clean[df_clean['class'] == class_name]
            
            if class_name in ['tiktok_over_4s_watched', 'tiktok_under_4s_watched']:
                # For TikTok classes: one segment per A keypress
                for a_press_idx in a_presses:
                    # Check if this A keypress belongs to this class
                    if a_press_idx < len(df) and df.loc[a_press_idx, 'class'] == class_name:
                        # Find the end of this segment (next A keypress or class change)
                        segment_end_idx = len(df)
                        
                        for next_idx in range(a_press_idx + 1, len(df)):
                            if (df.loc[next_idx, 'keypress_A'] == 1 or 
                                df.loc[next_idx, 'class'] != class_name):
                                segment_end_idx = next_idx
                                break
                        
                        # Extract segment from A keypress to end (only rows of this class)
                        segment_indices = []
                        for idx in range(a_press_idx, segment_end_idx):
                            if idx < len(df) and df.loc[idx, 'class'] == class_name:
                                segment_indices.append(idx)
                        
                        if len(segment_indices) >= min_segment_size:
                            segment_data_dict = {
                                'segment_id': len(segments),
                                'class': class_name,
                                'start_idx': segment_indices[0],
                                'end_idx': segment_indices[-1],
                                'timestamp_start': df.loc[segment_indices[0], 'lsl_timestamp'],
                                'timestamp_end': df.loc[segment_indices[-1], 'lsl_timestamp'],
                                'duration': df.loc[segment_indices[-1], 'lsl_timestamp'] - df.loc[segment_indices[0], 'lsl_timestamp']
                            }
                            
                            # Extract EEG data for this segment
                            for ch in self.eeg_channels:
                                segment_data_dict[f'{ch}_data'] = df.loc[segment_indices, ch].values
                            
                            segments.append(segment_data_dict)
            else:
                # For baseline classes: use continuous blocks
                indices = class_data.index.values
                breaks = np.where(np.diff(indices) > 1)[0] + 1
                blocks = np.split(indices, breaks)
                
                for block in blocks:
                    if len(block) >= min_segment_size:
                        segment_indices = block
                        
                        segment_data = {
                            'segment_id': len(segments),
                            'class': class_name,
                            'start_idx': segment_indices[0],
                            'end_idx': segment_indices[-1],
                            'timestamp_start': df.loc[segment_indices[0], 'lsl_timestamp'],
                            'timestamp_end': df.loc[segment_indices[-1], 'lsl_timestamp'],
                            'duration': df.loc[segment_indices[-1], 'lsl_timestamp'] - df.loc[segment_indices[0], 'lsl_timestamp']
                        }
                        
                        # Extract EEG data for this segment
                        for ch in self.eeg_channels:
                            segment_data[f'{ch}_data'] = df.loc[segment_indices, ch].values
                        
                        segments.append(segment_data)
        
        print(f"Extracted {len(segments)} segments:")
        segment_counts = {}
        for seg in segments:
            class_name = seg['class']
            segment_counts[class_name] = segment_counts.get(class_name, 0) + 1
        
        for class_name, count in segment_counts.items():
            print(f"  {class_name}: {count} segments")
        
        return segments
    
    def compute_band_powers(self, segments):
        """Compute Welch PSD band powers for each segment."""
        print(f"\nCOMPUTING WELCH BAND POWERS")
        print("="*40)
        
        feature_matrix = []
        
        for i, segment in enumerate(segments):
            if i % 50 == 0:
                print(f"Processing segment {i+1}/{len(segments)}...")
            
            segment_features = {
                'segment_id': segment['segment_id'],
                'class': segment['class'],
                'timestamp_start': segment['timestamp_start'],
                'timestamp_end': segment['timestamp_end']
            }
            
            # Compute band powers for each channel
            for ch in self.eeg_channels:
                eeg_data = segment[f'{ch}_data']
                
                # Welch PSD estimation
                freqs, psd = signal.welch(
                    eeg_data, 
                    fs=self.fs,
                    nperseg=len(eeg_data),  # Use full segment length
                    noverlap=0,
                    window='hann'
                )
                
                # Extract power in each frequency band
                for band_name, (low_freq, high_freq) in self.bands.items():
                    # Find frequency indices for this band
                    band_mask = (freqs >= low_freq) & (freqs <= high_freq)
                    
                    if np.any(band_mask):
                        # Integrate PSD over the band
                        band_power = np.trapz(psd[band_mask], freqs[band_mask])
                        
                        # Log transform (add small epsilon to avoid log(0))
                        log_power = np.log10(band_power + 1e-12)
                        
                        segment_features[f'{ch}_{band_name}'] = log_power
                    else:
                        segment_features[f'{ch}_{band_name}'] = np.nan
            
            feature_matrix.append(segment_features)
        
        df_features = pd.DataFrame(feature_matrix)
        
        print(f"Computed band powers for {len(df_features)} segments")
        print(f"Features per segment: {len(self.feature_names)}")
        
        return df_features
    
    def apply_baseline_drift_correction(self, df_features):
        """Apply linear baseline drift correction."""
        print(f"\nAPPLYING BASELINE DRIFT CORRECTION")
        print("="*40)
        
        df_corrected = df_features.copy()
        
        # Get baseline segments
        baseline1 = df_features[df_features['class'] == 'baseline_1']
        baseline2 = df_features[df_features['class'] == 'baseline_2']
        
        if len(baseline1) == 0 or len(baseline2) == 0:
            print("Warning: Missing baseline data, skipping drift correction")
            return df_corrected
        
        # Calculate median timestamps for baselines
        baseline1_time = baseline1['timestamp_start'].median()
        baseline2_time = baseline2['timestamp_start'].median()
        
        print(f"Baseline 1: {len(baseline1)} segments at t={baseline1_time:.1f}s")
        print(f"Baseline 2: {len(baseline2)} segments at t={baseline2_time:.1f}s")
        
        # Apply drift correction for each feature
        corrections_applied = 0
        
        for feature_name in self.feature_names:
            # Calculate baseline medians
            baseline1_median = baseline1[feature_name].median()
            baseline2_median = baseline2[feature_name].median()
            
            # Linear interpolation parameters
            slope = (baseline2_median - baseline1_median) / (baseline2_time - baseline1_time)
            intercept = baseline1_median - slope * baseline1_time
            
            # Apply correction to all segments
            for idx, row in df_corrected.iterrows():
                timestamp = row['timestamp_start']
                expected_baseline = slope * timestamp + intercept
                df_corrected.loc[idx, feature_name] = row[feature_name] - expected_baseline
            
            corrections_applied += 1
        
        print(f"Applied drift correction to {corrections_applied} features")
        
        return df_corrected
    
    def normalize_features(self, df_features):
        """Z-score normalize features using baseline data."""
        print(f"\nNORMALIZING FEATURES")
        print("="*25)
        
        df_normalized = df_features.copy()
        
        # Use all baseline data for normalization statistics
        baseline_data = df_features[df_features['class'].isin(['baseline_1', 'baseline_2'])]
        
        if len(baseline_data) == 0:
            print("Warning: No baseline data found, using global normalization")
            baseline_data = df_features
        
        # Normalize each feature
        for feature_name in self.feature_names:
            baseline_values = baseline_data[feature_name].dropna()
            
            if len(baseline_values) > 1:
                mean_val = baseline_values.mean()
                std_val = baseline_values.std()
                
                if std_val > 0:
                    df_normalized[feature_name] = (df_features[feature_name] - mean_val) / std_val
                else:
                    df_normalized[feature_name] = df_features[feature_name] - mean_val
        
        print(f"Normalized {len(self.feature_names)} features using baseline statistics")
        
        return df_normalized
    
    def create_tsne_visualization(self, df_features, output_file):
        """Create t-SNE visualization of band power features."""
        print(f"\nCREATING t-SNE VISUALIZATION")
        print("="*35)
        
        # Prepare feature matrix
        X = df_features[self.feature_names].values
        y = df_features['class'].values
        
        # Remove any NaN values
        valid_mask = ~np.isnan(X).any(axis=1)
        X_clean = X[valid_mask]
        y_clean = y[valid_mask]
        
        print(f"Using {len(X_clean)} segments for t-SNE")
        
        # Sample if too many points
        if len(X_clean) > 1000:
            indices = np.random.choice(len(X_clean), 1000, replace=False)
            X_clean = X_clean[indices]
            y_clean = y_clean[indices]
            print(f"Sampled 1000 points for visualization")
        
        # Standardize features for t-SNE
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_clean)
        
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
        
        for class_name in np.unique(y_clean):
            mask = y_clean == class_name
            color = color_map.get(class_name, '#808080')
            plt.scatter(X_tsne[mask, 0], X_tsne[mask, 1],
                       c=color, label=class_name.replace('_', ' ').title(),
                       alpha=0.7, s=40, edgecolors='white', linewidth=0.5)
        
        plt.title('t-SNE: EEG Frequency Band Features (Welch PSD, Baseline Corrected)',
                 fontsize=14, fontweight='bold')
        plt.xlabel('t-SNE Component 1')
        plt.ylabel('t-SNE Component 2')
        plt.legend(title='Behavioral Class', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"t-SNE plot saved: {output_file}")
        
        # Print class distribution
        class_counts = pd.Series(y_clean).value_counts()
        print(f"\nClass distribution in t-SNE:")
        for class_name, count in class_counts.items():
            print(f"  {class_name}: {count} segments")
    
    def create_statistical_comparison(self, df_features, output_file):
        """Create statistical comparison plots between TikTok classes."""
        print(f"\nCREATING STATISTICAL COMPARISON")
        print("="*40)
        
        # Filter to only TikTok classes for comparison
        tiktok_data = df_features[df_features['class'].isin([
            'tiktok_over_4s_watched', 'tiktok_under_4s_watched'
        ])].copy()
        
        if len(tiktok_data) == 0:
            print("ERROR: No TikTok data found for comparison")
            return
        
        # Get sample sizes
        class_counts = tiktok_data['class'].value_counts()
        n_over = class_counts.get('tiktok_over_4s_watched', 0)
        n_under = class_counts.get('tiktok_under_4s_watched', 0)
        
        print(f"Sample sizes: Over 4s: {n_over}, Under 4s: {n_under}")
        
        if n_over < 3 or n_under < 3:
            print("ERROR: Insufficient sample sizes for statistical testing")
            return
        
        # Perform statistical tests
        results = []
        n_comparisons = len(self.eeg_channels) * len(self.bands)
        alpha_corrected = 0.05 / n_comparisons  # Bonferroni correction
        
        print(f"Performing {n_comparisons} comparisons (Bonferroni α = {alpha_corrected:.4f})")
        
        for ch in self.eeg_channels:
            for band in self.bands.keys():
                feature_name = f'{ch}_{band}'
                
                # Extract data for each class
                over_4s = tiktok_data[tiktok_data['class'] == 'tiktok_over_4s_watched'][feature_name].dropna()
                under_4s = tiktok_data[tiktok_data['class'] == 'tiktok_under_4s_watched'][feature_name].dropna()
                
                if len(over_4s) < 3 or len(under_4s) < 3:
                    continue
                
                # Test assumptions
                # 1. Normality test (D'Agostino-Pearson)
                _, norm_p_over = normaltest(over_4s)
                _, norm_p_under = normaltest(under_4s)
                normal_assumption = (norm_p_over > 0.05) and (norm_p_under > 0.05)
                
                # 2. Equal variance test (Levene)
                _, levene_p = levene(over_4s, under_4s)
                equal_var_assumption = levene_p > 0.05
                
                # Mann-Whitney U test (non-parametric, robust for EEG)
                statistic, p_value = mannwhitneyu(over_4s, under_4s, alternative='two-sided')
                
                # Effect size (rank-biserial correlation for Mann-Whitney U)
                n1, n2 = len(over_4s), len(under_4s)
                r_rb = 1 - (2 * statistic) / (n1 * n2)  # Rank-biserial correlation
                
                # Cohen's d for interpretability
                pooled_std = np.sqrt(((n1-1)*over_4s.var() + (n2-1)*under_4s.var()) / (n1+n2-2))
                cohens_d = (over_4s.mean() - under_4s.mean()) / pooled_std if pooled_std > 0 else 0
                
                # Significance after correction
                significant = p_value < alpha_corrected
                
                results.append({
                    'channel': ch,
                    'band': band,
                    'feature': feature_name,
                    'n_over': len(over_4s),
                    'n_under': len(under_4s),
                    'median_over': over_4s.median(),
                    'median_under': under_4s.median(),
                    'mean_over': over_4s.mean(),
                    'mean_under': under_4s.mean(),
                    'std_over': over_4s.std(),
                    'std_under': under_4s.std(),
                    'u_statistic': statistic,
                    'p_value': p_value,
                    'p_corrected': p_value * n_comparisons,  # Bonferroni
                    'significant': significant,
                    'effect_size_r': abs(r_rb),
                    'cohens_d': abs(cohens_d),
                    'normal_over': norm_p_over > 0.05,
                    'normal_under': norm_p_under > 0.05,
                    'equal_variance': equal_var_assumption
                })
        
        results_df = pd.DataFrame(results)
        
        # Create the plot
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        
        # Color scheme
        colors = ['#FFB6C1', '#90EE90']  # Light pink, Light green
        
        for i, ch in enumerate(self.eeg_channels):
            ax = axes[i]
            
            # Prepare data for this channel
            channel_data = []
            channel_labels = []
            positions = []
            sig_annotations = []
            
            pos = 0
            for band in self.bands.keys():
                feature_name = f'{ch}_{band}'
                
                # Get data for both classes
                over_data = tiktok_data[tiktok_data['class'] == 'tiktok_over_4s_watched'][feature_name].dropna()
                under_data = tiktok_data[tiktok_data['class'] == 'tiktok_under_4s_watched'][feature_name].dropna()
                
                if len(over_data) > 0 and len(under_data) > 0:
                    channel_data.extend([under_data, over_data])
                    channel_labels.extend([f'{band}\nUnder 4s', f'{band}\nOver 4s'])
                    positions.extend([pos, pos + 0.4])
                    
                    # Get significance for this comparison
                    result = results_df[results_df['feature'] == feature_name]
                    if len(result) > 0:
                        p_val = result.iloc[0]['p_value']
                        corrected_p = result.iloc[0]['p_corrected']
                        effect_size = result.iloc[0]['effect_size_r']
                        
                        # Significance stars
                        if corrected_p < 0.001:
                            sig_star = '***'
                        elif corrected_p < 0.01:
                            sig_star = '**'
                        elif corrected_p < 0.05:
                            sig_star = '*'
                        else:
                            sig_star = 'ns'
                        
                        sig_annotations.append({
                            'x': pos + 0.2,
                            'text': f'{sig_star}\np={p_val:.3f}\nr={effect_size:.2f}',
                            'significant': corrected_p < 0.05
                        })
                    
                    pos += 1
            
            # Create box plot
            if channel_data:
                bp = ax.boxplot(channel_data, positions=positions, patch_artist=True,
                               labels=channel_labels, widths=0.3)
                
                # Color boxes
                for j, patch in enumerate(bp['boxes']):
                    patch.set_facecolor(colors[j % 2])
                    patch.set_alpha(0.7)
                
                # Add significance annotations
                y_max = max([max(data) for data in channel_data])
                y_range = y_max - min([min(data) for data in channel_data])
                
                for j, ann in enumerate(sig_annotations):
                    y_pos = y_max + 0.1 * y_range + (j % 2) * 0.05 * y_range
                    color = 'red' if ann['significant'] else 'gray'
                    ax.text(ann['x'], y_pos, ann['text'], 
                           ha='center', va='bottom', fontsize=8, color=color,
                           bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
            
            ax.set_title(f'{ch} Channel', fontweight='bold', fontsize=12)
            ax.set_ylabel('Normalized Log Band Power', fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis='x', rotation=45)
        
        # Overall title and info
        fig.suptitle('EEG Band Power Comparison: TikTok Under 4s vs Over 4s\n' +
                    f'Mann-Whitney U Tests, Bonferroni Corrected (α = {alpha_corrected:.4f})',
                    fontsize=14, fontweight='bold')
        
        # Add method info as text
        method_text = (
            f"Statistical Method: Mann-Whitney U test (non-parametric)\n"
            f"Multiple Comparison Correction: Bonferroni ({n_comparisons} tests)\n"
            f"Effect Size: Rank-biserial correlation (r)\n"
            f"Significance: * p<0.05, ** p<0.01, *** p<0.001, ns = not significant\n"
            f"Sample Sizes: Under 4s: n={n_under}, Over 4s: n={n_over}"
        )
        
        fig.text(0.02, 0.02, method_text, fontsize=9, verticalalignment='bottom',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.15, top=0.9)
        
        # Save plot
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Statistical comparison plot saved: {output_file}")
        
        # Print summary
        significant_results = results_df[results_df['significant']]
        print(f"\nSignificant differences found: {len(significant_results)}/{len(results_df)}")
        
        if len(significant_results) > 0:
            print("\nSignificant comparisons:")
            for _, row in significant_results.iterrows():
                print(f"  {row['feature']}: p={row['p_value']:.4f}, r={row['effect_size_r']:.3f}")
        
        # Save detailed results
        results_file = output_file.parent / f"{output_file.stem}_statistics.csv"
        results_df.to_csv(results_file, index=False)
        print(f"Detailed statistics saved: {results_file}")
        
        # Create punchy text summary
        summary_file = output_file.parent / f"{output_file.stem}_summary.txt"
        with open(summary_file, 'w') as f:
            f.write("="*70 + "\n")
            f.write("EEG STATISTICAL ANALYSIS SUMMARY\n")
            f.write("="*70 + "\n\n")
            
            # Quick overview
            n_total = len(results_df)
            n_sig = len(significant_results)
            sig_percent = (n_sig / n_total * 100) if n_total > 0 else 0
            
            f.write("📊 QUICK OVERVIEW\n")
            f.write("-" * 70 + "\n")
            f.write(f"Total Comparisons:     {n_total}\n")
            f.write(f"Significant Results:  {n_sig} ({sig_percent:.1f}%)\n")
            f.write(f"Non-Significant:      {n_total - n_sig} ({100-sig_percent:.1f}%)\n\n")
            
            # Test parameters
            f.write("🔬 TEST PARAMETERS\n")
            f.write("-" * 70 + "\n")
            f.write(f"Statistical Test:     Mann-Whitney U (non-parametric)\n")
            f.write(f"Correction Method:    Bonferroni\n")
            f.write(f"Number of Tests:       {n_comparisons}\n")
            f.write(f"Uncorrected α:        0.05\n")
            f.write(f"Corrected α:          {alpha_corrected:.4f}\n")
            f.write(f"Effect Size Metric:   Rank-biserial correlation (r)\n\n")
            
            # Sample sizes
            f.write("👥 SAMPLE SIZES\n")
            f.write("-" * 70 + "\n")
            f.write(f"TikTok Over 4s:        n = {n_over}\n")
            f.write(f"TikTok Under 4s:      n = {n_under}\n\n")
            
            # Significant findings
            if len(significant_results) > 0:
                f.write("✅ SIGNIFICANT FINDINGS\n")
                f.write("-" * 70 + "\n")
                for _, row in significant_results.iterrows():
                    direction = ">" if row['median_over'] > row['median_under'] else "<"
                    f.write(f"{row['feature']:25s}  ")
                    f.write(f"p={row['p_value']:.4f}  ")
                    f.write(f"r={row['effect_size_r']:.3f}  ")
                    f.write(f"d={row['cohens_d']:.3f}  ")
                    f.write(f"({row['median_over']:.3f} {direction} {row['median_under']:.3f})\n")
                f.write("\n")
            else:
                f.write("❌ NO SIGNIFICANT FINDINGS\n")
                f.write("-" * 70 + "\n")
                f.write("No significant differences found after Bonferroni correction.\n\n")
            
            # Top effect sizes (even if not significant)
            f.write("📈 TOP EFFECT SIZES (by rank-biserial correlation)\n")
            f.write("-" * 70 + "\n")
            top_effects = results_df.nlargest(5, 'effect_size_r')
            for i, (_, row) in enumerate(top_effects.iterrows(), 1):
                sig_marker = "★" if row['significant'] else " "
                f.write(f"{i}. {sig_marker} {row['feature']:25s}  ")
                f.write(f"r={row['effect_size_r']:.3f}  ")
                f.write(f"p={row['p_value']:.4f}\n")
            f.write("\n")
            
            # Method details
            f.write("📋 METHOD DETAILS\n")
            f.write("-" * 70 + "\n")
            f.write("• Frequency Bands:     ")
            f.write(", ".join([f"{k} ({v[0]}-{v[1]} Hz)" for k, v in self.bands.items()]) + "\n")
            f.write("• EEG Channels:       " + ", ".join(self.eeg_channels) + "\n")
            f.write("• Preprocessing:       DC removal, bandpass (1-40 Hz), baseline normalization\n")
            f.write("• Feature Extraction:  Welch PSD, baseline drift correction, z-score normalization\n")
            f.write("• Assumptions Checked: Normality (D'Agostino-Pearson), Equal variance (Levene)\n\n")
            
            # Interpretation guide
            f.write("💡 INTERPRETATION GUIDE\n")
            f.write("-" * 70 + "\n")
            f.write("Effect Size (r):       Small: <0.3, Medium: 0.3-0.5, Large: >0.5\n")
            f.write("Effect Size (Cohen's d): Small: <0.2, Medium: 0.2-0.8, Large: >0.8\n")
            f.write("Significance:          * p<0.05, ** p<0.01, *** p<0.001\n")
            f.write("                      (after Bonferroni correction)\n")
            f.write("="*70 + "\n")
        
        print(f"Summary report saved: {summary_file}")
        
        return results_df


def find_latest_preprocessed_file(recordings_dir="recordings"):
    """Find the most recently created preprocessed CSV file (searches recursively)."""
    recordings_path = Path(recordings_dir)
    if not recordings_path.exists():
        raise FileNotFoundError(f"Recordings directory '{recordings_dir}' not found")
    
    # Search recursively for preprocessed files
    preprocessed_files = list(recordings_path.rglob("*_preprocessed.csv"))
    
    if not preprocessed_files:
        raise FileNotFoundError(f"No preprocessed CSV files found in '{recordings_dir}'")
    
    # Sort by modification time, return most recent
    return max(preprocessed_files, key=lambda f: f.stat().st_mtime)


def main():
    parser = argparse.ArgumentParser(description='Extract EEG frequency band features using Welch PSD')
    parser.add_argument('--file', '-f', type=str, help='Specific preprocessed CSV file')
    parser.add_argument('--tsne', action='store_true', help='Generate t-SNE visualization')
    parser.add_argument('--stats', action='store_true', help='Generate statistical comparison plots')
    
    args = parser.parse_args()
    
    try:
        # Load data
        if args.file:
            input_file = Path(args.file)
        else:
            input_file = find_latest_preprocessed_file()
        
        print(f"Loading: {input_file}")
        df = pd.read_csv(input_file)
        
        # Validate required columns
        required_cols = ['lsl_timestamp', 'class'] + ['TP9', 'AF7', 'AF8', 'TP10']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"ERROR: Missing required columns: {missing_cols}")
            return 1
        
        # Initialize extractor
        extractor = EEGBandExtractor()
        
        # Extract segments
        segments = extractor.extract_segments(df)
        
        if len(segments) == 0:
            print("ERROR: No valid segments found")
            return 1
        
        # Compute band powers
        df_features = extractor.compute_band_powers(segments)
        
        # Apply baseline drift correction
        df_corrected = extractor.apply_baseline_drift_correction(df_features)
        
        # Normalize features
        df_final = extractor.normalize_features(df_corrected)
        
        # Save results
        output_file = input_file.parent / f"{input_file.stem}_bands.csv"
        df_final.to_csv(output_file, index=False)
        print(f"\nBand features saved: {output_file}")
        
        # Generate visualizations if requested
        if args.tsne:
            tsne_file = input_file.parent / f"{input_file.stem}_bands_tsne.png"
            extractor.create_tsne_visualization(df_final, tsne_file)
        
        if args.stats:
            stats_file = input_file.parent / f"{input_file.stem}_bands_stats.png"
            extractor.create_statistical_comparison(df_final, stats_file)
        
        print(f"\n{'='*60}")
        print("BAND EXTRACTION COMPLETE")
        print("="*60)
        print(f"Input samples: {len(df)}")
        print(f"Output segments: {len(df_final)}")
        print(f"Features per segment: {len(extractor.feature_names)}")
        
        return 0
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
