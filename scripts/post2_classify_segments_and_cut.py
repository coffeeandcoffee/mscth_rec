#!/usr/bin/env python3
"""
EEG Data Segment Classification and Cutting Script

Classifies EEG recording segments based on keypress patterns:
- baseline_1: Between first and second B press
- baseline_2: Between second-to-last and last B press  
- tiktok_over_4s_watched: Between A presses >4s apart
- tiktok_under_4s_watched: Between A presses <=4s apart

Then cuts TikTok segments to keep only first X seconds after each A keypress.

Usage:
    python post2_classify_segments_and_cut.py                    # Use latest CSV, default 0.5s cut
    python post2_classify_segments_and_cut.py --file path.csv    # Use specific CSV
    python post2_classify_segments_and_cut.py --cut-duration 3.0 # Keep first 3s instead of 0.5s
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import sys


def find_latest_eeg_file(recordings_dir=None):
    """Find the most recently created EEG CSV file (searches recursively)."""
    if recordings_dir is None:
        recordings_dir = Path(__file__).resolve().parent.parent / "recordings"
    recordings_path = Path(recordings_dir)
    if not recordings_path.exists():
        raise FileNotFoundError(f"Recordings directory '{recordings_dir}' not found")
    
    # Search recursively for eeg_*.csv files
    eeg_files = list(recordings_path.rglob("eeg_*.csv"))
    
    # Filter out files that are already processed (contain _classified, _preprocessed, etc.)
    raw_eeg_files = [f for f in eeg_files if not any(
        marker in f.name for marker in ['_classified', '_preprocessed', '_bands', '_cut']
    )]
    
    if not raw_eeg_files:
        raise FileNotFoundError(f"No raw EEG CSV files found in '{recordings_dir}'")
    
    # Sort by modification time, return most recent
    latest_file = max(raw_eeg_files, key=lambda f: f.stat().st_mtime)
    return latest_file


def classify_segments(df):
    """
    Classify EEG segments based on keypress patterns.
    
    Args:
        df: DataFrame with keypress_A and keypress_B columns
        
    Returns:
        DataFrame with added 'class' column
    """
    # Create class column filled with NaN
    df = df.copy()
    df['class'] = np.nan
    
    # Find all keypress events
    a_presses = df[df['keypress_A'] == 1].index.tolist()
    b_presses = df[df['keypress_B'] == 1].index.tolist()
    
    print(f"Found {len(a_presses)} A keypresses and {len(b_presses)} B keypresses")
    
    # Classify B press segments (baselines)
    if len(b_presses) >= 2:
        # baseline_1: Between first and second B press
        start_idx = b_presses[0]
        end_idx = b_presses[1]
        df.loc[start_idx:end_idx, 'class'] = 'baseline_1'
        print(f"baseline_1: rows {start_idx} to {end_idx}")
        
        if len(b_presses) >= 3:
            # baseline_2: Between second-to-last and last B press
            start_idx = b_presses[-2]
            end_idx = b_presses[-1]
            df.loc[start_idx:end_idx, 'class'] = 'baseline_2'
            print(f"baseline_2: rows {start_idx} to {end_idx}")
    
    # Classify A press segments (TikTok watching)
    if len(a_presses) >= 2:
        first_a = a_presses[0]
        last_a = a_presses[-1]
        
        # Only process A presses between first and last
        for i in range(len(a_presses) - 1):
            start_idx = a_presses[i]
            end_idx = a_presses[i + 1]
            
            # Calculate time difference in seconds
            start_time = df.loc[start_idx, 'lsl_timestamp']
            end_time = df.loc[end_idx, 'lsl_timestamp']
            time_diff = end_time - start_time
            
            # Classify based on duration
            if time_diff > 4.0:
                class_name = 'tiktok_over_4s_watched'
            else:
                class_name = 'tiktok_under_4s_watched'
            
            df.loc[start_idx:end_idx, 'class'] = class_name
            print(f"{class_name}: rows {start_idx} to {end_idx} (duration: {time_diff:.2f}s)")
    
    return df


def cut_tiktok_segments(df, cut_duration, fs=256):
    """
    Keep only first X seconds from each TikTok block starting with A keypress.
    Uses timestamps to ensure accurate duration cutting.
    
    Args:
        df: DataFrame with classified data
        cut_duration: Duration in seconds to keep (e.g., 0.5 or 3.0)
        fs: Sampling frequency (for display only, not used for cutting)
    
    Returns:
        DataFrame with TikTok sections cut to specified duration
    """
    print(f"\nCutting TikTok segments to first {cut_duration}s after A keypress (timestamp-based)...")
    
    # Find all A keypresses (start of TikTok videos)
    a_presses = df[df['keypress_A'] == 1].index.tolist()
    
    if len(a_presses) == 0:
        print("   No A keypresses found, skipping cutting")
        return df
    
    # Create mask for samples to keep
    keep_mask = np.ones(len(df), dtype=bool)
    
    segments_cut = 0
    
    for a_press_idx in a_presses:
        # Get the class at this A keypress
        current_class = df.loc[a_press_idx, 'class']
        
        if current_class in ['tiktok_over_4s_watched', 'tiktok_under_4s_watched']:
            # Get timestamp of A keypress
            a_press_timestamp = df.loc[a_press_idx, 'lsl_timestamp']
            max_timestamp = a_press_timestamp + cut_duration
            
            # Find the end of this TikTok section (next A keypress or end of same class)
            section_end_idx = len(df)
            
            # Look for next A keypress or class change
            for next_idx in range(a_press_idx + 1, len(df)):
                if (df.loc[next_idx, 'keypress_A'] == 1 or 
                    df.loc[next_idx, 'class'] != current_class):
                    section_end_idx = next_idx
                    break
            
            # Mark samples to keep: those with timestamp between a_press_timestamp and max_timestamp
            # AND within the TikTok section boundaries
            for idx in range(a_press_idx, section_end_idx):
                row_timestamp = df.loc[idx, 'lsl_timestamp']
                if row_timestamp > max_timestamp:
                    # This row is beyond the cut duration, mark for removal
                    keep_mask[idx] = False
            
            segments_cut += 1
            # Count how many samples were kept
            kept_samples = np.sum(keep_mask[a_press_idx:section_end_idx])
            actual_duration = max_timestamp - a_press_timestamp
            print(f"   Cut TikTok section at sample {a_press_idx} ({current_class}): "
                  f"kept {kept_samples} samples (timestamp range: {a_press_timestamp:.3f} to {max_timestamp:.3f}, duration: {actual_duration:.3f}s)")
    
    # Apply the mask to keep only desired samples
    df_cut = df[keep_mask].reset_index(drop=True)
    
    print(f"   Cut {segments_cut} TikTok segments")
    print(f"   Kept {len(df_cut)} samples (removed {len(df) - len(df_cut)} samples)")
    
    return df_cut


def main():
    parser = argparse.ArgumentParser(
        description='Classify EEG segments based on keypress patterns.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Classification Rules:
  baseline_1: Between 1st and 2nd B press
  baseline_2: Between 2nd-to-last and last B press
  tiktok_over_4s_watched: Between A presses >4s apart
  tiktok_under_4s_watched: Between A presses <=4s apart
  
After classification, TikTok segments are cut to keep only the first X seconds
after each A keypress (default: 0.5s, configurable via --cut-duration).
All other segments remain NaN.
        """
    )
    
    parser.add_argument(
        '--file', '-f',
        type=str,
        default=None,
        help='Path to specific EEG CSV file. If not provided, uses latest file in ./recordings'
    )
    
    parser.add_argument(
        '--cut-duration',
        type=float,
        default=0.5,
        help='Duration in seconds to keep from each TikTok block starting with A keypress (default: 0.5)'
    )
    
    args = parser.parse_args()
    
    try:
        # Determine input file
        if args.file:
            input_file = Path(args.file)
            if not input_file.exists():
                print(f"ERROR: File '{input_file}' not found")
                return 1
        else:
            input_file = find_latest_eeg_file()
            print(f"Using latest file: {input_file}")
        
        # Load data
        print(f"Loading data from: {input_file}")
        df = pd.read_csv(input_file)
        
        # Validate required columns
        required_cols = ['keypress_A', 'keypress_B', 'lsl_timestamp']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"ERROR: Missing required columns: {missing_cols}")
            print(f"Available columns: {list(df.columns)}")
            return 1
        
        print(f"Loaded {len(df)} samples")
        
        # Classify segments
        print("\nClassifying segments...")
        df_classified = classify_segments(df)
        
        # Cut TikTok segments to specified duration
        if args.cut_duration > 0:
            # Detect sampling rate from data
            time_diffs = np.diff(df['lsl_timestamp'].values)
            fs = 1.0 / np.median(time_diffs)
            print(f"\nDetected sampling rate: {fs:.2f} Hz")
            
            df_classified = cut_tiktok_segments(df_classified, args.cut_duration, fs=fs)
        else:
            print("\nSkipping cutting (cut-duration = 0)")
        
        # Generate output filename
        if args.cut_duration != 0.5:
            output_file = input_file.parent / f"{input_file.stem}_classified_cut{args.cut_duration}s.csv"
        else:
            output_file = input_file.parent / f"{input_file.stem}_classified.csv"
        
        # Save classified data
        df_classified.to_csv(output_file, index=False)
        print(f"\nClassified and cut data saved to: {output_file}")
        
        # Summary statistics
        class_counts = df_classified['class'].value_counts(dropna=False)
        print(f"\nClassification Summary:")
        for class_name, count in class_counts.items():
            if pd.isna(class_name):
                print(f"  NaN (unclassified): {count} samples")
            else:
                print(f"  {class_name}: {count} samples")
        
        return 0
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
