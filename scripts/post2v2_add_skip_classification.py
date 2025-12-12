#!/usr/bin/env python3
"""
Post2v2: Classify EEG Data + Add Skip Prediction Labels

Standalone script that takes RAW EEG CSV and:
1. Classifies segments (baseline_1, baseline_2, tiktok_over_4s, tiktok_under_4s)
2. DOES NOT CUT - keeps all data for skip prediction
3. Adds classification_2 for skip prediction (about_to_skip, not_about_to_skip)

Skip prediction logic (row by row):
- Default: 'not_about_to_skip' for all non-baseline rows
- When keypress_A=1: look BACK at all rows within window, set to 'about_to_skip'

Usage:
    python post2v2_add_skip_classification.py              # Default 3s window
    python post2v2_add_skip_classification.py --window 5.0 # Custom window
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path


def find_latest_raw_eeg_file(recordings_dir=None):
    """Find the most recently created RAW EEG CSV file."""
    if recordings_dir is None:
        recordings_dir = Path(__file__).resolve().parent.parent / "recordings"
    recordings_path = Path(recordings_dir)
    
    if not recordings_path.exists():
        raise FileNotFoundError(f"Recordings directory '{recordings_dir}' not found")
    
    # Search for eeg_*.csv files, excluding processed ones
    eeg_files = list(recordings_path.rglob("eeg_*.csv"))
    raw_eeg_files = [f for f in eeg_files if not any(
        marker in f.name for marker in ['_classified', '_preprocessed', '_bands', '_cut', '_skip', '_ml']
    )]
    
    if not raw_eeg_files:
        raise FileNotFoundError(f"No raw EEG CSV files found in '{recordings_dir}'")
    
    return max(raw_eeg_files, key=lambda f: f.stat().st_mtime)


def classify_segments(df):
    """
    Classify EEG segments based on keypress patterns.
    Same logic as post2_classify_segments_and_cut.py BUT NO CUTTING.
    
    Returns DataFrame with 'class' column.
    """
    df = df.copy()
    df['class'] = ''  # Empty string instead of NaN for consistency
    
    a_presses = df[df['keypress_A'] == 1].index.tolist()
    b_presses = df[df['keypress_B'] == 1].index.tolist()
    
    print(f"   Found {len(a_presses)} A keypresses and {len(b_presses)} B keypresses")
    
    # Classify B press segments (baselines)
    if len(b_presses) >= 2:
        # baseline_1: Between first and second B press
        start_idx = b_presses[0]
        end_idx = b_presses[1]
        df.loc[start_idx:end_idx, 'class'] = 'baseline_1'
        
        if len(b_presses) >= 3:
            # baseline_2: Between second-to-last and last B press
            start_idx = b_presses[-2]
            end_idx = b_presses[-1]
            df.loc[start_idx:end_idx, 'class'] = 'baseline_2'
    
    # Classify A press segments (TikTok watching)
    if len(a_presses) >= 2:
        for i in range(len(a_presses) - 1):
            start_idx = a_presses[i]
            end_idx = a_presses[i + 1]
            
            start_time = df.loc[start_idx, 'lsl_timestamp']
            end_time = df.loc[end_idx, 'lsl_timestamp']
            time_diff = end_time - start_time
            
            if time_diff > 4.0:
                class_name = 'tiktok_over_4s_watched'
            else:
                class_name = 'tiktok_under_4s_watched'
            
            df.loc[start_idx:end_idx, 'class'] = class_name
    
    return df


def add_skip_classification(df, window_seconds=3.0):
    """
    Add classification_2 column for skip prediction.
    
    Logic (row by row):
    1. Default: 'not_about_to_skip' for all non-baseline rows
    2. When keypress_A=1: look BACK, set rows within window to 'about_to_skip'
    """
    df = df.copy()
    timestamps = df['lsl_timestamp'].values
    keypress_A = df['keypress_A'].values
    original_class = df['class'].values
    
    # Initialize: baselines keep their class, others get 'not_about_to_skip'
    classification_2 = np.array([
        original_class[i] if original_class[i] in ['baseline_1', 'baseline_2'] 
        else 'not_about_to_skip' 
        for i in range(len(df))
    ], dtype=object)
    
    # Find keypress_A events and look back
    for i in range(len(df)):
        if keypress_A[i] == 1:
            current_time = timestamps[i]
            window_start = current_time - window_seconds
            
            # Look back and set rows within window to 'about_to_skip'
            for j in range(i - 1, -1, -1):
                if timestamps[j] < window_start:
                    break
                # Only override TikTok rows, not baselines
                if classification_2[j] not in ['baseline_1', 'baseline_2']:
                    classification_2[j] = 'about_to_skip'
    
    df['classification_2'] = classification_2
    return df


def main():
    parser = argparse.ArgumentParser(description='Classify EEG + Add Skip Prediction Labels')
    parser.add_argument('--file', '-f', type=str, help='Specific RAW EEG CSV file')
    parser.add_argument('--window', '-w', type=float, default=3.0,
                       help='Window (seconds) before skip (default: 3.0)')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("POST2V2: CLASSIFY + SKIP LABELS (NO CUTTING)")
    print("=" * 60)
    
    try:
        # Load RAW data
        if args.file:
            input_file = Path(args.file)
        else:
            input_file = find_latest_raw_eeg_file()
        
        print(f"\n1. LOADING RAW DATA")
        print(f"   File: {input_file.name}")
        
        df = pd.read_csv(input_file)
        print(f"   Total samples: {len(df)}")
        
        # Validate required columns
        required = ['keypress_A', 'keypress_B', 'lsl_timestamp']
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")
        
        # Step 1: Classify segments (NO CUTTING)
        print(f"\n2. CLASSIFYING SEGMENTS (NO CUTTING)")
        df_classified = classify_segments(df)
        
        class_counts = df_classified['class'].value_counts()
        print(f"   Results:")
        for cls, count in class_counts.items():
            if cls:  # Skip empty string
                print(f"     {cls}: {count}")
        
        # Step 2: Add skip classification
        print(f"\n3. ADDING CLASSIFICATION_2 (window={args.window}s)")
        df_labeled = add_skip_classification(df_classified, window_seconds=args.window)
        
        class2_counts = df_labeled['classification_2'].value_counts()
        print(f"   Results:")
        for cls, count in class2_counts.items():
            print(f"     {cls}: {count} ({100*count/len(df):.1f}%)")
        
        # Save
        window_str = f"{args.window:.1f}s".replace('.', '_')
        output_file = input_file.parent / f"{input_file.stem}_skip_labels_{window_str}.csv"
        df_labeled.to_csv(output_file, index=False)
        
        print(f"\n4. OUTPUT")
        print(f"   ✓ Saved: {output_file}")
        
        # Summary for training
        about_skip = class2_counts.get('about_to_skip', 0)
        not_skip = class2_counts.get('not_about_to_skip', 0)
        total_trainable = about_skip + not_skip
        
        print(f"\n   Trainable samples: {total_trainable}")
        print(f"   - about_to_skip: {about_skip} ({100*about_skip/total_trainable:.1f}%)")
        print(f"   - not_about_to_skip: {not_skip} ({100*not_skip/total_trainable:.1f}%)")
        
        return 0
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
