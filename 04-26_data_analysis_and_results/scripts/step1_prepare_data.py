#!/usr/bin/env python3
"""
Step 1: Data Preparation for Case B (STAY vs. SKIP)

Merges raw CSV parts, classifies baselines and viewing blocks,
and applies the rigorous STAY=1, SKIP=0 label mapping.
"""

import os
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import re

# Final 25 Included Participants
INCLUDED_PARTICIPANTS = [p for p in range(4, 32) if p not in [16, 19, 29]]

def get_participant_files(data_dir, pid):
    """Find and sort all P{pid}_*.csv files."""
    pattern = re.compile(rf'^P{pid}_(\d+)\.csv$')
    files = []
    
    for f in data_dir.glob(f"P{pid}_*.csv"):
        match = pattern.match(f.name)
        if match:
            part_num = int(match.group(1))
            files.append((part_num, f))
            
    # Sort files naturally by part number
    files.sort(key=lambda x: x[0])
    return [f for _, f in files]


def classify_segments(df):
    """
    Classify EEG segments based on keypress patterns.
    (Kept structurally identical to legacy post2v2 but applied to new df shape.)
    """
    df = df.copy()
    df['class'] = ''  # Empty string default
    
    a_presses = df[df['keypress_A'] == 1].index.tolist()
    b_presses = df[df['keypress_B'] == 1].index.tolist()
    
    print(f"      Found {len(a_presses)} 'A' keypresses and {len(b_presses)} 'B' keypresses")
    
    # Classify Baselines
    if len(b_presses) >= 2:
        df.loc[b_presses[0]:b_presses[1], 'class'] = 'baseline_1'
        if len(b_presses) >= 3:
            df.loc[b_presses[-2]:b_presses[-1], 'class'] = 'baseline_2'
            
    # Classify TikTok Viewing
    if len(a_presses) >= 2:
        for i in range(len(a_presses) - 1):
            start_idx = a_presses[i]
            end_idx = a_presses[i + 1]
            
            start_time = df.loc[start_idx, 'lsl_timestamp']
            end_time = df.loc[end_idx, 'lsl_timestamp']
            time_diff = end_time - start_time
            
            if time_diff > 4.0:
                df.loc[start_idx:end_idx, 'class'] = 'tiktok_over_4s_watched'
            else:
                df.loc[start_idx:end_idx, 'class'] = 'tiktok_under_4s_watched'
                
    return df


def add_engagement_labels(df, window_seconds=3.0):
    """
    Case B Logic: Predict Sustained Engagement (STAY=1) vs Disengagement (SKIP=0).
    """
    df = df.copy()
    timestamps = df['lsl_timestamp'].values
    keypress_A = df['keypress_A'].values
    original_class = df['class'].values
    
    # Initialize: -1 for baseline/ignored, 1 (STAY) for everything else
    engagement_state = np.full(len(df), -1, dtype=int)
    
    for i in range(len(df)):
        cls = original_class[i]
        if cls.startswith('tiktok_'):
            engagement_state[i] = 1 # Default to STAY
            
    # Apply SKIP overriding windows
    for i in range(len(df)):
        if keypress_A[i] == 1:
            current_time = timestamps[i]
            window_start = current_time - window_seconds
            
            # Look backwards and override strictly inside the 3s window
            for j in range(i - 1, -1, -1):
                if timestamps[j] < window_start:
                    break
                # Only override if it was actively classified as a viewing block
                if original_class[j].startswith('tiktok_'):
                    engagement_state[j] = 0 # Override to SKIP
                    
    df['engagement_state'] = engagement_state
    return df


def main():
    parser = argparse.ArgumentParser(description='Step 1: Parse and map data for STAY prediction.')
    parser.add_argument('--window', type=float, default=3.0, help='Window duration before skips.')
    args = parser.parse_args()
    
    project_root = Path(__file__).resolve().parent.parent.parent
    data_dir = project_root / "data"
    output_dir = project_root / "04-26_data_analysis_and_results" / "processed_data"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("STEP 1: MERGE & LABEL PREPARATION (STAY vs SKIP)")
    print("=" * 60)
    
    total_samples = 0
    total_stay = 0
    total_skip = 0
    
    for pid in INCLUDED_PARTICIPANTS:
        files = get_participant_files(data_dir, pid)
        if not files:
            print(f"⚠ Skipping P{pid} - No CSV files found.")
            continue
            
        print(f"\nProcessing P{pid} ({len(files)} parts)...")
        
        # Load and concat
        dfs = []
        for f in files:
            df_part = pd.read_csv(f)
            dfs.append(df_part)
            
        df = pd.concat(dfs, ignore_index=True)
        
        # 1. Classify
        df = classify_segments(df)
        
        # 2. Add Engagement Target (Case B)
        df = add_engagement_labels(df, window_seconds=args.window)
        
        # Tally metrics
        p_stay = np.sum(df['engagement_state'] == 1)
        p_skip = np.sum(df['engagement_state'] == 0)
        
        total_samples += len(df)
        total_stay += p_stay
        total_skip += p_skip
        
        print(f"      ✓ Mapped: {p_stay} STAY rows, {p_skip} SKIP rows")
        
        out_file = output_dir / f"P{pid}_labeled.csv"
        df.to_csv(out_file, index=False)
        print(f"      ✓ Saved: {out_file.name}")
        
    print(f"\n{'='*60}")
    print(f"COMPLETE. Total mapped samples across {len(INCLUDED_PARTICIPANTS)} participants:")
    print(f"STAY rows: {total_stay}")
    print(f"SKIP rows: {total_skip}")
    print(f"Total rows: {total_samples}")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()
