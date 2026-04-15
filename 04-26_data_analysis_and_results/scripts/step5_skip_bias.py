#!/usr/bin/env python3
"""
Step 5: Skip Behavior Bias Analysis

Quantifies user behavioral autocorrelation: How many sequential skip events 
occur uninterrupted within chained "SKIP" periods?
"""

import argparse
import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime

def analyze_bias(df):
    """
    Counts how many consecutive keypress_A events map inside contiguous SKIP blocks (engagement_state == 0).
    """
    is_skip = df['engagement_state'] == 0
    block_changes = is_skip.astype(int).diff().fillna(0)
    
    block_starts = df.index[block_changes == 1].tolist()
    block_ends = df.index[block_changes == -1].tolist()
    
    if is_skip.iloc[0]: block_starts.insert(0, df.index[0])
    if is_skip.iloc[-1]: block_ends.append(df.index[-1] + 1)
    
    n_blocks = min(len(block_starts), len(block_ends))
    
    skip_counts = []
    for i in range(n_blocks):
        block_data = df.loc[block_starts[i]:block_ends[i]-1]
        k_count = int(block_data['keypress_A'].sum())
        if k_count > 0:
            skip_counts.append(k_count)
            
    if not skip_counts:
        return None
        
    mode_val = int(max(set(skip_counts), key=skip_counts.count))
    return {
        'n_blocks': len(skip_counts),
        'total_skips': sum(skip_counts),
        'mean': float(np.mean(skip_counts)),
        'mode': mode_val,
        'mode_pct': 100 * skip_counts.count(mode_val) / len(skip_counts),
        'distribution': {str(k): skip_counts.count(k) for k in sorted(set(skip_counts))}
    }

def main():
    out_dir = Path(__file__).resolve().parent.parent.parent / "04-26_data_analysis_and_results" / "outputs" / f"skip_bias_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    data_dir = Path(__file__).resolve().parent.parent.parent / "04-26_data_analysis_and_results" / "processed_data"
    files = list(data_dir.glob("P*_labeled.csv"))
    
    print("=" * 60)
    print("STEP 5: SKIP BEHAVIOR BIAS")
    print("=" * 60)
    
    results = {}
    for f in files:
        pid = f.stem.split('_')[0]
        df = pd.read_csv(f)
        stats = analyze_bias(df)
        if stats:
            results[pid] = stats
            print(f"P{pid} -> Mean skips/block: {stats['mean']:.1f} | Mode: {stats['mode']} ({stats['mode_pct']:.0f}%)")
            
    with open(out_dir/'bias_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Outputs saved to {out_dir.name}")

if __name__ == "__main__":
    main()
