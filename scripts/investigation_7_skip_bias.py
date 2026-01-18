#!/usr/bin/env python3
"""
Investigation V7: Skip Behavior Bias Analysis

Investigates user skip patterns to understand behavioral biases:
- How many videos does each participant skip in a row before stopping?
- Is there a consistent "skip block length" preference?

Method:
1. Identify contiguous "about_to_skip" blocks in the data
2. Count keypress_A events within each block
3. Create distribution histogram showing skip block lengths

Usage:
    python scripts/investigation_7_skip_bias.py --file <skip_labels.csv>
    python scripts/investigation_7_skip_bias.py  # Uses most recent file
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import json

import warnings
warnings.filterwarnings('ignore')


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
    
    required = ['keypress_A', 'classification_2']
    missing = [c for c in required if c not in df.columns]
    assert len(missing) == 0, f"Missing columns: {missing}"
    
    print(f"   ✓ Required columns present")
    print(f"   Total rows: {len(df):,}")
    print(f"   Total keypress_A=1 events: {int(df['keypress_A'].sum()):,}")
    
    return df


# ============================================================================
# STEP 2: IDENTIFY SKIP BLOCKS AND COUNT KEYPRESSES
# ============================================================================

def identify_skip_blocks_and_count(df):
    """
    Identify contiguous 'about_to_skip' blocks and count keypresses within each.
    
    A "skip block" is a contiguous sequence of rows where classification_2 == 'about_to_skip'.
    Within each block, we count how many keypress_A=1 events occur.
    
    Returns:
        list of int: Number of keypresses in each skip block
    """
    skip_counts = []
    
    # Create a boolean mask for about_to_skip
    is_skip = df['classification_2'] == 'about_to_skip'
    
    # Find block boundaries using diff
    block_changes = is_skip.astype(int).diff().fillna(0)
    
    # Get indices where blocks start (change from 0 to 1)
    block_starts = df.index[block_changes == 1].tolist()
    
    # Get indices where blocks end (change from 1 to 0)
    block_ends = df.index[block_changes == -1].tolist()
    
    # Handle edge cases
    if is_skip.iloc[0]:
        block_starts.insert(0, df.index[0])
    if is_skip.iloc[-1]:
        block_ends.append(df.index[-1] + 1)
    
    # Ensure we have matching pairs
    n_blocks = min(len(block_starts), len(block_ends))
    
    print(f"   Found {n_blocks} skip blocks total")
    
    # Count keypresses in each block
    for i in range(n_blocks):
        start_idx = block_starts[i]
        end_idx = block_ends[i]
        
        block_data = df.loc[start_idx:end_idx-1]
        keypress_count = int(block_data['keypress_A'].sum())
        
        if keypress_count > 0:  # Only count blocks with at least 1 skip
            skip_counts.append(keypress_count)
    
    print(f"   Blocks with ≥1 keypress: {len(skip_counts)}")
    
    return skip_counts


# ============================================================================
# STEP 3: STATISTICS
# ============================================================================

def compute_statistics(skip_counts):
    """Compute distribution statistics."""
    if not skip_counts:
        return {}
    
    return {
        'n_blocks': len(skip_counts),
        'total_skips': sum(skip_counts),
        'mean': float(np.mean(skip_counts)),
        'median': float(np.median(skip_counts)),
        'mode': int(max(set(skip_counts), key=skip_counts.count)),
        'mode_count': skip_counts.count(max(set(skip_counts), key=skip_counts.count)),
        'mode_pct': 100 * skip_counts.count(max(set(skip_counts), key=skip_counts.count)) / len(skip_counts),
        'min': int(min(skip_counts)),
        'max': int(max(skip_counts)),
        'std': float(np.std(skip_counts)),
        'distribution': {str(k): skip_counts.count(k) for k in sorted(set(skip_counts))}
    }


# ============================================================================
# STEP 4: VISUALIZATION
# ============================================================================

def plot_skip_distribution(skip_counts, stats, output_dir):
    """Plot histogram of skip block lengths."""
    
    if not skip_counts:
        print("   ⚠ No skip blocks found, skipping plot")
        return
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Create histogram
    max_count = max(skip_counts)
    bins = np.arange(0.5, max_count + 1.5, 1)  # Centers at 1, 2, 3, etc.
    
    counts, edges, patches = ax.hist(skip_counts, bins=bins, 
                                      edgecolor='black', alpha=0.7, color='steelblue')
    
    # Add count labels on bars
    for count, patch in zip(counts, patches):
        if count > 0:
            x = patch.get_x() + patch.get_width() / 2
            y = patch.get_height()
            ax.annotate(f'{int(count)}', xy=(x, y), ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Add vertical lines for mean and median
    ax.axvline(stats['mean'], color='red', linestyle='--', linewidth=2.5, 
               label=f'Mean: {stats["mean"]:.1f}')
    ax.axvline(stats['median'], color='green', linestyle=':', linewidth=2.5, 
               label=f'Median: {stats["median"]:.1f}')
    
    # Styling
    ax.set_xlabel('Number of Videos Skipped in a Row (per block)', fontsize=12)
    ax.set_ylabel('Frequency (# of blocks)', fontsize=12)
    ax.set_title(f'Skip Behavior Bias: Distribution of Consecutive Skips\n'
                 f'n={stats["n_blocks"]} blocks | Mode={stats["mode"]} ({stats["mode_pct"]:.0f}% of blocks)', 
                 fontsize=13, fontweight='bold')
    ax.legend(loc='upper right', fontsize=11)
    ax.set_xticks(range(1, max_count + 1))
    ax.set_xlim(0.5, max_count + 0.5)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    output_path = output_dir / 'skip_bias_distribution.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"   ✓ Distribution plot saved")


def plot_cumulative_distribution(skip_counts, stats, output_dir):
    """Plot cumulative distribution function."""
    
    if not skip_counts:
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    sorted_counts = np.sort(skip_counts)
    cumulative = np.arange(1, len(sorted_counts) + 1) / len(sorted_counts)
    
    ax.plot(sorted_counts, cumulative, 'o-', color='steelblue', linewidth=2, markersize=6)
    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5, label='50th percentile')
    
    ax.set_xlabel('Number of Videos Skipped in a Row', fontsize=12)
    ax.set_ylabel('Cumulative Probability', fontsize=12)
    ax.set_title('Cumulative Distribution: Skip Block Lengths', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'skip_bias_cumulative.png', dpi=150)
    plt.close()
    
    print(f"   ✓ Cumulative distribution plot saved")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Investigation V7: Skip Behavior Bias Analysis')
    parser.add_argument('--file', '-f', type=str, help='Specific skip labels file')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("INVESTIGATION V7: SKIP BEHAVIOR BIAS ANALYSIS")
    print("=" * 60)
    
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
        
        # STEP 2: Identify skip blocks
        print(f"\n{'='*60}")
        print("STEP 2: IDENTIFY SKIP BLOCKS")
        print("=" * 60)
        
        skip_counts = identify_skip_blocks_and_count(df)
        
        if not skip_counts:
            print("   ❌ No skip blocks with keypresses found!")
            return 1
        
        # STEP 3: Compute statistics
        print(f"\n{'='*60}")
        print("STEP 3: COMPUTE STATISTICS")
        print("=" * 60)
        
        stats = compute_statistics(skip_counts)
        
        print(f"   Mean skips per block: {stats['mean']:.1f}")
        print(f"   Median: {stats['median']:.1f}")
        print(f"   Mode: {stats['mode']} ({stats['mode_pct']:.0f}% of blocks)")
        print(f"   Range: {stats['min']} - {stats['max']}")
        print(f"   Std Dev: {stats['std']:.2f}")
        
        # STEP 4: Save results
        print(f"\n{'='*60}")
        print("STEP 4: SAVE RESULTS")
        print("=" * 60)
        
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = input_file.parent / f"model_output_investigation_v7_BIAS_{timestamp_str}"
        output_dir.mkdir(exist_ok=True)
        
        # Generate plots
        plot_skip_distribution(skip_counts, stats, output_dir)
        plot_cumulative_distribution(skip_counts, stats, output_dir)
        
        # Save results JSON
        results_data = {
            'investigation': 'V7_Skip_Behavior_Bias',
            'timestamp': datetime.now().isoformat(),
            'input_file': str(input_file),
            'statistics': stats,
            'raw_skip_counts': skip_counts,
            'interpretation': f"User typically skips {stats['mode']} videos in a row ({stats['mode_pct']:.0f}% of skip blocks)"
        }
        
        with open(output_dir / 'skip_bias_results.json', 'w') as f:
            json.dump(results_data, f, indent=2)
        print(f"   ✓ Results JSON saved")
        
        # Save summary text
        with open(output_dir / 'skip_bias_summary.txt', 'w') as f:
            f.write("SKIP BEHAVIOR BIAS ANALYSIS\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Input: {input_file.name}\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"STATISTICS:\n")
            f.write(f"  Total skip blocks: {stats['n_blocks']}\n")
            f.write(f"  Total skips: {stats['total_skips']}\n")
            f.write(f"  Mean skips/block: {stats['mean']:.2f}\n")
            f.write(f"  Median: {stats['median']:.1f}\n")
            f.write(f"  Mode: {stats['mode']} ({stats['mode_pct']:.0f}% of blocks)\n")
            f.write(f"  Range: {stats['min']} - {stats['max']}\n")
            f.write(f"  Std Dev: {stats['std']:.2f}\n\n")
            f.write(f"DISTRIBUTION:\n")
            for k, v in stats['distribution'].items():
                pct = 100 * v / stats['n_blocks']
                f.write(f"  {k} skips: {v} blocks ({pct:.1f}%)\n")
            f.write(f"\nKEY FINDING:\n")
            f.write(f"  User typically skips {stats['mode']} videos in a row.\n")
        
        print(f"   ✓ Summary text saved")
        
        # Print key finding
        print(f"\n{'='*60}")
        print("KEY FINDING")
        print("=" * 60)
        print(f"   User typically skips {stats['mode']} videos in a row")
        print(f"   ({stats['mode_pct']:.0f}% of {stats['n_blocks']} skip blocks)")
        
        print(f"\n{'='*60}")
        print("ANALYSIS COMPLETE")
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
