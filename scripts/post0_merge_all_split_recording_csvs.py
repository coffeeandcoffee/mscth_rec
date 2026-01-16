#!/usr/bin/env python3
"""
Post0: Merge Split Recording CSVs

Merges all split recording CSVs (from headset reconnections) into a single file.

Process:
1. Find all recording folders with split CSVs (basename.csv + basename_2.csv, basename_3.csv, etc.)
2. Rename basename.csv → basename_1.csv (so merged file can take its place)
3. Concatenate all numbered files in order → basename.csv

Usage:
    python post0_merge_all_split_recording_csvs.py           # Process latest folder
    python post0_merge_all_split_recording_csvs.py --all     # Process all folders
    python post0_merge_all_split_recording_csvs.py --folder eeg_20251224_164549  # Specific folder
"""

import argparse
import pandas as pd
import re
from pathlib import Path


def find_recording_folders(recordings_dir=None):
    """Find all recording folders."""
    if recordings_dir is None:
        recordings_dir = Path(__file__).resolve().parent.parent / "recordings"
    recordings_path = Path(recordings_dir)
    
    if not recordings_path.exists():
        raise FileNotFoundError(f"Recordings directory '{recordings_dir}' not found")
    
    # Find all eeg_* folders
    folders = [f for f in recordings_path.iterdir() if f.is_dir() and f.name.startswith('eeg_')]
    return sorted(folders, key=lambda f: f.stat().st_mtime)


def get_split_files(folder):
    """
    Get all split CSV files in a folder.
    Returns (basename, list of numbered files) or None if no splits.
    
    Example: 
        folder contains: eeg_20251224_164549.csv, eeg_20251224_164549_2.csv, eeg_20251224_164549_3.csv
        returns: ('eeg_20251224_164549', [Path(..._2.csv), Path(..._3.csv)])
    """
    csv_files = list(folder.glob("*.csv"))
    
    # Exclude already processed files
    exclude_markers = ['_classified', '_preprocessed', '_bands', '_cut', '_skip', '_ml', '_merged']
    csv_files = [f for f in csv_files if not any(m in f.name for m in exclude_markers)]
    
    if len(csv_files) < 2:
        return None  # No splits to merge
    
    # Find the base file (no _N suffix)
    # Pattern: eeg_YYYYMMDD_HHMMSS.csv (no _2, _3, etc.)
    base_pattern = re.compile(r'^(eeg_\d{8}_\d{6})\.csv$')
    numbered_pattern = re.compile(r'^(eeg_\d{8}_\d{6})_(\d+)\.csv$')
    
    base_file = None
    numbered_files = []
    basename = None
    
    for f in csv_files:
        base_match = base_pattern.match(f.name)
        numbered_match = numbered_pattern.match(f.name)
        
        if base_match:
            base_file = f
            basename = base_match.group(1)
        elif numbered_match:
            numbered_files.append((int(numbered_match.group(2)), f))
    
    if base_file is None:
        return None  # No base file found (might already be merged)
    
    if len(numbered_files) == 0:
        return None  # No split files, nothing to merge
    
    # Sort numbered files by number
    numbered_files.sort(key=lambda x: x[0])
    
    return {
        'basename': basename,
        'base_file': base_file,
        'numbered_files': [f for _, f in numbered_files]
    }


def merge_folder(folder, dry_run=False):
    """
    Merge split CSVs in a folder.
    
    1. Rename basename.csv → basename_1.csv
    2. Concatenate all _N files in order → basename.csv
    """
    split_info = get_split_files(folder)
    
    if split_info is None:
        print(f"   ⏭ No splits to merge in {folder.name}")
        return False
    
    basename = split_info['basename']
    base_file = split_info['base_file']
    numbered_files = split_info['numbered_files']
    
    print(f"\n   📁 {folder.name}")
    print(f"      Base file: {base_file.name}")
    print(f"      Split files: {[f.name for f in numbered_files]}")
    
    # Step 1: Rename base file to _1
    new_base_name = f"{basename}_1.csv"
    new_base_path = folder / new_base_name
    
    if new_base_path.exists():
        print(f"      ⚠ {new_base_name} already exists - skipping (already processed?)")
        return False
    
    if dry_run:
        print(f"      [DRY RUN] Would rename {base_file.name} → {new_base_name}")
        print(f"      [DRY RUN] Would merge {len(numbered_files) + 1} files → {basename}.csv")
        return True
    
    print(f"      Renaming {base_file.name} → {new_base_name}")
    base_file.rename(new_base_path)
    
    # Step 2: Load and concatenate all files in order
    all_files = [new_base_path] + numbered_files
    print(f"      Merging {len(all_files)} files...")
    
    dfs = []
    total_rows = 0
    for f in all_files:
        df = pd.read_csv(f)
        dfs.append(df)
        print(f"         {f.name}: {len(df):,} rows")
        total_rows += len(df)
    
    merged_df = pd.concat(dfs, ignore_index=True)
    
    # Verify row count
    assert len(merged_df) == total_rows, f"Row count mismatch: {len(merged_df)} != {total_rows}"
    
    # Step 3: Save merged file
    output_path = folder / f"{basename}.csv"
    merged_df.to_csv(output_path, index=False)
    
    print(f"      ✓ Saved: {output_path.name} ({len(merged_df):,} rows)")
    
    return True


def main():
    parser = argparse.ArgumentParser(description='Merge split recording CSVs')
    parser.add_argument('--folder', '-f', type=str, help='Specific folder name (e.g., eeg_20251224_164549)')
    parser.add_argument('--all', '-a', action='store_true', help='Process all folders')
    parser.add_argument('--dry-run', '-d', action='store_true', help='Show what would be done without doing it')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("POST0: MERGE SPLIT RECORDING CSVs")
    print("=" * 60)
    
    if args.dry_run:
        print("🔍 DRY RUN MODE - no files will be modified")
    
    try:
        recordings_dir = Path(__file__).resolve().parent.parent / "recordings"
        
        if args.folder:
            # Process specific folder
            folder = recordings_dir / args.folder
            if not folder.exists():
                raise FileNotFoundError(f"Folder not found: {folder}")
            folders = [folder]
        elif args.all:
            # Process all folders
            folders = find_recording_folders(recordings_dir)
            print(f"\nFound {len(folders)} recording folders")
        else:
            # Process latest folder only
            folders = find_recording_folders(recordings_dir)
            if folders:
                folders = [folders[-1]]  # Most recent
                print(f"\nProcessing latest folder: {folders[0].name}")
            else:
                print("No recording folders found")
                return 1
        
        merged_count = 0
        for folder in folders:
            if merge_folder(folder, dry_run=args.dry_run):
                merged_count += 1
        
        print(f"\n{'='*60}")
        if args.dry_run:
            print(f"DRY RUN: Would merge {merged_count} folder(s)")
        else:
            print(f"✓ Merged {merged_count} folder(s)")
        
        return 0
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
