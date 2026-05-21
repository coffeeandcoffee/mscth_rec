#!/usr/bin/env python3
"""
step04_balance_split.py — Temporal blocked CV with undersampling.

Runs on primary manifest only. Creates temporal 5-fold CV blocks with 3s gap.
Test folds: zero overlap. Train folds: 80% overlap (already in window data).
Undersamples STAY to 50/50 per seed per fold.

OUT: balanced split manifests per participant per seed + balanced_counts.csv
"""

import numpy as np
import pickle
from pathlib import Path
import pandas as pd

import config


def create_smart_temporal_blocks(windows, n_folds=5, gap_s=3.0):
    """Divide windows into n_folds stratified temporal blocks with strict firewalls.
    Uses the Smart Stratified Temporal Split algorithm."""
    if not windows:
        return [[] for _ in range(n_folds)]

    # Sort windows by start_time
    sorted_wins = sorted(enumerate(windows), key=lambda x: x[1]['start_time'])
    times = np.array([w['start_time'] for _, w in sorted_wins])
    ids = [idx for idx, _ in sorted_wins]

    # Chronological timestamps of the minority class (SKIP)
    skip_times = [w['start_time'] for _, w in sorted_wins if w['label'] == 0]
    
    split_times = []
    if len(skip_times) >= n_folds:
        # Place splits at the exact percentiles of the SKIP behavior distribution
        for k in range(1, n_folds):
            idx = int(len(skip_times) * k / n_folds)
            split_times.append(skip_times[idx])
    else:
        # Fallback to simple time-based if there are almost no skips
        t_min, t_max = times[0], times[-1]
        split_times = [t_min + (t_max - t_min) * k / n_folds for k in range(1, n_folds)]

    blocks = [[] for _ in range(n_folds)]
    
    for wid, t in zip(ids, times):
        # Check if window falls in the strict deletion gap
        in_gap = False
        for st in split_times:
            if st - gap_s/2 <= t <= st + gap_s/2:
                in_gap = True
                break
        
        if in_gap:
            continue
            
        # Assign to the correct temporal block
        block_idx = 0
        for st in split_times:
            if t > st + gap_s/2:
                block_idx += 1
            else:
                break
                
        blocks[block_idx].append(wid)

    return blocks


def undersample_balance(window_ids, windows, seed):
    """Undersample majority class to achieve 50/50 balance."""
    rng = np.random.RandomState(seed)

    stay_ids = [wid for wid in window_ids if windows[wid]['label'] == 1]
    skip_ids = [wid for wid in window_ids if windows[wid]['label'] == 0]

    n_min = min(len(stay_ids), len(skip_ids))
    if n_min == 0:
        return window_ids  # Can't balance

    if len(stay_ids) > n_min:
        stay_ids = rng.choice(stay_ids, size=n_min, replace=False).tolist()
    if len(skip_ids) > n_min:
        skip_ids = rng.choice(skip_ids, size=n_min, replace=False).tolist()

    balanced = stay_ids + skip_ids
    rng.shuffle(balanced)
    return balanced


def validate_splits(seed_splits, windows, gap_s, info=""):
    """Mathematically validates that splits maintain the temporal firewall."""
    for fold in seed_splits:
        train_ids = set(fold['train_ids'])
        test_ids = set(fold['test_ids'])
        
        # 1. Absolute Independence
        overlap = train_ids.intersection(test_ids)
        assert len(overlap) == 0, f"Validation Failed: {len(overlap)} windows overlap between train and test in fold {fold['fold']} {info}"
        
        # 2. Temporal Distance Proof
        if not train_ids or not test_ids:
            continue
            
        train_times = np.array([windows[i]['start_time'] for i in train_ids])
        test_times = np.array([windows[i]['start_time'] for i in test_ids])
        
        train_times.sort()
        test_times.sort()
        
        min_dist = float('inf')
        i, j = 0, 0
        while i < len(train_times) and j < len(test_times):
            dist = abs(train_times[i] - test_times[j])
            if dist < min_dist:
                min_dist = dist
            if train_times[i] < test_times[j]:
                i += 1
            else:
                j += 1
                
        # 3. Assert distance >= gap_s - 0.1
        assert min_dist >= gap_s - 0.1, f"Validation Failed: Temporal firewall breached! min_dist={min_dist:.3f}s < {gap_s}s in fold {fold['fold']} {info}"


def build_cv_splits(windows, params, pid=None):
    """Build temporal blocked CV splits for all seeds."""
    p = params.get('step04', {})
    seeds = p.get('seeds', [0, 1, 7, 42, 99])
    n_folds = p.get('n_folds', 5)
    gap_s = p.get('gap_s', 3.0)

    blocks = create_smart_temporal_blocks(windows, n_folds, gap_s)

    all_splits = {}

    for seed in seeds:
        seed_splits = []
        for fold in range(n_folds):
            test_ids = blocks[fold]
            train_ids = []
            for k in range(n_folds):
                if k != fold:
                    train_ids.extend(blocks[k])

            # Balance independently
            train_balanced = undersample_balance(train_ids, windows, seed + fold)
            test_balanced = undersample_balance(test_ids, windows, seed + fold + 100)

            seed_splits.append({
                'fold': fold,
                'train_ids': train_balanced,
                'test_ids': test_balanced,
                'train_n_stay': sum(1 for i in train_balanced if windows[i]['label'] == 1),
                'train_n_skip': sum(1 for i in train_balanced if windows[i]['label'] == 0),
                'test_n_stay': sum(1 for i in test_balanced if windows[i]['label'] == 1),
                'test_n_skip': sum(1 for i in test_balanced if windows[i]['label'] == 0),
            })

        info = f"(PID {pid}, Seed {seed})" if pid is not None else f"(Seed {seed})"
        validate_splits(seed_splits, windows, gap_s, info)

        all_splits[seed] = seed_splits

    return all_splits


def run(run_dir, params):
    """Entry point called by run.py."""
    config.pprint_step(4, "BALANCE & TEMPORAL SPLIT")

    windows_dir = run_dir / "windows" / "primary"
    splits_dir = run_dir / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)

    balanced_rows = []

    for pid in config.INCLUDED_PARTICIPANTS:
        pkl_path = windows_dir / f"P{pid}.pkl"
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)

        windows = data['windows']
        all_splits = build_cv_splits(windows, params, pid=pid)

        # Save splits
        split_path = splits_dir / f"P{pid}_splits.pkl"
        with open(split_path, 'wb') as f:
            pickle.dump({
                'pid': pid,
                'splits': all_splits,
            }, f)

        # Compute balanced counts (use seed 0 as reference)
        seeds = params.get('step04', {}).get('seeds', [0, 1, 7, 42, 99])
        test_counts_per_seed = []
        for seed in seeds:
            total_test = sum(len(fold['test_ids']) for fold in all_splits[seed])
            test_counts_per_seed.append(total_test)

        mean_test_n = int(np.mean(test_counts_per_seed))
        # Total balanced windows across all folds for one seed (= total test windows)
        ref_splits = all_splits[seeds[0]]
        total_balanced = sum(len(f['test_ids']) for f in ref_splits)

        balanced_rows.append({
            'pid': pid,
            'n_windows_raw': len(windows),
            'n_stay_raw': sum(1 for w in windows if w['label'] == 1),
            'n_skip_raw': sum(1 for w in windows if w['label'] == 0),
            'n_balanced_test_total': total_balanced,
            'mean_test_n_per_seed': mean_test_n,
        })

        print(f"  P{pid}: raw={len(windows)}, "
              f"balanced_test={total_balanced}, "
              f"folds OK")

    df_counts = pd.DataFrame(balanced_rows)
    df_counts.to_csv(run_dir / "balanced_counts.csv", index=False)
    print(f"\n  ✓ Balanced counts saved. This N is used for binomial CIs in step09.")


if __name__ == "__main__":
    print("Use run.py to execute the pipeline.")
