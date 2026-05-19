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


def create_temporal_blocks(windows, n_folds=5, gap_s=3.0):
    """Divide windows into n_folds temporal blocks with gaps."""
    if not windows:
        return [[] for _ in range(n_folds)]

    # Sort windows by start_time
    sorted_wins = sorted(enumerate(windows), key=lambda x: x[1]['start_time'])
    times = np.array([w['start_time'] for _, w in sorted_wins])
    ids = [idx for idx, _ in sorted_wins]

    t_min, t_max = times[0], times[-1] + (windows[0].get('end_time', 0) - windows[0].get('start_time', 0))
    total_dur = t_max - t_min
    block_dur = (total_dur - gap_s * (n_folds - 1)) / n_folds

    if block_dur < 1.0:
        # Session too short — put everything in one block
        return [ids] + [[] for _ in range(n_folds - 1)]

    blocks = []
    for k in range(n_folds):
        b_start = t_min + k * (block_dur + gap_s)
        b_end = b_start + block_dur
        block_ids = [wid for wid, t in zip(ids, times) if b_start <= t < b_end]
        blocks.append(block_ids)

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


def build_cv_splits(windows, params):
    """Build temporal blocked CV splits for all seeds."""
    p = params.get('step04', {})
    seeds = p.get('seeds', [0, 1, 7, 42, 99])
    n_folds = p.get('n_folds', 5)
    gap_s = p.get('gap_s', 3.0)

    blocks = create_temporal_blocks(windows, n_folds, gap_s)

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
        all_splits = build_cv_splits(windows, params)

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
