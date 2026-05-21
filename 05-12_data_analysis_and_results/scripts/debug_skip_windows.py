#!/usr/bin/env python3
"""Quick diagnostic: verify step01 SKIP labeling around A-keypresses."""

import pickle, numpy as np, sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent))
import config

run_dir = __import__('pathlib').Path(__file__).resolve().parent.parent / "runs" / "run_20260519_070338"
processed_dir = run_dir / "processed" / "nonotch"

# Check first few participants
for pid in config.INCLUDED_PARTICIPANTS[:3]:
    pkl_path = processed_dir / f"P{pid}.pkl"
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    
    dfs = data['dfs']
    fs = data['fs']
    
    print(f"\n{'='*70}")
    print(f"  P{pid} — fs={fs}Hz, {len(dfs)} CSVs")
    print(f"{'='*70}")
    
    for ci, df in enumerate(dfs):
        ts = df['lsl_timestamp'].values
        classes = df['class'].values
        
        # Find SKIP regions
        skip_mask = np.array([c == 'SKIP' for c in classes])
        skip_idx = np.where(skip_mask)[0]
        
        if len(skip_idx) == 0:
            print(f"  CSV {ci}: No SKIP samples")
            continue
        
        # Find contiguous SKIP blocks
        breaks = np.where(np.diff(skip_idx) > 1)[0] + 1
        blocks = np.split(skip_idx, breaks)
        
        print(f"\n  CSV {ci}: {len(blocks)} SKIP blocks, total SKIP samples={len(skip_idx)}")
        print(f"  {'Block':>5}  {'Start(s)':>10}  {'End(s)':>10}  {'Duration(s)':>12}  {'N_samples':>10}")
        print(f"  {'─'*5}  {'─'*10}  {'─'*10}  {'─'*12}  {'─'*10}")
        
        for bi, block in enumerate(blocks[:20]):  # Show first 20 blocks
            t_start = ts[block[0]]
            t_end = ts[block[-1]]
            dur = t_end - t_start
            print(f"  {bi:>5}  {t_start - ts[0]:>10.3f}  {t_end - ts[0]:>10.3f}  {dur:>12.3f}  {len(block):>10}")
            
            if dur < 5.5:
                print(f"         ^^^ WARNING: duration {dur:.3f}s < 6s! Should be ~6s (±3s around A-press)")
        
        # Check: do all blocks have ~6s duration?
        durs = [ts[b[-1]] - ts[b[0]] for b in blocks]
        print(f"\n  Duration stats: min={min(durs):.3f}s, max={max(durs):.3f}s, mean={np.mean(durs):.3f}s")
        print(f"  Blocks < 5.5s: {sum(1 for d in durs if d < 5.5)}/{len(durs)}")
        print(f"  Blocks ≈ 6s (5.5-6.5): {sum(1 for d in durs if 5.5 <= d <= 6.5)}/{len(durs)}")
        print(f"  Blocks > 6.5s (merged): {sum(1 for d in durs if d > 6.5)}/{len(durs)}")
