import pickle, numpy as np
from pathlib import Path

run_dir = Path("/Users/gregorlederer/Documents/MSc Thesis - EEG Neuroscience/Data Recording and Quality Tests/05-12_data_analysis_and_results/runs/run_20260603_171603")
universe_dir = run_dir / "features_notch_burst"
splits_dir = run_dir / "splits"

pid = 4
feat_path = universe_dir / f"P{pid}.pkl"
split_path = splits_dir / f"P{pid}_splits.pkl"

with open(feat_path, 'rb') as f: fd = pickle.load(f)
with open(split_path, 'rb') as f: sd = pickle.load(f)

splits_seed0 = sd['splits'].get(0, [])
valid_test_ids = []
for fold in splits_seed0:
    valid_test_ids.extend(fold['test_ids'])

wid_to_idx = {int(w): i for i, w in enumerate(fd['window_ids'])}
valid_indices = [wid_to_idx[w] for w in valid_test_ids if w in wid_to_idx]
y = fd['labels'][valid_indices]
print(f"P{pid}: len(y)={len(y)}, sum(y)={sum(y)}, proportion={sum(y)/len(y):.3f}")

