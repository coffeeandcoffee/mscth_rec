import pickle, numpy as np
import json
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score, f1_score

run_dir = Path("/Users/gregorlederer/Documents/MSc Thesis - EEG Neuroscience/Data Recording and Quality Tests/05-12_data_analysis_and_results/runs/run_20260603_171603")
universe_dir = run_dir / "features_notch_burst"
splits_dir = run_dir / "splits"
gs_dir = run_dir / "grid_search"

all_data = {}
for pid in [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17, 18, 20, 21, 22, 23, 24, 25, 26, 27, 28, 30, 31]:
    feat_path = universe_dir / f"P{pid}.pkl"
    split_path = splits_dir / f"P{pid}_splits.pkl"
    if not feat_path.exists(): continue
    with open(feat_path, 'rb') as f: fd = pickle.load(f)
    with open(split_path, 'rb') as f: sd = pickle.load(f)
    splits_seed0 = sd['splits'].get(0, [])
    valid_test_ids = []
    for fold in splits_seed0: valid_test_ids.extend(fold['test_ids'])
    wid_to_idx = {int(w): i for i, w in enumerate(fd['window_ids'])}
    valid_indices = [wid_to_idx[w] for w in valid_test_ids if w in wid_to_idx]
    all_data[pid] = {'X': fd['features_full'][valid_indices], 'y': fd['labels'][valid_indices]}

test_pid = 4
X_tr_parts, y_tr_parts = [], []
for pid, data in all_data.items():
    if pid == test_pid: continue
    X_tr_parts.append(data['X'])
    y_tr_parts.append(data['y'])

X_tr = np.concatenate(X_tr_parts)
y_tr = np.concatenate(y_tr_parts)
X_te = all_data[test_pid]['X']
y_te = all_data[test_pid]['y']

print(f"Train y: len={len(y_tr)}, sum={sum(y_tr)}, prop={sum(y_tr)/len(y_tr):.3f}")

bp_path = gs_dir / f"P{test_pid}_best_params.json"
bp = json.load(open(bp_path)) if bp_path.exists() else {}
clf = RandomForestClassifier(n_estimators=bp.get('n_estimators', 200), max_depth=bp.get('max_depth', 7), min_samples_leaf=bp.get('min_samples_leaf', 5), random_state=0, n_jobs=-1)
clf.fit(X_tr, y_tr)
y_pred_te = clf.predict(X_te)
y_pred_tr = clf.predict(X_tr)

print(f"Test Pred: {np.unique(y_pred_te, return_counts=True)}")
print(f"Test Recall: {recall_score(y_te, y_pred_te, pos_label=1, zero_division=0):.3f}")
print(f"Test F1: {f1_score(y_te, y_pred_te, pos_label=1, zero_division=0):.3f}")
