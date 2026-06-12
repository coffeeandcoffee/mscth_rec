import pickle
import numpy as np
from pathlib import Path
import json

features_dir = Path("runs/run_20260609_182942/features")
features_notch_dir = Path("runs/run_20260609_182942/features_notch")
features_notch_art_dir = Path("runs/run_20260609_182942/features_notch_artifact")
features_notch_burst_dir = Path("runs/run_20260609_182942/features_notch_burst")
features_notch_ab_dir = Path("runs/run_20260609_182942/features_notch_artifact_burst")

for d in [features_dir, features_notch_dir, features_notch_art_dir, features_notch_burst_dir, features_notch_ab_dir]:
    all_X, all_y = [], []
    feature_names = None
    for pkl_file in d.glob("P*.pkl"):
        with open(pkl_file, 'rb') as f:
            data = pickle.load(f)
        if feature_names is None:
            feature_names = data['agg_names_full']
        all_X.append(data['features_full'])
        all_y.append(data['labels'])
        
    if not all_X: continue
    X_mat = np.concatenate(all_X)
    y_vec = np.concatenate(all_y)
    
    stay_mask = y_vec == 1
    skip_mask = y_vec == 0
    if np.sum(stay_mask) == 0 or np.sum(skip_mask) == 0: continue
        
    mean_stay = np.mean(X_mat[stay_mask], axis=0)
    mean_skip = np.mean(X_mat[skip_mask], axis=0)
    var_stay = np.var(X_mat[stay_mask], axis=0)
    var_skip = np.var(X_mat[skip_mask], axis=0)
    
    pooled_std = np.sqrt((var_stay + var_skip) / 2)
    pooled_std[pooled_std == 0] = 1e-9
    d_vals = np.abs((mean_stay - mean_skip) / pooled_std)
    
    ranking = [{"index": i, "name": name, "d": float(d_vals[i])} for i, name in enumerate(feature_names)]
    ranking.sort(key=lambda x: x['d'], reverse=True)
    
    with open(d / "best_features_ranking.json", 'w') as f:
        json.dump(ranking, f, indent=2)
    print(f"Generated ranking for {d.name}: Top feature is {ranking[0]['name']} (d={ranking[0]['d']:.3f})")

