#!/usr/bin/env python3
"""
step16_parallel_universes.py — Master 20-way evaluation matrix.

Systematically trains and evaluates 20 combinations:
- 5 Preprocessing Universes (nonotch, notch, notch_art, notch_burst, notch_ab)
- 2 Scales (Intra-subject vs Inter-subject LOGO-CV)
- 2 Models (Random Forest vs Engagement Index)

Maintains perfectly strict mathematical comparability by always undersampling
the test sets to exactly 50/50 balance after applying the universe-specific
artifact/burst filters, using the Step 4 validation splits as the single source of truth.

Performs stepwise Wilcoxon testing to build a "chain of proof" linking every
combination back to the baseline: Intra + Notch + NoArt + NoBurst + EI.
"""

import numpy as np
import pickle
import json
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import recall_score, f1_score, accuracy_score
from scipy import stats as sp_stats
import warnings

import config
warnings.filterwarnings('ignore')

UNIVERSES = {
    'nonotch': 'features',
    'notch': 'features_notch',
    'notch_art': 'features_notch_artifact',
    'notch_burst': 'features_notch_burst',
    'notch_ab': 'features_notch_artifact_burst',
}

def undersample_balance_test(X, y, seed):
    """Randomly undersample majority class to achieve exact 50/50 balance."""
    n0 = np.sum(y == 0)
    n1 = np.sum(y == 1)
    n_min = min(n0, n1)
    if n_min == 0:
        return X[:0], y[:0]
    
    rng = np.random.RandomState(seed)
    idx0 = rng.choice(np.where(y == 0)[0], size=n_min, replace=False)
    idx1 = rng.choice(np.where(y == 1)[0], size=n_min, replace=False)
    idx = np.concatenate([idx0, idx1])
    rng.shuffle(idx)
    return X[idx], y[idx]


def evaluate_intra(run_dir, universe_dir, model_type, seeds):
    """Run intra-subject evaluation (Temporal Blocked CV) for all participants."""
    gs_dir = run_dir / "grid_search"
    splits_dir = run_dir / "splits"
    
    results_per_pid = {}
    for pid in config.INCLUDED_PARTICIPANTS:
        feat_path = universe_dir / f"P{pid}.pkl"
        split_path = splits_dir / f"P{pid}_splits.pkl"
        bp_path = gs_dir / f"P{pid}_best_params.json"
        
        if not feat_path.exists() or not split_path.exists():
            continue
            
        with open(feat_path, 'rb') as f:
            fd = pickle.load(f)
        with open(split_path, 'rb') as f:
            sd = pickle.load(f)
            
        if model_type == 'RF':
            X_all = fd['features_full']
            bp = json.load(open(bp_path)) if bp_path.exists() else {}
        else:
            X_all = fd['ei_values'].reshape(-1, 1)
            bp = {}
            
        y_all = fd['labels']
        wids = fd['window_ids']
        wid_to_idx = {int(w): i for i, w in enumerate(wids)}
        
        seed_test_recalls = []
        seed_train_recalls = []
        seed_test_f1s = []
        seed_train_f1s = []
        seed_test_accs = []
        seed_train_accs = []
        seed_test_ns = []
        
        for seed in seeds:
            splits = sd['splits'].get(seed, [])
            fold_test_recalls = []
            fold_train_recalls = []
            fold_test_f1s = []
            fold_train_f1s = []
            fold_test_accs = []
            fold_train_accs = []
            fold_test_ns = []
            
            for fi in splits:
                tri = [wid_to_idx[w] for w in fi['train_ids'] if w in wid_to_idx]
                tei = [wid_to_idx[w] for w in fi['test_ids'] if w in wid_to_idx]
                
                if len(tri) < 4 or len(tei) < 4: continue
                
                X_tr, y_tr = X_all[tri], y_all[tri]
                X_te, y_te = X_all[tei], y_all[tei]
                
                # Critically rebalance test set
                X_te, y_te = undersample_balance_test(X_te, y_te, seed + fi['fold'])
                if len(X_te) < 2 or len(np.unique(y_tr)) < 2: continue
                
                if model_type == 'RF':
                    clf = RandomForestClassifier(
                        n_estimators=bp.get('n_estimators', 200),
                        max_depth=bp.get('max_depth', 7),
                        min_samples_leaf=bp.get('min_samples_leaf', 5),
                        random_state=seed, n_jobs=-1)
                else:
                    scaler = StandardScaler()
                    X_tr = scaler.fit_transform(X_tr)
                    X_te = scaler.transform(X_te)
                    clf = LogisticRegression(class_weight='balanced', random_state=seed)
                    
                clf.fit(X_tr, y_tr)
                y_pred_te = clf.predict(X_te)
                y_pred_tr = clf.predict(X_tr)
                
                fold_test_recalls.append(recall_score(y_te, y_pred_te, pos_label=1, zero_division=0))
                fold_train_recalls.append(recall_score(y_tr, y_pred_tr, pos_label=1, zero_division=0))
                fold_test_f1s.append(f1_score(y_te, y_pred_te, pos_label=1, zero_division=0))
                fold_train_f1s.append(f1_score(y_tr, y_pred_tr, pos_label=1, zero_division=0))
                fold_test_accs.append(accuracy_score(y_te, y_pred_te))
                fold_train_accs.append(accuracy_score(y_tr, y_pred_tr))
                fold_test_ns.append(len(y_te))
                
            if fold_test_recalls:
                seed_test_recalls.append(np.mean(fold_test_recalls))
                seed_train_recalls.append(np.mean(fold_train_recalls))
                seed_test_f1s.append(np.mean(fold_test_f1s))
                seed_train_f1s.append(np.mean(fold_train_f1s))
                seed_test_accs.append(np.mean(fold_test_accs))
                seed_train_accs.append(np.mean(fold_train_accs))
                seed_test_ns.append(np.sum(fold_test_ns))
                
        if seed_test_recalls:
            results_per_pid[pid] = {
                'test_recall': float(np.mean(seed_test_recalls)),
                'train_recall': float(np.mean(seed_train_recalls)),
                'test_f1': float(np.mean(seed_test_f1s)),
                'train_f1': float(np.mean(seed_train_f1s)),
                'test_accuracy': float(np.mean(seed_test_accs)),
                'train_accuracy': float(np.mean(seed_train_accs)),
                'test_n': float(np.mean(seed_test_ns)),
            }
            
    return results_per_pid


def evaluate_inter(run_dir, universe_dir, model_type, seed=0):
    """Run inter-subject evaluation (LOGO-CV)."""
    gs_dir = run_dir / "grid_search"
    splits_dir = run_dir / "splits"
    
    all_data = {}
    for pid in config.INCLUDED_PARTICIPANTS:
        feat_path = universe_dir / f"P{pid}.pkl"
        split_path = splits_dir / f"P{pid}_splits.pkl"
        
        if not feat_path.exists() or not split_path.exists():
            continue
            
        with open(feat_path, 'rb') as f:
            fd = pickle.load(f)
        with open(split_path, 'rb') as f:
            sd = pickle.load(f)
            
        splits_seed0 = sd['splits'].get(0, [])
        valid_test_ids = []
        for fold in splits_seed0:
            valid_test_ids.extend(fold['test_ids'])
            
        wid_to_idx = {int(w): i for i, w in enumerate(fd['window_ids'])}
        valid_indices = [wid_to_idx[w] for w in valid_test_ids if w in wid_to_idx]
        
        X = fd['features_full'][valid_indices] if model_type == 'RF' else fd['ei_values'][valid_indices].reshape(-1, 1)
        y = fd['labels'][valid_indices]
        
        all_data[pid] = {'X': X, 'y': y}

    results_per_pid = {}
    for test_pid in sorted(all_data.keys()):
        X_tr_parts, y_tr_parts = [], []
        for pid, data in all_data.items():
            if pid == test_pid: continue
            X_tr_parts.append(data['X'])
            y_tr_parts.append(data['y'])
            
        X_tr = np.concatenate(X_tr_parts)
        y_tr = np.concatenate(y_tr_parts)
        X_te = all_data[test_pid]['X']
        y_te = all_data[test_pid]['y']
        
        # Critically rebalance test set
        X_te, y_te = undersample_balance_test(X_te, y_te, seed + test_pid)
        if len(X_te) < 2 or len(np.unique(y_tr)) < 2: continue
        
        if model_type == 'RF':
            bp_path = gs_dir / f"P{test_pid}_best_params.json"
            bp = json.load(open(bp_path)) if bp_path.exists() else {}
            clf = RandomForestClassifier(
                n_estimators=bp.get('n_estimators', 200),
                max_depth=bp.get('max_depth', 7),
                min_samples_leaf=bp.get('min_samples_leaf', 5),
                random_state=seed, n_jobs=-1)
        else:
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_tr)
            X_te = scaler.transform(X_te)
            clf = LogisticRegression(class_weight='balanced', random_state=seed)
            
        clf.fit(X_tr, y_tr)
        y_pred_te = clf.predict(X_te)
        y_pred_tr = clf.predict(X_tr)
        
        results_per_pid[test_pid] = {
            'test_recall': float(recall_score(y_te, y_pred_te, pos_label=1, zero_division=0)),
            'train_recall': float(recall_score(y_tr, y_pred_tr, pos_label=1, zero_division=0)),
            'test_f1': float(f1_score(y_te, y_pred_te, pos_label=1, zero_division=0)),
            'train_f1': float(f1_score(y_tr, y_pred_tr, pos_label=1, zero_division=0)),
            'test_accuracy': float(accuracy_score(y_te, y_pred_te)),
            'train_accuracy': float(accuracy_score(y_tr, y_pred_tr)),
            'test_n': float(len(y_te)),
        }
        
    return results_per_pid


def parse_params(comb_name):
    """Parse combination name into its 5 parameters."""
    parts = comb_name.split('|')
    return {
        'scale': parts[0],
        'notch': parts[1],
        'art': parts[2],
        'burst': parts[3],
        'model': parts[4]
    }


def compute_param_diff(c1, c2):
    """Return number of differing parameters between two combinations."""
    p1 = parse_params(c1)
    p2 = parse_params(c2)
    return sum(1 for k in p1 if p1[k] != p2[k])


def run(run_dir, params):
    config.pprint_step(16, "MASTER PARALLEL UNIVERSE EVALUATION")
    
    seeds = params.get('step07', {}).get('seeds', [0, 1, 7, 42, 99])
    
    results = {}
    
    # 1. Run all 20 combinations
    print(f"  Evaluating 20 parallel combinations...")
    for scale in ['Intra', 'Inter']:
        for u_name, u_dir in UNIVERSES.items():
            notch_param = 'Notch' if 'notch' in u_name and u_name != 'nonotch' else 'NoNotch'
            art_param = 'Art' if 'art' in u_name or 'ab' in u_name else 'NoArt'
            burst_param = 'Burst' if 'burst' in u_name or 'ab' in u_name else 'NoBurst'
            
            universe_dir = run_dir / u_dir
            for model in ['EI', 'RF']:
                comb_name = f"{scale}|{notch_param}|{art_param}|{burst_param}|{model}"
                
                if scale == 'Intra':
                    metrics = evaluate_intra(run_dir, universe_dir, model, seeds)
                else:
                    metrics = evaluate_inter(run_dir, universe_dir, model, seed=0)
                    
                results[comb_name] = metrics
                
                if metrics:
                    mean_test = np.mean([m['test_recall'] for m in metrics.values()])
                    mean_train = np.mean([m['train_recall'] for m in metrics.values()])
                    mean_test_n = np.mean([m['test_n'] for m in metrics.values()])
                    delta = mean_train - mean_test
                else:
                    mean_test = mean_train = mean_test_n = delta = 0
                    
                print(f"    {comb_name:35} -> Test={mean_test:.4f} (Train={mean_train:.4f}, Δ={delta:.4f}, N_test={mean_test_n:.1f})")
                
    # Save raw recalls for viz16a
    with open(run_dir / "step16_parallel_recalls.pkl", 'wb') as f:
        viz_results = {}
        for k, v in results.items():
            viz_results[k] = {pid: m['test_recall'] for pid, m in v.items()}
        pickle.dump(viz_results, f)
        
    # Export comprehensive flat CSV
    flat_rows = []
    for comb, metrics in results.items():
        if metrics:
            for pid, m in metrics.items():
                row = parse_params(comb)
                row['combination'] = comb
                row['pid'] = pid
                row.update(m)
                flat_rows.append(row)
                
    df_metrics = pd.DataFrame(flat_rows)
    df_metrics.to_csv(run_dir / "parallel_universe_metrics.csv", index=False)
        
    # 2. Stepwise Significance Testing (Chain of Proof)
    baseline = 'Intra|Notch|NoArt|NoBurst|EI'
    
    print(f"\n  Computing Stepwise Significance Chain...")
    print(f"  Baseline: {baseline}")
    
    table_rows = []
    
    for comb, metrics in results.items():
        if metrics:
            mean_test = np.mean([m['test_recall'] for m in metrics.values()])
            mean_train = np.mean([m['train_recall'] for m in metrics.values()])
            mean_test_n = np.mean([m['test_n'] for m in metrics.values()])
        else:
            mean_test = mean_train = mean_test_n = 0
            
        diff = compute_param_diff(comb, baseline)
        
        row = {
            'Combination': comb,
            'Scale': parse_params(comb)['scale'],
            'Notch': parse_params(comb)['notch'],
            'Artifact_Filter': parse_params(comb)['art'],
            'Burst_Filter': parse_params(comb)['burst'],
            'Model': parse_params(comb)['model'],
            'Test_Recall': mean_test,
            'Train_Recall': mean_train,
            'Train_Test_Delta': mean_train - mean_test,
            'Test_N': mean_test_n,
            'Diff_from_Baseline': diff,
            'Compared_To': '',
            'Delta': '',
            'Wilcoxon_p': '',
            'Significant': ''
        }
        
        if diff == 0:
            table_rows.append(row)
            continue
            
        # Find valid "step before that" combinations (diff - 1 from baseline)
        parents = []
        for other_comb in results:
            if compute_param_diff(other_comb, baseline) == diff - 1:
                # Must be on the path (differ by exactly 1 parameter from current)
                if compute_param_diff(comb, other_comb) == 1:
                    parents.append(other_comb)
                    
        # Compare to all valid parents
        for parent in parents:
            p_metrics = results[parent]
            
            # Match PIDs
            matched_p = []
            matched_c = []
            for pid in sorted(metrics.keys()):
                if pid in p_metrics:
                    matched_p.append(p_metrics[pid]['test_recall'])
                    matched_c.append(metrics[pid]['test_recall'])
                    
            if len(matched_p) > 5:
                arr_p = np.array(matched_p)
                arr_c = np.array(matched_c)
                try:
                    w, p = sp_stats.wilcoxon(arr_c - arr_p)
                except:
                    p = 1.0
                
                delta = np.mean(arr_c) - np.mean(arr_p)
                
                r_copy = row.copy()
                r_copy['Compared_To'] = parent
                r_copy['Delta'] = delta
                r_copy['Wilcoxon_p'] = p
                r_copy['Significant'] = 'SIG' if p < 0.05 else 'ns'
                table_rows.append(r_copy)
            else:
                table_rows.append(row)
                
    df = pd.DataFrame(table_rows)
    df.to_csv(run_dir / "parallel_universe_comparisons.csv", index=False)
    
    print(f"\n  ✓ 20-way matrix evaluated and stepwise significance saved.")

if __name__ == "__main__":
    print("Use run.py to execute the pipeline.")
