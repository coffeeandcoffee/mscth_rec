import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
import ast
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def generate_diagnostics(scale, df_scale, best_configs, viz_dir):
    import config
    comb_rf = best_configs.get(f'{scale}_RF')
    comb_ei = best_configs.get(f'{scale}_EI')
    comb_lr = best_configs.get(f'{scale}_LR')
    comb_flr = best_configs.get(f'{scale}_BestFeatureLR')
    
    df_rf = df_scale[df_scale['combination'] == comb_rf] if comb_rf else pd.DataFrame()
    df_ei = df_scale[df_scale['combination'] == comb_ei] if comb_ei else pd.DataFrame()
    df_lr = df_scale[df_scale['combination'] == comb_lr] if comb_lr else pd.DataFrame()
    df_flr = df_scale[df_scale['combination'] == comb_flr] if comb_flr else pd.DataFrame()
    
    def calc_metrics(cm_str_series):
        f1_stay, f1_skip = [], []
        prec_stay, prec_skip = [], []
        rec_stay, rec_skip = [], []
        
        for s in cm_str_series:
            cm = ast.literal_eval(s)
            tn, fp = cm[0][0], cm[0][1]
            fn, tp = cm[1][0], cm[1][1]
            
            # STAY (Class 1)
            p_stay = tp / (tp + fp) if (tp + fp) > 0 else 0
            r_stay = tp / (tp + fn) if (tp + fn) > 0 else 0
            f_stay = 2 * p_stay * r_stay / (p_stay + r_stay) if (p_stay + r_stay) > 0 else 0
            
            # SKIP (Class 0)
            p_skip = tn / (tn + fn) if (tn + fn) > 0 else 0
            r_skip = tn / (tn + fp) if (tn + fp) > 0 else 0
            f_skip = 2 * p_skip * r_skip / (p_skip + r_skip) if (p_skip + r_skip) > 0 else 0
            
            prec_stay.append(p_stay * 100)
            rec_stay.append(r_stay * 100)
            f1_stay.append(f_stay * 100)
            
            prec_skip.append(p_skip * 100)
            rec_skip.append(r_skip * 100)
            f1_skip.append(f_skip * 100)
            
        return {
            'Precision (STAY)': np.array(prec_stay),
            'Recall (STAY)': np.array(rec_stay),
            'F1 Test (STAY)': np.array(f1_stay),
            'Precision (SKIP)': np.array(prec_skip),
            'Recall (SKIP)': np.array(rec_skip),
            'F1 Test (SKIP)': np.array(f1_skip)
        }
    
    def extract_all(df_model, is_dummy=False):
        if df_model is None or df_model.empty:
            return None
        
        test_col = 'dummy_test_cm' if is_dummy else 'test_cm'
        train_col = 'dummy_train_cm' if is_dummy else 'train_cm'
        
        if test_col not in df_model.columns:
            return None
            
        test_metrics = calc_metrics(df_model[test_col])
        train_metrics = calc_metrics(df_model[train_col])
        
        res = test_metrics.copy()
        res['F1 Train (STAY)'] = train_metrics['F1 Test (STAY)']
        res['F1 Train (SKIP)'] = train_metrics['F1 Test (SKIP)']
        return res

    is_ablation = getattr(config, 'ENABLE_EI_ABLATION', False) or getattr(config, 'ENABLE_EI_STAT_EXCHANGE', False)
    
    models = {}
    
    if not is_ablation:
        m_dummy = extract_all(df_rf if not df_rf.empty else (df_lr if not df_lr.empty else df_ei), is_dummy=True)
        models['Coin Flip'] = m_dummy
        models['EI LR'] = extract_all(df_ei)
        if getattr(config, 'ENABLE_BEST_FEATURE_LR', True):
            models['Best Feature LR'] = extract_all(df_flr)
        models['All Features LR'] = extract_all(df_lr)
        if getattr(config, 'ENABLE_RF', True):
            models['All Features RF'] = extract_all(df_rf)
    else:
        models['EI LR'] = extract_all(df_ei)
        if getattr(config, 'ENABLE_EI_ABLATION', False):
            ablation_names = ['Beta-only', 'Alpha-only', 'Theta-only', 'Beta/Alpha', 'Beta/Theta', 'EEI']
            for name in ablation_names:
                comb = best_configs.get(f'{scale}_{name}')
                df_m = df_scale[df_scale['combination'] == comb] if comb else pd.DataFrame()
                models[name] = extract_all(df_m)
                
        if getattr(config, 'ENABLE_EI_STAT_EXCHANGE', False):
            stats = ["mean", "std", "min", "max", "peakfreq", "macrofreq", "rel_power", "hjorth_act", "hjorth_mob", "hjorth_comp"]
            if getattr(config, 'DEFAULT_PARAMS', {}).get('experimental', {}).get('remove_min_max', False):
                stats = [s for s in stats if s not in ('min', 'max')]
            for st in stats:
                name = f'EI_{st}'
                comb = best_configs.get(f'{scale}_{name}')
                df_m = df_scale[df_scale['combination'] == comb] if comb else pd.DataFrame()
                models[name] = extract_all(df_m)
            
    # Filter out None
    models = {k: v for k, v in models.items() if v is not None}
        
        
    model_names = list(models.keys())
    
    metrics_order = [
        'F1 Train (STAY)', 'F1 Test (STAY)', 'Precision (STAY)', 'Recall (STAY)',
        'F1 Train (SKIP)', 'F1 Test (SKIP)', 'Precision (SKIP)', 'Recall (SKIP)'
    ]
    
    # --- TABLE 1: Metrics ---
    table_data = []
    columns = ['Metric']
    for m_name in model_names:
        columns.extend([f'{m_name} (Value)', f'{m_name} (Sig)'])
        
    tex_cols = "c" * (len(model_names) * 2)
    header_1 = " & ".join([f"\\multicolumn{{2}}{{c}}{{{m}}}" for m in model_names])
    header_2 = " & ".join(["Val & Sig"] * len(model_names))
    
    tex_lines = [
        "\\begin{table}[ht]",
        "\\centering",
        "\\begin{tabular}{l" + tex_cols + "}",
        "\\hline",
        f"Metric & {header_1} \\\\",
        f" & {header_2} \\\\",
        "\\hline"
    ]
    
    sig_better_cells = []
    
    for metric in metrics_order:
        row = [metric]
        tex_row = [metric.replace('%', '\\%')]
        for m_name in model_names:
            m_dict = models.get(m_name)
            if m_dict is None:
                row.extend(['-', '-'])
                tex_row.extend(['-', '-'])
                continue
                
            vals = m_dict[metric]
            mean_val = np.mean(vals)
            
            if m_name == 'Coin Flip' or (is_ablation and m_name == 'EI LR'):
                sig = '-'
            else:
                try:
                    baseline_m = models['EI LR'] if is_ablation else models['Coin Flip']
                    stat, p = wilcoxon(vals, baseline_m[metric])
                    if p < 0.05:
                        sig = 'TRUE'
                        if mean_val > np.mean(baseline_m[metric]):
                            sig_better_cells.append((len(table_data), len(row)))
                    else:
                        sig = 'FALSE'
                except Exception:
                    sig = 'FALSE'
            
            val_str = f"{mean_val:.1f}%"
            row.extend([val_str, sig])
            
            sig_tex = "\\textcolor{green}{TRUE}" if sig == 'TRUE' else sig
            tex_row.extend([val_str.replace('%', '\\%'), sig_tex])
            
        table_data.append(row)
        tex_lines.append(" & ".join(tex_row) + " \\\\")
        
    baseline_text = "Standard EI" if is_ablation else "Coin Flip"
    tex_lines.extend([
        "\\hline",
        "\\end{tabular}",
        "\\caption{Minimal Metrics for " + scale + " Models. Significance tested using Wilcoxon signed-rank test against " + baseline_text + " (two-sided, $\\alpha=0.05$, N=25).}",
        "\\label{tab:metrics_" + scale.lower() + ("_ablation" if is_ablation else "") + "}",
        "\\end{table}"
    ])
    
    suffix = "_ablation" if is_ablation else ""
    with open(viz_dir / f"viz17_{scale.lower()}_metrics_table{suffix}.tex", 'w') as f:
        f.write("\n".join(tex_lines))
        
    # Render PNG for Table 1
    fig_width = max(14, len(columns) * 0.8)
    fig, ax = plt.subplots(figsize=(fig_width, 4))
    ax.axis('tight')
    ax.axis('off')
    tbl = ax.table(cellText=table_data, colLabels=columns, loc='center', cellLoc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(12)
    tbl.scale(1.5, 1.5)
    tbl.auto_set_column_width(col=list(range(len(columns))))
    
    # Color significance cells
    for i in range(len(table_data)):
        for j in range(1, len(columns)):
            if "Sig" in columns[j]:
                val = table_data[i][j]
                if val == 'TRUE':
                    tbl[(i+1, j)].set_facecolor('#d4edda') # light green
                elif val == 'FALSE':
                    tbl[(i+1, j)].set_facecolor('#f8d7da') # light red
    
    # Highlight significantly better metric cells or gray out text of non-significant ones
    for i in range(len(table_data)):
        for j in range(1, len(columns)):
            if "Value" in columns[j]:
                m_idx = (j - 1) // 2
                m_name = model_names[m_idx]
                if m_name == 'Coin Flip' or (is_ablation and m_name == 'EI LR'):
                    continue
                
                if (i, j) in sig_better_cells:
                    cell = tbl[(i+1, j)]
                    cell.set_edgecolor('#00aa00')
                    cell.set_linewidth(2.5)
                else:
                    tbl[(i+1, j)].get_text().set_color('gray')
        
    plt.title(f"{scale} Model Metrics (Wilcoxon signed-rank test, N=25)", pad=20, fontsize=14, fontweight='bold')
    plt.savefig(viz_dir / f"viz17_{scale.lower()}_metrics_table{suffix}.png", bbox_inches='tight', dpi=200)
    plt.close()
    
    # --- TABLE 2: Optimal Parameters ---
    params_data = []
    param_names = ['Scale', 'Notch', 'Artifact', 'Burst']
    
    opt_models = [m for m in model_names if m != 'Coin Flip']
    
    for m_name in opt_models:
        comb = None
        if not is_ablation:
            if m_name == 'EI LR': comb = comb_ei
            elif m_name == 'Best Feature LR': comb = comb_flr
            elif m_name == 'All Features LR': comb = comb_lr
            elif m_name == 'All Features RF': comb = comb_rf
        else:
            if m_name == 'EI LR': comb = comb_ei
            else: comb = best_configs.get(f'{scale}_{m_name}')
        
        if comb:
            parts = comb.split('|')
            params_data.append(parts[:4])
        else:
            params_data.append(['-', '-', '-', '-'])
            
    # Transpose for table (Rows = Params, Cols = Models)
    params_data = np.array(params_data).T.tolist()
    
    # Add table headers
    p_columns = ['Parameter'] + opt_models
    p_table = []
    for i, p_name in enumerate(param_names):
        row = [p_name] + params_data[i]
        p_table.append(row)
        
    # Render PNG
    fig_width2 = max(10, len(p_columns) * 1.5)
    fig, ax = plt.subplots(figsize=(fig_width2, 3))
    ax.axis('tight')
    ax.axis('off')
    tbl2 = ax.table(cellText=p_table, colLabels=p_columns, loc='center', cellLoc='center')
    tbl2.auto_set_font_size(False)
    tbl2.set_fontsize(12)
    tbl2.scale(1.5, 1.8)
    plt.title(f"{scale} Optimal Preprocessing Parameters", pad=20, fontsize=14, fontweight='bold')
    plt.savefig(viz_dir / f"viz17_{scale.lower()}_optimal_params{suffix}.png", bbox_inches='tight', dpi=200)
    plt.close()
    
    # --- DIAGNOSTIC REPORT ---
    report_lines = []
    report_lines.append("\\section*{Diagnostic Report (" + scale + ")}")
    
    for m_name in opt_models:
        m_dict = models.get(m_name)
        if m_dict is None: continue
        
        tr_f1 = np.mean(m_dict['F1 Train (STAY)'])
        te_f1 = np.mean(m_dict['F1 Test (STAY)'])
        te_prec = np.mean(m_dict['Precision (STAY)'])
        te_rec = np.mean(m_dict['Recall (STAY)'])
        gap = tr_f1 - te_f1
        
        diagnosis = ""
        
        # Test for randomness (significance vs Coin Flip baseline)
        f1_vals = m_dict['F1 Test (STAY)']
        
        # In ablation mode, we don't have 'Coin Flip' explicitly in `models` if we didn't add it.
        # But wait, we always need a baseline to test for randomness.
        # Let's fallback to calculating the dummy metrics if it's missing.
        baseline_model_name = 'EI LR' if is_ablation else 'Coin Flip'
        baseline_m = models.get(baseline_model_name)
        
        try:
            if baseline_m is not None:
                stat, p_val = wilcoxon(f1_vals, baseline_m['F1 Test (STAY)'])
            else:
                p_val = 1.0 # default to not significant if baseline missing
            is_significant = p_val < 0.05
        except Exception:
            is_significant = False

        if not is_significant:
            diagnosis = "Random chance performance (No signal). The F1 score is not statistically significantly better than the Coin Flip baseline (p >= 0.05)."
        else:
            # Significant Signal Detected. Now characterize the failure or success modes.
            
            # Check for Class Collapse (STAY)
            if te_rec >= 85.0 and te_prec < 65.0:
                diagnosis = "Class collapse to positive (STAY). The model achieves high recall by predicting mostly STAY, sacrificing precision."
            # Check for Class Collapse (SKIP)
            elif te_rec < 35.0 and te_prec < 35.0:
                diagnosis = "Class collapse to negative (SKIP). The model fails to identify the positive class."
            else:
                # Good Predictive Power. Check Overfitting (Gap)
                if gap >= 40.0:
                    diagnosis = "Extreme memorization. The model demonstrates genuine predictive power but completely overfits the training data with massive performance drop on test."
                elif gap >= 20.0 and gap < 40.0:
                    diagnosis = "Severe overfit. The model demonstrates predictive power but struggles to generalize."
                elif gap >= 5.0 and gap < 20.0:
                    diagnosis = "Mild overfit. The model demonstrates genuine predictive power with expected minor performance degradation on unseen data."
                elif gap >= -5.0 and gap < 5.0:
                    diagnosis = "Genuine predictive power. Healthy generalization with statistically identical train and test performance."
                else:
                    diagnosis = "Anomalous performance (Test > Train). The model performs better on unseen data than training data, suggesting potential leakage or an anomalous fold distribution."
                
        report_lines.append(f"\\subsection*{{Model: {m_name}}}")
        report_lines.append(f"Train F1 = {tr_f1:.1f}\\%, Test F1 = {te_f1:.1f}\\%, Precision = {te_prec:.1f}\\%, Recall = {te_rec:.1f}\\%.")
        report_lines.append(f"\\newline")
        report_lines.append(f"\\textbf{{Significance vs Coin Flip:}} {'p < 0.05' if is_significant else 'p >= 0.05'}")
        report_lines.append(f"\\newline")
        report_lines.append(f"\\textbf{{Diagnosis:}} {diagnosis}")
        report_lines.append("")
        
    with open(viz_dir / f"viz17_{scale.lower()}_diagnostics.tex", 'w') as f:
        f.write("\n".join(report_lines))
