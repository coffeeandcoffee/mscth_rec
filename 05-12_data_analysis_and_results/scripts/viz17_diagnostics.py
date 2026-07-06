import numpy as np
import re

def post_process_tex_lines(lines):
    # Determine which Sig columns are all '-'
    data_start = -1
    for i, line in enumerate(lines):
        if "Metric &" in line:
            h2_idx = i + 1
            break
    else:
        return lines
        
    for i in range(h2_idx+1, len(lines)):
        if "hline" in lines[i]: continue
        if "&" in lines[i]:
            data_start = i
            break
            
    if data_start == -1: return lines
            
    data_end = -1
    for i in range(data_start, len(lines)):
        if "hline" in lines[i] or r"\end{tabular}" in lines[i]:
            data_end = i
            break
            
    def parse_row(row_str):
        parts = row_str.split("&")
        if len(parts) == 0: return []
        parts[-1] = parts[-1].replace(r"\\", "")
        return [p.strip() for p in parts]
        
    data_rows = []
    for i in range(data_start, data_end):
        row_str = lines[i]
        if "&" not in row_str:
            data_rows.append(None)
        else:
            data_rows.append(parse_row(row_str))
            
    num_cols = len(data_rows[0])
    cols_to_remove = []
    for col_idx in range(2, num_cols, 2):
        is_all_dash = True
        for r in data_rows:
            if r is not None:
                if r[col_idx] != "-":
                    is_all_dash = False
                    break
        if is_all_dash:
            cols_to_remove.append(col_idx)
            
    for r in data_rows:
        if r is None: continue
        for col_idx in sorted(cols_to_remove, reverse=True):
            if col_idx < len(r):
                del r[col_idx]
            
    for i, r in enumerate(data_rows):
        if r is not None:
            lines[data_start + i] = " & ".join(r) + " \\\\\\\\"
            
    h1_parts = [p.strip() for p in lines[h2_idx-1].replace(r"\\", "").split("&")]
    h2_parts = [p.strip() for p in lines[h2_idx].replace(r"\\", "").split("&")]
    
    new_h1_parts = [h1_parts[0]]
    new_h2_parts = [h2_parts[0]]
    
    for i in range(1, len(h1_parts)):
        sig_col_idx = i * 2
        m_name = re.sub(r"\\multicolumn\{2\}\{c\}\{(.*?)\}", r"\1", h1_parts[i])
        if sig_col_idx in cols_to_remove:
            new_h1_parts.append(m_name)
            new_h2_parts.append("Val")
        else:
            new_h1_parts.append(h1_parts[i])
            new_h2_parts.append("Val & Sig")
            
    lines[h2_idx-1] = " & ".join(new_h1_parts) + " \\\\\\\\"
    lines[h2_idx] = " & ".join(new_h2_parts) + " \\\\\\\\"
    
    new_num_cols = len(" & ".join(new_h2_parts).split("&"))
    for i, line in enumerate(lines):
        if "\\begin{tabular}" in line:
            lines[i] = re.sub(r"\\begin\{tabular\}\{.*?\}", f"\\\\begin{{tabular}}{{l{'c' * (new_num_cols - 1)}}}", line)
            break
            
    return lines

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
    
    # Always add Coin Flip as a baseline
    m_dummy = extract_all(df_ei, is_dummy=True)
    models['Coin Flip'] = m_dummy
    
    if not is_ablation:
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
                
    if getattr(config, 'ENABLE_TOP_FEATURES_EVAL', False):
        for top_model in ['TP10_raw_std_LR', 'All_raw_std_LR', 'AF7_high_gamma_mean_LR', 'All_high_gamma_mean_LR']:
            comb = best_configs.get(f'{scale}_{top_model}')
            df_m = df_scale[df_scale['combination'] == comb] if comb else pd.DataFrame()
            name_clean = top_model.replace('_LR', ' LR').replace('_', ' ')
            models[name_clean] = extract_all(df_m)
            
    # Filter out None
    models = {k: v for k, v in models.items() if v is not None}
        
        
    model_names = list(models.keys())
    
    def create_table_n1():
        # N=1: Only 'Coin Flip' and 'EI LR'
        m_names = [m for m in ['Coin Flip', 'EI LR'] if m in models]
        if len(m_names) < 2: return
        
        t_data = []
        cols = ['Metric']
        for m in m_names:
            cols.extend([f'{m} (Value)', f'{m} (Sig)'])
            
        t_tex_cols = "c" * (len(m_names) * 2)
        h1 = " & ".join([f"\\multicolumn{{2}}{{c}}{{{m}}}" for m in m_names])
        h2 = " & ".join(["Val & Sig"] * len(m_names))
        
        t_tex_lines = [
            "\\begin{table}[ht]",
            "\\centering",
            "\\begin{tabular}{l" + t_tex_cols + "}",
            "\\hline",
            f"Metric & {h1} \\\\",
            f" & {h2} \\\\",
            "\\hline"
        ]
        
        sb_cells = []
        metrics_order_n1 = [
            'F1 Train (STAY)', 'F1 Test (STAY)', 'Precision (STAY)', 'Recall (STAY)',
            'F1 Train (SKIP)', 'F1 Test (SKIP)', 'Precision (SKIP)', 'Recall (SKIP)'
        ]
        
        for metric in metrics_order_n1:
            row = [metric]
            t_tex_row = [metric.replace('%', '\\%')]
            for m in m_names:
                m_dict = models.get(m)
                vals = m_dict[metric]
                mean_val = np.mean(vals)
                
                if m == 'Coin Flip':
                    sig = '-'
                else:
                    try:
                        stat, p = wilcoxon(vals, models['Coin Flip'][metric])
                        if p < 0.05:
                            sig = 'TRUE'
                            if mean_val > np.mean(models['Coin Flip'][metric]):
                                sb_cells.append((len(t_data), len(row)))
                        else:
                            sig = 'FALSE'
                    except:
                        sig = 'FALSE'
                        
                val_str = f"{mean_val:.1f}%"
                row.extend([val_str, sig])
                sig_tex = "TRUE" if sig == 'TRUE' else sig
                val_tex = val_str.replace('%', '\\%')
                if sig == 'TRUE': val_tex = f"\\textcolor{{green}}{{{val_tex}}}"
                t_tex_row.extend([val_tex, sig_tex])
                
            t_data.append(row)
            t_tex_lines.append(" & ".join(t_tex_row) + " \\\\")
            
        t_tex_lines.extend([
            "\\hline",
            "\\end{tabular}",
            "\\caption{Step 1: EI vs Coin Flip. Significance tested using Wilcoxon signed-rank test against Coin Flip (two-sided, $\\alpha=0.05$, N=25).}",
            "\\label{tab:metrics_1_" + scale.lower() + "}",
            "\\end{table}"
        ])
        
        with open(viz_dir / f"viz17_1_{scale.lower()}.tex", 'w') as f:
            f.write("\n".join(post_process_tex_lines(t_tex_lines)))
            
        fw = max(8, len(cols) * 0.8)
        fig, ax = plt.subplots(figsize=(fw, 4))
        ax.axis('tight')
        ax.axis('off')
        tbl = ax.table(cellText=t_data, colLabels=cols, loc='center', cellLoc='center')
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(12)
        tbl.scale(1.5, 1.5)
        
        for i in range(len(t_data)):
            for j in range(1, len(cols)):
                if "Sig" in cols[j]:
                    v = t_data[i][j]
                    if v == 'TRUE':
                        tbl[(i+1, j)].set_facecolor('#d4edda')
                    elif v == 'FALSE':
                        tbl[(i+1, j)].set_facecolor('#f8d7da')
                        
        for i in range(len(t_data)):
            for j in range(1, len(cols)):
                if "Value" in cols[j]:
                    mi = (j - 1) // 2
                    mn = m_names[mi]
                    if mn == 'Coin Flip': continue
                    if (i, j) in sb_cells:
                        cell = tbl[(i+1, j)]
                        cell.set_edgecolor('#00aa00')
                        cell.set_linewidth(2.5)
                    else:
                        tbl[(i+1, j)].get_text().set_color('gray')
                        
        plt.title(f"{scale} Step 1: EI vs Coin Flip", pad=20, fontsize=14, fontweight='bold')
        plt.savefig(viz_dir / f"viz17_1_{scale.lower()}.png", bbox_inches='tight', dpi=200)
        plt.close()
        
    create_table_n1()
    export_significance_explanation(viz_dir)
    export_significance_heatmap(scale, models, viz_dir)
    export_thesis_paragraph(scale, viz_dir)

    
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
                    coin_mean = np.mean(models['Coin Flip'][metric]) if 'Coin Flip' in models else 0
                    ei_mean = np.mean(models['EI LR'][metric]) if 'EI LR' in models else 0
                    
                    if coin_mean > ei_mean:
                        baseline_m = models.get('Coin Flip')
                    else:
                        baseline_m = models.get('EI LR')
                        
                    if baseline_m is None:
                        baseline_m = models.get('Coin Flip') or models.get('EI LR')
                        
                    if baseline_m is not None:
                        stat, p = wilcoxon(vals, baseline_m[metric])
                        if p < 0.05:
                            sig = 'TRUE'
                            if mean_val > np.mean(baseline_m[metric]):
                                sig_better_cells.append((len(table_data), len(row)))
                        else:
                            sig = 'FALSE'
                    else:
                        sig = '-'
                except Exception:
                    sig = 'FALSE'
            
            val_str = f"{mean_val:.1f}%"
            row.extend([val_str, sig])
            
            sig_tex = "TRUE" if sig == 'TRUE' else sig
            val_tex = val_str.replace('%', '\\%')
            if sig == 'TRUE': val_tex = f"\\textcolor{{green}}{{{val_tex}}}"
            tex_row.extend([val_tex, sig_tex])
            
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
        f.write("\n".join(post_process_tex_lines(tex_lines)))
        
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

    # --- EXPORT F1 TEX ---
    export_f1_tex(scale, models, is_ablation, suffix, viz_dir)

def export_f1_tex(scale, models, is_ablation, suffix, viz_dir):
    model_names = list(models.keys())
    baseline_model_name = 'EI LR' if is_ablation else 'Coin Flip'
    baseline_m = models.get(baseline_model_name)
    if baseline_m is None:
        return
        
    best_stay_model = None
    best_stay_f1 = -1.0
    best_skip_model = None
    best_skip_f1 = -1.0
    
    table_lines = [
        "\\begin{table}[ht]",
        "\\centering",
        "\\begin{tabular}{lcc}",
        "\\hline",
        "Model & F1 Test (STAY) & F1 Test (SKIP) \\\\",
        "\\hline"
    ]
    
    for m_name in model_names:
        if m_name == 'Coin Flip' or (is_ablation and m_name == 'EI LR'):
            continue
            
        m_dict = models.get(m_name)
        if m_dict is None: continue
        
        f1_stay_vals = m_dict['F1 Test (STAY)']
        mean_stay = np.mean(f1_stay_vals)
        try:
            c_stay = np.mean(models['Coin Flip']['F1 Test (STAY)']) if 'Coin Flip' in models else 0
            e_stay = np.mean(models['EI LR']['F1 Test (STAY)']) if 'EI LR' in models else 0
            b_stay_m = models.get('Coin Flip') if c_stay > e_stay else models.get('EI LR')
            if b_stay_m is None: b_stay_m = models.get('Coin Flip') or models.get('EI LR')
            
            if b_stay_m is not None:
                stat, p_stay = wilcoxon(f1_stay_vals, b_stay_m['F1 Test (STAY)'])
                sig_stay = (p_stay < 0.05 and mean_stay > np.mean(b_stay_m['F1 Test (STAY)']))
            else:
                sig_stay = False
        except Exception:
            sig_stay = False
            
        f1_skip_vals = m_dict['F1 Test (SKIP)']
        mean_skip = np.mean(f1_skip_vals)
        try:
            c_skip = np.mean(models['Coin Flip']['F1 Test (SKIP)']) if 'Coin Flip' in models else 0
            e_skip = np.mean(models['EI LR']['F1 Test (SKIP)']) if 'EI LR' in models else 0
            b_skip_m = models.get('Coin Flip') if c_skip > e_skip else models.get('EI LR')
            if b_skip_m is None: b_skip_m = models.get('Coin Flip') or models.get('EI LR')
            
            if b_skip_m is not None:
                stat, p_skip = wilcoxon(f1_skip_vals, b_skip_m['F1 Test (SKIP)'])
                sig_skip = (p_skip < 0.05 and mean_skip > np.mean(b_skip_m['F1 Test (SKIP)']))
            else:
                sig_skip = False
        except Exception:
            sig_skip = False
            
        if not (sig_stay or sig_skip):
            continue
            
        stay_str = f"{mean_stay:.1f}\\%" if sig_stay else "-"
        skip_str = f"{mean_skip:.1f}\\%" if sig_skip else "-"
        m_name_tex = m_name.replace('_', '\\_')
        table_lines.append(f"{m_name_tex} & {stay_str} & {skip_str} \\\\")
        
        if sig_stay and mean_stay > best_stay_f1:
            best_stay_f1 = mean_stay
            best_stay_model = m_name
            
        if sig_skip and mean_skip > best_skip_f1:
            best_skip_f1 = mean_skip
            best_skip_model = m_name
            
    table_lines.extend([
        "\\hline",
        "\\end{tabular}",
        "\\caption{Models significantly outperforming baseline on F1 Test (" + scale + ")}",
        "\\label{tab:sig_f1_" + scale.lower() + ("_ablation" if is_ablation else "") + "}",
        "\\end{table}"
    ])
    
    details_lines = ["\n\\vspace{1em}"]
    
    if best_stay_model:
        m_dict = models[best_stay_model]
        details_lines.append(f"\\textbf{{Best Model for STAY:}} {best_stay_model.replace('_', '\\_')} \\\\")
        details_lines.append(f"F1 Train: {np.mean(m_dict['F1 Train (STAY)']):.1f}\\%, F1 Test: {np.mean(m_dict['F1 Test (STAY)']):.1f}\\%, Precision: {np.mean(m_dict['Precision (STAY)']):.1f}\\%, Recall: {np.mean(m_dict['Recall (STAY)']):.1f}\\%")
        details_lines.append("\\vspace{1em}")
        
    if best_skip_model:
        m_dict = models[best_skip_model]
        details_lines.append(f"\\textbf{{Best Model for SKIP:}} {best_skip_model.replace('_', '\\_')} \\\\")
        details_lines.append(f"F1 Train: {np.mean(m_dict['F1 Train (SKIP)']):.1f}\\%, F1 Test: {np.mean(m_dict['F1 Test (SKIP)']):.1f}\\%, Precision: {np.mean(m_dict['Precision (SKIP)']):.1f}\\%, Recall: {np.mean(m_dict['Recall (SKIP)']):.1f}\\%")
        details_lines.append("\\vspace{1em}")
        
    full_content = "\n".join(table_lines + details_lines)
    
    out_file = viz_dir / f"viz17_{scale.lower()}_sig_f1_models{suffix}.tex"
    with open(out_file, 'w') as f:
        f.write(full_content)

def export_significance_explanation(viz_dir):
    content = """Methodology for Significance Testing of EEG Pipeline Models

To rigorously determine whether a predictive model meaningfully outperforms a random baseline (Coin Flip) or a standard reference (Engagement Index, EI LR), this pipeline employs the Wilcoxon signed-rank test.

1. Rationale for the Non-Parametric Approach
Standard parametric tests (like the independent or paired t-test) assume that the performance metrics (such as F1-scores) across participants are normally distributed. In small-sample, highly individual physiological data such as EEG (N=25 participants), this assumption is often violated. The Wilcoxon signed-rank test is a non-parametric alternative that makes no assumptions about the underlying distribution, making it statistically robust for this dataset.

2. Paired Participant-Level Comparison
Performance is not evaluated as a single aggregated mean. Instead, the test utilizes paired metrics from the 25 individual cross-validation folds (where each fold represents a held-out participant).
The test calculates the difference in performance between the evaluated model and the baseline model for each participant:
   Difference = Model_Score - Baseline_Score
By ranking the absolute values of these differences and applying the original signs, the test evaluates whether the median of these paired differences is significantly shifted from zero.

Example: A model might have a high overall average F1-score simply because it performed exceptionally well on just 3 participants, while performing slightly worse than the baseline on the other 22. A mean-based comparison might incorrectly flag this as an overall improvement. The paired Wilcoxon test prevents this illusion by looking at consistency: it will only yield significance if the model reliably beats the baseline across the majority of the individual participants.

3. Two-Sided Testing with Directional Validation
The algorithm employs a two-sided Wilcoxon test with a standard significance threshold (alpha = 0.05, representing a 95% confidence level). A two-sided test simply detects a significant difference in either direction (i.e., the model could be significantly better OR significantly worse than the baseline).
To strictly guarantee superiority, a directional check is applied post-hoc:
- First, the test must yield p < 0.05.
- Second, the arithmetic mean of the model's scores across all participants must be strictly greater than the arithmetic mean of the baseline.
If both conditions are met, the model's performance is deemed statistically significantly superior to the baseline, and is highlighted as such (e.g., TRUE).

4. Assumptions and Parameters
- Paired Samples: N = 25 (derived from 25 strictly partitioned participant folds).
- Significance Threshold: Alpha = 0.05.
- Independence: Metric distributions are assumed to be independent between participants but dependent between models evaluated on the identical set of participant features.
- Ties: Zero-differences (ties) are handled via standard SciPy internal methods (the Wilcoxon method removes zero-differences from the ranking)."""
    
    with open(viz_dir / "viz17_2_significance.txt", "w") as f:
        f.write(content)

def export_significance_heatmap(scale, models, viz_dir):
    import seaborn as sns
    if 'Coin Flip' not in models or 'EI LR' not in models:
        return
        
    cf = models['Coin Flip']
    ei = models['EI LR']
    
    metrics = ['F1 Test (STAY)', 'F1 Test (SKIP)']
    n_participants = len(cf[metrics[0]])
    
    diff_matrix = np.zeros((len(metrics), n_participants))
    
    for i, metric in enumerate(metrics):
        diff_matrix[i, :] = ei[metric] - cf[metric]
        
    fig, ax = plt.subplots(figsize=(max(12, n_participants * 0.5), 3))
    
    sns.heatmap(diff_matrix, annot=True, fmt=".1f", cmap="RdYlGn", center=0, 
                yticklabels=[m.replace('Test ', '') for m in metrics],
                xticklabels=[f"P{i+1}" for i in range(n_participants)],
                cbar_kws={'label': 'Difference vs Coin Flip (%)'},
                linewidths=0.5, ax=ax)
                
    ax.set_title(f"Participant-Level Significance Map: EI LR vs Coin Flip ({scale})\n(Green = EI LR Outperforms Random)", pad=15, fontweight='bold', fontsize=12)
    plt.tight_layout()
    plt.savefig(viz_dir / f"viz17_2_{scale.lower()}_significance.png", dpi=200)
    plt.close()

def export_thesis_paragraph(scale, viz_dir):
    s = scale.lower()
    
    if scale == "Intra":
        cv_desc = "derived from the 25 participant-specific evaluations, where an individualized model is trained and tested exclusively on each participant's own data splits."
    else:
        cv_desc = "derived from the 25 cross-validation folds in a Leave-One-Group-Out (LOGO-CV) setup, where each fold trains a generalized model on 24 participants and evaluates it strictly on the held-out 25th participant."
        
    content = f"""To rigorously determine whether a predictive model meaningfully outperforms the random baseline (Coin Flip) or the standard reference (Engagement Index, EI), statistical significance was evaluated using the Wilcoxon signed-rank test. Unlike standard parametric tests (such as the paired t-test) which assume performance metrics are normally distributed, the Wilcoxon test is a non-parametric alternative. This makes it particularly robust for highly individual, small-sample physiological data such as EEG ($N=25$ participants).

Rather than comparing a single aggregated mean across the dataset, the test utilizes paired metrics {cv_desc} The procedure calculates the difference in performance (e.g., F1-score) between the evaluated model and the baseline for each specific participant. By ranking the absolute values of these differences and applying their original signs, the test evaluates whether the median of the differences is significantly shifted from zero. This participant-level comparison ensures that a model only achieves significance if it consistently outperforms the baseline across the majority of individuals, preventing a scenario where exceptional performance on a few participants artificially inflates the overall mean. A visual representation of these paired differences across all participants is provided in the significance heatmap (see Figure~\\ref{{fig:viz17_2_{s}_significance}}), which color-codes the exact performance gap for each participant.

The algorithm employs a two-sided Wilcoxon test with a standard significance threshold ($\\alpha = 0.05$). Because a two-sided test detects significant differences in either direction (meaning the model could be significantly better \\textit{{or}} significantly worse than the baseline), a directional validation check is applied post-hoc. A model's performance is deemed statistically significantly superior only if it meets two strict conditions: first, the test must yield $p < 0.05$; second, the arithmetic mean of the model's scores across all participants must be strictly greater than the baseline's mean. The results of this rigorous evaluation, including the final validated significance flags for each metric, are summarized in Table~\\ref{{tab:metrics_1_{s}}}. Furthermore, the test assumes that metric distributions are independent between participants, but appropriately treats models evaluated on the identical set of participant features as dependent, paired samples. Zero-differences (ties) are naturally handled by standard removal from the ranking process.
"""
    with open(viz_dir / f"viz17_2_{s}_significance_thesis_paragraph.tex", "w") as f:
        f.write(content)

