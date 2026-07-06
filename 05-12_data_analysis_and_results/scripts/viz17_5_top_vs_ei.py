#!/usr/bin/env python3
import json
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
from pathlib import Path
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon
import ast

def run(run_dir, params):
    mapping_file = run_dir / "viz" / "viz17_4_selected_features.json"
    if not mapping_file.exists():
        print(f"  viz17_5: {mapping_file.name} not found.")
        return
        
    with open(mapping_file, 'r') as f:
        selected_features = json.load(f)
        
    feature_names = list(selected_features.keys())
    
    csv_path = run_dir / "parallel_universe_metrics.csv"
    if not csv_path.exists():
        print(f"  viz17_5: {csv_path.name} not found. Ensure step 16 has been run.")
        return
        
    df = pd.read_csv(csv_path)
    viz_dir = run_dir / "viz"
    
    metrics_order = [
        'F1 Train (STAY)', 'F1 Test (STAY)', 'Precision (STAY)', 'Recall (STAY)',
        'F1 Train (SKIP)', 'F1 Test (SKIP)', 'Precision (SKIP)', 'Recall (SKIP)'
    ]
    
    def calc_metrics(cm_str_series):
        f1_stay, f1_skip = [], []
        prec_stay, prec_skip = [], []
        rec_stay, rec_skip = [], []
        
        for s in cm_str_series:
            cm = ast.literal_eval(s)
            tn, fp = cm[0][0], cm[0][1]
            fn, tp = cm[1][0], cm[1][1]
            
            p_stay = tp / (tp + fp) if (tp + fp) > 0 else 0
            r_stay = tp / (tp + fn) if (tp + fn) > 0 else 0
            f_stay = 2 * p_stay * r_stay / (p_stay + r_stay) if (p_stay + r_stay) > 0 else 0
            
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
        
    def extract_all(df_model):
        test_metrics = calc_metrics(df_model['test_cm'])
        train_metrics = calc_metrics(df_model['train_cm'])
        res = test_metrics.copy()
        res['F1 Train (STAY)'] = train_metrics['F1 Test (STAY)']
        res['F1 Train (SKIP)'] = train_metrics['F1 Test (SKIP)']
        return res
    
    def create_comparison_table(scale, feat_name, idx):
        model_name = f"{feat_name}_LR"
        df_scale = df[df['scale'] == scale]
        
        df_ei = df_scale[(df_scale['model'] == 'EI') & 
                         (df_scale['notch'] == 'NoNotch') & 
                         (df_scale['art'] == 'NoArt') & 
                         (df_scale['burst'] == 'NoBurst')].sort_values('pid')
                         
        df_feat = df_scale[(df_scale['model'] == model_name) & 
                           (df_scale['notch'] == 'NoNotch') & 
                           (df_scale['art'] == 'NoArt') & 
                           (df_scale['burst'] == 'NoBurst')].sort_values('pid')
                           
        if df_ei.empty or df_feat.empty:
            print(f"    Missing data for {feat_name} in {scale}.")
            return
            
        res_ei = extract_all(df_ei)
        res_feat = extract_all(df_feat)
            
        t_data = []
        cols = ['Metric', 'EI LR (Value)', 'EI LR (Sig)', 'Proposed Feature (Value)', 'Proposed Feature (Sig)']
        
        t_tex_cols = "c" * 4
        h1 = "\\multicolumn{2}{c}{EI LR} & \\multicolumn{2}{c}{" + feat_name.replace('_', ' ') + "}"
        h2 = "Val & Sig & Val & Sig"
        
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
        
        for metric in metrics_order:
            row = [metric]
            t_tex_row = [metric.replace('%', '\\%')]
            
            ei_vals = res_ei[metric]
            ei_mean = np.mean(ei_vals)
            val_str = f"{ei_mean:.1f}%"
            row.extend([val_str, '-'])
            t_tex_row.extend([val_str.replace('%', '\\%'), '-'])
            
            feat_vals = res_feat[metric]
            feat_mean = np.mean(feat_vals)
            
            try:
                stat, p = wilcoxon(feat_vals, ei_vals)
                if p < 0.05:
                    sig = 'TRUE'
                    if feat_mean > ei_mean:
                        sb_cells.append((len(t_data), 3)) 
                else:
                    sig = 'FALSE'
            except:
                sig = 'FALSE'
                
            val_str2 = f"{feat_mean:.1f}%"
            row.extend([val_str2, sig])
            sig_tex = "TRUE" if sig == 'TRUE' else sig
            val_tex = val_str2.replace('%', '\\%')
            if sig == 'TRUE': val_tex = f"\\textcolor{{green}}{{{val_tex}}}"
            t_tex_row.extend([val_tex, sig_tex])
            
            t_data.append(row)
            t_tex_lines.append(" & ".join(t_tex_row) + " \\\\")
            
        t_tex_lines.extend([
            "\\hline",
            "\\end{tabular}",
            f"\\caption{{Step 5 ({scale}): Proposed Feature ({feat_name.replace('_', ' ')}) vs EI. Significance tested using Wilcoxon signed-rank test against EI (two-sided, $\\alpha=0.05$, N=25).}}",
            "\\label{tab:metrics_5_" + scale.lower() + "_" + str(idx) + "}",
            "\\end{table}"
        ])
        
        with open(viz_dir / f"viz17_5_{scale.lower()}_feat{idx}.tex", 'w') as f:
            f.write("\n".join(post_process_tex_lines(t_tex_lines)))
            
        fw = max(10, len(cols) * 1.5)
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
                    if "EI LR" in cols[j]: continue
                    if (i, j) in sb_cells:
                        cell = tbl[(i+1, j)]
                        cell.set_edgecolor('#00aa00')
                        cell.set_linewidth(2.5)
                    else:
                        tbl[(i+1, j)].get_text().set_color('gray')
                        
        plt.title(f"{scale} Step 5: {feat_name.replace('_', ' ')} vs EI", pad=20, fontsize=14, fontweight='bold')
        plt.savefig(viz_dir / f"viz17_5_{scale.lower()}_feat{idx}.png", bbox_inches='tight', dpi=200)
        plt.close()

    for idx, fn in enumerate(feature_names, 1):
        create_comparison_table("Inter", fn, idx)
        create_comparison_table("Intra", fn, idx)

    print("  viz17_5: Generated exact-styled evaluation tables (1 per feature).")

if __name__ == "__main__":
    import sys
    run(Path(sys.argv[1]), {})
