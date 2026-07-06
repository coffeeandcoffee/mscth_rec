import os
import glob
import re

tex_files = glob.glob("../05-12_data_analysis_and_results/runs/run_20260611_220844/viz/*.tex")

def process_table(lines):
    # Find tabular definition
    for i, line in enumerate(lines):
        if line.startswith(r"\begin{tabular}"):
            break
    else:
        return lines # no table found
        
    tabular_idx = i
    
    # Header rows
    h1_idx = -1
    h2_idx = -1
    for i in range(tabular_idx+1, len(lines)):
        if "Metric &" in lines[i]:
            h1_idx = i
            h2_idx = i + 1
            break
            
    if h1_idx == -1: return lines
    
    # Gather data rows
    data_start = -1
    for i in range(h2_idx+1, len(lines)):
        if "hline" in lines[i]: continue
        if "&" in lines[i]:
            data_start = i
            break
            
    data_end = -1
    for i in range(data_start, len(lines)):
        if "hline" in lines[i] or r"\end{tabular}" in lines[i]:
            data_end = i
            break
            
    # Parse rows to columns
    def parse_row(row_str):
        parts = row_str.split("&")
        if len(parts) == 0: return []
        parts[-1] = parts[-1].replace(r"\\", "")
        return [p.strip() for p in parts]
        
    data_rows = []
    for i in range(data_start, data_end):
        row_str = lines[i]
        if "&" not in row_str:
            data_rows.append(None) # e.g. \hline
        else:
            data_rows.append(parse_row(row_str))
            
    # Determine which Sig columns are all '-'
    num_cols = len(data_rows[0])
    cols_to_remove = []
    for col_idx in range(2, num_cols, 2): # Sig columns are 2, 4, 6...
        is_all_dash = True
        for r in data_rows:
            if r is not None:
                if r[col_idx] != "-":
                    is_all_dash = False
                    break
        if is_all_dash:
            cols_to_remove.append(col_idx)
            
    # Fix the actual cells
    for r in data_rows:
        if r is None: continue
        for col_idx in range(2, num_cols, 2):
            val_col = col_idx - 1
            sig_val = r[col_idx]
            if "\\textcolor{green}{TRUE}" in sig_val:
                r[col_idx] = "TRUE"
                # mark green the value
                r[val_col] = "\\textcolor{green}{" + r[val_col] + "}"
            elif "\\textcolor" in sig_val:
                # clean up any other color in sig
                clean_sig = re.sub(r"\\textcolor\{.*?\}(.*)", r"\1", sig_val).replace("{", "").replace("}", "")
                r[col_idx] = clean_sig

    # Remove columns (backwards to not mess up indices)
    for r in data_rows:
        if r is None: continue
        for col_idx in sorted(cols_to_remove, reverse=True):
            del r[col_idx]
            
    # Reconstruct data rows
    for i, r in enumerate(data_rows):
        if r is not None:
            lines[data_start + i] = " & ".join(r) + " \\\\"
            
    # Fix headers
    h1_parts = lines[h1_idx].split("&")
    h1_parts[-1] = h1_parts[-1].replace(r"\\", "")
    h1_parts = [p.strip() for p in h1_parts]
    
    h2_parts = lines[h2_idx].split("&")
    h2_parts[-1] = h2_parts[-1].replace(r"\\", "")
    h2_parts = [p.strip() for p in h2_parts]
    
    # We mapped col_idx to original headers.
    # Original headers: Metric, Model1(2 cols), Model2(2 cols)
    # The models are at indices 1, 2, 3... in h1_parts
    # Their Sig columns are at 2, 4, 6... in original data columns
    # So Model i (1-indexed) corresponds to original data cols 2*i-1 and 2*i
    
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
            
    lines[h1_idx] = " & ".join(new_h1_parts) + " \\\\"
    lines[h2_idx] = " & ".join(new_h2_parts) + " \\\\"
    
    # Fix tabular definition
    # New number of columns
    if data_rows[0] is not None:
        new_num_cols = len(data_rows[0])
    else:
        new_num_cols = len(new_h2_parts) # fallback
        
    # Assume first is l, rest is c
    new_def = "l" + "c" * (new_num_cols - 1)
    lines[tabular_idx] = re.sub(r"\\begin\{tabular\}\{.*?\}", f"\\\\begin{{tabular}}{{{new_def}}}", lines[tabular_idx])
    
    return lines

for f in tex_files:
    if "thesis_paragraph" in f or "method" in f or "results" in f or "description" in f:
        continue # skip pure text blocks
    with open(f, 'r') as file:
        lines = file.read().splitlines()
        
    try:
        new_lines = process_table(lines)
    except Exception as e:
        print(f"Error processing {f}: {e}")
        continue
        
    with open(f, 'w') as file:
        file.write("\n".join(new_lines) + "\n")
        
print("Tables updated.")
