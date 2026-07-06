import re
import os

def fix_script(file_path):
    with open(file_path, "r") as f:
        content = f.read()
    
    # 1) Change \textcolor{green}{TRUE} to TRUE in the Sig column, and highlight the value in green instead.
    # We replace: val_tex = val_str.replace('%', '\\%')
    #          if sig == 'TRUE' and mean_val > np.mean(models['Coin Flip'][metric]):
    #              val_tex = f"\\textcolor{{green}}{{{val_tex}}}"
    # Wait, the current script does:
    # sig_tex = "\\textcolor{green}{TRUE}" if sig == 'TRUE' else sig
    # t_tex_row.extend([val_str.replace('%', '\\%'), sig_tex])
    
    # Let's replace:
    # sig_tex = "\\textcolor{green}{TRUE}" if sig == 'TRUE' else sig
    # with:
    # sig_tex = "TRUE" if sig == "TRUE" else sig
    # val_tex = val_str.replace("%", "\\%")
    # if sig == "TRUE": val_tex = f"\\textcolor{{green}}{{{val_tex}}}"
    # ...
    content = re.sub(
        r"sig_tex = \"\\\\textcolor\{green\}\{TRUE\}\" if sig == 'TRUE' else sig\n\s+t_tex_row\.extend\(\[val_str\.replace\('%', '\\\\%'\), sig_tex\]\)",
        r"""sig_tex = "TRUE" if sig == 'TRUE' else sig
                val_tex = val_str.replace('%', '\\%')
                if sig == 'TRUE': val_tex = f"\\textcolor{{green}}{{{val_tex}}}"
                t_tex_row.extend([val_tex, sig_tex])""",
        content
    )
    
    content = re.sub(
        r"sig_tex = \"\\\\textcolor\{green\}\{TRUE\}\" if sig == 'TRUE' else sig\n\s+tex_row\.extend\(\[val_str\.replace\('%', '\\\\%'\), sig_tex\]\)",
        r"""sig_tex = "TRUE" if sig == 'TRUE' else sig
            val_tex = val_str.replace('%', '\\%')
            if sig == 'TRUE': val_tex = f"\\textcolor{{green}}{{{val_tex}}}"
            tex_row.extend([val_tex, sig_tex])""",
        content
    )
    
    # And fix the baseline sig column to not be generated.
    # Actually, if we just remove the column at generation time:
    # "Coin Flip (Val)", "Coin Flip (Sig)"
    # We can just replace the final string before writing!
    # Let's add a post-processing function right before f.write("\n".join(tex_lines))
    
    post_process_func = """
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
        if "hline" in lines[i] or r"\\end{tabular}" in lines[i]:
            data_end = i
            break
            
    def parse_row(row_str):
        parts = row_str.split("&")
        if len(parts) == 0: return []
        parts[-1] = parts[-1].replace(r"\\\\", "")
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
            lines[data_start + i] = " & ".join(r) + " \\\\\\"
            
    h1_parts = [p.strip() for p in lines[h2_idx-1].replace(r"\\\\", "").split("&")]
    h2_parts = [p.strip() for p in lines[h2_idx].replace(r"\\\\", "").split("&")]
    
    new_h1_parts = [h1_parts[0]]
    new_h2_parts = [h2_parts[0]]
    
    for i in range(1, len(h1_parts)):
        sig_col_idx = i * 2
        m_name = re.sub(r"\\\\multicolumn\\{2\\}\\{c\\}\\{(.*?)\\}", r"\\1", h1_parts[i])
        if sig_col_idx in cols_to_remove:
            new_h1_parts.append(m_name)
            new_h2_parts.append("Val")
        else:
            new_h1_parts.append(h1_parts[i])
            new_h2_parts.append("Val & Sig")
            
    lines[h2_idx-1] = " & ".join(new_h1_parts) + " \\\\\\"
    lines[h2_idx] = " & ".join(new_h2_parts) + " \\\\\\"
    
    new_num_cols = len(new_h2_parts)
    for i, line in enumerate(lines):
        if "\\\\begin{tabular}" in line:
            lines[i] = re.sub(r"\\\\begin\\{tabular\\}\\{.*?\\}", f"\\\\\\\\begin{{tabular}}{{l{'c' * (new_num_cols - 1)}}}", line)
            break
            
    return lines
"""

    if "def post_process_tex_lines" not in content:
        # Add function after imports
        content = content.replace("import numpy as np", "import numpy as np\nimport re\n" + post_process_func)
        
    # Replace the writes
    content = re.sub(r"(f\.write\(\"\\n\"\.join\()([t_]*tex_lines)(\)\))", r"\1post_process_tex_lines(\2)\3", content)
    
    with open(file_path, "w") as f:
        f.write(content)

fix_script("../05-12_data_analysis_and_results/scripts/viz17_diagnostics.py")
fix_script("../05-12_data_analysis_and_results/scripts/viz17_5_top_vs_ei.py")

