import os
import json
import pickle
import pandas as pd
from pathlib import Path
import re
import textwrap

runs_dir = Path("/Users/gregorlederer/Documents/MSc Thesis - EEG Neuroscience/Data Recording and Quality Tests/05-12_data_analysis_and_results/runs")

def summarize_file(filepath):
    ext = filepath.suffix.lower()
    description = ""
    sample = ""
    try:
        if ext == '.csv':
            df = pd.read_csv(filepath, nrows=3)
            description = f"CSV with {len(df.columns)} columns: {', '.join(df.columns)}"
            sample = df.to_string(index=False)
        elif ext == '.json':
            with open(filepath, 'r') as f:
                data = json.load(f)
            if isinstance(data, dict):
                description = f"JSON Dictionary with top-level keys: {', '.join(list(data.keys()))}"
                sample = json.dumps(data, indent=2)
                lines = sample.split('\n')
                if len(lines) > 8:
                    sample = '\n'.join(lines[:8]) + "\n..."
            else:
                description = f"JSON Array of length {len(data)}"
                sample = str(data)[:200]
        elif ext == '.pkl':
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            if isinstance(data, dict):
                description = f"Pickle Dictionary with top-level keys: {', '.join(list(data.keys()))}"
                repr_str = str(data)
                sample = repr_str[:300] + ("..." if len(repr_str)>300 else "")
            elif isinstance(data, pd.DataFrame):
                description = f"Pickle DataFrame with {len(data.columns)} columns"
                sample = data.head(3).to_string(index=False)
            else:
                description = f"Pickle {type(data).__name__}"
                sample = str(data)[:200]
    except Exception as e:
        description = "Could not read content or parse"
        sample = str(e)
        
    return description, sample

def build_tree(run_path, f_out):
    f_out.write(f"## {run_path.name}/\n\n")
    
    patterns = {
        r'P\d+_(.*)\.pkl': r'P[X]_\1.pkl',
        r'P\d+_(.*)\.json': r'P[X]_\1.json',
        r'P\d+\.pkl': r'P[X].pkl',
        r'P\d+\.json': r'P[X].json',
        r'step\d+\.done': r'step[X].done',
        r'viz\d+_.*\.png': r'viz[X]_*.png'
    }
    
    def process_dir(directory, indent):
        items = sorted(list(directory.iterdir()))
        
        groups = {}
        dirs = []
        for item in items:
            if item.name.startswith('.'): continue
            if item.is_dir():
                dirs.append(item)
            else:
                grouped_name = item.name
                for pat, rep in patterns.items():
                    if re.match(pat, item.name):
                        grouped_name = rep
                        break
                if grouped_name not in groups:
                    groups[grouped_name] = []
                groups[grouped_name].append(item)
                
        for group_name, file_list in groups.items():
            if len(file_list) > 1:
                f_out.write(f"{indent}|-- **{group_name}** ({len(file_list)} files)\n")
                sample_file = file_list[0]
            else:
                f_out.write(f"{indent}|-- **{group_name}**\n")
                sample_file = file_list[0]
                
            if sample_file.suffix in ['.csv', '.json', '.pkl'] and 'done' not in group_name:
                desc, samp = summarize_file(sample_file)
                f_out.write(f"{indent}    *Content:* {desc}\n\n")
                f_out.write(f"{indent}    *Sample:*\n")
                indented_samp = textwrap.indent(samp, indent + "        ")
                f_out.write(f"{indented_samp}\n")
                
                # Add spacing and divider for contrast
                f_out.write(f"\n{indent}    " + "-"*40 + "\n\n")
                
        for d in dirs:
            if d.name == 'viz':
                f_out.write(f"{indent}|-- 📂 **{d.name}/** (Visualization output images)\n")
            elif d.name == 'tables':
                f_out.write(f"{indent}|-- 📂 **{d.name}/** (LaTeX table outputs)\n")
            else:
                f_out.write(f"{indent}|-- 📂 **{d.name}/**\n")
                process_dir(d, indent + "    ")

    process_dir(run_path, "   ")


# The 4 relevant runs identified in the directory
relevant_runs = [
    "run_20260525_115853 +-0.5s",
    "run_20260526_113959 +-0.5s (rerun)",
    "run_20260526_133703 smaller RF",
    "run_20260526_150106 3 True"
]

with open('schema_output.txt', 'w') as f_out:
    for run_name in relevant_runs:
        run_path = runs_dir / run_name
        if run_path.exists():
            build_tree(run_path, f_out)
            f_out.write("\n================================================================================\n\n")
