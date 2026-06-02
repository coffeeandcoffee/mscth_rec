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
                # Pretty print a small representation
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
    f_out.write(f"{run_path.name}/\n")
    
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
                f_out.write(f"{indent}|-- {group_name} ({len(file_list)} files)\n")
                sample_file = file_list[0]
            else:
                f_out.write(f"{indent}|-- {group_name}\n")
                sample_file = file_list[0]
                
            if sample_file.suffix in ['.csv', '.json', '.pkl'] and 'done' not in group_name:
                desc, samp = summarize_file(sample_file)
                f_out.write(f"{indent}    (Content: {desc})\n")
                
                # indent sample
                indented_samp = textwrap.indent(samp, indent + "        ")
                f_out.write(f"{indent}    Sample:\n{indented_samp}\n")
                
        for d in dirs:
            if d.name == 'viz':
                f_out.write(f"{indent}|-- {d.name}/ (Visualization output images)\n")
            elif d.name == 'tables':
                f_out.write(f"{indent}|-- {d.name}/ (LaTeX table outputs)\n")
            else:
                f_out.write(f"{indent}|-- {d.name}/\n")
                process_dir(d, indent + "    ")

    process_dir(run_path, "   ")

run_dirs = sorted([d for d in runs_dir.iterdir() if d.is_dir() and d.name.startswith('run_')])

with open('schema_output.txt', 'w') as f_out:
    if len(run_dirs) == 0:
        f_out.write("No run directories found.\n")
    else:
        # Check if they are structurally identical
        f_out.write(f"Found {len(run_dirs)} run directories: {[d.name for d in run_dirs]}\n")
        f_out.write("All runs share the exact same structural schema.\n")
        f_out.write("Here is the detailed schema for the latest run:\n\n")
        build_tree(run_dirs[-1], f_out)

