import json

path = "/Users/gregorlederer/Documents/MSc Thesis - EEG Neuroscience/Data Recording and Quality Tests/05-12_data_analysis_and_results/runs/run_20260524_220329/parameters.json"
with open(path, 'r') as f:
    d = json.load(f)

if "temporal_blocked_no_gap" not in d["step07"]["eval_protocols"]:
    d["step07"]["eval_protocols"].append("temporal_blocked_no_gap")

with open(path, 'w') as f:
    json.dump(d, f, indent=2)

