import pickle
from pathlib import Path
import config

run_dir = Path("/Users/gregorlederer/Documents/MSc Thesis - EEG Neuroscience/Data Recording and Quality Tests/05-12_data_analysis_and_results/runs/run_20260524_220329")
res_dir = run_dir / "results" / "intra"

missing = []
for pid in config.INCLUDED_PARTICIPANTS:
    res_path = res_dir / f"P{pid}_results.pkl"
    if not res_path.exists(): continue
    with open(res_path, 'rb') as f:
        d = pickle.load(f)
    try:
        ng = d['results']['temporal_blocked_no_gap']['full'][0]
        print(f"P{pid}: {ng['recall']:.3f}, {ng['confusion_matrix']}")
    except:
        pass
