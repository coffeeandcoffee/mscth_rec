import pandas as pd
import json
from pathlib import Path
import sys
sys.path.append("scripts")
import viz17_diagnostics
import viz17_5_top_vs_ei
run_dir = Path("runs/run_20260611_220844")
df = pd.read_csv(run_dir / "parallel_universe_metrics.csv")
with open(run_dir / "best_combinations.json") as f:
    best_configs = json.load(f)
for scale in ['Intra', 'Inter']:
    viz17_diagnostics.generate_diagnostics(scale, df, best_configs, run_dir / "viz")
viz17_5_top_vs_ei.run(run_dir, {})
print("Fixed files generated.")
