#!/usr/bin/env python3
"""Quick re-run of viz03 only, using existing run data."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import viz03_label_windows_viz
import config

run_dir = Path(__file__).resolve().parent.parent / "runs" / "run_20260519_073332"
params = __import__('json').load(open(run_dir / "parameters.json"))
viz03_label_windows_viz.run(run_dir, params)
print("Done — check viz/viz03_label_windows.png")
