#!/usr/bin/env python3
"""
regen_thesis_figures.py — Redraw every figure the thesis embeds, in place.

Why this exists instead of `run.py --resume`:
the figures marked with `% ADD:` in thesis.tex come from the visualisation
modules of steps 01-05 as well as step 17. Deleting step01.done ... step05.done
would re-run preprocessing, windowing, balancing and feature extraction — hours
of compute that also rewrites processed/, windows/, splits/ and features/, i.e.
the data every number in the thesis rests on. There is no need: each viz module
only *reads* that data, so it can be re-run on its own against an existing run
directory. This is the same pattern as scripts/regen_viz03.py and fix_gen.py.

Usage:
    python regen_thesis_figures.py                       # default run below
    python regen_thesis_figures.py run_20260611_220844   # explicit run folder
"""

import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

PIPELINE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PIPELINE_DIR / "scripts"))

DEFAULT_RUN = "run_20260611_220844"

# Modules run in pipeline order. Each exposes run(run_dir, params).
VIZ_MODULES = [
    ("viz01_preprocess_viz",         "viz01.3 — bluetooth dropout map"),
    ("viz02_artifact_flag_viz",      "viz02.1 — artifact rate per participant"),
    ("viz03_label_windows_viz",      "viz03.3, viz03.3.2_*, viz03.3.3_* — dwell times & bursts"),
    ("viz03b_labeling_strategy_viz", "viz03_exploration_7/9/12 — labeling strategy panels"),
    ("viz04_balance_split_viz",      "viz04.1, viz04.2_*, viz04.4, viz04.5 — balancing & splits"),
    ("viz05_feature_engineering_viz","viz05a.A2, viz05a.A3 — statistical moments & EI"),
]


def main():
    run_name = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_RUN
    run_dir = PIPELINE_DIR / "runs" / run_name
    if not run_dir.exists():
        print(f"  Run directory not found: {run_dir}")
        sys.exit(1)

    params = json.load(open(run_dir / "parameters.json"))

    print(f"\n  Regenerating thesis figures in: {run_dir.name}\n" + "-" * 70)

    import importlib
    failures = []

    for module_name, description in VIZ_MODULES:
        print(f"  {module_name:32s} {description}")
        try:
            importlib.import_module(module_name).run(run_dir, params)
        except Exception as exc:
            failures.append((module_name, exc))
            print(f"    FAILED: {exc}")

    # Step 17 regenerates its own figures and every generated .tex table.
    print(f"  {'step17_final_results':32s} viz17_* — rankings, top features, significance")
    try:
        import pandas as pd
        import step17_final_results
        step17_final_results.run(run_dir, params)
    except Exception as exc:
        failures.append(("step17_final_results", exc))
        print(f"    FAILED: {exc}")

    print("-" * 70)
    if failures:
        print(f"  {len(failures)} module(s) FAILED:")
        for name, exc in failures:
            print(f"    {name}: {exc}")
        sys.exit(1)
    print("  All thesis figures regenerated.\n")


if __name__ == "__main__":
    main()
