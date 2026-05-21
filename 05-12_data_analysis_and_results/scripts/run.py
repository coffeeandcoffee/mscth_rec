#!/usr/bin/env python3
"""
run.py — Single-entry orchestrator for the complete EEG analysis pipeline.

Creates a timestamped run folder, validates demographics, writes parameters.json,
then executes steps 01–15 sequentially with viz after each step.

Usage:
    python run.py                         # Fresh run
    python run.py --resume 20260512_1530  # Resume incomplete run
    python run.py --from-run 20260512_1530 --auto-approve  # Pre-fill params
"""

import argparse
import json
import sys
import os
import importlib
import traceback
from datetime import datetime
from pathlib import Path

# Ensure scripts dir is on path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config

# ──────────────────────────────────────────────
# Step registry
# ──────────────────────────────────────────────
STEPS = [
    ("step01_preprocess",          "viz01_preprocess_viz",          "Preprocessing"),
    ("step02_artifact_flag",       "viz02_artifact_flag_viz",       "Artifact Flagging"),
    ("step03_label_windows",       "viz03_label_windows_viz",       "Window Labelling"),
    ("step04_balance_split",       "viz04_balance_split_viz",       "Balance & Split"),
    ("step05_feature_engineering", "viz05_feature_engineering_viz", "Feature Engineering"),
    ("step06_grid_search",         "viz06_grid_search_viz",         "Grid Search"),
    ("step07_train_intra",         "viz07_train_intra_viz",         "Intra-Subject Training"),
    ("step08_train_ei",            "viz08_train_ei_viz",            "EI Comparison"),
    ("step09_significance",        "viz09_significance_viz",        "Significance Testing"),
    ("step10_ablation_notch",      "viz10_ablation_notch_viz",      "Notch Ablation"),
    ("step11_feature_importance",  "viz11_feature_importance_viz",  "Feature Importance"),
    ("step12_robustness_subgroups","viz12_robustness_subgroups_viz","Subgroup Robustness"),
    ("step13_sensitivity_manifests","viz13_sensitivity_manifests_viz","Sensitivity Analysis"),
    ("step14_logo_cv",             "viz14_logo_cv_viz",             "LOGO-CV"),
    ("step15_compile_report",      "viz15_compile_report_viz",      "Report Compilation"),
]


def validate_demographics():
    """Validate demographics CSV against participant list. Halt on mismatch."""
    import pandas as pd

    if not config.SURVEY_CSV.exists():
        print(f"  ✗ Demographics CSV not found: {config.SURVEY_CSV}")
        sys.exit(1)

    df = pd.read_csv(config.SURVEY_CSV)
    if 'ID' not in df.columns:
        print("  ✗ Demographics CSV missing 'ID' column.")
        sys.exit(1)

    survey_ids = set()
    for val in df['ID'].dropna():
        s = str(val).strip()
        if s.startswith('P'):
            try:
                survey_ids.add(int(s[1:]))
            except ValueError:
                pass

    pipeline_ids = set(config.INCLUDED_PARTICIPANTS)
    missing = pipeline_ids - survey_ids
    if missing:
        print(f"  ✗ Participants missing from demographics CSV: {sorted(missing)}")
        sys.exit(1)

    print(f"  ✓ Demographics validated: {len(pipeline_ids)} participants matched.")


def validate_raw_data():
    """Check that all participant session folders exist and have CSVs."""
    missing = []
    for pid in config.INCLUDED_PARTICIPANTS:
        session_dir = config.DATA_DIR / config.SESSION_MAP[pid]
        if not session_dir.exists():
            missing.append(pid)
            continue
        csvs = list(session_dir.glob("*.csv"))
        if not csvs:
            missing.append(pid)

    if missing:
        print(f"  ✗ Raw data missing for participants: {sorted(missing)}")
        sys.exit(1)

    print(f"  ✓ Raw data validated: {len(config.INCLUDED_PARTICIPANTS)} session folders found.")


def create_run_dir(timestamp=None):
    """Create timestamped run directory."""
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = config.PIPELINE_DIR / "runs" / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir, timestamp


def write_parameters(run_dir, params, auto_approve=False):
    """Write parameters.json and pause for user approval."""
    params_file = run_dir / "parameters.json"
    with open(params_file, 'w') as f:
        json.dump(params, f, indent=2)

    print(f"\n  Parameters written to: {params_file}")
    print(f"  Review parameters.json before proceeding.\n")

    if not auto_approve:
        response = input("  Approve parameters and start pipeline? [y/N]: ").strip().lower()
        if response != 'y':
            print("  Pipeline aborted by user.")
            sys.exit(0)
    else:
        print("  --auto-approve: Skipping approval pause.")


def load_params(run_dir):
    """Load parameters.json from run directory."""
    params_file = run_dir / "parameters.json"
    with open(params_file, 'r') as f:
        return json.load(f)


def step_done(run_dir, step_num):
    """Check if a step is already completed."""
    return (run_dir / f"step{step_num:02d}.done").exists()


def mark_done(run_dir, step_num):
    """Mark a step as completed."""
    (run_dir / f"step{step_num:02d}.done").touch()


def run_step(module_name, run_dir, params):
    """Import and execute a step module's run() function."""
    mod = importlib.import_module(module_name)
    return mod.run(run_dir, params)


def main():
    parser = argparse.ArgumentParser(description='EEG Pipeline Orchestrator')
    parser.add_argument('--resume', type=str, default=None,
                        help='Timestamp of run to resume (e.g. 20260512_1530)')
    parser.add_argument('--from-run', type=str, default=None,
                        help='Timestamp of prior run to pre-fill parameters from')
    parser.add_argument('--auto-approve', action='store_true',
                        help='Skip parameter approval pause')
    parser.add_argument('--step', action='store_true',
                        help='Execute only the next incomplete step, then stop')
    parser.add_argument('--restep', action='store_true',
                        help='Repeat the last completed step, then stop')
    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("  EEG ANALYSIS PIPELINE — STAY vs SKIP")
    print("  MSc Thesis — Corrected & Complete")
    print("=" * 70)

    # ── Validate inputs ──
    print("\n  Validating inputs...")
    validate_demographics()
    validate_raw_data()

    # ── Set up run directory ──
    if args.resume:
        run_dir = config.PIPELINE_DIR / "runs" / f"run_{args.resume}"
        if not run_dir.exists():
            print(f"  ✗ Run directory not found: {run_dir}")
            sys.exit(1)
        params = load_params(run_dir)
        timestamp = args.resume
        print(f"\n  Resuming run: {timestamp}")
    else:
        run_dir, timestamp = create_run_dir()
        if args.from_run:
            prior_dir = config.PIPELINE_DIR / "runs" / f"run_{args.from_run}"
            if not prior_dir.exists():
                print(f"  ✗ Prior run not found: {prior_dir}")
                sys.exit(1)
            params = load_params(prior_dir)
            print(f"  Pre-filled parameters from: {args.from_run}")
        else:
            params = config.DEFAULT_PARAMS.copy()

        write_parameters(run_dir, params, auto_approve=args.auto_approve)
        print(f"\n  Starting run: {timestamp}")
        print(f"  Run directory: {run_dir}")

    # ── Handle --restep ──
    if args.restep:
        last_done = 0
        for i in range(1, len(STEPS) + 1):
            if step_done(run_dir, i):
                last_done = i
        if last_done > 0:
            print(f"\n  --restep: Un-marking step {last_done:02d} to repeat it.")
            done_file = run_dir / f"step{last_done:02d}.done"
            if done_file.exists():
                done_file.unlink()
            args.step = True
        else:
            print("\n  --restep: No completed steps found to repeat.")

    # ── Execute pipeline ──
    for i, (step_module, viz_module, step_name) in enumerate(STEPS, start=1):
        if step_done(run_dir, i):
            print(f"\n  Step {i:02d} [{step_name}] — SKIPPED (already complete)")
            continue

        print(f"\n{'─'*70}")
        print(f"  Step {i:02d} — {step_name}")
        print(f"{'─'*70}")

        try:
            run_step(step_module, run_dir, params)
        except Exception as e:
            print(f"\n  ✗ Step {i:02d} FAILED: {e}")
            traceback.print_exc()
            print(f"\n  Resume with: python run.py --resume {timestamp}")
            sys.exit(1)

        # Run visualization
        try:
            run_step(viz_module, run_dir, params)
            print(f"  ✓ Visualization {i:02d} complete.")
        except Exception as e:
            print(f"  ⚠ Visualization {i:02d} failed (non-fatal): {e}")

        mark_done(run_dir, i)
        print(f"  ✓ Step {i:02d} [{step_name}] — COMPLETE")

        if args.step:
            flag_used = "--restep" if args.restep else "--step"
            print(f"\n  {flag_used}: Stopping after step {i:02d}.")
            print(f"  Resume with: ../venv/bin/python3 run.py --resume {timestamp}")
            break

    print(f"\n{'='*70}")
    print(f"  PIPELINE COMPLETE — {timestamp}")
    print(f"  Results: {run_dir}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
