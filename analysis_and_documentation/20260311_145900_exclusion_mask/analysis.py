#!/usr/bin/env python3
"""
Exclusion Mask — Step 3
========================
Reads the survey CSV, applies a priori exclusion criteria, and saves:
  - exclusion_mask.json  — {excluded_pid: [reasons], ...} + included list
  - images/exclusion_table.png — publication-ready table
  - description.txt — findings summary

Exclusion criteria (a priori):
  1. Diagnosed neurological disorder
  2. Self-reported keypress errors > THRESHOLD (default 10)

Designed to be re-run if more participants are added.
"""

from __future__ import annotations

import csv
import json
import re
from collections import OrderedDict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
SURVEY_CSV = SCRIPT_DIR.parent.parent / "survey_data" / "survey_p4_31.csv"
IMAGES_DIR = SCRIPT_DIR / "images"
IMAGES_DIR.mkdir(exist_ok=True)

WRONG_KEYPRESS_THRESHOLD = 10


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def parse_numeric(val: str) -> float:
    val = val.strip()
    if not val or val.lower() in ("never", "n/a", ""):
        return 0
    m = re.match(r"^(\d+)\s*[-–]\s*(\d+)$", val)
    if m:
        return (int(m.group(1)) + int(m.group(2))) / 2
    m = re.search(r"(\d+)", val)
    if m:
        return float(m.group(1))
    return 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("Exclusion Mask — Step 3")
    print("=" * 60)

    if not SURVEY_CSV.exists():
        print(f"❌ Survey file not found: {SURVEY_CSV}")
        return

    # --- Read survey ---
    rows = []
    with SURVEY_CSV.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames
        for row in reader:
            if not row.get("ID", "").strip():
                continue
            rows.append(row)

    print(f"Loaded {len(rows)} participants from survey.\n")

    # --- Find relevant columns ---
    neuro_col = "Do you have a diagnosed neurological disorder?"
    wrong_swipe_col = None
    wrong_a_col = None
    for h in headers:
        if "swipe without pressing A" in h:
            wrong_swipe_col = h
        elif "press A without swiping" in h:
            wrong_a_col = h

    # --- Apply exclusion criteria ---
    excluded = OrderedDict()   # pid -> [reasons]
    included = []              # list of pids

    for row in rows:
        pid = row["ID"].strip()
        reasons = []

        # Criterion 1: Neurological disorder
        neuro = row.get(neuro_col, "").strip().lower()
        if "yes" in neuro:
            reasons.append("neurological disorder")

        # Criterion 2: Excessive keypress errors
        wrong_swipe = parse_numeric(row.get(wrong_swipe_col, "0"))
        wrong_a = parse_numeric(row.get(wrong_a_col, "0"))
        total_wrong = wrong_swipe + wrong_a
        if total_wrong > WRONG_KEYPRESS_THRESHOLD:
            reasons.append(f"excessive keypress errors ({int(total_wrong)} total)")

        if reasons:
            excluded[pid] = reasons
        else:
            included.append(pid)

    # --- Save JSON ---
    mask = {
        "criteria": {
            "neurological_disorder": "exclude if diagnosed",
            "keypress_errors_threshold": WRONG_KEYPRESS_THRESHOLD,
        },
        "total_recruited": len(rows),
        "total_excluded": len(excluded),
        "total_included": len(included),
        "excluded": {pid: reasons for pid, reasons in excluded.items()},
        "included": included,
    }
    json_path = SCRIPT_DIR / "exclusion_mask.json"
    json_path.write_text(json.dumps(mask, indent=2))
    print(f"✅ Saved: {json_path}")

    # --- Render table ---
    render_table(rows, excluded, included)

    # --- Write description ---
    write_description(rows, excluded, included)

    print("\n✅ Done! Check exclusion_mask.json, images/, and description.txt")


# ---------------------------------------------------------------------------
# Table
# ---------------------------------------------------------------------------
def render_table(rows, excluded, included):
    col_labels = ["Participant", "Status", "Reason"]
    table_rows = []
    table_colors = []

    for row in rows:
        pid = row["ID"].strip()
        if pid in excluded:
            table_rows.append([pid, "EXCLUDED", "; ".join(excluded[pid])])
            table_colors.append(["#f8d7da"] * 3)
        else:
            table_rows.append([pid, "Included", "—"])
            table_colors.append(["#d4edda"] * 3)

    # Summary row
    table_rows.append([
        "TOTAL",
        f"{len(included)} included / {len(excluded)} excluded",
        f"of {len(rows)} recruited",
    ])
    table_colors.append(["#e8f0fe"] * 3)

    n_rows = len(table_rows)
    fig_height = max(4, 0.35 * n_rows + 2)
    fig, ax = plt.subplots(figsize=(10, fig_height))
    ax.axis("off")
    ax.set_title(
        f"Participant Exclusion Mask (n={len(rows)} recruited)",
        fontsize=13, fontweight="bold", pad=20,
    )

    table = ax.table(
        cellText=table_rows,
        colLabels=col_labels,
        cellColours=table_colors,
        colColours=["#4472c4"] * 3,
        loc="center",
        cellLoc="left",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.4)

    col_widths = [0.15, 0.30, 0.55]
    for ci, w in enumerate(col_widths):
        for ri in range(n_rows + 1):
            table[ri, ci].set_width(w)

    for ci in range(3):
        cell = table[0, ci]
        cell.set_text_props(color="white", fontweight="bold")
        cell.set_edgecolor("white")

    fig.tight_layout()
    out = IMAGES_DIR / "exclusion_table.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"✅ Saved table: {out}")


# ---------------------------------------------------------------------------
# Description
# ---------------------------------------------------------------------------
def write_description(rows, excluded, included):
    lines = []
    lines.append("=" * 70)
    lines.append("EXCLUSION MASK — EEG TikTok Study")
    lines.append("=" * 70)
    lines.append("")
    lines.append("OBJECTIVE")
    lines.append("---------")
    lines.append("Define the a priori exclusion mask for all recruited participants.")
    lines.append("This mask is saved as exclusion_mask.json and is consumed by")
    lines.append("downstream analysis scripts to parametrically select included")
    lines.append("participants.")
    lines.append("")
    lines.append("EXCLUSION CRITERIA (A PRIORI)")
    lines.append("-----------------------------")
    lines.append(f"  1. Diagnosed neurological disorder → exclude")
    lines.append(f"  2. Self-reported keypress errors > {WRONG_KEYPRESS_THRESHOLD} → exclude")
    lines.append("")
    lines.append(f"SAMPLE: {len(rows)} recruited, {len(excluded)} excluded, "
                 f"{len(included)} included")
    lines.append("")
    lines.append("EXCLUDED PARTICIPANTS")
    lines.append("---------------------")
    for pid, reasons in excluded.items():
        lines.append(f"  {pid}: {', '.join(reasons)}")
    lines.append("")
    lines.append("INCLUDED PARTICIPANTS")
    lines.append("---------------------")
    lines.append(f"  {', '.join(included)}")
    lines.append("")
    lines.append("=" * 70)
    lines.append("Generated by analysis.py (Step 3: exclusion_mask)")
    lines.append("=" * 70)

    out = SCRIPT_DIR / "description.txt"
    out.write_text("\n".join(lines))
    print(f"✅ Saved description: {out}")


if __name__ == "__main__":
    main()
