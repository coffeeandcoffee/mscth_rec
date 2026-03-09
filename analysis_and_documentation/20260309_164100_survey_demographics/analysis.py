#!/usr/bin/env python3
"""
Survey Demographics Summary — Step 2
======================================
Reads survey_data/survey_p4_31.csv, computes frequency distributions for each
survey category, flags exclusions (neurological disorder, excessive wrong
A-presses), and generates a publication-ready summary table.

Outputs:
  images/survey_summary_table.png  — per-category response distributions
  description.txt                  — findings and exclusion decisions
"""

from __future__ import annotations

import csv
import re
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
SURVEY_CSV = SCRIPT_DIR.parent.parent / "survey_data" / "survey_p4_31.csv"
IMAGES_DIR = SCRIPT_DIR / "images"
IMAGES_DIR.mkdir(exist_ok=True)

# Threshold: self-reported wrong A-presses above this → exclude
WRONG_A_THRESHOLD = 10


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------
def parse_numeric(val: str) -> float | None:
    """Try to extract a number from a string like '3', '1-2', '45', or free text."""
    val = val.strip()
    if not val or val.lower() in ("never", "n/a", ""):
        return 0
    # Check for range like '1-2' or '2-3' → take midpoint
    m = re.match(r"^(\d+)\s*[-–]\s*(\d+)$", val)
    if m:
        return (int(m.group(1)) + int(m.group(2))) / 2
    # Check for single number possibly embedded in text
    m = re.search(r"(\d+)", val)
    if m:
        return float(m.group(1))
    return None


def clean_category(val: str) -> str:
    """Normalise a categorical answer."""
    val = val.strip().strip('"')
    # Normalise dash variants
    val = val.replace("–", "–").replace("-", "–")
    return val


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("Survey Demographics Summary — P4 to P31")
    print("=" * 60)

    if not SURVEY_CSV.exists():
        print(f"❌ Survey file not found: {SURVEY_CSV}")
        return

    # --- Read CSV ---
    rows = []
    with SURVEY_CSV.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames
        for row in reader:
            # Skip empty rows
            if not row.get("ID", "").strip():
                continue
            rows.append(row)

    print(f"Loaded {len(rows)} participant rows.\n")

    # --- Map short names to original headers ---
    # We'll identify columns by position/content since headers are long
    col_map = {
        "ID": "ID",
        "Age": "How old are you?",
        "Last social media": "When did you last use social media?",
        "Daily short-form video": "How much time per day do you spend watching short-form videos?",
        "Visual correction": "Do you normally wear visual correction?",
        "Wearing correction": "Are you wearing them during the experiment?",
        "Neurological disorder": "Do you have a diagnosed neurological disorder?",
        "ADHD diagnosis": "Do you have ADHD or attention-related diagnosis?",
        "Substances (6h)": "In the last 6 hours, have you consumed:",
        "Psychoactive medication": "Are you currently taking psychoactive medication?",
        "Sleep last night": "How many hours did you sleep last night?",
        "Alertness": "Before starting, how alert do you feel?",
    }

    # Find the wrong-A and wrong-swipe columns by partial match
    wrong_swipe_col = None
    wrong_a_col = None
    for h in headers:
        if "swipe without pressing A" in h:
            wrong_swipe_col = h
        elif "press A without swiping" in h:
            wrong_a_col = h

    # --- Identify exclusions ---
    excluded = {}  # pid -> [reasons]
    included_rows = []

    for row in rows:
        pid = row["ID"].strip()
        reasons = []

        # Neurological disorder
        neuro = row.get(col_map["Neurological disorder"], "").strip().lower()
        if "yes" in neuro:
            reasons.append("neurological disorder")

        # Wrong A-presses (swipes without A + A without swipes)
        wrong_swipe = parse_numeric(row.get(wrong_swipe_col, "0")) or 0
        wrong_a = parse_numeric(row.get(wrong_a_col, "0")) or 0
        total_wrong = wrong_swipe + wrong_a
        if total_wrong > WRONG_A_THRESHOLD:
            reasons.append(f"excessive keypress errors ({int(total_wrong)} total)")

        if reasons:
            excluded[pid] = reasons
        else:
            included_rows.append(row)

    print(f"Excluded: {len(excluded)} participants")
    for pid, reasons in excluded.items():
        print(f"  {pid}: {', '.join(reasons)}")
    print(f"Included: {len(included_rows)} participants\n")

    # --- Compute distributions for included participants ---
    categories = {}  # short_name -> Counter of answers

    for short_name, col_header in col_map.items():
        if short_name == "ID":
            continue
        counter = Counter()
        for row in included_rows:
            val = clean_category(row.get(col_header, ""))
            if not val:
                val = "Not specified"
            # Split multi-value answers (comma separated)
            parts = [v.strip() for v in val.split(",") if v.strip()]
            for part in parts:
                counter[part] += 1
        categories[short_name] = counter

    # --- Age statistics (numeric) ---
    ages = []
    for row in included_rows:
        try:
            ages.append(int(row.get(col_map["Age"], "0")))
        except ValueError:
            pass
    age_stats = {
        "n": len(ages),
        "mean": float(np.mean(ages)) if ages else 0,
        "std": float(np.std(ages)) if ages else 0,
        "min": min(ages) if ages else 0,
        "max": max(ages) if ages else 0,
        "median": float(np.median(ages)) if ages else 0,
    }

    # --- Render table ---
    render_table(categories, age_stats, len(included_rows), excluded)

    # --- Write description ---
    write_description(categories, age_stats, len(included_rows), excluded, rows)

    print("\n✅ Done! Check images/ and description.txt")


# ---------------------------------------------------------------------------
# Table
# ---------------------------------------------------------------------------
def render_table(categories, age_stats, n_included, excluded):
    """Render a single table with one section per survey category."""

    # Build rows: Category | Response | Count | Percentage
    col_labels = ["Category", "Response", "n", "%"]
    table_rows = []
    table_colors = []

    # Age (numeric summary)
    table_rows.append([
        "Age",
        f"M={age_stats['mean']:.1f}, SD={age_stats['std']:.1f}, "
        f"range {age_stats['min']}–{age_stats['max']}",
        str(age_stats["n"]),
        "—",
    ])
    table_colors.append(["#e8f0fe"] * 4)

    for cat_name, counter in categories.items():
        if cat_name == "Age":
            continue
        # Sort by count descending
        sorted_items = counter.most_common()
        for i, (response, count) in enumerate(sorted_items):
            pct = count / n_included * 100
            bg = "#e8f0fe" if i == 0 else "white"
            # Truncate long responses
            resp_display = response[:50] + "…" if len(response) > 50 else response
            table_rows.append([
                cat_name if i == 0 else "",
                resp_display,
                str(count),
                f"{pct:.0f}",
            ])
            table_colors.append([bg] * 4)

    # Exclusions section
    table_rows.append(["EXCLUDED", "", "", ""])
    table_colors.append(["#f8d7da"] * 4)
    for pid, reasons in excluded.items():
        table_rows.append(["", f"{pid}: {', '.join(reasons)}", "—", "—"])
        table_colors.append(["#f8d7da"] * 4)

    # Render
    n_rows = len(table_rows)
    fig_height = max(6, 0.32 * n_rows + 2)
    fig, ax = plt.subplots(figsize=(14, fig_height))
    ax.axis("off")
    ax.set_title(f"Survey Summary — Included Participants (n={n_included})",
                 fontsize=14, fontweight="bold", pad=20)

    table = ax.table(
        cellText=table_rows,
        colLabels=col_labels,
        cellColours=table_colors,
        colColours=["#4472c4"] * 4,
        loc="center",
        cellLoc="left",
    )

    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.3)

    # Column widths
    col_widths = [0.18, 0.55, 0.08, 0.08]
    for col_idx, w in enumerate(col_widths):
        for row_idx in range(n_rows + 1):
            table[row_idx, col_idx].set_width(w)

    # Header styling
    for col_idx in range(4):
        cell = table[0, col_idx]
        cell.set_text_props(color="white", fontweight="bold", fontsize=9)
        cell.set_edgecolor("white")

    # Center n and % columns
    for row_idx in range(1, n_rows + 1):
        for col_idx in [2, 3]:
            table[row_idx, col_idx]._loc = "center"

    fig.tight_layout()
    out_path = IMAGES_DIR / "survey_summary_table.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"✅ Saved table: {out_path}")


# ---------------------------------------------------------------------------
# Description
# ---------------------------------------------------------------------------
def write_description(categories, age_stats, n_included, excluded, all_rows):
    lines = []
    lines.append("=" * 70)
    lines.append("SURVEY DEMOGRAPHICS SUMMARY — EEG TikTok Study")
    lines.append("=" * 70)
    lines.append("")
    lines.append("OBJECTIVE")
    lines.append("---------")
    lines.append("Summarise participant demographics and pre-experiment state from")
    lines.append("the survey questionnaire. Identify and document exclusions.")
    lines.append("")
    lines.append(f"SAMPLE: {len(all_rows)} recruited, {len(excluded)} excluded, "
                 f"{n_included} included")
    lines.append("")

    lines.append("EXCLUSIONS")
    lines.append("----------")
    for pid, reasons in excluded.items():
        lines.append(f"  {pid}: {', '.join(reasons)}")
    lines.append(f"\n  Exclusion criteria:")
    lines.append(f"    - Diagnosed neurological disorder")
    lines.append(f"    - Self-reported keypress errors > {WRONG_A_THRESHOLD}")
    lines.append("")

    lines.append("AGE")
    lines.append("---")
    lines.append(f"  n={age_stats['n']}, M={age_stats['mean']:.1f}, "
                 f"SD={age_stats['std']:.1f}, "
                 f"Mdn={age_stats['median']:.1f}, "
                 f"range={age_stats['min']}–{age_stats['max']}")
    lines.append("")

    lines.append("RESPONSE DISTRIBUTIONS (included participants)")
    lines.append("-----------------------------------------------")
    for cat_name, counter in categories.items():
        if cat_name == "Age":
            continue
        lines.append(f"\n{cat_name}:")
        for response, count in counter.most_common():
            pct = count / n_included * 100
            lines.append(f"  {response}: {count} ({pct:.0f}%)")

    lines.append("")
    lines.append("=" * 70)
    lines.append("Generated by analysis.py (Step 2: survey_demographics)")
    lines.append("=" * 70)

    out_path = SCRIPT_DIR / "description.txt"
    out_path.write_text("\n".join(lines))
    print(f"✅ Saved description: {out_path}")


if __name__ == "__main__":
    main()
