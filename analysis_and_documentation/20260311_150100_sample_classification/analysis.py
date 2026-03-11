#!/usr/bin/env python3
"""
Sample Classification & Counting — Step 5
============================================
Per included participant, per sub-recording:
  - Only data between first and last A press
  - Classify: 3s before each A = "about_to_skip"; gaps > 3s = "not_about_to_skip"
  - Consecutive A presses < 3s apart merge into one "about_to_skip" block
  - Calculate sample counts per class block using configurable window + overlap
  - Report per-participant totals and class balance

Outputs:
  sample_classification.json            — full block-level detail
  images/sample_classification_table.png — publication-ready summary
  description.txt                        — findings summary
"""

from __future__ import annotations

import csv
import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Config (changeable parameters)
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent.parent / "data"
MASK_JSON = SCRIPT_DIR.parent / "20260311_145900_exclusion_mask" / "exclusion_mask.json"
IMAGES_DIR = SCRIPT_DIR / "images"
IMAGES_DIR.mkdir(exist_ok=True)

PRE_SKIP_WINDOW_S = 3.0    # seconds before A press classified as about_to_skip
SAMPLE_LENGTH_S = 3.0       # sample block duration for ML
OVERLAP = 0.80              # 80% overlap → stride = 0.6s


# ---------------------------------------------------------------------------
# Sample count formula
# ---------------------------------------------------------------------------
def count_samples(block_duration: float,
                  sample_length: float = SAMPLE_LENGTH_S,
                  overlap: float = OVERLAP) -> int:
    """
    Number of samples from a class block of given duration.
    stride = sample_length * (1 - overlap)
    n = floor((duration - sample_length) / stride) + 1  if duration >= sample_length
    n = 0  otherwise
    """
    if block_duration < sample_length:
        return 0
    stride = sample_length * (1 - overlap)
    return int(math.floor((block_duration - sample_length) / stride)) + 1


# ---------------------------------------------------------------------------
# Classification logic
# ---------------------------------------------------------------------------
@dataclass
class ClassBlock:
    label: str          # "about_to_skip" or "not_about_to_skip"
    start_t: float
    end_t: float
    duration_s: float = 0.0
    n_samples: int = 0

    def __post_init__(self):
        self.duration_s = self.end_t - self.start_t
        self.n_samples = count_samples(self.duration_s)


def classify_subrecording(a_timestamps: List[float]) -> List[ClassBlock]:
    """
    Given sorted A-press timestamps for one sub-recording,
    produce alternating class blocks between first_A and last_A.

    Logic for each consecutive pair (t_i, t_{i+1}):
      gap = t_{i+1} - t_i
      if gap <= PRE_SKIP_WINDOW_S:
        entire interval is about_to_skip
      else:
        [t_i, t_{i+1} - PRE_SKIP_WINDOW_S] is not_about_to_skip
        [t_{i+1} - PRE_SKIP_WINDOW_S, t_{i+1}] is about_to_skip
    Adjacent same-label blocks get merged.
    """
    if len(a_timestamps) < 2:
        return []

    raw_blocks: List[ClassBlock] = []

    for i in range(len(a_timestamps) - 1):
        t_curr = a_timestamps[i]
        t_next = a_timestamps[i + 1]
        gap = t_next - t_curr

        if gap <= PRE_SKIP_WINDOW_S:
            raw_blocks.append(ClassBlock("about_to_skip", t_curr, t_next))
        else:
            raw_blocks.append(ClassBlock("not_about_to_skip", t_curr, t_next - PRE_SKIP_WINDOW_S))
            raw_blocks.append(ClassBlock("about_to_skip", t_next - PRE_SKIP_WINDOW_S, t_next))

    # Merge adjacent same-label blocks
    if not raw_blocks:
        return []

    merged: List[ClassBlock] = [raw_blocks[0]]
    for blk in raw_blocks[1:]:
        if blk.label == merged[-1].label:
            merged[-1] = ClassBlock(blk.label, merged[-1].start_t, blk.end_t)
        else:
            merged.append(blk)

    return merged


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------
def get_a_timestamps(csv_path: Path) -> List[float]:
    timestamps = []
    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            return []
        for row in reader:
            try:
                if int(float(row.get("keypress_A", "0"))) == 1:
                    ts = float(row.get("timestamp", "0"))
                    if ts > 0:
                        timestamps.append(ts)
            except (ValueError, TypeError):
                pass
    return sorted(timestamps)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("Sample Classification & Counting — Step 5")
    print("=" * 60)
    print(f"  Pre-skip window:  {PRE_SKIP_WINDOW_S}s")
    print(f"  Sample length:    {SAMPLE_LENGTH_S}s")
    print(f"  Overlap:          {OVERLAP*100:.0f}%")
    print(f"  Stride:           {SAMPLE_LENGTH_S * (1 - OVERLAP):.1f}s\n")

    # Load exclusion mask
    if not MASK_JSON.exists():
        print(f"❌ Exclusion mask not found: {MASK_JSON}")
        return

    mask = json.loads(MASK_JSON.read_text())
    included_pids = set(mask["included"])

    # Collect files per participant
    files = sorted(DATA_DIR.glob("P*_*.csv"), key=lambda p: (
        int(re.search(r"P(\d+)", p.name).group(1)),
        int(re.search(r"_(\d+)\.", p.name).group(1)),
    ))

    pid_files: dict[str, list[Path]] = defaultdict(list)
    for f in files:
        m = re.match(r"(P\d+)_", f.name, re.IGNORECASE)
        if m:
            pid = m.group(1).upper()
            if pid in included_pids:
                pid_files[pid].append(f)

    # --- Process each participant ---
    all_results = {}       # pid -> {sub_recordings, totals}
    participant_totals = []  # for table

    sorted_pids = sorted(pid_files.keys(), key=lambda p: int(p[1:]))

    for pid in sorted_pids:
        print(f"\n{pid}:")
        sub_results = []
        pid_skip_total = 0
        pid_noskip_total = 0
        pid_skip_dur = 0.0
        pid_noskip_dur = 0.0

        for csv_path in pid_files[pid]:
            sub_name = csv_path.stem  # e.g. P4_1
            print(f"  {sub_name} …")

            a_ts = get_a_timestamps(csv_path)
            if len(a_ts) < 2:
                print(f"    < 2 A-presses, skipping")
                sub_results.append({
                    "file": csv_path.name,
                    "a_press_count": len(a_ts),
                    "blocks": [],
                    "skip_samples": 0,
                    "noskip_samples": 0,
                    "skip_duration_s": 0,
                    "noskip_duration_s": 0,
                })
                continue

            blocks = classify_subrecording(a_ts)

            block_data = []
            sub_skip = 0
            sub_noskip = 0
            sub_skip_dur = 0.0
            sub_noskip_dur = 0.0

            for blk in blocks:
                block_data.append({
                    "label": blk.label,
                    "start_t": round(blk.start_t, 3),
                    "end_t": round(blk.end_t, 3),
                    "duration_s": round(blk.duration_s, 3),
                    "n_samples": blk.n_samples,
                })
                if blk.label == "about_to_skip":
                    sub_skip += blk.n_samples
                    sub_skip_dur += blk.duration_s
                else:
                    sub_noskip += blk.n_samples
                    sub_noskip_dur += blk.duration_s

            print(f"    {len(blocks)} blocks → skip={sub_skip} noskip={sub_noskip} samples")

            sub_results.append({
                "file": csv_path.name,
                "a_press_count": len(a_ts),
                "blocks": block_data,
                "skip_samples": sub_skip,
                "noskip_samples": sub_noskip,
                "skip_duration_s": round(sub_skip_dur, 3),
                "noskip_duration_s": round(sub_noskip_dur, 3),
            })

            pid_skip_total += sub_skip
            pid_noskip_total += sub_noskip
            pid_skip_dur += sub_skip_dur
            pid_noskip_dur += sub_noskip_dur

        total_samples = pid_skip_total + pid_noskip_total
        skip_pct = (pid_skip_total / total_samples * 100) if total_samples > 0 else 0

        all_results[pid] = {
            "sub_recordings": sub_results,
            "totals": {
                "skip_samples": pid_skip_total,
                "noskip_samples": pid_noskip_total,
                "total_samples": total_samples,
                "skip_pct": round(skip_pct, 1),
                "noskip_pct": round(100 - skip_pct, 1),
                "skip_duration_s": round(pid_skip_dur, 3),
                "noskip_duration_s": round(pid_noskip_dur, 3),
            },
        }

        participant_totals.append({
            "pid": pid,
            "skip": pid_skip_total,
            "noskip": pid_noskip_total,
            "total": total_samples,
            "skip_pct": skip_pct,
            "skip_dur": pid_skip_dur,
            "noskip_dur": pid_noskip_dur,
        })

    # --- Save JSON ---
    output_json = {
        "parameters": {
            "pre_skip_window_s": PRE_SKIP_WINDOW_S,
            "sample_length_s": SAMPLE_LENGTH_S,
            "overlap": OVERLAP,
            "stride_s": SAMPLE_LENGTH_S * (1 - OVERLAP),
        },
        "participants": all_results,
    }
    json_path = SCRIPT_DIR / "sample_classification.json"
    json_path.write_text(json.dumps(output_json, indent=2))
    print(f"\n✅ Saved: {json_path}")

    # --- Render table ---
    render_table(participant_totals)

    # --- Write description ---
    write_description(participant_totals)

    print("\n✅ Done! Check sample_classification.json, images/, and description.txt")


# ---------------------------------------------------------------------------
# Table
# ---------------------------------------------------------------------------
def render_table(totals: list[dict]):
    col_labels = [
        "P",
        "Skip\nDur (s)",
        "NoSkip\nDur (s)",
        "Skip\nSamples",
        "NoSkip\nSamples",
        "Total\nSamples",
        "Skip\n%",
        "NoSkip\n%",
    ]

    rows = []
    colors = []

    for t in totals:
        skip_pct = t["skip_pct"]

        row = [
            t["pid"],
            f"{t['skip_dur']:.1f}",
            f"{t['noskip_dur']:.1f}",
            str(t["skip"]),
            str(t["noskip"]),
            str(t["total"]),
            f"{skip_pct:.1f}",
            f"{100 - skip_pct:.1f}",
        ]
        rows.append(row)

        # Color: imbalance highlighting
        # Green if 40-60%, yellow if 30-40% or 60-70%, red if <30% or >70%
        if 40 <= skip_pct <= 60:
            bg = "#d4edda"  # green — balanced
        elif 30 <= skip_pct <= 70:
            bg = "#fff3cd"  # yellow — moderate imbalance
        else:
            bg = "#f8d7da"  # red — severe imbalance
        colors.append([bg] * len(col_labels))

    # Aggregate row
    total_skip = sum(t["skip"] for t in totals)
    total_noskip = sum(t["noskip"] for t in totals)
    grand_total = total_skip + total_noskip
    agg_pct = (total_skip / grand_total * 100) if grand_total > 0 else 0

    rows.append([
        "TOTAL",
        f"{sum(t['skip_dur'] for t in totals):.1f}",
        f"{sum(t['noskip_dur'] for t in totals):.1f}",
        str(total_skip),
        str(total_noskip),
        str(grand_total),
        f"{agg_pct:.1f}",
        f"{100 - agg_pct:.1f}",
    ])
    colors.append(["#e8f0fe"] * len(col_labels))

    n_rows = len(rows)
    fig_height = max(5, 0.40 * n_rows + 2)
    fig, ax = plt.subplots(figsize=(14, fig_height))
    ax.axis("off")
    ax.set_title(
        f"Sample Classification — Per Participant (n={len(totals)})\n"
        f"[window={SAMPLE_LENGTH_S}s, overlap={OVERLAP*100:.0f}%, "
        f"pre-skip={PRE_SKIP_WINDOW_S}s]",
        fontsize=12, fontweight="bold", pad=20,
    )

    table = ax.table(
        cellText=rows,
        colLabels=col_labels,
        cellColours=colors,
        colColours=["#4472c4"] * len(col_labels),
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.4)

    for ci in range(len(col_labels)):
        cell = table[0, ci]
        cell.set_text_props(color="white", fontweight="bold", fontsize=8)
        cell.set_edgecolor("white")

    fig.tight_layout()
    out = IMAGES_DIR / "sample_classification_table.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"✅ Saved table: {out}")


# ---------------------------------------------------------------------------
# Description
# ---------------------------------------------------------------------------
def write_description(totals: list[dict]):
    total_skip = sum(t["skip"] for t in totals)
    total_noskip = sum(t["noskip"] for t in totals)
    grand_total = total_skip + total_noskip
    agg_pct = (total_skip / grand_total * 100) if grand_total > 0 else 0
    stride = SAMPLE_LENGTH_S * (1 - OVERLAP)

    lines = []
    lines.append("=" * 70)
    lines.append("SAMPLE CLASSIFICATION & COUNTING — EEG TikTok Study")
    lines.append("=" * 70)
    lines.append("")
    lines.append("OBJECTIVE")
    lines.append("---------")
    lines.append("Classify EEG data between first and last A-press into")
    lines.append("'about_to_skip' and 'not_about_to_skip' class blocks,")
    lines.append("then count available ML samples per class per participant.")
    lines.append("")
    lines.append("PARAMETERS")
    lines.append("----------")
    lines.append(f"  Pre-skip window:  {PRE_SKIP_WINDOW_S}s (before each A press)")
    lines.append(f"  Sample length:    {SAMPLE_LENGTH_S}s")
    lines.append(f"  Overlap:          {OVERLAP*100:.0f}%")
    lines.append(f"  Stride:           {stride:.1f}s")
    lines.append("")
    lines.append("CLASSIFICATION LOGIC")
    lines.append("--------------------")
    lines.append("  For each consecutive A-press pair (t_i, t_{i+1}):")
    lines.append(f"    gap <= {PRE_SKIP_WINDOW_S}s → entire interval = about_to_skip")
    lines.append(f"    gap >  {PRE_SKIP_WINDOW_S}s → [t_i, t_{{i+1}}-{PRE_SKIP_WINDOW_S}] = not_about_to_skip")
    lines.append(f"                       [t_{{i+1}}-{PRE_SKIP_WINDOW_S}, t_{{i+1}}] = about_to_skip")
    lines.append("  Adjacent same-label blocks are merged.")
    lines.append("")
    lines.append("SAMPLE COUNT FORMULA")
    lines.append("--------------------")
    lines.append(f"  n = floor((block_duration - {SAMPLE_LENGTH_S}) / {stride}) + 1")
    lines.append(f"      if block_duration >= {SAMPLE_LENGTH_S}s, else n = 0")
    lines.append("")
    lines.append("AGGREGATE RESULTS")
    lines.append("-----------------")
    lines.append(f"  Participants:       {len(totals)}")
    lines.append(f"  Total skip samples: {total_skip}")
    lines.append(f"  Total noskip:       {total_noskip}")
    lines.append(f"  Grand total:        {grand_total}")
    lines.append(f"  Overall balance:    {agg_pct:.1f}% skip / {100-agg_pct:.1f}% noskip")
    lines.append("")

    lines.append("PER-PARTICIPANT")
    lines.append("---------------")
    for t in totals:
        lines.append(f"  {t['pid']}: skip={t['skip']} noskip={t['noskip']} "
                     f"total={t['total']} ({t['skip_pct']:.1f}% skip)")

    lines.append("")
    lines.append("COLOR CODING (table)")
    lines.append("  Green:  40-60% skip (balanced)")
    lines.append("  Yellow: 30-40% or 60-70% (moderate imbalance)")
    lines.append("  Red:    <30% or >70% (severe imbalance)")
    lines.append("")
    lines.append("=" * 70)
    lines.append("Generated by analysis.py (Step 5: sample_classification)")
    lines.append("=" * 70)

    out = SCRIPT_DIR / "description.txt"
    out.write_text("\n".join(lines))
    print(f"✅ Saved description: {out}")


if __name__ == "__main__":
    main()
