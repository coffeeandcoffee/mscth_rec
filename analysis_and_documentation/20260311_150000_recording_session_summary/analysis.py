#!/usr/bin/env python3
"""
Recording Session Summary — Step 4
====================================
For each INCLUDED participant (reads exclusion_mask.json from Step 3),
computes per-participant recording metrics with pass/fail criteria.

Metrics:
  1. Number of recording files
  2. Bluetooth loss (seconds + %, PASS if < 20%)
  3. Total usable duration (PASS if >= 20 min)
  4. Baseline availability (PASS if 100s uninterrupted baseline exists:
     10s after first B press with no B or A press within 110s of first B)
  5. Total keypress_A count

Outputs:
  images/recording_summary_table.png  — publication-ready table
  description.txt                     — findings summary
"""

from __future__ import annotations

import csv
import json
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
# Config
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent.parent / "data"
MASK_JSON = SCRIPT_DIR.parent / "20260311_145900_exclusion_mask" / "exclusion_mask.json"
IMAGES_DIR = SCRIPT_DIR / "images"
IMAGES_DIR.mkdir(exist_ok=True)

MAX_LOSS_PCT = 20.0       # pass if < this
MIN_DURATION_MIN = 20.0   # pass if >= this
BASELINE_REQUIRED_S = 100  # 100s uninterrupted baseline
BASELINE_OFFSET_S = 10     # starts 10s after first B press

EEG_CHANNELS = ["TP9", "AF7", "AF8", "TP10"]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass
class SubRecording:
    file_name: str
    total_samples: int = 0
    first_ts: Optional[float] = None
    last_ts: Optional[float] = None
    duration_s: float = 0.0
    a_timestamps: List[float] = field(default_factory=list)
    b_timestamps: List[float] = field(default_factory=list)
    keypress_a_count: int = 0


def parse_file(csv_path: Path) -> Optional[SubRecording]:
    match = re.match(r"(P\d+)_(\d+)\.csv", csv_path.name, re.IGNORECASE)
    if not match:
        return None

    sr = SubRecording(file_name=csv_path.name)

    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            return None
        for row in reader:
            sr.total_samples += 1
            ts_str = row.get("timestamp", "")
            ts = None
            if ts_str:
                try:
                    ts = float(ts_str)
                    if sr.first_ts is None:
                        sr.first_ts = ts
                    sr.last_ts = ts
                except ValueError:
                    pass

            try:
                if int(float(row.get("keypress_A", "0"))) == 1:
                    sr.keypress_a_count += 1
                    if ts is not None:
                        sr.a_timestamps.append(ts)
            except (ValueError, TypeError):
                pass
            try:
                if int(float(row.get("keypress_B", "0"))) == 1:
                    if ts is not None:
                        sr.b_timestamps.append(ts)
            except (ValueError, TypeError):
                pass

    if sr.first_ts is not None and sr.last_ts is not None and sr.last_ts > sr.first_ts:
        sr.duration_s = sr.last_ts - sr.first_ts

    return sr


# ---------------------------------------------------------------------------
# Per-participant summary
# ---------------------------------------------------------------------------
@dataclass
class ParticipantSummary:
    pid: str
    n_recordings: int = 0
    total_duration_s: float = 0.0
    lost_seconds: float = 0.0
    pct_lost: float = 0.0
    loss_pass: bool = False
    usable_duration_s: float = 0.0
    duration_pass: bool = False
    baseline_available: bool = False
    total_a_presses: int = 0


def compute_summary(pid: str, subs: list[SubRecording]) -> ParticipantSummary:
    s = ParticipantSummary(pid=pid)
    s.n_recordings = len(subs)
    s.total_duration_s = sum(sr.duration_s for sr in subs)
    s.total_a_presses = sum(sr.keypress_a_count for sr in subs)

    # --- Bluetooth loss ---
    all_a_ts = []
    for sr in subs:
        all_a_ts.extend(sr.a_timestamps)
    all_a_ts.sort()

    lost = 0.0
    for i in range(len(subs) - 1):
        curr_a = subs[i].a_timestamps
        next_a = subs[i + 1].a_timestamps
        if curr_a and next_a:
            gap = next_a[0] - curr_a[-1]
            if gap > 0:
                lost += gap
    s.lost_seconds = lost

    if len(all_a_ts) >= 2:
        total_span = all_a_ts[-1] - all_a_ts[0]
        s.pct_lost = (lost / total_span * 100) if total_span > 0 else 0
    s.loss_pass = s.pct_lost < MAX_LOSS_PCT

    # --- Usable duration ---
    s.usable_duration_s = s.total_duration_s - s.lost_seconds
    s.duration_pass = (s.usable_duration_s / 60) >= MIN_DURATION_MIN

    # --- Baseline check ---
    # Condition: 10s after first B press, no B or A press within 110s of first B
    all_b_ts = []
    for sr in subs:
        all_b_ts.extend(sr.b_timestamps)
    all_b_ts.sort()

    all_events = sorted(all_a_ts + all_b_ts)

    if all_b_ts:
        first_b = all_b_ts[0]
        baseline_start = first_b + BASELINE_OFFSET_S
        baseline_end = first_b + BASELINE_OFFSET_S + BASELINE_REQUIRED_S  # first_b + 110

        # Check: no A or B press in [baseline_start, baseline_end]
        interruption = False
        for ev_ts in all_events:
            if ev_ts <= first_b:
                continue  # skip the first B itself and anything before
            if baseline_start <= ev_ts <= baseline_end:
                interruption = True
                break
            if ev_ts > baseline_end:
                break
        s.baseline_available = not interruption
    else:
        s.baseline_available = False

    return s


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("Recording Session Summary — Step 4")
    print("=" * 60)

    # Load exclusion mask
    if not MASK_JSON.exists():
        print(f"❌ Exclusion mask not found: {MASK_JSON}")
        print("   Run Step 3 (exclusion_mask) first.")
        return

    mask = json.loads(MASK_JSON.read_text())
    included_pids = set(mask["included"])
    print(f"Included participants from mask: {len(included_pids)}\n")

    # Collect all CSVs
    files = sorted(DATA_DIR.glob("P*_*.csv"), key=lambda p: (
        int(re.search(r"P(\d+)", p.name).group(1)),
        int(re.search(r"_(\d+)\.", p.name).group(1)),
    ))

    participant_subs: dict[str, list[SubRecording]] = defaultdict(list)
    for f in files:
        match = re.match(r"(P\d+)_", f.name, re.IGNORECASE)
        if not match:
            continue
        pid = match.group(1).upper()
        if pid not in included_pids:
            continue
        print(f"  Parsing {f.name} …")
        sr = parse_file(f)
        if sr is not None:
            participant_subs[pid].append(sr)

    sorted_pids = sorted(participant_subs.keys(), key=lambda p: int(p[1:]))
    summaries = [compute_summary(pid, participant_subs[pid]) for pid in sorted_pids]

    print(f"\nAnalysed {sum(s.n_recordings for s in summaries)} files "
          f"across {len(summaries)} included participants.")

    render_table(summaries)
    write_description(summaries, mask)

    print("\n✅ Done! Check images/ and description.txt")


# ---------------------------------------------------------------------------
# Table
# ---------------------------------------------------------------------------
def render_table(summaries: list[ParticipantSummary]):
    col_labels = [
        "P",
        "Files",
        "Lost\n(s)",
        "Lost\n(%)",
        "Loss\n<20%",
        "Usable\nDur (min)",
        "Dur\n≥20m",
        "Baseline\n100s",
        "A Press\nCount",
    ]

    rows = []
    colors = []

    for s in summaries:
        loss_str = f"{s.pct_lost:.1f}"
        dur_min = s.usable_duration_s / 60

        row = [
            s.pid,
            str(s.n_recordings),
            f"{s.lost_seconds:.1f}" if s.lost_seconds > 0 else "0",
            loss_str,
            "PASS" if s.loss_pass else "FAIL",
            f"{dur_min:.1f}",
            "PASS" if s.duration_pass else "FAIL",
            "PASS" if s.baseline_available else "FAIL",
            str(s.total_a_presses),
        ]
        rows.append(row)

        # Row color: red if any criterion fails
        any_fail = not (s.loss_pass and s.duration_pass and s.baseline_available)
        bg = "#fff3cd" if any_fail else "white"
        colors.append([bg] * len(col_labels))

    # Summary row
    all_pass = sum(1 for s in summaries if s.loss_pass and s.duration_pass and s.baseline_available)
    rows.append([
        "TOTAL", "", "", "",
        f"{sum(s.loss_pass for s in summaries)}/{len(summaries)}",
        "",
        f"{sum(s.duration_pass for s in summaries)}/{len(summaries)}",
        f"{sum(s.baseline_available for s in summaries)}/{len(summaries)}",
        str(sum(s.total_a_presses for s in summaries)),
    ])
    colors.append(["#e8f0fe"] * len(col_labels))

    n_rows = len(rows)
    fig_height = max(5, 0.40 * n_rows + 2)
    fig, ax = plt.subplots(figsize=(14, fig_height))
    ax.axis("off")
    ax.set_title(
        f"Recording Session Summary — Included Participants (n={len(summaries)})",
        fontsize=13, fontweight="bold", pad=20,
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
    out = IMAGES_DIR / "recording_summary_table.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"✅ Saved table: {out}")


# ---------------------------------------------------------------------------
# Description
# ---------------------------------------------------------------------------
def write_description(summaries: list[ParticipantSummary], mask: dict):
    n = len(summaries)
    n_loss_pass = sum(s.loss_pass for s in summaries)
    n_dur_pass = sum(s.duration_pass for s in summaries)
    n_base_pass = sum(s.baseline_available for s in summaries)
    n_all_pass = sum(1 for s in summaries
                     if s.loss_pass and s.duration_pass and s.baseline_available)
    total_a = sum(s.total_a_presses for s in summaries)
    total_files = sum(s.n_recordings for s in summaries)
    total_dur = sum(s.usable_duration_s for s in summaries)

    lines = []
    lines.append("=" * 70)
    lines.append("RECORDING SESSION SUMMARY — EEG TikTok Study")
    lines.append("=" * 70)
    lines.append("")
    lines.append("OBJECTIVE")
    lines.append("---------")
    lines.append("Characterise each included participant's recording session")
    lines.append("with pass/fail criteria for protocol compliance.")
    lines.append("")
    lines.append("PASS/FAIL CRITERIA")
    lines.append("------------------")
    lines.append(f"  1. Bluetooth loss < {MAX_LOSS_PCT}%")
    lines.append(f"  2. Usable duration >= {MIN_DURATION_MIN} min")
    lines.append(f"  3. Baseline: {BASELINE_REQUIRED_S}s uninterrupted "
                 f"(starting {BASELINE_OFFSET_S}s after first B press)")
    lines.append("")
    lines.append(f"SAMPLE: {n} included participants "
                 f"(from {mask['total_recruited']} recruited)")
    lines.append("")
    lines.append("AGGREGATE RESULTS")
    lines.append("-----------------")
    lines.append(f"  Recording files:     {total_files}")
    lines.append(f"  Total usable time:   {total_dur:.0f}s ({total_dur/60:.1f} min)")
    lines.append(f"  Total A-presses:     {total_a}")
    lines.append(f"  Loss < {MAX_LOSS_PCT}%:        {n_loss_pass}/{n}")
    lines.append(f"  Duration >= {MIN_DURATION_MIN}m:   {n_dur_pass}/{n}")
    lines.append(f"  Baseline 100s:       {n_base_pass}/{n}")
    lines.append(f"  All criteria pass:   {n_all_pass}/{n}")
    lines.append("")

    lines.append("PER-PARTICIPANT")
    lines.append("---------------")
    for s in summaries:
        status = "PASS" if (s.loss_pass and s.duration_pass and s.baseline_available) else "FAIL"
        lines.append(f"  {s.pid}: {s.n_recordings} files, "
                     f"{s.usable_duration_s/60:.1f} min usable, "
                     f"{s.pct_lost:.1f}% lost, "
                     f"baseline={'yes' if s.baseline_available else 'NO'}, "
                     f"{s.total_a_presses} A-presses → {status}")

    # Flag failures
    failures = [s for s in summaries
                if not (s.loss_pass and s.duration_pass and s.baseline_available)]
    if failures:
        lines.append("")
        lines.append("PARTICIPANTS FAILING CRITERIA")
        lines.append("----------------------------")
        for s in failures:
            reasons = []
            if not s.loss_pass:
                reasons.append(f"loss {s.pct_lost:.1f}% >= {MAX_LOSS_PCT}%")
            if not s.duration_pass:
                reasons.append(f"duration {s.usable_duration_s/60:.1f} min < {MIN_DURATION_MIN} min")
            if not s.baseline_available:
                reasons.append("no 100s uninterrupted baseline")
            lines.append(f"  {s.pid}: {', '.join(reasons)}")

    lines.append("")
    lines.append("=" * 70)
    lines.append("Generated by analysis.py (Step 4: recording_session_summary)")
    lines.append("=" * 70)

    out = SCRIPT_DIR / "description.txt"
    out.write_text("\n".join(lines))
    print(f"✅ Saved description: {out}")


if __name__ == "__main__":
    main()
