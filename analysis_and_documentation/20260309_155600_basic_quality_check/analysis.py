#!/usr/bin/env python3
"""
Basic EEG Quality Check — Step 1
=================================
Per-participant summary table (P4–P31) with:
  • Sub-recording count, total samples, duration, sampling rate, channel coverage
  • Gap analysis: lost seconds between sub-recordings, % time lost
  • B-press count and baseline duration (between 2 B-presses, or B→first A)
  • A-press count and inter-skip interval stats (mean, min, max)

Outputs:
  images/quality_summary_table.png   — publication-ready table
  description.txt                    — findings summary
"""

from __future__ import annotations

import csv
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
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent.parent / "data"
IMAGES_DIR = SCRIPT_DIR / "images"
IMAGES_DIR.mkdir(exist_ok=True)

EEG_CHANNELS = ["TP9", "AF7", "AF8", "TP10"]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass
class RecordingInfo:
    """Raw info from a single sub-recording file."""
    participant: str
    sub_idx: int
    file_name: str
    total_samples: int = 0
    first_ts: Optional[float] = None
    last_ts: Optional[float] = None
    duration_s: float = 0.0
    sampling_rate_hz: float = 0.0
    channel_coverage: dict = field(default_factory=dict)
    keypress_a_count: int = 0
    keypress_b_count: int = 0
    a_timestamps: List[float] = field(default_factory=list)
    b_timestamps: List[float] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------
def parse_file(csv_path: Path) -> Optional[RecordingInfo]:
    match = re.match(r"(P\d+)_(\d+)\.csv", csv_path.name, re.IGNORECASE)
    if not match:
        return None

    ri = RecordingInfo(
        participant=match.group(1).upper(),
        sub_idx=int(match.group(2)),
        file_name=csv_path.name,
    )

    channel_counts = {ch: 0 for ch in EEG_CHANNELS}

    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            return None

        for row in reader:
            ri.total_samples += 1

            ts_str = row.get("timestamp", "")
            ts = None
            if ts_str:
                try:
                    ts = float(ts_str)
                    if ri.first_ts is None:
                        ri.first_ts = ts
                    ri.last_ts = ts
                except ValueError:
                    pass

            for ch in EEG_CHANNELS:
                val = row.get(ch, "")
                if val != "":
                    try:
                        float(val)
                        channel_counts[ch] += 1
                    except ValueError:
                        pass

            try:
                if int(float(row.get("keypress_A", "0"))) == 1:
                    ri.keypress_a_count += 1
                    if ts is not None:
                        ri.a_timestamps.append(ts)
            except (ValueError, TypeError):
                pass
            try:
                if int(float(row.get("keypress_B", "0"))) == 1:
                    ri.keypress_b_count += 1
                    if ts is not None:
                        ri.b_timestamps.append(ts)
            except (ValueError, TypeError):
                pass

    if ri.first_ts is not None and ri.last_ts is not None and ri.last_ts > ri.first_ts:
        ri.duration_s = ri.last_ts - ri.first_ts
        ri.sampling_rate_hz = ri.total_samples / ri.duration_s

    for ch in EEG_CHANNELS:
        ri.channel_coverage[ch] = (channel_counts[ch] / ri.total_samples * 100) if ri.total_samples else 0

    return ri


def collect_all() -> dict[str, list[RecordingInfo]]:
    files = sorted(DATA_DIR.glob("P*_*.csv"), key=lambda p: (
        int(re.search(r"P(\d+)", p.name).group(1)),
        int(re.search(r"_(\d+)\.", p.name).group(1)),
    ))
    participant_data: dict[str, list[RecordingInfo]] = defaultdict(list)
    for f in files:
        print(f"  Parsing {f.name} …")
        ri = parse_file(f)
        if ri is not None:
            participant_data[ri.participant].append(ri)
    return participant_data


# ---------------------------------------------------------------------------
# Per-participant aggregation
# ---------------------------------------------------------------------------
@dataclass
class ParticipantSummary:
    pid: str
    n_recordings: int = 0
    total_samples: int = 0
    total_duration_s: float = 0.0
    avg_sampling_rate: float = 0.0
    avg_channel_coverage: float = 0.0
    lost_seconds: float = 0.0
    pct_time_lost: float = 0.0
    total_b_presses: int = 0
    baseline_duration_s: float = 0.0
    baseline_duration_note: str = ""
    total_a_presses: int = 0
    mean_a_interval: float = 0.0
    min_a_interval: float = 0.0
    max_a_interval: float = 0.0


def compute_summary(pid: str, recs: list[RecordingInfo]) -> ParticipantSummary:
    s = ParticipantSummary(pid=pid)
    s.n_recordings = len(recs)
    s.total_samples = sum(r.total_samples for r in recs)
    s.total_duration_s = sum(r.duration_s for r in recs)
    s.avg_sampling_rate = s.total_samples / s.total_duration_s if s.total_duration_s > 0 else 0

    coverages = []
    for r in recs:
        coverages.extend(r.channel_coverage.values())
    s.avg_channel_coverage = float(np.mean(coverages)) if coverages else 0

    # --- Gap / lost time analysis ---
    # Collect all A-press timestamps across sub-recordings, ordered by sub-recording
    # Lost time = gap between last A of sub-recording N and first A of sub-recording N+1
    lost = 0.0
    for i in range(len(recs) - 1):
        curr_a = recs[i].a_timestamps
        next_a = recs[i + 1].a_timestamps
        if curr_a and next_a:
            gap = next_a[0] - curr_a[-1]
            if gap > 0:
                lost += gap
    s.lost_seconds = lost

    # Total span = first A of first sub-recording → last A of last sub-recording
    all_a_ts = []
    for r in recs:
        all_a_ts.extend(r.a_timestamps)
    all_a_ts.sort()

    if len(all_a_ts) >= 2:
        total_span = all_a_ts[-1] - all_a_ts[0]
        s.pct_time_lost = (lost / total_span * 100) if total_span > 0 else 0
    else:
        s.pct_time_lost = 0

    # --- B-press analysis ---
    all_b_ts = []
    for r in recs:
        all_b_ts.extend(r.b_timestamps)
    all_b_ts.sort()
    s.total_b_presses = sum(r.keypress_b_count for r in recs)

    if len(all_b_ts) >= 2:
        s.baseline_duration_s = all_b_ts[-1] - all_b_ts[0]
        s.baseline_duration_note = "B→B"
    elif len(all_b_ts) == 1 and all_a_ts:
        s.baseline_duration_s = abs(all_a_ts[0] - all_b_ts[0])
        s.baseline_duration_note = "B→1st A"
    else:
        s.baseline_duration_s = 0
        s.baseline_duration_note = "n/a"

    # --- A-press interval analysis ---
    s.total_a_presses = sum(r.keypress_a_count for r in recs)

    if len(all_a_ts) >= 2:
        intervals = [all_a_ts[i + 1] - all_a_ts[i] for i in range(len(all_a_ts) - 1)]
        s.mean_a_interval = float(np.mean(intervals))
        s.min_a_interval = float(np.min(intervals))
        s.max_a_interval = float(np.max(intervals))

    return s


# ---------------------------------------------------------------------------
# Table rendering
# ---------------------------------------------------------------------------
def render_table(summaries: list[ParticipantSummary]) -> Path:
    col_labels = [
        "P",
        "Sub\nRecs",
        "Total\nSamples",
        "Duration\n(s)",
        "Rate\n(Hz)",
        "Cov\n(%)",
        "Lost\n(s)",
        "Lost\n(%)",
        "B\nPress",
        "Baseline\nDur (s)",
        "Base\nType",
        "A\nPress",
        "A Intv\nMean (s)",
        "A Intv\nMin (s)",
        "A Intv\nMax (s)",
    ]

    rows = []
    colors = []
    for s in summaries:
        row = [
            s.pid,
            str(s.n_recordings),
            f"{s.total_samples:,}",
            f"{s.total_duration_s:.0f}",
            f"{s.avg_sampling_rate:.1f}",
            f"{s.avg_channel_coverage:.0f}",
            f"{s.lost_seconds:.1f}" if s.lost_seconds > 0 else "0",
            f"{s.pct_time_lost:.1f}" if s.pct_time_lost > 0 else "0",
            str(s.total_b_presses),
            f"{s.baseline_duration_s:.1f}",
            s.baseline_duration_note,
            str(s.total_a_presses),
            f"{s.mean_a_interval:.1f}" if s.total_a_presses >= 2 else "—",
            f"{s.min_a_interval:.1f}" if s.total_a_presses >= 2 else "—",
            f"{s.max_a_interval:.1f}" if s.total_a_presses >= 2 else "—",
        ]
        rows.append(row)

        # Color coding
        c = "white"
        if s.pct_time_lost > 10:
            c = "#fff3cd"  # yellow
        if s.total_a_presses < 20:
            c = "#f8d7da"  # red
        colors.append([c] * len(col_labels))

    n_rows = len(rows)
    fig_height = max(5, 0.42 * n_rows + 2)
    fig, ax = plt.subplots(figsize=(18, fig_height))
    ax.axis("off")
    ax.set_title("EEG Basic Quality Check — Per Participant (P4–P31)",
                 fontsize=14, fontweight="bold", pad=20)

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

    for col_idx in range(len(col_labels)):
        cell = table[0, col_idx]
        cell.set_text_props(color="white", fontweight="bold", fontsize=8)
        cell.set_edgecolor("white")

    for row_idx in range(1, n_rows + 1):
        for col_idx in range(len(col_labels)):
            cell = table[row_idx, col_idx]
            cell.set_edgecolor("#d0d0d0")

    fig.tight_layout()
    out_path = IMAGES_DIR / "quality_summary_table.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"\n✅ Saved table: {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Description
# ---------------------------------------------------------------------------
def write_description(summaries: list[ParticipantSummary]) -> Path:
    lines = []
    lines.append("=" * 70)
    lines.append("BASIC QUALITY CHECK — EEG TikTok Study (P4–P31)")
    lines.append("=" * 70)
    lines.append("")
    lines.append("OBJECTIVE")
    lines.append("---------")
    lines.append("Verify that all participant recordings are complete, correctly")
    lines.append("formatted, and usable for the ML pipeline.")
    lines.append("")
    lines.append("METRICS PER PARTICIPANT")
    lines.append("-----------------------")
    lines.append("  - Sub-recording count, total samples, duration, sampling rate")
    lines.append("  - Channel coverage (%)")
    lines.append("  - Lost seconds: gap between last A-press of one sub-recording")
    lines.append("    and first A-press of the next (Bluetooth reconnection time)")
    lines.append("  - % time lost relative to full recording span (first A → last A)")
    lines.append("  - B-press count and baseline duration")
    lines.append("    (between 2 B-presses, or single B → first A)")
    lines.append("  - A-press count and inter-skip interval (mean, min, max)")
    lines.append("")
    lines.append("COLOR CODING")
    lines.append("  Yellow: >10% time lost")
    lines.append("  Red:    <20 A-presses (insufficient skip data)")
    lines.append("  White:  looks good")
    lines.append("")

    # Aggregate stats
    total_participants = len(summaries)
    total_files = sum(s.n_recordings for s in summaries)
    total_dur = sum(s.total_duration_s for s in summaries)
    total_a = sum(s.total_a_presses for s in summaries)
    total_b = sum(s.total_b_presses for s in summaries)
    avg_lost = float(np.mean([s.pct_time_lost for s in summaries]))

    lines.append("OVERALL SUMMARY")
    lines.append("---------------")
    lines.append(f"  Participants:        {total_participants}")
    lines.append(f"  Recording files:     {total_files}")
    lines.append(f"  Total duration:      {total_dur:.0f}s ({total_dur/60:.1f} min)")
    lines.append(f"  Total A-presses:     {total_a}")
    lines.append(f"  Total B-presses:     {total_b}")
    lines.append(f"  Mean % time lost:    {avg_lost:.1f}%")
    lines.append("")

    # Per-participant
    lines.append("PER-PARTICIPANT SUMMARY")
    lines.append("-----------------------")
    for s in summaries:
        lines.append(f"\n{s.pid}:")
        lines.append(f"  Sub-recordings: {s.n_recordings}")
        lines.append(f"  Duration: {s.total_duration_s:.0f}s ({s.total_duration_s/60:.1f} min)")
        lines.append(f"  Coverage: {s.avg_channel_coverage:.0f}%")
        lines.append(f"  Lost: {s.lost_seconds:.1f}s ({s.pct_time_lost:.1f}%)")
        lines.append(f"  B-presses: {s.total_b_presses}  "
                      f"Baseline: {s.baseline_duration_s:.1f}s ({s.baseline_duration_note})")
        lines.append(f"  A-presses: {s.total_a_presses}  "
                      f"Interval mean={s.mean_a_interval:.1f}s "
                      f"min={s.min_a_interval:.1f}s "
                      f"max={s.max_a_interval:.1f}s")

    # Potential issues
    lines.append("")
    lines.append("POTENTIAL ISSUES")
    lines.append("----------------")
    issues = False
    for s in summaries:
        problems = []
        if s.total_a_presses < 20:
            problems.append(f"only {s.total_a_presses} A-presses")
        if s.pct_time_lost > 10:
            problems.append(f"{s.pct_time_lost:.1f}% time lost")
        if s.total_b_presses == 0:
            problems.append("no B-presses (no baseline)")
        if problems:
            lines.append(f"  {s.pid}: {', '.join(problems)}")
            issues = True
    if not issues:
        lines.append("  None detected.")

    lines.append("")
    lines.append("=" * 70)
    lines.append("Generated by analysis.py (Step 1: basic_quality_check)")
    lines.append("=" * 70)

    out_path = SCRIPT_DIR / "description.txt"
    out_path.write_text("\n".join(lines))
    print(f"✅ Saved description: {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("Basic EEG Quality Check — P4 to P31")
    print("=" * 60)
    print(f"Data directory: {DATA_DIR}\n")

    participant_data = collect_all()

    if not participant_data:
        print("❌ No recordings found!")
        return

    sorted_pids = sorted(participant_data.keys(), key=lambda p: int(p[1:]))
    summaries = [compute_summary(pid, participant_data[pid]) for pid in sorted_pids]

    print(f"\nAnalysed {sum(s.n_recordings for s in summaries)} files "
          f"across {len(summaries)} participants.")

    render_table(summaries)
    write_description(summaries)

    print("\n✅ Done! Check images/ and description.txt")


if __name__ == "__main__":
    main()
