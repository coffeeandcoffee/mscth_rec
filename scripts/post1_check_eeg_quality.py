#!/usr/bin/env python3
"""
Quick EEG recording quality report.

Usage:
    python check_eeg_quality.py               # analyze most recent EEG CSV
    python check_eeg_quality.py --file path/to/eeg_file.csv
"""

from __future__ import annotations

import argparse
import csv
import re
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


RECORDINGS_DIR = Path(__file__).parent / "recordings"


@dataclass
class ChannelStats:
    """Running stats for a single EEG column."""

    values: List[float]

    @property
    def count(self) -> int:
        return len(self.values)

    def median(self) -> Optional[float]:
        if not self.values:
            return None
        return statistics.median(self.values)

    def mean(self) -> Optional[float]:
        if not self.values:
            return None
        return statistics.fmean(self.values)


def find_latest_recording(directory: Path) -> Optional[Path]:
    """Return newest eeg_*.csv file in directory (searches recursively)."""
    if not directory.exists():
        return None
    
    # Search recursively for raw eeg_*.csv files (exclude processed files)
    candidates = [
        p for p in directory.rglob("eeg_*.csv")
        if not any(marker in p.name for marker in ['_classified', '_preprocessed', '_bands', '_cut'])
    ]
    
    if not candidates:
        return None
    
    # Sort by modification time, return most recent
    candidates = sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def sanitize_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_") or "channel"


def compute_percentiles(values: List[float]) -> Dict[str, float]:
    if not values:
        return {}
    perc_points = [0, 25, 50, 75, 100]
    qs = np.percentile(values, perc_points)
    return {f"{p}%": v for p, v in zip(perc_points, qs)}


def load_stats(csv_path: Path, *, output_graphs: bool) -> None:
    if not csv_path.exists():
        raise FileNotFoundError(f"EEG file not found: {csv_path}")

    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError("EEG file has no header row.")

        columns = [c for c in reader.fieldnames if c not in {"timestamp", "lsl_timestamp"}]
        if not columns:
            raise ValueError("EEG file does not contain channel columns.")

        stats: Dict[str, ChannelStats] = {col: ChannelStats(values=[]) for col in columns}
        channel_times: Dict[str, List[float]] = {col: [] for col in columns}
        channel_values: Dict[str, List[float]] = {col: [] for col in columns}

        total_rows = 0
        first_ts: Optional[float] = None
        last_ts: Optional[float] = None

        for row in reader:
            total_rows += 1

            ts_str = row.get("timestamp")
            if ts_str:
                try:
                    ts = float(ts_str)
                except ValueError:
                    ts = None
            else:
                ts = None

            if ts is not None:
                if first_ts is None:
                    first_ts = ts
                last_ts = ts

            relative_ts: Optional[float] = None
            if ts is not None:
                if first_ts is None:
                    first_ts = ts
                last_ts = ts
                if first_ts is not None:
                    relative_ts = ts - first_ts

            for col in columns:
                val = row.get(col)
                if val is None or val == "":
                    continue
                try:
                    val_float = float(val)
                except ValueError:
                    continue
                stats[col].values.append(val_float)
                if relative_ts is not None:
                    channel_times[col].append(relative_ts)
                    channel_values[col].append(val_float)

        if total_rows == 0:
            print("No samples in file.")
            return

        print("=" * 60)
        print(f"EEG quality summary for: {csv_path.name}")
        print(f"Total samples: {total_rows}")

        if first_ts is not None and last_ts is not None and last_ts > first_ts:
            duration = last_ts - first_ts
            frequency = total_rows / duration
            print(f"Duration: {duration:.2f} s")
            print(f"Approx. sampling rate: {frequency:.2f} Hz")
        else:
            print("Duration: unknown (missing timestamps)")

        print("\nPer-channel stats:")
        header = f"{'Channel':<12}{'Coverage%':>12}{'Median':>15}{'Mean':>15}"
        print(header)
        print("-" * len(header))

        for col in columns:
            coverage = (stats[col].count / total_rows * 100) if total_rows else 0.0
            median = stats[col].median()
            mean = stats[col].mean()
            median_str = f"{median:.4f}" if median is not None else "n/a"
            mean_str = f"{mean:.4f}" if mean is not None else "n/a"
            print(f"{col:<12}{coverage:>12.2f}{median_str:>15}{mean_str:>15}")

        print("=" * 60)

    if output_graphs:
        generate_graphs(csv_path, columns, channel_times, channel_values, stats)


def generate_graphs(
    csv_path: Path,
    columns: List[str],
    times: Dict[str, List[float]],
    values: Dict[str, List[float]],
    stats: Dict[str, ChannelStats],
) -> None:
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        print("Graph flag requested but seaborn/matplotlib are not installed.")
        return

    if not any(values[col] for col in columns):
        print("Cannot generate graphs (no timestamped samples).")
        return

    sns.set_theme(style="whitegrid")
    palette = sns.color_palette("pastel")

    for idx, col in enumerate(columns):
        col_times = times[col]
        col_values = values[col]
        if not col_times or not col_values:
            print(f"Skipping {col}: missing timestamped data.")
            continue

        fig, ax = plt.subplots(figsize=(10, 4))
        color = palette[idx % len(palette)]
        sns.lineplot(x=col_times, y=col_values, ax=ax, color=color, linewidth=1.1)
        ax.set_title(f"{col} amplitude over time")
        ax.set_xlabel("Time (s from start)")
        ax.set_ylabel("Amplitude")

        q_values = compute_percentiles(stats[col].values)
        if q_values:
            quant_text = " | ".join(f"{label}: {val:.4f}" for label, val in q_values.items())
        else:
            quant_text = "No quantiles available"

        fig.text(0.5, 0.02, quant_text, ha="center", fontsize=9)
        fig.tight_layout(rect=[0, 0.05, 1, 1])

        output_path = csv_path.with_name(f"{csv_path.stem}_{sanitize_name(col)}.jpg")
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
        print(f"Saved plot: {output_path.name}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Quick EEG recording quality report.")
    parser.add_argument(
        "--file",
        "-f",
        type=str,
        dest="target_file",
        help="Path to EEG CSV file. If omitted, analyze the most recent recording.",
    )
    parser.add_argument(
        "--graph",
        action="store_true",
        help="Also emit per-channel seaborn plots saved next to the EEG file.",
    )
    args = parser.parse_args()

    if args.target_file:
        csv_path = Path(args.target_file).expanduser()
    else:
        csv_path = find_latest_recording(RECORDINGS_DIR)
        if csv_path is None:
            print(f"No EEG recordings found in {RECORDINGS_DIR}")
            return 1

    try:
        load_stats(csv_path, output_graphs=args.graph)
    except Exception as exc:  # noqa: BLE001 - we want to surface errors to the console
        print(f"Error: {exc}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

