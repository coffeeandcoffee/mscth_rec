#!/usr/bin/env python3
"""
viz03b_labeling_strategy_viz.py — Labeling-strategy schematics for the thesis.

Replaces the previous two-step flow (a 19-panel composite PNG, cropped by
split_image.py into viz03_exploration/viz03_exploration_N.png). The composite's
generator was never in the repository, so those figures could not be regenerated.
This module draws the panels directly from the exploration window data.

Each panel shows the same 70-second excerpt from one participant under one
labeling parameterisation: STAY windows in blue, SKIP windows in red, the shaded
imminent-skip regions, and the registered A-keypresses as vertical red lines.

Output file names are kept identical to the ones thesis.tex already references.
"""

import pickle
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

import config
import viz_style

PALETTE = sns.color_palette("pastel")
STAY_COLOR = PALETTE[0]
SKIP_COLOR = PALETTE[3]

# Participant and excerpt used by the figures in the thesis.
REP_PID = 4
EXCERPT_S = 70.0

# (output name, universe folder, font size) — the universes match the three
# configurations described in Section "Parameter Sensitivity Analysis".
PANELS = [
    # default parameters: ws = 1s, gs = 2s, isp = 3s
    ("viz03_exploration_9.png",  "univ_a3.0_g2.0_w1.0", viz_style.FONT_3X),
    # reduced window size: ws = 0.3s
    ("viz03_exploration_7.png",  "univ_a3.0_g2.0_w0.3", viz_style.FONT_3X),
    # reduced gap, extended imminent-skip period: gs = 0.5s, isp = 4s
    ("viz03_exploration_12.png", "univ_a4.0_g0.5_w1.0", viz_style.FONT_3X),
]


def _parse_universe(univ_name):
    """univ_a3.0_g2.0_w1.0 -> (area, gap, window) as floats."""
    parts = univ_name.split('_')
    area = float(parts[1][1:])
    gap = float(parts[2][1:])
    window = float(parts[3][1:])
    return area, gap, window


def _draw_panel(ax, wdata, univ_name, font_size):
    area, gap, window = _parse_universe(univ_name)

    windows = wdata['windows']
    a_press_times = wdata.get('a_press_times', [])
    if not windows:
        return False

    t_min = min(w['start_time'] for w in windows)
    t_max = t_min + EXCERPT_S

    # Shaded imminent-skip regions: the isp interval ending `gap` before each
    # press. Overlapping regions are merged first so closely spaced presses do
    # not stack alpha into a darker band.
    spans = []
    for t in a_press_times:
        region_end = t - gap
        region_start = region_end - area
        if region_end < t_min or region_start > t_max:
            continue
        spans.append((region_start - t_min, region_end - t_min))

    merged = []
    for start, end in sorted(spans):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])

    for start, end in merged:
        ax.axvspan(start, end, alpha=0.12, color='red', zorder=0)

    # Labeled windows, drawn as short rules on the STAY / SKIP levels.
    for w in windows:
        if w['start_time'] > t_max or w['end_time'] < t_min:
            continue
        y = 1 if w['label'] == 1 else 0
        color = STAY_COLOR if w['label'] == 1 else SKIP_COLOR
        ax.hlines(y - 0.15, w['start_time'] - t_min, w['end_time'] - t_min,
                  colors=[color], linewidth=6, zorder=2)

    # Registered A-keypresses.
    drawn_label = False
    for t in a_press_times:
        t_rel = t - t_min
        if 0 <= t_rel <= EXCERPT_S:
            ax.axvline(t_rel, color='crimson', linewidth=1.6, zorder=3,
                       label='A-keypress' if not drawn_label else None)
            drawn_label = True

    ax.set_yticks([0, 1])
    ax.set_yticklabels(['SKIP', 'STAY'])
    ax.set_ylim(-0.6, 1.4)
    ax.set_xlim(-1, EXCERPT_S + 1)
    if drawn_label:
        ax.legend(loc='upper right')

    viz_style.style_axes(
        ax, font_size,
        title=f'P{REP_PID} | {univ_name} | '
              f'Window={window}s, Gap={gap}s, Area={area}s',
        xlabel='Time (s)',
        ylabel='Class Label',
    )
    return True


def run(run_dir, params):
    exploration_dir = run_dir / "exploration"
    if not exploration_dir.exists():
        print("  viz03b: exploration/ not found — skipping labeling-strategy panels.")
        return

    out_dir = run_dir / "viz" / "viz03_exploration"
    out_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    for out_name, univ_name, font_size in PANELS:
        pkl = exploration_dir / univ_name / "windows" / "primary" / f"P{REP_PID}.pkl"
        if not pkl.exists():
            print(f"  viz03b: {pkl.relative_to(run_dir)} not found — skipping {out_name}.")
            continue

        with open(pkl, 'rb') as f:
            wdata = pickle.load(f)

        fig, ax = plt.subplots(figsize=(20, 6))
        if _draw_panel(ax, wdata, univ_name, font_size):
            plt.tight_layout()
            plt.savefig(out_dir / out_name, dpi=200, bbox_inches='tight')
            written += 1
        plt.close()

    print(f"  viz03b: wrote {written} labeling-strategy panel(s) to viz/viz03_exploration/.")


if __name__ == "__main__":
    print("Use run.py to execute the pipeline.")
