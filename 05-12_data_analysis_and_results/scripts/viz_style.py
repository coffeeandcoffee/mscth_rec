#!/usr/bin/env python3
"""
viz_style.py — Consistent text sizing for the thesis figures.

Matplotlib's default is 10pt for ticks and axis labels, so FONT_2X / FONT_3X are
simply double and triple that. Figures are included at \\textwidth, and apparent
text size on the page scales with fontsize / figure_width_inches — so callers
should raise the font size while leaving the figure *width* alone. Figure height
may be increased freely to make room; it does not shrink the text on the page.
"""

BASE = 10.0
FONT_2X = 2.0 * BASE   # 20
FONT_3X = 3.0 * BASE   # 30


def style_axes(ax, size, title=None, xlabel=None, ylabel=None):
    """Force every text element belonging to `ax` to one consistent point size.

    Passing title/xlabel/ylabel sets them first; omitting them leaves whatever
    the caller already set in place.
    """
    if title is not None:
        ax.set_title(title)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)

    ax.title.set_fontsize(size)
    ax.xaxis.label.set_fontsize(size)
    ax.yaxis.label.set_fontsize(size)
    ax.tick_params(axis='both', which='major', labelsize=size)

    style_legend(ax.get_legend(), size)
    return ax


def style_legend(legend, size):
    """Match a legend's entries and its title to `size`."""
    if legend is None:
        return
    for text in legend.get_texts():
        text.set_fontsize(size)
    if legend.get_title() is not None:
        legend.get_title().set_fontsize(size)


def style_suptitle(fig, size):
    """Match a figure-level suptitle to `size`."""
    if fig._suptitle is not None:
        fig._suptitle.set_fontsize(size)
