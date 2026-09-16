"""Shared helpers for the thesis text-analysis scripts in this folder.

Every script here reads ../thesis.tex by default and agrees on what counts
as "the live thesis": the document body, with % comments and
\\iffalse ... \\fi draft blocks removed.

Stdlib only.
"""

import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_TEX = os.path.normpath(os.path.join(HERE, os.pardir, "thesis.tex"))
DEFAULT_BIB = os.path.normpath(os.path.join(HERE, os.pardir, "library.bib"))


def read(path=None):
    """Read a .tex file (defaults to ../thesis.tex)."""
    path = path or DEFAULT_TEX
    with open(path, encoding="utf-8", errors="replace") as fh:
        return fh.read()


def body_only(raw):
    """Everything between \\begin{document} and \\end{document}."""
    m = re.search(r"\\begin\{document\}", raw)
    body = raw[m.end():] if m else raw
    return re.sub(r"\\end\{document\}.*\Z", "", body, flags=re.S)


def strip_comments(text):
    """Drop % comments, keeping escaped \\% intact."""
    return re.sub(r"(?<!\\)%[^\n]*", "", text)


def split_iffalse(text):
    """Return (live, disabled): text outside vs. inside \\iffalse ... \\fi."""
    live, disabled, last = [], [], 0
    for m in re.finditer(r"\\iffalse\b.*?\\fi\b", text, flags=re.S):
        live.append(text[last:m.start()])
        disabled.append(m.group(0))
        last = m.end()
    live.append(text[last:])
    return "".join(live), "".join(disabled)


def live_and_disabled(raw):
    """Convenience: body, comments stripped, split into live and draft text."""
    return split_iffalse(strip_comments(body_only(raw)))


def relname(path):
    """Short path for printing in headers."""
    try:
        return os.path.relpath(path, HERE)
    except ValueError:
        return path
