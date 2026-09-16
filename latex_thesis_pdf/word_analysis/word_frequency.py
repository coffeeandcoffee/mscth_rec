#!/usr/bin/env python3
"""Word-frequency report for a LaTeX thesis.

Counts every word in a .tex file and splits each count into two buckets:

  TEXT  -- prose the reader actually reads
  TEX   -- LaTeX scaffolding: command names, labels, citation keys, file
           paths, math, verbatim, the preamble, and % comments

Usage:
    python3 word_frequency.py [../thesis.tex] [-n 100] [--csv out.csv]
                              [--include-iffalse] [--min-len 1]

By default, text inside \\iffalse ... \\fi is excluded from both buckets,
because that content is commented-out drafts that never reach the PDF.
Pass --include-iffalse to count it as prose anyway.

Stdlib only. No dependencies.
"""

import argparse
import csv
import os
import re
import sys
from collections import Counter, defaultdict

import texsource

# ---------------------------------------------------------------------------
# Commands whose *arguments* are machinery, not prose.
# ---------------------------------------------------------------------------
ARG_IS_CODE = {
    "label", "ref", "eqref", "pageref", "autoref", "nameref",
    "cite", "citep", "citet", "parencite", "textcite", "footcite", "nocite",
    "includegraphics", "input", "include", "bibliography", "addbibresource",
    "usepackage", "documentclass", "url", "path", "bibliographystyle",
    "printbibliography", "DeclareFieldFormat", "DeclareBibliographyDriver",
    "newcommand", "renewcommand", "providecommand", "setlength",
    "addcontentsline", "pagenumbering", "setcounter", "hypersetup",
    "geometry", "graphicspath", "definecolor", "clearfield", "clearlist",
    "printfield", "setunit", "iffieldundef", "ifentrytype", "renewbibmacro",
    "DefineBibliographyStrings", "AtEveryBibitem", "vspace", "hspace",
    "begin", "end",
}

# Environments whose whole body is machinery.
CODE_ENVIRONMENTS = {"equation", "equation*", "align", "align*", "eqnarray",
                     "eqnarray*", "gather", "gather*", "multline", "multline*",
                     "verbatim", "lstlisting", "tikzpicture"}

WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9''\u2019_-]*")


def strip_and_collect(pattern, text, sink, flags=0):
    """Remove every match of `pattern` from `text`, appending it to `sink`."""
    out = []
    last = 0
    for m in re.finditer(pattern, text, flags):
        out.append(text[last:m.start()])
        sink.append(m.group(0))
        last = m.end()
    out.append(text[last:])
    return "".join(out)


def match_braces(text, i):
    """Given index i of an opening '{', return index just past its match."""
    depth = 0
    while i < len(text):
        c = text[i]
        if c == "\\":
            i += 2
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return i + 1
        i += 1
    return len(text)


def separate(raw, include_iffalse=False):
    """Split a .tex source into (prose, code, skipped_lines)."""
    code = []

    # 1. Preamble is all machinery.
    m = re.search(r"\\begin\{document\}", raw)
    if m:
        code.append(raw[:m.end()])
        body = raw[m.end():]
    else:
        body = raw
    body = re.sub(r"\\end\{document\}.*\Z", "", body, flags=re.S)

    # 2. Comments: a % that is not escaped, through end of line.
    body = strip_and_collect(r"(?<!\\)%[^\n]*", body, code)

    # 3. \iffalse ... \fi -- drafts that never reach the PDF.
    skipped_lines = 0
    if not include_iffalse:
        kept, last = [], 0
        for mm in re.finditer(r"\\iffalse\b(.*?)\\fi\b", body, flags=re.S):
            kept.append(body[last:mm.start()])
            skipped_lines += mm.group(0).count("\n") + 1
            last = mm.end()
        kept.append(body[last:])
        body = "".join(kept)

    # 4. Math and verbatim-like environments.
    body = strip_and_collect(r"\$\$.*?\$\$", body, code, re.S)
    body = strip_and_collect(r"(?<!\\)\$.*?(?<!\\)\$", body, code, re.S)
    body = strip_and_collect(r"\\\(.*?\\\)", body, code, re.S)
    body = strip_and_collect(r"\\\[.*?\\\]", body, code, re.S)
    for env in CODE_ENVIRONMENTS:
        body = strip_and_collect(
            r"\\begin\{%s\}.*?\\end\{%s\}" % (re.escape(env), re.escape(env)),
            body, code, re.S)
    body = strip_and_collect(r"\\verb\|[^|]*\|", body, code)
    body = strip_and_collect(r"\\verb(.)(.*?)\1", body, code)

    # 5. Walk what is left, peeling off command names and code-arguments.
    prose = []
    i, n = 0, len(body)
    while i < n:
        c = body[i]
        if c != "\\":
            prose.append(c)
            i += 1
            continue
        cmd = re.match(r"\\([A-Za-z@]+)\*?", body[i:])
        if not cmd:                      # \\, \%, \&, \_ ... escapes
            code.append(body[i:i + 2])
            i += 2
            continue
        name = cmd.group(1)
        code.append(cmd.group(0))
        i += cmd.end()
        while i < n and body[i] in " \t":
            i += 1
        # Optional [..] arguments are always machinery.
        while i < n and body[i] == "[":
            close = body.find("]", i)
            close = n if close == -1 else close + 1
            code.append(body[i:close])
            i = close
            while i < n and body[i] in " \t":
                i += 1
        if name in ARG_IS_CODE:
            # Consume every brace group that follows; it is machinery.
            while i < n and body[i] == "{":
                end = match_braces(body, i)
                code.append(body[i:end])
                i = end
                while i < n and body[i] in " \t":
                    i += 1
    return "".join(prose), "\n".join(code), skipped_lines


def tally(text):
    """Return Counter(lowercased word) and the dominant original spelling."""
    counts = Counter()
    spellings = defaultdict(Counter)
    for m in WORD_RE.finditer(text):
        w = m.group(0).strip("-'\u2019_")
        if not w or not w[0].isalpha():
            continue
        key = w.lower()
        counts[key] += 1
        spellings[key][w] += 1
    return counts, spellings


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tex", nargs="?", default=texsource.DEFAULT_TEX)
    ap.add_argument("-n", "--top", type=int, default=100,
                    help="how many rows to print (default 100)")
    ap.add_argument("--csv", metavar="FILE",
                    help="also write the complete list to this CSV")
    ap.add_argument("--include-iffalse", action="store_true",
                    help="count \\iffalse ... \\fi drafts as prose")
    ap.add_argument("--min-len", type=int, default=1,
                    help="ignore words shorter than this (default 1)")
    ap.add_argument("--text-only", action="store_true",
                    help="rank by prose count instead of total count")
    args = ap.parse_args()

    with open(args.tex, encoding="utf-8", errors="replace") as fh:
        raw = fh.read()

    prose, code, skipped = separate(raw, args.include_iffalse)
    text_counts, text_spell = tally(prose)
    code_counts, code_spell = tally(code)

    keys = set(text_counts) | set(code_counts)
    if args.min_len > 1:
        keys = {k for k in keys if len(k) >= args.min_len}

    rows = []
    for k in keys:
        t, c = text_counts[k], code_counts[k]
        spell = (text_spell[k] + code_spell[k]).most_common(1)[0][0]
        if t and c:
            kind = "BOTH"
        elif t:
            kind = "TEXT"
        else:
            kind = "TEX"
        rows.append((spell, t + c, t, c, kind))

    sort_key = (lambda r: (-r[2], -r[1], r[0].lower())) if args.text_only \
        else (lambda r: (-r[1], r[0].lower()))
    rows.sort(key=sort_key)

    total_words = sum(text_counts.values()) + sum(code_counts.values())
    print()
    print("Word frequency  --  %s" % texsource.relname(args.tex))
    print("=" * 74)
    print("prose words: %-8d  tex/scaffolding words: %-8d  distinct: %d"
          % (sum(text_counts.values()), sum(code_counts.values()), len(keys)))
    if skipped:
        print("skipped %d lines inside \\iffalse ... \\fi "
              "(use --include-iffalse to count them)" % skipped)
    print()
    print("  #  word                         total     text      tex   kind")
    print("-" * 74)
    for rank, (word, total, t, c, kind) in enumerate(rows[:args.top], 1):
        pct = 100.0 * total / total_words if total_words else 0.0
        print("%3d  %-26.26s %6d   %6d   %6d   %-4s  %5.2f%%"
              % (rank, word, total, t, c, kind, pct))
    print("-" * 74)
    print("TEXT = prose only   TEX = LaTeX scaffolding only   BOTH = appears as both")

    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["rank", "word", "total", "text_count", "tex_count", "kind"])
            for rank, r in enumerate(rows, 1):
                w.writerow([rank, r[0], r[1], r[2], r[3], r[4]])
        print("full list -> %s (%d rows)" % (args.csv, len(rows)))


if __name__ == "__main__":
    sys.exit(main())
