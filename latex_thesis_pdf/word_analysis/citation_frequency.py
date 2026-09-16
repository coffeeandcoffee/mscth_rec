#!/usr/bin/env python3
"""Citation-frequency report for a LaTeX thesis.

Counts how often each BibTeX key is cited in the body of the .tex file and
ranks the keys. Multi-key calls like \\cite{a, b} count once for each key.
Resolves each key against library.bib so the list reads with author, year
and title rather than bare keys.

Usage:
    python3 citation_frequency.py [../thesis.tex] [-n 0] [--csv out.csv]
                                  [--keys-only] [--include-iffalse]
                                  [--bib ../library.bib] [--unused]

Citations inside \\iffalse ... \\fi are reported separately, in a trailing
column, because that text never reaches the PDF. --include-iffalse folds
them into the main count instead.

Stdlib only. No dependencies.
"""

import argparse
import csv
import os
import re
import sys
from collections import Counter

import texsource

CITE_RE = re.compile(
    r"\\(?:cite|citep|citet|citeauthor|citeyear|parencite|textcite|footcite|"
    r"autocite|smartcite|nocite)\*?"
    r"(?:\[[^\]]*\])*"      # optional [prenote][postnote]
    r"\{([^}]*)\}"
)

ENTRY_RE = re.compile(r"@(\w+)\s*\{\s*([^,\s]+)\s*,", re.I)


def count_keys(text):
    """Counter of bib key -> number of citing calls in `text`."""
    counts = Counter()
    for m in CITE_RE.finditer(text):
        for key in m.group(1).split(","):
            key = key.strip()
            if key:
                counts[key] += 1
    return counts


def field(entry, name):
    """Pull one brace-delimited field out of a .bib entry body."""
    m = re.search(r"\b%s\s*=\s*" % name, entry, re.I)
    if not m:
        return ""
    i = m.end()
    while i < len(entry) and entry[i] in " \t":
        i += 1
    if i >= len(entry):
        return ""
    if entry[i] == "{":
        depth, j = 0, i
        while j < len(entry):
            if entry[j] == "{":
                depth += 1
            elif entry[j] == "}":
                depth -= 1
                if depth == 0:
                    break
            j += 1
        val = entry[i + 1:j]
    elif entry[i] == '"':
        j = entry.find('"', i + 1)
        val = entry[i + 1:j if j != -1 else len(entry)]
    else:
        j = re.search(r"[,\n]", entry[i:])
        val = entry[i:i + (j.start() if j else len(entry) - i)]
    val = re.sub(r"[{}]", "", val)
    return " ".join(val.split())


def parse_bib(path):
    """Map bib key -> (first author surname, year, title)."""
    if not path or not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8", errors="replace") as fh:
        text = fh.read()
    out, starts = {}, [(m.start(), m.group(2)) for m in ENTRY_RE.finditer(text)]
    for n, (pos, key) in enumerate(starts):
        end = starts[n + 1][0] if n + 1 < len(starts) else len(text)
        entry = text[pos:end]
        author = field(entry, "author") or field(entry, "editor")
        first = author.split(" and ")[0] if author else ""
        if "," in first:
            surname = first.split(",")[0]
        else:
            surname = first.split()[-1] if first else ""
        year = field(entry, "year") or field(entry, "date")
        m = re.search(r"\d{4}", year)
        year = m.group(0) if m else ""
        title = field(entry, "shorttitle") or field(entry, "title")
        out[key] = (surname.strip(), year, title)
    return out


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tex", nargs="?", default=texsource.DEFAULT_TEX)
    ap.add_argument("-n", "--top", type=int, default=0,
                    help="how many rows to print (0 = all, the default)")
    ap.add_argument("--csv", metavar="FILE",
                    help="also write the complete list to this CSV")
    ap.add_argument("--bib", default=texsource.DEFAULT_BIB,
                    help="bibliography to resolve keys against")
    ap.add_argument("--keys-only", action="store_true",
                    help="print bare keys, skip author/year/title lookup")
    ap.add_argument("--include-iffalse", action="store_true",
                    help="count citations inside \\iffalse ... \\fi too")
    ap.add_argument("--unused", action="store_true",
                    help="also list bib entries that are never cited")
    args = ap.parse_args()

    raw = texsource.read(args.tex)
    live_src, dead_src = texsource.live_and_disabled(raw)
    if args.include_iffalse:
        live_src, dead_src = live_src + dead_src, ""

    live = count_keys(live_src)
    dead = count_keys(dead_src)
    bib = {} if args.keys_only else parse_bib(args.bib)

    rows = []
    for key, n in live.most_common():
        surname, year, title = bib.get(key, ("", "", ""))
        rows.append((key, n, dead.get(key, 0), surname, year, title))

    orphans = sorted(set(dead) - set(live))
    missing = sorted(k for k in live if bib and k not in bib)

    print()
    print("Citations  --  %s" % texsource.relname(args.tex))
    print("=" * 96)
    print("citing calls: %-6d  distinct keys: %-5d  mean per key: %.1f"
          % (sum(live.values()), len(live),
             sum(live.values()) / len(live) if live else 0))
    if bib:
        print("bibliography: %s (%d entries)" % (texsource.relname(args.bib), len(bib)))
    if dead:
        print("plus %d citing calls inside \\iffalse ... \\fi "
              "(shown as +N; --include-iffalse to merge)" % sum(dead.values()))
    print()

    shown = rows[:args.top] if args.top else rows
    if args.keys_only or not bib:
        print("  #  key                                  cites")
        print("-" * 96)
        for rank, (key, n, d, _, _, _) in enumerate(shown, 1):
            print("%3d  %-36.36s %4d%s"
                  % (rank, key, n, "  +%d" % d if d else ""))
    else:
        print("  #  cites  key                              source")
        print("-" * 96)
        for rank, (key, n, d, surname, year, title) in enumerate(shown, 1):
            tag = "%s %s" % (surname or "?", year or "n.d.")
            print("%3d  %4d%-4s %-32.32s %-.58s"
                  % (rank, n, "  +%d" % d if d else "", key, "%s — %s" % (tag, title)))
    print("-" * 96)

    if orphans:
        print("\ncited ONLY inside \\iffalse (not in the compiled thesis): %d"
              % len(orphans))
        for k in orphans:
            print("    %s" % k)
    if missing:
        print("\ncited but NOT found in %s: %d"
              % (texsource.relname(args.bib), len(missing)))
        for k in missing:
            print("    %s" % k)
    if args.unused and bib:
        never = sorted(set(bib) - set(live) - set(dead))
        print("\nin the .bib but never cited: %d of %d entries"
              % (len(never), len(bib)))
        for k in never:
            surname, year, title = bib[k]
            print("    %-34.34s %s %s — %.40s" % (k, surname or "?", year or "n.d.", title))

    singles = sum(1 for r in rows if r[1] == 1)
    print("\ncited once only: %d of %d keys (%.0f%%)"
          % (singles, len(rows), 100.0 * singles / len(rows) if rows else 0))

    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["rank", "key", "cites", "cites_in_iffalse",
                        "author", "year", "title"])
            for rank, r in enumerate(rows, 1):
                w.writerow([rank, r[0], r[1], r[2], r[3], r[4], r[5]])
        print("full list -> %s (%d rows)" % (args.csv, len(rows)))


if __name__ == "__main__":
    sys.exit(main())
