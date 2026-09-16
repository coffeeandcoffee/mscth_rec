# word_analysis

Read-only analysis scripts for `../thesis.tex`. Nothing in here writes to the
thesis, the bibliography, or the figures — each script reads the source and
prints a report (optionally a CSV next to itself).

Python 3, stdlib only. No venv, no dependencies.

## Scripts

| script | what it answers |
|---|---|
| `word_frequency.py` | Which words repeat, and how often — split into prose vs. LaTeX scaffolding |
| `citation_frequency.py` | Which sources carry the argument — bib keys ranked by citation count |
| `texsource.py` | Shared helper: default paths, comment stripping, `\iffalse` splitting. Not run directly |

```bash
python3 word_frequency.py                 # top 100 words
python3 word_frequency.py -n 500 --csv word_frequency.csv
python3 word_frequency.py --text-only --min-len 5 -n 60   # content words only

python3 citation_frequency.py             # all keys, with author/year/title
python3 citation_frequency.py -n 20 --csv citations.csv
python3 citation_frequency.py --unused    # bib entries never cited
```

## Shared conventions

Every script defaults to `../thesis.tex` and `../library.bib`, and agrees on
what "the live thesis" means:

- the document body only, preamble excluded
- `%` comments removed
- `\iffalse ... \fi` blocks removed — that text never reaches the PDF

`--include-iffalse` folds those draft blocks back in wherever it matters.
`texsource.live_and_disabled(raw)` returns both halves, so a new script gets
this behaviour for free.

## Adding a new analysis

Drop a new `*.py` in this folder, `import texsource`, take `tex` as an
optional positional argument defaulting to `texsource.DEFAULT_TEX`, and offer
`--csv`. Keep it read-only and stdlib-only.
