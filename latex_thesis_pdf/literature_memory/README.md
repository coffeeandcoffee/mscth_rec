# literature_memory

Dated snapshots of **what the cited sources actually say**, in their own words.

Each file is `YYYYMMDDHHMM_literature_memory.md`. Files are append-only history:
never edit an old one, write a new one. The newest file is the current picture;
older files show what was believed earlier and why it changed.

## What belongs in here

Verbatim quotes from source PDFs, with, for each source:

- full bibliographic record + Zotero storage key (so the PDF can be re-found)
- where it is cited in `../thesis.tex`, by chapter and section
- **SUPPORTS** — thesis claims the quote carries in full
- **DOES NOT SUPPORT** — claims the source cannot carry, and why
- **LIMIT** — domain, sample, task. A quote reused outside its limit overclaims.

The SUPPORTS / DOES NOT SUPPORT / LIMIT split is the point of the format. A bare
quote collection invites the same overclaims it was meant to prevent.

## What this is for

A later reader — human or model — that has read *other* sources and needs to fit
them into the existing argument without re-reading these PDFs. It answers: what
is already established, by whom, how strongly, and where the argument is still
carried by a single citation.

## Conventions

- Quotes are verbatim, including source typos and OCR artefacts. Do not clean them.
- Mark paraphrase as paraphrase.
- Record corrections with dates, e.g. *"thesis previously said X; corrected 2026-09-16"*.
- Record provenance problems loudly (e.g. a PDF that is an abstract, not the article).
- If a claim has **no** possible source (an assertion that nothing exists), say so
  rather than leaving it looking uncited by oversight.

## Hard rule

Never take evidence, claims or numbers from `\iffalse ... \fi` blocks in
`../thesis.tex`. They are a superseded study, not an earlier draft. See the
"Do not mine" section in the newest snapshot.

## Related

- `../word_analysis/` — text-statistics scripts over `thesis.tex`
- `../word_analysis/source_audit.md` — before/after of every claim-level correction
- `../word_analysis/precedent_rewrite.md` — the rewrite that produced the first snapshot
