"""
thesis_metric_table.py — one LaTeX style for every model-comparison table in the thesis
(viz17_1_* : EI model vs coin-flip baseline, viz17_5_* : top-feature model vs EI model).

Layout: a shaded box with the class name (STAY / SKIP), written vertically, left of the
four metrics of that class; one performance column per model, then the difference
(compared model minus reference, in percentage points); a significance column set apart
by a vertical rule.
Colours: significantly higher -> dark green cells, white text, up-triangle before the
difference; significantly lower -> red text, down-triangle before the difference;
not significant -> whole row gray.

Needs in the thesis preamble: colortbl, tabularx, multirow, amssymb (graphicx for \\rotatebox).
"""

METRIC_LABELS = {
    'F1 Train': r'F\(_1\)-train',
    'F1 Test': r'F\(_1\)-test',
    'Precision': 'Precision',
    'Recall': 'Recall',
}

COLOR_DEFS = [
    r"\providecolor{sigHigher}{HTML}{1E7B34}",
    r"\providecolor{sigLower}{HTML}{C62828}",
    r"\providecolor{nsGray}{gray}{0.55}",
    r"\providecolor{classBox}{gray}{0.90}",
    r"\providecolor{ruleGray}{gray}{0.78}",
    r"\providecolor{headGray}{gray}{0.45}",
]

SETTING_TITLES = {
    'intra': ("Performance within each participant",
              "intra-subject setting: each model trained and tested on one participant"),
    'inter': ("Performance across participants",
              "inter-subject setting: each model trained on 24 participants, tested on the 25th"),
}


def _comment(text, brackets=False, size=r"\scriptsize"):
    """Gray explanatory note: plain in the note row below the column names, bracketed under the title.
    \textcolor (not \color) so the note starts on the first line of a p-cell."""
    if brackets:
        text = f"({text})"
    return rf"{{{size}\textcolor{{headGray}}{{{text}}}}}"


def _pct(v):
    return f"{v:.1f}\\%"


def _pp(ref_v, mod_v):
    # Difference of the rounded values, so it matches the two numbers the reader sees.
    d = round(mod_v, 1) - round(ref_v, 1)
    sign = '+' if d >= 0 else '\\(-\\)'
    return f"{sign}{abs(d):.1f}"


def render(rows, scale, ref_header, model_header, sig_comment, caption, label):
    """
    scale: 'intra' or 'inter' (sets the title line above the table).
    ref_header, model_header: (primary, comment) pairs for the two model columns.
    sig_comment: gray comment under "Significance test"; may contain \\newline.
    rows: list of (metric, ref_value, model_value, verdict), in the order
          F1 Train/F1 Test/Precision/Recall for STAY, then the same for SKIP.
          metric looks like 'F1 Train (STAY)'; verdict is 'higher', 'lower' or 'ns'.
    Returns the full LaTeX table as a string.
    """
    groups = {}
    for metric, ref_v, mod_v, verdict in rows:
        name, cls = metric.rsplit(' (', 1)
        groups.setdefault(cls.rstrip(')'), []).append((name, ref_v, mod_v, verdict))

    out = [r"\begin{table}[H]", r"\centering", *COLOR_DEFS,
           r"\small",
           r"\renewcommand{\arraystretch}{1.12}",
           r"\setlength{\tabcolsep}{2pt}",
           # Title above the table (outside the tabular so it cannot widen a column),
           # with its gray comment on a line of its own.
           rf"{{\raggedright\textbf{{{SETTING_TITLES[scale][0]}}}\par {_comment(SETTING_TITLES[scale][1], brackets=True, size=r"\footnotesize")}\par}}",
           r"\vspace{3pt}",
           r"\begin{tabularx}{\textwidth}{c l >{\centering\arraybackslash}p{2.5cm} >{\centering\arraybackslash}X >{\centering\arraybackslash}p{1.5cm}|>{\centering\arraybackslash}p{2.95cm}}",
           r"\arrayrulecolor{black}\hline",
           # Row 1: column names in black. Row 2: gray comments.
           rf"\multicolumn{{2}}{{l}}{{\footnotesize\textbf{{Metric}}}} & {{\footnotesize\textbf{{{ref_header[0]}}}}} & {{\footnotesize\textbf{{{model_header[0]}}}}} & "
           r"{\footnotesize\textbf{\(\Delta\)\,[pp]}} & {\footnotesize\textbf{Significance test}} \\",
           rf"\multicolumn{{2}}{{l}}{{{{\scriptsize\textcolor{{headGray}}{{\textbf{{Note}}}}}}}} & {_comment(ref_header[1])} & {_comment(model_header[1])} & "
           rf" & {_comment(sig_comment)} \\",
           r"\hline"]

    for g_idx, (cls, items) in enumerate(groups.items()):
        n = len(items)
        for i, (name, ref_v, mod_v, verdict) in enumerate(items):
            if i < n - 1:
                box = r"\cellcolor{classBox}"
            else:
                # Negative multirow in the last row, so the cell colour of the rows
                # above does not paint over the label.
                box = rf"\cellcolor{{classBox}}\multirow{{-{n}}}{{*}}{{\rotatebox[origin=c]{{90}}{{\texttt{{{cls}}}}}}}"

            metric_tex = METRIC_LABELS.get(name, name)
            ref_tex, mod_tex, diff_tex = _pct(ref_v), _pct(mod_v), _pp(ref_v, mod_v)
            if verdict == 'higher':
                mod_tex = rf"\cellcolor{{sigHigher}}\textcolor{{white}}{{\textbf{{{mod_tex}}}}}"
                diff_tex = rf"\cellcolor{{sigHigher}}\textcolor{{white}}{{\(\blacktriangle\)\,\textbf{{{diff_tex}}}}}"
                sig_tex = r"{\footnotesize significantly higher}"
            elif verdict == 'lower':
                mod_tex = rf"\textcolor{{sigLower}}{{\textbf{{{mod_tex}}}}}"
                diff_tex = rf"\textcolor{{sigLower}}{{\(\blacktriangledown\)\,\textbf{{{diff_tex}}}}}"
                sig_tex = r"{\footnotesize significantly lower}"
            else:
                metric_tex = rf"\textcolor{{nsGray}}{{{metric_tex}}}"
                ref_tex = rf"\textcolor{{nsGray}}{{{ref_tex}}}"
                mod_tex = rf"\textcolor{{nsGray}}{{{mod_tex}}}"
                diff_tex = rf"\textcolor{{nsGray}}{{{diff_tex}}}"
                sig_tex = r"{\footnotesize\textcolor{nsGray}{not significant}}"

            out.append(f"{box} & {metric_tex} & {ref_tex} & {mod_tex} & {diff_tex} & {sig_tex} \\\\")
            if i < n - 1:
                out.append(r"\arrayrulecolor{ruleGray}\cline{2-6}")
        out.append(r"\arrayrulecolor{black}\hline")

    out += [r"\end{tabularx}",
            rf"\caption{{{caption}}}",
            rf"\label{{{label}}}",
            r"\end{table}"]
    return "\n".join(out)


def legend_sentence(ref_short):
    """Shared caption tail that explains the colours."""
    return (f"Performance is the mean over the 25 participants. \\(\\Delta\\): compared model minus "
            f"{ref_short}, in percentage points (pp). Significance test: two-sided "
            f"Wilcoxon signed-rank test against {ref_short} ($\\alpha=0.05$, $n=25$, uncorrected). "
            f"Green cells: significantly higher than {ref_short}; red values: significantly lower; "
            f"gray row: not significant.")
