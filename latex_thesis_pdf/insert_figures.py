import re

with open("/Users/gregorlederer/Documents/MSc Thesis - EEG Neuroscience/Data Recording and Quality Tests/latex_thesis_pdf/thesis.tex", "r") as f:
    content = f.read()

methods_insertion = r"""
% --- INSERTED METHODS FIGURES ---
\begin{figure}[h]
    \centering
    \includegraphics[width=\textwidth]{../05-12_data_analysis_and_results/runs/run_20260611_220844/viz/viz01.1.png}
    \caption{Sampling Rate Distribution across participants.}
\end{figure}

\begin{figure}[h]
    \centering
    \includegraphics[width=\textwidth]{../05-12_data_analysis_and_results/runs/run_20260611_220844/viz/viz01.3.png}
    \caption{Bluetooth Dropout Heatmap indicating signal loss events.}
\end{figure}

\begin{figure}[h]
    \centering
    \includegraphics[width=\textwidth]{../05-12_data_analysis_and_results/runs/run_20260611_220844/viz/viz02.1.png}
    \caption{Artifact rate distribution and signal quality flagging.}
\end{figure}

\begin{figure}[h]
    \centering
    \includegraphics[width=\textwidth]{../05-12_data_analysis_and_results/runs/run_20260611_220844/viz/viz03/viz03.1.png}
    \caption{Window extraction process from raw signal.}
\end{figure}

\begin{figure}[h]
    \centering
    \includegraphics[width=\textwidth]{../05-12_data_analysis_and_results/runs/run_20260611_220844/viz/viz03/viz03.2.png}
    \caption{Raw class distribution (STAY vs. SKIP) before balancing.}
\end{figure}

\begin{figure}[h]
    \centering
    \includegraphics[width=\textwidth]{../05-12_data_analysis_and_results/runs/run_20260611_220844/viz/viz03/viz03.3.png}
    \caption{Window duration analysis.}
\end{figure}

\begin{figure}[h]
    \centering
    \includegraphics[width=\textwidth]{../05-12_data_analysis_and_results/runs/run_20260611_220844/viz/viz03/viz03.4.png}
    \caption{Behavioral interaction distribution over time.}
\end{figure}

\begin{figure}[h]
    \centering
    \includegraphics[width=\textwidth]{../05-12_data_analysis_and_results/runs/run_20260611_220844/viz/viz03_exploration/viz03_exploration_9.png}
    \caption{Window parameter exploration (Slice 9).}
\end{figure}

\begin{figure}[h]
    \centering
    \includegraphics[width=\textwidth]{../05-12_data_analysis_and_results/runs/run_20260611_220844/viz/viz03_exploration/viz03_exploration_7.png}
    \caption{Window parameter exploration (Slice 7).}
\end{figure}

\begin{figure}[h]
    \centering
    \includegraphics[width=\textwidth]{../05-12_data_analysis_and_results/runs/run_20260611_220844/viz/viz03_exploration/viz03_exploration_12.png}
    \caption{Window parameter exploration (Slice 12).}
\end{figure}

\begin{figure}[h]
    \centering
    \includegraphics[width=\textwidth]{../05-12_data_analysis_and_results/runs/run_20260611_220844/viz/viz04/viz04.2_before_balancing.png}
    \caption{Raw window counts before balancing.}
\end{figure}

\begin{figure}[h]
    \centering
    \includegraphics[width=\textwidth]{../05-12_data_analysis_and_results/runs/run_20260611_220844/viz/viz04/viz04.2_after_balancing.png}
    \caption{Window counts after balancing.}
\end{figure}

\begin{figure}[h]
    \centering
    \includegraphics[width=\textwidth]{../05-12_data_analysis_and_results/runs/run_20260611_220844/viz/viz04/viz04.1.png}
    \caption{Temporal cross-validation split visualization.}
\end{figure}

\begin{figure}[h]
    \centering
    \includegraphics[width=\textwidth]{../05-12_data_analysis_and_results/runs/run_20260611_220844/viz/viz04/viz04.2.png}
    \caption{Balanced class counts across participants.}
\end{figure}

\begin{figure}[h]
    \centering
    \includegraphics[width=\textwidth]{../05-12_data_analysis_and_results/runs/run_20260611_220844/viz/viz04/viz04.3.png}
    \caption{Fold composition and data distribution.}
\end{figure}

\begin{figure}[h]
    \centering
    \includegraphics[width=\textwidth]{../05-12_data_analysis_and_results/runs/run_20260611_220844/viz/viz04/viz04.4.png}
    \caption{Temporal fold separation details.}
\end{figure}

\begin{figure}[h]
    \centering
    \includegraphics[width=\textwidth]{../05-12_data_analysis_and_results/runs/run_20260611_220844/viz/viz04/viz04.5.png}
    \caption{Temporal distance between train and test windows.}
\end{figure}

\begin{figure}[h]
    \centering
    \includegraphics[width=\textwidth]{../05-12_data_analysis_and_results/runs/run_20260611_220844/viz/viz05a.A2.png}
    \caption{Extracted statistical moments from EEG signals.}
\end{figure}

\begin{figure}[h]
    \centering
    \includegraphics[width=\textwidth]{../05-12_data_analysis_and_results/runs/run_20260611_220844/viz/viz05a.A3.png}
    \caption{Engagement Index (EI) calculation from band powers.}
\end{figure}
% --- END INSERTED METHODS FIGURES ---
\section{Internal Structuring: Results Narrative and Gap Analysis}
"""

results_insertion = r"""
% --- INSERTED RESULTS FIGURES ---
\begin{figure}[h]
    \centering
    \includegraphics[width=\textwidth]{../05-12_data_analysis_and_results/runs/run_20260611_220844/viz/viz17_0.1.1_cohend_bands.png}
    \caption{Cohen's d effect size by frequency band.}
\end{figure}

\begin{figure}[h]
    \centering
    \includegraphics[width=\textwidth]{../05-12_data_analysis_and_results/runs/run_20260611_220844/viz/viz17_0.1.2_cohend_electrodes.png}
    \caption{Cohen's d effect size by electrode.}
\end{figure}

\begin{figure}[h]
    \centering
    \includegraphics[width=\textwidth]{../05-12_data_analysis_and_results/runs/run_20260611_220844/viz/viz17_0.1.3_cohend_stats.png}
    \caption{Cohen's d effect size by statistical moment.}
\end{figure}

\begin{figure}[h]
    \centering
    \includegraphics[width=\textwidth]{../05-12_data_analysis_and_results/runs/run_20260611_220844/viz/viz17_0.1.4_cohend_interactions.png}
    \caption{Cohen's d effect size interactions.}
\end{figure}

\begin{figure}[h]
    \centering
    \includegraphics[width=\textwidth]{../05-12_data_analysis_and_results/runs/run_20260611_220844/viz/viz17_1_inter.png}
    \caption{Inter-subject performance metrics.}
\end{figure}

\input{../05-12_data_analysis_and_results/runs/run_20260611_220844/viz/viz17_1_inter.tex}

\begin{figure}[h]
    \centering
    \includegraphics[width=\textwidth]{../05-12_data_analysis_and_results/runs/run_20260611_220844/viz/viz17_1_intra.png}
    \caption{Intra-subject performance metrics.}
\end{figure}

\input{../05-12_data_analysis_and_results/runs/run_20260611_220844/viz/viz17_1_intra.tex}

\input{../05-12_data_analysis_and_results/runs/run_20260611_220844/viz/viz17_2_inter_significance_thesis_paragraph.tex}

\begin{figure}[h]
    \centering
    \includegraphics[width=\textwidth]{../05-12_data_analysis_and_results/runs/run_20260611_220844/viz/viz17_2_inter_significance.png}
    \caption{Inter-subject significance analysis.}
\end{figure}

\input{../05-12_data_analysis_and_results/runs/run_20260611_220844/viz/viz17_2_intra_significance_thesis_paragraph.tex}

\begin{figure}[h]
    \centering
    \includegraphics[width=\textwidth]{../05-12_data_analysis_and_results/runs/run_20260611_220844/viz/viz17_2_intra_significance.png}
    \caption{Intra-subject significance analysis.}
\end{figure}

\subsubsection*{Methodology for Significance Testing}
To rigorously determine whether a predictive model meaningfully outperforms a random baseline (Coin Flip) or a standard reference (Engagement Index, EI LR), this pipeline employs the Wilcoxon signed-rank test. Standard parametric tests assume that the performance metrics across participants are normally distributed. In small-sample physiological data (N=25 participants), this assumption is often violated. Performance is evaluated using paired metrics from the 25 individual cross-validation folds. The test calculates the difference in performance between the evaluated model and the baseline model for each participant. The algorithm employs a two-sided Wilcoxon test with a standard significance threshold ($\alpha = 0.05$). To strictly guarantee superiority, a directional check is applied post-hoc: the arithmetic mean of the model's scores across all participants must be strictly greater than the arithmetic mean of the baseline.

\input{../05-12_data_analysis_and_results/runs/run_20260611_220844/viz/viz17_3_cohend_method.tex}

\input{../05-12_data_analysis_and_results/runs/run_20260611_220844/viz/viz17_3_cohend_results.tex}

\begin{figure}[h]
    \centering
    \includegraphics[width=\textwidth]{../05-12_data_analysis_and_results/runs/run_20260611_220844/viz/viz17_3_cohend_top_features.png}
    \caption{Top features ranked by Cohen's d.}
\end{figure}

\input{../05-12_data_analysis_and_results/runs/run_20260611_220844/viz/viz17_3_cohend_top_features.tex}

\input{../05-12_data_analysis_and_results/runs/run_20260611_220844/viz/viz17_4_description.tex}

\begin{figure}[h]
    \centering
    \includegraphics[width=\textwidth]{../05-12_data_analysis_and_results/runs/run_20260611_220844/viz/viz17_4_identifying_top.png}
    \caption{Identifying top predictive features.}
\end{figure}

\subsubsection*{Selected Features}
\begin{verbatim}
{
    "AF7_high_gamma_mean": {
        "d": 0.16824331879615784,
        "labels": ["Top Band (high_gamma)", "Top Electrode (AF7)", "Overall Top Feature #2"]
    },
    "AF8_delta_peakfreq": {
        "d": 0.13990937173366547,
        "labels": ["Top Statistic (peakfreq)"]
    },
    "TP10_raw_std": {
        "d": 0.1959162801504135,
        "labels": ["Overall Top Feature #1"]
    }
}
\end{verbatim}

\begin{figure}[h]
    \centering
    \includegraphics[width=\textwidth]{../05-12_data_analysis_and_results/runs/run_20260611_220844/viz/viz17_5_inter_feat1.png}
    \caption{Inter-subject performance using Feature 1.}
\end{figure}

\input{../05-12_data_analysis_and_results/runs/run_20260611_220844/viz/viz17_5_inter_feat1.tex}

\begin{figure}[h]
    \centering
    \includegraphics[width=\textwidth]{../05-12_data_analysis_and_results/runs/run_20260611_220844/viz/viz17_5_inter_feat2.png}
    \caption{Inter-subject performance using Feature 2.}
\end{figure}

\input{../05-12_data_analysis_and_results/runs/run_20260611_220844/viz/viz17_5_inter_feat2.tex}

\begin{figure}[h]
    \centering
    \includegraphics[width=\textwidth]{../05-12_data_analysis_and_results/runs/run_20260611_220844/viz/viz17_5_inter_feat3.png}
    \caption{Inter-subject performance using Feature 3.}
\end{figure}

\input{../05-12_data_analysis_and_results/runs/run_20260611_220844/viz/viz17_5_inter_feat3.tex}

\begin{figure}[h]
    \centering
    \includegraphics[width=\textwidth]{../05-12_data_analysis_and_results/runs/run_20260611_220844/viz/viz17_5_intra_feat1.png}
    \caption{Intra-subject performance using Feature 1.}
\end{figure}

\input{../05-12_data_analysis_and_results/runs/run_20260611_220844/viz/viz17_5_intra_feat1.tex}

\begin{figure}[h]
    \centering
    \includegraphics[width=\textwidth]{../05-12_data_analysis_and_results/runs/run_20260611_220844/viz/viz17_5_intra_feat2.png}
    \caption{Intra-subject performance using Feature 2.}
\end{figure}

\input{../05-12_data_analysis_and_results/runs/run_20260611_220844/viz/viz17_5_intra_feat2.tex}

\begin{figure}[h]
    \centering
    \includegraphics[width=\textwidth]{../05-12_data_analysis_and_results/runs/run_20260611_220844/viz/viz17_5_intra_feat3.png}
    \caption{Intra-subject performance using Feature 3.}
\end{figure}

\input{../05-12_data_analysis_and_results/runs/run_20260611_220844/viz/viz17_5_intra_feat3.tex}

\input{../05-12_data_analysis_and_results/runs/run_20260611_220844/viz/viz17_inter_diagnostics.tex}

\begin{figure}[h]
    \centering
    \includegraphics[width=\textwidth]{../05-12_data_analysis_and_results/runs/run_20260611_220844/viz/viz17_inter_final.png}
    \caption{Final inter-subject performance summary.}
\end{figure}

\begin{figure}[h]
    \centering
    \includegraphics[width=\textwidth]{../05-12_data_analysis_and_results/runs/run_20260611_220844/viz/viz17_inter_metrics_table_ablation.png}
    \caption{Inter-subject metrics table ablation.}
\end{figure}

\input{../05-12_data_analysis_and_results/runs/run_20260611_220844/viz/viz17_inter_metrics_table_ablation.tex}

\begin{figure}[h]
    \centering
    \includegraphics[width=\textwidth]{../05-12_data_analysis_and_results/runs/run_20260611_220844/viz/viz17_inter_optimal_params_ablation.png}
    \caption{Inter-subject optimal parameters ablation.}
\end{figure}

\input{../05-12_data_analysis_and_results/runs/run_20260611_220844/viz/viz17_inter_sig_f1_models_ablation.tex}

\input{../05-12_data_analysis_and_results/runs/run_20260611_220844/viz/viz17_intra_diagnostics.tex}

\begin{figure}[h]
    \centering
    \includegraphics[width=\textwidth]{../05-12_data_analysis_and_results/runs/run_20260611_220844/viz/viz17_intra_final.png}
    \caption{Final intra-subject performance summary.}
\end{figure}

\begin{figure}[h]
    \centering
    \includegraphics[width=\textwidth]{../05-12_data_analysis_and_results/runs/run_20260611_220844/viz/viz17_intra_metrics_table_ablation.png}
    \caption{Intra-subject metrics table ablation.}
\end{figure}

\input{../05-12_data_analysis_and_results/runs/run_20260611_220844/viz/viz17_intra_metrics_table_ablation.tex}

\begin{figure}[h]
    \centering
    \includegraphics[width=\textwidth]{../05-12_data_analysis_and_results/runs/run_20260611_220844/viz/viz17_intra_optimal_params_ablation.png}
    \caption{Intra-subject optimal parameters ablation.}
\end{figure}

\input{../05-12_data_analysis_and_results/runs/run_20260611_220844/viz/viz17_intra_sig_f1_models_ablation.tex}
% --- END INSERTED RESULTS FIGURES ---
\subsection{8. Summary of Results}
"""

content = content.replace(r"\section{Internal Structuring: Results Narrative and Gap Analysis}", methods_insertion)
content = content.replace(r"\subsection{8. Summary of Results}", results_insertion)

with open("/Users/gregorlederer/Documents/MSc Thesis - EEG Neuroscience/Data Recording and Quality Tests/latex_thesis_pdf/thesis.tex", "w") as f:
    f.write(content)
print("Insertion completed.")
