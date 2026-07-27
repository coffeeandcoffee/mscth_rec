#!/usr/bin/env python3
import json
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import config
import pickle

def run(run_dir, params):
    json_file = run_dir / "features" / "best_features_ranking.json"
    if not json_file.exists():
        print(f"  viz17_feature_ranking: {json_file.name} not found.")
        return
        
    with open(json_file, 'r') as f:
        ranking = json.load(f)
        
    if not ranking:
        return
        
    # --- Calculate Baseline EI Cohen's d ---
    all_ei = []
    all_labels = []
    for pkl_file in (run_dir / "features").glob("P*.pkl"):
        with open(pkl_file, 'rb') as f:
            data = pickle.load(f)
            all_ei.extend(data['ei_values'])
            all_labels.extend(data['labels'])
            
    all_ei = np.array(all_ei)
    all_labels = np.array(all_labels)
    
    # We don't need to know which is STAY/SKIP because we take absolute difference
    unique_labels = np.unique(all_labels)
    if len(unique_labels) == 2:
        ei_0 = all_ei[all_labels == unique_labels[0]]
        ei_1 = all_ei[all_labels == unique_labels[1]]
        pooled_std = np.sqrt((np.var(ei_0) + np.var(ei_1)) / 2)
        ei_d = np.abs(np.mean(ei_1) - np.mean(ei_0)) / (pooled_std + 1e-9)
    else:
        ei_d = 0.0
        
    # Sort descending by absolute d
    ranking_full = sorted(ranking, key=lambda x: abs(x['d']), reverse=True)
    
    # Top 30 features to avoid clutter for the bar chart
    top_n = 30
    ranking = ranking_full[:top_n]
    
    names = [r['name'].replace('_', ' ') for r in ranking]
    ds = [abs(r['d']) for r in ranking]
    
    viz_dir = run_dir / "viz"
    viz_dir.mkdir(parents=True, exist_ok=True)
    
    # --- Feature Parsing for 0.1.x Visuals ---
    parsed = []
    for r in ranking_full: # use full ranking for aggregates
        name = r['name']
        d = abs(r['d'])
        parts = name.split('_')
        elec = parts[0]
        
        if 'high_gamma' in name:
            band = 'high_gamma'
            stat = name.split('high_gamma_')[1] if len(name.split('high_gamma_')) > 1 else 'unknown'
        elif 'low_gamma' in name:
            band = 'low_gamma'
            stat = name.split('low_gamma_')[1] if len(name.split('low_gamma_')) > 1 else 'unknown'
        elif 'very_high' in name:
            band = 'very_high'
            stat = name.split('very_high_')[1] if len(name.split('very_high_')) > 1 else 'unknown'
        elif 'raw' in name:
            band = 'raw'
            stat = name.split('raw_')[1] if len(name.split('raw_')) > 1 else 'unknown'
        else:
            band = parts[1] if len(parts) > 1 else 'unknown'
            stat = '_'.join(parts[2:]) if len(parts) > 2 else 'unknown'
            
        parsed.append({'Name': name, 'Electrode': elec, 'Band': band, 'Stat': stat, 'd': d})
        
    df = pd.DataFrame(parsed)
    
    # Extract frequencies from config for labeling
    freq_map = {name: f"{name} ({low}-{high} Hz)" for name, low, high in config.FREQUENCY_BANDS}
    freq_map['raw'] = 'raw (Broadband)'
    
    # Enforce logical band ordering
    band_order_base = ['raw', 'delta', 'theta', 'alpha', 'beta', 'low_gamma', 'gamma', 'high_gamma', 'very_high']
    bands_present = [b for b in band_order_base if b in df['Band'].unique()] + [b for b in df['Band'].unique() if b not in band_order_base]
    band_labels = [freq_map.get(b, b) for b in bands_present]
    
    # 0.1.1: Band Barplot
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=df, x='Band', y='d', order=bands_present, errorbar=None, color='#3498db', ax=ax)
    if ei_d > 0:
        ax.axhline(ei_d, color='red', linestyle='--', linewidth=2, label=f'Standard EI Baseline (|d|={ei_d:.3f})')
        ax.legend()
    ax.set_xticklabels(band_labels, rotation=45, ha='right')
    ax.set_ylabel("Mean |Cohen's d|")
    ax.set_title("Statistical Class Separation via Cohen's d (by Frequency Band)", pad=15, fontweight='bold')
    ax.grid(True, axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(viz_dir / "viz17_0.1.1_cohend_bands.png", dpi=200, bbox_inches='tight')
    plt.close()
    
    # 0.1.2: Electrode Barplot
    elec_order = ['TP9', 'AF7', 'AF8', 'TP10']
    elecs_present = [e for e in elec_order if e in df['Electrode'].unique()] + [e for e in df['Electrode'].unique() if e not in elec_order]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(data=df, x='Electrode', y='d', order=elecs_present, errorbar=None, color='#e74c3c', ax=ax)
    if ei_d > 0:
        ax.axhline(ei_d, color='red', linestyle='--', linewidth=2, label=f'Standard EI Baseline (|d|={ei_d:.3f})')
        ax.legend()
    ax.set_ylabel("Mean |Cohen's d|")
    ax.set_title("Statistical Class Separation via Cohen's d (by Electrode)", pad=15, fontweight='bold')
    ax.grid(True, axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(viz_dir / "viz17_0.1.2_cohend_electrodes.png", dpi=200, bbox_inches='tight')
    plt.close()
    
    # 0.1.3: Statistic Barplot
    stat_order = df.groupby('Stat')['d'].mean().sort_values(ascending=False).index
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(data=df, x='Stat', y='d', order=stat_order, errorbar=None, color='#2ecc71', ax=ax)
    if ei_d > 0:
        ax.axhline(ei_d, color='red', linestyle='--', linewidth=2, label=f'Standard EI Baseline (|d|={ei_d:.3f})')
        ax.legend()
    ax.set_xticklabels(stat_order, rotation=45, ha='right')
    ax.set_ylabel("Mean |Cohen's d|")
    ax.set_title("Statistical Class Separation via Cohen's d (by Statistic)", pad=15, fontweight='bold')
    ax.grid(True, axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(viz_dir / "viz17_0.1.3_cohend_stats.png", dpi=200, bbox_inches='tight')
    plt.close()
    
    # 0.1.4: Abstract Cross-Pattern Interaction Profile
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.pointplot(data=df, x='Band', y='d', hue='Electrode', order=bands_present, hue_order=elecs_present, 
                  dodge=True, markers=['o','s','D','^'], linestyles=['-','--','-.',':'], palette='Set1', ax=ax)
    if ei_d > 0:
        ax.axhline(ei_d, color='red', linestyle='--', linewidth=2, label=f'Standard EI Baseline (|d|={ei_d:.3f})')
        ax.legend()
    ax.set_xticklabels(band_labels, rotation=45, ha='right')
    ax.set_ylabel("Mean |Cohen's d|")
    ax.set_title("Interaction Profile: Frequency Band vs Electrode Topology", pad=15, fontweight='bold')
    ax.grid(True, axis='both', linestyle='--', alpha=0.6)
    plt.legend(title='Electrode', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(viz_dir / "viz17_0.1.4_cohend_interactions.png", dpi=200, bbox_inches='tight')
    plt.close()
    
    # Generate TEX Table
    tex_lines = [
        "\\begin{table}[H]",
        "\\centering",
        "\\begin{tabular}{llc}",
        "\\hline",
        "Rank & Feature Name & $|$Cohen's $d|$ \\\\",
        "\\hline"
    ]
    for i, (name, d) in enumerate(zip(names, ds), 1):
        safe_name = name.replace('_', '\\_')
        tex_lines.append(f"{i} & {safe_name} & {d:.3f} \\\\")
        
    tex_lines.extend([
        "\\hline",
        "\\end{tabular}",
        f"\\caption{{Top {top_n} Features by Separation ($|$Cohen's $d|$).}}",
        "\\label{tab:cohend_top_features}",
        "\\end{table}"
    ])
    
    with open(viz_dir / "viz17_3_cohend_top_features.tex", "w") as f:
        f.write("\n".join(tex_lines))
        
    # Generate Method TEX
    method_content = """To quantify the theoretical separability of the extracted features prior to any model training, Cohen's $d$ is utilized. As a standardized effect size, Cohen's $d$ provides a mathematically pure measure of class separation that is invariant to the absolute scale or unit of the feature (e.g., microvolts versus power spectral density). It answers precisely how many standard deviations separate the centers of the two distinct classes.

The calculation operates on the window level, utilizing the entire global pool of validated EEG windows from all participants simultaneously. For any given engineered feature, the arithmetic mean is calculated across all windows labeled as STAY ($\\mu_{stay}$) and across all windows labeled as SKIP ($\\mu_{skip}$). Next, the variance within the STAY class ($\\sigma^2_{stay}$) and the variance within the SKIP class ($\\sigma^2_{skip}$) are calculated. These are averaged to compute the pooled variance, whose square root yields the pooled standard deviation ($s_p$), representing the average noise of the feature regardless of its class label:
\\[ s_p = \\sqrt{\\frac{\\sigma^2_{stay} + \\sigma^2_{skip}}{2}} \\]

Finally, Cohen's $d$ is calculated by taking the absolute difference between the two class means and dividing it by this pooled standard deviation:
\\[ |d| = \\frac{|\\mu_{stay} - \\mu_{skip}|}{s_p} \\]

By standardizing the difference in means against the inherent variance of the data, this calculation simultaneously rewards features where the class centers are far apart while heavily penalizing features that exhibit excessive noise or overlap. This allows features of entirely different natures to be ranked on a single, standardized axis of separability.
"""
    with open(viz_dir / "viz17_3_cohend_method.tex", "w") as f:
        f.write(method_content)
        
    # Generate Results TEX dynamically
    if len(names) >= 3:
        f1_name, f1_d = names[0].replace('_', '\\_'), ds[0]
        f2_name, f2_d = names[1].replace('_', '\\_'), ds[1]
        f3_name, f3_d = names[2].replace('_', '\\_'), ds[2]
        
        diff_1_2 = f1_d - f2_d
        diff_2_3 = f2_d - f3_d
        
        results_content = f"""Applying this methodology to the full extracted feature set reveals clear frontrunners in class separability. The highest-ranking feature, {f1_name}, achieves a separation of $|d| = {f1_d:.3f}$ standard deviations between the STAY and SKIP classes. This leading feature provides a separation magnitude that is {diff_1_2:.3f} standard deviations wider than the second-best feature, {f2_name} ($|d| = {f2_d:.3f}$). 

Furthermore, the second-best feature maintains a separation advantage of {diff_2_3:.3f} standard deviations over the third-ranked feature, {f3_name} ($|d| = {f3_d:.3f}$). This objective standardization confirms that the uppermost features possess a notably distinct capacity to cleanly divide the dataset independent of any underlying machine learning classification bounds."""
        
        with open(viz_dir / "viz17_3_cohend_results.tex", "w") as f:
            f.write(results_content)
    
    # Reverse to have the highest on top in barh
    names = names[::-1]
    ds = ds[::-1]
    
    fig, ax = plt.subplots(figsize=(10, 12))
    fig.suptitle(f"Top {top_n} Features by Separation (|Cohen's d|)", fontsize=16, fontweight='bold')
    
    sns.barplot(x=ds, y=names, palette="viridis", ax=ax)
    
    ax.set_xlabel("|Cohen's d|", fontsize=12)
    ax.set_ylabel("Feature Name", fontsize=12)
    ax.grid(True, axis='x', linestyle='--', alpha=0.7)
    
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    
    plt.savefig(viz_dir / "viz17_3_cohend_top_features.png", dpi=200, bbox_inches='tight')
    plt.close()
    
if __name__ == "__main__":
    print("Use run.py to execute.")
