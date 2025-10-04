"""
Visualization utilities for TVD analysis.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from typing import Dict, List
from collections import defaultdict


def plot_tvd_distributions(results: Dict, save_path: str = None):
    """
    Plot histogram and density of TVD distributions for models and oracle.
    
    Args:
        results: Dictionary from evaluate_models_with_sampling
        save_path: Optional path to save figure
    """
    tvd_orig = results["original_tvds"]
    tvd_fine = results["finetuned_tvds"]
    oracle_tvds = results.get("oracle_tvds")

    # Histogram with mean lines
    plt.figure(figsize=(10, 6))
    colors = {
        "GPT2 vs Human": "skyblue",
        "Fine-tuned vs Human": "orange",
        "Oracle (Human vs Human)": "green"
    }
    
    all_tvds = {
        "GPT2 vs Human": tvd_orig,
        "Fine-tuned vs Human": tvd_fine
    }
    if oracle_tvds:
        all_tvds["Oracle (Human vs Human)"] = oracle_tvds

    for label, values in all_tvds.items():
        color = colors[label]
        plt.hist(values, bins=20, alpha=0.5, label=label, color=color)
        mean_val = np.mean(values)
        plt.axvline(x=mean_val, color=color, linestyle='--',
                   label=f"{label} Mean: {mean_val:.3f}")

    plt.xlabel("TVD")
    plt.ylabel("Frequency")
    plt.title("TVD Distribution Across Models")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_tvd_density(results: Dict, save_path: str = None):
    """
    Plot kernel density estimation of TVD distributions.
    
    Args:
        results: Dictionary from evaluate_models_with_sampling
        save_path: Optional path to save figure
    """
    tvd_orig = results["original_tvds"]
    tvd_fine = results["finetuned_tvds"]
    oracle_tvds = results.get("oracle_tvds")

    plt.figure(figsize=(12, 8))
    colors = ["skyblue", "orange", "green"]
    x = np.linspace(0, 1, 1000)

    all_tvds = {
        "GPT2 vs Human": tvd_orig,
        "Fine-tuned vs Human": tvd_fine
    }
    if oracle_tvds:
        all_tvds["Oracle (Human vs Human)"] = oracle_tvds

    for i, (label, values) in enumerate(all_tvds.items()):
        color = colors[i]
        if len(values) > 1:
            kde = stats.gaussian_kde(values)
            y = kde(x)
            plt.plot(x, y, label=label, color=color, linewidth=2)
            plt.fill_between(x, y, alpha=0.2, color=color)
            plt.axvline(x=np.mean(values), color=color, linestyle='--',
                       label=f"{label} Mean: {np.mean(values):.3f}")

    plt.title("TVD Density Comparison", fontsize=14)
    plt.xlabel("Total Variation Distance (TVD)")
    plt.ylabel("Density")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_feature_correlation(features: Dict, save_path: str = None):
    """
    Plot scatter plots of features vs TVD improvement with correlation.
    
    Args:
        features: Dictionary from extract_all_features
        save_path: Optional path prefix for saving figures
    """
    from scipy.stats import pearsonr
    
    numeric_features = ["ctx_len", "entropy", "avg_tok_len", "human_skew", 
                       "human_kurtosis", "word_position"]
    
    for feat in numeric_features:
        corr, pval = pearsonr(features[feat], features["tvd_improv"])
        print(f"Correlation between {feat} and TVD improvement: "
              f"r={corr:.3f}, p={pval:.4f}")

        plt.figure(figsize=(8, 6))
        plt.scatter(features[feat], features["tvd_improv"], alpha=0.6)
        plt.title(f"TVD Improvement vs {feat}\n(r={corr:.3f}, p={pval:.4f})")
        plt.xlabel(feat)
        plt.ylabel("Δ TVD (original - fine-tuned)")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(f"{save_path}_{feat}.png", dpi=300, bbox_inches='tight')
        plt.show()


def plot_pos_analysis(features: Dict, save_path: str = None):
    """
    Analyze and plot TVD improvement by POS tags.
    
    Args:
        features: Dictionary from extract_all_features
        save_path: Optional path to save figure
    """
    # Next word POS
    pos_buckets = defaultdict(list)
    for pos, tvd in zip(features["next_pos"], features["tvd_improv"]):
        pos_buckets[pos].append(tvd)

    print("\nAverage TVD improvement by POS tag (next word):")
    pos_stats = []
    for pos, vals in sorted(pos_buckets.items(), key=lambda x: -np.mean(x[1])):
        if len(vals) > 3:
            pos_stats.append({
                'POS': pos,
                'Mean': np.mean(vals),
                'Std': np.std(vals),
                'Count': len(vals)
            })
            print(f"{pos:5}: {np.mean(vals):.4f} ± {np.std(vals):.4f} (n={len(vals)})")
    
    # Plot
    if pos_stats:
        df_pos = pd.DataFrame(pos_stats)
        plt.figure(figsize=(12, 6))
        plt.bar(df_pos['POS'], df_pos['Mean'], yerr=df_pos['Std'], alpha=0.7, capsize=5)
        plt.title("Average TVD Improvement by POS Tag (Next Word)")
        plt.xlabel("POS Tag")
        plt.ylabel("Mean Δ TVD")
        plt.xticks(rotation=45)
        plt.grid(True, axis='y', alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


def plot_word_position_analysis(features: Dict, save_path: str = None):
    """
    Analyze TVD improvement by word position in paragraph.
    
    Args:
        features: Dictionary from extract_all_features
        save_path: Optional path to save figure
    """
    # Binned analysis
    df_pos = pd.DataFrame({
        "word_position": features["word_position"],
        "tvd_improvement": features["tvd_improv"]
    })

    # Create bins
    bin_edges = np.arange(0, df_pos["word_position"].max() + 10, 10)
    df_pos["pos_bin"] = pd.cut(df_pos["word_position"], bins=bin_edges)

    # Group and summarize
    grouped = df_pos.groupby("pos_bin")["tvd_improvement"].agg(
        ["mean", "std", "count"]
    ).reset_index()

    # Plot
    plt.figure(figsize=(10, 6))
    plt.errorbar(
        x=range(len(grouped)),
        y=grouped["mean"],
        yerr=grouped["std"],
        fmt='o-', capsize=5, linewidth=2, markersize=8
    )
    plt.xticks(range(len(grouped)), grouped["pos_bin"].astype(str), rotation=45)
    plt.title("TVD Improvement vs Word Position in Paragraph")
    plt.xlabel("Word Position Bin")
    plt.ylabel("Δ TVD (original - fine-tuned)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_correlation_heatmap(features: Dict, save_path: str = None):
    """
    Plot correlation heatmap of all numeric features.
    
    Args:
        features: Dictionary from extract_all_features
        save_path: Optional path to save figure
    """
    feature_df = pd.DataFrame({
        "Context Length": features["ctx_len"],
        "Entropy": features["entropy"],
        "Avg Token Length": features["avg_tok_len"],
        "Skewness": features["human_skew"],
        "Kurtosis": features["human_kurtosis"],
        "Word Position": features["word_position"],
        "TVD Improvement": features["tvd_improv"]
    })

    # Compute correlation matrix
    corr_matrix = feature_df.corr()

    # Plot heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f",
                square=True, cbar_kws={"shrink": 0.8})
    plt.title("Correlation Matrix of Features", fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
