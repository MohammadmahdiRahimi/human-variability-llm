"""
Statistical analysis utilities.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from typing import Dict, Tuple


def paired_ttest(original_tvds: np.ndarray, finetuned_tvds: np.ndarray) -> Dict:
    """
    Perform paired t-test to assess significance of TVD improvement.
    
    Args:
        original_tvds: Array of TVDs from original model
        finetuned_tvds: Array of TVDs from fine-tuned model
        
    Returns:
        Dictionary with test statistics
    """
    if len(original_tvds) != len(finetuned_tvds):
        raise ValueError("Arrays must have same length")
    
    tvd_improvements = original_tvds - finetuned_tvds
    
    # Paired t-test
    t_stat, p_val = stats.ttest_rel(original_tvds, finetuned_tvds)
    
    # Normality test on improvements
    shapiro_stat, shapiro_p = stats.shapiro(tvd_improvements)
    
    results = {
        "mean_original": np.mean(original_tvds),
        "mean_finetuned": np.mean(finetuned_tvds),
        "mean_improvement": np.mean(tvd_improvements),
        "std_improvement": np.std(tvd_improvements),
        "t_statistic": t_stat,
        "p_value": p_val,
        "significant": p_val < 0.05,
        "shapiro_statistic": shapiro_stat,
        "shapiro_p": shapiro_p,
        "normal_distribution": shapiro_p > 0.05
    }
    
    print("\n📈 Paired t-test Results:")
    print(f"Mean Original TVD: {results['mean_original']:.4f}")
    print(f"Mean Fine-tuned TVD: {results['mean_finetuned']:.4f}")
    print(f"Mean Improvement: {results['mean_improvement']:.4f}")
    print(f"T-statistic: {results['t_statistic']:.4f}")
    print(f"P-value: {results['p_value']:.4g}")
    
    if results['significant']:
        print("✅ Statistically significant (p < 0.05)")
    else:
        print("❌ Not statistically significant (p ≥ 0.05)")
    
    print(f"\n🔍 Shapiro-Wilk Test for Normality:")
    print(f"Statistic: {results['shapiro_statistic']:.4f}, "
          f"P-value: {results['shapiro_p']:.4f}")
    
    if results['normal_distribution']:
        print("✅ Distribution is approximately normal (validating t-test)")
    else:
        print("⚠️  Distribution may not be normal — consider Wilcoxon test")
    
    return results


def regression_analysis(features: Dict) -> Tuple[sm.regression.linear_model.RegressionResultsWrapper, pd.DataFrame]:
    """
    Perform linear regression to identify predictors of TVD improvement.
    
    Args:
        features: Dictionary from extract_all_features
        
    Returns:
        Tuple of (fitted model, cleaned dataframe)
    """
    # Create DataFrame
    df_features = pd.DataFrame({
        "ctx_len": features["ctx_len"],
        "entropy": features["entropy"],
        "avg_tok_len": features["avg_tok_len"],
        "human_skew": features["human_skew"],
        "human_kurtosis": features["human_kurtosis"],
        "word_position": features["word_position"],
        "tvd_improv": features["tvd_improv"]
    })

    # Drop rows with NaNs or Infs
    df_clean = df_features.replace([np.inf, -np.inf], np.nan).dropna()

    # Define predictors and response
    X = df_clean.drop(columns=["tvd_improv"])
    y = df_clean["tvd_improv"]

    # Add intercept
    X = sm.add_constant(X)

    # Fit model
    model = sm.OLS(y, X).fit()
    
    print("\n📊 Linear Regression Results:")
    print(model.summary())
    
    return model, df_clean


def compute_quartile_statistics(results: Dict) -> pd.DataFrame:
    """
    Compute quartile statistics for TVD distributions.
    
    Args:
        results: Dictionary from evaluate_models_with_sampling
        
    Returns:
        DataFrame with quartile statistics
    """
    df_dist = pd.DataFrame({
        "Original TVD": results.get('original_tvds', []),
        "Fine-tuned TVD": results.get('finetuned_tvds', []),
        "TVD Improvement": np.array(results.get('original_tvds', [])) - 
                          np.array(results.get('finetuned_tvds', []))
    })

    stats_list = []
    for col in df_dist.columns:
        desc = df_dist[col].describe()
        stats_list.append({
            'Metric': col,
            'Count': int(desc['count']),
            'Min': desc['min'],
            'Q1': desc['25%'],
            'Median': desc['50%'],
            'Q3': desc['75%'],
            'Max': desc['max'],
            'Mean': desc['mean'],
            'Std': desc['std']
        })
    
    return pd.DataFrame(stats_list)
