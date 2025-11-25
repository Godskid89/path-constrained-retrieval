"""
Statistical analysis for evaluation results.
"""

from typing import List, Dict
import numpy as np
import pandas as pd
from scipy import stats


def compute_statistical_significance(
    method1_scores: List[float],
    method2_scores: List[float],
    alpha: float = 0.05
) -> Dict[str, float]:
    """
    Compute statistical significance between two methods.
    
    Args:
        method1_scores: Scores from method 1
        method2_scores: Scores from method 2
        alpha: Significance level
        
    Returns:
        Dictionary with statistical test results
    """
    if len(method1_scores) != len(method2_scores):
        raise ValueError("Score lists must have same length")
    
    # Paired t-test
    differences = np.array(method1_scores) - np.array(method2_scores)
    t_stat, p_value = stats.ttest_1samp(differences, 0)
    
    # Effect size (Cohen's d)
    mean_diff = np.mean(differences)
    std_diff = np.std(differences, ddof=1)
    cohens_d = mean_diff / std_diff if std_diff > 0 else 0.0
    
    # Confidence interval (95%)
    n = len(differences)
    se = std_diff / np.sqrt(n)
    ci_lower = mean_diff - 1.96 * se
    ci_upper = mean_diff + 1.96 * se
    
    return {
        'mean_difference': float(mean_diff),
        'std_difference': float(std_diff),
        't_statistic': float(t_stat),
        'p_value': float(p_value),
        'significant': p_value < alpha,
        'cohens_d': float(cohens_d),
        'ci_lower': float(ci_lower),
        'ci_upper': float(ci_upper),
        'n_samples': n
    }


def aggregate_with_statistics(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate results with statistical measures.
    
    Args:
        results_df: DataFrame with per-query results
        
    Returns:
        Aggregated DataFrame with mean, std, sem, and confidence intervals
    """
    numeric_cols = [col for col in results_df.columns 
                   if col not in ['method', 'query_id']]
    
    aggregated = results_df.groupby('method')[numeric_cols].agg([
        'mean', 'std', 'count'
    ])
    
    # Compute standard error of mean and confidence intervals
    for col in numeric_cols:
        for method in results_df['method'].unique():
            method_data = results_df[results_df['method'] == method][col]
            n = len(method_data)
            if n > 0:
                mean = method_data.mean()
                std = method_data.std(ddof=1)
                sem = std / np.sqrt(n)  # Standard error of mean
                ci_lower = mean - 1.96 * sem
                ci_upper = mean + 1.96 * sem
                
                # Add to aggregated
                aggregated.loc[(method, col), 'sem'] = sem
                aggregated.loc[(method, col), 'ci_lower'] = ci_lower
                aggregated.loc[(method, col), 'ci_upper'] = ci_upper
    
    return aggregated


def compare_methods_statistically(
    results_df: pd.DataFrame,
    metric: str = 'relevance@10'
) -> pd.DataFrame:
    """
    Compare methods statistically for a given metric.
    
    Args:
        results_df: DataFrame with per-query results
        metric: Metric to compare
        
    Returns:
        DataFrame with statistical comparisons
    """
    methods = results_df['method'].unique()
    comparisons = []
    
    for i, method1 in enumerate(methods):
        for method2 in methods[i+1:]:
            scores1 = results_df[results_df['method'] == method1][metric].tolist()
            scores2 = results_df[results_df['method'] == method2][metric].tolist()
            
            if len(scores1) == len(scores2) and len(scores1) > 0:
                stats_result = compute_statistical_significance(scores1, scores2)
                comparisons.append({
                    'method1': method1,
                    'method2': method2,
                    'metric': metric,
                    **stats_result
                })
    
    return pd.DataFrame(comparisons)

