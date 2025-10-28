"""Statistical analysis with proper rigor.

Implements:
- Multiple independent runs per configuration
- Proper variance reporting with confidence intervals
- Power analysis for statistical tests
- Aggregation over different seeds
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Dict
import warnings

import numpy as np
import pandas as pd
from scipy import stats

from ..utils.logger import get_logger

LOGGER = get_logger("evaluation.statistical_analysis")


@dataclass
class ComparisonResult:
    """Result of statistical comparison between two methods."""
    
    method_a: str
    method_b: str
    metric: str
    
    # Sample statistics
    mean_a: float
    std_a: float
    n_a: int
    
    mean_b: float
    std_b: float
    n_b: int
    
    # Difference
    mean_diff: float
    ci_lower: float
    ci_upper: float
    
    # Statistical tests
    t_statistic: float
    p_value_ttest: float
    p_value_wilcoxon: float
    
    # Effect size
    cohens_d: float
    effect_size_interpretation: str
    
    # Power analysis
    statistical_power: float
    min_detectable_effect: float
    
    # Conclusion
    significant: bool
    alpha: float


def compute_confidence_interval(
    data: np.ndarray,
    confidence: float = 0.95
) -> Tuple[float, float, float]:
    """Compute mean and confidence interval.
    
    Args:
        data: Array of measurements
        confidence: Confidence level (default 0.95 for 95% CI)
    
    Returns:
        Tuple of (mean, ci_lower, ci_upper)
    """
    mean = np.mean(data)
    std_err = stats.sem(data)
    ci_range = std_err * stats.t.ppf((1 + confidence) / 2, len(data) - 1)
    return mean, mean - ci_range, mean + ci_range


def cohens_d(group_a: np.ndarray, group_b: np.ndarray) -> float:
    """Compute Cohen's d effect size.
    
    Args:
        group_a: First group measurements
        group_b: Second group measurements
    
    Returns:
        Cohen's d effect size
    """
    mean_a = np.mean(group_a)
    mean_b = np.mean(group_b)
    std_a = np.std(group_a, ddof=1)
    std_b = np.std(group_b, ddof=1)
    n_a = len(group_a)
    n_b = len(group_b)
    
    # Pooled standard deviation
    pooled_std = np.sqrt(((n_a - 1) * std_a**2 + (n_b - 1) * std_b**2) / (n_a + n_b - 2))
    
    if pooled_std == 0:
        return 0.0
    
    return (mean_a - mean_b) / pooled_std


def interpret_effect_size(d: float) -> str:
    """Interpret Cohen's d effect size.
    
    Args:
        d: Cohen's d value
    
    Returns:
        Interpretation string
    """
    abs_d = abs(d)
    if abs_d < 0.2:
        return "negligible"
    elif abs_d < 0.5:
        return "small"
    elif abs_d < 0.8:
        return "medium"
    else:
        return "large"


def compute_statistical_power(
    group_a: np.ndarray,
    group_b: np.ndarray,
    alpha: float = 0.05
) -> float:
    """Compute statistical power of the test.
    
    Uses post-hoc power analysis based on observed effect size.
    
    Args:
        group_a: First group measurements
        group_b: Second group measurements
        alpha: Significance level
    
    Returns:
        Statistical power (0-1)
    """
    n_a = len(group_a)
    n_b = len(group_b)
    effect_size = abs(cohens_d(group_a, group_b))
    
    # Degrees of freedom
    df = n_a + n_b - 2
    
    # Non-centrality parameter
    ncp = effect_size * np.sqrt((n_a * n_b) / (n_a + n_b))
    
    # Critical value
    t_crit = stats.t.ppf(1 - alpha/2, df)
    
    # Power = P(reject H0 | H1 is true)
    # = P(|T| > t_crit | ncp)
    power = 1 - stats.nct.cdf(t_crit, df, ncp) + stats.nct.cdf(-t_crit, df, ncp)
    
    return float(power)


def minimum_detectable_effect(
    n_a: int,
    n_b: int,
    alpha: float = 0.05,
    power: float = 0.80
) -> float:
    """Compute minimum detectable effect size.
    
    Args:
        n_a: Sample size group A
        n_b: Sample size group B
        alpha: Significance level
        power: Desired power
    
    Returns:
        Minimum detectable Cohen's d
    """
    df = n_a + n_b - 2
    t_crit = stats.t.ppf(1 - alpha/2, df)
    t_power = stats.t.ppf(power, df)
    
    # Approximate MDE
    mde = (t_crit + t_power) * np.sqrt((n_a + n_b) / (n_a * n_b))
    
    return float(mde)


def compare_methods(
    measurements_a: List[float],
    measurements_b: List[float],
    method_a_name: str,
    method_b_name: str,
    metric_name: str,
    alpha: float = 0.05,
    confidence: float = 0.95
) -> ComparisonResult:
    """Compare two methods with full statistical rigor.
    
    Args:
        measurements_a: List of measurements for method A
        measurements_b: List of measurements for method B
        method_a_name: Name of method A
        method_b_name: Name of method B
        metric_name: Name of metric being compared
        alpha: Significance level
        confidence: Confidence level for intervals
    
    Returns:
        ComparisonResult with all statistics
    """
    data_a = np.array(measurements_a)
    data_b = np.array(measurements_b)
    
    # Basic statistics
    mean_a, ci_a_lower, ci_a_upper = compute_confidence_interval(data_a, confidence)
    mean_b, ci_b_lower, ci_b_upper = compute_confidence_interval(data_b, confidence)
    
    # Difference
    diff = data_a - data_b if len(data_a) == len(data_b) else None
    if diff is not None:
        mean_diff, ci_diff_lower, ci_diff_upper = compute_confidence_interval(diff, confidence)
    else:
        mean_diff = mean_a - mean_b
        # Approximate CI for independent samples
        se_diff = np.sqrt(np.var(data_a, ddof=1)/len(data_a) + np.var(data_b, ddof=1)/len(data_b))
        t_crit = stats.t.ppf((1 + confidence) / 2, len(data_a) + len(data_b) - 2)
        ci_diff_lower = mean_diff - t_crit * se_diff
        ci_diff_upper = mean_diff + t_crit * se_diff
    
    # Statistical tests
    if diff is not None:
        # Paired t-test
        t_stat, p_ttest = stats.ttest_rel(data_a, data_b)
        # Wilcoxon signed-rank test
        try:
            _, p_wilcoxon = stats.wilcoxon(data_a, data_b, alternative='two-sided')
        except:
            p_wilcoxon = np.nan
    else:
        # Independent t-test
        t_stat, p_ttest = stats.ttest_ind(data_a, data_b)
        # Mann-Whitney U test
        try:
            _, p_wilcoxon = stats.mannwhitneyu(data_a, data_b, alternative='two-sided')
        except:
            p_wilcoxon = np.nan
    
    # Effect size
    d = cohens_d(data_a, data_b)
    effect_interp = interpret_effect_size(d)
    
    # Power analysis
    power = compute_statistical_power(data_a, data_b, alpha)
    mde = minimum_detectable_effect(len(data_a), len(data_b), alpha, 0.80)
    
    # Conclusion
    significant = p_ttest < alpha
    
    result = ComparisonResult(
        method_a=method_a_name,
        method_b=method_b_name,
        metric=metric_name,
        mean_a=float(mean_a),
        std_a=float(np.std(data_a, ddof=1)),
        n_a=len(data_a),
        mean_b=float(mean_b),
        std_b=float(np.std(data_b, ddof=1)),
        n_b=len(data_b),
        mean_diff=float(mean_diff),
        ci_lower=float(ci_diff_lower),
        ci_upper=float(ci_diff_upper),
        t_statistic=float(t_stat),
        p_value_ttest=float(p_ttest),
        p_value_wilcoxon=float(p_wilcoxon) if not np.isnan(p_wilcoxon) else None,
        cohens_d=float(d),
        effect_size_interpretation=effect_interp,
        statistical_power=float(power),
        min_detectable_effect=float(mde),
        significant=bool(significant),
        alpha=alpha
    )
    
    return result


def aggregate_multiple_seeds(
    results_by_seed: Dict[int, Dict],
    metric_name: str
) -> Dict:
    """Aggregate results across multiple random seeds.
    
    Args:
        results_by_seed: Dict mapping seed -> results dict
        metric_name: Name of metric to aggregate
    
    Returns:
        Dict with aggregated statistics
    """
    values = [results[metric_name] for results in results_by_seed.values()]
    data = np.array(values)
    
    mean, ci_lower, ci_upper = compute_confidence_interval(data)
    
    return {
        'metric': metric_name,
        'mean': float(mean),
        'std': float(np.std(data, ddof=1)),
        'ci_lower': float(ci_lower),
        'ci_upper': float(ci_upper),
        'n_seeds': len(values),
        'seeds': list(results_by_seed.keys()),
        'values': values
    }


def check_assumptions(
    data_a: np.ndarray,
    data_b: np.ndarray
) -> Dict[str, bool]:
    """Check statistical test assumptions.
    
    Args:
        data_a: First group data
        data_b: Second group data
    
    Returns:
        Dict with assumption check results
    """
    results = {}
    
    # Normality (Shapiro-Wilk test)
    if len(data_a) >= 3:
        _, p_norm_a = stats.shapiro(data_a)
        results['normality_a'] = p_norm_a > 0.05
    else:
        results['normality_a'] = None
    
    if len(data_b) >= 3:
        _, p_norm_b = stats.shapiro(data_b)
        results['normality_b'] = p_norm_b > 0.05
    else:
        results['normality_b'] = None
    
    # Equal variances (Levene's test)
    if len(data_a) >= 2 and len(data_b) >= 2:
        _, p_var = stats.levene(data_a, data_b)
        results['equal_variances'] = p_var > 0.05
    else:
        results['equal_variances'] = None
    
    return results

