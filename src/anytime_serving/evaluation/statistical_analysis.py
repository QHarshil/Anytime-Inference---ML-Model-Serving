"""Statistical comparison utilities: paired/unpaired tests, effect sizes, power."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

from ..utils.logger import get_logger

LOGGER = get_logger("evaluation.statistical_analysis")


@dataclass
class ComparisonResult:
    method_a: str
    method_b: str
    metric: str

    mean_a: float
    std_a: float
    n_a: int

    mean_b: float
    std_b: float
    n_b: int

    mean_diff: float
    ci_lower: float
    ci_upper: float

    t_statistic: float
    p_value_ttest: float
    p_value_wilcoxon: float | None

    cohens_d: float
    effect_size_interpretation: str

    statistical_power: float
    min_detectable_effect: float

    significant: bool
    alpha: float


def compute_confidence_interval(
    data: np.ndarray, confidence: float = 0.95
) -> tuple[float, float, float]:
    """Return (mean, ci_lower, ci_upper) for the given sample."""
    mean = float(np.mean(data))
    std_err = float(stats.sem(data))
    ci_range = std_err * stats.t.ppf((1 + confidence) / 2, len(data) - 1)
    return mean, mean - ci_range, mean + ci_range


def cohens_d(group_a: np.ndarray, group_b: np.ndarray) -> float:
    mean_a = np.mean(group_a)
    mean_b = np.mean(group_b)
    std_a = np.std(group_a, ddof=1)
    std_b = np.std(group_b, ddof=1)
    n_a = len(group_a)
    n_b = len(group_b)
    pooled_std = np.sqrt(((n_a - 1) * std_a**2 + (n_b - 1) * std_b**2) / (n_a + n_b - 2))
    if pooled_std == 0:
        return 0.0
    return float((mean_a - mean_b) / pooled_std)


def interpret_effect_size(d: float) -> str:
    abs_d = abs(d)
    if abs_d < 0.2:
        return "negligible"
    if abs_d < 0.5:
        return "small"
    if abs_d < 0.8:
        return "medium"
    return "large"


def compute_statistical_power(
    group_a: np.ndarray, group_b: np.ndarray, alpha: float = 0.05
) -> float:
    n_a = len(group_a)
    n_b = len(group_b)
    effect_size = abs(cohens_d(group_a, group_b))
    df = n_a + n_b - 2
    ncp = effect_size * np.sqrt((n_a * n_b) / (n_a + n_b))
    t_crit = stats.t.ppf(1 - alpha / 2, df)
    power = 1 - stats.nct.cdf(t_crit, df, ncp) + stats.nct.cdf(-t_crit, df, ncp)
    return float(power)


def minimum_detectable_effect(
    n_a: int, n_b: int, alpha: float = 0.05, power: float = 0.80
) -> float:
    df = n_a + n_b - 2
    t_crit = stats.t.ppf(1 - alpha / 2, df)
    t_power = stats.t.ppf(power, df)
    return float((t_crit + t_power) * np.sqrt((n_a + n_b) / (n_a * n_b)))


def compare_methods(
    measurements_a: list[float],
    measurements_b: list[float],
    method_a_name: str,
    method_b_name: str,
    metric_name: str,
    alpha: float = 0.05,
    confidence: float = 0.95,
) -> ComparisonResult:
    data_a = np.asarray(measurements_a)
    data_b = np.asarray(measurements_b)

    mean_a, _, _ = compute_confidence_interval(data_a, confidence)
    mean_b, _, _ = compute_confidence_interval(data_b, confidence)

    diff = data_a - data_b if len(data_a) == len(data_b) else None
    if diff is not None:
        mean_diff, ci_diff_lower, ci_diff_upper = compute_confidence_interval(diff, confidence)
        t_stat, p_ttest = stats.ttest_rel(data_a, data_b)
        try:
            _, p_wilcoxon = stats.wilcoxon(data_a, data_b, alternative="two-sided")
        except ValueError:
            p_wilcoxon = np.nan
    else:
        mean_diff = mean_a - mean_b
        se_diff = np.sqrt(
            np.var(data_a, ddof=1) / len(data_a) + np.var(data_b, ddof=1) / len(data_b)
        )
        t_crit = stats.t.ppf((1 + confidence) / 2, len(data_a) + len(data_b) - 2)
        ci_diff_lower = mean_diff - t_crit * se_diff
        ci_diff_upper = mean_diff + t_crit * se_diff
        t_stat, p_ttest = stats.ttest_ind(data_a, data_b)
        try:
            _, p_wilcoxon = stats.mannwhitneyu(data_a, data_b, alternative="two-sided")
        except ValueError:
            p_wilcoxon = np.nan

    d = cohens_d(data_a, data_b)
    return ComparisonResult(
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
        effect_size_interpretation=interpret_effect_size(d),
        statistical_power=compute_statistical_power(data_a, data_b, alpha),
        min_detectable_effect=minimum_detectable_effect(len(data_a), len(data_b), alpha, 0.80),
        significant=bool(p_ttest < alpha),
        alpha=alpha,
    )


def aggregate_multiple_seeds(results_by_seed: dict[int, dict], metric_name: str) -> dict:
    values = [results[metric_name] for results in results_by_seed.values()]
    data = np.asarray(values)
    mean, ci_lower, ci_upper = compute_confidence_interval(data)
    return {
        "metric": metric_name,
        "mean": float(mean),
        "std": float(np.std(data, ddof=1)),
        "ci_lower": float(ci_lower),
        "ci_upper": float(ci_upper),
        "n_seeds": len(values),
        "seeds": list(results_by_seed.keys()),
        "values": values,
    }


def check_assumptions(data_a: np.ndarray, data_b: np.ndarray) -> dict[str, bool | None]:
    """Run Shapiro-Wilk (per-group normality) and Levene's (equal variances).

    Keys returned: normality_a, normality_b, equal_variances, normality_ok,
    equal_variance_ok. The "_ok" keys are the AND/joint summaries.
    """
    results: dict[str, bool | None] = {}

    if len(data_a) >= 3:
        _, p_norm_a = stats.shapiro(data_a)
        results["normality_a"] = bool(p_norm_a > 0.05)
    else:
        results["normality_a"] = None

    if len(data_b) >= 3:
        _, p_norm_b = stats.shapiro(data_b)
        results["normality_b"] = bool(p_norm_b > 0.05)
    else:
        results["normality_b"] = None

    if len(data_a) >= 2 and len(data_b) >= 2:
        _, p_var = stats.levene(data_a, data_b)
        results["equal_variances"] = bool(p_var > 0.05)
    else:
        results["equal_variances"] = None

    na, nb = results["normality_a"], results["normality_b"]
    results["normality_ok"] = (
        (na is True) and (nb is True) if na is not None and nb is not None else None
    )
    results["equal_variance_ok"] = results["equal_variances"]
    return results
