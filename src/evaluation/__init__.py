"""Evaluation utilities (statistical analysis, real inference)."""

from .statistical_analysis import (
    ComparisonResult,
    aggregate_multiple_seeds,
    check_assumptions,
    cohens_d,
    compare_methods,
    compute_confidence_interval,
    compute_statistical_power,
    interpret_effect_size,
)

# real_inference is imported lazily because it pulls in torch, which is not
# required for code paths that only need statistical analysis.

__all__ = [
    "ComparisonResult",
    "aggregate_multiple_seeds",
    "check_assumptions",
    "cohens_d",
    "compare_methods",
    "compute_confidence_interval",
    "compute_statistical_power",
    "interpret_effect_size",
]
