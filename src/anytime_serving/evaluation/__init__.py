"""Evaluation utilities (statistical analysis, Pareto frontiers, real inference)."""

from .pareto import (
    compute_hypervolume,
    compute_pareto_frontier,
    dominance_ratio,
    is_dominated,
)
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
    "compute_hypervolume",
    "compute_pareto_frontier",
    "compute_statistical_power",
    "dominance_ratio",
    "interpret_effect_size",
    "is_dominated",
]
