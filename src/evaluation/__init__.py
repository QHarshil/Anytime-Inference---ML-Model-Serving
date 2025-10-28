"""Evaluation utilities for real inference measurements and statistical analysis."""

from .real_inference import RealInferenceEvaluator, CascadeInferenceEvaluator, InferenceResult
from .statistical_analysis import (
    ComparisonResult,
    compare_methods,
    aggregate_multiple_seeds,
    compute_confidence_interval,
    cohens_d,
    interpret_effect_size,
    compute_statistical_power,
    check_assumptions
)

__all__ = [
    "RealInferenceEvaluator",
    "CascadeInferenceEvaluator",
    "InferenceResult",
    "ComparisonResult",
    "compare_methods",
    "aggregate_multiple_seeds",
    "compute_confidence_interval",
    "cohens_d",
    "interpret_effect_size",
    "compute_statistical_power",
    "check_assumptions",
]

