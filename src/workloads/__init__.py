"""Synthetic workload generators."""

from .traces import SteadyWorkload, BurstyWorkload
from .trace_analyzer import compute_trace_statistics

__all__ = [
    "SteadyWorkload",
    "BurstyWorkload",
    "compute_trace_statistics",
]
