"""Synthetic workload generators."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class WorkloadTrace:
    timestamps_ms: np.ndarray
    rates_rps: np.ndarray


def steady_workload(duration_s: float, rate_rps: float, step_ms: float = 10.0) -> WorkloadTrace:
    timestamps = np.arange(0, duration_s * 1000, step_ms)
    rates = np.full_like(timestamps, rate_rps, dtype=float)
    return WorkloadTrace(timestamps_ms=timestamps, rates_rps=rates)


def bursty_workload(duration_s: float, low_rps: float, high_rps: float, period_s: float = 5.0, step_ms: float = 10.0) -> WorkloadTrace:
    timestamps = np.arange(0, duration_s * 1000, step_ms)
    rates = np.zeros_like(timestamps, dtype=float)
    half_period_ms = (period_s * 1000) / 2.0
    for idx, ts in enumerate(timestamps):
        phase = int((ts % (period_s * 1000)) < half_period_ms)
        rates[idx] = high_rps if phase else low_rps
    return WorkloadTrace(timestamps_ms=timestamps, rates_rps=rates)
