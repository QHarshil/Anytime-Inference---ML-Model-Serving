"""Workload trace analysis."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .traces import WorkloadTrace


@dataclass
class TraceStatistics:
    avg_rate_rps: float
    peak_rate_rps: float
    utilisation: float


def summarise_trace(trace: WorkloadTrace, service_time_ms: float) -> TraceStatistics:
    rates = trace.rates_rps
    avg_rate = float(np.mean(rates))
    peak_rate = float(np.max(rates))
    utilisation = avg_rate * (service_time_ms / 1000.0)
    return TraceStatistics(avg_rate_rps=avg_rate, peak_rate_rps=peak_rate, utilisation=utilisation)
