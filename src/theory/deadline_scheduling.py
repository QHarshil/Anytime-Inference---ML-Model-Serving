"""Deadline scheduling helpers."""
from __future__ import annotations

from typing import Dict, Iterable


def compute_utilization(configs: Iterable[Dict]) -> float:
    """Compute system utilisation U = Σ (execution_time / period)."""

    utilisation = 0.0
    for config in configs:
        exec_time = config.get("lat_p95_ms", 0.0) / 1000.0
        request_rate = config.get("request_rate", 1.0)
        period = 1.0 / max(request_rate, 1e-9)
        utilisation += exec_time / period
    return utilisation


def is_schedulable_rm(utilisation: float, n_tasks: int) -> bool:
    """Check the rate-monotonic schedulability bound."""

    if n_tasks <= 0:
        return False
    bound = n_tasks * ((2 ** (1.0 / n_tasks)) - 1.0)
    return utilisation <= bound
