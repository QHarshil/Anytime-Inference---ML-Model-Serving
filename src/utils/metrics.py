"""Evaluation metrics for the Anytime Inference Planner."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass
class PlannerMetrics:
    """Container for aggregated planner metrics."""

    deadline_hit_rate: float
    accuracy: float
    avg_latency_ms: float
    throughput_rps: float


def compute_hit_rate(latencies_ms: Iterable[float], deadline_ms: float) -> float:
    """Compute the fraction of latencies that meet the deadline."""

    latencies = np.asarray(list(latencies_ms))
    if latencies.size == 0:
        return 0.0
    return float(np.mean(latencies <= deadline_ms))


def compute_throughput(latencies_ms: Iterable[float]) -> float:
    """Estimate throughput in requests per second given per-request latency."""

    latencies = np.asarray(list(latencies_ms))
    if latencies.size == 0:
        return 0.0
    mean_latency = np.mean(latencies) / 1000.0  # convert to seconds
    if mean_latency <= 0:
        return 0.0
    return 1.0 / mean_latency


def summarise_results(df: pd.DataFrame) -> PlannerMetrics:
    """Summarise evaluation results stored in ``df``."""

    if df.empty:
        return PlannerMetrics(0.0, 0.0, 0.0, 0.0)

    return PlannerMetrics(
        deadline_hit_rate=float(df["deadline_hit_rate"].mean()),
        accuracy=float(df["accuracy"].mean()),
        avg_latency_ms=float(df["lat_p95_ms"].mean()),
        throughput_rps=float(df.get("throughput_rps", pd.Series(dtype=float)).mean()
                            if "throughput_rps" in df else 0.0),
    )
