"""Pareto frontier utilities."""
from __future__ import annotations

from typing import Iterable, List, Tuple
import pandas as pd


def is_dominated(point: Tuple[float, float], others: Iterable[Tuple[float, float]]) -> bool:
    latency, accuracy = point
    for other_latency, other_accuracy in others:
        if (
            other_latency <= latency
            and other_accuracy >= accuracy
            and (other_latency < latency or other_accuracy > accuracy)
        ):
            return True
    return False


def compute_pareto_frontier(
    df: pd.DataFrame,
    latency_col: str = "lat_p95_ms",
    accuracy_col: str = "accuracy",
) -> pd.DataFrame:
    points = list(zip(df[latency_col], df[accuracy_col]))
    mask = []
    for idx, point in enumerate(points):
        others = [p for j, p in enumerate(points) if j != idx]
        mask.append(not is_dominated(point, others))
    return df[mask].copy()


def compute_hypervolume(
    pareto_points: List[Tuple[float, float]],
    reference_point: Tuple[float, float],
) -> float:
    if not pareto_points:
        return 0.0
    sorted_points = sorted(pareto_points, key=lambda p: p[0])
    ref_latency, ref_accuracy = reference_point

    hypervolume = 0.0
    prev_latency = 0.0
    for latency, accuracy in sorted_points:
        width = latency - prev_latency
        height = max(0.0, accuracy - ref_accuracy)
        hypervolume += max(0.0, width) * height
        prev_latency = latency

    last_latency, last_accuracy = sorted_points[-1]
    width = max(0.0, ref_latency - last_latency)
    height = max(0.0, last_accuracy - ref_accuracy)
    hypervolume += width * height
    return max(0.0, hypervolume)


def dominance_ratio(
    method_df: pd.DataFrame,
    baseline_df: pd.DataFrame,
    latency_col: str = "lat_p95_ms",
    accuracy_col: str = "accuracy",
) -> float:
    method_points = list(zip(method_df[latency_col], method_df[accuracy_col]))
    baseline_points = list(zip(baseline_df[latency_col], baseline_df[accuracy_col]))
    if not baseline_points:
        return 0.0
    dominated = 0
    for point in baseline_points:
        if is_dominated(point, method_points):
            dominated += 1
    return dominated / len(baseline_points)
