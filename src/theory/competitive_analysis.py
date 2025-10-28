"""Competitive analysis helpers."""
from __future__ import annotations

import pandas as pd


def compute_competitive_ratio(online_results: pd.DataFrame, optimal_results: pd.DataFrame) -> float:
    """Compute cost(online) / cost(optimal)."""

    if online_results.empty or optimal_results.empty:
        return float("inf")
    online_cost = float(online_results["lat_p50_ms"].sum())
    optimal_cost = float(optimal_results["lat_p50_ms"].sum())
    if optimal_cost <= 0:
        return float("inf")
    return online_cost / optimal_cost
