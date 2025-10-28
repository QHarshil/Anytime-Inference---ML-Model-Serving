"""Baseline planners."""
from __future__ import annotations

import pandas as pd

from ..utils.logger import get_logger

LOGGER = get_logger("planner.baselines")


class StaticBaseline:
    """Always return the same configuration."""

    def __init__(self, profiles: pd.DataFrame, strategy: str) -> None:
        self.profiles = profiles
        if strategy not in {"fastest", "accurate"}:
            raise ValueError("strategy must be 'fastest' or 'accurate'")
        self.strategy = strategy

    def select(self, task: str) -> dict:
        candidates = self.profiles[self.profiles["task"] == task]
        if candidates.empty:
            raise ValueError(f"No profiles available for task {task}")
        if self.strategy == "fastest":
            best = candidates.loc[candidates["lat_p50_ms"].idxmin()]
        else:
            best = candidates.loc[candidates["accuracy"].idxmax()]
        return best.to_dict()


class ThroughputAutotuner:
    """Simple heuristic: choose configuration based on deadline buckets."""

    def __init__(self, profiles: pd.DataFrame) -> None:
        self.profiles = profiles

    def select(self, task: str, deadline_ms: float) -> dict:
        candidates = self.profiles[self.profiles["task"] == task]
        if candidates.empty:
            raise ValueError(f"No profiles available for task {task}")

        # Heuristic thresholds
        if deadline_ms <= 75:
            # prefer fastest config
            best = candidates.loc[candidates["lat_p50_ms"].idxmin()]
        elif deadline_ms >= 200:
            best = candidates.loc[candidates["accuracy"].idxmax()]
        else:
            feasible = candidates[candidates["lat_p95_ms"] <= deadline_ms]
            if feasible.empty:
                best = candidates.loc[candidates["lat_p50_ms"].idxmin()]
            else:
                # maximise accuracy subject to feasibility
                best = feasible.loc[feasible["accuracy"].idxmax()]
        return best.to_dict()
