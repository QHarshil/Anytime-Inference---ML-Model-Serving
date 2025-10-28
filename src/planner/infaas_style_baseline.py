"""INFaaS-style baseline adapted for offline profiles."""
from __future__ import annotations

import pandas as pd

from ..utils.logger import get_logger

LOGGER = get_logger("planner.infaas")


class INFaaSStyleBaseline:
    """Select the lowest-latency configuration that meets a deadline."""

    def __init__(self, profiles: pd.DataFrame) -> None:
        self.profiles = profiles

    def select_for_latency_target(self, task: str, latency_target_ms: float) -> dict:
        candidates = self.profiles[self.profiles["task"] == task]
        feasible = candidates[candidates["lat_p95_ms"] <= latency_target_ms]

        if not feasible.empty:
            best = feasible.loc[feasible["lat_p50_ms"].idxmin()]
        else:
            LOGGER.warning("No feasible config for %s under %s ms. Using fastest.", task, latency_target_ms)
            best = candidates.loc[candidates["lat_p50_ms"].idxmin()]

        return best.to_dict()

    def select(self, task: str, deadline_ms: float, workload: str = "steady") -> dict:
        return self.select_for_latency_target(task, deadline_ms)
