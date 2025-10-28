"""Deadline-aware configuration selection."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import pandas as pd

from ..utils.logger import get_logger
from .failure_handler import FailureHandler, FallbackResult

LOGGER = get_logger("planner.planner")


@dataclass
class PlannerDecision:
    config: Dict
    fallback: Optional[FallbackResult]


class CascadePlanner:
    """Select configurations that maximise accuracy under latency deadlines."""

    def __init__(self, profiles: pd.DataFrame) -> None:
        if profiles.empty:
            raise ValueError("Profiles dataframe must not be empty")
        self.profiles = profiles
        self.failure_handler = FailureHandler(profiles)

    def _filter_by_deadline(self, task: str, deadline_ms: float) -> pd.DataFrame:
        task_profiles = self.profiles[self.profiles["task"] == task]
        feasible = task_profiles[task_profiles["lat_p95_ms"] <= deadline_ms]
        return feasible

    def select(self, task: str, deadline_ms: float, workload: str = "steady") -> PlannerDecision:
        LOGGER.info("Selecting config for task=%s deadline=%sms workload=%s", task, deadline_ms, workload)
        feasible = self._filter_by_deadline(task, deadline_ms)

        if feasible.empty:
            fallback = self.failure_handler.handle_deadline_miss(task, deadline_ms)
            return PlannerDecision(config=fallback.config, fallback=fallback)

        # Workload adjustment: prefer larger batches for steady workloads
        if workload == "steady":
            feasible = feasible.sort_values(["batch_size", "accuracy"], ascending=[False, False])
        elif workload == "bursty":
            feasible = feasible.sort_values(["batch_size", "accuracy"], ascending=[True, False])
        else:
            feasible = feasible.sort_values(["accuracy"], ascending=False)

        best = feasible.iloc[0].to_dict()
        return PlannerDecision(config=best, fallback=None)
