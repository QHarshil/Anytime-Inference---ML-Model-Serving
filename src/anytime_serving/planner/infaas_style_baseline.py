"""INFaaS-style baseline adapted for offline profiles.

Stands in for the variant-selection policy of INFaaS (Romero et al., USENIX ATC '21,
https://www.usenix.org/conference/atc21/presentation/romero): pick the cheapest
variant that meets the latency target. Adapted rather than reimplemented -- INFaaS
selects over model variants and hardware, autoscaling in a cluster, while this reads
a table of offline profiles -- so it is a comparison point and not a reproduction.
"""

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
            LOGGER.warning(
                "No feasible config for %s under %s ms. Using fastest.", task, latency_target_ms
            )
            best = candidates.loc[candidates["lat_p50_ms"].idxmin()]

        return best.to_dict()

    def select(self, task: str, deadline_ms: float, workload: str = "steady") -> dict:
        return self.select_for_latency_target(task, deadline_ms)
