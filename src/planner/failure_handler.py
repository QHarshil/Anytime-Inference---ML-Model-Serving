"""Graceful degradation utilities."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd

from ..utils.logger import get_logger

LOGGER = get_logger("planner.failure")


@dataclass
class FallbackResult:
    config: Dict
    reason: str


class FailureHandler:
    """Handle planner failure modes and provide sensible fallbacks."""

    def __init__(self, profiles: pd.DataFrame) -> None:
        self.profiles = profiles
        self.fastest_configs: Dict[str, Dict] = {}
        for task in profiles["task"].unique():
            task_profiles = profiles[profiles["task"] == task]
            if task_profiles.empty:
                continue
            fastest = task_profiles.loc[task_profiles["lat_p50_ms"].idxmin()]
            self.fastest_configs[task] = fastest.to_dict()

    def handle_deadline_miss(
        self,
        task: str,
        deadline_ms: float,
        selected_config: Optional[Dict] = None,
    ) -> FallbackResult:
        """Handle the case where no configuration meets the deadline."""

        fallback = dict(self.fastest_configs[task])
        fallback["fallback_reason"] = "deadline_miss"
        fallback["original_deadline"] = deadline_ms
        fallback["expected_miss_ms"] = fallback.get("lat_p95_ms", np.nan) - deadline_ms

        LOGGER.warning("All configs miss %s ms deadline for %s", deadline_ms, task)
        LOGGER.warning("Falling back to %s", fallback.get("config_id", "<unknown>"))
        return FallbackResult(config=fallback, reason="deadline_miss")

    def handle_model_crash(self, task: str, crashed_config: Dict) -> FallbackResult:
        """Handle model crash by selecting an alternative configuration."""

        fallback = dict(self.fastest_configs[task])
        if fallback.get("config_id") == crashed_config.get("config_id"):
            task_profiles = self.profiles[self.profiles["task"] == task]
            task_profiles = task_profiles[task_profiles["config_id"] != crashed_config.get("config_id")]
            if not task_profiles.empty:
                fallback = task_profiles.loc[task_profiles["lat_p50_ms"].idxmin()].to_dict()

        fallback["fallback_reason"] = "model_crash"
        fallback["crashed_config"] = crashed_config.get("config_id")
        LOGGER.error("Model crashed: %s", crashed_config.get("config_id"))
        LOGGER.warning("Using fallback config %s", fallback.get("config_id"))
        return FallbackResult(config=fallback, reason="model_crash")

    def compute_degradation_metrics(self, results_df: pd.DataFrame) -> Dict[str, float]:
        """Compute degradation metrics from evaluation results."""

        if results_df.empty:
            return {
                "deadline_miss_rate": 0.0,
                "avg_miss_magnitude_ms": 0.0,
                "fallback_usage_rate": 0.0,
                "accuracy_under_degradation": 0.0,
            }

        miss_rate = 1.0 - results_df["deadline_hit_rate"].mean()
        misses = results_df[results_df["lat_p95_ms"] > results_df["deadline"]]
        avg_miss_ms = float((misses["lat_p95_ms"] - misses["deadline"]).mean()) if not misses.empty else 0.0

        if "fallback_reason" in results_df.columns:
            fallback_rate = float(results_df["fallback_reason"].notna().mean())
        else:
            fallback_rate = 0.0

        accuracy_under_degradation = (
            float(misses["accuracy"].mean()) if not misses.empty else float(results_df["accuracy"].mean())
        )

        return {
            "deadline_miss_rate": float(miss_rate),
            "avg_miss_magnitude_ms": float(avg_miss_ms),
            "fallback_usage_rate": float(fallback_rate),
            "accuracy_under_degradation": float(accuracy_under_degradation),
        }
