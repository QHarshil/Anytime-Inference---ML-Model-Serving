"""Latency profiling utilities."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

import pandas as pd

from ..models.model_zoo import ModelZoo
from ..utils.logger import get_logger
from .profiler_utils import measure_latencies, warmup

LOGGER = get_logger("profiler.latency")


@dataclass
class LatencyProfile:
    task: str
    model: str
    variant: str
    device: str
    batch_size: int
    lat_p50_ms: float
    lat_p95_ms: float
    throughput_rps: float


class LatencyProfiler:
    """Profile latency for text and image workloads."""

    def __init__(self, model_zoo: ModelZoo) -> None:
        self.model_zoo = model_zoo

    def profile_text(
        self,
        model: str,
        variant: str,
        device: str,
        texts: List[str],
        *,
        batch_size: int = 8,
    ) -> LatencyProfile:
        LOGGER.info("Profiling text latency for %s (%s)", model, variant)

        def run_once() -> None:
            self.model_zoo.predict_text(model, variant, device, texts, batch_size=batch_size)

        warmup(run_once)
        lat_p50, lat_p95, throughput = measure_latencies(run_once)

        return LatencyProfile(
            task="text",
            model=model,
            variant=variant,
            device=device,
            batch_size=batch_size,
            lat_p50_ms=lat_p50,
            lat_p95_ms=lat_p95,
            throughput_rps=throughput,
        )

    def profile_image(
        self,
        model: str,
        variant: str,
        device: str,
        images,
        *,
        batch_size: int = 8,
    ) -> LatencyProfile:
        LOGGER.info("Profiling image latency for %s (%s)", model, variant)

        def run_once() -> None:
            self.model_zoo.predict_image(model, variant, device, images)

        warmup(run_once)
        lat_p50, lat_p95, throughput = measure_latencies(run_once)

        return LatencyProfile(
            task="vision",
            model=model,
            variant=variant,
            device=device,
            batch_size=batch_size,
            lat_p50_ms=lat_p50,
            lat_p95_ms=lat_p95,
            throughput_rps=throughput,
        )

    @staticmethod
    def to_dataframe(profiles: Iterable[LatencyProfile]) -> pd.DataFrame:
        return pd.DataFrame([profile.__dict__ for profile in profiles])
