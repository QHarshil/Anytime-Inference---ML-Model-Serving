"""Shared utilities for latency and accuracy profilers."""
from __future__ import annotations

import time
from typing import Callable, Iterable, Tuple

import numpy as np


def warmup(fn: Callable[[], None], iterations: int = 10) -> None:
    """Run ``fn`` ``iterations`` times without recording measurements."""

    for _ in range(iterations):
        fn()


def measure_latencies(fn: Callable[[], None], iterations: int = 100) -> Tuple[float, float, float]:
    """Measure latency statistics for ``fn``.

    Returns a tuple ``(p50_ms, p95_ms, throughput_rps)``.
    """

    latencies = []
    for _ in range(iterations):
        start = time.perf_counter()
        fn()
        latencies.append((time.perf_counter() - start) * 1000.0)

    p50 = float(np.percentile(latencies, 50))
    p95 = float(np.percentile(latencies, 95))
    mean_latency = float(np.mean(latencies))
    throughput = 0.0 if mean_latency <= 0 else 1000.0 / mean_latency
    return p50, p95, throughput


def compute_accuracy(predictions: Iterable[int], labels: Iterable[int]) -> float:
    """Compute accuracy for integer predictions."""

    preds = np.asarray(list(predictions))
    labels = np.asarray(list(labels))
    if preds.size == 0 or preds.size != labels.size:
        return 0.0
    return float(np.mean(preds == labels))
