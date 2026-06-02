"""Closed-loop serving harness driving the runtime pool under concurrent load.

Combines the load monitor, the M/M/1 admission controller, the adaptive
selector, and the runtime pool into a single object. Requests are accepted
from a producer thread (or any external caller), pass through admission,
are dispatched to a free runtime worker, and the per-request result is
recorded for downstream analysis.
"""
from __future__ import annotations

import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np

from .admission import MM1AdmissionController
from .load_monitor import LoadMonitor
from .onnx_runtime import InferenceRequest, InferenceResponse, RuntimePool
from .selector import AdaptiveSelector, VariantProfile
from ..utils.logger import get_logger

LOGGER = get_logger("serving.server")


@dataclass
class ServedRequest:
    request_id: str
    submitted_at: float
    started_at: Optional[float]
    completed_at: Optional[float]
    deadline_ms: float
    variant: str
    admitted: bool
    rejection_reason: str
    runtime_latency_ms: float
    wall_latency_ms: float
    expected_sojourn_ms: float
    load_percent: float
    compute_cost: float


@dataclass
class ServerStats:
    accepted: int = 0
    rejected: int = 0
    deadline_hits: int = 0
    deadline_misses: int = 0
    cumulative_compute_cost: float = 0.0
    requests: List[ServedRequest] = field(default_factory=list)

    @property
    def total(self) -> int:
        return self.accepted + self.rejected

    @property
    def completion_rate(self) -> float:
        return self.accepted / self.total if self.total else 0.0

    @property
    def deadline_hit_rate(self) -> float:
        return self.deadline_hits / self.accepted if self.accepted else 0.0


class AdaptiveServer:
    """Threaded server tying together the runtime pool and the control plane."""

    def __init__(
        self,
        runtime_pool: RuntimePool,
        selector: AdaptiveSelector,
        load_monitor: LoadMonitor,
        *,
        max_in_flight: Optional[int] = None,
    ) -> None:
        self._pool = runtime_pool
        self._selector = selector
        self._load_monitor = load_monitor
        self._executor = ThreadPoolExecutor(
            max_workers=max_in_flight if max_in_flight else len(runtime_pool._clients)
        )
        self._stats = ServerStats()
        self._stats_lock = threading.Lock()
        self._arrival_lock = threading.Lock()
        self._arrival_window: List[float] = []
        self._window_seconds: float = 1.0
        self._cost_by_variant: Dict[str, float] = {
            v.name: v.compute_cost_per_request for v in selector.variants
        }

    @property
    def stats(self) -> ServerStats:
        with self._stats_lock:
            return ServerStats(
                accepted=self._stats.accepted,
                rejected=self._stats.rejected,
                deadline_hits=self._stats.deadline_hits,
                deadline_misses=self._stats.deadline_misses,
                cumulative_compute_cost=self._stats.cumulative_compute_cost,
                requests=list(self._stats.requests),
            )

    def submit(
        self,
        request: InferenceRequest,
        deadline_ms: float,
    ) -> Future:
        """Asynchronously schedule a request. Future resolves to a ServedRequest."""
        submitted_at = time.perf_counter()
        self._record_arrival(submitted_at)
        arrival_rate = self._estimate_arrival_rate(submitted_at)
        load_percent = self._load_monitor.smoothed_percent()
        decision = self._selector.select(deadline_ms, arrival_rate, load_percent)
        chosen_variant = decision.variant.name

        if decision.expected_sojourn_ms > deadline_ms:
            served = ServedRequest(
                request_id=request.request_id,
                submitted_at=submitted_at,
                started_at=None,
                completed_at=None,
                deadline_ms=deadline_ms,
                variant=chosen_variant,
                admitted=False,
                rejection_reason="expected sojourn exceeds deadline",
                runtime_latency_ms=0.0,
                wall_latency_ms=0.0,
                expected_sojourn_ms=decision.expected_sojourn_ms,
                load_percent=load_percent,
                compute_cost=0.0,
            )
            self._record_rejection(served)
            future: Future = Future()
            future.set_result(served)
            return future

        request.variant = chosen_variant
        return self._executor.submit(
            self._run,
            request=request,
            deadline_ms=deadline_ms,
            submitted_at=submitted_at,
            expected_sojourn_ms=decision.expected_sojourn_ms,
            load_percent=load_percent,
        )

    def shutdown(self) -> None:
        self._executor.shutdown(wait=True)

    def __enter__(self) -> "AdaptiveServer":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.shutdown()

    def _run(
        self,
        *,
        request: InferenceRequest,
        deadline_ms: float,
        submitted_at: float,
        expected_sojourn_ms: float,
        load_percent: float,
    ) -> ServedRequest:
        started_at = time.perf_counter()
        try:
            response: InferenceResponse = self._pool.infer(request)
        except Exception as exc:
            LOGGER.exception("runtime dispatch failed: %s", exc)
            served = ServedRequest(
                request_id=request.request_id,
                submitted_at=submitted_at,
                started_at=started_at,
                completed_at=time.perf_counter(),
                deadline_ms=deadline_ms,
                variant=request.variant,
                admitted=False,
                rejection_reason=f"runtime error: {exc}",
                runtime_latency_ms=0.0,
                wall_latency_ms=0.0,
                expected_sojourn_ms=expected_sojourn_ms,
                load_percent=load_percent,
                compute_cost=0.0,
            )
            self._record_rejection(served)
            return served

        completed_at = time.perf_counter()
        wall_latency_ms = (completed_at - submitted_at) * 1000.0
        cost = self._cost_by_variant.get(request.variant, 1.0)
        served = ServedRequest(
            request_id=response.request_id,
            submitted_at=submitted_at,
            started_at=started_at,
            completed_at=completed_at,
            deadline_ms=deadline_ms,
            variant=request.variant,
            admitted=True,
            rejection_reason="",
            runtime_latency_ms=response.runtime_latency_ms,
            wall_latency_ms=wall_latency_ms,
            expected_sojourn_ms=expected_sojourn_ms,
            load_percent=load_percent,
            compute_cost=cost,
        )
        self._record_completion(served)
        return served

    def _record_arrival(self, now: float) -> None:
        cutoff = now - self._window_seconds
        with self._arrival_lock:
            self._arrival_window.append(now)
            # Drop arrivals outside the sliding window.
            while self._arrival_window and self._arrival_window[0] < cutoff:
                self._arrival_window.pop(0)

    def _estimate_arrival_rate(self, now: float) -> float:
        with self._arrival_lock:
            if len(self._arrival_window) < 2:
                return 0.0
            duration = now - self._arrival_window[0]
            if duration <= 0.0:
                return 0.0
            return (len(self._arrival_window) - 1) / duration

    def _record_completion(self, served: ServedRequest) -> None:
        with self._stats_lock:
            self._stats.accepted += 1
            self._stats.cumulative_compute_cost += served.compute_cost
            if served.wall_latency_ms <= served.deadline_ms:
                self._stats.deadline_hits += 1
            else:
                self._stats.deadline_misses += 1
            self._stats.requests.append(served)

    def _record_rejection(self, served: ServedRequest) -> None:
        with self._stats_lock:
            self._stats.rejected += 1
            self._stats.requests.append(served)


def poisson_arrivals(
    duration_s: float, rate_rps: float, *, rng: Optional[np.random.Generator] = None
) -> List[float]:
    """Inter-arrival times drawn from an exponential distribution (Poisson process)."""
    rng = rng or np.random.default_rng()
    times: List[float] = []
    t = 0.0
    while True:
        gap = rng.exponential(1.0 / rate_rps)
        t += gap
        if t > duration_s:
            break
        times.append(t)
    return times


def drive_workload(
    server: AdaptiveServer,
    request_factory: Callable[[int], InferenceRequest],
    *,
    arrival_times: List[float],
    deadline_ms: float,
) -> None:
    """Synchronously drive a sequence of arrivals into the server.

    Holds back each request until its arrival time so the load monitor and
    M/M/1 controller observe a realistic Poisson interarrival pattern.
    """
    futures: List[Future] = []
    start = time.perf_counter()
    for i, arrival in enumerate(arrival_times):
        now = time.perf_counter()
        sleep_for = (start + arrival) - now
        if sleep_for > 0.0:
            time.sleep(sleep_for)
        futures.append(server.submit(request_factory(i), deadline_ms))
    for future in futures:
        future.result()
