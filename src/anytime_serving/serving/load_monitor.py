"""Background CPU load monitor used by the adaptive selector.

Samples system-wide CPU utilisation at a configurable interval and exposes an
exponentially-smoothed reading. Smoothing damps the high-frequency jitter that
``psutil.cpu_percent`` returns over short intervals while still tracking the
underlying trend a planner cares about.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass

import psutil

from ..utils.logger import get_logger

LOGGER = get_logger("serving.load_monitor")


@dataclass(frozen=True)
class LoadSample:
    timestamp: float
    cpu_percent: float
    smoothed_percent: float


class LoadMonitor:
    """Thread-based EWMA load monitor."""

    def __init__(self, interval_s: float = 0.25, alpha: float = 0.3) -> None:
        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha must be in (0, 1]")
        if interval_s <= 0.0:
            raise ValueError("interval_s must be positive")
        self._interval_s = interval_s
        self._alpha = alpha
        self._lock = threading.Lock()
        self._latest: LoadSample | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        # Prime psutil with a baseline read so the first sample reflects work
        # done since startup rather than returning 0.0.
        psutil.cpu_percent(interval=None)
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="LoadMonitor", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def sample(self) -> LoadSample:
        with self._lock:
            if self._latest is None:
                cpu = psutil.cpu_percent(interval=None)
                self._latest = LoadSample(time.time(), cpu, cpu)
            return self._latest

    def smoothed_percent(self) -> float:
        return self.sample().smoothed_percent

    def _run(self) -> None:
        while not self._stop.is_set():
            cpu = psutil.cpu_percent(interval=self._interval_s)
            with self._lock:
                prev = self._latest.smoothed_percent if self._latest else cpu
                smoothed = self._alpha * cpu + (1.0 - self._alpha) * prev
                self._latest = LoadSample(time.time(), cpu, smoothed)

    def __enter__(self) -> LoadMonitor:
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()
