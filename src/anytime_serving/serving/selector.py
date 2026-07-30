"""Load-aware variant selector for the adaptive planner.

Frames variant selection as a constrained optimisation: pick the
highest-accuracy variant whose expected sojourn time (queue waiting time plus
service time, derived from the current CPU load and arrival rate) stays under
the deadline. Falls back to the fastest variant when nothing else fits.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from .admission import MMcAdmissionController


@dataclass(frozen=True)
class VariantProfile:
    """Profiled cost/quality for one model variant on the target hardware."""

    name: str
    service_time_ms: float
    accuracy: float
    compute_cost_per_request: float

    def service_rate_rps(self) -> float:
        return 1000.0 / self.service_time_ms if self.service_time_ms > 0 else 0.0


@dataclass(frozen=True)
class SelectionDecision:
    variant: VariantProfile
    expected_sojourn_ms: float
    utilisation: float
    load_percent: float


class AdaptiveSelector:
    """Pick a variant given current load and arrival rate.

    Service times are scaled by a CPU-pressure factor (linear in load above a
    knee). Higher load inflates expected service time, which pushes the
    selector toward the cheaper INT8 variant. The fastest variant is always the
    fallback when no variant satisfies the deadline.

    ``servers`` must equal the number of workers in the runtime pool this
    selector feeds. It is taken here rather than on the admission controller so
    that a selector can never be paired with a controller that models a
    different pool size; ``AdaptiveServer`` asserts the value matches its pool.
    """

    def __init__(
        self,
        variants: Sequence[VariantProfile],
        *,
        servers: int,
        load_knee_percent: float = 50.0,
        load_slope: float = 0.02,
        safety_factor: float = 1.0,
    ) -> None:
        if not variants:
            raise ValueError("at least one variant is required")
        self._variants: list[VariantProfile] = sorted(variants, key=lambda v: -v.accuracy)
        self._fastest: VariantProfile = min(variants, key=lambda v: v.service_time_ms)
        self._load_knee = load_knee_percent
        self._load_slope = load_slope
        self._admission = MMcAdmissionController(servers=servers, safety_factor=safety_factor)

    @property
    def variants(self) -> list[VariantProfile]:
        return list(self._variants)

    @property
    def servers(self) -> int:
        return self._admission.servers

    def _sojourn_estimate_ms(
        self,
        variant: VariantProfile,
        *,
        arrival_rate_rps: float,
        queue_depth: int,
        pressure: float,
        deadline_ms: float,
    ) -> tuple[float, float]:
        """Estimate sojourn time for *variant*, returning (sojourn_ms, utilisation).

        Two estimators are combined and the more pessimistic one wins:

        - The M/M/c stationary bound, which captures the average behaviour implied
          by the observed arrival rate.
        - The work already queued. ``queue_depth`` requests ahead of this one,
          spread over ``servers`` workers, take
          ``ceil(queue_depth / servers) * service_time`` to clear.

        The stationary bound alone is not enough: it is computed from a
        one-second arrival-rate window and says nothing about the backlog that
        exists right now. Admitting on it while the dispatch queue is unbounded
        lets the backlog grow without limit, and every admitted request then
        misses its deadline even though the predicted sojourn looked fine.
        """
        effective_service_ms = variant.service_time_ms * (1.0 + pressure)
        effective_rate = 1000.0 / effective_service_ms
        decision = self._admission.decide(arrival_rate_rps, effective_rate, deadline_ms)

        servers = self._admission.servers
        rounds_ahead = math.ceil(queue_depth / servers) if queue_depth > 0 else 0
        backlog_ms = rounds_ahead * effective_service_ms
        queue_sojourn_ms = backlog_ms + effective_service_ms

        return max(decision.expected_sojourn_ms, queue_sojourn_ms), decision.utilisation

    def select(
        self,
        deadline_ms: float,
        arrival_rate_rps: float,
        load_percent: float,
        queue_depth: int = 0,
    ) -> SelectionDecision:
        """Pick a variant for one arrival.

        ``queue_depth`` is the number of requests already admitted but not yet
        completed. Passing it lets admission respond to the backlog that exists
        now rather than to a stationary average.
        """
        if deadline_ms <= 0.0:
            raise ValueError("deadline_ms must be positive")
        if queue_depth < 0:
            raise ValueError("queue_depth must be non-negative")

        pressure = max(0.0, load_percent - self._load_knee) * self._load_slope
        best: SelectionDecision | None = None
        for variant in self._variants:
            sojourn_ms, utilisation = self._sojourn_estimate_ms(
                variant,
                arrival_rate_rps=arrival_rate_rps,
                queue_depth=queue_depth,
                pressure=pressure,
                deadline_ms=deadline_ms,
            )
            candidate = SelectionDecision(
                variant=variant,
                expected_sojourn_ms=sojourn_ms,
                utilisation=utilisation,
                load_percent=load_percent,
            )
            if sojourn_ms <= deadline_ms:
                return candidate
            if best is None:
                best = candidate

        sojourn_ms, utilisation = self._sojourn_estimate_ms(
            self._fastest,
            arrival_rate_rps=arrival_rate_rps,
            queue_depth=queue_depth,
            pressure=pressure,
            deadline_ms=deadline_ms,
        )
        return SelectionDecision(
            variant=self._fastest,
            expected_sojourn_ms=sojourn_ms,
            utilisation=utilisation,
            load_percent=load_percent,
        )
