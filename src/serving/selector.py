"""Load-aware variant selector for the adaptive planner.

Frames variant selection as a constrained optimisation: pick the
highest-accuracy variant whose expected sojourn time (queue waiting time plus
service time, derived from the current CPU load and arrival rate) stays under
the deadline. Falls back to the fastest variant when nothing else fits.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

from .admission import MM1AdmissionController


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
    """

    def __init__(
        self,
        variants: Sequence[VariantProfile],
        *,
        load_knee_percent: float = 50.0,
        load_slope: float = 0.02,
        admission_controller: Optional[MM1AdmissionController] = None,
    ) -> None:
        if not variants:
            raise ValueError("at least one variant is required")
        self._variants: List[VariantProfile] = sorted(variants, key=lambda v: -v.accuracy)
        self._fastest: VariantProfile = min(variants, key=lambda v: v.service_time_ms)
        self._load_knee = load_knee_percent
        self._load_slope = load_slope
        self._admission = admission_controller or MM1AdmissionController()

    @property
    def variants(self) -> List[VariantProfile]:
        return list(self._variants)

    def select(
        self,
        deadline_ms: float,
        arrival_rate_rps: float,
        load_percent: float,
    ) -> SelectionDecision:
        if deadline_ms <= 0.0:
            raise ValueError("deadline_ms must be positive")

        pressure = max(0.0, load_percent - self._load_knee) * self._load_slope
        best: Optional[SelectionDecision] = None
        for variant in self._variants:
            effective_service_ms = variant.service_time_ms * (1.0 + pressure)
            effective_rate = 1000.0 / effective_service_ms
            decision = self._admission.decide(arrival_rate_rps, effective_rate, deadline_ms)
            if decision.admit:
                return SelectionDecision(
                    variant=variant,
                    expected_sojourn_ms=decision.expected_sojourn_ms,
                    utilisation=decision.utilisation,
                    load_percent=load_percent,
                )
            if best is None:
                best = SelectionDecision(
                    variant=variant,
                    expected_sojourn_ms=decision.expected_sojourn_ms,
                    utilisation=decision.utilisation,
                    load_percent=load_percent,
                )

        fastest_service_ms = self._fastest.service_time_ms * (1.0 + pressure)
        fastest_rate = 1000.0 / fastest_service_ms
        decision = self._admission.decide(arrival_rate_rps, fastest_rate, deadline_ms)
        return SelectionDecision(
            variant=self._fastest,
            expected_sojourn_ms=decision.expected_sojourn_ms,
            utilisation=decision.utilisation,
            load_percent=load_percent,
        )
