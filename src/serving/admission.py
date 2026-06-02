"""Admission control derived from M/M/1 queueing theory.

For an M/M/1 queue with arrival rate ``lambda`` (rps) and service rate ``mu``
(rps), utilisation ``rho = lambda / mu``. The expected waiting time in queue
(Pollaczek-Khinchine for M/M/1) is::

    E[W_q] = rho / (mu * (1 - rho)),    rho < 1

Total sojourn time for the request is ``E[W_q] + 1/mu``. If that exceeds the
SLO, the request is rejected. The controller also rejects whenever ``rho >= 1``
since the queue is unstable.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AdmissionDecision:
    admit: bool
    expected_wait_ms: float
    expected_sojourn_ms: float
    utilisation: float
    reason: str


class MM1AdmissionController:
    """Reject requests whose expected sojourn time would violate the deadline."""

    def __init__(self, safety_factor: float = 1.0) -> None:
        if safety_factor <= 0.0:
            raise ValueError("safety_factor must be positive")
        self._safety_factor = safety_factor

    def decide(
        self,
        arrival_rate_rps: float,
        service_rate_rps: float,
        deadline_ms: float,
    ) -> AdmissionDecision:
        if service_rate_rps <= 0.0:
            return AdmissionDecision(
                admit=False,
                expected_wait_ms=float("inf"),
                expected_sojourn_ms=float("inf"),
                utilisation=float("inf"),
                reason="zero service rate",
            )
        if arrival_rate_rps < 0.0:
            raise ValueError("arrival_rate_rps must be non-negative")

        rho = arrival_rate_rps / service_rate_rps
        service_time_ms = 1000.0 / service_rate_rps

        if rho >= 1.0:
            return AdmissionDecision(
                admit=False,
                expected_wait_ms=float("inf"),
                expected_sojourn_ms=float("inf"),
                utilisation=rho,
                reason="queue unstable (rho >= 1)",
            )

        expected_wait_ms = (rho / (service_rate_rps * (1.0 - rho))) * 1000.0
        expected_sojourn_ms = expected_wait_ms + service_time_ms
        budget_ms = deadline_ms / self._safety_factor

        if expected_sojourn_ms > budget_ms:
            return AdmissionDecision(
                admit=False,
                expected_wait_ms=expected_wait_ms,
                expected_sojourn_ms=expected_sojourn_ms,
                utilisation=rho,
                reason=f"expected sojourn {expected_sojourn_ms:.1f}ms exceeds budget {budget_ms:.1f}ms",
            )

        return AdmissionDecision(
            admit=True,
            expected_wait_ms=expected_wait_ms,
            expected_sojourn_ms=expected_sojourn_ms,
            utilisation=rho,
            reason="ok",
        )
