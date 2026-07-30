"""Admission control derived from M/M/c queueing theory.

The runtime pool serves requests from ``c`` independent workers, so the queue it
forms is M/M/c rather than M/M/1. For arrival rate ``lambda`` (rps) and
per-worker service rate ``mu`` (rps), the offered load is ``a = lambda / mu``
and the per-worker utilisation is ``rho = a / c``. The probability that an
arrival has to wait is given by Erlang's C formula::

    C(c, a) = B(c, a) / (1 - rho * (1 - B(c, a)))

where ``B`` is Erlang's B formula, evaluated with the recursion::

    B(0, a) = 1
    B(k, a) = a * B(k-1, a) / (k + a * B(k-1, a))

Expected waiting time in queue and total sojourn time are then::

    E[W_q] = C(c, a) / (c * mu - lambda)
    E[W]   = E[W_q] + 1 / mu

A request is rejected when ``E[W]`` exceeds the deadline budget, or whenever
``rho >= 1`` since the queue is then unstable.

Setting ``servers=1`` recovers the M/M/1 result ``E[W_q] = rho / (mu * (1 -
rho))`` exactly, so the single-worker case remains available as a special case
of the same controller.
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


def erlang_b(servers: int, offered_load: float) -> float:
    """Erlang's B formula, evaluated by recursion to avoid overflow.

    The closed form divides ``a**c / c!`` by a partial sum of the same terms,
    both of which overflow for even moderate ``c``. The recursion below is
    algebraically equivalent and numerically stable for any ``c``.
    """
    if servers < 0:
        raise ValueError("servers must be non-negative")
    if offered_load < 0.0:
        raise ValueError("offered_load must be non-negative")

    blocking = 1.0
    for k in range(1, servers + 1):
        scaled = offered_load * blocking
        blocking = scaled / (k + scaled)
    return blocking


def erlang_c(servers: int, offered_load: float) -> float:
    """Probability that an arriving request finds every worker busy and queues."""
    if servers <= 0:
        raise ValueError("servers must be positive")

    utilisation = offered_load / servers
    if utilisation >= 1.0:
        return 1.0

    blocking = erlang_b(servers, offered_load)
    denominator = 1.0 - utilisation * (1.0 - blocking)
    if denominator <= 0.0:
        return 1.0
    return blocking / denominator


class MMcAdmissionController:
    """Reject requests whose expected sojourn time would violate the deadline.

    ``servers`` must match the number of workers in the runtime pool that will
    actually serve the request. Passing a per-worker service rate while
    modelling the pool as a single server understates capacity by a factor of
    ``c`` and rejects traffic the pool could comfortably absorb.
    """

    def __init__(self, servers: int = 1, safety_factor: float = 1.0) -> None:
        if servers <= 0:
            raise ValueError("servers must be positive")
        if safety_factor <= 0.0:
            raise ValueError("safety_factor must be positive")
        self._servers = servers
        self._safety_factor = safety_factor

    @property
    def servers(self) -> int:
        return self._servers

    def decide(
        self,
        arrival_rate_rps: float,
        service_rate_rps: float,
        deadline_ms: float,
    ) -> AdmissionDecision:
        """Decide on one arrival.

        ``service_rate_rps`` is the rate of a *single* worker; the controller
        scales it by the pool size internally.
        """
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

        offered_load = arrival_rate_rps / service_rate_rps
        utilisation = offered_load / self._servers
        service_time_ms = 1000.0 / service_rate_rps

        if utilisation >= 1.0:
            return AdmissionDecision(
                admit=False,
                expected_wait_ms=float("inf"),
                expected_sojourn_ms=float("inf"),
                utilisation=utilisation,
                reason="queue unstable (rho >= 1)",
            )

        wait_probability = erlang_c(self._servers, offered_load)
        spare_capacity_rps = self._servers * service_rate_rps - arrival_rate_rps
        expected_wait_ms = (wait_probability / spare_capacity_rps) * 1000.0
        expected_sojourn_ms = expected_wait_ms + service_time_ms
        budget_ms = deadline_ms / self._safety_factor

        if expected_sojourn_ms > budget_ms:
            return AdmissionDecision(
                admit=False,
                expected_wait_ms=expected_wait_ms,
                expected_sojourn_ms=expected_sojourn_ms,
                utilisation=utilisation,
                reason=(
                    f"expected sojourn {expected_sojourn_ms:.1f}ms exceeds budget {budget_ms:.1f}ms"
                ),
            )

        return AdmissionDecision(
            admit=True,
            expected_wait_ms=expected_wait_ms,
            expected_sojourn_ms=expected_sojourn_ms,
            utilisation=utilisation,
            reason="ok",
        )
