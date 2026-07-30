"""Unit tests for the M/M/c admission controller."""

import math
import unittest

from anytime_serving.serving.admission import MMcAdmissionController, erlang_b, erlang_c


def erlang_c_closed_form(servers: int, offered_load: float) -> float:
    """Erlang C evaluated directly from factorials.

    Independent of the recursion used in the implementation, so a disagreement
    indicates a genuine error rather than a shared mistake. Overflows for large
    ``servers``, which is exactly why the implementation does not use it.
    """
    utilisation = offered_load / servers
    partial = sum(offered_load**k / math.factorial(k) for k in range(servers))
    tail = offered_load**servers / math.factorial(servers) / (1.0 - utilisation)
    return tail / (partial + tail)


class TestErlangFormulas(unittest.TestCase):
    def test_erlang_c_matches_closed_form(self):
        for servers in (1, 2, 3, 4, 8, 16, 32, 64):
            for offered_load in (0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0):
                if offered_load / servers >= 1.0:
                    continue
                with self.subTest(servers=servers, offered_load=offered_load):
                    self.assertAlmostEqual(
                        erlang_c(servers, offered_load),
                        erlang_c_closed_form(servers, offered_load),
                        places=12,
                    )

    def test_erlang_b_boundary_values(self):
        # No servers means every arrival is blocked; zero load means none is.
        self.assertEqual(erlang_b(0, 5.0), 1.0)
        self.assertEqual(erlang_b(4, 0.0), 0.0)

    def test_erlang_c_saturated_queue(self):
        # At or beyond capacity every arrival waits.
        self.assertEqual(erlang_c(4, 4.0), 1.0)
        self.assertEqual(erlang_c(4, 9.0), 1.0)

    def test_erlang_formulas_stable_at_large_server_counts(self):
        # The factorial form overflows past c ~= 170; the recursion must not.
        self.assertTrue(0.0 < erlang_b(500, 400.0) < 1.0)
        self.assertTrue(0.0 < erlang_c(1000, 900.0) < 1.0)

    def test_erlang_rejects_invalid_arguments(self):
        with self.assertRaises(ValueError):
            erlang_b(-1, 1.0)
        with self.assertRaises(ValueError):
            erlang_b(1, -1.0)
        with self.assertRaises(ValueError):
            erlang_c(0, 1.0)


class TestSingleServerEquivalence(unittest.TestCase):
    """With one worker the controller must reproduce the M/M/1 result exactly."""

    def test_low_load_admits(self):
        controller = MMcAdmissionController(servers=1)
        decision = controller.decide(
            arrival_rate_rps=20.0, service_rate_rps=100.0, deadline_ms=100.0
        )
        self.assertTrue(decision.admit)
        # E[W_q] = rho / (mu * (1 - rho)) seconds = 0.2 / (100 * 0.8) = 2.5 ms
        self.assertAlmostEqual(decision.expected_wait_ms, 2.5, places=9)
        # Sojourn = wait + service = 2.5 + 10 = 12.5 ms
        self.assertAlmostEqual(decision.expected_sojourn_ms, 12.5, places=9)
        self.assertAlmostEqual(decision.utilisation, 0.2, places=9)

    def test_matches_mm1_closed_form_across_utilisations(self):
        controller = MMcAdmissionController(servers=1)
        service_rate = 100.0
        for arrival_rate in (1.0, 20.0, 50.0, 80.0, 95.0, 99.0):
            with self.subTest(arrival_rate=arrival_rate):
                utilisation = arrival_rate / service_rate
                expected_ms = (utilisation / (service_rate * (1.0 - utilisation))) * 1000.0
                decision = controller.decide(arrival_rate, service_rate, deadline_ms=1e9)
                self.assertAlmostEqual(decision.expected_wait_ms, expected_ms, places=9)

    def test_overload_rejects(self):
        controller = MMcAdmissionController(servers=1)
        decision = controller.decide(
            arrival_rate_rps=150.0, service_rate_rps=100.0, deadline_ms=100.0
        )
        self.assertFalse(decision.admit)
        self.assertEqual(decision.reason, "queue unstable (rho >= 1)")
        self.assertTrue(math.isinf(decision.expected_wait_ms))

    def test_tight_deadline_rejects(self):
        controller = MMcAdmissionController(servers=1)
        decision = controller.decide(
            arrival_rate_rps=95.0, service_rate_rps=100.0, deadline_ms=20.0
        )
        self.assertFalse(decision.admit)
        self.assertIn("exceeds budget", decision.reason)
        self.assertGreater(decision.expected_sojourn_ms, 20.0)


class TestMultiServerCapacity(unittest.TestCase):
    def test_pool_capacity_scales_with_server_count(self):
        """A four-worker pool absorbs load a single worker cannot.

        Guards the defect this controller replaced: passing a per-worker service
        rate while modelling one server understates capacity by a factor of c
        and rejects traffic the pool serves comfortably.
        """
        service_rate = 1000.0 / 12.0  # 12 ms per request per worker
        arrival_rate = 90.0  # above one worker's 83.3 rps, far below four workers'
        deadline_ms = 45.0

        single = MMcAdmissionController(servers=1).decide(arrival_rate, service_rate, deadline_ms)
        pool = MMcAdmissionController(servers=4).decide(arrival_rate, service_rate, deadline_ms)

        self.assertFalse(single.admit)
        self.assertGreaterEqual(single.utilisation, 1.0)

        self.assertTrue(pool.admit)
        self.assertAlmostEqual(pool.utilisation, arrival_rate / (4 * service_rate), places=9)
        self.assertLess(pool.expected_sojourn_ms, deadline_ms)

    def test_utilisation_is_per_worker(self):
        controller = MMcAdmissionController(servers=4)
        decision = controller.decide(
            arrival_rate_rps=200.0, service_rate_rps=100.0, deadline_ms=1e9
        )
        # offered load 2.0 spread over 4 workers
        self.assertAlmostEqual(decision.utilisation, 0.5, places=9)

    def test_wait_falls_as_workers_are_added(self):
        service_rate = 100.0
        arrival_rate = 90.0
        waits = [
            MMcAdmissionController(servers=c)
            .decide(arrival_rate, service_rate, deadline_ms=1e9)
            .expected_wait_ms
            for c in (1, 2, 4, 8)
        ]
        self.assertEqual(waits, sorted(waits, reverse=True))
        self.assertLess(waits[-1], 1e-3)

    def test_saturated_pool_rejects(self):
        controller = MMcAdmissionController(servers=4)
        decision = controller.decide(
            arrival_rate_rps=400.0, service_rate_rps=100.0, deadline_ms=100.0
        )
        self.assertFalse(decision.admit)
        self.assertEqual(decision.reason, "queue unstable (rho >= 1)")


class TestControllerContract(unittest.TestCase):
    def test_safety_factor_tightens_budget(self):
        relaxed = MMcAdmissionController(servers=1, safety_factor=1.0)
        strict = MMcAdmissionController(servers=1, safety_factor=2.0)
        # Half the deadline budget under strict; should reject what relaxed accepts.
        self.assertTrue(relaxed.decide(20.0, 100.0, 20.0).admit)
        self.assertFalse(strict.decide(20.0, 100.0, 20.0).admit)

    def test_zero_service_rate(self):
        controller = MMcAdmissionController(servers=4)
        decision = controller.decide(arrival_rate_rps=10.0, service_rate_rps=0.0, deadline_ms=100.0)
        self.assertFalse(decision.admit)
        self.assertEqual(decision.reason, "zero service rate")

    def test_idle_pool_waits_only_for_service(self):
        controller = MMcAdmissionController(servers=4)
        decision = controller.decide(
            arrival_rate_rps=0.0, service_rate_rps=100.0, deadline_ms=100.0
        )
        self.assertTrue(decision.admit)
        self.assertAlmostEqual(decision.expected_wait_ms, 0.0, places=9)
        self.assertAlmostEqual(decision.expected_sojourn_ms, 10.0, places=9)

    def test_rejects_invalid_construction(self):
        with self.assertRaises(ValueError):
            MMcAdmissionController(servers=0)
        with self.assertRaises(ValueError):
            MMcAdmissionController(servers=-1)
        with self.assertRaises(ValueError):
            MMcAdmissionController(servers=1, safety_factor=0.0)

    def test_rejects_negative_arrival_rate(self):
        controller = MMcAdmissionController(servers=1)
        with self.assertRaises(ValueError):
            controller.decide(-1.0, 100.0, 100.0)


if __name__ == "__main__":
    unittest.main()
