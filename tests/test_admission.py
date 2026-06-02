"""Unit tests for the M/M/1 admission controller."""

import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.serving.admission import MM1AdmissionController


class TestMM1Admission(unittest.TestCase):
    def test_low_load_admits(self):
        controller = MM1AdmissionController()
        decision = controller.decide(arrival_rate_rps=20.0, service_rate_rps=100.0, deadline_ms=100.0)
        self.assertTrue(decision.admit)
        # E[W] = rho / (mu * (1 - rho)) seconds = 0.2 / (100 * 0.8) = 2.5 ms
        self.assertAlmostEqual(decision.expected_wait_ms, 2.5, places=3)
        # Sojourn = wait + service = 2.5 + 10 = 12.5 ms
        self.assertAlmostEqual(decision.expected_sojourn_ms, 12.5, places=3)
        self.assertAlmostEqual(decision.utilisation, 0.2, places=3)

    def test_overload_rejects(self):
        controller = MM1AdmissionController()
        decision = controller.decide(arrival_rate_rps=150.0, service_rate_rps=100.0, deadline_ms=100.0)
        self.assertFalse(decision.admit)
        self.assertEqual(decision.reason, "queue unstable (rho >= 1)")
        self.assertTrue(math.isinf(decision.expected_wait_ms))

    def test_tight_deadline_rejects(self):
        controller = MM1AdmissionController()
        decision = controller.decide(arrival_rate_rps=95.0, service_rate_rps=100.0, deadline_ms=20.0)
        self.assertFalse(decision.admit)
        self.assertIn("exceeds budget", decision.reason)
        self.assertTrue(decision.expected_sojourn_ms > 20.0)

    def test_safety_factor_tightens_budget(self):
        relaxed = MM1AdmissionController(safety_factor=1.0)
        strict = MM1AdmissionController(safety_factor=2.0)
        # Half the deadline budget under strict; should reject what relaxed accepts.
        decision_relaxed = relaxed.decide(20.0, 100.0, 20.0)
        decision_strict = strict.decide(20.0, 100.0, 20.0)
        self.assertTrue(decision_relaxed.admit)
        self.assertFalse(decision_strict.admit)

    def test_zero_service_rate(self):
        controller = MM1AdmissionController()
        decision = controller.decide(arrival_rate_rps=10.0, service_rate_rps=0.0, deadline_ms=100.0)
        self.assertFalse(decision.admit)
        self.assertEqual(decision.reason, "zero service rate")


if __name__ == "__main__":
    unittest.main()
