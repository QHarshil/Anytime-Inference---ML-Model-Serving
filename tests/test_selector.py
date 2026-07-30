"""Unit tests for the load-aware variant selector."""

import unittest

from anytime_serving.serving.selector import AdaptiveSelector, VariantProfile

FP32 = VariantProfile(
    name="fp32", service_time_ms=10.0, accuracy=0.91, compute_cost_per_request=1.0
)
INT8 = VariantProfile(
    name="int8", service_time_ms=4.0, accuracy=0.89, compute_cost_per_request=0.42
)


class TestAdaptiveSelector(unittest.TestCase):
    def test_picks_high_accuracy_when_lightly_loaded(self):
        selector = AdaptiveSelector([FP32, INT8], servers=4)
        decision = selector.select(deadline_ms=100.0, arrival_rate_rps=10.0, load_percent=10.0)
        self.assertEqual(decision.variant.name, "fp32")

    def test_falls_back_to_int8_when_only_int8_admits(self):
        """High load inflates effective service time past the deadline for FP32.

        Both variants are checked against the same M/M/c bound, so this asserts
        the fallback happens because INT8 admits and FP32 does not, rather than
        because neither fits and the fastest variant is returned by default.
        """
        selector = AdaptiveSelector([FP32, INT8], servers=2, load_knee_percent=20.0, load_slope=0.1)
        decision = selector.select(deadline_ms=50.0, arrival_rate_rps=20.0, load_percent=90.0)
        self.assertEqual(decision.variant.name, "int8")
        # Admitted, not the unmeetable-deadline fallback.
        self.assertLess(decision.expected_sojourn_ms, 50.0)

    def test_returns_fastest_when_nothing_fits(self):
        # Unmeetable deadline: selector should still resolve to a variant
        # (the fastest one) rather than raising.
        selector = AdaptiveSelector([FP32, INT8], servers=4)
        decision = selector.select(deadline_ms=0.5, arrival_rate_rps=200.0, load_percent=99.0)
        self.assertEqual(decision.variant.name, "int8")
        self.assertGreater(decision.expected_sojourn_ms, 0.5)

    def test_larger_pool_keeps_the_accurate_variant(self):
        """Pool size feeds the admission bound, so capacity buys accuracy.

        The same offered load that forces a two-worker pool onto INT8 is served
        by FP32 once enough workers are present.
        """
        # 190 rps is 0.95 utilisation on two FP32 workers (100 rps each) but
        # only 0.24 on eight, so the queue term dominates in one case and
        # vanishes in the other.
        arrival_rate = 190.0
        small = AdaptiveSelector([FP32, INT8], servers=2).select(
            deadline_ms=40.0, arrival_rate_rps=arrival_rate, load_percent=0.0
        )
        large = AdaptiveSelector([FP32, INT8], servers=8).select(
            deadline_ms=40.0, arrival_rate_rps=arrival_rate, load_percent=0.0
        )
        self.assertEqual(small.variant.name, "int8")
        self.assertEqual(large.variant.name, "fp32")
        self.assertLess(large.expected_sojourn_ms, small.expected_sojourn_ms + 40.0)

    def test_utilisation_accounts_for_pool_size(self):
        selector = AdaptiveSelector([FP32], servers=4)
        decision = selector.select(deadline_ms=1e6, arrival_rate_rps=200.0, load_percent=0.0)
        # FP32 at 10 ms serves 100 rps per worker; 200 rps over 4 workers is 0.5.
        self.assertAlmostEqual(decision.utilisation, 0.5, places=9)

    def test_exposes_server_count(self):
        self.assertEqual(AdaptiveSelector([FP32], servers=3).servers, 3)

    def test_validation(self):
        with self.assertRaises(ValueError):
            AdaptiveSelector([], servers=1)
        with self.assertRaises(ValueError):
            AdaptiveSelector([FP32], servers=0)
        selector = AdaptiveSelector([FP32, INT8], servers=4)
        with self.assertRaises(ValueError):
            selector.select(deadline_ms=0.0, arrival_rate_rps=1.0, load_percent=10.0)


if __name__ == "__main__":
    unittest.main()
