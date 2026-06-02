"""Unit tests for the load-aware variant selector."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.serving.selector import AdaptiveSelector, VariantProfile


FP32 = VariantProfile(name="fp32", service_time_ms=10.0, accuracy=0.91, compute_cost_per_request=1.0)
INT8 = VariantProfile(name="int8", service_time_ms=4.0, accuracy=0.89, compute_cost_per_request=0.42)


class TestAdaptiveSelector(unittest.TestCase):
    def test_picks_high_accuracy_when_lightly_loaded(self):
        selector = AdaptiveSelector([FP32, INT8])
        decision = selector.select(deadline_ms=100.0, arrival_rate_rps=10.0, load_percent=10.0)
        self.assertEqual(decision.variant.name, "fp32")

    def test_falls_back_to_int8_under_pressure(self):
        # High load inflates effective service time so the M/M/1 admission test
        # rejects FP32 and the selector chooses INT8.
        selector = AdaptiveSelector(
            [FP32, INT8], load_knee_percent=20.0, load_slope=0.1
        )
        decision = selector.select(deadline_ms=20.0, arrival_rate_rps=80.0, load_percent=90.0)
        self.assertEqual(decision.variant.name, "int8")

    def test_returns_fastest_when_nothing_fits(self):
        # Unmeetable deadline: selector should still resolve to a variant
        # (the fastest one) rather than raising.
        selector = AdaptiveSelector([FP32, INT8])
        decision = selector.select(deadline_ms=0.5, arrival_rate_rps=200.0, load_percent=99.0)
        self.assertEqual(decision.variant.name, "int8")

    def test_validation(self):
        with self.assertRaises(ValueError):
            AdaptiveSelector([])
        selector = AdaptiveSelector([FP32, INT8])
        with self.assertRaises(ValueError):
            selector.select(deadline_ms=0.0, arrival_rate_rps=1.0, load_percent=10.0)


if __name__ == "__main__":
    unittest.main()
