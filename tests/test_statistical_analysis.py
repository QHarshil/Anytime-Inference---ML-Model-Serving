"""Unit tests for the statistical analysis module."""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation.statistical_analysis import (
    check_assumptions,
    cohens_d,
    compare_methods,
    compute_confidence_interval,
    interpret_effect_size,
)


class TestConfidenceIntervals(unittest.TestCase):
    def test_ci_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        mean, ci_lower, ci_upper = compute_confidence_interval(data, confidence=0.95)
        self.assertAlmostEqual(mean, 3.0)
        self.assertLess(ci_lower, mean)
        self.assertGreater(ci_upper, mean)
        self.assertLess(ci_lower, ci_upper)

    def test_ci_coverage(self):
        rng = np.random.default_rng(0)
        true_mean = 100.0
        covered = 0
        trials = 200
        for _ in range(trials):
            data = rng.normal(true_mean, 10.0, size=30)
            _, ci_lower, ci_upper = compute_confidence_interval(data, confidence=0.95)
            if ci_lower <= true_mean <= ci_upper:
                covered += 1
        rate = covered / trials
        self.assertGreater(rate, 0.85)
        self.assertLessEqual(rate, 1.0)


class TestEffectSize(unittest.TestCase):
    def test_cohens_d_zero(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        self.assertAlmostEqual(cohens_d(data, data), 0.0, places=5)

    def test_cohens_d_large(self):
        rng = np.random.default_rng(1)
        a = rng.normal(0.0, 1.0, size=200)
        b = rng.normal(2.0, 1.0, size=200)
        self.assertGreater(abs(cohens_d(a, b)), 1.5)

    def test_interpret_effect_size(self):
        self.assertEqual(interpret_effect_size(0.1), "negligible")
        self.assertEqual(interpret_effect_size(0.3), "small")
        self.assertEqual(interpret_effect_size(0.6), "medium")
        self.assertEqual(interpret_effect_size(0.9), "large")


class TestAssumptionChecking(unittest.TestCase):
    def test_keys_present(self):
        rng = np.random.default_rng(2)
        data = rng.normal(0, 1, size=50)
        assumptions = check_assumptions(data, data)
        for key in ("normality_a", "normality_b", "equal_variances", "normality_ok", "equal_variance_ok"):
            self.assertIn(key, assumptions)


class TestMethodComparison(unittest.TestCase):
    def test_compare_indistinguishable(self):
        # Two independent draws from the same distribution: the planner should
        # report a non-significant difference.
        rng = np.random.default_rng(3)
        a = rng.normal(0.85, 0.05, size=50)
        b = rng.normal(0.85, 0.05, size=50)
        result = compare_methods(a, b, "Method1", "Method2", "accuracy")
        self.assertLess(abs(result.mean_diff), 0.02)
        self.assertGreater(result.p_value_ttest, 0.05)
        self.assertLess(abs(result.cohens_d), 0.5)

    def test_compare_different(self):
        rng = np.random.default_rng(4)
        a = rng.normal(0.80, 0.05, size=50)
        b = rng.normal(0.90, 0.05, size=50)
        result = compare_methods(a, b, "Baseline", "Improved", "accuracy")
        self.assertLess(result.mean_diff, 0.0)
        self.assertLess(result.p_value_ttest, 0.05)
        self.assertGreater(abs(result.cohens_d), 1.0)


if __name__ == "__main__":
    unittest.main()
