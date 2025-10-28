"""Unit tests for statistical analysis module."""

import unittest
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation.statistical_analysis import (
    compute_confidence_interval,
    cohens_d,
    interpret_effect_size,
    check_assumptions,
    compare_methods
)


class TestConfidenceIntervals(unittest.TestCase):
    """Test confidence interval computation."""
    
    def test_ci_basic(self):
        """Test basic CI computation."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        ci_lower, ci_upper = compute_confidence_interval(data, confidence=0.95)
        
        mean = np.mean(data)
        self.assertLess(ci_lower, mean)
        self.assertGreater(ci_upper, mean)
        self.assertLess(ci_lower, ci_upper)
    
    def test_ci_coverage(self):
        """Test CI coverage property."""
        # Generate many samples
        true_mean = 100.0
        samples = []
        
        for _ in range(100):
            data = np.random.normal(true_mean, 10.0, size=30)
            ci_lower, ci_upper = compute_confidence_interval(data, confidence=0.95)
            samples.append((ci_lower, ci_upper))
        
        # Count how many CIs contain true mean
        coverage = sum(1 for (lower, upper) in samples if lower <= true_mean <= upper)
        coverage_rate = coverage / len(samples)
        
        # Should be close to 0.95 (allow some random variation)
        self.assertGreater(coverage_rate, 0.85)
        self.assertLess(coverage_rate, 1.0)


class TestEffectSize(unittest.TestCase):
    """Test effect size computation."""
    
    def test_cohens_d_zero(self):
        """Test Cohen's d for identical distributions."""
        data1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        data2 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        
        d = cohens_d(data1, data2)
        self.assertAlmostEqual(d, 0.0, places=5)
    
    def test_cohens_d_large(self):
        """Test Cohen's d for well-separated distributions."""
        data1 = np.random.normal(0.0, 1.0, size=100)
        data2 = np.random.normal(2.0, 1.0, size=100)
        
        d = cohens_d(data1, data2)
        self.assertGreater(abs(d), 1.5)  # Should be large
    
    def test_effect_size_interpretation(self):
        """Test effect size interpretation."""
        self.assertEqual(interpret_effect_size(0.1), 'negligible')
        self.assertEqual(interpret_effect_size(0.3), 'small')
        self.assertEqual(interpret_effect_size(0.6), 'medium')
        self.assertEqual(interpret_effect_size(0.9), 'large')


class TestAssumptionChecking(unittest.TestCase):
    """Test statistical assumption checking."""
    
    def test_normality_check(self):
        """Test normality assumption check."""
        # Normal data
        normal_data = np.random.normal(0, 1, size=100)
        
        # Non-normal data (uniform)
        uniform_data = np.random.uniform(0, 1, size=100)
        
        assumptions_normal = check_assumptions(normal_data, normal_data)
        assumptions_uniform = check_assumptions(uniform_data, uniform_data)
        
        # Both should have results
        self.assertIn('normality_ok', assumptions_normal)
        self.assertIn('equal_variance_ok', assumptions_normal)
    
    def test_equal_variance_check(self):
        """Test equal variance assumption check."""
        # Equal variance
        data1 = np.random.normal(0, 1, size=100)
        data2 = np.random.normal(0, 1, size=100)
        
        # Unequal variance
        data3 = np.random.normal(0, 5, size=100)
        
        assumptions_equal = check_assumptions(data1, data2)
        assumptions_unequal = check_assumptions(data1, data3)
        
        self.assertIn('equal_variance_ok', assumptions_equal)
        self.assertIn('equal_variance_ok', assumptions_unequal)


class TestMethodComparison(unittest.TestCase):
    """Test method comparison."""
    
    def test_compare_identical(self):
        """Test comparison of identical methods."""
        data = np.random.normal(0.85, 0.05, size=50)
        
        result = compare_methods(data, data, 'Method1', 'Method2', 'accuracy')
        
        self.assertAlmostEqual(result.mean_diff, 0.0, places=10)
        self.assertGreater(result.p_value_ttest, 0.05)  # Not significant
        self.assertAlmostEqual(result.cohens_d, 0.0, places=5)
    
    def test_compare_different(self):
        """Test comparison of different methods."""
        data1 = np.random.normal(0.80, 0.05, size=50)
        data2 = np.random.normal(0.90, 0.05, size=50)
        
        result = compare_methods(data1, data2, 'Baseline', 'Improved', 'accuracy')
        
        self.assertLess(result.mean_diff, 0.0)  # data2 > data1
        self.assertLess(result.p_value_ttest, 0.05)  # Significant
        self.assertGreater(abs(result.cohens_d), 1.0)  # Large effect


if __name__ == '__main__':
    unittest.main()

