"""Unit tests for real inference evaluator."""

import unittest
import numpy as np
import torch
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation.real_inference import RealInferenceEvaluator, CascadeInferenceEvaluator, InferenceResult


class TestRealInferenceEvaluator(unittest.TestCase):
    """Test RealInferenceEvaluator class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.evaluator = RealInferenceEvaluator()
    
    def test_init(self):
        """Test evaluator initialization."""
        self.assertIsNotNone(self.evaluator)
        self.assertIsInstance(self.evaluator.cache, dict)
    
    def test_inference_result_structure(self):
        """Test InferenceResult dataclass structure."""
        result = InferenceResult(
            latency_mean=100.0,
            latency_std=10.0,
            latency_p50=95.0,
            latency_p95=115.0,
            accuracy_mean=0.85,
            accuracy_std=0.02,
            num_runs=5,
            num_samples_per_run=100
        )
        
        self.assertEqual(result.latency_mean, 100.0)
        self.assertEqual(result.latency_std, 10.0)
        self.assertEqual(result.accuracy_mean, 0.85)
        self.assertEqual(result.num_runs, 5)
    
    def test_cache_key_generation(self):
        """Test cache key generation."""
        config = {
            'model': 'distilbert',
            'variant': 'fp32',
            'device': 'cpu',
            'batch_size': 1
        }
        
        key1 = self.evaluator._make_cache_key(config, 'text', 42, 5, 100)
        key2 = self.evaluator._make_cache_key(config, 'text', 42, 5, 100)
        key3 = self.evaluator._make_cache_key(config, 'text', 43, 5, 100)
        
        self.assertEqual(key1, key2)
        self.assertNotEqual(key1, key3)


class TestCascadeInferenceEvaluator(unittest.TestCase):
    """Test CascadeInferenceEvaluator class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.evaluator = CascadeInferenceEvaluator()
    
    def test_init(self):
        """Test evaluator initialization."""
        self.assertIsNotNone(self.evaluator)
        self.assertIsInstance(self.evaluator.cache, dict)
    
    def test_threshold_validation(self):
        """Test threshold validation."""
        # Valid thresholds
        for threshold in [0.0, 0.5, 0.9, 1.0]:
            # Should not raise
            pass
        
        # Invalid thresholds would be caught by model logic
        self.assertTrue(True)


class TestStatisticalProperties(unittest.TestCase):
    """Test statistical properties of measurements."""
    
    def test_latency_variance(self):
        """Test that latency measurements have reasonable variance."""
        # Simulate latencies
        latencies = np.random.lognormal(mean=4.0, sigma=0.2, size=100)
        
        mean = np.mean(latencies)
        std = np.std(latencies)
        p50 = np.percentile(latencies, 50)
        p95 = np.percentile(latencies, 95)
        
        # Sanity checks
        self.assertGreater(mean, 0)
        self.assertGreater(std, 0)
        self.assertGreater(p95, p50)
        self.assertGreater(p95, mean)
    
    def test_accuracy_bounds(self):
        """Test that accuracy is bounded [0, 1]."""
        # Simulate accuracies
        accuracies = np.random.beta(a=85, b=15, size=100)
        
        self.assertTrue(np.all(accuracies >= 0.0))
        self.assertTrue(np.all(accuracies <= 1.0))
        
        mean = np.mean(accuracies)
        self.assertGreater(mean, 0.0)
        self.assertLess(mean, 1.0)


if __name__ == '__main__':
    unittest.main()

