"""Smoke tests for the real-inference evaluator dataclass and cache key."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation.real_inference import (
    CascadeInferenceEvaluator,
    InferenceResult,
    RealInferenceEvaluator,
)


class TestInferenceResult(unittest.TestCase):
    def test_dataclass_fields(self):
        result = InferenceResult(
            config_id="text_distilbert_fp32_cpu_b1_seed42",
            task="text",
            model="distilbert",
            variant="fp32",
            device="cpu",
            batch_size=1,
            latencies_ms=[100.0, 102.0],
            latency_samples_ms=[98.0, 101.0, 102.0],
            accuracies=[0.84, 0.86],
            latency_mean=100.0,
            latency_std=10.0,
            latency_p50=95.0,
            latency_p95=115.0,
            accuracy_mean=0.85,
            accuracy_std=0.02,
            num_runs=5,
            num_samples_per_run=100,
        )
        self.assertEqual(result.latency_mean, 100.0)
        self.assertEqual(result.accuracy_mean, 0.85)
        self.assertEqual(result.num_runs, 5)


class TestEvaluatorCacheKey(unittest.TestCase):
    def test_cache_key_stability(self):
        evaluator = RealInferenceEvaluator()
        config = {"model": "distilbert", "variant": "fp32", "device": "cpu", "batch_size": 1}
        key1 = evaluator._get_cache_key(config, "text", 42)
        key2 = evaluator._get_cache_key(config, "text", 42)
        key3 = evaluator._get_cache_key(config, "text", 43)
        self.assertEqual(key1, key2)
        self.assertNotEqual(key1, key3)


class TestCascadeEvaluatorInit(unittest.TestCase):
    def test_cache_is_dict(self):
        evaluator = CascadeInferenceEvaluator()
        self.assertIsInstance(evaluator.cache, dict)


if __name__ == "__main__":
    unittest.main()
