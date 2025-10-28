import pandas as pd

from src.planner.failure_handler import FailureHandler


def sample_profiles():
    return pd.DataFrame([
        {"task": "text", "config_id": "cfg1", "lat_p50_ms": 40, "lat_p95_ms": 60, "accuracy": 0.8},
        {"task": "text", "config_id": "cfg2", "lat_p50_ms": 80, "lat_p95_ms": 100, "accuracy": 0.9},
    ])


def sample_results():
    return pd.DataFrame([
        {"deadline_hit_rate": 0.9, "lat_p95_ms": 110, "deadline": 90, "accuracy": 0.85},
        {"deadline_hit_rate": 0.8, "lat_p95_ms": 70, "deadline": 100, "accuracy": 0.83},
    ])


def test_handle_deadline_miss():
    handler = FailureHandler(sample_profiles())
    fallback = handler.handle_deadline_miss("text", 30)
    assert fallback.config["config_id"] == "cfg1"


def test_compute_degradation_metrics():
    handler = FailureHandler(sample_profiles())
    metrics = handler.compute_degradation_metrics(sample_results())
    assert 0.0 <= metrics["deadline_miss_rate"] <= 1.0
