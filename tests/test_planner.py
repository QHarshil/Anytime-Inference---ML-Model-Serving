import pandas as pd

from src.planner.planner import CascadePlanner


def sample_profiles():
    return pd.DataFrame([
        {"task": "text", "config_id": "cfg1", "lat_p95_ms": 50, "lat_p50_ms": 40, "accuracy": 0.8, "batch_size": 1},
        {"task": "text", "config_id": "cfg2", "lat_p95_ms": 120, "lat_p50_ms": 90, "accuracy": 0.9, "batch_size": 4},
    ])


def test_planner_selects_feasible():
    planner = CascadePlanner(sample_profiles())
    decision = planner.select("text", deadline_ms=100)
    assert decision.config["config_id"] == "cfg1"


def test_planner_fallback_when_no_feasible():
    planner = CascadePlanner(sample_profiles())
    decision = planner.select("text", deadline_ms=10)
    assert decision.fallback is not None
