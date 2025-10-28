from src.theory.deadline_scheduling import compute_utilization, is_schedulable_rm


def test_compute_utilization():
    configs = [
        {"lat_p95_ms": 50, "request_rate": 5},
        {"lat_p95_ms": 100, "request_rate": 2},
    ]
    utilisation = compute_utilization(configs)
    assert utilisation > 0


def test_is_schedulable_rm():
    assert is_schedulable_rm(0.6, 2)
    assert not is_schedulable_rm(2.0, 1)
