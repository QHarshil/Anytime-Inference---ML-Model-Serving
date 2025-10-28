from src.profiler.profiler_utils import measure_latencies, compute_accuracy


def test_measure_latencies_runs():
    def noop():
        return None

    p50, p95, throughput = measure_latencies(noop, iterations=3)
    assert p50 >= 0
    assert p95 >= 0
    assert throughput >= 0


def test_compute_accuracy():
    acc = compute_accuracy([1, 0, 1], [1, 1, 1])
    assert 0 <= acc <= 1
