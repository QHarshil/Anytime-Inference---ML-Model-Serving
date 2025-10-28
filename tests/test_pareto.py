import pandas as pd

from src.theory.pareto import compute_pareto_frontier, compute_hypervolume, dominance_ratio


def test_compute_pareto_frontier_basic():
    df = pd.DataFrame({
        "lat_p95_ms": [50, 100, 200],
        "accuracy": [0.85, 0.88, 0.91],
    })
    frontier = compute_pareto_frontier(df)
    assert len(frontier) == 3


def test_compute_hypervolume():
    points = [(50, 0.85), (100, 0.88), (200, 0.91)]
    hv = compute_hypervolume(points, reference_point=(300, 0.80))
    assert hv > 0


def test_dominance_ratio():
    method_df = pd.DataFrame({"lat_p95_ms": [50], "accuracy": [0.9]})
    baseline_df = pd.DataFrame({"lat_p95_ms": [100, 120], "accuracy": [0.85, 0.86]})
    ratio = dominance_ratio(method_df, baseline_df)
    assert ratio == 1.0
