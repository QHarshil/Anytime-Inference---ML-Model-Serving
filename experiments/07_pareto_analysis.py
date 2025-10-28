"""Perform Pareto analysis on evaluation results."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.theory.pareto import compute_pareto_frontier, compute_hypervolume, dominance_ratio
from src.utils.io import read_dataframe, write_dataframe
from src.utils.logger import configure_logger


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--planner", default="results/planner_results.csv")
    parser.add_argument("--baseline", default="results/baseline_results.csv")
    parser.add_argument("--output", default="results/pareto_analysis.csv")
    args = parser.parse_args()

    configure_logger()
    planner_df = read_dataframe(Path(args.planner))
    baseline_df = read_dataframe(Path(args.baseline))

    frontier = compute_pareto_frontier(planner_df)
    reference = (float(planner_df["lat_p95_ms"].max()), float(planner_df["accuracy"].min()))
    hv = compute_hypervolume(list(zip(frontier["lat_p95_ms"], frontier["accuracy"])), reference)
    dom = dominance_ratio(planner_df, baseline_df)

    df = pd.DataFrame(
        [
            {
                "method": "CascadePlanner",
                "pareto_size": len(frontier),
                "hypervolume": hv,
                "dominance_ratio": dom,
            }
        ]
    )
    write_dataframe(df, Path(args.output))


if __name__ == "__main__":
    main()
