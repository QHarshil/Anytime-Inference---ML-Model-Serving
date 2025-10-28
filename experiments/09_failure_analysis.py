"""Analyse planner behaviour under failure scenarios."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.planner.failure_handler import FailureHandler
from src.utils.io import read_dataframe, write_dataframe
from src.utils.logger import configure_logger


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profiles", default="results/latency_profiles.csv")
    parser.add_argument("--results", default="results/planner_results.csv")
    parser.add_argument("--output", default="results/failure_analysis.csv")
    args = parser.parse_args()

    configure_logger()
    profiles = read_dataframe(Path(args.profiles))
    results = read_dataframe(Path(args.results)) if Path(args.results).exists() else pd.DataFrame()

    handler = FailureHandler(profiles)
    metrics = handler.compute_degradation_metrics(results)
    write_dataframe(pd.DataFrame([metrics]), Path(args.output))


if __name__ == "__main__":
    main()
