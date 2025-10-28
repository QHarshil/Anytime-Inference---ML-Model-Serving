"""Workload sensitivity analysis."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.planner.planner import CascadePlanner
from src.utils.io import read_dataframe, write_dataframe
from src.utils.logger import configure_logger
from src.workloads.traces import bursty_workload, steady_workload
from src.workloads.trace_analyzer import summarise_trace


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profiles", default="results/latency_profiles.csv")
    parser.add_argument("--output", default="results/workload_results.csv")
    args = parser.parse_args()

    configure_logger()
    profiles = read_dataframe(Path(args.profiles))
    planner = CascadePlanner(profiles)

    steady_trace = steady_workload(duration_s=60, rate_rps=10)
    bursty_trace = bursty_workload(duration_s=60, low_rps=5, high_rps=20)

    stats = []
    for name, trace in {"steady": steady_trace, "bursty": bursty_trace}.items():
        summary = summarise_trace(trace, service_time_ms=float(profiles["lat_p95_ms"].mean()))
        decision = planner.select(task="text", deadline_ms=100, workload=name)
        record = dict(summary.__dict__)
        record.update({"workload": name, "config_id": decision.config.get("config_id")})
        stats.append(record)

    df = pd.DataFrame(stats)
    write_dataframe(df, Path(args.output))


if __name__ == "__main__":
    main()
