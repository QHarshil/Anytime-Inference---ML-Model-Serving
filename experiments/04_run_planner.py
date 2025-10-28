"""Evaluate the cascade planner using profiling results."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml

from src.planner.planner import CascadePlanner
from src.utils.io import read_dataframe, write_dataframe
from src.utils.logger import configure_logger


def load_profiles(latency_path: Path, accuracy_path: Path) -> pd.DataFrame:
    latency_df = read_dataframe(latency_path)
    accuracy_df = read_dataframe(accuracy_path) if accuracy_path.exists() else pd.DataFrame()
    if "config_id" not in latency_df.columns:
        latency_df = latency_df.reset_index().rename(columns={"index": "config_id"})
    if not accuracy_df.empty:
        latency_df = latency_df.merge(
            accuracy_df[["model", "variant", "device", "accuracy"]],
            on=["model", "variant", "device"],
            how="left",
        )
    if "accuracy" not in latency_df.columns:
        latency_df["accuracy"] = 0.0
    return latency_df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--latency", default="results/latency_profiles.csv")
    parser.add_argument("--accuracy", default="results/accuracy_profiles.csv")
    parser.add_argument("--deadlines", default="configs/deadlines.yaml")
    parser.add_argument("--output", default="results/planner_results.csv")
    parser.add_argument("--workload", default="steady", choices=["steady", "bursty"])
    args = parser.parse_args()

    configure_logger()
    profiles = load_profiles(Path(args.latency), Path(args.accuracy))
    planner = CascadePlanner(profiles)
    deadlines_cfg = yaml.safe_load(Path(args.deadlines).read_text())

    records = []
    for task, deadlines in deadlines_cfg.items():
        for deadline in deadlines:
            decision = planner.select(task, deadline, workload=args.workload)
            record = dict(decision.config)
            record.update({"task": task, "deadline": deadline, "method": "CascadePlanner"})
            if decision.fallback is not None:
                record["fallback_reason"] = decision.fallback.reason
            records.append(record)

    df = pd.DataFrame(records)
    write_dataframe(df, Path(args.output))


if __name__ == "__main__":
    main()
