"""Evaluate baseline planners using profiling results."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml

from src.planner.baselines import StaticBaseline, ThroughputAutotuner
from src.utils.io import read_dataframe, write_dataframe
from src.utils.logger import configure_logger


def load_profiles(latency_path: Path, accuracy_path: Path) -> pd.DataFrame:
    latency_df = read_dataframe(latency_path) if latency_path.exists() else pd.DataFrame()
    accuracy_df = read_dataframe(accuracy_path) if accuracy_path.exists() else pd.DataFrame()
    if latency_df.empty:
        raise FileNotFoundError("Latency profiles are required. Run experiments/01_profile_latency.py first.")
    if "config_id" not in latency_df.columns:
        latency_df = latency_df.reset_index().rename(columns={"index": "config_id"})
    df = latency_df
    if not accuracy_df.empty:
        df = df.merge(accuracy_df[["model", "variant", "device", "accuracy"]], on=["model", "variant", "device"], how="left")
    if "accuracy" not in df.columns:
        df["accuracy"] = 0.0
    return df


def evaluate_baseline(baseline, task: str, deadlines) -> list[dict]:
    records = []
    for deadline in deadlines:
        config = baseline.select(task) if isinstance(baseline, StaticBaseline) else baseline.select(task, deadline)
        record = dict(config)
        record.update({"method": baseline.__class__.__name__, "deadline": deadline, "task": task})
        records.append(record)
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--latency", default="results/latency_profiles.csv")
    parser.add_argument("--accuracy", default="results/accuracy_profiles.csv")
    parser.add_argument("--deadlines", default="configs/deadlines.yaml")
    parser.add_argument("--output", default="results/baseline_results.csv")
    args = parser.parse_args()

    configure_logger()
    profiles = load_profiles(Path(args.latency), Path(args.accuracy))
    deadlines_cfg = yaml.safe_load(Path(args.deadlines).read_text())

    static_fast = StaticBaseline(profiles, strategy="fastest")
    static_large = StaticBaseline(profiles, strategy="accurate")
    heuristic = ThroughputAutotuner(profiles)

    records = []
    for task, deadlines in deadlines_cfg.items():
        records.extend(evaluate_baseline(static_fast, task, deadlines))
        records.extend(evaluate_baseline(static_large, task, deadlines))
        records.extend(evaluate_baseline(heuristic, task, deadlines))

    df = pd.DataFrame(records)
    write_dataframe(df, Path(args.output))


if __name__ == "__main__":
    main()
