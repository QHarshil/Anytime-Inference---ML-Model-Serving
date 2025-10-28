"""Run statistical significance tests between planner and baselines."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.utils.io import read_dataframe, write_dataframe
from src.utils.logger import configure_logger

try:  # pragma: no cover - optional dependency
    from scipy import stats
except Exception:  # pragma: no cover
    stats = None  # type: ignore

PRIMARY_METRICS = ["deadline_hit_rate", "accuracy"]


def paired_tests(values1: pd.Series, values2: pd.Series) -> dict:
    if stats is None:
        return {"t_pvalue": float("nan"), "wilcoxon_pvalue": float("nan")}
    t_stat, t_pvalue = stats.ttest_rel(values1, values2)
    try:
        _, w_pvalue = stats.wilcoxon(values1, values2)
    except ValueError:
        w_pvalue = float("nan")
    return {"t_pvalue": float(t_pvalue), "wilcoxon_pvalue": float(w_pvalue)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--planner", default="results/planner_results.csv")
    parser.add_argument("--baseline", default="results/baseline_results.csv")
    parser.add_argument("--output", default="results/statistical_tests.csv")
    args = parser.parse_args()

    configure_logger()
    planner_df = read_dataframe(Path(args.planner))
    baseline_df = read_dataframe(Path(args.baseline))

    records = []
    for metric in PRIMARY_METRICS:
        if metric not in planner_df.columns or metric not in baseline_df.columns:
            continue
        merged = planner_df.merge(
            baseline_df[["deadline", "task", metric]],
            on=["deadline", "task"],
            suffixes=("_planner", "_baseline"),
        )
        stats_record = paired_tests(merged[f"{metric}_planner"], merged[f"{metric}_baseline"])
        stats_record.update({"metric": metric})
        records.append(stats_record)

    df = pd.DataFrame(records)
    write_dataframe(df, Path(args.output))


if __name__ == "__main__":
    main()
