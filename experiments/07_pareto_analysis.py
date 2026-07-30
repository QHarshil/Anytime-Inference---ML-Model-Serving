"""Pareto frontier analysis: hypervolume, dominance ratio, Pareto efficiency."""

import argparse
from pathlib import Path

import pandas as pd

from anytime_serving.evaluation.pareto import (
    compute_hypervolume,
    compute_pareto_frontier,
    dominance_ratio,
)
from anytime_serving.utils.io import save_csv
from anytime_serving.utils.logger import get_logger

LOGGER = get_logger("experiments.pareto")

LATENCY_COL = "lat_p95_ms"
ACCURACY_COL = "accuracy"


def load_results(results_dir: Path):
    baseline_df = pd.read_csv(results_dir / "baseline_results.csv")
    planner_df = pd.read_csv(results_dir / "planner_results.csv")
    return baseline_df, planner_df


def _points(df: pd.DataFrame):
    return list(zip(df[LATENCY_COL].tolist(), df[ACCURACY_COL].tolist(), strict=True))


def analyze_pareto(baseline_df, planner_df, task: str):
    LOGGER.info("Analyzing task=%s", task)
    baseline_task = baseline_df[baseline_df["task"] == task]
    planner_task = planner_df[planner_df["task"] == task]

    baseline_agg = (
        baseline_task.groupby(["method", "deadline_ms"])[[LATENCY_COL, ACCURACY_COL]]
        .mean()
        .reset_index()
    )
    planner_agg = (
        planner_task.groupby(["threshold", "deadline_ms"])[[LATENCY_COL, ACCURACY_COL]]
        .mean()
        .reset_index()
    )

    combined = pd.concat(
        [baseline_agg[[LATENCY_COL, ACCURACY_COL]], planner_agg[[LATENCY_COL, ACCURACY_COL]]]
    )
    reference_point = (float(combined[LATENCY_COL].max()), float(combined[ACCURACY_COL].min()))
    LOGGER.info("  reference latency=%.1fms accuracy=%.3f", *reference_point)

    rows = []
    for method in baseline_agg["method"].unique():
        method_df = baseline_agg[baseline_agg["method"] == method]
        pareto_df = compute_pareto_frontier(
            method_df, latency_col=LATENCY_COL, accuracy_col=ACCURACY_COL
        )
        hv = compute_hypervolume(_points(pareto_df), reference_point)
        eff = len(pareto_df) / len(method_df) if len(method_df) else 0.0
        rows.append(
            {
                "method": method,
                "task": task,
                "hypervolume": hv,
                "num_points": len(method_df),
                "num_pareto_points": len(pareto_df),
                "pareto_efficiency": eff,
                "dominance_ratio": 0.0,
            }
        )
        LOGGER.info("  %s: HV=%.2f pareto=%d/%d", method, hv, len(pareto_df), len(method_df))

    planner_pareto = compute_pareto_frontier(
        planner_agg, latency_col=LATENCY_COL, accuracy_col=ACCURACY_COL
    )
    hv_planner = compute_hypervolume(_points(planner_pareto), reference_point)
    eff_planner = len(planner_pareto) / len(planner_agg) if len(planner_agg) else 0.0
    dom_ratio = dominance_ratio(
        planner_agg, baseline_agg, latency_col=LATENCY_COL, accuracy_col=ACCURACY_COL
    )
    rows.append(
        {
            "method": "CascadePlanner",
            "task": task,
            "hypervolume": hv_planner,
            "num_points": len(planner_agg),
            "num_pareto_points": len(planner_pareto),
            "pareto_efficiency": eff_planner,
            "dominance_ratio": dom_ratio,
        }
    )
    LOGGER.info(
        "  CascadePlanner: HV=%.2f pareto=%d/%d dom=%.2f",
        hv_planner,
        len(planner_pareto),
        len(planner_agg),
        dom_ratio,
    )
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Accepted for pipeline uniformity. This stage post-processes existing\n"
        "results, so its cost already follows the upstream sample size.",
    )
    parser.parse_args()

    results_dir = Path("results")
    baseline_df, planner_df = load_results(results_dir)

    all_rows = []
    for task in ("text", "vision"):
        all_rows.extend(analyze_pareto(baseline_df, planner_df, task))

    results_df = pd.DataFrame(all_rows)
    output_path = results_dir / "pareto_analysis.csv"
    save_csv(results_df, output_path)
    LOGGER.info("Wrote %d rows to %s", len(results_df), output_path)


if __name__ == "__main__":
    main()
