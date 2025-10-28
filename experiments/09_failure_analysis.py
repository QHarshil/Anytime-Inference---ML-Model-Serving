"""Failure analysis leveraging recorded latency traces."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.io import save_csv
from src.utils.logger import get_logger

LOGGER = get_logger("experiments.failure")


def load_cache(path: Path) -> dict:
    if path.exists():
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    LOGGER.warning("Cache not found at %s", path)
    return {}


def baseline_key(task: str, model: str, variant: str, device: str, batch_size: int, seed: int) -> str:
    return f"{task}_{model}_{variant}_{device}_{batch_size}_seed{seed}"


def cascade_key(task: str, threshold: float, device: str, seed: int) -> str:
    return f"cascade_{task}_{threshold}_{device}_seed{seed}"


def collect_samples(rows: pd.DataFrame, cache: dict) -> np.ndarray:
    samples = []
    for row in rows.itertuples():
        key = baseline_key(row.task, row.model, row.variant, row.device, int(row.batch_size), int(row.seed))
        entry = cache.get(key)
        if not entry:
            continue
        data = entry.get("latency_samples_ms") or entry.get("latencies_ms", [])
        samples.extend(data)
    return np.asarray(samples, dtype=float)


def collect_cascade_samples(rows: pd.DataFrame, cache: dict) -> np.ndarray:
    samples = []
    for row in rows.itertuples():
        key = cascade_key(row.task, float(row.threshold), row.small_device, int(row.seed))
        entry = cache.get(key)
        if not entry:
            continue
        data = entry.get("latency_samples_ms") or entry.get("latencies_ms", [])
        samples.extend(data)
    return np.asarray(samples, dtype=float)


def deadline_metrics(latencies: np.ndarray, deadline_ms: float) -> dict:
    if latencies.size == 0:
        return {
            "miss_rate": 0.0,
            "hit_rate": 0.0,
            "p50_latency_ms": 0.0,
            "p95_latency_ms": 0.0,
        }
    miss_rate = float(np.mean(latencies > deadline_ms))
    return {
        "miss_rate": miss_rate,
        "hit_rate": 1.0 - miss_rate,
        "p50_latency_ms": float(np.percentile(latencies, 50)),
        "p95_latency_ms": float(np.percentile(latencies, 95)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    results_dir = Path("results")
    baseline_df = pd.read_csv(results_dir / "baseline_results.csv")
    planner_df = pd.read_csv(results_dir / "planner_results.csv")
    inference_cache = load_cache(results_dir / "inference_cache" / "inference_measurements.json")
    cascade_cache = load_cache(results_dir / "inference_cache" / "cascade_measurements.json")

    miss_records = []

    for task in ["text", "vision"]:
        task_baseline = baseline_df[baseline_df["task"] == task]
        task_planner = planner_df[planner_df["task"] == task]

        for method in task_baseline["method"].unique():
            method_rows = task_baseline[task_baseline["method"] == method]
            for deadline_ms in sorted(method_rows["deadline_ms"].unique()):
                deadline_rows = method_rows[method_rows["deadline_ms"] == deadline_ms]
                samples = collect_samples(deadline_rows, inference_cache)
                metrics = deadline_metrics(samples, deadline_ms)
                miss_records.append(
                    {
                        "method": method,
                        "task": task,
                        "deadline_ms": float(deadline_ms),
                        "num_samples": int(samples.size),
                        **metrics,
                    }
                )

        if not task_planner.empty:
            for deadline_ms in sorted(task_planner["deadline_ms"].unique()):
                deadline_rows = task_planner[task_planner["deadline_ms"] == deadline_ms]
                if deadline_rows.empty:
                    continue
                best_threshold = (
                    deadline_rows.groupby("threshold")["deadline_hit_rate"].mean().idxmax()
                )
                selected_rows = deadline_rows[deadline_rows["threshold"] == best_threshold]
                samples = collect_cascade_samples(selected_rows, cascade_cache)
                metrics = deadline_metrics(samples, deadline_ms)
                miss_records.append(
                    {
                        "method": "CascadePlanner",
                        "task": task,
                        "deadline_ms": float(deadline_ms),
                        "threshold": float(best_threshold),
                        "num_samples": int(samples.size),
                        **metrics,
                    }
                )

    miss_df = pd.DataFrame(miss_records)
    save_csv(miss_df, results_dir / "failure_miss_analysis.csv")
    LOGGER.info("Saved deadline miss analysis to %s", results_dir / "failure_miss_analysis.csv")

    # Graceful degradation strategies
    degradation_records = []
    for task in ["text", "vision"]:
        task_baseline = baseline_df[baseline_df["task"] == task]
        static_small = task_baseline[task_baseline["method"] == "StaticSmall"]
        static_large = task_baseline[task_baseline["method"] == "StaticLarge"]

        if not static_small.empty:
            samples = collect_samples(static_small, inference_cache)
            metrics = deadline_metrics(samples, float(static_small["deadline_ms"].iloc[0]))
            degradation_records.append(
                {
                    "strategy": "fast_fallback",
                    "task": task,
                    "description": "Fallback to fastest configuration",
                    **metrics,
                }
            )

        if not static_large.empty:
            samples = collect_samples(static_large, inference_cache)
            metrics = deadline_metrics(samples, float(static_large["deadline_ms"].iloc[0]))
            degradation_records.append(
                {
                    "strategy": "accuracy_fallback",
                    "task": task,
                    "description": "Fallback to most accurate configuration",
                    **metrics,
                }
            )

        task_planner = planner_df[planner_df["task"] == task]
        if not task_planner.empty:
            for deadline_ms in sorted(task_planner["deadline_ms"].unique()):
                deadline_rows = task_planner[task_planner["deadline_ms"] == deadline_ms]
                if deadline_rows.empty:
                    continue
                best_threshold = (
                    deadline_rows.groupby("threshold")["deadline_hit_rate"].mean().idxmax()
                )
                selected_rows = deadline_rows[deadline_rows["threshold"] == best_threshold]
                samples = collect_cascade_samples(selected_rows, cascade_cache)
                metrics = deadline_metrics(samples, deadline_ms)
                degradation_records.append(
                    {
                        "strategy": "adaptive_threshold",
                        "task": task,
                        "deadline_ms": float(deadline_ms),
                        "description": "Adaptive cascade threshold",
                        "threshold": float(best_threshold),
                        **metrics,
                    }
                )

    degradation_df = pd.DataFrame(degradation_records)
    save_csv(degradation_df, results_dir / "failure_degradation_strategies.csv")
    LOGGER.info(
        "Saved graceful degradation strategies to %s",
        results_dir / "failure_degradation_strategies.csv",
    )


if __name__ == "__main__":
    main()
