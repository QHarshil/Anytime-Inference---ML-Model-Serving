"""Workload sensitivity analysis using recorded latency traces."""
from __future__ import annotations

import argparse
import json
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.io import save_csv
from src.utils.logger import get_logger

LOGGER = get_logger("experiments.workload")


def generate_steady_arrivals(duration_sec: float, arrival_rate: float) -> np.ndarray:
    """Generate Poisson arrivals for a steady workload."""

    estimate = int(duration_sec * arrival_rate * 1.2)
    inter_arrivals = np.random.exponential(1.0 / arrival_rate, size=estimate)
    arrivals = np.cumsum(inter_arrivals)
    return arrivals[arrivals < duration_sec]


def generate_bursty_arrivals(
    duration_sec: float,
    base_rate: float,
    burst_rate: float,
    burst_duration_sec: float = 5.0,
    burst_interval_sec: float = 20.0,
) -> np.ndarray:
    """Generate bursty arrivals alternating between low and high rates."""

    arrivals = []
    current = 0.0
    while current < duration_sec:
        burst_end = min(current + burst_duration_sec, duration_sec)
        burst = generate_steady_arrivals(burst_end - current, burst_rate) + current
        arrivals.extend(burst)
        current = burst_end

        quiet_end = min(current + (burst_interval_sec - burst_duration_sec), duration_sec)
        quiet = generate_steady_arrivals(quiet_end - current, base_rate) + current
        arrivals.extend(quiet)
        current = quiet_end

    return np.array(sorted(arrivals))


def simulate_queue(arrival_times: np.ndarray, latencies_ms: np.ndarray, deadline_ms: float) -> dict:
    """Simulate single-server queue fed by arrival_times and observed latencies."""

    queue = deque()
    completed = []
    queue_depths = []
    clock = 0.0

    for arrival, latency in zip(arrival_times, latencies_ms):
        clock = arrival

        while queue and queue[0]["completion_time"] <= clock:
            completed.append(queue.popleft())

        service_time = latency / 1000.0
        completion_time = clock + service_time
        queue.append(
            {
                "arrival_time": arrival,
                "completion_time": completion_time,
                "latency_ms": latency,
                "deadline_ms": deadline_ms,
            }
        )
        queue_depths.append(len(queue))

    while queue:
        completed.append(queue.popleft())

    if not completed:
        return {
            "deadline_hit_rate": 0.0,
            "avg_queue_depth": 0.0,
            "max_queue_depth": 0.0,
            "throughput": 0.0,
            "num_requests": 0,
        }

    completed_arrivals = np.array([req["arrival_time"] for req in completed])
    deadline_hits = np.mean([req["latency_ms"] <= req["deadline_ms"] for req in completed])
    duration = completed_arrivals[-1] - completed_arrivals[0] if len(completed_arrivals) > 1 else 0.0
    throughput = len(completed) / duration if duration > 0 else 0.0

    return {
        "deadline_hit_rate": float(deadline_hits),
        "avg_queue_depth": float(np.mean(queue_depths)) if queue_depths else 0.0,
        "max_queue_depth": float(np.max(queue_depths)) if queue_depths else 0.0,
        "throughput": float(throughput),
        "num_requests": int(len(completed)),
    }


def load_inference_cache(results_dir: Path) -> dict:
    cache_path = results_dir / "inference_cache" / "inference_measurements.json"
    if cache_path.exists():
        with cache_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    LOGGER.warning("Inference cache not found at %s", cache_path)
    return {}


def load_cascade_cache(results_dir: Path) -> dict:
    cache_path = results_dir / "inference_cache" / "cascade_measurements.json"
    if cache_path.exists():
        with cache_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    LOGGER.warning("Cascade cache not found at %s", cache_path)
    return {}


def get_latency_samples(cache: dict, task: str, model: str, variant: str, device: str, batch_size: int, seed: int) -> list:
    key = f"{task}_{model}_{variant}_{device}_{batch_size}_seed{seed}"
    entry = cache.get(key)
    if not entry:
        return []
    return entry.get("latency_samples_ms") or entry.get("latencies_ms", [])


def get_cascade_samples(cache: dict, task: str, threshold: float, device: str, seed: int) -> list:
    key = f"cascade_{task}_{threshold}_{device}_seed{seed}"
    entry = cache.get(key)
    if not entry:
        return []
    return entry.get("latency_samples_ms") or entry.get("latencies_ms", [])


def evaluate_workload(
    workload_type: str,
    task: str,
    deadline_ms: float,
    method_name: str,
    arrival_rate: float,
    latency_samples: list,
    fallback_stats: tuple[float, float],
    duration_sec: float,
    extra_meta: dict | None = None,
) -> dict:
    LOGGER.info("  %s workload @ %.1f req/s", workload_type, arrival_rate)

    if workload_type == "steady":
        arrivals = generate_steady_arrivals(duration_sec, arrival_rate)
    else:
        arrivals = generate_bursty_arrivals(duration_sec, base_rate=arrival_rate * 0.5, burst_rate=arrival_rate * 2.0)

    if arrivals.size == 0:
        arrivals = np.array([0.0])

    samples = np.array(latency_samples, dtype=float)
    if samples.size == 0:
        p50, p95 = fallback_stats
        sigma = np.log(p95 / p50) / 1.645 if p95 > p50 else 0.1
        mu = np.log(p50)
        latencies = np.random.lognormal(mu, sigma, size=len(arrivals))
    else:
        latencies = np.random.choice(samples, size=len(arrivals), replace=True)

    metrics = simulate_queue(arrivals, latencies, deadline_ms)
    result = {
        "workload_type": workload_type,
        "task": task,
        "deadline_ms": float(deadline_ms),
        "arrival_rate": float(arrival_rate),
        "method": method_name,
        **metrics,
    }
    if extra_meta:
        result.update(extra_meta)
    return result


def collect_latency_samples(rows: pd.DataFrame, cache: dict) -> list:
    samples = []
    for row in rows.itertuples():
        sample_values = get_latency_samples(
            cache,
            row.task,
            row.model,
            row.variant,
            row.device,
            int(row.batch_size),
            int(row.seed),
        )
        if sample_values:
            samples.extend(sample_values)
    return samples


def collect_cascade_samples(rows: pd.DataFrame, cache: dict) -> list:
    samples = []
    for row in rows.itertuples():
        sample_values = get_cascade_samples(cache, row.task, float(row.threshold), row.small_device, int(row.seed))
        if sample_values:
            samples.extend(sample_values)
    return samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    results_dir = Path("results")
    baseline_df = pd.read_csv(results_dir / "baseline_results.csv")
    planner_df = pd.read_csv(results_dir / "planner_results.csv")
    inference_cache = load_inference_cache(results_dir)
    cascade_cache = load_cascade_cache(results_dir)

    duration_sec = 30.0 if args.quick else 60.0
    arrival_rates = [5, 10] if args.quick else [5, 10, 20]
    workload_types = ["steady", "bursty"]

    records = []

    for task in ["text", "vision"]:
        LOGGER.info("\n%s workloads", task.upper())
        task_baseline = baseline_df[baseline_df["task"] == task]
        task_planner = planner_df[planner_df["task"] == task]

        for method in ["StaticSmall", "StaticLarge", "ThroughputAutotuner", "INFaaS-style"]:
            method_rows = task_baseline[task_baseline["method"] == method]
            if method_rows.empty:
                continue
            latency_samples = collect_latency_samples(method_rows, inference_cache)
            fallback_stats = (
                float(method_rows["lat_p50_ms"].mean()),
                float(method_rows["lat_p95_ms"].mean()),
            )
            deadline_ms = float(method_rows["deadline_ms"].iloc[0])

            for arrival_rate in arrival_rates:
                for workload in workload_types:
                    result = evaluate_workload(
                        workload,
                        task,
                        deadline_ms,
                        method,
                        arrival_rate,
                        latency_samples,
                        fallback_stats,
                        duration_sec,
                    )
                    records.append(result)
                    LOGGER.info(
                        "  %s %s: hit_rate=%.3f, queue_p95=%.1f",
                        method,
                        workload,
                        result["deadline_hit_rate"],
                        result["max_queue_depth"],
                    )

        if not task_planner.empty:
            for deadline_ms in sorted(task_planner["deadline_ms"].unique()):
                deadline_rows = task_planner[task_planner["deadline_ms"] == deadline_ms]
                if deadline_rows.empty:
                    continue
                threshold_scores = deadline_rows.groupby("threshold")["deadline_hit_rate"].mean()
                best_threshold = float(threshold_scores.idxmax())
                selected_rows = deadline_rows[deadline_rows["threshold"] == best_threshold]
                latency_samples = collect_cascade_samples(selected_rows, cascade_cache)
                fallback_stats = (
                    float(selected_rows["lat_p50_ms"].mean()),
                    float(selected_rows["lat_p95_ms"].mean()),
                )

                for arrival_rate in arrival_rates:
                    for workload in workload_types:
                        result = evaluate_workload(
                            workload,
                            task,
                            float(deadline_ms),
                            "CascadePlanner",
                            arrival_rate,
                            latency_samples,
                            fallback_stats,
                            duration_sec,
                            extra_meta={"threshold": best_threshold},
                        )
                        records.append(result)
                        LOGGER.info(
                            "  CascadePlanner τ=%.2f %s: hit_rate=%.3f, queue_p95=%.1f",
                            best_threshold,
                            workload,
                            result["deadline_hit_rate"],
                            result["max_queue_depth"],
                        )

    output_path = results_dir / "workload_sensitivity.csv"
    save_csv(pd.DataFrame(records), output_path)
    LOGGER.info("\nSaved workload sensitivity results to %s", output_path)


if __name__ == "__main__":
    main()
