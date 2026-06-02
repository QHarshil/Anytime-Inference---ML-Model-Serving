"""Concurrent-traffic benchmark for the adaptive planner.

Compares two policies under the same Poisson arrival stream:

- ``fp32-only``: every request goes to the full-precision variant.
- ``adaptive``: the planner picks FP32 or INT8 based on real-time CPU load,
  admission-controlled with an M/M/1 sojourn-time bound.

Outputs per-request traces and a summary CSV; prints the cost-reduction
percentage achieved by the adaptive policy.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd

from src.serving.admission import MM1AdmissionController
from src.serving.load_monitor import LoadMonitor
from src.serving.onnx_runtime import InferenceRequest, RuntimePool, find_runtime_binary
from src.serving.selector import AdaptiveSelector, VariantProfile
from src.serving.server import AdaptiveServer, drive_workload, poisson_arrivals
from src.utils.logger import get_logger

LOGGER = get_logger("scripts.run_benchmark")


def _build_selector(
    variants: List[VariantProfile], adaptive: bool
) -> AdaptiveSelector:
    if not adaptive:
        fp32 = next(v for v in variants if v.name == "fp32")
        return AdaptiveSelector([fp32], load_knee_percent=999.0, load_slope=0.0)
    return AdaptiveSelector(
        variants,
        load_knee_percent=50.0,
        load_slope=0.02,
        admission_controller=MM1AdmissionController(safety_factor=1.0),
    )


def _profiles_from_args(args: argparse.Namespace) -> List[VariantProfile]:
    return [
        VariantProfile(
            name="fp32",
            service_time_ms=args.fp32_service_ms,
            accuracy=args.fp32_accuracy,
            compute_cost_per_request=args.fp32_cost,
        ),
        VariantProfile(
            name="int8",
            service_time_ms=args.int8_service_ms,
            accuracy=args.int8_accuracy,
            compute_cost_per_request=args.int8_cost,
        ),
    ]


def _make_request_factory(input_shape: List[int], seed: int):
    rng = np.random.default_rng(seed)

    def factory(_i: int) -> InferenceRequest:
        data = rng.standard_normal(input_shape).astype(np.float32)
        return InferenceRequest(variant="fp32", data=data)

    return factory


def _resolve_model_paths(args: argparse.Namespace) -> Dict[str, Path]:
    paths: Dict[str, Path] = {}
    if args.fp32_model:
        paths["fp32"] = Path(args.fp32_model)
    if args.int8_model:
        paths["int8"] = Path(args.int8_model)
    if not paths:
        raise SystemExit("at least one of --fp32-model / --int8-model must be set")
    for variant, path in paths.items():
        if not path.exists():
            raise SystemExit(f"missing ONNX model for {variant}: {path}")
    return paths


def run_policy(
    *,
    label: str,
    adaptive: bool,
    args: argparse.Namespace,
    model_paths: Dict[str, Path],
    arrival_times: List[float],
    input_shape: List[int],
) -> pd.DataFrame:
    LOGGER.info("Running policy: %s (n=%d arrivals)", label, len(arrival_times))
    profiles = _profiles_from_args(args)
    selector = _build_selector(profiles, adaptive=adaptive)
    monitor = LoadMonitor(interval_s=0.25, alpha=0.3)
    monitor.start()
    try:
        with RuntimePool(args.workers, model_paths, binary=find_runtime_binary(),
                         input_name=args.input_name) as pool:
            server = AdaptiveServer(pool, selector, monitor)
            try:
                drive_workload(
                    server,
                    _make_request_factory(input_shape, args.seed),
                    arrival_times=arrival_times,
                    deadline_ms=args.deadline_ms,
                )
            finally:
                server.shutdown()
            stats = server.stats
    finally:
        monitor.stop()

    rows = [asdict(r) for r in stats.requests]
    df = pd.DataFrame(rows)
    df["policy"] = label
    LOGGER.info(
        "  accepted=%d rejected=%d completion=%.3f hit_rate=%.3f cost=%.2f",
        stats.accepted, stats.rejected, stats.completion_rate,
        stats.deadline_hit_rate, stats.cumulative_compute_cost,
    )
    return df


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fp32-model", type=Path, required=True)
    parser.add_argument("--int8-model", type=Path, required=True)
    parser.add_argument("--input-name", default="input")
    parser.add_argument("--input-shape", type=int, nargs="+", default=[1, 3, 224, 224])
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--duration", type=float, default=20.0,
                        help="Benchmark duration in seconds")
    parser.add_argument("--arrival-rate", type=float, default=80.0,
                        help="Mean Poisson arrival rate in rps")
    parser.add_argument("--deadline-ms", type=float, default=60.0)
    parser.add_argument("--seed", type=int, default=0)
    # Profile defaults are pessimistic placeholders; pass measured numbers from
    # latency profiling to get a faithful benchmark.
    parser.add_argument("--fp32-service-ms", type=float, default=12.0)
    parser.add_argument("--int8-service-ms", type=float, default=5.0)
    parser.add_argument("--fp32-accuracy", type=float, default=0.91)
    parser.add_argument("--int8-accuracy", type=float, default=0.89)
    parser.add_argument("--fp32-cost", type=float, default=1.0)
    parser.add_argument("--int8-cost", type=float, default=0.42)
    parser.add_argument("--output", type=Path, default=Path("results/serving_benchmark.csv"))
    args = parser.parse_args()

    model_paths = _resolve_model_paths(args)
    rng = np.random.default_rng(args.seed)
    arrival_times = poisson_arrivals(args.duration, args.arrival_rate, rng=rng)
    LOGGER.info("Generated %d Poisson arrivals over %.1fs (mean %.1f rps)",
                len(arrival_times), args.duration, args.arrival_rate)

    fp32_df = run_policy(
        label="fp32-only",
        adaptive=False,
        args=args,
        model_paths={k: v for k, v in model_paths.items() if k == "fp32"},
        arrival_times=arrival_times,
        input_shape=args.input_shape,
    )
    time.sleep(1.0)
    adaptive_df = run_policy(
        label="adaptive",
        adaptive=True,
        args=args,
        model_paths=model_paths,
        arrival_times=arrival_times,
        input_shape=args.input_shape,
    )

    combined = pd.concat([fp32_df, adaptive_df], ignore_index=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(args.output, index=False)

    def _per_served_cost(df: pd.DataFrame) -> float:
        served = df[df["admitted"]]
        return float(served["compute_cost"].mean()) if len(served) else 0.0

    fp32_served_cost = _per_served_cost(fp32_df)
    adaptive_served_cost = _per_served_cost(adaptive_df)
    cost_reduction = (
        (fp32_served_cost - adaptive_served_cost) / fp32_served_cost
        if fp32_served_cost > 0 else 0.0
    )

    fp32_completion = float(fp32_df["admitted"].mean()) if len(fp32_df) else 0.0
    adaptive_completion = float(adaptive_df["admitted"].mean()) if len(adaptive_df) else 0.0
    fp32_hit_rate = float(
        ((fp32_df["admitted"]) & (fp32_df["wall_latency_ms"] <= args.deadline_ms)).mean()
    ) if len(fp32_df) else 0.0
    adaptive_hit_rate = float(
        ((adaptive_df["admitted"]) & (adaptive_df["wall_latency_ms"] <= args.deadline_ms)).mean()
    ) if len(adaptive_df) else 0.0

    summary = {
        "arrivals": len(arrival_times),
        "deadline_ms": args.deadline_ms,
        "fp32_completion_rate": fp32_completion,
        "adaptive_completion_rate": adaptive_completion,
        "fp32_deadline_hit_rate": fp32_hit_rate,
        "adaptive_deadline_hit_rate": adaptive_hit_rate,
        "fp32_mean_cost_per_served": fp32_served_cost,
        "adaptive_mean_cost_per_served": adaptive_served_cost,
        "cost_reduction_pct": cost_reduction * 100.0,
    }
    print(json.dumps(summary, indent=2))
    LOGGER.info("Wrote per-request trace to %s", args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
