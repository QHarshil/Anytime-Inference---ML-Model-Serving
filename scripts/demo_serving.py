"""Self-contained demo of the adaptive serving stack.

Builds a tiny synthetic ONNX model (so the demo runs without torch or a real
model export) and drives Poisson traffic through both the FP32-only baseline
and the load-adaptive policy. Reports completion rate, deadline hit rate, and
per-served-request cost reduction.

Usage:
    python scripts/demo_serving.py
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper

from anytime_serving.serving.load_monitor import LoadMonitor
from anytime_serving.serving.onnx_runtime import InferenceRequest, RuntimePool
from anytime_serving.serving.selector import AdaptiveSelector, VariantProfile
from anytime_serving.serving.server import AdaptiveServer, drive_workload, poisson_arrivals
from anytime_serving.utils.logger import get_logger

LOGGER = get_logger("scripts.demo_serving")


def _build_identity_model(path: Path, input_dim: int = 4) -> None:
    inp = helper.make_tensor_value_info("input", TensorProto.FLOAT, [None, input_dim])
    out = helper.make_tensor_value_info("logits", TensorProto.FLOAT, [None, input_dim])
    node = helper.make_node("Identity", ["input"], ["logits"])
    graph = helper.make_graph([node], "identity", [inp], [out])
    opset = helper.make_opsetid("", 14)
    model = helper.make_model(graph, opset_imports=[opset], ir_version=8)
    onnx.save(model, str(path))


def _run_policy(
    *,
    label: str,
    adaptive: bool,
    model_paths: dict[str, Path],
    arrivals: list[float],
    deadline_ms: float,
    workers: int,
    seed: int,
):
    if adaptive:
        variants = [
            VariantProfile(
                "fp32", service_time_ms=12.0, accuracy=0.91, compute_cost_per_request=1.0
            ),
            VariantProfile(
                "int8", service_time_ms=5.0, accuracy=0.89, compute_cost_per_request=0.42
            ),
        ]
    else:
        variants = [VariantProfile("fp32", 12.0, 0.91, 1.0)]
    selector = AdaptiveSelector(
        variants,
        servers=workers,
        load_knee_percent=40.0,
        load_slope=0.015,
    )
    monitor = LoadMonitor(interval_s=0.05)
    monitor.start()
    rng = np.random.default_rng(seed)
    try:
        with RuntimePool(workers, model_paths) as pool:
            server = AdaptiveServer(pool, selector, monitor)
            try:

                def factory(_i: int) -> InferenceRequest:
                    return InferenceRequest(
                        variant="fp32",
                        data=rng.standard_normal((1, 4)).astype(np.float32),
                    )

                drive_workload(server, factory, arrival_times=arrivals, deadline_ms=deadline_ms)
            finally:
                server.shutdown()
            stats = server.stats
    finally:
        monitor.stop()

    served = [r for r in stats.requests if r.admitted]
    mean_cost = sum(r.compute_cost for r in served) / len(served) if served else 0.0
    int8_frac = sum(1 for r in served if r.variant == "int8") / len(served) if served else 0.0
    return {
        "label": label,
        "attempted": stats.total,
        "served": stats.accepted,
        "rejected": stats.rejected,
        "completion_rate": stats.completion_rate,
        "deadline_hit_rate": stats.deadline_hit_rate,
        "mean_cost_per_served": mean_cost,
        "int8_routing_fraction": int8_frac,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=float, default=3.0, help="Workload duration (s)")
    parser.add_argument("--arrival-rate", type=float, default=75.0, help="Mean Poisson rate (rps)")
    parser.add_argument("--deadline-ms", type=float, default=45.0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        fp32_path = tmp_path / "fp32.onnx"
        int8_path = tmp_path / "int8.onnx"
        _build_identity_model(fp32_path)
        _build_identity_model(int8_path)

        arrivals = poisson_arrivals(
            args.duration, args.arrival_rate, rng=np.random.default_rng(args.seed + 1)
        )
        LOGGER.info(
            "Driving %d Poisson arrivals over %.1fs (%.1f rps, deadline=%.1fms)",
            len(arrivals),
            args.duration,
            args.arrival_rate,
            args.deadline_ms,
        )

        fp32_result = _run_policy(
            label="fp32-only",
            adaptive=False,
            model_paths={"fp32": fp32_path},
            arrivals=arrivals,
            deadline_ms=args.deadline_ms,
            workers=args.workers,
            seed=args.seed,
        )
        adaptive_result = _run_policy(
            label="adaptive",
            adaptive=True,
            model_paths={"fp32": fp32_path, "int8": int8_path},
            arrivals=arrivals,
            deadline_ms=args.deadline_ms,
            workers=args.workers,
            seed=args.seed,
        )

    if fp32_result["mean_cost_per_served"] > 0:
        cost_reduction_pct = (
            (fp32_result["mean_cost_per_served"] - adaptive_result["mean_cost_per_served"])
            / fp32_result["mean_cost_per_served"]
            * 100.0
        )
    else:
        cost_reduction_pct = 0.0

    summary = {
        "deadline_ms": args.deadline_ms,
        "arrival_rate_rps": args.arrival_rate,
        "fp32_only": fp32_result,
        "adaptive": adaptive_result,
        "cost_reduction_pct": cost_reduction_pct,
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
