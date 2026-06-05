"""Sweep offered load and plot the adaptive policy against the FP32-only baseline.

Builds a synthetic ONNX model so the sweep runs without torch or the C++ build,
then drives the serving harness at a range of arrival rates under the variant
profiles in `configs/serving.yaml`. Produces:

  docs/serving_benchmark.png   - completion rate and cost reduction vs load
  docs/benchmark_results.csv   - the underlying numbers

Usage:
    python scripts/plot_benchmark.py
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from typing import Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import onnx
import pandas as pd
from onnx import TensorProto, helper

from src.serving.admission import MM1AdmissionController
from src.serving.load_monitor import LoadMonitor
from src.serving.onnx_runtime import InferenceRequest, RuntimePool
from src.serving.selector import AdaptiveSelector, VariantProfile
from src.serving.server import AdaptiveServer, drive_workload, poisson_arrivals
from src.utils.logger import get_logger

LOGGER = get_logger("scripts.plot_benchmark")

DEADLINE_MS = 45.0
WORKERS = 4
DURATION_S = 2.0
ARRIVAL_RATES = [20.0, 35.0, 50.0, 55.0, 60.0, 65.0, 75.0, 90.0]

FP32 = VariantProfile("fp32", service_time_ms=12.0, accuracy=0.91, compute_cost_per_request=1.0)
INT8 = VariantProfile("int8", service_time_ms=5.0, accuracy=0.89, compute_cost_per_request=0.42)


def _build_identity_model(path: Path, dim: int = 4) -> None:
    inp = helper.make_tensor_value_info("input", TensorProto.FLOAT, [None, dim])
    out = helper.make_tensor_value_info("logits", TensorProto.FLOAT, [None, dim])
    node = helper.make_node("Identity", ["input"], ["logits"])
    graph = helper.make_graph([node], "identity", [inp], [out])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 14)], ir_version=8)
    onnx.save(model, str(path))


def _run(adaptive: bool, model_paths: Dict[str, Path], arrivals: List[float], seed: int):
    variants = [FP32, INT8] if adaptive else [FP32]
    selector = AdaptiveSelector(
        variants,
        load_knee_percent=40.0,
        load_slope=0.015,
        admission_controller=MM1AdmissionController(),
    )
    monitor = LoadMonitor(interval_s=0.05)
    monitor.start()
    rng = np.random.default_rng(seed)
    try:
        with RuntimePool(WORKERS, model_paths) as pool:
            server = AdaptiveServer(pool, selector, monitor)
            try:
                def factory(_i: int) -> InferenceRequest:
                    return InferenceRequest("fp32", rng.standard_normal((1, 4)).astype(np.float32))

                drive_workload(server, factory, arrival_times=arrivals, deadline_ms=DEADLINE_MS)
            finally:
                server.shutdown()
            stats = server.stats
    finally:
        monitor.stop()
    served = [r for r in stats.requests if r.admitted]
    mean_cost = sum(r.compute_cost for r in served) / len(served) if served else 0.0
    int8_frac = sum(1 for r in served if r.variant == "int8") / len(served) if served else 0.0
    return {
        "completion_rate": stats.completion_rate,
        "deadline_hit_rate": stats.deadline_hit_rate,
        "mean_cost": mean_cost,
        "int8_frac": int8_frac,
    }


def main() -> int:
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        fp32_path = Path(tmp) / "fp32.onnx"
        int8_path = Path(tmp) / "int8.onnx"
        _build_identity_model(fp32_path)
        _build_identity_model(int8_path)

        for rate in ARRIVAL_RATES:
            arrivals = poisson_arrivals(DURATION_S, rate, rng=np.random.default_rng(int(rate)))
            fp32 = _run(False, {"fp32": fp32_path}, arrivals, seed=0)
            adaptive = _run(True, {"fp32": fp32_path, "int8": int8_path}, arrivals, seed=0)
            reduction = (
                (fp32["mean_cost"] - adaptive["mean_cost"]) / fp32["mean_cost"] * 100.0
                if fp32["mean_cost"] > 0 else 0.0
            )
            rows.append({
                "arrival_rate_rps": rate,
                "fp32_completion": fp32["completion_rate"],
                "adaptive_completion": adaptive["completion_rate"],
                "fp32_hit_rate": fp32["deadline_hit_rate"],
                "adaptive_hit_rate": adaptive["deadline_hit_rate"],
                "adaptive_int8_frac": adaptive["int8_frac"],
                "cost_reduction_pct": reduction,
            })
            LOGGER.info(
                "rate=%.0f  fp32_completion=%.2f  adaptive_completion=%.2f  cost_reduction=%.1f%%",
                rate, fp32["completion_rate"], adaptive["completion_rate"], reduction,
            )

    df = pd.DataFrame(rows)
    docs = REPO_ROOT / "docs"
    docs.mkdir(exist_ok=True)
    df.to_csv(docs / "benchmark_results.csv", index=False)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

    ax1.plot(df["arrival_rate_rps"], df["fp32_completion"] * 100, "o-", color="#c0392b", label="FP32-only")
    ax1.plot(df["arrival_rate_rps"], df["adaptive_completion"] * 100, "s-", color="#1f77b4", label="Adaptive")
    ax1.set_xlabel("Offered load (requests/s)")
    ax1.set_ylabel("Request completion (%)")
    ax1.set_title("Completion under concurrent load")
    ax1.set_ylim(0, 105)
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2.plot(df["arrival_rate_rps"], df["cost_reduction_pct"], "D-", color="#2e7d32")
    ax2.axhline(45.0, color="grey", linestyle="--", linewidth=1)
    ax2.annotate("45%", xy=(df["arrival_rate_rps"].iloc[0], 45.0), xytext=(2, 4),
                 textcoords="offset points", color="grey", fontsize=9)
    ax2.set_xlabel("Offered load (requests/s)")
    ax2.set_ylabel("Compute-cost reduction (%)")
    ax2.set_title("Cost reduction vs FP32-only")
    ax2.set_ylim(0, max(60.0, df["cost_reduction_pct"].max() * 1.1))
    ax2.grid(True, alpha=0.3)

    fig.suptitle(
        f"Adaptive serving under configured profiles "
        f"(FP32 {FP32.service_time_ms:.0f} ms / INT8 {INT8.service_time_ms:.0f} ms, "
        f"deadline {DEADLINE_MS:.0f} ms)",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out_path = docs / "serving_benchmark.png"
    fig.savefig(out_path, dpi=150)
    LOGGER.info("Wrote %s and %s", out_path, docs / "benchmark_results.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
