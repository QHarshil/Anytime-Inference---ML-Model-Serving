"""Measure the service time and accuracy of every model variant.

The serving planner needs a per-variant service time and accuracy to decide
anything. Those numbers must be measured on the machine that will serve traffic:
carrying a number over from a different host, thread count, or batch size makes
every downstream admission decision wrong.

This script measures every exported candidate under the same ONNX Runtime
configuration the serving worker uses (one intra-op thread per worker, batch size
one), computes the Pareto frontier over (service time, accuracy), and writes:

  results/variant_profiles.json   all measurements plus host metadata
  configs/serving.yaml            the frontier, as the serving harness reads it

Which variants end up on the frontier is a property of the hardware, not an
assumption. Dynamic INT8 quantisation, for instance, reliably shrinks a model
about fourfold but does not necessarily make it faster: on Apple Silicon it
measured slower than the FP32 graph it replaced, so it is dominated and excluded.

Accuracy is evaluated on SST-2 validation. Both text candidates are fine-tuned
for that task, so the numbers are meaningful; pairing an ImageNet-pretrained
vision model with CIFAR-10 would not be.

Usage:
    python scripts/profile_variants.py
    python scripts/profile_variants.py --quick
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import onnxruntime as ort

from anytime_serving.utils.logger import get_logger

LOGGER = get_logger("scripts.profile_variants")

SEQUENCE_LENGTH = 128
WARMUP_ITERATIONS = 20
MEASURE_ITERATIONS = 200
QUICK_WARMUP_ITERATIONS = 5
QUICK_MEASURE_ITERATIONS = 40
QUICK_ACCURACY_SAMPLES = 128


@dataclass
class VariantMeasurement:
    """Measured cost and quality for one variant."""

    name: str
    model: str
    precision: str
    graph: str
    size_mb: float
    service_time_ms: float
    latency_p50_ms: float
    latency_p95_ms: float
    latency_p99_ms: float
    latency_mean_ms: float
    latency_stdev_ms: float
    iterations: int
    accuracy: float
    accuracy_samples: int
    compute_cost_per_request: float = 1.0
    on_pareto_frontier: bool = False
    dominated_by: str | None = None


def _session(graph: Path) -> ort.InferenceSession:
    """Build a session matching the serving worker's configuration.

    runtime_cpp/src/main.cpp sets intra-op threads to one so that N pooled
    workers behave as N independent single-threaded servers. Profiling with a
    different thread count would measure a machine the planner never runs on.
    """
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(
        str(graph), sess_options=options, providers=["CPUExecutionProvider"]
    )


def _find_graph(directory: Path) -> Path:
    graphs = sorted(directory.glob("*.onnx"))
    if not graphs:
        raise SystemExit(f"no .onnx graph found in {directory}")
    for graph in graphs:
        if "quantized" in graph.name:
            return graph
    return graphs[0]


def _feeds(tokenizer, session: ort.InferenceSession, text: str) -> dict[str, np.ndarray]:
    encoded = tokenizer(
        text,
        return_tensors="np",
        padding="max_length",
        truncation=True,
        max_length=SEQUENCE_LENGTH,
    )
    expected = {i.name for i in session.get_inputs()}
    return {name: value.astype(np.int64) for name, value in encoded.items() if name in expected}


def _measure_latency(
    session: ort.InferenceSession,
    feeds: dict[str, np.ndarray],
    *,
    warmup: int,
    iterations: int,
) -> dict[str, float]:
    for _ in range(warmup):
        session.run(None, feeds)

    samples: list[float] = []
    for _ in range(iterations):
        start = time.perf_counter()
        session.run(None, feeds)
        samples.append((time.perf_counter() - start) * 1000.0)

    array = np.asarray(samples)
    return {
        "latency_p50_ms": float(np.percentile(array, 50)),
        "latency_p95_ms": float(np.percentile(array, 95)),
        "latency_p99_ms": float(np.percentile(array, 99)),
        "latency_mean_ms": statistics.fmean(samples),
        "latency_stdev_ms": statistics.stdev(samples) if len(samples) > 1 else 0.0,
        "iterations": float(len(samples)),
    }


def _measure_accuracy(
    session: ort.InferenceSession, tokenizer, samples: int | None
) -> tuple[float, int]:
    """Accuracy on the SST-2 validation split."""
    from datasets import load_dataset

    dataset = load_dataset("glue", "sst2", split="validation")
    if samples is not None:
        dataset = dataset.select(range(min(samples, len(dataset))))

    correct = 0
    for row in dataset:
        logits = session.run(None, _feeds(tokenizer, session, row["sentence"]))[0]
        if int(np.argmax(logits, axis=-1)[0]) == int(row["label"]):
            correct += 1
    return correct / len(dataset), len(dataset)


def _mark_pareto_frontier(measurements: list[VariantMeasurement]) -> None:
    """Flag variants that are not dominated on (service time, accuracy).

    A variant is dominated when another is at least as fast and at least as
    accurate, and strictly better on one of the two. Only frontier variants are
    worth offering the planner: a dominated variant is never the right choice.
    """
    for candidate in measurements:
        dominator = next(
            (
                other
                for other in measurements
                if other is not candidate
                and other.service_time_ms <= candidate.service_time_ms
                and other.accuracy >= candidate.accuracy
                and (
                    other.service_time_ms < candidate.service_time_ms
                    or other.accuracy > candidate.accuracy
                )
            ),
            None,
        )
        candidate.on_pareto_frontier = dominator is None
        candidate.dominated_by = dominator.name if dominator else None


def profile(model_dir: Path, quick: bool) -> list[VariantMeasurement]:
    from transformers import AutoTokenizer

    warmup = QUICK_WARMUP_ITERATIONS if quick else WARMUP_ITERATIONS
    iterations = QUICK_MEASURE_ITERATIONS if quick else MEASURE_ITERATIONS
    accuracy_samples = QUICK_ACCURACY_SAMPLES if quick else None

    directories = sorted(model_dir.glob("text_*_*"))
    if not directories:
        raise SystemExit(
            f"no exported variants under {model_dir}. Run: "
            f"python scripts/export_onnx.py --task text --output-dir {model_dir}"
        )

    measurements: list[VariantMeasurement] = []
    for directory in directories:
        if not directory.is_dir():
            continue
        # Directory names are text_<model>_<precision>.
        _, model_name, precision = directory.name.split("_", 2)
        graph = _find_graph(directory)
        tokenizer = AutoTokenizer.from_pretrained(directory)
        session = _session(graph)

        name = f"{model_name}_{precision}"
        LOGGER.info("Profiling %s (%s)", name, graph.name)
        latency = _measure_latency(
            session,
            _feeds(tokenizer, session, "a genuinely measured service time"),
            warmup=warmup,
            iterations=iterations,
        )
        accuracy, n_samples = _measure_accuracy(session, tokenizer, accuracy_samples)

        measurements.append(
            VariantMeasurement(
                name=name,
                model=model_name,
                precision=precision,
                graph=graph.name,
                size_mb=round(graph.stat().st_size / 1e6, 1),
                # The planner admits on expected service time, so use the median:
                # it is not skewed by occasional scheduler noise.
                service_time_ms=round(latency["latency_p50_ms"], 3),
                latency_p50_ms=round(latency["latency_p50_ms"], 3),
                latency_p95_ms=round(latency["latency_p95_ms"], 3),
                latency_p99_ms=round(latency["latency_p99_ms"], 3),
                latency_mean_ms=round(latency["latency_mean_ms"], 3),
                latency_stdev_ms=round(latency["latency_stdev_ms"], 3),
                iterations=int(latency["iterations"]),
                accuracy=round(accuracy, 4),
                accuracy_samples=n_samples,
            )
        )
        LOGGER.info(
            "  p50=%.2fms p95=%.2fms p99=%.2fms accuracy=%.4f (n=%d) size=%.1fMB",
            latency["latency_p50_ms"],
            latency["latency_p95_ms"],
            latency["latency_p99_ms"],
            accuracy,
            n_samples,
            graph.stat().st_size / 1e6,
        )

    _mark_pareto_frontier(measurements)

    # Cost is service time relative to the most accurate variant: the quantity
    # the adaptive policy actually saves when it downgrades a request.
    reference = max(measurements, key=lambda m: m.accuracy)
    for measurement in measurements:
        measurement.compute_cost_per_request = round(
            measurement.service_time_ms / reference.service_time_ms, 4
        )
    return measurements


def _host_metadata() -> dict[str, object]:
    return {
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "onnxruntime": ort.__version__,
        "providers": ort.get_available_providers(),
        "intra_op_num_threads": 1,
        "batch_size": 1,
        "sequence_length": SEQUENCE_LENGTH,
    }


def _write_serving_config(path: Path, frontier: list[VariantMeasurement], workers: int) -> None:
    """Write the measured frontier into the config the serving harness reads."""
    import yaml

    existing: dict = {}
    if path.exists():
        existing = yaml.safe_load(path.read_text()) or {}

    slowest = max(m.service_time_ms for m in frontier)
    # Enough headroom that a single request is not already at the limit; the load
    # sweep explores utilisation up to and past saturation from here.
    deadline_ms = float(round(3.0 * slowest, 1))

    existing.update(
        {
            "deadline_ms": deadline_ms,
            "workers": workers,
            "variants": {
                m.name: {
                    "service_time_ms": m.service_time_ms,
                    "accuracy": m.accuracy,
                    "compute_cost_per_request": m.compute_cost_per_request,
                }
                for m in sorted(frontier, key=lambda m: -m.accuracy)
            },
        }
    )
    existing.setdefault("load_knee_percent", 50.0)
    existing.setdefault("load_slope", 0.02)
    existing.setdefault("admission", {"safety_factor": 1.0})

    path.write_text(
        "# Generated by scripts/profile_variants.py. Values are measured on the\n"
        "# host recorded in results/variant_profiles.json; re-run after changing\n"
        "# hardware, thread count, or model variants.\n" + yaml.safe_dump(existing, sort_keys=False)
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=Path("models/onnx"))
    parser.add_argument("--workers", type=int, default=4, help="Pool size the profile targets")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Fewer iterations and a subset of the validation split",
    )
    parser.add_argument("--output", type=Path, default=Path("results/variant_profiles.json"))
    parser.add_argument("--serving-config", type=Path, default=Path("configs/serving.yaml"))
    args = parser.parse_args()

    measurements = profile(args.model_dir, args.quick)
    frontier = [m for m in measurements if m.on_pareto_frontier]

    payload = {
        "host": _host_metadata(),
        "quick": args.quick,
        "task": "text",
        "accuracy_dataset": "glue/sst2 validation",
        "variants": [asdict(m) for m in measurements],
        "pareto_frontier": [m.name for m in frontier],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    LOGGER.info("Wrote %s", args.output)

    _write_serving_config(args.serving_config, frontier, args.workers)
    LOGGER.info("Wrote %s", args.serving_config)

    LOGGER.info("")
    LOGGER.info("%-18s %10s %10s %9s  %s", "variant", "p50 (ms)", "accuracy", "size(MB)", "verdict")
    for m in sorted(measurements, key=lambda m: m.service_time_ms):
        verdict = "frontier" if m.on_pareto_frontier else f"dominated by {m.dominated_by}"
        LOGGER.info(
            "%-18s %10.2f %10.4f %9.1f  %s",
            m.name,
            m.service_time_ms,
            m.accuracy,
            m.size_mb,
            verdict,
        )

    if len(frontier) < 2:
        LOGGER.warning(
            "Only %d variant(s) on the frontier: the adaptive policy has nothing to "
            "trade off on this host.",
            len(frontier),
        )
    else:
        fastest = min(frontier, key=lambda m: m.service_time_ms)
        best = max(frontier, key=lambda m: m.accuracy)
        LOGGER.info("")
        LOGGER.info(
            "Frontier spans %.2fx service time for %.2f accuracy points (%s -> %s)",
            best.service_time_ms / fastest.service_time_ms,
            (best.accuracy - fastest.accuracy) * 100.0,
            best.name,
            fastest.name,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
