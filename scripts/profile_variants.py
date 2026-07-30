"""Measure the service time and accuracy of every model variant.

The serving planner needs a per-variant service time and accuracy to decide
anything. Those numbers must be measured on the machine that will serve traffic:
carrying a number over from a different host, thread count, or batch size makes
every downstream admission decision wrong.

Measurement goes **through the serving path**, not beside it. Every latency and
every logit here comes from the same `RuntimeClient` the server dispatches to.
Stage 1 profiled through a separate ONNX Runtime session instead, and that is what
let a 7.6x discrepancy hide: the C++ worker was built against ONNX Runtime 1.20.1
while the profiler used the 1.26.0 wheel, so DistilBERT measured 98.9 ms in the
worker against 13.0 ms in the profiler. Nothing failed. Every service time the
planner used was simply false.

To keep that from recurring in a form the engine cannot see, each variant is also
measured through a separate ONNX Runtime session and the two are required to agree.
A divergence beyond --agreement-tolerance fails the run rather than being written
to disk. This is the check whose absence made Stage 1's numbers wrong.

Writes:

  results/variant_profiles.json   all measurements, agreement, host metadata
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

from anytime_serving.serving.onnx_runtime import (
    InferenceRequest,
    RuntimeClient,
    extension_available,
    load_extension,
)
from anytime_serving.utils.logger import get_logger

LOGGER = get_logger("scripts.profile_variants")

SEQUENCE_LENGTH = 128
WARMUP_ITERATIONS = 20
MEASURE_ITERATIONS = 200
QUICK_WARMUP_ITERATIONS = 5
QUICK_MEASURE_ITERATIONS = 40
QUICK_ACCURACY_SAMPLES = 128
# Matched versions measured within 0.4% on this host, and Stage 1 saw 8% across a
# process boundary. 15% is loose enough not to fire on scheduler noise and tight
# enough that anything structural, let alone a 7.6x version mismatch, trips it.
DEFAULT_AGREEMENT_TOLERANCE = 0.15


@dataclass
class VariantMeasurement:
    """Measured cost and quality for one variant, as served."""

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
    # Time inside inference as the engine reports it, against the same graph run
    # through a separate session. Agreement is what says the number is real.
    engine_inference_p50_ms: float
    direct_session_p50_ms: float
    agreement_ratio: float
    compute_cost_per_request: float = 1.0
    on_pareto_frontier: bool = False
    dominated_by: str | None = None


def _session(graph: Path) -> ort.InferenceSession:
    """Build a cross-check session matching the engine's configuration.

    runtime/src/engine.cpp sets intra-op and inter-op threads to one so that N
    pooled workers behave as N independent single-threaded servers. This session
    exists only to be compared against the engine, so it has to be configured the
    same way or the comparison measures the configuration difference instead.
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


def _tokenize(tokenizer, text: str) -> dict[str, np.ndarray]:
    """Encode one string. Not filtered to any graph's declared inputs.

    The engine drops inputs a graph does not declare, so the union is what the
    server sends and therefore what should be measured.
    """
    encoded = tokenizer(
        text,
        return_tensors="np",
        padding="max_length",
        truncation=True,
        max_length=SEQUENCE_LENGTH,
    )
    return {name: value.astype(np.int64) for name, value in encoded.items()}


def _declared(session: ort.InferenceSession, feeds: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    expected = {spec.name for spec in session.get_inputs()}
    return {name: value for name, value in feeds.items() if name in expected}


def _percentiles(samples: list[float], key: str) -> dict[str, float]:
    array = np.asarray(samples)
    return {
        f"{key}_p50_ms": float(np.percentile(array, 50)),
        f"{key}_p95_ms": float(np.percentile(array, 95)),
        f"{key}_p99_ms": float(np.percentile(array, 99)),
        f"{key}_mean_ms": statistics.fmean(samples),
        f"{key}_stdev_ms": statistics.stdev(samples) if len(samples) > 1 else 0.0,
    }


def _measure_through_engine(
    client: RuntimeClient,
    variant: str,
    feeds: dict[str, np.ndarray],
    *,
    warmup: int,
    iterations: int,
) -> dict[str, float]:
    """Latency as the server experiences it, through the runtime client."""
    request = InferenceRequest(variant=variant, inputs=feeds)
    for _ in range(warmup):
        client.infer(request)

    wall: list[float] = []
    inference: list[float] = []
    for _ in range(iterations):
        response = client.infer(request)
        wall.append(response.wall_latency_ms)
        inference.append(response.runtime_latency_ms)

    stats = _percentiles(wall, "wall")
    stats.update(_percentiles(inference, "inference"))
    stats["iterations"] = float(iterations)
    return stats


def _measure_direct_session(
    session: ort.InferenceSession,
    feeds: dict[str, np.ndarray],
    *,
    warmup: int,
    iterations: int,
) -> dict[str, float]:
    """The same graph through a separate session, for the agreement check."""
    fed = _declared(session, feeds)
    for _ in range(warmup):
        session.run(None, fed)

    samples: list[float] = []
    for _ in range(iterations):
        start = time.perf_counter()
        session.run(None, fed)
        samples.append((time.perf_counter() - start) * 1000.0)
    return _percentiles(samples, "direct")


def _measure_accuracy(
    client: RuntimeClient, variant: str, tokenizer, samples: int | None
) -> tuple[float, int]:
    """Accuracy on the SST-2 validation split, scored through the engine."""
    from datasets import load_dataset

    dataset = load_dataset("glue", "sst2", split="validation")
    if samples is not None:
        dataset = dataset.select(range(min(samples, len(dataset))))

    correct = 0
    for row in dataset:
        feeds = _tokenize(tokenizer, row["sentence"])
        logits = client.infer(InferenceRequest(variant=variant, inputs=feeds)).logits
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


def _discover(model_dir: Path) -> list[tuple[str, str, str, Path]]:
    """Find exported variants as (name, model, precision, graph)."""
    directories = sorted(model_dir.glob("text_*_*"))
    if not directories:
        raise SystemExit(
            f"no exported variants under {model_dir}. Run: "
            f"python scripts/export_onnx.py --task text --output-dir {model_dir}"
        )
    found = []
    for directory in directories:
        if not directory.is_dir():
            continue
        # Directory names are text_<model>_<precision>.
        _, model_name, precision = directory.name.split("_", 2)
        found.append((f"{model_name}_{precision}", model_name, precision, _find_graph(directory)))
    return found


def profile(model_dir: Path, quick: bool, tolerance: float) -> tuple[list[VariantMeasurement], str]:
    from transformers import AutoTokenizer

    warmup = QUICK_WARMUP_ITERATIONS if quick else WARMUP_ITERATIONS
    iterations = QUICK_MEASURE_ITERATIONS if quick else MEASURE_ITERATIONS
    accuracy_samples = QUICK_ACCURACY_SAMPLES if quick else None

    variants = _discover(model_dir)
    # One client holding every variant, which is how the server loads them.
    client = RuntimeClient({name: graph for name, _, _, graph in variants})
    LOGGER.info("Profiling through the %s backend", client.backend_name)

    measurements: list[VariantMeasurement] = []
    disagreements: list[str] = []
    try:
        for name, model_name, precision, graph in variants:
            tokenizer = AutoTokenizer.from_pretrained(graph.parent)
            feeds = _tokenize(tokenizer, "a genuinely measured service time")

            LOGGER.info("Profiling %s (%s)", name, graph.name)
            engine = _measure_through_engine(
                client, name, feeds, warmup=warmup, iterations=iterations
            )
            direct = _measure_direct_session(
                _session(graph), feeds, warmup=warmup, iterations=iterations
            )
            accuracy, n_samples = _measure_accuracy(client, name, tokenizer, accuracy_samples)

            ratio = engine["inference_p50_ms"] / direct["direct_p50_ms"]
            if abs(ratio - 1.0) > tolerance:
                disagreements.append(
                    f"  {name}: engine {engine['inference_p50_ms']:.3f} ms vs separate "
                    f"session {direct['direct_p50_ms']:.3f} ms ({ratio:.3f}x)"
                )

            measurements.append(
                VariantMeasurement(
                    name=name,
                    model=model_name,
                    precision=precision,
                    graph=graph.name,
                    size_mb=round(graph.stat().st_size / 1e6, 1),
                    # The planner admits on expected service time, so use the
                    # median of what a request actually costs the pool: the wall
                    # time through the client, not just time inside inference.
                    service_time_ms=round(engine["wall_p50_ms"], 3),
                    latency_p50_ms=round(engine["wall_p50_ms"], 3),
                    latency_p95_ms=round(engine["wall_p95_ms"], 3),
                    latency_p99_ms=round(engine["wall_p99_ms"], 3),
                    latency_mean_ms=round(engine["wall_mean_ms"], 3),
                    latency_stdev_ms=round(engine["wall_stdev_ms"], 3),
                    iterations=int(engine["iterations"]),
                    accuracy=round(accuracy, 4),
                    accuracy_samples=n_samples,
                    engine_inference_p50_ms=round(engine["inference_p50_ms"], 3),
                    direct_session_p50_ms=round(direct["direct_p50_ms"], 3),
                    agreement_ratio=round(ratio, 4),
                )
            )
            LOGGER.info(
                "  p50=%.2fms p95=%.2fms accuracy=%.4f (n=%d) size=%.1fMB agreement=%.3fx",
                engine["wall_p50_ms"],
                engine["wall_p95_ms"],
                accuracy,
                n_samples,
                graph.stat().st_size / 1e6,
                ratio,
            )
    finally:
        client.close()

    if disagreements:
        raise SystemExit(
            "The engine and a separate ONNX Runtime session disagree by more than "
            f"the {tolerance} relative bound:\n"
            + "\n".join(disagreements)
            + "\n\nThis is the check that Stage 1 lacked. A 7.6x version mismatch "
            "between the C++ worker and the onnxruntime wheel went unnoticed for "
            "exactly this reason, and every service time the planner used was wrong. "
            "Do not write these numbers to disk: confirm the extension and the wheel "
            "report the same ONNX Runtime version, then re-run. Pass "
            "--agreement-tolerance to widen the bound only if the difference is "
            "understood."
        )

    _mark_pareto_frontier(measurements)

    # Cost is service time relative to the most accurate variant: the quantity
    # the adaptive policy actually saves when it downgrades a request.
    reference = max(measurements, key=lambda m: m.accuracy)
    for measurement in measurements:
        measurement.compute_cost_per_request = round(
            measurement.service_time_ms / reference.service_time_ms, 4
        )
    return measurements, client.backend_name


def _host_metadata(backend: str) -> dict[str, object]:
    metadata: dict[str, object] = {
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "onnxruntime": ort.__version__,
        "providers": ort.get_available_providers(),
        "backend": backend,
        "intra_op_num_threads": 1,
        "batch_size": 1,
        "sequence_length": SEQUENCE_LENGTH,
    }
    if extension_available():
        metadata["extension_onnxruntime"] = load_extension().onnxruntime_version()
    return metadata


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
        "# Generated by scripts/profile_variants.py. Values are measured through the\n"
        "# serving path on the host recorded in results/variant_profiles.json;\n"
        "# re-run after changing hardware, thread count, or model variants.\n"
        + yaml.safe_dump(existing, sort_keys=False)
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
    parser.add_argument(
        "--agreement-tolerance",
        type=float,
        default=DEFAULT_AGREEMENT_TOLERANCE,
        help=(
            "Maximum relative difference between inference time through the engine "
            "and through a separate session before the run fails, as a fraction "
            f"(default {DEFAULT_AGREEMENT_TOLERANCE})"
        ),
    )
    parser.add_argument(
        "--allow-fallback-backend",
        action="store_true",
        help=(
            "Profile even when the anytime_runtime extension is unavailable. The "
            "numbers then describe the Python fallback rather than the serving path"
        ),
    )
    parser.add_argument("--output", type=Path, default=Path("results/variant_profiles.json"))
    parser.add_argument("--serving-config", type=Path, default=Path("configs/serving.yaml"))
    args = parser.parse_args()

    if not extension_available() and not args.allow_fallback_backend:
        raise SystemExit(
            "anytime_runtime is not available, so profiling would measure the Python "
            "fallback rather than the path that serves traffic. That divergence is "
            "what made Stage 1's service times wrong. Build the extension with:\n"
            "    pip install -e .\n"
            "or pass --allow-fallback-backend to record fallback numbers deliberately."
        )

    measurements, backend = profile(args.model_dir, args.quick, args.agreement_tolerance)
    frontier = [m for m in measurements if m.on_pareto_frontier]

    payload = {
        "host": _host_metadata(backend),
        "quick": args.quick,
        "task": "text",
        "accuracy_dataset": "glue/sst2 validation",
        "agreement_tolerance": args.agreement_tolerance,
        "variants": [asdict(m) for m in measurements],
        "pareto_frontier": [m.name for m in frontier],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    LOGGER.info("Wrote %s", args.output)

    _write_serving_config(args.serving_config, frontier, args.workers)
    LOGGER.info("Wrote %s", args.serving_config)

    LOGGER.info("")
    LOGGER.info(
        "%-18s %10s %10s %9s %10s  %s",
        "variant",
        "p50 (ms)",
        "accuracy",
        "size(MB)",
        "agreement",
        "verdict",
    )
    for m in sorted(measurements, key=lambda m: m.service_time_ms):
        verdict = "frontier" if m.on_pareto_frontier else f"dominated by {m.dominated_by}"
        LOGGER.info(
            "%-18s %10.2f %10.4f %9.1f %9.3fx  %s",
            m.name,
            m.service_time_ms,
            m.accuracy,
            m.size_mb,
            m.agreement_ratio,
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
