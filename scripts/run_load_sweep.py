"""Sweep offered load through the real models and compare policies.

Offered load is expressed as a target utilisation of the *measured* pool capacity
rather than as a bare request rate. Capacity is ``workers / service_time`` for the
most accurate variant, so a four-worker pool serving a 12.8 ms model saturates
near 310 rps. A sweep that stops at 90 rps never leaves the idle regime and shows
nothing; the interesting behaviour is around and past rho = 1.

Both policies see the same Poisson arrival stream:

  accurate-only  every request goes to the highest-accuracy variant
  adaptive       the planner picks a variant under an M/M/c sojourn bound

Requests run through the exported ONNX graphs, so latencies, deadline hits, and
misses are measured rather than assumed.

Outputs:
  results/load_sweep.csv     one row per (policy, target utilisation)
  docs/img/load_sweep.png    completion, attainment, latency, and cost vs load

Usage:
    python scripts/run_load_sweep.py --duration 3
    python scripts/run_load_sweep.py --quick
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

from anytime_serving.serving.load_monitor import LoadMonitor
from anytime_serving.serving.onnx_runtime import InferenceRequest, RuntimePool, find_runtime_binary
from anytime_serving.serving.selector import AdaptiveSelector, VariantProfile
from anytime_serving.serving.server import AdaptiveServer, drive_workload, poisson_arrivals
from anytime_serving.utils.logger import get_logger

LOGGER = get_logger("scripts.run_load_sweep")

SEQUENCE_LENGTH = 128
DEFAULT_UTILISATIONS = (0.4, 0.6, 0.8, 0.95, 1.1, 1.3)
QUICK_UTILISATIONS = (0.6, 0.95, 1.3)


def _load_profiles(config_path: Path) -> tuple[list[VariantProfile], float, int]:
    if not config_path.exists():
        raise SystemExit(f"missing {config_path}. Run: python scripts/profile_variants.py")
    config = yaml.safe_load(config_path.read_text())
    variants = [
        VariantProfile(
            name=name,
            service_time_ms=float(spec["service_time_ms"]),
            accuracy=float(spec["accuracy"]),
            compute_cost_per_request=float(spec["compute_cost_per_request"]),
        )
        for name, spec in config["variants"].items()
    ]
    if len(variants) < 2:
        raise SystemExit(
            "serving.yaml lists fewer than two variants; the adaptive policy has "
            "nothing to trade off. Re-run scripts/profile_variants.py."
        )
    return variants, float(config["deadline_ms"]), int(config["workers"])


def _graph_for(model_dir: Path, variant: str) -> Path:
    directory = model_dir / f"text_{variant}"
    if not directory.is_dir():
        raise SystemExit(f"missing {directory}. Run scripts/export_onnx.py --task text")
    graphs = sorted(directory.glob("*.onnx"))
    for graph in graphs:
        if "quantized" in graph.name:
            return graph
    if not graphs:
        raise SystemExit(f"no .onnx graph in {directory}")
    return graphs[0]


def _tokenised_inputs(model_dir: Path, variant: str, count: int, seed: int) -> list[dict]:
    """Pre-tokenise a pool of real SST-2 sentences.

    Tokenisation happens before the run so the measured latency is inference, not
    text preprocessing.
    """
    from datasets import load_dataset
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_dir / f"text_{variant}")
    dataset = load_dataset("glue", "sst2", split="validation")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(dataset), size=count)
    # Send the union of tokenizer outputs. Variants declare different subsets
    # (DistilBERT omits token_type_ids, BERT-family models require it) and the
    # runtime drops whatever a given graph does not declare.
    encoded = tokenizer(
        [dataset[int(i)]["sentence"] for i in indices],
        return_tensors="np",
        padding="max_length",
        truncation=True,
        max_length=SEQUENCE_LENGTH,
        return_token_type_ids=True,
    )
    return [
        {
            name: np.ascontiguousarray(value[i : i + 1].astype(np.int64))
            for name, value in encoded.items()
        }
        for i in range(count)
    ]


def _run_policy(
    *,
    label: str,
    variants: list[VariantProfile],
    model_dir: Path,
    arrivals: list[float],
    inputs: list[dict],
    deadline_ms: float,
    workers: int,
) -> dict:
    model_paths = {v.name: _graph_for(model_dir, v.name) for v in variants}
    selector = AdaptiveSelector(
        variants,
        servers=workers,
        load_knee_percent=50.0,
        load_slope=0.02,
    )
    monitor = LoadMonitor(interval_s=0.1, alpha=0.3)
    monitor.start()
    try:
        with RuntimePool(workers, model_paths, binary=find_runtime_binary()) as pool:
            server = AdaptiveServer(pool, selector, monitor)
            try:

                def factory(index: int) -> InferenceRequest:
                    # DistilBERT and MiniLM both take input_ids and attention_mask,
                    # so the request carries the full named feed.
                    return InferenceRequest(
                        variant=variants[0].name, inputs=inputs[index % len(inputs)]
                    )

                drive_workload(server, factory, arrival_times=arrivals, deadline_ms=deadline_ms)
            finally:
                server.shutdown()
            stats = server.stats
    finally:
        monitor.stop()

    served = [r for r in stats.requests if r.admitted]
    latencies = np.asarray([r.wall_latency_ms for r in served]) if served else np.zeros(1)
    mix = {v.name: 0 for v in variants}
    for record in served:
        mix[record.variant] = mix.get(record.variant, 0) + 1

    return {
        "policy": label,
        "attempted": stats.total,
        "served": stats.accepted,
        "rejected": stats.rejected,
        "completion_rate": stats.completion_rate,
        "deadline_hit_rate": stats.deadline_hit_rate,
        "latency_p50_ms": float(np.percentile(latencies, 50)),
        "latency_p95_ms": float(np.percentile(latencies, 95)),
        "latency_p99_ms": float(np.percentile(latencies, 99)),
        "mean_cost_per_served": (
            sum(r.compute_cost for r in served) / len(served) if served else 0.0
        ),
        # Goodput: served within the deadline, per second of offered traffic.
        "goodput_rps": stats.deadline_hits / max(arrivals[-1], 1e-9) if arrivals else 0.0,
        "variant_mix": mix,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=Path("models/onnx"))
    parser.add_argument("--serving-config", type=Path, default=Path("configs/serving.yaml"))
    parser.add_argument("--duration", type=float, default=3.0, help="Seconds per sweep point")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--quick", action="store_true", help="Three sweep points instead of six")
    parser.add_argument("--output", type=Path, default=Path("results/load_sweep.csv"))
    parser.add_argument("--figure", type=Path, default=Path("docs/img/load_sweep.png"))
    args = parser.parse_args()

    variants, deadline_ms, workers = _load_profiles(args.serving_config)
    variants.sort(key=lambda v: -v.accuracy)
    accurate = variants[0]
    capacity_rps = workers * 1000.0 / accurate.service_time_ms

    LOGGER.info(
        "Frontier: %s",
        ", ".join(f"{v.name} {v.service_time_ms:.2f}ms/{v.accuracy:.4f}" for v in variants),
    )
    LOGGER.info(
        "Pool capacity for %s: %d workers / %.2f ms = %.0f rps (deadline %.1f ms)",
        accurate.name,
        workers,
        accurate.service_time_ms,
        capacity_rps,
        deadline_ms,
    )

    utilisations = QUICK_UTILISATIONS if args.quick else DEFAULT_UTILISATIONS
    max_requests = int(max(utilisations) * capacity_rps * args.duration * 1.3) + 64
    LOGGER.info("Pre-tokenising %d SST-2 inputs", max_requests)
    inputs = _tokenised_inputs(args.model_dir, accurate.name, max_requests, args.seed)

    rows: list[dict] = []
    for utilisation in utilisations:
        arrival_rate = utilisation * capacity_rps
        arrivals = poisson_arrivals(
            args.duration, arrival_rate, rng=np.random.default_rng(args.seed + 1)
        )
        LOGGER.info(
            "rho=%.2f  arrival_rate=%.0f rps  arrivals=%d", utilisation, arrival_rate, len(arrivals)
        )

        for label, policy_variants in (
            ("accurate-only", [accurate]),
            ("adaptive", variants),
        ):
            result = _run_policy(
                label=label,
                variants=policy_variants,
                model_dir=args.model_dir,
                arrivals=arrivals,
                inputs=inputs,
                deadline_ms=deadline_ms,
                workers=workers,
            )
            result["target_utilisation"] = utilisation
            result["arrival_rate_rps"] = arrival_rate
            rows.append(result)
            LOGGER.info(
                "  %-14s completion=%.3f attainment=%.3f p95=%.1fms "
                "goodput=%.0f rps cost=%.3f mix=%s",
                label,
                result["completion_rate"],
                result["deadline_hit_rate"],
                result["latency_p95_ms"],
                result["goodput_rps"],
                result["mean_cost_per_served"],
                result["variant_mix"],
            )

    _write_outputs(rows, args.output, args.figure, accurate, deadline_ms, workers, capacity_rps)
    return 0


def _write_outputs(
    rows: list[dict],
    csv_path: Path,
    figure_path: Path,
    accurate: VariantProfile,
    deadline_ms: float,
    workers: int,
    capacity_rps: float,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    frame = pd.DataFrame([{k: v for k, v in row.items() if k != "variant_mix"} for row in rows])
    frame["variant_mix"] = [json.dumps(row["variant_mix"]) for row in rows]
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(csv_path, index=False)
    LOGGER.info("Wrote %s", csv_path)

    accurate_rows = frame[frame["policy"] == "accurate-only"].sort_values("target_utilisation")
    adaptive_rows = frame[frame["policy"] == "adaptive"].sort_values("target_utilisation")

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.0))
    styles = {
        "accurate-only": {"color": "#b2432f", "marker": "o", "label": f"{accurate.name} only"},
        "adaptive": {"color": "#2f6fb2", "marker": "s", "label": "adaptive"},
    }

    panels = [
        ("completion_rate", "Admitted fraction", 100.0, "%"),
        ("deadline_hit_rate", "Deadline attainment", 100.0, "%"),
        ("latency_p95_ms", "p95 latency", 1.0, "ms"),
        ("goodput_rps", "Goodput (met deadline)", 1.0, "rps"),
    ]
    for axis, (column, title, scale, unit) in zip(axes.ravel(), panels, strict=True):
        for data, style in (
            (accurate_rows, styles["accurate-only"]),
            (adaptive_rows, styles["adaptive"]),
        ):
            axis.plot(
                data["target_utilisation"],
                data[column] * scale,
                marker=style["marker"],
                color=style["color"],
                label=style["label"],
            )
        axis.axvline(1.0, color="grey", linestyle=":", linewidth=1)
        axis.set_xlabel(r"Offered load / pool capacity ($\rho$)")
        axis.set_ylabel(f"{title} ({unit})")
        axis.set_title(title)
        axis.grid(True, alpha=0.3)
    if deadline_ms:
        axes.ravel()[2].axhline(deadline_ms, color="grey", linestyle="--", linewidth=1)
        axes.ravel()[2].annotate(
            f"deadline {deadline_ms:.0f} ms",
            xy=(axes.ravel()[2].get_xlim()[0], deadline_ms),
            xytext=(4, 4),
            textcoords="offset points",
            color="grey",
            fontsize=8,
        )
    axes.ravel()[0].legend(loc="lower left", fontsize=9)

    fig.suptitle(
        f"Measured serving behaviour vs offered load "
        f"({workers} workers, capacity {capacity_rps:.0f} rps, deadline {deadline_ms:.0f} ms)",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_path, dpi=150)
    LOGGER.info("Wrote %s", figure_path)


if __name__ == "__main__":
    sys.exit(main())
