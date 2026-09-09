"""Paired A/B: does sharing the pool's sessions cost latency under concurrent load?

`share_sessions` is measured for memory and saves 66% at four workers. What it does to
latency is the open question, and it has one specific mechanism: shared workers run on
one ONNX Runtime per-session CPU arena instead of one each. Nothing else changes,
`intra_op_num_threads` is 1 either way so a Run stays single-threaded, and the weights
are read-only.

The gate
--------

**Single-worker service time is the control, and it is logically independent of the
treatment: at one worker there is nothing to share**, so the two arms load the identical
one backend. A host that has drifted shows up here and nowhere else. This is the same
discipline `results/ab_copy_threads/check_arm.py` applies to the decoder path, for the
same reason, that A/B had three degraded arms and three healthy ones inside 28 minutes
and the load average ranked them backwards.

An arm whose control is outside tolerance is not a slow result, it is not a result, and
the driver retakes it instead of recording it.

Pairing
-------

Arms alternate order between passes, because whichever runs first pays for a colder page
cache. Reported as a ratio of shared to unshared per pass, so a host that moved between
passes cannot masquerade as an effect.

Usage:
    python scripts/ab_session_sharing.py
    python scripts/ab_session_sharing.py --passes 5 --requests 3000
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from anytime_serving.serving.onnx_runtime import (  # noqa: E402
    InferenceRequest,
    RuntimePool,
)
from anytime_serving.utils.logger import get_logger  # noqa: E402

LOGGER = get_logger("scripts.ab_session_sharing")

# configs/serving.yaml, measured through this same path on a host that was known good.
CONTROL_MS = 12.893
CONTROL_TOLERANCE = 0.08
SEQUENCE_LENGTH = 128


def _feeds(batch: int = 1) -> dict[str, np.ndarray]:
    return {
        "input_ids": np.ones((batch, SEQUENCE_LENGTH), dtype=np.int64),
        "attention_mask": np.ones((batch, SEQUENCE_LENGTH), dtype=np.int64),
    }


def _control(paths: dict[str, Path], variant: str) -> float:
    """One worker, one request at a time. Cannot depend on `share_sessions`."""
    with RuntimePool(1, paths) as pool:
        for _ in range(30):
            pool.infer(InferenceRequest(variant=variant, inputs=_feeds()))
        samples = [
            pool.infer(InferenceRequest(variant=variant, inputs=_feeds())).runtime_latency_ms
            for _ in range(120)
        ]
    return statistics.median(samples)


def _arm(paths: dict[str, Path], variant: str, *, workers: int, requests: int, share: bool) -> dict:
    """Saturate the pool and record what each request cost inside the runtime."""
    feeds = _feeds()
    with RuntimePool(workers, paths, share_sessions=share) as pool:
        for _ in range(workers * 20):
            pool.infer(InferenceRequest(variant=variant, inputs=feeds))

        def one(index: int) -> float:
            return pool.infer(
                InferenceRequest(variant=variant, inputs=feeds, request_id=f"r{index}")
            ).runtime_latency_ms

        started = time.perf_counter()
        with ThreadPoolExecutor(max_workers=workers) as threads:
            latencies = list(threads.map(one, range(requests)))
        elapsed = time.perf_counter() - started

    ordered = sorted(latencies)
    return {
        "share_sessions": share,
        "backends": 1 if share else workers,
        "p50_ms": round(statistics.median(ordered), 3),
        "p95_ms": round(ordered[int(0.95 * len(ordered)) - 1], 3),
        "mean_ms": round(statistics.mean(ordered), 3),
        "throughput_rps": round(requests / elapsed, 1),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=Path("models/onnx"))
    parser.add_argument("--variant", default="distilbert_fp32")
    parser.add_argument("--workers", type=int, nargs="+", default=[2, 4, 8])
    parser.add_argument("--requests", type=int, default=2000)
    parser.add_argument("--passes", type=int, default=3)
    parser.add_argument("--max-retakes", type=int, default=6)
    parser.add_argument("--output", type=Path, default=Path("results/ab_session_sharing.json"))
    args = parser.parse_args()

    graph = (
        args.model_dir
        / f"text_{args.variant.split('_')[0]}_{args.variant.split('_')[1]}"
        / "model.onnx"
    )
    if not graph.is_file():
        raise SystemExit(
            f"{graph} is missing. Export it first:\n    python scripts/export_onnx.py --task text"
        )
    paths = {args.variant: graph}

    by_workers = []
    for workers in args.workers:
        passes, retakes = _sweep_point(paths, args, workers)
        by_workers.append(
            {
                "workers": workers,
                "retakes": retakes,
                "passes": passes,
                "median_p50_ratio": round(statistics.median(p["p50_ratio"] for p in passes), 4),
                "median_p95_ratio": round(statistics.median(p["p95_ratio"] for p in passes), 4),
                "median_throughput_ratio": round(
                    statistics.median(p["throughput_ratio"] for p in passes), 4
                ),
            }
        )
        LOGGER.info(
            "%d workers: p50 %.3fx  p95 %.3fx  throughput %.3fx  (%d retakes)",
            workers,
            by_workers[-1]["median_p50_ratio"],
            by_workers[-1]["median_p95_ratio"],
            by_workers[-1]["median_throughput_ratio"],
            retakes,
        )

    payload = {
        "question": "does sharing the pool's sessions cost latency under concurrent load",
        "control": {
            "recorded_ms": CONTROL_MS,
            "tolerance": CONTROL_TOLERANCE,
            "why": "at one worker there is nothing to share, so this cannot depend on "
            "the treatment",
        },
        "requests_per_arm": args.requests,
        "ratios_are": "shared / unshared, so below 1 means sharing is faster",
        "by_workers": by_workers,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    LOGGER.info("Wrote %s", args.output)
    return 0


def _sweep_point(paths, args, workers: int):
    passes: list[dict] = []
    retakes = 0
    while len(passes) < args.passes:
        # Alternate which arm runs first: the leader pays for the colder cache.
        shared_first = len(passes) % 2 == 1
        order = [True, False] if shared_first else [False, True]

        control_before = _control(paths, args.variant)
        ratio = control_before / CONTROL_MS
        if abs(ratio - 1.0) > CONTROL_TOLERANCE:
            retakes += 1
            LOGGER.warning(
                "control %.2f ms is %.3fx the record; the host is not itself. Retaking (%d/%d).",
                control_before,
                ratio,
                retakes,
                args.max_retakes,
            )
            if retakes > args.max_retakes:
                raise SystemExit(
                    "the host failed its control too many times; measure when it is quiet"
                )
            time.sleep(20)
            continue

        arms = {
            share: _arm(paths, args.variant, workers=workers, requests=args.requests, share=share)
            for share in order
        }
        control_after = _control(paths, args.variant)
        drift = abs(control_after / control_before - 1.0)
        if drift > CONTROL_TOLERANCE:
            retakes += 1
            LOGGER.warning(
                "the control moved %.1f%% across the pair, so the arms are not comparable. "
                "Retaking (%d/%d).",
                100 * drift,
                retakes,
                args.max_retakes,
            )
            if retakes > args.max_retakes:
                raise SystemExit("the host would not hold still; measure when it is quiet")
            continue

        shared, unshared = arms[True], arms[False]
        passes.append(
            {
                "shared_first": shared_first,
                "control_before_ms": round(control_before, 3),
                "control_after_ms": round(control_after, 3),
                "shared": shared,
                "unshared": unshared,
                "p50_ratio": round(shared["p50_ms"] / unshared["p50_ms"], 4),
                "p95_ratio": round(shared["p95_ms"] / unshared["p95_ms"], 4),
                "throughput_ratio": round(shared["throughput_rps"] / unshared["throughput_rps"], 4),
            }
        )
        LOGGER.info(
            "pass %d: p50 %.3fx  p95 %.3fx  throughput %.3fx  (control %.2f -> %.2f ms)",
            len(passes),
            passes[-1]["p50_ratio"],
            passes[-1]["p95_ratio"],
            passes[-1]["throughput_ratio"],
            control_before,
            control_after,
        )
    return passes, retakes


if __name__ == "__main__":
    sys.exit(main())
