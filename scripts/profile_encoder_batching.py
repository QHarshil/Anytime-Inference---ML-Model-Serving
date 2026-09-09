"""What an encoder batch is worth per request, gated on a control that cannot depend on it.

The counting side of encoder batching is in `scripts/count_encoder_batching.py` and needs
no quiet host. This is the other half and it does: it times one `Session::Run` at several
batch widths and asks whether K requests in one Run beat K requests in K Runs.

The comparison this makes, and the one it deliberately does not
-----------------------------------------------------------------

Every row here is **sequence length 128 for every request**, which is what
`run_load_sweep.py` and `profile_variants.py` tokenise to. So no arm pays more padding
than any other and this isolates the GEMM effect alone. The padding a real ragged
workload would add is measured separately and exactly by the counting script, 18.7% at
width 2 rising to 47.8% at width 32, and the two multiply. Mixing them into one number
would hide which half moved.

What it does not measure is the pool. `intra_op_num_threads` is 1, as in serving, and the
pool already turns N cores into N concurrent single-request Runs. So a width-K speedup
below K is not a win: it is the same core-seconds spent less parallelisably. The decoder
gains from batching because a one-token step is a GEMV with nothing to divide; a
128-token encoder request is already a GEMM, which is why this was worth measuring,
not assuming.

The gate
--------

**This host runs about 2x slow in twenty-minute stretches and the load average does not
see it.** So every pass re-measures **DistilBERT FP32 at width 1** and compares it against
the 12.893 ms in `configs/serving.yaml`. That control is *logically* independent of the
treatment: at width 1 there is no batching, so it cannot move because batching worked or
failed, and it is the same graph and the same number
`scripts/ab_session_sharing.py` gates the encoder on. A pass whose control is outside the
tolerance is discarded, not recorded, which is the rule
`results/ab_copy_threads/check_arm.py` applies to the decoder arms.

**One control session, used for every variant.** The first version of this script gated
each variant on its own recorded service time, which silently gated nothing for the INT8
variants: `configs/serving.yaml` carries service times for the two FP32 variants only, so
`recorded` was None, the check was skipped, and the payload still declared itself gated.
A separate control arm is what makes the claim true for every row instead of for two of
them.

Arms alternate direction from pass to pass, so a host drifting monotonically through the
run cannot favour whichever width happened to be measured first.

Usage:
    python scripts/profile_encoder_batching.py
    python scripts/profile_encoder_batching.py --widths 1 2 4 8 16 --passes 5
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from anytime_serving.utils.logger import get_logger  # noqa: E402

LOGGER = get_logger("scripts.profile_encoder_batching")

SEQUENCE_LENGTH = 128
DEFAULT_WIDTHS = (1, 2, 4, 8, 16)

# The graph and the width the gate is taken on. DistilBERT FP32 because
# configs/serving.yaml records a service time for it and because that is the number
# scripts/ab_session_sharing.py already gates the encoder against.
GATE_VARIANT = "distilbert_fp32"

VARIANT_PATHS = {
    "distilbert_fp32": Path("models/onnx/text_distilbert_fp32/model.onnx"),
    "distilbert_int8": Path("models/onnx/text_distilbert_int8/model_quantized.onnx"),
    "minilm_fp32": Path("models/onnx/text_minilm_fp32/model.onnx"),
    "minilm_int8": Path("models/onnx/text_minilm_int8/model_quantized.onnx"),
}


def time_width(session, declared: set[str], width: int, *, reps: int, warmup: int) -> float:
    """Median per-request ms for one Run of `width` rows at SEQUENCE_LENGTH.

    Per request instead of per Run, so the number is directly comparable to the
    service time the planner uses and to every other width.
    """
    rng = np.random.default_rng(20260824 + width)
    feeds = {
        "input_ids": rng.integers(0, 1000, size=(width, SEQUENCE_LENGTH)).astype(np.int64),
        "attention_mask": np.ones((width, SEQUENCE_LENGTH), dtype=np.int64),
        "token_type_ids": np.zeros((width, SEQUENCE_LENGTH), dtype=np.int64),
    }
    fed = {name: value for name, value in feeds.items() if name in declared}
    for _ in range(warmup):
        session.run(None, fed)
    samples = []
    for _ in range(reps):
        start = time.perf_counter()
        session.run(None, fed)
        samples.append((time.perf_counter() - start) * 1000.0 / width)
    return statistics.median(samples)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--widths", type=int, nargs="+", default=list(DEFAULT_WIDTHS))
    parser.add_argument("--passes", type=int, default=5)
    parser.add_argument("--reps", type=int, default=25)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--variants", nargs="+", default=sorted(VARIANT_PATHS))
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.20,
        help="fractional band the width-1 control must land in to keep a pass",
    )
    parser.add_argument("--output", type=Path, default=Path("results/encoder_batching_timed.json"))
    args = parser.parse_args()

    if 1 not in args.widths:
        parser.error("width 1 is the control and cannot be omitted")

    import onnxruntime as ort
    import yaml

    config = yaml.safe_load(Path("configs/serving.yaml").read_text())

    def build(path: Path):
        options = ort.SessionOptions()
        # As in serving: one intra-op thread, so a Run is single-threaded and the pool
        # is what provides concurrency. Batching wider on one thread has to beat the
        # pool spending the same cores on separate requests.
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        session = ort.InferenceSession(
            str(path), sess_options=options, providers=["CPUExecutionProvider"]
        )
        return session, {spec.name for spec in session.get_inputs()}

    # The control. One graph, one width, one recorded number, used to accept or reject
    # every pass for every variant.
    control_variant = GATE_VARIANT
    control_path = VARIANT_PATHS[control_variant]
    control_recorded = config["variants"][control_variant]["service_time_ms"]
    if not control_path.exists():
        LOGGER.error(
            "the control graph %s is absent, so nothing here can be gated and nothing "
            "will be measured. models/ is not committed; run scripts/export_onnx.py.",
            control_path,
        )
        return 1
    control_session, control_declared = build(control_path)

    results = []
    for variant in args.variants:
        path = VARIANT_PATHS.get(variant)
        if path is None:
            parser.error(f"unknown variant {variant!r}; known: {sorted(VARIANT_PATHS)}")
        if not path.exists():
            LOGGER.error("missing %s; models/ is not committed. Skipping %s.", path, variant)
            continue

        session, declared = build(path)
        recorded = config["variants"].get(variant, {}).get("service_time_ms")
        per_width: dict[int, list[float]] = {w: [] for w in args.widths}
        controls: list[float] = []
        kept = discarded = 0

        for index in range(args.passes):
            # The control first, so a pass is accepted or rejected on the host state it
            # was actually measured in.
            control = time_width(
                control_session, control_declared, 1, reps=args.reps, warmup=args.warmup
            )
            if abs(control / control_recorded - 1.0) > args.tolerance:
                discarded += 1
                LOGGER.warning(
                    "  pass %d discarded: the %s width-1 control read %.3f ms against a "
                    "recorded %.3f ms, outside +-%.0f%%. The host is not itself.",
                    index + 1,
                    control_variant,
                    control,
                    control_recorded,
                    100.0 * args.tolerance,
                )
                continue
            # Alternate direction, so a host drifting through the run cannot favour
            # whichever width is measured first.
            order = args.widths if index % 2 == 0 else list(reversed(args.widths))
            pass_result = {
                w: time_width(session, declared, w, reps=args.reps, warmup=args.warmup)
                for w in order
            }
            kept += 1
            controls.append(control)
            for width, value in pass_result.items():
                per_width[width].append(value)

        if not kept:
            LOGGER.error(
                "%s: every pass failed the control. This is not a slow result, it is "
                "not a result. Re-run on an idle host.",
                variant,
            )
            results.append(
                {"variant": variant, "passes_kept": 0, "passes_discarded": discarded, "widths": []}
            )
            continue

        baseline = statistics.median(per_width[1])
        widths = []
        for width in sorted(args.widths):
            samples = per_width[width]
            median = statistics.median(samples)
            widths.append(
                {
                    "width": width,
                    "per_request_ms": round(median, 4),
                    "run_ms": round(median * width, 4),
                    "speedup_over_width_1": round(baseline / median, 4),
                    "spread_percent": round(
                        100.0 * (max(samples) - min(samples)) / median if median else 0.0, 2
                    ),
                    "passes": len(samples),
                }
            )
        LOGGER.info(
            "%s: %s control %.3f ms against a recorded %.3f ms; own width-1 %.3f ms "
            "against %s; %d pass(es) kept, %d discarded",
            variant,
            control_variant,
            statistics.median(controls),
            control_recorded,
            baseline,
            f"{recorded:.3f} ms" if recorded is not None else "nothing recorded",
            kept,
            discarded,
        )
        for row in widths:
            LOGGER.info(
                "    width %2d: %7.3f ms/request  (%8.3f ms/run)  %.3fx  spread %.1f%%",
                row["width"],
                row["per_request_ms"],
                row["run_ms"],
                row["speedup_over_width_1"],
                row["spread_percent"],
            )
        results.append(
            {
                "variant": variant,
                "recorded_service_time_ms": recorded,
                "control_variant": control_variant,
                "control_recorded_ms": control_recorded,
                "control_measured_ms": round(statistics.median(controls), 4),
                "own_width_1_ms": round(baseline, 4),
                "passes_kept": kept,
                "passes_discarded": discarded,
                "widths": widths,
            }
        )

    payload = {
        "measurement": "per-request ms of one Session::Run at several encoder batch widths",
        "sequence_length": SEQUENCE_LENGTH,
        "intra_op_threads": 1,
        "gated": True,
        "gate": (
            f"{GATE_VARIANT} at width 1, every pass, against the "
            f"configs/serving.yaml service_time_ms, within {100.0 * args.tolerance:.0f}%. "
            "Logically independent of the treatment: at width 1 there is no batching, so "
            "it cannot move because batching worked or failed. One control session serves "
            "every variant, including the INT8 ones the config records no service time for."
        ),
        "note": (
            "uniform sequence length, so no arm pays more padding than another. The "
            "padding a ragged workload adds is in results/encoder_batching.json and the "
            "two multiply."
        ),
        "variants": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    LOGGER.info("Wrote %s", args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
