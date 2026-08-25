"""What encoder batching costs and changes, in counts rather than seconds.

The encoder path had no batching, so every number in the variant frontier and the load
sweep describes batch size 1. `serving/batching.py` adds it. This measures what that
does to the *answers* and to the *work*, both of which are properties of the workload
and the graph rather than of how busy the host is -- so unlike a throughput claim, none
of this needs a quiet machine and none of it is gated.

Three questions, and the third is the one that produced a finding.

1. **How much of a batched tensor is padding?** A batched Run computes every row at the
   longest row's length, so batching buys amortisation and pays for it in wasted
   arithmetic. This counts the trade over the real SST-2 length distribution.

   It also puts a number on something that predates batching: the load sweep and the
   variant profiler tokenise with `padding="max_length"` at 128, while SST-2's median
   sentence is 24 tokens. **80% of every tensor the encoder benchmark has ever run is
   padding**, and that is a property of the measurement, not of batching.

2. **How many Runs does a batch width save?** Ceiling division, reported so the
   amortisation ceiling is explicit: nothing about batching can be worth more than the
   per-Run work it removes.

3. **Does a request's answer depend on who shares its batch?** For FP32, no -- to
   1.1e-05 of a logit, and it is the control here: logically independent of the
   quantisation question, so it says the harness is sound.

   That 1.1e-05 is reduction order rather than noise. A padded row makes the pooled
   sum longer, a vectorised reduction regroups its terms by lane, and float addition
   is not associative -- so how far the answer moves depends on which kernel ran, and
   differs by architecture. `tests/test_batching.py` bounds a synthetic case of it
   rather than asserting equality, having first asserted equality and failed CI.

   For INT8, **yes**. Both quantised variants carry 50 `DynamicQuantizeLinear` nodes,
   which compute an activation scale at runtime from the tensor actually fed. Batching
   changes that tensor, so it changes every row's scale. Measured over all 872
   validation sentences it moves a logit by up to 1.2 and flips 0.2-0.7% of
   predictions.

   **But the same is true of padding alone at batch 1**, which flips 4 of 872 -- so the
   effect belongs to dynamic quantisation meeting a padded tensor, not to batching, and
   the shipped `padding="max_length"` benchmark already sits inside it. Accuracy moves
   by at most +-0.5pp and does not systematically fall. That is why this is reported as
   a determinism property rather than an accuracy regression.

The FP32 accuracies this reproduces are 91.06% for DistilBERT and 90.14% for MiniLM,
which are the numbers already in `configs/serving.yaml`. That agreement is not
decoration: it says this script's tokenisation and feed match the profiler that wrote
the config, so the INT8 result beside it is measured the same way.

Needs `models/` and the `research` extra (`datasets`, `transformers`), neither of which
is committed. Runs single-threaded on purpose: the counts do not depend on it and a
pinned session makes the run reproducible.

Usage:
    python scripts/count_encoder_batching.py
    python scripts/count_encoder_batching.py --widths 1 2 4 8 --limit 200
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from anytime_serving.utils.logger import get_logger  # noqa: E402

LOGGER = get_logger("scripts.count_encoder_batching")

# What the load sweep and the variant profiler tokenise to. Repeated here rather than
# imported so this script does not drag their dependencies in, and asserted against
# them by tests/test_count_encoder_batching.py.
BENCHMARK_SEQUENCE_LENGTH = 128

DEFAULT_WIDTHS = (1, 2, 4, 8, 16, 32)

VARIANT_PATHS = {
    "distilbert_fp32": Path("models/onnx/text_distilbert_fp32/model.onnx"),
    "distilbert_int8": Path("models/onnx/text_distilbert_int8/model_quantized.onnx"),
    "distilbert_int8_static": Path("models/onnx/text_distilbert_int8_static/model_quantized.onnx"),
    "distilbert_int8_static_percentile": Path(
        "models/onnx/text_distilbert_int8_static_percentile/model_quantized.onnx"
    ),
    "minilm_fp32": Path("models/onnx/text_minilm_fp32/model.onnx"),
    "minilm_int8": Path("models/onnx/text_minilm_int8/model_quantized.onnx"),
    "minilm_int8_static": Path("models/onnx/text_minilm_int8_static/model_quantized.onnx"),
    "minilm_int8_static_percentile": Path(
        "models/onnx/text_minilm_int8_static_percentile/model_quantized.onnx"
    ),
}

# The op types the census below reports. `DynamicQuantizeLinear` is the mechanism
# under test; the rest are there because counting zero of it proves nothing on its
# own -- an FP32 graph also has zero. What distinguishes a static INT8 graph from an
# unquantised one is that the weights are 8-bit and the scales are initialisers.
CENSUS_OPS = (
    "DynamicQuantizeLinear",
    "QuantizeLinear",
    "DequantizeLinear",
    "MatMulInteger",
    "QLinearMatMul",
    "MatMul",
    "Gemm",
)


def group_in_arrival_order(count: int, width: int) -> list[list[int]]:
    """Split `count` requests into contiguous batches of at most `width`.

    Arrival order rather than length-sorted. Sorting by length is what the decoder's
    `length_bucketing` does and it would shrink the padding below; measuring the
    unsorted case first is what says how much there is to shrink.
    """
    if width < 1:
        raise ValueError("width must be at least 1")
    return [list(range(at, min(at + width, count))) for at in range(0, count, width)]


def padding_fraction(lengths: Sequence[int], width: int) -> float:
    """Share of a batched tensor's token slots that hold padding.

    Each group is charged its longest row across every row, which is what ONNX Runtime
    requires: one rectangular tensor per Run.
    """
    groups = group_in_arrival_order(len(lengths), width)
    slots = sum(max(lengths[i] for i in g) * len(g) for g in groups)
    useful = sum(lengths)
    return 0.0 if not slots else 1.0 - useful / slots


def fixed_padding_fraction(lengths: Sequence[int], sequence_length: int) -> float:
    """Padding share when every row is padded to a fixed width, batched or not.

    This is what the committed encoder benchmarks do at `sequence_length = 128`, and it
    is the baseline any batched number here has to beat rather than be compared against
    zero.
    """
    if not lengths:
        return 0.0
    return 1.0 - sum(lengths) / (sequence_length * len(lengths))


def _tokenise(variant: str, sentences: Sequence[str]) -> list[dict[str, np.ndarray]]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(f"models/onnx/text_{variant}")
    feeds = []
    for sentence in sentences:
        encoded = tokenizer(sentence, return_tensors="np", return_token_type_ids=True)
        feeds.append(
            {name: np.ascontiguousarray(value.astype(np.int64)) for name, value in encoded.items()}
        )
    return feeds


def _assemble(
    feeds: Sequence[dict[str, np.ndarray]], group: Sequence[int], width: int
) -> dict[str, np.ndarray]:
    """Right-pad and stack one group, the way `batching.pad_along_axis1` does."""
    batched = {}
    for name in feeds[group[0]]:
        out = np.zeros((len(group), width), dtype=np.int64)
        for row, index in enumerate(group):
            source = feeds[index][name]
            out[row, : source.shape[1]] = source[0]
        batched[name] = out
    return batched


def _quantise_nodes(path: Path) -> int:
    """How many `DynamicQuantizeLinear` nodes the graph carries.

    The mechanism behind the INT8 result, counted rather than asserted: each one
    computes an activation scale from the tensor it is handed, so a batched or padded
    tensor gives a different scale to every row in it.
    """
    import onnx

    model = onnx.load(str(path), load_external_data=False)
    return sum(1 for node in model.graph.node if node.op_type == "DynamicQuantizeLinear")


def graph_census(path: Path) -> dict:
    """Op-type counts and weight element counts by dtype, for one graph.

    Reported so that "the static export has no `DynamicQuantizeLinear`" is a claim
    with a control attached. Zero of them is also true of the FP32 graph, so the
    census carries the two things that separate the cases: how many 8-bit weight
    elements the graph holds, and how many `QuantizeLinear`/`DequantizeLinear` pairs
    surround the operators that were quantised.

    `int8_weight_elements` is what says the two INT8 flavours quantised the same
    operators. They land within a rounding error of each other, and the FP32 graph
    reads zero.
    """
    import onnx

    model = onnx.load(str(path), load_external_data=False)
    counts = dict.fromkeys(CENSUS_OPS, 0)
    for node in model.graph.node:
        if node.op_type in counts:
            counts[node.op_type] += 1

    quantised_elements = 0
    float_elements = 0
    for initialiser in model.graph.initializer:
        elements = 1
        for dimension in initialiser.dims:
            elements *= dimension
        name = onnx.TensorProto.DataType.Name(initialiser.data_type)
        if name in ("INT8", "UINT8"):
            quantised_elements += elements
        elif name in ("FLOAT", "FLOAT16"):
            float_elements += elements

    return {
        "nodes": counts,
        "int8_weight_elements": quantised_elements,
        "float_weight_elements": float_elements,
        "graph_bytes": path.stat().st_size,
    }


def measure_variant(
    variant: str,
    path: Path,
    sentences: Sequence[str],
    labels: np.ndarray,
    widths: Sequence[int],
) -> dict:
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(
        str(path), sess_options=options, providers=["CPUExecutionProvider"]
    )
    declared = {spec.name for spec in session.get_inputs()}

    def run(feed: dict[str, np.ndarray]) -> np.ndarray:
        return session.run(None, {k: v for k, v in feed.items() if k in declared})[0]

    feeds = _tokenise(variant, sentences)
    lengths = [f["input_ids"].shape[1] for f in feeds]

    # The reference every arm is compared against: one request, its own true length,
    # nothing else in the tensor. The only feed with no padding in it at all.
    reference = np.concatenate([run(f) for f in feeds])
    reference_accuracy = float((reference.argmax(1) == labels).mean())

    def arm(label: str, groups: list[list[int]], width_of) -> dict:
        outputs = [_assemble(feeds, g, width_of(g)) for g in groups]
        logits = np.concatenate([run(f) for f in outputs])
        flips = int((logits.argmax(1) != reference.argmax(1)).sum())
        accuracy = float((logits.argmax(1) == labels).mean())
        # Per request rather than over the whole array, because a maximum cannot tell
        # one sentence sitting on a rounding boundary apart from every sentence
        # wobbling. `rows_moved` is what separates those two, and they have different
        # causes: a threshold crossing in a fixed quantisation grid against float
        # reassociation, which moves everything a little.
        per_row = np.abs(logits - reference).max(axis=1)
        moved = per_row > 0.0
        return {
            "arm": label,
            "runs": len(groups),
            "flips": flips,
            "requests": len(reference),
            "flip_percent": round(100.0 * flips / len(reference), 3),
            "accuracy": round(accuracy, 4),
            "accuracy_delta_pp": round(100.0 * (accuracy - reference_accuracy), 3),
            "max_abs_logit_delta": float(per_row.max()),
            "rows_moved": int(moved.sum()),
            "rows_moved_percent": round(100.0 * float(moved.mean()), 3),
            "median_abs_logit_delta_of_moved": (
                float(np.median(per_row[moved])) if moved.any() else 0.0
            ),
        }

    arms = [
        arm(
            f"batch 1, pad to {BENCHMARK_SEQUENCE_LENGTH} (what the benchmarks run)",
            group_in_arrival_order(len(feeds), 1),
            lambda g: BENCHMARK_SEQUENCE_LENGTH,
        )
    ]
    for width in widths:
        groups = group_in_arrival_order(len(feeds), width)
        arms.append(
            arm(
                f"batch {width}, pad to longest in batch",
                groups,
                lambda g: max(lengths[i] for i in g),
            )
        )

    return {
        "variant": variant,
        "dynamic_quantize_nodes": _quantise_nodes(path),
        "graph": graph_census(path),
        "reference": "batch 1, true length, no padding",
        "reference_accuracy": round(reference_accuracy, 4),
        "arms": arms,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--widths", type=int, nargs="+", default=list(DEFAULT_WIDTHS), help="batch widths to count"
    )
    parser.add_argument(
        "--limit", type=int, default=0, help="use only the first N validation sentences"
    )
    parser.add_argument(
        "--variants", nargs="+", default=sorted(VARIANT_PATHS), help="variants to measure"
    )
    parser.add_argument(
        "--output", type=Path, default=Path("results/encoder_batching.json"), help="where to write"
    )
    args = parser.parse_args()

    for variant in args.variants:
        if variant not in VARIANT_PATHS:
            parser.error(f"unknown variant {variant!r}; known: {sorted(VARIANT_PATHS)}")
    missing = [v for v in args.variants if not VARIANT_PATHS[v].exists()]
    if missing:
        LOGGER.error(
            "missing exported models for %s. models/ is not committed; run "
            "python scripts/export_onnx.py first.",
            ", ".join(missing),
        )
        return 1

    from datasets import load_dataset

    dataset = load_dataset("glue", "sst2", split="validation")
    sentences = list(dataset["sentence"])
    labels = np.array(dataset["label"])
    if args.limit:
        sentences, labels = sentences[: args.limit], labels[: args.limit]

    # Lengths are the tokeniser's, so they are per variant; DistilBERT and MiniLM
    # share a WordPiece vocabulary, so in practice these agree. Reported from the
    # first variant and asserted equal across them by the geometry block below.
    per_variant_lengths: dict[str, list[int]] = {}
    variants = []
    for variant in args.variants:
        LOGGER.info("counting %s over %d sentence(s)", variant, len(sentences))
        per_variant_lengths[variant] = [
            f["input_ids"].shape[1] for f in _tokenise(variant, sentences)
        ]
        variants.append(
            measure_variant(variant, VARIANT_PATHS[variant], sentences, labels, args.widths)
        )
        latest = variants[-1]
        LOGGER.info(
            "  %s: %d DynamicQuantizeLinear node(s), %.1fM 8-bit weight element(s), "
            "%.1f MB on disk, reference accuracy %.2f%%",
            variant,
            latest["dynamic_quantize_nodes"],
            latest["graph"]["int8_weight_elements"] / 1e6,
            latest["graph"]["graph_bytes"] / 1e6,
            100.0 * latest["reference_accuracy"],
        )
        for entry in latest["arms"]:
            LOGGER.info(
                "    %-44s flips %3d/%d (%.2f%%)  moved %3d (%.1f%%)  accuracy %+.2fpp  "
                "max|dlogit| %.2e",
                entry["arm"],
                entry["flips"],
                entry["requests"],
                entry["flip_percent"],
                entry["rows_moved"],
                entry["rows_moved_percent"],
                entry["accuracy_delta_pp"],
                entry["max_abs_logit_delta"],
            )

    lengths = per_variant_lengths[args.variants[0]]
    geometry = {
        "sentences": len(lengths),
        "token_length": {
            "min": int(min(lengths)),
            "median": int(np.median(lengths)),
            "mean": round(float(np.mean(lengths)), 2),
            "p95": int(np.percentile(lengths, 95)),
            "max": int(max(lengths)),
        },
        "benchmark_sequence_length": BENCHMARK_SEQUENCE_LENGTH,
        "fixed_padding_fraction": round(
            fixed_padding_fraction(lengths, BENCHMARK_SEQUENCE_LENGTH), 4
        ),
        "widths": [
            {
                "width": width,
                "runs": len(group_in_arrival_order(len(lengths), width)),
                "runs_saved_fraction": round(
                    1.0 - len(group_in_arrival_order(len(lengths), width)) / len(lengths), 4
                ),
                "padding_fraction": round(padding_fraction(lengths, width), 4),
            }
            for width in args.widths
        ],
        "tokenisers_agree": len({tuple(v) for v in per_variant_lengths.values()}) == 1,
    }
    LOGGER.info(
        "padding: %.1f%% at a fixed %d tokens, against %s when padded to the longest in each batch",
        100.0 * geometry["fixed_padding_fraction"],
        BENCHMARK_SEQUENCE_LENGTH,
        ", ".join(
            f"{100.0 * row['padding_fraction']:.1f}% at width {row['width']}"
            for row in geometry["widths"]
        ),
    )

    import onnxruntime as ort

    payload = {
        "measurement": "what encoder batching costs in padding and changes in answers",
        "dataset": "glue/sst2 validation",
        "gated": False,
        "host": {
            # Recorded rather than pinned. The counts here do not depend on the host,
            # but `max_abs_logit_delta` does: it is reduction order, the kernel that
            # runs follows the ONNX Runtime version, and pyproject.toml carries a
            # floor rather than a pin. See docs/runtime.md, "Why the floor is not a
            # pin".
            "onnxruntime": ort.__version__,
            "machine": platform.machine(),
            "python": platform.python_version(),
        },
        "note": (
            "counts and numerics only. No timing here, so a contended host cannot "
            "corrupt it; the throughput arm of encoder batching is a separate, gated "
            "measurement."
        ),
        "geometry": geometry,
        "variants": variants,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    LOGGER.info("Wrote %s", args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
