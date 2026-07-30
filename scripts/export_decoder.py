"""Export a decoder-only model to ONNX with a KV cache, at several precisions.

The serving work that follows this script needs a decoder, because a KV cache and
continuous batching are meaningless for an encoder that runs once per request.
GPT-2 124M is the first target: small enough to iterate on and large enough that
the cache is not a rounding error.

Exports `text-generation-with-past`, so the graph carries its KV cache in its
signature rather than hiding it. For GPT-2 that is 27 inputs and 25 outputs:

    input_ids                 [batch, sequence]
    past_key_values.{i}.key   [batch, 12, past, 64]     i = 0..11
    past_key_values.{i}.value [batch, 12, past, 64]
    attention_mask            [batch, past + sequence]
    position_ids              [batch, sequence]
  ->
    logits                    [batch, sequence, 50257]
    present.{i}.key           [batch, 12, past + sequence, 64]
    present.{i}.value         [batch, 12, past + sequence, 64]

Worth reading that carefully before building on it. ONNX Runtime allocates the
`present` tensors itself, sized `past + sequence`, so a caller cannot hand this
graph a block table and have attention read scattered pages. Paging over a stock
exported decoder can only be a host-side block allocator that gathers a
sequence's blocks into these tensors before each run. That is what the KV cache
work does, and it is why it is called a block allocator rather than paged
attention.

Precisions:

    fp32   the exported graph as-is
    int8   dynamic quantisation of MatMul weights
    int4   block-wise weight-only quantisation via MatMulNBitsQuantizer

Each variant is measured for size and for WikiText-2 perplexity, reported as a
delta against FP32. Perplexity is scored through the same `RuntimeClient` the
server dispatches to, for the reason given in `profile_variants.py`: a number
measured beside the serving path is not a number about the serving path.

Usage:
    python scripts/export_decoder.py
    python scripts/export_decoder.py --precisions fp32 int4
    python scripts/export_decoder.py --quick
"""

from __future__ import annotations

import argparse
import functools
import json
import platform
import shutil
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from anytime_serving.serving.onnx_runtime import (
    InferenceRequest,
    RuntimeClient,
    extension_available,
)
from anytime_serving.utils.logger import get_logger

LOGGER = get_logger("scripts.export_decoder")

PRECISIONS = ("fp32", "int8", "int4")
DEFAULT_MODEL = "gpt2"
DEFAULT_OPSET = 14
# GPT-2's trained context. Perplexity is only comparable between variants when
# every variant is scored over the same windows of the same length.
DEFAULT_CONTEXT = 1024
DEFAULT_WINDOWS = 32
QUICK_WINDOWS = 4
# Weight-only INT4 groups this many values per scale. 32 is the common default and
# trades size against accuracy; larger blocks are smaller and blunter.
DEFAULT_INT4_BLOCK_SIZE = 32
# Above this, ONNX cannot hold the graph in a single protobuf and weights have to
# spill to a sidecar file. GPT-2 FP32 is about 0.65 GB and stays under it.
EXTERNAL_DATA_THRESHOLD_BYTES = 1_800_000_000


@dataclass
class DecoderVariant:
    """One exported precision of one decoder."""

    name: str
    model: str
    precision: str
    graph: str
    size_mb: float
    size_ratio_vs_fp32: float
    perplexity: float
    perplexity_delta: float
    perplexity_tokens: int
    context_length: int
    windows: int
    prefill_p50_ms: float
    layers: int
    kv_heads: int
    head_dim: int
    kv_bytes_per_token: int


def apply_partial_descriptor_shim() -> int:
    """Make optimum's decoder export configs work on Python 3.14.

    CPython 3.14 gave ``functools.partial`` a ``__get__``, so a partial held as a
    class attribute is now a method descriptor and binds the instance as its first
    positional argument. optimum stores ``NORMALIZED_CONFIG_CLASS`` that way for
    every model whose config needs renamed fields, and constructs it as
    ``self.NORMALIZED_CONFIG_CLASS(self._config)``. On 3.14 that call becomes
    ``NormalizedConfig(self, config, allow_new=False, ...)`` and fails with
    "got multiple values for argument 'allow_new'".

    Encoder configs are unaffected because they name a plain class rather than a
    partial, which is why the existing `export_onnx.py` still works and this is
    only visible on the decoder path. 42 of optimum's 176 ONNX config classes use
    a partial, GPT-2 among them.

    Wrapping each one in ``staticmethod`` suppresses the binding and restores the
    pre-3.14 call, without changing what is called or with what. A no-op below
    3.14. Returns the number of classes patched.
    """
    if sys.version_info < (3, 14):
        return 0

    from optimum.exporters.onnx import model_configs

    patched = 0
    for name in dir(model_configs):
        candidate = getattr(model_configs, name)
        if not isinstance(candidate, type):
            continue
        # Only classes declaring it themselves, so a subclass is not patched twice
        # through its parent.
        declared = vars(candidate).get("NORMALIZED_CONFIG_CLASS")
        if isinstance(declared, functools.partial):
            candidate.NORMALIZED_CONFIG_CLASS = staticmethod(declared)
            patched += 1
    return patched


def _graph_path(directory: Path) -> Path:
    graphs = sorted(directory.glob("*.onnx"))
    if not graphs:
        raise SystemExit(f"no .onnx graph in {directory}")
    return graphs[0]


def _graph_bytes(graph: Path) -> int:
    """Size of the graph including any external weight sidecar."""
    total = graph.stat().st_size
    for sidecar in graph.parent.glob(f"{graph.name}*data*"):
        if sidecar != graph:
            total += sidecar.stat().st_size
    return total


def export_fp32(model_id: str, out_dir: Path, opset: int) -> None:
    """Export the FP32 graph with past_key_values in its signature."""
    from optimum.exporters.onnx import main_export

    patched = apply_partial_descriptor_shim()
    if patched:
        LOGGER.info(
            "Applied the Python %d.%d functools.partial shim to %d optimum config "
            "class(es); see apply_partial_descriptor_shim",
            sys.version_info[0],
            sys.version_info[1],
            patched,
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    main_export(
        model_id,
        output=str(out_dir),
        task="text-generation-with-past",
        opset=opset,
    )


def find_output_projection(model) -> list[str]:
    """Names of the nodes that produce the logits.

    The output projection is left in float by both quantisers below. It maps the
    hidden state onto 50257 vocabulary entries and its error lands directly on the
    distribution being scored, so it is the one weight in a decoder where low
    precision is most expensive. Measured on GPT-2: quantising it to symmetric
    4-bit and nothing else took perplexity from 26.8 to 1265.
    """
    logit_outputs = {out.name for out in model.graph.output if out.name == "logits"}
    if not logit_outputs and model.graph.output:
        logit_outputs = {model.graph.output[0].name}
    return [n.name for n in model.graph.node if set(n.output) & logit_outputs and n.name]


def rewrite_gemm_as_matmul(model) -> int:
    """Rewrite ``Gemm(A, B, C)`` into ``Add(MatMul(A, B), C)``.

    Needed because ``MatMulNBitsQuantizer`` only rewrites ``MatMul`` nodes, and
    GPT-2's linear layers are ``Conv1D`` in PyTorch, which exports as ``Gemm``.
    Without this pass the quantiser finds exactly one eligible node in the whole
    graph -- the output projection -- and INT4 both fails to shrink the model and
    destroys it. Models built from ``nn.Linear``, the Llama family among them,
    export as ``MatMul`` and do not need this.

    Only applied where it is exactly equivalent: alpha and beta both 1, neither
    input transposed, and three inputs present. Verified lossless on GPT-2, where
    the rewritten graph produces bitwise identical logits.

    Returns the number of nodes rewritten.
    """
    from onnx import helper

    rewritten = 0
    nodes = []
    for node in model.graph.node:
        floats = {a.name: a.f for a in node.attribute if a.type == 1}
        transposed = any(a.name in ("transA", "transB") and a.i for a in node.attribute)
        equivalent = (
            node.op_type == "Gemm"
            and len(node.input) == 3
            and abs(floats.get("alpha", 1.0) - 1.0) < 1e-12
            and abs(floats.get("beta", 1.0) - 1.0) < 1e-12
            and not transposed
        )
        if not equivalent:
            nodes.append(node)
            continue
        product = node.output[0] + "__matmul"
        base = node.name or product
        nodes.append(
            helper.make_node(
                "MatMul", [node.input[0], node.input[1]], [product], name=base + "_MatMul"
            )
        )
        nodes.append(
            helper.make_node("Add", [product, node.input[2]], [node.output[0]], name=base + "_Add")
        )
        rewritten += 1

    del model.graph.node[:]
    model.graph.node.extend(nodes)
    return rewritten


def quantize_int8(source: Path, out_dir: Path) -> Path:
    """Dynamic INT8 on the weight matrices, per output channel.

    Per-channel because per-tensor scales are too blunt here: one scale across a
    768x2304 projection measured 44.4 perplexity against 26.8, while per-channel
    costs 0.1. Gemm is included alongside MatMul because GPT-2's linear layers
    export as Gemm, and the output projection is excluded for the reason in
    find_output_projection.
    """
    import onnx
    from onnxruntime.quantization import QuantType, quantize_dynamic

    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / "model.onnx"
    model = onnx.load(str(source), load_external_data=False)
    quantize_dynamic(
        model_input=str(source),
        model_output=str(target),
        op_types_to_quantize=["MatMul", "Gemm"],
        per_channel=True,
        weight_type=QuantType.QInt8,
        nodes_to_exclude=find_output_projection(model),
        use_external_data_format=_graph_bytes(source) > EXTERNAL_DATA_THRESHOLD_BYTES,
        extra_options={"MatMulConstBOnly": True},
    )
    return target


def quantize_int4(source: Path, out_dir: Path, block_size: int) -> Path:
    """Block-wise weight-only INT4 via MatMulNBitsQuantizer.

    Weights only: activations stay float, so this trades size and memory bandwidth
    for a rounding error on the weights rather than restructuring the arithmetic.

    Asymmetric, because GPT-2's weight distributions are not centred and symmetric
    4-bit spends half its range on values that do not occur.
    """
    import onnx
    from onnxruntime.quantization.matmul_nbits_quantizer import (
        MatMulNBitsQuantizer,
        RTNWeightOnlyQuantConfig,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / "model.onnx"

    model = onnx.load(str(source))
    excluded = find_output_projection(model)
    rewritten = rewrite_gemm_as_matmul(model)
    if rewritten:
        LOGGER.info("  rewrote %d Gemm node(s) as MatMul + Add so INT4 can reach them", rewritten)

    quantizer = MatMulNBitsQuantizer(
        model,
        block_size=block_size,
        is_symmetric=False,
        nodes_to_exclude=excluded,
        algo_config=RTNWeightOnlyQuantConfig(),
    )
    quantizer.process()
    quantizer.model.save_model_to_file(
        str(target),
        use_external_data_format=_graph_bytes(source) > EXTERNAL_DATA_THRESHOLD_BYTES,
    )
    return target


def _copy_tokenizer(source_dir: Path, target_dir: Path) -> None:
    """Every variant needs its own tokenizer beside the graph."""
    for pattern in ("*.json", "*.txt", "*.model"):
        for path in source_dir.glob(pattern):
            if path.suffix == ".onnx":
                continue
            shutil.copy2(path, target_dir / path.name)


def _kv_geometry(config) -> tuple[int, int, int]:
    """Layers, KV heads, and head dimension, whatever the config calls them."""
    layers = getattr(config, "num_hidden_layers", None) or config.n_layer
    heads = getattr(config, "num_attention_heads", None) or config.n_head
    # Grouped-query models publish fewer KV heads than attention heads, and it is
    # the KV heads that size the cache.
    kv_heads = getattr(config, "num_key_value_heads", None) or heads
    hidden = getattr(config, "hidden_size", None) or config.n_embd
    return int(layers), int(kv_heads), int(hidden) // int(heads)


def _empty_past(layers: int, kv_heads: int, head_dim: int) -> dict[str, np.ndarray]:
    """Zero-length KV inputs, which is what a prefill from cold looks like."""
    empty = np.zeros((1, kv_heads, 0, head_dim), dtype=np.float32)
    feeds: dict[str, np.ndarray] = {}
    for layer in range(layers):
        feeds[f"past_key_values.{layer}.key"] = empty
        feeds[f"past_key_values.{layer}.value"] = empty
    return feeds


def _token_nll(logits: np.ndarray, targets: np.ndarray) -> tuple[float, int]:
    """Summed negative log-likelihood of `targets` under `logits`.

    Computed with the log-sum-exp shift rather than by exponentiating directly:
    GPT-2 logits reach into the tens, and exp of that in float32 loses the tail
    the perplexity is measuring.
    """
    shifted = logits[:-1].astype(np.float64)
    wanted = targets[1:]
    peak = shifted.max(axis=-1, keepdims=True)
    log_partition = peak[:, 0] + np.log(np.exp(shifted - peak).sum(axis=-1))
    chosen = shifted[np.arange(shifted.shape[0]), wanted]
    return float((log_partition - chosen).sum()), int(wanted.size)


def measure_perplexity(
    client: RuntimeClient,
    variant: str,
    token_ids: np.ndarray,
    *,
    context: int,
    windows: int,
    layers: int,
    kv_heads: int,
    head_dim: int,
) -> tuple[float, int, float]:
    """WikiText-2 perplexity through the serving path.

    Non-overlapping windows, so every token is predicted exactly once and no
    token is scored with a longer prefix than another. A sliding window with
    overlap would give a lower number and a less comparable one.

    Returns perplexity, the number of tokens scored, and the median prefill time.
    """
    past = _empty_past(layers, kv_heads, head_dim)
    total_nll = 0.0
    total_tokens = 0
    latencies: list[float] = []

    for index in range(windows):
        start = index * context
        window = token_ids[start : start + context]
        if window.size < 2:
            break
        feeds = {
            "input_ids": window.reshape(1, -1),
            "attention_mask": np.ones((1, window.size), dtype=np.int64),
            "position_ids": np.arange(window.size, dtype=np.int64).reshape(1, -1),
            **past,
        }
        response = client.infer(InferenceRequest(variant=variant, inputs=feeds))
        latencies.append(response.runtime_latency_ms)
        nll, count = _token_nll(response.logits[0], window)
        total_nll += nll
        total_tokens += count

    if total_tokens == 0:
        raise SystemExit("no tokens scored; the corpus is shorter than one window")
    return (
        float(np.exp(total_nll / total_tokens)),
        total_tokens,
        float(np.median(latencies)),
    )


def _wikitext_tokens(tokenizer, needed: int) -> np.ndarray:
    """Tokenised WikiText-2 test split, joined as the corpus is conventionally read."""
    from datasets import load_dataset

    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(line for line in dataset["text"] if line.strip())
    encoded = tokenizer(text, return_tensors="np")["input_ids"][0].astype(np.int64)
    if encoded.size < needed:
        LOGGER.warning(
            "corpus has %d tokens but %d were requested; scoring what exists",
            encoded.size,
            needed,
        )
    return encoded


def _cross_check_logits(
    graph: Path, feeds: dict[str, np.ndarray], through_engine: np.ndarray
) -> float:
    """Largest absolute logit difference against a separate ONNX Runtime session.

    One window is enough. This is the same discipline as `profile_variants.py`:
    measuring through the engine is only trustworthy if something independent
    agrees with it.
    """
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(
        str(graph), sess_options=options, providers=["CPUExecutionProvider"]
    )
    declared = {spec.name for spec in session.get_inputs()}
    reference = session.run(None, {k: v for k, v in feeds.items() if k in declared})[0]
    return float(np.max(np.abs(reference - through_engine)))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Hugging Face model id")
    parser.add_argument(
        "--precisions",
        nargs="+",
        default=list(PRECISIONS),
        choices=PRECISIONS,
        help="Which precisions to export",
    )
    parser.add_argument("--model-dir", type=Path, default=Path("models/onnx"))
    parser.add_argument("--output", type=Path, default=Path("results/decoder_profiles.json"))
    parser.add_argument("--opset", type=int, default=DEFAULT_OPSET)
    parser.add_argument("--context", type=int, default=DEFAULT_CONTEXT)
    parser.add_argument(
        "--windows",
        type=int,
        default=DEFAULT_WINDOWS,
        help=(
            "Non-overlapping context-length windows of WikiText-2 to score. The "
            "absolute perplexity depends on this; the delta between precisions, "
            f"which is the point, does not (default {DEFAULT_WINDOWS})"
        ),
    )
    parser.add_argument("--int4-block-size", type=int, default=DEFAULT_INT4_BLOCK_SIZE)
    parser.add_argument(
        "--quick",
        action="store_true",
        help=f"Score {QUICK_WINDOWS} windows instead of {DEFAULT_WINDOWS}",
    )
    parser.add_argument(
        "--skip-export",
        action="store_true",
        help="Measure graphs already on disk rather than re-exporting",
    )
    args = parser.parse_args()

    if not extension_available():
        raise SystemExit(
            "anytime_runtime is not available, so perplexity would be scored through "
            "the Python fallback rather than the path that serves traffic. Build the "
            "extension with:\n    pip install -e ."
        )

    windows = QUICK_WINDOWS if args.quick else args.windows
    short_name = args.model.split("/")[-1].replace(".", "_").lower()
    directories = {
        precision: args.model_dir / f"decoder_{short_name}_{precision}"
        for precision in args.precisions
    }

    # FP32 is the source every quantised variant is derived from and the reference
    # every perplexity delta is measured against, so it is always produced.
    fp32_dir = args.model_dir / f"decoder_{short_name}_fp32"
    if not args.skip_export:
        if not fp32_dir.exists() or not sorted(fp32_dir.glob("*.onnx")):
            LOGGER.info("Exporting %s to FP32 ONNX with past_key_values", args.model)
            started = time.perf_counter()
            export_fp32(args.model, fp32_dir, args.opset)
            LOGGER.info("  took %.0f s", time.perf_counter() - started)
        else:
            LOGGER.info("FP32 graph already present at %s", fp32_dir)

        fp32_graph = _graph_path(fp32_dir)
        for precision in args.precisions:
            if precision == "fp32":
                continue
            target_dir = directories[precision]
            if target_dir.exists() and sorted(target_dir.glob("*.onnx")):
                LOGGER.info("%s graph already present at %s", precision, target_dir)
                continue
            LOGGER.info("Quantising to %s", precision)
            started = time.perf_counter()
            if precision == "int8":
                quantize_int8(fp32_graph, target_dir)
            else:
                quantize_int4(fp32_graph, target_dir, args.int4_block_size)
            _copy_tokenizer(fp32_dir, target_dir)
            LOGGER.info("  took %.0f s", time.perf_counter() - started)

    from transformers import AutoConfig, AutoTokenizer

    config = AutoConfig.from_pretrained(args.model)
    layers, kv_heads, head_dim = _kv_geometry(config)
    kv_bytes_per_token = 2 * layers * kv_heads * head_dim * 4
    LOGGER.info(
        "KV geometry: %d layers, %d kv heads, head dim %d -> %.1f KiB per token (fp32)",
        layers,
        kv_heads,
        head_dim,
        kv_bytes_per_token / 1024,
    )

    tokenizer = AutoTokenizer.from_pretrained(fp32_dir)
    token_ids = _wikitext_tokens(tokenizer, windows * args.context)

    graphs = {p: _graph_path(directories[p]) for p in args.precisions}
    fp32_bytes = _graph_bytes(graphs["fp32"]) if "fp32" in graphs else 0

    # One client holding every precision, which is how the server loads variants.
    client = RuntimeClient({p: graphs[p] for p in args.precisions})
    LOGGER.info("Scoring through the %s backend", client.backend_name)

    measurements: list[DecoderVariant] = []
    try:
        for precision in args.precisions:
            LOGGER.info("Scoring %s over %d windows of %d tokens", precision, windows, args.context)
            perplexity, tokens, prefill_ms = measure_perplexity(
                client,
                precision,
                token_ids,
                context=args.context,
                windows=windows,
                layers=layers,
                kv_heads=kv_heads,
                head_dim=head_dim,
            )
            size_bytes = _graph_bytes(graphs[precision])
            measurements.append(
                DecoderVariant(
                    name=f"{short_name}_{precision}",
                    model=short_name,
                    precision=precision,
                    graph=graphs[precision].name,
                    size_mb=round(size_bytes / 1e6, 1),
                    size_ratio_vs_fp32=round(size_bytes / fp32_bytes, 4) if fp32_bytes else 1.0,
                    perplexity=round(perplexity, 4),
                    perplexity_delta=0.0,
                    perplexity_tokens=tokens,
                    context_length=args.context,
                    windows=windows,
                    prefill_p50_ms=round(prefill_ms, 2),
                    layers=layers,
                    kv_heads=kv_heads,
                    head_dim=head_dim,
                    kv_bytes_per_token=kv_bytes_per_token,
                )
            )
            LOGGER.info(
                "  perplexity=%.4f (n=%d tokens) size=%.1f MB prefill p50=%.0f ms",
                perplexity,
                tokens,
                size_bytes / 1e6,
                prefill_ms,
            )

        # The cross-check runs on FP32 only: it establishes that the engine and an
        # independent session agree on this graph shape, which is what the other
        # precisions inherit.
        max_logit_diff = None
        if "fp32" in args.precisions:
            window = token_ids[: args.context]
            feeds = {
                "input_ids": window.reshape(1, -1),
                "attention_mask": np.ones((1, window.size), dtype=np.int64),
                "position_ids": np.arange(window.size, dtype=np.int64).reshape(1, -1),
                **_empty_past(layers, kv_heads, head_dim),
            }
            through_engine = client.infer(InferenceRequest(variant="fp32", inputs=feeds)).logits
            max_logit_diff = _cross_check_logits(graphs["fp32"], feeds, through_engine)
            LOGGER.info(
                "Engine vs separate session on FP32 logits: max abs diff %.3e", max_logit_diff
            )
            if max_logit_diff > 0.0:
                raise SystemExit(
                    f"the engine and a separate ONNX Runtime session disagree on the "
                    f"FP32 logits by {max_logit_diff:.3e}. They run the same library on "
                    f"the same graph and should be bitwise identical; investigate "
                    f"before trusting any perplexity above."
                )
    finally:
        client.close()

    reference = next((m for m in measurements if m.precision == "fp32"), None)
    if reference is not None:
        for measurement in measurements:
            measurement.perplexity_delta = round(measurement.perplexity - reference.perplexity, 4)

    payload = {
        "host": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "backend": "extension",
        },
        "model": args.model,
        "task": "text-generation-with-past",
        "opset": args.opset,
        "perplexity_dataset": "wikitext-2-raw-v1 test",
        "context_length": args.context,
        "windows": windows,
        "int4_block_size": args.int4_block_size,
        "engine_vs_session_max_logit_diff": max_logit_diff,
        "variants": [asdict(m) for m in measurements],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    LOGGER.info("Wrote %s", args.output)

    LOGGER.info("")
    LOGGER.info(
        "%-10s %10s %8s %12s %10s %12s",
        "precision",
        "size(MB)",
        "vs fp32",
        "perplexity",
        "delta",
        "prefill p50",
    )
    for measurement in measurements:
        LOGGER.info(
            "%-10s %10.1f %8.3f %12.4f %10.4f %9.0f ms",
            measurement.precision,
            measurement.size_mb,
            measurement.size_ratio_vs_fp32,
            measurement.perplexity,
            measurement.perplexity_delta,
            measurement.prefill_p50_ms,
        )
    LOGGER.info("")
    LOGGER.info(
        "Perplexity is over %d tokens of WikiText-2 test in %d non-overlapping "
        "windows of %d. The absolute value depends on that choice; the delta "
        "between precisions is what the quantisation costs.",
        measurements[0].perplexity_tokens if measurements else 0,
        windows,
        args.context,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
