"""A synthetic decoder graph, for testing the KV cache without exporting a model.

The block allocator and the prefill/decode split have to be tested somewhere CI can
reach, and CI cannot run `scripts/export_decoder.py` at all: it resolves optimum
2.2.0, which moved the ONNX exporter out to `optimum-onnx`, so
`optimum.exporters.onnx` does not exist there. Nothing in CI invokes the export and
nothing should start to.

So the graph is built here instead, with the same interface an
optimum-exported decoder declares -- `input_ids`, `past_key_values.{i}.{key,value}`,
`attention_mask` and `position_ids` in, `logits` and `present.{i}.{key,value}` out --
at a size that runs in microseconds.

Four properties make it a real test rather than a shape-checker:

- **Every cached position reaches the logits.** Each layer reduces its whole
  `present` tensor into the hidden state, so a gather that drops, duplicates or
  misplaces any token position changes the output. That reduction is deliberately
  *not* masked, which is what makes padding left in a reused staging buffer visible
  as a wrong answer rather than absorbed.
- **Layers and the two halves are not interchangeable.** Each layer scales its keys
  and values by a distinct constant, so reading layer 4's slab where layer 3's was
  meant, or a value where a key belonged, is visible rather than silently plausible.
- **`position_ids` and `attention_mask` are wired in.** A decode step that offset
  its positions wrongly, or sized its mask to the wrong total, changes the logits
  instead of being ignored.
- **The mask reaches the cache as well as the logits.** A second reduction weights
  `present` by the mask along the token axis. The plain mask term above only sees a
  row's mask *weight*, so it cannot tell a right-padded mask from a left-padded one
  of the same weight -- which is exactly the mistake a batched row invites, since
  its real tokens sit at [0, len) while a careless implementation masks [max-len,
  max). Weighting the cache by the mask makes the two differ.

Both reductions are needed, and neither is redundant. Measured against a padded
batch, the unmasked one alone misses a same-weight mask in the wrong columns, and
the masked one alone misses padding leaking out of the buffer, because masked
garbage multiplies to zero. Together they catch all four failures: leaked padding, a
mask of the wrong weight, a mask of the right weight in the wrong place, and a row
whose KV landed at the wrong offset. Neither makes a cache entry depend on what
follows it, so chunk-invariant prefill stays testable -- the mask is applied to the
reduction, never to the `present` the cache is scattered from.

`build_decoder_graph` also produces the malformed variants the rejection tests need:
a graph with no cache in its signature, one whose cache dimensions are dynamic, one
whose cache is float64, one missing a `present` output, and one that concatenates
its cache in the wrong order so the present-prefix invariant fails.
"""

from __future__ import annotations

from pathlib import Path

import pytest

LAYERS = 3
KV_HEADS = 2
HEAD_DIM = 4
# Wide enough that tests can pick token ids freely; the embedding Gather rejects an
# id at or above this, which is a distraction rather than a finding.
VOCAB = 64


def build_decoder_graph(
    path: Path,
    *,
    layers: int = LAYERS,
    kv_heads: int = KV_HEADS,
    head_dim: int = HEAD_DIM,
    vocab: int = VOCAB,
    include_past: bool = True,
    static_kv_dims: bool = True,
    double_cache: bool = False,
    omit_present_for_layer: int | None = None,
    reverse_present_concat: bool = False,
) -> None:
    """Write a decoder-with-past graph to *path*.

    Defaults produce a well-formed one. The keyword arguments each break a single
    assumption `DecoderSession` makes, so a rejection test can name which.
    """
    import numpy as np
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    hidden = kv_heads * head_dim
    cache_type = TensorProto.DOUBLE if double_cache else TensorProto.FLOAT
    rng = np.random.default_rng(20260730)

    inputs = [
        helper.make_tensor_value_info("input_ids", TensorProto.INT64, ["batch", "sequence"]),
    ]
    if include_past:
        past_dims = (
            ["batch", kv_heads, "past", head_dim]
            if static_kv_dims
            else ["batch", "kv_heads", "past", "head_dim"]
        )
        for layer in range(layers):
            for kind in ("key", "value"):
                inputs.append(
                    helper.make_tensor_value_info(
                        f"past_key_values.{layer}.{kind}", cache_type, list(past_dims)
                    )
                )
    inputs.append(
        helper.make_tensor_value_info("attention_mask", TensorProto.INT64, ["batch", "total"])
    )
    inputs.append(
        helper.make_tensor_value_info("position_ids", TensorProto.INT64, ["batch", "sequence"])
    )

    initializers = [
        numpy_helper.from_array(
            rng.standard_normal((vocab, hidden)).astype(np.float32), "embedding"
        ),
        numpy_helper.from_array(
            rng.standard_normal((hidden, vocab)).astype(np.float32), "projection"
        ),
        numpy_helper.from_array(np.array([2], dtype=np.int64), "axis_2"),
        numpy_helper.from_array(np.array([1], dtype=np.int64), "axis_1"),
        numpy_helper.from_array(
            np.array([0, 0, kv_heads, head_dim], dtype=np.int64), "shape_heads"
        ),
        numpy_helper.from_array(np.array([0, 1, -1], dtype=np.int64), "shape_summary"),
        # [batch, total] -> [batch, 1, total, 1], so the mask broadcasts over a
        # present tensor's heads and head dimension.
        numpy_helper.from_array(np.array([0, 1, -1, 1], dtype=np.int64), "shape_mask4"),
    ]

    nodes = [
        helper.make_node("Gather", ["embedding", "input_ids"], ["embedded"], axis=0),
        # position_ids folded in, so a decode step that mis-advances its positions
        # changes the logits rather than being ignored.
        helper.make_node("Cast", ["position_ids"], ["positions_f"], to=TensorProto.FLOAT),
        helper.make_node("Unsqueeze", ["positions_f", "axis_2"], ["positions_3d"]),
        # What the cache is derived from. Deliberately a function of this token and
        # this position only, which is the invariance a causal decoder has: what gets
        # cached for position p does not depend on how many positions follow it. A
        # fixture that broke it could not test chunked prefill, because splitting the
        # prompt would legitimately change the cache.
        helper.make_node("Add", ["embedded", "positions_3d"], ["hidden"]),
        # [batch, sequence, hidden] -> [batch, kv_heads, sequence, head_dim], the
        # layout the cache is stored and fed in.
        helper.make_node("Reshape", ["hidden", "shape_heads"], ["hidden_heads"]),
        helper.make_node("Transpose", ["hidden_heads"], ["new_kv"], perm=[0, 2, 1, 3]),
        # attention_mask reaches the logits but not the cache, so a step that sized
        # its mask to the wrong total is caught without making the cache
        # order-dependent.
        helper.make_node("Cast", ["attention_mask"], ["mask_f"], to=TensorProto.FLOAT),
        helper.make_node("ReduceSum", ["mask_f", "axis_1"], ["mask_width"], keepdims=1),
        helper.make_node("Unsqueeze", ["mask_width", "axis_2"], ["mask_width_3d"]),
        helper.make_node("Add", ["hidden", "mask_width_3d"], ["stated"]),
    ]

    if double_cache:
        nodes.append(helper.make_node("Cast", ["new_kv"], ["new_kv_cache"], to=cache_type))
    else:
        nodes.append(helper.make_node("Identity", ["new_kv"], ["new_kv_cache"]))

    # The mask again, this time shaped to weight a present tensor along the token
    # axis. In the cache's own dtype, so the Mul below does not mix types.
    nodes.append(helper.make_node("Reshape", ["mask_f", "shape_mask4"], ["mask_4d_f"]))
    if double_cache:
        nodes.append(helper.make_node("Cast", ["mask_4d_f"], ["mask_4d"], to=cache_type))
    else:
        nodes.append(helper.make_node("Identity", ["mask_4d_f"], ["mask_4d"]))

    outputs = [
        helper.make_tensor_value_info("logits", TensorProto.FLOAT, ["batch", "sequence", vocab])
    ]
    accumulated = "stated"

    for layer in range(layers):
        for index, kind in enumerate(("key", "value")):
            # Distinct per (layer, kind), so gathering the wrong slab is visible.
            scale = 1.0 + 0.25 * layer + 0.125 * index
            scale_name = f"scale_{layer}_{kind}"
            initializers.append(
                numpy_helper.from_array(
                    np.array(scale, dtype=np.float64 if double_cache else np.float32), scale_name
                )
            )
            nodes.append(
                helper.make_node("Mul", ["new_kv_cache", scale_name], [f"fresh.{layer}.{kind}"])
            )

            present = f"present.{layer}.{kind}"
            if include_past:
                # Reversed concatenation still produces the right shape, so the graph
                # runs; what it breaks is the invariant that the present tensor begins
                # with the past it was handed.
                order = (
                    [f"fresh.{layer}.{kind}", f"past_key_values.{layer}.{kind}"]
                    if reverse_present_concat
                    else [f"past_key_values.{layer}.{kind}", f"fresh.{layer}.{kind}"]
                )
                nodes.append(helper.make_node("Concat", order, [present], axis=2))
            else:
                nodes.append(helper.make_node("Identity", [f"fresh.{layer}.{kind}"], [present]))

            if omit_present_for_layer != layer:
                outputs.append(
                    helper.make_tensor_value_info(
                        present, cache_type, ["batch", kv_heads, "total", head_dim]
                    )
                )

            # Reduce the whole cache into the hidden state, so any misplaced token
            # position anywhere in the cache reaches the logits.
            reduced = f"reduced.{layer}.{kind}"
            nodes.append(helper.make_node("ReduceSum", [present, "axis_2"], [reduced], keepdims=0))
            if double_cache:
                nodes.append(
                    helper.make_node("Cast", [reduced], [reduced + ".f"], to=TensorProto.FLOAT)
                )
                reduced = reduced + ".f"
            flat = f"summary.{layer}.{kind}"
            nodes.append(helper.make_node("Reshape", [reduced, "shape_summary"], [flat]))
            merged = f"accumulated.{layer}.{kind}"
            nodes.append(helper.make_node("Add", [accumulated, flat], [merged]))
            accumulated = merged

            # The same reduction, weighted by the mask along the token axis. Applied
            # to a copy for the reduce only -- `present` itself is untouched, so what
            # gets cached for a position still does not depend on the mask, and
            # chunked prefill stays invariant.
            masked = f"masked.{layer}.{kind}"
            nodes.append(helper.make_node("Mul", [present, "mask_4d"], [masked]))
            mreduced = f"mreduced.{layer}.{kind}"
            nodes.append(helper.make_node("ReduceSum", [masked, "axis_2"], [mreduced], keepdims=0))
            if double_cache:
                nodes.append(
                    helper.make_node("Cast", [mreduced], [mreduced + ".f"], to=TensorProto.FLOAT)
                )
                mreduced = mreduced + ".f"
            mflat = f"msummary.{layer}.{kind}"
            nodes.append(helper.make_node("Reshape", [mreduced, "shape_summary"], [mflat]))
            mmerged = f"maccumulated.{layer}.{kind}"
            nodes.append(helper.make_node("Add", [accumulated, mflat], [mmerged]))
            accumulated = mmerged

    nodes.append(helper.make_node("MatMul", [accumulated, "projection"], ["logits"]))

    graph = helper.make_graph(nodes, "synthetic_decoder", inputs, outputs, initializer=initializers)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 14)], ir_version=8)
    onnx.checker.check_model(model, full_check=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(path))


@pytest.fixture(scope="session")
def decoder_graph(tmp_path_factory) -> Path:
    """A well-formed synthetic decoder graph, built once per session."""
    pytest.importorskip("onnx", reason="onnx is required to build the fixture graph")
    path = tmp_path_factory.mktemp("decoder") / "synthetic.onnx"
    build_decoder_graph(path)
    return path


@pytest.fixture(scope="session")
def gpt2_graph() -> Path:
    """The exported GPT-2 decoder, when it happens to be on disk.

    Skipped rather than exported: the export needs optimum-onnx, torch and a model
    download, none of which belong in a test. Local runs that have it get the real
    thing; CI gets the synthetic graph and the same assertions.
    """
    path = Path("models/onnx/decoder_gpt2_fp32/model.onnx")
    if not path.exists():
        pytest.skip(f"{path} is absent; run python scripts/export_decoder.py to measure on GPT-2")
    return path
