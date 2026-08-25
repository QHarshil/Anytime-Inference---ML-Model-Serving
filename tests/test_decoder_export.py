"""Guards on the graph surgery in `scripts/export_decoder.py`.

Two of the helpers there change or select parts of an ONNX graph, and a silent
mistake in either would not fail loudly -- it would quietly produce a model that
still runs and still emits plausible logits, just worse ones. That is exactly the
failure mode this project has already been burned by, so both are tested.

`rewrite_gemm_as_matmul` is the riskier one: it exists because
`MatMulNBitsQuantizer` only rewrites `MatMul`, and GPT-2's linear layers export as
`Gemm`. Without it, INT4 quantisation reached one node out of 49 and perplexity
went from 26.8 to 1265.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

onnx = pytest.importorskip("onnx", reason="onnx is required to build the fixture graphs")
ort = pytest.importorskip("onnxruntime", reason="onnxruntime is required to run them")

from onnx import TensorProto, helper, numpy_helper  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from export_decoder import (  # noqa: E402
    _kv_geometry,
    apply_partial_descriptor_shim,
    find_output_projection,
    kv_geometry_from_graph,
    rewrite_gemm_as_matmul,
)

RNG = np.random.default_rng(0)
WEIGHT = RNG.standard_normal((4, 6)).astype(np.float32)
BIAS = RNG.standard_normal((6,)).astype(np.float32)


def _gemm_graph(**attributes) -> onnx.ModelProto:
    """A single Gemm with constant weight and bias, plus the given attributes."""
    inp = helper.make_tensor_value_info("input", TensorProto.FLOAT, [None, 4])
    out = helper.make_tensor_value_info("logits", TensorProto.FLOAT, [None, 6])
    node = helper.make_node(
        "Gemm", ["input", "weight", "bias"], ["logits"], name="proj", **attributes
    )
    graph = helper.make_graph(
        [node],
        "gemm_only",
        [inp],
        [out],
        initializer=[
            numpy_helper.from_array(WEIGHT, "weight"),
            numpy_helper.from_array(BIAS, "bias"),
        ],
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 14)], ir_version=8)


def _run(model: onnx.ModelProto, data: np.ndarray) -> np.ndarray:
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    session = ort.InferenceSession(
        model.SerializeToString(), sess_options=options, providers=["CPUExecutionProvider"]
    )
    return session.run(None, {"input": data})[0]


def test_rewrite_is_bitwise_lossless():
    """The rewritten graph must compute exactly the same thing.

    Bitwise, not approximately: Add(MatMul(A, B), C) is the definition of
    Gemm(A, B, C) at alpha = beta = 1, so anything other than an exact match means
    the rewrite changed the arithmetic.
    """
    data = RNG.standard_normal((3, 4)).astype(np.float32)
    original = _gemm_graph()
    before = _run(original, data)

    rewritten = _gemm_graph()
    assert rewrite_gemm_as_matmul(rewritten) == 1
    onnx.checker.check_model(rewritten, full_check=False)
    after = _run(rewritten, data)

    np.testing.assert_array_equal(before, after)
    op_types = sorted(n.op_type for n in rewritten.graph.node)
    assert op_types == ["Add", "MatMul"]


@pytest.mark.parametrize(
    "attributes",
    [
        {"alpha": 2.0},
        {"beta": 0.5},
        {"transB": 1},
        {"transA": 1},
    ],
)
def test_rewrite_refuses_where_it_would_not_be_equivalent(attributes):
    """A Gemm that is not plain A @ B + C is left alone.

    Rewriting one of these would silently drop a scale factor or a transpose. The
    quantiser simply misses that node instead, which is recoverable; wrong
    arithmetic is not. The graph is never executed here, only inspected, so the
    shapes an alpha or a transpose would imply do not matter.
    """
    model = _gemm_graph(**attributes)
    assert rewrite_gemm_as_matmul(model) == 0
    assert [n.op_type for n in model.graph.node] == ["Gemm"]


def test_rewrite_leaves_a_two_input_gemm_alone():
    """Gemm without C has nothing to Add, so it is not eligible."""
    inp = helper.make_tensor_value_info("input", TensorProto.FLOAT, [None, 4])
    out = helper.make_tensor_value_info("logits", TensorProto.FLOAT, [None, 6])
    graph = helper.make_graph(
        [helper.make_node("Gemm", ["input", "weight"], ["logits"], name="proj")],
        "gemm_no_bias",
        [inp],
        [out],
        initializer=[numpy_helper.from_array(WEIGHT, "weight")],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 14)], ir_version=8)
    assert rewrite_gemm_as_matmul(model) == 0


def test_output_projection_is_the_node_producing_logits():
    """Both quantisers exclude this node, so selecting the wrong one is expensive."""
    model = _gemm_graph()
    assert find_output_projection(model) == ["proj"]


def test_output_projection_prefers_the_output_named_logits():
    """A graph with several outputs must still resolve to the logits producer.

    An exported decoder returns 24 present.* tensors alongside logits, and picking
    one of those would leave the output projection quantised.
    """
    inp = helper.make_tensor_value_info("input", TensorProto.FLOAT, [None, 4])
    logits = helper.make_tensor_value_info("logits", TensorProto.FLOAT, [None, 6])
    passthrough = helper.make_tensor_value_info("present.0.key", TensorProto.FLOAT, [None, 4])
    graph = helper.make_graph(
        [
            helper.make_node("Gemm", ["input", "weight", "bias"], ["logits"], name="proj"),
            helper.make_node("Identity", ["input"], ["present.0.key"], name="cache"),
        ],
        "two_outputs",
        [inp],
        # present.* first, so a naive "first output" choice would pick the wrong node.
        [passthrough, logits],
        initializer=[
            numpy_helper.from_array(WEIGHT, "weight"),
            numpy_helper.from_array(BIAS, "bias"),
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 14)], ir_version=8)
    assert find_output_projection(model) == ["proj"]


def test_partial_descriptor_shim_matches_the_interpreter():
    """The shim patches on 3.14 and does nothing below it.

    Also asserts the premise: that a ``functools.partial`` held as a class
    attribute binds the instance on 3.14 and not before. If CPython reverts that,
    this test says so rather than the shim silently becoming dead code.
    """
    import functools

    class Holder:
        FACTORY = functools.partial(lambda *args, **kwargs: (args, kwargs), flag=True)

    bound_args, _ = Holder().FACTORY("config")
    binds_self = len(bound_args) == 2

    assert binds_self == (sys.version_info >= (3, 14)), (
        "functools.partial descriptor behaviour changed; revisit "
        "apply_partial_descriptor_shim in scripts/export_decoder.py"
    )

    if sys.version_info < (3, 14):
        # Below 3.14 the shim returns before importing anything, so this half runs
        # everywhere.
        assert apply_partial_descriptor_shim() == 0
        return

    # On 3.14 the shim walks optimum's config classes, so it cannot run without
    # optimum -- which lives in the `research` extra, not `bench`. Skipping rather
    # than failing: this was a hard failure on a `pip install -e ".[bench]"` clone
    # under 3.14, and it was invisible everywhere else, because CI's matrix stops at
    # 3.13 and a development environment has optimum installed.
    pytest.importorskip(
        "optimum", reason="the 3.14 shim walks optimum's config classes; research extra"
    )
    # optimum declares a partial on every decoder config needing renamed fields; if it
    # stops doing so the shim is no longer needed.
    assert apply_partial_descriptor_shim() > 0


# -- KV geometry, read off the graph rather than off a config --------------------------
#
# `export_decoder.py` read this off the model config until a second model existed for
# the config to disagree with. The graph is what runs, so the graph is the authority;
# the config is the cross-check. `DecoderSession::derive_geometry` has said so on the
# C++ side since the arena was written, and the last test here is the one that keeps
# the Python mirror honest: it asserts the two implementations agree on the same file,
# so the mirror cannot drift into being a second opinion.


def test_geometry_comes_off_the_graphs_own_signature(tmp_path):
    """Grouped-query shape, so kv_heads cannot be confused with the head count."""
    from tests.conftest import build_decoder_graph

    path = tmp_path / "gqa.onnx"
    build_decoder_graph(path, layers=5, kv_heads=2, head_dim=8)
    assert kv_geometry_from_graph(path) == (5, 2, 8)


def test_a_graph_without_its_cache_in_the_signature_is_refused(tmp_path):
    from tests.conftest import build_decoder_graph

    path = tmp_path / "no_past.onnx"
    build_decoder_graph(path, include_past=False)
    with pytest.raises(SystemExit, match="with-past"):
        kv_geometry_from_graph(path)


def test_a_dynamic_kv_head_or_head_dim_is_refused(tmp_path):
    """Both size the block pool, so neither can be discovered per request."""
    from tests.conftest import build_decoder_graph

    path = tmp_path / "dynamic.onnx"
    build_decoder_graph(path, static_kv_dims=False)
    with pytest.raises(SystemExit, match="dynamic kv_heads or head_dim"):
        kv_geometry_from_graph(path)


@pytest.mark.parametrize(
    ("layers", "kv_heads", "head_dim"),
    [(3, 2, 4), (5, 1, 8), (2, 4, 16)],
)
def test_the_python_reader_agrees_with_the_c_plus_plus_one(tmp_path, layers, kv_heads, head_dim):
    """The mirror is asserted against the authority, not merely written to match it.

    `DecoderSession::derive_geometry` is what sizes the arena that actually holds the
    cache. If these two ever disagree, the numbers this script writes describe a
    different cache from the one the runtime allocates.
    """
    from anytime_serving.serving.onnx_runtime import extension_available, load_extension

    if not extension_available():  # pragma: no cover - present in this environment
        pytest.skip("anytime_runtime is not built; the C++ reader is what is compared")
    from tests.conftest import build_decoder_graph

    path = tmp_path / "mirror.onnx"
    build_decoder_graph(path, layers=layers, kv_heads=kv_heads, head_dim=head_dim)

    session = load_extension().DecoderSession(str(path), 8, 4)
    native = session.geometry
    assert kv_geometry_from_graph(path) == (native.layers, native.kv_heads, native.head_dim)
    assert (native.layers, native.kv_heads, native.head_dim) == (layers, kv_heads, head_dim)


def test_the_config_reader_still_agrees_on_a_model_it_was_written_for():
    """`_kv_geometry` is kept as the cross-check, so it has to keep working.

    GPT-2 names none of its geometry the way a Llama config does -- `n_layer`,
    `n_head`, `n_embd` against `num_hidden_layers`, `num_attention_heads`,
    `hidden_size` -- which is what the fallback chain in `_kv_geometry` is for.
    """
    transformers = pytest.importorskip("transformers")

    class _Gpt2Like:
        n_layer, n_head, n_embd = 12, 12, 768

    class _LlamaLike:
        num_hidden_layers, num_attention_heads = 22, 32
        num_key_value_heads, hidden_size = 4, 2048

    assert transformers is not None
    assert _kv_geometry(_Gpt2Like()) == (12, 12, 64)
    # Grouped-query: 4 KV heads sizing the cache, not the 32 attention heads.
    assert _kv_geometry(_LlamaLike()) == (22, 4, 64)


# -- the committed decoder profiles ----------------------------------------------------
#
# `results/decoder_profiles.json` and `results/decoder_profiles_tinyllama.json` are the
# two models the decoder page quotes. They are committed, so the claims about them are
# checkable without a 4.4 GB export. What is checked is the arithmetic that ties them
# together, not the values themselves -- a re-measurement should be free to move a
# perplexity without failing a test, but not free to move it in a direction that would
# make the write-up wrong.

import json  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PROFILES = {
    "gpt2": ROOT / "results" / "decoder_profiles.json",
    "tinyllama": ROOT / "results" / "decoder_profiles_tinyllama.json",
}


@pytest.fixture(scope="module")
def profiles() -> dict:
    missing = [n for n, p in PROFILES.items() if not p.exists()]
    if missing:  # pragma: no cover - both are committed in this repository
        pytest.skip(f"missing {missing}; run scripts/export_decoder.py")
    return {n: json.loads(p.read_text()) for n, p in PROFILES.items()}


def test_the_kv_geometry_agrees_across_every_precision_of_a_model(profiles):
    """The cache is a property of the architecture, not of the weights' precision.

    Weight-only quantisation leaves the KV cache in float, which is what makes one
    arena serve every precision. If a precision ever reported different geometry, the
    arena sized for one would be wrong for another.
    """
    for name, payload in profiles.items():
        shapes = {
            (v["layers"], v["kv_heads"], v["head_dim"], v["kv_bytes_per_token"])
            for v in payload["variants"]
        }
        assert len(shapes) == 1, f"{name} reports {len(shapes)} different KV geometries"


def test_kv_bytes_per_token_is_the_geometry_times_two_for_key_and_value(profiles):
    for name, payload in profiles.items():
        for variant in payload["variants"]:
            expected = 2 * variant["layers"] * variant["kv_heads"] * variant["head_dim"] * 4
            assert variant["kv_bytes_per_token"] == expected, name


def test_tinyllama_carries_less_kv_per_token_than_gpt2_on_more_layers(profiles):
    """The reason a second model was exported, as an assertion rather than a sentence.

    Grouped-query attention is the whole point: 4 KV heads against 12 on 22 layers
    against 12. If a re-export ever produced 22 KV heads, the graph would be
    multi-head after all and every expectation on the decoder page about where the
    gather's time goes would be built on the wrong shape.
    """
    gpt2 = profiles["gpt2"]["variants"][0]
    tiny = profiles["tinyllama"]["variants"][0]
    assert tiny["layers"] > gpt2["layers"]
    assert tiny["kv_heads"] < gpt2["kv_heads"]
    assert tiny["kv_bytes_per_token"] < gpt2["kv_bytes_per_token"]
    ratio = tiny["kv_bytes_per_token"] / gpt2["kv_bytes_per_token"]
    assert ratio == pytest.approx(0.61, abs=0.02), (
        f"KV per token is {ratio:.3f}x GPT-2's, not the 0.61x benchmarks.md quotes"
    )


def test_quantisation_costs_perplexity_and_never_improves_it(profiles):
    """Weight-only quantisation is a rounding error on the weights, so it cannot help.

    A negative delta would mean the measurement is noisy enough to be meaningless,
    which for a deterministic score over fixed windows would be a real defect.
    """
    for name, payload in profiles.items():
        by_precision = {v["precision"]: v for v in payload["variants"]}
        if "fp32" not in by_precision:  # pragma: no cover - always exported
            continue
        for precision in ("int8", "int4"):
            variant = by_precision.get(precision)
            if variant is None:  # pragma: no cover - both are exported
                continue
            assert variant["perplexity_delta"] > 0, f"{name}/{precision}"
            assert variant["size_ratio_vs_fp32"] < 1.0, f"{name}/{precision}"
        if "int8" in by_precision and "int4" in by_precision:
            assert (
                by_precision["int4"]["perplexity_delta"] > by_precision["int8"]["perplexity_delta"]
            ), f"{name}: INT4 costs no more perplexity than INT8, which inverts the ordering"


def test_weight_only_quantisation_compresses_the_larger_model_further(profiles):
    """The external-validity finding, pinned.

    What stays in float -- the embedding table and the excluded output projection -- is
    about a quarter of GPT-2 and about a twelfth of TinyLlama, so the size ratios
    quoted for GPT-2 are a small-model artefact. If a future export inverted this, the
    paragraph saying so on benchmarks.md would be wrong.
    """
    for precision in ("int8", "int4"):
        ratios = {}
        for name, payload in profiles.items():
            variant = next((v for v in payload["variants"] if v["precision"] == precision), None)
            if variant is None:  # pragma: no cover - both are exported
                pytest.skip(f"{name} has no {precision} variant")
            ratios[name] = variant["size_ratio_vs_fp32"]
        assert ratios["tinyllama"] < ratios["gpt2"], precision


def test_the_engine_and_a_separate_session_agree_bitwise(profiles):
    """The one bitwise claim the decoder makes, and the form of it that is safe.

    One graph through two sessions of the same library. Not across batch widths, where
    the GEMM shape changes and float addition is not associative -- see
    docs/runtime.md, and tests/test_batching.py for the assertion that had to be
    replaced with a bound after claiming otherwise.
    """
    for name, payload in profiles.items():
        assert payload["engine_vs_session_max_logit_diff"] == 0.0, name


def test_every_profile_records_the_runtime_it_was_measured_with(profiles):
    """pyproject.toml carries a floor, so the version is a measurement, not a constant."""
    for name, payload in profiles.items():
        assert payload["host"].get("onnxruntime"), (
            f"{name} does not record its ONNX Runtime version, so its numbers can be "
            f"neither re-derived nor falsified"
        )
