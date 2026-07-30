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
    apply_partial_descriptor_shim,
    find_output_projection,
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

    patched = apply_partial_descriptor_shim()
    if sys.version_info < (3, 14):
        assert patched == 0
    else:
        # optimum declares a partial on every decoder config needing renamed
        # fields; if it stops doing so the shim is no longer needed.
        assert patched > 0
