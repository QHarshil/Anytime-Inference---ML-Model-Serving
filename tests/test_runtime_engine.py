"""Tests for the in-process engine, against the reference implementation.

The point of most of this file is cross-backend agreement. The extension runs
ONNX Runtime in this process over borrowed buffers; the ``python`` backend runs the
same graph through the onnxruntime wheel. Both must produce bitwise identical
output, because they are the same library doing the same work and any difference
means the tensor path is corrupting something.

The Stage 1 subprocess worker was the third backend here until it was removed. It
agreed bitwise with the extension on DistilBERT, which is what justified deleting
it: a replacement should be checked against the thing it replaces before that
thing goes away.

Tests needing the compiled extension are skipped when it is absent. Build it with:

    pip install -e .
"""

import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

onnx = pytest.importorskip("onnx", reason="onnx is required to build the fixture graphs")
from onnx import TensorProto, helper, numpy_helper  # noqa: E402

from anytime_serving.serving.onnx_runtime import (  # noqa: E402
    BACKENDS,
    InferenceRequest,
    RuntimeClient,
    RuntimePool,
    extension_available,
    load_extension,
)

requires_extension = pytest.mark.skipif(
    not extension_available(), reason="anytime_runtime is not built; see module docstring"
)

# Fixed weights, so every backend is asked to compute exactly the same thing and
# the comparison is over arithmetic rather than over random inputs.
WEIGHT = np.arange(16, dtype=np.float32).reshape(4, 4) / 8.0 - 1.0
BIAS = np.array([0.25, -0.5, 0.125, 1.0], dtype=np.float32)


def _build_graph(path: Path) -> None:
    """A graph with real arithmetic in it.

    An Identity graph would round-trip a payload without exercising a kernel, so
    it cannot show that two backends compute the same result. MatMul, Add, and
    Relu do.
    """
    inp = helper.make_tensor_value_info("input", TensorProto.FLOAT, [None, 4])
    out = helper.make_tensor_value_info("logits", TensorProto.FLOAT, [None, 4])
    nodes = [
        helper.make_node("MatMul", ["input", "weight"], ["projected"]),
        helper.make_node("Add", ["projected", "bias"], ["biased"]),
        helper.make_node("Relu", ["biased"], ["logits"]),
    ]
    graph = helper.make_graph(
        nodes,
        "affine_relu",
        [inp],
        [out],
        initializer=[
            numpy_helper.from_array(WEIGHT, "weight"),
            numpy_helper.from_array(BIAS, "bias"),
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 14)], ir_version=8)
    onnx.save(model, str(path))


def _build_two_input_graph(path: Path) -> None:
    """Declares two inputs, so input filtering can be tested."""
    left = helper.make_tensor_value_info("input", TensorProto.FLOAT, [None, 4])
    right = helper.make_tensor_value_info("scale", TensorProto.FLOAT, [None, 4])
    out = helper.make_tensor_value_info("logits", TensorProto.FLOAT, [None, 4])
    graph = helper.make_graph(
        [helper.make_node("Mul", ["input", "scale"], ["logits"])],
        "scaled",
        [left, right],
        [out],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 14)], ir_version=8)
    onnx.save(model, str(path))


@pytest.fixture(scope="module")
def graphs():
    with tempfile.TemporaryDirectory() as tmp:
        affine = Path(tmp) / "affine.onnx"
        two_input = Path(tmp) / "two_input.onnx"
        _build_graph(affine)
        _build_two_input_graph(two_input)
        yield {"affine": affine, "two_input": two_input}


@pytest.fixture(scope="module")
def model_paths(graphs):
    return {"fp32": graphs["affine"], "int8": graphs["affine"]}


def _available_backends() -> list[str]:
    available = ["python"]
    if extension_available():
        available.append("extension")
    return available


def _client(model_paths, backend):
    return RuntimeClient(model_paths, backend=backend)


def _expected(data: np.ndarray) -> np.ndarray:
    return np.maximum(data @ WEIGHT + BIAS, 0.0)


# --- the premise the comparisons rest on ------------------------------------


def test_required_backends_are_actually_present():
    """Fail rather than skip where a backend is supposed to exist.

    Most tests here skip a backend that is not built, which is right locally but
    would let a CI job quietly compare one backend against itself and report
    success. ``ANYTIME_REQUIRE_BACKENDS`` names the backends an environment
    promises to provide, and CI sets it so the parity comparison cannot decay into
    a no-op without anyone noticing.
    """
    required = os.environ.get("ANYTIME_REQUIRE_BACKENDS", "").strip()
    if not required:
        pytest.skip("ANYTIME_REQUIRE_BACKENDS is not set")

    requested = [name.strip() for name in required.split(",") if name.strip()]
    unknown = sorted(set(requested) - set(BACKENDS))
    assert not unknown, f"ANYTIME_REQUIRE_BACKENDS names unknown backend(s): {unknown}"

    available = _available_backends()
    missing = sorted(set(requested) - set(available))
    assert not missing, (
        f"ANYTIME_REQUIRE_BACKENDS demands {requested} but only {available} are "
        f"available. The cross-backend comparison would have skipped instead of "
        f"failing."
    )


# --- version equality -------------------------------------------------------


@requires_extension
def test_extension_links_the_installed_onnxruntime():
    """The two ONNX Runtimes in this process must be the same version.

    Enforced at CMake configure time as well. This is the gate that still applies
    when a built extension is carried into an environment with a different wheel,
    which is how the Stage 1 mismatch would have been caught immediately.
    """
    import onnxruntime

    extension = load_extension()
    assert extension.onnxruntime_version() == onnxruntime.__version__


@requires_extension
def test_extension_reports_its_compiled_api_version():
    extension = load_extension()
    assert extension.ort_api_version() > 0


# --- cross-backend agreement ------------------------------------------------


def test_every_backend_computes_the_same_result(model_paths):
    """Bitwise agreement across every backend built in this environment."""
    backends = _available_backends()
    if len(backends) < 2:
        pytest.skip(f"only the {backends[0]!r} backend is available; nothing to compare")

    rng = np.random.default_rng(0)
    data = rng.standard_normal((3, 4)).astype(np.float32)
    request = InferenceRequest(variant="fp32", data=data)

    results = {}
    for backend in backends:
        with _client(model_paths, backend) as client:
            assert client.backend_name == backend
            results[backend] = client.infer(request).logits

    # Against the arithmetic, then against each other, so a shared error in two
    # backends cannot pass as agreement.
    reference = _expected(data)
    for backend, logits in results.items():
        np.testing.assert_allclose(logits, reference, rtol=1e-6, atol=1e-6)
    for backend in backends[1:]:
        np.testing.assert_array_equal(
            results[backend],
            results[backends[0]],
            err_msg=f"{backend} disagrees with {backends[0]} bitwise",
        )


@pytest.mark.parametrize("backend", BACKENDS)
def test_backend_round_trips_a_batch(model_paths, backend):
    if backend not in _available_backends():
        pytest.skip(f"the {backend!r} backend is not available here")
    rng = np.random.default_rng(1)
    data = rng.standard_normal((5, 4)).astype(np.float32)
    with _client(model_paths, backend) as client:
        response = client.infer(InferenceRequest(variant="fp32", data=data))
    np.testing.assert_allclose(response.logits, _expected(data), rtol=1e-6, atol=1e-6)
    assert response.runtime_latency_ms >= 0.0
    assert response.wall_latency_ms >= response.runtime_latency_ms - 1e-6


@pytest.mark.parametrize("backend", BACKENDS)
def test_backend_routes_between_variants(model_paths, backend):
    if backend not in _available_backends():
        pytest.skip(f"the {backend!r} backend is not available here")
    data = np.full((1, 4), 0.5, dtype=np.float32)
    with _client(model_paths, backend) as client:
        for variant in ("fp32", "int8"):
            response = client.infer(InferenceRequest(variant=variant, data=data))
            np.testing.assert_allclose(response.logits, _expected(data), rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("backend", BACKENDS)
def test_backend_reports_unknown_variant_and_stays_usable(model_paths, backend):
    """An unknown variant raises without taking the worker down."""
    if backend not in _available_backends():
        pytest.skip(f"the {backend!r} backend is not available here")
    data = np.ones((1, 4), dtype=np.float32)
    with _client(model_paths, backend) as client:
        with pytest.raises(RuntimeError, match="unknown variant"):
            client.infer(InferenceRequest(variant="does_not_exist", data=data))
        response = client.infer(InferenceRequest(variant="fp32", data=data))
        np.testing.assert_allclose(response.logits, _expected(data), rtol=1e-6, atol=1e-6)


# --- input handling ---------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS)
def test_backend_drops_undeclared_inputs(graphs, backend):
    """Callers send the union of every variant's inputs; extras are dropped.

    Variants of one task declare different inputs, so a graph must take the subset
    it declares rather than failing on the rest.
    """
    if backend not in _available_backends():
        pytest.skip(f"the {backend!r} backend is not available here")
    paths = {"affine": graphs["affine"]}
    data = np.full((1, 4), 2.0, dtype=np.float32)
    with _client(paths, backend) as client:
        response = client.infer(
            InferenceRequest(
                variant="affine",
                inputs={"input": data, "token_type_ids": np.zeros((1, 4), dtype=np.int64)},
            )
        )
    np.testing.assert_allclose(response.logits, _expected(data), rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("backend", BACKENDS)
def test_backend_rejects_a_missing_declared_input(graphs, backend):
    """A partial feed is an error, not a run on whatever was supplied."""
    if backend not in _available_backends():
        pytest.skip(f"the {backend!r} backend is not available here")
    paths = {"scaled": graphs["two_input"]}
    with _client(paths, backend) as client:
        with pytest.raises(RuntimeError, match="missing input"):
            client.infer(
                InferenceRequest(
                    variant="scaled", inputs={"input": np.ones((1, 4), dtype=np.float32)}
                )
            )


@requires_extension
def test_extension_accepts_a_non_contiguous_input(model_paths):
    """A strided array is made contiguous rather than misread.

    ONNX Runtime reads the buffer directly, so handing it a transposed view
    without converting would silently transpose the arithmetic.
    """
    rng = np.random.default_rng(2)
    base = rng.standard_normal((4, 4)).astype(np.float32)
    strided = base.T
    assert not strided.flags.c_contiguous

    with _client(model_paths, "extension") as client:
        from_strided = client.infer(InferenceRequest(variant="fp32", data=strided)).logits
        from_copy = client.infer(
            InferenceRequest(variant="fp32", data=np.ascontiguousarray(strided))
        ).logits
    np.testing.assert_array_equal(from_strided, from_copy)
    np.testing.assert_allclose(from_strided, _expected(strided), rtol=1e-6, atol=1e-6)


@requires_extension
def test_extension_rejects_an_unsupported_dtype(model_paths):
    """An unaccepted dtype fails loudly rather than being reinterpreted."""
    with _client(model_paths, "extension") as client:
        with pytest.raises(ValueError, match="float16|does not accept"):
            client.infer(InferenceRequest(variant="fp32", data=np.ones((1, 4), dtype=np.float16)))


# --- the properties the extension exists for --------------------------------


@requires_extension
def test_extension_output_is_a_view_not_a_copy(model_paths):
    """Outputs borrow ONNX Runtime's buffer.

    Copying every output would undo the reason for running in-process, so assert
    the array does not own its memory and is kept alive by its base.
    """
    data = np.ones((2, 4), dtype=np.float32)
    with _client(model_paths, "extension") as client:
        logits = client.infer(InferenceRequest(variant="fp32", data=data)).logits
    assert not logits.flags.owndata
    assert logits.base is not None
    # The buffer has to stay valid once the client is closed, since the response
    # outlives the call that produced it.
    np.testing.assert_allclose(logits, _expected(data), rtol=1e-6, atol=1e-6)


@requires_extension
def test_extension_releases_the_gil_during_inference(model_paths):
    """Concurrent pool workers must actually overlap.

    Holding the GIL through inference would serialise the pool, and the M/M/c
    model the admission controller uses would describe a machine that does not
    exist. Correctness under concurrency is asserted here; the throughput claim
    itself belongs to the benchmarks, not to a timing-sensitive unit test.
    """
    from concurrent.futures import ThreadPoolExecutor

    rng = np.random.default_rng(3)
    requests = [
        InferenceRequest(
            variant="int8" if i % 2 else "fp32",
            data=rng.standard_normal((1, 4)).astype(np.float32),
        )
        for i in range(16)
    ]
    with RuntimePool(size=4, model_paths=model_paths, backend="extension") as pool:
        with ThreadPoolExecutor(max_workers=4) as executor:
            responses = list(executor.map(pool.infer, requests))

    for request, response in zip(requests, responses, strict=True):
        assert response.request_id == request.request_id
        assert request.data is not None
        np.testing.assert_allclose(response.logits, _expected(request.data), rtol=1e-6, atol=1e-6)


# --- backend selection ------------------------------------------------------


@requires_extension
def test_the_extension_is_preferred_when_available(model_paths):
    """Automatic selection picks the in-process path, not the fallback."""
    with RuntimeClient(model_paths) as client:
        assert client.backend_name == "extension"


def test_an_unknown_backend_name_is_rejected(model_paths):
    with pytest.raises(ValueError, match="backend must be one of"):
        RuntimeClient(model_paths, backend="does_not_exist")
