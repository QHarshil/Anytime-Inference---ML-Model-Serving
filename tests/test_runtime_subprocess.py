"""Protocol tests against the compiled C++ runtime.

The Python fallback backend and the C++ subprocess backend implement the same
request/response protocol, but only the fallback is exercised by the rest of the
suite. These tests run the real binary so the framing, base64 codec, and
handshake are covered too.

Skipped when the binary has not been built. Build it with:

    cmake -S runtime_cpp -B runtime_cpp/build -DCMAKE_BUILD_TYPE=Release \\
        -DONNXRUNTIME_ROOT_PATH=/path/to/onnxruntime
    cmake --build runtime_cpp/build -j
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest

onnx = pytest.importorskip("onnx", reason="onnx is required to build the fixture model")
from onnx import TensorProto, helper  # noqa: E402

from anytime_serving.serving.onnx_runtime import (  # noqa: E402
    InferenceRequest,
    RuntimeClient,
    RuntimePool,
    find_runtime_binary,
)

pytestmark = pytest.mark.needs_runtime

BINARY = find_runtime_binary()
requires_binary = pytest.mark.skipif(
    BINARY is None, reason="C++ runtime binary not built; see module docstring"
)


def _build_identity_model(path: Path, dim: int = 4) -> None:
    inp = helper.make_tensor_value_info("input", TensorProto.FLOAT, [None, dim])
    out = helper.make_tensor_value_info("logits", TensorProto.FLOAT, [None, dim])
    graph = helper.make_graph(
        [helper.make_node("Identity", ["input"], ["logits"])], "identity", [inp], [out]
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 14)], ir_version=8)
    onnx.save(model, str(path))


@pytest.fixture(scope="module")
def model_paths():
    with tempfile.TemporaryDirectory() as tmp:
        fp32 = Path(tmp) / "fp32.onnx"
        int8 = Path(tmp) / "int8.onnx"
        _build_identity_model(fp32)
        _build_identity_model(int8)
        yield {"fp32": fp32, "int8": int8}


@requires_binary
def test_subprocess_round_trip_preserves_payload(model_paths):
    with RuntimeClient(model_paths, binary=BINARY) as client:
        data = np.arange(8, dtype=np.float32).reshape(2, 4)
        response = client.infer(InferenceRequest(variant="fp32", data=data))
        np.testing.assert_array_almost_equal(response.logits, data)
        assert response.runtime_latency_ms >= 0.0
        assert response.wall_latency_ms >= response.runtime_latency_ms - 1e-6


@requires_binary
def test_subprocess_base64_codec_survives_all_byte_values(model_paths):
    """Exercise the hand-rolled base64 codec across padding lengths.

    Payload sizes of 4, 5, and 6 floats give byte counts with remainders 0, 1,
    and 2 mod 3, covering every padding branch in the encoder.
    """
    try:
        for count in (4, 5, 6, 8, 16):
            data = np.linspace(-1e6, 1e6, count, dtype=np.float32).reshape(1, count)
            # The graph pins its input width, so rebuild it per payload size.
            _build_identity_model(model_paths["fp32"], dim=count)
            with RuntimeClient({"fp32": model_paths["fp32"]}, binary=BINARY) as sized:
                response = sized.infer(InferenceRequest(variant="fp32", data=data))
            np.testing.assert_array_equal(response.logits, data)
    finally:
        # Restore the shared module-scoped fixture for the other tests.
        _build_identity_model(model_paths["fp32"])


@requires_binary
def test_subprocess_routes_between_variants(model_paths):
    with RuntimeClient(model_paths, binary=BINARY) as client:
        for variant in ("fp32", "int8"):
            data = np.full((1, 4), 3.5, dtype=np.float32)
            response = client.infer(InferenceRequest(variant=variant, data=data))
            np.testing.assert_array_almost_equal(response.logits, data)


@requires_binary
def test_subprocess_pool_serves_concurrent_requests(model_paths):
    from concurrent.futures import ThreadPoolExecutor

    with RuntimePool(size=2, model_paths=model_paths, binary=BINARY) as pool:
        rng = np.random.default_rng(0)
        requests = [
            InferenceRequest(
                variant="int8" if i % 2 else "fp32",
                data=rng.standard_normal((1, 4)).astype(np.float32),
            )
            for i in range(8)
        ]
        with ThreadPoolExecutor(max_workers=4) as executor:
            responses = list(executor.map(pool.infer, requests))
    for request, response in zip(requests, responses, strict=True):
        np.testing.assert_array_almost_equal(response.logits, request.data)
        assert response.request_id == request.request_id


@requires_binary
def test_subprocess_reports_unknown_variant_and_stays_usable(model_paths):
    """An unknown variant raises a descriptive error without killing the worker."""
    with RuntimeClient(model_paths, binary=BINARY) as client:
        with pytest.raises(RuntimeError, match="unknown variant"):
            client.infer(
                InferenceRequest(variant="does_not_exist", data=np.zeros((1, 4), np.float32))
            )
        # The worker reports the problem and continues serving.
        data = np.ones((1, 4), dtype=np.float32)
        response = client.infer(InferenceRequest(variant="fp32", data=data))
        np.testing.assert_array_almost_equal(response.logits, data)


def test_find_runtime_binary_honours_env_override(monkeypatch, tmp_path):
    """The env override wins, and a missing path resolves to None."""
    monkeypatch.setenv("ANYTIME_RUNTIME_BIN", str(tmp_path / "absent"))
    assert find_runtime_binary() is None

    present = tmp_path / "anytime_runtime"
    present.write_text("")
    monkeypatch.setenv("ANYTIME_RUNTIME_BIN", str(present))
    assert find_runtime_binary() == present
