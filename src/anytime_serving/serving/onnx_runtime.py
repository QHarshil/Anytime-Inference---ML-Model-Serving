"""Client for the inference runtime.

Two backends implement one interface, both taking and returning numpy arrays:

``extension``
    The ``anytime_runtime`` pybind11 module, which runs ONNX Runtime in this
    process. Tensors are borrowed rather than copied and the GIL is released
    around inference, so a pool of workers runs concurrently. This is the serving
    path.
``python``
    ONNX Runtime through its own Python wheel. The reference implementation the
    tests compare the extension against, and the fallback where the extension has
    not been built.

The backend is chosen automatically unless ``backend=`` names one. Tests pin it
explicitly so a parity failure cannot be hidden by a silent fallback.
"""

from __future__ import annotations

import queue
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ..utils.logger import get_logger

LOGGER = get_logger("serving.onnx_runtime")

BACKENDS = ("extension", "python")

_extension: Any | None = None


def load_extension() -> Any:
    """Import ``anytime_runtime``, refusing a version mismatch.

    The extension and the ``onnxruntime`` wheel are two independent copies of
    ONNX Runtime loaded into one process. Stage 1 built the C++ worker against
    1.20.1 while profiling against the 1.26.0 wheel and measured DistilBERT at
    98.9 ms versus 13.0 ms inside ``session->Run()``; every service time the
    planner used was wrong by almost an order of magnitude, and nothing failed.
    The build enforces this equality at configure time, and this is the gate that
    still applies when a built extension is carried into an environment with a
    different wheel.
    """
    global _extension
    if _extension is not None:
        return _extension

    import anytime_runtime
    import onnxruntime

    linked = anytime_runtime.onnxruntime_version()
    installed = onnxruntime.__version__
    if linked != installed:
        raise RuntimeError(
            f"anytime_runtime links ONNX Runtime {linked} but the installed wheel is "
            f"{installed}. Both are loaded into this process, and a mismatch measured "
            f"a 7.6x difference in inference time during Stage 1. Rebuild the "
            f"extension against the current wheel: pip install -e . --no-cache-dir"
        )
    _extension = anytime_runtime
    return _extension


def extension_available() -> bool:
    """Whether the in-process extension can be imported and matches the wheel."""
    try:
        load_extension()
    except (ImportError, RuntimeError):
        return False
    return True


@dataclass
class InferenceRequest:
    """One inference request.

    Single-input models (an image classifier, say) set ``data`` and let the client
    name it. Models with several inputs -- a transformer taking ``input_ids`` and
    ``attention_mask`` -- set ``inputs`` instead, which is passed through
    verbatim. Setting ``inputs`` takes precedence over ``data``.
    """

    variant: str
    data: np.ndarray | None = None
    inputs: dict[str, np.ndarray] | None = None
    request_id: str = field(default_factory=lambda: uuid.uuid4().hex)

    def __post_init__(self) -> None:
        if self.data is None and not self.inputs:
            raise ValueError("InferenceRequest needs either data or inputs")

    def feed(self, default_input_name: str) -> dict[str, np.ndarray]:
        """Resolve the request to a name -> tensor mapping."""
        if self.inputs:
            return self.inputs
        assert self.data is not None  # guaranteed by __post_init__
        return {default_input_name: self.data}


@dataclass
class InferenceResponse:
    """Result of one inference.

    Under the extension backend ``logits`` is a view over the buffer ONNX Runtime
    allocated, not a copy, so holding the response holds that buffer. Copy it if
    it needs to outlive the request.
    """

    request_id: str
    logits: np.ndarray
    runtime_latency_ms: float
    wall_latency_ms: float


class _RuntimeBackend:
    """Runs one variant over a name -> tensor mapping.

    Implementations return the first graph output and the time spent inside
    inference, excluding anything the client adds around it.

    Anything meaning "the runtime could not serve this request" -- an unknown
    variant, a missing declared input -- raises ``RuntimeError``. A malformed
    argument, such as a dtype the engine does not accept, raises ``ValueError``.
    """

    name = "abstract"

    def infer(self, variant: str, feeds: dict[str, np.ndarray]) -> tuple[np.ndarray, float]:
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError


class _ExtensionBackend(_RuntimeBackend):
    """In-process ONNX Runtime through the ``anytime_runtime`` extension."""

    name = "extension"

    def __init__(self, model_paths: dict[str, Path]) -> None:
        extension = load_extension()
        self._engine = extension.Engine([(v, str(p)) for v, p in model_paths.items()])
        self._variants = frozenset(self._engine.variants)

    def infer(self, variant: str, feeds: dict[str, np.ndarray]) -> tuple[np.ndarray, float]:
        if variant not in self._variants:
            raise RuntimeError(f"unknown variant {variant!r}; loaded: {sorted(self._variants)}")
        outputs, latency_ms = self._engine.run(variant, feeds)
        return outputs[0], float(latency_ms)

    def close(self) -> None:
        # Dropping the engine releases the sessions and their arenas.
        self._engine = None


class _PythonBackend(_RuntimeBackend):
    """ONNX Runtime through its Python wheel. The reference implementation."""

    name = "python"

    def __init__(self, model_paths: dict[str, Path]) -> None:
        import onnxruntime as ort  # local import so the dep is optional at import time

        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self._sessions = {
            variant: ort.InferenceSession(
                str(path), sess_options=options, providers=["CPUExecutionProvider"]
            )
            for variant, path in model_paths.items()
        }
        self._declared_inputs = {
            variant: {spec.name for spec in session.get_inputs()}
            for variant, session in self._sessions.items()
        }

    def infer(self, variant: str, feeds: dict[str, np.ndarray]) -> tuple[np.ndarray, float]:
        if variant not in self._sessions:
            raise RuntimeError(f"unknown variant {variant!r}; loaded: {sorted(self._sessions)}")
        # Variants of one task can declare different inputs, so the caller sends
        # the union and each graph takes the subset it declares. Mirrors the
        # filtering in runtime/src/engine.cpp.
        declared = self._declared_inputs[variant]
        fed = {name: tensor for name, tensor in feeds.items() if name in declared}
        missing = declared - fed.keys()
        if missing:
            raise RuntimeError(f"variant {variant!r} is missing input(s): {sorted(missing)}")
        start = time.perf_counter()
        outputs = self._sessions[variant].run(None, fed)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        return outputs[0], elapsed_ms

    def close(self) -> None:
        self._sessions.clear()


def _make_backend(model_paths: dict[str, Path], requested: str | None) -> _RuntimeBackend:
    """Build the requested backend, or pick one."""
    if requested is not None:
        if requested not in BACKENDS:
            raise ValueError(f"backend must be one of {BACKENDS}, got {requested!r}")
        if requested == "extension":
            return _ExtensionBackend(model_paths)
        return _PythonBackend(model_paths)

    if extension_available():
        return _ExtensionBackend(model_paths)
    LOGGER.warning(
        "anytime_runtime is unavailable; falling back to the Python backend. Measured "
        "service times will not reflect the serving path. Build the extension with: "
        "pip install -e ."
    )
    return _PythonBackend(model_paths)


class RuntimeClient:
    """Single inference worker."""

    def __init__(
        self,
        model_paths: dict[str, Path],
        *,
        backend: str | None = None,
        input_name: str = "input",
    ) -> None:
        if not model_paths:
            raise ValueError("model_paths must be non-empty")
        self._input_name = input_name
        self._backend = _make_backend(model_paths, backend)
        self._lock = threading.Lock()

    @property
    def backend_name(self) -> str:
        """Which backend is serving. Recorded alongside every measurement."""
        return self._backend.name

    def infer(self, request: InferenceRequest) -> InferenceResponse:
        feeds = request.feed(self._input_name)
        start = time.perf_counter()
        with self._lock:
            logits, runtime_latency_ms = self._backend.infer(request.variant, feeds)
        wall_ms = (time.perf_counter() - start) * 1000.0
        return InferenceResponse(
            request_id=request.request_id,
            logits=logits,
            runtime_latency_ms=runtime_latency_ms,
            wall_latency_ms=wall_ms,
        )

    def close(self) -> None:
        self._backend.close()

    def __enter__(self) -> RuntimeClient:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


class RuntimePool:
    """Pool of RuntimeClients dispatched to from worker threads."""

    def __init__(
        self,
        size: int,
        model_paths: dict[str, Path],
        *,
        backend: str | None = None,
        input_name: str = "input",
    ) -> None:
        if size <= 0:
            raise ValueError("size must be positive")
        self._clients: list[RuntimeClient] = [
            RuntimeClient(model_paths, backend=backend, input_name=input_name) for _ in range(size)
        ]
        self._free: queue.Queue[RuntimeClient] = queue.Queue()
        for client in self._clients:
            self._free.put(client)

    @property
    def size(self) -> int:
        """Number of workers serving this pool.

        The admission controller needs this to model the queue as M/M/c.
        """
        return len(self._clients)

    @property
    def backend_name(self) -> str:
        """Which backend the workers use."""
        return self._clients[0].backend_name if self._clients else "none"

    def infer(self, request: InferenceRequest) -> InferenceResponse:
        client = self._free.get()
        try:
            return client.infer(request)
        finally:
            self._free.put(client)

    def close(self) -> None:
        for client in self._clients:
            client.close()
        self._clients = []

    def __enter__(self) -> RuntimePool:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
