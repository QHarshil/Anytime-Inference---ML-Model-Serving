"""Client for the inference runtime.

Two backends implement one interface, both taking and returning numpy arrays:

``extension``
    The ``anytime_runtime`` pybind11 module, which runs ONNX Runtime in this
    process. Tensors are borrowed, not copied, and the GIL is released
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
from .batching import BatchCounts, RequestBatcher

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
    name it. Models with several inputs, such as a transformer taking ``input_ids``
    and ``attention_mask``, set ``inputs`` instead, which is passed through
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
    # How many requests shared the Run this came out of. Recorded because
    # `runtime_latency_ms` is the batch's, so a service time read off a batched
    # response means something different at width 8 than at width 1, and nothing
    # else in the response says which it was.
    batch_size: int = 1


class _RuntimeBackend:
    """Runs one variant over a name -> tensor mapping.

    Implementations return the first graph output and the time spent inside
    inference, excluding anything the client adds around it.

    Anything meaning "the runtime could not serve this request" raises
    ``RuntimeError``: an unknown variant, or a missing declared input. A malformed
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
    """Single inference worker.

    Owns its backend unless it was built by `sharing`, in which case several clients
    hold the same one and none of them may close it.
    """

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
        self._owns_backend = True
        self._lock = threading.Lock()

    @classmethod
    def sharing(cls, backend: _RuntimeBackend, *, input_name: str = "input") -> RuntimeClient:
        """A worker over a backend somebody else owns and will close.

        The lock still guards this client, which is what makes the sharing safe to reason
        about. It is per client and not per backend, so it serialises nothing between
        workers and the concurrency the pool provides is unchanged. What the backend has
        to be is re-entrant, and both are. `Engine::run` reads a model map fixed at
        construction and calls `Session::Run`, which ONNX Runtime documents as safe to
        call concurrently, and the binding releases the GIL around it.
        """
        client = cls.__new__(cls)
        client._input_name = input_name
        client._backend = backend
        client._owns_backend = False
        client._lock = threading.Lock()
        return client

    @property
    def backend_name(self) -> str:
        """Which backend is serving. Recorded alongside every measurement."""
        return self._backend.name

    def infer(self, request: InferenceRequest) -> InferenceResponse:
        feeds = request.feed(self._input_name)
        start = time.perf_counter()
        logits, runtime_latency_ms = self.infer_feeds(request.variant, feeds)
        wall_ms = (time.perf_counter() - start) * 1000.0
        return InferenceResponse(
            request_id=request.request_id,
            logits=logits,
            runtime_latency_ms=runtime_latency_ms,
            wall_latency_ms=wall_ms,
        )

    def infer_feeds(self, variant: str, feeds: dict[str, np.ndarray]) -> tuple[np.ndarray, float]:
        """Run an already-assembled feed, whatever its batch width.

        `infer` is one request; this is the entry the batcher needs, where the
        leading axis holds several requests and no single `request_id` owns the
        result. Both go through the same lock and the same backend, so a batched
        run is not a second code path into the runtime.
        """
        with self._lock:
            return self._backend.infer(variant, feeds)

    def close(self) -> None:
        # A shared backend outlives every client over it, so closing here would pull the
        # sessions out from under the workers still holding it.
        if self._owns_backend:
            self._backend.close()

    def __enter__(self) -> RuntimeClient:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


class RuntimePool:
    """Pool of RuntimeClients dispatched to from worker threads.

    `share_sessions` decides whether the pool loads the graphs once or once per worker.
    It defaults to on, and both halves of that are measured.

    On, one backend serves every worker. The weights are read-only, so sharing them
    changes no answer, and `intra_op_num_threads` is 1 either way, so a Run stays
    single-threaded and the workers do not contend for an intra-op pool. N workers over
    one backend are still N independent single-threaded servers, which is the reading
    the M/M/c admission model rests on.

    Off, each worker holds its own sessions, which costs N copies of every variant's
    weights: 1598 MB against 549 MB at four workers over DistilBERT and MiniLM.

    The concern that kept this off was ONNX Runtime's per-session CPU arena, which
    sharing puts every worker on. Paired arms say it does not bite, and that sharing is
    faster as concurrency rises instead of slower. Shared over unshared throughput is
    1.015x at two workers, 1.017x at four and 1.074x at eight. `--no-share-sessions` on
    `run_load_sweep.py` is kept as the control that says so, the way `--no-spinning` is
    on the decoder path. See `scripts/ab_session_sharing.py`.
    """

    def __init__(
        self,
        size: int,
        model_paths: dict[str, Path],
        *,
        backend: str | None = None,
        input_name: str = "input",
        share_sessions: bool = True,
        max_batch_size: int = 1,
        max_batch_delay_ms: float = 0.0,
    ) -> None:
        if size <= 0:
            raise ValueError("size must be positive")
        if max_batch_size < 1:
            raise ValueError("max_batch_size must be at least 1")
        self._share_sessions = share_sessions
        self._input_name = input_name
        if share_sessions:
            if not model_paths:
                raise ValueError("model_paths must be non-empty")
            shared = _make_backend(model_paths, backend)
            # The pool owns the one backend; the clients borrow it. Ownership has to sit
            # somewhere singular or `close` runs once per worker over the same sessions.
            self._shared: _RuntimeBackend | None = shared
            self._clients = [
                RuntimeClient.sharing(shared, input_name=input_name) for _ in range(size)
            ]
        else:
            self._shared = None
            self._clients = [
                RuntimeClient(model_paths, backend=backend, input_name=input_name)
                for _ in range(size)
            ]
        self._free: queue.Queue[RuntimeClient] = queue.Queue()
        for client in self._clients:
            self._free.put(client)

        # `infer` keeps its per-request signature either way, which is what makes a
        # batched sweep and an unbatched one the same harness. At width 1 no batcher
        # is built at all, so the unbatched path is untouched and not merely
        # configured off.
        self._batcher: RequestBatcher | None = None
        if max_batch_size > 1:
            self._batcher = RequestBatcher(
                self._run_batch,
                max_batch_size=max_batch_size,
                max_delay_ms=max_batch_delay_ms,
                # One batch per worker at most: the batcher may not hand the pool
                # more concurrent work than it has clients to run it on, or callers
                # would block inside `_run_batch` holding a batch together.
                max_concurrent_batches=size,
            )

    @property
    def share_sessions(self) -> bool:
        """Whether one backend serves every worker. Recorded beside a measurement,
        because it changes what the pool costs in memory and may change what it costs
        in latency."""
        return self._share_sessions

    @property
    def loaded_backends(self) -> int:
        """How many times the graphs are loaded, which is the whole point of sharing.

        Exposed so the saving is a number a test can assert, not a claim in a
        docstring.
        """
        return 1 if self._share_sessions else len(self._clients)

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

    @property
    def max_batch_size(self) -> int:
        """Widest Run the pool will form. 1 means no batching at all."""
        return self._batcher.max_batch_size if self._batcher is not None else 1

    @property
    def batch_counts(self) -> BatchCounts | None:
        """What the batcher did, in counts. None when batching is off.

        Exposed because the achieved width is the only honest statement of what
        batching did on a given workload: a pool configured for width 8 that never
        saw eight requests at once ran at width 1 and should say so.
        """
        return self._batcher.counts if self._batcher is not None else None

    def _run_batch(self, variant: str, feeds: dict[str, np.ndarray]) -> tuple[np.ndarray, float]:
        """Run one assembled batch on whichever worker is free."""
        client = self._free.get()
        try:
            return client.infer_feeds(variant, feeds)
        finally:
            self._free.put(client)

    def infer(self, request: InferenceRequest) -> InferenceResponse:
        if self._batcher is not None:
            start = time.perf_counter()
            logits, runtime_latency_ms, width = self._batcher.infer(
                request.variant, request.feed(self._input_name)
            )
            return InferenceResponse(
                request_id=request.request_id,
                logits=logits,
                runtime_latency_ms=runtime_latency_ms,
                # Wall time includes the wait for a batch to form, which is the
                # whole cost of batching and is invisible in the runtime latency.
                wall_latency_ms=(time.perf_counter() - start) * 1000.0,
                batch_size=width,
            )
        client = self._free.get()
        try:
            return client.infer(request)
        finally:
            self._free.put(client)

    def close(self) -> None:
        # The batcher first: it dispatches onto these clients, so releasing their
        # sessions while a batch is in flight would run a closed session.
        if self._batcher is not None:
            self._batcher.close()
            self._batcher = None
        for client in self._clients:
            client.close()
        self._clients = []
        # Borrowing clients close nothing, so the one backend they shared is the pool's
        # to release. Doing it after the loop instead of inside it is what makes the
        # release happen exactly once however many workers there were.
        if self._shared is not None:
            self._shared.close()
            self._shared = None

    def __enter__(self) -> RuntimePool:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
