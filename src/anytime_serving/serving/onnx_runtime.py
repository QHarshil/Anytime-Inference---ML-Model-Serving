"""Client for the C++ ONNX runtime worker.

The C++ binary (``runtime_cpp/``) accepts line-delimited JSON requests on
stdin and writes line-delimited JSON responses to stdout. This module wraps
that protocol and provides a worker pool for concurrent dispatch.

A pure-Python fallback worker is also provided. It runs ONNX Runtime in-process
and is used when the C++ binary has not been built (e.g. on CI without a C++
toolchain). The protocol it implements is identical, so unit tests exercise
the same code path the production server uses.
"""

from __future__ import annotations

import base64
import json
import os
import queue
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO

import numpy as np

from ..utils.logger import get_logger

LOGGER = get_logger("serving.onnx_runtime")


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
    request_id: str
    logits: np.ndarray
    runtime_latency_ms: float
    wall_latency_ms: float


def _encode(array: np.ndarray) -> dict:
    contiguous = np.ascontiguousarray(array)
    return {
        "shape": list(contiguous.shape),
        "dtype": str(contiguous.dtype),
        "data": base64.b64encode(contiguous.tobytes()).decode("ascii"),
    }


def _decode(payload: dict) -> np.ndarray:
    raw = base64.b64decode(payload["data"])
    dtype = np.dtype(payload["dtype"])
    return np.frombuffer(raw, dtype=dtype).reshape(payload["shape"])


class _RuntimeBackend:
    """Abstract backend that takes a request dict and returns a response dict."""

    def submit(self, request: dict) -> dict:
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError


class _SubprocessBackend(_RuntimeBackend):
    """Backend backed by the C++ binary, communicating via stdin/stdout."""

    def __init__(self, binary: Path, model_paths: dict[str, Path]) -> None:
        args = [str(binary)]
        for variant, path in model_paths.items():
            args.extend(["--model", f"{variant}={path}"])
        self._process = subprocess.Popen(
            args,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=1,
            text=True,
        )
        # Popen types the pipes as Optional. All three were requested above, so
        # bind them once to non-optional handles rather than narrowing at every
        # use site.
        assert self._process.stdin is not None
        assert self._process.stdout is not None
        assert self._process.stderr is not None
        self._stdin: IO[str] = self._process.stdin
        self._stdout: IO[str] = self._process.stdout
        self._stderr: IO[str] = self._process.stderr

        # Drain handshake line (the C++ binary prints "ready" once both models
        # have loaded).
        handshake = self._stdout.readline().strip()
        if handshake != "ready":
            stderr = self._stderr.read()
            raise RuntimeError(f"C++ runtime failed to start: {handshake!r} ({stderr})")

    def submit(self, request: dict) -> dict:
        line = json.dumps(request) + "\n"
        self._stdin.write(line)
        self._stdin.flush()
        response_line = self._stdout.readline()
        if not response_line:
            stderr = self._stderr.read()
            raise RuntimeError(f"C++ runtime closed unexpectedly: {stderr}")
        response = json.loads(response_line)
        # The worker reports a recoverable problem (an unknown variant, say) as an
        # error field and stays alive for the next request.
        if "error" in response:
            raise RuntimeError(
                f"C++ runtime rejected request {response.get('request_id', '?')}: "
                f"{response['error']}"
            )
        return response

    def close(self) -> None:
        if self._process.poll() is None:
            try:
                self._stdin.close()
            except OSError:
                pass
            self._process.terminate()
            try:
                self._process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                self._process.kill()
        # Popen leaves stdout/stderr open after the child exits; close them so a
        # long-lived pool does not accumulate file descriptors.
        for stream in (self._stdin, self._stdout, self._stderr):
            try:
                stream.close()
            except OSError:
                pass


class _PythonBackend(_RuntimeBackend):
    """Pure-Python ONNX Runtime backend. Used when the C++ binary is absent."""

    def __init__(self, model_paths: dict[str, Path]) -> None:
        import onnxruntime as ort  # local import so the dep is optional at import time

        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        self._sessions = {
            variant: ort.InferenceSession(str(path), sess_options=options)
            for variant, path in model_paths.items()
        }
        self._declared_inputs = {
            variant: {spec.name for spec in session.get_inputs()}
            for variant, session in self._sessions.items()
        }

    def submit(self, request: dict) -> dict:
        variant = request["variant"]
        session = self._sessions[variant]
        # Variants of the same task can declare different inputs, so the caller
        # sends the union and each graph takes the subset it declares. Mirrors the
        # filtering in runtime_cpp/src/main.cpp.
        declared = self._declared_inputs[variant]
        feeds = {
            name: _decode(payload)
            for name, payload in request["inputs"].items()
            if name in declared
        }
        missing = declared - feeds.keys()
        if missing:
            raise RuntimeError(f"request is missing inputs {sorted(missing)} for {variant!r}")
        start = time.perf_counter()
        outputs = session.run(None, feeds)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        return {
            "request_id": request["request_id"],
            "logits": _encode(outputs[0]),
            "latency_ms": elapsed_ms,
        }

    def close(self) -> None:
        self._sessions.clear()


class RuntimeClient:
    """Single inference worker, backed by either the C++ binary or ONNX Runtime."""

    def __init__(
        self,
        model_paths: dict[str, Path],
        *,
        binary: Path | None = None,
        input_name: str = "input",
    ) -> None:
        if not model_paths:
            raise ValueError("model_paths must be non-empty")
        self._input_name = input_name
        if binary is not None and binary.exists():
            LOGGER.info("Using C++ runtime binary at %s", binary)
            self._backend: _RuntimeBackend = _SubprocessBackend(binary, model_paths)
        else:
            if binary is not None:
                LOGGER.warning("C++ binary %s not found; falling back to Python backend", binary)
            self._backend = _PythonBackend(model_paths)
        self._lock = threading.Lock()

    def infer(self, request: InferenceRequest) -> InferenceResponse:
        payload = {
            "request_id": request.request_id,
            "variant": request.variant,
            "inputs": {
                name: _encode(tensor) for name, tensor in request.feed(self._input_name).items()
            },
        }
        start = time.perf_counter()
        with self._lock:
            response = self._backend.submit(payload)
        wall_ms = (time.perf_counter() - start) * 1000.0
        return InferenceResponse(
            request_id=response["request_id"],
            logits=_decode(response["logits"]),
            runtime_latency_ms=float(response["latency_ms"]),
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
        binary: Path | None = None,
        input_name: str = "input",
    ) -> None:
        if size <= 0:
            raise ValueError("size must be positive")
        self._clients: list[RuntimeClient] = [
            RuntimeClient(model_paths, binary=binary, input_name=input_name) for _ in range(size)
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


def _build_dir_candidates(root: Path) -> tuple[Path, ...]:
    build = root / "runtime_cpp" / "build"
    return (
        build / "anytime_runtime",
        build / "Release" / "anytime_runtime.exe",
    )


def find_runtime_binary() -> Path | None:
    """Locate the compiled C++ runtime, if it exists.

    ``ANYTIME_RUNTIME_BIN`` takes precedence and is the only reliable option for
    a non-editable install, where the package no longer lives inside the source
    tree. Otherwise search upward from this file (covering an editable install)
    and from the working directory (covering a plain checkout).
    """
    env_override = os.environ.get("ANYTIME_RUNTIME_BIN")
    if env_override:
        candidate = Path(env_override)
        return candidate if candidate.exists() else None

    roots: list[Path] = []
    # Walk up from this module: src/anytime_serving/serving/ -> repo root. Done by
    # search rather than a fixed parent index so moving the package does not
    # silently break resolution.
    here = Path(__file__).resolve()
    roots.extend(here.parents)
    cwd = Path.cwd().resolve()
    roots.append(cwd)
    roots.extend(cwd.parents)

    seen: set[Path] = set()
    for root in roots:
        if root in seen:
            continue
        seen.add(root)
        for candidate in _build_dir_candidates(root):
            if candidate.exists():
                return candidate
    return None
