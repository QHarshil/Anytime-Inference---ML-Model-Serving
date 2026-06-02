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
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..utils.logger import get_logger

LOGGER = get_logger("serving.onnx_runtime")


@dataclass
class InferenceRequest:
    variant: str
    data: np.ndarray
    request_id: str = field(default_factory=lambda: uuid.uuid4().hex)


@dataclass
class InferenceResponse:
    request_id: str
    logits: np.ndarray
    runtime_latency_ms: float
    wall_latency_ms: float


def _encode(array: np.ndarray) -> Dict:
    contiguous = np.ascontiguousarray(array)
    return {
        "shape": list(contiguous.shape),
        "dtype": str(contiguous.dtype),
        "data": base64.b64encode(contiguous.tobytes()).decode("ascii"),
    }


def _decode(payload: Dict) -> np.ndarray:
    raw = base64.b64decode(payload["data"])
    dtype = np.dtype(payload["dtype"])
    return np.frombuffer(raw, dtype=dtype).reshape(payload["shape"])


class _RuntimeBackend:
    """Abstract backend that takes a request dict and returns a response dict."""

    def submit(self, request: Dict) -> Dict:
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError


class _SubprocessBackend(_RuntimeBackend):
    """Backend backed by the C++ binary, communicating via stdin/stdout."""

    def __init__(self, binary: Path, model_paths: Dict[str, Path]) -> None:
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
        # Drain handshake line (the C++ binary prints "ready" once both models
        # have loaded).
        handshake = self._process.stdout.readline().strip()
        if handshake != "ready":
            stderr = self._process.stderr.read()
            raise RuntimeError(f"C++ runtime failed to start: {handshake!r} ({stderr})")

    def submit(self, request: Dict) -> Dict:
        line = json.dumps(request) + "\n"
        self._process.stdin.write(line)
        self._process.stdin.flush()
        response_line = self._process.stdout.readline()
        if not response_line:
            stderr = self._process.stderr.read()
            raise RuntimeError(f"C++ runtime closed unexpectedly: {stderr}")
        return json.loads(response_line)

    def close(self) -> None:
        if self._process.poll() is None:
            try:
                self._process.stdin.close()
            except Exception:
                pass
            self._process.terminate()
            try:
                self._process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                self._process.kill()


class _PythonBackend(_RuntimeBackend):
    """Pure-Python ONNX Runtime backend. Used when the C++ binary is absent."""

    def __init__(self, model_paths: Dict[str, Path]) -> None:
        import onnxruntime as ort  # local import so the dep is optional at import time

        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        self._sessions = {
            variant: ort.InferenceSession(str(path), sess_options=options)
            for variant, path in model_paths.items()
        }

    def submit(self, request: Dict) -> Dict:
        variant = request["variant"]
        session = self._sessions[variant]
        feeds = {}
        for name, payload in request["inputs"].items():
            feeds[name] = _decode(payload)
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
        model_paths: Dict[str, Path],
        *,
        binary: Optional[Path] = None,
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
            "inputs": {self._input_name: _encode(request.data)},
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

    def __enter__(self) -> "RuntimeClient":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


class RuntimePool:
    """Pool of RuntimeClients dispatched to from worker threads."""

    def __init__(
        self,
        size: int,
        model_paths: Dict[str, Path],
        *,
        binary: Optional[Path] = None,
        input_name: str = "input",
    ) -> None:
        if size <= 0:
            raise ValueError("size must be positive")
        self._clients: List[RuntimeClient] = [
            RuntimeClient(model_paths, binary=binary, input_name=input_name) for _ in range(size)
        ]
        self._free: "queue.Queue[RuntimeClient]" = queue.Queue()
        for client in self._clients:
            self._free.put(client)

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

    def __enter__(self) -> "RuntimePool":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def find_runtime_binary() -> Optional[Path]:
    """Locate the compiled C++ runtime, if it exists."""
    env_override = os.environ.get("ANYTIME_RUNTIME_BIN")
    if env_override:
        candidate = Path(env_override)
        return candidate if candidate.exists() else None
    repo_root = Path(__file__).resolve().parents[2]
    for candidate in (
        repo_root / "runtime_cpp" / "build" / "anytime_runtime",
        repo_root / "runtime_cpp" / "build" / "Release" / "anytime_runtime.exe",
    ):
        if candidate.exists():
            return candidate
    return None
