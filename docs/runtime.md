# C++ runtime worker

`runtime_cpp/` builds `anytime_runtime`, a single-threaded ONNX Runtime worker
that reads line-delimited JSON on stdin and writes it on stdout. The Python
control plane spawns one process per pool slot.

## Build

```bash
cmake -S runtime_cpp -B runtime_cpp/build -DCMAKE_BUILD_TYPE=Release \
    -DONNXRUNTIME_ROOT_PATH=/path/to/onnxruntime
cmake --build runtime_cpp/build -j
```

Download the ONNX Runtime release matching your platform from
<https://github.com/microsoft/onnxruntime/releases> and unpack it. Release tarball
layouts differ between versions: some nest a versioned directory. Point
`ONNXRUNTIME_ROOT_PATH` at whichever directory contains `include/` and `lib/`.

The binary lands at `runtime_cpp/build/anytime_runtime`. The Python client finds
it via `ANYTIME_RUNTIME_BIN`, or by searching upward from the package and the
working directory. Set the environment variable explicitly for a non-editable
install, where the package no longer sits inside the source tree.

## Match the ONNX Runtime version to the Python wheel

This is the single most important build detail. Building the worker against
1.20.1 while the Python side used the 1.26.0 wheel measured DistilBERT at
**98.9 ms inside the worker against 13.0 ms in the profiler**, a 7.6x gap. The
timing is taken around `session->Run()`, so this was not transport overhead; the
older release simply lacks the optimisations the newer one applies to this graph
on arm64.

Because the profiler and the serving path disagreed, every service time the
planner used was wrong by almost an order of magnitude and no request met its
deadline. With matched versions the worker measures 14.05 ms, within 8% of the
in-process session, and the JSON transport costs 0.27 ms per request.

`CMakeLists.txt` guards a related trap: `find_library` caches its result, so
reconfiguring with a different `ONNXRUNTIME_ROOT_PATH` would keep linking the
previously found library while compiling against the new headers. That mismatch
surfaces only at runtime, as `The requested API version [N] is not available`.
The cache entry is now invalidated when the root path changes, and the resolved
library is printed at configure time.

## Protocol

Handshake: the worker prints `ready` on stdout once every model has loaded.

Request, one JSON object per line:

```json
{"request_id": "r-1",
 "variant": "distilbert_fp32",
 "inputs": {"input_ids":      {"shape": [1,128], "dtype": "int64", "data": "<base64>"},
            "attention_mask": {"shape": [1,128], "dtype": "int64", "data": "<base64>"}}}
```

Response:

```json
{"request_id": "r-1",
 "logits": {"shape": [1,2], "dtype": "float32", "data": "<base64>"},
 "latency_ms": 14.05}
```

Error response, for anything recoverable:

```json
{"request_id": "r-1", "error": "unknown variant"}
```

The Python client raises `RuntimeError` on an error response.

## Behaviour

- **One inference at a time.** Run a pool from the Python side for concurrency.
  Intra-op threads are set to 1 so N workers behave as N independent servers,
  which is what makes the M/M/c queueing model in
  [`planner.md`](planner.md) valid.
- **Per-request error isolation.** A malformed request, an unknown variant, or an
  ONNX Runtime failure produces an error response and the worker continues. An
  earlier version wrapped the whole request loop in one `try`, so a single bad
  request terminated the worker and failed every subsequent request routed to it.
- **Input filtering.** Variants can declare different inputs. The worker keeps the
  subset its graph declares and drops the rest, and errors if a declared input is
  missing rather than running on a partial feed.
- **Supported dtypes.** Inputs `float32` and `int64`; outputs `float32`.
- **No external dependencies.** JSON parsing and base64 are implemented in
  `main.cpp` so the worker needs only ONNX Runtime.

## Tests

`tests/test_runtime_subprocess.py` exercises the real binary: round-trip
fidelity, the base64 codec across all three padding cases, variant routing,
concurrent dispatch through a pool, and error isolation. Marked `needs_runtime`
and skipped when the binary is absent.

```bash
pytest -q tests/test_runtime_subprocess.py
```

## Superseded by the in-process engine

This worker is no longer the serving path. `anytime_runtime`, documented in
[`../runtime/README.md`](../runtime/README.md), runs ONNX Runtime in the caller's
process over borrowed numpy buffers and is selected automatically.

The subprocess boundary costs a base64 encode, a JSON parse, and a copy in each
direction. Measured on DistilBERT at batch size one, over 60 requests after
warm-up:

| Backend | Inference p50 | Wall p50 | Transport |
| --- | --- | --- | --- |
| `python` | 13.667 ms | 13.673 ms | 0.006 ms |
| `extension` | 13.609 ms | 13.628 ms | 0.018 ms |
| `subprocess` | 13.629 ms | 13.926 ms | 0.296 ms |

All three agree bitwise on the logits, and their inference times agree within
0.4%, which is what a matched ONNX Runtime version looks like. The transport cost
falls from 0.296 ms to 0.018 ms.

Against 14 ms of inference that saving is not the reason for the change. The
reason is that a subprocess boundary rules out batching across requests and
sharing a KV cache between them, which is what the rest of Stage 2 needs.

The worker is kept for now because it is what the engine is validated against:
`tests/test_runtime_engine.py` asserts all three backends agree, on the principle
that a replacement should be checked against the thing it replaces before that
thing is deleted.
