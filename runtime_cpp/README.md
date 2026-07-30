# anytime_runtime (C++)

Single-threaded ONNX Runtime worker. Loads one ONNX graph per variant and answers
line-delimited JSON requests on stdin. `RuntimePool` in
`src/anytime_serving/serving/onnx_runtime.py` spawns one process per pool slot and
dispatches across them.

## Build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
    -DONNXRUNTIME_ROOT_PATH=/path/to/onnxruntime
cmake --build build -j
```

Download the release matching your platform from
<https://github.com/microsoft/onnxruntime/releases>. Point
`ONNXRUNTIME_ROOT_PATH` at the directory containing `include/` and `lib/`.

**The version must match the `onnxruntime` Python wheel.** A mismatch is a silent
performance cliff, not a link error. See
[`../docs/runtime.md`](../docs/runtime.md).

The binary lands at `build/anytime_runtime`. The Python client finds it via
`ANYTIME_RUNTIME_BIN` or by searching upward from the package and the working
directory.

## Protocol

Handshake: the worker prints `ready` once every model has loaded.

```json
{"request_id":"r-1","variant":"distilbert_fp32",
 "inputs":{"input_ids":{"shape":[1,128],"dtype":"int64","data":"<base64>"},
           "attention_mask":{"shape":[1,128],"dtype":"int64","data":"<base64>"}}}
```

```json
{"request_id":"r-1",
 "logits":{"shape":[1,2],"dtype":"float32","data":"<base64>"},
 "latency_ms":14.05}
```

Recoverable failures return `{"request_id":"r-1","error":"..."}` and the worker
continues serving.

## Notes

- One inference per process. Run a pool from the Python side for concurrency.
- Input dtypes `float32` and `int64`; output dtype `float32`.
- Inputs a graph does not declare are dropped; a missing declared input is an
  error.
- No dependencies beyond ONNX Runtime. JSON and base64 are implemented in
  `src/main.cpp`.

Full details, including the version-matching rationale and the CMake cache
pitfall: [`../docs/runtime.md`](../docs/runtime.md).
