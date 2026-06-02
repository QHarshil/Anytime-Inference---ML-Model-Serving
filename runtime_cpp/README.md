# anytime_runtime (C++)

Minimal ONNX Runtime worker. Loads one or more ONNX models (FP32 and INT8
variants of the same architecture) and answers line-delimited JSON requests on
stdin. The Python control plane in `src/serving/onnx_runtime.py` spawns one or
more instances and dispatches requests across them.

## Build

```bash
# Download the ONNX Runtime release matching your platform from
# https://github.com/microsoft/onnxruntime/releases and unpack it.
export ONNXRUNTIME_ROOT_PATH=/path/to/onnxruntime-osx-arm64-1.16.0

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

The resulting binary is `build/anytime_runtime`. Point the Python client at it
via the `ANYTIME_RUNTIME_BIN` environment variable, or rely on the default
path lookup in `find_runtime_binary()`.

## Protocol

Handshake: the worker prints `ready\n` once every model has loaded.

Request (one JSON object per line):
```json
{"request_id":"r-1","variant":"fp32",
 "inputs":{"input_ids":{"shape":[1,128],"dtype":"int64","data":"<base64>"},
           "attention_mask":{"shape":[1,128],"dtype":"int64","data":"<base64>"}}}
```

Response:
```json
{"request_id":"r-1",
 "logits":{"shape":[1,2],"dtype":"float32","data":"<base64>"},
 "latency_ms":12.3}
```

## Notes

- Single inference per process. Run a pool from the Python side for parallelism.
- Supported input dtypes: `float32`, `int64`. Output dtype: `float32`.
