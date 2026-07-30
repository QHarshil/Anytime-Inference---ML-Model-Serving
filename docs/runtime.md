# Inference runtime

`runtime/` builds `anytime_runtime`, a pybind11 extension that runs ONNX Runtime in
the caller's process. `RuntimePool` in `src/anytime_serving/serving/onnx_runtime.py`
holds one engine per worker slot and dispatches across them.

Build and layout details are in [`../runtime/README.md`](../runtime/README.md).
This page is about the two decisions that shape everything downstream.

## Build

```bash
pip install -e .
```

That is the whole thing. `scikit-build-core` runs CMake, and CMake resolves an ONNX
Runtime SDK matching the installed `onnxruntime` wheel, downloading it once into
`~/.cache/anytime-inference-planner/` if it is not already there.

## Match the ONNX Runtime version to the Python wheel

This is the single most important build detail, and it is now enforced rather than
documented.

Building the Stage 1 worker against 1.20.1 while the Python side used the 1.26.0
wheel measured DistilBERT at **98.9 ms inside the worker against 13.0 ms in the
profiler**, a 7.6x gap. The timing was taken around `session->Run()`, so this was
not transport overhead; the older release simply lacks the optimisations the newer
one applies to this graph on arm64. Because the profiler and the serving path
disagreed, every service time the planner used was wrong by almost an order of
magnitude and no request met its deadline. Nothing crashed and no test failed. The
numbers were just false.

With the extension, both copies of ONNX Runtime are loaded into the same process,
so this stopped being a performance question and became a correctness one. The
version is therefore never written down twice. It is read from the wheel the target
interpreter will import, and checked in three places:

1. **Configure time.** The SDK version must equal the wheel version, or CMake
   fails.
2. **Compile time.** `ORT_API_VERSION` is read out of the resolved headers and
   asserted in `src/tensor.cpp`, so a header from a different include path breaks
   the build rather than the run.
3. **Import time.** `load_extension()` compares
   `anytime_runtime.onnxruntime_version()` against `onnxruntime.__version__`,
   which covers an extension carried into an environment with a different wheel.

A related consequence: CI no longer pins a version anywhere. It had been building
the C++ side against 1.26.0 while `pip` installed 1.28.0, since the dependency
floor is `>=1.16`. That was harmless only because the two never shared a process.

## Cost of the transport it replaced

Measured on DistilBERT at batch size one, 60 requests after warm-up:

| Backend | Inference p50 | Wall p50 | Transport |
| --- | --- | --- | --- |
| `python` | 13.667 ms | 13.673 ms | 0.006 ms |
| `extension` | 13.609 ms | 13.628 ms | 0.018 ms |
| `subprocess` | 13.629 ms | 13.926 ms | 0.296 ms |

All three agreed bitwise on the logits, and their inference times agreed within
0.4%. That agreement is what a matched version looks like, and it is the check that
was missing in Stage 1.

Transport fell from 0.296 ms to 0.018 ms. Against 14 ms of inference that saving is
not why the change was made: a subprocess boundary rules out batching across
requests and sharing a KV cache between them, which is what the rest of Stage 2
needs. The `subprocess` row is from the Stage 1 worker, which was removed once it
had served as the reference the extension was validated against.

## Behaviour

- **Tensors are borrowed, not copied.** Inputs point into the numpy buffer;
  outputs are numpy views over ONNX Runtime's buffers, kept alive by a capsule
  owning the `Ort::Value`. A response therefore holds runtime memory until it is
  dropped, so copy `logits` if it needs to outlive the request.
- **The GIL is released during inference.** Without that a pool would serialise on
  the interpreter lock, and the M/M/c model in [`planner.md`](planner.md) would
  describe a machine that does not exist.
- **One engine per worker.** Intra-op threads are set to 1 so that N workers behave
  as N independent servers, which is what makes that queueing model valid, and
  matches how the service times in `configs/serving.yaml` were measured.
- **Input filtering.** Variants can declare different inputs, so callers pass the
  union and each graph takes the subset it declares. A declared input that is
  missing is an error rather than a run on a partial feed.
- **Error contract.** An unknown variant or a missing declared input raises
  `RuntimeError`; a dtype the engine does not accept raises `ValueError`. Both
  backends behave identically, which is what makes them comparable.
- **Supported dtypes.** Inputs and outputs: float32, float64, int32, int64, bool.

## Backends

`RuntimeClient` picks the extension when it imports and matches the wheel, and
otherwise warns and falls back to ONNX Runtime's own Python API. The fallback exists
so the control plane runs where the extension has not been built; it is not the
serving path, and a measurement taken through it does not describe one.

`backend="extension"` or `backend="python"` pins the choice. Tests pin it so a
parity failure cannot hide behind a silent fallback.

## Tests

```bash
pytest -q tests/test_runtime_engine.py
ANYTIME_REQUIRE_BACKENDS=extension,python pytest -q tests/test_runtime_engine.py
```

`tests/test_runtime_engine.py` asserts the backends agree bitwise on a graph with
real arithmetic in it, that outputs are views rather than copies, that a strided
input is made contiguous rather than misread, and that an unsupported dtype is
refused. `ANYTIME_REQUIRE_BACKENDS` turns a missing backend from a skip into a
failure, so the comparison cannot decay into one backend checked against itself.
