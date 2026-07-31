# anytime_runtime (C++ extension)

In-process [ONNX Runtime](https://onnxruntime.ai/docs/) engine, exposed to Python
through [pybind11](https://pybind11.readthedocs.io/en/stable/). Replaces the Stage 1
subprocess worker: the transport is a function call, so there is no JSON framing, no
base64, and no copy in either direction.

## Build

The extension is built by installing the package.
[`scikit-build-core`](https://scikit-build-core.readthedocs.io/en/latest/) drives
CMake, so there is no separate build step:

```bash
pip install -e .
```

`onnxruntime` must already be installed in the target environment, because the
extension is linked against the version that environment will import.

## The version is derived, never written down twice

`cmake/ResolveOnnxRuntime.cmake` reads `onnxruntime.__version__` from the target
interpreter and resolves an SDK to match:

- `ONNXRUNTIME_ROOT_PATH`, if set, must contain a `VERSION_NUMBER` equal to the
  wheel version. Configuration fails otherwise.
- Otherwise the matching release archive is downloaded once into
  `~/.cache/anytime-inference-planner/` and reused. `ANYTIME_ORT_CACHE` overrides
  the location.

This matters more than it looks. Stage 1 built the worker against 1.20.1 while the
Python side used the 1.26.0 wheel and measured DistilBERT at **98.9 ms against
13.0 ms** inside `session->Run()`, a 7.6x gap that made every admission decision
wrong. Nothing failed; the numbers were just false. With the extension both copies
of the library are loaded into one process, so the equality is enforced in three
places:

1. **Configure time.** The SDK version must equal the wheel version.
2. **Compile time.** `ORT_API_VERSION` is read out of the resolved headers and
   asserted in `src/tensor.cpp`, so a header from another include path fails the
   build rather than the run.
3. **Import time.** `load_extension()` in `serving/onnx_runtime.py` compares
   `anytime_runtime.onnxruntime_version()` against `onnxruntime.__version__` and
   raises on a mismatch, which covers an extension carried into a different
   environment.

The probe deliberately clears `PYTHONPATH`. pip builds in an isolated environment
layered onto the target interpreter through that variable, and reading its
onnxruntime instead of the target's would reintroduce the mismatch through the
build system.

## Layout

```text
include/anytime/tensor.hpp     element types, borrowed tensor views
include/anytime/engine.hpp     sessions, input filtering, one run
include/anytime/kv_cache.hpp   the block arena, gather and scatter
include/anytime/decoder.hpp    prefill/decode over that arena
src/                           implementations
bindings/module.cpp            the pybind11 module
cmake/                         ONNX Runtime resolution
```

Headers arrive as the work that needs them lands, rather than as empty
placeholders: the scheduler and batch assembly are not here yet.

## Behaviour

- **Tensors are borrowed, not copied.** Inputs point into the numpy buffer, whose
  references the bindings hold for the whole call. Outputs are numpy views over
  ONNX Runtime's own buffers, kept alive by a capsule owning the `Ort::Value`. A
  response therefore holds runtime memory until it is dropped.
- **A non-contiguous input is made contiguous.** ONNX Runtime reads the buffer
  directly, so a strided array would otherwise be misread.
- **The GIL is released around inference.** Without that a pool would serialise on
  the interpreter lock, and the M/M/c model the admission controller uses would
  describe a machine that does not exist.
- **One session set per engine.** The Python pool holds one engine per slot, so N
  workers stay N independent single-threaded servers. Intra-op and inter-op
  threads default to one each, matching how the service times in
  `configs/serving.yaml` were measured.
- **Input filtering.** Variants of one task can declare different inputs, so
  callers pass the union and each graph takes the subset it declares. A declared
  input that is missing raises instead, since running on a partial feed would
  silently produce wrong output.
- **Error contract.** Anything meaning "the runtime could not serve this request"
  (an unknown variant, a missing input) raises `RuntimeError`. A malformed
  argument (a dtype the engine does not accept) raises `ValueError`.
  `CacheExhausted` derives from `RuntimeError` and means specifically "the arena has
  no room", which is the one such failure a policy is expected to handle by evicting
  rather than propagate.
- **Supported input dtypes.** float32, float64, int32, int64, bool. Outputs are
  mapped back from the same set. The KV arena is float32, which is what weight-only
  quantisation leaves the cache as; a graph declaring a narrower cache is refused
  rather than reinterpreted.

## The decoder path

`DecoderSession` runs a decoder-only graph over a fixed arena of KV blocks. It is a
host-side block allocator and not paged attention, for the reason at the top of
`include/anytime/kv_cache.hpp`: ONNX Runtime allocates the `present` tensors itself,
so there is no block table to hand the graph.

```python
import anytime_runtime as ar

session = ar.DecoderSession("model.onnx", block_tokens=64, num_blocks=24)
session.geometry  # read off the graph, not from a config
session.open("seq", reserve_tokens=1024)
result = session.prefill("seq", prompt)  # chunked at 256 by default
result.logits  # the next token's row, only
result.timings.gather_ms  # what block accounting cost
session.decode("seq", token)
session.release("seq")  # blocks back; tokens are the caller's
```

- **`open` refuses, it does not raise.** Returning `False` is the admission
  controller's answer. A sequence that outgrows its reservation mid-decode raises
  `CacheExhausted` instead, because by then something has already promised it room.
- **The arena is fixed at construction** and zero-filled, so its pages are resident
  before the first run rather than faulting in during the opening decode steps.
- **Gather is timed, not assumed.** `StepTimings` breaks out gather, run, scatter and
  the once-per-sequence invariant check separately. On GPT-2 the gather is 4% of a
  decode step at 128 cached tokens and 11% at 960.
- **Only the new tail is scattered back.** That rests on `present` beginning with the
  `past` it was given, which is verified once per sequence and raises on mismatch.
- **One arena per session, shared by every sequence in it.** `open` registers a
  sequence and takes its blocks from the same pool as every other, so several
  sequences are already accounted against one fixed budget. What is not shared is
  the session: the Python pool holds one per worker slot. A scheduler that gathers
  several sequences into a single run needs one session driving one arena instead,
  which is a change to the concurrency model rather than to the allocator. The
  policy half lives in `src/anytime_serving/serving/kv_admission.py`.

## Tests

`tests/test_runtime_engine.py` compares the extension against the backend it
replaces on a graph with real arithmetic in it, and asserts they agree bitwise.
`ANYTIME_REQUIRE_BACKENDS` turns a missing backend from a skip into a failure, so
the comparison cannot decay into a backend checked against itself.

That bitwise assertion is worth reading precisely, because it does not generalise.
This extension links its own ONNX Runtime SDK and the wheel ships a separate build
of the same version, so on x86-64 the two can dispatch to different MLAS kernels.
A 4x4 matmul with a bias and a Relu gives them nothing to disagree about, and the
assertion is kept there because it catches tensor-path corruption sharply. The
decoder fixture's reduction does give them something to disagree about, measured at
around seven float32 ULP, so those comparisons are held to token identity plus
float32 agreement instead. Within one instance -- the same session with its cache
held two ways -- bitwise is the right bar and stays. See the same distinction in
[`../docs/runtime.md`](../docs/runtime.md).

`tests/test_kv_cache.py` and `tests/test_decoder_session.py` cover the decoder path
against the contiguous cache it replaces, on the synthetic graph in
`tests/conftest.py` and on the exported GPT-2 graph when that is on disk.

```bash
pytest -q tests/test_runtime_engine.py tests/test_kv_cache.py tests/test_decoder_session.py
ANYTIME_REQUIRE_BACKENDS=extension,python pytest -q tests/test_runtime_engine.py
```
