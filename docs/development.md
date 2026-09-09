# Development

## Install

```bash
python3 -m venv venv && source venv/bin/activate
pip install -e ".[dev]"
```

Or run `./setup.sh`, which does the same.

Installing compiles the `anytime_runtime` extension, so a C++17 compiler and CMake
3.20 or newer are needed. The build runs through
[scikit-build-core](https://scikit-build-core.readthedocs.io/en/latest/) and the
bindings are [pybind11](https://pybind11.readthedocs.io/en/stable/). An
[ONNX Runtime](https://github.com/microsoft/onnxruntime/releases) SDK matching the
`onnxruntime` wheel is downloaded once into
`~/.cache/anytime-inference-planner/`; nothing needs to be fetched by hand. See
[`../runtime/README.md`](../runtime/README.md) for why the version is derived and not
pinned.

### Dependency groups

The serving control plane deliberately depends on very little. Extras add the
rest:

| Extra | Contents | Needed for |
| --- | --- | --- |
| *(base)* | numpy, onnx, onnxruntime, psutil, pyyaml | running the server |
| `bench` | matplotlib, pandas, scipy, seaborn | benchmark drivers, figures |
| `research` | torch, transformers, datasets, optimum, onnx-ir, pillow, tqdm | model export, quantisation, offline profiling |
| `dev` | everything plus ruff, mypy, pytest | contributing |

`tests/test_import_boundaries.py` enforces the base boundary: importing anything
on the serving path must not load torch or pandas. It imports the modules in a
subprocess and asserts neither appears in `sys.modules`. Both have leaked before,
torch through a module-scope import and pandas through a package `__init__`
re-export, so the guard is not hypothetical.

## Tests

```bash
pytest -q                          # everything available in this environment
pytest -q -m "not slow"            # skip long-running tests
pytest -q tests/test_admission.py  # one module
```

Markers, declared in `pytest.ini` with `--strict-markers`:

| Marker | Meaning |
| --- | --- |
| `slow` | takes more than a few seconds |
| `needs_torch` | requires torch, torchvision, or transformers |
| `needs_runtime` | requires the compiled `anytime_runtime` extension |

Tests skip cleanly when an optional dependency is absent, instead of failing. The
serving tests build a tiny ONNX graph on the fly, so they need neither torch nor a
compiled runtime.

Skipping is the wrong default for the cross-backend comparison in
`tests/test_runtime_engine.py`, which would silently pass by checking one backend
against itself. Set `ANYTIME_REQUIRE_BACKENDS` to the backends an environment is
supposed to provide and a missing one fails instead:

```bash
ANYTIME_REQUIRE_BACKENDS=extension,python pytest -q tests/test_runtime_engine.py
```

Two switches go the other way. They make an assertion stricter than the suite can
afford by default, because what they demand is a property of one machine and not of the
code.

| Variable | Demands |
| --- | --- |
| `ANYTIME_FIGURE_BYTES=1` | a redrawn figure matches the committed PNG byte for byte, not just in dimensions. A PNG's bytes follow the freetype its text was rasterised with. |
| `ANYTIME_BATCH_BITWISE=1` | a batched encoder row equals the same request run alone bit for bit, not within `tests/test_batching.py`'s reassociation bound. Padding changes the length of a float reduction, and a NEON reduction and an AVX one regroup it differently. |

Both are worth setting on the machine that produced the committed artefacts and
nowhere else. Both exist because a test that only passes where it was written was
shipped once each: the first cost a clean clone, the second cost a red CI run.

## A teardown abort that leaves every test passing

Running the `test-minimal` job's test list on macOS aborts at process teardown in about
3 of 10 runs. Seen on Python 3.12 and 3.14, and on ONNX Runtime 1.26.0 and 1.29.0:

```
libc++abi: terminating due to uncaught exception of type std::__1::system_error:
recursive_mutex lock failed: Invalid argument
```

Every test passes and pytest reports success. Exit code 134 is the only signal, so:

- Check `$?`, and tell 0, 134 and everything else apart. A loop that counts non-zero
  cannot tell an abort from a harness that never ran the tests.
- `pytest -q 2>&1 | tail -3` reports tail's exit code, not pytest's.
- In zsh an unquoted `$VAR` holding a list of paths is a single argument. `pytest -q
  $TESTS` then exits 4 with "file or directory not found", which looks like a failure
  if you only check the exit code. Use an array and `"${TESTS[@]}"`.

What is known so far:

- The full suite is much less affected than the minimal list. Running more tests after
  the minimal list makes the abort less likely, which is unexplained.
- Dropping `tests/test_batching.py` from the list still gives 2 of 10, so the abort is
  not confined to that module.
- Arms measured at 3/10, 1/10 and 2/10 cannot be separated at n = 10. Do not read an
  ordering into them.
- `recursive_mutex` is a C++ mutex, so `EINVAL` means locking storage that is already
  destroyed. Something touches ONNX Runtime state at static-destructor time.
- It has never fired on CI. The likeliest reason is that CI runs Linux while this
  message comes from libc++, so the next check is one CI run repeating the minimal
  list ten times.

The aborting environment, for comparison against a future one: Python 3.14.0, macOS
arm64, `pip install -e . pytest`, onnxruntime 1.29.0 for both the wheel and the linked
SDK, onnx 1.22.0, numpy 2.5.2, protobuf 7.36.0, pytest 9.1.1.

## Lint and types

```bash
ruff check .
ruff format --check .
mypy
```

Configuration lives in `pyproject.toml`. `mypy` runs over
`src/anytime_serving` only.

## Continuous integration

`.github/workflows/ci.yml` runs four jobs. Every one of them compiles the
extension, because installing the package is what builds it.

| Job | What it does |
| --- | --- |
| `lint` | ruff check, ruff format, mypy |
| `test` | full suite on Python 3.10 through 3.13 with `[bench]`; asserts the extension built |
| `test-minimal` | base dependencies only; asserts torch and pandas are absent, then runs the serving tests and boundary guards |
| `engine` | asserts the extension links the installed wheel, and compares it against the reference backend |

`test-minimal` exists because the boundary it protects is easy to break by
accident and impossible to notice locally, where the research stack is installed.

`engine` exists for the same reason in the other direction. The backend comparison
skips a backend that is not built, so a job that failed to build one would report
success having compared nothing; `ANYTIME_REQUIRE_BACKENDS` makes that a failure.
No job pins an ONNX Runtime version any more: the version is read from the
installed wheel at configure time, so the two copies in the process cannot drift
apart. See [`../runtime/README.md`](../runtime/README.md).

## Layout

```text
src/anytime_serving/
  serving/        load monitor, admission control, selector, runtime client,
                  server, and the decoder path: decoder client, KV admission
  planner/        offline deadline-aware planner and baselines
  models/         model zoo, cascade evaluator, quantisation helpers
  evaluation/     statistical analysis, Pareto frontiers, real inference
  profiler/       offline latency and accuracy profilers
  utils/          io, logging, metrics, visualisation
  workloads/      synthetic Poisson and bursty trace generators
runtime/          C++ engine, KV arena, and pybind11 bindings (built by pip install)
scripts/          export, profiling, load sweep, demo
experiments/      offline profiling and statistical evaluation pipeline
training/         fine-tuning entry points; not needed to serve or benchmark
configs/          serving.yaml: the measured frontier the planner reads
data/             dataset download helper; nothing here is versioned
docs/             the six pages this one belongs to, and img/ for their figures
tests/            unit, integration, engine-parity, and import-boundary tests
```

## Generated files

`models/` is ignored, and so is most of `results/`. Both are reproducible:

```bash
python scripts/export_onnx.py --task text     # models/, encoder variants, FP32 + dynamic INT8
python scripts/export_onnx.py --task text --quantization static
                                              # models/text_*_int8_static/; calibrated on SST-2 train
python scripts/export_onnx.py --task text --quantization static --calibration-method percentile
                                              # models/text_*_int8_static_percentile/
python scripts/export_decoder.py              # models/, decoder variants + results/decoder_profiles.json
python scripts/export_decoder.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --output results/decoder_profiles_tinyllama.json
python scripts/profile_variants.py            # results/variant_profiles.json, configs/serving.yaml
python scripts/profile_decode.py              # results/decode_profiles.json
python scripts/plot_decode_profiles.py        # docs/img/ decoder figures, from that JSON
python scripts/draw_arena_geometry.py         # docs/img/arena_geometry.png; a schematic, reads nothing
python scripts/run_load_sweep.py              # results/load_sweep.csv, docs/img/
python scripts/run_load_sweep.py --replot     # redraw docs/img/load_sweep.png from that CSV
python scripts/measure_session_sharing.py     # results/session_sharing.json; memory, not timing
python scripts/ab_session_sharing.py          # results/ab_session_sharing.json; paired, gated, needs a quiet host
python scripts/count_encoder_batching.py      # results/encoder_batching.json; counts and numerics, ungated
python scripts/profile_encoder_batching.py    # results/encoder_batching_timed.json; timed, gated, needs a quiet host
```

`export_onnx.py` reuses an FP32 graph that is already on disk instead of rebuilding
it. Every committed encoder number was measured against the graph that is there, and a
re-export differing by so much as a node would silently make those numbers describe
something else. Delete the directory to force a rebuild.

`export_decoder.py` takes `--model` and is model-agnostic. That was a claim until a
second model existed to check it against. It holds, with one defect found and fixed.
The KV geometry was read off the model config. It now reads the graph, the way
`DecoderSession::derive_geometry` does on the C++ side, and asserts the config agrees.
Give a second model its own `--output`, or the default overwrites the GPT-2 profile.

The last two are the two halves of the encoder-batching question and they are
separate on purpose. `count_encoder_batching.py` measures padding shares, run counts and
whether a batched answer differs from an unbatched one. Those are all properties of the
workload and the graph, so it carries no host gate and a contended machine cannot
corrupt it. `profile_encoder_batching.py` times a Run at several widths, so it does need
a gate. Every pass re-measures width 1 against `configs/serving.yaml` and a pass outside
the band is discarded. Splitting them is what let the counting half be finished on a
busy host.

Ten files under `results/` are committed, listed in `.gitignore` and totalling
240 KB. They hold the measurements every figure in `docs/img/` is drawn from, plus the
tables on `docs/benchmarks.md` that have no figure, meaning session sharing and encoder
batching. A checkout can then redraw each figure and check the numbers in the docs
against the data behind them, instead of taking both on trust. Everything else under
`results/` stays ignored: the A/B arm directories, the per-request CSVs, and the
inference cache.

`decoder_profiles.json` and `decode_profiles.json` are different files by one
letter: the first is what the export measured about each precision (size,
perplexity), the second is what the profiler measured about serving it (TTFT,
TPOT, the fitted cache cost).

### Exporting the decoder needs a dependency set the extra does not pin

`export_decoder.py` needs the `research` extra, and that is not sufficient. The
ONNX exporter moved out of `optimum` into `optimum-onnx`, so `optimum 2.2.0` has
no `optimum.exporters.onnx` at all, while `optimum-onnx 0.1.0` requires
`optimum~=2.1.0` and `transformers<4.58`. The only self-consistent set is:

```bash
pip install "optimum==2.1.0" "optimum-onnx==0.1.0" "transformers==4.57.*"
```

The `research` extra asks for `optimum>=1.20` and does not mention
`optimum-onnx`, so a fresh resolve lands on 2.2.0 and the export fails on an
import. That is what CI resolves, which is why nothing in CI exports anything and
why every decoder test builds a synthetic graph instead (`tests/conftest.py`).

On Python 3.14 the script also applies a shim to optimum before exporting, and it
is a no-op below that: CPython 3.14 made `functools.partial` a descriptor, and
optimum holds its decoder config factories as class-level partials, so they bind
`self` and fail. See `apply_partial_descriptor_shim`. Encoder export is
unaffected, which is why `export_onnx.py` never broke.

Figures referenced by the docs are committed under `docs/img/`.

## Offline pipeline

`run_all.py` drives the `experiments/` stages. Profiling and evaluation stages are
required, because later stages read their output; analysis stages are reported and
skipped on failure. `--quick-test` forwards `--quick` to every stage that accepts
it.

```bash
python run_all.py --quick-test
python run_all.py --skip-download --skip-profiling
```

This pipeline needs the `research` extra and downloads SST-2 and CIFAR-10 on
first run.
