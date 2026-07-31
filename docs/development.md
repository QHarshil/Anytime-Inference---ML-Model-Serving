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
[`../runtime/README.md`](../runtime/README.md) for why the version is derived rather
than pinned.

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

Tests skip cleanly rather than failing when an optional dependency is absent. The
serving tests build a tiny ONNX graph on the fly, so they need neither torch nor a
compiled runtime.

Skipping is the wrong default for the cross-backend comparison in
`tests/test_runtime_engine.py`, which would silently pass by checking one backend
against itself. Set `ANYTIME_REQUIRE_BACKENDS` to the backends an environment is
supposed to provide and a missing one fails instead:

```bash
ANYTIME_REQUIRE_BACKENDS=extension,python pytest -q tests/test_runtime_engine.py
```

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

`models/` and `results/` are ignored. Both are reproducible:

```bash
python scripts/export_onnx.py --task text     # models/, encoder variants
python scripts/export_decoder.py              # models/, decoder variants + results/decoder_profiles.json
python scripts/profile_variants.py            # results/variant_profiles.json, configs/serving.yaml
python scripts/profile_decode.py              # results/decode_profiles.json
python scripts/run_load_sweep.py              # results/load_sweep.csv, docs/img/
```

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
