# Development

## Install

```bash
python3 -m venv venv && source venv/bin/activate
pip install -e ".[dev]"
```

Or run `./setup.sh`, which does the same.

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
| `needs_runtime` | requires the compiled C++ worker |

Tests skip cleanly rather than failing when an optional dependency is absent. The
serving tests build a tiny ONNX model on the fly, so they need neither torch nor
the C++ binary.

## Lint and types

```bash
ruff check .
ruff format --check .
mypy
```

Configuration lives in `pyproject.toml`. `mypy` runs over
`src/anytime_serving` only.

## Continuous integration

`.github/workflows/ci.yml` runs four jobs:

| Job | What it does |
| --- | --- |
| `lint` | ruff check, ruff format, mypy |
| `test` | full suite on Python 3.10 through 3.13 with `[bench]` |
| `test-minimal` | base dependencies only; asserts torch and pandas are absent, then runs the serving tests and boundary guards |
| `runtime` | builds the C++ worker against a pinned ONNX Runtime and checks the CLI contract |

`test-minimal` exists because the boundary it protects is easy to break by
accident and impossible to notice locally, where the research stack is installed.

The `runtime` job pins `ONNXRUNTIME_VERSION` to match the wheel. See
[`runtime.md`](runtime.md) for why that pin matters.

## Layout

```text
src/anytime_serving/
  serving/        load monitor, admission control, selector, runtime client, server
  planner/        offline deadline-aware planner and baselines
  models/         model zoo, cascade evaluator, quantisation helpers
  evaluation/     statistical analysis, Pareto frontiers, real inference
  profiler/       offline latency and accuracy profilers
  utils/          io, logging, metrics, visualisation
  workloads/      synthetic Poisson and bursty trace generators
runtime_cpp/      C++ ONNX Runtime worker (CMake project)
scripts/          export, profiling, load sweep, demo
experiments/      offline profiling and statistical evaluation pipeline
configs/          deadlines, model zoo, measured serving profiles
docs/             architecture, planner, runtime, quantisation, benchmarks
tests/            unit, integration, protocol, and import-boundary tests
```

## Generated files

`models/` and `results/` are ignored. Both are reproducible:

```bash
python scripts/export_onnx.py --task text     # models/
python scripts/profile_variants.py            # results/variant_profiles.json, configs/serving.yaml
python scripts/run_load_sweep.py              # results/load_sweep.csv, docs/img/
```

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
