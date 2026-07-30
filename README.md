# Anytime Inference Planner

Latency-bounded ML serving. A Python control plane watches queue depth and CPU
load, then routes each request to one of several profiled model variants so that
as much traffic as possible completes inside a deadline. Inference runs in a C++
ONNX Runtime worker pool.

Variant selection is a constrained optimisation: maximise expected accuracy
subject to an M/M/c sojourn-time bound and the measured backlog.

![Measured serving behaviour vs offered load](docs/img/load_sweep.png)

## Results

DistilBERT-SST-2 and MiniLM-L6 served from a 4-worker pool on an Apple M4 Pro,
real SST-2 validation traffic, 38 ms deadline. Offered load is a fraction of the
measured pool capacity of 313 rps. Goodput counts only requests that completed
within the deadline.

| Offered load | Goodput, accurate-only | Goodput, adaptive | Compute cost | p95 |
| --- | --- | --- | --- | --- |
| ρ = 0.40 | 126 rps | 127 rps | 1.00 | 28 ms |
| ρ = 0.80 | 99 rps | 214 rps | 0.76 | 44 ms |
| ρ = 0.95 | 37 rps | 287 rps | 0.48 | 38 ms |
| ρ = 1.30 | 3 rps | 403 rps | 0.41 | 18 ms |

Below ρ ≈ 0.5 the planner keeps every request on the most accurate variant and
matches the baseline exactly. As the queue builds it shifts traffic to the cheaper
variant: at ρ = 0.95 that is **7.8x the goodput at 48% of the compute cost**, for
at most 0.92 accuracy points (91.06% to 90.14% on SST-2 validation).

Full tables, methodology, and host details: [`docs/benchmarks.md`](docs/benchmarks.md).

## Install

```bash
python3 -m venv venv && source venv/bin/activate
pip install -e ".[dev]"
```

Extras: `bench` for benchmark drivers and figures, `research` for model export and
offline profiling, `dev` for everything plus lint, types, and tests.

## Quick start

```bash
python scripts/demo_serving.py
```

Self-contained: builds a small synthetic ONNX model, drives Poisson traffic
through both policies, and prints a JSON summary. Needs no model download and no
C++ build.

Reproduce the measured results end to end:

```bash
python scripts/export_onnx.py --task text     # export FP32 and INT8 variants
python scripts/profile_variants.py            # measure the Pareto frontier
python scripts/run_load_sweep.py              # sweep load, write the figure
```

## Tests

```bash
pytest -q
```

The serving tests build a tiny ONNX model on the fly, so they run without torch
or the C++ binary. Tests needing either are marked and skip when it is absent.

## Documentation

| | |
| --- | --- |
| [Architecture](docs/architecture.md) | control plane and worker split, request lifecycle |
| [Planner](docs/planner.md) | admission control, Erlang-C, variant selection |
| [Runtime](docs/runtime.md) | C++ worker, build, wire protocol |
| [Quantisation](docs/quantization.md) | why the variant frontier is measured, not assumed |
| [Benchmarks](docs/benchmarks.md) | methodology, full results, reproduction |
| [Development](docs/development.md) | dependency groups, tests, CI, layout |

## Notes

Two findings shaped the current design and are worth knowing before extending it:

- **Worker count belongs in the queueing model.** Modelling a c-worker pool as a
  single server understates capacity by a factor of c. `AdaptiveServer` now
  refuses a selector that models a different pool size.
- **Match the C++ ONNX Runtime version to the Python wheel.** A mismatch measured
  DistilBERT at 98.9 ms in the worker against 13.0 ms in the profiler, which
  silently invalidated every admission decision.

## License

MIT. See [LICENSE](LICENSE).
