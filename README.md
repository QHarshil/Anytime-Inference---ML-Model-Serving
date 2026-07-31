# Anytime Inference Planner

[![CI](https://github.com/QHarshil/Anytime-Inference-ML-Model-Serving/actions/workflows/ci.yml/badge.svg)](https://github.com/QHarshil/Anytime-Inference-ML-Model-Serving/actions/workflows/ci.yml)

Latency-bounded ML serving on CPU. A Python control plane watches queue depth and
CPU load, then routes each request to one of several profiled model variants so that
as much traffic as possible completes inside a deadline: maximise expected accuracy
subject to an M/M/c sojourn-time bound and the measured backlog. Inference runs
in-process in a C++ ONNX Runtime engine.

At 95% of measured pool capacity that buys **6.5x the goodput at 49% of the compute
cost** — 281 requests per second against 43 — for at most 0.92 accuracy points, on a
4-worker pool on an Apple M4 Pro against a 39 ms deadline. Below half capacity it
keeps every request on the most accurate variant and matches the baseline exactly.

A second lane decodes. GPT-2 is exported with its KV cache in the graph signature and
that cache is held in a fixed arena of blocks, so admission can refuse a sequence it
cannot hold and eviction can pick a victim on evidence rather than on hope. Time to
first token and time per output token are 39x apart at FP32, so they are never
reported as one number. There is no batching across requests yet, and the decoder
lane is not wired into the server.

![Measured serving behaviour vs offered load](docs/img/load_sweep.png)

## Results

DistilBERT-SST-2 and MiniLM-L6 served from a 4-worker pool on an Apple M4 Pro,
real SST-2 validation traffic, 39 ms deadline. Offered load is a fraction of the
measured pool capacity of 310 rps. Goodput counts only requests that completed
within the deadline; p95 is the adaptive policy's.

| Offered load | Goodput, accurate-only | Goodput, adaptive | Compute cost | p95 |
| --- | --- | --- | --- | --- |
| ρ = 0.40 | 122 rps | 124 rps | 1.00 | 31 ms |
| ρ = 0.80 | 144 rps | 217 rps | 0.79 | 53 ms |
| ρ = 0.95 | 43 rps | 281 rps | 0.49 | 38 ms |
| ρ = 1.30 | 2 rps | 394 rps | 0.41 | 19 ms |

Below ρ ≈ 0.5 the two policies are indistinguishable. As the queue builds the planner
shifts traffic to the cheaper variant, and the accuracy it gives up is bounded by the
gap between the variants: 91.06% against 90.14% on SST-2 validation, and only for the
shifted fraction.

Measured through the serving path itself, with every variant cross-checked against
a separate ONNX Runtime session. Repeating the sweep moved goodput by at most 6%,
and service times carry a 1-4% run-to-run spread, so treat the third digit as
noise.

### The decoder is two measurements, not one

GPT-2 124M through the block-allocated KV cache. A 1024-token prompt is prefilled in
chunks of 256; TPOT is measured against 960 cached tokens, where a decode step costs
the most.

| Precision | Size | TTFT | TPOT | TTFT / TPOT | Cost of block accounting |
| --- | --- | --- | --- | --- | --- |
| `fp32` | 653 MB | 372 ms | 9.54 ms | 39.0x | 11.1% |
| `int8` | 399 MB | 346 ms | 8.45 ms | 40.9x | 12.6% |
| `int4` | 367 MB | 1772 ms | 16.63 ms | 106.6x | 6.5% |

INT8 wins on every axis here, which contradicts the encoder finding on the same host
and does not overturn it: a decode step is a matrix-vector product bound by weight
bandwidth, and an encoder at sequence length 128 is not. The arena is not a speedup
and is not offered as one — feeding the graph's own `present` tensors back costs no
gather at all. What blocks buy is an occupancy number a policy can act on, and the
last column is what that costs.

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
through both policies, and prints a JSON summary. Needs no model download.

Reproduce the measured results end to end:

```bash
python scripts/export_onnx.py --task text     # export FP32 and INT8 variants
python scripts/profile_variants.py            # measure the Pareto frontier
python scripts/run_load_sweep.py              # sweep load, write the figure
```

The decoder path, which needs the `research` extra for the export:

```bash
python scripts/export_decoder.py              # GPT-2 in FP32, INT8, INT4
python scripts/profile_decode.py              # TTFT and TPOT through the KV cache
```

## Tests

```bash
pytest -q
```

The serving tests build a tiny ONNX graph on the fly, so they run without torch, and
the decoder tests build a synthetic decoder graph for the same reason. Tests needing
the compiled extension are marked and skip when it is absent — except where a skip
would mean comparing a backend against itself:

```bash
ANYTIME_REQUIRE_BACKENDS=extension,python pytest -q
```

turns a missing backend into a failure instead.

## Documentation

| | |
| --- | --- |
| [Architecture](docs/architecture.md) | control plane and worker split, request lifecycle, related work |
| [Planner](docs/planner.md) | admission control, Erlang-C, variant selection, KV-block admission |
| [Runtime](docs/runtime.md) | C++ engine, block-allocated KV cache, build, version matching |
| [Quantisation](docs/quantization.md) | why the variant frontier is measured, not assumed |
| [Benchmarks](docs/benchmarks.md) | methodology, full results, reproduction |
| [Development](docs/development.md) | dependency groups, tests, CI, layout |

## Notes

Four findings shaped the current design and are worth knowing before extending it:

- **Worker count belongs in the queueing model.** Modelling a c-worker pool as a
  single server understates capacity by a factor of c. `AdaptiveServer` now
  refuses a selector that models a different pool size.
- **Match the C++ ONNX Runtime version to the Python wheel.** A mismatch measured
  DistilBERT at 98.9 ms in the worker against 13.0 ms in the profiler, which
  silently invalidated every admission decision. The version is now derived from
  the installed wheel at build time and checked again at import, so it cannot
  drift.
- **Measure through the serving path, not beside it.** That mismatch survived
  because profiling used a separate process. Profiling now runs through the same
  client the server dispatches to, and fails if a cross-check disagrees.
- **Assert what is derivable, not what happens to hold.** One session with its KV
  cache held two ways must agree bitwise, because only the source of the bytes
  differs. Two ONNX Runtime builds must not be held to that: the extension links its
  own SDK, the wheel ships another, and on x86-64 they dispatch to different kernels,
  measured at seven float32 ULP. Asserting bitwise there passed on arm64 and reddened
  every x86-64 CI job. Token identity is the portable hard assertion.

## License

MIT. See [LICENSE](LICENSE).
