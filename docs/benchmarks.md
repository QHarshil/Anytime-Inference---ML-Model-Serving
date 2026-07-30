# Benchmarks

Every number here is measured on the host recorded below. None is modelled,
assumed, or carried over from another machine.

## Host and configuration

| | |
| --- | --- |
| Platform | macOS 26.5.2, arm64 (Apple M4 Pro, 14 cores, 24 GB) |
| ONNX Runtime | 1.26.0, CPU execution provider |
| Python | 3.14.0 |
| Workers | 4, one intra-op thread each |
| Batch size | 1 |
| Sequence length | 128 tokens |
| Task | SST-2 binary sentiment |
| Accuracy split | GLUE SST-2 validation, all 872 examples |
| Latency samples | 200 per variant, after 20 warmup iterations |
| Deadline | 38.4 ms (3x the slowest frontier variant) |

Recorded in full in `results/variant_profiles.json`.

## Variant frontier

Service time is the median of 200 single-request measurements taken through a
session configured exactly like a serving worker.

| Variant | p50 | p95 | p99 | stdev | Accuracy | Size | Relative cost | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `distilbert_fp32` | 12.79 ms | 13.54 ms | 14.73 ms | 0.45 ms | 91.06% | 268 MB | 1.000 | frontier |
| `minilm_fp32` | 5.21 ms | 5.43 ms | 5.54 ms | 0.09 ms | 90.14% | 91 MB | 0.407 | frontier |
| `distilbert_int8` | 16.42 ms | 17.26 ms | 17.57 ms | 0.36 ms | 90.60% | 67 MB | 1.284 | dominated |
| `minilm_int8` | 6.93 ms | 7.07 ms | 7.19 ms | 0.08 ms | 90.14% | 23 MB | 0.542 | dominated |

The measured 91.06% for DistilBERT-SST-2 matches its published SST-2 dev accuracy
of roughly 91.3%, which is the main sanity check on the measurement path.

Both INT8 variants are strictly dominated: slower with no accuracy gain. See
[`quantization.md`](quantization.md).

## Load sweep

Offered load is expressed as a fraction of measured pool capacity
(4 workers / 12.79 ms = 313 rps) rather than as a bare request rate. Both
policies see the same Poisson arrival stream of real SST-2 sentences, tokenised
before the run so the measurement is inference rather than preprocessing.

Goodput counts only requests that completed within the deadline; it is the metric
that matters, because admitted-and-late is worth nothing to a caller.

| ρ | Rate | Policy | Admitted | Attainment | p50 | p95 | Goodput | Cost |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.40 | 125 rps | accurate-only | 99.6% | 99.2% | 16.8 ms | 28.7 ms | 126 rps | 1.000 |
| 0.40 | 125 rps | adaptive | 99.6% | 100.0% | 16.5 ms | 28.4 ms | 127 rps | 1.000 |
| 0.60 | 188 rps | accurate-only | 94.5% | 82.1% | 29.6 ms | 46.8 ms | 150 rps | 1.000 |
| 0.60 | 188 rps | adaptive | 100.0% | 87.8% | 27.0 ms | 44.7 ms | 169 rps | 0.941 |
| 0.80 | 250 rps | accurate-only | 74.8% | 51.7% | 38.2 ms | 52.6 ms | 99 rps | 1.000 |
| 0.80 | 250 rps | adaptive | 99.8% | 83.6% | 30.3 ms | 44.0 ms | 214 rps | 0.757 |
| 0.95 | 297 rps | accurate-only | 17.5% | 68.9% | 33.9 ms | 49.1 ms | 37 rps | 1.000 |
| 0.95 | 297 rps | adaptive | 99.8% | 95.0% | 9.4 ms | 38.4 ms | 287 rps | 0.483 |
| 1.10 | 344 rps | accurate-only | 3.6% | 76.0% | 30.5 ms | 47.3 ms | 10 rps | 1.000 |
| 1.10 | 344 rps | adaptive | 99.9% | 99.4% | 7.2 ms | 31.2 ms | 345 rps | 0.426 |
| 1.30 | 407 rps | accurate-only | 0.6% | 100.0% | 24.7 ms | 25.2 ms | 3 rps | 1.000 |
| 1.30 | 407 rps | adaptive | 99.9% | 100.0% | 8.1 ms | 17.9 ms | 403 rps | 0.411 |

![Measured serving behaviour vs offered load](img/load_sweep.png)

### Reading the results

- **Below ρ ≈ 0.5 the policies are identical.** The adaptive planner routes 253
  of 253 requests to DistilBERT and matches the baseline. It does not degrade
  quality it does not need to.
- **The gap opens as the queue builds.** At ρ = 0.95 adaptive delivers 287 rps of
  goodput against 37 rps, at 48% of the compute cost, having moved 526 of 603
  requests to MiniLM.
- **The accuracy cost is bounded and small.** MiniLM measures 90.14% against
  DistilBERT's 91.06%, so the worst case is 0.92 accuracy points, and only for
  the shifted fraction.
- **Attainment for `accurate-only` is noisy above ρ = 0.9** because it is
  conditioned on admitted requests, and very few are admitted there. At ρ = 1.30
  it admits 5 requests and all 5 hit, giving 100% attainment and 3 rps of
  goodput. This is why goodput, not attainment, is the headline.

## Reproducing

```bash
python scripts/export_onnx.py --task text     # export FP32 and INT8 variants
python scripts/profile_variants.py            # measure frontier, write serving.yaml
python scripts/run_load_sweep.py              # sweep load, write CSV and figure
```

Add `--quick` to any of the last two for a reduced run during development.
Outputs land in `results/` and `docs/img/`. Both scripts are deterministic given
a seed except for wall-clock effects, which is why latency is reported as
percentiles over hundreds of samples rather than as single figures.

## Known limitations

- Single host, single task. No GPU, no multi-node.
- The figure above was produced with the Stage 1 subprocess worker, which has since
  been replaced by the in-process engine. The two agreed bitwise on logits and
  within 0.4% on inference time, so the shape of the result stands, but the sweep
  has not been re-run through the engine.
- `configs/serving.yaml` still holds the service times measured through a separate
  ONNX Runtime session. Re-running `scripts/profile_variants.py` now measures
  through the engine and reports wall time per request rather than time inside a
  bare session, which moves the numbers by roughly 4%. The config, the sweep, and
  this page should be regenerated together so they continue to describe one
  measurement.

Closed since Stage 1: service profiles are no longer measured beside the serving
path. `scripts/profile_variants.py` measures through the same `RuntimeClient` the
server dispatches to, and cross-checks every variant against a separate ONNX
Runtime session, failing the run rather than writing numbers to disk when the two
diverge by more than 15%. Measured agreement on this host is within 1.6%.
