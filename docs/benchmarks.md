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
| Latency samples | 200 per pass, 3 independent passes per variant, 20 warmup iterations each |
| Deadline | 38.7 ms (3x the slowest frontier variant) |
| Measured through | the `anytime_runtime` extension, the same path the server uses |

Recorded in full in `results/variant_profiles.json`.

## Variant frontier

Service time is wall time per request through `RuntimeClient`, which is what the
pool actually spends. It is reported as the median of three independent
measurement passes; the spread column is the range across those passes.

| Variant | p50 | Spread | p95 | p99 | stdev | Accuracy | Size | Relative cost | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `distilbert_fp32` | 12.89 ms | 4.0% | 14.24 ms | 15.41 ms | 1.02 ms | 91.06% | 268 MB | 1.000 | frontier |
| `minilm_fp32` | 5.19 ms | 2.0% | 5.46 ms | 5.62 ms | 0.13 ms | 90.14% | 91 MB | 0.403 | frontier |
| `distilbert_int8` | 17.17 ms | 1.4% | 18.38 ms | 19.51 ms | 0.85 ms | 90.60% | 67 MB | 1.332 | dominated |
| `minilm_int8` | 7.16 ms | 0.9% | 7.45 ms | 7.75 ms | 0.22 ms | 90.14% | 23 MB | 0.556 | dominated |

The measured 91.06% for DistilBERT-SST-2 matches its published SST-2 dev accuracy
of roughly 91.3%, which is the main sanity check on the measurement path. All four
accuracies reproduce Stage 1 exactly, which is the check that the engine's logits
are the same logits.

Both INT8 variants are strictly dominated: slower with no accuracy gain. See
[`quantization.md`](quantization.md).

### Why three passes

One pass is not enough on this host. Measuring the same 200-request p50 eight
times gave 13.84 to 14.73 ms for DistilBERT and 5.62 to 5.83 ms for MiniLM, a
range of 6.5% and 3.6%, driven by thermal state and scheduler placement rather
than by anything in the code. The within-pass standard deviation is around 0.45 ms
and says nothing about that, so a single p50 quoted with a within-pass spread
overstates its own precision. Reporting the median of three passes, with the range
across them, is why the spread column exists.

This also sets the cross-check tolerance. Engine and separate session agreed to
within 2% here (0.980x to 1.007x), and the 15% bound is loose enough not to fire
on that drift while still catching anything structural.

## Load sweep

Offered load is expressed as a fraction of measured pool capacity
(4 workers / 12.89 ms = 310 rps) rather than as a bare request rate. Both
policies see the same Poisson arrival stream of real SST-2 sentences, tokenised
before the run so the measurement is inference rather than preprocessing. Three
seconds of arrivals per point, which is the script's default.

Goodput counts only requests that completed within the deadline; it is the metric
that matters, because admitted-and-late is worth nothing to a caller.

| ρ | Rate | Policy | Admitted | Attainment | p50 | p95 | Goodput | Cost |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.40 | 124 rps | accurate-only | 99.7% | 96.0% | 16.3 ms | 34.4 ms | 122 rps | 1.000 |
| 0.40 | 124 rps | adaptive | 99.7% | 97.9% | 15.6 ms | 31.0 ms | 124 rps | 1.000 |
| 0.60 | 186 rps | accurate-only | 94.4% | 85.1% | 24.1 ms | 53.0 ms | 153 rps | 1.000 |
| 0.60 | 186 rps | adaptive | 99.8% | 84.5% | 25.1 ms | 53.9 ms | 161 rps | 0.942 |
| 0.80 | 248 rps | accurate-only | 77.9% | 73.4% | 34.3 ms | 68.3 ms | 144 rps | 1.000 |
| 0.80 | 248 rps | adaptive | 99.9% | 86.5% | 29.1 ms | 52.8 ms | 217 rps | 0.790 |
| 0.95 | 295 rps | accurate-only | 18.4% | 79.0% | 30.9 ms | 53.3 ms | 43 rps | 1.000 |
| 0.95 | 295 rps | adaptive | 99.9% | 95.3% | 7.3 ms | 37.6 ms | 281 rps | 0.489 |
| 1.10 | 341 rps | accurate-only | 2.5% | 88.0% | 28.8 ms | 50.3 ms | 7 rps | 1.000 |
| 1.10 | 341 rps | adaptive | 99.9% | 99.5% | 6.3 ms | 24.5 ms | 335 rps | 0.415 |
| 1.30 | 403 rps | accurate-only | 0.4% | 100.0% | 17.0 ms | 21.1 ms | 2 rps | 1.000 |
| 1.30 | 403 rps | adaptive | 99.9% | 99.9% | 7.4 ms | 19.0 ms | 394 rps | 0.405 |

Repeating the whole sweep twice more with the same seed moved goodput by at most
6%, and the ρ = 0.95 adaptive figure by 2% (281, 286, 285 rps). The accurate-only
column is the noisier one, because it is computed from however few requests
admission happens to let through.

![Measured serving behaviour vs offered load](img/load_sweep.png)

### Reading the results

- **Below ρ ≈ 0.5 the policies are identical.** The adaptive planner routes 378
  of 378 requests to DistilBERT and matches the baseline. It does not degrade
  quality it does not need to.
- **The gap opens as the queue builds.** At ρ = 0.95 adaptive delivers 281 rps of
  goodput against 43 rps, at 49% of the compute cost, having moved 753 of 881
  requests to MiniLM.
- **The accuracy cost is bounded and small.** MiniLM measures 90.14% against
  DistilBERT's 91.06%, so the worst case is 0.92 accuracy points, and only for
  the shifted fraction.
- **At ρ = 0.60 the attainment curves cross.** Adaptive attains 84.5% against
  85.1%, marginally worse, while admitting 99.8% against 94.4%. Admitting nearly
  everything puts more work in the queue, so attainment among admitted requests
  does not improve; goodput still does, 161 rps against 153. This is worth stating
  because it is the one point where the adaptive policy is not uniformly better on
  every axis.
- **Attainment for `accurate-only` is noisy above ρ = 0.9** because it is
  conditioned on admitted requests, and very few are admitted there. At ρ = 1.30
  it admits 5 requests and all 5 hit, giving 100% attainment and 2 rps of
  goodput. This is why goodput, not attainment, is the headline.

## Reproducing

```bash
python scripts/export_onnx.py --task text     # export FP32 and INT8 variants
python scripts/profile_variants.py            # measure frontier, write serving.yaml
python scripts/run_load_sweep.py              # sweep load, write CSV and figure
```

Add `--quick` to either of the last two for a reduced run during development. Do
not report `--quick` numbers: it drops accuracy to 128 of the 872 validation
examples, which moves the measured accuracy by half a point.

Outputs land in `results/` and `docs/img/`. Both scripts are deterministic given a
seed except for wall-clock effects, which is why latency is reported as
percentiles over repeated passes rather than as single figures.

## Known limitations

- Single host, single task. No GPU, no multi-node.
- Absolute service times drift with thermal state by a few percent, which is why
  they are reported as a median over passes with the range attached. Ratios between
  variants are steadier than absolutes.
- Batch size is fixed at 1. There is no batching across requests yet, so these
  numbers describe one request in flight per worker.
- Attainment for `accurate-only` above ρ = 0.9 is conditioned on a small admitted
  sample and should not be read as a quality signal.

Closed since Stage 1:

- **Profiling no longer happens beside the serving path.**
  `scripts/profile_variants.py` measures through the same `RuntimeClient` the
  server dispatches to, scores accuracy through it, and cross-checks every variant
  against a separate ONNX Runtime session, aborting rather than writing numbers to
  disk when the two diverge by more than 15%.
- **The config, the sweep, and this page describe one measurement again.** All
  three were regenerated together through the engine. Service times reproduce
  Stage 1 within their own spread (12.89 against 12.79 ms for DistilBERT, 5.19
  against 5.21 ms for MiniLM) and accuracies reproduce exactly, so replacing the
  subprocess transport did not move the result.
