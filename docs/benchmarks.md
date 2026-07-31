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
| Task | [SST-2](https://huggingface.co/datasets/stanfordnlp/sst2) binary sentiment |
| Accuracy split | [GLUE](https://gluebenchmark.com/) SST-2 validation, all 872 examples |
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

The measured 91.06% for
[DistilBERT-SST-2](https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english)
matches the roughly 91.3% its model card reports on SST-2 dev, which is the main
sanity check on the measurement path. All four accuracies reproduce Stage 1 exactly,
which is the check that the engine's logits are the same logits.

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

## Decoder: prefill and decode are two different measurements

GPT-2 124M through the block-allocated KV cache, measured by
`scripts/profile_decode.py` through the same `DecoderClient` that serves decoding.
Recorded in full in `results/decode_profiles.json`.

The headline is the gap. Time to first token and time per output token differ by a
factor of 39 at FP32 and 107 at INT4, so a single latency figure would describe
neither phase.

| Precision | TTFT, 1024-token prompt | TPOT, 960 cached | TTFT / TPOT | Arena cost |
| --- | --- | --- | --- | --- |
| `fp32` | 372.2 ms (0.5%) | 9.54 ms (0.3%) | 39.0x | 11.1% |
| `int8` | 345.6 ms (0.8%) | 8.45 ms (0.2%) | 40.9x | 12.6% |
| `int4` | 1772.3 ms (0.2%) | 16.63 ms (0.5%) | 106.6x | 6.5% |

TTFT is a chunked prefill at the 256-token default; the percentage is the range
across three passes. "Arena cost" is the gather plus scatter as a share of the
decode step -- what block accounting costs.

### TPOT grows with the cache, so one figure is not enough

A decode step re-reads the whole cache before it runs, so its cost is a line in the
number of cached tokens rather than a constant:

| Cached tokens | Cache size | `fp32` TPOT | Gather | Scatter | Arena cost |
| --- | --- | --- | --- | --- | --- |
| 128 | 9 MB | 4.83 ms | 0.185 ms | 0.009 ms | 4.0% |
| 512 | 38 MB | 7.22 ms | 0.600 ms | 0.015 ms | 8.5% |
| 960 | 71 MB | 9.54 ms | 1.044 ms | 0.018 ms | 11.1% |

The gather is pure `memcpy` and comes out the same at every precision -- 0.19, 0.59
and 1.05 ms at the three lengths -- because it moves the same bytes whatever the
weights are. 71 MB in 1.044 ms is 68 GB/s, close to this host's memory bandwidth.
The scatter stays near zero because it writes only the new token rather than the
whole `present` tensor, which is worth about 1 ms a step at full context.

So the price of block accounting is 4% of a decode step at short context and 11-13%
at long. `scripts/profile_decode.py` also runs the same generation over contiguous KV
and fails rather than reporting anything if the two disagree on the tokens they emit,
or if time inside `Session::Run` differs by more than 15%. Measured, the two agree on
tokens exactly and on graph time to within 3% (0.995x to 1.029x).

`kv_admission.CacheCost` is fitted to these points rather than written down: FP32
comes out at 4.18 ms plus 0.00565 ms per cached token, which reproduces all three
measurements to within 0.14 ms. Recompute is 0.35 ms per token, fitted from the
chunked prefill because that is the width a resume actually runs at. Drawing it from
the single-pass sweep beside it instead would overstate every recompute by 13% and
leave the eviction policy needlessly unwilling to act.

The whole run was repeated to check it. After a two-minute pause every figure
reproduced within 2.5% -- FP32 TPOT at 960 cached read 9.54 then 9.46 ms, INT8 8.45
then 8.43. Started back to back with no pause it drifts up to 10%, which is worth
knowing before comparing two runs.

### Chunked prefill is faster and smaller

Splitting a prefill into chunks re-reads the growing cache, so it ought to cost more.
It does not, because a single pass over 1024 tokens also allocates logits for every
position -- 206 MB -- when sampling reads one row of them. Chunking a prefill is the
first half of [SARATHI](https://arxiv.org/abs/2308.16369); the second half, sharing
a run with decode steps, is not reachable over this graph, for the reason in
[`architecture.md`](architecture.md).

| Prefill width | `fp32` TTFT | vs one pass | Peak logits |
| --- | --- | --- | --- |
| one pass | 433.6 ms | 1.000x | 206 MB |
| 512 | 388.4 ms | 0.896x | 103 MB |
| 256 | 372.2 ms | **0.858x** | 52 MB |
| 128 | 391.3 ms | 0.902x | 26 MB |

256 tokens is the default. It is clearly fastest for FP32 and INT4; for INT8, 128 and
256 are indistinguishable (345.4 against 344.1 ms in one run and a dead heat in the
other), so 256 is chosen for the two reasons that are not about speed: at four blocks
of 64 it aligns with the allocator, and it gives the scheduler a preemption point
inside a long prefill.

The gain is 1.17x at FP32 and 1.19x at INT8 but only 1.02x at INT4, where the cost is
unpacking 4-bit weights once per run and more runs multiply it. Going below 256 stops
helping for the same reason at every precision: eight passes over a 1024-token prompt
re-read the growing cache eight times.

This was checked twice with the chunk widths interleaved, because the first ordering
measured them in sequence and could have been recording thermal drift.

### INT8 is not dominated on the decoder

Stage 1 found INT8 strictly dominated on this host, and
[`quantization.md`](quantization.md) records it. That was an encoder at sequence
length 128. Decode at sequence length 1 is a different kernel regime -- a
matrix-vector product bound by weight bandwidth, where moving a quarter of the bytes
helps -- and there INT8 wins:

| Precision | TPOT at 128 cached | at 512 | at 960 | vs FP32 |
| --- | --- | --- | --- | --- |
| `fp32` | 4.83 ms | 7.22 ms | 9.54 ms | 1.000x |
| `int8` | 3.60 ms | 5.79 ms | 8.45 ms | 0.75x to 0.89x |
| `int4` | 11.56 ms | 13.85 ms | 16.63 ms | 1.74x to 2.39x |

INT8 leads by 25% at 128 cached tokens and 11% at 960; the advantage narrows as cache
reads take a larger share of the step. At +0.063 perplexity for 0.61x the graph size,
INT8 is the better decode variant on this host on every axis.

INT4 is still dominated, but the penalty shrinks from 4.8x on prefill to 1.7x on
decode, in the same direction and for the same reason. It buys memory and nothing
else, which is what the export measured and this does not change.

## Reproducing

```bash
python scripts/export_onnx.py --task text     # export FP32 and INT8 variants
python scripts/profile_variants.py            # measure frontier, write serving.yaml
python scripts/run_load_sweep.py              # sweep load, write CSV and figure
python scripts/export_decoder.py              # export GPT-2, measure perplexity
python scripts/profile_decode.py              # measure TTFT and TPOT
```

Add `--quick` to any of these for a reduced run during development. Do not report
`--quick` numbers: for `profile_variants.py` it drops accuracy to 128 of the 872
validation examples, which moves the measured accuracy by half a point, and for
`profile_decode.py` it drops both the prompt lengths and the number of passes.

Outputs land in `results/` and `docs/img/`. Every script writes only the paths named
in its `--help`, all of which are overridable, and the decoder profiler deliberately
writes no config: the decoder path is not wired into the adaptive serving harness yet.
The scripts are deterministic given a seed except for wall-clock effects, which is why
latency is reported as percentiles over repeated passes rather than as single figures.

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
