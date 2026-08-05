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

![Prefill and decode measured separately, across three precisions](img/decoder_phases.png)

### TPOT grows with the cache, so one figure is not enough

A decode step re-reads the whole cache before it runs, so its cost is a line in the
number of cached tokens rather than a constant:

| Cached tokens | Cache size | `fp32` TPOT | Gather | Scatter | Arena cost |
| --- | --- | --- | --- | --- | --- |
| 128 | 9 MB | 4.83 ms | 0.185 ms | 0.009 ms | 4.0% |
| 512 | 38 MB | 7.22 ms | 0.600 ms | 0.015 ms | 8.5% |
| 960 | 71 MB | 9.54 ms | 1.044 ms | 0.018 ms | 11.1% |

The gather is pure `memcpy`, so it comes out the same at every precision to within a
few percent -- 0.19, 0.59 and 1.05 ms at the three lengths, spreading 5% across
precisions at 128 tokens and 2% at 960 -- because it moves the same bytes whatever the
weights are.

70.8 MB in 1.044 ms is 68 GB/s. That is **not** near this host's peak memory
bandwidth, and the comparison that matters is not the peak but what one thread can do:
a plain 70.8 MB `np.copyto` on the same host measures 1.049 ms as the median of three
passes, against the gather's 1.044. The gather is therefore already running at the copy
rate, and the only ways to make it cheaper are to move fewer bytes or to use more than
one thread.

The scatter stays near zero because it writes only the new token rather than the
whole `present` tensor, which is worth about 1 ms a step at full context.

![What the gather costs, absolutely and as a share of a decode step](img/arena_cost.png)

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

![TTFT by chunk width, indexed to one pass, and the logits each width allocates](img/chunked_prefill.png)

### INT8 is not dominated on the decoder

Stage 1 found INT8 strictly dominated on this host, and
[`quantization.md`](quantization.md) records it. That was an encoder at sequence length
128, and on the decoder the ordering reverses:

| Precision | TPOT at 128 cached | at 512 | at 960 | vs FP32 |
| --- | --- | --- | --- | --- |
| `fp32` | 4.83 ms | 7.22 ms | 9.54 ms | 1.000x |
| `int8` | 3.60 ms | 5.79 ms | 8.45 ms | 0.75x to 0.89x |
| `int4` | 11.56 ms | 13.85 ms | 16.63 ms | 1.74x to 2.39x |

INT8 leads by 25% at 128 cached tokens and 11% at 960. At +0.063 perplexity for 0.61x
the graph size it is the decode variant to serve here — but it does not dominate, and
the difference matters when reading the frontier. INT4 is smaller (367 MB against 399)
and FP32 is more accurate (31.307 against 31.371), so all three sit on the frontier and
what INT8 wins is speed. "Not dominated" is the claim; "wins on every axis" would not
be true of any of them.

### What the reversal does and does not establish

Three things differ between the encoder measurement and this one, not one:

- **Model.** DistilBERT and MiniLM against GPT-2.
- **Shape.** Sequence length 128 in one pass, against a 256-token prefill chunk and
  then single-token decode steps.
- **The quantisation recipe itself.** `export_onnx.py` asks optimum for
  `AutoQuantizationConfig.arm64(is_static=False, per_channel=False)`, which is
  per-tensor across the graph. `export_decoder.py` runs `quantize_dynamic` with
  `per_channel=True` over `MatMul` and `Gemm` only, with the output projection
  excluded, because including it measured 44.4 perplexity against 26.8 for the
  unquantised graph in that same early run — a smaller scoring configuration than the
  32 windows above, so those two numbers compare with each other and not with this
  page's table.

So the honest conclusion is that the encoder result does not generalise to the
decoder. It is not evidence that decode shape alone is responsible, and the usual
explanation — that decode at length 1 is a matrix-vector product bound by the
bandwidth to read weights — does not even cover the whole result, because INT8 also
wins the 256-token chunked prefill, which is not a matrix-vector product.

What the fitted cost model does isolate is the part of the step precision acts on:

| Precision | Cache-independent term | Per cached token | Cache-independent share at 960 |
| --- | --- | --- | --- |
| `fp32` | 4.18 ms | 5.65 µs | 44% |
| `int8` | 2.83 ms | 5.84 µs | 34% |
| `int4` | 10.76 ms | 6.10 µs | 65% |

The per-token coefficients agree within 8% across precisions while the constant terms
span 3.8x. Reading the cache is precision-invariant, as it must be — the arena is
float32 whatever the weights are — and quantisation acts on the constant part. That is
enough to explain the narrowing lead without claiming to have identified the
bottleneck: by the same fit, the term INT8 shrinks is 85% of an FP32 step at 128 cached
tokens and 44% at 960.

INT4 is still dominated on speed, but the penalty shrinks from 4.8x on prefill to 1.7x
on decode, consistent with the same split: its constant term is the one carrying the
per-run cost of unpacking 4-bit weights, and more of the step is cache traffic as
context grows. It buys memory and nothing else, which is what the export measured and
this does not change.

## Batching a decode step: a throughput win that decays with the cache

`scripts/profile_batching.py` measures this through `ContinuousBatchScheduler`, which
is the thing that would do it in service. An earlier round of these numbers was taken
by calling `Engine.run` with hand-assembled batch tensors, and it overstated the result
by about 20% at short context because it left out the gather, the padding, the scatter
and the scheduler. Those numbers are gone; these replace them.

Speedup is against the same sequences stepped **one at a time at the same cached
length**, through the same scheduler — not against nothing, and not against a batch
divided by its width. A batched step's duration is what every sequence in it waited.

| Batch | `fp32` 128 | 512 | 960 | `int8` 128 | 512 | 960 | `int4` 128 | 512 | 960 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 0.92x | 0.93x | 0.91x | 1.20x | 1.10x | 1.05x | 1.15x | 1.14x | 1.12x |
| 4 | 1.52x | 1.27x | 1.11x | 1.75x | 1.35x | 1.16x | 2.08x | 1.65x | 1.56x |
| 8 | 2.32x | 1.52x | 1.25x | 2.25x | 1.46x | 1.22x | 2.89x | 2.16x | 1.78x |
| 16 | 2.87x | 1.68x | 1.33x | **2.50x** | **1.51x** | **1.24x** | 3.43x | 2.39x | **1.93x** |
| 32 | **3.09x** | **1.76x** | **1.37x** | 2.43x | 1.49x | 1.21x | **3.75x** | **2.49x** | 1.92x |

In tokens per second, `fp32` runs 202 at batch 1 and 624 at batch 32 with 128 cached
tokens; at 960 cached it runs 100 and 138.

**The gain decays with cache occupancy, which is what the fitted split predicts.** Only
the cache-independent term of a decode step can be shared across a batch; the
per-cached-token term is per sequence, because each sequence reads its own cache. So the
more cache there is, the less of a step is amortisable. INT4 gains most at full context
— 1.9x against FP32's 1.4x — because its cache-independent term is the largest share of
its step, which is the same fitted split that explains its prefill penalty.

**Four things the curve says that the prediction did not.**

- **A batch of two is a loss at FP32**, at every occupancy: 0.92x, 0.93x, 0.91x. Two
  sequences in one step cost more than two separate steps. INT8 and INT4 gain slightly
  at two, so this is specific to the FP32 path.
- **Returns stop by batch 16 for FP32 and INT8.** INT8 is *slower* at 32 than at 16 at
  all three occupancies. Only INT4, whose step is dominated by the shared term, is still
  gaining at 32.
- **Measured always falls short of predicted**, by 40-45% at short context and 15-20%
  at long. The shortfall is inside `Session::Run` rather than in the gather, the padding
  or the scheduler, all three of which are timed separately and add up to a small
  fraction of it.
- **A per-row constant does not explain the shortfall either.** Fitting
  `shared + per_row × B + per_token × B × L` over every point leaves worst-case errors
  of 21-29%, and fitting it on batches of 4 and above and extrapolating down puts batch
  1 off the line in *opposite directions* for FP32 and INT8. So the honest statement is
  that the split predicts the shape — the decay with occupancy, and INT4 gaining most —
  and does not predict the level. A likely mechanism is that a batch-1 decode step takes
  a different kernel from a batched one, which would make the denominator of every
  speedup here unusually fast; that is consistent with the batch-2 loss and is **not
  measured**, so it is offered as a hypothesis and not as a finding.

![Speedup against batch width per cache occupancy, and measured against predicted](img/batch_scaling.png)

### Right-padding costs a third of a step, and it is not the zeroing

A batch runs at its longest row. Three regimes at batch 8 and 960 cached tokens, `fp32`,
where the spread and the uniform-mean regime hold the same mean so the difference
between them is variance alone:

| Rows | Step | Graph run | Gather | Zeroing |
| --- | --- | --- | --- | --- |
| all 960 | 64.31 ms | 54.36 ms | 10.02 ms | 0.12 ms |
| spread 240-960, mean 600 | 59.80 ms | 52.60 ms | 6.37 ms | 1.02 ms |
| all 600 | 43.12 ms | 36.07 ms | 6.37 ms | 0.16 ms |

Length variance costs **16.7 ms on a 43.1 ms step, 39%**, and of that 16.5 ms is the
graph running every row at 960 positions instead of 600 while 0.9 ms is clearing the
padding. The gather is identical between those two regimes, as it must be: they hold
the same number of real tokens.

So `pad_ms` is the cheap part and bucketing by length would be worth roughly the whole
16.7 ms, up to a ceiling of 21.2 ms — the difference between a batch of the mean length
and a batch of the longest. INT8 and INT4 measure the same effect at 45% and 29% of
their steps. The same cost appears again under load, where it takes about a quarter of
capacity; see below.

### Reproducibility, and what moved between two runs

The whole measurement was run twice, about 30 minutes apart, with a 28-minute load
sweep in between so the second run started from a warmer machine. Absolute step times
moved by at most 8.5% and speedups by at most 9.2%, the largest on INT4 at 960 cached
tokens where a batch of 32 read 2.12x and then 1.92x. Every qualitative statement above
held in both: the FP32 batch-2 loss, the decay with occupancy, INT8 peaking at 16, and
INT4 gaining most at full context. **Prefer the ratios to the milliseconds**, and treat
a difference under 10% between any two runs on this host as noise.

Two figures worth quoting for what they cross-check rather than for themselves. Refitting
the cost split from this script's own batch-1 points gives FP32 4.176 ms + 6.010 µs per
cached token, against the 4.181 ms + 5.654 µs `profile_decode.py` fitted independently
in a different session — the constant term agrees to 0.1%. And the scheduler's own
overhead, the wall time around `step()` beyond what the runtime reports, is 0.05 to
0.79 ms per step across all 54 points, so the Python control loop is not what any of
this measures.

### What alternating costs a sequence that is already decoding

A prefill chunk and a decode step cannot share a run over this graph, for the reason in
[`architecture.md`](architecture.md), so the scheduler alternates and a resident
sequence waits while somebody else's prompt is being read. Six generations over one
arena at batch width 4, `fp32`:

| Chunks per decode step | Gap between a sequence's tokens, median | worst | Prefill chunk | Decode step |
| --- | --- | --- | --- | --- |
| 1 | 27.4 ms | 192.0 ms | 76.2 ms | 22.8 ms |
| 4 | 39.9 ms | 244.4 ms | 75.3 ms | 26.6 ms |

At `int4` a chunk is 428 ms and the worst gap is 790 ms at one chunk per decode step,
1145 ms at four.

The worst gap is larger than one chunk plus one step, and the reason is not only
prefill: this trace has six sequences resident against a batch width of four, so a
sequence can also miss a turn to round-robin. The figure records what was measured —
the longest a decoding sequence went without a token — rather than attributing it.

![One schedule over time, at two settings of the alternation knob](img/alternation.png)

## Batching under load: fairness is the larger half of the result

The mechanism above is one measurement; what a scheduler does to traffic is another.
`scripts/run_decode_sweep.py` drives `ContinuousBatchScheduler` with an **open-loop
Poisson arrival stream** at a swept rate, the same shape as the encoder load sweep and
for the same reason: a closed-loop burst measures a makespan and cannot show a
saturation knee or a queueing tail, and the tail is what a scheduler is judged on.

Load is a fraction of *measured* capacity — the 3.59 completions/s the batched policy
sustains with a full backlog, measured before the sweep rather than derived from a
service time. GPT-2 FP32, 256-token prompts, 64 generated, 150 requests per point, and
all three policies see the same arrivals and the same prompts.

| Policy | Arena holds | Batch width |
| --- | --- | --- |
| `serial` | 1 sequence | 1 |
| `batched-8` | 8 sequences | 8 |
| `batched-8-preempting` | 4 sequences | 8, so `BlockAdmission` must evict |

Attainment counts a request only if it finished **and** met both targets: 500 ms to
first token and 50 ms a token. Those are stated absolutely rather than derived from the
unloaded measurement (86 ms and 6.1 ms here), because a target set at a multiple of what
one sequence achieves alone is a target defined by the absence of batching, which
batching would then fail by construction.

| ρ | TTFT p95, serial | batched | preempting | Attainment, serial | batched | preempting |
| --- | --- | --- | --- | --- | --- | --- |
| 0.40 | 1034 ms | **168 ms** | 179 ms | 76.0% | **100.0%** | 98.7% |
| 0.60 | 2356 ms | **207 ms** | 649 ms | 32.7% | **98.7%** | 94.0% |
| 0.80 | 8972 ms | **792 ms** | 10126 ms | 4.7% | **93.3%** | 86.0% |
| 0.95 | 18712 ms | **1182 ms** | 16435 ms | 3.3% | 66.7% | **84.0%** |
| 1.10 | 24223 ms | **2214 ms** | 19979 ms | 2.0% | 34.0% | **79.3%** |
| 1.30 | 29688 ms | **6655 ms** | 24296 ms | 1.3% | 14.0% | **73.3%** |

![Latency, attainment and goodput against offered load, three scheduling policies](img/decode_sweep.png)

Three things worth separating out of that table.

**Serial decoding collapses immediately, and the throughput result understates why.**
Its time per output token is the best of the three at every load — 5.6 to 6.0 ms,
flat, because it never shares a step with anybody. It fails anyway: by ρ = 0.8 its p95
time to first token is 9.0 seconds against a 500 ms target. Its own capacity is about
1.3 requests/s, so it is already past saturation at the lowest point of a sweep scaled
to the batched policy. Goodput at ρ = 0.8 is 2.38 requests/s batched against 0.10
serial, a factor of 24, and essentially all of that is queueing rather than speed.

**Batching trades time per output token for time to first token, and the trade is
worth it until saturation.** As load rises the batch fills — mean decode batch goes
from 1.5 at ρ = 0.4 to 6.9 at ρ = 1.3 — and each sequence's tokens arrive further
apart, 12.3 ms to 25.4 ms. That is the mechanism from the section above running in
reverse: a decode step's cost grows with the number of sequences in it, so a fuller
batch is slower per step for every member of it.

**Past saturation, limiting concurrency beats sharing.** This is the result that was
not predicted. The preempting policy is slightly *worse* below ρ = 0.8 — eviction is
not free, and it paid 9 to 132 preemptions to hold a resident set of four — and then
it wins by increasingly large margins: 79% attainment against 34% at ρ = 1.1, and 73%
against 14% at ρ = 1.3. Its time per output token stays near 17 ms while the
unconstrained batch degrades to 25 ms, because it never lets more than four sequences
into a step.

The cost is a bimodal distribution rather than a uniformly better one. Its *median*
time to first token stays at 126 ms at every load while its p95 runs to 24 seconds:
most requests are served promptly and a tail waits a very long time. That is what
admission control does, and whether it is the right shape depends on whether the tail
is a queue or a dropped request — here it is a queue, because nothing is shed.

### Prompt and generation length move capacity more than any of this

Measured at ρ = 0.95 under `batched-8`, each shape against its own measured capacity:

| Prompt | Generated | Capacity | TTFT p95 | TPOT p95 | Attainment | Output |
| --- | --- | --- | --- | --- | --- | --- |
| 128 | 64 | 5.17 rps | 948 ms | 19.8 ms | 78.7% | 278 tok/s |
| 256 | 64 | 3.53 rps | 1220 ms | 26.1 ms | 76.0% | 192 tok/s |
| 512 | 64 | 2.02 rps | 1907 ms | 40.4 ms | 68.0% | 112 tok/s |
| 896 | 64 | 1.21 rps | 3889 ms | 64.1 ms | 10.7% | 66 tok/s |
| 256 | 192 | 1.30 rps | 5238 ms | 31.3 ms | 56.0% | 204 tok/s |

Capacity falls 4.3x between a 128-token prompt and an 896-token one, and at 896 the
50 ms per-token target is missed outright: a batch of eight at that occupancy cannot
step faster than 64 ms. The last row is the other axis — three times the generated
tokens is the highest output rate in the table and the worst but one attainment,
because every request holds its blocks three times as long.

**Length variance costs about a quarter of capacity.** Prompts drawn from
`{64, 192, 320, 448}` have the same 253-token mean as the fixed 256-token workload and
sustain 2.72 requests/s against 3.53, with p95 time per output token of 66.0 ms against
26.1. That is the right-padding cost from the section above, under load: the batch runs
at its longest row, so one long prompt slows every sequence sharing the step with it.
It is the strongest argument here for length-bucketed batching, and it is measured
rather than assumed.

### What is weak about these numbers

- **Attainment near the knee is the noisiest thing on this page.** The sweep measures
  ρ = 0.95 under `batched-8` twice — once in the main sweep and once as the first row
  of the shape table — and the two read 66.7% and 76.0% about ten minutes apart. Treat
  differences of that size as noise, and the ordering between policies, which is much
  larger, as the result.
- **150 requests a point** supports p50 and p95. The p99 in
  `results/decode_sweep.json` is six samples from the tail and is recorded rather than
  reported.
- **One precision.** The sweep is FP32 only. It answers a scheduling question, and the
  precision comparison is the section above.
- Every per-request row is in `results/decode_sweep_requests.csv`, so attainment at a
  different pair of targets, or any other percentile, can be recomputed without
  measuring again.

## Reproducing

```bash
python scripts/export_onnx.py --task text     # export FP32 and INT8 variants
python scripts/profile_variants.py            # measure frontier, write serving.yaml
python scripts/run_load_sweep.py              # sweep load, write CSV and figure
python scripts/export_decoder.py              # export GPT-2, measure perplexity
python scripts/profile_decode.py              # measure TTFT and TPOT
python scripts/plot_decode_profiles.py        # draw the three decoder figures
python scripts/profile_batching.py            # measure batch scaling and padding
python scripts/run_decode_sweep.py            # sweep load through the scheduler
python scripts/plot_batching.py               # draw the three batching figures
```

`profile_batching.py` takes about 25 minutes for three precisions and
`run_decode_sweep.py` about 28 for one, most of it spent deliberately idle while the
arrival process waits. `run_decode_sweep.py` reads the fitted cost model out of
`results/decode_profiles.json`, so `profile_decode.py` has to have run first; it says
so rather than substituting coefficients.

The last step draws from `results/decode_profiles.json` and measures nothing, so a
figure can be redrawn without re-measuring. That separation is the point: runs drift by
up to 10% back to back, so a figure regenerated by re-running the profiler would
disagree with the tables beside it for reasons that have nothing to do with the code.

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
- **The encoder path still has no batching**: one request in flight per worker, so
  every number in the variant frontier and load sweep sections describes batch size 1.
  Batching exists on the decoder path only.
- The decoder lane is not served by `AdaptiveServer`, so the two lanes' load sweeps
  are measured through different harnesses and their capacities are not comparable.
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
