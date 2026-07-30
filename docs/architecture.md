# Architecture

Two processes with a narrow interface between them: a Python control plane that
decides, and a C++ worker pool that computes.

```text
                +-----------------------------------+
  requests ---> | AdaptiveServer                    |
                |                                   |
                |   LoadMonitor      CPU load EWMA  |
                |   AdaptiveSelector variant choice |
                |   MMcAdmission     sojourn bound  |
                |   in-flight count  backlog        |
                +----------------+------------------+
                                 |  ThreadPoolExecutor
                                 v
                +-----------------------------------+
                | RuntimePool                       |
                |   N RuntimeClients, one per worker|
                +----------------+------------------+
                                 |  line-delimited JSON on stdin/stdout
                                 v
                +-----------------------------------+
                | anytime_runtime (C++)             |
                |   one ONNX Runtime session per    |
                |   variant, 1 intra-op thread      |
                +-----------------------------------+
```

## Request lifecycle

1. `AdaptiveServer.submit` timestamps the arrival and records it in a
   one-second sliding window to estimate the arrival rate.
2. It reads the smoothed CPU load and the current in-flight count.
3. `AdaptiveSelector.select` walks variants from most to least accurate and
   returns the first whose estimated sojourn time fits the deadline. If none
   fits, it returns the fastest variant and the request is rejected.
4. Admitted requests are dispatched to a free `RuntimeClient`. The in-flight
   count is incremented before queueing and released on every exit path.
5. The result is recorded as a `ServedRequest`: chosen variant, measured
   runtime and wall latency, predicted sojourn, load, and compute cost.

## Why the split

The control plane needs to be cheap and observable; inference needs to be fast
and isolated. Keeping them apart means:

- A worker crash costs one worker, not the process.
- Each worker is a single-threaded ONNX Runtime session, so N workers behave as
  N independent servers and the queueing model is a clean M/M/c.
- The control plane has no torch or pandas dependency and runs anywhere.
  `tests/test_import_boundaries.py` enforces this.

The cost is a serialisation hop. With matched ONNX Runtime versions it measures
0.27 ms per request against 14 ms of inference, so it is not currently the
bottleneck. Replacing it with in-process bindings is planned.

## Worker count is a first-class parameter

The number of workers determines queue capacity, so it appears explicitly in the
admission maths rather than being inferred. `AdaptiveSelector` takes `servers`
and constructs its own controller; `AdaptiveServer` raises if that value does not
equal `RuntimePool.size`:

```python
selector = AdaptiveSelector(variants, servers=4)
server = AdaptiveServer(pool, selector, monitor)  # raises if pool.size != 4
```

This is deliberate. An earlier version passed a per-worker service rate into a
single-server queueing formula, which understated capacity by the worker count
and shed traffic the pool could serve comfortably. Making the mismatch
unrepresentable is cheaper than detecting it.

## Multiple inputs per variant

Variants of the same task can declare different graph inputs: DistilBERT takes
`input_ids` and `attention_mask`, BERT-family models additionally take
`token_type_ids`. The variant is chosen after the request is built, so callers
send the union and the runtime keeps the subset its graph declares. Both the C++
worker and the Python fallback implement identical filtering, and a request
missing a declared input is an error rather than a silent wrong answer.

## Further reading

- [`planner.md`](planner.md) — admission control and variant selection
- [`runtime.md`](runtime.md) — the C++ worker and its wire protocol
- [`benchmarks.md`](benchmarks.md) — measurement methodology and results
