# Architecture

One process, two halves with a narrow interface between them: a Python control
plane that decides, and a C++ engine that computes. Stage 1 ran the engine as a
pool of subprocesses speaking JSON; it is now an extension module loaded into the
same interpreter.

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
                                 |  in-process call, borrowed tensors
                                 v
                +-----------------------------------+
                | anytime_runtime (C++ extension)   |
                |   one ONNX Runtime session per    |
                |   variant, 1 intra-op thread,     |
                |   GIL released during inference   |
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

The control plane needs to be cheap and observable; inference needs to be fast.
Keeping them separate, even inside one process, means:

- Each worker holds its own single-threaded ONNX Runtime sessions, so N workers
  behave as N independent servers and the queueing model is a clean M/M/c.
- The control plane has no torch or pandas dependency and runs anywhere.
  `tests/test_import_boundaries.py` enforces this.
- The engine can be replaced or bypassed. `RuntimeClient` also has a backend that
  runs ONNX Runtime through its Python wheel, which is what the extension is
  tested against.

### What moving in-process cost

The subprocess boundary bought fault isolation, and that is now gone: a crash
inside ONNX Runtime takes the whole server down rather than one worker. Stage 1's
worker caught per-request exceptions and kept serving; the engine still reports
recoverable errors as exceptions, but a genuine segfault is no longer contained.
That is a real regression, accepted deliberately.

What it bought is the thing the rest of Stage 2 needs. A process boundary makes
batching across requests and sharing a KV cache between them impossible, because
the tensors live in the wrong address space. Transport also fell from 0.296 ms to
0.018 ms per request, but against 14 ms of inference that was never the argument.

See [`runtime.md`](runtime.md) for the measured comparison.

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
send the union and the runtime keeps the subset its graph declares. The C++
engine and the Python reference backend implement identical filtering, and a
request missing a declared input is an error rather than a silent wrong answer.

## Further reading

- [`planner.md`](planner.md) — admission control and variant selection
- [`runtime.md`](runtime.md) — the C++ engine, version matching, and the measured transport cost
- [`benchmarks.md`](benchmarks.md) — measurement methodology and results
