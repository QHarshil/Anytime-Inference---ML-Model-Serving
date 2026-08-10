"""Measure what batching a decode step buys, and what right-padding costs.

Everything here runs through `ContinuousBatchScheduler` over `DecoderClient`, which
is the path that would serve batched decoding traffic. An earlier round of these
numbers was taken by calling `Engine.run` directly with hand-assembled batch
tensors, and those are not reportable: they leave out the gather, the padding, the
scatter and the scheduler, which is most of what a batched step costs beyond the
graph. Stage 1 profiled beside the serving path once and it hid a 7.6x version
mismatch for a whole stage.

What is measured here, and what is not
--------------------------------------

This script measures the *mechanism*: how a batched decode step scales with the
batch, and what a batch of unequal lengths pays for being right-padded. It does not
measure serving behaviour under load -- there are no arrivals, no deadlines and no
queueing here. `run_decode_sweep.py` does that, with an open-loop Poisson arrival
process, and the two are deliberately separate: a mechanism is best measured in
isolation and a policy is only meaningful under load.

Assembling the batch
--------------------

A steady-state batched step needs every sequence resident before the first
measurement, and the scheduler already has the knob for that:
`prefill_chunks_per_decode`. Set high, prefill runs ahead of decode, so the batch
fills and then every subsequent iteration is a decode step over all of it. That is
the scheduler's own configuration rather than a bypass of it, and it is stated in
the results as `assembled_prefill_first`.

It is not free of consequences and they are recorded rather than hidden. Each
sequence's prefill completing costs one decode step over whoever is resident at the
time, so by the time a batch of B is assembled the first sequence has emitted B
tokens and the last has emitted one. The batch therefore carries a spread of about
B cached tokens, `cached_min` and `cached_max` report it, and the padding that
spread implies is real: `pad_ms` is non-zero at every point here without anyone
asking for it.

What the split predicts
-----------------------

A decode step at batch 1 fits `base + per_token * cached` (measured in P3, refitted
here from this script's own batch-1 points so the prediction and the measurement come
from one run). Only `base` can be amortised across a batch; the per-cached-token term
is per sequence, because each sequence reads its own cache. So

    step(B, L) ~ base + B * per_token * L

and the speedup over B separate steps is `B * step(1, L) / step(B, L)`, which decays
as L grows. `predicted_speedup` carries that, beside the measured one. The prediction
is here to be wrong if it is wrong: a batched step that beats it is doing something
the split does not describe, and one that falls short of it is paying an overhead this
script should be able to name.

Writes:

  results/batch_profiles.json   scaling, padding, the refitted split, two traces

Refuses to write if the same request emits different tokens at different batch sizes.
Greedy decoding depends only on a sequence's own history, so batching must not change
it; a padding or masking fault that corrupted a neighbour's cache would otherwise
still produce plausible timings, which is how a wrong number gets written down.

Usage:
    python scripts/profile_batching.py
    python scripts/profile_batching.py --precisions fp32 int8
    python scripts/profile_batching.py --quick --output /tmp/batch.json
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

# One definition of what a reportable latency is. `Spread` refuses to let a single
# pass look like a result, and duplicating it here would let the two drift.
from profile_decode import Spread  # noqa: E402

from anytime_serving.serving.batch_scheduler import ContinuousBatchScheduler  # noqa: E402
from anytime_serving.serving.decoder import (  # noqa: E402
    DEFAULT_COPY_THREADS,
    DEFAULT_INTRA_OP_THREADS,
    DecoderClient,
    GenerationRequest,
)
from anytime_serving.serving.onnx_runtime import extension_available, load_extension  # noqa: E402
from anytime_serving.utils.logger import get_logger  # noqa: E402

LOGGER = get_logger("scripts.profile_batching")

PRECISIONS = ("fp32", "int8", "int4")
DEFAULT_BATCH_SIZES = (1, 2, 4, 8, 16, 32)
QUICK_BATCH_SIZES = (1, 2, 4)
# Cached lengths to measure against, matching profile_decode.py so the batch-1 points
# here can be compared with the single-sequence measurement there.
DEFAULT_CACHED = (128, 512, 960)
QUICK_CACHED = (128, 512)
DEFAULT_REPEATS = 3
# Decode steps measured per pass, after the batch is fully resident. Small enough that
# the cache barely grows during one and large enough for a median.
DEFAULT_STEPS = 16
DEFAULT_BLOCK_TOKENS = 64
DEFAULT_MAX_CONTEXT = 1024
# Batch width for the padding regimes. 8 rather than the largest measured width: the
# question is what length variance costs, and a wider batch would confound it with the
# scaling being measured separately above.
PADDING_BATCH = 8
# Shortest row in the spread regime, as a fraction of the longest. 0.25 gives a mean of
# 0.625 and a 4x ratio between the shortest and longest row, which is a harsher spread
# than a real mixed workload and therefore an upper bound on what bucketing could win.
PADDING_SPREAD_FLOOR = 0.25
# Prefill-first assembly. Any value past the number of chunks a prompt needs has the
# same effect; this one is past every prompt this script can construct.
PREFILL_FIRST = 1 << 30
# Prompt lengths for the alternation traces. Deliberately unequal, deliberately not
# multiples of the chunk width so the last chunk of each prompt is short, and
# deliberately more requests than the batch width so the schedule has something
# queued. The arena is sized to hold all of them: if it could not, the trace would be
# a picture of admission rather than of alternation.
TRACE_PROMPTS = (700, 300, 520, 180, 640, 420)
TRACE_NEW_TOKENS = 24
TRACE_BATCH = 4
# Iterations allowed for a batch to become resident before the run is failed. Ten
# times the chunks a full-context batch of 32 needs, so hitting it means something is
# wrong rather than slow.
ASSEMBLY_STEP_BUDGET = 4096


@dataclass
class ScalingPoint:
    """One batch width at one cached length.

    `step` is what every sequence in the batch waited: a batched step's duration is
    not divided by the batch, because each sequence genuinely waited all of it.
    Throughput is the other direction and `tokens_per_s` carries it.

    `scheduler_overhead_p50_ms` is the wall time around `step()` minus the duration
    the runtime reported for the invocation inside it -- the Python scheduler's own
    cost, measured rather than assumed to be negligible. Differenced per step and then
    taken as a median, not the difference of two medians: the second cannot be
    negative and the first can, which is how the mistake announced itself.
    """

    batch_size: int
    cached_tokens: int
    cached_min: int
    cached_max: int
    step: Spread
    tokens_per_s: float
    pad_p50_ms: float
    gather_p50_ms: float
    run_p50_ms: float
    scatter_p50_ms: float
    scheduler_overhead_p50_ms: float
    steps_per_pass: int
    speedup_vs_serial: float = 1.0
    predicted_speedup: float = 1.0
    # Measured over predicted. A number near 1.0 means the split explains the scaling.
    prediction_ratio: float = 1.0


@dataclass
class PaddingPoint:
    """One length regime at a fixed batch width.

    Three regimes make the cost of variance separable from the cost of length. Rows
    all at the longest length pay no padding and the most cache traffic; rows all at
    the mean pay no padding and the least; rows spread between them pay padding and
    sit somewhere in between. The difference between the spread regime and the
    uniform-at-mean regime is what variance costs, and the difference between
    uniform-at-max and uniform-at-mean bounds what any bucketing scheme could recover.
    """

    regime: str
    batch_size: int
    longest_tokens: int
    shortest_tokens: int
    mean_tokens: float
    step: Spread
    pad_p50_ms: float
    run_p50_ms: float
    gather_p50_ms: float
    tokens_per_s: float


@dataclass
class StepSplit:
    """A decode step's cost, split into what a batch can amortise and what it cannot.

    Refitted here from this script's own batch-1 points rather than read from
    `results/decode_profiles.json`, so the prediction and the measurement it is
    compared against come from one run at one thermal state. That it also reproduces
    the P3 fit is a cross-check worth having; that it is fitted from the same run is
    what makes `prediction_ratio` mean something.
    """

    base_ms: float
    per_token_ms: float
    max_residual_ms: float
    fitted_from_points: int

    def step_ms(self, batch_size: int, cached_tokens: int) -> float:
        return self.base_ms + batch_size * self.per_token_ms * cached_tokens

    def predicted_speedup(self, batch_size: int, cached_tokens: int) -> float:
        serial = batch_size * self.step_ms(1, cached_tokens)
        batched = self.step_ms(batch_size, cached_tokens)
        return serial / batched if batched > 0 else 0.0


@dataclass
class TraceStep:
    """One scheduler iteration, as a caller driving it would see it."""

    index: int
    kind: str
    batch_size: int
    start_ms: float
    end_ms: float
    total_ms: float
    cached_max: int
    completed: int
    preempted: int


@dataclass
class AlternationTrace:
    """A whole schedule, and what alternation cost the sequences in it.

    Recorded at two settings of `prefill_chunks_per_decode` because that knob is the
    trade the scheduler's docstring describes: more chunks per decode step is faster
    to first token and stalls resident sequences for longer. `max_decode_gap_ms` is
    the measured version of that stall -- the longest a sequence that was already
    decoding went without a token.
    """

    prefill_chunks_per_decode: int
    max_batch_size: int
    chunk_tokens: int
    requests: int
    prompt_tokens: list[int]
    max_new_tokens: int
    steps: list[TraceStep]
    wall_ms: float
    prefill_steps: int
    decode_steps: int
    mean_decode_batch: float
    prefill_step_p50_ms: float
    decode_step_p50_ms: float
    # Longest interval between two consecutive tokens of one sequence, over sequences
    # that had already emitted at least one. A sequence that has not started decoding
    # is not being stalled, it is waiting to be admitted, which is a different thing.
    max_decode_gap_ms: float
    decode_gap_p50_ms: float
    stalled_sequences: int


@dataclass
class PrecisionProfile:
    precision: str
    graph: str
    size_mb: float
    scaling: list[ScalingPoint] = field(default_factory=list)
    padding: list[PaddingPoint] = field(default_factory=list)
    step_split: StepSplit | None = None
    alternation: list[AlternationTrace] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)


def _graph_path(directory: Path) -> Path:
    graphs = sorted(directory.glob("*.onnx"))
    if not graphs:
        raise SystemExit(f"no .onnx graph in {directory}")
    return graphs[0]


def _graph_bytes(graph: Path) -> int:
    total = graph.stat().st_size
    for sidecar in graph.parent.glob(f"{graph.name}*data*"):
        if sidecar != graph:
            total += sidecar.stat().st_size
    return total


def _prompt(length: int, *, vocab: int, seed: int) -> list[int]:
    """A fixed pseudo-random prompt, distinct per seed.

    Distinct rather than one prompt repeated across the batch: identical rows would
    time the same and hide a gather or mask fault that mixed one row's cache into
    another's. Random rather than real text because this measures cost, and cost
    depends on how many tokens there are rather than on which ones.
    """
    rng = np.random.default_rng(seed)
    return rng.integers(0, vocab, size=length).astype(np.int64).tolist()


def _request(prompt: list[int], max_new_tokens: int, *, request_id: str) -> GenerationRequest:
    # A deadline far enough out not to interfere. Admission against a real deadline is
    # what run_decode_sweep.py measures; here it would only add a variable.
    return GenerationRequest(
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        deadline_ms=1e9,
        request_id=request_id,
    )


def _release_all(client: DecoderClient, request_ids: list[str]) -> None:
    """Free every sequence, whether it finished or was measured part way through.

    A point measures a fixed number of steps out of a longer generation, so most
    sequences are still resident when it ends. Leaving them would shrink the arena
    every later point sees.
    """
    for request_id in request_ids:
        client.release(request_id)


def _assemble(
    client: DecoderClient,
    *,
    prompts: list[list[int]],
    max_new_tokens: int,
    label: str,
    max_batch_size: int,
    chunk_tokens: int | None = None,
) -> tuple[ContinuousBatchScheduler, list[str]]:
    """Submit a batch and step until every sequence of it is decoding.

    Returns the scheduler positioned so that the next iteration is a decode step over
    the whole batch, plus the request ids in submission order.
    """
    scheduler = ContinuousBatchScheduler(
        client,
        chunk_tokens=chunk_tokens,
        max_batch_size=max_batch_size,
        prefill_chunks_per_decode=PREFILL_FIRST,
    )
    request_ids = [f"{label}-{index}" for index in range(len(prompts))]
    for request_id, prompt in zip(request_ids, prompts, strict=True):
        scheduler.submit(_request(prompt, max_new_tokens, request_id=request_id))

    for _ in range(ASSEMBLY_STEP_BUDGET):
        if len(scheduler.decoding) == len(prompts):
            return scheduler, request_ids
        if scheduler.step() is None:
            break
    raise SystemExit(
        f"{label}: only {len(scheduler.decoding)} of {len(prompts)} sequence(s) reached "
        f"the decode phase ({scheduler.waiting} still waiting, {client.free_blocks} of "
        f"{client.capacity_blocks} blocks free). The arena has to hold the whole batch "
        f"at once for a batched step to be measurable; size it from the widest batch "
        f"and longest prompt in the sweep rather than from one point."
    )


def _measure_steps(
    scheduler: ContinuousBatchScheduler, *, expect_batch: int, steps: int
) -> dict[str, list[float]]:
    """Time `steps` decode steps over an assembled batch.

    Wall time is taken around `step()` as well as read out of the runtime, because
    the difference is the scheduler's own cost and the point of measuring through the
    scheduler is not to assume it away.
    """
    samples: dict[str, list[float]] = {
        "total": [],
        "pad": [],
        "gather": [],
        "run": [],
        "scatter": [],
        "wall": [],
        "overhead": [],
        "cached_min": [],
        "cached_max": [],
    }
    for _ in range(steps):
        started = time.perf_counter()
        step = scheduler.step()
        wall_ms = (time.perf_counter() - started) * 1000.0
        if step is None:
            raise SystemExit(
                "the schedule ran out of work while a batched step was being measured; "
                "max_new_tokens has to cover the assembly steps as well as the "
                "measured ones"
            )
        if step.kind != "decode" or step.batch_size != expect_batch:
            raise SystemExit(
                f"expected a decode step over {expect_batch} sequence(s) and got a "
                f"{step.kind} step over {step.batch_size}. With prefill running ahead "
                f"of decode there should be nothing left to prefill by now, so this is "
                f"a fault in the measurement rather than a slow result."
            )
        record = step.records[0]
        samples["total"].append(record.total_ms)
        samples["pad"].append(record.pad_ms)
        samples["gather"].append(record.gather_ms)
        samples["run"].append(record.run_ms)
        samples["scatter"].append(record.scatter_ms)
        samples["wall"].append(wall_ms)
        # Paired, per step, rather than one aggregate minus another. Taking the median
        # of the wall times and subtracting the median of the durations mixes two
        # aggregations -- a pooled median against a median of per-pass medians -- and
        # they disagree by more than the quantity being measured when the passes drift.
        # It produced a negative overhead at the widest batch, which is not a thing that
        # can happen: the wall clock around a call cannot be shorter than what the call
        # reports taking.
        samples["overhead"].append(wall_ms - record.total_ms)
        cached = [row.cached_tokens for row in step.records]
        samples["cached_min"].append(float(min(cached)))
        samples["cached_max"].append(float(max(cached)))
    return samples


def _pooled(passes: list[dict[str, list[float]]], key: str) -> list[float]:
    return [value for sample in passes for value in sample[key]]


def measure_scaling_point(
    client: DecoderClient,
    precision: str,
    *,
    batch_size: int,
    cached_tokens: int,
    vocab: int,
    repeats: int,
    steps: int,
) -> tuple[ScalingPoint, list[int]]:
    """One batch width at one cached length, as a median over independent passes.

    Also returns what the first sequence of the batch emitted, which is compared
    across batch widths: greedy decoding depends only on a sequence's own history, so
    the same prompt has to produce the same tokens whether it ran alone or in a batch
    of 32.
    """
    prompts = [_prompt(cached_tokens, vocab=vocab, seed=index + 1) for index in range(batch_size)]
    # One decode step happens for each sequence that finishes prefilling, so the batch
    # is `batch_size` tokens into its generation before a measurement starts.
    max_new_tokens = batch_size + steps + 4
    per_pass: list[dict[str, list[float]]] = []
    emitted: list[int] = []

    # One warm-up pass, discarded. The first batched step of a session sizes the
    # padded staging buffers, which is real and is not a steady-state step.
    for index in range(repeats + 1):
        label = f"{precision}-scale-{cached_tokens}-{batch_size}-{index}"
        scheduler, request_ids = _assemble(
            client,
            prompts=prompts,
            max_new_tokens=max_new_tokens,
            label=label,
            max_batch_size=batch_size,
        )
        samples = _measure_steps(scheduler, expect_batch=batch_size, steps=steps)
        first = client.emitted(request_ids[0])
        _release_all(client, request_ids)
        if client.free_blocks != client.capacity_blocks:
            raise SystemExit(
                f"{label}: {client.capacity_blocks - client.free_blocks} block(s) were "
                f"still held after releasing every sequence, so a later point would be "
                f"measured against a smaller arena than this one"
            )
        if index == 0:
            continue
        per_pass.append(samples)
        if not emitted:
            emitted = first
        elif first != emitted:
            raise SystemExit(
                f"{label}: the same prompt emitted different tokens on two passes of "
                f"one configuration, which is not a batching question. Greedy decoding "
                f"over the same history is deterministic, so this is a fault."
            )

    step = Spread.of([statistics.median(sample["total"]) for sample in per_pass])
    return (
        ScalingPoint(
            batch_size=batch_size,
            cached_tokens=cached_tokens,
            cached_min=int(min(_pooled(per_pass, "cached_min"))),
            cached_max=int(max(_pooled(per_pass, "cached_max"))),
            step=step,
            tokens_per_s=round(batch_size / step.p50_ms * 1000.0 if step.p50_ms else 0.0, 1),
            pad_p50_ms=round(statistics.median(_pooled(per_pass, "pad")), 4),
            gather_p50_ms=round(statistics.median(_pooled(per_pass, "gather")), 3),
            run_p50_ms=round(statistics.median(_pooled(per_pass, "run")), 3),
            scatter_p50_ms=round(statistics.median(_pooled(per_pass, "scatter")), 4),
            scheduler_overhead_p50_ms=round(statistics.median(_pooled(per_pass, "overhead")), 4),
            steps_per_pass=steps,
        ),
        emitted,
    )


def measure_padding_point(
    client: DecoderClient,
    precision: str,
    *,
    regime: str,
    lengths: list[int],
    vocab: int,
    repeats: int,
    steps: int,
) -> PaddingPoint:
    """One length regime at a fixed batch width."""
    prompts = [
        _prompt(length, vocab=vocab, seed=index + 101) for index, length in enumerate(lengths)
    ]
    max_new_tokens = len(lengths) + steps + 4
    per_pass: list[dict[str, list[float]]] = []

    for index in range(repeats + 1):
        label = f"{precision}-pad-{regime}-{index}"
        scheduler, request_ids = _assemble(
            client,
            prompts=prompts,
            max_new_tokens=max_new_tokens,
            label=label,
            max_batch_size=len(lengths),
        )
        samples = _measure_steps(scheduler, expect_batch=len(lengths), steps=steps)
        _release_all(client, request_ids)
        if index == 0:
            continue
        per_pass.append(samples)

    step = Spread.of([statistics.median(sample["total"]) for sample in per_pass])
    return PaddingPoint(
        regime=regime,
        batch_size=len(lengths),
        longest_tokens=max(lengths),
        shortest_tokens=min(lengths),
        mean_tokens=round(statistics.fmean(lengths), 1),
        step=step,
        pad_p50_ms=round(statistics.median(_pooled(per_pass, "pad")), 4),
        run_p50_ms=round(statistics.median(_pooled(per_pass, "run")), 3),
        gather_p50_ms=round(statistics.median(_pooled(per_pass, "gather")), 3),
        tokens_per_s=round(len(lengths) / step.p50_ms * 1000.0 if step.p50_ms else 0.0, 1),
    )


def padding_regimes(*, longest: int, batch_size: int, floor: float) -> dict[str, list[int]]:
    """Three batches: all longest, spread up to longest, all at the spread's mean.

    The spread is linear from `floor * longest` to `longest` so the mean is stated
    rather than sampled, which keeps the uniform-at-mean regime exactly comparable to
    it. A sampled spread would differ in mean from run to run and the subtraction
    would carry that difference.
    """
    if batch_size < 2:
        raise ValueError("a padding regime needs at least two rows to have a spread")
    if not 0.0 < floor <= 1.0:
        raise ValueError("floor must be a fraction of the longest row, in (0, 1]")
    shortest = max(1, int(longest * floor))
    stride = (longest - shortest) / (batch_size - 1)
    spread = [int(round(shortest + stride * index)) for index in range(batch_size)]
    mean = int(round(statistics.fmean(spread)))
    return {
        "uniform-max": [longest] * batch_size,
        "spread": spread,
        "uniform-mean": [mean] * batch_size,
    }


def tokens_disagree(expected: list[int], actual: list[int]) -> bool:
    """Whether two runs of one prompt emitted different tokens.

    Compared over the prefix both produced. Batch widths are measured with different
    token budgets -- assembling a wider batch spends more of them -- so the shorter run
    bounds the comparison. Trimming to the shorter is not a weakening of the check: a
    divergence at step k shows up in every run that reached step k, and the first
    batched step is step one.
    """
    shared = min(len(expected), len(actual))
    return expected[:shared] != actual[:shared]


def fit_step_split(scaling: list[ScalingPoint]) -> StepSplit:
    """Fit `base + per_token * cached` to the batch-1 points.

    A line, because a decode step re-reads the whole cache. Fitted from batch 1 only:
    the batched points are what the fit is used to predict, so fitting from them would
    make the prediction unfalsifiable.
    """
    serial = sorted(
        [point for point in scaling if point.batch_size == 1],
        key=lambda point: point.cached_tokens,
    )
    if not serial:
        raise ValueError(
            "the split is fitted from batch-1 points and there are none; a sweep that "
            "omits batch 1 also has nothing to compute a speedup against"
        )
    lengths = np.array([point.cached_tokens for point in serial], dtype=np.float64)
    latencies = np.array([point.step.p50_ms for point in serial], dtype=np.float64)
    if lengths.size >= 2:
        slope, intercept = np.polyfit(lengths, latencies, 1)
    else:
        slope, intercept = 0.0, float(latencies[0])
    residual = float(np.max(np.abs(latencies - (intercept + slope * lengths))))
    return StepSplit(
        base_ms=round(float(intercept), 4),
        per_token_ms=round(float(slope), 6),
        max_residual_ms=round(residual, 4),
        fitted_from_points=len(serial),
    )


def apply_scaling_derivations(scaling: list[ScalingPoint], split: StepSplit) -> None:
    """Fill in the speedups, once every point at a cached length has been measured.

    Derived rather than measured separately: the denominator is the batch-1 point at
    the same cached length, so a run that measured them at different times would be
    comparing across thermal states.
    """
    serial = {point.cached_tokens: point.step.p50_ms for point in scaling if point.batch_size == 1}
    # One cached length cannot separate the constant term from the per-token one, so a
    # single-point fit reports a slope of zero and would predict perfect amortisation
    # at every width. That is an artefact of the sweep rather than a prediction, so the
    # comparison is withheld instead of being printed as a failure of the model.
    predictable = split.fitted_from_points >= 2
    for point in scaling:
        reference = serial.get(point.cached_tokens)
        if reference is None or point.step.p50_ms <= 0.0:
            continue
        point.speedup_vs_serial = round(point.batch_size * reference / point.step.p50_ms, 4)
        if not predictable:
            point.predicted_speedup = 0.0
            point.prediction_ratio = 0.0
            continue
        point.predicted_speedup = round(
            split.predicted_speedup(point.batch_size, point.cached_tokens), 4
        )
        if point.predicted_speedup > 0:
            point.prediction_ratio = round(point.speedup_vs_serial / point.predicted_speedup, 4)


def record_alternation(
    client: DecoderClient,
    precision: str,
    *,
    prompt_tokens: list[int],
    max_new_tokens: int,
    max_batch_size: int,
    prefill_chunks_per_decode: int,
    vocab: int,
    chunk_tokens: int | None = None,
) -> AlternationTrace:
    """Drive a whole schedule and record every iteration of it.

    One trace, driven to completion, at the scheduler's own alternation setting rather
    than the prefill-first assembly the scaling points use. This is where a decode
    step stalling behind a prefill chunk is visible, so it is measured here and not
    inferred from the chunk width.
    """
    scheduler = ContinuousBatchScheduler(
        client,
        chunk_tokens=chunk_tokens,
        max_batch_size=max_batch_size,
        prefill_chunks_per_decode=prefill_chunks_per_decode,
    )
    request_ids = []
    for index, length in enumerate(prompt_tokens):
        request_id = f"{precision}-alt-{prefill_chunks_per_decode}-{index}"
        request_ids.append(request_id)
        scheduler.submit(
            _request(
                _prompt(length, vocab=vocab, seed=index + 201),
                max_new_tokens,
                request_id=request_id,
            )
        )

    steps: list[TraceStep] = []
    # Wall clock of each token, per sequence, so the gap a stalled sequence saw is
    # measured rather than derived from the chunk width.
    token_times: dict[str, list[float]] = {request_id: [] for request_id in request_ids}
    origin = time.perf_counter()
    while not scheduler.idle():
        started = time.perf_counter()
        step = scheduler.step()
        finished = time.perf_counter()
        if step is None:
            break
        end_ms = (finished - origin) * 1000.0
        if step.kind == "decode":
            for request_id in step.request_ids:
                token_times[request_id].append(end_ms)
        steps.append(
            TraceStep(
                index=len(steps),
                kind=step.kind,
                batch_size=step.batch_size,
                start_ms=round((started - origin) * 1000.0, 3),
                end_ms=round(end_ms, 3),
                total_ms=round(step.total_ms, 3),
                cached_max=max((row.cached_tokens for row in step.records), default=0),
                completed=len(step.completed),
                preempted=len(step.preempted),
            )
        )
    wall_ms = (time.perf_counter() - origin) * 1000.0
    _release_all(client, request_ids)

    gaps: list[float] = []
    stalled = 0
    for times in token_times.values():
        if len(times) < 2:
            continue
        sequence_gaps = [
            later - earlier for earlier, later in zip(times[:-1], times[1:], strict=True)
        ]
        gaps.extend(sequence_gaps)
        stalled += 1
    prefill_steps = [step for step in steps if step.kind == "prefill"]
    decode_steps = [step for step in steps if step.kind == "decode"]
    return AlternationTrace(
        prefill_chunks_per_decode=prefill_chunks_per_decode,
        max_batch_size=max_batch_size,
        chunk_tokens=scheduler.chunk_tokens,
        requests=len(prompt_tokens),
        prompt_tokens=list(prompt_tokens),
        max_new_tokens=max_new_tokens,
        steps=steps,
        wall_ms=round(wall_ms, 2),
        prefill_steps=len(prefill_steps),
        decode_steps=len(decode_steps),
        mean_decode_batch=round(scheduler.stats.mean_decode_batch, 3),
        prefill_step_p50_ms=round(
            statistics.median([step.total_ms for step in prefill_steps]) if prefill_steps else 0.0,
            3,
        ),
        decode_step_p50_ms=round(
            statistics.median([step.total_ms for step in decode_steps]) if decode_steps else 0.0,
            3,
        ),
        max_decode_gap_ms=round(max(gaps), 3) if gaps else 0.0,
        decode_gap_p50_ms=round(statistics.median(gaps), 3) if gaps else 0.0,
        stalled_sequences=stalled,
    )


def blocks_for_sweep(
    *,
    batch_sizes: tuple[int, ...],
    cached_lengths: tuple[int, ...],
    steps: int,
    block_tokens: int,
    max_context: int,
    padding_batch: int,
) -> int:
    """Arena size the widest configuration in this sweep needs.

    Derived rather than defaulted: a fixed block count silently caps how wide a batch
    can be measured, and the failure mode is a sweep that stops early with a message
    about the arena rather than a wrong number, which is better but still avoidable.
    """
    longest = max(cached_lengths)
    per_sequence = min(max_context, longest + max(batch_sizes) + steps + 4)
    blocks_each = -(-per_sequence // block_tokens)
    widest = max(max(batch_sizes), padding_batch)
    scaling_blocks = blocks_each * widest
    # The traces hold every one of their sequences at once, and their prompts are
    # unrelated to the sweep's cached lengths.
    trace_each = -(-min(max_context, max(TRACE_PROMPTS) + TRACE_NEW_TOKENS) // block_tokens)
    return max(scaling_blocks, trace_each * len(TRACE_PROMPTS))


def feasible(cached_tokens: int, batch_size: int, steps: int, max_context: int) -> bool:
    """Whether a point fits under the model's position limit.

    Assembling a batch of B costs the first sequence B tokens before the measurement
    starts, so a wide batch at a nearly full cache runs past the position table. GPT-2
    stops at 1024 and exceeding it is an out-of-bounds Gather inside ONNX Runtime
    rather than a graceful stop.
    """
    return cached_tokens + batch_size + steps + 4 <= max_context


def host_metadata(
    intra_op_threads: int,
    copy_threads: int = DEFAULT_COPY_THREADS,
    allow_spinning: bool = True,
) -> dict[str, object]:
    """What the run was taken on, including the settings that change the numbers.

    `intra_op_num_threads` is read from the run rather than written as a constant. It
    was a hardcoded 1 while the thread count was not configurable, and leaving it that
    way once it was would have made every recorded artefact describe a configuration
    it had not been measured under. `copy_threads` is here for the same reason and was
    added at the same time as the setting: it moves `gather_p50_ms`, which is a
    reported number.
    """
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "backend": "extension",
        "onnxruntime": load_extension().onnxruntime_version(),
        "intra_op_num_threads": intra_op_threads,
        "copy_threads": copy_threads,
        "allow_spinning": allow_spinning,
    }


def profile_precision(
    precision: str,
    graph: Path,
    *,
    batch_sizes: tuple[int, ...],
    cached_lengths: tuple[int, ...],
    repeats: int,
    steps: int,
    block_tokens: int,
    num_blocks: int,
    max_context: int,
    padding_batch: int,
    trace_settings: tuple[int, ...],
    intra_op_threads: int,
    copy_threads: int = DEFAULT_COPY_THREADS,
    allow_spinning: bool = True,
) -> tuple[PrecisionProfile, list[str]]:
    """Every measurement for one precision. Returns it plus any divergence found."""
    profile = PrecisionProfile(
        precision=precision,
        graph=graph.name,
        size_mb=round(_graph_bytes(graph) / 1e6, 1),
    )
    divergences: list[str] = []

    with DecoderClient(
        graph,
        block_tokens=block_tokens,
        num_blocks=num_blocks,
        max_context_tokens=max_context,
        intra_op_threads=intra_op_threads,
        copy_threads=copy_threads,
        allow_spinning=allow_spinning,
    ) as client:
        geometry = client.geometry
        LOGGER.info(
            "  %s: %d layers, %d kv heads, head dim %d -> %.1f KiB per token; arena "
            "%d blocks of %d tokens (%.0f MB)",
            precision,
            geometry.layers,
            geometry.kv_heads,
            geometry.head_dim,
            geometry.bytes_per_token / 1024,
            client.capacity_blocks,
            geometry.block_tokens,
            client.arena_bytes / 1e6,
        )

        probe_id = f"{precision}-vocab-probe"
        client.admit(_request([0] * 8, 1, request_id=probe_id))
        client.prefill(probe_id)
        vocab = int(client.next_token_logits(probe_id).size)
        client.release(probe_id)

        # Tokens the first sequence emits, per cached length, at whatever batch width
        # measured it first. Every other width has to agree with it.
        reference: dict[int, tuple[int, list[int]]] = {}
        for cached_tokens in cached_lengths:
            for batch_size in batch_sizes:
                if not feasible(cached_tokens, batch_size, steps, max_context):
                    note = (
                        f"{precision}: batch {batch_size} at {cached_tokens} cached "
                        f"needs {cached_tokens + batch_size + steps + 4} positions and "
                        f"the model has {max_context}"
                    )
                    profile.skipped.append(note)
                    LOGGER.info("  skipped: %s", note)
                    continue
                point, emitted = measure_scaling_point(
                    client,
                    precision,
                    batch_size=batch_size,
                    cached_tokens=cached_tokens,
                    vocab=vocab,
                    repeats=repeats,
                    steps=steps,
                )
                profile.scaling.append(point)
                LOGGER.info(
                    "  batch %2d at %4d cached: step %s  %6.1f tok/s  pad %.3f  "
                    "gather %.3f  scheduler %.3f",
                    batch_size,
                    cached_tokens,
                    point.step,
                    point.tokens_per_s,
                    point.pad_p50_ms,
                    point.gather_p50_ms,
                    point.scheduler_overhead_p50_ms,
                )
                if cached_tokens not in reference:
                    reference[cached_tokens] = (batch_size, emitted)
                    continue
                first_batch, expected = reference[cached_tokens]
                if tokens_disagree(expected, emitted):
                    divergences.append(
                        f"  {precision} at {cached_tokens} cached: the same prompt "
                        f"emitted {emitted[:6]} in a batch of {batch_size} and "
                        f"{expected[:6]} in a batch of {first_batch}"
                    )

        profile.step_split = fit_step_split(profile.scaling)
        apply_scaling_derivations(profile.scaling, profile.step_split)
        LOGGER.info(
            "  step splits into %.3f ms amortisable + %.5f ms per cached token per "
            "sequence (max residual %.3f over %d point(s))",
            profile.step_split.base_ms,
            profile.step_split.per_token_ms,
            profile.step_split.max_residual_ms,
            profile.step_split.fitted_from_points,
        )
        for point in profile.scaling:
            if point.batch_size == 1 or point.predicted_speedup <= 0.0:
                continue
            LOGGER.info(
                "  batch %2d at %4d cached: %.2fx measured against %.2fx predicted (%.2fx)",
                point.batch_size,
                point.cached_tokens,
                point.speedup_vs_serial,
                point.predicted_speedup,
                point.prediction_ratio,
            )

        longest = max(cached_lengths)
        if feasible(longest, padding_batch, steps, max_context):
            regimes = padding_regimes(
                longest=longest, batch_size=padding_batch, floor=PADDING_SPREAD_FLOOR
            )
            for regime, lengths in regimes.items():
                point = measure_padding_point(
                    client,
                    precision,
                    regime=regime,
                    lengths=lengths,
                    vocab=vocab,
                    repeats=repeats,
                    steps=steps,
                )
                profile.padding.append(point)
                LOGGER.info(
                    "  padding %-12s rows %4d-%4d (mean %6.1f): step %s  pad %.3f  run %.3f",
                    regime,
                    point.shortest_tokens,
                    point.longest_tokens,
                    point.mean_tokens,
                    point.step,
                    point.pad_p50_ms,
                    point.run_p50_ms,
                )
        else:
            note = (
                f"{precision}: the padding regimes need "
                f"{longest + padding_batch + steps + 4} positions at batch "
                f"{padding_batch} and the model has {max_context}"
            )
            profile.skipped.append(note)
            LOGGER.info("  skipped: %s", note)

        for setting in trace_settings:
            trace = record_alternation(
                client,
                precision,
                prompt_tokens=list(TRACE_PROMPTS),
                max_new_tokens=TRACE_NEW_TOKENS,
                max_batch_size=TRACE_BATCH,
                prefill_chunks_per_decode=setting,
                vocab=vocab,
            )
            profile.alternation.append(trace)
            LOGGER.info(
                "  alternation at %d chunk(s) per decode: %d prefill + %d decode steps "
                "in %.0f ms, mean batch %.2f, decode gap p50 %.1f ms max %.1f ms",
                setting,
                trace.prefill_steps,
                trace.decode_steps,
                trace.wall_ms,
                trace.mean_decode_batch,
                trace.decode_gap_p50_ms,
                trace.max_decode_gap_ms,
            )

    return profile, divergences


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="gpt2", help="Model short name used in the graph paths")
    parser.add_argument("--model-dir", type=Path, default=Path("models/onnx"))
    parser.add_argument("--precisions", nargs="+", default=list(PRECISIONS), choices=PRECISIONS)
    parser.add_argument("--output", type=Path, default=Path("results/batch_profiles.json"))
    parser.add_argument(
        "--batch-sizes",
        nargs="+",
        type=int,
        default=list(DEFAULT_BATCH_SIZES),
        help=(
            "Batch widths to measure. Must include 1: it is both the denominator of "
            "every speedup and what the cost split is fitted from"
        ),
    )
    parser.add_argument(
        "--cached", nargs="+", type=int, default=list(DEFAULT_CACHED), help="Cached lengths"
    )
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--block-tokens", type=int, default=DEFAULT_BLOCK_TOKENS)
    parser.add_argument(
        "--blocks",
        type=int,
        default=0,
        help="Arena blocks; 0 derives it from the widest configuration in the sweep",
    )
    parser.add_argument("--max-context", type=int, default=DEFAULT_MAX_CONTEXT)
    parser.add_argument(
        "--intra-op-threads",
        type=int,
        default=DEFAULT_INTRA_OP_THREADS,
        help=(
            "Threads ONNX Runtime may use inside one operator. Defaults to what the "
            "serving path uses, so a recorded run describes the configuration that "
            "would actually serve. The scatter is this process's own memcpy loop and "
            "stays serial whatever this is set to, so it doubles as a drift control: "
            "it should not move with this. The gather and the padding were that too "
            "until --copy-threads existed, and are only a control while it is 1 "
            f"(default {DEFAULT_INTRA_OP_THREADS})"
        ),
    )
    parser.add_argument(
        "--copy-threads",
        type=int,
        default=DEFAULT_COPY_THREADS,
        help=(
            "Runners the KV gather may split across, the calling thread included. A "
            "separate budget from --intra-op-threads because it divides a different "
            "thing: that one divides the graph, this one divides the memcpy that "
            "stages the batch's past, and the two never run at the same moment. One "
            "is the serial copy every recorded number was taken with "
            f"(default {DEFAULT_COPY_THREADS})"
        ),
    )
    parser.add_argument(
        "--no-spinning",
        action="store_true",
        help=(
            "Stop ONNX Runtime's intra-op workers busy-waiting between parallel "
            "sections. They do by default, which starts the next section sooner and "
            "holds the cores in between -- and what runs between two Runs here is a "
            "bandwidth-bound gather. Measured at --copy-threads 8: it buys the gather "
            "1.9% and costs Run 1.79x, for a 1.65x worse step, so this is the control "
            "that establishes the default rather than a lever. Off by default, "
            "matching ONNX Runtime and every recorded number"
        ),
    )
    parser.add_argument("--padding-batch", type=int, default=PADDING_BATCH)
    parser.add_argument(
        "--quick", action="store_true", help="Fewer batch widths, cached lengths and passes"
    )
    args = parser.parse_args()

    if not extension_available():
        raise SystemExit(
            "anytime_runtime is not available. Batched decoding is the extension, so "
            "there is nothing to measure without it. Build it with:\n"
            "    pip install -e ."
        )
    if args.repeats < 1:
        raise SystemExit("--repeats must be at least 1")
    if args.intra_op_threads < 1:
        raise SystemExit("--intra-op-threads must be at least 1")
    if args.copy_threads < 1:
        raise SystemExit("--copy-threads must be at least 1")

    batch_sizes = tuple(QUICK_BATCH_SIZES if args.quick else args.batch_sizes)
    cached = tuple(QUICK_CACHED if args.quick else args.cached)
    repeats = 2 if args.quick else args.repeats
    if 1 not in batch_sizes:
        raise SystemExit(
            "--batch-sizes must include 1. Every speedup here is against the same "
            "sequences run one at a time, and the cost split is fitted from batch 1, "
            "so a sweep without it can report neither."
        )
    batch_sizes = tuple(sorted(set(batch_sizes)))
    cached = tuple(sorted({length for length in cached if length < args.max_context}))
    if not cached:
        raise SystemExit(f"no cached length below --max-context {args.max_context}")

    num_blocks = args.blocks or blocks_for_sweep(
        batch_sizes=batch_sizes,
        cached_lengths=cached,
        steps=args.steps,
        block_tokens=args.block_tokens,
        max_context=args.max_context,
        padding_batch=args.padding_batch,
    )

    graphs: dict[str, Path] = {}
    for precision in args.precisions:
        directory = args.model_dir / f"decoder_{args.model}_{precision}"
        if not directory.is_dir():
            raise SystemExit(
                f"{directory} is missing. Export it with:\n"
                f"    python scripts/export_decoder.py --precisions {precision}"
            )
        graphs[precision] = _graph_path(directory)

    trace_settings = (1,) if args.quick else (1, 4)
    LOGGER.info(
        "Measuring %s through the batching scheduler: batches %s at %s cached "
        "token(s), %d pass(es) per point, %d block(s) of %d tokens",
        ", ".join(args.precisions),
        list(batch_sizes),
        list(cached),
        repeats,
        num_blocks,
        args.block_tokens,
    )

    profiles: list[PrecisionProfile] = []
    divergences: list[str] = []
    for precision in args.precisions:
        profile, found = profile_precision(
            precision,
            graphs[precision],
            batch_sizes=batch_sizes,
            cached_lengths=cached,
            repeats=repeats,
            steps=args.steps,
            block_tokens=args.block_tokens,
            num_blocks=num_blocks,
            max_context=args.max_context,
            padding_batch=args.padding_batch,
            trace_settings=trace_settings,
            intra_op_threads=args.intra_op_threads,
            copy_threads=args.copy_threads,
            allow_spinning=not args.no_spinning,
        )
        profiles.append(profile)
        divergences.extend(found)

    if divergences:
        raise SystemExit(
            "Batching changed what a sequence emitted:\n"
            + "\n".join(divergences)
            + "\n\nGreedy decoding depends only on a sequence's own history, so the "
            "batch it shared a run with must not change its tokens. Do not report the "
            "latencies above: a padded row that leaked into a neighbour's attention "
            "would still produce plausible timings. Run pytest -q "
            "tests/test_batch_scheduler.py tests/test_decoder_session.py, which compare "
            "batched against unbatched directly."
        )

    payload = {
        "host": host_metadata(args.intra_op_threads, args.copy_threads, not args.no_spinning),
        "model": args.model,
        "measurement_passes": repeats,
        "measured_steps_per_pass": args.steps,
        "batch_sizes": list(batch_sizes),
        "cached_lengths": list(cached),
        "block_tokens": args.block_tokens,
        "arena_blocks": num_blocks,
        "max_context_tokens": args.max_context,
        "assembled_prefill_first": True,
        "tokens_agree_across_batch_sizes": True,
        "precisions": [asdict(profile) for profile in profiles],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    LOGGER.info("Wrote %s", args.output)

    LOGGER.info("")
    for profile in profiles:
        for cached_tokens in cached:
            points = [p for p in profile.scaling if p.cached_tokens == cached_tokens]
            if not points:
                continue
            widest = max(points, key=lambda point: point.batch_size)
            LOGGER.info(
                "%-5s at %4d cached: batch %2d gives %.2fx the tokens per second of "
                "one at a time (%.1f against %.1f tok/s), predicted %.2fx",
                profile.precision,
                cached_tokens,
                widest.batch_size,
                widest.speedup_vs_serial,
                widest.tokens_per_s,
                next(p.tokens_per_s for p in points if p.batch_size == 1),
                widest.predicted_speedup,
            )
    LOGGER.info("")
    LOGGER.info(
        "Speedup is against the same sequences stepped one at a time at the same "
        "cached length, through the same scheduler. It decays as the cache fills "
        "because only the cache-independent part of a step amortises across a batch. "
        "Repeating this run after a pause reproduces each figure within a few percent; "
        "back to back it drifts further, so leave a gap before comparing two runs."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
