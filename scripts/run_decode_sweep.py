"""Sweep offered load through the batching scheduler and compare scheduling policies.

The mechanism -- what a batched decode step costs and what padding adds to it -- is
measured by `profile_batching.py`. This measures the thing a scheduler exists for:
what happens to time to first token and time per output token when requests arrive
faster than one at a time can serve them.

Arrivals are an open-loop Poisson process at a swept rate, independent of completions,
which is the same shape `run_load_sweep.py` uses for the encoder lane and for the same
reason. A closed-loop burst measures a makespan; it cannot show a saturation knee or a
queueing tail, and the tail is what a scheduler is judged on. Load is expressed as a
fraction of *measured* capacity rather than as a bare request rate: capacity here is
the completion rate the batched policy sustains with a full backlog, measured before
the sweep starts rather than derived from a formula.

Three policies, one arrival stream
----------------------------------

  serial              arena for one sequence, batch width 1. What the decoder path
                      did before the scheduler existed: one generation at a time,
                      start to finish, everybody else waiting.
  batched             arena for the whole batch, batch width B. Iteration-level
                      scheduling with no eviction, because nothing has to be evicted.
  batched-preempting  arena for fewer sequences than the batch width, with
                      BlockAdmission deciding who is resident. Past rho = 1 the arena
                      fills, so this is the configuration where preempt-and-recompute
                      is on the critical path rather than a curiosity.

All three see the same arrival times and the same prompts, from the same seed.

Why the arena size is part of a policy
--------------------------------------

`DecoderClient` reserves a sequence's whole projected generation, so arena capacity is
the concurrency limit. That interacts with the batch width and the interaction is not
optional: an arena holding many more sequences than the batch width leaves the
scheduler round-robining over a resident set it cannot serve in one step, and each
sequence's time per output token degrades in proportion. So each policy pairs a width
with a capacity deliberately, and `resident_capacity` records it.

The SLO is stated, not derived
------------------------------

Attainment is measured against an absolute pair -- a time to first token target and a
time per output token target -- and both are arguments. They are deliberately not
derived from the unloaded latency: a target set at a multiple of what one sequence
alone achieves is a target defined by not batching, and batching would fail it by
construction at every rate. The unloaded numbers are still measured and recorded, as
the reference the loaded percentiles are read against.

Every per-request row is written out, so attainment at a different SLO, or any other
percentile, can be recomputed without measuring again. `run_load_sweep.py` cannot
redraw its figure without re-measuring and that is a wart worth not repeating.

Writes:

  results/decode_sweep.json           metadata, capacity, one row per point
  results/decode_sweep_requests.csv   one row per request, every point

Usage:
    python scripts/run_decode_sweep.py
    python scripts/run_decode_sweep.py --precision int8 --utilisations 0.95 1.3
    python scripts/run_decode_sweep.py --quick --output-dir /tmp/sweep
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import statistics
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from anytime_serving.serving.batch_scheduler import ContinuousBatchScheduler  # noqa: E402
from anytime_serving.serving.decoder import (  # noqa: E402
    DEFAULT_INTRA_OP_THREADS,
    DecoderClient,
    GenerationRequest,
)
from anytime_serving.serving.kv_admission import BlockAdmission, CacheCost  # noqa: E402
from anytime_serving.serving.onnx_runtime import extension_available, load_extension  # noqa: E402
from anytime_serving.serving.server import poisson_arrivals  # noqa: E402
from anytime_serving.utils.logger import get_logger  # noqa: E402

LOGGER = get_logger("scripts.run_decode_sweep")

PRECISIONS = ("fp32", "int8", "int4")
# The same utilisations run_load_sweep.py uses, so the two lanes' figures read against
# each other. Past rho = 1 is where the interesting behaviour is.
DEFAULT_UTILISATIONS = (0.4, 0.6, 0.8, 0.95, 1.1, 1.3)
QUICK_UTILISATIONS = (0.95, 1.3)
DEFAULT_REQUESTS = 150
QUICK_REQUESTS = 24
# Primary workload. 256 prompt tokens and 64 generated: long enough that decode
# dominates the request, short enough that a sweep of six rates against three policies
# finishes in one sitting. The prompt-length and output-length trade is measured
# separately by the matrix below rather than by moving this.
DEFAULT_PROMPT_TOKENS = 256
DEFAULT_NEW_TOKENS = 64
DEFAULT_BATCH = 8
# Sequences the preempting policy's arena holds, against a batch width of 8. Half the
# width, so eviction is reachable well before rho = 1 rather than only at the extreme.
PREEMPTING_RESIDENT = 4
DEFAULT_BLOCK_TOKENS = 64
DEFAULT_MAX_CONTEXT = 1024
# Stated absolutely. 50 ms a token is about reading speed; 500 ms to first token is the
# usual threshold for an interaction feeling immediate.
DEFAULT_TTFT_SLO_MS = 500.0
DEFAULT_TPOT_SLO_MS = 50.0
# Prompt/generation pairs for the shape matrix, measured at one utilisation. The last
# one is prefill-light and decode-heavy; the third is the reverse.
SHAPE_MATRIX = ((128, 64), (512, 64), (896, 64), (256, 192))
# Requests spread over four prompt lengths with the same mean as the primary workload,
# to put the padding cost from profile_batching.py under load.
VARIANCE_LENGTHS = (64, 192, 320, 448)
# Wall-clock ceiling for one point. Generous against the ~200s a slow point takes at
# the default workload, tight enough that a mis-measured capacity fails in minutes
# rather than overnight. See `drive` for why a point that overruns is failed rather
# than truncated.
DEFAULT_POINT_BUDGET_S = 900.0


@dataclass
class PolicySpec:
    """A scheduling policy, as an arena size and a batch width.

    `resident_capacity` is how many full sequences of the workload the arena holds.
    It is part of the policy rather than of the host: with the whole generation
    reserved up front, that number is the concurrency limit.
    """

    name: str
    max_batch_size: int
    resident_capacity: int
    use_admission: bool


@dataclass
class WorkloadSpec:
    """What arrives. Fixed lengths, or a spread with a stated mean."""

    label: str
    prompt_tokens: int
    max_new_tokens: int
    requests: int
    spread: tuple[int, ...] = ()

    def lengths(self) -> list[int]:
        if not self.spread:
            return [self.prompt_tokens] * self.requests
        return [self.spread[index % len(self.spread)] for index in range(self.requests)]

    @property
    def mean_prompt_tokens(self) -> float:
        return statistics.fmean(self.lengths())

    @property
    def longest_prompt_tokens(self) -> int:
        return max(self.lengths())


@dataclass
class RequestOutcome:
    """One request, as the client that sent it would have experienced it.

    `ttft_ms` is from arrival to first token and therefore includes every queue the
    request sat in. `GenerationRecord.ttft_ms` is a different number -- the graph time
    of the prefill alone -- and under load the two diverge by however long the request
    waited, which is the quantity this sweep exists to measure.
    """

    policy: str
    target_utilisation: float
    workload: str
    request_id: str
    prompt_tokens: int
    max_new_tokens: int
    arrival_ms: float
    ttft_ms: float
    tpot_p50_ms: float
    e2e_ms: float
    normalised_ms_per_token: float
    output_tokens: int
    preemptions: int
    completed: bool
    rejection_reason: str


@dataclass
class SweepRow:
    """One (policy, utilisation) point."""

    policy: str
    workload: str
    target_utilisation: float
    offered_rps: float
    requests: int
    completed: int
    rejected: int
    arrival_window_s: float
    makespan_s: float
    achieved_rps: float
    output_tokens_per_s: float
    ttft_p50_ms: float
    ttft_p95_ms: float
    ttft_p99_ms: float
    tpot_p50_ms: float
    tpot_p95_ms: float
    tpot_p99_ms: float
    e2e_p50_ms: float
    e2e_p95_ms: float
    slo_attainment: float
    goodput_rps: float
    mean_decode_batch: float
    prefill_steps: int
    decode_steps: int
    preemptions: int
    max_waiting: int
    max_resident: int
    prompt_tokens: int
    mean_prompt_tokens: float
    max_new_tokens: int


@dataclass
class DriveResult:
    """What one run produced: per-request timings and what the schedule looked like."""

    outcomes: list[RequestOutcome]
    makespan_s: float
    arrival_window_s: float
    prefill_steps: int
    decode_steps: int
    mean_decode_batch: float
    preemptions: int
    max_waiting: int
    max_resident: int
    steps: list[tuple[float, float, str, int]] = field(default_factory=list)


def _graph_path(directory: Path) -> Path:
    graphs = sorted(directory.glob("*.onnx"))
    if not graphs:
        raise SystemExit(f"no .onnx graph in {directory}")
    return graphs[0]


def _prompt(length: int, *, vocab: int, seed: int) -> list[int]:
    rng = np.random.default_rng(seed)
    return rng.integers(0, vocab, size=length).astype(np.int64).tolist()


def build_requests(
    workload: WorkloadSpec, *, vocab: int, deadline_ms: float, seed: int
) -> list[GenerationRequest]:
    """The arriving requests. Distinct prompts, so no two rows of a batch are alike."""
    return [
        GenerationRequest(
            prompt=_prompt(length, vocab=vocab, seed=seed + index),
            max_new_tokens=workload.max_new_tokens,
            deadline_ms=deadline_ms,
            request_id=f"{workload.label}-{index}",
        )
        for index, length in enumerate(workload.lengths())
    ]


def blocks_for(*, sequences: int, tokens_each: int, block_tokens: int) -> int:
    return -(-tokens_each // block_tokens) * sequences


def drive(
    scheduler: ContinuousBatchScheduler,
    requests: list[GenerationRequest],
    arrivals: list[float],
    *,
    policy: str,
    utilisation: float,
    workload: str,
    budget_s: float = DEFAULT_POINT_BUDGET_S,
) -> DriveResult:
    """Release each request at its arrival time and step the scheduler in between.

    Open loop: a request arrives when its time comes whether or not anything finished.
    Arrivals are only noticed between iterations, which is a real property of a
    synchronous scheduler rather than an artefact of this driver -- a server can admit
    only between the runs it is already doing.

    While nothing is resident the driver sleeps to the next arrival rather than
    spinning, so an idle interval costs wall time and not CPU.

    `budget_s` bounds how long one point may take. A point's duration is
    `requests / (utilisation * capacity)` and capacity is *measured*, so a capacity
    that comes out low makes the arrival window grow without anything noticing: one
    run took 9h34m against a 28-minute predecessor because of it. Exceeding the
    budget raises rather than truncating the point, because a partial point is a
    latency distribution missing its slowest requests -- which is the half that
    matters -- and reporting it would be worse than not measuring.
    """
    arrival_of: dict[str, float] = {}
    token_times: dict[str, list[float]] = defaultdict(list)
    prefill_steps = 0
    decode_steps = 0
    batched_rows = 0
    max_waiting = 0
    max_resident = 0
    steps: list[tuple[float, float, str, int]] = []

    origin = time.perf_counter()
    submitted = 0
    while submitted < len(requests) or not scheduler.idle():
        elapsed = (time.perf_counter() - origin) * 1000.0
        if elapsed > budget_s * 1000.0:
            raise SystemExit(
                f"{policy} at rho={utilisation} on {workload} exceeded its "
                f"{budget_s:.0f}s budget with {submitted}/{len(requests)} submitted "
                f"and {scheduler.waiting} waiting. A point's length is "
                f"requests / (rho * measured capacity), so a capacity measured far "
                f"below the truth stretches it without bound. Re-measure capacity, or "
                f"raise --point-budget-s if this workload genuinely needs longer."
            )
        while submitted < len(requests) and arrivals[submitted] * 1000.0 <= elapsed:
            request = requests[submitted]
            scheduler.submit(request)
            arrival_of[request.request_id] = arrivals[submitted] * 1000.0
            submitted += 1

        if scheduler.idle():
            if submitted >= len(requests):
                break
            # Nothing to do until the next arrival. Sleeping is what makes this an
            # offered load rather than a backlog drained as fast as possible.
            gap = arrivals[submitted] - (time.perf_counter() - origin)
            if gap > 0:
                time.sleep(gap)
            continue

        max_waiting = max(max_waiting, scheduler.waiting)
        max_resident = max(max_resident, scheduler.resident)
        started = time.perf_counter()
        step = scheduler.step()
        finished = (time.perf_counter() - origin) * 1000.0
        if step is None:
            raise SystemExit(
                f"{policy} at rho={utilisation}: the scheduler had "
                f"{scheduler.waiting} request(s) waiting and "
                f"{len(scheduler.preempted)} preempted but no work to do. That is a "
                f"deadlock rather than an idle moment, so the run is failed instead of "
                f"spinning."
            )
        steps.append(
            (
                round((started - origin) * 1000.0, 3),
                round(finished, 3),
                step.kind,
                step.batch_size,
            )
        )
        if step.kind == "prefill":
            prefill_steps += 1
        else:
            decode_steps += 1
            batched_rows += step.batch_size
            for request_id in step.request_ids:
                token_times[request_id].append(finished)

    makespan_s = time.perf_counter() - origin
    records = scheduler.records()
    outcomes = []
    for request in requests:
        record = records[request.request_id]
        times = token_times.get(request.request_id, [])
        arrival = arrival_of.get(request.request_id, 0.0)
        gaps = [later - earlier for earlier, later in zip(times[:-1], times[1:], strict=True)]
        output_tokens = len(times)
        e2e = times[-1] - arrival if times else 0.0
        outcomes.append(
            RequestOutcome(
                policy=policy,
                target_utilisation=utilisation,
                workload=workload,
                request_id=request.request_id,
                prompt_tokens=request.prompt_tokens,
                max_new_tokens=request.max_new_tokens,
                arrival_ms=round(arrival, 3),
                ttft_ms=round(times[0] - arrival, 3) if times else 0.0,
                tpot_p50_ms=round(statistics.median(gaps), 3) if gaps else 0.0,
                e2e_ms=round(e2e, 3),
                normalised_ms_per_token=round(e2e / output_tokens, 3) if output_tokens else 0.0,
                output_tokens=output_tokens,
                preemptions=record.preemptions,
                completed=bool(times) and output_tokens >= request.max_new_tokens,
                rejection_reason=record.rejection_reason,
            )
        )

    return DriveResult(
        outcomes=outcomes,
        makespan_s=makespan_s,
        arrival_window_s=arrivals[-1] if arrivals else 0.0,
        prefill_steps=prefill_steps,
        decode_steps=decode_steps,
        mean_decode_batch=round(batched_rows / decode_steps, 3) if decode_steps else 0.0,
        preemptions=scheduler.stats.preemptions,
        max_waiting=max_waiting,
        max_resident=max_resident,
        steps=steps,
    )


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    return round(float(np.percentile(np.asarray(values, dtype=np.float64), q)), 3)


def meets_slo(outcome: RequestOutcome, *, ttft_slo_ms: float, tpot_slo_ms: float) -> bool:
    """Both halves, or neither counts.

    A request that arrived promptly and then trickled has not been served well, and
    neither has one that streamed smoothly after a two-second wait. Reporting them
    separately as well as together is why both percentiles are in the row.
    """
    if not outcome.completed:
        return False
    return outcome.ttft_ms <= ttft_slo_ms and outcome.tpot_p50_ms <= tpot_slo_ms


def summarise(
    result: DriveResult,
    *,
    policy: str,
    workload: WorkloadSpec,
    utilisation: float,
    offered_rps: float,
    ttft_slo_ms: float,
    tpot_slo_ms: float,
) -> SweepRow:
    """Aggregate one point.

    Percentiles are over completed requests: a request that never finished has no time
    per output token, and counting it as zero would improve the tail by dropping the
    traffic that suffered most. It is counted in `completed` and in the attainment
    denominator instead, which is where an unserved request belongs.
    """
    completed = [outcome for outcome in result.outcomes if outcome.completed]
    ttfts = [outcome.ttft_ms for outcome in completed]
    tpots = [outcome.tpot_p50_ms for outcome in completed]
    e2es = [outcome.e2e_ms for outcome in completed]
    met = [
        outcome
        for outcome in result.outcomes
        if meets_slo(outcome, ttft_slo_ms=ttft_slo_ms, tpot_slo_ms=tpot_slo_ms)
    ]
    tokens = sum(outcome.output_tokens for outcome in result.outcomes)
    window = max(result.makespan_s, 1e-9)
    return SweepRow(
        policy=policy,
        workload=workload.label,
        target_utilisation=utilisation,
        offered_rps=round(offered_rps, 4),
        requests=len(result.outcomes),
        completed=len(completed),
        rejected=sum(1 for outcome in result.outcomes if outcome.rejection_reason),
        arrival_window_s=round(result.arrival_window_s, 3),
        makespan_s=round(result.makespan_s, 3),
        achieved_rps=round(len(completed) / window, 4),
        output_tokens_per_s=round(tokens / window, 2),
        ttft_p50_ms=_percentile(ttfts, 50),
        ttft_p95_ms=_percentile(ttfts, 95),
        ttft_p99_ms=_percentile(ttfts, 99),
        tpot_p50_ms=_percentile(tpots, 50),
        tpot_p95_ms=_percentile(tpots, 95),
        tpot_p99_ms=_percentile(tpots, 99),
        e2e_p50_ms=_percentile(e2es, 50),
        e2e_p95_ms=_percentile(e2es, 95),
        slo_attainment=round(len(met) / len(result.outcomes), 4) if result.outcomes else 0.0,
        goodput_rps=round(len(met) / window, 4),
        mean_decode_batch=result.mean_decode_batch,
        prefill_steps=result.prefill_steps,
        decode_steps=result.decode_steps,
        preemptions=result.preemptions,
        max_waiting=result.max_waiting,
        max_resident=result.max_resident,
        prompt_tokens=workload.prompt_tokens,
        mean_prompt_tokens=round(workload.mean_prompt_tokens, 1),
        max_new_tokens=workload.max_new_tokens,
    )


def _cost_model(path: Path, precision: str) -> CacheCost:
    """The fitted cost model the eviction policy needs.

    Read from `profile_decode.py`'s output rather than written down here. A policy
    weighing recompute against deadline slack with made-up coefficients would evict the
    wrong sequences and look like it was working, which is the failure this project has
    already had once.
    """
    if not path.is_file():
        raise SystemExit(
            f"{path} is missing and the preempting policy needs the fitted cost model "
            f"in it. Measure it first:\n    python scripts/profile_decode.py"
        )
    data = json.loads(path.read_text())
    for profile in data.get("precisions", []):
        if profile["precision"] == precision and profile.get("cache_cost"):
            fit = profile["cache_cost"]
            return CacheCost(
                decode_base_ms=float(fit["decode_base_ms"]),
                decode_per_token_ms=float(fit["decode_per_token_ms"]),
                prefill_per_token_ms=float(fit["prefill_per_token_ms"]),
            )
    raise SystemExit(
        f"{path} has no fitted cost model for {precision}. Measure it with:\n"
        f"    python scripts/profile_decode.py --precisions {precision}"
    )


def host_metadata(intra_op_threads: int) -> dict[str, object]:
    """What the run was taken on, including the settings that change the numbers.

    `intra_op_num_threads` is read from the run rather than written as a constant. It
    was a hardcoded 1 while the thread count was not configurable, and leaving it that
    way once it was would have made every recorded artefact describe a configuration
    it had not been measured under.
    """
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "backend": "extension",
        "onnxruntime": load_extension().onnxruntime_version(),
        "intra_op_num_threads": intra_op_threads,
    }


def _client(
    graph: Path,
    *,
    policy: PolicySpec,
    tokens_each: int,
    block_tokens: int,
    max_context: int,
    cost: CacheCost | None,
    intra_op_threads: int = 1,
) -> DecoderClient:
    blocks = blocks_for(
        sequences=policy.resident_capacity, tokens_each=tokens_each, block_tokens=block_tokens
    )
    admission = None
    if policy.use_admission:
        if cost is None:
            raise SystemExit(f"policy {policy.name} needs a cost model and none was loaded")
        admission = BlockAdmission(capacity_blocks=blocks, block_tokens=block_tokens, cost=cost)
    return DecoderClient(
        graph,
        block_tokens=block_tokens,
        num_blocks=blocks,
        admission=admission,
        max_context_tokens=max_context,
        intra_op_threads=intra_op_threads,
    )


def measure_reference(
    graph: Path,
    *,
    workload: WorkloadSpec,
    vocab: int,
    block_tokens: int,
    max_context: int,
    deadline_ms: float,
    intra_op_threads: int = 1,
) -> tuple[float, float]:
    """Time to first token and time per output token for one request, alone.

    Not the SLO -- the SLO is stated absolutely, because deriving it from this would
    define the target by the absence of batching. This is the reference the loaded
    percentiles are read against.
    """
    policy = PolicySpec("reference", max_batch_size=1, resident_capacity=1, use_admission=False)
    tokens_each = min(max_context, workload.longest_prompt_tokens + workload.max_new_tokens)
    with _client(
        graph,
        policy=policy,
        tokens_each=tokens_each,
        block_tokens=block_tokens,
        max_context=max_context,
        cost=None,
        intra_op_threads=intra_op_threads,
    ) as client:
        requests = build_requests(
            WorkloadSpec(
                label="reference",
                prompt_tokens=workload.prompt_tokens,
                max_new_tokens=workload.max_new_tokens,
                requests=2,
            ),
            vocab=vocab,
            deadline_ms=deadline_ms,
            seed=1,
        )
        ttfts: list[float] = []
        tpots: list[float] = []
        for index, request in enumerate(requests):
            scheduler = ContinuousBatchScheduler(client, max_batch_size=1)
            result = drive(
                scheduler,
                [request],
                [0.0],
                policy="reference",
                utilisation=0.0,
                workload="reference",
            )
            if index == 0:
                # First request of a session sizes the staging buffers.
                continue
            ttfts.append(result.outcomes[0].ttft_ms)
            tpots.append(result.outcomes[0].tpot_p50_ms)
    return statistics.median(ttfts), statistics.median(tpots)


def measure_capacity(
    graph: Path,
    *,
    workload: WorkloadSpec,
    policy: PolicySpec,
    vocab: int,
    block_tokens: int,
    max_context: int,
    deadline_ms: float,
    requests: int,
    intra_op_threads: int = 1,
) -> float:
    """Completion rate with a full backlog: the capacity the sweep is expressed against.

    Measured rather than derived. A closed-loop backlog is the most favourable arrival
    pattern there is -- the batch is always as full as the arena allows -- so a Poisson
    stream at rho = 1 of this figure will not achieve it, which is the point of
    expressing load as a fraction of it.
    """
    tokens_each = min(max_context, workload.longest_prompt_tokens + workload.max_new_tokens)
    with _client(
        graph,
        policy=policy,
        tokens_each=tokens_each,
        block_tokens=block_tokens,
        max_context=max_context,
        cost=None,
        intra_op_threads=intra_op_threads,
    ) as client:
        spec = WorkloadSpec(
            label="capacity",
            prompt_tokens=workload.prompt_tokens,
            max_new_tokens=workload.max_new_tokens,
            requests=requests,
            spread=workload.spread,
        )
        scheduler = ContinuousBatchScheduler(client, max_batch_size=policy.max_batch_size)
        result = drive(
            scheduler,
            build_requests(spec, vocab=vocab, deadline_ms=deadline_ms, seed=1),
            [0.0] * requests,
            policy="capacity",
            utilisation=0.0,
            workload="capacity",
        )
    completed = sum(1 for outcome in result.outcomes if outcome.completed)
    return completed / max(result.makespan_s, 1e-9)


def probe_vocabulary(
    graph: Path, *, block_tokens: int, max_context: int, intra_op_threads: int = 1
) -> int:
    """The width of the logits row, learned by running one throwaway prefill."""
    with DecoderClient(
        graph,
        block_tokens=block_tokens,
        num_blocks=8,
        max_context_tokens=max_context,
        intra_op_threads=intra_op_threads,
    ) as client:
        request_id = "vocab-probe"
        client.admit(
            GenerationRequest(
                prompt=[0] * 8, max_new_tokens=1, deadline_ms=1e9, request_id=request_id
            )
        )
        client.prefill(request_id)
        return int(client.next_token_logits(request_id).size)


def run_point(
    graph: Path,
    *,
    policy: PolicySpec,
    workload: WorkloadSpec,
    utilisation: float,
    capacity_rps: float,
    vocab: int,
    block_tokens: int,
    max_context: int,
    deadline_ms: float,
    ttft_slo_ms: float,
    tpot_slo_ms: float,
    cost: CacheCost | None,
    seed: int,
    intra_op_threads: int = 1,
    budget_s: float = DEFAULT_POINT_BUDGET_S,
) -> tuple[SweepRow, list[RequestOutcome], DriveResult]:
    """One policy at one utilisation, over its own fresh arena."""
    offered_rps = utilisation * capacity_rps
    arrivals = poisson_arrivals(
        workload.requests / max(offered_rps, 1e-9),
        offered_rps,
        rng=np.random.default_rng(seed),
    )[: workload.requests]
    if len(arrivals) < workload.requests:
        # The Poisson draw came up short of the request count; pad the window rather
        # than measuring fewer requests at this point than at the others.
        gap = 1.0 / max(offered_rps, 1e-9)
        last = arrivals[-1] if arrivals else 0.0
        arrivals = arrivals + [
            last + gap * (index + 1) for index in range(workload.requests - len(arrivals))
        ]

    tokens_each = min(max_context, workload.longest_prompt_tokens + workload.max_new_tokens)
    with _client(
        graph,
        policy=policy,
        tokens_each=tokens_each,
        block_tokens=block_tokens,
        max_context=max_context,
        cost=cost,
        intra_op_threads=intra_op_threads,
    ) as client:
        scheduler = ContinuousBatchScheduler(client, max_batch_size=policy.max_batch_size)
        result = drive(
            scheduler,
            build_requests(workload, vocab=vocab, deadline_ms=deadline_ms, seed=seed + 1000),
            arrivals,
            policy=policy.name,
            utilisation=utilisation,
            workload=workload.label,
            budget_s=budget_s,
        )
    row = summarise(
        result,
        policy=policy.name,
        workload=workload,
        utilisation=utilisation,
        offered_rps=offered_rps,
        ttft_slo_ms=ttft_slo_ms,
        tpot_slo_ms=tpot_slo_ms,
    )
    return row, result.outcomes, result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="gpt2")
    parser.add_argument("--model-dir", type=Path, default=Path("models/onnx"))
    parser.add_argument("--precision", default="fp32", choices=PRECISIONS)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument(
        "--decode-profiles",
        type=Path,
        default=Path("results/decode_profiles.json"),
        help="Where the fitted cost model the eviction policy needs is read from",
    )
    parser.add_argument("--utilisations", nargs="+", type=float, default=None)
    parser.add_argument("--requests", type=int, default=DEFAULT_REQUESTS)
    parser.add_argument("--prompt-tokens", type=int, default=DEFAULT_PROMPT_TOKENS)
    parser.add_argument("--new-tokens", type=int, default=DEFAULT_NEW_TOKENS)
    parser.add_argument("--batch", type=int, default=DEFAULT_BATCH)
    parser.add_argument("--block-tokens", type=int, default=DEFAULT_BLOCK_TOKENS)
    parser.add_argument("--max-context", type=int, default=DEFAULT_MAX_CONTEXT)
    parser.add_argument(
        "--point-budget-s",
        type=float,
        default=DEFAULT_POINT_BUDGET_S,
        help=(
            "Wall-clock ceiling for one point, after which the run fails rather than "
            f"continuing (default {DEFAULT_POINT_BUDGET_S:.0f})"
        ),
    )
    parser.add_argument(
        "--intra-op-threads",
        type=int,
        default=DEFAULT_INTRA_OP_THREADS,
        help=(
            "Threads ONNX Runtime may use inside one operator. Defaults to what the "
            "serving path uses. Capacity is measured under the same setting as the "
            "sweep, so a load fraction stays a fraction of what this configuration "
            f"can actually do (default {DEFAULT_INTRA_OP_THREADS})"
        ),
    )
    parser.add_argument("--ttft-slo-ms", type=float, default=DEFAULT_TTFT_SLO_MS)
    parser.add_argument("--tpot-slo-ms", type=float, default=DEFAULT_TPOT_SLO_MS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--skip-shapes", action="store_true", help="Skip the prompt/generation matrix"
    )
    parser.add_argument(
        "--quick", action="store_true", help="Two utilisations and far fewer requests"
    )
    args = parser.parse_args()

    if not extension_available():
        raise SystemExit(
            "anytime_runtime is not available. Batched decoding is the extension, so "
            "there is nothing to sweep without it. Build it with:\n    pip install -e ."
        )
    if args.intra_op_threads < 1:
        raise SystemExit("--intra-op-threads must be at least 1")

    directory = args.model_dir / f"decoder_{args.model}_{args.precision}"
    if not directory.is_dir():
        raise SystemExit(
            f"{directory} is missing. Export it with:\n"
            f"    python scripts/export_decoder.py --precisions {args.precision}"
        )
    graph = _graph_path(directory)

    utilisations = tuple(
        args.utilisations
        if args.utilisations
        else (QUICK_UTILISATIONS if args.quick else DEFAULT_UTILISATIONS)
    )
    requests = QUICK_REQUESTS if args.quick else args.requests
    primary = WorkloadSpec(
        label="fixed",
        prompt_tokens=args.prompt_tokens,
        max_new_tokens=args.new_tokens,
        requests=requests,
    )
    if primary.longest_prompt_tokens + primary.max_new_tokens > args.max_context:
        raise SystemExit(
            f"a {primary.prompt_tokens}-token prompt plus {primary.max_new_tokens} "
            f"generated tokens is past the model's {args.max_context} positions"
        )

    # The deadline the eviction policy reasons with: what the stated SLO implies for a
    # whole generation. A deadline unrelated to the SLO would have the policy protecting
    # sequences the SLO does not care about.
    deadline_ms = args.ttft_slo_ms + args.new_tokens * args.tpot_slo_ms
    cost = _cost_model(args.decode_profiles, args.precision)
    vocab = probe_vocabulary(
        graph,
        block_tokens=args.block_tokens,
        max_context=args.max_context,
        intra_op_threads=args.intra_op_threads,
    )

    policies = (
        PolicySpec("serial", max_batch_size=1, resident_capacity=1, use_admission=False),
        PolicySpec(
            f"batched-{args.batch}",
            max_batch_size=args.batch,
            resident_capacity=args.batch,
            use_admission=False,
        ),
        PolicySpec(
            f"batched-{args.batch}-preempting",
            max_batch_size=args.batch,
            resident_capacity=min(PREEMPTING_RESIDENT, args.batch),
            use_admission=True,
        ),
    )
    batched = policies[1]

    LOGGER.info(
        "%s %s: %d-token prompts, %d generated, batch %d, %d request(s) per point",
        args.model,
        args.precision,
        primary.prompt_tokens,
        primary.max_new_tokens,
        args.batch,
        requests,
    )
    reference_ttft, reference_tpot = measure_reference(
        graph,
        workload=primary,
        vocab=vocab,
        block_tokens=args.block_tokens,
        max_context=args.max_context,
        deadline_ms=deadline_ms,
        intra_op_threads=args.intra_op_threads,
    )
    LOGGER.info(
        "Unloaded: %.1f ms to first token, %.2f ms a token. SLO is %.0f / %.0f ms, "
        "stated rather than derived from those",
        reference_ttft,
        reference_tpot,
        args.ttft_slo_ms,
        args.tpot_slo_ms,
    )

    capacity_rps = measure_capacity(
        graph,
        workload=primary,
        policy=batched,
        vocab=vocab,
        block_tokens=args.block_tokens,
        max_context=args.max_context,
        deadline_ms=deadline_ms,
        requests=max(args.batch * 2, 8),
        intra_op_threads=args.intra_op_threads,
    )
    LOGGER.info(
        "Capacity with a full backlog under %s: %.3f completions/s. Load below is a "
        "fraction of that",
        batched.name,
        capacity_rps,
    )

    rows: list[SweepRow] = []
    outcomes: list[RequestOutcome] = []
    for utilisation in utilisations:
        LOGGER.info("rho=%.2f  offered=%.3f rps", utilisation, utilisation * capacity_rps)
        for policy in policies:
            row, per_request, _ = run_point(
                graph,
                policy=policy,
                workload=primary,
                utilisation=utilisation,
                capacity_rps=capacity_rps,
                vocab=vocab,
                block_tokens=args.block_tokens,
                max_context=args.max_context,
                deadline_ms=deadline_ms,
                ttft_slo_ms=args.ttft_slo_ms,
                tpot_slo_ms=args.tpot_slo_ms,
                cost=cost,
                seed=args.seed,
                intra_op_threads=args.intra_op_threads,
                budget_s=args.point_budget_s,
            )
            rows.append(row)
            outcomes.extend(per_request)
            LOGGER.info(
                "  %-24s completed %3d/%3d  TTFT p50 %7.0f p95 %8.0f  TPOT p50 %6.1f "
                "p95 %6.1f  attainment %.3f  goodput %.2f rps  batch %.2f  preempted %d",
                policy.name,
                row.completed,
                row.requests,
                row.ttft_p50_ms,
                row.ttft_p95_ms,
                row.tpot_p50_ms,
                row.tpot_p95_ms,
                row.slo_attainment,
                row.goodput_rps,
                row.mean_decode_batch,
                row.preemptions,
            )

    shape_rows: list[SweepRow] = []
    if not args.skip_shapes:
        shape_utilisation = 0.95
        shapes = [(args.prompt_tokens, args.new_tokens), *SHAPE_MATRIX]
        LOGGER.info("Prompt and generation shapes at rho=%.2f, %s", shape_utilisation, batched.name)
        for prompt_tokens, new_tokens in shapes:
            if prompt_tokens + new_tokens > args.max_context:
                LOGGER.info("  skipped %d/%d: past the position limit", prompt_tokens, new_tokens)
                continue
            spec = WorkloadSpec(
                label=f"shape-{prompt_tokens}-{new_tokens}",
                prompt_tokens=prompt_tokens,
                max_new_tokens=new_tokens,
                requests=max(requests // 2, 8),
            )
            shape_capacity = measure_capacity(
                graph,
                workload=spec,
                policy=batched,
                vocab=vocab,
                block_tokens=args.block_tokens,
                max_context=args.max_context,
                deadline_ms=args.ttft_slo_ms + new_tokens * args.tpot_slo_ms,
                requests=max(args.batch * 2, 8),
                intra_op_threads=args.intra_op_threads,
            )
            row, per_request, _ = run_point(
                graph,
                policy=batched,
                workload=spec,
                utilisation=shape_utilisation,
                capacity_rps=shape_capacity,
                vocab=vocab,
                block_tokens=args.block_tokens,
                max_context=args.max_context,
                deadline_ms=args.ttft_slo_ms + new_tokens * args.tpot_slo_ms,
                ttft_slo_ms=args.ttft_slo_ms,
                tpot_slo_ms=args.tpot_slo_ms,
                cost=cost,
                seed=args.seed,
                intra_op_threads=args.intra_op_threads,
                budget_s=args.point_budget_s,
            )
            shape_rows.append(row)
            outcomes.extend(per_request)
            LOGGER.info(
                "  %4d prompt / %3d generated: capacity %.3f rps  TTFT p95 %8.0f  "
                "TPOT p95 %6.1f  attainment %.3f  %.1f tok/s",
                prompt_tokens,
                new_tokens,
                shape_capacity,
                row.ttft_p95_ms,
                row.tpot_p95_ms,
                row.slo_attainment,
                row.output_tokens_per_s,
            )

        # The same mean prompt length, spread four ways, to put the padding cost from
        # profile_batching.py under load.
        variance = WorkloadSpec(
            label="variance",
            prompt_tokens=int(statistics.fmean(VARIANCE_LENGTHS)),
            max_new_tokens=args.new_tokens,
            requests=max(requests // 2, 8),
            spread=VARIANCE_LENGTHS,
        )
        variance_capacity = measure_capacity(
            graph,
            workload=variance,
            policy=batched,
            vocab=vocab,
            block_tokens=args.block_tokens,
            max_context=args.max_context,
            deadline_ms=deadline_ms,
            requests=max(args.batch * 2, 8),
            intra_op_threads=args.intra_op_threads,
        )
        row, per_request, _ = run_point(
            graph,
            policy=batched,
            workload=variance,
            utilisation=shape_utilisation,
            capacity_rps=variance_capacity,
            vocab=vocab,
            block_tokens=args.block_tokens,
            max_context=args.max_context,
            deadline_ms=deadline_ms,
            ttft_slo_ms=args.ttft_slo_ms,
            tpot_slo_ms=args.tpot_slo_ms,
            cost=cost,
            seed=args.seed,
            intra_op_threads=args.intra_op_threads,
            budget_s=args.point_budget_s,
        )
        shape_rows.append(row)
        outcomes.extend(per_request)
        LOGGER.info(
            "  prompts spread %s (mean %.0f): capacity %.3f rps  TPOT p95 %6.1f  "
            "attainment %.3f  %.1f tok/s",
            list(VARIANCE_LENGTHS),
            variance.mean_prompt_tokens,
            variance_capacity,
            row.tpot_p95_ms,
            row.slo_attainment,
            row.output_tokens_per_s,
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "host": host_metadata(args.intra_op_threads),
        "model": args.model,
        "precision": args.precision,
        "requests_per_point": requests,
        "block_tokens": args.block_tokens,
        "max_context_tokens": args.max_context,
        "batch_size": args.batch,
        "preempting_resident_capacity": min(PREEMPTING_RESIDENT, args.batch),
        "arrival_process": "poisson, open loop",
        "capacity_rps": round(capacity_rps, 4),
        "capacity_measured_under": batched.name,
        "reference_ttft_ms": round(reference_ttft, 3),
        "reference_tpot_ms": round(reference_tpot, 3),
        "ttft_slo_ms": args.ttft_slo_ms,
        "tpot_slo_ms": args.tpot_slo_ms,
        "deadline_ms": deadline_ms,
        "policies": [asdict(policy) for policy in policies],
        "sweep": [asdict(row) for row in rows],
        "shapes": [asdict(row) for row in shape_rows],
    }
    summary_path = args.output_dir / "decode_sweep.json"
    summary_path.write_text(json.dumps(payload, indent=2) + "\n")
    LOGGER.info("Wrote %s", summary_path)

    requests_path = args.output_dir / "decode_sweep_requests.csv"
    with requests_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(outcomes[0]).keys()))
        writer.writeheader()
        for outcome in outcomes:
            writer.writerow(asdict(outcome))
    LOGGER.info("Wrote %s (%d request(s))", requests_path, len(outcomes))

    LOGGER.info("")
    LOGGER.info(
        "Load is a fraction of the completion rate the batched policy sustains with a "
        "full backlog (%.3f rps), measured before the sweep. Every policy saw the same "
        "arrivals and the same prompts. TTFT is from arrival, so it includes queueing; "
        "TPOT is the median interval between one sequence's tokens. Attainment counts a "
        "request only if it met both targets and finished.",
        capacity_rps,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
