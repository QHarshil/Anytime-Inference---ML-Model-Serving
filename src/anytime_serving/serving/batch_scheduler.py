"""Continuous batching over the decoder path: alternate, do not fuse.

What the graph allows
---------------------

An optimum-exported decoder takes one `sequence` dimension as well as one
`past_sequence_length`. So a prefill chunk and a decode step cannot share a `Run`:
a 256-token chunk beside a one-token step is not merely a wasteful pairing, it is
unrepresentable without padding the decode row out to 256. vLLM and SARATHI fuse the
two with a flattened varlen layout and custom kernels; a stock exported graph has
neither.

This scheduler therefore **alternates**. Each iteration is either one prefill chunk
or one batched decode step, never both. That is a consequence of the graph rather
than a preference, and it is what makes chunk width the central tuning knob here:
while a chunk runs, every resident sequence waits.

The trade chunk width sets
--------------------------

At the 256-token default, one chunk of GPT-2 at FP32 is about 93 ms -- a quarter of
the 372.2 ms a 1024-token prefill takes. So a decode step queued behind a chunk waits
up to that long, against the 9.5 ms it would take on its own. Narrower chunks cut
that stall and lengthen time to first token, because chunked prefill's advantage
stops improving and the per-run overhead starts telling. Wider chunks do the reverse.
`chunk_tokens` is the knob and `prefill_chunks_per_decode` is the other half of it.

What batching is worth
----------------------

Measured through this scheduler on GPT-2 at FP32 with the decoder session on eight
threads, against the same sequences stepped one at a time: at batch 8, 3.00x the tokens
per second at 128 cached tokens, 2.15x at 512, 1.67x at 960; at batch 32, 3.47x / 2.26x
/ 1.71x. The gain decays with cache occupancy because only the cache-independent term of
a decode step amortises across a batch, while the per-cached-token term is per sequence
and grows with the batch's total cache.

Thread count belongs in that sentence rather than beside it. With the session pinned to
one thread the same points read 2.32x / 1.52x / 1.25x at batch 8: a batch-1 decode is a
skinny GEMV with little for a thread pool to divide, while a wide batch is a real GEMM,
so batching supplies the parallelism that threading exploits and the two compound.

So the throughput win is real and it stops: returns flatten by batch 16 at FP32 and
INT8. Which means this scheduler's other job -- deciding who waits -- is still the
larger half of what it is for. Under an open-loop arrival sweep, batching holds time to
first token near its unloaded value where one-at-a-time decoding collapses, and past
saturation the configuration that additionally *limits* how many sequences are resident
wins again by as much: 71% of requests meeting their targets against 13%.
`docs/benchmarks.md` has both, and the caveat that a batched-against-serial ratio is
scaled to the batched policy's own capacity.

What length bucketing is for
----------------------------

A batched step runs every row at the longest row's cached length, so a batch holding one
long sequence charges every short one for positions it does not have. Measured at a 4:1
spread that is 9.0 ms of a 27.7 ms step, and only 1.0 ms of it is clearing the padding --
the rest is the graph doing arithmetic on absent tokens. `length_bucketing` chooses *who
shares a step* to shrink that.

It can only help when more sequences are decoding than fit in one step. With an arena
sized to the batch width every batch holds everyone, and the ordering rule cannot change
what the step costs.

Fairness is the part admission already owned
--------------------------------------------

Who is resident is `kv_admission.BlockAdmission`'s decision, not this module's. The
scheduler asks, carries out the eviction plan it is given, and puts the victims back
when there is room. What is added here is only the ordering: which of the admitted
sequences runs next, and whether this iteration goes to a prefill chunk or to a
decode step.
"""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass

from ..utils.logger import get_logger
from .decoder import DecoderClient, GenerationRecord, GenerationRequest, StepRecord

LOGGER = get_logger("serving.batch_scheduler")

__all__ = [
    "ContinuousBatchScheduler",
    "SchedulerStep",
    "SchedulerStats",
]


@dataclass(frozen=True)
class SchedulerStep:
    """What one iteration of the scheduler did.

    `kind` is "prefill" or "decode". There is no kind for admitting or readmitting:
    those consume no graph invocation, so reporting them as iterations would make the
    step count stop meaning "runs of the model".
    """

    kind: str
    request_ids: tuple[str, ...]
    records: tuple[StepRecord, ...]
    # Sequences that finished on this step, and sequences preempted to make room for
    # it. Both are how a caller learns about work it did not ask for.
    completed: tuple[str, ...] = ()
    preempted: tuple[str, ...] = ()

    @property
    def batch_size(self) -> int:
        return len(self.request_ids)

    @property
    def total_ms(self) -> float:
        """Wall time of the graph invocation this iteration made.

        One number, not a sum over records: a batched decode step's records all carry
        the same duration because they all waited for the same `Run`.
        """
        return self.records[0].total_ms if self.records else 0.0


@dataclass
class SchedulerStats:
    """Counts worth having without walking every record."""

    prefill_steps: int = 0
    decode_steps: int = 0
    tokens_emitted: int = 0
    preemptions: int = 0
    rejections: int = 0
    # Decode steps summed by how many sequences were in them, so the mean batch size
    # is derivable. A scheduler that batched one sequence at a time would otherwise
    # look identical to one that batched eight.
    batched_rows: int = 0

    @property
    def mean_decode_batch(self) -> float:
        return self.batched_rows / self.decode_steps if self.decode_steps else 0.0


class ContinuousBatchScheduler:
    """Runs many generations over one arena, one chunk or one batched step at a time.

    Deliberately synchronous and single-threaded. The arena and the graph are shared
    mutable state, and a scheduler is a sequence of decisions about them; making the
    loop concurrent would mean locking both and would not add throughput, because
    `Run` is where the time goes and batching is how this path uses more of the
    machine.
    """

    def __init__(
        self,
        client: DecoderClient,
        *,
        chunk_tokens: int | None = None,
        max_batch_size: int = 8,
        prefill_chunks_per_decode: int = 1,
        length_bucketing: bool = False,
    ) -> None:
        if max_batch_size <= 0:
            raise ValueError("max_batch_size must be positive")
        if prefill_chunks_per_decode <= 0:
            raise ValueError(
                "prefill_chunks_per_decode must be positive; zero would admit "
                "sequences and never run their prompts"
            )
        if not client.reserves_full_generation:
            raise ValueError(
                "the scheduler needs a client that reserves the full generation. "
                "Otherwise a resident sequence can outgrow its blocks mid-batch, and "
                "a decode step over eight sequences would exhaust the arena on behalf "
                "of one of them with no way to say which."
            )

        self._client = client
        self._chunk_tokens = client.default_chunk_tokens if chunk_tokens is None else chunk_tokens
        self._max_batch_size = max_batch_size
        self._prefill_chunks_per_decode = prefill_chunks_per_decode
        self._length_bucketing = length_bucketing

        self._waiting: deque[GenerationRequest] = deque()
        # Admitted, prompt not yet fully cached. Includes sequences re-running a
        # history after preemption, because that is the same work.
        self._pending: deque[str] = deque()
        self._decoding: list[str] = []
        # Lost their blocks and are waiting for room to come back.
        self._preempted: list[str] = []
        self._records: dict[str, GenerationRecord] = {}
        self._started_at: dict[str, float] = {}
        self._chunks_since_decode = 0
        self._stats = SchedulerStats()

    # --- state ---------------------------------------------------------------

    @property
    def stats(self) -> SchedulerStats:
        return self._stats

    @property
    def chunk_tokens(self) -> int:
        """Prefill chunk width, which is also the decode-jitter knob."""
        return self._chunk_tokens

    @property
    def length_bucketing(self) -> bool:
        """Whether a decode batch is filled by cached length or by queue order."""
        return self._length_bucketing

    @property
    def waiting(self) -> int:
        return len(self._waiting)

    @property
    def resident(self) -> int:
        """Sequences holding blocks, whether prefilling or decoding."""
        return len(self._pending) + len(self._decoding)

    @property
    def decoding(self) -> tuple[str, ...]:
        return tuple(self._decoding)

    @property
    def preempted(self) -> tuple[str, ...]:
        return tuple(self._preempted)

    def record(self, request_id: str) -> GenerationRecord:
        return self._records[request_id]

    def records(self) -> dict[str, GenerationRecord]:
        return dict(self._records)

    def idle(self) -> bool:
        """Whether there is nothing left to do."""
        return not (self._waiting or self._pending or self._decoding or self._preempted)

    # --- driving --------------------------------------------------------------

    def submit(self, request: GenerationRequest) -> None:
        """Queue a generation. Nothing runs until `step` is called."""
        if request.request_id in self._records:
            raise ValueError(f"request {request.request_id} has already been submitted")
        self._records[request.request_id] = GenerationRecord(
            request_id=request.request_id,
            admitted=False,
            prompt_tokens=request.prompt_tokens,
            deadline_ms=request.deadline_ms,
        )
        self._waiting.append(request)

    def step(self) -> SchedulerStep | None:
        """Do one unit of work: one prefill chunk, or one batched decode step.

        Returns None when there is nothing to do, which is also what `idle` reports.
        Admission and readmission happen first and are not iterations of their own;
        they cost no graph invocation.
        """
        preempted = self._admit_waiting()
        self._readmit_preempted()

        if self._should_prefill():
            return self._run_prefill_chunk(preempted)
        if self._decoding:
            return self._run_decode_batch(preempted)
        return None

    def drain(self, *, max_steps: int | None = None) -> dict[str, GenerationRecord]:
        """Step until nothing is left, or until `max_steps` iterations have run.

        `max_steps` is a guard for a caller that would rather fail than hang; reaching
        it is not an error and the records describe how far things got.
        """
        steps = 0
        while not self.idle():
            if max_steps is not None and steps >= max_steps:
                break
            if self.step() is None:
                break
            steps += 1
        return self.records()

    # --- internals ------------------------------------------------------------

    def _should_prefill(self) -> bool:
        """Whether this iteration goes to a prefill chunk rather than a decode step.

        With nothing decoding, prefill runs unconditionally. With both available they
        alternate: `prefill_chunks_per_decode` chunks, then a decode step. Strict
        alternation at the default of 1 bounds how long a resident sequence waits to
        one chunk plus one step, which is the guarantee chunk width is chosen against.
        Raising it favours time to first token and widens decode jitter by the same
        arithmetic.
        """
        if not self._pending:
            return False
        if not self._decoding:
            return True
        return self._chunks_since_decode < self._prefill_chunks_per_decode

    def _admit_waiting(self) -> tuple[str, ...]:
        """Admit what fits, in arrival order, and carry out any eviction plan.

        Arrival order rather than shortest-first: reordering by size would starve long
        prompts under load, and the deadline reasoning that would justify a different
        order already lives in the admission policy rather than here.
        """
        preempted: list[str] = []
        while self._waiting:
            request = self._waiting[0]
            plan = self._client.admit(request)
            if not plan.admit:
                record = self._records[request.request_id]
                if plan.blocks_needed > self._client.capacity_blocks:
                    # No eviction can ever make room, so queueing it forever would be
                    # a hang dressed as backpressure.
                    self._waiting.popleft()
                    record.rejection_reason = plan.reason
                    record.admitted = False
                    self._stats.rejections += 1
                    LOGGER.info("rejected %s: %s", request.request_id, plan.reason)
                    continue
                # It might fit once something finishes. Stop here rather than trying
                # later arrivals, so admission stays first-come.
                break

            self._waiting.popleft()
            record = self._records[request.request_id]
            record.admitted = True
            self._started_at[request.request_id] = time.perf_counter()
            self._pending.append(request.request_id)
            for victim in plan.evict:
                self._move_to_preempted(victim)
                preempted.append(victim)
                self._stats.preemptions += 1
        return tuple(preempted)

    def _readmit_preempted(self) -> None:
        """Give blocks back to preempted sequences, oldest first.

        Oldest first because a sequence that has been out longest has already paid the
        most waiting, and its recompute cost does not shrink by leaving it there.
        """
        still_out = []
        for request_id in self._preempted:
            if self._client.readmit(request_id):
                self._pending.append(request_id)
            else:
                still_out.append(request_id)
        self._preempted = still_out

    def _move_to_preempted(self, request_id: str) -> None:
        if request_id in self._decoding:
            self._decoding.remove(request_id)
        elif request_id in self._pending:
            self._pending.remove(request_id)
        else:
            return
        self._preempted.append(request_id)
        self._records[request_id].preemptions += 1

    def _run_prefill_chunk(self, preempted: tuple[str, ...]) -> SchedulerStep:
        request_id = self._pending[0]
        record = self._client.prefill_chunk(request_id, chunk_tokens=self._chunk_tokens)
        if record is None:
            # Everything this sequence holds is cached, so it is ready to emit. No
            # graph invocation happened, and reporting an empty iteration would make
            # the step count stop meaning "runs of the model", so the iteration goes
            # to the decode step this sequence has just become eligible for.
            self._pending.popleft()
            self._decoding.append(request_id)
            return self._run_decode_batch(preempted)

        self._records[request_id].steps.append(record)
        self._stats.prefill_steps += 1
        self._chunks_since_decode += 1
        return SchedulerStep(
            kind="prefill",
            request_ids=(request_id,),
            records=(record,),
            preempted=preempted,
        )

    def _select_batch(self) -> list[str]:
        """Which of the decoding sequences share the next step.

        Queue order unless `length_bucketing` is on, and queue order either way when
        everyone fits: with nothing left over there is no choice to make, which is why
        an arena sized to the batch width sees no effect from this setting at all.

        Bucketed, the batch is **anchored** on the head of the queue -- the sequence
        that has waited longest -- and the remaining slots go to the nearest cached
        lengths. Anchoring is what makes starvation impossible rather than unlikely:

            The anchor is always index 0 and is always removed from the queue. So a
            sequence at index k is either picked as filler this step, or watches at
            least one element ahead of it leave, which drops it to index k-1 or lower.
            An unserved sequence's index therefore strictly decreases, and it becomes
            the anchor within k steps. With N decoding, **every sequence is served at
            least once in any window of N decode steps.**

        That bound is a property of the rule, not of a tuning constant, which is why
        there is no age guard here. Sequences admitted or readmitted later join at the
        back, so they never push an incumbent's index up.

        Ties on distance break on queue position, so the sequence that has waited
        longer takes the slot.
        """
        if len(self._decoding) <= self._max_batch_size:
            return list(self._decoding)
        if not self._length_bucketing:
            return self._decoding[: self._max_batch_size]

        anchor = self._decoding[0]
        lengths = self._cached_lengths()
        anchor_length = lengths[anchor]
        ranked = sorted(
            enumerate(self._decoding[1:]),
            key=lambda pair: (abs(lengths[pair[1]] - anchor_length), pair[0]),
        )
        return [anchor] + [request_id for _, request_id in ranked[: self._max_batch_size - 1]]

    def _cached_lengths(self) -> dict[str, int]:
        """Cached tokens per decoding sequence, which is what sets the padded width.

        Read from the client's own view rather than tracked alongside it, so the two
        cannot disagree about how long a sequence is after a preemption and recompute.
        """
        decoding = set(self._decoding)
        return {
            state.sequence_id: state.cached_tokens
            for state in self._client.states()
            if state.sequence_id in decoding
        }

    def _run_decode_batch(self, preempted: tuple[str, ...]) -> SchedulerStep:
        batch = self._select_batch()
        records = self._client.emit_batch(batch)
        self._chunks_since_decode = 0
        self._stats.decode_steps += 1
        self._stats.batched_rows += len(batch)
        self._stats.tokens_emitted += len(batch)

        completed = []
        for request_id, record in zip(batch, records, strict=True):
            generation = self._records[request_id]
            generation.steps.append(record)
            if self._finished(request_id, record):
                completed.append(request_id)

        for request_id in completed:
            self._retire(request_id)

        # Round-robin, so a batch wider than max_batch_size does not starve its tail.
        # Written against a subset rather than a prefix because a bucketed batch is one;
        # with bucketing off the two are the same list, since ids are unique and the
        # batch is then exactly the front of the queue.
        served = set(batch)
        survivors = [r for r in batch if r not in completed]
        self._decoding = [r for r in self._decoding if r not in served] + survivors

        return SchedulerStep(
            kind="decode",
            request_ids=tuple(batch),
            records=tuple(records),
            completed=tuple(completed),
            preempted=preempted,
        )

    def _finished(self, request_id: str, record: StepRecord) -> bool:
        generation = self._records[request_id]
        emitted = self._client.emitted(request_id)
        if record.token is not None and record.token in self._request(request_id).stop_tokens:
            generation.stopped_early = True
            return True
        if len(emitted) >= self._request(request_id).max_new_tokens:
            return True
        if self._client.at_context_limit(request_id):
            generation.hit_context_limit = True
            return True
        return False

    def _request(self, request_id: str) -> GenerationRequest:
        return self._client.request(request_id)

    def _retire(self, request_id: str) -> None:
        generation = self._records[request_id]
        generation.tokens = self._client.emitted(request_id)
        # Derived from the steps rather than tracked alongside them, so the two cannot
        # disagree about what a recompute cost.
        generation.recompute_ms = sum(
            step.total_ms for step in generation.steps if step.phase == "recompute"
        )
        started = self._started_at.get(request_id)
        if started is not None:
            generation.wall_ms = (time.perf_counter() - started) * 1000.0
        self._client.release(request_id)

    # --- convenience ----------------------------------------------------------

    def run(self, requests: Sequence[GenerationRequest]) -> dict[str, GenerationRecord]:
        """Submit everything and drain. For a fixed workload rather than a live one."""
        for request in requests:
            self.submit(request)
        return self.drain()
