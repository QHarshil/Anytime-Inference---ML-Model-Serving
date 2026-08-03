"""Client for the decoder path: prefill, decode, preempt, resume.

Wraps `anytime_runtime.DecoderSession` the way `RuntimeClient` wraps `Engine`, and
adds the two things a block-allocated cache needs that an encoder never did: token
history, so a preempted sequence can be recomputed, and a policy to consult when the
arena has no room.

There is no Python fallback here, unlike `onnx_runtime.py`. The arena is the
extension -- reimplementing it over numpy would allocate the whole cache afresh
every step (measured 2.88 ms with a 478% spread against 1.08 ms for the arena) and
would put the accounting somewhere other than where the runtime is. A missing
extension is therefore an error rather than a slower path.

TTFT and TPOT are reported separately because they are not the same measurement. On
this host GPT-2 FP32 prefills a 1024-token prompt in 372.2 ms and then emits each
token in 9.5 ms at full context, a factor of 39. A single latency figure would
describe neither.

Preemption
----------

`preempt` releases a sequence's blocks and keeps its tokens; `resume` re-runs the
history and carries on. The output is token-identical to an uninterrupted run --
asserted in `tests/test_decoder_session.py` on both the synthetic graph and GPT-2 --
which is what makes eviction a scheduling decision rather than a correctness bug. It
is not free: recomputing a 960-token sequence costs about 340 ms against a 9.5 ms
decode step, which is why `kv_admission.BlockAdmission` weighs slack against
recompute before naming a victim.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np

from ..utils.logger import get_logger
from .kv_admission import AdmissionPlan, BlockAdmission, SequenceState
from .onnx_runtime import load_extension

LOGGER = get_logger("serving.decoder")

__all__ = [
    "DecoderClient",
    "GenerationRecord",
    "GenerationRequest",
    "Occupancy",
    "StepRecord",
]


@dataclass
class GenerationRequest:
    """One generation, with the deadline it is meant to meet."""

    prompt: Sequence[int]
    max_new_tokens: int
    deadline_ms: float
    request_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    # Stopping early frees blocks early, which is the whole reason to bother.
    stop_tokens: frozenset[int] = frozenset()

    def __post_init__(self) -> None:
        if len(self.prompt) == 0:
            raise ValueError("a generation needs at least one prompt token")
        if self.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        if self.deadline_ms <= 0.0:
            raise ValueError("deadline_ms must be positive")

    @property
    def prompt_tokens(self) -> int:
        return len(self.prompt)


@dataclass(frozen=True)
class StepRecord:
    """One prefill, decode or recompute, as the runtime measured it.

    The phases are broken out rather than summed because the gather is the price of
    block accounting and reporting only a total would hide it. `verify_ms` is
    non-zero on the one step per sequence that checks the present-prefix invariant,
    and `pad_ms` only on a batched step, where rows shorter than the batch's longest
    are right-padded and the padding is cleared.

    `batch_size` is how many sequences shared the graph invocation this record
    describes. It is 1 for everything except a batched decode step, and it is here
    because it is what separates two different numbers that a single field could not
    carry. For a sequence in a batch of eight taking 50 ms, the durations above are
    its real latency -- it genuinely waited 50 ms for its token, and dividing by
    eight would understate what it experienced. Throughput is the other direction:
    eight tokens came out in that 50 ms. Latency is the duration, throughput is
    `batch_size` over the duration, and neither is derivable from the other without
    this field.
    """

    request_id: str
    phase: str
    token: int | None
    cached_tokens: int
    runs: int
    gather_ms: float
    run_ms: float
    scatter_ms: float
    verify_ms: float
    total_ms: float
    pad_ms: float = 0.0
    batch_size: int = 1


@dataclass(frozen=True)
class Occupancy:
    """Arena state, as admission sees it."""

    capacity_blocks: int
    free_blocks: int
    block_tokens: int
    bytes_per_block: int
    resident_sequences: int

    @property
    def used_blocks(self) -> int:
        return self.capacity_blocks - self.free_blocks

    @property
    def used_fraction(self) -> float:
        return self.used_blocks / self.capacity_blocks if self.capacity_blocks else 0.0


@dataclass
class GenerationRecord:
    """A whole generation: what came out, and what it cost."""

    request_id: str
    admitted: bool
    prompt_tokens: int
    deadline_ms: float
    tokens: list[int] = field(default_factory=list)
    steps: list[StepRecord] = field(default_factory=list)
    rejection_reason: str = ""
    preemptions: int = 0
    recompute_ms: float = 0.0
    wall_ms: float = 0.0
    stopped_early: bool = False
    hit_context_limit: bool = False

    @property
    def ttft_ms(self) -> float:
        """Time to first token: the prefill, including its gather and scatter.

        The prefill is what stands between arrival and the first token, so a chunked
        prefill's chunks all count towards it.
        """
        prefill = [step for step in self.steps if step.phase == "prefill"]
        return sum(step.total_ms for step in prefill)

    @property
    def decode_steps(self) -> list[StepRecord]:
        return [step for step in self.steps if step.phase == "decode"]

    @property
    def tpot_ms(self) -> float:
        """Median time per output token over the decode phase.

        A median, not a mean: the first decode step of a sequence carries the
        present-prefix check and the staging-buffer allocation, and a mean would
        spread those across every token.
        """
        steps = self.decode_steps
        return median(step.total_ms for step in steps) if steps else 0.0

    @property
    def tpot_range_ms(self) -> tuple[float, float]:
        steps = self.decode_steps
        if not steps:
            return (0.0, 0.0)
        totals = [step.total_ms for step in steps]
        return (min(totals), max(totals))

    @property
    def gather_fraction(self) -> float:
        """Share of the decode phase spent gathering blocks.

        The cost of making the arena accountable, which is the thing the block
        allocator trades for and the number not to assume.
        """
        steps = self.decode_steps
        total = sum(step.total_ms for step in steps)
        return sum(step.gather_ms for step in steps) / total if total else 0.0

    @property
    def met_deadline(self) -> bool:
        return self.admitted and self.wall_ms <= self.deadline_ms


@dataclass
class _Sequence:
    """Bookkeeping the runtime deliberately does not do.

    The cache holds bytes; this holds the tokens those bytes were derived from. That
    separation is what makes preempt-and-recompute possible at all: the arena can
    give a sequence's blocks away because the sequence itself is still here.
    """

    request: GenerationRequest
    deadline_at: float
    tokens: list[int]
    emitted: list[int] = field(default_factory=list)
    logits: np.ndarray | None = None
    resident: bool = False
    finished: bool = False
    preemptions: int = 0
    recompute_ms: float = 0.0
    # Whether the tokens still to be cached are being re-run after a preemption
    # rather than run for the first time. Only affects how a step is labelled: the
    # work is identical, which is why one code path does both.
    recomputing: bool = False

    @property
    def remaining(self) -> int:
        return max(0, self.request.max_new_tokens - len(self.emitted))


class DecoderClient:
    """Decoding over a block-allocated KV cache, with admission and eviction.

    `admission` is optional. Without it the client runs sequences until the arena
    refuses one, which is what a profiling run needs before it has measured the costs
    a policy would be built from. With it, a request that does not fit triggers an
    eviction plan and the victims are preempted rather than the request being
    refused outright.
    """

    def __init__(
        self,
        graph: Path | str,
        *,
        block_tokens: int | None = None,
        num_blocks: int = 256,
        intra_op_threads: int = 1,
        inter_op_threads: int = 1,
        admission: BlockAdmission | None = None,
        reserve_full_generation: bool = True,
        max_context_tokens: int | None = None,
    ) -> None:
        extension = load_extension()
        width = extension.DEFAULT_BLOCK_TOKENS if block_tokens is None else block_tokens
        self._session: Any = extension.DecoderSession(
            str(graph),
            block_tokens=width,
            num_blocks=num_blocks,
            intra_op_threads=intra_op_threads,
            inter_op_threads=inter_op_threads,
        )
        self._exhausted = extension.CacheExhausted
        self._default_chunk = extension.DEFAULT_PREFILL_CHUNK_TOKENS
        self._admission = admission
        self._reserve_full_generation = reserve_full_generation
        self._max_context_tokens = max_context_tokens
        self._sequences: dict[str, _Sequence] = {}

        if admission is not None and admission.block_tokens != self.geometry.block_tokens:
            raise ValueError(
                f"the policy reasons in {admission.block_tokens}-token blocks but the "
                f"arena uses {self.geometry.block_tokens}. Every admission decision "
                f"would be off by a factor of "
                f"{admission.block_tokens / self.geometry.block_tokens:.3g}."
            )
        if admission is not None and admission.capacity_blocks != self.capacity_blocks:
            raise ValueError(
                f"the policy models {admission.capacity_blocks} block(s) but the arena "
                f"holds {self.capacity_blocks}; use BlockAdmission.with_capacity"
            )

    # --- state ---------------------------------------------------------------

    @property
    def geometry(self) -> Any:
        """KV geometry, as read off the graph."""
        return self._session.geometry

    @property
    def capacity_blocks(self) -> int:
        return int(self._session.capacity_blocks)

    @property
    def free_blocks(self) -> int:
        return int(self._session.free_blocks)

    @property
    def arena_bytes(self) -> int:
        return int(self._session.arena_bytes)

    @property
    def default_chunk_tokens(self) -> int:
        return int(self._default_chunk)

    @property
    def reserves_full_generation(self) -> bool:
        """Whether a sequence's whole projected generation is reserved up front.

        The batch scheduler requires this. With it, a resident sequence always has
        room for the tokens it still owes, so a batched decode step cannot run out of
        blocks part way through; without it, a step over eight sequences could
        exhaust the arena on behalf of any of them and the scheduler would have to
        unpick which.
        """
        return self._reserve_full_generation

    def occupancy(self) -> Occupancy:
        return Occupancy(
            capacity_blocks=self.capacity_blocks,
            free_blocks=self.free_blocks,
            block_tokens=int(self.geometry.block_tokens),
            bytes_per_block=int(self.geometry.bytes_per_block),
            resident_sequences=sum(1 for s in self._sequences.values() if s.resident),
        )

    def states(self, *, resident_only: bool = True) -> list[SequenceState]:
        """The sequences in flight, as the policy wants to see them."""
        states = []
        for request_id, sequence in self._sequences.items():
            if sequence.finished:
                continue
            if resident_only and not sequence.resident:
                continue
            states.append(
                SequenceState(
                    sequence_id=request_id,
                    deadline_at=sequence.deadline_at,
                    cached_tokens=len(sequence.tokens),
                    remaining_tokens=sequence.remaining,
                    blocks_held=(
                        int(self._session.blocks_held(request_id)) if sequence.resident else 0
                    ),
                )
            )
        return states

    def tokens(self, request_id: str) -> list[int]:
        """Prompt plus everything emitted, which is what a recompute re-runs."""
        return list(self._lookup(request_id).tokens)

    def request(self, request_id: str) -> GenerationRequest:
        """The request as submitted, including its stop tokens and token budget."""
        return self._lookup(request_id).request

    def at_context_limit(self, request_id: str) -> bool:
        """Whether another token would run past what the model was trained for.

        Public because a scheduler decides when a sequence stops, and this is not
        derivable from the graph: GPT-2's position table is an initializer rather than
        a declared shape, so exceeding it surfaces as an out-of-bounds Gather from
        inside ONNX Runtime.
        """
        return self._at_context_limit(self._lookup(request_id))

    def emitted(self, request_id: str) -> list[int]:
        return list(self._lookup(request_id).emitted)

    def next_token_logits(self, request_id: str) -> np.ndarray:
        """The distribution the next token would be drawn from.

        One row, not one per position: the graph returns logits for every position it
        was given, and the runtime copies out only the last. A caller wanting
        something other than the greedy `emit` samples from this.
        """
        sequence = self._lookup(request_id)
        if sequence.logits is None:
            raise RuntimeError(
                f"request {request_id} has no logits yet; prefill or resume it first"
            )
        return sequence.logits

    # --- admission -----------------------------------------------------------

    def admit(self, request: GenerationRequest, *, now: float | None = None) -> AdmissionPlan:
        """Reserve blocks for a request, evicting to make room if the policy allows.

        Returns the plan that was carried out. `plan.admit` false means the request
        was not opened and `plan.reason` says why.
        """
        if request.request_id in self._sequences:
            raise RuntimeError(f"request {request.request_id} is already in flight")

        moment = time.perf_counter() if now is None else now
        wanted = self._reserve_tokens(request.prompt_tokens, request.max_new_tokens)

        if self._admission is None:
            opened = bool(self._session.open(request.request_id, wanted))
            plan = AdmissionPlan(
                admit=opened,
                blocks_needed=int(self._session.blocks_for(wanted)),
                blocks_free=self.free_blocks,
                reason="" if opened else f"the arena cannot hold {wanted} more token(s)",
            )
        else:
            plan = self._admission.plan(
                tokens=wanted,
                blocks_free=self.free_blocks,
                sequences=self.states(),
                now=moment,
            )
            if plan.admit:
                for victim in plan.evict:
                    self.preempt(victim)
                if not self._session.open(request.request_id, wanted):
                    # The policy and the arena disagreed, which means one of them is
                    # wrong about the same numbers. Better to say so than to retry.
                    raise RuntimeError(
                        f"admission planned {plan.blocks_needed} block(s) for "
                        f"{request.request_id} after evicting {list(plan.evict)}, but the "
                        f"arena refused with {self.free_blocks} free of "
                        f"{self.capacity_blocks}"
                    )

        if plan.admit:
            self._sequences[request.request_id] = _Sequence(
                request=request,
                deadline_at=moment + request.deadline_ms / 1000.0,
                tokens=list(request.prompt),
                resident=True,
            )
        return plan

    def release(self, request_id: str) -> int:
        """Free a sequence's blocks and forget it. Idempotent."""
        blocks = int(self._session.release(request_id))
        self._sequences.pop(request_id, None)
        return blocks

    def preempt(self, request_id: str) -> int:
        """Take a sequence's blocks and keep its tokens.

        The sequence stays in flight and can be resumed; what it loses is the cache,
        which `resume` rebuilds by re-running the history.
        """
        sequence = self._lookup(request_id)
        blocks = int(self._session.release(request_id))
        sequence.resident = False
        sequence.logits = None
        sequence.preemptions += 1
        return blocks

    def resume(self, request_id: str, *, chunk_tokens: int | None = None) -> StepRecord:
        """Re-admit a preempted sequence and rebuild its cache from its tokens."""
        sequence = self._lookup(request_id)
        if sequence.resident:
            raise RuntimeError(f"request {request_id} was not preempted")

        wanted = self._reserve_tokens(len(sequence.tokens), sequence.remaining)
        if not self._session.open(request_id, wanted):
            raise self._exhausted(
                f"cannot readmit {request_id}: it needs "
                f"{self._session.blocks_for(wanted)} block(s) and "
                f"{self.free_blocks} of {self.capacity_blocks} are free"
            )
        sequence.resident = True
        record = self._run_prefill(sequence, "recompute", chunk_tokens)
        sequence.recompute_ms += record.total_ms
        return record

    # --- stepping ------------------------------------------------------------

    def prefill(self, request_id: str, *, chunk_tokens: int | None = None) -> StepRecord:
        """Run the prompt, filling the cache. Emits nothing."""
        sequence = self._lookup(request_id)
        if not sequence.resident:
            raise RuntimeError(f"request {request_id} holds no blocks; resume it first")
        return self._run_prefill(sequence, "prefill", chunk_tokens)

    def prefill_chunk(
        self, request_id: str, *, chunk_tokens: int | None = None
    ) -> StepRecord | None:
        """Advance the cached prefix by one chunk, or return None when it is complete.

        `prefill` runs every chunk before returning, so nothing else can happen in
        between. A scheduler needs the chunks one at a time, because the chunk
        boundary is where a resident sequence can get a turn: driving a whole prompt
        stalls every other sequence for its duration, 372 ms on GPT-2 at FP32 against
        93 ms for one 256-token chunk.

        A fresh prompt and a preempted sequence's history go through here identically.
        Both are "run the tokens this sequence holds that are not cached yet", and the
        only difference is what the step is called, so there is one path rather than
        two that could drift apart.
        """
        sequence = self._lookup(request_id)
        if not sequence.resident:
            raise RuntimeError(f"request {request_id} holds no blocks; readmit it first")

        cached = int(self._session.length(request_id))
        if cached >= len(sequence.tokens):
            sequence.recomputing = False
            return None

        width = self._default_chunk if chunk_tokens is None else chunk_tokens
        if width <= 0:
            width = len(sequence.tokens) - cached
        chunk = sequence.tokens[cached : cached + width]

        phase = "recompute" if sequence.recomputing else "prefill"
        result = self._session.extend(request_id, chunk)
        sequence.logits = np.asarray(result.logits)
        record = self._record(request_id, phase, None, result)
        if phase == "recompute":
            sequence.recompute_ms += record.total_ms
        if int(self._session.length(request_id)) >= len(sequence.tokens):
            sequence.recomputing = False
        return record

    def readmit(self, request_id: str) -> bool:
        """Take blocks back for a preempted sequence without running anything.

        `resume` reopens and recomputes in one call, which is what a caller driving
        one sequence wants. A scheduler wants the two apart, so that the recompute
        goes through `prefill_chunk` like any other prefill and can be interleaved.

        Returns False when the arena cannot supply the blocks, which is the same
        answer `admit` gives and for the same reason: it is a scheduling fact, not a
        failure.
        """
        sequence = self._lookup(request_id)
        if sequence.resident:
            raise RuntimeError(f"request {request_id} was not preempted")
        wanted = self._reserve_tokens(len(sequence.tokens), sequence.remaining)
        if not self._session.open(request_id, wanted):
            return False
        sequence.resident = True
        sequence.recomputing = True
        return True

    def emit_batch(self, request_ids: Sequence[str]) -> list[StepRecord]:
        """Emit one token for each sequence, in a single graph invocation.

        Every record carries the whole step's timings rather than a share of them,
        because that is what each sequence actually waited. See `StepRecord` on why
        `batch_size` travels with them.
        """
        ids = list(request_ids)
        if not ids:
            raise ValueError("emit_batch needs at least one request")

        sequences = []
        tokens = []
        for request_id in ids:
            sequence = self._lookup(request_id)
            if sequence.logits is None:
                raise RuntimeError(
                    f"request {request_id} has no logits to sample; prefill or resume it first"
                )
            if not sequence.resident:
                raise RuntimeError(f"request {request_id} holds no blocks; readmit it first")
            sequences.append(sequence)
            tokens.append(int(np.argmax(sequence.logits)))

        result = self._session.decode_batch(ids, tokens)

        records = []
        for request_id, sequence, token, row in zip(
            ids, sequences, tokens, result.rows, strict=True
        ):
            sequence.emitted.append(token)
            sequence.tokens.append(token)
            sequence.logits = np.asarray(row.logits)
            records.append(
                self._record(
                    request_id,
                    "decode",
                    token,
                    row,
                    timings=result.timings,
                    batch_size=len(ids),
                )
            )
        return records

    def emit(self, request_id: str) -> StepRecord:
        """Emit one token and extend the cache by it.

        The token is the argmax of the logits the previous step produced, so this is
        greedy decoding. Deterministic on purpose: a preempted sequence has to be
        comparable with an uninterrupted one, and sampling would make that a
        statement about a random seed.
        """
        sequence = self._lookup(request_id)
        if sequence.logits is None:
            raise RuntimeError(
                f"request {request_id} has no logits to sample; prefill or resume it first"
            )
        if not sequence.resident:
            raise RuntimeError(f"request {request_id} holds no blocks; resume it first")

        token = int(np.argmax(sequence.logits))
        sequence.emitted.append(token)
        sequence.tokens.append(token)

        try:
            result = self._session.decode(request_id, token)
        except self._exhausted:
            self._make_room_for(request_id)
            result = self._session.decode(request_id, token)

        sequence.logits = np.asarray(result.logits)
        return self._record(request_id, "decode", token, result)

    def generate(
        self, request: GenerationRequest, *, chunk_tokens: int | None = None
    ) -> GenerationRecord:
        """Admit, prefill and decode to completion. One sequence, start to finish."""
        started = time.perf_counter()
        plan = self.admit(request, now=started)
        record = GenerationRecord(
            request_id=request.request_id,
            admitted=plan.admit,
            prompt_tokens=request.prompt_tokens,
            deadline_ms=request.deadline_ms,
            rejection_reason="" if plan.admit else plan.reason,
        )
        if not plan.admit:
            record.wall_ms = (time.perf_counter() - started) * 1000.0
            return record

        sequence = self._sequences[request.request_id]
        record.steps.append(self.prefill(request.request_id, chunk_tokens=chunk_tokens))

        for _ in range(request.max_new_tokens):
            if self._at_context_limit(sequence):
                record.hit_context_limit = True
                break
            step = self.emit(request.request_id)
            record.steps.append(step)
            assert step.token is not None
            if step.token in request.stop_tokens:
                record.stopped_early = True
                break

        record.tokens = list(sequence.emitted)
        record.preemptions = sequence.preemptions
        record.recompute_ms = sequence.recompute_ms
        record.wall_ms = (time.perf_counter() - started) * 1000.0
        sequence.finished = True
        return record

    def close(self) -> None:
        for request_id in list(self._sequences):
            self._session.release(request_id)
        self._sequences.clear()
        self._session = None

    def __enter__(self) -> DecoderClient:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    # --- internals -----------------------------------------------------------

    def _lookup(self, request_id: str) -> _Sequence:
        sequence = self._sequences.get(request_id)
        if sequence is None:
            raise RuntimeError(f"unknown request {request_id}; admit it first")
        return sequence

    def _reserve_tokens(self, prompt_tokens: int, remaining: int) -> int:
        """How many token positions to reserve for a sequence.

        Reserving the whole projected generation admits fewer sequences but never
        strands one part way through; reserving only the prompt admits more and
        relies on eviction when they grow. The conservative choice is the default
        because a stranded sequence costs a full recompute to rescue.
        """
        wanted = prompt_tokens + (remaining if self._reserve_full_generation else 0)
        if self._max_context_tokens is not None:
            wanted = min(wanted, self._max_context_tokens)
        return max(wanted, prompt_tokens)

    def _at_context_limit(self, sequence: _Sequence) -> bool:
        """Whether another token would run past what the model was trained for.

        Not derivable from the graph: the position table is an initializer, not a
        declared shape, so exceeding it surfaces as an out-of-bounds Gather from
        inside ONNX Runtime. Stopping here turns that into a recorded outcome.
        """
        if self._max_context_tokens is None:
            return False
        return len(sequence.tokens) >= self._max_context_tokens

    def _make_room_for(self, request_id: str) -> None:
        """Evict on behalf of a sequence that outgrew its own reservation."""
        sequence = self._lookup(request_id)
        if self._admission is None:
            raise self._exhausted(
                f"request {request_id} outgrew its reservation and there is no admission "
                f"policy to evict with; {self.free_blocks} of {self.capacity_blocks} "
                f"blocks free"
            )
        plan = self._admission.plan(
            tokens=len(sequence.tokens),
            blocks_free=self.free_blocks,
            sequences=self.states(),
            now=time.perf_counter(),
            exclude=request_id,
            # Its requirement is a total, so what it already holds counts towards it.
            # Without this a sequence holding six blocks and wanting a seventh would
            # look like it needed all seven.
            already_held=int(self._session.blocks_held(request_id)),
        )
        if not plan.admit:
            raise self._exhausted(
                f"request {request_id} needs another block and none can be freed: {plan.reason}"
            )
        for victim in plan.evict:
            self.preempt(victim)
        LOGGER.info(
            "preempted %s to make room for %s (%s)",
            list(plan.evict),
            request_id,
            plan.reason,
        )

    def _run_prefill(self, sequence: _Sequence, phase: str, chunk_tokens: int | None) -> StepRecord:
        width = self._default_chunk if chunk_tokens is None else chunk_tokens
        result = self._session.prefill(
            sequence.request.request_id, list(sequence.tokens), chunk_tokens=width
        )
        sequence.logits = np.asarray(result.logits)
        return self._record(sequence.request.request_id, phase, None, result)

    @staticmethod
    def _record(
        request_id: str,
        phase: str,
        token: int | None,
        result: Any,
        *,
        timings: Any = None,
        batch_size: int = 1,
    ) -> StepRecord:
        # `timings` is passed separately for a batched step, where the rows carry none
        # of their own: one Run happened and it belongs to all of them.
        measured = result.timings if timings is None else timings
        return StepRecord(
            request_id=request_id,
            phase=phase,
            token=token,
            cached_tokens=int(result.length),
            runs=int(result.runs),
            gather_ms=float(measured.gather_ms),
            run_ms=float(measured.run_ms),
            scatter_ms=float(measured.scatter_ms),
            verify_ms=float(measured.verify_ms),
            total_ms=float(measured.total_ms),
            pad_ms=float(measured.pad_ms),
            batch_size=batch_size,
        )
