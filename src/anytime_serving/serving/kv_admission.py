"""Block-granular admission and eviction by deadline slack.

The KV arena is fixed by design; that is what makes it something to decide about.
This module holds the decision, and `anytime_runtime.DecoderSession` holds the
mechanism. Deadlines live here rather than in C++ because this is where the rest of
the control plane already reasons about them.

Two decisions, one currency
---------------------------

**Admission** is block-granular: a request needs `ceil(tokens / block_tokens)`
blocks, and either they exist or they do not. Unlike the encoder path, there is no
useful "admit and see" -- a decoding sequence occupies its blocks for hundreds of
steps, so admitting one the arena cannot hold means evicting somebody later at a
worse moment.

**Eviction** picks the sequence that can most afford to be interrupted. Preemption
here is preempt-and-recompute: the victim's blocks are released and its tokens are
kept, so resuming it means re-running its whole history. That cost is real. On GPT-2
a 960-token sequence recomputes in about 261 ms against an 8.1 ms decode step, so
evicting the wrong sequence spends 32 steps' worth of somebody else's budget.

So the currency is slack: how much of a sequence's deadline is left after allowing
for the work it still has to do. A victim is only eligible if it survives its own
recompute, because evicting a sequence into a certain deadline miss turns one
problem into two. Sequences that will miss regardless are evicted first: their
blocks are pure gain, since what would be lost is already lost.

Costs are measured, not assumed
-------------------------------

`CacheCost` carries no defaults. Stage 1's headline result was invalid because a
service time was carried over from somewhere it did not apply, and a decode cost is
just as host-specific: on this machine GPT-2 FP32 steps in 5.12 ms at 128 cached
tokens and 8.15 ms at 960. A single scalar would be wrong at one end or the other,
so the decode cost is linear in cached tokens, which fits those measurements to
within 0.09 ms. `scripts/profile_decode.py` produces the coefficients.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

__all__ = [
    "AdmissionPlan",
    "BlockAdmission",
    "CacheCost",
    "SequenceState",
]


@dataclass(frozen=True)
class CacheCost:
    """What a decode step and a recompute cost, on this host, for this model.

    ``decode_ms`` is linear in the number of cached tokens rather than constant,
    because a decode step re-reads the whole cache: measured 5.12 ms at 128 tokens,
    6.66 at 512 and 8.15 at 960, which a line through them reproduces within
    0.09 ms.

    ``prefill_ms`` is linear too, which is the coarser of the two. Prefill is mildly
    superlinear -- 0.280 ms per token at 128, 0.364 at 1024 -- so the single fitted
    rate of 0.35 overstates a 128-token recompute by 25% and understates a
    1024-token one by 4%. It is used to compare eviction candidates against each
    other rather than to promise a completion time, and it errs on the cheap side for
    long sequences, which is the direction that makes the policy more willing to
    evict them. Worth revisiting if that shows up as churn.
    """

    decode_base_ms: float
    decode_per_token_ms: float
    prefill_per_token_ms: float

    def __post_init__(self) -> None:
        for name in ("decode_base_ms", "decode_per_token_ms", "prefill_per_token_ms"):
            value = getattr(self, name)
            if value < 0.0:
                raise ValueError(f"{name} must not be negative, got {value}")

    def decode_ms(self, cached_tokens: int) -> float:
        """Cost of one decode step against a cache of this size."""
        if cached_tokens < 0:
            raise ValueError("cached_tokens must not be negative")
        return self.decode_base_ms + self.decode_per_token_ms * cached_tokens

    def decode_span_ms(self, cached_tokens: int, steps: int) -> float:
        """Cost of `steps` decode steps, the cache growing by one token each time.

        Closed form rather than a loop: the per-step cost is linear in the cache
        size, so the total is an arithmetic series. Summing it step by step would
        give the same answer and make the caller's cost depend on how many tokens
        are left.
        """
        if steps <= 0:
            return 0.0
        if cached_tokens < 0:
            raise ValueError("cached_tokens must not be negative")
        positions = steps * cached_tokens + steps * (steps - 1) / 2.0
        return steps * self.decode_base_ms + self.decode_per_token_ms * positions

    def prefill_ms(self, tokens: int) -> float:
        """Cost of running `tokens` through the graph from a cold cache."""
        if tokens < 0:
            raise ValueError("tokens must not be negative")
        return self.prefill_per_token_ms * tokens


@dataclass
class SequenceState:
    """What the policy needs to know about one sequence in flight.

    `cached_tokens` is what a recompute would have to re-run, so it is the prompt
    plus everything emitted so far. `remaining_tokens` is how many steps are still
    owed.
    """

    sequence_id: str
    # On the same clock as the `now` passed to `plan`; `time.perf_counter` in the
    # serving path.
    deadline_at: float
    cached_tokens: int
    remaining_tokens: int
    blocks_held: int

    def __post_init__(self) -> None:
        if not self.sequence_id:
            raise ValueError("sequence_id must not be empty")
        for name in ("cached_tokens", "remaining_tokens", "blocks_held"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must not be negative")


@dataclass(frozen=True)
class AdmissionPlan:
    """The answer, and what it would cost to carry out.

    `evict` is empty when the request fits as things stand. When it is not, those
    sequences have to be preempted before the request is opened, in that order.
    """

    admit: bool
    blocks_needed: int
    blocks_free: int
    evict: tuple[str, ...] = ()
    reason: str = ""
    # Recompute the plan spends on the sequences it evicts. Zero when nothing is
    # evicted; worth surfacing because a plan that frees room by throwing away a
    # second of work is a decision, not a detail.
    recompute_cost_ms: float = 0.0

    @property
    def preempts(self) -> bool:
        return bool(self.evict)


@dataclass
class BlockAdmission:
    """Decides who is admitted and, when the arena is full, who makes room.

    Holds no state about sequences: `plan` is given the arena's occupancy and the
    sequences in flight, so the caller stays the single owner of that. The reason a
    plan came out the way it did is returned with it, because "rejected" on its own
    is not a diagnosis.
    """

    capacity_blocks: int
    block_tokens: int
    cost: CacheCost
    # Inflates every block requirement. Above 1.0 it reserves headroom, so a
    # sequence that runs longer than projected does not immediately have to evict
    # someone. Matches the `admission.safety_factor` already in configs/serving.yaml.
    safety_factor: float = 1.0
    _log: list[str] = field(default_factory=list, repr=False)

    def __post_init__(self) -> None:
        if self.capacity_blocks <= 0:
            raise ValueError("capacity_blocks must be positive")
        if self.block_tokens <= 0:
            raise ValueError("block_tokens must be positive")
        if self.safety_factor < 1.0:
            raise ValueError(
                f"safety_factor below 1.0 would reserve less than a sequence needs, "
                f"got {self.safety_factor}"
            )

    @property
    def capacity_tokens(self) -> int:
        """Token positions the whole arena can hold at once."""
        return self.capacity_blocks * self.block_tokens

    def blocks_needed(self, tokens: int) -> int:
        """Blocks a sequence of this many tokens occupies, with headroom applied."""
        if tokens < 0:
            raise ValueError("tokens must not be negative")
        wanted = int(tokens * self.safety_factor + 0.5)
        return -(-wanted // self.block_tokens)

    def slack_ms(self, state: SequenceState, now: float) -> float:
        """Milliseconds of deadline left once the work still owed is allowed for.

        Negative means the sequence is going to miss even if nothing interrupts it.
        """
        remaining = self.cost.decode_span_ms(state.cached_tokens, state.remaining_tokens)
        return (state.deadline_at - now) * 1000.0 - remaining

    def recompute_ms(self, state: SequenceState) -> float:
        """What resuming this sequence would cost after its blocks were taken."""
        return self.cost.prefill_ms(state.cached_tokens)

    def plan(
        self,
        *,
        tokens: int,
        blocks_free: int,
        sequences: list[SequenceState] | None = None,
        now: float,
        exclude: str | None = None,
        already_held: int = 0,
    ) -> AdmissionPlan:
        """Whether a sequence needing `tokens` positions can be admitted.

        `exclude` names a sequence that must not be considered as a victim, which is
        how a sequence growing past its own reservation asks for room without being
        offered its own blocks.

        `already_held` is what that sequence has already got. `tokens` is a total
        rather than an increment, so a sequence holding six blocks and asking to hold
        seven has a shortfall of one; without this it would appear to need all seven
        from scratch and the plan would evict far more than necessary, or refuse.
        """
        needed = self.blocks_needed(tokens)
        candidates = list(sequences or [])
        available = blocks_free + already_held

        if needed > self.capacity_blocks:
            return AdmissionPlan(
                admit=False,
                blocks_needed=needed,
                blocks_free=blocks_free,
                reason=(
                    f"needs {needed} block(s) but the arena holds {self.capacity_blocks}; "
                    f"no eviction can make room for a sequence larger than the pool"
                ),
            )
        if needed <= available:
            return AdmissionPlan(
                admit=True,
                blocks_needed=needed,
                blocks_free=blocks_free,
                reason=f"{available} block(s) available, {needed} needed",
            )

        shortfall = needed - available
        victims: list[str] = []
        freed = 0
        spent = 0.0
        for state in self._victim_order(candidates, now, exclude):
            victims.append(state.sequence_id)
            freed += state.blocks_held
            spent += self.recompute_ms(state)
            if freed >= shortfall:
                break

        if freed < shortfall:
            return AdmissionPlan(
                admit=False,
                blocks_needed=needed,
                blocks_free=blocks_free,
                reason=(
                    f"needs {shortfall} more block(s); evicting every eligible sequence "
                    f"would free {freed}. A sequence that would miss its deadline once "
                    f"recomputed is not eligible, since evicting it turns one miss into two"
                ),
            )

        return AdmissionPlan(
            admit=True,
            blocks_needed=needed,
            blocks_free=blocks_free,
            evict=tuple(victims),
            reason=(
                f"needs {shortfall} more block(s); preempting {len(victims)} sequence(s) "
                f"frees {freed}"
            ),
            recompute_cost_ms=spent,
        )

    def _victim_order(
        self, sequences: list[SequenceState], now: float, exclude: str | None
    ) -> list[SequenceState]:
        """Eviction candidates, best first.

        Doomed sequences go first: they will miss their deadline whether or not they
        are touched, so their blocks are free to take. After them come the sequences
        that survive their own recompute, most surviving slack first, and more blocks
        first where that ties -- freeing the same room from fewer victims means fewer
        recomputes.

        Everything else is withheld. A sequence with slack now but none after a
        recompute would be converted from a deadline it can meet into one it cannot,
        which is a worse outcome than refusing the newcomer.

        Not an optimum: freeing the shortfall from the fewest possible victims is a
        packing problem, and solving it would sometimes pick sequences with tighter
        deadlines. This orders by harm instead.
        """
        doomed: list[tuple[float, int, SequenceState]] = []
        survivors: list[tuple[float, int, SequenceState]] = []

        for state in sequences:
            if exclude is not None and state.sequence_id == exclude:
                continue
            if state.blocks_held == 0:
                continue
            slack = self.slack_ms(state, now)
            if slack <= 0.0:
                # Most-missed first, so the least salvageable goes first.
                doomed.append((slack, -state.blocks_held, state))
                continue
            surviving = slack - self.recompute_ms(state)
            if surviving <= 0.0:
                continue
            survivors.append((-surviving, -state.blocks_held, state))

        doomed.sort(key=lambda entry: (entry[0], entry[1], entry[2].sequence_id))
        survivors.sort(key=lambda entry: (entry[0], entry[1], entry[2].sequence_id))
        return [entry[2] for entry in doomed] + [entry[2] for entry in survivors]

    def with_capacity(self, capacity_blocks: int) -> BlockAdmission:
        """The same policy against a differently sized arena."""
        return replace(self, capacity_blocks=capacity_blocks)
