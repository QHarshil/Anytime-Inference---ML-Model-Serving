"""Tests for block-granular admission and eviction by deadline slack.

Pure policy, no runtime: the arena is fixed and the questions are which sequences fit
and, when they do not, who can most afford to be interrupted. Both answers are cheap
to get wrong in a way that looks reasonable, so the cases here are mostly about the
orderings the policy claims to make rather than about the arithmetic.

The one it must never get wrong is `test_a_sequence_that_could_not_survive_recompute
_is_not_evicted`. Preemption is preempt-and-recompute, so evicting a sequence with
slack now but none after re-running its history converts a deadline it would have met
into one it cannot -- two misses where there was one.
"""

import pytest

from anytime_serving.serving.kv_admission import (
    BlockAdmission,
    CacheCost,
    SequenceState,
)

# Close to what this host measures for GPT-2 FP32, rounded: a decode step costs
# 4.0 ms plus 0.005 ms per cached token, and a recompute 0.35 ms per token.
COST = CacheCost(decode_base_ms=4.0, decode_per_token_ms=0.005, prefill_per_token_ms=0.35)
BLOCK_TOKENS = 64


def _policy(*, capacity_blocks=16, safety_factor=1.0):
    return BlockAdmission(
        capacity_blocks=capacity_blocks,
        block_tokens=BLOCK_TOKENS,
        cost=COST,
        safety_factor=safety_factor,
    )


def _state(name, *, deadline_at, cached=256, remaining=32, blocks=4):
    return SequenceState(
        sequence_id=name,
        deadline_at=deadline_at,
        cached_tokens=cached,
        remaining_tokens=remaining,
        blocks_held=blocks,
    )


# --- the cost model ---------------------------------------------------------


def test_decode_cost_grows_with_the_cache():
    """A decode step re-reads the whole cache, so a single scalar would be wrong.

    Measured on this host: 4.48 ms at 128 cached tokens, 6.37 at 512, 8.84 at 1023.
    A line through those reproduces them within 0.03 ms, which is what the model is
    for.
    """
    measured = CacheCost(
        decode_base_ms=3.86, decode_per_token_ms=0.00487, prefill_per_token_ms=0.35
    )
    assert measured.decode_ms(128) == pytest.approx(4.48, abs=0.03)
    assert measured.decode_ms(512) == pytest.approx(6.37, abs=0.03)
    assert measured.decode_ms(1023) == pytest.approx(8.84, abs=0.03)


def test_decode_span_matches_stepping_one_at_a_time():
    """The closed form has to equal the loop it replaces.

    Summed step by step the cost depends on how many tokens are left, which is fine
    for tens and wrong to do inside an eviction decision taken per request.
    """
    stepwise = sum(COST.decode_ms(200 + step) for step in range(37))
    assert COST.decode_span_ms(200, 37) == pytest.approx(stepwise, rel=1e-12)
    assert COST.decode_span_ms(200, 0) == 0.0
    assert COST.decode_span_ms(200, -5) == 0.0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"decode_base_ms": -1.0},
        {"decode_per_token_ms": -0.1},
        {"prefill_per_token_ms": -0.5},
    ],
)
def test_a_negative_cost_is_rejected(kwargs):
    fields = {"decode_base_ms": 1.0, "decode_per_token_ms": 1.0, "prefill_per_token_ms": 1.0}
    fields.update(kwargs)
    with pytest.raises(ValueError):
        CacheCost(**fields)


# --- block arithmetic -------------------------------------------------------


def test_blocks_needed_rounds_up():
    policy = _policy()
    assert policy.blocks_needed(0) == 0
    assert policy.blocks_needed(1) == 1
    assert policy.blocks_needed(64) == 1
    assert policy.blocks_needed(65) == 2
    assert policy.capacity_tokens == 16 * BLOCK_TOKENS


def test_the_safety_factor_reserves_headroom():
    """Headroom so a sequence running longer than projected need not evict anyone."""
    tight = _policy()
    roomy = _policy(safety_factor=1.5)
    assert tight.blocks_needed(128) == 2
    assert roomy.blocks_needed(128) == 3


def test_a_safety_factor_below_one_is_rejected():
    """It would reserve less than the sequence needs, which is not a trade-off."""
    with pytest.raises(ValueError, match="less than a sequence needs"):
        _policy(safety_factor=0.9)


@pytest.mark.parametrize("capacity,block", [(0, 64), (-1, 64), (16, 0), (16, -8)])
def test_a_degenerate_arena_is_rejected(capacity, block):
    with pytest.raises(ValueError):
        BlockAdmission(capacity_blocks=capacity, block_tokens=block, cost=COST)


# --- admission --------------------------------------------------------------


def test_a_request_that_fits_is_admitted_without_evicting_anyone():
    policy = _policy()
    plan = policy.plan(tokens=128, blocks_free=8, sequences=[], now=0.0)
    assert plan.admit
    assert plan.blocks_needed == 2
    assert plan.evict == ()
    assert not plan.preempts
    assert plan.recompute_cost_ms == 0.0


def test_a_request_larger_than_the_whole_arena_is_refused_outright():
    """No eviction can help, so the plan should not propose any.

    Worth separating from ordinary pressure: this request will never fit, however
    quiet the arena gets, and retrying it later is wasted work.
    """
    policy = _policy(capacity_blocks=4)
    plan = policy.plan(
        tokens=4096, blocks_free=4, sequences=[_state("a", deadline_at=100.0)], now=0.0
    )
    assert not plan.admit
    assert plan.evict == ()
    assert "larger than the pool" in plan.reason


def test_slack_is_what_is_left_after_the_work_still_owed():
    policy = _policy()
    # One second of budget, 32 steps still to run against a 256-token cache.
    state = _state("a", deadline_at=1.0, cached=256, remaining=32)
    expected = 1000.0 - COST.decode_span_ms(256, 32)
    assert policy.slack_ms(state, 0.0) == pytest.approx(expected)
    assert policy.recompute_ms(state) == pytest.approx(256 * 0.35)


# --- who gets evicted -------------------------------------------------------


def test_the_sequence_with_the_most_slack_is_evicted():
    """Least harm first. Both hold the same blocks, so only the deadline differs."""
    policy = _policy(capacity_blocks=8)
    plan = policy.plan(
        tokens=256,
        blocks_free=0,
        sequences=[
            _state("tight", deadline_at=0.4, blocks=4),
            _state("roomy", deadline_at=5.0, blocks=4),
        ],
        now=0.0,
    )
    assert plan.admit
    assert plan.evict == ("roomy",)
    assert plan.recompute_cost_ms == pytest.approx(256 * 0.35)


def test_a_sequence_that_could_not_survive_recompute_is_not_evicted():
    """Evicting it would turn one deadline miss into two.

    `fragile` has slack, so it is on course to finish in time, but not enough slack to
    absorb re-running its history. Taking its blocks would guarantee it misses, and the
    newcomer is refused instead.
    """
    policy = _policy(capacity_blocks=8)
    # 150 ms of budget, 52.6 ms of decoding still owed, and a 179.2 ms recompute: it
    # finishes comfortably if left alone and misses if interrupted.
    fragile = _state("fragile", deadline_at=0.15, cached=512, remaining=8, blocks=8)
    assert policy.slack_ms(fragile, 0.0) > 0.0
    assert policy.slack_ms(fragile, 0.0) - policy.recompute_ms(fragile) < 0.0

    plan = policy.plan(tokens=256, blocks_free=0, sequences=[fragile], now=0.0)
    assert not plan.admit
    assert plan.evict == ()
    assert "turns one miss into two" in plan.reason


def test_a_sequence_already_past_saving_is_evicted_first():
    """Its blocks are pure gain: what would be lost is already lost.

    `doomed` cannot meet its deadline whether or not it is touched, so it goes ahead
    of a healthy sequence with plenty of slack.
    """
    policy = _policy(capacity_blocks=8)
    doomed = _state("doomed", deadline_at=0.01, cached=256, remaining=64, blocks=4)
    healthy = _state("healthy", deadline_at=60.0, cached=256, remaining=64, blocks=4)
    assert policy.slack_ms(doomed, 0.0) < 0.0

    plan = policy.plan(tokens=256, blocks_free=0, sequences=[healthy, doomed], now=0.0)
    assert plan.admit
    assert plan.evict == ("doomed",)


def test_the_same_room_is_freed_from_fewer_victims():
    """At equal slack, prefer the sequence holding more blocks.

    Freeing four blocks from one sequence costs one recompute; from four sequences it
    costs four. Not an optimum -- packing the shortfall into the fewest victims would
    sometimes pick tighter deadlines -- but at equal harm it is free to prefer.
    """
    policy = _policy(capacity_blocks=8)
    plan = policy.plan(
        tokens=256,
        blocks_free=0,
        sequences=[
            _state("small_a", deadline_at=9.0, blocks=1),
            _state("small_b", deadline_at=9.0, blocks=1),
            _state("large", deadline_at=9.0, blocks=4),
        ],
        now=0.0,
    )
    assert plan.admit
    assert plan.evict == ("large",)


def test_several_sequences_are_evicted_when_one_is_not_enough():
    policy = _policy(capacity_blocks=8)
    plan = policy.plan(
        tokens=256,
        blocks_free=1,
        sequences=[
            _state("a", deadline_at=9.0, blocks=1),
            _state("b", deadline_at=9.0, blocks=1),
            _state("c", deadline_at=9.0, blocks=1),
        ],
        now=0.0,
    )
    assert plan.admit
    assert len(plan.evict) == 3
    assert plan.recompute_cost_ms == pytest.approx(3 * 256 * 0.35)


def test_eviction_stops_once_enough_has_been_freed():
    """Nobody is evicted whose blocks are not needed."""
    policy = _policy(capacity_blocks=16)
    plan = policy.plan(
        tokens=64,
        blocks_free=0,
        sequences=[
            _state("a", deadline_at=9.0, blocks=4),
            _state("b", deadline_at=9.0, blocks=4),
        ],
        now=0.0,
    )
    assert plan.admit
    assert len(plan.evict) == 1


def test_a_request_is_refused_when_eviction_cannot_free_enough():
    policy = _policy(capacity_blocks=16)
    plan = policy.plan(
        tokens=1024,
        blocks_free=0,
        sequences=[_state("a", deadline_at=9.0, blocks=2)],
        now=0.0,
    )
    assert not plan.admit
    assert "would free 2" in plan.reason


def test_a_sequence_is_never_offered_its_own_blocks():
    """A sequence outgrowing its reservation asks for room, not for its own cache."""
    policy = _policy(capacity_blocks=8)
    itself = _state("grower", deadline_at=9.0, blocks=8)
    plan = policy.plan(tokens=576, blocks_free=0, sequences=[itself], now=0.0, exclude="grower")
    assert not plan.admit
    assert plan.evict == ()


def test_what_a_growing_sequence_already_holds_counts_towards_its_requirement():
    """`tokens` is a total, so a sequence asking for one more block needs one more.

    Without this the shortfall is the sequence's entire requirement rather than its
    increment: a sequence holding six blocks and wanting a seventh would look like it
    needed all seven from scratch, and the plan would either evict far more than
    necessary or refuse outright.
    """
    policy = _policy(capacity_blocks=8)
    bystander = _state("bystander", deadline_at=60.0, cached=256, remaining=4, blocks=2)

    # 385 tokens is seven blocks; the grower holds six and the arena is full.
    increment = policy.plan(
        tokens=385,
        blocks_free=0,
        sequences=[bystander],
        now=0.0,
        exclude="grower",
        already_held=6,
    )
    assert increment.admit
    assert increment.evict == ("bystander",)

    # Read as a request from scratch, the same numbers cannot be satisfied.
    from_scratch = policy.plan(
        tokens=385, blocks_free=0, sequences=[bystander], now=0.0, exclude="grower"
    )
    assert not from_scratch.admit


def test_a_sequence_that_already_holds_enough_needs_no_eviction():
    policy = _policy(capacity_blocks=8)
    plan = policy.plan(
        tokens=200,
        blocks_free=0,
        sequences=[_state("other", deadline_at=60.0)],
        now=0.0,
        exclude="grower",
        already_held=4,
    )
    assert plan.admit
    assert plan.evict == ()


def test_sequences_holding_nothing_are_not_candidates():
    """A preempted sequence has no blocks left to take."""
    policy = _policy(capacity_blocks=8)
    plan = policy.plan(
        tokens=256,
        blocks_free=0,
        sequences=[
            _state("preempted", deadline_at=9.0, blocks=0),
            _state("resident", deadline_at=9.0, blocks=4),
        ],
        now=0.0,
    )
    assert plan.admit
    assert plan.evict == ("resident",)


def test_eviction_order_is_deterministic_when_candidates_tie():
    """Two identical candidates must not be chosen by dict ordering."""
    policy = _policy(capacity_blocks=8)
    forward = policy.plan(
        tokens=256,
        blocks_free=0,
        sequences=[_state("b", deadline_at=9.0), _state("a", deadline_at=9.0)],
        now=0.0,
    )
    reverse = policy.plan(
        tokens=256,
        blocks_free=0,
        sequences=[_state("a", deadline_at=9.0), _state("b", deadline_at=9.0)],
        now=0.0,
    )
    assert forward.evict == reverse.evict == ("a",)


def test_with_capacity_leaves_the_original_alone():
    policy = _policy(capacity_blocks=16)
    resized = policy.with_capacity(64)
    assert resized.capacity_blocks == 64
    assert policy.capacity_blocks == 16
    assert resized.block_tokens == policy.block_tokens


# --- state validation -------------------------------------------------------


def test_a_sequence_state_needs_an_id():
    with pytest.raises(ValueError, match="sequence_id"):
        SequenceState(
            sequence_id="", deadline_at=1.0, cached_tokens=1, remaining_tokens=1, blocks_held=1
        )


@pytest.mark.parametrize("field", ["cached_tokens", "remaining_tokens", "blocks_held"])
def test_negative_sequence_state_is_rejected(field):
    fields = {
        "sequence_id": "a",
        "deadline_at": 1.0,
        "cached_tokens": 1,
        "remaining_tokens": 1,
        "blocks_held": 1,
    }
    fields[field] = -1
    with pytest.raises(ValueError, match=field):
        SequenceState(**fields)
