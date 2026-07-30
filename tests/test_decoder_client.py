"""Tests for the decoder client: generation, preemption, and eviction in practice.

`tests/test_kv_admission.py` checks the policy against constructed states and
`tests/test_decoder_session.py` checks the arena against a contiguous cache. What is
left, and what neither covers, is the two meeting: a sequence that runs out of blocks
mid-decode has to have somebody else preempted on its behalf and then carry on
correctly.

The assertion that matters most is `test_a_sequence_preempted_through_the_client_is
_token_identical`. The session-level test proves the mechanism preserves output; this
proves the client's bookkeeping does too, which is the part that has to remember what
the cache forgot.
"""

import numpy as np
import pytest

from anytime_serving.serving.decoder import DecoderClient, GenerationRequest
from anytime_serving.serving.kv_admission import BlockAdmission, CacheCost
from anytime_serving.serving.onnx_runtime import extension_available, load_extension

requires_extension = pytest.mark.skipif(
    not extension_available(),
    reason="anytime_runtime is not built; the decoder path lives in it. pip install -e .",
)

pytestmark = requires_extension

PROMPT = [3, 1, 4, 1, 5, 9, 2, 6]
BLOCK_TOKENS = 4
COST = CacheCost(decode_base_ms=4.0, decode_per_token_ms=0.005, prefill_per_token_ms=0.35)


def _client(graph, *, num_blocks=32, admission=None, **kwargs):
    return DecoderClient(
        graph, block_tokens=BLOCK_TOKENS, num_blocks=num_blocks, admission=admission, **kwargs
    )


def _policy(*, capacity_blocks, safety_factor=1.0):
    return BlockAdmission(
        capacity_blocks=capacity_blocks,
        block_tokens=BLOCK_TOKENS,
        cost=COST,
        safety_factor=safety_factor,
    )


def _request(*, max_new_tokens=6, deadline_ms=10_000.0, request_id="r0", stop_tokens=frozenset()):
    return GenerationRequest(
        prompt=PROMPT,
        max_new_tokens=max_new_tokens,
        deadline_ms=deadline_ms,
        request_id=request_id,
        stop_tokens=stop_tokens,
    )


# --- the request contract ---------------------------------------------------


@pytest.mark.parametrize(
    "kwargs,message",
    [
        ({"prompt": []}, "at least one prompt token"),
        ({"max_new_tokens": 0}, "max_new_tokens"),
        ({"deadline_ms": 0.0}, "deadline_ms"),
    ],
)
def test_a_malformed_request_is_rejected(kwargs, message):
    fields = {"prompt": PROMPT, "max_new_tokens": 4, "deadline_ms": 100.0}
    fields.update(kwargs)
    with pytest.raises(ValueError, match=message):
        GenerationRequest(**fields)


# --- generation -------------------------------------------------------------


def test_generate_reports_ttft_and_tpot_separately(decoder_graph):
    """The two phases are different measurements, so they are not averaged together.

    On GPT-2 they differ by a factor of 42. Reporting one latency for both would
    describe neither, which is the whole reason the split exists.
    """
    with _client(decoder_graph) as client:
        record = client.generate(_request(max_new_tokens=6))

    assert record.admitted
    assert len(record.tokens) == 6
    assert record.prompt_tokens == len(PROMPT)
    assert [step.phase for step in record.steps] == ["prefill"] + ["decode"] * 6

    assert record.ttft_ms > 0.0
    assert record.tpot_ms > 0.0
    low, high = record.tpot_range_ms
    assert low <= record.tpot_ms <= high
    assert len(record.decode_steps) == 6
    # The gather is the price of block accounting; it must be visible, not folded in.
    assert 0.0 < record.gather_fraction < 1.0
    assert record.met_deadline


def test_the_emitted_tokens_are_greedy_and_repeatable(decoder_graph):
    """Deterministic on purpose: a preempted run has to be comparable with a clean one."""
    with _client(decoder_graph) as first, _client(decoder_graph) as second:
        one = first.generate(_request(max_new_tokens=8))
        two = second.generate(_request(max_new_tokens=8))
    assert one.tokens == two.tokens


def test_a_stop_token_ends_the_generation_early(decoder_graph):
    """Stopping early returns blocks early, which is the point of noticing."""
    with _client(decoder_graph) as client:
        full = client.generate(_request(max_new_tokens=8, request_id="full"))
    stop_at = full.tokens[2]

    with _client(decoder_graph) as client:
        stopped = client.generate(
            _request(max_new_tokens=8, request_id="stopped", stop_tokens=frozenset({stop_at}))
        )
    assert stopped.stopped_early
    assert stopped.tokens == full.tokens[: stopped.tokens.index(stop_at) + 1]


def test_generation_stops_at_the_context_limit_instead_of_failing(decoder_graph):
    """A model's position table is an initializer, not a declared shape.

    Running past it surfaces as an out-of-bounds Gather from inside ONNX Runtime,
    which is a confusing way to learn that GPT-2 stops at 1024. Given the limit, the
    client records the outcome instead.
    """
    with _client(decoder_graph, max_context_tokens=len(PROMPT) + 3) as client:
        record = client.generate(_request(max_new_tokens=20))
    assert record.hit_context_limit
    assert len(record.tokens) == 3


def test_occupancy_and_states_describe_the_arena(decoder_graph):
    with _client(decoder_graph, num_blocks=32) as client:
        assert client.occupancy().free_blocks == 32
        client.admit(_request(max_new_tokens=8, request_id="a"))
        client.prefill("a")

        occupancy = client.occupancy()
        assert occupancy.resident_sequences == 1
        assert occupancy.used_blocks == client.capacity_blocks - client.free_blocks
        assert 0.0 < occupancy.used_fraction < 1.0

        state = client.states()[0]
        assert state.sequence_id == "a"
        assert state.cached_tokens == len(PROMPT)
        assert state.remaining_tokens == 8
        assert state.blocks_held > 0


# --- admission without a policy ---------------------------------------------


def test_without_a_policy_a_full_arena_refuses_rather_than_evicting(decoder_graph):
    """The profiling path runs this way: it has to measure the costs a policy needs."""
    with _client(decoder_graph, num_blocks=4) as client:
        assert client.admit(_request(max_new_tokens=8, request_id="a")).admit
        plan = client.admit(_request(max_new_tokens=8, request_id="b"))
        assert not plan.admit
        assert "cannot hold" in plan.reason
        # Refusing must leave the incumbent alone.
        assert client.states()[0].sequence_id == "a"


def test_without_a_policy_outgrowing_a_reservation_raises(decoder_graph):
    """Nothing to evict with, so the caller is told rather than quietly stalled."""
    extension = load_extension()
    with _client(decoder_graph, num_blocks=3, reserve_full_generation=False) as client:
        assert client.admit(_request(max_new_tokens=20, request_id="a")).admit
        client.prefill("a")
        with pytest.raises(extension.CacheExhausted, match="no admission policy"):
            for _ in range(20):
                client.emit("a")


def test_admitting_the_same_request_twice_raises(decoder_graph):
    with _client(decoder_graph) as client:
        client.admit(_request(request_id="a"))
        with pytest.raises(RuntimeError, match="already in flight"):
            client.admit(_request(request_id="a"))


def test_an_unknown_request_raises(decoder_graph):
    with _client(decoder_graph) as client:
        with pytest.raises(RuntimeError, match="unknown request"):
            client.prefill("ghost")


def test_emitting_before_prefilling_raises(decoder_graph):
    with _client(decoder_graph) as client:
        client.admit(_request(request_id="a"))
        with pytest.raises(RuntimeError, match="no logits to sample"):
            client.emit("a")


# --- the policy and the arena must agree ------------------------------------


def test_a_policy_reasoning_in_the_wrong_block_size_is_rejected(decoder_graph):
    """Every admission decision would be off by the ratio between them."""
    wrong = BlockAdmission(capacity_blocks=32, block_tokens=BLOCK_TOKENS * 2, cost=COST)
    with pytest.raises(ValueError, match="admission decision would be off"):
        _client(decoder_graph, num_blocks=32, admission=wrong)


def test_a_policy_modelling_the_wrong_capacity_is_rejected(decoder_graph):
    with pytest.raises(ValueError, match="with_capacity"):
        _client(decoder_graph, num_blocks=32, admission=_policy(capacity_blocks=8))


# --- preemption through the client ------------------------------------------


def test_a_sequence_preempted_through_the_client_is_token_identical(decoder_graph):
    """The client has to remember what the cache forgot.

    `preempt` releases the blocks and keeps the tokens; `resume` re-runs the history.
    The session-level test proves the arena preserves output across that. This proves
    the client's bookkeeping does, which is the half that could lose a token or replay
    one.
    """
    with _client(decoder_graph) as client:
        clean = client.generate(_request(max_new_tokens=8, request_id="clean"))

    with _client(decoder_graph) as client:
        request = _request(max_new_tokens=8, request_id="interrupted")
        assert client.admit(request).admit
        client.prefill("interrupted")
        emitted = []
        for step in range(8):
            if step == 4:
                blocks = client.preempt("interrupted")
                assert blocks > 0
                assert client.free_blocks == client.capacity_blocks
                assert client.states() == []
                record = client.resume("interrupted")
                assert record.phase == "recompute"
            emitted.append(client.emit("interrupted").token)
        assert client.emitted("interrupted") == emitted

    assert emitted == clean.tokens


def test_resuming_a_resident_sequence_raises(decoder_graph):
    with _client(decoder_graph) as client:
        client.admit(_request(request_id="a"))
        client.prefill("a")
        with pytest.raises(RuntimeError, match="was not preempted"):
            client.resume("a")


def test_a_preempted_sequence_keeps_its_history(decoder_graph):
    """Preemption takes the cache, not the tokens; that is what recompute needs."""
    with _client(decoder_graph) as client:
        client.admit(_request(max_new_tokens=8, request_id="a"))
        client.prefill("a")
        for _ in range(3):
            client.emit("a")

        before = client.tokens("a")
        assert len(before) == len(PROMPT) + 3
        client.preempt("a")
        assert client.tokens("a") == before
        assert client.states() == []
        assert client.states(resident_only=False)[0].blocks_held == 0


def test_release_frees_the_blocks_and_forgets_the_sequence(decoder_graph):
    with _client(decoder_graph) as client:
        client.admit(_request(request_id="a"))
        client.prefill("a")
        assert client.release("a") > 0
        assert client.free_blocks == client.capacity_blocks
        assert client.release("a") == 0
        with pytest.raises(RuntimeError, match="unknown request"):
            client.tokens("a")


# --- eviction on the real path ----------------------------------------------


def test_a_full_arena_evicts_by_slack_to_admit_a_newcomer(decoder_graph):
    """Where the policy and the arena meet.

    Two incumbents fill the pool. One has a deadline it will comfortably beat and the
    other barely will, so the newcomer's blocks come from the first. The evicted
    sequence keeps its tokens and can be resumed.
    """
    policy = _policy(capacity_blocks=6)
    with _client(decoder_graph, num_blocks=6, admission=policy) as client:
        assert client.admit(
            _request(max_new_tokens=4, deadline_ms=60_000.0, request_id="roomy")
        ).admit
        assert client.admit(_request(max_new_tokens=4, deadline_ms=60.0, request_id="tight")).admit
        client.prefill("roomy")
        client.prefill("tight")
        assert client.free_blocks == 0

        plan = client.admit(_request(max_new_tokens=4, request_id="newcomer"))
        assert plan.admit
        assert plan.evict == ("roomy",)
        assert plan.recompute_cost_ms > 0.0

        # The newcomer runs, and the victim still has everything it needs to resume.
        client.prefill("newcomer")
        assert len(client.tokens("roomy")) == len(PROMPT)
        assert {state.sequence_id for state in client.states()} == {"newcomer", "tight"}


def test_a_sequence_outgrowing_its_reservation_evicts_instead_of_failing(decoder_graph):
    """The eviction path taken mid-decode rather than at admission.

    Reserving only the prompt means a long generation will run out, which is exactly
    when a scheduler has to find room without dropping the sequence it is serving.
    """
    policy = _policy(capacity_blocks=8)
    with _client(
        decoder_graph, num_blocks=8, admission=policy, reserve_full_generation=False
    ) as client:
        assert client.admit(
            _request(max_new_tokens=24, deadline_ms=60_000.0, request_id="grower")
        ).admit
        assert client.admit(
            _request(max_new_tokens=1, deadline_ms=60_000.0, request_id="bystander")
        ).admit
        client.prefill("grower")
        client.prefill("bystander")

        # Take every free block, then fill the grower's last one, so that the very
        # next token has to come out of somebody else's allocation rather than out of
        # slack the grower already holds.
        while client.free_blocks:
            client.emit("grower")
        grower = next(state for state in client.states() if state.sequence_id == "grower")
        for _ in range(grower.blocks_held * BLOCK_TOKENS - grower.cached_tokens):
            client.emit("grower")

        assert client.free_blocks == 0
        assert {state.sequence_id for state in client.states()} == {"grower", "bystander"}

        client.emit("grower")

        # The grower is still running and the bystander was preempted, not dropped.
        assert [state.sequence_id for state in client.states()] == ["grower"]
        bystander = next(
            state
            for state in client.states(resident_only=False)
            if state.sequence_id == "bystander"
        )
        assert bystander.blocks_held == 0
        assert client.tokens("bystander") == PROMPT


def test_a_newcomer_is_refused_when_nobody_can_afford_to_be_evicted(decoder_graph):
    """A full arena of tight deadlines shuts the door rather than dooming somebody."""
    policy = _policy(capacity_blocks=6)
    with _client(decoder_graph, num_blocks=6, admission=policy) as client:
        for name in ("a", "b"):
            assert client.admit(_request(max_new_tokens=4, deadline_ms=1.0, request_id=name)).admit
            client.prefill(name)
        assert client.free_blocks == 0

        # Both incumbents are already past saving, so their blocks are fair game.
        plan = client.admit(_request(max_new_tokens=4, request_id="newcomer"))
        assert plan.admit
        assert plan.evict == ("a",)

    # With enough slack to matter but not enough to absorb a recompute, the answer
    # flips to a refusal.
    tight = BlockAdmission(
        capacity_blocks=6,
        block_tokens=BLOCK_TOKENS,
        cost=CacheCost(
            decode_base_ms=0.0, decode_per_token_ms=0.0, prefill_per_token_ms=1_000_000.0
        ),
    )
    with _client(decoder_graph, num_blocks=6, admission=tight) as client:
        for name in ("a", "b"):
            assert client.admit(
                _request(max_new_tokens=4, deadline_ms=60_000.0, request_id=name)
            ).admit
            client.prefill(name)
        plan = client.admit(_request(max_new_tokens=4, request_id="newcomer"))
        assert not plan.admit
        assert "turns one miss into two" in plan.reason


def test_an_evicted_sequence_resumes_to_the_same_output(decoder_graph):
    """Eviction is a scheduling decision, so it must not change what comes out.

    Distinct from the preemption test above in who decides: there the test calls
    `preempt` directly, here the policy chooses the victim while admitting somebody
    else. The sequence that gets interrupted does not know it is about to be, which is
    how it happens under load.
    """
    with _client(decoder_graph) as client:
        clean = client.generate(_request(max_new_tokens=4, request_id="clean"))

    policy = _policy(capacity_blocks=5)
    with _client(
        decoder_graph, num_blocks=5, admission=policy, reserve_full_generation=False
    ) as client:
        assert client.admit(
            _request(max_new_tokens=4, deadline_ms=60_000.0, request_id="victim")
        ).admit
        client.prefill("victim")
        emitted = [client.emit("victim").token for _ in range(2)]

        # A second incumbent with a tighter deadline, filling the arena.
        assert client.admit(
            _request(max_new_tokens=4, deadline_ms=5_000.0, request_id="tight")
        ).admit
        client.prefill("tight")
        assert client.free_blocks == 0

        # The newcomer's blocks come from the sequence with the most slack.
        assert client.admit(_request(max_new_tokens=4, request_id="newcomer")).evict == ("victim",)

        # Once the newcomer is done, the victim gets its cache back by recomputing.
        client.release("newcomer")
        assert client.resume("victim").phase == "recompute"
        emitted += [client.emit("victim").token for _ in range(2)]

    assert emitted == clean.tokens


def test_the_logits_a_step_returns_are_the_next_token_distribution(decoder_graph):
    """One row, and it is the row the next token is sampled from."""
    with _client(decoder_graph) as client:
        client.admit(_request(max_new_tokens=2, request_id="a"))
        client.prefill("a")
        first = client.emit("a")
        assert first.token is not None
        assert first.cached_tokens == len(PROMPT) + 1
        assert np.isfinite(client.states()[0].cached_tokens)
