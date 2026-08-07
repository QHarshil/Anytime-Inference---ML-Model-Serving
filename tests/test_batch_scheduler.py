"""Tests for the continuous batching scheduler.

The scheduler owns one decision -- what runs next -- and the assertion that matters
is that owning it changes nothing about the answers. Every sequence has to emit the
same tokens it would have emitted alone, whether it was batched with seven others,
stalled behind somebody else's prefill chunk, evicted part way through and recomputed,
or all three.

`test_every_sequence_emits_what_it_would_have_alone` is the one that covers that. The
rest pin down the shape of the schedule: that a prefill chunk and a decode step never
share an invocation, that alternation bounds how long a resident sequence waits, and
that the counts a caller reads describe what actually ran.
"""

import statistics

import numpy as np
import pytest

from anytime_serving.serving.batch_scheduler import ContinuousBatchScheduler
from anytime_serving.serving.decoder import DecoderClient, GenerationRequest
from anytime_serving.serving.kv_admission import BlockAdmission, CacheCost
from anytime_serving.serving.onnx_runtime import extension_available

requires_extension = pytest.mark.skipif(
    not extension_available(),
    reason="anytime_runtime is not built; the decoder path lives in it. pip install -e .",
)

pytestmark = requires_extension

BLOCK_TOKENS = 4
COST = CacheCost(decode_base_ms=4.0, decode_per_token_ms=0.005, prefill_per_token_ms=0.35)


def _client(graph, *, num_blocks=64, admission=None, **kwargs):
    return DecoderClient(
        graph, block_tokens=BLOCK_TOKENS, num_blocks=num_blocks, admission=admission, **kwargs
    )


def _policy(*, capacity_blocks):
    return BlockAdmission(capacity_blocks=capacity_blocks, block_tokens=BLOCK_TOKENS, cost=COST)


def _prompt(index, length):
    return [(index * 7 + step) % 60 + 1 for step in range(length)]


def _request(index, *, length=8, max_new_tokens=5, deadline_ms=10_000.0):
    return GenerationRequest(
        prompt=_prompt(index, length),
        max_new_tokens=max_new_tokens,
        deadline_ms=deadline_ms,
        request_id=f"r{index}",
    )


def _alone(graph, request):
    """What this request emits with nothing else in the arena."""
    with _client(graph) as client:
        record = client.generate(request, chunk_tokens=0)
        return list(record.tokens)


# --- the assertion the scheduler exists to preserve --------------------------


@pytest.mark.parametrize("length_bucketing", [False, True])
@pytest.mark.parametrize("chunk_tokens", [3, 4, 64])
@pytest.mark.parametrize("max_batch_size", [1, 3, 8])
def test_every_sequence_emits_what_it_would_have_alone(
    decoder_graph, chunk_tokens, max_batch_size, length_bucketing
):
    """Interleaving must not change any answer, at every width and batch cap.

    Prompt lengths differ so decode batches are right-padded, and a chunk width of 3
    straddles the 4-token block boundary. A batch cap of 1 is the degenerate schedule
    and has to agree too, since it is the reference the batched ones are measured
    against. Bucketing reorders who shares a step, which is exactly the kind of change
    that must not reach the answers: at a cap of 3 there are four sequences to choose
    three from, so it is choosing here rather than taking what it is given.
    """
    requests = [_request(index, length=length) for index, length in enumerate([9, 6, 11, 4])]
    expected = {request.request_id: _alone(decoder_graph, request) for request in requests}

    with _client(decoder_graph) as client:
        scheduler = ContinuousBatchScheduler(
            client,
            chunk_tokens=chunk_tokens,
            max_batch_size=max_batch_size,
            length_bucketing=length_bucketing,
        )
        records = scheduler.run(requests)

    assert set(records) == set(expected)
    for request_id, tokens in expected.items():
        assert records[request_id].admitted is True
        assert records[request_id].tokens == tokens, (
            f"{request_id} emitted differently under the scheduler than alone"
        )


def test_sequences_evicted_and_recomputed_still_agree(decoder_graph):
    """The same claim with the arena too small to hold everybody at once.

    Four sequences reserving three blocks each against an arena of six, so admission
    has to evict and the victims have to be recomputed from their token history. The
    output is still the output.
    """
    requests = [_request(index, length=8, max_new_tokens=4) for index in range(4)]
    expected = {request.request_id: _alone(decoder_graph, request) for request in requests}

    with _client(decoder_graph, num_blocks=6, admission=_policy(capacity_blocks=6)) as client:
        scheduler = ContinuousBatchScheduler(client, chunk_tokens=4, max_batch_size=4)
        records = scheduler.run(requests)

    admitted = [r for r in records.values() if r.admitted]
    assert len(admitted) == 4, "all four fit one at a time, so none should be rejected"
    for request_id, tokens in expected.items():
        assert records[request_id].tokens == tokens


# --- the shape of the schedule ----------------------------------------------


def test_a_step_is_either_a_prefill_chunk_or_a_decode_batch_never_both(decoder_graph):
    """The graph cannot represent a fused step, so the scheduler must never claim one."""
    requests = [_request(index, length=9) for index in range(3)]
    with _client(decoder_graph) as client:
        scheduler = ContinuousBatchScheduler(client, chunk_tokens=4, max_batch_size=3)
        for request in requests:
            scheduler.submit(request)

        kinds = []
        while (step := scheduler.step()) is not None:
            assert step.kind in {"prefill", "decode"}
            if step.kind == "prefill":
                # A prefill chunk is one sequence. Batching prefills would pad on
                # prompt-length variance for no gain: a chunk is already a wide GEMM.
                assert step.batch_size == 1
                assert all(record.phase in {"prefill", "recompute"} for record in step.records)
            else:
                assert all(record.phase == "decode" for record in step.records)
                assert {record.batch_size for record in step.records} == {step.batch_size}
            kinds.append(step.kind)

    assert "prefill" in kinds and "decode" in kinds


def _trace(scheduler):
    """Each iteration's kind, with how many sequences were decoding when it began.

    The condition matters. Consecutive prefill chunks are correct when nothing is
    decoding -- there is nobody to starve, and refusing to get on with the prompts
    would be worse. The guarantee is about a sequence that *is* decoding, so the
    assertion has to know which iterations had one.
    """
    observed = []
    while True:
        decoding_before = len(scheduler.decoding)
        step = scheduler.step()
        if step is None:
            return observed
        observed.append((step.kind, decoding_before))


def test_alternation_bounds_how_long_a_resident_sequence_waits(decoder_graph):
    """At the default of one chunk per decode step, prefill cannot starve decode.

    Enough waiting work that a prefill-priority scheduler would run every chunk of
    every prompt before emitting anything. What has to hold instead is that a sequence
    already decoding never waits through two prefill chunks in a row.
    """
    requests = [_request(index, length=12, max_new_tokens=3) for index in range(4)]
    with _client(decoder_graph) as client:
        scheduler = ContinuousBatchScheduler(
            client, chunk_tokens=3, max_batch_size=8, prefill_chunks_per_decode=1
        )
        for request in requests:
            scheduler.submit(request)
        observed = _trace(scheduler)

    stalled = 0
    for kind, decoding_before in observed:
        if kind == "prefill" and decoding_before > 0:
            stalled += 1
            assert stalled <= 1, (
                f"a decoding sequence waited through {stalled} prefill chunks in a "
                f"row at a budget of 1: {observed}"
            )
        else:
            stalled = 0
    # And the situation the guarantee is about did arise, so this is not vacuous.
    assert any(kind == "prefill" and decoding > 0 for kind, decoding in observed)


def test_raising_the_chunk_budget_lets_prefill_run_ahead(decoder_graph):
    """The knob has to actually move the schedule, or it is decoration.

    With three chunks allowed per decode step, consecutive prefill chunks are exactly
    what should happen -- that is the trade being bought.
    """
    requests = [_request(index, length=12, max_new_tokens=3) for index in range(4)]
    with _client(decoder_graph) as client:
        scheduler = ContinuousBatchScheduler(
            client, chunk_tokens=3, max_batch_size=8, prefill_chunks_per_decode=3
        )
        for request in requests:
            scheduler.submit(request)
        observed = _trace(scheduler)

    # Consecutive chunks while something was decoding, which is the thing a budget of
    # 1 forbids and a budget of 3 is bought in order to allow.
    stalled = 0
    longest = 0
    for kind, decoding_before in observed:
        if kind == "prefill" and decoding_before > 0:
            stalled += 1
            longest = max(longest, stalled)
        else:
            stalled = 0
    assert longest > 1, f"a budget of 3 should let chunks run consecutively: {observed}"
    assert longest <= 3, f"a budget of 3 should not allow more than 3: {observed}"


def test_decode_steps_batch_more_than_one_sequence_when_several_are_ready(decoder_graph):
    """Otherwise this is the old one-at-a-time loop with extra bookkeeping."""
    requests = [_request(index, length=5, max_new_tokens=6) for index in range(4)]
    with _client(decoder_graph) as client:
        scheduler = ContinuousBatchScheduler(client, chunk_tokens=64, max_batch_size=4)
        scheduler.run(requests)
        stats = scheduler.stats

    assert stats.decode_steps > 0
    assert stats.mean_decode_batch > 1.0
    assert stats.tokens_emitted == 4 * 6


def test_the_batch_cap_is_respected(decoder_graph):
    requests = [_request(index, length=5, max_new_tokens=4) for index in range(6)]
    with _client(decoder_graph) as client:
        scheduler = ContinuousBatchScheduler(client, chunk_tokens=64, max_batch_size=2)
        for request in requests:
            scheduler.submit(request)
        while (step := scheduler.step()) is not None:
            if step.kind == "decode":
                assert step.batch_size <= 2


def test_a_sequence_too_large_for_the_arena_is_rejected_rather_than_queued_forever(
    decoder_graph,
):
    """Queueing something no eviction can ever fit would be a hang dressed as patience."""
    small = _request(0, length=4, max_new_tokens=2)
    huge = GenerationRequest(
        prompt=_prompt(1, 60),
        max_new_tokens=20,
        deadline_ms=10_000.0,
        request_id="huge",
    )
    with _client(decoder_graph, num_blocks=6, admission=_policy(capacity_blocks=6)) as client:
        scheduler = ContinuousBatchScheduler(client, chunk_tokens=4)
        records = scheduler.run([small, huge])

    assert records["huge"].admitted is False
    assert "no eviction can make room" in records["huge"].rejection_reason
    assert scheduler.stats.rejections == 1
    assert records["r0"].admitted is True
    assert scheduler.idle() is True


def test_a_stop_token_retires_a_sequence_from_the_batch(decoder_graph):
    """One row finishing must not disturb the others."""
    with _client(decoder_graph) as client:
        scheduler = ContinuousBatchScheduler(client, chunk_tokens=64, max_batch_size=4)
        plain = [_request(index, length=6, max_new_tokens=5) for index in range(3)]
        expected = {r.request_id: _alone(decoder_graph, r) for r in plain}
        first_token = expected["r0"][0]

        stopping = GenerationRequest(
            prompt=_prompt(0, 6),
            max_new_tokens=5,
            deadline_ms=10_000.0,
            request_id="stops",
            stop_tokens=frozenset({first_token}),
        )
        records = scheduler.run([*plain, stopping])

    assert records["stops"].stopped_early is True
    assert records["stops"].tokens == [first_token]
    for request_id, tokens in expected.items():
        assert records[request_id].tokens == tokens


def test_the_context_limit_retires_a_sequence_instead_of_failing(decoder_graph):
    with _client(decoder_graph, max_context_tokens=10) as client:
        scheduler = ContinuousBatchScheduler(client, chunk_tokens=64, max_batch_size=2)
        records = scheduler.run([_request(0, length=8, max_new_tokens=20)])

    assert records["r0"].hit_context_limit is True
    assert len(records["r0"].tokens) == 2


def test_records_describe_what_ran(decoder_graph):
    with _client(decoder_graph) as client:
        scheduler = ContinuousBatchScheduler(client, chunk_tokens=4, max_batch_size=2)
        records = scheduler.run([_request(index, length=9, max_new_tokens=4) for index in range(2)])

    for record in records.values():
        assert record.ttft_ms > 0.0
        assert record.tpot_ms > 0.0
        assert len(record.decode_steps) == 4
        assert record.wall_ms > 0.0
        # A chunked prefill of 9 tokens at width 4 is three chunks.
        assert len([s for s in record.steps if s.phase == "prefill"]) == 3


def test_the_scheduler_reports_idle_only_when_nothing_is_left(decoder_graph):
    with _client(decoder_graph) as client:
        scheduler = ContinuousBatchScheduler(client, chunk_tokens=64)
        assert scheduler.idle() is True
        scheduler.submit(_request(0, length=5, max_new_tokens=2))
        assert scheduler.idle() is False
        assert scheduler.step() is not None
        scheduler.drain()
        assert scheduler.idle() is True
        assert scheduler.step() is None


# --- construction and misuse -------------------------------------------------


def test_a_client_that_does_not_reserve_the_whole_generation_is_refused(decoder_graph):
    """A batched step could then exhaust the arena for one row with no way to say which."""
    with _client(decoder_graph, reserve_full_generation=False) as client:
        with pytest.raises(ValueError, match="reserves the full generation"):
            ContinuousBatchScheduler(client)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"max_batch_size": 0}, "max_batch_size must be positive"),
        ({"prefill_chunks_per_decode": 0}, "prefill_chunks_per_decode must be positive"),
    ],
)
def test_a_malformed_scheduler_is_refused(decoder_graph, kwargs, message):
    with _client(decoder_graph) as client:
        with pytest.raises(ValueError, match=message):
            ContinuousBatchScheduler(client, **kwargs)


def test_submitting_the_same_request_twice_raises(decoder_graph):
    with _client(decoder_graph) as client:
        scheduler = ContinuousBatchScheduler(client)
        scheduler.submit(_request(0))
        with pytest.raises(ValueError, match="already been submitted"):
            scheduler.submit(_request(0))


def test_drain_respects_a_step_budget(decoder_graph):
    """A caller that would rather stop than hang gets to say so."""
    with _client(decoder_graph) as client:
        scheduler = ContinuousBatchScheduler(client, chunk_tokens=3)
        for index in range(3):
            scheduler.submit(_request(index, length=12, max_new_tokens=8))
        scheduler.drain(max_steps=2)
        assert scheduler.idle() is False
        assert scheduler.stats.prefill_steps + scheduler.stats.decode_steps <= 2


def test_the_default_chunk_width_comes_from_the_runtime(decoder_graph):
    with _client(decoder_graph) as client:
        scheduler = ContinuousBatchScheduler(client)
        assert scheduler.chunk_tokens == client.default_chunk_tokens


def test_logits_stay_the_next_token_distribution_through_a_batched_schedule(decoder_graph):
    with _client(decoder_graph) as client:
        scheduler = ContinuousBatchScheduler(client, chunk_tokens=64, max_batch_size=2)
        scheduler.submit(_request(0, length=5, max_new_tokens=3))
        scheduler.submit(_request(1, length=5, max_new_tokens=3))
        scheduler.step()
        scheduler.step()
        step = scheduler.step()
        assert step is not None
        for record in step.records:
            row = client.next_token_logits(record.request_id)
            assert row.ndim == 1
            assert np.isfinite(row).all()


# --- length bucketing --------------------------------------------------------
#
# Every assertion here is about which sequences share a step, never about how long a
# step took. A batch's composition is what the rule decides; wall time on a fixture
# whose graph runs in microseconds is noise, and asserting on it is what put a flaky
# test through CI last round.


# Iterations allowed to get every sequence decoding. Alternation spends about three
# per sequence, so this is generous enough that reaching it means a hang rather than
# a slow start.
_ASSEMBLY_BUDGET = 400


def _batch_trace(client, scheduler, requests, *, steps):
    """Assemble every sequence into the decoding set, then record the next `steps` batches.

    Returns `(request_ids, cached_length_range)` per decode step. Assembly is separate
    because the ordering rule has nothing to decide until residency exceeds the batch
    width, which is the whole condition under which bucketing does anything at all.

    The range is snapshotted per step rather than recomputed at the end. Rows served
    more often have grown more, so one set of final lengths does not describe a batch
    from twenty steps ago -- and a rule that keeps picking the same rows is exactly the
    one that would be flattered by the mistake. Taken immediately after the step, when
    every row of that batch has gained the same single token, it is the range that was
    actually selected.
    """
    for request in requests:
        scheduler.submit(request)

    for _ in range(_ASSEMBLY_BUDGET):
        if len(scheduler.decoding) == len(requests):
            break
        if scheduler.step() is None:
            break
    else:  # pragma: no cover - a budget this size means something is wrong
        raise AssertionError("the sequences never all reached the decoding set")
    assert len(scheduler.decoding) == len(requests), (
        f"only {len(scheduler.decoding)} of {len(requests)} became resident; "
        "the arena is too small for this test to be about ordering"
    )

    trace = []
    while len(trace) < steps:
        step = scheduler.step()
        if step is None:
            break
        if step.kind != "decode":
            continue
        lengths = {state.sequence_id: state.cached_tokens for state in client.states()}
        served = [lengths[r] for r in step.request_ids if r in lengths]
        trace.append((step.request_ids, max(served) - min(served) if served else 0))
    return trace


def _longest_gap(trace, request_ids):
    """The most decode steps any sequence went unserved.

    Counted from the start of the window, so a sequence never served at all is charged
    the whole of it rather than being silently skipped.
    """
    worst = 0
    for request_id in request_ids:
        last = -1
        for index, (batch, _) in enumerate(trace):
            if request_id in batch:
                worst = max(worst, index - last)
                last = index
        worst = max(worst, len(trace) - 1 - last)
    return worst


def test_bucketing_puts_similar_lengths_in_one_step(decoder_graph):
    """The mechanism: a batch runs at its longest row, so cluster the rows.

    Three clusters far enough apart that the tokens emitted during the trace cannot
    blur them. Arrival order interleaves the clusters and pays the full range on most
    steps; bucketing should mostly stay inside one.
    """
    lengths = [8, 12, 16, 64, 68, 72, 128, 132]
    requests = [
        _request(index, length=length, max_new_tokens=24) for index, length in enumerate(lengths)
    ]

    spreads = {}
    for bucketing in (False, True):
        with _client(decoder_graph, num_blocks=192) as client:
            scheduler = ContinuousBatchScheduler(
                client, chunk_tokens=256, max_batch_size=3, length_bucketing=bucketing
            )
            trace = _batch_trace(client, scheduler, requests, steps=12)
            spreads[bucketing] = [spread for _, spread in trace]

    assert len(spreads[True]) >= 8, "too few decode steps observed to compare"
    bucketed = statistics.fmean(spreads[True])
    arrival = statistics.fmean(spreads[False])
    assert bucketed < arrival, (
        f"bucketing left a mean cached-length range of {bucketed:.1f} against arrival "
        f"order's {arrival:.1f}; it is not clustering anything"
    )


def test_no_sequence_waits_longer_than_the_decoding_set(decoder_graph):
    """The starvation bound, which is a property of anchoring rather than a guard.

    The anchor is always the head of the queue and is always moved to the back, so an
    unserved sequence's position strictly decreases and it becomes the anchor within N
    steps. One sequence is deliberately far from the pack: a rule that only looked at
    length would never pick it.
    """
    lengths = [8, 12, 16, 20, 24, 28, 32, 160]
    requests = [
        _request(index, length=length, max_new_tokens=64) for index, length in enumerate(lengths)
    ]
    with _client(decoder_graph, num_blocks=224) as client:
        scheduler = ContinuousBatchScheduler(
            client, chunk_tokens=256, max_batch_size=3, length_bucketing=True
        )
        trace = _batch_trace(client, scheduler, requests, steps=32)
        still_decoding = len(scheduler.decoding)

    assert len(trace) > 3 * len(requests), (
        "the window has to be several times the bound for the bound to be tested"
    )
    assert still_decoding == len(requests), (
        "a sequence retired inside the window, so the set the bound is stated over "
        "changed underneath it"
    )
    gap = _longest_gap(trace, [request.request_id for request in requests])
    assert gap <= len(requests), (
        f"a sequence went {gap} decode steps unserved with {len(requests)} decoding; "
        f"anchoring bounds that at {len(requests)}"
    )


def test_a_pure_length_sort_starves_the_outlier(decoder_graph, monkeypatch):
    """Fault injection: delete the anchor and the bound has to fail.

    Otherwise the test above is passing on something other than what it claims. With
    the anchor gone and the batch taken by length alone, the sequence 128 tokens clear
    of the pack is never among the three shortest and never runs again.
    """
    lengths = [8, 12, 16, 20, 24, 28, 32, 160]
    requests = [
        _request(index, length=length, max_new_tokens=64) for index, length in enumerate(lengths)
    ]
    with _client(decoder_graph, num_blocks=224) as client:
        scheduler = ContinuousBatchScheduler(
            client, chunk_tokens=256, max_batch_size=3, length_bucketing=True
        )

        def pure_length_sort(self=scheduler):
            lengths_now = self._cached_lengths()
            ordered = sorted(self._decoding, key=lambda r: (lengths_now[r], r))
            return ordered[: self._max_batch_size]

        monkeypatch.setattr(scheduler, "_select_batch", pure_length_sort)
        trace = _batch_trace(client, scheduler, requests, steps=32)

    gap = _longest_gap(trace, [request.request_id for request in requests])
    assert gap > len(requests), (
        "a pure length sort was supposed to starve the outlier and did not, so the "
        "starvation test is not testing the anchor"
    )


def test_bucketing_is_a_no_op_when_everyone_fits_in_a_step(decoder_graph):
    """With residency at or below the batch width there is no choice to make.

    This is what lets `profile_batching.py` stand without re-measuring: it assembles
    exactly `max_batch_size` sequences, so every batch holds all of them whatever the
    ordering rule is, and the padding regimes it measures cannot move.
    """
    lengths = [8, 40, 12, 72]
    requests = [
        _request(index, length=length, max_new_tokens=16) for index, length in enumerate(lengths)
    ]

    traces = {}
    for bucketing in (False, True):
        with _client(decoder_graph, num_blocks=128) as client:
            scheduler = ContinuousBatchScheduler(
                client,
                chunk_tokens=256,
                max_batch_size=len(requests),
                length_bucketing=bucketing,
            )
            traces[bucketing] = _batch_trace(client, scheduler, requests, steps=8)

    assert traces[True] == traces[False]
    assert all(len(batch) == len(requests) for batch, _ in traces[True])


def test_bucketing_off_still_takes_the_front_of_the_queue(decoder_graph):
    """The default has to be the schedule every recorded number was measured under.

    Checked only on steps that did not just promote a sequence out of prefill. A
    promotion appends to the decoding set inside the same iteration, so the queue this
    loop saw beforehand is genuinely not the one the batch was drawn from -- and once
    the queue is already at least a batch wide, appending to its back cannot change
    its front anyway.
    """
    lengths = [8, 40, 12, 72, 20, 96]
    requests = [
        _request(index, length=length, max_new_tokens=16) for index, length in enumerate(lengths)
    ]
    with _client(decoder_graph, num_blocks=192) as client:
        scheduler = ContinuousBatchScheduler(client, chunk_tokens=256, max_batch_size=3)
        for request in requests:
            scheduler.submit(request)
        assert scheduler.length_bucketing is False

        observed = 0
        for _ in range(_ASSEMBLY_BUDGET):
            before = scheduler.decoding
            step = scheduler.step()
            if step is None:
                break
            if step.kind == "decode" and len(before) >= 3:
                assert step.request_ids == before[:3]
                observed += 1
            if observed >= 12:
                break
    assert observed >= 12, "not enough decode steps to be sure of the ordering"
