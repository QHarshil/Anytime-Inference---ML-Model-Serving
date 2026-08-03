"""Tests for the prefill/decode split, against the cache it replaces.

The block allocator's claim is narrow and worth stating precisely: it computes the
same thing as holding a sequence's KV contiguously, while making the arena's
occupancy a number that admission and eviction can act on. It does not claim to be
faster -- feeding the `present` tensors straight back costs no gather at all.

So the reference here is that faster path. `tests/decoder_reference.py` holds both
loops.

Which bar applies depends on what is being compared, and getting that wrong is how
this file first went red on CI:

- **Same ONNX Runtime instance, cache held two ways.** Only the source of the bytes
  differs, so the bar is **bitwise**. Anything less means the gather is corrupting
  something. This is the test that actually covers the block allocator.
- **Two independent ONNX Runtime builds.** The extension links its own SDK and the
  wheel ships another, and on x86-64 they dispatch to different MLAS kernels, so a
  reduction accumulates in a different order. Measured 8.6e-07 relative, around seven
  float32 ULP. Bitwise held on arm64 only because both took the same NEON path, which
  made it look like a guarantee. The bar is token identity plus float32 agreement.
- **A recomputed sequence against an uninterrupted one.** One wide pass against many
  narrow ones, about 6e-05 on GPT-2. Token identity, with the winning margin asserted
  so it is not passing by luck.

Token identity is the hard assertion in all three cases: a gather that reads the
wrong block moves logits by hundreds, not by ULPs, so it changes the argmax.
"""

import numpy as np
import pytest

from anytime_serving.serving.onnx_runtime import extension_available, load_extension
from tests.conftest import HEAD_DIM, KV_HEADS, LAYERS, VOCAB
from tests.decoder_reference import (
    engine_runner,
    reference_generate,
    session_generate,
    top_two_margin,
    wheel_runner,
)

requires_extension = pytest.mark.skipif(
    not extension_available(),
    reason="anytime_runtime is not built; the decoder session lives in it. pip install -e .",
)

pytestmark = requires_extension

PROMPT = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3]
STEPS = 8
SYNTHETIC_GEOMETRY = {"layers": LAYERS, "kv_heads": KV_HEADS, "head_dim": HEAD_DIM}


def _session(graph, *, block_tokens=4, num_blocks=64):
    return load_extension().DecoderSession(
        str(graph), block_tokens=block_tokens, num_blocks=num_blocks
    )


# Bound for comparisons across two independent ONNX Runtime builds. Derived rather
# than tuned until it passed: float32 carries about 1.2e-07 per operation, the
# measured disagreement on x86-64 is 8.6e-07 (roughly seven ULP through a reduction),
# and this is 12x that. It is also five orders of magnitude below a real fault -- a
# gather reading the wrong block moved these logits by 1.8e+03 when it was tried.
CROSS_BUILD_RTOL = 1e-5
CROSS_BUILD_ATOL = 1e-3


def _assert_bitwise(reference, measured, label):
    for step, (want, got) in enumerate(zip(reference, measured, strict=True)):
        np.testing.assert_array_equal(
            got, want, err_msg=f"{label}: step {step} differs (0 = prefill, then decode steps)"
        )


def _assert_same_to_float32(reference, measured, label):
    for step, (want, got) in enumerate(zip(reference, measured, strict=True)):
        np.testing.assert_allclose(
            got,
            want,
            rtol=CROSS_BUILD_RTOL,
            atol=CROSS_BUILD_ATOL,
            err_msg=f"{label}: step {step} differs by more than float32 accumulation "
            f"error (0 = prefill, then decode steps)",
        )


# --- against a contiguous cache ---------------------------------------------


@pytest.mark.parametrize("chunk_tokens", [0, 3, 4, 8, 64])
def test_the_block_allocated_cache_matches_a_contiguous_one_bitwise(decoder_graph, chunk_tokens):
    """The core claim, across prefill widths that do and do not align to blocks.

    Chunk widths of 3 and 4 straddle the 4-token block boundary in different ways, so
    a gather or scatter that assumed a chunk started at a block boundary is caught
    here. A width above the prompt length collapses to a single pass.

    Both sides run through the same in-process engine, so this isolates the block
    allocator itself rather than comparing two ONNX Runtime instances.
    """
    engine = load_extension().Engine([("m", str(decoder_graph))])
    expected_tokens, expected_logits = reference_generate(
        engine_runner(engine, "m"),
        PROMPT,
        STEPS,
        chunk_tokens=chunk_tokens,
        **SYNTHETIC_GEOMETRY,
    )
    tokens, logits = session_generate(
        _session(decoder_graph), "a", PROMPT, STEPS, chunk_tokens=chunk_tokens
    )
    assert tokens == expected_tokens
    _assert_bitwise(expected_logits, logits, f"chunk_tokens={chunk_tokens}")


def test_the_session_matches_an_independent_onnxruntime(decoder_graph):
    """Cross-checked against the wheel, not just against the engine beside it.

    The same discipline as `tests/test_runtime_engine.py`: a replacement is validated
    against the thing it replaces, through a separate copy of the library. This also
    covers `position_ids` and `attention_mask`, which the reference loop sets
    explicitly and the session derives -- a decode step that offset its positions
    wrongly would diverge here, and by far more than float32 noise.

    Not bitwise, and the reason is the point. The extension links its own ONNX Runtime
    SDK while the wheel ships a separate build of the same version, and on x86-64 the
    two dispatch to different MLAS kernels: the reduction in this graph accumulates in
    a different order and the last one or two float32 digits differ. This test asserted
    bitwise equality at first and passed locally on arm64, where both builds take the
    same NEON path -- then failed on every x86-64 CI job. Bitwise was never derivable
    across two builds; it only looked that way on one architecture.

    What is derivable is that both compute the same function, so the tokens match
    exactly and the logits match to float32 accumulation error.
    """
    ort = pytest.importorskip("onnxruntime")
    session = ort.InferenceSession(str(decoder_graph), providers=["CPUExecutionProvider"])
    expected_tokens, expected_logits = reference_generate(
        wheel_runner(session), PROMPT, STEPS, chunk_tokens=0, **SYNTHETIC_GEOMETRY
    )
    tokens, logits = session_generate(_session(decoder_graph), "a", PROMPT, STEPS, chunk_tokens=0)
    assert tokens == expected_tokens
    _assert_same_to_float32(expected_logits, logits, "wheel reference")


@pytest.mark.parametrize("chunk_tokens", [3, 4, 8])
def test_chunked_prefill_fills_the_cache_the_same_way(decoder_graph, chunk_tokens):
    """Splitting a prompt must not change what ends up cached.

    Prefill is chunked by default because it measured both faster and smaller, so the
    default path has to be the one that is checked, not the fallback.
    """
    single = _session(decoder_graph)
    single.open("a", 64)
    chunked = _session(decoder_graph)
    chunked.open("a", 64)

    from_single = single.prefill("a", PROMPT, chunk_tokens=0)
    from_chunks = chunked.prefill("a", PROMPT, chunk_tokens=chunk_tokens)

    assert from_single.runs == 1
    assert from_chunks.runs == -(-len(PROMPT) // chunk_tokens)
    assert from_single.length == from_chunks.length == len(PROMPT)
    np.testing.assert_array_equal(from_chunks.logits, from_single.logits)

    # And the cache itself, read through the next step rather than inspected.
    np.testing.assert_array_equal(chunked.decode("a", 7).logits, single.decode("a", 7).logits)


# --- preemption -------------------------------------------------------------


@pytest.mark.parametrize("recompute_at", [0, 3, STEPS - 1])
def test_a_preempted_sequence_emits_token_identical_output(decoder_graph, recompute_at):
    """A sequence whose cache was thrown away must finish saying the same thing.

    This is what makes eviction usable at all. Under memory pressure the policy
    releases a victim's blocks and keeps its tokens; when it is readmitted, its whole
    history is re-run. If that produced different output, eviction would be a
    correctness bug rather than a scheduling decision, and the difference would be
    invisible -- the model would carry on emitting fluent text.

    Token identity is the claim, not bitwise logits: recomputing runs one wide pass
    where the uninterrupted path ran many narrow ones. The margin is asserted too, so
    that if this ever flakes it is clear whether the cause is a real divergence or a
    prompt where two candidates were always within float noise of each other.
    """
    clean_tokens, clean_logits = session_generate(
        _session(decoder_graph), "a", PROMPT, STEPS, chunk_tokens=0
    )
    preempted_tokens, preempted_logits = session_generate(
        _session(decoder_graph), "a", PROMPT, STEPS, chunk_tokens=0, recompute_at=recompute_at
    )

    drift = max(
        float(np.max(np.abs(a - b))) for a, b in zip(clean_logits, preempted_logits, strict=True)
    )
    margin = min(top_two_margin(row) for row in clean_logits)
    assert margin > 100.0 * max(drift, 1e-9), (
        f"the winning logit led by only {margin:.3e} while recompute moved the logits "
        f"by {drift:.3e}, so token identity here would be luck rather than a result. "
        f"Choose a prompt with a clearer argmax."
    )
    assert preempted_tokens == clean_tokens


def test_preemption_returns_the_blocks_while_it_is_out(decoder_graph):
    """The point of evicting is that the blocks become available meanwhile."""
    session = _session(decoder_graph, num_blocks=8)
    assert session.open("victim", 16) is True
    session.prefill("victim", PROMPT, chunk_tokens=0)
    assert session.free_blocks == 4

    assert session.release("victim") == 4
    assert session.free_blocks == 8
    # Which is enough for someone else to be admitted in the gap.
    assert session.open("newcomer", 32) is True
    assert session.free_blocks == 0


# --- the contract -----------------------------------------------------------


def test_prefill_returns_only_the_next_token_distribution(decoder_graph):
    """One row, not one per position.

    The graph returns logits for every position it was given, which is 206 MB for a
    1024-token GPT-2 prefill when sampling reads one row.
    """
    session = _session(decoder_graph)
    session.open("a", 64)
    result = session.prefill("a", PROMPT)
    assert result.logits.shape == (VOCAB,)
    assert result.logits.dtype == np.float32
    assert session.decode("a", 1).logits.shape == (VOCAB,)


def test_timings_account_for_the_step(decoder_graph):
    """Every phase is reported, and a cold prefill has nothing to gather.

    Not exactly zero: the phase still sizes its staging buffers, which on the first
    step of a sequence means allocating them. What must hold is that copying nothing
    costs far less than the run it precedes.
    """
    session = _session(decoder_graph)
    session.open("a", 64)

    cold = session.prefill("a", PROMPT, chunk_tokens=0)
    assert cold.timings.run_ms > 0.0
    assert cold.timings.gather_ms < cold.timings.run_ms
    assert cold.timings.scatter_ms > 0.0
    assert cold.timings.total_ms >= cold.timings.run_ms

    warm = session.decode("a", 1)
    assert warm.timings.gather_ms > 0.0
    assert warm.timings.total_ms >= warm.timings.gather_ms + warm.timings.run_ms


def test_a_chunked_prefill_reports_how_many_runs_it_took(decoder_graph):
    session = _session(decoder_graph)
    session.open("a", 64)
    assert session.prefill("a", PROMPT, chunk_tokens=4).runs == 3
    assert session.decode("a", 1).runs == 1


def test_the_present_prefix_invariant_is_verified_once_per_sequence(decoder_graph):
    """Checked, then trusted, then checked again after readmission.

    Verifying every step would cost a full-cache comparison per token; never
    verifying would let the scatter's tail-only assumption rot silently. Once per
    sequence is the compromise, and a preempted sequence is a new one.
    """
    session = _session(decoder_graph)
    session.open("a", 64)

    # A single-pass prefill has an empty past, so there is no prefix to check yet.
    assert session.prefill("a", PROMPT, chunk_tokens=0).timings.verify_ms == 0.0
    assert session.decode("a", 1).timings.verify_ms > 0.0
    assert session.decode("a", 2).timings.verify_ms == 0.0
    assert session.decode("a", 3).timings.verify_ms == 0.0

    session.release("a")
    session.open("a", 64)
    session.prefill("a", PROMPT, chunk_tokens=0)
    assert session.decode("a", 1).timings.verify_ms > 0.0


def test_a_chunked_prefill_verifies_on_its_second_chunk(decoder_graph):
    """The first chunk with a non-empty past is where the check belongs."""
    session = _session(decoder_graph)
    session.open("a", 64)
    assert session.prefill("a", PROMPT, chunk_tokens=4).timings.verify_ms > 0.0
    assert session.decode("a", 1).timings.verify_ms == 0.0


def test_decoding_before_prefilling_raises(decoder_graph):
    session = _session(decoder_graph)
    session.open("a", 64)
    with pytest.raises(RuntimeError, match="empty cache"):
        session.decode("a", 1)


def test_prefilling_a_second_time_raises(decoder_graph):
    """Extending a sequence is decode's job; starting over means releasing it."""
    session = _session(decoder_graph)
    session.open("a", 64)
    session.prefill("a", PROMPT)
    with pytest.raises(RuntimeError, match="already holds"):
        session.prefill("a", PROMPT)


def test_prefill_needs_at_least_one_token(decoder_graph):
    session = _session(decoder_graph)
    session.open("a", 64)
    with pytest.raises(ValueError, match="at least one token"):
        session.prefill("a", [])


# --- driving prefill a chunk at a time ---------------------------------------


@pytest.mark.parametrize("width", [3, 4, 7])
def test_extend_chunk_by_chunk_matches_prefill_driving_its_own_chunks(decoder_graph, width):
    """A scheduler drives the chunks; prefill drives them itself. Same cache either way.

    This is what makes the chunk boundary usable as a preemption point: if advancing a
    prompt from outside produced a different cache from advancing it inside, a
    scheduler could not interleave without changing the answer.

    Compared through the step after, not just on the prefill's own logits, so the
    cache is what is being checked rather than one output row.
    """
    inside = _session(decoder_graph)
    inside.open("a", 64)
    expected = inside.prefill("a", PROMPT, chunk_tokens=width)

    outside = _session(decoder_graph)
    outside.open("a", 64)
    last = None
    runs = 0
    for start in range(0, len(PROMPT), width):
        last = outside.extend("a", PROMPT[start : start + width])
        runs += 1

    assert last is not None
    assert runs == expected.runs
    assert last.length == expected.length == len(PROMPT)
    np.testing.assert_array_equal(last.logits, expected.logits)

    for token in (11, 12, 13):
        np.testing.assert_array_equal(
            outside.decode("a", token).logits, inside.decode("a", token).logits
        )


def test_extend_continues_a_sequence_that_prefill_would_refuse(decoder_graph):
    """prefill refuses a non-empty sequence; extend is how a chunk after the first runs."""
    session = _session(decoder_graph)
    session.open("a", 64)
    session.extend("a", PROMPT[:4])
    with pytest.raises(RuntimeError, match="already holds"):
        session.prefill("a", PROMPT[4:])
    assert session.extend("a", PROMPT[4:]).length == len(PROMPT)


def test_extend_needs_at_least_one_token(decoder_graph):
    session = _session(decoder_graph)
    session.open("a", 64)
    with pytest.raises(ValueError, match="at least one token"):
        session.extend("a", [])


def test_extend_on_an_unknown_sequence_raises(decoder_graph):
    session = _session(decoder_graph)
    with pytest.raises(RuntimeError, match="unknown sequence"):
        session.extend("ghost", [1, 2])


# --- batched decode ----------------------------------------------------------
#
# The bar here is token identity plus float32 agreement, not bitwise, and that is a
# deliberate weakening. Batching changes the GEMM shape, which can change which MLAS
# kernel runs and therefore the order a reduction accumulates in. Measured on this
# host every configuration below came out bitwise equal, and asserting that is
# exactly the mistake `Hold cross-build comparisons to float32` was fixing: bitwise
# held on arm64 and reddened every x86-64 job. What the assertion has to catch is a
# row reading the wrong offset or padding leaking into the cache, and the fixture
# moves those by 6e-02 to 4e-01 relative -- four orders above the bound below.


def _prefilled(graph, names, lengths, *, num_blocks=256, slack=8):
    """Open and prefill one sequence per name, each with a distinct prompt.

    `slack` is how many token positions beyond the prompt each sequence reserves.
    Zero means every sequence sits exactly at its reservation, so the next decode
    step has to take a block -- which is how the all-or-nothing test arranges a
    batch that cannot fit without any single row being at fault.
    """
    session = _session(graph, num_blocks=num_blocks)
    prompts = {}
    for index, (name, length) in enumerate(zip(names, lengths, strict=True)):
        prompt = [(index * 7 + step) % (VOCAB - 2) + 1 for step in range(length)]
        prompts[name] = prompt
        assert session.open(name, length + slack) is True
        session.prefill(name, prompt, chunk_tokens=0)
    return session, prompts


@pytest.mark.parametrize(
    "lengths",
    [
        pytest.param([6, 6, 6, 6], id="uniform"),
        pytest.param([13, 9, 5, 2], id="mixed"),
        pytest.param([17, 1], id="one-long-one-minimal"),
        pytest.param([8], id="batch-of-one"),
    ],
)
def test_a_batched_decode_step_matches_the_same_sequences_run_alone(decoder_graph, lengths):
    """The claim batching rests on, across the length spreads that pad differently.

    Uniform lengths pad nothing. A mixed batch right-pads three of four rows, and the
    one-long-one-minimal case pads sixteen of seventeen positions, which is where an
    offset that confused the row's own length with the batch's width shows up.

    Four steps rather than one: the first step's logits would agree even if the new
    KV were scattered to the wrong index, because that write is only read back on the
    step after. Continuing is what makes the cache itself the thing under test.
    """
    names = [f"s{i}" for i in range(len(lengths))]
    batched, prompts = _prefilled(decoder_graph, names, lengths)
    solo, _ = _prefilled(decoder_graph, [f"t{i}" for i in range(len(lengths))], lengths)

    feed = [(index * 3 + 5) % (VOCAB - 2) + 1 for index in range(len(lengths))]
    for step in range(4):
        result = batched.decode_batch(names, feed)
        assert len(result.rows) == len(lengths)
        for index, name in enumerate(names):
            alone = solo.decode(f"t{index}", feed[index])
            got = np.asarray(result.rows[index].logits)
            want = np.asarray(alone.logits)
            assert int(np.argmax(got)) == int(np.argmax(want)), (
                f"step {step}, row {index}: batched and sequential picked different tokens"
            )
            np.testing.assert_allclose(
                got,
                want,
                rtol=CROSS_BUILD_RTOL,
                atol=CROSS_BUILD_ATOL,
                err_msg=f"step {step}, row {index} differs by more than float32 accumulation error",
            )
            assert result.rows[index].length == len(prompts[name]) + step + 1
            assert alone.length == result.rows[index].length
        feed = [int(np.argmax(np.asarray(row.logits))) % (VOCAB - 2) + 1 for row in result.rows]


def test_padding_is_cleared_rather_than_left_from_the_previous_step(decoder_graph):
    """A long batch first, then a short one reusing the same staging buffers.

    The buffers are reused across steps and only grow, so after a wide batch the
    region a narrow one right-pads holds the earlier batch's KV. The fixture reduces
    its whole `present` unmasked, so anything left there reaches the logits: if this
    agrees with the same sequences run alone, the padding was cleared.
    """
    names = ["a", "b", "c", "d"]
    session, _ = _prefilled(decoder_graph, names, [20, 18, 16, 14])
    session.decode_batch(names, [2, 3, 4, 5])

    short = ["p", "q"]
    batched, prompts = _prefilled(decoder_graph, short, [9, 3])
    # Same session, so the buffers a wide batch grew are the ones this narrow batch
    # right-pads into.
    for name, length in zip(short, [9, 3], strict=True):
        prompt = prompts[name]
        assert session.open(name, length + 8) is True
        session.prefill(name, prompt, chunk_tokens=0)
    reused = session.decode_batch(short, [7, 11])
    fresh = batched.decode_batch(short, [7, 11])

    for index in range(len(short)):
        got = np.asarray(reused.rows[index].logits)
        want = np.asarray(fresh.rows[index].logits)
        assert int(np.argmax(got)) == int(np.argmax(want))
        np.testing.assert_allclose(got, want, rtol=CROSS_BUILD_RTOL, atol=CROSS_BUILD_ATOL)


def test_padding_costs_nothing_when_every_row_is_the_same_length(decoder_graph):
    """pad_ms is reported apart from gather_ms because they scale with different things.

    A uniform batch pads no positions at all, so the clear has nothing to do. This
    asserts the accounting, not a duration: the two are timed separately so that a
    slow step says which of the two it was paying for.
    """
    names = ["a", "b", "c"]
    session, _ = _prefilled(decoder_graph, names, [7, 7, 7])
    uniform = session.decode_batch(names, [2, 3, 4])
    assert uniform.timings.pad_ms == 0.0

    spread, _ = _prefilled(decoder_graph, ["x", "y", "z"], [30, 6, 2])
    varied = spread.decode_batch(["x", "y", "z"], [2, 3, 4])
    assert varied.timings.pad_ms > 0.0


def test_the_batch_reports_one_set_of_timings_and_no_per_row_ones(decoder_graph):
    """The rows share a Run, so a per-row total would be an invented number."""
    names = ["a", "b"]
    session, _ = _prefilled(decoder_graph, names, [5, 5])
    result = session.decode_batch(names, [2, 3])

    assert result.timings.run_ms > 0.0
    assert result.timings.total_ms >= result.timings.run_ms
    for row in result.rows:
        assert row.runs == 1
        assert row.timings.run_ms == 0.0
        assert row.timings.total_ms == 0.0


def test_a_sequence_may_not_appear_twice_in_one_batch(decoder_graph):
    """Both rows would scatter into the same blocks and the second would win."""
    session, _ = _prefilled(decoder_graph, ["a", "b"], [5, 5])
    with pytest.raises(ValueError, match="appears twice"):
        session.decode_batch(["a", "b", "a"], [1, 2, 3])


def test_a_batch_needs_one_token_per_sequence(decoder_graph):
    session, _ = _prefilled(decoder_graph, ["a", "b"], [5, 5])
    with pytest.raises(ValueError, match="one token is emitted per sequence"):
        session.decode_batch(["a", "b"], [1])


def test_an_empty_batch_raises(decoder_graph):
    session, _ = _prefilled(decoder_graph, ["a"], [5])
    with pytest.raises(ValueError, match="at least one sequence"):
        session.decode_batch([], [])


def test_a_batch_containing_an_unprefilled_sequence_raises(decoder_graph):
    session, _ = _prefilled(decoder_graph, ["a"], [5])
    session.open("b", 16)
    with pytest.raises(RuntimeError, match="empty cache"):
        session.decode_batch(["a", "b"], [1, 2])


def test_a_batch_containing_an_unknown_sequence_raises(decoder_graph):
    session, _ = _prefilled(decoder_graph, ["a"], [5])
    with pytest.raises(RuntimeError, match="unknown sequence"):
        session.decode_batch(["a", "ghost"], [1, 2])


def test_a_batch_that_does_not_fit_reserves_nothing(decoder_graph):
    """All or nothing, so a refused batch leaves the arena exactly as it was.

    Reserving row by row and failing part way would leave the earlier rows holding
    blocks for a step that never runs, and the caller with no way to know which.
    """
    exhausted = load_extension().CacheExhausted
    names = ["a", "b", "c"]
    # Sixteen tokens at four per block is exactly four blocks, reserved with no
    # slack, so every row needs a fifth block to take another token: a shortfall of
    # three against the two that are free. Two is enough for either of the first two
    # rows alone, so this fails as a batch rather than because any one row is
    # individually impossible.
    session, _ = _prefilled(decoder_graph, names, [16, 16, 16], num_blocks=14, slack=0)
    assert session.free_blocks == 2

    before_free = session.free_blocks
    before_held = {name: session.blocks_held(name) for name in names}
    before_length = {name: session.length(name) for name in names}

    with pytest.raises(exhausted, match="Nothing was reserved"):
        session.decode_batch(names, [1, 2, 3])

    assert session.free_blocks == before_free
    for name in names:
        assert session.blocks_held(name) == before_held[name]
        assert session.length(name) == before_length[name]


# --- on the real graph, where it is on disk ---------------------------------


def test_gpt2_block_cache_matches_contiguous_kv_bitwise(gpt2_graph):
    """The same equality on a real decoder, at real KV geometry.

    The synthetic graph has three layers of two heads; GPT-2 has twelve of twelve,
    at 72 KiB of cache per token. A gather whose per-layer or per-head stride was
    subtly wrong could survive the small geometry and fail here.
    """
    extension = load_extension()
    session = extension.DecoderSession(str(gpt2_graph), block_tokens=64, num_blocks=16)
    geometry = session.geometry
    assert (geometry.layers, geometry.kv_heads, geometry.head_dim) == (12, 12, 64)
    assert geometry.bytes_per_token == 72 * 1024

    prompt = list(range(100, 148))
    engine = extension.Engine([("m", str(gpt2_graph))])
    expected_tokens, expected_logits = reference_generate(
        engine_runner(engine, "m"),
        prompt,
        6,
        chunk_tokens=0,
        layers=geometry.layers,
        kv_heads=geometry.kv_heads,
        head_dim=geometry.head_dim,
    )
    tokens, logits = session_generate(session, "a", prompt, 6, chunk_tokens=0)
    assert tokens == expected_tokens
    _assert_bitwise(expected_logits, logits, "gpt2 contiguous reference")


def test_gpt2_preempted_sequence_emits_token_identical_output(gpt2_graph):
    """Preempt-and-recompute on a real decoder.

    Recorded rather than smoothed over: on GPT-2 the recomputed logits differ from
    the uninterrupted ones by around 6e-05, because one wide pass sums in a different
    order than many narrow ones. Token identity survives that; bitwise equality does
    not, and asserting it would fail for a legitimate reason.
    """
    extension = load_extension()
    prompt = list(range(200, 248))

    clean_tokens, clean_logits = session_generate(
        extension.DecoderSession(str(gpt2_graph), block_tokens=64, num_blocks=16),
        "a",
        prompt,
        8,
        chunk_tokens=0,
    )
    preempted_tokens, preempted_logits = session_generate(
        extension.DecoderSession(str(gpt2_graph), block_tokens=64, num_blocks=16),
        "a",
        prompt,
        8,
        chunk_tokens=0,
        recompute_at=4,
    )

    drift = max(
        float(np.max(np.abs(a - b))) for a, b in zip(clean_logits, preempted_logits, strict=True)
    )
    margin = min(top_two_margin(row) for row in clean_logits)
    assert margin > 100.0 * max(drift, 1e-9), (
        f"winning logit led by {margin:.3e} while recompute moved the logits by "
        f"{drift:.3e}; token identity would be luck rather than a result"
    )
    assert preempted_tokens == clean_tokens
