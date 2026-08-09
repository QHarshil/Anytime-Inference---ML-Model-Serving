"""Guards on `scripts/profile_batching.py`: the arithmetic, and one real schedule.

Two kinds of test here, and they fail for different reasons.

The arithmetic tests cover the functions that turn timings into claims -- the speedup
against one-at-a-time, the fitted split the speedup is compared against, and the length
regimes the padding cost is subtracted between. None of these would crash if they were
wrong; they would produce a plausible number, which is the shape of mistake this
project has already been burned by.

The rest drive the real scheduler over the synthetic decoder graph. They are slow
relative to the arithmetic and they are worth it: the measurement functions assemble a
batch by running prefill ahead of decode and then assume every following iteration is a
decode step over the whole batch. That assumption is about the scheduler's behaviour
rather than about arithmetic, so it is tested against the scheduler.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from profile_batching import (  # noqa: E402
    TRACE_PROMPTS,
    ScalingPoint,
    StepSplit,
    _assemble,
    _measure_steps,
    apply_scaling_derivations,
    blocks_for_sweep,
    feasible,
    fit_step_split,
    host_metadata,
    measure_scaling_point,
    padding_regimes,
    record_alternation,
    tokens_disagree,
)
from profile_decode import Spread  # noqa: E402

from anytime_serving.serving.decoder import DecoderClient  # noqa: E402
from anytime_serving.serving.onnx_runtime import extension_available  # noqa: E402

requires_extension = pytest.mark.skipif(
    not extension_available(),
    reason="anytime_runtime is not built; batched decoding lives in it. pip install -e .",
)

# The synthetic fixture's geometry is tiny, so every length here is too. Block width 4
# rather than the runtime default of 64 for the same reason: a sequence has to span
# several blocks for the arena to be doing anything.
BLOCK_TOKENS = 4
FIXTURE_VOCAB = 64
FIXTURE_CONTEXT = 64


def _point(batch_size: int, cached_tokens: int, step_ms: float) -> ScalingPoint:
    return ScalingPoint(
        batch_size=batch_size,
        cached_tokens=cached_tokens,
        cached_min=cached_tokens,
        cached_max=cached_tokens,
        step=Spread.of([step_ms]),
        tokens_per_s=batch_size / step_ms * 1000.0,
        pad_p50_ms=0.0,
        gather_p50_ms=0.0,
        run_p50_ms=step_ms,
        scatter_p50_ms=0.0,
        scheduler_overhead_p50_ms=0.0,
        steps_per_pass=16,
    )


def _client(graph, *, num_blocks=64, max_context=FIXTURE_CONTEXT):
    return DecoderClient(
        graph,
        block_tokens=BLOCK_TOKENS,
        num_blocks=num_blocks,
        max_context_tokens=max_context,
    )


# --- the speedup and what it is compared against -----------------------------


def test_the_speedup_is_against_the_same_sequences_stepped_one_at_a_time():
    """B sequences in one step, against B steps of one sequence. Nothing else.

    A batched step serving 4 sequences in 12 ms is worth 4 x 4.64 ms of serial
    stepping, so the speedup is 1.55x. Dividing the step by the batch instead would
    report 3.0 ms a token and call it a 1.55x latency win, which no sequence
    experienced -- each of the four waited the full 12 ms.
    """
    scaling = [_point(1, 128, 4.64), _point(4, 128, 12.0), _point(1, 512, 6.82)]
    split = fit_step_split(scaling)
    apply_scaling_derivations(scaling, split)

    batched = scaling[1]
    assert batched.speedup_vs_serial == pytest.approx(4 * 4.64 / 12.0, abs=1e-4)
    assert batched.step.p50_ms == 12.0


def test_a_batch_slower_than_serial_stepping_reports_below_one():
    """The measurement has to be able to say batching lost, because it can.

    Nothing in the arithmetic clamps this at 1.0. A batched step that costs more than
    the serial steps it replaces is a real outcome on a host whose batch-1 path hits a
    different kernel, and a floor of 1.0 would hide it.
    """
    scaling = [_point(1, 128, 4.64), _point(2, 128, 9.83), _point(1, 512, 6.82)]
    apply_scaling_derivations(scaling, fit_step_split(scaling))
    assert scaling[1].speedup_vs_serial < 1.0


def test_the_split_is_fitted_from_batch_one_only():
    """Fitting from the batched points would make the prediction unfalsifiable.

    The batched points are what the split is used to predict. These batch-4 timings are
    deliberately nothing like the batch-1 line, and the fit has to ignore them.
    """
    scaling = [
        _point(1, 128, 4.0 + 0.005 * 128),
        _point(1, 512, 4.0 + 0.005 * 512),
        _point(4, 128, 99.0),
        _point(4, 512, 99.0),
    ]
    split = fit_step_split(scaling)
    assert split.base_ms == pytest.approx(4.0, abs=1e-3)
    assert split.per_token_ms == pytest.approx(0.005, abs=1e-5)
    assert split.fitted_from_points == 2


def test_the_prediction_is_withheld_when_one_length_cannot_separate_the_terms():
    """A single cached length fits a slope of zero and would predict perfect scaling.

    That is an artefact of the sweep, not a prediction, and printing it as one would
    make every point look like a failure of the cost model.
    """
    scaling = [_point(1, 128, 4.64), _point(4, 128, 12.0)]
    split = fit_step_split(scaling)
    assert split.fitted_from_points == 1
    apply_scaling_derivations(scaling, split)
    assert scaling[1].speedup_vs_serial > 0.0
    assert scaling[1].predicted_speedup == 0.0
    assert scaling[1].prediction_ratio == 0.0


def test_only_the_cache_independent_term_amortises():
    """The prediction the whole batching story rests on, in one assertion.

    With no cache to read, a batch of 8 costs what one step costs, so the speedup is
    8x. As the cache fills, the per-sequence term dominates and the speedup decays
    towards 1. If this ever stops holding, the sentence in the docs about why batching
    pays less at long context is no longer derivable from the model.
    """
    split = StepSplit(base_ms=4.0, per_token_ms=0.005, max_residual_ms=0.0, fitted_from_points=3)
    assert split.predicted_speedup(8, 0) == pytest.approx(8.0)
    assert split.predicted_speedup(1, 960) == pytest.approx(1.0)
    decaying = [split.predicted_speedup(8, length) for length in (128, 512, 960)]
    assert decaying == sorted(decaying, reverse=True)
    assert decaying[-1] < 2.0


def test_a_point_with_no_serial_reference_is_left_alone():
    """A cached length measured only at batch 4 has nothing to divide by.

    Defaulting it to 1.0x would put a made-up speedup in the results next to measured
    ones, indistinguishable from a batch that genuinely broke even.
    """
    scaling = [_point(1, 128, 4.64), _point(1, 512, 6.82), _point(4, 960, 20.0)]
    apply_scaling_derivations(scaling, fit_step_split(scaling))
    orphan = scaling[-1]
    assert orphan.speedup_vs_serial == 1.0
    assert orphan.tokens_per_s > 0.0


# --- the length regimes the padding cost is subtracted between ---------------


def test_the_spread_and_the_uniform_mean_regime_share_a_mean():
    """The subtraction is only about variance if the two regimes agree on the mean.

    `spread` minus `uniform-mean` is reported as what length variance costs. If the
    spread's mean drifted from the uniform regime's length, that difference would
    carry a change in total cache traffic as well, and the number would be a
    combination of the two with no way to separate them.
    """
    regimes = padding_regimes(longest=960, batch_size=8, floor=0.25)
    spread_mean = sum(regimes["spread"]) / len(regimes["spread"])
    assert regimes["uniform-mean"][0] == round(spread_mean)
    assert len(set(regimes["uniform-mean"])) == 1
    assert set(regimes["uniform-max"]) == {960}
    assert min(regimes["spread"]) == 240
    assert max(regimes["spread"]) == 960


def test_the_spread_runs_at_the_same_width_as_uniform_max():
    """No row of the spread is longer than the uniform-max row, and one of them equals it.

    That is what makes uniform-max the right reference: right-padding runs every row of
    a batch at the batch's longest past, so the spread and uniform-max regimes hand the
    graph the same shape and differ only in how much of it is padding.

    A stronger bracket is tempting and false. The spread's rows are not all above its
    own mean -- half of them are below it by construction -- and its measured *step* is
    not bounded above by uniform-max's either, because the padding it has to clear is
    work uniform-max does not do. The two facts asserted here are the ones the
    subtraction actually rests on.
    """
    regimes = padding_regimes(longest=512, batch_size=4, floor=0.5)
    assert max(regimes["spread"]) == max(regimes["uniform-max"]) == 512
    assert min(regimes["spread"]) == 256
    assert max(regimes["uniform-mean"]) <= max(regimes["spread"])


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"batch_size": 1, "floor": 0.25}, "two rows"),
        ({"batch_size": 8, "floor": 0.0}, "fraction"),
        ({"batch_size": 8, "floor": 1.5}, "fraction"),
    ],
)
def test_a_regime_that_cannot_have_a_spread_is_refused(kwargs, message):
    with pytest.raises(ValueError, match=message):
        padding_regimes(longest=512, **kwargs)


# --- what the sweep is allowed to ask for ------------------------------------


def test_the_arena_is_sized_for_the_traces_as_well_as_the_sweep():
    """The traces hold every sequence at once and their prompts are their own.

    Sizing from the scaling sweep alone left the traces admission-limited, which made
    the alternation figure a picture of a full arena rather than of alternation.
    """
    blocks = blocks_for_sweep(
        batch_sizes=(1, 2),
        cached_lengths=(128,),
        steps=4,
        block_tokens=64,
        max_context=1024,
        padding_batch=8,
    )
    per_trace_sequence = -(-max(TRACE_PROMPTS) // 64)
    assert blocks >= per_trace_sequence * len(TRACE_PROMPTS)


def test_the_arena_grows_with_the_widest_batch():
    narrow = blocks_for_sweep(
        batch_sizes=(1, 4),
        cached_lengths=(960,),
        steps=16,
        block_tokens=64,
        max_context=1024,
        padding_batch=8,
    )
    wide = blocks_for_sweep(
        batch_sizes=(1, 32),
        cached_lengths=(960,),
        steps=16,
        block_tokens=64,
        max_context=1024,
        padding_batch=8,
    )
    assert wide > narrow


def test_a_point_past_the_position_table_is_not_attempted():
    """Assembling a wide batch spends tokens before the measurement starts.

    GPT-2 stops at 1024 positions and exceeding it is an out-of-bounds Gather from
    inside ONNX Runtime, so the point is skipped with a reason rather than run.
    """
    assert feasible(960, 32, 16, 1024)
    assert not feasible(960, 64, 16, 1024)
    assert not feasible(1000, 32, 16, 1024)


# --- the token-agreement guard ----------------------------------------------


def test_the_guard_compares_the_prefix_both_runs_produced():
    """Different batch widths spend different token budgets assembling themselves.

    So the comparison is over the shorter run. A divergence at the first batched step
    still shows up, which is what the guard is for.
    """
    assert not tokens_disagree([1, 2, 3], [1, 2, 3, 4, 5])
    assert not tokens_disagree([1, 2, 3, 4, 5], [1, 2, 3])
    assert tokens_disagree([1, 2, 3], [1, 9, 3])
    assert tokens_disagree([1, 2, 3, 4], [9])
    # Nothing to compare is not a disagreement; a run that emitted nothing is caught
    # by the step count instead.
    assert not tokens_disagree([], [1, 2, 3])


@requires_extension
@pytest.mark.parametrize("threads", [1, 4])
def test_the_recorded_host_reports_the_thread_count_the_run_used(threads):
    """A settings field written as a constant describes a run that may not have happened.

    This one was literally `"intra_op_num_threads": 1` while the count was fixed at 1,
    and it stayed correct only for as long as nothing could change it. Reading it from
    the run is what keeps the artefact a record rather than an assumption.
    """
    assert host_metadata(threads)["intra_op_num_threads"] == threads


@requires_extension
@pytest.mark.parametrize("copy_threads", [1, 4])
def test_the_recorded_host_reports_the_copy_thread_count_too(copy_threads):
    """It moves `gather_p50_ms`, which is a reported number, so it has to be recorded.

    Same species as the field above and added at the same time as the setting, rather
    than after an artefact had already been written describing the wrong one.
    """
    assert host_metadata(8, copy_threads)["copy_threads"] == copy_threads
    # Defaulted rather than required, so an older caller records the serial copy it
    # actually ran instead of nothing at all.
    assert host_metadata(8)["copy_threads"] == 1


@requires_extension
@pytest.mark.parametrize("allow_spinning", [True, False])
def test_the_recorded_host_reports_whether_the_pool_was_spinning(allow_spinning):
    """It changes when cores are free, so it changes the gather beside the graph."""
    assert host_metadata(8, 1, allow_spinning)["allow_spinning"] is allow_spinning
    assert host_metadata(8)["allow_spinning"] is True


# --- against the real scheduler ----------------------------------------------


@requires_extension
def test_a_measured_point_is_a_batched_decode_step_of_the_width_it_claims(decoder_graph):
    """The assembly assumption, checked against the scheduler rather than asserted.

    `_measure_steps` refuses anything that is not a decode step over the whole batch,
    so this passing means prefill really did run ahead and the batch really was
    assembled. It also exercises the arena bookkeeping: the point releases every
    sequence and fails if a block is left held.
    """
    with _client(decoder_graph) as client:
        point, emitted = measure_scaling_point(
            client,
            "fixture",
            batch_size=3,
            cached_tokens=8,
            vocab=FIXTURE_VOCAB,
            repeats=1,
            steps=2,
        )
    assert point.batch_size == 3
    assert point.cached_tokens == 8
    assert point.step.p50_ms > 0.0
    assert point.tokens_per_s > 0.0
    # The wall clock around a call cannot be shorter than the call says it took. This
    # read negative at the widest batch when the overhead was the difference of two
    # differently-pooled medians rather than a per-step difference.
    assert point.scheduler_overhead_p50_ms >= 0.0
    # Assembling three sequences costs the first one three tokens, then two more are
    # measured, so it has emitted at least five.
    assert len(emitted) >= 5
    # The batch carries the spread its own assembly created, which is why pad_ms is
    # non-zero at every point in this measurement.
    assert point.cached_max >= point.cached_min


@requires_extension
def test_batching_does_not_change_what_a_sequence_emits(decoder_graph):
    """The guard the whole measurement is refused on, proved to hold on a real graph.

    Same prompt, once alone and once in a batch of four. Greedy decoding depends only
    on a sequence's own history, so the tokens have to match.
    """
    with _client(decoder_graph) as client:
        _, alone = measure_scaling_point(
            client,
            "fixture",
            batch_size=1,
            cached_tokens=8,
            vocab=FIXTURE_VOCAB,
            repeats=1,
            steps=3,
        )
        _, batched = measure_scaling_point(
            client,
            "fixture",
            batch_size=4,
            cached_tokens=8,
            vocab=FIXTURE_VOCAB,
            repeats=1,
            steps=3,
        )
    assert alone
    assert not tokens_disagree(alone, batched)


@requires_extension
def test_a_step_of_the_wrong_width_fails_the_measurement(decoder_graph):
    """A measurement that silently timed the wrong thing is worse than one that stops.

    The guard is what makes the assembly assumption load-bearing rather than hopeful,
    so it is worth proving it fires.
    """
    with _client(decoder_graph) as client:
        scheduler, _ = _assemble(
            client,
            prompts=[[1, 2, 3, 4], [5, 6, 7, 8]],
            max_new_tokens=8,
            label="fixture-mismatch",
            max_batch_size=2,
        )
        with pytest.raises(SystemExit, match="decode step over 3"):
            _measure_steps(scheduler, expect_batch=3, steps=1)


@requires_extension
def test_a_batch_the_arena_cannot_hold_is_a_failure_with_a_reason(decoder_graph):
    """Silently measuring a narrower batch than asked for would be the bad outcome.

    Two sequences of 32 tokens need 16 blocks between them at this width, and the
    arena is given 4, so the second never becomes resident.
    """
    with _client(decoder_graph, num_blocks=4) as client:
        with pytest.raises(SystemExit, match="reached the decode phase"):
            _assemble(
                client,
                prompts=[list(range(1, 33)), list(range(1, 33))],
                max_new_tokens=4,
                label="fixture-starved",
                max_batch_size=2,
            )


@requires_extension
def test_a_trace_records_both_kinds_of_step_and_the_gaps_between_tokens(decoder_graph):
    """The alternation trace, on a schedule with more requests than batch width.

    Chunked at 4 tokens so a 12-token prompt takes three chunks, which is what makes
    the stall the figure is about visible at all.
    """
    trace = record_alternation(
        _client(decoder_graph, num_blocks=64),
        "fixture",
        prompt_tokens=[12, 8, 16],
        max_new_tokens=4,
        max_batch_size=2,
        prefill_chunks_per_decode=1,
        vocab=FIXTURE_VOCAB,
        chunk_tokens=4,
    )
    assert trace.prefill_steps > len(trace.prompt_tokens)
    assert trace.decode_steps > 0
    assert trace.wall_ms > 0.0
    assert [step.index for step in trace.steps] == list(range(len(trace.steps)))
    assert {step.kind for step in trace.steps} == {"prefill", "decode"}
    # Every sequence emitted four tokens, so every one of them has gaps to report.
    assert trace.stalled_sequences == len(trace.prompt_tokens)
    assert trace.max_decode_gap_ms >= trace.decode_gap_p50_ms > 0.0


@requires_extension
def test_a_trace_with_one_token_each_reports_no_gap_rather_than_inventing_one(decoder_graph):
    """A gap needs two tokens to exist between.

    Worth pinning: an unconditional summary would report the interval between two
    different sequences' tokens as a stall, which is not what the number means.
    """
    trace = record_alternation(
        _client(decoder_graph, num_blocks=64),
        "fixture",
        prompt_tokens=[8, 8],
        max_new_tokens=1,
        max_batch_size=2,
        prefill_chunks_per_decode=1,
        vocab=FIXTURE_VOCAB,
        chunk_tokens=4,
    )
    assert trace.decode_steps > 0
    assert trace.stalled_sequences == 0
    assert trace.max_decode_gap_ms == 0.0
    assert trace.decode_gap_p50_ms == 0.0
