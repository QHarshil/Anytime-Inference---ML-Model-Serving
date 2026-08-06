"""Guards on `scripts/run_decode_sweep.py`, the open-loop load sweep.

The property everything else depends on is that arrivals are open loop: a request
enters when its time comes, whether or not anything finished. A driver that ignored
arrival times would drain a backlog as fast as it could and every latency in the
results would be a makespan measurement wearing a queueing label -- and it would look
entirely plausible, because the numbers would still be ordered the way one expects.
`test_the_driver_waits_for_an_arrival_rather_than_draining_a_backlog` is the one that
pins it.

The rest cover the arithmetic that turns per-request timings into a claim: which
requests a percentile is taken over, what the attainment denominator is, and that time
to first token is measured from arrival rather than from the start of the run.
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from run_decode_sweep import (  # noqa: E402
    PolicySpec,
    RequestOutcome,
    WorkloadSpec,
    _cost_model,
    _percentile,
    blocks_for,
    build_requests,
    drive,
    host_metadata,
    meets_slo,
    summarise,
)

from anytime_serving.serving.batch_scheduler import ContinuousBatchScheduler  # noqa: E402
from anytime_serving.serving.decoder import DecoderClient  # noqa: E402
from anytime_serving.serving.onnx_runtime import extension_available  # noqa: E402

requires_extension = pytest.mark.skipif(
    not extension_available(),
    reason="anytime_runtime is not built; batched decoding lives in it. pip install -e .",
)

BLOCK_TOKENS = 4
FIXTURE_VOCAB = 64
FIXTURE_CONTEXT = 64


@requires_extension
@pytest.mark.parametrize("threads", [1, 4])
def test_the_recorded_host_reports_the_thread_count_the_run_used(threads):
    """Capacity is measured under this setting, so the sweep is only readable with it.

    Every load figure here is a fraction of measured capacity. Capacity measured with
    one thread and reported beside a metadata block claiming another would make the
    whole sweep describe a machine it never ran on.
    """
    assert host_metadata(threads)["intra_op_num_threads"] == threads


def _outcome(
    *,
    ttft_ms: float = 100.0,
    tpot_ms: float = 10.0,
    completed: bool = True,
    output_tokens: int = 8,
    e2e_ms: float = 200.0,
) -> RequestOutcome:
    return RequestOutcome(
        policy="batched-8",
        target_utilisation=0.95,
        workload="fixed",
        request_id="r0",
        prompt_tokens=256,
        max_new_tokens=8,
        arrival_ms=0.0,
        ttft_ms=ttft_ms,
        tpot_p50_ms=tpot_ms,
        e2e_ms=e2e_ms,
        normalised_ms_per_token=e2e_ms / max(output_tokens, 1),
        output_tokens=output_tokens,
        preemptions=0,
        completed=completed,
        rejection_reason="",
    )


def _workload(requests: int = 4) -> WorkloadSpec:
    return WorkloadSpec(label="fixed", prompt_tokens=256, max_new_tokens=8, requests=requests)


def _drive_result(outcomes, *, makespan_s: float = 1.0, arrival_window_s: float = 1.0):
    from run_decode_sweep import DriveResult

    return DriveResult(
        outcomes=list(outcomes),
        makespan_s=makespan_s,
        arrival_window_s=arrival_window_s,
        prefill_steps=4,
        decode_steps=8,
        mean_decode_batch=2.0,
        preemptions=0,
        max_waiting=2,
        max_resident=4,
    )


# --- the workload -----------------------------------------------------------


def test_a_spread_workload_keeps_the_mean_it_claims():
    """The variance regime is only about variance if it matches the fixed one's mean.

    Its whole purpose is to be compared against a fixed-length workload of the same
    mean, so that the difference is the length spread and not the amount of work.
    """
    lengths = (64, 192, 320, 448)
    spread = WorkloadSpec(
        label="variance", prompt_tokens=256, max_new_tokens=64, requests=8, spread=lengths
    )
    assert spread.mean_prompt_tokens == pytest.approx(256.0)
    assert spread.longest_prompt_tokens == 448
    # Cycled rather than sampled, so the mean is exact at any request count that is a
    # multiple of the number of lengths.
    assert spread.lengths()[:4] == list(lengths)


def test_a_fixed_workload_is_every_request_the_same_length():
    fixed = _workload(requests=3)
    assert fixed.lengths() == [256, 256, 256]
    assert fixed.mean_prompt_tokens == 256.0


def test_the_arena_holds_whole_sequences_rounded_up():
    """A sequence needing 260 tokens at 64 a block occupies five, not 4.06."""
    assert blocks_for(sequences=1, tokens_each=260, block_tokens=64) == 5
    assert blocks_for(sequences=4, tokens_each=256, block_tokens=64) == 16
    assert blocks_for(sequences=8, tokens_each=1, block_tokens=64) == 8


def test_requests_carry_distinct_prompts():
    """Identical rows would hide a gather fault that mixed one row into another."""
    requests = build_requests(_workload(requests=3), vocab=64, deadline_ms=1000.0, seed=7)
    prompts = {tuple(request.prompt) for request in requests}
    assert len(prompts) == 3
    assert {request.prompt_tokens for request in requests} == {256}


# --- the SLO ----------------------------------------------------------------


def test_both_halves_of_the_slo_are_required():
    """A prompt reply that then trickles has not been served, and neither has the reverse."""
    assert meets_slo(_outcome(ttft_ms=100, tpot_ms=10), ttft_slo_ms=500, tpot_slo_ms=50)
    assert not meets_slo(_outcome(ttft_ms=900, tpot_ms=10), ttft_slo_ms=500, tpot_slo_ms=50)
    assert not meets_slo(_outcome(ttft_ms=100, tpot_ms=80), ttft_slo_ms=500, tpot_slo_ms=50)


def test_the_target_is_inclusive_at_the_boundary():
    """A request exactly on target met it. Nothing hinges on it, but it should not drift."""
    assert meets_slo(_outcome(ttft_ms=500, tpot_ms=50), ttft_slo_ms=500, tpot_slo_ms=50)


def test_an_unfinished_request_cannot_meet_the_slo():
    """Otherwise a request that emitted one fast token and stalled would count as served."""
    assert not meets_slo(
        _outcome(ttft_ms=10, tpot_ms=1, completed=False), ttft_slo_ms=500, tpot_slo_ms=50
    )


# --- what the aggregate is over ----------------------------------------------


def test_percentiles_are_over_completed_requests_and_attainment_is_over_all_of_them():
    """An unserved request must not improve the tail by being counted as zero.

    It has no time per output token to contribute, so it is excluded from the
    percentiles -- and included in the attainment denominator, which is where traffic
    that suffered most belongs. Getting this backwards would make a policy that drops
    half its load look like the best one.
    """
    outcomes = [
        _outcome(ttft_ms=100, tpot_ms=10),
        _outcome(ttft_ms=200, tpot_ms=20),
        _outcome(ttft_ms=0, tpot_ms=0, completed=False, output_tokens=1),
    ]
    row = summarise(
        _drive_result(outcomes),
        policy="batched-8",
        workload=_workload(requests=3),
        utilisation=0.95,
        offered_rps=2.0,
        ttft_slo_ms=500,
        tpot_slo_ms=50,
    )
    assert row.completed == 2
    assert row.ttft_p50_ms == pytest.approx(150.0)
    assert row.tpot_p50_ms == pytest.approx(15.0)
    # Two of three met both targets, and the third counts against it. The recorded
    # value is rounded to four places for the JSON, hence the tolerance.
    assert row.slo_attainment == pytest.approx(2 / 3, abs=1e-4)


def test_goodput_counts_only_requests_that_met_the_slo():
    """Throughput and goodput are different numbers and the sweep reports both."""
    outcomes = [
        _outcome(ttft_ms=100, tpot_ms=10),
        _outcome(ttft_ms=9000, tpot_ms=10),
    ]
    row = summarise(
        _drive_result(outcomes, makespan_s=2.0),
        policy="batched-8",
        workload=_workload(requests=2),
        utilisation=0.95,
        offered_rps=1.0,
        ttft_slo_ms=500,
        tpot_slo_ms=50,
    )
    assert row.achieved_rps == pytest.approx(1.0)
    assert row.goodput_rps == pytest.approx(0.5)
    assert row.goodput_rps < row.achieved_rps


def test_a_point_where_nothing_completed_reports_zeros_rather_than_failing():
    """A saturated policy is a result, not a crash, and it has to be recordable."""
    outcomes = [_outcome(completed=False, output_tokens=0, e2e_ms=0.0)]
    row = summarise(
        _drive_result(outcomes),
        policy="serial",
        workload=_workload(requests=1),
        utilisation=1.3,
        offered_rps=3.0,
        ttft_slo_ms=500,
        tpot_slo_ms=50,
    )
    assert row.completed == 0
    assert row.ttft_p50_ms == 0.0
    assert row.slo_attainment == 0.0
    assert row.goodput_rps == 0.0


def test_an_empty_percentile_is_zero_rather_than_an_error():
    assert _percentile([], 95) == 0.0
    assert _percentile([4.0], 95) == 4.0


# --- the cost model the eviction policy needs --------------------------------


def test_the_cost_model_is_read_from_the_measurement(tmp_path):
    path = tmp_path / "decode_profiles.json"
    path.write_text(
        json.dumps(
            {
                "precisions": [
                    {
                        "precision": "fp32",
                        "cache_cost": {
                            "decode_base_ms": 4.18,
                            "decode_per_token_ms": 0.00565,
                            "prefill_per_token_ms": 0.364,
                        },
                    }
                ]
            }
        )
    )
    cost = _cost_model(path, "fp32")
    assert cost.decode_ms(960) == pytest.approx(4.18 + 0.00565 * 960)


def test_a_missing_cost_model_names_the_script_that_produces_it(tmp_path):
    """Inventing coefficients would give the eviction policy a cost model of nothing."""
    with pytest.raises(SystemExit, match="profile_decode"):
        _cost_model(tmp_path / "absent.json", "fp32")


def test_a_cost_model_for_another_precision_is_not_substituted(tmp_path):
    """INT4's constant term is 2.6x FP32's; swapping them would misprice every eviction."""
    path = tmp_path / "decode_profiles.json"
    path.write_text(
        json.dumps(
            {
                "precisions": [
                    {
                        "precision": "fp32",
                        "cache_cost": {
                            "decode_base_ms": 4.18,
                            "decode_per_token_ms": 0.00565,
                            "prefill_per_token_ms": 0.364,
                        },
                    }
                ]
            }
        )
    )
    with pytest.raises(SystemExit, match="no fitted cost model for int4"):
        _cost_model(path, "int4")


# --- the driver -------------------------------------------------------------


@requires_extension
def test_the_driver_waits_for_an_arrival_rather_than_draining_a_backlog(decoder_graph):
    """Open loop, which is the property every latency in the results depends on.

    Two requests, the second arriving 400 ms in. A driver that ignored arrival times
    would finish in whatever the work takes -- a few milliseconds on this graph -- and
    report a queueing latency it never measured. The run cannot finish before the last
    arrival.
    """
    with DecoderClient(
        decoder_graph,
        block_tokens=BLOCK_TOKENS,
        num_blocks=32,
        max_context_tokens=FIXTURE_CONTEXT,
    ) as client:
        scheduler = ContinuousBatchScheduler(client, max_batch_size=2)
        workload = WorkloadSpec(label="fixed", prompt_tokens=8, max_new_tokens=2, requests=2)
        requests = build_requests(workload, vocab=FIXTURE_VOCAB, deadline_ms=1e6, seed=3)
        result = drive(
            scheduler,
            requests,
            [0.0, 0.4],
            policy="batched-2",
            utilisation=0.5,
            workload="fixed",
        )
    assert result.makespan_s >= 0.4
    assert len(result.outcomes) == 2
    assert all(outcome.completed for outcome in result.outcomes)


@requires_extension
def test_a_point_that_overruns_its_budget_fails_rather_than_running_on(decoder_graph):
    """A point's length depends on a measured capacity, so it can grow without bound.

    Duration is `requests / (rho * capacity)` and capacity is measured, not assumed.
    A capacity that comes out far below the truth stretches the arrival window with
    nothing to notice: one run took 9h34m against a 28-minute predecessor before it
    was killed by hand. The budget turns that into a failure with a diagnostic.

    Failing rather than truncating is deliberate. A truncated point is a latency
    distribution missing its slowest requests, which is the half a scheduler is
    judged on, so a short point would read *better* than an honest one.
    """
    with DecoderClient(
        decoder_graph,
        block_tokens=BLOCK_TOKENS,
        num_blocks=32,
        max_context_tokens=FIXTURE_CONTEXT,
    ) as client:
        scheduler = ContinuousBatchScheduler(client, max_batch_size=2)
        workload = WorkloadSpec(label="fixed", prompt_tokens=8, max_new_tokens=2, requests=2)
        with pytest.raises(SystemExit, match="exceeded its"):
            drive(
                scheduler,
                build_requests(workload, vocab=FIXTURE_VOCAB, deadline_ms=1e6, seed=7),
                # The second request arrives well past a budget of a tenth of a second.
                [0.0, 5.0],
                policy="batched-2",
                utilisation=0.5,
                workload="fixed",
                budget_s=0.1,
            )


@requires_extension
def test_a_point_inside_its_budget_is_untouched_by_the_guard(decoder_graph):
    """The guard must not truncate an honest point, which is the failure that would hide.

    Same shape as the overrun case with a budget the run comfortably meets, so a guard
    that fired on elapsed time regardless would show up here rather than as quietly
    missing requests in a sweep.
    """
    with DecoderClient(
        decoder_graph,
        block_tokens=BLOCK_TOKENS,
        num_blocks=32,
        max_context_tokens=FIXTURE_CONTEXT,
    ) as client:
        scheduler = ContinuousBatchScheduler(client, max_batch_size=2)
        workload = WorkloadSpec(label="fixed", prompt_tokens=8, max_new_tokens=2, requests=2)
        result = drive(
            scheduler,
            build_requests(workload, vocab=FIXTURE_VOCAB, deadline_ms=1e6, seed=7),
            [0.0, 0.2],
            policy="batched-2",
            utilisation=0.5,
            workload="fixed",
            budget_s=60.0,
        )
    assert len(result.outcomes) == 2
    assert all(outcome.completed for outcome in result.outcomes)


@requires_extension
def test_time_to_first_token_is_measured_from_arrival(decoder_graph):
    """Not from the start of the run, which is a different number for a late arrival.

    The second request arrives 400 ms in and is served straight away, so its TTFT is a
    few milliseconds. Measured from the origin it would read over 400, and the sweep
    would report queueing that never happened.
    """
    with DecoderClient(
        decoder_graph,
        block_tokens=BLOCK_TOKENS,
        num_blocks=32,
        max_context_tokens=FIXTURE_CONTEXT,
    ) as client:
        scheduler = ContinuousBatchScheduler(client, max_batch_size=2)
        workload = WorkloadSpec(label="fixed", prompt_tokens=8, max_new_tokens=2, requests=2)
        result = drive(
            scheduler,
            build_requests(workload, vocab=FIXTURE_VOCAB, deadline_ms=1e6, seed=5),
            [0.0, 0.4],
            policy="batched-2",
            utilisation=0.5,
            workload="fixed",
        )
    late = result.outcomes[1]
    assert late.arrival_ms == pytest.approx(400.0, abs=1.0)
    assert late.ttft_ms < 100.0
    assert late.e2e_ms < 100.0


@requires_extension
def test_a_serial_policy_queues_what_a_batched_one_overlaps(decoder_graph):
    """The comparison the sweep exists to make, at the smallest scale that shows it.

    Four requests arriving together. With a batch width of one and an arena for one
    sequence, the fourth waits for the first three; batched, they overlap. This is
    the mechanism behind the time-to-first-token difference in the results, checked
    here on a graph that runs in microseconds so it cannot be a timing artefact.
    """
    workload = WorkloadSpec(label="fixed", prompt_tokens=8, max_new_tokens=3, requests=4)
    arrivals = [0.0, 0.0, 0.0, 0.0]
    waits = {}
    for policy in (
        PolicySpec("serial", max_batch_size=1, resident_capacity=1, use_admission=False),
        PolicySpec("batched-4", max_batch_size=4, resident_capacity=4, use_admission=False),
    ):
        with DecoderClient(
            decoder_graph,
            block_tokens=BLOCK_TOKENS,
            num_blocks=blocks_for(
                sequences=policy.resident_capacity, tokens_each=11, block_tokens=BLOCK_TOKENS
            ),
            max_context_tokens=FIXTURE_CONTEXT,
        ) as client:
            scheduler = ContinuousBatchScheduler(client, max_batch_size=policy.max_batch_size)
            result = drive(
                scheduler,
                build_requests(workload, vocab=FIXTURE_VOCAB, deadline_ms=1e6, seed=9),
                arrivals,
                policy=policy.name,
                utilisation=1.0,
                workload="fixed",
            )
        assert all(outcome.completed for outcome in result.outcomes)
        waits[policy.name] = max(outcome.ttft_ms for outcome in result.outcomes)
        if policy.max_batch_size > 1:
            assert result.mean_decode_batch > 1.0
        else:
            assert result.mean_decode_batch == 1.0
    assert waits["batched-4"] < waits["serial"]
