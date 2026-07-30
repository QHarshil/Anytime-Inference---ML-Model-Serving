"""Guards on the two helpers in `scripts/profile_decode.py` that produce numbers.

`Spread` is what stops a single-pass latency being reported as a result, and
`fit_cache_cost` produces the coefficients the admission policy makes eviction
decisions with. Both are arithmetic over measurements, which is the shape of mistake
this project has already been burned by: a wrong fit would not fail, it would produce
a policy that evicts the wrong sequences and looks like it is working.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from profile_decode import (  # noqa: E402
    CacheCostFit,
    DecodeMeasurement,
    PrefillMeasurement,
    Spread,
    fit_cache_cost,
)

from anytime_serving.serving.kv_admission import CacheCost  # noqa: E402


def _decode(cached_tokens: float, tpot_ms: float) -> DecodeMeasurement:
    return DecodeMeasurement(
        precision="fp32",
        cached_tokens=int(cached_tokens),
        tpot=Spread.of([tpot_ms]),
        gather_p50_ms=0.0,
        run_p50_ms=0.0,
        scatter_p50_ms=0.0,
        arena_cost_pct=0.0,
        contiguous_run=Spread.of([tpot_ms]),
        graph_run_agreement=1.0,
        steps_per_pass=16,
        cache_mb=0.0,
    )


def _prefill(prompt_tokens: int, ttft_ms: float, *, chunk_tokens: int = 0) -> PrefillMeasurement:
    return PrefillMeasurement(
        precision="fp32",
        prompt_tokens=prompt_tokens,
        chunk_tokens=chunk_tokens,
        graph_runs=1,
        ttft=Spread.of([ttft_ms]),
        gather_p50_ms=0.0,
        run_p50_ms=0.0,
        scatter_p50_ms=0.0,
        ms_per_prompt_token=ttft_ms / prompt_tokens,
        peak_logits_mb=0.0,
    )


# --- the spread -------------------------------------------------------------


def test_spread_reports_the_median_and_the_range():
    """The range is the honest uncertainty, so it travels with the median."""
    spread = Spread.of([10.0, 12.0, 11.0])
    assert spread.p50_ms == 11.0
    assert (spread.min_ms, spread.max_ms) == (10.0, 12.0)
    assert spread.spread_pct == pytest.approx(20.0)
    assert spread.passes == 3


def test_a_single_pass_reports_a_zero_spread_rather_than_hiding_it():
    """Not an error, but visibly one pass: `passes` says so and the range collapses."""
    spread = Spread.of([7.5])
    assert spread.p50_ms == spread.min_ms == spread.max_ms == 7.5
    assert spread.spread_pct == 0.0
    assert spread.passes == 1


def test_spread_survives_a_zero_measurement():
    """A gather on an empty past can measure zero; that must not divide by it."""
    assert Spread.of([0.0, 1.0]).spread_pct == 0.0


# --- the fitted cost model --------------------------------------------------


def test_the_decode_fit_recovers_a_line_it_was_given():
    """Exact on synthetic points, so a real residual means the model is wrong."""
    fit = fit_cache_cost(
        [_decode(128, 4.0 + 0.005 * 128), _decode(512, 4.0 + 0.005 * 512)],
        [_prefill(1024, 350.0)],
        recompute_chunk_tokens=0,
    )
    assert fit.decode_base_ms == pytest.approx(4.0, abs=1e-3)
    assert fit.decode_per_token_ms == pytest.approx(0.005, abs=1e-5)
    assert fit.decode_max_residual_ms == pytest.approx(0.0, abs=1e-6)
    assert fit.fitted_from_points == 2


def test_the_decode_residual_shows_when_a_line_is_the_wrong_model():
    """A line is only right while this stays small, so it is reported rather than assumed.

    These points curve, and the fit has to say so instead of quietly averaging them.
    """
    fit = fit_cache_cost(
        [_decode(128, 4.0), _decode(512, 6.0), _decode(960, 20.0)],
        [_prefill(1024, 350.0)],
        recompute_chunk_tokens=0,
    )
    assert fit.decode_max_residual_ms > 1.0


def test_the_fitted_coefficients_feed_the_policy_directly():
    """The output of the fit has to be constructible as the policy's input.

    Nothing else checks that the two agree on what the numbers mean, and a sign or a
    unit slipping between them would be invisible.
    """
    fit = fit_cache_cost(
        [_decode(128, 4.64), _decode(512, 6.56), _decode(960, 8.80)],
        [_prefill(128, 35.2), _prefill(512, 164.5), _prefill(1024, 350.0)],
        recompute_chunk_tokens=0,
    )
    cost = CacheCost(
        decode_base_ms=fit.decode_base_ms,
        decode_per_token_ms=fit.decode_per_token_ms,
        prefill_per_token_ms=fit.prefill_per_token_ms,
    )
    assert cost.decode_ms(512) == pytest.approx(6.56, abs=fit.decode_max_residual_ms + 1e-6)
    assert cost.prefill_ms(1024) > cost.prefill_ms(512)


def test_the_prefill_rate_is_weighted_towards_long_prompts():
    """Long sequences are where an eviction misjudged by recompute cost is expensive.

    A least-squares rate through the origin weighted by prompt length lands nearer the
    long end than a plain mean of the per-length rates would, which is the direction
    that matters: getting a 128-token recompute wrong costs 30 ms, getting a
    1024-token one wrong costs 350.
    """
    prefill = [
        _prefill(128, 128 * 0.28, chunk_tokens=256),
        _prefill(1024, 1024 * 0.36, chunk_tokens=256),
    ]
    fit = fit_cache_cost(
        [_decode(128, 4.0), _decode(512, 6.0)], prefill, recompute_chunk_tokens=256
    )
    unweighted = (0.28 + 0.36) / 2
    assert fit.prefill_per_token_ms > unweighted
    assert fit.prefill_per_token_ms < 0.36


def test_the_recompute_rate_comes_from_the_width_a_resume_would_run():
    """The chunk sweep measures the same prompt several ways; they are not the same rate.

    `DecoderClient.resume` re-runs a history at the default prefill width, so that is
    the configuration the estimate has to come from. On this host a 1024-token GPT-2
    prefill is 0.364 ms per token chunked at 256 and 0.417 in one pass, so drawing the
    rate from the single-pass sweep would overstate every recompute by 13% and leave
    the policy needlessly unwilling to evict.
    """
    prefill = [
        _prefill(1024, 372.4, chunk_tokens=256),
        _prefill(1024, 427.4, chunk_tokens=0),
    ]
    chunked = fit_cache_cost([_decode(128, 4.0)], prefill, recompute_chunk_tokens=256)
    single = fit_cache_cost([_decode(128, 4.0)], prefill, recompute_chunk_tokens=0)

    assert chunked.prefill_per_token_ms == pytest.approx(372.4 / 1024, abs=1e-4)
    assert single.prefill_per_token_ms == pytest.approx(427.4 / 1024, abs=1e-4)
    assert chunked.prefill_per_token_ms < single.prefill_per_token_ms


def test_fitting_at_a_width_that_was_never_measured_is_refused():
    """A rate averaged over whatever happened to be there is not a rate."""
    with pytest.raises(ValueError, match="chunk width"):
        fit_cache_cost(
            [_decode(128, 4.0)],
            [_prefill(512, 164.0, chunk_tokens=256)],
            recompute_chunk_tokens=64,
        )


def test_fitting_without_a_decode_measurement_is_refused():
    """Silently returning zeros would give the policy a cost model of nothing."""
    with pytest.raises(ValueError, match="cannot be fitted"):
        fit_cache_cost([], [_prefill(1024, 350.0)], recompute_chunk_tokens=0)


def test_a_single_decode_point_yields_a_flat_cost():
    """One point cannot show a slope, so it must not invent one."""
    fit = fit_cache_cost([_decode(512, 6.5)], [_prefill(512, 164.0)], recompute_chunk_tokens=0)
    assert isinstance(fit, CacheCostFit)
    assert fit.decode_per_token_ms == 0.0
    assert fit.decode_base_ms == pytest.approx(6.5)
