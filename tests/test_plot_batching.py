"""Guards on `scripts/plot_batching.py`, which turns the batching measurements into pictures.

A figure is read by people who will not open the JSON behind it, so the failure that
matters is not a crash. It is a figure that draws something other than what was
measured, or one that encodes an ordered quantity as if it were unordered.

Four of those are checkable without looking at pixels: that the cache-occupancy ramp is
assigned shortest to longest so darker always means more cache, that a precision with
no colour assigned is refused rather than given whatever matplotlib cycles to next,
that a missing input skips its own figures instead of failing the run, and that the
files land where they were asked for. Reading the render is still required and is not
something a test does.

`--output-dir` is a tmp path throughout. The default writes into `docs/img/`, and a
test that let it do so would overwrite committed figures with fixture data.
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

pytest.importorskip("matplotlib", reason="figures need the bench extra")

import plot_batching as plot  # noqa: E402


def _spread(value: float) -> dict:
    return {
        "p50_ms": value,
        "min_ms": value * 0.99,
        "max_ms": value * 1.01,
        "spread_pct": 2.0,
        "passes": 3,
    }


def _scaling(batch_size: int, cached_tokens: int, step_ms: float, *, speedup: float) -> dict:
    return {
        "batch_size": batch_size,
        "cached_tokens": cached_tokens,
        "cached_min": cached_tokens,
        "cached_max": cached_tokens + batch_size,
        "step": _spread(step_ms),
        "tokens_per_s": batch_size / step_ms * 1000.0,
        "pad_p50_ms": 0.03,
        "gather_p50_ms": 0.2 * batch_size,
        "run_p50_ms": step_ms * 0.9,
        "scatter_p50_ms": 0.01,
        "scheduler_overhead_p50_ms": 0.05,
        "steps_per_pass": 16,
        "speedup_vs_serial": speedup,
        "predicted_speedup": speedup * 1.3,
        "prediction_ratio": 1 / 1.3,
    }


def _trace(prefill_chunks_per_decode: int) -> dict:
    steps = []
    clock = 0.0
    for index in range(8):
        kind = "prefill" if index % 2 == 0 else "decode"
        duration = 90.0 if kind == "prefill" else 20.0
        steps.append(
            {
                "index": index,
                "kind": kind,
                "batch_size": 1 if kind == "prefill" else 3,
                "start_ms": round(clock, 3),
                "end_ms": round(clock + duration, 3),
                "total_ms": duration,
                "cached_max": 500 + index,
                "completed": 0,
                "preempted": 0,
            }
        )
        clock += duration
    return {
        "prefill_chunks_per_decode": prefill_chunks_per_decode,
        "max_batch_size": 4,
        "chunk_tokens": 256,
        "requests": 6,
        "prompt_tokens": [700, 300, 520, 180, 640, 420],
        "max_new_tokens": 24,
        "steps": steps,
        "wall_ms": clock,
        "prefill_steps": 4,
        "decode_steps": 4,
        "mean_decode_batch": 3.0,
        "prefill_step_p50_ms": 90.0,
        "decode_step_p50_ms": 20.0,
        "max_decode_gap_ms": 182.2,
        "decode_gap_p50_ms": 25.9,
        "stalled_sequences": 6,
    }


def _profile(precision: str) -> dict:
    return {
        "precision": precision,
        "graph": "model.onnx",
        "size_mb": 640.0,
        "scaling": [
            _scaling(1, 128, 4.6, speedup=1.0),
            _scaling(4, 128, 11.7, speedup=1.57),
            _scaling(8, 128, 15.8, speedup=2.33),
            _scaling(1, 512, 6.8, speedup=1.0),
            _scaling(4, 512, 20.8, speedup=1.31),
            _scaling(8, 512, 35.0, speedup=1.55),
        ],
        "padding": [
            {
                "regime": "uniform-max",
                "batch_size": 8,
                "longest_tokens": 512,
                "shortest_tokens": 512,
                "mean_tokens": 512.0,
                "step": _spread(35.0),
                "pad_p50_ms": 0.14,
                "run_p50_ms": 29.7,
                "gather_p50_ms": 4.4,
                "tokens_per_s": 228.0,
            }
        ],
        "step_split": {
            "base_ms": 3.92,
            "per_token_ms": 0.00566,
            "max_residual_ms": 0.0,
            "fitted_from_points": 2,
        },
        "alternation": [_trace(1), _trace(4)],
        "skipped": [],
    }


def _mechanism(**overrides) -> dict:
    data = {
        "host": {"platform": "test", "onnxruntime": "1.26.0"},
        "model": "gpt2",
        "measurement_passes": 3,
        "measured_steps_per_pass": 16,
        "batch_sizes": [1, 4, 8],
        "cached_lengths": [128, 512],
        "block_tokens": 64,
        "arena_blocks": 128,
        "max_context_tokens": 1024,
        "assembled_prefill_first": True,
        "tokens_agree_across_batch_sizes": True,
        "precisions": [_profile("fp32"), _profile("int8"), _profile("int4")],
    }
    data.update(overrides)
    return data


def _sweep_row(policy: str, utilisation: float) -> dict:
    return {
        "policy": policy,
        "workload": "fixed",
        "target_utilisation": utilisation,
        "offered_rps": utilisation * 2.0,
        "requests": 150,
        "completed": 150,
        "rejected": 0,
        "arrival_window_s": 60.0,
        "makespan_s": 70.0,
        "achieved_rps": 2.1,
        "output_tokens_per_s": 130.0,
        "ttft_p50_ms": 200.0 * utilisation,
        "ttft_p95_ms": 400.0 * utilisation,
        "ttft_p99_ms": 500.0 * utilisation,
        "tpot_p50_ms": 12.0,
        "tpot_p95_ms": 14.0,
        "tpot_p99_ms": 16.0,
        "e2e_p50_ms": 900.0,
        "e2e_p95_ms": 1200.0,
        "slo_attainment": max(0.0, 1.0 - utilisation / 4),
        "goodput_rps": 1.9,
        "mean_decode_batch": 5.4,
        "prefill_steps": 300,
        "decode_steps": 1200,
        "preemptions": 0,
        "max_waiting": 12,
        "max_resident": 8,
        "prompt_tokens": 256,
        "mean_prompt_tokens": 256.0,
        "max_new_tokens": 64,
    }


def _sweep(**overrides) -> dict:
    data = {
        "host": {"platform": "test", "onnxruntime": "1.26.0"},
        "model": "gpt2",
        "precision": "fp32",
        "requests_per_point": 150,
        "block_tokens": 64,
        "max_context_tokens": 1024,
        "batch_size": 8,
        "preempting_resident_capacity": 4,
        "arrival_process": "poisson, open loop",
        "capacity_rps": 2.04,
        "capacity_measured_under": "batched-8",
        "reference_ttft_ms": 95.0,
        "reference_tpot_ms": 5.4,
        "ttft_slo_ms": 500.0,
        "tpot_slo_ms": 50.0,
        "deadline_ms": 3700.0,
        "policies": [],
        "sweep": [
            _sweep_row(policy, utilisation)
            for policy in ("serial", "batched-8", "batched-8-preempting")
            for utilisation in (0.4, 0.95, 1.3)
        ],
        "shapes": [],
    }
    data.update(overrides)
    return data


def _write(tmp_path: Path, name: str, data: dict) -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(data))
    return path


def _run(tmp_path, monkeypatch, *, mechanism=None, sweep=None, output=None):
    output = output or tmp_path / "img"
    argv = ["plot", "--output-dir", str(output)]
    argv += [
        "--batch-profiles",
        str(
            _write(tmp_path, "batch_profiles.json", mechanism)
            if mechanism is not None
            else tmp_path / "absent-mechanism.json"
        ),
    ]
    argv += [
        "--decode-sweep",
        str(
            _write(tmp_path, "decode_sweep.json", sweep)
            if sweep is not None
            else tmp_path / "absent-sweep.json"
        ),
    ]
    monkeypatch.setattr(sys, "argv", argv)
    return plot.main(), output


# --- the encodings ----------------------------------------------------------


def test_the_occupancy_ramp_runs_shortest_to_longest():
    """Cache occupancy is a magnitude, so darker has to mean more of it.

    Assigned by rank rather than by value, so a sweep at other lengths still reads in
    order. If this were assigned in whatever order the measurements happened to appear,
    the panel would still draw and would encode nothing.
    """
    colours = plot._occupancy_colours([960, 128, 512])
    assert [colours[length] for length in (128, 512, 960)] == list(plot.OCCUPANCY_RAMP)


def test_more_lengths_than_ramp_steps_repeat_the_darkest_rather_than_inventing_a_hue():
    """A generated fourth hue would break the one-hue rule a sequential ramp rests on.

    Two lines the same colour is a visible symptom that the ramp needs another step,
    which is the intended failure mode.
    """
    colours = plot._occupancy_colours([128, 256, 512, 960])
    assert colours[512] == colours[960] == plot.OCCUPANCY_RAMP[-1]
    assert len(set(colours.values())) == len(plot.OCCUPANCY_RAMP)


def test_batch_widths_are_drawn_as_steps_not_as_a_linear_axis():
    """1, 2, 4, 8, 16, 32 on a linear axis crushes everything below 8 into the corner."""
    positions, labels = plot._batch_positions([1, 2, 4, 8, 16, 32])
    assert positions == [0, 1, 2, 3, 4, 5]
    assert labels == ["1", "2", "4", "8", "16", "32"]


def test_the_occupancy_ramp_is_not_a_precision_colour():
    """A reader who learned that blue means fp32 must not meet blue meaning 512 tokens."""
    assert not set(plot.OCCUPANCY_RAMP) & set(plot.SERIES_COLOURS.values())


# --- what gets drawn --------------------------------------------------------


def test_every_figure_is_written(tmp_path, monkeypatch):
    status, output = _run(tmp_path, monkeypatch, mechanism=_mechanism(), sweep=_sweep())
    assert status == 0
    for name in ("batch_scaling.png", "alternation.png", "decode_sweep.png"):
        assert (output / name).stat().st_size > 0, f"{name} was not drawn"


def test_a_missing_sweep_still_draws_the_mechanism_figures(tmp_path, monkeypatch):
    """The two measurements are separate runs, so one must not block the other's figures."""
    status, output = _run(tmp_path, monkeypatch, mechanism=_mechanism())
    assert status == 0
    assert (output / "batch_scaling.png").exists()
    assert (output / "alternation.png").exists()
    assert not (output / "decode_sweep.png").exists()


def test_a_missing_mechanism_still_draws_the_sweep_figure(tmp_path, monkeypatch):
    status, output = _run(tmp_path, monkeypatch, sweep=_sweep())
    assert status == 0
    assert (output / "decode_sweep.png").exists()
    assert not (output / "batch_scaling.png").exists()


def test_nothing_to_draw_names_both_scripts(tmp_path, monkeypatch):
    with pytest.raises(SystemExit, match="profile_batching.py"):
        _run(tmp_path, monkeypatch)


def test_a_precision_with_no_colour_is_refused(tmp_path, monkeypatch):
    """Rather than letting matplotlib cycle one and break colour-to-entity."""
    data = _mechanism(precisions=[_profile("fp32"), _profile("int2")])
    with pytest.raises(SystemExit, match="int2"):
        _run(tmp_path, monkeypatch, mechanism=data)


def test_a_trace_with_no_steps_is_skipped_rather_than_drawn_empty(tmp_path, monkeypatch):
    """An empty panel reads as a missing measurement, not as a bug in the plotting."""
    profile = _profile("fp32")
    profile["alternation"] = []
    status, output = _run(tmp_path, monkeypatch, mechanism=_mechanism(precisions=[profile]))
    assert status == 0
    assert (output / "batch_scaling.png").exists()
    assert not (output / "alternation.png").exists()


def test_the_scaling_figure_handles_one_precision(tmp_path, monkeypatch):
    """A sweep of one precision is a normal thing to run, and the layout has to hold."""
    status, output = _run(
        tmp_path, monkeypatch, mechanism=_mechanism(precisions=[_profile("fp32")])
    )
    assert status == 0
    assert (output / "batch_scaling.png").stat().st_size > 0


def test_the_headline_sentence_quotes_the_widest_batch_at_both_extremes(tmp_path):
    """The suptitle states the claim, so it has to come from the data rather than a guess."""
    sentence = plot._scaling_sentence(_mechanism())
    assert "batch 8" in sentence
    assert "2.33x" in sentence
    assert "1.55x" in sentence
    assert "128 cached" in sentence
