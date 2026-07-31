"""Guards on the figure script, which turns measurements into pictures.

A figure is read by people who will not open the JSON behind it, so the failure that
matters is not a crash -- it is a figure that draws something other than what was
measured. Three of those are checkable without looking at pixels: that the chunk width
the prefill panel plots is read off the data rather than assumed, that a precision with
no colour assigned is refused rather than given whatever matplotlib cycles to next, and
that the files are written where they were asked for.

`--output-dir` is exercised with a tmp path throughout. The script's default writes into
`docs/img/`, and a test that let it do so would overwrite committed figures with
whatever the fixture happens to contain.
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

pytest.importorskip("matplotlib", reason="figures need the bench extra")

import plot_decode_profiles as plot  # noqa: E402

FIGURES = ("decoder_phases.png", "arena_cost.png", "chunked_prefill.png")


def _spread(value: float) -> dict:
    return {
        "p50_ms": value,
        "min_ms": value,
        "max_ms": value,
        "spread_pct": 0.0,
        "passes": 3,
    }


def _prefill(prompt_tokens: int, chunk_tokens: int, ttft_ms: float) -> dict:
    return {
        "prompt_tokens": prompt_tokens,
        "chunk_tokens": chunk_tokens,
        "graph_runs": 1,
        "ttft": _spread(ttft_ms),
        "gather_p50_ms": 0.1,
        "run_p50_ms": ttft_ms,
        "scatter_p50_ms": 0.1,
        "ms_per_prompt_token": ttft_ms / prompt_tokens,
        "peak_logits_mb": 51.5,
    }


def _decode(cached_tokens: int, tpot_ms: float) -> dict:
    return {
        "cached_tokens": cached_tokens,
        "tpot": _spread(tpot_ms),
        "gather_p50_ms": 0.2,
        "run_p50_ms": tpot_ms,
        "scatter_p50_ms": 0.01,
        "arena_cost_pct": 4.0,
        "contiguous_run": _spread(tpot_ms),
        "graph_run_agreement": 1.0,
        "steps_per_pass": 16,
        "cache_mb": 9.4,
    }


def _profile(precision: str, *, default_chunk: int = 256) -> dict:
    return {
        "precision": precision,
        "graph": "model.onnx",
        "size_mb": 100.0,
        "prefill": [
            _prefill(128, default_chunk, 40.0),
            _prefill(512, default_chunk, 160.0),
            _prefill(1024, default_chunk, 370.0),
            _prefill(1024, 0, 430.0),
            _prefill(1024, 512, 390.0),
        ],
        "decode": [_decode(128, 4.8), _decode(512, 7.2), _decode(960, 9.5)],
        "cache_cost": {
            "decode_base_ms": 4.18,
            "decode_per_token_ms": 0.00565,
            "prefill_per_token_ms": 0.35,
            "decode_max_residual_ms": 0.14,
            "fitted_from_points": 3,
        },
        "best_chunk_tokens": default_chunk,
        "chunk_speedup_vs_single_pass": 1.16,
    }


def _measurements(**overrides) -> dict:
    data = {
        "host": {"platform": "test", "onnxruntime": "1.26.0"},
        "model": "gpt2",
        "measurement_passes": 3,
        "block_tokens": 64,
        "precisions": [_profile("fp32"), _profile("int8"), _profile("int4")],
    }
    data.update(overrides)
    return data


def _write(tmp_path: Path, data: dict) -> Path:
    path = tmp_path / "decode_profiles.json"
    path.write_text(json.dumps(data))
    return path


def test_every_figure_is_written(tmp_path, monkeypatch):
    output = tmp_path / "img"
    monkeypatch.setattr(
        sys,
        "argv",
        ["plot", "--input", str(_write(tmp_path, _measurements())), "--output-dir", str(output)],
    )
    assert plot.main() == 0
    for name in FIGURES:
        assert (output / name).stat().st_size > 0, f"{name} was not drawn"


def test_the_prefill_panel_follows_the_measured_chunk_width(tmp_path):
    """The default width is read off the data, not assumed to be 256.

    The profiler sweeps prompt lengths at whatever its default is and then sweeps widths
    at the longest prompt. A script that hardcoded 256 would draw an empty panel the day
    that default moved, and an empty panel looks like a missing measurement rather than
    like a bug in the plotting.
    """
    assert plot._default_chunk(_profile("fp32", default_chunk=256)) == 256
    assert plot._default_chunk(_profile("fp32", default_chunk=192)) == 192


def test_a_precision_with_no_colour_is_refused(tmp_path, monkeypatch):
    """Rather than letting matplotlib cycle one and break colour-to-entity.

    A precision keeps one colour across every panel and figure. If a fourth appeared and
    the script simply drew it, the reader who learned that orange is int8 would be
    misled, which is worse than not drawing the figure.
    """
    data = _measurements(precisions=[_profile("fp32"), _profile("int2")])
    monkeypatch.setattr(
        sys,
        "argv",
        ["plot", "--input", str(_write(tmp_path, data)), "--output-dir", str(tmp_path / "img")],
    )
    with pytest.raises(SystemExit, match="int2"):
        plot.main()


def test_a_missing_input_says_how_to_produce_it(tmp_path, monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["plot", "--input", str(tmp_path / "absent.json"), "--output-dir", str(tmp_path)],
    )
    with pytest.raises(SystemExit, match="profile_decode.py"):
        plot.main()
