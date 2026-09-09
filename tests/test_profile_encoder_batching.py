"""Guards on `scripts/profile_encoder_batching.py` and the timed result it commits.

This is the gated half of the encoder-batching question, so most of what matters is
whether the gate is real. The first version of this script gated each variant against
its own `configs/serving.yaml` service time, which meant it gated *nothing* for the two
INT8 variants, the config records service times for the FP32 pair only, so the lookup
returned None, the check was skipped, and the payload still said `gated: true`. The
tests below are shaped by that: they check the control exists, that it is a variant the
config actually carries a number for, and that the recorded control landed inside the
band it claims to enforce.

The rest pin the finding, which is that this measurement argues against the feature it
measures:

- Width 1 is exactly 1.000x, by construction. If it drifts, the baseline is not the
  baseline.
- The peak speedup is small. `benchmarks.md` says 1.08x against the decoder's 3.00x, and
  a change that made encoder batching suddenly worth 2x would mean something else moved.
- `run_ms` grows very nearly linearly in the width, which is the mechanism: almost
  nothing amortises, so a batched Run occupies one worker for K times as long.
- **The widest width whose Run fits inside the deadline is 2.** That is the arithmetic
  that decides the feature, so it is asserted against the committed config instead of
  quoted.
"""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from profile_encoder_batching import (  # noqa: E402
    GATE_VARIANT,
    SEQUENCE_LENGTH,
    VARIANT_PATHS,
)

RESULTS = ROOT / "results" / "encoder_batching_timed.json"


@pytest.fixture(scope="module")
def config() -> dict:
    yaml = pytest.importorskip("yaml")
    return yaml.safe_load((ROOT / "configs" / "serving.yaml").read_text())


@pytest.fixture(scope="module")
def timed() -> dict:
    if not RESULTS.exists():  # pragma: no cover - committed in this repository
        pytest.skip(f"{RESULTS} is absent; run python scripts/profile_encoder_batching.py")
    return json.loads(RESULTS.read_text())


def test_the_timed_run_uses_the_length_the_benchmarks_tokenise_to():
    """Otherwise the per-request ms here is not comparable to the recorded service time."""
    import profile_variants
    import run_load_sweep

    assert SEQUENCE_LENGTH == run_load_sweep.SEQUENCE_LENGTH
    assert SEQUENCE_LENGTH == profile_variants.SEQUENCE_LENGTH


def test_the_gate_variant_has_a_recorded_service_time_to_be_gated_against(config):
    """The defect this file was written around: a gate with nothing behind it."""
    assert GATE_VARIANT in VARIANT_PATHS
    assert config["variants"][GATE_VARIANT]["service_time_ms"] > 0.0


def test_the_committed_run_declares_its_gate_and_names_the_control(timed):
    assert timed["gated"] is True
    assert GATE_VARIANT in timed["gate"]
    assert timed["sequence_length"] == SEQUENCE_LENGTH
    # Serving runs one intra-op thread, and a batching claim taken on more would be
    # measuring threading instead.
    assert timed["intra_op_threads"] == 1


def test_every_recorded_variant_kept_a_pass_and_names_its_control(timed, config):
    assert timed["variants"], "nothing was measured"
    for variant in timed["variants"]:
        assert variant["passes_kept"] > 0, variant["variant"]
        # Including the INT8 variants, which is the whole point of a separate control.
        assert variant["control_variant"] == GATE_VARIANT
        assert variant["control_recorded_ms"] == pytest.approx(
            config["variants"][GATE_VARIANT]["service_time_ms"]
        )


def test_the_control_landed_inside_the_band_it_enforces(timed):
    for variant in timed["variants"]:
        measured = variant["control_measured_ms"]
        recorded = variant["control_recorded_ms"]
        # 20% is the script's default tolerance. A pass outside it is discarded, so a
        # committed run cannot contain one. This is the check that the discard works.
        assert abs(measured / recorded - 1.0) <= 0.20, variant["variant"]


def test_width_one_is_the_baseline_by_construction(timed):
    for variant in timed["variants"]:
        first = [w for w in variant["widths"] if w["width"] == 1]
        assert first, variant["variant"]
        assert first[0]["speedup_over_width_1"] == pytest.approx(1.0, abs=1e-6)


def test_batching_the_encoder_is_worth_far_less_than_batching_a_decode_step(timed):
    """The finding. The decoder measures 3.00x at width 8; this is a different regime.

    A generous ceiling on purpose: what would be interesting is not the third decimal
    but a result that stopped being small, and that is what this would catch.
    """
    for variant in timed["variants"]:
        peak = max(w["speedup_over_width_1"] for w in variant["widths"])
        assert peak < 1.5, f"{variant['variant']} now reaches {peak:.3f}x; re-read the section"


def test_almost_nothing_amortises_across_an_encoder_batch(timed):
    """`run_ms` grows nearly linearly in the width, which is why the speedup is small.

    The complement of the test above: it says *why* the ceiling is where it is and not
    just that it is low.
    """
    for variant in timed["variants"]:
        rows = {w["width"]: w["run_ms"] for w in variant["widths"]}
        base = rows[1]
        for width, run_ms in rows.items():
            if width == 1:
                continue
            # Perfect amortisation would be run_ms == base; none at all would be
            # base * width. It sits just under the latter.
            assert run_ms > 0.6 * base * width, (variant["variant"], width)


def test_only_a_width_of_two_leaves_the_run_inside_the_deadline(timed, config):
    """The arithmetic that decides the feature.

    A batch's Run is indivisible: every request in it waits for the whole thing. So a
    width whose Run alone exceeds the deadline cannot meet it however empty the queue
    is. For DistilBERT FP32 against the 38.7 ms deadline that admits width 2 and
    nothing above it, while the speedup does not peak until 8.
    """
    deadline = config["deadline_ms"]
    for variant in timed["variants"]:
        if variant["variant"] != GATE_VARIANT:
            continue
        fitting = [w["width"] for w in variant["widths"] if w["run_ms"] <= deadline]
        assert fitting, "not even width 1 fits the deadline; the config and the host disagree"
        assert max(fitting) == 2, (
            f"widths fitting the {deadline} ms deadline are now {fitting}; the "
            "conclusion on benchmarks.md rests on this being [1, 2]"
        )
        # And the width that is actually fastest per request is outside that set, which
        # is the whole tension.
        best = max(variant["widths"], key=lambda w: w["speedup_over_width_1"])["width"]
        assert best > max(fitting), "the fastest width now fits the deadline; re-read the section"
