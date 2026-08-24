"""Guards on `scripts/count_encoder_batching.py` and the counts it commits.

Two jobs here, and the second is the one that catches drift.

**The arithmetic.** `padding_fraction` and `group_in_arrival_order` are the whole
cost side of encoder batching, and both are pure functions of a length list, so they
are checked against hand-computed values rather than against a run.

**The numbers this repository quotes.** `results/encoder_batching.json` is committed,
so the padding shares and the INT8 flip counts on `benchmarks.md` are checkable
without models. What is checked is not that the file exists but that it is
self-consistent and still agrees with everything else in the tree:

- Its padding fractions recompute from its own length distribution.
- Its FP32 reference accuracies are the accuracies in `configs/serving.yaml`. That is
  the tie that says the script's tokenisation matches the profiler that wrote the
  config, and it is what makes the INT8 result beside it comparable rather than
  merely adjacent.
- Its FP32 arms flip nothing, at every width. That is the control: it is logically
  independent of the quantisation question, so if it ever starts flipping, the
  finding below is an artefact and not a finding.
- Its INT8 variants carry the `DynamicQuantizeLinear` nodes that are the stated
  mechanism, so the explanation cannot outlive the thing it explains.

`BENCHMARK_SEQUENCE_LENGTH` is asserted against the two scripts that actually
tokenise. The 80% padding figure is a claim *about them*; a script that changed its
`SEQUENCE_LENGTH` would leave that claim true of nothing.
"""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from count_encoder_batching import (  # noqa: E402
    BENCHMARK_SEQUENCE_LENGTH,
    fixed_padding_fraction,
    group_in_arrival_order,
    padding_fraction,
)

RESULTS = ROOT / "results" / "encoder_batching.json"


# -- the arithmetic --------------------------------------------------------------------


def test_grouping_is_contiguous_and_covers_every_request():
    groups = group_in_arrival_order(10, 4)
    assert groups == [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9]]
    assert [i for g in groups for i in g] == list(range(10))


def test_a_width_of_one_is_a_group_per_request():
    assert group_in_arrival_order(3, 1) == [[0], [1], [2]]


def test_a_width_wider_than_the_workload_is_one_group():
    assert group_in_arrival_order(3, 99) == [[0, 1, 2]]


def test_grouping_below_width_one_is_an_error():
    with pytest.raises(ValueError, match="at least 1"):
        group_in_arrival_order(4, 0)


def test_equal_lengths_pad_nothing_at_any_width():
    lengths = [7] * 12
    for width in (1, 2, 3, 4, 6, 12):
        assert padding_fraction(lengths, width) == pytest.approx(0.0)


def test_padding_is_charged_at_the_longest_row_in_each_group():
    # One group of [2, 6]: 8 useful tokens in 2 x 6 = 12 slots.
    assert padding_fraction([2, 6], 2) == pytest.approx(1.0 - 8 / 12)
    # Served alone, the same two requests waste nothing.
    assert padding_fraction([2, 6], 1) == pytest.approx(0.0)


def test_padding_grows_with_width_on_a_spread_workload():
    lengths = [1, 2, 4, 8, 16, 32]
    shares = [padding_fraction(lengths, w) for w in (1, 2, 3, 6)]
    assert shares == sorted(shares), shares
    assert shares[0] == pytest.approx(0.0)


def test_a_fixed_width_charges_every_row_the_same_whatever_the_batch():
    # 4 rows of 24 tokens padded to 128: the batch width cannot change this.
    assert fixed_padding_fraction([24] * 4, 128) == pytest.approx(1.0 - 24 / 128)
    assert fixed_padding_fraction([], 128) == pytest.approx(0.0)


def test_a_fixed_width_is_never_beaten_by_padding_to_the_longest_in_a_batch():
    """Padding to the longest row in a batch cannot waste more than padding to 128.

    Which is the comparison that matters and the one that is easy to get backwards:
    batching adds padding against *true lengths*, but the committed benchmarks do not
    run true lengths -- they run a fixed 128. Against what actually ships, a batched
    tensor of real lengths is the cheaper of the two.
    """
    lengths = [4, 9, 17, 24, 31, 55]
    fixed = fixed_padding_fraction(lengths, BENCHMARK_SEQUENCE_LENGTH)
    for width in (1, 2, 3, 6):
        assert padding_fraction(lengths, width) <= fixed


def test_the_benchmark_sequence_length_is_the_one_the_benchmarks_tokenise_to():
    """The 80%-padding claim is about these two scripts, so it is pinned to them."""
    import profile_variants
    import run_load_sweep

    assert BENCHMARK_SEQUENCE_LENGTH == run_load_sweep.SEQUENCE_LENGTH
    assert BENCHMARK_SEQUENCE_LENGTH == profile_variants.SEQUENCE_LENGTH


# -- the committed counts --------------------------------------------------------------


@pytest.fixture(scope="module")
def committed() -> dict:
    if not RESULTS.exists():  # pragma: no cover - committed in this repository
        pytest.skip(f"{RESULTS} is absent; run python scripts/count_encoder_batching.py")
    return json.loads(RESULTS.read_text())


def test_the_committed_counts_declare_themselves_ungated(committed):
    # Nothing here is timed, which is why it carries no host gate. A future edit that
    # added a latency column would have to change this flag and be noticed.
    assert committed["gated"] is False
    assert "timing" not in committed["note"].lower() or "no timing" in committed["note"].lower()


def test_the_committed_padding_fractions_recompute_from_the_committed_geometry(committed):
    geometry = committed["geometry"]
    mean = geometry["token_length"]["mean"]
    count = geometry["sentences"]
    # The fixed-width share needs only the mean, so it is checkable from the summary.
    expected = 1.0 - mean / geometry["benchmark_sequence_length"]
    assert geometry["fixed_padding_fraction"] == pytest.approx(expected, abs=5e-4)
    assert count > 0
    for row in geometry["widths"]:
        assert row["runs"] == len(group_in_arrival_order(count, row["width"]))
        assert 0.0 <= row["padding_fraction"] < 1.0


def test_padding_to_a_fixed_128_wastes_more_than_any_batched_width(committed):
    geometry = committed["geometry"]
    for row in geometry["widths"]:
        assert row["padding_fraction"] < geometry["fixed_padding_fraction"], row


def test_a_wider_batch_runs_fewer_times(committed):
    rows = sorted(committed["geometry"]["widths"], key=lambda r: r["width"])
    runs = [r["runs"] for r in rows]
    assert runs == sorted(runs, reverse=True)
    assert rows[0]["runs_saved_fraction"] == pytest.approx(0.0)


def test_the_fp32_control_flips_nothing_at_any_width(committed):
    fp32 = [v for v in committed["variants"] if v["variant"].endswith("fp32")]
    assert fp32, "the control is missing; every INT8 number beside it is uninterpretable"
    for variant in fp32:
        assert variant["dynamic_quantize_nodes"] == 0
        for arm in variant["arms"]:
            assert arm["flips"] == 0, f"{variant['variant']} {arm['arm']}"
            assert arm["max_abs_logit_delta"] < 1e-3, f"{variant['variant']} {arm['arm']}"


def test_the_fp32_accuracies_are_the_ones_the_serving_config_carries(committed):
    """Ties this measurement to the config the planner actually reads.

    If these ever disagree, one of the two was measured through a different feed and
    the frontier and this page stop describing the same system.
    """
    yaml = pytest.importorskip("yaml")
    config = yaml.safe_load((ROOT / "configs" / "serving.yaml").read_text())
    for variant in committed["variants"]:
        entry = config["variants"].get(variant["variant"])
        if entry is None:
            continue
        assert variant["reference_accuracy"] == pytest.approx(entry["accuracy"], abs=5e-4)


def test_the_int8_variants_carry_the_mechanism_the_finding_names(committed):
    int8 = [v for v in committed["variants"] if v["variant"].endswith("int8")]
    assert int8, "no quantised variant was measured"
    for variant in int8:
        # The stated cause of the flips. A re-export that folded these away would
        # make the explanation stale, and this is what would say so.
        assert variant["dynamic_quantize_nodes"] > 0


def test_int8_answers_move_at_batch_one_as_well_as_batched(committed):
    """The finding, and the reason it is not filed under batching.

    Padding alone at batch 1 -- which is what the committed benchmarks already run --
    moves INT8 predictions. If a future change made batching the only arm that
    flipped, the attribution on `benchmarks.md` would be wrong and this would fail.
    """
    for variant in committed["variants"]:
        if not variant["variant"].endswith("int8"):
            continue
        unbatched = [a for a in variant["arms"] if a["arm"].startswith("batch 1, pad to 128")]
        assert unbatched, variant["variant"]
        assert unbatched[0]["flips"] > 0, (
            f"{variant['variant']}: padding alone no longer perturbs INT8, so the "
            "finding is now specific to batching and benchmarks.md misattributes it"
        )


def test_int8_accuracy_moves_little_in_either_direction(committed):
    """The bound the write-up quotes: a determinism property, not a regression."""
    for variant in committed["variants"]:
        if not variant["variant"].endswith("int8"):
            continue
        for arm in variant["arms"]:
            assert abs(arm["accuracy_delta_pp"]) <= 1.0, f"{variant['variant']} {arm['arm']}"
            assert arm["flip_percent"] < 2.0, f"{variant['variant']} {arm['arm']}"
