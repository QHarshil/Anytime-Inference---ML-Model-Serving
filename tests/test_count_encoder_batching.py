"""Guards on `scripts/count_encoder_batching.py` and the counts it commits.

Two jobs here, and the second is the one that catches drift.

**The arithmetic.** `padding_fraction` and `group_in_arrival_order` are the whole
cost side of encoder batching, and both are pure functions of a length list, so they
are checked against hand-computed values instead of against a run.

**The numbers this repository quotes.** `results/encoder_batching.json` is committed,
so the padding shares and the INT8 flip counts on `benchmarks.md` are checkable
without models. What is checked is not that the file exists but that it is
self-consistent and still agrees with everything else in the tree:

- Its padding fractions recompute from its own length distribution.
- Its FP32 reference accuracies are the accuracies in `configs/serving.yaml`. That is
  the tie that says the script's tokenisation matches the profiler that wrote the
  config, and it is what makes the INT8 result beside it comparable instead of
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
    run true lengths, they run a fixed 128. Against what actually ships, a batched
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

    Padding alone at batch 1, which is what the committed benchmarks already run,
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


# -- the static export, and whether it removes the mechanism ---------------------------
#
# `distilbert_int8_static` and `minilm_int8_static` are exported by
# `scripts/export_onnx.py --quantization static`. They exist to answer one question
# that `benchmarks.md` carried as untested: the INT8 answers move because the
# activation scale is computed at runtime, so does calibrating it away remove the
# movement?
#
# The tests below are written so that the *controls* fail first. A static graph with
# no `DynamicQuantizeLinear` is not evidence on its own, the FP32 graph has none
# either, so what is asserted first is that the static graph is genuinely 8-bit and
# that it quantised the same operators as the dynamic one. Only then does an
# assertion about flips mean anything.


def _static(committed) -> list[dict]:
    return [v for v in committed["variants"] if "int8_static" in v["variant"]]


def _dynamic(committed) -> list[dict]:
    return [v for v in committed["variants"] if v["variant"].endswith("int8")]


def test_the_static_variants_are_genuinely_quantised(committed):
    """The control for the claim below. Zero DynamicQuantizeLinear is also true of FP32."""
    variants = _static(committed)
    if not variants:  # pragma: no cover - present in this repository
        pytest.skip("no static variant measured; run scripts/export_onnx.py --quantization static")
    for variant in variants:
        census = variant["graph"]
        assert census["int8_weight_elements"] > 0, (
            f"{variant['variant']} carries no 8-bit weights, so 'no DynamicQuantizeLinear' "
            f"says only that it is not quantised"
        )
        # QDQ keeps float MatMul in the saved graph and fuses at session load, so the
        # evidence of quantisation is the Q/DQ pairs and the 8-bit initialisers.
        assert census["nodes"]["QuantizeLinear"] > 0
        assert census["nodes"]["DequantizeLinear"] > 0


def test_the_two_int8_flavours_quantised_the_same_operators(committed):
    """What makes static-versus-dynamic a controlled comparison instead of two changes.

    Static QDQ left to its own defaults quantises 24 operator types, dynamic quantises
    three. Exported that way the two would differ in how much of the graph is 8-bit as
    well as in where the activation scale comes from, and the flip counts could not be
    attributed to either. `QUANTISED_OPERATORS` pins them together and this checks the
    result: the same weights, to within the handful of elements the two conventions
    round differently.
    """
    by_name = {v["variant"]: v for v in committed["variants"]}
    for static in _static(committed):
        dynamic = by_name.get(static["variant"].removesuffix("_static"))
        if dynamic is None:  # pragma: no cover - both are exported together
            pytest.skip(f"no dynamic counterpart for {static['variant']}")
        one = static["graph"]["int8_weight_elements"]
        other = dynamic["graph"]["int8_weight_elements"]
        assert one == pytest.approx(other, rel=0.01), (
            f"{static['variant']} holds {one} 8-bit weight elements against "
            f"{other} for {dynamic['variant']}; the two are not quantising the "
            f"same operators, so their flip counts are not comparable"
        )


def test_the_static_export_removes_the_mechanism(committed):
    """The countable half of the question: no scale is computed from the tensor fed."""
    variants = _static(committed)
    if not variants:  # pragma: no cover - present in this repository
        pytest.skip("no static variant measured")
    for variant in variants:
        assert variant["dynamic_quantize_nodes"] == 0
        assert variant["graph"]["nodes"]["MatMulInteger"] == 0


def test_a_static_answer_does_not_depend_on_who_shares_its_batch(committed):
    """The prediction this experiment was run to test, on the axis it holds on.

    With the activation scales calibrated into the graph, no prediction moves with the
    shape of the tensor, at any width. That is the half of the prediction that came
    out right.
    """
    variants = _static(committed)
    if not variants:  # pragma: no cover - present in this repository
        pytest.skip("no static variant measured")
    for variant in variants:
        for arm in variant["arms"]:
            assert arm["flips"] == 0, (
                f"{variant['variant']} {arm['arm']}: a statically quantised graph still "
                f"changes {arm['flips']} prediction(s) with the shape of the tensor, so "
                f"DynamicQuantizeLinear is not the whole cause the write-up names"
            )


def test_what_static_quantisation_actually_removes_is_the_incidence(committed):
    """The half of the prediction that came out wrong, pinned so the write-up cannot drift.

    The prediction was that `max_abs_logit_delta` would fall to the FP32 control's
    1.1e-05. **It does not.** It falls from around 1.0 to around 0.3, which is four
    orders of magnitude short. What collapses instead is how many requests are affected
    at all: a dynamically quantised graph moves *every* request's logits, because every
    request gets its own activation scale, and a statically quantised one moves a
    handful, because the only way a fixed grid can move is if a float perturbation
    crosses a rounding boundary.

    So the two are not the same effect at different sizes. `rows_moved` separates them
    and the maximum does not, which is why both are recorded.
    """
    static, dynamic = _static(committed), _dynamic(committed)
    if not static or not dynamic:  # pragma: no cover - present in this repository
        pytest.skip("both flavours are needed to compare them")

    for variant in dynamic:
        for arm in variant["arms"]:
            if arm["arm"].startswith("batch 1, pad to longest"):
                continue  # the reference compared against itself
            assert arm["rows_moved"] == arm["requests"], (
                f"{variant['variant']} {arm['arm']}: dynamic quantisation no longer "
                f"moves every request, so the stated mechanism has changed"
            )

    for variant in static:
        for arm in variant["arms"]:
            assert arm["rows_moved"] <= 0.05 * arm["requests"], (
                f"{variant['variant']} {arm['arm']}: {arm['rows_moved']} of "
                f"{arm['requests']} requests moved, which is not the handful a fixed "
                f"quantisation grid should produce"
            )


def test_the_fp32_control_moves_everything_a_little_and_static_moves_little_a_lot(committed):
    """The two signatures, told apart by their shape instead of by their size.

    Float reassociation touches most requests and moves each of them by about a
    millionth of a logit. A rounding-boundary crossing touches almost none and moves
    those by a hundredth or more. This asserts the difference in kind, which is what
    the attribution on `benchmarks.md` rests on.
    """
    for variant in committed["variants"]:
        arm = next(a for a in variant["arms"] if a["arm"].startswith("batch 1, pad to 128"))
        if variant["variant"].endswith("fp32"):
            assert arm["rows_moved"] > 0.4 * arm["requests"]
            assert arm["median_abs_logit_delta_of_moved"] < 1e-5
        elif "int8_static" in variant["variant"]:
            assert arm["rows_moved"] < 0.05 * arm["requests"]
            assert arm["median_abs_logit_delta_of_moved"] > 1e-3


def test_static_quantisation_costs_accuracy_and_the_cost_is_recorded(committed):
    """The trade, so that "it is deterministic" cannot be quoted without its price.

    Neither calibration method wins on both models: percentile clipping helps
    DistilBERT and hurts MiniLM. What is asserted is the bound the write-up quotes,
    the best static variant of each model is within 0.5pp of that model's dynamic one.
    """
    by_name = {v["variant"]: v["reference_accuracy"] for v in committed["variants"]}
    for model in ("distilbert", "minilm"):
        dynamic = by_name.get(f"{model}_int8")
        static = [a for n, a in by_name.items() if n.startswith(f"{model}_int8_static")]
        if dynamic is None or not static:  # pragma: no cover - present in this repository
            pytest.skip(f"{model} was not measured in both flavours")
        assert max(static) >= dynamic - 0.005, (
            f"{model}: the best static export scores {100 * max(static):.2f}% against "
            f"{100 * dynamic:.2f}% for the dynamic one, a wider gap than benchmarks.md "
            f"records"
        )


def test_calibration_never_touches_the_split_the_accuracy_is_scored_on(committed):
    """Calibrating on validation is how a quantisation result reports a borrowed accuracy."""
    import export_onnx

    assert export_onnx.CALIBRATION_SPLIT == "train"
    assert committed["dataset"].endswith("validation")


def test_calibration_tokenises_the_way_the_benchmarks_feed_the_model():
    """A fourth script now depends on this length; all four are pinned together."""
    import export_onnx

    assert export_onnx.CALIBRATION_SEQUENCE_LENGTH == BENCHMARK_SEQUENCE_LENGTH
