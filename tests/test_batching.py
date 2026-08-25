"""Guards on `serving/batching.py`, which coalesces encoder requests into one Run.

The failure this module invites is not a crash. A batcher that attributes row 2 to the
request that sent row 3, or pads a short row into a graph that reads its padding,
returns logits of the right shape and the right dtype for every request, and every
caller gets an answer. So most of what follows is about identity and isolation rather
than about shapes:

- **A request gets its own row back.** Every request here is given a feed no other
  request could produce, and the answer is checked against that feed rather than
  against a shape. A batcher that returned the batch's rows in the wrong order would
  pass a shape check and fail these.
- **A row is unaffected by its neighbours.** Which is the property batching has to
  preserve and the one padding threatens.
- **Padding the mask into the wrong columns is caught.** The synthetic graph weights
  each position by its index before the mask is applied, so a mask of the right total
  weight in the wrong place is a different answer. `test_the_fixture_can_see_a_mask_in
  _the_wrong_columns` is what says that assertion has teeth: it injects the bug and
  requires the graph to notice.

The synthetic graph gives **bitwise** equality between a batched row and the same
request run alone -- its pooling is a masked sum, so appending masked zeros to a row
adds exactly nothing. That is asserted exactly rather than within a tolerance, because
a tolerance here would hide the very drift it was chosen to absorb. The real encoder
variants are a separate group, skipped when `models/` is absent, and they are *not*
bitwise: see the docstring on that group for the int8 result, which is a finding rather
than a tolerance.

The batcher's own policy tests use a stub in place of a runtime. They are the ones that
have to run everywhere and they carry no ONNX dependency at all. **Coalescing is made to
happen rather than hoped for.** Each of them starts its submitters on a
`threading.Barrier` and gives the batcher a window wide enough that the group closes on
the last arrival rather than on the clock, so an asserted width is a property of the
arrivals. The version before this one held the first batch for 0.15 s and assumed eight
threads would start inside it. On CI they did not: eight requests ran at width 1, and six
of the nine tests below assert nothing a width-1 run would violate, so they were passing
over a batcher that had not batched. Patching `RequestBatcher` to never put two requests
in a group is the check on that -- **three of the nine caught it before this change and
nine of nine do now**, which is what the barrier and the window bought.
"""

from __future__ import annotations

import math
import sys
import threading
import time
from collections import Counter
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from anytime_serving.serving.batching import (
    MASK_INPUT_NAMES,
    BatchingClosed,
    RequestBatcher,
    pad_along_axis1,
)

VOCAB, HIDDEN, CLASSES = 32, 8, 3


def build_encoder_graph(path: Path, *, include_mask: bool = True) -> None:
    """Write a tiny encoder-shaped graph: ids (and a mask) in, pooled logits out.

    Two properties make it a real test of a batcher rather than a shape-checker:

    - **Each position is weighted by its index before pooling**, so a row whose real
      tokens sit at the wrong offset, or whose mask covers the wrong columns, gives a
      different answer instead of a plausible one.
    - **The mask is applied to the pooled sum**, so a masked position contributes
      exactly zero and a batched row equals the same row run alone bit for bit. That
      is what makes the equality assertion exact.

    `include_mask=False` produces the graph that has no way to ignore padding, which
    is the one the batcher has to refuse to pad rather than serve wrongly.
    """
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    rng = np.random.default_rng(20260824)
    inputs = [helper.make_tensor_value_info("input_ids", TensorProto.INT64, ["batch", "sequence"])]
    if include_mask:
        inputs.append(
            helper.make_tensor_value_info(
                "attention_mask", TensorProto.INT64, ["batch", "sequence"]
            )
        )

    initializers = [
        # Offset off zero, so a padded position that leaked past the mask would
        # contribute something rather than nothing.
        numpy_helper.from_array(
            (rng.standard_normal((VOCAB, HIDDEN)) + 1.0).astype(np.float32), "embedding"
        ),
        numpy_helper.from_array(
            rng.standard_normal((HIDDEN, CLASSES)).astype(np.float32), "projection"
        ),
        numpy_helper.from_array(np.array([1], dtype=np.int64), "axis_1"),
        numpy_helper.from_array(np.array([2], dtype=np.int64), "axis_2"),
        numpy_helper.from_array(np.array(0, dtype=np.int64), "zero"),
        numpy_helper.from_array(np.array(1, dtype=np.int64), "one"),
        numpy_helper.from_array(np.array(1.0, dtype=np.float32), "one_f"),
        numpy_helper.from_array(np.array([1], dtype=np.int64), "start_1"),
        numpy_helper.from_array(np.array([2], dtype=np.int64), "end_2"),
        numpy_helper.from_array(np.array([], dtype=np.int64), "scalar_shape"),
    ]

    nodes = [
        helper.make_node("Gather", ["embedding", "input_ids"], ["embedded"], axis=0),
        # Positions taken from the fed shape rather than an input, so the weighting
        # follows whatever width the batch happened to have.
        helper.make_node("Shape", ["input_ids"], ["ids_shape"]),
        helper.make_node("Slice", ["ids_shape", "start_1", "end_2"], ["sequence_1d"]),
        helper.make_node("Reshape", ["sequence_1d", "scalar_shape"], ["sequence"]),
        helper.make_node("Range", ["zero", "sequence", "one"], ["positions"]),
        helper.make_node("Cast", ["positions"], ["positions_f"], to=TensorProto.FLOAT),
        helper.make_node("Add", ["positions_f", "one_f"], ["position_weight"]),
        helper.make_node("Unsqueeze", ["position_weight", "axis_1"], ["position_weight_2d"]),
        helper.make_node("Mul", ["embedded", "position_weight_2d"], ["weighted"]),
    ]
    pooled_from = "weighted"
    if include_mask:
        nodes += [
            helper.make_node("Cast", ["attention_mask"], ["mask_f"], to=TensorProto.FLOAT),
            helper.make_node("Unsqueeze", ["mask_f", "axis_2"], ["mask_3d"]),
            helper.make_node("Mul", ["weighted", "mask_3d"], ["masked"]),
        ]
        pooled_from = "masked"
    nodes += [
        helper.make_node("ReduceSum", [pooled_from, "axis_1"], ["pooled"], keepdims=0),
        helper.make_node("MatMul", ["pooled", "projection"], ["logits"]),
    ]

    outputs = [helper.make_tensor_value_info("logits", TensorProto.FLOAT, ["batch", CLASSES])]
    graph = helper.make_graph(nodes, "synthetic_encoder", inputs, outputs, initializer=initializers)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 14)], ir_version=8)
    onnx.checker.check_model(model, full_check=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(path))


# --------------------------------------------------------------------------------------
# pad_along_axis1: the one place padding happens
# --------------------------------------------------------------------------------------


def test_equal_length_rows_are_stacked_unchanged():
    a = np.array([[1, 2, 3]], dtype=np.int64)
    b = np.array([[4, 5, 6]], dtype=np.int64)
    out = pad_along_axis1([a, b], "input_ids")
    np.testing.assert_array_equal(out, np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int64))
    assert out.dtype == np.int64


def test_short_rows_are_padded_on_the_right_and_the_long_row_is_untouched():
    short = np.array([[1, 2]], dtype=np.int64)
    long = np.array([[3, 4, 5, 6]], dtype=np.int64)
    out = pad_along_axis1([short, long], "input_ids")
    # Right-padded, which is the half of the convention attention_mask has to agree
    # with: the real tokens stay at [0, len).
    np.testing.assert_array_equal(out[0], np.array([1, 2, 0, 0], dtype=np.int64))
    np.testing.assert_array_equal(out[1], np.array([3, 4, 5, 6], dtype=np.int64))


def test_padding_keeps_the_feeds_dtype():
    out = pad_along_axis1(
        [np.zeros((1, 2), dtype=np.float32), np.zeros((1, 5), dtype=np.float32)], "input"
    )
    assert out.dtype == np.float32
    assert out.shape == (2, 5)


def test_a_multi_row_request_keeps_its_rows_adjacent():
    pair = np.array([[1, 1], [2, 2]], dtype=np.int64)
    single = np.array([[3, 3, 3]], dtype=np.int64)
    out = pad_along_axis1([pair, single], "input_ids")
    assert out.shape == (3, 3)
    np.testing.assert_array_equal(out[:2, :2], pair)
    np.testing.assert_array_equal(out[2], np.array([3, 3, 3], dtype=np.int64))


def test_padding_nothing_is_an_error():
    with pytest.raises(ValueError, match="at least one array"):
        pad_along_axis1([], "input_ids")


# --------------------------------------------------------------------------------------
# The batching policy, against a stub runtime. No ONNX here.
# --------------------------------------------------------------------------------------


class _StubRuntime:
    """Records every batch it is asked to run and answers row-wise.

    The answer for a row is derived from that row's own feed, so a batcher that
    handed a row to the wrong request produces a detectable answer rather than a
    plausible one.
    """

    def __init__(self, *, fail_on: str | None = None) -> None:
        self.batches: list[tuple[str, dict[str, np.ndarray]]] = []
        self._lock = threading.Lock()
        self._fail_on = fail_on

    def run(self, variant: str, feeds: dict[str, np.ndarray]) -> tuple[np.ndarray, float]:
        with self._lock:
            self.batches.append((variant, {k: v.copy() for k, v in feeds.items()}))
        if self._fail_on is not None and variant == self._fail_on:
            raise RuntimeError(f"unknown variant {variant!r}")
        ids = feeds["input_ids"]
        # Row answer = the row's own first token. Identity, not shape.
        return ids[:, :1].astype(np.float32), 1.0

    @property
    def widths(self) -> list[int]:
        return [f["input_ids"].shape[0] for _, f in self.batches]


def _request(token: int, length: int = 2, *, mask: bool = True) -> dict[str, np.ndarray]:
    feeds = {"input_ids": np.full((1, length), token, dtype=np.int64)}
    if mask:
        feeds["attention_mask"] = np.ones((1, length), dtype=np.int64)
    return feeds


# A barrier that never fills should fail the test rather than hang the suite, so the wait
# is timed. It is generous because nothing here measures how long anything took.
_BARRIER_TIMEOUT_S = 60.0

# Wide enough that the clock never closes a group. Paired with `max_batch_size` set to
# the number of arrivals, `_take_group` returns the moment the last one lands, so the
# width these tests assert is a property of the arrivals rather than of how fast the host
# ran them.
#
# Two seconds rather than something larger, because it bounds the cost of a *failing*
# run: a batcher that stopped coalescing waits the window out once per group. At two
# seconds this file fails such a run in under a minute; at thirty it had not finished in
# two. It is still far longer than the arrivals need -- the barrier has already paid for
# starting the threads, so what is left is an append under a lock.
_UNTIL_FULL_MS = 2_000.0

# Where the group is split -- two variants, or two input signatures, cannot share a Run
# -- the remainder is smaller than `max_batch_size` and does wait the window out, once.
# So those tests need a window short enough to pay for, and assert only what a window
# that expired early would still leave true.
_UNTIL_SPLIT_MS = 250.0


def _submit_together(call: Callable[[int], Any], count: int) -> list[Future]:
    """Run ``call(i)`` for ``i`` in ``range(count)`` on threads that start together.

    The batcher can only coalesce requests that have already arrived, so a test of what
    it coalesces is really a test of what the host let arrive. `ThreadPoolExecutor`
    creates a thread per `submit`, and on a loaded runner the eighth can start long after
    the first has been answered -- which is exactly what happened on CI, where eight
    requests that pass here at width 8 ran at width 1.

    The barrier removes that: every submitter is inside `call` before any of them is let
    through, so what the batcher sees is a burst. Returns the futures, all of them
    finished; a caller expecting an exception asks for its own results.
    """
    barrier = threading.Barrier(count)

    def arrive(index: int) -> Any:
        barrier.wait(timeout=_BARRIER_TIMEOUT_S)
        return call(index)

    with ThreadPoolExecutor(max_workers=count) as pool:
        return [pool.submit(arrive, index) for index in range(count)]


def test_a_lone_request_runs_at_width_one():
    stub = _StubRuntime()
    with RequestBatcher(stub.run, max_batch_size=8) as batcher:
        logits, _, width = batcher.infer("v", _request(7))
    assert width == 1
    assert stub.widths == [1]
    assert logits[0, 0] == 7.0


def test_concurrent_requests_share_a_run():
    stub = _StubRuntime()
    with RequestBatcher(
        stub.run, max_batch_size=8, max_delay_ms=_UNTIL_FULL_MS, max_concurrent_batches=1
    ) as batcher:
        futures = _submit_together(lambda i: batcher.infer("v", _request(i)), 8)
        results = [future.result() for future in futures]
    # One Run for all eight, not "two of them happened to overlap". The window closes on
    # the eighth arrival, so this is what the batcher does rather than what the host's
    # scheduler allowed on the day.
    assert stub.widths == [8], f"eight simultaneous requests ran as {stub.widths}"
    # Every request still got its own answer back, and every one of them was told the
    # width it came from.
    assert sorted(int(logits[0, 0]) for logits, _, _ in results) == list(range(8))
    assert {width for _, _, width in results} == {8}


def test_a_batch_never_exceeds_max_batch_size():
    stub = _StubRuntime()
    with RequestBatcher(
        stub.run, max_batch_size=3, max_delay_ms=_UNTIL_FULL_MS, max_concurrent_batches=1
    ) as batcher:
        futures = _submit_together(lambda i: batcher.infer("v", _request(i)), 12)
        for future in futures:
            future.result()
    # Twelve arrivals against a cap of three, and the window closes each group the moment
    # it is full: four runs of three exactly, rather than "at most three" over whatever
    # split the host produced. `max <= 3` alone would pass over a batcher that never
    # coalesced at all, which is how it passed on CI.
    assert stub.widths == [3, 3, 3, 3], stub.widths


def test_requests_for_different_variants_do_not_share_a_run():
    stub = _StubRuntime()
    with RequestBatcher(
        stub.run, max_batch_size=8, max_delay_ms=_UNTIL_SPLIT_MS, max_concurrent_batches=1
    ) as batcher:
        futures = _submit_together(
            lambda i: batcher.infer("even" if i % 2 == 0 else "odd", _request(i)), 8
        )
        for future in futures:
            future.result()
    # Two graphs cannot share a Run, so eight simultaneous arrivals against a cap of
    # eight still come out as two runs. Which variant leads is the arrival order's to
    # decide, and the remainder is what waits the window out.
    assert sum(stub.widths) == 8
    assert {v for v, _ in stub.batches} == {"even", "odd"}
    # A batch of one here would mean the window closed on a single arrival, 250 ms after
    # eight already-running threads were released together. Three queued is enough for
    # two of them to share a variant.
    assert max(stub.widths) >= 2, f"nothing coalesced: widths {stub.widths}"
    # Identity, not shape: `_request(i)` fills its row with `i`, so a row that reached
    # the wrong variant's batch is the wrong parity.
    for variant, feeds in stub.batches:
        parity = 0 if variant == "even" else 1
        assert (feeds["input_ids"] % 2 == parity).all(), f"{variant} batch holds another's row"


def test_requests_declaring_different_inputs_do_not_share_a_run():
    stub = _StubRuntime()
    with RequestBatcher(
        stub.run, max_batch_size=4, max_delay_ms=_UNTIL_SPLIT_MS, max_concurrent_batches=1
    ) as batcher:
        futures = _submit_together(lambda i: batcher.infer("v", _request(i, mask=i % 2 == 0)), 4)
        for future in futures:
            future.result()
    # A batch is all-masked or all-unmasked. A mixed one would leave the graph missing a
    # declared input for some of its rows, which is not a thing a feed can express.
    signatures = {tuple(sorted(feeds)) for _, feeds in stub.batches}
    assert signatures == {("attention_mask", "input_ids"), ("input_ids",)}
    assert sum(stub.widths) == 4
    assert max(stub.widths) >= 2, f"nothing coalesced: widths {stub.widths}"
    # The masked requests are the even-numbered ones, so a row in the wrong group shows
    # up as the wrong parity rather than only as the wrong shape.
    for _, feeds in stub.batches:
        parity = 0 if "attention_mask" in feeds else 1
        assert (feeds["input_ids"] % 2 == parity).all(), "a row joined the wrong signature"


def test_ragged_rows_without_a_mask_input_are_refused():
    stub = _StubRuntime()
    # A window rather than a held first batch: with two requests and no window the
    # first is dispatched alone before the second arrives, and a batch of one is
    # never ragged. The window is what makes the group form.
    lengths = (2, 5)
    with RequestBatcher(
        stub.run, max_batch_size=2, max_delay_ms=_UNTIL_FULL_MS, max_concurrent_batches=1
    ) as batcher:
        futures = _submit_together(
            lambda i: batcher.infer("v", _request(i + 1, length=lengths[i], mask=False)), 2
        )
        errors = []
        for future in futures:
            try:
                future.result()
            except RuntimeError as exc:
                errors.append(str(exc))
    # Refusing beats padding: without a mask the graph reads the zeros, and wrong
    # logits of the right shape are worse than an exception.
    assert errors, "a maskless ragged batch was served rather than refused"
    assert MASK_INPUT_NAMES[0] in errors[0]


def test_ragged_rows_with_a_mask_input_are_padded_to_the_longest():
    stub = _StubRuntime()
    lengths = (2, 5, 3)
    with RequestBatcher(
        stub.run, max_batch_size=3, max_delay_ms=_UNTIL_FULL_MS, max_concurrent_batches=1
    ) as batcher:
        futures = _submit_together(
            lambda i: batcher.infer("v", _request(i + 1, length=lengths[i])), 3
        )
        for future in futures:
            future.result()
    assert stub.widths == [3], f"the three rows did not share a run: {stub.widths}"
    for _, feeds in stub.batches:
        width = feeds["input_ids"].shape[1]
        # The mask marks the real extent of each row, and it is right-aligned to
        # zero rather than to the width.
        for row in range(feeds["input_ids"].shape[0]):
            real = int(feeds["attention_mask"][row].sum())
            assert feeds["attention_mask"][row, :real].all()
            assert not feeds["attention_mask"][row, real:width].any()
            assert not feeds["input_ids"][row, real:width].any()


def test_a_failing_run_reaches_every_request_that_shared_it():
    stub = _StubRuntime(fail_on="broken")
    with RequestBatcher(stub.run, max_batch_size=4) as batcher:
        with pytest.raises(RuntimeError, match="unknown variant"):
            batcher.infer("broken", _request(1))
        # The batcher survives it: a failed batch must not take the dispatcher with
        # it, or one bad variant name stops the whole encoder lane.
        logits, _, _ = batcher.infer("fine", _request(9))
    assert logits[0, 0] == 9.0


def test_a_run_returning_the_wrong_row_count_is_refused():
    def wrong_rows(variant: str, feeds: dict[str, np.ndarray]) -> tuple[np.ndarray, float]:
        del variant, feeds
        return np.zeros((99, 2), dtype=np.float32), 1.0

    with RequestBatcher(wrong_rows, max_batch_size=4) as batcher:
        with pytest.raises(RuntimeError, match="not batch-major"):
            batcher.infer("v", _request(1))


def test_delivered_rows_are_copies_not_views_into_the_batch():
    buffers: list[np.ndarray] = []

    def keep(variant: str, feeds: dict[str, np.ndarray]) -> tuple[np.ndarray, float]:
        del variant
        out = feeds["input_ids"][:, :1].astype(np.float32)
        buffers.append(out)
        return out, 1.0

    with RequestBatcher(keep, max_batch_size=4) as batcher:
        logits, _, _ = batcher.infer("v", _request(5))
    # A row handed out as a view would keep the whole batch's output buffer alive
    # behind one response, and would change under it if the runtime reused the
    # buffer. Overwriting the source proves the delivered row is independent.
    buffers[0][:] = -1.0
    assert logits[0, 0] == 5.0


def test_the_batch_latency_is_reported_rather_than_a_share_of_it():
    stub = _StubRuntime()
    with RequestBatcher(
        stub.run, max_batch_size=4, max_delay_ms=_UNTIL_FULL_MS, max_concurrent_batches=1
    ) as batcher:
        futures = _submit_together(lambda i: batcher.infer("v", _request(i)), 4)
        results = [future.result() for future in futures]
    # The width assertion is what gives the rest of this teeth. The stub returns 1.0
    # whatever it is fed, so at width 1 the latency check below is true of a batcher
    # that never batched -- which is what it was on CI.
    assert stub.widths == [4], stub.widths
    # Every member of one batch reports the same runtime latency: the batch's, not
    # a per-request share. Dividing it by the width would report a service time no
    # request experienced.
    assert all(latency == 1.0 for _, latency, _ in results)
    assert all(width == 4 for _, _, width in results)


def test_counts_record_the_achieved_width_and_the_padding_it_cost():
    stub = _StubRuntime()
    with RequestBatcher(
        stub.run, max_batch_size=4, max_delay_ms=_UNTIL_FULL_MS, max_concurrent_batches=1
    ) as batcher:
        futures = _submit_together(lambda i: batcher.infer("v", _request(i + 1, length=4)), 4)
        for future in futures:
            future.result()
        counts = batcher.counts
    assert counts.requests == 4
    assert counts.rows_run == 4
    # One run of four, so the padding figures below are read off a batch that actually
    # formed. `sum(widths.values()) == runs` holds at any width and says nothing.
    assert counts.runs == 1
    assert counts.widths == Counter({4: 1}), counts.widths
    # Equal lengths pad nothing, whatever the width.
    assert counts.padding_fraction == pytest.approx(0.0)
    assert counts.padded_rows == 0


def test_padding_fraction_counts_the_slots_the_longest_row_forced():
    stub = _StubRuntime()
    lengths = (2, 6)
    with RequestBatcher(
        stub.run, max_batch_size=2, max_delay_ms=_UNTIL_FULL_MS, max_concurrent_batches=1
    ) as batcher:
        futures = _submit_together(
            lambda i: batcher.infer("v", _request(i + 1, length=lengths[i])), 2
        )
        for future in futures:
            future.result()
        counts = batcher.counts
    # The version before this one had a branch here for "the arrivals did not overlap on
    # this run", which is the whole disease: a test that reports zero padding when it
    # failed to measure any. The window closes on the second arrival, so one run of two
    # is the only outcome.
    assert counts.runs == 1
    # 2 + 6 useful of 2 x 6 slots.
    assert counts.token_slots == 12
    assert counts.useful_token_slots == 8
    assert counts.padding_fraction == pytest.approx(1 / 3)
    assert counts.padded_rows == 1


def test_a_request_arriving_after_close_is_refused():
    stub = _StubRuntime()
    batcher = RequestBatcher(stub.run, max_batch_size=4)
    batcher.close()
    with pytest.raises(BatchingClosed):
        batcher.infer("v", _request(1))


def test_closing_twice_is_harmless():
    batcher = RequestBatcher(_StubRuntime().run, max_batch_size=4)
    batcher.close()
    batcher.close()


def test_max_batch_size_below_one_is_rejected():
    with pytest.raises(ValueError, match="max_batch_size"):
        RequestBatcher(_StubRuntime().run, max_batch_size=0)


def test_a_negative_delay_is_rejected():
    with pytest.raises(ValueError, match="max_delay_ms"):
        RequestBatcher(_StubRuntime().run, max_batch_size=2, max_delay_ms=-1.0)


def test_a_positive_delay_still_returns_a_lone_request():
    stub = _StubRuntime()
    with RequestBatcher(stub.run, max_batch_size=8, max_delay_ms=25.0) as batcher:
        start = time.perf_counter()
        logits, _, width = batcher.infer("v", _request(3))
        waited_ms = (time.perf_counter() - start) * 1000.0
    assert width == 1
    assert logits[0, 0] == 3.0
    # The window is a bound, not a sleep the request cannot escape. Generous
    # against a loaded host; what it rules out is waiting indefinitely for company
    # that never comes.
    assert waited_ms < 5000.0


# --------------------------------------------------------------------------------------
# Through the pool, against a real ONNX graph
# --------------------------------------------------------------------------------------

onnx = pytest.importorskip("onnx", reason="onnx is needed to build the fixture graph")
pytest.importorskip("onnxruntime", reason="onnxruntime is needed to run the fixture graph")

from anytime_serving.serving.onnx_runtime import (  # noqa: E402
    InferenceRequest,
    RuntimePool,
)


@pytest.fixture(scope="module")
def encoder_graph(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("encoder") / "synthetic_encoder.onnx"
    build_encoder_graph(path)
    return path


@pytest.fixture(scope="module")
def maskless_graph(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("encoder") / "maskless_encoder.onnx"
    build_encoder_graph(path, include_mask=False)
    return path


def _feeds(tokens: list[int]) -> dict[str, np.ndarray]:
    return {
        "input_ids": np.array([tokens], dtype=np.int64),
        "attention_mask": np.ones((1, len(tokens)), dtype=np.int64),
    }


def test_the_fixture_can_see_a_mask_in_the_wrong_columns(encoder_graph):
    """Fixture validity: inject the bug the padding tests are aimed at.

    A batcher that right-padded the ids but left-padded the mask produces a mask of
    the right total weight in the wrong columns. If the graph could not tell the two
    apart, every assertion below would pass against a broken batcher.
    """
    import onnxruntime as ort

    session = ort.InferenceSession(str(encoder_graph), providers=["CPUExecutionProvider"])
    alone = session.run(None, _feeds([3, 4, 5]))[0]

    ids = np.zeros((1, 5), dtype=np.int64)
    ids[0, :3] = [3, 4, 5]
    right = np.zeros((1, 5), dtype=np.int64)
    right[0, :3] = 1
    wrong = np.zeros((1, 5), dtype=np.int64)
    wrong[0, 2:] = 1  # same weight, wrong columns

    correct = session.run(None, {"input_ids": ids, "attention_mask": right})[0]
    injected = session.run(None, {"input_ids": ids, "attention_mask": wrong})[0]
    np.testing.assert_array_equal(correct, alone)
    assert not np.allclose(injected, alone), "the graph cannot see a misplaced mask"


def test_batched_rows_equal_the_same_requests_run_alone(encoder_graph):
    lengths = [2, 7, 4, 9, 3, 6, 5, 8]
    requests = [_feeds(list(range(1, n + 1))) for n in lengths]

    with RuntimePool(1, {"enc": encoder_graph}, backend="python") as pool:
        alone = [pool.infer(InferenceRequest(variant="enc", inputs=f)).logits for f in requests]

    with RuntimePool(
        1,
        {"enc": encoder_graph},
        backend="python",
        max_batch_size=len(requests),
        max_batch_delay_ms=_UNTIL_FULL_MS,
    ) as pool:
        futures = _submit_together(
            lambda i: pool.infer(InferenceRequest(variant="enc", inputs=requests[i])),
            len(requests),
        )
        batched = [future.result() for future in futures]
        counts = pool.batch_counts

    assert counts is not None
    # One Run for all eight. Without this the comparison below could be a row against
    # itself at the same width, which is the case where nothing can go wrong.
    assert counts.widths == Counter({len(requests): 1}), counts.widths
    assert counts.rows_run == len(requests)

    # Bitwise, not approximate. The pooling is a masked sum, so a padded row differs
    # from the same row run alone by the addition of exact zeros.
    for index, (one, many) in enumerate(zip(alone, batched, strict=True)):
        np.testing.assert_array_equal(
            many.logits,
            one,
            err_msg=f"row {index} (length {lengths[index]}) changed under batching",
        )


def test_a_short_request_is_unaffected_by_a_long_neighbour(encoder_graph):
    short = _feeds([1, 2])
    long = _feeds(list(range(1, 25)))
    both = (short, long)

    with RuntimePool(1, {"enc": encoder_graph}, backend="python") as pool:
        alone = pool.infer(InferenceRequest(variant="enc", inputs=short)).logits.copy()

    with RuntimePool(
        1,
        {"enc": encoder_graph},
        backend="python",
        max_batch_size=2,
        max_batch_delay_ms=_UNTIL_FULL_MS,
    ) as pool:
        futures = _submit_together(
            lambda i: pool.infer(InferenceRequest(variant="enc", inputs=both[i])), 2
        )
        results = [future.result() for future in futures]

    assert results[0].batch_size == 2, "the two requests did not share a run"
    np.testing.assert_array_equal(results[0].logits, alone)


def test_the_response_records_the_width_it_came_from(encoder_graph):
    with RuntimePool(1, {"enc": encoder_graph}, backend="python", max_batch_size=4) as pool:
        response = pool.infer(InferenceRequest(variant="enc", inputs=_feeds([1, 2, 3])))
    # A service time read off this response means something different at width 4
    # than at width 1, and this is the only field that says which it was.
    assert response.batch_size == 1


def test_batching_is_off_unless_asked_for(encoder_graph):
    with RuntimePool(2, {"enc": encoder_graph}, backend="python") as pool:
        assert pool.max_batch_size == 1
        assert pool.batch_counts is None
        response = pool.infer(InferenceRequest(variant="enc", inputs=_feeds([1, 2])))
    assert response.batch_size == 1


def test_a_pool_configured_for_width_one_builds_no_batcher(encoder_graph):
    with RuntimePool(1, {"enc": encoder_graph}, backend="python", max_batch_size=1) as pool:
        assert pool.batch_counts is None


def test_a_zero_width_pool_is_rejected(encoder_graph):
    with pytest.raises(ValueError, match="max_batch_size"):
        RuntimePool(1, {"enc": encoder_graph}, backend="python", max_batch_size=0)


def test_a_maskless_graph_serves_one_length_and_refuses_a_ragged_batch(maskless_graph):
    same = {"input_ids": np.array([[1, 2, 3]], dtype=np.int64)}
    with RuntimePool(
        1,
        {"enc": maskless_graph},
        backend="python",
        max_batch_size=2,
        max_batch_delay_ms=_UNTIL_FULL_MS,
    ) as pool:
        # Equal lengths need no padding, so a maskless graph batches fine.
        futures = _submit_together(
            lambda i: pool.infer(InferenceRequest(variant="enc", inputs=same)), 2
        )
        for future in futures:
            future.result()
        assert pool.batch_counts is not None
        assert pool.batch_counts.widths == Counter({2: 1}), pool.batch_counts.widths

    with RuntimePool(
        1,
        {"enc": maskless_graph},
        backend="python",
        max_batch_size=2,
        max_batch_delay_ms=_UNTIL_FULL_MS,
    ) as pool:
        ragged = [
            {"input_ids": np.array([[1, 2]], dtype=np.int64)},
            {"input_ids": np.array([[1, 2, 3, 4, 5]], dtype=np.int64)},
        ]
        futures = _submit_together(
            lambda i: pool.infer(InferenceRequest(variant="enc", inputs=ragged[i])), 2
        )
        errors = []
        for future in futures:
            try:
                future.result()
            except RuntimeError as exc:
                errors.append(str(exc))
    assert errors, "a maskless graph was fed padding rather than refusing it"


def test_closing_a_batched_pool_stops_the_batcher(encoder_graph):
    pool = RuntimePool(1, {"enc": encoder_graph}, backend="python", max_batch_size=4)
    pool.infer(InferenceRequest(variant="enc", inputs=_feeds([1, 2])))
    pool.close()
    with pytest.raises((BatchingClosed, RuntimeError, IndexError)):
        pool.infer(InferenceRequest(variant="enc", inputs=_feeds([1, 2])))


def test_a_batched_pool_still_reports_its_size_and_sharing(encoder_graph):
    with RuntimePool(3, {"enc": encoder_graph}, backend="python", max_batch_size=4) as pool:
        # Batching does not change what the admission controller models: the pool
        # still has three workers and still loads the graph once.
        assert pool.size == 3
        assert pool.loaded_backends == 1
        assert pool.max_batch_size == 4


# --------------------------------------------------------------------------------------
# What bounds the achieved width, which is the question the AdaptiveServer merge asks
# --------------------------------------------------------------------------------------
#
# Batching the encoder was expected to move its concurrency model toward the decoder's
# -- from "N workers x 1 request" to "one scheduler, K requests per Run" -- and so to
# dissolve part of what makes merging the two lanes hard. These two say it does not.
#
# A batch can only hold requests that have already arrived at the runtime, and
# `AdaptiveServer` keeps at most `max_in_flight` of them there. So the widest batch the
# encoder lane can ever form is `max_in_flight`, which defaults to the pool size -- the
# same number `AdaptiveSelector` models as M/M/c's `c`. Batching wider means admitting
# more requests than there are servers, which is the identical decision the decoder
# merge needs. The wall is in the same place; batching only makes it one number.
#
# The pair is the point: the first shows the bound, the second shows it is
# `max_in_flight` that sets it and not the pool, the graph, or the batcher's cap.


def _server_with(pool, servers: int, max_in_flight: int | None):
    from anytime_serving.serving.load_monitor import LoadMonitor
    from anytime_serving.serving.selector import AdaptiveSelector, VariantProfile
    from anytime_serving.serving.server import AdaptiveServer

    selector = AdaptiveSelector(
        [VariantProfile("enc", service_time_ms=1.0, accuracy=0.9, compute_cost_per_request=1.0)],
        servers=servers,
    )
    monitor = LoadMonitor(interval_s=0.05)
    return AdaptiveServer(pool, selector, monitor, max_in_flight=max_in_flight), monitor


def _widest_batch_through_a_server(encoder_graph, *, workers: int, max_in_flight: int | None):
    with RuntimePool(
        workers,
        {"enc": encoder_graph},
        backend="python",
        max_batch_size=16,
        # A window, so the batcher waits for whatever the server is willing to have
        # in flight rather than racing a graph that runs in microseconds.
        max_batch_delay_ms=120.0,
    ) as pool:
        server, monitor = _server_with(pool, workers, max_in_flight)
        monitor.start()
        try:
            futures = [
                server.submit(
                    InferenceRequest(variant="enc", inputs=_feeds([1, 2, 3])), deadline_ms=5000.0
                )
                for _ in range(24)
            ]
            served = [f.result() for f in futures]
        finally:
            server.shutdown()
            monitor.stop()
        counts = pool.batch_counts
    assert counts is not None
    admitted = [s for s in served if s.admitted]
    assert admitted, "nothing was admitted, so the width bound was not exercised"
    return max(counts.widths), counts


def test_the_widest_batch_is_bounded_by_what_the_server_holds_in_flight(encoder_graph):
    # max_in_flight defaults to the pool size, which is also the `c` the admission
    # controller models. Two workers therefore cap the batch at two, however wide the
    # batcher was configured for.
    widest, counts = _widest_batch_through_a_server(encoder_graph, workers=2, max_in_flight=None)
    assert counts.requests > 0
    assert widest <= 2, f"width {widest} exceeded the two requests the server admits at once"


def test_admission_caps_the_batch_even_with_max_in_flight_lifted():
    """The second bound, and the one that cannot be configured away.

    Lifting `max_in_flight` does not free the width, because the selector's backlog
    term rejects an arrival once the requests already admitted would take longer than
    the deadline to clear. So the widest batch the encoder lane can form is a function
    of the deadline and the service time -- no timing, no host, just the shipped
    configuration.

    These are the numbers `benchmarks.md` quotes. The decoder needs width 8 to reach
    3.00x and flattens by 16; DistilBERT's ceiling here is 9 and the shipped default
    is 4.
    """
    yaml = pytest.importorskip("yaml")
    from anytime_serving.serving.selector import AdaptiveSelector, VariantProfile

    root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load((root / "configs" / "serving.yaml").read_text())
    deadline = config["deadline_ms"]
    workers = config["workers"]
    expected = {"distilbert_fp32": 8, "minilm_fp32": 24}

    for name, entry in config["variants"].items():
        selector = AdaptiveSelector(
            [
                VariantProfile(
                    name,
                    service_time_ms=entry["service_time_ms"],
                    accuracy=entry["accuracy"],
                    compute_cost_per_request=entry["compute_cost_per_request"],
                )
            ],
            servers=workers,
            safety_factor=config["admission"]["safety_factor"],
        )
        depth = selector.max_admissible_queue_depth(deadline)
        # The closed form the docstring states, checked against the scan so neither
        # can drift from the other.
        closed_form = workers * math.floor(deadline / entry["service_time_ms"] - 1)
        assert depth == closed_form, name
        assert depth == expected[name], f"{name}: widest batch moved to {depth + 1}"


def test_the_admissible_depth_falls_as_the_deadline_tightens():
    """Direction check on the bound, independent of the shipped numbers.

    A tighter deadline can only allow a shallower backlog. This is what says the
    quantity above is the deadline's doing rather than an artefact of one config.
    """
    from anytime_serving.serving.selector import AdaptiveSelector, VariantProfile

    selector = AdaptiveSelector(
        [VariantProfile("v", service_time_ms=10.0, accuracy=0.9, compute_cost_per_request=1.0)],
        servers=4,
    )
    depths = [selector.max_admissible_queue_depth(d) for d in (200.0, 100.0, 50.0, 25.0)]
    assert depths == sorted(depths, reverse=True), depths
    # A deadline under two service times admits no backlog at all: the arrival itself
    # already costs one service time and the first queued round costs another.
    assert selector.max_admissible_queue_depth(15.0) == 0


def test_a_non_positive_deadline_has_no_admissible_depth():
    from anytime_serving.serving.selector import AdaptiveSelector, VariantProfile

    selector = AdaptiveSelector(
        [VariantProfile("v", service_time_ms=10.0, accuracy=0.9, compute_cost_per_request=1.0)],
        servers=1,
    )
    with pytest.raises(ValueError, match="deadline_ms"):
        selector.max_admissible_queue_depth(0.0)


# --------------------------------------------------------------------------------------
# Interpreter shutdown
# --------------------------------------------------------------------------------------


def test_an_unclosed_batcher_does_not_abort_the_interpreter_at_exit(tmp_path):
    """A leaked batcher must not take the process down on the way out.

    A subprocess because interpreter shutdown is not something an in-process assertion
    can reach, and stderr as well as status because a teardown abort prints there
    without moving the exit code.

    **What this does not do is reproduce the abort that prompted it.** A clean clone
    printed `std::recursive_mutex lock failed` at teardown in about one full-suite run
    in ten; removing the `atexit` registration this guards does *not* make this test
    fail, and 15 runs of this module alone produced none. So the abort is not a leaked
    batcher, and this is a guard on the narrower thing that is actually checkable here:
    a process that leaks a batcher still exits cleanly and quietly. The abort is
    recorded as open in `.claude/PROGRESS.md` with its reproduction rate.
    """
    import subprocess

    script = tmp_path / "leak.py"
    script.write_text(
        "import numpy as np\n"
        "from anytime_serving.serving.batching import RequestBatcher\n"
        "b = RequestBatcher(\n"
        "    lambda v, f: (f['input_ids'][:, :1].astype(np.float32), 1.0), max_batch_size=4\n"
        ")\n"
        "b.infer('v', {'input_ids': np.ones((1, 2), dtype=np.int64)})\n"
        "print('ok')\n"
    )
    completed = subprocess.run(
        [sys.executable, str(script)], capture_output=True, text=True, timeout=60
    )
    assert completed.returncode == 0, completed.stderr
    assert "ok" in completed.stdout
    for symptom in ("terminating", "recursive_mutex", "Fatal Python error"):
        assert symptom not in completed.stderr, completed.stderr


def test_closing_a_batcher_leaves_no_thread_behind():
    before = {t.name for t in threading.enumerate()}
    batcher = RequestBatcher(_StubRuntime().run, max_batch_size=4)
    batcher.infer("v", _request(1))
    assert any(t.name == "request-batcher" for t in threading.enumerate())
    batcher.close()
    after = {t.name for t in threading.enumerate()}
    assert "request-batcher" not in after - before
