"""Coalescing several encoder requests into one ``Session::Run``.

Why this is not the decoder's scheduler
---------------------------------------

The decoder path batches because it has to: a one-token decode step is a
``[1, hidden] x [hidden, hidden]`` GEMV, which reads a layer's whole weight matrix
to do one row of arithmetic, so widening the batch is nearly free and measured at
3.00x. The encoder path is not in that regime. A 128-token request is already a
``[128, hidden]`` GEMM, and the pool already extracts parallelism a different way --
N workers running N independent single-threaded Runs on N cores.

So batching here competes with the pool for the same requests. Holding K requests
back to fill a batch is K requests not being served in parallel, and at
``intra_op_num_threads = 1`` a batched Run uses one core for all K. That trade is
what `scripts/count_encoder_batching.py` counts and `docs/benchmarks.md` records;
this module is the mechanism, deliberately built so the trade can be measured
rather than argued.

What it does
------------

`RequestBatcher` sits between a caller and something that can run a batched feed.
`infer` stays blocking and per-request, which is what makes it a drop-in for
`RuntimePool.infer`: no caller changes, so a batched load sweep and an unbatched
one are the same harness and their numbers are comparable.

A dispatcher thread pops arrivals, groups them, and hands each group to an
executor, so several batches can be in flight across the pool's workers at once
and batching does not serialise what the pool parallelised.

Grouping rules, and why each is forced
--------------------------------------

- **By variant.** Each variant is a separate loaded graph. Two variants cannot
  share a Run at all.
- **By input signature.** Requests feeding different input names cannot be stacked;
  the graph would be missing a declared input for some rows and there is no such
  thing as a per-row feed.
- **By trailing shape.** Only axis 0 (batch) and axis 1 (sequence) are joined. A
  mismatch anywhere further right is a different tensor layout, not a wider batch.

Padding, and the correctness condition
--------------------------------------

Rows in one batch may differ in length along axis 1, and ONNX Runtime needs one
rectangular tensor, so short rows are right-padded to the group's longest with
zeros.

**That is only correct if the graph masks what it is fed.** For the encoder
variants here it is: `attention_mask` is padded with 0, and a masked position
contributes nothing to the ``[CLS]`` position the classifier head reads. A graph
without a mask input, fed rows of differing length, would silently read the
padding -- so the batcher refuses to pad a group whose feeds do not include a mask
input, rather than producing plausible wrong logits. Equal-length rows need no
padding and are batched either way.

`pad_along_axis1` is where that happens and it is deliberately one function, so
the test that says a short row is unaffected by a long neighbour has one place to
aim at.
"""

from __future__ import annotations

import atexit
import threading
import time
import weakref
from collections import Counter
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..utils.logger import get_logger

LOGGER = get_logger("serving.batching")

# Input names that mark a feed as maskable, so right-padding a short row is safe.
# `attention_mask` is what every encoder variant here declares; the tuple exists so
# the condition is a named fact rather than a string buried in a branch.
MASK_INPUT_NAMES = ("attention_mask",)

# Feeds padded with something other than zero. Nothing does yet -- token ids are
# masked out and a zero id is in every vocabulary here -- but the batcher having a
# per-name pad value is what keeps a future graph that needs one from being a
# rewrite.
PAD_VALUES: dict[str, int] = {}


class BatchingClosed(RuntimeError):
    """Raised when a request arrives after `close`, or is pending during one."""


# Every batcher that has not been closed, so a leaked one is shut down at `atexit`
# rather than having its thread killed mid-`Condition.wait()` by interpreter teardown.
#
# The dispatcher is a daemon thread, which guarantees it can never hang an exit but says
# nothing about what state it is in when the exit happens. Closing it explicitly is the
# difference between a thread that returned and a thread that was abandoned holding a
# lock. That is worth having on its own terms; a caller that forgets `close()` should
# still get a clean shutdown.
#
# It is *not* known to fix the `std::recursive_mutex lock failed` abort a clean clone
# prints at teardown in roughly one run in ten -- see `.claude/PROGRESS.md`. That is a
# C++ mutex, so ONNX Runtime rather than `threading`, and it was not reproducible from a
# leaked batcher alone.
#
# A WeakSet rather than a list because the registry must not be what keeps a batcher
# alive. It achieves little on its own -- the dispatcher holds a bound method, so a
# running batcher is reachable regardless -- but a closed one becomes collectable at
# once.
_LIVE_BATCHERS: weakref.WeakSet = weakref.WeakSet()


def _close_live_batchers() -> None:
    for batcher in list(_LIVE_BATCHERS):
        try:
            batcher.close()
        except Exception:  # noqa: BLE001 - nothing useful to do while exiting
            pass


atexit.register(_close_live_batchers)


def pad_along_axis1(arrays: Sequence[np.ndarray], name: str) -> np.ndarray:
    """Stack per-request arrays into one batch, right-padding axis 1.

    Each input is one request's feed with its own leading axis of 1, as
    `InferenceRequest` supplies them. The result has the group's length along
    axis 1 and the padding sits to the right of every short row, which is the
    half of the convention `attention_mask` has to agree with: a mask of the right
    total weight in the wrong columns is a wrong answer that shapes cannot catch.
    """
    if not arrays:
        raise ValueError("pad_along_axis1 needs at least one array")
    first = arrays[0]
    if first.ndim < 2:
        # A 1-D feed has no sequence axis to join, so stacking is all there is.
        return np.ascontiguousarray(np.concatenate([a.reshape(1, -1) for a in arrays], axis=0))

    width = max(a.shape[1] for a in arrays)
    rows = sum(a.shape[0] for a in arrays)
    out = np.full((rows, width, *first.shape[2:]), PAD_VALUES.get(name, 0), dtype=first.dtype)
    at = 0
    for array in arrays:
        take = array.shape[0]
        out[at : at + take, : array.shape[1]] = array
        at += take
    return out


@dataclass
class _Pending:
    """One request waiting for a batch, and the slot its answer comes back in."""

    variant: str
    feeds: dict[str, np.ndarray]
    done: threading.Event = field(default_factory=threading.Event)
    logits: np.ndarray | None = None
    error: BaseException | None = None
    runtime_latency_ms: float = 0.0
    batch_size: int = 0

    @property
    def rows(self) -> int:
        first = next(iter(self.feeds.values()))
        return first.shape[0] if first.ndim else 1

    @property
    def length(self) -> int:
        first = next(iter(self.feeds.values()))
        return first.shape[1] if first.ndim >= 2 else 1

    def signature(self) -> tuple:
        """What has to match for two requests to share a Run."""
        return (
            self.variant,
            tuple(sorted(self.feeds)),
            tuple(
                (name, array.dtype.str, array.shape[2:])
                for name, array in sorted(self.feeds.items())
            ),
        )


@dataclass
class BatchCounts:
    """What the batcher did, in counts rather than seconds.

    Every field here is a property of the arrival pattern and the batching policy,
    not of how fast the host was, so these are the numbers that mean the same thing
    on a loaded machine as on an idle one.
    """

    requests: int = 0
    runs: int = 0
    rows_run: int = 0
    padded_rows: int = 0
    token_slots: int = 0
    useful_token_slots: int = 0
    widths: Counter = field(default_factory=Counter)

    @property
    def mean_width(self) -> float:
        """Rows per Run. This is the whole of what batching can amortise."""
        return self.rows_run / self.runs if self.runs else 0.0

    @property
    def padding_fraction(self) -> float:
        """Share of the batched tensors' token slots that hold padding.

        The cost side of the same trade: a batched Run computes every row at the
        longest row's length, so a wider batch amortises more per-Run work over
        more wasted arithmetic.
        """
        if not self.token_slots:
            return 0.0
        return 1.0 - self.useful_token_slots / self.token_slots

    def as_dict(self) -> dict[str, Any]:
        return {
            "requests": self.requests,
            "runs": self.runs,
            "rows_run": self.rows_run,
            "padded_rows": self.padded_rows,
            "token_slots": self.token_slots,
            "useful_token_slots": self.useful_token_slots,
            "mean_width": self.mean_width,
            "padding_fraction": self.padding_fraction,
            "widths": {str(k): v for k, v in sorted(self.widths.items())},
        }


class RequestBatcher:
    """Groups concurrent `infer` calls into batched runs of a shared graph.

    `run_batch` takes a variant and a stacked feed and returns
    ``(logits, latency_ms)`` for the whole batch, exactly as a runtime worker does
    for one request. `max_batch_size` caps a group; `max_delay_ms` caps how long
    the first request in a group waits for company.

    **`max_delay_ms` defaults to 0**, which means a group is whatever has already
    arrived when the dispatcher looks. That is the setting that cannot make a
    single request slower than no batching at all, and it is why the default is
    safe to leave on: with one caller the batcher is a pass-through of width 1.
    Any positive delay is a latency bet, and against this project's 38.7 ms
    deadline and 12.893 ms service time there is only so much slack to bet with.
    """

    def __init__(
        self,
        run_batch: Callable[[str, dict[str, np.ndarray]], tuple[np.ndarray, float]],
        *,
        max_batch_size: int,
        max_delay_ms: float = 0.0,
        max_concurrent_batches: int = 1,
    ) -> None:
        if max_batch_size < 1:
            raise ValueError("max_batch_size must be at least 1")
        if max_delay_ms < 0.0:
            raise ValueError("max_delay_ms cannot be negative")
        if max_concurrent_batches < 1:
            raise ValueError("max_concurrent_batches must be at least 1")
        self._run_batch = run_batch
        self._max_batch_size = max_batch_size
        self._max_delay_s = max_delay_ms / 1000.0
        self._max_concurrent = max_concurrent_batches

        self._lock = threading.Lock()
        self._arrived = threading.Condition(self._lock)
        # A pool rather than a thread per batch, and it is what makes `close` sound.
        # The first version started a thread per group and joined them from inside the
        # dispatch loop with a five-second timeout, while `close` waited five seconds for
        # the dispatcher -- so with several batches in flight the dispatcher could
        # outlast that wait, `close` would return, and `RuntimePool.close` would then
        # release the sessions underneath a Run that was still going.
        # `shutdown(wait=True)` has no timeout to overrun, so that window is gone.
        #
        # **It is not the fix for the SIGABRT recorded in `.claude/PROGRESS.md`**, which
        # was what prompted the change and which still reproduces after it. Kept because
        # the window above was real and a nested timeout is not a shutdown.
        self._batches = ThreadPoolExecutor(
            max_workers=max_concurrent_batches, thread_name_prefix="request-batch"
        )
        self._queue: list[_Pending] = []
        self._closed = False
        self._counts = BatchCounts()
        self._dispatcher = threading.Thread(
            target=self._dispatch_loop, name="request-batcher", daemon=True
        )
        self._dispatcher.start()
        _LIVE_BATCHERS.add(self)

    @property
    def max_batch_size(self) -> int:
        return self._max_batch_size

    @property
    def counts(self) -> BatchCounts:
        """A snapshot of what the batcher has done so far."""
        with self._lock:
            return BatchCounts(
                requests=self._counts.requests,
                runs=self._counts.runs,
                rows_run=self._counts.rows_run,
                padded_rows=self._counts.padded_rows,
                token_slots=self._counts.token_slots,
                useful_token_slots=self._counts.useful_token_slots,
                widths=Counter(self._counts.widths),
            )

    def infer(self, variant: str, feeds: dict[str, np.ndarray]) -> tuple[np.ndarray, float, int]:
        """Run one request, possibly inside somebody else's batch.

        Returns the request's own logits, the time the batch spent inside the
        runtime, and how many requests shared it. The latency is the batch's, not
        a share of it: that is how long this request actually waited on the
        runtime, and dividing it by the width would report a service time no
        request experienced.
        """
        pending = _Pending(variant=variant, feeds=feeds)
        with self._lock:
            if self._closed:
                raise BatchingClosed("the batcher is closed")
            self._queue.append(pending)
            self._counts.requests += 1
            self._arrived.notify()
        pending.done.wait()
        if pending.error is not None:
            raise pending.error
        assert pending.logits is not None  # set whenever error is None
        return pending.logits, pending.runtime_latency_ms, pending.batch_size

    def close(self) -> None:
        """Stop the dispatcher and fail anything still queued.

        Failing rather than draining: a queued request whose caller is still
        blocked in `infer` would otherwise wait for a thread that has stopped
        looking, and a serving harness shutting down wants an exception, not a
        hang.
        """
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._arrived.notify_all()
        # No timeout: the dispatcher only waits on the condition, which has just been
        # notified, so it returns promptly. A timeout here is what created the window
        # described above.
        self._dispatcher.join()
        # Every batch already handed to the pool finishes before this returns, which is
        # the guarantee `RuntimePool.close` needs before it releases the sessions those
        # batches are running on.
        self._batches.shutdown(wait=True)
        with self._lock:
            stranded, self._queue = self._queue, []
        for pending in stranded:
            pending.error = BatchingClosed("the batcher closed before this request ran")
            pending.done.set()
        _LIVE_BATCHERS.discard(self)

    def __enter__(self) -> RequestBatcher:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    # -- dispatcher ------------------------------------------------------------

    def _dispatch_loop(self) -> None:
        while True:
            group = self._take_group()
            if group is None:
                break
            # The pool bounds how many batches are in the runtime at once, so the
            # batcher cannot hand out more concurrent work than the thing behind it
            # has workers to run it on. Submitting rather than blocking lets the
            # dispatcher go straight back to forming the next group.
            self._batches.submit(self._run_group, group)

    def _take_group(self) -> list[_Pending] | None:
        """Block until there is something to run, then take one runnable group.

        Returns None when the batcher is closing. The waiting rule is the policy:
        with `max_delay_ms = 0` the group is whatever is already queued, so a lone
        request is dispatched at width 1 and pays nothing for the batcher's
        existence.
        """
        with self._lock:
            while not self._queue and not self._closed:
                self._arrived.wait()
            if self._closed:
                return None

            if self._max_delay_s > 0.0 and len(self._queue) < self._max_batch_size:
                deadline = time.monotonic() + self._max_delay_s
                while (
                    len(self._queue) < self._max_batch_size
                    and not self._closed
                    and time.monotonic() < deadline
                ):
                    self._arrived.wait(timeout=max(0.0, deadline - time.monotonic()))
                if self._closed:
                    return None

            # The head request decides the signature; anything behind it that
            # cannot share a Run stays queued in arrival order.
            head = self._queue[0]
            signature = head.signature()
            group: list[_Pending] = []
            remainder: list[_Pending] = []
            rows = 0
            for pending in self._queue:
                if pending.signature() == signature and rows + pending.rows <= self._max_batch_size:
                    group.append(pending)
                    rows += pending.rows
                else:
                    remainder.append(pending)
            self._queue = remainder
            return group

    def _run_group(self, group: list[_Pending]) -> None:
        try:
            self._execute(group)
        except BaseException as exc:  # noqa: BLE001 - delivered to the callers
            LOGGER.exception("batched run failed: %s", exc)
            for pending in group:
                pending.error = exc
        finally:
            for pending in group:
                pending.done.set()

    def _execute(self, group: list[_Pending]) -> None:
        variant = group[0].variant
        names = list(group[0].feeds)
        width = max(p.length for p in group)
        lengths = [p.length for p in group]

        if width != min(lengths) and not any(name in MASK_INPUT_NAMES for name in names):
            # Refusing beats padding: without a mask input the graph reads the
            # zeros and the logits are wrong in a way no shape check would see.
            raise RuntimeError(
                f"variant {variant!r} declares no mask input ({', '.join(MASK_INPUT_NAMES)}), "
                f"so rows of lengths {sorted(set(lengths))} cannot share a batch: the graph "
                f"would read the padding. Pad the requests to one length before submitting, "
                f"or serve this variant unbatched."
            )

        batched = {name: pad_along_axis1([p.feeds[name] for p in group], name) for name in names}
        logits, latency_ms = self._run_batch(variant, batched)

        rows_total = sum(p.rows for p in group)
        if logits.shape[0] != rows_total:
            raise RuntimeError(
                f"variant {variant!r} returned {logits.shape[0]} row(s) for a batch of "
                f"{rows_total}; its first output is not batch-major, so a row cannot be "
                f"attributed to a request"
            )

        with self._lock:
            self._counts.runs += 1
            self._counts.rows_run += rows_total
            self._counts.widths[rows_total] += 1
            self._counts.padded_rows += sum(1 for length in lengths if length < width)
            self._counts.token_slots += rows_total * width
            self._counts.useful_token_slots += sum(p.rows * p.length for p in group)

        at = 0
        for pending in group:
            take = pending.rows
            # Copied rather than viewed. Every row of one batched run points into
            # one buffer ONNX Runtime allocated, so handing out views would keep
            # the whole batch's output alive behind any single response and make
            # one request's lifetime depend on its neighbours'.
            #
            # `np.ascontiguousarray` is not enough and was the first version of this
            # line: a full-width row slice of a C-contiguous array is already
            # contiguous, so it returns the input untouched and every response held
            # a view after all. `copy=True` is the part that does the work.
            pending.logits = np.array(logits[at : at + take], copy=True, order="C")
            pending.runtime_latency_ms = latency_ms
            pending.batch_size = rows_total
            at += take
