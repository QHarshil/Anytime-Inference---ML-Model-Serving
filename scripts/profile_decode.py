"""Measure TTFT and TPOT for the decoder path, and what block accounting costs.

Prefill and decode are not one measurement. On this host GPT-2 124M at FP32 runs a
1024-token prompt in about 370 ms and then emits each following token in about 9 ms:
a factor of 40. A single latency figure would describe neither phase, so time to
first token and time per output token are measured and reported separately.

Everything goes through `DecoderClient`, which is the path that serves decoding
traffic. Stage 1 profiled through a separate ONNX Runtime session and that is how a
7.6x version mismatch stayed hidden for a whole stage; `profile_variants.py` carries
the same rule for the encoder path.

What the block allocator costs
------------------------------

The honest comparison for a block-allocated cache is not "against nothing", it is
against the faster thing it replaces: holding a sequence's KV contiguously by feeding
the `present` tensors ONNX Runtime returns straight back as the next `past`. That
costs no gather at all. So this script runs both, reports the difference, and refuses
to write results if the two disagree on the tokens they emit -- the same divergence
guard `profile_variants.py` uses, for the same reason.

The block allocator does not claim to be faster. It claims to make the arena's
occupancy a number admission and eviction can act on, and this is what that costs.

Every latency is a median over repeated passes with the range attached. A single pass
is not reportable on this host: run-to-run spread is 4-6%, far larger than the spread
within a pass, and driven by thermal state rather than by anything the code does.

Writes:

  results/decode_profiles.json   all measurements, the fitted cost model, host metadata

Nothing else. In particular this does not touch `configs/serving.yaml`: the decoder
path is not wired into the adaptive serving harness yet, and writing a config for it
would describe something that does not run.

Usage:
    python scripts/profile_decode.py
    python scripts/profile_decode.py --precisions fp32 int8
    python scripts/profile_decode.py --quick --output /tmp/decode.json
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from anytime_serving.serving.decoder import (  # noqa: E402
    DEFAULT_INTRA_OP_THREADS,
    DecoderClient,
    GenerationRequest,
)
from anytime_serving.serving.onnx_runtime import extension_available, load_extension
from anytime_serving.utils.logger import get_logger

LOGGER = get_logger("scripts.profile_decode")

PRECISIONS = ("fp32", "int8", "int4")
DEFAULT_PROMPTS = (128, 256, 512, 1024)
QUICK_PROMPTS = (128, 512)
# Prefill widths to sweep at the longest prompt. Zero means one pass, which is the
# configuration the others are compared against.
DEFAULT_CHUNKS = (0, 512, 256, 128)
QUICK_CHUNKS = (0, 256)
# Cached lengths to measure a decode step against. TPOT grows with the cache, so a
# single figure would be wrong at one end or the other.
DEFAULT_CACHED = (128, 512, 960)
QUICK_CACHED = (128, 512)
DEFAULT_REPEATS = 3
# Decode steps per pass. Small enough that the cache barely grows during one -- 16
# tokens at 960 cached moves the step by under 1% -- and large enough for a median.
DECODE_STEPS = 16
# GPT-2's position table. Exceeding it is an out-of-bounds Gather inside ONNX Runtime
# rather than a graceful stop, and it is an initializer so it cannot be read off the
# graph.
DEFAULT_MAX_CONTEXT = 1024
DEFAULT_BLOCK_TOKENS = 64
# Blocks the arena holds. 24 covers a 1024-token GPT-2 sequence at 64 tokens a block
# with room to grow, at 108 MB. The measurement is single-sequence, so a larger pool
# would only mean a longer memset at startup.
DEFAULT_BLOCKS = 24
# How much *slower* the arena path may be inside Session::Run than contiguous KV before
# the run is failed. Same bound as profile_variants.py, loose enough not to fire on
# scheduler noise and tight enough that anything structural trips it.
#
# One-sided, and that is a correction rather than a loosening. The check used to be
# two-sided on the premise that the two paths run the same graph on the same shapes, so
# the only reason to differ was noise. That premise is false once the session is
# threaded: measured over three runs, INT4 at 512 and 960 cached tokens comes out at
# 0.83-0.88x, the arena being consistently *faster*, while the arena's own Run time is
# stable to 2% and the contiguous side is the one that moves. The cause is where the
# bytes were: the gather writes the staging buffer immediately before Run, leaving it
# hot in cache, while the contiguous path feeds freshly allocated numpy arrays that are
# cold. It shows at INT4 and not at FP32 or INT8 because INT4's weights are compressed,
# so cache traffic is a much larger share of what the step reads.
#
# The direction that still means a fault is the arena being slower: identical shapes
# cannot make the graph do more work, so extra time there would mean the arena is
# handing it something other than what it looks like. Being faster has a measured
# explanation, and token identity is asserted separately and unconditionally, so
# failing on it would be failing on a cache effect. Widening the bound to 0.35 instead
# would have hidden the asymmetry and let a genuine regression through in the other
# direction.
GRAPH_AGREEMENT_TOLERANCE = 0.15


@dataclass
class Spread:
    """A median with the range it was drawn from.

    Never a single pass: the same measurement on this host moves 4-6% between runs
    from thermal state, and the spread within a pass says nothing about that.
    """

    p50_ms: float
    min_ms: float
    max_ms: float
    spread_pct: float
    passes: int

    @classmethod
    def of(cls, per_pass: list[float]) -> Spread:
        low, high = min(per_pass), max(per_pass)
        return cls(
            p50_ms=round(statistics.median(per_pass), 3),
            min_ms=round(low, 3),
            max_ms=round(high, 3),
            spread_pct=round((high - low) / low * 100.0 if low > 0 else 0.0, 2),
            passes=len(per_pass),
        )

    def __str__(self) -> str:
        return (
            f"{self.p50_ms:8.2f} ms [{self.min_ms:.2f}-{self.max_ms:.2f}, {self.spread_pct:.1f}%]"
        )


@dataclass
class PrefillMeasurement:
    """Time to first token for one prompt length at one prefill width."""

    precision: str
    prompt_tokens: int
    chunk_tokens: int
    graph_runs: int
    ttft: Spread
    gather_p50_ms: float
    run_p50_ms: float
    scatter_p50_ms: float
    ms_per_prompt_token: float
    peak_logits_mb: float


@dataclass
class DecodeMeasurement:
    """Time per output token at one cached length, and what the arena added to it.

    `arena_cost_pct` is the honest statement of that: gather plus scatter as a share
    of the step, both measured inside the same run rather than by differencing two
    paths. Differencing was tried first and is worse -- the contiguous reference is
    driven from a Python loop that builds 27 numpy feeds per step, so the comparison
    picks up that overhead as well as the arena's and moved 5 points between runs
    while the gather itself moved 0.03 ms.

    The contiguous path is still measured, for `graph_run_agreement`: both paths hand
    the same graph the same shapes, so the arena cannot make Session::Run do more work.
    If it takes materially longer, the arena is feeding the graph something different
    from what it looks like.

    The other direction is not a fault and is not treated as one. The arena comes out
    0.83-0.88x at INT4 with a full cache, reproducibly, because the gather leaves the
    staging buffer hot in cache while the contiguous path feeds cold, freshly allocated
    numpy arrays. See `GRAPH_AGREEMENT_TOLERANCE`. This does not make the arena a
    speedup -- `arena_cost_pct` is the honest statement of what it costs, and it is
    positive -- it means the comparison is not the controlled experiment its name
    suggests once the session is threaded.
    """

    precision: str
    cached_tokens: int
    tpot: Spread
    gather_p50_ms: float
    run_p50_ms: float
    scatter_p50_ms: float
    arena_cost_pct: float
    contiguous_run: Spread
    graph_run_agreement: float
    steps_per_pass: int
    cache_mb: float


@dataclass
class CacheCostFit:
    """Coefficients for `kv_admission.CacheCost`, fitted to the measurements.

    The policy needs a decode cost and a recompute cost, and both are host- and
    model-specific. Fitting them here rather than writing them down is the point:
    Stage 1's result was invalid because a service time was carried over from
    somewhere it did not apply.
    """

    decode_base_ms: float
    decode_per_token_ms: float
    prefill_per_token_ms: float
    # Largest gap between the fitted decode line and a measured point. A line is only
    # the right model while this stays small.
    decode_max_residual_ms: float
    fitted_from_points: int


@dataclass
class PrecisionProfile:
    precision: str
    graph: str
    size_mb: float
    prefill: list[PrefillMeasurement] = field(default_factory=list)
    decode: list[DecodeMeasurement] = field(default_factory=list)
    cache_cost: CacheCostFit | None = None
    best_chunk_tokens: int = 0
    chunk_speedup_vs_single_pass: float = 1.0


def _graph_path(directory: Path) -> Path:
    graphs = sorted(directory.glob("*.onnx"))
    if not graphs:
        raise SystemExit(f"no .onnx graph in {directory}")
    return graphs[0]


def _graph_bytes(graph: Path) -> int:
    total = graph.stat().st_size
    for sidecar in graph.parent.glob(f"{graph.name}*data*"):
        if sidecar != graph:
            total += sidecar.stat().st_size
    return total


def _prompt(length: int, *, vocab: int, seed: int = 0) -> list[int]:
    """A fixed pseudo-random prompt.

    Random rather than real text on purpose: this measures cost, not quality, and
    cost depends on the number of tokens rather than on which ones. Perplexity is
    `export_decoder.py`'s job and is scored on real text there.
    """
    rng = np.random.default_rng(seed)
    return rng.integers(0, vocab, size=length).astype(np.int64).tolist()


def _request(prompt: list[int], max_new_tokens: int, *, request_id: str) -> GenerationRequest:
    # A deadline large enough not to interfere: this run measures cost, and admission
    # against a real deadline is what the policy tests cover.
    return GenerationRequest(
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        deadline_ms=1e9,
        request_id=request_id,
    )


def measure_prefill(
    client: DecoderClient,
    precision: str,
    *,
    prompt_tokens: int,
    chunk_tokens: int,
    vocab: int,
    repeats: int,
) -> PrefillMeasurement:
    """TTFT for one prompt length, as a median over independent passes."""
    prompt = _prompt(prompt_tokens, vocab=vocab)
    totals: list[float] = []
    gathers: list[float] = []
    runs: list[float] = []
    scatters: list[float] = []
    graph_runs = 0

    # One warm-up pass, discarded. The first prefill of a session allocates the
    # staging buffers and faults in the arena, which is real but is not what a
    # steady-state TTFT means.
    for index in range(repeats + 1):
        request_id = f"{precision}-prefill-{prompt_tokens}-{chunk_tokens}-{index}"
        client.admit(_request(prompt, 1, request_id=request_id))
        record = client.prefill(request_id, chunk_tokens=chunk_tokens)
        client.release(request_id)
        if index == 0:
            continue
        totals.append(record.total_ms)
        gathers.append(record.gather_ms)
        runs.append(record.run_ms)
        scatters.append(record.scatter_ms)
        graph_runs = record.runs

    width = chunk_tokens if chunk_tokens > 0 else prompt_tokens
    return PrefillMeasurement(
        precision=precision,
        prompt_tokens=prompt_tokens,
        chunk_tokens=chunk_tokens,
        graph_runs=graph_runs,
        ttft=Spread.of(totals),
        gather_p50_ms=round(statistics.median(gathers), 3),
        run_p50_ms=round(statistics.median(runs), 3),
        scatter_p50_ms=round(statistics.median(scatters), 3),
        ms_per_prompt_token=round(statistics.median(totals) / prompt_tokens, 4),
        peak_logits_mb=round(min(width, prompt_tokens) * vocab * 4 / 1e6, 1),
    )


def _contiguous_decode_pass(
    engine, precision: str, prompt: list[int], steps: int, *, geometry
) -> tuple[list[float], list[int]]:
    """Prefill and decode with the cache held contiguously, timing each Session::Run.

    The `present` tensors are fed straight back as the next `past`, so there is no
    gather at all: this is the faster thing the block allocator replaces, and the
    reference its graph time is checked against.

    Times what the engine reports for Session::Run rather than the wall clock around
    the loop. The loop is Python and builds 27 numpy feeds per step, and that
    overhead belongs to the reference implementation rather than to the graph, so
    including it would flatter the arena.

    Deliberately written out here rather than shared with
    `tests/decoder_reference.py`, which asserts equality rather than measuring time.
    """
    layers, kv_heads, head_dim = geometry.layers, geometry.kv_heads, geometry.head_dim
    names = list(engine.output_names(precision))

    def run(tokens: list[int], offset: int, past: dict[str, np.ndarray]):
        feeds = {
            "input_ids": np.asarray(tokens, dtype=np.int64).reshape(1, -1),
            "attention_mask": np.ones((1, offset + len(tokens)), dtype=np.int64),
            "position_ids": np.arange(offset, offset + len(tokens), dtype=np.int64).reshape(1, -1),
            **past,
        }
        outputs, latency_ms = engine.run(precision, feeds)
        return dict(zip(names, outputs, strict=True)), latency_ms

    def past_from(outputs) -> dict[str, np.ndarray]:
        return {
            f"past_key_values.{layer}.{kind}": np.ascontiguousarray(
                outputs[f"present.{layer}.{kind}"]
            )
            for layer in range(layers)
            for kind in ("key", "value")
        }

    empty = np.zeros((1, kv_heads, 0, head_dim), dtype=np.float32)
    past = {
        f"past_key_values.{layer}.{kind}": empty
        for layer in range(layers)
        for kind in ("key", "value")
    }
    outputs, _ = run(prompt, 0, past)
    past = past_from(outputs)

    history = list(prompt)
    latencies: list[float] = []
    emitted: list[int] = []
    for _ in range(steps):
        token = int(np.asarray(outputs["logits"])[0, -1].argmax())
        emitted.append(token)
        history.append(token)
        outputs, run_ms = run([token], len(history) - 1, past)
        past = past_from(outputs)
        latencies.append(run_ms)
    return latencies, emitted


def measure_decode(
    client: DecoderClient,
    engine,
    precision: str,
    *,
    cached_tokens: int,
    vocab: int,
    repeats: int,
    steps: int,
) -> tuple[DecodeMeasurement, list[list[int]], list[list[int]]]:
    """TPOT at one cached length, through the arena and through contiguous KV.

    Returns the measurement plus both sets of emitted tokens, so the caller can refuse
    to publish numbers from two paths that disagree.
    """
    prompt = _prompt(cached_tokens, vocab=vocab)
    geometry = client.geometry

    block_pass_p50: list[float] = []
    gathers: list[float] = []
    runs: list[float] = []
    scatters: list[float] = []
    totals: list[float] = []
    block_tokens: list[list[int]] = []

    for index in range(repeats + 1):
        request_id = f"{precision}-decode-{cached_tokens}-{index}"
        client.admit(_request(prompt, steps, request_id=request_id))
        client.prefill(request_id)
        # The first decode step of a sequence carries the present-prefix check, which
        # runs once and is not part of a steady-state step.
        first = client.emit(request_id)
        assert first.verify_ms >= 0.0
        records = [client.emit(request_id) for _ in range(steps)]
        emitted = [first.token, *(record.token for record in records)]
        client.release(request_id)
        if index == 0:
            continue
        block_pass_p50.append(statistics.median(record.total_ms for record in records))
        gathers.extend(record.gather_ms for record in records)
        runs.extend(record.run_ms for record in records)
        scatters.extend(record.scatter_ms for record in records)
        totals.extend(record.total_ms for record in records)
        block_tokens.append([token for token in emitted if token is not None])

    contiguous_pass_p50: list[float] = []
    contiguous_tokens: list[list[int]] = []
    for index in range(repeats + 1):
        latencies, emitted_reference = _contiguous_decode_pass(
            engine, precision, prompt, steps + 1, geometry=geometry
        )
        if index == 0:
            continue
        contiguous_pass_p50.append(statistics.median(latencies[1:]))
        contiguous_tokens.append(emitted_reference)

    block = Spread.of(block_pass_p50)
    contiguous = Spread.of(contiguous_pass_p50)
    run_p50 = statistics.median(runs)
    arena_ms = statistics.median(gathers) + statistics.median(scatters)
    return (
        DecodeMeasurement(
            precision=precision,
            cached_tokens=cached_tokens,
            tpot=block,
            gather_p50_ms=round(statistics.median(gathers), 3),
            run_p50_ms=round(run_p50, 3),
            scatter_p50_ms=round(statistics.median(scatters), 4),
            arena_cost_pct=round(arena_ms / block.p50_ms * 100.0 if block.p50_ms > 0 else 0.0, 2),
            contiguous_run=contiguous,
            graph_run_agreement=round(
                run_p50 / contiguous.p50_ms if contiguous.p50_ms > 0 else 0.0, 4
            ),
            steps_per_pass=steps,
            cache_mb=round(cached_tokens * geometry.bytes_per_token / 1e6, 1),
        ),
        block_tokens,
        contiguous_tokens,
    )


def fit_cache_cost(
    decode: list[DecodeMeasurement],
    prefill: list[PrefillMeasurement],
    *,
    recompute_chunk_tokens: int,
) -> CacheCostFit:
    """Fit the coefficients `kv_admission.CacheCost` needs.

    The decode cost is a line in cached tokens, because a decode step re-reads the
    whole cache. The prefill rate is a single rate through the origin, least-squares
    weighted by prompt length, which puts the weight where recompute cost actually
    matters -- the estimate is used to compare eviction candidates, and the long ones
    are the expensive mistakes.

    `recompute_chunk_tokens` selects which prefill measurements the rate is drawn
    from, and it has to be the width `DecoderClient.resume` actually runs. The chunk
    sweep in the same results measures the same prompt several ways, and they are not
    interchangeable: on this host a 1024-token GPT-2 prefill is 0.364 ms per token
    chunked at 256 and 0.417 in a single pass, so fitting the rate from the wrong
    configuration would overstate every recompute by 13% and make the policy
    needlessly unwilling to evict.
    """
    if not decode:
        raise ValueError("a decode cost cannot be fitted from no measurements")

    lengths = np.array([m.cached_tokens for m in decode], dtype=np.float64)
    latencies = np.array([m.tpot.p50_ms for m in decode], dtype=np.float64)
    if lengths.size >= 2:
        slope, intercept = np.polyfit(lengths, latencies, 1)
    else:
        slope, intercept = 0.0, float(latencies[0])
    residual = float(np.max(np.abs(latencies - (intercept + slope * lengths))))

    comparable = [m for m in prefill if m.chunk_tokens == recompute_chunk_tokens]
    if not comparable:
        raise ValueError(
            f"no prefill measurement at the {recompute_chunk_tokens}-token chunk width "
            f"a recompute would use; measured widths were "
            f"{sorted({m.chunk_tokens for m in prefill})}"
        )
    tokens = np.array([m.prompt_tokens for m in comparable], dtype=np.float64)
    ttft = np.array([m.ttft.p50_ms for m in comparable], dtype=np.float64)
    rate = float((tokens * ttft).sum() / (tokens * tokens).sum())

    return CacheCostFit(
        decode_base_ms=round(float(intercept), 4),
        decode_per_token_ms=round(float(slope), 6),
        prefill_per_token_ms=round(rate, 4),
        decode_max_residual_ms=round(residual, 4),
        fitted_from_points=len(decode),
    )


def profile_precision(
    precision: str,
    graph: Path,
    *,
    prompts: tuple[int, ...],
    chunks: tuple[int, ...],
    cached: tuple[int, ...],
    repeats: int,
    steps: int,
    block_tokens: int,
    num_blocks: int,
    max_context: int,
    intra_op_threads: int,
) -> tuple[PrecisionProfile, list[str]]:
    """Every measurement for one precision. Returns it plus any divergences found."""
    extension = load_extension()
    # Same thread count as the session below. This engine is the contiguous-KV
    # reference the arena is cross-checked against, and the check is on time inside
    # Session::Run as well as on tokens -- so a reference running on a different
    # number of threads makes the arena look 0.38-0.82x its cost and fails the run.
    # Engine's own default is one thread, deliberately, because the encoder pool
    # depends on it; it is the caller's job to match them here.
    engine = extension.Engine([(precision, str(graph))], intra_op_threads=intra_op_threads)
    profile = PrecisionProfile(
        precision=precision,
        graph=graph.name,
        size_mb=round(_graph_bytes(graph) / 1e6, 1),
    )
    divergences: list[str] = []
    recompute_chunk = 0

    with DecoderClient(
        graph,
        block_tokens=block_tokens,
        num_blocks=num_blocks,
        max_context_tokens=max_context,
        intra_op_threads=intra_op_threads,
    ) as client:
        geometry = client.geometry
        # The width a resume would run at, which is what the recompute rate has to be
        # fitted from rather than from the single-pass sweep beside it.
        recompute_chunk = client.default_chunk_tokens
        LOGGER.info(
            "  %s: %d layers, %d kv heads, head dim %d -> %.1f KiB per token; arena "
            "%d blocks of %d tokens (%.0f MB)",
            precision,
            geometry.layers,
            geometry.kv_heads,
            geometry.head_dim,
            geometry.bytes_per_token / 1024,
            client.capacity_blocks,
            geometry.block_tokens,
            client.arena_bytes / 1e6,
        )

        # The vocabulary is the width of the logits row the graph returns, so a
        # throwaway prefill is the cheapest way to learn it. Token 0 exists in every
        # vocabulary, which avoids needing to know the size before asking.
        probe_id = f"{precision}-vocab-probe"
        client.admit(_request([0] * 8, 1, request_id=probe_id))
        client.prefill(probe_id)
        vocab = int(client.next_token_logits(probe_id).size)
        client.release(probe_id)
        LOGGER.info("  vocabulary %d", vocab)

        for prompt_tokens in prompts:
            measurement = measure_prefill(
                client,
                precision,
                prompt_tokens=prompt_tokens,
                chunk_tokens=client.default_chunk_tokens,
                vocab=vocab,
                repeats=repeats,
            )
            profile.prefill.append(measurement)
            LOGGER.info(
                "  prefill %5d tokens (chunk %d, %d run(s)): TTFT %s  %.3f ms/token",
                prompt_tokens,
                measurement.chunk_tokens,
                measurement.graph_runs,
                measurement.ttft,
                measurement.ms_per_prompt_token,
            )

        longest = max(prompts)
        for chunk in chunks:
            if chunk == client.default_chunk_tokens:
                continue
            measurement = measure_prefill(
                client,
                precision,
                prompt_tokens=longest,
                chunk_tokens=chunk,
                vocab=vocab,
                repeats=repeats,
            )
            profile.prefill.append(measurement)
            LOGGER.info(
                "  prefill %5d tokens, chunk %4d (%d run(s)): TTFT %s  peak logits %.0f MB",
                longest,
                chunk,
                measurement.graph_runs,
                measurement.ttft,
                measurement.peak_logits_mb,
            )

        at_longest = [m for m in profile.prefill if m.prompt_tokens == longest]
        single = next((m for m in at_longest if m.chunk_tokens == 0), None)
        best = min(at_longest, key=lambda m: m.ttft.p50_ms)
        profile.best_chunk_tokens = best.chunk_tokens
        if single is not None and best.ttft.p50_ms > 0:
            profile.chunk_speedup_vs_single_pass = round(single.ttft.p50_ms / best.ttft.p50_ms, 4)

        for cached_tokens in cached:
            measurement, block_tokens_seen, contiguous_seen = measure_decode(
                client,
                engine,
                precision,
                cached_tokens=cached_tokens,
                vocab=vocab,
                repeats=repeats,
                steps=steps,
            )
            profile.decode.append(measurement)
            for pass_index, (mine, theirs) in enumerate(
                zip(block_tokens_seen, contiguous_seen, strict=True)
            ):
                if mine != theirs:
                    divergences.append(
                        f"  {precision} at {cached_tokens} cached tokens, pass "
                        f"{pass_index}: the block-allocated cache emitted {mine[:6]} "
                        f"and contiguous KV emitted {theirs[:6]}"
                    )
            if measurement.graph_run_agreement - 1.0 > GRAPH_AGREEMENT_TOLERANCE:
                divergences.append(
                    f"  {precision} at {cached_tokens} cached tokens: Session::Run took "
                    f"{measurement.run_p50_ms:.3f} ms through the arena and "
                    f"{measurement.contiguous_run.p50_ms:.3f} ms over contiguous KV "
                    f"({measurement.graph_run_agreement:.3f}x). The graph and the shapes "
                    f"are the same either way, so the arena is feeding it something "
                    f"different from what it looks like"
                )
            LOGGER.info(
                "  decode at %5d cached: TPOT %s  arena costs %.1f%% of the step "
                "(gather %.3f, scatter %.4f)  graph agrees %.3fx",
                cached_tokens,
                measurement.tpot,
                measurement.arena_cost_pct,
                measurement.gather_p50_ms,
                measurement.scatter_p50_ms,
                measurement.graph_run_agreement,
            )

    profile.cache_cost = fit_cache_cost(
        profile.decode, profile.prefill, recompute_chunk_tokens=recompute_chunk
    )
    LOGGER.info(
        "  cost model: decode %.3f + %.5f per cached token ms (max residual %.3f), "
        "recompute %.4f ms/token",
        profile.cache_cost.decode_base_ms,
        profile.cache_cost.decode_per_token_ms,
        profile.cache_cost.decode_max_residual_ms,
        profile.cache_cost.prefill_per_token_ms,
    )
    return profile, divergences


def host_metadata(intra_op_threads: int) -> dict[str, object]:
    """What the run was taken on, including the settings that change the numbers.

    `intra_op_num_threads` is read from the run rather than written as a constant. It
    was a hardcoded 1, which stayed true only for as long as the thread count could
    not change; once it could, the field would have described a configuration this
    file had not been measured under. The cost model fitted here is consumed by
    `run_decode_sweep.py`, so a wrong thread count would not merely be a wrong
    annotation -- it would have the admission policy reasoning about a different
    machine from the one it is running on.
    """
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "backend": "extension",
        "onnxruntime": load_extension().onnxruntime_version(),
        "intra_op_num_threads": intra_op_threads,
        "batch_size": 1,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="gpt2", help="Model short name used in the graph paths")
    parser.add_argument("--model-dir", type=Path, default=Path("models/onnx"))
    parser.add_argument("--precisions", nargs="+", default=list(PRECISIONS), choices=PRECISIONS)
    parser.add_argument("--output", type=Path, default=Path("results/decode_profiles.json"))
    parser.add_argument(
        "--intra-op-threads",
        type=int,
        default=DEFAULT_INTRA_OP_THREADS,
        help=(
            "Threads ONNX Runtime may use inside one operator. Defaults to what the "
            "serving path uses: the cost model fitted here is what run_decode_sweep.py "
            "hands to BlockAdmission, and a model fitted under a different thread count "
            f"describes a different machine (default {DEFAULT_INTRA_OP_THREADS})"
        ),
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=DEFAULT_REPEATS,
        help=(
            "Independent passes per measurement; every latency is the median of their "
            f"medians with the range attached (default {DEFAULT_REPEATS})"
        ),
    )
    parser.add_argument("--decode-steps", type=int, default=DECODE_STEPS)
    parser.add_argument("--block-tokens", type=int, default=DEFAULT_BLOCK_TOKENS)
    parser.add_argument("--blocks", type=int, default=DEFAULT_BLOCKS)
    parser.add_argument(
        "--max-context",
        type=int,
        default=DEFAULT_MAX_CONTEXT,
        help=(
            "Positions the model was trained for. Exceeding it is an out-of-bounds "
            f"Gather inside ONNX Runtime rather than a graceful stop (default "
            f"{DEFAULT_MAX_CONTEXT})"
        ),
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Fewer prompt lengths, chunk widths and repeats",
    )
    args = parser.parse_args()

    if not extension_available():
        raise SystemExit(
            "anytime_runtime is not available. The block-allocated cache is the "
            "extension, so there is nothing to measure without it. Build it with:\n"
            "    pip install -e ."
        )
    if args.repeats < 1:
        raise SystemExit("--repeats must be at least 1")

    prompts = QUICK_PROMPTS if args.quick else DEFAULT_PROMPTS
    chunks = QUICK_CHUNKS if args.quick else DEFAULT_CHUNKS
    cached = QUICK_CACHED if args.quick else DEFAULT_CACHED
    repeats = 2 if args.quick else args.repeats
    cached = tuple(length for length in cached if length < args.max_context)
    if not cached:
        raise SystemExit(f"no cached length below --max-context {args.max_context}")

    graphs: dict[str, Path] = {}
    for precision in args.precisions:
        directory = args.model_dir / f"decoder_{args.model}_{precision}"
        if not directory.is_dir():
            raise SystemExit(
                f"{directory} is missing. Export it with:\n"
                f"    python scripts/export_decoder.py --precisions {precision}"
            )
        graphs[precision] = _graph_path(directory)

    LOGGER.info(
        "Measuring %s through the decoder path, %d pass(es) per point",
        ", ".join(args.precisions),
        repeats,
    )
    profiles: list[PrecisionProfile] = []
    divergences: list[str] = []
    for precision in args.precisions:
        profile, found = profile_precision(
            precision,
            graphs[precision],
            prompts=prompts,
            chunks=chunks,
            cached=cached,
            repeats=repeats,
            steps=args.decode_steps,
            block_tokens=args.block_tokens,
            num_blocks=args.blocks,
            max_context=args.max_context,
            intra_op_threads=args.intra_op_threads,
        )
        profiles.append(profile)
        divergences.extend(found)

    if divergences:
        raise SystemExit(
            "The block-allocated cache and contiguous KV do not agree:\n"
            + "\n".join(divergences)
            + "\n\nThese run the same graph over the same prompt and differ only in "
            "where the cache bytes are kept, so they must agree on both the tokens "
            "and the time inside Session::Run. Do not report the latencies above: a "
            "gather that corrupts the cache would still produce plausible timings, "
            "which is how a wrong number gets written down. Run pytest -q "
            "tests/test_kv_cache.py tests/test_decoder_session.py, which compares the "
            "two bitwise."
        )

    payload = {
        "host": host_metadata(args.intra_op_threads),
        "model": args.model,
        "measurement_passes": repeats,
        "decode_steps_per_pass": args.decode_steps,
        "block_tokens": args.block_tokens,
        "arena_blocks": args.blocks,
        "max_context_tokens": args.max_context,
        "block_cache_matches_contiguous_kv": True,
        "precisions": [asdict(profile) for profile in profiles],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    LOGGER.info("Wrote %s", args.output)

    LOGGER.info("")
    LOGGER.info(
        "%-8s %14s %14s %10s %12s %10s",
        "precision",
        "TTFT (longest)",
        "TPOT (longest)",
        "ratio",
        "arena cost",
        "best chunk",
    )
    for profile in profiles:
        prefill = max(
            (m for m in profile.prefill if m.chunk_tokens == 0),
            key=lambda m: m.prompt_tokens,
            default=None,
        )
        decode = max(profile.decode, key=lambda m: m.cached_tokens, default=None)
        if prefill is None or decode is None:
            continue
        LOGGER.info(
            "%-8s %11.1f ms %11.2f ms %9.1fx %11.1f%% %10d",
            profile.precision,
            prefill.ttft.p50_ms,
            decode.tpot.p50_ms,
            prefill.ttft.p50_ms / decode.tpot.p50_ms,
            decode.arena_cost_pct,
            profile.best_chunk_tokens,
        )
    LOGGER.info("")
    LOGGER.info(
        "TTFT is a single-pass prefill of the longest prompt; TPOT is the median "
        "decode step at the largest cached length. 'arena cost' is the gather and "
        "scatter as a share of that step: what block accounting costs, against a "
        "contiguous cache that needs neither and is the faster path. Repeating this "
        "whole run after a pause reproduces each figure within about 2.5%%; started "
        "back to back it drifts up to 10%% from thermal state, so leave a gap before "
        "comparing two runs, and prefer the ratios above to the milliseconds."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
