"""A reference decode loop with contiguous KV, and the block-allocated one beside it.

`DecoderSession` replaces the obvious way to decode over an exported graph: keep the
`present` tensors ONNX Runtime hands back and feed them straight in as the next
`past`. That costs no gather at all and is the fastest thing available, so it is the
right reference. The block allocator does not claim to beat it -- it claims to
compute the same thing while making the arena's occupancy a number somebody can
admit or evict against.

Both loops are written here so a test can run one against the other. Same discipline
that put the in-process engine against the subprocess worker it replaced before that
worker was deleted: validate the replacement against the thing it replaces, while
both still exist.

`reference_generate` is parametrised over how the graph is run, so the same loop
serves two references: the in-process engine, which isolates the gather because it
is the same ONNX Runtime instance, and the `onnxruntime` wheel, which is an
independent one.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

RunNamed = Callable[[dict[str, np.ndarray]], dict[str, np.ndarray]]


def engine_runner(engine: Any, variant: str) -> RunNamed:
    """Run the graph through the in-process engine, returning outputs by name."""
    names = list(engine.output_names(variant))

    def run(feeds: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        outputs, _ = engine.run(variant, feeds)
        return dict(zip(names, outputs, strict=True))

    return run


def wheel_runner(session: Any) -> RunNamed:
    """Run the graph through the onnxruntime wheel: an independent ONNX Runtime."""
    declared = {spec.name for spec in session.get_inputs()}
    names = [spec.name for spec in session.get_outputs()]

    def run(feeds: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        fed = {name: value for name, value in feeds.items() if name in declared}
        return dict(zip(names, session.run(None, fed), strict=True))

    return run


def reference_generate(
    run: RunNamed,
    prompt: list[int],
    steps: int,
    *,
    layers: int,
    kv_heads: int,
    head_dim: int,
    chunk_tokens: int = 0,
    recompute_at: int | None = None,
) -> tuple[list[int], list[np.ndarray]]:
    """Greedy generation with the cache held as contiguous tensors.

    Mirrors `DecoderSession` step for step, including how a chunked prefill splits
    the prompt, so a difference between the two is the block allocator's and not the
    loop's.

    `recompute_at` drops the cache after that step and re-runs the whole history
    through the graph, which is what preemption does. Returns the emitted tokens and
    the next-token logits from the prefill and from each step.
    """

    def empty_past() -> dict[str, np.ndarray]:
        zero = np.zeros((1, kv_heads, 0, head_dim), dtype=np.float32)
        return {
            f"past_key_values.{layer}.{kind}": zero
            for layer in range(layers)
            for kind in ("key", "value")
        }

    def past_from(outputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        # Copied because the engine's outputs are views over runtime buffers that the
        # next run is free to reuse.
        return {
            f"past_key_values.{layer}.{kind}": np.ascontiguousarray(
                outputs[f"present.{layer}.{kind}"]
            )
            for layer in range(layers)
            for kind in ("key", "value")
        }

    def run_span(
        tokens: list[int], offset: int, past: dict[str, np.ndarray]
    ) -> dict[str, np.ndarray]:
        return run(
            {
                "input_ids": np.asarray(tokens, dtype=np.int64).reshape(1, -1),
                "attention_mask": np.ones((1, offset + len(tokens)), dtype=np.int64),
                "position_ids": np.arange(offset, offset + len(tokens), dtype=np.int64).reshape(
                    1, -1
                ),
                **past,
            }
        )

    def prefill(tokens: list[int]) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        width = chunk_tokens if chunk_tokens > 0 else len(tokens)
        past = empty_past()
        outputs: dict[str, np.ndarray] = {}
        done = 0
        while done < len(tokens):
            take = min(width, len(tokens) - done)
            outputs = run_span(tokens[done : done + take], done, past)
            past = past_from(outputs)
            done += take
        return outputs, past

    history = list(prompt)
    outputs, past = prefill(history)
    logits = [np.asarray(outputs["logits"])[0, -1].copy()]
    emitted: list[int] = []

    for step in range(steps):
        token = int(logits[-1].argmax())
        emitted.append(token)
        history.append(token)
        if recompute_at is not None and step == recompute_at:
            outputs, past = prefill(history)
        else:
            outputs = run_span([token], len(history) - 1, past)
            past = past_from(outputs)
        logits.append(np.asarray(outputs["logits"])[0, -1].copy())

    return emitted, logits


def session_generate(
    session: Any,
    sequence_id: str,
    prompt: list[int],
    steps: int,
    *,
    chunk_tokens: int = 0,
    recompute_at: int | None = None,
    reserve: int | None = None,
) -> tuple[list[int], list[np.ndarray]]:
    """The same generation through the block-allocated cache.

    `recompute_at` releases the sequence's blocks after that step and re-prefills its
    whole history, which is exactly what preempt-and-recompute does under memory
    pressure: the tokens are kept, the cache is not.
    """
    reserved = reserve if reserve is not None else len(prompt) + steps + 1
    if not session.contains(sequence_id):
        if not session.open(sequence_id, reserved):
            raise AssertionError(
                f"could not reserve {reserved} token(s) for {sequence_id}; "
                f"{session.free_blocks} of {session.capacity_blocks} blocks free"
            )

    result = session.prefill(sequence_id, prompt, chunk_tokens=chunk_tokens)
    logits = [np.asarray(result.logits).copy()]
    history = list(prompt)
    emitted: list[int] = []

    for step in range(steps):
        token = int(logits[-1].argmax())
        emitted.append(token)
        history.append(token)
        if recompute_at is not None and step == recompute_at:
            session.release(sequence_id)
            if not session.open(sequence_id, reserved):
                raise AssertionError(f"could not re-admit {sequence_id} after preemption")
            result = session.prefill(sequence_id, history, chunk_tokens=chunk_tokens)
        else:
            result = session.decode(sequence_id, token)
        logits.append(np.asarray(result.logits).copy())

    return emitted, logits


def top_two_margin(logits: np.ndarray) -> float:
    """Gap between the best and second-best logit.

    Reported by the preemption test so a flake is diagnosable. Token identity across
    a recompute is not a float guarantee: recomputing runs one wide pass where the
    uninterrupted path ran many narrow ones, and on GPT-2 those differ by about
    6e-05. That is only invisible while the winning margin is comfortably larger.
    """
    ordered = np.sort(np.asarray(logits, dtype=np.float64))
    return float(ordered[-1] - ordered[-2])
