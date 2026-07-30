"""Tests for the block-allocated KV cache.

The arena is the part of the decoder path where a mistake is quiet. A gather that
reads the wrong block, the wrong layer or the wrong half still returns a
well-shaped tensor, and the graph still emits plausible logits from it. So the
assertions here are mostly about equality against a path that cannot have the bug,
rather than about shapes.

The sharpest of them is `test_a_sequence_built_from_scattered_blocks_matches_one_in
_order`: a sequence whose blocks are non-adjacent and out of order has to produce
bitwise the same output as one whose blocks are contiguous. That is the whole claim
of block allocation, and nothing else in the suite would notice if it broke.

Everything runs on the synthetic graph in conftest.py, so it works where the
GPT-2 export does not.
"""

import numpy as np
import pytest

from anytime_serving.serving.onnx_runtime import extension_available, load_extension
from tests.conftest import HEAD_DIM, KV_HEADS, LAYERS, build_decoder_graph

requires_extension = pytest.mark.skipif(
    not extension_available(),
    reason="anytime_runtime is not built; the arena lives in the extension. pip install -e .",
)

pytestmark = requires_extension

PROMPT = [1, 2, 3, 4, 5, 6, 7, 8]


def _session(graph, *, block_tokens=4, num_blocks=16):
    return load_extension().DecoderSession(
        str(graph), block_tokens=block_tokens, num_blocks=num_blocks
    )


# --- geometry ---------------------------------------------------------------


def test_geometry_is_read_off_the_graph(decoder_graph):
    """Derived from the graph, not from a config that could disagree with it."""
    session = _session(decoder_graph)
    geometry = session.geometry
    assert geometry.layers == LAYERS
    assert geometry.kv_heads == KV_HEADS
    assert geometry.head_dim == HEAD_DIM
    assert geometry.block_tokens == 4

    # Both halves of every layer, in float32.
    assert geometry.bytes_per_token == 2 * LAYERS * KV_HEADS * HEAD_DIM * 4
    assert geometry.bytes_per_block == geometry.bytes_per_token * 4
    assert session.arena_bytes == geometry.bytes_per_block * 16


def test_blocks_for_rounds_up(decoder_graph):
    """A partially filled block is still held; that is what blocks trade away."""
    session = _session(decoder_graph)
    assert session.blocks_for(0) == 0
    assert session.blocks_for(1) == 1
    assert session.blocks_for(4) == 1
    assert session.blocks_for(5) == 2
    assert session.blocks_for(16) == 4


def test_the_graph_declares_the_optional_inputs_the_session_builds(decoder_graph):
    session = _session(decoder_graph)
    assert session.declares_attention_mask
    assert session.declares_position_ids


# --- admission and occupancy ------------------------------------------------


def test_open_reserves_blocks_and_release_returns_them(decoder_graph):
    session = _session(decoder_graph, num_blocks=8)
    assert session.free_blocks == 8

    assert session.open("a", 8) is True
    assert session.blocks_held("a") == 2
    assert session.free_blocks == 6
    assert session.contains("a")
    assert session.sequences == ["a"]

    assert session.release("a") == 2
    assert session.free_blocks == 8
    assert not session.contains("a")


def test_open_refuses_when_the_arena_cannot_hold_the_sequence(decoder_graph):
    """Refusing is an answer, not a failure.

    The admission controller asks whether there is room, and a sequence it cannot
    hold has to come back as False rather than as an exception, so the caller can
    shed or evict instead of unwinding.
    """
    session = _session(decoder_graph, num_blocks=4)
    assert session.open("a", 16) is True
    assert session.free_blocks == 0

    assert session.open("b", 4) is False
    assert not session.contains("b")
    # The refusal must not have consumed anything or disturbed the incumbent.
    assert session.free_blocks == 0
    assert session.blocks_held("a") == 4
    result = session.prefill("a", PROMPT, chunk_tokens=0)
    assert result.length == len(PROMPT)


def test_released_blocks_are_reused(decoder_graph):
    session = _session(decoder_graph, num_blocks=4)
    assert session.open("a", 16) is True
    assert session.open("b", 4) is False

    session.release("a")
    assert session.open("b", 16) is True
    assert session.free_blocks == 0


def test_release_is_idempotent(decoder_graph):
    """A policy that releases a finished sequence and then unwinds is not an error."""
    session = _session(decoder_graph)
    session.open("a", 4)
    assert session.release("a") == 1
    assert session.release("a") == 0
    assert session.release("never-existed") == 0


def test_opening_a_sequence_twice_raises(decoder_graph):
    session = _session(decoder_graph)
    session.open("a", 4)
    with pytest.raises(RuntimeError, match="already open"):
        session.open("a", 4)


@pytest.mark.parametrize("call", ["length", "blocks_held", "decode", "prefill"])
def test_an_unknown_sequence_raises(decoder_graph, call):
    session = _session(decoder_graph)
    with pytest.raises(RuntimeError, match="unknown sequence"):
        if call == "length":
            session.length("ghost")
        elif call == "blocks_held":
            session.blocks_held("ghost")
        elif call == "decode":
            session.decode("ghost", 1)
        else:
            session.prefill("ghost", PROMPT)


def test_a_sequence_grows_into_more_blocks_as_it_decodes(decoder_graph):
    """Reserving the prompt is enough to start; decoding takes blocks as it needs them."""
    session = _session(decoder_graph, num_blocks=8)
    assert session.open("a", len(PROMPT)) is True
    assert session.blocks_held("a") == 2

    session.prefill("a", PROMPT, chunk_tokens=0)
    assert session.length("a") == 8
    assert session.blocks_held("a") == 2

    # Position 8 opens a third block; the four after it fit inside it.
    for step in range(5):
        session.decode("a", 1)
    assert session.length("a") == 13
    assert session.blocks_held("a") == 4


def test_outgrowing_the_arena_mid_decode_raises_cache_exhausted(decoder_graph):
    """The fixed arena is the point, so running out has to be visible.

    CacheExhausted derives from RuntimeError, keeping the error contract the other
    backends share, but is a distinct type because it is the one runtime error the
    admission policy is meant to handle rather than propagate.
    """
    extension = load_extension()
    session = _session(decoder_graph, num_blocks=2)
    assert session.open("a", 8) is True
    assert session.free_blocks == 0
    session.prefill("a", PROMPT, chunk_tokens=0)

    with pytest.raises(extension.CacheExhausted, match="needs 1 more block"):
        session.decode("a", 1)
    assert issubclass(extension.CacheExhausted, RuntimeError)


def test_a_prompt_larger_than_the_arena_fails_before_running_anything(decoder_graph):
    """Reserved up front, so a chunked prefill cannot die half way through."""
    extension = load_extension()
    session = _session(decoder_graph, num_blocks=2)
    assert session.open("a", 4) is True
    with pytest.raises(extension.CacheExhausted):
        session.prefill("a", list(range(1, 21)), chunk_tokens=4)
    assert session.length("a") == 0


# --- the property fixed blocks exist for ------------------------------------


def test_a_sequence_built_from_scattered_blocks_matches_one_in_order(decoder_graph):
    """Non-adjacent, out-of-order blocks must compute the same thing. Bitwise.

    This is the claim block allocation rests on: because every block is the same
    size, any free block serves any request, so the arena cannot reach a state where
    free space exists and nothing fits. The cost is that a gather has to walk a block
    list, and a gather that walked it wrongly would still return a well-shaped tensor
    full of plausible numbers.

    The free list is a stack, so releasing blocks 0 and 2 in that order and then
    asking for two hands back [2, 0] -- non-adjacent and reversed, which is the
    adversarial case rather than a lucky one.
    """
    ordered = _session(decoder_graph, num_blocks=8)
    assert ordered.open("x", 8) is True
    assert ordered.blocks_held("x") == 2
    reference = [ordered.prefill("x", PROMPT, chunk_tokens=0).logits]
    for token in (9, 10, 11):
        reference.append(ordered.decode("x", token).logits)

    scattered = _session(decoder_graph, num_blocks=8)
    for name in ("f0", "f1", "f2", "f3"):
        assert scattered.open(name, 4) is True
    scattered.release("f0")
    scattered.release("f2")
    assert scattered.open("x", 8) is True
    assert scattered.blocks_held("x") == 2

    measured = [scattered.prefill("x", PROMPT, chunk_tokens=0).logits]
    for token in (9, 10, 11):
        measured.append(scattered.decode("x", token).logits)

    for step, (want, got) in enumerate(zip(reference, measured, strict=True)):
        np.testing.assert_array_equal(
            got, want, err_msg=f"step {step} differs between scattered and ordered blocks"
        )


def test_blocks_are_not_shared_between_sequences(decoder_graph):
    """Two sequences decoding in turn must not read each other's cache.

    Interleaved deliberately: a gather that used a stale block list, or a scatter
    that wrote past the sequence's own blocks, would show up here and nowhere else.
    """
    session = _session(decoder_graph, num_blocks=16)
    alone = _session(decoder_graph, num_blocks=16)

    assert session.open("a", 8) is True
    assert session.open("b", 8) is True
    assert alone.open("a", 8) is True

    other = [40, 41, 42, 43, 44, 45, 46, 47]
    interleaved = [session.prefill("a", PROMPT, chunk_tokens=0).logits]
    session.prefill("b", other, chunk_tokens=0)
    for token in (9, 10, 11):
        interleaved.append(session.decode("a", token).logits)
        session.decode("b", token + 20)

    solo = [alone.prefill("a", PROMPT, chunk_tokens=0).logits]
    for token in (9, 10, 11):
        solo.append(alone.decode("a", token).logits)

    for step, (want, got) in enumerate(zip(solo, interleaved, strict=True)):
        np.testing.assert_array_equal(
            got, want, err_msg=f"step {step} of sequence a was disturbed by sequence b"
        )


# --- graphs the session must refuse -----------------------------------------


def test_a_graph_without_a_cache_in_its_signature_is_rejected(tmp_path):
    """An encoder, or a decoder exported without `-with-past`, cannot be paged."""
    graph = tmp_path / "no_past.onnx"
    build_decoder_graph(graph, include_past=False)
    with pytest.raises(ValueError, match="past_key_values.0.key"):
        _session(graph)


def test_a_graph_with_dynamic_cache_dimensions_is_rejected(tmp_path):
    """kv_heads and head_dim size the arena, so neither can be per-request."""
    graph = tmp_path / "dynamic.onnx"
    build_decoder_graph(graph, static_kv_dims=False)
    with pytest.raises(ValueError, match="dynamic kv_heads or head_dim"):
        _session(graph)


def test_a_graph_with_a_float64_cache_is_rejected(tmp_path):
    """The arena stores float32, and reinterpreting a wider cache would be silent."""
    graph = tmp_path / "double.onnx"
    build_decoder_graph(graph, double_cache=True)
    with pytest.raises(ValueError, match="arena stores float32"):
        _session(graph)


def test_a_graph_missing_a_present_output_is_rejected(tmp_path):
    """Without every present tensor, part of the cache could never be updated."""
    graph = tmp_path / "missing_present.onnx"
    build_decoder_graph(graph, omit_present_for_layer=1)
    with pytest.raises(ValueError, match="present.1.key"):
        _session(graph)


def test_a_graph_that_does_not_concatenate_its_cache_is_rejected(tmp_path):
    """The scatter writes only the new tail, so the invariant behind it is checked.

    `present[..., :past_len, :]` equalling the past that produced it is a property of
    how these graphs are exported, not of the ONNX specification. Measured bitwise
    true on GPT-2, and verified once per sequence rather than trusted -- because if it
    stopped holding, every token before the current one would silently rot while the
    model kept emitting fluent text.

    This graph concatenates the other way round. It runs, its shapes are right, and
    its logits are plausible. It has to be refused anyway.
    """
    graph = tmp_path / "reversed.onnx"
    build_decoder_graph(graph, reverse_present_concat=True)
    session = _session(graph)
    assert session.open("a", 16) is True
    session.prefill("a", PROMPT, chunk_tokens=0)
    with pytest.raises(RuntimeError, match="does not begin with the past"):
        session.decode("a", 9)
