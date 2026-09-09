"""Guards on `scripts/draw_arena_geometry.py`, which draws the arena instead of plots it.

A schematic has a failure mode a measured figure does not: nothing contradicts it. There
is no JSON behind it to disagree with, so a drawing that shows the scatter landing at
`max_past`, or a sequence holding four blocks for nine tokens, is wrong in a way only a
reader who already knows the answer would catch.

So the drawing takes every number from `Batch` and `Geometry`, and this checks those
instead of the pixels: the block arithmetic against the ceiling division the sweep script
already uses and the extension's own constant, the padding against the batch's longest,
and the scatter's landing against `past_len` instead of the width the batch happened to
have. The last group checks the illustration still illustrates, a sequence whose blocks
are scattered, a tail block held and not full, a row that pads more than it copies,
because an edit that tidied those away would leave a figure that draws nothing worth
drawing and no other test would notice.

Reading the render is still required and is not something a test does.

`--output-dir` is a tmp path throughout. The default writes into `docs/img/`, and a test
that let it do so would overwrite the committed figure.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

pytest.importorskip("matplotlib", reason="figures need the bench extra")

import draw_arena_geometry as draw  # noqa: E402
from run_decode_sweep import blocks_for  # noqa: E402


def test_gpt2_geometry_matches_the_figures_the_runtime_records():
    """The caption's numbers, against the ones `runtime/` and `docs/runtime.md` state."""
    gpt2 = draw.GPT2
    assert gpt2.slots == 24
    assert gpt2.bytes_per_token == 72 * 1024
    assert gpt2.bytes_per_block == int(4.5 * 1024**2)
    assert gpt2.blocks_for(1024) == 16
    assert gpt2.floats_per_block == 2 * gpt2.layers * gpt2.floats_per_slab


def test_the_caption_block_width_is_the_runtimes_default():
    """A caption that drifted from `kDefaultBlockTokens` would quote a cache nobody runs."""
    extension = pytest.importorskip(
        "anytime_runtime", reason="the block width comes from the built extension"
    )
    assert draw.GPT2.block_tokens == extension.DEFAULT_BLOCK_TOKENS


@pytest.mark.parametrize("tokens", [0, 1, 63, 64, 65, 260, 1024])
def test_ceiling_division_agrees_with_the_sweep_scripts(tokens):
    """One sequence's blocks, against the helper `run_decode_sweep` already ships.

    Two ceiling divisions in one repository are two chances to round the other way.
    """
    assert draw.GPT2.blocks_for(tokens) == blocks_for(
        sequences=1, tokens_each=tokens, block_tokens=draw.GPT2.block_tokens
    )


def test_a_negative_length_is_refused_rather_than_rounded():
    with pytest.raises(ValueError, match="cannot hold"):
        draw.GPT2.blocks_for(-1)


def test_padding_is_the_gap_to_the_batchs_longest_row():
    batch = draw.BATCH
    for row in batch.rows:
        assert batch.padding(row) == batch.max_past - row.length
    assert min(batch.padding(row) for row in batch.rows) == 0, "the longest row pads nothing"
    assert batch.total == batch.max_past + 1


def test_the_scatter_lands_at_past_len_and_not_at_the_batchs_width():
    """`past_len` is the row's own, `max_past` is the batch's. Confusing them is the bug
    this panel exists to make hard to have."""
    batch = draw.BATCH
    for row in batch.rows:
        ordinal, offset, block = batch.landing(row)
        assert ordinal * batch.geometry.block_tokens + offset == row.length
        assert 0 <= offset < batch.geometry.block_tokens
        assert block == row.blocks[ordinal], "the landing block is the one the ordinal names"
        assert block in row.blocks, "a row cannot scatter into a block it does not hold"


def test_every_block_but_the_last_is_drawn_full():
    batch = draw.BATCH
    for row in batch.rows:
        written = [batch.written_in(row, held) for held in range(len(row.blocks))]
        assert all(count == batch.geometry.block_tokens for count in written[:-1])
        assert sum(written) == row.length


def test_a_length_landing_on_a_boundary_fills_its_last_written_block():
    """The remainder is `block_tokens`, not zero, when a sequence ends exactly on one.

    Such a row holds a third block that is entirely empty, because the reservation covers
    the token about to be emitted, and that empty block is where the scatter goes.
    """
    batch = draw.Batch(
        geometry=draw.DRAWN,
        rows=(draw.Row(name="A", blocks=(0, 1, 2), length=8),),
        arena_blocks=4,
    )
    row = batch.rows[0]
    assert batch.written_in(row, 1) == draw.DRAWN.block_tokens
    assert batch.written_in(row, 2) == 0
    assert batch.landing(row) == (2, 0, 2)


def test_a_row_with_no_room_for_the_token_it_is_about_to_emit_is_refused():
    """Eight tokens in two four-token blocks is full. The runtime reserves a third before
    it gathers; a figure that skipped that would draw a scatter into a block nobody holds.
    """
    with pytest.raises(ValueError, match="the one being emitted"):
        draw.Batch(
            geometry=draw.DRAWN,
            rows=(draw.Row(name="A", blocks=(0, 1), length=8),),
            arena_blocks=4,
        )


def test_a_row_whose_blocks_do_not_match_its_length_is_refused():
    with pytest.raises(ValueError, match="blocks but"):
        draw.Batch(
            geometry=draw.DRAWN,
            rows=(draw.Row(name="A", blocks=(0, 1, 2, 3), length=9),),
            arena_blocks=8,
        )


def test_two_rows_cannot_be_drawn_holding_one_block():
    with pytest.raises(ValueError, match="same block"):
        draw.Batch(
            geometry=draw.DRAWN,
            rows=(
                draw.Row(name="A", blocks=(0,), length=3),
                draw.Row(name="B", blocks=(0,), length=2),
            ),
            arena_blocks=8,
        )


def test_a_block_outside_the_arena_is_refused():
    with pytest.raises(ValueError, match="outside the arena"):
        draw.Batch(
            geometry=draw.DRAWN,
            rows=(draw.Row(name="A", blocks=(9,), length=2),),
            arena_blocks=4,
        )


def test_a_drawn_sequence_holds_blocks_that_are_not_adjacent():
    """The first panel's whole subject. Tidying the blocks into a run would leave a
    picture that quietly says a sequence is contiguous in the arena, which is the thing
    block allocation is for not being."""
    batch = draw.BATCH

    def contiguous(blocks: tuple[int, ...]) -> bool:
        return tuple(sorted(blocks)) == tuple(range(min(blocks), max(blocks) + 1))

    scattered = [row for row in batch.rows if len(row.blocks) > 1 and not contiguous(row.blocks)]
    assert scattered, "no drawn sequence has a gap between its blocks"


def test_a_drawn_sequence_holds_a_block_it_has_not_filled():
    """Reserved-against-written is what the pale slots mean; a batch of exact multiples
    would draw every block full and the distinction would vanish."""
    batch = draw.BATCH
    assert any(row.length % batch.geometry.block_tokens for row in batch.rows), (
        "every drawn sequence ends on a block boundary"
    )


def test_a_drawn_row_pads_more_positions_than_it_copies():
    """The padding trap the second panel is for. Rows of equal length pad nothing and
    the hatched hole, and the reason the clear is timed apart from the copy,
    disappears from the figure."""
    batch = draw.BATCH
    assert any(batch.padding(row) > row.length for row in batch.rows)


def test_the_figure_lands_where_it_was_asked_for(tmp_path):
    path = tmp_path / "arena_geometry.png"
    draw.draw_arena_geometry(path)
    assert path.is_file()
    assert path.stat().st_size > 10_000, "a figure this size should not be nearly empty"


def test_drawing_writes_nothing_but_the_path_it_was_given(tmp_path):
    draw.draw_arena_geometry(tmp_path / "figure.png")
    assert [entry.name for entry in tmp_path.iterdir()] == ["figure.png"]
