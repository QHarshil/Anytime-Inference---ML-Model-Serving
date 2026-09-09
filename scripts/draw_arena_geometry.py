"""Draw the arena's geometry: blocks, the gather into the padded batch tensor, the scatter.

This is a schematic, not a plot. It reads no measurement file and it draws no
measured quantity, which is why it lives here instead of in `plot_batching.py`, whose
`main` loads `results/*.json` and has nothing to hand a figure that takes no input.

Writes, into `docs/img/`:

  arena_geometry.png  where a token's KV lives, what the gather has to do to hand it to
                      the graph, and how much of that work comes back

Why this one is a rendered figure
---------------------------------

`architecture.md` draws layered stacks as text and graphs-with-cycles as Mermaid, and
says so. This is neither. It is spatial and proportional: the point of it is that a
sequence's blocks are scattered through the arena while its tokens are contiguous in the
staged tensor, and that the padding is a hole whose size depends on the batch's length
spread. Mermaid lays out a graph, so it draws neither of those; ASCII can hold a grid but
not the proportions. So it is drawn, committed, and referenced like the measured figures.

Colour means one thing here
---------------------------

In this figure a hue names a sequence and nothing else. The phases are named by the panel
titles and drawn in chrome, padding is a hatched hole, arrows are ink, so no reader
has to work out whether a colour means "sequence B" or "the pad phase". That is the
opposite convention from `step_composition.png`, where hue is the phase, and the two do
not meet: this figure has no phase colours and that one has no sequences.

The three sequence hues are Okabe-Ito steps, the same published palette
`plot_batching.PHASE_COLOURS` draws from, at 5.19:1, 3.87:1 and 3.42:1 against the white
surface. They are the same three hexes that figure gives `run`, `gather` and `pad`, which
is accepted, not overlooked: the two figures live in different documents, encode
different things, and every region here carries its sequence letter inside it, so colour
is a second cue instead of the only one.

The numbers are illustrative, the caption is not
------------------------------------------------

The drawing uses four token positions per block and a twelve-block arena, because sixty-
four and a real arena draw as a grey smear. Every figure quoted in the caption is GPT-2's
actual geometry, computed from `GPT2` below instead of typed, and
`tests/test_draw_arena_geometry.py` pins that against the extension's own constant and
against the geometry `tests/test_decoder_session.py` reads off the exported graph.

Usage:
    python scripts/draw_arena_geometry.py
    python scripts/draw_arena_geometry.py --output-dir /tmp/figures
"""

from __future__ import annotations

import argparse
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from plot_decode_profiles import GRID, INK, INK_MUTED, _save  # noqa: E402

from anytime_serving.utils.logger import get_logger  # noqa: E402

LOGGER = get_logger("scripts.draw_arena_geometry")

# A hue names a sequence. See the module docstring for the palette and the trade-off it
# accepts against step_composition.png.
SEQUENCE_COLOURS = ("#0072B2", "#D55E00", "#009E73")
# Reserved but not yet written, and the free list. Both are absence, so neither gets a
# hue: one is the owner's colour gone pale, the other is bare surface.
UNWRITTEN_ALPHA = 0.22
FREE = "#f4f3ee"
# Padding is drawn, not filled: it is a hole the clear pass has to walk. The hatch sits
# between the grid and the muted ink so it reads as absence instead of as a fourth
# category competing with the three sequences.
HATCH = "#b9b7ad"


@dataclass(frozen=True)
class Geometry:
    """One decoder's cache shape, mirroring `KvGeometry` in `runtime/include/anytime`.

    Mirrored, not imported: the extension exposes `KvGeometry` with no
    constructor, because a real one is read off a loaded graph, and a drawing script
    should not need a 500 MB ONNX file and a session to caption a schematic. The test
    checks this against the extension's `DEFAULT_BLOCK_TOKENS` and against the values
    `tests/test_decoder_session.py` asserts the exported GPT-2 graph actually has.
    """

    layers: int
    kv_heads: int
    head_dim: int
    block_tokens: int

    @property
    def slots(self) -> int:
        """Graph past inputs, and so staging buffers: one per layer per key/value half.

        This is what the gather is split across, which is why it is a property of the
        geometry instead of of the batch. It does not depend on how many rows there are.
        """
        return self.layers * 2

    @property
    def floats_per_slab(self) -> int:
        """One (layer, kind) piece of one block: the unit gather and scatter address."""
        return self.kv_heads * self.block_tokens * self.head_dim

    @property
    def floats_per_block(self) -> int:
        return 2 * self.layers * self.floats_per_slab

    @property
    def floats_per_token(self) -> int:
        return 2 * self.layers * self.kv_heads * self.head_dim

    @property
    def bytes_per_block(self) -> int:
        return self.floats_per_block * 4

    @property
    def bytes_per_token(self) -> int:
        return self.floats_per_token * 4

    def blocks_for(self, tokens: int) -> int:
        """Ceiling division, as `KvGeometry::blocks_for` does it.

        A partly filled block is still held. That is the space blocks trade for never
        having to move a growing sequence, and it is the reason a tail block is drawn
        pale instead of short.
        """
        if tokens < 0:
            raise ValueError(f"a sequence cannot hold {tokens} tokens")
        return (tokens + self.block_tokens - 1) // self.block_tokens


# What the caption quotes. Twelve layers, twelve KV heads, head dim 64 is what
# `tests/test_decoder_session.py` reads off the exported graph; 64 token positions per
# block is `kDefaultBlockTokens`.
GPT2 = Geometry(layers=12, kv_heads=12, head_dim=64, block_tokens=64)

# What the picture draws. Sixty-four positions across a real arena is a grey smear, so
# the drawing shrinks both and the caption carries the real ones.
DRAWN = Geometry(layers=12, kv_heads=12, head_dim=64, block_tokens=4)
ARENA_BLOCKS = 12


@dataclass(frozen=True)
class Row:
    """One sequence's residency, mirroring `SequenceCache`.

    `blocks` is arena indices in token order, which is the whole point of the first
    panel: the order a sequence reads its blocks in has nothing to do with where they
    sit in the arena. `length` is positions written; `len(blocks) * block_tokens` is
    positions reserved.
    """

    name: str
    blocks: tuple[int, ...]
    length: int


@dataclass(frozen=True)
class Batch:
    """The three sequences the figure draws, and everything derived from them.

    Every number the drawing places is taken from here instead of written into the
    drawing code, so the test can assert the arithmetic without rendering anything.
    """

    geometry: Geometry
    rows: tuple[Row, ...]
    arena_blocks: int

    def __post_init__(self) -> None:
        for row in self.rows:
            # `length + 1`, not `length`, and the +1 is the whole reason the third panel
            # has a slot to point at. `decode_batch` reserves `blocks_for(length + 1)`
            # before it gathers, because every row grows by one token; a row drawn with
            # only the blocks its current tokens need would have nowhere to scatter to
            # the moment its length landed on a block boundary.
            needed = self.geometry.blocks_for(row.length + 1)
            if needed != len(row.blocks):
                raise ValueError(
                    f"sequence {row.name} holds {len(row.blocks)} blocks but "
                    f"{row.length} tokens and the one being emitted need {needed}"
                )
        held = [block for row in self.rows for block in row.blocks]
        if len(set(held)) != len(held):
            raise ValueError("two sequences are drawn holding the same block")
        if held and max(held) >= self.arena_blocks:
            raise ValueError("a sequence is drawn holding a block outside the arena")

    @property
    def max_past(self) -> int:
        """The batch's longest past. Every row is right-padded out to this."""
        return max(row.length for row in self.rows)

    @property
    def total(self) -> int:
        """Past width plus the one token being emitted: the attention mask's width."""
        return self.max_past + 1

    def padding(self, row: Row) -> int:
        """Token positions this row pads, which exist only because the batch shares one
        `past_sequence_length`. Zero when the row is the longest."""
        return self.max_past - row.length

    def owner(self, block: int) -> Row | None:
        for row in self.rows:
            if block in row.blocks:
                return row
        return None

    def written_in(self, row: Row, ordinal: int) -> int:
        """Token positions written in the `ordinal`-th block this row holds.

        Full for every block but the last, and the remainder in the last, which is
        `block_tokens` instead of zero when the length lands exactly on a boundary.
        """
        start = ordinal * self.geometry.block_tokens
        return max(0, min(self.geometry.block_tokens, row.length - start))

    def landing(self, row: Row) -> tuple[int, int, int]:
        """Where this row's new token is scattered: (ordinal, offset, arena block).

        Its home is `past_len`, not `max_past`. The present tensor is the other way
        round, the new KV sits at index `max_past` there for every row, because that
        is how wide the batch made its past, and keeping the two apart is the whole
        content of the third panel.
        """
        ordinal, offset = divmod(row.length, self.geometry.block_tokens)
        return ordinal, offset, row.blocks[ordinal]


# Three sequences, chosen to show the four things the panels are for: blocks that are
# not adjacent, a tail block that is held but not full, a batch whose rows disagree
# about length, and a row that pads more positions than it copies.
BATCH = Batch(
    geometry=DRAWN,
    rows=(
        Row(name="A", blocks=(0, 2, 5), length=11),
        Row(name="B", blocks=(1, 4), length=5),
        Row(name="C", blocks=(3, 6, 8), length=9),
    ),
    arena_blocks=ARENA_BLOCKS,
)


# Drawing units. A token slot is one unit wide, but only within a panel: the first draws
# a whole arena and the others draw one batch, so the same unit lands at a different size
# on the page in each. Colour and the row labels are what carry a sequence from panel to
# panel; slot width is not, and is not asked to.
#
# Blocks are separated by surface. Token positions inside a staged row are not, because
# there the tensor really is contiguous, that difference is the first panel's subject.
SLOT = 1.0
BLOCK_GAP = 0.5
ROW_HEIGHT = 0.62
MASK_HEIGHT = 0.30


def _cell(axis: Any, x: float, y: float, width: float, height: float, **kwargs: Any) -> None:
    from matplotlib.patches import Rectangle

    axis.add_patch(Rectangle((x, y), width, height, **kwargs))


def _panel(axis: Any, title: str) -> None:
    """A schematic panel: a title in the house style and no axes at all.

    `plot_decode_profiles._style` is not used here, and that is deliberate instead of an
    oversight. It fits a plot, ticks, a grid, two spines, and every one of those would
    be furniture around a drawing that has no coordinates a reader should read.
    """
    axis.set_title(title, color=INK, fontsize=10, loc="left")
    axis.set_axis_off()


NOTE_COLUMNS = 116


def _note(axis: Any, y: float, text: str, **kwargs: Any) -> None:
    """Prose under a panel: left margin in axes fractions, baseline in data units.

    The blend is the point. Each panel sets its own data limits: the arena is four
    times as wide in slots as one staged row, so a note placed at a data x lands at a
    different left margin in each panel, and three paragraphs that should form one column
    down the figure come out ragged. Axes fractions fix the margin; the baseline still
    has to follow the drawing, which is in data units.

    Matplotlib does not wrap either, and an unwrapped line runs off the figure at any
    width the author did not happen to test at, so the line length is fixed in characters
    here. Each paragraph is wrapped separately and rejoined with a blank line; one
    `textwrap.fill` over the lot would reflow them into a single block.
    """
    from matplotlib.transforms import blended_transform_factory

    wrapped = "\n\n".join(
        textwrap.fill(paragraph, width=NOTE_COLUMNS) for paragraph in text.split("\n")
    )
    axis.text(
        0.0,
        y,
        wrapped,
        color=INK_MUTED,
        fontsize=8,
        va="top",
        linespacing=1.45,
        transform=blended_transform_factory(axis.transAxes, axis.transData),
        **kwargs,
    )


def _block_span(count: int) -> float:
    """Page width of `count` blocks laid side by side with surface between them."""
    geometry = BATCH.geometry
    return count * geometry.block_tokens * SLOT + max(0, count - 1) * BLOCK_GAP


def _draw_block(
    axis: Any,
    x: float,
    y: float,
    *,
    written: int,
    colour: str | None,
    height: float = ROW_HEIGHT,
) -> None:
    """One block as its token slots: `written` of them filled, the rest held but empty.

    A free block is drawn the same size as an owned one instead of omitted, because the
    arena is a fixed allocation and the free list is space that exists.
    """
    slots = BATCH.geometry.block_tokens
    for slot in range(slots):
        if colour is None:
            face, alpha = FREE, 1.0
        elif slot < written:
            face, alpha = colour, 1.0
        else:
            face, alpha = colour, UNWRITTEN_ALPHA
        _cell(
            axis,
            x + slot * SLOT,
            y,
            SLOT,
            height,
            facecolor=face,
            alpha=alpha,
            edgecolor="white",
            linewidth=0.7,
        )
    _cell(
        axis,
        x,
        y,
        slots * SLOT,
        height,
        facecolor="none",
        edgecolor=GRID if colour is None else INK_MUTED,
        linewidth=0.9,
    )


def draw_arena(axis: Any, batch: Batch) -> None:
    """Panel one: the arena, and the fact that a sequence's blocks are scattered in it."""
    _panel(axis, "The arena: one allocation, equal blocks, a free list")

    colours = {row.name: SEQUENCE_COLOURS[i] for i, row in enumerate(batch.rows)}
    slots = batch.geometry.block_tokens
    width = slots * SLOT
    for index in range(batch.arena_blocks):
        x = index * (width + BLOCK_GAP)
        owner = batch.owner(index)
        if owner is None:
            _draw_block(axis, x, 0.0, written=0, colour=None)
            label, colour = "free", INK_MUTED
        else:
            ordinal = owner.blocks.index(index)
            _draw_block(
                axis, x, 0.0, written=batch.written_in(owner, ordinal), colour=colours[owner.name]
            )
            label, colour = f"{owner.name}{ordinal}", colours[owner.name]
        axis.text(x + width / 2, ROW_HEIGHT + 0.14, label, color=colour, fontsize=8.5, ha="center")
        axis.text(
            x + width / 2, -0.16, str(index), color=INK_MUTED, fontsize=7.5, ha="center", va="top"
        )

    span = _block_span(batch.arena_blocks)
    order = ", ".join(str(block) for block in batch.rows[0].blocks)
    free = ", ".join(str(i) for i in range(batch.arena_blocks) if batch.owner(i) is None)
    first = batch.rows[0]
    _note(
        axis,
        -0.62,
        f"{first.name} reads its blocks in the order {order}, which is not the order they "
        f"sit in. Any free block serves any request, so the arena cannot reach a state "
        f"where space exists and nothing fits.\n"
        f"Pale slots are held and not yet written: {first.name} reserved "
        f"{len(first.blocks) * slots} positions to hold {first.length}. Free: {free}.",
    )
    axis.set_xlim(-0.5, span + 0.5)
    axis.set_ylim(-2.0, ROW_HEIGHT + 0.6)


NEW_TOKEN_GAP = 0.9
ROW_PITCH = ROW_HEIGHT + MASK_HEIGHT + 0.62


def draw_gather(axis: Any, batch: Batch) -> None:
    """Panel two: the staged tensor the graph is handed, and the hole padding leaves."""
    _panel(
        axis,
        "The gather: each row's blocks into one staged tensor, right-padded to the batch's longest",
    )

    colours = {row.name: SEQUENCE_COLOURS[i] for i, row in enumerate(batch.rows)}
    max_past = batch.max_past
    new_x = max_past * SLOT + NEW_TOKEN_GAP

    for index, row in enumerate(batch.rows):
        colour = colours[row.name]
        kv_y = -index * ROW_PITCH
        mask_y = kv_y - MASK_HEIGHT - 0.10

        for position in range(row.length):
            _cell(
                axis,
                position * SLOT,
                kv_y,
                SLOT,
                ROW_HEIGHT,
                facecolor=colour,
                edgecolor="white",
                linewidth=0.7,
            )
        padding = batch.padding(row)
        if padding:
            # One rectangle instead of one per position. The padding is a single hole
            # that gets cleared in a single pass, and cell-by-cell hatching reads as
            # texture instead of as absence.
            _cell(
                axis,
                row.length * SLOT,
                kv_y,
                padding * SLOT,
                ROW_HEIGHT,
                facecolor="white",
                edgecolor=HATCH,
                linewidth=0.0,
                hatch="//",
            )
            # The border is a second patch so the hatch can be paler than the outline.
            # Matplotlib draws both in `edgecolor`, and one colour dark enough to bound
            # the hole makes a texture that outshouts the copied tokens beside it.
            _cell(
                axis,
                row.length * SLOT,
                kv_y,
                padding * SLOT,
                ROW_HEIGHT,
                facecolor="none",
                edgecolor=INK_MUTED,
                linewidth=0.8,
            )
        _cell(
            axis,
            new_x,
            kv_y,
            SLOT,
            ROW_HEIGHT,
            facecolor="white",
            edgecolor=INK,
            linewidth=1.2,
        )

        for position in range(max_past + 1):
            held = position < row.length
            x = new_x if position == max_past else position * SLOT
            _cell(
                axis,
                x,
                mask_y,
                SLOT,
                MASK_HEIGHT,
                facecolor="white",
                edgecolor=GRID,
                linewidth=0.7,
            )
            digit = "1" if held or position == max_past else "0"
            axis.text(
                x + SLOT / 2,
                mask_y + MASK_HEIGHT / 2,
                digit,
                color=INK if digit == "1" else INK_MUTED,
                fontsize=6.5,
                ha="center",
                va="center",
            )

        axis.text(
            -0.5,
            kv_y + ROW_HEIGHT / 2,
            row.name,
            color=colour,
            fontsize=10,
            ha="right",
            va="center",
        )
        axis.text(
            -0.5,
            mask_y + MASK_HEIGHT / 2,
            "mask",
            color=INK_MUTED,
            fontsize=7,
            ha="right",
            va="center",
        )
        summary = f"{row.length} copied"
        if padding:
            summary += f", {padding} padded"
        axis.text(
            new_x + SLOT + 0.5,
            kv_y + ROW_HEIGHT / 2,
            summary,
            color=INK_MUTED,
            fontsize=8,
            va="center",
        )

    top = ROW_HEIGHT + 0.30
    axis.annotate(
        "",
        xy=(0, top),
        xytext=(max_past * SLOT, top),
        arrowprops={"arrowstyle": "<->", "color": INK_MUTED, "linewidth": 0.9},
    )
    axis.text(
        max_past * SLOT / 2,
        top + 0.10,
        f"past_sequence_length = max_past = {max_past}",
        color=INK_MUTED,
        fontsize=8,
        ha="center",
    )
    axis.text(
        new_x + SLOT / 2,
        top + 0.10,
        "the token\nbeing emitted",
        color=INK_MUTED,
        fontsize=8,
        ha="center",
    )

    worst = max(batch.rows, key=batch.padding)
    bottom = -(len(batch.rows) - 1) * ROW_PITCH - MASK_HEIGHT - 0.55
    _note(
        axis,
        bottom,
        f"One slot's staging buffer, shape [batch, kv_heads, max_past, head_dim]. There "
        f"are layers x 2 = {batch.geometry.slots} of them on GPT-2, one per graph past "
        f"input, and the gather is split across slots instead of across rows: a batch may "
        f"be 1, and there are always {batch.geometry.slots}.\n"
        f"The mask is 1 over the positions a row holds, 0 over its padding and 1 for the "
        f"token being emitted, so the padding sits between the two. {worst.name} pads "
        f"{batch.padding(worst)} positions to copy {worst.length}.",
    )
    axis.set_xlim(-2.2, new_x + SLOT + 5.0)
    axis.set_ylim(bottom - 2.7, top + 0.75)


SCATTER_PITCH = ROW_HEIGHT + 0.98
SCATTER_SEPARATION = 4.5
# How far a scatter arrow bows above its own row. It is a height, not a curvature,
# on purpose. matplotlib's arc3 `rad` is a ratio, so one value shared by three
# arrows of different lengths bows the longest one clean off the figure and sends it
# through the rows above on the way. Fixing the height and solving for `rad` per arrow
# keeps every one of them inside the gutter above the row it belongs to, which is what
# stops the reader seeing B scatter into A's blocks.
SCATTER_BOW = 0.40


def draw_scatter(axis: Any, batch: Batch) -> None:
    """Panel three: one token position per row, from present index max_past to past_len."""
    _panel(axis, "The scatter: the new tail only, from max_past back to each row's past_len")

    from matplotlib.patches import FancyArrowPatch

    colours = {row.name: SEQUENCE_COLOURS[i] for i, row in enumerate(batch.rows)}
    slots = batch.geometry.block_tokens
    right = batch.total * SLOT + SCATTER_SEPARATION

    for index, row in enumerate(batch.rows):
        colour = colours[row.name]
        y = -index * SCATTER_PITCH

        # The present row, contiguous: the past it was fed, then the token it produced.
        # Unlike the staged tensor in the panel above, the emitted position is part of
        # this tensor instead of a separate input, so nothing is set apart here.
        for position in range(batch.total):
            if position == batch.max_past:
                continue
            padded = position >= row.length
            _cell(
                axis,
                position * SLOT,
                y,
                SLOT,
                ROW_HEIGHT,
                facecolor="white",
                edgecolor=HATCH if padded else GRID,
                linewidth=0.7,
                hatch="//" if padded else None,
            )
        _cell(
            axis,
            batch.max_past * SLOT,
            y,
            SLOT,
            ROW_HEIGHT,
            facecolor=colour,
            edgecolor=INK,
            linewidth=1.1,
        )
        _cell(
            axis,
            0.0,
            y,
            batch.total * SLOT,
            ROW_HEIGHT,
            facecolor="none",
            edgecolor=INK_MUTED,
            linewidth=0.8,
        )

        ordinal, offset, arena_block = batch.landing(row)
        for held, block in enumerate(row.blocks):
            x = right + held * (slots * SLOT + BLOCK_GAP)
            _draw_block(axis, x, y, written=batch.written_in(row, held), colour=colour)
            axis.text(
                x + slots * SLOT / 2,
                y - 0.16,
                str(block),
                color=INK_MUTED,
                fontsize=7.5,
                ha="center",
                va="top",
            )

        landing_x = right + ordinal * (slots * SLOT + BLOCK_GAP) + offset * SLOT
        _cell(
            axis,
            landing_x,
            y,
            SLOT,
            ROW_HEIGHT,
            facecolor="white",
            edgecolor=INK,
            linewidth=1.3,
        )
        source_x = batch.max_past * SLOT + SLOT / 2
        target_x = landing_x + SLOT / 2
        # arc3 puts its control point `rad * distance` off the midpoint, so the curve
        # peaks at half of that. Invert it for the height wanted.
        rad = -2.0 * SCATTER_BOW / max(target_x - source_x, SLOT)
        axis.add_patch(
            FancyArrowPatch(
                (source_x, y + ROW_HEIGHT + 0.05),
                (target_x, y + ROW_HEIGHT + 0.05),
                connectionstyle=f"arc3,rad={rad:.4f}",
                arrowstyle="-|>",
                mutation_scale=11,
                color=colour,
                linewidth=1.1,
                shrinkA=2,
                shrinkB=2,
            )
        )
        axis.text(
            -0.5, y + ROW_HEIGHT / 2, row.name, color=colour, fontsize=10, ha="right", va="center"
        )
        axis.text(
            right + _block_span(len(row.blocks)) + 0.6,
            y + ROW_HEIGHT / 2,
            f"past_len {row.length} -> block {arena_block}, slot {offset}",
            color=INK_MUTED,
            fontsize=8,
            va="center",
        )

    top = ROW_HEIGHT + 1.05
    axis.text(
        batch.total * SLOT / 2,
        top,
        f"present, width total = max_past + 1 = {batch.total}",
        color=INK_MUTED,
        fontsize=8,
        ha="center",
    )
    axis.text(
        right + _block_span(3) / 2,
        top,
        "the blocks the sequence holds",
        color=INK_MUTED,
        fontsize=8,
        ha="center",
    )

    bottom = -(len(batch.rows) - 1) * SCATTER_PITCH - 0.75
    _note(
        axis,
        bottom,
        f"The new KV is at present index max_past = {batch.max_past} for every row, because "
        f"that is how wide the batch made its past; its home is past_len, which differs per "
        f"row. Hatched positions are padding the graph concatenated through.\n"
        f"Everything before it is left alone, so the scatter moves one token position per "
        f"row where the gather moved up to {batch.max_past}. The gather's loop over the "
        f"{batch.geometry.slots} slots is the threaded one; the scatter's is serial.",
    )
    axis.set_xlim(-2.2, right + _block_span(3) + 11.0)
    axis.set_ylim(bottom - 2.5, top + 0.5)


def draw_arena_geometry(path: Path, batch: Batch = BATCH, caption: Geometry = GPT2) -> None:
    """Draw all three panels into one figure and write it.

    The panels are stacked instead of placed side by side because they are a sequence:
    a token's KV is somewhere, then it is somewhere else so the graph can read it, then
    what came back goes home. Read left to right that would be three unrelated pictures.
    """
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(
        3,
        1,
        figsize=(12.5, 12.4),
        # Height ratios only. `hspace` here would be doing nothing except emitting a
        # warning: `_save` runs tight_layout, which recomputes the spacing between rows
        # and then reports the figure as incompatible with it. The gap under a panel is
        # set by that panel's own lower y limit, which tight_layout leaves alone.
        gridspec_kw={"height_ratios": (0.95, 1.55, 1.35)},
    )
    draw_arena(axes[0], batch)
    draw_gather(axes[1], batch)
    draw_scatter(axes[2], batch)

    figure.suptitle(
        "How a decode step reaches its KV cache", color=INK, fontsize=12.5, x=0.011, ha="left"
    )
    figure.text(
        0.011,
        0.012,
        textwrap.fill(
            f"Schematic: {batch.geometry.block_tokens} token positions per block and a "
            f"{batch.arena_blocks}-block arena, because the real ones draw as a grey smear. "
            f"GPT-2 is {caption.layers} layers, {caption.kv_heads} KV heads, head dim "
            f"{caption.head_dim}, {caption.block_tokens} positions per block: "
            f"{caption.bytes_per_token / 1024:.0f} KiB per token, "
            f"{caption.bytes_per_block / 1024**2:.1f} MiB per block, and "
            f"{caption.blocks_for(1024)} blocks for a 1024-token sequence. Geometry is read "
            f"off the loaded graph instead of a model config, because a config can "
            f"disagree with the graph it is meant to describe.",
            width=150,
        ),
        color=INK_MUTED,
        fontsize=8,
        va="bottom",
        linespacing=1.45,
    )
    _save(figure, path, top=0.965, bottom=0.055)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("docs/img"),
        help="Where the figure is written",
    )
    args = parser.parse_args()

    try:
        import matplotlib
    except ImportError as error:  # pragma: no cover - depends on the environment
        raise SystemExit(
            "matplotlib is required to draw figures. Install it with:\n"
            '    pip install -e ".[bench]"'
        ) from error
    matplotlib.use("Agg")
    # The default 1.0 draws a padding hole heavier than the tokens beside it.
    matplotlib.rcParams["hatch.linewidth"] = 0.6

    draw_arena_geometry(args.output_dir / "arena_geometry.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
