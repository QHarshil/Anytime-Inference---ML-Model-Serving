// Block-allocated KV cache: a fixed arena, fixed-size blocks, and a free list.
//
// This is a host-side block allocator, not paged attention, and the distinction
// is structural rather than terminological. An optimum-exported decoder declares
// `past_key_values.{i}.{key,value}` as graph inputs and ONNX Runtime allocates the
// matching `present.{i}.*` outputs itself, sized `[batch, kv_heads, past +
// sequence, head_dim]`. There is no block table to hand such a graph and no hook
// by which attention could read scattered pages. What is reachable over a stock
// graph is exactly this: keep a sequence's KV in fixed blocks, gather those blocks
// into the batch-shaped tensor the graph expects before each run, and copy the new
// tail back out afterwards.
//
// The allocator does not make decoding faster. Feeding the `present` tensors
// straight back into the next run costs no gather at all and is the fastest thing
// available. What blocks buy is accounting: a fixed arena whose occupancy is
// known, so admission can refuse a sequence it cannot hold and eviction can pick a
// victim on evidence. On this host that accounting costs about 15% of a decode step
// at full context, which `scripts/profile_decode.py` measures against the
// no-gather reference rather than assuming.
//
// Two invariants of the exported graph are load-bearing here, both measured rather
// than assumed:
//
//   - One decode step from a gathered cache is bitwise identical to the same step
//     over contiguous KV. Only the source of the bytes differs, so anything less
//     than bitwise equality means the gather is corrupting something.
//   - `present[..., :past_len, :]` is bitwise equal to the `past` that was fed,
//     because the graph concatenates rather than rewriting. That is what lets
//     scatter copy only the new tail instead of the whole tensor, and it is
//     verified per sequence rather than trusted; see decoder.hpp.

#ifndef ANYTIME_KV_CACHE_HPP
#define ANYTIME_KV_CACHE_HPP

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace anytime {

// Raised when a sequence needs a block the arena cannot supply. Distinct from a
// generic failure because it is the one runtime error the admission policy is
// expected to handle rather than propagate: the answer is to evict or refuse, not
// to abort. Derives from RuntimeError on the Python side, so the error contract in
// serving/onnx_runtime.py still holds.
class CacheExhausted : public std::runtime_error {
public:
    explicit CacheExhausted(const std::string& what) : std::runtime_error(what) {}
};

// Which half of a layer's cache a slab holds. Ordering matters: it indexes the
// arena, so Key must stay 0.
enum class KvKind : int { Key = 0, Value = 1 };

// Shape of one decoder's cache, derived from the graph rather than from a model
// config. `decoder.cpp` counts the `past_key_values.{i}` inputs for `layers` and
// reads `kv_heads` and `head_dim` off the static dimensions of the first one; a
// config could disagree with the graph it is meant to describe.
struct KvGeometry {
    int layers = 0;
    int kv_heads = 0;
    int head_dim = 0;
    int block_tokens = 0;

    // One (layer, kind) slab of one block, as float count. This is the unit both
    // gather and scatter address.
    std::size_t floats_per_slab() const {
        return static_cast<std::size_t>(kv_heads) * static_cast<std::size_t>(block_tokens) *
               static_cast<std::size_t>(head_dim);
    }

    // A block holds `block_tokens` token positions across every layer, for both
    // key and value. That is deliberately the unit admission reasons about: "this
    // sequence needs 16 blocks" is a decision, whereas "this sequence needs 384
    // per-layer slabs" is bookkeeping. For GPT-2 at block_tokens = 64 a block is
    // 4.5 MiB and a 1024-token sequence is 16 of them.
    std::size_t floats_per_block() const {
        return 2 * static_cast<std::size_t>(layers) * floats_per_slab();
    }

    std::size_t floats_per_token() const {
        return 2 * static_cast<std::size_t>(layers) * static_cast<std::size_t>(kv_heads) *
               static_cast<std::size_t>(head_dim);
    }

    std::size_t bytes_per_block() const { return floats_per_block() * sizeof(float); }
    std::size_t bytes_per_token() const { return floats_per_token() * sizeof(float); }

    // Blocks a sequence of `tokens` positions occupies. Ceiling division: a
    // partially filled block is still held, which is the space blocks trade for
    // not having to move a growing sequence around.
    std::size_t blocks_for(int tokens) const;

    // Throws if the geometry could not describe a real cache. Called once at
    // construction so a malformed graph fails there rather than mid-gather.
    void validate() const;
};

// The arena. One contiguous allocation carved into equal blocks, with a free list
// on top.
//
// Fragmentation is not a concern and that is the point of fixed blocks: a
// sequence's blocks need not be adjacent, so any free block serves any request and
// the arena cannot reach a state where free space exists but nothing fits. The free
// list is a stack, so a released block is the next one handed out and stays warm in
// cache.
class BlockPool {
public:
    BlockPool(const KvGeometry& geometry, std::size_t num_blocks);

    const KvGeometry& geometry() const { return geometry_; }
    std::size_t capacity_blocks() const { return capacity_; }
    std::size_t free_blocks() const { return free_.size(); }
    std::size_t bytes() const { return arena_.size() * sizeof(float); }

    // Takes `count` blocks, or none at all. All-or-nothing because a partial
    // allocation would leave a sequence that cannot run and blocks that cannot be
    // attributed to anyone.
    std::vector<std::uint32_t> allocate(std::size_t count);
    void release(const std::vector<std::uint32_t>& blocks);

    float* block(std::uint32_t index);
    const float* block(std::uint32_t index) const;

private:
    KvGeometry geometry_;
    std::size_t capacity_;
    std::vector<float> arena_;
    std::vector<std::uint32_t> free_;
};

// One sequence's residency: the blocks it holds, in token order, and how many
// token positions are actually written. `blocks.size() * block_tokens` is what it
// reserved; `length` is what it has filled.
struct SequenceCache {
    std::vector<std::uint32_t> blocks;
    int length = 0;

    int capacity_tokens(int block_tokens) const {
        return static_cast<int>(blocks.size()) * block_tokens;
    }
};

// Copies `length` token positions of one (layer, kind) out of the sequence's
// blocks into `dest`, laid out as the graph's `[1, kv_heads, length, head_dim]`
// input expects.
//
// Both source and destination are contiguous within a (head, block) pair, so this
// is one memcpy per pair rather than per token: `kv_heads * ceil(length /
// block_tokens)` copies of up to `block_tokens * head_dim` floats. At GPT-2's
// geometry with 64-token blocks that is 16 KiB a copy, which is large enough to
// run at memory bandwidth.
void gather(const BlockPool& pool, const SequenceCache& sequence, int layer, KvKind kind,
            int length, float* dest);

// Copies `count` new token positions, starting at position `start`, out of a
// `present` tensor shaped `[1, kv_heads, present_length, head_dim]` and into the
// sequence's blocks.
//
// Only the tail is copied. The prefix of `present` is bitwise equal to the `past`
// that produced it, which is already in the blocks, so rewriting it would move
// tens of megabytes to no effect. `prefix_matches` below is what keeps that from
// being an assumption.
void scatter(BlockPool& pool, const SequenceCache& sequence, int layer, KvKind kind, int start,
             int count, const float* present, int present_length);

// Whether the first `length` positions of `present` equal what the blocks hold for
// this (layer, kind). Bitwise: these are the same bytes by construction, so any
// difference means the graph is not concatenating the way the scatter above
// assumes.
bool prefix_matches(const BlockPool& pool, const SequenceCache& sequence, int layer, KvKind kind,
                    int length, const float* present, int present_length);

}  // namespace anytime

#endif  // ANYTIME_KV_CACHE_HPP
