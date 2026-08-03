#include "anytime/kv_cache.hpp"

#include <algorithm>
#include <cstring>

namespace anytime {
namespace {

// Offset of one (layer, kind) slab within a block. Key and value are separate
// slabs because the graph declares them as separate inputs, so keeping them apart
// means gather writes one destination per graph input with no interleaving.
std::size_t slab_offset(const KvGeometry& geometry, int layer, KvKind kind) {
    const std::size_t index =
        static_cast<std::size_t>(kind) * static_cast<std::size_t>(geometry.layers) +
        static_cast<std::size_t>(layer);
    return index * geometry.floats_per_slab();
}

void check_layer(const KvGeometry& geometry, int layer) {
    if (layer < 0 || layer >= geometry.layers) {
        throw std::invalid_argument("layer " + std::to_string(layer) + " is outside 0.." +
                                    std::to_string(geometry.layers - 1));
    }
}

// A gather or scatter walking past the blocks a sequence holds would read or write
// another sequence's cache, which would not fail loudly on its own.
void check_span(const KvGeometry& geometry, const SequenceCache& sequence, int last_token) {
    const int capacity = sequence.capacity_tokens(geometry.block_tokens);
    if (last_token > capacity) {
        throw std::out_of_range("token position " + std::to_string(last_token) +
                                " exceeds the " + std::to_string(capacity) +
                                " positions this sequence reserved (" +
                                std::to_string(sequence.blocks.size()) + " block(s))");
    }
}

}  // namespace

std::size_t KvGeometry::blocks_for(int tokens) const {
    if (tokens < 0) {
        throw std::invalid_argument("token count must not be negative");
    }
    if (block_tokens <= 0) {
        throw std::invalid_argument("block_tokens must be positive");
    }
    return (static_cast<std::size_t>(tokens) + static_cast<std::size_t>(block_tokens) - 1) /
           static_cast<std::size_t>(block_tokens);
}

void KvGeometry::validate() const {
    if (layers <= 0) {
        throw std::invalid_argument("KV geometry needs at least one layer, got " +
                                    std::to_string(layers));
    }
    if (kv_heads <= 0) {
        throw std::invalid_argument("KV geometry needs at least one kv head, got " +
                                    std::to_string(kv_heads));
    }
    if (head_dim <= 0) {
        throw std::invalid_argument("KV geometry needs a positive head dimension, got " +
                                    std::to_string(head_dim));
    }
    if (block_tokens <= 0) {
        throw std::invalid_argument("block_tokens must be positive, got " +
                                    std::to_string(block_tokens));
    }
}

BlockPool::BlockPool(const KvGeometry& geometry, std::size_t num_blocks)
    : geometry_(geometry), capacity_(num_blocks) {
    geometry_.validate();
    if (num_blocks == 0) {
        throw std::invalid_argument("a block pool with no blocks can hold no sequences");
    }

    // Zero-initialised, which the std::vector constructor does anyway. Worth
    // keeping deliberately: it makes every arena page resident before the first
    // run, so first-touch page faults do not land inside the opening decode steps
    // and show up as latency that has nothing to do with inference.
    arena_.assign(num_blocks * geometry_.floats_per_block(), 0.0F);

    // Reversed, so the first allocation hands out block 0 upward. The free list is
    // a stack: a released block is the next one reused and is still warm.
    free_.reserve(num_blocks);
    for (std::size_t i = num_blocks; i-- > 0;) {
        free_.push_back(static_cast<std::uint32_t>(i));
    }
}

std::vector<std::uint32_t> BlockPool::allocate(std::size_t count) {
    if (count > free_.size()) {
        return {};
    }
    std::vector<std::uint32_t> taken;
    taken.reserve(count);
    for (std::size_t i = 0; i < count; ++i) {
        taken.push_back(free_.back());
        free_.pop_back();
    }
    return taken;
}

void BlockPool::release(const std::vector<std::uint32_t>& blocks) {
    for (const std::uint32_t index : blocks) {
        if (index >= capacity_) {
            throw std::out_of_range("cannot release block " + std::to_string(index) +
                                    "; the pool holds " + std::to_string(capacity_));
        }
        free_.push_back(index);
    }
}

float* BlockPool::block(std::uint32_t index) {
    if (index >= capacity_) {
        throw std::out_of_range("block " + std::to_string(index) + " is outside a pool of " +
                                std::to_string(capacity_));
    }
    return arena_.data() + static_cast<std::size_t>(index) * geometry_.floats_per_block();
}

const float* BlockPool::block(std::uint32_t index) const {
    if (index >= capacity_) {
        throw std::out_of_range("block " + std::to_string(index) + " is outside a pool of " +
                                std::to_string(capacity_));
    }
    return arena_.data() + static_cast<std::size_t>(index) * geometry_.floats_per_block();
}

void gather(const BlockPool& pool, const SequenceCache& sequence, int layer, KvKind kind,
            int length, float* dest, int dest_row_tokens) {
    const KvGeometry& geometry = pool.geometry();
    check_layer(geometry, layer);
    if (length < 0) {
        throw std::invalid_argument("gather length must not be negative");
    }
    if (dest_row_tokens < length) {
        throw std::invalid_argument("gather destination row holds " +
                                    std::to_string(dest_row_tokens) +
                                    " token position(s), which cannot take " +
                                    std::to_string(length) + " gathered token(s)");
    }
    check_span(geometry, sequence, length);
    if (dest_row_tokens == 0) {
        // A prefill from cold. The graph accepts zero-length past tensors, so
        // there is nothing to copy, nothing to pad, and nothing to complain about.
        return;
    }
    if (dest == nullptr) {
        throw std::invalid_argument("gather needs a destination buffer");
    }

    const std::size_t slab = slab_offset(geometry, layer, kind);
    const std::size_t head_dim = static_cast<std::size_t>(geometry.head_dim);
    const std::size_t block_stride = static_cast<std::size_t>(geometry.block_tokens) * head_dim;
    const std::size_t dest_stride = static_cast<std::size_t>(dest_row_tokens) * head_dim;

    for (int head = 0; head < geometry.kv_heads; ++head) {
        float* out = dest + static_cast<std::size_t>(head) * dest_stride;
        int copied = 0;
        std::size_t block = 0;
        while (copied < length) {
            const float* source =
                pool.block(sequence.blocks[block]) + slab + static_cast<std::size_t>(head) * block_stride;
            const int take = std::min(geometry.block_tokens, length - copied);
            std::memcpy(out + static_cast<std::size_t>(copied) * head_dim, source,
                        static_cast<std::size_t>(take) * head_dim * sizeof(float));
            copied += take;
            ++block;
        }
    }
}

void zero_pad(const KvGeometry& geometry, float* dest, int length, int dest_row_tokens) {
    if (length < 0) {
        throw std::invalid_argument("zero_pad length must not be negative");
    }
    if (dest_row_tokens < length) {
        throw std::invalid_argument("zero_pad row holds " + std::to_string(dest_row_tokens) +
                                    " token position(s), fewer than the " +
                                    std::to_string(length) + " said to be filled");
    }
    const int pad_tokens = dest_row_tokens - length;
    if (pad_tokens == 0) {
        return;
    }
    if (dest == nullptr) {
        throw std::invalid_argument("zero_pad needs a destination buffer");
    }

    const std::size_t head_dim = static_cast<std::size_t>(geometry.head_dim);
    const std::size_t dest_stride = static_cast<std::size_t>(dest_row_tokens) * head_dim;
    const std::size_t bytes = static_cast<std::size_t>(pad_tokens) * head_dim * sizeof(float);
    for (int head = 0; head < geometry.kv_heads; ++head) {
        std::memset(dest + static_cast<std::size_t>(head) * dest_stride +
                        static_cast<std::size_t>(length) * head_dim,
                    0, bytes);
    }
}

void scatter(BlockPool& pool, const SequenceCache& sequence, int layer, KvKind kind,
             int dest_start, int count, const float* present, int present_length,
             int source_start) {
    const KvGeometry& geometry = pool.geometry();
    check_layer(geometry, layer);
    if (dest_start < 0 || count < 0 || source_start < 0) {
        throw std::invalid_argument(
            "scatter offsets and count must not be negative");
    }
    if (source_start + count > present_length) {
        throw std::invalid_argument("scatter would read position " +
                                    std::to_string(source_start + count) +
                                    " of a present tensor holding " +
                                    std::to_string(present_length));
    }
    check_span(geometry, sequence, dest_start + count);
    if (count == 0) {
        return;
    }
    if (present == nullptr) {
        throw std::invalid_argument("scatter needs a source buffer");
    }

    const std::size_t slab = slab_offset(geometry, layer, kind);
    const std::size_t head_dim = static_cast<std::size_t>(geometry.head_dim);
    const std::size_t block_stride = static_cast<std::size_t>(geometry.block_tokens) * head_dim;
    const std::size_t present_stride = static_cast<std::size_t>(present_length) * head_dim;
    const int end = dest_start + count;

    for (int head = 0; head < geometry.kv_heads; ++head) {
        const float* source = present + static_cast<std::size_t>(head) * present_stride;
        int token = dest_start;
        while (token < end) {
            const std::size_t block = static_cast<std::size_t>(token / geometry.block_tokens);
            const int offset = token % geometry.block_tokens;
            // Consecutive positions are contiguous in both the block and the
            // present tensor, so a run inside one block is a single copy. The two
            // sides advance together but from different origins, which is why the
            // read is offset by source_start - dest_start.
            const int run = std::min(geometry.block_tokens - offset, end - token);
            float* out = pool.block(sequence.blocks[block]) + slab +
                         static_cast<std::size_t>(head) * block_stride +
                         static_cast<std::size_t>(offset) * head_dim;
            std::memcpy(out,
                        source + static_cast<std::size_t>(token - dest_start + source_start) *
                                     head_dim,
                        static_cast<std::size_t>(run) * head_dim * sizeof(float));
            token += run;
        }
    }
}

bool prefix_matches(const BlockPool& pool, const SequenceCache& sequence, int layer, KvKind kind,
                    int length, const float* present, int present_length) {
    const KvGeometry& geometry = pool.geometry();
    check_layer(geometry, layer);
    if (length < 0 || length > present_length) {
        throw std::invalid_argument("prefix length " + std::to_string(length) +
                                    " does not fit a present tensor holding " +
                                    std::to_string(present_length));
    }
    check_span(geometry, sequence, length);
    if (length == 0) {
        return true;
    }

    const std::size_t slab = slab_offset(geometry, layer, kind);
    const std::size_t head_dim = static_cast<std::size_t>(geometry.head_dim);
    const std::size_t block_stride = static_cast<std::size_t>(geometry.block_tokens) * head_dim;
    const std::size_t present_stride = static_cast<std::size_t>(present_length) * head_dim;

    for (int head = 0; head < geometry.kv_heads; ++head) {
        const float* source = present + static_cast<std::size_t>(head) * present_stride;
        int compared = 0;
        std::size_t block = 0;
        while (compared < length) {
            const float* held =
                pool.block(sequence.blocks[block]) + slab + static_cast<std::size_t>(head) * block_stride;
            const int take = std::min(geometry.block_tokens, length - compared);
            if (std::memcmp(held, source + static_cast<std::size_t>(compared) * head_dim,
                            static_cast<std::size_t>(take) * head_dim * sizeof(float)) != 0) {
                return false;
            }
            compared += take;
            ++block;
        }
    }
    return true;
}

}  // namespace anytime
