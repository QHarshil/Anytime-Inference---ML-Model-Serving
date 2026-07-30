// Decoder-only inference over a block-allocated KV cache.
//
// Splits a generation into the two phases that have almost nothing in common. On
// this host, GPT-2 124M at FP32 prefills a 1024-token prompt in 434 ms and then
// emits each following token in 8.8 ms: a factor of 49. Reporting one latency for
// both would describe neither, which is why prefill and decode are separate calls
// returning separate timings.
//
// The cache is a host-side block allocator; see kv_cache.hpp for why a stock
// exported decoder admits nothing else. This class is the mechanical half of the
// design -- gather, run, scatter, and the arena's occupancy. The policy half, which
// decides who is admitted and who is evicted, lives in Python next to the deadlines
// it reasons about: src/anytime_serving/serving/kv_admission.py.
//
// Prefill runs in chunks by default because that measured faster as well as
// smaller. One pass over 1024 tokens took 444.9 ms and allocated 206 MB of logits
// the sampler never reads; four passes of 256 took 372.7 ms, 0.838x, with a 51 MB
// peak. The redundant re-reads of the growing past cost less than the full-window
// attention and that discarded allocation. Chunk boundaries also give the scheduler
// a preemption point inside a long prefill, which P4 needs.

#ifndef ANYTIME_DECODER_HPP
#define ANYTIME_DECODER_HPP

#include <onnxruntime_cxx_api.h>

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include "anytime/engine.hpp"
#include "anytime/kv_cache.hpp"

namespace anytime {

// Prefill chunk width, in tokens. 256 is four blocks at the default block size and
// measured fastest of 128, 256, 512 and one pass; see the file comment.
constexpr int kDefaultPrefillChunkTokens = 256;

// Token positions per block. 64 keeps a GPT-2 block at 4.5 MiB, so a 1024-token
// sequence is 16 blocks and admission has useful granularity without the free list
// growing large enough to matter.
constexpr int kDefaultBlockTokens = 64;

// Where one step's time went. `run_ms` is time inside Session::Run, matching what
// Engine::run reports, so the two are comparable. The gather is broken out because
// it is the price of block accounting and the whole point is not to assume it is
// free.
//
// Two costs land on a sequence's first steps rather than on its steady state, and
// both stay inside the phase that pays them rather than being hidden: `gather_ms`
// on the first step includes sizing the staging buffers, and `verify_ms` is non-zero
// only on the one step that checks the present-prefix invariant. A per-step
// distribution shows each as a single outlier, which is what it is.
struct StepTimings {
    double gather_ms = 0.0;
    double run_ms = 0.0;
    double scatter_ms = 0.0;
    double verify_ms = 0.0;
    double total_ms = 0.0;
};

// Result of one prefill or one decode step.
struct StepResult {
    // Logits for the next token only, copied out of the graph's output.
    //
    // This is the one place the runtime copies rather than borrowing, and it is a
    // saving rather than a cost. The graph returns logits for every position it was
    // given -- 206 MB for a 1024-token prefill -- when sampling reads one row of
    // 50257. Handing that back as a zero-copy view would keep tens of megabytes
    // alive to read 200 KB of it. So the last row is copied and the rest is dropped
    // with the Ort::Value.
    std::vector<float> logits;
    StepTimings timings;
    // Tokens in the cache after this step.
    int length = 0;
    // Graph invocations. Greater than one for a chunked prefill.
    int runs = 0;
};

class DecoderSession {
public:
    // `num_blocks` fixes the arena: the cache never grows past it, which is what
    // makes admission a decision rather than a hope. Thread counts default to one
    // of each, matching Engine, so a measurement here is comparable with one taken
    // through the encoder path.
    DecoderSession(const std::string& path, int block_tokens = kDefaultBlockTokens,
                   std::size_t num_blocks = 256, int intra_op_threads = 1,
                   int inter_op_threads = 1);

    const KvGeometry& geometry() const { return geometry_; }
    std::size_t capacity_blocks() const { return pool_->capacity_blocks(); }
    std::size_t free_blocks() const { return pool_->free_blocks(); }
    std::size_t arena_bytes() const { return pool_->bytes(); }
    std::size_t blocks_for(int tokens) const { return geometry_.blocks_for(tokens); }
    // Whether the graph declares these; a decoder that derives positions itself
    // does not, and feeding an input the graph never asked for is an error.
    bool declares_attention_mask() const { return has_attention_mask_; }
    bool declares_position_ids() const { return has_position_ids_; }

    // Reserves blocks for `reserve_tokens` positions and registers the sequence.
    // Returns false when the arena cannot supply them, leaving the pool and every
    // other sequence untouched -- refusing is the admission controller's answer,
    // not an exception.
    bool open(const std::string& id, int reserve_tokens);
    // Returns the blocks the sequence held. Idempotent for an unknown id, since a
    // policy releasing a sequence twice is not a runtime failure.
    std::size_t release(const std::string& id);
    bool contains(const std::string& id) const;
    int length(const std::string& id) const;
    std::size_t blocks_held(const std::string& id) const;
    std::vector<std::string> sequences() const;

    // Runs the prompt through the graph, filling the cache. `chunk_tokens` at or
    // below zero runs it in one pass.
    StepResult prefill(const std::string& id, const std::vector<std::int64_t>& tokens,
                       int chunk_tokens = kDefaultPrefillChunkTokens);
    // Extends the sequence by one token, reading the cache for everything before it.
    StepResult decode(const std::string& id, std::int64_t token);

private:
    SequenceCache& lookup(const std::string& id);
    // Takes blocks so the sequence can hold `tokens` positions. Throws
    // CacheExhausted rather than returning false: by the time a sequence is
    // mid-decode, the policy has already promised it room.
    void reserve(const std::string& id, SequenceCache& sequence, int tokens);
    StepResult step(const std::string& id, SequenceCache& sequence, const std::int64_t* tokens,
                    int count);
    void derive_geometry(int block_tokens);

    Ort::Env env_;
    Ort::MemoryInfo memory_;
    std::unique_ptr<Model> model_;
    KvGeometry geometry_;
    // Held indirectly because its size comes from the geometry, which is read off
    // the loaded graph rather than passed in, so the arena cannot be built until
    // the session exists.
    std::unique_ptr<BlockPool> pool_;
    std::map<std::string, SequenceCache> sequences_;

    // Sequences whose first non-empty step has already confirmed that
    // `present[..., :past_len, :]` equals what the blocks hold. Checked once per
    // sequence: the scatter only writes the new tail, so if the graph ever stopped
    // concatenating, every token before the current one would silently rot. A
    // mismatch raises rather than falling back to a full-present scatter, because a
    // silent fallback would change what TPOT measures without saying so.
    std::set<std::string> prefix_verified_;

    // Reused across steps, so a decode loop in its steady state does not allocate.
    // One buffer per graph past input, sized to the current past length.
    std::vector<std::vector<float>> past_buffers_;
    std::vector<std::int64_t> attention_mask_;
    std::vector<std::int64_t> position_ids_;

    // Graph indices resolved once, by name rather than by position.
    std::vector<std::string> past_input_names_;  // 2 * layers, key then value per layer
    std::size_t logits_output_ = 0;
    std::vector<std::size_t> present_outputs_;  // 2 * layers, same ordering
    bool has_attention_mask_ = false;
    bool has_position_ids_ = false;
};

}  // namespace anytime

#endif  // ANYTIME_DECODER_HPP
