// Decoder-only inference over a block-allocated KV cache.
//
// Splits a generation into the two phases that have almost nothing in common. On
// this host, GPT-2 124M at FP32 prefills a 1024-token prompt in 372 ms and then
// emits each following token in 9.5 ms at full context: a factor of 39. Reporting
// one latency for both would describe neither, which is why prefill and decode are
// separate calls returning separate timings.
//
// The cache is a host-side block allocator; see kv_cache.hpp for why a stock
// exported decoder admits nothing else. This class is the mechanical half of the
// design -- gather, run, scatter, and the arena's occupancy. The policy half, which
// decides who is admitted and who is evicted, lives in Python next to the deadlines
// it reasons about: src/anytime_serving/serving/kv_admission.py.
//
// Prefill runs in chunks by default because that measured faster as well as
// smaller. One pass over 1024 tokens took 433.6 ms and allocated 206 MB of logits
// the sampler never reads; four passes of 256 took 372.2 ms, 0.858x, with a 51 MB
// peak. The redundant re-reads of the growing past cost less than the full-window
// attention and that discarded allocation. Chunk boundaries also give a scheduler a
// preemption point inside a long prefill, which continuous batching needs.

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

// Prefill chunk width, in tokens. 256 is four blocks at the default block size, and
// of 128, 256, 512 and one pass it measured fastest at FP32 and INT4 and tied with
// 128 at INT8; see the file comment.
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
//
// `pad_ms` is only ever non-zero for a batched step, and it is timed rather than
// inferred. Right-padding every row to the batch's longest past means clearing the
// difference, so the cost is set by the batch's length variance rather than by its
// size: eight sequences of equal length pad nothing, and one long sequence beside
// seven short ones pads almost as many token positions as it copies. Measuring it
// inside the same run as the gather keeps it off the wrong side of a subtraction.
struct StepTimings {
    double gather_ms = 0.0;
    double pad_ms = 0.0;
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

// Result of one batched decode step: one row of logits per sequence, and a single
// set of timings for the whole step.
//
// The timings are deliberately not per sequence. The rows share one `Session::Run`,
// so there is no honest way to attribute part of it to one of them; dividing by the
// batch size would produce a number that looks per-sequence and is not. `rows` is
// the divisor a caller needs to form an average and say so.
//
// Batching a decode step is worth doing, and how much depends on how full the caches
// are. Measured through the scheduler that drives this, GPT-2 at FP32 with the session
// on eight threads, batch 8 against the same eight sequences stepped one at a time:
// 3.00x at 128 cached tokens, 2.15x at 512 and 1.67x at 960. Only the
// cache-independent term of a step amortises across a batch; the per-cached-token term
// is per sequence and grows with the batch's total cache, which is why the curve
// decays.
//
// Thread count is part of that measurement rather than beside it. At one intra-op
// thread the same points read 2.32x / 1.52x / 1.25x, because a batch-1 decode is a
// skinny GEMV with little for a thread pool to divide while a wide batch is a real
// GEMM: batching supplies the parallelism threading then exploits, and the two
// compound. serving/decoder.py sets the count and records why.
//
// One thing not to infer: the gain falls short of what the split alone predicts, by
// more than the gather, the padding and the scatter together account for.
// scripts/profile_batching.py measures all of it, and docs/benchmarks.md says plainly
// that the shortfall is unexplained.
struct BatchStepResult {
    std::vector<StepResult> rows;
    StepTimings timings;
};

class DecoderSession {
public:
    // `num_blocks` fixes the arena: the cache never grows past it, which is what
    // makes admission a decision rather than a hope.
    //
    // Thread counts default to one of each, which is the neutral mechanism default
    // rather than a recommendation. On the encoder path that pin is load-bearing --
    // N single-threaded workers are N independent servers, which is what makes the
    // M/M/c model valid. The decoder lane has no pool, so the reason does not carry
    // over, and serving.DecoderClient overrides this with a measured count. Policy
    // in Python, mechanism here.
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
    // Runs `tokens` through the graph, appending to whatever the sequence already
    // holds. Prefill's inner step, exposed on its own.
    //
    // `prefill` loops over the chunks itself and refuses a sequence that is not
    // empty, which is right for a caller that wants a prompt run and nothing
    // interleaved. A scheduler wants the opposite: the chunk boundary is the point
    // where a long prefill can be interrupted, so it drives the chunks and decides
    // what happens between them. Without this a resident sequence stalls for a whole
    // prompt rather than for one chunk -- 372 ms against 93 ms on GPT-2 at FP32, at
    // the 256-token default.
    //
    // Unlike `prefill` this reserves per chunk rather than for the whole prompt, so a
    // prompt too large for the arena fails part way through instead of before it
    // starts. A scheduler is expected to have asked admission first.
    StepResult extend(const std::string& id, const std::vector<std::int64_t>& tokens);
    // Extends the sequence by one token, reading the cache for everything before it.
    StepResult decode(const std::string& id, std::int64_t token);
    // Extends each of `ids` by the matching token, in one graph invocation.
    //
    // Decode only. A prefill chunk and a decode step cannot share a `Run`: the graph
    // takes one `sequence` dimension as well as one `past_sequence_length`, so a
    // 256-token chunk and a one-token step are not merely a wasteful pairing, they
    // are unrepresentable without padding the decode row out to the chunk width.
    // Fusing them is what a flattened varlen layout and custom kernels buy, and a
    // stock exported graph has neither. A scheduler over this alternates instead.
    //
    // Rows are right-padded to the longest past in the batch. Every sequence must be
    // open and non-empty, and no id may repeat -- two rows of one sequence would
    // scatter twice into the same blocks and the second write would win.
    //
    // All or nothing on blocks. The whole batch's shortfall is checked against the
    // pool before any row takes a block, so a batch that cannot fit throws
    // CacheExhausted with the arena untouched rather than part way through.
    BatchStepResult decode_batch(const std::vector<std::string>& ids,
                                 const std::vector<std::int64_t>& tokens);

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
