#include "anytime/decoder.hpp"

#include <algorithm>
#include <chrono>
#include <map>
#include <stdexcept>

namespace anytime {
namespace {

using Clock = std::chrono::steady_clock;

double elapsed_ms(Clock::time_point from, Clock::time_point to) {
    return std::chrono::duration<double, std::milli>(to - from).count();
}

std::string past_name(int layer, KvKind kind) {
    return "past_key_values." + std::to_string(layer) +
           (kind == KvKind::Key ? ".key" : ".value");
}

std::string present_name(int layer, KvKind kind) {
    return "present." + std::to_string(layer) + (kind == KvKind::Key ? ".key" : ".value");
}

std::map<std::string, std::size_t> index_by_name(const std::vector<std::string>& names) {
    std::map<std::string, std::size_t> index;
    for (std::size_t i = 0; i < names.size(); ++i) {
        index[names[i]] = i;
    }
    return index;
}

}  // namespace

DecoderSession::DecoderSession(const std::string& path, int block_tokens, std::size_t num_blocks,
                               int intra_op_threads, int inter_op_threads)
    : env_(ORT_LOGGING_LEVEL_WARNING, "anytime_decoder"),
      memory_(Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault)) {
    model_ = std::make_unique<Model>(env_, path, intra_op_threads, inter_op_threads);
    derive_geometry(block_tokens);
    pool_ = std::make_unique<BlockPool>(geometry_, num_blocks);
    past_buffers_.resize(static_cast<std::size_t>(geometry_.layers) * 2);
}

void DecoderSession::derive_geometry(int block_tokens) {
    // Read off the graph rather than taken from a model config. A config that
    // disagrees with the graph it describes would produce a cache of the wrong
    // shape, and the failure would be wrong logits rather than an error.
    const auto inputs = index_by_name(model_->input_names());
    const auto outputs = index_by_name(model_->output_names());

    if (inputs.find("input_ids") == inputs.end()) {
        throw std::invalid_argument(
            "graph declares no input_ids, so it is not a decoder this session can drive");
    }

    int layers = 0;
    while (inputs.find(past_name(layers, KvKind::Key)) != inputs.end()) {
        ++layers;
    }
    if (layers == 0) {
        throw std::invalid_argument(
            "graph declares no past_key_values.0.key. Export with the "
            "'-with-past' task so the KV cache is in the signature; a graph that "
            "hides its cache cannot be block-allocated.");
    }

    for (int layer = 0; layer < layers; ++layer) {
        for (const KvKind kind : {KvKind::Key, KvKind::Value}) {
            if (inputs.find(past_name(layer, kind)) == inputs.end()) {
                throw std::invalid_argument("graph is missing input " + past_name(layer, kind) +
                                            " while declaring " + std::to_string(layers) +
                                            " layer(s)");
            }
            if (outputs.find(present_name(layer, kind)) == outputs.end()) {
                throw std::invalid_argument("graph is missing output " +
                                            present_name(layer, kind) +
                                            ", so the cache could not be updated after a run");
            }
        }
    }

    const auto info = model_->session().GetInputTypeInfo(inputs.at(past_name(0, KvKind::Key)));
    const auto tensor = info.GetTensorTypeAndShapeInfo();
    if (tensor.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
        throw std::invalid_argument(
            "past_key_values.0.key is " + ort_type_name(tensor.GetElementType()) +
            ", but the arena stores float32. Weight-only quantisation leaves the "
            "cache in float; a graph with a narrower cache needs an arena to match.");
    }
    const std::vector<int64_t> shape = tensor.GetShape();
    if (shape.size() != 4) {
        throw std::invalid_argument("past_key_values.0.key has " + std::to_string(shape.size()) +
                                    " dimension(s); [batch, kv_heads, past, head_dim] is "
                                    "expected");
    }
    // Dimensions 1 and 3 must be static, because they size the arena and cannot be
    // discovered per request. Dimension 2 is the one that grows and is dynamic by
    // construction. ONNX Runtime reports a dynamic dimension as -1.
    if (shape[1] <= 0 || shape[3] <= 0) {
        throw std::invalid_argument(
            "past_key_values.0.key has a dynamic kv_heads or head_dim (shape [" +
            std::to_string(shape[0]) + ", " + std::to_string(shape[1]) + ", " +
            std::to_string(shape[2]) + ", " + std::to_string(shape[3]) +
            "]). Both size the block pool, so neither can be resolved per request.");
    }

    geometry_.layers = layers;
    geometry_.kv_heads = static_cast<int>(shape[1]);
    geometry_.head_dim = static_cast<int>(shape[3]);
    geometry_.block_tokens = block_tokens;
    geometry_.validate();

    const auto logits = outputs.find("logits");
    if (logits == outputs.end()) {
        throw std::invalid_argument("graph declares no logits output, so nothing can be sampled");
    }
    logits_output_ = logits->second;

    past_input_names_.clear();
    present_outputs_.clear();
    past_input_names_.reserve(static_cast<std::size_t>(layers) * 2);
    present_outputs_.reserve(static_cast<std::size_t>(layers) * 2);
    for (int layer = 0; layer < layers; ++layer) {
        for (const KvKind kind : {KvKind::Key, KvKind::Value}) {
            past_input_names_.push_back(past_name(layer, kind));
            present_outputs_.push_back(outputs.at(present_name(layer, kind)));
        }
    }

    // Optional, so a decoder that derives its own positions still runs. Engine
    // drops undeclared feeds; this session builds only what the graph asked for.
    has_attention_mask_ = inputs.find("attention_mask") != inputs.end();
    has_position_ids_ = inputs.find("position_ids") != inputs.end();
}

bool DecoderSession::open(const std::string& id, int reserve_tokens) {
    if (id.empty()) {
        throw std::invalid_argument("a sequence id must not be empty");
    }
    if (reserve_tokens < 0) {
        throw std::invalid_argument("reserve_tokens must not be negative");
    }
    if (sequences_.find(id) != sequences_.end()) {
        throw std::runtime_error("sequence " + id + " is already open; release it first");
    }

    std::vector<std::uint32_t> blocks;
    const std::size_t needed = geometry_.blocks_for(reserve_tokens);
    if (needed > 0) {
        blocks = pool_->allocate(needed);
        if (blocks.empty()) {
            // Refusing, not throwing: this is the admission controller asking
            // whether there is room, and "no" is an answer rather than a failure.
            return false;
        }
    }

    SequenceCache cache;
    cache.blocks = std::move(blocks);
    sequences_.emplace(id, std::move(cache));
    prefix_verified_.erase(id);
    return true;
}

std::size_t DecoderSession::release(const std::string& id) {
    const auto entry = sequences_.find(id);
    if (entry == sequences_.end()) {
        // Idempotent. A policy that releases a completed sequence and then unwinds
        // is not making a runtime error.
        return 0;
    }
    const std::size_t held = entry->second.blocks.size();
    pool_->release(entry->second.blocks);
    sequences_.erase(entry);
    prefix_verified_.erase(id);
    return held;
}

bool DecoderSession::contains(const std::string& id) const {
    return sequences_.find(id) != sequences_.end();
}

SequenceCache& DecoderSession::lookup(const std::string& id) {
    const auto entry = sequences_.find(id);
    if (entry == sequences_.end()) {
        throw std::runtime_error("unknown sequence " + id + "; open it before running it");
    }
    return entry->second;
}

int DecoderSession::length(const std::string& id) const {
    const auto entry = sequences_.find(id);
    if (entry == sequences_.end()) {
        throw std::runtime_error("unknown sequence " + id + "; open it before running it");
    }
    return entry->second.length;
}

std::size_t DecoderSession::blocks_held(const std::string& id) const {
    const auto entry = sequences_.find(id);
    if (entry == sequences_.end()) {
        throw std::runtime_error("unknown sequence " + id + "; open it before running it");
    }
    return entry->second.blocks.size();
}

std::vector<std::string> DecoderSession::sequences() const {
    std::vector<std::string> ids;
    ids.reserve(sequences_.size());
    for (const auto& [id, cache] : sequences_) {
        ids.push_back(id);
    }
    return ids;
}

void DecoderSession::reserve(const std::string& id, SequenceCache& sequence, int tokens) {
    const std::size_t needed = geometry_.blocks_for(tokens);
    if (needed <= sequence.blocks.size()) {
        return;
    }
    const std::size_t extra = needed - sequence.blocks.size();
    std::vector<std::uint32_t> taken = pool_->allocate(extra);
    if (taken.empty()) {
        throw CacheExhausted("sequence " + id + " needs " + std::to_string(extra) +
                             " more block(s) to hold " + std::to_string(tokens) +
                             " token(s), and " + std::to_string(pool_->free_blocks()) + " of " +
                             std::to_string(pool_->capacity_blocks()) +
                             " are free. The arena is fixed by design: evict a sequence or "
                             "refuse this one.");
    }
    sequence.blocks.insert(sequence.blocks.end(), taken.begin(), taken.end());
}

StepResult DecoderSession::step(const std::string& id, SequenceCache& sequence,
                                const std::int64_t* tokens, int count) {
    const auto step_start = Clock::now();

    const int past_len = sequence.length;
    const int total = past_len + count;
    reserve(id, sequence, total);

    const std::size_t past_floats = static_cast<std::size_t>(geometry_.kv_heads) *
                                    static_cast<std::size_t>(past_len) *
                                    static_cast<std::size_t>(geometry_.head_dim);
    const std::size_t halves = static_cast<std::size_t>(geometry_.layers) * 2;

    // Sized to what the sequence reserved rather than to what this step needs.
    // Growing to the exact past length would zero-fill a region the gather is about
    // to overwrite, and a decode step adds one token, so that cost would land on
    // every step. Sizing to the reservation instead means one growth per sequence
    // and none in the steady state. The floor of one element keeps data() a real
    // address: the first prefill hands the graph a zero-length past tensor, and a
    // null pointer is not a safe thing to give ONNX Runtime.
    const std::size_t reserved_floats =
        static_cast<std::size_t>(geometry_.kv_heads) *
        static_cast<std::size_t>(sequence.capacity_tokens(geometry_.block_tokens)) *
        static_cast<std::size_t>(geometry_.head_dim);
    const std::size_t wanted = std::max<std::size_t>(std::max(reserved_floats, past_floats), 1);

    const auto gather_start = Clock::now();
    for (std::size_t slot = 0; slot < halves; ++slot) {
        std::vector<float>& buffer = past_buffers_[slot];
        if (buffer.size() < wanted) {
            buffer.resize(wanted);
        }
        gather(*pool_, sequence, static_cast<int>(slot / 2), static_cast<KvKind>(slot % 2),
               past_len, buffer.data());
    }
    const auto gather_end = Clock::now();

    std::vector<const char*> input_names;
    std::vector<Ort::Value> inputs;
    input_names.reserve(model_->input_names().size());
    inputs.reserve(model_->input_names().size());

    TensorView ids;
    ids.data = tokens;
    ids.shape = {1, count};
    ids.dtype = DType::Int64;
    input_names.push_back("input_ids");
    inputs.push_back(borrow_as_tensor(ids, memory_));

    for (std::size_t slot = 0; slot < halves; ++slot) {
        TensorView past;
        past.data = past_buffers_[slot].data();
        past.shape = {1, geometry_.kv_heads, past_len, geometry_.head_dim};
        past.dtype = DType::Float32;
        input_names.push_back(past_input_names_[slot].c_str());
        inputs.push_back(borrow_as_tensor(past, memory_));
    }

    if (has_attention_mask_) {
        // Every cached position is attended to. Eviction here drops a whole
        // sequence rather than part of one, so there is no hole to mask out.
        attention_mask_.assign(static_cast<std::size_t>(total), 1);
        TensorView mask;
        mask.data = attention_mask_.data();
        mask.shape = {1, total};
        mask.dtype = DType::Int64;
        input_names.push_back("attention_mask");
        inputs.push_back(borrow_as_tensor(mask, memory_));
    }

    if (has_position_ids_) {
        position_ids_.resize(static_cast<std::size_t>(count));
        for (int i = 0; i < count; ++i) {
            position_ids_[static_cast<std::size_t>(i)] = past_len + i;
        }
        TensorView positions;
        positions.data = position_ids_.data();
        positions.shape = {1, count};
        positions.dtype = DType::Int64;
        input_names.push_back("position_ids");
        inputs.push_back(borrow_as_tensor(positions, memory_));
    }

    if (inputs.size() != model_->input_names().size()) {
        throw std::runtime_error(
            "built " + std::to_string(inputs.size()) + " feed(s) for a graph declaring " +
            std::to_string(model_->input_names().size()) +
            ". A decoder input beyond input_ids, the past tensors, attention_mask and "
            "position_ids is not handled; running on a partial feed would produce wrong "
            "logits silently.");
    }

    std::vector<const char*> output_names;
    output_names.reserve(model_->output_names().size());
    for (const std::string& name : model_->output_names()) {
        output_names.push_back(name.c_str());
    }

    const auto run_start = Clock::now();
    std::vector<Ort::Value> outputs =
        model_->session().Run(Ort::RunOptions{nullptr}, input_names.data(), inputs.data(),
                              inputs.size(), output_names.data(), output_names.size());
    const auto run_end = Clock::now();

    double verify_ms = 0.0;
    if (past_len > 0 && prefix_verified_.find(id) == prefix_verified_.end()) {
        // The scatter below writes only the new tail, which is only correct if the
        // graph concatenates the past it was given rather than rewriting it.
        // Measured bitwise true on GPT-2, but it is a property of the exported
        // graph and not of the ONNX specification, so it is checked once per
        // sequence instead of trusted. A mismatch raises: scattering the whole
        // present instead would keep running while quietly changing what a decode
        // step costs, and every TPOT number after that would be incomparable.
        const auto verify_start = Clock::now();
        for (std::size_t slot = 0; slot < halves; ++slot) {
            const float* present = outputs[present_outputs_[slot]].GetTensorData<float>();
            if (!prefix_matches(*pool_, sequence, static_cast<int>(slot / 2),
                                static_cast<KvKind>(slot % 2), past_len, present, total)) {
                throw std::runtime_error(
                    "graph output " + past_input_names_[slot] +
                    "'s present tensor does not begin with the past it was given, so this "
                    "cache cannot append only the new tail. The block allocator assumes the "
                    "graph concatenates; this one does not.");
            }
        }
        verify_ms = elapsed_ms(verify_start, Clock::now());
        prefix_verified_.insert(id);
    }

    const auto scatter_start = Clock::now();
    for (std::size_t slot = 0; slot < halves; ++slot) {
        const float* present = outputs[present_outputs_[slot]].GetTensorData<float>();
        scatter(*pool_, sequence, static_cast<int>(slot / 2), static_cast<KvKind>(slot % 2),
                past_len, count, present, total);
    }
    sequence.length = total;
    const auto scatter_end = Clock::now();

    const auto logits_info = outputs[logits_output_].GetTensorTypeAndShapeInfo();
    const std::vector<int64_t> logits_shape = logits_info.GetShape();
    if (logits_shape.size() != 3 || logits_shape[1] != count || logits_shape[2] <= 0) {
        throw std::runtime_error("logits came back with an unexpected shape; [1, " +
                                 std::to_string(count) + ", vocab] is expected");
    }
    const std::size_t vocab = static_cast<std::size_t>(logits_shape[2]);
    const std::size_t rows = static_cast<std::size_t>(logits_shape[1]);
    const float* logits = outputs[logits_output_].GetTensorData<float>();

    StepResult result;
    // Only the last row. See StepResult for why this one copy is a saving.
    result.logits.assign(logits + (rows - 1) * vocab, logits + rows * vocab);
    result.length = sequence.length;
    result.runs = 1;
    result.timings.gather_ms = elapsed_ms(gather_start, gather_end);
    result.timings.run_ms = elapsed_ms(run_start, run_end);
    result.timings.scatter_ms = elapsed_ms(scatter_start, scatter_end);
    result.timings.verify_ms = verify_ms;
    result.timings.total_ms = elapsed_ms(step_start, Clock::now());
    return result;
}

StepResult DecoderSession::prefill(const std::string& id, const std::vector<std::int64_t>& tokens,
                                   int chunk_tokens) {
    SequenceCache& sequence = lookup(id);
    if (tokens.empty()) {
        throw std::invalid_argument("prefill needs at least one token");
    }
    if (sequence.length != 0) {
        throw std::runtime_error("sequence " + id + " already holds " +
                                 std::to_string(sequence.length) +
                                 " token(s); decode to extend it, or release and reopen it to "
                                 "start over");
    }

    const int prompt = static_cast<int>(tokens.size());
    // Reserved up front so a prompt too large for the arena fails before any of it
    // has been run, rather than part way through a chunked prefill.
    reserve(id, sequence, prompt);

    const int width = chunk_tokens > 0 ? std::min(chunk_tokens, prompt) : prompt;
    StepResult total;
    int done = 0;
    while (done < prompt) {
        const int take = std::min(width, prompt - done);
        StepResult chunk = step(id, sequence, tokens.data() + done, take);
        total.timings.gather_ms += chunk.timings.gather_ms;
        total.timings.run_ms += chunk.timings.run_ms;
        total.timings.scatter_ms += chunk.timings.scatter_ms;
        total.timings.verify_ms += chunk.timings.verify_ms;
        total.timings.total_ms += chunk.timings.total_ms;
        total.runs += chunk.runs;
        total.length = chunk.length;
        // Every chunk but the last predicts a token the prompt already supplies, so
        // only the final row is worth keeping.
        total.logits = std::move(chunk.logits);
        done += take;
    }
    return total;
}

StepResult DecoderSession::decode(const std::string& id, std::int64_t token) {
    SequenceCache& sequence = lookup(id);
    if (sequence.length == 0) {
        throw std::runtime_error("sequence " + id +
                                 " has an empty cache; prefill a prompt before decoding");
    }
    return step(id, sequence, &token, 1);
}

}  // namespace anytime
