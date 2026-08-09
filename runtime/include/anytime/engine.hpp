// In-process inference engine.
//
// Owns one ONNX Runtime session per named variant and runs them synchronously.
// This replaces the line-delimited JSON worker from Stage 1: the transport is now
// a function call, so there is no framing, no base64, and no copy in either
// direction.
//
// Threading in this phase mirrors what the subprocess pool did. One Engine holds
// its own sessions, and the Python pool holds one Engine per slot, so N workers
// remain N independent single-threaded servers and the M/M/c model the admission
// controller uses stays valid. Sharing one session across the pool would save
// memory and is what the scheduler will do once it lands, but it would change the
// concurrency model in the same commit that introduces the engine.

#ifndef ANYTIME_ENGINE_HPP
#define ANYTIME_ENGINE_HPP

#include <onnxruntime_cxx_api.h>

#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "anytime/tensor.hpp"

namespace anytime {

// One loaded graph, with the input and output names it declares.
class Model {
public:
    // `allow_spinning` is ONNX Runtime's own default: after a Run returns, its
    // intra-op workers busy-wait for the next one rather than sleeping, which makes
    // the next Run start sooner and costs cores in between. That trade is only
    // obviously right when Run is the only thing happening. On the decoder path it is
    // not -- a gather runs between two Runs, and it is bandwidth-bound -- so this is
    // exposed to be measured rather than assumed. Defaults to true, which is what
    // every recorded number was taken with.
    Model(Ort::Env& env, const std::string& path, int intra_op_threads,
          int inter_op_threads, bool allow_spinning = true);

    const std::vector<std::string>& input_names() const { return input_names_; }
    const std::vector<std::string>& output_names() const { return output_names_; }
    bool declares_input(const std::string& name) const;
    Ort::Session& session() { return *session_; }

private:
    std::unique_ptr<Ort::Session> session_;
    std::vector<std::string> input_names_;
    std::vector<std::string> output_names_;
};

// Outputs of one run, in the order the graph declares them, plus the time spent
// inside Session::Run. The values own their buffers; the bindings hand those
// buffers to numpy without copying.
struct RunResult {
    std::vector<Ort::Value> outputs;
    double latency_ms = 0.0;
};

class Engine {
public:
    // Each entry maps a variant name to an ONNX graph path. Threads default to
    // one of each so a pooled engine behaves as a single server, matching how
    // the service times in configs/serving.yaml were measured.
    explicit Engine(const std::vector<std::pair<std::string, std::string>>& models,
                    int intra_op_threads = 1, int inter_op_threads = 1);

    // Runs `variant` over `feeds`.
    //
    // Feeds a graph does not declare are dropped rather than rejected: variants
    // of one task can declare different inputs (a DistilBERT graph takes
    // input_ids and attention_mask, a BERT graph also takes token_type_ids), so
    // callers pass the union. A declared input that is missing is an error, since
    // running on a partial feed would silently produce wrong logits.
    RunResult run(const std::string& variant,
                  const std::vector<std::pair<std::string, TensorView>>& feeds);

    std::vector<std::string> variants() const;
    const std::vector<std::string>& input_names(const std::string& variant) const;
    const std::vector<std::string>& output_names(const std::string& variant) const;

private:
    Model& lookup(const std::string& variant);
    const Model& lookup(const std::string& variant) const;

    Ort::Env env_;
    Ort::MemoryInfo memory_;
    std::map<std::string, std::unique_ptr<Model>> models_;
};

}  // namespace anytime

#endif  // ANYTIME_ENGINE_HPP
