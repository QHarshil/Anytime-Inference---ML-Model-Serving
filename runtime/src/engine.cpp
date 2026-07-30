#include "anytime/engine.hpp"

#include <algorithm>
#include <chrono>
#include <sstream>
#include <stdexcept>

namespace anytime {
namespace {

std::string join(const std::vector<std::string>& parts) {
    std::ostringstream out;
    for (std::size_t i = 0; i < parts.size(); ++i) {
        if (i) out << ", ";
        out << parts[i];
    }
    return out.str();
}

}  // namespace

Model::Model(Ort::Env& env, const std::string& path, int intra_op_threads,
             int inter_op_threads) {
    Ort::SessionOptions options;
    options.SetIntraOpNumThreads(intra_op_threads);
    options.SetInterOpNumThreads(inter_op_threads);
    options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    session_ = std::make_unique<Ort::Session>(env, path.c_str(), options);

    Ort::AllocatorWithDefaultOptions allocator;
    const std::size_t input_count = session_->GetInputCount();
    input_names_.reserve(input_count);
    for (std::size_t i = 0; i < input_count; ++i) {
        auto name = session_->GetInputNameAllocated(i, allocator);
        input_names_.emplace_back(name.get());
    }
    const std::size_t output_count = session_->GetOutputCount();
    output_names_.reserve(output_count);
    for (std::size_t i = 0; i < output_count; ++i) {
        auto name = session_->GetOutputNameAllocated(i, allocator);
        output_names_.emplace_back(name.get());
    }
}

bool Model::declares_input(const std::string& name) const {
    return std::find(input_names_.begin(), input_names_.end(), name) !=
           input_names_.end();
}

Engine::Engine(const std::vector<std::pair<std::string, std::string>>& models,
               int intra_op_threads, int inter_op_threads)
    : env_(ORT_LOGGING_LEVEL_WARNING, "anytime_runtime"),
      memory_(Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault)) {
    if (models.empty()) {
        throw std::invalid_argument("Engine needs at least one model");
    }
    for (const auto& [variant, path] : models) {
        if (models_.count(variant)) {
            throw std::invalid_argument("duplicate variant: " + variant);
        }
        models_[variant] = std::make_unique<Model>(env_, path, intra_op_threads,
                                                   inter_op_threads);
    }
}

Model& Engine::lookup(const std::string& variant) {
    const auto it = models_.find(variant);
    if (it == models_.end()) {
        // runtime_error rather than out_of_range: pybind11 maps it to
        // RuntimeError, which is the contract every backend shares for "the
        // runtime could not serve this request". Malformed arguments (a dtype the
        // engine does not accept) stay invalid_argument and surface as ValueError.
        throw std::runtime_error("unknown variant: " + variant + " (loaded: " +
                                 join(variants()) + ")");
    }
    return *it->second;
}

const Model& Engine::lookup(const std::string& variant) const {
    const auto it = models_.find(variant);
    if (it == models_.end()) {
        // runtime_error rather than out_of_range: pybind11 maps it to
        // RuntimeError, which is the contract every backend shares for "the
        // runtime could not serve this request". Malformed arguments (a dtype the
        // engine does not accept) stay invalid_argument and surface as ValueError.
        throw std::runtime_error("unknown variant: " + variant + " (loaded: " +
                                 join(variants()) + ")");
    }
    return *it->second;
}

std::vector<std::string> Engine::variants() const {
    std::vector<std::string> names;
    names.reserve(models_.size());
    for (const auto& [variant, model] : models_) {
        names.push_back(variant);
    }
    return names;
}

const std::vector<std::string>& Engine::input_names(const std::string& variant) const {
    return lookup(variant).input_names();
}

const std::vector<std::string>& Engine::output_names(const std::string& variant) const {
    return lookup(variant).output_names();
}

RunResult Engine::run(const std::string& variant,
                      const std::vector<std::pair<std::string, TensorView>>& feeds) {
    Model& model = lookup(variant);

    std::vector<const char*> input_names;
    std::vector<Ort::Value> inputs;
    input_names.reserve(feeds.size());
    inputs.reserve(feeds.size());

    for (const auto& [name, view] : feeds) {
        if (!model.declares_input(name)) continue;
        input_names.push_back(name.c_str());
        inputs.push_back(borrow_as_tensor(view, memory_));
    }

    if (inputs.size() != model.input_names().size()) {
        std::vector<std::string> missing;
        for (const auto& declared : model.input_names()) {
            const bool supplied = std::any_of(
                feeds.begin(), feeds.end(),
                [&declared](const auto& feed) { return feed.first == declared; });
            if (!supplied) missing.push_back(declared);
        }
        throw std::runtime_error("variant " + variant + " is missing input(s): " +
                                 join(missing) + " (declares: " +
                                 join(model.input_names()) + ")");
    }

    std::vector<const char*> output_names;
    output_names.reserve(model.output_names().size());
    for (const auto& name : model.output_names()) {
        output_names.push_back(name.c_str());
    }

    const auto start = std::chrono::steady_clock::now();
    std::vector<Ort::Value> outputs = model.session().Run(
        Ort::RunOptions{nullptr},
        input_names.data(), inputs.data(), inputs.size(),
        output_names.data(), output_names.size());
    const auto end = std::chrono::steady_clock::now();

    RunResult result;
    result.outputs = std::move(outputs);
    result.latency_ms =
        std::chrono::duration<double, std::milli>(end - start).count();
    return result;
}

}  // namespace anytime
