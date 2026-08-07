// pybind11 bindings for the in-process engine.
//
// Two things here are load-bearing beyond wiring:
//
// Tensors cross the boundary without copying. Inputs are borrowed straight from
// the numpy buffer, so the references are held on the stack for the whole call.
// Outputs are numpy views over the buffers ONNX Runtime allocated, kept alive by
// a capsule that owns the Ort::Value.
//
// The GIL is released around Session::Run. Without that, a pool of workers would
// serialise on the interpreter lock and the M/M/c model the admission controller
// relies on would describe a machine that does not exist.

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "anytime/decoder.hpp"
#include "anytime/engine.hpp"
#include "anytime/kv_cache.hpp"
#include "anytime/tensor.hpp"

namespace py = pybind11;

namespace {

anytime::DType dtype_from_numpy(const py::array& array, const std::string& name) {
    const py::dtype dtype = array.dtype();
    if (dtype.is(py::dtype::of<float>()))    return anytime::DType::Float32;
    if (dtype.is(py::dtype::of<double>()))   return anytime::DType::Float64;
    if (dtype.is(py::dtype::of<int32_t>()))  return anytime::DType::Int32;
    if (dtype.is(py::dtype::of<int64_t>()))  return anytime::DType::Int64;
    if (dtype.is(py::dtype::of<bool>()))     return anytime::DType::Bool;
    throw std::invalid_argument(
        "feed '" + name + "' has dtype " +
        py::str(dtype).cast<std::string>() +
        ", which the engine does not accept. Supported: float32, float64, int32, "
        "int64, bool. Cast before submitting rather than relying on an implicit "
        "conversion.");
}

py::dtype numpy_dtype_from_ort(ONNXTensorElementDataType type) {
    switch (type) {
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:  return py::dtype::of<float>();
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE: return py::dtype::of<double>();
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:  return py::dtype::of<int32_t>();
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:  return py::dtype::of<int64_t>();
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:   return py::dtype::of<bool>();
        default:
            throw std::runtime_error(
                "graph produced an output of type " + anytime::ort_type_name(type) +
                ", which the bindings cannot map to numpy");
    }
}

// Hands ONNX Runtime's output buffer to numpy without copying it. The capsule
// owns the Ort::Value, so the buffer outlives this call for exactly as long as
// the returned array is referenced.
py::array wrap_output(Ort::Value&& value) {
    const auto info = value.GetTensorTypeAndShapeInfo();
    const ONNXTensorElementDataType element_type = info.GetElementType();
    const py::dtype dtype = numpy_dtype_from_ort(element_type);
    const std::vector<int64_t> raw_shape = info.GetShape();

    std::vector<py::ssize_t> shape(raw_shape.begin(), raw_shape.end());
    std::vector<py::ssize_t> strides(shape.size());
    py::ssize_t stride = static_cast<py::ssize_t>(dtype.itemsize());
    for (std::size_t i = shape.size(); i-- > 0;) {
        strides[i] = stride;
        stride *= shape[i];
    }

    if (info.GetElementCount() == 0) {
        // Nothing to share. An owning empty array avoids handing numpy a base
        // object for a buffer that may not exist.
        return py::array(dtype, shape, strides);
    }

    auto* owned = new Ort::Value(std::move(value));
    py::capsule owner(owned, [](void* pointer) {
        delete static_cast<Ort::Value*>(pointer);
    });
    return py::array(dtype, shape, strides, owned->GetTensorMutableRawData(), owner);
}

// Hands a vector's buffer to numpy without a second copy. The logits in a
// StepResult are already the one deliberate copy the decoder makes -- the last row
// of a much larger output -- so copying them again into a numpy array would be
// gratuitous.
py::array wrap_logits(std::vector<float>&& logits) {
    auto* owned = new std::vector<float>(std::move(logits));
    py::capsule owner(owned, [](void* pointer) {
        delete static_cast<std::vector<float>*>(pointer);
    });
    return py::array(py::dtype::of<float>(),
                     {static_cast<py::ssize_t>(owned->size())},
                     {static_cast<py::ssize_t>(sizeof(float))},
                     owned->data(), owner);
}

// What a prefill or decode step returns to Python. Separate from
// anytime::StepResult so the logits cross as a numpy view rather than being copied
// again by the stl caster.
struct PyStepResult {
    py::array logits;
    anytime::StepTimings timings;
    int length = 0;
    int runs = 0;
};

PyStepResult make_step_result(anytime::StepResult&& result) {
    PyStepResult wrapped;
    wrapped.logits = wrap_logits(std::move(result.logits));
    wrapped.timings = result.timings;
    wrapped.length = result.length;
    wrapped.runs = result.runs;
    return wrapped;
}

// What a batched decode step returns. The rows carry no timings of their own; see
// the class docstring for why splitting one Run between them would be an invention.
struct PyBatchStepResult {
    std::vector<PyStepResult> rows;
    anytime::StepTimings timings;
};

PyBatchStepResult make_batch_step_result(anytime::BatchStepResult&& result) {
    PyBatchStepResult wrapped;
    wrapped.rows.reserve(result.rows.size());
    for (anytime::StepResult& row : result.rows) {
        wrapped.rows.push_back(make_step_result(std::move(row)));
    }
    wrapped.timings = result.timings;
    return wrapped;
}

py::tuple run_engine(anytime::Engine& engine, const std::string& variant,
                     const py::dict& feeds) {
    // Built while the GIL is held. `retained` keeps every numpy buffer alive
    // across the release below; the views point into those buffers.
    std::vector<py::array> retained;
    std::vector<std::pair<std::string, anytime::TensorView>> views;
    retained.reserve(feeds.size());
    views.reserve(feeds.size());

    for (const auto item : feeds) {
        const auto name = py::cast<std::string>(item.first);
        // ONNX Runtime reads the buffer directly, so it has to be C-contiguous.
        // ensure() returns the same object when it already is, and converts only
        // when it is not, rather than misreading a strided array.
        py::array array = py::array::ensure(item.second, py::array::c_style);
        if (!array) {
            throw std::invalid_argument("feed '" + name + "' is not array-like");
        }

        anytime::TensorView view;
        view.dtype = dtype_from_numpy(array, name);
        view.shape.assign(array.shape(), array.shape() + array.ndim());
        view.data = array.data();

        retained.push_back(std::move(array));
        views.emplace_back(name, std::move(view));
    }

    anytime::RunResult result;
    {
        py::gil_scoped_release release;
        result = engine.run(variant, views);
    }

    py::list outputs;
    for (auto& value : result.outputs) {
        outputs.append(wrap_output(std::move(value)));
    }
    return py::make_tuple(std::move(outputs), result.latency_ms);
}

}  // namespace

PYBIND11_MODULE(anytime_runtime, module) {
    module.doc() =
        "In-process ONNX Runtime engine for the Anytime Inference Planner.\n\n"
        "Tensors cross the boundary as borrowed buffers rather than copies, and "
        "the GIL is released around inference so a pool of workers runs "
        "concurrently.";

    module.def("onnxruntime_version", []() {
        return std::string(OrtGetApiBase()->GetVersionString());
    }, "Version of the ONNX Runtime this extension is linked against.\n\n"
       "Must equal onnxruntime.__version__: both libraries are loaded into this "
       "process, and Stage 1 measured a 7.6x difference in inference time from a "
       "mismatch between them.");

    module.def("ort_api_version", []() { return ORT_API_VERSION; },
               "ORT_API_VERSION of the headers this extension was compiled "
               "against.");

    py::class_<anytime::Engine>(module, "Engine",
        "Holds one ONNX Runtime session per named variant.\n\n"
        "Thread counts default to one of each so that N pooled engines behave as "
        "N independent single-threaded servers, which is what makes the queueing "
        "model in the admission controller valid.")
        .def(py::init([](const std::vector<std::pair<std::string, std::string>>& models,
                         int intra_op_threads, int inter_op_threads) {
                 return std::make_unique<anytime::Engine>(models, intra_op_threads,
                                                          inter_op_threads);
             }),
             py::arg("models"), py::arg("intra_op_threads") = 1,
             py::arg("inter_op_threads") = 1,
             "models is a sequence of (variant, onnx_path) pairs.")
        .def("run", &run_engine, py::arg("variant"), py::arg("feeds"),
             "Run one variant.\n\n"
             "Returns (outputs, latency_ms), where outputs are numpy views over "
             "ONNX Runtime's buffers in the order the graph declares them, and "
             "latency_ms is the time spent inside Session::Run.\n\n"
             "Feeds the graph does not declare are dropped, since variants of one "
             "task can declare different inputs and callers pass the union. A "
             "declared input that is missing raises instead: running on a partial "
             "feed would silently produce wrong output.")
        .def_property_readonly("variants", &anytime::Engine::variants,
                               "Loaded variant names.")
        .def("input_names", &anytime::Engine::input_names, py::arg("variant"),
             "Input names the variant's graph declares.")
        .def("output_names", &anytime::Engine::output_names, py::arg("variant"),
             "Output names the variant's graph declares, in run order.");

    // --- decoder path -------------------------------------------------------

    module.attr("DEFAULT_BLOCK_TOKENS") = anytime::kDefaultBlockTokens;
    module.attr("DEFAULT_PREFILL_CHUNK_TOKENS") = anytime::kDefaultPrefillChunkTokens;

    // Derives from RuntimeError, so the error contract in serving/onnx_runtime.py
    // still holds: this means "the runtime could not serve this request". It is a
    // distinct type because it is the one such error the admission policy is meant
    // to handle -- by evicting or refusing -- rather than propagate.
    py::register_exception<anytime::CacheExhausted>(module, "CacheExhausted",
                                                   PyExc_RuntimeError);

    py::class_<anytime::KvGeometry>(module, "KvGeometry",
        "Shape of a decoder's KV cache, read off the graph rather than a config.")
        .def_readonly("layers", &anytime::KvGeometry::layers)
        .def_readonly("kv_heads", &anytime::KvGeometry::kv_heads)
        .def_readonly("head_dim", &anytime::KvGeometry::head_dim)
        .def_readonly("block_tokens", &anytime::KvGeometry::block_tokens)
        .def_property_readonly("bytes_per_token", &anytime::KvGeometry::bytes_per_token,
                               "Cache bytes one token position costs across every layer.")
        .def_property_readonly("bytes_per_block", &anytime::KvGeometry::bytes_per_block,
                               "Cache bytes one block holds.")
        .def("blocks_for", &anytime::KvGeometry::blocks_for, py::arg("tokens"),
             "Blocks a sequence of this many tokens occupies, rounded up.")
        .def("__repr__", [](const anytime::KvGeometry& geometry) {
            return "KvGeometry(layers=" + std::to_string(geometry.layers) +
                   ", kv_heads=" + std::to_string(geometry.kv_heads) +
                   ", head_dim=" + std::to_string(geometry.head_dim) +
                   ", block_tokens=" + std::to_string(geometry.block_tokens) + ")";
        });

    py::class_<anytime::StepTimings>(module, "StepTimings",
        "Where one step's time went.\n\n"
        "gather_ms is the price of block accounting and is reported separately "
        "because the point is not to assume it is free. run_ms is time inside "
        "Session::Run, matching what Engine.run reports, so the two are "
        "comparable. verify_ms is non-zero only on the step that checks the "
        "present-prefix invariant, once per sequence. pad_ms is non-zero only for a "
        "batched step, where rows shorter than the batch's longest are right-padded "
        "and the padding is cleared; it scales with the batch's length variance "
        "rather than with its size.")
        .def_readonly("gather_ms", &anytime::StepTimings::gather_ms)
        .def_readonly("pad_ms", &anytime::StepTimings::pad_ms)
        .def_readonly("run_ms", &anytime::StepTimings::run_ms)
        .def_readonly("scatter_ms", &anytime::StepTimings::scatter_ms)
        .def_readonly("verify_ms", &anytime::StepTimings::verify_ms)
        .def_readonly("total_ms", &anytime::StepTimings::total_ms);

    py::class_<PyStepResult>(module, "StepResult",
        "Result of one prefill or one decode step.\n\n"
        "logits holds the distribution for the next token only. The graph returns "
        "one row per position it was given -- 206 MB for a 1024-token prefill -- "
        "when sampling reads one row of 50257, so the last row is copied out and "
        "the rest is dropped.")
        .def_readonly("logits", &PyStepResult::logits)
        .def_readonly("timings", &PyStepResult::timings)
        .def_readonly("length", &PyStepResult::length,
                      "Tokens in the cache after this step.")
        .def_readonly("runs", &PyStepResult::runs,
                      "Graph invocations. Greater than one for a chunked prefill.");

    py::class_<PyBatchStepResult>(module, "BatchStepResult",
        "Result of one batched decode step: one row per sequence, one set of "
        "timings for the step.\n\n"
        "The timings are not per sequence and are not divided by the batch size. "
        "The rows share a single Session::Run, so attributing part of it to one of "
        "them would produce a number that reads as per-sequence and is not; "
        "len(rows) is the divisor for an average, stated as an average.\n\n"
        "Batching a decode step pays, and how much depends on how full the caches "
        "are: only the cache-independent term of a step amortises across a batch, "
        "while the per-cached-token term is per sequence and grows with the batch's "
        "total cache. The measured curve lives in docs/benchmarks.md rather than "
        "here. It is deliberately not quoted in this docstring -- an earlier version "
        "cited a Run-only probe that measurement through the scheduler later "
        "overturned by about 20%, and a number embedded in a C++ string literal is "
        "the last copy anyone thinks to update.")
        .def_readonly("rows", &PyBatchStepResult::rows,
                      "One StepResult per sequence, in the order they were given.")
        .def_readonly("timings", &PyBatchStepResult::timings,
                      "Timings for the whole step, not for any one row.");

    py::class_<anytime::DecoderSession>(module, "DecoderSession",
        "Decoder-only inference over a block-allocated KV cache.\n\n"
        "The cache is a host-side block allocator, not paged attention: an "
        "exported decoder takes past_key_values as graph inputs and ONNX Runtime "
        "allocates the present tensors itself, so a sequence's blocks are gathered "
        "into the batch-shaped tensor before each run and the new tail is copied "
        "back afterwards.\n\n"
        "The arena is fixed at construction. That is what makes admission a "
        "decision: open() returns False when there is no room, and a sequence that "
        "outgrows its reservation mid-decode raises CacheExhausted.")
        .def(py::init([](const std::string& path, int block_tokens, std::size_t num_blocks,
                         int intra_op_threads, int inter_op_threads, int copy_threads,
                         std::size_t parallel_copy_floor) {
                 return std::make_unique<anytime::DecoderSession>(
                     path, block_tokens, num_blocks, intra_op_threads, inter_op_threads,
                     copy_threads, parallel_copy_floor);
             }),
             py::arg("path"), py::arg("block_tokens") = anytime::kDefaultBlockTokens,
             py::arg("num_blocks") = 256, py::arg("intra_op_threads") = 1,
             py::arg("inter_op_threads") = 1, py::arg("copy_threads") = 1,
             py::arg("parallel_copy_floor") = anytime::kDefaultParallelCopyFloor)
        .def_property_readonly("copy_threads", &anytime::DecoderSession::copy_threads,
                               "Runners the gather may split across, the calling thread "
                               "included. One means the copy is a plain loop.")
        .def_property_readonly("parallel_copy_floor",
                               &anytime::DecoderSession::parallel_copy_floor,
                               "Staged floats below which the gather runs inline whatever "
                               "the copy pool holds.")
        .def_property_readonly("geometry", &anytime::DecoderSession::geometry,
                               py::return_value_policy::copy)
        .def_property_readonly("capacity_blocks", &anytime::DecoderSession::capacity_blocks)
        .def_property_readonly("free_blocks", &anytime::DecoderSession::free_blocks)
        .def_property_readonly("arena_bytes", &anytime::DecoderSession::arena_bytes)
        .def_property_readonly("declares_attention_mask",
                               &anytime::DecoderSession::declares_attention_mask)
        .def_property_readonly("declares_position_ids",
                               &anytime::DecoderSession::declares_position_ids)
        .def("blocks_for", &anytime::DecoderSession::blocks_for, py::arg("tokens"))
        .def("open", &anytime::DecoderSession::open, py::arg("sequence_id"),
             py::arg("reserve_tokens"),
             "Reserve blocks for a sequence.\n\n"
             "Returns False when the arena cannot supply them, leaving the pool and "
             "every other sequence untouched. Refusing is the admission "
             "controller's answer, not an exception.")
        .def("release", &anytime::DecoderSession::release, py::arg("sequence_id"),
             "Free a sequence's blocks, returning how many. Idempotent.")
        .def("contains", &anytime::DecoderSession::contains, py::arg("sequence_id"))
        .def("length", &anytime::DecoderSession::length, py::arg("sequence_id"),
             "Tokens cached for this sequence.")
        .def("blocks_held", &anytime::DecoderSession::blocks_held, py::arg("sequence_id"))
        .def_property_readonly("sequences", &anytime::DecoderSession::sequences,
                               "Open sequence ids.")
        .def("prefill",
             [](anytime::DecoderSession& self, const std::string& sequence_id,
                const std::vector<std::int64_t>& tokens, int chunk_tokens) {
                 anytime::StepResult result;
                 {
                     py::gil_scoped_release release;
                     result = self.prefill(sequence_id, tokens, chunk_tokens);
                 }
                 return make_step_result(std::move(result));
             },
             py::arg("sequence_id"), py::arg("tokens"),
             py::arg("chunk_tokens") = anytime::kDefaultPrefillChunkTokens,
             "Run a prompt through the graph, filling the cache.\n\n"
             "Chunked by default: one pass over 1024 GPT-2 tokens measured 433.6 ms "
             "and allocated 206 MB of logits the sampler never reads, while four "
             "passes of 256 measured 372.2 ms with a 51 MB peak. Pass "
             "chunk_tokens=0 for a single pass.")
        .def("extend",
             [](anytime::DecoderSession& self, const std::string& sequence_id,
                const std::vector<std::int64_t>& tokens) {
                 anytime::StepResult result;
                 {
                     py::gil_scoped_release release;
                     result = self.extend(sequence_id, tokens);
                 }
                 return make_step_result(std::move(result));
             },
             py::arg("sequence_id"), py::arg("tokens"),
             "Run tokens through the graph, appending to what the sequence holds.\n\n"
             "Prefill's inner step. prefill() loops over the chunks itself and "
             "refuses a non-empty sequence, which is what a caller wanting a whole "
             "prompt run needs. A scheduler needs the chunks driven from outside, "
             "because the chunk boundary is where a long prefill can be interrupted: "
             "otherwise a resident sequence stalls for a whole prompt rather than "
             "for one chunk, 372 ms against 93 ms on GPT-2 at FP32.\n\n"
             "Reserves per chunk rather than for the whole prompt, so ask admission "
             "before driving a prompt through this.")
        .def("decode",
             [](anytime::DecoderSession& self, const std::string& sequence_id,
                std::int64_t token) {
                 anytime::StepResult result;
                 {
                     py::gil_scoped_release release;
                     result = self.decode(sequence_id, token);
                 }
                 return make_step_result(std::move(result));
             },
             py::arg("sequence_id"), py::arg("token"),
             "Extend a sequence by one token, reading the cache for the rest.")
        .def("decode_batch",
             [](anytime::DecoderSession& self, const std::vector<std::string>& sequence_ids,
                const std::vector<std::int64_t>& tokens) {
                 anytime::BatchStepResult result;
                 {
                     py::gil_scoped_release release;
                     result = self.decode_batch(sequence_ids, tokens);
                 }
                 return make_batch_step_result(std::move(result));
             },
             py::arg("sequence_ids"), py::arg("tokens"),
             "Extend each sequence by one token, in a single graph invocation.\n\n"
             "Decode only. The graph takes one sequence dimension as well as one "
             "past_sequence_length, so a prefill chunk and a decode step cannot "
             "share a run without padding the decode row out to the chunk width; a "
             "scheduler over this alternates rather than fusing.\n\n"
             "Rows are right-padded to the longest past in the batch, with a "
             "per-row attention_mask and true absolute position_ids. Every sequence "
             "must be open and non-empty and no id may repeat. Blocks are all or "
             "nothing: a batch that does not fit raises CacheExhausted with the "
             "arena untouched.");
}
