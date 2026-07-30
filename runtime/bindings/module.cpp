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

#include "anytime/engine.hpp"
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
}
