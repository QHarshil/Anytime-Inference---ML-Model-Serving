// Element types and borrowed tensor views.
//
// The engine never owns input buffers. Callers hand it a pointer, a shape, and a
// dtype; ONNX Runtime reads through that pointer during Run. This is what makes
// the numpy boundary free of copies, and it means the caller has to keep the
// buffer alive for the duration of the call. The bindings do that by holding the
// numpy references on the stack across the whole run.

#ifndef ANYTIME_TENSOR_HPP
#define ANYTIME_TENSOR_HPP

#include <onnxruntime_cxx_api.h>

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace anytime {

// Element types accepted on graph inputs. Deliberately narrow: exported graphs
// declare int64 token ids and masks and float32 activations, and an unsupported
// dtype has to fail loudly rather than be reinterpreted. Outputs are mapped back
// from whatever the graph produces, which is a wider set.
enum class DType {
    Float32,
    Float64,
    Int32,
    Int64,
    Bool,
};

std::size_t dtype_size(DType dtype);
const char* dtype_name(DType dtype);
ONNXTensorElementDataType dtype_to_ort(DType dtype);
DType dtype_from_ort(ONNXTensorElementDataType type);

// Name of an ONNX element type, for error messages about types the engine does
// not accept.
std::string ort_type_name(ONNXTensorElementDataType type);

// A borrowed, C-contiguous buffer. Non-owning by design; see the file comment.
struct TensorView {
    const void* data = nullptr;
    std::vector<int64_t> shape;
    DType dtype = DType::Float32;

    std::size_t element_count() const;
    std::size_t byte_count() const;
};

// Wraps a borrowed buffer as an Ort::Value. No copy: the returned value points
// into view.data, which must outlive it.
Ort::Value borrow_as_tensor(const TensorView& view, const OrtMemoryInfo* memory);

}  // namespace anytime

#endif  // ANYTIME_TENSOR_HPP
