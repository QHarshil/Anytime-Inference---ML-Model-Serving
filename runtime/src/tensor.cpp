#include "anytime/tensor.hpp"

#include <stdexcept>

// The headers this compiles against must be the ones CMake resolved from the SDK
// that matches the onnxruntime wheel. A stray onnxruntime_c_api.h arriving from
// another include path would compile cleanly and then disagree with the linked
// library at runtime, which is the failure mode this whole build path exists to
// prevent.
#ifndef ANYTIME_EXPECTED_ORT_API_VERSION
#error "ANYTIME_EXPECTED_ORT_API_VERSION is not defined; configure through runtime/CMakeLists.txt"
#endif
static_assert(ORT_API_VERSION == ANYTIME_EXPECTED_ORT_API_VERSION,
              "onnxruntime_c_api.h does not match the SDK CMake resolved. The "
              "include path is picking up a different ONNX Runtime.");

namespace anytime {

std::size_t dtype_size(DType dtype) {
    switch (dtype) {
        case DType::Float32: return sizeof(float);
        case DType::Float64: return sizeof(double);
        case DType::Int32:   return sizeof(int32_t);
        case DType::Int64:   return sizeof(int64_t);
        case DType::Bool:    return sizeof(bool);
    }
    throw std::runtime_error("unhandled dtype");
}

const char* dtype_name(DType dtype) {
    switch (dtype) {
        case DType::Float32: return "float32";
        case DType::Float64: return "float64";
        case DType::Int32:   return "int32";
        case DType::Int64:   return "int64";
        case DType::Bool:    return "bool";
    }
    throw std::runtime_error("unhandled dtype");
}

ONNXTensorElementDataType dtype_to_ort(DType dtype) {
    switch (dtype) {
        case DType::Float32: return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
        case DType::Float64: return ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE;
        case DType::Int32:   return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32;
        case DType::Int64:   return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
        case DType::Bool:    return ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL;
    }
    throw std::runtime_error("unhandled dtype");
}

DType dtype_from_ort(ONNXTensorElementDataType type) {
    switch (type) {
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:  return DType::Float32;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE: return DType::Float64;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:  return DType::Int32;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:  return DType::Int64;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:   return DType::Bool;
        default:
            throw std::runtime_error("unsupported ONNX element type: " +
                                     ort_type_name(type));
    }
}

std::string ort_type_name(ONNXTensorElementDataType type) {
    switch (type) {
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:      return "float32";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:     return "float64";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:    return "float16";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16:   return "bfloat16";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:       return "int8";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:      return "uint8";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:      return "int16";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:     return "uint16";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:      return "int32";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:     return "uint32";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:      return "int64";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:     return "uint64";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:       return "bool";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING:     return "string";
        default:                                       return "unknown";
    }
}

std::size_t TensorView::element_count() const {
    std::size_t count = 1;
    for (const int64_t dimension : shape) {
        if (dimension < 0) {
            throw std::runtime_error("tensor shape has a negative dimension");
        }
        count *= static_cast<std::size_t>(dimension);
    }
    return count;
}

std::size_t TensorView::byte_count() const {
    return element_count() * dtype_size(dtype);
}

Ort::Value borrow_as_tensor(const TensorView& view, const OrtMemoryInfo* memory) {
    if (view.data == nullptr) {
        throw std::runtime_error("tensor view has no data pointer");
    }
    // CreateTensor over caller memory: ONNX Runtime reads through the pointer and
    // does not take ownership, so nothing is copied on the way in.
    return Ort::Value::CreateTensor(
        memory,
        const_cast<void*>(view.data),
        view.byte_count(),
        view.shape.data(),
        view.shape.size(),
        dtype_to_ort(view.dtype));
}

}  // namespace anytime
