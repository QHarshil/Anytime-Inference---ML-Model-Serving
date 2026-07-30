cmake_minimum_required(VERSION 3.20)
project(anytime_runtime CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release)
endif()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)

# Resolves an SDK whose version equals the onnxruntime wheel's. See the module
# for why that equality is enforced at configure time rather than documented.
include(ResolveOnnxRuntime)
anytime_resolve_onnxruntime()

add_library(anytime_core STATIC
    src/tensor.cpp
    src/engine.cpp
    src/kv_cache.cpp
    src/decoder.cpp)

target_include_directories(anytime_core
    PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
        "${ANYTIME_ORT_ROOT}/include")

# Asserted in tensor.cpp against the value ORT_API_VERSION actually has, so a
# header arriving from some other include path fails the build rather than the
# run.
target_compile_definitions(anytime_core
    PUBLIC ANYTIME_EXPECTED_ORT_API_VERSION=${ANYTIME_ORT_API_VERSION})

target_link_libraries(anytime_core PUBLIC "${ANYTIME_ORT_LIBRARY}")

pybind11_add_module(anytime_runtime MODULE bindings/module.cpp)
target_link_libraries(anytime_runtime PRIVATE anytime_core)

# The extension loads libonnxruntime from the resolved SDK. The path is absolute
# because the SDK lives outside the wheel; ONNXRUNTIME_ROOT_PATH is how a
# relocatable build points somewhere else.
cmake_path(GET ANYTIME_ORT_LIBRARY PARENT_PATH ANYTIME_ORT_LIBDIR)
if(APPLE)
    set_target_properties(anytime_runtime PROPERTIES
        BUILD_RPATH "${ANYTIME_ORT_LIBDIR}"
        INSTALL_RPATH "${ANYTIME_ORT_LIBDIR}")
elseif(UNIX)
    set_target_properties(anytime_runtime PROPERTIES
        BUILD_RPATH "${ANYTIME_ORT_LIBDIR}"
        INSTALL_RPATH "${ANYTIME_ORT_LIBDIR}")
endif()

install(TARGETS anytime_runtime DESTINATION .)
