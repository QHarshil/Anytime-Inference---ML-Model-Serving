cmake_minimum_required(VERSION 3.14)
project(anytime_runtime CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Locate ONNX Runtime. Either set ONNXRUNTIME_ROOT_PATH to the unpacked release
# tarball (which contains include/ and lib/) or install ONNX Runtime such that
# CMake can find it via find_package.
if(NOT DEFINED ONNXRUNTIME_ROOT_PATH AND DEFINED ENV{ONNXRUNTIME_ROOT_PATH})
    set(ONNXRUNTIME_ROOT_PATH $ENV{ONNXRUNTIME_ROOT_PATH})
endif()

if(NOT ONNXRUNTIME_ROOT_PATH)
    message(FATAL_ERROR
        "ONNXRUNTIME_ROOT_PATH is not set. Download ONNX Runtime from "
        "https://github.com/microsoft/onnxruntime/releases and pass "
        "-DONNXRUNTIME_ROOT_PATH=/path/to/onnxruntime to cmake.")
endif()

add_executable(anytime_runtime src/main.cpp)

target_include_directories(anytime_runtime PRIVATE ${ONNXRUNTIME_ROOT_PATH}/include)

find_library(ONNXRUNTIME_LIB
    NAMES onnxruntime
    PATHS ${ONNXRUNTIME_ROOT_PATH}/lib
    NO_DEFAULT_PATH
    REQUIRED)

target_link_libraries(anytime_runtime PRIVATE ${ONNXRUNTIME_LIB})

if(APPLE)
    set_target_properties(anytime_runtime PROPERTIES
        BUILD_RPATH "${ONNXRUNTIME_ROOT_PATH}/lib"
        INSTALL_RPATH "${ONNXRUNTIME_ROOT_PATH}/lib")
elseif(UNIX)
    target_link_options(anytime_runtime PRIVATE
        "-Wl,-rpath,${ONNXRUNTIME_ROOT_PATH}/lib")
endif()
