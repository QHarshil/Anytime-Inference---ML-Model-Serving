# Resolve an ONNX Runtime SDK that matches the installed onnxruntime wheel.
#
# The extension and the Python wheel run in one process, so their ONNX Runtime
# versions cannot be allowed to differ. Stage 1 built the worker against 1.20.1
# while the wheel was 1.26.0 and measured DistilBERT at 98.9 ms against 13.0 ms
# inside session->Run(); every service time the planner used was wrong by almost
# an order of magnitude. Under pybind11 a mismatch is worse than a performance
# cliff, because both copies of the library are loaded at once.
#
# The version is therefore never written down twice. It is read from the wheel
# that the target interpreter will import at runtime, and everything else is
# derived from it:
#
#   1. ONNXRUNTIME_ROOT_PATH, if set, must contain a VERSION_NUMBER equal to the
#      wheel version. Configuration fails otherwise.
#   2. Otherwise the matching release archive is downloaded once into a cache
#      directory and reused.
#
# Sets, in the parent scope:
#   ANYTIME_ORT_ROOT         directory holding include/ and lib/
#   ANYTIME_ORT_VERSION      version string, as reported by the wheel
#   ANYTIME_ORT_LIBRARY      full path to the shared library
#   ANYTIME_ORT_API_VERSION  ORT_API_VERSION as written in the resolved headers

include_guard(GLOBAL)

function(_anytime_ort_wheel_version out_var)
    if(NOT Python_EXECUTABLE)
        message(FATAL_ERROR
            "Python_EXECUTABLE is not set, so the onnxruntime wheel version cannot "
            "be read. Build through pip (which supplies it via scikit-build-core) "
            "or pass -DPython_EXECUTABLE=/path/to/python.")
    endif()

    # Probed twice, in order of authority.
    #
    # First with PYTHONPATH cleared. pip builds in an isolated environment layered
    # onto the target interpreter through that variable, and the overlay carries
    # its own onnxruntime, which can be a different version from the one the target
    # environment imports at runtime. Reading the overlay when the target already
    # has a wheel would reintroduce the mismatch this module exists to prevent,
    # through the build system instead of through the developer.
    execute_process(
        COMMAND "${CMAKE_COMMAND}" -E env --unset=PYTHONPATH
                "${Python_EXECUTABLE}" -c
                "import onnxruntime; print(onnxruntime.__version__)"
        OUTPUT_VARIABLE version
        ERROR_VARIABLE  target_error
        RESULT_VARIABLE probe_status
        OUTPUT_STRIP_TRAILING_WHITESPACE)

    if(probe_status EQUAL 0)
        message(STATUS "onnxruntime resolved from the target environment")
        set(${out_var} "${version}" PARENT_SCOPE)
        return()
    endif()

    # Then with the build environment left intact. On a first install into a clean
    # environment there is no wheel to read yet: pip installs build requirements,
    # builds, and only then installs runtime dependencies. The build requirement
    # pins the same specifier as the runtime dependency, so the version resolved
    # here is the one pip is about to install. load_extension() re-checks the
    # equality at import time, which is what catches the case where it is not.
    execute_process(
        COMMAND "${Python_EXECUTABLE}" -c
                "import onnxruntime; print(onnxruntime.__version__)"
        OUTPUT_VARIABLE version
        ERROR_VARIABLE  build_error
        RESULT_VARIABLE probe_status
        OUTPUT_STRIP_TRAILING_WHITESPACE)

    if(probe_status EQUAL 0)
        message(STATUS
            "onnxruntime is not installed in the target environment yet; resolved "
            "${version} from the build environment instead. The version is "
            "re-checked when anytime_runtime is imported.")
        set(${out_var} "${version}" PARENT_SCOPE)
        return()
    endif()

    message(FATAL_ERROR
        "Could not import onnxruntime with ${Python_EXECUTABLE}, with or without "
        "the build environment on the path. The extension is linked against the "
        "same ONNX Runtime that interpreter imports at runtime, so a wheel has to "
        "be resolvable:\n"
        "    ${Python_EXECUTABLE} -m pip install onnxruntime\n"
        "target environment:\n${target_error}\n"
        "build environment:\n${build_error}")
endfunction()

# Release archives are published per platform under a stable naming scheme.
function(_anytime_ort_archive_name version out_name out_extension)
    if(APPLE)
        if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64")
            set(platform "osx-arm64")
        else()
            set(platform "osx-x86_64")
        endif()
        set(extension "tgz")
    elseif(UNIX)
        if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
            set(platform "linux-aarch64")
        else()
            set(platform "linux-x64")
        endif()
        set(extension "tgz")
    else()
        # Windows archives are zips with a different internal layout. Rather than
        # guess, ask for an explicit SDK.
        message(FATAL_ERROR
            "Automatic ONNX Runtime download is implemented for macOS and Linux "
            "only. Download the ${version} release for this platform from "
            "https://github.com/microsoft/onnxruntime/releases and pass "
            "-DONNXRUNTIME_ROOT_PATH=/path/to/onnxruntime.")
    endif()

    set(${out_name} "onnxruntime-${platform}-${version}" PARENT_SCOPE)
    set(${out_extension} "${extension}" PARENT_SCOPE)
endfunction()

# The unpacked tree is the directory holding include/. Some releases nest a
# versioned directory inside the archive and some do not, so locate it rather
# than assuming either layout.
function(_anytime_ort_find_root search_dir out_var)
    if(EXISTS "${search_dir}/include/onnxruntime_c_api.h")
        set(${out_var} "${search_dir}" PARENT_SCOPE)
        return()
    endif()
    file(GLOB candidates LIST_DIRECTORIES true "${search_dir}/*")
    foreach(candidate IN LISTS candidates)
        if(IS_DIRECTORY "${candidate}" AND
           EXISTS "${candidate}/include/onnxruntime_c_api.h")
            set(${out_var} "${candidate}" PARENT_SCOPE)
            return()
        endif()
    endforeach()
    set(${out_var} "" PARENT_SCOPE)
endfunction()

function(_anytime_ort_download version out_root)
    _anytime_ort_archive_name("${version}" archive_name archive_extension)

    # A durable cache, so the SDK is fetched once per version per machine rather
    # than once per build tree. Stage 1 kept it in a temporary directory and lost
    # it between sessions.
    if(DEFINED ENV{ANYTIME_ORT_CACHE})
        set(cache_root "$ENV{ANYTIME_ORT_CACHE}")
    elseif(DEFINED ENV{XDG_CACHE_HOME})
        set(cache_root "$ENV{XDG_CACHE_HOME}/anytime-inference-planner")
    else()
        set(cache_root "$ENV{HOME}/.cache/anytime-inference-planner")
    endif()
    set(unpacked "${cache_root}/${archive_name}")

    _anytime_ort_find_root("${unpacked}" cached_root)
    if(cached_root)
        message(STATUS "ONNX Runtime ${version}: using cached SDK at ${cached_root}")
        set(${out_root} "${cached_root}" PARENT_SCOPE)
        return()
    endif()

    set(url "https://github.com/microsoft/onnxruntime/releases/download/v${version}/${archive_name}.${archive_extension}")
    set(archive "${cache_root}/${archive_name}.${archive_extension}")
    message(STATUS "ONNX Runtime ${version}: downloading ${url}")

    file(MAKE_DIRECTORY "${cache_root}")
    file(DOWNLOAD "${url}" "${archive}" STATUS download_status SHOW_PROGRESS)
    list(GET download_status 0 download_code)
    if(NOT download_code EQUAL 0)
        list(GET download_status 1 download_message)
        file(REMOVE "${archive}")
        message(FATAL_ERROR
            "Failed to download ONNX Runtime ${version} from ${url}: "
            "${download_message}\nDownload it manually and pass "
            "-DONNXRUNTIME_ROOT_PATH=/path/to/onnxruntime instead.")
    endif()

    file(MAKE_DIRECTORY "${unpacked}")
    file(ARCHIVE_EXTRACT INPUT "${archive}" DESTINATION "${unpacked}")
    file(REMOVE "${archive}")

    _anytime_ort_find_root("${unpacked}" downloaded_root)
    if(NOT downloaded_root)
        message(FATAL_ERROR
            "Downloaded ONNX Runtime ${version} but found no include/ directory "
            "under ${unpacked}.")
    endif()
    set(${out_root} "${downloaded_root}" PARENT_SCOPE)
endfunction()

function(anytime_resolve_onnxruntime)
    _anytime_ort_wheel_version(wheel_version)
    message(STATUS "onnxruntime wheel reports version ${wheel_version}")

    if(NOT ONNXRUNTIME_ROOT_PATH AND DEFINED ENV{ONNXRUNTIME_ROOT_PATH})
        set(ONNXRUNTIME_ROOT_PATH "$ENV{ONNXRUNTIME_ROOT_PATH}")
    endif()

    if(ONNXRUNTIME_ROOT_PATH)
        _anytime_ort_find_root("${ONNXRUNTIME_ROOT_PATH}" root)
        if(NOT root)
            message(FATAL_ERROR
                "ONNXRUNTIME_ROOT_PATH=${ONNXRUNTIME_ROOT_PATH} contains no "
                "include/onnxruntime_c_api.h.")
        endif()

        if(NOT EXISTS "${root}/VERSION_NUMBER")
            message(FATAL_ERROR
                "${root} has no VERSION_NUMBER, so its version cannot be checked "
                "against the ${wheel_version} wheel. Point ONNXRUNTIME_ROOT_PATH at "
                "an unpacked release archive, or unset it to download one.")
        endif()
        file(READ "${root}/VERSION_NUMBER" sdk_version)
        string(STRIP "${sdk_version}" sdk_version)

        if(NOT sdk_version VERSION_EQUAL wheel_version)
            message(FATAL_ERROR
                "ONNX Runtime version mismatch. The SDK at ${root} is "
                "${sdk_version}; the onnxruntime wheel that will be imported in the "
                "same process is ${wheel_version}. These must be equal: the two "
                "libraries are loaded together, and Stage 1 measured a 7.6x "
                "difference in session->Run() from exactly this mismatch. Unset "
                "ONNXRUNTIME_ROOT_PATH to download the matching SDK.")
        endif()
        message(STATUS "ONNX Runtime ${sdk_version}: using SDK at ${root}")
    else()
        _anytime_ort_download("${wheel_version}" root)
    endif()

    # find_library caches its result, so a changed root would otherwise keep
    # linking the previously resolved library against the new headers. That
    # mismatch surfaces only at runtime, as "The requested API version [N] is not
    # available".
    if(DEFINED CACHE{ANYTIME_ORT_LIBRARY} AND
       NOT ANYTIME_ORT_LIBRARY MATCHES "^${root}/")
        message(STATUS "ONNX Runtime root changed; re-resolving the library")
        unset(ANYTIME_ORT_LIBRARY CACHE)
    endif()
    find_library(ANYTIME_ORT_LIBRARY
        NAMES onnxruntime
        PATHS "${root}/lib"
        NO_DEFAULT_PATH
        REQUIRED)

    # Read ORT_API_VERSION out of the headers that were just resolved and assert
    # it in the translation units. This catches a stray onnxruntime_c_api.h
    # arriving from some other include path, without assuming anything about how
    # release versions map onto API versions.
    file(STRINGS "${root}/include/onnxruntime_c_api.h" api_version_line
         REGEX "^#define ORT_API_VERSION[ \t]+[0-9]+")
    if(NOT api_version_line)
        message(FATAL_ERROR
            "Could not read ORT_API_VERSION from "
            "${root}/include/onnxruntime_c_api.h.")
    endif()
    string(REGEX MATCH "[0-9]+" api_version "${api_version_line}")

    message(STATUS "ONNX Runtime library: ${ANYTIME_ORT_LIBRARY}")
    message(STATUS "ONNX Runtime API version: ${api_version}")

    set(ANYTIME_ORT_ROOT "${root}" PARENT_SCOPE)
    set(ANYTIME_ORT_VERSION "${wheel_version}" PARENT_SCOPE)
    set(ANYTIME_ORT_API_VERSION "${api_version}" PARENT_SCOPE)
endfunction()
