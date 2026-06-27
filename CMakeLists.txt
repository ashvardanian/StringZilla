# StringZilla CMakeLists.txt
#
# This file defines several library build & installation targets:
#
# * stringzilla_header: A header-only library with the StringZilla C and C++ headers.
# * stringzilla_shared: A shared library with the StringZilla C and C++ headers and dynamic SIMD dispatch.
# * stringzilla_bare: A shared library with the StringZilla headers, but without linking the standard C library.
# * stringzillas_cpus_shared / stringzillas_cpus_static: Shared and static libraries with the StringZillas parallel
#   algorithms for multi-threaded CPUs. Both reuse one stringzillas_cpus_objects OBJECT compilation; their ISA coverage
#   is identical because each per-ISA backend is selected at runtime via per-function target attributes.
# * stringzillas_cuda_shared / stringzillas_cuda_static: Shared and static libraries with the StringZillas parallel
#   algorithms for CUDA-capable GPUs. The static library (for local testing and benchmarking) ships real SASS for the
#   current and older compatible GPUs; the shared library (for distribution) also packages forward-compatible PTX.
# * stringzillas_rocm_shared: A shared library with the StringZillas parallel algorithms for ROCm-capable GPUs (TODO).
#
# Tests for different C++ standards:
#
# * stringzilla_test_cpp11: C++11 baseline support.
# * stringzilla_test_cpp14: C++14 support with `std::less<std::string>`-like function objects.
# * stringzilla_test_cpp17: C++17 support with `std::string_view` compatibility.
# * stringzilla_test_cpp20: C++20 support with `<=>` operator and more `constexpr` features.
#
# Tests for parallel algorithms:
#
# * stringzillas_test_cpp17 / stringzillas_test_cpp20: Parallel-algorithm tests for the host's best CPU backend.
# * stringzillas_test_cu17 / stringzillas_test_cu20: Parallel-algorithm tests on CUDA-capable GPUs (base SIMT, Kepler,
#   and Hopper tiers selected at runtime).
#
# Serial Benchmarks:
#
# * stringzilla_bench_find_cpp20: A benchmark for substring search operations.
# * stringzilla_bench_sequence_cpp20: A benchmark for string array-level operations.
# * stringzilla_bench_token_cpp20: A benchmark for comparators and hash functions.
# * stringzilla_bench_container_cpp20: A benchmark for STL containers powered by StringZilla.
# * stringzilla_bench_memory_cpp20: A benchmark for LibC-style low-level memory operations.
# * stringzilla_bench_utf8_uncased_cpp20: A benchmark for the `sz_utf8_uncased_*` folding/search family.
# * stringzilla_bench_utf8_iterate_cpp20: A benchmark for the `sz_utf8_*` iteration/segmentation family.
# * stringzilla_bench_utf8_norm_cpp20: A benchmark for the `sz_utf8_norm` normalization family.
# * stringzilla_bench_utf8_delimiters_cpp20: A benchmark for the `sz_find_delimiter_utf8` family.
#
# Parallel Benchmarks:
#
# * stringzillas_bench_similarities_cpp20: A benchmark for similarity operations.
# * stringzillas_bench_similarities_cu20: A benchmark for similarity operations on GPU.
# * stringzillas_bench_fingerprints_cpp20: A benchmark for finding many substrings.
# * stringzillas_bench_fingerprints_cu20: A benchmark for finding many substrings on GPU.
#
# For higher-level language bindings separate build scripts are provided, native to each toolchain.
cmake_minimum_required(VERSION 3.14 FATAL_ERROR)
project(
    stringzilla
    VERSION 4.6.2
    LANGUAGES C CXX ASM
    DESCRIPTION "Search, hash, sort, fingerprint, and fuzzy-match strings faster via SWAR, SIMD, and GPGPU"
    HOMEPAGE_URL "https://github.com/ashvardanian/stringzilla"
)

set(CMAKE_C_STANDARD 99)
set(CMAKE_CXX_STANDARD 11)

set(CMAKE_C_EXTENSIONS OFF)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_COMPILE_WARNING_AS_ERROR)
set(DEV_USER_NAME $ENV{USER})

message(STATUS "C Compiler ID: ${CMAKE_C_COMPILER_ID}")
message(STATUS "C Compiler Version: ${CMAKE_C_COMPILER_VERSION}")
message(STATUS "C Compiler: ${CMAKE_C_COMPILER}")
message(STATUS "C++ Compiler ID: ${CMAKE_CXX_COMPILER_ID}")
message(STATUS "C++ Compiler Version: ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "C++ Compiler: ${CMAKE_CXX_COMPILER}")

# Detect CUDA Support
set(STRINGZILLA_CAN_BUILD_CUDA OFF)
include(CheckLanguage)
check_language(CUDA)
if (CMAKE_CUDA_COMPILER)
    set(STRINGZILLA_CAN_BUILD_CUDA ON)
    message(STATUS "CUDA compiler available")
else ()
    message(STATUS "CUDA compiler not available")
endif ()

if (CMAKE_SIZEOF_VOID_P EQUAL 8)
    message(STATUS "Pointer size: 64-bit")
else ()
    message(STATUS "Pointer size: 32-bit")
endif ()

# Set a default build type to "Release" if none was specified
if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(STATUS "Setting build type to 'Release' as none was specified.")
    set(CMAKE_BUILD_TYPE
        Release
        CACHE STRING "Choose the type of build." FORCE
    )
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif ()
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

# MSVC does not set `CMAKE_SYSTEM_PROCESSOR` correctly so use `CMAKE_<LANG>_COMPILER_ARCHITECTURE_ID` instead
if (MSVC)
    if (CMAKE_C_COMPILER_ARCHITECTURE_ID MATCHES "x64")
        set(CMAKE_SYSTEM_PROCESSOR "AMD64")
    elseif (CMAKE_C_COMPILER_ARCHITECTURE_ID MATCHES "X86")
        set(CMAKE_SYSTEM_PROCESSOR "X86")
    elseif (CMAKE_C_COMPILER_ARCHITECTURE_ID MATCHES "ARM64")
        set(CMAKE_SYSTEM_PROCESSOR "ARM64")
    else ()
        message(WARNING "Unknown CMAKE_C_COMPILER_ARCHITECTURE_ID=${CMAKE_C_COMPILER_ARCHITECTURE_ID}")
    endif ()
endif ()

# Detect target architecture from `CMAKE_SYSTEM_PROCESSOR`... safer for cross-compilation!
if (CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|X86_64|AMD64|amd64")
    set(SZ_IS_64BIT_X86_ TRUE)
    message(STATUS "Platform: x86_64 (CMAKE_SYSTEM_PROCESSOR=${CMAKE_SYSTEM_PROCESSOR})")
elseif (CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|AARCH64|arm64|ARM64")
    set(SZ_IS_64BIT_ARM_ TRUE)
    message(STATUS "Platform: ARM64 (CMAKE_SYSTEM_PROCESSOR=${CMAKE_SYSTEM_PROCESSOR})")
elseif (CMAKE_SYSTEM_PROCESSOR MATCHES "riscv64|RISCV64")
    set(SZ_IS_64BIT_RISCV_ TRUE)
    message(STATUS "Platform: RISC-V 64 (CMAKE_SYSTEM_PROCESSOR=${CMAKE_SYSTEM_PROCESSOR})")
elseif (CMAKE_SYSTEM_PROCESSOR MATCHES "loongarch64|LOONGARCH64")
    set(SZ_IS_64BIT_LOONGARCH_ TRUE)
    message(STATUS "Platform: LoongArch64 (CMAKE_SYSTEM_PROCESSOR=${CMAKE_SYSTEM_PROCESSOR})")
elseif (CMAKE_SYSTEM_PROCESSOR MATCHES "ppc64le|powerpc64le|PPC64LE|POWERPC64LE")
    set(SZ_IS_64BIT_POWER_ TRUE)
    message(STATUS "Platform: PowerPC64 LE (CMAKE_SYSTEM_PROCESSOR=${CMAKE_SYSTEM_PROCESSOR})")
elseif (CMAKE_SYSTEM_PROCESSOR MATCHES "wasm")
    # WebAssembly is 32-bit; SIMD comes from the toolchain's `-msimd128` / `-mrelaxed-simd`, and the
    # `set_architecture_simd_definitions` wasm branch keys off `CMAKE_SYSTEM_NAME`, not a 64-bit flag.
    message(STATUS "Platform: WebAssembly (CMAKE_SYSTEM_PROCESSOR=${CMAKE_SYSTEM_PROCESSOR})")
else ()
    message(WARNING "Unknown CMAKE_SYSTEM_PROCESSOR: ${CMAKE_SYSTEM_PROCESSOR}")
endif ()

# Determine if StringZilla is built as a sub-project (using `add_subdirectory`) or if it is the main project
set(STRINGZILLA_IS_MAIN_PROJECT OFF)

if (CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
    set(STRINGZILLA_IS_MAIN_PROJECT ON)
endif ()

# Installation options
option(STRINGZILLA_INSTALL "Install CMake targets" OFF)
option(STRINGZILLA_BUILD_TEST "Compile a native unit test in C++" ${STRINGZILLA_IS_MAIN_PROJECT})
option(STRINGZILLA_BUILD_BENCHMARK "Compile a native benchmark in C++" ${STRINGZILLA_IS_MAIN_PROJECT})
option(STRINGZILLA_BUILD_SHARED "Compile a dynamic library" ${STRINGZILLA_IS_MAIN_PROJECT})
option(STRINGZILLAS_BUILD_SHARED "Compile dynamic parallel libraries" ${STRINGZILLA_IS_MAIN_PROJECT})
option(STRINGZILLA_BUILD_CUDA "Build CUDA-accelerated targets" ${STRINGZILLA_CAN_BUILD_CUDA})
option(STRINGZILLA_USE_SANITIZERS "Enable AddressSanitizer and UndefinedBehaviorSanitizer in Debug builds" ON)

# Clang source-based coverage for the header-only library. Recipe (single-arch, native backends only):
#   cmake -B build_cov -DSTRINGZILLA_BUILD_COVERAGE=1 -DSTRINGZILLA_USE_SANITIZERS=0 -DCMAKE_CXX_COMPILER=clang++
#   cmake --build build_cov --target stringzilla_test_cpp17 -j
#   LLVM_PROFILE_FILE=/tmp/sz.profraw ./build_cov/stringzilla_test_cpp17
#   llvm-profdata merge -sparse /tmp/sz.profraw -o /tmp/sz.profdata
#   llvm-cov report ./build_cov/stringzilla_test_cpp17 -instr-profile=/tmp/sz.profdata include/stringzilla/
# `-march=native` compiles out the non-native backends, so a faithful picture needs `llvm-profdata merge`
# across per-ISA builds (native x86 + qemu-aarch64 for NEON/SVE + the RVV qemu toolchain).
option(STRINGZILLA_BUILD_COVERAGE "Instrument with Clang source-based coverage" OFF)
set(STRINGZILLA_TARGET_ARCH
    ""
    CACHE STRING "Architecture to tell the compiler to optimize for (-march)"
)

# Enable CUDA if requested
if (STRINGZILLA_BUILD_CUDA)
    if (NOT STRINGZILLA_CAN_BUILD_CUDA)
        message(FATAL_ERROR "CUDA support requested but CUDA compiler not found")
    endif ()
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 20)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(CMAKE_CUDA_EXTENSIONS OFF)
    set(CMAKE_CUDA_ARCHITECTURES 90a) # Hopper is the newest architecture we specialize for
    set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
    find_package(CUDAToolkit REQUIRED)
    message(STATUS "CUDA support enabled")
    message(STATUS "CUDA Compiler: ${CMAKE_CUDA_COMPILER}")
    message(STATUS "CUDA Compiler ID: ${CMAKE_CUDA_COMPILER_ID}")
    message(STATUS "CUDA Toolkit Version: ${CUDAToolkit_VERSION}")
    message(STATUS "CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
endif ()

# Includes
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake ${CMAKE_MODULE_PATH})
include(ExternalProject)
include(CheckCSourceCompiles)

# Allow CMake 3.13+ to override options when using FetchContent / add_subdirectory
if (POLICY CMP0077)
    cmake_policy(SET CMP0077 NEW)
endif ()

# Configuration
include(GNUInstallDirs)
set(STRINGZILLA_INCLUDE_BUILD_DIR "${PROJECT_SOURCE_DIR}/include/")
set(STRINGZILLA_INCLUDE_INSTALL_DIR "${CMAKE_INSTALL_INCLUDEDIR}")

if (CMAKE_VERSION VERSION_EQUAL 3.13 OR CMAKE_VERSION VERSION_GREATER 3.13)
    include(CTest)
    enable_testing()
endif ()

if (MSVC)
    # Remove /RTC* from MSVC debug flags by default (it will be added back in the set_compiler_flags function) Because
    # /RTC* cannot be used without the crt so it needs to be disabled for that specific target
    string(REGEX REPLACE "/RTC[^ ]*" "" CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG}")
    string(REGEX REPLACE "/RTC[^ ]*" "" CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG}")
endif ()

# Function to set the default compiler-specific flags
function (set_compiler_flags target cpp_standard target_arch compiler_id)
    get_target_property(target_type ${target} TYPE)

    target_include_directories(${target} PRIVATE test bench)
    target_include_directories(${target} PRIVATE fork_union/include)

    # Set output directory for single-configuration generators (like Make)
    set_target_properties(${target} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/$<0:>)
    set_target_properties(${target} PROPERTIES ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/$<0:>)

    # Set output directory for multi-configuration generators (like Visual Studio)
    foreach (config IN LISTS CMAKE_CONFIGURATION_TYPES)
        string(TOUPPER ${config} config_upper)
        set_target_properties(${target} PROPERTIES RUNTIME_OUTPUT_DIRECTORY_${config_upper} ${CMAKE_BINARY_DIR}/$<0:>)
        set_target_properties(${target} PROPERTIES ARCHIVE_OUTPUT_DIRECTORY_${config_upper} ${CMAKE_BINARY_DIR}/$<0:>)
    endforeach ()

    # Set the C++ standard
    if (NOT cpp_standard STREQUAL "")
        if (compiler_id STREQUAL "NVIDIA")
            set_target_properties(${target} PROPERTIES CUDA_STANDARD ${cpp_standard})
        elseif (compiler_id MATCHES "MSVC")
            # For MSVC, explicitly set the /std: flag - don't set CXX_STANDARD property to avoid conflicts
            target_compile_options(${target} PRIVATE "/std:c++${cpp_standard}")
        else ()
            set_target_properties(${target} PROPERTIES CXX_STANDARD ${cpp_standard})
        endif ()
    endif ()

    # Use the `/Zc:__cplusplus` flag to correctly define the `__cplusplus` macro in MSVC
    if (compiler_id MATCHES "MSVC")
        target_compile_options(${target} PRIVATE "/Zc:__cplusplus")
    endif ()

    # Make sure CUDA C++ allows calling `constexpr` from device code
    if (compiler_id STREQUAL "NVIDIA")
        target_compile_options(${target} PRIVATE "--expt-relaxed-constexpr")
    endif ()

    # Maximum warnings level & warnings as error.
    #
    # MSVC uses numeric values: > 4068 for "unknown pragmas". > 4146 for "unary minus operator applied to unsigned type,
    # result still unsigned". We also specify `/utf-8` to properly UTF-8 symbols in tests.
    if (compiler_id STREQUAL "GNU")
        target_compile_options(
            ${target}
            PRIVATE
                "-Wall;-Wextra;-Werror;-Wfatal-errors;-Wno-unknown-pragmas;-Wno-cast-function-type;-Wno-unused-function;-Wno-sign-conversion"
        )
        target_compile_options(${target} PRIVATE "-Wno-cast-function-type;-Wno-unused-function") # ? Unique to GCC
    elseif (compiler_id STREQUAL "Clang" OR compiler_id STREQUAL "AppleClang")
        target_compile_options(
            ${target} PRIVATE "-Wall;-Wextra;-Werror;-Wfatal-errors;-Wno-unknown-pragmas;-Wno-sign-conversion"
        )
    elseif (compiler_id MATCHES "MSVC")
        target_compile_options(
            ${target}
            PRIVATE "/Bt" # Display build timings
                    "/wd4068" # Disable warning: unknown pragma
                    "/wd4146" # Disable warning: unary minus operator applied to unsigned type
                    "/wd4996" # Disable warning: 'unsafe' functions like getenv, fopen (use _s variants)
                    "/wd4244" # Disable warning: conversion with possible loss of data (e.g., float to int)
                    "/wd4267" # Disable warning: conversion from 'size_t' to smaller type, possible loss of data
                    "/utf-8" # Set source and execution character sets to UTF-8
                    "/WX" # Treat warnings as errors
        )
    elseif (compiler_id STREQUAL "NVIDIA")
        target_compile_options(
            ${target}
            PRIVATE
                "-Xcompiler=-Wfatal-errors;-Xcompiler=-Wall;-Xcompiler=-Wextra;-Wno-unknown-pragmas;-Wno-cast-function-type;-Wno-unused-function"
        )
    endif ()

    # Set optimization options for different compilers differently
    if (compiler_id MATCHES "MSVC")
        if (CMAKE_BUILD_TYPE STREQUAL "Debug")
            target_compile_options(${target} PRIVATE "/Od;/Zi")
            if (NOT target_type STREQUAL "SHARED_LIBRARY")
                target_compile_options(${target} PRIVATE "/RTC1")
            endif ()
        elseif (CMAKE_BUILD_TYPE STREQUAL "Release" OR CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
            target_compile_options(${target} PRIVATE "/O2;/Zi")
        endif ()
    elseif (
        compiler_id STREQUAL "GNU"
        OR compiler_id STREQUAL "Clang"
        OR compiler_id STREQUAL "AppleClang"
    )
        if (CMAKE_BUILD_TYPE STREQUAL "Debug" OR CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
            target_compile_options(${target} PRIVATE "-O0;-g")
        endif ()
        if (CMAKE_BUILD_TYPE STREQUAL "Release" OR CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
            target_compile_options(${target} PRIVATE "-O2")
        endif ()
    elseif (compiler_id STREQUAL "NVIDIA")
        target_compile_options(
            ${target} PRIVATE "-Xcompiler=-Wall" # All warnings (host)
                              "-Xcompiler=-Wextra" # Extra warnings (host)
        )

        if (CMAKE_BUILD_TYPE STREQUAL "Debug" OR CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
            target_compile_options(
                ${target}
                PRIVATE "-G" # Device debug symbols, which will add `-lineinfo` symbols to PTX
                        "-no-compress" # No compression of debug info
                        "-Xcompiler=-g" # Host debugging symbols explicitly
                        "-Xcompiler=-fno-omit-frame-pointer" # Stack trace clarity
                        "-Xcompiler=-fno-inline" # Prevent host inlining
                        "-maxrregcount=0" # No register count limits
            )
        endif ()
        if (CMAKE_BUILD_TYPE STREQUAL "Release" OR CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
            target_compile_options(
                ${target}
                PRIVATE "-O2" # Disable NVCC optimizations explicitly
                        "-Xptxas=-O2" # Disable PTX assembler optimizations
                        "-Xcompiler=-O2" # Host optimizations off
            )
        endif ()
    endif ()

    # If available, enable Position Independent Code
    get_target_property(target_pic ${target} POSITION_INDEPENDENT_CODE)
    if (target_pic)
        target_compile_definitions(${target} PRIVATE "SZ_PIC")
    endif ()

    # Avoid builtin functions where we know what we are doing.
    if (compiler_id MATCHES "MSVC")
        target_compile_options(${target} PRIVATE "/Oi-")
    else ()
        target_compile_options(${target} PRIVATE "-fno-builtin-memcmp")
        target_compile_options(${target} PRIVATE "-fno-builtin-memchr")
        target_compile_options(${target} PRIVATE "-fno-builtin-memcpy")
        target_compile_options(${target} PRIVATE "-fno-builtin-memset")
    endif ()

    # On macOS, when using non-AppleClang compilers (e.g., Homebrew LLVM), explicitly link against libc++. AppleClang
    # automatically links the system libc++, but Homebrew LLVM requires explicit configuration.
    if (CMAKE_SYSTEM_NAME MATCHES "Darwin"
        AND compiler_id STREQUAL "Clang"
        AND NOT compiler_id STREQUAL "AppleClang"
    )
        if (NOT target_type STREQUAL "SHARED_LIBRARY")
            target_compile_options(${target} PRIVATE "-stdlib=libc++")
            target_link_options(${target} PRIVATE "-stdlib=libc++")
            # Find and link the C++ standard library from the compiler's installation Homebrew LLVM stores libc++ in
            # lib/c++ subdirectory
            get_filename_component(COMPILER_DIR ${CMAKE_CXX_COMPILER} DIRECTORY)
            get_filename_component(COMPILER_ROOT ${COMPILER_DIR} DIRECTORY)
            if (EXISTS "${COMPILER_ROOT}/lib/c++/libc++.dylib")
                target_link_options(${target} PRIVATE "-L${COMPILER_ROOT}/lib/c++")
                target_link_libraries(${target} PRIVATE c++abi)
            elseif (EXISTS "${COMPILER_ROOT}/lib/libc++.dylib")
                target_link_options(${target} PRIVATE "-L${COMPILER_ROOT}/lib")
            endif ()
        endif ()
    endif ()

    # Check for ${target_arch} and set it or use the current system if not defined
    if ("${target_arch}" STREQUAL "")
        # Only use the current system if we are not cross compiling
        if (((NOT MSVC) AND (NOT CMAKE_CROSSCOMPILING)) OR (CMAKE_SYSTEM_PROCESSOR MATCHES
                                                            ${CMAKE_HOST_SYSTEM_PROCESSOR})
        )
            if (compiler_id STREQUAL "NVIDIA")
                # For NVCC, pass native flag to host compiler
                include(CheckCXXCompilerFlag)
                check_cxx_compiler_flag("-march=native" supports_march_native)
                if (supports_march_native)
                    target_compile_options(${target} PRIVATE "-Xcompiler=-march=native")
                endif ()
            elseif (NOT (compiler_id MATCHES "MSVC"))
                include(CheckCXXCompilerFlag)
                check_cxx_compiler_flag("-march=native" supports_march_native)
                if (supports_march_native)
                    target_compile_options(${target} PRIVATE "-march=native")
                endif ()
            else ()
                # MSVC does not have a direct equivalent to -march=native
                if (SZ_IS_64BIT_ARM_)
                    target_compile_options(${target} PRIVATE "/arch:armv8.0")
                else ()
                    target_compile_options(${target} PRIVATE "/arch:AVX2")
                endif ()
            endif ()
        endif ()
    else ()
        if (compiler_id MATCHES "MSVC")
            target_compile_options(${target} PRIVATE "/arch:${target_arch}")
        elseif (compiler_id STREQUAL "NVIDIA")
            # NVCC handles CPU architecture through host compiler flags
            target_compile_options(${target} PRIVATE "-Xcompiler=-march=${target_arch}")
        else ()
            target_compile_options(${target} PRIVATE "-march=${target_arch}")
        endif ()
    endif ()

    # Define SZ_IS_BIG_ENDIAN_ macro based on system byte order
    if (CMAKE_C_BYTE_ORDER STREQUAL "BIG_ENDIAN")
        set(SZ_IS_BIG_ENDIAN_ 1)
    else ()
        set(SZ_IS_BIG_ENDIAN_ 0)
    endif ()

    target_compile_definitions(${target} PRIVATE "SZ_IS_BIG_ENDIAN_=${SZ_IS_BIG_ENDIAN_}")

    # Sanitizer options for Debug mode
    if (CMAKE_BUILD_TYPE STREQUAL "Debug")
        target_compile_definitions(${target} PRIVATE "SZ_DEBUG=1")
        if (STRINGZILLA_USE_SANITIZERS AND NOT target_type STREQUAL "SHARED_LIBRARY")
            if (compiler_id MATCHES "MSVC")
                target_compile_options(${target} PRIVATE "/fsanitize=address;/fsanitize=leak")
                target_link_options(${target} PRIVATE "/fsanitize=address;/fsanitize=leak")
            elseif (compiler_id STREQUAL "NVIDIA")
                # ! NVCC can't handle sanitizers?!
                # https://stackoverflow.com/questions/75590579/cuda-fails-to-initialise-when-address-sanitizer-is-enabled
            else ()
                target_compile_options(${target} PRIVATE "-fsanitize=address" "-fsanitize=undefined")
                target_link_options(${target} PRIVATE "-fsanitize=address" "-fsanitize=undefined")
            endif ()
        endif ()
    else ()
        target_compile_definitions(${target} PRIVATE "SZ_DEBUG=0")
    endif ()

    if (STRINGZILLA_BUILD_COVERAGE AND (compiler_id STREQUAL "Clang" OR compiler_id STREQUAL "AppleClang"))
        target_compile_options(${target} PRIVATE "-fprofile-instr-generate" "-fcoverage-mapping")
        target_link_options(${target} PRIVATE "-fprofile-instr-generate" "-fcoverage-mapping")
    endif ()
endfunction ()

function (define_launcher exec_name source cpp_standard target_arch)
    add_executable(${exec_name})
    target_sources(${exec_name} PRIVATE ${source})
    set_compiler_flags(${exec_name} ${cpp_standard} "${target_arch}" "${CMAKE_CXX_COMPILER_ID}")
    target_link_libraries(${exec_name} PRIVATE stringzilla_header)
    add_test(NAME ${exec_name} COMMAND ${exec_name})
endfunction ()

function (define_gpu_launcher exec_name source cuda_standard target_arch)
    add_executable(${exec_name})
    target_sources(${exec_name} PRIVATE ${source})
    set_source_files_properties(${source} TARGET_DIRECTORY ${exec_name} PROPERTIES LANGUAGE CUDA)
    target_compile_definitions(${exec_name} PRIVATE "SZ_USE_CUDA=1")
    set_target_properties(${exec_name} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    target_include_directories(${exec_name} PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
    set_compiler_flags(${exec_name} ${cuda_standard} "${target_arch}" "${CMAKE_CUDA_COMPILER_ID}")
    target_link_libraries(${exec_name} PRIVATE CUDA::cudart CUDA::cuda_driver)
    # Match the StringZillas CUDA static library's base tier (real SASS for Ampere sm_80 and Hopper sm_90), so the test
    # harness and the precompiled engines it links target the same GPUs.
    set_property(TARGET ${exec_name} PROPERTY CUDA_ARCHITECTURES "80-real;90-real")
    target_link_libraries(${exec_name} PRIVATE stringzilla_header)
    add_test(NAME ${exec_name} COMMAND ${exec_name})
endfunction ()

# Tests and benchmarks are defined after the libraries (below), so the StringZillas consumers can link the precompiled
# static archives instead of recompiling the engines header-only.

# Helper function to set architecture-specific SIMD compile definitions for dynamic dispatch. This is shared between
# stringzilla_shared (serial) and stringzillas_shared (parallel) libraries. This function ONLY sets the SIMD-related
# preprocessor definitions, not compiler flags.
function (set_architecture_simd_definitions target)
    if (SZ_IS_64BIT_X86_)
        target_compile_definitions(
            ${target}
            PRIVATE "SZ_IS_64BIT_X86_=1"
                    "SZ_IS_64BIT_ARM_=0"
                    "SZ_USE_WESTMERE=1"
                    "SZ_USE_GOLDMONT=1"
                    "SZ_USE_HASWELL=1"
                    "SZ_USE_SKYLAKE=1"
                    "SZ_USE_ICELAKE=1"
                    "SZ_USE_NEON=0"
                    "SZ_USE_NEONAES=0"
                    "SZ_USE_NEONSHA=0"
                    "SZ_USE_SVE=0"
                    "SZ_USE_SVE2=0"
                    "SZ_USE_SVE2AES=0"
        )
    elseif (SZ_IS_64BIT_ARM_)
        # SVE is not supported with MSVC
        if (MSVC)
            set(supports_sve "0")
        else ()
            set(supports_sve "1")
        endif ()
        target_compile_definitions(
            ${target}
            PRIVATE "SZ_IS_64BIT_ARM_=1"
                    "SZ_IS_64BIT_X86_=0"
                    "SZ_USE_WESTMERE=0"
                    "SZ_USE_GOLDMONT=0"
                    "SZ_USE_HASWELL=0"
                    "SZ_USE_SKYLAKE=0"
                    "SZ_USE_ICELAKE=0"
                    "SZ_USE_NEON=1"
                    "SZ_USE_NEONAES=1"
                    "SZ_USE_NEONSHA=1"
                    "SZ_USE_SVE=${supports_sve}"
                    "SZ_USE_SVE2=${supports_sve}"
                    "SZ_USE_SVE2AES=${supports_sve}"
        )
    elseif (SZ_IS_64BIT_RISCV_)
        # RISC-V with the RVV 1.0 Vector extension (the toolchain passes `-march=rv64gcv`).
        target_compile_definitions(
            ${target} PRIVATE "SZ_USE_RVV=1" "SZ_USE_LASX=0" "SZ_USE_POWERVSX=0" "SZ_USE_V128=0" "SZ_USE_V128RELAXED=0"
        )
    elseif (SZ_IS_64BIT_LOONGARCH_)
        # LoongArch64 with the LASX 256-bit SIMD extension (the toolchain passes `-mlasx`).
        target_compile_definitions(
            ${target} PRIVATE "SZ_USE_RVV=0" "SZ_USE_LASX=1" "SZ_USE_POWERVSX=0" "SZ_USE_V128=0" "SZ_USE_V128RELAXED=0"
        )
    elseif (SZ_IS_64BIT_POWER_)
        # IBM POWER (little-endian) with VSX (the toolchain passes `-mcpu=power9 -mvsx`).
        target_compile_definitions(
            ${target} PRIVATE "SZ_USE_RVV=0" "SZ_USE_LASX=0" "SZ_USE_POWERVSX=1" "SZ_USE_V128=0" "SZ_USE_V128RELAXED=0"
        )
    elseif (CMAKE_SYSTEM_NAME STREQUAL "WASI" OR CMAKE_SYSTEM_NAME STREQUAL "Emscripten" OR CMAKE_SYSTEM_PROCESSOR
                                                                                            MATCHES "wasm")
        # WebAssembly: the toolchain passes `-msimd128` (+ `-mrelaxed-simd`), and `types.h` auto-enables
        # SZ_USE_V128 / SZ_USE_V128RELAXED from the predefined `__wasm_simd128__` / `__wasm_relaxed_simd__`
        # macros. Leave those to auto-detect; just disable every native tier. wasm has no runtime probe, so
        # the header-only test/bench TUs use compile-time dispatch (SZ_DYNAMIC_DISPATCH defaults to 0).
        target_compile_definitions(
            ${target}
            PRIVATE "SZ_IS_64BIT_X86_=0"
                    "SZ_IS_64BIT_ARM_=0"
                    "SZ_USE_WESTMERE=0"
                    "SZ_USE_GOLDMONT=0"
                    "SZ_USE_HASWELL=0"
                    "SZ_USE_SKYLAKE=0"
                    "SZ_USE_ICELAKE=0"
                    "SZ_USE_NEON=0"
                    "SZ_USE_NEONAES=0"
                    "SZ_USE_NEONSHA=0"
                    "SZ_USE_SVE=0"
                    "SZ_USE_SVE2=0"
                    "SZ_USE_SVE2AES=0"
                    "SZ_USE_RVV=0"
                    "SZ_USE_LASX=0"
                    "SZ_USE_POWERVSX=0"
        )
    else ()
        # Fallback: disable all SIMD and use serial execution only
        message(
            WARNING
                "Neither SZ_IS_64BIT_X86_ nor SZ_IS_64BIT_ARM_ is set for ${target}, falling back to serial execution"
        )
        target_compile_definitions(
            ${target}
            PRIVATE "SZ_IS_64BIT_X86_=0"
                    "SZ_IS_64BIT_ARM_=0"
                    "SZ_USE_WESTMERE=0"
                    "SZ_USE_GOLDMONT=0"
                    "SZ_USE_HASWELL=0"
                    "SZ_USE_SKYLAKE=0"
                    "SZ_USE_ICELAKE=0"
                    "SZ_USE_NEON=0"
                    "SZ_USE_NEONAES=0"
                    "SZ_USE_NEONSHA=0"
                    "SZ_USE_SVE=0"
                    "SZ_USE_SVE2=0"
                    "SZ_USE_SVE2AES=0"
        )
    endif ()
endfunction ()

# Apply the conservative baseline architecture (`-march`/`-mcpu`) that lets one shared/OBJECT compilation host every
# per-ISA SIMD tier: the per-function target attributes inside the kernels pick the actual instruction set, so the
# baseline only has to be old enough for every tier's intrinsics headers to parse. Cross-compiled targets (RISC-V,
# LoongArch, POWER) get their arch from the toolchain file's `CMAKE_<LANG>_FLAGS_INIT`, so they pass an empty arch.
function (set_baseline_architecture_flags target compiler_id)
    if (SZ_IS_64BIT_X86_)
        if (MSVC)
            set_compiler_flags(${target} "" "SSE2" "${compiler_id}")
        else ()
            set_compiler_flags(${target} "" "ivybridge" "${compiler_id}")
        endif ()
    elseif (SZ_IS_64BIT_ARM_)
        if (MSVC)
            set_compiler_flags(${target} "" "armv8.0" "${compiler_id}")
        else ()
            set_compiler_flags(${target} "" "armv8-a" "${compiler_id}")
        endif ()
    else ()
        set_compiler_flags(${target} "" "" "${compiler_id}")
    endif ()
endfunction ()

# Define our libraries, first the header-only version
add_library(stringzilla_header INTERFACE)
add_library(${PROJECT_NAME}::stringzilla_header ALIAS stringzilla_header)
target_include_directories(
    stringzilla_header INTERFACE $<BUILD_INTERFACE:${STRINGZILLA_INCLUDE_BUILD_DIR}> $<INSTALL_INTERFACE:include>
)

# The compiled core is split into one translation unit per domain, so editing one domain only recompiles that domain.
# The thin `c/stringzilla/runtime.c` owns the shared dispatch table & glue.
set(STRINGZILLA_SHIM_SOURCES
    c/stringzilla/runtime.c
    c/stringzilla/compare.c
    c/stringzilla/memory.c
    c/stringzilla/hash.c
    c/stringzilla/find.c
    c/stringzilla/sort.c
    c/stringzilla/intersect.c
    c/stringzilla/utf8_norm.c
    c/stringzilla/utf8_runes.c
    c/stringzilla/utf8_tokens.c
    c/stringzilla/utf8_words.c
    c/stringzilla/utf8_graphemes.c
    c/stringzilla/utf8_sentences.c
    c/stringzilla/utf8_linebreaks.c
    c/stringzilla/utf8_uncased_fold.c
    c/stringzilla/utf8_uncased.c
)

# Helper function used for `stringzilla_shared` and `stringzilla_bare` targets
function (define_stringzilla_shared target)
    add_library(${target} SHARED ${STRINGZILLA_SHIM_SOURCES})
    add_library(${PROJECT_NAME}::${target} ALIAS ${target})

    set_target_properties(
        ${target}
        PROPERTIES VERSION ${PROJECT_VERSION}
                   SOVERSION 1
                   POSITION_INDEPENDENT_CODE ON
    )

    # Set the baseline architecture; the dynamic-dispatch core selects each SIMD tier at runtime.
    set_baseline_architecture_flags(${target} "${CMAKE_CXX_COMPILER_ID}")

    # Set architecture-specific SIMD flags using shared helper function
    set_architecture_simd_definitions(${target})

endfunction ()

if (STRINGZILLA_BUILD_SHARED)

    define_stringzilla_shared(stringzilla_shared)
    target_compile_definitions(stringzilla_shared PRIVATE "SZ_AVOID_LIBC=0")
    target_compile_definitions(stringzilla_shared PRIVATE "SZ_OVERRIDE_LIBC=1")
    target_include_directories(stringzilla_shared PUBLIC include)

    # Force aggressive inlining for SHA-256 functions to avoid stack overflow in Go CGO (-mno-red-zone) The SHA-256
    # block processor has ~200-350 instructions and must be inlined to avoid nested stack frames
    if (CMAKE_C_COMPILER_ID STREQUAL "GNU")
        target_compile_options(
            stringzilla_shared PRIVATE "--param=max-inline-insns-single=2000" "--param=max-inline-insns-auto=2000"
                                       "-finline-functions" "-fno-inline-functions-called-once"
        )
    elseif (CMAKE_C_COMPILER_ID MATCHES "Clang")
        target_compile_options(stringzilla_shared PRIVATE "-mllvm" "-inline-threshold=2000" "-finline-functions")
    endif ()

    # Try compiling a version without linking the LibC ! This is only for Linux/MSVC, as on modern Arm-based MacOS
    # machines ! We can't legally access Arm's "feature registers" without `sysctl` or `sysctlbyname`.
    if (NOT CMAKE_SYSTEM_NAME MATCHES "Darwin")
        define_stringzilla_shared(stringzilla_bare)
        target_compile_definitions(stringzilla_bare PRIVATE "SZ_AVOID_LIBC=1")
        target_compile_definitions(stringzilla_bare PRIVATE "SZ_OVERRIDE_LIBC=1")
        target_include_directories(stringzilla_bare PUBLIC include)

        # For stringzilla_bare, enforce strict C99 standard conformance with `-pedantic`. This ensures the bare build
        # remains strictly standards-compliant without POSIX extensions. Explore `SZ_HAS_POSIX_EXTENSIONS_` macro usage
        # for details.
        if (CMAKE_C_COMPILER_ID STREQUAL "GNU" OR CMAKE_C_COMPILER_ID MATCHES "Clang")
            target_compile_options(stringzilla_bare PRIVATE "-pedantic")
        endif ()

        # Avoid built-ins
        target_compile_options(stringzilla_bare PRIVATE "$<$<CXX_COMPILER_ID:GNU,Clang>:-fno-builtin;-nostdlib>")
        target_compile_options(stringzilla_bare PRIVATE "$<$<CXX_COMPILER_ID:MSVC>:/Oi-;/GS->")
        target_link_options(stringzilla_bare PRIVATE "$<$<CXX_COMPILER_ID:GNU,Clang>:-nostdlib>")
        target_link_options(stringzilla_bare PRIVATE "$<$<CXX_COMPILER_ID:MSVC>:/NODEFAULTLIB>")
    endif ()
endif ()

# Helper function used for `stringzillas_cpus_shared`, `stringzillas_cuda_shared`, and `stringzillas_rocm_shared`
# targets
function (define_stringzillas_shared target source_file backend_flags)
    # Compile the translation units once into an OBJECT library, then expose a SHARED and a STATIC variant that reuse
    # those objects. Tests and benchmarks link the STATIC variant and inherit `SZ_DYNAMIC_DISPATCH=1` (set INTERFACE
    # below), so the heavy precompiled engine instantiations are linked rather than recompiled. The objects are linked
    # PRIVATE into each variant (baked into the `.so`/`.a`) to avoid the OBJECT-library transitive duplicate-symbol
    # gotcha; consumer-facing usage requirements are declared explicitly on the variants' INTERFACE.
    string(REPLACE "_shared" "" base "${target}")
    set(objects "${base}_objects")
    set(static "${base}_static")

    add_library(${objects} OBJECT ${source_file})
    add_library(${target} SHARED)
    add_library(${static} STATIC)
    add_library(${PROJECT_NAME}::${target} ALIAS ${target})
    target_link_libraries(${target} PRIVATE ${objects})
    target_link_libraries(${static} PRIVATE ${objects})

    set_target_properties(${objects} ${target} ${static} PROPERTIES POSITION_INDEPENDENT_CODE ON)
    set_target_properties(${objects} PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON)
    set_target_properties(${target} PROPERTIES VERSION ${PROJECT_VERSION} SOVERSION 1)

    # Compile settings for the library's own translation units (the OBJECT compilation).
    target_include_directories(${objects} PRIVATE include fork_union/include)
    target_compile_definitions(${objects} PRIVATE "SZ_DYNAMIC_DISPATCH=1")
    target_compile_definitions(${objects} PRIVATE "SZ_AVOID_LIBC=0")
    # The precompiled per-capability providers pull in fork_union directly for `fu::basic_pool_t`; keep NUMA off so
    # libnuma-free cross-builds (e.g. aarch64) compile (matches the `stringzillas.cuh` default).
    target_compile_definitions(${objects} PRIVATE "FU_ENABLE_NUMA=0")
    if (NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
        target_compile_definitions(${objects} PRIVATE "SZ_DEBUG=0")
    endif ()
    foreach (flag ${backend_flags})
        target_compile_definitions(${objects} PRIVATE ${flag})
    endforeach ()
    target_compile_options(
        ${objects} PRIVATE "$<$<CXX_COMPILER_ID:GNU,Clang>:-O2;-fPIC>" "$<$<CXX_COMPILER_ID:MSVC>:/O2>"
    )
    set_architecture_simd_definitions(${objects})

    # Usage requirements seen by consumers (tests and benchmarks linking the library). Defining `SZ_DYNAMIC_DISPATCH=1`
    # makes a consumer extern, rather than re-instantiate, the engines, and fork_union is needed by the extern block in
    # `similarities.hpp`. The consumer still picks its own ISA via `set_compiler_flags`.
    foreach (variant ${target} ${static})
        target_include_directories(${variant} PUBLIC include)
        target_include_directories(${variant} INTERFACE fork_union/include)
        target_compile_definitions(${variant} INTERFACE "SZ_DYNAMIC_DISPATCH=1")
    endforeach ()

    find_package(Threads REQUIRED)
    target_link_libraries(${target} PUBLIC Threads::Threads)
    target_link_libraries(${static} PUBLIC Threads::Threads)
    if (WIN32 AND CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
        target_link_libraries(${target} PRIVATE msvcrt.lib vcruntime.lib ucrt.lib)
        target_link_libraries(${static} PRIVATE msvcrt.lib vcruntime.lib ucrt.lib)
    endif ()
endfunction ()

# Compiles a group of StringZillas CUDA translation units into an OBJECT library for one set of GPU architectures. Each
# tier gets its own architecture floor (base SIMT and Kepler from sm_70, Hopper DPX from sm_90), and the static library
# (local testing) and shared library (distribution) request different architecture sets, so this helper is invoked once
# per (tier, variant) combination.
function (define_stringzillas_cuda_objects name sources gpu_archs)
    add_library(${name} OBJECT ${sources})
    set_source_files_properties(${sources} TARGET_DIRECTORY ${name} PROPERTIES LANGUAGE CUDA)
    set_target_properties(
        ${name}
        PROPERTIES POSITION_INDEPENDENT_CODE ON
                   CXX_STANDARD 20
                   CXX_STANDARD_REQUIRED ON
                   CUDA_STANDARD 20
                   CUDA_STANDARD_REQUIRED ON
                   CUDA_SEPARABLE_COMPILATION OFF
                   CUDA_ARCHITECTURES "${gpu_archs}"
    )
    target_include_directories(${name} PRIVATE include fork_union/include)
    target_compile_definitions(
        ${name} PRIVATE "SZ_DYNAMIC_DISPATCH=1" "SZ_USE_CUDA=1" "SZ_USE_ROCM=0" "SZ_AVOID_LIBC=0" "FU_ENABLE_NUMA=0"
    )
    if (NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
        target_compile_definitions(${name} PRIVATE "SZ_DEBUG=0")
    endif ()
    target_compile_options(${name} PRIVATE "--expt-relaxed-constexpr")
    set_architecture_simd_definitions(${name})
    set_baseline_architecture_flags(${name} "${CMAKE_CUDA_COMPILER_ID}")
endfunction ()

# The parallel shim is split into one translation unit per algorithm, so each engine (and its CUDA kernels) compiles
# independently and in parallel. Each `.cpp`/`.cu` is a thin wrapper around the matching `c/szs_<algorithm>.cuh`, which
# includes the shared `c/stringzillas/stringzillas.cuh` scaffolding.
set(STRINGZILLAS_CPU_SOURCES
    c/stringzillas/runtime.cpp
    c/stringzillas/levenshtein.cpp
    c/stringzillas/needleman_wunsch.cpp
    c/stringzillas/smith_waterman.cpp
    c/stringzillas/fingerprints.cpp
    # Per-capability instantiation units: each emits one ISA's heavy single-pair anti-diagonal SIMD core so the
    # algorithm entry TUs above only declare them (`extern template`) and compile in parallel. Each file is internally
    # guarded by its `SZ_USE_*` macro, so the off-platform ones compile to empty objects.
    c/stringzillas/levenshtein_serial.cpp
    c/stringzillas/levenshtein_icelake.cpp
    c/stringzillas/levenshtein_neon.cpp
    c/stringzillas/needleman_wunsch_serial.cpp
    c/stringzillas/needleman_wunsch_icelake.cpp
    c/stringzillas/needleman_wunsch_haswell.cpp
    c/stringzillas/needleman_wunsch_neon.cpp
    c/stringzillas/smith_waterman_serial.cpp
    c/stringzillas/smith_waterman_icelake.cpp
    c/stringzillas/smith_waterman_haswell.cpp
    c/stringzillas/smith_waterman_neon.cpp
)
# CUDA translation units are grouped by their architecture floor. The C-API entry units, the base SIMT providers and the
# Kepler video-SIMD providers run from sm_70, so they live in the "base" group. The Hopper DPX providers
# (`__viaddmax_s16x2`, `__vimax3`, `__reduce_max_sync`) need sm_90, so they form a separate group with its own floor.
set(STRINGZILLAS_CUDA_BASE_SOURCES
    c/stringzillas/runtime.cu
    c/stringzillas/levenshtein.cu
    c/stringzillas/needleman_wunsch.cu
    c/stringzillas/smith_waterman.cu
    c/stringzillas/fingerprints.cu
    c/stringzillas/levenshtein_cuda.cu
    c/stringzillas/levenshtein_kepler.cu
    c/stringzillas/needleman_wunsch_cuda.cu
    c/stringzillas/smith_waterman_cuda.cu
)
set(STRINGZILLAS_CUDA_HOPPER_SOURCES c/stringzillas/levenshtein_hopper.cu c/stringzillas/needleman_wunsch_hopper.cu
                                     c/stringzillas/smith_waterman_hopper.cu
)

# Assemble one StringZillas CUDA library variant from per-tier OBJECT libraries. The base tier (C-API entry units, base
# SIMT and Kepler providers) and the Hopper DPX tier each compile once for the requested architecture set, then link
# into the named variant. The static and shared variants pass different architecture sets (real SASS only versus real
# SASS plus forward-compatible PTX), so this is invoked once per variant.
function (define_stringzillas_cuda_library target kind base_archs hopper_archs)
    define_stringzillas_cuda_objects(${target}_base "${STRINGZILLAS_CUDA_BASE_SOURCES}" "${base_archs}")
    define_stringzillas_cuda_objects(${target}_hopper "${STRINGZILLAS_CUDA_HOPPER_SOURCES}" "${hopper_archs}")

    add_library(${target} ${kind})
    target_link_libraries(${target} PRIVATE ${target}_base ${target}_hopper)
    set_target_properties(${target} PROPERTIES POSITION_INDEPENDENT_CODE ON LINKER_LANGUAGE CUDA)
    if (kind STREQUAL "SHARED")
        add_library(${PROJECT_NAME}::${target} ALIAS ${target})
        set_target_properties(${target} PROPERTIES VERSION ${PROJECT_VERSION} SOVERSION 1)
    endif ()

    # Consumers (tests/benches) extern the engines and link cudart themselves; the static archive does not resolve
    # cudart, so expose it (plus the dynamic-dispatch macro and headers) on the variant's interface.
    target_include_directories(${target} PUBLIC include)
    target_include_directories(${target} INTERFACE fork_union/include)
    target_compile_definitions(${target} INTERFACE "SZ_DYNAMIC_DISPATCH=1")
    target_link_libraries(${target} PUBLIC CUDA::cudart Threads::Threads)
endfunction ()

if (STRINGZILLAS_BUILD_SHARED)

    # Define StringZillas CPU shared + static libraries (sharing one OBJECT compilation)
    define_stringzillas_shared(stringzillas_cpus_shared "${STRINGZILLAS_CPU_SOURCES}" "SZ_USE_CUDA=0;SZ_USE_ROCM=0")

    # Set the baseline architecture on the OBJECT compilation; the per-ISA SIMD cores carry per-function target
    # attributes, so this conservative baseline covers all tiers.
    set_baseline_architecture_flags(stringzillas_cpus_objects "${CMAKE_CXX_COMPILER_ID}")
endif ()

# Define StringZillas CUDA shared + static libraries (only if CUDA is available). The static library is for local
# testing and benchmarking, so it ships real SASS for the current and older compatible datacenter GPUs (Ampere sm_80 and
# Hopper sm_90) and runs without any load-time JIT. The shared library is for distribution, so on top of that real SASS
# it also packages forward-compatible PTX (the `-virtual` entries) that the driver JITs on newer GPUs. The Hopper DPX
# tier exists only from sm_90, so it never carries an sm_80 entry.
if (STRINGZILLAS_BUILD_SHARED AND STRINGZILLA_BUILD_CUDA)
    find_package(Threads REQUIRED)
    define_stringzillas_cuda_library(stringzillas_cuda_static STATIC "80-real;90-real" "90-real")
    define_stringzillas_cuda_library(stringzillas_cuda_shared SHARED "80-real;90-real;90-virtual" "90-real;90-virtual")
endif ()

# Tests and benchmarks come after the libraries so the StringZillas consumers can link the precompiled static archives
# (which carry the engine instantiations and export SZ_DYNAMIC_DISPATCH=1 on their interface) instead of recompiling the
# engines header-only.
if (STRINGZILLA_BUILD_BENCHMARK)
    define_launcher(stringzilla_bench_find_cpp20 bench/find.cpp 20 "${STRINGZILLA_TARGET_ARCH}")
    define_launcher(stringzilla_bench_sequence_cpp20 bench/sequence.cpp 20 "${STRINGZILLA_TARGET_ARCH}")
    define_launcher(stringzilla_bench_token_cpp20 bench/token.cpp 20 "${STRINGZILLA_TARGET_ARCH}")
    define_launcher(stringzilla_bench_utf8_uncased_cpp20 bench/utf8_uncased.cpp 20 "${STRINGZILLA_TARGET_ARCH}")
    define_launcher(stringzilla_bench_utf8_iterate_cpp20 bench/utf8_iterate.cpp 20 "${STRINGZILLA_TARGET_ARCH}")
    define_launcher(stringzilla_bench_utf8_norm_cpp20 bench/utf8_norm.cpp 20 "${STRINGZILLA_TARGET_ARCH}")
    define_launcher(stringzilla_bench_utf8_delimiters_cpp20 bench/utf8_delimiters.cpp 20 "${STRINGZILLA_TARGET_ARCH}")
    define_launcher(stringzilla_bench_container_cpp20 bench/container.cpp 20 "${STRINGZILLA_TARGET_ARCH}")
    define_launcher(stringzilla_bench_memory_cpp20 bench/memory.cpp 20 "${STRINGZILLA_TARGET_ARCH}")

    # Parallel benchmarks link the precompiled static libraries when those are built.
    define_launcher(
        stringzillas_bench_similarities_cpp20 bench/similarities.cpp 20 "${STRINGZILLA_TARGET_ARCH}"
    )
    define_launcher(
        stringzillas_bench_fingerprints_cpp20 bench/fingerprints.cpp 20 "${STRINGZILLA_TARGET_ARCH}"
    )
    if (STRINGZILLAS_BUILD_SHARED)
        target_link_libraries(stringzillas_bench_similarities_cpp20 PRIVATE stringzillas_cpus_static)
        target_link_libraries(stringzillas_bench_fingerprints_cpp20 PRIVATE stringzillas_cpus_static)
    endif ()
    if (STRINGZILLA_BUILD_CUDA)
        define_gpu_launcher(
            stringzillas_bench_similarities_cu20 bench/similarities.cu 20 "${STRINGZILLA_TARGET_ARCH}"
        )
        define_gpu_launcher(
            stringzillas_bench_fingerprints_cu20 bench/fingerprints.cu 20 "${STRINGZILLA_TARGET_ARCH}"
        )
        if (STRINGZILLAS_BUILD_SHARED)
            target_link_libraries(stringzillas_bench_similarities_cu20 PRIVATE stringzillas_cuda_static)
            target_link_libraries(stringzillas_bench_fingerprints_cu20 PRIVATE stringzillas_cuda_static)
        endif ()
        # The host object pulls in `std::log` etc.; unlike the host-compiler driver used for the `cpp20`
        # benches, nvcc's final link does not implicitly add libm. Link it only where it exists as a
        # separate library (Linux/glibc); on macOS/MSVC the math symbols live in the C runtime.
        find_library(STRINGZILLA_LIBM m)
        if (STRINGZILLA_LIBM)
            target_link_libraries(stringzillas_bench_similarities_cu20 PRIVATE ${STRINGZILLA_LIBM})
            target_link_libraries(stringzillas_bench_fingerprints_cu20 PRIVATE ${STRINGZILLA_LIBM})
        endif ()
    endif ()
endif ()

if (STRINGZILLA_BUILD_TEST)
    # The serial test suite is split across per-domain translation units that compile into the same binary,
    # so the standards-matrix builds below parallelize and rebuild incrementally instead of recompiling one
    # 5K-line file. `define_launcher`'s `target_sources` accepts this list verbatim.
    set(STRINGZILLA_TEST_SOURCES
        test/stringzilla.cpp
        test/hash.cpp
        test/utf8_runes.cpp
        test/utf8_tokens.cpp
        test/utf8_words.cpp
        test/utf8_graphemes.cpp
        test/utf8_sentences.cpp
        test/utf8_linebreaks.cpp
        test/utf8_norm.cpp
        test/utf8_delimiters.cpp
        test/uncased.cpp
        test/string.cpp
        test/find.cpp
        test/sort.cpp
    )

    # Make sure compilation passes for different C++ standards; MSVC only supports C++11 and newer.
    define_launcher(stringzilla_test_cpp11 "${STRINGZILLA_TEST_SOURCES}" 11 "${STRINGZILLA_TARGET_ARCH}")
    define_launcher(stringzilla_test_cpp14 "${STRINGZILLA_TEST_SOURCES}" 14 "${STRINGZILLA_TARGET_ARCH}")
    define_launcher(stringzilla_test_cpp17 "${STRINGZILLA_TEST_SOURCES}" 17 "${STRINGZILLA_TARGET_ARCH}")
    define_launcher(stringzilla_test_cpp20 "${STRINGZILLA_TEST_SOURCES}" 20 "${STRINGZILLA_TARGET_ARCH}")

    # Test parallel algorithms separately; link the static archive when the libraries are built.
    define_launcher(stringzillas_test_cpp17 test/stringzillas.cpp 17 "${STRINGZILLA_TARGET_ARCH}")
    define_launcher(stringzillas_test_cpp20 test/stringzillas.cpp 20 "${STRINGZILLA_TARGET_ARCH}")
    if (STRINGZILLAS_BUILD_SHARED)
        target_link_libraries(stringzillas_test_cpp17 PRIVATE stringzillas_cpus_static)
        target_link_libraries(stringzillas_test_cpp20 PRIVATE stringzillas_cpus_static)
    endif ()

    # We only target C++14 and newer so `constexpr` functions compile on both host and device without `__device__`
    # annotations, and C++17 and newer so the `if constexpr` compile-time SIMD dispatch works.
    if (STRINGZILLA_BUILD_CUDA)
        define_gpu_launcher(stringzillas_test_cu17 test/stringzillas.cu 17 "${STRINGZILLA_TARGET_ARCH}")
        define_gpu_launcher(stringzillas_test_cu20 test/stringzillas.cu 20 "${STRINGZILLA_TARGET_ARCH}")
        if (STRINGZILLAS_BUILD_SHARED)
            target_link_libraries(stringzillas_test_cu17 PRIVATE stringzillas_cuda_static)
            target_link_libraries(stringzillas_test_cu20 PRIVATE stringzillas_cuda_static)
        endif ()
    endif ()
endif ()

# TODO: Define StringZillas ROCm shared library when ROCm support is added if (ENABLE_ROCM)
# define_stringzillas_shared(stringzillas_rocm_shared "SZ_USE_CUDA=0;SZ_USE_ROCM=1") endif ()

if (STRINGZILLA_INSTALL)
    if (TARGET stringzilla_header)
        install(
            TARGETS stringzilla_shared
            ARCHIVE
            BUNDLE
            FRAMEWORK
            LIBRARY
            OBJECTS
            PRIVATE_HEADER
            PUBLIC_HEADER
            RESOURCE
            RUNTIME
        )
    endif ()
    if (TARGET stringzilla_bare)
        install(
            TARGETS stringzilla_bare
            ARCHIVE
            BUNDLE
            FRAMEWORK
            LIBRARY
            OBJECTS
            PRIVATE_HEADER
            PUBLIC_HEADER
            RESOURCE
            RUNTIME
        )
    endif ()

    # Install StringZillas shared libraries if they were built
    if (TARGET stringzillas_cpus_shared)
        install(
            TARGETS stringzillas_cpus_shared
            ARCHIVE
            BUNDLE
            FRAMEWORK
            LIBRARY
            OBJECTS
            PRIVATE_HEADER
            PUBLIC_HEADER
            RESOURCE
            RUNTIME
        )
    endif ()

    if (TARGET stringzillas_cuda_shared)
        install(
            TARGETS stringzillas_cuda_shared
            ARCHIVE
            BUNDLE
            FRAMEWORK
            LIBRARY
            OBJECTS
            PRIVATE_HEADER
            PUBLIC_HEADER
            RESOURCE
            RUNTIME
        )
    endif ()

    install(DIRECTORY ${STRINGZILLA_INCLUDE_BUILD_DIR} DESTINATION ${STRINGZILLA_INCLUDE_INSTALL_DIR})
    install(DIRECTORY c/ DESTINATION ${CMAKE_INSTALL_DATADIR}/${PROJECT_NAME}/src)
endif ()
