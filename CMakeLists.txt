# StringZilla CMakeLists.txt
#
# This file defines several library build & installation targets:
#
# * stringzilla_header: A header-only library with the StringZilla C and C++ headers.
# * stringzilla_shared: A shared library with the StringZilla C and C++ headers and dynamic SIMD dispatch.
# * stringzilla_bare: A shared library with the StringZilla headers, but without linking the standard C library.
# * stringzillas_cpus_shared: A shared library with the StringZillas parallel algorithms for multi-threaded CPUs.
# * stringzillas_cuda_shared: A shared library with the StringZillas parallel algorithms for CUDA-capable GPUs.
# * stringzillas_rocm_shared: A shared library with the StringZillas parallel algorithms for ROCm-capable GPUs.
#
# Tests for different C++ standards:
#
# * stringzilla_test_cpp11: C++11 baseline support.
# * stringzilla_test_cpp14: C++14 support with `std::less<std::string>`-like function objects.
# * stringzilla_test_cpp17: C++17 support with `std::string_view` compatibility.
# * stringzilla_test_cpp20: C++20 support with `<=>` operator and more `constexpr` features.
#
# Tests for different SIMD architectures:
#
# * stringzilla_test_cpp20_serial: A test executable for serial execution.
# * stringzilla_test_cpp20_westmere: A test executable for SSE4.2.
# * stringzilla_test_cpp20_goldmont: A test executable for SHA-NI.
# * stringzilla_test_cpp20_haswell: A test executable for AVX2.
# * stringzilla_test_cpp20_icelake: A test executable for AVX-512.
# * stringzilla_test_cpp20_neon: A test executable for ARM Neon.
# * stringzilla_test_cpp20_sve: A test executable for ARM Scalable Vector Extension.
#
# Serial Benchmarks:
#
# * stringzilla_bench_find_cpp20: A benchmark for substring search operations.
# * stringzilla_bench_sequence_cpp20: A benchmark for string array-level operations.
# * stringzilla_bench_token_cpp20: A benchmark for comparators and hash functions.
# * stringzilla_bench_container_cpp20: A benchmark for STL containers powered by StringZilla.
# * stringzilla_bench_memory_cpp20: A benchmark for LibC-style low-level memory operations.
# * stringzilla_bench_utf8_case_cpp20: A benchmark for the `sz_utf8_case_*` folding/search family.
# * stringzilla_bench_utf8_iterate_cpp20: A benchmark for the `sz_utf8_*` iteration/segmentation family.
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

    target_include_directories(${target} PRIVATE scripts)
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
    # Only targeting Ampere and Hopper architectures for now
    set_property(TARGET ${exec_name} PROPERTY CUDA_ARCHITECTURES 80 90)
    target_link_libraries(${exec_name} PRIVATE stringzilla_header)
    add_test(NAME ${exec_name} COMMAND ${exec_name})
endfunction ()

if (STRINGZILLA_BUILD_BENCHMARK)
    define_launcher(stringzilla_bench_find_cpp20 scripts/bench_find.cpp 20 "${STRINGZILLA_TARGET_ARCH}")
    define_launcher(stringzilla_bench_sequence_cpp20 scripts/bench_sequence.cpp 20 "${STRINGZILLA_TARGET_ARCH}")
    define_launcher(stringzilla_bench_token_cpp20 scripts/bench_token.cpp 20 "${STRINGZILLA_TARGET_ARCH}")
    define_launcher(stringzilla_bench_utf8_case_cpp20 scripts/bench_utf8_case.cpp 20 "${STRINGZILLA_TARGET_ARCH}")
    define_launcher(stringzilla_bench_utf8_iterate_cpp20 scripts/bench_utf8_iterate.cpp 20 "${STRINGZILLA_TARGET_ARCH}")
    define_launcher(stringzilla_bench_container_cpp20 scripts/bench_container.cpp 20 "${STRINGZILLA_TARGET_ARCH}")
    define_launcher(stringzilla_bench_memory_cpp20 scripts/bench_memory.cpp 20 "${STRINGZILLA_TARGET_ARCH}")

    # The parallel StringZillas benchmarks were template-based and were removed with the C++ engine in
    # the v5 lowering; reintroduce as plain-C benchmarks over the `szs_*` API when needed.
endif ()

if (STRINGZILLA_BUILD_TEST)
    # Make sure that the compilation passes for different C++ standards!
    #
    # Keep in mind, MSVC only supports C++11 and newer.
    define_launcher(stringzilla_test_cpp11 scripts/test_stringzilla.cpp 11 "${STRINGZILLA_TARGET_ARCH}")
    define_launcher(stringzilla_test_cpp14 scripts/test_stringzilla.cpp 14 "${STRINGZILLA_TARGET_ARCH}")
    define_launcher(stringzilla_test_cpp17 scripts/test_stringzilla.cpp 17 "${STRINGZILLA_TARGET_ARCH}")
    define_launcher(stringzilla_test_cpp20 scripts/test_stringzilla.cpp 20 "${STRINGZILLA_TARGET_ARCH}")

    # Check system architecture to avoid complex cross-compilation workflows, but compile multiple backends: disabling
    # all SIMD, enabling only AVX2, only AVX-512, only Arm Neon.
    if (SZ_IS_64BIT_X86_)
        # x86 specific backends
        if (MSVC)
            define_launcher(stringzilla_test_cpp20_serial scripts/test_stringzilla.cpp 20 "AVX")
            define_launcher(stringzilla_test_cpp20_westmere scripts/test_stringzilla.cpp 20 "SSE4.2")
            define_launcher(stringzilla_test_cpp20_haswell scripts/test_stringzilla.cpp 20 "AVX2")
            define_launcher(stringzilla_test_cpp20_icelake scripts/test_stringzilla.cpp 20 "AVX512")
        else ()
            define_launcher(stringzilla_test_cpp20_serial scripts/test_stringzilla.cpp 20 "ivybridge")
            define_launcher(stringzilla_test_cpp20_westmere scripts/test_stringzilla.cpp 20 "westmere")
            define_launcher(stringzilla_test_cpp20_goldmont scripts/test_stringzilla.cpp 20 "goldmont")
            define_launcher(stringzilla_test_cpp20_haswell scripts/test_stringzilla.cpp 20 "haswell")
            define_launcher(stringzilla_test_cpp20_icelake scripts/test_stringzilla.cpp 20 "sapphirerapids")
        endif ()
    elseif (SZ_IS_64BIT_ARM_)
        # ARM specific backends
        if (MSVC)
            # SVE is not supported with MSVC
            define_launcher(stringzilla_test_cpp20_serial scripts/test_stringzilla.cpp 20 "armv8.0")
            define_launcher(stringzilla_test_cpp20_neon scripts/test_stringzilla.cpp 20 "armv8.0")
        else ()
            define_launcher(stringzilla_test_cpp20_serial scripts/test_stringzilla.cpp 20 "armv8-a")
            define_launcher(stringzilla_test_cpp20_neon scripts/test_stringzilla.cpp 20 "armv8-a+simd")
            # SVE is not supported on Apple Silicon, only compile on non-Darwin ARM platforms
            if (NOT CMAKE_SYSTEM_NAME MATCHES "Darwin")
                define_launcher(stringzilla_test_cpp20_sve scripts/test_stringzilla.cpp 20 "armv8.2-a+sve")
            endif ()
        endif ()
    endif ()
endif ()

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
    c/stringzilla/utf8_iterate.c
    c/stringzilla/utf8_case_fold.c
    c/stringzilla/utf8_case_insensitive.c
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

    # Set compiler flags for appropriate architecture
    if (SZ_IS_64BIT_X86_)
        if (MSVC)
            set_compiler_flags(${target} "" "SSE2" "${CMAKE_CXX_COMPILER_ID}")
        else ()
            set_compiler_flags(${target} "" "ivybridge" "${CMAKE_CXX_COMPILER_ID}")
        endif ()
    elseif (SZ_IS_64BIT_ARM_)
        if (MSVC)
            set_compiler_flags(${target} "" "armv8.0" "${CMAKE_CXX_COMPILER_ID}")
        else ()
            set_compiler_flags(${target} "" "armv8-a" "${CMAKE_CXX_COMPILER_ID}")
        endif ()
    else ()
        # Cross-compiled targets (RISC-V, LoongArch, POWER, ...) get their `-march`/`-mcpu` from the CMake toolchain
        # file's `CMAKE_<LANG>_FLAGS_INIT`, so pass an empty arch.
        set_compiler_flags(${target} "" "" "${CMAKE_CXX_COMPILER_ID}")
    endif ()

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
    add_library(${target} SHARED ${source_file})
    add_library(${PROJECT_NAME}::${target} ALIAS ${target})

    set_target_properties(
        ${target}
        PROPERTIES VERSION ${PROJECT_VERSION}
                   SOVERSION 1
                   POSITION_INDEPENDENT_CODE ON
    )

    target_include_directories(${target} PUBLIC include)
    target_include_directories(${target} PRIVATE fork_union/include)
    target_compile_definitions(${target} PRIVATE "SZ_DYNAMIC_DISPATCH=1")
    target_compile_definitions(${target} PRIVATE "SZ_AVOID_LIBC=0")
    # Only define SZ_DEBUG=0 in Release builds; Debug builds inherit from types.h
    if (NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
        target_compile_definitions(${target} PRIVATE "SZ_DEBUG=0")
    endif ()

    # Set backend-specific compilation flags
    foreach (flag ${backend_flags})
        target_compile_definitions(${target} PRIVATE ${flag})
    endforeach ()

    # Use C++20 for StringZillas
    set_target_properties(${target} PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON)

    # Architecture-specific optimizations
    target_compile_options(
        ${target} PRIVATE "$<$<CXX_COMPILER_ID:GNU,Clang>:-O3;-fPIC>" "$<$<CXX_COMPILER_ID:MSVC>:/O2>"
    )

    # Hide non-public symbols so the fork_union C-ABI TU (fork_union/c/lib.cpp) and other internals do
    # not leak into the exported ABI; only `SZ_DYNAMIC` (visibility=default) `szs_*` symbols are exported.
    target_compile_options(
        ${target} PRIVATE "$<$<COMPILE_LANGUAGE:C,CXX>:$<$<CXX_COMPILER_ID:GNU,Clang>:-fvisibility=hidden>>"
    )

    # Set architecture-specific SIMD flags using shared helper function
    set_architecture_simd_definitions(${target})

    # Link threading libraries for CPU backend
    find_package(Threads REQUIRED)
    target_link_libraries(${target} PRIVATE Threads::Threads)

    # fork_union's C ABI (fork_union/c/lib.cpp) is NUMA-aware; link libnuma when available, else disable.
    if (UNIX AND NOT APPLE)
        find_library(NUMA_LIBRARY NAMES numa)
        if (NUMA_LIBRARY)
            target_compile_definitions(${target} PRIVATE "FU_ENABLE_NUMA=1")
            target_link_libraries(${target} PRIVATE ${NUMA_LIBRARY})
        else ()
            target_compile_definitions(${target} PRIVATE "FU_ENABLE_NUMA=0")
        endif ()
    endif ()

    # Platform-specific runtime libraries (similar to define_stringzilla_shared)
    if (WIN32 AND CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
        target_link_libraries(${target} PRIVATE msvcrt.lib vcruntime.lib ucrt.lib)
    endif ()

endfunction ()

# The parallel shim is split into one translation unit per algorithm, so each engine (and its CUDA kernels) compiles
# independently and in parallel. Each `.cpp`/`.cu` is a thin wrapper around the matching `c/szs_<algorithm>.cuh`, which
# includes the shared `c/stringzillas/stringzillas.cuh` scaffolding.
# v5 lowering: the CPU library is now plain C (runtime + per-domain dispatch TUs), mirroring the C core.
# `fork_union/c/lib.cpp` is the only C++ TU; it provides the `fu_*` C ABI the dispatch entries call.
set(STRINGZILLAS_CPU_SOURCES
    c/stringzillas/runtime.c c/stringzillas/similarities.c c/stringzillas/fingerprints.c fork_union/c/lib.cpp
)
# v5 lowering: the CUDA library is the SAME lowered C TUs as the CPU lib, compiled as host C++ so they can
# pull in the Driver-API launchers (`cuda_host.cuh`). GPU device code does NOT live in the lib: it ships as
# one PTX module per tier (built standalone with `nvcc -ptx`) and is `bin2c`-embedded + JIT-loaded at runtime.
# The concrete source list is assembled inside the CUDA block below (after the PTX blobs are generated).

if (STRINGZILLAS_BUILD_SHARED)

    # Define StringZillas CPU shared library
    define_stringzillas_shared(stringzillas_cpus_shared "${STRINGZILLAS_CPU_SOURCES}" "SZ_USE_CUDA=0;SZ_USE_ROCM=0")

    # Set compiler flags for C++ target
    if (SZ_IS_64BIT_X86_)
        if (MSVC)
            set_compiler_flags(stringzillas_cpus_shared "" "SSE2" "${CMAKE_CXX_COMPILER_ID}")
        else ()
            set_compiler_flags(stringzillas_cpus_shared "" "ivybridge" "${CMAKE_CXX_COMPILER_ID}")
        endif ()
    elseif (SZ_IS_64BIT_ARM_)
        if (MSVC)
            set_compiler_flags(stringzillas_cpus_shared "" "armv8.0" "${CMAKE_CXX_COMPILER_ID}")
        else ()
            set_compiler_flags(stringzillas_cpus_shared "" "armv8-a" "${CMAKE_CXX_COMPILER_ID}")
        endif ()
    else ()
        set_compiler_flags(stringzillas_cpus_shared "" "" "${CMAKE_CXX_COMPILER_ID}")
    endif ()
endif ()

# Define StringZillas CUDA shared library (only if CUDA is available)
if (STRINGZILLA_BUILD_SHARED AND STRINGZILLA_BUILD_CUDA)
    find_package(CUDAToolkit REQUIRED)

    # Each GPU tier compiles standalone to ONE PTX module (`nvcc -ptx`), then `bin2c`-embedded into the .so
    # as a NUL-terminated C array. No device code (no fatbin) is linked into the library itself; the host
    # side JIT-loads the matching tier's PTX at runtime via the CUDA Driver API (`cuda_host.cuh`).
    set(SZS_PTX_DIR "${CMAKE_CURRENT_BINARY_DIR}/szs_ptx")
    file(MAKE_DIRECTORY "${SZS_PTX_DIR}")
    set(SZS_PTX_BLOBS "")

    # The `.cu` TUs are thin wrappers that pull in the device kernels from the `.cuh` headers, so the PTX
    # must regenerate when any of those headers change (nvcc's `-ptx` custom command has no implicit header
    # tracking). Track the whole similarity/fingerprint device-header set as explicit dependencies.
    file(GLOB SZS_PTX_HEADERS
        "${CMAKE_CURRENT_SOURCE_DIR}/include/stringzillas/similarities/*.cuh"
        "${CMAKE_CURRENT_SOURCE_DIR}/include/stringzillas/fingerprints/*.cuh"
    )

    function (szs_add_ptx cu arch sym)
        set(ptx "${SZS_PTX_DIR}/${sym}.ptx")
        set(blob "${SZS_PTX_DIR}/${sym}.c")
        add_custom_command(
            OUTPUT "${ptx}"
            COMMAND "${CMAKE_CUDA_COMPILER}" -ptx -std=c++20 --expt-relaxed-constexpr -O3 -arch=${arch}
                    -ccbin "${CMAKE_CUDA_HOST_COMPILER}" -DSZ_USE_CUDA=1
                    -I "${CMAKE_CURRENT_SOURCE_DIR}/include" -I "${CMAKE_CURRENT_SOURCE_DIR}/c"
                    "${CMAKE_CURRENT_SOURCE_DIR}/${cu}" -o "${ptx}"
            DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/${cu}" ${SZS_PTX_HEADERS}
            COMMENT "nvcc -ptx ${sym} (${arch})"
            VERBATIM
        )
        # bin2c writes to stdout; wrap in a shell so the redirection works under both Make and Ninja.
        add_custom_command(
            OUTPUT "${blob}"
            COMMAND bash -c "'${CUDAToolkit_BIN_DIR}/bin2c' --name ${sym} '${ptx}' > '${blob}'"
            DEPENDS "${ptx}"
            COMMENT "bin2c ${sym}"
            VERBATIM
        )
        set(SZS_PTX_BLOBS ${SZS_PTX_BLOBS} "${blob}" PARENT_SCOPE)
    endfunction ()

    szs_add_ptx(c/stringzillas/cuda/similarities_base.cu compute_50 sim_base_ptx)
    szs_add_ptx(c/stringzillas/cuda/similarities_kepler.cu compute_70 sim_kepler_ptx)
    szs_add_ptx(c/stringzillas/cuda/similarities_hopper.cu compute_90a sim_hopper_ptx)
    szs_add_ptx(c/stringzillas/cuda/fingerprints_cuda.cu compute_50 fp_cuda_ptx)

    # The lowered C TUs (compiled as host C++ for the Driver-API launchers) + fork_union C ABI + PTX blobs.
    set(STRINGZILLAS_CUDA_SOURCES
        c/stringzillas/runtime.c c/stringzillas/similarities.c c/stringzillas/fingerprints.c
        fork_union/c/lib.cpp ${SZS_PTX_BLOBS}
    )
    define_stringzillas_shared(stringzillas_cuda_shared "${STRINGZILLAS_CUDA_SOURCES}" "SZ_USE_CUDA=1;SZ_USE_ROCM=0")

    # Compile the three lowered C TUs as host C++ (no device code in the lib) so they can include the
    # Driver-API launchers in `cuda_host.cuh`.
    set_source_files_properties(
        c/stringzillas/runtime.c c/stringzillas/similarities.c c/stringzillas/fingerprints.c
        TARGET_DIRECTORY stringzillas_cuda_shared PROPERTIES LANGUAGE CXX
    )
    target_include_directories(stringzillas_cuda_shared PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
    target_link_libraries(stringzillas_cuda_shared PRIVATE CUDA::cudart CUDA::cuda_driver)

    # x86/ARM SIMD flags for the host C++ (the icelake/skylake/haswell CPU kernels stay available here too).
    if (SZ_IS_64BIT_X86_)
        if (MSVC)
            set_compiler_flags(stringzillas_cuda_shared "" "SSE2" "${CMAKE_CXX_COMPILER_ID}")
        else ()
            set_compiler_flags(stringzillas_cuda_shared "" "ivybridge" "${CMAKE_CXX_COMPILER_ID}")
        endif ()
    elseif (SZ_IS_64BIT_ARM_)
        set_compiler_flags(stringzillas_cuda_shared "" "armv8-a" "${CMAKE_CXX_COMPILER_ID}")
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
