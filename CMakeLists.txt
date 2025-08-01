# StringZilla CMakeLists.txt
#
# This file defines several library build & installation targets:
#
# * stringzilla_header: A header-only library with the StringZilla C and C++ headers.
# * stringzilla_shared: A shared library with the StringZilla C and C++ headers and dynamic SIMD dispatch.
# * stringzilla_bare: A shared library with the StringZilla headers, but without linking the standard C library.
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
# * stringzilla_test_cpp20_haswell: A test executable for AVX2.
# * stringzilla_test_cpp20_ice: A test executable for AVX-512.
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
    VERSION 3.11.3
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
set(ENABLE_CUDA OFF)
include(CheckLanguage)
check_language(CUDA)

if (CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    set(ENABLE_CUDA ON)
    set(CMAKE_CUDA_STANDARD 20)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(CMAKE_CUDA_EXTENSIONS OFF)
    set(CMAKE_CUDA_ARCHITECTURES 80 90)
    set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
    find_package(CUDAToolkit REQUIRED)
    message(STATUS "CUDA detected!")
    message(STATUS "CUDA Compiler: ${CMAKE_CUDA_COMPILER}")
    message(STATUS "CUDA Compiler ID: ${CMAKE_CUDA_COMPILER_ID}")
    message(STATUS "CUDA Toolkit Version: ${CUDAToolkit_VERSION}")
    message(STATUS "CUDA Toolkit Include Path: ${CUDAToolkit_INCLUDE_DIRS}")
    message(STATUS "CUDA Toolkit Libraries Path: ${CUDAToolkit_LIBRARY_DIR}")
else ()
    message(STATUS "CUDA not detected. Skipping CUDA-specific builds.")
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

if (CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|amd64")
    set(SZ_PLATFORM_X86 TRUE)
    message(STATUS "Platform: x86")
elseif (CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|AARCH64|arm64|ARM64")
    set(SZ_PLATFORM_ARM TRUE)
    message(STATUS "Platform: ARM")
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
set(STRINGZILLA_TARGET_ARCH
    ""
    CACHE STRING "Architecture to tell the compiler to optimize for (-march)"
)

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

if (${CMAKE_VERSION} VERSION_EQUAL 3.13 OR ${CMAKE_VERSION} VERSION_GREATER 3.13)
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

    # Set output directory for multi-configuration generators (like Visual Studio)
    foreach (config IN LISTS CMAKE_CONFIGURATION_TYPES)
        string(TOUPPER ${config} config_upper)
        set_target_properties(${target} PROPERTIES RUNTIME_OUTPUT_DIRECTORY_${config_upper} ${CMAKE_BINARY_DIR}/$<0:>)
    endforeach ()

    # Set the C++ standard
    if (NOT ${cpp_standard} STREQUAL "")
        if (${compiler_id} STREQUAL "NVIDIA")
            set_target_properties(${target} PROPERTIES CUDA_STANDARD ${cpp_standard})
        else ()
            set_target_properties(${target} PROPERTIES CXX_STANDARD ${cpp_standard})
        endif ()
    endif ()

    # Use the `/Zc:__cplusplus` flag to correctly define the `__cplusplus` macro in MSVC
    if (${compiler_id} STREQUAL "MSVC")
        target_compile_options(${target} PRIVATE "/Zc:__cplusplus")
    endif ()

    # Make sure CUDA C++ allows calling `constexpr` from device code
    if (${compiler_id} STREQUAL "NVIDIA")
        target_compile_options(${target} PRIVATE "--expt-relaxed-constexpr")
    endif ()

    # Maximum warnings level & warnings as error.
    #
    # MSVC uses numeric values: > 4068 for "unknown pragmas". > 4146 for "unary minus operator applied to unsigned type,
    # result still unsigned". We also specify `/utf-8` to properly UTF-8 symbols in tests.
    if (${compiler_id} STREQUAL "GNU")
        target_compile_options(
            ${target}
            PRIVATE
                "-Wall;-Wextra;-pedantic;-Werror;-Wfatal-errors;-Wno-unknown-pragmas;-Wno-cast-function-type;-Wno-unused-function"
        )
        target_compile_options(${target} PRIVATE "-Wno-cast-function-type;-Wno-unused-function") # ? Unique to GCC
    elseif (${compiler_id} STREQUAL "Clang" OR ${compiler_id} STREQUAL "AppleClang")
        target_compile_options(${target} PRIVATE "-Wall;-Wextra;-pedantic;-Werror;-Wfatal-errors;-Wno-unknown-pragmas")
    elseif (${compiler_id} STREQUAL "MSVC")
        target_compile_options(${target} PRIVATE "/Bt;/wd4068;/wd4146;/utf-8;/WX")
    elseif (${compiler_id} STREQUAL "NVIDIA")
        target_compile_options(
            ${target}
            PRIVATE
                "-Xcompiler=-Wfatal-errors;-Xcompiler=-Wall;-Xcompiler=-Wextra;-Wno-unknown-pragmas;-Wno-cast-function-type;-Wno-unused-function"
        )
    endif ()

    # Enable OpenMP if available
    if (NOT target_type STREQUAL "SHARED_LIBRARY")
        if (${compiler_id} STREQUAL "GNU"
            OR ${compiler_id} STREQUAL "Clang"
            OR ${compiler_id} STREQUAL "NVIDIA"
        )
            target_compile_options(${target} PRIVATE "-fopenmp")
            target_link_libraries(${target} PRIVATE "-fopenmp")
        elseif (${compiler_id} STREQUAL "MSVC")
            target_compile_options(${target} PRIVATE "/openmp")
            target_link_libraries(${target} PRIVATE "/openmp")
        endif ()
    endif ()

    # Set optimization options for different compilers differently
    if (${compiler_id} STREQUAL "MSVC")
        if (${CMAKE_BUILD_TYPE} STREQUAL "Debug" OR ${CMAKE_BUILD_TYPE} STREQUAL "RelWithDebInfo")
            target_compile_options(${target} PRIVATE "/Od;/Zi")
            if (NOT target_type STREQUAL "SHARED_LIBRARY")
                target_compile_options(${target} PRIVATE "/RTC1")
            endif ()
        endif ()
        if (${CMAKE_BUILD_TYPE} STREQUAL "Release" OR ${CMAKE_BUILD_TYPE} STREQUAL "RelWithDebInfo")
            target_compile_options(${target} PRIVATE "/O2;/Zi")
        endif ()
    elseif (
        ${compiler_id} STREQUAL "GNU"
        OR ${compiler_id} STREQUAL "Clang"
        OR ${compiler_id} STREQUAL "AppleClang"
    )
        if (${CMAKE_BUILD_TYPE} STREQUAL "Debug" OR ${CMAKE_BUILD_TYPE} STREQUAL "RelWithDebInfo")
            target_compile_options(${target} PRIVATE "-O0;-g")
        endif ()
        if (${CMAKE_BUILD_TYPE} STREQUAL "Release" OR ${CMAKE_BUILD_TYPE} STREQUAL "RelWithDebInfo")
            target_compile_options(${target} PRIVATE "-O2")
        endif ()
    elseif (${compiler_id} STREQUAL "NVIDIA")
        if (${CMAKE_BUILD_TYPE} STREQUAL "Debug" OR ${CMAKE_BUILD_TYPE} STREQUAL "RelWithDebInfo")
            target_compile_options(
                ${target}
                PRIVATE "-G" # Device debug symbols
                        "-lineinfo" # Include source line info in PTX
                        "-O0" # Disable NVCC optimizations explicitly
                        "-no-compress" # No compression of debug info
                        "-Xptxas=-O0" # Disable PTX assembler optimizations
                        "-Xcompiler=-g" # Host debugging symbols explicitly
                        "-Xcompiler=-O0" # Host optimizations off
                        "-Xcompiler=-fno-omit-frame-pointer" # Stack trace clarity
                        "-Xcompiler=-Wall" # All warnings (host)
                        "-Xcompiler=-Wextra" # Extra warnings (host)
                        "-Xcompiler=-fno-inline" # Prevent host inlining
                        "-maxrregcount=0" # No register count limits
            )
        endif ()
        if (${CMAKE_BUILD_TYPE} STREQUAL "Release" OR ${CMAKE_BUILD_TYPE} STREQUAL "RelWithDebInfo")
            target_compile_options(${target} PRIVATE "-O2")
        endif ()
    endif ()

    # If available, enable Position Independent Code
    get_target_property(target_pic ${target} POSITION_INDEPENDENT_CODE)
    if (target_pic)
        target_compile_definitions(${target} PRIVATE "SZ_PIC")
    endif ()

    # Avoid builtin functions where we know what we are doing.
    if (${compiler_id} STREQUAL "MSVC")
        target_compile_options(${target} PRIVATE "/Oi-")
    else ()
        target_compile_options(${target} PRIVATE "-fno-builtin-memcmp")
        target_compile_options(${target} PRIVATE "-fno-builtin-memchr")
        target_compile_options(${target} PRIVATE "-fno-builtin-memcpy")
        target_compile_options(${target} PRIVATE "-fno-builtin-memset")
    endif ()

    # Check for ${target_arch} and set it or use the current system if not defined
    if ("${target_arch}" STREQUAL "")
        # Only use the current system if we are not cross compiling
        if ((NOT CMAKE_CROSSCOMPILING) OR (CMAKE_SYSTEM_PROCESSOR MATCHES CMAKE_HOST_SYSTEM_PROCESSOR))
            if (NOT (${compiler_id} STREQUAL "MSVC"))
                include(CheckCXXCompilerFlag)
                check_cxx_compiler_flag("-march=native" supports_march_native)
                if (supports_march_native)
                    target_compile_options(${target} PRIVATE "-march=native")
                endif ()
            else ()
                # MSVC does not have a direct equivalent to -march=native
                target_compile_options(${target} PRIVATE "/arch:AVX2")
            endif ()
        endif ()
    else ()
        if (${compiler_id} STREQUAL "MSVC")
            target_compile_options(${target} PRIVATE "/arch:${target_arch}")
        else ()
            target_compile_options(${target} PRIVATE "-march=${target_arch}")
        endif ()
    endif ()

    # Define SZ_DETECT_BIG_ENDIAN macro based on system byte order
    if (CMAKE_C_BYTE_ORDER STREQUAL "BIG_ENDIAN")
        set(SZ_DETECT_BIG_ENDIAN 1)
    else ()
        set(SZ_DETECT_BIG_ENDIAN 0)
    endif ()

    target_compile_definitions(${target} PRIVATE "SZ_DETECT_BIG_ENDIAN=${SZ_DETECT_BIG_ENDIAN}")

    # Sanitizer options for Debug mode
    if (CMAKE_BUILD_TYPE STREQUAL "Debug")
        target_compile_definitions(${target} PRIVATE "SZ_DEBUG=1")
        if (NOT target_type STREQUAL "SHARED_LIBRARY")
            if (${compiler_id} STREQUAL "MSVC")
                target_compile_options(${target} PRIVATE "/fsanitize=address;/fsanitize=leak")
                target_link_options(${target} PRIVATE "/fsanitize=address;/fsanitize=leak")
            elseif (${compiler_id} STREQUAL "NVIDIA")
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
    # TODO: How do we constrain this scope?! set_source_files_properties(${source} TARGET_DIRECTORY ${exec_name}
    # PROPERTIES LANGUAGE CXX)
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

if (${STRINGZILLA_BUILD_BENCHMARK})
    define_launcher(stringzilla_bench_find_cpp20 scripts/bench_find.cpp 20 "${STRINGZILLA_TARGET_ARCH}")
    define_launcher(stringzilla_bench_sequence_cpp20 scripts/bench_sequence.cpp 20 "${STRINGZILLA_TARGET_ARCH}")
    define_launcher(stringzilla_bench_token_cpp20 scripts/bench_token.cpp 20 "${STRINGZILLA_TARGET_ARCH}")
    define_launcher(stringzilla_bench_container_cpp20 scripts/bench_container.cpp 20 "${STRINGZILLA_TARGET_ARCH}")
    define_launcher(stringzilla_bench_memory_cpp20 scripts/bench_memory.cpp 20 "${STRINGZILLA_TARGET_ARCH}")

    # Parallel benchmarks
    define_launcher(
        stringzillas_bench_similarities_cpp20 scripts/bench_similarities.cpp 20 "${STRINGZILLA_TARGET_ARCH}"
    )
    define_launcher(
        stringzillas_bench_fingerprints_cpp20 scripts/bench_fingerprints.cpp 20 "${STRINGZILLA_TARGET_ARCH}"
    )
    if (ENABLE_CUDA)
        define_gpu_launcher(
            stringzillas_bench_similarities_cu20 scripts/bench_similarities.cu 20 "${STRINGZILLA_TARGET_ARCH}"
        )
        define_gpu_launcher(
            stringzillas_bench_fingerprints_cu20 scripts/bench_fingerprints.cu 20 "${STRINGZILLA_TARGET_ARCH}"
        )
    endif ()
endif ()

if (${STRINGZILLA_BUILD_TEST})
    # Make sure that the compilation passes for different C++ standards!
    #
    # Keep in mind, MSVC only supports C++11 and newer.
    define_launcher(stringzilla_test_cpp11 scripts/test_stringzilla.cpp 11 "${STRINGZILLA_TARGET_ARCH}")
    define_launcher(stringzilla_test_cpp14 scripts/test_stringzilla.cpp 14 "${STRINGZILLA_TARGET_ARCH}")
    define_launcher(stringzilla_test_cpp17 scripts/test_stringzilla.cpp 17 "${STRINGZILLA_TARGET_ARCH}")
    define_launcher(stringzilla_test_cpp20 scripts/test_stringzilla.cpp 20 "${STRINGZILLA_TARGET_ARCH}")

    # Test parallel algorithms separately
    define_launcher(stringzillas_test_cpp17 scripts/test_stringzillas.cpp 17 "${STRINGZILLA_TARGET_ARCH}")
    define_launcher(stringzillas_test_cpp20 scripts/test_stringzillas.cpp 20 "${STRINGZILLA_TARGET_ARCH}")

    # To avoid bloating our codebase with `__device__` function annotations, we only target C++14 and newer to compile
    # `constexpr` functions on both host and device side. To avoid the complexity of defining too many template objects
    # and complex SFINAE, we only target C++17 anf newer to compile `if constexpr` compile-time SIMD dispatch.
    if (ENABLE_CUDA)
        define_gpu_launcher(stringzillas_test_cu17 scripts/test_stringzillas.cu 17 "${STRINGZILLA_TARGET_ARCH}")
        define_gpu_launcher(stringzillas_test_cu20 scripts/test_stringzillas.cu 20 "${STRINGZILLA_TARGET_ARCH}")
    endif ()

    # Check system architecture to avoid complex cross-compilation workflows, but compile multiple backends: disabling
    # all SIMD, enabling only AVX2, only AVX-512, only Arm Neon.
    if (SZ_PLATFORM_X86)
        # x86 specific backends
        if (MSVC)
            define_launcher(stringzilla_test_cpp20_serial scripts/test_stringzilla.cpp 20 "AVX")
            define_launcher(stringzilla_test_cpp20_haswell scripts/test_stringzilla.cpp 20 "AVX2")
            define_launcher(stringzilla_test_cpp20_ice scripts/test_stringzilla.cpp 20 "AVX512")
            if (ENABLE_CUDA)
                define_gpu_launcher(stringzillas_test_cu20_serial scripts/test_stringzillas.cu 20 "AVX")
                define_gpu_launcher(stringzillas_test_cu20_haswell scripts/test_stringzillas.cu 20 "AVX2")
                define_gpu_launcher(stringzillas_test_cu20_ice scripts/test_stringzillas.cu 20 "AVX512")
            endif ()
        else ()
            define_launcher(stringzilla_test_cpp20_serial scripts/test_stringzilla.cpp 20 "ivybridge")
            define_launcher(stringzilla_test_cpp20_haswell scripts/test_stringzilla.cpp 20 "haswell")
            define_launcher(stringzilla_test_cpp20_ice scripts/test_stringzilla.cpp 20 "sapphirerapids")
            if (ENABLE_CUDA)
                define_gpu_launcher(stringzillas_test_cu20_serial scripts/test_stringzillas.cu 20 "ivybridge")
                define_gpu_launcher(stringzillas_test_cu20_haswell scripts/test_stringzillas.cu 20 "haswell")
                define_gpu_launcher(stringzillas_test_cu20_ice scripts/test_stringzillas.cu 20 "sapphirerapids")
            endif ()
        endif ()
    elseif (SZ_PLATFORM_ARM)
        # ARM specific backends
        define_launcher(stringzilla_test_cpp20_serial scripts/test_stringzilla.cpp 20 "armv8-a")
        define_launcher(stringzilla_test_cpp20_neon scripts/test_stringzilla.cpp 20 "armv8-a+simd")
        define_launcher(stringzilla_test_cpp20_sve scripts/test_stringzilla.cpp 20 "armv8.2-a+sve")
        if (ENABLE_CUDA)
            define_gpu_launcher(stringzillas_test_cu20_serial scripts/test_stringzillas.cu 20 "armv8-a")
            define_gpu_launcher(stringzillas_test_cu20_neon scripts/test_stringzillas.cu 20 "armv8-a+simd")
            define_gpu_launcher(stringzillas_test_cu20_sve scripts/test_stringzillas.cu 20 "armv8.2-a+sve")
        endif ()
    endif ()
endif ()

# Define our libraries, first the header-only version
add_library(stringzilla_header INTERFACE)
add_library(${PROJECT_NAME}::stringzilla_header ALIAS stringzilla_header)
target_include_directories(
    stringzilla_header INTERFACE $<BUILD_INTERFACE:${STRINGZILLA_INCLUDE_BUILD_DIR}> $<INSTALL_INTERFACE:include>
)

if (${STRINGZILLA_BUILD_SHARED})

    function (define_shared target)
        add_library(${target} SHARED c/stringzilla.c)
        add_library(${PROJECT_NAME}::${target} ALIAS ${target})

        set_target_properties(
            ${target}
            PROPERTIES VERSION ${PROJECT_VERSION}
                       SOVERSION 1
                       POSITION_INDEPENDENT_CODE ON
        )

        if (SZ_PLATFORM_X86)
            if (MSVC)
                set_compiler_flags(${target} "" "SSE2" "${CMAKE_CXX_COMPILER_ID}")
            else ()
                set_compiler_flags(${target} "" "ivybridge" "${CMAKE_CXX_COMPILER_ID}")
            endif ()

            target_compile_definitions(
                ${target} PRIVATE "SZ_USE_HASWELL=1" "SZ_USE_SKYLAKE=1" "SZ_USE_ICE=1" "SZ_USE_NEON=0" "SZ_USE_SVE=0"
                                  "SZ_USE_SVE2=0"
            )
        elseif (SZ_PLATFORM_ARM)
            set_compiler_flags(${target} "" "armv8-a" "${CMAKE_CXX_COMPILER_ID}")

            target_compile_definitions(
                ${target} PRIVATE "SZ_USE_HASWELL=0" "SZ_USE_SKYLAKE=0" "SZ_USE_ICE=0" "SZ_USE_NEON=1" "SZ_USE_SVE=1"
                                  "SZ_USE_SVE2=1"
            )
        endif ()

        if (MSVC)
            # Add dependencies for necessary runtime libraries in case of static linking. This ensures that basic
            # runtime functions are available:
            #
            # * msvcrt.lib: Microsoft Visual C Runtime, required for basic C runtime functions on Windows.
            # * vcruntime.lib: Microsoft Visual C++ Runtime library for basic runtime functions.
            # * ucrt.lib: Universal C Runtime, necessary for linking basic C functions like I/O.
            target_link_libraries(${target} PRIVATE msvcrt.lib vcruntime.lib ucrt.lib)
        endif ()

    endfunction ()

    define_shared(stringzilla_shared)
    target_compile_definitions(stringzilla_shared PRIVATE "SZ_AVOID_LIBC=0")
    target_compile_definitions(stringzilla_shared PRIVATE "SZ_OVERRIDE_LIBC=1")
    target_include_directories(stringzilla_shared PUBLIC include)

    # Try compiling a version without linking the LibC ! This is only for Linux and Windows, as on modern Arm-based
    # MacOS machines ! We can't legally access Arm's "feature registers" without `sysctl` or `sysctlbyname`. So let's
    # check if we are compiling for a Darwin-based OS.
    if (NOT ${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
        define_shared(stringzilla_bare)
        target_compile_definitions(stringzilla_bare PRIVATE "SZ_AVOID_LIBC=1")
        target_compile_definitions(stringzilla_bare PRIVATE "SZ_OVERRIDE_LIBC=1")
        target_include_directories(stringzilla_bare PUBLIC include)

        # Avoid built-ins on MSVC and other compilers, as that will cause compilation errors
        target_compile_options(
            stringzilla_bare PRIVATE "$<$<CXX_COMPILER_ID:GNU,Clang>:-fno-builtin;-nostdlib>"
                                     "$<$<CXX_COMPILER_ID:MSVC>:/Oi-;/GS->"
        )
        target_link_options(stringzilla_bare PRIVATE "$<$<CXX_COMPILER_ID:GNU,Clang>:-nostdlib>")
        target_link_options(stringzilla_bare PRIVATE "$<$<CXX_COMPILER_ID:MSVC>:/NODEFAULTLIB>")
    endif ()
endif ()

if (STRINGZILLA_INSTALL)
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
    install(DIRECTORY ${STRINGZILLA_INCLUDE_BUILD_DIR} DESTINATION ${STRINGZILLA_INCLUDE_INSTALL_DIR})
    install(DIRECTORY ./c/ DESTINATION /usr/src/${PROJECT_NAME}/)
endif ()
