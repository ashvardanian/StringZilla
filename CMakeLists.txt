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
# Consumers link the same namespaced spellings whether pulled in with `add_subdirectory` or found with `find_package`:
#
# * stringzilla::header, stringzilla::shared, stringzilla::bare - from `find_package(stringzilla)`.
# * stringzillas::cpus_shared, stringzillas::cuda_shared - from `find_package(stringzillas)`. The `cpus_static` and
#   `cuda_static` siblings are in-tree only; see the install rules below.
#
# The longer spellings (stringzilla::stringzilla_header and the rest) stay available as aliases.
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
# * stringzillas_test_cpp20 (and stringzillas_test_cpp23 where the compiler advertises C++23): Parallel-algorithm tests
#   for the host's best CPU backend.
# * stringzillas_test_cu20 (and stringzillas_test_cu23 where nvcc advertises CUDA23): Parallel-algorithm tests on
#   CUDA-capable GPUs (base SIMT, Kepler, and Hopper tiers selected at runtime).
#
# Serial Benchmarks:
#
# * stringzilla_bench_find_cpp20: A benchmark for substring search operations.
# * stringzilla_bench_sequence_cpp20: A benchmark for string array-level operations.
# * stringzilla_bench_token_cpp20: A benchmark for comparators and hash functions.
# * stringzilla_bench_container_cpp20: A benchmark for STL containers powered by StringZilla.
# * stringzilla_bench_memory_cpp20: A benchmark for LibC-style low-level memory operations.
# * stringzilla_bench_utf8_uncased_cpp20: A benchmark for the `sz_utf8_uncased_*` folding/search family.
# * stringzilla_bench_utf8_traverse_cpp20: A benchmark for the `sz_utf8_*` traversal/transcode family
#   (count/seek/decode).
# * stringzilla_bench_utf8_scan_cpp20: A benchmark for the `sz_utf8_*` class-scan family
#   (newlines/whitespaces/delimiters).
# * stringzilla_bench_utf8_segment_cpp20: A benchmark for the `sz_utf8_*` boundary-segmentation family
#   (words/graphemes/sentences/linebreaks).
# * stringzilla_bench_utf8_norm_cpp20: A benchmark for the `sz_utf8_norm` normalization family.
#
# Parallel Benchmarks:
#
# * stringzillas_bench_similarities_cpp20: A benchmark for similarity operations.
# * stringzillas_bench_similarities_cu20: A benchmark for similarity operations on GPU.
# * stringzillas_bench_fingerprints_cpp20: A benchmark for finding many substrings.
# * stringzillas_bench_fingerprints_cu20: A benchmark for finding many substrings on GPU.
#
# For higher-level language bindings separate build scripts are provided, native to each toolchain.
cmake_minimum_required(VERSION 3.21 FATAL_ERROR)
project(
    stringzilla
    VERSION 5.1.2
    LANGUAGES C CXX ASM
    DESCRIPTION "Search, hash, sort, fingerprint, and fuzzy-match strings faster via SWAR, SIMD, and GPGPU"
    HOMEPAGE_URL "https://github.com/ashvardanian/stringzilla"
)

set(CMAKE_C_STANDARD 99)
set(CMAKE_CXX_STANDARD 11)

set(CMAKE_C_EXTENSIONS OFF)
set(CMAKE_CXX_EXTENSIONS OFF)

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

# Set a default build type to "Release" if none was specified - as a plain directory-scope variable,
# so a subproject build never forces its default into the parent project's cache.
if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(STATUS "Setting build type to 'Release' as none was specified.")
    set(CMAKE_BUILD_TYPE Release)
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
elseif (CMAKE_SYSTEM_PROCESSOR MATCHES "loongarch64|LOONGARCH64|loong64|LOONG64")
    # "loongarch64" is the GNU triple and `uname -m` spelling; "loong64" is the Debian/dpkg port name.
    set(SZ_IS_64BIT_LOONGARCH_ TRUE)
    message(STATUS "Platform: LoongArch64 (CMAKE_SYSTEM_PROCESSOR=${CMAKE_SYSTEM_PROCESSOR})")
elseif (CMAKE_SYSTEM_PROCESSOR MATCHES "ppc64|powerpc64|PPC64|POWERPC64")
    # Matches both endiannesses ("ppc64le" contains "ppc64"); the kernels handle big-endian explicitly
    # (`find/powervsx.h`) and `SZ_IS_BIG_ENDIAN_` is derived from `CMAKE_C_BYTE_ORDER` separately.
    set(SZ_IS_64BIT_POWER_ TRUE)
    message(STATUS "Platform: PowerPC64 (CMAKE_SYSTEM_PROCESSOR=${CMAKE_SYSTEM_PROCESSOR})")
elseif (CMAKE_SYSTEM_PROCESSOR MATCHES "wasm")
    # WebAssembly is 32-bit; SIMD comes from the toolchain's `-msimd128` / `-mrelaxed-simd`, and the
    # `set_architecture_simd_definitions` wasm branch keys off `CMAKE_SYSTEM_NAME`, not a 64-bit flag.
    message(STATUS "Platform: WebAssembly (CMAKE_SYSTEM_PROCESSOR=${CMAKE_SYSTEM_PROCESSOR})")
else ()
    message(WARNING "Unknown CMAKE_SYSTEM_PROCESSOR: ${CMAKE_SYSTEM_PROCESSOR}")
endif ()

# ISA capability probes: which SIMD capabilities can this toolchain COMPILE, and which can this machine RUN. The probe
# sources under `probes/` are shared with the Cargo build (`build.rs`); the model and the helper macros are documented
# in `cmake/sz_isa_probe.cmake`. Each module fills the cached `SZ_COMPILABLE_CAPABILITIES` set and `SZ_ISA_CAPABILITIES`.
if (SZ_IS_64BIT_X86_)
    include(cmake/sz_x86_isa_probes.cmake)
elseif (SZ_IS_64BIT_ARM_)
    include(cmake/sz_arm_isa_probes.cmake)
elseif (SZ_IS_64BIT_RISCV_)
    include(cmake/sz_riscv_isa_probes.cmake)
elseif (SZ_IS_64BIT_LOONGARCH_)
    include(cmake/sz_loongarch_isa_probes.cmake)
elseif (SZ_IS_64BIT_POWER_)
    include(cmake/sz_power_isa_probes.cmake)
elseif (
    CMAKE_SYSTEM_NAME STREQUAL "WASI"
    OR CMAKE_SYSTEM_NAME STREQUAL "Emscripten"
    OR CMAKE_SYSTEM_PROCESSOR MATCHES "wasm"
)
    include(cmake/sz_wasm_isa_probes.cmake)
else ()
    # Unknown architecture: no SIMD capability probes, but the shared probe module must still load so the
    # `sz_runtime_detectable_` / `sz_machine_capabilities_` calls below resolve and configuration
    # degrades to a serial-only build instead of failing.
    include(cmake/sz_isa_probe.cmake)
    set(SZ_ISA_CAPABILITIES "")
endif ()

# SIMD capability verdicts: fold the probe results, the machine answer, and any `-D SZ_USE_<X>=0/1` overrides into two
# capability sets:
#   SZ_CAPABILITIES_TO_COMPILE → runtime-dispatched libraries (the load-time table masks what the CPU lacks)
#   SZ_CAPABILITIES_TO_RUN     → comptime-dispatched tests & benchmarks (the picked capability must also run HERE)
# When a set is unknowable — no runtime detection on this platform, or a machine we cannot probe — the matching
# `SZ_*_CAPABILITIES_KNOWN` flag stays 0, targets get no explicit definitions, and `types.h` auto-detection under
# their own flags decides, as it did before the probes existed.
sz_runtime_detectable_()
set(SZ_COMPILE_CAPABILITIES_KNOWN ${SZ_RUNTIME_DETECTABLE})
set(SZ_RUN_CAPABILITIES_KNOWN 0)
if (SZ_RUNTIME_DETECTABLE)
    sz_machine_capabilities_()
    if (NOT "${SZ_MACHINE_CAPABILITIES}" STREQUAL "")
        set(SZ_RUN_CAPABILITIES_KNOWN 1)
        string(REPLACE "," ";" sz_machine_tokens_ "${SZ_MACHINE_CAPABILITIES}")
    endif ()
endif ()
set(SZ_CAPABILITIES_TO_COMPILE "")
set(SZ_CAPABILITIES_TO_RUN "")
foreach (sz_capability_ IN LISTS SZ_ISA_CAPABILITIES)
    if (sz_capability_ IN_LIST SZ_COMPILABLE_CAPABILITIES)
        set(sz_to_compile_ 1)
    else ()
        set(sz_to_compile_ 0)
    endif ()
    set(sz_to_run_ ${sz_to_compile_})
    string(TOLOWER "${sz_capability_}" sz_token_)
    if (sz_to_run_
        AND SZ_RUN_CAPABILITIES_KNOWN
        AND NOT sz_token_ IN_LIST sz_machine_tokens_
    )
        set(sz_to_run_ 0)
    endif ()
    if (DEFINED SZ_USE_${sz_capability_}) # The user override wins both verdicts, except past a failed compile probe.
        # A command-line `-D SZ_USE_X` becomes a cache entry that outlives its command line, so every
        # configure names the overrides in effect - a stale one is visible instead of silently sticky.
        message(STATUS "SZ_USE_${sz_capability_} override in effect: ${SZ_USE_${sz_capability_}}")
        if (SZ_USE_${sz_capability_} AND NOT sz_to_compile_)
            message(WARNING "SZ_USE_${sz_capability_}=1 requested, but its probe does not compile here; ignoring")
        elseif (SZ_USE_${sz_capability_})
            set(sz_to_compile_ 1)
            set(sz_to_run_ 1)
        else ()
            set(sz_to_compile_ 0)
            set(sz_to_run_ 0)
        endif ()
    endif ()
    if (sz_to_compile_)
        list(APPEND SZ_CAPABILITIES_TO_COMPILE ${sz_capability_})
    endif ()
    if (sz_to_run_)
        list(APPEND SZ_CAPABILITIES_TO_RUN ${sz_capability_})
    endif ()
endforeach ()
if (SZ_COMPILE_CAPABILITIES_KNOWN)
    message(STATUS "SIMD capabilities to compile: [${SZ_CAPABILITIES_TO_COMPILE}]")
endif ()
if (SZ_RUN_CAPABILITIES_KNOWN)
    message(STATUS "SIMD capabilities to run here: [${SZ_CAPABILITIES_TO_RUN}]")
endif ()
unset(sz_to_compile_)
unset(sz_to_run_)
unset(sz_token_)
unset(sz_machine_tokens_)

# Determine if StringZilla is built as a sub-project (using `add_subdirectory`) or if it is the main project
set(STRINGZILLA_IS_MAIN_PROJECT ${PROJECT_IS_TOP_LEVEL})

# CUDA auto-enables only for top-level builds: a parent project consuming StringZilla for strings
# should not sprout CUDA targets just because `nvcc` happens to be on the PATH.
set(STRINGZILLA_DEFAULT_CUDA OFF)

if (STRINGZILLA_IS_MAIN_PROJECT AND STRINGZILLA_CAN_BUILD_CUDA)
    set(STRINGZILLA_DEFAULT_CUDA ON)
endif ()

# Installation options
option(STRINGZILLA_INSTALL "Install CMake targets" OFF)
option(STRINGZILLA_BUILD_TEST "Compile a native unit test in C++" ${STRINGZILLA_IS_MAIN_PROJECT})
option(STRINGZILLA_BUILD_BENCHMARK "Compile a native benchmark in C++" ${STRINGZILLA_IS_MAIN_PROJECT})
option(STRINGZILLA_BUILD_SHARED "Compile a dynamic library" ${STRINGZILLA_IS_MAIN_PROJECT})
option(STRINGZILLAS_BUILD_SHARED "Compile dynamic parallel libraries" ${STRINGZILLA_IS_MAIN_PROJECT})
option(STRINGZILLA_BUILD_CUDA "Build CUDA-accelerated targets" ${STRINGZILLA_DEFAULT_CUDA})
option(STRINGZILLA_USE_SANITIZERS "Enable AddressSanitizer and UndefinedBehaviorSanitizer in Debug builds" ON)

# ForkUnion supplies the CPU thread pools, consumed through its C API: `forkunion::header` carries the headers, and
# `forkunion::static` carries the compiled runtime with the capability dispatch (NUMA-aware placement, colocated
# scheduling, huge pages - all via sysfs and raw syscalls, no libnuma), so no StringZilla translation unit
# instantiates its C++ core.
# A parent project may already provide ForkUnion - as its own submodule or an installed package - and
# adding the vendored copy on top would collide on target names. The vendored copy is only the
# fallback, and an unpopulated nested submodule fails with an actionable message instead of a
# cryptic `add_subdirectory` error.
#
# Only the StringZillas libraries and the launchers need ForkUnion - the header-only target does not, and a
# submodule-less tree is exactly what a release tarball, a vcpkg port, and a Conan recipe unpack. Tests and benchmarks
# count because `define_launcher` links `forkunion::header` into every executable.
set(STRINGZILLA_NEEDS_FORKUNION ${STRINGZILLAS_BUILD_SHARED})
if (STRINGZILLA_BUILD_TEST OR STRINGZILLA_BUILD_BENCHMARK)
    set(STRINGZILLA_NEEDS_FORKUNION ON)
endif ()

if (STRINGZILLA_NEEDS_FORKUNION AND NOT TARGET forkunion_header)
    if (NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/forkunion/CMakeLists.txt")
        message(FATAL_ERROR "The `forkunion` submodule is not populated - run: git submodule update --init --recursive")
    endif ()
    # `forkunion::static` is PUBLIC on the StringZillas libraries, so `find_dependency(forkunion)` needs it installed
    # alongside. ForkUnion's own default is top-level-only, which would leave the export dangling.
    set(FORKUNION_INSTALL ${STRINGZILLA_INSTALL})
    add_subdirectory(forkunion)
endif ()

if (TARGET forkunion_static)
    set_target_properties(forkunion_static PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif ()

# Clang source-based coverage for the header-only library. Recipe (single-arch, native backends only): cmake -B
# build_cov -DSTRINGZILLA_BUILD_COVERAGE=1 -DSTRINGZILLA_USE_SANITIZERS=0 -DCMAKE_CXX_COMPILER=clang++ cmake --build
# build_cov --target stringzilla_test_cpp17 -j LLVM_PROFILE_FILE=/tmp/sz.profraw ./build_cov/stringzilla_test_cpp17
# llvm-profdata merge -sparse /tmp/sz.profraw -o /tmp/sz.profdata llvm-cov report ./build_cov/stringzilla_test_cpp17
# -instr-profile=/tmp/sz.profdata include/stringzilla/ `-march=native` compiles out the non-native backends, so a
# faithful picture needs `llvm-profdata merge` across per-ISA builds (native x86 + qemu-aarch64 for NEON/SVE + the RVV
# qemu toolchain).
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
    # Real SASS for the supported datacenter generations - Ampere sm_80 and Hopper sm_90 (the DPX kernels need only
    # sm_90, not the H100-locked sm_90a). One constant serves the fallback below, the static library's base tier, and
    # the GPU launchers; per-tier targets override it with their own sets.
    set(STRINGZILLA_CUDA_ARCHS "80-real;90-real")
    set(CMAKE_CUDA_ARCHITECTURES ${STRINGZILLA_CUDA_ARCHS})
    set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
    find_package(CUDAToolkit REQUIRED)
    # CUDA 13+ / CCCL 3.x ships CUB, Thrust, and libcudacxx under include/cccl/
    find_path(
        STRINGZILLA_CCCL_INCLUDE_DIR
        NAMES cuda/iterator
        PATHS ${CUDAToolkit_INCLUDE_DIRS}
        PATH_SUFFIXES cccl
        NO_DEFAULT_PATH
    )
    message(STATUS "CUDA support enabled")
    message(STATUS "CUDA Compiler: ${CMAKE_CUDA_COMPILER}")
    message(STATUS "CUDA Compiler ID: ${CMAKE_CUDA_COMPILER_ID}")
    message(STATUS "CUDA Toolkit Version: ${CUDAToolkit_VERSION}")
    message(STATUS "CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
endif ()

# Configuration
include(GNUInstallDirs)
include(CheckCXXCompilerFlag)
set(STRINGZILLA_INCLUDE_BUILD_DIR "${PROJECT_SOURCE_DIR}/include/")
set(STRINGZILLA_INCLUDE_INSTALL_DIR "${CMAKE_INSTALL_INCLUDEDIR}")

# CTest machinery only for top-level builds: a parent project decides its own testing story, and
# `add_test` calls below are inert without `enable_testing()`.
if (STRINGZILLA_IS_MAIN_PROJECT)
    include(CTest)
    enable_testing()
endif ()

if (MSVC)
    # Remove /RTC* from MSVC debug flags by default (it will be added back in the set_compiler_flags function) Because
    # /RTC* cannot be used without the crt so it needs to be disabled for that specific target
    string(REGEX REPLACE "/RTC[^ ]*" "" CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG}")
    string(REGEX REPLACE "/RTC[^ ]*" "" CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG}")
endif ()

include(cmake/sz_compiler_flags.cmake)

function (define_launcher exec_name cpp_standard target_arch)
    add_executable(${exec_name})
    target_sources(${exec_name} PRIVATE ${ARGN})
    target_include_directories(${exec_name} PRIVATE test bench)
    target_link_libraries(${exec_name} PRIVATE forkunion::header)
    set_compiler_flags(${exec_name} ${cpp_standard} "${target_arch}" "${CMAKE_CXX_COMPILER_ID}")
    # Launchers dispatch at compile time, so a capability must be BOTH compilable by this toolchain AND runnable on
    # this machine — the intersection the RUN verdicts encode. When the caller pins an explicit `target_arch`, that
    # `-march` (plus `types.h` auto-detection) is the capability request, so the machine gate must not override it.
    if ("${target_arch}" STREQUAL "")
        set_architecture_simd_definitions(${exec_name} RUN)
    endif ()
    target_link_libraries(${exec_name} PRIVATE stringzilla_header)
    add_test(NAME ${exec_name} COMMAND ${exec_name})
endfunction ()

function (define_gpu_launcher exec_name cuda_standard target_arch)
    add_executable(${exec_name})
    target_sources(${exec_name} PRIVATE ${ARGN})
    set_source_files_properties(${ARGN} TARGET_DIRECTORY ${exec_name} PROPERTIES LANGUAGE CUDA)
    target_include_directories(${exec_name} PRIVATE test bench)
    target_link_libraries(${exec_name} PRIVATE forkunion::header)
    target_compile_definitions(${exec_name} PRIVATE ${STRINGZILLAS_COMMON_DEFINITIONS} "SZ_USE_CUDA=1" "SZ_USE_ROCM=0")
    set_target_properties(${exec_name} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    target_include_directories(
        ${exec_name} PRIVATE ${CUDAToolkit_INCLUDE_DIRS}
                             $<$<BOOL:${STRINGZILLA_CCCL_INCLUDE_DIR}>:${STRINGZILLA_CCCL_INCLUDE_DIR}>
    )
    set_compiler_flags(${exec_name} ${cuda_standard} "${target_arch}" "${CMAKE_CUDA_COMPILER_ID}")
    target_link_libraries(${exec_name} PRIVATE CUDA::cudart CUDA::cuda_driver)
    # Match the StringZillas CUDA static library's base tier, so the test harness and the precompiled
    # engines it links target the same GPUs.
    set_property(TARGET ${exec_name} PROPERTY CUDA_ARCHITECTURES "${STRINGZILLA_CUDA_ARCHS}")
    target_link_libraries(${exec_name} PRIVATE stringzilla_header)
    add_test(NAME ${exec_name} COMMAND ${exec_name})
endfunction ()

# Tests and benchmarks are defined after the libraries (below), so the StringZillas consumers can link the precompiled
# static archives instead of recompiling the engines header-only.

# Define our libraries, first the header-only version.
#
# Published as `stringzilla::header`: the namespace already says "stringzilla". The target keeps the long name, since a
# bare `header` would be a poor global one, and `EXPORT_NAME` advertises the short spelling to installed consumers.
add_library(stringzilla_header INTERFACE)
add_library(${PROJECT_NAME}::header ALIAS stringzilla_header)
add_library(${PROJECT_NAME}::stringzilla_header ALIAS stringzilla_header)
set_target_properties(stringzilla_header PROPERTIES EXPORT_NAME header)
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
    c/stringzilla/cipher.c
    c/stringzilla/find.c
    c/stringzilla/sort.c
    c/stringzilla/intersect.c
    c/stringzilla/utf8_norm.c
    c/stringzilla/utf8_runes.c
    c/stringzilla/utf8_tokens.c
    c/stringzilla/utf8_wordbreaks.c
    c/stringzilla/utf8_graphemes.c
    c/stringzilla/utf8_sentences.c
    c/stringzilla/utf8_linebreaks.c
    c/stringzilla/utf8_uncased_fold.c
    c/stringzilla/utf8_uncased.c
)

# Helper function used for `stringzilla_shared` and `stringzilla_bare` targets
function (define_stringzilla_shared target public_name)
    add_library(${target} SHARED ${STRINGZILLA_SHIM_SOURCES})

    # `stringzilla_shared` is published as `stringzilla::shared`. Both spellings resolve, in-tree and installed.
    add_library(${PROJECT_NAME}::${public_name} ALIAS ${target})
    add_library(${PROJECT_NAME}::${target} ALIAS ${target})
    set_target_properties(${target} PROPERTIES EXPORT_NAME ${public_name})

    # `SOVERSION` carries the major alone - the only break the C ABI admits - so it tracks the project rather than
    # sitting at a literal that stopped matching four majors ago.
    set_target_properties(
        ${target}
        PROPERTIES VERSION ${PROJECT_VERSION}
                   SOVERSION ${PROJECT_VERSION_MAJOR}
                   POSITION_INDEPENDENT_CODE ON
    )

    # Set the baseline architecture; the dynamic-dispatch core selects each SIMD capability at runtime.
    set_baseline_architecture_flags(${target} "${CMAKE_CXX_COMPILER_ID}")

    # Enable every capability the toolchain can compile; the runtime dispatch table masks what the CPU lacks.
    set_architecture_simd_definitions(${target} COMPILE)

endfunction ()

if (STRINGZILLA_BUILD_SHARED)

    define_stringzilla_shared(stringzilla_shared shared)
    target_compile_definitions(stringzilla_shared PRIVATE "SZ_AVOID_LIBC=0")
    target_compile_definitions(stringzilla_shared PRIVATE "SZ_OVERRIDE_LIBC=1")
    target_include_directories(
        stringzilla_shared PUBLIC $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
                                  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    )

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
        define_stringzilla_shared(stringzilla_bare bare)
        target_compile_definitions(stringzilla_bare PRIVATE "SZ_AVOID_LIBC=1")
        target_compile_definitions(stringzilla_bare PRIVATE "SZ_OVERRIDE_LIBC=1")
        target_include_directories(
            stringzilla_bare PUBLIC $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
                                    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
        )

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

# Definitions every StringZillas translation unit shares.
set(STRINGZILLAS_COMMON_DEFINITIONS "SZ_DYNAMIC_DISPATCH=1" "SZ_AVOID_LIBC=0"
                                    "$<IF:$<CONFIG:Debug>,SZ_DEBUG=1,SZ_DEBUG=0>"
)

# Helper function used for `stringzillas_cpus_shared`, `stringzillas_cuda_shared`, and `stringzillas_rocm_shared`
# targets
function (define_stringzillas_shared base public_base)
    # Compile the ARGN translation units once into `<base>_objects`, then expose `<base>_shared` and
    # `<base>_static` variants that reuse those objects. Tests and benchmarks link the STATIC variant and
    # inherit `SZ_DYNAMIC_DISPATCH=1` (set INTERFACE below), so the heavy precompiled engine instantiations
    # are linked rather than recompiled. The objects are linked PRIVATE into each variant (baked into the
    # `.so`/`.a`) to avoid the OBJECT-library transitive duplicate-symbol gotcha; consumer-facing usage
    # requirements are declared explicitly on the variants' INTERFACE.
    set(objects "${base}_objects")
    set(target "${base}_shared")
    set(static "${base}_static")

    add_library(${objects} OBJECT ${ARGN})
    add_library(${target} SHARED)
    add_library(${static} STATIC)

    # StringZillas is its own installed package, so its targets live under `stringzillas::`. The `_shared`/`_static`
    # suffix stays on the leaf: which one a consumer links is a real choice, not noise.
    add_library(stringzillas::${public_base}_shared ALIAS ${target})
    add_library(stringzillas::${public_base}_static ALIAS ${static})
    add_library(${PROJECT_NAME}::${target} ALIAS ${target})
    set_target_properties(${target} PROPERTIES EXPORT_NAME ${public_base}_shared)
    set_target_properties(${static} PROPERTIES EXPORT_NAME ${public_base}_static)

    target_link_libraries(${target} PRIVATE ${objects})
    target_link_libraries(${static} PRIVATE ${objects})

    set_target_properties(${objects} ${target} ${static} PROPERTIES POSITION_INDEPENDENT_CODE ON)
    set_target_properties(${objects} PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON)
    set_target_properties(${target} PROPERTIES VERSION ${PROJECT_VERSION} SOVERSION ${PROJECT_VERSION_MAJOR})

    # Compile settings for the library's own translation units (the OBJECT compilation). A future ROCm
    # backend declares its own `SZ_USE_*` pair here.
    target_include_directories(${objects} PRIVATE include)
    target_link_libraries(${objects} PRIVATE forkunion::header)
    target_compile_definitions(${objects} PRIVATE ${STRINGZILLAS_COMMON_DEFINITIONS} "SZ_USE_CUDA=0" "SZ_USE_ROCM=0")
    target_compile_options(
        ${objects} PRIVATE "$<$<CXX_COMPILER_ID:GNU,Clang>:-O2;-fPIC>" "$<$<CXX_COMPILER_ID:MSVC>:/O2>"
    )
    set_architecture_simd_definitions(${objects} COMPILE)
    set_baseline_architecture_flags(${objects} "${CMAKE_CXX_COMPILER_ID}")

    # Usage requirements seen by consumers (tests and benchmarks linking the library). Defining `SZ_DYNAMIC_DISPATCH=1`
    # makes a consumer extern, rather than re-instantiate, the engines; `forkunion::static` brings the ForkUnion
    # headers and its compiled runtime to both the library and its consumers. The consumer still picks its own ISA
    # via `set_compiler_flags`.
    foreach (variant ${target} ${static})
        target_include_directories(
            ${variant} PUBLIC $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
                              $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
        )
        target_link_libraries(${variant} PUBLIC forkunion::static)
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
function (define_stringzillas_cuda_objects name gpu_archs)
    add_library(${name} OBJECT ${ARGN})
    set_source_files_properties(${ARGN} TARGET_DIRECTORY ${name} PROPERTIES LANGUAGE CUDA)
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
    target_include_directories(${name} PRIVATE include)
    target_link_libraries(${name} PRIVATE forkunion::header)
    target_include_directories(
        ${name} SYSTEM PRIVATE ${CUDAToolkit_INCLUDE_DIRS}
                               $<$<BOOL:${STRINGZILLA_CCCL_INCLUDE_DIR}>:${STRINGZILLA_CCCL_INCLUDE_DIR}>
    )
    target_compile_definitions(${name} PRIVATE ${STRINGZILLAS_COMMON_DEFINITIONS} "SZ_USE_CUDA=1" "SZ_USE_ROCM=0")
    target_compile_options(${name} PRIVATE "--expt-relaxed-constexpr" "-Xfatbin=--compress-all")
    set_architecture_simd_definitions(${name} COMPILE)
    set_baseline_architecture_flags(${name} "${CMAKE_CUDA_COMPILER_ID}")
endfunction ()

# StringZillas C-API entry units, compiled once per library - as C++ into the CPU one, as CUDA into the GPU one.
set(STRINGZILLAS_API_CPP_SOURCES
    c/stringzillas/runtime.cpp c/stringzillas/levenshtein.cpp c/stringzillas/needleman_wunsch.cpp
    c/stringzillas/smith_waterman.cpp c/stringzillas/fingerprints.cpp
)
set(STRINGZILLAS_API_CU_SOURCES
    c/stringzillas/runtime.cu c/stringzillas/levenshtein.cu c/stringzillas/needleman_wunsch.cu
    c/stringzillas/smith_waterman.cu c/stringzillas/fingerprints.cu
)
# Per-ISA CPU instantiation units, host C++ in every library; off-platform files compile to empty objects.
set(STRINGZILLAS_CPUS_SOURCES
    c/stringzillas/levenshtein_serial.cpp
    c/stringzillas/levenshtein_icelake.cpp
    c/stringzillas/levenshtein_haswell.cpp
    c/stringzillas/levenshtein_neon.cpp
    c/stringzillas/levenshtein_rvv.cpp
    c/stringzillas/needleman_wunsch_serial.cpp
    c/stringzillas/needleman_wunsch_icelake.cpp
    c/stringzillas/needleman_wunsch_haswell.cpp
    c/stringzillas/needleman_wunsch_neon.cpp
    c/stringzillas/needleman_wunsch_rvv.cpp
    c/stringzillas/smith_waterman_serial.cpp
    c/stringzillas/smith_waterman_icelake.cpp
    c/stringzillas/smith_waterman_haswell.cpp
    c/stringzillas/smith_waterman_neon.cpp
    c/stringzillas/smith_waterman_rvv.cpp
)
# Per-tier GPU instantiation units, grouped by architecture floor: Hopper DPX needs sm_90, the rest run
# from the base set.
set(STRINGZILLAS_CUDA_SOURCES c/stringzillas/levenshtein_cuda.cu c/stringzillas/needleman_wunsch_cuda.cu
                              c/stringzillas/smith_waterman_cuda.cu
)
set(STRINGZILLAS_KEPLER_SOURCES c/stringzillas/levenshtein_kepler.cu)
set(STRINGZILLAS_HOPPER_SOURCES c/stringzillas/levenshtein_hopper.cu c/stringzillas/needleman_wunsch_hopper.cu
                                c/stringzillas/smith_waterman_hopper.cu
)

# The per-ISA CPU instantiation units stay host C++ even inside the CUDA library: NVCC's frontend cannot
# parse the CPU SIMD headers.
function (define_stringzillas_cpus_objects name)
    add_library(${name} OBJECT ${ARGN})
    set_target_properties(
        ${name}
        PROPERTIES POSITION_INDEPENDENT_CODE ON
                   CXX_STANDARD 20
                   CXX_STANDARD_REQUIRED ON
    )
    target_include_directories(${name} PRIVATE include)
    target_link_libraries(${name} PRIVATE forkunion::header)
    target_compile_definitions(${name} PRIVATE ${STRINGZILLAS_COMMON_DEFINITIONS} "SZ_USE_CUDA=1" "SZ_USE_ROCM=0")
    set_architecture_simd_definitions(${name} COMPILE)
    set_baseline_architecture_flags(${name} "${CMAKE_CXX_COMPILER_ID}")
endfunction ()

# Assemble one StringZillas CUDA library variant from four groups of object files, one per (compiler,
# architecture-set) pair; the static and shared variants pass different architecture sets.
function (
    define_stringzillas_cuda_library
    target
    public_name
    kind
    cuda_archs
    kepler_archs
    hopper_archs
)
    define_stringzillas_cuda_objects(
        ${target}_cuda "${cuda_archs}" ${STRINGZILLAS_API_CU_SOURCES} ${STRINGZILLAS_CUDA_SOURCES}
    )
    define_stringzillas_cuda_objects(${target}_kepler "${kepler_archs}" ${STRINGZILLAS_KEPLER_SOURCES})
    define_stringzillas_cuda_objects(${target}_hopper "${hopper_archs}" ${STRINGZILLAS_HOPPER_SOURCES})
    define_stringzillas_cpus_objects(${target}_cpus ${STRINGZILLAS_CPUS_SOURCES})

    add_library(${target} ${kind})
    target_link_libraries(${target} PRIVATE ${target}_cuda ${target}_kepler ${target}_hopper ${target}_cpus)
    set_target_properties(${target} PROPERTIES POSITION_INDEPENDENT_CODE ON LINKER_LANGUAGE CUDA)

    # Same `stringzillas::` publishing as the CPU variants; the static archive needs a name too.
    add_library(stringzillas::${public_name} ALIAS ${target})
    set_target_properties(${target} PROPERTIES EXPORT_NAME ${public_name})
    if (kind STREQUAL "SHARED")
        add_library(${PROJECT_NAME}::${target} ALIAS ${target})
        set_target_properties(${target} PROPERTIES VERSION ${PROJECT_VERSION} SOVERSION ${PROJECT_VERSION_MAJOR})
    endif ()

    # Consumers (tests/benches) extern the engines and link cudart themselves; the static archive does not resolve
    # cudart, so expose it (plus the dynamic-dispatch macro and headers) on the variant's interface. The include path is
    # split like the CPU variants': a bare source path would be baked into the export, pointing at this build tree.
    target_include_directories(
        ${target} PUBLIC $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
                         $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    )
    target_compile_definitions(${target} INTERFACE "SZ_DYNAMIC_DISPATCH=1")
    target_link_libraries(${target} PUBLIC forkunion::static CUDA::cudart Threads::Threads)
endfunction ()

if (STRINGZILLAS_BUILD_SHARED)

    # Define StringZillas CPU shared + static libraries (sharing one OBJECT compilation)
    define_stringzillas_shared(stringzillas_cpus cpus ${STRINGZILLAS_API_CPP_SOURCES} ${STRINGZILLAS_CPUS_SOURCES})
endif ()

# Define StringZillas CUDA shared + static libraries (only if CUDA is available). The static library is for local
# testing and benchmarking, so it ships real SASS only and runs without any load-time JIT. The shared library is for
# distribution, so on top of that real SASS it also packages forward-compatible PTX (the `-virtual` entries) that the
# driver JITs on newer GPUs. The base tier executes natively on both supported generations; the Kepler tier is
# superseded by Hopper on sm_90, so it ships sm_80 SASS only; the Hopper DPX tier exists only from sm_90.
if (STRINGZILLAS_BUILD_SHARED AND STRINGZILLA_BUILD_CUDA)
    find_package(Threads REQUIRED)
    define_stringzillas_cuda_library(
        stringzillas_cuda_static cuda_static STATIC "${STRINGZILLA_CUDA_ARCHS}" "80-real" "90-real"
    )
    define_stringzillas_cuda_library(
        stringzillas_cuda_shared cuda_shared SHARED "80-real;90-real;90-virtual" "80-real;80-virtual"
        "90-real;90-virtual"
    )
endif ()

# Tests and benchmarks come after the libraries so the StringZillas consumers can link the precompiled static archives
# (which carry the engine instantiations and export SZ_DYNAMIC_DISPATCH=1 on their interface) instead of recompiling the
# engines header-only.
if (STRINGZILLA_BUILD_BENCHMARK)
    define_launcher(stringzilla_bench_find_cpp20 20 "${STRINGZILLA_TARGET_ARCH}" bench/find.cpp)
    define_launcher(stringzilla_bench_sequence_cpp20 20 "${STRINGZILLA_TARGET_ARCH}" bench/sequence.cpp)
    define_launcher(stringzilla_bench_token_cpp20 20 "${STRINGZILLA_TARGET_ARCH}" bench/token.cpp)
    define_launcher(stringzilla_bench_utf8_uncased_cpp20 20 "${STRINGZILLA_TARGET_ARCH}" bench/utf8_uncased.cpp)
    define_launcher(stringzilla_bench_utf8_traverse_cpp20 20 "${STRINGZILLA_TARGET_ARCH}" bench/utf8_traverse.cpp)
    define_launcher(stringzilla_bench_utf8_scan_cpp20 20 "${STRINGZILLA_TARGET_ARCH}" bench/utf8_scan.cpp)
    define_launcher(stringzilla_bench_utf8_segment_cpp20 20 "${STRINGZILLA_TARGET_ARCH}" bench/utf8_segment.cpp)
    define_launcher(stringzilla_bench_utf8_norm_cpp20 20 "${STRINGZILLA_TARGET_ARCH}" bench/utf8_norm.cpp)
    define_launcher(stringzilla_bench_container_cpp20 20 "${STRINGZILLA_TARGET_ARCH}" bench/container.cpp)
    define_launcher(stringzilla_bench_memory_cpp20 20 "${STRINGZILLA_TARGET_ARCH}" bench/memory.cpp)
    define_launcher(stringzilla_bench_cipher_cpp20 20 "${STRINGZILLA_TARGET_ARCH}" bench/cipher.cpp)

    # Parallel benchmarks link the precompiled static libraries when those are built.
    define_launcher(stringzillas_bench_similarities_cpp20 20 "${STRINGZILLA_TARGET_ARCH}" bench/similarities.cpp)
    define_launcher(stringzillas_bench_fingerprints_cpp20 20 "${STRINGZILLA_TARGET_ARCH}" bench/fingerprints.cpp)
    if (STRINGZILLAS_BUILD_SHARED)
        target_link_libraries(stringzillas_bench_similarities_cpp20 PRIVATE stringzillas_cpus_static)
        target_link_libraries(stringzillas_bench_fingerprints_cpp20 PRIVATE stringzillas_cpus_static)
    endif ()
    if (STRINGZILLA_BUILD_CUDA)
        define_gpu_launcher(stringzillas_bench_similarities_cu20 20 "${STRINGZILLA_TARGET_ARCH}" bench/similarities.cu)
        define_gpu_launcher(stringzillas_bench_fingerprints_cu20 20 "${STRINGZILLA_TARGET_ARCH}" bench/fingerprints.cu)
        if (STRINGZILLAS_BUILD_SHARED)
            target_link_libraries(stringzillas_bench_similarities_cu20 PRIVATE stringzillas_cuda_static)
            target_link_libraries(stringzillas_bench_fingerprints_cu20 PRIVATE stringzillas_cuda_static)
        endif ()
        # The host object pulls in `std::log` etc.; unlike the host-compiler driver used for the `cpp20` benches, nvcc's
        # final link does not implicitly add libm. Link it only where it exists as a separate library (Linux/glibc); on
        # macOS/MSVC the math symbols live in the C runtime.
        find_library(STRINGZILLA_LIBM m)
        if (STRINGZILLA_LIBM)
            target_link_libraries(stringzillas_bench_similarities_cu20 PRIVATE ${STRINGZILLA_LIBM})
            target_link_libraries(stringzillas_bench_fingerprints_cu20 PRIVATE ${STRINGZILLA_LIBM})
        endif ()
    endif ()
endif ()

if (STRINGZILLA_BUILD_TEST)
    # The serial test suite is split across per-domain translation units that compile into the same binary, so the
    # standards-matrix builds below parallelize and rebuild incrementally instead of recompiling one 5K-line file.
    # `define_launcher`'s `target_sources` accepts this list verbatim.
    set(STRINGZILLA_TEST_SOURCES
        test/stringzilla.cpp
        test/hash.cpp
        test/cipher.cpp
        test/utf8_runes.cpp
        test/utf8_tokens.cpp
        test/utf8_wordbreaks.cpp
        test/utf8_graphemes.cpp
        test/utf8_sentences.cpp
        test/utf8_linebreaks.cpp
        test/utf8_norm.cpp
        test/uncased.cpp
        test/string.cpp
        test/find.cpp
        test/sort.cpp
    )

    # Make sure compilation passes for different C++ standards; MSVC only supports C++11 and newer.
    define_launcher(stringzilla_test_cpp11 11 "${STRINGZILLA_TARGET_ARCH}" ${STRINGZILLA_TEST_SOURCES})
    define_launcher(stringzilla_test_cpp14 14 "${STRINGZILLA_TARGET_ARCH}" ${STRINGZILLA_TEST_SOURCES})
    define_launcher(stringzilla_test_cpp17 17 "${STRINGZILLA_TARGET_ARCH}" ${STRINGZILLA_TEST_SOURCES})
    define_launcher(stringzilla_test_cpp20 20 "${STRINGZILLA_TARGET_ARCH}" ${STRINGZILLA_TEST_SOURCES})

    # Test parallel algorithms separately; link the static archive when the libraries are built. StringZillas requires
    # C++20 as the minimum language standard for concepts, designated initializers, and templated lambdas. We also test
    # C++23 to validate forward compatibility, but only where the toolchain actually advertises support for it — older
    # MSVC and nvcc releases lack a C++23 dialect, and forcing it there silently decays to an earlier standard or hard
    # errors at configure time.
    define_launcher(stringzillas_test_cpp20 20 "${STRINGZILLA_TARGET_ARCH}" test/stringzillas.cpp)
    if (STRINGZILLAS_BUILD_SHARED)
        target_link_libraries(stringzillas_test_cpp20 PRIVATE stringzillas_cpus_static)
    endif ()
    if ("cxx_std_23" IN_LIST CMAKE_CXX_COMPILE_FEATURES)
        define_launcher(stringzillas_test_cpp23 23 "${STRINGZILLA_TARGET_ARCH}" test/stringzillas.cpp)
        if (STRINGZILLAS_BUILD_SHARED)
            target_link_libraries(stringzillas_test_cpp23 PRIVATE stringzillas_cpus_static)
        endif ()
    endif ()

    if (STRINGZILLA_BUILD_CUDA)
        define_gpu_launcher(stringzillas_test_cu20 20 "${STRINGZILLA_TARGET_ARCH}" test/stringzillas.cu)
        if (STRINGZILLAS_BUILD_SHARED)
            target_link_libraries(stringzillas_test_cu20 PRIVATE stringzillas_cuda_static)
        endif ()
        if ("cuda_std_23" IN_LIST CMAKE_CUDA_COMPILE_FEATURES)
            define_gpu_launcher(stringzillas_test_cu23 23 "${STRINGZILLA_TARGET_ARCH}" test/stringzillas.cu)
            if (STRINGZILLAS_BUILD_SHARED)
                target_link_libraries(stringzillas_test_cu23 PRIVATE stringzillas_cuda_static)
            endif ()
        endif ()
    endif ()
endif ()

# TODO: Define a `stringzillas_rocm` library via `define_stringzillas_shared` when ROCm support is added.

if (STRINGZILLA_INSTALL)
    include(CMakePackageConfigHelpers)

    # Two packages, not one. `find_package(stringzilla)` always resolves; `find_package(stringzillas)` exists only where
    # the parallel engines were built, so a header-only prefix advertises no empty package. Every installed library
    # belongs to an `EXPORT` set - a `.so` with no imported target leaves consumers hand-rolling `find_library`.
    #
    # Destinations come from the `GNUInstallDirs` defaults, so each call names only its target and export set.
    install(TARGETS stringzilla_header EXPORT stringzillaTargets)
    if (TARGET stringzilla_shared)
        install(TARGETS stringzilla_shared EXPORT stringzillaTargets)
    endif ()
    if (TARGET stringzilla_bare)
        install(TARGETS stringzilla_bare EXPORT stringzillaTargets)
    endif ()

    install(
        EXPORT stringzillaTargets
        FILE stringzillaTargets.cmake
        NAMESPACE stringzilla::
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/stringzilla
    )
    write_basic_package_version_file(
        "${CMAKE_CURRENT_BINARY_DIR}/stringzillaConfigVersion.cmake"
        VERSION ${PROJECT_VERSION}
        COMPATIBILITY SameMajorVersion
    )
    configure_package_config_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/cmake/stringzillaConfig.cmake.in"
        "${CMAKE_CURRENT_BINARY_DIR}/stringzillaConfig.cmake"
        INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/stringzilla
    )
    install(FILES "${CMAKE_CURRENT_BINARY_DIR}/stringzillaConfig.cmake"
                  "${CMAKE_CURRENT_BINARY_DIR}/stringzillaConfigVersion.cmake"
            DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/stringzilla
    )

    # Shared variants only: the static archives consume the `_objects` OBJECT libraries, and CMake will not export a
    # target whose object library is not exported too - that would ship every `.o`. The `_static` aliases remain for
    # `add_subdirectory` consumers, whose objects are in the same build tree.
    if (TARGET stringzillas_cpus_shared OR TARGET stringzillas_cuda_shared)
        if (TARGET stringzillas_cpus_shared)
            install(TARGETS stringzillas_cpus_shared EXPORT stringzillasTargets)
        endif ()
        if (TARGET stringzillas_cuda_shared)
            install(TARGETS stringzillas_cuda_shared EXPORT stringzillasTargets)
        endif ()

        install(
            EXPORT stringzillasTargets
            FILE stringzillasTargets.cmake
            NAMESPACE stringzillas::
            DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/stringzillas
        )
        write_basic_package_version_file(
            "${CMAKE_CURRENT_BINARY_DIR}/stringzillasConfigVersion.cmake"
            VERSION ${PROJECT_VERSION}
            COMPATIBILITY SameMajorVersion
        )
        configure_package_config_file(
            "${CMAKE_CURRENT_SOURCE_DIR}/cmake/stringzillasConfig.cmake.in"
            "${CMAKE_CURRENT_BINARY_DIR}/stringzillasConfig.cmake"
            INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/stringzillas
        )
        install(FILES "${CMAKE_CURRENT_BINARY_DIR}/stringzillasConfig.cmake"
                      "${CMAKE_CURRENT_BINARY_DIR}/stringzillasConfigVersion.cmake"
                DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/stringzillas
        )
    endif ()

    # SwiftPM reads `module.modulemap` from the source tree, via `publicHeadersPath` in `Package.swift`. At an installed
    # include root it is a hazard: Clang treats whichever one it finds there as authoritative for the whole prefix.
    install(
        DIRECTORY ${STRINGZILLA_INCLUDE_BUILD_DIR}
        DESTINATION ${STRINGZILLA_INCLUDE_INSTALL_DIR}
        PATTERN "module.modulemap" EXCLUDE
    )
    install(DIRECTORY c/ DESTINATION ${CMAKE_INSTALL_DATADIR}/${PROJECT_NAME}/src)

    # A `.pc` for the consumers CMake never reaches - cgo among them. Spelling the prefix against `${pcfiledir}` keeps it
    # correct after `cmake --install --prefix` moves the tree. `GNUInstallDirs` returns a relative path unless a distro
    # forced an absolute one, and only the former may follow a prefix. No `Libs.private`: the shared library is C and
    # links nothing but libc, so a static caller inherits no closure to name.
    if (TARGET stringzilla_shared)
        file(RELATIVE_PATH STRINGZILLA_PC_PREFIX "${CMAKE_INSTALL_FULL_LIBDIR}/pkgconfig" "${CMAKE_INSTALL_PREFIX}")
        string(REGEX REPLACE "/+$" "" STRINGZILLA_PC_PREFIX "${STRINGZILLA_PC_PREFIX}")
        set(STRINGZILLA_PC_PREFIX "\${pcfiledir}/${STRINGZILLA_PC_PREFIX}")

        if (IS_ABSOLUTE "${CMAKE_INSTALL_LIBDIR}")
            set(STRINGZILLA_PC_LIBDIR "${CMAKE_INSTALL_LIBDIR}")
        else ()
            set(STRINGZILLA_PC_LIBDIR "\${exec_prefix}/${CMAKE_INSTALL_LIBDIR}")
        endif ()
        if (IS_ABSOLUTE "${CMAKE_INSTALL_INCLUDEDIR}")
            set(STRINGZILLA_PC_INCLUDEDIR "${CMAKE_INSTALL_INCLUDEDIR}")
        else ()
            set(STRINGZILLA_PC_INCLUDEDIR "\${prefix}/${CMAKE_INSTALL_INCLUDEDIR}")
        endif ()

        configure_file(cmake/stringzilla.pc.in "${CMAKE_CURRENT_BINARY_DIR}/stringzilla.pc" @ONLY)
        install(FILES "${CMAKE_CURRENT_BINARY_DIR}/stringzilla.pc" DESTINATION ${CMAKE_INSTALL_LIBDIR}/pkgconfig)
    endif ()

endif ()
