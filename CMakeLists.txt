# StringZilla CMakeLists.txt
#
# This file defines several library build & installation targets:
#
# * stringzilla_header: A header-only library with the StringZilla C and C++ headers.
# * stringzilla_shared: A shared library with the StringZilla C and C++ headers and dynamic SIMD dispatch.
# * stringzilla_bare: A shared library with the StringZilla headers, but without linking the standard C library.
#
# Consumers link the same namespaced spellings whether pulled in with `add_subdirectory` or found with `find_package`:
#
# * stringzilla::header, stringzilla::shared, stringzilla::bare - from `find_package(stringzilla)`.
#
# Tests:
#
# * stringzilla_test_cpp20: The C++ interface, at its C++20 floor.
# * stringzilla_test_cu20: The CUDA backends of the core families, against the serial answers.
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
# * stringzilla_bench_levenshtein_cpp20: A benchmark for the `sz_levenshtein_*` edit-distance family.
# * stringzilla_bench_overlap_cpp20: A benchmark for the `sz_overlap_*` window-overlap family.
# * stringzilla_bench_levenshtein_cu20: The `sz_levenshtein_*` family on a CUDA GPU.
# * stringzilla_bench_overlap_cu20: The `sz_overlap_*` family on a CUDA GPU.
# * stringzilla_bench_substrings_cpp20: The `sz_substrings_*` family on a CPU.
# * stringzilla_bench_substrings_cu20: The `sz_substrings_*` family on a CUDA GPU.
#
# For higher-level language bindings separate build scripts are provided, native to each toolchain.
cmake_minimum_required(VERSION 3.21 FATAL_ERROR)
project(
    stringzilla
    VERSION 5.1.2
    LANGUAGES C CXX ASM
    DESCRIPTION "Search, hash, sort, and fuzzy-match strings faster via SWAR, SIMD, and GPGPU"
    HOMEPAGE_URL "https://github.com/ashvardanian/stringzilla"
)

set(CMAKE_C_STANDARD 99)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_C_EXTENSIONS OFF)
set(CMAKE_CXX_EXTENSIONS OFF)

message(STATUS "C Compiler ID: ${CMAKE_C_COMPILER_ID}")
message(STATUS "C Compiler Version: ${CMAKE_C_COMPILER_VERSION}")
message(STATUS "C Compiler: ${CMAKE_C_COMPILER}")
message(STATUS "C++ Compiler ID: ${CMAKE_CXX_COMPILER_ID}")
message(STATUS "C++ Compiler Version: ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "C++ Compiler: ${CMAKE_CXX_COMPILER}")

# Detect CUDA Support
# `check_language` caches the compiler it finds together with the host compiler it happened to pair with it,
# and that pairing then outranks a caller's `CMAKE_CUDA_HOST_COMPILER` - which is why the probe runs only to
# pick the default below. A caller who asks for CUDA outright gets the real detection from `enable_language`,
# whose own failure message says everything this probe would have.
set(STRINGZILLA_CAN_BUILD_CUDA OFF)
if (DEFINED STRINGZILLA_BUILD_CUDA)
    set(STRINGZILLA_CAN_BUILD_CUDA ${STRINGZILLA_BUILD_CUDA})
else ()
    include(CheckLanguage)
    check_language(CUDA)
    if (CMAKE_CUDA_COMPILER)
        set(STRINGZILLA_CAN_BUILD_CUDA ON)
        message(STATUS "CUDA compiler available")
    else ()
        message(STATUS "CUDA compiler not available")
    endif ()
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
    # WebAssembly is 32-bit; its SIMD tier is the toolchain's `-msimd128` / `-mrelaxed-simd`, which `types.h`
    # reads off the predefined macros, so no 64-bit flag and no probe verdict is set here.
    message(STATUS "Platform: WebAssembly (CMAKE_SYSTEM_PROCESSOR=${CMAKE_SYSTEM_PROCESSOR})")
else ()
    message(WARNING "Unknown CMAKE_SYSTEM_PROCESSOR: ${CMAKE_SYSTEM_PROCESSOR}")
endif ()

# One compile probe per tier under `probes/`, `SZ_USE_<TIER>=0/1`: the verdicts reach the units that admit tiers at
# runtime, never `stringzilla_header`, whose consumers derive each tier from their own flags in `types.h`.
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
    # Unknown architecture: no tier probes, but the shared module still loads to clear both verdict lists, so
    # configuration degrades to a serial-only build instead of failing.
    include(cmake/sz_isa_probe.cmake)
endif ()

# Determine if StringZilla is built as a sub-project (using `add_subdirectory`) or if it is the main project
set(STRINGZILLA_IS_MAIN_PROJECT ${PROJECT_IS_TOP_LEVEL})

# Enable the in-repo Git hooks. Git will not auto-run hooks shipped inside a clone (that would let any cloned repo
# execute code), so point `core.hooksPath` at `.githooks` here - configuring the build arms the `pre-commit` checks.
# Skipped when built as a subproject or without `.git` (tarball / export builds).
if (STRINGZILLA_IS_MAIN_PROJECT AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/.git")
    execute_process(
        COMMAND git config core.hooksPath .githooks
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
        RESULT_VARIABLE STRINGZILLA_HOOKS_STATUS
        OUTPUT_QUIET ERROR_QUIET
    )
    if (STRINGZILLA_HOOKS_STATUS EQUAL 0)
        message(STATUS "stringzilla: Git hooks enabled (core.hooksPath -> .githooks)")
    endif ()
endif ()

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
option(STRINGZILLA_BUILD_STATIC "Compile a static library with runtime dispatch" OFF)
option(STRINGZILLA_BUILD_CUDA "Build CUDA-accelerated targets" ${STRINGZILLA_DEFAULT_CUDA})
option(STRINGZILLA_USE_SANITIZERS "Enable AddressSanitizer and UndefinedBehaviorSanitizer in Debug builds" ON)

# Clang source-based coverage for the header-only library. Recipe (single-arch, native backends only): cmake -B
# build_cov -DSTRINGZILLA_BUILD_COVERAGE=1 -DSTRINGZILLA_USE_SANITIZERS=0 -DCMAKE_CXX_COMPILER=clang++ cmake --build
# build_cov --target stringzilla_test_cpp20 -j LLVM_PROFILE_FILE=/tmp/sz.profraw ./build_cov/stringzilla_test_cpp20
# llvm-profdata merge -sparse /tmp/sz.profraw -o /tmp/sz.profdata llvm-cov report ./build_cov/stringzilla_test_cpp20
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
    # Real SASS per tier, in lockstep with `_CUDA_ARCHES` in setup.py and build.rs. The base tier is the only
    # implementation for some ops even on sm_90, so it ships both generations; Kepler is superseded by Hopper there
    # and ships sm_80 alone; the DPX tier starts at sm_90 (plain sm_90, not the H100-locked sm_90a). The shared
    # library adds forward PTX on top of these; the static one, built for local testing, runs without any JIT.
    set(STRINGZILLA_CUDA_ARCHS "80-real;90-real"
        CACHE STRING "Real SASS tiers every CUDA target carries; override to match the machine at hand"
    )
    set(STRINGZILLA_KEPLER_ARCHS "80-real")
    set(STRINGZILLA_HOPPER_ARCHS "90-real")
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
    include(cmake/sz_cuda_probes.cmake)
    sz_cuda_probe_native_arch_()

    message(STATUS "CUDA support enabled")
    message(STATUS "CUDA Compiler: ${CMAKE_CUDA_COMPILER}")
    message(STATUS "CUDA Compiler ID: ${CMAKE_CUDA_COMPILER_ID}")
    message(STATUS "CUDA Toolkit Version: ${CUDAToolkit_VERSION}")
    message(STATUS "CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
endif ()

# Stamps the whole GPU switch family on one target. They are exclusive - a build carries NVIDIA kernels, AMD kernels,
# or neither - and every translation unit linked into one library has to agree on all three, so no site picks a subset
# and leans on the header's defaults for the rest. `SZ_USE_HOPPER` follows the architectures compiled rather than the
# toolkit version, since a newer nvcc asked for sm_80 alone emits no DPX, and it takes the library-wide set because a
# tier's narrower one would make the Kepler objects disagree with their siblings about which engines exist.
function (set_gpu_definitions name backend)
    set(uses_cuda 0)
    set(carries_hopper 0)
    set(uses_rocm 0)
    if (backend STREQUAL "cuda")
        set(uses_cuda 1)
        foreach (architecture IN LISTS STRINGZILLA_CUDA_ARCHS)
            string(REGEX MATCH "^[0-9]+" architecture_number "${architecture}")
            if (architecture_number AND NOT architecture_number LESS 90)
                set(carries_hopper 1)
            endif ()
        endforeach ()
    elseif (backend STREQUAL "rocm")
        set(uses_rocm 1)
    endif ()
    target_compile_definitions(
        ${name} PRIVATE "SZ_USE_CUDA=${uses_cuda}" "SZ_USE_HOPPER=${carries_hopper}" "SZ_USE_ROCM=${uses_rocm}"
    )
endfunction ()

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

# {fmt} formats the suites' assertion messages and reports. `FIND_PACKAGE_ARGS` lets a system copy satisfy it without
# touching the network, but only from the pinned version up, since older copies lack `fmt::println`.
if (STRINGZILLA_BUILD_TEST OR STRINGZILLA_BUILD_BENCHMARK)
    include(FetchContent)
    FetchContent_Declare(
        fmt
        GIT_REPOSITORY https://github.com/fmtlib/fmt.git
        GIT_TAG 12.2.0
        FIND_PACKAGE_ARGS 12.2.0
    )
    FetchContent_MakeAvailable(fmt)
endif ()

function (define_launcher exec_name cpp_standard target_arch)
    add_executable(${exec_name})
    target_sources(${exec_name} PRIVATE ${ARGN})
    target_include_directories(${exec_name} PRIVATE test bench)
    set_compiler_flags(${exec_name} ${cpp_standard} "${target_arch}" "${CMAKE_CXX_COMPILER_ID}")
    # Launchers dispatch at compile time, so a capability must be BOTH compilable by this toolchain AND runnable on
    # this machine — the intersection the RUN verdicts encode. When the caller pins an explicit `target_arch`, that
    # `-march` (plus `types.h` auto-detection) is the capability request, so the machine gate must not override it.
    if ("${target_arch}" STREQUAL "")
        set_architecture_simd_definitions(${exec_name} RUN)
    endif ()
    target_link_libraries(${exec_name} PRIVATE stringzilla_header fmt::fmt)
    add_test(NAME ${exec_name} COMMAND ${exec_name})
endfunction ()

function (define_gpu_launcher exec_name cuda_standard target_arch)
    add_executable(${exec_name})
    target_sources(${exec_name} PRIVATE ${ARGN})
    set_source_files_properties(${ARGN} TARGET_DIRECTORY ${exec_name} PROPERTIES LANGUAGE CUDA)
    target_include_directories(${exec_name} PRIVATE test bench)
    set_gpu_definitions(${exec_name} cuda)
    set_target_properties(${exec_name} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    target_include_directories(
        ${exec_name} PRIVATE ${CUDAToolkit_INCLUDE_DIRS}
                             $<$<BOOL:${STRINGZILLA_CCCL_INCLUDE_DIR}>:${STRINGZILLA_CCCL_INCLUDE_DIR}>
    )
    set_compiler_flags(${exec_name} ${cuda_standard} "${target_arch}" "${CMAKE_CUDA_COMPILER_ID}")
    target_link_libraries(${exec_name} PRIVATE CUDA::cudart CUDA::cuda_driver)
    # Match the CUDA library's base tier, so the test harness and the precompiled engines it links target the same GPUs.
    set_property(TARGET ${exec_name} PROPERTY CUDA_ARCHITECTURES "${STRINGZILLA_CUDA_ARCHS}")
    target_link_libraries(${exec_name} PRIVATE stringzilla_header fmt::fmt)
    add_test(NAME ${exec_name} COMMAND ${exec_name})
endfunction ()

# Tests and benchmarks are defined after the libraries (below), so the GPU launchers can link the precompiled static
# archive instead of recompiling the engines header-only.

# Define our libraries, first the header-only version.
#
# Published as `stringzilla::header`: the namespace already says "stringzilla". The target keeps the long name, since a
# bare `header` would be a poor global one, and `EXPORT_NAME` advertises the short spelling to installed consumers.
add_library(stringzilla_header INTERFACE)
add_library(${PROJECT_NAME}::header ALIAS stringzilla_header)
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
    c/stringzilla/levenshtein.c
    c/stringzilla/overlap.c
    c/stringzilla/substrings.c
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

# Helper function used for the `stringzilla_shared`, `stringzilla_bare`, and `stringzilla_static` targets
function (define_stringzilla_library target public_name library_type)
    add_library(${target} ${library_type} ${STRINGZILLA_SHIM_SOURCES})
    target_compile_definitions(${target} PRIVATE "SZ_DYNAMIC_DISPATCH=1")

    # `stringzilla_shared` is published as `stringzilla::shared`, in-tree and installed alike.
    add_library(${PROJECT_NAME}::${public_name} ALIAS ${target})
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

    # The shim units default `SZ_OVERRIDE_LIBC` to `SZ_AVOID_LIBC`, so this library exports no `memcpy` of its
    # own and interposes nothing. That is what lets anything link it - the tests, a consumer's program -
    # without ELF redirecting `libstdc++`'s internal calls into a dispatch table that is not up yet.
    # `stringzilla_bare` is the artifact that replaces LibC, and it still does.
    define_stringzilla_library(stringzilla_shared shared SHARED)
    target_compile_definitions(stringzilla_shared PRIVATE "SZ_AVOID_LIBC=0")
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
        define_stringzilla_library(stringzilla_bare bare SHARED)
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

# The dispatch core as an archive, for consumers that ship one self-contained artifact. Tests and benchmarks link it
# too: the cross-compiled runs are `-static` executables, which cannot take a `.so` at all.
if (STRINGZILLA_BUILD_STATIC
    OR STRINGZILLA_BUILD_TEST
    OR STRINGZILLA_BUILD_BENCHMARK
)
    define_stringzilla_library(stringzilla_static static STATIC)
    target_compile_definitions(stringzilla_static PRIVATE "SZ_AVOID_LIBC=0")
    target_include_directories(stringzilla_static PUBLIC $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>)
endif ()

# The same dispatch core, compiled by nvcc so its CUDA arms exist. `set_source_files_properties` retargets the very
# same `.c` files per target, so there is no second source list and no trampoline pair to keep in step - the shims
# already carry their `#if SZ_USE_CUDA` arms and only ever lacked a target that defines it. `runtime.c` has to be in
# here too: `sz_capabilities` ands a compile-time mask that carries `sz_cap_cuda_k` only when its own translation
# unit saw `SZ_USE_CUDA`, so a CPU-compiled runtime would report no device however the other shims were built.
function (define_stringzilla_cuda_library target public_name library_type gpu_archs)
    add_library(${target} ${library_type} ${STRINGZILLA_SHIM_SOURCES})
    set_source_files_properties(${STRINGZILLA_SHIM_SOURCES} TARGET_DIRECTORY ${target} PROPERTIES LANGUAGE CUDA)

    add_library(${PROJECT_NAME}::${public_name} ALIAS ${target})
    set_target_properties(${target} PROPERTIES EXPORT_NAME ${public_name})
    set_target_properties(
        ${target}
        PROPERTIES POSITION_INDEPENDENT_CODE ON
                   CUDA_STANDARD 20
                   CUDA_STANDARD_REQUIRED ON
                   CUDA_SEPARABLE_COMPILATION OFF
                   CUDA_ARCHITECTURES "${gpu_archs}"
    )
    if (library_type STREQUAL "SHARED")
        set_target_properties(${target} PROPERTIES VERSION ${PROJECT_VERSION} SOVERSION ${PROJECT_VERSION_MAJOR})
    endif ()

    target_include_directories(
        ${target} PUBLIC $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
                         $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    )
    target_include_directories(${target} SYSTEM PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
    target_compile_definitions(${target} PRIVATE "SZ_AVOID_LIBC=0" "SZ_DYNAMIC_DISPATCH=1")
    set_gpu_definitions(${target} cuda)

    # The device tiers reach the serial arithmetic through relaxed constexpr, so the scores are bit-identical.
    target_compile_options(${target} PRIVATE "--expt-relaxed-constexpr")
    set_architecture_simd_definitions(${target} COMPILE)
    set_baseline_architecture_flags(${target} "${CMAKE_CUDA_COMPILER_ID}")
    target_link_libraries(${target} PUBLIC CUDA::cudart CUDA::cuda_driver)
endfunction ()

if (STRINGZILLA_BUILD_CUDA)
    # Static carries real SASS alone, so a local run needs no load-time JIT; shared adds forward PTX for distribution.
    define_stringzilla_cuda_library(stringzilla_cuda_static cuda_static STATIC "${STRINGZILLA_CUDA_ARCHS}")
    define_stringzilla_cuda_library(
        stringzilla_cuda_shared cuda_shared SHARED "${STRINGZILLA_CUDA_ARCHS};90-virtual"
    )
endif ()

# Tests and benchmarks come after the libraries so they link the precompiled archives rather than recompiling.
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
    define_launcher(stringzilla_bench_levenshtein_cpp20 20 "${STRINGZILLA_TARGET_ARCH}" bench/levenshtein.cpp)
    define_launcher(stringzilla_bench_overlap_cpp20 20 "${STRINGZILLA_TARGET_ARCH}" bench/overlap.cpp)
    define_launcher(stringzilla_bench_substrings_cpp20 20 "${STRINGZILLA_TARGET_ARCH}" bench/substrings.cpp)

    if (STRINGZILLA_BUILD_CUDA)
        define_gpu_launcher(stringzilla_bench_overlap_cu20 20 "${STRINGZILLA_TARGET_ARCH}" bench/overlap.cu)
        define_gpu_launcher(stringzilla_bench_levenshtein_cu20 20 "${STRINGZILLA_TARGET_ARCH}" bench/levenshtein.cu)
        define_gpu_launcher(stringzilla_bench_substrings_cu20 20 "${STRINGZILLA_TARGET_ARCH}" bench/substrings.cu)
        # The CUDA-compiled core, so `sz_capabilities` reports the device and the dispatched verbs are its own.
        target_link_libraries(stringzilla_bench_overlap_cu20 PRIVATE stringzilla_cuda_static)
        target_link_libraries(stringzilla_bench_levenshtein_cu20 PRIVATE stringzilla_cuda_static)
        target_link_libraries(stringzilla_bench_substrings_cu20 PRIVATE stringzilla_cuda_static)
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
        test/levenshtein.cpp
        test/overlap.cpp
        test/substrings.cpp
        test/sort.cpp
    )

    define_launcher(stringzilla_test_cpp20 20 "${STRINGZILLA_TARGET_ARCH}" ${STRINGZILLA_TEST_SOURCES})

    if (STRINGZILLA_BUILD_CUDA)
        # The core library's own GPU backends: one launcher, one source per family beside its CPU test.
        define_gpu_launcher(
            stringzilla_test_cu20 20 "${STRINGZILLA_TARGET_ARCH}" test/stringzilla.cu test/overlap.cu
            test/levenshtein.cu test/substrings.cu
        )
        target_link_libraries(stringzilla_test_cu20 PRIVATE stringzilla_cuda_static)
    endif ()
endif ()


if (STRINGZILLA_INSTALL)
    include(CMakePackageConfigHelpers)

    # Every installed library belongs to an `EXPORT` set - a `.so` with no imported target leaves consumers
    # hand-rolling `find_library`.
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
