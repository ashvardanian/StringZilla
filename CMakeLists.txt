# StringZilla CMakeLists.txt
#
# This file defines several library build & installation targets:
#
# * stringzilla_header: A header-only library with the StringZilla C and C++ headers.
# * stringzilla_static: A static library with dynamic SIMD dispatch, and the GPU backends STRINGZILLA_BUILD_CUDA,
#   STRINGZILLA_BUILD_ROCM and STRINGZILLA_BUILD_METAL ask for, side by side.
# * stringzilla_shared: The same as a shared library.
# * stringzilla_python: The CPython extension over `stringzilla_static`, with STRINGZILLA_BUILD_PYTHON.
# * stringzilla_node: The Node addon over `stringzilla_static`, with STRINGZILLA_BUILD_NODE, which cmake-js sets.
# * stringzilla_swift: `stringzilla_static` packaged for SwiftPM, with STRINGZILLA_BUILD_SWIFT.
#
# Consumers link the same namespaced spellings whether pulled in with `add_subdirectory` or found with `find_package`:
#
# * stringzilla::header, stringzilla::static, stringzilla::shared.
#
# Tests:
#
# * stringzilla_test: The dispatch points, the C++ interface, and every kernel the static library defines.
# * stringzilla_shared_test: The dispatch points and GPU kernels, through the shared library.
# * stringzilla_cpu_header_test: The kernels compiled inline, header-only, and the dispatch-point stubs.
#
# Benchmarks:
#
# * stringzilla_bench: The dispatch points against the standard baselines, and every kernel the static library
#   defines against its serial twin.
# * stringzilla_cpu_header_bench: The public capability kernels compiled inline, header-only.
#
# Python, Node, and Swift build their artifacts through this file, while Rust's `build.rs` and Go's cgo link
# `stringzilla_static`; Java and C# load `stringzilla_shared`.
cmake_minimum_required(VERSION 3.21 FATAL_ERROR)

# Every binding reads its version from `VERSION`, which the release bumps.
file(STRINGS "${CMAKE_CURRENT_SOURCE_DIR}/VERSION" stringzilla_version_line_ LIMIT_COUNT 1)
# CMake versions are numeric, so the pre-release suffix of `VERSION` stays out of `PROJECT_VERSION`.
string(REGEX MATCH "^[0-9]+\\.[0-9]+\\.[0-9]+" stringzilla_version_ "${stringzilla_version_line_}")
project(
    stringzilla
    VERSION ${stringzilla_version_}
    LANGUAGES C CXX
    DESCRIPTION "Search, hash, sort, and fuzzy-match strings faster via SWAR, SIMD, and GPGPU"
    HOMEPAGE_URL "https://github.com/ashvardanian/StringZilla"
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

# On macOS `CMAKE_SYSTEM_PROCESSOR` names the host, while `CMAKE_OSX_ARCHITECTURES` names the slice `-arch` compiles,
# as when an Apple Silicon runner builds the x86-64 wheels.
if (APPLE AND CMAKE_OSX_ARCHITECTURES)
    list(LENGTH CMAKE_OSX_ARCHITECTURES stringzilla_osx_architectures_count_)
    if (NOT stringzilla_osx_architectures_count_ EQUAL 1)
        message(
            FATAL_ERROR "CMAKE_OSX_ARCHITECTURES='${CMAKE_OSX_ARCHITECTURES}' holds several slices; build one per slice"
        )
    endif ()
    set(CMAKE_SYSTEM_PROCESSOR "${CMAKE_OSX_ARCHITECTURES}")
endif ()

# Detect target architecture from `CMAKE_SYSTEM_PROCESSOR`... safer for cross-compilation!
if (CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|X86_64|AMD64|amd64")
    set(stringzilla_arch_x8664_ TRUE)
    message(STATUS "Platform: x86_64 (CMAKE_SYSTEM_PROCESSOR=${CMAKE_SYSTEM_PROCESSOR})")
elseif (CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|AARCH64|arm64|ARM64")
    set(stringzilla_arch_arm64_ TRUE)
    message(STATUS "Platform: ARM64 (CMAKE_SYSTEM_PROCESSOR=${CMAKE_SYSTEM_PROCESSOR})")
elseif (CMAKE_SYSTEM_PROCESSOR MATCHES "riscv64|RISCV64")
    set(stringzilla_arch_riscv64_ TRUE)
    message(STATUS "Platform: RISC-V 64 (CMAKE_SYSTEM_PROCESSOR=${CMAKE_SYSTEM_PROCESSOR})")
elseif (CMAKE_SYSTEM_PROCESSOR MATCHES "loongarch64|LOONGARCH64|loong64|LOONG64")
    # "loongarch64" is the GNU triple and `uname -m` spelling; "loong64" is the Debian/dpkg port name.
    set(stringzilla_arch_loongarch64_ TRUE)
    message(STATUS "Platform: LoongArch64 (CMAKE_SYSTEM_PROCESSOR=${CMAKE_SYSTEM_PROCESSOR})")
elseif (CMAKE_SYSTEM_PROCESSOR MATCHES "ppc64|powerpc64|PPC64|POWERPC64")
    # Matches both endiannesses ("ppc64le" contains "ppc64"); the kernels handle big-endian explicitly
    # (`find/powervsx.h`) and `STRINGZILLA_ARCH_BIG_ENDIAN_` is derived from `CMAKE_C_BYTE_ORDER` separately.
    set(stringzilla_arch_ppc64_ TRUE)
    message(STATUS "Platform: PowerPC64 (CMAKE_SYSTEM_PROCESSOR=${CMAKE_SYSTEM_PROCESSOR})")
elseif (CMAKE_SYSTEM_PROCESSOR MATCHES "^wasm(32|64)$")
    # A module's one SIMD capability is `STRINGZILLA_TARGET_ARCH`, which `types.h` reads back off the predefined
    # macros, so no probe verdict is set here.
    set(stringzilla_arch_wasm_ TRUE)
    message(STATUS "Platform: WebAssembly (CMAKE_SYSTEM_PROCESSOR=${CMAKE_SYSTEM_PROCESSOR})")
else ()
    message(WARNING "Unknown CMAKE_SYSTEM_PROCESSOR: ${CMAKE_SYSTEM_PROCESSOR}")
endif ()

# A `Generic` target, like Rust's `wasm32-unknown-unknown`, has no LibC for the libraries to reach.
if (CMAKE_SYSTEM_NAME STREQUAL "Generic")
    set(stringzilla_with_libc_ 0)
else ()
    set(stringzilla_with_libc_ 1)
endif ()

# One compile probe per capability under `probes/`, defining `STRINGZILLA_TARGET_<CAPABILITY>` as the toolchain
# compiles it through `stringzilla::cpu_capabilities_compiled`, and as 0 through
# `stringzilla_cpu_capabilities_disabled_`.
include(cmake/sz_isa_probe.cmake)

# Determine if StringZilla is built as a sub-project (using `add_subdirectory`) or if it is the main project
set(STRINGZILLA_IS_MAIN_PROJECT ${PROJECT_IS_TOP_LEVEL})

# Enable the in-repo Git hooks. Git will not auto-run hooks shipped inside a clone (that would let any cloned repo
# execute code), so point `core.hooksPath` at `.githooks` here - configuring the build arms the `pre-commit` checks.
# Skipped when built as a subproject or without `.git` (tarball / export builds).
if (STRINGZILLA_IS_MAIN_PROJECT AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/.git")
    execute_process(
        COMMAND git config core.hooksPath .githooks
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
        RESULT_VARIABLE stringzilla_hooks_status_
        OUTPUT_QUIET ERROR_QUIET
    )
    if (stringzilla_hooks_status_ EQUAL 0)
        message(STATUS "stringzilla: Git hooks enabled (core.hooksPath -> .githooks)")
    endif ()
endif ()

# Every binding's build runs this file as the top-level project, so the tests, the benchmarks and the GPUs stay off
# unless asked for, as the presets do: none of them fetches {fmt} or sprouts CUDA just because `nvcc` is on the PATH.
option(STRINGZILLA_INSTALL "Install CMake targets" OFF)
option(STRINGZILLA_BUILD_TEST "Compile a native unit test in C++" OFF)
option(STRINGZILLA_BUILD_BENCH "Compile a native benchmark in C++" OFF)
option(STRINGZILLA_BUILD_SHARED "Compile a dynamic library" ${STRINGZILLA_IS_MAIN_PROJECT})
option(STRINGZILLA_BUILD_PYTHON "Compile the CPython extension over the static library" OFF)
option(STRINGZILLA_BUILD_NODE "Compile the Node addon over the static library; cmake-js switches it on" OFF)
option(STRINGZILLA_BUILD_SWIFT "Package the static library for SwiftPM, as an XCFramework or an artifact bundle" OFF)
option(STRINGZILLA_BUILD_CUDA "Add CUDA to the libraries and to the tests and benchmarks built" OFF)
option(STRINGZILLA_BUILD_ROCM "Add ROCm for AMD GPUs to the libraries and to the tests and benchmarks built" OFF)
option(STRINGZILLA_BUILD_METAL "Add Metal for Apple GPUs to the libraries and to the tests and benchmarks built" OFF)
option(STRINGZILLA_BUILD_SANITIZERS "Enable AddressSanitizer and UndefinedBehaviorSanitizer in Debug builds" ON)

# Clang source-based coverage for the header-only library. Recipe (single-arch, native backends only): cmake -B
# build_cov -DSTRINGZILLA_BUILD_COVERAGE=1 -DSTRINGZILLA_BUILD_SANITIZERS=0 -DCMAKE_CXX_COMPILER=clang++ cmake --build
# build_cov --target stringzilla_test -j LLVM_PROFILE_FILE=/tmp/sz.profraw ./build_cov/stringzilla_test
# llvm-profdata merge -sparse /tmp/sz.profraw -o /tmp/sz.profdata llvm-cov report ./build_cov/stringzilla_test
# -instr-profile=/tmp/sz.profdata include/stringzilla/ `-march=native` compiles out the non-native backends, so a
# faithful picture needs `llvm-profdata merge` across per-ISA builds (native x86 + qemu-aarch64 for NEON/SVE + the RVV
# qemu toolchain).
option(STRINGZILLA_BUILD_COVERAGE "Instrument with Clang source-based coverage" OFF)
# Seeded from the environment at the first configure, so `cargo` and `cmake-js` honour `STRINGZILLA_TARGET_ARCH=native`.
set(STRINGZILLA_TARGET_ARCH
    "$ENV{STRINGZILLA_TARGET_ARCH}"
    CACHE STRING "Host-tuned `-march`, like native, or a WebAssembly SIMD capability: serial, v128 or v128relaxed"
)

# An engine validates a WebAssembly module whole, so every unit carries the module's one SIMD capability, whatever
# flags a binding passes, and `types.h` reads it back off the predefined macros.
if (stringzilla_arch_wasm_)
    if (STRINGZILLA_TARGET_ARCH STREQUAL "v128relaxed")
        add_compile_options(-msimd128 -mrelaxed-simd)
    elseif (STRINGZILLA_TARGET_ARCH STREQUAL "v128")
        add_compile_options(-msimd128)
    elseif (NOT STRINGZILLA_TARGET_ARCH STREQUAL "serial")
        message(
            FATAL_ERROR
                "STRINGZILLA_TARGET_ARCH on WebAssembly: serial, v128 or v128relaxed, not `${STRINGZILLA_TARGET_ARCH}`"
        )
    endif ()
endif ()

# Enable CUDA if requested
if (STRINGZILLA_BUILD_CUDA)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 20)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(CMAKE_CUDA_EXTENSIONS OFF)
    # Real SASS for every generation CUDA 13 targets, one family-wide cubin each for Blackwell (`f`) rather than the
    # arch-locked `a`, and forward PTX for the Hopper and Blackwell tiers, which a listed GPU never JITs.
    set(STRINGZILLA_CUDA_ARCHITECTURES
        "75-real;80-real;90-real;100f-real;120f-real;90-virtual;100-virtual"
        CACHE STRING "Codes every CUDA target carries; override to match the machine at hand"
    )
    set(CMAKE_CUDA_ARCHITECTURES ${STRINGZILLA_CUDA_ARCHITECTURES})
    set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
    find_package(CUDAToolkit REQUIRED)
    include(cmake/sz_cuda_probes.cmake)
    stringzilla_probe_cuda_native_arch()
    # "native", "all" and "all-major" stand for the lists CMake resolved them to, which the capability rows read.
    set(stringzilla_cuda_resolved_architectures_)
    foreach (code_ IN LISTS STRINGZILLA_CUDA_ARCHITECTURES)
        if (code_ STREQUAL "native")
            list(APPEND stringzilla_cuda_resolved_architectures_ ${CMAKE_CUDA_ARCHITECTURES_NATIVE})
        elseif (code_ STREQUAL "all")
            list(APPEND stringzilla_cuda_resolved_architectures_ ${CMAKE_CUDA_ARCHITECTURES_ALL})
        elseif (code_ STREQUAL "all-major")
            list(APPEND stringzilla_cuda_resolved_architectures_ ${CMAKE_CUDA_ARCHITECTURES_ALL_MAJOR})
        else ()
            list(APPEND stringzilla_cuda_resolved_architectures_ ${code_})
        endif ()
    endforeach ()
    # Clusters and their distributed shared memory: H100, H200, GH200, and every GPU since.
    stringzilla_add_gpu_capability(
        hopper STRINGZILLA_TARGET_HOPPER
        CUDA_ARCHITECTURES
            90
            90a
            100
            100a
            100f
            103
            103a
            103f
            110
            110a
            110f
            120
            120a
            120f
            121
            121a
            121f
    )
    # Cluster launch control, letting a block take over the tile of one not yet started: B200, B300, RTX 50, and every
    # GPU since.
    stringzilla_add_gpu_capability(
        blackwell STRINGZILLA_TARGET_BLACKWELL
        CUDA_ARCHITECTURES
            100
            100a
            100f
            103
            103a
            103f
            110
            110a
            110f
            120
            120a
            120f
            121
            121a
            121f
    )

    message(STATUS "CUDA support enabled")
    message(STATUS "CUDA Compiler: ${CMAKE_CUDA_COMPILER}")
    message(STATUS "CUDA Compiler ID: ${CMAKE_CUDA_COMPILER_ID}")
    message(STATUS "CUDA Toolkit Version: ${CUDAToolkit_VERSION}")
    message(STATUS "CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
endif ()

# HIP comes on before any target, as the libraries carry the ROCm unit. The package lands `hip_DIR` in the cache, where
# the Rust `rocm` feature finds the runtime to link.
if (STRINGZILLA_BUILD_ROCM)
    enable_language(HIP)
    set(STRINGZILLA_ROCM_ARCHITECTURES
        "gfx942;gfx950"
        CACHE STRING "AMD targets the ROCm unit and test carry; override to match the machine at hand"
    )
    set(CMAKE_HIP_ARCHITECTURES "${STRINGZILLA_ROCM_ARCHITECTURES}")
    find_package(hip REQUIRED)
endif ()

if (STRINGZILLA_BUILD_METAL AND NOT APPLE)
    message(FATAL_ERROR "STRINGZILLA_BUILD_METAL needs an Apple platform")
endif ()

# Configuration
include(GNUInstallDirs)
include(CheckCXXCompilerFlag)
set(stringzilla_include_build_dir_ "${PROJECT_SOURCE_DIR}/include/")
set(stringzilla_include_install_dir_ "${CMAKE_INSTALL_INCLUDEDIR}")

# CTest machinery only for top-level builds: a parent project decides its own testing story, and
# `add_test` calls below are inert without `enable_testing()`.
if (STRINGZILLA_IS_MAIN_PROJECT)
    include(CTest)
    enable_testing()
endif ()

if (MSVC)
    # Remove /RTC* from MSVC debug flags by default (it will be added back in the stringzilla_set_compiler_flags
    # function) Because /RTC* cannot be used without the crt so it needs to be disabled for that specific target
    string(REGEX REPLACE "/RTC[^ ]*" "" CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG}")
    string(REGEX REPLACE "/RTC[^ ]*" "" CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG}")
endif ()

include(cmake/sz_compiler_flags.cmake)

# {fmt} formats the suites' assertion messages and reports. `FIND_PACKAGE_ARGS` lets a system copy satisfy it without
# touching the network, but only from the pinned version up, since older copies lack `fmt::println`.
if (STRINGZILLA_BUILD_TEST OR STRINGZILLA_BUILD_BENCH)
    include(FetchContent)
    FetchContent_Declare(
        fmt
        GIT_REPOSITORY https://github.com/fmtlib/fmt.git
        GIT_TAG 12.2.0
        FIND_PACKAGE_ARGS 12.2.0
    )
    FetchContent_MakeAvailable(fmt)
endif ()

# Native vendor sources remain separate objects so one C++ runner can link both CUDA and HIP.
# MSVC tests use checked iterators, so fmt must compile with the same setting rather than link its unchecked library.
function (stringzilla_add_gpu_suite target_ language_ source_)
    add_library(${target_} OBJECT ${source_})
    stringzilla_set_compiler_flags(${target_} 20 "${STRINGZILLA_TARGET_ARCH}")
    set(format_library_ fmt::fmt)
    if (MSVC AND source_ MATCHES "^test/")
        target_compile_definitions(${target_} PRIVATE _ITERATOR_DEBUG_LEVEL=1)
        set(format_library_ fmt::fmt-header-only)
    endif ()
    set_target_properties(
        ${target_}
        PROPERTIES CUDA_ARCHITECTURES "${STRINGZILLA_CUDA_ARCHITECTURES}"
                   HIP_ARCHITECTURES "${STRINGZILLA_ROCM_ARCHITECTURES}"
                   HIP_STANDARD 20
    )
    if (MSVC AND language_ STREQUAL "CUDA")
        target_compile_definitions(${target_} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:FMT_UNICODE=0>)
    endif ()
    target_link_libraries(
        ${target_} PRIVATE stringzilla_cpu_capabilities_disabled_ stringzilla_cuda_capabilities_compiled_
                           stringzilla_static ${format_library_}
    )
endfunction ()

# One CPU test or benchmark linking `library_`, whose capability `verdicts_` declare every kernel the
# toolchain builds (COMPILE), as the harnesses skip those this CPU lacks, or turn them all off, leaving
# the dispatch points (SERIAL).
function (stringzilla_add_cpu_launcher executable_ library_ verdicts_)
    add_executable(${executable_} ${ARGN})
    stringzilla_set_compiler_flags(${executable_} 20 "${STRINGZILLA_TARGET_ARCH}")
    set(format_library_ fmt::fmt)
    if (MSVC AND executable_ MATCHES "_test$")
        target_compile_definitions(${executable_} PRIVATE _ITERATOR_DEBUG_LEVEL=1)
        set(format_library_ fmt::fmt-header-only)
    endif ()
    if (verdicts_ STREQUAL "SERIAL")
        target_link_libraries(${executable_} PRIVATE stringzilla_cpu_capabilities_disabled_)
    else ()
        stringzilla_set_architecture_simd_definitions(${executable_})
    endif ()
    target_link_libraries(${executable_} PRIVATE ${library_} ${format_library_})
endfunction ()

# Tests and benchmarks link the precompiled libraries below.

# Define our libraries, first the header-only version.
#
# Published as `stringzilla::header`: the namespace already says "stringzilla". The target keeps the long name, since a
# bare `header` would be a poor global one, and `EXPORT_NAME` advertises the short spelling to installed consumers.
add_library(stringzilla_header INTERFACE)
add_library(${PROJECT_NAME}::header ALIAS stringzilla_header)
set_target_properties(stringzilla_header PROPERTIES EXPORT_NAME header)
target_include_directories(
    stringzilla_header INTERFACE $<BUILD_INTERFACE:${stringzilla_include_build_dir_}>
                                 $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
# Consumers of the header target compile every kernel themselves, and link no dispatch points.
target_compile_definitions(stringzilla_header INTERFACE STRINGZILLA_HEADER_ONLY=1)
if (APPLE)
    set(stringzilla_embed_compiler_
        "$<AND:$<PLATFORM_ID:Darwin>,$<OR:$<COMPILE_LANG_AND_ID:C,Clang,AppleClang>,$<COMPILE_LANG_AND_ID:CXX,Clang,AppleClang>>>"
    )
    if (IS_ABSOLUTE "${CMAKE_INSTALL_INCLUDEDIR}")
        set(stringzilla_embed_install_dir_ "${CMAKE_INSTALL_INCLUDEDIR}")
    else ()
        set(stringzilla_embed_install_dir_ "$<INSTALL_PREFIX>/${CMAKE_INSTALL_INCLUDEDIR}")
    endif ()
    set(stringzilla_embed_options_
        "$<${stringzilla_embed_compiler_}:$<BUILD_INTERFACE:--embed-dir=${stringzilla_include_build_dir_}>>"
        "$<${stringzilla_embed_compiler_}:$<INSTALL_INTERFACE:--embed-dir=${stringzilla_embed_install_dir_}>>"
    )
    target_compile_options(stringzilla_header INTERFACE ${stringzilla_embed_options_})
endif ()

# The compiled core: `c/stringzilla.c` owns the cross-family exports (versions, capabilities, the kernel finder), each
# `c/target/<capability>.c` defines that capability's kernels once, and each `c/dispatch/<family>.c` holds its family's
# lists, dispatch points and finder. Globbed, so a new unit needs no entry; the Metal unit joins below, with its own
# frameworks, beside the CUDA and ROCm ones.
file(
    GLOB stringzilla_globbed_units_ CONFIGURE_DEPENDS
    RELATIVE ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/c/target/*.c ${PROJECT_SOURCE_DIR}/c/dispatch/*.c
)
set(stringzilla_shim_sources_ c/stringzilla.c)
foreach (unit_ IN LISTS stringzilla_globbed_units_)
    if (NOT unit_ STREQUAL "c/target/metal.c")
        list(APPEND stringzilla_shim_sources_ ${unit_})
    endif ()
endforeach ()

# Defines `stringzilla_shared` or `stringzilla_static` over the sources that follow.
function (stringzilla_add_library target_ public_name_ library_type_)
    add_library(${target_} ${library_type_} ${ARGN})

    # The library, unlike the headers, may use C11: `c/stringzilla.c` caches detection in a <stdatomic.h> atomic, which
    # MSVC gates behind a flag.
    set_target_properties(${target_} PROPERTIES C_STANDARD 11)
    target_compile_options(${target_} PRIVATE $<$<COMPILE_LANG_AND_ID:C,MSVC>:/experimental:c11atomics>)

    # The dispatch units share the private `c/dispatch.h`.
    target_include_directories(${target_} PRIVATE ${PROJECT_SOURCE_DIR}/c)

    # Only `STRINGZILLA_API` functions leave the library; Windows exports every symbol, as it spells no `__declspec`.
    set_target_properties(
        ${target_}
        PROPERTIES C_VISIBILITY_PRESET hidden
                   CXX_VISIBILITY_PRESET hidden
                   CUDA_VISIBILITY_PRESET hidden
                   WINDOWS_EXPORT_ALL_SYMBOLS ON
    )

    # `stringzilla_shared` is published as `stringzilla::shared`, in-tree and installed alike.
    add_library(${PROJECT_NAME}::${public_name_} ALIAS ${target_})
    set_target_properties(${target_} PROPERTIES EXPORT_NAME ${public_name_})

    # `SOVERSION` carries the major alone - the only break the C ABI admits - so it tracks the project rather than
    # sitting at a literal that stopped matching four majors ago.
    set_target_properties(
        ${target_}
        PROPERTIES VERSION ${PROJECT_VERSION}
                   SOVERSION ${PROJECT_VERSION_MAJOR}
                   POSITION_INDEPENDENT_CODE ON
    )

    # Set the baseline architecture; the dispatch points pick each SIMD capability by the mask each call passes.
    stringzilla_set_baseline_architecture_flags(${target_})

    # Enable every capability the toolchain can compile; `sz_capabilities_enabled_cpu` masks what the CPU lacks.
    stringzilla_set_architecture_simd_definitions(${target_})

endfunction ()

# The dispatch core as an archive, which the tests, the benchmarks, and the Go and Python bindings link - the
# cross-compiled test runs are `-static` executables, which cannot take a `.so` at all - and as a shared library. A
# parent project builds the archive only if it links it.
stringzilla_add_library(stringzilla_static static STATIC ${stringzilla_shim_sources_})
if (NOT STRINGZILLA_IS_MAIN_PROJECT)
    set_target_properties(stringzilla_static PROPERTIES EXCLUDE_FROM_ALL ON)
endif ()
if (STRINGZILLA_BUILD_SHARED)
    stringzilla_add_library(stringzilla_shared shared SHARED ${stringzilla_shim_sources_})
    set(stringzilla_libraries_ stringzilla_static stringzilla_shared)
else ()
    set(stringzilla_libraries_ stringzilla_static)
endif ()

foreach (library_ IN LISTS stringzilla_libraries_)
    target_compile_definitions(${library_} PRIVATE "STRINGZILLA_WITH_LIBC=${stringzilla_with_libc_}")
    target_include_directories(
        ${library_} PUBLIC $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
                           $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    )

    # Force aggressive inlining for SHA-256 functions to avoid stack overflow in Go CGO (-mno-red-zone) The SHA-256
    # block processor has ~200-350 instructions and must be inlined to avoid nested stack frames
    target_compile_options(
        ${library_}
        PRIVATE
            "$<$<COMPILE_LANG_AND_ID:C,GNU>:--param=max-inline-insns-single=2000;--param=max-inline-insns-auto=2000>"
            "$<$<COMPILE_LANG_AND_ID:C,GNU>:-finline-functions;-fno-inline-functions-called-once>"
            "$<$<COMPILE_LANG_AND_ID:C,Clang,AppleClang>:-mllvm;-inline-threshold=2000;-finline-functions>"
    )

    # MSVC ARM64: <intrin.h> pulls in <winnt.h>, which needs `_ARM64_`, and the Visual Studio generator scikit-build-core
    # picks leaves it undefined for C targets. Public, so the binding units that include the headers see it too.
    if (MSVC AND stringzilla_arch_arm64_)
        target_compile_definitions(${library_} PUBLIC _ARM64_)
    endif ()
endforeach ()

# Adds one GPU unit, like `c/target/cuda.cu`, to the archive and the shared library: compiled for `codes_` alone,
# without the CPU capabilities and with the GPU ones the build's codes run, and linked with the runtime libraries that
# follow. A unit without codes to compile is skipped. Both the unit and the libraries' host units see `definition_`,
# so the dispatch lists name the unit's kernels.
function (stringzilla_add_gpu_unit source_ codes_ definition_)
    if (NOT codes_)
        return()
    endif ()
    get_filename_component(unit_ "${source_}" NAME_WE)
    add_library(stringzilla_${unit_}_ OBJECT ${source_})
    target_include_directories(stringzilla_${unit_}_ PRIVATE ${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/c)
    set_target_properties(
        stringzilla_${unit_}_
        PROPERTIES POSITION_INDEPENDENT_CODE ON
                   C_VISIBILITY_PRESET hidden
                   CUDA_VISIBILITY_PRESET hidden
                   HIP_VISIBILITY_PRESET hidden
                   CUDA_STANDARD 20
                   HIP_STANDARD 20
                   CUDA_SEPARABLE_COMPILATION OFF
                   CUDA_ARCHITECTURES "${codes_}"
                   HIP_ARCHITECTURES "${codes_}"
    )
    stringzilla_set_compiler_flags(stringzilla_${unit_}_ "" "")
    target_compile_definitions(stringzilla_${unit_}_ PRIVATE "STRINGZILLA_WITH_LIBC=1" "${definition_}")
    target_link_libraries(
        stringzilla_${unit_}_ PRIVATE stringzilla_cpu_capabilities_disabled_ stringzilla_cuda_capabilities_compiled_
                                      ${ARGN}
    )
    foreach (library_ IN LISTS stringzilla_libraries_)
        target_sources(${library_} PRIVATE $<TARGET_OBJECTS:stringzilla_${unit_}_>)
        target_compile_definitions(${library_} PRIVATE "${definition_}")
        target_link_libraries(
            ${library_}
            PRIVATE $<BUILD_INTERFACE:stringzilla_cuda_capabilities_compiled_>
            PUBLIC ${ARGN}
        )
    endforeach ()
endfunction ()

if (STRINGZILLA_BUILD_CUDA)
    # One unit per tier, compiled only for the codes running it, so no cubin carries another tier's kernels.
    stringzilla_select_gpu_codes(
        stringzilla_hopper_unit_codes_ "${stringzilla_cuda_resolved_architectures_}"
        "${sz_target_hopper_architectures}"
    )
    stringzilla_select_gpu_codes(
        stringzilla_blackwell_unit_codes_ "${stringzilla_cuda_resolved_architectures_}"
        "${sz_target_blackwell_architectures}"
    )
    stringzilla_add_gpu_unit(
        c/target/cuda.cu "${stringzilla_cuda_resolved_architectures_}" "STRINGZILLA_ARCH_CUDA_=1" CUDA::cudart
    )
    stringzilla_add_gpu_unit(
        c/target/hopper.cu "${stringzilla_hopper_unit_codes_}" "STRINGZILLA_ARCH_CUDA_=1" CUDA::cudart
    )
    stringzilla_add_gpu_unit(
        c/target/blackwell.cu "${stringzilla_blackwell_unit_codes_}" "STRINGZILLA_ARCH_CUDA_=1" CUDA::cudart
    )
endif ()
if (STRINGZILLA_BUILD_ROCM)
    stringzilla_add_gpu_unit(c/target/rocm.hip "${STRINGZILLA_ROCM_ARCHITECTURES}" "STRINGZILLA_ARCH_ROCM_=1" hip::host)
endif ()
if (STRINGZILLA_BUILD_METAL)
    # Metal compiles on the device, so its one code names the family the baseline needs.
    stringzilla_add_gpu_unit(
        c/target/metal.c "apple7" "STRINGZILLA_ARCH_METAL_=1" "-framework Metal" "-framework Foundation"
    )
    target_compile_options(stringzilla_metal_ PRIVATE ${stringzilla_embed_options_})
    foreach (library_ IN LISTS stringzilla_libraries_)
        target_compile_options(${library_} PUBLIC ${stringzilla_embed_options_})
    endforeach ()
endif ()

# The `stringzilla` extension module, which scikit-build-core builds and packages from `pyproject.toml`, installed at the
# wheel's root, where `import stringzilla` finds it.
if (STRINGZILLA_BUILD_PYTHON)
    find_package(Python 3.10 REQUIRED COMPONENTS Interpreter Development.Module)
    find_package(OpenMP COMPONENTS C)
    python_add_library(
        stringzilla_python
        MODULE
        WITH_SOABI
        python/stringzilla/stringzilla.c
        python/stringzilla/shared.c
        python/stringzilla/file.c
        python/stringzilla/string.c
        python/stringzilla/strings.c
        python/stringzilla/hash.c
        python/stringzilla/cipher.c
        python/stringzilla/compare.c
        python/stringzilla/find.c
        python/stringzilla/sort.c
        python/stringzilla/intersect.c
        python/stringzilla/levenshtein.c
        python/stringzilla/overlap.c
        python/stringzilla/substrings.c
        python/stringzilla/regex.c
        python/stringzilla/subdocuments.c
        python/stringzilla/utf8_runes.c
        python/stringzilla/utf8_tokens.c
        python/stringzilla/utf8_boundaries.c
        python/stringzilla/utf8_wordbreaks.c
        python/stringzilla/utf8_graphemes.c
        python/stringzilla/utf8_sentences.c
        python/stringzilla/utf8_linebreaks.c
        python/stringzilla/utf8_uncased_fold.c
        python/stringzilla/utf8_uncased.c
        python/stringzilla/utf8_norm.c
        c/parallel.c
    )
    set_target_properties(stringzilla_python PROPERTIES OUTPUT_NAME stringzilla C_STANDARD 11)
    # Strict C11 hides `mmap` from glibc and musl headers included ahead of <Python.h>.
    target_compile_definitions(stringzilla_python PRIVATE $<$<PLATFORM_ID:Linux>:_GNU_SOURCE>)
    # The object types fill CPython's slots and pass through its `PyObject *` parameters uncast, which GCC 14 and Clang
    # 16 reject by default.
    target_compile_options(
        stringzilla_python PRIVATE $<$<C_COMPILER_ID:GNU,Clang,AppleClang>:-Wno-incompatible-pointer-types>
                                   $<$<C_COMPILER_ID:Clang,AppleClang>:-Wno-incompatible-function-pointer-types>
    )
    target_link_libraries(
        stringzilla_python PRIVATE stringzilla_static
                                   $<$<AND:$<BOOL:${OpenMP_C_FOUND}>,$<PLATFORM_ID:Linux,FreeBSD>>:OpenMP::OpenMP_C>
    )
    install(
        TARGETS stringzilla_python
        DESTINATION .
        COMPONENT stringzilla_python
    )
endif ()

# The `stringzilla.node` addon, which cmake-js builds from `package.json`. It stays off
# `stringzilla_set_compiler_flags`, whose output directories would move the Windows addon and whose `-Werror` and
# sanitizers belong to the tests.
if (STRINGZILLA_BUILD_NODE)
    if (NOT CMAKE_JS_INC)
        message(FATAL_ERROR "The Node addon is built by cmake-js, which supplies the headers and import library")
    endif ()
    find_package(OpenMP COMPONENTS C)
    # `CMAKE_JS_SRC` is the Windows delay-load hook and empty everywhere else.
    add_library(stringzilla_node SHARED javascript/stringzilla.c c/parallel.c ${CMAKE_JS_SRC})
    set_target_properties(
        stringzilla_node
        PROPERTIES PREFIX ""
                   SUFFIX ".node"
                   OUTPUT_NAME stringzilla
                   C_STANDARD 11
    )
    target_include_directories(stringzilla_node PRIVATE ${CMAKE_JS_INC})
    # The BigInt conversions need Node-API 6.
    target_compile_definitions(stringzilla_node PRIVATE NAPI_VERSION=8)
    target_link_libraries(
        stringzilla_node PRIVATE stringzilla_static ${CMAKE_JS_LIB}
                                 $<$<AND:$<BOOL:${OpenMP_C_FOUND}>,$<PLATFORM_ID:Linux,FreeBSD>>:OpenMP::OpenMP_C>
    )
endif ()

# The static library as SwiftPM links it, beside the headers and their module map: an XCFramework on Apple platforms,
# and elsewhere an SE-0482 artifact bundle. Each build adds its target to the one in STRINGZILLA_SWIFT_DIRECTORY, so a
# release fills one artifact from a build per target.
if (STRINGZILLA_BUILD_SWIFT)
    set(STRINGZILLA_SWIFT_DIRECTORY
        "${PROJECT_BINARY_DIR}"
        CACHE PATH "Where the Swift artifact lands"
    )
    if (APPLE)
        # The SDK names the XCFramework library an archive joins, and Mac Catalyst builds with the macOS one.
        get_filename_component(stringzilla_swift_sysroot_name_ "${CMAKE_OSX_SYSROOT}" NAME)
        string(TOLOWER "${stringzilla_swift_sysroot_name_}" stringzilla_swift_sysroot_lowercase_)
        string(REGEX REPLACE "[0-9.]*\\.sdk$" "" stringzilla_swift_sdk_ "${stringzilla_swift_sysroot_lowercase_}")
        if (CMAKE_C_FLAGS MATCHES "-macabi")
            set(stringzilla_swift_sdk_ maccatalyst)
        elseif (NOT stringzilla_swift_sdk_)
            set(stringzilla_swift_sdk_ macosx)
        endif ()
        set(stringzilla_swift_target_ -D "STRINGZILLA_SWIFT_SDK=${stringzilla_swift_sdk_}" -D
                                      "STRINGZILLA_SWIFT_ARCHITECTURE=${CMAKE_SYSTEM_PROCESSOR}"
        )
    else ()
        # SwiftPM matches a variant by the exact triple it builds for, spelled as Swift spells it.
        if (stringzilla_arch_x8664_)
            set(stringzilla_swift_arch_ x86_64)
        elseif (stringzilla_arch_arm64_)
            set(stringzilla_swift_arch_ aarch64)
        else ()
            set(stringzilla_swift_arch_ "${CMAKE_SYSTEM_PROCESSOR}")
        endif ()
        set(stringzilla_swift_triple_)
        if (CMAKE_SYSTEM_NAME STREQUAL "Linux")
            set(stringzilla_swift_triple_ "${stringzilla_swift_arch_}-unknown-linux-gnu")
        elseif (CMAKE_SYSTEM_NAME STREQUAL "Android")
            set(stringzilla_swift_triple_ "${stringzilla_swift_arch_}-unknown-linux-android${ANDROID_NATIVE_API_LEVEL}")
        elseif (CMAKE_SYSTEM_NAME STREQUAL "Windows")
            set(stringzilla_swift_triple_ "${stringzilla_swift_arch_}-unknown-windows-msvc")
        elseif (CMAKE_SYSTEM_NAME STREQUAL "WASI")
            set(stringzilla_swift_triple_ "wasm32-unknown-wasip1")
        endif ()
        set(STRINGZILLA_SWIFT_TRIPLES
            "${stringzilla_swift_triple_}"
            CACHE STRING "Swift target triples the artifact bundle serves"
        )
        if (NOT STRINGZILLA_SWIFT_TRIPLES)
            message(FATAL_ERROR "STRINGZILLA_BUILD_SWIFT needs STRINGZILLA_SWIFT_TRIPLES on ${CMAKE_SYSTEM_NAME}")
        endif ()
        list(JOIN STRINGZILLA_SWIFT_TRIPLES "," stringzilla_swift_triples_)
        set(stringzilla_swift_target_ -D "STRINGZILLA_SWIFT_TRIPLES=${stringzilla_swift_triples_}" -D
                                      "STRINGZILLA_SWIFT_VERSION=${PROJECT_VERSION}"
        )
    endif ()
    add_custom_target(
        stringzilla_swift ALL
        COMMAND
            "${CMAKE_COMMAND}" -D "STRINGZILLA_SWIFT_ARCHIVE=$<TARGET_FILE:stringzilla_static>" -D
            "STRINGZILLA_SWIFT_HEADERS=${PROJECT_SOURCE_DIR}/include" -D
            "STRINGZILLA_SWIFT_DIRECTORY=${STRINGZILLA_SWIFT_DIRECTORY}" ${stringzilla_swift_target_} -P
            "${PROJECT_SOURCE_DIR}/cmake/sz_swift_artifact.cmake"
        DEPENDS stringzilla_static
        VERBATIM
    )
endif ()

# Tests and benchmarks come after the libraries so they link the precompiled archives rather than recompiling.
if (STRINGZILLA_BUILD_BENCH)
    # The family files time the dispatch points against the standard baselines, and each `cross_<arch>.cpp` times its
    # architecture's kernels by name, one section per capability, against the serial ones.
    set(stringzilla_cross_bench_sources_
        bench/cross_serial.cpp
        bench/cross_x8664.cpp
        bench/cross_arm64.cpp
        bench/cross_riscv64.cpp
        bench/cross_loongarch64.cpp
        bench/cross_ppc64.cpp
        bench/cross_wasm.cpp
    )
    set(stringzilla_cpu_bench_sources_
        bench/main.cpp
        bench/find.cpp
        bench/token.cpp
        bench/sort.cpp
        bench/strings.cpp
        bench/cipher.cpp
        bench/container.cpp
        bench/levenshtein.cpp
        bench/overlap.cpp
        bench/substrings.cpp
        bench/regex.cpp
        bench/subdocuments.cpp
        bench/utf8_traverse.cpp
        bench/utf8_scan.cpp
        bench/utf8_segment.cpp
        bench/utf8_norm.cpp
        bench/utf8_uncased.cpp
        ${stringzilla_cross_bench_sources_}
    )
    stringzilla_add_cpu_launcher(stringzilla_bench stringzilla_static COMPILE ${stringzilla_cpu_bench_sources_})
    # Header-only builds have no dispatch points to time, but their kernels compile inline, which also reaches the
    # private helpers the library keeps to itself, like the SIMD sorts of pgrams.
    stringzilla_add_cpu_launcher(
        stringzilla_cpu_header_bench stringzilla_header COMPILE bench/main.cpp ${stringzilla_cross_bench_sources_}
    )

endif ()

if (STRINGZILLA_BUILD_TEST)
    # The family files test the dispatch points and `stringzilla.hpp`, and each `cross_<arch>.cpp` holds its
    # architecture's kernels to the serial ones, a section per capability, skipping those this CPU lacks.
    set(stringzilla_cross_test_sources_
        test/cross_serial.cpp
        test/cross_x8664.cpp
        test/cross_arm64.cpp
        test/cross_riscv64.cpp
        test/cross_loongarch64.cpp
        test/cross_ppc64.cpp
        test/cross_wasm.cpp
    )
    set(stringzilla_cpu_test_sources_
        test/main_cpu.cpp
        test/hash.cpp
        test/cipher.cpp
        test/sort.cpp
        test/levenshtein.cpp
        test/overlap.cpp
        test/substrings.cpp
        test/regex.cpp
        test/subdocuments.cpp
        test/string.cpp
        test/find.cpp
        test/utf8_runes.cpp
        test/utf8_tokens.cpp
        test/utf8_wordbreaks.cpp
        test/utf8_graphemes.cpp
        test/utf8_sentences.cpp
        test/utf8_linebreaks.cpp
        test/utf8_norm.cpp
        test/utf8_uncased.cpp
        ${stringzilla_cross_test_sources_}
    )

    stringzilla_add_cpu_launcher(stringzilla_test stringzilla_static COMPILE ${stringzilla_cpu_test_sources_})
    add_test(NAME stringzilla_test COMMAND stringzilla_test)
    # Cross builds link their executables statically, to run them under an emulator without the target's loader.
    if (TARGET stringzilla_shared AND NOT CMAKE_CROSSCOMPILING)
        stringzilla_add_cpu_launcher(stringzilla_shared_test stringzilla_shared SERIAL ${stringzilla_cpu_test_sources_})
        add_test(NAME stringzilla_shared_test COMMAND stringzilla_shared_test)
    endif ()
    # The family files drive `stringzilla.hpp`, which always goes through the library, so the header-only test keeps
    # the kernel cross-checks and the dispatch-point stubs.
    stringzilla_add_cpu_launcher(
        stringzilla_cpu_header_test stringzilla_header COMPILE test/main_cpu.cpp ${stringzilla_cross_test_sources_}
    )
    add_test(NAME stringzilla_cpu_header_test COMMAND stringzilla_cpu_header_test)

endif ()

foreach (suite_ bench test)
    if (NOT TARGET stringzilla_${suite_})
        continue()
    endif ()
    foreach (vendor_ cuda rocm metal)
        set(language_ CXX)
        if (vendor_ STREQUAL "cuda")
            set(source_ "${suite_}/cross_cuda.cu")
            if (STRINGZILLA_BUILD_CUDA)
                set(language_ CUDA)
            else ()
                set_source_files_properties(${source_} PROPERTIES LANGUAGE CXX)
            endif ()
        elseif (vendor_ STREQUAL "rocm")
            set(source_ "${suite_}/cross_rocm.hip")
            if (STRINGZILLA_BUILD_ROCM)
                set(language_ HIP)
            else ()
                set_source_files_properties(${source_} PROPERTIES LANGUAGE CXX)
            endif ()
        else ()
            set(source_ "${suite_}/cross_metal.cpp")
        endif ()
        set(unit_ "stringzilla_${vendor_}_${suite_}_")
        stringzilla_add_gpu_suite(${unit_} ${language_} ${source_})
        if (vendor_ STREQUAL "metal" AND STRINGZILLA_BUILD_METAL)
            target_compile_definitions(${unit_} PRIVATE STRINGZILLA_ARCH_METAL_=1)
        endif ()
        target_sources(stringzilla_${suite_} PRIVATE $<TARGET_OBJECTS:${unit_}>)
        if (suite_ STREQUAL "test" AND TARGET stringzilla_shared_test)
            target_sources(stringzilla_shared_test PRIVATE $<TARGET_OBJECTS:${unit_}>)
        endif ()
    endforeach ()
endforeach ()

if (STRINGZILLA_INSTALL)
    include(CMakePackageConfigHelpers)

    # Every installed library belongs to an `EXPORT` set - a `.so` with no imported target leaves consumers
    # hand-rolling `find_library`.
    # Destinations come from the `GNUInstallDirs` defaults, so each call names only its target and export set.
    install(TARGETS stringzilla_header stringzilla_static EXPORT stringzillaTargets)
    if (TARGET stringzilla_shared)
        install(TARGETS stringzilla_shared EXPORT stringzillaTargets)
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

    # SwiftPM reads `module.modulemap` from the XCFramework or artifact bundle `stringzilla_swift` packs. At an installed
    # include root it is a hazard: Clang treats whichever one it finds there as authoritative for the whole prefix.
    install(
        DIRECTORY ${stringzilla_include_build_dir_}
        DESTINATION ${stringzilla_include_install_dir_}
        PATTERN "module.modulemap" EXCLUDE
    )
    install(DIRECTORY c/ DESTINATION ${CMAKE_INSTALL_DATADIR}/${PROJECT_NAME}/src)

    # A `.pc` for the consumers CMake never reaches - cgo among them. Spelling the prefix against `${pcfiledir}` keeps it
    # correct after `cmake --install --prefix` moves the tree. `GNUInstallDirs` returns a relative path unless a distro
    # forced an absolute one, and only the former may follow a prefix. No `Libs.private`: the shared library is C and
    # links nothing but libc, so a static caller inherits no closure to name.
    if (TARGET stringzilla_shared)
        file(RELATIVE_PATH stringzilla_pc_relative_prefix_ "${CMAKE_INSTALL_FULL_LIBDIR}/pkgconfig"
             "${CMAKE_INSTALL_PREFIX}"
        )
        string(REGEX REPLACE "/+$" "" stringzilla_pc_trimmed_prefix_ "${stringzilla_pc_relative_prefix_}")
        set(stringzilla_pc_prefix_ "\${pcfiledir}/${stringzilla_pc_trimmed_prefix_}")

        if (IS_ABSOLUTE "${CMAKE_INSTALL_LIBDIR}")
            set(stringzilla_pc_libdir_ "${CMAKE_INSTALL_LIBDIR}")
        else ()
            set(stringzilla_pc_libdir_ "\${exec_prefix}/${CMAKE_INSTALL_LIBDIR}")
        endif ()
        if (IS_ABSOLUTE "${CMAKE_INSTALL_INCLUDEDIR}")
            set(stringzilla_pc_includedir_ "${CMAKE_INSTALL_INCLUDEDIR}")
        else ()
            set(stringzilla_pc_includedir_ "\${prefix}/${CMAKE_INSTALL_INCLUDEDIR}")
        endif ()

        configure_file(cmake/stringzilla.pc.in "${CMAKE_CURRENT_BINARY_DIR}/stringzilla.pc" @ONLY)
        install(FILES "${CMAKE_CURRENT_BINARY_DIR}/stringzilla.pc" DESTINATION ${CMAKE_INSTALL_LIBDIR}/pkgconfig)
    endif ()

endif ()
