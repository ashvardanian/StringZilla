# StringZilla CMakeLists.txt
#
# This file defines several library build & installation targets:
#
# * stringzilla_header: A header-only library with the StringZilla C and C++ headers.
# * stringzilla_static: A static library with dynamic SIMD dispatch, and the GPU backends STRINGZILLA_BUILD_CUDA,
#   STRINGZILLA_BUILD_ROCM and STRINGZILLA_BUILD_METAL ask for, side by side.
# * stringzilla_shared: The same as a shared library.
# * stringzilla_bare: A shared library with the StringZilla headers, but without linking the standard C library.
# * stringzilla_python: The CPython extension over `stringzilla_static`, with STRINGZILLA_BUILD_PYTHON.
# * stringzilla_node: The Node addon over `stringzilla_static`, with STRINGZILLA_BUILD_NODE, which cmake-js sets.
# * stringzilla_swift: `stringzilla_static` packaged for SwiftPM, with STRINGZILLA_BUILD_SWIFT.
#
# Consumers link the same namespaced spellings whether pulled in with `add_subdirectory` or found with `find_package`:
#
# * stringzilla::header, stringzilla::static, stringzilla::shared, stringzilla::bare.
#
# Tests:
#
# * stringzilla_cpu_test: The dispatch points, the C++ interface, and every kernel the static library defines.
# * stringzilla_cpu_shared_test: The dispatch points alone, through the shared library.
# * stringzilla_cpu_header_test: The kernels compiled inline, header-only, and the dispatch-point stubs.
# * stringzilla_cuda_test: The CUDA kernels and the dispatch points over them, against the serial answers.
# * stringzilla_rocm_test: The same checks over the ROCm kernels.
# * stringzilla_metal_test: The same checks over the Metal kernels.
#
# Benchmarks:
#
# * stringzilla_cpu_bench: The dispatch points against the standard baselines, and every kernel the static library
#   defines against its serial twin.
# * stringzilla_cpu_header_bench: The kernels compiled inline, header-only, with the private helpers only the tier
#   headers define.
# * stringzilla_cuda_bench: The CUDA kernels of the engine families against the CPU tiers this build carries.
# * stringzilla_rocm_bench: The same over the ROCm kernels.
# * stringzilla_metal_bench: The Metal kernels of the window-overlap family against the serial one.
#
# Python, Node, and Swift build their artifacts through this file, while Rust's `build.rs` and Go's cgo link
# `stringzilla_static`; Java and C# load `stringzilla_shared`.
cmake_minimum_required(VERSION 3.21 FATAL_ERROR)

# Every binding reads its version from `VERSION`, which the release bumps.
file(STRINGS "${CMAKE_CURRENT_SOURCE_DIR}/VERSION" STRINGZILLA_VERSION_ LIMIT_COUNT 1)
project(
    stringzilla
    VERSION ${STRINGZILLA_VERSION_}
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
    list(LENGTH CMAKE_OSX_ARCHITECTURES STRINGZILLA_OSX_ARCHITECTURES_COUNT_)
    if (NOT STRINGZILLA_OSX_ARCHITECTURES_COUNT_ EQUAL 1)
        message(
            FATAL_ERROR "CMAKE_OSX_ARCHITECTURES='${CMAKE_OSX_ARCHITECTURES}' holds several slices; build one per slice"
        )
    endif ()
    set(CMAKE_SYSTEM_PROCESSOR "${CMAKE_OSX_ARCHITECTURES}")
endif ()

# Detect target architecture from `CMAKE_SYSTEM_PROCESSOR`... safer for cross-compilation!
if (CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|X86_64|AMD64|amd64")
    set(STRINGZILLA_ARCH_X8664_ TRUE)
    message(STATUS "Platform: x86_64 (CMAKE_SYSTEM_PROCESSOR=${CMAKE_SYSTEM_PROCESSOR})")
elseif (CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|AARCH64|arm64|ARM64")
    set(STRINGZILLA_ARCH_ARM64_ TRUE)
    message(STATUS "Platform: ARM64 (CMAKE_SYSTEM_PROCESSOR=${CMAKE_SYSTEM_PROCESSOR})")
elseif (CMAKE_SYSTEM_PROCESSOR MATCHES "riscv64|RISCV64")
    set(STRINGZILLA_ARCH_RISCV64_ TRUE)
    message(STATUS "Platform: RISC-V 64 (CMAKE_SYSTEM_PROCESSOR=${CMAKE_SYSTEM_PROCESSOR})")
elseif (CMAKE_SYSTEM_PROCESSOR MATCHES "loongarch64|LOONGARCH64|loong64|LOONG64")
    # "loongarch64" is the GNU triple and `uname -m` spelling; "loong64" is the Debian/dpkg port name.
    set(STRINGZILLA_ARCH_LOONGARCH64_ TRUE)
    message(STATUS "Platform: LoongArch64 (CMAKE_SYSTEM_PROCESSOR=${CMAKE_SYSTEM_PROCESSOR})")
elseif (CMAKE_SYSTEM_PROCESSOR MATCHES "ppc64|powerpc64|PPC64|POWERPC64")
    # Matches both endiannesses ("ppc64le" contains "ppc64"); the kernels handle big-endian explicitly
    # (`find/powervsx.h`) and `STRINGZILLA_ARCH_BIG_ENDIAN_` is derived from `CMAKE_C_BYTE_ORDER` separately.
    set(STRINGZILLA_ARCH_PPC64_ TRUE)
    message(STATUS "Platform: PowerPC64 (CMAKE_SYSTEM_PROCESSOR=${CMAKE_SYSTEM_PROCESSOR})")
elseif (CMAKE_SYSTEM_PROCESSOR MATCHES "^wasm(32|64)$")
    # A module's one SIMD kit is `STRINGZILLA_TARGET_ARCH`, which `types.h` reads back off the predefined macros, so
    # no probe verdict is set here.
    set(STRINGZILLA_ARCH_WASM_ TRUE)
    message(STATUS "Platform: WebAssembly (CMAKE_SYSTEM_PROCESSOR=${CMAKE_SYSTEM_PROCESSOR})")
else ()
    message(WARNING "Unknown CMAKE_SYSTEM_PROCESSOR: ${CMAKE_SYSTEM_PROCESSOR}")
endif ()

# A `Generic` target, like Rust's `wasm32-unknown-unknown`, has no LibC for the libraries to reach.
if (CMAKE_SYSTEM_NAME STREQUAL "Generic")
    set(STRINGZILLA_WITH_LIBC_ 0)
else ()
    set(STRINGZILLA_WITH_LIBC_ 1)
endif ()

# One probe per kit under `probes/`, whose `STRINGZILLA_TARGET_<KIT>=0/1` verdicts reach the libraries and
# the CPU tests and benchmarks, but not `stringzilla_header`, whose consumers derive each kit from their
# own flags in `types.h`.
include(cmake/sz_isa_probe.cmake)

# LASX and POWER9 compile file-wide, so only the kit's own unit and the header-only cross-checks of its
# architecture take the flag; every other header-only unit follows its own flags in `types.h`, which leave
# those kits off.
set(sz_in_header_only_suite_
    "$<IN_LIST:$<TARGET_PROPERTY:NAME>,stringzilla_cpu_header_test;stringzilla_cpu_header_bench>"
)
if ("STRINGZILLA_TARGET_LOONGSONASX=1" IN_LIST sz_compile_definitions_)
    set_source_files_properties(c/cpu/loongsonasx.c PROPERTIES COMPILE_OPTIONS ${sz_kit_flags_LOONGSONASX})
    set_source_files_properties(
        test/cross_loongarch64.cpp bench/cross_loongarch64.cpp
        PROPERTIES COMPILE_OPTIONS "$<${sz_in_header_only_suite_}:${sz_kit_flags_LOONGSONASX}>"
    )
endif ()
if ("STRINGZILLA_TARGET_POWERVSX=1" IN_LIST sz_compile_definitions_)
    set_source_files_properties(c/cpu/powervsx.c PROPERTIES COMPILE_OPTIONS ${sz_kit_flags_POWERVSX})
    set_source_files_properties(
        test/cross_ppc64.cpp bench/cross_ppc64.cpp PROPERTIES COMPILE_OPTIONS
                                                              "$<${sz_in_header_only_suite_}:${sz_kit_flags_POWERVSX}>"
    )
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
option(STRINGZILLA_USE_SANITIZERS "Enable AddressSanitizer and UndefinedBehaviorSanitizer in Debug builds" ON)

# Clang source-based coverage for the header-only library. Recipe (single-arch, native backends only): cmake -B
# build_cov -DSTRINGZILLA_BUILD_COVERAGE=1 -DSTRINGZILLA_USE_SANITIZERS=0 -DCMAKE_CXX_COMPILER=clang++ cmake --build
# build_cov --target stringzilla_cpu_test -j LLVM_PROFILE_FILE=/tmp/sz.profraw ./build_cov/stringzilla_cpu_test
# llvm-profdata merge -sparse /tmp/sz.profraw -o /tmp/sz.profdata llvm-cov report ./build_cov/stringzilla_cpu_test
# -instr-profile=/tmp/sz.profdata include/stringzilla/ `-march=native` compiles out the non-native backends, so a
# faithful picture needs `llvm-profdata merge` across per-ISA builds (native x86 + qemu-aarch64 for NEON/SVE + the RVV
# qemu toolchain).
option(STRINGZILLA_BUILD_COVERAGE "Instrument with Clang source-based coverage" OFF)
# Seeded from the environment at the first configure, so `cargo` and `cmake-js` honour `STRINGZILLA_TARGET_ARCH=native`.
set(STRINGZILLA_TARGET_ARCH
    "$ENV{STRINGZILLA_TARGET_ARCH}"
    CACHE STRING "Host-tuned `-march`, like native, or a WebAssembly module's SIMD kit, serial, v128 or v128relaxed"
)

# An engine validates a WebAssembly module whole, so every unit carries the module's one SIMD kit, whatever flags a
# binding passes, and `types.h` reads it back off the predefined macros.
if (STRINGZILLA_ARCH_WASM_)
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
    # arch-locked `a`. The libraries' CUDA unit adds forward PTX on top of these, which a listed GPU never JITs.
    set(STRINGZILLA_CUDA_ARCHITECTURES
        "75-real;80-real;90-real;100f-real;120f-real"
        CACHE STRING "Real SASS tiers every CUDA target carries; override to match the machine at hand"
    )
    set(CMAKE_CUDA_ARCHITECTURES ${STRINGZILLA_CUDA_ARCHITECTURES})
    set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
    find_package(CUDAToolkit REQUIRED)
    include(cmake/sz_cuda_probes.cmake)
    sz_cuda_probe_native_arch_()

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

# One GPU test or benchmark over the static library, whose vendor unit holds the kernels it names: compiled as
# `language`, which its sources' extension spells, for the architectures the library's vendor unit carries.
function (define_gpu_launcher exec_name language)
    add_executable(${exec_name} ${ARGN})
    set_compiler_flags(${exec_name} 20 "${STRINGZILLA_TARGET_ARCH}" "${CMAKE_${language}_COMPILER_ID}")
    set_target_properties(
        ${exec_name}
        PROPERTIES CUDA_ARCHITECTURES "${STRINGZILLA_CUDA_ARCHITECTURES}"
                   HIP_ARCHITECTURES "${STRINGZILLA_ROCM_ARCHITECTURES}"
                   HIP_STANDARD 20
    )
    # NVCC's front end cannot honour MSVC's `/utf-8`, without which {fmt} refuses to build its Unicode support.
    if (MSVC AND language STREQUAL "CUDA")
        target_compile_definitions(${exec_name} PRIVATE FMT_UNICODE=0)
    endif ()
    target_link_libraries(${exec_name} PRIVATE stringzilla_static fmt::fmt)
endfunction ()

# One CPU test or benchmark linking `library`, whose capability `verdicts` declare every kernel the
# toolchain builds (COMPILE), as the harnesses skip those this CPU lacks, or turn them all off, leaving
# the dispatch points (SERIAL).
function (define_cpu_launcher exec_name library verdicts)
    add_executable(${exec_name} ${ARGN})
    set_compiler_flags(${exec_name} 20 "${STRINGZILLA_TARGET_ARCH}" "${CMAKE_CXX_COMPILER_ID}")
    if (verdicts STREQUAL "SERIAL")
        target_compile_definitions(${exec_name} PRIVATE ${STRINGZILLA_SERIAL_DEFINITIONS_})
    elseif (library STREQUAL "stringzilla_header")
        target_compile_definitions(${exec_name} PRIVATE ${sz_header_only_definitions_})
    else ()
        set_architecture_simd_definitions(${exec_name})
    endif ()
    target_link_libraries(${exec_name} PRIVATE ${library} fmt::fmt)
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
# Consumers of the header target compile every kernel themselves, and link no dispatch points.
target_compile_definitions(stringzilla_header INTERFACE STRINGZILLA_HEADER_ONLY=1)

# The compiled core: `c/stringzilla.c` owns the cross-family exports (versions, capabilities, the kernel finder and the
# LibC overrides), each `c/cpu/<capability>.c` defines that capability's kernels once, and each
# `c/dispatch/<family>.c` holds its family's lists, dispatch points and finder. Globbed, so a new unit needs no entry.
file(
    GLOB STRINGZILLA_LIBRARY_UNITS_ CONFIGURE_DEPENDS
    RELATIVE ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/c/cpu/*.c ${PROJECT_SOURCE_DIR}/c/dispatch/*.c
)
set(STRINGZILLA_SHIM_SOURCES c/stringzilla.c ${STRINGZILLA_LIBRARY_UNITS_})

# Helper function used for the `stringzilla_shared`, `stringzilla_bare`, and `stringzilla_static` targets, which compile
# the sources that follow with the compiler `compiler_id` names
function (define_stringzilla_library target public_name library_type compiler_id)
    add_library(${target} ${library_type} ${ARGN})

    # The library, unlike the headers, may use C11: `c/stringzilla.c` caches detection in a <stdatomic.h> atomic, which
    # MSVC gates behind a flag.
    set_target_properties(${target} PROPERTIES C_STANDARD 11)
    target_compile_options(${target} PRIVATE $<$<COMPILE_LANG_AND_ID:C,MSVC>:/experimental:c11atomics>)

    # The dispatch units share the private `c/dispatch.h`.
    target_include_directories(${target} PRIVATE ${PROJECT_SOURCE_DIR}/c)

    # Only `STRINGZILLA_API` functions leave the library; Windows exports every symbol, as it spells no `__declspec`.
    set_target_properties(
        ${target}
        PROPERTIES C_VISIBILITY_PRESET hidden
                   CXX_VISIBILITY_PRESET hidden
                   CUDA_VISIBILITY_PRESET hidden
                   WINDOWS_EXPORT_ALL_SYMBOLS ON
    )

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

    # Set the baseline architecture; the dispatch points pick each SIMD capability by the mask each call passes.
    set_baseline_architecture_flags(${target} "${compiler_id}")

    # Enable every capability the toolchain can compile; `sz_cpu_capabilities_enabled` masks what the CPU lacks.
    set_architecture_simd_definitions(${target})

endfunction ()

# The dispatch core as an archive, which the tests, the benchmarks, and the Go and Python bindings link - the
# cross-compiled test runs are `-static` executables, which cannot take a `.so` at all - and as a shared library. A
# parent project builds the archive only if it links it.
define_stringzilla_library(stringzilla_static static STATIC "${CMAKE_CXX_COMPILER_ID}" ${STRINGZILLA_SHIM_SOURCES})
if (NOT STRINGZILLA_IS_MAIN_PROJECT)
    set_target_properties(stringzilla_static PROPERTIES EXCLUDE_FROM_ALL ON)
endif ()
if (STRINGZILLA_BUILD_SHARED)
    define_stringzilla_library(stringzilla_shared shared SHARED "${CMAKE_CXX_COMPILER_ID}" ${STRINGZILLA_SHIM_SOURCES})
    set(STRINGZILLA_LIBRARIES_ stringzilla_static stringzilla_shared)
else ()
    set(STRINGZILLA_LIBRARIES_ stringzilla_static)
endif ()

# The shim units default `STRINGZILLA_OVERRIDE_LIBC` to `!STRINGZILLA_WITH_LIBC`, so these libraries export no `memcpy` of
# their own and interpose nothing. That is what lets anything link them - the tests, a consumer's program - without ELF
# redirecting `libstdc++`'s internal calls into StringZilla. `stringzilla_bare` is the artifact that replaces LibC, and
# it still does.
foreach (library_ IN LISTS STRINGZILLA_LIBRARIES_)
    target_compile_definitions(${library_} PRIVATE "STRINGZILLA_WITH_LIBC=${STRINGZILLA_WITH_LIBC_}")
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
    if (MSVC AND STRINGZILLA_ARCH_ARM64_)
        target_compile_definitions(${library_} PUBLIC _ARM64_)
    endif ()
endforeach ()

# Every probed kit switched off, for the units that must stay at the platform baseline: the GPU vendor units and the
# CPU tests of the dispatch points alone.
string(REPLACE "=1" "=0" STRINGZILLA_SERIAL_DEFINITIONS_ "${sz_compile_definitions_}")

# Adds one GPU vendor's unit, like `c/nvidia/cuda.cu`, to the archive and the shared library: compiled as `language`,
# without the CPU kits, and linked with the runtime libraries that follow. Both the unit and the libraries' host units
# see `definition`, so the dispatch lists name the unit's kernels. The bare library stays without them, as every GPU
# runtime needs the LibC it replaces.
function (sz_add_gpu_unit_ source language definition)
    get_filename_component(unit_ "${source}" NAME_WE)
    add_library(stringzilla_${unit_}_ OBJECT ${source})
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
                   CUDA_ARCHITECTURES "${STRINGZILLA_CUDA_ARCHITECTURES};90-virtual"
                   HIP_ARCHITECTURES "${STRINGZILLA_ROCM_ARCHITECTURES}"
    )
    set_compiler_flags(stringzilla_${unit_}_ "" "" "${CMAKE_${language}_COMPILER_ID}")
    target_compile_definitions(
        stringzilla_${unit_}_ PRIVATE ${STRINGZILLA_SERIAL_DEFINITIONS_} "STRINGZILLA_WITH_LIBC=1" "${definition}"
    )
    target_link_libraries(stringzilla_${unit_}_ PRIVATE ${ARGN})
    foreach (library_ IN LISTS STRINGZILLA_LIBRARIES_)
        target_sources(${library_} PRIVATE $<TARGET_OBJECTS:stringzilla_${unit_}_>)
        target_compile_definitions(${library_} PRIVATE "${definition}")
        target_link_libraries(${library_} PUBLIC ${ARGN})
    endforeach ()
endfunction ()

if (STRINGZILLA_BUILD_CUDA)
    sz_add_gpu_unit_(c/nvidia/cuda.cu CUDA "STRINGZILLA_ARCH_CUDA_=1" CUDA::cudart)
endif ()
if (STRINGZILLA_BUILD_ROCM)
    sz_add_gpu_unit_(c/amd/rocm.hip HIP "STRINGZILLA_ARCH_ROCM_=1" hip::host)
endif ()
if (STRINGZILLA_BUILD_METAL)
    sz_add_gpu_unit_(c/apple/metal.c C "STRINGZILLA_WITH_METAL=1" "-framework Metal" "-framework Foundation")
endif ()

if (STRINGZILLA_BUILD_SHARED)
    # Try compiling a version without linking the LibC ! This is only for Linux/MSVC, as on modern Arm-based MacOS
    # machines ! We can't legally access Arm's "feature registers" without `sysctl` or `sysctlbyname`.
    if (NOT CMAKE_SYSTEM_NAME MATCHES "Darwin")
        define_stringzilla_library(stringzilla_bare bare SHARED "${CMAKE_CXX_COMPILER_ID}" ${STRINGZILLA_SHIM_SOURCES})
        target_compile_definitions(stringzilla_bare PRIVATE "STRINGZILLA_WITH_LIBC=0")
        target_compile_definitions(stringzilla_bare PRIVATE "STRINGZILLA_OVERRIDE_LIBC=1")
        target_include_directories(
            stringzilla_bare PUBLIC $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
                                    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
        )

        # For stringzilla_bare, enforce strict standard conformance with `-pedantic`, so the bare build stays free of
        # the POSIX and GNU extensions a LibC would otherwise lend it.
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
        python/stringzilla/str.c
        python/stringzilla/strs.c
        python/stringzilla/memory.c
        python/stringzilla/hash.c
        python/stringzilla/cipher.c
        python/stringzilla/compare.c
        python/stringzilla/find.c
        python/stringzilla/sort.c
        python/stringzilla/intersect.c
        python/stringzilla/levenshtein.c
        python/stringzilla/overlap.c
        python/stringzilla/substrings.c
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

# The `stringzilla.node` addon, which cmake-js builds from `package.json`. It stays off `set_compiler_flags`, whose output
# directories would move the Windows addon and whose `-Werror` and sanitizers belong to the tests.
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
        get_filename_component(STRINGZILLA_SWIFT_SDK_ "${CMAKE_OSX_SYSROOT}" NAME)
        string(TOLOWER "${STRINGZILLA_SWIFT_SDK_}" STRINGZILLA_SWIFT_SDK_)
        string(REGEX REPLACE "[0-9.]*\\.sdk$" "" STRINGZILLA_SWIFT_SDK_ "${STRINGZILLA_SWIFT_SDK_}")
        if (CMAKE_C_FLAGS MATCHES "-macabi")
            set(STRINGZILLA_SWIFT_SDK_ maccatalyst)
        elseif (NOT STRINGZILLA_SWIFT_SDK_)
            set(STRINGZILLA_SWIFT_SDK_ macosx)
        endif ()
        set(STRINGZILLA_SWIFT_TARGET_ -D "STRINGZILLA_SWIFT_SDK=${STRINGZILLA_SWIFT_SDK_}" -D
                                      "STRINGZILLA_SWIFT_ARCHITECTURE=${CMAKE_SYSTEM_PROCESSOR}"
        )
    else ()
        # SwiftPM matches a variant by the exact triple it builds for, spelled as Swift spells it.
        if (STRINGZILLA_ARCH_X8664_)
            set(STRINGZILLA_SWIFT_ARCH_ x86_64)
        elseif (STRINGZILLA_ARCH_ARM64_)
            set(STRINGZILLA_SWIFT_ARCH_ aarch64)
        else ()
            set(STRINGZILLA_SWIFT_ARCH_ "${CMAKE_SYSTEM_PROCESSOR}")
        endif ()
        set(STRINGZILLA_SWIFT_TRIPLE_)
        if (CMAKE_SYSTEM_NAME STREQUAL "Linux")
            set(STRINGZILLA_SWIFT_TRIPLE_ "${STRINGZILLA_SWIFT_ARCH_}-unknown-linux-gnu")
        elseif (CMAKE_SYSTEM_NAME STREQUAL "Android")
            set(STRINGZILLA_SWIFT_TRIPLE_ "${STRINGZILLA_SWIFT_ARCH_}-unknown-linux-android${ANDROID_NATIVE_API_LEVEL}")
        elseif (CMAKE_SYSTEM_NAME STREQUAL "Windows")
            set(STRINGZILLA_SWIFT_TRIPLE_ "${STRINGZILLA_SWIFT_ARCH_}-unknown-windows-msvc")
        elseif (CMAKE_SYSTEM_NAME STREQUAL "WASI")
            set(STRINGZILLA_SWIFT_TRIPLE_ "wasm32-unknown-wasip1")
        endif ()
        set(STRINGZILLA_SWIFT_TRIPLES
            "${STRINGZILLA_SWIFT_TRIPLE_}"
            CACHE STRING "Swift target triples the artifact bundle serves"
        )
        if (NOT STRINGZILLA_SWIFT_TRIPLES)
            message(FATAL_ERROR "STRINGZILLA_BUILD_SWIFT needs STRINGZILLA_SWIFT_TRIPLES on ${CMAKE_SYSTEM_NAME}")
        endif ()
        set(STRINGZILLA_SWIFT_TARGET_ -D "STRINGZILLA_SWIFT_TRIPLES=$<JOIN:${STRINGZILLA_SWIFT_TRIPLES},,>" -D
                                      "STRINGZILLA_SWIFT_VERSION=${PROJECT_VERSION}"
        )
    endif ()
    add_custom_target(
        stringzilla_swift ALL
        COMMAND
            "${CMAKE_COMMAND}" -D "STRINGZILLA_SWIFT_ARCHIVE=$<TARGET_FILE:stringzilla_static>" -D
            "STRINGZILLA_SWIFT_HEADERS=${PROJECT_SOURCE_DIR}/include" -D
            "STRINGZILLA_SWIFT_DIRECTORY=${STRINGZILLA_SWIFT_DIRECTORY}" ${STRINGZILLA_SWIFT_TARGET_} -P
            "${PROJECT_SOURCE_DIR}/cmake/sz_swift_artifact.cmake"
        DEPENDS stringzilla_static
        VERBATIM
    )
endif ()

# Tests and benchmarks come after the libraries so they link the precompiled archives rather than recompiling.
if (STRINGZILLA_BUILD_BENCH)
    # The family files time the dispatch points against the standard baselines, and each `cross_<arch>.cpp` times its
    # architecture's kernels by name, one section per capability, against the serial ones.
    set(STRINGZILLA_CROSS_BENCH_SOURCES_
        bench/cross_serial.cpp
        bench/cross_x8664.cpp
        bench/cross_arm64.cpp
        bench/cross_riscv64.cpp
        bench/cross_loongarch64.cpp
        bench/cross_ppc64.cpp
        bench/cross_wasm.cpp
    )
    set(STRINGZILLA_CPU_BENCH_SOURCES_
        bench/main.cpp
        bench/find.cpp
        bench/token.cpp
        bench/sequence.cpp
        bench/memory.cpp
        bench/cipher.cpp
        bench/container.cpp
        bench/levenshtein.cpp
        bench/overlap.cpp
        bench/substrings.cpp
        bench/utf8_traverse.cpp
        bench/utf8_scan.cpp
        bench/utf8_segment.cpp
        bench/utf8_norm.cpp
        bench/utf8_uncased.cpp
        ${STRINGZILLA_CROSS_BENCH_SOURCES_}
    )
    define_cpu_launcher(stringzilla_cpu_bench stringzilla_static COMPILE ${STRINGZILLA_CPU_BENCH_SOURCES_})
    # Header-only builds have no dispatch points to time, but their kernels compile inline, which also reaches the
    # private helpers the library keeps to itself, like the SIMD sorts of pgrams.
    define_cpu_launcher(
        stringzilla_cpu_header_bench stringzilla_header COMPILE bench/main.cpp ${STRINGZILLA_CROSS_BENCH_SOURCES_}
    )

    if (STRINGZILLA_BUILD_CUDA)
        define_gpu_launcher(stringzilla_cuda_bench CUDA bench/cross_cuda.cu)
    endif ()
    if (STRINGZILLA_BUILD_ROCM)
        define_gpu_launcher(stringzilla_rocm_bench HIP bench/cross_rocm.hip)
    endif ()
    if (STRINGZILLA_BUILD_METAL)
        define_gpu_launcher(stringzilla_metal_bench CXX bench/cross_metal.cpp)
        target_compile_definitions(stringzilla_metal_bench PRIVATE STRINGZILLA_WITH_METAL=1)
    endif ()
endif ()

if (STRINGZILLA_BUILD_TEST)
    # The family files test the dispatch points and `stringzilla.hpp`, and each `cross_<arch>.cpp` holds its
    # architecture's kernels to the serial ones, a section per capability, skipping those this CPU lacks.
    set(STRINGZILLA_CROSS_TEST_SOURCES_
        test/cross_serial.cpp
        test/cross_x8664.cpp
        test/cross_arm64.cpp
        test/cross_riscv64.cpp
        test/cross_loongarch64.cpp
        test/cross_ppc64.cpp
        test/cross_wasm.cpp
    )
    set(STRINGZILLA_CPU_TEST_SOURCES_
        test/main_cpu.cpp
        test/hash.cpp
        test/cipher.cpp
        test/sort.cpp
        test/levenshtein.cpp
        test/overlap.cpp
        test/substrings.cpp
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
        ${STRINGZILLA_CROSS_TEST_SOURCES_}
    )

    define_cpu_launcher(stringzilla_cpu_test stringzilla_static COMPILE ${STRINGZILLA_CPU_TEST_SOURCES_})
    add_test(NAME stringzilla_cpu_test COMMAND stringzilla_cpu_test)
    # Cross builds link their executables statically, to run them under an emulator without the target's loader.
    if (TARGET stringzilla_shared AND NOT CMAKE_CROSSCOMPILING)
        define_cpu_launcher(stringzilla_cpu_shared_test stringzilla_shared SERIAL ${STRINGZILLA_CPU_TEST_SOURCES_})
        add_test(NAME stringzilla_cpu_shared_test COMMAND stringzilla_cpu_shared_test)
    endif ()
    # The family files drive `stringzilla.hpp`, which always goes through the library, so the header-only test keeps
    # the kernel cross-checks and the dispatch-point stubs.
    define_cpu_launcher(
        stringzilla_cpu_header_test stringzilla_header COMPILE test/main_cpu.cpp ${STRINGZILLA_CROSS_TEST_SOURCES_}
    )
    add_test(NAME stringzilla_cpu_header_test COMMAND stringzilla_cpu_header_test)

    # One GPU test per vendor, each checking that vendor's kernels by name and the dispatch points over them.
    if (STRINGZILLA_BUILD_CUDA)
        define_gpu_launcher(stringzilla_cuda_test CUDA test/main_cuda.cu test/cross_cuda.cu)
        add_test(NAME stringzilla_cuda_test COMMAND stringzilla_cuda_test)
    endif ()
    if (STRINGZILLA_BUILD_ROCM)
        define_gpu_launcher(stringzilla_rocm_test HIP test/main_rocm.hip test/cross_rocm.hip)
        add_test(NAME stringzilla_rocm_test COMMAND stringzilla_rocm_test)
    endif ()
    if (STRINGZILLA_BUILD_METAL)
        # The kernels travel as embedded `.metal` source and compile on the device at first use, so the test only
        # needs to see the Metal declarations the library defines.
        define_gpu_launcher(stringzilla_metal_test CXX test/main_metal.cpp test/cross_metal.cpp)
        target_compile_definitions(stringzilla_metal_test PRIVATE STRINGZILLA_WITH_METAL=1)
        add_test(NAME stringzilla_metal_test COMMAND stringzilla_metal_test)
    endif ()
endif ()

if (STRINGZILLA_INSTALL)
    include(CMakePackageConfigHelpers)

    # Every installed library belongs to an `EXPORT` set - a `.so` with no imported target leaves consumers
    # hand-rolling `find_library`.
    # Destinations come from the `GNUInstallDirs` defaults, so each call names only its target and export set.
    install(TARGETS stringzilla_header stringzilla_static EXPORT stringzillaTargets)
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

    # SwiftPM reads `module.modulemap` from the XCFramework or artifact bundle `stringzilla_swift` packs. At an installed
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
