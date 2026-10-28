# StringZilla CMakeLists.txt
#
# This file defines several library build & installation targets:
#
# * stringzilla_header: A header-only library with the StringZilla C and C++ headers.
# * stringzilla_static: A static library with dynamic SIMD dispatch, and the GPU backends STRINGZILLA_CUDA_ARCH,
#   STRINGZILLA_ROCM_ARCH and STRINGZILLA_METAL_ARCH ask for, side by side.
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
cmake_minimum_required(VERSION 3.28)

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

# Release compiles at -O2 rather than CMake's -O3, set here rather than per target so that presets and parent projects
# can still lower RelWithDebInfo and Debug, which keep CMake's defaults.
if (NOT MSVC)
    set(CMAKE_C_FLAGS_RELEASE "-O2 -DNDEBUG")
    set(CMAKE_CXX_FLAGS_RELEASE "-O2 -DNDEBUG")
endif ()

# Standards are floors each target declares, so a parent project's units and the fetched dependencies build at the
# standard they pick for themselves.
set(CMAKE_C_EXTENSIONS OFF)
set(CMAKE_CXX_EXTENSIONS OFF)
# No target uses C++20 modules, and a Clang installed without `clang-scan-deps` would fail every C++ unit.
set(CMAKE_CXX_SCAN_FOR_MODULES OFF)

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

# Enable the in-repo Git hooks. Git will not auto-run hooks shipped inside a clone (that would let any cloned repo
# execute code), so point `core.hooksPath` at `.githooks` here - configuring the build arms the `pre-commit` checks.
# Skipped when built as a subproject or without `.git` (tarball / export builds).
if (PROJECT_IS_TOP_LEVEL AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/.git")
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
option(STRINGZILLA_INSTALL "Install CMake targets" ${PROJECT_IS_TOP_LEVEL})
option(STRINGZILLA_BUILD_TEST "Compile a native unit test in C++" OFF)
option(STRINGZILLA_BUILD_BENCH "Compile a native benchmark in C++" OFF)
option(STRINGZILLA_BUILD_SHARED "Compile a dynamic library" ${PROJECT_IS_TOP_LEVEL})
option(STRINGZILLA_BUILD_PYTHON "Compile the CPython extension over the static library" OFF)
option(STRINGZILLA_BUILD_NODE "Compile the Node addon over the static library; cmake-js switches it on" OFF)
option(STRINGZILLA_BUILD_SWIFT "Package the static library for SwiftPM, as an XCFramework or an artifact bundle" OFF)
set(STRINGZILLA_CUDA_ARCH
    ""
    CACHE STRING "NVIDIA code the libraries, tests and benchmarks build for: all, native or one"
)
set(STRINGZILLA_ROCM_ARCH
    ""
    CACHE STRING "AMD target the libraries, tests and benchmarks build for: all, native or one"
)
set(STRINGZILLA_METAL_ARCH
    ""
    CACHE STRING "Apple GPU family the Metal unit serves: all or one"
)
option(STRINGZILLA_WERROR "Treat compiler warnings as errors" OFF)
set(STRINGZILLA_BUILD_SANITIZERS
    "off"
    CACHE STRING "Sanitizers for StringZilla's tests, benchmarks and archives: off, address with undefined, or thread"
)
set_property(CACHE STRINGZILLA_BUILD_SANITIZERS PROPERTY STRINGS off address thread)
if (NOT STRINGZILLA_BUILD_SANITIZERS MATCHES "^(off|address|thread)$")
    message(FATAL_ERROR "STRINGZILLA_BUILD_SANITIZERS is off, address or thread, not '${STRINGZILLA_BUILD_SANITIZERS}'")
endif ()

# Clang source-based coverage for the header-only library. Recipe (single-arch, native backends only): cmake -B
# build_cov -DSTRINGZILLA_BUILD_COVERAGE=1 -DCMAKE_CXX_COMPILER=clang++ cmake --build
# build_cov --target stringzilla_test -j LLVM_PROFILE_FILE=/tmp/sz.profraw ./build_cov/stringzilla_test
# llvm-profdata merge -sparse /tmp/sz.profraw -o /tmp/sz.profdata llvm-cov report ./build_cov/stringzilla_test
# -instr-profile=/tmp/sz.profdata include/stringzilla/ `-march=native` compiles out the non-native backends, so a
# faithful picture needs `llvm-profdata merge` across per-ISA builds (native x86 + qemu-aarch64 for NEON/SVE + the RVV
# qemu toolchain).
option(STRINGZILLA_BUILD_COVERAGE "Instrument with Clang source-based coverage" OFF)
# Seeded from the environment at the first configure, so `cargo` and `cmake-js` honour `STRINGZILLA_MARCH=native`.
set(STRINGZILLA_MARCH
    "$ENV{STRINGZILLA_MARCH}"
    CACHE STRING "The `-march` floor, like x86-64-v3 or native, or a WebAssembly module's SIMD capability, like v128"
)
option(STRINGZILLA_MULTIVERSIONED
       "Keep every capability above the floor, or only the best one STRINGZILLA_MARCH implies" ON
)

include(cmake/stringzilla_platform.cmake)

# One compile probe per capability under `probes/`, defining `STRINGZILLA_TARGET_<CAPABILITY>` as the toolchain
# compiles it through `stringzilla::cpu_capabilities_compiled`, and as 0 through
# `stringzilla_cpu_capabilities_disabled_`. Below the options, as a native build also asks which ones this host runs.
include(cmake/stringzilla_probes.cmake)

include(cmake/stringzilla_gpu.cmake)

if (STRINGZILLA_MARCH)
    set(stringzilla_floor_ "${STRINGZILLA_MARCH}")
else ()
    set(stringzilla_floor_ "the portable floor")
endif ()
get_property(stringzilla_capabilities_enabled_ GLOBAL PROPERTY stringzilla_capabilities_enabled)
list(JOIN stringzilla_capabilities_enabled_ " " stringzilla_capabilities_summary_)
if (STRINGZILLA_MULTIVERSIONED)
    message(STATUS "StringZilla: multiversioned above ${stringzilla_floor_}")
else ()
    message(STATUS "StringZilla: monoversioned for ${stringzilla_floor_}: ${stringzilla_capabilities_summary_}")
endif ()

# Configuration
include(GNUInstallDirs)
set(stringzilla_include_build_dir_ "${PROJECT_SOURCE_DIR}/include/")
set(stringzilla_include_install_dir_ "${CMAKE_INSTALL_INCLUDEDIR}")

# CTest machinery only for top-level builds: a parent project decides its own testing story, and
# `add_test` calls below are inert without `enable_testing()`.
if (PROJECT_IS_TOP_LEVEL)
    include(CTest)
    enable_testing()
endif ()

if (MSVC)
    # Remove /RTC* from MSVC debug flags by default (it will be added back in the stringzilla_set_compiler_flags
    # function) Because /RTC* cannot be used without the crt so it needs to be disabled for that specific target
    string(REGEX REPLACE "/RTC[^ ]*" "" CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG}")
    string(REGEX REPLACE "/RTC[^ ]*" "" CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG}")
endif ()

include(cmake/stringzilla_compiler_flags.cmake)

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
function (stringzilla_add_gpu_suite target_ language_ source_)
    add_library(${target_} OBJECT ${source_})
    stringzilla_set_compiler_flags(${target_} 20)
    set_target_properties(
        ${target_}
        PROPERTIES CUDA_ARCHITECTURES "${stringzilla_cuda_resolved_architectures_}"
                   HIP_ARCHITECTURES "${stringzilla_rocm_resolved_architectures_}"
                   HIP_STANDARD 20
    )
    if (MSVC AND language_ STREQUAL "CUDA")
        target_compile_definitions(${target_} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:FMT_UNICODE=0>)
    endif ()
    target_link_libraries(
        ${target_} PRIVATE stringzilla_cpu_capabilities_disabled_ stringzilla_cuda_capabilities_compiled_
                           stringzilla_static fmt::fmt
    )
endfunction ()

# One CPU test or benchmark linking `library_`, whose capability `verdicts_` declare every kernel the
# toolchain builds (COMPILE), as the harnesses skip those this CPU lacks, or turn them all off, leaving
# the dispatch points (SERIAL).
function (stringzilla_add_cpu_launcher executable_ library_ verdicts_)
    add_executable(${executable_} ${ARGN})
    stringzilla_set_compiler_flags(${executable_} 20)
    if (verdicts_ STREQUAL "SERIAL")
        target_link_libraries(${executable_} PRIVATE stringzilla_cpu_capabilities_disabled_)
    else ()
        target_link_libraries(${executable_} PRIVATE stringzilla_cpu_capabilities_compiled)
    endif ()
    target_link_libraries(${executable_} PRIVATE ${library_} fmt::fmt)
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
# enabled capability's `c/target/<capability>.c` defines that capability's kernels once, and each
# `c/dispatch/<family>.c` holds its family's lists, dispatch points and finder. Globbed, so a new unit needs no entry;
# WebAssembly enables its capabilities through STRINGZILLA_MARCH, so its units follow the platform, and the Metal
# unit joins below, with its own frameworks, beside the CUDA and ROCm ones.
file(
    GLOB stringzilla_globbed_units_ CONFIGURE_DEPENDS
    RELATIVE ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/c/target/*.c ${PROJECT_SOURCE_DIR}/c/dispatch/*.c
)
set(stringzilla_shim_sources_ c/stringzilla.c)
foreach (unit_ IN LISTS stringzilla_globbed_units_)
    if (unit_ MATCHES "^c/target/([a-z0-9]+)")
        set(capability_ ${CMAKE_MATCH_1})
        if (NOT capability_ STREQUAL "serial"
            AND NOT (stringzilla_arch_wasm_ AND capability_ MATCHES "^v128")
            AND NOT stringzilla_target_${capability_}_enabled
        )
            continue()
        endif ()
    endif ()
    list(APPEND stringzilla_shim_sources_ ${unit_})
endforeach ()

# Defines `stringzilla_shared` or `stringzilla_static` over the sources that follow.
function (stringzilla_add_library target_ public_name_ library_type_)
    add_library(${target_} ${library_type_} ${ARGN})

    # The library, unlike the headers, may use C11: `c/stringzilla.c` caches detection in a <stdatomic.h> atomic, which
    # MSVC gates behind a flag.
    target_compile_features(${target_} PRIVATE c_std_11)
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

    # The units hold the directory's architecture floor, and the dispatch points pick each SIMD capability by the mask
    # each call passes: every capability the toolchain compiles, which `sz_capabilities_enabled_cpu` masks to the CPU's.
    stringzilla_set_compiler_flags(${target_} "")
    target_link_libraries(${target_} PRIVATE $<BUILD_INTERFACE:stringzilla_cpu_capabilities_compiled>)

endfunction ()

# The dispatch core as an archive, which the tests, the benchmarks, and the Go and Python bindings link - the
# cross-compiled test runs are `-static` executables, which cannot take a `.so` at all - and as a shared library. A
# parent project builds the archive only if it links it.
stringzilla_add_library(stringzilla_static static STATIC ${stringzilla_shim_sources_})
if (NOT PROJECT_IS_TOP_LEVEL)
    set_target_properties(stringzilla_static PROPERTIES EXCLUDE_FROM_ALL ON)
endif ()
if (STRINGZILLA_BUILD_SHARED)
    # Both compile alike, so the shared library links the archive's objects, except where sanitizers, a monoversioned
    # build's link-time optimization, or MSVC's debug checks apply to the archive alone.
    set(shared_sources_ $<TARGET_OBJECTS:stringzilla_static>)
    if (MSVC
        OR NOT STRINGZILLA_MULTIVERSIONED
        OR NOT STRINGZILLA_BUILD_SANITIZERS STREQUAL "off"
    )
        set(shared_sources_ ${stringzilla_shim_sources_})
    endif ()
    stringzilla_add_library(stringzilla_shared shared SHARED ${shared_sources_})
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

# A monoversioned build folds each dispatch table to the kernel of the best enabled capability implementing its
# operation. Its static archive links as bitcode, so the folded tables leave the kernels they skip out of the products'
# links; any other static archive ships machine code any toolchain, cGo included, can link. MSVC's `/GL` objects hide
# the symbols `WINDOWS_EXPORT_ALL_SYMBOLS` must list.
if (NOT STRINGZILLA_MULTIVERSIONED)
    foreach (library_ IN LISTS stringzilla_libraries_)
        target_compile_definitions(${library_} PRIVATE STRINGZILLA_MULTIVERSIONED=0)
    endforeach ()
    include(CheckIPOSupported)
    check_ipo_supported(RESULT stringzilla_ipo_supported_ OUTPUT stringzilla_ipo_support_output_)
    if (stringzilla_ipo_supported_ AND NOT MSVC)
        set_target_properties(stringzilla_static PROPERTIES INTERPROCEDURAL_OPTIMIZATION TRUE)
    else ()
        message(STATUS "Link-time optimization not available: ${stringzilla_ipo_support_output_}")
    endif ()
endif ()

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
    stringzilla_set_compiler_flags(stringzilla_${unit_}_ "")
    target_compile_definitions(stringzilla_${unit_}_ PRIVATE "STRINGZILLA_WITH_LIBC=1" "${definition_}")
    # A monoversioned build links the host side as bitcode too, but only bitcode the C units' compiler can read back.
    if (NOT STRINGZILLA_MULTIVERSIONED
        AND stringzilla_ipo_supported_
        AND CMAKE_CUDA_HOST_COMPILER_ID STREQUAL CMAKE_C_COMPILER_ID
        AND CMAKE_CUDA_HOST_COMPILER_VERSION STREQUAL CMAKE_C_COMPILER_VERSION
    )
        if (CMAKE_C_COMPILER_ID STREQUAL "Clang")
            target_compile_options(stringzilla_${unit_}_ PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-flto=thin>")
        elseif (CMAKE_C_COMPILER_ID STREQUAL "GNU")
            target_compile_options(stringzilla_${unit_}_ PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-flto>")
        endif ()
    endif ()
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

if (STRINGZILLA_CUDA_ARCH)
    # One unit per tier, compiled only for the codes running it, so no cubin carries another tier's kernels.
    # A capability switched off keeps its unit out, as no code then carries its kernels.
    set(stringzilla_hopper_unit_codes_)
    set(stringzilla_blackwell_unit_codes_)
    if (stringzilla_target_hopper_enabled)
        stringzilla_select_gpu_codes(
            stringzilla_hopper_unit_codes_ "${stringzilla_cuda_resolved_architectures_}"
            "${stringzilla_target_hopper_architectures}"
        )
    endif ()
    if (stringzilla_target_blackwell_enabled)
        stringzilla_select_gpu_codes(
            stringzilla_blackwell_unit_codes_ "${stringzilla_cuda_resolved_architectures_}"
            "${stringzilla_target_blackwell_architectures}"
        )
    endif ()
    string(REGEX REPLACE "-(real|virtual)" "" stringzilla_cuda_codes_ "${stringzilla_cuda_resolved_architectures_}")
    stringzilla_select_gpu_codes(
        stringzilla_cuda_unit_codes_ "${stringzilla_cuda_resolved_architectures_}" "${stringzilla_cuda_codes_}"
    )
    stringzilla_add_gpu_unit(c/target/cuda.cu "${stringzilla_cuda_unit_codes_}" "STRINGZILLA_ARCH_CUDA_=1" CUDA::cudart)
    stringzilla_add_gpu_unit(
        c/target/hopper.cu "${stringzilla_hopper_unit_codes_}" "STRINGZILLA_ARCH_CUDA_=1" CUDA::cudart
    )
    stringzilla_add_gpu_unit(
        c/target/blackwell.cu "${stringzilla_blackwell_unit_codes_}" "STRINGZILLA_ARCH_CUDA_=1" CUDA::cudart
    )
endif ()
if (STRINGZILLA_ROCM_ARCH)
    stringzilla_add_gpu_unit(
        c/target/rocm.hip "${stringzilla_rocm_resolved_architectures_}" "STRINGZILLA_ARCH_ROCM_=1" hip::host
    )
endif ()
if (STRINGZILLA_METAL_ARCH)
    # Metal compiles on the device, so its codes name the families the unit serves.
    stringzilla_add_gpu_unit(
        c/target/metal.c "${stringzilla_metal_resolved_architectures_}" "STRINGZILLA_ARCH_METAL_=1" "-framework Metal"
        "-framework Foundation"
    )
    target_compile_options(stringzilla_metal_ PRIVATE ${stringzilla_embed_options_})
    foreach (library_ IN LISTS stringzilla_libraries_)
        target_compile_options(${library_} PUBLIC ${stringzilla_embed_options_})
    endforeach ()
endif ()

# The `stringzilla` extension module, which scikit-build-core builds and packages from `pyproject.toml`, installed at the
# wheel's root, where `import stringzilla` finds it.
if (STRINGZILLA_BUILD_PYTHON)
    find_package(Python 3.12 REQUIRED COMPONENTS Interpreter Development.Module)
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
    set_target_properties(stringzilla_python PROPERTIES OUTPUT_NAME stringzilla)
    target_compile_features(stringzilla_python PRIVATE c_std_11)
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
    # The module's one export is its init function; the static library's kernels stay inside it.
    set_target_properties(stringzilla_python PROPERTIES C_VISIBILITY_PRESET hidden)
    if (APPLE)
        target_link_options(stringzilla_python PRIVATE "LINKER:-exported_symbol,_PyInit_stringzilla")
    elseif (NOT MSVC)
        target_link_options(stringzilla_python PRIVATE "LINKER:--exclude-libs,ALL")
    endif ()
    install(
        TARGETS stringzilla_python
        DESTINATION .
        COMPONENT stringzilla_python
    )
    install(
        FILES python/stringzilla.pyi
        DESTINATION .
        COMPONENT stringzilla_python
    )
endif ()

# The `stringzilla.node` addon, which cmake-js builds from `package.json`. It stays off
# `stringzilla_set_compiler_flags`, whose output directories would move the Windows addon and whose warnings and
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
    )
    target_compile_features(stringzilla_node PRIVATE c_std_11)
    target_include_directories(stringzilla_node PRIVATE ${CMAKE_JS_INC})
    # The BigInt conversions need Node-API 6.
    target_compile_definitions(stringzilla_node PRIVATE NAPI_VERSION=8)
    target_link_libraries(
        stringzilla_node PRIVATE stringzilla_static ${CMAKE_JS_LIB}
                                 $<$<AND:$<BOOL:${OpenMP_C_FOUND}>,$<PLATFORM_ID:Linux,FreeBSD>>:OpenMP::OpenMP_C>
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
            if (STRINGZILLA_CUDA_ARCH)
                set(language_ CUDA)
            else ()
                set_source_files_properties(${source_} PROPERTIES LANGUAGE CXX)
            endif ()
        elseif (vendor_ STREQUAL "rocm")
            set(source_ "${suite_}/cross_rocm.hip")
            if (STRINGZILLA_ROCM_ARCH)
                set(language_ HIP)
            else ()
                set_source_files_properties(${source_} PROPERTIES LANGUAGE CXX)
            endif ()
        else ()
            set(source_ "${suite_}/cross_metal.cpp")
        endif ()
        set(unit_ "stringzilla_${vendor_}_${suite_}_")
        stringzilla_add_gpu_suite(${unit_} ${language_} ${source_})
        if (vendor_ STREQUAL "metal" AND STRINGZILLA_METAL_ARCH)
            target_compile_definitions(${unit_} PRIVATE STRINGZILLA_ARCH_METAL_=1)
        endif ()
        target_sources(stringzilla_${suite_} PRIVATE $<TARGET_OBJECTS:${unit_}>)
        if (suite_ STREQUAL "test" AND TARGET stringzilla_shared_test)
            target_sources(stringzilla_shared_test PRIVATE $<TARGET_OBJECTS:${unit_}>)
        endif ()
    endforeach ()
endforeach ()

include(cmake/stringzilla_install.cmake)
