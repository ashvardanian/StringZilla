# cmake/sz_isa_probe.cmake — shared ISA probe infrastructure over the checked-in `probes/` sources, the
# same files the Cargo build compiles. A per-tier compile probe asks what the toolchain can emit; one
# executed machine probe asks what this machine can run. Runtime-dispatched targets enable everything
# compilable and trust the load-time dispatch table; comptime-dispatched targets require a tier to pass
# both probes. Per-architecture modules declare the tiers and set `SZ_ISA_TIERS`, newest first.

# Try-compile one tier's probe into the cached `<result>`, conventionally `SZ_CAN_COMPILE_<TIER>`:
#
#   sz_isa_probe_(<result> SOURCE <probes/file.c> [GNU_FLAGS <flags...>] [MSVC_FLAGS <flags...>])
#
# `GNU_FLAGS` reach GCC and Clang, both `GNU` frontend variants in CMake terms; only the wasm tiers need
# any. The probe file is compiled as-is, byte-identical to what `build.rs` sees: a string round-trip
# would swallow the backslash line-continuations inside multi-line pragmas and mis-fail the probe. The
# Release configuration pin is function-scoped, so Debug-only sanitizer runtimes cannot interfere.
function (sz_isa_probe_ result_)
    if (DEFINED ${result_})
        return()
    endif ()
    cmake_parse_arguments(PARSE_ARGV 1 sz_arg "" "SOURCE" "GNU_FLAGS;MSVC_FLAGS")
    if (NOT sz_arg_SOURCE)
        message(FATAL_ERROR "sz_isa_probe_(${result_}) requires SOURCE <probes/file.c>")
    endif ()
    if (sz_arg_UNPARSED_ARGUMENTS)
        message(FATAL_ERROR "sz_isa_probe_(${result_}) got unexpected arguments: ${sz_arg_UNPARSED_ARGUMENTS}")
    endif ()
    if (MSVC)
        set(sz_probe_flags_ "${sz_arg_MSVC_FLAGS}")
    else ()
        set(sz_probe_flags_ "${sz_arg_GNU_FLAGS}")
    endif ()
    set(CMAKE_TRY_COMPILE_CONFIGURATION "Release")
    try_compile(
        ${result_} ${CMAKE_BINARY_DIR}/sz_probes
        ${CMAKE_CURRENT_SOURCE_DIR}/${sz_arg_SOURCE}
        COMPILE_DEFINITIONS "${sz_probe_flags_}" C_STANDARD 99
        OUTPUT_VARIABLE sz_probe_output_
    )
    set(${result_}
        ${${result_}}
        CACHE INTERNAL "The toolchain compiles ${sz_arg_SOURCE}"
    )
    if (${result_})
        message(STATUS "Performing Test ${result_} - Success")
    else ()
        message(STATUS "Performing Test ${result_} - Failed")
    endif ()
endfunction ()

# Compile-probe `probes/runtime_detection.c` into the cached `SZ_RUNTIME_DETECTABLE`: whether the built
# library performs real runtime capability detection, the header-owned
# `SZ_CAPABILITIES_RUNTIME_DETECTABLE_`. Without it, dispatch tables mirror the compile-time mask and
# tier selection must stay with the toolchain flags and `types.h` auto-detection.
function (sz_runtime_detectable_)
    if (DEFINED SZ_RUNTIME_DETECTABLE)
        return()
    endif ()
    try_compile(
        sz_detectable_ ${CMAKE_BINARY_DIR}/sz_probes
        ${CMAKE_CURRENT_SOURCE_DIR}/probes/runtime_detection.c
        COMPILE_DEFINITIONS "-DSZ_AVOID_LIBC=0"
        CMAKE_FLAGS "-DINCLUDE_DIRECTORIES=${CMAKE_CURRENT_SOURCE_DIR}/include" C_STANDARD 99
        OUTPUT_VARIABLE sz_probe_output_
    )
    if (sz_detectable_)
        set(SZ_RUNTIME_DETECTABLE
            1
            CACHE INTERNAL "Runtime capability detection exists for this target"
        )
    else ()
        set(SZ_RUNTIME_DETECTABLE
            0
            CACHE INTERNAL "Runtime capability detection exists for this target"
        )
        message(STATUS "No runtime capability detection for this target; SIMD tiers follow the toolchain flags")
    endif ()
endfunction ()

# Compile and run `probes/run_capabilities.c`, caching the machine's comma-separated tier tokens in
# `SZ_MACHINE_CAPABILITIES`, for example "serial,neon,neonaes". Left empty when the answer is
# unknowable, like cross-compiling without an emulator or any probe failure; callers then fall back to
# `types.h` auto-detection under the target's own `-march` flags.
function (sz_machine_capabilities_)
    if (DEFINED SZ_MACHINE_CAPABILITIES)
        return()
    endif ()
    if (CMAKE_CROSSCOMPILING AND NOT CMAKE_CROSSCOMPILING_EMULATOR)
        set(SZ_MACHINE_CAPABILITIES
            ""
            CACHE INTERNAL "Tier tokens the build machine can run"
        )
        message(STATUS "Machine capabilities: unknown, cross-compiling without an emulator")
        return()
    endif ()
    try_run(
        sz_run_exit_ sz_run_compiled_ ${CMAKE_BINARY_DIR}/sz_probes
        ${CMAKE_CURRENT_SOURCE_DIR}/probes/run_capabilities.c
        CMAKE_FLAGS "-DINCLUDE_DIRECTORIES=${CMAKE_CURRENT_SOURCE_DIR}/include"
        RUN_OUTPUT_VARIABLE sz_run_output_
    )
    if (sz_run_compiled_ AND sz_run_exit_ EQUAL 0)
        string(STRIP "${sz_run_output_}" sz_run_output_)
        set(SZ_MACHINE_CAPABILITIES
            "${sz_run_output_}"
            CACHE INTERNAL "Tier tokens the build machine can run"
        )
        message(STATUS "Machine capabilities: ${SZ_MACHINE_CAPABILITIES}")
    else ()
        set(SZ_MACHINE_CAPABILITIES
            ""
            CACHE INTERNAL "Tier tokens the build machine can run"
        )
        message(STATUS "Machine capabilities: unknown, run probe unavailable")
    endif ()
endfunction ()
