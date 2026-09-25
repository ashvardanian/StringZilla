# cmake/sz_isa_probe.cmake — shared ISA probe infrastructure over the checked-in `probes/` sources, the
# same files the Cargo build compiles. A per-capability compile probe asks what the toolchain can emit;
# one executed machine probe asks what this machine can run. Runtime-dispatched targets enable everything
# compilable and trust the load-time dispatch table; comptime-dispatched targets require a capability to
# pass both probes. Per-architecture modules call `sz_instruction_set_probe_` per tier, newest first, then
# `sz_build_instruction_set_definitions_`, which fills the cached `sz_compile_definitions_` and
# `sz_run_definitions_` lists of `STRINGZILLA_TARGET_<TIER>=0/1` that the targets take.

include_guard(GLOBAL)

# Probe verdicts are cached, but they answer for one set of compiler flags: changing them in the same
# build tree must re-ask every question, or the wrong kernels get enabled with no message. A fresh tree
# holds nothing stale, and sweeping it would erase a preset `-D sz_target_<tier>_compiles` verdict.
set(sz_probe_key_ "${CMAKE_C_COMPILER}|${CMAKE_C_FLAGS}|${CMAKE_TOOLCHAIN_FILE}")
if (NOT "${STRINGZILLA_PROBE_KEY}" STREQUAL "${sz_probe_key_}")
    if (DEFINED STRINGZILLA_PROBE_KEY)
        message(STATUS "Toolchain changed - re-running the ISA probes")
        get_cmake_property(sz_cache_entries_ CACHE_VARIABLES)
        list(FILTER sz_cache_entries_ INCLUDE REGEX "^sz_target_.*_(compiles|flags)$")
        foreach (sz_cache_entry_ IN LISTS sz_cache_entries_)
            unset(${sz_cache_entry_} CACHE)
        endforeach ()
    endif ()
    unset(STRINGZILLA_HAS_RUNTIME_DETECTION_ CACHE)
    unset(STRINGZILLA_MACHINE_CAPABILITIES CACHE)
    set(STRINGZILLA_PROBE_KEY
        "${sz_probe_key_}"
        CACHE INTERNAL "Toolchain the cached probe verdicts answer for"
    )
endif ()
set(sz_compile_definitions_
    ""
    CACHE INTERNAL "STRINGZILLA_TARGET_<TIER>=0/1 verdicts for runtime-dispatched units"
)
set(sz_run_definitions_
    ""
    CACHE INTERNAL "STRINGZILLA_TARGET_<TIER>=0/1 verdicts for comptime-dispatched executables"
)

# Try-compile one capability's probe, caching the verdict as `sz_target_<tier>_compiles` and the flags it
# took as `sz_target_<tier>_flags`; a preset `-D sz_target_<tier>_compiles=0` skips the probe:
#
#   sz_instruction_set_probe_(<TIER> SOURCE <probes/file.c> [GNU_FLAGS <flags...>] [MSVC_FLAGS <flags...>])
#
# `GNU_FLAGS` reach GCC and Clang, both `GNU` frontend variants in CMake terms; only the wasm capabilities
# need any. The probe file is compiled as-is, byte-identical to what `build.rs` sees: a string round-trip
# would swallow the backslash line-continuations inside multi-line pragmas and mis-fail the probe. The
# Release configuration pin is function-scoped, so Debug-only sanitizer runtimes cannot interfere.
function (sz_instruction_set_probe_ capability_)
    string(TOLOWER "${capability_}" tier_lowercase_)
    if (DEFINED sz_target_${tier_lowercase_}_compiles)
        return()
    endif ()
    cmake_parse_arguments(PARSE_ARGV 1 sz_arg "" "SOURCE" "GNU_FLAGS;MSVC_FLAGS")
    if (NOT sz_arg_SOURCE)
        message(FATAL_ERROR "sz_instruction_set_probe_(${capability_}) requires SOURCE <probes/file.c>")
    endif ()
    if (sz_arg_UNPARSED_ARGUMENTS)
        message(
            FATAL_ERROR
                "sz_instruction_set_probe_(${capability_}) got unexpected arguments: ${sz_arg_UNPARSED_ARGUMENTS}"
        )
    endif ()
    if (MSVC)
        set(sz_probe_flags_ "${sz_arg_MSVC_FLAGS}")
    else ()
        set(sz_probe_flags_ "${sz_arg_GNU_FLAGS}")
    endif ()
    set(CMAKE_TRY_COMPILE_CONFIGURATION "Release")
    try_compile(
        sz_probe_succeeded_ ${CMAKE_BINARY_DIR}/sz_probes
        ${CMAKE_CURRENT_SOURCE_DIR}/${sz_arg_SOURCE}
        COMPILE_DEFINITIONS "${sz_probe_flags_}" C_STANDARD 99
        OUTPUT_VARIABLE sz_probe_output_
    )
    set(sz_probe_verdict_ 0)
    set(sz_probe_outcome_ "Failed")
    if (sz_probe_succeeded_)
        set(sz_probe_verdict_ 1)
        set(sz_probe_outcome_ "Success")
    endif ()
    set(sz_target_${tier_lowercase_}_compiles
        ${sz_probe_verdict_}
        CACHE INTERNAL "Whether the ${capability_} probe compiles with this toolchain"
    )
    set(sz_target_${tier_lowercase_}_flags
        "${sz_probe_flags_}"
        CACHE INTERNAL "Flags the ${capability_} probe compiled under"
    )
    message(STATUS "Performing ISA probe ${capability_} - ${sz_probe_outcome_}")
endfunction ()

# Compile-probe `probes/runtime_detection.c` into the cached `STRINGZILLA_HAS_RUNTIME_DETECTION_`: whether the built
# library performs real runtime capability detection, the header-owned
# `STRINGZILLA_HAS_RUNTIME_DETECTION_`. Without it, dispatch tables mirror the compile-time mask and
# capability selection must stay with the toolchain flags and `types.h` auto-detection.
function (sz_runtime_detectable_)
    if (DEFINED STRINGZILLA_HAS_RUNTIME_DETECTION_)
        return()
    endif ()
    try_compile(
        sz_detectable_ ${CMAKE_BINARY_DIR}/sz_probes
        ${CMAKE_CURRENT_SOURCE_DIR}/probes/runtime_detection.c
        COMPILE_DEFINITIONS "-DSTRINGZILLA_WITH_LIBC=1"
        CMAKE_FLAGS "-DINCLUDE_DIRECTORIES=${CMAKE_CURRENT_SOURCE_DIR}/include" C_STANDARD 99
        OUTPUT_VARIABLE sz_probe_output_
    )
    if (sz_detectable_)
        set(STRINGZILLA_HAS_RUNTIME_DETECTION_
            1
            CACHE INTERNAL "Runtime capability detection exists for this target"
        )
    else ()
        set(STRINGZILLA_HAS_RUNTIME_DETECTION_
            0
            CACHE INTERNAL "Runtime capability detection exists for this target"
        )
        message(STATUS "No runtime capability detection for this target; capabilities follow the toolchain flags")
    endif ()
endfunction ()

# Compile and run `probes/run_capabilities.c`, caching the machine's comma-separated capability tokens in
# `STRINGZILLA_MACHINE_CAPABILITIES`, for example "serial,neon,neonaes". Left empty when the answer is
# unknowable, like cross-compiling without an emulator or any probe failure; callers then fall back to
# `types.h` auto-detection under the target's own `-march` flags.
function (sz_machine_capabilities_)
    if (DEFINED STRINGZILLA_MACHINE_CAPABILITIES)
        return()
    endif ()
    if (CMAKE_CROSSCOMPILING AND NOT CMAKE_CROSSCOMPILING_EMULATOR)
        set(STRINGZILLA_MACHINE_CAPABILITIES
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
        set(STRINGZILLA_MACHINE_CAPABILITIES
            "${sz_run_output_}"
            CACHE INTERNAL "Tier tokens the build machine can run"
        )
        message(STATUS "Machine capabilities: ${STRINGZILLA_MACHINE_CAPABILITIES}")
    else ()
        set(STRINGZILLA_MACHINE_CAPABILITIES
            ""
            CACHE INTERNAL "Tier tokens the build machine can run"
        )
        message(STATUS "Machine capabilities: unknown, run probe unavailable")
    endif ()
endfunction ()

# Fold the cached verdicts into `sz_compile_definitions_`, the `STRINGZILLA_TARGET_<TIER>=0/1` list for
# runtime-dispatched units, and `sz_run_definitions_`, the same for comptime-dispatched executables that
# must also run here. A list stays empty when its answer is unknowable, so `types.h` decides under the
# unit's own flags; `-D STRINGZILLA_TARGET_<TIER>=0/1` outranks both lists, except past a failed compile probe.
function (sz_build_instruction_set_definitions_ architecture_name_ tier_names_)
    sz_runtime_detectable_()
    set(compile_verdicts_known_ ${STRINGZILLA_HAS_RUNTIME_DETECTION_})
    set(run_verdicts_known_ 0)
    set(machine_tiers_ "")
    if (STRINGZILLA_HAS_RUNTIME_DETECTION_)
        sz_machine_capabilities_()
    endif ()
    if (NOT "${STRINGZILLA_MACHINE_CAPABILITIES}" STREQUAL "")
        set(run_verdicts_known_ 1)
        string(REPLACE "," ";" machine_tiers_ "${STRINGZILLA_MACHINE_CAPABILITIES}")
    endif ()
    set(compile_definitions_ "")
    set(run_definitions_ "")
    foreach (tier_ IN LISTS tier_names_)
        string(TOLOWER "${tier_}" tier_lowercase_)
        set(toolchain_compiles_ ${sz_target_${tier_lowercase_}_compiles})
        set(machine_runs_ ${toolchain_compiles_})
        if (machine_runs_
            AND run_verdicts_known_
            AND NOT tier_lowercase_ IN_LIST machine_tiers_
        )
            set(machine_runs_ 0)
        endif ()
        if (DEFINED STRINGZILLA_TARGET_${tier_})
            message(STATUS "STRINGZILLA_TARGET_${tier_} override in effect: ${STRINGZILLA_TARGET_${tier_}}")
            set(toolchain_compiles_ 0)
            set(machine_runs_ 0)
        endif ()
        if (DEFINED STRINGZILLA_TARGET_${tier_} AND STRINGZILLA_TARGET_${tier_})
            set(toolchain_compiles_ 1)
            set(machine_runs_ 1)
        endif ()
        if (toolchain_compiles_ AND NOT sz_target_${tier_lowercase_}_compiles)
            message(WARNING "STRINGZILLA_TARGET_${tier_}=1 requested, but its probe does not compile here; ignoring")
            set(toolchain_compiles_ 0)
            set(machine_runs_ 0)
        endif ()
        if (compile_verdicts_known_ OR DEFINED STRINGZILLA_TARGET_${tier_})
            list(APPEND compile_definitions_ "STRINGZILLA_TARGET_${tier_}=${toolchain_compiles_}")
        endif ()
        if (run_verdicts_known_ OR DEFINED STRINGZILLA_TARGET_${tier_})
            list(APPEND run_definitions_ "STRINGZILLA_TARGET_${tier_}=${machine_runs_}")
        endif ()
    endforeach ()
    if (NOT "${compile_definitions_}" STREQUAL "")
        list(JOIN compile_definitions_ " " compile_summary_)
        message(STATUS "${architecture_name_} compile verdicts: ${compile_summary_}")
    endif ()
    if (NOT "${run_definitions_}" STREQUAL "")
        list(JOIN run_definitions_ " " run_summary_)
        message(STATUS "${architecture_name_} run verdicts: ${run_summary_}")
    endif ()
    set(sz_compile_definitions_
        "${compile_definitions_}"
        CACHE INTERNAL "STRINGZILLA_TARGET_<TIER>=0/1 verdicts for runtime-dispatched units"
    )
    set(sz_run_definitions_
        "${run_definitions_}"
        CACHE INTERNAL "STRINGZILLA_TARGET_<TIER>=0/1 verdicts for comptime-dispatched executables"
    )
endfunction ()
