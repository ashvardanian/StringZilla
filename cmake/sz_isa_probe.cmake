# cmake/sz_isa_probe.cmake — shared ISA probe infrastructure
#
# Two independently-probed facts decide every SIMD tier, answered with the checked-in `probes/` sources
# that the Cargo build (`build.rs`) reuses verbatim (the layout mirrors the sibling NumKong project):
#   - the COMPILE set — can this toolchain emit the tier? `sz_isa_probe_()` try-compiles
#     `probes/<arch>_<tier>.c`, which reuses the real kernels' per-function `target` pragmas and
#     intrinsics, so ICE-prone toolchains and missing intrinsics surface at configure time.
#   - the RUN set — can this machine execute the tier? `sz_machine_capabilities_()` compiles and RUNS
#     `probes/run_capabilities.c` (under `CMAKE_CROSSCOMPILING_EMULATOR` when cross-compiling with one,
#     which pins the same CPU model CTest will use).
# Runtime-dispatched targets enable the COMPILE set — the load-time dispatch table masks whatever the
# CPU lacks. Comptime-dispatched targets (tests, benchmarks) enable the intersection of the two sets,
# because the picked tier is baked into every call with no runtime guard.
#
# Each per-architecture file (sz_arm_isa_probes.cmake, …) calls `sz_isa_probe_()` once per tier between
# `sz_isa_probes_begin_()`/`sz_isa_probes_end_()` and sets `SZ_ISA_TIERS` — newest tier first, each name
# doubling as the `SZ_USE_<TIER>` macro suffix and (lowercased) as the run probe's capability token.

include(CheckCSourceCompiles)

# Save and restore `CMAKE_REQUIRED_FLAGS` around probes, and force a Release configuration for
# try_compile so Debug-only sanitizer runtimes never fail a probe for unrelated reasons.
macro (sz_isa_probes_begin_)
    set(sz_saved_required_flags_ "${CMAKE_REQUIRED_FLAGS}")
    set(sz_saved_try_compile_config_ "${CMAKE_TRY_COMPILE_CONFIGURATION}")
    set(CMAKE_TRY_COMPILE_CONFIGURATION "Release")
endmacro ()

macro (sz_isa_probes_end_)
    set(CMAKE_REQUIRED_FLAGS "${sz_saved_required_flags_}")
    set(CMAKE_TRY_COMPILE_CONFIGURATION "${sz_saved_try_compile_config_}")
endmacro ()

# Try-compile one tier's probe program; the (cached) result lands in `${var_}` (`SZ_CAN_COMPILE_<TIER>`
# by convention). Native tiers pass empty flag columns — their ISA rides in the per-function `target`
# pragmas over baseline flags, exactly like the real kernels — while the wasm tiers need
# `-msimd128`/`-mrelaxed-simd`.
macro (sz_isa_probe_ var_ msvc_flags_ gcc_flags_ probe_file_)
    file(READ "${CMAKE_CURRENT_SOURCE_DIR}/${probe_file_}" sz_probe_source_)
    if (MSVC)
        set(CMAKE_REQUIRED_FLAGS "${msvc_flags_}")
    else ()
        set(CMAKE_REQUIRED_FLAGS "${gcc_flags_}")
    endif ()
    check_c_source_compiles("${sz_probe_source_}" ${var_})
endmacro ()

# Compile-probe `probes/runtime_detection.c`, caching the result in `SZ_RUNTIME_DETECTABLE`: whether the
# library built for this target performs real runtime capability detection (the header-owned
# `SZ_CAPABILITIES_RUNTIME_DETECTABLE_`, defined next to the detectors). Where it does, load-time
# dispatch masks whatever the CPU lacks, so every compilable tier is safe to enable; where it does not
# (WebAssembly by nature, OS-less exotic targets), dispatch tables mirror the compile-time mask and tier
# selection must stay with the toolchain flags + `types.h` auto-detection. Probed with `SZ_AVOID_LIBC=0`,
# matching every regular target; `stringzilla_bare` differs only on niche OS-less combinations.
function (sz_runtime_detectable_)
    if (DEFINED SZ_RUNTIME_DETECTABLE)
        return ()
    endif ()
    file(READ "${CMAKE_CURRENT_SOURCE_DIR}/probes/runtime_detection.c" sz_probe_source_)
    set(sz_saved_required_includes_ "${CMAKE_REQUIRED_INCLUDES}")
    set(sz_saved_required_definitions_ "${CMAKE_REQUIRED_DEFINITIONS}")
    set(CMAKE_REQUIRED_INCLUDES "${CMAKE_CURRENT_SOURCE_DIR}/include")
    set(CMAKE_REQUIRED_DEFINITIONS "-DSZ_AVOID_LIBC=0")
    check_c_source_compiles("${sz_probe_source_}" sz_runtime_detectable_probe_)
    set(CMAKE_REQUIRED_INCLUDES "${sz_saved_required_includes_}")
    set(CMAKE_REQUIRED_DEFINITIONS "${sz_saved_required_definitions_}")
    if (sz_runtime_detectable_probe_)
        set(SZ_RUNTIME_DETECTABLE 1 CACHE INTERNAL "Runtime capability detection exists for this target")
    else ()
        set(SZ_RUNTIME_DETECTABLE 0 CACHE INTERNAL "Runtime capability detection exists for this target")
        message(STATUS "No runtime capability detection for this target; SIMD tiers follow the toolchain flags")
    endif ()
endfunction ()

# Compile & RUN `probes/run_capabilities.c`, caching the machine's comma-separated tier tokens in
# `SZ_MACHINE_CAPABILITIES` (e.g. "serial,neon,neonaes"). Left EMPTY when the answer is unknowable —
# cross-compiling without an emulator, any probe failure, or the probe itself reporting that this
# platform has no hardware introspection — and callers then fall back to `types.h` auto-detection under
# the target's own `-march` flags.
function (sz_machine_capabilities_)
    if (DEFINED SZ_MACHINE_CAPABILITIES)
        return ()
    endif ()
    if (CMAKE_CROSSCOMPILING AND NOT CMAKE_CROSSCOMPILING_EMULATOR)
        set(SZ_MACHINE_CAPABILITIES "" CACHE INTERNAL "Tier tokens the build machine can run")
        message(STATUS "Machine capabilities: unknown (cross-compiling without an emulator)")
        return ()
    endif ()
    try_run(
        sz_run_probe_exit_ sz_run_probe_compiled_ ${CMAKE_BINARY_DIR}/sz_probes
        ${CMAKE_CURRENT_SOURCE_DIR}/probes/run_capabilities.c
        CMAKE_FLAGS "-DINCLUDE_DIRECTORIES=${CMAKE_CURRENT_SOURCE_DIR}/include"
        RUN_OUTPUT_VARIABLE sz_run_probe_output_
    )
    if (sz_run_probe_compiled_ AND sz_run_probe_exit_ EQUAL 0)
        string(STRIP "${sz_run_probe_output_}" sz_run_probe_output_)
        set(SZ_MACHINE_CAPABILITIES "${sz_run_probe_output_}" CACHE INTERNAL "Tier tokens the build machine can run")
        message(STATUS "Machine capabilities: ${SZ_MACHINE_CAPABILITIES}")
    else ()
        set(SZ_MACHINE_CAPABILITIES "" CACHE INTERNAL "Tier tokens the build machine can run")
        message(STATUS "Machine capabilities: unknown (run probe unavailable)")
    endif ()
endfunction ()
