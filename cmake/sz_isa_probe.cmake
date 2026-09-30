# cmake/sz_isa_probe.cmake — probes every kit of the target architecture over `probes/<kit>.c` and
# folds the verdicts into the cached `sz_compile_definitions_` list of
# `STRINGZILLA_TARGET_<KIT>=0/1`, reading the `STRINGZILLA_ARCH_*_` facts `CMakeLists.txt` computes.

include_guard(GLOBAL)

# Whether the toolchain builds one kit's kernels: the probe calls one of them, compiled header-only at the baseline
# flags, as the library compiles it. `try_compile` reruns even with its verdict cached, hence the guard.
function (sz_instruction_set_probe_ kit_)
    string(TOLOWER "${kit_}" kit_lowercase_)
    if (NOT DEFINED sz_target_${kit_lowercase_}_compiles)
        set(CMAKE_TRY_COMPILE_CONFIGURATION "Release")
        try_compile(
            sz_target_${kit_lowercase_}_compiles ${CMAKE_BINARY_DIR}/sz_probes
            ${PROJECT_SOURCE_DIR}/probes/${kit_lowercase_}.c
            CMAKE_FLAGS "-DINCLUDE_DIRECTORIES=${PROJECT_SOURCE_DIR}/include" C_STANDARD 99
            COMPILE_DEFINITIONS ${ARGN}
        )
    endif ()
    message(STATUS "Performing ISA probe ${kit_} - compiles: ${sz_target_${kit_lowercase_}_compiles}")
endfunction ()

# LASX and POWER9 compile file-wide, as `lasxintrin.h` and `altivec.h` hide their contents without
# the flag; `CMakeLists.txt` gives it to the kit's unit and header-only cross-checks alone.
set(sz_kit_flags_LOONGSONASX -mlasx)
set(sz_kit_flags_POWERVSX -mcpu=power9)

# WebAssembly carries the one kit `STRINGZILLA_TARGET_ARCH` names, so it probes none.
if (STRINGZILLA_ARCH_X8664_)
    set(sz_kits_ ICELAKE SKYLAKE HASWELL GOLDMONT WESTMERE)
elseif (STRINGZILLA_ARCH_ARM64_)
    set(sz_kits_ SVE2AES SVE2 SVE NEONSHA NEONAES NEON)
elseif (STRINGZILLA_ARCH_RISCV64_)
    set(sz_kits_ RVVCRYPTO RVV)
elseif (STRINGZILLA_ARCH_LOONGARCH64_)
    set(sz_kits_ LOONGSONASX)
elseif (STRINGZILLA_ARCH_PPC64_)
    set(sz_kits_ POWERVSX)
else ()
    set(sz_kits_ "")
endif ()

# `-D STRINGZILLA_TARGET_<KIT>=0/1` outranks a verdict, except past a failed probe. The header-only
# suites leave the file-wide kits to each unit's own flags.
set(sz_verdicts_ "")
set(sz_header_only_definitions_ "")
foreach (kit_ IN LISTS sz_kits_)
    sz_instruction_set_probe_(${kit_} ${sz_kit_flags_${kit_}})
    string(TOLOWER "${kit_}" kit_lowercase_)
    set(compiles_ 0)
    if (sz_target_${kit_lowercase_}_compiles)
        set(compiles_ 1)
    endif ()
    if (DEFINED STRINGZILLA_TARGET_${kit_})
        message(STATUS "STRINGZILLA_TARGET_${kit_} override in effect: ${STRINGZILLA_TARGET_${kit_}}")
        if (NOT STRINGZILLA_TARGET_${kit_})
            set(compiles_ 0)
        elseif (NOT compiles_)
            message(WARNING "STRINGZILLA_TARGET_${kit_}=1 requested, but its probe does not compile here; ignoring")
        endif ()
    endif ()
    list(APPEND sz_verdicts_ "STRINGZILLA_TARGET_${kit_}=${compiles_}")
    if (NOT DEFINED sz_kit_flags_${kit_})
        list(APPEND sz_header_only_definitions_ "STRINGZILLA_TARGET_${kit_}=${compiles_}")
    endif ()
endforeach ()
if (sz_verdicts_)
    list(JOIN sz_verdicts_ " " sz_verdicts_summary_)
    message(STATUS "Compile verdicts: ${sz_verdicts_summary_}")
endif ()
set(sz_compile_definitions_
    "${sz_verdicts_}"
    CACHE INTERNAL "STRINGZILLA_TARGET_<KIT>=0/1 verdicts"
)
