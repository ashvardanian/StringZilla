# cmake/stringzilla_probes.cmake — which CPU capabilities of the target architecture the toolchain compiles, and which
# GPU capabilities the listed codes run
#
# One `stringzilla_add_cpu_capability` row per capability, and two verdicts each, both defined on every unit linking
# the verdict target:
#
#   compiles  -> STRINGZILLA_ARCH_<ARCH>_<CAPABILITY>_  the toolchain builds its code at the unit's floor
#   compiled  -> STRINGZILLA_TARGET_<CAPABILITY>        it compiles, its target is not 0, and the build kind keeps it
#
# Each row probes `probes/<capability>.c`, which raises only its own ARCH flag, into the cached
# `stringzilla_target_<capability>_compiles`, and caches the flags enabling the capability across a unit as
# `stringzilla_target_<capability>_flags`. `-D STRINGZILLA_TARGET_<CAPABILITY>=0` leaves its kernels out, and
# `stringzilla_target_<capability>_compiled` caches the verdict after it. A monoversioned build also compiles
# `probes/implied.c` at the floor into the cached `stringzilla_target_<capability>_implied`, and keeps only the kernels
# the floor's predefined macros imply. MSVC has no `-march=native`, so its native build runs each enabled probe, caching
# whether this host runs the capability as `stringzilla_target_<capability>_runs`, and a monoversioned one keeps those.
#
# A compiled capability is not one the unit's own flags enable, nor one the CPU runs. `stringzilla_header` carries no
# macro, so its consumers enable what their flags name, and the runtime mask picks among the compiled ones.

include_guard(GLOBAL)

# Linked by the units that call CPU capability kernels and pick them at runtime: each macro is 1 where the toolchain
# compiles the capability, else 0. The twin of `sz_capabilities_compiled_cpu()`, published as
# `stringzilla::cpu_capabilities_compiled` for projects building StringZilla in their own tree, like USearch, and
# never installed, as an installed header meets other toolchains.
add_library(stringzilla_cpu_capabilities_compiled INTERFACE)
add_library(stringzilla::cpu_capabilities_compiled ALIAS stringzilla_cpu_capabilities_compiled)
# Linked by the GPU units: every CPU capability's ARCH flag is 0, so no CPU intrinsics header reaches a device compiler,
# and the targets follow by default.
add_library(stringzilla_cpu_capabilities_disabled_ INTERFACE)
# Serial, then each capability row's name as its verdict enables it, for the configure summary.
set_property(GLOBAL PROPERTY stringzilla_capabilities_compiled serial)

# Probes one CPU capability, named `capability_name_` in its probe and cache variables, and defines
# `architecture_macro_` and `capability_macro_` from the two verdicts. `GCC_FLAGS`, or `MSVC_FLAGS` under MSVC, enable
# the capability across a whole unit. The probe compiles without them, as the library does, the capability's target
# pragmas scoping it per function; a capability this toolchain cannot scope that way stays off, and the serial kernels
# serve its calls.
function (stringzilla_add_cpu_capability capability_name_ capability_macro_ architecture_macro_)
    cmake_parse_arguments(PARSE_ARGV 3 arg "" "" "GCC_FLAGS;MSVC_FLAGS")
    if (MSVC)
        set(unit_flags_ ${arg_MSVC_FLAGS})
    else ()
        set(unit_flags_ ${arg_GCC_FLAGS})
    endif ()
    set(stringzilla_target_${capability_name_}_flags
        "${unit_flags_}"
        CACHE INTERNAL "Flags enabling ${capability_macro_} across a unit"
    )
    if (NOT DEFINED stringzilla_target_${capability_name_}_compiles)
        set(CMAKE_TRY_COMPILE_CONFIGURATION "Release")
        try_compile(
            stringzilla_target_${capability_name_}_compiles ${CMAKE_BINARY_DIR}/stringzilla_probes
            ${PROJECT_SOURCE_DIR}/probes/${capability_name_}.c
            CMAKE_FLAGS "-DINCLUDE_DIRECTORIES=${PROJECT_SOURCE_DIR}/c;${PROJECT_SOURCE_DIR}/include" C_STANDARD 99
            COMPILE_DEFINITIONS ${stringzilla_unit_arch_flags_}
        )
    endif ()
    message(
        STATUS
            "Performing ISA probe ${capability_macro_} - compiles: ${stringzilla_target_${capability_name_}_compiles}"
    )
    set(capability_compiled_ ${stringzilla_target_${capability_name_}_compiles})
    if (DEFINED ${capability_macro_} AND NOT ${capability_macro_})
        set(capability_compiled_ FALSE)
    endif ()
    # The host's exit code is cached per build folder, and a probe killed by a missing instruction never exits with 0.
    set(capability_runs_ FALSE)
    if (capability_compiled_
        AND MSVC
        AND STRINGZILLA_MARCH STREQUAL "native"
    )
        if (NOT DEFINED stringzilla_target_${capability_name_}_exit_code)
            try_run(
                stringzilla_target_${capability_name_}_exit_code stringzilla_target_${capability_name_}_built
                ${CMAKE_BINARY_DIR}/stringzilla_probes ${PROJECT_SOURCE_DIR}/probes/${capability_name_}.c
                CMAKE_FLAGS "-DINCLUDE_DIRECTORIES=${PROJECT_SOURCE_DIR}/c;${PROJECT_SOURCE_DIR}/include" C_STANDARD 99
            )
        endif ()
        if (stringzilla_target_${capability_name_}_exit_code STREQUAL "0")
            set(capability_runs_ TRUE)
        endif ()
        set(capability_implied_ ${capability_runs_})
        message(STATUS "Performing ISA probe ${capability_macro_} - runs: ${capability_runs_}")
    elseif (capability_compiled_ AND NOT STRINGZILLA_MULTIVERSIONED)
        # A verdict holds for the floor it was probed at, so a build tree whose floor changed probes again.
        if (NOT "${stringzilla_unit_arch_flags_}" STREQUAL "${stringzilla_target_${capability_name_}_implied_floor}")
            unset(stringzilla_target_${capability_name_}_implied CACHE)
        endif ()
        if (NOT DEFINED stringzilla_target_${capability_name_}_implied)
            # The probe holds no `main`, as it only asks the preprocessor.
            set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)
            try_compile(
                stringzilla_target_${capability_name_}_implied ${CMAKE_BINARY_DIR}/stringzilla_probes
                ${PROJECT_SOURCE_DIR}/probes/implied.c
                CMAKE_FLAGS "-DINCLUDE_DIRECTORIES=${PROJECT_SOURCE_DIR}/c;${PROJECT_SOURCE_DIR}/include" C_STANDARD 99
                COMPILE_DEFINITIONS ${stringzilla_unit_arch_flags_} -DSTRINGZILLA_PROBE_MACRO=${capability_macro_}
            )
        endif ()
        set(stringzilla_target_${capability_name_}_implied_floor
            "${stringzilla_unit_arch_flags_}"
            CACHE INTERNAL "The floor ${capability_macro_} was last probed at"
        )
        set(capability_implied_ ${stringzilla_target_${capability_name_}_implied})
        message(STATUS "Performing ISA probe ${capability_macro_} - implied: ${capability_implied_}")
    endif ()
    # `-D STRINGZILLA_TARGET_<CAP>=1` vouches for what no compiler macro reports, like WFET or MSVC's AVX-512 subsets.
    if (NOT STRINGZILLA_MULTIVERSIONED
        AND NOT capability_implied_
        AND NOT ${capability_macro_}
    )
        set(capability_compiled_ FALSE)
    endif ()
    set(stringzilla_target_${capability_name_}_compiled
        ${capability_compiled_}
        CACHE INTERNAL "Whether ${capability_macro_} compiles, is not switched off, and the build kind keeps it"
    )
    if (capability_compiled_)
        set_property(GLOBAL APPEND PROPERTY stringzilla_capabilities_compiled ${capability_name_})
    endif ()
    set(stringzilla_target_${capability_name_}_runs
        ${capability_runs_}
        CACHE INTERNAL "Whether ${capability_macro_} is enabled and this host runs it, for MSVC's native build"
    )
    target_compile_definitions(
        stringzilla_cpu_capabilities_compiled
        INTERFACE "${architecture_macro_}=$<BOOL:${stringzilla_target_${capability_name_}_compiles}>"
                  "${capability_macro_}=$<BOOL:${capability_compiled_}>"
    )
    target_compile_definitions(stringzilla_cpu_capabilities_disabled_ INTERFACE ${architecture_macro_}=0)
endfunction ()

# Flags are the union of each capability's target pragmas. LASX and POWER9 scope per function from GCC 15 on, which
# opens `lasxintrin.h` and `altivec.h` inside their target regions; Clang never does, so its builds leave both off.
# WebAssembly carries the one capability `STRINGZILLA_MARCH` names, so it probes none.
if (stringzilla_arch_x8664_)
    stringzilla_add_cpu_capability(
        westmere STRINGZILLA_TARGET_WESTMERE
        STRINGZILLA_ARCH_X8664_WESTMERE_
        GCC_FLAGS -msse4.2 -maes -mpclmul
    )
    stringzilla_add_cpu_capability(
        goldmont STRINGZILLA_TARGET_GOLDMONT
        STRINGZILLA_ARCH_X8664_GOLDMONT_
        GCC_FLAGS -msse3 -mssse3 -msse4.1 -msha
    )
    stringzilla_add_cpu_capability(
        haswell STRINGZILLA_TARGET_HASWELL
        STRINGZILLA_ARCH_X8664_HASWELL_
        GCC_FLAGS -mavx2
                  -mfma
                  -mbmi
                  -mbmi2
                  -mpopcnt
                  -mlzcnt
                  -mpclmul
        MSVC_FLAGS /arch:AVX2
    )
    stringzilla_add_cpu_capability(
        skylake STRINGZILLA_TARGET_SKYLAKE
        STRINGZILLA_ARCH_X8664_SKYLAKE_
        GCC_FLAGS -mavx2
                  -mavx512f
                  -mavx512vl
                  -mavx512bw
                  -mavx512dq
                  -mbmi
                  -mbmi2
                  -mpopcnt
                  -mlzcnt
                  -maes
        MSVC_FLAGS /arch:AVX512
    )
    stringzilla_add_cpu_capability(
        icelake STRINGZILLA_TARGET_ICELAKE
        STRINGZILLA_ARCH_X8664_ICELAKE_
        GCC_FLAGS -mavx2
                  -mavx512f
                  -mavx512vl
                  -mavx512bw
                  -mavx512dq
                  -mavx512vbmi
                  -mavx512vbmi2
                  -mavx512vnni
                  -mbmi
                  -mbmi2
                  -mpopcnt
                  -mlzcnt
                  -maes
                  -mvaes
                  -mpclmul
                  -mvpclmulqdq
                  -msha
        MSVC_FLAGS /arch:AVX512
    )
elseif (stringzilla_arch_arm64_)
    stringzilla_add_cpu_capability(
        neon STRINGZILLA_TARGET_NEON
        STRINGZILLA_ARCH_ARM64_NEON_
        GCC_FLAGS -march=armv8-a+simd
    )
    stringzilla_add_cpu_capability(
        neonaes STRINGZILLA_TARGET_NEONAES
        STRINGZILLA_ARCH_ARM64_NEONAES_
        GCC_FLAGS -march=armv8-a+simd+crypto+aes
    )
    stringzilla_add_cpu_capability(
        neonsha STRINGZILLA_TARGET_NEONSHA
        STRINGZILLA_ARCH_ARM64_NEONSHA_
        GCC_FLAGS -march=armv8-a+simd+crypto+sha2
    )
    stringzilla_add_cpu_capability(
        sve STRINGZILLA_TARGET_SVE
        STRINGZILLA_ARCH_ARM64_SVE_
        GCC_FLAGS -march=armv8.2-a+sve
    )
    stringzilla_add_cpu_capability(
        sve2 STRINGZILLA_TARGET_SVE2
        STRINGZILLA_ARCH_ARM64_SVE2_
        GCC_FLAGS -march=armv8.2-a+sve+sve2
    )
    stringzilla_add_cpu_capability(
        sve2aes STRINGZILLA_TARGET_SVE2AES
        STRINGZILLA_ARCH_ARM64_SVE2AES_
        GCC_FLAGS -march=armv8.2-a+sve+sve2+sve2-aes+crypto
    )
elseif (stringzilla_arch_riscv64_)
    stringzilla_add_cpu_capability(
        rvv STRINGZILLA_TARGET_RVV
        STRINGZILLA_ARCH_RISCV64_RVV_
        GCC_FLAGS -march=rv64gcv
    )
    stringzilla_add_cpu_capability(
        rvvcrypto STRINGZILLA_TARGET_RVVCRYPTO
        STRINGZILLA_ARCH_RISCV64_RVVCRYPTO_
        GCC_FLAGS -march=rv64gcv_zvkned_zvknhb_zvkg
    )
elseif (stringzilla_arch_loongarch64_)
    stringzilla_add_cpu_capability(
        loongsonasx STRINGZILLA_TARGET_LOONGSONASX
        STRINGZILLA_ARCH_LOONGARCH64_LOONGSONASX_
        GCC_FLAGS -mlasx
    )
elseif (stringzilla_arch_ppc64_)
    stringzilla_add_cpu_capability(
        powervsx STRINGZILLA_TARGET_POWERVSX
        STRINGZILLA_ARCH_PPC64_POWERVSX_
        GCC_FLAGS -mcpu=power9
    )
endif ()

# Linked by the GPU units, the libraries and the GPU suites: each CUDA or ROCm capability's macro is 1 where some code
# the build compiles, without CMake's "-real" or "-virtual" or ROCm's ":xnack-" features, runs the capability, else 0.
add_library(stringzilla_cuda_capabilities_compiled_ INTERFACE)
add_library(stringzilla_rocm_capabilities_compiled_ INTERFACE)

# One GPU capability, named `capability_name_` in its cache variable: caches the `CUDA_ARCHITECTURES` or
# `ROCM_ARCHITECTURES` running it as `stringzilla_target_<capability>_architectures`, the twin of a CPU capability's
# `stringzilla_target_<capability>_flags`, and defines `capability_macro_` from the codes in that vendor's
# `stringzilla_<vendor>_resolved_architectures_`. `-D STRINGZILLA_TARGET_<CAPABILITY>=0` leaves it out, as on the CPU,
# and `stringzilla_target_<capability>_compiled` caches the verdict.
function (stringzilla_add_gpu_capability capability_name_ capability_macro_)
    cmake_parse_arguments(PARSE_ARGV 2 arg "" "" "CUDA_ARCHITECTURES;ROCM_ARCHITECTURES")
    if (arg_CUDA_ARCHITECTURES)
        set(capability_architectures_ ${arg_CUDA_ARCHITECTURES})
        set(compiled_architectures_ ${stringzilla_cuda_resolved_architectures_})
        set(capabilities_target_ stringzilla_cuda_capabilities_compiled_)
    else ()
        set(capability_architectures_ ${arg_ROCM_ARCHITECTURES})
        set(compiled_architectures_ ${stringzilla_rocm_resolved_architectures_})
        set(capabilities_target_ stringzilla_rocm_capabilities_compiled_)
    endif ()
    set(stringzilla_target_${capability_name_}_architectures
        "${capability_architectures_}"
        CACHE INTERNAL "Architectures running ${capability_macro_}"
    )
    set(capability_compiled_ FALSE)
    foreach (code_ IN LISTS compiled_architectures_)
        string(REGEX REPLACE "(-(real|virtual)|:.+)$" "" architecture_ "${code_}")
        if (architecture_ IN_LIST capability_architectures_)
            set(capability_compiled_ TRUE)
        endif ()
    endforeach ()
    if (DEFINED ${capability_macro_} AND NOT ${capability_macro_})
        set(capability_compiled_ FALSE)
    endif ()
    set(stringzilla_target_${capability_name_}_compiled
        ${capability_compiled_}
        CACHE INTERNAL "Whether some compiled architecture runs ${capability_macro_} and it is not switched off"
    )
    if (capability_compiled_)
        set_property(GLOBAL APPEND PROPERTY stringzilla_capabilities_compiled ${capability_name_})
    endif ()
    target_compile_definitions(${capabilities_target_} INTERFACE "${capability_macro_}=$<BOOL:${capability_compiled_}>")
endfunction ()

# Sets `output_` to the entries of `codes_` that `capability_codes_` names, keeping CMake's "-real" and "-virtual"
# suffixes and a ROCm target ID's ":xnack-" features. NVIDIA's release set, whose codes alone carry the suffixes, keeps
# only the first, the lowest code the capability needs; every ROCm target ID is kept.
function (stringzilla_select_gpu_codes output_ codes_ capability_codes_)
    set(kept_)
    foreach (code_ IN LISTS codes_)
        string(REGEX REPLACE "(-(real|virtual)|:.+)$" "" architecture_ "${code_}")
        if (architecture_ IN_LIST capability_codes_)
            list(APPEND kept_ "${code_}")
            if (STRINGZILLA_CUDA_ARCH STREQUAL "all" AND code_ MATCHES "-(real|virtual)$")
                break()
            endif ()
        endif ()
    endforeach ()
    set(${output_}
        "${kept_}"
        PARENT_SCOPE
    )
endfunction ()

if (MSVC AND STRINGZILLA_MARCH STREQUAL "native")
    # MSVC has no `/arch:native`, so it takes the `/arch:` of the widest capability this host runs.
    if (stringzilla_target_icelake_runs)
        set(stringzilla_host_arch_flags_ ${stringzilla_target_icelake_flags})
    elseif (stringzilla_target_skylake_runs)
        set(stringzilla_host_arch_flags_ ${stringzilla_target_skylake_flags})
    elseif (stringzilla_target_haswell_runs)
        set(stringzilla_host_arch_flags_ ${stringzilla_target_haswell_flags})
    endif ()
    add_compile_options("$<$<COMPILE_LANGUAGE:C,CXX>:${stringzilla_host_arch_flags_}>")
endif ()
