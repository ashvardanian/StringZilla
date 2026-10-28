# cmake/stringzilla_probes.cmake — which CPU capabilities of the target architecture the toolchain compiles, and which
# GPU capabilities the listed codes run
#
# One `stringzilla_add_cpu_capability` row per capability. Each probes `probes/<capability>.c` into the cached
# `stringzilla_target_<capability>_compiles`, caches the flags enabling the capability across a unit as
# `stringzilla_target_<capability>_flags`, and defines its `STRINGZILLA_TARGET_<CAPABILITY>` macro on the targets below.
# `-D STRINGZILLA_TARGET_<CAPABILITY>=0` leaves a capability out, and `stringzilla_target_<capability>_enabled` caches
# the verdict after it. A native build that is not cross-compiling also runs each enabled probe, caching whether this
# host runs the capability as `stringzilla_target_<capability>_runs`.
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
# Linked by the units held at the platform baseline: every CPU capability's macro is 0.
add_library(stringzilla_cpu_capabilities_disabled_ INTERFACE)
# Serial, then each capability row's `BIT` from `capabilities.h` as its verdict enables it, for a machine profile's
# `STRINGZILLA_PROFILE_CAPABILITIES`.
set(stringzilla_capabilities_enabled_ 1)

# Probes one CPU capability, named `capability_name_` in its probe and cache variables, and defines
# `capability_macro_` from the verdict. `BIT` is its `sz_cap_<capability>_k` bit. `GCC_FLAGS`, or `MSVC_FLAGS` under
# MSVC, enable the capability across a whole unit. The probe compiles without them, as the library does, the
# capability's target pragmas scoping it per function; a capability this toolchain cannot scope that way stays off, and
# the serial kernels serve its calls.
function (stringzilla_add_cpu_capability capability_name_ capability_macro_)
    cmake_parse_arguments(PARSE_ARGV 2 arg "" "BIT" "GCC_FLAGS;MSVC_FLAGS")
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
            CMAKE_FLAGS "-DINCLUDE_DIRECTORIES=${PROJECT_SOURCE_DIR}/include" C_STANDARD 99
        )
    endif ()
    message(
        STATUS
            "Performing ISA probe ${capability_macro_} - compiles: ${stringzilla_target_${capability_name_}_compiles}"
    )
    set(capability_enabled_ ${stringzilla_target_${capability_name_}_compiles})
    if (DEFINED ${capability_macro_} AND NOT ${capability_macro_})
        set(capability_enabled_ FALSE)
    endif ()
    set(stringzilla_target_${capability_name_}_enabled
        ${capability_enabled_}
        CACHE INTERNAL "Whether ${capability_macro_} compiles and is not switched off"
    )
    if (capability_enabled_)
        math(EXPR capabilities_ "${stringzilla_capabilities_enabled_} | (1 << ${arg_BIT})" OUTPUT_FORMAT HEXADECIMAL)
        set(stringzilla_capabilities_enabled_
            ${capabilities_}
            PARENT_SCOPE
        )
    endif ()
    # The host's exit code is cached per build folder, and a probe killed by a missing instruction never exits with 0.
    set(capability_runs_ FALSE)
    if (capability_enabled_
        AND STRINGZILLA_TARGET_ARCH STREQUAL "native"
        AND NOT CMAKE_CROSSCOMPILING
    )
        if (NOT DEFINED stringzilla_target_${capability_name_}_exit_code)
            try_run(stringzilla_target_${capability_name_}_exit_code stringzilla_target_${capability_name_}_built
                    ${CMAKE_BINARY_DIR}/stringzilla_probes ${PROJECT_SOURCE_DIR}/probes/${capability_name_}.c
                    CMAKE_FLAGS "-DINCLUDE_DIRECTORIES=${PROJECT_SOURCE_DIR}/include" C_STANDARD 99
            )
        endif ()
        if (stringzilla_target_${capability_name_}_exit_code STREQUAL "0")
            set(capability_runs_ TRUE)
        endif ()
        message(STATUS "Performing ISA probe ${capability_macro_} - runs: ${capability_runs_}")
    endif ()
    set(stringzilla_target_${capability_name_}_runs
        ${capability_runs_}
        CACHE INTERNAL "Whether ${capability_macro_} is enabled and this host runs it"
    )
    target_compile_definitions(
        stringzilla_cpu_capabilities_compiled INTERFACE "${capability_macro_}=$<BOOL:${capability_enabled_}>"
    )
    target_compile_definitions(stringzilla_cpu_capabilities_disabled_ INTERFACE ${capability_macro_}=0)
endfunction ()

# Flags are the union of each capability's target pragmas. LASX and POWER9 scope per function from GCC 15 on, which
# opens `lasxintrin.h` and `altivec.h` inside their target regions; Clang never does, so its builds leave both off.
# WebAssembly carries the one capability `STRINGZILLA_TARGET_ARCH` names, so it probes none.
if (stringzilla_arch_x8664_)
    stringzilla_add_cpu_capability(
        westmere STRINGZILLA_TARGET_WESTMERE
        BIT 1
        GCC_FLAGS -msse4.2 -maes -mpclmul -mbmi -mlzcnt
    )
    stringzilla_add_cpu_capability(
        goldmont STRINGZILLA_TARGET_GOLDMONT
        BIT 2
        GCC_FLAGS -msse3 -mssse3 -msse4.1 -msha
    )
    stringzilla_add_cpu_capability(
        haswell STRINGZILLA_TARGET_HASWELL
        BIT 3
        GCC_FLAGS -mavx2 -mfma -mbmi -mbmi2 -mpopcnt -mlzcnt
        MSVC_FLAGS /arch:AVX2
    )
    stringzilla_add_cpu_capability(
        skylake STRINGZILLA_TARGET_SKYLAKE
        BIT 4
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
        BIT 5
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
        BIT 6
        GCC_FLAGS -march=armv8-a+simd
    )
    stringzilla_add_cpu_capability(
        neonaes STRINGZILLA_TARGET_NEONAES
        BIT 7
        GCC_FLAGS -march=armv8-a+simd+crypto+aes
    )
    stringzilla_add_cpu_capability(
        neonsha STRINGZILLA_TARGET_NEONSHA
        BIT 8
        GCC_FLAGS -march=armv8-a+simd+crypto+sha2
    )
    stringzilla_add_cpu_capability(
        sve STRINGZILLA_TARGET_SVE
        BIT 9
        GCC_FLAGS -march=armv8.2-a+sve
    )
    stringzilla_add_cpu_capability(
        sve2 STRINGZILLA_TARGET_SVE2
        BIT 10
        GCC_FLAGS -march=armv8.2-a+sve+sve2
    )
    stringzilla_add_cpu_capability(
        sve2aes STRINGZILLA_TARGET_SVE2AES
        BIT 11
        GCC_FLAGS -march=armv8.2-a+sve+sve2+sve2-aes+crypto
    )
elseif (stringzilla_arch_riscv64_)
    stringzilla_add_cpu_capability(
        rvv STRINGZILLA_TARGET_RVV
        BIT 12
        GCC_FLAGS -march=rv64gcv
    )
    stringzilla_add_cpu_capability(
        rvvcrypto STRINGZILLA_TARGET_RVVCRYPTO
        BIT 13
        GCC_FLAGS -march=rv64gcv_zvkned_zvknhb_zvkg
    )
elseif (stringzilla_arch_loongarch64_)
    stringzilla_add_cpu_capability(
        loongsonasx STRINGZILLA_TARGET_LOONGSONASX
        BIT 16
        GCC_FLAGS -mlasx
    )
elseif (stringzilla_arch_ppc64_)
    stringzilla_add_cpu_capability(
        powervsx STRINGZILLA_TARGET_POWERVSX
        BIT 17
        GCC_FLAGS -mcpu=power9
    )
endif ()

# Linked by the GPU units, the libraries and the GPU suites: each CUDA capability's macro is 1 where some code the
# build compiles, without CMake's "-real" or "-virtual", runs the capability, else 0.
add_library(stringzilla_cuda_capabilities_compiled_ INTERFACE)

# One CUDA capability, named `capability_name_` in its cache variable: caches the `CUDA_ARCHITECTURES` running it as
# `stringzilla_target_<capability>_architectures`, the twin of a CPU capability's `stringzilla_target_<capability>_flags`, and defines
# `capability_macro_` from the codes in `stringzilla_cuda_resolved_architectures_`. `-D STRINGZILLA_TARGET_<CAPABILITY>=0`
# leaves it out, as on the CPU, `stringzilla_target_<capability>_enabled` caches the verdict, and `BIT` is its
# `sz_cap_<capability>_k` bit.
function (stringzilla_add_gpu_capability capability_name_ capability_macro_)
    cmake_parse_arguments(PARSE_ARGV 2 arg "" "BIT" "CUDA_ARCHITECTURES")
    set(stringzilla_target_${capability_name_}_architectures
        "${arg_CUDA_ARCHITECTURES}"
        CACHE INTERNAL "Architectures running ${capability_macro_}"
    )
    set(capability_enabled_ FALSE)
    foreach (code_ IN LISTS stringzilla_cuda_resolved_architectures_)
        string(REGEX REPLACE "-(real|virtual)$" "" architecture_ "${code_}")
        if (architecture_ IN_LIST arg_CUDA_ARCHITECTURES)
            set(capability_enabled_ TRUE)
        endif ()
    endforeach ()
    if (DEFINED ${capability_macro_} AND NOT ${capability_macro_})
        set(capability_enabled_ FALSE)
    endif ()
    set(stringzilla_target_${capability_name_}_enabled
        ${capability_enabled_}
        CACHE INTERNAL "Whether some compiled architecture runs ${capability_macro_} and it is not switched off"
    )
    if (capability_enabled_)
        math(EXPR capabilities_ "${stringzilla_capabilities_enabled_} | (1 << ${arg_BIT})" OUTPUT_FORMAT HEXADECIMAL)
        set(stringzilla_capabilities_enabled_
            ${capabilities_}
            PARENT_SCOPE
        )
    endif ()
    target_compile_definitions(
        stringzilla_cuda_capabilities_compiled_ INTERFACE "${capability_macro_}=$<BOOL:${capability_enabled_}>"
    )
endfunction ()

# Sets `output_` to the entries of `codes_` that `capability_codes_` names, keeping CMake's "-real" and "-virtual"
# suffixes.
function (stringzilla_select_gpu_codes output_ codes_ capability_codes_)
    set(kept_)
    foreach (code_ IN LISTS codes_)
        string(REGEX REPLACE "-(real|virtual)$" "" architecture_ "${code_}")
        if (architecture_ IN_LIST capability_codes_)
            list(APPEND kept_ "${code_}")
        endif ()
    endforeach ()
    set(${output_}
        "${kept_}"
        PARENT_SCOPE
    )
endfunction ()

if (MSVC AND STRINGZILLA_TARGET_ARCH STREQUAL "native")
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
