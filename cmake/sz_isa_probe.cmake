# cmake/sz_isa_probe.cmake — which CPU capabilities of the target architecture the toolchain compiles, and which GPU
# capabilities the listed codes run
#
# One `sz_cpu_capability_` row per capability. Each probes `probes/<capability>.c` into the cached
# `sz_target_<capability>_compiles`, caches the flags enabling the capability across a unit as
# `sz_target_<capability>_flags`, and defines its `STRINGZILLA_TARGET_<CAPABILITY>` macro on the targets below.
# `-D STRINGZILLA_TARGET_<CAPABILITY>=0` leaves a capability out.
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

# Probes one CPU capability, named `capability_name_` in its probe and cache variables, and defines
# `capability_macro_` from the verdict. `GCC_FLAGS`, or `MSVC_FLAGS` under MSVC, enable the capability across a whole
# unit. The probe compiles without them, as the library does, the capability's target pragmas scoping it per
# function; a capability this toolchain cannot scope that way stays off, and the serial kernels serve its calls.
function (sz_cpu_capability_ capability_name_ capability_macro_)
    cmake_parse_arguments(PARSE_ARGV 2 argument "" "" "GCC_FLAGS;MSVC_FLAGS")
    if (MSVC)
        set(unit_flags_ ${argument_MSVC_FLAGS})
    else ()
        set(unit_flags_ ${argument_GCC_FLAGS})
    endif ()
    set(sz_target_${capability_name_}_flags
        "${unit_flags_}"
        CACHE INTERNAL "Flags enabling ${capability_macro_} across a unit"
    )
    if (NOT DEFINED sz_target_${capability_name_}_compiles)
        set(CMAKE_TRY_COMPILE_CONFIGURATION "Release")
        try_compile(
            sz_target_${capability_name_}_compiles ${CMAKE_BINARY_DIR}/sz_probes
            ${PROJECT_SOURCE_DIR}/probes/${capability_name_}.c
            CMAKE_FLAGS "-DINCLUDE_DIRECTORIES=${PROJECT_SOURCE_DIR}/include" C_STANDARD 99
        )
    endif ()
    message(STATUS "Performing ISA probe ${capability_macro_} - compiles: ${sz_target_${capability_name_}_compiles}")
    set(capability_enabled_ ${sz_target_${capability_name_}_compiles})
    if (DEFINED ${capability_macro_} AND NOT ${capability_macro_})
        set(capability_enabled_ FALSE)
    endif ()
    target_compile_definitions(
        stringzilla_cpu_capabilities_compiled INTERFACE "${capability_macro_}=$<BOOL:${capability_enabled_}>"
    )
    target_compile_definitions(stringzilla_cpu_capabilities_disabled_ INTERFACE ${capability_macro_}=0)
endfunction ()

# Flags are the union of each capability's target pragmas. LASX and POWER9 scope per function from GCC 15 on, which
# opens `lasxintrin.h` and `altivec.h` inside their target regions; Clang never does, so its builds leave both off.
# WebAssembly carries the one capability `STRINGZILLA_TARGET_ARCH` names, so it probes none.
if (STRINGZILLA_ARCH_X8664_)
    sz_cpu_capability_(westmere STRINGZILLA_TARGET_WESTMERE GCC_FLAGS -msse4.2 -maes -mpclmul -mbmi -mlzcnt)
    sz_cpu_capability_(goldmont STRINGZILLA_TARGET_GOLDMONT GCC_FLAGS -msse3 -mssse3 -msse4.1 -msha)
    sz_cpu_capability_(
        haswell STRINGZILLA_TARGET_HASWELL
        GCC_FLAGS -mavx2 -mfma -mbmi -mbmi2 -mpopcnt -mlzcnt
        MSVC_FLAGS /arch:AVX2
    )
    sz_cpu_capability_(
        skylake STRINGZILLA_TARGET_SKYLAKE
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
    sz_cpu_capability_(
        icelake STRINGZILLA_TARGET_ICELAKE
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
elseif (STRINGZILLA_ARCH_ARM64_)
    sz_cpu_capability_(neon STRINGZILLA_TARGET_NEON GCC_FLAGS -march=armv8-a+simd)
    sz_cpu_capability_(neonaes STRINGZILLA_TARGET_NEONAES GCC_FLAGS -march=armv8-a+simd+crypto+aes)
    sz_cpu_capability_(neonsha STRINGZILLA_TARGET_NEONSHA GCC_FLAGS -march=armv8-a+simd+crypto+sha2)
    sz_cpu_capability_(sve STRINGZILLA_TARGET_SVE GCC_FLAGS -march=armv8.2-a+sve)
    sz_cpu_capability_(sve2 STRINGZILLA_TARGET_SVE2 GCC_FLAGS -march=armv8.2-a+sve+sve2)
    sz_cpu_capability_(sve2aes STRINGZILLA_TARGET_SVE2AES GCC_FLAGS -march=armv8.2-a+sve+sve2+sve2-aes+crypto)
elseif (STRINGZILLA_ARCH_RISCV64_)
    sz_cpu_capability_(rvv STRINGZILLA_TARGET_RVV GCC_FLAGS -march=rv64gcv)
    sz_cpu_capability_(rvvcrypto STRINGZILLA_TARGET_RVVCRYPTO GCC_FLAGS -march=rv64gcv_zvkned_zvknhb_zvkg)
elseif (STRINGZILLA_ARCH_LOONGARCH64_)
    sz_cpu_capability_(loongsonasx STRINGZILLA_TARGET_LOONGSONASX GCC_FLAGS -mlasx)
elseif (STRINGZILLA_ARCH_PPC64_)
    sz_cpu_capability_(powervsx STRINGZILLA_TARGET_POWERVSX GCC_FLAGS -mcpu=power9)
endif ()

# Linked by the GPU units, the libraries and the GPU suites: each CUDA capability's macro is 1 where some code the
# build compiles, without CMake's "-real" or "-virtual", runs the capability, else 0.
add_library(stringzilla_cuda_capabilities_compiled_ INTERFACE)

# One CUDA capability, named `capability_name_` in its cache variable: caches the `CUDA_ARCHITECTURES` running it as
# `sz_target_<capability>_architectures`, the twin of a CPU capability's `sz_target_<capability>_flags`, and defines
# `capability_macro_` from the codes in `STRINGZILLA_CUDA_RESOLVED_ARCHITECTURES_`.
function (sz_gpu_capability_ capability_name_ capability_macro_)
    cmake_parse_arguments(PARSE_ARGV 2 argument "" "" "CUDA_ARCHITECTURES")
    set(sz_target_${capability_name_}_architectures
        "${argument_CUDA_ARCHITECTURES}"
        CACHE INTERNAL "Architectures running ${capability_macro_}"
    )
    set(capability_enabled_ FALSE)
    foreach (code_ IN LISTS STRINGZILLA_CUDA_RESOLVED_ARCHITECTURES_)
        string(REGEX REPLACE "-(real|virtual)$" "" architecture_ "${code_}")
        if (architecture_ IN_LIST argument_CUDA_ARCHITECTURES)
            set(capability_enabled_ TRUE)
        endif ()
    endforeach ()
    target_compile_definitions(
        stringzilla_cuda_capabilities_compiled_ INTERFACE "${capability_macro_}=$<BOOL:${capability_enabled_}>"
    )
endfunction ()

# The entries of `codes` that `capability_codes` names, keeping CMake's "-real" and "-virtual" suffixes.
function (sz_gpu_codes_ codes capability_codes output)
    set(kept_)
    foreach (code_ IN LISTS codes)
        string(REGEX REPLACE "-(real|virtual)$" "" architecture_ "${code_}")
        if (architecture_ IN_LIST capability_codes)
            list(APPEND kept_ "${code_}")
        endif ()
    endforeach ()
    set(${output}
        "${kept_}"
        PARENT_SCOPE
    )
endfunction ()
