# Architecture and OS facts, and the ABI floor every C and C++ unit compiles at.

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
    # A module's one SIMD capability is `STRINGZILLA_MARCH`, which `types.h` reads back off the predefined
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

# Each unit gets the architecture's ABI floor, and SIMD kernels opt in via per-function target pragmas, so the library
# runs on every CPU of its architecture. STRINGZILLA_MARCH raises the floor: an ISA level like `x86-64-v3`, `native` for
# this host, or a CPU name like `graniterapids`, also when cross-compiling. Only C and C++ units take the flags, as nvcc
# would read them as its own; the CUDA block below hands them to its host compiler. The probes compile at
# `stringzilla_unit_arch_flags_`, the floor either way.
set(stringzilla_host_arch_flags_)
set(stringzilla_unit_arch_flags_)
if (STRINGZILLA_MARCH STREQUAL "native" AND CMAKE_CROSSCOMPILING)
    message(FATAL_ERROR "STRINGZILLA_MARCH=native needs the host as the target; name a CPU instead")
elseif (STRINGZILLA_MARCH AND NOT stringzilla_arch_wasm_)
    message(
        WARNING "STRINGZILLA_MARCH=${STRINGZILLA_MARCH}: baking specific instructions; result may not run elsewhere"
    )
    if (MSVC AND STRINGZILLA_MARCH STREQUAL "native")
        # MSVC has no `/arch:native`, so it takes its flags from the probes below.
    elseif (MSVC)
        message(
            FATAL_ERROR
                "MSVC takes STRINGZILLA_MARCH=native only; pass an /arch: value such as AVX2, AVX512 or AVX10.2 "
                "through CMAKE_C_FLAGS instead of `${STRINGZILLA_MARCH}`"
        )
    elseif (stringzilla_arch_arm64_)
        # AArch64 names CPUs through `-mcpu`, like `neoverse-v2`, as `-march` takes only architecture versions; Apple
        # Clang's `-march=native` also advertises just a subset of host features.
        set(stringzilla_host_arch_flags_ -mcpu=${STRINGZILLA_MARCH})
    else ()
        set(stringzilla_host_arch_flags_ -march=${STRINGZILLA_MARCH})
    endif ()
    set(stringzilla_unit_arch_flags_ ${stringzilla_host_arch_flags_})
elseif (stringzilla_arch_wasm_)
    # An engine validates a WebAssembly module whole, so every unit carries the module's one SIMD capability, whatever
    # flags a binding passes, and `types.h` reads it back off the predefined macros. Empty is the portable floor.
    set(stringzilla_wasm_simd_ ${STRINGZILLA_MARCH})
    if (NOT stringzilla_wasm_simd_)
        set(stringzilla_wasm_simd_ serial)
    endif ()
    if (stringzilla_wasm_simd_ STREQUAL "v128relaxed")
        add_compile_options(-msimd128 -mrelaxed-simd)
    elseif (stringzilla_wasm_simd_ STREQUAL "v128")
        add_compile_options(-msimd128)
    elseif (NOT stringzilla_wasm_simd_ STREQUAL "serial")
        message(FATAL_ERROR "STRINGZILLA_MARCH on WebAssembly: serial, v128 or v128relaxed, not `${STRINGZILLA_MARCH}`")
    endif ()
elseif (MSVC)
    if (stringzilla_arch_x8664_)
        set(stringzilla_unit_arch_flags_ /arch:SSE2)
    elseif (stringzilla_arch_arm64_)
        set(stringzilla_unit_arch_flags_ /arch:armv8.0)
    endif ()
elseif (APPLE)
    # `-arch` already pins the ABI floor of the one slice this build compiles.
elseif (stringzilla_arch_x8664_)
    set(stringzilla_unit_arch_flags_ -march=x86-64)
elseif (stringzilla_arch_arm64_)
    set(stringzilla_unit_arch_flags_ -march=armv8-a)
elseif (stringzilla_arch_riscv64_)
    set(stringzilla_unit_arch_flags_ -march=rv64gc)
elseif (stringzilla_arch_loongarch64_)
    set(stringzilla_unit_arch_flags_ -march=loongarch64)
elseif (stringzilla_arch_ppc64_)
    set(stringzilla_unit_arch_flags_ -mcpu=power8)
endif ()
add_compile_options("$<$<COMPILE_LANGUAGE:C,CXX>:${stringzilla_unit_arch_flags_}>")
# Every capability's kernels are written by hand, so auto-vectorizing the portable build only bloats its scalar code.
if (NOT STRINGZILLA_MARCH AND CMAKE_C_COMPILER_ID MATCHES "^(GNU|Clang|AppleClang|IntelLLVM)$")
    add_compile_options("$<$<COMPILE_LANGUAGE:C,CXX>:-fno-tree-vectorize;-fno-tree-slp-vectorize>")
endif ()
