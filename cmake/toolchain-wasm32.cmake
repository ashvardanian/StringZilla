# Clang cross-compilation toolchain for WebAssembly (wasm32) with SIMD128 + relaxed-SIMD.
#
# Usage: cmake -B build_wasm -DCMAKE_TOOLCHAIN_FILE=cmake/toolchain-wasm32.cmake ... ; ctest --test-dir build_wasm
# # builds the wasm32 test binaries and runs each .wasm under the selected runtime via CTest.
#
# Pick the runtime with -DSZ_WASM_RUNTIME=wasmtime (default) or =wasmer; both execute a `.wasm` WASI command
# directly, so CTest can use them as a CMAKE_CROSSCOMPILING_EMULATOR with no wrapper script. (node cannot run a
# bare `.wasm` from the CLI, so it is intentionally not wired here.)
#
# Requires the wasi-sdk (self-contained clang + wasi-sysroot + libc). Point at it with -DWASI_SDK_PREFIX=... or the
# WASI_SDK_PREFIX environment variable; the default falls back to ~/wasi-sdk then /opt/wasi-sdk. Using the wasi-sdk
# clang keeps the toolchain dependency-free: no merged resource directory, no separate builtins archive.

set(CMAKE_SYSTEM_NAME WASI)
set(CMAKE_SYSTEM_PROCESSOR wasm32)

# Locate the wasi-sdk (cache var > environment > common install locations).
if(NOT DEFINED WASI_SDK_PREFIX)
    if(DEFINED ENV{WASI_SDK_PREFIX})
        set(WASI_SDK_PREFIX "$ENV{WASI_SDK_PREFIX}")
    elseif(EXISTS "$ENV{HOME}/wasi-sdk/bin/clang")
        set(WASI_SDK_PREFIX "$ENV{HOME}/wasi-sdk")
    else()
        set(WASI_SDK_PREFIX "/opt/wasi-sdk")
    endif()
endif()

set(CMAKE_C_COMPILER "${WASI_SDK_PREFIX}/bin/clang")
set(CMAKE_CXX_COMPILER "${WASI_SDK_PREFIX}/bin/clang++")
set(SZ_SYSROOT "${WASI_SDK_PREFIX}/share/wasi-sysroot")

# `-msimd128` defines `__wasm_simd128__` so `SZ_USE_V128` auto-detects to 1 in types.h; `-mrelaxed-simd` likewise
# enables `SZ_USE_V128RELAXED`. Both backends are therefore exercised by the same build.
set(SZ_TARGET_FLAGS "--target=wasm32-wasip1 --sysroot=${SZ_SYSROOT} -msimd128 -mrelaxed-simd")
set(CMAKE_C_FLAGS_INIT "${SZ_TARGET_FLAGS}")
set(CMAKE_CXX_FLAGS_INIT "${SZ_TARGET_FLAGS}")

# Choose the runtime that CTest invokes on each cross binary.
set(SZ_WASM_RUNTIME "wasmtime" CACHE STRING "WASM runtime used to run tests via CTest (wasmtime or wasmer)")
if(SZ_WASM_RUNTIME STREQUAL "wasmer")
    find_program(SZ_WASMER wasmer PATHS "$ENV{HOME}/.cargo/bin" "$ENV{HOME}/.wasmer/bin")
    set(CMAKE_CROSSCOMPILING_EMULATOR "${SZ_WASMER};run")
else()
    find_program(SZ_WASMTIME wasmtime PATHS "$ENV{HOME}/.wasmtime/bin")
    set(CMAKE_CROSSCOMPILING_EMULATOR "${SZ_WASMTIME}")
endif()

# Look for headers/libraries inside the target sysroot, host tools on the host.
set(CMAKE_FIND_ROOT_PATH ${SZ_SYSROOT})
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)
