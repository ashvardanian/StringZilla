# Clang cross-compilation toolchain for WebAssembly (wasm32-wasip1) with SIMD128, plus relaxed-SIMD by default.
#
# Usage: cmake -B build_wasm -DCMAKE_TOOLCHAIN_FILE=cmake/toolchain-wasm32.cmake ... ; ctest --test-dir build_wasm
# # builds the wasm32 test binaries and runs each .wasm under the selected runtime via CTest.
#
# One module carries one SIMD tier. `-msimd128` defines `__wasm_simd128__`, so `STRINGZILLA_TARGET_V128` auto-detects to 1 in
# types.h, and `-mrelaxed-simd` likewise enables `STRINGZILLA_TARGET_V128RELAXED`. Pass -DSTRINGZILLA_TARGET_V128RELAXED=0 for the strict
# module: the same override that zeroes the macro also drops the flag here, so no relaxed opcode reaches the binary.
#
# Single-threaded; a threaded variant lives in cmake/toolchain-wasm32-threads.cmake.
#
# Pick the runtime with -DSTRINGZILLA_WASM_RUNTIME=wasmtime (default) or =wasmer; both execute a `.wasm` WASI command
# directly, so CTest can use them as a CMAKE_CROSSCOMPILING_EMULATOR with no wrapper script. (node cannot run a
# bare `.wasm` from the CLI, so it is intentionally not wired here.)
#
# Requires the wasi-sdk (self-contained clang + wasi-sysroot + libc). Point at it with -DWASI_SDK_PREFIX=... or the
# WASI_SDK_PREFIX / WASI_SDK_PATH environment variables; the default falls back to ~/wasi-sdk then /opt/wasi-sdk*.
# Using the wasi-sdk clang keeps the toolchain dependency-free: no merged resource directory, no separate builtins
# archive.

set(CMAKE_SYSTEM_NAME WASI)
set(CMAKE_SYSTEM_PROCESSOR wasm32)

# Locate the wasi-sdk (cache var > environment > common install locations).
if (NOT DEFINED WASI_SDK_PREFIX)
    if (DEFINED ENV{WASI_SDK_PREFIX})
        set(WASI_SDK_PREFIX "$ENV{WASI_SDK_PREFIX}")
    elseif (DEFINED ENV{WASI_SDK_PATH})
        set(WASI_SDK_PREFIX "$ENV{WASI_SDK_PATH}")
    elseif (EXISTS "$ENV{HOME}/wasi-sdk/bin/clang")
        set(WASI_SDK_PREFIX "$ENV{HOME}/wasi-sdk")
    else ()
        file(GLOB STRINGZILLA_WASI_SDK_CANDIDATES_ "/opt/wasi-sdk*")
        list(SORT STRINGZILLA_WASI_SDK_CANDIDATES_)
        list(POP_BACK STRINGZILLA_WASI_SDK_CANDIDATES_ WASI_SDK_PREFIX)
        if (NOT WASI_SDK_PREFIX)
            set(WASI_SDK_PREFIX "/opt/wasi-sdk")
        endif ()
    endif ()
endif ()

# Nested try-compile projects reread this file and inherit the environment, but not this cache variable.
set(ENV{WASI_SDK_PREFIX} "${WASI_SDK_PREFIX}")

set(CMAKE_C_COMPILER "${WASI_SDK_PREFIX}/bin/clang")
set(CMAKE_CXX_COMPILER "${WASI_SDK_PREFIX}/bin/clang++")
set(CMAKE_AR "${WASI_SDK_PREFIX}/bin/llvm-ar")
set(CMAKE_RANLIB "${WASI_SDK_PREFIX}/bin/llvm-ranlib")
set(STRINGZILLA_SYSROOT "${WASI_SDK_PREFIX}/share/wasi-sysroot")

set(STRINGZILLA_TARGET_FLAGS "--target=wasm32-wasip1 --sysroot=${STRINGZILLA_SYSROOT} -msimd128")
if (NOT DEFINED STRINGZILLA_TARGET_V128RELAXED OR STRINGZILLA_TARGET_V128RELAXED)
    string(APPEND STRINGZILLA_TARGET_FLAGS " -mrelaxed-simd")
endif ()

# The C++ test harness pulls in libc features wasm lacks natively; WASI provides emulation libraries, opted into with
# `-D_WASI_EMULATED_*` at compile time and `-lwasi-emulated-*` at link time. The kernels themselves need none of this.
set(STRINGZILLA_EMULATION_DEFS "-D_WASI_EMULATED_SIGNAL -D_WASI_EMULATED_MMAN -D_WASI_EMULATED_PROCESS_CLOCKS")
set(CMAKE_C_FLAGS_INIT "${STRINGZILLA_TARGET_FLAGS} ${STRINGZILLA_EMULATION_DEFS}")
# StringZilla's C++ API throws, so exceptions stay on through the wasm exception-handling proposal in its exnref
# flavor, which the wasi-sdk `eh` multilib and Wasmtime's `-W exceptions=y` both speak.
set(CMAKE_CXX_FLAGS_INIT
    "${STRINGZILLA_TARGET_FLAGS} ${STRINGZILLA_EMULATION_DEFS} -fwasm-exceptions -mllvm -wasm-use-legacy-eh=false"
)
set(CMAKE_EXE_LINKER_FLAGS_INIT
    "-fwasm-exceptions -lunwind -lwasi-emulated-signal -lwasi-emulated-mman -lwasi-emulated-process-clocks"
)

# Choose the runtime that CTest invokes on each cross binary. Datasets live in the source tree, so map it into the
# guest via `--dir`.
set(STRINGZILLA_WASM_RUNTIME
    "wasmtime"
    CACHE STRING "WASM runtime used to run tests via CTest (wasmtime or wasmer)"
)
if (STRINGZILLA_WASM_RUNTIME STREQUAL "wasmer")
    find_program(STRINGZILLA_WASMER wasmer PATHS "$ENV{HOME}/.cargo/bin" "$ENV{HOME}/.wasmer/bin")
    set(CMAKE_CROSSCOMPILING_EMULATOR "${STRINGZILLA_WASMER};run;--dir;${CMAKE_CURRENT_LIST_DIR}/..")
else ()
    find_program(STRINGZILLA_WASMTIME wasmtime PATHS "$ENV{HOME}/.wasmtime/bin")
    set(CMAKE_CROSSCOMPILING_EMULATOR
        "${STRINGZILLA_WASMTIME};-W;relaxed-simd=y;-W;exceptions=y;--dir;${CMAKE_CURRENT_LIST_DIR}/.."
    )
endif ()

# Look for headers/libraries inside the target sysroot, host tools on the host.
set(CMAKE_FIND_ROOT_PATH ${STRINGZILLA_SYSROOT})
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)
