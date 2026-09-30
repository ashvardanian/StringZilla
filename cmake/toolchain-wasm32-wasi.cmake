# Clang cross-compilation toolchain for WebAssembly (wasm32-wasip1), driving the wasi-sdk's Clang.
#
# Usage: cmake --preset wasm32_wasi, then ctest --preset wasm32_wasi, which runs each .wasm under
# the selected runtime.
#
# One module carries one SIMD kit, `STRINGZILLA_TARGET_ARCH`: `serial`, `v128`, or the default
# `v128relaxed`, taken from `-D` or the environment. `CMakeLists.txt` turns it into `-msimd128` and
# `-mrelaxed-simd` for every unit, as an engine validates a module whole. Both kits build with
# Clang 20 or newer.
#
# Single-threaded; a threaded variant lives in cmake/toolchain-wasm32-wasi-threads.cmake.
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

set(STRINGZILLA_TARGET_FLAGS "--target=wasm32-wasip1 --sysroot=${STRINGZILLA_SYSROOT}")

# Set here, ahead of `project()`, so the environment reaches it as it would `CMakeLists.txt`'s own default.
set(STRINGZILLA_TARGET_ARCH_DEFAULT_ "$ENV{STRINGZILLA_TARGET_ARCH}")
if (NOT STRINGZILLA_TARGET_ARCH_DEFAULT_)
    set(STRINGZILLA_TARGET_ARCH_DEFAULT_ "v128relaxed")
endif ()
set(STRINGZILLA_TARGET_ARCH
    "${STRINGZILLA_TARGET_ARCH_DEFAULT_}"
    CACHE STRING "The module's SIMD kit: serial, v128 or v128relaxed"
)

# The C++ test harness pulls in libc features wasm lacks natively; WASI provides emulation libraries, opted into with
# `-D_WASI_EMULATED_*` at compile time and `-lwasi-emulated-*` at link time. The kernels themselves need none of this.
set(STRINGZILLA_EMULATION_DEFS "-D_WASI_EMULATED_SIGNAL -D_WASI_EMULATED_MMAN -D_WASI_EMULATED_PROCESS_CLOCKS")
set(CMAKE_C_FLAGS_INIT "${STRINGZILLA_TARGET_FLAGS} ${STRINGZILLA_EMULATION_DEFS}")
# StringZilla's C++ API throws, so exceptions stay on through the wasm exception-handling proposal in its exnref
# flavor, which the wasi-sdk `eh` multilib and Wasmtime's `-W exceptions=y` both speak.
set(CMAKE_CXX_FLAGS_INIT "${STRINGZILLA_TARGET_FLAGS} ${STRINGZILLA_EMULATION_DEFS} -fwasm-exceptions \
     --start-no-unused-arguments -mllvm -wasm-use-legacy-eh=false --end-no-unused-arguments"
)
set(CMAKE_EXE_LINKER_FLAGS_INIT
    "-fwasm-exceptions -lunwind -lwasi-emulated-signal -lwasi-emulated-mman -lwasi-emulated-process-clocks"
)

# Choose the runtime that CTest invokes on each cross binary. Datasets live in the source tree, so map it into the
# guest via `--dir`; Wasmtime also inherits the environment the test presets set.
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
        "${STRINGZILLA_WASMTIME};-W;relaxed-simd=y;-W;exceptions=y;-S;inherit-env=y;--dir;${CMAKE_CURRENT_LIST_DIR}/.."
    )
endif ()

# Look for headers/libraries inside the target sysroot, host tools on the host.
set(CMAKE_FIND_ROOT_PATH ${STRINGZILLA_SYSROOT})
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)
