# Clang cross-compilation toolchain for WebAssembly (wasm32-wasip1-threads) with SIMD128 and relaxed-SIMD.
#
# Usage: cmake -B build_wasm_threads -DCMAKE_TOOLCHAIN_FILE=cmake/toolchain-wasm32-threads.cmake ... ; ctest --test-dir build_wasm_threads
# # builds the wasm32 test binaries and runs each .wasm under Wasmtime.
#
# Threads are a whole-module choice: `-pthread` puts every worker on its own wasi-thread over one imported shared
# memory. Nothing in the library needs it today, so the single-threaded cmake/toolchain-wasm32.cmake is the usual
# one; this file repeats its SIMD tier and exception handling for a caller that shards work across threads itself.
#
# Only Wasmtime runs wasi-threads modules today, with `-W threads=y -S threads=y`; there is no runtime choice here.
# Shared libraries stay off: WASI has no dynamic loader, so the tests link statically.
#
# Requires the wasi-sdk (self-contained clang + wasi-sysroot + libc). Point at it with -DWASI_SDK_PREFIX=... or the
# WASI_SDK_PREFIX / WASI_SDK_PATH environment variables; the default falls back to ~/wasi-sdk then /opt/wasi-sdk*.

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

set(STRINGZILLA_TARGET_FLAGS "--target=wasm32-wasip1-threads --sysroot=${STRINGZILLA_SYSROOT} -pthread -msimd128")
if (NOT DEFINED STRINGZILLA_TARGET_V128RELAXED OR STRINGZILLA_TARGET_V128RELAXED)
    string(APPEND STRINGZILLA_TARGET_FLAGS " -mrelaxed-simd")
endif ()

set(STRINGZILLA_EMULATION_DEFS "-D_WASI_EMULATED_SIGNAL -D_WASI_EMULATED_MMAN -D_WASI_EMULATED_PROCESS_CLOCKS")
set(CMAKE_C_FLAGS_INIT "${STRINGZILLA_TARGET_FLAGS} ${STRINGZILLA_EMULATION_DEFS}")
set(CMAKE_CXX_FLAGS_INIT
    "${STRINGZILLA_TARGET_FLAGS} ${STRINGZILLA_EMULATION_DEFS} -fwasm-exceptions -mllvm -wasm-use-legacy-eh=false"
)
# wasi-threads imports the shared memory from the host and still exports it for WASI itself.
set(CMAKE_EXE_LINKER_FLAGS_INIT
    "-pthread -Wl,--import-memory -Wl,--export-memory -Wl,--max-memory=2147483648 -fwasm-exceptions -lunwind \
     -lwasi-emulated-signal -lwasi-emulated-mman -lwasi-emulated-process-clocks"
)

# CTest runs each cross binary under Wasmtime with the threads proposal and the WASI threading imports on. Datasets
# live in the source tree, so map it into the guest via `--dir`.
find_program(STRINGZILLA_WASMTIME wasmtime PATHS "$ENV{HOME}/.wasmtime/bin")
set(CMAKE_CROSSCOMPILING_EMULATOR
    "${STRINGZILLA_WASMTIME};-W;relaxed-simd=y;-W;threads=y;-W;exceptions=y;-S;threads=y;--dir;${CMAKE_CURRENT_LIST_DIR}/.."
)

# Look for headers/libraries inside the target sysroot, host tools on the host.
set(CMAKE_FIND_ROOT_PATH ${STRINGZILLA_SYSROOT})
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)
