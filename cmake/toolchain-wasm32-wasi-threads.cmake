# Clang cross-compilation toolchain for WebAssembly with threads (wasm32-wasip1-threads), driving
# the wasi-sdk's Clang.
#
# Usage: cmake --preset wasm32_wasi_threads, then ctest --preset wasm32_wasi_threads, which runs
# each .wasm under Wasmtime. The module's one SIMD kit, `v128relaxed` by default, builds with
# Clang 20 or newer.
#
# Threads are a whole-module choice: `-pthread` puts every worker on its own wasi-thread over one imported shared
# memory. Nothing in the library needs it today, so the single-threaded cmake/toolchain-wasm32-wasi.cmake is the usual
# one; this file repeats its SIMD kit and exception handling for a caller that shards work across threads itself.
#
# Only Wasmtime runs wasi-threads modules, with `-W threads=y -W shared-memory=y -S threads=y`, and only before 48,
# which removed wasi-threads; there is no runtime choice here.
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
        file(GLOB stringzilla_wasi_sdk_candidates_ "/opt/wasi-sdk*")
        list(SORT stringzilla_wasi_sdk_candidates_)
        list(POP_BACK stringzilla_wasi_sdk_candidates_ WASI_SDK_PREFIX)
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
set(stringzilla_sysroot_ "${WASI_SDK_PREFIX}/share/wasi-sysroot")

set(stringzilla_target_flags_ "--target=wasm32-wasip1-threads --sysroot=${stringzilla_sysroot_} -pthread")

# Set here, ahead of `project()`, so the environment reaches it as it would `CMakeLists.txt`'s own default.
set(stringzilla_march_default_ "$ENV{STRINGZILLA_MARCH}")
if (NOT stringzilla_march_default_)
    set(stringzilla_march_default_ "v128relaxed")
endif ()
set(STRINGZILLA_MARCH
    "${stringzilla_march_default_}"
    CACHE STRING "The module's SIMD kit: serial, v128 or v128relaxed"
)

set(stringzilla_emulation_definitions_ "-D_WASI_EMULATED_SIGNAL -D_WASI_EMULATED_MMAN -D_WASI_EMULATED_PROCESS_CLOCKS")
set(CMAKE_C_FLAGS_INIT "${stringzilla_target_flags_} ${stringzilla_emulation_definitions_}")
set(CMAKE_CXX_FLAGS_INIT "${stringzilla_target_flags_} ${stringzilla_emulation_definitions_} -fwasm-exceptions \
     --start-no-unused-arguments -mllvm -wasm-use-legacy-eh=false --end-no-unused-arguments"
)
# wasi-threads imports the shared memory from the host and still exports it for WASI itself.
set(CMAKE_EXE_LINKER_FLAGS_INIT
    "-pthread -Wl,--import-memory -Wl,--export-memory -Wl,--max-memory=2147483648 -fwasm-exceptions -lunwind \
     -lwasi-emulated-signal -lwasi-emulated-mman -lwasi-emulated-process-clocks"
)

# CTest runs each cross binary under Wasmtime with the threads proposal and the WASI threading imports on. Datasets
# live in the source tree, so map it into the guest via `--dir`, and inherit the environment the test presets set.
find_program(stringzilla_wasmtime_ wasmtime PATHS "$ENV{HOME}/.wasmtime/bin")
set(CMAKE_CROSSCOMPILING_EMULATOR
    "${stringzilla_wasmtime_};-W;relaxed-simd=y;-W;threads=y;-W;shared-memory=y;-W;exceptions=y;-S;threads=y;-S;inherit-env=y;--dir;${CMAKE_CURRENT_LIST_DIR}/.."
)

# Look for headers/libraries inside the target sysroot, host tools on the host.
set(CMAKE_FIND_ROOT_PATH ${stringzilla_sysroot_})
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)
