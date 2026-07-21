# WASI toolchain for StringZilla — builds the C/C++ test & bench binaries for standalone WASM
# runtimes (Wasmtime/Wasmer/Node-WASI) so the v128 / v128relaxed kernels are exercised under wasm.
#
# Usage:
#   export WASI_SDK_PATH=~/wasi-sdk
#   cmake -B build_wasi -DCMAKE_TOOLCHAIN_FILE=cmake/toolchain-wasi.cmake \
#         -DSTRINGZILLA_BUILD_TEST=ON -DCMAKE_BUILD_TYPE=Release
#   cmake --build build_wasi
#   ctest --test-dir build_wasi   # runs each .wasm under Wasmtime

set(CMAKE_SYSTEM_NAME WASI)
set(CMAKE_SYSTEM_VERSION 1)
set(CMAKE_SYSTEM_PROCESSOR wasm32)

# Locate the WASI SDK.
if (NOT DEFINED WASI_SDK_PATH)
    if (DEFINED ENV{WASI_SDK_PATH})
        set(WASI_SDK_PATH "$ENV{WASI_SDK_PATH}")
    elseif (EXISTS "$ENV{HOME}/wasi-sdk")
        set(WASI_SDK_PATH "$ENV{HOME}/wasi-sdk")
    else ()
        message(FATAL_ERROR
            "WASI_SDK_PATH not set and ~/wasi-sdk not found.\n"
            "Download from https://github.com/WebAssembly/wasi-sdk/releases, then "
            "`export WASI_SDK_PATH=~/wasi-sdk`.")
    endif ()
endif ()
file(TO_CMAKE_PATH "${WASI_SDK_PATH}" WASI_SDK_PATH)

if (CMAKE_HOST_WIN32)
    set(WASI_TOOL_SUFFIX ".exe")
else ()
    set(WASI_TOOL_SUFFIX "")
endif ()

set(CMAKE_C_COMPILER "${WASI_SDK_PATH}/bin/clang${WASI_TOOL_SUFFIX}")
set(CMAKE_CXX_COMPILER "${WASI_SDK_PATH}/bin/clang++${WASI_TOOL_SUFFIX}")
set(CMAKE_AR "${WASI_SDK_PATH}/bin/llvm-ar${WASI_TOOL_SUFFIX}")
set(CMAKE_RANLIB "${WASI_SDK_PATH}/bin/llvm-ranlib${WASI_TOOL_SUFFIX}")
set(CMAKE_SYSROOT "${WASI_SDK_PATH}/share/wasi-sysroot")
set(CMAKE_FIND_ROOT_PATH "${WASI_SDK_PATH}")

# Single-threaded, self-contained memory: the SAME `.wasm` runs under wasmtime, wasmer, and node-WASI.
# StringZilla's single-string core (`stringzilla`) is single-threaded; the parallel `stringzillas`
# backend is not built for wasm.
set(SZ_WASI_TARGET_ "wasm32-wasip1")

# SIMD128 + relaxed-SIMD. `types.h` keys `SZ_USE_V128` / `SZ_USE_V128RELAXED` off the predefined
# `__wasm_simd128__` / `__wasm_relaxed_simd__` macros these flags set.
set(WASM_SIMD_FLAGS "-msimd128 -mrelaxed-simd")
set(CMAKE_C_FLAGS_INIT "${WASM_SIMD_FLAGS} --target=${SZ_WASI_TARGET_}")
# StringZilla's C++ API throws (e.g. `throw std::bad_alloc()` in stringzilla.hpp), so keep exceptions
# enabled via the wasm EH proposal (unlike NumKong's C-centric build, which compiles `-fno-exceptions`).
set(CMAKE_CXX_FLAGS_INIT "${WASM_SIMD_FLAGS} --target=${SZ_WASI_TARGET_} -fwasm-exceptions")

set(CMAKE_C_FLAGS_RELEASE "-O3 -DNDEBUG")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
set(CMAKE_C_FLAGS_DEBUG "-O0 -g")
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g")

# The C++ test harness pulls in libc features wasm lacks natively (signals); WASI provides minimal
# emulation libraries, opted into with `-D_WASI_EMULATED_*` at compile time and `-lwasi-emulated-*` at
# link time. The single-string kernels themselves need none of this.
set(WASM_EMULATION_DEFS "-D_WASI_EMULATED_SIGNAL -D_WASI_EMULATED_MMAN -D_WASI_EMULATED_PROCESS_CLOCKS")
set(CMAKE_C_FLAGS_INIT "${CMAKE_C_FLAGS_INIT} ${WASM_EMULATION_DEFS}")
set(CMAKE_CXX_FLAGS_INIT "${CMAKE_CXX_FLAGS_INIT} ${WASM_EMULATION_DEFS}")
set(CMAKE_EXE_LINKER_FLAGS_INIT
    "-Wl,--allow-undefined -Wl,--export=main -Wl,--export=_start \
     -lwasi-emulated-signal -lwasi-emulated-mman -lwasi-emulated-process-clocks")

set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)

if (EXISTS "${WASI_SDK_PATH}/VERSION")
    file(READ "${WASI_SDK_PATH}/VERSION" WASI_SDK_VERSION)
    string(STRIP "${WASI_SDK_VERSION}" WASI_SDK_VERSION)
    message(STATUS "StringZilla WASI: WASI-SDK ${WASI_SDK_VERSION}, SIMD128 + relaxed-SIMD enabled")
endif ()

# CTest runs each cross binary under Wasmtime (relaxed-SIMD enabled). Datasets live in the source tree,
# so map it into the guest via `--dir`.
find_program(SZ_WASMTIME_EXE_ wasmtime PATHS "$ENV{HOME}/.wasmtime/bin")
set(CMAKE_CROSSCOMPILING_EMULATOR "${SZ_WASMTIME_EXE_};-W;relaxed-simd=y;--dir;${CMAKE_CURRENT_LIST_DIR}/..")
