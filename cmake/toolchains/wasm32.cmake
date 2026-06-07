# Clang cross-compilation toolchain for WebAssembly (wasm32) with SIMD128.
#
# WebAssembly has no full libc/loader in this environment that CMake's compiler checks expect, so this toolchain is
# intended for *compile-checking* the headers (and a self-contained probe), NOT for driving the full CMake test/library
# suite. Functional WASM validation compiles the kernel-vs-serial probe against the wasi-sdk sysroot and runs it under
# wasmtime + node for both `-msimd128` (v128) and `-msimd128 -mrelaxed-simd` (v128relaxed).

set(CMAKE_SYSTEM_NAME WASI)
set(CMAKE_SYSTEM_PROCESSOR wasm32)

set(CMAKE_C_COMPILER clang-23)
set(CMAKE_CXX_COMPILER clang++-23)

# A wasi-sdk sysroot supplies libc; clang-23's own resource dir lacks the wasm builtins archive, so we redirect the
# resource dir to one that combines clang-23 headers with the wasi-sdk builtins.
set(SZ_WASI_SYSROOT /home/ubuntu/wasi-sdk/share/wasi-sysroot)
set(SZ_WASM_RESOURCE_DIR /home/ubuntu/.sz-wasm-resource)

set(SZ_TARGET_FLAGS
    "--target=wasm32-wasip1 --sysroot=${SZ_WASI_SYSROOT} -resource-dir=${SZ_WASM_RESOURCE_DIR} -msimd128"
)

set(CMAKE_C_FLAGS_INIT "${SZ_TARGET_FLAGS}")
set(CMAKE_CXX_FLAGS_INIT "${SZ_TARGET_FLAGS}")

set(CMAKE_CROSSCOMPILING_EMULATOR "/home/ubuntu/.wasmtime/bin/wasmtime")

set(CMAKE_FIND_ROOT_PATH ${SZ_WASI_SYSROOT})
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)
