# Clang cross-compilation toolchain for LoongArch64 with the LASX 256-bit SIMD extension.
#
# Usage: cmake -B build_loongarch64 -DCMAKE_TOOLCHAIN_FILE=cmake/toolchains/loongarch64.cmake ... ctest --test-dir
# build_loongarch64   # runs cross binaries under qemu-loongarch64-static

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR loongarch64)

set(SZ_TRIPLE loongarch64-linux-gnu)
set(SZ_SYSROOT /usr/${SZ_TRIPLE})
set(SZ_GCC_LIBDIR /usr/lib/gcc-cross/${SZ_TRIPLE}/14)

set(CMAKE_C_COMPILER clang-23)
set(CMAKE_CXX_COMPILER clang++-23)

# `-mlasx` defines `__loongarch_asx` so `SZ_USE_LASX` auto-detects to 1.
set(SZ_TARGET_FLAGS "--target=${SZ_TRIPLE} --sysroot=${SZ_SYSROOT} --gcc-toolchain=/usr -mlasx")

set(CMAKE_C_FLAGS_INIT "${SZ_TARGET_FLAGS}")
set(CMAKE_CXX_FLAGS_INIT "${SZ_TARGET_FLAGS} -stdlib=libstdc++")

set(CMAKE_EXE_LINKER_FLAGS_INIT "-static -stdlib=libstdc++ -L${SZ_GCC_LIBDIR} -lstdc++ -lm")

set(CMAKE_CROSSCOMPILING_EMULATOR "qemu-loongarch64-static;-L;${SZ_SYSROOT}")

set(CMAKE_FIND_ROOT_PATH ${SZ_SYSROOT})
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)
