# Clang cross-compilation toolchain for LoongArch64 with the LASX 256-bit SIMD extension.
#
# Usage: cmake -B build_loongarch64 -DCMAKE_TOOLCHAIN_FILE=cmake/toolchain-loongarch64.cmake ... ctest --test-dir
# build_loongarch64   # runs cross binaries under qemu-loongarch64-static

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR loongarch64)

set(STRINGZILLA_TRIPLE loongarch64-linux-gnu)
set(STRINGZILLA_SYSROOT /usr/${STRINGZILLA_TRIPLE})
set(STRINGZILLA_GCC_LIBDIR /usr/lib/gcc-cross/${STRINGZILLA_TRIPLE}/14)

set(CMAKE_C_COMPILER clang-23)
set(CMAKE_CXX_COMPILER clang++-23)

# `-mlasx` defines `__loongarch_asx` so `STRINGZILLA_TARGET_LASX` auto-detects to 1.
set(STRINGZILLA_TARGET_FLAGS
    "--target=${STRINGZILLA_TRIPLE} --sysroot=${STRINGZILLA_SYSROOT} --gcc-toolchain=/usr -mlasx"
)

set(CMAKE_C_FLAGS_INIT "${STRINGZILLA_TARGET_FLAGS}")
set(CMAKE_CXX_FLAGS_INIT "${STRINGZILLA_TARGET_FLAGS} -stdlib=libstdc++")

set(CMAKE_EXE_LINKER_FLAGS_INIT "-static -stdlib=libstdc++ -L${STRINGZILLA_GCC_LIBDIR} -lstdc++ -lm")

set(CMAKE_CROSSCOMPILING_EMULATOR "qemu-loongarch64-static;-L;${STRINGZILLA_SYSROOT}")

set(CMAKE_FIND_ROOT_PATH ${STRINGZILLA_SYSROOT})
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)
