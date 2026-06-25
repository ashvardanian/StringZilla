# Clang cross-compilation toolchain for 64-bit Arm (AArch64) exercising the full NEON + crypto + SVE/SVE2 + SVE2-AES
# feature set.
#
# Usage: cmake -B build_aarch64 -DCMAKE_TOOLCHAIN_FILE=cmake/toolchains/aarch64.cmake ... ctest --test-dir build_aarch64
# # runs cross binaries under qemu-aarch64-static
#
# `-march=armv9-a+sve2+sve2-aes+sha3+aes` makes clang define `__ARM_NEON`, `__ARM_FEATURE_SVE`, `__ARM_FEATURE_SVE2`,
# `__ARM_FEATURE_SVE2AES`, `__ARM_FEATURE_AES` and `__ARM_FEATURE_SHA2`, so the NEON / ARMV8AES / ARMV8SHA / SVE / SVE2
# / SVE2AES backends in `types.h` all auto-detect to 1.

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)

set(SZ_TRIPLE aarch64-linux-gnu)
set(SZ_SYSROOT /usr/${SZ_TRIPLE})
set(SZ_GCC_LIBDIR /usr/lib/gcc-cross/${SZ_TRIPLE}/13)

set(CMAKE_C_COMPILER clang-23)
set(CMAKE_CXX_COMPILER clang++-23)

set(SZ_TARGET_FLAGS
    "--target=${SZ_TRIPLE} --sysroot=${SZ_SYSROOT} --gcc-toolchain=/usr -march=armv9-a+sve2+sve2-aes+sha3+aes"
)

set(CMAKE_C_FLAGS_INIT "${SZ_TARGET_FLAGS}")
set(CMAKE_CXX_FLAGS_INIT "${SZ_TARGET_FLAGS} -stdlib=libstdc++")

set(CMAKE_EXE_LINKER_FLAGS_INIT "-static -stdlib=libstdc++ -L${SZ_GCC_LIBDIR} -lstdc++ -lm")

# Run under qemu with an SVE2-capable CPU model so the SVE/SVE2 paths execute.
set(CMAKE_CROSSCOMPILING_EMULATOR "qemu-aarch64-static;-cpu;max,sve=on,sve256=on;-L;${SZ_SYSROOT}")

set(CMAKE_FIND_ROOT_PATH ${SZ_SYSROOT})
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)
