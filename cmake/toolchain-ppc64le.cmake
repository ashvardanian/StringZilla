# Clang cross-compilation toolchain for little-endian 64-bit PowerPC (POWER9) with VSX.
#
# Usage: cmake -B build_ppc64le -DCMAKE_TOOLCHAIN_FILE=cmake/toolchain-ppc64le.cmake ... ctest --test-dir build_ppc64le
# # runs cross binaries under qemu-ppc64le-static

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR ppc64le)

set(SZ_TRIPLE powerpc64le-linux-gnu)
set(SZ_SYSROOT /usr/${SZ_TRIPLE})
set(SZ_GCC_LIBDIR /usr/lib/gcc-cross/${SZ_TRIPLE}/13)

set(CMAKE_C_COMPILER clang-23)
set(CMAKE_CXX_COMPILER clang++-23)

# `-mcpu=power9 -mvsx` defines `__VSX__` so `SZ_USE_POWER` auto-detects to 1. POWER9 also exposes the AES (`vcipher`)
# builtins used by the real `sz_hash_power`.
#
# Note: this machine has BOTH gcc-cross 13 and a partial 14 for powerpc64le, but only 13 ships the C++
# headers/`bits/c++config.h`. `--gcc-toolchain` would auto-pick 14 and fail, so we pin the GCC install explicitly with
# `--gcc-install-dir`. The Power backend relies on implicit VSX vector-type conversions (`__vector unsigned long long`
# <-> `__vector unsigned char`). clang flags these via `-Wdeprecate-lax-vec-conv-all`, which the project's `-Werror`
# would turn fatal, so we both allow lax conversions and silence that specific deprecation warning.
set(SZ_TARGET_FLAGS
    "--target=${SZ_TRIPLE} --sysroot=${SZ_SYSROOT} --gcc-install-dir=${SZ_GCC_LIBDIR} -mcpu=power9 -mvsx -flax-vector-conversions -Wno-deprecate-lax-vec-conv-all"
)

set(CMAKE_C_FLAGS_INIT "${SZ_TARGET_FLAGS}")
set(CMAKE_CXX_FLAGS_INIT "${SZ_TARGET_FLAGS} -stdlib=libstdc++")

set(CMAKE_EXE_LINKER_FLAGS_INIT "-static -stdlib=libstdc++ -L${SZ_GCC_LIBDIR} -lstdc++ -lm")

set(CMAKE_CROSSCOMPILING_EMULATOR "qemu-ppc64le-static;-L;${SZ_SYSROOT}")

set(CMAKE_FIND_ROOT_PATH ${SZ_SYSROOT})
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)
