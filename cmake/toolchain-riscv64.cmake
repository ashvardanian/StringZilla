# Clang cross-compilation toolchain for 64-bit RISC-V with the RVV 1.0 Vector extension.
#
# Usage: cmake -B build_riscv64 -DCMAKE_TOOLCHAIN_FILE=cmake/toolchain-riscv64.cmake ... ctest --test-dir build_riscv64
# # runs cross binaries under qemu-riscv64-static
#
# Requires: clang-23 / clang++-23, the riscv64-linux-gnu sysroot + gcc-14 cross libstdc++, and qemu-riscv64-static. See
# cmake/toolchains/README is not needed — all paths below are absolute and validated for this machine.

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR riscv64)

set(STRINGZILLA_TRIPLE riscv64-linux-gnu)
set(STRINGZILLA_SYSROOT /usr/${STRINGZILLA_TRIPLE})
set(STRINGZILLA_GCC_LIBDIR /usr/lib/gcc-cross/${STRINGZILLA_TRIPLE}/14)

set(CMAKE_C_COMPILER clang-23)
set(CMAKE_CXX_COMPILER clang++-23)

# `-march=rv64gcv` turns on the V extension so `__riscv_vector` is defined and `STRINGZILLA_TARGET_RVV` auto-detects to 1 inside
# `types.h`.
set(STRINGZILLA_TARGET_FLAGS
    "--target=${STRINGZILLA_TRIPLE} --sysroot=${STRINGZILLA_SYSROOT} --gcc-toolchain=/usr -march=rv64gcv"
)

set(CMAKE_C_FLAGS_INIT "${STRINGZILLA_TARGET_FLAGS}")
set(CMAKE_CXX_FLAGS_INIT "${STRINGZILLA_TARGET_FLAGS} -stdlib=libstdc++")

# Static linking sidesteps qemu/dynamic-loader/sysroot headaches. We still must point clang at the cross libstdc++ and
# pull it in explicitly, because clang drives the GNU linker directly and does not add `-lstdc++` on its own.
set(CMAKE_EXE_LINKER_FLAGS_INIT "-static -stdlib=libstdc++ -L${STRINGZILLA_GCC_LIBDIR} -lstdc++ -lm")

# Run cross binaries (e.g. via `ctest`) under qemu with the V extension enabled.
set(CMAKE_CROSSCOMPILING_EMULATOR "qemu-riscv64-static;-cpu;rv64,v=true,vlen=128;-L;${STRINGZILLA_SYSROOT}")

# Look for headers/libraries inside the target sysroot, host tools on the host.
set(CMAKE_FIND_ROOT_PATH ${STRINGZILLA_SYSROOT})
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)
