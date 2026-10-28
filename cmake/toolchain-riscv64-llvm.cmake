# Clang cross-compilation toolchain for 64-bit RISC-V, driving `clang-23`.
#
# Usage: cmake --preset linux_riscv64, then ctest --preset linux_riscv64, which runs the static
# binaries under qemu-riscv64-static.
#
# The RVV and RVV-crypto kernels need Clang 21 or newer; with an older compiler those kits probe as
# 0. Every unit compiles at the `rv64gc` floor, and each kit's kernels pick their own extensions per
# function.
#
# `-D RISCV_QEMU_CPU=...`, or the environment variable, picks the emulated CPU: `max`, the
# default, has every extension QEMU knows, so the tests exercise every kit, while `rv64` runs the
# floor alone.

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR riscv64)

set(stringzilla_triple_ riscv64-linux-gnu)
set(stringzilla_sysroot_ /usr/${stringzilla_triple_})
set(stringzilla_gcc_library_directory_ /usr/lib/gcc-cross/${stringzilla_triple_}/14)

set(CMAKE_C_COMPILER clang-23)
set(CMAKE_CXX_COMPILER clang++-23)

# No `--sysroot`: the distribution cross packages install into the build root, and their `libc.so` script names
# `/usr/riscv64-linux-gnu/lib` absolutely, which the linker would prefix with the sysroot a second time.
set(stringzilla_target_flags_ "--target=${stringzilla_triple_} --gcc-toolchain=/usr -march=rv64gc")

set(CMAKE_C_FLAGS_INIT "${stringzilla_target_flags_}")
set(CMAKE_CXX_FLAGS_INIT "${stringzilla_target_flags_} -stdlib=libstdc++")

# Static linking sidesteps qemu/dynamic-loader/sysroot headaches. We still must point clang at the cross libstdc++ and
# pull it in explicitly, because clang drives the GNU linker directly and does not add `-lstdc++` on its own.
set(CMAKE_EXE_LINKER_FLAGS_INIT "-static -stdlib=libstdc++ -L${stringzilla_gcc_library_directory_} -lstdc++ -lm")

if (DEFINED ENV{RISCV_QEMU_CPU})
    set(stringzilla_qemu_cpu_default_ "$ENV{RISCV_QEMU_CPU}")
else ()
    set(stringzilla_qemu_cpu_default_ "max")
endif ()
set(RISCV_QEMU_CPU
    "${stringzilla_qemu_cpu_default_}"
    CACHE STRING "CPU model for `qemu-riscv64 -cpu`"
)
# Nested try-compile projects reread this file and inherit the environment, not the cache.
set(ENV{RISCV_QEMU_CPU} "${RISCV_QEMU_CPU}")
set(CMAKE_CROSSCOMPILING_EMULATOR "qemu-riscv64-static;-cpu;${RISCV_QEMU_CPU};-L;${stringzilla_sysroot_}")

# Look for headers/libraries inside the target sysroot, host tools on the host.
set(CMAKE_FIND_ROOT_PATH ${stringzilla_sysroot_})
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)
