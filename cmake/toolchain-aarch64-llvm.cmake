# Clang cross-compilation toolchain for 64-bit Arm (AArch64), driving `clang-23`.
#
# Usage: cmake --preset linux_arm64, then ctest --preset linux_arm64, which runs the static binaries
# under qemu-aarch64-static.
#
# Every Arm kit, NEON with its AES and SHA2 extensions and SVE, SVE2 and SVE2-AES, builds with
# Clang 20 or newer; with an older compiler a kit it cannot build probes as 0. Every unit compiles
# at the `armv8-a` floor, and each kit's kernels pick their own instructions per function.
#
# `-D AARCH64_QEMU_CPU=...`, or the environment variable, picks the emulated CPU: `max`, the
# default, has every extension QEMU knows, so the tests exercise every kit, while `cortex-a53`, an
# Armv8.0-A core, runs the floor alone.

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)

set(stringzilla_triple_ aarch64-linux-gnu)
set(stringzilla_sysroot_ /usr/${stringzilla_triple_})
set(stringzilla_gcc_library_directory_ /usr/lib/gcc-cross/${stringzilla_triple_}/13)

set(CMAKE_C_COMPILER clang-23)
set(CMAKE_CXX_COMPILER clang++-23)

set(stringzilla_target_flags_
    "--target=${stringzilla_triple_} --sysroot=${stringzilla_sysroot_} --gcc-toolchain=/usr -march=armv8-a"
)

set(CMAKE_C_FLAGS_INIT "${stringzilla_target_flags_}")
set(CMAKE_CXX_FLAGS_INIT "${stringzilla_target_flags_} -stdlib=libstdc++")

set(CMAKE_EXE_LINKER_FLAGS_INIT "-static -stdlib=libstdc++ -L${stringzilla_gcc_library_directory_} -lstdc++ -lm")

if (DEFINED ENV{AARCH64_QEMU_CPU})
    set(stringzilla_qemu_cpu_default_ "$ENV{AARCH64_QEMU_CPU}")
else ()
    set(stringzilla_qemu_cpu_default_ "max")
endif ()
set(AARCH64_QEMU_CPU
    "${stringzilla_qemu_cpu_default_}"
    CACHE STRING "CPU model for `qemu-aarch64 -cpu`"
)
# Nested try-compile projects reread this file and inherit the environment, not the cache.
set(ENV{AARCH64_QEMU_CPU} "${AARCH64_QEMU_CPU}")
set(CMAKE_CROSSCOMPILING_EMULATOR "qemu-aarch64-static;-cpu;${AARCH64_QEMU_CPU};-L;${stringzilla_sysroot_}")

set(CMAKE_FIND_ROOT_PATH ${stringzilla_sysroot_})
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)
