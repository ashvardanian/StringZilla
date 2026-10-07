# GCC cross-compilation toolchain for LoongArch64, driving `loongarch64-linux-gnu-gcc-15`.
#
# Usage: cmake --preset linux_loongarch64, then ctest --preset linux_loongarch64, which runs the
# static binaries under qemu-loongarch64.
#
# GCC 15 is the floor: its `#pragma GCC target("lasx")` opens `lasxintrin.h` inside the LASX kernels' region, so
# every unit compiles at the generic `loongarch64` floor and the LASX kernels scope themselves per function, as
# every other capability does. GCC 14 has no LoongArch target pragma and Clang never opens the header without
# `-mlasx`, so either leaves LASX off.
#
# `-D LOONGARCH_QEMU_CPU=...`, or the environment variable, picks the emulated CPU: `la464`, the
# default, has LSX and LASX, so the tests exercise the capability, while `la464,lsx=off,lasx=off`
# runs the floor alone.

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR loongarch64)

set(STRINGZILLA_TRIPLE loongarch64-linux-gnu)
set(STRINGZILLA_SYSROOT /usr/${STRINGZILLA_TRIPLE})

set(CMAKE_C_COMPILER ${STRINGZILLA_TRIPLE}-gcc-15)
set(CMAKE_CXX_COMPILER ${STRINGZILLA_TRIPLE}-g++-15)

set(CMAKE_C_FLAGS_INIT "-march=loongarch64")
set(CMAKE_CXX_FLAGS_INIT "-march=loongarch64")
set(CMAKE_EXE_LINKER_FLAGS_INIT "-static")

if (DEFINED ENV{LOONGARCH_QEMU_CPU})
    set(STRINGZILLA_QEMU_CPU_DEFAULT_ "$ENV{LOONGARCH_QEMU_CPU}")
else ()
    set(STRINGZILLA_QEMU_CPU_DEFAULT_ "la464")
endif ()
set(LOONGARCH_QEMU_CPU
    "${STRINGZILLA_QEMU_CPU_DEFAULT_}"
    CACHE STRING "CPU model for `qemu-loongarch64 -cpu`"
)
# Nested try-compile projects reread this file and inherit the environment, not the cache.
set(ENV{LOONGARCH_QEMU_CPU} "${LOONGARCH_QEMU_CPU}")
set(CMAKE_CROSSCOMPILING_EMULATOR "qemu-loongarch64;-cpu;${LOONGARCH_QEMU_CPU};-L;${STRINGZILLA_SYSROOT}")

set(CMAKE_FIND_ROOT_PATH ${STRINGZILLA_SYSROOT})
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)
