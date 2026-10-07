# GCC cross-compilation toolchain for little-endian 64-bit PowerPC, driving `powerpc64le-linux-gnu-gcc-15`.
#
# Usage: cmake --preset linux_ppc64le, then ctest --preset linux_ppc64le, which runs the static
# binaries under qemu-ppc64le.
#
# GCC 15 is the floor: its `#pragma GCC target("power9-vector")` opens the POWER9 half of `altivec.h` inside the POWER9
# kernels' region, so every unit compiles at the `power8` floor and those kernels scope themselves per function, as
# every other capability does. Clang never opens it without `-mcpu=power9`, so it leaves POWER9 off.
#
# `-D PPC_QEMU_CPU=...`, or the environment variable, picks the emulated CPU: `power10`, the
# default, runs every capability, while `power8` runs the floor alone.

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR ppc64le)

set(STRINGZILLA_TRIPLE powerpc64le-linux-gnu)
set(STRINGZILLA_SYSROOT /usr/${STRINGZILLA_TRIPLE})

set(CMAKE_C_COMPILER ${STRINGZILLA_TRIPLE}-gcc-15)
set(CMAKE_CXX_COMPILER ${STRINGZILLA_TRIPLE}-g++-15)

set(CMAKE_C_FLAGS_INIT "-mcpu=power8")
set(CMAKE_CXX_FLAGS_INIT "-mcpu=power8")
set(CMAKE_EXE_LINKER_FLAGS_INIT "-static")

if (DEFINED ENV{PPC_QEMU_CPU})
    set(STRINGZILLA_QEMU_CPU_DEFAULT_ "$ENV{PPC_QEMU_CPU}")
else ()
    set(STRINGZILLA_QEMU_CPU_DEFAULT_ "power10")
endif ()
set(PPC_QEMU_CPU
    "${STRINGZILLA_QEMU_CPU_DEFAULT_}"
    CACHE STRING "CPU model for `qemu-ppc64le -cpu`"
)
# Nested try-compile projects reread this file and inherit the environment, not the cache.
set(ENV{PPC_QEMU_CPU} "${PPC_QEMU_CPU}")
set(CMAKE_CROSSCOMPILING_EMULATOR "qemu-ppc64le;-cpu;${PPC_QEMU_CPU};-L;${STRINGZILLA_SYSROOT}")

set(CMAKE_FIND_ROOT_PATH ${STRINGZILLA_SYSROOT})
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)
