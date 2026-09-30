# Clang cross-compilation toolchain for LoongArch64, driving `clang-23`.
#
# Usage: cmake --preset linux_loongarch64, then ctest --preset linux_loongarch64, which runs the
# static binaries under qemu-loongarch64-static.
#
# Every unit compiles at the generic `loongarch64` floor, without LSX or LASX; only the LASX unit,
# its probe and its header-only cross-checks add `-mlasx`, as `lasxintrin.h` hides without it.
#
# `-D LOONGARCH_QEMU_CPU=...`, or the environment variable, picks the emulated CPU: `la464`, the
# default, has LSX and LASX, so the tests exercise the kit, while `la464,lsx=off,lasx=off` runs the
# floor alone.

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR loongarch64)

set(STRINGZILLA_TRIPLE loongarch64-linux-gnu)
set(STRINGZILLA_SYSROOT /usr/${STRINGZILLA_TRIPLE})
set(STRINGZILLA_GCC_LIBDIR /usr/lib/gcc-cross/${STRINGZILLA_TRIPLE}/14)

set(CMAKE_C_COMPILER clang-23)
set(CMAKE_CXX_COMPILER clang++-23)

set(STRINGZILLA_TARGET_FLAGS
    "--target=${STRINGZILLA_TRIPLE} --sysroot=${STRINGZILLA_SYSROOT} --gcc-toolchain=/usr -march=loongarch64"
)

set(CMAKE_C_FLAGS_INIT "${STRINGZILLA_TARGET_FLAGS}")
set(CMAKE_CXX_FLAGS_INIT "${STRINGZILLA_TARGET_FLAGS} -stdlib=libstdc++")

set(CMAKE_EXE_LINKER_FLAGS_INIT "-static -stdlib=libstdc++ -L${STRINGZILLA_GCC_LIBDIR} -lstdc++ -lm")

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
set(CMAKE_CROSSCOMPILING_EMULATOR "qemu-loongarch64-static;-cpu;${LOONGARCH_QEMU_CPU};-L;${STRINGZILLA_SYSROOT}")

set(CMAKE_FIND_ROOT_PATH ${STRINGZILLA_SYSROOT})
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)
