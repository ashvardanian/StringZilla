# Clang cross-compilation toolchain for x86_64, driving `clang-23` from a foreign Linux host.
#
# Usage: cmake --preset linux_x86_64, then ctest --preset linux_x86_64, which runs the static
# binaries under qemu-x86_64-static.
#
# `-D X86_QEMU_CPU=...`, or the environment variable, picks the emulated CPU: `max`, the default,
# has every extension QEMU knows, while `Haswell-noTSX-IBRS` runs that tier alone. QEMU's TCG has
# no AVX-512, so those tiers need Intel SDE on an x86 host.

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR x86_64)

set(stringzilla_triple_ x86_64-linux-gnu)
set(stringzilla_sysroot_ /usr/${stringzilla_triple_})
set(stringzilla_gcc_library_directory_ /usr/lib/gcc-cross/${stringzilla_triple_}/15)

set(CMAKE_C_COMPILER clang-23)
set(CMAKE_CXX_COMPILER clang++-23)

# No `--sysroot`: the distribution cross packages install into the build root, and their linker
# scripts name their libraries absolutely, which the linker would prefix with the sysroot again.
set(stringzilla_target_flags_ "--target=${stringzilla_triple_} --gcc-toolchain=/usr")

set(CMAKE_C_FLAGS_INIT "${stringzilla_target_flags_}")
set(CMAKE_CXX_FLAGS_INIT "${stringzilla_target_flags_} -stdlib=libstdc++")

# Clang drives the GNU linker directly and adds no `-lstdc++` of its own.
set(CMAKE_EXE_LINKER_FLAGS_INIT "-static -stdlib=libstdc++ -L${stringzilla_gcc_library_directory_} -lstdc++ -lm")

if (DEFINED ENV{X86_QEMU_CPU})
    set(stringzilla_qemu_cpu_default_ "$ENV{X86_QEMU_CPU}")
else ()
    set(stringzilla_qemu_cpu_default_ "max")
endif ()
set(X86_QEMU_CPU
    "${stringzilla_qemu_cpu_default_}"
    CACHE STRING "CPU model for `qemu-x86_64 -cpu`"
)
# Nested try-compile projects reread this file and inherit the environment, not the cache.
set(ENV{X86_QEMU_CPU} "${X86_QEMU_CPU}")
set(CMAKE_CROSSCOMPILING_EMULATOR "qemu-x86_64-static;-cpu;${X86_QEMU_CPU};-L;${stringzilla_sysroot_}")

# Look for headers/libraries inside the target sysroot, host tools on the host.
set(CMAKE_FIND_ROOT_PATH ${stringzilla_sysroot_})
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)
