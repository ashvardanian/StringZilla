/**
 *  @file probes/run_capabilities.c
 *  @author Ash Vardanian
 *  @date July 9, 2026
 *  @brief Machine probe printing the tiers the running CPU supports, as one comma-separated list.
 *
 *  Unlike the sibling `<arch>_<tier>.c` files, which are try-compiled to learn what the toolchain
 *  can emit, this program is try-run by the build systems, through CMake @c try_run and Cargo
 *  @c build.rs, to learn what the build machine can execute. Static and comptime dispatch then
 *  enable the intersection of the two sets.
 *
 *  The translation unit is serial-only: every `STRINGZILLA_TARGET_*` is off, so no SIMD kernel or
 *  intrinsics header is pulled in and it compiles at baseline flags everywhere. The GPU tiers are
 *  off for a second reason: this program answers for the CPU that will run the build, and a GPU
 *  attached to the build machine says nothing about the one the binary will meet. Yet the runtime
 *  detectors still report the full hardware capability set - detection is independent of the
 *  compiled tiers by design: cpuid/xgetbv on x86, sysctl on Apple, @c mrs with a SIGILL guard on
 *  Linux Arm, and auxiliary-vector HWCAPs on RISC-V, LoongArch, and POWER. Token names come from
 *  the library's own capability map, so build systems parse tokens they know and ignore the rest.
 *
 *  On platforms where the header performs no real hardware introspection, such as WebAssembly and
 *  the OS-less targets @c STRINGZILLA_HAS_RUNTIME_DETECTION_ excludes, the program exits non-zero
 *  instead of printing a misleading "serial", and build systems treat that like any other probe
 *  failure: no answer, fall back to the target description.
 */
#define STRINGZILLA_RUNTIME_DISPATCH 0
#define STRINGZILLA_WITH_LIBC 1

#define STRINGZILLA_TARGET_WESTMERE 0
#define STRINGZILLA_TARGET_GOLDMONT 0
#define STRINGZILLA_TARGET_HASWELL 0
#define STRINGZILLA_TARGET_SKYLAKE 0
#define STRINGZILLA_TARGET_ICELAKE 0
#define STRINGZILLA_TARGET_NEON 0
#define STRINGZILLA_TARGET_NEONAES 0
#define STRINGZILLA_TARGET_NEONSHA 0
#define STRINGZILLA_TARGET_SVE 0
#define STRINGZILLA_TARGET_SVE2 0
#define STRINGZILLA_TARGET_SVE2AES 0
#define STRINGZILLA_TARGET_V128 0
#define STRINGZILLA_TARGET_V128RELAXED 0
#define STRINGZILLA_TARGET_RVV 0
#define STRINGZILLA_TARGET_RVVCRYPTO 0
#define STRINGZILLA_TARGET_LASX 0
#define STRINGZILLA_TARGET_POWERVSX 0
#define STRINGZILLA_TARGET_CUDA 0
#define STRINGZILLA_TARGET_KEPLER 0
#define STRINGZILLA_TARGET_HOPPER 0
#define STRINGZILLA_TARGET_ROCM 0

#include <stringzilla/stringzilla.h>

#include <stdio.h>

int main(void) {
#if STRINGZILLA_HAS_RUNTIME_DETECTION_
    sz_capability_t caps = sz_capabilities_runtime_cpu_();
    char names[256];
    sz_capabilities_to_string_implementation_(caps, names, sizeof(names));
    printf("%s\n", names);
    return 0;
#else
    return 1; // No hardware introspection here - the build must trust the target description.
#endif
}
