/* StringZilla machine probe: prints the tiers the RUNNING CPU supports, one comma-separated list.
 *
 * Unlike the sibling `<arch>_<tier>.c` files - which are try-COMPILED to learn what the toolchain can
 * emit - this program is try-RUN by the build systems (CMake `try_run`, Cargo `build.rs`) to learn what
 * the build machine can execute. Static/comptime dispatch then enables the intersection of the two sets.
 *
 * The translation unit is serial-only (every `SZ_USE_*` is off, so no SIMD kernel or intrinsics header
 * is pulled in and it compiles at baseline flags everywhere), yet the runtime detectors still report the
 * FULL hardware capability set - detection is independent of the compiled tiers by design: cpuid/xgetbv
 * on x86, sysctl on Apple, `mrs` with a SIGILL guard on Linux Arm, auxiliary-vector HWCAPs on RISC-V,
 * LoongArch, and POWER. Token names come from the library's own capability map, so build systems parse
 * tokens they know and ignore the rest. On platforms where the header performs no real hardware
 * introspection (WebAssembly, OS-less targets - see `SZ_CAPABILITIES_RUNTIME_DETECTABLE_`) the program
 * exits non-zero instead of printing a misleading "serial", and build systems treat that like any other
 * probe failure: no answer, fall back to the target description.
 */
#define SZ_DYNAMIC_DISPATCH 0
#define SZ_AVOID_LIBC 0

#define SZ_USE_WESTMERE 0
#define SZ_USE_GOLDMONT 0
#define SZ_USE_HASWELL 0
#define SZ_USE_SKYLAKE 0
#define SZ_USE_ICELAKE 0
#define SZ_USE_NEON 0
#define SZ_USE_NEONAES 0
#define SZ_USE_NEONSHA 0
#define SZ_USE_SVE 0
#define SZ_USE_SVE2 0
#define SZ_USE_SVE2AES 0
#define SZ_USE_V128 0
#define SZ_USE_V128RELAXED 0
#define SZ_USE_RVV 0
#define SZ_USE_RVVCRYPTO 0
#define SZ_USE_LASX 0
#define SZ_USE_POWERVSX 0

#include <stringzilla/stringzilla.h>

#include <stdio.h>

int main(void) {
#if SZ_CAPABILITIES_RUNTIME_DETECTABLE_
    sz_capability_t caps = sz_capabilities_runtime_implementation_();
    printf("%s\n", sz_capabilities_to_string_implementation_(caps));
    return 0;
#else
    return 1; // No hardware introspection here - the build must trust the target description.
#endif
}
