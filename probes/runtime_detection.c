/* StringZilla platform probe: COMPILES only when `<stringzilla/stringzilla.h>` performs real runtime
 * capability detection for this target (`SZ_CAPABILITIES_RUNTIME_DETECTABLE_`).
 *
 * The sibling `run_capabilities.c` answers "what does this machine support" by executing - which a cross
 * build can never do. This probe answers the prior question, "will the built library mask unsupported
 * tiers at load?", with a plain try-compile that works for any target. Runtime-dispatched builds use it
 * to pick their gate: where detection exists, the load-time dispatch table masks whatever the CPU lacks,
 * so every compilable tier is safe to enable; where it does not (WebAssembly by nature - a module with
 * unsupported SIMD opcodes fails validation at instantiation - or OS-less exotic targets), the dispatch
 * table just mirrors the compile-time mask and the build must stay within the target description.
 *
 * The answer lives in the header, next to the detectors themselves, so neither CMake nor `build.rs`
 * carries a platform list that could drift out of sync. Compile with the same `SZ_AVOID_LIBC` value as
 * the real build - detectability depends on it where the probe reads the auxiliary vector.
 */
#define SZ_DYNAMIC_DISPATCH 0

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

#if !SZ_CAPABILITIES_RUNTIME_DETECTABLE_
#error "No runtime capability detection on this target - dispatch tables would mirror the compile-time mask"
#endif

int main(void) { return 0; }
