/**
 *  @file probes/runtime_detection.c
 *  @author Ash Vardanian
 *  @date July 9, 2026
 *  @brief Platform probe that compiles only when `<stringzilla/stringzilla.h>` performs real
 *      runtime capability detection for this target, per @c STRINGZILLA_HAS_RUNTIME_DETECTION_.
 *
 *  The sibling `run_capabilities.c` answers "what does this machine support" by executing, which a
 *  cross build can never do. This probe answers the prior question, "will the built library mask
 *  unsupported tiers at load?", with a plain try-compile that works for any target.
 *  Runtime-dispatched builds use it to pick their gate: where detection exists, the load-time
 *  dispatch table masks whatever the CPU lacks, so every compilable tier is safe to enable; where
 *  it does not, as in WebAssembly by nature - a module with unsupported SIMD opcodes fails
 *  validation at instantiation - or OS-less exotic targets, the dispatch table just mirrors the
 *  compile-time mask and the build must stay within the target description.
 *
 *  The answer lives in the header, next to the detectors themselves, so neither CMake nor
 *  @c build.rs carries a platform list that could drift out of sync. Compile with the same
 *  @c STRINGZILLA_WITH_LIBC value as the real build - detectability depends on it where the probe
 *  reads the auxiliary vector.
 */
#define STRINGZILLA_RUNTIME_DISPATCH 0

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

#if !STRINGZILLA_HAS_RUNTIME_DETECTION_
#error "No runtime capability detection on this target - dispatch tables would mirror the compile-time mask"
#endif

int main(void) { return 0; }
