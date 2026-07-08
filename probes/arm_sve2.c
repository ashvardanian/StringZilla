/* StringZilla ISA probe: SVE2 (Armv9-A scalable vectors), mirroring `include/stringzilla/hash/sve2.h` */
#if defined(_MSC_VER)
#error "SVE target attributes and intrinsics are unavailable under MSVC"
#endif
#if defined(__APPLE__)
#error "No Apple CPU implements SVE - kernels could compile but never run"
#endif
#include <arm_sve.h>

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("+sve+sve2"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("+sve+sve2")
#endif

static svbool_t sz_probe_match_(svuint8_t haystack, svuint8_t needles) {
    return svmatch_u8(svptrue_b8(), haystack, needles);
}

int main(void) {
    svbool_t hits = sz_probe_match_(svdup_n_u8(1), svdup_n_u8(1));
    return svptest_any(svptrue_b8(), hits) ? 0 : 1;
}

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
