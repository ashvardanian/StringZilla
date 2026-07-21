/* StringZilla ISA probe: NEON + SHA2 (Armv8-A crypto), mirroring `include/stringzilla/hash/neonsha.h` */
#include <arm_neon.h>

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("+simd+crypto+sha2"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("+simd+crypto+sha2")
#endif

static uint32x4_t sz_probe_sha_rounds_(uint32x4_t hash, uint32x4_t wk) { return vsha256hq_u32(hash, hash, wk); }

int main(void) {
    uint32x4_t state = sz_probe_sha_rounds_(vdupq_n_u32(1), vdupq_n_u32(2));
    return vaddvq_u32(state) != 0 ? 0 : 1;
}

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
