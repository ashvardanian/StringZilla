/* StringZilla ISA probe: NEON (AArch64 baseline SIMD), mirroring `include/stringzilla/find/neon.h` */
#include <arm_neon.h>

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("+simd"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("+simd")
#endif

static uint8x16_t sz_probe_lookup_(uint8x16_t table, uint8x16_t indices) { return vqtbl1q_u8(table, indices); }

int main(void) {
    uint8x16_t gathered = sz_probe_lookup_(vdupq_n_u8(1), vdupq_n_u8(2));
    return vaddvq_u8(gathered) != 0 ? 0 : 1;
}

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
