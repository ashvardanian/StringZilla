/* StringZilla ISA probe: NEON + AES (Armv8-A crypto), mirroring `include/stringzilla/hash/neonaes.h` */
#include <arm_neon.h>

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("+simd+crypto+aes"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("+simd+crypto+aes")
#endif

static uint8x16_t sz_probe_aes_round_(uint8x16_t state, uint8x16_t key) { return vaesmcq_u8(vaeseq_u8(state, key)); }

int main(void) {
    uint8x16_t mixed = sz_probe_aes_round_(vdupq_n_u8(1), vdupq_n_u8(2));
    return vaddvq_u8(mixed) != 0 ? 0 : 1;
}

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
