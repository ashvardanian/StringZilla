/* StringZilla ISA probe: SVE2 + AES (Armv9-A crypto), mirroring `include/stringzilla/hash/sve2aes.h` */
#if defined(_MSC_VER)
#error "SVE target attributes and intrinsics are unavailable under MSVC"
#endif
#if defined(__APPLE__)
#error "No Apple CPU implements SVE - kernels could compile but never run"
#endif
#include <arm_sve.h>

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("+sve+sve2+sve2-aes"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("+sve+sve2+sve2-aes")
#endif

static svuint8_t sz_probe_aes_round_(svuint8_t state, svuint8_t key) { return svaesmc_u8(svaese_u8(state, key)); }

int main(void) {
    svuint8_t mixed = sz_probe_aes_round_(svdup_n_u8(1), svdup_n_u8(2));
    return svlasta_u8(svpfalse_b(), mixed) != 0 ? 0 : 1;
}

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
