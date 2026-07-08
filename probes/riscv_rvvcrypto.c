/* StringZilla ISA probe: RISC-V Vector Crypto (Zvkned AES + Zvknhb SHA-2), mirroring
 * `include/stringzilla/hash/rvvcrypto.h` */
#include <riscv_vector.h>

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("arch=+v,+zvkned,+zvknhb"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("arch=+v,+zvkned,+zvknhb")
#endif

static vuint32m1_t sz_probe_aes_round_(vuint32m1_t state, vuint32m1_t key, unsigned long count) {
    return __riscv_vaesem_vv_u32m1(state, key, count);
}
static vuint32m1_t sz_probe_sha_rounds_(vuint32m1_t hash, vuint32m1_t a, vuint32m1_t b, unsigned long count) {
    return __riscv_vsha2ch_vv_u32m1(hash, a, b, count);
}

int main(void) {
    unsigned long count = __riscv_vsetvlmax_e32m1();
    vuint32m1_t mixed = sz_probe_aes_round_(__riscv_vmv_v_x_u32m1(1u, count), __riscv_vmv_v_x_u32m1(2u, count), count);
    vuint32m1_t state = sz_probe_sha_rounds_(mixed, mixed, mixed, count);
    return __riscv_vmv_x_s_u32m1_u32(state) != 0 ? 0 : 1;
}

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
