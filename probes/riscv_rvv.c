/* StringZilla ISA probe: RISC-V Vector 1.0 (RVV), mirroring `include/stringzilla/find/rvv.h` */
#include <riscv_vector.h>

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("arch=+v"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("arch=+v")
#endif

static vuint8m1_t sz_probe_mask_(vuint8m1_t bytes, unsigned long count) {
    return __riscv_vand_vx_u8m1(bytes, 0x0Fu, count);
}

int main(void) {
    unsigned long count = __riscv_vsetvlmax_e8m1();
    vuint8m1_t masked = sz_probe_mask_(__riscv_vmv_v_x_u8m1(0xFFu, count), count);
    return __riscv_vmv_x_s_u8m1_u8(masked) != 0 ? 0 : 1;
}

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
