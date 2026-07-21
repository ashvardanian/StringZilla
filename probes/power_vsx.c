/* StringZilla ISA probe: IBM POWER9 VSX, mirroring `include/stringzilla/find/powervsx.h` */
#include <altivec.h>

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("power9-vector"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("power9-vector")
#endif

static __vector unsigned long long sz_probe_bit_gather_(__vector unsigned char cmp, __vector unsigned char idx) {
    return vec_vbpermq(cmp, idx);
}

int main(void) {
    __vector unsigned char const indices = {120, 112, 104, 96, 88, 80, 72, 64, 56, 48, 40, 32, 24, 16, 8, 0};
    __vector unsigned long long gathered = sz_probe_bit_gather_(vec_splats((unsigned char)0xFF), indices);
    // `vec_vbpermq` deposits the bits into element [1] on little-endian and [0] on big-endian - check both.
    return (gathered[0] | gathered[1]) != 0 ? 0 : 1;
}

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
