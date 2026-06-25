/**
 *  @brief RISC-V Vector (RVV 1.0) backend for compare.
 *  @file include/stringzilla/compare/rvv.h
 *  @author Ash Vardanian
 *  @sa include/stringzilla/compare.h
 */
#ifndef STRINGZILLA_COMPARE_RVV_H_
#define STRINGZILLA_COMPARE_RVV_H_

#include "stringzilla/types.h"
#include "stringzilla/compare/serial.h"

#ifdef __cplusplus
extern "C" {
#endif

#if SZ_USE_RVV
#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("arch=+v"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("arch=+v")
#endif

SZ_PUBLIC sz_bool_t sz_equal_rvv(sz_cptr_t a, sz_cptr_t b, sz_size_t length) {
    sz_u8_t const *a_u8 = (sz_u8_t const *)a;
    sz_u8_t const *b_u8 = (sz_u8_t const *)b;
    while (length) {
        sz_size_t vector_length = __riscv_vsetvl_e8m8(length);
        vuint8m8_t a_u8m8 = __riscv_vle8_v_u8m8(a_u8, vector_length);
        vuint8m8_t b_u8m8 = __riscv_vle8_v_u8m8(b_u8, vector_length);
        vbool1_t not_equal_mask_b1 = __riscv_vmsne_vv_u8m8_b1(a_u8m8, b_u8m8, vector_length);
        // `vfirst` returns the index of the first set bit, or -1 if none is set.
        if (__riscv_vfirst_m_b1(not_equal_mask_b1, vector_length) >= 0) return sz_false_k;
        a_u8 += vector_length, b_u8 += vector_length, length -= vector_length;
    }
    return sz_true_k;
}

SZ_PUBLIC sz_ordering_t sz_order_rvv(sz_cptr_t a, sz_size_t a_length, sz_cptr_t b, sz_size_t b_length) {
    sz_u8_t const *a_u8 = (sz_u8_t const *)a;
    sz_u8_t const *b_u8 = (sz_u8_t const *)b;
    // Scan the common prefix in `e8m8` strips; `vmsne` + `vfirst` locate the first differing byte natively, and
    // `vsetvl` folds the tail with no scalar remainder (mirrors `sz_equal_rvv` and the Skylake order).
    sz_size_t remaining = a_length < b_length ? a_length : b_length;
    while (remaining) {
        sz_size_t vector_length = __riscv_vsetvl_e8m8(remaining);
        vuint8m8_t a_u8m8 = __riscv_vle8_v_u8m8(a_u8, vector_length);
        vuint8m8_t b_u8m8 = __riscv_vle8_v_u8m8(b_u8, vector_length);
        vbool1_t not_equal_mask_b1 = __riscv_vmsne_vv_u8m8_b1(a_u8m8, b_u8m8, vector_length);
        long const first_difference = __riscv_vfirst_m_b1(not_equal_mask_b1, vector_length);
        // Order the two differing bytes, read back from the just-loaded (hot) cache line, as the Skylake order does.
        if (first_difference >= 0) return sz_order_scalars_(a_u8[first_difference], b_u8[first_difference]);
        a_u8 += vector_length, b_u8 += vector_length, remaining -= vector_length;
    }
    // Common prefix equal: the shorter string is the smaller one.
    return sz_order_scalars_(a_length, b_length);
}

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#endif // SZ_USE_RVV

#ifdef __cplusplus
}
#endif

#endif // STRINGZILLA_COMPARE_RVV_H_
