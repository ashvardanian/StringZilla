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
        vbool1_t ne_mask_b1 = __riscv_vmsne_vv_u8m8_b1(a_u8m8, b_u8m8, vector_length);
        // `vfirst` returns the index of the first set bit, or -1 if none is set.
        if (__riscv_vfirst_m_b1(ne_mask_b1, vector_length) >= 0) return sz_false_k;
        a_u8 += vector_length, b_u8 += vector_length, length -= vector_length;
    }
    return sz_true_k;
}

SZ_PUBLIC sz_ordering_t sz_order_rvv(sz_cptr_t a, sz_size_t a_length, sz_cptr_t b, sz_size_t b_length) {
    return sz_order_serial(a, a_length, b, b_length);
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
