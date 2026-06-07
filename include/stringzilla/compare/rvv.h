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

/*  Length-agnostic equality. We walk the buffers with a `vsetvl`-driven loop, so the same code
 *  is correct for any `VLEN`. On each iteration we compare a strip of bytes, OR-reduce the
 *  "not-equal" mask, and bail out the moment any lane differs. No scalar tail is needed. */
SZ_PUBLIC sz_bool_t sz_equal_rvv(sz_cptr_t a, sz_cptr_t b, sz_size_t length) {
    sz_u8_t const *a_u8 = (sz_u8_t const *)a;
    sz_u8_t const *b_u8 = (sz_u8_t const *)b;
    while (length) {
        size_t vl = __riscv_vsetvl_e8m8(length);
        vuint8m8_t a_vec = __riscv_vle8_v_u8m8(a_u8, vl);
        vuint8m8_t b_vec = __riscv_vle8_v_u8m8(b_u8, vl);
        vbool1_t ne_mask = __riscv_vmsne_vv_u8m8_b1(a_vec, b_vec, vl);
        // `vfirst` returns the index of the first set bit, or -1 if none is set.
        if (__riscv_vfirst_m_b1(ne_mask, vl) >= 0) return sz_false_k;
        a_u8 += vl, b_u8 += vl, length -= vl;
    }
    return sz_true_k;
}

/*  Lexicographic ordering. As noted in the NEON backend, see the "Operations Not Worth Optimizing"
 *  section of the Contributions Guide — branch-heavy mismatch handling makes a vector pass rarely
 *  worthwhile, so we delegate to the serial baseline. */
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
