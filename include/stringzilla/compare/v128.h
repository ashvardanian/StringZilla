/**
 *  @brief WebAssembly SIMD128 backend for compare.
 *  @file include/stringzilla/compare/v128.h
 *  @author Ash Vardanian
 *  @sa include/stringzilla/compare.h
 */
#ifndef STRINGZILLA_COMPARE_V128_H_
#define STRINGZILLA_COMPARE_V128_H_

#include "stringzilla/types.h"
#include "stringzilla/compare/serial.h"

#ifdef __cplusplus
extern "C" {
#endif

#if SZ_USE_V128
#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("simd128"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("simd128")
#endif

SZ_PUBLIC sz_ordering_t sz_order_v128(sz_cptr_t a, sz_size_t a_length, sz_cptr_t b, sz_size_t b_length) {
    //! Lexicographic ordering needs the byte-reversed comparison of the first differing word,
    //! which has no cheap SIMD128 equivalent that beats the serial SWAR variant. Delegate.
    return sz_order_serial(a, a_length, b, b_length);
}

SZ_PUBLIC sz_bool_t sz_equal_v128(sz_cptr_t a, sz_cptr_t b, sz_size_t length) {
    if (length < 16) return sz_equal_serial(a, b, length);

    sz_u128_vec_t a_vec, b_vec;
    sz_size_t offset = 0;
    do {
        a_vec.v128 = wasm_v128_load(a + offset);
        b_vec.v128 = wasm_v128_load(b + offset);
        // `wasm_i8x16_eq` yields 0xFF where bytes match; `wasm_i8x16_all_true` of the equality
        // mask is true only if every lane matched.
        if (!wasm_i8x16_all_true(wasm_i8x16_eq(a_vec.v128, b_vec.v128))) return sz_false_k;
        offset += 16;
    } while (offset + 16 <= length);

    // Final overlapping check on the last 16-byte window.
    a_vec.v128 = wasm_v128_load(a + length - 16);
    b_vec.v128 = wasm_v128_load(b + length - 16);
    if (!wasm_i8x16_all_true(wasm_i8x16_eq(a_vec.v128, b_vec.v128))) return sz_false_k;
    return sz_true_k;
}

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#endif // SZ_USE_V128

#ifdef __cplusplus
}
#endif

#endif // STRINGZILLA_COMPARE_V128_H_
