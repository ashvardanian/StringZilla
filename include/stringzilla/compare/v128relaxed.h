/**
 *  @brief WebAssembly relaxed-SIMD backend for compare (level above SIMD128).
 *  @file include/stringzilla/compare/v128relaxed.h
 *  @author Ash Vardanian
 *  @sa include/stringzilla/compare.h
 */
#ifndef STRINGZILLA_COMPARE_V128RELAXED_H_
#define STRINGZILLA_COMPARE_V128RELAXED_H_

#include "stringzilla/types.h"
#include "stringzilla/compare/serial.h"
#include "stringzilla/compare/v128.h" // baseline SIMD128 fallbacks

#ifdef __cplusplus
extern "C" {
#endif

#if SZ_USE_V128RELAXED
#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("relaxed-simd"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("relaxed-simd")
#endif

/*  Equality is `wasm_i8x16_eq` + `all_true`, and ordering needs a byte-reversed compare of the first
 *  differing word; neither gains anything from relaxed-simd. Both delegate to the baseline. */

SZ_PUBLIC sz_ordering_t sz_order_v128relaxed(sz_cptr_t a, sz_size_t a_length, sz_cptr_t b, sz_size_t b_length) {
    return sz_order_v128(a, a_length, b, b_length);
}

SZ_PUBLIC sz_bool_t sz_equal_v128relaxed(sz_cptr_t a, sz_cptr_t b, sz_size_t length) {
    return sz_equal_v128(a, b, length);
}

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#endif // SZ_USE_V128RELAXED

#ifdef __cplusplus
}
#endif

#endif // STRINGZILLA_COMPARE_V128RELAXED_H_
