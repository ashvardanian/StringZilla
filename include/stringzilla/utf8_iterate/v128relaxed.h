/**
 *  @brief WebAssembly relaxed-SIMD backend for utf8 (level above SIMD128).
 *  @file include/stringzilla/utf8_iterate/v128relaxed.h
 *  @author Ash Vardanian
 *  @sa utf8.h
 */
#ifndef STRINGZILLA_UTF8_ITERATE_V128RELAXED_H_
#define STRINGZILLA_UTF8_ITERATE_V128RELAXED_H_

#include "stringzilla/types.h"
#include "stringzilla/utf8_iterate/serial.h"
#include "stringzilla/utf8_iterate/v128.h" // baseline SIMD128 fallbacks

#ifdef __cplusplus
extern "C" {
#endif

#if SZ_USE_V128RELAXED
#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("relaxed-simd"))), apply_to = function)
#endif

/*  The UTF-8 kernels are built from `wasm_i8x16_eq`, range compares, `wasm_i8x16_shuffle` rotations,
 *  and `wasm_i8x16_bitmask`. None of these map onto a relaxed-simd op that would be a win (the
 *  rotations use compile-time-constant shuffles, not data-dependent swizzles), so every kernel
 *  delegates to the baseline SIMD128 implementation. */

SZ_PUBLIC sz_size_t sz_utf8_count_v128relaxed(sz_cptr_t text, sz_size_t length) {
    return sz_utf8_count_v128(text, length);
}

SZ_PUBLIC sz_cptr_t sz_utf8_find_nth_v128relaxed(sz_cptr_t text, sz_size_t length, sz_size_t n) {
    return sz_utf8_find_nth_v128(text, length, n);
}

SZ_PUBLIC sz_cptr_t sz_utf8_find_newline_v128relaxed(sz_cptr_t text, sz_size_t length, sz_size_t *matched_length) {
    return sz_utf8_find_newline_v128(text, length, matched_length);
}

SZ_PUBLIC sz_cptr_t sz_utf8_find_whitespace_v128relaxed(sz_cptr_t text, sz_size_t length, sz_size_t *matched_length) {
    return sz_utf8_find_whitespace_v128(text, length, matched_length);
}

#if defined(__clang__)
#pragma clang attribute pop
#endif
#endif // SZ_USE_V128RELAXED

#ifdef __cplusplus
}
#endif

#endif // STRINGZILLA_UTF8_ITERATE_V128RELAXED_H_
