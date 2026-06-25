/**
 *  @brief WebAssembly relaxed-SIMD backend for UTF-8 codepoint mechanics (level above SIMD128).
 *  @file include/stringzilla/utf8_runes/v128relaxed.h
 *  @author Ash Vardanian
 */
#ifndef STRINGZILLA_UTF8_RUNES_V128RELAXED_H_
#define STRINGZILLA_UTF8_RUNES_V128RELAXED_H_

#include "stringzilla/types.h"
#include "stringzilla/utf8_runes/serial.h"
#include "stringzilla/utf8_runes/v128.h" // baseline SIMD128 fallbacks

#ifdef __cplusplus
extern "C" {
#endif

#if SZ_USE_V128RELAXED
#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("relaxed-simd"))), apply_to = function)
#endif

/*  Relaxed-SIMD offers no win for the count / find-nth kernels (they use `wasm_i8x16_eq`, range compares,
 *  compile-time-constant `wasm_i8x16_shuffle` rotations, and `wasm_i8x16_bitmask` - none of which map onto a
 *  relaxed op), so they delegate to the baseline SIMD128. The multistep newline/whitespace iterators are not
 *  defined here at all: the dispatch table routes the `v128relaxed` capability straight to the `v128` kernels. */

SZ_PUBLIC sz_size_t sz_utf8_count_v128relaxed(sz_cptr_t text, sz_size_t length) {
    return sz_utf8_count_v128(text, length);
}

SZ_PUBLIC sz_cptr_t sz_utf8_find_nth_v128relaxed(sz_cptr_t text, sz_size_t length, sz_size_t n) {
    return sz_utf8_find_nth_v128(text, length, n);
}

#if defined(__clang__)
#pragma clang attribute pop
#endif
#endif // SZ_USE_V128RELAXED

#ifdef __cplusplus
}
#endif

#endif // STRINGZILLA_UTF8_RUNES_V128RELAXED_H_
