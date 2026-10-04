/**
 *  @file include/stringzilla/utf8_runes/v128relaxed.h
 *  @author Ash Vardanian
 *  @date June 7, 2026
 *  @brief WebAssembly relaxed-SIMD backend for UTF-8 codepoint mechanics (level above SIMD128).
 */
#ifndef STRINGZILLA_UTF8_RUNES_V128RELAXED_H_
#define STRINGZILLA_UTF8_RUNES_V128RELAXED_H_

#include "stringzilla/types.h"
#include "stringzilla/utf8_runes/serial.h"
#include "stringzilla/utf8_runes/v128.h" // baseline SIMD128 fallbacks

#ifdef __cplusplus
extern "C" {
#endif

/*  Relaxed-SIMD offers no win for the count / find-nth kernels (they use @c wasm_i8x16_eq, range
 *  compares, compile-time-constant @c wasm_i8x16_shuffle rotations, and @c wasm_i8x16_bitmask -
 *  none of which map onto a relaxed op), so they delegate to the baseline SIMD128. The decoder and
 *  the multistep newline/whitespace iterators are not defined here at all: a @c v128relaxed mask
 *  also carries @c v128, so their lists pick the @c v128 kernels. */
#if STRINGZILLA_TARGET_V128RELAXED
#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("relaxed-simd"))), apply_to = function)
#endif

STRINGZILLA_API sz_status_t sz_utf8_count_v128relaxed(sz_cptr_t text, sz_size_t length, sz_size_t *count,
                                                      sz_stream_t stream) {
    sz_assert_(stream == STRINGZILLA_NULL);
    *count = sz_utf8_count_v128_(text, length);
    return sz_success_k;
}

STRINGZILLA_API sz_status_t sz_utf8_seek_v128relaxed(sz_cptr_t text, sz_size_t length, sz_size_t n, sz_cptr_t *position,
                                                     sz_stream_t stream) {
    sz_assert_(stream == STRINGZILLA_NULL);
    *position = sz_utf8_seek_v128_(text, length, n);
    return sz_success_k;
}

#if defined(__clang__)
#pragma clang attribute pop
#endif
#endif // STRINGZILLA_TARGET_V128RELAXED

#ifdef __cplusplus
}
#endif

#endif // STRINGZILLA_UTF8_RUNES_V128RELAXED_H_
