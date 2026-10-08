/**
 *  @file include/stringzilla/compare/v128relaxed.h
 *  @author Ash Vardanian
 *  @date June 7, 2026
 *  @brief WebAssembly relaxed-SIMD backend for compare (level above SIMD128).
 *
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

#if STRINGZILLA_ARCH_WASM_
#if STRINGZILLA_TARGET_V128RELAXED
#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("relaxed-simd"))), apply_to = function)
#endif

STRINGZILLA_INLINE sz_ordering_t sz_order_v128relaxed_(sz_cptr_t a, sz_size_t a_length, sz_cptr_t b,
                                                       sz_size_t b_length) {
    return sz_order_v128_(a, a_length, b, b_length);
}

STRINGZILLA_API sz_status_t sz_order_v128relaxed(sz_cptr_t a, sz_size_t a_length, sz_cptr_t b, sz_size_t b_length,
                                                 sz_ordering_t *ordering, sz_stream_t stream) {
    sz_assert_(stream == STRINGZILLA_NULL);
    *ordering = sz_order_v128relaxed_(a, a_length, b, b_length);
    return sz_success_k;
}

STRINGZILLA_INLINE sz_bool_t sz_equal_v128relaxed_(sz_cptr_t a, sz_cptr_t b, sz_size_t length) {
    return sz_equal_v128_(a, b, length);
}

STRINGZILLA_API sz_status_t sz_equal_v128relaxed(sz_cptr_t a, sz_cptr_t b, sz_size_t length, sz_bool_t *equal,
                                                 sz_stream_t stream) {
    sz_assert_(stream == STRINGZILLA_NULL);
    *equal = sz_equal_v128relaxed_(a, b, length);
    return sz_success_k;
}

#if defined(__clang__)
#pragma clang attribute pop
#endif
#endif // STRINGZILLA_TARGET_V128RELAXED
#endif // STRINGZILLA_ARCH_WASM_

#ifdef __cplusplus
}
#endif

#endif // STRINGZILLA_COMPARE_V128RELAXED_H_
