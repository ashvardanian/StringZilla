/**
 *  @file include/stringzilla/compare/sve.h
 *  @author Ash Vardanian
 *  @date October 3, 2024
 *  @brief SVE backend for string comparison utilities.
 *
 *  @sa include/stringzilla/compare.h
 */
#ifndef STRINGZILLA_COMPARE_SVE_H_
#define STRINGZILLA_COMPARE_SVE_H_

#include "stringzilla/types.h"
#include "stringzilla/compare/serial.h"
#include "stringzilla/compare/neon.h" // `sz_equal_neon_`

#ifdef __cplusplus
extern "C" {
#endif

#if STRINGZILLA_ARCH_ARM64_SVE_
#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("+sve"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("+sve")
#endif

STRINGZILLA_INLINE sz_bool_t sz_equal_sve_(sz_cptr_t a, sz_cptr_t b, sz_size_t length) {
    // Determine the number of bytes in an SVE vector.
    sz_size_t const vector_bytes = svcntb();
    sz_size_t progress = 0;
    do {
        svbool_t active_b8x = svwhilelt_b8((sz_u64_t)progress, (sz_u64_t)length);
        svuint8_t a_u8x = svld1(active_b8x, (sz_u8_t const *)(a + progress));
        svuint8_t b_u8x = svld1(active_b8x, (sz_u8_t const *)(b + progress));
        // Compare: generate a predicate marking lanes where a!=b
        svbool_t not_equal_b8x = svcmpne(active_b8x, a_u8x, b_u8x);
        if (svptest_any(active_b8x, not_equal_b8x)) return sz_false_k;
        progress += vector_bytes;
    } while (progress < length);
    return sz_true_k;
}

STRINGZILLA_INLINE sz_ordering_t sz_order_sve_(sz_cptr_t a, sz_size_t a_length, sz_cptr_t b, sz_size_t b_length) {
    //! Before optimizing this, read the "Operations Not Worth Optimizing" in Contributions Guide:
    //! https://github.com/ashvardanian/StringZilla/blob/main/CONTRIBUTING.md#general-performance-observations
    return sz_order_serial_(a, a_length, b, b_length);
}

#if STRINGZILLA_TARGET_SVE

STRINGZILLA_API sz_status_t sz_equal_sve(sz_cptr_t a, sz_cptr_t b, sz_size_t length, sz_bool_t *equal, void *stream) {
    sz_assert_(stream == STRINGZILLA_NULL);
    // Graviton 5: the scalable comparison only outruns NEON on registers wider than 128 bits.
    *equal = svcntb() <= 16 ? sz_equal_neon_(a, b, length) : sz_equal_sve_(a, b, length);
    return sz_success_k;
}

STRINGZILLA_API sz_status_t sz_order_sve(sz_cptr_t a, sz_size_t a_length, sz_cptr_t b, sz_size_t b_length,
                                         sz_ordering_t *ordering, void *stream) {
    sz_assert_(stream == STRINGZILLA_NULL);
    *ordering = sz_order_sve_(a, a_length, b, b_length);
    return sz_success_k;
}

#endif // STRINGZILLA_TARGET_SVE

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#endif // STRINGZILLA_ARCH_ARM64_SVE_

#ifdef __cplusplus
}
#endif

#endif // STRINGZILLA_COMPARE_SVE_H_
