/**
 *  @brief SVE backend for string comparison utilities.
 *  @file include/stringzilla/compare/sve.h
 *  @author Ash Vardanian
 *  @sa include/stringzilla/compare.h
 */
#ifndef STRINGZILLA_COMPARE_SVE_H_
#define STRINGZILLA_COMPARE_SVE_H_

#include "stringzilla/types.h"
#include "stringzilla/compare/serial.h"

#ifdef __cplusplus
extern "C" {
#endif

#if SZ_USE_SVE
#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("+sve"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("+sve")
#endif

SZ_PUBLIC sz_bool_t sz_equal_sve(sz_cptr_t a, sz_cptr_t b, sz_size_t length) {
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

SZ_PUBLIC sz_ordering_t sz_order_sve(sz_cptr_t a, sz_size_t a_length, sz_cptr_t b, sz_size_t b_length) {
    //! Before optimizing this, read the "Operations Not Worth Optimizing" in Contributions Guide:
    //! https://github.com/ashvardanian/StringZilla/blob/main/CONTRIBUTING.md#general-performance-observations
    return sz_order_serial(a, a_length, b, b_length);
}

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#endif // SZ_USE_SVE

#ifdef __cplusplus
}
#endif

#endif // STRINGZILLA_COMPARE_SVE_H_
