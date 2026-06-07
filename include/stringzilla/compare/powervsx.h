/**
 *  @brief IBM Power VSX backend for compare.
 *  @file include/stringzilla/compare/powervsx.h
 *  @author Ash Vardanian
 *  @sa include/stringzilla/compare.h
 */
#ifndef STRINGZILLA_COMPARE_POWERVSX_H_
#define STRINGZILLA_COMPARE_POWERVSX_H_

#include "stringzilla/types.h"
#include "stringzilla/compare/serial.h"

#ifdef __cplusplus
extern "C" {
#endif

#if SZ_USE_POWERVSX
#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("power9-vector"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("power9-vector")
#endif

/**
 *  @brief Byte-level equality comparison using IBM Power VSX, processing 16 bytes at a time.
 *         Mirrors the NEON reference; the final 16-byte window handles the tail without a
 *         separate scalar loop, exactly matching `sz_equal_serial`.
 */
SZ_PUBLIC sz_bool_t sz_equal_powervsx(sz_cptr_t a, sz_cptr_t b, sz_size_t length) {
    if (length < 16) return sz_equal_serial(a, b, length);

    __vector unsigned char a_vec, b_vec;
    sz_size_t offset = 0;
    do {
        a_vec = vec_xl(0, (unsigned char const *)(a + offset));
        b_vec = vec_xl(0, (unsigned char const *)(b + offset));
        // On Power10 `vec_first_mismatch_index` reports the first differing lane (16 if none) in a
        // single instruction, replacing the `vec_all_eq` compare + branch reduction. The Power9
        // fallback below keeps the byte-for-byte identical `vec_all_eq` semantics.
#if defined(_ARCH_PWR10)
        if (vec_first_mismatch_index(a_vec, b_vec) != 16) return sz_false_k;
#else
        if (!vec_all_eq(a_vec, b_vec)) return sz_false_k; // True only when every byte matched.
#endif
        offset += 16;
    } while (offset + 16 <= length);

    // Final check - load the last register-long window from the end.
    a_vec = vec_xl(0, (unsigned char const *)(a + length - 16));
    b_vec = vec_xl(0, (unsigned char const *)(b + length - 16));
#if defined(_ARCH_PWR10)
    if (vec_first_mismatch_index(a_vec, b_vec) != 16) return sz_false_k;
#else
    if (!vec_all_eq(a_vec, b_vec)) return sz_false_k;
#endif
    return sz_true_k;
}

SZ_PUBLIC sz_ordering_t sz_order_powervsx(sz_cptr_t a, sz_size_t a_length, sz_cptr_t b, sz_size_t b_length) {
    //! Lexicographic ordering is endian-sensitive and not worth vectorizing — see the
    //! "Operations Not Worth Optimizing" note in the Contributions Guide, mirroring NEON.
    return sz_order_serial(a, a_length, b, b_length);
}

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#endif // SZ_USE_POWERVSX

#ifdef __cplusplus
}
#endif

#endif // STRINGZILLA_COMPARE_POWERVSX_H_
