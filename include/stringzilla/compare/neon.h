/**
 *  @file include/stringzilla/compare/neon.h
 *  @author Ash Vardanian
 *  @date October 12, 2024
 *  @brief NEON backend for string comparison utilities.
 *
 *  @sa c/stringzilla/compare.h
 */
#ifndef STRINGZILLA_COMPARE_NEON_H_
#define STRINGZILLA_COMPARE_NEON_H_

#include "stringzilla/types.h"
#include "stringzilla/compare/serial.h"

#ifdef __cplusplus
extern "C" {
#endif

#if STRINGZILLA_ARCH_ARM64_
#if STRINGZILLA_ARCH_ARM64_NEON_
#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("+simd"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("+simd")
#endif

/** The @p length bytes, at most a lane, of @p text, zeros past them, from overlapping reads inside
 *  it, as a byte loop would stall on store forwarding. */
STRINGZILLA_INLINE uint8x16_t sz_load_short_neon_(sz_cptr_t text, sz_size_t length) {
    sz_u64_t low = 0, high = 0;
    if (length >= 8) {
        // Two half shifts, as a span of eight bytes shifts the whole word out.
        sz_size_t const passed = 16 - length;
        low = sz_u64_load(text).u64;
        high = (sz_u64_load(text + length - 8).u64 >> (passed * 4)) >> (passed * 4);
    }
    else if (length >= 4) {
        low = (sz_u64_t)sz_u32_load(text).u32 | ((sz_u64_t)sz_u32_load(text + length - 4).u32 << ((length - 4) * 8));
    }
    else if (length) {
        sz_size_t const middle = length / 2;
        low = (sz_u64_t)(sz_u8_t)text[0] | ((sz_u64_t)(sz_u8_t)text[middle] << (middle * 8)) |
              ((sz_u64_t)(sz_u8_t)text[length - 1] << ((length - 1) * 8));
    }
    return vcombine_u8(vcreate_u8(low), vcreate_u8(high));
}

STRINGZILLA_INLINE sz_ordering_t sz_order_neon_(sz_cptr_t a, sz_size_t a_length, sz_cptr_t b, sz_size_t b_length) {
    //! Before optimizing this, read the "Operations Not Worth Optimizing" in Contributions Guide:
    //! https://github.com/ashvardanian/StringZilla/blob/main/CONTRIBUTING.md#general-performance-observations
    return sz_order_serial_(a, a_length, b, b_length);
}

STRINGZILLA_INLINE sz_bool_t sz_equal_neon_(sz_cptr_t a, sz_cptr_t b, sz_size_t length) {
    if (length < 16) return sz_equal_serial_(a, b, length);

    sz_u128_vec_t a_vec, b_vec;
    sz_size_t offset = 0;
    // Past 128 bytes one reduction covers 64 bytes, after one register for what differs early.
    if (length >= 128) {
        a_vec.u8x16 = vld1q_u8((sz_u8_t const *)a);
        b_vec.u8x16 = vld1q_u8((sz_u8_t const *)b);
        if (vminvq_u8(vceqq_u8(a_vec.u8x16, b_vec.u8x16)) != 255) return sz_false_k;
        for (offset = 16; offset + 64 + 16 <= length; offset += 64) {
            uint8x16x4_t const a_u8x16x4 = vld1q_u8_x4((sz_u8_t const *)(a + offset));
            uint8x16x4_t const b_u8x16x4 = vld1q_u8_x4((sz_u8_t const *)(b + offset));
            uint8x16_t const any_u8x16 = vorrq_u8(
                vorrq_u8(veorq_u8(a_u8x16x4.val[0], b_u8x16x4.val[0]), veorq_u8(a_u8x16x4.val[1], b_u8x16x4.val[1])),
                vorrq_u8(veorq_u8(a_u8x16x4.val[2], b_u8x16x4.val[2]), veorq_u8(a_u8x16x4.val[3], b_u8x16x4.val[3])));
            if (vmaxvq_u8(any_u8x16)) return sz_false_k;
        }
    }
    do {
        a_vec.u8x16 = vld1q_u8((sz_u8_t const *)(a + offset));
        b_vec.u8x16 = vld1q_u8((sz_u8_t const *)(b + offset));
        uint8x16_t matches_u8x16 = vceqq_u8(a_vec.u8x16, b_vec.u8x16);
        if (vminvq_u8(matches_u8x16) != 255) return sz_false_k; // Check if all bytes match
        offset += 16;
    } while (offset + 16 <= length);

    // For final check - load the last register-long piece of content from the end
    a_vec.u8x16 = vld1q_u8((sz_u8_t const *)(a + length - 16));
    b_vec.u8x16 = vld1q_u8((sz_u8_t const *)(b + length - 16));
    uint8x16_t matches_u8x16 = vceqq_u8(a_vec.u8x16, b_vec.u8x16);
    if (vminvq_u8(matches_u8x16) != 255) return sz_false_k;
    return sz_true_k;
}

#if STRINGZILLA_TARGET_NEON

STRINGZILLA_API sz_status_t sz_order_neon(sz_cptr_t a, sz_size_t a_length, sz_cptr_t b, sz_size_t b_length,
                                          sz_ordering_t *ordering, sz_stream_t stream) {
    sz_assert_(stream == STRINGZILLA_NULL);
    *ordering = sz_order_neon_(a, a_length, b, b_length);
    return sz_success_k;
}

STRINGZILLA_API sz_status_t sz_equal_neon(sz_cptr_t a, sz_cptr_t b, sz_size_t length, sz_bool_t *equal,
                                          sz_stream_t stream) {
    sz_assert_(stream == STRINGZILLA_NULL);
    *equal = sz_equal_neon_(a, b, length);
    return sz_success_k;
}

#endif // STRINGZILLA_TARGET_NEON

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#endif // STRINGZILLA_ARCH_ARM64_NEON_
#endif // STRINGZILLA_ARCH_ARM64_

#ifdef __cplusplus
}
#endif

#endif // STRINGZILLA_COMPARE_NEON_H_
