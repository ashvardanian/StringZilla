/**
 *  @file include/stringzilla/compare/powervsx.h
 *  @author Ash Vardanian
 *  @date June 7, 2026
 *  @brief IBM Power VSX backend for compare.
 *
 *  @sa c/stringzilla/compare.h
 */
#ifndef STRINGZILLA_COMPARE_POWERVSX_H_
#define STRINGZILLA_COMPARE_POWERVSX_H_

#include "stringzilla/types.h"
#include "stringzilla/compare/serial.h"

#ifdef __cplusplus
extern "C" {
#endif

#if STRINGZILLA_ARCH_PPC64_
#if STRINGZILLA_TARGET_POWERVSX
#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("power9-vector"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("power9-vector")
#endif

STRINGZILLA_INLINE sz_bool_t sz_equal_vec16_powervsx_(__vector unsigned char a_u8x16, __vector unsigned char b_u8x16) {
    return (sz_bool_t)(vec_first_mismatch_index(a_u8x16, b_u8x16) == 16);
}

STRINGZILLA_INLINE sz_bool_t sz_equal_powervsx_(sz_cptr_t a, sz_cptr_t b, sz_size_t length) {
    // `lxvl` reads exactly `length` bytes and zeroes the rest of the register.
    if (length <= 16)
        return sz_equal_vec16_powervsx_(vec_xl_len((unsigned char *)a, length), vec_xl_len((unsigned char *)b, length));
    // Two overlapping 128-bit loads cover the whole [16,32) window with one branch.
    else if (length <= 32) {
        __vector unsigned char a_first_u8x16 = vec_xl(0, (unsigned char const *)(a));
        __vector unsigned char b_first_u8x16 = vec_xl(0, (unsigned char const *)(b));
        __vector unsigned char a_second_u8x16 = vec_xl(0, (unsigned char const *)(a + length - 16));
        __vector unsigned char b_second_u8x16 = vec_xl(0, (unsigned char const *)(b + length - 16));
        __vector unsigned char eq_u8x16 = vec_and( //
            (__vector unsigned char)vec_cmpeq(a_first_u8x16, b_first_u8x16),
            (__vector unsigned char)vec_cmpeq(a_second_u8x16, b_second_u8x16));
        return (sz_bool_t)vec_all_eq(eq_u8x16, vec_splats((unsigned char)0xFF));
    }
    // Long inputs: stride two 16-byte vectors per iteration to expose more memory-level parallelism,
    // then close with one overlapping tail window read from the end.
    else {
        sz_size_t offset = 0;
        // Past 128 bytes one reduction covers 64 bytes, after one register for what differs early.
        if (length >= 128) {
            __vector unsigned char const zeros_u8x16 = vec_splats((unsigned char)0);
            if (!sz_equal_vec16_powervsx_(vec_xl(0, (unsigned char const *)a), vec_xl(0, (unsigned char const *)b)))
                return sz_false_k;
            for (offset = 16; offset + 64 <= length; offset += 64) {
                __vector unsigned char const first_u8x16 = vec_xor(vec_xl(0, (unsigned char const *)(a + offset + 0)),
                                                                   vec_xl(0, (unsigned char const *)(b + offset + 0)));
                __vector unsigned char const second_u8x16 = vec_xor(
                    vec_xl(0, (unsigned char const *)(a + offset + 16)),
                    vec_xl(0, (unsigned char const *)(b + offset + 16)));
                __vector unsigned char const third_u8x16 = vec_xor(vec_xl(0, (unsigned char const *)(a + offset + 32)),
                                                                   vec_xl(0, (unsigned char const *)(b + offset + 32)));
                __vector unsigned char const fourth_u8x16 = vec_xor(
                    vec_xl(0, (unsigned char const *)(a + offset + 48)),
                    vec_xl(0, (unsigned char const *)(b + offset + 48)));
                if (!vec_all_eq(vec_or(vec_or(first_u8x16, second_u8x16), vec_or(third_u8x16, fourth_u8x16)),
                                zeros_u8x16))
                    return sz_false_k;
            }
        }
        while (offset + 32 <= length) {
            __vector unsigned char a_low_u8x16 = vec_xl(0, (unsigned char const *)(a + offset));
            __vector unsigned char b_low_u8x16 = vec_xl(0, (unsigned char const *)(b + offset));
            __vector unsigned char a_high_u8x16 = vec_xl(0, (unsigned char const *)(a + offset + 16));
            __vector unsigned char b_high_u8x16 = vec_xl(0, (unsigned char const *)(b + offset + 16));
            __vector unsigned char eq_u8x16 = vec_and( //
                (__vector unsigned char)vec_cmpeq(a_low_u8x16, b_low_u8x16),
                (__vector unsigned char)vec_cmpeq(a_high_u8x16, b_high_u8x16));
            if (!vec_all_eq(eq_u8x16, vec_splats((unsigned char)0xFF))) return sz_false_k;
            offset += 32;
        }
        for (; offset + 16 <= length; offset += 16) {
            __vector unsigned char a_u8x16 = vec_xl(0, (unsigned char const *)(a + offset));
            __vector unsigned char b_u8x16 = vec_xl(0, (unsigned char const *)(b + offset));
            if (!sz_equal_vec16_powervsx_(a_u8x16, b_u8x16)) return sz_false_k;
        }
        // Final check - load the last register-long window from the end.
        __vector unsigned char a_tail_u8x16 = vec_xl(0, (unsigned char const *)(a + length - 16));
        __vector unsigned char b_tail_u8x16 = vec_xl(0, (unsigned char const *)(b + length - 16));
        return sz_equal_vec16_powervsx_(a_tail_u8x16, b_tail_u8x16);
    }
}

STRINGZILLA_API sz_status_t sz_equal_powervsx(sz_cptr_t a, sz_cptr_t b, sz_size_t length, sz_bool_t *equal,
                                              sz_stream_t stream) {
    sz_assert_(stream == STRINGZILLA_NULL);
    *equal = sz_equal_powervsx_(a, b, length);
    return sz_success_k;
}

STRINGZILLA_INLINE sz_ordering_t sz_order_powervsx_(sz_cptr_t a, sz_size_t a_length, sz_cptr_t b, sz_size_t b_length) {
    sz_u8_t const *a_u8 = (sz_u8_t const *)a;
    sz_u8_t const *b_u8 = (sz_u8_t const *)b;
    sz_size_t const shared_length = sz_min_of_two(a_length, b_length);
    sz_size_t offset = 0;

    // `vctzlsbb` reports 16 for an equal first half, so its fifth bit admits the second half index.
    for (; offset + 32 <= shared_length; offset += 32) {
        sz_size_t const first_index = vec_first_mismatch_index(vec_xl(0, a_u8 + offset), vec_xl(0, b_u8 + offset));
        sz_size_t const second_index = vec_first_mismatch_index(vec_xl(16, a_u8 + offset), vec_xl(16, b_u8 + offset));
        sz_size_t const mismatch_index = first_index + (first_index >> 4) * second_index;
        if (mismatch_index < 32) return sz_order_scalars_(a_u8[offset + mismatch_index], b_u8[offset + mismatch_index]);
    }

    // Two `lxvl` read the bytes left, padding both sides with the same zeros.
    sz_size_t const tail_length = shared_length - offset;
    sz_size_t const first_length = sz_min_of_two(tail_length, (sz_size_t)16);
    sz_size_t const first_index = vec_first_mismatch_index(vec_xl_len((unsigned char *)(a_u8 + offset), first_length),
                                                           vec_xl_len((unsigned char *)(b_u8 + offset), first_length));
    sz_size_t const second_index = vec_first_mismatch_index(
        vec_xl_len((unsigned char *)(a_u8 + offset + first_length), tail_length - first_length),
        vec_xl_len((unsigned char *)(b_u8 + offset + first_length), tail_length - first_length));
    sz_size_t const mismatch_index = first_index + (first_index >> 4) * second_index;
    if (mismatch_index < tail_length)
        return sz_order_scalars_(a_u8[offset + mismatch_index], b_u8[offset + mismatch_index]);
    return sz_order_scalars_(a_length, b_length);
}

STRINGZILLA_API sz_status_t sz_order_powervsx(sz_cptr_t a, sz_size_t a_length, sz_cptr_t b, sz_size_t b_length,
                                              sz_ordering_t *ordering, sz_stream_t stream) {
    sz_assert_(stream == STRINGZILLA_NULL);
    *ordering = sz_order_powervsx_(a, a_length, b, b_length);
    return sz_success_k;
}

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#endif // STRINGZILLA_TARGET_POWERVSX
#endif // STRINGZILLA_ARCH_PPC64_

#ifdef __cplusplus
}
#endif

#endif // STRINGZILLA_COMPARE_POWERVSX_H_
