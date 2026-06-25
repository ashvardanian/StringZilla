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

SZ_INTERNAL sz_bool_t sz_equal_powervsx_vec16_(__vector unsigned char a_u8x16, __vector unsigned char b_u8x16) {
    // On Power10 `vec_first_mismatch_index` reports the first differing lane (16 if none) in a
    // single instruction, replacing the `vec_all_eq` compare + branch reduction. The Power9
    // fallback keeps the byte-for-byte identical `vec_all_eq` semantics.
#if defined(_ARCH_PWR10)
    return (sz_bool_t)(vec_first_mismatch_index(a_u8x16, b_u8x16) == 16);
#else
    return (sz_bool_t)vec_all_eq(a_u8x16, b_u8x16); // True only when every byte matched.
#endif
}

SZ_PUBLIC sz_bool_t sz_equal_powervsx(sz_cptr_t a, sz_cptr_t b, sz_size_t length) {

    if (length < 8) {
        sz_cptr_t const a_end = a + length;
        while (a != a_end && *a == *b) a++, b++;
        return (sz_bool_t)(a_end == a);
    }
    // Two overlapping 64-bit loads cover the whole [8,16) window with one branch, mirroring the
    // x86 sibling (`compare/haswell.h`) and the GLibC `memcmp-avx2` tiering.
    else if (length <= 16) {
        sz_u64_t a_first_word = sz_u64_load(a).u64;
        sz_u64_t b_first_word = sz_u64_load(b).u64;
        sz_u64_t a_second_word = sz_u64_load(a + length - 8).u64;
        sz_u64_t b_second_word = sz_u64_load(b + length - 8).u64;
        return (sz_bool_t)((a_first_word == b_first_word) & (a_second_word == b_second_word));
    }
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
            if (!sz_equal_powervsx_vec16_(a_u8x16, b_u8x16)) return sz_false_k;
        }
        // Final check - load the last register-long window from the end.
        __vector unsigned char a_tail_u8x16 = vec_xl(0, (unsigned char const *)(a + length - 16));
        __vector unsigned char b_tail_u8x16 = vec_xl(0, (unsigned char const *)(b + length - 16));
        return sz_equal_powervsx_vec16_(a_tail_u8x16, b_tail_u8x16);
    }
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
