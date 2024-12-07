/**
 *  @brief  Hardware-accelerated string comparison utilities.
 *  @file   compare.h
 *  @author Ash Vardanian
 *
 *  Includes core APIs:
 *
 *  - `sz_equal` - for equality comparison of two strings.
 *  - `sz_order` - for the relative order of two strings, similar to `memcmp`.
 *  - TODO: `sz_mismatch`, `sz_rmismatch` - to supersede `sz_equal`.
 *  - TODO: `sz_order_utf8` - for the relative order of two UTF-8 strings.
 */
#ifndef STRINGZILLA_COMPARE_H_
#define STRINGZILLA_COMPARE_H_

#include "types.h"

#ifdef __cplusplus
extern "C" {
#endif

#pragma region Core API

/**
 *  @brief  Checks if two string are equal.
 *          Similar to `memcmp(a, b, length) == 0` in LibC and `a == b` in STL.
 *
 *  The implementation of this function is very similar to `sz_order`, but the usage patterns are different.
 *  This function is more often used in parsing, while `sz_order` is often used in sorting.
 *  It works best on platforms with cheap
 *
 *  @param a        First string to compare.
 *  @param b        Second string to compare.
 *  @param length   Number of bytes in both strings.
 *  @return         1 if strings match, 0 otherwise.
 */
SZ_DYNAMIC sz_bool_t sz_equal(sz_cptr_t a, sz_cptr_t b, sz_size_t length);

/**
 *  @brief  Estimates the relative order of two strings. Equivalent to `memcmp(a, b, length)` in LibC.
 *          Can be used on different length strings.
 *
 *  @param a        First string to compare.
 *  @param a_length Number of bytes in the first string.
 *  @param b        Second string to compare.
 *  @param b_length Number of bytes in the second string.
 *  @return         Negative if (a < b), positive if (a > b), zero if they are equal.
 */
SZ_DYNAMIC sz_ordering_t sz_order(sz_cptr_t a, sz_size_t a_length, sz_cptr_t b, sz_size_t b_length);

/** @copydoc sz_equal */
SZ_PUBLIC sz_bool_t sz_equal_serial(sz_cptr_t a, sz_cptr_t b, sz_size_t length);
/** @copydoc sz_order */
SZ_PUBLIC sz_ordering_t sz_order_serial(sz_cptr_t a, sz_size_t a_length, sz_cptr_t b, sz_size_t b_length);

#if SZ_USE_HASWELL
/** @copydoc sz_equal */
SZ_PUBLIC sz_bool_t sz_equal_haswell(sz_cptr_t a, sz_cptr_t b, sz_size_t length);
/** @copydoc sz_order */
SZ_PUBLIC sz_ordering_t sz_order_haswell(sz_cptr_t a, sz_size_t a_length, sz_cptr_t b, sz_size_t b_length);
#endif

#if SZ_USE_SKYLAKE
/** @copydoc sz_equal */
SZ_PUBLIC sz_bool_t sz_equal_skylake(sz_cptr_t a, sz_cptr_t b, sz_size_t length);
/** @copydoc sz_order */
SZ_PUBLIC sz_ordering_t sz_order_skylake(sz_cptr_t a, sz_size_t a_length, sz_cptr_t b, sz_size_t b_length);
#endif

#if SZ_USE_NEON
/** @copydoc sz_equal */
SZ_PUBLIC sz_bool_t sz_equal_neon(sz_cptr_t a, sz_cptr_t b, sz_size_t length);
/** @copydoc sz_order */
SZ_PUBLIC sz_ordering_t sz_order_neon(sz_cptr_t a, sz_size_t a_length, sz_cptr_t b, sz_size_t b_length);
#endif

#pragma endregion // Core API

#pragma region Serial Implementation

/**
 *  @brief  Byte-level equality comparison between two strings.
 *          If unaligned loads are allowed, uses a switch-table to avoid loops on short strings.
 */
SZ_PUBLIC sz_bool_t sz_equal_serial(sz_cptr_t a, sz_cptr_t b, sz_size_t length) {
    sz_cptr_t const a_end = a + length;
#if SZ_USE_MISALIGNED_LOADS
    if (length >= SZ_SWAR_THRESHOLD) {
        sz_u64_vec_t a_vec, b_vec;
        for (; a + 8 <= a_end; a += 8, b += 8) {
            a_vec = sz_u64_load(a);
            b_vec = sz_u64_load(b);
            if (a_vec.u64 != b_vec.u64) return sz_false_k;
        }
    }
#endif
    while (a != a_end && *a == *b) a++, b++;
    return (sz_bool_t)(a_end == a);
}

SZ_PUBLIC sz_ordering_t sz_order_serial(sz_cptr_t a, sz_size_t a_length, sz_cptr_t b, sz_size_t b_length) {
    sz_bool_t a_shorter = (sz_bool_t)(a_length < b_length);
    sz_size_t min_length = a_shorter ? a_length : b_length;
    sz_cptr_t min_end = a + min_length;
#if SZ_USE_MISALIGNED_LOADS && !_SZ_IS_BIG_ENDIAN
    for (sz_u64_vec_t a_vec, b_vec; a + 8 <= min_end; a += 8, b += 8) {
        a_vec = sz_u64_load(a);
        b_vec = sz_u64_load(b);
        if (a_vec.u64 != b_vec.u64)
            return _sz_order_scalars(sz_u64_bytes_reverse(a_vec.u64), sz_u64_bytes_reverse(b_vec.u64));
    }
#endif
    for (; a != min_end; ++a, ++b)
        if (*a != *b) return _sz_order_scalars(*a, *b);

    // If the strings are equal up to `min_end`, then the shorter string is smaller
    return _sz_order_scalars(a_length, b_length);
}

#pragma endregion // Serial Implementation

/*  AVX2 implementation of the string search algorithms for Haswell processors and newer.
 *  Very minimalistic (compared to AVX-512), but still faster than the serial implementation.
 */
#pragma region Haswell Implementation
#if SZ_USE_HASWELL
#pragma GCC push_options
#pragma GCC target("avx2")
#pragma clang attribute push(__attribute__((target("avx2"))), apply_to = function)

SZ_PUBLIC sz_ordering_t sz_order_haswell(sz_cptr_t a, sz_size_t a_length, sz_cptr_t b, sz_size_t b_length) {
    //! Before optimizing this, read the "Operations Not Worth Optimizing" in Contributions Guide:
    //! https://github.com/ashvardanian/StringZilla/blob/main/CONTRIBUTING.md#general-performance-observations
    return sz_order_serial(a, a_length, b, b_length);
}

SZ_PUBLIC sz_bool_t sz_equal_haswell(sz_cptr_t a, sz_cptr_t b, sz_size_t length) {
    sz_u256_vec_t a_vec, b_vec;

    while (length >= 32) {
        a_vec.ymm = _mm256_lddqu_si256((__m256i const *)a);
        b_vec.ymm = _mm256_lddqu_si256((__m256i const *)b);
        // One approach can be to use "movemasks", but we could also use a bitwise matching like `_mm256_testnzc_si256`.
        int difference_mask = ~_mm256_movemask_epi8(_mm256_cmpeq_epi8(a_vec.ymm, b_vec.ymm));
        if (difference_mask == 0) { a += 32, b += 32, length -= 32; }
        else { return sz_false_k; }
    }

    if (length) return sz_equal_serial(a, b, length);
    return sz_true_k;
}

#pragma clang attribute pop
#pragma GCC pop_options
#endif            // SZ_USE_HASWELL
#pragma endregion // Haswell Implementation

/*  AVX512 implementation of the string search algorithms for Skylake and newer CPUs.
 *  Includes extensions: F, CD, ER, PF, VL, DQ, BW.
 *
 *  This is the "starting level" for the advanced algorithms using K-mask registers on x86.
 */
#pragma region Skylake Implementation
#if SZ_USE_SKYLAKE
#pragma GCC push_options
#pragma GCC target("avx", "avx512f", "avx512vl", "avx512bw", "bmi", "bmi2")
#pragma clang attribute push(__attribute__((target("avx,avx512f,avx512vl,avx512bw,bmi,bmi2"))), apply_to = function)

SZ_PUBLIC sz_ordering_t sz_order_skylake(sz_cptr_t a, sz_size_t a_length, sz_cptr_t b, sz_size_t b_length) {
    sz_u512_vec_t a_vec, b_vec;

    // Pointer arithmetic is cheap, fetching memory is not!
    // So we can use the masked loads to fetch at most one cache-line for each string,
    // compare the prefixes, and only then move forward.
    sz_size_t a_head_length = 64 - ((sz_size_t)a % 64); // 63 or less.
    sz_size_t b_head_length = 64 - ((sz_size_t)b % 64); // 63 or less.
    a_head_length = a_head_length < a_length ? a_head_length : a_length;
    b_head_length = b_head_length < b_length ? b_head_length : b_length;
    sz_size_t head_length = a_head_length < b_head_length ? a_head_length : b_head_length;
    __mmask64 head_mask = _sz_u64_mask_until(head_length);
    a_vec.zmm = _mm512_maskz_loadu_epi8(head_mask, a);
    b_vec.zmm = _mm512_maskz_loadu_epi8(head_mask, b);
    __mmask64 mask_not_equal = _mm512_cmpneq_epi8_mask(a_vec.zmm, b_vec.zmm);
    if (mask_not_equal != 0) {
        sz_u64_t first_diff = _tzcnt_u64(mask_not_equal);
        char a_char = a_vec.u8s[first_diff];
        char b_char = b_vec.u8s[first_diff];
        return _sz_order_scalars(a_char, b_char);
    }
    else if (head_length == a_length && head_length == b_length) { return sz_equal_k; }
    else { a += head_length, b += head_length, a_length -= head_length, b_length -= head_length; }

    // The rare case, when both string are very long.
    __mmask64 a_mask, b_mask;
    while ((a_length >= 64) & (b_length >= 64)) {
        a_vec.zmm = _mm512_loadu_si512(a);
        b_vec.zmm = _mm512_loadu_si512(b);
        mask_not_equal = _mm512_cmpneq_epi8_mask(a_vec.zmm, b_vec.zmm);
        if (mask_not_equal != 0) {
            sz_u64_t first_diff = _tzcnt_u64(mask_not_equal);
            char a_char = a_vec.u8s[first_diff];
            char b_char = b_vec.u8s[first_diff];
            return _sz_order_scalars(a_char, b_char);
        }
        a += 64, b += 64, a_length -= 64, b_length -= 64;
    }

    // In most common scenarios at least one of the strings is under 64 bytes.
    if (a_length | b_length) {
        a_mask = _sz_u64_clamp_mask_until(a_length);
        b_mask = _sz_u64_clamp_mask_until(b_length);
        a_vec.zmm = _mm512_maskz_loadu_epi8(a_mask, a);
        b_vec.zmm = _mm512_maskz_loadu_epi8(b_mask, b);
        // The AVX-512 `_mm512_mask_cmpneq_epi8_mask` intrinsics are generally handy in such environments.
        // They, however, have latency 3 on most modern CPUs. Using AVX2: `_mm256_cmpeq_epi8` would have
        // been cheaper, if we didn't have to apply `_mm256_movemask_epi8` afterwards.
        mask_not_equal = _mm512_cmpneq_epi8_mask(a_vec.zmm, b_vec.zmm);
        if (mask_not_equal != 0) {
            sz_u64_t first_diff = _tzcnt_u64(mask_not_equal);
            char a_char = a_vec.u8s[first_diff];
            char b_char = b_vec.u8s[first_diff];
            return _sz_order_scalars(a_char, b_char);
        }
        // From logic perspective, the hardest cases are "abc\0" and "abc".
        // The result must be `sz_greater_k`, as the latter is shorter.
        else { return _sz_order_scalars(a_length, b_length); }
    }

    return sz_equal_k;
}

SZ_PUBLIC sz_bool_t sz_equal_skylake(sz_cptr_t a, sz_cptr_t b, sz_size_t length) {
    __mmask64 mask;
    sz_u512_vec_t a_vec, b_vec;

    while (length >= 64) {
        a_vec.zmm = _mm512_loadu_si512(a);
        b_vec.zmm = _mm512_loadu_si512(b);
        mask = _mm512_cmpneq_epi8_mask(a_vec.zmm, b_vec.zmm);
        if (mask != 0) return sz_false_k;
        a += 64, b += 64, length -= 64;
    }

    if (length) {
        mask = _sz_u64_mask_until(length);
        a_vec.zmm = _mm512_maskz_loadu_epi8(mask, a);
        b_vec.zmm = _mm512_maskz_loadu_epi8(mask, b);
        // Reuse the same `mask` variable to find the bit that doesn't match
        mask = _mm512_mask_cmpneq_epi8_mask(mask, a_vec.zmm, b_vec.zmm);
        return (sz_bool_t)(mask == 0);
    }

    return sz_true_k;
}

#pragma clang attribute pop
#pragma GCC pop_options
#endif            // SZ_USE_SKYLAKE
#pragma endregion // Skylake Implementation

/*  AVX512 implementation of the string search algorithms for Ice Lake and newer CPUs.
 *  Includes extensions:
 *      - 2017 Skylake: F, CD, ER, PF, VL, DQ, BW,
 *      - 2018 CannonLake: IFMA, VBMI,
 *      - 2019 Ice Lake: VPOPCNTDQ, VNNI, VBMI2, BITALG, GFNI, VPCLMULQDQ, VAES.
 */
#pragma region Ice Lake Implementation
#if SZ_USE_ICE
#pragma GCC push_options
#pragma GCC target("avx", "avx512f", "avx512vl", "avx512bw", "avx512dq", "avx512vbmi", "bmi", "bmi2")
#pragma clang attribute push(__attribute__((target("avx,avx512f,avx512vl,avx512bw,avx512dq,avx512vbmi,bmi,bmi2"))), \
                             apply_to = function)

/* Nothing here for now. */

#pragma clang attribute pop
#pragma GCC pop_options
#endif            // SZ_USE_ICE
#pragma endregion // Ice Lake Implementation

/*  Implementation of the string search algorithms using the Arm NEON instruction set, available on 64-bit
 *  Arm processors. Covers billions of mobile CPUs worldwide, including Apple's A-series, and Qualcomm's Snapdragon.
 */
#pragma region NEON Implementation
#if SZ_USE_NEON
#pragma GCC push_options
#pragma GCC target("arch=armv8.2-a+simd")
#pragma clang attribute push(__attribute__((target("arch=armv8.2-a+simd"))), apply_to = function)

SZ_PUBLIC sz_ordering_t sz_order_neon(sz_cptr_t a, sz_size_t a_length, sz_cptr_t b, sz_size_t b_length) {
    //! Before optimizing this, read the "Operations Not Worth Optimizing" in Contributions Guide:
    //! https://github.com/ashvardanian/StringZilla/blob/main/CONTRIBUTING.md#general-performance-observations
    return sz_order_serial(a, a_length, b, b_length);
}

SZ_PUBLIC sz_bool_t sz_equal_neon(sz_cptr_t a, sz_cptr_t b, sz_size_t length) {
    sz_u128_vec_t a_vec, b_vec;
    for (; length >= 16; a += 16, b += 16, length -= 16) {
        a_vec.u8x16 = vld1q_u8((sz_u8_t const *)a);
        b_vec.u8x16 = vld1q_u8((sz_u8_t const *)b);
        uint8x16_t cmp = vceqq_u8(a_vec.u8x16, b_vec.u8x16);
        if (vminvq_u8(cmp) != 255) { return sz_false_k; } // Check if all bytes match
    }

    // Handle remaining bytes
    if (length) return sz_equal_serial(a, b, length);
    return sz_true_k;
}

#pragma clang attribute pop
#pragma GCC pop_options
#endif            // SZ_USE_NEON
#pragma endregion // NEON Implementation

/*  Implementation of the string search algorithms using the Arm SVE variable-length registers,
 *  available in Arm v9 processors, like in Apple M4+ and Graviton 3+ CPUs.
 */
#pragma region SVE Implementation
#if SZ_USE_SVE
#pragma GCC push_options
#pragma GCC target("arch=armv8.2-a+sve")
#pragma clang attribute push(__attribute__((target("arch=armv8.2-a+sve"))), apply_to = function)

/* Nothing here for now. */

#pragma clang attribute pop
#pragma GCC pop_options
#endif            // SZ_USE_SVE
#pragma endregion // SVE Implementation

/*  Pick the right implementation for the string search algorithms.
 *  To override this behavior and precompile all backends - set `SZ_DYNAMIC_DISPATCH` to 1.
 */
#pragma region Compile Time Dispatching
#if !SZ_DYNAMIC_DISPATCH

SZ_DYNAMIC sz_bool_t sz_equal(sz_cptr_t a, sz_cptr_t b, sz_size_t length) {
#if SZ_USE_SKYLAKE
    return sz_equal_skylake(a, b, length);
#elif SZ_USE_HASWELL
    return sz_equal_haswell(a, b, length);
#elif SZ_USE_NEON
    return sz_equal_neon(a, b, length);
#else
    return sz_equal_serial(a, b, length);
#endif
}

SZ_DYNAMIC sz_ordering_t sz_order(sz_cptr_t a, sz_size_t a_length, sz_cptr_t b, sz_size_t b_length) {
#if SZ_USE_SKYLAKE
    return sz_order_skylake(a, a_length, b, b_length);
#elif SZ_USE_HASWELL
    return sz_order_haswell(a, a_length, b, b_length);
#elif SZ_USE_NEON
    return sz_order_neon(a, a_length, b, b_length);
#else
    return sz_order_serial(a, a_length, b, b_length);
#endif
}

#endif            // !SZ_DYNAMIC_DISPATCH
#pragma endregion // Compile Time Dispatching

#ifdef __cplusplus
}
#endif // __cplusplus
#endif // STRINGZILLA_COMPARE_H_
