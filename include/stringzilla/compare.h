/**
 *  @brief  Hardware-accelerated string comparison utilities.
 *  @file   compare.h
 *  @author Ash Vardanian
 *
 *  Includes core APIs:
 *
 *  - `sz_equal` - for equality comparison of two strings.
 *  - `sz_order` - for the relative order of two strings, similar to `memcmp`.
 *
 *  A valid suggestion may be to add an `sz_mismatch`, as the shared part of the `sz_order` and `sz_equal`.
 *  That would be great for a general-purpose library, but has little practical use for string processing.
 *
 *  The functions in this file can be used for both UTF-8 and other inputs.
 *  On platforms without masked loads they use interleaved prefix and suffix vector-loads
 *  to avoid scalar code, similar to the kernels in `memory.h`.
 */
#ifndef STRINGZILLA_COMPARE_H_
#define STRINGZILLA_COMPARE_H_

#include "types.h"

#ifdef __cplusplus
extern "C" {
#endif

#pragma region Core API

/**
 *  @brief  Checks if two strings are equal. Equivalent to `memcmp(a, b, length) == 0` in LibC and `a == b` in STL.
 *  @see    https://en.cppreference.com/w/c/string/byte/memcmp
 *
 *  @param[in] a First string to compare.
 *  @param[in] b Second string to compare.
 *  @param[in] length Number of bytes to compare in both strings.
 *
 *  @retval `sz_true_k` if strings are equal.
 *  @retval `sz_false_k` if strings are different.
 *
 *  Example usage:
 *
 *  @code{.c}
 *      #include <stringzilla/compare.h>
 *      int main() {
 *          return sz_equal("hello", "hello", 5) && !sz_equal("hello", "world", 5);
 *      }
 *  @endcode
 *
 *  @note   Selects the fastest implementation at compile- or run-time based on `SZ_DYNAMIC_DISPATCH`.
 *  @sa     sz_equal_serial, sz_equal_haswell, sz_equal_skylake, sz_equal_neon, sz_equal_sve
 */
SZ_DYNAMIC sz_bool_t sz_equal(sz_cptr_t a, sz_cptr_t b, sz_size_t length);

/**
 *  @brief  Compares two strings lexicographically. Equivalent to `memcmp(a, b, length)` in LibC.
 *          Mostly used in sorting and associative containers. Can be used for @b UTF-8 inputs.
 *  @see    https://en.cppreference.com/w/c/string/byte/memcmp
 *
 *  This function uses scalar code on most platforms, as in the majority of cases the strings that
 *  differ - will have differences among the very first characters and fetching more than one cache
 *  line may not be justified.
 *
 *  @param[in] a First string to compare.
 *  @param[in] a_length Number of bytes in the first string.
 *  @param[in] b Second string to compare.
 *  @param[in] b_length Number of bytes in the second string.
 *
 *  @retval `sz_less_k` if @p a is lexicographically smaller than @p b.
 *  @retval `sz_greater_k` if @p a is lexicographically greater than @p b.
 *  @retval `sz_equal_k` if strings @p a and @p b are identical.
 *
 *  Example usage:
 *
 *  @code{.c}
 *      #include <stringzilla/compare.h>
 *      int main() {
 *          return sz_order("apple", 5, "banana", 6) < 0 &&
 *                 sz_order("grape", 5, "grape", 5) == 0 &&
 *                 sz_order("zebra", 5, "apple", 5) > 0;
 *      }
 *  @endcode
 *
 *  @note   Selects the fastest implementation at compile- or run-time based on `SZ_DYNAMIC_DISPATCH`.
 *  @sa     sz_order_serial, sz_order_haswell, sz_order_skylake, sz_order_neon, sz_order_sve
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
#if SZ_USE_MISALIGNED_LOADS && !SZ_IS_BIG_ENDIAN_
    for (sz_u64_vec_t a_vec, b_vec; a + 8 <= min_end; a += 8, b += 8) {
        a_vec = sz_u64_load(a);
        b_vec = sz_u64_load(b);
        if (a_vec.u64 != b_vec.u64)
            return sz_order_scalars_(sz_u64_bytes_reverse(a_vec.u64), sz_u64_bytes_reverse(b_vec.u64));
    }
#endif
    for (; a != min_end; ++a, ++b)
        if (*a != *b) return sz_order_scalars_(*a, *b);

    // If the strings are equal up to `min_end`, then the shorter string is smaller
    return sz_order_scalars_(a_length, b_length);
}

#pragma endregion // Serial Implementation

/*  AVX2 implementation of the string search algorithms for Haswell processors and newer.
 *  Very minimalistic (compared to AVX-512), but still faster than the serial implementation.
 */
#pragma region Haswell Implementation
#if SZ_USE_HASWELL
#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("avx2"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("avx2")
#endif

SZ_PUBLIC sz_ordering_t sz_order_haswell(sz_cptr_t a, sz_size_t a_length, sz_cptr_t b, sz_size_t b_length) {
    //! Before optimizing this, read the "Operations Not Worth Optimizing" in Contributions Guide:
    //! https://github.com/ashvardanian/StringZilla/blob/main/CONTRIBUTING.md#general-performance-observations
    return sz_order_serial(a, a_length, b, b_length);
}

SZ_PUBLIC sz_bool_t sz_equal_haswell(sz_cptr_t a, sz_cptr_t b, sz_size_t length) {

    if (length < 8) {
        sz_cptr_t const a_end = a + length;
        while (a != a_end && *a == *b) a++, b++;
        return (sz_bool_t)(a_end == a);
    }
    // We can use 2x 64-bit interleaving loads for each string, and then compare them for equality.
    // The same approach is used in GLibC and was suggest by Denis Yaroshevskiy.
    // https://codebrowser.dev/glibc/glibc/sysdeps/x86_64/multiarch/memcmp-avx2-movbe.S.html#518
    // It shouldn't improve performance on microbenchmarks, but should be better in practice.
    else if (length <= 16) {
        sz_u64_t a_first_word = sz_u64_load(a).u64;
        sz_u64_t b_first_word = sz_u64_load(b).u64;
        sz_u64_t a_second_word = sz_u64_load(a + length - 8).u64;
        sz_u64_t b_second_word = sz_u64_load(b + length - 8).u64;
        return (sz_bool_t)((a_first_word == b_first_word) & (a_second_word == b_second_word));
    }
    // We can use 2x 128-bit interleaving loads for each string, and then compare them for equality.
    else if (length <= 32) {
        sz_u128_vec_t a_first_vec, b_first_vec, a_second_vec, b_second_vec;
        a_first_vec.xmm = _mm_lddqu_si128((__m128i const *)(a));
        b_first_vec.xmm = _mm_lddqu_si128((__m128i const *)(b));
        a_second_vec.xmm = _mm_lddqu_si128((__m128i const *)(a + length - 16));
        b_second_vec.xmm = _mm_lddqu_si128((__m128i const *)(b + length - 16));
        return (sz_bool_t)(_mm_movemask_epi8(_mm_and_si128( //
                               _mm_cmpeq_epi8(a_first_vec.xmm, b_first_vec.xmm),
                               _mm_cmpeq_epi8(a_second_vec.xmm, b_second_vec.xmm))) == 0xFFFF);
    }
    // We can use 2x 256-bit interleaving loads for each string, and then compare them for equality.
    else if (length <= 64) {
        sz_u256_vec_t a_first_vec, b_first_vec, a_second_vec, b_second_vec;
        a_first_vec.ymm = _mm256_lddqu_si256((__m256i const *)(a));
        b_first_vec.ymm = _mm256_lddqu_si256((__m256i const *)(b));
        a_second_vec.ymm = _mm256_lddqu_si256((__m256i const *)(a + length - 32));
        b_second_vec.ymm = _mm256_lddqu_si256((__m256i const *)(b + length - 32));
        return (sz_bool_t)(_mm256_movemask_epi8(_mm256_and_si256( //
                               _mm256_cmpeq_epi8(a_first_vec.ymm, b_first_vec.ymm),
                               _mm256_cmpeq_epi8(a_second_vec.ymm, b_second_vec.ymm))) == (int)0xFFFFFFFF);
    }
    else {
        sz_size_t i = 0;
        sz_u256_vec_t a_vec, b_vec;
        do {
            a_vec.ymm = _mm256_lddqu_si256((__m256i const *)(a + i));
            b_vec.ymm = _mm256_lddqu_si256((__m256i const *)(b + i));
            // One approach can be to use "movemasks", but we could also use a bitwise
            // matching like `_mm256_testnzc_si256`.
            if (_mm256_movemask_epi8(_mm256_cmpeq_epi8(a_vec.ymm, b_vec.ymm)) != (int)0xFFFFFFFF) return sz_false_k;
            i += 32;
        } while (i + 32 <= length);
        a_vec.ymm = _mm256_lddqu_si256((__m256i const *)(a + length - 32));
        b_vec.ymm = _mm256_lddqu_si256((__m256i const *)(b + length - 32));
        return (sz_bool_t)(_mm256_movemask_epi8(_mm256_cmpeq_epi8(a_vec.ymm, b_vec.ymm)) == (int)0xFFFFFFFF);
    }
}

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#endif            // SZ_USE_HASWELL
#pragma endregion // Haswell Implementation

/*  AVX512 implementation of the string search algorithms for Skylake and newer CPUs.
 *  Includes extensions: F, CD, ER, PF, VL, DQ, BW.
 *
 *  This is the "starting level" for the advanced algorithms using K-mask registers on x86.
 */
#pragma region Skylake Implementation
#if SZ_USE_SKYLAKE
#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("avx,avx512f,avx512vl,avx512bw,bmi,bmi2"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("avx", "avx512f", "avx512vl", "avx512bw", "bmi", "bmi2")
#endif

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
    __mmask64 head_mask = sz_u64_mask_until_(head_length);
    a_vec.zmm = _mm512_maskz_loadu_epi8(head_mask, a);
    b_vec.zmm = _mm512_maskz_loadu_epi8(head_mask, b);
    __mmask64 mask_not_equal = _mm512_cmpneq_epi8_mask(a_vec.zmm, b_vec.zmm);
    if (mask_not_equal != 0) {
        sz_u64_t first_diff = _tzcnt_u64(mask_not_equal);
        char a_char = a_vec.u8s[first_diff];
        char b_char = b_vec.u8s[first_diff];
        return sz_order_scalars_(a_char, b_char);
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
            return sz_order_scalars_(a_char, b_char);
        }
        a += 64, b += 64, a_length -= 64, b_length -= 64;
    }

    // In most common scenarios at least one of the strings is under 64 bytes.
    if (a_length | b_length) {
        a_mask = sz_u64_clamp_mask_until_(a_length);
        b_mask = sz_u64_clamp_mask_until_(b_length);
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
            return sz_order_scalars_(a_char, b_char);
        }
        // From logic perspective, the hardest cases are "abc\0" and "abc".
        // The result must be `sz_greater_k`, as the latter is shorter.
        else { return sz_order_scalars_(a_length, b_length); }
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
        mask = sz_u64_mask_until_(length);
        a_vec.zmm = _mm512_maskz_loadu_epi8(mask, a);
        b_vec.zmm = _mm512_maskz_loadu_epi8(mask, b);
        // Reuse the same `mask` variable to find the bit that doesn't match
        mask = _mm512_mask_cmpneq_epi8_mask(mask, a_vec.zmm, b_vec.zmm);
        return (sz_bool_t)(mask == 0);
    }

    return sz_true_k;
}

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#endif            // SZ_USE_SKYLAKE
#pragma endregion // Skylake Implementation

/*  Implementation of the string search algorithms using the Arm NEON instruction set, available on 64-bit
 *  Arm processors. Covers billions of mobile CPUs worldwide, including Apple's A-series, and Qualcomm's Snapdragon.
 */
#pragma region NEON Implementation
#if SZ_USE_NEON
#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("arch=armv8.2-a+simd"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("arch=armv8.2-a+simd")
#endif

SZ_PUBLIC sz_ordering_t sz_order_neon(sz_cptr_t a, sz_size_t a_length, sz_cptr_t b, sz_size_t b_length) {
    //! Before optimizing this, read the "Operations Not Worth Optimizing" in Contributions Guide:
    //! https://github.com/ashvardanian/StringZilla/blob/main/CONTRIBUTING.md#general-performance-observations
    return sz_order_serial(a, a_length, b, b_length);
}

SZ_PUBLIC sz_bool_t sz_equal_neon(sz_cptr_t a, sz_cptr_t b, sz_size_t length) {
    if (length < 16) return sz_equal_serial(a, b, length);

    sz_u128_vec_t a_vec, b_vec;
    sz_size_t offset = 0;
    do {
        a_vec.u8x16 = vld1q_u8((sz_u8_t const *)(a + offset));
        b_vec.u8x16 = vld1q_u8((sz_u8_t const *)(b + offset));
        uint8x16_t cmp = vceqq_u8(a_vec.u8x16, b_vec.u8x16);
        if (vminvq_u8(cmp) != 255) return sz_false_k; // Check if all bytes match
        offset += 16;
    } while (offset + 16 <= length);

    // For final check - load the last register-long piece of content from the end
    a_vec.u8x16 = vld1q_u8((sz_u8_t const *)(a + length - 16));
    b_vec.u8x16 = vld1q_u8((sz_u8_t const *)(b + length - 16));
    uint8x16_t cmp = vceqq_u8(a_vec.u8x16, b_vec.u8x16);
    if (vminvq_u8(cmp) != 255) return sz_false_k;
    return sz_true_k;
}

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#endif            // SZ_USE_NEON
#pragma endregion // NEON Implementation

/*  Implementation of the string search algorithms using the Arm SVE variable-length registers,
 *  available in Arm v9 processors, like in Apple M4+ and Graviton 3+ CPUs.
 */
#pragma region SVE Implementation
#if SZ_USE_SVE
#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("arch=armv8.2-a+sve"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("arch=armv8.2-a+sve")
#endif

SZ_PUBLIC sz_bool_t sz_equal_sve(sz_cptr_t a, sz_cptr_t b, sz_size_t length) {
    // Determine the number of bytes in an SVE vector.
    sz_size_t const vector_bytes = svcntb();
    sz_size_t progress = 0;
    do {
        svbool_t progress_vec = svwhilelt_b8((sz_u64_t)progress, (sz_u64_t)length);
        svuint8_t a_vec = svld1(progress_vec, (sz_u8_t const *)(a + progress));
        svuint8_t b_vec = svld1(progress_vec, (sz_u8_t const *)(b + progress));
        // Compare: generate a predicate marking lanes where a!=b
        svbool_t not_equal_vec = svcmpne(progress_vec, a_vec, b_vec);
        if (svptest_any(progress_vec, not_equal_vec)) return sz_false_k;
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
#elif SZ_USE_SVE
    return sz_equal_sve(a, b, length);
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
#elif SZ_USE_SVE
    return sz_order_sve(a, a_length, b, b_length);
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
