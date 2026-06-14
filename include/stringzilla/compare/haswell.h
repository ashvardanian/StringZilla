/**
 *  @brief Haswell (AVX2) backend for string comparison utilities.
 *  @file include/stringzilla/compare/haswell.h
 *  @author Ash Vardanian
 *  @sa include/stringzilla/compare.h
 */
#ifndef STRINGZILLA_COMPARE_HASWELL_H_
#define STRINGZILLA_COMPARE_HASWELL_H_

#include "stringzilla/types.h"
#include "stringzilla/compare/serial.h"

#ifdef __cplusplus
extern "C" {
#endif

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
        sz_size_t byte_index = 0;
        sz_u256_vec_t a_vec, b_vec;
        do {
            a_vec.ymm = _mm256_lddqu_si256((__m256i const *)(a + byte_index));
            b_vec.ymm = _mm256_lddqu_si256((__m256i const *)(b + byte_index));
            // One approach can be to use "movemasks", but we could also use a bitwise
            // matching like `_mm256_testnzc_si256`.
            if (_mm256_movemask_epi8(_mm256_cmpeq_epi8(a_vec.ymm, b_vec.ymm)) != (int)0xFFFFFFFF) return sz_false_k;
            byte_index += 32;
        } while (byte_index + 32 <= length);
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
#endif // SZ_USE_HASWELL

#ifdef __cplusplus
}
#endif

#endif // STRINGZILLA_COMPARE_HASWELL_H_
