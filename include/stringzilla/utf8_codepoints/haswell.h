/**
 *  @brief Haswell (AVX2) backend for UTF-8 codepoint mechanics.
 *  @file include/stringzilla/utf8_codepoints/haswell.h
 *  @author Ash Vardanian
 */
#ifndef STRINGZILLA_UTF8_CODEPOINTS_HASWELL_H_
#define STRINGZILLA_UTF8_CODEPOINTS_HASWELL_H_

#include "stringzilla/types.h"
#include "stringzilla/utf8_codepoints/serial.h"

#ifdef __cplusplus
extern "C" {
#endif

#if SZ_USE_HASWELL
#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("avx2,bmi,bmi2,popcnt"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("avx2,bmi,bmi2,popcnt")
#endif

/** @brief  Mask for `_mm256_maskstore_epi64` selecting the low `count` (0..4) of four 64-bit lanes. */
SZ_INTERNAL __m256i sz_mm256_store_mask_epi64_(sz_size_t count) {
    return _mm256_cmpgt_epi64(_mm256_set1_epi64x((long long)count), _mm256_setr_epi64x(0, 1, 2, 3));
}

SZ_PUBLIC sz_size_t sz_utf8_count_haswell(sz_cptr_t text, sz_size_t length) {
    sz_u256_vec_t continuation_mask_vec, continuation_pattern_vec;
    continuation_mask_vec.ymm = _mm256_set1_epi8((char)0xC0);
    continuation_pattern_vec.ymm = _mm256_set1_epi8((char)0x80);

    sz_u8_t const *text_u8 = (sz_u8_t const *)text;
    sz_size_t char_count = 0;

    // Process 32 bytes at a time
    sz_u256_vec_t text_vec, headers_vec;
    while (length >= 32) {
        text_vec.ymm = _mm256_loadu_si256((__m256i const *)text_u8);

        // Apply mask (byte & 0xC0) to extract top 2 bits of each byte
        headers_vec.ymm = _mm256_and_si256(text_vec.ymm, continuation_mask_vec.ymm);

        // Compare with 0x80 (0b10000000) to find continuation bytes
        sz_u32_t start_byte_mask = ~(sz_u32_t)_mm256_movemask_epi8(
            _mm256_cmpeq_epi8(headers_vec.ymm, continuation_pattern_vec.ymm));

        char_count += _mm_popcnt_u32(start_byte_mask);
        text_u8 += 32;
        length -= 32;
    }

    // Process remaining bytes with serial
    char_count += sz_utf8_count_serial((sz_cptr_t)text_u8, length);
    return char_count;
}

SZ_PUBLIC sz_cptr_t sz_utf8_find_nth_haswell(sz_cptr_t text, sz_size_t length, sz_size_t n) {
    // The logic of this function is similar to `sz_utf8_count_haswell`, but uses PDEP
    // instruction in the inner loop to locate Nth character start byte efficiently
    // without one more loop.
    sz_u256_vec_t continuation_mask_vec, continuation_pattern_vec;
    continuation_mask_vec.ymm = _mm256_set1_epi8((char)0xC0);
    continuation_pattern_vec.ymm = _mm256_set1_epi8((char)0x80);

    sz_u8_t const *text_u8 = (sz_u8_t const *)text;

    // Process 32 bytes at a time
    sz_u256_vec_t text_vec, headers_vec;
    while (length >= 32) {
        text_vec.ymm = _mm256_loadu_si256((__m256i const *)text_u8);

        // Apply mask (byte & 0xC0) to extract top 2 bits of each byte
        headers_vec.ymm = _mm256_and_si256(text_vec.ymm, continuation_mask_vec.ymm);

        // Compare with 0x80 (0b10000000) to find continuation bytes
        sz_u32_t start_byte_mask = ~(sz_u32_t)_mm256_movemask_epi8(
            _mm256_cmpeq_epi8(headers_vec.ymm, continuation_pattern_vec.ymm));
        sz_size_t start_byte_count = _mm_popcnt_u32(start_byte_mask);

        // The Nth start byte is not in this window: advance to the next block.
        if (n >= start_byte_count) {
            n -= start_byte_count;
            text_u8 += 32;
            length -= 32;
            continue;
        }

        // `_pdep_u32` isolates the Nth set bit; `_tzcnt_u32` then gives its lane.
        sz_u32_t deposited_bits = _pdep_u32((sz_u32_t)1 << n, start_byte_mask);
        int byte_offset = (int)_tzcnt_u32(deposited_bits);
        return (sz_cptr_t)(text_u8 + byte_offset);
    }

    // Process remaining bytes with serial
    return sz_utf8_find_nth_serial((sz_cptr_t)text_u8, length, n);
}

/*  UAX-29 word boundary detection (vectorized). The same outer walk as the serial reference, but all-ASCII
 *  windows resolve their boundaries with `_mm256_shuffle_epi8` Word_Break classification plus a local pair
 *  decision. Any stateful rule (WB4/6/7/7a/11/12/15/16/3c), non-ASCII, or window-edge position defers to
 *  `sz_utf8_is_word_boundary_serial`, keeping the output byte-exact versus serial. */

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#endif // SZ_USE_HASWELL

#ifdef __cplusplus
}
#endif

#endif // STRINGZILLA_UTF8_CODEPOINTS_HASWELL_H_
