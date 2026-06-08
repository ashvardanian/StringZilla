/**
 *  @brief Haswell (AVX2) backend for UTF-8 traversal.
 *  @file include/stringzilla/utf8_iterate/haswell.h
 *  @author Ash Vardanian
 *  @sa utf8.h
 */
#ifndef STRINGZILLA_UTF8_ITERATE_HASWELL_H_
#define STRINGZILLA_UTF8_ITERATE_HASWELL_H_

#include "stringzilla/types.h"
#include "stringzilla/utf8_iterate/serial.h"

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

/**
 *  @brief Unsigned byte greater-than-or-equal comparison for AVX2.
 *
 *  AVX2 lacks unsigned comparison intrinsics like `_mm256_cmpge_epu8`.
 *  This uses the identity: a >= b  ⟺  max(a, b) == a
 *  Since `_mm256_max_epu8` treats bytes as unsigned, this gives correct results.
 */
SZ_INTERNAL __m256i sz_mm256_cmpge_epu8_(__m256i a, __m256i b) { return _mm256_cmpeq_epi8(_mm256_max_epu8(a, b), a); }

SZ_PUBLIC sz_cptr_t sz_utf8_find_newline_haswell(sz_cptr_t text, sz_size_t length, sz_size_t *matched_length) {

    // We need to check if the ASCII chars in [10,13] (same as '\n', '\v', '\f', '\r') are present.
    // The last one - '\r' - needs special handling to differentiate between "\r" and "\r\n".
    __m256i n_u8x32 = _mm256_set1_epi8('\n');
    __m256i v_u8x32 = _mm256_set1_epi8('\v');
    __m256i f_u8x32 = _mm256_set1_epi8('\f');
    __m256i r_u8x32 = _mm256_set1_epi8('\r');

    // We also need to match the 2-byte newline character 0xC285 (NEL),
    // as well as the 3-byte characters 0xE280A8 (PS) and 0xE280A9 (LS).
    __m256i x_c2_u8x32 = _mm256_set1_epi8('\xC2');
    __m256i x_85_u8x32 = _mm256_set1_epi8('\x85');
    __m256i x_e2_u8x32 = _mm256_set1_epi8('\xE2');
    __m256i x_80_u8x32 = _mm256_set1_epi8('\x80');
    __m256i x_a8_u8x32 = _mm256_set1_epi8('\xA8');
    __m256i x_a9_u8x32 = _mm256_set1_epi8('\xA9');

    // We check 32 bytes of data at once, but only step forward by 30 bytes for split-register matches.
    while (length >= 32) {
        __m256i text_u8x32 = _mm256_loadu_si256((__m256i const *)text);

        // 1-byte indicators & matches
        __m256i n_cmp = _mm256_cmpeq_epi8(text_u8x32, n_u8x32);
        __m256i v_cmp = _mm256_cmpeq_epi8(text_u8x32, v_u8x32);
        __m256i f_cmp = _mm256_cmpeq_epi8(text_u8x32, f_u8x32);
        __m256i r_cmp = _mm256_cmpeq_epi8(text_u8x32, r_u8x32);

        sz_u32_t n_mask = (sz_u32_t)_mm256_movemask_epi8(n_cmp);
        sz_u32_t v_mask = (sz_u32_t)_mm256_movemask_epi8(v_cmp);
        sz_u32_t f_mask = (sz_u32_t)_mm256_movemask_epi8(f_cmp);
        sz_u32_t r_mask = (sz_u32_t)_mm256_movemask_epi8(r_cmp) & 0x7FFFFFFF; // Ignore last byte
        sz_u32_t one_byte_mask = n_mask | v_mask | f_mask | r_mask;

        // 2-byte indicators
        __m256i x_c2_cmp = _mm256_cmpeq_epi8(text_u8x32, x_c2_u8x32);
        __m256i x_85_cmp = _mm256_cmpeq_epi8(text_u8x32, x_85_u8x32);
        __m256i x_e2_cmp = _mm256_cmpeq_epi8(text_u8x32, x_e2_u8x32);
        __m256i x_80_cmp = _mm256_cmpeq_epi8(text_u8x32, x_80_u8x32);
        __m256i x_a8_cmp = _mm256_cmpeq_epi8(text_u8x32, x_a8_u8x32);
        __m256i x_a9_cmp = _mm256_cmpeq_epi8(text_u8x32, x_a9_u8x32);

        sz_u32_t x_c2_mask = (sz_u32_t)_mm256_movemask_epi8(x_c2_cmp);
        sz_u32_t x_85_mask = (sz_u32_t)_mm256_movemask_epi8(x_85_cmp);
        sz_u32_t x_e2_mask = (sz_u32_t)_mm256_movemask_epi8(x_e2_cmp);
        sz_u32_t x_80_mask = (sz_u32_t)_mm256_movemask_epi8(x_80_cmp);
        sz_u32_t x_a8_mask = (sz_u32_t)_mm256_movemask_epi8(x_a8_cmp);
        sz_u32_t x_a9_mask = (sz_u32_t)_mm256_movemask_epi8(x_a9_cmp);

        // 2-byte matches
        sz_u32_t rn_mask = r_mask & (n_mask >> 1);
        sz_u32_t x_c285_mask = x_c2_mask & (x_85_mask >> 1);
        sz_u32_t two_byte_mask = rn_mask | x_c285_mask;

        // 3-byte matches
        sz_u32_t x_e280_mask = x_e2_mask & (x_80_mask >> 1);
        sz_u32_t x_e280a8_mask = x_e280_mask & (x_a8_mask >> 2);
        sz_u32_t x_e280a9_mask = x_e280_mask & (x_a9_mask >> 2);
        sz_u32_t three_byte_mask = x_e280a8_mask | x_e280a9_mask;

        // Find the earliest match regardless of length
        sz_u32_t combined_mask = one_byte_mask | two_byte_mask | three_byte_mask;
        if (combined_mask) {
            int first_offset = sz_u32_ctz(combined_mask);
            sz_u32_t first_match_mask = (sz_u32_t)(1) << first_offset;

            // Determine matched length
            sz_size_t length_value = 1;
            length_value += (first_match_mask & (two_byte_mask | three_byte_mask)) != 0;
            length_value += (first_match_mask & (three_byte_mask)) != 0;
            *matched_length = length_value;
            return text + first_offset;
        }
        else {
            text += 30;
            length -= 30;
        }
    }

    return sz_utf8_find_newline_serial(text, length, matched_length);
}

SZ_PUBLIC sz_cptr_t sz_utf8_find_whitespace_haswell(sz_cptr_t text, sz_size_t length, sz_size_t *matched_length) {

    // We need to check if the ASCII chars in [9,13] (same as '\t', '\n', '\v', '\f', '\r') are present.
    // There is also the canonical space ' ' (0x20).
    __m256i x_20_u8x32 = _mm256_set1_epi8(' ');

    // We also need to match the 2-byte characters 0xC285 (NEL) and 0xC2A0 (NBSP)
    __m256i x_c2_u8x32 = _mm256_set1_epi8('\xC2');
    __m256i x_85_u8x32 = _mm256_set1_epi8('\x85');
    __m256i x_a0_u8x32 = _mm256_set1_epi8('\xA0');

    // 3-byte character prefixes and suffixes
    __m256i x_e1_u8x32 = _mm256_set1_epi8('\xE1');
    __m256i x_e2_u8x32 = _mm256_set1_epi8('\xE2');
    __m256i x_e3_u8x32 = _mm256_set1_epi8('\xE3');
    __m256i x_9a_u8x32 = _mm256_set1_epi8('\x9A');
    __m256i x_80_u8x32 = _mm256_set1_epi8('\x80');
    __m256i x_81_u8x32 = _mm256_set1_epi8('\x81');
    __m256i x_a8_u8x32 = _mm256_set1_epi8('\xA8');
    __m256i x_a9_u8x32 = _mm256_set1_epi8('\xA9');
    __m256i x_af_u8x32 = _mm256_set1_epi8('\xAF');
    __m256i x_9f_u8x32 = _mm256_set1_epi8('\x9F');

    // We check 32 bytes of data at once, but only step forward by 30 bytes for split-register matches.
    while (length >= 32) {
        __m256i text_u8x32 = _mm256_loadu_si256((__m256i const *)text);

        // 1-byte indicators & matches
        // Range [9,13] covers \t, \n, \v, \f, \r
        __m256i x_20_cmp = _mm256_cmpeq_epi8(text_u8x32, x_20_u8x32);
        __m256i t_cmp = _mm256_cmpgt_epi8(text_u8x32, _mm256_set1_epi8((char)0x08)); // >= '\t' (0x09)
        __m256i r_cmp = _mm256_cmpgt_epi8(_mm256_set1_epi8((char)0x0E), text_u8x32); // <= '\r' (0x0D)
        __m256i tr_range = _mm256_and_si256(t_cmp, r_cmp);
        __m256i one_byte_cmp = _mm256_or_si256(x_20_cmp, tr_range);

        sz_u32_t one_byte_mask = (sz_u32_t)_mm256_movemask_epi8(one_byte_cmp);

        // 2-byte and 3-byte prefix indicators
        __m256i x_c2_cmp = _mm256_cmpeq_epi8(text_u8x32, x_c2_u8x32);
        __m256i x_e1_cmp = _mm256_cmpeq_epi8(text_u8x32, x_e1_u8x32);
        __m256i x_e2_cmp = _mm256_cmpeq_epi8(text_u8x32, x_e2_u8x32);
        __m256i x_e3_cmp = _mm256_cmpeq_epi8(text_u8x32, x_e3_u8x32);

        sz_u32_t x_c2_mask = (sz_u32_t)_mm256_movemask_epi8(x_c2_cmp) & 0x7FFFFFFF;
        sz_u32_t x_e1_mask = (sz_u32_t)_mm256_movemask_epi8(x_e1_cmp) & 0x3FFFFFFF;
        sz_u32_t x_e2_mask = (sz_u32_t)_mm256_movemask_epi8(x_e2_cmp) & 0x3FFFFFFF;
        sz_u32_t x_e3_mask = (sz_u32_t)_mm256_movemask_epi8(x_e3_cmp) & 0x3FFFFFFF;
        sz_u32_t prefix_byte_mask = x_c2_mask | x_e1_mask | x_e2_mask | x_e3_mask;

        // Check for fast path: one-byte match before any prefix
        if (one_byte_mask) {
            if (prefix_byte_mask) {
                int first_one_byte_offset = sz_u32_ctz(one_byte_mask);
                int first_prefix_offset = sz_u32_ctz(prefix_byte_mask);
                if (first_one_byte_offset < first_prefix_offset) {
                    *matched_length = 1;
                    return text + first_one_byte_offset;
                }
            }
            else {
                int first_one_byte_offset = sz_u32_ctz(one_byte_mask);
                *matched_length = 1;
                return text + first_one_byte_offset;
            }
        }

        // 2-byte suffixes
        __m256i x_85_cmp = _mm256_cmpeq_epi8(text_u8x32, x_85_u8x32);
        __m256i x_a0_cmp = _mm256_cmpeq_epi8(text_u8x32, x_a0_u8x32);
        sz_u32_t x_85_mask = (sz_u32_t)_mm256_movemask_epi8(x_85_cmp);
        sz_u32_t x_a0_mask = (sz_u32_t)_mm256_movemask_epi8(x_a0_cmp);

        sz_u32_t x_c285_mask = x_c2_mask & (x_85_mask >> 1); // U+0085 NEL
        sz_u32_t x_c2a0_mask = x_c2_mask & (x_a0_mask >> 1); // U+00A0 NBSP
        sz_u32_t two_byte_mask = x_c285_mask | x_c2a0_mask;

        // 3-byte suffixes
        __m256i x_9a_cmp = _mm256_cmpeq_epi8(text_u8x32, x_9a_u8x32);
        __m256i x_80_cmp = _mm256_cmpeq_epi8(text_u8x32, x_80_u8x32);
        __m256i x_81_cmp = _mm256_cmpeq_epi8(text_u8x32, x_81_u8x32);
        __m256i x_80_ge_cmp = sz_mm256_cmpge_epu8_(text_u8x32, x_80_u8x32);                   // >= 0x80
        __m256i x_8d_le_cmp = sz_mm256_cmpge_epu8_(_mm256_set1_epi8((char)0x8D), text_u8x32); // <= 0x8D
        __m256i x_8d_range = _mm256_and_si256(x_80_ge_cmp, x_8d_le_cmp);
        __m256i x_a8_cmp = _mm256_cmpeq_epi8(text_u8x32, x_a8_u8x32);
        __m256i x_a9_cmp = _mm256_cmpeq_epi8(text_u8x32, x_a9_u8x32);
        __m256i x_af_cmp = _mm256_cmpeq_epi8(text_u8x32, x_af_u8x32);
        __m256i x_9f_cmp = _mm256_cmpeq_epi8(text_u8x32, x_9f_u8x32);

        sz_u32_t x_9a_mask = (sz_u32_t)_mm256_movemask_epi8(x_9a_cmp);
        sz_u32_t x_80_mask = (sz_u32_t)_mm256_movemask_epi8(x_80_cmp);
        sz_u32_t x_81_mask = (sz_u32_t)_mm256_movemask_epi8(x_81_cmp);
        sz_u32_t x_8d_range_mask = (sz_u32_t)_mm256_movemask_epi8(x_8d_range);
        sz_u32_t x_a8_mask = (sz_u32_t)_mm256_movemask_epi8(x_a8_cmp);
        sz_u32_t x_a9_mask = (sz_u32_t)_mm256_movemask_epi8(x_a9_cmp);
        sz_u32_t x_af_mask = (sz_u32_t)_mm256_movemask_epi8(x_af_cmp);
        sz_u32_t x_9f_mask = (sz_u32_t)_mm256_movemask_epi8(x_9f_cmp);

        // 3-byte matches
        sz_u32_t ogham_mask = x_e1_mask & (x_9a_mask >> 1) & (x_80_mask >> 2);            // E1 9A 80
        sz_u32_t range_e280_mask = x_e2_mask & (x_80_mask >> 1) & (x_8d_range_mask >> 2); // E2 80 [80-8D]
        sz_u32_t line_mask = x_e2_mask & (x_80_mask >> 1) & (x_a8_mask >> 2);             // E2 80 A8
        sz_u32_t paragraph_mask = x_e2_mask & (x_80_mask >> 1) & (x_a9_mask >> 2);        // E2 80 A9
        sz_u32_t nnbsp_mask = x_e2_mask & (x_80_mask >> 1) & (x_af_mask >> 2);            // E2 80 AF
        sz_u32_t mmsp_mask = x_e2_mask & (x_81_mask >> 1) & (x_9f_mask >> 2);             // E2 81 9F
        sz_u32_t ideographic_mask = x_e3_mask & (x_80_mask >> 1) & (x_80_mask >> 2);      // E3 80 80
        sz_u32_t three_byte_mask = ogham_mask | range_e280_mask | nnbsp_mask | mmsp_mask | line_mask | paragraph_mask |
                                   ideographic_mask;

        // Find the earliest match regardless of length
        sz_u32_t combined_mask = one_byte_mask | two_byte_mask | three_byte_mask;
        if (combined_mask) {
            int first_offset = sz_u32_ctz(combined_mask);
            sz_u32_t first_match_mask = (sz_u32_t)(1) << first_offset;

            // Determine matched length
            sz_size_t length_value = 1;
            length_value += (first_match_mask & (two_byte_mask | three_byte_mask)) != 0;
            length_value += (first_match_mask & (three_byte_mask)) != 0;
            *matched_length = length_value;
            return text + first_offset;
        }
        else {
            text += 30;
            length -= 30;
        }
    }

    return sz_utf8_find_whitespace_serial(text, length, matched_length);
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

        // Check if we've reached the terminal part of our search
        if (n < start_byte_count) {
            // PDEP directly gives us the nth set bit position
            // Example: _pdep_u32(0b10, 0b00010101) = 0b00000100
            sz_u32_t deposited_bits = _pdep_u32((sz_u32_t)1 << n, start_byte_mask);
            int byte_offset = sz_u32_ctz(deposited_bits);
            return (sz_cptr_t)(text_u8 + byte_offset);
        }
        // Jump to the next block
        else {
            n -= start_byte_count;
            text_u8 += 32;
            length -= 32;
        }
    }

    // Process remaining bytes with serial
    return sz_utf8_find_nth_serial((sz_cptr_t)text_u8, length, n);
}

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif

#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("avx2,bmi,bmi2,popcnt"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("avx2,bmi,bmi2,popcnt")
#endif

/*  UAX-29 word boundary detection (vectorized).
 *
 *  The serial reference walks the text codepoint-by-codepoint and, at each candidate position,
 *  evaluates `sz_utf8_is_word_boundary_serial`. The dominant runtime cost for typical (ASCII / Latin)
 *  text is the per-codepoint Word_Break property classification.
 *
 *  The Haswell backend keeps the same outer walk but accelerates it in two ways while remaining
 *  byte-exact versus the serial reference:
 *
 *   1) Vectorized property classification. A 32-byte window is classified with `_mm256_shuffle_epi8`.
 *      ASCII bytes (<0x80) are mapped to their Word_Break property through an 8-row nibble LUT (one
 *      `_mm256_shuffle_epi8` per high-nibble row, selected by an equality mask). The full ASCII
 *      property table is an 8x16 grid, so a single shuffle is insufficient; eight row-shuffles
 *      reproduce it exactly. Continuation bytes (0x80-0xBF) and non-ASCII lead bytes are tagged with
 *      a sentinel so the scalar fallback handles them. The result is cached into a small stack window
 *      of per-byte properties.
 *
 *   2) Vectorized local boundary decision. For positions whose previous and following effective
 *      properties are both "plain" (Other/CR/LF/Newline/ALetter/Hebrew/Numeric/Katakana/ExtendNumLet)
 *      the boundary is fully determined by the immediate pair via rules WB3/3a/3b/5/8/9/10/13/13a/13b
 *      (encoded in `sz_wb_pair_decision_`). These are resolved without any look-around.
 *
 *  Every position that could be governed by a stateful rule -- WB4 (Extend/Format/ZWJ skipping),
 *  WB6/WB7 (MidLetter look-around), WB7a, WB11/WB12 (MidNum look-around), WB15/WB16 (Regional
 *  Indicator parity), the emoji-ZWJ rule WB3c, or any non-ASCII / window-edge position where local
 *  context is insufficient -- defers to `sz_utf8_is_word_boundary_serial`, which re-reads the full
 *  `text`/`length` and is therefore always exact. Sub-rules left to serial: WB3c, WB4, WB6, WB7,
 *  WB7a, WB11, WB12, WB15, WB16 (and all non-BMP / SMP property cases).
 */

/* Per-byte Word_Break property of ASCII codepoints, laid out as eight 16-entry rows indexed by the
 * low nibble; row R serves high nibble R (covering bytes 0x00-0x7F). */
SZ_INTERNAL __m256i sz_wb_classify_ascii_haswell_(__m256i bytes) {
    // Eight rows of the ASCII Word_Break property table (high nibble -> 16 low-nibble entries).
    // clang-format off
    __m256i row0 = _mm256_setr_epi8(0,0,0,0,0,0,0,0,0,0,2,3,3,1,0,0, 0,0,0,0,0,0,0,0,0,0,2,3,3,1,0,0);
    __m256i row1 = _mm256_setzero_si256();
    __m256i row2 = _mm256_setr_epi8(0,0,15,0,0,0,0,15,0,0,0,0,14,0,15,0, 0,0,15,0,0,0,0,15,0,0,0,0,14,0,15,0);
    __m256i row3 = _mm256_setr_epi8(10,10,10,10,10,10,10,10,10,10,13,14,0,0,0,0, 10,10,10,10,10,10,10,10,10,10,13,14,0,0,0,0);
    __m256i row4 = _mm256_setr_epi8(0,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8, 0,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8);
    __m256i row5 = _mm256_setr_epi8(8,8,8,8,8,8,8,8,8,8,8,0,0,0,0,12, 8,8,8,8,8,8,8,8,8,8,8,0,0,0,0,12);
    __m256i row6 = _mm256_setr_epi8(0,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8, 0,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8);
    __m256i row7 = _mm256_setr_epi8(8,8,8,8,8,8,8,8,8,8,8,0,0,0,0,0, 8,8,8,8,8,8,8,8,8,8,8,0,0,0,0,0);
    // clang-format on

    __m256i lo_nibble = _mm256_and_si256(bytes, _mm256_set1_epi8(0x0F));
    __m256i hi_nibble = _mm256_and_si256(_mm256_srli_epi16(bytes, 4), _mm256_set1_epi8(0x0F));

    __m256i result = _mm256_setzero_si256();
    __m256i shuf0 = _mm256_shuffle_epi8(row0, lo_nibble);
    __m256i shuf1 = _mm256_shuffle_epi8(row1, lo_nibble);
    __m256i shuf2 = _mm256_shuffle_epi8(row2, lo_nibble);
    __m256i shuf3 = _mm256_shuffle_epi8(row3, lo_nibble);
    __m256i shuf4 = _mm256_shuffle_epi8(row4, lo_nibble);
    __m256i shuf5 = _mm256_shuffle_epi8(row5, lo_nibble);
    __m256i shuf6 = _mm256_shuffle_epi8(row6, lo_nibble);
    __m256i shuf7 = _mm256_shuffle_epi8(row7, lo_nibble);

    result = _mm256_blendv_epi8(result, shuf0, _mm256_cmpeq_epi8(hi_nibble, _mm256_set1_epi8(0)));
    result = _mm256_blendv_epi8(result, shuf1, _mm256_cmpeq_epi8(hi_nibble, _mm256_set1_epi8(1)));
    result = _mm256_blendv_epi8(result, shuf2, _mm256_cmpeq_epi8(hi_nibble, _mm256_set1_epi8(2)));
    result = _mm256_blendv_epi8(result, shuf3, _mm256_cmpeq_epi8(hi_nibble, _mm256_set1_epi8(3)));
    result = _mm256_blendv_epi8(result, shuf4, _mm256_cmpeq_epi8(hi_nibble, _mm256_set1_epi8(4)));
    result = _mm256_blendv_epi8(result, shuf5, _mm256_cmpeq_epi8(hi_nibble, _mm256_set1_epi8(5)));
    result = _mm256_blendv_epi8(result, shuf6, _mm256_cmpeq_epi8(hi_nibble, _mm256_set1_epi8(6)));
    result = _mm256_blendv_epi8(result, shuf7, _mm256_cmpeq_epi8(hi_nibble, _mm256_set1_epi8(7)));
    return result;
}

/* Pairwise boundary decision for "plain" Word_Break property pairs.
 * Index: (prev_prop << 4) | after_prop. Value: 0 = no break, 1 = break, 2 = defer to serial. */
static const sz_u8_t sz_wb_pair_decision_[256] = {
    1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 1, 1, 1, 2, 2, 2, //
    1, 1, 0, 1, 2, 2, 2, 2, 1, 1, 1, 1, 1, 2, 2, 2, //
    1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 1, 1, 1, 2, 2, 2, //
    1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 1, 1, 1, 2, 2, 2, //
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, //
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, //
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, //
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, //
    1, 1, 1, 1, 2, 2, 2, 2, 0, 0, 0, 1, 0, 2, 2, 2, //
    1, 1, 1, 1, 2, 2, 2, 2, 0, 0, 0, 1, 0, 2, 2, 2, //
    1, 1, 1, 1, 2, 2, 2, 2, 0, 0, 0, 1, 0, 2, 2, 2, //
    1, 1, 1, 1, 2, 2, 2, 2, 1, 1, 1, 0, 0, 2, 2, 2, //
    1, 1, 1, 1, 2, 2, 2, 2, 0, 0, 0, 0, 0, 2, 2, 2, //
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, //
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, //
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, //
};

/* Sentinel marking a byte whose property must be resolved by the scalar fallback
 * (continuation byte or non-ASCII lead byte). */
enum { sz_wb_prop_sentinel_k = 0x80 };
#define SZ_WB_PROP_SENTINEL_ ((sz_u8_t)sz_wb_prop_sentinel_k)

/* Classify a window of up to 32 bytes into per-byte Word_Break properties.
 * ASCII bytes get their exact property; every other byte gets SZ_WB_PROP_SENTINEL_. */
SZ_INTERNAL void sz_wb_classify_window_haswell_(sz_cptr_t window, sz_size_t valid, sz_u8_t *props_out) {
    __m256i bytes = _mm256_loadu_si256((__m256i const *)window);
    __m256i is_ascii = _mm256_cmpgt_epi8(bytes, _mm256_set1_epi8(-1)); // top bit clear -> non-negative
    __m256i ascii_props = sz_wb_classify_ascii_haswell_(bytes);
    __m256i sentinel = _mm256_set1_epi8((char)SZ_WB_PROP_SENTINEL_);
    __m256i classified = _mm256_blendv_epi8(sentinel, ascii_props, is_ascii);
    _mm256_storeu_si256((__m256i *)props_out, classified);
    // Mark trailing (invalid) lanes as sentinel so they are never trusted.
    for (sz_size_t i = valid; i < 32; ++i) props_out[i] = SZ_WB_PROP_SENTINEL_;
}

SZ_PUBLIC sz_cptr_t sz_utf8_word_find_boundary_haswell(sz_cptr_t text, sz_size_t length, sz_size_t *boundary_width) {
    if (length == 0) {
        if (boundary_width) *boundary_width = 0;
        return text;
    }

    sz_u8_t const *text_u8 = (sz_u8_t const *)text;
    sz_size_t position = 0;
    // Skip first codepoint (position 0 is always a boundary), matching the serial reference.
    position += sz_utf8_char_length_(text_u8[0]);

    // Window of classified properties covering [base, base+32).
    sz_u8_t props[32];
    sz_size_t base = (sz_size_t)-1; // No window loaded yet.

    while (position < length) {
        // (Re)load the property window if the byte at `position` and its immediate predecessor are covered.
        // We need props[position-base] (after) and props[position-1-base] (prev byte) to be valid.
        if (base == (sz_size_t)-1 || position < base + 1 || position >= base + 32) {
            base = (position > 0) ? (position - 1) : 0;
            if (base + 32 <= length) { sz_wb_classify_window_haswell_(text + base, 32, props); }
            else {
                sz_size_t valid = length - base;
                sz_u8_t window_scratch[32];
                for (sz_size_t i = 0; i < valid; ++i) window_scratch[i] = text_u8[base + i];
                for (sz_size_t i = valid; i < 32; ++i) window_scratch[i] = 0;
                sz_wb_classify_window_haswell_((sz_cptr_t)window_scratch, valid, props);
            }
        }

        sz_u8_t after_prop = props[position - base];
        sz_u8_t prev_byte_prop = props[position - 1 - base];
        // The byte before `position` must itself be a lead byte (not a continuation) for `prev_byte_prop`
        // to be the immediate previous codepoint's property. We approximate the serial `prev_prop`
        // (which skips ignorables) only when both sides are plain ASCII; otherwise defer.
        sz_u8_t decision = 2;
        if (after_prop != SZ_WB_PROP_SENTINEL_ && prev_byte_prop != SZ_WB_PROP_SENTINEL_) {
            decision = sz_wb_pair_decision_[((sz_size_t)prev_byte_prop << 4) | after_prop];
        }

        sz_bool_t is_boundary;
        if (decision != 2) { is_boundary = (sz_bool_t)decision; }
        else { is_boundary = sz_utf8_is_word_boundary_serial(text, length, position); }

        if (is_boundary) {
            if (boundary_width) *boundary_width = position;
            return text + position;
        }
        position += sz_utf8_char_length_(text_u8[position]);
    }

    if (boundary_width) *boundary_width = length;
    return text + length;
}

SZ_PUBLIC sz_cptr_t sz_utf8_word_rfind_boundary_haswell(sz_cptr_t text, sz_size_t length, sz_size_t *boundary_width) {
    if (length == 0) {
        if (boundary_width) *boundary_width = 0;
        return text;
    }

    sz_u8_t const *text_u8 = (sz_u8_t const *)text;
    sz_size_t position = length;
    // Move back one codepoint (position length is always a boundary).
    position--;
    while (position > 0 && (text_u8[position] & 0xC0) == 0x80) position--;

    sz_u8_t props[32];
    sz_size_t base = (sz_size_t)-1;

    while (position > 0) {
        if (base == (sz_size_t)-1 || position < base + 1 || position >= base + 32) {
            base = position - 1;
            if (base + 32 <= length) { sz_wb_classify_window_haswell_(text + base, 32, props); }
            else {
                sz_size_t valid = length - base;
                sz_u8_t window_scratch[32];
                for (sz_size_t i = 0; i < valid; ++i) window_scratch[i] = text_u8[base + i];
                for (sz_size_t i = valid; i < 32; ++i) window_scratch[i] = 0;
                sz_wb_classify_window_haswell_((sz_cptr_t)window_scratch, valid, props);
            }
        }

        sz_u8_t after_prop = props[position - base];
        sz_u8_t prev_byte_prop = props[position - 1 - base];
        sz_u8_t decision = 2;
        if (after_prop != SZ_WB_PROP_SENTINEL_ && prev_byte_prop != SZ_WB_PROP_SENTINEL_) {
            decision = sz_wb_pair_decision_[((sz_size_t)prev_byte_prop << 4) | after_prop];
        }

        sz_bool_t is_boundary;
        if (decision != 2) { is_boundary = (sz_bool_t)decision; }
        else { is_boundary = sz_utf8_is_word_boundary_serial(text, length, position); }

        if (is_boundary) {
            if (boundary_width) *boundary_width = length - position;
            return text + position;
        }
        position--;
        while (position > 0 && (text_u8[position] & 0xC0) == 0x80) position--;
    }

    if (boundary_width) *boundary_width = length;
    return text;
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

#endif // STRINGZILLA_UTF8_ITERATE_HASWELL_H_
