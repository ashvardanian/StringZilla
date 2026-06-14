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
SZ_INTERNAL __m256i sz_mm256_cmpge_epu8_haswell_(__m256i a, __m256i b) {
    return _mm256_cmpeq_epi8(_mm256_max_epu8(a, b), a);
}

SZ_PUBLIC sz_cptr_t sz_utf8_find_newline_haswell(sz_cptr_t text, sz_size_t length, sz_size_t *matched_length) {

    // We need to check if the ASCII chars in [10,13] (same as '\n', '\v', '\f', '\r') are present.
    // The last one - '\r' - needs special handling to differentiate between "\r" and "\r\n".
    __m256i newline_u8x32 = _mm256_set1_epi8('\n');
    __m256i vertical_tab_u8x32 = _mm256_set1_epi8('\v');
    __m256i form_feed_u8x32 = _mm256_set1_epi8('\f');
    __m256i carriage_return_u8x32 = _mm256_set1_epi8('\r');

    // We also need to match the 2-byte newline character 0xC285 (NEL),
    // as well as the 3-byte characters 0xE280A8 (PS) and 0xE280A9 (LS).
    __m256i lead_c2_u8x32 = _mm256_set1_epi8('\xC2');
    __m256i x_85_u8x32 = _mm256_set1_epi8('\x85');
    __m256i lead_e2_u8x32 = _mm256_set1_epi8('\xE2');
    __m256i byte_80_u8x32 = _mm256_set1_epi8('\x80');
    __m256i x_a8_u8x32 = _mm256_set1_epi8('\xA8');
    __m256i x_a9_u8x32 = _mm256_set1_epi8('\xA9');

    // We check 32 bytes of data at once, but only step forward by 30 bytes for split-register matches.
    while (length >= 32) {
        __m256i text_u8x32 = _mm256_loadu_si256((__m256i const *)text);

        // 1-byte indicators & matches
        __m256i newline_cmp = _mm256_cmpeq_epi8(text_u8x32, newline_u8x32);
        __m256i vertical_tab_cmp = _mm256_cmpeq_epi8(text_u8x32, vertical_tab_u8x32);
        __m256i form_feed_cmp = _mm256_cmpeq_epi8(text_u8x32, form_feed_u8x32);
        __m256i carriage_return_cmp = _mm256_cmpeq_epi8(text_u8x32, carriage_return_u8x32);

        sz_u32_t newline_mask = (sz_u32_t)_mm256_movemask_epi8(newline_cmp);
        sz_u32_t vertical_tab_mask = (sz_u32_t)_mm256_movemask_epi8(vertical_tab_cmp);
        sz_u32_t form_feed_mask = (sz_u32_t)_mm256_movemask_epi8(form_feed_cmp);
        sz_u32_t carriage_return_mask = (sz_u32_t)_mm256_movemask_epi8(carriage_return_cmp) &
                                        0x7FFFFFFF; // Ignore last byte
        sz_u32_t one_byte_mask = newline_mask | vertical_tab_mask | form_feed_mask | carriage_return_mask;

        // 2-byte indicators
        __m256i lead_c2_cmp = _mm256_cmpeq_epi8(text_u8x32, lead_c2_u8x32);
        __m256i x_85_cmp = _mm256_cmpeq_epi8(text_u8x32, x_85_u8x32);
        __m256i lead_e2_cmp = _mm256_cmpeq_epi8(text_u8x32, lead_e2_u8x32);
        __m256i byte_80_cmp = _mm256_cmpeq_epi8(text_u8x32, byte_80_u8x32);
        __m256i x_a8_cmp = _mm256_cmpeq_epi8(text_u8x32, x_a8_u8x32);
        __m256i x_a9_cmp = _mm256_cmpeq_epi8(text_u8x32, x_a9_u8x32);

        sz_u32_t lead_c2_mask = (sz_u32_t)_mm256_movemask_epi8(lead_c2_cmp);
        sz_u32_t x_85_mask = (sz_u32_t)_mm256_movemask_epi8(x_85_cmp);
        sz_u32_t lead_e2_mask = (sz_u32_t)_mm256_movemask_epi8(lead_e2_cmp);
        sz_u32_t byte_80_mask = (sz_u32_t)_mm256_movemask_epi8(byte_80_cmp);
        sz_u32_t x_a8_mask = (sz_u32_t)_mm256_movemask_epi8(x_a8_cmp);
        sz_u32_t x_a9_mask = (sz_u32_t)_mm256_movemask_epi8(x_a9_cmp);

        // 2-byte matches
        sz_u32_t rn_mask = carriage_return_mask & (newline_mask >> 1);
        sz_u32_t nel_c2_85_mask = lead_c2_mask & (x_85_mask >> 1);
        sz_u32_t two_byte_mask = rn_mask | nel_c2_85_mask;

        // 3-byte matches
        sz_u32_t lead_e280_mask = lead_e2_mask & (byte_80_mask >> 1);
        sz_u32_t x_e280a8_mask = lead_e280_mask & (x_a8_mask >> 2);
        sz_u32_t x_e280a9_mask = lead_e280_mask & (x_a9_mask >> 2);
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
    __m256i lead_c2_u8x32 = _mm256_set1_epi8('\xC2');
    __m256i x_85_u8x32 = _mm256_set1_epi8('\x85');
    __m256i x_a0_u8x32 = _mm256_set1_epi8('\xA0');

    // 3-byte character prefixes and suffixes
    __m256i x_e1_u8x32 = _mm256_set1_epi8('\xE1');
    __m256i lead_e2_u8x32 = _mm256_set1_epi8('\xE2');
    __m256i x_e3_u8x32 = _mm256_set1_epi8('\xE3');
    __m256i x_9a_u8x32 = _mm256_set1_epi8('\x9A');
    __m256i byte_80_u8x32 = _mm256_set1_epi8('\x80');
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
        __m256i tab_lower_bound_cmp = _mm256_cmpgt_epi8(text_u8x32, _mm256_set1_epi8((char)0x08)); // >= '\t' (0x09)
        __m256i carriage_return_upper_bound_cmp = _mm256_cmpgt_epi8(_mm256_set1_epi8((char)0x0E),
                                                                    text_u8x32); // <= '\r' (0x0D)
        __m256i tab_carriage_return_range = _mm256_and_si256(tab_lower_bound_cmp, carriage_return_upper_bound_cmp);
        __m256i one_byte_cmp = _mm256_or_si256(x_20_cmp, tab_carriage_return_range);

        sz_u32_t one_byte_mask = (sz_u32_t)_mm256_movemask_epi8(one_byte_cmp);

        // 2-byte and 3-byte prefix indicators
        __m256i lead_c2_cmp = _mm256_cmpeq_epi8(text_u8x32, lead_c2_u8x32);
        __m256i x_e1_cmp = _mm256_cmpeq_epi8(text_u8x32, x_e1_u8x32);
        __m256i lead_e2_cmp = _mm256_cmpeq_epi8(text_u8x32, lead_e2_u8x32);
        __m256i x_e3_cmp = _mm256_cmpeq_epi8(text_u8x32, x_e3_u8x32);

        sz_u32_t lead_c2_mask = (sz_u32_t)_mm256_movemask_epi8(lead_c2_cmp) & 0x7FFFFFFF;
        sz_u32_t x_e1_mask = (sz_u32_t)_mm256_movemask_epi8(x_e1_cmp) & 0x3FFFFFFF;
        sz_u32_t lead_e2_mask = (sz_u32_t)_mm256_movemask_epi8(lead_e2_cmp) & 0x3FFFFFFF;
        sz_u32_t x_e3_mask = (sz_u32_t)_mm256_movemask_epi8(x_e3_cmp) & 0x3FFFFFFF;
        sz_u32_t prefix_byte_mask = lead_c2_mask | x_e1_mask | lead_e2_mask | x_e3_mask;

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

        sz_u32_t nel_c2_85_mask = lead_c2_mask & (x_85_mask >> 1); // U+0085 NEL
        sz_u32_t x_c2a0_mask = lead_c2_mask & (x_a0_mask >> 1);    // U+00A0 NBSP
        sz_u32_t two_byte_mask = nel_c2_85_mask | x_c2a0_mask;

        // 3-byte suffixes
        __m256i x_9a_cmp = _mm256_cmpeq_epi8(text_u8x32, x_9a_u8x32);
        __m256i byte_80_cmp = _mm256_cmpeq_epi8(text_u8x32, byte_80_u8x32);
        __m256i x_81_cmp = _mm256_cmpeq_epi8(text_u8x32, x_81_u8x32);
        __m256i x_80_ge_cmp = sz_mm256_cmpge_epu8_haswell_(text_u8x32, byte_80_u8x32);                // >= 0x80
        __m256i x_8d_le_cmp = sz_mm256_cmpge_epu8_haswell_(_mm256_set1_epi8((char)0x8D), text_u8x32); // <= 0x8D
        __m256i x_8d_range = _mm256_and_si256(x_80_ge_cmp, x_8d_le_cmp);
        __m256i x_a8_cmp = _mm256_cmpeq_epi8(text_u8x32, x_a8_u8x32);
        __m256i x_a9_cmp = _mm256_cmpeq_epi8(text_u8x32, x_a9_u8x32);
        __m256i x_af_cmp = _mm256_cmpeq_epi8(text_u8x32, x_af_u8x32);
        __m256i x_9f_cmp = _mm256_cmpeq_epi8(text_u8x32, x_9f_u8x32);

        sz_u32_t x_9a_mask = (sz_u32_t)_mm256_movemask_epi8(x_9a_cmp);
        sz_u32_t byte_80_mask = (sz_u32_t)_mm256_movemask_epi8(byte_80_cmp);
        sz_u32_t x_81_mask = (sz_u32_t)_mm256_movemask_epi8(x_81_cmp);
        sz_u32_t x_8d_range_mask = (sz_u32_t)_mm256_movemask_epi8(x_8d_range);
        sz_u32_t x_a8_mask = (sz_u32_t)_mm256_movemask_epi8(x_a8_cmp);
        sz_u32_t x_a9_mask = (sz_u32_t)_mm256_movemask_epi8(x_a9_cmp);
        sz_u32_t x_af_mask = (sz_u32_t)_mm256_movemask_epi8(x_af_cmp);
        sz_u32_t x_9f_mask = (sz_u32_t)_mm256_movemask_epi8(x_9f_cmp);

        // 3-byte matches
        sz_u32_t ogham_mask = x_e1_mask & (x_9a_mask >> 1) & (byte_80_mask >> 2);               // E1 9A 80
        sz_u32_t range_e280_mask = lead_e2_mask & (byte_80_mask >> 1) & (x_8d_range_mask >> 2); // E2 80 [80-8D]
        sz_u32_t line_mask = lead_e2_mask & (byte_80_mask >> 1) & (x_a8_mask >> 2);             // E2 80 A8
        sz_u32_t paragraph_mask = lead_e2_mask & (byte_80_mask >> 1) & (x_a9_mask >> 2);        // E2 80 A9
        sz_u32_t nnbsp_mask = lead_e2_mask & (byte_80_mask >> 1) & (x_af_mask >> 2);            // E2 80 AF
        sz_u32_t mmsp_mask = lead_e2_mask & (x_81_mask >> 1) & (x_9f_mask >> 2);                // E2 81 9F
        sz_u32_t ideographic_mask = x_e3_mask & (byte_80_mask >> 1) & (byte_80_mask >> 2);      // E3 80 80
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
 *      (encoded in `sz_utf8_word_break_pair_decision_`). These are resolved without any look-around.
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
SZ_INTERNAL __m256i sz_utf8_word_break_classify_ascii_haswell_(__m256i bytes) {
    // Eight rows of the ASCII Word_Break property table (high nibble → 16 low-nibble entries).
    __m256i row0 = _mm256_setr_epi8(                    //
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 3, 3, 1, 0, 0, //
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 3, 3, 1, 0, 0);
    __m256i row1 = _mm256_setzero_si256();
    __m256i row2 = _mm256_setr_epi8(                        //
        0, 0, 15, 0, 0, 0, 0, 15, 0, 0, 0, 0, 14, 0, 15, 0, //
        0, 0, 15, 0, 0, 0, 0, 15, 0, 0, 0, 0, 14, 0, 15, 0);
    __m256i row3 = _mm256_setr_epi8(                                //
        10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 13, 14, 0, 0, 0, 0, //
        10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 13, 14, 0, 0, 0, 0);
    __m256i row4 = _mm256_setr_epi8(                    //
        0, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, //
        0, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8);
    __m256i row5 = _mm256_setr_epi8(                     //
        8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 0, 0, 0, 0, 12, //
        8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 0, 0, 0, 0, 12);
    __m256i row6 = _mm256_setr_epi8(                    //
        0, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, //
        0, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8);
    __m256i row7 = _mm256_setr_epi8(                    //
        8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 0, 0, 0, 0, 0, //
        8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 0, 0, 0, 0, 0);

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

/* Per-class lane mask: bit i set where lane i has Word_Break property `value`. */
SZ_INTERNAL sz_u32_t sz_utf8_word_break_class_mask_haswell_(__m256i classes, int value) {
    return (sz_u32_t)_mm256_movemask_epi8(_mm256_cmpeq_epi8(classes, _mm256_set1_epi8((char)value)));
}

/* 32-bit "joined" (guaranteed non-boundary) mask for an all-ASCII 32-byte window: bit i set => the boundary
 * before lane i is suppressed by a UAX-29 no-break rule. ASCII has no Extend/Format/ZWJ/Regional_Indicator/
 * Hebrew/Katakana, so WB4 and WB15/16 never apply and WB6/7/11/12 reduce to neighbour bit-shifts (`<< 1` =
 * previous lane, `>> 1` = next, `<< 2` = two back). Exact for lanes whose i-2 and i+1 neighbours are in-window. */
SZ_INTERNAL sz_u32_t sz_utf8_word_break_join_mask_ascii_haswell_(__m256i classes) {
    // Reduce each class to a 32-lane bitmask and defer the rule logic to the shared portable routine; the
    // caller restricts the result to its trusted lane window, so the wider u64 math is harmless.
    return (sz_u32_t)sz_utf8_word_break_join_from_class_masks_(
        sz_utf8_word_break_class_mask_haswell_(classes, sz_tr29_word_break_aletter_k),
        sz_utf8_word_break_class_mask_haswell_(classes, sz_tr29_word_break_numeric_k),
        sz_utf8_word_break_class_mask_haswell_(classes, sz_tr29_word_break_extendnumlet_k),
        sz_utf8_word_break_class_mask_haswell_(classes, sz_tr29_word_break_midletter_k),
        sz_utf8_word_break_class_mask_haswell_(classes, sz_tr29_word_break_midnum_k),
        sz_utf8_word_break_class_mask_haswell_(classes, sz_tr29_word_break_mid_quotes_k),
        sz_utf8_word_break_class_mask_haswell_(classes, sz_tr29_word_break_cr_k),
        sz_utf8_word_break_class_mask_haswell_(classes, sz_tr29_word_break_lf_k));
}

SZ_PUBLIC sz_size_t sz_utf8_word_find_boundaries_haswell( //
    sz_cptr_t text, sz_size_t length,                     //
    sz_size_t *word_starts, sz_size_t *word_lengths,      //
    sz_size_t words_capacity, sz_size_t *bytes_consumed) {

    sz_size_t words = 0;
    if (length == 0 || words_capacity == 0) {
        if (bytes_consumed) *bytes_consumed = 0;
        return 0;
    }

    sz_u8_t const *text_u8 = (sz_u8_t const *)text;
    sz_size_t word_start = 0; // Start of the word currently being accumulated (always a boundary).
    // Skip first codepoint (position 0 is always a boundary), matching the serial reference.
    sz_size_t position = sz_utf8_codepoint_length_(text_u8[0]);

    while (position < length) {
        // Oracle-free fast path: a window [position-2, position+30) gives lanes [2,30] full +/-2 context, so an
        // all-ASCII window resolves boundaries at positions [position, position+28] directly from the mask.
        if (position >= 2 && position + 30 <= length) {
            __m256i window = _mm256_loadu_si256(
                (__m256i const *)(text_u8 + position - 2)); // lane j = byte position-2+j
            if (_mm256_movemask_epi8(window) == 0) {        // all ASCII
                __m256i classes = sz_utf8_word_break_classify_ascii_haswell_(window);
                sz_u32_t join = sz_utf8_word_break_join_mask_ascii_haswell_(classes);
                sz_u32_t boundary = (~join) & 0x7FFFFFFCu; // trusted lanes [2,30]
                while (boundary) {
                    sz_size_t boundary_position = position - 2 + (sz_size_t)sz_u32_ctz(boundary);
                    if (words == words_capacity) {
                        if (bytes_consumed) *bytes_consumed = word_start;
                        return words;
                    }
                    word_starts[words] = word_start;
                    word_lengths[words] = boundary_position - word_start;
                    ++words;
                    word_start = boundary_position;
                    boundary &= boundary - 1;
                }
                position += 29; // Resolved [position, position+28]; next unresolved boundary is at position+29.
                continue;
            }
        }
        // Scalar step (non-ASCII window, or near the leading/trailing edges).
        if (sz_utf8_is_word_boundary_serial(text, length, position)) {
            if (words == words_capacity) {
                if (bytes_consumed) *bytes_consumed = word_start;
                return words;
            }
            word_starts[words] = word_start;
            word_lengths[words] = position - word_start;
            ++words;
            word_start = position;
        }
        position += sz_utf8_codepoint_length_(text_u8[position]);
    }

    if (words == words_capacity) {
        if (bytes_consumed) *bytes_consumed = word_start;
        return words;
    }
    word_starts[words] = word_start;
    word_lengths[words] = length - word_start;
    ++words;
    if (bytes_consumed) *bytes_consumed = length;
    return words;
}

SZ_PUBLIC sz_size_t sz_utf8_word_rfind_boundaries_haswell( //
    sz_cptr_t text, sz_size_t length,                      //
    sz_size_t *word_starts, sz_size_t *word_lengths,       //
    sz_size_t words_capacity, sz_size_t *bytes_consumed) {

    sz_size_t words = 0;
    if (length == 0 || words_capacity == 0) {
        if (bytes_consumed) *bytes_consumed = length;
        return 0;
    }

    sz_u8_t const *text_u8 = (sz_u8_t const *)text;
    sz_size_t word_end = length; // End of the word currently being accumulated (always a boundary).
    sz_size_t position = length - 1;
    while (position > 0 && (text_u8[position] & 0xC0) == 0x80) position--;

    while (position > 0) {
        // Oracle-free fast path: a window [position-30, position+2) gives lanes [2,30] full +/-2 context,
        // resolving boundaries at positions [position-28, position]; emit them high-to-low.
        if (position >= 30 && position + 2 <= length) {
            sz_size_t base = position - 30; // lane j = byte base+j; trusted lanes [2,30] → [position-28, position]
            __m256i window = _mm256_loadu_si256((__m256i const *)(text_u8 + base));
            if (_mm256_movemask_epi8(window) == 0) {
                __m256i classes = sz_utf8_word_break_classify_ascii_haswell_(window);
                sz_u32_t join = sz_utf8_word_break_join_mask_ascii_haswell_(classes);
                sz_u32_t boundary = (~join) & 0x7FFFFFFCu; // trusted lanes [2,30]
                // Emit high-to-low: clear from the most-significant set bit downward.
                while (boundary) {
                    int lane = 31 - sz_u32_clz(boundary);
                    sz_size_t boundary_position = base + (sz_size_t)lane;
                    if (words == words_capacity) {
                        if (bytes_consumed) *bytes_consumed = word_end;
                        return words;
                    }
                    word_starts[words] = boundary_position;
                    word_lengths[words] = word_end - boundary_position;
                    ++words;
                    word_end = boundary_position;
                    boundary &= ~((sz_u32_t)1 << lane);
                }
                position = base + 1; // Resolved down to position-28; next unresolved boundary is at position-29.
                continue;
            }
        }
        if (sz_utf8_is_word_boundary_serial(text, length, position)) {
            if (words == words_capacity) {
                if (bytes_consumed) *bytes_consumed = word_end;
                return words;
            }
            word_starts[words] = position;
            word_lengths[words] = word_end - position;
            ++words;
            word_end = position;
        }
        position--;
        while (position > 0 && (text_u8[position] & 0xC0) == 0x80) position--;
    }

    if (words == words_capacity) {
        if (bytes_consumed) *bytes_consumed = word_end;
        return words;
    }
    word_starts[words] = 0;
    word_lengths[words] = word_end;
    ++words;
    if (bytes_consumed) *bytes_consumed = 0;
    return words;
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
