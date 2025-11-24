/**
 *  @brief  Hardware-accelerated UTF-8 text processing utilities.
 *  @file   utf8.h
 *  @author Ash Vardanian
 *
 *  Includes core APIs:
 *
 *  - `sz_utf8_count` - count UTF-8 characters in a string
 *  - `sz_utf8_find_nth` - skip to Nth UTF-8 character
 *  - `sz_utf8_find_newline` - skip to first newline (7 Unicode newline characters + CRLF)
 *  - `sz_utf8_find_whitespace` - skip to first whitespace (25 Unicode White_Space characters)
 *
 *  StringZilla focuses on analyzing character classes rather than UAX#14 (Line Breaking Algorithm) or UAX#29 (Text
 *  Segmentation) in full detail. It doesn't currently implement traditional "grapheme cluster", "word boundary",
 *  or "sentence boundary", but focuses on SIMD-optimized detection of newlines and whitespaces in UTF-8 strings.
 *
 *  StringZilla detects all of "White_Space" characters defined by Unicode, matching ICU's `u_isspace()` and
 *  Python's `str.isspace()`. It does NOT include U+001C-U+001F (FILE/GROUP/RECORD/UNIT SEPARATOR) unlike Java.
 *  These are data structure delimiters (used in USV format), not whitespace.
 *
 *  UTF-8 processing operates directly on UTF-8 encoded strings without decoding into UTF-32 or UTF-16 codepoints.
 *  SIMD operations check for matches at different granularities: every byte (1-byte chars), every 2 bytes,
 *  and every 3 bytes. 4-byte UTF-8 sequences are handled by the validation and skip functions.
 */
#ifndef STRINGZILLA_UTF8_H_
#define STRINGZILLA_UTF8_H_

#include "types.h"

#ifdef __cplusplus
extern "C" {
#endif

#pragma region Core API

/**
 *  @brief  Count the number of UTF-8 characters in a string.
 *
 *  The logic is to count the number of "continuation bytes" matching the 10xxxxxx pattern,
 *  and then subtract that from the total byte length to get the number of "start bytes" -
 *  coinciding with the number of UTF-8 characters.
 *
 *  @param[in] text String to be scanned.
 *  @param[in] length Number of bytes in the string.
 *  @return Number of UTF-8 characters in the string.
 *
 *  @example Count characters:
 *  @code
 *      size_t char_count = sz_utf8_count(text, length);
 *      printf("String has %zu characters\n", char_count);
 *  @endcode
 */
SZ_DYNAMIC sz_size_t sz_utf8_count(sz_cptr_t text, sz_size_t length);

/**
 *  @brief  Skip forward to the Nth UTF-8 character.
 *
 *  @param[in] text String to be scanned.
 *  @param[in] length Number of bytes in the string.
 *  @param[in] n Number of UTF-8 characters to skip (0-indexed, so n=0 returns text).
 *  @return Pointer to the Nth character, or NULL if the string has fewer than n characters.
 *
 *  @example Skip to character 1000 (e.g., pagination):
 *  @code
 *      char const *pos = sz_utf8_find_nth(text, length, 1000);
 *      if (!pos) {
 *          // String has fewer than 1000 characters
 *      }
 *  @endcode
 *
 *  @example Truncate to 280 characters (Twitter-style):
 *  @code
 *      char const *end = sz_utf8_find_nth(text, length, 280);
 *      size_t truncated_bytes = end ? (end - text) : length;
 *  @endcode
 */
SZ_DYNAMIC sz_cptr_t sz_utf8_find_nth(sz_cptr_t text, sz_size_t length, sz_size_t n);

/**
 *  @brief  Skips to the first occurrence of a UTF-8 newline character in a string.
 *
 *  Here are all the UTF-8 newline characters we are looking for (7 characters + CRLF):
 *  - single-byte chars (4 total):
 *    - U+000A for @c "\n" (LINE FEED)
 *    - U+000B for @c "\v" (VERTICAL TAB / LINE TABULATION)
 *    - U+000C for @c "\f" (FORM FEED)
 *    - U+000D for @c "\r" (CARRIAGE RETURN)
 *  - double-byte chars (1 total):
 *    - U+0085 for @c 0xC285 (NEXT LINE)
 *  - triple-byte chars (2 total):
 *    - U+2028 for @c 0xE280A8 (LINE SEPARATOR)
 *    - U+2029 for @c 0xE280A9 (PARAGRAPH SEPARATOR)
 *  - double-character sequence:
 *    - U+000D U+000A for @c "\r\n" that should be treated as a single new line!
 *
 *  @note U+001C, U+001D, U+001E (FILE/GROUP/RECORD SEPARATOR) are NOT included.
 *        These are data structure delimiters used in formats like USV (Unicode Separated Values),
 *        not line breaks. Use @c sz_find_byte() if you need to find these separators.
 *
 *  @param[in] text String to be scanned.
 *  @param[in] length Number of bytes in the string.
 *  @param[out] matched_length Number of bytes in the matched newline delimiter.
 *  @return Pointer to the first matching newline character from @p text, or @c SZ_NULL_CHAR if not found.
 */
SZ_DYNAMIC sz_cptr_t sz_utf8_find_newline(sz_cptr_t text, sz_size_t length, sz_size_t *matched_length);

/**
 *  @brief  Skips to the first occurrence of a UTF-8 whitespace character in a string.
 *
 *  Implements the Unicode White_Space property (25 characters total).
 *  Per Unicode standard, whitespace includes all newline characters plus horizontal spaces.
 *  Matches the behavior of ICU's u_isspace() and Python's str.isspace().
 *
 *  - single-byte chars (6 total):
 *    - U+0009 tab @c "\t" (CHARACTER TABULATION)
 *    - U+000A @c "\n" (LINE FEED - newline)
 *    - U+000B @c "\v" (LINE TABULATION - newline)
 *    - U+000C @c "\f" (FORM FEED - newline)
 *    - U+000D @c "\r" (CARRIAGE RETURN - newline)
 *    - U+0020 (SPACE)
 *  - double-byte chars (2 total):
 *    - U+0085 @c 0xC285 (NEXT LINE - newline)
 *    - U+00A0 @c 0xC2A0 (NO-BREAK SPACE)
 *  - triple-byte chars (17 total):
 *    - U+1680 @c 0xE19A80 (OGHAM SPACE MARK)
 *    - U+2000 @c 0xE28080 (EN QUAD)
 *    - U+2001 @c 0xE28081 (EM QUAD)
 *    - U+2002 @c 0xE28082 (EN SPACE)
 *    - U+2003 @c 0xE28083 (EM SPACE)
 *    - U+2004 @c 0xE28084 (THREE-PER-EM SPACE)
 *    - U+2005 @c 0xE28085 (FOUR-PER-EM SPACE)
 *    - U+2006 @c 0xE28086 (SIX-PER-EM SPACE)
 *    - U+2007 @c 0xE28087 (FIGURE SPACE)
 *    - U+2008 @c 0xE28088 (PUNCTUATION SPACE)
 *    - U+2009 @c 0xE28089 (THIN SPACE)
 *    - U+200A @c 0xE2808A (HAIR SPACE)
 *    - U+2028 @c 0xE280A8 (LINE SEPARATOR - newline)
 *    - U+2029 @c 0xE280A9 (PARAGRAPH SEPARATOR - newline)
 *    - U+202F @c 0xE280AF (NARROW NO-BREAK SPACE)
 *    - U+205F @c 0xE2819F (MEDIUM MATHEMATICAL SPACE)
 *    - U+3000 @c 0xE38080 (IDEOGRAPHIC SPACE)
 *
 *  The last one, the IDEOGRAPHIC SPACE (U+3000), is commonly used in East Asian typography,
 *  like Japanese formatted text or Chinese traditional poetry alignments.
 *
 *  @note NOT included (despite some implementations treating them as whitespace):
 *    - U+001C, U+001D, U+001E, U+001F (FILE/GROUP/RECORD/UNIT SEPARATOR):
 *      These are data structure delimiters for formats like USV (Unicode Separated Values).
 *      Only Java's Character.isWhitespace() includes them; Unicode, ICU, and Python do not.
 *    - U+200B, U+200C, U+200D (ZERO WIDTH SPACE/NON-JOINER/JOINER):
 *      These are Format characters, not whitespace. They have no width and affect rendering,
 *      not spacing.
 *
 *  @param[in] text String to be scanned.
 *  @param[in] length Number of bytes in the string.
 *  @param[out] matched_length Number of bytes in the matched whitespace character.
 *  @return Pointer to the first matching whitespace character from @p text, or @c SZ_NULL_CHAR if not found.
 */
SZ_DYNAMIC sz_cptr_t sz_utf8_find_whitespace(sz_cptr_t text, sz_size_t length, sz_size_t *matched_length);

#pragma endregion

#pragma region Platform-Specific Backends

/** @copydoc sz_utf8_count */
SZ_PUBLIC sz_size_t sz_utf8_count_serial(sz_cptr_t text, sz_size_t length);
/** @copydoc sz_utf8_find_nth */
SZ_PUBLIC sz_cptr_t sz_utf8_find_nth_serial(sz_cptr_t text, sz_size_t length, sz_size_t n);
/** @copydoc sz_utf8_find_newline */
SZ_PUBLIC sz_cptr_t sz_utf8_find_newline_serial(sz_cptr_t text, sz_size_t length, sz_size_t *matched_length);
/** @copydoc sz_utf8_find_whitespace */
SZ_PUBLIC sz_cptr_t sz_utf8_find_whitespace_serial(sz_cptr_t text, sz_size_t length, sz_size_t *matched_length);

#if SZ_USE_HASWELL
/** @copydoc sz_utf8_count */
SZ_PUBLIC sz_size_t sz_utf8_count_haswell(sz_cptr_t text, sz_size_t length);
/** @copydoc sz_utf8_find_nth */
SZ_PUBLIC sz_cptr_t sz_utf8_find_nth_haswell(sz_cptr_t text, sz_size_t length, sz_size_t n);
/** @copydoc sz_utf8_find_newline */
SZ_PUBLIC sz_cptr_t sz_utf8_find_newline_haswell(sz_cptr_t text, sz_size_t length, sz_size_t *matched_length);
/** @copydoc sz_utf8_find_whitespace */
SZ_PUBLIC sz_cptr_t sz_utf8_find_whitespace_haswell(sz_cptr_t text, sz_size_t length, sz_size_t *matched_length);
#endif

#if SZ_USE_ICE
/** @copydoc sz_utf8_count */
SZ_PUBLIC sz_size_t sz_utf8_count_ice(sz_cptr_t text, sz_size_t length);
/** @copydoc sz_utf8_find_nth */
SZ_PUBLIC sz_cptr_t sz_utf8_find_nth_ice(sz_cptr_t text, sz_size_t length, sz_size_t n);
/** @copydoc sz_utf8_find_newline */
SZ_PUBLIC sz_cptr_t sz_utf8_find_newline_ice(sz_cptr_t text, sz_size_t length, sz_size_t *matched_length);
/** @copydoc sz_utf8_find_whitespace */
SZ_PUBLIC sz_cptr_t sz_utf8_find_whitespace_ice(sz_cptr_t text, sz_size_t length, sz_size_t *matched_length);
#endif

#if SZ_USE_NEON
/** @copydoc sz_utf8_count */
SZ_PUBLIC sz_size_t sz_utf8_count_neon(sz_cptr_t text, sz_size_t length);
/** @copydoc sz_utf8_find_nth */
SZ_PUBLIC sz_cptr_t sz_utf8_find_nth_neon(sz_cptr_t text, sz_size_t length, sz_size_t n);
/** @copydoc sz_utf8_find_newline */
SZ_PUBLIC sz_cptr_t sz_utf8_find_newline_neon(sz_cptr_t text, sz_size_t length, sz_size_t *matched_length);
/** @copydoc sz_utf8_find_whitespace */
SZ_PUBLIC sz_cptr_t sz_utf8_find_whitespace_neon(sz_cptr_t text, sz_size_t length, sz_size_t *matched_length);
#endif

/**
 *  SVE2 provides a lot of UTF-8 friendly instructions superior to NEON, including:
 *  - svcmpeq_n: Compare vector elements to a scalar byte value without broadcast overhead.
 *  - svmatch: Compare each value against up to 16 other byte values in a single instruction.
 *  - svbrkb: Find byte positions of break characters in UTF-8 strings.
 */
#if SZ_USE_SVE2
/** @copydoc sz_utf8_count */
SZ_PUBLIC sz_size_t sz_utf8_count_sve2(sz_cptr_t text, sz_size_t length);
/** @copydoc sz_utf8_find_nth */
SZ_PUBLIC sz_cptr_t sz_utf8_find_nth_sve2(sz_cptr_t text, sz_size_t length, sz_size_t n);
/** @copydoc sz_utf8_find_newline */
SZ_PUBLIC sz_cptr_t sz_utf8_find_newline_sve2(sz_cptr_t text, sz_size_t length, sz_size_t *matched_length);
/** @copydoc sz_utf8_find_whitespace */
SZ_PUBLIC sz_cptr_t sz_utf8_find_whitespace_sve2(sz_cptr_t text, sz_size_t length, sz_size_t *matched_length);
#endif

#pragma endregion

/*  Implementation Section
 *  Following the same pattern as find.h with implementations in the header file.
 */

#pragma region Serial Implementation

SZ_PUBLIC sz_cptr_t sz_utf8_find_newline_serial(sz_cptr_t text, sz_size_t length, sz_size_t *matched_length) {
    // TODO: Optimize with a SWAR variant
    sz_u8_t const *text_u8 = (sz_u8_t const *)text;
    sz_u8_t const *end_u8 = text_u8 + length;
    while (text_u8 != end_u8) {
        sz_u8_t const c = *text_u8;
        switch (c) {
        case '\n':
        case '\v':
        case '\f': *matched_length = 1; return (sz_cptr_t)text_u8;
        // Differentiate between "\r" and "\r\n"
        case '\r':
            if (text_u8 + 1 != end_u8 && text_u8[1] == '\n') {
                *matched_length = 2;
                return (sz_cptr_t)text_u8;
            }
            else {
                *matched_length = 1;
                return (sz_cptr_t)text_u8;
            }
        // Matching the 0xC285 character
        case 0xC2:
            if (text_u8 + 1 != end_u8 && text_u8[1] == 0x85) {
                *matched_length = 2;
                return (sz_cptr_t)text_u8;
            }
            else {
                ++text_u8;
                continue;
            }
        // Matching 3-byte newline characters
        case 0xE2:
            if (text_u8 + 2 < end_u8 && text_u8[1] == 0x80 && text_u8[2] == 0xA8) {
                *matched_length = 3;
                return (sz_cptr_t)text_u8;
            }
            else if (text_u8 + 2 < end_u8 && text_u8[1] == 0x80 && text_u8[2] == 0xA9) {
                *matched_length = 3;
                return (sz_cptr_t)text_u8;
            }
            else {
                ++text_u8;
                continue;
            }
        default: ++text_u8; continue;
        }
    }
    *matched_length = 0;
    return SZ_NULL_CHAR;
}

SZ_PUBLIC sz_cptr_t sz_utf8_find_whitespace_serial(sz_cptr_t text, sz_size_t length, sz_size_t *matched_length) {
    // TODO: Optimize with a SWAR variant
    sz_u8_t const *text_u8 = (sz_u8_t const *)text;
    sz_u8_t const *end_u8 = text_u8 + length;
    while (text_u8 != end_u8) {
        sz_u8_t const c = *text_u8;
        switch (c) {
        case ' ':
        case '\t':
        case '\n':
        case '\v':
        case '\f':
        case '\r': *matched_length = 1; return (sz_cptr_t)text_u8;
        // Matching 2-byte whitespace characters
        case 0xC2:
            if (text_u8 + 1 != end_u8) {
                // U+0085: NEXT LINE (NEL)
                if (text_u8[1] == 0x85) {
                    *matched_length = 2;
                    return (sz_cptr_t)text_u8;
                }
                // U+00A0: NO-BREAK SPACE
                else if (text_u8[1] == 0xA0) {
                    *matched_length = 2;
                    return (sz_cptr_t)text_u8;
                }
            }
            ++text_u8;
            continue;
        // Matching the 0xE19A80 ogham space mark
        case 0xE1:
            if (text_u8 + 2 < end_u8 && text_u8[1] == 0x9A && text_u8[2] == 0x80) {
                *matched_length = 3;
                return (sz_cptr_t)text_u8;
            }
            else {
                ++text_u8;
                continue;
            }
        // Match the 3-byte whitespace characters starting with 0xE2
        case 0xE2:
            if (text_u8 + 2 < end_u8) {
                // U+2000 to U+200D: 0xE28080 to 0xE2808D
                if (text_u8[1] == 0x80 && text_u8[2] >= 0x80 && text_u8[2] <= 0x8D) {
                    *matched_length = 3;
                    return (sz_cptr_t)text_u8;
                }
                // U+2028: LINE SEPARATOR (0xE280A8)
                else if (text_u8[1] == 0x80 && text_u8[2] == 0xA8) {
                    *matched_length = 3;
                    return (sz_cptr_t)text_u8;
                }
                // U+2029: PARAGRAPH SEPARATOR (0xE280A9)
                else if (text_u8[1] == 0x80 && text_u8[2] == 0xA9) {
                    *matched_length = 3;
                    return (sz_cptr_t)text_u8;
                }
                // U+202F: NARROW NO-BREAK SPACE (0xE280AF)
                else if (text_u8[1] == 0x80 && text_u8[2] == 0xAF) {
                    *matched_length = 3;
                    return (sz_cptr_t)text_u8;
                }
                // U+205F: MEDIUM MATHEMATICAL SPACE (0xE2819F)
                else if (text_u8[1] == 0x81 && text_u8[2] == 0x9F) {
                    *matched_length = 3;
                    return (sz_cptr_t)text_u8;
                }
            }
            ++text_u8;
            continue;
        // Match the 3-byte ideographic space
        case 0xE3:
            if (text_u8 + 2 < end_u8 && text_u8[1] == 0x80 && text_u8[2] == 0x80) {
                *matched_length = 3;
                return (sz_cptr_t)text_u8;
            }
            else {
                ++text_u8;
                continue;
            }
        }
        ++text_u8;
    }
    *matched_length = 0;
    return SZ_NULL_CHAR;
}

SZ_PUBLIC sz_size_t sz_utf8_count_serial(sz_cptr_t text, sz_size_t length) {
    sz_u8_t const *text_u8 = (sz_u8_t const *)text;
    sz_u8_t const *end_u8 = text_u8 + length;
    sz_size_t char_count = 0;

    while (text_u8 < end_u8) {
        // Count this byte if it's NOT a continuation byte
        if ((*text_u8 & 0xC0) != 0x80) char_count++;
        text_u8++;
    }

    return char_count;
}

SZ_PUBLIC sz_cptr_t sz_utf8_find_nth_serial(sz_cptr_t text, sz_size_t length, sz_size_t n) {
    sz_u8_t const *text_u8 = (sz_u8_t const *)text;
    sz_u8_t const *end_u8 = text_u8 + length;
    sz_size_t char_count = 0;

    while (text_u8 < end_u8) {
        // Check if this is NOT a continuation byte
        if ((*text_u8 & 0xC0) != 0x80) {
            if (char_count == n) return (sz_cptr_t)text_u8;
            char_count++;
        }
        text_u8++;
    }

    // If we reached the end without finding the nth character, return NULL
    return SZ_NULL_CHAR;
}

#pragma endregion // Serial Implementation

#pragma region Ice Lake Implementation
#if SZ_USE_ICE
#if defined(__clang__)
#pragma clang attribute push(                                                                          \
    __attribute__((target("avx,avx512f,avx512vl,avx512bw,avx512dq,avx512vbmi,avx512vbmi2,bmi,bmi2"))), \
    apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("avx", "avx512f", "avx512vl", "avx512bw", "avx512dq", "avx512vbmi", "avx512vbmi2", "bmi", "bmi2")
#endif

SZ_PUBLIC sz_cptr_t sz_utf8_find_newline_ice(sz_cptr_t text, sz_size_t length, sz_size_t *matched_length) {

    // We need to check if the ASCII chars in [10,13] (same as '\n', '\v', '\f', '\r') are present.
    // The last one - '\r' - needs special handling to differentiate between "\r" and "\r\n".
    sz_u512_vec_t n_vec, v_vec, f_vec, r_vec;
    n_vec.zmm = _mm512_set1_epi8('\n');
    v_vec.zmm = _mm512_set1_epi8('\v');
    f_vec.zmm = _mm512_set1_epi8('\f');
    r_vec.zmm = _mm512_set1_epi8('\r');

    // We also need to match the 2-byte newline character 0xC285 (NEL),
    // as well as the 3-byte characters 0xE280A8 (PS) and 0xE280A9 (LS).
    sz_u512_vec_t x_c2_vec, x_85_vec, x_e2_vec, x_80_vec, x_a8_vec, x_a9_vec;
    x_c2_vec.zmm = _mm512_set1_epi8('\xC2');
    x_85_vec.zmm = _mm512_set1_epi8('\x85');
    x_e2_vec.zmm = _mm512_set1_epi8('\xE2');
    x_80_vec.zmm = _mm512_set1_epi8('\x80');
    x_a8_vec.zmm = _mm512_set1_epi8('\xA8');
    x_a9_vec.zmm = _mm512_set1_epi8('\xA9');

    // We check 64 bytes of data at once, but only step forward by 62 bytes for split-register matches.
    sz_u512_vec_t text_vec;
    while (length >= 64) {
        text_vec.zmm = _mm512_loadu_epi8(text);

        // 1-byte indicators & matches
        __mmask64 n_mask = _mm512_cmpeq_epi8_mask(text_vec.zmm, n_vec.zmm);
        __mmask64 v_mask = _mm512_cmpeq_epi8_mask(text_vec.zmm, v_vec.zmm);
        __mmask64 f_mask = _mm512_cmpeq_epi8_mask(text_vec.zmm, f_vec.zmm);
        __mmask64 r_mask = _mm512_mask_cmpeq_epi8_mask(0x7FFFFFFFFFFFFFFF, text_vec.zmm, r_vec.zmm); // Ignore last
        sz_u64_t one_byte_mask = _cvtmask64_u64(_kor_mask64(_kor_mask64(n_mask, v_mask), _kor_mask64(f_mask, r_mask)));

        // 2-byte indicators
        __mmask64 x_c2_mask = _mm512_cmpeq_epi8_mask(text_vec.zmm, x_c2_vec.zmm);
        __mmask64 x_85_mask = _mm512_cmpeq_epi8_mask(text_vec.zmm, x_85_vec.zmm);
        __mmask64 x_e2_mask = _mm512_cmpeq_epi8_mask(text_vec.zmm, x_e2_vec.zmm);
        __mmask64 x_80_mask = _mm512_cmpeq_epi8_mask(text_vec.zmm, x_80_vec.zmm);
        __mmask64 x_a8_mask = _mm512_cmpeq_epi8_mask(text_vec.zmm, x_a8_vec.zmm);
        __mmask64 x_a9_mask = _mm512_cmpeq_epi8_mask(text_vec.zmm, x_a9_vec.zmm);

        // 2-byte matches
        __mmask64 rn_mask = _kand_mask64(r_mask, _kshiftri_mask64(n_mask, 1));
        __mmask64 x_c285_mask = _kand_mask64(x_c2_mask, _kshiftri_mask64(x_85_mask, 1));
        sz_u64_t two_byte_mask = _cvtmask64_u64(_kor_mask64(rn_mask, x_c285_mask));

        // 3-byte matches
        __mmask64 x_e280_mask = _kand_mask64(x_e2_mask, _kshiftri_mask64(x_80_mask, 1));
        __mmask64 x_e280a8_mask = _kand_mask64(x_e280_mask, _kshiftri_mask64(x_a8_mask, 2));
        __mmask64 x_e280a9_mask = _kand_mask64(x_e280_mask, _kshiftri_mask64(x_a9_mask, 2));
        sz_u64_t three_byte_mask = _cvtmask64_u64(_kor_mask64(x_e280a8_mask, x_e280a9_mask));

        // Find the earliest match regardless of length
        sz_u64_t combined_mask = one_byte_mask | two_byte_mask | three_byte_mask;
        if (combined_mask) {
            int first_offset = sz_u64_ctz(combined_mask);
            sz_u64_t first_match_mask = (sz_u64_t)(1) << first_offset;

            // We don't want to produce too much divergent control flow,
            // but need to achieve a behavior similar to this:
            //
            //  if (first_match_mask & three_byte_mask) { *matched_length = 3; }
            //  else if (first_match_mask & two_byte_mask) { *matched_length = 2; }
            //  else { *matched_length = 1; }
            sz_size_t length_value = 1;
            length_value += (first_match_mask & (two_byte_mask | three_byte_mask)) != 0;
            length_value += (first_match_mask & (three_byte_mask)) != 0;
            *matched_length = length_value;
            return text + first_offset;
        }
        else {
            text += 62;
            length -= 62;
        }
    }

    return sz_utf8_find_newline_serial(text, length, matched_length);
}

SZ_PUBLIC sz_cptr_t sz_utf8_find_whitespace_ice(sz_cptr_t text, sz_size_t length, sz_size_t *matched_length) {

    // We need to check if the ASCII chars in [9,13] (same as '\t', '\n', '\v', '\f', '\r') are present.
    // There is also the canonical space ' ' (0x20).
    sz_u512_vec_t t_vec, r_vec, x_20_vec;
    t_vec.zmm = _mm512_set1_epi8('\t');
    r_vec.zmm = _mm512_set1_epi8('\r');
    x_20_vec.zmm = _mm512_set1_epi8(' ');

    // We also need to match the 2-byte characters 0xC285 (NEL) and 0xC2A0 (NBSP),
    sz_u512_vec_t x_c2_vec, x_85_vec, x_a0_vec;
    x_c2_vec.zmm = _mm512_set1_epi8('\xC2');
    x_85_vec.zmm = _mm512_set1_epi8('\x85');
    x_a0_vec.zmm = _mm512_set1_epi8('\xA0');

    // We also need to match 3-byte ogham space mark 0xE19A80 (OGHAM SPACE MARK),
    // a range of 3-byte characters from 0xE28080 to 0xE2808D (various spaces),
    // U+202F (0xE280AF), U+205F (0xE2819F),
    // U+2028 (0xE280A8) LINE SEPARATOR, U+2029 (0xE280A9) PARAGRAPH SEPARATOR,
    // and the 3-byte ideographic space 0xE38080 (IDEOGRAPHIC SPACE).
    sz_u512_vec_t x_e1_vec, x_e2_vec, x_e3_vec,           // ? possible first byte values
        x_9a_vec, x_80_vec, x_81_vec,                     // ? possible second byte values
        x_8d_vec, x_a8_vec, x_a9_vec, x_af_vec, x_9f_vec; // ? third byte values for ranges and specific matches
    x_e1_vec.zmm = _mm512_set1_epi8('\xE1');
    x_e2_vec.zmm = _mm512_set1_epi8('\xE2');
    x_e3_vec.zmm = _mm512_set1_epi8('\xE3');
    x_9a_vec.zmm = _mm512_set1_epi8('\x9A');
    x_80_vec.zmm = _mm512_set1_epi8('\x80');
    x_81_vec.zmm = _mm512_set1_epi8('\x81');
    x_8d_vec.zmm = _mm512_set1_epi8('\x8D');
    x_a8_vec.zmm = _mm512_set1_epi8('\xA8');
    x_a9_vec.zmm = _mm512_set1_epi8('\xA9');
    x_af_vec.zmm = _mm512_set1_epi8('\xAF');
    x_9f_vec.zmm = _mm512_set1_epi8('\x9F');

    // We check 64 bytes of data at once, but only step forward by 62 bytes for split-register matches.
    sz_u512_vec_t text_vec;
    while (length >= 64) {
        text_vec.zmm = _mm512_loadu_epi8(text);

        // 1-byte indicators & matches
        // Range [9,13] covers \t, \n, \v, \f, \r
        __mmask64 x_20_mask = _mm512_cmpeq_epi8_mask(text_vec.zmm, x_20_vec.zmm);
        __mmask64 t_mask = _mm512_cmpge_epu8_mask(text_vec.zmm, t_vec.zmm);
        __mmask64 r_mask = _mm512_cmple_epu8_mask(text_vec.zmm, r_vec.zmm);
        sz_u64_t one_byte_mask = _cvtmask64_u64(_kor_mask64(x_20_mask, _kand_mask64(t_mask, r_mask)));

        // Instead of immediately checking for 2-byte and 3-byte matches with a ridiculous number of masks and
        // comparisons, let's define a "fast path" for following cases:
        // - no whitespaces are found in the range
        // - a one-byte match comes before any possible prefix byte of a multi-byte match
        __mmask64 x_c2_mask = _mm512_mask_cmpeq_epi8_mask(0x7FFFFFFFFFFFFFFF, text_vec.zmm, x_c2_vec.zmm);
        __mmask64 x_e1_mask = _mm512_mask_cmpeq_epi8_mask(0x3FFFFFFFFFFFFFFF, text_vec.zmm, x_e1_vec.zmm);
        __mmask64 x_e2_mask = _mm512_mask_cmpeq_epi8_mask(0x3FFFFFFFFFFFFFFF, text_vec.zmm, x_e2_vec.zmm);
        __mmask64 x_e3_mask = _mm512_mask_cmpeq_epi8_mask(0x3FFFFFFFFFFFFFFF, text_vec.zmm, x_e3_vec.zmm);

        // Check if we matched the "fast path"
        if (one_byte_mask) {
            sz_u64_t prefix_byte_mask =
                _cvtmask64_u64(_kor_mask64(_kor_mask64(x_c2_mask, x_e1_mask), _kor_mask64(x_e2_mask, x_e3_mask)));
            if (prefix_byte_mask) {
                int first_one_byte_offset = sz_u64_ctz(one_byte_mask);
                int first_prefix_offset = sz_u64_ctz(prefix_byte_mask);
                if (first_one_byte_offset < first_prefix_offset) {
                    *matched_length = 1;
                    return text + first_one_byte_offset;
                }
            }
            else {
                int first_one_byte_offset = sz_u64_ctz(one_byte_mask);
                *matched_length = 1;
                return text + first_one_byte_offset;
            }
        }

        // 2-byte indicators suffixes & matches
        __mmask64 x_85_mask = _mm512_cmpeq_epi8_mask(text_vec.zmm, x_85_vec.zmm);
        __mmask64 x_a0_mask = _mm512_cmpeq_epi8_mask(text_vec.zmm, x_a0_vec.zmm);
        sz_u64_t two_byte_mask = _cvtmask64_u64(             //
            _kand_mask64(x_c2_mask,                          //
                         _kor_mask64(                        //
                             _kshiftri_mask64(x_85_mask, 1), // U+0085 NEL
                             _kshiftri_mask64(x_a0_mask, 1)  // U+00A0 NBSP
                             )));

        // 3-byte indicators suffixes
        __mmask64 x_9a_mask = _mm512_cmpeq_epi8_mask(text_vec.zmm, x_9a_vec.zmm);
        __mmask64 x_80_mask = _mm512_cmpeq_epi8_mask(text_vec.zmm, x_80_vec.zmm);
        __mmask64 x_81_mask = _mm512_cmpeq_epi8_mask(text_vec.zmm, x_81_vec.zmm);
        __mmask64 x_80_ge_mask = _mm512_cmpge_epu8_mask(text_vec.zmm, x_80_vec.zmm);
        __mmask64 x_8d_le_mask = _mm512_cmple_epu8_mask(text_vec.zmm, x_8d_vec.zmm);
        __mmask64 x_a8_mask = _mm512_cmpeq_epi8_mask(text_vec.zmm, x_a8_vec.zmm);
        __mmask64 x_a9_mask = _mm512_cmpeq_epi8_mask(text_vec.zmm, x_a9_vec.zmm);
        __mmask64 x_af_mask = _mm512_cmpeq_epi8_mask(text_vec.zmm, x_af_vec.zmm);
        __mmask64 x_9f_mask = _mm512_cmpeq_epi8_mask(text_vec.zmm, x_9f_vec.zmm);

        // 3-byte matches
        __mmask64 ogham_mask =
            _kand_mask64(x_e1_mask, _kand_mask64(_kshiftri_mask64(x_9a_mask, 1), _kshiftri_mask64(x_80_mask, 2)));
        // U+2000 to U+200D: E2 80 [80-8D]
        __mmask64 range_e280_mask = _kand_mask64(
            x_e2_mask, _kand_mask64(_kshiftri_mask64(x_80_mask, 1), _kand_mask64(_kshiftri_mask64(x_80_ge_mask, 2),
                                                                                 _kshiftri_mask64(x_8d_le_mask, 2))));
        // U+202F: E2 80 AF (NARROW NO-BREAK SPACE)
        __mmask64 nnbsp_mask =
            _kand_mask64(x_e2_mask, _kand_mask64(_kshiftri_mask64(x_80_mask, 1), _kshiftri_mask64(x_af_mask, 2)));
        // U+205F: E2 81 9F (MEDIUM MATHEMATICAL SPACE)
        __mmask64 mmsp_mask =
            _kand_mask64(x_e2_mask, _kand_mask64(_kshiftri_mask64(x_81_mask, 1), _kshiftri_mask64(x_9f_mask, 2)));
        // U+2028: E2 80 A8 (LINE SEPARATOR)
        __mmask64 line_mask =
            _kand_mask64(x_e2_mask, _kand_mask64(_kshiftri_mask64(x_80_mask, 1), _kshiftri_mask64(x_a8_mask, 2)));
        // U+2029: E2 80 A9 (PARAGRAPH SEPARATOR)
        __mmask64 paragraph_mask =
            _kand_mask64(x_e2_mask, _kand_mask64(_kshiftri_mask64(x_80_mask, 1), _kshiftri_mask64(x_a9_mask, 2)));
        __mmask64 ideographic_mask =
            _kand_mask64(x_e3_mask, _kand_mask64(_kshiftri_mask64(x_80_mask, 1), _kshiftri_mask64(x_80_mask, 2)));
        sz_u64_t three_byte_mask = _cvtmask64_u64(_kor_mask64(
            _kor_mask64(_kor_mask64(_kor_mask64(ogham_mask, range_e280_mask), _kor_mask64(nnbsp_mask, mmsp_mask)),
                        _kor_mask64(line_mask, paragraph_mask)),
            ideographic_mask));

        // Find the earliest match regardless of length
        sz_u64_t combined_mask = one_byte_mask | two_byte_mask | three_byte_mask;
        if (combined_mask) {
            int first_offset = sz_u64_ctz(combined_mask);
            sz_u64_t first_match_mask = (sz_u64_t)(1) << first_offset;

            // We don't want to produce too much divergent control flow,
            // but need to achieve a behavior similar to this:
            //
            //  if (first_match_mask & three_byte_mask) { *matched_length = 3; }
            //  else if (first_match_mask & two_byte_mask) { *matched_length = 2; }
            //  else { *matched_length = 1; }
            sz_size_t length_value = 1;
            length_value += (first_match_mask & (two_byte_mask | three_byte_mask)) != 0;
            length_value += (first_match_mask & (three_byte_mask)) != 0;
            *matched_length = length_value;
            return text + first_offset;
        }
        else {
            text += 62;
            length -= 62;
        }
    }

    return sz_utf8_find_whitespace_serial(text, length, matched_length);
}

SZ_PUBLIC sz_size_t sz_utf8_count_ice(sz_cptr_t text, sz_size_t length) {
    // UTF-8 character counting strategy:
    // Count every byte that is NOT a continuation byte (i.e., character start bytes).
    //
    // UTF-8 byte patterns:
    //   ASCII:        0xxxxxxx (0x00-0x7F)  - single byte character
    //   Start 2-byte: 110xxxxx (0xC0-0xDF)  - first byte of 2-byte sequence
    //   Start 3-byte: 1110xxxx (0xE0-0xEF)  - first byte of 3-byte sequence
    //   Start 4-byte: 11110xxx (0xF0-0xF7)  - first byte of 4-byte sequence
    //   Continuation: 10xxxxxx (0x80-0xBF)  - continuation byte (NOT a character start)
    //
    // To detect continuation bytes: (byte & 0xC0) == 0x80
    //   0xC0 = 11000000  - masks the top 2 bits
    //   0x80 = 10000000  - pattern for continuation bytes after masking

    sz_u512_vec_t continuation_mask_vec, continuation_pattern_vec;
    continuation_mask_vec.zmm = _mm512_set1_epi8((char)0xC0);    // 0xC0 = 0b11000000 - mask top 2 bits
    continuation_pattern_vec.zmm = _mm512_set1_epi8((char)0x80); // 0x80 = 0b10000000 - continuation pattern

    sz_u8_t const *text_u8 = (sz_u8_t const *)text;
    sz_size_t char_count = 0;

    // Process 64 bytes at a time
    sz_u512_vec_t text_vec, headers_vec;
    while (length >= 64) {
        text_vec.zmm = _mm512_loadu_epi8(text_u8);

        // Apply mask (byte & 0xC0) to extract top 2 bits of each byte
        headers_vec.zmm = _mm512_and_si512(text_vec.zmm, continuation_mask_vec.zmm);

        // Compare with 0x80 (0b10000000) to find continuation bytes
        sz_u64_t start_byte_mask =
            _cvtmask64_u64(_mm512_cmpneq_epi8_mask(headers_vec.zmm, continuation_pattern_vec.zmm));

        // Count non-continuation bytes (i.e., character starts)
        char_count += _mm_popcnt_u64(start_byte_mask);
        text_u8 += 64;
        length -= 64;
    }

    // Process remaining bytes with a masked variant
    if (length) {
        __mmask64 load_mask = sz_u64_mask_until_(length);
        text_vec.zmm = _mm512_maskz_loadu_epi8(load_mask, text_u8);
        headers_vec.zmm = _mm512_and_si512(text_vec.zmm, continuation_mask_vec.zmm);
        __mmask64 start_byte_mask =
            _mm512_mask_cmpneq_epi8_mask(load_mask, headers_vec.zmm, continuation_pattern_vec.zmm);
        char_count += _mm_popcnt_u64(_cvtmask64_u64(start_byte_mask));
    }
    return char_count;
}

SZ_PUBLIC sz_cptr_t sz_utf8_find_nth_ice(sz_cptr_t text, sz_size_t length, sz_size_t n) {

    // The logic of this function is similar to `sz_utf8_count_ice`, but uses PDEP & PEXT
    // instructions in the inner loop to locate Nth character start byte efficiently
    // without one more loop.
    sz_u512_vec_t continuation_mask_vec, continuation_pattern_vec;
    continuation_mask_vec.zmm = _mm512_set1_epi8((char)0xC0);
    continuation_pattern_vec.zmm = _mm512_set1_epi8((char)0x80);

    sz_u8_t const *text_u8 = (sz_u8_t const *)text;

    // Process 64 bytes at a time
    sz_u512_vec_t text_vec, headers_vec;
    while (length >= 64) {
        text_vec.zmm = _mm512_loadu_epi8(text_u8);

        // Apply mask (byte & 0xC0) to extract top 2 bits of each byte
        headers_vec.zmm = _mm512_and_si512(text_vec.zmm, continuation_mask_vec.zmm);

        // Compare with 0x80 (0b10000000) to find continuation bytes
        sz_u64_t start_byte_mask =
            _cvtmask64_u64(_mm512_cmpneq_epi8_mask(headers_vec.zmm, continuation_pattern_vec.zmm));
        sz_size_t start_byte_count = _mm_popcnt_u64(start_byte_mask);

        // Check if we've reached the terminal part of our search
        if (n < start_byte_count) {
            // PDEP directly gives us the nth set bit position
            // Example: _pdep_u64(0b10, 0b0001010100) = 0b0000010000
            sz_u64_t deposited_bits = _pdep_u64((sz_u64_t)1 << n, start_byte_mask);
            int byte_offset = sz_u64_ctz(deposited_bits);
            return (sz_cptr_t)(text_u8 + byte_offset);
        }
        // Jump to the next block
        else {
            n -= start_byte_count;
            text_u8 += 64;
            length -= 64;
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
#endif            // SZ_USE_ICE
#pragma endregion // Ice Lake Implementation

#pragma region Haswell Implementation
#if SZ_USE_HASWELL
#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("avx2,bmi,bmi2,popcnt"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("avx2,bmi,bmi2,popcnt")
#endif

/**
 *  @brief  Unsigned byte greater-than-or-equal comparison for AVX2.
 *
 *  AVX2 lacks unsigned comparison intrinsics like `_mm256_cmpge_epu8`.
 *  This uses the identity: a >= b  ⟺  max(a, b) == a
 *  Since `_mm256_max_epu8` treats bytes as unsigned, this gives correct results.
 */
SZ_INTERNAL __m256i sz_mm256_cmpge_epu8_(__m256i a, __m256i b) { return _mm256_cmpeq_epi8(_mm256_max_epu8(a, b), a); }

SZ_PUBLIC sz_cptr_t sz_utf8_find_newline_haswell(sz_cptr_t text, sz_size_t length, sz_size_t *matched_length) {

    // We need to check if the ASCII chars in [10,13] (same as '\n', '\v', '\f', '\r') are present.
    // The last one - '\r' - needs special handling to differentiate between "\r" and "\r\n".
    __m256i n_vec = _mm256_set1_epi8('\n');
    __m256i v_vec = _mm256_set1_epi8('\v');
    __m256i f_vec = _mm256_set1_epi8('\f');
    __m256i r_vec = _mm256_set1_epi8('\r');

    // We also need to match the 2-byte newline character 0xC285 (NEL),
    // as well as the 3-byte characters 0xE280A8 (PS) and 0xE280A9 (LS).
    __m256i x_c2_vec = _mm256_set1_epi8('\xC2');
    __m256i x_85_vec = _mm256_set1_epi8('\x85');
    __m256i x_e2_vec = _mm256_set1_epi8('\xE2');
    __m256i x_80_vec = _mm256_set1_epi8('\x80');
    __m256i x_a8_vec = _mm256_set1_epi8('\xA8');
    __m256i x_a9_vec = _mm256_set1_epi8('\xA9');

    // We check 32 bytes of data at once, but only step forward by 30 bytes for split-register matches.
    while (length >= 32) {
        __m256i text_vec = _mm256_loadu_si256((__m256i const *)text);

        // 1-byte indicators & matches
        __m256i n_cmp = _mm256_cmpeq_epi8(text_vec, n_vec);
        __m256i v_cmp = _mm256_cmpeq_epi8(text_vec, v_vec);
        __m256i f_cmp = _mm256_cmpeq_epi8(text_vec, f_vec);
        __m256i r_cmp = _mm256_cmpeq_epi8(text_vec, r_vec);

        sz_u32_t n_mask = (sz_u32_t)_mm256_movemask_epi8(n_cmp);
        sz_u32_t v_mask = (sz_u32_t)_mm256_movemask_epi8(v_cmp);
        sz_u32_t f_mask = (sz_u32_t)_mm256_movemask_epi8(f_cmp);
        sz_u32_t r_mask = (sz_u32_t)_mm256_movemask_epi8(r_cmp) & 0x7FFFFFFF; // Ignore last byte
        sz_u32_t one_byte_mask = n_mask | v_mask | f_mask | r_mask;

        // 2-byte indicators
        __m256i x_c2_cmp = _mm256_cmpeq_epi8(text_vec, x_c2_vec);
        __m256i x_85_cmp = _mm256_cmpeq_epi8(text_vec, x_85_vec);
        __m256i x_e2_cmp = _mm256_cmpeq_epi8(text_vec, x_e2_vec);
        __m256i x_80_cmp = _mm256_cmpeq_epi8(text_vec, x_80_vec);
        __m256i x_a8_cmp = _mm256_cmpeq_epi8(text_vec, x_a8_vec);
        __m256i x_a9_cmp = _mm256_cmpeq_epi8(text_vec, x_a9_vec);

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
    __m256i x_20_vec = _mm256_set1_epi8(' ');

    // We also need to match the 2-byte characters 0xC285 (NEL) and 0xC2A0 (NBSP)
    __m256i x_c2_vec = _mm256_set1_epi8('\xC2');
    __m256i x_85_vec = _mm256_set1_epi8('\x85');
    __m256i x_a0_vec = _mm256_set1_epi8('\xA0');

    // 3-byte character prefixes and suffixes
    __m256i x_e1_vec = _mm256_set1_epi8('\xE1');
    __m256i x_e2_vec = _mm256_set1_epi8('\xE2');
    __m256i x_e3_vec = _mm256_set1_epi8('\xE3');
    __m256i x_9a_vec = _mm256_set1_epi8('\x9A');
    __m256i x_80_vec = _mm256_set1_epi8('\x80');
    __m256i x_81_vec = _mm256_set1_epi8('\x81');
    __m256i x_a8_vec = _mm256_set1_epi8('\xA8');
    __m256i x_a9_vec = _mm256_set1_epi8('\xA9');
    __m256i x_af_vec = _mm256_set1_epi8('\xAF');
    __m256i x_9f_vec = _mm256_set1_epi8('\x9F');

    // We check 32 bytes of data at once, but only step forward by 30 bytes for split-register matches.
    while (length >= 32) {
        __m256i text_vec = _mm256_loadu_si256((__m256i const *)text);

        // 1-byte indicators & matches
        // Range [9,13] covers \t, \n, \v, \f, \r
        __m256i x_20_cmp = _mm256_cmpeq_epi8(text_vec, x_20_vec);
        __m256i t_cmp = _mm256_cmpgt_epi8(text_vec, _mm256_set1_epi8((char)0x08)); // >= '\t' (0x09)
        __m256i r_cmp = _mm256_cmpgt_epi8(_mm256_set1_epi8((char)0x0E), text_vec); // <= '\r' (0x0D)
        __m256i tr_range = _mm256_and_si256(t_cmp, r_cmp);
        __m256i one_byte_cmp = _mm256_or_si256(x_20_cmp, tr_range);

        sz_u32_t one_byte_mask = (sz_u32_t)_mm256_movemask_epi8(one_byte_cmp);

        // 2-byte and 3-byte prefix indicators
        __m256i x_c2_cmp = _mm256_cmpeq_epi8(text_vec, x_c2_vec);
        __m256i x_e1_cmp = _mm256_cmpeq_epi8(text_vec, x_e1_vec);
        __m256i x_e2_cmp = _mm256_cmpeq_epi8(text_vec, x_e2_vec);
        __m256i x_e3_cmp = _mm256_cmpeq_epi8(text_vec, x_e3_vec);

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
        __m256i x_85_cmp = _mm256_cmpeq_epi8(text_vec, x_85_vec);
        __m256i x_a0_cmp = _mm256_cmpeq_epi8(text_vec, x_a0_vec);
        sz_u32_t x_85_mask = (sz_u32_t)_mm256_movemask_epi8(x_85_cmp);
        sz_u32_t x_a0_mask = (sz_u32_t)_mm256_movemask_epi8(x_a0_cmp);

        sz_u32_t x_c285_mask = x_c2_mask & (x_85_mask >> 1); // U+0085 NEL
        sz_u32_t x_c2a0_mask = x_c2_mask & (x_a0_mask >> 1); // U+00A0 NBSP
        sz_u32_t two_byte_mask = x_c285_mask | x_c2a0_mask;

        // 3-byte suffixes
        __m256i x_9a_cmp = _mm256_cmpeq_epi8(text_vec, x_9a_vec);
        __m256i x_80_cmp = _mm256_cmpeq_epi8(text_vec, x_80_vec);
        __m256i x_81_cmp = _mm256_cmpeq_epi8(text_vec, x_81_vec);
        __m256i x_80_ge_cmp = sz_mm256_cmpge_epu8_(text_vec, x_80_vec);                     // >= 0x80
        __m256i x_8d_le_cmp = sz_mm256_cmpge_epu8_(_mm256_set1_epi8((char)0x8D), text_vec); // <= 0x8D
        __m256i x_8d_range = _mm256_and_si256(x_80_ge_cmp, x_8d_le_cmp);
        __m256i x_a8_cmp = _mm256_cmpeq_epi8(text_vec, x_a8_vec);
        __m256i x_a9_cmp = _mm256_cmpeq_epi8(text_vec, x_a9_vec);
        __m256i x_af_cmp = _mm256_cmpeq_epi8(text_vec, x_af_vec);
        __m256i x_9f_cmp = _mm256_cmpeq_epi8(text_vec, x_9f_vec);

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
        sz_u32_t three_byte_mask =
            ogham_mask | range_e280_mask | nnbsp_mask | mmsp_mask | line_mask | paragraph_mask | ideographic_mask;

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
        sz_u32_t start_byte_mask =
            ~(sz_u32_t)_mm256_movemask_epi8(_mm256_cmpeq_epi8(headers_vec.ymm, continuation_pattern_vec.ymm));

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
        sz_u32_t start_byte_mask =
            ~(sz_u32_t)_mm256_movemask_epi8(_mm256_cmpeq_epi8(headers_vec.ymm, continuation_pattern_vec.ymm));
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
#endif            // SZ_USE_HASWELL
#pragma endregion // Haswell Implementation

#pragma region NEON Implementation
#if SZ_USE_NEON
#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("+simd"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("+simd")
#endif

SZ_INTERNAL sz_u64_t sz_utf8_vreinterpretq_u8_u4_(uint8x16_t vec) {
    // Use `vshrn` to produce a bitmask, similar to `movemask` in SSE.
    // https://community.arm.com/arm-community-blogs/b/infrastructure-solutions-blog/posts/porting-x86-vector-bitmask-optimizations-to-arm-neon
    return vget_lane_u64(vreinterpret_u64_u8(vshrn_n_u16(vreinterpretq_u16_u8(vec), 4)), 0) & 0x8888888888888888ull;
}

SZ_PUBLIC sz_cptr_t sz_utf8_find_newline_neon(sz_cptr_t text, sz_size_t length, sz_size_t *matched_length) {

    sz_u128_vec_t text_vec;
    uint8x16_t n_vec = vdupq_n_u8('\n');
    uint8x16_t v_vec = vdupq_n_u8('\v');
    uint8x16_t f_vec = vdupq_n_u8('\f');
    uint8x16_t r_vec = vdupq_n_u8('\r');
    uint8x16_t x_c2_vec = vdupq_n_u8(0xC2);
    uint8x16_t x_85_vec = vdupq_n_u8(0x85);
    uint8x16_t x_e2_vec = vdupq_n_u8(0xE2);
    uint8x16_t x_80_vec = vdupq_n_u8(0x80);
    uint8x16_t x_a8_vec = vdupq_n_u8(0xA8);
    uint8x16_t x_a9_vec = vdupq_n_u8(0xA9);

    uint8x16_t drop1_vec = {0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
                            0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x00};
    uint8x16_t drop2_vec = {0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
                            0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x00, 0x00};

    while (length >= 16) {
        text_vec.u8x16 = vld1q_u8((sz_u8_t const *)text);

        // 1-byte matches
        uint8x16_t n_cmp = vceqq_u8(text_vec.u8x16, n_vec);
        uint8x16_t v_cmp = vceqq_u8(text_vec.u8x16, v_vec);
        uint8x16_t f_cmp = vceqq_u8(text_vec.u8x16, f_vec);
        uint8x16_t r_cmp = vceqq_u8(text_vec.u8x16, r_vec);
        uint8x16_t one_byte_cmp = vorrq_u8(vorrq_u8(n_cmp, v_cmp), vorrq_u8(f_cmp, r_cmp));

        // 2- & 3-byte matches with shifted views
        uint8x16_t text1 = vextq_u8(text_vec.u8x16, text_vec.u8x16, 1);
        uint8x16_t text2 = vextq_u8(text_vec.u8x16, text_vec.u8x16, 2);
        uint8x16_t rn_vec = vandq_u8(r_cmp, vceqq_u8(text1, n_vec));
        uint8x16_t x_c285_vec = vandq_u8(vceqq_u8(text_vec.u8x16, x_c2_vec), vceqq_u8(text1, x_85_vec));
        uint8x16_t two_byte_cmp = vandq_u8(vorrq_u8(rn_vec, x_c285_vec), drop1_vec); // Ignore last split match

        uint8x16_t x_e2_cmp = vceqq_u8(text_vec.u8x16, x_e2_vec);
        uint8x16_t x_e280_cmp = vandq_u8(x_e2_cmp, vceqq_u8(text1, x_80_vec));
        uint8x16_t x_e280a8_cmp = vandq_u8(x_e280_cmp, vceqq_u8(text2, x_a8_vec));
        uint8x16_t x_e280a9_cmp = vandq_u8(x_e280_cmp, vceqq_u8(text2, x_a9_vec));
        uint8x16_t three_byte_cmp =
            vandq_u8(vorrq_u8(x_e280a8_cmp, x_e280a9_cmp), drop2_vec); // Ignore last two split matches

        // Quick presence check
        uint8x16_t combined_vec = vorrq_u8(one_byte_cmp, vorrq_u8(two_byte_cmp, three_byte_cmp));
        if (vmaxvq_u8(combined_vec)) {

            // Late mask extraction only when a match exists
            sz_u64_t one_byte_mask = sz_utf8_vreinterpretq_u8_u4_(one_byte_cmp);
            sz_u64_t two_mask = sz_utf8_vreinterpretq_u8_u4_(two_byte_cmp);
            sz_u64_t three_mask = sz_utf8_vreinterpretq_u8_u4_(three_byte_cmp);
            sz_u64_t combined_mask = one_byte_mask | two_mask | three_mask;

            int bit_index = sz_u64_ctz(combined_mask);
            sz_u64_t first_match_mask = (sz_u64_t)1 << bit_index;
            sz_size_t length_value = 1;
            length_value += (first_match_mask & (two_mask | three_mask)) != 0;
            length_value += (first_match_mask & three_mask) != 0;
            *matched_length = length_value;
            return text + (bit_index / 4);
        }
        text += 14;
        length -= 14;
    }

    return sz_utf8_find_newline_serial(text, length, matched_length);
}

SZ_PUBLIC sz_cptr_t sz_utf8_find_whitespace_neon(sz_cptr_t text, sz_size_t length, sz_size_t *matched_length) {

    sz_u128_vec_t text_vec;
    uint8x16_t t_vec = vdupq_n_u8('\t');
    uint8x16_t r_vec = vdupq_n_u8('\r');
    uint8x16_t x_20_vec = vdupq_n_u8(' ');
    uint8x16_t x_c2_vec = vdupq_n_u8(0xC2);
    uint8x16_t x_85_vec = vdupq_n_u8(0x85);
    uint8x16_t x_a0_vec = vdupq_n_u8(0xA0);
    uint8x16_t x_e1_vec = vdupq_n_u8(0xE1);
    uint8x16_t x_e2_vec = vdupq_n_u8(0xE2);
    uint8x16_t x_e3_vec = vdupq_n_u8(0xE3);
    uint8x16_t x_9a_vec = vdupq_n_u8(0x9A);
    uint8x16_t x_80_vec = vdupq_n_u8(0x80);
    uint8x16_t x_81_vec = vdupq_n_u8(0x81);
    uint8x16_t x_8d_vec = vdupq_n_u8(0x8D);
    uint8x16_t x_a8_vec = vdupq_n_u8(0xA8);
    uint8x16_t x_a9_vec = vdupq_n_u8(0xA9);
    uint8x16_t x_af_vec = vdupq_n_u8(0xAF);
    uint8x16_t x_9f_vec = vdupq_n_u8(0x9F);

    uint8x16_t drop1_vec = {0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
                            0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x00};
    uint8x16_t drop2_vec = {0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
                            0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x00, 0x00};

    while (length >= 16) {
        text_vec.u8x16 = vld1q_u8((sz_u8_t const *)text);

        // 1-byte matches
        uint8x16_t x_20_cmp = vceqq_u8(text_vec.u8x16, x_20_vec);
        uint8x16_t range_cmp = vandq_u8(vcgeq_u8(text_vec.u8x16, t_vec), vcleq_u8(text_vec.u8x16, r_vec));
        uint8x16_t one_byte_cmp = vorrq_u8(x_20_cmp, range_cmp);

        // 2-byte and 3-byte prefix indicators
        uint8x16_t x_c2_cmp = vandq_u8(vceqq_u8(text_vec.u8x16, x_c2_vec), drop1_vec);
        uint8x16_t x_e1_cmp = vandq_u8(vceqq_u8(text_vec.u8x16, x_e1_vec), drop2_vec);
        uint8x16_t x_e2_cmp = vandq_u8(vceqq_u8(text_vec.u8x16, x_e2_vec), drop2_vec);
        uint8x16_t x_e3_cmp = vandq_u8(vceqq_u8(text_vec.u8x16, x_e3_vec), drop2_vec);
        uint8x16_t prefix_byte_cmp = vorrq_u8(vorrq_u8(x_c2_cmp, x_e1_cmp), vorrq_u8(x_e2_cmp, x_e3_cmp));

        sz_u64_t one_byte_mask = sz_utf8_vreinterpretq_u8_u4_(one_byte_cmp);
        sz_u64_t prefix_byte_mask = sz_utf8_vreinterpretq_u8_u4_(prefix_byte_cmp);

        // Check for fast path: one-byte match before any prefix
        if (one_byte_mask) {
            if (prefix_byte_mask) {
                int first_one_byte_offset = sz_u64_ctz(one_byte_mask);
                int first_prefix_offset = sz_u64_ctz(prefix_byte_mask);
                if (first_one_byte_offset < first_prefix_offset) {
                    *matched_length = 1;
                    return text + (first_one_byte_offset / 4);
                }
            }
            else {
                int first_one_byte_offset = sz_u64_ctz(one_byte_mask);
                *matched_length = 1;
                return text + (first_one_byte_offset / 4);
            }
        }
        else if (!prefix_byte_mask) {
            text += 14;
            length -= 14;
            continue;
        }

        // 2-byte matches
        uint8x16_t text1 = vextq_u8(text_vec.u8x16, text_vec.u8x16, 1);
        uint8x16_t two_vec =
            vorrq_u8(vandq_u8(x_c2_cmp, vceqq_u8(text1, x_85_vec)), vandq_u8(x_c2_cmp, vceqq_u8(text1, x_a0_vec)));

        // 3-byte matches
        uint8x16_t text2 = vextq_u8(text_vec.u8x16, text_vec.u8x16, 2);
        uint8x16_t x_80_ge_cmp = vcgeq_u8(text2, x_80_vec);
        uint8x16_t x_8d_le_cmp = vcleq_u8(text2, x_8d_vec);

        uint8x16_t ogham_cmp = vandq_u8(x_e1_cmp, vandq_u8(vceqq_u8(text1, x_9a_vec), vceqq_u8(text2, x_80_vec)));
        uint8x16_t range_e280_cmp =
            vandq_u8(x_e2_cmp, vandq_u8(vceqq_u8(text1, x_80_vec), vandq_u8(x_80_ge_cmp, x_8d_le_cmp)));
        uint8x16_t line_cmp = vandq_u8(x_e2_cmp, vandq_u8(vceqq_u8(text1, x_80_vec), vceqq_u8(text2, x_a8_vec)));
        uint8x16_t paragraph_cmp = vandq_u8(x_e2_cmp, vandq_u8(vceqq_u8(text1, x_80_vec), vceqq_u8(text2, x_a9_vec)));
        uint8x16_t nnbsp_cmp = vandq_u8(x_e2_cmp, vandq_u8(vceqq_u8(text1, x_80_vec), vceqq_u8(text2, x_af_vec)));
        uint8x16_t mmsp_cmp = vandq_u8(x_e2_cmp, vandq_u8(vceqq_u8(text1, x_81_vec), vceqq_u8(text2, x_9f_vec)));
        uint8x16_t ideographic_vec = vandq_u8(x_e3_cmp, vandq_u8(vceqq_u8(text1, x_80_vec), vceqq_u8(text2, x_80_vec)));
        uint8x16_t three_vec =
            vorrq_u8(vorrq_u8(vorrq_u8(ogham_cmp, range_e280_cmp), vorrq_u8(line_cmp, paragraph_cmp)),
                     vorrq_u8(vorrq_u8(nnbsp_cmp, mmsp_cmp), ideographic_vec));

        sz_u64_t two_byte_mask = sz_utf8_vreinterpretq_u8_u4_(two_vec);
        sz_u64_t three_byte_mask = sz_utf8_vreinterpretq_u8_u4_(three_vec);
        sz_u64_t combined_mask = one_byte_mask | two_byte_mask | three_byte_mask;

        if (combined_mask) {
            int bit_index = sz_u64_ctz(combined_mask);
            sz_u64_t first_match_mask = (sz_u64_t)1 << bit_index;
            sz_size_t length_value = 1;
            length_value += (first_match_mask & (two_byte_mask | three_byte_mask)) != 0;
            length_value += (first_match_mask & three_byte_mask) != 0;
            *matched_length = length_value;
            return text + (bit_index / 4);
        }
        text += 14;
        length -= 14;
    }

    return sz_utf8_find_whitespace_serial(text, length, matched_length);
}

SZ_PUBLIC sz_size_t sz_utf8_count_neon(sz_cptr_t text, sz_size_t length) {
    sz_u128_vec_t text_vec, headers_vec, continuation_vec;
    uint8x16_t continuation_mask_vec = vdupq_n_u8(0xC0);
    uint8x16_t continuation_pattern_vec = vdupq_n_u8(0x80);
    sz_u8_t const *text_u8 = (sz_u8_t const *)text;
    uint64x2_t char_count_vec = vdupq_n_u64(0);
    while (length >= 16) {
        text_vec.u8x16 = vld1q_u8(text_u8);
        headers_vec.u8x16 = vandq_u8(text_vec.u8x16, continuation_mask_vec);
        continuation_vec.u8x16 = vceqq_u8(headers_vec.u8x16, continuation_pattern_vec);
        // Convert 0xFF/0x00 into 1/0 and sum.
        uint8x16_t start_flags = vshrq_n_u8(vmvnq_u8(continuation_vec.u8x16), 7);
        uint16x8_t sum16 = vpaddlq_u8(start_flags);
        uint32x4_t sum32 = vpaddlq_u16(sum16);
        uint64x2_t sum64 = vpaddlq_u32(sum32);
        char_count_vec = vaddq_u64(char_count_vec, sum64);
        text_u8 += 16;
        length -= 16;
    }

    sz_size_t char_count = vgetq_lane_u64(char_count_vec, 0) + vgetq_lane_u64(char_count_vec, 1);
    if (length) char_count += sz_utf8_count_serial((sz_cptr_t)text_u8, length);
    return char_count;
}

SZ_PUBLIC sz_cptr_t sz_utf8_find_nth_neon(sz_cptr_t text, sz_size_t length, sz_size_t n) {
    // TODO: Implement a NEON-accelerated version of sz_utf8_find_nth in absense of PDEP instruction.
    return sz_utf8_find_nth_serial(text, length, n);
}

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#endif // SZ_USE_NEON

#pragma endregion // NEON Implementation

#pragma region SVE2 Implementation
#if SZ_USE_SVE2
#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("+sve+sve2"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("+sve+sve2")
#endif

SZ_PUBLIC sz_size_t sz_utf8_count_sve2(sz_cptr_t text, sz_size_t length) {
    sz_u8_t const *text_u8 = (sz_u8_t const *)text;
    sz_size_t const step = svcntb();
    sz_size_t char_count = 0;

    // Count bytes that are NOT continuation bytes: (byte & 0xC0) != 0x80
    for (sz_size_t offset = 0; offset < length; offset += step) {
        svbool_t pg = svwhilelt_b8((sz_u64_t)offset, (sz_u64_t)length);
        svuint8_t text_vec = svld1_u8(pg, text_u8 + offset);
        svbool_t is_start = svcmpne_n_u8(pg, svand_n_u8_x(pg, text_vec, 0xC0), 0x80);
        char_count += svcntp_b8(pg, is_start);
    }
    return char_count;
}

SZ_PUBLIC sz_cptr_t sz_utf8_find_nth_sve2(sz_cptr_t text, sz_size_t length, sz_size_t n) {
    sz_u8_t const *text_u8 = (sz_u8_t const *)text;
    sz_size_t const step = svcntb();

    // Find character start bytes: (byte & 0xC0) != 0x80
    for (sz_size_t offset = 0; offset < length; offset += step) {
        svbool_t pg = svwhilelt_b8((sz_u64_t)offset, (sz_u64_t)length);
        svuint8_t text_vec = svld1_u8(pg, text_u8 + offset);
        svbool_t is_start = svcmpne_n_u8(pg, svand_n_u8_x(pg, text_vec, 0xC0), 0x80);
        sz_size_t start_count = svcntp_b8(pg, is_start);

        // When we find the chunk containing the Nth character, let serial handle extraction.
        // There is no `svcompact_u8` in SVE2 (only 32/64-bit variants), and no direct instruction
        // to find the position of the Nth set bit in a predicate.
        if (n < start_count) return sz_utf8_find_nth_serial((sz_cptr_t)(text_u8 + offset), length - offset, n);
        n -= start_count;
    }

    return SZ_NULL_CHAR;
}

SZ_PUBLIC sz_cptr_t sz_utf8_find_newline_sve2(sz_cptr_t text, sz_size_t length, sz_size_t *matched_length) {
    sz_u8_t const *text_u8 = (sz_u8_t const *)text;
    sz_size_t const step = svcntb();

    // SVE2 kernels are a bit different from both NEON and Ice Lake due to presense of
    // a few very convenient and cheap instructions. Most importantly, we have `svmatch` instruction
    // that can match against a set of bytes in one go, similar to many invocations of `vceqq` in NEON
    // with subsequent mask combination.
    svuint8_t prefix_byte_set =
        svdupq_n_u8('\n', '\v', '\f', '\r', 0xC2, 0xE2, '\n', '\n', '\n', '\n', '\n', '\n', '\n', '\n', '\n', '\n');
    svuint8_t one_byte_set =
        svdupq_n_u8('\n', '\v', '\f', '\r', '\n', '\n', '\n', '\n', '\n', '\n', '\n', '\n', '\n', '\n', '\n', '\n');
    svuint8_t zeros = svdup_n_u8(0);

    for (sz_size_t offset = 0; offset < length;) {
        svbool_t pg = svwhilelt_b8_u64((sz_u64_t)offset, (sz_u64_t)length);
        svuint8_t text0 = svld1_u8(pg, text_u8 + offset);

        // Fast rejection: any potential first byte?
        if (!svptest_any(pg, svmatch_u8(pg, text0, prefix_byte_set))) {
            offset += step;
            continue;
        }

        svbool_t pg_1 = svwhilelt_b8_u64((sz_u64_t)offset + 1, (sz_u64_t)length);
        svbool_t pg_2 = svwhilelt_b8_u64((sz_u64_t)offset + 2, (sz_u64_t)length);
        svuint8_t text1 = svext_u8(text0, zeros, 1);
        svuint8_t text2 = svext_u8(text0, zeros, 2);

        // 1-byte newlines: MATCH against {\n, \v, \f, \r}
        svbool_t one_byte_mask = svmatch_u8(pg, text0, one_byte_set);

        // 2-byte: \r\n, NEL (0xC2 0x85)
        svbool_t rn_mask = svand_b_z(pg_1, svcmpeq_n_u8(pg, text0, '\r'), svcmpeq_n_u8(pg, text1, '\n'));
        svbool_t x_c285_mask = svand_b_z(pg_1, svcmpeq_n_u8(pg, text0, 0xC2), svcmpeq_n_u8(pg, text1, 0x85));
        svbool_t two_byte_mask = svorr_b_z(pg_1, rn_mask, x_c285_mask);

        // 3-byte: 0xE2 0x80 + {0xA8, 0xA9} - use OR instead of MATCH for 2-element set
        svbool_t x_e280_mask = svand_b_z(pg_2, svcmpeq_n_u8(pg, text0, 0xE2), svcmpeq_n_u8(pg, text1, 0x80));
        svbool_t third_byte_mask =
            svand_b_z(pg_2, x_e280_mask, svorr_b_z(pg, svcmpeq_n_u8(pg, text2, 0xA8), svcmpeq_n_u8(pg, text2, 0xA9)));

        svbool_t combined_mask = svorr_b_z(pg, one_byte_mask, svorr_b_z(pg, two_byte_mask, third_byte_mask));
        if (svptest_any(pg, combined_mask)) {
            sz_size_t pos = svcntp_b8(pg, svbrkb_b_z(pg, combined_mask));
            svbool_t at_pos = svcmpeq_n_u8(svptrue_b8(), svindex_u8(0, 1), (sz_u8_t)pos);

            if (svptest_any(at_pos, third_byte_mask)) { *matched_length = 3; }
            else if (svptest_any(at_pos, two_byte_mask)) { *matched_length = 2; }
            else { *matched_length = 1; }

            return (sz_cptr_t)(text_u8 + offset + pos);
        }
        else { offset += step - 2; }
    }

    *matched_length = 0;
    return SZ_NULL_CHAR;
}

SZ_PUBLIC sz_cptr_t sz_utf8_find_whitespace_sve2(sz_cptr_t text, sz_size_t length, sz_size_t *matched_length) {
    sz_u8_t const *text_u8 = (sz_u8_t const *)text;
    sz_size_t const step = svcntb();

    // Character sets for MATCH (DUPQ replicates 128-bit pattern, no stack/loads)
    svuint8_t any_whitespace_byte_set =
        svdupq_n_u8(' ', '\t', '\n', '\v', '\f', '\r', 0xC2, 0xE1, 0xE2, 0xE3, ' ', ' ', ' ', ' ', ' ', ' ');
    svuint8_t one_byte_set =
        svdupq_n_u8(' ', '\t', '\n', '\v', '\f', '\r', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ');
    svuint8_t multi_byte_prefix_set =
        svdupq_n_u8(0xC2, 0xE1, 0xE2, 0xE3, 0xC2, 0xC2, 0xC2, 0xC2, 0xC2, 0xC2, 0xC2, 0xC2, 0xC2, 0xC2, 0xC2, 0xC2);
    svuint8_t zeros = svdup_n_u8(0);

    for (sz_size_t offset = 0; offset < length;) {
        svbool_t pg = svwhilelt_b8_u64((sz_u64_t)offset, (sz_u64_t)length);
        svuint8_t text0 = svld1_u8(pg, text_u8 + offset);

        // Fast rejection: skip if no whitespace-related bytes at all
        if (!svptest_any(pg, svmatch_u8(pg, text0, any_whitespace_byte_set))) {
            offset += step;
            continue;
        }

        // 1-byte whitespace: space, tab, newlines
        svbool_t one_byte_mask = svmatch_u8(pg, text0, one_byte_set);

        // Fast path: if we have a 1-byte match and it comes before any multi-byte prefix, return it
        if (svptest_any(pg, one_byte_mask)) {
            svbool_t multi_byte_prefix_mask = svmatch_u8(pg, text0, multi_byte_prefix_set);
            if (!svptest_any(pg, multi_byte_prefix_mask)) {
                // No multi-byte prefix in this chunk - return first 1-byte match immediately
                sz_size_t pos = svcntp_b8(pg, svbrkb_b_z(pg, one_byte_mask));
                *matched_length = 1;
                return (sz_cptr_t)(text_u8 + offset + pos);
            }
            // Check if 1-byte match comes before any multi-byte prefix
            sz_size_t one_byte_pos = svcntp_b8(pg, svbrkb_b_z(pg, one_byte_mask));
            sz_size_t multi_byte_prefix_pos = svcntp_b8(pg, svbrkb_b_z(pg, multi_byte_prefix_mask));
            if (one_byte_pos < multi_byte_prefix_pos) {
                *matched_length = 1;
                return (sz_cptr_t)(text_u8 + offset + one_byte_pos);
            }
        }

        // Slow path: need to check multi-byte sequences
        svbool_t pg_1 = svwhilelt_b8_u64((sz_u64_t)offset + 1, (sz_u64_t)length);
        svbool_t pg_2 = svwhilelt_b8_u64((sz_u64_t)offset + 2, (sz_u64_t)length);
        svuint8_t text1 = svext_u8(text0, zeros, 1);
        svuint8_t text2 = svext_u8(text0, zeros, 2);

        // 2-byte: C2 + {85, A0} (NEL, NBSP)
        svbool_t x_c2_mask = svcmpeq_n_u8(pg, text0, 0xC2);
        svbool_t x_85_mask = svcmpeq_n_u8(pg, text1, 0x85);
        svbool_t x_a0_mask = svcmpeq_n_u8(pg, text1, 0xA0);
        svbool_t two_byte_mask = svand_b_z(pg_1, x_c2_mask, svorr_b_z(pg, x_85_mask, x_a0_mask));

        // 3-byte: E1 9A 80 (Ogham Space Mark)
        svbool_t ogham_mask =
            svand_b_z(pg_2, svand_b_z(pg_2, svcmpeq_n_u8(pg, text0, 0xE1), svcmpeq_n_u8(pg, text1, 0x9A)),
                      svcmpeq_n_u8(pg, text2, 0x80));

        // 3-byte: E2 80 XX - various Unicode spaces
        svbool_t x_e2_mask = svcmpeq_n_u8(pg, text0, 0xE2);
        svbool_t x_80_mask = svcmpeq_n_u8(pg, text1, 0x80);
        svbool_t x_e280_mask = svand_b_z(pg_2, x_e2_mask, x_80_mask);
        // U+2000 to U+200D: E2 80 [80-8D]
        svbool_t range_e280_mask =
            svand_b_z(pg_2, x_e280_mask, svand_b_z(pg, svcmpge_n_u8(pg, text2, 0x80), svcmple_n_u8(pg, text2, 0x8D)));
        // U+2028: E2 80 A8 (LINE SEPARATOR)
        svbool_t line_mask = svand_b_z(pg_2, x_e280_mask, svcmpeq_n_u8(pg, text2, 0xA8));
        // U+2029: E2 80 A9 (PARAGRAPH SEPARATOR)
        svbool_t paragraph_mask = svand_b_z(pg_2, x_e280_mask, svcmpeq_n_u8(pg, text2, 0xA9));
        // U+202F: E2 80 AF (NARROW NO-BREAK SPACE)
        svbool_t nnbsp_mask = svand_b_z(pg_2, x_e280_mask, svcmpeq_n_u8(pg, text2, 0xAF));
        // U+205F: E2 81 9F (MEDIUM MATHEMATICAL SPACE)
        svbool_t mmsp_mask =
            svand_b_z(pg_2, svand_b_z(pg_2, x_e2_mask, svcmpeq_n_u8(pg, text1, 0x81)), svcmpeq_n_u8(pg, text2, 0x9F));

        // 3-byte: E3 80 80 (IDEOGRAPHIC SPACE)
        svbool_t ideographic_mask =
            svand_b_z(pg_2, svand_b_z(pg_2, svcmpeq_n_u8(pg, text0, 0xE3), svcmpeq_n_u8(pg, text1, 0x80)),
                      svcmpeq_n_u8(pg, text2, 0x80));

        svbool_t three_byte_mask = svorr_b_z(
            pg_2,
            svorr_b_z(pg_2, svorr_b_z(pg_2, ogham_mask, range_e280_mask), svorr_b_z(pg_2, line_mask, paragraph_mask)),
            svorr_b_z(pg_2, svorr_b_z(pg_2, nnbsp_mask, mmsp_mask), ideographic_mask));
        svbool_t combined_mask = svorr_b_z(pg, one_byte_mask, svorr_b_z(pg, two_byte_mask, three_byte_mask));

        if (svptest_any(pg, combined_mask)) {
            sz_size_t pos = svcntp_b8(pg, svbrkb_b_z(pg, combined_mask));
            svbool_t at_pos = svcmpeq_n_u8(svptrue_b8(), svindex_u8(0, 1), (sz_u8_t)pos);

            if (svptest_any(at_pos, three_byte_mask)) { *matched_length = 3; }
            else if (svptest_any(at_pos, two_byte_mask)) { *matched_length = 2; }
            else { *matched_length = 1; }

            return (sz_cptr_t)(text_u8 + offset + pos);
        }
        else { offset += step - 2; }
    }

    *matched_length = 0;
    return SZ_NULL_CHAR;
}

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#endif // SZ_USE_SVE2

#pragma endregion // SVE2 Implementation

#pragma region Dynamic Dispatch

#if !SZ_DYNAMIC_DISPATCH

SZ_DYNAMIC sz_size_t sz_utf8_count(sz_cptr_t text, sz_size_t length) {
#if SZ_USE_ICE
    return sz_utf8_count_ice(text, length);
#elif SZ_USE_HASWELL
    return sz_utf8_count_haswell(text, length);
#elif SZ_USE_SVE2
    return sz_utf8_count_sve2(text, length);
#elif SZ_USE_NEON
    return sz_utf8_count_neon(text, length);
#else
    return sz_utf8_count_serial(text, length);
#endif
}

SZ_DYNAMIC sz_cptr_t sz_utf8_find_nth(sz_cptr_t text, sz_size_t length, sz_size_t n) {
#if SZ_USE_ICE
    return sz_utf8_find_nth_ice(text, length, n);
#elif SZ_USE_HASWELL
    return sz_utf8_find_nth_haswell(text, length, n);
#elif SZ_USE_SVE2
    return sz_utf8_find_nth_sve2(text, length, n);
#elif SZ_USE_NEON
    return sz_utf8_find_nth_neon(text, length, n);
#else
    return sz_utf8_find_nth_serial(text, length, n);
#endif
}

SZ_DYNAMIC sz_cptr_t sz_utf8_find_newline(sz_cptr_t text, sz_size_t length, sz_size_t *matched_length) {
#if SZ_USE_ICE
    return sz_utf8_find_newline_ice(text, length, matched_length);
#elif SZ_USE_HASWELL
    return sz_utf8_find_newline_haswell(text, length, matched_length);
#elif SZ_USE_SVE2
    return sz_utf8_find_newline_sve2(text, length, matched_length);
#elif SZ_USE_NEON
    return sz_utf8_find_newline_neon(text, length, matched_length);
#else
    return sz_utf8_find_newline_serial(text, length, matched_length);
#endif
}

SZ_DYNAMIC sz_cptr_t sz_utf8_find_whitespace(sz_cptr_t text, sz_size_t length, sz_size_t *matched_length) {
#if SZ_USE_ICE
    return sz_utf8_find_whitespace_ice(text, length, matched_length);
#elif SZ_USE_HASWELL
    return sz_utf8_find_whitespace_haswell(text, length, matched_length);
#elif SZ_USE_SVE2
    return sz_utf8_find_whitespace_sve2(text, length, matched_length);
#elif SZ_USE_NEON
    return sz_utf8_find_whitespace_neon(text, length, matched_length);
#else
    return sz_utf8_find_whitespace_serial(text, length, matched_length);
#endif
}

#endif // !SZ_DYNAMIC_DISPATCH

#pragma endregion // Dynamic Dispatch

#ifdef __cplusplus
}
#endif

#endif // STRINGZILLA_UTF8_H_
