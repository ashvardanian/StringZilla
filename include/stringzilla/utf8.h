/**
 *  @brief  Hardware-accelerated UTF-8 text processing utilities.
 *  @file   utf8.h
 *  @author Ash Vardanian
 *
 *  Includes core APIs:
 *
 *  - `sz_utf8_valid` - validate UTF-8 encoding correctness
 *  - `sz_utf8_find_nth` - skip to Nth UTF-8 character (unified skip/count API)
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
 *    - U+3000 @c 0xE38080 (IDEOGRAPHIC SPACE, common in CJK texts)
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

/**
 *  @brief  Skip forward to the Nth UTF-8 character, returning the pointer and character count.
 *
 *  This is a unified API that handles both character counting and skipping:
 *  - To count all characters: pass max_chars = SIZE_MAX
 *  - To skip N characters: pass max_chars = N
 *  - Always returns BOTH pointer position and character count
 *
 *  @param[in] text String to be scanned.
 *  @param[in] length Number of bytes in the string.
 *  @param[in] max_chars Maximum number of UTF-8 characters to skip (SIZE_MAX for "count all").
 *  @param[out] chars_skipped Actual number of UTF-8 characters skipped/counted.
 *  @return Pointer to position after skipping (use for slicing), or end of string if max_chars exceeded.
 *
 *  @example Count all characters:
 *  @code
 *      size_t char_count;
 *      sz_utf8_find_nth(text, length, SIZE_MAX, &char_count);
 *      printf("String has %zu characters\n", char_count);
 *  @endcode
 *
 *  @example Skip to character 1000 (e.g., pagination):
 *  @code
 *      size_t actual_chars;
 *      char const *pos = sz_utf8_find_nth(text, length, 1000, &actual_chars);
 *      // Continue processing from pos
 *  @endcode
 *
 *  @example Truncate to 280 characters (Twitter-style):
 *  @code
 *      size_t actual_chars;
 *      char const *end = sz_utf8_find_nth(text, length, 280, &actual_chars);
 *      size_t truncated_bytes = end - text;
 *  @endcode
 */
SZ_DYNAMIC sz_cptr_t sz_utf8_find_nth(sz_cptr_t text, sz_size_t length, sz_size_t max_chars, sz_size_t *chars_skipped);

/**
 *  @brief  Validate UTF-8 encoding correctness.
 *
 *  Checks for:
 *  - Correct continuation byte patterns (10xxxxxx after lead bytes)
 *  - No overlong encodings (e.g., 2-byte encoding of ASCII char)
 *  - No invalid code points (surrogates U+D800-U+DFFF, values > U+10FFFF)
 *  - Proper sequence lengths (1-4 bytes only)
 *
 *  @param[in] text String to validate.
 *  @param[in] length Number of bytes in the string.
 *  @return 1 if valid UTF-8, 0 if invalid.
 *
 *  @example Validate user input:
 *  @code
 *      if (!sz_utf8_valid(user_input, input_len)) {
 *          fprintf(stderr, "Invalid UTF-8 encoding\n");
 *          return ERROR;
 *      }
 *  @endcode
 */
SZ_DYNAMIC sz_bool_t sz_utf8_valid(sz_cptr_t text, sz_size_t length);

#pragma endregion

#pragma region Platform-Specific Backends

// Serial (portable) implementations
SZ_PUBLIC sz_cptr_t sz_utf8_find_newline_serial(sz_cptr_t text, sz_size_t length, sz_size_t *matched_length);
SZ_PUBLIC sz_cptr_t sz_utf8_find_whitespace_serial(sz_cptr_t text, sz_size_t length, sz_size_t *matched_length);
SZ_PUBLIC sz_cptr_t sz_utf8_find_nth_serial(sz_cptr_t text, sz_size_t length, sz_size_t max_chars,
                                            sz_size_t *chars_skipped);
SZ_PUBLIC sz_bool_t sz_utf8_valid_serial(sz_cptr_t text, sz_size_t length);

// Haswell (AVX2) implementations
SZ_PUBLIC sz_cptr_t sz_utf8_find_newline_haswell(sz_cptr_t text, sz_size_t length, sz_size_t *matched_length);
SZ_PUBLIC sz_cptr_t sz_utf8_find_whitespace_haswell(sz_cptr_t text, sz_size_t length, sz_size_t *matched_length);
SZ_PUBLIC sz_cptr_t sz_utf8_find_nth_haswell(sz_cptr_t text, sz_size_t length, sz_size_t max_chars,
                                             sz_size_t *chars_skipped);
SZ_PUBLIC sz_bool_t sz_utf8_valid_haswell(sz_cptr_t text, sz_size_t length);

// Ice Lake (AVX-512) implementations
SZ_PUBLIC sz_cptr_t sz_utf8_find_newline_ice(sz_cptr_t text, sz_size_t length, sz_size_t *matched_length);
SZ_PUBLIC sz_cptr_t sz_utf8_find_whitespace_ice(sz_cptr_t text, sz_size_t length, sz_size_t *matched_length);
SZ_PUBLIC sz_cptr_t sz_utf8_find_nth_ice(sz_cptr_t text, sz_size_t length, sz_size_t max_chars,
                                         sz_size_t *chars_skipped);
SZ_PUBLIC sz_bool_t sz_utf8_valid_ice(sz_cptr_t text, sz_size_t length);

// NEON (ARM) implementations - fall back to serial
SZ_PUBLIC sz_cptr_t sz_utf8_find_newline_neon(sz_cptr_t text, sz_size_t length, sz_size_t *matched_length);
SZ_PUBLIC sz_cptr_t sz_utf8_find_whitespace_neon(sz_cptr_t text, sz_size_t length, sz_size_t *matched_length);

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

SZ_PUBLIC sz_cptr_t sz_utf8_find_nth_serial(sz_cptr_t text, sz_size_t length, sz_size_t max_chars,
                                            sz_size_t *chars_skipped) {
    // Count UTF-8 characters by counting non-continuation bytes
    // Continuation bytes have pattern 10xxxxxx (i.e., (byte & 0xC0) == 0x80)
    sz_u8_t const *text_u8 = (sz_u8_t const *)text;
    sz_u8_t const *end_u8 = text_u8 + length;
    sz_size_t char_count = 0;

    while (text_u8 < end_u8 && char_count < max_chars) {
        // Count this byte if it's NOT a continuation byte
        if ((*text_u8 & 0xC0) != 0x80) { char_count++; }
        text_u8++;
    }

    *chars_skipped = char_count;
    return (sz_cptr_t)text_u8;
}

SZ_PUBLIC sz_bool_t sz_utf8_valid_serial(sz_cptr_t text, sz_size_t length) {
    sz_u8_t const *text_u8 = (sz_u8_t const *)text;
    sz_u8_t const *end_u8 = text_u8 + length;

    while (text_u8 < end_u8) {
        sz_u8_t byte1 = *text_u8;

        // 1-byte sequence (0x00-0x7F)
        if (byte1 <= 0x7F) {
            text_u8 += 1;
            continue;
        }

        // 2-byte sequence (0xC2-0xDF)
        if (byte1 >= 0xC2 && byte1 <= 0xDF) {
            if (text_u8 + 1 >= end_u8) return sz_false_k;
            sz_u8_t byte2 = text_u8[1];
            if ((byte2 & 0xC0) != 0x80) return sz_false_k; // Invalid continuation
            text_u8 += 2;
            continue;
        }

        // 3-byte sequence (0xE0-0xEF)
        if (byte1 >= 0xE0 && byte1 <= 0xEF) {
            if (text_u8 + 2 >= end_u8) return sz_false_k;
            sz_u8_t byte2 = text_u8[1];
            sz_u8_t byte3 = text_u8[2];
            if ((byte2 & 0xC0) != 0x80 || (byte3 & 0xC0) != 0x80) return sz_false_k;

            // Check for overlong encodings and surrogates
            if (byte1 == 0xE0 && byte2 < 0xA0) return sz_false_k;  // Overlong
            if (byte1 == 0xED && byte2 >= 0xA0) return sz_false_k; // Surrogate (U+D800-U+DFFF)

            text_u8 += 3;
            continue;
        }

        // 4-byte sequence (0xF0-0xF4)
        if (byte1 >= 0xF0 && byte1 <= 0xF4) {
            if (text_u8 + 3 >= end_u8) return sz_false_k;
            sz_u8_t byte2 = text_u8[1];
            sz_u8_t byte3 = text_u8[2];
            sz_u8_t byte4 = text_u8[3];
            if ((byte2 & 0xC0) != 0x80 || (byte3 & 0xC0) != 0x80 || (byte4 & 0xC0) != 0x80) return sz_false_k;

            // Check for overlong and out-of-range
            if (byte1 == 0xF0 && byte2 < 0x90) return sz_false_k;  // Overlong
            if (byte1 == 0xF4 && byte2 >= 0x90) return sz_false_k; // > U+10FFFF

            text_u8 += 4;
            continue;
        }

        // Invalid lead byte
        return sz_false_k;
    }

    return sz_true_k;
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
    sz_u512_vec_t xc2_vec, x85_vec, xe2_vec, x80_vec, xa8_vec, xa9_vec;
    xc2_vec.zmm = _mm512_set1_epi8('\xC2');
    x85_vec.zmm = _mm512_set1_epi8('\x85');
    xe2_vec.zmm = _mm512_set1_epi8('\xE2');
    x80_vec.zmm = _mm512_set1_epi8('\x80');
    xa8_vec.zmm = _mm512_set1_epi8('\xA8');
    xa9_vec.zmm = _mm512_set1_epi8('\xA9');

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
        __mmask64 xc2_mask = _mm512_cmpeq_epi8_mask(text_vec.zmm, xc2_vec.zmm);
        __mmask64 x85_mask = _mm512_cmpeq_epi8_mask(text_vec.zmm, x85_vec.zmm);
        __mmask64 xe2_mask = _mm512_cmpeq_epi8_mask(text_vec.zmm, xe2_vec.zmm);
        __mmask64 x80_mask = _mm512_cmpeq_epi8_mask(text_vec.zmm, x80_vec.zmm);
        __mmask64 xa8_mask = _mm512_cmpeq_epi8_mask(text_vec.zmm, xa8_vec.zmm);
        __mmask64 xa9_mask = _mm512_cmpeq_epi8_mask(text_vec.zmm, xa9_vec.zmm);

        // 2-byte matches
        __mmask64 rn_mask = _kand_mask64(r_mask, _kshiftri_mask64(n_mask, 1));
        __mmask64 xc285_mask = _kand_mask64(xc2_mask, _kshiftri_mask64(x85_mask, 1));
        sz_u64_t two_byte_mask = _cvtmask64_u64(_kor_mask64(rn_mask, xc285_mask));

        // 3-byte matches
        __mmask64 xe280_mask = _kand_mask64(xe2_mask, _kshiftri_mask64(x80_mask, 1));
        __mmask64 e280a8_mask = _kand_mask64(xe280_mask, _kshiftri_mask64(xa8_mask, 2));
        __mmask64 e280a9_mask = _kand_mask64(xe280_mask, _kshiftri_mask64(xa9_mask, 2));
        sz_u64_t three_byte_mask = _cvtmask64_u64(_kor_mask64(e280a8_mask, e280a9_mask));

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
    sz_u512_vec_t t_vec, r_vec, x20_vec;
    t_vec.zmm = _mm512_set1_epi8('\t');
    r_vec.zmm = _mm512_set1_epi8('\r');
    x20_vec.zmm = _mm512_set1_epi8(' ');

    // We also need to match the 2-byte characters 0xC285 (NEL) and 0xC2A0 (NBSP),
    sz_u512_vec_t xc2_vec, x85_vec, xa0_vec;
    xc2_vec.zmm = _mm512_set1_epi8('\xC2');
    x85_vec.zmm = _mm512_set1_epi8('\x85');
    xa0_vec.zmm = _mm512_set1_epi8('\xA0');

    // We also need to match 3-byte ogham space mark 0xE19A80 (OGHAM SPACE MARK),
    // a range of 3-byte characters from 0xE28080 to 0xE2808D (various spaces),
    // U+202F (0xE280AF), U+205F (0xE2819F),
    // U+2028 (0xE280A8) LINE SEPARATOR, U+2029 (0xE280A9) PARAGRAPH SEPARATOR,
    // and the 3-byte ideographic space 0xE38080 (IDEOGRAPHIC SPACE).
    sz_u512_vec_t xe1_vec, xe2_vec, xe3_vec,         // ? possible first byte values
        x9a_vec, x80_vec, x81_vec,                   // ? possible second byte values
        x8d_vec, xa8_vec, xa9_vec, xaf_vec, x9f_vec; // ? third byte values for ranges and specific matches
    xe1_vec.zmm = _mm512_set1_epi8('\xE1');
    xe2_vec.zmm = _mm512_set1_epi8('\xE2');
    xe3_vec.zmm = _mm512_set1_epi8('\xE3');
    x9a_vec.zmm = _mm512_set1_epi8('\x9A');
    x80_vec.zmm = _mm512_set1_epi8('\x80');
    x81_vec.zmm = _mm512_set1_epi8('\x81');
    x8d_vec.zmm = _mm512_set1_epi8('\x8D');
    xa8_vec.zmm = _mm512_set1_epi8('\xA8');
    xa9_vec.zmm = _mm512_set1_epi8('\xA9');
    xaf_vec.zmm = _mm512_set1_epi8('\xAF');
    x9f_vec.zmm = _mm512_set1_epi8('\x9F');

    // We check 64 bytes of data at once, but only step forward by 62 bytes for split-register matches.
    sz_u512_vec_t text_vec;
    while (length >= 64) {
        text_vec.zmm = _mm512_loadu_epi8(text);

        // 1-byte indicators & matches
        // Range [9,13] covers \t, \n, \v, \f, \r
        __mmask64 x20_mask = _mm512_cmpeq_epi8_mask(text_vec.zmm, x20_vec.zmm);
        __mmask64 t_mask = _mm512_cmpge_epu8_mask(text_vec.zmm, t_vec.zmm);
        __mmask64 r_mask = _mm512_cmple_epu8_mask(text_vec.zmm, r_vec.zmm);
        sz_u64_t one_byte_mask = _cvtmask64_u64(_kor_mask64(x20_mask, _kand_mask64(t_mask, r_mask)));

        // Instead of immediately checking for 2-byte and 3-byte matches with a ridiculous number of masks and
        // comparisons, let's define a "fast path" for following cases:
        // - no whitespaces are found in the range
        // - a one-byte match comes before any possible prefix byte of a multi-byte match
        __mmask64 xc2_mask = _mm512_mask_cmpeq_epi8_mask(0x7FFFFFFFFFFFFFFF, text_vec.zmm, xc2_vec.zmm);
        __mmask64 xe1_mask = _mm512_mask_cmpeq_epi8_mask(0x3FFFFFFFFFFFFFFF, text_vec.zmm, xe1_vec.zmm);
        __mmask64 xe2_mask = _mm512_mask_cmpeq_epi8_mask(0x3FFFFFFFFFFFFFFF, text_vec.zmm, xe2_vec.zmm);
        __mmask64 xe3_mask = _mm512_mask_cmpeq_epi8_mask(0x3FFFFFFFFFFFFFFF, text_vec.zmm, xe3_vec.zmm);
        sz_u64_t prefix_mask =
            _cvtmask64_u64(_kor_mask64(_kor_mask64(xc2_mask, xe1_mask), _kor_mask64(xe2_mask, xe3_mask)));

        // Check if we matched the "fast path"
        if (one_byte_mask) {
            if (prefix_mask) {
                int first_one_byte_offset = sz_u64_ctz(one_byte_mask);
                int first_prefix_offset = sz_u64_ctz(prefix_mask);
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
        __mmask64 x85_mask = _mm512_cmpeq_epi8_mask(text_vec.zmm, x85_vec.zmm);
        __mmask64 xa0_mask = _mm512_cmpeq_epi8_mask(text_vec.zmm, xa0_vec.zmm);
        __mmask64 xc285_mask = _kand_mask64(xc2_mask, _kshiftri_mask64(x85_mask, 1)); // U+0085 NEL
        __mmask64 xc2a0_mask = _kand_mask64(xc2_mask, _kshiftri_mask64(xa0_mask, 1)); // U+00A0 NBSP
        sz_u64_t two_byte_mask = _cvtmask64_u64(_kor_mask64(xc285_mask, xc2a0_mask));

        // 3-byte indicators suffixes
        __mmask64 x9a_mask = _mm512_cmpeq_epi8_mask(text_vec.zmm, x9a_vec.zmm);
        __mmask64 x80_mask = _mm512_cmpeq_epi8_mask(text_vec.zmm, x80_vec.zmm);
        __mmask64 x81_mask = _mm512_cmpeq_epi8_mask(text_vec.zmm, x81_vec.zmm);
        __mmask64 x80_ge_mask = _mm512_cmpge_epu8_mask(text_vec.zmm, x80_vec.zmm);
        __mmask64 x8d_le_mask = _mm512_cmple_epu8_mask(text_vec.zmm, x8d_vec.zmm);
        __mmask64 xa8_mask = _mm512_cmpeq_epi8_mask(text_vec.zmm, xa8_vec.zmm);
        __mmask64 xa9_mask = _mm512_cmpeq_epi8_mask(text_vec.zmm, xa9_vec.zmm);
        __mmask64 xaf_mask = _mm512_cmpeq_epi8_mask(text_vec.zmm, xaf_vec.zmm);
        __mmask64 x9f_mask = _mm512_cmpeq_epi8_mask(text_vec.zmm, x9f_vec.zmm);

        // 3-byte matches
        __mmask64 ogham_mask =
            _kand_mask64(xe1_mask, _kand_mask64(_kshiftri_mask64(x9a_mask, 1), _kshiftri_mask64(x80_mask, 2)));
        // U+2000 to U+200D: E2 80 [80-8D]
        __mmask64 range_e280_mask = _kand_mask64(
            xe2_mask, _kand_mask64(_kshiftri_mask64(x80_mask, 1),
                                   _kand_mask64(_kshiftri_mask64(x80_ge_mask, 2), _kshiftri_mask64(x8d_le_mask, 2))));
        // U+202F: E2 80 AF
        __mmask64 u202f_mask =
            _kand_mask64(xe2_mask, _kand_mask64(_kshiftri_mask64(x80_mask, 1), _kshiftri_mask64(xaf_mask, 2)));
        // U+205F: E2 81 9F
        __mmask64 u205f_mask =
            _kand_mask64(xe2_mask, _kand_mask64(_kshiftri_mask64(x81_mask, 1), _kshiftri_mask64(x9f_mask, 2)));
        // U+2028: E2 80 A8 (LINE SEPARATOR)
        __mmask64 u2028_mask =
            _kand_mask64(xe2_mask, _kand_mask64(_kshiftri_mask64(x80_mask, 1), _kshiftri_mask64(xa8_mask, 2)));
        // U+2029: E2 80 A9 (PARAGRAPH SEPARATOR)
        __mmask64 u2029_mask =
            _kand_mask64(xe2_mask, _kand_mask64(_kshiftri_mask64(x80_mask, 1), _kshiftri_mask64(xa9_mask, 2)));
        __mmask64 ideographic_mask =
            _kand_mask64(xe3_mask, _kand_mask64(_kshiftri_mask64(x80_mask, 1), _kshiftri_mask64(x80_mask, 2)));
        sz_u64_t three_byte_mask = _cvtmask64_u64(_kor_mask64(
            _kor_mask64(_kor_mask64(_kor_mask64(ogham_mask, range_e280_mask), _kor_mask64(u202f_mask, u205f_mask)),
                        _kor_mask64(u2028_mask, u2029_mask)),
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

SZ_PUBLIC sz_cptr_t sz_utf8_find_nth_ice(sz_cptr_t text, sz_size_t length, sz_size_t max_chars,
                                         sz_size_t *chars_skipped) {
    sz_u512_vec_t continuation_mask_vec;
    continuation_mask_vec.zmm = _mm512_set1_epi8((char)0xC0);
    sz_u512_vec_t continuation_pattern_vec;
    continuation_pattern_vec.zmm = _mm512_set1_epi8((char)0x80);

    sz_u8_t const *text_u8 = (sz_u8_t const *)text;
    sz_size_t char_count = 0;

    // Process 64 bytes at a time
    while (length >= 64 && char_count < max_chars) {
        sz_u512_vec_t text_vec;
        text_vec.zmm = _mm512_loadu_epi8(text_u8);

        // Apply mask (byte & 0xC0)
        sz_u512_vec_t masked;
        masked.zmm = _mm512_and_si512(text_vec.zmm, continuation_mask_vec.zmm);

        // Compare with 0x80 to find continuation bytes
        __mmask64 is_continuation = _mm512_cmpeq_epi8_mask(masked.zmm, continuation_pattern_vec.zmm);

        // Count non-continuation bytes (inverted mask)
        sz_u32_t non_continuation_count = 64 - _mm_popcnt_u64(_cvtmask64_u64(is_continuation));

        // Check if we would exceed max_chars
        if (char_count + non_continuation_count > max_chars) {
            // Fall back to serial for the remainder
            sz_size_t remaining_chars = max_chars - char_count;
            sz_cptr_t result = sz_utf8_find_nth_serial((sz_cptr_t)text_u8, length, remaining_chars, chars_skipped);
            *chars_skipped = char_count + *chars_skipped;
            return result;
        }

        char_count += non_continuation_count;
        text_u8 += 64;
        length -= 64;
    }

    // Process remaining bytes with serial
    if (length > 0 && char_count < max_chars) {
        sz_size_t remaining_chars = max_chars - char_count;
        sz_size_t serial_chars;
        sz_cptr_t result = sz_utf8_find_nth_serial((sz_cptr_t)text_u8, length, remaining_chars, &serial_chars);
        char_count += serial_chars;
        *chars_skipped = char_count;
        return result;
    }

    *chars_skipped = char_count;
    return (sz_cptr_t)text_u8;
}

SZ_PUBLIC sz_bool_t sz_utf8_valid_ice(sz_cptr_t text, sz_size_t length) {
    // For now, fall back to serial implementation
    // TODO: Implement full SIMD UTF-8 validation using AVX-512
    // (This requires complex state machine for multi-byte sequences)
    return sz_utf8_valid_serial(text, length);
}

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#endif            // SZ_USE_ICE
#pragma endregion // Ice Lake Implementation

#pragma region NEON Implementation

SZ_PUBLIC sz_cptr_t sz_utf8_find_newline_neon(sz_cptr_t text, sz_size_t length, sz_size_t *matched_length) {
    return sz_utf8_find_newline_serial(text, length, matched_length);
}

SZ_PUBLIC sz_cptr_t sz_utf8_find_whitespace_neon(sz_cptr_t text, sz_size_t length, sz_size_t *matched_length) {
    return sz_utf8_find_whitespace_serial(text, length, matched_length);
}

#pragma endregion // NEON Implementation

#pragma region Haswell Implementation
#if SZ_USE_HASWELL
#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("avx2"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("avx2")
#endif

SZ_PUBLIC sz_cptr_t sz_utf8_find_newline_haswell(sz_cptr_t text, sz_size_t length, sz_size_t *matched_length) {

    // We need to check if the ASCII chars in [10,13] (same as '\n', '\v', '\f', '\r') are present.
    // The last one - '\r' - needs special handling to differentiate between "\r" and "\r\n".
    __m256i n_vec = _mm256_set1_epi8('\n');
    __m256i v_vec = _mm256_set1_epi8('\v');
    __m256i f_vec = _mm256_set1_epi8('\f');
    __m256i r_vec = _mm256_set1_epi8('\r');

    // We also need to match the 2-byte newline character 0xC285 (NEL),
    // as well as the 3-byte characters 0xE280A8 (PS) and 0xE280A9 (LS).
    __m256i xc2_vec = _mm256_set1_epi8('\xC2');
    __m256i x85_vec = _mm256_set1_epi8('\x85');
    __m256i xe2_vec = _mm256_set1_epi8('\xE2');
    __m256i x80_vec = _mm256_set1_epi8('\x80');
    __m256i xa8_vec = _mm256_set1_epi8('\xA8');
    __m256i xa9_vec = _mm256_set1_epi8('\xA9');

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
        __m256i xc2_cmp = _mm256_cmpeq_epi8(text_vec, xc2_vec);
        __m256i x85_cmp = _mm256_cmpeq_epi8(text_vec, x85_vec);
        __m256i xe2_cmp = _mm256_cmpeq_epi8(text_vec, xe2_vec);
        __m256i x80_cmp = _mm256_cmpeq_epi8(text_vec, x80_vec);
        __m256i xa8_cmp = _mm256_cmpeq_epi8(text_vec, xa8_vec);
        __m256i xa9_cmp = _mm256_cmpeq_epi8(text_vec, xa9_vec);

        sz_u32_t xc2_mask = (sz_u32_t)_mm256_movemask_epi8(xc2_cmp);
        sz_u32_t x85_mask = (sz_u32_t)_mm256_movemask_epi8(x85_cmp);
        sz_u32_t xe2_mask = (sz_u32_t)_mm256_movemask_epi8(xe2_cmp);
        sz_u32_t x80_mask = (sz_u32_t)_mm256_movemask_epi8(x80_cmp);
        sz_u32_t xa8_mask = (sz_u32_t)_mm256_movemask_epi8(xa8_cmp);
        sz_u32_t xa9_mask = (sz_u32_t)_mm256_movemask_epi8(xa9_cmp);

        // 2-byte matches
        sz_u32_t rn_mask = r_mask & (n_mask >> 1);
        sz_u32_t xc285_mask = xc2_mask & (x85_mask >> 1);
        sz_u32_t two_byte_mask = rn_mask | xc285_mask;

        // 3-byte matches
        sz_u32_t xe280_mask = xe2_mask & (x80_mask >> 1);
        sz_u32_t e280a8_mask = xe280_mask & (xa8_mask >> 2);
        sz_u32_t e280a9_mask = xe280_mask & (xa9_mask >> 2);
        sz_u32_t three_byte_mask = e280a8_mask | e280a9_mask;

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
    __m256i x20_vec = _mm256_set1_epi8(' ');

    // We also need to match the 2-byte characters 0xC285 (NEL) and 0xC2A0 (NBSP)
    __m256i xc2_vec = _mm256_set1_epi8('\xC2');
    __m256i x85_vec = _mm256_set1_epi8('\x85');
    __m256i xa0_vec = _mm256_set1_epi8('\xA0');

    // 3-byte character prefixes and suffixes
    __m256i xe1_vec = _mm256_set1_epi8('\xE1');
    __m256i xe2_vec = _mm256_set1_epi8('\xE2');
    __m256i xe3_vec = _mm256_set1_epi8('\xE3');
    __m256i x9a_vec = _mm256_set1_epi8('\x9A');
    __m256i x80_vec = _mm256_set1_epi8('\x80');
    __m256i x81_vec = _mm256_set1_epi8('\x81');
    __m256i xa8_vec = _mm256_set1_epi8('\xA8');
    __m256i xa9_vec = _mm256_set1_epi8('\xA9');
    __m256i xaf_vec = _mm256_set1_epi8('\xAF');
    __m256i x9f_vec = _mm256_set1_epi8('\x9F');

    // We check 32 bytes of data at once, but only step forward by 30 bytes for split-register matches.
    while (length >= 32) {
        __m256i text_vec = _mm256_loadu_si256((__m256i const *)text);

        // 1-byte indicators & matches
        // Range [9,13] covers \t, \n, \v, \f, \r
        __m256i x20_cmp = _mm256_cmpeq_epi8(text_vec, x20_vec);
        __m256i t_cmp = _mm256_cmpgt_epi8(text_vec, _mm256_set1_epi8((char)0x08)); // >= '\t' (0x09)
        __m256i r_cmp = _mm256_cmpgt_epi8(_mm256_set1_epi8((char)0x0E), text_vec); // <= '\r' (0x0D)
        __m256i tr_range = _mm256_and_si256(t_cmp, r_cmp);
        __m256i one_byte_cmp = _mm256_or_si256(x20_cmp, tr_range);

        sz_u32_t one_byte_mask = (sz_u32_t)_mm256_movemask_epi8(one_byte_cmp);

        // 2-byte and 3-byte prefix indicators
        __m256i xc2_cmp = _mm256_cmpeq_epi8(text_vec, xc2_vec);
        __m256i xe1_cmp = _mm256_cmpeq_epi8(text_vec, xe1_vec);
        __m256i xe2_cmp = _mm256_cmpeq_epi8(text_vec, xe2_vec);
        __m256i xe3_cmp = _mm256_cmpeq_epi8(text_vec, xe3_vec);

        sz_u32_t xc2_mask = (sz_u32_t)_mm256_movemask_epi8(xc2_cmp) & 0x7FFFFFFF;
        sz_u32_t xe1_mask = (sz_u32_t)_mm256_movemask_epi8(xe1_cmp) & 0x3FFFFFFF;
        sz_u32_t xe2_mask = (sz_u32_t)_mm256_movemask_epi8(xe2_cmp) & 0x3FFFFFFF;
        sz_u32_t xe3_mask = (sz_u32_t)_mm256_movemask_epi8(xe3_cmp) & 0x3FFFFFFF;
        sz_u32_t prefix_mask = xc2_mask | xe1_mask | xe2_mask | xe3_mask;

        // Check for fast path: one-byte match before any prefix
        if (one_byte_mask) {
            if (prefix_mask) {
                int first_one_byte_offset = sz_u32_ctz(one_byte_mask);
                int first_prefix_offset = sz_u32_ctz(prefix_mask);
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
        __m256i x85_cmp = _mm256_cmpeq_epi8(text_vec, x85_vec);
        __m256i xa0_cmp = _mm256_cmpeq_epi8(text_vec, xa0_vec);
        sz_u32_t x85_mask = (sz_u32_t)_mm256_movemask_epi8(x85_cmp);
        sz_u32_t xa0_mask = (sz_u32_t)_mm256_movemask_epi8(xa0_cmp);

        sz_u32_t xc285_mask = xc2_mask & (x85_mask >> 1); // U+0085 NEL
        sz_u32_t xc2a0_mask = xc2_mask & (xa0_mask >> 1); // U+00A0 NBSP
        sz_u32_t two_byte_mask = xc285_mask | xc2a0_mask;

        // 3-byte suffixes
        __m256i x9a_cmp = _mm256_cmpeq_epi8(text_vec, x9a_vec);
        __m256i x80_cmp = _mm256_cmpeq_epi8(text_vec, x80_vec);
        __m256i x81_cmp = _mm256_cmpeq_epi8(text_vec, x81_vec);
        __m256i x80_ge_cmp = _mm256_cmpgt_epi8(text_vec, _mm256_set1_epi8((char)0x7F)); // >= 0x80
        __m256i x8d_le_cmp = _mm256_cmpgt_epi8(_mm256_set1_epi8((char)0x8E), text_vec); // <= 0x8D
        __m256i x8d_range = _mm256_and_si256(x80_ge_cmp, x8d_le_cmp);
        __m256i xa8_cmp = _mm256_cmpeq_epi8(text_vec, xa8_vec);
        __m256i xa9_cmp = _mm256_cmpeq_epi8(text_vec, xa9_vec);
        __m256i xaf_cmp = _mm256_cmpeq_epi8(text_vec, xaf_vec);
        __m256i x9f_cmp = _mm256_cmpeq_epi8(text_vec, x9f_vec);

        sz_u32_t x9a_mask = (sz_u32_t)_mm256_movemask_epi8(x9a_cmp);
        sz_u32_t x80_mask = (sz_u32_t)_mm256_movemask_epi8(x80_cmp);
        sz_u32_t x81_mask = (sz_u32_t)_mm256_movemask_epi8(x81_cmp);
        sz_u32_t x8d_range_mask = (sz_u32_t)_mm256_movemask_epi8(x8d_range);
        sz_u32_t xa8_mask = (sz_u32_t)_mm256_movemask_epi8(xa8_cmp);
        sz_u32_t xa9_mask = (sz_u32_t)_mm256_movemask_epi8(xa9_cmp);
        sz_u32_t xaf_mask = (sz_u32_t)_mm256_movemask_epi8(xaf_cmp);
        sz_u32_t x9f_mask = (sz_u32_t)_mm256_movemask_epi8(x9f_cmp);

        // 3-byte matches
        sz_u32_t ogham_mask = xe1_mask & (x9a_mask >> 1) & (x80_mask >> 2);            // E1 9A 80
        sz_u32_t range_e280_mask = xe2_mask & (x80_mask >> 1) & (x8d_range_mask >> 2); // E2 80 [80-8D]
        sz_u32_t u202f_mask = xe2_mask & (x80_mask >> 1) & (xaf_mask >> 2);            // E2 80 AF
        sz_u32_t u205f_mask = xe2_mask & (x81_mask >> 1) & (x9f_mask >> 2);            // E2 81 9F
        sz_u32_t u2028_mask = xe2_mask & (x80_mask >> 1) & (xa8_mask >> 2);            // E2 80 A8
        sz_u32_t u2029_mask = xe2_mask & (x80_mask >> 1) & (xa9_mask >> 2);            // E2 80 A9
        sz_u32_t ideographic_mask = xe3_mask & (x80_mask >> 1) & (x80_mask >> 2);      // E3 80 80
        sz_u32_t three_byte_mask =
            ogham_mask | range_e280_mask | u202f_mask | u205f_mask | u2028_mask | u2029_mask | ideographic_mask;

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

SZ_PUBLIC sz_cptr_t sz_utf8_find_nth_haswell(sz_cptr_t text, sz_size_t length, sz_size_t max_chars,
                                             sz_size_t *chars_skipped) {
    __m256i continuation_mask_vec = _mm256_set1_epi8((char)0xC0);
    __m256i continuation_pattern_vec = _mm256_set1_epi8((char)0x80);

    sz_u8_t const *text_u8 = (sz_u8_t const *)text;
    sz_size_t char_count = 0;

    // Process 32 bytes at a time
    while (length >= 32 && char_count < max_chars) {
        __m256i text_vec = _mm256_loadu_si256((__m256i const *)text_u8);

        // Apply mask (byte & 0xC0)
        __m256i masked = _mm256_and_si256(text_vec, continuation_mask_vec);

        // Compare with 0x80 to find continuation bytes
        __m256i is_continuation_vec = _mm256_cmpeq_epi8(masked, continuation_pattern_vec);
        sz_u32_t is_continuation_mask = (sz_u32_t)_mm256_movemask_epi8(is_continuation_vec);

        // Count non-continuation bytes (inverted mask)
        sz_u32_t non_continuation_count = 32 - _mm_popcnt_u32(is_continuation_mask);

        // Check if we would exceed max_chars
        if (char_count + non_continuation_count > max_chars) {
            // Fall back to serial for the remainder
            sz_size_t remaining_chars = max_chars - char_count;
            sz_cptr_t result = sz_utf8_find_nth_serial((sz_cptr_t)text_u8, length, remaining_chars, chars_skipped);
            *chars_skipped = char_count + *chars_skipped;
            return result;
        }

        char_count += non_continuation_count;
        text_u8 += 32;
        length -= 32;
    }

    // Process remaining bytes with serial
    if (length > 0 && char_count < max_chars) {
        sz_size_t remaining_chars = max_chars - char_count;
        sz_size_t serial_chars;
        sz_cptr_t result = sz_utf8_find_nth_serial((sz_cptr_t)text_u8, length, remaining_chars, &serial_chars);
        char_count += serial_chars;
        *chars_skipped = char_count;
        return result;
    }

    *chars_skipped = char_count;
    return (sz_cptr_t)text_u8;
}

SZ_PUBLIC sz_bool_t sz_utf8_valid_haswell(sz_cptr_t text, sz_size_t length) {
    // TODO: Implement AVX2 version - for now fall back to serial
    return sz_utf8_valid_serial(text, length);
}

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#endif            // SZ_USE_HASWELL
#pragma endregion // Haswell Implementation

#pragma region Dynamic Dispatch

#if !SZ_DYNAMIC_DISPATCH

SZ_DYNAMIC sz_cptr_t sz_utf8_find_newline(sz_cptr_t text, sz_size_t length, sz_size_t *matched_length) {
#if SZ_USE_ICE
    return sz_utf8_find_newline_ice(text, length, matched_length);
#elif SZ_USE_HASWELL
    return sz_utf8_find_newline_haswell(text, length, matched_length);
#else
    return sz_utf8_find_newline_serial(text, length, matched_length);
#endif
}

SZ_DYNAMIC sz_cptr_t sz_utf8_find_whitespace(sz_cptr_t text, sz_size_t length, sz_size_t *matched_length) {
#if SZ_USE_ICE
    return sz_utf8_find_whitespace_ice(text, length, matched_length);
#elif SZ_USE_HASWELL
    return sz_utf8_find_whitespace_haswell(text, length, matched_length);
#else
    return sz_utf8_find_whitespace_serial(text, length, matched_length);
#endif
}

SZ_DYNAMIC sz_cptr_t sz_utf8_find_nth(sz_cptr_t text, sz_size_t length, sz_size_t max_chars, sz_size_t *chars_skipped) {
#if SZ_USE_ICE
    return sz_utf8_find_nth_ice(text, length, max_chars, chars_skipped);
#elif SZ_USE_HASWELL
    return sz_utf8_find_nth_haswell(text, length, max_chars, chars_skipped);
#else
    return sz_utf8_find_nth_serial(text, length, max_chars, chars_skipped);
#endif
}

SZ_DYNAMIC sz_bool_t sz_utf8_valid(sz_cptr_t text, sz_size_t length) {
#if SZ_USE_ICE
    return sz_utf8_valid_ice(text, length);
#elif SZ_USE_HASWELL
    return sz_utf8_valid_haswell(text, length);
#else
    return sz_utf8_valid_serial(text, length);
#endif
}

#endif // !SZ_DYNAMIC_DISPATCH

#pragma endregion // Dynamic Dispatch

#ifdef __cplusplus
}
#endif

#endif // STRINGZILLA_UTF8_H_
