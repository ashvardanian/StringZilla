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

/**
 *  @brief  Unpack a UTF-8 string into UTF-32 codepoints.
 *
 *  This function is designed for streaming-like decoding with smart iterators built on top of it.
 *  The iterator would unpack a continuous slice of UTF-8 text into UTF-32 codepoints in chunks,
 *  yielding them upstream - only one at a time. This avoids allocating large buffers for the entire
 *  UTF-32 string, which can be 4x the size of the UTF-8 input.
 *
 *  This functionality is similar to the `simdutf` library's UTF-8 to UTF-32 conversion routines,
 *  but unlike most of them - performs no validity checks, and leverages an assumption that absolute
 *  majority of written text doesn't mix codepoints of every length in each register-sized chunk.
 *
 *  - English text and source code is predominantly 1-byte ASCII characters.
 *  - Broader European languages with diacritics mostly use 2-byte characters with 1-byte punctuation.
 *  - Chinese & Jamapanese mostly use 3-byte characters with rare punctuation, which can be 1- or 3-byte.
 *  - Korean uses 3-byte characters with 1-byte spaces; word are 2-6 syllables or 6-16 bytes.
 *
 *  It's a different story for emoji-heavy texts, which can mix 4-byte characters more frequently.
 *
 *  @param[in] text UTF-8 string to unpack.
 *  @param[in] length Number of bytes in the string (up to 64).
 *  @param[out] runes Output buffer for UTF-32 codepoints (recommended to be at least @b 64 entries wide).
 *  @param[in] runes_capacity Capacity of the @p runes buffer (number of sz_rune_t entries).
 *  @param[out] runes_unpacked Number of runes unpacked.
 *  @return Pointer to the byte after the last unpacked byte in @p text.
 */
SZ_DYNAMIC sz_cptr_t sz_utf8_unpack_chunk(      //
    sz_cptr_t text, sz_size_t length,           //
    sz_rune_t *runes, sz_size_t runes_capacity, //
    sz_size_t *runes_unpacked);

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
/** @copydoc sz_utf8_unpack_chunk */
SZ_PUBLIC sz_cptr_t sz_utf8_unpack_chunk_serial( //
    sz_cptr_t text, sz_size_t length,            //
    sz_rune_t *runes, sz_size_t runes_capacity,  //
    sz_size_t *runes_unpacked);

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
/** @copydoc sz_utf8_unpack_chunk */
SZ_PUBLIC sz_cptr_t sz_utf8_unpack_chunk_ice( //
    sz_cptr_t text, sz_size_t length,         //
    sz_rune_t *runes, sz_size_t runes_capacity, sz_size_t *runes_unpacked);
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
/** @copydoc sz_utf8_unpack_chunk */
SZ_PUBLIC sz_cptr_t sz_utf8_unpack_chunk_neon( //
    sz_cptr_t text, sz_size_t length,          //
    sz_rune_t *runes, sz_size_t runes_capacity, sz_size_t *runes_unpacked);
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

SZ_PUBLIC sz_bool_t sz_utf8_valid_serial(sz_cptr_t text, sz_size_t length) {
    sz_u8_t const *text_u8 = (sz_u8_t const *)text;
    sz_u8_t const *end_u8 = text_u8 + length;

    while (text_u8 < end_u8) {
        sz_u8_t byte1 = *text_u8;

        // 1-byte sequence (0x00-0x7F)
        if (byte1 <= 0x7F) { text_u8 += 1; }

        // 2-byte sequence (0xC2-0xDF)
        else if (byte1 >= 0xC2 && byte1 <= 0xDF) {
            if (text_u8 + 1 >= end_u8) return sz_false_k;
            sz_u8_t byte2 = text_u8[1];
            if ((byte2 & 0xC0) != 0x80) return sz_false_k; // Invalid continuation
            text_u8 += 2;
        }

        // 3-byte sequence (0xE0-0xEF)
        else if (byte1 >= 0xE0 && byte1 <= 0xEF) {
            if (text_u8 + 2 >= end_u8) return sz_false_k;
            sz_u8_t byte2 = text_u8[1];
            sz_u8_t byte3 = text_u8[2];
            if ((byte2 & 0xC0) != 0x80 || (byte3 & 0xC0) != 0x80) return sz_false_k;

            // Check for overlong encodings and surrogates
            if (byte1 == 0xE0 && byte2 < 0xA0) return sz_false_k;  // Overlong
            if (byte1 == 0xED && byte2 >= 0xA0) return sz_false_k; // Surrogate (U+D800-U+DFFF)

            text_u8 += 3;
        }

        // 4-byte sequence (0xF0-0xF4)
        else if (byte1 >= 0xF0 && byte1 <= 0xF4) {
            if (text_u8 + 3 >= end_u8) return sz_false_k;
            sz_u8_t byte2 = text_u8[1];
            sz_u8_t byte3 = text_u8[2];
            sz_u8_t byte4 = text_u8[3];
            if ((byte2 & 0xC0) != 0x80 || (byte3 & 0xC0) != 0x80 || (byte4 & 0xC0) != 0x80) return sz_false_k;

            // Check for overlong and out-of-range
            if (byte1 == 0xF0 && byte2 < 0x90) return sz_false_k;  // Overlong
            if (byte1 == 0xF4 && byte2 >= 0x90) return sz_false_k; // > U+10FFFF

            text_u8 += 4;
        }

        // Invalid lead byte
        else
            return sz_false_k;
    }

    return sz_true_k;
}

SZ_PUBLIC sz_cptr_t sz_utf8_unpack_chunk_serial( //
    sz_cptr_t text, sz_size_t length,            //
    sz_rune_t *runes, sz_size_t runes_capacity,  //
    sz_size_t *runes_unpacked) {

    sz_cptr_t src = text;
    sz_cptr_t src_end = text + length;
    sz_size_t runes_written = 0;

    // Process up to runes_capacity codepoints or end of input
    while (src < src_end && runes_written < runes_capacity) {
        sz_rune_t rune;
        sz_rune_length_t rune_length;
        sz_rune_parse(src, &rune, &rune_length);
        if (rune_length == sz_utf8_invalid_k) break;
        if (src + rune_length > src_end) break; // Incomplete sequence
        runes[runes_written++] = rune;
        src += rune_length;
    }

    *runes_unpacked = runes_written;
    return src;
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

SZ_PUBLIC sz_cptr_t sz_utf8_unpack_chunk_ice(   //
    sz_cptr_t text, sz_size_t length,           //
    sz_rune_t *runes, sz_size_t runes_capacity, //
    sz_size_t *runes_unpacked) {

    // Filter out obsolte calls
    if (!runes_capacity || !length) return text;

    // Process up to the minimum of: available bytes, (output capacity * 4), or optimal chunk size (64)
    sz_size_t chunk_size = sz_min_of_three(length, runes_capacity * 4, 64);
    sz_u512_vec_t text_vec, runes_vec;
    __mmask64 load_mask = sz_u64_mask_until_(chunk_size);
    text_vec.zmm = _mm512_maskz_loadu_epi8(load_mask, (sz_u8_t const *)text);
    __mmask64 is_non_ascii = _mm512_movepi8_mask(text_vec.zmm);

    // Check if its our lucky day and we have an entire register worth of ASCII text,
    // that we will output into runes directly. English is responsible for roughly 60% of the text
    // on the Internet, so this will often be our primary execution path.
    if (is_non_ascii == 0) {
        // For ASCII, 1 byte = 1 rune, so limit to runes_capacity
        sz_size_t runes_to_unpack = sz_min_of_two(chunk_size, runes_capacity);
        _mm512_mask_storeu_epi32(runes, sz_u16_clamp_mask_until_(runes_to_unpack),
                                 _mm512_cvtepu8_epi32(_mm512_castsi512_si128(text_vec.zmm)));
        if (runes_to_unpack > 16)
            _mm512_mask_storeu_epi32(runes + 16, sz_u16_clamp_mask_until_(runes_to_unpack - 16),
                                     _mm512_cvtepu8_epi32(_mm512_extracti32x4_epi32(text_vec.zmm, 1)));
        if (runes_to_unpack > 32)
            _mm512_mask_storeu_epi32(runes + 32, sz_u16_clamp_mask_until_(runes_to_unpack - 32),
                                     _mm512_cvtepu8_epi32(_mm512_extracti32x4_epi32(text_vec.zmm, 2)));
        if (runes_to_unpack > 48)
            _mm512_mask_storeu_epi32(runes + 48, sz_u16_clamp_mask_until_(runes_to_unpack - 48),
                                     _mm512_cvtepu8_epi32(_mm512_extracti32x4_epi32(text_vec.zmm, 3)));
        *runes_unpacked = runes_to_unpack;
        return text + runes_to_unpack;
    }

    // Russian, Spanish, German, and French are the 2nd, 3rd, 4th, and 5th most common languages on the Internet,
    // and all of them are composed of a mixture of 2-byte and 1-byte UTF-8 characters. When dealing with such text
    // we plan the algorithm with respect to the number of decoded entries we can fit in a single output register.
    // We don't need to validate the UTF-8 encoding, just classify the inputs to locate the first 3- or 4-byte
    // character in the input:
    // - ASCII: bit 7 = 0, i.e., 0xxxxxxx (0x00-0x7F)
    // - 2-byte lead: bits 7-5 = 110, i.e., 110xxxxx (0xC0-0xDF)
    // - Continuation: bits 7-6 = 10, i.e., 10xxxxxx (0x80-0xBF)
    __mmask64 is_ascii = ~is_non_ascii & load_mask;
    __mmask64 is_two_byte_start = _mm512_mask_cmpeq_epi8_mask(
        load_mask, _mm512_and_si512(text_vec.zmm, _mm512_set1_epi8((char)0xE0)), _mm512_set1_epi8((char)0xC0));
    __mmask64 is_continuation = _mm512_mask_cmpeq_epi8_mask(
        load_mask, _mm512_and_si512(text_vec.zmm, _mm512_set1_epi8((char)0xC0)), _mm512_set1_epi8((char)0x80));

    // Find longest prefix containing only ASCII and complete 2-byte sequences - let's call it the "Mixed 12" case
    __mmask64 is_expected_continuation = is_two_byte_start << 1;
    __mmask64 is_valid_mixed12 = is_ascii | is_two_byte_start | (is_continuation & is_expected_continuation);
    sz_size_t mixed12_prefix_length = sz_u64_ctz(~is_valid_mixed12 | ~load_mask);
    mixed12_prefix_length -= mixed12_prefix_length && ((is_two_byte_start >> (mixed12_prefix_length - 1)) & 1);

    if (mixed12_prefix_length >= 2) {
        __mmask64 prefix_mask = sz_u64_mask_until_(mixed12_prefix_length);
        __mmask64 is_char_start = (is_ascii | is_two_byte_start) & prefix_mask;
        sz_size_t num_runes = (sz_size_t)sz_u64_popcount(is_char_start);
        sz_size_t runes_to_unpack = sz_min_of_three(num_runes, runes_capacity, 16);

        // Compress character start positions into sequential indices, then gather bytes
        sz_u512_vec_t char_indices;
        char_indices.zmm = _mm512_set_epi8(                                 //
            63, 62, 61, 60, 59, 58, 57, 56, 55, 54, 53, 52, 51, 50, 49, 48, //
            47, 46, 45, 44, 43, 42, 41, 40, 39, 38, 37, 36, 35, 34, 33, 32, //
            31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, //
            15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
        char_indices.zmm = _mm512_maskz_compress_epi8(is_char_start, char_indices.zmm);

        sz_u512_vec_t first_bytes, second_bytes;
        first_bytes.zmm = _mm512_permutexvar_epi8(char_indices.zmm, text_vec.zmm);
        second_bytes.zmm =
            _mm512_permutexvar_epi8(_mm512_add_epi8(char_indices.zmm, _mm512_set1_epi8(1)), text_vec.zmm);

        // Expand to 32-bit and decode 2-byte sequences: ((first & 0x1F) << 6) | (second & 0x3F)
        __m512i first_bytes_wide = _mm512_cvtepu8_epi32(_mm512_castsi512_si128(first_bytes.zmm));
        __m512i second_bytes_wide = _mm512_cvtepu8_epi32(_mm512_castsi512_si128(second_bytes.zmm));
        __mmask16 is_two_byte_char = (__mmask16)_pext_u64(is_two_byte_start, is_char_start);
        __m512i decoded_two_byte =
            _mm512_or_si512(_mm512_slli_epi32(_mm512_and_si512(first_bytes_wide, _mm512_set1_epi32(0x1F)), 6),
                            _mm512_and_si512(second_bytes_wide, _mm512_set1_epi32(0x3F)));

        // Blend: ASCII positions keep byte value, 2-byte positions get decoded rune
        runes_vec.zmm = _mm512_mask_blend_epi32(is_two_byte_char, first_bytes_wide, decoded_two_byte);
        _mm512_mask_storeu_epi32(runes, sz_u16_mask_until_(runes_to_unpack), runes_vec.zmm);

        // Bytes consumed: one per ASCII, two per 2-byte sequence
        sz_size_t two_byte_count = (sz_size_t)sz_u64_popcount(is_two_byte_char & sz_u16_mask_until_(runes_to_unpack));
        *runes_unpacked = runes_to_unpack;
        return text + runes_to_unpack + two_byte_count;
    }

    // Check for the number of 3-byte characters - in this case we can't easily cast to 16-bit integers
    // and check for equality, but we can pre-define the masks and values we expect at each byte position.
    // For 3-byte UTF-8 sequences, we check if bytes match the pattern: 1110xxxx 10xxxxxx 10xxxxxx
    // We need to check every 3rd byte starting from position 0.
    sz_u512_vec_t three_byte_mask_vec, three_byte_pattern_vec;
    three_byte_mask_vec.zmm = _mm512_set1_epi32(0x00C0C0F0);    // Mask: [F0, C0, C0, 00] per 4-byte slot
    three_byte_pattern_vec.zmm = _mm512_set1_epi32(0x008080E0); // Pattern: [E0, 80, 80, 00] per 4-byte slot

    // Create permutation indices to gather 3-byte sequences into 4-byte slots
    // Input:  [b0 b1 b2]    [b3 b4 b5]    [b6 b7 b8]    ... (up to 16 triplets from 48 bytes)
    // Output: [b0 b1 b2 XX] [b3 b4 b5 XX] [b6 b7 b8 XX] ... (16 slots, 4th byte zeroed)
    sz_u512_vec_t permute_indices;
    permute_indices.zmm = _mm512_setr_epi32(
        // Triplets 0-3:  [0,1,2,_] [3,4,5,_] [6,7,8,_] [9,10,11,_]
        0x40020100, 0x40050403, 0x40080706, 0x400B0A09,
        // Triplets 4-7:  [12,13,14,_] [15,16,17,_] [18,19,20,_] [21,22,23,_]
        0x400E0D0C, 0x40111010, 0x40141312, 0x40171615,
        // Triplets 8-11: [24,25,26,_] [27,28,29,_] [30,31,32,_] [33,34,35,_]
        0x401A1918, 0x401D1C1B, 0x40201F1E, 0x40232221,
        // Triplets 12-15: [36,37,38,_] [39,40,41,_] [42,43,44,_] [45,46,47,_]
        0x40262524, 0x40292827, 0x402C2B2A, 0x402F2E2D);

    // Permute to gather triplets into slots
    sz_u512_vec_t gathered_triplets;
    gathered_triplets.zmm = _mm512_permutexvar_epi8(permute_indices.zmm, text_vec.zmm);

    // Check if gathered bytes match 3-byte UTF-8 pattern
    sz_u512_vec_t masked_triplets;
    masked_triplets.zmm = _mm512_and_si512(gathered_triplets.zmm, three_byte_mask_vec.zmm);
    __mmask16 three_byte_match_mask = _mm512_cmpeq_epi32_mask(masked_triplets.zmm, three_byte_pattern_vec.zmm);
    sz_size_t three_byte_prefix_length = sz_u64_ctz(~three_byte_match_mask);

    if (three_byte_prefix_length) {
        // Unpack up to 16 three-byte characters (48 bytes of input).
        sz_size_t runes_to_place = sz_min_of_three(three_byte_prefix_length, 16, runes_capacity);
        // Decode: ((b0 & 0x0F) << 12) | ((b1 & 0x3F) << 6) | (b2 & 0x3F)
        // gathered_triplets has: [b0, b1, b2, XX] in each 32-bit slot (little-endian: 0xXXb2b1b0)
        // Extract: b0 from bits 7-0, b1 from bits 15-8, b2 from bits 23-16
        runes_vec.zmm = _mm512_or_si512(
            _mm512_or_si512(
                // (b0 & 0x0F) << 12
                _mm512_slli_epi32(_mm512_and_si512(gathered_triplets.zmm, _mm512_set1_epi32(0x0FU)), 12),
                // (b1 & 0x3F) << 6
                _mm512_slli_epi32(
                    _mm512_and_si512(_mm512_srli_epi32(gathered_triplets.zmm, 8), _mm512_set1_epi32(0x3FU)), 6)),
            _mm512_and_si512(_mm512_srli_epi32(gathered_triplets.zmm, 16), _mm512_set1_epi32(0x3FU))); // (b2 & 0x3F)
        _mm512_mask_storeu_epi32(runes, sz_u16_mask_until_(runes_to_place), runes_vec.zmm);
        *runes_unpacked = runes_to_place;
        return text + runes_to_place * 3;
    }

    // Check for the number of 4-byte characters
    // For 4-byte UTF-8 sequences: 11110xxx 10xxxxxx 10xxxxxx 10xxxxxx
    // With a homogeneous 4-byte prefix, we have perfect 4-byte alignment (up to 16 sequences in 64 bytes)
    sz_u512_vec_t four_byte_mask_vec, four_byte_pattern_vec;
    four_byte_mask_vec.zmm = _mm512_set1_epi32((int)0xC0C0C0F8);    // Mask: [F8, C0, C0, C0] per 4-byte slot
    four_byte_pattern_vec.zmm = _mm512_set1_epi32((int)0x808080F0); // Pattern: [F0, 80, 80, 80] per 4-byte slot

    // Mask and check for 4-byte pattern in each 32-bit slot
    sz_u512_vec_t masked_quads;
    masked_quads.zmm = _mm512_and_si512(text_vec.zmm, four_byte_mask_vec.zmm);
    __mmask16 four_byte_match_mask = _mm512_cmpeq_epi32_mask(masked_quads.zmm, four_byte_pattern_vec.zmm);
    sz_size_t four_byte_prefix_length = sz_u64_ctz(~four_byte_match_mask);

    if (four_byte_prefix_length) {
        // Unpack up to 16 four-byte characters (64 bytes of input).
        sz_size_t runes_to_place = sz_min_of_three(four_byte_prefix_length, 16, runes_capacity);
        // Decode: ((b0 & 0x07) << 18) | ((b1 & 0x3F) << 12) | ((b2 & 0x3F) << 6) | (b3 & 0x3F)
        runes_vec.zmm = _mm512_or_si512(
            _mm512_or_si512(
                // (b0 & 0x07) << 18
                _mm512_slli_epi32(_mm512_and_si512(text_vec.zmm, _mm512_set1_epi32(0x07U)), 18),
                // (b1 & 0x3F) << 12
                _mm512_slli_epi32(_mm512_and_si512(_mm512_srli_epi32(text_vec.zmm, 8), _mm512_set1_epi32(0x3FU)), 12)),
            _mm512_or_si512(
                // (b2 & 0x3F) << 6
                _mm512_slli_epi32(_mm512_and_si512(_mm512_srli_epi32(text_vec.zmm, 16), _mm512_set1_epi32(0x3FU)), 6),
                // (b3 & 0x3F)
                _mm512_and_si512(_mm512_srli_epi32(text_vec.zmm, 24), _mm512_set1_epi32(0x3FU))));
        _mm512_mask_storeu_epi32(runes, sz_u16_mask_until_(runes_to_place), runes_vec.zmm);
        *runes_unpacked = runes_to_place;
        return text + runes_to_place * 4;
    }

    // Fallback to serial for mixed/malformed content
    return sz_utf8_unpack_chunk_serial(text, length, runes, runes_capacity, runes_unpacked);
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

    uint8x16_t drop1_vec = vsetq_lane_u8(0x00, vdupq_n_u8(0xFF), 15);
    uint8x16_t drop2_vec = vsetq_lane_u8(0x00, drop1_vec, 14);

    while (length >= 16) {
        text_vec.u8x16 = vld1q_u8((sz_u8_t const *)text);

        // 1-byte matches
        uint8x16_t n_cmp = vceqq_u8(text_vec.u8x16, n_vec);
        uint8x16_t v_cmp = vceqq_u8(text_vec.u8x16, v_vec);
        uint8x16_t f_cmp = vceqq_u8(text_vec.u8x16, f_vec);
        uint8x16_t r_cmp = vandq_u8(vceqq_u8(text_vec.u8x16, r_vec), drop1_vec); // Mask out \r at position 15
        uint8x16_t one_byte_cmp = vorrq_u8(vorrq_u8(n_cmp, v_cmp), vorrq_u8(f_cmp, r_cmp));

        // 2- & 3-byte matches with shifted views
        uint8x16_t text1 = vextq_u8(text_vec.u8x16, text_vec.u8x16, 1);
        uint8x16_t text2 = vextq_u8(text_vec.u8x16, text_vec.u8x16, 2);
        uint8x16_t rn_vec = vandq_u8(r_cmp, vceqq_u8(text1, n_vec));
        uint8x16_t x_c285_vec = vandq_u8(vceqq_u8(text_vec.u8x16, x_c2_vec), vceqq_u8(text1, x_85_vec));
        uint8x16_t two_byte_cmp = vandq_u8(vorrq_u8(rn_vec, x_c285_vec), drop1_vec); // Ignore last split match

        uint8x16_t x_e280_cmp = vandq_u8(vceqq_u8(text_vec.u8x16, x_e2_vec), vceqq_u8(text1, x_80_vec));
        uint8x16_t x_e280ax_cmp = vandq_u8(x_e280_cmp, vorrq_u8(vceqq_u8(text2, x_a8_vec), vceqq_u8(text2, x_a9_vec)));
        uint8x16_t three_byte_cmp = vandq_u8(x_e280ax_cmp, drop2_vec); // Ignore last two split matches

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

    uint8x16_t drop1_vec = vsetq_lane_u8(0x00, vdupq_n_u8(0xFF), 15);
    uint8x16_t drop2_vec = vsetq_lane_u8(0x00, drop1_vec, 14);

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
        uint8x16_t prefix_byte_cmp =
            vorrq_u8(one_byte_cmp, vorrq_u8(vorrq_u8(x_c2_cmp, x_e1_cmp), vorrq_u8(x_e2_cmp, x_e3_cmp)));

        sz_u64_t one_byte_mask = sz_utf8_vreinterpretq_u8_u4_(one_byte_cmp);
        sz_u64_t prefix_byte_mask = sz_utf8_vreinterpretq_u8_u4_(prefix_byte_cmp);

        // Check for fast path - no whitespaces in this chunk
        if (!prefix_byte_mask) {
            text += 14;
            length -= 14;
            continue;
        }
        // Another simple common case - we have spotted a one-byte match before any prefix
        else if (one_byte_mask) {
            int first_one_byte_offset = sz_u64_ctz(one_byte_mask);
            int first_prefix_offset = sz_u64_ctz(prefix_byte_mask);
            if (first_one_byte_offset < first_prefix_offset) {
                *matched_length = 1;
                return text + (first_one_byte_offset / 4);
            }
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
            vandq_u8(vorrq_u8(vorrq_u8(vorrq_u8(ogham_cmp, range_e280_cmp), vorrq_u8(line_cmp, paragraph_cmp)),
                              vorrq_u8(vorrq_u8(nnbsp_cmp, mmsp_cmp), ideographic_vec)),
                     drop2_vec);

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

    // Early return for short inputs
    if (length < step) return sz_utf8_find_newline_serial(text, length, matched_length);

    // SVE2 kernels are a bit different from both NEON and Ice Lake due to presence of
    // a few very convenient and cheap instructions. Most importantly, we have `svmatch` instruction
    // that can match against a set of bytes in one go, similar to many invocations of `vceqq` in NEON
    // with subsequent mask combination.
    svuint8_t prefix_byte_set =
        svdupq_n_u8('\n', '\v', '\f', '\r', 0xC2, 0xE2, '\n', '\n', '\n', '\n', '\n', '\n', '\n', '\n', '\n', '\n');
    svuint8_t one_byte_set =
        svdupq_n_u8('\n', '\v', '\f', '\r', '\n', '\n', '\n', '\n', '\n', '\n', '\n', '\n', '\n', '\n', '\n', '\n');
    svuint8_t zeros = svdup_n_u8(0);

    // We load full `step` bytes but only match on first `step - 2` positions.
    // This allows using svext for shifted views without extra loads.
    sz_size_t const usable_step = step - 2;
    sz_size_t offset = 0;
    while (offset + step <= length) {
        svbool_t pg = svwhilelt_b8_u64(0, usable_step);             // First step-2 lanes active
        svuint8_t text0 = svld1_u8(svptrue_b8(), text_u8 + offset); // Load full step bytes

        // Fast rejection: any potential first byte?
        if (!svptest_any(pg, svmatch_u8(pg, text0, prefix_byte_set))) {
            offset += usable_step;
            continue;
        }

        // Shifted views via svext - zeros fill unused lanes at end, but pg masks them out
        svuint8_t text1 = svext_u8(text0, zeros, 1);
        svuint8_t text2 = svext_u8(text0, zeros, 2);

        // 1-byte matches
        svbool_t one_byte_mask = svmatch_u8(pg, text0, one_byte_set);

        // 2-byte matches
        svbool_t rn_mask = svand_b_z(pg, svcmpeq_n_u8(pg, text0, '\r'), svcmpeq_n_u8(pg, text1, '\n'));
        svbool_t x_c285_mask = svand_b_z(pg, svcmpeq_n_u8(pg, text0, 0xC2), svcmpeq_n_u8(pg, text1, 0x85));
        svbool_t two_byte_mask = svorr_b_z(pg, rn_mask, x_c285_mask);

        // 3-byte matches
        svbool_t x_e280_mask = svand_b_z(pg, svcmpeq_n_u8(pg, text0, 0xE2), svcmpeq_n_u8(pg, text1, 0x80));
        svbool_t three_byte_mask =
            svand_b_z(pg, x_e280_mask, svorr_b_z(pg, svcmpeq_n_u8(pg, text2, 0xA8), svcmpeq_n_u8(pg, text2, 0xA9)));

        // Technically, we may want to exclude "\r" that is part of "\r\n" from one-byte matches,
        // but we don't really need it here - it won't affect the exstimates.
        //
        //      one_byte_mask = svbic_b_z(pg, one_byte_mask, rn_mask);
        svbool_t combined_mask = svorr_b_z(pg, one_byte_mask, svorr_b_z(pg, two_byte_mask, three_byte_mask));
        if (svptest_any(pg, combined_mask)) {
            sz_size_t pos = svcntp_b8(pg, svbrkb_b_z(pg, combined_mask));
            svbool_t at_pos = svcmpeq_n_u8(svptrue_b8(), svindex_u8(0, 1), (sz_u8_t)pos);
            sz_size_t has_two_byte = svptest_any(at_pos, two_byte_mask);
            sz_size_t has_three_byte = svptest_any(at_pos, three_byte_mask);
            sz_size_t length_value = 1;
            length_value += has_two_byte | has_three_byte;
            length_value += has_three_byte;
            *matched_length = length_value;
            return (sz_cptr_t)(text_u8 + offset + pos);
        }
        offset += usable_step;
    }

    // Handle remaining bytes with serial fallback
    return sz_utf8_find_newline_serial(text + offset, length - offset, matched_length);
}

SZ_PUBLIC sz_cptr_t sz_utf8_find_whitespace_sve2(sz_cptr_t text, sz_size_t length, sz_size_t *matched_length) {
    sz_u8_t const *text_u8 = (sz_u8_t const *)text;
    sz_size_t const step = svcntb();

    // Early return for short inputs
    if (length < step) return sz_utf8_find_whitespace_serial(text, length, matched_length);

    // Character sets for MATCH (DUPQ replicates 128-bit pattern, no stack/loads)
    svuint8_t any_byte_set =
        svdupq_n_u8(' ', '\t', '\n', '\v', '\f', '\r', 0xC2, 0xE1, 0xE2, 0xE3, ' ', ' ', ' ', ' ', ' ', ' ');
    svuint8_t one_byte_set =
        svdupq_n_u8(' ', '\t', '\n', '\v', '\f', '\r', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ');
    // Valid third bytes for E2 80 XX: U+2000-U+200D (0x80-0x8D), U+2028 (0xA8), U+2029 (0xA9)
    svuint8_t e280_third_bytes =
        svdupq_n_u8(0x80, 0x81, 0x82, 0x83, 0x84, 0x85, 0x86, 0x87, 0x88, 0x89, 0x8A, 0x8B, 0x8C, 0x8D, 0xA8, 0xA9);
    svuint8_t zeros = svdup_n_u8(0);

    // We load full `step` bytes but only match on first `step - 2` positions.
    // This allows using svext for shifted views without extra loads.
    sz_size_t const usable_step = step - 2;
    sz_size_t offset = 0;
    while (offset + step <= length) {
        svbool_t pg = svwhilelt_b8_u64(0, usable_step);             // First step-2 lanes active
        svuint8_t text0 = svld1_u8(svptrue_b8(), text_u8 + offset); // Load full step bytes

        // Fast rejection: skip if no whitespace-related bytes at all
        if (!svptest_any(pg, svmatch_u8(pg, text0, any_byte_set))) {
            offset += usable_step;
            continue;
        }

        // 1-byte whitespace: space, tab, newlines
        svbool_t one_byte_mask = svmatch_u8(pg, text0, one_byte_set);

        // Shifted views via svext - zeros fill unused lanes at end, but pg masks them out
        svuint8_t text1 = svext_u8(text0, zeros, 1);
        svuint8_t text2 = svext_u8(text0, zeros, 2);

        // 2-byte: C2 + {85, A0} (NEL, NBSP)
        svbool_t x_c2_mask = svcmpeq_n_u8(pg, text0, 0xC2);
        svbool_t x_85_mask = svcmpeq_n_u8(pg, text1, 0x85);
        svbool_t x_a0_mask = svcmpeq_n_u8(pg, text1, 0xA0);
        svbool_t two_byte_mask = svand_b_z(pg, x_c2_mask, svorr_b_z(pg, x_85_mask, x_a0_mask));

        // 3-byte: E1 9A 80 (Ogham Space Mark)
        svbool_t ogham_mask = svand_b_z(pg, svand_b_z(pg, svcmpeq_n_u8(pg, text0, 0xE1), svcmpeq_n_u8(pg, text1, 0x9A)),
                                        svcmpeq_n_u8(pg, text2, 0x80));

        // 3-byte: E2 80 XX - various Unicode spaces (U+2000-U+200D, U+2028, U+2029, U+202F)
        svbool_t x_e2_mask = svcmpeq_n_u8(pg, text0, 0xE2);
        svbool_t x_e280_mask = svand_b_z(pg, x_e2_mask, svcmpeq_n_u8(pg, text1, 0x80));
        svbool_t x_e280xx_mask = svand_b_z(pg, x_e280_mask, svmatch_u8(pg, text2, e280_third_bytes));
        // U+202F: E2 80 AF (NARROW NO-BREAK SPACE) - doesn't fit in the 16-byte set
        svbool_t nnbsp_mask = svand_b_z(pg, x_e280_mask, svcmpeq_n_u8(pg, text2, 0xAF));
        // U+205F: E2 81 9F (MEDIUM MATHEMATICAL SPACE)
        svbool_t mmsp_mask =
            svand_b_z(pg, svand_b_z(pg, x_e2_mask, svcmpeq_n_u8(pg, text1, 0x81)), svcmpeq_n_u8(pg, text2, 0x9F));

        // 3-byte: E3 80 80 (IDEOGRAPHIC SPACE)
        svbool_t ideographic_mask =
            svand_b_z(pg, svand_b_z(pg, svcmpeq_n_u8(pg, text0, 0xE3), svcmpeq_n_u8(pg, text1, 0x80)),
                      svcmpeq_n_u8(pg, text2, 0x80));

        svbool_t three_byte_mask = svorr_b_z(pg, svorr_b_z(pg, ogham_mask, svorr_b_z(pg, x_e280xx_mask, nnbsp_mask)),
                                             svorr_b_z(pg, mmsp_mask, ideographic_mask));
        svbool_t combined_mask = svorr_b_z(pg, one_byte_mask, svorr_b_z(pg, two_byte_mask, three_byte_mask));

        if (svptest_any(pg, combined_mask)) {
            sz_size_t pos = svcntp_b8(pg, svbrkb_b_z(pg, combined_mask));
            svbool_t at_pos = svcmpeq_n_u8(svptrue_b8(), svindex_u8(0, 1), (sz_u8_t)pos);
            sz_size_t has_two_byte = svptest_any(at_pos, two_byte_mask);
            sz_size_t has_three_byte = svptest_any(at_pos, three_byte_mask);
            sz_size_t length_value = 1;
            length_value += has_two_byte | has_three_byte;
            length_value += has_three_byte;
            *matched_length = length_value;
            return (sz_cptr_t)(text_u8 + offset + pos);
        }
        offset += usable_step;
    }

    // Handle remaining bytes with serial fallback
    return sz_utf8_find_whitespace_serial(text + offset, length - offset, matched_length);
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
#elif SZ_USE_SVE2 && SZ_ENFORCE_SVE_OVER_NEON
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
#elif SZ_USE_SVE2 && SZ_ENFORCE_SVE_OVER_NEON
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
