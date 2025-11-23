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
 *  - `sz_utf8_find_case_insensitive` - case-insensitive substring search in UTF-8 strings
 *  - `sz_utf8_unpack_chunk` - convert UTF-8 to UTF-32 in a streaming manner
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
 *  @brief  Validate UTF-8 encoding correctness per RFC 3629.
 *
 *  Checks continuation byte patterns (10xxxxxx), rejects overlong encodings, surrogates (U+D800-U+DFFF),
 *  values beyond U+10FFFF, and truncated sequences. Only accepts proper 1-4 byte UTF-8 sequences.
 *
 *  Overlong encodings occur when a codepoint is encoded with more bytes than necessary.
 *  For example, "/" (U+002F) can be incorrectly encoded as 0xC0 0xAF (2 bytes) or 0xE0 0x80 0xAF (3 bytes)
 *  instead of the valid 0x2F (1 byte). These create security vulnerabilities (directory traversal attacks)
 *  by bypassing filters that check for "/" in its canonical form.
 *
 *  Surrogates (U+D800-U+DFFF) are UTF-16 encoding artifacts - they're not actual Unicode characters.
 *  UTF-16 uses "surrogate pairs" (high surrogate + low surrogate) to represent codepoints beyond U+FFFF.
 *  UTF-8 encodes these codepoints directly in 4 bytes without needing surrogates, so the range
 *  U+D800-U+DFFF is permanently invalid in UTF-8 to prevent multiple representations of the same character.
 *
 *  @param[in] text String to validate.
 *  @param[in] length Number of bytes in the string.
 *  @return 1 if valid UTF-8, 0 if invalid.
 *
 *  @code
 *      if (!sz_utf8_valid(user_input, input_len)) {
 *          fprintf(stderr, "Invalid UTF-8 encoding\n");
 *          return ERROR;
 *      }
 *  @endcode
 *
 *  @note Invalid patterns include: overlong encodings (0xC0 0x80 for null), surrogates (0xED 0xA0 0x80),
 *        invalid continuation bytes, out-of-range codepoints (0xF4 0x90+), and truncated sequences.
 */
SZ_DYNAMIC sz_bool_t sz_utf8_valid(sz_cptr_t text, sz_size_t length);

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

/**
 *  @brief  Case-insensitive substring search in UTF-8 strings.
 *
 *  In applications where the haystack remains largely static and memory/storage is cheap, it is recommended
 *  to pre-process the haystack into a case-folded version using Unicode Case Folding (e.g., via the ICU
 *  library) and subsequently use the simpler `sz_find()` function for repeated searches. This avoids the cost
 *  of performing full folding logic during every search operation.
 *
 *  This function applies full Unicode Case Folding as defined in the Unicode Standard (UAX #21 and
 *  CaseFolding.txt), covering all bicameral scripts, all offset-based one-to-one folds, all table-based
 *  one-to-one folds, and all normative one-to-many expansions.
 *
 *  The following character mappings are supported:
 *
 *  - ASCII Latin letters A–Z (U+0041–U+005A) are folded to a–z (U+0061–U+007A) using a trivial +32 offset.
 *  - Fullwidth Latin letters Ａ–Ｚ (U+FF21–U+FF3A) are folded to ａ–ｚ (U+FF41–U+FF5A) with the same +32 offset.
 *  - Cyrillic uppercase А–Я (U+0410–U+042F) are folded to а–я (U+0430–U+044F) using a +32 offset.
 *  - Armenian uppercase Ա–Ֆ (U+0531–U+0556) are folded to ա–ֆ (U+0561–U+0586) using a +48 offset.
 *  - Georgian Mtavruli letters Ა-Ჿ (U+1C90–U+1CBF, excluding 2) are folded to their Mkhedruli equivalents
 *    (U+10D0–U+10FF) using a fixed linear translation defined by the Unicode Standard.
 *  - Greek uppercase Α–Ω (U+0391–U+03A9) are folded to α–ω (U+03B1–U+03C9) via a +32 offset, with a normative
 *    context-sensitive rule for sigma: Σ (U+03A3) folds to σ (U+03C3) or ς (U+03C2) depending on word-final
 *    position, as required by SpecialCasing.txt.
 *  - Latin Extended characters include numerous one-to-one folds and several one-to-many expansions, including:
 *      ß  (U+00DF) → "ss" (U+0073 U+0073)
 *      ẞ  (U+1E9E) → "ss"
 *    as well as mixed-case digraphs and trigraphs normalized to lowercase sequences.
 *  - Turkish and Azerbaijani dotted/dotless-I rules follow SpecialCasing.txt, including:
 *      İ (U+0130)  → "i̇" (U+0069 U+0307)
 *      I (U+0049)  →  i   (U+0069)
 *      ı (U+0131)  →  ı   (already lowercase)
 *    with full locale-correct behavior.
 *  - Lithuanian accented I/J mappings that require combining-dot additions or removals are processed
 *    as multi-codepoint expansions exactly as specified in SpecialCasing.txt.
 *  - Additional bicameral scripts—Cherokee, Deseret, Osage, Warang Citi, Adlam—use their normative
 *    one-to-one uppercase-to-lowercase mappings defined in CaseFolding.txt.
 *
 *  Folding is applied during matching without rewriting the entire haystack. Multi-codepoint expansions,
 *  contextual folds, and combining-mark adjustments are handled at comparison time.
 *
 *  @param[in] haystack UTF-8 string to be searched.
 *  @param[in] haystack_length Number of bytes in the haystack buffer.
 *  @param[in] needle UTF-8 substring to search for.
 *  @param[in] needle_length Number of bytes in the needle substring.
 *  @param[out] matched_length Number of bytes in the matched region.
 *  @return Pointer to the first matching substring from @p haystack, or @c SZ_NULL_CHAR if not found.
 */
SZ_DYNAMIC sz_cptr_t sz_utf8_find_case_insensitive( //
    sz_cptr_t haystack, sz_size_t haystack_length,  //
    sz_cptr_t needle, sz_size_t needle_length, sz_size_t *matched_length);

#pragma endregion

#pragma region Platform-Specific Backends

// Serial (portable) implementations
SZ_PUBLIC sz_size_t sz_utf8_count_serial(sz_cptr_t text, sz_size_t length);
SZ_PUBLIC sz_cptr_t sz_utf8_find_nth_serial(sz_cptr_t text, sz_size_t length, sz_size_t n);
SZ_PUBLIC sz_cptr_t sz_utf8_find_newline_serial(sz_cptr_t text, sz_size_t length, sz_size_t *matched_length);
SZ_PUBLIC sz_cptr_t sz_utf8_find_whitespace_serial(sz_cptr_t text, sz_size_t length, sz_size_t *matched_length);
SZ_PUBLIC sz_cptr_t sz_utf8_unpack_chunk_serial( //
    sz_cptr_t text, sz_size_t length,            //
    sz_rune_t *runes, sz_size_t runes_capacity,  //
    sz_size_t *runes_unpacked);

// Haswell (AVX2) implementations
SZ_PUBLIC sz_size_t sz_utf8_count_haswell(sz_cptr_t text, sz_size_t length);
SZ_PUBLIC sz_cptr_t sz_utf8_find_nth_haswell(sz_cptr_t text, sz_size_t length, sz_size_t n);
SZ_PUBLIC sz_cptr_t sz_utf8_find_newline_haswell(sz_cptr_t text, sz_size_t length, sz_size_t *matched_length);
SZ_PUBLIC sz_cptr_t sz_utf8_find_whitespace_haswell(sz_cptr_t text, sz_size_t length, sz_size_t *matched_length);
SZ_PUBLIC sz_cptr_t sz_utf8_unpack_chunk_haswell( //
    sz_cptr_t text, sz_size_t length,             //
    sz_rune_t *runes, sz_size_t runes_capacity,   //
    sz_size_t *runes_unpacked);

// Ice Lake (AVX-512) implementations
SZ_PUBLIC sz_size_t sz_utf8_count_ice(sz_cptr_t text, sz_size_t length);
SZ_PUBLIC sz_cptr_t sz_utf8_find_nth_ice(sz_cptr_t text, sz_size_t length, sz_size_t n);
SZ_PUBLIC sz_cptr_t sz_utf8_find_newline_ice(sz_cptr_t text, sz_size_t length, sz_size_t *matched_length);
SZ_PUBLIC sz_cptr_t sz_utf8_find_whitespace_ice(sz_cptr_t text, sz_size_t length, sz_size_t *matched_length);
SZ_PUBLIC sz_cptr_t sz_utf8_unpack_chunk_ice(   //
    sz_cptr_t text, sz_size_t length,           //
    sz_rune_t *runes, sz_size_t runes_capacity, //
    sz_size_t *runes_unpacked);

// NEON (ARM) implementations - fall back to serial
SZ_PUBLIC sz_size_t sz_utf8_count_neon(sz_cptr_t text, sz_size_t length);
SZ_PUBLIC sz_cptr_t sz_utf8_find_nth_neon(sz_cptr_t text, sz_size_t length, sz_size_t n);
SZ_PUBLIC sz_cptr_t sz_utf8_find_newline_neon(sz_cptr_t text, sz_size_t length, sz_size_t *matched_length);
SZ_PUBLIC sz_cptr_t sz_utf8_find_whitespace_neon(sz_cptr_t text, sz_size_t length, sz_size_t *matched_length);
SZ_PUBLIC sz_cptr_t sz_utf8_unpack_chunk_neon(  //
    sz_cptr_t text, sz_size_t length,           //
    sz_rune_t *runes, sz_size_t runes_capacity, //
    sz_size_t *runes_unpacked);

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

SZ_PUBLIC sz_cptr_t sz_utf8_unpack_chunk_serial( //
    sz_cptr_t text, sz_size_t length,            //
    sz_rune_t *runes, sz_size_t runes_capacity,  //
    sz_size_t *runes_unpacked) {
    // Process up to the minimum of: available bytes, output capacity, or optimal chunk size (64)
    sz_cptr_t const end = text + sz_min_of_three(length, runes_capacity * 4, 64);
    sz_size_t count = 0;
    sz_rune_length_t rune_length;
    for (; text != end && count < runes_capacity; text += rune_length, runes++, count++)
        sz_rune_parse(text, runes, &rune_length);
    *runes_unpacked = count;
    return text;
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

SZ_INTERNAL sz_rune_t sz_unicode_fold_codepoint_(sz_rune_t cp, sz_rune_t *expansion, sz_size_t *expansion_count) {

    *expansion_count = 0;

    // Offset-based ranges
    if (cp >= 0x0041 && cp <= 0x005A) return cp + 0x20;       // offset +32
    if (cp >= 0x00C0 && cp <= 0x00D6) return cp + 0x20;       // offset +32
    if (cp >= 0x00D8 && cp <= 0x00DE) return cp + 0x20;       // offset +32
    if (cp >= 0x0391 && cp <= 0x03A1) return cp + 0x20;       // offset +32
    if (cp >= 0x03A3 && cp <= 0x03AB) return cp + 0x20;       // offset +32
    if (cp >= 0x0410 && cp <= 0x042F) return cp + 0x20;       // offset +32
    if (cp >= 0xFF21 && cp <= 0xFF3A) return cp + 0x20;       // offset +32
    if (cp >= 0x10D50 && cp <= 0x10D65) return cp + 0x20;     // offset +32
    if (cp >= 0x118A0 && cp <= 0x118BF) return cp + 0x20;     // offset +32
    if (cp >= 0x16E40 && cp <= 0x16E5F) return cp + 0x20;     // offset +32
    if (cp >= 0x0531 && cp <= 0x0556) return cp + 0x30;       // offset +48
    if (cp >= 0x2C00 && cp <= 0x2C2F) return cp + 0x30;       // offset +48
    if (cp >= 0xAB70 && cp <= 0xABBF) return cp + 0xFFFF6830; // offset -38864
    if (cp >= 0x10400 && cp <= 0x10427) return cp + 0x28;     // offset +40
    if (cp >= 0x104B0 && cp <= 0x104D3) return cp + 0x28;     // offset +40
    if (cp >= 0x13F8 && cp <= 0x13FD) return cp + 0xFFFFFFF8; // offset -8
    if (cp >= 0x1F08 && cp <= 0x1F0F) return cp + 0xFFFFFFF8; // offset -8
    if (cp >= 0x1F18 && cp <= 0x1F1D) return cp + 0xFFFFFFF8; // offset -8
    if (cp >= 0x1F28 && cp <= 0x1F2F) return cp + 0xFFFFFFF8; // offset -8
    if (cp >= 0x1F38 && cp <= 0x1F3F) return cp + 0xFFFFFFF8; // offset -8
    if (cp >= 0x1F48 && cp <= 0x1F4D) return cp + 0xFFFFFFF8; // offset -8
    if (cp >= 0x1F68 && cp <= 0x1F6F) return cp + 0xFFFFFFF8; // offset -8
    if (cp >= 0x10C80 && cp <= 0x10CB2) return cp + 0x40;     // offset +64
    if (cp >= 0x1C90 && cp <= 0x1CBA) return cp + 0xFFFFF440; // offset -3008
    if (cp >= 0x1CBD && cp <= 0x1CBF) return cp + 0xFFFFF440; // offset -3008
    if (cp >= 0x10A0 && cp <= 0x10C5) return cp + 0x1C60;     // offset +7264
    if (cp >= 0x10570 && cp <= 0x1057A) return cp + 0x27;     // offset +39
    if (cp >= 0x1057C && cp <= 0x1058A) return cp + 0x27;     // offset +39
    if (cp >= 0x1058C && cp <= 0x10592) return cp + 0x27;     // offset +39
    if (cp >= 0x1E900 && cp <= 0x1E921) return cp + 0x22;     // offset +34
    if (cp >= 0x24B6 && cp <= 0x24CF) return cp + 0x1A;       // offset +26
    if (cp >= 0x16EA0 && cp <= 0x16EB8) return cp + 0x1B;     // offset +27
    if (cp >= 0x2160 && cp <= 0x216F) return cp + 0x10;       // offset +16
    if (cp >= 0x0400 && cp <= 0x040F) return cp + 0x50;       // offset +80
    if (cp >= 0x03FD && cp <= 0x03FF) return cp + 0xFFFFFF7E; // offset -130
    if (cp >= 0x1FC8 && cp <= 0x1FCB) return cp + 0xFFFFFFAA; // offset -86
    if (cp >= 0x0388 && cp <= 0x038A) return cp + 0x25;       // offset +37

    // Irregular one-to-one mappings
    switch (cp) {
    case 0x00B5: return 0x03BC;   // 'µ' → 'μ'
    case 0x0100: return 0x0101;   // 'Ā' → 'ā'
    case 0x0102: return 0x0103;   // 'Ă' → 'ă'
    case 0x0104: return 0x0105;   // 'Ą' → 'ą'
    case 0x0106: return 0x0107;   // 'Ć' → 'ć'
    case 0x0108: return 0x0109;   // 'Ĉ' → 'ĉ'
    case 0x010A: return 0x010B;   // 'Ċ' → 'ċ'
    case 0x010C: return 0x010D;   // 'Č' → 'č'
    case 0x010E: return 0x010F;   // 'Ď' → 'ď'
    case 0x0110: return 0x0111;   // 'Đ' → 'đ'
    case 0x0112: return 0x0113;   // 'Ē' → 'ē'
    case 0x0114: return 0x0115;   // 'Ĕ' → 'ĕ'
    case 0x0116: return 0x0117;   // 'Ė' → 'ė'
    case 0x0118: return 0x0119;   // 'Ę' → 'ę'
    case 0x011A: return 0x011B;   // 'Ě' → 'ě'
    case 0x011C: return 0x011D;   // 'Ĝ' → 'ĝ'
    case 0x011E: return 0x011F;   // 'Ğ' → 'ğ'
    case 0x0120: return 0x0121;   // 'Ġ' → 'ġ'
    case 0x0122: return 0x0123;   // 'Ģ' → 'ģ'
    case 0x0124: return 0x0125;   // 'Ĥ' → 'ĥ'
    case 0x0126: return 0x0127;   // 'Ħ' → 'ħ'
    case 0x0128: return 0x0129;   // 'Ĩ' → 'ĩ'
    case 0x012A: return 0x012B;   // 'Ī' → 'ī'
    case 0x012C: return 0x012D;   // 'Ĭ' → 'ĭ'
    case 0x012E: return 0x012F;   // 'Į' → 'į'
    case 0x0132: return 0x0133;   // 'Ĳ' → 'ĳ'
    case 0x0134: return 0x0135;   // 'Ĵ' → 'ĵ'
    case 0x0136: return 0x0137;   // 'Ķ' → 'ķ'
    case 0x0139: return 0x013A;   // 'Ĺ' → 'ĺ'
    case 0x013B: return 0x013C;   // 'Ļ' → 'ļ'
    case 0x013D: return 0x013E;   // 'Ľ' → 'ľ'
    case 0x013F: return 0x0140;   // 'Ŀ' → 'ŀ'
    case 0x0141: return 0x0142;   // 'Ł' → 'ł'
    case 0x0143: return 0x0144;   // 'Ń' → 'ń'
    case 0x0145: return 0x0146;   // 'Ņ' → 'ņ'
    case 0x0147: return 0x0148;   // 'Ň' → 'ň'
    case 0x014A: return 0x014B;   // 'Ŋ' → 'ŋ'
    case 0x014C: return 0x014D;   // 'Ō' → 'ō'
    case 0x014E: return 0x014F;   // 'Ŏ' → 'ŏ'
    case 0x0150: return 0x0151;   // 'Ő' → 'ő'
    case 0x0152: return 0x0153;   // 'Œ' → 'œ'
    case 0x0154: return 0x0155;   // 'Ŕ' → 'ŕ'
    case 0x0156: return 0x0157;   // 'Ŗ' → 'ŗ'
    case 0x0158: return 0x0159;   // 'Ř' → 'ř'
    case 0x015A: return 0x015B;   // 'Ś' → 'ś'
    case 0x015C: return 0x015D;   // 'Ŝ' → 'ŝ'
    case 0x015E: return 0x015F;   // 'Ş' → 'ş'
    case 0x0160: return 0x0161;   // 'Š' → 'š'
    case 0x0162: return 0x0163;   // 'Ţ' → 'ţ'
    case 0x0164: return 0x0165;   // 'Ť' → 'ť'
    case 0x0166: return 0x0167;   // 'Ŧ' → 'ŧ'
    case 0x0168: return 0x0169;   // 'Ũ' → 'ũ'
    case 0x016A: return 0x016B;   // 'Ū' → 'ū'
    case 0x016C: return 0x016D;   // 'Ŭ' → 'ŭ'
    case 0x016E: return 0x016F;   // 'Ů' → 'ů'
    case 0x0170: return 0x0171;   // 'Ű' → 'ű'
    case 0x0172: return 0x0173;   // 'Ų' → 'ų'
    case 0x0174: return 0x0175;   // 'Ŵ' → 'ŵ'
    case 0x0176: return 0x0177;   // 'Ŷ' → 'ŷ'
    case 0x0178: return 0x00FF;   // 'Ÿ' → 'ÿ'
    case 0x0179: return 0x017A;   // 'Ź' → 'ź'
    case 0x017B: return 0x017C;   // 'Ż' → 'ż'
    case 0x017D: return 0x017E;   // 'Ž' → 'ž'
    case 0x017F: return 0x0073;   // 'ſ' → 's'
    case 0x0181: return 0x0253;   // 'Ɓ' → 'ɓ'
    case 0x0182: return 0x0183;   // 'Ƃ' → 'ƃ'
    case 0x0184: return 0x0185;   // 'Ƅ' → 'ƅ'
    case 0x0186: return 0x0254;   // 'Ɔ' → 'ɔ'
    case 0x0187: return 0x0188;   // 'Ƈ' → 'ƈ'
    case 0x0189: return 0x0256;   // 'Ɖ' → 'ɖ'
    case 0x018A: return 0x0257;   // 'Ɗ' → 'ɗ'
    case 0x018B: return 0x018C;   // 'Ƌ' → 'ƌ'
    case 0x018E: return 0x01DD;   // 'Ǝ' → 'ǝ'
    case 0x018F: return 0x0259;   // 'Ə' → 'ə'
    case 0x0190: return 0x025B;   // 'Ɛ' → 'ɛ'
    case 0x0191: return 0x0192;   // 'Ƒ' → 'ƒ'
    case 0x0193: return 0x0260;   // 'Ɠ' → 'ɠ'
    case 0x0194: return 0x0263;   // 'Ɣ' → 'ɣ'
    case 0x0196: return 0x0269;   // 'Ɩ' → 'ɩ'
    case 0x0197: return 0x0268;   // 'Ɨ' → 'ɨ'
    case 0x0198: return 0x0199;   // 'Ƙ' → 'ƙ'
    case 0x019C: return 0x026F;   // 'Ɯ' → 'ɯ'
    case 0x019D: return 0x0272;   // 'Ɲ' → 'ɲ'
    case 0x019F: return 0x0275;   // 'Ɵ' → 'ɵ'
    case 0x01A0: return 0x01A1;   // 'Ơ' → 'ơ'
    case 0x01A2: return 0x01A3;   // 'Ƣ' → 'ƣ'
    case 0x01A4: return 0x01A5;   // 'Ƥ' → 'ƥ'
    case 0x01A6: return 0x0280;   // 'Ʀ' → 'ʀ'
    case 0x01A7: return 0x01A8;   // 'Ƨ' → 'ƨ'
    case 0x01A9: return 0x0283;   // 'Ʃ' → 'ʃ'
    case 0x01AC: return 0x01AD;   // 'Ƭ' → 'ƭ'
    case 0x01AE: return 0x0288;   // 'Ʈ' → 'ʈ'
    case 0x01AF: return 0x01B0;   // 'Ư' → 'ư'
    case 0x01B1: return 0x028A;   // 'Ʊ' → 'ʊ'
    case 0x01B2: return 0x028B;   // 'Ʋ' → 'ʋ'
    case 0x01B3: return 0x01B4;   // 'Ƴ' → 'ƴ'
    case 0x01B5: return 0x01B6;   // 'Ƶ' → 'ƶ'
    case 0x01B7: return 0x0292;   // 'Ʒ' → 'ʒ'
    case 0x01B8: return 0x01B9;   // 'Ƹ' → 'ƹ'
    case 0x01BC: return 0x01BD;   // 'Ƽ' → 'ƽ'
    case 0x01C4: return 0x01C6;   // 'Ǆ' → 'ǆ'
    case 0x01C5: return 0x01C6;   // 'ǅ' → 'ǆ'
    case 0x01C7: return 0x01C9;   // 'Ǉ' → 'ǉ'
    case 0x01C8: return 0x01C9;   // 'ǈ' → 'ǉ'
    case 0x01CA: return 0x01CC;   // 'Ǌ' → 'ǌ'
    case 0x01CB: return 0x01CC;   // 'ǋ' → 'ǌ'
    case 0x01CD: return 0x01CE;   // 'Ǎ' → 'ǎ'
    case 0x01CF: return 0x01D0;   // 'Ǐ' → 'ǐ'
    case 0x01D1: return 0x01D2;   // 'Ǒ' → 'ǒ'
    case 0x01D3: return 0x01D4;   // 'Ǔ' → 'ǔ'
    case 0x01D5: return 0x01D6;   // 'Ǖ' → 'ǖ'
    case 0x01D7: return 0x01D8;   // 'Ǘ' → 'ǘ'
    case 0x01D9: return 0x01DA;   // 'Ǚ' → 'ǚ'
    case 0x01DB: return 0x01DC;   // 'Ǜ' → 'ǜ'
    case 0x01DE: return 0x01DF;   // 'Ǟ' → 'ǟ'
    case 0x01E0: return 0x01E1;   // 'Ǡ' → 'ǡ'
    case 0x01E2: return 0x01E3;   // 'Ǣ' → 'ǣ'
    case 0x01E4: return 0x01E5;   // 'Ǥ' → 'ǥ'
    case 0x01E6: return 0x01E7;   // 'Ǧ' → 'ǧ'
    case 0x01E8: return 0x01E9;   // 'Ǩ' → 'ǩ'
    case 0x01EA: return 0x01EB;   // 'Ǫ' → 'ǫ'
    case 0x01EC: return 0x01ED;   // 'Ǭ' → 'ǭ'
    case 0x01EE: return 0x01EF;   // 'Ǯ' → 'ǯ'
    case 0x01F1: return 0x01F3;   // 'Ǳ' → 'ǳ'
    case 0x01F2: return 0x01F3;   // 'ǲ' → 'ǳ'
    case 0x01F4: return 0x01F5;   // 'Ǵ' → 'ǵ'
    case 0x01F6: return 0x0195;   // 'Ƕ' → 'ƕ'
    case 0x01F7: return 0x01BF;   // 'Ƿ' → 'ƿ'
    case 0x01F8: return 0x01F9;   // 'Ǹ' → 'ǹ'
    case 0x01FA: return 0x01FB;   // 'Ǻ' → 'ǻ'
    case 0x01FC: return 0x01FD;   // 'Ǽ' → 'ǽ'
    case 0x01FE: return 0x01FF;   // 'Ǿ' → 'ǿ'
    case 0x0200: return 0x0201;   // 'Ȁ' → 'ȁ'
    case 0x0202: return 0x0203;   // 'Ȃ' → 'ȃ'
    case 0x0204: return 0x0205;   // 'Ȅ' → 'ȅ'
    case 0x0206: return 0x0207;   // 'Ȇ' → 'ȇ'
    case 0x0208: return 0x0209;   // 'Ȉ' → 'ȉ'
    case 0x020A: return 0x020B;   // 'Ȋ' → 'ȋ'
    case 0x020C: return 0x020D;   // 'Ȍ' → 'ȍ'
    case 0x020E: return 0x020F;   // 'Ȏ' → 'ȏ'
    case 0x0210: return 0x0211;   // 'Ȑ' → 'ȑ'
    case 0x0212: return 0x0213;   // 'Ȓ' → 'ȓ'
    case 0x0214: return 0x0215;   // 'Ȕ' → 'ȕ'
    case 0x0216: return 0x0217;   // 'Ȗ' → 'ȗ'
    case 0x0218: return 0x0219;   // 'Ș' → 'ș'
    case 0x021A: return 0x021B;   // 'Ț' → 'ț'
    case 0x021C: return 0x021D;   // 'Ȝ' → 'ȝ'
    case 0x021E: return 0x021F;   // 'Ȟ' → 'ȟ'
    case 0x0220: return 0x019E;   // 'Ƞ' → 'ƞ'
    case 0x0222: return 0x0223;   // 'Ȣ' → 'ȣ'
    case 0x0224: return 0x0225;   // 'Ȥ' → 'ȥ'
    case 0x0226: return 0x0227;   // 'Ȧ' → 'ȧ'
    case 0x0228: return 0x0229;   // 'Ȩ' → 'ȩ'
    case 0x022A: return 0x022B;   // 'Ȫ' → 'ȫ'
    case 0x022C: return 0x022D;   // 'Ȭ' → 'ȭ'
    case 0x022E: return 0x022F;   // 'Ȯ' → 'ȯ'
    case 0x0230: return 0x0231;   // 'Ȱ' → 'ȱ'
    case 0x0232: return 0x0233;   // 'Ȳ' → 'ȳ'
    case 0x023A: return 0x2C65;   // 'Ⱥ' → 'ⱥ'
    case 0x023B: return 0x023C;   // 'Ȼ' → 'ȼ'
    case 0x023D: return 0x019A;   // 'Ƚ' → 'ƚ'
    case 0x023E: return 0x2C66;   // 'Ⱦ' → 'ⱦ'
    case 0x0241: return 0x0242;   // 'Ɂ' → 'ɂ'
    case 0x0243: return 0x0180;   // 'Ƀ' → 'ƀ'
    case 0x0244: return 0x0289;   // 'Ʉ' → 'ʉ'
    case 0x0245: return 0x028C;   // 'Ʌ' → 'ʌ'
    case 0x0246: return 0x0247;   // 'Ɇ' → 'ɇ'
    case 0x0248: return 0x0249;   // 'Ɉ' → 'ɉ'
    case 0x024A: return 0x024B;   // 'Ɋ' → 'ɋ'
    case 0x024C: return 0x024D;   // 'Ɍ' → 'ɍ'
    case 0x024E: return 0x024F;   // 'Ɏ' → 'ɏ'
    case 0x0345: return 0x03B9;   // 'ͅ' → 'ι'
    case 0x0370: return 0x0371;   // 'Ͱ' → 'ͱ'
    case 0x0372: return 0x0373;   // 'Ͳ' → 'ͳ'
    case 0x0376: return 0x0377;   // 'Ͷ' → 'ͷ'
    case 0x037F: return 0x03F3;   // 'Ϳ' → 'ϳ'
    case 0x0386: return 0x03AC;   // 'Ά' → 'ά'
    case 0x038C: return 0x03CC;   // 'Ό' → 'ό'
    case 0x038E: return 0x03CD;   // 'Ύ' → 'ύ'
    case 0x038F: return 0x03CE;   // 'Ώ' → 'ώ'
    case 0x03C2: return 0x03C3;   // 'ς' → 'σ'
    case 0x03CF: return 0x03D7;   // 'Ϗ' → 'ϗ'
    case 0x03D0: return 0x03B2;   // 'ϐ' → 'β'
    case 0x03D1: return 0x03B8;   // 'ϑ' → 'θ'
    case 0x03D5: return 0x03C6;   // 'ϕ' → 'φ'
    case 0x03D6: return 0x03C0;   // 'ϖ' → 'π'
    case 0x03D8: return 0x03D9;   // 'Ϙ' → 'ϙ'
    case 0x03DA: return 0x03DB;   // 'Ϛ' → 'ϛ'
    case 0x03DC: return 0x03DD;   // 'Ϝ' → 'ϝ'
    case 0x03DE: return 0x03DF;   // 'Ϟ' → 'ϟ'
    case 0x03E0: return 0x03E1;   // 'Ϡ' → 'ϡ'
    case 0x03E2: return 0x03E3;   // 'Ϣ' → 'ϣ'
    case 0x03E4: return 0x03E5;   // 'Ϥ' → 'ϥ'
    case 0x03E6: return 0x03E7;   // 'Ϧ' → 'ϧ'
    case 0x03E8: return 0x03E9;   // 'Ϩ' → 'ϩ'
    case 0x03EA: return 0x03EB;   // 'Ϫ' → 'ϫ'
    case 0x03EC: return 0x03ED;   // 'Ϭ' → 'ϭ'
    case 0x03EE: return 0x03EF;   // 'Ϯ' → 'ϯ'
    case 0x03F0: return 0x03BA;   // 'ϰ' → 'κ'
    case 0x03F1: return 0x03C1;   // 'ϱ' → 'ρ'
    case 0x03F4: return 0x03B8;   // 'ϴ' → 'θ'
    case 0x03F5: return 0x03B5;   // 'ϵ' → 'ε'
    case 0x03F7: return 0x03F8;   // 'Ϸ' → 'ϸ'
    case 0x03F9: return 0x03F2;   // 'Ϲ' → 'ϲ'
    case 0x03FA: return 0x03FB;   // 'Ϻ' → 'ϻ'
    case 0x0460: return 0x0461;   // 'Ѡ' → 'ѡ'
    case 0x0462: return 0x0463;   // 'Ѣ' → 'ѣ'
    case 0x0464: return 0x0465;   // 'Ѥ' → 'ѥ'
    case 0x0466: return 0x0467;   // 'Ѧ' → 'ѧ'
    case 0x0468: return 0x0469;   // 'Ѩ' → 'ѩ'
    case 0x046A: return 0x046B;   // 'Ѫ' → 'ѫ'
    case 0x046C: return 0x046D;   // 'Ѭ' → 'ѭ'
    case 0x046E: return 0x046F;   // 'Ѯ' → 'ѯ'
    case 0x0470: return 0x0471;   // 'Ѱ' → 'ѱ'
    case 0x0472: return 0x0473;   // 'Ѳ' → 'ѳ'
    case 0x0474: return 0x0475;   // 'Ѵ' → 'ѵ'
    case 0x0476: return 0x0477;   // 'Ѷ' → 'ѷ'
    case 0x0478: return 0x0479;   // 'Ѹ' → 'ѹ'
    case 0x047A: return 0x047B;   // 'Ѻ' → 'ѻ'
    case 0x047C: return 0x047D;   // 'Ѽ' → 'ѽ'
    case 0x047E: return 0x047F;   // 'Ѿ' → 'ѿ'
    case 0x0480: return 0x0481;   // 'Ҁ' → 'ҁ'
    case 0x048A: return 0x048B;   // 'Ҋ' → 'ҋ'
    case 0x048C: return 0x048D;   // 'Ҍ' → 'ҍ'
    case 0x048E: return 0x048F;   // 'Ҏ' → 'ҏ'
    case 0x0490: return 0x0491;   // 'Ґ' → 'ґ'
    case 0x0492: return 0x0493;   // 'Ғ' → 'ғ'
    case 0x0494: return 0x0495;   // 'Ҕ' → 'ҕ'
    case 0x0496: return 0x0497;   // 'Җ' → 'җ'
    case 0x0498: return 0x0499;   // 'Ҙ' → 'ҙ'
    case 0x049A: return 0x049B;   // 'Қ' → 'қ'
    case 0x049C: return 0x049D;   // 'Ҝ' → 'ҝ'
    case 0x049E: return 0x049F;   // 'Ҟ' → 'ҟ'
    case 0x04A0: return 0x04A1;   // 'Ҡ' → 'ҡ'
    case 0x04A2: return 0x04A3;   // 'Ң' → 'ң'
    case 0x04A4: return 0x04A5;   // 'Ҥ' → 'ҥ'
    case 0x04A6: return 0x04A7;   // 'Ҧ' → 'ҧ'
    case 0x04A8: return 0x04A9;   // 'Ҩ' → 'ҩ'
    case 0x04AA: return 0x04AB;   // 'Ҫ' → 'ҫ'
    case 0x04AC: return 0x04AD;   // 'Ҭ' → 'ҭ'
    case 0x04AE: return 0x04AF;   // 'Ү' → 'ү'
    case 0x04B0: return 0x04B1;   // 'Ұ' → 'ұ'
    case 0x04B2: return 0x04B3;   // 'Ҳ' → 'ҳ'
    case 0x04B4: return 0x04B5;   // 'Ҵ' → 'ҵ'
    case 0x04B6: return 0x04B7;   // 'Ҷ' → 'ҷ'
    case 0x04B8: return 0x04B9;   // 'Ҹ' → 'ҹ'
    case 0x04BA: return 0x04BB;   // 'Һ' → 'һ'
    case 0x04BC: return 0x04BD;   // 'Ҽ' → 'ҽ'
    case 0x04BE: return 0x04BF;   // 'Ҿ' → 'ҿ'
    case 0x04C0: return 0x04CF;   // 'Ӏ' → 'ӏ'
    case 0x04C1: return 0x04C2;   // 'Ӂ' → 'ӂ'
    case 0x04C3: return 0x04C4;   // 'Ӄ' → 'ӄ'
    case 0x04C5: return 0x04C6;   // 'Ӆ' → 'ӆ'
    case 0x04C7: return 0x04C8;   // 'Ӈ' → 'ӈ'
    case 0x04C9: return 0x04CA;   // 'Ӊ' → 'ӊ'
    case 0x04CB: return 0x04CC;   // 'Ӌ' → 'ӌ'
    case 0x04CD: return 0x04CE;   // 'Ӎ' → 'ӎ'
    case 0x04D0: return 0x04D1;   // 'Ӑ' → 'ӑ'
    case 0x04D2: return 0x04D3;   // 'Ӓ' → 'ӓ'
    case 0x04D4: return 0x04D5;   // 'Ӕ' → 'ӕ'
    case 0x04D6: return 0x04D7;   // 'Ӗ' → 'ӗ'
    case 0x04D8: return 0x04D9;   // 'Ә' → 'ә'
    case 0x04DA: return 0x04DB;   // 'Ӛ' → 'ӛ'
    case 0x04DC: return 0x04DD;   // 'Ӝ' → 'ӝ'
    case 0x04DE: return 0x04DF;   // 'Ӟ' → 'ӟ'
    case 0x04E0: return 0x04E1;   // 'Ӡ' → 'ӡ'
    case 0x04E2: return 0x04E3;   // 'Ӣ' → 'ӣ'
    case 0x04E4: return 0x04E5;   // 'Ӥ' → 'ӥ'
    case 0x04E6: return 0x04E7;   // 'Ӧ' → 'ӧ'
    case 0x04E8: return 0x04E9;   // 'Ө' → 'ө'
    case 0x04EA: return 0x04EB;   // 'Ӫ' → 'ӫ'
    case 0x04EC: return 0x04ED;   // 'Ӭ' → 'ӭ'
    case 0x04EE: return 0x04EF;   // 'Ӯ' → 'ӯ'
    case 0x04F0: return 0x04F1;   // 'Ӱ' → 'ӱ'
    case 0x04F2: return 0x04F3;   // 'Ӳ' → 'ӳ'
    case 0x04F4: return 0x04F5;   // 'Ӵ' → 'ӵ'
    case 0x04F6: return 0x04F7;   // 'Ӷ' → 'ӷ'
    case 0x04F8: return 0x04F9;   // 'Ӹ' → 'ӹ'
    case 0x04FA: return 0x04FB;   // 'Ӻ' → 'ӻ'
    case 0x04FC: return 0x04FD;   // 'Ӽ' → 'ӽ'
    case 0x04FE: return 0x04FF;   // 'Ӿ' → 'ӿ'
    case 0x0500: return 0x0501;   // 'Ԁ' → 'ԁ'
    case 0x0502: return 0x0503;   // 'Ԃ' → 'ԃ'
    case 0x0504: return 0x0505;   // 'Ԅ' → 'ԅ'
    case 0x0506: return 0x0507;   // 'Ԇ' → 'ԇ'
    case 0x0508: return 0x0509;   // 'Ԉ' → 'ԉ'
    case 0x050A: return 0x050B;   // 'Ԋ' → 'ԋ'
    case 0x050C: return 0x050D;   // 'Ԍ' → 'ԍ'
    case 0x050E: return 0x050F;   // 'Ԏ' → 'ԏ'
    case 0x0510: return 0x0511;   // 'Ԑ' → 'ԑ'
    case 0x0512: return 0x0513;   // 'Ԓ' → 'ԓ'
    case 0x0514: return 0x0515;   // 'Ԕ' → 'ԕ'
    case 0x0516: return 0x0517;   // 'Ԗ' → 'ԗ'
    case 0x0518: return 0x0519;   // 'Ԙ' → 'ԙ'
    case 0x051A: return 0x051B;   // 'Ԛ' → 'ԛ'
    case 0x051C: return 0x051D;   // 'Ԝ' → 'ԝ'
    case 0x051E: return 0x051F;   // 'Ԟ' → 'ԟ'
    case 0x0520: return 0x0521;   // 'Ԡ' → 'ԡ'
    case 0x0522: return 0x0523;   // 'Ԣ' → 'ԣ'
    case 0x0524: return 0x0525;   // 'Ԥ' → 'ԥ'
    case 0x0526: return 0x0527;   // 'Ԧ' → 'ԧ'
    case 0x0528: return 0x0529;   // 'Ԩ' → 'ԩ'
    case 0x052A: return 0x052B;   // 'Ԫ' → 'ԫ'
    case 0x052C: return 0x052D;   // 'Ԭ' → 'ԭ'
    case 0x052E: return 0x052F;   // 'Ԯ' → 'ԯ'
    case 0x10C7: return 0x2D27;   // 'Ⴧ' → 'ⴧ'
    case 0x10CD: return 0x2D2D;   // 'Ⴭ' → 'ⴭ'
    case 0x1C80: return 0x0432;   // 'ᲀ' → 'в'
    case 0x1C81: return 0x0434;   // 'ᲁ' → 'д'
    case 0x1C82: return 0x043E;   // 'ᲂ' → 'о'
    case 0x1C83: return 0x0441;   // 'ᲃ' → 'с'
    case 0x1C84: return 0x0442;   // 'ᲄ' → 'т'
    case 0x1C85: return 0x0442;   // 'ᲅ' → 'т'
    case 0x1C86: return 0x044A;   // 'ᲆ' → 'ъ'
    case 0x1C87: return 0x0463;   // 'ᲇ' → 'ѣ'
    case 0x1C88: return 0xA64B;   // 'ᲈ' → 'ꙋ'
    case 0x1C89: return 0x1C8A;   // 'Ᲊ' → 'ᲊ'
    case 0x1E00: return 0x1E01;   // 'Ḁ' → 'ḁ'
    case 0x1E02: return 0x1E03;   // 'Ḃ' → 'ḃ'
    case 0x1E04: return 0x1E05;   // 'Ḅ' → 'ḅ'
    case 0x1E06: return 0x1E07;   // 'Ḇ' → 'ḇ'
    case 0x1E08: return 0x1E09;   // 'Ḉ' → 'ḉ'
    case 0x1E0A: return 0x1E0B;   // 'Ḋ' → 'ḋ'
    case 0x1E0C: return 0x1E0D;   // 'Ḍ' → 'ḍ'
    case 0x1E0E: return 0x1E0F;   // 'Ḏ' → 'ḏ'
    case 0x1E10: return 0x1E11;   // 'Ḑ' → 'ḑ'
    case 0x1E12: return 0x1E13;   // 'Ḓ' → 'ḓ'
    case 0x1E14: return 0x1E15;   // 'Ḕ' → 'ḕ'
    case 0x1E16: return 0x1E17;   // 'Ḗ' → 'ḗ'
    case 0x1E18: return 0x1E19;   // 'Ḙ' → 'ḙ'
    case 0x1E1A: return 0x1E1B;   // 'Ḛ' → 'ḛ'
    case 0x1E1C: return 0x1E1D;   // 'Ḝ' → 'ḝ'
    case 0x1E1E: return 0x1E1F;   // 'Ḟ' → 'ḟ'
    case 0x1E20: return 0x1E21;   // 'Ḡ' → 'ḡ'
    case 0x1E22: return 0x1E23;   // 'Ḣ' → 'ḣ'
    case 0x1E24: return 0x1E25;   // 'Ḥ' → 'ḥ'
    case 0x1E26: return 0x1E27;   // 'Ḧ' → 'ḧ'
    case 0x1E28: return 0x1E29;   // 'Ḩ' → 'ḩ'
    case 0x1E2A: return 0x1E2B;   // 'Ḫ' → 'ḫ'
    case 0x1E2C: return 0x1E2D;   // 'Ḭ' → 'ḭ'
    case 0x1E2E: return 0x1E2F;   // 'Ḯ' → 'ḯ'
    case 0x1E30: return 0x1E31;   // 'Ḱ' → 'ḱ'
    case 0x1E32: return 0x1E33;   // 'Ḳ' → 'ḳ'
    case 0x1E34: return 0x1E35;   // 'Ḵ' → 'ḵ'
    case 0x1E36: return 0x1E37;   // 'Ḷ' → 'ḷ'
    case 0x1E38: return 0x1E39;   // 'Ḹ' → 'ḹ'
    case 0x1E3A: return 0x1E3B;   // 'Ḻ' → 'ḻ'
    case 0x1E3C: return 0x1E3D;   // 'Ḽ' → 'ḽ'
    case 0x1E3E: return 0x1E3F;   // 'Ḿ' → 'ḿ'
    case 0x1E40: return 0x1E41;   // 'Ṁ' → 'ṁ'
    case 0x1E42: return 0x1E43;   // 'Ṃ' → 'ṃ'
    case 0x1E44: return 0x1E45;   // 'Ṅ' → 'ṅ'
    case 0x1E46: return 0x1E47;   // 'Ṇ' → 'ṇ'
    case 0x1E48: return 0x1E49;   // 'Ṉ' → 'ṉ'
    case 0x1E4A: return 0x1E4B;   // 'Ṋ' → 'ṋ'
    case 0x1E4C: return 0x1E4D;   // 'Ṍ' → 'ṍ'
    case 0x1E4E: return 0x1E4F;   // 'Ṏ' → 'ṏ'
    case 0x1E50: return 0x1E51;   // 'Ṑ' → 'ṑ'
    case 0x1E52: return 0x1E53;   // 'Ṓ' → 'ṓ'
    case 0x1E54: return 0x1E55;   // 'Ṕ' → 'ṕ'
    case 0x1E56: return 0x1E57;   // 'Ṗ' → 'ṗ'
    case 0x1E58: return 0x1E59;   // 'Ṙ' → 'ṙ'
    case 0x1E5A: return 0x1E5B;   // 'Ṛ' → 'ṛ'
    case 0x1E5C: return 0x1E5D;   // 'Ṝ' → 'ṝ'
    case 0x1E5E: return 0x1E5F;   // 'Ṟ' → 'ṟ'
    case 0x1E60: return 0x1E61;   // 'Ṡ' → 'ṡ'
    case 0x1E62: return 0x1E63;   // 'Ṣ' → 'ṣ'
    case 0x1E64: return 0x1E65;   // 'Ṥ' → 'ṥ'
    case 0x1E66: return 0x1E67;   // 'Ṧ' → 'ṧ'
    case 0x1E68: return 0x1E69;   // 'Ṩ' → 'ṩ'
    case 0x1E6A: return 0x1E6B;   // 'Ṫ' → 'ṫ'
    case 0x1E6C: return 0x1E6D;   // 'Ṭ' → 'ṭ'
    case 0x1E6E: return 0x1E6F;   // 'Ṯ' → 'ṯ'
    case 0x1E70: return 0x1E71;   // 'Ṱ' → 'ṱ'
    case 0x1E72: return 0x1E73;   // 'Ṳ' → 'ṳ'
    case 0x1E74: return 0x1E75;   // 'Ṵ' → 'ṵ'
    case 0x1E76: return 0x1E77;   // 'Ṷ' → 'ṷ'
    case 0x1E78: return 0x1E79;   // 'Ṹ' → 'ṹ'
    case 0x1E7A: return 0x1E7B;   // 'Ṻ' → 'ṻ'
    case 0x1E7C: return 0x1E7D;   // 'Ṽ' → 'ṽ'
    case 0x1E7E: return 0x1E7F;   // 'Ṿ' → 'ṿ'
    case 0x1E80: return 0x1E81;   // 'Ẁ' → 'ẁ'
    case 0x1E82: return 0x1E83;   // 'Ẃ' → 'ẃ'
    case 0x1E84: return 0x1E85;   // 'Ẅ' → 'ẅ'
    case 0x1E86: return 0x1E87;   // 'Ẇ' → 'ẇ'
    case 0x1E88: return 0x1E89;   // 'Ẉ' → 'ẉ'
    case 0x1E8A: return 0x1E8B;   // 'Ẋ' → 'ẋ'
    case 0x1E8C: return 0x1E8D;   // 'Ẍ' → 'ẍ'
    case 0x1E8E: return 0x1E8F;   // 'Ẏ' → 'ẏ'
    case 0x1E90: return 0x1E91;   // 'Ẑ' → 'ẑ'
    case 0x1E92: return 0x1E93;   // 'Ẓ' → 'ẓ'
    case 0x1E94: return 0x1E95;   // 'Ẕ' → 'ẕ'
    case 0x1E9B: return 0x1E61;   // 'ẛ' → 'ṡ'
    case 0x1EA0: return 0x1EA1;   // 'Ạ' → 'ạ'
    case 0x1EA2: return 0x1EA3;   // 'Ả' → 'ả'
    case 0x1EA4: return 0x1EA5;   // 'Ấ' → 'ấ'
    case 0x1EA6: return 0x1EA7;   // 'Ầ' → 'ầ'
    case 0x1EA8: return 0x1EA9;   // 'Ẩ' → 'ẩ'
    case 0x1EAA: return 0x1EAB;   // 'Ẫ' → 'ẫ'
    case 0x1EAC: return 0x1EAD;   // 'Ậ' → 'ậ'
    case 0x1EAE: return 0x1EAF;   // 'Ắ' → 'ắ'
    case 0x1EB0: return 0x1EB1;   // 'Ằ' → 'ằ'
    case 0x1EB2: return 0x1EB3;   // 'Ẳ' → 'ẳ'
    case 0x1EB4: return 0x1EB5;   // 'Ẵ' → 'ẵ'
    case 0x1EB6: return 0x1EB7;   // 'Ặ' → 'ặ'
    case 0x1EB8: return 0x1EB9;   // 'Ẹ' → 'ẹ'
    case 0x1EBA: return 0x1EBB;   // 'Ẻ' → 'ẻ'
    case 0x1EBC: return 0x1EBD;   // 'Ẽ' → 'ẽ'
    case 0x1EBE: return 0x1EBF;   // 'Ế' → 'ế'
    case 0x1EC0: return 0x1EC1;   // 'Ề' → 'ề'
    case 0x1EC2: return 0x1EC3;   // 'Ể' → 'ể'
    case 0x1EC4: return 0x1EC5;   // 'Ễ' → 'ễ'
    case 0x1EC6: return 0x1EC7;   // 'Ệ' → 'ệ'
    case 0x1EC8: return 0x1EC9;   // 'Ỉ' → 'ỉ'
    case 0x1ECA: return 0x1ECB;   // 'Ị' → 'ị'
    case 0x1ECC: return 0x1ECD;   // 'Ọ' → 'ọ'
    case 0x1ECE: return 0x1ECF;   // 'Ỏ' → 'ỏ'
    case 0x1ED0: return 0x1ED1;   // 'Ố' → 'ố'
    case 0x1ED2: return 0x1ED3;   // 'Ồ' → 'ồ'
    case 0x1ED4: return 0x1ED5;   // 'Ổ' → 'ổ'
    case 0x1ED6: return 0x1ED7;   // 'Ỗ' → 'ỗ'
    case 0x1ED8: return 0x1ED9;   // 'Ộ' → 'ộ'
    case 0x1EDA: return 0x1EDB;   // 'Ớ' → 'ớ'
    case 0x1EDC: return 0x1EDD;   // 'Ờ' → 'ờ'
    case 0x1EDE: return 0x1EDF;   // 'Ở' → 'ở'
    case 0x1EE0: return 0x1EE1;   // 'Ỡ' → 'ỡ'
    case 0x1EE2: return 0x1EE3;   // 'Ợ' → 'ợ'
    case 0x1EE4: return 0x1EE5;   // 'Ụ' → 'ụ'
    case 0x1EE6: return 0x1EE7;   // 'Ủ' → 'ủ'
    case 0x1EE8: return 0x1EE9;   // 'Ứ' → 'ứ'
    case 0x1EEA: return 0x1EEB;   // 'Ừ' → 'ừ'
    case 0x1EEC: return 0x1EED;   // 'Ử' → 'ử'
    case 0x1EEE: return 0x1EEF;   // 'Ữ' → 'ữ'
    case 0x1EF0: return 0x1EF1;   // 'Ự' → 'ự'
    case 0x1EF2: return 0x1EF3;   // 'Ỳ' → 'ỳ'
    case 0x1EF4: return 0x1EF5;   // 'Ỵ' → 'ỵ'
    case 0x1EF6: return 0x1EF7;   // 'Ỷ' → 'ỷ'
    case 0x1EF8: return 0x1EF9;   // 'Ỹ' → 'ỹ'
    case 0x1EFA: return 0x1EFB;   // 'Ỻ' → 'ỻ'
    case 0x1EFC: return 0x1EFD;   // 'Ỽ' → 'ỽ'
    case 0x1EFE: return 0x1EFF;   // 'Ỿ' → 'ỿ'
    case 0x1F59: return 0x1F51;   // 'Ὑ' → 'ὑ'
    case 0x1F5B: return 0x1F53;   // 'Ὓ' → 'ὓ'
    case 0x1F5D: return 0x1F55;   // 'Ὕ' → 'ὕ'
    case 0x1F5F: return 0x1F57;   // 'Ὗ' → 'ὗ'
    case 0x1FB8: return 0x1FB0;   // 'Ᾰ' → 'ᾰ'
    case 0x1FB9: return 0x1FB1;   // 'Ᾱ' → 'ᾱ'
    case 0x1FBA: return 0x1F70;   // 'Ὰ' → 'ὰ'
    case 0x1FBB: return 0x1F71;   // 'Ά' → 'ά'
    case 0x1FBE: return 0x03B9;   // 'ι' → 'ι'
    case 0x1FD8: return 0x1FD0;   // 'Ῐ' → 'ῐ'
    case 0x1FD9: return 0x1FD1;   // 'Ῑ' → 'ῑ'
    case 0x1FDA: return 0x1F76;   // 'Ὶ' → 'ὶ'
    case 0x1FDB: return 0x1F77;   // 'Ί' → 'ί'
    case 0x1FE8: return 0x1FE0;   // 'Ῠ' → 'ῠ'
    case 0x1FE9: return 0x1FE1;   // 'Ῡ' → 'ῡ'
    case 0x1FEA: return 0x1F7A;   // 'Ὺ' → 'ὺ'
    case 0x1FEB: return 0x1F7B;   // 'Ύ' → 'ύ'
    case 0x1FEC: return 0x1FE5;   // 'Ῥ' → 'ῥ'
    case 0x1FF8: return 0x1F78;   // 'Ὸ' → 'ὸ'
    case 0x1FF9: return 0x1F79;   // 'Ό' → 'ό'
    case 0x1FFA: return 0x1F7C;   // 'Ὼ' → 'ὼ'
    case 0x1FFB: return 0x1F7D;   // 'Ώ' → 'ώ'
    case 0x2126: return 0x03C9;   // 'Ω' → 'ω'
    case 0x212A: return 0x006B;   // 'K' → 'k'
    case 0x212B: return 0x00E5;   // 'Å' → 'å'
    case 0x2132: return 0x214E;   // 'Ⅎ' → 'ⅎ'
    case 0x2183: return 0x2184;   // 'Ↄ' → 'ↄ'
    case 0x2C60: return 0x2C61;   // 'Ⱡ' → 'ⱡ'
    case 0x2C62: return 0x026B;   // 'Ɫ' → 'ɫ'
    case 0x2C63: return 0x1D7D;   // 'Ᵽ' → 'ᵽ'
    case 0x2C64: return 0x027D;   // 'Ɽ' → 'ɽ'
    case 0x2C67: return 0x2C68;   // 'Ⱨ' → 'ⱨ'
    case 0x2C69: return 0x2C6A;   // 'Ⱪ' → 'ⱪ'
    case 0x2C6B: return 0x2C6C;   // 'Ⱬ' → 'ⱬ'
    case 0x2C6D: return 0x0251;   // 'Ɑ' → 'ɑ'
    case 0x2C6E: return 0x0271;   // 'Ɱ' → 'ɱ'
    case 0x2C6F: return 0x0250;   // 'Ɐ' → 'ɐ'
    case 0x2C70: return 0x0252;   // 'Ɒ' → 'ɒ'
    case 0x2C72: return 0x2C73;   // 'Ⱳ' → 'ⱳ'
    case 0x2C75: return 0x2C76;   // 'Ⱶ' → 'ⱶ'
    case 0x2C7E: return 0x023F;   // 'Ȿ' → 'ȿ'
    case 0x2C7F: return 0x0240;   // 'Ɀ' → 'ɀ'
    case 0x2C80: return 0x2C81;   // 'Ⲁ' → 'ⲁ'
    case 0x2C82: return 0x2C83;   // 'Ⲃ' → 'ⲃ'
    case 0x2C84: return 0x2C85;   // 'Ⲅ' → 'ⲅ'
    case 0x2C86: return 0x2C87;   // 'Ⲇ' → 'ⲇ'
    case 0x2C88: return 0x2C89;   // 'Ⲉ' → 'ⲉ'
    case 0x2C8A: return 0x2C8B;   // 'Ⲋ' → 'ⲋ'
    case 0x2C8C: return 0x2C8D;   // 'Ⲍ' → 'ⲍ'
    case 0x2C8E: return 0x2C8F;   // 'Ⲏ' → 'ⲏ'
    case 0x2C90: return 0x2C91;   // 'Ⲑ' → 'ⲑ'
    case 0x2C92: return 0x2C93;   // 'Ⲓ' → 'ⲓ'
    case 0x2C94: return 0x2C95;   // 'Ⲕ' → 'ⲕ'
    case 0x2C96: return 0x2C97;   // 'Ⲗ' → 'ⲗ'
    case 0x2C98: return 0x2C99;   // 'Ⲙ' → 'ⲙ'
    case 0x2C9A: return 0x2C9B;   // 'Ⲛ' → 'ⲛ'
    case 0x2C9C: return 0x2C9D;   // 'Ⲝ' → 'ⲝ'
    case 0x2C9E: return 0x2C9F;   // 'Ⲟ' → 'ⲟ'
    case 0x2CA0: return 0x2CA1;   // 'Ⲡ' → 'ⲡ'
    case 0x2CA2: return 0x2CA3;   // 'Ⲣ' → 'ⲣ'
    case 0x2CA4: return 0x2CA5;   // 'Ⲥ' → 'ⲥ'
    case 0x2CA6: return 0x2CA7;   // 'Ⲧ' → 'ⲧ'
    case 0x2CA8: return 0x2CA9;   // 'Ⲩ' → 'ⲩ'
    case 0x2CAA: return 0x2CAB;   // 'Ⲫ' → 'ⲫ'
    case 0x2CAC: return 0x2CAD;   // 'Ⲭ' → 'ⲭ'
    case 0x2CAE: return 0x2CAF;   // 'Ⲯ' → 'ⲯ'
    case 0x2CB0: return 0x2CB1;   // 'Ⲱ' → 'ⲱ'
    case 0x2CB2: return 0x2CB3;   // 'Ⲳ' → 'ⲳ'
    case 0x2CB4: return 0x2CB5;   // 'Ⲵ' → 'ⲵ'
    case 0x2CB6: return 0x2CB7;   // 'Ⲷ' → 'ⲷ'
    case 0x2CB8: return 0x2CB9;   // 'Ⲹ' → 'ⲹ'
    case 0x2CBA: return 0x2CBB;   // 'Ⲻ' → 'ⲻ'
    case 0x2CBC: return 0x2CBD;   // 'Ⲽ' → 'ⲽ'
    case 0x2CBE: return 0x2CBF;   // 'Ⲿ' → 'ⲿ'
    case 0x2CC0: return 0x2CC1;   // 'Ⳁ' → 'ⳁ'
    case 0x2CC2: return 0x2CC3;   // 'Ⳃ' → 'ⳃ'
    case 0x2CC4: return 0x2CC5;   // 'Ⳅ' → 'ⳅ'
    case 0x2CC6: return 0x2CC7;   // 'Ⳇ' → 'ⳇ'
    case 0x2CC8: return 0x2CC9;   // 'Ⳉ' → 'ⳉ'
    case 0x2CCA: return 0x2CCB;   // 'Ⳋ' → 'ⳋ'
    case 0x2CCC: return 0x2CCD;   // 'Ⳍ' → 'ⳍ'
    case 0x2CCE: return 0x2CCF;   // 'Ⳏ' → 'ⳏ'
    case 0x2CD0: return 0x2CD1;   // 'Ⳑ' → 'ⳑ'
    case 0x2CD2: return 0x2CD3;   // 'Ⳓ' → 'ⳓ'
    case 0x2CD4: return 0x2CD5;   // 'Ⳕ' → 'ⳕ'
    case 0x2CD6: return 0x2CD7;   // 'Ⳗ' → 'ⳗ'
    case 0x2CD8: return 0x2CD9;   // 'Ⳙ' → 'ⳙ'
    case 0x2CDA: return 0x2CDB;   // 'Ⳛ' → 'ⳛ'
    case 0x2CDC: return 0x2CDD;   // 'Ⳝ' → 'ⳝ'
    case 0x2CDE: return 0x2CDF;   // 'Ⳟ' → 'ⳟ'
    case 0x2CE0: return 0x2CE1;   // 'Ⳡ' → 'ⳡ'
    case 0x2CE2: return 0x2CE3;   // 'Ⳣ' → 'ⳣ'
    case 0x2CEB: return 0x2CEC;   // 'Ⳬ' → 'ⳬ'
    case 0x2CED: return 0x2CEE;   // 'Ⳮ' → 'ⳮ'
    case 0x2CF2: return 0x2CF3;   // 'Ⳳ' → 'ⳳ'
    case 0xA640: return 0xA641;   // 'Ꙁ' → 'ꙁ'
    case 0xA642: return 0xA643;   // 'Ꙃ' → 'ꙃ'
    case 0xA644: return 0xA645;   // 'Ꙅ' → 'ꙅ'
    case 0xA646: return 0xA647;   // 'Ꙇ' → 'ꙇ'
    case 0xA648: return 0xA649;   // 'Ꙉ' → 'ꙉ'
    case 0xA64A: return 0xA64B;   // 'Ꙋ' → 'ꙋ'
    case 0xA64C: return 0xA64D;   // 'Ꙍ' → 'ꙍ'
    case 0xA64E: return 0xA64F;   // 'Ꙏ' → 'ꙏ'
    case 0xA650: return 0xA651;   // 'Ꙑ' → 'ꙑ'
    case 0xA652: return 0xA653;   // 'Ꙓ' → 'ꙓ'
    case 0xA654: return 0xA655;   // 'Ꙕ' → 'ꙕ'
    case 0xA656: return 0xA657;   // 'Ꙗ' → 'ꙗ'
    case 0xA658: return 0xA659;   // 'Ꙙ' → 'ꙙ'
    case 0xA65A: return 0xA65B;   // 'Ꙛ' → 'ꙛ'
    case 0xA65C: return 0xA65D;   // 'Ꙝ' → 'ꙝ'
    case 0xA65E: return 0xA65F;   // 'Ꙟ' → 'ꙟ'
    case 0xA660: return 0xA661;   // 'Ꙡ' → 'ꙡ'
    case 0xA662: return 0xA663;   // 'Ꙣ' → 'ꙣ'
    case 0xA664: return 0xA665;   // 'Ꙥ' → 'ꙥ'
    case 0xA666: return 0xA667;   // 'Ꙧ' → 'ꙧ'
    case 0xA668: return 0xA669;   // 'Ꙩ' → 'ꙩ'
    case 0xA66A: return 0xA66B;   // 'Ꙫ' → 'ꙫ'
    case 0xA66C: return 0xA66D;   // 'Ꙭ' → 'ꙭ'
    case 0xA680: return 0xA681;   // 'Ꚁ' → 'ꚁ'
    case 0xA682: return 0xA683;   // 'Ꚃ' → 'ꚃ'
    case 0xA684: return 0xA685;   // 'Ꚅ' → 'ꚅ'
    case 0xA686: return 0xA687;   // 'Ꚇ' → 'ꚇ'
    case 0xA688: return 0xA689;   // 'Ꚉ' → 'ꚉ'
    case 0xA68A: return 0xA68B;   // 'Ꚋ' → 'ꚋ'
    case 0xA68C: return 0xA68D;   // 'Ꚍ' → 'ꚍ'
    case 0xA68E: return 0xA68F;   // 'Ꚏ' → 'ꚏ'
    case 0xA690: return 0xA691;   // 'Ꚑ' → 'ꚑ'
    case 0xA692: return 0xA693;   // 'Ꚓ' → 'ꚓ'
    case 0xA694: return 0xA695;   // 'Ꚕ' → 'ꚕ'
    case 0xA696: return 0xA697;   // 'Ꚗ' → 'ꚗ'
    case 0xA698: return 0xA699;   // 'Ꚙ' → 'ꚙ'
    case 0xA69A: return 0xA69B;   // 'Ꚛ' → 'ꚛ'
    case 0xA722: return 0xA723;   // 'Ꜣ' → 'ꜣ'
    case 0xA724: return 0xA725;   // 'Ꜥ' → 'ꜥ'
    case 0xA726: return 0xA727;   // 'Ꜧ' → 'ꜧ'
    case 0xA728: return 0xA729;   // 'Ꜩ' → 'ꜩ'
    case 0xA72A: return 0xA72B;   // 'Ꜫ' → 'ꜫ'
    case 0xA72C: return 0xA72D;   // 'Ꜭ' → 'ꜭ'
    case 0xA72E: return 0xA72F;   // 'Ꜯ' → 'ꜯ'
    case 0xA732: return 0xA733;   // 'Ꜳ' → 'ꜳ'
    case 0xA734: return 0xA735;   // 'Ꜵ' → 'ꜵ'
    case 0xA736: return 0xA737;   // 'Ꜷ' → 'ꜷ'
    case 0xA738: return 0xA739;   // 'Ꜹ' → 'ꜹ'
    case 0xA73A: return 0xA73B;   // 'Ꜻ' → 'ꜻ'
    case 0xA73C: return 0xA73D;   // 'Ꜽ' → 'ꜽ'
    case 0xA73E: return 0xA73F;   // 'Ꜿ' → 'ꜿ'
    case 0xA740: return 0xA741;   // 'Ꝁ' → 'ꝁ'
    case 0xA742: return 0xA743;   // 'Ꝃ' → 'ꝃ'
    case 0xA744: return 0xA745;   // 'Ꝅ' → 'ꝅ'
    case 0xA746: return 0xA747;   // 'Ꝇ' → 'ꝇ'
    case 0xA748: return 0xA749;   // 'Ꝉ' → 'ꝉ'
    case 0xA74A: return 0xA74B;   // 'Ꝋ' → 'ꝋ'
    case 0xA74C: return 0xA74D;   // 'Ꝍ' → 'ꝍ'
    case 0xA74E: return 0xA74F;   // 'Ꝏ' → 'ꝏ'
    case 0xA750: return 0xA751;   // 'Ꝑ' → 'ꝑ'
    case 0xA752: return 0xA753;   // 'Ꝓ' → 'ꝓ'
    case 0xA754: return 0xA755;   // 'Ꝕ' → 'ꝕ'
    case 0xA756: return 0xA757;   // 'Ꝗ' → 'ꝗ'
    case 0xA758: return 0xA759;   // 'Ꝙ' → 'ꝙ'
    case 0xA75A: return 0xA75B;   // 'Ꝛ' → 'ꝛ'
    case 0xA75C: return 0xA75D;   // 'Ꝝ' → 'ꝝ'
    case 0xA75E: return 0xA75F;   // 'Ꝟ' → 'ꝟ'
    case 0xA760: return 0xA761;   // 'Ꝡ' → 'ꝡ'
    case 0xA762: return 0xA763;   // 'Ꝣ' → 'ꝣ'
    case 0xA764: return 0xA765;   // 'Ꝥ' → 'ꝥ'
    case 0xA766: return 0xA767;   // 'Ꝧ' → 'ꝧ'
    case 0xA768: return 0xA769;   // 'Ꝩ' → 'ꝩ'
    case 0xA76A: return 0xA76B;   // 'Ꝫ' → 'ꝫ'
    case 0xA76C: return 0xA76D;   // 'Ꝭ' → 'ꝭ'
    case 0xA76E: return 0xA76F;   // 'Ꝯ' → 'ꝯ'
    case 0xA779: return 0xA77A;   // 'Ꝺ' → 'ꝺ'
    case 0xA77B: return 0xA77C;   // 'Ꝼ' → 'ꝼ'
    case 0xA77D: return 0x1D79;   // 'Ᵹ' → 'ᵹ'
    case 0xA77E: return 0xA77F;   // 'Ꝿ' → 'ꝿ'
    case 0xA780: return 0xA781;   // 'Ꞁ' → 'ꞁ'
    case 0xA782: return 0xA783;   // 'Ꞃ' → 'ꞃ'
    case 0xA784: return 0xA785;   // 'Ꞅ' → 'ꞅ'
    case 0xA786: return 0xA787;   // 'Ꞇ' → 'ꞇ'
    case 0xA78B: return 0xA78C;   // 'Ꞌ' → 'ꞌ'
    case 0xA78D: return 0x0265;   // 'Ɥ' → 'ɥ'
    case 0xA790: return 0xA791;   // 'Ꞑ' → 'ꞑ'
    case 0xA792: return 0xA793;   // 'Ꞓ' → 'ꞓ'
    case 0xA796: return 0xA797;   // 'Ꞗ' → 'ꞗ'
    case 0xA798: return 0xA799;   // 'Ꞙ' → 'ꞙ'
    case 0xA79A: return 0xA79B;   // 'Ꞛ' → 'ꞛ'
    case 0xA79C: return 0xA79D;   // 'Ꞝ' → 'ꞝ'
    case 0xA79E: return 0xA79F;   // 'Ꞟ' → 'ꞟ'
    case 0xA7A0: return 0xA7A1;   // 'Ꞡ' → 'ꞡ'
    case 0xA7A2: return 0xA7A3;   // 'Ꞣ' → 'ꞣ'
    case 0xA7A4: return 0xA7A5;   // 'Ꞥ' → 'ꞥ'
    case 0xA7A6: return 0xA7A7;   // 'Ꞧ' → 'ꞧ'
    case 0xA7A8: return 0xA7A9;   // 'Ꞩ' → 'ꞩ'
    case 0xA7AA: return 0x0266;   // 'Ɦ' → 'ɦ'
    case 0xA7AB: return 0x025C;   // 'Ɜ' → 'ɜ'
    case 0xA7AC: return 0x0261;   // 'Ɡ' → 'ɡ'
    case 0xA7AD: return 0x026C;   // 'Ɬ' → 'ɬ'
    case 0xA7AE: return 0x026A;   // 'Ɪ' → 'ɪ'
    case 0xA7B0: return 0x029E;   // 'Ʞ' → 'ʞ'
    case 0xA7B1: return 0x0287;   // 'Ʇ' → 'ʇ'
    case 0xA7B2: return 0x029D;   // 'Ʝ' → 'ʝ'
    case 0xA7B3: return 0xAB53;   // 'Ꭓ' → 'ꭓ'
    case 0xA7B4: return 0xA7B5;   // 'Ꞵ' → 'ꞵ'
    case 0xA7B6: return 0xA7B7;   // 'Ꞷ' → 'ꞷ'
    case 0xA7B8: return 0xA7B9;   // 'Ꞹ' → 'ꞹ'
    case 0xA7BA: return 0xA7BB;   // 'Ꞻ' → 'ꞻ'
    case 0xA7BC: return 0xA7BD;   // 'Ꞽ' → 'ꞽ'
    case 0xA7BE: return 0xA7BF;   // 'Ꞿ' → 'ꞿ'
    case 0xA7C0: return 0xA7C1;   // 'Ꟁ' → 'ꟁ'
    case 0xA7C2: return 0xA7C3;   // 'Ꟃ' → 'ꟃ'
    case 0xA7C4: return 0xA794;   // 'Ꞔ' → 'ꞔ'
    case 0xA7C5: return 0x0282;   // 'Ʂ' → 'ʂ'
    case 0xA7C6: return 0x1D8E;   // 'Ᶎ' → 'ᶎ'
    case 0xA7C7: return 0xA7C8;   // 'Ꟈ' → 'ꟈ'
    case 0xA7C9: return 0xA7CA;   // 'Ꟊ' → 'ꟊ'
    case 0xA7CB: return 0x0264;   // 'Ɤ' → 'ɤ'
    case 0xA7CC: return 0xA7CD;   // 'Ꟍ' → 'ꟍ'
    case 0xA7CE: return 0xA7CF;   // '꟎' → '꟏'
    case 0xA7D0: return 0xA7D1;   // 'Ꟑ' → 'ꟑ'
    case 0xA7D2: return 0xA7D3;   // '꟒' → 'ꟓ'
    case 0xA7D4: return 0xA7D5;   // '꟔' → 'ꟕ'
    case 0xA7D6: return 0xA7D7;   // 'Ꟗ' → 'ꟗ'
    case 0xA7D8: return 0xA7D9;   // 'Ꟙ' → 'ꟙ'
    case 0xA7DA: return 0xA7DB;   // 'Ꟛ' → 'ꟛ'
    case 0xA7DC: return 0x019B;   // 'Ƛ' → 'ƛ'
    case 0xA7F5: return 0xA7F6;   // 'Ꟶ' → 'ꟶ'
    case 0x10594: return 0x105BB; // '𐖔' → '𐖻'
    case 0x10595: return 0x105BC; // '𐖕' → '𐖼'
    }

    // One-to-many expansions
    switch (cp) {
    case 0x00DF: expansion[0] = 0x0073, expansion[1] = 0x0073, *expansion_count = 2; return 0x0073; // ß → ss
    case 0x0130: expansion[0] = 0x0069, expansion[1] = 0x0307, *expansion_count = 2; return 0x0069; // İ → i + combining
    case 0x0149: expansion[0] = 0x02BC, expansion[1] = 0x006E, *expansion_count = 2; return 0x02BC; // ŉ → ʼn
    case 0x01F0: expansion[0] = 0x006A, expansion[1] = 0x030C, *expansion_count = 2; return 0x006A; // ǰ → j + combining
    case 0x0390:
        expansion[0] = 0x03B9, expansion[1] = 0x0308, expansion[2] = 0x0301, *expansion_count = 3;
        return 0x03B9; // ΐ → ι + 2 combining
    case 0x03B0:
        expansion[0] = 0x03C5, expansion[1] = 0x0308, expansion[2] = 0x0301, *expansion_count = 3;
        return 0x03C5; // ΰ → υ + 2 combining
    case 0x0587: expansion[0] = 0x0565, expansion[1] = 0x0582, *expansion_count = 2; return 0x0565; // և → եւ
    case 0x1E96: expansion[0] = 0x0068, expansion[1] = 0x0331, *expansion_count = 2; return 0x0068; // ẖ → h + combining
    case 0x1E97: expansion[0] = 0x0074, expansion[1] = 0x0308, *expansion_count = 2; return 0x0074; // ẗ → t + combining
    case 0x1E98: expansion[0] = 0x0077, expansion[1] = 0x030A, *expansion_count = 2; return 0x0077; // ẘ → w + combining
    case 0x1E99: expansion[0] = 0x0079, expansion[1] = 0x030A, *expansion_count = 2; return 0x0079; // ẙ → y + combining
    case 0x1E9A: expansion[0] = 0x0061, expansion[1] = 0x02BE, *expansion_count = 2; return 0x0061; // ẚ → aʾ
    case 0x1E9E: expansion[0] = 0x0073, expansion[1] = 0x0073, *expansion_count = 2; return 0x0073; // ẞ → ss
    case 0x1F50: expansion[0] = 0x03C5, expansion[1] = 0x0313, *expansion_count = 2; return 0x03C5; // ὐ → υ + combining
    case 0x1F52:
        expansion[0] = 0x03C5, expansion[1] = 0x0313, expansion[2] = 0x0300, *expansion_count = 3;
        return 0x03C5; // ὒ → υ + 2 combining
    case 0x1F54:
        expansion[0] = 0x03C5, expansion[1] = 0x0313, expansion[2] = 0x0301, *expansion_count = 3;
        return 0x03C5; // ὔ → υ + 2 combining
    case 0x1F56:
        expansion[0] = 0x03C5, expansion[1] = 0x0313, expansion[2] = 0x0342, *expansion_count = 3;
        return 0x03C5; // ὖ → υ + 2 combining
    case 0x1F80: expansion[0] = 0x1F00, expansion[1] = 0x03B9, *expansion_count = 2; return 0x1F00; // ᾀ → ἀι
    case 0x1F81: expansion[0] = 0x1F01, expansion[1] = 0x03B9, *expansion_count = 2; return 0x1F01; // ᾁ → ἁι
    case 0x1F82: expansion[0] = 0x1F02, expansion[1] = 0x03B9, *expansion_count = 2; return 0x1F02; // ᾂ → ἂι
    case 0x1F83: expansion[0] = 0x1F03, expansion[1] = 0x03B9, *expansion_count = 2; return 0x1F03; // ᾃ → ἃι
    case 0x1F84: expansion[0] = 0x1F04, expansion[1] = 0x03B9, *expansion_count = 2; return 0x1F04; // ᾄ → ἄι
    case 0x1F85: expansion[0] = 0x1F05, expansion[1] = 0x03B9, *expansion_count = 2; return 0x1F05; // ᾅ → ἅι
    case 0x1F86: expansion[0] = 0x1F06, expansion[1] = 0x03B9, *expansion_count = 2; return 0x1F06; // ᾆ → ἆι
    case 0x1F87: expansion[0] = 0x1F07, expansion[1] = 0x03B9, *expansion_count = 2; return 0x1F07; // ᾇ → ἇι
    case 0x1F88: expansion[0] = 0x1F00, expansion[1] = 0x03B9, *expansion_count = 2; return 0x1F00; // ᾈ → ἀι
    case 0x1F89: expansion[0] = 0x1F01, expansion[1] = 0x03B9, *expansion_count = 2; return 0x1F01; // ᾉ → ἁι
    case 0x1F8A: expansion[0] = 0x1F02, expansion[1] = 0x03B9, *expansion_count = 2; return 0x1F02; // ᾊ → ἂι
    case 0x1F8B: expansion[0] = 0x1F03, expansion[1] = 0x03B9, *expansion_count = 2; return 0x1F03; // ᾋ → ἃι
    case 0x1F8C: expansion[0] = 0x1F04, expansion[1] = 0x03B9, *expansion_count = 2; return 0x1F04; // ᾌ → ἄι
    case 0x1F8D: expansion[0] = 0x1F05, expansion[1] = 0x03B9, *expansion_count = 2; return 0x1F05; // ᾍ → ἅι
    case 0x1F8E: expansion[0] = 0x1F06, expansion[1] = 0x03B9, *expansion_count = 2; return 0x1F06; // ᾎ → ἆι
    case 0x1F8F: expansion[0] = 0x1F07, expansion[1] = 0x03B9, *expansion_count = 2; return 0x1F07; // ᾏ → ἇι
    case 0x1F90: expansion[0] = 0x1F20, expansion[1] = 0x03B9, *expansion_count = 2; return 0x1F20; // ᾐ → ἠι
    case 0x1F91: expansion[0] = 0x1F21, expansion[1] = 0x03B9, *expansion_count = 2; return 0x1F21; // ᾑ → ἡι
    case 0x1F92: expansion[0] = 0x1F22, expansion[1] = 0x03B9, *expansion_count = 2; return 0x1F22; // ᾒ → ἢι
    case 0x1F93: expansion[0] = 0x1F23, expansion[1] = 0x03B9, *expansion_count = 2; return 0x1F23; // ᾓ → ἣι
    case 0x1F94: expansion[0] = 0x1F24, expansion[1] = 0x03B9, *expansion_count = 2; return 0x1F24; // ᾔ → ἤι
    case 0x1F95: expansion[0] = 0x1F25, expansion[1] = 0x03B9, *expansion_count = 2; return 0x1F25; // ᾕ → ἥι
    case 0x1F96: expansion[0] = 0x1F26, expansion[1] = 0x03B9, *expansion_count = 2; return 0x1F26; // ᾖ → ἦι
    case 0x1F97: expansion[0] = 0x1F27, expansion[1] = 0x03B9, *expansion_count = 2; return 0x1F27; // ᾗ → ἧι
    case 0x1F98: expansion[0] = 0x1F20, expansion[1] = 0x03B9, *expansion_count = 2; return 0x1F20; // ᾘ → ἠι
    case 0x1F99: expansion[0] = 0x1F21, expansion[1] = 0x03B9, *expansion_count = 2; return 0x1F21; // ᾙ → ἡι
    case 0x1F9A: expansion[0] = 0x1F22, expansion[1] = 0x03B9, *expansion_count = 2; return 0x1F22; // ᾚ → ἢι
    case 0x1F9B: expansion[0] = 0x1F23, expansion[1] = 0x03B9, *expansion_count = 2; return 0x1F23; // ᾛ → ἣι
    case 0x1F9C: expansion[0] = 0x1F24, expansion[1] = 0x03B9, *expansion_count = 2; return 0x1F24; // ᾜ → ἤι
    case 0x1F9D: expansion[0] = 0x1F25, expansion[1] = 0x03B9, *expansion_count = 2; return 0x1F25; // ᾝ → ἥι
    case 0x1F9E: expansion[0] = 0x1F26, expansion[1] = 0x03B9, *expansion_count = 2; return 0x1F26; // ᾞ → ἦι
    case 0x1F9F: expansion[0] = 0x1F27, expansion[1] = 0x03B9, *expansion_count = 2; return 0x1F27; // ᾟ → ἧι
    case 0x1FA0: expansion[0] = 0x1F60, expansion[1] = 0x03B9, *expansion_count = 2; return 0x1F60; // ᾠ → ὠι
    case 0x1FA1: expansion[0] = 0x1F61, expansion[1] = 0x03B9, *expansion_count = 2; return 0x1F61; // ᾡ → ὡι
    case 0x1FA2: expansion[0] = 0x1F62, expansion[1] = 0x03B9, *expansion_count = 2; return 0x1F62; // ᾢ → ὢι
    case 0x1FA3: expansion[0] = 0x1F63, expansion[1] = 0x03B9, *expansion_count = 2; return 0x1F63; // ᾣ → ὣι
    case 0x1FA4: expansion[0] = 0x1F64, expansion[1] = 0x03B9, *expansion_count = 2; return 0x1F64; // ᾤ → ὤι
    case 0x1FA5: expansion[0] = 0x1F65, expansion[1] = 0x03B9, *expansion_count = 2; return 0x1F65; // ᾥ → ὥι
    case 0x1FA6: expansion[0] = 0x1F66, expansion[1] = 0x03B9, *expansion_count = 2; return 0x1F66; // ᾦ → ὦι
    case 0x1FA7: expansion[0] = 0x1F67, expansion[1] = 0x03B9, *expansion_count = 2; return 0x1F67; // ᾧ → ὧι
    case 0x1FA8: expansion[0] = 0x1F60, expansion[1] = 0x03B9, *expansion_count = 2; return 0x1F60; // ᾨ → ὠι
    case 0x1FA9: expansion[0] = 0x1F61, expansion[1] = 0x03B9, *expansion_count = 2; return 0x1F61; // ᾩ → ὡι
    case 0x1FAA: expansion[0] = 0x1F62, expansion[1] = 0x03B9, *expansion_count = 2; return 0x1F62; // ᾪ → ὢι
    case 0x1FAB: expansion[0] = 0x1F63, expansion[1] = 0x03B9, *expansion_count = 2; return 0x1F63; // ᾫ → ὣι
    case 0x1FAC: expansion[0] = 0x1F64, expansion[1] = 0x03B9, *expansion_count = 2; return 0x1F64; // ᾬ → ὤι
    case 0x1FAD: expansion[0] = 0x1F65, expansion[1] = 0x03B9, *expansion_count = 2; return 0x1F65; // ᾭ → ὥι
    case 0x1FAE: expansion[0] = 0x1F66, expansion[1] = 0x03B9, *expansion_count = 2; return 0x1F66; // ᾮ → ὦι
    case 0x1FAF: expansion[0] = 0x1F67, expansion[1] = 0x03B9, *expansion_count = 2; return 0x1F67; // ᾯ → ὧι
    case 0x1FB2: expansion[0] = 0x1F70, expansion[1] = 0x03B9, *expansion_count = 2; return 0x1F70; // ᾲ → ὰι
    case 0x1FB3: expansion[0] = 0x03B1, expansion[1] = 0x03B9, *expansion_count = 2; return 0x03B1; // ᾳ → αι
    case 0x1FB4: expansion[0] = 0x03AC, expansion[1] = 0x03B9, *expansion_count = 2; return 0x03AC; // ᾴ → άι
    case 0x1FB6: expansion[0] = 0x03B1, expansion[1] = 0x0342, *expansion_count = 2; return 0x03B1; // ᾶ → α + combining
    case 0x1FB7:
        expansion[0] = 0x03B1, expansion[1] = 0x0342, expansion[2] = 0x03B9, *expansion_count = 3;
        return 0x03B1; // ᾷ → α + 2 combining
    case 0x1FBC: expansion[0] = 0x03B1, expansion[1] = 0x03B9, *expansion_count = 2; return 0x03B1; // ᾼ → αι
    case 0x1FC2: expansion[0] = 0x1F74, expansion[1] = 0x03B9, *expansion_count = 2; return 0x1F74; // ῂ → ὴι
    case 0x1FC3: expansion[0] = 0x03B7, expansion[1] = 0x03B9, *expansion_count = 2; return 0x03B7; // ῃ → ηι
    case 0x1FC4: expansion[0] = 0x03AE, expansion[1] = 0x03B9, *expansion_count = 2; return 0x03AE; // ῄ → ήι
    case 0x1FC6: expansion[0] = 0x03B7, expansion[1] = 0x0342, *expansion_count = 2; return 0x03B7; // ῆ → η + combining
    case 0x1FC7:
        expansion[0] = 0x03B7, expansion[1] = 0x0342, expansion[2] = 0x03B9, *expansion_count = 3;
        return 0x03B7; // ῇ → η + 2 combining
    case 0x1FCC: expansion[0] = 0x03B7, expansion[1] = 0x03B9, *expansion_count = 2; return 0x03B7; // ῌ → ηι
    case 0x1FD2:
        expansion[0] = 0x03B9, expansion[1] = 0x0308, expansion[2] = 0x0300, *expansion_count = 3;
        return 0x03B9; // ῒ → ι + 2 combining
    case 0x1FD3:
        expansion[0] = 0x03B9, expansion[1] = 0x0308, expansion[2] = 0x0301, *expansion_count = 3;
        return 0x03B9; // ΐ → ι + 2 combining
    case 0x1FD6: expansion[0] = 0x03B9, expansion[1] = 0x0342, *expansion_count = 2; return 0x03B9; // ῖ → ι + combining
    case 0x1FD7:
        expansion[0] = 0x03B9, expansion[1] = 0x0308, expansion[2] = 0x0342, *expansion_count = 3;
        return 0x03B9; // ῗ → ι + 2 combining
    case 0x1FE2:
        expansion[0] = 0x03C5, expansion[1] = 0x0308, expansion[2] = 0x0300, *expansion_count = 3;
        return 0x03C5; // ῢ → υ + 2 combining
    case 0x1FE3:
        expansion[0] = 0x03C5, expansion[1] = 0x0308, expansion[2] = 0x0301, *expansion_count = 3;
        return 0x03C5; // ΰ → υ + 2 combining
    case 0x1FE4: expansion[0] = 0x03C1, expansion[1] = 0x0313, *expansion_count = 2; return 0x03C1; // ῤ → ρ + combining
    case 0x1FE6: expansion[0] = 0x03C5, expansion[1] = 0x0342, *expansion_count = 2; return 0x03C5; // ῦ → υ + combining
    case 0x1FE7:
        expansion[0] = 0x03C5, expansion[1] = 0x0308, expansion[2] = 0x0342, *expansion_count = 3;
        return 0x03C5; // ῧ → υ + 2 combining
    case 0x1FF2: expansion[0] = 0x1F7C, expansion[1] = 0x03B9, *expansion_count = 2; return 0x1F7C; // ῲ → ὼι
    case 0x1FF3: expansion[0] = 0x03C9, expansion[1] = 0x03B9, *expansion_count = 2; return 0x03C9; // ῳ → ωι
    case 0x1FF4: expansion[0] = 0x03CE, expansion[1] = 0x03B9, *expansion_count = 2; return 0x03CE; // ῴ → ώι
    case 0x1FF6: expansion[0] = 0x03C9, expansion[1] = 0x0342, *expansion_count = 2; return 0x03C9; // ῶ → ω + combining
    case 0x1FF7:
        expansion[0] = 0x03C9, expansion[1] = 0x0342, expansion[2] = 0x03B9, *expansion_count = 3;
        return 0x03C9; // ῷ → ω + 2 combining
    case 0x1FFC: expansion[0] = 0x03C9, expansion[1] = 0x03B9, *expansion_count = 2; return 0x03C9; // ῼ → ωι
    case 0xFB00: expansion[0] = 0x0066, expansion[1] = 0x0066, *expansion_count = 2; return 0x0066; // ﬀ → ff
    case 0xFB01: expansion[0] = 0x0066, expansion[1] = 0x0069, *expansion_count = 2; return 0x0066; // ﬁ → fi
    case 0xFB02: expansion[0] = 0x0066, expansion[1] = 0x006C, *expansion_count = 2; return 0x0066; // ﬂ → fl
    case 0xFB03:
        expansion[0] = 0x0066, expansion[1] = 0x0066, expansion[2] = 0x0069, *expansion_count = 3;
        return 0x0066; // ﬃ → ffi
    case 0xFB04:
        expansion[0] = 0x0066, expansion[1] = 0x0066, expansion[2] = 0x006C, *expansion_count = 3;
        return 0x0066;                                                                              // ﬄ → ffl
    case 0xFB05: expansion[0] = 0x0073, expansion[1] = 0x0074, *expansion_count = 2; return 0x0073; // ﬅ → st
    case 0xFB06: expansion[0] = 0x0073, expansion[1] = 0x0074, *expansion_count = 2; return 0x0073; // ﬆ → st
    case 0xFB13: expansion[0] = 0x0574, expansion[1] = 0x0576, *expansion_count = 2; return 0x0574; // ﬓ → մն
    case 0xFB14: expansion[0] = 0x0574, expansion[1] = 0x0565, *expansion_count = 2; return 0x0574; // ﬔ → մե
    case 0xFB15: expansion[0] = 0x0574, expansion[1] = 0x056B, *expansion_count = 2; return 0x0574; // ﬕ → մի
    case 0xFB16: expansion[0] = 0x057E, expansion[1] = 0x0576, *expansion_count = 2; return 0x057E; // ﬖ → վն
    case 0xFB17: expansion[0] = 0x0574, expansion[1] = 0x056D, *expansion_count = 2; return 0x0574; // ﬗ → մխ
    }

    return cp; // No folding
}

/**
 *  @brief  Case-insensitive UTF-8 substring search (serial implementation).
 *
 *  This function performs a case-insensitive search by case-folding both the needle and haystack
 *  according to Unicode case-folding rules. It handles one-to-many expansions (e.g., ß → ss).
 *
 *  @param haystack         The string to search in.
 *  @param haystack_length  The length of the haystack in bytes.
 *  @param needle           The substring to search for.
 *  @param needle_length    The length of the needle in bytes.
 *  @param matched_length   Output: the length of the matched substring in the haystack (in bytes).
 *  @return                 Pointer to the first match, or SZ_NULL_CHAR if not found.
 */
SZ_PUBLIC sz_cptr_t sz_utf8_find_case_insensitive_serial( //
    sz_cptr_t haystack, sz_size_t haystack_length,        //
    sz_cptr_t needle, sz_size_t needle_length, sz_size_t *matched_length) {

    if (needle_length == 0) {
        *matched_length = 0;
        return haystack;
    }

    // Pre-fold the needle into a buffer of codepoints
    sz_rune_t folded_needle[1024]; // Should be enough for most needles
    sz_size_t folded_needle_count = 0;

    sz_cptr_t needle_ptr = needle;
    sz_cptr_t needle_end = needle + needle_length;

    while (needle_ptr < needle_end && folded_needle_count < 1024) {
        sz_rune_t cp;
        sz_rune_length_t rune_length;
        sz_rune_parse(needle_ptr, &cp, &rune_length);
        if (rune_length == 0) break; // Invalid UTF-8

        sz_rune_t expansion[4];
        sz_size_t expansion_count = 0;
        sz_rune_t folded = sz_unicode_fold_codepoint_(cp, expansion, &expansion_count);

        // If expansion_count > 0, the expansion array contains ALL folded codepoints
        // If expansion_count == 0, use the return value (simple one-to-one mapping)
        if (expansion_count > 0) {
            for (sz_size_t i = 0; i < expansion_count; i++) {
                if (folded_needle_count < 1024) { folded_needle[folded_needle_count++] = expansion[i]; }
            }
        }
        else { folded_needle[folded_needle_count++] = folded; }

        needle_ptr += rune_length;
    }

    if (folded_needle_count == 0) {
        *matched_length = 0;
        return SZ_NULL_CHAR;
    }

    // Search through the haystack
    sz_cptr_t haystack_ptr = haystack;
    sz_cptr_t haystack_end = haystack + haystack_length;

    while (haystack_ptr < haystack_end) {
        sz_cptr_t match_start = haystack_ptr;
        sz_cptr_t match_ptr = haystack_ptr;
        sz_size_t needle_idx = 0;

        // Try to match the folded needle at this position
        while (needle_idx < folded_needle_count && match_ptr < haystack_end) {
            sz_rune_t cp;
            sz_rune_length_t rune_length;
            sz_rune_parse(match_ptr, &cp, &rune_length);
            if (rune_length == 0) break; // Invalid UTF-8

            sz_rune_t expansion[4];
            sz_size_t expansion_count = 0;
            sz_rune_t folded = sz_unicode_fold_codepoint_(cp, expansion, &expansion_count);

            // If expansion_count > 0, check expansion array; otherwise check return value
            if (expansion_count > 0) {
                // Check all codepoints in the expansion array
                for (sz_size_t i = 0; i < expansion_count && needle_idx < folded_needle_count; i++) {
                    if (expansion[i] != folded_needle[needle_idx]) {
                        needle_idx = folded_needle_count + 1; // Signal mismatch
                        break;
                    }
                    needle_idx++;
                }
                if (needle_idx > folded_needle_count) break; // Mismatch occurred
            }
            else {
                // Simple one-to-one mapping
                if (folded != folded_needle[needle_idx]) {
                    break; // Mismatch
                }
                needle_idx++;
            }

            match_ptr += rune_length;
        }

        // Did we match the entire needle?
        if (needle_idx == folded_needle_count) {
            *matched_length = (sz_size_t)(match_ptr - match_start);
            return match_start;
        }

        // Move to next codepoint in haystack
        sz_rune_t cp;
        sz_rune_length_t rune_length;
        sz_rune_parse(haystack_ptr, &cp, &rune_length);
        if (rune_length == 0) break; // Invalid UTF-8
        haystack_ptr += rune_length;
    }

    *matched_length = 0;
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

    // Process up to the minimum of: available bytes, output capacity * 4, or optimal chunk size (64)
    sz_size_t chunk_size = sz_min_of_three(length, runes_capacity * 4, 64);
    sz_u512_vec_t text_vec, runes_vec;
    __mmask64 load_mask = sz_u64_mask_until_(chunk_size);
    text_vec.zmm = _mm512_maskz_loadu_epi8(load_mask, (sz_u8_t const *)text);

    // Check, how many of the next characters are single byte (ASCII) codepoints
    // ASCII bytes have bit 7 clear (0x00-0x7F), non-ASCII have bit 7 set (0x80-0xFF)
    __mmask64 non_ascii_mask = _mm512_movepi8_mask(text_vec.zmm);
    // Find first non-ASCII byte or end of loaded data
    sz_size_t ascii_prefix_length = sz_u64_ctz(non_ascii_mask | ~load_mask);

    if (ascii_prefix_length) {
        // Unpack the last 16 bytes of text into the next 16 runes.
        // Even if we have more than 16 ASCII characters, we don't want to overcomplicate control flow here.
        sz_size_t runes_to_place = sz_min_of_three(ascii_prefix_length, 16, runes_capacity);
        runes_vec.zmm = _mm512_cvtepu8_epi32(_mm512_castsi512_si128(text_vec.zmm));
        _mm512_mask_storeu_epi32(runes, sz_u16_mask_until_(runes_to_place), runes_vec.zmm);
        *runes_unpacked = runes_to_place;
        return text + runes_to_place;
    }

    // Check for the number of 2-byte characters
    // 2-byte UTF-8: [lead, cont] where lead=110xxxxx (0xC0-0xDF), cont=10xxxxxx (0x80-0xBF)
    // In 16-bit little-endian: 0xCCLL where LL=lead, CC=cont
    // Mask: 0xC0E0 (cont & 0xC0, lead & 0xE0), Pattern: 0x80C0 (cont=0x80, lead=0xC0)
    __mmask32 non_two_byte_mask = _mm512_cmpneq_epi16_mask(
        _mm512_and_si512(text_vec.zmm, _mm512_set1_epi16((short)0xC0E0)), _mm512_set1_epi16((short)0x80C0));
    sz_size_t two_byte_prefix_length = sz_u64_ctz(non_two_byte_mask);
    if (two_byte_prefix_length) {
        // Unpack the last 32 bytes of text into the next 32 runes.
        // Even if we have more than 32 two-byte characters, we don't want to overcomplicate control flow here.
        sz_size_t runes_to_place = sz_min_of_three(two_byte_prefix_length, 32, runes_capacity);
        runes_vec.zmm = _mm512_cvtepu16_epi32(_mm512_castsi512_si256(text_vec.zmm));
        // Decode 2-byte UTF-8: ((lead & 0x1F) << 6) | (cont & 0x3F)
        // After cvtepu16_epi32: value = 0x0000CCLL where LL=lead (bits 7-0), CC=cont (bits 15-8)
        runes_vec.zmm = _mm512_or_si512(                                                      //
            _mm512_slli_epi32(_mm512_and_si512(runes_vec.zmm, _mm512_set1_epi32(0x1FU)), 6),  // (lead & 0x1F) << 6
            _mm512_and_si512(_mm512_srli_epi32(runes_vec.zmm, 8), _mm512_set1_epi32(0x3FU))); // (cont & 0x3F)
        _mm512_mask_storeu_epi32(runes, sz_u32_mask_until_(runes_to_place), runes_vec.zmm);
        *runes_unpacked = runes_to_place;
        return text + runes_to_place * 2;
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

    // Seems like broken unicoode?
    *runes_unpacked = 0;
    return text;
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

SZ_PUBLIC sz_cptr_t sz_utf8_unpack_chunk_haswell( //
    sz_cptr_t text, sz_size_t length,             //
    sz_rune_t *runes, sz_size_t runes_capacity,   //
    sz_size_t *runes_unpacked) {
    // Fallback to serial implementation for now
    // A future optimization could use AVX2 for decoding
    return sz_utf8_unpack_chunk_serial(text, length, runes, runes_capacity, runes_unpacked);
}

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#endif            // SZ_USE_HASWELL
#pragma endregion // Haswell Implementation

#pragma region NEON Implementation

SZ_PUBLIC sz_cptr_t sz_utf8_find_newline_neon(sz_cptr_t text, sz_size_t length, sz_size_t *matched_length) {
    return sz_utf8_find_newline_serial(text, length, matched_length);
}

SZ_PUBLIC sz_cptr_t sz_utf8_find_whitespace_neon(sz_cptr_t text, sz_size_t length, sz_size_t *matched_length) {
    return sz_utf8_find_whitespace_serial(text, length, matched_length);
}

SZ_PUBLIC sz_size_t sz_utf8_count_neon(sz_cptr_t text, sz_size_t length) { return sz_utf8_count_serial(text, length); }

SZ_PUBLIC sz_cptr_t sz_utf8_find_nth_neon(sz_cptr_t text, sz_size_t length, sz_size_t n) {
    return sz_utf8_find_nth_serial(text, length, n);
}

SZ_PUBLIC sz_cptr_t sz_utf8_unpack_chunk_neon(  //
    sz_cptr_t text, sz_size_t length,           //
    sz_rune_t *runes, sz_size_t runes_capacity, //
    sz_size_t *runes_unpacked) {
    return sz_utf8_unpack_chunk_serial(text, length, runes, runes_capacity, runes_unpacked);
}

#pragma endregion // NEON Implementation

#pragma region Dynamic Dispatch

#if !SZ_DYNAMIC_DISPATCH

SZ_DYNAMIC sz_size_t sz_utf8_count(sz_cptr_t text, sz_size_t length) {
#if SZ_USE_ICE
    return sz_utf8_count_ice(text, length);
#elif SZ_USE_HASWELL
    return sz_utf8_count_haswell(text, length);
#else
    return sz_utf8_count_serial(text, length);
#endif
}

SZ_DYNAMIC sz_cptr_t sz_utf8_find_nth(sz_cptr_t text, sz_size_t length, sz_size_t n) {
#if SZ_USE_ICE
    return sz_utf8_find_nth_ice(text, length, n);
#elif SZ_USE_HASWELL
    return sz_utf8_find_nth_haswell(text, length, n);
#else
    return sz_utf8_find_nth_serial(text, length, n);
#endif
}

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

SZ_DYNAMIC sz_cptr_t sz_utf8_unpack_chunk(sz_cptr_t text, sz_size_t length, sz_rune_t *runes, sz_size_t runes_capacity,
                                          sz_size_t *runes_unpacked) {
#if SZ_USE_ICE
    return sz_utf8_unpack_chunk_ice(text, length, runes, runes_capacity, runes_unpacked);
#elif SZ_USE_HASWELL
    return sz_utf8_unpack_chunk_haswell(text, length, runes, runes_capacity, runes_unpacked);
#else
    return sz_utf8_unpack_chunk_serial(text, length, runes, runes_capacity, runes_unpacked);
#endif
}

#endif // !SZ_DYNAMIC_DISPATCH

#pragma endregion // Dynamic Dispatch

#ifdef __cplusplus
}
#endif

#endif // STRINGZILLA_UTF8_H_
