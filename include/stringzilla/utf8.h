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
 *  Work in progress:
 *
 *  - `sz_utf8_case_fold` - Unicode case folding for codepoints
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
 *  @brief  Apply Unicode case folding to a UTF-8 string in-place or to a separate buffer.
 *
 *  This function reads a UTF-8 encoded source string, applies Unicode case folding to each
 *  codepoint, and writes the result to the destination buffer. Case folding normalizes text
 *  for case-insensitive comparisons by mapping uppercase letters to their lowercase equivalents
 *  and handling special cases like German ß → ss expansion.
 *
 *  The destination buffer must be at least 1.5x the source length to accommodate potential
 *  one-to-many expansions (e.g., ß → ss, ﬁ → fi). If the destination buffer is too small,
 *  the function returns an error status.
 *
 *  @param[in] source UTF-8 string to be case-folded.
 *  @param[in] source_length Number of bytes in the source buffer.
 *  @param[out] destination Buffer to write the case-folded UTF-8 string.
 *  @param[in] destination_capacity Size of the destination buffer in bytes.
 *  @param[out] destination_length Number of bytes written to the destination buffer.
 *  @return sz_success_k on success, sz_bad_alloc_k if destination buffer is too small,
 *          sz_invalid_utf8_k if source contains invalid UTF-8 sequences.
 *
 *  @example Basic usage:
 *  @code
 *      char const *source = "HELLO WORLD";
 *      char destination[32];
 *      sz_size_t result_length;
 *      sz_status_t status = sz_utf8_case_fold(source, 11, destination, 32, &result_length);
 *      // destination now contains "hello world", result_length = 11
 *  @endcode
 *
 *  @example Handling expansions:
 *  @code
 *      char const *source = "STRAẞE";  // German "street" with capital ß
 *      char destination[32];
 *      sz_size_t result_length;
 *      sz_status_t status = sz_utf8_case_fold(source, strlen(source), destination, 32, &result_length);
 *      // destination now contains "strasse" (ẞ expanded to ss)
 *  @endcode
 */
SZ_DYNAMIC sz_status_t sz_utf8_case_fold(      //
    sz_cptr_t source, sz_size_t source_length, //
    sz_ptr_t destination, sz_size_t destination_capacity, sz_size_t *destination_length);

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
SZ_PUBLIC sz_status_t sz_utf8_case_fold_serial( //
    sz_cptr_t source, sz_size_t source_length,  //
    sz_ptr_t destination, sz_size_t destination_capacity, sz_size_t *destination_length);
SZ_PUBLIC sz_cptr_t sz_utf8_find_case_insensitive_serial( //
    sz_cptr_t haystack, sz_size_t haystack_length,        //
    sz_cptr_t needle, sz_size_t needle_length, sz_size_t *matched_length);

// Haswell (AVX2) implementations
SZ_PUBLIC sz_size_t sz_utf8_count_haswell(sz_cptr_t text, sz_size_t length);
SZ_PUBLIC sz_cptr_t sz_utf8_find_nth_haswell(sz_cptr_t text, sz_size_t length, sz_size_t n);
SZ_PUBLIC sz_cptr_t sz_utf8_find_newline_haswell(sz_cptr_t text, sz_size_t length, sz_size_t *matched_length);
SZ_PUBLIC sz_cptr_t sz_utf8_find_whitespace_haswell(sz_cptr_t text, sz_size_t length, sz_size_t *matched_length);
SZ_PUBLIC sz_cptr_t sz_utf8_unpack_chunk_haswell( //
    sz_cptr_t text, sz_size_t length,             //
    sz_rune_t *runes, sz_size_t runes_capacity,   //
    sz_size_t *runes_unpacked);
SZ_PUBLIC sz_status_t sz_utf8_case_fold_haswell( //
    sz_cptr_t source, sz_size_t source_length,   //
    sz_ptr_t destination, sz_size_t destination_capacity, sz_size_t *destination_length);
SZ_PUBLIC sz_cptr_t sz_utf8_find_case_insensitive_haswell( //
    sz_cptr_t haystack, sz_size_t haystack_length,         //
    sz_cptr_t needle, sz_size_t needle_length, sz_size_t *matched_length);

// Ice Lake (AVX-512) implementations
SZ_PUBLIC sz_size_t sz_utf8_count_ice(sz_cptr_t text, sz_size_t length);
SZ_PUBLIC sz_cptr_t sz_utf8_find_nth_ice(sz_cptr_t text, sz_size_t length, sz_size_t n);
SZ_PUBLIC sz_cptr_t sz_utf8_find_newline_ice(sz_cptr_t text, sz_size_t length, sz_size_t *matched_length);
SZ_PUBLIC sz_cptr_t sz_utf8_find_whitespace_ice(sz_cptr_t text, sz_size_t length, sz_size_t *matched_length);
SZ_PUBLIC sz_cptr_t sz_utf8_unpack_chunk_ice(   //
    sz_cptr_t text, sz_size_t length,           //
    sz_rune_t *runes, sz_size_t runes_capacity, //
    sz_size_t *runes_unpacked);
SZ_PUBLIC sz_status_t sz_utf8_case_fold_ice(   //
    sz_cptr_t source, sz_size_t source_length, //
    sz_ptr_t destination, sz_size_t destination_capacity, sz_size_t *destination_length);
SZ_PUBLIC sz_cptr_t sz_utf8_find_case_insensitive_ice( //
    sz_cptr_t haystack, sz_size_t haystack_length,     //
    sz_cptr_t needle, sz_size_t needle_length, sz_size_t *matched_length);

// NEON (ARM) implementations - fall back to serial
SZ_PUBLIC sz_size_t sz_utf8_count_neon(sz_cptr_t text, sz_size_t length);
SZ_PUBLIC sz_cptr_t sz_utf8_find_nth_neon(sz_cptr_t text, sz_size_t length, sz_size_t n);
SZ_PUBLIC sz_cptr_t sz_utf8_find_newline_neon(sz_cptr_t text, sz_size_t length, sz_size_t *matched_length);
SZ_PUBLIC sz_cptr_t sz_utf8_find_whitespace_neon(sz_cptr_t text, sz_size_t length, sz_size_t *matched_length);
SZ_PUBLIC sz_cptr_t sz_utf8_unpack_chunk_neon(  //
    sz_cptr_t text, sz_size_t length,           //
    sz_rune_t *runes, sz_size_t runes_capacity, //
    sz_size_t *runes_unpacked);
SZ_PUBLIC sz_status_t sz_utf8_case_fold_neon(  //
    sz_cptr_t source, sz_size_t source_length, //
    sz_ptr_t destination, sz_size_t destination_capacity, sz_size_t *destination_length);
SZ_PUBLIC sz_cptr_t sz_utf8_find_case_insensitive_neon( //
    sz_cptr_t haystack, sz_size_t haystack_length,      //
    sz_cptr_t needle, sz_size_t needle_length, sz_size_t *matched_length);

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

/**
 *  @brief  Encode a UTF-32 codepoint to UTF-8.
 *
 *  Writes 1-4 bytes to the output buffer depending on the codepoint value.
 *  This is the inverse of @c sz_rune_parse.
 *
 *  @param[in] cp The UTF-32 codepoint to encode.
 *  @param[out] out Output buffer (must have space for at least 4 bytes).
 *  @return Number of bytes written (1-4), or 0 if the codepoint is invalid.
 */
SZ_INTERNAL sz_size_t sz_rune_export_(sz_rune_t cp, sz_u8_t *out) {
    if (cp <= 0x7F) {
        out[0] = (sz_u8_t)cp;
        return 1;
    }
    else if (cp <= 0x7FF) {
        out[0] = (sz_u8_t)(0xC0 | (cp >> 6));
        out[1] = (sz_u8_t)(0x80 | (cp & 0x3F));
        return 2;
    }
    else if (cp <= 0xFFFF) {
        // Reject surrogate codepoints
        if (cp >= 0xD800 && cp <= 0xDFFF) return 0;
        out[0] = (sz_u8_t)(0xE0 | (cp >> 12));
        out[1] = (sz_u8_t)(0x80 | ((cp >> 6) & 0x3F));
        out[2] = (sz_u8_t)(0x80 | (cp & 0x3F));
        return 3;
    }
    else if (cp <= 0x10FFFF) {
        out[0] = (sz_u8_t)(0xF0 | (cp >> 18));
        out[1] = (sz_u8_t)(0x80 | ((cp >> 12) & 0x3F));
        out[2] = (sz_u8_t)(0x80 | ((cp >> 6) & 0x3F));
        out[3] = (sz_u8_t)(0x80 | (cp & 0x3F));
        return 4;
    }
    return 0; // Invalid codepoint
}

// clang-format off
SZ_INTERNAL sz_size_t sz_unicode_fold_codepoint_(sz_rune_t rune, sz_rune_t *folded) {
    // Offset +32 ranges
    if (rune >= 0x0041 && rune <= 0x005A) { folded[0] = rune + 0x20; return 1; } // ASCII A-Z → a-z
    if (rune >= 0x00C0 && rune <= 0x00D6) { folded[0] = rune + 0x20; return 1; } // Latin-1 À-Ö → à-ö
    if (rune >= 0x00D8 && rune <= 0x00DE) { folded[0] = rune + 0x20; return 1; } // Latin-1 Ø-Þ → ø-þ
    if (rune >= 0x0391 && rune <= 0x03A1) { folded[0] = rune + 0x20; return 1; } // Greek Α-Ρ → α-ρ
    if (rune >= 0x03A3 && rune <= 0x03AB) { folded[0] = rune + 0x20; return 1; } // Greek Σ-Ϋ → σ-ϋ
    if (rune >= 0x0410 && rune <= 0x042F) { folded[0] = rune + 0x20; return 1; } // Cyrillic А-Я → а-я
    if (rune >= 0xFF21 && rune <= 0xFF3A) { folded[0] = rune + 0x20; return 1; } // Fullwidth Ａ-Ｚ → ａ-ｚ
    if (rune >= 0x10D50 && rune <= 0x10D65) { folded[0] = rune + 0x20; return 1; } // Garay
    if (rune >= 0x118A0 && rune <= 0x118BF) { folded[0] = rune + 0x20; return 1; } // Warang Citi
    if (rune >= 0x16E40 && rune <= 0x16E5F) { folded[0] = rune + 0x20; return 1; } // Medefaidrin
    // Offset +48 ranges
    if (rune >= 0x0531 && rune <= 0x0556) { folded[0] = rune + 0x30; return 1; } // Armenian Ա-Ֆ → ա-ֆ
    if (rune >= 0x2C00 && rune <= 0x2C2F) { folded[0] = rune + 0x30; return 1; } // Glagolitic Ⰰ-Ⱟ → ⰰ-ⱟ
    // Other offset ranges
    if (rune >= 0xAB70 && rune <= 0xABBF) { folded[0] = rune + 0xFFFF6830; return 1; } // Cherokee Ꭰ-Ᏼ (offset -38864)
    if (rune >= 0x10400 && rune <= 0x10427) { folded[0] = rune + 0x28; return 1; } // Deseret 𐐀-𐐧 → 𐐨-𐑏 (+40)
    if (rune >= 0x104B0 && rune <= 0x104D3) { folded[0] = rune + 0x28; return 1; } // Osage 𐒰-𐓓 → 𐓘-𐓻 (+40)
    if (rune >= 0x13F8 && rune <= 0x13FD) { folded[0] = rune + 0xFFFFFFF8; return 1; } // Cherokee Ᏸ-Ᏽ (-8)
    if (rune >= 0x1F08 && rune <= 0x1F0F) { folded[0] = rune + 0xFFFFFFF8; return 1; } // Greek Ἀ-Ἇ (-8)
    if (rune >= 0x1F18 && rune <= 0x1F1D) { folded[0] = rune + 0xFFFFFFF8; return 1; } // Greek Ἐ-Ἕ (-8)
    if (rune >= 0x1F28 && rune <= 0x1F2F) { folded[0] = rune + 0xFFFFFFF8; return 1; } // Greek Ἠ-Ἧ (-8)
    if (rune >= 0x1F38 && rune <= 0x1F3F) { folded[0] = rune + 0xFFFFFFF8; return 1; } // Greek Ἰ-Ἷ (-8)
    if (rune >= 0x1F48 && rune <= 0x1F4D) { folded[0] = rune + 0xFFFFFFF8; return 1; } // Greek Ὀ-Ὅ (-8)
    if (rune >= 0x1F68 && rune <= 0x1F6F) { folded[0] = rune + 0xFFFFFFF8; return 1; } // Greek Ὠ-Ὧ (-8)
    if (rune >= 0x10C80 && rune <= 0x10CB2) { folded[0] = rune + 0x40; return 1; } // Old Hungarian (+64)
    if (rune >= 0x1C90 && rune <= 0x1CBA) { folded[0] = rune + 0xFFFFF440; return 1; } // Georgian Mtavruli Ა-Ჺ (-3008)
    if (rune >= 0x1CBD && rune <= 0x1CBF) { folded[0] = rune + 0xFFFFF440; return 1; } // Georgian Mtavruli Ჽ-Ჿ (-3008)
    if (rune >= 0x10A0 && rune <= 0x10C5) { folded[0] = rune + 0x1C60; return 1; } // Georgian Ⴀ-Ⴥ (+7264)
    if (rune >= 0x10570 && rune <= 0x1057A) { folded[0] = rune + 0x27; return 1; } // Vithkuqi (+39)
    if (rune >= 0x1057C && rune <= 0x1058A) { folded[0] = rune + 0x27; return 1; } // Vithkuqi (+39)
    if (rune >= 0x1058C && rune <= 0x10592) { folded[0] = rune + 0x27; return 1; } // Vithkuqi (+39)
    if (rune >= 0x1E900 && rune <= 0x1E921) { folded[0] = rune + 0x22; return 1; } // Adlam 𞤀-𞤡 → 𞤢-𞥃 (+34)
    if (rune >= 0x24B6 && rune <= 0x24CF) { folded[0] = rune + 0x1A; return 1; } // Circled Ⓐ-Ⓩ → ⓐ-ⓩ (+26)
    if (rune >= 0x16EA0 && rune <= 0x16EB8) { folded[0] = rune + 0x1B; return 1; } // Kawi (+27)
    if (rune >= 0x2160 && rune <= 0x216F) { folded[0] = rune + 0x10; return 1; } // Roman numerals Ⅰ-Ⅿ → ⅰ-ⅿ (+16)
    if (rune >= 0x0400 && rune <= 0x040F) { folded[0] = rune + 0x50; return 1; } // Cyrillic Ѐ-Џ → ѐ-џ (+80)
    if (rune >= 0x03FD && rune <= 0x03FF) { folded[0] = rune + 0xFFFFFF7E; return 1; } // Greek Ͻ-Ͽ (-130)
    if (rune >= 0x1FC8 && rune <= 0x1FCB) { folded[0] = rune + 0xFFFFFFAA; return 1; } // Greek Ὲ-Ή (-86)
    if (rune >= 0x0388 && rune <= 0x038A) { folded[0] = rune + 0x25; return 1; } // Greek Έ-Ί (+37)

    // Even/odd +1 mappings: uppercase at even codepoint, lowercase at odd (or vice versa)
    sz_u32_t is_even = ((rune & 1) == 0), is_odd = !is_even;
    // Latin Extended-A: Ą Ć Ę Ł Ń Ś Ź Ż, Č Ď Ě Ň Ř Š Ť Ž, Ő Ű, Ş Ğ
    if (rune >= 0x0100 && rune <= 0x012E && is_even) { folded[0] = rune + 1; return 1; } // Ā-Į
    if (rune >= 0x0132 && rune <= 0x0136 && is_even) { folded[0] = rune + 1; return 1; } // Ĳ-Ķ
    if (rune >= 0x0139 && rune <= 0x0147 && is_odd)  { folded[0] = rune + 1; return 1; } // Ĺ-Ň
    if (rune >= 0x014A && rune <= 0x0176 && is_even) { folded[0] = rune + 1; return 1; } // Ŋ-Ŷ
    if (rune >= 0x0179 && rune <= 0x017D && is_odd)  { folded[0] = rune + 1; return 1; } // Ź-Ž
    // Latin Extended-B: Pinyin, Romanian, Serbian/Croatian
    if (rune >= 0x01CD && rune <= 0x01DB && is_odd)  { folded[0] = rune + 1; return 1; } // Ǎ-Ǜ
    if (rune >= 0x01DE && rune <= 0x01EE && is_even) { folded[0] = rune + 1; return 1; } // Ǟ-Ǯ
    if (rune >= 0x01F8 && rune <= 0x01FE && is_even) { folded[0] = rune + 1; return 1; } // Ǹ-Ǿ
    if (rune >= 0x0200 && rune <= 0x021E && is_even) { folded[0] = rune + 1; return 1; } // Ȁ-Ȟ
    if (rune >= 0x0222 && rune <= 0x0232 && is_even) { folded[0] = rune + 1; return 1; } // Ȣ-Ȳ
    if (rune >= 0x0246 && rune <= 0x024E && is_even) { folded[0] = rune + 1; return 1; } // Ɇ-Ɏ
    // Greek archaic
    if (rune >= 0x0370 && rune <= 0x0372 && is_even) { folded[0] = rune + 1; return 1; } // Ͱ-Ͳ
    if (rune == 0x0376) { folded[0] = 0x0377; return 1; } // Ͷ → ͷ
    if (rune >= 0x03D8 && rune <= 0x03EE && is_even) { folded[0] = rune + 1; return 1; } // Ϙ-Ϯ
    // Cyrillic extended
    if (rune >= 0x0460 && rune <= 0x0480 && is_even) { folded[0] = rune + 1; return 1; } // Ѡ-Ҁ
    if (rune >= 0x048A && rune <= 0x04BE && is_even) { folded[0] = rune + 1; return 1; } // Ҋ-Ҿ
    if (rune >= 0x04C1 && rune <= 0x04CD && is_odd)  { folded[0] = rune + 1; return 1; } // Ӂ-Ӎ
    if (rune >= 0x04D0 && rune <= 0x04FE && is_even) { folded[0] = rune + 1; return 1; } // Ӑ-Ӿ
    if (rune >= 0x0500 && rune <= 0x052E && is_even) { folded[0] = rune + 1; return 1; } // Ԁ-Ԯ
    // Latin Extended Additional: Vietnamese Ạ Ả Ấ Ầ...
    if (rune >= 0x1E00 && rune <= 0x1E94 && is_even) { folded[0] = rune + 1; return 1; } // Ḁ-Ẕ
    if (rune >= 0x1EA0 && rune <= 0x1EFE && is_even) { folded[0] = rune + 1; return 1; } // Ạ-Ỿ
    // Coptic
    if (rune >= 0x2C80 && rune <= 0x2CE2 && is_even) { folded[0] = rune + 1; return 1; } // Ⲁ-Ⳣ
    // Cyrillic Extended-B
    if (rune >= 0xA640 && rune <= 0xA66C && is_even) { folded[0] = rune + 1; return 1; } // Ꙁ-Ꙭ
    if (rune >= 0xA680 && rune <= 0xA69A && is_even) { folded[0] = rune + 1; return 1; } // Ꚁ-Ꚛ
    // Latin Extended-D
    if (rune >= 0xA722 && rune <= 0xA72E && is_even) { folded[0] = rune + 1; return 1; } // Ꜣ-Ꜯ
    if (rune >= 0xA732 && rune <= 0xA76E && is_even) { folded[0] = rune + 1; return 1; } // Ꜳ-Ꝯ
    if (rune >= 0xA77E && rune <= 0xA786 && is_even) { folded[0] = rune + 1; return 1; } // Ꝿ-Ꞇ
    if (rune >= 0xA790 && rune <= 0xA792 && is_even) { folded[0] = rune + 1; return 1; } // Ꞑ-Ꞓ
    if (rune >= 0xA796 && rune <= 0xA7A8 && is_even) { folded[0] = rune + 1; return 1; } // Ꞗ-Ꞩ
    if (rune >= 0xA7B4 && rune <= 0xA7C2 && is_even) { folded[0] = rune + 1; return 1; } // Ꞵ-Ꟃ
    if (rune == 0xA7C7 || rune == 0xA7C9) { folded[0] = rune + 1; return 1; } // Ꟈ, Ꟊ
    if (rune >= 0xA7CC && rune <= 0xA7D8 && is_even) { folded[0] = rune + 1; return 1; } // Ꟍ-Ꟙ
    if (rune == 0xA7DA) { folded[0] = 0xA7DB; return 1; } // Ꟛ → ꟛ
    if (rune == 0xA7F5) { folded[0] = 0xA7F6; return 1; } // Ꟶ → ꟶ

    // Irregular one-to-one mappings: ~90 cases that don't follow even/odd patterns
    switch (rune) {
    // Latin-1 Supplement & specials
    case 0x00B5: folded[0] = 0x03BC; return 1; // µ → μ (micro sign to Greek mu)
    case 0x0178: folded[0] = 0x00FF; return 1; // Ÿ → ÿ
    case 0x017F: folded[0] = 0x0073; return 1; // ſ → s (long s)
    // Latin Extended-B: African/IPA letters with irregular mappings (0x0181-0x01BF)
    case 0x0181: folded[0] = 0x0253; return 1; // Ɓ → ɓ
    case 0x0182: folded[0] = 0x0183; return 1; // Ƃ → ƃ
    case 0x0184: folded[0] = 0x0185; return 1; // Ƅ → ƅ
    case 0x0186: folded[0] = 0x0254; return 1; // Ɔ → ɔ
    case 0x0187: folded[0] = 0x0188; return 1; // Ƈ → ƈ
    case 0x0189: folded[0] = 0x0256; return 1; // Ɖ → ɖ
    case 0x018A: folded[0] = 0x0257; return 1; // Ɗ → ɗ
    case 0x018B: folded[0] = 0x018C; return 1; // Ƌ → ƌ
    case 0x018E: folded[0] = 0x01DD; return 1; // Ǝ → ǝ
    case 0x018F: folded[0] = 0x0259; return 1; // Ə → ə (schwa, Azerbaijani)
    case 0x0190: folded[0] = 0x025B; return 1; // Ɛ → ɛ
    case 0x0191: folded[0] = 0x0192; return 1; // Ƒ → ƒ
    case 0x0193: folded[0] = 0x0260; return 1; // Ɠ → ɠ
    case 0x0194: folded[0] = 0x0263; return 1; // Ɣ → ɣ
    case 0x0196: folded[0] = 0x0269; return 1; // Ɩ → ɩ
    case 0x0197: folded[0] = 0x0268; return 1; // Ɨ → ɨ
    case 0x0198: folded[0] = 0x0199; return 1; // Ƙ → ƙ
    case 0x019C: folded[0] = 0x026F; return 1; // Ɯ → ɯ
    case 0x019D: folded[0] = 0x0272; return 1; // Ɲ → ɲ
    case 0x019F: folded[0] = 0x0275; return 1; // Ɵ → ɵ
    case 0x01A0: folded[0] = 0x01A1; return 1; // Ơ → ơ (Vietnamese)
    case 0x01A2: folded[0] = 0x01A3; return 1; // Ƣ → ƣ
    case 0x01A4: folded[0] = 0x01A5; return 1; // Ƥ → ƥ
    case 0x01A6: folded[0] = 0x0280; return 1; // Ʀ → ʀ
    case 0x01A7: folded[0] = 0x01A8; return 1; // Ƨ → ƨ
    case 0x01A9: folded[0] = 0x0283; return 1; // Ʃ → ʃ
    case 0x01AC: folded[0] = 0x01AD; return 1; // Ƭ → ƭ
    case 0x01AE: folded[0] = 0x0288; return 1; // Ʈ → ʈ
    case 0x01AF: folded[0] = 0x01B0; return 1; // Ư → ư (Vietnamese)
    case 0x01B1: folded[0] = 0x028A; return 1; // Ʊ → ʊ
    case 0x01B2: folded[0] = 0x028B; return 1; // Ʋ → ʋ
    case 0x01B3: folded[0] = 0x01B4; return 1; // Ƴ → ƴ
    case 0x01B5: folded[0] = 0x01B6; return 1; // Ƶ → ƶ
    case 0x01B7: folded[0] = 0x0292; return 1; // Ʒ → ʒ
    case 0x01B8: folded[0] = 0x01B9; return 1; // Ƹ → ƹ
    case 0x01BC: folded[0] = 0x01BD; return 1; // Ƽ → ƽ

    // Digraphs: Serbian/Croatian DŽ, LJ, NJ and DZ
    case 0x01C4: folded[0] = 0x01C6; return 1; // Ǆ → ǆ
    case 0x01C5: folded[0] = 0x01C6; return 1; // ǅ → ǆ (titlecase)
    case 0x01C7: folded[0] = 0x01C9; return 1; // Ǉ → ǉ
    case 0x01C8: folded[0] = 0x01C9; return 1; // ǈ → ǉ (titlecase)
    case 0x01CA: folded[0] = 0x01CC; return 1; // Ǌ → ǌ
    case 0x01CB: folded[0] = 0x01CC; return 1; // ǋ → ǌ (titlecase)
    case 0x01F1: folded[0] = 0x01F3; return 1; // Ǳ → ǳ
    case 0x01F2: folded[0] = 0x01F3; return 1; // ǲ → ǳ (titlecase)
    // Latin Extended-B: isolated irregulars
    case 0x01F4: folded[0] = 0x01F5; return 1; // Ǵ → ǵ (between ranges)
    case 0x01F6: folded[0] = 0x0195; return 1; // Ƕ → ƕ (hwair)
    case 0x01F7: folded[0] = 0x01BF; return 1; // Ƿ → ƿ (wynn)
    case 0x0220: folded[0] = 0x019E; return 1; // Ƞ → ƞ
    case 0x023A: folded[0] = 0x2C65; return 1; // Ⱥ → ⱥ
    case 0x023B: folded[0] = 0x023C; return 1; // Ȼ → ȼ
    case 0x023D: folded[0] = 0x019A; return 1; // Ƚ → ƚ
    case 0x023E: folded[0] = 0x2C66; return 1; // Ⱦ → ⱦ
    case 0x0241: folded[0] = 0x0242; return 1; // Ɂ → ɂ
    case 0x0243: folded[0] = 0x0180; return 1; // Ƀ → ƀ
    case 0x0244: folded[0] = 0x0289; return 1; // Ʉ → ʉ
    case 0x0245: folded[0] = 0x028C; return 1; // Ʌ → ʌ

    // Greek: combining iota, accented vowels, variant forms
    case 0x0345: folded[0] = 0x03B9; return 1; // ͅ → ι (combining iota subscript)
    case 0x037F: folded[0] = 0x03F3; return 1; // Ϳ → ϳ
    case 0x0386: folded[0] = 0x03AC; return 1; // Ά → ά
    case 0x038C: folded[0] = 0x03CC; return 1; // Ό → ό
    case 0x038E: folded[0] = 0x03CD; return 1; // Ύ → ύ
    case 0x038F: folded[0] = 0x03CE; return 1; // Ώ → ώ
    case 0x03C2: folded[0] = 0x03C3; return 1; // ς → σ (final sigma)
    case 0x03CF: folded[0] = 0x03D7; return 1; // Ϗ → ϗ
    case 0x03D0: folded[0] = 0x03B2; return 1; // ϐ → β (beta symbol)
    case 0x03D1: folded[0] = 0x03B8; return 1; // ϑ → θ (theta symbol)
    case 0x03D5: folded[0] = 0x03C6; return 1; // ϕ → φ (phi symbol)
    case 0x03D6: folded[0] = 0x03C0; return 1; // ϖ → π (pi symbol)
    case 0x03F0: folded[0] = 0x03BA; return 1; // ϰ → κ (kappa symbol)
    case 0x03F1: folded[0] = 0x03C1; return 1; // ϱ → ρ (rho symbol)
    case 0x03F4: folded[0] = 0x03B8; return 1; // ϴ → θ
    case 0x03F5: folded[0] = 0x03B5; return 1; // ϵ → ε (lunate epsilon)
    case 0x03F7: folded[0] = 0x03F8; return 1; // Ϸ → ϸ
    case 0x03F9: folded[0] = 0x03F2; return 1; // Ϲ → ϲ
    case 0x03FA: folded[0] = 0x03FB; return 1; // Ϻ → ϻ
    // Cyrillic: palochka (irregular +15 offset)
    case 0x04C0: folded[0] = 0x04CF; return 1; // Ӏ → ӏ
    // Georgian: large offsets to lowercase block
    case 0x10C7: folded[0] = 0x2D27; return 1; // Ⴧ → ⴧ
    case 0x10CD: folded[0] = 0x2D2D; return 1; // Ⴭ → ⴭ
    // Cyrillic Extended-C: Old Slavonic variant forms (map to basic Cyrillic)
    case 0x1C80: folded[0] = 0x0432; return 1; // ᲀ → в
    case 0x1C81: folded[0] = 0x0434; return 1; // ᲁ → д
    case 0x1C82: folded[0] = 0x043E; return 1; // ᲂ → о
    case 0x1C83: folded[0] = 0x0441; return 1; // ᲃ → с
    case 0x1C84: folded[0] = 0x0442; return 1; // ᲄ → т
    case 0x1C85: folded[0] = 0x0442; return 1; // ᲅ → т
    case 0x1C86: folded[0] = 0x044A; return 1; // ᲆ → ъ
    case 0x1C87: folded[0] = 0x0463; return 1; // ᲇ → ѣ
    case 0x1C88: folded[0] = 0xA64B; return 1; // ᲈ → ꙋ
    case 0x1C89: folded[0] = 0x1C8A; return 1; // Ᲊ → ᲊ
    // Latin Extended Additional: long s with dot above (irregular target)
    case 0x1E9B: folded[0] = 0x1E61; return 1; // ẛ → ṡ

    // Greek Extended: vowels with breathing marks (irregular offsets)
    case 0x1F59: folded[0] = 0x1F51; return 1; // 'Ὑ' → 'ὑ'
    case 0x1F5B: folded[0] = 0x1F53; return 1; // 'Ὓ' → 'ὓ'
    case 0x1F5D: folded[0] = 0x1F55; return 1; // 'Ὕ' → 'ὕ'
    case 0x1F5F: folded[0] = 0x1F57; return 1; // 'Ὗ' → 'ὗ'
    case 0x1FB8: folded[0] = 0x1FB0; return 1; // 'Ᾰ' → 'ᾰ'
    case 0x1FB9: folded[0] = 0x1FB1; return 1; // 'Ᾱ' → 'ᾱ'
    case 0x1FBA: folded[0] = 0x1F70; return 1; // 'Ὰ' → 'ὰ'
    case 0x1FBB: folded[0] = 0x1F71; return 1; // 'Ά' → 'ά'
    case 0x1FBE: folded[0] = 0x03B9; return 1; // 'ι' → 'ι'
    case 0x1FD8: folded[0] = 0x1FD0; return 1; // 'Ῐ' → 'ῐ'
    case 0x1FD9: folded[0] = 0x1FD1; return 1; // 'Ῑ' → 'ῑ'
    case 0x1FDA: folded[0] = 0x1F76; return 1; // 'Ὶ' → 'ὶ'
    case 0x1FDB: folded[0] = 0x1F77; return 1; // 'Ί' → 'ί'
    case 0x1FE8: folded[0] = 0x1FE0; return 1; // 'Ῠ' → 'ῠ'
    case 0x1FE9: folded[0] = 0x1FE1; return 1; // 'Ῡ' → 'ῡ'
    case 0x1FEA: folded[0] = 0x1F7A; return 1; // 'Ὺ' → 'ὺ'
    case 0x1FEB: folded[0] = 0x1F7B; return 1; // 'Ύ' → 'ύ'
    case 0x1FEC: folded[0] = 0x1FE5; return 1; // 'Ῥ' → 'ῥ'
    case 0x1FF8: folded[0] = 0x1F78; return 1; // 'Ὸ' → 'ὸ'
    case 0x1FF9: folded[0] = 0x1F79; return 1; // 'Ό' → 'ό'
    case 0x1FFA: folded[0] = 0x1F7C; return 1; // 'Ὼ' → 'ὼ'
    case 0x1FFB:
        folded[0] = 0x1F7D; return 1; // 'Ώ' → 'ώ'
    // Letterlike Symbols: compatibility mappings
    case 0x2126: folded[0] = 0x03C9; return 1; // 'Ω' → 'ω'
    case 0x212A: folded[0] = 0x006B; return 1; // 'K' → 'k'
    case 0x212B: folded[0] = 0x00E5; return 1; // 'Å' → 'å'
    case 0x2132: folded[0] = 0x214E; return 1; // 'Ⅎ' → 'ⅎ'
    case 0x2183:
        folded[0] = 0x2184; return 1; // 'Ↄ' → 'ↄ'

    // Latin Extended-C: irregular mappings to IPA/other blocks
    case 0x2C60: folded[0] = 0x2C61; return 1; // 'Ⱡ' → 'ⱡ'
    case 0x2C62: folded[0] = 0x026B; return 1; // 'Ɫ' → 'ɫ'
    case 0x2C63: folded[0] = 0x1D7D; return 1; // 'Ᵽ' → 'ᵽ'
    case 0x2C64: folded[0] = 0x027D; return 1; // 'Ɽ' → 'ɽ'
    case 0x2C67: folded[0] = 0x2C68; return 1; // 'Ⱨ' → 'ⱨ'
    case 0x2C69: folded[0] = 0x2C6A; return 1; // 'Ⱪ' → 'ⱪ'
    case 0x2C6B: folded[0] = 0x2C6C; return 1; // 'Ⱬ' → 'ⱬ'
    case 0x2C6D: folded[0] = 0x0251; return 1; // 'Ɑ' → 'ɑ'
    case 0x2C6E: folded[0] = 0x0271; return 1; // 'Ɱ' → 'ɱ'
    case 0x2C6F: folded[0] = 0x0250; return 1; // 'Ɐ' → 'ɐ'
    case 0x2C70: folded[0] = 0x0252; return 1; // 'Ɒ' → 'ɒ'
    case 0x2C72: folded[0] = 0x2C73; return 1; // 'Ⱳ' → 'ⱳ'
    case 0x2C75: folded[0] = 0x2C76; return 1; // 'Ⱶ' → 'ⱶ'
    case 0x2C7E: folded[0] = 0x023F; return 1; // 'Ȿ' → 'ȿ'
    case 0x2C7F:
        folded[0] = 0x0240; return 1; // 'Ɀ' → 'ɀ'

    // Coptic: irregular cases outside the even/odd range
    case 0x2CEB: folded[0] = 0x2CEC; return 1; // 'Ⳬ' → 'ⳬ'
    case 0x2CED: folded[0] = 0x2CEE; return 1; // 'Ⳮ' → 'ⳮ'
    case 0x2CF2:
        folded[0] = 0x2CF3; return 1; // 'Ⳳ' → 'ⳳ'

    // Latin Extended-D: isolated irregulars with non-standard offsets
    case 0xA779: folded[0] = 0xA77A; return 1; // 'Ꝺ' → 'ꝺ'
    case 0xA77B: folded[0] = 0xA77C; return 1; // 'Ꝼ' → 'ꝼ'
    case 0xA77D: folded[0] = 0x1D79; return 1; // 'Ᵹ' → 'ᵹ'
    case 0xA78B: folded[0] = 0xA78C; return 1; // 'Ꞌ' → 'ꞌ'
    case 0xA78D: folded[0] = 0x0265; return 1; // 'Ɥ' → 'ɥ'
    case 0xA7AA: folded[0] = 0x0266; return 1; // 'Ɦ' → 'ɦ'
    case 0xA7AB: folded[0] = 0x025C; return 1; // 'Ɜ' → 'ɜ'
    case 0xA7AC: folded[0] = 0x0261; return 1; // 'Ɡ' → 'ɡ'
    case 0xA7AD: folded[0] = 0x026C; return 1; // 'Ɬ' → 'ɬ'
    case 0xA7AE: folded[0] = 0x026A; return 1; // 'Ɪ' → 'ɪ'
    case 0xA7B0: folded[0] = 0x029E; return 1; // 'Ʞ' → 'ʞ'
    case 0xA7B1: folded[0] = 0x0287; return 1; // 'Ʇ' → 'ʇ'
    case 0xA7B2: folded[0] = 0x029D; return 1; // 'Ʝ' → 'ʝ'
    case 0xA7B3: folded[0] = 0xAB53; return 1; // 'Ꭓ' → 'ꭓ'
    case 0xA7C4: folded[0] = 0xA794; return 1; // 'Ꞔ' → 'ꞔ'
    case 0xA7C5: folded[0] = 0x0282; return 1; // 'Ʂ' → 'ʂ'
    case 0xA7C6: folded[0] = 0x1D8E; return 1; // 'Ᶎ' → 'ᶎ'
    case 0xA7CB: folded[0] = 0x0264; return 1; // 'Ɤ' → 'ɤ'
    case 0xA7DC:
        folded[0] = 0x019B; return 1; // 'Ƛ' → 'ƛ'

    // Vithkuqi: Albanian historical script
    case 0x10594: folded[0] = 0x105BB; return 1; // '𐖔' → '𐖻'
    case 0x10595: folded[0] = 0x105BC; return 1; // '𐖕' → '𐖼'
    }

    // One-to-many expansions
    switch (rune) {
    case 0x00DF: folded[0] = 0x0073; folded[1] = 0x0073; return 2; // ß → ss (German)
    case 0x0130: folded[0] = 0x0069; folded[1] = 0x0307; return 2; // İ → i + combining (Turkish)
    case 0x0149: folded[0] = 0x02BC; folded[1] = 0x006E; return 2; // ŉ → ʼn (Afrikaans)
    case 0x01F0: folded[0] = 0x006A; folded[1] = 0x030C; return 2; // ǰ → j + combining
    case 0x0390: folded[0] = 0x03B9; folded[1] = 0x0308; folded[2] = 0x0301; return 3; // ΐ → ι + 2 combining (Greek)
    case 0x03B0: folded[0] = 0x03C5; folded[1] = 0x0308; folded[2] = 0x0301; return 3; // ΰ → υ + 2 combining (Greek)
    case 0x0587: folded[0] = 0x0565; folded[1] = 0x0582; return 2; // և → եdelays (Armenian)
    case 0x1E96: folded[0] = 0x0068; folded[1] = 0x0331; return 2; // ẖ → h + combining
    case 0x1E97: folded[0] = 0x0074; folded[1] = 0x0308; return 2; // ẗ → t + combining
    case 0x1E98: folded[0] = 0x0077; folded[1] = 0x030A; return 2; // ẘ → w + combining
    case 0x1E99: folded[0] = 0x0079; folded[1] = 0x030A; return 2; // ẙ → y + combining
    case 0x1E9A: folded[0] = 0x0061; folded[1] = 0x02BE; return 2; // ẚ → aʾ
    case 0x1E9E: folded[0] = 0x0073; folded[1] = 0x0073; return 2; // ẞ → ss (German capital Eszett)
    case 0x1F50: folded[0] = 0x03C5; folded[1] = 0x0313; return 2; // ὐ → υ + combining (Greek)
    case 0x1F52: folded[0] = 0x03C5; folded[1] = 0x0313; folded[2] = 0x0300; return 3; // ὒ → υ + 2 combining
    case 0x1F54: folded[0] = 0x03C5; folded[1] = 0x0313; folded[2] = 0x0301; return 3; // ὔ → υ + 2 combining
    case 0x1F56: folded[0] = 0x03C5; folded[1] = 0x0313; folded[2] = 0x0342; return 3; // ὖ → υ + 2 combining
    case 0x1F80: folded[0] = 0x1F00; folded[1] = 0x03B9; return 2; // ᾀ → ἀι (Greek iota subscript)
    case 0x1F81: folded[0] = 0x1F01; folded[1] = 0x03B9; return 2; // ᾁ → ἁι
    case 0x1F82: folded[0] = 0x1F02; folded[1] = 0x03B9; return 2; // ᾂ → ἂι
    case 0x1F83: folded[0] = 0x1F03; folded[1] = 0x03B9; return 2; // ᾃ → ἃι
    case 0x1F84: folded[0] = 0x1F04; folded[1] = 0x03B9; return 2; // ᾄ → ἄι
    case 0x1F85: folded[0] = 0x1F05; folded[1] = 0x03B9; return 2; // ᾅ → ἅι
    case 0x1F86: folded[0] = 0x1F06; folded[1] = 0x03B9; return 2; // ᾆ → ἆι
    case 0x1F87: folded[0] = 0x1F07; folded[1] = 0x03B9; return 2; // ᾇ → ἇι
    case 0x1F88: folded[0] = 0x1F00; folded[1] = 0x03B9; return 2; // ᾈ → ἀι
    case 0x1F89: folded[0] = 0x1F01; folded[1] = 0x03B9; return 2; // ᾉ → ἁι
    case 0x1F8A: folded[0] = 0x1F02; folded[1] = 0x03B9; return 2; // ᾊ → ἂι
    case 0x1F8B: folded[0] = 0x1F03; folded[1] = 0x03B9; return 2; // ᾋ → ἃι
    case 0x1F8C: folded[0] = 0x1F04; folded[1] = 0x03B9; return 2; // ᾌ → ἄι
    case 0x1F8D: folded[0] = 0x1F05; folded[1] = 0x03B9; return 2; // ᾍ → ἅι
    case 0x1F8E: folded[0] = 0x1F06; folded[1] = 0x03B9; return 2; // ᾎ → ἆι
    case 0x1F8F: folded[0] = 0x1F07; folded[1] = 0x03B9; return 2; // ᾏ → ἇι
    case 0x1F90: folded[0] = 0x1F20; folded[1] = 0x03B9; return 2; // ᾐ → ἠι
    case 0x1F91: folded[0] = 0x1F21; folded[1] = 0x03B9; return 2; // ᾑ → ἡι
    case 0x1F92: folded[0] = 0x1F22; folded[1] = 0x03B9; return 2; // ᾒ → ἢι
    case 0x1F93: folded[0] = 0x1F23; folded[1] = 0x03B9; return 2; // ᾓ → ἣι
    case 0x1F94: folded[0] = 0x1F24; folded[1] = 0x03B9; return 2; // ᾔ → ἤι
    case 0x1F95: folded[0] = 0x1F25; folded[1] = 0x03B9; return 2; // ᾕ → ἥι
    case 0x1F96: folded[0] = 0x1F26; folded[1] = 0x03B9; return 2; // ᾖ → ἦι
    case 0x1F97: folded[0] = 0x1F27; folded[1] = 0x03B9; return 2; // ᾗ → ἧι
    case 0x1F98: folded[0] = 0x1F20; folded[1] = 0x03B9; return 2; // ᾘ → ἠι
    case 0x1F99: folded[0] = 0x1F21; folded[1] = 0x03B9; return 2; // ᾙ → ἡι
    case 0x1F9A: folded[0] = 0x1F22; folded[1] = 0x03B9; return 2; // ᾚ → ἢι
    case 0x1F9B: folded[0] = 0x1F23; folded[1] = 0x03B9; return 2; // ᾛ → ἣι
    case 0x1F9C: folded[0] = 0x1F24; folded[1] = 0x03B9; return 2; // ᾜ → ἤι
    case 0x1F9D: folded[0] = 0x1F25; folded[1] = 0x03B9; return 2; // ᾝ → ἥι
    case 0x1F9E: folded[0] = 0x1F26; folded[1] = 0x03B9; return 2; // ᾞ → ἦι
    case 0x1F9F: folded[0] = 0x1F27; folded[1] = 0x03B9; return 2; // ᾟ → ἧι
    case 0x1FA0: folded[0] = 0x1F60; folded[1] = 0x03B9; return 2; // ᾠ → ὠι
    case 0x1FA1: folded[0] = 0x1F61; folded[1] = 0x03B9; return 2; // ᾡ → ὡι
    case 0x1FA2: folded[0] = 0x1F62; folded[1] = 0x03B9; return 2; // ᾢ → ὢι
    case 0x1FA3: folded[0] = 0x1F63; folded[1] = 0x03B9; return 2; // ᾣ → ὣι
    case 0x1FA4: folded[0] = 0x1F64; folded[1] = 0x03B9; return 2; // ᾤ → ὤι
    case 0x1FA5: folded[0] = 0x1F65; folded[1] = 0x03B9; return 2; // ᾥ → ὥι
    case 0x1FA6: folded[0] = 0x1F66; folded[1] = 0x03B9; return 2; // ᾦ → ὦι
    case 0x1FA7: folded[0] = 0x1F67; folded[1] = 0x03B9; return 2; // ᾧ → ὧι
    case 0x1FA8: folded[0] = 0x1F60; folded[1] = 0x03B9; return 2; // ᾨ → ὠι
    case 0x1FA9: folded[0] = 0x1F61; folded[1] = 0x03B9; return 2; // ᾩ → ὡι
    case 0x1FAA: folded[0] = 0x1F62; folded[1] = 0x03B9; return 2; // ᾪ → ὢι
    case 0x1FAB: folded[0] = 0x1F63; folded[1] = 0x03B9; return 2; // ᾫ → ὣι
    case 0x1FAC: folded[0] = 0x1F64; folded[1] = 0x03B9; return 2; // ᾬ → ὤι
    case 0x1FAD: folded[0] = 0x1F65; folded[1] = 0x03B9; return 2; // ᾭ → ὥι
    case 0x1FAE: folded[0] = 0x1F66; folded[1] = 0x03B9; return 2; // ᾮ → ὦι
    case 0x1FAF: folded[0] = 0x1F67; folded[1] = 0x03B9; return 2; // ᾯ → ὧι
    case 0x1FB2: folded[0] = 0x1F70; folded[1] = 0x03B9; return 2; // ᾲ → ὰι
    case 0x1FB3: folded[0] = 0x03B1; folded[1] = 0x03B9; return 2; // ᾳ → αι
    case 0x1FB4: folded[0] = 0x03AC; folded[1] = 0x03B9; return 2; // ᾴ → άι
    case 0x1FB6: folded[0] = 0x03B1; folded[1] = 0x0342; return 2; // ᾶ → α + combining
    case 0x1FB7: folded[0] = 0x03B1; folded[1] = 0x0342; folded[2] = 0x03B9; return 3; // ᾷ → α + 2 combining
    case 0x1FBC: folded[0] = 0x03B1; folded[1] = 0x03B9; return 2; // ᾼ → αι
    case 0x1FC2: folded[0] = 0x1F74; folded[1] = 0x03B9; return 2; // ῂ → ὴι
    case 0x1FC3: folded[0] = 0x03B7; folded[1] = 0x03B9; return 2; // ῃ → ηι
    case 0x1FC4: folded[0] = 0x03AE; folded[1] = 0x03B9; return 2; // ῄ → ήι
    case 0x1FC6: folded[0] = 0x03B7; folded[1] = 0x0342; return 2; // ῆ → η + combining
    case 0x1FC7: folded[0] = 0x03B7; folded[1] = 0x0342; folded[2] = 0x03B9; return 3; // ῇ → η + 2 combining
    case 0x1FCC: folded[0] = 0x03B7; folded[1] = 0x03B9; return 2; // ῌ → ηι
    case 0x1FD2: folded[0] = 0x03B9; folded[1] = 0x0308; folded[2] = 0x0300; return 3; // ῒ → ι + 2 combining
    case 0x1FD3: folded[0] = 0x03B9; folded[1] = 0x0308; folded[2] = 0x0301; return 3; // ΐ → ι + 2 combining
    case 0x1FD6: folded[0] = 0x03B9; folded[1] = 0x0342; return 2; // ῖ → ι + combining
    case 0x1FD7: folded[0] = 0x03B9; folded[1] = 0x0308; folded[2] = 0x0342; return 3; // ῗ → ι + 2 combining
    case 0x1FE2: folded[0] = 0x03C5; folded[1] = 0x0308; folded[2] = 0x0300; return 3; // ῢ → υ + 2 combining
    case 0x1FE3: folded[0] = 0x03C5; folded[1] = 0x0308; folded[2] = 0x0301; return 3; // ΰ → υ + 2 combining
    case 0x1FE4: folded[0] = 0x03C1; folded[1] = 0x0313; return 2; // ῤ → ρ + combining
    case 0x1FE6: folded[0] = 0x03C5; folded[1] = 0x0342; return 2; // ῦ → υ + combining
    case 0x1FE7: folded[0] = 0x03C5; folded[1] = 0x0308; folded[2] = 0x0342; return 3; // ῧ → υ + 2 combining
    case 0x1FF2: folded[0] = 0x1F7C; folded[1] = 0x03B9; return 2; // ῲ → ὼι
    case 0x1FF3: folded[0] = 0x03C9; folded[1] = 0x03B9; return 2; // ῳ → ωι
    case 0x1FF4: folded[0] = 0x03CE; folded[1] = 0x03B9; return 2; // ῴ → ώι
    case 0x1FF6: folded[0] = 0x03C9; folded[1] = 0x0342; return 2; // ῶ → ω + combining
    case 0x1FF7: folded[0] = 0x03C9; folded[1] = 0x0342; folded[2] = 0x03B9; return 3; // ῷ → ω + 2 combining
    case 0x1FFC: folded[0] = 0x03C9; folded[1] = 0x03B9; return 2; // ῼ → ωι
    case 0xFB00: folded[0] = 0x0066; folded[1] = 0x0066; return 2; // ﬀ → ff
    case 0xFB01: folded[0] = 0x0066; folded[1] = 0x0069; return 2; // ﬁ → fi
    case 0xFB02: folded[0] = 0x0066; folded[1] = 0x006C; return 2; // ﬂ → fl
    case 0xFB03: folded[0] = 0x0066; folded[1] = 0x0066; folded[2] = 0x0069; return 3; // ﬃ → ffi
    case 0xFB04: folded[0] = 0x0066; folded[1] = 0x0066; folded[2] = 0x006C; return 3; // ﬄ → ffl
    case 0xFB05: folded[0] = 0x0073; folded[1] = 0x0074; return 2; // ﬅ → st
    case 0xFB06: folded[0] = 0x0073; folded[1] = 0x0074; return 2; // ﬆ → st
    case 0xFB13: folded[0] = 0x0574; folded[1] = 0x0576; return 2; // ﬓ → մն
    case 0xFB14: folded[0] = 0x0574; folded[1] = 0x0565; return 2; // ﬔ → մե
    case 0xFB15: folded[0] = 0x0574; folded[1] = 0x056B; return 2; // ﬕ → մի
    case 0xFB16: folded[0] = 0x057E; folded[1] = 0x0576; return 2; // ﬖ → վն
    case 0xFB17: folded[0] = 0x0574; folded[1] = 0x056D; return 2; // ﬗ → մխ
    }

    folded[0] = rune; return 1; // No folding
    // clang-format on
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
    sz_rune_t folded_needle[1024];
    sz_size_t folded_needle_count = 0;
    sz_cptr_t needle_ptr = needle;
    sz_cptr_t needle_end = needle + needle_length;

    while (needle_ptr < needle_end && folded_needle_count < 1024) {
        sz_rune_t cp;
        sz_rune_length_t rune_length;
        sz_rune_parse(needle_ptr, &cp, &rune_length);
        if (rune_length == 0) break;

        // Apply case folding
        sz_rune_t folded[4];
        sz_size_t folded_count = sz_unicode_fold_codepoint_(cp, folded);
        for (sz_size_t i = 0; i < folded_count && folded_needle_count < 1024; ++i)
            folded_needle[folded_needle_count++] = folded[i];

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
        sz_bool_t mismatch = sz_false_k;

        // Try to match the folded needle at this position
        while (needle_idx < folded_needle_count && match_ptr < haystack_end && !mismatch) {
            sz_rune_t cp;
            sz_rune_length_t rune_length;
            sz_rune_parse(match_ptr, &cp, &rune_length);
            if (rune_length == 0) break;

            // Apply case folding
            sz_rune_t folded[4];
            sz_size_t folded_count = sz_unicode_fold_codepoint_(cp, folded);

            // Compare all folded codepoints against the needle
            for (sz_size_t i = 0; i < folded_count && needle_idx < folded_needle_count; ++i) {
                if (folded[i] != folded_needle[needle_idx]) {
                    mismatch = sz_true_k;
                    break;
                }
                needle_idx++;
            }

            match_ptr += rune_length;
        }

        // Did we match the entire needle?
        if (needle_idx == folded_needle_count && !mismatch) {
            *matched_length = (sz_size_t)(match_ptr - match_start);
            return match_start;
        }

        // Move to next codepoint in haystack
        sz_rune_t cp;
        sz_rune_length_t rune_length;
        sz_rune_parse(haystack_ptr, &cp, &rune_length);
        if (rune_length == 0) break;
        haystack_ptr += rune_length;
    }

    *matched_length = 0;
    return SZ_NULL_CHAR;
}

SZ_PUBLIC sz_status_t sz_utf8_case_fold_serial( //
    sz_cptr_t source, sz_size_t source_length,  //
    sz_ptr_t destination, sz_size_t destination_capacity, sz_size_t *destination_length) {

    sz_u8_t const *src = (sz_u8_t const *)source;
    sz_u8_t const *src_end = src + source_length;
    sz_u8_t *dst = (sz_u8_t *)destination;
    sz_u8_t *dst_end = dst + destination_capacity;
    sz_u8_t *dst_start = dst;

    while (src < src_end) {
        // Decode one UTF-8 codepoint
        sz_rune_t cp;
        sz_rune_length_t rune_length;
        sz_rune_parse((sz_cptr_t)src, &cp, &rune_length);
        if (rune_length == 0) {
            *destination_length = (sz_size_t)(dst - dst_start);
            return sz_invalid_utf8_k;
        }
        src += rune_length;

        // Apply case folding
        sz_rune_t folded[4];
        sz_size_t folded_count = sz_unicode_fold_codepoint_(cp, folded);

        // Encode all folded codepoints
        for (sz_size_t i = 0; i < folded_count; ++i) {
            if (dst + 4 > dst_end) {
                *destination_length = (sz_size_t)(dst - dst_start);
                return sz_bad_alloc_k;
            }
            sz_size_t written = sz_rune_export_(folded[i], dst);
            if (written == 0) {
                *destination_length = (sz_size_t)(dst - dst_start);
                return sz_invalid_utf8_k;
            }
            dst += written;
        }
    }

    *destination_length = (sz_size_t)(dst - dst_start);
    return sz_success_k;
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

SZ_PUBLIC sz_status_t sz_utf8_case_fold_ice(   //
    sz_cptr_t source, sz_size_t source_length, //
    sz_ptr_t destination, sz_size_t destination_capacity, sz_size_t *destination_length) {
    return sz_utf8_case_fold_serial(source, source_length, destination, destination_capacity, destination_length);
}

SZ_PUBLIC sz_cptr_t sz_utf8_find_case_insensitive_ice( //
    sz_cptr_t haystack, sz_size_t haystack_length,     //
    sz_cptr_t needle, sz_size_t needle_length, sz_size_t *matched_length) {
    return sz_utf8_find_case_insensitive_serial(haystack, haystack_length, needle, needle_length, matched_length);
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

SZ_PUBLIC sz_status_t sz_utf8_case_fold_haswell( //
    sz_cptr_t source, sz_size_t source_length,   //
    sz_ptr_t destination, sz_size_t destination_capacity, sz_size_t *destination_length) {
    return sz_utf8_case_fold_serial(source, source_length, destination, destination_capacity, destination_length);
}

SZ_PUBLIC sz_cptr_t sz_utf8_find_case_insensitive_haswell( //
    sz_cptr_t haystack, sz_size_t haystack_length,         //
    sz_cptr_t needle, sz_size_t needle_length, sz_size_t *matched_length) {
    return sz_utf8_find_case_insensitive_serial(haystack, haystack_length, needle, needle_length, matched_length);
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

SZ_PUBLIC sz_status_t sz_utf8_case_fold_neon(  //
    sz_cptr_t source, sz_size_t source_length, //
    sz_ptr_t destination, sz_size_t destination_capacity, sz_size_t *destination_length) {
    return sz_utf8_case_fold_serial(source, source_length, destination, destination_capacity, destination_length);
}

SZ_PUBLIC sz_cptr_t sz_utf8_find_case_insensitive_neon( //
    sz_cptr_t haystack, sz_size_t haystack_length,      //
    sz_cptr_t needle, sz_size_t needle_length, sz_size_t *matched_length) {
    return sz_utf8_find_case_insensitive_serial(haystack, haystack_length, needle, needle_length, matched_length);
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

SZ_DYNAMIC sz_status_t sz_utf8_case_fold(sz_cptr_t source, sz_size_t source_length, sz_ptr_t destination,
                                         sz_size_t destination_capacity, sz_size_t *destination_length) {
#if SZ_USE_ICE
    return sz_utf8_case_fold_ice(source, source_length, destination, destination_capacity, destination_length);
#elif SZ_USE_HASWELL
    return sz_utf8_case_fold_haswell(source, source_length, destination, destination_capacity, destination_length);
#else
    return sz_utf8_case_fold_serial(source, source_length, destination, destination_capacity, destination_length);
#endif
}

SZ_DYNAMIC sz_cptr_t sz_utf8_find_case_insensitive(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle,
                                                   sz_size_t needle_length, sz_size_t *matched_length) {
#if SZ_USE_ICE
    return sz_utf8_find_case_insensitive_ice(haystack, haystack_length, needle, needle_length, matched_length);
#elif SZ_USE_HASWELL
    return sz_utf8_find_case_insensitive_haswell(haystack, haystack_length, needle, needle_length, matched_length);
#else
    return sz_utf8_find_case_insensitive_serial(haystack, haystack_length, needle, needle_length, matched_length);
#endif
}

#endif // !SZ_DYNAMIC_DISPATCH

#pragma endregion // Dynamic Dispatch

#ifdef __cplusplus
}
#endif

#endif // STRINGZILLA_UTF8_H_
