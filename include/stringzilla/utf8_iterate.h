/**
 *  @brief Hardware-accelerated UTF-8 text processing utilities.
 *  @file include/stringzilla/utf8_iterate.h
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
#ifndef STRINGZILLA_UTF8_ITERATE_H_
#define STRINGZILLA_UTF8_ITERATE_H_

#include "stringzilla/types.h"

#ifdef __cplusplus
extern "C" {
#endif

#pragma region Word_Break Property Encoding

/**
 *  @brief Unicode TR29 Word_Break property values (4-bit encoding, 0-15).
 *
 *  These values correspond to the Word_Break property from Unicode TR29.
 *  Used by `sz_rune_word_break_property()` for full TR29-compliant boundary detection.
 */
typedef enum sz_tr29_word_break_t {
    sz_tr29_word_break_other_k = 0,         /**< Default - creates word boundary */
    sz_tr29_word_break_cr_k = 1,            /**< Carriage Return (U+000D) */
    sz_tr29_word_break_lf_k = 2,            /**< Line Feed (U+000A) */
    sz_tr29_word_break_newline_k = 3,       /**< Other newlines (VT, FF, NEL, LS, PS) */
    sz_tr29_word_break_extend_k = 4,        /**< Combining marks (Mn, Me, Mc) */
    sz_tr29_word_break_zwj_k = 5,           /**< Zero Width Joiner (U+200D) */
    sz_tr29_word_break_format_k = 6,        /**< Format characters (Cf) */
    sz_tr29_word_break_regional_ind_k = 7,  /**< Regional Indicator (U+1F1E6-U+1F1FF) */
    sz_tr29_word_break_aletter_k = 8,       /**< Alphabetic letters */
    sz_tr29_word_break_hebrew_letter_k = 9, /**< Hebrew script letters */
    sz_tr29_word_break_numeric_k = 10,      /**< Digits (0-9 and other scripts) */
    sz_tr29_word_break_katakana_k = 11,     /**< Japanese Katakana */
    sz_tr29_word_break_extendnumlet_k = 12, /**< Underscore, connector punctuation */
    sz_tr29_word_break_midletter_k = 13,    /**< Mid-letter punctuation (colon, etc.) */
    sz_tr29_word_break_midnum_k = 14,       /**< Mid-number punctuation (comma, etc.) */
    sz_tr29_word_break_mid_quotes_k = 15,   /**< MidNumLet + Single_Quote + Double_Quote */
} sz_tr29_word_break_t;

#pragma endregion

#pragma region Core API

/**
 *  @brief Count the number of UTF-8 characters in a string.
 *
 *  The logic is to count the number of "continuation bytes" matching the 10xxxxxx pattern,
 *  and then subtract that from the total byte length to get the number of "start bytes" -
 *  coinciding with the number of UTF-8 characters.
 *
 *  @param text String to be scanned.
 *  @param length Number of bytes in the string.
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
 *  @brief Skip forward to the Nth UTF-8 character.
 *
 *  @param text String to be scanned.
 *  @param length Number of bytes in the string.
 *  @param n Number of UTF-8 characters to skip (0-indexed, so n=0 returns text).
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
 *  @brief Skips to the first occurrence of a UTF-8 newline character in a string.
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
 *  @param text String to be scanned.
 *  @param length Number of bytes in the string.
 *  @param matched_length Number of bytes in the matched newline delimiter.
 *  @return Pointer to the first matching newline character from @p text, or @c SZ_NULL_CHAR if not found.
 */
SZ_DYNAMIC sz_cptr_t sz_utf8_find_newline(sz_cptr_t text, sz_size_t length, sz_size_t *matched_length);

/**
 *  @brief Skips to the first occurrence of a UTF-8 whitespace character in a string.
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
 *  @param text String to be scanned.
 *  @param length Number of bytes in the string.
 *  @param matched_length Number of bytes in the matched whitespace character.
 *  @return Pointer to the first matching whitespace character from @p text, or @c SZ_NULL_CHAR if not found.
 */
SZ_DYNAMIC sz_cptr_t sz_utf8_find_whitespace(sz_cptr_t text, sz_size_t length, sz_size_t *matched_length);

/**
 *  @brief Unpack a UTF-8 string into UTF-32 codepoints.
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
 *  @param text UTF-8 string to unpack.
 *  @param length Number of bytes in the string (up to 64).
 *  @param runes Output buffer for UTF-32 codepoints (recommended to be at least @b 64 entries wide).
 *  @param runes_capacity Capacity of the @p runes buffer (number of sz_rune_t entries).
 *  @param runes_unpacked Number of runes unpacked.
 *  @return Pointer to the byte after the last unpacked byte in @p text.
 */
SZ_DYNAMIC sz_cptr_t sz_utf8_unpack_chunk(      //
    sz_cptr_t text, sz_size_t length,           //
    sz_rune_t *runes, sz_size_t runes_capacity, //
    sz_size_t *runes_unpacked);

/**
 *  @brief Get the Unicode TR29 Word_Break property for a codepoint.
 *
 *  Returns one of the 16 Word_Break property values (sz_tr29_word_break_other_k through
 *  sz_tr29_word_break_mid_quotes_k). This is the foundation for TR29-compliant word boundary detection.
 *
 *  @param rune The Unicode codepoint to classify.
 *  @return The Word_Break property value (0-15).
 *
 *  @see https://www.unicode.org/reports/tr29/ - Unicode Text Segmentation
 */
SZ_PUBLIC sz_u8_t sz_rune_word_break_property(sz_rune_t rune);

/**
 *  @brief Check if a codepoint is a "word character" (has word-forming property).
 *
 *  Returns true if the codepoint has a Word_Break property that typically forms words:
 *  ALetter, Hebrew_Letter, Numeric, Katakana, ExtendNumLet, or mid-word punctuation.
 *
 *  @param rune The Unicode codepoint to check.
 *  @return sz_true_k if the codepoint is a word character, sz_false_k otherwise.
 */
SZ_PUBLIC sz_bool_t sz_rune_is_word_char(sz_rune_t rune);

/**
 *  @brief Suggested default batch size for callers that stream words through `sz_utf8_word_*_boundaries`.
 *
 *  Iterators that emit one word at a time (the Python and C++ bindings) buffer this many boundaries per call
 *  so the per-word overhead amortizes without an unbounded output buffer. It is only a default — any capacity
 *  works, and the kernels report `bytes_consumed` so the caller can resume past a full buffer.
 */
enum { sz_utf8_word_boundaries_batch_k = 16 };

/**
 *  @brief Segment UTF-8 text into UAX-29 words in a single pass (dispatch function).
 *
 *  Walks the whole input left-to-right and writes one entry per word into two parallel output arrays:
 *  `word_starts[i]` is the byte offset of the i-th word and `word_lengths[i]` its byte length. Words are the
 *  spans between consecutive TR29 boundaries, so a single call segments the entire input without the caller
 *  having to loop and restart a scan for every word.
 *
 *  @param text UTF-8 encoded text.
 *  @param length Byte length of @p text.
 *  @param word_starts Output array of word byte offsets (at least @p words_capacity entries).
 *  @param word_lengths Output array of word byte lengths (at least @p words_capacity entries).
 *  @param words_capacity Capacity of the output arrays, in entries.
 *  @param bytes_consumed Optional output: byte offset up to which the input was segmented. Equals @p length
 *         when everything fit; otherwise it is the start of the first word that did not fit (a TR29 boundary),
 *         so the caller may resume from @c text+*bytes_consumed.
 *  @return Number of words written (at most @p words_capacity).
 *
 *  @note No zero-length words are emitted; @p length == 0 returns 0.
 */
SZ_DYNAMIC sz_size_t sz_utf8_word_find_boundaries(   //
    sz_cptr_t text, sz_size_t length,                //
    sz_size_t *word_starts, sz_size_t *word_lengths, //
    sz_size_t words_capacity, sz_size_t *bytes_consumed);

/**
 *  @brief Segment UTF-8 text into UAX-29 words from the end backward (dispatch function).
 *
 *  Like @c sz_utf8_word_find_boundaries but emits words starting from the end of the text: `word_starts[0]` is
 *  the last word, `word_starts[1]` the one before it, and so on. On a full buffer @p bytes_consumed is the end
 *  offset of the earliest emitted word (a TR29 boundary); resume by calling again with @c length == *bytes_consumed.
 *
 *  @param text UTF-8 encoded text.
 *  @param length Byte length of @p text.
 *  @param word_starts Output array of word byte offsets (at least @p words_capacity entries).
 *  @param word_lengths Output array of word byte lengths (at least @p words_capacity entries).
 *  @param words_capacity Capacity of the output arrays, in entries.
 *  @param bytes_consumed Optional output: byte offset down to which the input was segmented (0 when done).
 *  @return Number of words written (at most @p words_capacity).
 */
SZ_DYNAMIC sz_size_t sz_utf8_word_rfind_boundaries(  //
    sz_cptr_t text, sz_size_t length,                //
    sz_size_t *word_starts, sz_size_t *word_lengths, //
    sz_size_t words_capacity, sz_size_t *bytes_consumed);

/**
 *  @brief Check if a position in UTF-8 text is a word boundary per Unicode TR29.
 *
 *  Implements the full TR29 word boundary algorithm including:
 *  - WB3: Do not break between CR and LF
 *  - WB4: Ignore Extend/Format/ZWJ characters for boundary purposes
 *  - WB5-WB13: Letter, number, and punctuation rules
 *  - WB15-WB16: Regional Indicator pair rules
 *
 *  @param text UTF-8 encoded text.
 *  @param length Byte length of text.
 *  @param position Byte offset to check (must be start of a UTF-8 codepoint).
 *  @return sz_true_k if position is a word boundary, sz_false_k otherwise.
 *
 *  @note Position 0 and position == length are always boundaries (SOT/EOT).
 *  @note This is an internal helper used by the iterators; not part of stable ABI.
 */
SZ_PUBLIC sz_bool_t sz_utf8_is_word_boundary_serial(sz_cptr_t text, sz_size_t length, sz_size_t position);

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
/** @copydoc sz_utf8_word_find_boundaries */
SZ_PUBLIC sz_size_t sz_utf8_word_find_boundaries_serial(sz_cptr_t text, sz_size_t length, sz_size_t *word_starts,
                                                        sz_size_t *word_lengths, sz_size_t words_capacity,
                                                        sz_size_t *bytes_consumed);
/** @copydoc sz_utf8_word_rfind_boundaries */
SZ_PUBLIC sz_size_t sz_utf8_word_rfind_boundaries_serial(sz_cptr_t text, sz_size_t length, sz_size_t *word_starts,
                                                         sz_size_t *word_lengths, sz_size_t words_capacity,
                                                         sz_size_t *bytes_consumed);

#if SZ_USE_HASWELL
/** @copydoc sz_utf8_count */
SZ_PUBLIC sz_size_t sz_utf8_count_haswell(sz_cptr_t text, sz_size_t length);
/** @copydoc sz_utf8_find_nth */
SZ_PUBLIC sz_cptr_t sz_utf8_find_nth_haswell(sz_cptr_t text, sz_size_t length, sz_size_t n);
/** @copydoc sz_utf8_find_newline */
SZ_PUBLIC sz_cptr_t sz_utf8_find_newline_haswell(sz_cptr_t text, sz_size_t length, sz_size_t *matched_length);
/** @copydoc sz_utf8_find_whitespace */
SZ_PUBLIC sz_cptr_t sz_utf8_find_whitespace_haswell(sz_cptr_t text, sz_size_t length, sz_size_t *matched_length);
/** @copydoc sz_utf8_word_find_boundaries */
SZ_PUBLIC sz_size_t sz_utf8_word_find_boundaries_haswell(sz_cptr_t text, sz_size_t length, sz_size_t *word_starts,
                                                         sz_size_t *word_lengths, sz_size_t words_capacity,
                                                         sz_size_t *bytes_consumed);
/** @copydoc sz_utf8_word_rfind_boundaries */
SZ_PUBLIC sz_size_t sz_utf8_word_rfind_boundaries_haswell(sz_cptr_t text, sz_size_t length, sz_size_t *word_starts,
                                                          sz_size_t *word_lengths, sz_size_t words_capacity,
                                                          sz_size_t *bytes_consumed);
#endif

#if SZ_USE_ICELAKE
/** @copydoc sz_utf8_count */
SZ_PUBLIC sz_size_t sz_utf8_count_icelake(sz_cptr_t text, sz_size_t length);
/** @copydoc sz_utf8_find_nth */
SZ_PUBLIC sz_cptr_t sz_utf8_find_nth_icelake(sz_cptr_t text, sz_size_t length, sz_size_t n);
/** @copydoc sz_utf8_find_newline */
SZ_PUBLIC sz_cptr_t sz_utf8_find_newline_icelake(sz_cptr_t text, sz_size_t length, sz_size_t *matched_length);
/** @copydoc sz_utf8_find_whitespace */
SZ_PUBLIC sz_cptr_t sz_utf8_find_whitespace_icelake(sz_cptr_t text, sz_size_t length, sz_size_t *matched_length);
/** @copydoc sz_utf8_unpack_chunk */
SZ_PUBLIC sz_cptr_t sz_utf8_unpack_chunk_icelake( //
    sz_cptr_t text, sz_size_t length,             //
    sz_rune_t *runes, sz_size_t runes_capacity, sz_size_t *runes_unpacked);
/** @copydoc sz_utf8_word_find_boundaries */
SZ_PUBLIC sz_size_t sz_utf8_word_find_boundaries_icelake(sz_cptr_t text, sz_size_t length, sz_size_t *word_starts,
                                                         sz_size_t *word_lengths, sz_size_t words_capacity,
                                                         sz_size_t *bytes_consumed);
/** @copydoc sz_utf8_word_rfind_boundaries */
SZ_PUBLIC sz_size_t sz_utf8_word_rfind_boundaries_icelake(sz_cptr_t text, sz_size_t length, sz_size_t *word_starts,
                                                          sz_size_t *word_lengths, sz_size_t words_capacity,
                                                          sz_size_t *bytes_consumed);
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
/** @copydoc sz_utf8_word_find_boundaries */
SZ_PUBLIC sz_size_t sz_utf8_word_find_boundaries_neon(sz_cptr_t text, sz_size_t length, sz_size_t *word_starts,
                                                      sz_size_t *word_lengths, sz_size_t words_capacity,
                                                      sz_size_t *bytes_consumed);
/** @copydoc sz_utf8_word_rfind_boundaries */
SZ_PUBLIC sz_size_t sz_utf8_word_rfind_boundaries_neon(sz_cptr_t text, sz_size_t length, sz_size_t *word_starts,
                                                       sz_size_t *word_lengths, sz_size_t words_capacity,
                                                       sz_size_t *bytes_consumed);
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
/** @copydoc sz_utf8_word_find_boundaries */
SZ_PUBLIC sz_size_t sz_utf8_word_find_boundaries_sve2(sz_cptr_t text, sz_size_t length, sz_size_t *word_starts,
                                                      sz_size_t *word_lengths, sz_size_t words_capacity,
                                                      sz_size_t *bytes_consumed);
/** @copydoc sz_utf8_word_rfind_boundaries */
SZ_PUBLIC sz_size_t sz_utf8_word_rfind_boundaries_sve2(sz_cptr_t text, sz_size_t length, sz_size_t *word_starts,
                                                       sz_size_t *word_lengths, sz_size_t words_capacity,
                                                       sz_size_t *bytes_consumed);
#endif

#if SZ_USE_V128
/** @copydoc sz_utf8_word_find_boundaries */
SZ_PUBLIC sz_size_t sz_utf8_word_find_boundaries_v128(sz_cptr_t text, sz_size_t length, sz_size_t *word_starts,
                                                      sz_size_t *word_lengths, sz_size_t words_capacity,
                                                      sz_size_t *bytes_consumed);
/** @copydoc sz_utf8_word_rfind_boundaries */
SZ_PUBLIC sz_size_t sz_utf8_word_rfind_boundaries_v128(sz_cptr_t text, sz_size_t length, sz_size_t *word_starts,
                                                       sz_size_t *word_lengths, sz_size_t words_capacity,
                                                       sz_size_t *bytes_consumed);
#endif

#if SZ_USE_RVV
/** @copydoc sz_utf8_word_find_boundaries */
SZ_PUBLIC sz_size_t sz_utf8_word_find_boundaries_rvv(sz_cptr_t text, sz_size_t length, sz_size_t *word_starts,
                                                     sz_size_t *word_lengths, sz_size_t words_capacity,
                                                     sz_size_t *bytes_consumed);
/** @copydoc sz_utf8_word_rfind_boundaries */
SZ_PUBLIC sz_size_t sz_utf8_word_rfind_boundaries_rvv(sz_cptr_t text, sz_size_t length, sz_size_t *word_starts,
                                                      sz_size_t *word_lengths, sz_size_t words_capacity,
                                                      sz_size_t *bytes_consumed);
#endif

#if SZ_USE_LASX
/** @copydoc sz_utf8_word_find_boundaries */
SZ_PUBLIC sz_size_t sz_utf8_word_find_boundaries_lasx(sz_cptr_t text, sz_size_t length, sz_size_t *word_starts,
                                                      sz_size_t *word_lengths, sz_size_t words_capacity,
                                                      sz_size_t *bytes_consumed);
/** @copydoc sz_utf8_word_rfind_boundaries */
SZ_PUBLIC sz_size_t sz_utf8_word_rfind_boundaries_lasx(sz_cptr_t text, sz_size_t length, sz_size_t *word_starts,
                                                       sz_size_t *word_lengths, sz_size_t words_capacity,
                                                       sz_size_t *bytes_consumed);
#endif

#if SZ_USE_POWERVSX
/** @copydoc sz_utf8_word_find_boundaries */
SZ_PUBLIC sz_size_t sz_utf8_word_find_boundaries_powervsx(sz_cptr_t text, sz_size_t length, sz_size_t *word_starts,
                                                          sz_size_t *word_lengths, sz_size_t words_capacity,
                                                          sz_size_t *bytes_consumed);
/** @copydoc sz_utf8_word_rfind_boundaries */
SZ_PUBLIC sz_size_t sz_utf8_word_rfind_boundaries_powervsx(sz_cptr_t text, sz_size_t length, sz_size_t *word_starts,
                                                           sz_size_t *word_lengths, sz_size_t words_capacity,
                                                           sz_size_t *bytes_consumed);
#endif

#pragma endregion

/*  Implementation Section
 *  Each ISA backend lives in its own header, included serial-first.
 */
#include "stringzilla/utf8_iterate/serial.h"
#include "stringzilla/utf8_iterate/icelake.h"
#include "stringzilla/utf8_iterate/haswell.h"
#include "stringzilla/utf8_iterate/neon.h"
#include "stringzilla/utf8_iterate/sve2.h"
#include "stringzilla/utf8_iterate/v128.h"
#include "stringzilla/utf8_iterate/v128relaxed.h"
#include "stringzilla/utf8_iterate/rvv.h"
#include "stringzilla/utf8_iterate/lasx.h"
#include "stringzilla/utf8_iterate/powervsx.h"

#pragma region Dynamic Dispatch

#if !SZ_DYNAMIC_DISPATCH

SZ_DYNAMIC sz_size_t sz_utf8_count(sz_cptr_t text, sz_size_t length) {
#if SZ_USE_V128RELAXED
    return sz_utf8_count_v128relaxed(text, length);
#elif SZ_USE_V128
    return sz_utf8_count_v128(text, length);
#elif SZ_USE_RVV
    return sz_utf8_count_rvv(text, length);
#elif SZ_USE_LASX
    return sz_utf8_count_lasx(text, length);
#elif SZ_USE_POWERVSX
    return sz_utf8_count_powervsx(text, length);
#elif SZ_USE_ICELAKE
    return sz_utf8_count_icelake(text, length);
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
#if SZ_USE_V128RELAXED
    return sz_utf8_find_nth_v128relaxed(text, length, n);
#elif SZ_USE_V128
    return sz_utf8_find_nth_v128(text, length, n);
#elif SZ_USE_RVV
    return sz_utf8_find_nth_rvv(text, length, n);
#elif SZ_USE_LASX
    return sz_utf8_find_nth_lasx(text, length, n);
#elif SZ_USE_POWERVSX
    return sz_utf8_find_nth_powervsx(text, length, n);
#elif SZ_USE_ICELAKE
    return sz_utf8_find_nth_icelake(text, length, n);
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

SZ_DYNAMIC sz_cptr_t sz_utf8_unpack_chunk(sz_cptr_t text, sz_size_t length, sz_rune_t *runes, sz_size_t runes_capacity,
                                          sz_size_t *runes_unpacked) {
#if SZ_USE_V128
    return sz_utf8_unpack_chunk_v128(text, length, runes, runes_capacity, runes_unpacked);
#elif SZ_USE_RVV
    return sz_utf8_unpack_chunk_rvv(text, length, runes, runes_capacity, runes_unpacked);
#elif SZ_USE_LASX
    return sz_utf8_unpack_chunk_lasx(text, length, runes, runes_capacity, runes_unpacked);
#elif SZ_USE_POWERVSX
    return sz_utf8_unpack_chunk_powervsx(text, length, runes, runes_capacity, runes_unpacked);
#elif SZ_USE_ICELAKE
    return sz_utf8_unpack_chunk_icelake(text, length, runes, runes_capacity, runes_unpacked);
#elif SZ_USE_NEON
    return sz_utf8_unpack_chunk_neon(text, length, runes, runes_capacity, runes_unpacked);
#else
    return sz_utf8_unpack_chunk_serial(text, length, runes, runes_capacity, runes_unpacked);
#endif
}

SZ_DYNAMIC sz_cptr_t sz_utf8_find_newline(sz_cptr_t text, sz_size_t length, sz_size_t *matched_length) {
#if SZ_USE_V128RELAXED
    return sz_utf8_find_newline_v128relaxed(text, length, matched_length);
#elif SZ_USE_V128
    return sz_utf8_find_newline_v128(text, length, matched_length);
#elif SZ_USE_RVV
    return sz_utf8_find_newline_rvv(text, length, matched_length);
#elif SZ_USE_LASX
    return sz_utf8_find_newline_lasx(text, length, matched_length);
#elif SZ_USE_POWERVSX
    return sz_utf8_find_newline_powervsx(text, length, matched_length);
#elif SZ_USE_ICELAKE
    return sz_utf8_find_newline_icelake(text, length, matched_length);
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
#if SZ_USE_V128RELAXED
    return sz_utf8_find_whitespace_v128relaxed(text, length, matched_length);
#elif SZ_USE_V128
    return sz_utf8_find_whitespace_v128(text, length, matched_length);
#elif SZ_USE_RVV
    return sz_utf8_find_whitespace_rvv(text, length, matched_length);
#elif SZ_USE_LASX
    return sz_utf8_find_whitespace_lasx(text, length, matched_length);
#elif SZ_USE_POWERVSX
    return sz_utf8_find_whitespace_powervsx(text, length, matched_length);
#elif SZ_USE_ICELAKE
    return sz_utf8_find_whitespace_icelake(text, length, matched_length);
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

SZ_DYNAMIC sz_size_t sz_utf8_word_find_boundaries(sz_cptr_t text, sz_size_t length, sz_size_t *word_starts,
                                                  sz_size_t *word_lengths, sz_size_t words_capacity,
                                                  sz_size_t *bytes_consumed) {
#if SZ_USE_ICELAKE
    return sz_utf8_word_find_boundaries_icelake(text, length, word_starts, word_lengths, words_capacity,
                                                bytes_consumed);
#elif SZ_USE_HASWELL
    return sz_utf8_word_find_boundaries_haswell(text, length, word_starts, word_lengths, words_capacity,
                                                bytes_consumed);
#elif SZ_USE_SVE2
    return sz_utf8_word_find_boundaries_sve2(text, length, word_starts, word_lengths, words_capacity, bytes_consumed);
#elif SZ_USE_NEON
    return sz_utf8_word_find_boundaries_neon(text, length, word_starts, word_lengths, words_capacity, bytes_consumed);
#elif SZ_USE_V128
    return sz_utf8_word_find_boundaries_v128(text, length, word_starts, word_lengths, words_capacity, bytes_consumed);
#elif SZ_USE_RVV
    return sz_utf8_word_find_boundaries_rvv(text, length, word_starts, word_lengths, words_capacity, bytes_consumed);
#elif SZ_USE_LASX
    return sz_utf8_word_find_boundaries_lasx(text, length, word_starts, word_lengths, words_capacity, bytes_consumed);
#elif SZ_USE_POWERVSX
    return sz_utf8_word_find_boundaries_powervsx(text, length, word_starts, word_lengths, words_capacity,
                                                 bytes_consumed);
#else
    return sz_utf8_word_find_boundaries_serial(text, length, word_starts, word_lengths, words_capacity, bytes_consumed);
#endif
}

SZ_DYNAMIC sz_size_t sz_utf8_word_rfind_boundaries(sz_cptr_t text, sz_size_t length, sz_size_t *word_starts,
                                                   sz_size_t *word_lengths, sz_size_t words_capacity,
                                                   sz_size_t *bytes_consumed) {
#if SZ_USE_ICELAKE
    return sz_utf8_word_rfind_boundaries_icelake(text, length, word_starts, word_lengths, words_capacity,
                                                 bytes_consumed);
#elif SZ_USE_HASWELL
    return sz_utf8_word_rfind_boundaries_haswell(text, length, word_starts, word_lengths, words_capacity,
                                                 bytes_consumed);
#elif SZ_USE_SVE2
    return sz_utf8_word_rfind_boundaries_sve2(text, length, word_starts, word_lengths, words_capacity, bytes_consumed);
#elif SZ_USE_NEON
    return sz_utf8_word_rfind_boundaries_neon(text, length, word_starts, word_lengths, words_capacity, bytes_consumed);
#elif SZ_USE_V128
    return sz_utf8_word_rfind_boundaries_v128(text, length, word_starts, word_lengths, words_capacity, bytes_consumed);
#elif SZ_USE_RVV
    return sz_utf8_word_rfind_boundaries_rvv(text, length, word_starts, word_lengths, words_capacity, bytes_consumed);
#elif SZ_USE_LASX
    return sz_utf8_word_rfind_boundaries_lasx(text, length, word_starts, word_lengths, words_capacity, bytes_consumed);
#elif SZ_USE_POWERVSX
    return sz_utf8_word_rfind_boundaries_powervsx(text, length, word_starts, word_lengths, words_capacity,
                                                  bytes_consumed);
#else
    return sz_utf8_word_rfind_boundaries_serial(text, length, word_starts, word_lengths, words_capacity,
                                                bytes_consumed);
#endif
}

#endif // !SZ_DYNAMIC_DISPATCH

#pragma endregion // Dynamic Dispatch

#ifdef __cplusplus
}
#endif

#endif // STRINGZILLA_UTF8_ITERATE_H_
