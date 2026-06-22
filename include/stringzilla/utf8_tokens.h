/**
 *  @brief Hardware-accelerated UTF-8 newline and whitespace delimiter scanning.
 *  @file utf8_tokens.h
 *  @author Ash Vardanian
 */
#ifndef STRINGZILLA_UTF8_TOKENS_H_
#define STRINGZILLA_UTF8_TOKENS_H_

#include "stringzilla/types.h"

#ifdef __cplusplus
extern "C" {
#endif

#pragma region Core API

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
 *  Enumerates every newline delimiter in a single sweep, mirroring `sz_utf8_words`: writes the
 *  byte offset and byte length of each match into the parallel `match_offsets` / `match_lengths` arrays (a
 *  @c "\r\n" CRLF is one match of length 2). Returns the number of delimiters emitted. When the output fills
 *  before the input is exhausted, `*bytes_consumed` is set to the resume offset - always past the last emitted
 *  delimiter and at a byte that begins fresh content - so a caller resumes from `text + *bytes_consumed` and
 *  obtains the identical remainder.
 *
 *  @param text String to be scanned.
 *  @param length Number of bytes in the string.
 *  @param match_offsets Output array of delimiter start offsets (at least @p matches_capacity entries).
 *  @param match_lengths Output array of delimiter byte lengths (at least @p matches_capacity entries).
 *  @param matches_capacity Capacity of the output arrays.
 *  @param bytes_consumed Output: byte offset to resume scanning from.
 *  @return Number of delimiters written to the output arrays.
 */
SZ_DYNAMIC sz_size_t sz_utf8_newlines(sz_cptr_t text, sz_size_t length, sz_size_t *match_offsets,
                                      sz_size_t *match_lengths, sz_size_t matches_capacity, sz_size_t *bytes_consumed);

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
 *  Enumerates every whitespace delimiter in a single sweep, with the same contract as `sz_utf8_newlines`:
 *  writes the byte offset and byte length of each match into the parallel `match_offsets` / `match_lengths`
 *  arrays, returns the count, and sets `*bytes_consumed` to the resume offset. Each whitespace codepoint is one
 *  match (CR and LF are independent length-1 matches; there is no CRLF merging in the whitespace set).
 *
 *  @param text String to be scanned.
 *  @param length Number of bytes in the string.
 *  @param match_offsets Output array of delimiter start offsets (at least @p matches_capacity entries).
 *  @param match_lengths Output array of delimiter byte lengths (at least @p matches_capacity entries).
 *  @param matches_capacity Capacity of the output arrays.
 *  @param bytes_consumed Output: byte offset to resume scanning from.
 *  @return Number of delimiters written to the output arrays.
 */
SZ_DYNAMIC sz_size_t sz_utf8_whitespaces(sz_cptr_t text, sz_size_t length, sz_size_t *match_offsets,
                                         sz_size_t *match_lengths, sz_size_t matches_capacity,
                                         sz_size_t *bytes_consumed);

#pragma endregion

#pragma region Platform Specific Backends

/** @copydoc sz_utf8_newlines */
SZ_PUBLIC sz_size_t sz_utf8_newlines_serial(sz_cptr_t text, sz_size_t length, sz_size_t *match_offsets,
                                            sz_size_t *match_lengths, sz_size_t matches_capacity,
                                            sz_size_t *bytes_consumed);
/** @copydoc sz_utf8_whitespaces */
SZ_PUBLIC sz_size_t sz_utf8_whitespaces_serial(sz_cptr_t text, sz_size_t length, sz_size_t *match_offsets,
                                               sz_size_t *match_lengths, sz_size_t matches_capacity,
                                               sz_size_t *bytes_consumed);

#if SZ_USE_HASWELL
/** @copydoc sz_utf8_newlines */
SZ_PUBLIC sz_size_t sz_utf8_newlines_haswell(sz_cptr_t text, sz_size_t length, sz_size_t *match_offsets,
                                             sz_size_t *match_lengths, sz_size_t matches_capacity,
                                             sz_size_t *bytes_consumed);
/** @copydoc sz_utf8_whitespaces */
SZ_PUBLIC sz_size_t sz_utf8_whitespaces_haswell(sz_cptr_t text, sz_size_t length, sz_size_t *match_offsets,
                                                sz_size_t *match_lengths, sz_size_t matches_capacity,
                                                sz_size_t *bytes_consumed);
#endif

#if SZ_USE_ICELAKE
/** @copydoc sz_utf8_newlines */
SZ_PUBLIC sz_size_t sz_utf8_newlines_icelake(sz_cptr_t text, sz_size_t length, sz_size_t *match_offsets,
                                             sz_size_t *match_lengths, sz_size_t matches_capacity,
                                             sz_size_t *bytes_consumed);
/** @copydoc sz_utf8_whitespaces */
SZ_PUBLIC sz_size_t sz_utf8_whitespaces_icelake(sz_cptr_t text, sz_size_t length, sz_size_t *match_offsets,
                                                sz_size_t *match_lengths, sz_size_t matches_capacity,
                                                sz_size_t *bytes_consumed);
#endif

#if SZ_USE_NEON
/** @copydoc sz_utf8_newlines */
SZ_PUBLIC sz_size_t sz_utf8_newlines_neon(sz_cptr_t text, sz_size_t length, sz_size_t *match_offsets,
                                          sz_size_t *match_lengths, sz_size_t matches_capacity,
                                          sz_size_t *bytes_consumed);
/** @copydoc sz_utf8_whitespaces */
SZ_PUBLIC sz_size_t sz_utf8_whitespaces_neon(sz_cptr_t text, sz_size_t length, sz_size_t *match_offsets,
                                             sz_size_t *match_lengths, sz_size_t matches_capacity,
                                             sz_size_t *bytes_consumed);
#endif

#if SZ_USE_SVE2
/** @copydoc sz_utf8_newlines */
SZ_PUBLIC sz_size_t sz_utf8_newlines_sve2(sz_cptr_t text, sz_size_t length, sz_size_t *match_offsets,
                                          sz_size_t *match_lengths, sz_size_t matches_capacity,
                                          sz_size_t *bytes_consumed);
/** @copydoc sz_utf8_whitespaces */
SZ_PUBLIC sz_size_t sz_utf8_whitespaces_sve2(sz_cptr_t text, sz_size_t length, sz_size_t *match_offsets,
                                             sz_size_t *match_lengths, sz_size_t matches_capacity,
                                             sz_size_t *bytes_consumed);
#endif

#if SZ_USE_V128
/** @copydoc sz_utf8_newlines */
SZ_PUBLIC sz_size_t sz_utf8_newlines_v128(sz_cptr_t text, sz_size_t length, sz_size_t *match_offsets,
                                          sz_size_t *match_lengths, sz_size_t matches_capacity,
                                          sz_size_t *bytes_consumed);
/** @copydoc sz_utf8_whitespaces */
SZ_PUBLIC sz_size_t sz_utf8_whitespaces_v128(sz_cptr_t text, sz_size_t length, sz_size_t *match_offsets,
                                             sz_size_t *match_lengths, sz_size_t matches_capacity,
                                             sz_size_t *bytes_consumed);
#endif

#if SZ_USE_RVV
/** @copydoc sz_utf8_newlines */
SZ_PUBLIC sz_size_t sz_utf8_newlines_rvv(sz_cptr_t text, sz_size_t length, sz_size_t *match_offsets,
                                         sz_size_t *match_lengths, sz_size_t matches_capacity,
                                         sz_size_t *bytes_consumed);
/** @copydoc sz_utf8_whitespaces */
SZ_PUBLIC sz_size_t sz_utf8_whitespaces_rvv(sz_cptr_t text, sz_size_t length, sz_size_t *match_offsets,
                                            sz_size_t *match_lengths, sz_size_t matches_capacity,
                                            sz_size_t *bytes_consumed);
#endif

#if SZ_USE_LASX
/** @copydoc sz_utf8_newlines */
SZ_PUBLIC sz_size_t sz_utf8_newlines_lasx(sz_cptr_t text, sz_size_t length, sz_size_t *match_offsets,
                                          sz_size_t *match_lengths, sz_size_t matches_capacity,
                                          sz_size_t *bytes_consumed);
/** @copydoc sz_utf8_whitespaces */
SZ_PUBLIC sz_size_t sz_utf8_whitespaces_lasx(sz_cptr_t text, sz_size_t length, sz_size_t *match_offsets,
                                             sz_size_t *match_lengths, sz_size_t matches_capacity,
                                             sz_size_t *bytes_consumed);
#endif

#if SZ_USE_POWERVSX
/** @copydoc sz_utf8_newlines */
SZ_PUBLIC sz_size_t sz_utf8_newlines_powervsx(sz_cptr_t text, sz_size_t length, sz_size_t *match_offsets,
                                              sz_size_t *match_lengths, sz_size_t matches_capacity,
                                              sz_size_t *bytes_consumed);
/** @copydoc sz_utf8_whitespaces */
SZ_PUBLIC sz_size_t sz_utf8_whitespaces_powervsx(sz_cptr_t text, sz_size_t length, sz_size_t *match_offsets,
                                                 sz_size_t *match_lengths, sz_size_t matches_capacity,
                                                 sz_size_t *bytes_consumed);
#endif

#pragma endregion

/*  Implementation Section - each ISA backend lives in its own header, included serial-first. */
#include "stringzilla/utf8_tokens/serial.h"
#include "stringzilla/utf8_tokens/icelake.h"
#include "stringzilla/utf8_tokens/haswell.h"
#include "stringzilla/utf8_tokens/neon.h"
#include "stringzilla/utf8_tokens/sve2.h"
#include "stringzilla/utf8_tokens/v128.h"
#include "stringzilla/utf8_tokens/rvv.h"
#include "stringzilla/utf8_tokens/lasx.h"
#include "stringzilla/utf8_tokens/powervsx.h"

#pragma region Dynamic Dispatch

#if !SZ_DYNAMIC_DISPATCH

SZ_DYNAMIC sz_size_t sz_utf8_newlines(sz_cptr_t text, sz_size_t length, sz_size_t *match_offsets,
                                      sz_size_t *match_lengths, sz_size_t matches_capacity, sz_size_t *bytes_consumed) {
#if SZ_USE_V128
    return sz_utf8_newlines_v128(text, length, match_offsets, match_lengths, matches_capacity, bytes_consumed);
#elif SZ_USE_RVV
    return sz_utf8_newlines_rvv(text, length, match_offsets, match_lengths, matches_capacity, bytes_consumed);
#elif SZ_USE_LASX
    return sz_utf8_newlines_lasx(text, length, match_offsets, match_lengths, matches_capacity, bytes_consumed);
#elif SZ_USE_POWERVSX
    return sz_utf8_newlines_powervsx(text, length, match_offsets, match_lengths, matches_capacity, bytes_consumed);
#elif SZ_USE_ICELAKE
    return sz_utf8_newlines_icelake(text, length, match_offsets, match_lengths, matches_capacity, bytes_consumed);
#elif SZ_USE_HASWELL
    return sz_utf8_newlines_haswell(text, length, match_offsets, match_lengths, matches_capacity, bytes_consumed);
#elif SZ_USE_SVE2 && SZ_ENFORCE_SVE_OVER_NEON
    return sz_utf8_newlines_sve2(text, length, match_offsets, match_lengths, matches_capacity, bytes_consumed);
#elif SZ_USE_NEON
    return sz_utf8_newlines_neon(text, length, match_offsets, match_lengths, matches_capacity, bytes_consumed);
#else
    return sz_utf8_newlines_serial(text, length, match_offsets, match_lengths, matches_capacity, bytes_consumed);
#endif
}

SZ_DYNAMIC sz_size_t sz_utf8_whitespaces(sz_cptr_t text, sz_size_t length, sz_size_t *match_offsets,
                                         sz_size_t *match_lengths, sz_size_t matches_capacity,
                                         sz_size_t *bytes_consumed) {
#if SZ_USE_V128
    return sz_utf8_whitespaces_v128(text, length, match_offsets, match_lengths, matches_capacity, bytes_consumed);
#elif SZ_USE_RVV
    return sz_utf8_whitespaces_rvv(text, length, match_offsets, match_lengths, matches_capacity, bytes_consumed);
#elif SZ_USE_LASX
    return sz_utf8_whitespaces_lasx(text, length, match_offsets, match_lengths, matches_capacity, bytes_consumed);
#elif SZ_USE_POWERVSX
    return sz_utf8_whitespaces_powervsx(text, length, match_offsets, match_lengths, matches_capacity, bytes_consumed);
#elif SZ_USE_ICELAKE
    return sz_utf8_whitespaces_icelake(text, length, match_offsets, match_lengths, matches_capacity, bytes_consumed);
#elif SZ_USE_HASWELL
    return sz_utf8_whitespaces_haswell(text, length, match_offsets, match_lengths, matches_capacity, bytes_consumed);
#elif SZ_USE_SVE2 && SZ_ENFORCE_SVE_OVER_NEON
    return sz_utf8_whitespaces_sve2(text, length, match_offsets, match_lengths, matches_capacity, bytes_consumed);
#elif SZ_USE_NEON
    return sz_utf8_whitespaces_neon(text, length, match_offsets, match_lengths, matches_capacity, bytes_consumed);
#else
    return sz_utf8_whitespaces_serial(text, length, match_offsets, match_lengths, matches_capacity, bytes_consumed);
#endif
}

#endif // !SZ_DYNAMIC_DISPATCH

#pragma endregion

#ifdef __cplusplus
}
#endif

#endif // STRINGZILLA_UTF8_TOKENS_H_
