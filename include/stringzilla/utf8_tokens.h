/**
 *  @file include/stringzilla/utf8_tokens.h
 *  @author Ash Vardanian
 *  @date November 18, 2025
 *  @brief Hardware-accelerated UTF-8 newline, whitespace, and general delimiter scanning.
 */
#ifndef STRINGZILLA_UTF8_TOKENS_H_
#define STRINGZILLA_UTF8_TOKENS_H_

#include "stringzilla/types.h"
#include "stringzilla/capabilities.h" // `sz_capability_t`

#ifdef __cplusplus
extern "C" {
#endif

#pragma region Core API

/**
 *  @brief Enumerates every UTF-8 newline delimiter in a string.
 *
 *  Enumerates every newline delimiter in a single sweep: writes the byte offset and byte length of
 *  each match into the parallel @p match_offsets and @p match_lengths arrays, a @c "\r\n" CRLF
 *  being one match of length 2, and their number into @p matches_count. A caller resuming from
 *  `text + *bytes_consumed` after a full output obtains the identical remainder.
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
 *  U+001C, U+001D, U+001E (FILE/GROUP/RECORD SEPARATOR) are not included. These are data structure
 *  delimiters used in formats like USV (Unicode Separated Values), not line breaks. Use
 *  @c sz_find_byte_best if you need to find these separators.
 *
 *  @param[in] text String to be scanned.
 *  @param[in] length Number of bytes in the string.
 *  @param[out] match_offsets Delimiter start offsets, at least @p matches_capacity entries.
 *  @param[out] match_lengths Delimiter byte lengths, at least @p matches_capacity entries.
 *  @param[in] matches_capacity Capacity of the output arrays.
 *  @param[out] matches_count Number of delimiters written to the output arrays.
 *  @param[out] bytes_consumed Optional resume offset: once @p matches_count reaches
 *      @p matches_capacity, the end of the last match, or zero with no capacity; else @p length.
 *  @param[in] capabilities One device's capabilities, like @c sz_cpu_capabilities_enabled reports.
 *  @param[in] stream Null on the CPU, or the GPU stream of that device to queue on.
 *  @return @c sz_success_k, or @c sz_missing_kernel_k when no capability in @p capabilities has it.
 */
STRINGZILLA_API sz_status_t sz_utf8_newlines_best(                                  //
    sz_cptr_t text, sz_size_t length,                                               //
    sz_size_t *match_offsets, sz_size_t *match_lengths, sz_size_t matches_capacity, //
    sz_size_t *matches_count, sz_size_t *bytes_consumed,                            //
    sz_capability_t capabilities, void *stream);

/**
 *  @brief Enumerates every UTF-8 whitespace delimiter in a string.
 *
 *  Enumerates every whitespace delimiter in a single sweep, with the same contract as
 *  @c sz_utf8_newlines_best: writes the byte offset and byte length of each match into the parallel
 *  @p match_offsets and @p match_lengths arrays, their number into @p matches_count, and sets
 *  @p bytes_consumed to the resume offset. Each whitespace codepoint is one match: CR and LF are
 *  independent length-1 matches, as there is no CRLF merging in the whitespace set.
 *
 *  Implements the Unicode White_Space property, 25 characters in total. Per the Unicode standard,
 *  whitespace includes all newline characters plus horizontal spaces. Matches the behavior of ICU's
 *  `u_isspace()` and Python's `str.isspace()`.
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
 *  Some implementations treat more codepoints as whitespace, but these are not included:
 *
 *  - U+001C, U+001D, U+001E, U+001F (FILE/GROUP/RECORD/UNIT SEPARATOR) are data structure
 *    delimiters for formats like USV (Unicode Separated Values). Only Java's
 *    `Character.isWhitespace()` includes them; Unicode, ICU, and Python do not.
 *  - U+200B, U+200C, U+200D (ZERO WIDTH SPACE/NON-JOINER/JOINER) are Format characters, not
 *    whitespace. They have no width and affect rendering rather than spacing.
 *
 *  @param[in] text String to be scanned.
 *  @param[in] length Number of bytes in the string.
 *  @param[out] match_offsets Delimiter start offsets, at least @p matches_capacity entries.
 *  @param[out] match_lengths Delimiter byte lengths, at least @p matches_capacity entries.
 *  @param[in] matches_capacity Capacity of the output arrays.
 *  @param[out] matches_count Number of delimiters written to the output arrays.
 *  @param[out] bytes_consumed Optional resume offset: once @p matches_count reaches
 *      @p matches_capacity, the end of the last match, or zero with no capacity; else @p length.
 *  @param[in] capabilities One device's capabilities, like @c sz_cpu_capabilities_enabled reports.
 *  @param[in] stream Null on the CPU, or the GPU stream of that device to queue on.
 *  @return @c sz_success_k, or @c sz_missing_kernel_k when no capability in @p capabilities has it.
 */
STRINGZILLA_API sz_status_t sz_utf8_whitespaces_best(                               //
    sz_cptr_t text, sz_size_t length,                                               //
    sz_size_t *match_offsets, sz_size_t *match_lengths, sz_size_t matches_capacity, //
    sz_size_t *matches_count, sz_size_t *bytes_consumed,                            //
    sz_capability_t capabilities, void *stream);

/**
 *  @brief Enumerates every UTF-8 delimiter codepoint (punctuation, symbol, separator, whitespace).
 *
 *  The general superset of @c sz_utf8_newlines_best and @c sz_utf8_whitespaces_best: every
 *  codepoint whose Unicode general category is a punctuation (P*), symbol (S*), or separator (Z*)
 *  is a delimiter. Shares the contract of @c sz_utf8_newlines_best: writes the byte offset and byte
 *  length of each match into the parallel @p match_offsets and @p match_lengths arrays, their
 *  number into @p matches_count, and sets @p bytes_consumed to the resume offset. A byte that does
 *  not begin a well-formed codepoint is skipped and never reported.
 *
 *  @param[in] text String to be scanned.
 *  @param[in] length Number of bytes in the string.
 *  @param[out] match_offsets Delimiter start offsets, at least @p matches_capacity entries.
 *  @param[out] match_lengths Delimiter byte lengths, at least @p matches_capacity entries.
 *  @param[in] matches_capacity Capacity of the output arrays.
 *  @param[out] matches_count Number of delimiters written to the output arrays.
 *  @param[out] bytes_consumed Optional resume offset: once @p matches_count reaches
 *      @p matches_capacity, the end of the last match, or zero with no capacity; else @p length.
 *  @param[in] capabilities One device's capabilities, like @c sz_cpu_capabilities_enabled reports.
 *  @param[in] stream Null on the CPU, or the GPU stream of that device to queue on.
 *  @return @c sz_success_k, or @c sz_missing_kernel_k when no capability in @p capabilities has it.
 */
STRINGZILLA_API sz_status_t sz_utf8_delimiters_best(                                //
    sz_cptr_t text, sz_size_t length,                                               //
    sz_size_t *match_offsets, sz_size_t *match_lengths, sz_size_t matches_capacity, //
    sz_size_t *matches_count, sz_size_t *bytes_consumed,                            //
    sz_capability_t capabilities, void *stream);

/**
 *  @brief Finds the UTF-8 tokens kernel of @p kind, from the best of @p capabilities.
 *  @param[out] kernel The kernel, or null when none of @p capabilities has it.
 *  @param[out] capability The capability the kernel belongs to, or zero.
 *  @return @c sz_success_k, @c sz_missing_kernel_k, or @c sz_missing_library_k when header-only.
 */
STRINGZILLA_API sz_status_t sz_utf8_tokens_find_kernel(sz_kernel_kind_t kind, sz_capability_t capabilities,
                                                       sz_kernel_punned_t *kernel, sz_capability_t *capability);

#pragma endregion

#pragma region Platform Specific Backends

/** @copydoc sz_utf8_newlines_best */
STRINGZILLA_API sz_status_t sz_utf8_newlines_serial(                                //
    sz_cptr_t text, sz_size_t length,                                               //
    sz_size_t *match_offsets, sz_size_t *match_lengths, sz_size_t matches_capacity, //
    sz_size_t *matches_count, sz_size_t *bytes_consumed, void *stream);
/** @copydoc sz_utf8_whitespaces_best */
STRINGZILLA_API sz_status_t sz_utf8_whitespaces_serial(                             //
    sz_cptr_t text, sz_size_t length,                                               //
    sz_size_t *match_offsets, sz_size_t *match_lengths, sz_size_t matches_capacity, //
    sz_size_t *matches_count, sz_size_t *bytes_consumed, void *stream);
/** @copydoc sz_utf8_delimiters_best */
STRINGZILLA_API sz_status_t sz_utf8_delimiters_serial(                              //
    sz_cptr_t text, sz_size_t length,                                               //
    sz_size_t *match_offsets, sz_size_t *match_lengths, sz_size_t matches_capacity, //
    sz_size_t *matches_count, sz_size_t *bytes_consumed, void *stream);

#if STRINGZILLA_TARGET_HASWELL
/** @copydoc sz_utf8_newlines_best */
STRINGZILLA_API sz_status_t sz_utf8_newlines_haswell(                               //
    sz_cptr_t text, sz_size_t length,                                               //
    sz_size_t *match_offsets, sz_size_t *match_lengths, sz_size_t matches_capacity, //
    sz_size_t *matches_count, sz_size_t *bytes_consumed, void *stream);
/** @copydoc sz_utf8_whitespaces_best */
STRINGZILLA_API sz_status_t sz_utf8_whitespaces_haswell(                            //
    sz_cptr_t text, sz_size_t length,                                               //
    sz_size_t *match_offsets, sz_size_t *match_lengths, sz_size_t matches_capacity, //
    sz_size_t *matches_count, sz_size_t *bytes_consumed, void *stream);
/** @copydoc sz_utf8_delimiters_best */
STRINGZILLA_API sz_status_t sz_utf8_delimiters_haswell(                             //
    sz_cptr_t text, sz_size_t length,                                               //
    sz_size_t *match_offsets, sz_size_t *match_lengths, sz_size_t matches_capacity, //
    sz_size_t *matches_count, sz_size_t *bytes_consumed, void *stream);
#endif

#if STRINGZILLA_TARGET_ICELAKE
/** @copydoc sz_utf8_newlines_best */
STRINGZILLA_API sz_status_t sz_utf8_newlines_icelake(                               //
    sz_cptr_t text, sz_size_t length,                                               //
    sz_size_t *match_offsets, sz_size_t *match_lengths, sz_size_t matches_capacity, //
    sz_size_t *matches_count, sz_size_t *bytes_consumed, void *stream);
/** @copydoc sz_utf8_whitespaces_best */
STRINGZILLA_API sz_status_t sz_utf8_whitespaces_icelake(                            //
    sz_cptr_t text, sz_size_t length,                                               //
    sz_size_t *match_offsets, sz_size_t *match_lengths, sz_size_t matches_capacity, //
    sz_size_t *matches_count, sz_size_t *bytes_consumed, void *stream);
/** @copydoc sz_utf8_delimiters_best */
STRINGZILLA_API sz_status_t sz_utf8_delimiters_icelake(                             //
    sz_cptr_t text, sz_size_t length,                                               //
    sz_size_t *match_offsets, sz_size_t *match_lengths, sz_size_t matches_capacity, //
    sz_size_t *matches_count, sz_size_t *bytes_consumed, void *stream);
#endif

#if STRINGZILLA_TARGET_NEON
/** @copydoc sz_utf8_newlines_best */
STRINGZILLA_API sz_status_t sz_utf8_newlines_neon(                                  //
    sz_cptr_t text, sz_size_t length,                                               //
    sz_size_t *match_offsets, sz_size_t *match_lengths, sz_size_t matches_capacity, //
    sz_size_t *matches_count, sz_size_t *bytes_consumed, void *stream);
/** @copydoc sz_utf8_whitespaces_best */
STRINGZILLA_API sz_status_t sz_utf8_whitespaces_neon(                               //
    sz_cptr_t text, sz_size_t length,                                               //
    sz_size_t *match_offsets, sz_size_t *match_lengths, sz_size_t matches_capacity, //
    sz_size_t *matches_count, sz_size_t *bytes_consumed, void *stream);
/** @copydoc sz_utf8_delimiters_best */
STRINGZILLA_API sz_status_t sz_utf8_delimiters_neon(                                //
    sz_cptr_t text, sz_size_t length,                                               //
    sz_size_t *match_offsets, sz_size_t *match_lengths, sz_size_t matches_capacity, //
    sz_size_t *matches_count, sz_size_t *bytes_consumed, void *stream);
#endif

#if STRINGZILLA_TARGET_SVE2
/** @copydoc sz_utf8_newlines_best */
STRINGZILLA_API sz_status_t sz_utf8_newlines_sve2(                                  //
    sz_cptr_t text, sz_size_t length,                                               //
    sz_size_t *match_offsets, sz_size_t *match_lengths, sz_size_t matches_capacity, //
    sz_size_t *matches_count, sz_size_t *bytes_consumed, void *stream);
/** @copydoc sz_utf8_whitespaces_best */
STRINGZILLA_API sz_status_t sz_utf8_whitespaces_sve2(                               //
    sz_cptr_t text, sz_size_t length,                                               //
    sz_size_t *match_offsets, sz_size_t *match_lengths, sz_size_t matches_capacity, //
    sz_size_t *matches_count, sz_size_t *bytes_consumed, void *stream);
/** @copydoc sz_utf8_delimiters_best */
STRINGZILLA_API sz_status_t sz_utf8_delimiters_sve2(                                //
    sz_cptr_t text, sz_size_t length,                                               //
    sz_size_t *match_offsets, sz_size_t *match_lengths, sz_size_t matches_capacity, //
    sz_size_t *matches_count, sz_size_t *bytes_consumed, void *stream);
#endif

#if STRINGZILLA_TARGET_RVV
/** @copydoc sz_utf8_newlines_best */
STRINGZILLA_API sz_status_t sz_utf8_newlines_rvv(                                   //
    sz_cptr_t text, sz_size_t length,                                               //
    sz_size_t *match_offsets, sz_size_t *match_lengths, sz_size_t matches_capacity, //
    sz_size_t *matches_count, sz_size_t *bytes_consumed, void *stream);
/** @copydoc sz_utf8_whitespaces_best */
STRINGZILLA_API sz_status_t sz_utf8_whitespaces_rvv(                                //
    sz_cptr_t text, sz_size_t length,                                               //
    sz_size_t *match_offsets, sz_size_t *match_lengths, sz_size_t matches_capacity, //
    sz_size_t *matches_count, sz_size_t *bytes_consumed, void *stream);
#endif

#if STRINGZILLA_TARGET_V128
/** @copydoc sz_utf8_newlines_best */
STRINGZILLA_API sz_status_t sz_utf8_newlines_v128(                                  //
    sz_cptr_t text, sz_size_t length,                                               //
    sz_size_t *match_offsets, sz_size_t *match_lengths, sz_size_t matches_capacity, //
    sz_size_t *matches_count, sz_size_t *bytes_consumed, void *stream);
/** @copydoc sz_utf8_whitespaces_best */
STRINGZILLA_API sz_status_t sz_utf8_whitespaces_v128(                               //
    sz_cptr_t text, sz_size_t length,                                               //
    sz_size_t *match_offsets, sz_size_t *match_lengths, sz_size_t matches_capacity, //
    sz_size_t *matches_count, sz_size_t *bytes_consumed, void *stream);
#endif

#if STRINGZILLA_TARGET_LOONGSONASX
/** @copydoc sz_utf8_newlines_best */
STRINGZILLA_API sz_status_t sz_utf8_newlines_loongsonasx(                           //
    sz_cptr_t text, sz_size_t length,                                               //
    sz_size_t *match_offsets, sz_size_t *match_lengths, sz_size_t matches_capacity, //
    sz_size_t *matches_count, sz_size_t *bytes_consumed, void *stream);
/** @copydoc sz_utf8_whitespaces_best */
STRINGZILLA_API sz_status_t sz_utf8_whitespaces_loongsonasx(                        //
    sz_cptr_t text, sz_size_t length,                                               //
    sz_size_t *match_offsets, sz_size_t *match_lengths, sz_size_t matches_capacity, //
    sz_size_t *matches_count, sz_size_t *bytes_consumed, void *stream);
#endif

#if STRINGZILLA_TARGET_POWERVSX
/** @copydoc sz_utf8_newlines_best */
STRINGZILLA_API sz_status_t sz_utf8_newlines_powervsx(                              //
    sz_cptr_t text, sz_size_t length,                                               //
    sz_size_t *match_offsets, sz_size_t *match_lengths, sz_size_t matches_capacity, //
    sz_size_t *matches_count, sz_size_t *bytes_consumed, void *stream);
/** @copydoc sz_utf8_whitespaces_best */
STRINGZILLA_API sz_status_t sz_utf8_whitespaces_powervsx(                           //
    sz_cptr_t text, sz_size_t length,                                               //
    sz_size_t *match_offsets, sz_size_t *match_lengths, sz_size_t matches_capacity, //
    sz_size_t *matches_count, sz_size_t *bytes_consumed, void *stream);
#endif

#pragma endregion

/*  Header-only builds define each kernel inline from its tier header, while the library defines
 *  every kernel once, in its capability's unit under `c/target/`. */
#include "stringzilla/utf8_tokens/serial.h"
#if STRINGZILLA_HEADER_ONLY
#include "stringzilla/utf8_tokens/icelake.h"
#include "stringzilla/utf8_tokens/haswell.h"
#include "stringzilla/utf8_tokens/neon.h"
#include "stringzilla/utf8_tokens/sve2.h"
#include "stringzilla/utf8_tokens/v128.h"
#include "stringzilla/utf8_tokens/rvv.h"
#include "stringzilla/utf8_tokens/loongsonasx.h"
#include "stringzilla/utf8_tokens/powervsx.h"
#endif // STRINGZILLA_HEADER_ONLY

#if STRINGZILLA_HEADER_ONLY

STRINGZILLA_API sz_status_t sz_utf8_newlines_best(                                  //
    sz_cptr_t text, sz_size_t length,                                               //
    sz_size_t *match_offsets, sz_size_t *match_lengths, sz_size_t matches_capacity, //
    sz_size_t *matches_count, sz_size_t *bytes_consumed,                            //
    sz_capability_t capabilities, void *stream) {
    sz_unused_(text), sz_unused_(length), sz_unused_(match_offsets), sz_unused_(match_lengths),
        sz_unused_(matches_capacity), sz_unused_(matches_count), sz_unused_(bytes_consumed), sz_unused_(capabilities),
        sz_unused_(stream);
    return sz_missing_library_k;
}

STRINGZILLA_API sz_status_t sz_utf8_whitespaces_best(                               //
    sz_cptr_t text, sz_size_t length,                                               //
    sz_size_t *match_offsets, sz_size_t *match_lengths, sz_size_t matches_capacity, //
    sz_size_t *matches_count, sz_size_t *bytes_consumed,                            //
    sz_capability_t capabilities, void *stream) {
    sz_unused_(text), sz_unused_(length), sz_unused_(match_offsets), sz_unused_(match_lengths),
        sz_unused_(matches_capacity), sz_unused_(matches_count), sz_unused_(bytes_consumed), sz_unused_(capabilities),
        sz_unused_(stream);
    return sz_missing_library_k;
}

STRINGZILLA_API sz_status_t sz_utf8_delimiters_best(                                //
    sz_cptr_t text, sz_size_t length,                                               //
    sz_size_t *match_offsets, sz_size_t *match_lengths, sz_size_t matches_capacity, //
    sz_size_t *matches_count, sz_size_t *bytes_consumed,                            //
    sz_capability_t capabilities, void *stream) {
    sz_unused_(text), sz_unused_(length), sz_unused_(match_offsets), sz_unused_(match_lengths),
        sz_unused_(matches_capacity), sz_unused_(matches_count), sz_unused_(bytes_consumed), sz_unused_(capabilities),
        sz_unused_(stream);
    return sz_missing_library_k;
}

STRINGZILLA_API sz_status_t sz_utf8_tokens_find_kernel(sz_kernel_kind_t kind, sz_capability_t capabilities,
                                                       sz_kernel_punned_t *kernel, sz_capability_t *capability) {
    sz_unused_(kind), sz_unused_(capabilities);
    *kernel = STRINGZILLA_NULL, *capability = 0;
    return sz_missing_library_k;
}

#endif // STRINGZILLA_HEADER_ONLY

#ifdef __cplusplus
}
#endif

#endif // STRINGZILLA_UTF8_TOKENS_H_
