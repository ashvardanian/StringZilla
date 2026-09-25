/**
 *  @file include/stringzilla/utf8_wordbreaks.h
 *  @author Ash Vardanian
 *  @date November 30, 2025
 *  @brief Hardware-accelerated UAX-29 word boundary segmentation.
 */
#ifndef STRINGZILLA_UTF8_WORDBREAKS_H_
#define STRINGZILLA_UTF8_WORDBREAKS_H_

#include "stringzilla/types.h"

#ifdef __cplusplus
extern "C" {
#endif

#pragma region Core API

/**
 *  @brief Segment UTF-8 text into UAX-29 words in a single pass (dispatch function).
 *
 *  Walks the whole input left-to-right and writes one entry per word into two parallel output
 *  arrays: `word_starts[i]` is the byte offset of the i-th word and `word_lengths[i]` its byte
 *  length. Words are the spans between consecutive TR29 boundaries, so a single call segments the
 *  entire input without the caller having to loop and restart a scan for every word.
 *
 *  @param[in] text UTF-8 encoded text.
 *  @param[in] length Byte length of @p text.
 *  @param[out] word_starts Word byte offsets, at least @p words_capacity entries.
 *  @param[out] word_lengths Word byte lengths, at least @p words_capacity entries.
 *  @param[in] words_capacity Capacity of the output arrays, in entries.
 *  @param[out] bytes_consumed Optional byte offset up to which the input was segmented: @p length
 *      when everything fit, else the start of the first word that did not fit (a TR29 boundary), so
 *      the caller may resume from `text + *bytes_consumed`.
 *  @return Number of words written (at most @p words_capacity).
 *
 *  @note No zero-length words are emitted; @p length == 0 returns 0.
 */
STRINGZILLA_API_RUNTIME sz_size_t sz_utf8_wordbreaks( //
    sz_cptr_t text, sz_size_t length,                 //
    sz_size_t *word_starts, sz_size_t *word_lengths,  //
    sz_size_t words_capacity, sz_size_t *bytes_consumed);

/**
 *  @brief Get the Unicode TR29 Word_Break property for a codepoint.
 *
 *  Returns one of the 16 Word_Break property values, @c sz_utf8_word_break_other_k through
 *  @c sz_utf8_word_break_mid_quotes_k: the foundation of TR29-compliant word boundary detection.
 *
 *  @param[in] rune The Unicode codepoint to classify.
 *  @return The Word_Break property value (0-15).
 *
 *  @see Unicode Text Segmentation: https://www.unicode.org/reports/tr29/
 */
STRINGZILLA_API_COMPTIME sz_u8_t sz_rune_word_break_property(sz_rune_t rune);

/**
 *  @brief Check if a codepoint is a "word character" (has word-forming property).
 *
 *  Returns true if the codepoint has a Word_Break property that typically forms words:
 *  ALetter, Hebrew_Letter, Numeric, Katakana, ExtendNumLet, or mid-word punctuation.
 *
 *  @param[in] rune The Unicode codepoint to check.
 *  @return @c sz_true_k if the codepoint is a word character, @c sz_false_k otherwise.
 */
STRINGZILLA_API_COMPTIME sz_bool_t sz_rune_is_word_char(sz_rune_t rune);

/**
 *  @brief Suggested batch size for streaming boundaries through the @c sz_utf8_find_* kernels.
 *
 *  Iterators that emit one segment/delimiter at a time buffer this many boundaries per call so
 *  the per-item overhead amortizes without an unbounded output buffer. It is only a default -
 *  any capacity works, and the kernels report @c bytes_consumed so the caller can resume past
 *  a full buffer.
 */
enum { sz_iterators_default_steps_k = 64 };

/**
 *  @brief Check if a position in UTF-8 text is a word boundary per Unicode TR29.
 *
 *  Implements the full TR29 word boundary algorithm including:
 *  - WB3: Do not break between CR and LF
 *  - WB4: Ignore Extend/Format/ZWJ characters for boundary purposes
 *  - WB5-WB13: Letter, number, and punctuation rules
 *  - WB15-WB16: Regional Indicator pair rules
 *
 *  @param[in] text UTF-8 encoded text.
 *  @param[in] length Byte length of @p text.
 *  @param[in] position Byte offset to check (must be start of a UTF-8 codepoint).
 *  @return @c sz_true_k if @p position is a word boundary, @c sz_false_k otherwise.
 *
 *  @note Position 0 and position == length are always boundaries (SOT/EOT).
 *  @note This is an internal helper used by the iterators; not part of stable ABI.
 */
STRINGZILLA_API_COMPTIME sz_bool_t sz_utf8_is_word_boundary_serial(sz_cptr_t text, sz_size_t length,
                                                                   sz_size_t position);

#pragma endregion

#pragma region Platform Specific Backends

/** @copydoc sz_utf8_wordbreaks */
STRINGZILLA_API_COMPTIME sz_size_t sz_utf8_wordbreaks_serial(sz_cptr_t text, sz_size_t length, sz_size_t *word_starts,
                                                             sz_size_t *word_lengths, sz_size_t words_capacity,
                                                             sz_size_t *bytes_consumed);

#if STRINGZILLA_TARGET_HASWELL
/** @copydoc sz_utf8_wordbreaks */
STRINGZILLA_API_COMPTIME sz_size_t sz_utf8_wordbreaks_haswell(sz_cptr_t text, sz_size_t length, sz_size_t *word_starts,
                                                              sz_size_t *word_lengths, sz_size_t words_capacity,
                                                              sz_size_t *bytes_consumed);
#endif

#if STRINGZILLA_TARGET_ICELAKE
/** @copydoc sz_utf8_wordbreaks */
STRINGZILLA_API_COMPTIME sz_size_t sz_utf8_wordbreaks_icelake(sz_cptr_t text, sz_size_t length, sz_size_t *word_starts,
                                                              sz_size_t *word_lengths, sz_size_t words_capacity,
                                                              sz_size_t *bytes_consumed);
#endif

#if STRINGZILLA_TARGET_NEON
/** @copydoc sz_utf8_wordbreaks */
STRINGZILLA_API_COMPTIME sz_size_t sz_utf8_wordbreaks_neon(sz_cptr_t text, sz_size_t length, sz_size_t *word_starts,
                                                           sz_size_t *word_lengths, sz_size_t words_capacity,
                                                           sz_size_t *bytes_consumed);
#endif

#if STRINGZILLA_TARGET_SVE2
/** @copydoc sz_utf8_wordbreaks */
STRINGZILLA_API_COMPTIME sz_size_t sz_utf8_wordbreaks_sve2(sz_cptr_t text, sz_size_t length, sz_size_t *word_starts,
                                                           sz_size_t *word_lengths, sz_size_t words_capacity,
                                                           sz_size_t *bytes_consumed);
#endif

#if STRINGZILLA_TARGET_V128
/** @copydoc sz_utf8_wordbreaks */
STRINGZILLA_API_COMPTIME sz_size_t sz_utf8_wordbreaks_v128(sz_cptr_t text, sz_size_t length, sz_size_t *word_starts,
                                                           sz_size_t *word_lengths, sz_size_t words_capacity,
                                                           sz_size_t *bytes_consumed);
#endif

#if STRINGZILLA_TARGET_RVV
/** @copydoc sz_utf8_wordbreaks */
STRINGZILLA_API_COMPTIME sz_size_t sz_utf8_wordbreaks_rvv(sz_cptr_t text, sz_size_t length, sz_size_t *word_starts,
                                                          sz_size_t *word_lengths, sz_size_t words_capacity,
                                                          sz_size_t *bytes_consumed);
#endif

#if STRINGZILLA_TARGET_LASX
/** @copydoc sz_utf8_wordbreaks */
STRINGZILLA_API_COMPTIME sz_size_t sz_utf8_wordbreaks_lasx(sz_cptr_t text, sz_size_t length, sz_size_t *word_starts,
                                                           sz_size_t *word_lengths, sz_size_t words_capacity,
                                                           sz_size_t *bytes_consumed);
#endif

#if STRINGZILLA_TARGET_POWERVSX
/** @copydoc sz_utf8_wordbreaks */
STRINGZILLA_API_COMPTIME sz_size_t sz_utf8_wordbreaks_powervsx(sz_cptr_t text, sz_size_t length, sz_size_t *word_starts,
                                                               sz_size_t *word_lengths, sz_size_t words_capacity,
                                                               sz_size_t *bytes_consumed);
#endif

#pragma endregion

/*  Implementation Section - each ISA backend lives in its own header, included serial-first. */
#include "stringzilla/utf8_wordbreaks/serial.h"
#include "stringzilla/utf8_wordbreaks/icelake.h"
#include "stringzilla/utf8_wordbreaks/haswell.h"
#include "stringzilla/utf8_wordbreaks/neon.h"
#include "stringzilla/utf8_wordbreaks/sve2.h"
#include "stringzilla/utf8_wordbreaks/v128.h"
#include "stringzilla/utf8_wordbreaks/rvv.h"
#include "stringzilla/utf8_wordbreaks/lasx.h"
#include "stringzilla/utf8_wordbreaks/powervsx.h"

#pragma region Dynamic Dispatch

#if !STRINGZILLA_RUNTIME_DISPATCH

STRINGZILLA_API_RUNTIME sz_size_t sz_utf8_wordbreaks(sz_cptr_t text, sz_size_t length, sz_size_t *word_starts,
                                                     sz_size_t *word_lengths, sz_size_t words_capacity,
                                                     sz_size_t *bytes_consumed) {
#if STRINGZILLA_TARGET_ICELAKE
    return sz_utf8_wordbreaks_icelake(text, length, word_starts, word_lengths, words_capacity, bytes_consumed);
#elif STRINGZILLA_TARGET_HASWELL
    return sz_utf8_wordbreaks_haswell(text, length, word_starts, word_lengths, words_capacity, bytes_consumed);
#elif STRINGZILLA_TARGET_SVE2
    return sz_utf8_wordbreaks_sve2(text, length, word_starts, word_lengths, words_capacity, bytes_consumed);
#elif STRINGZILLA_TARGET_NEON
    return sz_utf8_wordbreaks_neon(text, length, word_starts, word_lengths, words_capacity, bytes_consumed);
#elif STRINGZILLA_TARGET_V128
    return sz_utf8_wordbreaks_v128(text, length, word_starts, word_lengths, words_capacity, bytes_consumed);
#elif STRINGZILLA_TARGET_RVV
    return sz_utf8_wordbreaks_rvv(text, length, word_starts, word_lengths, words_capacity, bytes_consumed);
#elif STRINGZILLA_TARGET_LASX
    return sz_utf8_wordbreaks_lasx(text, length, word_starts, word_lengths, words_capacity, bytes_consumed);
#elif STRINGZILLA_TARGET_POWERVSX
    return sz_utf8_wordbreaks_powervsx(text, length, word_starts, word_lengths, words_capacity, bytes_consumed);
#else
    return sz_utf8_wordbreaks_serial(text, length, word_starts, word_lengths, words_capacity, bytes_consumed);
#endif
}

#endif // !STRINGZILLA_RUNTIME_DISPATCH

#pragma endregion

#ifdef __cplusplus
}
#endif

#endif // STRINGZILLA_UTF8_WORDBREAKS_H_
