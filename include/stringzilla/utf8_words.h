/**
 *  @brief Hardware-accelerated UAX-29 word boundary segmentation.
 *  @file utf8_words.h
 *  @author Ash Vardanian
 */
#ifndef STRINGZILLA_UTF8_WORDS_H_
#define STRINGZILLA_UTF8_WORDS_H_

#include "stringzilla/types.h"

#ifdef __cplusplus
extern "C" {
#endif

#pragma region Core API

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
 *  @brief Get the Unicode TR29 Word_Break property for a codepoint.
 *
 *  Returns one of the 16 Word_Break property values (sz_utf8_word_break_other_k through
 *  sz_utf8_word_break_mid_quotes_k). This is the foundation for TR29-compliant word boundary detection.
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
 *  @brief Suggested default batch size for callers that stream boundaries through the `sz_utf8_find_*` kernels.
 *
 *  Iterators that emit one segment/delimiter at a time (the Python and C++ bindings) buffer this many boundaries
 *  per call so the per-item overhead amortizes without an unbounded output buffer. It is only a default - any
 *  capacity works, and the kernels report `bytes_consumed` so the caller can resume past a full buffer.
 */
enum { sz_iterators_default_steps_k = 16 };

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

/** @copydoc sz_utf8_word_find_boundaries */
SZ_PUBLIC sz_size_t sz_utf8_word_find_boundaries_serial(sz_cptr_t text, sz_size_t length, sz_size_t *word_starts,
                                                        sz_size_t *word_lengths, sz_size_t words_capacity,
                                                        sz_size_t *bytes_consumed);
/** @copydoc sz_utf8_word_rfind_boundaries */
SZ_PUBLIC sz_size_t sz_utf8_word_rfind_boundaries_serial(sz_cptr_t text, sz_size_t length, sz_size_t *word_starts,
                                                         sz_size_t *word_lengths, sz_size_t words_capacity,
                                                         sz_size_t *bytes_consumed);

#if SZ_USE_HASWELL
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
/** @copydoc sz_utf8_word_find_boundaries */
SZ_PUBLIC sz_size_t sz_utf8_word_find_boundaries_neon(sz_cptr_t text, sz_size_t length, sz_size_t *word_starts,
                                                      sz_size_t *word_lengths, sz_size_t words_capacity,
                                                      sz_size_t *bytes_consumed);
/** @copydoc sz_utf8_word_rfind_boundaries */
SZ_PUBLIC sz_size_t sz_utf8_word_rfind_boundaries_neon(sz_cptr_t text, sz_size_t length, sz_size_t *word_starts,
                                                       sz_size_t *word_lengths, sz_size_t words_capacity,
                                                       sz_size_t *bytes_consumed);
#endif

#if SZ_USE_SVE2
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

/*  Implementation Section - each ISA backend lives in its own header, included serial-first. */
#include "stringzilla/utf8_words/serial.h"
#include "stringzilla/utf8_words/icelake.h"
#include "stringzilla/utf8_words/haswell.h"
#include "stringzilla/utf8_words/neon.h"
#include "stringzilla/utf8_words/sve2.h"
#include "stringzilla/utf8_words/v128.h"
#include "stringzilla/utf8_words/rvv.h"
#include "stringzilla/utf8_words/lasx.h"
#include "stringzilla/utf8_words/powervsx.h"

#pragma region Dynamic Dispatch

#if !SZ_DYNAMIC_DISPATCH

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

#pragma endregion

#ifdef __cplusplus
}
#endif

#endif // STRINGZILLA_UTF8_WORDS_H_
