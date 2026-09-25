/**
 *  @file include/stringzilla/utf8_sentences.h
 *  @author Ash Vardanian
 *  @date June 20, 2026
 *  @brief Hardware-accelerated UAX-29 sentence segmentation.
 */
#ifndef STRINGZILLA_UTF8_SENTENCES_H_
#define STRINGZILLA_UTF8_SENTENCES_H_

#include "stringzilla/types.h"

#ifdef __cplusplus
extern "C" {
#endif

#pragma region Core API

/**
 *  @brief Segment UTF-8 text into UAX-29 sentences in a single pass (dispatch function).
 *
 *  Walks the whole input left-to-right and writes one entry per sentence into two parallel output
 *  arrays: `sentence_starts[i]` is the byte offset of the i-th sentence and `sentence_lengths[i]`
 *  its byte length. Sentences are the spans between consecutive UAX-29 sentence boundaries
 *  (SB1-SB998), so a single call segments the entire input without the caller having to loop and
 *  restart a scan for every sentence.
 *
 *  @param[in] text UTF-8 encoded text.
 *  @param[in] length Byte length of @p text.
 *  @param[out] sentence_starts Sentence byte offsets, at least @p sentences_capacity entries.
 *  @param[out] sentence_lengths Sentence byte lengths, at least @p sentences_capacity entries.
 *  @param[in] sentences_capacity Capacity of the output arrays, in entries.
 *  @param[out] bytes_consumed Optional byte offset up to which the input was segmented: @p length
 *      when everything fit, else the start of the first sentence that did not fit (a sentence
 *      boundary), so the caller may resume from `text + *bytes_consumed`.
 *  @return Number of sentences written (at most @p sentences_capacity).
 *
 *  @note No zero-length sentences are emitted; @p length == 0 returns 0.
 *  @note Sentence segmentation is forward-only.
 */
STRINGZILLA_API_RUNTIME sz_size_t sz_utf8_sentences(         //
    sz_cptr_t text, sz_size_t length,                        //
    sz_size_t *sentence_starts, sz_size_t *sentence_lengths, //
    sz_size_t sentences_capacity, sz_size_t *bytes_consumed);

#pragma endregion

#pragma region Platform Specific Backends

/** @copydoc sz_utf8_sentences */
STRINGZILLA_API_COMPTIME sz_size_t sz_utf8_sentences_serial(sz_cptr_t text, sz_size_t length,
                                                            sz_size_t *sentence_starts, sz_size_t *sentence_lengths,
                                                            sz_size_t sentences_capacity, sz_size_t *bytes_consumed);

#if STRINGZILLA_TARGET_HASWELL
/** @copydoc sz_utf8_sentences */
STRINGZILLA_API_COMPTIME sz_size_t sz_utf8_sentences_haswell(sz_cptr_t text, sz_size_t length,
                                                             sz_size_t *sentence_starts, sz_size_t *sentence_lengths,
                                                             sz_size_t sentences_capacity, sz_size_t *bytes_consumed);
#endif

#if STRINGZILLA_TARGET_NEON
/** @copydoc sz_utf8_sentences */
STRINGZILLA_API_COMPTIME sz_size_t sz_utf8_sentences_neon(sz_cptr_t text, sz_size_t length, sz_size_t *sentence_starts,
                                                          sz_size_t *sentence_lengths, sz_size_t sentences_capacity,
                                                          sz_size_t *bytes_consumed);
#endif

#if STRINGZILLA_TARGET_ICELAKE
/** @copydoc sz_utf8_sentences */
STRINGZILLA_API_COMPTIME sz_size_t sz_utf8_sentences_icelake(sz_cptr_t text, sz_size_t length,
                                                             sz_size_t *sentence_starts, sz_size_t *sentence_lengths,
                                                             sz_size_t sentences_capacity, sz_size_t *bytes_consumed);
#endif

#if STRINGZILLA_TARGET_SVE2
/** @copydoc sz_utf8_sentences */
STRINGZILLA_API_COMPTIME sz_size_t sz_utf8_sentences_sve2(sz_cptr_t text, sz_size_t length, sz_size_t *sentence_starts,
                                                          sz_size_t *sentence_lengths, sz_size_t sentences_capacity,
                                                          sz_size_t *bytes_consumed);
#endif

#pragma endregion

/*  Implementation Section - each ISA backend lives in its own header, included serial-first. */
#include "stringzilla/utf8_sentences/serial.h"
#include "stringzilla/utf8_sentences/haswell.h"
#include "stringzilla/utf8_sentences/neon.h"
#include "stringzilla/utf8_sentences/icelake.h"
#include "stringzilla/utf8_sentences/sve2.h"

#pragma region Dynamic Dispatch

#if !STRINGZILLA_RUNTIME_DISPATCH

STRINGZILLA_API_RUNTIME sz_size_t sz_utf8_sentences(sz_cptr_t text, sz_size_t length, sz_size_t *sentence_starts,
                                                    sz_size_t *sentence_lengths, sz_size_t sentences_capacity,
                                                    sz_size_t *bytes_consumed) {
#if STRINGZILLA_TARGET_ICELAKE
    return sz_utf8_sentences_icelake(text, length, sentence_starts, sentence_lengths, sentences_capacity,
                                     bytes_consumed);
#elif STRINGZILLA_TARGET_HASWELL
    return sz_utf8_sentences_haswell(text, length, sentence_starts, sentence_lengths, sentences_capacity,
                                     bytes_consumed);
#elif STRINGZILLA_TARGET_SVE2
    return sz_utf8_sentences_sve2(text, length, sentence_starts, sentence_lengths, sentences_capacity, bytes_consumed);
#elif STRINGZILLA_TARGET_NEON
    return sz_utf8_sentences_neon(text, length, sentence_starts, sentence_lengths, sentences_capacity, bytes_consumed);
#else
    return sz_utf8_sentences_serial(text, length, sentence_starts, sentence_lengths, sentences_capacity,
                                    bytes_consumed);
#endif
}

#endif // !STRINGZILLA_RUNTIME_DISPATCH

#pragma endregion

#ifdef __cplusplus
}
#endif

#endif // STRINGZILLA_UTF8_SENTENCES_H_
