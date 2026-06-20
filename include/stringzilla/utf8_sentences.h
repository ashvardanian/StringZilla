/**
 *  @brief Hardware-accelerated UAX-29 sentence segmentation.
 *  @file utf8_sentences.h
 *  @author Ash Vardanian
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
 *  Walks the whole input left-to-right and writes one entry per sentence into two parallel output arrays:
 *  `sentence_starts[i]` is the byte offset of the i-th sentence and `sentence_lengths[i]` its byte length.
 *  Sentences are the spans between consecutive UAX-29 sentence boundaries (SB1-SB998), so a single call
 *  segments the entire input without the caller having to loop and restart a scan for every sentence.
 *
 *  @param text UTF-8 encoded text.
 *  @param length Byte length of @p text.
 *  @param sentence_starts Output array of sentence byte offsets (at least @p sentences_capacity entries).
 *  @param sentence_lengths Output array of sentence byte lengths (at least @p sentences_capacity entries).
 *  @param sentences_capacity Capacity of the output arrays, in entries.
 *  @param bytes_consumed Optional output: byte offset up to which the input was segmented. Equals @p length
 *         when everything fit; otherwise it is the start of the first sentence that did not fit (a sentence
 *         boundary), so the caller may resume from @c text+*bytes_consumed.
 *  @return Number of sentences written (at most @p sentences_capacity).
 *
 *  @note No zero-length sentences are emitted; @p length == 0 returns 0. Sentence segmentation is forward-only.
 */
SZ_DYNAMIC sz_size_t sz_utf8_sentence_find_boundaries(       //
    sz_cptr_t text, sz_size_t length,                        //
    sz_size_t *sentence_starts, sz_size_t *sentence_lengths, //
    sz_size_t sentences_capacity, sz_size_t *bytes_consumed);

#pragma endregion

#pragma region Platform-Specific Backends

/** @copydoc sz_utf8_sentence_find_boundaries */
SZ_PUBLIC sz_size_t sz_utf8_sentence_find_boundaries_serial(sz_cptr_t text, sz_size_t length,
                                                            sz_size_t *sentence_starts, sz_size_t *sentence_lengths,
                                                            sz_size_t sentences_capacity, sz_size_t *bytes_consumed);

#if SZ_USE_ICELAKE
/** @copydoc sz_utf8_sentence_find_boundaries */
SZ_PUBLIC sz_size_t sz_utf8_sentence_find_boundaries_icelake(sz_cptr_t text, sz_size_t length,
                                                             sz_size_t *sentence_starts, sz_size_t *sentence_lengths,
                                                             sz_size_t sentences_capacity, sz_size_t *bytes_consumed);
#endif

#pragma endregion

/*  Implementation Section - each ISA backend lives in its own header, included serial-first. */
#include "stringzilla/utf8_sentences/serial.h"
#include "stringzilla/utf8_sentences/icelake.h"

#pragma region Dynamic Dispatch

#if !SZ_DYNAMIC_DISPATCH

SZ_DYNAMIC sz_size_t sz_utf8_sentence_find_boundaries(sz_cptr_t text, sz_size_t length, sz_size_t *sentence_starts,
                                                      sz_size_t *sentence_lengths, sz_size_t sentences_capacity,
                                                      sz_size_t *bytes_consumed) {
#if SZ_USE_ICELAKE
    return sz_utf8_sentence_find_boundaries_icelake(text, length, sentence_starts, sentence_lengths, sentences_capacity,
                                                    bytes_consumed);
#else
    return sz_utf8_sentence_find_boundaries_serial(text, length, sentence_starts, sentence_lengths, sentences_capacity,
                                                   bytes_consumed);
#endif
}

#endif // !SZ_DYNAMIC_DISPATCH

#pragma endregion

#ifdef __cplusplus
}
#endif

#endif // STRINGZILLA_UTF8_SENTENCES_H_
