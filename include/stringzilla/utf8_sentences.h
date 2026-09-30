/**
 *  @file include/stringzilla/utf8_sentences.h
 *  @author Ash Vardanian
 *  @date June 20, 2026
 *  @brief Hardware-accelerated UAX-29 sentence segmentation.
 */
#ifndef STRINGZILLA_UTF8_SENTENCES_H_
#define STRINGZILLA_UTF8_SENTENCES_H_

#include "stringzilla/types.h"
#include "stringzilla/capabilities.h" // `sz_capability_t`

#ifdef __cplusplus
extern "C" {
#endif

#pragma region Core API

/**
 *  @brief Segment UTF-8 text into UAX-29 sentences in a single pass.
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
 *  @param[out] sentences_count Number of sentences written, at most @p sentences_capacity.
 *  @param[out] bytes_consumed Optional byte offset up to which the input was segmented: @p length
 *      when everything fit, else the start of the first sentence that did not fit (a sentence
 *      boundary), so the caller may resume from `text + *bytes_consumed`.
 *  @param[in] capabilities One device's capabilities, like @c sz_cpu_capabilities_enabled reports.
 *  @param[in] stream Null on the CPU, or the GPU stream of that device to queue on.
 *  @return @c sz_success_k, or @c sz_missing_kernel_k when no capability in @p capabilities has it.
 *
 *  @note No zero-length sentences are emitted; @p length == 0 yields no sentences.
 *  @note Sentence segmentation is forward-only.
 */
STRINGZILLA_API sz_status_t sz_utf8_sentences_best(                                        //
    sz_cptr_t text, sz_size_t length,                                                      //
    sz_size_t *sentence_starts, sz_size_t *sentence_lengths, sz_size_t sentences_capacity, //
    sz_size_t *sentences_count, sz_size_t *bytes_consumed,                                 //
    sz_capability_t capabilities, void *stream);

/**
 *  @brief Finds the UTF-8 sentence kernel of @p kind, from the best of @p capabilities.
 *  @param[out] kernel The kernel, or null when none of @p capabilities has it.
 *  @param[out] capability The capability the kernel belongs to, or zero.
 *  @return @c sz_success_k, @c sz_missing_kernel_k, or @c sz_missing_library_k when header-only.
 */
STRINGZILLA_API sz_status_t sz_utf8_sentences_find_kernel(sz_kernel_kind_t kind, sz_capability_t capabilities,
                                                          sz_kernel_punned_t *kernel, sz_capability_t *capability);

#pragma endregion

#pragma region Platform Specific Backends

/** @copydoc sz_utf8_sentences_best */
STRINGZILLA_API sz_status_t sz_utf8_sentences_serial(                                      //
    sz_cptr_t text, sz_size_t length,                                                      //
    sz_size_t *sentence_starts, sz_size_t *sentence_lengths, sz_size_t sentences_capacity, //
    sz_size_t *sentences_count, sz_size_t *bytes_consumed, void *stream);

#if STRINGZILLA_TARGET_HASWELL
/** @copydoc sz_utf8_sentences_best */
STRINGZILLA_API sz_status_t sz_utf8_sentences_haswell(                                     //
    sz_cptr_t text, sz_size_t length,                                                      //
    sz_size_t *sentence_starts, sz_size_t *sentence_lengths, sz_size_t sentences_capacity, //
    sz_size_t *sentences_count, sz_size_t *bytes_consumed, void *stream);
#endif

#if STRINGZILLA_TARGET_ICELAKE
/** @copydoc sz_utf8_sentences_best */
STRINGZILLA_API sz_status_t sz_utf8_sentences_icelake(                                     //
    sz_cptr_t text, sz_size_t length,                                                      //
    sz_size_t *sentence_starts, sz_size_t *sentence_lengths, sz_size_t sentences_capacity, //
    sz_size_t *sentences_count, sz_size_t *bytes_consumed, void *stream);
#endif

#if STRINGZILLA_TARGET_NEON
/** @copydoc sz_utf8_sentences_best */
STRINGZILLA_API sz_status_t sz_utf8_sentences_neon(                                        //
    sz_cptr_t text, sz_size_t length,                                                      //
    sz_size_t *sentence_starts, sz_size_t *sentence_lengths, sz_size_t sentences_capacity, //
    sz_size_t *sentences_count, sz_size_t *bytes_consumed, void *stream);
#endif

#if STRINGZILLA_TARGET_SVE2
/** @copydoc sz_utf8_sentences_best */
STRINGZILLA_API sz_status_t sz_utf8_sentences_sve2(                                        //
    sz_cptr_t text, sz_size_t length,                                                      //
    sz_size_t *sentence_starts, sz_size_t *sentence_lengths, sz_size_t sentences_capacity, //
    sz_size_t *sentences_count, sz_size_t *bytes_consumed, void *stream);
#endif

#pragma endregion

/*  Header-only builds define each kernel inline from its tier header, while the library defines
 *  every kernel once, in its capability's unit under `c/cpu/`. */
#include "stringzilla/utf8_sentences/serial.h"
#if STRINGZILLA_HEADER_ONLY
#include "stringzilla/utf8_sentences/haswell.h"
#include "stringzilla/utf8_sentences/neon.h"
#include "stringzilla/utf8_sentences/icelake.h"
#include "stringzilla/utf8_sentences/sve2.h"
#endif // STRINGZILLA_HEADER_ONLY

#if STRINGZILLA_HEADER_ONLY

STRINGZILLA_API sz_status_t sz_utf8_sentences_best(                                        //
    sz_cptr_t text, sz_size_t length,                                                      //
    sz_size_t *sentence_starts, sz_size_t *sentence_lengths, sz_size_t sentences_capacity, //
    sz_size_t *sentences_count, sz_size_t *bytes_consumed,                                 //
    sz_capability_t capabilities, void *stream) {
    sz_unused_(text), sz_unused_(length), sz_unused_(sentence_starts), sz_unused_(sentence_lengths),
        sz_unused_(sentences_capacity), sz_unused_(sentences_count), sz_unused_(bytes_consumed),
        sz_unused_(capabilities), sz_unused_(stream);
    return sz_missing_library_k;
}

STRINGZILLA_API sz_status_t sz_utf8_sentences_find_kernel(sz_kernel_kind_t kind, sz_capability_t capabilities,
                                                          sz_kernel_punned_t *kernel, sz_capability_t *capability) {
    sz_unused_(kind), sz_unused_(capabilities);
    *kernel = STRINGZILLA_NULL, *capability = 0;
    return sz_missing_library_k;
}

#endif // STRINGZILLA_HEADER_ONLY

#ifdef __cplusplus
}
#endif

#endif // STRINGZILLA_UTF8_SENTENCES_H_
