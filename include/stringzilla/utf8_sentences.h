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
 *  Walks the input left-to-right and writes the byte length of each sentence into @p lengths.
 *  Sentences are the spans between consecutive UAX-29 sentence boundaries (SB1-SB998) and tile the
 *  input: each starts where the previous one ended, the first at @p text. A call covers all of
 *  @p text unless it fills @p lengths, in which case it covers the sum of the lengths it wrote, and
 *  the caller resumes from @p text advanced by that sum.
 *
 *  @param[in] text UTF-8 encoded text.
 *  @param[in] length Byte length of @p text.
 *  @param[out] lengths Sentence byte lengths, at least @p capacity entries.
 *  @param[in] capacity Capacity of @p lengths, in entries.
 *  @param[out] count Number of sentences written, at most @p capacity.
 *  @param[in] capabilities One device's capabilities, like @c sz_capabilities_enabled_cpu reports.
 *  @param[in] stream Null on the CPU, or the GPU stream of that device to queue on.
 *  @return @c sz_success_k, or @c sz_missing_kernel_k when no capability in @p capabilities has it.
 *
 *  @note No zero-length sentences are emitted; @p length == 0 yields no sentences.
 *  @note Sentence segmentation is forward-only.
 */
STRINGZILLA_API sz_status_t sz_utf8_sentences_best(sz_cptr_t text, sz_size_t length, sz_size_t *lengths,
                                                   sz_size_t capacity, sz_size_t *count, sz_capability_t capabilities,
                                                   sz_stream_t stream);

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
STRINGZILLA_API sz_status_t sz_utf8_sentences_serial(sz_cptr_t text, sz_size_t length, sz_size_t *lengths,
                                                     sz_size_t capacity, sz_size_t *count, sz_stream_t stream);

#if STRINGZILLA_TARGET_HASWELL
/** @copydoc sz_utf8_sentences_best */
STRINGZILLA_API sz_status_t sz_utf8_sentences_haswell(sz_cptr_t text, sz_size_t length, sz_size_t *lengths,
                                                      sz_size_t capacity, sz_size_t *count, sz_stream_t stream);
#endif

#if STRINGZILLA_TARGET_ICELAKE
/** @copydoc sz_utf8_sentences_best */
STRINGZILLA_API sz_status_t sz_utf8_sentences_icelake(sz_cptr_t text, sz_size_t length, sz_size_t *lengths,
                                                      sz_size_t capacity, sz_size_t *count, sz_stream_t stream);
#endif

#if STRINGZILLA_TARGET_NEON
/** @copydoc sz_utf8_sentences_best */
STRINGZILLA_API sz_status_t sz_utf8_sentences_neon(sz_cptr_t text, sz_size_t length, sz_size_t *lengths,
                                                   sz_size_t capacity, sz_size_t *count, sz_stream_t stream);
#endif

#if STRINGZILLA_TARGET_SVE2
/** @copydoc sz_utf8_sentences_best */
STRINGZILLA_API sz_status_t sz_utf8_sentences_sve2(sz_cptr_t text, sz_size_t length, sz_size_t *lengths,
                                                   sz_size_t capacity, sz_size_t *count, sz_stream_t stream);
#endif

#pragma endregion

/*  Header-only builds define each kernel inline from its tier header, while the library defines
 *  every kernel once, in its capability's unit under `c/target/`. */
#include "stringzilla/utf8_sentences/serial.h"
#if STRINGZILLA_HEADER_ONLY
#include "stringzilla/utf8_sentences/haswell.h"
#include "stringzilla/utf8_sentences/neon.h"
#include "stringzilla/utf8_sentences/icelake.h"
#include "stringzilla/utf8_sentences/sve2.h"
#endif // STRINGZILLA_HEADER_ONLY

#if STRINGZILLA_HEADER_ONLY

STRINGZILLA_API sz_status_t sz_utf8_sentences_best(sz_cptr_t text, sz_size_t length, sz_size_t *lengths,
                                                   sz_size_t capacity, sz_size_t *count, sz_capability_t capabilities,
                                                   sz_stream_t stream) {
    sz_unused_(text), sz_unused_(length), sz_unused_(lengths), sz_unused_(capacity), sz_unused_(count),
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
