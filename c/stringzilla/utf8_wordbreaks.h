/**
 *  @file c/stringzilla/utf8_wordbreaks.h
 *  @author Ash Vardanian
 *  @date November 30, 2025
 *  @brief Hardware-accelerated UAX-29 word boundary segmentation.
 */
#ifndef STRINGZILLA_UTF8_WORDBREAKS_H_
#define STRINGZILLA_UTF8_WORDBREAKS_H_

#include "stringzilla/types.h"
#include "stringzilla/capabilities.h" // `sz_capability_t`

#ifdef __cplusplus
extern "C" {
#endif

#pragma region Core API

/**
 *  @brief Segment UTF-8 text into UAX-29 words in a single pass.
 *
 *  Walks the input left-to-right and writes the byte length of each word into @p lengths. Words
 *  are the spans between consecutive TR29 boundaries and tile the input: each starts where the
 *  previous one ended, the first at @p text. A call covers all of @p text unless it fills
 *  @p lengths, in which case it covers the sum of the lengths it wrote, and the caller resumes
 *  from @p text advanced by that sum.
 *
 *  @param[in] text UTF-8 encoded text.
 *  @param[in] length Byte length of @p text.
 *  @param[out] lengths Word byte lengths, at least @p capacity entries.
 *  @param[in] capacity Capacity of @p lengths, in entries.
 *  @param[out] count Number of words written, at most @p capacity.
 *  @param[in] capabilities One device's capabilities, like @c sz_capabilities_enabled_cpu reports.
 *  @param[in] stream Null on the CPU, or the GPU stream of that device to queue on.
 *
 *  @note No zero-length words are emitted; @p length == 0 yields no words.
 *
 *  @par Statuses:
 *  - @c sz_missing_kernel_k No capability in @p capabilities has it.
 */
STRINGZILLA_API sz_status_t sz_utf8_wordbreaks_best(sz_cptr_t text, sz_size_t length, sz_size_t *lengths,
                                                    sz_size_t capacity, sz_size_t *count, sz_capability_t capabilities,
                                                    sz_stream_t stream);

/**
 *  @brief Suggested batch size for streaming boundaries through the segmenting kernels.
 *
 *  Iterators that emit one segment/delimiter at a time buffer this many boundaries per call so
 *  the per-item overhead amortizes without an unbounded output buffer. It is only a default -
 *  any capacity works, and every kernel reports how far it got, so a caller can resume past a
 *  full output buffer.
 */
enum { sz_iterators_default_steps_k = 64 };

/**
 *  @brief Finds the UTF-8 word break kernel of @p kind, from the best of @p capabilities.
 *  @param[in] kind The verb whose kernel to find.
 *  @param[in] capabilities The mask to pick the best capability from.
 *  @param[out] kernel The kernel, or null when none of @p capabilities has it.
 *  @param[out] capability The capability the kernel belongs to, or zero.
 *
 *  @par Statuses:
 *  - @c sz_missing_kernel_k
 *  - @c sz_missing_library_k Header-only.
 */
STRINGZILLA_API sz_status_t sz_utf8_wordbreaks_find_kernel(sz_kernel_kind_t kind, sz_capability_t capabilities,
                                                           sz_kernel_punned_t *kernel, sz_capability_t *capability);

#pragma endregion

#pragma region Platform Specific Backends

/** @copydoc sz_utf8_wordbreaks_best */
STRINGZILLA_API sz_status_t sz_utf8_wordbreaks_serial(sz_cptr_t text, sz_size_t length, sz_size_t *lengths,
                                                      sz_size_t capacity, sz_size_t *count, sz_stream_t stream);

#if STRINGZILLA_TARGET_HASWELL
/** @copydoc sz_utf8_wordbreaks_best */
STRINGZILLA_API sz_status_t sz_utf8_wordbreaks_haswell(sz_cptr_t text, sz_size_t length, sz_size_t *lengths,
                                                       sz_size_t capacity, sz_size_t *count, sz_stream_t stream);
#endif

#if STRINGZILLA_TARGET_ICELAKE
/** @copydoc sz_utf8_wordbreaks_best */
STRINGZILLA_API sz_status_t sz_utf8_wordbreaks_icelake(sz_cptr_t text, sz_size_t length, sz_size_t *lengths,
                                                       sz_size_t capacity, sz_size_t *count, sz_stream_t stream);
#endif

#if STRINGZILLA_TARGET_NEON
/** @copydoc sz_utf8_wordbreaks_best */
STRINGZILLA_API sz_status_t sz_utf8_wordbreaks_neon(sz_cptr_t text, sz_size_t length, sz_size_t *lengths,
                                                    sz_size_t capacity, sz_size_t *count, sz_stream_t stream);
#endif

#if STRINGZILLA_TARGET_SVE2
/** @copydoc sz_utf8_wordbreaks_best */
STRINGZILLA_API sz_status_t sz_utf8_wordbreaks_sve2(sz_cptr_t text, sz_size_t length, sz_size_t *lengths,
                                                    sz_size_t capacity, sz_size_t *count, sz_stream_t stream);
#endif

#if STRINGZILLA_TARGET_RVV
/** @copydoc sz_utf8_wordbreaks_best */
STRINGZILLA_API sz_status_t sz_utf8_wordbreaks_rvv(sz_cptr_t text, sz_size_t length, sz_size_t *lengths,
                                                   sz_size_t capacity, sz_size_t *count, sz_stream_t stream);
#endif

#if STRINGZILLA_TARGET_V128
/** @copydoc sz_utf8_wordbreaks_best */
STRINGZILLA_API sz_status_t sz_utf8_wordbreaks_v128(sz_cptr_t text, sz_size_t length, sz_size_t *lengths,
                                                    sz_size_t capacity, sz_size_t *count, sz_stream_t stream);
#endif

#if STRINGZILLA_TARGET_LOONGSONASX
/** @copydoc sz_utf8_wordbreaks_best */
STRINGZILLA_API sz_status_t sz_utf8_wordbreaks_loongsonasx(sz_cptr_t text, sz_size_t length, sz_size_t *lengths,
                                                           sz_size_t capacity, sz_size_t *count, sz_stream_t stream);
#endif

#if STRINGZILLA_TARGET_POWERVSX
/** @copydoc sz_utf8_wordbreaks_best */
STRINGZILLA_API sz_status_t sz_utf8_wordbreaks_powervsx(sz_cptr_t text, sz_size_t length, sz_size_t *lengths,
                                                        sz_size_t capacity, sz_size_t *count, sz_stream_t stream);
#endif

#pragma endregion

/*  Header-only builds define each kernel inline from its tier header, while the library defines
 *  every kernel once, in its capability's unit under `src/capabilities/`. */
#if STRINGZILLA_HEADER_ONLY
#include "stringzilla/utf8_wordbreaks/serial.h"
#include "stringzilla/utf8_wordbreaks/icelake.h"
#include "stringzilla/utf8_wordbreaks/haswell.h"
#include "stringzilla/utf8_wordbreaks/neon.h"
#include "stringzilla/utf8_wordbreaks/sve2.h"
#include "stringzilla/utf8_wordbreaks/v128.h"
#include "stringzilla/utf8_wordbreaks/rvv.h"
#include "stringzilla/utf8_wordbreaks/loongsonasx.h"
#include "stringzilla/utf8_wordbreaks/powervsx.h"
#endif // STRINGZILLA_HEADER_ONLY

#if STRINGZILLA_HEADER_ONLY

STRINGZILLA_API sz_status_t sz_utf8_wordbreaks_best(sz_cptr_t text, sz_size_t length, sz_size_t *lengths,
                                                    sz_size_t capacity, sz_size_t *count, sz_capability_t capabilities,
                                                    sz_stream_t stream) {
    sz_unused_(text), sz_unused_(length), sz_unused_(lengths), sz_unused_(capacity), sz_unused_(count),
        sz_unused_(capabilities), sz_unused_(stream);
    return sz_missing_library_k;
}

STRINGZILLA_API sz_status_t sz_utf8_wordbreaks_find_kernel(sz_kernel_kind_t kind, sz_capability_t capabilities,
                                                           sz_kernel_punned_t *kernel, sz_capability_t *capability) {
    sz_unused_(kind), sz_unused_(capabilities);
    *kernel = STRINGZILLA_NULL, *capability = 0;
    return sz_missing_library_k;
}

#endif // STRINGZILLA_HEADER_ONLY

#ifdef __cplusplus
}
#endif

#endif // STRINGZILLA_UTF8_WORDBREAKS_H_
