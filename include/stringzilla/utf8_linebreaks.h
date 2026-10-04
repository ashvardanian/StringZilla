/**
 *  @file include/stringzilla/utf8_linebreaks.h
 *  @author Ash Vardanian
 *  @date June 20, 2026
 *  @brief Hardware-accelerated UAX-14 line break segmentation.
 */
#ifndef STRINGZILLA_UTF8_LINEBREAKS_H_
#define STRINGZILLA_UTF8_LINEBREAKS_H_

#include "stringzilla/types.h"
#include "stringzilla/capabilities.h" // `sz_capability_t`

#ifdef __cplusplus
extern "C" {
#endif

#pragma region Core API

/**
 *  @brief Segment UTF-8 text into UAX-14 line-break opportunities in one pass.
 *
 *  Walks the input left-to-right and writes the byte length of each line segment into @p lengths.
 *  Segments are the spans between consecutive UAX-14 break opportunities and tile the input: each
 *  starts where the previous one ended, the first at @p text. A call covers all of @p text unless
 *  it fills @p lengths, in which case it covers the sum of the lengths it wrote, and the caller
 *  resumes from @p text advanced by that sum.
 *
 *  This emits every wrap opportunity (both the mandatory LB4/LB5 hard breaks and the allowed
 *  soft-wrap points). To split only on the hard breaks (the @c str.splitlines behavior) use
 *  @c sz_utf8_newlines_best instead, which enumerates exactly the LB4/LB5 break positions.
 *
 *  @param[in] text UTF-8 encoded text.
 *  @param[in] length Byte length of @p text.
 *  @param[out] lengths Segment byte lengths, at least @p capacity entries.
 *  @param[in] capacity Capacity of @p lengths, in entries.
 *  @param[out] count Number of segments written, at most @p capacity.
 *  @param[in] capabilities One device's capabilities, like @c sz_capabilities_enabled_cpu reports.
 *  @param[in] stream Null on the CPU, or the GPU stream of that device to queue on.
 *  @return @c sz_success_k, or @c sz_missing_kernel_k when no capability in @p capabilities has it.
 *
 *  @note No zero-length segments are emitted; @p length == 0 yields no segments.
 *  @note Line segmentation is forward-only.
 */
STRINGZILLA_API sz_status_t sz_utf8_linebreaks_best(sz_cptr_t text, sz_size_t length, sz_size_t *lengths,
                                                    sz_size_t capacity, sz_size_t *count, sz_capability_t capabilities,
                                                    sz_stream_t stream);

/**
 *  @brief Finds the UTF-8 line break kernel of @p kind, from the best of @p capabilities.
 *  @param[out] kernel The kernel, or null when none of @p capabilities has it.
 *  @param[out] capability The capability the kernel belongs to, or zero.
 *  @return @c sz_success_k, @c sz_missing_kernel_k, or @c sz_missing_library_k when header-only.
 */
STRINGZILLA_API sz_status_t sz_utf8_linebreaks_find_kernel(sz_kernel_kind_t kind, sz_capability_t capabilities,
                                                           sz_kernel_punned_t *kernel, sz_capability_t *capability);

#pragma endregion

#pragma region Platform Specific Backends

/** @copydoc sz_utf8_linebreaks_best */
STRINGZILLA_API sz_status_t sz_utf8_linebreaks_serial(sz_cptr_t text, sz_size_t length, sz_size_t *lengths,
                                                      sz_size_t capacity, sz_size_t *count, sz_stream_t stream);

#if STRINGZILLA_TARGET_HASWELL
/** @copydoc sz_utf8_linebreaks_best */
STRINGZILLA_API sz_status_t sz_utf8_linebreaks_haswell(sz_cptr_t text, sz_size_t length, sz_size_t *lengths,
                                                       sz_size_t capacity, sz_size_t *count, sz_stream_t stream);
#endif

#if STRINGZILLA_TARGET_ICELAKE
/** @copydoc sz_utf8_linebreaks_best */
STRINGZILLA_API sz_status_t sz_utf8_linebreaks_icelake(sz_cptr_t text, sz_size_t length, sz_size_t *lengths,
                                                       sz_size_t capacity, sz_size_t *count, sz_stream_t stream);
#endif

#if STRINGZILLA_TARGET_NEON
/** @copydoc sz_utf8_linebreaks_best */
STRINGZILLA_API sz_status_t sz_utf8_linebreaks_neon(sz_cptr_t text, sz_size_t length, sz_size_t *lengths,
                                                    sz_size_t capacity, sz_size_t *count, sz_stream_t stream);
#endif

#if STRINGZILLA_TARGET_SVE2
/** @copydoc sz_utf8_linebreaks_best */
STRINGZILLA_API sz_status_t sz_utf8_linebreaks_sve2(sz_cptr_t text, sz_size_t length, sz_size_t *lengths,
                                                    sz_size_t capacity, sz_size_t *count, sz_stream_t stream);
#endif

#pragma endregion

/*  Header-only builds define each kernel inline from its tier header, while the library defines
 *  every kernel once, in its capability's unit under `c/target/`. */
#include "stringzilla/utf8_linebreaks/serial.h"
#if STRINGZILLA_HEADER_ONLY
#include "stringzilla/utf8_linebreaks/haswell.h"
#include "stringzilla/utf8_linebreaks/neon.h"
#include "stringzilla/utf8_linebreaks/icelake.h"
#include "stringzilla/utf8_linebreaks/sve2.h"
#endif // STRINGZILLA_HEADER_ONLY

#if STRINGZILLA_HEADER_ONLY

STRINGZILLA_API sz_status_t sz_utf8_linebreaks_best(sz_cptr_t text, sz_size_t length, sz_size_t *lengths,
                                                    sz_size_t capacity, sz_size_t *count, sz_capability_t capabilities,
                                                    sz_stream_t stream) {
    sz_unused_(text), sz_unused_(length), sz_unused_(lengths), sz_unused_(capacity), sz_unused_(count),
        sz_unused_(capabilities), sz_unused_(stream);
    return sz_missing_library_k;
}

STRINGZILLA_API sz_status_t sz_utf8_linebreaks_find_kernel(sz_kernel_kind_t kind, sz_capability_t capabilities,
                                                           sz_kernel_punned_t *kernel, sz_capability_t *capability) {
    sz_unused_(kind), sz_unused_(capabilities);
    *kernel = STRINGZILLA_NULL, *capability = 0;
    return sz_missing_library_k;
}

#endif // STRINGZILLA_HEADER_ONLY

#ifdef __cplusplus
}
#endif

#endif // STRINGZILLA_UTF8_LINEBREAKS_H_
