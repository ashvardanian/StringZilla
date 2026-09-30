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
 *  Walks the whole input left-to-right and writes one entry per line segment into two parallel
 *  output arrays: `line_starts[i]` is the byte offset of the i-th segment and `line_lengths[i]`
 *  its byte length. Segments are the spans between consecutive UAX-14 break opportunities, so a
 *  single call segments the entire input without the caller having to loop and restart a scan
 *  for every break.
 *
 *  This emits every wrap opportunity (both the mandatory LB4/LB5 hard breaks and the allowed
 *  soft-wrap points). To split only on the hard breaks (the @c str.splitlines behavior) use
 *  @c sz_utf8_newlines_best instead, which enumerates exactly the LB4/LB5 break positions.
 *
 *  @param[in] text UTF-8 encoded text.
 *  @param[in] length Byte length of @p text.
 *  @param[out] line_starts Segment byte offsets, at least @p lines_capacity entries.
 *  @param[out] line_lengths Segment byte lengths, at least @p lines_capacity entries.
 *  @param[in] lines_capacity Capacity of the output arrays, in entries.
 *  @param[out] lines_count Number of segments written, at most @p lines_capacity.
 *  @param[out] bytes_consumed Optional byte offset up to which the input was segmented: @p length
 *      when everything fit, else the start of the first segment that did not fit (a break
 *      opportunity), so the caller may resume from `text + *bytes_consumed`.
 *  @param[in] capabilities One device's capabilities, like @c sz_cpu_capabilities_enabled reports.
 *  @param[in] stream Null on the CPU, or the GPU stream of that device to queue on.
 *  @return @c sz_success_k, or @c sz_missing_kernel_k when no capability in @p capabilities has it.
 *
 *  @note No zero-length segments are emitted; @p length == 0 yields no segments.
 *  @note Line segmentation is forward-only.
 */
STRINGZILLA_API sz_status_t sz_utf8_linebreaks_best(                           //
    sz_cptr_t text, sz_size_t length,                                          //
    sz_size_t *line_starts, sz_size_t *line_lengths, sz_size_t lines_capacity, //
    sz_size_t *lines_count, sz_size_t *bytes_consumed,                         //
    sz_capability_t capabilities, void *stream);

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
STRINGZILLA_API sz_status_t sz_utf8_linebreaks_serial(                         //
    sz_cptr_t text, sz_size_t length,                                          //
    sz_size_t *line_starts, sz_size_t *line_lengths, sz_size_t lines_capacity, //
    sz_size_t *lines_count, sz_size_t *bytes_consumed, void *stream);

#if STRINGZILLA_TARGET_HASWELL
/** @copydoc sz_utf8_linebreaks_best */
STRINGZILLA_API sz_status_t sz_utf8_linebreaks_haswell(                        //
    sz_cptr_t text, sz_size_t length,                                          //
    sz_size_t *line_starts, sz_size_t *line_lengths, sz_size_t lines_capacity, //
    sz_size_t *lines_count, sz_size_t *bytes_consumed, void *stream);
#endif

#if STRINGZILLA_TARGET_ICELAKE
/** @copydoc sz_utf8_linebreaks_best */
STRINGZILLA_API sz_status_t sz_utf8_linebreaks_icelake(                        //
    sz_cptr_t text, sz_size_t length,                                          //
    sz_size_t *line_starts, sz_size_t *line_lengths, sz_size_t lines_capacity, //
    sz_size_t *lines_count, sz_size_t *bytes_consumed, void *stream);
#endif

#if STRINGZILLA_TARGET_NEON
/** @copydoc sz_utf8_linebreaks_best */
STRINGZILLA_API sz_status_t sz_utf8_linebreaks_neon(                           //
    sz_cptr_t text, sz_size_t length,                                          //
    sz_size_t *line_starts, sz_size_t *line_lengths, sz_size_t lines_capacity, //
    sz_size_t *lines_count, sz_size_t *bytes_consumed, void *stream);
#endif

#if STRINGZILLA_TARGET_SVE2
/** @copydoc sz_utf8_linebreaks_best */
STRINGZILLA_API sz_status_t sz_utf8_linebreaks_sve2(                           //
    sz_cptr_t text, sz_size_t length,                                          //
    sz_size_t *line_starts, sz_size_t *line_lengths, sz_size_t lines_capacity, //
    sz_size_t *lines_count, sz_size_t *bytes_consumed, void *stream);
#endif

#pragma endregion

/*  Header-only builds define each kernel inline from its tier header, while the library defines
 *  every kernel once, in its capability's unit under `c/cpu/`. */
#include "stringzilla/utf8_linebreaks/serial.h"
#if STRINGZILLA_HEADER_ONLY
#include "stringzilla/utf8_linebreaks/haswell.h"
#include "stringzilla/utf8_linebreaks/neon.h"
#include "stringzilla/utf8_linebreaks/icelake.h"
#include "stringzilla/utf8_linebreaks/sve2.h"
#endif // STRINGZILLA_HEADER_ONLY

#if STRINGZILLA_HEADER_ONLY

STRINGZILLA_API sz_status_t sz_utf8_linebreaks_best(                           //
    sz_cptr_t text, sz_size_t length,                                          //
    sz_size_t *line_starts, sz_size_t *line_lengths, sz_size_t lines_capacity, //
    sz_size_t *lines_count, sz_size_t *bytes_consumed,                         //
    sz_capability_t capabilities, void *stream) {
    sz_unused_(text), sz_unused_(length), sz_unused_(line_starts), sz_unused_(line_lengths), sz_unused_(lines_capacity),
        sz_unused_(lines_count), sz_unused_(bytes_consumed), sz_unused_(capabilities), sz_unused_(stream);
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
