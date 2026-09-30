/**
 *  @file include/stringzilla/utf8_graphemes.h
 *  @author Ash Vardanian
 *  @date June 20, 2026
 *  @brief Hardware-accelerated UAX-29 grapheme cluster segmentation.
 */
#ifndef STRINGZILLA_UTF8_GRAPHEMES_H_
#define STRINGZILLA_UTF8_GRAPHEMES_H_

#include "stringzilla/types.h"
#include "stringzilla/capabilities.h" // `sz_capability_t`

#ifdef __cplusplus
extern "C" {
#endif

#pragma region Core API

/**
 *  @brief Segment UTF-8 text into UAX-29 grapheme clusters in a single pass.
 *
 *  Walks the whole input left-to-right and writes one entry per grapheme cluster into two parallel
 *  output arrays: `cluster_starts[i]` is the byte offset of the i-th cluster and
 *  `cluster_lengths[i]` its byte length. Clusters are the spans between consecutive UAX-29 grapheme
 *  boundaries (GB1-GB999), so a single call segments the entire input without the caller having to
 *  loop and restart a scan for every cluster.
 *
 *  @param[in] text UTF-8 encoded text.
 *  @param[in] length Byte length of @p text.
 *  @param[out] cluster_starts Cluster byte offsets, at least @p clusters_capacity entries.
 *  @param[out] cluster_lengths Cluster byte lengths, at least @p clusters_capacity entries.
 *  @param[in] clusters_capacity Capacity of the output arrays, in entries.
 *  @param[out] clusters_count Number of clusters written, at most @p clusters_capacity.
 *  @param[out] bytes_consumed Optional byte offset up to which the input was segmented: @p length
 *      when everything fit, else the start of the first cluster that did not fit (a grapheme
 *      boundary), so the caller may resume from `text + *bytes_consumed`.
 *  @param[in] capabilities One device's capabilities, like @c sz_cpu_capabilities_enabled reports.
 *  @param[in] stream Null on the CPU, or the GPU stream of that device to queue on.
 *  @return @c sz_success_k, or @c sz_missing_kernel_k when no capability in @p capabilities has it.
 *
 *  @note No zero-length clusters are emitted; @p length == 0 yields no clusters.
 */
STRINGZILLA_API sz_status_t sz_utf8_graphemes_best(                                     //
    sz_cptr_t text, sz_size_t length,                                                   //
    sz_size_t *cluster_starts, sz_size_t *cluster_lengths, sz_size_t clusters_capacity, //
    sz_size_t *clusters_count, sz_size_t *bytes_consumed,                               //
    sz_capability_t capabilities, void *stream);

/**
 *  @brief Finds the UTF-8 grapheme kernel of @p kind, from the best of @p capabilities.
 *  @param[out] kernel The kernel, or null when none of @p capabilities has it.
 *  @param[out] capability The capability the kernel belongs to, or zero.
 *  @return @c sz_success_k, @c sz_missing_kernel_k, or @c sz_missing_library_k when header-only.
 */
STRINGZILLA_API sz_status_t sz_utf8_graphemes_find_kernel(sz_kernel_kind_t kind, sz_capability_t capabilities,
                                                          sz_kernel_punned_t *kernel, sz_capability_t *capability);

#pragma endregion

#pragma region Platform Specific Backends

/** @copydoc sz_utf8_graphemes_best */
STRINGZILLA_API sz_status_t sz_utf8_graphemes_serial(                                   //
    sz_cptr_t text, sz_size_t length,                                                   //
    sz_size_t *cluster_starts, sz_size_t *cluster_lengths, sz_size_t clusters_capacity, //
    sz_size_t *clusters_count, sz_size_t *bytes_consumed, void *stream);

#if STRINGZILLA_TARGET_HASWELL
/** @copydoc sz_utf8_graphemes_best */
STRINGZILLA_API sz_status_t sz_utf8_graphemes_haswell(                                  //
    sz_cptr_t text, sz_size_t length,                                                   //
    sz_size_t *cluster_starts, sz_size_t *cluster_lengths, sz_size_t clusters_capacity, //
    sz_size_t *clusters_count, sz_size_t *bytes_consumed, void *stream);
#endif

#if STRINGZILLA_TARGET_ICELAKE
/** @copydoc sz_utf8_graphemes_best */
STRINGZILLA_API sz_status_t sz_utf8_graphemes_icelake(                                  //
    sz_cptr_t text, sz_size_t length,                                                   //
    sz_size_t *cluster_starts, sz_size_t *cluster_lengths, sz_size_t clusters_capacity, //
    sz_size_t *clusters_count, sz_size_t *bytes_consumed, void *stream);
#endif

#if STRINGZILLA_TARGET_SVE2
/** @copydoc sz_utf8_graphemes_best */
STRINGZILLA_API sz_status_t sz_utf8_graphemes_sve2(                                     //
    sz_cptr_t text, sz_size_t length,                                                   //
    sz_size_t *cluster_starts, sz_size_t *cluster_lengths, sz_size_t clusters_capacity, //
    sz_size_t *clusters_count, sz_size_t *bytes_consumed, void *stream);
#endif

#pragma endregion

/*  Header-only builds define each kernel inline from its tier header, while the library defines
 *  every kernel once, in its capability's unit under `c/cpu/`. */
#include "stringzilla/utf8_graphemes/serial.h"
#if STRINGZILLA_HEADER_ONLY
#include "stringzilla/utf8_graphemes/haswell.h"
#include "stringzilla/utf8_graphemes/icelake.h"
#include "stringzilla/utf8_graphemes/sve2.h"
#endif // STRINGZILLA_HEADER_ONLY

#if STRINGZILLA_HEADER_ONLY

STRINGZILLA_API sz_status_t sz_utf8_graphemes_best(                                     //
    sz_cptr_t text, sz_size_t length,                                                   //
    sz_size_t *cluster_starts, sz_size_t *cluster_lengths, sz_size_t clusters_capacity, //
    sz_size_t *clusters_count, sz_size_t *bytes_consumed,                               //
    sz_capability_t capabilities, void *stream) {
    sz_unused_(text), sz_unused_(length), sz_unused_(cluster_starts), sz_unused_(cluster_lengths),
        sz_unused_(clusters_capacity), sz_unused_(clusters_count), sz_unused_(bytes_consumed), sz_unused_(capabilities),
        sz_unused_(stream);
    return sz_missing_library_k;
}

STRINGZILLA_API sz_status_t sz_utf8_graphemes_find_kernel(sz_kernel_kind_t kind, sz_capability_t capabilities,
                                                          sz_kernel_punned_t *kernel, sz_capability_t *capability) {
    sz_unused_(kind), sz_unused_(capabilities);
    *kernel = STRINGZILLA_NULL, *capability = 0;
    return sz_missing_library_k;
}

#endif // STRINGZILLA_HEADER_ONLY

#ifdef __cplusplus
}
#endif

#endif // STRINGZILLA_UTF8_GRAPHEMES_H_
