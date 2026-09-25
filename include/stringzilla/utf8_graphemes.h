/**
 *  @file include/stringzilla/utf8_graphemes.h
 *  @author Ash Vardanian
 *  @date June 20, 2026
 *  @brief Hardware-accelerated UAX-29 grapheme cluster segmentation.
 */
#ifndef STRINGZILLA_UTF8_GRAPHEMES_H_
#define STRINGZILLA_UTF8_GRAPHEMES_H_

#include "stringzilla/types.h"

#ifdef __cplusplus
extern "C" {
#endif

#pragma region Core API

/**
 *  @brief Segment UTF-8 text into UAX-29 grapheme clusters in a single pass (dispatch function).
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
 *  @param[out] bytes_consumed Optional byte offset up to which the input was segmented: @p length
 *      when everything fit, else the start of the first cluster that did not fit (a grapheme
 *      boundary), so the caller may resume from `text + *bytes_consumed`.
 *  @return Number of clusters written (at most @p clusters_capacity).
 *
 *  @note No zero-length clusters are emitted; @p length == 0 returns 0.
 */
STRINGZILLA_API_RUNTIME sz_size_t sz_utf8_graphemes(       //
    sz_cptr_t text, sz_size_t length,                      //
    sz_size_t *cluster_starts, sz_size_t *cluster_lengths, //
    sz_size_t clusters_capacity, sz_size_t *bytes_consumed);

#pragma endregion

#pragma region Platform Specific Backends

/** @copydoc sz_utf8_graphemes */
STRINGZILLA_API_COMPTIME sz_size_t sz_utf8_graphemes_serial(sz_cptr_t text, sz_size_t length, sz_size_t *cluster_starts,
                                                            sz_size_t *cluster_lengths, sz_size_t clusters_capacity,
                                                            sz_size_t *bytes_consumed);

#if STRINGZILLA_TARGET_HASWELL
/** @copydoc sz_utf8_graphemes */
STRINGZILLA_API_COMPTIME sz_size_t sz_utf8_graphemes_haswell(sz_cptr_t text, sz_size_t length,
                                                             sz_size_t *cluster_starts, sz_size_t *cluster_lengths,
                                                             sz_size_t clusters_capacity, sz_size_t *bytes_consumed);
#endif

#if STRINGZILLA_TARGET_NEON
/** @copydoc sz_utf8_graphemes */
STRINGZILLA_API_COMPTIME sz_size_t sz_utf8_graphemes_neon(sz_cptr_t text, sz_size_t length, sz_size_t *cluster_starts,
                                                          sz_size_t *cluster_lengths, sz_size_t clusters_capacity,
                                                          sz_size_t *bytes_consumed);
#endif

#if STRINGZILLA_TARGET_ICELAKE
/** @copydoc sz_utf8_graphemes */
STRINGZILLA_API_COMPTIME sz_size_t sz_utf8_graphemes_icelake(sz_cptr_t text, sz_size_t length,
                                                             sz_size_t *cluster_starts, sz_size_t *cluster_lengths,
                                                             sz_size_t clusters_capacity, sz_size_t *bytes_consumed);
#endif

#if STRINGZILLA_TARGET_SVE2
/** @copydoc sz_utf8_graphemes */
STRINGZILLA_API_COMPTIME sz_size_t sz_utf8_graphemes_sve2(sz_cptr_t text, sz_size_t length, sz_size_t *cluster_starts,
                                                          sz_size_t *cluster_lengths, sz_size_t clusters_capacity,
                                                          sz_size_t *bytes_consumed);
#endif

#pragma endregion

/*  Implementation Section - each ISA backend lives in its own header, included serial-first. */
#include "stringzilla/utf8_graphemes/serial.h"
#include "stringzilla/utf8_graphemes/haswell.h"
#include "stringzilla/utf8_graphemes/neon.h"
#include "stringzilla/utf8_graphemes/icelake.h"
#include "stringzilla/utf8_graphemes/sve2.h"

#pragma region Dynamic Dispatch

#if !STRINGZILLA_RUNTIME_DISPATCH

STRINGZILLA_API_RUNTIME sz_size_t sz_utf8_graphemes(sz_cptr_t text, sz_size_t length, sz_size_t *cluster_starts,
                                                    sz_size_t *cluster_lengths, sz_size_t clusters_capacity,
                                                    sz_size_t *bytes_consumed) {
#if STRINGZILLA_TARGET_ICELAKE
    return sz_utf8_graphemes_icelake(text, length, cluster_starts, cluster_lengths, clusters_capacity, bytes_consumed);
#elif STRINGZILLA_TARGET_HASWELL
    return sz_utf8_graphemes_haswell(text, length, cluster_starts, cluster_lengths, clusters_capacity, bytes_consumed);
#elif STRINGZILLA_TARGET_SVE2 && STRINGZILLA_SVE_WIDER_THAN_NEON_
    return sz_utf8_graphemes_sve2(text, length, cluster_starts, cluster_lengths, clusters_capacity, bytes_consumed);
#else
    // Not NEON: grapheme clusters are dense enough that the scalar walk outruns the 128-bit windowed
    // fronts on every corpus and capacity; the vector kernels remain compiled and tested reserves.
    return sz_utf8_graphemes_serial(text, length, cluster_starts, cluster_lengths, clusters_capacity, bytes_consumed);
#endif
}

#endif // !STRINGZILLA_RUNTIME_DISPATCH

#pragma endregion

#ifdef __cplusplus
}
#endif

#endif // STRINGZILLA_UTF8_GRAPHEMES_H_
