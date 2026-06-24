/**
 *  @brief Hardware-accelerated UAX-29 grapheme cluster segmentation.
 *  @file utf8_graphemes.h
 *  @author Ash Vardanian
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
 *  Walks the whole input left-to-right and writes one entry per grapheme cluster into two parallel output
 *  arrays: `cluster_starts[i]` is the byte offset of the i-th cluster and `cluster_lengths[i]` its byte length.
 *  Clusters are the spans between consecutive UAX-29 grapheme boundaries (GB1-GB999), so a single call segments
 *  the entire input without the caller having to loop and restart a scan for every cluster.
 *
 *  @param text UTF-8 encoded text.
 *  @param length Byte length of @p text.
 *  @param cluster_starts Output array of cluster byte offsets (at least @p clusters_capacity entries).
 *  @param cluster_lengths Output array of cluster byte lengths (at least @p clusters_capacity entries).
 *  @param clusters_capacity Capacity of the output arrays, in entries.
 *  @param bytes_consumed Optional output: byte offset up to which the input was segmented. Equals @p length
 *         when everything fit; otherwise it is the start of the first cluster that did not fit (a grapheme
 *         boundary), so the caller may resume from @c text+*bytes_consumed.
 *  @return Number of clusters written (at most @p clusters_capacity).
 *
 *  @note No zero-length clusters are emitted; @p length == 0 returns 0.
 */
SZ_DYNAMIC sz_size_t sz_utf8_graphemes(                    //
    sz_cptr_t text, sz_size_t length,                      //
    sz_size_t *cluster_starts, sz_size_t *cluster_lengths, //
    sz_size_t clusters_capacity, sz_size_t *bytes_consumed);

#pragma endregion

#pragma region Platform Specific Backends

/** @copydoc sz_utf8_graphemes */
SZ_PUBLIC sz_size_t sz_utf8_graphemes_serial(sz_cptr_t text, sz_size_t length, sz_size_t *cluster_starts,
                                             sz_size_t *cluster_lengths, sz_size_t clusters_capacity,
                                             sz_size_t *bytes_consumed);

#if SZ_USE_HASWELL
/** @copydoc sz_utf8_graphemes */
SZ_PUBLIC sz_size_t sz_utf8_graphemes_haswell(sz_cptr_t text, sz_size_t length, sz_size_t *cluster_starts,
                                              sz_size_t *cluster_lengths, sz_size_t clusters_capacity,
                                              sz_size_t *bytes_consumed);
#endif

#if SZ_USE_NEON
/** @copydoc sz_utf8_graphemes */
SZ_PUBLIC sz_size_t sz_utf8_graphemes_neon(sz_cptr_t text, sz_size_t length, sz_size_t *cluster_starts,
                                           sz_size_t *cluster_lengths, sz_size_t clusters_capacity,
                                           sz_size_t *bytes_consumed);
#endif

#if SZ_USE_ICELAKE
/** @copydoc sz_utf8_graphemes */
SZ_PUBLIC sz_size_t sz_utf8_graphemes_icelake(sz_cptr_t text, sz_size_t length, sz_size_t *cluster_starts,
                                              sz_size_t *cluster_lengths, sz_size_t clusters_capacity,
                                              sz_size_t *bytes_consumed);
#endif

#pragma endregion

/*  Implementation Section - each ISA backend lives in its own header, included serial-first. */
#include "stringzilla/utf8_graphemes/serial.h"
#include "stringzilla/utf8_graphemes/haswell.h"
#include "stringzilla/utf8_graphemes/neon.h"
#include "stringzilla/utf8_graphemes/icelake.h"

#pragma region Dynamic Dispatch

#if !SZ_DYNAMIC_DISPATCH

SZ_DYNAMIC sz_size_t sz_utf8_graphemes(sz_cptr_t text, sz_size_t length, sz_size_t *cluster_starts,
                                       sz_size_t *cluster_lengths, sz_size_t clusters_capacity,
                                       sz_size_t *bytes_consumed) {
#if SZ_USE_ICELAKE
    return sz_utf8_graphemes_icelake(text, length, cluster_starts, cluster_lengths, clusters_capacity, bytes_consumed);
#elif SZ_USE_HASWELL
    return sz_utf8_graphemes_haswell(text, length, cluster_starts, cluster_lengths, clusters_capacity, bytes_consumed);
#elif SZ_USE_NEON
    return sz_utf8_graphemes_neon(text, length, cluster_starts, cluster_lengths, clusters_capacity, bytes_consumed);
#else
    return sz_utf8_graphemes_serial(text, length, cluster_starts, cluster_lengths, clusters_capacity, bytes_consumed);
#endif
}

#endif // !SZ_DYNAMIC_DISPATCH

#pragma endregion

#ifdef __cplusplus
}
#endif

#endif // STRINGZILLA_UTF8_GRAPHEMES_H_
