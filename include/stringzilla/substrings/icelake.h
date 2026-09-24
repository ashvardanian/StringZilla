/**
 *  @brief AVX-512 (Ice Lake) backend for multi-pattern search: root skips through the VBMI2 byte-set search.
 *  @file include/stringzilla/substrings/icelake.h
 *  @author Ash Vardanian
 *  @sa include/stringzilla/substrings.h
 *
 *  The transition stays scalar: a data-dependent chase has no vector form that beats eight scalar chains.
 *  This tier replaces the text-side stages around it, through @ref sz_substrings_walks_t, and inherits
 *  every other stage from the serial one.
 */
#ifndef STRINGZILLA_SUBSTRINGS_ICELAKE_H_
#define STRINGZILLA_SUBSTRINGS_ICELAKE_H_

#include "stringzilla/types.h"

#include "stringzilla/find/icelake.h" // `sz_find_byteset_icelake`
#include "stringzilla/substrings/serial.h"

#ifdef __cplusplus
extern "C" {
#endif

#if SZ_USE_ICELAKE
#if defined(__clang__) && SZ_CLANG_HAS_EVEX512_
#pragma clang attribute push(                                                                                        \
    __attribute__((target("avx,avx512f,avx512vl,avx512bw,avx512dq,avx512vbmi,avx512vbmi2,bmi,bmi2,lzcnt,evex512"))), \
    apply_to = function)
#elif defined(__clang__)
#pragma clang attribute push(                                                                                \
    __attribute__((target("avx,avx512f,avx512vl,avx512bw,avx512dq,avx512vbmi,avx512vbmi2,bmi,bmi2,lzcnt"))), \
    apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("avx", "avx512f", "avx512vl", "avx512bw", "avx512dq", "avx512vbmi", "avx512vbmi2", "bmi", "bmi2", \
                   "lzcnt")
#endif

#pragma region Ice Lake

/** Overlapping count of one haystack, skipping from the root wherever live bytes are sparse. */
SZ_API_COMPTIME sz_size_t sz_substrings_count_bytes_icelake(sz_substrings_engine_t const *engine,
                                                            sz_cptr_t haystack, sz_size_t length) {
    if (sz_substrings_skipping_pays_(engine, haystack, length))
        return sz_substrings_count_skipping_(engine, haystack, length, &sz_find_byteset_icelake);
    return sz_substrings_count_bytes_serial(engine, haystack, length);
}

/** Overlapping reports of one haystack, skipping from the root wherever live bytes are sparse. */
SZ_API_COMPTIME void sz_substrings_find_bytes_icelake(sz_substrings_engine_t const *engine, sz_cptr_t haystack,
                                                      sz_size_t length, sz_substrings_report_order_t order,
                                                      sz_substrings_reporter_t reporter, void *context) {
    // One chain reports in ascending end order, which satisfies either order a consumer asks for.
    if (sz_substrings_skipping_pays_(engine, haystack, length))
        sz_substrings_find_skipping_(engine, haystack, length, reporter, context, &sz_find_byteset_icelake);
    else sz_substrings_find_bytes_serial(engine, haystack, length, order, reporter, context);
}

/** The Ice Lake stages. */
SZ_API_COMPTIME sz_substrings_walks_t sz_substrings_walks_icelake_(void) {
    sz_substrings_walks_t walks;
    walks.count_bytes = &sz_substrings_count_bytes_icelake;
    walks.find_bytes = &sz_substrings_find_bytes_icelake;
    return walks;
}

SZ_API_COMPTIME sz_status_t sz_substrings_counts_icelake(sz_substrings_engine_t *engine,
                                                         sz_sequence_t const *haystacks, sz_size_t *counts,
                                                         sz_size_t counts_stride) {
    sz_substrings_walks_t const walks = sz_substrings_walks_icelake_();
    return sz_substrings_counts_with_(engine, &walks, haystacks, counts, counts_stride);
}

SZ_API_COMPTIME sz_status_t sz_substrings_find_icelake(sz_substrings_engine_t *engine, sz_sequence_t const *haystacks,
                                                       sz_substrings_match_t *matches, sz_size_t matches_capacity,
                                                       sz_size_t *matches_offsets) {
    sz_substrings_walks_t const walks = sz_substrings_walks_icelake_();
    return sz_substrings_find_with_(engine, &walks, haystacks, matches, matches_capacity, matches_offsets);
}

SZ_API_COMPTIME sz_status_t sz_substrings_replace_icelake(sz_substrings_engine_t *engine,
                                                          sz_sequence_t const *haystacks,
                                                          sz_sequence_t const *replacements, sz_ptr_t tape,
                                                          sz_size_t tape_capacity, sz_size_t *offsets) {
    sz_substrings_walks_t const walks = sz_substrings_walks_icelake_();
    return sz_substrings_replace_with_(engine, &walks, haystacks, replacements, tape, tape_capacity, offsets);
}

SZ_API_COMPTIME sz_status_t sz_substrings_bm25_scores_icelake(sz_substrings_engine_t *engine,
                                                              sz_sequence_t const *haystacks,
                                                              sz_f32_t const *document_lengths,
                                                              sz_substrings_bm25_t const *parameters,
                                                              sz_f32_t const *needle_weights, sz_f32_t *scores,
                                                              sz_size_t scores_stride) {
    sz_substrings_walks_t const walks = sz_substrings_walks_icelake_();
    return sz_substrings_bm25_scores_with_(engine, &walks, haystacks, document_lengths, parameters, needle_weights,
                                           scores, scores_stride);
}

#pragma endregion Ice Lake

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#endif // SZ_USE_ICELAKE

#ifdef __cplusplus
}
#endif

#endif // STRINGZILLA_SUBSTRINGS_ICELAKE_H_
