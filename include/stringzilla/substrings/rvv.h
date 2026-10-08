/**
 *  @file include/stringzilla/substrings/rvv.h
 *  @author Ash Vardanian
 *  @date October 8, 2026
 *  @brief RVV backend for multi-pattern search: root skips through the RVV byte-set search.
 *
 *  The transition stays scalar: a data-dependent chase has no vector form that beats eight scalar
 *  chains. This tier replaces the text-side stages around it, through @ref sz_substrings_walks_t,
 *  and inherits every other stage from the serial one.
 *
 *  @sa include/stringzilla/substrings.h
 */
#ifndef STRINGZILLA_SUBSTRINGS_RVV_H_
#define STRINGZILLA_SUBSTRINGS_RVV_H_

#include "stringzilla/types.h"

#include "stringzilla/find/rvv.h" // `sz_find_byteset_rvv_`
#include "stringzilla/substrings/serial.h"

#ifdef __cplusplus
extern "C" {
#endif

#if STRINGZILLA_ARCH_RISCV64_
#if STRINGZILLA_ARCH_RISCV64_RVV_
#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("arch=+v"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("arch=+v")
#endif

#pragma region RVV

STRINGZILLA_INLINE sz_size_t sz_substrings_count_bytes_rvv_(sz_substrings_engine_t const *engine, sz_cptr_t haystack,
                                                            sz_size_t length) {
    if (sz_substrings_skipping_pays_(engine, haystack, length))
        return sz_substrings_count_skipping_(engine, haystack, length, &sz_find_byteset_rvv_);
    return sz_substrings_count_bytes_serial_(engine, haystack, length);
}

/** Overlapping count of one haystack, skipping from the root wherever live bytes are sparse. */
STRINGZILLA_INLINE sz_size_t sz_substrings_count_bytes_rvv(sz_substrings_engine_t const *engine, sz_cptr_t haystack,
                                                           sz_size_t length) {
    return sz_substrings_count_bytes_rvv_(engine, haystack, length);
}

STRINGZILLA_INLINE void sz_substrings_find_bytes_rvv_(sz_substrings_engine_t const *engine, sz_cptr_t haystack,
                                                      sz_size_t length, sz_substrings_report_order_t order,
                                                      sz_substrings_reporter_t reporter, void *context) {
    // One chain reports in ascending end order, which satisfies either order a consumer asks for.
    if (sz_substrings_skipping_pays_(engine, haystack, length))
        sz_substrings_find_skipping_(engine, haystack, length, reporter, context, &sz_find_byteset_rvv_);
    else sz_substrings_find_bytes_serial_(engine, haystack, length, order, reporter, context);
}

/** Overlapping reports of one haystack, skipping from the root wherever live bytes are sparse. */
STRINGZILLA_INLINE void sz_substrings_find_bytes_rvv(sz_substrings_engine_t const *engine, sz_cptr_t haystack,
                                                     sz_size_t length, sz_substrings_report_order_t order,
                                                     sz_substrings_reporter_t reporter, void *context) {
    sz_substrings_find_bytes_rvv_(engine, haystack, length, order, reporter, context);
}

/** The RVV stages. */
STRINGZILLA_INLINE sz_substrings_walks_t sz_substrings_walks_rvv_(void) {
    sz_substrings_walks_t walks;
    walks.count_bytes = &sz_substrings_count_bytes_rvv_;
    walks.find_bytes = &sz_substrings_find_bytes_rvv_;
    return walks;
}

#if STRINGZILLA_TARGET_RVV

STRINGZILLA_API sz_status_t sz_substrings_engine_init_rvv(sz_substrings_engine_t *engine, sz_sequence_t const *needles,
                                                          sz_substrings_case_sensitivity_t case_sensitivity,
                                                          sz_substrings_overlap_policy_t overlap_policy,
                                                          sz_size_t hot_states, sz_size_t matches_budget,
                                                          sz_size_t haystacks_budget, sz_allocator_t *allocator,
                                                          sz_stream_t stream) {
    sz_unused_(haystacks_budget);
    return sz_substrings_engine_init_cpu_(engine, needles, case_sensitivity, overlap_policy, hot_states, matches_budget,
                                          sz_cap_rvv_k, allocator, stream);
}

STRINGZILLA_API sz_status_t sz_substrings_counts_rvv(sz_substrings_engine_t *engine, sz_sequence_t const *haystacks,
                                                     sz_size_t *counts, sz_size_t counts_stride, sz_stream_t stream) {
    sz_assert_(stream == STRINGZILLA_NULL);
    sz_substrings_walks_t const walks = sz_substrings_walks_rvv_();
    return sz_substrings_counts_with_(engine, &walks, haystacks, counts, counts_stride);
}

STRINGZILLA_API sz_status_t sz_substrings_find_rvv(sz_substrings_engine_t *engine, sz_sequence_t const *haystacks,
                                                   sz_substrings_match_t *matches, sz_size_t matches_capacity,
                                                   sz_size_t *matches_offsets, sz_stream_t stream) {
    sz_assert_(stream == STRINGZILLA_NULL);
    sz_substrings_walks_t const walks = sz_substrings_walks_rvv_();
    return sz_substrings_find_with_(engine, &walks, haystacks, matches, matches_capacity, matches_offsets);
}

STRINGZILLA_API sz_status_t sz_substrings_replace_rvv(sz_substrings_engine_t *engine, sz_sequence_t const *haystacks,
                                                      sz_sequence_t const *replacements, sz_ptr_t target,
                                                      sz_size_t target_capacity, sz_size_t *offsets,
                                                      sz_stream_t stream) {
    sz_assert_(stream == STRINGZILLA_NULL);
    sz_substrings_walks_t const walks = sz_substrings_walks_rvv_();
    return sz_substrings_replace_with_(engine, &walks, haystacks, replacements, target, target_capacity, offsets);
}

STRINGZILLA_API sz_status_t sz_substrings_bm25_scores_rvv(sz_substrings_engine_t *engine,
                                                          sz_sequence_t const *haystacks,
                                                          sz_f32_t const *document_lengths,
                                                          sz_substrings_bm25_t const *parameters,
                                                          sz_f32_t const *needle_weights, sz_f32_t *scores,
                                                          sz_size_t scores_stride, sz_stream_t stream) {
    sz_assert_(stream == STRINGZILLA_NULL);
    sz_substrings_walks_t const walks = sz_substrings_walks_rvv_();
    return sz_substrings_bm25_scores_with_(engine, &walks, haystacks, document_lengths, parameters, needle_weights,
                                           scores, scores_stride);
}

#endif // STRINGZILLA_TARGET_RVV

#pragma endregion RVV

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#endif // STRINGZILLA_ARCH_RISCV64_RVV_
#endif // STRINGZILLA_ARCH_RISCV64_

#ifdef __cplusplus
}
#endif

#endif // STRINGZILLA_SUBSTRINGS_RVV_H_
