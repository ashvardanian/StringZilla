/**
 *  @file include/stringzilla/substrings/cdna4.cuh
 *  @author Ash Vardanian
 *  @date November 14, 2026
 *  @brief CDNA4 tier of multi-pattern search: the scans of every round over CDNA3's DPP block scan,
 *      for gfx950, and the @c _cdna4 exports, over the ROCm host side.
 *
 *  gfx950 keeps the 64-lane wavefront and the row broadcasts of gfx942, so its scans run
 *  @c sz_block_scan_cdna3_ , and only the kernels compiled for gfx950 are its own.
 *
 *  @sa include/stringzilla/substrings/cdna3.cuh
 */
#ifndef STRINGZILLA_SUBSTRINGS_CDNA4_CUH_
#define STRINGZILLA_SUBSTRINGS_CDNA4_CUH_

#include "stringzilla/substrings/cdna3.cuh"
#include "stringzilla/strings/cdna3.cuh" // `sz_block_scan_bounded_cdna3_`, `sz_block_scan_cdna3_` & kin

#if STRINGZILLA_ARCH_ROCM_
#if STRINGZILLA_ARCH_ROCM_CDNA4_

#ifdef __cplusplus
extern "C" {
#endif

#if STRINGZILLA_TARGET_CDNA4

#pragma region Entry Points

static __global__ void __launch_bounds__(sz_substrings_threads_per_block_simt_k)
    sz_substrings_scan_cdna4_kernel_(sz_size_t *values, sz_size_t count, sz_size_t const *live,
                                     sz_size_t elements_per_tile, sz_size_t *chain, sz_u64_t *states, sz_u64_t *stale) {
    sz_substrings_scan_tile_simt_(values, count, live, elements_per_tile, chain, states, stale, sz_block_scan_cdna3_,
                                  sz_block_sum_cdna3_, sz_chain_tile_simt_, sz_chain_look_back_rocm_);
}

static __global__ void __launch_bounds__(sz_substrings_threads_per_block_simt_k)
    sz_substrings_scan_bounded_cdna4_kernel_(sz_size_t *values, sz_size_t count, sz_size_t const *live,
                                             sz_size_t elements_per_tile, sz_size_t *chain, sz_u64_t *states,
                                             sz_u64_t *stale) {
    sz_substrings_scan_tile_simt_(values, count, live, elements_per_tile, chain, states, stale,
                                  sz_block_scan_bounded_cdna3_, sz_block_sum_cdna3_, sz_chain_tile_simt_,
                                  sz_chain_look_back_rocm_);
}

static __global__ void __launch_bounds__(sz_substrings_threads_per_block_simt_k)
    sz_substrings_scan_sized_cdna4_kernel_(sz_size_t *chunk_slots, sz_size_t count, sz_size_t const *chunks_count,
                                           sz_size_t elements_per_tile, sz_size_t *chain, sz_u64_t *states,
                                           sz_u64_t *stale, sz_size_t matches_budget, sz_bool_t emitting,
                                           sz_substrings_report_t *report) {
    sz_substrings_scan_sized_simt_(chunk_slots, count, chunks_count, elements_per_tile, chain, states, stale,
                                   matches_budget, emitting, report, sz_block_scan_cdna3_, sz_block_sum_cdna3_,
                                   sz_chain_tile_simt_, sz_chain_look_back_rocm_);
}

static __global__ void __launch_bounds__(sz_substrings_threads_per_block_simt_k)
    sz_substrings_rewrite_offsets_cdna4_kernel_(sz_strings_view_t haystacks, sz_strings_view_t replacements,
                                                sz_size_t const *haystack_offsets, sz_match_t const *matches,
                                                sz_substrings_report_t const *report, sz_size_t *gap_offsets,
                                                sz_size_t *output_sizes) {
    sz_substrings_rewrite_offsets_simt_(haystacks, replacements, haystack_offsets, matches, report, gap_offsets,
                                        output_sizes, sz_block_scan_cdna3_);
}

/** The CDNA4 tier's entry points, scanning through DPP. */
static sz_substrings_entry_points_rocm_t const sz_substrings_entry_points_cdna4_ = {
    (void const *)sz_substrings_scan_cdna4_kernel_,
    (void const *)sz_substrings_scan_bounded_cdna4_kernel_,
    (void const *)sz_substrings_scan_sized_cdna4_kernel_,
    (void const *)sz_substrings_rewrite_offsets_cdna4_kernel_,
    sz_cap_cdna4_k,
};

#pragma endregion Entry Points

STRINGZILLA_API sz_status_t sz_substrings_engine_init_cdna4(
    sz_substrings_engine_t *engine, sz_strings_view_t const *needles, sz_case_sensitivity_t case_sensitivity,
    sz_substrings_overlap_policy_t overlap_policy, sz_size_t hot_states, sz_size_t matches_budget,
    sz_size_t haystacks_budget, sz_allocator_t *allocator, sz_stream_t stream) {
    return sz_substrings_engine_init_scoped_rocm_(engine, needles, case_sensitivity, overlap_policy, hot_states,
                                                  matches_budget, haystacks_budget, &sz_substrings_entry_points_cdna4_,
                                                  allocator, stream);
}

STRINGZILLA_API sz_status_t sz_substrings_contains_cdna4(sz_substrings_engine_t *engine,
                                                         sz_strings_view_t const *haystacks, sz_u1x8_t *contained,
                                                         sz_stream_t stream) {
    return sz_substrings_contains_scoped_rocm_(engine, haystacks, contained, &sz_substrings_entry_points_cdna4_,
                                               stream);
}

STRINGZILLA_API sz_status_t sz_substrings_counts_cdna4(sz_substrings_engine_t *engine,
                                                       sz_strings_view_t const *haystacks, sz_size_t *counts,
                                                       sz_size_t counts_stride, sz_stream_t stream) {
    return sz_substrings_counts_scoped_rocm_(engine, haystacks, counts, counts_stride,
                                             &sz_substrings_entry_points_cdna4_, stream);
}

STRINGZILLA_API sz_status_t sz_substrings_find_cdna4(sz_substrings_engine_t *engine, sz_strings_view_t const *haystacks,
                                                     sz_match_t *matches, sz_size_t matches_capacity,
                                                     sz_size_t *matches_offsets, sz_stream_t stream) {
    return sz_substrings_find_scoped_rocm_(engine, haystacks, matches, matches_capacity, matches_offsets,
                                           &sz_substrings_entry_points_cdna4_, stream);
}

STRINGZILLA_API sz_status_t sz_substrings_replace_cdna4(sz_substrings_engine_t *engine,
                                                        sz_strings_view_t const *haystacks,
                                                        sz_strings_view_t const *replacements, sz_ptr_t target,
                                                        sz_size_t target_capacity, sz_size_t *offsets,
                                                        sz_stream_t stream) {
    return sz_substrings_replace_scoped_rocm_(engine, haystacks, replacements, target, target_capacity, offsets,
                                              &sz_substrings_entry_points_cdna4_, stream);
}

STRINGZILLA_API sz_status_t sz_substrings_bm25_scores_cdna4(sz_substrings_engine_t *engine,
                                                            sz_strings_view_t const *haystacks,
                                                            sz_f32_t const *document_lengths,
                                                            sz_substrings_bm25_t const *parameters,
                                                            sz_f32_t const *needle_weights, sz_f32_t *scores,
                                                            sz_size_t scores_stride, sz_stream_t stream) {
    return sz_substrings_bm25_scores_scoped_rocm_(engine, haystacks, document_lengths, parameters, needle_weights,
                                                  scores, scores_stride, &sz_substrings_entry_points_cdna4_, stream);
}

#endif // STRINGZILLA_TARGET_CDNA4

#ifdef __cplusplus
}
#endif
#endif // STRINGZILLA_ARCH_ROCM_CDNA4_
#endif // STRINGZILLA_ARCH_ROCM_
#endif // STRINGZILLA_SUBSTRINGS_CDNA4_CUH_
