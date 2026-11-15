/**
 *  @file include/stringzilla/overlap/cdna4.cuh
 *  @author Ash Vardanian
 *  @date November 12, 2026
 *  @brief CDNA4 tier of window overlap: the scoring kernels over the one-lane shift of gfx950, and
 *      the @c _cdna4 exports, over the ROCm host side.
 *
 *  gfx950 keeps the DPP wavefront shift of gfx942, so the scan takes its first step one lane up
 *  through Levenshtein's CDNA3 shift, and these kernels differ from the CDNA3 tier's only by the
 *  targets they compile for.
 *
 *  @sa include/stringzilla/overlap/cdna3.cuh
 *  @sa include/stringzilla/levenshtein/cdna3.cuh
 */
#ifndef STRINGZILLA_OVERLAP_CDNA4_CUH_
#define STRINGZILLA_OVERLAP_CDNA4_CUH_

#include "stringzilla/overlap/rocm.cuh"
#include "stringzilla/levenshtein/cdna3.cuh" // `sz_shuffle_up_cdna3_`

#if STRINGZILLA_ARCH_ROCM_
#if STRINGZILLA_ARCH_ROCM_CDNA4_

#ifdef __cplusplus
extern "C" {
#endif

#if STRINGZILLA_TARGET_CDNA4

static __global__ void __launch_bounds__(sz_overlap_candidates_per_block_rocm_k)
    sz_overlap_scores_cdna4_kernel_(sz_overlap_engine_t engine, sz_strings_view_t candidates, sz_u32_t *matches,
                                    sz_size_t matches_query_stride, sz_size_t matches_candidate_stride,
                                    sz_size_t staged_nodes_count) {
    sz_overlap_scores_simt_(engine, candidates, matches, matches_query_stride, matches_candidate_stride,
                            staged_nodes_count, sz_shuffle_up_cdna3_, sz_tile_queue_open_simt_,
                            sz_tile_queue_next_simt_);
}

static __global__ void __launch_bounds__(sz_overlap_candidates_per_block_rocm_k)
    sz_overlap_bounded_scores_cdna4_kernel_(sz_overlap_engine_t engine, sz_strings_view_t candidates, sz_u32_t *matches,
                                            sz_size_t matches_query_stride, sz_size_t matches_candidate_stride,
                                            sz_size_t staged_nodes_count, sz_u32_t misses_bound) {
    sz_overlap_bounded_scores_simt_(engine, candidates, matches, matches_query_stride, matches_candidate_stride,
                                    staged_nodes_count, misses_bound, sz_group_shuffle_index_rocm_, sz_group_any_rocm_,
                                    STRINGZILLA_NULL, STRINGZILLA_NULL, sz_tile_queue_open_simt_,
                                    sz_tile_queue_next_simt_);
}

STRINGZILLA_API sz_status_t sz_overlap_engine_init_cdna4(sz_overlap_engine_t *engine, sz_strings_view_t const *queries,
                                                         sz_size_t const *window_widths, sz_size_t window_widths_count,
                                                         sz_size_t candidates_budget, sz_allocator_t *allocator,
                                                         sz_stream_t stream) {
    return sz_overlap_engine_init_scoped_rocm_(engine, queries, window_widths, window_widths_count, candidates_budget,
                                               (void const *)sz_overlap_scores_cdna4_kernel_, sz_cap_cdna4_k, allocator,
                                               stream);
}

STRINGZILLA_API sz_status_t sz_overlap_scores_cdna4(sz_overlap_engine_t *engine, sz_strings_view_t const *candidates,
                                                    sz_f32_t radius, sz_u32_t *matches, sz_size_t matches_query_stride,
                                                    sz_size_t matches_candidate_stride, sz_stream_t stream) {
    return sz_overlap_scores_scoped_rocm_(engine, candidates, radius, matches, matches_query_stride,
                                          matches_candidate_stride, (void const *)sz_overlap_scores_cdna4_kernel_,
                                          (void const *)sz_overlap_bounded_scores_cdna4_kernel_, stream);
}

#endif // STRINGZILLA_TARGET_CDNA4

#ifdef __cplusplus
}
#endif
#endif // STRINGZILLA_ARCH_ROCM_CDNA4_
#endif // STRINGZILLA_ARCH_ROCM_
#endif // STRINGZILLA_OVERLAP_CDNA4_CUH_
