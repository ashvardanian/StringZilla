/**
 *  @file include/stringzilla/overlap/ampere.cuh
 *  @author Ash Vardanian
 *  @date November 14, 2026
 *  @brief Ampere tier of window overlap: the scoring kernels with warp reductions, and the
 *      @c _ampere exports, over the CUDA host side.
 *
 *  A bounded walk compares its lanes' totals every few dozen endings, which the CUDA tier sums by
 *  halves in ten shuffles and here in two @c redux.sync , and each width's count in one. The tree
 *  stays staged once per query, as on the CUDA tier: a second buffer prefetching the next query's
 *  tree would double the shared memory of every block staging one, and the walk, which waits on
 *  dependent loads, relies on the resident blocks that memory would cost.
 *
 *  @sa include/stringzilla/overlap/cuda.cuh
 *  @sa include/stringzilla/cuda.cuh
 */
#ifndef STRINGZILLA_OVERLAP_AMPERE_CUH_
#define STRINGZILLA_OVERLAP_AMPERE_CUH_

#include "stringzilla/types.h"

#if STRINGZILLA_ARCH_CUDA_
#if STRINGZILLA_ARCH_CUDA_AMPERE_

#include "stringzilla/overlap/cuda.cuh"
#include "stringzilla/strings/ampere.cuh" // `sz_u32_lanes_sum_ampere_`, `sz_u64_lanes_sum_ampere_`

#if defined(__cplusplus)
extern "C" {
#endif

#if STRINGZILLA_TARGET_AMPERE

static __global__ __launch_bounds__(
    sz_overlap_candidates_per_block_cuda_k,
    sz_overlap_blocks_per_multiprocessor_cuda_k) void sz_overlap_scores_ampere_kernel_( //
    sz_overlap_engine_t engine, sz_strings_view_t candidates, sz_u32_t *matches, sz_size_t matches_query_stride,
    sz_size_t matches_candidate_stride, sz_size_t staged_nodes_count) {
    sz_overlap_scores_simt_(engine, candidates, matches, matches_query_stride, matches_candidate_stride,
                            staged_nodes_count, sz_shuffle_up_cuda_, sz_tile_queue_open_simt_,
                            sz_tile_queue_next_simt_);
}

static __global__ __launch_bounds__(
    sz_overlap_candidates_per_block_cuda_k) void sz_overlap_bounded_scores_ampere_kernel_( //
    sz_overlap_engine_t engine, sz_strings_view_t candidates, sz_u32_t *matches, sz_size_t matches_query_stride,
    sz_size_t matches_candidate_stride, sz_size_t staged_nodes_count, sz_u32_t misses_bound) {
    sz_overlap_bounded_scores_simt_(engine, candidates, matches, matches_query_stride, matches_candidate_stride,
                                    staged_nodes_count, misses_bound, sz_group_shuffle_index_cuda_, sz_group_any_cuda_,
                                    sz_u32_lanes_sum_ampere_, sz_u64_lanes_sum_ampere_, sz_tile_queue_open_simt_,
                                    sz_tile_queue_next_simt_);
}

STRINGZILLA_API sz_status_t sz_overlap_engine_init_ampere(sz_overlap_engine_t *engine, sz_strings_view_t const *queries,
                                                          sz_size_t const *window_widths, sz_size_t window_widths_count,
                                                          sz_size_t candidates_budget, sz_allocator_t *allocator,
                                                          sz_stream_t stream) {
    return sz_overlap_engine_init_scoped_cuda_(engine, queries, window_widths, window_widths_count, candidates_budget,
                                               (void const *)sz_overlap_scores_ampere_kernel_, sz_cap_ampere_k,
                                               allocator, stream);
}

STRINGZILLA_API sz_status_t sz_overlap_scores_ampere(sz_overlap_engine_t *engine, sz_strings_view_t const *candidates,
                                                     sz_f32_t radius, sz_u32_t *matches, sz_size_t matches_query_stride,
                                                     sz_size_t matches_candidate_stride, sz_stream_t stream) {
    return sz_overlap_scores_scoped_cuda_(engine, candidates, radius, matches, matches_query_stride,
                                          matches_candidate_stride, (void const *)sz_overlap_scores_ampere_kernel_,
                                          (void const *)sz_overlap_bounded_scores_ampere_kernel_, stream);
}

#endif // STRINGZILLA_TARGET_AMPERE

#if defined(__cplusplus)
} // extern "C"
#endif
#endif // STRINGZILLA_ARCH_CUDA_AMPERE_
#endif // STRINGZILLA_ARCH_CUDA_
#endif // STRINGZILLA_OVERLAP_AMPERE_CUH_
