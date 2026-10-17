/**
 *  @file include/stringzilla/overlap/blackwell.cuh
 *  @author Ash Vardanian
 *  @date October 8, 2026
 *  @brief Blackwell tier of window overlap: the scoring kernel compiled for Blackwell, and the
 *      @c _blackwell exports, over the CUDA host side.
 *
 *  Its blocks run their own tiles, as the CUDA tier's do, under a register budget sized for
 *  Blackwell's multiprocessors alone. Taking over unstarted blocks through cluster launch control
 *  does not pay here: a takeover lands a resident grid away, so it reuses neither the chunks nor
 *  the tree laid out, and the hardware already starts blocks in that order.
 *
 *  @sa include/stringzilla/overlap/cuda.cuh
 *  @sa include/stringzilla/cuda.cuh
 */
#ifndef STRINGZILLA_OVERLAP_BLACKWELL_CUH_
#define STRINGZILLA_OVERLAP_BLACKWELL_CUH_

#include "stringzilla/overlap/cuda.cuh"

#if STRINGZILLA_ARCH_CUDA_
#if STRINGZILLA_TARGET_BLACKWELL

#ifdef __cplusplus
extern "C" {
#endif

/** Blocks one multiprocessor keeps resident, which caps the walk at 40 registers: it waits on
 *  dependent loads, so more resident blocks hide more of them, up to where spills cost more. */
enum { sz_overlap_blocks_per_multiprocessor_blackwell_k = 12 };

static __global__ __launch_bounds__(
    sz_overlap_candidates_per_block_cuda_k,
    sz_overlap_blocks_per_multiprocessor_blackwell_k) void sz_overlap_scores_blackwell_kernel_( //
    sz_overlap_engine_t engine, sz_strings_view_t candidates, sz_f32_t *scores, sz_size_t scores_query_stride,
    sz_size_t scores_candidate_stride, sz_size_t staged_nodes_count) {
    sz_overlap_scores_simt_(engine, candidates, scores, scores_query_stride, scores_candidate_stride,
                            staged_nodes_count, sz_shuffle_up_cuda_, sz_tile_queue_open_simt_,
                            sz_tile_queue_next_simt_);
}

static __global__ __launch_bounds__(
    sz_overlap_candidates_per_block_cuda_k) void sz_overlap_bounded_scores_blackwell_kernel_( //
    sz_overlap_engine_t engine, sz_strings_view_t candidates, sz_f32_t *scores, sz_size_t scores_query_stride,
    sz_size_t scores_candidate_stride, sz_size_t staged_nodes_count, sz_size_t bound) {
    sz_overlap_bounded_scores_simt_(engine, candidates, scores, scores_query_stride, scores_candidate_stride,
                                    staged_nodes_count, bound, sz_shuffle_index_cuda_, sz_tile_queue_open_simt_,
                                    sz_tile_queue_next_simt_);
}

STRINGZILLA_API sz_status_t sz_overlap_engine_init_blackwell(sz_overlap_engine_t *engine,
                                                             sz_strings_view_t const *queries,
                                                             sz_size_t const *window_widths,
                                                             sz_size_t window_widths_count, sz_size_t candidates_budget,
                                                             sz_allocator_t *allocator, sz_stream_t stream) {
    return sz_overlap_engine_init_scoped_cuda_(engine, queries, window_widths, window_widths_count, candidates_budget,
                                               (void const *)sz_overlap_scores_blackwell_kernel_, sz_cap_blackwell_k,
                                               allocator, stream);
}

STRINGZILLA_API sz_status_t sz_overlap_scores_blackwell(sz_overlap_engine_t *engine,
                                                        sz_strings_view_t const *candidates, sz_size_t bound,
                                                        sz_f32_t *scores, sz_size_t scores_query_stride,
                                                        sz_size_t scores_candidate_stride, sz_stream_t stream) {
    return sz_overlap_scores_scoped_cuda_(engine, candidates, bound, scores, scores_query_stride,
                                          scores_candidate_stride, (void const *)sz_overlap_scores_blackwell_kernel_,
                                          (void const *)sz_overlap_bounded_scores_blackwell_kernel_, stream);
}

#ifdef __cplusplus
}
#endif
#endif // STRINGZILLA_TARGET_BLACKWELL
#endif // STRINGZILLA_ARCH_CUDA_
#endif // STRINGZILLA_OVERLAP_BLACKWELL_CUH_
