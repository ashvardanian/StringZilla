/**
 *  @file include/stringzilla/overlap/blackwell.cuh
 *  @author Ash Vardanian
 *  @date October 8, 2026
 *  @brief Blackwell tier of window overlap: the scoring kernel over the cluster launch control tile
 *      queue, and the @c _blackwell exports, over the CUDA host side.
 *
 *  A block done with its tile cancels a block not yet started and takes over its tile, keeping the
 *  chunks or the tree it already laid out whenever the two share candidates or a query.
 *
 *  @sa include/stringzilla/overlap/cuda.cuh
 *  @sa include/stringzilla/levenshtein/blackwell.cuh
 */
#ifndef STRINGZILLA_OVERLAP_BLACKWELL_CUH_
#define STRINGZILLA_OVERLAP_BLACKWELL_CUH_

#include "stringzilla/overlap/cuda.cuh"
#include "stringzilla/levenshtein/blackwell.cuh" // `sz_tile_queue_open_blackwell_`, `sz_tile_queue_next_blackwell_`

#if STRINGZILLA_ARCH_CUDA_
#if STRINGZILLA_TARGET_BLACKWELL

#ifdef __cplusplus
extern "C" {
#endif

static __global__ void sz_overlap_scores_blackwell_kernel_(sz_overlap_engine_t engine, sz_sequence_t candidates,
                                                           sz_f32_t *scores, sz_size_t scores_query_stride,
                                                           sz_size_t scores_candidate_stride,
                                                           sz_size_t staged_nodes_count) {
    sz_overlap_scores_simt_(engine, candidates, scores, scores_query_stride, scores_candidate_stride,
                            staged_nodes_count, sz_shuffle_up_cuda_, sz_tile_queue_open_blackwell_,
                            sz_tile_queue_next_blackwell_);
}

STRINGZILLA_API sz_status_t sz_overlap_engine_init_blackwell(sz_overlap_engine_t *engine, sz_sequence_t const *queries,
                                                             sz_size_t const *window_widths,
                                                             sz_size_t window_widths_count, sz_size_t candidates_budget,
                                                             sz_allocator_t *allocator, sz_stream_t stream) {
    return sz_overlap_engine_init_scoped_cuda_(engine, queries, window_widths, window_widths_count, candidates_budget,
                                               (void const *)sz_overlap_scores_blackwell_kernel_, sz_cap_blackwell_k,
                                               allocator, stream);
}

STRINGZILLA_API sz_status_t sz_overlap_scores_blackwell(sz_overlap_engine_t *engine, sz_sequence_t const *candidates,
                                                        sz_f32_t *scores, sz_size_t scores_query_stride,
                                                        sz_size_t scores_candidate_stride, sz_stream_t stream) {
    return sz_overlap_scores_scoped_cuda_(engine, candidates, scores, scores_query_stride, scores_candidate_stride,
                                          (void const *)sz_overlap_scores_blackwell_kernel_, stream);
}

#ifdef __cplusplus
}
#endif
#endif // STRINGZILLA_TARGET_BLACKWELL
#endif // STRINGZILLA_ARCH_CUDA_
#endif // STRINGZILLA_OVERLAP_BLACKWELL_CUH_
