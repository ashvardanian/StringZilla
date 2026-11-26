/**
 *  @file include/stringzilla/ngrams/ampere.cuh
 *  @author Ash Vardanian
 *  @date November 14, 2026
 *  @brief Ampere tier of N-grams: the overlaps and sketches kernels compiled for Ampere and newer,
 *      and the @c _ampere exports, over the CUDA host side.
 *
 *  Its kernels compile for Ampere's codes alone, where a warp sums each width's first hits in one
 *  @c redux.sync . The table stays staged once per query, as on the CUDA tier: a second buffer
 *  prefetching the next query's table would double the shared memory of every block staging one,
 *  and the walk, which waits on dependent loads, relies on the resident blocks that it costs.
 *
 *  @sa include/stringzilla/ngrams/cuda.cuh
 *  @sa include/stringzilla/cuda.cuh
 */
#ifndef STRINGZILLA_NGRAMS_AMPERE_CUH_
#define STRINGZILLA_NGRAMS_AMPERE_CUH_

#include "stringzilla/types.h"

#if STRINGZILLA_ARCH_CUDA_
#if STRINGZILLA_ARCH_CUDA_AMPERE_

#include "stringzilla/ngrams/cuda.cuh"

#if defined(__cplusplus)
extern "C" {
#endif

#if STRINGZILLA_TARGET_AMPERE

static __global__ __launch_bounds__(
    sz_ngrams_candidates_per_block_cuda_k,
    sz_ngrams_blocks_per_multiprocessor_cuda_k) void sz_ngrams_overlaps_ampere_kernel_( //
    sz_ngrams_engine_t engine, sz_strings_view_t candidates, sz_u32_t *matches, sz_size_t matches_query_stride,
    sz_size_t matches_candidate_stride, sz_size_t staged_slots_count, sz_size_t hit_words_per_warp,
    sz_u32_t misses_bound) {
    sz_simt_group_t const group = {32,
                                   threadIdx.x & 31u,
                                   sz_lanes_any_cuda_,
                                   sz_lanes_ballot_cuda_,
                                   sz_lanes_first_cuda_,
                                   sz_lanes_below_cuda_,
                                   sz_shuffle_index_cuda_,
                                   sz_shuffle_up_cuda_,
                                   sz_u32_lanes_prefix_sum_cuda_,
                                   sz_i32_lanes_prefix_min_cuda_,
                                   sz_u64_lanes_sum_cuda_,
                                   sz_group_sync_cuda_};
    sz_ngrams_overlaps_simt_(engine, candidates, matches, matches_query_stride, matches_candidate_stride,
                             staged_slots_count, hit_words_per_warp, misses_bound, &group);
}

static __global__ __launch_bounds__(
    sz_ngrams_sketch_threads_per_block_cuda_k) void sz_ngrams_sketches_ampere_kernel_( //
    sz_size_t const *widths, sz_size_t widths_count, sz_strings_view_t texts, sz_size_t slots, sz_u8_t *sketches,
    sz_size_t sketches_stride) {
    sz_ngrams_sketches_simt_(widths, widths_count, texts, slots, sketches, sketches_stride);
}

STRINGZILLA_API sz_status_t sz_ngrams_engine_init_ampere(sz_ngrams_engine_t *engine, sz_strings_view_t const *queries,
                                                         sz_size_t const *window_widths, sz_size_t window_widths_count,
                                                         sz_size_t candidates_budget, sz_allocator_t *allocator,
                                                         sz_stream_t stream) {
    return sz_ngrams_engine_init_scoped_cuda_(engine, queries, window_widths, window_widths_count, candidates_budget,
                                              (void const *)sz_ngrams_overlaps_ampere_kernel_, sz_cap_ampere_k,
                                              allocator, stream);
}

STRINGZILLA_API sz_status_t sz_ngrams_overlaps_ampere(sz_ngrams_engine_t *engine, sz_strings_view_t const *candidates,
                                                      sz_f32_t radius, sz_u32_t *matches,
                                                      sz_size_t matches_query_stride,
                                                      sz_size_t matches_candidate_stride, sz_stream_t stream) {
    return sz_ngrams_overlaps_scoped_cuda_(engine, candidates, radius, matches, matches_query_stride,
                                           matches_candidate_stride, (void const *)sz_ngrams_overlaps_ampere_kernel_,
                                           stream);
}

STRINGZILLA_API sz_status_t sz_ngrams_sketches_ampere(sz_ngrams_engine_t *engine, sz_strings_view_t const *texts,
                                                      sz_size_t slots, sz_u8_t *sketches, sz_size_t sketches_stride,
                                                      sz_stream_t stream) {
    return sz_ngrams_sketches_scoped_cuda_(engine, texts, slots, sketches, sketches_stride,
                                           (void const *)sz_ngrams_sketches_ampere_kernel_, stream);
}

#endif // STRINGZILLA_TARGET_AMPERE

#if defined(__cplusplus)
} // extern "C"
#endif
#endif // STRINGZILLA_ARCH_CUDA_AMPERE_
#endif // STRINGZILLA_ARCH_CUDA_
#endif // STRINGZILLA_NGRAMS_AMPERE_CUH_
