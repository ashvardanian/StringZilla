/**
 *  @file include/stringzilla/ngrams/blackwell.cuh
 *  @author Ash Vardanian
 *  @date October 8, 2026
 *  @brief Blackwell tier of N-grams: the overlaps and sketches kernels compiled for Blackwell, and
 *      the @c _blackwell exports, over the CUDA host side.
 *
 *  Its blocks run their own tiles, as the CUDA tier's do, under a register budget sized for
 *  Blackwell's multiprocessors alone. Taking over unstarted blocks through cluster launch control
 *  does not pay here: a takeover lands a resident grid away, so it reuses no table staged, and the
 *  hardware already starts blocks in that order.
 *
 *  @sa include/stringzilla/ngrams/cuda.cuh
 *  @sa include/stringzilla/cuda.cuh
 */
#ifndef STRINGZILLA_NGRAMS_BLACKWELL_CUH_
#define STRINGZILLA_NGRAMS_BLACKWELL_CUH_

#include "stringzilla/types.h"

#if STRINGZILLA_ARCH_CUDA_
#if STRINGZILLA_ARCH_CUDA_BLACKWELL_

#include "stringzilla/ngrams/cuda.cuh"

#if defined(__cplusplus)
extern "C" {
#endif

/** Blocks one multiprocessor keeps resident, which caps the walk at 40 registers: it waits on
 *  dependent loads, so more resident blocks hide more of them, up to where spills cost more. */
enum { sz_ngrams_blocks_per_multiprocessor_blackwell_k = 12 };

#if STRINGZILLA_TARGET_BLACKWELL

static __global__ __launch_bounds__(
    sz_ngrams_candidates_per_block_cuda_k,
    sz_ngrams_blocks_per_multiprocessor_blackwell_k) void sz_ngrams_overlaps_blackwell_kernel_( //
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
    sz_ngrams_sketch_threads_per_block_cuda_k) void sz_ngrams_sketches_blackwell_kernel_( //
    sz_size_t const *widths, sz_size_t widths_count, sz_strings_view_t texts, sz_size_t slots, sz_u8_t *sketches,
    sz_size_t sketches_stride) {
    sz_ngrams_sketches_simt_(widths, widths_count, texts, slots, sketches, sketches_stride);
}

STRINGZILLA_API sz_status_t sz_ngrams_engine_init_blackwell(sz_ngrams_engine_t *engine,
                                                            sz_strings_view_t const *queries,
                                                            sz_size_t const *window_widths,
                                                            sz_size_t window_widths_count, sz_size_t candidates_budget,
                                                            sz_allocator_t *allocator, sz_stream_t stream) {
    return sz_ngrams_engine_init_scoped_cuda_(engine, queries, window_widths, window_widths_count, candidates_budget,
                                              (void const *)sz_ngrams_overlaps_blackwell_kernel_, sz_cap_blackwell_k,
                                              allocator, stream);
}

STRINGZILLA_API sz_status_t sz_ngrams_overlaps_blackwell(sz_ngrams_engine_t *engine,
                                                         sz_strings_view_t const *candidates, sz_f32_t radius,
                                                         sz_u32_t *matches, sz_size_t matches_query_stride,
                                                         sz_size_t matches_candidate_stride, sz_stream_t stream) {
    return sz_ngrams_overlaps_scoped_cuda_(engine, candidates, radius, matches, matches_query_stride,
                                           matches_candidate_stride, (void const *)sz_ngrams_overlaps_blackwell_kernel_,
                                           stream);
}

STRINGZILLA_API sz_status_t sz_ngrams_sketches_blackwell(sz_ngrams_engine_t *engine, sz_strings_view_t const *texts,
                                                         sz_size_t slots, sz_u8_t *sketches, sz_size_t sketches_stride,
                                                         sz_stream_t stream) {
    return sz_ngrams_sketches_scoped_cuda_(engine, texts, slots, sketches, sketches_stride,
                                           (void const *)sz_ngrams_sketches_blackwell_kernel_, stream);
}

#endif // STRINGZILLA_TARGET_BLACKWELL

#if defined(__cplusplus)
} // extern "C"
#endif
#endif // STRINGZILLA_ARCH_CUDA_BLACKWELL_
#endif // STRINGZILLA_ARCH_CUDA_
#endif // STRINGZILLA_NGRAMS_BLACKWELL_CUH_
