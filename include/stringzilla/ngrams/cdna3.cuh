/**
 *  @file include/stringzilla/ngrams/cdna3.cuh
 *  @author Ash Vardanian
 *  @date November 12, 2026
 *  @brief CDNA3 tier of N-grams: the overlaps and sketches kernels compiled for gfx942, and the
 *      @c _cdna3 exports, over the ROCm host side.
 *
 *  The walks only read lanes by index, so its kernels differ from the ROCm tier's by their names
 *  and the targets they compile for alone.
 *
 *  @sa include/stringzilla/ngrams/rocm.cuh
 */
#ifndef STRINGZILLA_NGRAMS_CDNA3_CUH_
#define STRINGZILLA_NGRAMS_CDNA3_CUH_

#include "stringzilla/types.h"

#if STRINGZILLA_ARCH_ROCM_
#if STRINGZILLA_ARCH_ROCM_CDNA3_

#include "stringzilla/ngrams/rocm.cuh"

#if defined(__cplusplus)
extern "C" {
#endif

#if STRINGZILLA_TARGET_CDNA3

static __global__ void __launch_bounds__(sz_ngrams_candidates_per_block_rocm_k)
    sz_ngrams_overlaps_cdna3_kernel_(sz_ngrams_engine_t engine, sz_strings_view_t candidates, sz_u32_t *matches,
                                     sz_size_t matches_query_stride, sz_size_t matches_candidate_stride,
                                     sz_size_t staged_slots_count, sz_size_t hit_words_per_warp,
                                     sz_u32_t misses_bound) {
    sz_simt_group_t const group = {32,
                                   threadIdx.x & 31u,
                                   sz_lanes_any_rocm_,
                                   sz_lanes_ballot_rocm_,
                                   sz_lanes_first_rocm_,
                                   sz_lanes_below_rocm_,
                                   sz_shuffle_index_rocm_,
                                   sz_shuffle_up_rocm_,
                                   sz_u32_lanes_prefix_sum_rocm_,
                                   sz_i32_lanes_prefix_min_rocm_,
                                   sz_u64_lanes_sum_rocm_,
                                   sz_group_sync_rocm_};
    sz_ngrams_overlaps_simt_(engine, candidates, matches, matches_query_stride, matches_candidate_stride,
                             staged_slots_count, hit_words_per_warp, misses_bound, &group);
}

static __global__ void __launch_bounds__(sz_ngrams_sketch_threads_per_block_rocm_k)
    sz_ngrams_sketches_cdna3_kernel_(sz_size_t const *widths, sz_size_t widths_count, sz_strings_view_t texts,
                                     sz_size_t slots, sz_u8_t *sketches, sz_size_t sketches_stride) {
    sz_ngrams_sketches_simt_(widths, widths_count, texts, slots, sketches, sketches_stride);
}

STRINGZILLA_API sz_status_t sz_ngrams_engine_init_cdna3(sz_ngrams_engine_t *engine, sz_strings_view_t const *queries,
                                                        sz_size_t const *window_widths, sz_size_t window_widths_count,
                                                        sz_size_t candidates_budget, sz_allocator_t *allocator,
                                                        sz_stream_t stream) {
    return sz_ngrams_engine_init_scoped_rocm_(engine, queries, window_widths, window_widths_count, candidates_budget,
                                              (void const *)sz_ngrams_overlaps_cdna3_kernel_, sz_cap_cdna3_k, allocator,
                                              stream);
}

STRINGZILLA_API sz_status_t sz_ngrams_overlaps_cdna3(sz_ngrams_engine_t *engine, sz_strings_view_t const *candidates,
                                                     sz_f32_t radius, sz_u32_t *matches, sz_size_t matches_query_stride,
                                                     sz_size_t matches_candidate_stride, sz_stream_t stream) {
    return sz_ngrams_overlaps_scoped_rocm_(engine, candidates, radius, matches, matches_query_stride,
                                           matches_candidate_stride, (void const *)sz_ngrams_overlaps_cdna3_kernel_,
                                           stream);
}

STRINGZILLA_API sz_status_t sz_ngrams_sketches_cdna3(sz_ngrams_engine_t *engine, sz_strings_view_t const *texts,
                                                     sz_size_t slots, sz_u8_t *sketches, sz_size_t sketches_stride,
                                                     sz_stream_t stream) {
    return sz_ngrams_sketches_scoped_rocm_(engine, texts, slots, sketches, sketches_stride,
                                           (void const *)sz_ngrams_sketches_cdna3_kernel_, stream);
}

#endif // STRINGZILLA_TARGET_CDNA3

#if defined(__cplusplus)
} // extern "C"
#endif
#endif // STRINGZILLA_ARCH_ROCM_CDNA3_
#endif // STRINGZILLA_ARCH_ROCM_
#endif // STRINGZILLA_NGRAMS_CDNA3_CUH_
