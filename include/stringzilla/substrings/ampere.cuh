/**
 *  @file include/stringzilla/substrings/ampere.cuh
 *  @author Ash Vardanian
 *  @date November 14, 2026
 *  @brief Ampere tier of multi-pattern search: the scans totalling each tile with @c redux.sync ,
 *      and the @c _ampere exports, over the CUDA host side.
 *
 *  A scan's tile first learns its own total, which the CUDA tier reads off a scan of every warp
 *  and this tier takes from one reduction per warp: one @c redux.sync where the tile's chunks or
 *  keep flags bound it to 32 bits, two over the halves of a size or a count otherwise. BM25 keeps
 *  the CUDA tier's sums, as its fixed-point terms are signed and a split reduction is exact only
 *  for unsigned sums below 2^59.
 *
 *  @sa include/stringzilla/substrings/cuda.cuh
 *  @sa include/stringzilla/cuda.cuh
 */
#ifndef STRINGZILLA_SUBSTRINGS_AMPERE_CUH_
#define STRINGZILLA_SUBSTRINGS_AMPERE_CUH_

#include "stringzilla/substrings/cuda.cuh"
#include "stringzilla/strings/ampere.cuh" // `sz_block_sum_ampere_`, `sz_block_sum_bounded_ampere_`

#if STRINGZILLA_ARCH_CUDA_
#if STRINGZILLA_ARCH_CUDA_AMPERE_

#ifdef __cplusplus
extern "C" {
#endif

#if STRINGZILLA_TARGET_AMPERE

/** @ref sz_substrings_scan_tile_simt_ over CUDA's warps, each tile totalled through
 *  @ref sz_block_sum_ampere_ . */
static __global__ void __launch_bounds__(sz_substrings_threads_per_block_simt_k)
    sz_substrings_scan_ampere_kernel_(sz_size_t *values, sz_size_t count, sz_size_t const *live,
                                      sz_size_t elements_per_tile, sz_size_t *chain, sz_u64_t *states,
                                      sz_u64_t *stale) {
    sz_substrings_scan_tile_simt_(values, count, live, elements_per_tile, chain, states, stale, sz_block_scan_cuda_,
                                  sz_block_sum_ampere_, sz_chain_tile_simt_, sz_chain_look_back_cuda_);
}

/** @ref sz_substrings_scan_bounded_cuda_kernel_ with each tile totalled through
 *  @ref sz_block_sum_bounded_ampere_ , as a round's chunks and keep flags sum below 2^32. */
static __global__ void __launch_bounds__(sz_substrings_threads_per_block_simt_k)
    sz_substrings_scan_bounded_ampere_kernel_(sz_size_t *values, sz_size_t count, sz_size_t const *live,
                                              sz_size_t elements_per_tile, sz_size_t *chain, sz_u64_t *states,
                                              sz_u64_t *stale) {
    sz_substrings_scan_tile_simt_(values, count, live, elements_per_tile, chain, states, stale,
                                  sz_block_scan_bounded_cuda_, sz_block_sum_bounded_ampere_, sz_chain_tile_simt_,
                                  sz_chain_look_back_cuda_);
}

/** @ref sz_substrings_scan_sized_simt_ over CUDA's warps, each tile totalled through
 *  @ref sz_block_sum_ampere_ . */
static __global__ void __launch_bounds__(sz_substrings_threads_per_block_simt_k)
    sz_substrings_scan_sized_ampere_kernel_(sz_size_t *chunk_slots, sz_size_t count, sz_size_t const *chunks_count,
                                            sz_size_t elements_per_tile, sz_size_t *chain, sz_u64_t *states,
                                            sz_u64_t *stale, sz_size_t matches_budget, sz_bool_t emitting,
                                            sz_substrings_report_t *report) {
    sz_substrings_scan_sized_simt_(chunk_slots, count, chunks_count, elements_per_tile, chain, states, stale,
                                   matches_budget, emitting, report, sz_block_scan_cuda_, sz_block_sum_ampere_,
                                   sz_chain_tile_simt_, sz_chain_look_back_cuda_);
}

/** The Ampere tier's entry points, totalling each scan's tiles with @c redux.sync . */
static sz_substrings_entry_points_cuda_t const sz_substrings_entry_points_ampere_ = {
    (void const *)sz_substrings_scan_ampere_kernel_,
    (void const *)sz_substrings_scan_bounded_ampere_kernel_,
    (void const *)sz_substrings_scan_sized_ampere_kernel_,
    (void const *)sz_substrings_rewrite_offsets_cuda_kernel_,
    1,
    sz_cap_ampere_k,
};

STRINGZILLA_API sz_status_t sz_substrings_engine_init_ampere(
    sz_substrings_engine_t *engine, sz_strings_view_t const *needles, sz_case_sensitivity_t case_sensitivity,
    sz_substrings_overlap_policy_t overlap_policy, sz_size_t hot_states, sz_size_t matches_budget,
    sz_size_t haystacks_budget, sz_allocator_t *allocator, sz_stream_t stream) {
    return sz_substrings_engine_init_scoped_cuda_(engine, needles, case_sensitivity, overlap_policy, hot_states,
                                                  matches_budget, haystacks_budget, &sz_substrings_entry_points_ampere_,
                                                  allocator, stream);
}

STRINGZILLA_API sz_status_t sz_substrings_contains_ampere(sz_substrings_engine_t *engine,
                                                          sz_strings_view_t const *haystacks, sz_u1x8_t *contained,
                                                          sz_stream_t stream) {
    return sz_substrings_contains_scoped_cuda_(engine, haystacks, contained, &sz_substrings_entry_points_ampere_,
                                               stream);
}

STRINGZILLA_API sz_status_t sz_substrings_counts_ampere(sz_substrings_engine_t *engine,
                                                        sz_strings_view_t const *haystacks, sz_size_t *counts,
                                                        sz_size_t counts_stride, sz_stream_t stream) {
    return sz_substrings_counts_scoped_cuda_(engine, haystacks, counts, counts_stride,
                                             &sz_substrings_entry_points_ampere_, stream);
}

STRINGZILLA_API sz_status_t sz_substrings_find_ampere(sz_substrings_engine_t *engine,
                                                      sz_strings_view_t const *haystacks, sz_match_t *matches,
                                                      sz_size_t matches_capacity, sz_size_t *matches_offsets,
                                                      sz_stream_t stream) {
    return sz_substrings_find_scoped_cuda_(engine, haystacks, matches, matches_capacity, matches_offsets,
                                           &sz_substrings_entry_points_ampere_, stream);
}

STRINGZILLA_API sz_status_t sz_substrings_replace_ampere(sz_substrings_engine_t *engine,
                                                         sz_strings_view_t const *haystacks,
                                                         sz_strings_view_t const *replacements, sz_ptr_t target,
                                                         sz_size_t target_capacity, sz_size_t *offsets,
                                                         sz_stream_t stream) {
    return sz_substrings_replace_scoped_cuda_(engine, haystacks, replacements, target, target_capacity, offsets,
                                              &sz_substrings_entry_points_ampere_, stream);
}

STRINGZILLA_API sz_status_t sz_substrings_bm25_scores_ampere(sz_substrings_engine_t *engine,
                                                             sz_strings_view_t const *haystacks,
                                                             sz_f32_t const *document_lengths,
                                                             sz_substrings_bm25_t const *parameters,
                                                             sz_f32_t const *needle_weights, sz_f32_t *scores,
                                                             sz_size_t scores_stride, sz_stream_t stream) {
    return sz_substrings_bm25_scores_scoped_cuda_(engine, haystacks, document_lengths, parameters, needle_weights,
                                                  scores, scores_stride, STRINGZILLA_NULL,
                                                  &sz_substrings_entry_points_ampere_, stream);
}

#endif // STRINGZILLA_TARGET_AMPERE

#ifdef __cplusplus
}
#endif
#endif // STRINGZILLA_ARCH_CUDA_AMPERE_
#endif // STRINGZILLA_ARCH_CUDA_
#endif // STRINGZILLA_SUBSTRINGS_AMPERE_CUH_
