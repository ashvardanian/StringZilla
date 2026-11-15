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

#include "stringzilla/types.h"

#if STRINGZILLA_ARCH_CUDA_
#if STRINGZILLA_ARCH_CUDA_AMPERE_

#include "stringzilla/substrings/cuda.cuh"
#include "stringzilla/strings/ampere.cuh" // `sz_block_sum_ampere_`, `sz_block_sum_bounded_ampere_`

#if defined(__cplusplus)
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

/** @ref sz_substrings_chunk_counts_simt_ . */
static __global__ void __launch_bounds__(sz_substrings_threads_per_block_simt_k)
    sz_substrings_chunk_counts_ampere_kernel_(sz_substrings_engine_t engine, sz_strings_view_t haystacks,
                                              sz_size_t *chunk_offsets) {
    sz_substrings_chunk_counts_simt_(engine, haystacks, chunk_offsets);
}

static __global__ void __launch_bounds__(sz_substrings_threads_per_block_simt_k)
    sz_substrings_walk_cased_ampere_kernel_(sz_substrings_engine_t engine, sz_u32_t staged_count,
                                            sz_strings_view_t haystacks, sz_size_t const *chunk_offsets,
                                            sz_size_t *chunk_slots) {
    sz_substrings_walk_chunks_simt_(&engine, sz_cased_k, sz_substrings_gpu_sizing_k, staged_count, haystacks,
                                    chunk_offsets, chunk_slots, 0, STRINGZILLA_NULL, STRINGZILLA_NULL, STRINGZILLA_NULL,
                                    0, STRINGZILLA_NULL, STRINGZILLA_NULL);
}

static __global__ void __launch_bounds__(sz_substrings_threads_per_block_simt_k)
    sz_substrings_walk_uncased_ampere_kernel_(sz_substrings_engine_t engine, sz_u32_t staged_count,
                                              sz_strings_view_t haystacks, sz_size_t const *chunk_offsets,
                                              sz_size_t *chunk_slots) {
    sz_substrings_walk_chunks_simt_(&engine, sz_uncased_k, sz_substrings_gpu_sizing_k, staged_count, haystacks,
                                    chunk_offsets, chunk_slots, 0, STRINGZILLA_NULL, STRINGZILLA_NULL, STRINGZILLA_NULL,
                                    0, STRINGZILLA_NULL, STRINGZILLA_NULL);
}

static __global__ void __launch_bounds__(sz_substrings_threads_per_block_simt_k)
    sz_substrings_mark_cased_ampere_kernel_(sz_substrings_engine_t engine, sz_u32_t staged_count,
                                            sz_strings_view_t haystacks, sz_size_t const *chunk_offsets,
                                            sz_size_t *chunk_slots) {
    sz_substrings_walk_chunks_simt_(&engine, sz_cased_k, sz_substrings_gpu_marking_k, staged_count, haystacks,
                                    chunk_offsets, chunk_slots, 0, STRINGZILLA_NULL, STRINGZILLA_NULL, STRINGZILLA_NULL,
                                    0, STRINGZILLA_NULL, STRINGZILLA_NULL);
}

static __global__ void __launch_bounds__(sz_substrings_threads_per_block_simt_k)
    sz_substrings_mark_uncased_ampere_kernel_(sz_substrings_engine_t engine, sz_u32_t staged_count,
                                              sz_strings_view_t haystacks, sz_size_t const *chunk_offsets,
                                              sz_size_t *chunk_slots) {
    sz_substrings_walk_chunks_simt_(&engine, sz_uncased_k, sz_substrings_gpu_marking_k, staged_count, haystacks,
                                    chunk_offsets, chunk_slots, 0, STRINGZILLA_NULL, STRINGZILLA_NULL, STRINGZILLA_NULL,
                                    0, STRINGZILLA_NULL, STRINGZILLA_NULL);
}

static __global__ void __launch_bounds__(sz_substrings_threads_per_block_simt_k)
    sz_substrings_keep_cased_ampere_kernel_(sz_substrings_engine_t engine, sz_u32_t staged_count,
                                            sz_strings_view_t haystacks, sz_size_t const *chunk_offsets,
                                            sz_size_t *chunk_slots, sz_size_t slots_count, sz_u32_t *kept,
                                            sz_substrings_spill_simt_t *spills, sz_size_t *spilled,
                                            sz_size_t spills_capacity) {
    sz_substrings_walk_chunks_simt_(&engine, sz_cased_k, sz_substrings_gpu_sizing_k, staged_count, haystacks,
                                    chunk_offsets, chunk_slots, slots_count, kept, spills, spilled, spills_capacity,
                                    STRINGZILLA_NULL, STRINGZILLA_NULL);
}

static __global__ void __launch_bounds__(sz_substrings_threads_per_block_simt_k)
    sz_substrings_keep_uncased_ampere_kernel_(sz_substrings_engine_t engine, sz_u32_t staged_count,
                                              sz_strings_view_t haystacks, sz_size_t const *chunk_offsets,
                                              sz_size_t *chunk_slots, sz_size_t slots_count, sz_u32_t *kept,
                                              sz_substrings_spill_simt_t *spills, sz_size_t *spilled,
                                              sz_size_t spills_capacity) {
    sz_substrings_walk_chunks_simt_(&engine, sz_uncased_k, sz_substrings_gpu_sizing_k, staged_count, haystacks,
                                    chunk_offsets, chunk_slots, slots_count, kept, spills, spilled, spills_capacity,
                                    STRINGZILLA_NULL, STRINGZILLA_NULL);
}

/** @ref sz_substrings_gather_simt_ . */
static __global__ void __launch_bounds__(sz_substrings_threads_per_block_simt_k)
    sz_substrings_gather_ampere_kernel_(sz_substrings_engine_t engine, sz_strings_view_t haystacks,
                                        sz_size_t const *chunk_offsets, sz_substrings_report_t const *report,
                                        sz_size_t const *chunk_slots, sz_size_t slots_count, sz_u32_t const *kept,
                                        sz_substrings_spill_simt_t const *spills, sz_size_t const *spilled,
                                        sz_match_t *matches) {
    sz_substrings_gather_simt_(engine, haystacks, chunk_offsets, report, chunk_slots, slots_count, kept, spills,
                               spilled, matches);
}

/** @ref sz_substrings_bm25_pairs_simt_ . */
static __global__ void __launch_bounds__(sz_substrings_threads_per_block_simt_k)
    sz_substrings_bm25_pairs_ampere_kernel_(sz_substrings_pairs_simt_t pairs, sz_strings_view_t haystacks,
                                            sz_f32_t const *document_lengths, sz_substrings_bm25_t parameters,
                                            sz_f32_t const *needle_weights, unsigned long long *sums) {
    sz_substrings_bm25_pairs_simt_(pairs, haystacks, document_lengths, parameters, needle_weights, sums);
}

/** @ref sz_substrings_link_simt_ . */
static __global__ void __launch_bounds__(sz_substrings_threads_per_block_simt_k)
    sz_substrings_link_ampere_kernel_(sz_match_t const *matches, sz_substrings_report_t const *report,
                                      sz_size_t longest, sz_substrings_overlap_policy_t policy, sz_size_t *jumps,
                                      sz_size_t *keep) {
    sz_substrings_link_simt_(matches, report, longest, policy, jumps, keep);
}

/** @ref sz_substrings_double_simt_ . */
static __global__ void __launch_bounds__(sz_substrings_threads_per_block_simt_k)
    sz_substrings_double_ampere_kernel_(sz_substrings_report_t const *report, sz_size_t const *jumps,
                                        sz_size_t *doubled, sz_size_t *keep) {
    sz_substrings_double_simt_(report, jumps, doubled, keep);
}

/** @ref sz_substrings_compact_simt_ . */
static __global__ void __launch_bounds__(sz_substrings_threads_per_block_simt_k)
    sz_substrings_compact_ampere_kernel_(sz_match_t const *matches, sz_substrings_report_t *report,
                                         sz_size_t const *keep_offsets, sz_match_t *survivors) {
    sz_substrings_compact_simt_(matches, report, keep_offsets, survivors);
}

/** @ref sz_substrings_haystack_offsets_simt_ . */
static __global__ void __launch_bounds__(sz_substrings_threads_per_block_simt_k)
    sz_substrings_haystack_offsets_ampere_kernel_(sz_size_t const *chunk_offsets, sz_size_t const *chunk_slots,
                                                  sz_size_t const *keep_offsets, sz_substrings_report_t const *report,
                                                  sz_size_t *haystack_offsets, sz_size_t count) {
    sz_substrings_haystack_offsets_simt_(chunk_offsets, chunk_slots, keep_offsets, report, haystack_offsets, count);
}

/** @ref sz_substrings_counts_simt_ . */
static __global__ void __launch_bounds__(sz_substrings_threads_per_block_simt_k)
    sz_substrings_counts_ampere_kernel_(sz_size_t const *haystack_offsets, sz_size_t *counts, sz_size_t counts_stride,
                                        sz_size_t count) {
    sz_substrings_counts_simt_(haystack_offsets, counts, counts_stride, count);
}

/** @ref sz_substrings_store_matches_simt_ . */
static __global__ void __launch_bounds__(sz_substrings_threads_per_block_simt_k)
    sz_substrings_store_matches_ampere_kernel_(sz_match_t const *reported, sz_substrings_report_t const *report,
                                               sz_match_t *matches, sz_size_t matches_capacity) {
    sz_substrings_store_matches_simt_(reported, report, matches, matches_capacity);
}

/** @ref sz_substrings_stored_simt_ . */
static __global__ void __launch_bounds__(sz_substrings_threads_per_block_simt_k)
    sz_substrings_stored_ampere_kernel_(sz_substrings_report_t *report, sz_size_t matches_capacity) {
    sz_substrings_stored_simt_(report, matches_capacity);
}

/** @ref sz_substrings_rewrite_copy_simt_ . */
static __global__ void __launch_bounds__(sz_substrings_threads_per_block_simt_k)
    sz_substrings_rewrite_copy_ampere_kernel_(sz_strings_view_t haystacks, sz_strings_view_t replacements,
                                              sz_size_t const *haystack_offsets, sz_match_t const *matches,
                                              sz_size_t const *gap_offsets, sz_size_t const *output_offsets,
                                              sz_substrings_report_t const *report, sz_ptr_t output) {
    sz_substrings_rewrite_copy_simt_(haystacks, replacements, haystack_offsets, matches, gap_offsets, output_offsets,
                                     report, output);
}

/** @ref sz_substrings_target_simt_ . */
static __global__ void __launch_bounds__(sz_substrings_threads_per_block_simt_k)
    sz_substrings_target_ampere_kernel_(sz_size_t const *rewritten_at, sz_size_t target_ceiling,
                                        sz_substrings_report_t *report) {
    sz_substrings_target_simt_(rewritten_at, target_ceiling, report);
}

/** One lane per haystack marks whether its boundaries enclose a match, with CUDA's ballot. */
static __global__ void __launch_bounds__(sz_substrings_threads_per_block_simt_k)
    sz_substrings_contains_ampere_kernel_(sz_size_t const *haystack_offsets, sz_u1x8_t *contained, sz_size_t count) {
    sz_substrings_contains_simt_(haystack_offsets, contained, count, sz_lanes_ballot_cuda_);
}

/** @ref sz_substrings_bm25_simt_ over CUDA's warps. */
static __global__ void __launch_bounds__(sz_substrings_threads_per_block_simt_k)
    sz_substrings_bm25_ampere_kernel_(sz_substrings_engine_t engine, sz_strings_view_t haystacks,
                                      sz_f32_t const *document_lengths, sz_substrings_bm25_t parameters,
                                      sz_f32_t const *needle_weights, sz_u32_t *overflow_rows, sz_f32_t *scores,
                                      sz_size_t scores_stride, sz_u32_t staged_accepts_words, sz_u32_t staged_count,
                                      unsigned long long const *sums, sz_size_t const *overflowed) {
    sz_substrings_bm25_simt_(engine, haystacks, document_lengths, parameters, needle_weights, overflow_rows, scores,
                             scores_stride, staged_accepts_words, staged_count, sums, overflowed,
                             sz_u64_lanes_sum_cuda_);
}

/** The cased pairs walk, counting each pair once per counter among the warp's lanes. */
static __global__ void __launch_bounds__(sz_substrings_threads_per_block_simt_k)
    sz_substrings_pairs_cased_ampere_kernel_(sz_substrings_engine_t engine, sz_u32_t staged_count,
                                             sz_strings_view_t haystacks, sz_size_t const *chunk_offsets,
                                             sz_substrings_pairs_simt_t pairs) {
    sz_substrings_walk_chunks_simt_(&engine, sz_cased_k, sz_substrings_gpu_tallying_k, staged_count, haystacks,
                                    chunk_offsets, STRINGZILLA_NULL, 0, STRINGZILLA_NULL, STRINGZILLA_NULL,
                                    STRINGZILLA_NULL, 0, &pairs, sz_u32_increment_cuda_);
}

/** The uncased pairs walk, counting each pair once per counter among the warp's lanes. */
static __global__ void __launch_bounds__(sz_substrings_threads_per_block_simt_k)
    sz_substrings_pairs_uncased_ampere_kernel_(sz_substrings_engine_t engine, sz_u32_t staged_count,
                                               sz_strings_view_t haystacks, sz_size_t const *chunk_offsets,
                                               sz_substrings_pairs_simt_t pairs) {
    sz_substrings_walk_chunks_simt_(&engine, sz_uncased_k, sz_substrings_gpu_tallying_k, staged_count, haystacks,
                                    chunk_offsets, STRINGZILLA_NULL, 0, STRINGZILLA_NULL, STRINGZILLA_NULL,
                                    STRINGZILLA_NULL, 0, &pairs, sz_u32_increment_cuda_);
}

/** @ref sz_substrings_rewrite_offsets_simt_ over CUDA's warps. */
static __global__ void __launch_bounds__(sz_substrings_threads_per_block_simt_k)
    sz_substrings_rewrite_offsets_ampere_kernel_(sz_strings_view_t haystacks, sz_strings_view_t replacements,
                                                 sz_size_t const *haystack_offsets, sz_match_t const *matches,
                                                 sz_substrings_report_t const *report, sz_size_t *gap_offsets,
                                                 sz_size_t *output_sizes) {
    sz_substrings_rewrite_offsets_simt_(haystacks, replacements, haystack_offsets, matches, report, gap_offsets,
                                        output_sizes, sz_block_scan_cuda_);
}

/** The Ampere tier's entry points, totalling each scan's tiles with @c redux.sync . */
static sz_substrings_entry_points_cuda_t const sz_substrings_entry_points_ampere_ = {
    (void const *)sz_substrings_scan_ampere_kernel_,
    (void const *)sz_substrings_scan_bounded_ampere_kernel_,
    (void const *)sz_substrings_scan_sized_ampere_kernel_,
    (void const *)sz_substrings_rewrite_offsets_ampere_kernel_,
    (void const *)sz_substrings_chunk_counts_ampere_kernel_,
    (void const *)sz_substrings_walk_cased_ampere_kernel_,
    (void const *)sz_substrings_walk_uncased_ampere_kernel_,
    (void const *)sz_substrings_mark_cased_ampere_kernel_,
    (void const *)sz_substrings_mark_uncased_ampere_kernel_,
    (void const *)sz_substrings_keep_cased_ampere_kernel_,
    (void const *)sz_substrings_keep_uncased_ampere_kernel_,
    (void const *)sz_substrings_gather_ampere_kernel_,
    (void const *)sz_substrings_link_ampere_kernel_,
    (void const *)sz_substrings_double_ampere_kernel_,
    (void const *)sz_substrings_compact_ampere_kernel_,
    (void const *)sz_substrings_haystack_offsets_ampere_kernel_,
    (void const *)sz_substrings_counts_ampere_kernel_,
    (void const *)sz_substrings_store_matches_ampere_kernel_,
    (void const *)sz_substrings_stored_ampere_kernel_,
    (void const *)sz_substrings_rewrite_copy_ampere_kernel_,
    (void const *)sz_substrings_target_ampere_kernel_,
    (void const *)sz_substrings_contains_ampere_kernel_,
    (void const *)sz_substrings_pairs_cased_ampere_kernel_,
    (void const *)sz_substrings_pairs_uncased_ampere_kernel_,
    (void const *)sz_substrings_bm25_pairs_ampere_kernel_,
    (void const *)sz_substrings_bm25_ampere_kernel_,
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

#if defined(__cplusplus)
} // extern "C"
#endif
#endif // STRINGZILLA_ARCH_CUDA_AMPERE_
#endif // STRINGZILLA_ARCH_CUDA_
#endif // STRINGZILLA_SUBSTRINGS_AMPERE_CUH_
