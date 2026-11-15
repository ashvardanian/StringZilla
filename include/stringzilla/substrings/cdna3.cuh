/**
 *  @file include/stringzilla/substrings/cdna3.cuh
 *  @author Ash Vardanian
 *  @date November 14, 2026
 *  @brief CDNA3 tier of multi-pattern search: the scans of every round over the DPP block scan of
 *      gfx942, and the @c _cdna3 exports, over the ROCm host side.
 *
 *  Every round scans its chunk counts, its sizing walk's slots and its cover, and a rewrite also
 *  scans each haystack's drift, each through @c sz_block_scan_simt_ , whose groups the ROCm tier
 *  scans through @c ds_bpermute . Here each 64-lane wavefront scans itself through DPP, through
 *  @c sz_block_scan_cdna3_ . The walks themselves take no scan, so every other launch runs this
 *  unit's own copies of the shared kernels.
 *
 *  @sa include/stringzilla/substrings/rocm.cuh
 */
#ifndef STRINGZILLA_SUBSTRINGS_CDNA3_CUH_
#define STRINGZILLA_SUBSTRINGS_CDNA3_CUH_

#include "stringzilla/types.h"

#if STRINGZILLA_ARCH_ROCM_
#if STRINGZILLA_ARCH_ROCM_CDNA3_

#include "stringzilla/substrings/rocm.cuh"
#include "stringzilla/strings/cdna3.cuh" // `sz_block_scan_bounded_cdna3_`, `sz_block_scan_cdna3_` & kin

#if defined(__cplusplus)
extern "C" {
#endif

#if STRINGZILLA_TARGET_CDNA3

#pragma region Entry Points

static __global__ void __launch_bounds__(sz_substrings_threads_per_block_simt_k)
    sz_substrings_scan_cdna3_kernel_(sz_size_t *values, sz_size_t count, sz_size_t const *live,
                                     sz_size_t elements_per_tile, sz_size_t *chain, sz_u64_t *states, sz_u64_t *stale) {
    sz_substrings_scan_tile_simt_(values, count, live, elements_per_tile, chain, states, stale, sz_block_scan_cdna3_,
                                  sz_block_sum_cdna3_, sz_chain_tile_simt_, sz_chain_look_back_rocm_);
}

static __global__ void __launch_bounds__(sz_substrings_threads_per_block_simt_k)
    sz_substrings_scan_bounded_cdna3_kernel_(sz_size_t *values, sz_size_t count, sz_size_t const *live,
                                             sz_size_t elements_per_tile, sz_size_t *chain, sz_u64_t *states,
                                             sz_u64_t *stale) {
    sz_substrings_scan_tile_simt_(values, count, live, elements_per_tile, chain, states, stale,
                                  sz_block_scan_bounded_cdna3_, sz_block_sum_cdna3_, sz_chain_tile_simt_,
                                  sz_chain_look_back_rocm_);
}

static __global__ void __launch_bounds__(sz_substrings_threads_per_block_simt_k)
    sz_substrings_scan_sized_cdna3_kernel_(sz_size_t *chunk_slots, sz_size_t count, sz_size_t const *chunks_count,
                                           sz_size_t elements_per_tile, sz_size_t *chain, sz_u64_t *states,
                                           sz_u64_t *stale, sz_size_t matches_budget, sz_bool_t emitting,
                                           sz_substrings_report_t *report) {
    sz_substrings_scan_sized_simt_(chunk_slots, count, chunks_count, elements_per_tile, chain, states, stale,
                                   matches_budget, emitting, report, sz_block_scan_cdna3_, sz_block_sum_cdna3_,
                                   sz_chain_tile_simt_, sz_chain_look_back_rocm_);
}

static __global__ void __launch_bounds__(sz_substrings_threads_per_block_simt_k)
    sz_substrings_rewrite_offsets_cdna3_kernel_(sz_strings_view_t haystacks, sz_strings_view_t replacements,
                                                sz_size_t const *haystack_offsets, sz_match_t const *matches,
                                                sz_substrings_report_t const *report, sz_size_t *gap_offsets,
                                                sz_size_t *output_sizes) {
    sz_substrings_rewrite_offsets_simt_(haystacks, replacements, haystack_offsets, matches, report, gap_offsets,
                                        output_sizes, sz_block_scan_cdna3_);
}

/** @ref sz_substrings_chunk_counts_simt_ . */
static __global__ void __launch_bounds__(sz_substrings_threads_per_block_simt_k)
    sz_substrings_chunk_counts_cdna3_kernel_(sz_substrings_engine_t engine, sz_strings_view_t haystacks,
                                             sz_size_t *chunk_offsets) {
    sz_substrings_chunk_counts_simt_(engine, haystacks, chunk_offsets);
}

static __global__ void __launch_bounds__(sz_substrings_threads_per_block_simt_k)
    sz_substrings_walk_cased_cdna3_kernel_(sz_substrings_engine_t engine, sz_u32_t staged_count,
                                           sz_strings_view_t haystacks, sz_size_t const *chunk_offsets,
                                           sz_size_t *chunk_slots) {
    sz_substrings_walk_chunks_simt_(&engine, sz_cased_k, sz_substrings_gpu_sizing_k, staged_count, haystacks,
                                    chunk_offsets, chunk_slots, 0, STRINGZILLA_NULL, STRINGZILLA_NULL, STRINGZILLA_NULL,
                                    0, STRINGZILLA_NULL, STRINGZILLA_NULL);
}

static __global__ void __launch_bounds__(sz_substrings_threads_per_block_simt_k)
    sz_substrings_walk_uncased_cdna3_kernel_(sz_substrings_engine_t engine, sz_u32_t staged_count,
                                             sz_strings_view_t haystacks, sz_size_t const *chunk_offsets,
                                             sz_size_t *chunk_slots) {
    sz_substrings_walk_chunks_simt_(&engine, sz_uncased_k, sz_substrings_gpu_sizing_k, staged_count, haystacks,
                                    chunk_offsets, chunk_slots, 0, STRINGZILLA_NULL, STRINGZILLA_NULL, STRINGZILLA_NULL,
                                    0, STRINGZILLA_NULL, STRINGZILLA_NULL);
}

static __global__ void __launch_bounds__(sz_substrings_threads_per_block_simt_k)
    sz_substrings_mark_cased_cdna3_kernel_(sz_substrings_engine_t engine, sz_u32_t staged_count,
                                           sz_strings_view_t haystacks, sz_size_t const *chunk_offsets,
                                           sz_size_t *chunk_slots) {
    sz_substrings_walk_chunks_simt_(&engine, sz_cased_k, sz_substrings_gpu_marking_k, staged_count, haystacks,
                                    chunk_offsets, chunk_slots, 0, STRINGZILLA_NULL, STRINGZILLA_NULL, STRINGZILLA_NULL,
                                    0, STRINGZILLA_NULL, STRINGZILLA_NULL);
}

static __global__ void __launch_bounds__(sz_substrings_threads_per_block_simt_k)
    sz_substrings_mark_uncased_cdna3_kernel_(sz_substrings_engine_t engine, sz_u32_t staged_count,
                                             sz_strings_view_t haystacks, sz_size_t const *chunk_offsets,
                                             sz_size_t *chunk_slots) {
    sz_substrings_walk_chunks_simt_(&engine, sz_uncased_k, sz_substrings_gpu_marking_k, staged_count, haystacks,
                                    chunk_offsets, chunk_slots, 0, STRINGZILLA_NULL, STRINGZILLA_NULL, STRINGZILLA_NULL,
                                    0, STRINGZILLA_NULL, STRINGZILLA_NULL);
}

static __global__ void __launch_bounds__(sz_substrings_threads_per_block_simt_k)
    sz_substrings_keep_cased_cdna3_kernel_(sz_substrings_engine_t engine, sz_u32_t staged_count,
                                           sz_strings_view_t haystacks, sz_size_t const *chunk_offsets,
                                           sz_size_t *chunk_slots, sz_size_t slots_count, sz_u32_t *kept,
                                           sz_substrings_spill_simt_t *spills, sz_size_t *spilled,
                                           sz_size_t spills_capacity) {
    sz_substrings_walk_chunks_simt_(&engine, sz_cased_k, sz_substrings_gpu_sizing_k, staged_count, haystacks,
                                    chunk_offsets, chunk_slots, slots_count, kept, spills, spilled, spills_capacity,
                                    STRINGZILLA_NULL, STRINGZILLA_NULL);
}

static __global__ void __launch_bounds__(sz_substrings_threads_per_block_simt_k)
    sz_substrings_keep_uncased_cdna3_kernel_(sz_substrings_engine_t engine, sz_u32_t staged_count,
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
    sz_substrings_gather_cdna3_kernel_(sz_substrings_engine_t engine, sz_strings_view_t haystacks,
                                       sz_size_t const *chunk_offsets, sz_substrings_report_t const *report,
                                       sz_size_t const *chunk_slots, sz_size_t slots_count, sz_u32_t const *kept,
                                       sz_substrings_spill_simt_t const *spills, sz_size_t const *spilled,
                                       sz_match_t *matches) {
    sz_substrings_gather_simt_(engine, haystacks, chunk_offsets, report, chunk_slots, slots_count, kept, spills,
                               spilled, matches);
}

/** @ref sz_substrings_bm25_pairs_simt_ . */
static __global__ void __launch_bounds__(sz_substrings_threads_per_block_simt_k)
    sz_substrings_bm25_pairs_cdna3_kernel_(sz_substrings_pairs_simt_t pairs, sz_strings_view_t haystacks,
                                           sz_f32_t const *document_lengths, sz_substrings_bm25_t parameters,
                                           sz_f32_t const *needle_weights, unsigned long long *sums) {
    sz_substrings_bm25_pairs_simt_(pairs, haystacks, document_lengths, parameters, needle_weights, sums);
}

/** @ref sz_substrings_link_simt_ . */
static __global__ void __launch_bounds__(sz_substrings_threads_per_block_simt_k)
    sz_substrings_link_cdna3_kernel_(sz_match_t const *matches, sz_substrings_report_t const *report, sz_size_t longest,
                                     sz_substrings_overlap_policy_t policy, sz_size_t *jumps, sz_size_t *keep) {
    sz_substrings_link_simt_(matches, report, longest, policy, jumps, keep);
}

/** @ref sz_substrings_double_simt_ . */
static __global__ void __launch_bounds__(sz_substrings_threads_per_block_simt_k)
    sz_substrings_double_cdna3_kernel_(sz_substrings_report_t const *report, sz_size_t const *jumps, sz_size_t *doubled,
                                       sz_size_t *keep) {
    sz_substrings_double_simt_(report, jumps, doubled, keep);
}

/** @ref sz_substrings_compact_simt_ . */
static __global__ void __launch_bounds__(sz_substrings_threads_per_block_simt_k)
    sz_substrings_compact_cdna3_kernel_(sz_match_t const *matches, sz_substrings_report_t *report,
                                        sz_size_t const *keep_offsets, sz_match_t *survivors) {
    sz_substrings_compact_simt_(matches, report, keep_offsets, survivors);
}

/** @ref sz_substrings_haystack_offsets_simt_ . */
static __global__ void __launch_bounds__(sz_substrings_threads_per_block_simt_k)
    sz_substrings_haystack_offsets_cdna3_kernel_(sz_size_t const *chunk_offsets, sz_size_t const *chunk_slots,
                                                 sz_size_t const *keep_offsets, sz_substrings_report_t const *report,
                                                 sz_size_t *haystack_offsets, sz_size_t count) {
    sz_substrings_haystack_offsets_simt_(chunk_offsets, chunk_slots, keep_offsets, report, haystack_offsets, count);
}

/** @ref sz_substrings_counts_simt_ . */
static __global__ void __launch_bounds__(sz_substrings_threads_per_block_simt_k)
    sz_substrings_counts_cdna3_kernel_(sz_size_t const *haystack_offsets, sz_size_t *counts, sz_size_t counts_stride,
                                       sz_size_t count) {
    sz_substrings_counts_simt_(haystack_offsets, counts, counts_stride, count);
}

/** @ref sz_substrings_store_matches_simt_ . */
static __global__ void __launch_bounds__(sz_substrings_threads_per_block_simt_k)
    sz_substrings_store_matches_cdna3_kernel_(sz_match_t const *reported, sz_substrings_report_t const *report,
                                              sz_match_t *matches, sz_size_t matches_capacity) {
    sz_substrings_store_matches_simt_(reported, report, matches, matches_capacity);
}

/** @ref sz_substrings_stored_simt_ . */
static __global__ void __launch_bounds__(sz_substrings_threads_per_block_simt_k)
    sz_substrings_stored_cdna3_kernel_(sz_substrings_report_t *report, sz_size_t matches_capacity) {
    sz_substrings_stored_simt_(report, matches_capacity);
}

/** @ref sz_substrings_rewrite_copy_simt_ . */
static __global__ void __launch_bounds__(sz_substrings_threads_per_block_simt_k)
    sz_substrings_rewrite_copy_cdna3_kernel_(sz_strings_view_t haystacks, sz_strings_view_t replacements,
                                             sz_size_t const *haystack_offsets, sz_match_t const *matches,
                                             sz_size_t const *gap_offsets, sz_size_t const *output_offsets,
                                             sz_substrings_report_t const *report, sz_ptr_t output) {
    sz_substrings_rewrite_copy_simt_(haystacks, replacements, haystack_offsets, matches, gap_offsets, output_offsets,
                                     report, output);
}

/** @ref sz_substrings_target_simt_ . */
static __global__ void __launch_bounds__(sz_substrings_threads_per_block_simt_k)
    sz_substrings_target_cdna3_kernel_(sz_size_t const *rewritten_at, sz_size_t target_ceiling,
                                       sz_substrings_report_t *report) {
    sz_substrings_target_simt_(rewritten_at, target_ceiling, report);
}

/** One lane per haystack marks whether its boundaries enclose a match, with HIP's ballot. */
static __global__ void __launch_bounds__(sz_substrings_threads_per_block_simt_k)
    sz_substrings_contains_cdna3_kernel_(sz_size_t const *haystack_offsets, sz_u1x8_t *contained, sz_size_t count) {
    sz_substrings_contains_simt_(haystack_offsets, contained, count, sz_lanes_ballot_rocm_);
}

/** @ref sz_substrings_bm25_simt_ over ROCm's 32-lane halves of wavefronts. */
static __global__ void __launch_bounds__(sz_substrings_threads_per_block_simt_k)
    sz_substrings_bm25_cdna3_kernel_(sz_substrings_engine_t engine, sz_strings_view_t haystacks,
                                     sz_f32_t const *document_lengths, sz_substrings_bm25_t parameters,
                                     sz_f32_t const *needle_weights, sz_u32_t *overflow_rows, sz_f32_t *scores,
                                     sz_size_t scores_stride, sz_u32_t staged_accepts_words, sz_u32_t staged_count,
                                     unsigned long long const *sums, sz_size_t const *overflowed) {
    sz_substrings_bm25_simt_(engine, haystacks, document_lengths, parameters, needle_weights, overflow_rows, scores,
                             scores_stride, staged_accepts_words, staged_count, sums, overflowed,
                             sz_u64_lanes_sum_rocm_);
}

/** The cased pairs walk, counting each pair with its own atomic. */
static __global__ void __launch_bounds__(sz_substrings_threads_per_block_simt_k)
    sz_substrings_pairs_cased_cdna3_kernel_(sz_substrings_engine_t engine, sz_u32_t staged_count,
                                            sz_strings_view_t haystacks, sz_size_t const *chunk_offsets,
                                            sz_substrings_pairs_simt_t pairs) {
    sz_substrings_walk_chunks_simt_(&engine, sz_cased_k, sz_substrings_gpu_tallying_k, staged_count, haystacks,
                                    chunk_offsets, STRINGZILLA_NULL, 0, STRINGZILLA_NULL, STRINGZILLA_NULL,
                                    STRINGZILLA_NULL, 0, &pairs, sz_u32_increment_rocm_);
}

/** The uncased pairs walk, counting each pair with its own atomic. */
static __global__ void __launch_bounds__(sz_substrings_threads_per_block_simt_k)
    sz_substrings_pairs_uncased_cdna3_kernel_(sz_substrings_engine_t engine, sz_u32_t staged_count,
                                              sz_strings_view_t haystacks, sz_size_t const *chunk_offsets,
                                              sz_substrings_pairs_simt_t pairs) {
    sz_substrings_walk_chunks_simt_(&engine, sz_uncased_k, sz_substrings_gpu_tallying_k, staged_count, haystacks,
                                    chunk_offsets, STRINGZILLA_NULL, 0, STRINGZILLA_NULL, STRINGZILLA_NULL,
                                    STRINGZILLA_NULL, 0, &pairs, sz_u32_increment_rocm_);
}

/** The CDNA3 tier's entry points, scanning through DPP. */
static sz_substrings_entry_points_rocm_t const sz_substrings_entry_points_cdna3_ = {
    (void const *)sz_substrings_scan_cdna3_kernel_,
    (void const *)sz_substrings_scan_bounded_cdna3_kernel_,
    (void const *)sz_substrings_scan_sized_cdna3_kernel_,
    (void const *)sz_substrings_rewrite_offsets_cdna3_kernel_,
    (void const *)sz_substrings_chunk_counts_cdna3_kernel_,
    (void const *)sz_substrings_walk_cased_cdna3_kernel_,
    (void const *)sz_substrings_walk_uncased_cdna3_kernel_,
    (void const *)sz_substrings_mark_cased_cdna3_kernel_,
    (void const *)sz_substrings_mark_uncased_cdna3_kernel_,
    (void const *)sz_substrings_keep_cased_cdna3_kernel_,
    (void const *)sz_substrings_keep_uncased_cdna3_kernel_,
    (void const *)sz_substrings_gather_cdna3_kernel_,
    (void const *)sz_substrings_link_cdna3_kernel_,
    (void const *)sz_substrings_double_cdna3_kernel_,
    (void const *)sz_substrings_compact_cdna3_kernel_,
    (void const *)sz_substrings_haystack_offsets_cdna3_kernel_,
    (void const *)sz_substrings_counts_cdna3_kernel_,
    (void const *)sz_substrings_store_matches_cdna3_kernel_,
    (void const *)sz_substrings_stored_cdna3_kernel_,
    (void const *)sz_substrings_rewrite_copy_cdna3_kernel_,
    (void const *)sz_substrings_target_cdna3_kernel_,
    (void const *)sz_substrings_contains_cdna3_kernel_,
    (void const *)sz_substrings_pairs_cased_cdna3_kernel_,
    (void const *)sz_substrings_pairs_uncased_cdna3_kernel_,
    (void const *)sz_substrings_bm25_pairs_cdna3_kernel_,
    (void const *)sz_substrings_bm25_cdna3_kernel_,
    sz_cap_cdna3_k,
};

#pragma endregion Entry Points

STRINGZILLA_API sz_status_t sz_substrings_engine_init_cdna3(
    sz_substrings_engine_t *engine, sz_strings_view_t const *needles, sz_case_sensitivity_t case_sensitivity,
    sz_substrings_overlap_policy_t overlap_policy, sz_size_t hot_states, sz_size_t matches_budget,
    sz_size_t haystacks_budget, sz_allocator_t *allocator, sz_stream_t stream) {
    return sz_substrings_engine_init_scoped_rocm_(engine, needles, case_sensitivity, overlap_policy, hot_states,
                                                  matches_budget, haystacks_budget, &sz_substrings_entry_points_cdna3_,
                                                  allocator, stream);
}

STRINGZILLA_API sz_status_t sz_substrings_contains_cdna3(sz_substrings_engine_t *engine,
                                                         sz_strings_view_t const *haystacks, sz_u1x8_t *contained,
                                                         sz_stream_t stream) {
    return sz_substrings_contains_scoped_rocm_(engine, haystacks, contained, &sz_substrings_entry_points_cdna3_,
                                               stream);
}

STRINGZILLA_API sz_status_t sz_substrings_counts_cdna3(sz_substrings_engine_t *engine,
                                                       sz_strings_view_t const *haystacks, sz_size_t *counts,
                                                       sz_size_t counts_stride, sz_stream_t stream) {
    return sz_substrings_counts_scoped_rocm_(engine, haystacks, counts, counts_stride,
                                             &sz_substrings_entry_points_cdna3_, stream);
}

STRINGZILLA_API sz_status_t sz_substrings_find_cdna3(sz_substrings_engine_t *engine, sz_strings_view_t const *haystacks,
                                                     sz_match_t *matches, sz_size_t matches_capacity,
                                                     sz_size_t *matches_offsets, sz_stream_t stream) {
    return sz_substrings_find_scoped_rocm_(engine, haystacks, matches, matches_capacity, matches_offsets,
                                           &sz_substrings_entry_points_cdna3_, stream);
}

STRINGZILLA_API sz_status_t sz_substrings_replace_cdna3(sz_substrings_engine_t *engine,
                                                        sz_strings_view_t const *haystacks,
                                                        sz_strings_view_t const *replacements, sz_ptr_t target,
                                                        sz_size_t target_capacity, sz_size_t *offsets,
                                                        sz_stream_t stream) {
    return sz_substrings_replace_scoped_rocm_(engine, haystacks, replacements, target, target_capacity, offsets,
                                              &sz_substrings_entry_points_cdna3_, stream);
}

STRINGZILLA_API sz_status_t sz_substrings_bm25_scores_cdna3(sz_substrings_engine_t *engine,
                                                            sz_strings_view_t const *haystacks,
                                                            sz_f32_t const *document_lengths,
                                                            sz_substrings_bm25_t const *parameters,
                                                            sz_f32_t const *needle_weights, sz_f32_t *scores,
                                                            sz_size_t scores_stride, sz_stream_t stream) {
    return sz_substrings_bm25_scores_scoped_rocm_(engine, haystacks, document_lengths, parameters, needle_weights,
                                                  scores, scores_stride, &sz_substrings_entry_points_cdna3_, stream);
}

#endif // STRINGZILLA_TARGET_CDNA3

#if defined(__cplusplus)
} // extern "C"
#endif
#endif // STRINGZILLA_ARCH_ROCM_CDNA3_
#endif // STRINGZILLA_ARCH_ROCM_
#endif // STRINGZILLA_SUBSTRINGS_CDNA3_CUH_
