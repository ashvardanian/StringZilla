/**
 *  @file include/stringzilla/regex/hopper.cuh
 *  @author Ash Vardanian
 *  @date November 15, 2026
 *  @brief Hopper tier of regular expressions: the scans looking back once per cluster, and the
 *      @c _hopper exports, over the CUDA host side.
 *
 *  From compute capability 9.0 a grid's blocks group into clusters, each block addressing the
 *  others' shared memory. A scan's first block of each cluster takes one ticket for all of them,
 *  the blocks hand it their tile totals, and it looks back once over their aggregate, so the chain
 *  of look-backs is a cluster's blocks shorter. The start-marking walk is the Ampere tier's.
 *
 *  @sa include/stringzilla/regex/ampere.cuh
 *  @sa include/stringzilla/cuda.cuh
 */
#ifndef STRINGZILLA_REGEX_HOPPER_CUH_
#define STRINGZILLA_REGEX_HOPPER_CUH_

#include "stringzilla/types.h"

#if STRINGZILLA_ARCH_CUDA_
#if STRINGZILLA_ARCH_CUDA_HOPPER_

#include "stringzilla/regex/ampere.cuh"
#include "stringzilla/strings/hopper.cuh" // `sz_chain_look_back_hopper_`, `sz_chain_tile_hopper_`

#if defined(__cplusplus)
extern "C" {
#endif

#if STRINGZILLA_TARGET_HOPPER

/** @ref sz_regex_scan_tile_simt_ in 32 bits, looking back once per cluster; launched only in
 *  clusters of @c sz_cluster_blocks_cuda_k blocks. */
static __global__ void __launch_bounds__(sz_regex_threads_per_block_simt_k)
    sz_regex_scan_bounded_hopper_kernel_(sz_size_t *values, sz_size_t count, sz_size_t elements_per_tile,
                                         sz_size_t *chain, sz_u64_t *states, sz_u64_t *stale) {
    sz_regex_scan_tile_simt_(values, count, elements_per_tile, chain, states, stale, sz_block_scan_bounded_cuda_,
                             sz_block_sum_cuda_, sz_chain_tile_hopper_, sz_chain_look_back_hopper_);
}

/** @ref sz_regex_scan_tile_simt_ looking back once per cluster; launched only in clusters of
 *  @c sz_cluster_blocks_cuda_k blocks. */
static __global__ void __launch_bounds__(sz_regex_threads_per_block_simt_k)
    sz_regex_scan_hopper_kernel_(sz_size_t *values, sz_size_t count, sz_size_t elements_per_tile, sz_size_t *chain,
                                 sz_u64_t *states, sz_u64_t *stale) {
    sz_regex_scan_tile_simt_(values, count, elements_per_tile, chain, states, stale, sz_block_scan_cuda_,
                             sz_block_sum_cuda_, sz_chain_tile_hopper_, sz_chain_look_back_hopper_);
}

/** The warp walk of the links that outlived their first thread, with CUDA's cross-lane steps. */
static __global__ void __launch_bounds__(sz_regex_threads_per_block_simt_k)
    sz_regex_continue_hopper_kernel_(sz_regex_engine_t engine, sz_strings_view_t haystacks,
                                     sz_regex_round_simt_t round) {
    sz_regex_continue_simt_(&engine, haystacks, &round, sz_false_k, sz_shuffle_down_cuda_, sz_lanes_broadcast_cuda_,
                            sz_lanes_any_cuda_);
}

/** The warp walk of the folded links that outlived their first thread, with CUDA's cross-lane
 *  steps. */
static __global__ void __launch_bounds__(sz_regex_threads_per_block_simt_k)
    sz_regex_continue_folded_hopper_kernel_(sz_regex_engine_t engine, sz_strings_view_t haystacks,
                                            sz_regex_round_simt_t round) {
    sz_regex_continue_simt_(&engine, haystacks, &round, sz_true_k, sz_shuffle_down_cuda_, sz_lanes_broadcast_cuda_,
                            sz_lanes_any_cuda_);
}

/** @ref sz_regex_bound_simt_ over the pass's NFA or positions. */
static __global__ void __launch_bounds__(sz_regex_threads_per_block_simt_k)
    sz_regex_bound_hopper_kernel_(sz_regex_engine_t engine, sz_strings_view_t haystacks, sz_regex_round_simt_t round) {
    sz_regex_bound_simt_(&engine, haystacks, &round, sz_false_k);
}

/** @ref sz_regex_bound_simt_ over a body's DFA. */
static __global__ void __launch_bounds__(sz_regex_threads_per_block_simt_k)
    sz_regex_bound_dfa_hopper_kernel_(sz_regex_engine_t engine, sz_strings_view_t haystacks,
                                      sz_regex_round_simt_t round) {
    sz_regex_bound_simt_(&engine, haystacks, &round, sz_true_k);
}

/** @ref sz_regex_bound_pieces_simt_ over the pass's NFA or positions. */
static __global__ void __launch_bounds__(sz_regex_threads_per_block_simt_k)
    sz_regex_bound_pieces_hopper_kernel_(sz_regex_engine_t engine, sz_strings_view_t haystacks,
                                         sz_regex_round_simt_t round) {
    sz_regex_bound_pieces_simt_(&engine, haystacks, &round, sz_false_k);
}

/** @ref sz_regex_bound_pieces_simt_ over a body's DFA. */
static __global__ void __launch_bounds__(sz_regex_threads_per_block_simt_k)
    sz_regex_bound_pieces_dfa_hopper_kernel_(sz_regex_engine_t engine, sz_strings_view_t haystacks,
                                             sz_regex_round_simt_t round) {
    sz_regex_bound_pieces_simt_(&engine, haystacks, &round, sz_true_k);
}

/** @ref sz_regex_affine_simt_ over the pass's NFA or positions. */
static __global__ void __launch_bounds__(sz_regex_threads_per_block_simt_k)
    sz_regex_affine_hopper_kernel_(sz_regex_engine_t engine, sz_strings_view_t haystacks, sz_regex_round_simt_t round) {
    sz_regex_affine_simt_(&engine, haystacks, &round, sz_false_k);
}

/** @ref sz_regex_affine_simt_ over a body's DFA. */
static __global__ void __launch_bounds__(sz_regex_threads_per_block_simt_k)
    sz_regex_affine_dfa_hopper_kernel_(sz_regex_engine_t engine, sz_strings_view_t haystacks,
                                       sz_regex_round_simt_t round) {
    sz_regex_affine_simt_(&engine, haystacks, &round, sz_true_k);
}

/** @ref sz_regex_affine_double_simt_ . */
static __global__ void __launch_bounds__(sz_regex_threads_per_block_simt_k)
    sz_regex_affine_double_hopper_kernel_(sz_regex_round_simt_t round) {
    sz_regex_affine_double_simt_(round);
}

/** @ref sz_regex_looks_simt_ over the pass's NFA or positions. */
static __global__ void __launch_bounds__(sz_regex_threads_per_block_simt_k)
    sz_regex_looks_hopper_kernel_(sz_regex_engine_t engine, sz_strings_view_t haystacks, sz_regex_round_simt_t round) {
    sz_regex_looks_simt_(&engine, haystacks, &round, sz_false_k);
}

/** @ref sz_regex_looks_simt_ over a body's DFA. */
static __global__ void __launch_bounds__(sz_regex_threads_per_block_simt_k)
    sz_regex_looks_dfa_hopper_kernel_(sz_regex_engine_t engine, sz_strings_view_t haystacks,
                                      sz_regex_round_simt_t round) {
    sz_regex_looks_simt_(&engine, haystacks, &round, sz_true_k);
}

/** @ref sz_regex_fill_singles_simt_ . */
static __global__ void __launch_bounds__(sz_regex_threads_per_block_simt_k)
    sz_regex_singles_hopper_kernel_(sz_regex_engine_t engine, sz_strings_view_t haystacks,
                                    sz_regex_round_simt_t round) {
    sz_regex_fill_singles_simt_(engine, haystacks, round);
}

/** @ref sz_regex_prime_chunks_simt_ . */
static __global__ void __launch_bounds__(sz_regex_threads_per_block_simt_k)
    sz_regex_prime_hopper_kernel_(sz_regex_engine_t engine, sz_strings_view_t haystacks, sz_regex_round_simt_t round) {
    sz_regex_prime_chunks_simt_(engine, haystacks, round);
}

/** @ref sz_regex_prime_pieces_simt_ . */
static __global__ void __launch_bounds__(sz_regex_threads_per_block_simt_k)
    sz_regex_prime_pieces_hopper_kernel_(sz_regex_engine_t engine, sz_strings_view_t haystacks,
                                         sz_regex_round_simt_t round) {
    sz_regex_prime_pieces_simt_(engine, haystacks, round);
}

/** @ref sz_regex_compose_simt_ . */
static __global__ void __launch_bounds__(sz_regex_threads_per_block_simt_k)
    sz_regex_compose_hopper_kernel_(sz_regex_engine_t engine, sz_strings_view_t haystacks,
                                    sz_regex_round_simt_t round) {
    sz_regex_compose_simt_(engine, haystacks, round);
}

/** @ref sz_regex_compose_double_simt_ . */
static __global__ void __launch_bounds__(sz_regex_threads_per_block_simt_k)
    sz_regex_compose_double_hopper_kernel_(sz_regex_round_simt_t round) {
    sz_regex_compose_double_simt_(round);
}

/** @ref sz_regex_mark_prefixes_simt_ . */
static __global__ void __launch_bounds__(sz_regex_threads_per_block_simt_k)
    sz_regex_mark_prefixes_hopper_kernel_(sz_regex_engine_t engine, sz_strings_view_t haystacks,
                                          sz_regex_round_simt_t round) {
    sz_regex_mark_prefixes_simt_(engine, haystacks, round);
}

/** @ref sz_regex_mark_folded_simt_ . */
static __global__ void __launch_bounds__(sz_regex_threads_per_block_simt_k)
    sz_regex_mark_folded_hopper_kernel_(sz_regex_engine_t engine, sz_strings_view_t haystacks,
                                        sz_regex_round_simt_t round) {
    sz_regex_mark_folded_simt_(engine, haystacks, round);
}

/** @ref sz_regex_mark_positions_simt_ over bytes. */
static __global__ void __launch_bounds__(sz_regex_threads_per_block_simt_k)
    sz_regex_mark_positions_hopper_kernel_(sz_regex_engine_t engine, sz_strings_view_t haystacks,
                                           sz_regex_round_simt_t round) {
    sz_regex_mark_positions_simt_(&engine, haystacks, &round, sz_false_k);
}

/** @ref sz_regex_mark_positions_simt_ over folded runes. */
static __global__ void __launch_bounds__(sz_regex_threads_per_block_simt_k)
    sz_regex_mark_positions_folded_hopper_kernel_(sz_regex_engine_t engine, sz_strings_view_t haystacks,
                                                  sz_regex_round_simt_t round) {
    sz_regex_mark_positions_simt_(&engine, haystacks, &round, sz_true_k);
}

/** @ref sz_regex_list_simt_ . */
static __global__ void __launch_bounds__(sz_regex_threads_per_block_simt_k)
    sz_regex_list_hopper_kernel_(sz_regex_round_simt_t round) {
    sz_regex_list_simt_(round);
}

/** @ref sz_regex_ends_pike_simt_ . */
static __global__ void __launch_bounds__(sz_regex_threads_per_block_simt_k)
    sz_regex_ends_pike_hopper_kernel_(sz_regex_engine_t engine, sz_strings_view_t haystacks,
                                      sz_regex_round_simt_t round) {
    sz_regex_ends_pike_simt_(engine, haystacks, round);
}

/** @ref sz_regex_ends_simt_ . */
static __global__ void __launch_bounds__(sz_regex_threads_per_block_simt_k)
    sz_regex_ends_hopper_kernel_(sz_regex_engine_t engine, sz_strings_view_t haystacks, sz_regex_round_simt_t round) {
    sz_regex_ends_simt_(engine, haystacks, round);
}

/** @ref sz_regex_links_simt_ . */
static __global__ void __launch_bounds__(sz_regex_threads_per_block_simt_k)
    sz_regex_links_hopper_kernel_(sz_regex_engine_t engine, sz_strings_view_t haystacks, sz_regex_round_simt_t round) {
    sz_regex_links_simt_(engine, haystacks, round);
}

/** @ref sz_regex_resume_simt_ . */
static __global__ void __launch_bounds__(sz_regex_threads_per_block_simt_k)
    sz_regex_resume_hopper_kernel_(sz_regex_engine_t engine, sz_strings_view_t haystacks, sz_regex_round_simt_t round) {
    sz_regex_resume_simt_(engine, haystacks, round);
}

/** @ref sz_regex_links_double_simt_ . */
static __global__ void __launch_bounds__(sz_regex_threads_per_block_simt_k)
    sz_regex_links_double_hopper_kernel_(sz_regex_round_simt_t round) {
    sz_regex_links_double_simt_(round);
}

/** @ref sz_regex_group_simt_ . */
static __global__ void __launch_bounds__(sz_regex_threads_per_block_simt_k)
    sz_regex_group_hopper_kernel_(sz_regex_engine_t engine, sz_regex_round_simt_t round) {
    sz_regex_group_simt_(engine, round);
}

/** @ref sz_regex_point_simt_ . */
static __global__ void __launch_bounds__(sz_regex_threads_per_block_simt_k)
    sz_regex_point_hopper_kernel_(sz_regex_engine_t engine, sz_regex_round_simt_t round) {
    sz_regex_point_simt_(engine, round);
}

/** @ref sz_regex_groups_double_simt_ . */
static __global__ void __launch_bounds__(sz_regex_threads_per_block_simt_k)
    sz_regex_groups_double_hopper_kernel_(sz_regex_round_simt_t round) {
    sz_regex_groups_double_simt_(round);
}

/** @ref sz_regex_take_simt_ . */
static __global__ void __launch_bounds__(sz_regex_threads_per_block_simt_k)
    sz_regex_take_hopper_kernel_(sz_regex_round_simt_t round) {
    sz_regex_take_simt_(round);
}

/** @ref sz_regex_retry_folded_simt_ . */
static __global__ void __launch_bounds__(sz_regex_threads_per_block_simt_k)
    sz_regex_retry_folded_hopper_kernel_(sz_regex_engine_t engine, sz_strings_view_t haystacks,
                                         sz_regex_round_simt_t round) {
    sz_regex_retry_folded_simt_(engine, haystacks, round);
}

/** @ref sz_regex_chain_simt_ . */
static __global__ void __launch_bounds__(sz_regex_threads_per_block_simt_k)
    sz_regex_chain_hopper_kernel_(sz_strings_view_t haystacks, sz_regex_round_simt_t round) {
    sz_regex_chain_simt_(haystacks, round);
}

/** @ref sz_regex_double_simt_ . */
static __global__ void __launch_bounds__(sz_regex_threads_per_block_simt_k)
    sz_regex_double_hopper_kernel_(sz_regex_round_simt_t round) {
    sz_regex_double_simt_(round);
}

/** @ref sz_regex_sweep_simt_ . */
static __global__ void __launch_bounds__(sz_regex_threads_per_block_simt_k)
    sz_regex_sweep_hopper_kernel_(sz_strings_view_t haystacks, sz_regex_round_simt_t round) {
    sz_regex_sweep_simt_(haystacks, round);
}

/** @ref sz_regex_carry_simt_ . */
static __global__ void __launch_bounds__(sz_regex_carry_threads_per_block_simt_k)
    sz_regex_carry_hopper_kernel_(sz_strings_view_t haystacks, sz_regex_round_simt_t round) {
    sz_regex_carry_simt_(haystacks, round);
}

/** @ref sz_regex_counts_edges_simt_ . */
static __global__ void __launch_bounds__(sz_regex_threads_per_block_simt_k)
    sz_regex_counts_edges_hopper_kernel_(sz_size_t const *counts, sz_size_t counts_stride, sz_size_t count,
                                         sz_size_t *edges) {
    sz_regex_counts_edges_simt_(counts, counts_stride, count, edges);
}

/** @ref sz_regex_counts_simt_ . */
static __global__ void __launch_bounds__(sz_regex_threads_per_block_simt_k)
    sz_regex_counts_hopper_kernel_(sz_size_t *counts, sz_size_t counts_stride, sz_size_t count,
                                   sz_size_t const *edges) {
    sz_regex_counts_simt_(counts, counts_stride, count, edges);
}

/** @ref sz_regex_expand_kept_simt_ . */
static __global__ void __launch_bounds__(sz_regex_threads_per_block_simt_k)
    sz_regex_expand_hopper_kernel_(sz_regex_engine_t engine, sz_strings_view_t haystacks, sz_regex_round_simt_t round) {
    sz_regex_expand_kept_simt_(engine, haystacks, round);
}

/** @ref sz_regex_drift_simt_ . */
static __global__ void __launch_bounds__(sz_regex_threads_per_block_simt_k)
    sz_regex_drift_hopper_kernel_(sz_regex_round_simt_t round) {
    sz_regex_drift_simt_(round);
}

/** @ref sz_regex_rewrite_simt_ . */
static __global__ void __launch_bounds__(sz_regex_threads_per_block_simt_k)
    sz_regex_rewrite_hopper_kernel_(sz_regex_engine_t engine, sz_strings_view_t haystacks,
                                    sz_regex_round_simt_t round) {
    sz_regex_rewrite_simt_(engine, haystacks, round);
}

/** @ref sz_regex_firsts_simt_ . */
static __global__ void __launch_bounds__(sz_regex_threads_per_block_simt_k)
    sz_regex_firsts_hopper_kernel_(sz_regex_engine_t engine, sz_strings_view_t haystacks, sz_regex_round_simt_t round) {
    sz_regex_firsts_simt_(engine, haystacks, round);
}

/** @ref sz_regex_rows_simt_ . */
static __global__ void __launch_bounds__(sz_regex_threads_per_block_simt_k)
    sz_regex_rows_hopper_kernel_(sz_regex_engine_t engine, sz_strings_view_t haystacks, sz_regex_round_simt_t round) {
    sz_regex_rows_simt_(engine, haystacks, round);
}

/** @ref sz_regex_mark_ampere_ . */
static __global__ void __launch_bounds__(sz_regex_threads_per_block_simt_k)
    sz_regex_mark_hopper_kernel_(sz_regex_engine_t engine, sz_strings_view_t haystacks, sz_regex_round_simt_t round) {
    sz_regex_mark_ampere_(engine, haystacks, round);
}

/** The Hopper tier's entry points, the Ampere tier's marking walk over scans clustered by eight. */
static sz_regex_entry_points_cuda_t const sz_regex_entry_points_hopper_ = {
    (void const *)sz_regex_mark_hopper_kernel_,
    (void const *)sz_regex_scan_hopper_kernel_,
    (void const *)sz_regex_scan_bounded_hopper_kernel_,
    (void const *)sz_regex_bound_hopper_kernel_,
    (void const *)sz_regex_bound_dfa_hopper_kernel_,
    (void const *)sz_regex_bound_pieces_hopper_kernel_,
    (void const *)sz_regex_bound_pieces_dfa_hopper_kernel_,
    (void const *)sz_regex_affine_hopper_kernel_,
    (void const *)sz_regex_affine_dfa_hopper_kernel_,
    (void const *)sz_regex_affine_double_hopper_kernel_,
    (void const *)sz_regex_looks_hopper_kernel_,
    (void const *)sz_regex_looks_dfa_hopper_kernel_,
    (void const *)sz_regex_singles_hopper_kernel_,
    (void const *)sz_regex_prime_hopper_kernel_,
    (void const *)sz_regex_prime_pieces_hopper_kernel_,
    (void const *)sz_regex_compose_hopper_kernel_,
    (void const *)sz_regex_compose_double_hopper_kernel_,
    (void const *)sz_regex_mark_prefixes_hopper_kernel_,
    (void const *)sz_regex_mark_folded_hopper_kernel_,
    (void const *)sz_regex_mark_positions_hopper_kernel_,
    (void const *)sz_regex_mark_positions_folded_hopper_kernel_,
    (void const *)sz_regex_list_hopper_kernel_,
    (void const *)sz_regex_ends_pike_hopper_kernel_,
    (void const *)sz_regex_ends_hopper_kernel_,
    (void const *)sz_regex_links_hopper_kernel_,
    (void const *)sz_regex_continue_hopper_kernel_,
    (void const *)sz_regex_continue_folded_hopper_kernel_,
    (void const *)sz_regex_resume_hopper_kernel_,
    (void const *)sz_regex_links_double_hopper_kernel_,
    (void const *)sz_regex_group_hopper_kernel_,
    (void const *)sz_regex_point_hopper_kernel_,
    (void const *)sz_regex_groups_double_hopper_kernel_,
    (void const *)sz_regex_take_hopper_kernel_,
    (void const *)sz_regex_retry_folded_hopper_kernel_,
    (void const *)sz_regex_chain_hopper_kernel_,
    (void const *)sz_regex_double_hopper_kernel_,
    (void const *)sz_regex_sweep_hopper_kernel_,
    (void const *)sz_regex_carry_hopper_kernel_,
    (void const *)sz_regex_counts_edges_hopper_kernel_,
    (void const *)sz_regex_counts_hopper_kernel_,
    (void const *)sz_regex_expand_hopper_kernel_,
    (void const *)sz_regex_drift_hopper_kernel_,
    (void const *)sz_regex_rewrite_hopper_kernel_,
    (void const *)sz_regex_firsts_hopper_kernel_,
    (void const *)sz_regex_rows_hopper_kernel_,
    sz_cluster_blocks_cuda_k,
    sz_cap_hopper_k,
};

STRINGZILLA_API sz_status_t sz_regex_engine_init_hopper(sz_regex_engine_t *engine, sz_strings_view_t const *patterns,
                                                        sz_case_sensitivity_t case_sensitivity,
                                                        sz_size_t *malformed_index, sz_allocator_t *allocator,
                                                        sz_stream_t stream) {
    int caller = 0;
    sz_status_t status = sz_device_enter_cuda_(stream, &caller);
    if (status != sz_success_k) return status;
    status = sz_regex_engine_init_scoped_cuda_(engine, patterns, case_sensitivity, malformed_index,
                                               &sz_regex_entry_points_hopper_, allocator, stream);
    sz_device_leave_cuda_(caller);
    return status;
}

STRINGZILLA_API sz_status_t sz_regex_contains_hopper(sz_regex_engine_t *engine, sz_strings_view_t const *haystacks,
                                                     sz_u1x8_t *contained, sz_stream_t stream) {
    int caller = 0;
    sz_status_t status = sz_device_enter_cuda_(stream, &caller);
    if (status != sz_success_k) return status;
    status = sz_regex_contains_scoped_cuda_(engine, haystacks, contained, &sz_regex_entry_points_hopper_, stream);
    sz_device_leave_cuda_(caller);
    return status;
}

STRINGZILLA_API sz_status_t sz_regex_counts_hopper(sz_regex_engine_t *engine, sz_strings_view_t const *haystacks,
                                                   sz_size_t *counts, sz_size_t counts_stride, sz_stream_t stream) {
    int caller = 0;
    sz_status_t status = sz_device_enter_cuda_(stream, &caller);
    if (status != sz_success_k) return status;
    status = sz_regex_counts_scoped_cuda_(engine, haystacks, counts, counts_stride, &sz_regex_entry_points_hopper_,
                                          stream);
    sz_device_leave_cuda_(caller);
    return status;
}

STRINGZILLA_API sz_status_t sz_regex_find_hopper(sz_regex_engine_t *engine, sz_strings_view_t const *haystacks,
                                                 sz_match_t *matches, sz_size_t matches_capacity,
                                                 sz_size_t *matches_offsets, sz_stream_t stream) {
    int caller = 0;
    sz_status_t status = sz_device_enter_cuda_(stream, &caller);
    if (status != sz_success_k) return status;
    status = sz_regex_find_scoped_cuda_(engine, haystacks, matches, matches_capacity, matches_offsets,
                                        &sz_regex_entry_points_hopper_, stream);
    sz_device_leave_cuda_(caller);
    return status;
}

STRINGZILLA_API sz_status_t sz_regex_replace_hopper(sz_regex_engine_t *engine, sz_strings_view_t const *haystacks,
                                                    sz_strings_view_t const *replacements, sz_ptr_t target,
                                                    sz_size_t target_capacity, sz_size_t *offsets, sz_stream_t stream) {
    int caller = 0;
    sz_status_t status = sz_device_enter_cuda_(stream, &caller);
    if (status != sz_success_k) return status;
    status = sz_regex_replace_scoped_cuda_(engine, haystacks, replacements, target, target_capacity, offsets,
                                           &sz_regex_entry_points_hopper_, stream);
    sz_device_leave_cuda_(caller);
    return status;
}

STRINGZILLA_API sz_status_t sz_regex_extract_hopper(sz_regex_engine_t *engine, sz_strings_view_t const *haystacks,
                                                    sz_size_t pattern_index, sz_match_t *groups,
                                                    sz_size_t groups_capacity, sz_size_t *groups_offsets,
                                                    sz_stream_t stream) {
    int caller = 0;
    sz_status_t status = sz_device_enter_cuda_(stream, &caller);
    if (status != sz_success_k) return status;
    status = sz_regex_extract_scoped_cuda_(engine, haystacks, pattern_index, groups, groups_capacity, groups_offsets,
                                           &sz_regex_entry_points_hopper_, stream);
    sz_device_leave_cuda_(caller);
    return status;
}

#endif // STRINGZILLA_TARGET_HOPPER

#if defined(__cplusplus)
} // extern "C"
#endif
#endif // STRINGZILLA_ARCH_CUDA_HOPPER_
#endif // STRINGZILLA_ARCH_CUDA_
#endif // STRINGZILLA_REGEX_HOPPER_CUH_
