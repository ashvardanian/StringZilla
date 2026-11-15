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

#include "stringzilla/regex/ampere.cuh"
#include "stringzilla/strings/hopper.cuh" // `sz_chain_look_back_hopper_`, `sz_chain_tile_hopper_`

#if STRINGZILLA_ARCH_CUDA_
#if STRINGZILLA_ARCH_CUDA_HOPPER_

#ifdef __cplusplus
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

/** The Hopper tier's entry points, the Ampere tier's marking walk over scans clustered by eight. */
static sz_regex_entry_points_cuda_t const sz_regex_entry_points_hopper_ = {
    (void const *)sz_regex_mark_ampere_kernel_,
    (void const *)sz_regex_scan_hopper_kernel_,
    (void const *)sz_regex_scan_bounded_hopper_kernel_,
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

#ifdef __cplusplus
}
#endif
#endif // STRINGZILLA_ARCH_CUDA_HOPPER_
#endif // STRINGZILLA_ARCH_CUDA_
#endif // STRINGZILLA_REGEX_HOPPER_CUH_
