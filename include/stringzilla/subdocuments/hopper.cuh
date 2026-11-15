/**
 *  @file include/stringzilla/subdocuments/hopper.cuh
 *  @author Ash Vardanian
 *  @date November 15, 2026
 *  @brief Hopper tier of path queries: the counts scan looking back once per cluster, and the
 *      @c _hopper exports, over the CUDA host side.
 *
 *  From compute capability 9.0 a grid's blocks group into clusters, each block addressing the
 *  others' shared memory. The counts scan's first block of each cluster takes one ticket for all
 *  of them, the blocks hand it their tile totals, and it looks back once over their aggregate, so
 *  the chain of look-backs is a cluster's blocks shorter. Tiles total and depths publish with
 *  @c redux.sync as on the Ampere tier.
 *
 *  @sa include/stringzilla/subdocuments/ampere.cuh
 *  @sa include/stringzilla/cuda.cuh
 */
#ifndef STRINGZILLA_SUBDOCUMENTS_HOPPER_CUH_
#define STRINGZILLA_SUBDOCUMENTS_HOPPER_CUH_

#include "stringzilla/subdocuments/ampere.cuh"
#include "stringzilla/strings/ampere.cuh" // `sz_block_sum_ampere_`
#include "stringzilla/strings/hopper.cuh" // `sz_chain_look_back_hopper_`, `sz_chain_tile_hopper_`

#if STRINGZILLA_ARCH_CUDA_
#if STRINGZILLA_ARCH_CUDA_HOPPER_

#ifdef __cplusplus
extern "C" {
#endif

#if STRINGZILLA_TARGET_HOPPER

/** @ref sz_subdocuments_scan_tile_simt_ looking back once per cluster, each tile totalled through
 *  @ref sz_block_sum_ampere_ ; launched only in clusters of @c sz_cluster_blocks_cuda_k blocks. */
static __global__ void __launch_bounds__(sz_subdocuments_threads_per_block_simt_k)
    sz_subdocuments_scan_hopper_kernel_(sz_size_t *values, sz_size_t count, sz_size_t elements_per_tile,
                                        sz_size_t *chain, sz_u64_t *states) {
    sz_subdocuments_scan_tile_simt_(values, count, elements_per_tile, chain, states, sz_block_scan_cuda_,
                                    sz_block_sum_ampere_, sz_chain_tile_hopper_, sz_chain_look_back_hopper_);
}

/** The Hopper tier's entry points, the Ampere tier's depths over a scan clustered by eight. */
static sz_subdocuments_entry_points_cuda_t const sz_subdocuments_entry_points_hopper_ = {
    (void const *)sz_subdocuments_scan_hopper_kernel_,
    (void const *)sz_subdocuments_large_drop_ampere_kernel_,
    sz_cluster_blocks_cuda_k,
    sz_cap_hopper_k,
};

STRINGZILLA_API sz_status_t sz_subdocuments_engine_init_hopper(
    sz_subdocuments_engine_t *engine, sz_strings_view_t const *paths, sz_subdocuments_format_t format,
    sz_subdocuments_options_t options, sz_size_t *malformed_index, sz_allocator_t *allocator, sz_stream_t stream) {
    int caller = 0;
    sz_status_t status = sz_device_enter_cuda_(stream, &caller);
    if (status != sz_success_k) return status;
    status = sz_subdocuments_engine_init_scoped_cuda_(engine, paths, format, options, malformed_index,
                                                      &sz_subdocuments_entry_points_hopper_, allocator, stream);
    sz_device_leave_cuda_(caller);
    return status;
}

STRINGZILLA_API sz_status_t sz_subdocuments_contains_hopper(sz_subdocuments_engine_t *engine,
                                                            sz_strings_view_t const *documents, sz_u1x8_t *contained,
                                                            sz_size_t *malformed_index, sz_stream_t stream) {
    int caller = 0;
    sz_status_t status = sz_device_enter_cuda_(stream, &caller);
    if (status != sz_success_k) return status;
    status = sz_subdocuments_contains_scoped_cuda_(engine, documents, contained, malformed_index,
                                                   &sz_subdocuments_entry_points_hopper_, stream);
    sz_device_leave_cuda_(caller);
    return status;
}

STRINGZILLA_API sz_status_t sz_subdocuments_counts_hopper(sz_subdocuments_engine_t *engine,
                                                          sz_strings_view_t const *documents, sz_size_t *counts,
                                                          sz_size_t counts_stride, sz_size_t *malformed_index,
                                                          sz_stream_t stream) {
    int caller = 0;
    sz_status_t status = sz_device_enter_cuda_(stream, &caller);
    if (status != sz_success_k) return status;
    status = sz_subdocuments_counts_scoped_cuda_(engine, documents, counts, counts_stride, malformed_index,
                                                 &sz_subdocuments_entry_points_hopper_, stream);
    sz_device_leave_cuda_(caller);
    return status;
}

STRINGZILLA_API sz_status_t sz_subdocuments_find_hopper(sz_subdocuments_engine_t *engine,
                                                        sz_strings_view_t const *documents, sz_match_t *matches,
                                                        sz_size_t matches_capacity, sz_size_t *matches_offsets,
                                                        sz_size_t *malformed_index, sz_stream_t stream) {
    int caller = 0;
    sz_status_t status = sz_device_enter_cuda_(stream, &caller);
    if (status != sz_success_k) return status;
    status = sz_subdocuments_find_scoped_cuda_(engine, documents, matches, matches_capacity, matches_offsets,
                                               malformed_index, &sz_subdocuments_entry_points_hopper_, stream);
    sz_device_leave_cuda_(caller);
    return status;
}

#endif // STRINGZILLA_TARGET_HOPPER

#ifdef __cplusplus
}
#endif
#endif // STRINGZILLA_ARCH_CUDA_HOPPER_
#endif // STRINGZILLA_ARCH_CUDA_
#endif // STRINGZILLA_SUBDOCUMENTS_HOPPER_CUH_
