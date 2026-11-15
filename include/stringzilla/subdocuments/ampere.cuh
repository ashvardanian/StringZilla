/**
 *  @file include/stringzilla/subdocuments/ampere.cuh
 *  @author Ash Vardanian
 *  @date November 14, 2026
 *  @brief Ampere tier of path queries: tiles totalled and depths published with @c redux.sync ,
 *      and the @c _ampere exports, over the CUDA host side.
 *
 *  The counts scan learns each tile's total from two reductions per warp rather than a scan of
 *  every warp. Dropping the refused documents' chunks publishes how deep the kept ones nest, which
 *  the CUDA tier does with an atomic per chunk and this tier with one @c redux.sync and one atomic
 *  per warp. The walks themselves keep the CUDA tier's kernels: their group sums are prefix sums,
 *  which no reduction answers.
 *
 *  @sa include/stringzilla/subdocuments/cuda.cuh
 *  @sa include/stringzilla/cuda.cuh
 */
#ifndef STRINGZILLA_SUBDOCUMENTS_AMPERE_CUH_
#define STRINGZILLA_SUBDOCUMENTS_AMPERE_CUH_

#include "stringzilla/subdocuments/cuda.cuh"
#include "stringzilla/strings/ampere.cuh" // `sz_block_sum_ampere_`, `sz_u32_lanes_max_ampere_`

#if STRINGZILLA_ARCH_CUDA_
#if STRINGZILLA_ARCH_CUDA_AMPERE_

#ifdef __cplusplus
extern "C" {
#endif

/** @ref sz_subdocuments_large_drop_simt_kernel_ with each warp's deepest nesting reduced first. */
static __global__ void __launch_bounds__(sz_subdocuments_threads_per_block_simt_k)
    sz_subdocuments_large_drop_ampere_kernel_(sz_subdocuments_round_simt_t round, sz_subdocuments_large_simt_t large) {
    sz_u32_t deepest = 0;
    for (sz_size_t chunk = (sz_size_t)blockIdx.x * blockDim.x + threadIdx.x; chunk < large.chunks;
         chunk += large.threads)
        deepest = sz_max_of_two(deepest, sz_subdocuments_large_drop_chunk_simt_(&large, chunk));
    deepest = sz_u32_lanes_max_ampere_(deepest);
    if ((threadIdx.x & 31u) == 0 && deepest)
        atomicMax((unsigned long long *)&round.header->deepest, (unsigned long long)deepest);
}

#if STRINGZILLA_TARGET_AMPERE

/** @ref sz_subdocuments_scan_tile_simt_ over CUDA's warps, each tile totalled through
 *  @ref sz_block_sum_ampere_ . */
static __global__ void __launch_bounds__(sz_subdocuments_threads_per_block_simt_k)
    sz_subdocuments_scan_ampere_kernel_(sz_size_t *values, sz_size_t count, sz_size_t elements_per_tile,
                                        sz_size_t *chain, sz_u64_t *states) {
    sz_subdocuments_scan_tile_simt_(values, count, elements_per_tile, chain, states, sz_block_scan_cuda_,
                                    sz_block_sum_ampere_, sz_chain_tile_simt_, sz_chain_look_back_cuda_);
}

/** The Ampere tier's entry points, its scans and depths reduced with @c redux.sync . */
static sz_subdocuments_entry_points_cuda_t const sz_subdocuments_entry_points_ampere_ = {
    (void const *)sz_subdocuments_scan_ampere_kernel_,
    (void const *)sz_subdocuments_large_drop_ampere_kernel_,
    1,
    sz_cap_ampere_k,
};

STRINGZILLA_API sz_status_t sz_subdocuments_engine_init_ampere(
    sz_subdocuments_engine_t *engine, sz_strings_view_t const *paths, sz_subdocuments_format_t format,
    sz_subdocuments_options_t options, sz_size_t *malformed_index, sz_allocator_t *allocator, sz_stream_t stream) {
    int caller = 0;
    sz_status_t status = sz_device_enter_cuda_(stream, &caller);
    if (status != sz_success_k) return status;
    status = sz_subdocuments_engine_init_scoped_cuda_(engine, paths, format, options, malformed_index,
                                                      &sz_subdocuments_entry_points_ampere_, allocator, stream);
    sz_device_leave_cuda_(caller);
    return status;
}

STRINGZILLA_API sz_status_t sz_subdocuments_contains_ampere(sz_subdocuments_engine_t *engine,
                                                            sz_strings_view_t const *documents, sz_u1x8_t *contained,
                                                            sz_size_t *malformed_index, sz_stream_t stream) {
    int caller = 0;
    sz_status_t status = sz_device_enter_cuda_(stream, &caller);
    if (status != sz_success_k) return status;
    status = sz_subdocuments_contains_scoped_cuda_(engine, documents, contained, malformed_index,
                                                   &sz_subdocuments_entry_points_ampere_, stream);
    sz_device_leave_cuda_(caller);
    return status;
}

STRINGZILLA_API sz_status_t sz_subdocuments_counts_ampere(sz_subdocuments_engine_t *engine,
                                                          sz_strings_view_t const *documents, sz_size_t *counts,
                                                          sz_size_t counts_stride, sz_size_t *malformed_index,
                                                          sz_stream_t stream) {
    int caller = 0;
    sz_status_t status = sz_device_enter_cuda_(stream, &caller);
    if (status != sz_success_k) return status;
    status = sz_subdocuments_counts_scoped_cuda_(engine, documents, counts, counts_stride, malformed_index,
                                                 &sz_subdocuments_entry_points_ampere_, stream);
    sz_device_leave_cuda_(caller);
    return status;
}

STRINGZILLA_API sz_status_t sz_subdocuments_find_ampere(sz_subdocuments_engine_t *engine,
                                                        sz_strings_view_t const *documents, sz_match_t *matches,
                                                        sz_size_t matches_capacity, sz_size_t *matches_offsets,
                                                        sz_size_t *malformed_index, sz_stream_t stream) {
    int caller = 0;
    sz_status_t status = sz_device_enter_cuda_(stream, &caller);
    if (status != sz_success_k) return status;
    status = sz_subdocuments_find_scoped_cuda_(engine, documents, matches, matches_capacity, matches_offsets,
                                               malformed_index, &sz_subdocuments_entry_points_ampere_, stream);
    sz_device_leave_cuda_(caller);
    return status;
}

#endif // STRINGZILLA_TARGET_AMPERE

#ifdef __cplusplus
}
#endif
#endif // STRINGZILLA_ARCH_CUDA_AMPERE_
#endif // STRINGZILLA_ARCH_CUDA_
#endif // STRINGZILLA_SUBDOCUMENTS_AMPERE_CUH_
