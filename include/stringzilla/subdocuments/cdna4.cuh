/**
 *  @file include/stringzilla/subdocuments/cdna4.cuh
 *  @author Ash Vardanian
 *  @date November 14, 2026
 *  @brief CDNA4 tier of path queries: the entry points passing CDNA3's DPP lane and block scans to
 *      the shared walks for gfx950, and the @c _cdna4 exports over the ROCm host side.
 *
 *  gfx950 keeps the 64-lane wavefront and the row broadcasts of gfx942, so the scans live once,
 *  under closures covering this tier, in <tt>rocm.cuh</tt> and <tt>subdocuments/cdna3.cuh</tt>.
 *  Only the kernels compiled for gfx950 are its own.
 *
 *  @sa include/stringzilla/subdocuments/cdna3.cuh
 */
#ifndef STRINGZILLA_SUBDOCUMENTS_CDNA4_CUH_
#define STRINGZILLA_SUBDOCUMENTS_CDNA4_CUH_

#include "stringzilla/types.h"

#if STRINGZILLA_ARCH_ROCM_
#if STRINGZILLA_ARCH_ROCM_CDNA4_

#include "stringzilla/subdocuments/cdna3.cuh" // `sz_subdocuments_block_scan_cdna3_` and its bounded form
#include "stringzilla/strings/cdna3.cuh"      // `sz_block_scan_cdna3_`, `sz_block_sum_cdna3_` & kin

#if defined(__cplusplus)
extern "C" {
#endif

#if STRINGZILLA_TARGET_CDNA4

/*  The shared passes, each over its body in <tt>subdocuments/kernels.cuh</tt>. */
#pragma region Entry Points

static __global__ void __launch_bounds__(sz_subdocuments_threads_per_block_simt_k)
    sz_subdocuments_walk_cdna4_kernel_(sz_subdocuments_engine_t engine, sz_strings_view_t documents,
                                       sz_subdocuments_round_simt_t round) {
    sz_subdocuments_walk_documents_simt_(&engine, &documents, &round, (sz_size_t)blockIdx.x * blockDim.x + threadIdx.x,
                                         0, 1, (sz_simt_group_t const *)STRINGZILLA_NULL);
}

static __global__ void __launch_bounds__(sz_subdocuments_threads_per_block_simt_k)
    sz_subdocuments_contains_cdna4_kernel_(sz_size_t const *counted, sz_u1x8_t *contained, sz_size_t count) {
    sz_subdocuments_contains_simt_(counted, contained, count, sz_lanes_ballot_rocm_);
}

static __global__ void __launch_bounds__(sz_subdocuments_threads_per_block_simt_k)
    sz_subdocuments_route_cdna4_kernel_(sz_strings_view_t documents, sz_subdocuments_round_simt_t round) {
    sz_subdocuments_route_simt_(documents, round);
}

static __global__ void __launch_bounds__(sz_subdocuments_threads_per_block_simt_k)
    sz_subdocuments_large_lengths_cdna4_kernel_(sz_subdocuments_large_simt_t large) {
    sz_subdocuments_large_lengths_simt_(large);
}

static __global__ void __launch_bounds__(sz_subdocuments_threads_per_block_simt_k)
    sz_subdocuments_large_summary_cdna4_kernel_(sz_strings_view_t documents, sz_subdocuments_large_simt_t large) {
    sz_subdocuments_large_summary_simt_(documents, large);
}

static __global__ void __launch_bounds__(sz_subdocuments_threads_per_block_simt_k)
    sz_subdocuments_large_count_cdna4_kernel_(sz_strings_view_t documents, sz_subdocuments_large_simt_t large) {
    sz_subdocuments_large_count_simt_(documents, large);
}

static __global__ void __launch_bounds__(sz_subdocuments_threads_per_block_simt_k)
    sz_subdocuments_large_check_cdna4_kernel_(sz_subdocuments_round_simt_t round, sz_subdocuments_large_simt_t large) {
    sz_subdocuments_large_check_simt_(round, large);
}

static __global__ void __launch_bounds__(sz_subdocuments_threads_per_block_simt_k)
    sz_subdocuments_large_tokens_cdna4_kernel_(sz_strings_view_t documents, sz_subdocuments_large_simt_t large) {
    sz_subdocuments_large_tokens_simt_(documents, large);
}

static __global__ void __launch_bounds__(sz_subdocuments_threads_per_block_simt_k)
    sz_subdocuments_large_minima_cdna4_kernel_(sz_subdocuments_large_simt_t large) {
    sz_subdocuments_large_minima_simt_(large);
}

static __global__ void __launch_bounds__(sz_subdocuments_threads_per_block_simt_k)
    sz_subdocuments_large_parents_cdna4_kernel_(sz_subdocuments_large_simt_t large) {
    sz_subdocuments_large_parents_simt_(large);
}

static __global__ void __launch_bounds__(sz_subdocuments_threads_per_block_simt_k)
    sz_subdocuments_large_closes_cdna4_kernel_(sz_subdocuments_large_simt_t large) {
    sz_subdocuments_large_closes_simt_(large);
}

static __global__ void __launch_bounds__(sz_subdocuments_threads_per_block_simt_k)
    sz_subdocuments_large_sums_cdna4_kernel_(sz_subdocuments_large_simt_t large) {
    sz_subdocuments_large_sums_simt_(large);
}

static __global__ void __launch_bounds__(sz_subdocuments_threads_per_block_simt_k)
    sz_subdocuments_large_offsets_cdna4_kernel_(sz_subdocuments_large_simt_t large) {
    sz_subdocuments_large_offsets_simt_(large);
}

static __global__ void __launch_bounds__(sz_subdocuments_threads_per_block_simt_k)
    sz_subdocuments_large_selectors_cdna4_kernel_(sz_subdocuments_engine_t engine, sz_strings_view_t documents,
                                                  sz_subdocuments_large_simt_t large) {
    sz_subdocuments_large_selectors_simt_(engine, documents, large);
}

static __global__ void __launch_bounds__(sz_subdocuments_threads_per_block_simt_k)
    sz_subdocuments_large_jump_cdna4_kernel_(sz_subdocuments_engine_t engine, sz_subdocuments_large_simt_t large) {
    sz_subdocuments_large_jump_simt_(engine, large);
}

static __global__ void __launch_bounds__(sz_subdocuments_threads_per_block_simt_k)
    sz_subdocuments_large_advance_cdna4_kernel_(sz_subdocuments_engine_t engine, sz_subdocuments_large_simt_t large) {
    sz_subdocuments_large_advance_simt_(engine, large);
}

static __global__ void __launch_bounds__(sz_subdocuments_threads_per_block_simt_k)
    sz_subdocuments_large_walk_cdna4_kernel_(sz_subdocuments_engine_t engine, sz_strings_view_t documents,
                                             sz_subdocuments_round_simt_t round, sz_subdocuments_large_simt_t large) {
    sz_subdocuments_large_walk_simt_(engine, documents, round, large);
}

static __global__ void __launch_bounds__(sz_subdocuments_threads_per_block_simt_k)
    sz_subdocuments_large_finish_cdna4_kernel_(sz_subdocuments_round_simt_t round, sz_subdocuments_large_simt_t large) {
    sz_subdocuments_large_finish_simt_(round, large);
}

static __global__ void __launch_bounds__(sz_subdocuments_threads_per_block_simt_k)
    sz_subdocuments_large_drop_cdna4_kernel_(sz_subdocuments_round_simt_t round, sz_subdocuments_large_simt_t large) {
    sz_subdocuments_large_drop_simt_(round, large);
}

/** @ref sz_subdocuments_walk_documents_simt_ with 32 lanes walking each JSON document, scanning
 *  them through DPP. */
static __global__ void __launch_bounds__(sz_subdocuments_threads_per_block_simt_k)
    sz_subdocuments_warp_walk_cdna4_kernel_(sz_subdocuments_engine_t engine, sz_strings_view_t documents,
                                            sz_subdocuments_round_simt_t round) {
    sz_size_t const thread = (sz_size_t)blockIdx.x * blockDim.x + threadIdx.x;
    sz_simt_group_t const group = {32,
                                   (unsigned)(thread & 31),
                                   sz_lanes_any_rocm_,
                                   sz_lanes_ballot_rocm_,
                                   sz_lanes_first_rocm_,
                                   sz_lanes_below_rocm_,
                                   sz_shuffle_index_rocm_,
                                   sz_shuffle_up_rocm_,
                                   sz_u32_lanes_prefix_sum_cdna3_,
                                   sz_i32_lanes_prefix_min_cdna3_,
                                   sz_u64_lanes_sum_cdna3_,
                                   sz_group_sync_rocm_};
    sz_subdocuments_walk_documents_simt_(&engine, &documents, &round, thread / 32, thread & 31, 32, &group);
}

static __global__ void __launch_bounds__(sz_subdocuments_threads_per_block_simt_k)
    sz_subdocuments_scan_cdna4_kernel_(sz_size_t *values, sz_size_t count, sz_size_t elements_per_tile,
                                       sz_size_t *chain, sz_u64_t *states) {
    sz_subdocuments_scan_tile_simt_(values, count, elements_per_tile, chain, states, sz_block_scan_cdna3_,
                                    sz_block_sum_cdna3_, sz_chain_tile_simt_, sz_chain_look_back_rocm_);
}

static __global__ void __launch_bounds__(sz_subdocuments_threads_per_block_simt_k)
    sz_subdocuments_large_reduce_cdna4_kernel_(sz_subdocuments_large_simt_t large) {
    sz_subdocuments_large_reduce_simt_(&large, sz_subdocuments_block_scan_bounded_cdna3_);
}

static __global__ void __launch_bounds__(sz_subdocuments_threads_per_block_simt_k)
    sz_subdocuments_large_carry_cdna4_kernel_(sz_subdocuments_large_simt_t large) {
    sz_subdocuments_large_carry_simt_(&large, sz_subdocuments_block_scan_cdna3_);
}

static __global__ void __launch_bounds__(sz_subdocuments_threads_per_block_simt_k)
    sz_subdocuments_large_apply_cdna4_kernel_(sz_subdocuments_large_simt_t large) {
    sz_subdocuments_large_apply_simt_(&large, sz_subdocuments_block_scan_bounded_cdna3_);
}

/** The CDNA4 tier's entry points, walking and scanning through DPP. */
static sz_subdocuments_entry_points_rocm_t const sz_subdocuments_entry_points_cdna4_ = {
    (void const *)sz_subdocuments_warp_walk_cdna4_kernel_,
    (void const *)sz_subdocuments_scan_cdna4_kernel_,
    (void const *)sz_subdocuments_large_reduce_cdna4_kernel_,
    (void const *)sz_subdocuments_large_carry_cdna4_kernel_,
    (void const *)sz_subdocuments_large_apply_cdna4_kernel_,
    (void const *)sz_subdocuments_walk_cdna4_kernel_,
    (void const *)sz_subdocuments_contains_cdna4_kernel_,
    (void const *)sz_subdocuments_route_cdna4_kernel_,
    (void const *)sz_subdocuments_large_lengths_cdna4_kernel_,
    (void const *)sz_subdocuments_large_summary_cdna4_kernel_,
    (void const *)sz_subdocuments_large_count_cdna4_kernel_,
    (void const *)sz_subdocuments_large_check_cdna4_kernel_,
    (void const *)sz_subdocuments_large_tokens_cdna4_kernel_,
    (void const *)sz_subdocuments_large_minima_cdna4_kernel_,
    (void const *)sz_subdocuments_large_parents_cdna4_kernel_,
    (void const *)sz_subdocuments_large_closes_cdna4_kernel_,
    (void const *)sz_subdocuments_large_sums_cdna4_kernel_,
    (void const *)sz_subdocuments_large_offsets_cdna4_kernel_,
    (void const *)sz_subdocuments_large_selectors_cdna4_kernel_,
    (void const *)sz_subdocuments_large_jump_cdna4_kernel_,
    (void const *)sz_subdocuments_large_advance_cdna4_kernel_,
    (void const *)sz_subdocuments_large_walk_cdna4_kernel_,
    (void const *)sz_subdocuments_large_finish_cdna4_kernel_,
    (void const *)sz_subdocuments_large_drop_cdna4_kernel_,
    sz_cap_cdna4_k,
};

#pragma endregion Entry Points

STRINGZILLA_API sz_status_t sz_subdocuments_engine_init_cdna4(
    sz_subdocuments_engine_t *engine, sz_strings_view_t const *paths, sz_subdocuments_format_t format,
    sz_subdocuments_options_t options, sz_size_t *malformed_index, sz_allocator_t *allocator, sz_stream_t stream) {
    int caller = 0;
    sz_status_t status = sz_device_enter_rocm_(stream, &caller);
    if (status != sz_success_k) return status;
    status = sz_subdocuments_engine_init_scoped_rocm_(engine, paths, format, options, malformed_index,
                                                      &sz_subdocuments_entry_points_cdna4_, allocator, stream);
    sz_device_leave_rocm_(caller);
    return status;
}

STRINGZILLA_API sz_status_t sz_subdocuments_contains_cdna4(sz_subdocuments_engine_t *engine,
                                                           sz_strings_view_t const *documents, sz_u1x8_t *contained,
                                                           sz_size_t *malformed_index, sz_stream_t stream) {
    int caller = 0;
    sz_status_t status = sz_device_enter_rocm_(stream, &caller);
    if (status != sz_success_k) return status;
    status = sz_subdocuments_contains_scoped_rocm_(engine, documents, contained, malformed_index,
                                                   &sz_subdocuments_entry_points_cdna4_, stream);
    sz_device_leave_rocm_(caller);
    return status;
}

STRINGZILLA_API sz_status_t sz_subdocuments_counts_cdna4(sz_subdocuments_engine_t *engine,
                                                         sz_strings_view_t const *documents, sz_size_t *counts,
                                                         sz_size_t counts_stride, sz_size_t *malformed_index,
                                                         sz_stream_t stream) {
    int caller = 0;
    sz_status_t status = sz_device_enter_rocm_(stream, &caller);
    if (status != sz_success_k) return status;
    status = sz_subdocuments_counts_scoped_rocm_(engine, documents, counts, counts_stride, malformed_index,
                                                 &sz_subdocuments_entry_points_cdna4_, stream);
    sz_device_leave_rocm_(caller);
    return status;
}

STRINGZILLA_API sz_status_t sz_subdocuments_find_cdna4(sz_subdocuments_engine_t *engine,
                                                       sz_strings_view_t const *documents, sz_match_t *matches,
                                                       sz_size_t matches_capacity, sz_size_t *matches_offsets,
                                                       sz_size_t *malformed_index, sz_stream_t stream) {
    int caller = 0;
    sz_status_t status = sz_device_enter_rocm_(stream, &caller);
    if (status != sz_success_k) return status;
    status = sz_subdocuments_find_scoped_rocm_(engine, documents, matches, matches_capacity, matches_offsets,
                                               malformed_index, &sz_subdocuments_entry_points_cdna4_, stream);
    sz_device_leave_rocm_(caller);
    return status;
}

#endif // STRINGZILLA_TARGET_CDNA4

#if defined(__cplusplus)
} // extern "C"
#endif
#endif // STRINGZILLA_ARCH_ROCM_CDNA4_
#endif // STRINGZILLA_ARCH_ROCM_
#endif // STRINGZILLA_SUBDOCUMENTS_CDNA4_CUH_
