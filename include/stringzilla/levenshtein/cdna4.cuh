/**
 *  @file include/stringzilla/levenshtein/cdna4.cuh
 *  @author Ash Vardanian
 *  @date November 12, 2026
 *  @brief CDNA4 tier of Levenshtein distances: the entry points passing CDNA3's one-lane shift to
 *      the shared walks for gfx950, and the @c _cdna4 exports over the ROCm host side.
 *
 *  gfx950 keeps the 64-lane wavefront and the DPP wavefront shift of gfx942, so the shifts live
 *  once in <tt>levenshtein/cdna3.cuh</tt>, under a closure covering this tier, and only the kernels
 *  compiled for gfx950 are its own. Overlap takes the shift from there too.
 *
 *  @sa include/stringzilla/levenshtein/cdna3.cuh
 */
#ifndef STRINGZILLA_LEVENSHTEIN_CDNA4_CUH_
#define STRINGZILLA_LEVENSHTEIN_CDNA4_CUH_

#include "stringzilla/levenshtein/cdna3.cuh" // `sz_shuffle_up_cdna3_`, `sz_group_shuffle_up_cdna3_`

#if STRINGZILLA_ARCH_ROCM_
#if STRINGZILLA_ARCH_ROCM_CDNA4_

#ifdef __cplusplus
extern "C" {
#endif

#if STRINGZILLA_TARGET_CDNA4

#pragma region Entry Points

/*  Warped byte rung: one entry point per words-per-lane. */
static __global__ void __launch_bounds__(sz_levenshtein_candidates_per_block_max_simt_k)
    sz_levenshtein_distances_k1_cdna4_kernel_( //
        sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_strings_view_t candidates, sz_size_t *distances,
        sz_size_t distances_stride, sz_size_t bound) {
    sz_levenshtein_warp_sweep_simt_(engine, order, candidates, distances, distances_stride, bound, 1,
                                    sz_group_shuffle_up_cdna3_, sz_group_shuffle_index_rocm_, STRINGZILLA_NULL,
                                    sz_lanes_broadcast_rocm_, sz_tile_queue_open_simt_, sz_tile_queue_draw_simt_);
}

static __global__ void __launch_bounds__(sz_levenshtein_candidates_per_block_max_simt_k)
    sz_levenshtein_distances_k2_cdna4_kernel_( //
        sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_strings_view_t candidates, sz_size_t *distances,
        sz_size_t distances_stride, sz_size_t bound) {
    sz_levenshtein_warp_sweep_simt_(engine, order, candidates, distances, distances_stride, bound, 2,
                                    sz_group_shuffle_up_cdna3_, sz_group_shuffle_index_rocm_, STRINGZILLA_NULL,
                                    sz_lanes_broadcast_rocm_, sz_tile_queue_open_simt_, sz_tile_queue_draw_simt_);
}

static __global__ void __launch_bounds__(sz_levenshtein_candidates_per_block_max_simt_k)
    sz_levenshtein_distances_k3_cdna4_kernel_( //
        sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_strings_view_t candidates, sz_size_t *distances,
        sz_size_t distances_stride, sz_size_t bound) {
    sz_levenshtein_warp_sweep_simt_(engine, order, candidates, distances, distances_stride, bound, 3,
                                    sz_group_shuffle_up_cdna3_, sz_group_shuffle_index_rocm_, STRINGZILLA_NULL,
                                    sz_lanes_broadcast_rocm_, sz_tile_queue_open_simt_, sz_tile_queue_draw_simt_);
}

static __global__ void __launch_bounds__(sz_levenshtein_candidates_per_block_max_simt_k)
    sz_levenshtein_distances_k4_cdna4_kernel_( //
        sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_strings_view_t candidates, sz_size_t *distances,
        sz_size_t distances_stride, sz_size_t bound) {
    sz_levenshtein_warp_sweep_simt_(engine, order, candidates, distances, distances_stride, bound, 4,
                                    sz_group_shuffle_up_cdna3_, sz_group_shuffle_index_rocm_, STRINGZILLA_NULL,
                                    sz_lanes_broadcast_rocm_, sz_tile_queue_open_simt_, sz_tile_queue_draw_simt_);
}

static __global__ void __launch_bounds__(sz_levenshtein_candidates_per_block_max_simt_k)
    sz_levenshtein_distances_k5_cdna4_kernel_( //
        sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_strings_view_t candidates, sz_size_t *distances,
        sz_size_t distances_stride, sz_size_t bound) {
    sz_levenshtein_warp_sweep_simt_(engine, order, candidates, distances, distances_stride, bound, 5,
                                    sz_group_shuffle_up_cdna3_, sz_group_shuffle_index_rocm_, STRINGZILLA_NULL,
                                    sz_lanes_broadcast_rocm_, sz_tile_queue_open_simt_, sz_tile_queue_draw_simt_);
}

static __global__ void __launch_bounds__(sz_levenshtein_candidates_per_block_max_simt_k)
    sz_levenshtein_distances_k6_cdna4_kernel_( //
        sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_strings_view_t candidates, sz_size_t *distances,
        sz_size_t distances_stride, sz_size_t bound) {
    sz_levenshtein_warp_sweep_simt_(engine, order, candidates, distances, distances_stride, bound, 6,
                                    sz_group_shuffle_up_cdna3_, sz_group_shuffle_index_rocm_, STRINGZILLA_NULL,
                                    sz_lanes_broadcast_rocm_, sz_tile_queue_open_simt_, sz_tile_queue_draw_simt_);
}

static __global__ void __launch_bounds__(sz_levenshtein_candidates_per_block_max_simt_k)
    sz_levenshtein_distances_k7_cdna4_kernel_( //
        sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_strings_view_t candidates, sz_size_t *distances,
        sz_size_t distances_stride, sz_size_t bound) {
    sz_levenshtein_warp_sweep_simt_(engine, order, candidates, distances, distances_stride, bound, 7,
                                    sz_group_shuffle_up_cdna3_, sz_group_shuffle_index_rocm_, STRINGZILLA_NULL,
                                    sz_lanes_broadcast_rocm_, sz_tile_queue_open_simt_, sz_tile_queue_draw_simt_);
}

static __global__ void __launch_bounds__(sz_levenshtein_candidates_per_block_max_simt_k)
    sz_levenshtein_distances_k8_cdna4_kernel_( //
        sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_strings_view_t candidates, sz_size_t *distances,
        sz_size_t distances_stride, sz_size_t bound) {
    sz_levenshtein_warp_sweep_simt_(engine, order, candidates, distances, distances_stride, bound, 8,
                                    sz_group_shuffle_up_cdna3_, sz_group_shuffle_index_rocm_, STRINGZILLA_NULL,
                                    sz_lanes_broadcast_rocm_, sz_tile_queue_open_simt_, sz_tile_queue_draw_simt_);
}

/*  Warped rune rung: one entry point per words-per-lane. */
static __global__ void __launch_bounds__(sz_levenshtein_candidates_per_block_max_simt_k)
    sz_levenshtein_distances_utf8_k1_cdna4_kernel_( //
        sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_strings_view_t candidates, sz_size_t *distances,
        sz_size_t distances_stride, sz_size_t bound) {
    sz_levenshtein_warp_sweep_utf8_simt_(engine, order, candidates, distances, distances_stride, bound, 1,
                                         sz_group_shuffle_up_cdna3_, sz_group_shuffle_index_rocm_, STRINGZILLA_NULL,
                                         sz_lanes_broadcast_rocm_, sz_tile_queue_open_simt_, sz_tile_queue_draw_simt_);
}

static __global__ void __launch_bounds__(sz_levenshtein_candidates_per_block_max_simt_k)
    sz_levenshtein_distances_utf8_k2_cdna4_kernel_( //
        sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_strings_view_t candidates, sz_size_t *distances,
        sz_size_t distances_stride, sz_size_t bound) {
    sz_levenshtein_warp_sweep_utf8_simt_(engine, order, candidates, distances, distances_stride, bound, 2,
                                         sz_group_shuffle_up_cdna3_, sz_group_shuffle_index_rocm_, STRINGZILLA_NULL,
                                         sz_lanes_broadcast_rocm_, sz_tile_queue_open_simt_, sz_tile_queue_draw_simt_);
}

static __global__ void __launch_bounds__(sz_levenshtein_candidates_per_block_max_simt_k)
    sz_levenshtein_distances_utf8_k3_cdna4_kernel_( //
        sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_strings_view_t candidates, sz_size_t *distances,
        sz_size_t distances_stride, sz_size_t bound) {
    sz_levenshtein_warp_sweep_utf8_simt_(engine, order, candidates, distances, distances_stride, bound, 3,
                                         sz_group_shuffle_up_cdna3_, sz_group_shuffle_index_rocm_, STRINGZILLA_NULL,
                                         sz_lanes_broadcast_rocm_, sz_tile_queue_open_simt_, sz_tile_queue_draw_simt_);
}

static __global__ void __launch_bounds__(sz_levenshtein_candidates_per_block_max_simt_k)
    sz_levenshtein_distances_utf8_k4_cdna4_kernel_( //
        sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_strings_view_t candidates, sz_size_t *distances,
        sz_size_t distances_stride, sz_size_t bound) {
    sz_levenshtein_warp_sweep_utf8_simt_(engine, order, candidates, distances, distances_stride, bound, 4,
                                         sz_group_shuffle_up_cdna3_, sz_group_shuffle_index_rocm_, STRINGZILLA_NULL,
                                         sz_lanes_broadcast_rocm_, sz_tile_queue_open_simt_, sz_tile_queue_draw_simt_);
}

static __global__ void __launch_bounds__(sz_levenshtein_candidates_per_block_max_simt_k)
    sz_levenshtein_distances_utf8_k5_cdna4_kernel_( //
        sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_strings_view_t candidates, sz_size_t *distances,
        sz_size_t distances_stride, sz_size_t bound) {
    sz_levenshtein_warp_sweep_utf8_simt_(engine, order, candidates, distances, distances_stride, bound, 5,
                                         sz_group_shuffle_up_cdna3_, sz_group_shuffle_index_rocm_, STRINGZILLA_NULL,
                                         sz_lanes_broadcast_rocm_, sz_tile_queue_open_simt_, sz_tile_queue_draw_simt_);
}

static __global__ void __launch_bounds__(sz_levenshtein_candidates_per_block_max_simt_k)
    sz_levenshtein_distances_utf8_k6_cdna4_kernel_( //
        sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_strings_view_t candidates, sz_size_t *distances,
        sz_size_t distances_stride, sz_size_t bound) {
    sz_levenshtein_warp_sweep_utf8_simt_(engine, order, candidates, distances, distances_stride, bound, 6,
                                         sz_group_shuffle_up_cdna3_, sz_group_shuffle_index_rocm_, STRINGZILLA_NULL,
                                         sz_lanes_broadcast_rocm_, sz_tile_queue_open_simt_, sz_tile_queue_draw_simt_);
}

static __global__ void __launch_bounds__(sz_levenshtein_candidates_per_block_max_simt_k)
    sz_levenshtein_distances_utf8_k7_cdna4_kernel_( //
        sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_strings_view_t candidates, sz_size_t *distances,
        sz_size_t distances_stride, sz_size_t bound) {
    sz_levenshtein_warp_sweep_utf8_simt_(engine, order, candidates, distances, distances_stride, bound, 7,
                                         sz_group_shuffle_up_cdna3_, sz_group_shuffle_index_rocm_, STRINGZILLA_NULL,
                                         sz_lanes_broadcast_rocm_, sz_tile_queue_open_simt_, sz_tile_queue_draw_simt_);
}

static __global__ void __launch_bounds__(sz_levenshtein_candidates_per_block_max_simt_k)
    sz_levenshtein_distances_utf8_k8_cdna4_kernel_( //
        sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_strings_view_t candidates, sz_size_t *distances,
        sz_size_t distances_stride, sz_size_t bound) {
    sz_levenshtein_warp_sweep_utf8_simt_(engine, order, candidates, distances, distances_stride, bound, 8,
                                         sz_group_shuffle_up_cdna3_, sz_group_shuffle_index_rocm_, STRINGZILLA_NULL,
                                         sz_lanes_broadcast_rocm_, sz_tile_queue_open_simt_, sz_tile_queue_draw_simt_);
}

static __global__
__launch_bounds__(sz_levenshtein_tiled_threads_max_rocm_k) void sz_levenshtein_tiled_batch_cdna4_kernel_(
    sz_levenshtein_engine_t engine, sz_strings_view_t candidates, sz_size_t *distances, sz_ptr_t workspace,
    sz_levenshtein_long_arguments_simt_t batch) {
    sz_levenshtein_tiled_batch_simt_(engine, candidates, distances, workspace, batch, warpSize, sz_shuffle_up_cdna3_,
                                     sz_levenshtein_publish_rocm_, sz_levenshtein_await_rocm_);
}

/** The CDNA4 tier's entry points: the shared threaded rungs, and warped and tiled ones shifting
 *  through CDNA3's DPP wavefront shift. */
static sz_levenshtein_entry_points_rocm_t const sz_levenshtein_entry_points_cdna4_ = {
    {
        (void const *)sz_levenshtein_distances_w1_simt_kernel_,
        (void const *)sz_levenshtein_distances_w2_simt_kernel_,
        (void const *)sz_levenshtein_distances_w3_simt_kernel_,
        (void const *)sz_levenshtein_distances_w4_simt_kernel_,
        (void const *)sz_levenshtein_distances_w5_simt_kernel_,
        (void const *)sz_levenshtein_distances_w6_simt_kernel_,
        (void const *)sz_levenshtein_distances_w7_simt_kernel_,
        (void const *)sz_levenshtein_distances_w8_simt_kernel_,
        (void const *)sz_levenshtein_distances_w9_simt_kernel_,
        (void const *)sz_levenshtein_distances_w10_simt_kernel_,
        (void const *)sz_levenshtein_distances_w11_simt_kernel_,
        (void const *)sz_levenshtein_distances_w12_simt_kernel_,
        (void const *)sz_levenshtein_distances_w13_simt_kernel_,
        (void const *)sz_levenshtein_distances_w14_simt_kernel_,
        (void const *)sz_levenshtein_distances_w15_simt_kernel_,
        (void const *)sz_levenshtein_distances_w16_simt_kernel_,
    },
    {
        (void const *)sz_levenshtein_distances_utf8_w1_simt_kernel_,
        (void const *)sz_levenshtein_distances_utf8_w2_simt_kernel_,
        (void const *)sz_levenshtein_distances_utf8_w3_simt_kernel_,
        (void const *)sz_levenshtein_distances_utf8_w4_simt_kernel_,
        (void const *)sz_levenshtein_distances_utf8_w5_simt_kernel_,
        (void const *)sz_levenshtein_distances_utf8_w6_simt_kernel_,
        (void const *)sz_levenshtein_distances_utf8_w7_simt_kernel_,
        (void const *)sz_levenshtein_distances_utf8_w8_simt_kernel_,
        (void const *)sz_levenshtein_distances_utf8_w9_simt_kernel_,
        (void const *)sz_levenshtein_distances_utf8_w10_simt_kernel_,
        (void const *)sz_levenshtein_distances_utf8_w11_simt_kernel_,
        (void const *)sz_levenshtein_distances_utf8_w12_simt_kernel_,
        (void const *)sz_levenshtein_distances_utf8_w13_simt_kernel_,
        (void const *)sz_levenshtein_distances_utf8_w14_simt_kernel_,
        (void const *)sz_levenshtein_distances_utf8_w15_simt_kernel_,
        (void const *)sz_levenshtein_distances_utf8_w16_simt_kernel_,
    },
    {
        (void const *)sz_levenshtein_distances_k1_cdna4_kernel_,
        (void const *)sz_levenshtein_distances_k2_cdna4_kernel_,
        (void const *)sz_levenshtein_distances_k3_cdna4_kernel_,
        (void const *)sz_levenshtein_distances_k4_cdna4_kernel_,
        (void const *)sz_levenshtein_distances_k5_cdna4_kernel_,
        (void const *)sz_levenshtein_distances_k6_cdna4_kernel_,
        (void const *)sz_levenshtein_distances_k7_cdna4_kernel_,
        (void const *)sz_levenshtein_distances_k8_cdna4_kernel_,
    },
    {
        (void const *)sz_levenshtein_distances_utf8_k1_cdna4_kernel_,
        (void const *)sz_levenshtein_distances_utf8_k2_cdna4_kernel_,
        (void const *)sz_levenshtein_distances_utf8_k3_cdna4_kernel_,
        (void const *)sz_levenshtein_distances_utf8_k4_cdna4_kernel_,
        (void const *)sz_levenshtein_distances_utf8_k5_cdna4_kernel_,
        (void const *)sz_levenshtein_distances_utf8_k6_cdna4_kernel_,
        (void const *)sz_levenshtein_distances_utf8_k7_cdna4_kernel_,
        (void const *)sz_levenshtein_distances_utf8_k8_cdna4_kernel_,
    },
    (void const *)sz_levenshtein_tiled_batch_cdna4_kernel_,
    sz_cap_cdna4_k,
};

#pragma endregion Entry Points

STRINGZILLA_API sz_status_t sz_levenshtein_engine_init_cdna4(sz_levenshtein_engine_t *engine,
                                                             sz_strings_view_t const *queries,
                                                             sz_levenshtein_symbol_t symbol, sz_allocator_t *allocator,
                                                             sz_stream_t stream) {
    return sz_levenshtein_engine_init_scoped_rocm_(engine, queries, symbol, &sz_levenshtein_entry_points_cdna4_,
                                                   allocator, stream);
}

STRINGZILLA_API sz_status_t sz_levenshtein_distances_cdna4(sz_levenshtein_engine_t *engine,
                                                           sz_strings_view_t const *candidates,
                                                           sz_size_t const *symbol_counts, sz_size_t const *order,
                                                           sz_size_t bound, sz_size_t *distances,
                                                           sz_size_t distances_stride, sz_stream_t stream) {
    sz_unused_(symbol_counts), sz_unused_(order);
    return sz_levenshtein_distances_scoped_rocm_(engine, candidates, distances, distances_stride, bound,
                                                 &sz_levenshtein_entry_points_cdna4_, stream);
}

#endif // STRINGZILLA_TARGET_CDNA4

#ifdef __cplusplus
}
#endif
#endif // STRINGZILLA_ARCH_ROCM_CDNA4_
#endif // STRINGZILLA_ARCH_ROCM_
#endif // STRINGZILLA_LEVENSHTEIN_CDNA4_CUH_
