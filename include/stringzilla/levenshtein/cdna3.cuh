/**
 *  @file include/stringzilla/levenshtein/cdna3.cuh
 *  @author Ash Vardanian
 *  @date November 12, 2026
 *  @brief CDNA3 tier of Levenshtein distances: the one-lane shift of the gfx942 and gfx950
 *      wavefront, the entry points passing it to the shared walks for gfx942, and the @c _cdna3
 *      exports over the ROCm host side.
 *
 *  Every wavefront march hands each lane's carry one lane up per step, which @c __shfl_up sends
 *  through @c ds_bpermute and the LDS crossbar. A GFX9 wavefront moves all 64 lanes one up in a
 *  single DPP shift instead, and the first lane of each group keeps its own. The shift sits under
 *  a closure covering CDNA4, whose gfx950 kernels take it from here, as overlap does.
 *
 *  @sa include/stringzilla/levenshtein/rocm.cuh
 */
#ifndef STRINGZILLA_LEVENSHTEIN_CDNA3_CUH_
#define STRINGZILLA_LEVENSHTEIN_CDNA3_CUH_

#include "stringzilla/levenshtein/rocm.cuh"

#if STRINGZILLA_ARCH_ROCM_
#if STRINGZILLA_ARCH_ROCM_CDNA3_

#ifdef __cplusplus
extern "C" {
#endif

#pragma region Lane Shifts

/** The DPP control moving every lane's value one lane up across the whole 64-lane wavefront. */
enum { sz_shuffle_up_wavefront_dpp_control_cdna3_k = 0x138 };

/** Lane <tt>lane - delta</tt>'s @p value within this lane's aligned group of @p width lanes, a lane's own
 *  below @p delta. A one-lane step moves through one DPP wavefront shift rather than the LDS
 *  crossbar, a group's first lane keeping its own; both steps feed one select, as a branch would
 *  stop a shuffle ladder over @p delta unrolling. */
STRINGZILLA_DEVICE sz_u32_t sz_group_shuffle_up_cdna3_(sz_u32_t value, unsigned delta, unsigned width) {
    int const shifted = __builtin_amdgcn_update_dpp((int)value, (int)value, sz_shuffle_up_wavefront_dpp_control_cdna3_k,
                                                    0xF, 0xF, false);
    sz_u32_t const one_up = (__lane_id() & (width - 1u)) ? (sz_u32_t)shifted : value;
    sz_u32_t const delta_up = __shfl_up(value, delta, (int)width);
    return delta == 1 ? one_up : delta_up;
}

/** Lane <tt>lane - delta</tt>'s @p value among this thread's 32 lanes, a lane's own below @p delta. */
STRINGZILLA_DEVICE sz_u32_t sz_shuffle_up_cdna3_(sz_u32_t value, unsigned delta) {
    return sz_group_shuffle_up_cdna3_(value, delta, 32);
}

#pragma endregion Lane Shifts

#if STRINGZILLA_TARGET_CDNA3

#pragma region Entry Points

/*  Warped byte rung: one entry point per words-per-lane. */
static __global__ void __launch_bounds__(sz_levenshtein_candidates_per_block_max_simt_k)
    sz_levenshtein_distances_k1_cdna3_kernel_( //
        sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_strings_view_t candidates, sz_size_t *distances,
        sz_size_t distances_stride, sz_size_t bound) {
    sz_levenshtein_warp_sweep_simt_(engine, order, candidates, distances, distances_stride, bound, 1,
                                    sz_group_shuffle_up_cdna3_, sz_group_shuffle_index_rocm_, STRINGZILLA_NULL,
                                    sz_lanes_broadcast_rocm_, sz_tile_queue_open_simt_, sz_tile_queue_draw_simt_);
}

static __global__ void __launch_bounds__(sz_levenshtein_candidates_per_block_max_simt_k)
    sz_levenshtein_distances_k2_cdna3_kernel_( //
        sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_strings_view_t candidates, sz_size_t *distances,
        sz_size_t distances_stride, sz_size_t bound) {
    sz_levenshtein_warp_sweep_simt_(engine, order, candidates, distances, distances_stride, bound, 2,
                                    sz_group_shuffle_up_cdna3_, sz_group_shuffle_index_rocm_, STRINGZILLA_NULL,
                                    sz_lanes_broadcast_rocm_, sz_tile_queue_open_simt_, sz_tile_queue_draw_simt_);
}

static __global__ void __launch_bounds__(sz_levenshtein_candidates_per_block_max_simt_k)
    sz_levenshtein_distances_k3_cdna3_kernel_( //
        sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_strings_view_t candidates, sz_size_t *distances,
        sz_size_t distances_stride, sz_size_t bound) {
    sz_levenshtein_warp_sweep_simt_(engine, order, candidates, distances, distances_stride, bound, 3,
                                    sz_group_shuffle_up_cdna3_, sz_group_shuffle_index_rocm_, STRINGZILLA_NULL,
                                    sz_lanes_broadcast_rocm_, sz_tile_queue_open_simt_, sz_tile_queue_draw_simt_);
}

static __global__ void __launch_bounds__(sz_levenshtein_candidates_per_block_max_simt_k)
    sz_levenshtein_distances_k4_cdna3_kernel_( //
        sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_strings_view_t candidates, sz_size_t *distances,
        sz_size_t distances_stride, sz_size_t bound) {
    sz_levenshtein_warp_sweep_simt_(engine, order, candidates, distances, distances_stride, bound, 4,
                                    sz_group_shuffle_up_cdna3_, sz_group_shuffle_index_rocm_, STRINGZILLA_NULL,
                                    sz_lanes_broadcast_rocm_, sz_tile_queue_open_simt_, sz_tile_queue_draw_simt_);
}

static __global__ void __launch_bounds__(sz_levenshtein_candidates_per_block_max_simt_k)
    sz_levenshtein_distances_k5_cdna3_kernel_( //
        sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_strings_view_t candidates, sz_size_t *distances,
        sz_size_t distances_stride, sz_size_t bound) {
    sz_levenshtein_warp_sweep_simt_(engine, order, candidates, distances, distances_stride, bound, 5,
                                    sz_group_shuffle_up_cdna3_, sz_group_shuffle_index_rocm_, STRINGZILLA_NULL,
                                    sz_lanes_broadcast_rocm_, sz_tile_queue_open_simt_, sz_tile_queue_draw_simt_);
}

static __global__ void __launch_bounds__(sz_levenshtein_candidates_per_block_max_simt_k)
    sz_levenshtein_distances_k6_cdna3_kernel_( //
        sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_strings_view_t candidates, sz_size_t *distances,
        sz_size_t distances_stride, sz_size_t bound) {
    sz_levenshtein_warp_sweep_simt_(engine, order, candidates, distances, distances_stride, bound, 6,
                                    sz_group_shuffle_up_cdna3_, sz_group_shuffle_index_rocm_, STRINGZILLA_NULL,
                                    sz_lanes_broadcast_rocm_, sz_tile_queue_open_simt_, sz_tile_queue_draw_simt_);
}

static __global__ void __launch_bounds__(sz_levenshtein_candidates_per_block_max_simt_k)
    sz_levenshtein_distances_k7_cdna3_kernel_( //
        sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_strings_view_t candidates, sz_size_t *distances,
        sz_size_t distances_stride, sz_size_t bound) {
    sz_levenshtein_warp_sweep_simt_(engine, order, candidates, distances, distances_stride, bound, 7,
                                    sz_group_shuffle_up_cdna3_, sz_group_shuffle_index_rocm_, STRINGZILLA_NULL,
                                    sz_lanes_broadcast_rocm_, sz_tile_queue_open_simt_, sz_tile_queue_draw_simt_);
}

static __global__ void __launch_bounds__(sz_levenshtein_candidates_per_block_max_simt_k)
    sz_levenshtein_distances_k8_cdna3_kernel_( //
        sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_strings_view_t candidates, sz_size_t *distances,
        sz_size_t distances_stride, sz_size_t bound) {
    sz_levenshtein_warp_sweep_simt_(engine, order, candidates, distances, distances_stride, bound, 8,
                                    sz_group_shuffle_up_cdna3_, sz_group_shuffle_index_rocm_, STRINGZILLA_NULL,
                                    sz_lanes_broadcast_rocm_, sz_tile_queue_open_simt_, sz_tile_queue_draw_simt_);
}

/*  Warped rune rung: one entry point per words-per-lane. */
static __global__ void __launch_bounds__(sz_levenshtein_candidates_per_block_max_simt_k)
    sz_levenshtein_distances_utf8_k1_cdna3_kernel_( //
        sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_strings_view_t candidates, sz_size_t *distances,
        sz_size_t distances_stride, sz_size_t bound) {
    sz_levenshtein_warp_sweep_utf8_simt_(engine, order, candidates, distances, distances_stride, bound, 1,
                                         sz_group_shuffle_up_cdna3_, sz_group_shuffle_index_rocm_, STRINGZILLA_NULL,
                                         sz_lanes_broadcast_rocm_, sz_tile_queue_open_simt_, sz_tile_queue_draw_simt_);
}

static __global__ void __launch_bounds__(sz_levenshtein_candidates_per_block_max_simt_k)
    sz_levenshtein_distances_utf8_k2_cdna3_kernel_( //
        sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_strings_view_t candidates, sz_size_t *distances,
        sz_size_t distances_stride, sz_size_t bound) {
    sz_levenshtein_warp_sweep_utf8_simt_(engine, order, candidates, distances, distances_stride, bound, 2,
                                         sz_group_shuffle_up_cdna3_, sz_group_shuffle_index_rocm_, STRINGZILLA_NULL,
                                         sz_lanes_broadcast_rocm_, sz_tile_queue_open_simt_, sz_tile_queue_draw_simt_);
}

static __global__ void __launch_bounds__(sz_levenshtein_candidates_per_block_max_simt_k)
    sz_levenshtein_distances_utf8_k3_cdna3_kernel_( //
        sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_strings_view_t candidates, sz_size_t *distances,
        sz_size_t distances_stride, sz_size_t bound) {
    sz_levenshtein_warp_sweep_utf8_simt_(engine, order, candidates, distances, distances_stride, bound, 3,
                                         sz_group_shuffle_up_cdna3_, sz_group_shuffle_index_rocm_, STRINGZILLA_NULL,
                                         sz_lanes_broadcast_rocm_, sz_tile_queue_open_simt_, sz_tile_queue_draw_simt_);
}

static __global__ void __launch_bounds__(sz_levenshtein_candidates_per_block_max_simt_k)
    sz_levenshtein_distances_utf8_k4_cdna3_kernel_( //
        sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_strings_view_t candidates, sz_size_t *distances,
        sz_size_t distances_stride, sz_size_t bound) {
    sz_levenshtein_warp_sweep_utf8_simt_(engine, order, candidates, distances, distances_stride, bound, 4,
                                         sz_group_shuffle_up_cdna3_, sz_group_shuffle_index_rocm_, STRINGZILLA_NULL,
                                         sz_lanes_broadcast_rocm_, sz_tile_queue_open_simt_, sz_tile_queue_draw_simt_);
}

static __global__ void __launch_bounds__(sz_levenshtein_candidates_per_block_max_simt_k)
    sz_levenshtein_distances_utf8_k5_cdna3_kernel_( //
        sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_strings_view_t candidates, sz_size_t *distances,
        sz_size_t distances_stride, sz_size_t bound) {
    sz_levenshtein_warp_sweep_utf8_simt_(engine, order, candidates, distances, distances_stride, bound, 5,
                                         sz_group_shuffle_up_cdna3_, sz_group_shuffle_index_rocm_, STRINGZILLA_NULL,
                                         sz_lanes_broadcast_rocm_, sz_tile_queue_open_simt_, sz_tile_queue_draw_simt_);
}

static __global__ void __launch_bounds__(sz_levenshtein_candidates_per_block_max_simt_k)
    sz_levenshtein_distances_utf8_k6_cdna3_kernel_( //
        sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_strings_view_t candidates, sz_size_t *distances,
        sz_size_t distances_stride, sz_size_t bound) {
    sz_levenshtein_warp_sweep_utf8_simt_(engine, order, candidates, distances, distances_stride, bound, 6,
                                         sz_group_shuffle_up_cdna3_, sz_group_shuffle_index_rocm_, STRINGZILLA_NULL,
                                         sz_lanes_broadcast_rocm_, sz_tile_queue_open_simt_, sz_tile_queue_draw_simt_);
}

static __global__ void __launch_bounds__(sz_levenshtein_candidates_per_block_max_simt_k)
    sz_levenshtein_distances_utf8_k7_cdna3_kernel_( //
        sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_strings_view_t candidates, sz_size_t *distances,
        sz_size_t distances_stride, sz_size_t bound) {
    sz_levenshtein_warp_sweep_utf8_simt_(engine, order, candidates, distances, distances_stride, bound, 7,
                                         sz_group_shuffle_up_cdna3_, sz_group_shuffle_index_rocm_, STRINGZILLA_NULL,
                                         sz_lanes_broadcast_rocm_, sz_tile_queue_open_simt_, sz_tile_queue_draw_simt_);
}

static __global__ void __launch_bounds__(sz_levenshtein_candidates_per_block_max_simt_k)
    sz_levenshtein_distances_utf8_k8_cdna3_kernel_( //
        sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_strings_view_t candidates, sz_size_t *distances,
        sz_size_t distances_stride, sz_size_t bound) {
    sz_levenshtein_warp_sweep_utf8_simt_(engine, order, candidates, distances, distances_stride, bound, 8,
                                         sz_group_shuffle_up_cdna3_, sz_group_shuffle_index_rocm_, STRINGZILLA_NULL,
                                         sz_lanes_broadcast_rocm_, sz_tile_queue_open_simt_, sz_tile_queue_draw_simt_);
}

static __global__
__launch_bounds__(sz_levenshtein_tiled_threads_max_rocm_k) void sz_levenshtein_tiled_batch_cdna3_kernel_(
    sz_levenshtein_engine_t engine, sz_strings_view_t candidates, sz_size_t *distances, sz_ptr_t workspace,
    sz_levenshtein_long_arguments_simt_t batch) {
    sz_levenshtein_tiled_batch_simt_(engine, candidates, distances, workspace, batch, warpSize, sz_shuffle_up_cdna3_,
                                     sz_levenshtein_publish_rocm_, sz_levenshtein_await_rocm_);
}

/** The CDNA3 tier's entry points: the shared threaded rungs, and warped and tiled ones shifting
 *  through DPP. */
static sz_levenshtein_entry_points_rocm_t const sz_levenshtein_entry_points_cdna3_ = {
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
        (void const *)sz_levenshtein_distances_k1_cdna3_kernel_,
        (void const *)sz_levenshtein_distances_k2_cdna3_kernel_,
        (void const *)sz_levenshtein_distances_k3_cdna3_kernel_,
        (void const *)sz_levenshtein_distances_k4_cdna3_kernel_,
        (void const *)sz_levenshtein_distances_k5_cdna3_kernel_,
        (void const *)sz_levenshtein_distances_k6_cdna3_kernel_,
        (void const *)sz_levenshtein_distances_k7_cdna3_kernel_,
        (void const *)sz_levenshtein_distances_k8_cdna3_kernel_,
    },
    {
        (void const *)sz_levenshtein_distances_utf8_k1_cdna3_kernel_,
        (void const *)sz_levenshtein_distances_utf8_k2_cdna3_kernel_,
        (void const *)sz_levenshtein_distances_utf8_k3_cdna3_kernel_,
        (void const *)sz_levenshtein_distances_utf8_k4_cdna3_kernel_,
        (void const *)sz_levenshtein_distances_utf8_k5_cdna3_kernel_,
        (void const *)sz_levenshtein_distances_utf8_k6_cdna3_kernel_,
        (void const *)sz_levenshtein_distances_utf8_k7_cdna3_kernel_,
        (void const *)sz_levenshtein_distances_utf8_k8_cdna3_kernel_,
    },
    (void const *)sz_levenshtein_tiled_batch_cdna3_kernel_,
    sz_cap_cdna3_k,
};

#pragma endregion Entry Points

STRINGZILLA_API sz_status_t sz_levenshtein_engine_init_cdna3(sz_levenshtein_engine_t *engine,
                                                             sz_strings_view_t const *queries,
                                                             sz_levenshtein_symbol_t symbol, sz_allocator_t *allocator,
                                                             sz_stream_t stream) {
    return sz_levenshtein_engine_init_scoped_rocm_(engine, queries, symbol, &sz_levenshtein_entry_points_cdna3_,
                                                   allocator, stream);
}

STRINGZILLA_API sz_status_t sz_levenshtein_distances_cdna3(sz_levenshtein_engine_t *engine,
                                                           sz_strings_view_t const *candidates,
                                                           sz_size_t const *symbol_counts, sz_size_t const *order,
                                                           sz_size_t bound, sz_size_t *distances,
                                                           sz_size_t distances_stride, sz_stream_t stream) {
    sz_unused_(symbol_counts), sz_unused_(order);
    return sz_levenshtein_distances_scoped_rocm_(engine, candidates, distances, distances_stride, bound,
                                                 &sz_levenshtein_entry_points_cdna3_, stream);
}

#endif // STRINGZILLA_TARGET_CDNA3

#ifdef __cplusplus
}
#endif
#endif // STRINGZILLA_ARCH_ROCM_CDNA3_
#endif // STRINGZILLA_ARCH_ROCM_
#endif // STRINGZILLA_LEVENSHTEIN_CDNA3_CUH_
