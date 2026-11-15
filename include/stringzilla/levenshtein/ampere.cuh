/**
 *  @file include/stringzilla/levenshtein/ampere.cuh
 *  @author Ash Vardanian
 *  @date November 14, 2026
 *  @brief Ampere tier of Levenshtein distances: the warped rungs summing a candidate's deltas with
 *      @c redux.sync , and the @c _ampere exports over the CUDA host side.
 *
 *  An unbounded warped candidate ends by summing its lanes' vertical deltas, which the CUDA tier
 *  scans up the warp and broadcasts, and this tier reduces in one instruction. The tiled wavefront
 *  keeps the CUDA tier's staging: the query bytes a tile-row reads load beside the frontier band,
 *  which only the acquire that ends the spin may read, so copying them before the spin would hide
 *  no latency the band's loads do not already pay.
 *
 *  @sa include/stringzilla/levenshtein/cuda.cuh
 */
#ifndef STRINGZILLA_LEVENSHTEIN_AMPERE_CUH_
#define STRINGZILLA_LEVENSHTEIN_AMPERE_CUH_

#include "stringzilla/levenshtein/cuda.cuh"
#include "stringzilla/strings/ampere.cuh" // `sz_u32_lanes_sum_ampere_`

#if STRINGZILLA_ARCH_CUDA_
#if STRINGZILLA_ARCH_CUDA_AMPERE_

#ifdef __cplusplus
extern "C" {
#endif

#if STRINGZILLA_TARGET_AMPERE

#pragma region Entry Points

/*  Warped byte rung: one entry point per words-per-lane. */
static __global__ void __launch_bounds__(sz_levenshtein_candidates_per_block_max_simt_k)
    sz_levenshtein_distances_k1_ampere_kernel_( //
        sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_strings_view_t candidates, sz_size_t *distances,
        sz_size_t distances_stride, sz_size_t bound) {
    sz_levenshtein_warp_sweep_simt_(engine, order, candidates, distances, distances_stride, bound, 1,
                                    sz_group_shuffle_up_cuda_, sz_group_shuffle_index_cuda_, sz_u32_lanes_sum_ampere_,
                                    sz_lanes_broadcast_cuda_, sz_tile_queue_open_simt_, sz_tile_queue_draw_simt_);
}

static __global__ void __launch_bounds__(sz_levenshtein_candidates_per_block_max_simt_k)
    sz_levenshtein_distances_k2_ampere_kernel_( //
        sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_strings_view_t candidates, sz_size_t *distances,
        sz_size_t distances_stride, sz_size_t bound) {
    sz_levenshtein_warp_sweep_simt_(engine, order, candidates, distances, distances_stride, bound, 2,
                                    sz_group_shuffle_up_cuda_, sz_group_shuffle_index_cuda_, sz_u32_lanes_sum_ampere_,
                                    sz_lanes_broadcast_cuda_, sz_tile_queue_open_simt_, sz_tile_queue_draw_simt_);
}

static __global__ void __launch_bounds__(sz_levenshtein_candidates_per_block_max_simt_k)
    sz_levenshtein_distances_k3_ampere_kernel_( //
        sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_strings_view_t candidates, sz_size_t *distances,
        sz_size_t distances_stride, sz_size_t bound) {
    sz_levenshtein_warp_sweep_simt_(engine, order, candidates, distances, distances_stride, bound, 3,
                                    sz_group_shuffle_up_cuda_, sz_group_shuffle_index_cuda_, sz_u32_lanes_sum_ampere_,
                                    sz_lanes_broadcast_cuda_, sz_tile_queue_open_simt_, sz_tile_queue_draw_simt_);
}

static __global__ void __launch_bounds__(sz_levenshtein_candidates_per_block_max_simt_k)
    sz_levenshtein_distances_k4_ampere_kernel_( //
        sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_strings_view_t candidates, sz_size_t *distances,
        sz_size_t distances_stride, sz_size_t bound) {
    sz_levenshtein_warp_sweep_simt_(engine, order, candidates, distances, distances_stride, bound, 4,
                                    sz_group_shuffle_up_cuda_, sz_group_shuffle_index_cuda_, sz_u32_lanes_sum_ampere_,
                                    sz_lanes_broadcast_cuda_, sz_tile_queue_open_simt_, sz_tile_queue_draw_simt_);
}

static __global__ void __launch_bounds__(sz_levenshtein_candidates_per_block_max_simt_k)
    sz_levenshtein_distances_k5_ampere_kernel_( //
        sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_strings_view_t candidates, sz_size_t *distances,
        sz_size_t distances_stride, sz_size_t bound) {
    sz_levenshtein_warp_sweep_simt_(engine, order, candidates, distances, distances_stride, bound, 5,
                                    sz_group_shuffle_up_cuda_, sz_group_shuffle_index_cuda_, sz_u32_lanes_sum_ampere_,
                                    sz_lanes_broadcast_cuda_, sz_tile_queue_open_simt_, sz_tile_queue_draw_simt_);
}

static __global__ void __launch_bounds__(sz_levenshtein_candidates_per_block_max_simt_k)
    sz_levenshtein_distances_k6_ampere_kernel_( //
        sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_strings_view_t candidates, sz_size_t *distances,
        sz_size_t distances_stride, sz_size_t bound) {
    sz_levenshtein_warp_sweep_simt_(engine, order, candidates, distances, distances_stride, bound, 6,
                                    sz_group_shuffle_up_cuda_, sz_group_shuffle_index_cuda_, sz_u32_lanes_sum_ampere_,
                                    sz_lanes_broadcast_cuda_, sz_tile_queue_open_simt_, sz_tile_queue_draw_simt_);
}

static __global__ void __launch_bounds__(sz_levenshtein_candidates_per_block_max_simt_k)
    sz_levenshtein_distances_k7_ampere_kernel_( //
        sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_strings_view_t candidates, sz_size_t *distances,
        sz_size_t distances_stride, sz_size_t bound) {
    sz_levenshtein_warp_sweep_simt_(engine, order, candidates, distances, distances_stride, bound, 7,
                                    sz_group_shuffle_up_cuda_, sz_group_shuffle_index_cuda_, sz_u32_lanes_sum_ampere_,
                                    sz_lanes_broadcast_cuda_, sz_tile_queue_open_simt_, sz_tile_queue_draw_simt_);
}

static __global__ void __launch_bounds__(sz_levenshtein_candidates_per_block_max_simt_k)
    sz_levenshtein_distances_k8_ampere_kernel_( //
        sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_strings_view_t candidates, sz_size_t *distances,
        sz_size_t distances_stride, sz_size_t bound) {
    sz_levenshtein_warp_sweep_simt_(engine, order, candidates, distances, distances_stride, bound, 8,
                                    sz_group_shuffle_up_cuda_, sz_group_shuffle_index_cuda_, sz_u32_lanes_sum_ampere_,
                                    sz_lanes_broadcast_cuda_, sz_tile_queue_open_simt_, sz_tile_queue_draw_simt_);
}

/*  Warped rune rung: one entry point per words-per-lane. */
static __global__ void __launch_bounds__(sz_levenshtein_candidates_per_block_max_simt_k)
    sz_levenshtein_distances_utf8_k1_ampere_kernel_( //
        sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_strings_view_t candidates, sz_size_t *distances,
        sz_size_t distances_stride, sz_size_t bound) {
    sz_levenshtein_warp_sweep_utf8_simt_(engine, order, candidates, distances, distances_stride, bound, 1,
                                         sz_group_shuffle_up_cuda_, sz_group_shuffle_index_cuda_,
                                         sz_u32_lanes_sum_ampere_, sz_lanes_broadcast_cuda_, sz_tile_queue_open_simt_,
                                         sz_tile_queue_draw_simt_);
}

static __global__ void __launch_bounds__(sz_levenshtein_candidates_per_block_max_simt_k)
    sz_levenshtein_distances_utf8_k2_ampere_kernel_( //
        sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_strings_view_t candidates, sz_size_t *distances,
        sz_size_t distances_stride, sz_size_t bound) {
    sz_levenshtein_warp_sweep_utf8_simt_(engine, order, candidates, distances, distances_stride, bound, 2,
                                         sz_group_shuffle_up_cuda_, sz_group_shuffle_index_cuda_,
                                         sz_u32_lanes_sum_ampere_, sz_lanes_broadcast_cuda_, sz_tile_queue_open_simt_,
                                         sz_tile_queue_draw_simt_);
}

static __global__ void __launch_bounds__(sz_levenshtein_candidates_per_block_max_simt_k)
    sz_levenshtein_distances_utf8_k3_ampere_kernel_( //
        sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_strings_view_t candidates, sz_size_t *distances,
        sz_size_t distances_stride, sz_size_t bound) {
    sz_levenshtein_warp_sweep_utf8_simt_(engine, order, candidates, distances, distances_stride, bound, 3,
                                         sz_group_shuffle_up_cuda_, sz_group_shuffle_index_cuda_,
                                         sz_u32_lanes_sum_ampere_, sz_lanes_broadcast_cuda_, sz_tile_queue_open_simt_,
                                         sz_tile_queue_draw_simt_);
}

static __global__ void __launch_bounds__(sz_levenshtein_candidates_per_block_max_simt_k)
    sz_levenshtein_distances_utf8_k4_ampere_kernel_( //
        sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_strings_view_t candidates, sz_size_t *distances,
        sz_size_t distances_stride, sz_size_t bound) {
    sz_levenshtein_warp_sweep_utf8_simt_(engine, order, candidates, distances, distances_stride, bound, 4,
                                         sz_group_shuffle_up_cuda_, sz_group_shuffle_index_cuda_,
                                         sz_u32_lanes_sum_ampere_, sz_lanes_broadcast_cuda_, sz_tile_queue_open_simt_,
                                         sz_tile_queue_draw_simt_);
}

static __global__ void __launch_bounds__(sz_levenshtein_candidates_per_block_max_simt_k)
    sz_levenshtein_distances_utf8_k5_ampere_kernel_( //
        sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_strings_view_t candidates, sz_size_t *distances,
        sz_size_t distances_stride, sz_size_t bound) {
    sz_levenshtein_warp_sweep_utf8_simt_(engine, order, candidates, distances, distances_stride, bound, 5,
                                         sz_group_shuffle_up_cuda_, sz_group_shuffle_index_cuda_,
                                         sz_u32_lanes_sum_ampere_, sz_lanes_broadcast_cuda_, sz_tile_queue_open_simt_,
                                         sz_tile_queue_draw_simt_);
}

static __global__ void __launch_bounds__(sz_levenshtein_candidates_per_block_max_simt_k)
    sz_levenshtein_distances_utf8_k6_ampere_kernel_( //
        sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_strings_view_t candidates, sz_size_t *distances,
        sz_size_t distances_stride, sz_size_t bound) {
    sz_levenshtein_warp_sweep_utf8_simt_(engine, order, candidates, distances, distances_stride, bound, 6,
                                         sz_group_shuffle_up_cuda_, sz_group_shuffle_index_cuda_,
                                         sz_u32_lanes_sum_ampere_, sz_lanes_broadcast_cuda_, sz_tile_queue_open_simt_,
                                         sz_tile_queue_draw_simt_);
}

static __global__ void __launch_bounds__(sz_levenshtein_candidates_per_block_max_simt_k)
    sz_levenshtein_distances_utf8_k7_ampere_kernel_( //
        sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_strings_view_t candidates, sz_size_t *distances,
        sz_size_t distances_stride, sz_size_t bound) {
    sz_levenshtein_warp_sweep_utf8_simt_(engine, order, candidates, distances, distances_stride, bound, 7,
                                         sz_group_shuffle_up_cuda_, sz_group_shuffle_index_cuda_,
                                         sz_u32_lanes_sum_ampere_, sz_lanes_broadcast_cuda_, sz_tile_queue_open_simt_,
                                         sz_tile_queue_draw_simt_);
}

static __global__ void __launch_bounds__(sz_levenshtein_candidates_per_block_max_simt_k)
    sz_levenshtein_distances_utf8_k8_ampere_kernel_( //
        sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_strings_view_t candidates, sz_size_t *distances,
        sz_size_t distances_stride, sz_size_t bound) {
    sz_levenshtein_warp_sweep_utf8_simt_(engine, order, candidates, distances, distances_stride, bound, 8,
                                         sz_group_shuffle_up_cuda_, sz_group_shuffle_index_cuda_,
                                         sz_u32_lanes_sum_ampere_, sz_lanes_broadcast_cuda_, sz_tile_queue_open_simt_,
                                         sz_tile_queue_draw_simt_);
}

static __global__
__launch_bounds__(sz_levenshtein_tiled_threads_per_block_cuda_k) void sz_levenshtein_tiled_batch_ampere_kernel_(
    sz_levenshtein_engine_t engine, sz_strings_view_t candidates, sz_size_t *distances, sz_ptr_t workspace,
    sz_levenshtein_long_arguments_simt_t batch) {
    sz_levenshtein_tiled_batch_simt_(engine, candidates, distances, workspace, batch, sz_levenshtein_lanes_simt_k,
                                     sz_shuffle_up_cuda_, sz_levenshtein_publish_cuda_, sz_levenshtein_await_cuda_);
}

/** The Ampere tier's entry points: the shared threaded rungs, warped ones summing their final
 *  deltas in one @c redux.sync , and the tiled one of the CUDA tier. */
static sz_levenshtein_entry_points_cuda_t const sz_levenshtein_entry_points_ampere_ = {
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
        (void const *)sz_levenshtein_distances_k1_ampere_kernel_,
        (void const *)sz_levenshtein_distances_k2_ampere_kernel_,
        (void const *)sz_levenshtein_distances_k3_ampere_kernel_,
        (void const *)sz_levenshtein_distances_k4_ampere_kernel_,
        (void const *)sz_levenshtein_distances_k5_ampere_kernel_,
        (void const *)sz_levenshtein_distances_k6_ampere_kernel_,
        (void const *)sz_levenshtein_distances_k7_ampere_kernel_,
        (void const *)sz_levenshtein_distances_k8_ampere_kernel_,
    },
    {
        (void const *)sz_levenshtein_distances_utf8_k1_ampere_kernel_,
        (void const *)sz_levenshtein_distances_utf8_k2_ampere_kernel_,
        (void const *)sz_levenshtein_distances_utf8_k3_ampere_kernel_,
        (void const *)sz_levenshtein_distances_utf8_k4_ampere_kernel_,
        (void const *)sz_levenshtein_distances_utf8_k5_ampere_kernel_,
        (void const *)sz_levenshtein_distances_utf8_k6_ampere_kernel_,
        (void const *)sz_levenshtein_distances_utf8_k7_ampere_kernel_,
        (void const *)sz_levenshtein_distances_utf8_k8_ampere_kernel_,
    },
    (void const *)sz_levenshtein_tiled_batch_ampere_kernel_,
    sz_cap_ampere_k,
};

#pragma endregion Entry Points

STRINGZILLA_API sz_status_t sz_levenshtein_engine_init_ampere(sz_levenshtein_engine_t *engine,
                                                              sz_strings_view_t const *queries,
                                                              sz_levenshtein_symbol_t symbol, sz_allocator_t *allocator,
                                                              sz_stream_t stream) {
    return sz_levenshtein_engine_init_scoped_cuda_(engine, queries, symbol, &sz_levenshtein_entry_points_ampere_,
                                                   allocator, stream);
}

STRINGZILLA_API sz_status_t sz_levenshtein_distances_ampere(sz_levenshtein_engine_t *engine,
                                                            sz_strings_view_t const *candidates,
                                                            sz_size_t const *symbol_counts, sz_size_t const *order,
                                                            sz_size_t bound, sz_size_t *distances,
                                                            sz_size_t distances_stride, sz_stream_t stream) {
    sz_unused_(symbol_counts), sz_unused_(order);
    return sz_levenshtein_distances_scoped_cuda_(engine, candidates, distances, distances_stride, bound,
                                                 &sz_levenshtein_entry_points_ampere_, stream);
}

#endif // STRINGZILLA_TARGET_AMPERE

#ifdef __cplusplus
}
#endif
#endif // STRINGZILLA_ARCH_CUDA_AMPERE_
#endif // STRINGZILLA_ARCH_CUDA_
#endif // STRINGZILLA_LEVENSHTEIN_AMPERE_CUH_
