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

#include "stringzilla/types.h"

#if STRINGZILLA_ARCH_ROCM_
#if STRINGZILLA_ARCH_ROCM_CDNA3_

#include "stringzilla/levenshtein/rocm.cuh"

#if defined(__cplusplus)
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

/*  Threaded byte rung: one entry point per word count. */
static __global__ void __launch_bounds__(sz_levenshtein_candidates_per_block_max_simt_k)
    sz_levenshtein_distances_w1_cdna3_kernel_( //
        sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_strings_view_t candidates, sz_size_t *distances,
        sz_size_t distances_stride, sz_size_t bound, sz_size_t const *candidate_order) {
    sz_levenshtein_sweep_simt_(engine, order, candidates, distances, distances_stride, bound, candidate_order, 1,
                               sz_tile_queue_open_simt_, sz_tile_queue_draw_simt_);
}

static __global__ void __launch_bounds__(sz_levenshtein_candidates_per_block_max_simt_k)
    sz_levenshtein_distances_w2_cdna3_kernel_( //
        sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_strings_view_t candidates, sz_size_t *distances,
        sz_size_t distances_stride, sz_size_t bound, sz_size_t const *candidate_order) {
    sz_levenshtein_sweep_simt_(engine, order, candidates, distances, distances_stride, bound, candidate_order, 2,
                               sz_tile_queue_open_simt_, sz_tile_queue_draw_simt_);
}

static __global__ void __launch_bounds__(sz_levenshtein_candidates_per_block_max_simt_k)
    sz_levenshtein_distances_w3_cdna3_kernel_( //
        sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_strings_view_t candidates, sz_size_t *distances,
        sz_size_t distances_stride, sz_size_t bound, sz_size_t const *candidate_order) {
    sz_levenshtein_sweep_simt_(engine, order, candidates, distances, distances_stride, bound, candidate_order, 3,
                               sz_tile_queue_open_simt_, sz_tile_queue_draw_simt_);
}

static __global__ void __launch_bounds__(sz_levenshtein_candidates_per_block_max_simt_k)
    sz_levenshtein_distances_w4_cdna3_kernel_( //
        sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_strings_view_t candidates, sz_size_t *distances,
        sz_size_t distances_stride, sz_size_t bound, sz_size_t const *candidate_order) {
    sz_levenshtein_sweep_simt_(engine, order, candidates, distances, distances_stride, bound, candidate_order, 4,
                               sz_tile_queue_open_simt_, sz_tile_queue_draw_simt_);
}

static __global__ void __launch_bounds__(sz_levenshtein_candidates_per_block_max_simt_k)
    sz_levenshtein_distances_w5_cdna3_kernel_( //
        sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_strings_view_t candidates, sz_size_t *distances,
        sz_size_t distances_stride, sz_size_t bound, sz_size_t const *candidate_order) {
    sz_levenshtein_sweep_simt_(engine, order, candidates, distances, distances_stride, bound, candidate_order, 5,
                               sz_tile_queue_open_simt_, sz_tile_queue_draw_simt_);
}

static __global__ void __launch_bounds__(sz_levenshtein_candidates_per_block_max_simt_k)
    sz_levenshtein_distances_w6_cdna3_kernel_( //
        sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_strings_view_t candidates, sz_size_t *distances,
        sz_size_t distances_stride, sz_size_t bound, sz_size_t const *candidate_order) {
    sz_levenshtein_sweep_simt_(engine, order, candidates, distances, distances_stride, bound, candidate_order, 6,
                               sz_tile_queue_open_simt_, sz_tile_queue_draw_simt_);
}

static __global__ void __launch_bounds__(sz_levenshtein_candidates_per_block_max_simt_k)
    sz_levenshtein_distances_w7_cdna3_kernel_( //
        sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_strings_view_t candidates, sz_size_t *distances,
        sz_size_t distances_stride, sz_size_t bound, sz_size_t const *candidate_order) {
    sz_levenshtein_sweep_simt_(engine, order, candidates, distances, distances_stride, bound, candidate_order, 7,
                               sz_tile_queue_open_simt_, sz_tile_queue_draw_simt_);
}

static __global__ void __launch_bounds__(sz_levenshtein_candidates_per_block_max_simt_k)
    sz_levenshtein_distances_w8_cdna3_kernel_( //
        sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_strings_view_t candidates, sz_size_t *distances,
        sz_size_t distances_stride, sz_size_t bound, sz_size_t const *candidate_order) {
    sz_levenshtein_sweep_simt_(engine, order, candidates, distances, distances_stride, bound, candidate_order, 8,
                               sz_tile_queue_open_simt_, sz_tile_queue_draw_simt_);
}

static __global__ void __launch_bounds__(sz_levenshtein_candidates_per_block_max_simt_k)
    sz_levenshtein_distances_w9_cdna3_kernel_( //
        sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_strings_view_t candidates, sz_size_t *distances,
        sz_size_t distances_stride, sz_size_t bound, sz_size_t const *candidate_order) {
    sz_levenshtein_sweep_simt_(engine, order, candidates, distances, distances_stride, bound, candidate_order, 9,
                               sz_tile_queue_open_simt_, sz_tile_queue_draw_simt_);
}

static __global__ void __launch_bounds__(sz_levenshtein_candidates_per_block_max_simt_k)
    sz_levenshtein_distances_w10_cdna3_kernel_( //
        sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_strings_view_t candidates, sz_size_t *distances,
        sz_size_t distances_stride, sz_size_t bound, sz_size_t const *candidate_order) {
    sz_levenshtein_sweep_simt_(engine, order, candidates, distances, distances_stride, bound, candidate_order, 10,
                               sz_tile_queue_open_simt_, sz_tile_queue_draw_simt_);
}

static __global__ void __launch_bounds__(sz_levenshtein_candidates_per_block_max_simt_k)
    sz_levenshtein_distances_w11_cdna3_kernel_( //
        sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_strings_view_t candidates, sz_size_t *distances,
        sz_size_t distances_stride, sz_size_t bound, sz_size_t const *candidate_order) {
    sz_levenshtein_sweep_simt_(engine, order, candidates, distances, distances_stride, bound, candidate_order, 11,
                               sz_tile_queue_open_simt_, sz_tile_queue_draw_simt_);
}

static __global__ void __launch_bounds__(sz_levenshtein_candidates_per_block_max_simt_k)
    sz_levenshtein_distances_w12_cdna3_kernel_( //
        sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_strings_view_t candidates, sz_size_t *distances,
        sz_size_t distances_stride, sz_size_t bound, sz_size_t const *candidate_order) {
    sz_levenshtein_sweep_simt_(engine, order, candidates, distances, distances_stride, bound, candidate_order, 12,
                               sz_tile_queue_open_simt_, sz_tile_queue_draw_simt_);
}

static __global__ void __launch_bounds__(sz_levenshtein_candidates_per_block_max_simt_k)
    sz_levenshtein_distances_w13_cdna3_kernel_( //
        sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_strings_view_t candidates, sz_size_t *distances,
        sz_size_t distances_stride, sz_size_t bound, sz_size_t const *candidate_order) {
    sz_levenshtein_sweep_simt_(engine, order, candidates, distances, distances_stride, bound, candidate_order, 13,
                               sz_tile_queue_open_simt_, sz_tile_queue_draw_simt_);
}

static __global__ void __launch_bounds__(sz_levenshtein_candidates_per_block_max_simt_k)
    sz_levenshtein_distances_w14_cdna3_kernel_( //
        sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_strings_view_t candidates, sz_size_t *distances,
        sz_size_t distances_stride, sz_size_t bound, sz_size_t const *candidate_order) {
    sz_levenshtein_sweep_simt_(engine, order, candidates, distances, distances_stride, bound, candidate_order, 14,
                               sz_tile_queue_open_simt_, sz_tile_queue_draw_simt_);
}

static __global__ void __launch_bounds__(sz_levenshtein_candidates_per_block_max_simt_k)
    sz_levenshtein_distances_w15_cdna3_kernel_( //
        sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_strings_view_t candidates, sz_size_t *distances,
        sz_size_t distances_stride, sz_size_t bound, sz_size_t const *candidate_order) {
    sz_levenshtein_sweep_simt_(engine, order, candidates, distances, distances_stride, bound, candidate_order, 15,
                               sz_tile_queue_open_simt_, sz_tile_queue_draw_simt_);
}

static __global__ void __launch_bounds__(sz_levenshtein_candidates_per_block_max_simt_k)
    sz_levenshtein_distances_w16_cdna3_kernel_( //
        sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_strings_view_t candidates, sz_size_t *distances,
        sz_size_t distances_stride, sz_size_t bound, sz_size_t const *candidate_order) {
    sz_levenshtein_sweep_simt_(engine, order, candidates, distances, distances_stride, bound, candidate_order, 16,
                               sz_tile_queue_open_simt_, sz_tile_queue_draw_simt_);
}

/*  Threaded rune rung: one entry point per word count. */
static __global__ void __launch_bounds__(sz_levenshtein_candidates_per_block_max_simt_k)
    sz_levenshtein_distances_utf8_w1_cdna3_kernel_( //
        sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_strings_view_t candidates, sz_size_t *distances,
        sz_size_t distances_stride, sz_size_t bound, sz_size_t const *candidate_order) {
    sz_levenshtein_sweep_utf8_simt_(engine, order, candidates, distances, distances_stride, bound, candidate_order, 1,
                                    sz_tile_queue_open_simt_, sz_tile_queue_draw_simt_);
}

static __global__ void __launch_bounds__(sz_levenshtein_candidates_per_block_max_simt_k)
    sz_levenshtein_distances_utf8_w2_cdna3_kernel_( //
        sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_strings_view_t candidates, sz_size_t *distances,
        sz_size_t distances_stride, sz_size_t bound, sz_size_t const *candidate_order) {
    sz_levenshtein_sweep_utf8_simt_(engine, order, candidates, distances, distances_stride, bound, candidate_order, 2,
                                    sz_tile_queue_open_simt_, sz_tile_queue_draw_simt_);
}

static __global__ void __launch_bounds__(sz_levenshtein_candidates_per_block_max_simt_k)
    sz_levenshtein_distances_utf8_w3_cdna3_kernel_( //
        sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_strings_view_t candidates, sz_size_t *distances,
        sz_size_t distances_stride, sz_size_t bound, sz_size_t const *candidate_order) {
    sz_levenshtein_sweep_utf8_simt_(engine, order, candidates, distances, distances_stride, bound, candidate_order, 3,
                                    sz_tile_queue_open_simt_, sz_tile_queue_draw_simt_);
}

static __global__ void __launch_bounds__(sz_levenshtein_candidates_per_block_max_simt_k)
    sz_levenshtein_distances_utf8_w4_cdna3_kernel_( //
        sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_strings_view_t candidates, sz_size_t *distances,
        sz_size_t distances_stride, sz_size_t bound, sz_size_t const *candidate_order) {
    sz_levenshtein_sweep_utf8_simt_(engine, order, candidates, distances, distances_stride, bound, candidate_order, 4,
                                    sz_tile_queue_open_simt_, sz_tile_queue_draw_simt_);
}

static __global__ void __launch_bounds__(sz_levenshtein_candidates_per_block_max_simt_k)
    sz_levenshtein_distances_utf8_w5_cdna3_kernel_( //
        sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_strings_view_t candidates, sz_size_t *distances,
        sz_size_t distances_stride, sz_size_t bound, sz_size_t const *candidate_order) {
    sz_levenshtein_sweep_utf8_simt_(engine, order, candidates, distances, distances_stride, bound, candidate_order, 5,
                                    sz_tile_queue_open_simt_, sz_tile_queue_draw_simt_);
}

static __global__ void __launch_bounds__(sz_levenshtein_candidates_per_block_max_simt_k)
    sz_levenshtein_distances_utf8_w6_cdna3_kernel_( //
        sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_strings_view_t candidates, sz_size_t *distances,
        sz_size_t distances_stride, sz_size_t bound, sz_size_t const *candidate_order) {
    sz_levenshtein_sweep_utf8_simt_(engine, order, candidates, distances, distances_stride, bound, candidate_order, 6,
                                    sz_tile_queue_open_simt_, sz_tile_queue_draw_simt_);
}

static __global__ void __launch_bounds__(sz_levenshtein_candidates_per_block_max_simt_k)
    sz_levenshtein_distances_utf8_w7_cdna3_kernel_( //
        sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_strings_view_t candidates, sz_size_t *distances,
        sz_size_t distances_stride, sz_size_t bound, sz_size_t const *candidate_order) {
    sz_levenshtein_sweep_utf8_simt_(engine, order, candidates, distances, distances_stride, bound, candidate_order, 7,
                                    sz_tile_queue_open_simt_, sz_tile_queue_draw_simt_);
}

static __global__ void __launch_bounds__(sz_levenshtein_candidates_per_block_max_simt_k)
    sz_levenshtein_distances_utf8_w8_cdna3_kernel_( //
        sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_strings_view_t candidates, sz_size_t *distances,
        sz_size_t distances_stride, sz_size_t bound, sz_size_t const *candidate_order) {
    sz_levenshtein_sweep_utf8_simt_(engine, order, candidates, distances, distances_stride, bound, candidate_order, 8,
                                    sz_tile_queue_open_simt_, sz_tile_queue_draw_simt_);
}

static __global__ void __launch_bounds__(sz_levenshtein_candidates_per_block_max_simt_k)
    sz_levenshtein_distances_utf8_w9_cdna3_kernel_( //
        sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_strings_view_t candidates, sz_size_t *distances,
        sz_size_t distances_stride, sz_size_t bound, sz_size_t const *candidate_order) {
    sz_levenshtein_sweep_utf8_simt_(engine, order, candidates, distances, distances_stride, bound, candidate_order, 9,
                                    sz_tile_queue_open_simt_, sz_tile_queue_draw_simt_);
}

static __global__ void __launch_bounds__(sz_levenshtein_candidates_per_block_max_simt_k)
    sz_levenshtein_distances_utf8_w10_cdna3_kernel_( //
        sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_strings_view_t candidates, sz_size_t *distances,
        sz_size_t distances_stride, sz_size_t bound, sz_size_t const *candidate_order) {
    sz_levenshtein_sweep_utf8_simt_(engine, order, candidates, distances, distances_stride, bound, candidate_order, 10,
                                    sz_tile_queue_open_simt_, sz_tile_queue_draw_simt_);
}

static __global__ void __launch_bounds__(sz_levenshtein_candidates_per_block_max_simt_k)
    sz_levenshtein_distances_utf8_w11_cdna3_kernel_( //
        sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_strings_view_t candidates, sz_size_t *distances,
        sz_size_t distances_stride, sz_size_t bound, sz_size_t const *candidate_order) {
    sz_levenshtein_sweep_utf8_simt_(engine, order, candidates, distances, distances_stride, bound, candidate_order, 11,
                                    sz_tile_queue_open_simt_, sz_tile_queue_draw_simt_);
}

static __global__ void __launch_bounds__(sz_levenshtein_candidates_per_block_max_simt_k)
    sz_levenshtein_distances_utf8_w12_cdna3_kernel_( //
        sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_strings_view_t candidates, sz_size_t *distances,
        sz_size_t distances_stride, sz_size_t bound, sz_size_t const *candidate_order) {
    sz_levenshtein_sweep_utf8_simt_(engine, order, candidates, distances, distances_stride, bound, candidate_order, 12,
                                    sz_tile_queue_open_simt_, sz_tile_queue_draw_simt_);
}

static __global__ void __launch_bounds__(sz_levenshtein_candidates_per_block_max_simt_k)
    sz_levenshtein_distances_utf8_w13_cdna3_kernel_( //
        sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_strings_view_t candidates, sz_size_t *distances,
        sz_size_t distances_stride, sz_size_t bound, sz_size_t const *candidate_order) {
    sz_levenshtein_sweep_utf8_simt_(engine, order, candidates, distances, distances_stride, bound, candidate_order, 13,
                                    sz_tile_queue_open_simt_, sz_tile_queue_draw_simt_);
}

static __global__ void __launch_bounds__(sz_levenshtein_candidates_per_block_max_simt_k)
    sz_levenshtein_distances_utf8_w14_cdna3_kernel_( //
        sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_strings_view_t candidates, sz_size_t *distances,
        sz_size_t distances_stride, sz_size_t bound, sz_size_t const *candidate_order) {
    sz_levenshtein_sweep_utf8_simt_(engine, order, candidates, distances, distances_stride, bound, candidate_order, 14,
                                    sz_tile_queue_open_simt_, sz_tile_queue_draw_simt_);
}

static __global__ void __launch_bounds__(sz_levenshtein_candidates_per_block_max_simt_k)
    sz_levenshtein_distances_utf8_w15_cdna3_kernel_( //
        sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_strings_view_t candidates, sz_size_t *distances,
        sz_size_t distances_stride, sz_size_t bound, sz_size_t const *candidate_order) {
    sz_levenshtein_sweep_utf8_simt_(engine, order, candidates, distances, distances_stride, bound, candidate_order, 15,
                                    sz_tile_queue_open_simt_, sz_tile_queue_draw_simt_);
}

static __global__ void __launch_bounds__(sz_levenshtein_candidates_per_block_max_simt_k)
    sz_levenshtein_distances_utf8_w16_cdna3_kernel_( //
        sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_strings_view_t candidates, sz_size_t *distances,
        sz_size_t distances_stride, sz_size_t bound, sz_size_t const *candidate_order) {
    sz_levenshtein_sweep_utf8_simt_(engine, order, candidates, distances, distances_stride, bound, candidate_order, 16,
                                    sz_tile_queue_open_simt_, sz_tile_queue_draw_simt_);
}

/*  Narrow byte rungs: one entry point per lane width. */
static __global__ void __launch_bounds__(sz_levenshtein_candidates_per_block_max_simt_k)
    sz_levenshtein_distances_u8x4_cdna3_kernel_(sz_levenshtein_engine_t engine, sz_u32_t const *order,
                                                sz_strings_view_t candidates, sz_size_t *distances,
                                                sz_size_t distances_stride, sz_size_t bound) {
    sz_levenshtein_lanes_sweep_simt_(engine, order, candidates, distances, distances_stride, bound, 8, 4);
}

static __global__ void __launch_bounds__(sz_levenshtein_candidates_per_block_max_simt_k)
    sz_levenshtein_distances_u16x2_cdna3_kernel_(sz_levenshtein_engine_t engine, sz_u32_t const *order,
                                                 sz_strings_view_t candidates, sz_size_t *distances,
                                                 sz_size_t distances_stride, sz_size_t bound) {
    sz_levenshtein_lanes_sweep_simt_(engine, order, candidates, distances, distances_stride, bound, 16, 2);
}

/** @ref sz_levenshtein_masks_simt_ , one block per query. */
static __global__ void __launch_bounds__(sz_levenshtein_masks_threads_simt_k)
    sz_levenshtein_masks_cdna3_kernel_(sz_levenshtein_engine_t engine, sz_string_view_t const *texts, sz_size_t first) {
    sz_levenshtein_masks_simt_(engine, texts, first);
}

/** @ref sz_levenshtein_length_counts_simt_ over a resident wave of blocks. */
static __global__ void __launch_bounds__(sz_levenshtein_length_keys_simt_k)
    sz_levenshtein_length_counts_cdna3_kernel_(sz_strings_view_t candidates, sz_size_t *counts) {
    sz_levenshtein_length_counts_simt_(candidates, counts);
}

/** @ref sz_levenshtein_length_offsets_rocm_ in one block. */
static __global__ void __launch_bounds__(sz_levenshtein_length_keys_simt_k)
    sz_levenshtein_length_offsets_cdna3_kernel_(sz_size_t *counts) {
    sz_levenshtein_length_offsets_rocm_(counts);
}

/** @ref sz_levenshtein_length_scatter_simt_. */
static __global__ void __launch_bounds__(sz_levenshtein_length_keys_simt_k)
    sz_levenshtein_length_scatter_cdna3_kernel_(sz_strings_view_t candidates, sz_size_t *cursors,
                                                sz_size_t *candidate_order) {
    sz_levenshtein_length_scatter_simt_(candidates, cursors, candidate_order, sz_u32_increment_rocm_);
}

/** @ref sz_levenshtein_order_simt_ over the long pairs' candidates. */
static __global__ void __launch_bounds__(sz_levenshtein_long_threads_simt_k)
    sz_levenshtein_order_cdna3_kernel_(sz_strings_view_t candidates, sz_levenshtein_candidate_buckets_simt_t buckets,
                                       sz_size_t *cursors, sz_size_t *order) {
    sz_levenshtein_order_simt_(candidates, buckets, cursors, order);
}

/** @ref sz_levenshtein_long_simt_ , one thread per pair. */
static __global__ void __launch_bounds__(sz_levenshtein_long_threads_simt_k)
    sz_levenshtein_long_cdna3_kernel_(sz_levenshtein_engine_t engine, sz_strings_view_t candidates,
                                      sz_size_t *distances, sz_ptr_t workspace,
                                      sz_levenshtein_long_arguments_simt_t batch) {
    sz_levenshtein_long_simt_(engine, candidates, distances, workspace, batch);
}

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

/** The CDNA3 tier's entry points: threaded rungs over the shared sweeps, and warped and tiled ones
 *  shifting through DPP. */
static sz_levenshtein_entry_points_rocm_t const sz_levenshtein_entry_points_cdna3_ = {
    {
        (void const *)sz_levenshtein_distances_w1_cdna3_kernel_,
        (void const *)sz_levenshtein_distances_w2_cdna3_kernel_,
        (void const *)sz_levenshtein_distances_w3_cdna3_kernel_,
        (void const *)sz_levenshtein_distances_w4_cdna3_kernel_,
        (void const *)sz_levenshtein_distances_w5_cdna3_kernel_,
        (void const *)sz_levenshtein_distances_w6_cdna3_kernel_,
        (void const *)sz_levenshtein_distances_w7_cdna3_kernel_,
        (void const *)sz_levenshtein_distances_w8_cdna3_kernel_,
        (void const *)sz_levenshtein_distances_w9_cdna3_kernel_,
        (void const *)sz_levenshtein_distances_w10_cdna3_kernel_,
        (void const *)sz_levenshtein_distances_w11_cdna3_kernel_,
        (void const *)sz_levenshtein_distances_w12_cdna3_kernel_,
        (void const *)sz_levenshtein_distances_w13_cdna3_kernel_,
        (void const *)sz_levenshtein_distances_w14_cdna3_kernel_,
        (void const *)sz_levenshtein_distances_w15_cdna3_kernel_,
        (void const *)sz_levenshtein_distances_w16_cdna3_kernel_,
    },
    {
        (void const *)sz_levenshtein_distances_utf8_w1_cdna3_kernel_,
        (void const *)sz_levenshtein_distances_utf8_w2_cdna3_kernel_,
        (void const *)sz_levenshtein_distances_utf8_w3_cdna3_kernel_,
        (void const *)sz_levenshtein_distances_utf8_w4_cdna3_kernel_,
        (void const *)sz_levenshtein_distances_utf8_w5_cdna3_kernel_,
        (void const *)sz_levenshtein_distances_utf8_w6_cdna3_kernel_,
        (void const *)sz_levenshtein_distances_utf8_w7_cdna3_kernel_,
        (void const *)sz_levenshtein_distances_utf8_w8_cdna3_kernel_,
        (void const *)sz_levenshtein_distances_utf8_w9_cdna3_kernel_,
        (void const *)sz_levenshtein_distances_utf8_w10_cdna3_kernel_,
        (void const *)sz_levenshtein_distances_utf8_w11_cdna3_kernel_,
        (void const *)sz_levenshtein_distances_utf8_w12_cdna3_kernel_,
        (void const *)sz_levenshtein_distances_utf8_w13_cdna3_kernel_,
        (void const *)sz_levenshtein_distances_utf8_w14_cdna3_kernel_,
        (void const *)sz_levenshtein_distances_utf8_w15_cdna3_kernel_,
        (void const *)sz_levenshtein_distances_utf8_w16_cdna3_kernel_,
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
    {
        (void const *)sz_levenshtein_distances_u8x4_cdna3_kernel_,
        (void const *)sz_levenshtein_distances_u16x2_cdna3_kernel_,
    },
    (void const *)sz_levenshtein_long_cdna3_kernel_,
    (void const *)sz_levenshtein_masks_cdna3_kernel_,
    (void const *)sz_levenshtein_length_counts_cdna3_kernel_,
    (void const *)sz_levenshtein_length_offsets_cdna3_kernel_,
    (void const *)sz_levenshtein_length_scatter_cdna3_kernel_,
    (void const *)sz_levenshtein_order_cdna3_kernel_,
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

#if defined(__cplusplus)
} // extern "C"
#endif
#endif // STRINGZILLA_ARCH_ROCM_CDNA3_
#endif // STRINGZILLA_ARCH_ROCM_
#endif // STRINGZILLA_LEVENSHTEIN_CDNA3_CUH_
