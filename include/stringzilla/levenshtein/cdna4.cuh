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

#include "stringzilla/types.h"

#if STRINGZILLA_ARCH_ROCM_
#if STRINGZILLA_ARCH_ROCM_CDNA4_

#include "stringzilla/levenshtein/cdna3.cuh" // `sz_shuffle_up_cdna3_`, `sz_group_shuffle_up_cdna3_`

#if defined(__cplusplus)
extern "C" {
#endif

#if STRINGZILLA_TARGET_CDNA4

#pragma region Entry Points

/*  Threaded byte rung: one entry point per word count. */
static __global__ void __launch_bounds__(sz_levenshtein_candidates_per_block_max_simt_k)
    sz_levenshtein_distances_w1_cdna4_kernel_( //
        sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_strings_view_t candidates, sz_size_t *distances,
        sz_size_t distances_stride, sz_size_t bound, sz_size_t const *candidate_order) {
    sz_levenshtein_sweep_simt_(engine, order, candidates, distances, distances_stride, bound, candidate_order, 1,
                               sz_tile_queue_open_simt_, sz_tile_queue_draw_simt_);
}

static __global__ void __launch_bounds__(sz_levenshtein_candidates_per_block_max_simt_k)
    sz_levenshtein_distances_w2_cdna4_kernel_( //
        sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_strings_view_t candidates, sz_size_t *distances,
        sz_size_t distances_stride, sz_size_t bound, sz_size_t const *candidate_order) {
    sz_levenshtein_sweep_simt_(engine, order, candidates, distances, distances_stride, bound, candidate_order, 2,
                               sz_tile_queue_open_simt_, sz_tile_queue_draw_simt_);
}

static __global__ void __launch_bounds__(sz_levenshtein_candidates_per_block_max_simt_k)
    sz_levenshtein_distances_w3_cdna4_kernel_( //
        sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_strings_view_t candidates, sz_size_t *distances,
        sz_size_t distances_stride, sz_size_t bound, sz_size_t const *candidate_order) {
    sz_levenshtein_sweep_simt_(engine, order, candidates, distances, distances_stride, bound, candidate_order, 3,
                               sz_tile_queue_open_simt_, sz_tile_queue_draw_simt_);
}

static __global__ void __launch_bounds__(sz_levenshtein_candidates_per_block_max_simt_k)
    sz_levenshtein_distances_w4_cdna4_kernel_( //
        sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_strings_view_t candidates, sz_size_t *distances,
        sz_size_t distances_stride, sz_size_t bound, sz_size_t const *candidate_order) {
    sz_levenshtein_sweep_simt_(engine, order, candidates, distances, distances_stride, bound, candidate_order, 4,
                               sz_tile_queue_open_simt_, sz_tile_queue_draw_simt_);
}

static __global__ void __launch_bounds__(sz_levenshtein_candidates_per_block_max_simt_k)
    sz_levenshtein_distances_w5_cdna4_kernel_( //
        sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_strings_view_t candidates, sz_size_t *distances,
        sz_size_t distances_stride, sz_size_t bound, sz_size_t const *candidate_order) {
    sz_levenshtein_sweep_simt_(engine, order, candidates, distances, distances_stride, bound, candidate_order, 5,
                               sz_tile_queue_open_simt_, sz_tile_queue_draw_simt_);
}

static __global__ void __launch_bounds__(sz_levenshtein_candidates_per_block_max_simt_k)
    sz_levenshtein_distances_w6_cdna4_kernel_( //
        sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_strings_view_t candidates, sz_size_t *distances,
        sz_size_t distances_stride, sz_size_t bound, sz_size_t const *candidate_order) {
    sz_levenshtein_sweep_simt_(engine, order, candidates, distances, distances_stride, bound, candidate_order, 6,
                               sz_tile_queue_open_simt_, sz_tile_queue_draw_simt_);
}

static __global__ void __launch_bounds__(sz_levenshtein_candidates_per_block_max_simt_k)
    sz_levenshtein_distances_w7_cdna4_kernel_( //
        sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_strings_view_t candidates, sz_size_t *distances,
        sz_size_t distances_stride, sz_size_t bound, sz_size_t const *candidate_order) {
    sz_levenshtein_sweep_simt_(engine, order, candidates, distances, distances_stride, bound, candidate_order, 7,
                               sz_tile_queue_open_simt_, sz_tile_queue_draw_simt_);
}

static __global__ void __launch_bounds__(sz_levenshtein_candidates_per_block_max_simt_k)
    sz_levenshtein_distances_w8_cdna4_kernel_( //
        sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_strings_view_t candidates, sz_size_t *distances,
        sz_size_t distances_stride, sz_size_t bound, sz_size_t const *candidate_order) {
    sz_levenshtein_sweep_simt_(engine, order, candidates, distances, distances_stride, bound, candidate_order, 8,
                               sz_tile_queue_open_simt_, sz_tile_queue_draw_simt_);
}

static __global__ void __launch_bounds__(sz_levenshtein_candidates_per_block_max_simt_k)
    sz_levenshtein_distances_w9_cdna4_kernel_( //
        sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_strings_view_t candidates, sz_size_t *distances,
        sz_size_t distances_stride, sz_size_t bound, sz_size_t const *candidate_order) {
    sz_levenshtein_sweep_simt_(engine, order, candidates, distances, distances_stride, bound, candidate_order, 9,
                               sz_tile_queue_open_simt_, sz_tile_queue_draw_simt_);
}

static __global__ void __launch_bounds__(sz_levenshtein_candidates_per_block_max_simt_k)
    sz_levenshtein_distances_w10_cdna4_kernel_( //
        sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_strings_view_t candidates, sz_size_t *distances,
        sz_size_t distances_stride, sz_size_t bound, sz_size_t const *candidate_order) {
    sz_levenshtein_sweep_simt_(engine, order, candidates, distances, distances_stride, bound, candidate_order, 10,
                               sz_tile_queue_open_simt_, sz_tile_queue_draw_simt_);
}

static __global__ void __launch_bounds__(sz_levenshtein_candidates_per_block_max_simt_k)
    sz_levenshtein_distances_w11_cdna4_kernel_( //
        sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_strings_view_t candidates, sz_size_t *distances,
        sz_size_t distances_stride, sz_size_t bound, sz_size_t const *candidate_order) {
    sz_levenshtein_sweep_simt_(engine, order, candidates, distances, distances_stride, bound, candidate_order, 11,
                               sz_tile_queue_open_simt_, sz_tile_queue_draw_simt_);
}

static __global__ void __launch_bounds__(sz_levenshtein_candidates_per_block_max_simt_k)
    sz_levenshtein_distances_w12_cdna4_kernel_( //
        sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_strings_view_t candidates, sz_size_t *distances,
        sz_size_t distances_stride, sz_size_t bound, sz_size_t const *candidate_order) {
    sz_levenshtein_sweep_simt_(engine, order, candidates, distances, distances_stride, bound, candidate_order, 12,
                               sz_tile_queue_open_simt_, sz_tile_queue_draw_simt_);
}

static __global__ void __launch_bounds__(sz_levenshtein_candidates_per_block_max_simt_k)
    sz_levenshtein_distances_w13_cdna4_kernel_( //
        sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_strings_view_t candidates, sz_size_t *distances,
        sz_size_t distances_stride, sz_size_t bound, sz_size_t const *candidate_order) {
    sz_levenshtein_sweep_simt_(engine, order, candidates, distances, distances_stride, bound, candidate_order, 13,
                               sz_tile_queue_open_simt_, sz_tile_queue_draw_simt_);
}

static __global__ void __launch_bounds__(sz_levenshtein_candidates_per_block_max_simt_k)
    sz_levenshtein_distances_w14_cdna4_kernel_( //
        sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_strings_view_t candidates, sz_size_t *distances,
        sz_size_t distances_stride, sz_size_t bound, sz_size_t const *candidate_order) {
    sz_levenshtein_sweep_simt_(engine, order, candidates, distances, distances_stride, bound, candidate_order, 14,
                               sz_tile_queue_open_simt_, sz_tile_queue_draw_simt_);
}

static __global__ void __launch_bounds__(sz_levenshtein_candidates_per_block_max_simt_k)
    sz_levenshtein_distances_w15_cdna4_kernel_( //
        sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_strings_view_t candidates, sz_size_t *distances,
        sz_size_t distances_stride, sz_size_t bound, sz_size_t const *candidate_order) {
    sz_levenshtein_sweep_simt_(engine, order, candidates, distances, distances_stride, bound, candidate_order, 15,
                               sz_tile_queue_open_simt_, sz_tile_queue_draw_simt_);
}

static __global__ void __launch_bounds__(sz_levenshtein_candidates_per_block_max_simt_k)
    sz_levenshtein_distances_w16_cdna4_kernel_( //
        sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_strings_view_t candidates, sz_size_t *distances,
        sz_size_t distances_stride, sz_size_t bound, sz_size_t const *candidate_order) {
    sz_levenshtein_sweep_simt_(engine, order, candidates, distances, distances_stride, bound, candidate_order, 16,
                               sz_tile_queue_open_simt_, sz_tile_queue_draw_simt_);
}

/*  Threaded rune rung: one entry point per word count. */
static __global__ void __launch_bounds__(sz_levenshtein_candidates_per_block_max_simt_k)
    sz_levenshtein_distances_utf8_w1_cdna4_kernel_( //
        sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_strings_view_t candidates, sz_size_t *distances,
        sz_size_t distances_stride, sz_size_t bound, sz_size_t const *candidate_order) {
    sz_levenshtein_sweep_utf8_simt_(engine, order, candidates, distances, distances_stride, bound, candidate_order, 1,
                                    sz_tile_queue_open_simt_, sz_tile_queue_draw_simt_);
}

static __global__ void __launch_bounds__(sz_levenshtein_candidates_per_block_max_simt_k)
    sz_levenshtein_distances_utf8_w2_cdna4_kernel_( //
        sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_strings_view_t candidates, sz_size_t *distances,
        sz_size_t distances_stride, sz_size_t bound, sz_size_t const *candidate_order) {
    sz_levenshtein_sweep_utf8_simt_(engine, order, candidates, distances, distances_stride, bound, candidate_order, 2,
                                    sz_tile_queue_open_simt_, sz_tile_queue_draw_simt_);
}

static __global__ void __launch_bounds__(sz_levenshtein_candidates_per_block_max_simt_k)
    sz_levenshtein_distances_utf8_w3_cdna4_kernel_( //
        sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_strings_view_t candidates, sz_size_t *distances,
        sz_size_t distances_stride, sz_size_t bound, sz_size_t const *candidate_order) {
    sz_levenshtein_sweep_utf8_simt_(engine, order, candidates, distances, distances_stride, bound, candidate_order, 3,
                                    sz_tile_queue_open_simt_, sz_tile_queue_draw_simt_);
}

static __global__ void __launch_bounds__(sz_levenshtein_candidates_per_block_max_simt_k)
    sz_levenshtein_distances_utf8_w4_cdna4_kernel_( //
        sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_strings_view_t candidates, sz_size_t *distances,
        sz_size_t distances_stride, sz_size_t bound, sz_size_t const *candidate_order) {
    sz_levenshtein_sweep_utf8_simt_(engine, order, candidates, distances, distances_stride, bound, candidate_order, 4,
                                    sz_tile_queue_open_simt_, sz_tile_queue_draw_simt_);
}

static __global__ void __launch_bounds__(sz_levenshtein_candidates_per_block_max_simt_k)
    sz_levenshtein_distances_utf8_w5_cdna4_kernel_( //
        sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_strings_view_t candidates, sz_size_t *distances,
        sz_size_t distances_stride, sz_size_t bound, sz_size_t const *candidate_order) {
    sz_levenshtein_sweep_utf8_simt_(engine, order, candidates, distances, distances_stride, bound, candidate_order, 5,
                                    sz_tile_queue_open_simt_, sz_tile_queue_draw_simt_);
}

static __global__ void __launch_bounds__(sz_levenshtein_candidates_per_block_max_simt_k)
    sz_levenshtein_distances_utf8_w6_cdna4_kernel_( //
        sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_strings_view_t candidates, sz_size_t *distances,
        sz_size_t distances_stride, sz_size_t bound, sz_size_t const *candidate_order) {
    sz_levenshtein_sweep_utf8_simt_(engine, order, candidates, distances, distances_stride, bound, candidate_order, 6,
                                    sz_tile_queue_open_simt_, sz_tile_queue_draw_simt_);
}

static __global__ void __launch_bounds__(sz_levenshtein_candidates_per_block_max_simt_k)
    sz_levenshtein_distances_utf8_w7_cdna4_kernel_( //
        sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_strings_view_t candidates, sz_size_t *distances,
        sz_size_t distances_stride, sz_size_t bound, sz_size_t const *candidate_order) {
    sz_levenshtein_sweep_utf8_simt_(engine, order, candidates, distances, distances_stride, bound, candidate_order, 7,
                                    sz_tile_queue_open_simt_, sz_tile_queue_draw_simt_);
}

static __global__ void __launch_bounds__(sz_levenshtein_candidates_per_block_max_simt_k)
    sz_levenshtein_distances_utf8_w8_cdna4_kernel_( //
        sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_strings_view_t candidates, sz_size_t *distances,
        sz_size_t distances_stride, sz_size_t bound, sz_size_t const *candidate_order) {
    sz_levenshtein_sweep_utf8_simt_(engine, order, candidates, distances, distances_stride, bound, candidate_order, 8,
                                    sz_tile_queue_open_simt_, sz_tile_queue_draw_simt_);
}

static __global__ void __launch_bounds__(sz_levenshtein_candidates_per_block_max_simt_k)
    sz_levenshtein_distances_utf8_w9_cdna4_kernel_( //
        sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_strings_view_t candidates, sz_size_t *distances,
        sz_size_t distances_stride, sz_size_t bound, sz_size_t const *candidate_order) {
    sz_levenshtein_sweep_utf8_simt_(engine, order, candidates, distances, distances_stride, bound, candidate_order, 9,
                                    sz_tile_queue_open_simt_, sz_tile_queue_draw_simt_);
}

static __global__ void __launch_bounds__(sz_levenshtein_candidates_per_block_max_simt_k)
    sz_levenshtein_distances_utf8_w10_cdna4_kernel_( //
        sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_strings_view_t candidates, sz_size_t *distances,
        sz_size_t distances_stride, sz_size_t bound, sz_size_t const *candidate_order) {
    sz_levenshtein_sweep_utf8_simt_(engine, order, candidates, distances, distances_stride, bound, candidate_order, 10,
                                    sz_tile_queue_open_simt_, sz_tile_queue_draw_simt_);
}

static __global__ void __launch_bounds__(sz_levenshtein_candidates_per_block_max_simt_k)
    sz_levenshtein_distances_utf8_w11_cdna4_kernel_( //
        sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_strings_view_t candidates, sz_size_t *distances,
        sz_size_t distances_stride, sz_size_t bound, sz_size_t const *candidate_order) {
    sz_levenshtein_sweep_utf8_simt_(engine, order, candidates, distances, distances_stride, bound, candidate_order, 11,
                                    sz_tile_queue_open_simt_, sz_tile_queue_draw_simt_);
}

static __global__ void __launch_bounds__(sz_levenshtein_candidates_per_block_max_simt_k)
    sz_levenshtein_distances_utf8_w12_cdna4_kernel_( //
        sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_strings_view_t candidates, sz_size_t *distances,
        sz_size_t distances_stride, sz_size_t bound, sz_size_t const *candidate_order) {
    sz_levenshtein_sweep_utf8_simt_(engine, order, candidates, distances, distances_stride, bound, candidate_order, 12,
                                    sz_tile_queue_open_simt_, sz_tile_queue_draw_simt_);
}

static __global__ void __launch_bounds__(sz_levenshtein_candidates_per_block_max_simt_k)
    sz_levenshtein_distances_utf8_w13_cdna4_kernel_( //
        sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_strings_view_t candidates, sz_size_t *distances,
        sz_size_t distances_stride, sz_size_t bound, sz_size_t const *candidate_order) {
    sz_levenshtein_sweep_utf8_simt_(engine, order, candidates, distances, distances_stride, bound, candidate_order, 13,
                                    sz_tile_queue_open_simt_, sz_tile_queue_draw_simt_);
}

static __global__ void __launch_bounds__(sz_levenshtein_candidates_per_block_max_simt_k)
    sz_levenshtein_distances_utf8_w14_cdna4_kernel_( //
        sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_strings_view_t candidates, sz_size_t *distances,
        sz_size_t distances_stride, sz_size_t bound, sz_size_t const *candidate_order) {
    sz_levenshtein_sweep_utf8_simt_(engine, order, candidates, distances, distances_stride, bound, candidate_order, 14,
                                    sz_tile_queue_open_simt_, sz_tile_queue_draw_simt_);
}

static __global__ void __launch_bounds__(sz_levenshtein_candidates_per_block_max_simt_k)
    sz_levenshtein_distances_utf8_w15_cdna4_kernel_( //
        sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_strings_view_t candidates, sz_size_t *distances,
        sz_size_t distances_stride, sz_size_t bound, sz_size_t const *candidate_order) {
    sz_levenshtein_sweep_utf8_simt_(engine, order, candidates, distances, distances_stride, bound, candidate_order, 15,
                                    sz_tile_queue_open_simt_, sz_tile_queue_draw_simt_);
}

static __global__ void __launch_bounds__(sz_levenshtein_candidates_per_block_max_simt_k)
    sz_levenshtein_distances_utf8_w16_cdna4_kernel_( //
        sz_levenshtein_engine_t engine, sz_u32_t const *order, sz_strings_view_t candidates, sz_size_t *distances,
        sz_size_t distances_stride, sz_size_t bound, sz_size_t const *candidate_order) {
    sz_levenshtein_sweep_utf8_simt_(engine, order, candidates, distances, distances_stride, bound, candidate_order, 16,
                                    sz_tile_queue_open_simt_, sz_tile_queue_draw_simt_);
}

/*  Narrow byte rungs: one entry point per lane width. */
static __global__ void __launch_bounds__(sz_levenshtein_candidates_per_block_max_simt_k)
    sz_levenshtein_distances_u8x4_cdna4_kernel_(sz_levenshtein_engine_t engine, sz_u32_t const *order,
                                                sz_strings_view_t candidates, sz_size_t *distances,
                                                sz_size_t distances_stride, sz_size_t bound) {
    sz_levenshtein_lanes_sweep_simt_(engine, order, candidates, distances, distances_stride, bound, 8, 4);
}

static __global__ void __launch_bounds__(sz_levenshtein_candidates_per_block_max_simt_k)
    sz_levenshtein_distances_u16x2_cdna4_kernel_(sz_levenshtein_engine_t engine, sz_u32_t const *order,
                                                 sz_strings_view_t candidates, sz_size_t *distances,
                                                 sz_size_t distances_stride, sz_size_t bound) {
    sz_levenshtein_lanes_sweep_simt_(engine, order, candidates, distances, distances_stride, bound, 16, 2);
}

/** @ref sz_levenshtein_masks_simt_ , one block per query. */
static __global__ void __launch_bounds__(sz_levenshtein_masks_threads_simt_k)
    sz_levenshtein_masks_cdna4_kernel_(sz_levenshtein_engine_t engine, sz_string_view_t const *texts, sz_size_t first) {
    sz_levenshtein_masks_simt_(engine, texts, first);
}

/** @ref sz_levenshtein_length_counts_simt_ over a resident wave of blocks. */
static __global__ void __launch_bounds__(sz_levenshtein_length_keys_simt_k)
    sz_levenshtein_length_counts_cdna4_kernel_(sz_strings_view_t candidates, sz_size_t *counts) {
    sz_levenshtein_length_counts_simt_(candidates, counts);
}

/** @ref sz_levenshtein_length_offsets_rocm_ in one block. */
static __global__ void __launch_bounds__(sz_levenshtein_length_keys_simt_k)
    sz_levenshtein_length_offsets_cdna4_kernel_(sz_size_t *counts) {
    sz_levenshtein_length_offsets_rocm_(counts);
}

/** @ref sz_levenshtein_length_scatter_simt_. */
static __global__ void __launch_bounds__(sz_levenshtein_length_keys_simt_k)
    sz_levenshtein_length_scatter_cdna4_kernel_(sz_strings_view_t candidates, sz_size_t *cursors,
                                                sz_size_t *candidate_order) {
    sz_levenshtein_length_scatter_simt_(candidates, cursors, candidate_order, sz_u32_increment_rocm_);
}

/** @ref sz_levenshtein_order_simt_ over the long pairs' candidates. */
static __global__ void __launch_bounds__(sz_levenshtein_long_threads_simt_k)
    sz_levenshtein_order_cdna4_kernel_(sz_strings_view_t candidates, sz_levenshtein_candidate_buckets_simt_t buckets,
                                       sz_size_t *cursors, sz_size_t *order) {
    sz_levenshtein_order_simt_(candidates, buckets, cursors, order);
}

/** @ref sz_levenshtein_long_simt_ , one thread per pair. */
static __global__ void __launch_bounds__(sz_levenshtein_long_threads_simt_k)
    sz_levenshtein_long_cdna4_kernel_(sz_levenshtein_engine_t engine, sz_strings_view_t candidates,
                                      sz_size_t *distances, sz_ptr_t workspace,
                                      sz_levenshtein_long_arguments_simt_t batch) {
    sz_levenshtein_long_simt_(engine, candidates, distances, workspace, batch);
}

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

/** The CDNA4 tier's entry points: threaded rungs over the shared sweeps, and warped and tiled ones
 *  shifting through CDNA3's DPP wavefront shift. */
static sz_levenshtein_entry_points_rocm_t const sz_levenshtein_entry_points_cdna4_ = {
    {
        (void const *)sz_levenshtein_distances_w1_cdna4_kernel_,
        (void const *)sz_levenshtein_distances_w2_cdna4_kernel_,
        (void const *)sz_levenshtein_distances_w3_cdna4_kernel_,
        (void const *)sz_levenshtein_distances_w4_cdna4_kernel_,
        (void const *)sz_levenshtein_distances_w5_cdna4_kernel_,
        (void const *)sz_levenshtein_distances_w6_cdna4_kernel_,
        (void const *)sz_levenshtein_distances_w7_cdna4_kernel_,
        (void const *)sz_levenshtein_distances_w8_cdna4_kernel_,
        (void const *)sz_levenshtein_distances_w9_cdna4_kernel_,
        (void const *)sz_levenshtein_distances_w10_cdna4_kernel_,
        (void const *)sz_levenshtein_distances_w11_cdna4_kernel_,
        (void const *)sz_levenshtein_distances_w12_cdna4_kernel_,
        (void const *)sz_levenshtein_distances_w13_cdna4_kernel_,
        (void const *)sz_levenshtein_distances_w14_cdna4_kernel_,
        (void const *)sz_levenshtein_distances_w15_cdna4_kernel_,
        (void const *)sz_levenshtein_distances_w16_cdna4_kernel_,
    },
    {
        (void const *)sz_levenshtein_distances_utf8_w1_cdna4_kernel_,
        (void const *)sz_levenshtein_distances_utf8_w2_cdna4_kernel_,
        (void const *)sz_levenshtein_distances_utf8_w3_cdna4_kernel_,
        (void const *)sz_levenshtein_distances_utf8_w4_cdna4_kernel_,
        (void const *)sz_levenshtein_distances_utf8_w5_cdna4_kernel_,
        (void const *)sz_levenshtein_distances_utf8_w6_cdna4_kernel_,
        (void const *)sz_levenshtein_distances_utf8_w7_cdna4_kernel_,
        (void const *)sz_levenshtein_distances_utf8_w8_cdna4_kernel_,
        (void const *)sz_levenshtein_distances_utf8_w9_cdna4_kernel_,
        (void const *)sz_levenshtein_distances_utf8_w10_cdna4_kernel_,
        (void const *)sz_levenshtein_distances_utf8_w11_cdna4_kernel_,
        (void const *)sz_levenshtein_distances_utf8_w12_cdna4_kernel_,
        (void const *)sz_levenshtein_distances_utf8_w13_cdna4_kernel_,
        (void const *)sz_levenshtein_distances_utf8_w14_cdna4_kernel_,
        (void const *)sz_levenshtein_distances_utf8_w15_cdna4_kernel_,
        (void const *)sz_levenshtein_distances_utf8_w16_cdna4_kernel_,
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
    {
        (void const *)sz_levenshtein_distances_u8x4_cdna4_kernel_,
        (void const *)sz_levenshtein_distances_u16x2_cdna4_kernel_,
    },
    (void const *)sz_levenshtein_long_cdna4_kernel_,
    (void const *)sz_levenshtein_masks_cdna4_kernel_,
    (void const *)sz_levenshtein_length_counts_cdna4_kernel_,
    (void const *)sz_levenshtein_length_offsets_cdna4_kernel_,
    (void const *)sz_levenshtein_length_scatter_cdna4_kernel_,
    (void const *)sz_levenshtein_order_cdna4_kernel_,
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

#if defined(__cplusplus)
} // extern "C"
#endif
#endif // STRINGZILLA_ARCH_ROCM_CDNA4_
#endif // STRINGZILLA_ARCH_ROCM_
#endif // STRINGZILLA_LEVENSHTEIN_CDNA4_CUH_
