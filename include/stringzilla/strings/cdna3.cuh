/**
 *  @file include/stringzilla/strings/cdna3.cuh
 *  @author Ash Vardanian
 *  @date November 14, 2026
 *  @brief The wavefront scans of 64-lane CDNA3 and CDNA4, which other families' CDNA tiers include:
 *      DPP row steps carried across rows by the GFX9 row broadcasts.
 *
 *  Each scan runs within its row of 16 lanes through @c sz_u32_rows_prefix_sum_rocm_ and kin, then
 *  carries the row totals on. Every lane of the wavefront calls a step at once.
 *
 *  @sa include/stringzilla/rocm.cuh
 */
#ifndef STRINGZILLA_STRINGS_CDNA3_CUH_
#define STRINGZILLA_STRINGS_CDNA3_CUH_

#include "stringzilla/rocm.cuh"

#if STRINGZILLA_ARCH_ROCM_
#if STRINGZILLA_ARCH_ROCM_CDNA3_

#ifdef __cplusplus
extern "C" {
#endif

enum {

    /** The DPP control of 64-lane wavefronts moving lane 15 of each row into the row after it. */
    sz_dpp_row_broadcast_15_cdna3_k = 0x142,

    /** The DPP control of 64-lane wavefronts moving lane 31 into rows 2 and 3. */
    sz_dpp_row_broadcast_31_cdna3_k = 0x143,
};

/** The sum of @p value over this lane and every lane below it among this thread's 32 lanes: rows 1
 *  and 3 add the last lane of the row before, which stops at each half of the wavefront. */
STRINGZILLA_DEVICE sz_u32_t sz_u32_lanes_prefix_sum_cdna3_(sz_u32_t value) {
    value = sz_u32_rows_prefix_sum_rocm_(value);
    return value + __builtin_amdgcn_update_dpp(0u, value, sz_dpp_row_broadcast_15_cdna3_k, 0xA, 0xF, false);
}

/** @ref sz_u32_lanes_prefix_sum_cdna3_ over 64-bit values, each move two DPP moves. */
STRINGZILLA_DEVICE sz_u64_t sz_u64_lanes_prefix_sum_cdna3_(sz_u64_t value) {
    value = sz_u64_rows_prefix_sum_rocm_(value);
    return value + __builtin_amdgcn_update_dpp((sz_u64_t)0, value, sz_dpp_row_broadcast_15_cdna3_k, 0xA, 0xF, false);
}

/** The least of @p value over this lane and every lane below it among this thread's 32 lanes. */
STRINGZILLA_DEVICE sz_i32_t sz_i32_lanes_prefix_min_cdna3_(sz_i32_t value) {
    sz_i32_t below;
    value = sz_i32_rows_prefix_min_rocm_(value);
    below = __builtin_amdgcn_update_dpp(0x7FFFFFFF, value, sz_dpp_row_broadcast_15_cdna3_k, 0xA, 0xF, false);
    return below < value ? below : value;
}

/** The sum of @p value over this thread's 32 lanes, in every one of them, read off the last lane of
 *  its half of the wavefront. */
STRINGZILLA_DEVICE sz_u32_t sz_u32_lanes_sum_cdna3_(sz_u32_t value) {
    int const inclusive = (int)sz_u32_lanes_prefix_sum_cdna3_(value);
    return (sz_u32_t)(__lane_id() & 32u ? __builtin_amdgcn_readlane(inclusive, 63)
                                        : __builtin_amdgcn_readlane(inclusive, 31));
}

/** @ref sz_u32_lanes_sum_cdna3_ over 64-bit values, each half read off its last lane. */
STRINGZILLA_DEVICE sz_u64_t sz_u64_lanes_sum_cdna3_(sz_u64_t value) {
    sz_u64_t const inclusive = sz_u64_lanes_prefix_sum_cdna3_(value);
    int const low = (int)inclusive, high = (int)(inclusive >> 32);
    return __lane_id() & 32u ? (sz_u64_t)(sz_u32_t)__builtin_amdgcn_readlane(high, 63) << 32 |
                                   (sz_u32_t)__builtin_amdgcn_readlane(low, 63)
                             : (sz_u64_t)(sz_u32_t)__builtin_amdgcn_readlane(high, 31) << 32 |
                                   (sz_u32_t)__builtin_amdgcn_readlane(low, 31);
}

/** The sum of @p value over this lane and every lane below it across the 64-lane wavefront: past
 *  each half's own scan, rows 2 and 3 add lane 31. */
STRINGZILLA_DEVICE sz_u32_t sz_u32_wavefront_prefix_sum_cdna3(sz_u32_t value) {
    value = sz_u32_lanes_prefix_sum_cdna3_(value);
    return value + __builtin_amdgcn_update_dpp(0u, value, sz_dpp_row_broadcast_31_cdna3_k, 0xC, 0xF, false);
}

/** @ref sz_u32_wavefront_prefix_sum_cdna3 over 64-bit values, each move two DPP moves. */
STRINGZILLA_DEVICE sz_u64_t sz_u64_wavefront_prefix_sum_cdna3(sz_u64_t value) {
    value = sz_u64_lanes_prefix_sum_cdna3_(value);
    return value + __builtin_amdgcn_update_dpp((sz_u64_t)0, value, sz_dpp_row_broadcast_31_cdna3_k, 0xC, 0xF, false);
}

/** The least of @p value over this lane and every lane below it across the 64-lane wavefront. */
STRINGZILLA_DEVICE sz_i32_t sz_i32_wavefront_prefix_min_cdna3_(sz_i32_t value) {
    sz_i32_t below;
    value = sz_i32_lanes_prefix_min_cdna3_(value);
    below = __builtin_amdgcn_update_dpp(0x7FFFFFFF, value, sz_dpp_row_broadcast_31_cdna3_k, 0xC, 0xF, false);
    return below < value ? below : value;
}

/** The sum of @p value over the 64-lane wavefront, in every lane. */
STRINGZILLA_DEVICE sz_u32_t sz_u32_wavefront_sum_cdna3_(sz_u32_t value) {
    return (sz_u32_t)__builtin_amdgcn_readlane((int)sz_u32_wavefront_prefix_sum_cdna3(value), 63);
}

/** @ref sz_block_scan_simt_ over 64-lane wavefronts scanning themselves through DPP. */
STRINGZILLA_DEVICE sz_size_t sz_block_scan_cdna3_(sz_size_t value, sz_size_t *shared, sz_size_t *total,
                                                  unsigned threads) {
    return sz_block_scan_simt_(value, shared, total, 64, threads, sz_u64_wavefront_prefix_sum_cdna3);
}

/** @ref sz_block_scan_bounded_simt_ over 64-lane wavefronts scanning themselves through DPP, one
 *  move per step. */
STRINGZILLA_DEVICE sz_size_t sz_block_scan_bounded_cdna3_(sz_size_t value, sz_size_t *shared, sz_size_t *total,
                                                          unsigned threads) {
    return sz_block_scan_bounded_simt_(value, (sz_u32_t *)shared, total, 64, threads,
                                       sz_u32_wavefront_prefix_sum_cdna3);
}

/** @ref sz_block_sum_simt_ over 64-lane wavefronts summing themselves through DPP. */
STRINGZILLA_DEVICE sz_size_t sz_block_sum_cdna3_(sz_size_t value, sz_size_t *shared, unsigned threads) {
    return sz_block_sum_simt_(value, shared, 64, threads, sz_u64_wavefront_prefix_sum_cdna3);
}

#ifdef __cplusplus
} // extern "C"
#endif
#endif // STRINGZILLA_ARCH_ROCM_CDNA3_
#endif // STRINGZILLA_ARCH_ROCM_
#endif // STRINGZILLA_STRINGS_CDNA3_CUH_
