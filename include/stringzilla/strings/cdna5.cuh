/**
 *  @file include/stringzilla/strings/cdna5.cuh
 *  @author Ash Vardanian
 *  @date November 14, 2026
 *  @brief The wavefront scans of 32-lane CDNA5, which other families' CDNA5 tiers include: DPP row
 *      steps whose two rows swap totals through @c v_permlanex16 , as CDNA5 broadcasts nothing.
 *
 *  A CDNA5 wavefront is 32 lanes, so its group steps scan the whole wavefront, every lane calling
 *  each step at once.
 *
 *  @sa include/stringzilla/strings/cdna3.cuh
 */
#ifndef STRINGZILLA_STRINGS_CDNA5_CUH_
#define STRINGZILLA_STRINGS_CDNA5_CUH_

#include "stringzilla/rocm.cuh"

#if STRINGZILLA_ARCH_ROCM_
#if STRINGZILLA_TARGET_CDNA5

#ifdef __cplusplus
extern "C" {
#endif

/** The sum of @p value over this lane and every lane below it among the wavefront's 32 lanes: each
 *  row takes lane 15 of the other through @c v_permlanex16 , and only row 1 adds it. */
STRINGZILLA_DEVICE sz_u32_t sz_u32_lanes_prefix_sum_cdna5_(sz_u32_t value) {
    sz_u32_t other_row;
    value = sz_u32_rows_prefix_sum_rocm_(value);
    other_row = (sz_u32_t)__builtin_amdgcn_permlanex16((int)value, (int)value, 0xFFFFFFFFu, 0xFFFFFFFFu, false, false);
    return value + __builtin_amdgcn_update_dpp(0u, other_row, sz_dpp_identity_rocm_k, 0x2, 0xF, false);
}

/** @ref sz_u32_lanes_prefix_sum_cdna5_ over 64-bit values, @c v_permlanex16 moving each half. */
STRINGZILLA_DEVICE sz_u64_t sz_u64_lanes_prefix_sum_cdna5_(sz_u64_t value) {
    sz_u32_t low, high;
    value = sz_u64_rows_prefix_sum_rocm_(value);
    low = (sz_u32_t)__builtin_amdgcn_permlanex16((int)value, (int)value, 0xFFFFFFFFu, 0xFFFFFFFFu, false, false);
    high = (sz_u32_t)__builtin_amdgcn_permlanex16((int)(value >> 32), (int)(value >> 32), 0xFFFFFFFFu, 0xFFFFFFFFu,
                                                  false, false);
    return value + __builtin_amdgcn_update_dpp((sz_u64_t)0, (sz_u64_t)high << 32 | low, sz_dpp_identity_rocm_k, 0x2,
                                               0xF, false);
}

/** The least of @p value over this lane and every lane below it among the wavefront's 32 lanes. */
STRINGZILLA_DEVICE sz_i32_t sz_i32_lanes_prefix_min_cdna5_(sz_i32_t value) {
    sz_i32_t other_row, below;
    value = sz_i32_rows_prefix_min_rocm_(value);
    other_row = __builtin_amdgcn_permlanex16(value, value, 0xFFFFFFFFu, 0xFFFFFFFFu, false, false);
    below = __builtin_amdgcn_update_dpp(0x7FFFFFFF, other_row, sz_dpp_identity_rocm_k, 0x2, 0xF, false);
    return below < value ? below : value;
}

/** The sum of @p value over the wavefront's 32 lanes, in every one of them. */
STRINGZILLA_DEVICE sz_u32_t sz_u32_lanes_sum_cdna5_(sz_u32_t value) {
    return (sz_u32_t)__builtin_amdgcn_readlane((int)sz_u32_lanes_prefix_sum_cdna5_(value), 31);
}

/** @ref sz_u32_lanes_sum_cdna5_ over 64-bit values, each half read off lane 31. */
STRINGZILLA_DEVICE sz_u64_t sz_u64_lanes_sum_cdna5_(sz_u64_t value) {
    sz_u64_t const inclusive = sz_u64_lanes_prefix_sum_cdna5_(value);
    return (sz_u64_t)(sz_u32_t)__builtin_amdgcn_readlane((int)(inclusive >> 32), 31) << 32 |
           (sz_u32_t)__builtin_amdgcn_readlane((int)inclusive, 31);
}

/** @ref sz_block_scan_simt_ over 32-lane wavefronts scanning themselves through DPP. */
STRINGZILLA_DEVICE sz_size_t sz_block_scan_cdna5_(sz_size_t value, sz_size_t *shared, sz_size_t *total,
                                                  unsigned threads) {
    return sz_block_scan_simt_(value, shared, total, 32, threads, sz_u64_lanes_prefix_sum_cdna5_);
}

/** @ref sz_block_scan_bounded_simt_ over 32-lane wavefronts scanning themselves through DPP, one
 *  move per step. */
STRINGZILLA_DEVICE sz_size_t sz_block_scan_bounded_cdna5_(sz_size_t value, sz_size_t *shared, sz_size_t *total,
                                                          unsigned threads) {
    return sz_block_scan_bounded_simt_(value, (sz_u32_t *)shared, total, 32, threads, sz_u32_lanes_prefix_sum_cdna5_);
}

/** @ref sz_block_sum_simt_ over 32-lane wavefronts summing themselves through DPP. */
STRINGZILLA_DEVICE sz_size_t sz_block_sum_cdna5_(sz_size_t value, sz_size_t *shared, unsigned threads) {
    return sz_block_sum_simt_(value, shared, 32, threads, sz_u64_lanes_prefix_sum_cdna5_);
}

#ifdef __cplusplus
} // extern "C"
#endif

#endif // STRINGZILLA_TARGET_CDNA5
#endif // STRINGZILLA_ARCH_ROCM_
#endif // STRINGZILLA_STRINGS_CDNA5_CUH_
