/**
 *  @brief NEON (Arm) backend for sorting string collections.
 *  @file include/stringzilla/sort/neon.h
 *  @author Ash Vardanian
 *  @sa include/stringzilla/sort.h
 *
 *  Mirrors the SVE backend's out-of-place 3-way QuickSort partition, but NEON has no `svcompact`, so the
 *  per-block compaction uses a small `static const` left-pack table fed to `vqtbl2q_u8` (4 u64 = 32 bytes =
 *  exactly one `uint8x16x2_t`, so one table lookup gathers any subset of 4 lanes to the front; a second
 *  moves the order indices). The 4-wide table is 16x32 = 512 bytes - small enough to keep `const` and warm
 *  in cache - versus the 16 KB an 8-wide table would need. Compaction preserves lane order, so the partition
 *  is @b stable - matching the stable-by-default contract - and the recursion/stability/reverse/top-K/
 *  uncased machinery is reused verbatim from `sort/serial.h`.
 */
#ifndef STRINGZILLA_SORT_NEON_H_
#define STRINGZILLA_SORT_NEON_H_

#include "stringzilla/types.h"
#include "stringzilla/compare.h" // `sz_compare`
#include "stringzilla/memory.h"  // `sz_copy_neon`

#include "stringzilla/sort/serial.h"

#ifdef __cplusplus
extern "C" {
#endif

#if SZ_USE_NEON
#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("+simd"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("+simd")
#endif

/** @brief Collapses two `uint64x2_t` compare results (lanes are 0 or ~0) into a 4-bit lane mask. */
SZ_INTERNAL sz_u32_t sz_sort_neon_lane_mask4_(uint64x2_t lower_u64x2, uint64x2_t upper_u64x2) {
    static sz_u16_t const lane_weights[4] = {1, 2, 4, 8};
    uint16x4_t flags_u16x4 = vmovn_u32(vcombine_u32(vmovn_u64(lower_u64x2), vmovn_u64(upper_u64x2)));
    return (sz_u32_t)vaddv_u16(vand_u16(flags_u16x4, vld1_u16(lane_weights)));
}

/** @brief Left-packs the @p mask4 -selected lanes of one 4-u64 block (keys @p keys_u8x16x2 and order
 *         @p order_u8x16x2) to the @p out_pgrams / @p out_order cursors, preserving order; returns how many
 *         lanes were written. Both half-vectors are stored unconditionally (the cursor only advances by the
 *         surviving count, so the unwritten tail is overwritten by the next call or lands in the region
 *         slack): on a wide out-of-order core the branchless full-vector stores beat data-dependent
 *         "store only what survives" branches, whose misprediction cost dwarfs the few wasted stores. */
SZ_INTERNAL sz_size_t sz_sort_neon_compact4_(                          //
    uint8x16x2_t const keys_u8x16x2, uint8x16x2_t const order_u8x16x2, //
    sz_u32_t const mask4, sz_pgram_t *const out_pgrams, sz_sorted_idx_t *const out_order) {

    // 16x32 = 512-byte left-pack table for `vqtbl2q_u8`: row `[m]` holds the 32 byte indices that gather the
    // `m`-selected u64 lanes (of 4) to the front - first 16 bytes feed lookup 0 (out lanes 0-1), last 16 feed
    // lookup 1 (out lanes 2-3). 512 bytes stays cache-warm; an 8-wide `vqtbl4q` table would need 16 KB.
    static sz_u8_t const compact_lut[16][32] = {
        {0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7},
        {0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7},
        {8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7},
        {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7},
        {16, 17, 18, 19, 20, 21, 22, 23, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7},
        {0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7},
        {8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7},
        {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 0, 1, 2, 3, 4, 5, 6, 7},
        {24, 25, 26, 27, 28, 29, 30, 31, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7},
        {0, 1, 2, 3, 4, 5, 6, 7, 24, 25, 26, 27, 28, 29, 30, 31, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7},
        {8, 9, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29, 30, 31, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7},
        {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29, 30, 31, 0, 1, 2, 3, 4, 5, 6, 7},
        {16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
         0,  1,  2,  3,  4,  5,  6,  7,  0,  1,  2,  3,  4,  5,  6,  7},
        {0,  1,  2,  3,  4,  5,  6,  7,  16, 17, 18, 19, 20, 21, 22, 23,
         24, 25, 26, 27, 28, 29, 30, 31, 0,  1,  2,  3,  4,  5,  6,  7},
        {8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
         24, 25, 26, 27, 28, 29, 30, 31, 0,  1,  2,  3,  4,  5,  6,  7},
        {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15,
         16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31},
    };

    // Unlike x86, AArch64 has no scalar population-count instruction (`cnt` is vector-only), so a 16-byte LUT
    // beats spilling the 4-bit mask to a vector for one `cnt`+`addv`.
    static sz_u8_t const popcount_lut[16] = {0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4};

    sz_size_t const taken = popcount_lut[mask4];
    sz_u8_t const *const indices = compact_lut[mask4];
    uint8x16_t const indices_u8x16 = vld1q_u8(indices);
    uint8x16_t const indices_high_u8x16 = vld1q_u8(indices + 16);
    vst1q_u64((sz_u64_t *)(out_pgrams), vreinterpretq_u64_u8(vqtbl2q_u8(keys_u8x16x2, indices_u8x16)));
    vst1q_u64((sz_u64_t *)(out_order), vreinterpretq_u64_u8(vqtbl2q_u8(order_u8x16x2, indices_u8x16)));
    vst1q_u64((sz_u64_t *)(out_pgrams + 2), vreinterpretq_u64_u8(vqtbl2q_u8(keys_u8x16x2, indices_high_u8x16)));
    vst1q_u64((sz_u64_t *)(out_order + 2), vreinterpretq_u64_u8(vqtbl2q_u8(order_u8x16x2, indices_high_u8x16)));
    return taken;
}

/**
 *  @brief 3-way partition around the Sedgewick pivot using NEON table-compaction.
 *  @note Out-of-place into @p partitioned_* with three regions (smaller, equal, greater) laid out by the
 *        count pass and separated by 8-lane slack gaps that absorb each region's trailing full-vector spill;
 *        the three regions are then copied back contiguously. A single block-major pass left-packs all three
 *        comparison kinds together, so each block is loaded once and the equal mask is derived for free.
 */
SZ_INTERNAL void sz_sequence_argsort_neon_3way_partition_(                          //
    sz_pgram_t *const initial_pgrams, sz_sorted_idx_t *const initial_order,         //
    sz_pgram_t *const partitioned_pgrams, sz_sorted_idx_t *const partitioned_order, //
    sz_size_t const start_in_sequence, sz_size_t const end_in_sequence,             //
    sz_size_t *const first_pivot_offset, sz_size_t *const last_pivot_offset) {

    sz_size_t const count = end_in_sequence - start_in_sequence;
    sz_pgram_t const pivot_pgram = *sz_sequence_partitioning_pivot_(initial_pgrams + start_in_sequence, count);
    uint64x2_t const pivot_u64x2 = vdupq_n_u64(pivot_pgram);

    // Count pass: tally smaller and greater lanes by accumulating the compare masks (each lane 0 or ~0, so
    // subtracting it adds one per set lane), then horizontally reduce - no scalar population counts needed.
    uint64x2_t smaller_accumulator_u64x2 = vdupq_n_u64(0), greater_accumulator_u64x2 = vdupq_n_u64(0);
    sz_size_t i = start_in_sequence;
    for (; i + 8 <= end_in_sequence; i += 8) {
        uint64x2_t keys0_u64x2 = vld1q_u64((sz_u64_t const *)(initial_pgrams + i));
        uint64x2_t keys1_u64x2 = vld1q_u64((sz_u64_t const *)(initial_pgrams + i + 2));
        uint64x2_t keys2_u64x2 = vld1q_u64((sz_u64_t const *)(initial_pgrams + i + 4));
        uint64x2_t keys3_u64x2 = vld1q_u64((sz_u64_t const *)(initial_pgrams + i + 6));
        smaller_accumulator_u64x2 = vsubq_u64(smaller_accumulator_u64x2, vcltq_u64(keys0_u64x2, pivot_u64x2));
        smaller_accumulator_u64x2 = vsubq_u64(smaller_accumulator_u64x2, vcltq_u64(keys1_u64x2, pivot_u64x2));
        smaller_accumulator_u64x2 = vsubq_u64(smaller_accumulator_u64x2, vcltq_u64(keys2_u64x2, pivot_u64x2));
        smaller_accumulator_u64x2 = vsubq_u64(smaller_accumulator_u64x2, vcltq_u64(keys3_u64x2, pivot_u64x2));
        greater_accumulator_u64x2 = vsubq_u64(greater_accumulator_u64x2, vcgtq_u64(keys0_u64x2, pivot_u64x2));
        greater_accumulator_u64x2 = vsubq_u64(greater_accumulator_u64x2, vcgtq_u64(keys1_u64x2, pivot_u64x2));
        greater_accumulator_u64x2 = vsubq_u64(greater_accumulator_u64x2, vcgtq_u64(keys2_u64x2, pivot_u64x2));
        greater_accumulator_u64x2 = vsubq_u64(greater_accumulator_u64x2, vcgtq_u64(keys3_u64x2, pivot_u64x2));
    }
    sz_size_t count_smaller = vaddvq_u64(smaller_accumulator_u64x2);
    sz_size_t count_greater = vaddvq_u64(greater_accumulator_u64x2);
    for (; i < end_in_sequence; ++i)
        count_smaller += initial_pgrams[i]<pivot_pgram, count_greater += initial_pgrams[i]> pivot_pgram;
    sz_size_t const count_equal = count - count_smaller - count_greater;

    // Lay the three regions out with 8-lane slack gaps; each region's final compaction may spill up to a
    // full vector past its true end, and the slack (plus the buffer's trailing slack) absorbs it.
    sz_size_t const equal_region = count_smaller + 8;
    sz_size_t const greater_region = equal_region + count_equal + 8;
    sz_pgram_t *smaller_pgrams = partitioned_pgrams + start_in_sequence;
    sz_sorted_idx_t *smaller_order = partitioned_order + start_in_sequence;
    sz_pgram_t *equal_pgrams = smaller_pgrams + equal_region;
    sz_sorted_idx_t *equal_order = smaller_order + equal_region;
    sz_pgram_t *greater_pgrams = smaller_pgrams + greater_region;
    sz_sorted_idx_t *greater_order = smaller_order + greater_region;

    // Block-major compaction: each 8-lane block is loaded once and dispatched to all three regions.
    i = start_in_sequence;
    for (; i + 8 <= end_in_sequence; i += 8) {
        uint64x2_t keys0_u64x2 = vld1q_u64((sz_u64_t const *)(initial_pgrams + i));
        uint64x2_t keys1_u64x2 = vld1q_u64((sz_u64_t const *)(initial_pgrams + i + 2));
        uint64x2_t keys2_u64x2 = vld1q_u64((sz_u64_t const *)(initial_pgrams + i + 4));
        uint64x2_t keys3_u64x2 = vld1q_u64((sz_u64_t const *)(initial_pgrams + i + 6));
        uint8x16x2_t keys_lower_u8x16x2 = {{vreinterpretq_u8_u64(keys0_u64x2), vreinterpretq_u8_u64(keys1_u64x2)}};
        uint8x16x2_t keys_upper_u8x16x2 = {{vreinterpretq_u8_u64(keys2_u64x2), vreinterpretq_u8_u64(keys3_u64x2)}};
        uint8x16x2_t order_lower_u8x16x2 = {
            {vreinterpretq_u8_u64(vld1q_u64((sz_u64_t const *)(initial_order + i))),
             vreinterpretq_u8_u64(vld1q_u64((sz_u64_t const *)(initial_order + i + 2)))}};
        uint8x16x2_t order_upper_u8x16x2 = {
            {vreinterpretq_u8_u64(vld1q_u64((sz_u64_t const *)(initial_order + i + 4))),
             vreinterpretq_u8_u64(vld1q_u64((sz_u64_t const *)(initial_order + i + 6)))}};

        // Eight compares per block; the equal mask is the complement of (smaller | greater), so no third set.
        sz_u32_t smaller_lower = sz_sort_neon_lane_mask4_(vcltq_u64(keys0_u64x2, pivot_u64x2),
                                                          vcltq_u64(keys1_u64x2, pivot_u64x2));
        sz_u32_t smaller_upper = sz_sort_neon_lane_mask4_(vcltq_u64(keys2_u64x2, pivot_u64x2),
                                                          vcltq_u64(keys3_u64x2, pivot_u64x2));
        sz_u32_t greater_lower = sz_sort_neon_lane_mask4_(vcgtq_u64(keys0_u64x2, pivot_u64x2),
                                                          vcgtq_u64(keys1_u64x2, pivot_u64x2));
        sz_u32_t greater_upper = sz_sort_neon_lane_mask4_(vcgtq_u64(keys2_u64x2, pivot_u64x2),
                                                          vcgtq_u64(keys3_u64x2, pivot_u64x2));
        sz_u32_t equal_lower = (~(smaller_lower | greater_lower)) & 0xF;
        sz_u32_t equal_upper = (~(smaller_upper | greater_upper)) & 0xF;

        sz_size_t taken;
        taken = sz_sort_neon_compact4_(keys_lower_u8x16x2, order_lower_u8x16x2, smaller_lower, smaller_pgrams,
                                       smaller_order);
        smaller_pgrams += taken, smaller_order += taken;
        taken = sz_sort_neon_compact4_(keys_upper_u8x16x2, order_upper_u8x16x2, smaller_upper, smaller_pgrams,
                                       smaller_order);
        smaller_pgrams += taken, smaller_order += taken;
        taken = sz_sort_neon_compact4_(keys_lower_u8x16x2, order_lower_u8x16x2, equal_lower, equal_pgrams, equal_order);
        equal_pgrams += taken, equal_order += taken;
        taken = sz_sort_neon_compact4_(keys_upper_u8x16x2, order_upper_u8x16x2, equal_upper, equal_pgrams, equal_order);
        equal_pgrams += taken, equal_order += taken;
        taken = sz_sort_neon_compact4_(keys_lower_u8x16x2, order_lower_u8x16x2, greater_lower, greater_pgrams,
                                       greater_order);
        greater_pgrams += taken, greater_order += taken;
        taken = sz_sort_neon_compact4_(keys_upper_u8x16x2, order_upper_u8x16x2, greater_upper, greater_pgrams,
                                       greater_order);
        greater_pgrams += taken, greater_order += taken;
    }
    for (; i < end_in_sequence; ++i) {
        sz_pgram_t const value = initial_pgrams[i];
        sz_sorted_idx_t const index = initial_order[i];
        if (value < pivot_pgram) *smaller_pgrams++ = value, *smaller_order++ = index;
        else if (value > pivot_pgram) *greater_pgrams++ = value, *greater_order++ = index;
        else *equal_pgrams++ = value, *equal_order++ = index;
    }

    // Copy the three slack-separated regions back into one contiguous run.
    sz_pgram_t *const destination_pgrams = initial_pgrams + start_in_sequence;
    sz_sorted_idx_t *const destination_order = initial_order + start_in_sequence;
    sz_copy_neon((sz_ptr_t)(destination_pgrams), (sz_cptr_t)(partitioned_pgrams + start_in_sequence),
                 count_smaller * sizeof(sz_pgram_t));
    sz_copy_neon((sz_ptr_t)(destination_pgrams + count_smaller),
                 (sz_cptr_t)(partitioned_pgrams + start_in_sequence + equal_region), count_equal * sizeof(sz_pgram_t));
    sz_copy_neon((sz_ptr_t)(destination_pgrams + count_smaller + count_equal),
                 (sz_cptr_t)(partitioned_pgrams + start_in_sequence + greater_region),
                 count_greater * sizeof(sz_pgram_t));
    sz_copy_neon((sz_ptr_t)(destination_order), (sz_cptr_t)(partitioned_order + start_in_sequence),
                 count_smaller * sizeof(sz_sorted_idx_t));
    sz_copy_neon((sz_ptr_t)(destination_order + count_smaller),
                 (sz_cptr_t)(partitioned_order + start_in_sequence + equal_region),
                 count_equal * sizeof(sz_sorted_idx_t));
    sz_copy_neon((sz_ptr_t)(destination_order + count_smaller + count_equal),
                 (sz_cptr_t)(partitioned_order + start_in_sequence + greater_region),
                 count_greater * sizeof(sz_sorted_idx_t));

    *first_pivot_offset = start_in_sequence + count_smaller;
    *last_pivot_offset = start_in_sequence + count_smaller + count_equal - 1;
}

SZ_PUBLIC void sz_sequence_argsort_neon_quicksort_pgrams_(sz_pgram_t *initial_pgrams, sz_sorted_idx_t *initial_order,
                                                          sz_pgram_t *temporary_pgrams,
                                                          sz_sorted_idx_t *temporary_order,
                                                          sz_size_t const start_in_sequence,
                                                          sz_size_t const end_in_sequence, sz_size_t const top_count) {
    sz_size_t const count = end_in_sequence - start_in_sequence;
    if (count <= 32) { // Below the vectorization break-even, a stable insertion sort wins.
        sz_pgrams_sort_with_insertion(initial_pgrams + start_in_sequence, count, initial_order + start_in_sequence);
        return;
    }

    sz_size_t first_pivot_index, last_pivot_index;
    sz_sequence_argsort_neon_3way_partition_(initial_pgrams, initial_order, temporary_pgrams, temporary_order,
                                             start_in_sequence, end_in_sequence, &first_pivot_index, &last_pivot_index);

    if (start_in_sequence + 1 < first_pivot_index)
        sz_sequence_argsort_neon_quicksort_pgrams_(initial_pgrams, initial_order, temporary_pgrams, temporary_order,
                                                   start_in_sequence, first_pivot_index, top_count);
    if (last_pivot_index + 2 < end_in_sequence && (top_count == 0 || last_pivot_index + 1 < top_count))
        sz_sequence_argsort_neon_quicksort_pgrams_(initial_pgrams, initial_order, temporary_pgrams, temporary_order,
                                                   last_pivot_index + 1, end_in_sequence, top_count);
}

SZ_PUBLIC sz_status_t sz_pgrams_sort_neon(sz_pgram_t *pgrams, sz_size_t count, sz_memory_allocator_t *alloc,
                                          sz_sorted_idx_t *order) {
    for (sz_size_t pgram_index = 0; pgram_index != count; ++pgram_index) order[pgram_index] = pgram_index;

    sz_memory_allocator_t global_alloc;
    if (!alloc) {
        sz_memory_allocator_init_default(&global_alloc);
        alloc = &global_alloc;
    }

    // `+ 24` of slack absorbs the two inter-region gaps (8 each) plus the final compaction spill past `count`.
    sz_size_t memory_usage = sizeof(sz_pgram_t) * (count + 24) + sizeof(sz_sorted_idx_t) * (count + 24);
    sz_pgram_t *temporary_pgrams = (sz_pgram_t *)alloc->allocate(memory_usage, alloc);
    sz_sorted_idx_t *temporary_order = (sz_sorted_idx_t *)(temporary_pgrams + count + 24);
    if (!temporary_pgrams) return sz_bad_alloc_k;

    sz_sequence_argsort_neon_quicksort_pgrams_(pgrams, order, temporary_pgrams, temporary_order, 0, count, 0);

    alloc->free(temporary_pgrams, memory_usage, alloc);
    return sz_success_k;
}

SZ_PUBLIC void sz_sequence_argsort_neon_sort_byte_windows_(
    sz_sequence_t const *const sequence, sz_pgram_t *const global_pgrams, sz_sorted_idx_t *const global_order,
    sz_pgram_t *const temporary_pgrams, sz_sorted_idx_t *const temporary_order, sz_size_t const start_in_sequence,
    sz_size_t const end_in_sequence, sz_size_t const start_character, sz_size_t const top_count,
    sz_bool_t const reverse) {

    sz_sequence_argsort_serial_export_byte_window_(sequence, global_pgrams, global_order, start_in_sequence,
                                                   end_in_sequence, start_character, reverse);

    sz_sequence_argsort_neon_quicksort_pgrams_(global_pgrams, global_order, temporary_pgrams, temporary_order,
                                               start_in_sequence, end_in_sequence, top_count);

    sz_size_t const pgram_capacity = sizeof(sz_pgram_t) - 1;
    sz_size_t nested_start = start_in_sequence;
    sz_size_t nested_end = start_in_sequence;
    while (nested_end != end_in_sequence) {
        if (top_count != 0 && nested_start >= top_count) break;

        sz_pgram_t current_pgram = global_pgrams[nested_start];
        while (nested_end != end_in_sequence && current_pgram == global_pgrams[nested_end]) ++nested_end;

        sz_pgram_t const length_source = reverse ? ~current_pgram : current_pgram;
        sz_cptr_t const length_str = (sz_cptr_t)&length_source;
#if !SZ_IS_BIG_ENDIAN_
        sz_size_t current_pgram_length = (sz_size_t)(sz_u8_t)length_str[0];
#else
        sz_size_t current_pgram_length = (sz_size_t)(sz_u8_t)length_str[pgram_capacity];
#endif
        int has_multiple_strings = nested_end - nested_start > 1;
        int has_more_characters_in_each = current_pgram_length == pgram_capacity;
        if (has_multiple_strings && has_more_characters_in_each)
            sz_sequence_argsort_neon_sort_byte_windows_(sequence, global_pgrams, global_order, temporary_pgrams,
                                                        temporary_order, nested_start, nested_end,
                                                        start_character + pgram_capacity, top_count, reverse);
        else if (has_multiple_strings)
            sz_order_indices_ascending_(global_order + nested_start, nested_end - nested_start);
        nested_start = nested_end;
    }
}

SZ_PUBLIC sz_status_t sz_sequence_argsort_neon(sz_sequence_t const *sequence, sz_memory_allocator_t *alloc,
                                               sz_sorted_idx_t *order, sz_size_t top_count, sz_bool_t reverse) {
    sz_size_t count = sequence->count;
    for (sz_size_t sequence_index = 0; sequence_index != count; ++sequence_index)
        order[sequence_index] = sequence_index;

    if (count <= 32 && !reverse) {
        sz_sequence_argsort_with_insertion(sequence, order);
        return sz_success_k;
    }

    sz_memory_allocator_t global_alloc;
    if (!alloc) {
        sz_memory_allocator_init_default(&global_alloc);
        alloc = &global_alloc;
    }

    // `global_pgrams` (count) + two scratch buffers (count + 24 slack each: two inter-region gaps + spill).
    sz_size_t memory_usage = sizeof(sz_pgram_t) * (count + count + 24) + sizeof(sz_sorted_idx_t) * (count + 24);
    sz_pgram_t *global_pgrams = (sz_pgram_t *)alloc->allocate(memory_usage, alloc);
    sz_pgram_t *temporary_pgrams = global_pgrams + count;
    sz_sorted_idx_t *temporary_order = (sz_sorted_idx_t *)(temporary_pgrams + count + 24);
    if (!global_pgrams) return sz_bad_alloc_k;

    sz_sequence_argsort_neon_sort_byte_windows_(sequence, global_pgrams, order, temporary_pgrams, temporary_order, 0,
                                                count, 0, top_count, reverse);

    alloc->free(global_pgrams, memory_usage, alloc);
    return sz_success_k;
}

/**
 *  @brief Uncased twin of `sz_sequence_argsort_neon_sort_byte_windows_`: the folded code-point export
 *      stays scalar (and is shared with the serial backend), but the pgrams it produces are sorted with the
 *      NEON partition - which is where NEON beats the fully-serial uncased path.
 */
SZ_PUBLIC void sz_sequence_argsort_neon_sort_casefold_windows_(
    sz_sequence_t const *const sequence, sz_pgram_t *const global_pgrams, sz_sorted_idx_t *const global_order,
    sz_pgram_t *const temporary_pgrams, sz_sorted_idx_t *const temporary_order, sz_size_t const start_in_sequence,
    sz_size_t const end_in_sequence, sz_size_t const folded_skip_count, sz_size_t const top_count,
    sz_bool_t const reverse) {

    sz_sequence_argsort_serial_export_casefold_window_(sequence, global_pgrams, global_order, start_in_sequence,
                                                       end_in_sequence, folded_skip_count, reverse);
    sz_sequence_argsort_neon_quicksort_pgrams_(global_pgrams, global_order, temporary_pgrams, temporary_order,
                                               start_in_sequence, end_in_sequence, top_count);

    // A window's lowest 21-bit field is non-zero only when it was filled to capacity, so the equal group may
    // still carry more folded code-points and must recurse one window deeper (mirrors the serial folded sort).
    sz_size_t const fields_per_pgram = sz_argsort_casefold_fields_(sz_pgram_t);
    sz_pgram_t const lowest_field_mask = ((sz_pgram_t)1 << sz_argsort_casefold_field_bits_) - 1;
    sz_size_t nested_start = start_in_sequence;
    sz_size_t nested_end = start_in_sequence;
    while (nested_end != end_in_sequence) {
        if (top_count != 0 && nested_start >= top_count) break;

        sz_pgram_t current_pgram = global_pgrams[nested_start];
        while (nested_end != end_in_sequence && current_pgram == global_pgrams[nested_end]) ++nested_end;

        sz_pgram_t const decoded_pgram = reverse ? ~current_pgram : current_pgram;
        int has_multiple_strings = nested_end - nested_start > 1;
        int has_more_characters_in_each = (decoded_pgram & lowest_field_mask) != 0;
        if (has_multiple_strings && has_more_characters_in_each)
            sz_sequence_argsort_neon_sort_casefold_windows_(sequence, global_pgrams, global_order, temporary_pgrams,
                                                            temporary_order, nested_start, nested_end,
                                                            folded_skip_count + fields_per_pgram, top_count, reverse);
        else if (has_multiple_strings)
            sz_order_indices_ascending_(global_order + nested_start, nested_end - nested_start);
        nested_start = nested_end;
    }
}

SZ_PUBLIC sz_status_t sz_sequence_argsort_utf8_uncased_neon(     //
    sz_sequence_t const *sequence, sz_memory_allocator_t *alloc, //
    sz_sorted_idx_t *order, sz_size_t top_count, sz_bool_t reverse) {

    sz_size_t const count = sequence->count;
    for (sz_size_t sequence_index = 0; sequence_index != count; ++sequence_index)
        order[sequence_index] = sequence_index;
    if (count < 2) return sz_success_k;

    sz_memory_allocator_t global_alloc;
    if (!alloc) {
        sz_memory_allocator_init_default(&global_alloc);
        alloc = &global_alloc;
    }

    // Same layout as the byte arg-sort - `global_pgrams` (count) + two scratch buffers (count + 24 slack each:
    // two inter-region gaps + spill, since the NEON table compaction overruns). The folded export is stateless
    // (re-folds the prefix on demand), so unlike the earlier design there is no per-string cursor array.
    sz_size_t const memory_usage = sizeof(sz_pgram_t) * (count + count + 24) + sizeof(sz_sorted_idx_t) * (count + 24);
    sz_pgram_t *global_pgrams = (sz_pgram_t *)alloc->allocate(memory_usage, alloc);
    if (!global_pgrams) return sz_bad_alloc_k;
    sz_pgram_t *temporary_pgrams = global_pgrams + count;
    sz_sorted_idx_t *temporary_order = (sz_sorted_idx_t *)(temporary_pgrams + count + 24);

    sz_sequence_argsort_neon_sort_casefold_windows_(sequence, global_pgrams, order, temporary_pgrams, temporary_order,
                                                    0, count, 0, top_count, reverse);

    alloc->free(global_pgrams, memory_usage, alloc);
    return sz_success_k;
}

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#endif // SZ_USE_NEON

#ifdef __cplusplus
}
#endif

#endif // STRINGZILLA_SORT_NEON_H_
