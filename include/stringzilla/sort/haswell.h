/**
 *  @brief Haswell (x86 AVX2) backend for sorting string collections.
 *  @file include/stringzilla/sort/haswell.h
 *  @author Ash Vardanian
 *  @sa include/stringzilla/sort.h
 *
 *  Mirrors the NEON backend's out-of-place 3-way QuickSort partition: AVX2 has no AVX-512 `compressstore`,
 *  so the per-block compaction uses a small `static const` left-pack table fed to `vpermd`
 *  (`_mm256_permutevar8x32_epi32`). A 4-u64 block is treated as 8 u32, and row `[m]` of the table holds the
 *  8 dword indices that gather the selected u64 lanes (of 4) to the front. The 4-wide table is 16x32 = 512
 *  bytes - small enough to keep `const` and warm in cache - versus the 16 KB an 8-wide table would need.
 *  Compaction preserves lane order, so the partition is @b stable - matching the stable-by-default contract -
 *  and the recursion/stability/reverse/top-K/uncased machinery is reused verbatim from `sort/serial.h`.
 *
 *  AVX2 has only a @b signed 64-bit compare (`_mm256_cmpgt_epi64`), so unsigned pgram keys are biased by the
 *  sign bit (XOR `0x8000...`) before comparing, which maps unsigned order onto signed order.
 */
#ifndef STRINGZILLA_SORT_HASWELL_H_
#define STRINGZILLA_SORT_HASWELL_H_

#include "stringzilla/types.h"
#include "stringzilla/compare.h" // `sz_compare`
#include "stringzilla/memory.h"  // `sz_copy_haswell`

#include "stringzilla/sort/serial.h"

#ifdef __cplusplus
extern "C" {
#endif

#if SZ_USE_HASWELL
#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("avx2"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("avx2")
#endif

/**
 *  @brief Number of `sz_pgram_t` lanes in one AVX2 vector (4 on 64-bit) - the granularity at which the
 *      3-way partition loads, `vpermd`-permutes, and stores.
 */
#define sz_sort_haswell_vector_lanes_ ((sz_size_t)(sizeof(__m256i) / sizeof(sz_pgram_t)))

/**
 *  @brief Gap, in pgram lanes, that the out-of-place 3-way partition leaves between its scratch regions.
 *
 *  AVX2 has @b no masked or compress store, so the compaction (`sz_sort_haswell_compact4_`) left-packs with
 *  `vpermd` and then stores a @b full vector, advancing the output cursor only by the surviving-lane count.
 *  A region's final store can therefore overrun its true end by up to one whole vector, so consecutive
 *  regions are separated by exactly one vector width to absorb that overrun. The gap is purely a consequence
 *  of the missing masked store - Skylake, whose `vpcompressstoreu` writes exactly the survivors, needs none.
 */
#define sz_sort_haswell_region_gap_ sz_sort_haswell_vector_lanes_

/** @brief Per-`count` scratch over-allocation: two inter-region gaps + the trailing region's overrun. */
#define sz_sort_haswell_partition_slack_ (3 * sz_sort_haswell_region_gap_)

/** @brief Collapses a 64-bit-lane compare result (each lane 0 or ~0) into a 4-bit lane mask. */
SZ_INTERNAL sz_u32_t sz_sort_haswell_lane_mask4_(__m256i const compared_u64x4) {
    return (sz_u32_t)_mm256_movemask_pd(_mm256_castsi256_pd(compared_u64x4));
}

/** @brief Left-packs the @p mask4 -selected lanes of one 4-u64 block (keys @p keys_u64x4 and order
 *         @p order_u64x4) to the @p out_pgrams / @p out_order cursors, preserving order; returns how many
 *         lanes were written. The full vector is stored unconditionally (the cursor only advances by the
 *         surviving count, so the unwritten tail is overwritten by the next call or lands in the region
 *         slack): on a wide out-of-order core the branchless full-vector stores beat data-dependent
 *         "store only what survives" branches, whose misprediction cost dwarfs the few wasted stores. */
SZ_INTERNAL sz_size_t sz_sort_haswell_compact4_(         //
    __m256i const keys_u64x4, __m256i const order_u64x4, //
    sz_u32_t const mask4, sz_pgram_t *const out_pgrams, sz_sorted_idx_t *const out_order) {

    // Left-pack table for `vpermd`: row `[m]` holds the 8 dword indices that gather the `m`-selected u64 lanes
    // (of 4, each a pair of dwords) to the front. 16 x 32 = 512 bytes, a single cache-resident load; an 8-wide
    // variant would need 16 KB, which is why the partition stays 4-wide. `taken` comes from POPCNT, not a table.
    static sz_u32_t const compact_lut[16][8] = {
        {0, 0, 0, 0, 0, 0, 0, 0}, // 0b0000: none
        {0, 1, 0, 0, 0, 0, 0, 0}, // 0b0001: lane 0
        {2, 3, 0, 0, 0, 0, 0, 0}, // 0b0010: lane 1
        {0, 1, 2, 3, 0, 0, 0, 0}, // 0b0011: lanes 0,1
        {4, 5, 0, 0, 0, 0, 0, 0}, // 0b0100: lane 2
        {0, 1, 4, 5, 0, 0, 0, 0}, // 0b0101: lanes 0,2
        {2, 3, 4, 5, 0, 0, 0, 0}, // 0b0110: lanes 1,2
        {0, 1, 2, 3, 4, 5, 0, 0}, // 0b0111: lanes 0,1,2
        {6, 7, 0, 0, 0, 0, 0, 0}, // 0b1000: lane 3
        {0, 1, 6, 7, 0, 0, 0, 0}, // 0b1001: lanes 0,3
        {2, 3, 6, 7, 0, 0, 0, 0}, // 0b1010: lanes 1,3
        {0, 1, 2, 3, 6, 7, 0, 0}, // 0b1011: lanes 0,1,3
        {4, 5, 6, 7, 0, 0, 0, 0}, // 0b1100: lanes 2,3
        {0, 1, 4, 5, 6, 7, 0, 0}, // 0b1101: lanes 0,2,3
        {2, 3, 4, 5, 6, 7, 0, 0}, // 0b1110: lanes 1,2,3
        {0, 1, 2, 3, 4, 5, 6, 7}, // 0b1111: all
    };

    sz_size_t const taken = (sz_size_t)_mm_popcnt_u32(mask4);
    __m256i const permutation = _mm256_loadu_si256((__m256i const *)compact_lut[mask4]);
    _mm256_storeu_si256((__m256i *)out_pgrams, _mm256_permutevar8x32_epi32(keys_u64x4, permutation));
    _mm256_storeu_si256((__m256i *)out_order, _mm256_permutevar8x32_epi32(order_u64x4, permutation));
    return taken;
}

/**
 *  @brief 3-way partition around the Sedgewick pivot using AVX2 table-compaction.
 *  @note Out-of-place into @p partitioned_* with three regions (smaller, equal, greater) laid out by the
 *        count pass and separated by one-vector slack gaps that absorb each region's trailing full-vector
 *        store overrun; the three regions are then copied back contiguously. A single block-major pass
 *        left-packs all three comparison kinds together, so each block is loaded once and the equal mask
 *        is derived for free.
 */
SZ_INTERNAL void sz_sequence_argsort_haswell_3way_partition_(                       //
    sz_pgram_t *const initial_pgrams, sz_sorted_idx_t *const initial_order,         //
    sz_pgram_t *const partitioned_pgrams, sz_sorted_idx_t *const partitioned_order, //
    sz_size_t const start_in_sequence, sz_size_t const end_in_sequence,             //
    sz_size_t *const first_pivot_offset, sz_size_t *const last_pivot_offset) {

    sz_size_t const count = end_in_sequence - start_in_sequence;
    sz_pgram_t const pivot_pgram = *sz_sequence_partitioning_pivot_(initial_pgrams + start_in_sequence, count);

    // AVX2 has only a signed 64-bit compare, so bias the unsigned keys and pivot by the sign bit.
    __m256i const sign_u64x4 = _mm256_set1_epi64x((long long)0x8000000000000000ull);
    __m256i const pivot_u64x4 = _mm256_set1_epi64x((long long)pivot_pgram);
    __m256i const pivot_biased_u64x4 = _mm256_xor_si256(pivot_u64x4, sign_u64x4);

    // Count pass: tally smaller and greater lanes by population-counting the compare masks - mirrors the
    // Skylake backend, with `movemask_pd` standing in for the AVX-512 mask register.
    sz_size_t count_smaller = 0, count_greater = 0;
    sz_size_t i = start_in_sequence;
    for (; i + 8 <= end_in_sequence; i += 8) {
        __m256i keys_lower_u64x4 = _mm256_loadu_si256((__m256i const *)(initial_pgrams + i));
        __m256i keys_upper_u64x4 = _mm256_loadu_si256((__m256i const *)(initial_pgrams + i + 4));
        __m256i lower_biased_u64x4 = _mm256_xor_si256(keys_lower_u64x4, sign_u64x4);
        __m256i upper_biased_u64x4 = _mm256_xor_si256(keys_upper_u64x4, sign_u64x4);
        count_smaller += (sz_size_t)_mm_popcnt_u32(
            sz_sort_haswell_lane_mask4_(_mm256_cmpgt_epi64(pivot_biased_u64x4, lower_biased_u64x4)));
        count_smaller += (sz_size_t)_mm_popcnt_u32(
            sz_sort_haswell_lane_mask4_(_mm256_cmpgt_epi64(pivot_biased_u64x4, upper_biased_u64x4)));
        count_greater += (sz_size_t)_mm_popcnt_u32(
            sz_sort_haswell_lane_mask4_(_mm256_cmpgt_epi64(lower_biased_u64x4, pivot_biased_u64x4)));
        count_greater += (sz_size_t)_mm_popcnt_u32(
            sz_sort_haswell_lane_mask4_(_mm256_cmpgt_epi64(upper_biased_u64x4, pivot_biased_u64x4)));
    }
    for (; i < end_in_sequence; ++i)
        count_smaller += initial_pgrams[i]<pivot_pgram, count_greater += initial_pgrams[i]> pivot_pgram;
    sz_size_t const count_equal = count - count_smaller - count_greater;

    // Lay the three regions out with `sz_sort_haswell_region_gap_` slack between them; each region's final
    // compaction may spill up to a full vector past its true end, and the gap (plus the buffer's trailing
    // slack) absorbs it.
    sz_size_t const equal_region = count_smaller + sz_sort_haswell_region_gap_;
    sz_size_t const greater_region = equal_region + count_equal + sz_sort_haswell_region_gap_;
    sz_pgram_t *smaller_pgrams = partitioned_pgrams + start_in_sequence;
    sz_sorted_idx_t *smaller_order = partitioned_order + start_in_sequence;
    sz_pgram_t *equal_pgrams = smaller_pgrams + equal_region;
    sz_sorted_idx_t *equal_order = smaller_order + equal_region;
    sz_pgram_t *greater_pgrams = smaller_pgrams + greater_region;
    sz_sorted_idx_t *greater_order = smaller_order + greater_region;

    // Block-major compaction: each 8-lane block is loaded once and dispatched to all three regions.
    i = start_in_sequence;
    for (; i + 8 <= end_in_sequence; i += 8) {
        __m256i keys_lower_u64x4 = _mm256_loadu_si256((__m256i const *)(initial_pgrams + i));
        __m256i keys_upper_u64x4 = _mm256_loadu_si256((__m256i const *)(initial_pgrams + i + 4));
        __m256i order_lower_u64x4 = _mm256_loadu_si256((__m256i const *)(initial_order + i));
        __m256i order_upper_u64x4 = _mm256_loadu_si256((__m256i const *)(initial_order + i + 4));
        __m256i lower_biased_u64x4 = _mm256_xor_si256(keys_lower_u64x4, sign_u64x4);
        __m256i upper_biased_u64x4 = _mm256_xor_si256(keys_upper_u64x4, sign_u64x4);

        // Four compares per block; the equal mask is the complement of (smaller | greater), so no third compare.
        sz_u32_t smaller_lower = sz_sort_haswell_lane_mask4_(
            _mm256_cmpgt_epi64(pivot_biased_u64x4, lower_biased_u64x4));
        sz_u32_t smaller_upper = sz_sort_haswell_lane_mask4_(
            _mm256_cmpgt_epi64(pivot_biased_u64x4, upper_biased_u64x4));
        sz_u32_t greater_lower = sz_sort_haswell_lane_mask4_(
            _mm256_cmpgt_epi64(lower_biased_u64x4, pivot_biased_u64x4));
        sz_u32_t greater_upper = sz_sort_haswell_lane_mask4_(
            _mm256_cmpgt_epi64(upper_biased_u64x4, pivot_biased_u64x4));
        sz_u32_t equal_lower = (~(smaller_lower | greater_lower)) & 0xF;
        sz_u32_t equal_upper = (~(smaller_upper | greater_upper)) & 0xF;

        sz_size_t taken;
        taken = sz_sort_haswell_compact4_(keys_lower_u64x4, order_lower_u64x4, smaller_lower, smaller_pgrams,
                                          smaller_order);
        smaller_pgrams += taken, smaller_order += taken;
        taken = sz_sort_haswell_compact4_(keys_upper_u64x4, order_upper_u64x4, smaller_upper, smaller_pgrams,
                                          smaller_order);
        smaller_pgrams += taken, smaller_order += taken;
        taken = sz_sort_haswell_compact4_(keys_lower_u64x4, order_lower_u64x4, equal_lower, equal_pgrams, equal_order);
        equal_pgrams += taken, equal_order += taken;
        taken = sz_sort_haswell_compact4_(keys_upper_u64x4, order_upper_u64x4, equal_upper, equal_pgrams, equal_order);
        equal_pgrams += taken, equal_order += taken;
        taken = sz_sort_haswell_compact4_(keys_lower_u64x4, order_lower_u64x4, greater_lower, greater_pgrams,
                                          greater_order);
        greater_pgrams += taken, greater_order += taken;
        taken = sz_sort_haswell_compact4_(keys_upper_u64x4, order_upper_u64x4, greater_upper, greater_pgrams,
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
    sz_copy_haswell((sz_ptr_t)(destination_pgrams), (sz_cptr_t)(partitioned_pgrams + start_in_sequence),
                    count_smaller * sizeof(sz_pgram_t));
    sz_copy_haswell((sz_ptr_t)(destination_pgrams + count_smaller),
                    (sz_cptr_t)(partitioned_pgrams + start_in_sequence + equal_region),
                    count_equal * sizeof(sz_pgram_t));
    sz_copy_haswell((sz_ptr_t)(destination_pgrams + count_smaller + count_equal),
                    (sz_cptr_t)(partitioned_pgrams + start_in_sequence + greater_region),
                    count_greater * sizeof(sz_pgram_t));
    sz_copy_haswell((sz_ptr_t)(destination_order), (sz_cptr_t)(partitioned_order + start_in_sequence),
                    count_smaller * sizeof(sz_sorted_idx_t));
    sz_copy_haswell((sz_ptr_t)(destination_order + count_smaller),
                    (sz_cptr_t)(partitioned_order + start_in_sequence + equal_region),
                    count_equal * sizeof(sz_sorted_idx_t));
    sz_copy_haswell((sz_ptr_t)(destination_order + count_smaller + count_equal),
                    (sz_cptr_t)(partitioned_order + start_in_sequence + greater_region),
                    count_greater * sizeof(sz_sorted_idx_t));

    *first_pivot_offset = start_in_sequence + count_smaller;
    *last_pivot_offset = start_in_sequence + count_smaller + count_equal - 1;
}

SZ_PUBLIC void sz_sequence_argsort_haswell_quicksort_pgrams_(sz_pgram_t *initial_pgrams, sz_sorted_idx_t *initial_order,
                                                             sz_pgram_t *temporary_pgrams,
                                                             sz_sorted_idx_t *temporary_order,
                                                             sz_size_t const start_in_sequence,
                                                             sz_size_t const end_in_sequence,
                                                             sz_size_t const top_count) {
    sz_size_t const count = end_in_sequence - start_in_sequence;
    if (count <= 32) { // Below the vectorization break-even, a stable insertion sort wins.
        sz_pgrams_sort_with_insertion(initial_pgrams + start_in_sequence, count, initial_order + start_in_sequence);
        return;
    }

    sz_size_t first_pivot_index, last_pivot_index;
    sz_sequence_argsort_haswell_3way_partition_(initial_pgrams, initial_order, temporary_pgrams, temporary_order,
                                                start_in_sequence, end_in_sequence, &first_pivot_index,
                                                &last_pivot_index);

    if (start_in_sequence + 1 < first_pivot_index)
        sz_sequence_argsort_haswell_quicksort_pgrams_(initial_pgrams, initial_order, temporary_pgrams, temporary_order,
                                                      start_in_sequence, first_pivot_index, top_count);
    if (last_pivot_index + 2 < end_in_sequence && (top_count == 0 || last_pivot_index + 1 < top_count))
        sz_sequence_argsort_haswell_quicksort_pgrams_(initial_pgrams, initial_order, temporary_pgrams, temporary_order,
                                                      last_pivot_index + 1, end_in_sequence, top_count);
}

SZ_PUBLIC sz_status_t sz_pgrams_sort_haswell(sz_pgram_t *pgrams, sz_size_t count, sz_memory_allocator_t *alloc,
                                             sz_sorted_idx_t *order) {
    for (sz_size_t pgram_index = 0; pgram_index != count; ++pgram_index) order[pgram_index] = pgram_index;

    sz_memory_allocator_t global_alloc;
    if (!alloc) {
        sz_memory_allocator_init_default(&global_alloc);
        alloc = &global_alloc;
    }

    // Two scratch buffers, each over-allocated by `sz_sort_haswell_partition_slack_` (region gaps + spill).
    sz_size_t const slack = sz_sort_haswell_partition_slack_;
    sz_size_t memory_usage = sizeof(sz_pgram_t) * (count + slack) + sizeof(sz_sorted_idx_t) * (count + slack);
    sz_pgram_t *temporary_pgrams = (sz_pgram_t *)alloc->allocate(memory_usage, alloc);
    sz_sorted_idx_t *temporary_order = (sz_sorted_idx_t *)(temporary_pgrams + count + slack);
    if (!temporary_pgrams) return sz_bad_alloc_k;

    sz_sequence_argsort_haswell_quicksort_pgrams_(pgrams, order, temporary_pgrams, temporary_order, 0, count, 0);

    alloc->free(temporary_pgrams, memory_usage, alloc);
    return sz_success_k;
}

SZ_PUBLIC void sz_sequence_argsort_haswell_sort_byte_windows_(
    sz_sequence_t const *const sequence, sz_pgram_t *const global_pgrams, sz_sorted_idx_t *const global_order,
    sz_pgram_t *const temporary_pgrams, sz_sorted_idx_t *const temporary_order, sz_size_t const start_in_sequence,
    sz_size_t const end_in_sequence, sz_size_t const start_character, sz_size_t const top_count,
    sz_bool_t const reverse) {

    sz_sequence_argsort_serial_export_byte_window_(sequence, global_pgrams, global_order, start_in_sequence,
                                                   end_in_sequence, start_character, reverse);

    sz_sequence_argsort_haswell_quicksort_pgrams_(global_pgrams, global_order, temporary_pgrams, temporary_order,
                                                  start_in_sequence, end_in_sequence, top_count);

    sz_size_t const pgram_capacity = sizeof(sz_pgram_t) - 1;
    sz_size_t nested_start = start_in_sequence;
    sz_size_t nested_end = start_in_sequence;
    while (nested_end != end_in_sequence) {
        if (top_count != 0 && nested_start >= top_count) break;

        sz_pgram_t current_pgram = global_pgrams[nested_start];
        while (nested_end != end_in_sequence && current_pgram == global_pgrams[nested_end]) ++nested_end;

        // x86 is always little-endian, so the packed length byte is the least-significant byte of the
        // (byte-reversed) pgram - a plain cast reads it with no endianness branch.
        sz_pgram_t const length_source = reverse ? ~current_pgram : current_pgram;
        sz_size_t current_pgram_length = (sz_size_t)(sz_u8_t)length_source;
        int has_multiple_strings = nested_end - nested_start > 1;
        int has_more_characters_in_each = current_pgram_length == pgram_capacity;
        if (has_multiple_strings && has_more_characters_in_each)
            sz_sequence_argsort_haswell_sort_byte_windows_(sequence, global_pgrams, global_order, temporary_pgrams,
                                                           temporary_order, nested_start, nested_end,
                                                           start_character + pgram_capacity, top_count, reverse);
        else if (has_multiple_strings)
            sz_order_indices_ascending_(global_order + nested_start, nested_end - nested_start);
        nested_start = nested_end;
    }
}

SZ_PUBLIC sz_status_t sz_sequence_argsort_haswell(sz_sequence_t const *sequence, sz_memory_allocator_t *alloc,
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

    // One allocation backs three arrays laid out back-to-back:
    //   - `global_pgrams`    : the working pgram keys                           → `count`
    //   - `temporary_pgrams` : the 3-way partition's out-of-place pgram scratch → `count + slack`
    //   - `temporary_order`  : the matching order scratch                       → `count + slack`
    sz_size_t const slack = sz_sort_haswell_partition_slack_;
    sz_size_t const memory_usage = sizeof(sz_pgram_t) * (count + (count + slack)) // global + pgram scratch
                                   + sizeof(sz_sorted_idx_t) * (count + slack);   // order scratch
    sz_pgram_t *global_pgrams = (sz_pgram_t *)alloc->allocate(memory_usage, alloc);
    sz_pgram_t *temporary_pgrams = global_pgrams + count;
    sz_sorted_idx_t *temporary_order = (sz_sorted_idx_t *)(temporary_pgrams + count + slack);
    if (!global_pgrams) return sz_bad_alloc_k;

    sz_sequence_argsort_haswell_sort_byte_windows_(sequence, global_pgrams, order, temporary_pgrams, temporary_order, 0,
                                                   count, 0, top_count, reverse);

    alloc->free(global_pgrams, memory_usage, alloc);
    return sz_success_k;
}

/**
 *  @brief Uncased twin of `sz_sequence_argsort_haswell_sort_byte_windows_`: the folded code-point export
 *      stays scalar (and is shared with the serial backend), but the pgrams it produces are sorted with the
 *      AVX2 partition - which is where Haswell beats the fully-serial uncased path.
 */
SZ_PUBLIC void sz_sequence_argsort_haswell_sort_casefold_windows_(
    sz_sequence_t const *const sequence, sz_pgram_t *const global_pgrams, sz_sorted_idx_t *const global_order,
    sz_pgram_t *const temporary_pgrams, sz_sorted_idx_t *const temporary_order, sz_size_t const start_in_sequence,
    sz_size_t const end_in_sequence, sz_size_t const folded_skip_count, sz_size_t const top_count,
    sz_bool_t const reverse) {

    sz_sequence_argsort_serial_export_casefold_window_(sequence, global_pgrams, global_order, start_in_sequence,
                                                       end_in_sequence, folded_skip_count, reverse);
    sz_sequence_argsort_haswell_quicksort_pgrams_(global_pgrams, global_order, temporary_pgrams, temporary_order,
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
            sz_sequence_argsort_haswell_sort_casefold_windows_(
                sequence, global_pgrams, global_order, temporary_pgrams, temporary_order, nested_start, nested_end,
                folded_skip_count + fields_per_pgram, top_count, reverse);
        else if (has_multiple_strings)
            sz_order_indices_ascending_(global_order + nested_start, nested_end - nested_start);
        nested_start = nested_end;
    }
}

SZ_PUBLIC sz_status_t sz_sequence_argsort_utf8_uncased_haswell( //
    sz_sequence_t const *sequence, sz_memory_allocator_t *alloc,         //
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

    // Same layout as the byte arg-sort - working pgrams (count) + the partition's two scratch regions
    // (count + slack each). The folded export is stateless (re-folds the prefix on demand), so unlike the
    // earlier design there is no per-string cursor array.
    sz_size_t const slack = sz_sort_haswell_partition_slack_;
    sz_size_t const memory_usage = sizeof(sz_pgram_t) * (count + (count + slack)) // global + pgram scratch
                                   + sizeof(sz_sorted_idx_t) * (count + slack);   // order scratch
    sz_pgram_t *global_pgrams = (sz_pgram_t *)alloc->allocate(memory_usage, alloc);
    if (!global_pgrams) return sz_bad_alloc_k;
    sz_pgram_t *temporary_pgrams = global_pgrams + count;
    sz_sorted_idx_t *temporary_order = (sz_sorted_idx_t *)(temporary_pgrams + count + slack);

    sz_sequence_argsort_haswell_sort_casefold_windows_(sequence, global_pgrams, order, temporary_pgrams,
                                                       temporary_order, 0, count, 0, top_count, reverse);

    alloc->free(global_pgrams, memory_usage, alloc);
    return sz_success_k;
}

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#endif // SZ_USE_HASWELL

#ifdef __cplusplus
}
#endif

#endif // STRINGZILLA_SORT_HASWELL_H_
