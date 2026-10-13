/**
 *  @file include/stringzilla/sort/haswell.h
 *  @author Ash Vardanian
 *  @date June 14, 2026
 *  @brief Haswell (x86 AVX2) backend for sorting string collections.
 *
 *  Mirrors the NEON backend's out-of-place 3-way QuickSort partition: AVX2 has no AVX-512
 *  @c compressstore, so the per-block compaction uses a small `static const` left-pack table fed to
 *  @c vpermd, the @c _mm256_permutevar8x32_epi32 intrinsic. A 4-u64 block is treated as 8 u32, and
 *  row `[m]` of the table holds the 8 dword indices that gather the selected u64 lanes, of 4, to
 *  the front. The 4-wide table is 16 × 32 = 512 bytes, small enough to keep @c const and warm in
 *  cache, versus the 16 KB an 8-wide table would need. Compaction preserves lane order, so the
 *  partition is @b stable, matching the stable-by-default contract, and the recursion, stability,
 *  reverse, top-K and uncased machinery is reused verbatim from `sort/serial.h`.
 *
 *  AVX2 has only a @b signed 64-bit compare, @c _mm256_cmpgt_epi64, so unsigned pgram keys are
 *  biased by the sign bit, an XOR with `0x8000...`, before comparing, which maps unsigned order
 *  onto signed order.
 *
 *  @sa include/stringzilla/sort.h
 */
#ifndef STRINGZILLA_SORT_HASWELL_H_
#define STRINGZILLA_SORT_HASWELL_H_

#include "stringzilla/types.h"
#include "stringzilla/sort/serial.h"

#ifdef __cplusplus
extern "C" {
#endif

#if STRINGZILLA_ARCH_X8664_
#if STRINGZILLA_ARCH_X8664_HASWELL_
#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("avx2,popcnt"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("avx2", "popcnt")
#endif

/** Number of @c sz_pgram_t lanes in one AVX2 vector (4 on 64-bit) - the granularity at which the
 *  3-way partition loads, @c vpermd-permutes, and stores. */
#define sz_sort_vector_lanes_haswell_ ((sz_size_t)(sizeof(__m256i) / sizeof(sz_pgram_t)))

/**
 *  @brief Gap, in pgram lanes, that the out-of-place 3-way partition leaves between
 *      its scratch regions.
 *
 *  AVX2 has @b no masked or compress store, so the compaction in @c sz_sort_compact4_haswell_
 *  left-packs with @c vpermd and then stores a @b full vector, advancing the output cursor only by
 *  the surviving-lane count. A region's final store can therefore overrun its true end by up to one
 *  whole vector, so consecutive regions are separated by exactly one vector width to absorb that
 *  overrun. The gap is purely a consequence of the missing masked store - Skylake, whose
 *  @c vpcompressstoreu writes exactly the survivors, needs none.
 */
#define sz_sort_region_gap_haswell_ sz_sort_vector_lanes_haswell_

/** Scratch over-allocation per @c count: two inter-region gaps and the
 *  trailing region's overrun. */
#define sz_sort_partition_slack_haswell_ (3 * sz_sort_region_gap_haswell_)

/** Collapses a 64-bit-lane compare result (each lane 0 or ~0) into a 4-bit lane mask. */
STRINGZILLA_INLINE sz_u32_t sz_sort_lane_mask4_haswell_(__m256i const compared_u64x4) {
    return (sz_u32_t)_mm256_movemask_pd(_mm256_castsi256_pd(compared_u64x4));
}

/** Left-packs the @p mask4 -selected lanes of one 4-u64 block (keys @p keys_u64x4 and order
 *  @p order_u64x4) to the @p out_pgrams / @p out_order cursors, preserving order; returns how many
 *  lanes were written. The full vector is stored unconditionally (the cursor only advances by the
 *  surviving count, so the unwritten tail is overwritten by the next call or lands in the region
 *  slack): on a wide out-of-order core the branchless full-vector stores beat data-dependent "store
 *  only what survives" branches, whose misprediction cost dwarfs the few wasted stores. */
STRINGZILLA_INLINE sz_size_t sz_sort_compact4_haswell_(  //
    __m256i const keys_u64x4, __m256i const order_u64x4, //
    sz_u32_t const mask4, sz_pgram_t *const out_pgrams, sz_size_t *const out_order) {

    // Stays 4-wide to keep the `vpermd` left-pack table at 512 bytes; `taken` comes from POPCNT.
    sz_size_t const taken = (sz_size_t)_mm_popcnt_u32(mask4);
    __m256i const permutation_u32x8 = _mm256_loadu_si256((__m256i const *)sz_compact4_dword_indices_[mask4]);
    _mm256_storeu_si256((__m256i *)out_pgrams, _mm256_permutevar8x32_epi32(keys_u64x4, permutation_u32x8));
    _mm256_storeu_si256((__m256i *)out_order, _mm256_permutevar8x32_epi32(order_u64x4, permutation_u32x8));
    return taken;
}

/** Per-region pair of output cursors (keys and matching order) the 3-way
 *  partition left-packs into. */
typedef struct sz_sort_region_cursor_haswell_t {
    sz_pgram_t *pgrams;
    sz_size_t *order;
} sz_sort_region_cursor_haswell_t;

/** The smaller/equal/greater lane masks of one 8-lane block, split into its lower
 *  and upper 4-lanes. */
typedef struct sz_sort_block_masks_haswell_t {
    sz_u32_t smaller_lower, smaller_upper;
    sz_u32_t equal_lower, equal_upper;
    sz_u32_t greater_lower, greater_upper;
} sz_sort_block_masks_haswell_t;

/** Classifies one 8-lane block (its two key vectors @p keys_lower_u64x4 / @p keys_upper_u64x4)
 *  against the sign-biased @p pivot_biased_u64x4, returning the smaller/equal/greater lane
 *  masks for both 4-lane halves; the equal mask is the complement of (smaller | greater), so
 *  only four compares run. */
STRINGZILLA_INLINE sz_sort_block_masks_haswell_t sz_sort_classify_block_haswell_( //
    __m256i const keys_lower_u64x4, __m256i const keys_upper_u64x4,               //
    __m256i const pivot_biased_u64x4, __m256i const sign_u64x4) {

    __m256i const lower_biased_u64x4 = _mm256_xor_si256(keys_lower_u64x4, sign_u64x4);
    __m256i const upper_biased_u64x4 = _mm256_xor_si256(keys_upper_u64x4, sign_u64x4);
    sz_sort_block_masks_haswell_t masks;
    masks.smaller_lower = sz_sort_lane_mask4_haswell_(_mm256_cmpgt_epi64(pivot_biased_u64x4, lower_biased_u64x4));
    masks.smaller_upper = sz_sort_lane_mask4_haswell_(_mm256_cmpgt_epi64(pivot_biased_u64x4, upper_biased_u64x4));
    masks.greater_lower = sz_sort_lane_mask4_haswell_(_mm256_cmpgt_epi64(lower_biased_u64x4, pivot_biased_u64x4));
    masks.greater_upper = sz_sort_lane_mask4_haswell_(_mm256_cmpgt_epi64(upper_biased_u64x4, pivot_biased_u64x4));
    masks.equal_lower = (~(masks.smaller_lower | masks.greater_lower)) & 0xF;
    masks.equal_upper = (~(masks.smaller_upper | masks.greater_upper)) & 0xF;
    return masks;
}

/** Left-packs both 4-lane halves of one 8-lane block into @p cursor under @p mask_lower /
 *  @p mask_upper, advancing the cursor by the surviving-lane counts (the lower half first,
 *  preserving lane order). */
STRINGZILLA_INLINE void sz_sort_compact_block_into_haswell_(         //
    sz_sort_region_cursor_haswell_t *const cursor,                   //
    __m256i const keys_lower_u64x4, __m256i const order_lower_u64x4, //
    __m256i const keys_upper_u64x4, __m256i const order_upper_u64x4, //
    sz_u32_t const mask_lower, sz_u32_t const mask_upper) {

    sz_size_t taken;
    taken = sz_sort_compact4_haswell_(keys_lower_u64x4, order_lower_u64x4, mask_lower, cursor->pgrams, cursor->order);
    cursor->pgrams += taken, cursor->order += taken;
    taken = sz_sort_compact4_haswell_(keys_upper_u64x4, order_upper_u64x4, mask_upper, cursor->pgrams, cursor->order);
    cursor->pgrams += taken, cursor->order += taken;
}

/**
 *  @brief 3-way partition around the Sedgewick pivot using AVX2 table-compaction.
 *
 *  Out-of-place into @p partitioned_* with three regions, smaller, equal and greater, laid out by
 *  the count pass and separated by one-vector slack gaps that absorb each region's trailing
 *  full-vector store overrun; the three regions are then copied back contiguously. A single
 *  block-major pass left-packs all three comparison kinds together, so each block is loaded once
 *  and the equal mask is derived for free.
 */
STRINGZILLA_INLINE void sz_sort_3way_partition_haswell_(                      //
    sz_pgram_t *const initial_pgrams, sz_size_t *const initial_order,         //
    sz_pgram_t *const partitioned_pgrams, sz_size_t *const partitioned_order, //
    sz_size_t const start_in_strings, sz_size_t const end_in_strings,         //
    sz_size_t *const first_pivot_offset, sz_size_t *const last_pivot_offset) {

    sz_size_t const count = end_in_strings - start_in_strings;
    sz_pgram_t const pivot_pgram = *sz_strings_partitioning_pivot_(initial_pgrams + start_in_strings, count);

    // AVX2 has only a signed 64-bit compare, so bias the unsigned keys and pivot by the sign bit.
    __m256i const sign_u64x4 = _mm256_set1_epi64x((long long)0x8000000000000000ull);
    __m256i const pivot_u64x4 = _mm256_set1_epi64x((long long)pivot_pgram);
    __m256i const pivot_biased_u64x4 = _mm256_xor_si256(pivot_u64x4, sign_u64x4);

    // Count pass: tally smaller and greater lanes by population-counting the compare masks - mirrors the
    // Skylake backend, with `movemask_pd` standing in for the AVX-512 mask register.
    sz_size_t count_smaller = 0, count_greater = 0;
    sz_size_t i = start_in_strings;
    for (; i + 8 <= end_in_strings; i += 8) {
        __m256i keys_lower_u64x4 = _mm256_loadu_si256((__m256i const *)(initial_pgrams + i));
        __m256i keys_upper_u64x4 = _mm256_loadu_si256((__m256i const *)(initial_pgrams + i + 4));
        sz_sort_block_masks_haswell_t const masks = sz_sort_classify_block_haswell_(keys_lower_u64x4, keys_upper_u64x4,
                                                                                    pivot_biased_u64x4, sign_u64x4);
        count_smaller += (sz_size_t)_mm_popcnt_u32(masks.smaller_lower);
        count_smaller += (sz_size_t)_mm_popcnt_u32(masks.smaller_upper);
        count_greater += (sz_size_t)_mm_popcnt_u32(masks.greater_lower);
        count_greater += (sz_size_t)_mm_popcnt_u32(masks.greater_upper);
    }
    for (; i < end_in_strings; ++i)
        count_smaller += initial_pgrams[i]<pivot_pgram, count_greater += initial_pgrams[i]> pivot_pgram;
    sz_size_t const count_equal = count - count_smaller - count_greater;

    // Lay the three regions out with `sz_sort_region_gap_haswell_` slack between them; each
    // region's final compaction may spill up to a full vector past its true end, and the gap
    // (plus the buffer's trailing slack) absorbs it.
    sz_size_t const equal_region = count_smaller + sz_sort_region_gap_haswell_;
    sz_size_t const greater_region = equal_region + count_equal + sz_sort_region_gap_haswell_;
    sz_sort_region_cursor_haswell_t smaller_cursor = {partitioned_pgrams + start_in_strings,
                                                      partitioned_order + start_in_strings};
    sz_sort_region_cursor_haswell_t equal_cursor = {smaller_cursor.pgrams + equal_region,
                                                    smaller_cursor.order + equal_region};
    sz_sort_region_cursor_haswell_t greater_cursor = {smaller_cursor.pgrams + greater_region,
                                                      smaller_cursor.order + greater_region};

    // Block-major compaction: each 8-lane block is loaded once and dispatched to all three regions.
    i = start_in_strings;
    for (; i + 8 <= end_in_strings; i += 8) {
        __m256i keys_lower_u64x4 = _mm256_loadu_si256((__m256i const *)(initial_pgrams + i));
        __m256i keys_upper_u64x4 = _mm256_loadu_si256((__m256i const *)(initial_pgrams + i + 4));
        __m256i order_lower_u64x4 = _mm256_loadu_si256((__m256i const *)(initial_order + i));
        __m256i order_upper_u64x4 = _mm256_loadu_si256((__m256i const *)(initial_order + i + 4));
        sz_sort_block_masks_haswell_t const masks = sz_sort_classify_block_haswell_(keys_lower_u64x4, keys_upper_u64x4,
                                                                                    pivot_biased_u64x4, sign_u64x4);

        sz_sort_compact_block_into_haswell_(&smaller_cursor, keys_lower_u64x4, order_lower_u64x4, keys_upper_u64x4,
                                            order_upper_u64x4, masks.smaller_lower, masks.smaller_upper);
        sz_sort_compact_block_into_haswell_(&equal_cursor, keys_lower_u64x4, order_lower_u64x4, keys_upper_u64x4,
                                            order_upper_u64x4, masks.equal_lower, masks.equal_upper);
        sz_sort_compact_block_into_haswell_(&greater_cursor, keys_lower_u64x4, order_lower_u64x4, keys_upper_u64x4,
                                            order_upper_u64x4, masks.greater_lower, masks.greater_upper);
    }
    for (; i < end_in_strings; ++i) {
        sz_pgram_t const value = initial_pgrams[i];
        sz_size_t const index = initial_order[i];
        if (value < pivot_pgram) *smaller_cursor.pgrams++ = value, *smaller_cursor.order++ = index;
        else if (value > pivot_pgram) *greater_cursor.pgrams++ = value, *greater_cursor.order++ = index;
        else *equal_cursor.pgrams++ = value, *equal_cursor.order++ = index;
    }

    // Copy the three slack-separated regions back into one contiguous run.
    sz_pgram_t *const destination_pgrams = initial_pgrams + start_in_strings;
    sz_size_t *const destination_order = initial_order + start_in_strings;
    memcpy(destination_pgrams, partitioned_pgrams + start_in_strings, count_smaller * sizeof(sz_pgram_t));
    memcpy(destination_pgrams + count_smaller, partitioned_pgrams + start_in_strings + equal_region,
           count_equal * sizeof(sz_pgram_t));
    memcpy(destination_pgrams + count_smaller + count_equal, partitioned_pgrams + start_in_strings + greater_region,
           count_greater * sizeof(sz_pgram_t));
    memcpy(destination_order, partitioned_order + start_in_strings, count_smaller * sizeof(sz_size_t));
    memcpy(destination_order + count_smaller, partitioned_order + start_in_strings + equal_region,
           count_equal * sizeof(sz_size_t));
    memcpy(destination_order + count_smaller + count_equal, partitioned_order + start_in_strings + greater_region,
           count_greater * sizeof(sz_size_t));

    *first_pivot_offset = start_in_strings + count_smaller;
    *last_pivot_offset = start_in_strings + count_smaller + count_equal - 1;
}

STRINGZILLA_OUTLINED_ void sz_sort_quicksort_pgrams_haswell_(sz_pgram_t *initial_pgrams, sz_size_t *initial_order,
                                                             sz_pgram_t *temporary_pgrams, sz_size_t *temporary_order,
                                                             sz_size_t const start_in_strings,
                                                             sz_size_t const end_in_strings,
                                                             sz_size_t const top_count) {
    sz_size_t const count = end_in_strings - start_in_strings;
    if (count <= 32) { // Below the vectorization break-even, a stable insertion sort wins.
        sz_sort_pgrams_insertion_(initial_pgrams + start_in_strings, count, initial_order + start_in_strings);
        return;
    }

    sz_size_t first_pivot_index, last_pivot_index;
    sz_sort_3way_partition_haswell_(initial_pgrams, initial_order, temporary_pgrams, temporary_order, start_in_strings,
                                    end_in_strings, &first_pivot_index, &last_pivot_index);

    if (start_in_strings + 1 < first_pivot_index)
        sz_sort_quicksort_pgrams_haswell_(initial_pgrams, initial_order, temporary_pgrams, temporary_order,
                                          start_in_strings, first_pivot_index, top_count);
    if (last_pivot_index + 2 < end_in_strings && (top_count == 0 || last_pivot_index + 1 < top_count))
        sz_sort_quicksort_pgrams_haswell_(initial_pgrams, initial_order, temporary_pgrams, temporary_order,
                                          last_pivot_index + 1, end_in_strings, top_count);
}

STRINGZILLA_INLINE sz_status_t sz_sort_pgrams_haswell_(sz_pgram_t *pgrams, sz_size_t count, sz_allocator_t *allocator,
                                                       sz_size_t *order) {
    for (sz_size_t pgram_index = 0; pgram_index != count; ++pgram_index) order[pgram_index] = pgram_index;

    sz_allocator_t global_alloc;
    if (!allocator) {
        sz_status_t const status = sz_allocator_init_heap(&global_alloc);
        if (status != sz_success_k) return status;
        allocator = &global_alloc;
    }

    // Two scratch buffers, each over-allocated by `sz_sort_partition_slack_haswell_`
    // (region gaps + spill).
    sz_size_t const slack = sz_sort_partition_slack_haswell_;
    sz_size_t memory_usage = sizeof(sz_pgram_t) * (count + slack) + sizeof(sz_size_t) * (count + slack);
    sz_pgram_t *temporary_pgrams = (sz_pgram_t *)allocator->allocate(memory_usage, allocator->handle, STRINGZILLA_NULL);
    sz_size_t *temporary_order = (sz_size_t *)(temporary_pgrams + count + slack);
    if (!temporary_pgrams) return sz_bad_alloc_k;

    sz_sort_quicksort_pgrams_haswell_(pgrams, order, temporary_pgrams, temporary_order, 0, count, 0);

    allocator->free(temporary_pgrams, memory_usage, allocator->handle, STRINGZILLA_NULL);
    return sz_success_k;
}

STRINGZILLA_OUTLINED_ void sz_sort_byte_windows_haswell_(
    sz_strings_view_t const *const strings, sz_pgram_t *const global_pgrams, sz_size_t *const global_order,
    sz_pgram_t *const temporary_pgrams, sz_size_t *const temporary_order, sz_size_t const start_in_strings,
    sz_size_t const end_in_strings, sz_size_t const start_character, sz_size_t const top_count,
    sz_bool_t const reverse) {

    sz_sort_export_byte_window_serial_(strings, global_pgrams, global_order, start_in_strings, end_in_strings,
                                       start_character, reverse);

    sz_sort_quicksort_pgrams_haswell_(global_pgrams, global_order, temporary_pgrams, temporary_order, start_in_strings,
                                      end_in_strings, top_count);

    sz_size_t const pgram_capacity = sizeof(sz_pgram_t) - 1;
    sz_size_t nested_start = start_in_strings;
    sz_size_t nested_end = start_in_strings;
    while (nested_end != end_in_strings) {
        if (top_count != 0 && nested_start >= top_count) break;

        sz_pgram_t current_pgram = global_pgrams[nested_start];
        while (nested_end != end_in_strings && current_pgram == global_pgrams[nested_end]) ++nested_end;

        // x86 is always little-endian, so the packed length byte is the least-significant byte of the
        // (byte-reversed) pgram - a plain cast reads it with no endianness branch.
        sz_pgram_t const length_source = reverse ? ~current_pgram : current_pgram;
        sz_size_t current_pgram_length = (sz_size_t)(sz_u8_t)length_source;
        int has_multiple_strings = nested_end - nested_start > 1;
        int has_more_characters_in_each = current_pgram_length == pgram_capacity;
        if (has_multiple_strings && has_more_characters_in_each)
            sz_sort_byte_windows_haswell_(strings, global_pgrams, global_order, temporary_pgrams, temporary_order,
                                          nested_start, nested_end, start_character + pgram_capacity, top_count,
                                          reverse);
        else if (has_multiple_strings)
            sz_order_indices_ascending_(global_order + nested_start, nested_end - nested_start);
        nested_start = nested_end;
    }
}

/** Uncased twin of @c sz_sort_byte_windows_haswell_: the folded code-point export stays scalar and
 *  shared with the serial backend, while the pgrams it produces are sorted with the AVX2 partition,
 *  which is where Haswell beats the fully serial uncased path. */
STRINGZILLA_OUTLINED_ void sz_sort_casefold_windows_haswell_(
    sz_strings_view_t const *const strings, sz_pgram_t *const global_pgrams, sz_size_t *const global_order,
    sz_pgram_t *const temporary_pgrams, sz_size_t *const temporary_order, sz_size_t const start_in_strings,
    sz_size_t const end_in_strings, sz_size_t const folded_skip_count, sz_size_t const top_count,
    sz_bool_t const reverse) {

    sz_sort_export_casefold_window_serial_(strings, global_pgrams, global_order, start_in_strings, end_in_strings,
                                           folded_skip_count, reverse);
    sz_sort_quicksort_pgrams_haswell_(global_pgrams, global_order, temporary_pgrams, temporary_order, start_in_strings,
                                      end_in_strings, top_count);

    // A window's lowest 21-bit field is non-zero only when it was filled to capacity, so the equal group may
    // still carry more folded code-points and must recurse one window deeper (mirrors the serial folded sort).
    sz_size_t const fields_per_pgram = sz_argsort_casefold_fields_(sz_pgram_t);
    sz_pgram_t const lowest_field_mask = ((sz_pgram_t)1 << sz_argsort_casefold_field_bits_) - 1;
    sz_size_t nested_start = start_in_strings;
    sz_size_t nested_end = start_in_strings;
    while (nested_end != end_in_strings) {
        if (top_count != 0 && nested_start >= top_count) break;

        sz_pgram_t current_pgram = global_pgrams[nested_start];
        while (nested_end != end_in_strings && current_pgram == global_pgrams[nested_end]) ++nested_end;

        sz_pgram_t const decoded_pgram = reverse ? ~current_pgram : current_pgram;
        int has_multiple_strings = nested_end - nested_start > 1;
        int has_more_characters_in_each = (decoded_pgram & lowest_field_mask) != 0;
        if (has_multiple_strings && has_more_characters_in_each)
            sz_sort_casefold_windows_haswell_(strings, global_pgrams, global_order, temporary_pgrams, temporary_order,
                                              nested_start, nested_end, folded_skip_count + fields_per_pgram, top_count,
                                              reverse);
        else if (has_multiple_strings)
            sz_order_indices_ascending_(global_order + nested_start, nested_end - nested_start);
        nested_start = nested_end;
    }
}

#if STRINGZILLA_TARGET_HASWELL

STRINGZILLA_API sz_status_t sz_sort_order_haswell(sz_strings_view_t const *strings, sz_size_t top_count,
                                                  sz_bool_t reverse, sz_allocator_t *allocator, sz_size_t *order,
                                                  sz_stream_t stream) {
    sz_assert_(stream == STRINGZILLA_NULL);
    sz_size_t count = strings->count;
    for (sz_size_t strings_index = 0; strings_index != count; ++strings_index) order[strings_index] = strings_index;

    if (count <= 32 && !reverse) {
        sz_sort_insertion_(strings, order);
        return sz_success_k;
    }

    sz_allocator_t global_alloc;
    if (!allocator) {
        sz_status_t const status = sz_allocator_init_heap(&global_alloc);
        if (status != sz_success_k) return status;
        allocator = &global_alloc;
    }

    // One allocation backs three arrays laid out back-to-back:
    //   - `global_pgrams`    : the working pgram keys                           → `count`
    //   - `temporary_pgrams` : the 3-way partition's out-of-place pgram scratch → `count + slack`
    //   - `temporary_order`  : the matching order scratch                       → `count + slack`
    sz_size_t const slack = sz_sort_partition_slack_haswell_;
    sz_size_t const memory_usage = sizeof(sz_pgram_t) * (count + (count + slack)) // global + pgram scratch
                                   + sizeof(sz_size_t) * (count + slack);         // order scratch
    sz_pgram_t *global_pgrams = (sz_pgram_t *)allocator->allocate(memory_usage, allocator->handle, STRINGZILLA_NULL);
    sz_pgram_t *temporary_pgrams = global_pgrams + count;
    sz_size_t *temporary_order = (sz_size_t *)(temporary_pgrams + count + slack);
    if (!global_pgrams) return sz_bad_alloc_k;

    sz_sort_byte_windows_haswell_(strings, global_pgrams, order, temporary_pgrams, temporary_order, 0, count, 0,
                                  top_count, reverse);

    allocator->free(global_pgrams, memory_usage, allocator->handle, STRINGZILLA_NULL);
    return sz_success_k;
}

STRINGZILLA_API sz_status_t sz_sort_uncased_order_haswell(                    //
    sz_strings_view_t const *strings, sz_size_t top_count, sz_bool_t reverse, //
    sz_allocator_t *allocator, sz_size_t *order, sz_stream_t stream) {
    sz_assert_(stream == STRINGZILLA_NULL);

    sz_size_t const count = strings->count;
    for (sz_size_t strings_index = 0; strings_index != count; ++strings_index) order[strings_index] = strings_index;
    if (count < 2) return sz_success_k;

    sz_allocator_t global_alloc;
    if (!allocator) {
        sz_status_t const status = sz_allocator_init_heap(&global_alloc);
        if (status != sz_success_k) return status;
        allocator = &global_alloc;
    }

    // Same layout as the byte arg-sort - working pgrams (count) + the partition's two scratch regions
    // (count + slack each). The folded export is stateless (re-folds the prefix on demand), so unlike the
    // earlier design there is no per-string cursor array.
    sz_size_t const slack = sz_sort_partition_slack_haswell_;
    sz_size_t const memory_usage = sizeof(sz_pgram_t) * (count + (count + slack)) // global + pgram scratch
                                   + sizeof(sz_size_t) * (count + slack);         // order scratch
    sz_pgram_t *global_pgrams = (sz_pgram_t *)allocator->allocate(memory_usage, allocator->handle, STRINGZILLA_NULL);
    if (!global_pgrams) return sz_bad_alloc_k;
    sz_pgram_t *temporary_pgrams = global_pgrams + count;
    sz_size_t *temporary_order = (sz_size_t *)(temporary_pgrams + count + slack);

    sz_sort_casefold_windows_haswell_(strings, global_pgrams, order, temporary_pgrams, temporary_order, 0, count, 0,
                                      top_count, reverse);

    allocator->free(global_pgrams, memory_usage, allocator->handle, STRINGZILLA_NULL);
    return sz_success_k;
}

#endif // STRINGZILLA_TARGET_HASWELL

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#endif // STRINGZILLA_ARCH_X8664_HASWELL_
#endif // STRINGZILLA_ARCH_X8664_

#ifdef __cplusplus
}
#endif

#endif // STRINGZILLA_SORT_HASWELL_H_
