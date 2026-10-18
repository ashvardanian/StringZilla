/**
 *  @brief SVE (Arm) backend for sorting string collections.
 *  @file include/stringzilla/sort/sve.h
 *  @author Ash Vardanian
 *  @sa include/stringzilla/sort.h
 */
#ifndef STRINGZILLA_SORT_SVE_H_
#define STRINGZILLA_SORT_SVE_H_

#include "stringzilla/types.h"
#include "stringzilla/compare.h" // `sz_compare`
#include "stringzilla/memory.h"  // `sz_copy`

#include "stringzilla/sort/serial.h"

#ifdef __cplusplus
extern "C" {
#endif

#if SZ_USE_SVE
#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("+sve"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("+sve")
#endif

/**
 *  @brief The most important part of the QuickSort algorithm partitioning the elements around the pivot.
 *  @note Unlike the serial algorithm, uses compressed stores to filter and move the elements around the pivot.
 *  @sa Identical to @b Skylake implementation, but uses variable length SVE registers.
 *
 *  @param initial_pgrams Source pgram array; updated in place after copy-back.
 *  @param initial_order Corresponding order array; updated in place after copy-back.
 *  @param partitioned_pgrams Temporary output buffer for the three-way partitioned pgrams.
 *  @param partitioned_order Temporary output buffer for the corresponding order entries.
 *  @param start_in_sequence First index (inclusive) of the range to partition.
 *  @param end_in_sequence One-past-the-last index of the range to partition.
 *  @param first_pivot_offset Receives the index of the first element equal to the pivot.
 *  @param last_pivot_offset Receives the index of the last element equal to the pivot.
 */
SZ_HELPER_AUTO void sz_sequence_argsort_sve_3way_partition_(
    sz_pgram_t *const initial_pgrams, sz_sorted_idx_t *const initial_order, sz_pgram_t *const partitioned_pgrams,
    sz_sorted_idx_t *const partitioned_order, sz_size_t const start_in_sequence, sz_size_t const end_in_sequence,
    sz_size_t *const first_pivot_offset, sz_size_t *const last_pivot_offset) {
    sz_size_t const count = end_in_sequence - start_in_sequence;

    // Use `svcntd()` to obtain the number of 64-bit elements in one SVE vector.
    sz_size_t const pgrams_per_vector = svcntd();

    // Choose the pivot with Sedgewick's method.
    sz_pgram_t const *pivot_pgram_ptr = sz_sequence_partitioning_pivot_(initial_pgrams + start_in_sequence, count);
    sz_pgram_t const pivot_pgram = *pivot_pgram_ptr;
    svuint64_t pivot_u64x = svdup_n_u64(pivot_pgram);

    // Count elements smaller and greater than the pivot.
    sz_size_t count_smaller = 0, count_greater = 0;
    for (sz_size_t block_index = start_in_sequence; block_index < end_in_sequence; block_index += pgrams_per_vector) {
        svbool_t load_mask_b64x = svwhilelt_b64((sz_u64_t)block_index, (sz_u64_t)end_in_sequence);
        svuint64_t pgrams_u64x = svld1_u64(load_mask_b64x, (sz_u64_t const *)(initial_pgrams + block_index));
        svbool_t smaller_b64x = svcmplt_u64(load_mask_b64x, pgrams_u64x, pivot_u64x);
        svbool_t greater_b64x = svcmpgt_u64(load_mask_b64x, pgrams_u64x, pivot_u64x);
        count_smaller = svqincp_n_u64_b64(count_smaller, smaller_b64x); // Smarter than `svcntp_b64`
        count_greater = svqincp_n_u64_b64(count_greater, greater_b64x); // Smarter than `svcntp_b64`
    }

    sz_size_t const count_equal = count - count_smaller - count_greater;
    sz_assert_(count_equal >= 1 && "The pivot must be present in the collection.");
    sz_assert_(count_smaller + count_equal + count_greater == count && "The partitioning must be exhaustive.");

    // Set offsets for each partition.
    sz_size_t smaller_offset = start_in_sequence;
    sz_size_t equal_offset = start_in_sequence + count_smaller;
    sz_size_t greater_offset = start_in_sequence + count_smaller + count_equal;

    // Partition elements into three segments.
    for (sz_size_t block_index = start_in_sequence; block_index < end_in_sequence; block_index += pgrams_per_vector) {
        svbool_t load_mask_b64x = svwhilelt_b64((sz_u64_t)block_index, (sz_u64_t)end_in_sequence);
        svuint64_t pgrams_u64x = svld1_u64(load_mask_b64x, (sz_u64_t const *)(initial_pgrams + block_index));
        svuint64_t order_u64x = svld1_u64(load_mask_b64x, (sz_u64_t const *)(initial_order + block_index));

        svbool_t smaller_b64x = svcmplt_u64(load_mask_b64x, pgrams_u64x, pivot_u64x);
        svbool_t greater_b64x = svcmpgt_u64(load_mask_b64x, pgrams_u64x, pivot_u64x);
        // The equal lanes are the active lanes that are neither smaller nor greater, so a zeroing NOR governed
        // by the load mask derives them without a third compare (inactive lanes stay false).
        svbool_t equal_b64x = svnor_b_z(load_mask_b64x, smaller_b64x, greater_b64x);

        // Compress the elements that satisfy the predicate and store them contiguously. A store masked to a
        // zero count writes nothing, so no per-segment branch guards the stores.
        sz_size_t block_count_smaller = svcntp_b64(smaller_b64x, smaller_b64x);
        sz_size_t block_count_equal = svcntp_b64(equal_b64x, equal_b64x);
        sz_size_t block_count_greater = svcntp_b64(greater_b64x, greater_b64x);

        svuint64_t comp_smaller_pgrams_u64x = svcompact_u64(smaller_b64x, pgrams_u64x);
        svuint64_t comp_smaller_order_u64x = svcompact_u64(smaller_b64x, order_u64x);
        svbool_t smaller_store_mask_b64x = svwhilelt_b64((sz_u64_t)0, (sz_u64_t)block_count_smaller);
        svst1_u64(smaller_store_mask_b64x, (sz_u64_t *)(partitioned_pgrams + smaller_offset), comp_smaller_pgrams_u64x);
        svst1_u64(smaller_store_mask_b64x, (sz_u64_t *)(partitioned_order + smaller_offset), comp_smaller_order_u64x);
        smaller_offset += block_count_smaller;

        svuint64_t comp_equal_pgrams_u64x = svcompact_u64(equal_b64x, pgrams_u64x);
        svuint64_t comp_equal_order_u64x = svcompact_u64(equal_b64x, order_u64x);
        svbool_t equal_store_mask_b64x = svwhilelt_b64((sz_u64_t)0, (sz_u64_t)block_count_equal);
        svst1_u64(equal_store_mask_b64x, (sz_u64_t *)(partitioned_pgrams + equal_offset), comp_equal_pgrams_u64x);
        svst1_u64(equal_store_mask_b64x, (sz_u64_t *)(partitioned_order + equal_offset), comp_equal_order_u64x);
        equal_offset += block_count_equal;

        svuint64_t comp_greater_pgrams_u64x = svcompact_u64(greater_b64x, pgrams_u64x);
        svuint64_t comp_greater_order_u64x = svcompact_u64(greater_b64x, order_u64x);
        svbool_t greater_store_mask_b64x = svwhilelt_b64((sz_u64_t)0, (sz_u64_t)block_count_greater);
        svst1_u64(greater_store_mask_b64x, (sz_u64_t *)(partitioned_pgrams + greater_offset), comp_greater_pgrams_u64x);
        svst1_u64(greater_store_mask_b64x, (sz_u64_t *)(partitioned_order + greater_offset), comp_greater_order_u64x);
        greater_offset += block_count_greater;
    }

    // Copy back.
    sz_copy_sve((sz_ptr_t)(initial_pgrams + start_in_sequence),      //
                (sz_cptr_t)(partitioned_pgrams + start_in_sequence), //
                count * sizeof(sz_pgram_t));
    sz_copy_sve((sz_ptr_t)(initial_order + start_in_sequence),      //
                (sz_cptr_t)(partitioned_order + start_in_sequence), //
                count * sizeof(sz_sorted_idx_t));

    // Return the offsets of the equal elements.
    *first_pivot_offset = start_in_sequence + count_smaller;
    *last_pivot_offset = start_in_sequence + count_smaller + count_equal - 1;
}

/**
 *  @brief Recursive Quick-Sort implementation backing both the `sz_sequence_argsort_sve` and
 *      `sz_pgrams_sort_sve`, and using the `sz_sequence_argsort_sve_3way_partition_` under the hood.
 *  @sa Identical to @b Skylake implementation, but uses variable length SVE registers.
 *
 *  @param initial_pgrams Pgram array to sort in place.
 *  @param initial_order Corresponding order array, permuted in sync with `initial_pgrams`.
 *  @param temporary_pgrams Scratch buffer of the same size as `initial_pgrams`, used during partitioning.
 *  @param temporary_order Scratch buffer of the same size as `initial_order`, used during partitioning.
 *  @param start_in_sequence First index (inclusive) of the range to sort.
 *  @param end_in_sequence One-past-the-last index of the range to sort.
 */
SZ_API_COMPTIME void sz_sequence_argsort_sve_quicksort_pgrams_(
    sz_pgram_t *initial_pgrams, sz_sorted_idx_t *initial_order, sz_pgram_t *temporary_pgrams,
    sz_sorted_idx_t *temporary_order, sz_size_t const start_in_sequence, sz_size_t const end_in_sequence,
    sz_size_t const top_count) {
    sz_size_t const count = end_in_sequence - start_in_sequence;
    sz_size_t const pgrams_per_vector = svcntd();
    if (count <= pgrams_per_vector) {
        // For very small arrays use a simple insertion sort.
        sz_pgrams_sort_with_insertion(initial_pgrams + start_in_sequence, count, initial_order + start_in_sequence);
        return;
    }

    sz_size_t first_pivot_index, last_pivot_index;
    sz_sequence_argsort_sve_3way_partition_(initial_pgrams, initial_order, temporary_pgrams, temporary_order,
                                            start_in_sequence, end_in_sequence, &first_pivot_index, &last_pivot_index);

    if (start_in_sequence + 1 < first_pivot_index)
        sz_sequence_argsort_sve_quicksort_pgrams_(initial_pgrams, initial_order, temporary_pgrams, temporary_order,
                                                  start_in_sequence, first_pivot_index, top_count);
    if (last_pivot_index + 2 < end_in_sequence && (top_count == 0 || last_pivot_index + 1 < top_count))
        sz_sequence_argsort_sve_quicksort_pgrams_(initial_pgrams, initial_order, temporary_pgrams, temporary_order,
                                                  last_pivot_index + 1, end_in_sequence, top_count);
}

SZ_API_COMPTIME sz_status_t sz_pgrams_sort_sve(sz_pgram_t *pgrams, sz_size_t count, sz_memory_allocator_t *alloc,
                                               sz_sorted_idx_t *order) {
    // Initialize the order with 0,1,2,...
    for (sz_size_t pgram_index = 0; pgram_index != count; ++pgram_index) order[pgram_index] = pgram_index;

    sz_memory_allocator_t global_alloc;
    if (!alloc) {
        sz_memory_allocator_init_default(&global_alloc);
        alloc = &global_alloc;
    }

    // Allocate temporary memory for partitioning.
    sz_size_t memory_usage = sizeof(sz_pgram_t) * count + sizeof(sz_sorted_idx_t) * count;
    sz_pgram_t *temporary_pgrams = (sz_pgram_t *)alloc->allocate(memory_usage, alloc);
    sz_sorted_idx_t *temporary_order = (sz_sorted_idx_t *)(temporary_pgrams + count);
    if (!temporary_pgrams) return sz_bad_alloc_k;

    sz_sequence_argsort_sve_quicksort_pgrams_(pgrams, order, temporary_pgrams, temporary_order, 0, count, 0);

    alloc->free(temporary_pgrams, memory_usage, alloc);
    return sz_success_k;
}

typedef struct sz_argsort_sve_scratch_t {
    sz_pgram_t *temporary_pgrams;
    sz_sorted_idx_t *temporary_order;
} sz_argsort_sve_scratch_t;

SZ_HELPER_AUTO void sz_argsort_sve_sort_range_(sz_pgram_t *pgrams, sz_sorted_idx_t *order, sz_size_t start,
                                               sz_size_t end, sz_size_t top_count, void *context) {
    sz_argsort_sve_scratch_t *scratch = (sz_argsort_sve_scratch_t *)context;
    sz_sequence_argsort_sve_quicksort_pgrams_(pgrams, order, scratch->temporary_pgrams, scratch->temporary_order, start,
                                              end, top_count);
}

/**
 *  @brief Quick-Sort adaptation for strings, that processes the strings a few N-grams at a time.
 *      It combines `sz_sequence_argsort_serial_export_byte_window_` and `sz_sequence_argsort_sve_quicksort_pgrams_`.
 *      Equal pgrams are walked iteratively, so a long shared prefix does not grow the call stack.
 *  @sa Identical to @b Skylake implementation, but uses variable length SVE registers.
 *
 *  @param sequence The collection of strings to sort.
 *  @param global_pgrams Working pgram array, length at least `sequence->count`.
 *  @param global_order Current permutation array, updated in place.
 *  @param temporary_pgrams Scratch buffer of the same size as `global_pgrams`.
 *  @param temporary_order Scratch buffer of the same size as `global_order`.
 *  @param start_in_sequence First index (inclusive) of the range to process.
 *  @param end_in_sequence One-past-the-last index of the range to process.
 *  @param start_character Byte offset into each string for the current pgram window.
 *  @param top_count Global top-K cut-off forwarded to the partitioner; 0 fully sorts the range.
 *  @param reverse Whether to export complemented keys for descending order.
 *  @param frames Scratch stack of at least `end_in_sequence - start_in_sequence` window frames.
 */
SZ_API_COMPTIME void sz_sequence_argsort_sve_sort_byte_windows_(
    sz_sequence_t const *const sequence, sz_pgram_t *const global_pgrams, sz_sorted_idx_t *const global_order,
    sz_pgram_t *const temporary_pgrams, sz_sorted_idx_t *const temporary_order, sz_size_t const start_in_sequence,
    sz_size_t const end_in_sequence, sz_size_t const start_character, sz_size_t const top_count,
    sz_bool_t const reverse, sz_argsort_window_frame_t *const frames) {

    sz_argsort_sve_scratch_t scratch = {temporary_pgrams, temporary_order};
    sz_argsort_walk_byte_windows_(sequence, global_pgrams, global_order, start_in_sequence, end_in_sequence,
                                  start_character, top_count, reverse, sz_argsort_sve_sort_range_, &scratch, frames);
}

SZ_API_COMPTIME sz_status_t sz_sequence_argsort_sve(sz_sequence_t const *sequence, sz_memory_allocator_t *alloc,
                                                    sz_sorted_idx_t *order, sz_size_t top_count, sz_bool_t reverse) {
    sz_size_t count = sequence->count;
    for (sz_size_t sequence_index = 0; sequence_index != count; ++sequence_index)
        order[sequence_index] = sequence_index;
    if (count < 2) return sz_success_k;

    if (count <= 32 && !reverse) {
        sz_sequence_argsort_with_insertion(sequence, order);
        return sz_success_k;
    }

    sz_memory_allocator_t global_alloc;
    if (!alloc) {
        sz_memory_allocator_init_default(&global_alloc);
        alloc = &global_alloc;
    }

    sz_size_t memory_usage = sizeof(sz_pgram_t) * count * 2 + sizeof(sz_sorted_idx_t) * count +
                             sizeof(sz_argsort_window_frame_t) * count;
    sz_pgram_t *global_pgrams = (sz_pgram_t *)alloc->allocate(memory_usage, alloc);
    if (!global_pgrams) return sz_bad_alloc_k;
    sz_pgram_t *temporary_pgrams = global_pgrams + count;
    sz_sorted_idx_t *temporary_order = (sz_sorted_idx_t *)(temporary_pgrams + count);
    sz_argsort_window_frame_t *frames = (sz_argsort_window_frame_t *)(temporary_order + count);

    sz_sequence_argsort_sve_sort_byte_windows_(sequence, global_pgrams, order, temporary_pgrams, temporary_order, 0,
                                               count, 0, top_count, reverse, frames);

    alloc->free(global_pgrams, memory_usage, alloc);
    return sz_success_k;
}

/**
 *  @brief Uncased twin of `sz_sequence_argsort_sve_sort_byte_windows_`: the folded code-point export
 *      stays scalar (and is shared with the serial backend), but the pgrams it produces are sorted with the
 *      SVE partition - which is where SVE beats the fully-serial uncased path.
 */
SZ_API_COMPTIME void sz_sequence_argsort_sve_sort_casefold_windows_(
    sz_sequence_t const *const sequence, sz_pgram_t *const global_pgrams, sz_sorted_idx_t *const global_order,
    sz_pgram_t *const temporary_pgrams, sz_sorted_idx_t *const temporary_order, sz_size_t const start_in_sequence,
    sz_size_t const end_in_sequence, sz_size_t const folded_skip_count, sz_size_t const top_count,
    sz_bool_t const reverse, sz_argsort_window_frame_t *const frames) {

    sz_argsort_sve_scratch_t scratch = {temporary_pgrams, temporary_order};
    sz_argsort_walk_casefold_windows_(sequence, global_pgrams, global_order, start_in_sequence, end_in_sequence,
                                      folded_skip_count, top_count, reverse, sz_argsort_sve_sort_range_, &scratch,
                                      frames);
}

SZ_API_COMPTIME sz_status_t sz_sequence_argsort_uncased_sve(     //
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

    // Same layout as the byte arg-sort - working pgrams (count) + the partition's two scratch regions. The
    // SVE compress-store is exact, so no slack is needed. The folded export is stateless (re-folds the prefix
    // on demand), so unlike the earlier design there is no per-string cursor array.
    sz_size_t const memory_usage = sizeof(sz_pgram_t) * count * 2 + sizeof(sz_sorted_idx_t) * count +
                                   sizeof(sz_argsort_window_frame_t) * count;
    sz_pgram_t *global_pgrams = (sz_pgram_t *)alloc->allocate(memory_usage, alloc);
    if (!global_pgrams) return sz_bad_alloc_k;
    sz_pgram_t *temporary_pgrams = global_pgrams + count;
    sz_sorted_idx_t *temporary_order = (sz_sorted_idx_t *)(temporary_pgrams + count);
    sz_argsort_window_frame_t *frames = (sz_argsort_window_frame_t *)(temporary_order + count);

    sz_sequence_argsort_sve_sort_casefold_windows_(sequence, global_pgrams, order, temporary_pgrams, temporary_order, 0,
                                                   count, 0, top_count, reverse, frames);

    alloc->free(global_pgrams, memory_usage, alloc);
    return sz_success_k;
}

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#endif // SZ_USE_SVE

#ifdef __cplusplus
}
#endif

#endif // STRINGZILLA_SORT_SVE_H_
