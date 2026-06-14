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
SZ_INTERNAL void sz_sequence_argsort_sve_3way_partition_(
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
        svbool_t equal_b64x = svcmpeq_u64(load_mask_b64x, pgrams_u64x, pivot_u64x);
        svbool_t greater_b64x = svcmpgt_u64(load_mask_b64x, pgrams_u64x, pivot_u64x);

        // Compress the elements that satisfy the predicate and store them contiguously.
        sz_size_t block_count_smaller = svcntp_b64(smaller_b64x, smaller_b64x);
        sz_size_t block_count_equal = svcntp_b64(equal_b64x, equal_b64x);
        sz_size_t block_count_greater = svcntp_b64(greater_b64x, greater_b64x);
        if (block_count_smaller) {
            svuint64_t comp_pgrams_u64x = svcompact_u64(smaller_b64x, pgrams_u64x);
            svuint64_t comp_order_u64x = svcompact_u64(smaller_b64x, order_u64x);
            svbool_t store_mask_b64x = svwhilelt_b64((sz_u64_t)0, (sz_u64_t)block_count_smaller);
            svst1_u64(store_mask_b64x, (sz_u64_t *)(partitioned_pgrams + smaller_offset), comp_pgrams_u64x);
            svst1_u64(store_mask_b64x, (sz_u64_t *)(partitioned_order + smaller_offset), comp_order_u64x);
            smaller_offset += block_count_smaller;
        }
        if (block_count_equal) {
            svuint64_t comp_pgrams_u64x = svcompact_u64(equal_b64x, pgrams_u64x);
            svuint64_t comp_order_u64x = svcompact_u64(equal_b64x, order_u64x);
            svbool_t store_mask_b64x = svwhilelt_b64((sz_u64_t)0, (sz_u64_t)block_count_equal);
            svst1_u64(store_mask_b64x, (sz_u64_t *)(partitioned_pgrams + equal_offset), comp_pgrams_u64x);
            svst1_u64(store_mask_b64x, (sz_u64_t *)(partitioned_order + equal_offset), comp_order_u64x);
            equal_offset += block_count_equal;
        }
        if (block_count_greater) {
            svuint64_t comp_pgrams_u64x = svcompact_u64(greater_b64x, pgrams_u64x);
            svuint64_t comp_order_u64x = svcompact_u64(greater_b64x, order_u64x);
            svbool_t store_mask_b64x = svwhilelt_b64((sz_u64_t)0, (sz_u64_t)block_count_greater);
            svst1_u64(store_mask_b64x, (sz_u64_t *)(partitioned_pgrams + greater_offset), comp_pgrams_u64x);
            svst1_u64(store_mask_b64x, (sz_u64_t *)(partitioned_order + greater_offset), comp_order_u64x);
            greater_offset += block_count_greater;
        }
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
SZ_PUBLIC void sz_sequence_argsort_sve_recursively_(sz_pgram_t *initial_pgrams, sz_sorted_idx_t *initial_order,
                                                    sz_pgram_t *temporary_pgrams, sz_sorted_idx_t *temporary_order,
                                                    sz_size_t const start_in_sequence, sz_size_t const end_in_sequence,
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
        sz_sequence_argsort_sve_recursively_(initial_pgrams, initial_order, temporary_pgrams, temporary_order,
                                             start_in_sequence, first_pivot_index, top_count);
    if (last_pivot_index + 2 < end_in_sequence && (top_count == 0 || last_pivot_index + 1 < top_count))
        sz_sequence_argsort_sve_recursively_(initial_pgrams, initial_order, temporary_pgrams, temporary_order,
                                             last_pivot_index + 1, end_in_sequence, top_count);
}

SZ_PUBLIC sz_status_t sz_pgrams_sort_sve(sz_pgram_t *pgrams, sz_size_t count, sz_memory_allocator_t *alloc,
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

    sz_sequence_argsort_sve_recursively_(pgrams, order, temporary_pgrams, temporary_order, 0, count, 0);

    alloc->free(temporary_pgrams, memory_usage, alloc);
    return sz_success_k;
}

/**
 *  @brief Recursive Quick-Sort adaptation for strings, that processes the strings a few N-grams at a time.
 *      It combines `sz_sequence_argsort_serial_export_next_pgrams_` and `sz_sequence_argsort_sve_recursively_`,
 *      recursively diving into groups of identical pgrams.
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
 */
SZ_PUBLIC void sz_sequence_argsort_sve_next_pgrams_(
    sz_sequence_t const *const sequence, sz_pgram_t *const global_pgrams, sz_sorted_idx_t *const global_order,
    sz_pgram_t *const temporary_pgrams, sz_sorted_idx_t *const temporary_order, sz_size_t const start_in_sequence,
    sz_size_t const end_in_sequence, sz_size_t const start_character, sz_size_t const top_count,
    sz_bool_t const reverse) {

    // Export the next pgrams from the sequence.
    sz_sequence_argsort_serial_export_next_pgrams_(sequence, global_pgrams, global_order, start_in_sequence,
                                                   end_in_sequence, start_character, reverse);

    // Sort the current pgrams with the SVE quicksort.
    sz_sequence_argsort_sve_recursively_(global_pgrams, global_order, temporary_pgrams, temporary_order,
                                         start_in_sequence, end_in_sequence, top_count);

    // For each group of equal pgrams, if there are multiple strings and more characters,
    // recursively sort the next pgrams.
    sz_size_t const pgram_capacity = sizeof(sz_pgram_t) - 1;
    sz_size_t nested_start = start_in_sequence;
    sz_size_t nested_end = start_in_sequence;
    while (nested_end != end_in_sequence) {
        // Everything from `top_count` onwards needs no ordering - the wanted elements are already in front.
        if (top_count != 0 && nested_start >= top_count) break;

        sz_pgram_t current_pgram = global_pgrams[nested_start];
        while (nested_end != end_in_sequence && current_pgram == global_pgrams[nested_end]) ++nested_end;

        // The packed length byte lives in the low byte after the byte-reversal; under `reverse` the
        // whole key was complemented, so we complement back before reading it.
        sz_pgram_t const length_source = reverse ? ~current_pgram : current_pgram;
        sz_cptr_t const length_str = (sz_cptr_t)&length_source;
#if !SZ_IS_BIG_ENDIAN_
        sz_size_t current_pgram_length = (sz_size_t)(sz_u8_t)length_str[0]; //! The byte order was swapped
#else
        sz_size_t current_pgram_length = (sz_size_t)(sz_u8_t)length_str[pgram_capacity]; //! No swaps on big-endian
#endif
        int has_multiple_strings = nested_end - nested_start > 1;
        int has_more_characters_in_each = current_pgram_length == pgram_capacity;
        if (has_multiple_strings && has_more_characters_in_each)
            sz_sequence_argsort_sve_next_pgrams_(sequence, global_pgrams, global_order, temporary_pgrams,
                                                 temporary_order, nested_start, nested_end,
                                                 start_character + pgram_capacity, top_count, reverse);
        else if (has_multiple_strings)
            // Terminal run of byte-identical strings: restore stable order by original index.
            sz_order_indices_ascending_(global_order + nested_start, nested_end - nested_start);
        nested_start = nested_end;
    }
}

SZ_PUBLIC sz_status_t sz_sequence_argsort_sve(sz_sequence_t const *sequence, sz_memory_allocator_t *alloc,
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

    sz_size_t memory_usage = sizeof(sz_pgram_t) * count * 2 + sizeof(sz_sorted_idx_t) * count;
    sz_pgram_t *global_pgrams = (sz_pgram_t *)alloc->allocate(memory_usage, alloc);
    sz_pgram_t *temporary_pgrams = global_pgrams + count;
    sz_sorted_idx_t *temporary_order = (sz_sorted_idx_t *)(temporary_pgrams + count);
    if (!global_pgrams) return sz_bad_alloc_k;

    sz_sequence_argsort_sve_next_pgrams_(sequence, global_pgrams, order, temporary_pgrams, temporary_order, 0, count, 0,
                                         top_count, reverse);

    alloc->free(global_pgrams, memory_usage, alloc);
    return sz_success_k;
}

SZ_PUBLIC sz_status_t sz_sequence_argsort_utf8_case_insensitive_sve( //
    sz_sequence_t const *sequence, sz_memory_allocator_t *alloc,     //
    sz_sorted_idx_t *order, sz_size_t top_count, sz_bool_t reverse) {
    // Case-folding the pgram window on the fly is inherently scalar, so the SIMD partition buys little
    // over the folding cost; reuse the serial folded sort verbatim.
    return sz_sequence_argsort_utf8_case_insensitive_serial(sequence, alloc, order, top_count, reverse);
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
