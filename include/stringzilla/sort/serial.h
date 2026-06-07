/**
 *  @brief Serial backend for sorting string collections.
 *  @file include/stringzilla/sort/serial.h
 *  @author Ash Vardanian
 *  @sa include/stringzilla/sort.h
 */
#ifndef STRINGZILLA_SORT_SERIAL_H_
#define STRINGZILLA_SORT_SERIAL_H_

#include "stringzilla/types.h"
#include "stringzilla/compare.h" // `sz_compare`
#include "stringzilla/memory.h"  // `sz_copy`

#ifdef __cplusplus
extern "C" {
#endif

SZ_INTERNAL void sz_sequence_argsort_serial_export_next_pgrams_(                //
    sz_sequence_t const *const sequence,                                        //
    sz_pgram_t *const global_pgrams, sz_sorted_idx_t const *const global_order, //
    sz_size_t const start_in_sequence, sz_size_t const end_in_sequence,         //
    sz_size_t const start_character) {

    // Depending on the architecture, we will export a different number of bytes.
    // On 32-bit architectures, we will export 3 bytes, and on 64-bit architectures - 7 bytes.
    sz_size_t const pgram_capacity = sizeof(sz_pgram_t) - 1;

    // Perform the same operation for every string.
    for (sz_size_t i = start_in_sequence; i < end_in_sequence; ++i) {

        // On the first recursion level, the `global_order` is the identity permutation.
        sz_sorted_idx_t const partial_order_index = global_order[i];
        if (SZ_DEBUG && start_character == 0)
            sz_assert_(partial_order_index == i && "At start this must be an identity permutation.");

        // Get the string slice in global memory.
        sz_cptr_t const source_str = sequence->get_start(sequence->handle, partial_order_index);
        sz_size_t const length = sequence->get_length(sequence->handle, partial_order_index);
        sz_size_t const remaining_length = length > start_character ? length - start_character : 0;
        sz_size_t const exported_length = remaining_length > pgram_capacity ? pgram_capacity : remaining_length;

        // Fill with zeros, export a slice, and mark the exported length.
        sz_pgram_t *target_pgram = &global_pgrams[i];
        sz_ptr_t target_str = (sz_ptr_t)target_pgram;
        *target_pgram = 0;
        for (sz_size_t j = 0; j < exported_length; ++j) target_str[j] = source_str[j + start_character];
        target_str[pgram_capacity] = (char)exported_length;
#if !SZ_IS_BIG_ENDIAN_
#if SZ_IS_64BIT_
        *target_pgram = sz_u64_bytes_reverse(*target_pgram);
#else
        *target_pgram = sz_u32_bytes_reverse(*target_pgram);
#endif
#endif
        sz_assert_(                                                //
            (length <= start_character) == (*target_pgram == 0) && //
            "We can have a zero value if only the string is shorter than other strings at this position.");
    }

    // As our goal is to sort the strings using the exported integer "pgrams",
    // this is a good place to validate the correctness of the exported data.
    if (SZ_DEBUG && start_character == 0)
        for (sz_size_t i = start_in_sequence + 1; i < end_in_sequence; ++i) {
            sz_pgram_t const previous_pgram = global_pgrams[i - 1];
            sz_pgram_t const current_pgram = global_pgrams[i];
            sz_cptr_t const previous_str = sequence->get_start(sequence->handle, i - 1);
            sz_size_t const previous_length = sequence->get_length(sequence->handle, i - 1);
            sz_cptr_t const current_str = sequence->get_start(sequence->handle, i);
            sz_size_t const current_length = sequence->get_length(sequence->handle, i);
            sz_ordering_t const ordering = sz_order(                                               //
                previous_str, previous_length > pgram_capacity ? pgram_capacity : previous_length, //
                current_str, current_length > pgram_capacity ? pgram_capacity : current_length);
            sz_assert_(                                                        //
                (previous_pgram < current_pgram) == (ordering == sz_less_k) && //
                "The exported pgrams should be in the same order as the original strings.");
        }
}

/**
 *  @brief Picks the "pivot" value for the QuickSort algorithm's partitioning step using Robert Sedgewick's method,
 *         the median of three elements - the first, the middle, and the last element of the given range.
 */
SZ_INTERNAL sz_pgram_t const *sz_sequence_partitioning_pivot_(sz_pgram_t const *pgrams, sz_size_t count) {
    sz_size_t const middle_offset = count / 2;
    sz_pgram_t const *first_pgram = &pgrams[0];
    sz_pgram_t const *middle_pgram = &pgrams[middle_offset];
    sz_pgram_t const *last_pgram = &pgrams[count - 1];
    if (*first_pgram < *middle_pgram) {
        if (*middle_pgram < *last_pgram) { return middle_pgram; }
        else if (*first_pgram < *last_pgram) { return last_pgram; }
        else { return first_pgram; }
    }
    else {
        if (*first_pgram < *last_pgram) { return first_pgram; }
        else if (*middle_pgram < *last_pgram) { return last_pgram; }
        else { return middle_pgram; }
    }
}

/**
 *  @brief The most important part of the QuickSort algorithm partitioning the elements around the pivot.
 *
 *  The classical variant uses the normal 2-way partitioning, but it will scatter the values equal to the pivot
 *  into the left and right partitions. Instead we use the Dutch National Flag @b 3-way partitioning, outputting
 *  the range of values equal to the pivot.
 *
 *  @see https://en.wikipedia.org/wiki/Dutch_national_flag_problem
 */
SZ_INTERNAL void sz_sequence_argsort_serial_3way_partition_(              //
    sz_pgram_t *const global_pgrams, sz_sorted_idx_t *const global_order, //
    sz_size_t const start_in_sequence, sz_size_t const end_in_sequence,   //
    sz_size_t *first_pivot_offset, sz_size_t *last_pivot_offset) {

    // On very small inputs this procedure is rudimentary.
    sz_size_t const count = end_in_sequence - start_in_sequence;
    if (count <= 4) {
        sz_pgram_t *const pgrams = global_pgrams + start_in_sequence;
        sz_sorted_idx_t *const offsets = global_order + start_in_sequence;
        if (count == 2) { sz_sequence_sorting_network_2x_(pgrams, offsets); }
        else if (count == 3) { sz_sequence_sorting_network_3x_(pgrams, offsets); }
        else if (count == 4) { sz_sequence_sorting_network_4x_(pgrams, offsets); }
        *first_pivot_offset = start_in_sequence;
        *last_pivot_offset = end_in_sequence;
        return;
    }

    // Chose the pivot offset with Sedgewick's method.
    sz_pgram_t const pivot_pgram = *sz_sequence_partitioning_pivot_(global_pgrams + start_in_sequence, count);

    // Loop through the collection and move the elements around the pivot with the 3-way partitioning.
    sz_size_t partitioning_progress = start_in_sequence; // Current index.
    sz_size_t smaller_offset = start_in_sequence;        // Boundary for elements < `pivot_pgram`.
    sz_size_t greater_offset = end_in_sequence - 1;      // Boundary for elements > `pivot_pgram`.

    while (partitioning_progress <= greater_offset) {
        // Element is less than pivot: swap into the < pivot region.
        if (global_pgrams[partitioning_progress] < pivot_pgram) {
            sz_swap_(sz_sorted_idx_t, global_order[partitioning_progress], global_order[smaller_offset]);
            sz_swap_(sz_pgram_t, global_pgrams[partitioning_progress], global_pgrams[smaller_offset]);
            ++partitioning_progress;
            ++smaller_offset;
        }
        // Element is greater than pivot: swap into the > pivot region.
        else if (global_pgrams[partitioning_progress] > pivot_pgram) {
            sz_swap_(sz_sorted_idx_t, global_order[partitioning_progress], global_order[greater_offset]);
            sz_swap_(sz_pgram_t, global_pgrams[partitioning_progress], global_pgrams[greater_offset]);
            --greater_offset;
        }
        // Element equals `pivot_pgram`: leave it in place.
        else { ++partitioning_progress; }
    }

    *first_pivot_offset = smaller_offset;
    *last_pivot_offset = greater_offset;
}

/**
 *  @brief Recursive Quick-Sort implementation backing both the `sz_sequence_argsort` and `sz_pgrams_sort`,
 *         and using the `sz_sequence_argsort_serial_3way_partition_` under the hood.
 */
SZ_PUBLIC void sz_sequence_argsort_serial_recursively_(                   //
    sz_pgram_t *const global_pgrams, sz_sorted_idx_t *const global_order, //
    sz_size_t const start_in_sequence, sz_size_t const end_in_sequence) {

    // Partition the collection around some pivot or 2 pivots in a 3-way partitioning
    sz_size_t first_pivot_index, last_pivot_index;
    sz_sequence_argsort_serial_3way_partition_( //
        global_pgrams, global_order,            //
        start_in_sequence, end_in_sequence,     //
        &first_pivot_index, &last_pivot_index);

    // Recursively sort the left partition
    if (start_in_sequence < first_pivot_index)
        sz_sequence_argsort_serial_recursively_(global_pgrams, global_order, start_in_sequence, first_pivot_index);

    // Recursively sort the right partition
    if (last_pivot_index + 1 < end_in_sequence)
        sz_sequence_argsort_serial_recursively_(global_pgrams, global_order, last_pivot_index + 1, end_in_sequence);
}

/**
 *  @brief Recursive Quick-Sort adaptation for strings, that processes the strings a few N-grams at a time.
 *         It combines `sz_sequence_argsort_serial_export_next_pgrams_` and `sz_sequence_argsort_serial_recursively_`,
 *         recursively diving into the identical pgrams.
 */
SZ_PUBLIC void sz_sequence_argsort_serial_next_pgrams_(                   //
    sz_sequence_t const *const sequence,                                  //
    sz_pgram_t *const global_pgrams, sz_sorted_idx_t *const global_order, //
    sz_size_t const start_in_sequence, sz_size_t const end_in_sequence,   //
    sz_size_t const start_character) {

    // Prepare the new range of pgrams
    sz_sequence_argsort_serial_export_next_pgrams_(sequence, global_pgrams, global_order, start_in_sequence,
                                                   end_in_sequence, start_character);

    // Sort current pgrams with a quicksort
    sz_sequence_argsort_serial_recursively_(global_pgrams, global_order, start_in_sequence, end_in_sequence);

    // Depending on the architecture, we will export a different number of bytes.
    // On 32-bit architectures, we will export 3 bytes, and on 64-bit architectures - 7 bytes.
    sz_size_t const pgram_capacity = sizeof(sz_pgram_t) - 1;

    // Repeat the procedure for the identical pgrams
    sz_size_t nested_start = start_in_sequence;
    sz_size_t nested_end = start_in_sequence;
    while (nested_end != end_in_sequence) {
        // Find the end of the identical pgrams
        sz_pgram_t current_pgram = global_pgrams[nested_start];
        while (nested_end != end_in_sequence && current_pgram == global_pgrams[nested_end]) ++nested_end;

        // If the identical pgrams are not trivial and each string has more characters, sort them recursively
        sz_cptr_t current_pgram_str = (sz_cptr_t)&current_pgram;
#if !SZ_IS_BIG_ENDIAN_
        sz_size_t current_pgram_length = (sz_size_t)current_pgram_str[0]; //! The byte order was swapped
#else
        sz_size_t current_pgram_length = (sz_size_t)current_pgram_str[pgram_capacity]; //! No byte swaps on big-endian
#endif
        int has_multiple_strings = nested_end - nested_start > 1;
        int has_more_characters_in_each = current_pgram_length == pgram_capacity;
        if (has_multiple_strings && has_more_characters_in_each) {
            sz_sequence_argsort_serial_next_pgrams_(sequence, global_pgrams, global_order, nested_start, nested_end,
                                                    start_character + pgram_capacity);
        }
        // Move to the next
        nested_start = nested_end;
    }
}

SZ_PUBLIC sz_status_t sz_sequence_argsort_serial(sz_sequence_t const *sequence, sz_memory_allocator_t *alloc,
                                                 sz_sorted_idx_t *order) {

    // First, initialize the `order` with `std::iota`-like behavior.
    for (sz_size_t i = 0; i != sequence->count; ++i) order[i] = i;

    // On very small collections - just use the quadratic-complexity insertion sort
    // without any smart optimizations or memory allocations.
    if (sequence->count <= 32) {
        sz_sequence_argsort_with_insertion(sequence, order);
        return sz_success_k;
    }

    // Simplify usage in higher-level libraries, where wrapping custom allocators may be troublesome.
    sz_memory_allocator_t global_alloc;
    if (!alloc) {
        sz_memory_allocator_init_default(&global_alloc);
        alloc = &global_alloc;
    }

    // One of the reasons for slow string operations is the significant overhead of branching when performing
    // individual string comparisons.
    //
    // The core idea of our algorithm is to minimize character-level loops in string comparisons and
    // instead operate on larger integer words - 4 or 8 bytes at once, on 32-bit or 64-bit architectures, respectively.
    // Let's say we have N strings and the pointer size is P.
    //
    // Our recursive algorithm will take the first P bytes of each string and sort them as integers.
    // Assuming that some strings may contain or even end with NULL bytes, we need to make sure, that their length
    // is included in those P-long words. So, in reality, we will be taking (P-1) bytes from each string on every
    // iteration of a recursive algorithm.
    sz_size_t memory_usage = sequence->count * sizeof(sz_pgram_t);
    sz_pgram_t *pgrams = (sz_pgram_t *)alloc->allocate(memory_usage, alloc);
    if (!pgrams) return sz_bad_alloc_k;

    // Recursively sort the whole sequence.
    sz_sequence_argsort_serial_next_pgrams_(sequence, pgrams, order, 0, sequence->count, 0);

    // Free temporary storage.
    alloc->free(pgrams, memory_usage, alloc);
    return sz_success_k;
}

SZ_PUBLIC sz_status_t sz_pgrams_sort_serial(sz_pgram_t *pgrams, sz_size_t count, sz_memory_allocator_t *alloc,
                                            sz_sorted_idx_t *order) {
    sz_unused_(alloc);
    // First, initialize the `order` with `std::iota`-like behavior.
    for (sz_size_t i = 0; i != count; ++i) order[i] = i;
    // Reuse the string sorting algorithm for sorting the "pgrams".
    sz_sequence_argsort_serial_recursively_((sz_pgram_t *)pgrams, order, 0, count);
    return sz_success_k;
}

/**
 *  @brief Helper function similar to `std::set_union` over pairs of integers and their original indices.
 *  @see https://en.cppreference.com/w/cpp/algorithm/set_union
 */
SZ_INTERNAL void sz_pgrams_union_serial_(                                                           //
    sz_pgram_t const *first_pgrams, sz_sorted_idx_t const *first_indices, sz_size_t first_count,    //
    sz_pgram_t const *second_pgrams, sz_sorted_idx_t const *second_indices, sz_size_t second_count, //
    sz_pgram_t *result_pgrams, sz_sorted_idx_t *result_indices) {

    // Compute the end pointers for each input array
    sz_pgram_t const *const first_end = first_pgrams + first_count;
    sz_pgram_t const *const second_end = second_pgrams + second_count;
    sz_pgram_t *const merged_begin = result_pgrams;

    // Merge until one array is exhausted
    while (first_pgrams < first_end && second_pgrams < second_end) {
        if (*first_pgrams < *second_pgrams) {
            *result_pgrams++ = *first_pgrams++;
            *result_indices++ = *first_indices++;
        }
        else if (*second_pgrams < *first_pgrams) {
            *result_pgrams++ = *second_pgrams++;
            *result_indices++ = *second_indices++;
        }
        else {
            // Equal keys: for stability, choose the one from the first array, and don't increment the second array
            *result_pgrams++ = *first_pgrams;
            *result_indices++ = *first_indices;
            ++first_pgrams;
            ++first_indices;
        }
    }

    // Copy any remaining elements from the first array
    while (first_pgrams < first_end) {
        *result_pgrams++ = *first_pgrams++;
        *result_indices++ = *first_indices++;
    }

    // Copy any remaining elements from the second array
    while (second_pgrams < second_end) {
        *result_pgrams++ = *second_pgrams++;
        *result_indices++ = *second_indices++;
    }

    // Validate the merged result.
    if (SZ_DEBUG)
        for (sz_size_t i = 1; i < first_count + second_count; ++i)
            sz_assert_(merged_begin[i - 1] <= merged_begin[i] && "The merged pgrams must be in ascending order.");
}

#ifdef __cplusplus
}
#endif

#endif // STRINGZILLA_SORT_SERIAL_H_
