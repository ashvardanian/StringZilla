/**
 *  @brief  Hardware-accelerated string collection sorting.
 *  @file   sort.h
 *  @author Ash Vardanian
 *
 *  Includes core APIs for `sz_sequence_t` string collections:
 *
 *  - `sz_sequence_argsort` - to get the sorting permutation of a string collection with QuickSort.
 *  - `sz_sequence_argsort_stable` - to get the stable-sorting permutation of a string collection with a MergeSort.
 *
 *  The core idea of all following string algorithms is to sort strings not based on 1 character at a time,
 *  but on a larger "Pointer-sized N-grams" fitting in 4 or 8 bytes at once, on 32-bit or 64-bit architectures,
 *  respectively. In reality we may not use the full pointer size, but only a few bytes from it, and keep the rest
 *  for some metadata.
 *
 *  That, however, means, that unsigned integer sorting is a constituent part of our string sorting and we can
 *  expose it as an additional set of APIs for the users:
 *
 *  - `sz_pgrams_sort` - to inplace sort continuous pointer-sized integers with QuickSort.
 *  - `sz_pgrams_sort_stable` - to inplace stable-sort continuous pointer-sized integers with a MergeSort.
 *
 *  For cases, when the input is known to be tiny, we provide quadratic-complexity insertion sort adaptations:
 *
 *  - `sz_sequence_argsort_with_insertion` - for string collections.
 *  - `sz_pgrams_sort_stable_with_insertion` - for continuous unsigned integers.
 *
 */
#ifndef STRINGZILLA_SORT_H_
#define STRINGZILLA_SORT_H_

#include "types.h"

#include "compare.h" // `sz_compare`
#include "memory.h"  // `sz_copy`

#ifdef __cplusplus
extern "C" {
#endif

#pragma region Core API

/**
 *  @brief  Faster @b arg-sort for an arbitrary @b string sequence, using QuickSort.
 *          Outputs the ::order of elements in the immutable ::sequence, that would sort it.
 *          The algorithm doesn't guarantee stability, meaning that the relative order of equal elements
 *          may not be preserved.
 *
 *  @param sequence The sequence of strings to sort.
 *  @param alloc Memory allocator for temporary storage.
 *  @param order The output - indices of the sorted sequence elements.
 *  @return Whether the operation was successful.
 */
SZ_DYNAMIC sz_bool_t sz_sequence_argsort(sz_sequence_t const *sequence, sz_memory_allocator_t *alloc,
                                         sz_sorted_idx_t *order);

/**
 *  @brief  Faster @b inplace `std::sort` for a continuous @b unsigned-integer sequence, using QuickSort.
 *          Overwrites the input ::sequence with the sorted sequence and exports the permutation ::order.
 *          The algorithm doesn't guarantee stability, meaning that the relative order of equal elements
 *          may not be preserved.
 *
 *  @param pgrams The continuous buffer of unsigned integers to sort in place.
 *  @param count The number of elements in the sequence.
 *  @param alloc Memory allocator for temporary storage.
 *  @param order The output - indices of the sorted sequence elements.
 *  @return Whether the operation was successful.
 */
SZ_DYNAMIC sz_bool_t sz_pgrams_sort(sz_pgram_t *pgrams, sz_size_t count, sz_memory_allocator_t *alloc,
                                    sz_sorted_idx_t *order);

/**
 *  @brief  Faster @b arg-sort for an arbitrary @b string sequence, using MergeSort.
 *          Outputs the ::order of elements in the immutable ::sequence, that would sort it.
 *          The algorithm guarantees stability, meaning that the relative order of equal elements is preserved.
 *
 *  This algorithm uses more memory than `sz_sequence_argsort`, but it's performance is more predictable.
 *  It's also preferred for very large inputs, as most memory access happens in a predictable sequential order.
 *
 *  @param sequence The sequence of strings to sort.
 *  @param alloc Memory allocator for temporary storage.
 *  @param order The output - indices of the sorted sequence elements.
 *  @return Whether the operation was successful.
 */
SZ_DYNAMIC sz_bool_t sz_sequence_argsort_stable(sz_sequence_t const *sequence, sz_memory_allocator_t *alloc,
                                                sz_sorted_idx_t *order);

/**
 *  @brief  Faster @b inplace `std::stable_sort sort` for a continuous @b unsigned-integer sequence, using MergeSort.
 *          Overwrites the input ::sequence with the sorted sequence and exports the permutation ::order.
 *          The algorithm guarantees stability, meaning that the relative order of equal elements is preserved.
 *
 *  This algorithm uses more memory than `sz_pgrams_sort`, but it's performance is more predictable.
 *  It's also preferred for very large inputs, as most memory access happens in a predictable sequential order.
 *
 *  @param pgrams The continuous buffer of unsigned integers to sort in place.
 *  @param count The number of elements in the sequence.
 *  @param alloc Memory allocator for temporary storage.
 *  @param order The output - indices of the sorted sequence elements.
 *  @return Whether the operation was successful.
 */
SZ_DYNAMIC sz_bool_t sz_pgrams_sort_stable(sz_pgram_t *pgrams, sz_size_t count, sz_memory_allocator_t *alloc,
                                           sz_sorted_idx_t *order);

/** @copydoc sz_sequence_argsort */
SZ_PUBLIC sz_bool_t sz_sequence_argsort_serial(sz_sequence_t const *sequence, sz_memory_allocator_t *alloc,
                                               sz_sorted_idx_t *order);

/** @copydoc sz_pgrams_sort */
SZ_PUBLIC sz_bool_t sz_pgrams_sort_serial(sz_pgram_t *pgrams, sz_size_t count, sz_memory_allocator_t *alloc,
                                          sz_sorted_idx_t *order);

/** @copydoc sz_sequence_argsort */
SZ_PUBLIC sz_bool_t sz_sequence_argsort_ice(sz_sequence_t const *sequence, sz_memory_allocator_t *alloc,
                                            sz_sorted_idx_t *order);

/** @copydoc sz_pgrams_sort */
SZ_PUBLIC sz_bool_t sz_pgrams_sort_ice(sz_pgram_t *pgrams, sz_size_t count, sz_memory_allocator_t *alloc,
                                       sz_sorted_idx_t *order);

/** @copydoc sz_sequence_argsort */
SZ_PUBLIC sz_bool_t sz_sequence_argsort_sve(sz_sequence_t const *sequence, sz_memory_allocator_t *alloc,
                                            sz_sorted_idx_t *order);

/** @copydoc sz_pgrams_sort */
SZ_PUBLIC sz_bool_t sz_pgrams_sort_sve(sz_pgram_t *pgrams, sz_size_t count, sz_memory_allocator_t *alloc,
                                       sz_sorted_idx_t *order);

/** @copydoc sz_sequence_argsort_stable */
SZ_PUBLIC sz_bool_t sz_sequence_argsort_stable_serial(sz_sequence_t const *sequence, sz_memory_allocator_t *alloc,
                                                      sz_sorted_idx_t *order);

/** @copydoc sz_pgrams_sort_stable */
SZ_PUBLIC sz_bool_t sz_pgrams_sort_stable_serial(sz_pgram_t *pgrams, sz_size_t count, sz_memory_allocator_t *alloc,
                                                 sz_sorted_idx_t *order);

/** @copydoc sz_sequence_argsort_stable */
SZ_PUBLIC sz_bool_t sz_sequence_argsort_stable_ice(sz_sequence_t const *sequence, sz_memory_allocator_t *alloc,
                                                   sz_sorted_idx_t *order);

/** @copydoc sz_pgrams_sort_stable */
SZ_PUBLIC sz_bool_t sz_pgrams_sort_stable_ice(sz_pgram_t *pgrams, sz_size_t count, sz_memory_allocator_t *alloc,
                                              sz_sorted_idx_t *order);

/** @copydoc sz_sequence_argsort_stable */
SZ_PUBLIC sz_bool_t sz_sequence_argsort_stable_sve(sz_sequence_t const *sequence, sz_memory_allocator_t *alloc,
                                                   sz_sorted_idx_t *order);

/** @copydoc sz_pgrams_sort_stable */
SZ_PUBLIC sz_bool_t sz_pgrams_sort_stable_sve(sz_pgram_t *pgrams, sz_size_t count, sz_memory_allocator_t *alloc,
                                              sz_sorted_idx_t *order);

#pragma endregion

#pragma region Generic Public Helpers

/**
 *  @brief  Quadratic complexity insertion sort adjust for our @b argsort usecase.
 *          Needs no extra memory and is used as a fallback for small inputs.
 */
SZ_PUBLIC void sz_sequence_argsort_with_insertion(sz_sequence_t const *sequence, sz_sorted_idx_t *order) {
    // Assume `order` is already initialized with 0, 1, 2, ... N.
    for (sz_size_t i = 1; i < sequence->count; ++i) {
        sz_sorted_idx_t current_idx = order[i];
        sz_size_t j = i;
        while (j > 0) {
            // Get the two strings to compare.
            sz_sorted_idx_t previous_idx = order[j - 1];
            sz_cptr_t previous_start = sequence->get_start(sequence, previous_idx);
            sz_cptr_t current_start = sequence->get_start(sequence, current_idx);
            sz_size_t previous_length = sequence->get_length(sequence, previous_idx);
            sz_size_t current_length = sequence->get_length(sequence, current_idx);

            // Use the provided sz_order to compare.
            sz_ordering_t ordering = sz_order(previous_start, previous_length, current_start, current_length);

            // If the previous string is not greater than current_idx, we're done.
            if (ordering != sz_greater_k) break;

            // Otherwise, shift the previous element to the right.
            order[j] = order[j - 1];
            --j;
        }
        order[j] = current_idx;
    }
}

/**
 *  @brief  Quadratic complexity insertion sort adjust for our @b pgram-sorting usecase.
 *          Needs no extra memory and is used as a fallback for small inputs.
 */

SZ_PUBLIC void sz_pgrams_sort_stable_with_insertion(sz_pgram_t *pgrams, sz_size_t count, sz_sorted_idx_t *order) {

    // Assume `order` is already initialized with 0, 1, 2, ... N.
    for (sz_size_t i = 1; i < count; ++i) {
        // Save the current key and corresponding index.
        sz_pgram_t current_key = pgrams[i];
        sz_sorted_idx_t current_idx = order[i];
        sz_size_t j = i;

        // Shift elements of the sorted region that are greater than the current key
        // to the right. This loop stops as soon as the correct insertion point is found.
        while (j > 0 && pgrams[j - 1] > current_key) {
            pgrams[j] = pgrams[j - 1];
            order[j] = order[j - 1];
            --j;
        }

        // Insert the current key and index into their proper location.
        pgrams[j] = current_key;
        order[j] = current_idx;
    }

#if SZ_DEBUG
    for (sz_size_t i = 1; i < count; ++i)
        _sz_assert(pgrams[i - 1] <= pgrams[i] && "The pgrams should be sorted in ascending order.");
#endif
}

#pragma endregion // Generic Public Helpers

#pragma region Generic Internal Helpers

/**
 *  @brief  Convenience macro for of conditional swap of "pgrams" and their indices for a sorting network.
 *  @see    https://en.wikipedia.org/wiki/Sorting_network
 */
#define _sz_sequence_sorting_network_conditional_swap(i, j)    \
    do {                                                       \
        if (pgrams[i] > pgrams[j]) {                           \
            _sz_swap(sz_pgram_t, pgrams[i], pgrams[j]);        \
            _sz_swap(sz_sorted_idx_t, offsets[i], offsets[j]); \
        }                                                      \
    } while (0)

/**
 *  @brief  Sorting network for 2 elements is just a single compare–swap.
 */
SZ_INTERNAL void _sz_sequence_sorting_network_2x(sz_pgram_t *pgrams, sz_sorted_idx_t *offsets) {
    _sz_sequence_sorting_network_conditional_swap(0, 1);
}

/**
 *  @brief  Sorting network for 3 elements.
 *
 *  The network uses 3 compare–swap operations:
 *
 *      Stage 1: (0, 1)
 *      Stage 2: (0, 2)
 *      Stage 3: (1, 2)
 */
SZ_INTERNAL void _sz_sequence_sorting_network_3x(sz_pgram_t *pgrams, sz_sorted_idx_t *offsets) {

    _sz_sequence_sorting_network_conditional_swap(0, 1);
    _sz_sequence_sorting_network_conditional_swap(0, 2);
    _sz_sequence_sorting_network_conditional_swap(1, 2);

#if SZ_DEBUG
    for (sz_size_t i = 1; i < 3; ++i)
        _sz_assert(pgrams[i - 1] <= pgrams[i] && "Sorting network for 3 elements failed.");
#endif
}

/**
 *  @brief  Sorting network for 4 elements.
 *
 *  The network uses 5 compare–swap operations:
 *
 *      Stage 1: (0, 1) and (2, 3)
 *      Stage 2: (0, 2)
 *      Stage 3: (1, 3)
 *      Stage 4: (1, 2)
 */
SZ_INTERNAL void _sz_sequence_sorting_network_4x(sz_pgram_t *pgrams, sz_sorted_idx_t *offsets) {

    // Stage 1: Compare–swap adjacent pairs.
    _sz_sequence_sorting_network_conditional_swap(0, 1);
    _sz_sequence_sorting_network_conditional_swap(2, 3);

    // Stage 2: Compare–swap (0, 2)
    _sz_sequence_sorting_network_conditional_swap(0, 2);

    // Stage 3: Compare–swap (1, 3)
    _sz_sequence_sorting_network_conditional_swap(1, 3);

    // Stage 4: Final compare–swap (1, 2)
    _sz_sequence_sorting_network_conditional_swap(1, 2);

#if SZ_DEBUG
    for (sz_size_t i = 1; i < 4; ++i)
        _sz_assert(pgrams[i - 1] <= pgrams[i] && "Sorting network for 4 elements failed.");
#endif
}

/**
 *  @brief  A scalar sorting network for 8 elements that reorders both the pgrams
 *          and their corresponding offsets in only 19 comparisons, the most efficient
 *          variant currently known.
 *
 *  The network consists of 6 stages with the following compare–swap pairs:
 *
 *      Stage 1: (0,1), (2,3), (4,5), (6,7)
 *      Stage 2: (0,2), (1,3), (4,6), (5,7)
 *      Stage 3: (1,2), (5,6)
 *      Stage 4: (0,4), (1,5), (2,6), (3,7)
 *      Stage 5: (2,4), (3,5)
 *      Stage 6: (1,2), (3,4), (5,6)
 */
SZ_INTERNAL void _sz_sequence_sorting_network_8x(sz_pgram_t *pgrams, sz_sorted_idx_t *offsets) {

    // Stage 1: Compare–swap adjacent pairs.
    _sz_sequence_sorting_network_conditional_swap(0, 1);
    _sz_sequence_sorting_network_conditional_swap(2, 3);
    _sz_sequence_sorting_network_conditional_swap(4, 5);
    _sz_sequence_sorting_network_conditional_swap(6, 7);

    // Stage 2: Compare–swap with stride 2.
    _sz_sequence_sorting_network_conditional_swap(0, 2);
    _sz_sequence_sorting_network_conditional_swap(1, 3);
    _sz_sequence_sorting_network_conditional_swap(4, 6);
    _sz_sequence_sorting_network_conditional_swap(5, 7);

    // Stage 3: Compare–swap between middle elements.
    _sz_sequence_sorting_network_conditional_swap(1, 2);
    _sz_sequence_sorting_network_conditional_swap(5, 6);

    // Stage 4: Compare–swap across the two halves.
    _sz_sequence_sorting_network_conditional_swap(0, 4);
    _sz_sequence_sorting_network_conditional_swap(1, 5);
    _sz_sequence_sorting_network_conditional_swap(2, 6);
    _sz_sequence_sorting_network_conditional_swap(3, 7);

    // Stage 5: Compare–swap within each half.
    _sz_sequence_sorting_network_conditional_swap(2, 4);
    _sz_sequence_sorting_network_conditional_swap(3, 5);

    // Stage 6: Final compare–swap of adjacent elements.
    _sz_sequence_sorting_network_conditional_swap(1, 2);
    _sz_sequence_sorting_network_conditional_swap(3, 4);
    _sz_sequence_sorting_network_conditional_swap(5, 6);

#if SZ_DEBUG
    // Validate the sorting network.
    for (sz_size_t i = 1; i < 8; ++i)
        _sz_assert(pgrams[i - 1] <= pgrams[i] && "The sorting network must sort the pgrams in ascending order.");
#endif
}

#undef _sz_sequence_sorting_network_conditional_swap

#pragma endregion // Generic Internal Helpers

#pragma region Serial QuickSort Implementation

SZ_INTERNAL void _sz_sequence_argsort_serial_export_next_pgrams(                //
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
            _sz_assert(partial_order_index == i && "At start this must be an identity permutation.");

        // Get the string slice in global memory.
        sz_cptr_t const source_str = sequence->get_start(sequence, partial_order_index);
        sz_size_t const length = sequence->get_length(sequence, partial_order_index);
        sz_size_t const remaining_length = length > start_character ? length - start_character : 0;
        sz_size_t const exported_length = remaining_length > pgram_capacity ? pgram_capacity : remaining_length;

        // Fill with zeros, export a slice, and mark the exported length.
        sz_pgram_t *target_pgram = &global_pgrams[i];
        sz_ptr_t target_str = (sz_ptr_t)target_pgram;
        *target_pgram = 0;
        for (sz_size_t j = 0; j < exported_length; ++j) target_str[j] = source_str[j + start_character];
        target_str[pgram_capacity] = exported_length;
#if defined(_SZ_IS_64_BIT)
        *target_pgram = sz_u64_bytes_reverse(*target_pgram);
#else
        *target_pgram = sz_u32_bytes_reverse(*target_pgram);
#endif
        _sz_assert(                                                //
            (length <= start_character) == (*target_pgram == 0) && //
            "We can have a zero value if only the string is shorter than other strings at this position.");
    }

    // As our goal is to sort the strings using the exported integer "pgrams",
    // this is a good place to validate the correctness of the exported data.
    if (SZ_DEBUG && start_character == 0)
        for (sz_size_t i = start_in_sequence + 1; i < end_in_sequence; ++i) {
            sz_pgram_t const previous_pgram = global_pgrams[i - 1];
            sz_pgram_t const current_pgram = global_pgrams[i];
            sz_cptr_t const previous_str = sequence->get_start(sequence, i - 1);
            sz_size_t const previous_length = sequence->get_length(sequence, i - 1);
            sz_cptr_t const current_str = sequence->get_start(sequence, i);
            sz_size_t const current_length = sequence->get_length(sequence, i);
            sz_ordering_t const ordering = sz_order(                                               //
                previous_str, previous_length > pgram_capacity ? pgram_capacity : previous_length, //
                current_str, current_length > pgram_capacity ? pgram_capacity : current_length);
            _sz_assert(                                                        //
                (previous_pgram < current_pgram) == (ordering == sz_less_k) && //
                "The exported pgrams should be in the same order as the original strings.");
        }
}

/**
 *  @brief  Picks the "pivot" value for the QuickSort algorithm's partitioning step using Robert Sedgewick's method,
 *          the median of three elements - the first, the middle, and the last element of the given range.
 */
SZ_INTERNAL sz_pgram_t const *_sz_sequence_partitioning_pivot(sz_pgram_t const *pgrams, sz_size_t count) {
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
 *  @brief  The most important part of the QuickSort algorithm partitioning the elements around the pivot.
 *
 *  The classical variant uses the normal 2-way partitioning, but it will scatter the values equal to the pivot
 *  into the left and right partitions. Instead we use the Dutch National Flag @b 3-way partitioning, outputting
 *  the range of values equal to the pivot.
 *
 *  @see https://en.wikipedia.org/wiki/Dutch_national_flag_problem
 */
SZ_INTERNAL void _sz_sequence_argsort_serial_3way_partition(              //
    sz_pgram_t *const global_pgrams, sz_sorted_idx_t *const global_order, //
    sz_size_t const start_in_sequence, sz_size_t const end_in_sequence,   //
    sz_size_t *first_pivot_offset, sz_size_t *last_pivot_offset) {

    // On very small inputs this procedure is rudimentary.
    sz_size_t const count = end_in_sequence - start_in_sequence;
    if (count <= 4) {
        sz_pgram_t *const pgrams = global_pgrams + start_in_sequence;
        sz_sorted_idx_t *const offsets = global_order + start_in_sequence;
        if (count == 2) { _sz_sequence_sorting_network_2x(pgrams, offsets); }
        else if (count == 3) { _sz_sequence_sorting_network_3x(pgrams, offsets); }
        else if (count == 4) { _sz_sequence_sorting_network_4x(pgrams, offsets); }
        *first_pivot_offset = start_in_sequence;
        *last_pivot_offset = end_in_sequence;
        return;
    }

    // Chose the pivot offset with Sedgewick's method.
    sz_pgram_t const pivot_pgram = *_sz_sequence_partitioning_pivot(global_pgrams + start_in_sequence, count);

    // Loop through the collection and move the elements around the pivot with the 3-way partitioning.
    sz_size_t partitioning_progress = start_in_sequence; // Current index.
    sz_size_t smaller_offset = start_in_sequence;        // Boundary for elements < `pivot_pgram`.
    sz_size_t greater_offset = end_in_sequence - 1;      // Boundary for elements > `pivot_pgram`.

    while (partitioning_progress <= greater_offset) {
        // Element is less than pivot: swap into the < pivot region.
        if (global_pgrams[partitioning_progress] < pivot_pgram) {
            _sz_swap(sz_sorted_idx_t, global_order[partitioning_progress], global_order[smaller_offset]);
            _sz_swap(sz_pgram_t, global_pgrams[partitioning_progress], global_pgrams[smaller_offset]);
            ++partitioning_progress;
            ++smaller_offset;
        }
        // Element is greater than pivot: swap into the > pivot region.
        else if (global_pgrams[partitioning_progress] > pivot_pgram) {
            _sz_swap(sz_sorted_idx_t, global_order[partitioning_progress], global_order[greater_offset]);
            _sz_swap(sz_pgram_t, global_pgrams[partitioning_progress], global_pgrams[greater_offset]);
            --greater_offset;
        }
        // Element equals `pivot_pgram`: leave it in place.
        else { ++partitioning_progress; }
    }

    *first_pivot_offset = smaller_offset;
    *last_pivot_offset = greater_offset;
}

/**
 *  @brief  Recursive Quick-Sort implementation backing both the `sz_sequence_argsort` and `sz_pgrams_sort`,
 *          and using the `_sz_sequence_argsort_serial_3way_partition` under the hood.
 */
SZ_PUBLIC void _sz_sequence_argsort_serial_recursively(                   //
    sz_pgram_t *const global_pgrams, sz_sorted_idx_t *const global_order, //
    sz_size_t const start_in_sequence, sz_size_t const end_in_sequence) {

    // Partition the collection around some pivot or 2 pivots in a 3-way partitioning
    sz_size_t first_pivot_index, last_pivot_index;
    _sz_sequence_argsort_serial_3way_partition( //
        global_pgrams, global_order,            //
        start_in_sequence, end_in_sequence,     //
        &first_pivot_index, &last_pivot_index);

    // Recursively sort the left partition
    if (start_in_sequence < first_pivot_index)
        _sz_sequence_argsort_serial_recursively(global_pgrams, global_order, start_in_sequence, first_pivot_index);

    // Recursively sort the right partition
    if (last_pivot_index + 1 < end_in_sequence)
        _sz_sequence_argsort_serial_recursively(global_pgrams, global_order, last_pivot_index + 1, end_in_sequence);
}

/**
 *  @brief  Recursive Quick-Sort adaptation for strings, that processes the strings a few N-grams at a time.
 *          It combines `_sz_sequence_argsort_serial_export_next_pgrams` and `_sz_sequence_argsort_serial_recursively`,
 *          recursively diving into the identical pgrams.
 */
SZ_PUBLIC void _sz_sequence_argsort_serial_next_pgrams(                   //
    sz_sequence_t const *const sequence,                                  //
    sz_pgram_t *const global_pgrams, sz_sorted_idx_t *const global_order, //
    sz_size_t const start_in_sequence, sz_size_t const end_in_sequence,   //
    sz_size_t const start_character) {

    // Prepare the new range of pgrams
    _sz_sequence_argsort_serial_export_next_pgrams(sequence, global_pgrams, global_order, start_in_sequence,
                                                   end_in_sequence, start_character);

    // Sort current pgrams with a quicksort
    _sz_sequence_argsort_serial_recursively(global_pgrams, global_order, start_in_sequence, end_in_sequence);

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
        sz_size_t current_pgram_length = (sz_size_t)current_pgram_str[0]; //! The byte order was swapped
        int has_multiple_strings = nested_end - nested_start > 1;
        int has_more_characters_in_each = current_pgram_length == pgram_capacity;
        if (has_multiple_strings && has_more_characters_in_each) {
            _sz_sequence_argsort_serial_next_pgrams(sequence, global_pgrams, global_order, nested_start, nested_end,
                                                    start_character + pgram_capacity);
        }
        // Move to the next
        nested_start = nested_end;
    }
}

SZ_PUBLIC sz_bool_t sz_sequence_argsort_serial(sz_sequence_t const *sequence, sz_memory_allocator_t *alloc,
                                               sz_sorted_idx_t *order) {

    // First, initialize the `order` with `std::iota`-like behavior.
    for (sz_size_t i = 0; i != sequence->count; ++i) order[i] = i;

    // On very small collections - just use the quadratic-complexity insertion sort
    // without any smart optimizations or memory allocations.
    if (sequence->count <= 32) {
        sz_sequence_argsort_with_insertion(sequence, order);
        return sz_true_k;
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
    if (!pgrams) return sz_false_k;

    // Recursively sort the whole sequence.
    _sz_sequence_argsort_serial_next_pgrams(sequence, pgrams, order, 0, sequence->count, 0);

    // Free temporary storage.
    alloc->free(pgrams, memory_usage, alloc);
    return sz_true_k;
}

SZ_PUBLIC sz_bool_t sz_pgrams_sort_serial(sz_pgram_t *pgrams, sz_size_t count, sz_memory_allocator_t *alloc,
                                          sz_sorted_idx_t *order) {
    sz_unused(alloc);
    // First, initialize the `order` with `std::iota`-like behavior.
    for (sz_size_t i = 0; i != count; ++i) order[i] = i;
    // Reuse the string sorting algorithm for sorting the "pgrams".
    _sz_sequence_argsort_serial_recursively((sz_pgram_t *)pgrams, order, 0, count);
    return sz_true_k;
}

#pragma endregion // Serial QuickSort Implementation

#pragma region Serial MergeSort Implementation

/**
 *  @brief  Helper function similar to `std::set_union` over pairs of integers and their original indices.
 *  @see    https://en.cppreference.com/w/cpp/algorithm/set_union
 */
SZ_INTERNAL void _sz_sequence_argsort_stable_serial_merge(                                          //
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
            _sz_assert(merged_begin[i - 1] <= merged_begin[i] && "The merged pgrams must be in ascending order.");
}

SZ_PUBLIC sz_bool_t sz_pgrams_sort_stable_serial(sz_pgram_t *pgrams, sz_size_t count, sz_memory_allocator_t *alloc,
                                                 sz_sorted_idx_t *order) {

    // First, initialize the `order` with `std::iota`-like behavior.
    for (sz_size_t i = 0; i != count; ++i) order[i] = i;

    // On very small collections - just use the quadratic-complexity insertion sort
    // without any smart optimizations or memory allocations.
    if (count <= 32) {
        sz_pgrams_sort_stable_with_insertion(pgrams, count, order);
        return sz_true_k;
    }

    // Go through short chunks of 8 elements and sort them with a sorting network.
    for (sz_size_t i = 0; i + 8u <= count; i += 8u) _sz_sequence_sorting_network_8x(pgrams + i, order + i);

    // For the tail of the array, sort it with insertion sort.
    sz_size_t const tail_count = count & 7u;
    sz_pgrams_sort_stable_with_insertion(pgrams + count - tail_count, tail_count, order + count - tail_count);

    // At this point, the array is partitioned into sorted runs.
    // We'll now merge these runs until the whole array is sorted.
    // Allocate temporary memory to hold merged results:
    //    - one block for keys (`sz_pgram_t`)
    //    - one block for indices (`sz_sorted_idx_t`)
    sz_size_t memory_usage = sizeof(sz_pgram_t) * count + sizeof(sz_sorted_idx_t) * count;
    sz_pgram_t *pgrams_temporary = (sz_pgram_t *)alloc->allocate(memory_usage, alloc);
    sz_sorted_idx_t *order_temporary = (sz_sorted_idx_t *)(pgrams_temporary + count);
    if (!pgrams_temporary) return sz_false_k;

    // Set initial run size (the sorted chunks).
    sz_size_t run_size = 8;

    // Pointers for current source and destination arrays.
    sz_pgram_t *src_pgrams = pgrams;
    sz_sorted_idx_t *src_order = order;
    sz_pgram_t *dst_pgrams = pgrams_temporary;
    sz_sorted_idx_t *dst_order = order_temporary;

    // Merge sorted runs in a bottom-up manner until the run size covers the whole array.
    while (run_size < count) {
        // Process adjacent runs.
        for (sz_size_t i = 0; i < count; i += run_size * 2) {
            // Determine the number of elements in the left run.
            sz_size_t left_count = run_size;
            if (i + left_count > count) { left_count = count - i; }

            // Determine the number of elements in the right run.
            sz_size_t right_count = run_size;
            if (i + left_count >= count) { right_count = 0; }
            else if (i + left_count + right_count > count) { right_count = count - (i + left_count); }

            // Merge the two runs:
            _sz_sequence_argsort_stable_serial_merge(                             //
                src_pgrams + i, src_order + i, left_count,                        //
                src_pgrams + i + run_size, src_order + i + run_size, right_count, //
                dst_pgrams + i, dst_order + i);
        }

        // Swap the roles of the source and destination arrays.
        _sz_swap(sz_pgram_t *, src_pgrams, dst_pgrams);
        _sz_swap(sz_sorted_idx_t *, src_order, dst_order);

        // Double the run size for the next pass.
        run_size *= 2;
    }

    // If the final sorted result is not in the original array, copy the sorted results back.
    if (src_pgrams != pgrams)
        for (sz_size_t i = 0; i < count; ++i) pgrams[i] = src_pgrams[i], order[i] = src_order[i];

    // Free the temporary memory used for merging.
    alloc->free(pgrams_temporary, memory_usage, alloc);
    return sz_true_k;
}

#pragma endregion // Serial MergeSort Implementation

/*  AVX512 implementation of the string search algorithms for Ice Lake and newer CPUs.
 *  Includes extensions:
 *      - 2017 Skylake: F, CD, ER, PF, VL, DQ, BW,
 *      - 2018 CannonLake: IFMA, VBMI,
 *      - 2019 Ice Lake: VPOPCNTDQ, VNNI, VBMI2, BITALG, GFNI, VPCLMULQDQ, VAES.
 *
 *  We are going to use VBMI2 for `_mm256_maskz_compress_epi8`.
 */
#pragma region Ice Lake Implementation
#if SZ_USE_ICE
#pragma GCC push_options
#pragma GCC target("avx", "avx512f", "avx512vl", "avx512bw", "avx512dq", "avx512vbmi", "avx512vbmi2", "bmi", "bmi2")
#pragma clang attribute push(                                                                          \
    __attribute__((target("avx,avx512f,avx512vl,avx512bw,avx512dq,avx512vbmi,avx512vbmi2,bmi,bmi2"))), \
    apply_to = function)

/**
 *  @brief  The most important part of the QuickSort algorithm partitioning the elements around the pivot.
 *          Unlike the serial algorithm, uses compressed stores to filter and move the elements around the pivot.
 */
SZ_INTERNAL void _sz_sequence_argsort_ice_3way_partition(                           //
    sz_pgram_t *const initial_pgrams, sz_sorted_idx_t *const initial_order,         //
    sz_pgram_t *const partitioned_pgrams, sz_sorted_idx_t *const partitioned_order, //
    sz_size_t const start_in_sequence, sz_size_t const end_in_sequence,             //
    sz_size_t *const first_pivot_offset, sz_size_t *const last_pivot_offset) {

    sz_size_t const count = end_in_sequence - start_in_sequence;
    sz_size_t const pgrams_per_register = sizeof(sz_u512_vec_t) / sizeof(sz_pgram_t);

    // Choose the pivot offset with Sedgewick's method.
    sz_pgram_t const *pivot_pgram_ptr = _sz_sequence_partitioning_pivot(initial_pgrams + start_in_sequence, count);
    sz_pgram_t const pivot_pgram = *pivot_pgram_ptr;
    sz_u512_vec_t pivot_vec;
    pivot_vec.zmm = _mm512_set1_epi64(pivot_pgram);

    // Reading data is always cheaper than writing, so we can further minimize the writes, if
    // we know exactly, how many elements are smaller or greater than the pivot.
    sz_size_t count_smaller = 0, count_greater = 0;
    sz_size_t const tail_count = count & 7u;
    __mmask8 const tail_mask = _sz_u8_mask_until(tail_count);

    sz_u512_vec_t pgrams_vec, order_vec;
    for (sz_size_t i = start_in_sequence; i + pgrams_per_register <= end_in_sequence; i += pgrams_per_register) {
        pgrams_vec.zmm = _mm512_loadu_si512(initial_pgrams + i);
        count_smaller += sz_u32_popcount(_mm512_cmplt_epu64_mask(pgrams_vec.zmm, pivot_vec.zmm));
        count_greater += sz_u32_popcount(_mm512_cmpgt_epu64_mask(pgrams_vec.zmm, pivot_vec.zmm));
    }
    if (tail_count) {
        pgrams_vec.zmm = _mm512_maskz_loadu_epi64(tail_mask, initial_pgrams + end_in_sequence - tail_count);
        count_smaller += sz_u32_popcount(_mm512_mask_cmplt_epu64_mask(tail_mask, pgrams_vec.zmm, pivot_vec.zmm));
        count_greater += sz_u32_popcount(_mm512_mask_cmpgt_epu64_mask(tail_mask, pgrams_vec.zmm, pivot_vec.zmm));
    }

    // Now all we need to do is to loop through the collection and export them into the temporary buffer
    // in 3 separate segments - smaller, equal, and greater than the pivot.
    sz_size_t const count_equal = count - count_smaller - count_greater;
    _sz_assert(count_equal >= 1 && "The pivot must be present in the collection.");
    _sz_assert(count_smaller + count_equal + count_greater == count && "The partitioning must be exhaustive.");
    sz_size_t smaller_offset = start_in_sequence;
    sz_size_t equal_offset = start_in_sequence + count_smaller;
    sz_size_t greater_offset = start_in_sequence + count_smaller + count_equal;

    // The naive algorithm - unzip the elements into 3 separate buffers.
    for (sz_size_t i = start_in_sequence; i < end_in_sequence; i += pgrams_per_register) {
        __mmask8 const load_mask = i + pgrams_per_register <= end_in_sequence ? 0xFF : tail_mask;
        pgrams_vec.zmm = _mm512_maskz_loadu_epi64(load_mask, initial_pgrams + i);
        order_vec.zmm = _mm512_maskz_loadu_epi64(load_mask, initial_order + i);

        __mmask8 const smaller_mask = _mm512_mask_cmplt_epu64_mask(load_mask, pgrams_vec.zmm, pivot_vec.zmm);
        __mmask8 const equal_mask = _mm512_mask_cmpeq_epu64_mask(load_mask, pgrams_vec.zmm, pivot_vec.zmm);
        __mmask8 const greater_mask = _mm512_mask_cmpgt_epu64_mask(load_mask, pgrams_vec.zmm, pivot_vec.zmm);

        // Compress the elements into the temporary buffer.
        _mm512_mask_compressstoreu_epi64(partitioned_pgrams + smaller_offset, smaller_mask, pgrams_vec.zmm);
        _mm512_mask_compressstoreu_epi64(partitioned_order + smaller_offset, smaller_mask, order_vec.zmm);
        smaller_offset += _mm_popcnt_u32(smaller_mask);

        _mm512_mask_compressstoreu_epi64(partitioned_pgrams + equal_offset, equal_mask, pgrams_vec.zmm);
        _mm512_mask_compressstoreu_epi64(partitioned_order + equal_offset, equal_mask, order_vec.zmm);
        equal_offset += _mm_popcnt_u32(equal_mask);

        _mm512_mask_compressstoreu_epi64(partitioned_pgrams + greater_offset, greater_mask, pgrams_vec.zmm);
        _mm512_mask_compressstoreu_epi64(partitioned_order + greater_offset, greater_mask, order_vec.zmm);
        greater_offset += _mm_popcnt_u32(greater_mask);
    }

    // Copy back.
    sz_copy_skylake((sz_ptr_t)(initial_pgrams + start_in_sequence),      //
                    (sz_cptr_t)(partitioned_pgrams + start_in_sequence), //
                    count_smaller * sizeof(sz_pgram_t));
    sz_copy_skylake((sz_ptr_t)(initial_order + start_in_sequence),      //
                    (sz_cptr_t)(partitioned_order + start_in_sequence), //
                    count_smaller * sizeof(sz_sorted_idx_t));

    sz_copy_skylake((sz_ptr_t)(initial_pgrams + start_in_sequence + count_smaller),      //
                    (sz_cptr_t)(partitioned_pgrams + start_in_sequence + count_smaller), //
                    count_equal * sizeof(sz_pgram_t));
    sz_copy_skylake((sz_ptr_t)(initial_order + start_in_sequence + count_smaller),      //
                    (sz_cptr_t)(partitioned_order + start_in_sequence + count_smaller), //
                    count_equal * sizeof(sz_sorted_idx_t));

    sz_copy_skylake((sz_ptr_t)(initial_pgrams + start_in_sequence + count_smaller + count_equal),      //
                    (sz_cptr_t)(partitioned_pgrams + start_in_sequence + count_smaller + count_equal), //
                    count_greater * sizeof(sz_pgram_t));
    sz_copy_skylake((sz_ptr_t)(initial_order + start_in_sequence + count_smaller + count_equal),      //
                    (sz_cptr_t)(partitioned_order + start_in_sequence + count_smaller + count_equal), //
                    count_greater * sizeof(sz_sorted_idx_t));

    // Return the offsets of the equal elements.
    *first_pivot_offset = start_in_sequence + count_smaller;
    *last_pivot_offset = start_in_sequence + count_smaller + count_equal - 1;
}

/**
 *  @brief  Recursive Quick-Sort implementation backing both the `sz_sequence_argsort_ice` and `sz_pgrams_sort_ice`,
 *          and using the `_sz_sequence_argsort_ice_3way_partition` under the hood.
 */
SZ_INTERNAL void _sz_sequence_argsort_ice_recursively(              //
    sz_pgram_t *initial_pgrams, sz_sorted_idx_t *initial_order,     //
    sz_pgram_t *temporary_pgrams, sz_sorted_idx_t *temporary_order, //
    sz_size_t const start_in_sequence, sz_size_t const end_in_sequence) {

    // On very small inputs, when we don't even have enough input for a single ZMM register,
    // use simple insertion sort without any extra memory.
    sz_size_t const count = end_in_sequence - start_in_sequence;
    sz_size_t const pgrams_per_register = sizeof(sz_u512_vec_t) / sizeof(sz_pgram_t);
    if (count <= pgrams_per_register) {
        sz_pgrams_sort_stable_with_insertion( //
            initial_pgrams + start_in_sequence, count, initial_order + start_in_sequence);
        return;
    }

    // Partition the collection around some pivot
    sz_size_t first_pivot_index, last_pivot_index;
    _sz_sequence_argsort_ice_3way_partition(                              //
        initial_pgrams, initial_order, temporary_pgrams, temporary_order, //
        start_in_sequence, end_in_sequence,                               //
        &first_pivot_index, &last_pivot_index);

    // Recursively sort the left and right partitions, if there are at least 2 elements in each
    if (start_in_sequence + 1 < first_pivot_index)
        _sz_sequence_argsort_ice_recursively(                                 //
            initial_pgrams, initial_order, temporary_pgrams, temporary_order, //
            start_in_sequence, first_pivot_index);
    if (last_pivot_index + 2 < end_in_sequence)
        _sz_sequence_argsort_ice_recursively(                                 //
            initial_pgrams, initial_order, temporary_pgrams, temporary_order, //
            last_pivot_index + 1, end_in_sequence);
}

SZ_PUBLIC sz_bool_t sz_pgrams_sort_ice(sz_pgram_t *pgrams, sz_size_t count, sz_memory_allocator_t *alloc,
                                       sz_sorted_idx_t *order) {

    // First, initialize the `order` with `std::iota`-like behavior.
    for (sz_size_t i = 0; i != count; ++i) order[i] = i;

    // Allocate memory for partitioning the elements around the pivot.
    sz_size_t memory_usage = sizeof(sz_pgram_t) * count + sizeof(sz_sorted_idx_t) * count;
    sz_pgram_t *temporary_pgrams = (sz_pgram_t *)alloc->allocate(memory_usage, alloc);
    sz_sorted_idx_t *temporary_order = (sz_sorted_idx_t *)(temporary_pgrams + count);
    if (!temporary_pgrams) return sz_false_k;

    // Reuse the string sorting algorithm for sorting the "pgrams".
    _sz_sequence_argsort_ice_recursively(pgrams, order, temporary_pgrams, temporary_order, 0, count);

    // Deallocate the temporary memory used for partitioning.
    alloc->free(temporary_pgrams, memory_usage, alloc);
    return sz_true_k;
}

/**
 *  @brief  Recursive Quick-Sort adaptation for strings, that processes the strings a few N-grams at a time.
 *          It combines `_sz_sequence_argsort_serial_export_next_pgrams` and `_sz_sequence_argsort_serial_recursively`,
 *          recursively diving into the identical pgrams.
 */
SZ_PUBLIC void _sz_sequence_argsort_ice_next_pgrams(                            //
    sz_sequence_t const *const sequence,                                        //
    sz_pgram_t *const global_pgrams, sz_sorted_idx_t *const global_order,       //
    sz_pgram_t *const temporary_pgrams, sz_sorted_idx_t *const temporary_order, //
    sz_size_t const start_in_sequence, sz_size_t const end_in_sequence,         //
    sz_size_t const start_character) {

    // Prepare the new range of pgrams
    _sz_sequence_argsort_serial_export_next_pgrams(sequence, global_pgrams, global_order, start_in_sequence,
                                                   end_in_sequence, start_character);

    // Sort current pgrams with a quicksort
    _sz_sequence_argsort_ice_recursively(global_pgrams, global_order, temporary_pgrams, temporary_order,
                                         start_in_sequence, end_in_sequence);

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
        sz_size_t current_pgram_length = (sz_size_t)current_pgram_str[0]; //! The byte order was swapped
        int has_multiple_strings = nested_end - nested_start > 1;
        int has_more_characters_in_each = current_pgram_length == pgram_capacity;
        if (has_multiple_strings && has_more_characters_in_each) {
            _sz_sequence_argsort_ice_next_pgrams(sequence, global_pgrams, global_order, temporary_pgrams,
                                                 temporary_order, nested_start, nested_end,
                                                 start_character + pgram_capacity);
        }
        // Move to the next
        nested_start = nested_end;
    }
}

SZ_PUBLIC sz_bool_t sz_sequence_argsort_ice(sz_sequence_t const *sequence, sz_memory_allocator_t *alloc,
                                            sz_sorted_idx_t *order) {

    // First, initialize the `order` with `std::iota`-like behavior.
    sz_size_t count = sequence->count;
    for (sz_size_t i = 0; i != count; ++i) order[i] = i;

    // On very small collections - just use the quadratic-complexity insertion sort
    // without any smart optimizations or memory allocations.
    if (count <= 32) {
        sz_sequence_argsort_with_insertion(sequence, order);
        return sz_true_k;
    }

    // Allocate memory for partitioning the elements around the pivot.
    sz_size_t memory_usage = sizeof(sz_pgram_t) * count * 2 + sizeof(sz_sorted_idx_t) * count;
    sz_pgram_t *global_pgrams = (sz_pgram_t *)alloc->allocate(memory_usage, alloc);
    sz_pgram_t *temporary_pgrams = global_pgrams + count;
    sz_sorted_idx_t *temporary_order = (sz_sorted_idx_t *)(temporary_pgrams + count);
    if (!global_pgrams) return sz_false_k;

    // Recursively sort the whole sequence.
    _sz_sequence_argsort_ice_next_pgrams(sequence, global_pgrams, order, temporary_pgrams, temporary_order, //
                                         0, count, 0);

    // Free temporary storage.
    alloc->free(global_pgrams, memory_usage, alloc);
    return sz_true_k;
}

#pragma clang attribute pop
#pragma GCC pop_options
#endif            // SZ_USE_ICE
#pragma endregion // Ice Lake Implementation

/*  Pick the right implementation for the string search algorithms.
 *  To override this behavior and precompile all backends - set `SZ_DYNAMIC_DISPATCH` to 1.
 */
#pragma region Compile Time Dispatching
#if !SZ_DYNAMIC_DISPATCH

SZ_DYNAMIC sz_bool_t sz_sequence_argsort(sz_sequence_t const *sequence, sz_memory_allocator_t *alloc,
                                         sz_sorted_idx_t *order) {
#if SZ_USE_ICE
    return sz_sequence_argsort_ice(sequence, alloc, order);
#elif SZ_USE_SVE
    return sz_sequence_argsort_sve(sequence, alloc, order);
#else
    return sz_sequence_argsort_serial(sequence, alloc, order);
#endif
}

SZ_DYNAMIC sz_bool_t sz_pgrams_sort(sz_pgram_t *pgrams, sz_size_t count, sz_memory_allocator_t *alloc,
                                    sz_sorted_idx_t *order) {
#if SZ_USE_ICE
    return sz_pgrams_sort_ice(pgrams, count, alloc, order);
#elif SZ_USE_SVE
    return sz_pgrams_sort_sve(pgrams, count, alloc, order);
#else
    return sz_pgrams_sort_serial(pgrams, count, alloc, order);
#endif
}

SZ_DYNAMIC sz_bool_t sz_sequence_argsort_stable(sz_sequence_t const *sequence, sz_memory_allocator_t *alloc,
                                                sz_sorted_idx_t *order) {
#if SZ_USE_ICE
    return sz_sequence_argsort_ice(sequence, alloc, order);
#elif SZ_USE_SVE
    return sz_sequence_argsort_sve(sequence, alloc, order);
#else
    return sz_sequence_argsort_serial(sequence, alloc, order);
#endif
}

SZ_DYNAMIC sz_bool_t sz_pgrams_sort_stable(sz_pgram_t *pgrams, sz_size_t count, sz_memory_allocator_t *alloc,
                                           sz_sorted_idx_t *order) {
#if SZ_USE_ICE
    return sz_pgrams_sort_ice(pgrams, count, alloc, order);
#elif SZ_USE_SVE
    return sz_pgrams_sort_sve(pgrams, count, alloc, order);
#else
    return sz_pgrams_sort_serial(pgrams, count, alloc, order);
#endif
}

#endif            // !SZ_DYNAMIC_DISPATCH
#pragma endregion // Compile Time Dispatching

#ifdef __cplusplus
}
#endif // __cplusplus
#endif // STRINGZILLA_SORT_H_
