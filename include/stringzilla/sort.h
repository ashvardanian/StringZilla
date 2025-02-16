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
 */
#ifndef STRINGZILLA_SORT_H_
#define STRINGZILLA_SORT_H_

#include "types.h"

#include "compare.h" // `sz_compare`

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

#pragma region Serial QuickSort Implementation

SZ_PUBLIC void _sz_sequence_argsort_serial_export_next_pgrams(                  //
    sz_sequence_t const *const sequence,                                        //
    sz_pgram_t *const global_pgrams, sz_sorted_idx_t const *const global_order, //
    sz_size_t const start_in_sequence, sz_size_t const end_in_sequence,         //
    sz_size_t const start_character) {

    // Depending on the architecture, we will export a different number of bytes.
    // On 32-bit architectures, we will export 3 bytes, and on 64-bit architectures - 7 bytes.
    sz_size_t const window_capacity = sizeof(sz_pgram_t) - 1;

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
        sz_size_t const exported_length = remaining_length > window_capacity ? window_capacity : remaining_length;

        // Fill with zeros, export a slice, and mark the exported length.
        sz_pgram_t *target_pgram = &global_pgrams[i];
        sz_ptr_t target_str = (sz_ptr_t)target_pgram;
        *target_pgram = 0;
        for (sz_size_t j = 0; j < exported_length; ++j) target_str[j] = source_str[j + start_character];
        target_str[window_capacity] = exported_length;
#if defined(_SZ_IS_64_BIT)
        *target_pgram = sz_u64_bytes_reverse(*target_pgram);
#else
        *target_pgram = sz_u32_bytes_reverse(*target_pgram);
#endif
        _sz_assert(                                                //
            (length <= start_character) == (*target_pgram == 0) && //
            "We can have a zero value if only the string is shorter than other strings at this position.");
    }

    // As our goal is to sort the strings using the exported integer "windows",
    // this is a good place to validate the correctness of the exported data.
    if (SZ_DEBUG && start_character == 0)
        for (sz_size_t i = start_in_sequence + 1; i < end_in_sequence; ++i) {
            sz_pgram_t const previous_window = global_pgrams[i - 1];
            sz_pgram_t const current_window = global_pgrams[i];
            sz_cptr_t const previous_str = sequence->get_start(sequence, i - 1);
            sz_size_t const previous_length = sequence->get_length(sequence, i - 1);
            sz_cptr_t const current_str = sequence->get_start(sequence, i);
            sz_size_t const current_length = sequence->get_length(sequence, i);
            sz_ordering_t const ordering = sz_order(                                                 //
                previous_str, previous_length > window_capacity ? window_capacity : previous_length, //
                current_str, current_length > window_capacity ? window_capacity : current_length);
            _sz_assert(                                                          //
                (previous_window < current_window) == (ordering == sz_less_k) && //
                "The exported windows should be in the same order as the original strings.");
        }
}

/**
 *  @brief  The most important part of the QuickSort algorithm, that rearranges the elements in
 *          such a way, that all entries around the pivot are less than the pivot.
 *
 *  It means that no relative order among the elements on the left or right side of the pivot is preserved.
 *  We chose the pivot point using Robert Sedgewick's method - the median of three elements - the first,
 *  the middle, and the last element of the given range.
 *
 *  Moreover, considering our iterative refinement procedure, we can't just use the normal 2-way partitioning,
 *  as it will scatter the values equal to the pivot into the left and right partitions. Instead we use the
 *  Dutch National Flag @b 3-way partitioning, outputting the range of values equal to the pivot.
 *
 *  @see https://en.wikipedia.org/wiki/Dutch_national_flag_problem
 */
SZ_PUBLIC void _sz_sequence_argsort_serial_3way_partition(                //
    sz_pgram_t *const global_pgrams, sz_sorted_idx_t *const global_order, //
    sz_size_t const start_in_sequence, sz_size_t const end_in_sequence,   //
    sz_size_t *first_pivot_offset, sz_size_t *last_pivot_offset) {

    // Chose the pivot offset with Sedgewick's method.
    sz_pgram_t pivot_window;
    {
        sz_size_t const middle_offset = start_in_sequence + (end_in_sequence - start_in_sequence) / 2;
        sz_size_t const last_offset = end_in_sequence - 1;
        sz_size_t const first_offset = start_in_sequence;
        sz_pgram_t const first_window = global_pgrams[first_offset];
        sz_pgram_t const middle_window = global_pgrams[middle_offset];
        sz_pgram_t const last_window = global_pgrams[last_offset];
        if (first_window < middle_window) {
            if (middle_window < last_window) { pivot_window = middle_window; }
            else if (first_window < last_window) { pivot_window = last_window; }
            else { pivot_window = first_window; }
        }
        else {
            if (first_window < last_window) { pivot_window = first_window; }
            else if (middle_window < last_window) { pivot_window = last_window; }
            else { pivot_window = middle_window; }
        }
    }

    // Loop through the collection and move the elements around the pivot with the 3-way partitioning.
    sz_size_t partitioning_progress = start_in_sequence; // Current index.
    sz_size_t smaller_offset = start_in_sequence;        // Boundary for elements < pivot_window.
    sz_size_t greater_offset = end_in_sequence - 1;      // Boundary for elements > pivot_window.

    while (partitioning_progress <= greater_offset) {
        // Element is less than pivot: swap into the < pivot region.
        if (global_pgrams[partitioning_progress] < pivot_window) {
            _sz_swap(sz_sorted_idx_t, global_order[partitioning_progress], global_order[smaller_offset]);
            _sz_swap(sz_pgram_t, global_pgrams[partitioning_progress], global_pgrams[smaller_offset]);
            ++partitioning_progress;
            ++smaller_offset;
        }
        // Element is greater than pivot: swap into the > pivot region.
        else if (global_pgrams[partitioning_progress] > pivot_window) {
            _sz_swap(sz_sorted_idx_t, global_order[partitioning_progress], global_order[greater_offset]);
            _sz_swap(sz_pgram_t, global_pgrams[partitioning_progress], global_pgrams[greater_offset]);
            --greater_offset;
        }
        // Element equals pivot_window: leave it in place.
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
 *          recursively diving into the identical windows.
 */
SZ_PUBLIC void _sz_sequence_argsort_serial_next_pgrams(                   //
    sz_sequence_t const *const sequence,                                  //
    sz_pgram_t *const global_pgrams, sz_sorted_idx_t *const global_order, //
    sz_size_t const start_in_sequence, sz_size_t const end_in_sequence,   //
    sz_size_t const start_character) {

    // Prepare the new range of windows
    _sz_sequence_argsort_serial_export_next_pgrams(sequence, global_pgrams, global_order, start_in_sequence,
                                                   end_in_sequence, start_character);

    // Sort current windows with a quicksort
    _sz_sequence_argsort_serial_recursively(global_pgrams, global_order, start_in_sequence, end_in_sequence);

    // Depending on the architecture, we will export a different number of bytes.
    // On 32-bit architectures, we will export 3 bytes, and on 64-bit architectures - 7 bytes.
    sz_size_t const window_capacity = sizeof(sz_pgram_t) - 1;

    // Repeat the procedure for the identical windows
    sz_size_t nested_start = start_in_sequence;
    sz_size_t nested_end = start_in_sequence;
    while (nested_end != end_in_sequence) {
        // Find the end of the identical windows
        sz_pgram_t current_window_integer = global_pgrams[nested_start];
        while (nested_end != end_in_sequence && current_window_integer == global_pgrams[nested_end]) ++nested_end;

        // If the identical windows are not trivial and each string has more characters, sort them recursively
        sz_cptr_t current_window_str = (sz_cptr_t)&current_window_integer;
        sz_size_t current_window_length = (sz_size_t)current_window_str[0]; //! The byte order was swapped
        int has_multiple_strings = nested_end - nested_start > 1;
        int has_more_characters_in_each = current_window_length == window_capacity;
        if (has_multiple_strings && has_more_characters_in_each) {
            _sz_sequence_argsort_serial_next_pgrams(sequence, global_pgrams, global_order, nested_start, nested_end,
                                                    start_character + window_capacity);
        }
        // Move to the next
        nested_start = nested_end;
    }
}

SZ_PUBLIC void _sz_sequence_argsort_serial_insertion(sz_sequence_t const *sequence, sz_memory_allocator_t *alloc,
                                                     sz_sorted_idx_t *order) {
    // This algorithm needs no memory allocations:
    sz_unused(alloc);

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

SZ_PUBLIC sz_bool_t sz_sequence_argsort_serial(sz_sequence_t const *sequence, sz_memory_allocator_t *alloc,
                                               sz_sorted_idx_t *order) {

    // First, initialize the `order` with `std::iota`-like behavior.
    for (sz_size_t i = 0; i != sequence->count; ++i) order[i] = i;

    // On very small collections - just use the quadratic-complexity insertion sort
    // without any smart optimizations or memory allocations.
    if (sequence->count <= 32) {
        _sz_sequence_argsort_serial_insertion(sequence, alloc, order);
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
    sz_pgram_t *windows = (sz_pgram_t *)alloc->allocate(memory_usage, alloc);
    if (!windows) return sz_false_k;

    // Recursively sort the whole sequence.
    _sz_sequence_argsort_serial_next_pgrams(sequence, windows, order, 0, sequence->count, 0);

    // Free temporary storage.
    alloc->free(windows, memory_usage, alloc);
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
 *  @brief  A scalar sorting network for 8 elements that reorders both the keys
 *          and their corresponding offsets in only 19 comparisons, the most efficient
 *          variant currently known.
 *  @see    https://en.wikipedia.org/wiki/Sorting_network
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
void _sz_sequence_argsort_stable_serial_8x_network(sz_pgram_t *keys, sz_sorted_idx_t *offsets) {

#define _sz_sequence_argsort_stable_8x_conditional_swap(i, j)  \
    do {                                                       \
        if (keys[i] > keys[j]) {                               \
            _sz_swap(sz_pgram_t, keys[i], keys[j]);            \
            _sz_swap(sz_sorted_idx_t, offsets[i], offsets[j]); \
        }                                                      \
    } while (0)

    // Stage 1: Compare–swap adjacent pairs.
    _sz_sequence_argsort_stable_8x_conditional_swap(0, 1);
    _sz_sequence_argsort_stable_8x_conditional_swap(2, 3);
    _sz_sequence_argsort_stable_8x_conditional_swap(4, 5);
    _sz_sequence_argsort_stable_8x_conditional_swap(6, 7);

    // Stage 2: Compare–swap with stride 2.
    _sz_sequence_argsort_stable_8x_conditional_swap(0, 2);
    _sz_sequence_argsort_stable_8x_conditional_swap(1, 3);
    _sz_sequence_argsort_stable_8x_conditional_swap(4, 6);
    _sz_sequence_argsort_stable_8x_conditional_swap(5, 7);

    // Stage 3: Compare–swap between middle elements.
    _sz_sequence_argsort_stable_8x_conditional_swap(1, 2);
    _sz_sequence_argsort_stable_8x_conditional_swap(5, 6);

    // Stage 4: Compare–swap across the two halves.
    _sz_sequence_argsort_stable_8x_conditional_swap(0, 4);
    _sz_sequence_argsort_stable_8x_conditional_swap(1, 5);
    _sz_sequence_argsort_stable_8x_conditional_swap(2, 6);
    _sz_sequence_argsort_stable_8x_conditional_swap(3, 7);

    // Stage 5: Compare–swap within each half.
    _sz_sequence_argsort_stable_8x_conditional_swap(2, 4);
    _sz_sequence_argsort_stable_8x_conditional_swap(3, 5);

    // Stage 6: Final compare–swap of adjacent elements.
    _sz_sequence_argsort_stable_8x_conditional_swap(1, 2);
    _sz_sequence_argsort_stable_8x_conditional_swap(3, 4);
    _sz_sequence_argsort_stable_8x_conditional_swap(5, 6);

#undef _sz_sequence_argsort_stable_8x_conditional_swap

    // Validate the sorting network.
    if (SZ_DEBUG)
        for (sz_size_t i = 1; i < 8; ++i)
            _sz_assert(keys[i - 1] <= keys[i] && "The sorting network must sort the keys in ascending order.");
}

/**
 *  @brief  Helper function similar to `std::set_union` over pairs of integers and their original indices.
 *  @see    https://en.cppreference.com/w/cpp/algorithm/set_union
 */
void _sz_sequence_argsort_stable_serial_merge(                                                      //
    sz_pgram_t const *first_pgrams, sz_sorted_idx_t const *first_indices, sz_size_t first_count,    //
    sz_pgram_t const *second_pgrams, sz_sorted_idx_t const *second_indices, sz_size_t second_count, //
    sz_pgram_t *result_pgrams, sz_sorted_idx_t *result_indices) {

    // Compute the end pointers for each input array
    sz_pgram_t const *const first_end = first_pgrams + first_count;
    sz_pgram_t const *const second_end = second_pgrams + second_count;

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
            // Equal keys: for stability, choose the one from the first array
            *result_pgrams++ = *first_pgrams;
            *result_indices++ = *first_indices;
            ++first_pgrams;
            ++first_indices;
            ++second_pgrams;
            ++second_indices;
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
}

SZ_PUBLIC sz_bool_t sz_pgrams_sort_stable_serial(sz_pgram_t *pgrams, sz_size_t count, sz_memory_allocator_t *alloc,
                                                 sz_sorted_idx_t *order) {

    // First, initialize the `order` with `std::iota`-like behavior.
    for (sz_size_t i = 0; i != count; ++i) order[i] = i;

    // Go through short chunks of 8 elements and sort them with a sorting network.
    for (sz_size_t i = 0; i + 8 <= count; i += 8) _sz_sequence_argsort_stable_serial_8x_network(pgrams + i, order + i);

    // For the tail of the array, sort it with insertion sort.
    for (sz_size_t i = count & ~7; i < count; i++) {
        sz_pgram_t current_address = pgrams[i];
        sz_sorted_idx_t current_idx = order[i];
        sz_size_t j = i;
        while (j > 0 && pgrams[j - 1] > current_address) {
            pgrams[j] = pgrams[j - 1];
            order[j] = order[j - 1];
            --j;
        }
        pgrams[j] = current_address;
        order[j] = current_idx;
    }

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
            if (i + run_size >= count) { right_count = 0; }
            else if (i + run_size + right_count > count) { right_count = count - (i + run_size); }

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

#pragma region Ice Lake Implementation

SZ_PUBLIC void _sz_sequence_argsort_ice_recursively(                    //
    sz_sequence_t const *const collection,                              //
    sz_pgram_t *const global_pgrams, sz_size_t *const global_order,     //
    sz_size_t const start_in_sequence, sz_size_t const end_in_sequence, //
    sz_size_t const start_character) {

    // Prepare the new range of windows
    _sz_sequence_argsort_serial_export_next_pgrams(collection, global_pgrams, global_order, start_in_sequence,
                                                   end_in_sequence, start_character);

    // We can implement a form of a Radix sort here, that will count the number of elements with
    // a certain bit set. The naive approach may require too many loops over data. A more "vectorized"
    // approach would be to maintain a histogram for several bits at once. For 4 bits we will
    // need 2^4 = 16 counters.
    sz_size_t histogram[16] = {0};
    for (sz_size_t byte_in_window = 0; byte_in_window != sizeof(sz_pgram_t); ++byte_in_window) {
        // First sort based on the low nibble of each byte.
        for (sz_size_t i = start_in_sequence; i < end_in_sequence; ++i) {
            sz_size_t const byte = (global_pgrams[i] >> (byte_in_window * 8)) & 0xFF;
            ++histogram[byte];
        }
        sz_size_t offset = start_in_sequence;
        for (sz_size_t i = 0; i != 16; ++i) {
            sz_size_t const count = histogram[i];
            histogram[i] = offset;
            offset += count;
        }
        for (sz_size_t i = start_in_sequence; i < end_in_sequence; ++i) {
            sz_size_t const byte = (global_pgrams[i] >> (byte_in_window * 8)) & 0xFF;
            global_order[histogram[byte]] = i;
            ++histogram[byte];
        }
    }
}

#pragma endregion // Ice Lake Implementation

/*  Pick the right implementation for the string search algorithms.
 *  To override this behavior and precompile all backends - set `SZ_DYNAMIC_DISPATCH` to 1.
 */
#pragma region Compile Time Dispatching
#if !SZ_DYNAMIC_DISPATCH

SZ_DYNAMIC sz_bool_t sz_sequence_argsort(sz_sequence_t const *sequence, sz_memory_allocator_t *alloc,
                                         sz_sorted_idx_t *order) {
    return sz_sequence_argsort_serial(sequence, alloc, order);
}

#endif            // !SZ_DYNAMIC_DISPATCH
#pragma endregion // Compile Time Dispatching

#ifdef __cplusplus
}
#endif // __cplusplus
#endif // STRINGZILLA_SORT_H_
