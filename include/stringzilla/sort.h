/**
 *  @brief  Hardware-accelerated string collection sorting.
 *  @file   sort.h
 *  @author Ash Vardanian
 *
 *  Provides the @b `sz_sequence_argsort` API to get the sorting permutation of `sz_sequence_t` binary
 *  string collections in lexicographical order.
 *
 *  The core idea of all following string algorithms is to process strings not based on 1 character at a time,
 *  but on a larger "Pointer-sized N-grams" fitting in 4 or 8 bytes at once, on 32-bit or 64-bit architectures,
 *  respectively. In reality we may not use the full pointer size, but only a few bytes from it, and keep the
 *  rest for some metadata.
 *
 *  That, however, means, that unsigned integer sorting & matching is a constituent part of our sequence
 *  algorithms and we can expose them as an additional APIs for the users:
 *
 *  - `sz_pgrams_sort` - to inplace sort continuous pointer-sized integers.
 *  - `sz_pgrams_join` - to compute the intersection of two arbitrary integer collections.
 *
 *  Other helpers include:
 *
 *  - `sz_pgrams_sort_with_insertion` - for quadratic-complexity sorting of small continuous integer arrays.
 *  - `sz_sequence_argsort_with_insertion` - for quadratic-complexity sorting of small string collections.
 *  - `sz_sequence_argsort_stabilize` - updates the sorting permutation to be stable.
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
 *          Outputs the @p order of elements in the immutable @p sequence, that would sort it.
 *
 *  @param[in] sequence Immutable sequence of strings to sort.
 *  @param[in] alloc Optional memory allocator for temporary storage.
 *  @param[out] order Output permutation that sorts the elements.
 *
 *  @retval `sz_success_k` if the operation was successful.
 *  @retval `sz_bad_alloc_k` if the operation failed due to memory allocation failure.
 *  @pre The @p order array must fit at least `sequence->count` integers.
 *  @post The @p order array will contain a valid permutation of `[0, sequence->count - 1]`.
 *
 *  Example usage:
 *
 *  @code{.c}
 *      #include <stringzilla/sort.h>
 *      int main() {
 *          char const *strings[] = {"banana", "apple", "cherry"};
 *          sz_sequence_t sequence;
 *          sz_sequence_from_null_terminated_strings(strings, 3, &sequence);
 *          sz_sorted_idx_t order[3];
 *          sz_status_t status = sz_sequence_argsort(&sequence, NULL, order);
 *          return status == sz_success_k && order[0] == 1 && order[1] == 0 && order[2] == 2 ? 0 : 1;
 *      }
 *  @endcode
 *
 *  @note   The algorithm has linear memory complexity, quadratic worst-case and log-linear average time complexity.
 *  @see    https://en.wikipedia.org/wiki/Quicksort
 *
 *  @note   This algorithm is @b unstable: equal elements may change relative order.
 *  @sa     sz_sequence_argsort_stabilize
 *
 *  @note   Selects the fastest implementation at compile- or run-time based on `SZ_DYNAMIC_DISPATCH`.
 *  @sa     sz_sequence_argsort_serial, sz_sequence_argsort_skylake, sz_sequence_argsort_sve
 */
SZ_DYNAMIC sz_status_t sz_sequence_argsort(sz_sequence_t const *sequence, sz_memory_allocator_t *alloc,
                                           sz_sorted_idx_t *order);

/**
 *  @brief  Faster @b inplace `std::sort` for a continuous @b unsigned-integer sequence, using QuickSort.
 *          Overwrites the input @p pgrams with the sorted sequence and exports the @p order permutation.
 *
 *  @param[inout] pgrams Continuous buffer of unsigned integers to sort in place.
 *  @param[in] count Number of elements in the sequence.
 *  @param[in] alloc Optional memory allocator for temporary storage.
 *  @param[out] order Output permutation that sorts the elements.
 *
 *  @retval `sz_success_k` if the operation was successful.
 *  @retval `sz_bad_alloc_k` if the operation failed due to memory allocation failure.
 *  @pre The @p order array must fit at least `count` integers.
 *  @post The @p order array will contain a valid permutation of `[0, count - 1]`.
 *
 *  Example usage:
 *
 *  @code{.c}
 *      #include <stringzilla/sort.h>
 *      int main() {
 *          sz_pgram_t pgrams[] = {42, 17, 99, 8};
 *          sz_sorted_idx_t order[4];
 *          sz_status_t status = sz_pgrams_sort(pgrams, 4, NULL, order);
 *          return status == sz_success_k && order[0] == 3 && order[1] == 1 && order[2] == 0 && order[3] == 2 ? 0 : 1;
 *      }
 *  @endcode
 *
 *  @note   The algorithm has linear memory complexity, quadratic worst-case and log-linear average time complexity.
 *  @see    https://en.wikipedia.org/wiki/Quicksort
 *
 *  @note   Selects the fastest implementation at compile- or run-time based on `SZ_DYNAMIC_DISPATCH`.
 *  @sa     sz_pgrams_sort_serial, sz_pgrams_sort_skylake, sz_pgrams_sort_sve
 */
SZ_DYNAMIC sz_status_t sz_pgrams_sort(sz_pgram_t *pgrams, sz_size_t count, sz_memory_allocator_t *alloc,
                                      sz_sorted_idx_t *order);

/** @copydoc sz_sequence_argsort */
SZ_PUBLIC sz_status_t sz_sequence_argsort_serial(sz_sequence_t const *sequence, sz_memory_allocator_t *alloc,
                                                 sz_sorted_idx_t *order);

/** @copydoc sz_pgrams_sort */
SZ_PUBLIC sz_status_t sz_pgrams_sort_serial(sz_pgram_t *pgrams, sz_size_t count, sz_memory_allocator_t *alloc,
                                            sz_sorted_idx_t *order);

#if SZ_USE_SKYLAKE

/** @copydoc sz_sequence_argsort */
SZ_PUBLIC sz_status_t sz_sequence_argsort_skylake(sz_sequence_t const *sequence, sz_memory_allocator_t *alloc,
                                                  sz_sorted_idx_t *order);

/** @copydoc sz_pgrams_sort */
SZ_PUBLIC sz_status_t sz_pgrams_sort_skylake(sz_pgram_t *pgrams, sz_size_t count, sz_memory_allocator_t *alloc,
                                             sz_sorted_idx_t *order);

#endif

#if SZ_USE_SVE

/** @copydoc sz_sequence_argsort */
SZ_PUBLIC sz_status_t sz_sequence_argsort_sve(sz_sequence_t const *sequence, sz_memory_allocator_t *alloc,
                                              sz_sorted_idx_t *order);

/** @copydoc sz_pgrams_sort */
SZ_PUBLIC sz_status_t sz_pgrams_sort_sve(sz_pgram_t *pgrams, sz_size_t count, sz_memory_allocator_t *alloc,
                                         sz_sorted_idx_t *order);

#endif

#pragma endregion

#pragma region Generic Public Helpers

/**
 *  @brief  Quadratic complexity @b stable insertion sort adjust for our @b argsort usecase.
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
            sz_cptr_t previous_start = sequence->get_start(sequence->handle, previous_idx);
            sz_cptr_t current_start = sequence->get_start(sequence->handle, current_idx);
            sz_size_t previous_length = sequence->get_length(sequence->handle, previous_idx);
            sz_size_t current_length = sequence->get_length(sequence->handle, current_idx);

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
 *  @brief  Quadratic complexity @b stable insertion sort adjust for our @b pgram-sorting usecase.
 *          Needs no extra memory and is used as a fallback for small inputs.
 */

SZ_PUBLIC void sz_pgrams_sort_with_insertion(sz_pgram_t *pgrams, sz_size_t count, sz_sorted_idx_t *order) {

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
        sz_assert_(pgrams[i - 1] <= pgrams[i] && "The pgrams should be sorted in ascending order.");
#endif
}

#pragma endregion // Generic Public Helpers

#pragma region Generic Internal Helpers

/**
 *  @brief  Convenience macro for of conditional swap of "pgrams" and their indices for a sorting network.
 *  @see    https://en.wikipedia.org/wiki/Sorting_network
 */
#define sz_sequence_sorting_network_conditional_swap_(i, j)    \
    do {                                                       \
        if (pgrams[i] > pgrams[j]) {                           \
            sz_swap_(sz_pgram_t, pgrams[i], pgrams[j]);        \
            sz_swap_(sz_sorted_idx_t, offsets[i], offsets[j]); \
        }                                                      \
    } while (0)

/**
 *  @brief  Sorting network for 2 elements is just a single compare-swap.
 */
SZ_INTERNAL void sz_sequence_sorting_network_2x_(sz_pgram_t *pgrams, sz_sorted_idx_t *offsets) {
    sz_sequence_sorting_network_conditional_swap_(0, 1);
}

/**
 *  @brief  Sorting network for 3 elements.
 *
 *  The network uses 3 compare-swap operations:
 *
 *      Stage 1: (0, 1)
 *      Stage 2: (0, 2)
 *      Stage 3: (1, 2)
 */
SZ_INTERNAL void sz_sequence_sorting_network_3x_(sz_pgram_t *pgrams, sz_sorted_idx_t *offsets) {

    sz_sequence_sorting_network_conditional_swap_(0, 1);
    sz_sequence_sorting_network_conditional_swap_(0, 2);
    sz_sequence_sorting_network_conditional_swap_(1, 2);

#if SZ_DEBUG
    for (sz_size_t i = 1; i < 3; ++i)
        sz_assert_(pgrams[i - 1] <= pgrams[i] && "Sorting network for 3 elements failed.");
#endif
}

/**
 *  @brief  Sorting network for 4 elements.
 *
 *  The network uses 5 compare-swap operations:
 *
 *      Stage 1: (0, 1) and (2, 3)
 *      Stage 2: (0, 2)
 *      Stage 3: (1, 3)
 *      Stage 4: (1, 2)
 */
SZ_INTERNAL void sz_sequence_sorting_network_4x_(sz_pgram_t *pgrams, sz_sorted_idx_t *offsets) {

    // Stage 1: Compare-swap adjacent pairs.
    sz_sequence_sorting_network_conditional_swap_(0, 1);
    sz_sequence_sorting_network_conditional_swap_(2, 3);

    // Stage 2: Compare-swap (0, 2)
    sz_sequence_sorting_network_conditional_swap_(0, 2);

    // Stage 3: Compare-swap (1, 3)
    sz_sequence_sorting_network_conditional_swap_(1, 3);

    // Stage 4: Final compare-swap (1, 2)
    sz_sequence_sorting_network_conditional_swap_(1, 2);

#if SZ_DEBUG
    for (sz_size_t i = 1; i < 4; ++i)
        sz_assert_(pgrams[i - 1] <= pgrams[i] && "Sorting network for 4 elements failed.");
#endif
}

/**
 *  @brief  A scalar sorting network for 8 elements that reorders both the pgrams
 *          and their corresponding offsets in only 19 comparisons, the most efficient
 *          variant currently known.
 *
 *  The network consists of 6 stages with the following compare-swap pairs:
 *
 *      Stage 1: (0,1), (2,3), (4,5), (6,7)
 *      Stage 2: (0,2), (1,3), (4,6), (5,7)
 *      Stage 3: (1,2), (5,6)
 *      Stage 4: (0,4), (1,5), (2,6), (3,7)
 *      Stage 5: (2,4), (3,5)
 *      Stage 6: (1,2), (3,4), (5,6)
 */
SZ_INTERNAL void sz_sequence_sorting_network_8x_(sz_pgram_t *pgrams, sz_sorted_idx_t *offsets) {

    // Stage 1: Compare-swap adjacent pairs.
    sz_sequence_sorting_network_conditional_swap_(0, 1);
    sz_sequence_sorting_network_conditional_swap_(2, 3);
    sz_sequence_sorting_network_conditional_swap_(4, 5);
    sz_sequence_sorting_network_conditional_swap_(6, 7);

    // Stage 2: Compare-swap with stride 2.
    sz_sequence_sorting_network_conditional_swap_(0, 2);
    sz_sequence_sorting_network_conditional_swap_(1, 3);
    sz_sequence_sorting_network_conditional_swap_(4, 6);
    sz_sequence_sorting_network_conditional_swap_(5, 7);

    // Stage 3: Compare-swap between middle elements.
    sz_sequence_sorting_network_conditional_swap_(1, 2);
    sz_sequence_sorting_network_conditional_swap_(5, 6);

    // Stage 4: Compare-swap across the two halves.
    sz_sequence_sorting_network_conditional_swap_(0, 4);
    sz_sequence_sorting_network_conditional_swap_(1, 5);
    sz_sequence_sorting_network_conditional_swap_(2, 6);
    sz_sequence_sorting_network_conditional_swap_(3, 7);

    // Stage 5: Compare-swap within each half.
    sz_sequence_sorting_network_conditional_swap_(2, 4);
    sz_sequence_sorting_network_conditional_swap_(3, 5);

    // Stage 6: Final compare-swap of adjacent elements.
    sz_sequence_sorting_network_conditional_swap_(1, 2);
    sz_sequence_sorting_network_conditional_swap_(3, 4);
    sz_sequence_sorting_network_conditional_swap_(5, 6);

#if SZ_DEBUG
    // Validate the sorting network.
    for (sz_size_t i = 1; i < 8; ++i)
        sz_assert_(pgrams[i - 1] <= pgrams[i] && "The sorting network must sort the pgrams in ascending order.");
#endif
}

#undef sz_sequence_sorting_network_conditional_swap_

#pragma endregion // Generic Internal Helpers

#pragma region Serial QuickSort Implementation

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
 *  @brief  Picks the "pivot" value for the QuickSort algorithm's partitioning step using Robert Sedgewick's method,
 *          the median of three elements - the first, the middle, and the last element of the given range.
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
 *  @brief  The most important part of the QuickSort algorithm partitioning the elements around the pivot.
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
 *  @brief  Recursive Quick-Sort implementation backing both the `sz_sequence_argsort` and `sz_pgrams_sort`,
 *          and using the `sz_sequence_argsort_serial_3way_partition_` under the hood.
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
 *  @brief  Recursive Quick-Sort adaptation for strings, that processes the strings a few N-grams at a time.
 *          It combines `sz_sequence_argsort_serial_export_next_pgrams_` and `sz_sequence_argsort_serial_recursively_`,
 *          recursively diving into the identical pgrams.
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

#pragma endregion // Serial QuickSort Implementation

#pragma region Serial MergeSort Implementation

/**
 *  @brief  Helper function similar to `std::set_union` over pairs of integers and their original indices.
 *  @see    https://en.cppreference.com/w/cpp/algorithm/set_union
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

#pragma endregion // Serial MergeSort Implementation

/*  AVX512 implementation of the string search algorithms for Ice Lake and newer CPUs.
 *  Includes extensions:
 *      - 2017 Skylake: F, CD, ER, PF, VL, DQ, BW,
 *      - 2018 CannonLake: IFMA, VBMI,
 *      - 2019 Ice Lake: VPOPCNTDQ, VNNI, VBMI2, BITALG, GFNI, VPCLMULQDQ, VAES.
 *
 *  We are going to use VBMI2 for `_mm256_maskz_compress_epi8`.
 */
#pragma region Skylake Implementation
#if SZ_USE_SKYLAKE
#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("avx,avx512f,avx512vl,avx512bw,bmi,bmi2"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("avx", "avx512f", "avx512vl", "avx512bw", "bmi", "bmi2")
#endif

/**
 *  @brief The most important part of the QuickSort algorithm partitioning the elements around the pivot.
 *  @note Unlike the serial algorithm, uses compressed stores to filter and move the elements around the pivot.
 */
SZ_INTERNAL void sz_sequence_argsort_skylake_3way_partition_(                       //
    sz_pgram_t *const initial_pgrams, sz_sorted_idx_t *const initial_order,         //
    sz_pgram_t *const partitioned_pgrams, sz_sorted_idx_t *const partitioned_order, //
    sz_size_t const start_in_sequence, sz_size_t const end_in_sequence,             //
    sz_size_t *const first_pivot_offset, sz_size_t *const last_pivot_offset) {

    sz_size_t const count = end_in_sequence - start_in_sequence;
    sz_size_t const pgrams_per_register = sizeof(sz_u512_vec_t) / sizeof(sz_pgram_t);

    // Choose the pivot offset with Sedgewick's method.
    sz_pgram_t const *pivot_pgram_ptr = sz_sequence_partitioning_pivot_(initial_pgrams + start_in_sequence, count);
    sz_pgram_t const pivot_pgram = *pivot_pgram_ptr;
    sz_u512_vec_t pivot_vec;
    pivot_vec.zmm = _mm512_set1_epi64(pivot_pgram);

    // Reading data is always cheaper than writing, so we can further minimize the writes, if
    // we know exactly, how many elements are smaller or greater than the pivot.
    sz_size_t count_smaller = 0, count_greater = 0;
    sz_size_t const tail_count = count & 7u;
    __mmask8 const tail_mask = sz_u8_mask_until_(tail_count);

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
    sz_assert_(count_equal >= 1 && "The pivot must be present in the collection.");
    sz_assert_(count_smaller + count_equal + count_greater == count && "The partitioning must be exhaustive.");
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
                    count * sizeof(sz_pgram_t));
    sz_copy_skylake((sz_ptr_t)(initial_order + start_in_sequence),      //
                    (sz_cptr_t)(partitioned_order + start_in_sequence), //
                    count * sizeof(sz_sorted_idx_t));

    // Return the offsets of the equal elements.
    *first_pivot_offset = start_in_sequence + count_smaller;
    *last_pivot_offset = start_in_sequence + count_smaller + count_equal - 1;
}

/**
 *  @brief Recursive Quick-Sort implementation backing both the `sz_sequence_argsort_skylake` and
 * `sz_pgrams_sort_skylake`, and using the `sz_sequence_argsort_skylake_3way_partition_` under the hood.
 */
SZ_PUBLIC void sz_sequence_argsort_skylake_recursively_(            //
    sz_pgram_t *initial_pgrams, sz_sorted_idx_t *initial_order,     //
    sz_pgram_t *temporary_pgrams, sz_sorted_idx_t *temporary_order, //
    sz_size_t const start_in_sequence, sz_size_t const end_in_sequence) {

    // On very small inputs, when we don't even have enough input for a single ZMM register,
    // use simple insertion sort without any extra memory.
    sz_size_t const count = end_in_sequence - start_in_sequence;
    sz_size_t const pgrams_per_register = sizeof(sz_u512_vec_t) / sizeof(sz_pgram_t);
    if (count <= pgrams_per_register) {
        sz_pgrams_sort_with_insertion( //
            initial_pgrams + start_in_sequence, count, initial_order + start_in_sequence);
        return;
    }

    // Partition the collection around some pivot
    sz_size_t first_pivot_index, last_pivot_index;
    sz_sequence_argsort_skylake_3way_partition_(                          //
        initial_pgrams, initial_order, temporary_pgrams, temporary_order, //
        start_in_sequence, end_in_sequence,                               //
        &first_pivot_index, &last_pivot_index);

    // Recursively sort the left and right partitions, if there are at least 2 elements in each
    if (start_in_sequence + 1 < first_pivot_index)
        sz_sequence_argsort_skylake_recursively_(                             //
            initial_pgrams, initial_order, temporary_pgrams, temporary_order, //
            start_in_sequence, first_pivot_index);
    if (last_pivot_index + 2 < end_in_sequence)
        sz_sequence_argsort_skylake_recursively_(                             //
            initial_pgrams, initial_order, temporary_pgrams, temporary_order, //
            last_pivot_index + 1, end_in_sequence);
}

SZ_PUBLIC sz_status_t sz_pgrams_sort_skylake(sz_pgram_t *pgrams, sz_size_t count, sz_memory_allocator_t *alloc,
                                             sz_sorted_idx_t *order) {

    // First, initialize the `order` with `std::iota`-like behavior.
    for (sz_size_t i = 0; i != count; ++i) order[i] = i;

    // Simplify usage in higher-level libraries, where wrapping custom allocators may be troublesome.
    sz_memory_allocator_t global_alloc;
    if (!alloc) {
        sz_memory_allocator_init_default(&global_alloc);
        alloc = &global_alloc;
    }

    // Allocate memory for partitioning the elements around the pivot.
    sz_size_t memory_usage = sizeof(sz_pgram_t) * count + sizeof(sz_sorted_idx_t) * count;
    sz_pgram_t *temporary_pgrams = (sz_pgram_t *)alloc->allocate(memory_usage, alloc);
    sz_sorted_idx_t *temporary_order = (sz_sorted_idx_t *)(temporary_pgrams + count);
    if (!temporary_pgrams) return sz_bad_alloc_k;

    // Reuse the string sorting algorithm for sorting the "pgrams".
    sz_sequence_argsort_skylake_recursively_(pgrams, order, temporary_pgrams, temporary_order, 0, count);

    // Deallocate the temporary memory used for partitioning.
    alloc->free(temporary_pgrams, memory_usage, alloc);
    return sz_success_k;
}

/**
 *  @brief Recursive Quick-Sort adaptation for strings, that processes the strings a few N-grams at a time.
 *
 *  It combines `sz_sequence_argsort_serial_export_next_pgrams_` and `sz_sequence_argsort_serial_recursively_`,
 *  recursively diving into the identical pgrams.
 */
SZ_PUBLIC void sz_sequence_argsort_skylake_next_pgrams_(                        //
    sz_sequence_t const *const sequence,                                        //
    sz_pgram_t *const global_pgrams, sz_sorted_idx_t *const global_order,       //
    sz_pgram_t *const temporary_pgrams, sz_sorted_idx_t *const temporary_order, //
    sz_size_t const start_in_sequence, sz_size_t const end_in_sequence,         //
    sz_size_t const start_character) {

    // Prepare the new range of pgrams
    sz_sequence_argsort_serial_export_next_pgrams_( //
        sequence, global_pgrams, global_order, start_in_sequence, end_in_sequence, start_character);

    // Sort current pgrams with a quicksort
    sz_sequence_argsort_skylake_recursively_( //
        global_pgrams, global_order, temporary_pgrams, temporary_order, start_in_sequence, end_in_sequence);

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
        if (has_multiple_strings && has_more_characters_in_each)
            sz_sequence_argsort_skylake_next_pgrams_( //
                sequence, global_pgrams, global_order, temporary_pgrams, temporary_order, nested_start, nested_end,
                start_character + pgram_capacity);

        // Move to the next
        nested_start = nested_end;
    }
}

SZ_PUBLIC sz_status_t sz_sequence_argsort_skylake(sz_sequence_t const *sequence, sz_memory_allocator_t *alloc,
                                                  sz_sorted_idx_t *order) {

    // First, initialize the `order` with `std::iota`-like behavior.
    sz_size_t count = sequence->count;
    for (sz_size_t i = 0; i != count; ++i) order[i] = i;

    // On very small collections - just use the quadratic-complexity insertion sort
    // without any smart optimizations or memory allocations.
    if (count <= 32) {
        sz_sequence_argsort_with_insertion(sequence, order);
        return sz_success_k;
    }

    // Simplify usage in higher-level libraries, where wrapping custom allocators may be troublesome.
    sz_memory_allocator_t global_alloc;
    if (!alloc) {
        sz_memory_allocator_init_default(&global_alloc);
        alloc = &global_alloc;
    }

    // Allocate memory for partitioning the elements around the pivot.
    sz_size_t memory_usage = sizeof(sz_pgram_t) * count * 2 + sizeof(sz_sorted_idx_t) * count;
    sz_pgram_t *global_pgrams = (sz_pgram_t *)alloc->allocate(memory_usage, alloc);
    sz_pgram_t *temporary_pgrams = global_pgrams + count;
    sz_sorted_idx_t *temporary_order = (sz_sorted_idx_t *)(temporary_pgrams + count);
    if (!global_pgrams) return sz_bad_alloc_k;

    // Recursively sort the whole sequence.
    sz_sequence_argsort_skylake_next_pgrams_(sequence, global_pgrams, order, temporary_pgrams, temporary_order, //
                                             0, count, 0);

    // Free temporary storage.
    alloc->free(global_pgrams, memory_usage, alloc);
    return sz_success_k;
}

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#endif            // SZ_USE_SKYLAKE
#pragma endregion // Skylake Implementation

#pragma region SVE Implementation
#if SZ_USE_SVE
#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("arch=armv8.2-a+sve"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("arch=armv8.2-a+sve")
#endif

/**
 *  @brief The most important part of the QuickSort algorithm partitioning the elements around the pivot.
 *  @note Unlike the serial algorithm, uses compressed stores to filter and move the elements around the pivot.
 *  @sa Identical to @b Skylake implementation, but uses variable length SVE registers.
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
    svuint64_t pivot_vec = svdup_n_u64(pivot_pgram);

    // Count elements smaller and greater than the pivot.
    sz_size_t count_smaller = 0, count_greater = 0;
    for (sz_size_t i = start_in_sequence; i < end_in_sequence; i += pgrams_per_vector) {
        svbool_t load_mask = svwhilelt_b64((sz_u64_t)i, (sz_u64_t)end_in_sequence);
        svuint64_t pgrams_vec = svld1_u64(load_mask, (sz_u64_t const *)(initial_pgrams + i));
        svbool_t smaller_mask = svcmplt_u64(load_mask, pgrams_vec, pivot_vec);
        svbool_t greater_mask = svcmpgt_u64(load_mask, pgrams_vec, pivot_vec);
        count_smaller = svqincp_n_u64_b64(count_smaller, smaller_mask); // Smarter than `svcntp_b64`
        count_greater = svqincp_n_u64_b64(count_greater, greater_mask); // Smarter than `svcntp_b64`
    }

    sz_size_t const count_equal = count - count_smaller - count_greater;
    sz_assert_(count_equal >= 1 && "The pivot must be present in the collection.");
    sz_assert_(count_smaller + count_equal + count_greater == count && "The partitioning must be exhaustive.");

    // Set offsets for each partition.
    sz_size_t smaller_offset = start_in_sequence;
    sz_size_t equal_offset = start_in_sequence + count_smaller;
    sz_size_t greater_offset = start_in_sequence + count_smaller + count_equal;

    // Partition elements into three segments.
    for (sz_size_t i = start_in_sequence; i < end_in_sequence; i += pgrams_per_vector) {
        svbool_t load_mask = svwhilelt_b64((sz_u64_t)i, (sz_u64_t)end_in_sequence);
        svuint64_t pgrams_vec = svld1_u64(load_mask, (sz_u64_t const *)(initial_pgrams + i));
        svuint64_t order_vec = svld1_u64(load_mask, (sz_u64_t const *)(initial_order + i));

        svbool_t smaller_mask = svcmplt_u64(load_mask, pgrams_vec, pivot_vec);
        svbool_t equal_mask = svcmpeq_u64(load_mask, pgrams_vec, pivot_vec);
        svbool_t greater_mask = svcmpgt_u64(load_mask, pgrams_vec, pivot_vec);

        // Compress the elements that satisfy the predicate and store them contiguously.
        sz_size_t count_smaller = svcntp_b64(smaller_mask, smaller_mask);
        sz_size_t count_equal = svcntp_b64(equal_mask, equal_mask);
        sz_size_t count_greater = svcntp_b64(greater_mask, greater_mask);
        if (count_smaller) {
            svuint64_t comp_pgrams = svcompact_u64(smaller_mask, pgrams_vec);
            svuint64_t comp_order = svcompact_u64(smaller_mask, order_vec);
            svbool_t store_mask = svwhilelt_b64((sz_u64_t)0, (sz_u64_t)count_smaller);
            svst1_u64(store_mask, (sz_u64_t *)(partitioned_pgrams + smaller_offset), comp_pgrams);
            svst1_u64(store_mask, (sz_u64_t *)(partitioned_order + smaller_offset), comp_order);
            smaller_offset += count_smaller;
        }
        if (count_equal) {
            svuint64_t comp_pgrams = svcompact_u64(equal_mask, pgrams_vec);
            svuint64_t comp_order = svcompact_u64(equal_mask, order_vec);
            svbool_t store_mask = svwhilelt_b64((sz_u64_t)0, (sz_u64_t)count_equal);
            svst1_u64(store_mask, (sz_u64_t *)(partitioned_pgrams + equal_offset), comp_pgrams);
            svst1_u64(store_mask, (sz_u64_t *)(partitioned_order + equal_offset), comp_order);
            equal_offset += count_equal;
        }
        if (count_greater) {
            svuint64_t comp_pgrams = svcompact_u64(greater_mask, pgrams_vec);
            svuint64_t comp_order = svcompact_u64(greater_mask, order_vec);
            svbool_t store_mask = svwhilelt_b64((sz_u64_t)0, (sz_u64_t)count_greater);
            svst1_u64(store_mask, (sz_u64_t *)(partitioned_pgrams + greater_offset), comp_pgrams);
            svst1_u64(store_mask, (sz_u64_t *)(partitioned_order + greater_offset), comp_order);
            greater_offset += count_greater;
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
 *  @brief Recursive Quick-Sort implementation backing both the `sz_sequence_argsort_skylake` and
 * `sz_pgrams_sort_skylake`, and using the `sz_sequence_argsort_skylake_3way_partition_` under the hood.
 *  @sa Identical to @b Skylake implementation, but uses variable length SVE registers.
 */
SZ_PUBLIC void sz_sequence_argsort_sve_recursively_(sz_pgram_t *initial_pgrams, sz_sorted_idx_t *initial_order,
                                                    sz_pgram_t *temporary_pgrams, sz_sorted_idx_t *temporary_order,
                                                    sz_size_t const start_in_sequence,
                                                    sz_size_t const end_in_sequence) {
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
                                             start_in_sequence, first_pivot_index);
    if (last_pivot_index + 2 < end_in_sequence)
        sz_sequence_argsort_sve_recursively_(initial_pgrams, initial_order, temporary_pgrams, temporary_order,
                                             last_pivot_index + 1, end_in_sequence);
}

SZ_PUBLIC sz_status_t sz_pgrams_sort_sve(sz_pgram_t *pgrams, sz_size_t count, sz_memory_allocator_t *alloc,
                                         sz_sorted_idx_t *order) {
    // Initialize the order with 0,1,2,...
    for (sz_size_t i = 0; i != count; ++i) order[i] = i;

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

    sz_sequence_argsort_sve_recursively_(pgrams, order, temporary_pgrams, temporary_order, 0, count);

    alloc->free(temporary_pgrams, memory_usage, alloc);
    return sz_success_k;
}

/**
 *  @brief Recursive Quick-Sort adaptation for strings, that processes the strings a few N-grams at a time.
 *  @sa Identical to @b Skylake implementation, but uses variable length SVE registers.
 *
 *  It combines `sz_sequence_argsort_serial_export_next_pgrams_` and `sz_sequence_argsort_serial_recursively_`,
 *  recursively diving into the identical pgrams.
 */
SZ_PUBLIC void sz_sequence_argsort_sve_next_pgrams_(
    sz_sequence_t const *const sequence, sz_pgram_t *const global_pgrams, sz_sorted_idx_t *const global_order,
    sz_pgram_t *const temporary_pgrams, sz_sorted_idx_t *const temporary_order, sz_size_t const start_in_sequence,
    sz_size_t const end_in_sequence, sz_size_t const start_character) {

    // Export the next pgrams from the sequence.
    sz_sequence_argsort_serial_export_next_pgrams_(sequence, global_pgrams, global_order, start_in_sequence,
                                                   end_in_sequence, start_character);

    // Sort the current pgrams with the SVE quicksort.
    sz_sequence_argsort_sve_recursively_(global_pgrams, global_order, temporary_pgrams, temporary_order,
                                         start_in_sequence, end_in_sequence);

    // For each group of equal pgrams, if there are multiple strings and more characters,
    // recursively sort the next pgrams.
    sz_size_t const pgram_capacity = sizeof(sz_pgram_t) - 1;
    sz_size_t nested_start = start_in_sequence;
    sz_size_t nested_end = start_in_sequence;
    while (nested_end != end_in_sequence) {
        sz_pgram_t current_pgram = global_pgrams[nested_start];
        while (nested_end != end_in_sequence && current_pgram == global_pgrams[nested_end]) ++nested_end;

        sz_cptr_t current_pgram_str = (sz_cptr_t)&current_pgram;
#if !SZ_IS_BIG_ENDIAN_
        sz_size_t current_pgram_length = (sz_size_t)current_pgram_str[0]; //! The byte order was swapped
#else
        sz_size_t current_pgram_length = (sz_size_t)current_pgram_str[pgram_capacity]; // ! No byte swaps on big-endian
#endif
        int has_multiple_strings = nested_end - nested_start > 1;
        int has_more_characters_in_each = current_pgram_length == pgram_capacity;
        if (has_multiple_strings && has_more_characters_in_each)
            sz_sequence_argsort_sve_next_pgrams_(sequence, global_pgrams, global_order, temporary_pgrams,
                                                 temporary_order, nested_start, nested_end,
                                                 start_character + pgram_capacity);
        nested_start = nested_end;
    }
}

SZ_PUBLIC sz_status_t sz_sequence_argsort_sve(sz_sequence_t const *sequence, sz_memory_allocator_t *alloc,
                                              sz_sorted_idx_t *order) {
    sz_size_t count = sequence->count;
    for (sz_size_t i = 0; i != count; ++i) order[i] = i;

    if (count <= 32) {
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

    sz_sequence_argsort_sve_next_pgrams_(sequence, global_pgrams, order, temporary_pgrams, temporary_order, 0, count,
                                         0);

    alloc->free(global_pgrams, memory_usage, alloc);
    return sz_success_k;
}

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#endif            // SZ_USE_SVE
#pragma endregion // SVE Implementation

/*  Pick the right implementation for the string search algorithms.
 *  To override this behavior and precompile all backends - set `SZ_DYNAMIC_DISPATCH` to 1.
 */
#pragma region Compile Time Dispatching
#if !SZ_DYNAMIC_DISPATCH

SZ_DYNAMIC sz_status_t sz_sequence_argsort(sz_sequence_t const *sequence, sz_memory_allocator_t *alloc,
                                           sz_sorted_idx_t *order) {
#if SZ_USE_SKYLAKE
    return sz_sequence_argsort_skylake(sequence, alloc, order);
#elif SZ_USE_SVE
    return sz_sequence_argsort_sve(sequence, alloc, order);
#else
    return sz_sequence_argsort_serial(sequence, alloc, order);
#endif
}

SZ_DYNAMIC sz_status_t sz_pgrams_sort(sz_pgram_t *pgrams, sz_size_t count, sz_memory_allocator_t *alloc,
                                      sz_sorted_idx_t *order) {
#if SZ_USE_SKYLAKE
    return sz_pgrams_sort_skylake(pgrams, count, alloc, order);
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
