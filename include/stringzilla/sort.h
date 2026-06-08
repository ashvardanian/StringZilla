/**
 *  @brief Hardware-accelerated string collection sorting.
 *  @file include/stringzilla/sort.h
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

#include "stringzilla/types.h"

#include "stringzilla/compare.h" // `sz_compare`
#include "stringzilla/memory.h"  // `sz_copy`

#ifdef __cplusplus
extern "C" {
#endif

#pragma region Core API

/**
 *  @brief Faster @b arg-sort for an arbitrary @b string sequence, using QuickSort.
 *         Outputs the @p order of elements in the immutable @p sequence, that would sort it.
 *
 *  @param sequence Immutable sequence of strings to sort.
 *  @param alloc Optional memory allocator for temporary storage.
 *  @param order Output permutation that sorts the elements.
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
 *  @note The algorithm has linear memory complexity, quadratic worst-case and log-linear average time complexity.
 *  @see https://en.wikipedia.org/wiki/Quicksort
 *
 *  @note This algorithm is @b unstable: equal elements may change relative order.
 *  @sa sz_sequence_argsort_stabilize
 *
 *  @note Selects the fastest implementation at compile- or run-time based on `SZ_DYNAMIC_DISPATCH`.
 *  @sa sz_sequence_argsort_serial, sz_sequence_argsort_skylake, sz_sequence_argsort_sve
 */
SZ_DYNAMIC sz_status_t sz_sequence_argsort(sz_sequence_t const *sequence, sz_memory_allocator_t *alloc,
                                           sz_sorted_idx_t *order);

/**
 *  @brief Faster @b inplace `std::sort` for a continuous @b unsigned-integer sequence, using QuickSort.
 *         Overwrites the input @p pgrams with the sorted sequence and exports the @p order permutation.
 *
 *  @param pgrams Continuous buffer of unsigned integers to sort in place.
 *  @param count Number of elements in the sequence.
 *  @param alloc Optional memory allocator for temporary storage.
 *  @param order Output permutation that sorts the elements.
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
 *  @note The algorithm has linear memory complexity, quadratic worst-case and log-linear average time complexity.
 *  @see https://en.wikipedia.org/wiki/Quicksort
 *
 *  @note Selects the fastest implementation at compile- or run-time based on `SZ_DYNAMIC_DISPATCH`.
 *  @sa sz_pgrams_sort_serial, sz_pgrams_sort_skylake, sz_pgrams_sort_sve
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
 *  @brief Quadratic complexity @b stable insertion sort adjust for our @b argsort usecase.
 *         Needs no extra memory and is used as a fallback for small inputs.
 */
SZ_PUBLIC void sz_sequence_argsort_with_insertion(sz_sequence_t const *sequence, sz_sorted_idx_t *order) {
    // Assume `order` is already initialized with 0, 1, 2, ... N.
    for (sz_size_t element_index = 1; element_index < sequence->count; ++element_index) {
        sz_sorted_idx_t current_idx = order[element_index];
        sz_size_t position_index = element_index;
        while (position_index > 0) {
            // Get the two strings to compare.
            sz_sorted_idx_t previous_idx = order[position_index - 1];
            sz_cptr_t previous_start = sequence->get_start(sequence->handle, previous_idx);
            sz_cptr_t current_start = sequence->get_start(sequence->handle, current_idx);
            sz_size_t previous_length = sequence->get_length(sequence->handle, previous_idx);
            sz_size_t current_length = sequence->get_length(sequence->handle, current_idx);

            // Use the provided sz_order to compare.
            sz_ordering_t ordering = sz_order(previous_start, previous_length, current_start, current_length);

            // If the previous string is not greater than current_idx, we're done.
            if (ordering != sz_greater_k) break;

            // Otherwise, shift the previous element to the right.
            order[position_index] = order[position_index - 1];
            --position_index;
        }
        order[position_index] = current_idx;
    }
}

/**
 *  @brief Quadratic complexity @b stable insertion sort adjust for our @b pgram-sorting usecase.
 *         Needs no extra memory and is used as a fallback for small inputs.
 */

SZ_PUBLIC void sz_pgrams_sort_with_insertion(sz_pgram_t *pgrams, sz_size_t count, sz_sorted_idx_t *order) {

    // Assume `order` is already initialized with 0, 1, 2, ... N.
    for (sz_size_t element_index = 1; element_index < count; ++element_index) {
        // Save the current key and corresponding index.
        sz_pgram_t current_key = pgrams[element_index];
        sz_sorted_idx_t current_idx = order[element_index];
        sz_size_t position_index = element_index;

        // Shift elements of the sorted region that are greater than the current key
        // to the right. This loop stops as soon as the correct insertion point is found.
        while (position_index > 0 && pgrams[position_index - 1] > current_key) {
            pgrams[position_index] = pgrams[position_index - 1];
            order[position_index] = order[position_index - 1];
            --position_index;
        }

        // Insert the current key and index into their proper location.
        pgrams[position_index] = current_key;
        order[position_index] = current_idx;
    }

#if SZ_DEBUG
    for (sz_size_t element_index = 1; element_index < count; ++element_index)
        sz_assert_(pgrams[element_index - 1] <= pgrams[element_index] &&
                   "The pgrams should be sorted in ascending order.");
#endif
}

#pragma endregion // Generic Public Helpers

#pragma region Generic Internal Helpers

/**
 *  @brief Convenience macro for of conditional swap of "pgrams" and their indices for a sorting network.
 *  @see https://en.wikipedia.org/wiki/Sorting_network
 */
#define sz_sequence_sorting_network_conditional_swap_(i, j)    \
    do {                                                       \
        if (pgrams[i] > pgrams[j]) {                           \
            sz_swap_(sz_pgram_t, pgrams[i], pgrams[j]);        \
            sz_swap_(sz_sorted_idx_t, offsets[i], offsets[j]); \
        }                                                      \
    } while (0)

/**
 *  @brief Sorting network for 2 elements is just a single compare-swap.
 */
SZ_INTERNAL void sz_sequence_sorting_network_2x_(sz_pgram_t *pgrams, sz_sorted_idx_t *offsets) {
    sz_sequence_sorting_network_conditional_swap_(0, 1);
}

/**
 *  @brief Sorting network for 3 elements.
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
    for (sz_size_t element_index = 1; element_index < 3; ++element_index)
        sz_assert_(pgrams[element_index - 1] <= pgrams[element_index] && "Sorting network for 3 elements failed.");
#endif
}

/**
 *  @brief Sorting network for 4 elements.
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
    for (sz_size_t element_index = 1; element_index < 4; ++element_index)
        sz_assert_(pgrams[element_index - 1] <= pgrams[element_index] && "Sorting network for 4 elements failed.");
#endif
}

/**
 *  @brief A scalar sorting network for 8 elements that reorders both the pgrams
 *         and their corresponding offsets in only 19 comparisons, the most efficient
 *         variant currently known.
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
    for (sz_size_t element_index = 1; element_index < 8; ++element_index)
        sz_assert_(pgrams[element_index - 1] <= pgrams[element_index] &&
                   "The sorting network must sort the pgrams in ascending order.");
#endif
}

#undef sz_sequence_sorting_network_conditional_swap_

#pragma endregion // Generic Internal Helpers

#include "stringzilla/sort/serial.h"
#include "stringzilla/sort/skylake.h"
#include "stringzilla/sort/sve.h"

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
