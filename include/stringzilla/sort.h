/**
 *  @brief  Hardware-accelerated string collection sorting and intersections.
 *  @file   sort.h
 *  @author Ash Vardanian
 *
 *  Includes core APIs:
 *
 *  - `sz_qsort` - to sort an arbitrary string collection with QuickSort-like algorithm.
 *  - `sz_qsort_addresses` - to sort a collection of continuous pointer-sized integers with QuickSort-like algorithm.
 *  - `sz_msort` - to sort an arbitrary string collection with a MergeSort-like algorithm.
 *  - `sz_msort_addresses` - to sort a collection of continuous pointer-sized integers with a MergeSort-like algorithm.
 *
 *  The `qsort` variants are not guaranteed to be stable.
 *  The `msort` variants are guaranteed to be stable.
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
 *  @brief  Faster `std::sort` for an arbitrary string sequence.
 *
 *  @param collection The collection of strings to sort.
 *  @param alloc Memory allocator for temporary storage.
 *  @param order The output - indices of the sorted collection elements.
 *  @return Whether the operation was successful.
 */
SZ_DYNAMIC sz_bool_t sz_qsort(sz_sequence_t const *collection, sz_memory_allocator_t *alloc, sz_sorted_idx_t *order);

/** @copydoc sz_qsort */
SZ_PUBLIC sz_bool_t sz_qsort_serial(sz_sequence_t const *collection, sz_memory_allocator_t *alloc,
                                    sz_sorted_idx_t *order);

/** @copydoc sz_qsort */
SZ_PUBLIC sz_bool_t sz_qsort_skylake(sz_sequence_t const *collection, sz_memory_allocator_t *alloc,
                                     sz_sorted_idx_t *order);

/** @copydoc sz_qsort */
SZ_PUBLIC sz_bool_t sz_qsort_sve(sz_sequence_t const *collection, sz_memory_allocator_t *alloc, sz_sorted_idx_t *order);

#pragma endregion

#pragma region Serial Implementation

/**
 *  The core idea of all following string algorithms is to sort strings not based on 1 character at a time,
 *  but on a larger integer word fitting in 4 or 8 bytes at once, on 32-bit or 64-bit architectures, respectively.
 *  That word is pointer-sized, but it may contain extra information aside from N characters.
 */
typedef sz_size_t _sz_sort_ngram_t;

SZ_PUBLIC void _sz_qsort_serial_export_prefixes(                                       //
    sz_sequence_t const *const collection,                                             //
    _sz_sort_ngram_t *const global_windows, sz_sorted_idx_t const *const global_order, //
    sz_size_t const start_in_collection, sz_size_t const end_in_collection,            //
    sz_size_t const start_character) {

    // Depending on the architecture, we will export a different number of bytes.
    // On 32-bit architectures, we will export 3 bytes, and on 64-bit architectures - 7 bytes.
    sz_size_t const window_capacity = sizeof(_sz_sort_ngram_t) - 1;

    // Perform the same operation for every string.
    for (sz_size_t i = start_in_collection; i < end_in_collection; ++i) {

        // On the first recursion level, the `global_order` is the identity permutation.
        sz_sorted_idx_t const partial_order_index = global_order[i];
        if (SZ_DEBUG && start_character == 0)
            _sz_assert(partial_order_index == i && "At start this must be an identity permutation.");

        // Get the string slice in global memory.
        sz_cptr_t const source_str = collection->get_start(collection, partial_order_index);
        sz_size_t const length = collection->get_length(collection, partial_order_index);
        sz_size_t const remaining_length = length > start_character ? length - start_character : 0;
        sz_size_t const exported_length = remaining_length > window_capacity ? window_capacity : remaining_length;

        // Fill with zeros, export a slice, and mark the exported length.
        sz_size_t *target_integer = &global_windows[i];
        sz_ptr_t target_str = (sz_ptr_t)target_integer;
        *target_integer = 0;
        for (sz_size_t j = 0; j < exported_length; ++j) target_str[j] = source_str[j + start_character];
        target_str[window_capacity] = exported_length;
#if defined(_SZ_IS_64_BIT)
        *target_integer = sz_u64_bytes_reverse(*target_integer);
#else
        *target_integer = sz_u32_bytes_reverse(*target_integer);
#endif
        _sz_assert(                                                  //
            (length <= start_character) == (*target_integer == 0) && //
            "We can have a zero value if only the string is shorter than other strings at this position.");
    }

    // As our goal is to sort the strings using the exported integer "windows",
    // this is a good place to validate the correctness of the exported data.
    if (SZ_DEBUG && start_character == 0)
        for (sz_size_t i = start_in_collection + 1; i < end_in_collection; ++i) {
            _sz_sort_ngram_t const previous_window = global_windows[i - 1];
            _sz_sort_ngram_t const current_window = global_windows[i];
            sz_cptr_t const previous_str = collection->get_start(collection, i - 1);
            sz_size_t const previous_length = collection->get_length(collection, i - 1);
            sz_cptr_t const current_str = collection->get_start(collection, i);
            sz_size_t const current_length = collection->get_length(collection, i);
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
SZ_PUBLIC void _sz_qsort_serial_3way_partition(                                  //
    _sz_sort_ngram_t *const global_windows, sz_sorted_idx_t *const global_order, //
    sz_size_t const start_in_collection, sz_size_t const end_in_collection,      //
    sz_size_t *first_pivot_offset, sz_size_t *last_pivot_offset) {

    // Chose the pivot offset with Sedgewick's method.
    _sz_sort_ngram_t pivot_window;
    {
        sz_size_t const middle_offset = start_in_collection + (end_in_collection - start_in_collection) / 2;
        sz_size_t const last_offset = end_in_collection - 1;
        sz_size_t const first_offset = start_in_collection;
        _sz_sort_ngram_t const first_window = global_windows[first_offset];
        _sz_sort_ngram_t const middle_window = global_windows[middle_offset];
        _sz_sort_ngram_t const last_window = global_windows[last_offset];
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
    sz_size_t partitioning_progress = start_in_collection;       // Current index.
    sz_size_t less_than_pivot_offset = start_in_collection;      // Boundary for elements < pivot_window.
    sz_size_t greater_than_pivot_offset = end_in_collection - 1; // Boundary for elements > pivot_window.

    while (partitioning_progress <= greater_than_pivot_offset) {
        // Element is less than pivot: swap into the < pivot region.
        if (global_windows[partitioning_progress] < pivot_window) {
#if defined(_SZ_IS_64_BIT)
            sz_u64_swap(&global_order[partitioning_progress], &global_order[less_than_pivot_offset]);
            sz_u64_swap(&global_windows[partitioning_progress], &global_windows[less_than_pivot_offset]);
#else
            sz_u32_swap(&global_order[partitioning_progress], &global_order[less_than_pivot_offset]);
            sz_u32_swap(&global_windows[partitioning_progress], &global_windows[less_than_pivot_offset]);
#endif
            ++partitioning_progress;
            ++less_than_pivot_offset;
        }
        // Element is greater than pivot: swap into the > pivot region.
        else if (global_windows[partitioning_progress] > pivot_window) {
#if defined(_SZ_IS_64_BIT)
            sz_u64_swap(&global_order[partitioning_progress], &global_order[greater_than_pivot_offset]);
            sz_u64_swap(&global_windows[partitioning_progress], &global_windows[greater_than_pivot_offset]);
#else
            sz_u32_swap(&global_order[partitioning_progress], &global_order[greater_than_pivot_offset]);
            sz_u32_swap(&global_windows[partitioning_progress], &global_windows[greater_than_pivot_offset]);
#endif
            --greater_than_pivot_offset;
        }
        // Element equals pivot_window: leave it in place.
        else { ++partitioning_progress; }
    }

    *first_pivot_offset = less_than_pivot_offset;
    *last_pivot_offset = greater_than_pivot_offset;
}

SZ_PUBLIC void _sz_qsort_serial_recursively(                                //
    sz_sequence_t const *const collection,                                  //
    _sz_sort_ngram_t *const global_windows, sz_size_t *const global_order,  //
    sz_size_t const start_in_collection, sz_size_t const end_in_collection, //
    sz_size_t const start_character) {

    // Partition the collection around some pivot or 2 pivots in a 3-way partitioning
    sz_size_t first_pivot_index, last_pivot_index;
    _sz_qsort_serial_3way_partition(            //
        global_windows, global_order,           //
        start_in_collection, end_in_collection, //
        &first_pivot_index, &last_pivot_index);

    // Recursively sort the left partition
    if (start_in_collection < first_pivot_index)
        _sz_qsort_serial_recursively(collection, global_windows, global_order, start_in_collection, first_pivot_index,
                                     start_character);

    // Recursively sort the right partition
    if (last_pivot_index + 1 < end_in_collection)
        _sz_qsort_serial_recursively(collection, global_windows, global_order, last_pivot_index + 1, end_in_collection,
                                     start_character);
}

SZ_PUBLIC void _sz_qsort_serial_next_window(                                //
    sz_sequence_t const *const collection,                                  //
    _sz_sort_ngram_t *const global_windows, sz_size_t *const global_order,  //
    sz_size_t const start_in_collection, sz_size_t const end_in_collection, //
    sz_size_t const start_character) {

    // Prepare the new range of windows
    _sz_qsort_serial_export_prefixes(collection, global_windows, global_order, start_in_collection, end_in_collection,
                                     start_character);

    // Sort current windows with a quicksort
    _sz_qsort_serial_recursively(collection, global_windows, global_order, start_in_collection, end_in_collection,
                                 start_character);

    // Depending on the architecture, we will export a different number of bytes.
    // On 32-bit architectures, we will export 3 bytes, and on 64-bit architectures - 7 bytes.
    sz_size_t const window_capacity = sizeof(_sz_sort_ngram_t) - 1;

    // Repeat the procedure for the identical windows
    sz_size_t nested_start = start_in_collection;
    sz_size_t nested_end = start_in_collection;
    while (nested_end != end_in_collection) {
        // Find the end of the identical windows
        _sz_sort_ngram_t current_window_integer = global_windows[nested_start];
        while (nested_end != end_in_collection && current_window_integer == global_windows[nested_end]) ++nested_end;

        // If the identical windows are not trivial and each string has more characters, sort them recursively
        sz_cptr_t current_window_str = (sz_cptr_t)&current_window_integer;
        sz_size_t current_window_length = (sz_size_t)current_window_str[0]; //! The byte order was swapped
        if (nested_end - nested_start > 1 && current_window_length == window_capacity) {
            _sz_qsort_serial_next_window(collection, global_windows, global_order, nested_start, nested_end,
                                         start_character + window_capacity);
        }
        // Move to the next
        nested_start = nested_end;
    }
}

SZ_PUBLIC void _sz_qsort_serial_insertion(sz_sequence_t const *collection, sz_memory_allocator_t *alloc,
                                          sz_sorted_idx_t *order) {
    // This algorithm needs no memory allocations:
    sz_unused(alloc);

    // Assume `order` is already initialized with 0, 1, 2, ... N.
    for (sz_size_t i = 1; i < collection->count; ++i) {
        sz_sorted_idx_t current_idx = order[i];
        sz_size_t j = i;
        while (j > 0) {
            // Get the two strings to compare.
            sz_sorted_idx_t previous_idx = order[j - 1];
            sz_cptr_t previous_start = collection->get_start(collection, previous_idx);
            sz_cptr_t current_start = collection->get_start(collection, current_idx);
            sz_size_t previous_length = collection->get_length(collection, previous_idx);
            sz_size_t current_length = collection->get_length(collection, current_idx);

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

SZ_PUBLIC sz_bool_t sz_qsort_serial(sz_sequence_t const *collection, sz_memory_allocator_t *alloc,
                                    sz_sorted_idx_t *order) {

    // First, initialize the `order` with `std::iota`-like behavior.
    for (sz_size_t i = 0; i != collection->count; ++i) order[i] = i;

    // On very small collections - just use the quadratic-complexity insertion sort
    // without any smart optimizations or memory allocations.
    if (collection->count <= 32) {
        _sz_qsort_serial_insertion(collection, alloc, order);
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
    _sz_sort_ngram_t *windows =
        (_sz_sort_ngram_t *)alloc->allocate(collection->count * sizeof(_sz_sort_ngram_t), alloc);
    if (!windows) return sz_false_k;

    // Recursively sort the whole collection.
    _sz_qsort_serial_next_window(collection, windows, order, 0, collection->count, 0);

    // Free temporary storage.
    alloc->free(windows, collection->count * sizeof(_sz_sort_ngram_t), alloc);
    return sz_true_k;
}

#pragma endregion // Serial Implementation

#pragma region Ice Lake Implementation

SZ_PUBLIC void _sz_qsort_ice_recursively(                                   //
    sz_sequence_t const *const collection,                                  //
    _sz_sort_ngram_t *const global_windows, sz_size_t *const global_order,  //
    sz_size_t const start_in_collection, sz_size_t const end_in_collection, //
    sz_size_t const start_character) {

    // Prepare the new range of windows
    _sz_qsort_serial_export_prefixes(collection, global_windows, global_order, start_in_collection, end_in_collection,
                                     start_character);

    // We can implement a form of a Radix sort here, that will count the number of elements with
    // a certain bit set. The naive approach may require too many loops over data. A more "vectorized"
    // approach would be to maintain a histogram for several bits at once. For 4 bits we will
    // need 2^4 = 16 counters.
    sz_size_t histogram[16] = {0};
    for (sz_size_t byte_in_window = 0; byte_in_window != sizeof(_sz_sort_ngram_t); ++byte_in_window) {
        // First sort based on the low nibble of each byte.
        for (sz_size_t i = start_in_collection; i < end_in_collection; ++i) {
            sz_size_t const byte = (global_windows[i] >> (byte_in_window * 8)) & 0xFF;
            ++histogram[byte];
        }
        sz_size_t offset = start_in_collection;
        for (sz_size_t i = 0; i != 16; ++i) {
            sz_size_t const count = histogram[i];
            histogram[i] = offset;
            offset += count;
        }
        for (sz_size_t i = start_in_collection; i < end_in_collection; ++i) {
            sz_size_t const byte = (global_windows[i] >> (byte_in_window * 8)) & 0xFF;
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

SZ_DYNAMIC sz_bool_t sz_qsort(sz_sequence_t const *collection, sz_memory_allocator_t *alloc, sz_sorted_idx_t *order) {
    return sz_qsort_serial(collection, alloc, order);
}

#endif            // !SZ_DYNAMIC_DISPATCH
#pragma endregion // Compile Time Dispatching

#ifdef __cplusplus
}
#endif // __cplusplus
#endif // STRINGZILLA_SORT_H_
