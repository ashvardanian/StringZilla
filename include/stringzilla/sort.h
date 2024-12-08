/**
 *  @brief  Hardware-accelerated string sorting.
 *  @file   sort.h
 *  @author Ash Vardanian
 *
 *  Includes core APIs:
 *
 *  - `sz_partition` - to split the sequence into two parts based on a predicate.
 *  - `sz_merge` - to merge two consecutive sorted chunks forming the same continuous `sequence`.
 *  - `sz_sort` - to sort an arbitrary string sequence.
 *  - `sz_sort_partial` - to partially sort an arbitrary string sequence.
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
 *  @brief  Similar to `std::partition`, given a predicate splits the sequence into two parts.
 *          The algorithm is unstable, meaning that elements may change relative order, as long
 *          as they are in the right partition. This is the simpler algorithm for partitioning.
 */
SZ_PUBLIC sz_size_t sz_partition(sz_sequence_t *sequence, sz_sequence_predicate_t predicate);

/**
 *  @brief  Inplace `std::set_union` for two consecutive chunks forming the same continuous `sequence`.
 *
 *  @param partition The number of elements in the first sub-sequence in `sequence`.
 *  @param less Comparison function, to determine the lexicographic ordering.
 */
SZ_PUBLIC void sz_merge(sz_sequence_t *sequence, sz_size_t partition, sz_sequence_comparator_t less);

/**
 *  @brief  Sorting algorithm, combining Radix Sort for the first 32 bits of every word
 *          and a follow-up by a more conventional sorting procedure on equally prefixed parts.
 */
SZ_PUBLIC void sz_sort(sz_sequence_t *sequence);

/**
 *  @brief  Partial sorting algorithm, combining Radix Sort for the first 32 bits of every word
 *          and a follow-up by a more conventional sorting procedure on equally prefixed parts.
 */
SZ_PUBLIC void sz_sort_partial(sz_sequence_t *sequence, sz_size_t n);

/**
 *  @brief  Intro-Sort algorithm that supports custom comparators.
 */
SZ_PUBLIC void sz_sort_intro(sz_sequence_t *sequence, sz_sequence_comparator_t less);

#pragma endregion

#pragma region Serial Implementation

SZ_PUBLIC sz_size_t sz_partition(sz_sequence_t *sequence, sz_sequence_predicate_t predicate) {

    sz_size_t matches = 0;
    while (matches != sequence->count && predicate(sequence, sequence->order[matches])) ++matches;

    for (sz_size_t i = matches + 1; i < sequence->count; ++i)
        if (predicate(sequence, sequence->order[i]))
            sz_u64_swap(sequence->order + i, sequence->order + matches), ++matches;

    return matches;
}

SZ_PUBLIC void sz_merge(sz_sequence_t *sequence, sz_size_t partition, sz_sequence_comparator_t less) {

    sz_size_t start_b = partition + 1;

    // If the direct merge is already sorted
    if (!less(sequence, sequence->order[start_b], sequence->order[partition])) return;

    sz_size_t start_a = 0;
    while (start_a <= partition && start_b <= sequence->count) {

        // If element 1 is in right place
        if (!less(sequence, sequence->order[start_b], sequence->order[start_a])) { start_a++; }
        else {
            sz_size_t value = sequence->order[start_b];
            sz_size_t index = start_b;

            // Shift all the elements between element 1
            // element 2, right by 1.
            while (index != start_a) { sequence->order[index] = sequence->order[index - 1], index--; }
            sequence->order[start_a] = value;

            // Update all the pointers
            start_a++;
            partition++;
            start_b++;
        }
    }
}

SZ_PUBLIC void sz_sort_insertion(sz_sequence_t *sequence, sz_sequence_comparator_t less) {
    sz_u64_t *keys = sequence->order;
    sz_size_t keys_count = sequence->count;
    for (sz_size_t i = 1; i < keys_count; i++) {
        sz_u64_t i_key = keys[i];
        sz_size_t j = i;
        for (; j > 0 && less(sequence, i_key, keys[j - 1]); --j) keys[j] = keys[j - 1];
        keys[j] = i_key;
    }
}

SZ_INTERNAL void _sz_sift_down( //
    sz_sequence_t *sequence, sz_sequence_comparator_t less, sz_u64_t *order, sz_size_t start, sz_size_t end) {
    sz_size_t root = start;
    while (2 * root + 1 <= end) {
        sz_size_t child = 2 * root + 1;
        if (child + 1 <= end && less(sequence, order[child], order[child + 1])) { child++; }
        if (!less(sequence, order[root], order[child])) { return; }
        sz_u64_swap(order + root, order + child);
        root = child;
    }
}

SZ_INTERNAL void _sz_heapify(sz_sequence_t *sequence, sz_sequence_comparator_t less, sz_u64_t *order, sz_size_t count) {
    sz_size_t start = (count - 2) / 2;
    while (1) {
        _sz_sift_down(sequence, less, order, start, count - 1);
        if (start == 0) return;
        start--;
    }
}

SZ_INTERNAL void _sz_heapsort(sz_sequence_t *sequence, sz_sequence_comparator_t less, sz_size_t first, sz_size_t last) {
    sz_u64_t *order = sequence->order;
    sz_size_t count = last - first;
    _sz_heapify(sequence, less, order + first, count);
    sz_size_t end = count - 1;
    while (end > 0) {
        sz_u64_swap(order + first, order + first + end);
        end--;
        _sz_sift_down(sequence, less, order + first, 0, end);
    }
}

SZ_PUBLIC void sz_sort_introsort_recursion( //
    sz_sequence_t *sequence, sz_sequence_comparator_t less, sz_size_t first, sz_size_t last, sz_size_t depth) {

    sz_size_t length = last - first;
    switch (length) {
    case 0:
    case 1: return;
    case 2:
        if (less(sequence, sequence->order[first + 1], sequence->order[first]))
            sz_u64_swap(&sequence->order[first], &sequence->order[first + 1]);
        return;
    case 3: {
        sz_u64_t a = sequence->order[first];
        sz_u64_t b = sequence->order[first + 1];
        sz_u64_t c = sequence->order[first + 2];
        if (less(sequence, b, a)) sz_u64_swap(&a, &b);
        if (less(sequence, c, b)) sz_u64_swap(&c, &b);
        if (less(sequence, b, a)) sz_u64_swap(&a, &b);
        sequence->order[first] = a;
        sequence->order[first + 1] = b;
        sequence->order[first + 2] = c;
        return;
    }
    }
    // Until a certain length, the quadratic-complexity insertion-sort is fine
    if (length <= 16) {
        sz_sequence_t sub_seq = *sequence;
        sub_seq.order += first;
        sub_seq.count = length;
        sz_sort_insertion(&sub_seq, less);
        return;
    }

    // Fallback to N-logN-complexity heap-sort
    if (depth == 0) {
        _sz_heapsort(sequence, less, first, last);
        return;
    }

    --depth;

    // Median-of-three logic to choose pivot
    sz_size_t median = first + length / 2;
    if (less(sequence, sequence->order[median], sequence->order[first]))
        sz_u64_swap(&sequence->order[first], &sequence->order[median]);
    if (less(sequence, sequence->order[last - 1], sequence->order[first]))
        sz_u64_swap(&sequence->order[first], &sequence->order[last - 1]);
    if (less(sequence, sequence->order[median], sequence->order[last - 1]))
        sz_u64_swap(&sequence->order[median], &sequence->order[last - 1]);

    // Partition using the median-of-three as the pivot
    sz_u64_t pivot = sequence->order[median];
    sz_size_t left = first;
    sz_size_t right = last - 1;
    while (1) {
        while (less(sequence, sequence->order[left], pivot)) left++;
        while (less(sequence, pivot, sequence->order[right])) right--;
        if (left >= right) break;
        sz_u64_swap(&sequence->order[left], &sequence->order[right]);
        left++;
        right--;
    }

    // Recursively sort the partitions
    sz_sort_introsort_recursion(sequence, less, first, left, depth);
    sz_sort_introsort_recursion(sequence, less, right + 1, last, depth);
}

SZ_PUBLIC void sz_sort_introsort(sz_sequence_t *sequence, sz_sequence_comparator_t less) {
    if (sequence->count == 0) return;
    sz_size_t size_is_not_power_of_two = (sequence->count & (sequence->count - 1)) != 0;
    sz_size_t depth_limit = sz_size_log2i_nonzero(sequence->count) + size_is_not_power_of_two;
    sz_sort_introsort_recursion(sequence, less, 0, sequence->count, depth_limit);
}

SZ_PUBLIC void sz_sort_recursion( //
    sz_sequence_t *sequence, sz_size_t bit_idx, sz_size_t bit_max, sz_sequence_comparator_t comparator,
    sz_size_t partial_order_length) {

    if (!sequence->count) return;

    // Array of size one doesn't need sorting - only needs the prefix to be discarded.
    if (sequence->count == 1) {
        sz_u32_t *order_half_words = (sz_u32_t *)sequence->order;
        order_half_words[1] = 0;
        return;
    }

    // Partition a range of integers according to a specific bit value
    sz_size_t split = 0;
    sz_u64_t mask = (1ull << 63) >> bit_idx;

    // The clean approach would be to perform a single pass over the sequence.
    //
    //    while (split != sequence->count && !(sequence->order[split] & mask)) ++split;
    //    for (sz_size_t i = split + 1; i < sequence->count; ++i)
    //        if (!(sequence->order[i] & mask)) sz_u64_swap(sequence->order + i, sequence->order + split), ++split;
    //
    // This, however, doesn't take into account the high relative cost of writes and swaps.
    // To circumvent that, we can first count the total number entries to be mapped into either part.
    // And then walk through both parts, swapping the entries that are in the wrong part.
    // This would often lead to ~15% performance gain.
    sz_size_t count_with_bit_set = 0;
    for (sz_size_t i = 0; i != sequence->count; ++i) count_with_bit_set += (sequence->order[i] & mask) != 0;
    split = sequence->count - count_with_bit_set;

    // It's possible that the sequence is already partitioned.
    if (split != 0 && split != sequence->count) {
        // Use two pointers to efficiently reposition elements.
        // On pointer walks left-to-right from the start, and the other walks right-to-left from the end.
        sz_size_t left = 0;
        sz_size_t right = sequence->count - 1;
        while (1) {
            // Find the next element with the bit set on the left side.
            while (left < split && !(sequence->order[left] & mask)) ++left;
            // Find the next element without the bit set on the right side.
            while (right >= split && (sequence->order[right] & mask)) --right;
            // Swap the mispositioned elements.
            if (left < split && right >= split) {
                sz_u64_swap(sequence->order + left, sequence->order + right);
                ++left;
                --right;
            }
            else { break; }
        }
    }

    // Go down recursively.
    if (bit_idx < bit_max) {
        sz_sequence_t a = *sequence;
        a.count = split;
        sz_sort_recursion(&a, bit_idx + 1, bit_max, comparator, partial_order_length);

        sz_sequence_t b = *sequence;
        b.order += split;
        b.count -= split;
        sz_sort_recursion(&b, bit_idx + 1, bit_max, comparator, partial_order_length);
    }
    // Reached the end of recursion.
    else {
        // Discard the prefixes.
        sz_u32_t *order_half_words = (sz_u32_t *)sequence->order;
        for (sz_size_t i = 0; i != sequence->count; ++i) { order_half_words[i * 2 + 1] = 0; }

        sz_sequence_t a = *sequence;
        a.count = split;
        sz_sort_introsort(&a, comparator);

        sz_sequence_t b = *sequence;
        b.order += split;
        b.count -= split;
        sz_sort_introsort(&b, comparator);
    }
}

SZ_INTERNAL sz_bool_t _sz_sort_is_less(sz_sequence_t *sequence, sz_size_t i_key, sz_size_t j_key) {
    sz_cptr_t i_str = sequence->get_start(sequence, i_key);
    sz_cptr_t j_str = sequence->get_start(sequence, j_key);
    sz_size_t i_len = sequence->get_length(sequence, i_key);
    sz_size_t j_len = sequence->get_length(sequence, j_key);
    return (sz_bool_t)(sz_order_serial(i_str, i_len, j_str, j_len) == sz_less_k);
}

SZ_PUBLIC void sz_sort_partial(sz_sequence_t *sequence, sz_size_t partial_order_length) {

#if _SZ_IS_BIG_ENDIAN
    // TODO: Implement partial sort for big-endian systems. For now this sorts the whole thing.
    sz_unused(partial_order_length);
    sz_sort_introsort(sequence, (sz_sequence_comparator_t)_sz_sort_is_less);
#else

    // Export up to 4 bytes into the `sequence` bits themselves
    for (sz_size_t i = 0; i != sequence->count; ++i) {
        sz_cptr_t begin = sequence->get_start(sequence, sequence->order[i]);
        sz_size_t length = sequence->get_length(sequence, sequence->order[i]);
        length = length > 4u ? 4u : length;
        sz_ptr_t prefix = (sz_ptr_t)&sequence->order[i];
        for (sz_size_t j = 0; j != length; ++j) prefix[7 - j] = begin[j];
    }

    // Perform optionally-parallel radix sort on them
    sz_sort_recursion(sequence, 0, 32, (sz_sequence_comparator_t)_sz_sort_is_less, partial_order_length);
#endif
}

SZ_PUBLIC void sz_sort(sz_sequence_t *sequence) {
#if _SZ_IS_BIG_ENDIAN
    sz_sort_introsort(sequence, (sz_sequence_comparator_t)_sz_sort_is_less);
#else
    sz_sort_partial(sequence, sequence->count);
#endif
}

#pragma endregion // Serial Implementation

#ifdef __cplusplus
}
#endif // __cplusplus
#endif // STRINGZILLA_SORT_H_
