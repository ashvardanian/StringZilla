#include <stringzilla/stringzilla.h>

/**
 *  @brief  Helper, that swaps two 64-bit integers representing the order of elements in the sequence.
 */
void _sz_swap_order(sz_u64_t *a, sz_u64_t *b) {
    sz_u64_t t = *a;
    *a = *b;
    *b = t;
}

SZ_EXPORT sz_size_t sz_partition(sz_sequence_t *sequence, sz_sequence_predicate_t predicate) {

    sz_size_t matches = 0;
    while (matches != sequence->count && predicate(sequence, sequence->order[matches])) ++matches;

    for (sz_size_t i = matches + 1; i < sequence->count; ++i)
        if (predicate(sequence, sequence->order[i]))
            _sz_swap_order(sequence->order + i, sequence->order + matches), ++matches;

    return matches;
}

SZ_EXPORT void sz_merge(sz_sequence_t *sequence, sz_size_t partition, sz_sequence_comparator_t less) {

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

void sz_sort_insertion(sz_sequence_t *sequence, sz_sequence_comparator_t less) {
    sz_u64_t *keys = sequence->order;
    sz_size_t keys_count = sequence->count;
    for (sz_size_t i = 1; i < keys_count; i++) {
        sz_u64_t i_key = keys[i];
        sz_size_t j = i;
        for (; j > 0 && less(sequence, i_key, keys[j - 1]); --j) keys[j] = keys[j - 1];
        keys[j] = i_key;
    }
}

void _sz_sift_down(sz_sequence_t *sequence, sz_sequence_comparator_t less, sz_u64_t *order, sz_size_t start,
                   sz_size_t end) {
    sz_size_t root = start;
    while (2 * root + 1 <= end) {
        sz_size_t child = 2 * root + 1;
        if (child + 1 <= end && less(sequence, order[child], order[child + 1])) { child++; }
        if (!less(sequence, order[root], order[child])) { return; }
        _sz_swap_order(order + root, order + child);
        root = child;
    }
}

void _sz_heapify(sz_sequence_t *sequence, sz_sequence_comparator_t less, sz_u64_t *order, sz_size_t count) {
    sz_size_t start = (count - 2) / 2;
    while (1) {
        _sz_sift_down(sequence, less, order, start, count - 1);
        if (start == 0) return;
        start--;
    }
}

void _sz_heapsort(sz_sequence_t *sequence, sz_sequence_comparator_t less, sz_size_t first, sz_size_t last) {
    sz_u64_t *order = sequence->order;
    sz_size_t count = last - first;
    _sz_heapify(sequence, less, order + first, count);
    sz_size_t end = count - 1;
    while (end > 0) {
        _sz_swap_order(order + first, order + first + end);
        end--;
        _sz_sift_down(sequence, less, order + first, 0, end);
    }
}

void _sz_introsort(sz_sequence_t *sequence, sz_sequence_comparator_t less, sz_size_t first, sz_size_t last,
                   sz_size_t depth) {

    sz_size_t length = last - first;
    switch (length) {
    case 0:
    case 1: return;
    case 2:
        if (less(sequence, sequence->order[first + 1], sequence->order[first]))
            _sz_swap_order(&sequence->order[first], &sequence->order[first + 1]);
        return;
    case 3: {
        sz_u64_t a = sequence->order[first];
        sz_u64_t b = sequence->order[first + 1];
        sz_u64_t c = sequence->order[first + 2];
        if (less(sequence, b, a)) _sz_swap_order(&a, &b);
        if (less(sequence, c, b)) _sz_swap_order(&c, &b);
        if (less(sequence, b, a)) _sz_swap_order(&a, &b);
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
        _sz_swap_order(&sequence->order[first], &sequence->order[median]);
    if (less(sequence, sequence->order[last - 1], sequence->order[first]))
        _sz_swap_order(&sequence->order[first], &sequence->order[last - 1]);
    if (less(sequence, sequence->order[median], sequence->order[last - 1]))
        _sz_swap_order(&sequence->order[median], &sequence->order[last - 1]);

    // Partition using the median-of-three as the pivot
    sz_u64_t pivot = sequence->order[median];
    sz_size_t left = first;
    sz_size_t right = last - 1;
    while (1) {
        while (less(sequence, sequence->order[left], pivot)) left++;
        while (less(sequence, pivot, sequence->order[right])) right--;
        if (left >= right) break;
        _sz_swap_order(&sequence->order[left], &sequence->order[right]);
        left++;
        right--;
    }

    // Recursively sort the partitions
    _sz_introsort(sequence, less, first, left, depth);
    _sz_introsort(sequence, less, right + 1, last, depth);
}

SZ_EXPORT void sz_sort_introsort(sz_sequence_t *sequence, sz_sequence_comparator_t less) {
    sz_size_t depth_limit = 2 * sz_log2i(sequence->count);
    _sz_introsort(sequence, less, 0, sequence->count, depth_limit);
}

void _sz_sort_recursion( //
    sz_sequence_t *sequence, sz_size_t bit_idx, sz_size_t bit_max, sz_sequence_comparator_t comparator,
    sz_size_t partial_order_length) {

    if (!sequence->count) return;

    // Partition a range of integers according to a specific bit value
    sz_size_t split = 0;
    {
        sz_u64_t mask = (1ull << 63) >> bit_idx;
        while (split != sequence->count && !(sequence->order[split] & mask)) ++split;
        for (sz_size_t i = split + 1; i < sequence->count; ++i)
            if (!(sequence->order[i] & mask)) _sz_swap_order(sequence->order + i, sequence->order + split), ++split;
    }

    // Go down recursively
    if (bit_idx < bit_max) {
        sz_sequence_t a = *sequence;
        a.count = split;
        _sz_sort_recursion(&a, bit_idx + 1, bit_max, comparator, partial_order_length);

        sz_sequence_t b = *sequence;
        b.order += split;
        b.count -= split;
        _sz_sort_recursion(&b, bit_idx + 1, bit_max, comparator, partial_order_length);
    }
    // Reached the end of recursion
    else {
        // Discard the prefixes
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

sz_bool_t _sz_sort_is_less(sz_sequence_t *sequence, sz_size_t i_key, sz_size_t j_key) {
    sz_cptr_t i_str = sequence->get_start(sequence, i_key);
    sz_cptr_t j_str = sequence->get_start(sequence, j_key);
    sz_size_t i_len = sequence->get_length(sequence, i_key);
    sz_size_t j_len = sequence->get_length(sequence, j_key);
    return sz_order(i_str, j_str, sz_min_of_two(i_len, j_len)) > 0 ? 0 : 1;
}

SZ_EXPORT void sz_sort_partial(sz_sequence_t *sequence, sz_size_t partial_order_length) {

    // Export up to 4 bytes into the `sequence` bits themselves
    for (sz_size_t i = 0; i != sequence->count; ++i) {
        sz_cptr_t begin = sequence->get_start(sequence, sequence->order[i]);
        sz_size_t length = sequence->get_length(sequence, sequence->order[i]);
        length = length > 4ull ? 4ull : length;
        sz_ptr_t prefix = (sz_ptr_t)&sequence->order[i];
        for (sz_size_t j = 0; j != length; ++j) prefix[7 - j] = begin[j];
    }

    // Perform optionally-parallel radix sort on them
    _sz_sort_recursion(sequence, 0, 32, (sz_sequence_comparator_t)_sz_sort_is_less, partial_order_length);
}

SZ_EXPORT void sz_sort(sz_sequence_t *sequence) { sz_sort_partial(sequence, sequence->count); }