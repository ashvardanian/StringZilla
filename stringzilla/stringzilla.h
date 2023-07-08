#ifndef STRINGZILLA_H_
#define STRINGZILLA_H_

#include <stdint.h> // `uint8_t`
#include <stddef.h> // `size_t`
#include <string.h> // `memcpy`

#if defined(__AVX2__)
#include <x86intrin.h>
#endif
#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif

#ifdef _MSC_VER
#include <intrin.h>
#define popcount64 __popcnt64
#define ctz64 _tzcnt_u64
#else
#define popcount64 __builtin_popcountll
#define ctz64 __builtin_ctzll
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef uint32_t strzl_anomaly_t;

inline static size_t strzl_divide_round_up(size_t x, size_t divisor) { return (x + (divisor - 1)) / divisor; }

/**
 *  @brief This is a faster alternative to `strncmp(a, b, len) == 0`.
 */
inline static bool strzl_equal(char const *a, char const *b, size_t len) {
    char const *const a_end = a + len;
    while (a != a_end && *a == *b)
        a++, b++;
    return a_end == a;
}

typedef struct strzl_haystack_t {
    char const *ptr;
    size_t len;
} strzl_haystack_t;

typedef struct strzl_needle_t {
    char const *ptr;
    size_t len;
    size_t anomaly_offset;
} strzl_needle_t;

/**
 *  @brief  A naive subtring matching algorithm with O(|h|*|n|) comparisons.
 *          Matching performance fluctuates between 200 MB/s and 2 GB/s.
 */
inline static size_t strzl_naive_count_char(strzl_haystack_t h, char n) {

    size_t result = 0;
    char const *h_ptr = h.ptr;
    char const *h_end = h.ptr + h.len;

    for (; (uint64_t)h_ptr % 8 != 0 && h_ptr < h_end; ++h_ptr)
        result += *h_ptr == n;

    // This code simulates hyperscalar execution, comparing 8 characters at a time.
    uint64_t nnnnnnnn = n;
    nnnnnnnn |= nnnnnnnn << 8;
    nnnnnnnn |= nnnnnnnn << 16;
    nnnnnnnn |= nnnnnnnn << 32;
    for (; h_ptr + 8 <= h_end; h_ptr += 8) {
        uint64_t h_slice = *(uint64_t const *)h_ptr;
        uint64_t match_indicators = ~(h_slice ^ nnnnnnnn);
        match_indicators &= match_indicators >> 1;
        match_indicators &= match_indicators >> 2;
        match_indicators &= match_indicators >> 4;
        match_indicators &= 0x0101010101010101;
        result += popcount64(match_indicators);
    }

    for (; h_ptr < h_end; ++h_ptr)
        result += *h_ptr == n;
    return result;
}

inline static size_t strzl_naive_find_char(strzl_haystack_t h, char n) {

    char const *h_ptr = h.ptr;
    char const *h_end = h.ptr + h.len;

    for (; (uint64_t)h_ptr % 8 != 0 && h_ptr < h_end; ++h_ptr)
        if (*h_ptr == n)
            return h_ptr - h.ptr;

    // This code simulates hyperscalar execution, analyzing 8 offsets at a time.
    uint64_t nnnnnnnn = n;
    nnnnnnnn |= nnnnnnnn << 8;
    nnnnnnnn |= nnnnnnnn << 16;
    nnnnnnnn |= nnnnnnnn << 32;
    for (; h_ptr + 8 <= h_end; h_ptr += 8) {
        uint64_t h_slice = *(uint64_t const *)h_ptr;
        uint64_t match_indicators = ~(h_slice ^ nnnnnnnn);
        match_indicators &= match_indicators >> 1;
        match_indicators &= match_indicators >> 2;
        match_indicators &= match_indicators >> 4;
        match_indicators &= 0x0101010101010101;

        if (match_indicators != 0)
            return h_ptr - h.ptr + ctz64(match_indicators) / 8;
    }

    for (; h_ptr < h_end; ++h_ptr)
        if (*h_ptr == n)
            return h_ptr - h.ptr;
    return h.len;
}

inline static size_t strzl_naive_find_2chars(strzl_haystack_t h, char const *n) {

    char const *h_ptr = h.ptr;
    char const *h_end = h.ptr + h.len;

    // This code simulates hyperscalar execution, analyzing 7 offsets at a time.
    uint64_t nnnn = (uint64_t(n[0]) << 0) | (uint64_t(n[1]) << 8);
    nnnn |= nnnn << 16;
    nnnn |= nnnn << 32;
    uint64_t h_slice;
    for (; h_ptr + 8 <= h_end; h_ptr += 7) {
        memcpy(&h_slice, h_ptr, 8);
        uint64_t even_indicators = ~(h_slice ^ nnnn);
        uint64_t odd_indicators = ~((h_slice << 8) ^ nnnn);
        // For every even match - 2 char (16 bits) must be identical.
        even_indicators &= even_indicators >> 1;
        even_indicators &= even_indicators >> 2;
        even_indicators &= even_indicators >> 4;
        even_indicators &= even_indicators >> 8;
        even_indicators &= 0x0001000100010001;
        // For every odd match - 2 char (16 bits) must be identical.
        odd_indicators &= odd_indicators >> 1;
        odd_indicators &= odd_indicators >> 2;
        odd_indicators &= odd_indicators >> 4;
        odd_indicators &= odd_indicators >> 8;
        odd_indicators &= 0x0001000100010000;

        if (even_indicators + odd_indicators) {
            uint64_t match_indicators = even_indicators | (odd_indicators >> 8);
            return h_ptr - h.ptr + ctz64(match_indicators) / 8;
        }
    }

    for (; h_ptr + 2 <= h_end; ++h_ptr)
        if (h_ptr[0] == n[0] && h_ptr[1] == n[1])
            return h_ptr - h.ptr;
    return h.len;
}

inline static size_t strzl_naive_find_3chars(strzl_haystack_t h, char const *n) {

    char const *h_ptr = h.ptr;
    char const *h_end = h.ptr + h.len;

    // This code simulates hyperscalar execution, analyzing 6 offsets at a time.
    // We have two unused bytes at the end.
    uint64_t nn = uint64_t(n[0] << 0) | (uint64_t(n[1]) << 8) | (uint64_t(n[2]) << 16);
    nn |= nn << 24;
    nn <<= 16;

    for (; h_ptr + 8 <= h_end; h_ptr += 6) {
        uint64_t h_slice;
        memcpy(&h_slice, h_ptr, 8);
        uint64_t first_indicators = ~(h_slice ^ nn);
        uint64_t second_indicators = ~((h_slice << 8) ^ nn);
        uint64_t third_indicators = ~((h_slice << 16) ^ nn);
        // For every first match - 3 chars (24 bits) must be identical.
        // For that merge every byte state and then combine those three-way.
        first_indicators &= first_indicators >> 1;
        first_indicators &= first_indicators >> 2;
        first_indicators &= first_indicators >> 4;
        first_indicators =
            (first_indicators >> 16) & (first_indicators >> 8) & (first_indicators >> 0) & 0x0000010000010000;

        // For every second match - 3 chars (24 bits) must be identical.
        // For that merge every byte state and then combine those three-way.
        second_indicators &= second_indicators >> 1;
        second_indicators &= second_indicators >> 2;
        second_indicators &= second_indicators >> 4;
        second_indicators =
            (second_indicators >> 16) & (second_indicators >> 8) & (second_indicators >> 0) & 0x0000010000010000;

        // For every third match - 3 chars (24 bits) must be identical.
        // For that merge every byte state and then combine those three-way.
        third_indicators &= third_indicators >> 1;
        third_indicators &= third_indicators >> 2;
        third_indicators &= third_indicators >> 4;
        third_indicators =
            (third_indicators >> 16) & (third_indicators >> 8) & (third_indicators >> 0) & 0x0000010000010000;

        uint64_t match_indicators = first_indicators | (second_indicators >> 8) | (third_indicators >> 16);
        if (match_indicators != 0)
            return h_ptr - h.ptr + ctz64(match_indicators) / 8;
    }

    for (; h_ptr + 3 <= h_end; ++h_ptr)
        if (h_ptr[0] == n[0] && h_ptr[1] == n[1] && h_ptr[2] == n[2])
            return h_ptr - h.ptr;
    return h.len;
}

inline static size_t strzl_naive_find_4chars(strzl_haystack_t h, char const *n) {

    char const *h_ptr = h.ptr;
    char const *h_end = h.ptr + h.len;

    // This code simulates hyperscalar execution, analyzing 4 offsets at a time.
    uint64_t nn = uint64_t(n[0] << 0) | (uint64_t(n[1]) << 8) | (uint64_t(n[2]) << 16) | (uint64_t(n[3]) << 24);
    nn |= nn << 32;
    nn = nn;

    //
    uint8_t lookup[16] = {0};
    lookup[0b0010] = lookup[0b0110] = lookup[0b1010] = lookup[0b1110] = 1;
    lookup[0b0100] = lookup[0b1100] = 2;
    lookup[0b1000] = 3;

    // We can perform 5 comparisons per load, but it's easir to perform 4, minimizing the size of the lookup table.
    for (; h_ptr + 8 <= h_end; h_ptr += 4) {
        uint64_t h_slice;
        memcpy(&h_slice, h_ptr, 8);
        uint64_t h01 = (h_slice & 0x00000000FFFFFFFF) | ((h_slice & 0x000000FFFFFFFF00) << 24);
        uint64_t h23 = ((h_slice & 0x0000FFFFFFFF0000) >> 16) | ((h_slice & 0x00FFFFFFFF000000) << 8);
        uint64_t h01_indicators = ~(h01 ^ nn);
        uint64_t h23_indicators = ~(h23 ^ nn);

        // For every first match - 4 chars (32 bits) must be identical.
        h01_indicators &= h01_indicators >> 1;
        h01_indicators &= h01_indicators >> 2;
        h01_indicators &= h01_indicators >> 4;
        h01_indicators &= h01_indicators >> 8;
        h01_indicators &= h01_indicators >> 16;
        h01_indicators &= 0x0000000100000001;

        // For every first match - 4 chars (32 bits) must be identical.
        h23_indicators &= h23_indicators >> 1;
        h23_indicators &= h23_indicators >> 2;
        h23_indicators &= h23_indicators >> 4;
        h23_indicators &= h23_indicators >> 8;
        h23_indicators &= h23_indicators >> 16;
        h23_indicators &= 0x0000000100000001;

        if (h01_indicators + h23_indicators) {
            // Assuming we have performed 4 comparisons, we can only have 2^4=16 outcomes.
            // Which is small enought for a lookup table.
            uint8_t match_indicators =
                (h01_indicators >> 31) | (h01_indicators << 0) | (h23_indicators >> 29) | (h23_indicators << 2);
            return h_ptr - h.ptr + lookup[match_indicators];
        }
    }

    for (; h_ptr + 4 <= h_end; ++h_ptr)
        if (h_ptr[0] == n[0] && h_ptr[1] == n[1] && h_ptr[2] == n[2] && h_ptr[3] == n[3])
            return h_ptr - h.ptr;
    return h.len;
}

/**
 *  @brief  Trivial substring search with scalar code. Instead of comparing characters one-by-one
 *          it compares 4-byte anomalies first, most commonly prefixes. It's computationally cheaper.
 *          Matching performance fluctuates between 1 GB/s and 3,5 GB/s per core.
 */
inline static size_t strzl_naive_find_substr(strzl_haystack_t h, strzl_needle_t n) {

    if (h.len < n.len)
        return h.len;

    char const *h_ptr = h.ptr;
    char const *const h_end = h.ptr + h.len;
    switch (n.len) {
    case 0: return 0;
    case 1: return strzl_naive_find_char(h, *n.ptr);
    case 2: return strzl_naive_find_2chars(h, n.ptr);
    case 3: return strzl_naive_find_3chars(h, n.ptr);
    case 4: return strzl_naive_find_4chars(h, n.ptr);
    default: {
        strzl_anomaly_t n_anomaly, h_anomaly;
        size_t const n_suffix_len = n.len - 4 - n.anomaly_offset;
        char const *n_suffix_ptr = n.ptr + 4 + n.anomaly_offset;
        memcpy(&n_anomaly, n.ptr + n.anomaly_offset, 4);

        h_ptr += n.anomaly_offset;
        for (; h_ptr + n.len <= h_end; h_ptr++) {
            memcpy(&h_anomaly, h_ptr, 4);
            if (h_anomaly == n_anomaly)                                                 // Match anomaly.
                if (strzl_equal(h_ptr + 4, n_suffix_ptr, n_suffix_len))                 // Match suffix.
                    if (strzl_equal(h_ptr - n.anomaly_offset, n.ptr, n.anomaly_offset)) // Match prefix.
                        return h_ptr - h.ptr - n.anomaly_offset;
        }
        return h.len;
    }
    }
}

#if defined(__AVX2__)

/**
 *  @brief  Substring-search implementation, leveraging x86 AVX2 instrinsics and speculative
 *          execution capabilities on modern CPUs. Performing 4 unaligned vector loads per cycle
 *          was practically more efficient than loading once and shifting around, as introduces
 *          less data dependencies.
 */
size_t strzl_avx2_find_substr(strzl_haystack_t h, strzl_needle_t n) {

    if (n.len < 4)
        return strzl_naive_find_substr(h, n);

    // Precomputed constants.
    char const *const h_end = h.ptr + h.len;
    __m256i const n_prefix = _mm256_set1_epi32(*(strzl_anomaly_t const *)(n.ptr));

    // Top level for-loop changes dramatically.
    // In sequentail computing model for 32 offsets we would do:
    //  + 32 comparions.
    //  + 32 branches.
    // In vectorized computations models:
    //  + 4 vectorized comparisons.
    //  + 4 movemasks.
    //  + 3 bitwise ANDs.
    //  + 1 heavy (but very unlikely) branch.
    char const *h_ptr = h.ptr;
    for (; (h_ptr + n.len + 32) <= h_end; h_ptr += 32) {

        // Performing many unaligned loads ends up being faster than loading once and shuffling around.
        __m256i h0_prefixes = _mm256_loadu_si256((__m256i const *)(h_ptr));
        int masks0 = _mm256_movemask_epi8(_mm256_cmpeq_epi32(h0_prefixes, n_prefix));
        __m256i h1_prefixes = _mm256_loadu_si256((__m256i const *)(h_ptr + 1));
        int masks1 = _mm256_movemask_epi8(_mm256_cmpeq_epi32(h1_prefixes, n_prefix));
        __m256i h2_prefixes = _mm256_loadu_si256((__m256i const *)(h_ptr + 2));
        int masks2 = _mm256_movemask_epi8(_mm256_cmpeq_epi32(h2_prefixes, n_prefix));
        __m256i h3_prefixes = _mm256_loadu_si256((__m256i const *)(h_ptr + 3));
        int masks3 = _mm256_movemask_epi8(_mm256_cmpeq_epi32(h3_prefixes, n_prefix));

        if (masks0 | masks1 | masks2 | masks3) {
            for (size_t i = 0; i < 32; i++) {
                if (strzl_equal(h_ptr + i, n.ptr, n.len))
                    return i + (h_ptr - h.ptr);
            }
        }
    }

    // Don't forget the last (up to 35) characters.
    size_t tail_len = h_end - h_ptr;
    size_t tail_match = strzl_naive_find_substr({h_ptr, tail_len}, n);
    return h_ptr + tail_match - h.ptr;
}

#endif // x86 AVX2

#if defined(__ARM_NEON)

/**
 *  @brief  Character-counting routine, leveraging Arm Neon instrinsics and checking 16 characters at once.
 */
inline static size_t strzl_neon_count_char(strzl_haystack_t h, char n) {
    char const *const h_end = h.ptr + h.len;

    // The plan is simple, skim through the misaligned part of the string.
    char const *aligned_start = (char const *)(strzl_divide_round_up((size_t)h.ptr, 16) * 16);
    size_t misaligned_len = (size_t)(aligned_start - h.ptr) < h.len ? (size_t)(aligned_start - h.ptr) : h.len;
    size_t result = strzl_naive_count_char({h.ptr, misaligned_len}, n);
    if (h.ptr + misaligned_len >= h_end)
        return result;

    // Count matches in the aligned part.
    char const *h_ptr = aligned_start;
    uint8x16_t n_vector = vld1q_dup_u8((uint8_t const *)&n);
    for (; (h_ptr + 16) <= h_end; h_ptr += 16) {
        uint8x16_t masks = vceqq_u8(vld1q_u8((uint8_t const *)h_ptr), n_vector);
        uint64x2_t masks64x2 = vreinterpretq_u64_u8(masks);
        result += popcount64(vgetq_lane_u64(masks64x2, 0)) / 8;
        result += popcount64(vgetq_lane_u64(masks64x2, 1)) / 8;
    }

    // Count matches in the misaligned tail.
    size_t tail_len = h_end - h_ptr;
    result += strzl_naive_count_char({h_ptr, tail_len}, n);
    return result;
}

/**
 *  @brief  Substring-search implementation, leveraging Arm Neon instrinsics and speculative
 *          execution capabilities on modern CPUs. Performing 4 unaligned vector loads per cycle
 *          was practically more efficient than loading once and shifting around, as introduces
 *          less data dependencies.
 */
inline static size_t strzl_neon_find_substr(strzl_haystack_t h, strzl_needle_t n) {

    if (n.len < 4)
        return strzl_naive_find_substr(h, n);

    // Precomputed constants.
    char const *const h_end = h.ptr + h.len;
    uint32x4_t const n_prefix = vld1q_dup_u32((strzl_anomaly_t const *)(n.ptr));

    char const *h_ptr = h.ptr;
    for (; (h_ptr + n.len + 16) <= h_end; h_ptr += 16) {

        uint32x4_t masks0 = vceqq_u32(vld1q_u32((strzl_anomaly_t const *)(h_ptr)), n_prefix);
        uint32x4_t masks1 = vceqq_u32(vld1q_u32((strzl_anomaly_t const *)(h_ptr + 1)), n_prefix);
        uint32x4_t masks2 = vceqq_u32(vld1q_u32((strzl_anomaly_t const *)(h_ptr + 2)), n_prefix);
        uint32x4_t masks3 = vceqq_u32(vld1q_u32((strzl_anomaly_t const *)(h_ptr + 3)), n_prefix);

        // Extracting matches from masks:
        // vmaxvq_u32 (only a64)
        // vgetq_lane_u32 (all)
        // vorrq_u32 (all)
        uint32x4_t masks = vorrq_u32(vorrq_u32(masks0, masks1), vorrq_u32(masks2, masks3));
        uint64x2_t masks64x2 = vreinterpretq_u64_u32(masks);
        bool has_match = vgetq_lane_u64(masks64x2, 0) | vgetq_lane_u64(masks64x2, 1);

        if (has_match) {
            for (size_t i = 0; i < 16; i++) {
                if (strzl_equal(h_ptr + i, n.ptr, n.len))
                    return i + (h_ptr - h.ptr);
            }
        }
    }

    // Don't forget the last (up to 16+3=19) characters.
    size_t tail_len = h_end - h_ptr;
    size_t tail_match = strzl_naive_find_substr({h_ptr, tail_len}, n);
    return h_ptr + tail_match - h.ptr;
}

#endif // Arm Neon

typedef char const *(*stringzilla_array_get_start_t)(void const *, size_t);
typedef size_t (*stringzilla_array_get_length_t)(void const *, size_t);
typedef bool (*stringzilla_array_predicate_t)(char const *, size_t);

inline static bool strzl_has_under_one_char(char const *, size_t len) { return len < 1; }
inline static bool strzl_has_under_two_chars(char const *, size_t len) { return len < 2; }
inline static bool strzl_has_under_three_chars(char const *, size_t len) { return len < 3; }
inline static bool strzl_has_under_four_chars(char const *, size_t len) { return len < 4; }

inline static bool strzl_less_one_char(char const *a, size_t, char const *b, size_t) { return *a < *b; }

inline static bool strzl_less_two_chars(char const *a, size_t, char const *b, size_t) {
    uint16_t aa, bb;
    memcpy(&aa, a, 2);
    memcpy(&bb, b, 2);
    return aa < bb;
}

inline static bool strzl_less_three_chars(char const *a, size_t, char const *b, size_t) {
    uint32_t aaa = 0, bbb = 0;
    memcpy(&aaa, a, 3);
    memcpy(&bbb, b, 3);
    return aaa < bbb;
}

inline static bool strzl_less_four_chars(char const *a, size_t, char const *b, size_t) {
    uint32_t aaaa, bbbb;
    memcpy(&aaaa, a, 4);
    memcpy(&bbbb, b, 4);
    return aaaa < bbbb;
}

inline static bool strzl_less_entire(char const *a, size_t an, char const *b, size_t bn) {
    size_t n = an < bn ? an : bn;
    char const *end = a + an;
    while (a != end && *a == *b)
        ++a, ++b;
    if (a == end)
        return an < bn;
    return *a < *b;
}

inline static void strzl_swap(size_t *a, size_t *b) {
    *a ^= *b;
    *b ^= *a;
    *a ^= *b;
}

inline static size_t strzl_partition( //
    void const *array,
    size_t count,
    stringzilla_array_get_start_t get_start,
    stringzilla_array_get_length_t get_length,
    stringzilla_array_predicate_t predicate,
    size_t *order) {

    size_t matches = 0;
    while (matches != count && predicate(get_start(array, order[matches]), get_length(array, order[matches])))
        ++matches;

    for (size_t i = matches + 1; i < count; ++i)
            if (predicate(get_start(array, order[i]), get_length(array, order[i])))
            strzl_swap(order + i, order + matches), ++matches;

    return matches;
}

inline static void strzl_qsort( //
    void const *array,
    size_t count,
    stringzilla_array_get_start_t begin,
    stringzilla_array_get_length_t length,
    bool (*less)(char const *, size_t, char const *, size_t),
    size_t *order,
    ssize_t const l,
    ssize_t const h) {

    if (l >= h)
        return;

    ssize_t i = (l - 1);
    for (ssize_t j = l; j < h; j++) {
        if (less( //
                begin(array, order[j]),
                length(array, order[j]),
                begin(array, order[h]),
                length(array, order[h]))) {
            i++;
            strzl_swap(order + i, order + j);
        }
    }
    strzl_swap(order + i + 1, order + h);
    ssize_t p = i + 1;

    strzl_qsort(array, count, begin, length, less, order, l, p - 1);
    strzl_qsort(array, count, begin, length, less, order, p + 1, h);
        }

inline static size_t _strzl_integers_partition(size_t *array, size_t count, size_t bit_idx) {

    size_t split = 0;
    size_t mask = (1ul << 63) >> bit_idx;
    while (split != count && (array[split] & mask))
        ++split;

    for (size_t i = split + 1; i < count; ++i)
        if (array[i] & mask)
            strzl_swap(array + i, array + split), ++split;

    return split;
            }

void _strzl_integers_qsort(size_t *array, ssize_t count) {

    ssize_t l = 0;
    ssize_t h = count;
    if (h <= 0)
        return;

    ssize_t i = (l - 1);
    for (ssize_t j = l; j < h; j++) {
        if (*(uint32_t *)(array + j) < *(uint32_t *)(array + h)) {
            i++;
            strzl_swap(array + i, array + j);
        }
    }
    strzl_swap(array + i + 1, array + h);
    ssize_t p = i + 1;

    _strzl_integers_qsort(array + l, p - 1);
    _strzl_integers_qsort(array + p + 1, h);
}

inline static void _strzl_sort_radix_recursion( //
    void const *array,
    size_t array_size,
    stringzilla_array_get_start_t get_begin,
    stringzilla_array_get_length_t get_length,
    size_t *order,
    size_t order_size,
    size_t bit_idx,
    size_t bit_max) {

    if (!order_size)
        return;

    size_t split = _strzl_integers_partition(order, order_size, bit_idx);
    if (bit_idx < bit_max) {
        _strzl_sort_radix_recursion( //
            array,
            array_size,
            get_begin,
            get_length,
            order,
            split,
            bit_idx + 1,
            bit_max);
        _strzl_sort_radix_recursion( //
            array,
            array_size,
            get_begin,
            get_length,
            order + split,
            order_size - split,
            bit_idx + 1,
            bit_max);
}
    else {
        // Discard the prefixes
        for (size_t i = 0; i != order_size; ++i)
            memset(&order[i], 0, 4ul);

        // Perform sorts on smaller chunks instead of the whole array
        auto sorter = [=](size_t i, size_t j) {
            auto a = std::string_view(get_begin(array, i), get_length(array, i));
            auto b = std::string_view(get_begin(array, j), get_length(array, j));
            return a < b;
        };
        std::sort(order, order + split, sorter);
        std::sort(order + split, order + order_size, sorter);
    }
}

/**
 *  @brief  Sorting algorithm, combining Radix Sort for the first 32 bits of every word
 *          and a follow-up Quick Sort on resulting structure.
 */
inline static void strzl_sort( //
    void const *array,
    size_t array_size,
    stringzilla_array_get_start_t get_begin,
    stringzilla_array_get_length_t get_length,
    size_t *order) {

    // Export up to 4 bytes into the `order` bits themselves
    for (size_t i = 0; i != array_size; ++i)
        memcpy( //
            &order[i],
            get_begin(array, order[i]),
            get_length(array, order[i]) > 4ul ? 4ul : get_length(array, order[i]));

    // Perform optionally-parallel radix sort on them
    _strzl_sort_radix_recursion(array, array_size, get_begin, get_length, order, array_size, 0, 32);
}

#ifdef __cplusplus
}
#endif

#undef popcount64
#undef ctz64

#endif // STRINGZILLA_H_