#ifndef STRINGZILLA_H_
#define STRINGZILLA_H_

#include <stdint.h> // `uint8_t`
#include <stddef.h> // `size_t`
#include <string.h> // `memcpy`
#include <stdlib.h> // `qsort_r`
#include <search.h> // `qsort_s`
#include <ctype.h>  // `tolower`

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
#define strncasecmp _strnicmp
#define strcasecmp _stricmp
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
    char const *text = h.ptr;
    char const *end = h.ptr + h.len;

    for (; (uint64_t)text % 8 != 0 && text < end; ++text)
        result += *text == n;

    // This code simulates hyperscalar execution, comparing 8 characters at a time.
    uint64_t nnnnnnnn = n;
    nnnnnnnn |= nnnnnnnn << 8;
    nnnnnnnn |= nnnnnnnn << 16;
    nnnnnnnn |= nnnnnnnn << 32;
    for (; text + 8 <= end; text += 8) {
        uint64_t text_slice = *(uint64_t const *)text;
        uint64_t match_indicators = ~(text_slice ^ nnnnnnnn);
        match_indicators &= match_indicators >> 1;
        match_indicators &= match_indicators >> 2;
        match_indicators &= match_indicators >> 4;
        match_indicators &= 0x0101010101010101;
        result += popcount64(match_indicators);
    }

    for (; text < end; ++text)
        result += *text == n;
    return result;
}

inline static size_t strzl_naive_find_char(strzl_haystack_t h, char n) {

    char const *text = h.ptr;
    char const *end = h.ptr + h.len;

    for (; (uint64_t)text % 8 != 0 && text < end; ++text)
        if (*text == n)
            return text - h.ptr;

    // This code simulates hyperscalar execution, analyzing 8 offsets at a time.
    uint64_t nnnnnnnn = n;
    nnnnnnnn |= nnnnnnnn << 8;  // broadcast `n` into `nnnnnnnn`
    nnnnnnnn |= nnnnnnnn << 16; // broadcast `n` into `nnnnnnnn`
    nnnnnnnn |= nnnnnnnn << 32; // broadcast `n` into `nnnnnnnn`
    for (; text + 8 <= end; text += 8) {
        uint64_t text_slice = *(uint64_t const *)text;
        uint64_t match_indicators = ~(text_slice ^ nnnnnnnn);
        match_indicators &= match_indicators >> 1;
        match_indicators &= match_indicators >> 2;
        match_indicators &= match_indicators >> 4;
        match_indicators &= 0x0101010101010101;

        if (match_indicators != 0)
            return text - h.ptr + ctz64(match_indicators) / 8;
    }

    for (; text < end; ++text)
        if (*text == n)
            return text - h.ptr;
    return h.len;
}

inline static size_t strzl_naive_find_2chars(strzl_haystack_t h, char const *n) {

    char const *text = h.ptr;
    char const *end = h.ptr + h.len;

    // This code simulates hyperscalar execution, analyzing 7 offsets at a time.
    uint64_t nnnn = ((uint64_t)(n[0]) << 0) | ((uint64_t)(n[1]) << 8); // broadcast `n` into `nnnn`
    nnnn |= nnnn << 16;                                                // broadcast `n` into `nnnn`
    nnnn |= nnnn << 32;                                                // broadcast `n` into `nnnn`
    uint64_t text_slice;
    for (; text + 8 <= end; text += 7) {
        memcpy(&text_slice, text, 8);
        uint64_t even_indicators = ~(text_slice ^ nnnn);
        uint64_t odd_indicators = ~((text_slice << 8) ^ nnnn);
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
            return text - h.ptr + ctz64(match_indicators) / 8;
        }
    }

    for (; text + 2 <= end; ++text)
        if (text[0] == n[0] && text[1] == n[1])
            return text - h.ptr;
    return h.len;
}

inline static size_t strzl_naive_find_3chars(strzl_haystack_t h, char const *n) {

    char const *text = h.ptr;
    char const *end = h.ptr + h.len;

    // This code simulates hyperscalar execution, analyzing 6 offsets at a time.
    // We have two unused bytes at the end.
    uint64_t nn = (uint64_t)(n[0] << 0) | ((uint64_t)(n[1]) << 8) | ((uint64_t)(n[2]) << 16); // broadcast `n` into `nn`
    nn |= nn << 24;                                                                           // broadcast `n` into `nn`
    nn <<= 16;                                                                                // broadcast `n` into `nn`

    for (; text + 8 <= end; text += 6) {
        uint64_t text_slice;
        memcpy(&text_slice, text, 8);
        uint64_t first_indicators = ~(text_slice ^ nn);
        uint64_t second_indicators = ~((text_slice << 8) ^ nn);
        uint64_t third_indicators = ~((text_slice << 16) ^ nn);
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
            return text - h.ptr + ctz64(match_indicators) / 8;
    }

    for (; text + 3 <= end; ++text)
        if (text[0] == n[0] && text[1] == n[1] && text[2] == n[2])
            return text - h.ptr;
    return h.len;
}

inline static size_t strzl_naive_find_4chars(strzl_haystack_t h, char const *n) {

    char const *text = h.ptr;
    char const *end = h.ptr + h.len;

    // This code simulates hyperscalar execution, analyzing 4 offsets at a time.
    uint64_t nn = (uint64_t)(n[0] << 0) | ((uint64_t)(n[1]) << 8) | ((uint64_t)(n[2]) << 16) | ((uint64_t)(n[3]) << 24);
    nn |= nn << 32;

    //
    uint8_t lookup[16] = {0};
    lookup[0b0010] = lookup[0b0110] = lookup[0b1010] = lookup[0b1110] = 1;
    lookup[0b0100] = lookup[0b1100] = 2;
    lookup[0b1000] = 3;

    // We can perform 5 comparisons per load, but it's easir to perform 4, minimizing the size of the lookup table.
    for (; text + 8 <= end; text += 4) {
        uint64_t text_slice;
        memcpy(&text_slice, text, 8);
        uint64_t text01 = (text_slice & 0x00000000FFFFFFFF) | ((text_slice & 0x000000FFFFFFFF00) << 24);
        uint64_t text23 = ((text_slice & 0x0000FFFFFFFF0000) >> 16) | ((text_slice & 0x00FFFFFFFF000000) << 8);
        uint64_t text01_indicators = ~(text01 ^ nn);
        uint64_t text23_indicators = ~(text23 ^ nn);

        // For every first match - 4 chars (32 bits) must be identical.
        text01_indicators &= text01_indicators >> 1;
        text01_indicators &= text01_indicators >> 2;
        text01_indicators &= text01_indicators >> 4;
        text01_indicators &= text01_indicators >> 8;
        text01_indicators &= text01_indicators >> 16;
        text01_indicators &= 0x0000000100000001;

        // For every first match - 4 chars (32 bits) must be identical.
        text23_indicators &= text23_indicators >> 1;
        text23_indicators &= text23_indicators >> 2;
        text23_indicators &= text23_indicators >> 4;
        text23_indicators &= text23_indicators >> 8;
        text23_indicators &= text23_indicators >> 16;
        text23_indicators &= 0x0000000100000001;

        if (text01_indicators + text23_indicators) {
            // Assuming we have performed 4 comparisons, we can only have 2^4=16 outcomes.
            // Which is small enought for a lookup table.
            uint8_t match_indicators = (uint8_t)(                      //
                (text01_indicators >> 31) | (text01_indicators << 0) | //
                (text23_indicators >> 29) | (text23_indicators << 2));
            return text - h.ptr + lookup[match_indicators];
        }
    }

    for (; text + 4 <= end; ++text)
        if (text[0] == n[0] && text[1] == n[1] && text[2] == n[2] && text[3] == n[3])
            return text - h.ptr;
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

    char const *text = h.ptr;
    char const *const end = h.ptr + h.len;
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

        text += n.anomaly_offset;
        for (; text + n.len <= end; text++) {
            memcpy(&h_anomaly, text, 4);
            if (h_anomaly == n_anomaly)                                                // Match anomaly.
                if (strzl_equal(text + 4, n_suffix_ptr, n_suffix_len))                 // Match suffix.
                    if (strzl_equal(text - n.anomaly_offset, n.ptr, n.anomaly_offset)) // Match prefix.
                        return text - h.ptr - n.anomaly_offset;
        }
        return h.len;
    }
    }
}

#if defined(__AVX2__)

/**
 *  @brief  Substring-search implementation, leveraging x86 AVX2 intrinsics and speculative
 *          execution capabilities on modern CPUs. Performing 4 unaligned vector loads per cycle
 *          was practically more efficient than loading once and shifting around, as introduces
 *          less data dependencies.
 */
size_t strzl_avx2_find_substr(strzl_haystack_t h, strzl_needle_t n) {

    // Precomputed constants
    char const *const end = h.ptr + h.len;
    uint32_t anomaly = 0;
    uint32_t mask = 0;
    switch (n.len) {
    case 1: memset(&mask, 0xFF, 1), memcpy(&anomaly, n.ptr, 1); break;
    case 2: memset(&mask, 0xFF, 2), memcpy(&anomaly, n.ptr, 2); break;
    case 3: memset(&mask, 0xFF, 3), memcpy(&anomaly, n.ptr, 3); break;
    default: memset(&mask, 0xFF, 4), memcpy(&anomaly, n.ptr, 4); break;
    }

    __m256i const anomalies = _mm256_set1_epi32(*(uint32_t const *)&anomaly);
    __m256i const masks = _mm256_set1_epi32(*(uint32_t const *)&mask);

    // Top level for-loop changes dramatically.
    // In sequential computing model for 32 offsets we would do:
    //  + 32 comparions.
    //  + 32 branches.
    // In vectorized computations models:
    //  + 4 vectorized comparisons.
    //  + 4 movemasks.
    //  + 3 bitwise ANDs.
    //  + 1 heavy (but very unlikely) branch.
    char const *text = h.ptr;
    for (; (text + n.len + 32) <= end; text += 32) {

        // Performing many unaligned loads ends up being faster than loading once and shuffling around.
        __m256i texts0 = _mm256_and_si256(_mm256_loadu_si256((__m256i const *)(text + 0)), masks);
        int matches0 = _mm256_movemask_epi8(_mm256_cmpeq_epi32(texts0, anomalies));
        __m256i texts1 = _mm256_and_si256(_mm256_loadu_si256((__m256i const *)(text + 1)), masks);
        int matches1 = _mm256_movemask_epi8(_mm256_cmpeq_epi32(texts1, anomalies));
        __m256i text2 = _mm256_and_si256(_mm256_loadu_si256((__m256i const *)(text + 2)), masks);
        int matches2 = _mm256_movemask_epi8(_mm256_cmpeq_epi32(text2, anomalies));
        __m256i texts3 = _mm256_and_si256(_mm256_loadu_si256((__m256i const *)(text + 3)), masks);
        int matches3 = _mm256_movemask_epi8(_mm256_cmpeq_epi32(texts3, anomalies));

        if (matches0 | matches1 | matches2 | matches3) {
            for (size_t i = 0; i < 32; i++) {
                if (strzl_equal(text + i, n.ptr, n.len))
                    return i + (text - h.ptr);
            }
        }
    }

    // Don't forget the last (up to 35) characters.
    strzl_haystack_t tail;
    tail.ptr = text;
    tail.len = end - text;
    size_t tail_match = strzl_naive_find_substr(tail, n);
    return text + tail_match - h.ptr;
}

#endif // x86 AVX2

#if defined(__ARM_NEON)

/**
 *  @brief  Substring-search implementation, leveraging Arm Neon intrinsics and speculative
 *          execution capabilities on modern CPUs. Performing 4 unaligned vector loads per cycle
 *          was practically more efficient than loading once and shifting around, as introduces
 *          less data dependencies.
 */
inline static size_t strzl_neon_find_substr(strzl_haystack_t h, strzl_needle_t n) {

    // Precomputed constants
    char const *const end = h.ptr + h.len;
    uint32_t anomaly = 0;
    uint32_t mask = 0;
    switch (n.len) {
    case 1: memset(&mask, 0xFF, 1), memcpy(&anomaly, n.ptr, 1); break;
    case 2: memset(&mask, 0xFF, 2), memcpy(&anomaly, n.ptr, 2); break;
    case 3: memset(&mask, 0xFF, 3), memcpy(&anomaly, n.ptr, 3); break;
    default: memset(&mask, 0xFF, 4), memcpy(&anomaly, n.ptr, 4); break;
    }

    uint32x4_t const anomalies = vld1q_dup_u32(&anomaly);
    uint32x4_t const masks = vld1q_dup_u32(&mask);

    char const *text = h.ptr;
    for (; (text + n.len + 16) <= end; text += 16) {

        uint32x4_t matches0 = vceqq_u32(vandq_u32(vld1q_u32((uint32_t const *)(text + 0)), masks), anomalies);
        uint32x4_t matches1 = vceqq_u32(vandq_u32(vld1q_u32((uint32_t const *)(text + 1)), masks), anomalies);
        uint32x4_t matches2 = vceqq_u32(vandq_u32(vld1q_u32((uint32_t const *)(text + 2)), masks), anomalies);
        uint32x4_t matches3 = vceqq_u32(vandq_u32(vld1q_u32((uint32_t const *)(text + 3)), masks), anomalies);

        // Extracting matches from matches:
        // vmaxvq_u32 (only a64)
        // vgetq_lane_u32 (all)
        // vorrq_u32 (all)
        uint32x4_t matches = vorrq_u32(vorrq_u32(matches0, matches1), vorrq_u32(matches2, matches3));
        uint64x2_t matches64x2 = vreinterpretq_u64_u32(matches);
        bool has_match = vgetq_lane_u64(matches64x2, 0) | vgetq_lane_u64(matches64x2, 1);

        if (has_match) {
            for (size_t i = 0; i < 16; i++) {
                if (strzl_equal(text + i, n.ptr, n.len))
                    return i + (text - h.ptr);
            }
        }
    }

    // Don't forget the last (up to 16+3=19) characters.
    strzl_haystack_t tail;
    tail.ptr = text;
    tail.len = end - text;
    size_t tail_match = strzl_naive_find_substr(tail, n);
    return text + tail_match - h.ptr;
}

#endif // Arm Neon

inline static void strzl_swap(size_t *a, size_t *b) {
    size_t t = *a;
    *a = *b;
    *b = t;
}

typedef char const *(*strzl_array_get_begin_t)(void const *, size_t);
typedef size_t (*strzl_array_get_length_t)(void const *, size_t);
typedef bool (*strzl_array_predicate_t)(void const *, size_t);
typedef bool (*strzl_array_comparator_t)(void const *, size_t, size_t);

typedef struct strzl_array_t {
    size_t *order;
    size_t count;
    strzl_array_get_begin_t get_begin;
    strzl_array_get_length_t get_length;
    void const *handle;
} strzl_array_t;

/**
 *  @brief  Similar to `std::partition`, given a predicate splits the
 *          array into two parts.
 */
inline static size_t strzl_partition(strzl_array_t *array, strzl_array_predicate_t predicate) {

    size_t matches = 0;
    while (matches != array->count && predicate(array->handle, array->order[matches]))
        ++matches;

    for (size_t i = matches + 1; i < array->count; ++i)
        if (predicate(array->handle, array->order[i]))
            strzl_swap(array->order + i, array->order + matches), ++matches;

    return matches;
}

/**
 *  @brief  Inplace `std::set_union` for two consecutive chunks forming
 *          the same continuous array.
 */
inline static void strzl_merge(strzl_array_t *array, size_t partition, strzl_array_comparator_t less) {

    size_t start_b = partition + 1;

    // If the direct merge is already sorted
    if (!less(array->handle, array->order[start_b], array->order[partition]))
        return;

    size_t start_a = 0;
    while (start_a <= partition && start_b <= array->count) {

        // If element 1 is in right place
        if (!less(array->handle, array->order[start_b], array->order[start_a])) {
            start_a++;
        }
        else {
            size_t value = array->order[start_b];
            size_t index = start_b;

            // Shift all the elements between element 1
            // element 2, right by 1.
            while (index != start_a) {
                array->order[index] = array->order[index - 1];
                index--;
            }
            array->order[start_a] = value;

            // Update all the pointers
            start_a++;
            partition++;
            start_b++;
        }
    }
}

inline static void _strzl_sort_recursion( //
    strzl_array_t *array,
    size_t bit_idx,
    size_t bit_max,
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__) || defined(__NT__) || __APPLE__
    int (*libc_comparator)(void *, void const *, void const *)
#else
    int (*libc_comparator)(void const *, void const *, void *)
#endif
) {

    if (!array->count)
        return;

    // Partition a range of integers according to a specific bit value
    size_t split = 0;
    {
        size_t mask = (1ul << 63) >> bit_idx;
        while (split != array->count && !(array->order[split] & mask))
            ++split;

        for (size_t i = split + 1; i < array->count; ++i)
            if (!(array->order[i] & mask))
                strzl_swap(array->order + i, array->order + split), ++split;
    }

    // Go down recursively
    if (bit_idx < bit_max) {
        strzl_array_t a = *array;
        a.count = split;
        _strzl_sort_recursion(&a, bit_idx + 1, bit_max, libc_comparator);

        strzl_array_t b = *array;
        b.order += split;
        b.count -= split;
        _strzl_sort_recursion(&b, bit_idx + 1, bit_max, libc_comparator);
    }
    // Reached the end of recursion
    else {
        // Discard the prefixes
        for (size_t i = 0; i != array->count; ++i)
            memset((char *)(&array->order[i]) + 4, 0, 4ul);

#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__) || defined(__NT__)
        // Perform sorts on smaller chunks instead of the whole handle
        // https://stackoverflow.com/a/39561369
        // https://learn.microsoft.com/en-us/cpp/c-runtime-library/reference/qsort-s?view=msvc-170
        qsort_s(array->order, split, sizeof(size_t), libc_comparator, (void *)array);
        qsort_s(array->order + split, array->count - split, sizeof(size_t), libc_comparator, (void *)array);
#elif __APPLE__
        qsort_r(array->order, split, sizeof(size_t), (void *)array, libc_comparator);
        qsort_r(array->order + split, array->count - split, sizeof(size_t), (void *)array, libc_comparator);
#else
        // https://linux.die.net/man/3/qsort_r
        qsort_r(array->order, split, sizeof(size_t), libc_comparator, (void *)array);
        qsort_r(array->order + split, array->count - split, sizeof(size_t), libc_comparator, (void *)array);
#endif
    }
}

inline static int _strzl_sort_array_strncmp(
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__) || defined(__NT__) || __APPLE__
    void *array_raw, void const *a_raw, void const *b_raw
#else
    void const *a_raw, void const *b_raw, void *array_raw
#endif
) {
    // https://man.freebsd.org/cgi/man.cgi?query=qsort_s&sektion=3&n=1
    // https://www.man7.org/linux/man-pages/man3/strcmp.3.html
    strzl_array_t *array = (strzl_array_t *)array_raw;
    size_t a = *(size_t *)a_raw;
    size_t b = *(size_t *)b_raw;
    size_t a_len = array->get_length(array->handle, a);
    size_t b_len = array->get_length(array->handle, b);
    int res = strncmp( //
        array->get_begin(array->handle, a),
        array->get_begin(array->handle, b),
        a_len > b_len ? b_len : a_len);
    return res ? res : a_len - b_len;
}

inline static int _strzl_sort_array_strncasecmp(
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__) || defined(__NT__) || __APPLE__
    void *array_raw, void const *a_raw, void const *b_raw
#else
    void const *a_raw, void const *b_raw, void *array_raw
#endif
) {
    // https://man.freebsd.org/cgi/man.cgi?query=qsort_s&sektion=3&n=1
    // https://www.man7.org/linux/man-pages/man3/strcmp.3.html
    strzl_array_t *array = (strzl_array_t *)array_raw;
    size_t a = *(size_t *)a_raw;
    size_t b = *(size_t *)b_raw;
    size_t a_len = array->get_length(array->handle, a);
    size_t b_len = array->get_length(array->handle, b);
    int res = strncasecmp( //
        array->get_begin(array->handle, a),
        array->get_begin(array->handle, b),
        a_len > b_len ? b_len : a_len);
    return res ? res : a_len - b_len;
}

struct strzl_sort_config_t {
    bool case_insensitive;
};

/**
 *  @brief  Sorting algorithm, combining Radix Sort for the first 32 bits of every word
 *          and a follow-up Quick Sort on resulting structure.
 */
inline static void strzl_sort(strzl_array_t *array, strzl_sort_config_t const *config) {

    bool case_insensitive = config && config->case_insensitive;

    // Export up to 4 bytes into the `array` bits themselves
    for (size_t i = 0; i != array->count; ++i) {
        char const *begin = array->get_begin(array->handle, array->order[i]);
        size_t length = array->get_length(array->handle, array->order[i]);
        length = length > 4ul ? 4ul : length;
        char *prefix = (char *)&array->order[i];
        for (size_t j = 0; j != length; ++j)
            prefix[7 - j] = begin[j];
        if (case_insensitive) {
            prefix[0] = tolower(prefix[0]);
            prefix[1] = tolower(prefix[1]);
            prefix[2] = tolower(prefix[2]);
            prefix[3] = tolower(prefix[3]);
        }
    }

#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__) || defined(__NT__) || __APPLE__
    int (*comparator)(void *, void const *, void const *);
#else
    int (*comparator)(void const *, void const *, void *);
#endif
    comparator = _strzl_sort_array_strncmp;
    if (case_insensitive)
        comparator = _strzl_sort_array_strncasecmp;

    // Perform optionally-parallel radix sort on them
    _strzl_sort_recursion(array, 0, 32, comparator);
}

typedef uint8_t levenstein_distance_t;

/**
 *  @return Amount of temporary memory (in bytes) needed to efficiently compute
 *          the Levenstein distance between two strings of given size.
 */
inline static size_t strzl_levenstein_memory_needed(size_t, size_t b_length) { return b_length + b_length + 2; }

/**
 *  @brief  Auxiliary function, that computes the minimum of three values.
 */
inline static levenstein_distance_t _strzl_levenstein_minimum( //
    levenstein_distance_t a,
    levenstein_distance_t b,
    levenstein_distance_t c) {

    return (a < b ? (a < c ? a : c) : (b < c ? b : c));
}

/**
 *  @brief  Levenshtein String Similarity function, implemented with linear memory consumption.
 *          It accepts an upper bound on the possible error. Quadratic complexity in time, linear in space.
 */
inline static levenstein_distance_t strzl_levenstein( //
    char const *a,
    size_t a_length,
    char const *b,
    size_t b_length,
    levenstein_distance_t bound,
    void *buffer) {

    if (a_length == 0)
        return b_length <= bound ? b_length : bound + 1;
    if (b_length == 0)
        return a_length <= bound ? a_length : bound + 1;

    levenstein_distance_t *previous_distances = (levenstein_distance_t *)buffer;
    levenstein_distance_t *current_distances = previous_distances + b_length + 1;

    for (size_t idx_b = 0; idx_b != (b_length + 1); ++idx_b)
        previous_distances[idx_b] = idx_b;

    for (size_t idx_a = 0; idx_a != a_length; ++idx_a) {
        current_distances[0] = idx_a + 1;

        for (size_t idx_b = 0; idx_b != b_length; ++idx_b) {
            levenstein_distance_t cost_deletion = previous_distances[idx_b + 1] + 1;
            levenstein_distance_t cost_insertion = current_distances[idx_b] + 1;
            levenstein_distance_t cost_substitution = previous_distances[idx_b] + (a[idx_a] != b[idx_b]);
            current_distances[idx_b + 1] = _strzl_levenstein_minimum(cost_deletion, cost_insertion, cost_substitution);
        }

        // Swap previous_distances and current_distances pointers
        levenstein_distance_t *temp = previous_distances;
        previous_distances = current_distances;
        current_distances = temp;
    }

    return previous_distances[b_length];
}

#ifdef __cplusplus
}
#endif

#ifdef _MSC_VER
#undef strncasecmp
#undef strcasecmp
#endif
#undef popcount64
#undef ctz64

#endif // STRINGZILLA_H_
