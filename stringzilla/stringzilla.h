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
    for (; h_ptr != h_end; ++h_ptr)
        result += *h_ptr == n;
    return result;
}

inline static size_t strzl_naive_find_char(strzl_haystack_t h, char n) {
    char const *h_ptr = h.ptr;
    char const *h_end = h.ptr + h.len;
    for (; h_ptr != h_end; ++h_ptr)
        if (*h_ptr == n)
            return h_ptr - h.ptr;
    return h.len;
}

inline static size_t strzl_naive_rfind_char(strzl_haystack_t h, char n) {
    char const *h_end = h.ptr + h.len;
    for (char const *h_ptr = h_end; h_ptr != h_end; --h_ptr)
        if (*(h_ptr - 1) == n)
            return h_ptr - h.ptr - 1;
    return h.len;
}

inline static size_t strzl_naive_count_substr(strzl_haystack_t h, strzl_needle_t n, bool overlap = false) {

    if (n.len == 1)
        return strzl_naive_count_char(h, *n.ptr);
    if (h.len < n.len)
        return 0;

    size_t result = 0;
    if (!overlap)

        for (size_t off = 0; off <= h.len - n.len;)
            if (strzl_equal(h.ptr + off, n.ptr, n.len))
                off += n.len, result++;
            else
                off++;

    else
        for (size_t off = 0; off <= h.len - n.len; off++)
            result += strzl_equal(h.ptr + off, n.ptr, n.len);

    return result;
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
    case 2: {
        // On very short patterns, it's easier to manually unroll the loop, even if we keep the scalar code.
        uint16_t word, needle;
        memcpy(&needle, n.ptr, 2);
        for (; h_ptr + 2 <= h_end; h_ptr++) {
            memcpy(&word, h_ptr, 2);
            if (word == needle)
                return h_ptr - h.ptr;
        }
        return h.len;
    }
    case 3: {
        // On very short patterns, it's easier to manually unroll the loop, even if we keep the scalar code.
        uint32_t word = 0, needle = 0; // Initialize for the last byte to deterministically match.
        memcpy(&needle, n.ptr, 3);
        for (; h_ptr + 3 <= h_end; h_ptr++) {
            memcpy(&word, h_ptr, 3);
            if (word == needle)
                return h_ptr - h.ptr;
        }
        return h.len;
    }
    case 4: {
        // On very short patterns, it's easier to manually unroll the loop, even if we keep the scalar code.
        uint32_t word, needle;
        memcpy(&needle, n.ptr, 4);
        for (; h_ptr + 4 <= h_end; h_ptr++) {
            memcpy(&word, h_ptr, 4);
            if (word == needle)
                return h_ptr - h.ptr;
        }
        return h.len;
    }
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
    size_t tail_start = h_ptr - h.ptr;
    size_t tail_match = prefixed_t {}.find_offset(h.after_n(tail_start), n);
    return tail_match + tail_start;
}

#endif // x86 AVX2

#if defined(__ARM_NEON)

/**
 *  @brief  Character-counting routing, leveraging Arm Neon instrinsics and checking 16 characters at once.
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
        result += __builtin_popcountll(vgetq_lane_u64(masks64x2, 0)) / 8;
        result += __builtin_popcountll(vgetq_lane_u64(masks64x2, 1)) / 8;
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

#ifdef __cplusplus
}
#endif

#endif // STRINGZILLA_H_