#ifndef STRINGZILLA_H_
#define STRINGZILLA_H_

#if defined(__AVX2__)
#include <x86intrin.h>
#endif

#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif

#if defined(__ARM_FEATURE_CRC32)
#include <arm_acle.h>
#endif

/**
 *  Intrinsics aliases for MSVC, GCC, and Clang.
 */
#ifdef _MSC_VER
#include <intrin.h>
#define popcount64 __popcnt64
#define ctz64 _tzcnt_u64
#define clz64 _lzcnt_u64
#else
#define popcount64 __builtin_popcountll
#define ctz64 __builtin_ctzll
#define clz64 __builtin_clzll
#endif

/**
 *  @brief  Generally `NULL` is coming from locale.h, stddef.h, stdio.h, stdlib.h, string.h, time.h, and wchar.h,
 *          according to the C standard.
 */
#ifndef NULL
#define NULL ((void *)0)
#endif

/**
 * @brief   Compile-time assert macro.
 */
#define SZ_STATIC_ASSERT(condition, name)                \
    typedef struct {                                     \
        int static_assert_##name : (condition) ? 1 : -1; \
    } sz_static_assert_##name##_t

#ifdef __cplusplus
extern "C" {
#endif

/**
 *  @brief  Analogous to `sz_size_t` and `std::sz_size_t`, unsigned integer, identical to pointer size.
 *          64-bit on most platforms where pointers are 64-bit.
 *          32-bit on platforms where pointers are 32-bit.
 */
#if defined(__LP64__) || defined(_LP64) || defined(__x86_64__) || defined(_WIN64)
typedef unsigned long long sz_size_t;
#else
typedef unsigned sz_size_t;
#endif
SZ_STATIC_ASSERT(sizeof(sz_size_t) == sizeof(void *), sz_size_t_must_be_pointer_size);

typedef int sz_bool_t;                 // Only one relevant bit
typedef unsigned sz_u32_t;             // Always 32 bits
typedef unsigned long long sz_u64_t;   // Always 64 bits
typedef char const *sz_string_start_t; // A type alias for `char const * `

/**
 *  @brief  For faster bounded Levenshtein (Edit) distance computation no more than 255 characters are supported.
 */
typedef unsigned char levenshtein_distance_t;

/**
 *  @brief  Helper construct for higher-level bindings.
 */
typedef struct sz_string_view_t {
    sz_string_start_t start;
    sz_size_t length;
} sz_string_view_t;

/**
 *  @brief  Internal data-structure, used to address "anomalies" (often prefixes),
 *          during substring search. Always a 32-bit unsigned integer, containing 4 chars.
 */
typedef union _sz_anomaly_t {
    unsigned u32;
    unsigned char u8s[4];
} _sz_anomaly_t;

/**
 *  @brief  This is a slightly faster alternative to `strncmp(a, b, length) == 0`.
 *          Doesn't provide major performance improvements, but helps avoid the LibC dependency.
 *  @return 1 for `true`, and 0 for `false`.
 */
inline static sz_bool_t sz_equal(sz_string_start_t a, sz_string_start_t b, sz_size_t length) {
    sz_string_start_t const a_end = a + length;
    while (a != a_end && *a == *b) a++, b++;
    return a_end == a;
}

/**
 *  @brief  Count the number of occurrences of a @b single-character needle in an arbitrary length haystack.
 *          This implementation uses hardware-agnostic SWAR technique, to process 8 characters at a time.
 */
inline static sz_size_t sz_count_char_swar(sz_string_start_t const haystack,
                                           sz_size_t const haystack_length,
                                           sz_string_start_t const needle) {

    sz_size_t result = 0;
    sz_string_start_t text = haystack;
    sz_string_start_t const end = haystack + haystack_length;

    // Process the misaligned head, to void UB on unaligned 64-bit loads.
    for (; ((sz_size_t)text & 7ull) && text < end; ++text) result += *text == *needle;

    // This code simulates hyper-scalar execution, comparing 8 characters at a time.
    sz_u64_t nnnnnnnn = *needle;
    nnnnnnnn |= nnnnnnnn << 8;
    nnnnnnnn |= nnnnnnnn << 16;
    nnnnnnnn |= nnnnnnnn << 32;
    for (; text + 8 <= end; text += 8) {
        sz_u64_t text_slice = *(sz_u64_t const *)text;
        sz_u64_t match_indicators = ~(text_slice ^ nnnnnnnn);
        match_indicators &= match_indicators >> 1;
        match_indicators &= match_indicators >> 2;
        match_indicators &= match_indicators >> 4;
        match_indicators &= 0x0101010101010101;
        result += popcount64(match_indicators);
    }

    for (; text < end; ++text) result += *text == *needle;
    return result;
}

/**
 *  @brief  Find the first occurrence of a @b single-character needle in an arbitrary length haystack.
 *          This implementation uses hardware-agnostic SWAR technique, to process 8 characters at a time.
 *          Identical to `memchr(haystack, needle[0], haystack_length)`.
 */
inline static sz_string_start_t sz_find_1char_swar(sz_string_start_t const haystack,
                                                   sz_size_t const haystack_length,
                                                   sz_string_start_t const needle) {

    sz_string_start_t text = haystack;
    sz_string_start_t const end = haystack + haystack_length;

    // Process the misaligned head, to void UB on unaligned 64-bit loads.
    for (; ((sz_size_t)text & 7ull) && text < end; ++text)
        if (*text == *needle) return text;

    // This code simulates hyper-scalar execution, analyzing 8 offsets at a time.
    sz_u64_t nnnnnnnn = *needle;
    nnnnnnnn |= nnnnnnnn << 8;  // broadcast `needle` into `nnnnnnnn`
    nnnnnnnn |= nnnnnnnn << 16; // broadcast `needle` into `nnnnnnnn`
    nnnnnnnn |= nnnnnnnn << 32; // broadcast `needle` into `nnnnnnnn`
    for (; text + 8 <= end; text += 8) {
        sz_u64_t text_slice = *(sz_u64_t const *)text;
        sz_u64_t match_indicators = ~(text_slice ^ nnnnnnnn);
        match_indicators &= match_indicators >> 1;
        match_indicators &= match_indicators >> 2;
        match_indicators &= match_indicators >> 4;
        match_indicators &= 0x0101010101010101;

        if (match_indicators != 0) return text + ctz64(match_indicators) / 8;
    }

    for (; text < end; ++text)
        if (*text == *needle) return text;
    return NULL;
}

/**
 *  @brief  Find the last occurrence of a @b single-character needle in an arbitrary length haystack.
 *          This implementation uses hardware-agnostic SWAR technique, to process 8 characters at a time.
 *          Identical to `memrchr(haystack, needle[0], haystack_length)`.
 */
inline static sz_string_start_t sz_rfind_1char_swar(sz_string_start_t const haystack,
                                                    sz_size_t const haystack_length,
                                                    sz_string_start_t const needle) {

    sz_string_start_t const end = haystack + haystack_length;
    sz_string_start_t text = end - 1;

    // Process the misaligned head, to void UB on unaligned 64-bit loads.
    for (; ((sz_size_t)text & 7ull) && text >= haystack; --text)
        if (*text == *needle) return text;

    // This code simulates hyper-scalar execution, analyzing 8 offsets at a time.
    sz_u64_t nnnnnnnn = *needle;
    nnnnnnnn |= nnnnnnnn << 8;  // broadcast `needle` into `nnnnnnnn`
    nnnnnnnn |= nnnnnnnn << 16; // broadcast `needle` into `nnnnnnnn`
    nnnnnnnn |= nnnnnnnn << 32; // broadcast `needle` into `nnnnnnnn`
    for (; text - 8 >= haystack; text -= 8) {
        sz_u64_t text_slice = *(sz_u64_t const *)text;
        sz_u64_t match_indicators = ~(text_slice ^ nnnnnnnn);
        match_indicators &= match_indicators >> 1;
        match_indicators &= match_indicators >> 2;
        match_indicators &= match_indicators >> 4;
        match_indicators &= 0x0101010101010101;

        if (match_indicators != 0) return text - 8 + clz64(match_indicators) / 8;
    }

    for (; text >= haystack; --text)
        if (*text == *needle) return text;
    return NULL;
}

/**
 *  @brief  Find the first occurrence of a @b two-character needle in an arbitrary length haystack.
 *          This implementation uses hardware-agnostic SWAR technique, to process 8 characters at a time.
 */
inline static sz_string_start_t sz_find_2char_swar(sz_string_start_t const haystack,
                                                   sz_size_t const haystack_length,
                                                   sz_string_start_t const needle) {

    sz_string_start_t text = haystack;
    sz_string_start_t const end = haystack + haystack_length;

    // Process the misaligned head, to void UB on unaligned 64-bit loads.
    for (; ((sz_size_t)text & 7ull) && text + 2 <= end; ++text)
        if (text[0] == needle[0] && text[1] == needle[1]) return text;

    // This code simulates hyper-scalar execution, analyzing 7 offsets at a time.
    sz_u64_t nnnn = ((sz_u64_t)(needle[0]) << 0) | ((sz_u64_t)(needle[1]) << 8); // broadcast `needle` into `nnnn`
    nnnn |= nnnn << 16;                                                          // broadcast `needle` into `nnnn`
    nnnn |= nnnn << 32;                                                          // broadcast `needle` into `nnnn`
    for (; text + 8 <= end; text += 7) {
        sz_u64_t text_slice = *(sz_u64_t const *)text;
        sz_u64_t even_indicators = ~(text_slice ^ nnnn);
        sz_u64_t odd_indicators = ~((text_slice << 8) ^ nnnn);

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
            sz_u64_t match_indicators = even_indicators | (odd_indicators >> 8);
            return text + ctz64(match_indicators) / 8;
        }
    }

    for (; text + 2 <= end; ++text)
        if (text[0] == needle[0] && text[1] == needle[1]) return text;
    return NULL;
}

/**
 *  @brief  Find the first occurrence of a three-character needle in an arbitrary length haystack.
 *          This implementation uses hardware-agnostic SWAR technique, to process 8 characters at a time.
 */
inline static sz_string_start_t sz_find_3char_swar(sz_string_start_t const haystack,
                                                   sz_size_t const haystack_length,
                                                   sz_string_start_t const needle) {

    sz_string_start_t text = haystack;
    sz_string_start_t end = haystack + haystack_length;

    // Process the misaligned head, to void UB on unaligned 64-bit loads.
    for (; ((sz_size_t)text & 7ull) && text + 3 <= end; ++text)
        if (text[0] == needle[0] && text[1] == needle[1] && text[2] == needle[2]) return text;

    // This code simulates hyper-scalar execution, analyzing 6 offsets at a time.
    // We have two unused bytes at the end.
    sz_u64_t nn =                      // broadcast `needle` into `nn`
        (sz_u64_t)(needle[0] << 0) |   // broadcast `needle` into `nn`
        ((sz_u64_t)(needle[1]) << 8) | // broadcast `needle` into `nn`
        ((sz_u64_t)(needle[2]) << 16); // broadcast `needle` into `nn`
    nn |= nn << 24;                    // broadcast `needle` into `nn`
    nn <<= 16;                         // broadcast `needle` into `nn`

    for (; text + 8 <= end; text += 6) {
        sz_u64_t text_slice = *(sz_u64_t const *)text;
        sz_u64_t first_indicators = ~(text_slice ^ nn);
        sz_u64_t second_indicators = ~((text_slice << 8) ^ nn);
        sz_u64_t third_indicators = ~((text_slice << 16) ^ nn);
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

        sz_u64_t match_indicators = first_indicators | (second_indicators >> 8) | (third_indicators >> 16);
        if (match_indicators != 0) return text + ctz64(match_indicators) / 8;
    }

    for (; text + 3 <= end; ++text)
        if (text[0] == needle[0] && text[1] == needle[1] && text[2] == needle[2]) return text;
    return NULL;
}

/**
 *  @brief  Find the first occurrence of a @b four-character needle in an arbitrary length haystack.
 *          This implementation uses hardware-agnostic SWAR technique, to process 8 characters at a time.
 */
inline static sz_string_start_t sz_find_4char_swar(sz_string_start_t const haystack,
                                                   sz_size_t const haystack_length,
                                                   sz_string_start_t const needle) {

    sz_string_start_t text = haystack;
    sz_string_start_t end = haystack + haystack_length;

    // Process the misaligned head, to void UB on unaligned 64-bit loads.
    for (; ((sz_size_t)text & 7ull) && text + 4 <= end; ++text)
        if (text[0] == needle[0] && text[1] == needle[1] && text[2] == needle[2] && text[3] == needle[3]) return text;

    // This code simulates hyper-scalar execution, analyzing 4 offsets at a time.
    sz_u64_t nn = (sz_u64_t)(needle[0] << 0) | ((sz_u64_t)(needle[1]) << 8) | ((sz_u64_t)(needle[2]) << 16) |
                  ((sz_u64_t)(needle[3]) << 24);
    nn |= nn << 32;

    //
    unsigned char offset_in_slice[16] = {0};
    offset_in_slice[0x2] = offset_in_slice[0x6] = offset_in_slice[0xA] = offset_in_slice[0xE] = 1;
    offset_in_slice[0x4] = offset_in_slice[0xC] = 2;
    offset_in_slice[0x8] = 3;

    // We can perform 5 comparisons per load, but it's easier to perform 4, minimizing the size of the lookup table.
    for (; text + 8 <= end; text += 4) {
        sz_u64_t text_slice = *(sz_u64_t const *)text;
        sz_u64_t text01 = (text_slice & 0x00000000FFFFFFFF) | ((text_slice & 0x000000FFFFFFFF00) << 24);
        sz_u64_t text23 = ((text_slice & 0x0000FFFFFFFF0000) >> 16) | ((text_slice & 0x00FFFFFFFF000000) << 8);
        sz_u64_t text01_indicators = ~(text01 ^ nn);
        sz_u64_t text23_indicators = ~(text23 ^ nn);

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
            // Which is small enough for a lookup table.
            unsigned char match_indicators = (unsigned char)(          //
                (text01_indicators >> 31) | (text01_indicators << 0) | //
                (text23_indicators >> 29) | (text23_indicators << 2));
            return text + offset_in_slice[match_indicators];
        }
    }

    for (; text + 4 <= end; ++text)
        if (text[0] == needle[0] && text[1] == needle[1] && text[2] == needle[2] && text[3] == needle[3]) return text;
    return NULL;
}

/**
 *  @brief  Trivial substring search with scalar SWAR code. Instead of comparing characters one-by-one
 *          it compares 4-byte anomalies first, most commonly prefixes. It's computationally cheaper.
 *          Matching performance fluctuates between 1 GB/s and 3,5 GB/s per core.
 */
inline static sz_string_start_t sz_find_substring_swar( //
    sz_string_start_t const haystack,
    sz_size_t const haystack_length,
    sz_string_start_t const needle,
    sz_size_t const needle_length) {

    if (haystack_length < needle_length) return NULL;

    sz_size_t anomaly_offset = 0;
    switch (needle_length) {
    case 0: return NULL;
    case 1: return sz_find_1char_swar(haystack, haystack_length, needle);
    case 2: return sz_find_2char_swar(haystack, haystack_length, needle);
    case 3: return sz_find_3char_swar(haystack, haystack_length, needle);
    case 4: return sz_find_4char_swar(haystack, haystack_length, needle);
    default: {
        sz_string_start_t text = haystack;
        sz_string_start_t const end = haystack + haystack_length;

        _sz_anomaly_t n_anomaly, h_anomaly;
        sz_size_t const n_suffix_len = needle_length - 4 - anomaly_offset;
        sz_string_start_t n_suffix_ptr = needle + 4 + anomaly_offset;
        n_anomaly.u8s[0] = needle[anomaly_offset];
        n_anomaly.u8s[1] = needle[anomaly_offset + 1];
        n_anomaly.u8s[2] = needle[anomaly_offset + 2];
        n_anomaly.u8s[3] = needle[anomaly_offset + 3];
        h_anomaly.u8s[0] = haystack[0];
        h_anomaly.u8s[1] = haystack[1];
        h_anomaly.u8s[2] = haystack[2];
        h_anomaly.u8s[3] = haystack[3];

        text += anomaly_offset;
        while (text + needle_length <= end) {
            h_anomaly.u8s[3] = text[3];
            if (h_anomaly.u32 == n_anomaly.u32)                     // Match anomaly.
                if (sz_equal(text + 4, n_suffix_ptr, n_suffix_len)) // Match suffix.
                    return text;

            h_anomaly.u32 >>= 8;
            ++text;
        }
        return NULL;
    }
    }
}

/**
 *  Helper function, used in substring search operations.
 */
inline static void _sz_find_substring_populate_anomaly( //
    sz_string_start_t const needle,
    sz_size_t const needle_length,
    _sz_anomaly_t *anomaly_out,
    _sz_anomaly_t *mask_out) {

    _sz_anomaly_t anomaly;
    _sz_anomaly_t mask;
    switch (needle_length) {
    case 1:
        mask.u8s[0] = 0xFF, mask.u8s[1] = mask.u8s[2] = mask.u8s[3] = 0;
        anomaly.u8s[0] = needle[0], anomaly.u8s[1] = anomaly.u8s[2] = anomaly.u8s[3] = 0;
        break;
    case 2:
        mask.u8s[0] = mask.u8s[1] = 0xFF, mask.u8s[2] = mask.u8s[3] = 0;
        anomaly.u8s[0] = needle[0], anomaly.u8s[1] = needle[1], anomaly.u8s[2] = anomaly.u8s[3] = 0;
        break;
    case 3:
        mask.u8s[0] = mask.u8s[1] = mask.u8s[2] = 0xFF, mask.u8s[3] = 0;
        anomaly.u8s[0] = needle[0], anomaly.u8s[1] = needle[1], anomaly.u8s[2] = needle[2], anomaly.u8s[3] = 0;
        break;
    default:
        mask.u32 = 0xFFFFFFFF;
        anomaly.u8s[0] = needle[0], anomaly.u8s[1] = needle[1], anomaly.u8s[2] = needle[2], anomaly.u8s[3] = needle[3];
        break;
    }
    *anomaly_out = anomaly;
    *mask_out = mask;
}

#if defined(__AVX2__)

/**
 *  @brief  Substring-search implementation, leveraging x86 AVX2 intrinsics and speculative
 *          execution capabilities on modern CPUs. Performing 4 unaligned vector loads per cycle
 *          was practically more efficient than loading once and shifting around, as introduces
 *          less data dependencies.
 */
inline static sz_string_start_t sz_find_substring_avx2(sz_string_start_t const haystack,
                                                       sz_size_t const haystack_length,
                                                       sz_string_start_t const needle,
                                                       sz_size_t const needle_length) {

    // Precomputed constants
    sz_string_start_t const end = haystack + haystack_length;
    _sz_anomaly_t anomaly;
    _sz_anomaly_t mask;
    _sz_find_substring_populate_anomaly(needle, needle_length, &anomaly, &mask);
    __m256i const anomalies = _mm256_set1_epi32(anomaly.u32);
    __m256i const masks = _mm256_set1_epi32(mask.u32);

    // Top level for-loop changes dramatically.
    // In sequential computing model for 32 offsets we would do:
    //  + 32 comparions.
    //  + 32 branches.
    // In vectorized computations models:
    //  + 4 vectorized comparisons.
    //  + 4 movemasks.
    //  + 3 bitwise ANDs.
    //  + 1 heavy (but very unlikely) branch.
    sz_string_start_t text = haystack;
    while (text + needle_length + 32 <= end) {

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
            int matches =                 //
                (matches0 & 0x11111111) | //
                (matches1 & 0x22222222) | //
                (matches2 & 0x44444444) | //
                (matches3 & 0x88888888);
            sz_size_t first_match_offset = ctz64(matches);
            if (needle_length > 4) {
                if (sz_equal(text + first_match_offset + 4, needle + 4, needle_length - 4)) {
                    return text + first_match_offset;
                }
                else { text += first_match_offset + 1; }
            }
            else { return text + first_match_offset; }
        }
        else { text += 32; }
    }

    // Don't forget the last (up to 35) characters.
    return sz_find_substring_swar(text, end - text, needle, needle_length);
}

#endif // x86 AVX2

#if defined(__ARM_NEON)

/**
 *  @brief  Substring-search implementation, leveraging Arm Neon intrinsics and speculative
 *          execution capabilities on modern CPUs. Performing 4 unaligned vector loads per cycle
 *          was practically more efficient than loading once and shifting around, as introduces
 *          less data dependencies.
 */
inline static sz_string_start_t sz_find_substring_neon(sz_string_start_t const haystack,
                                                       sz_size_t const haystack_length,
                                                       sz_string_start_t const needle,
                                                       sz_size_t const needle_length) {

    // Precomputed constants
    sz_string_start_t const end = haystack + haystack_length;
    _sz_anomaly_t anomaly;
    _sz_anomaly_t mask;
    _sz_find_substring_populate_anomaly(needle, needle_length, &anomaly, &mask);
    uint32x4_t const anomalies = vld1q_dup_u32(&anomaly.u32);
    uint32x4_t const masks = vld1q_dup_u32(&mask.u32);
    uint32x4_t matches, matches0, matches1, matches2, matches3;

    sz_string_start_t text = haystack;
    while (text + needle_length + 16 <= end) {

        // Each of the following `matchesX` contains only 4 relevant bits - one per word.
        // Each signifies a match at the given offset.
        matches0 = vceqq_u32(vandq_u32(vreinterpretq_u32_u8(vld1q_u8((unsigned char *)text + 0)), masks), anomalies);
        matches1 = vceqq_u32(vandq_u32(vreinterpretq_u32_u8(vld1q_u8((unsigned char *)text + 1)), masks), anomalies);
        matches2 = vceqq_u32(vandq_u32(vreinterpretq_u32_u8(vld1q_u8((unsigned char *)text + 2)), masks), anomalies);
        matches3 = vceqq_u32(vandq_u32(vreinterpretq_u32_u8(vld1q_u8((unsigned char *)text + 3)), masks), anomalies);
        matches = vorrq_u32(vorrq_u32(matches0, matches1), vorrq_u32(matches2, matches3));

        if (vmaxvq_u32(matches)) {
            // Let's isolate the match from every word
            matches0 = vandq_u32(matches0, vdupq_n_u32(0x00000001));
            matches1 = vandq_u32(matches1, vdupq_n_u32(0x00000002));
            matches2 = vandq_u32(matches2, vdupq_n_u32(0x00000004));
            matches3 = vandq_u32(matches3, vdupq_n_u32(0x00000008));
            matches = vorrq_u32(vorrq_u32(matches0, matches1), vorrq_u32(matches2, matches3));

            // By now, every 32-bit word of `matches` no more than 4 set bits.
            // Meaning that we can narrow it down to a single 16-bit word.
            uint16x4_t matches_u16x4 = vmovn_u32(matches);
            uint16_t matches_u16 =                       //
                (vget_lane_u16(matches_u16x4, 0) << 0) | //
                (vget_lane_u16(matches_u16x4, 1) << 4) | //
                (vget_lane_u16(matches_u16x4, 2) << 8) | //
                (vget_lane_u16(matches_u16x4, 3) << 12);

            // Find the first match
            sz_size_t first_match_offset = ctz64(matches_u16);
            if (needle_length > 4) {
                if (sz_equal(text + first_match_offset + 4, needle + 4, needle_length - 4)) {
                    return text + first_match_offset;
                }
                else { text += first_match_offset + 1; }
            }
            else { return text + first_match_offset; }
        }
        else { text += 16; }
    }

    // Don't forget the last (up to 16+3=19) characters.
    return sz_find_substring_swar(text, end - text, needle, needle_length);
}

#endif // Arm Neon

inline static sz_size_t sz_count_char(sz_string_start_t const haystack,
                                      sz_size_t const haystack_length,
                                      sz_string_start_t const needle) {
    return sz_count_char_swar(haystack, haystack_length, needle);
}

inline static sz_string_start_t sz_find_1char(sz_string_start_t const haystack,
                                              sz_size_t const haystack_length,
                                              sz_string_start_t const needle) {
    return sz_find_1char_swar(haystack, haystack_length, needle);
}

inline static sz_string_start_t sz_rfind_1char(sz_string_start_t const haystack,
                                               sz_size_t const haystack_length,
                                               sz_string_start_t const needle) {
    return sz_rfind_1char_swar(haystack, haystack_length, needle);
}

inline static sz_string_start_t sz_find_substring(sz_string_start_t const haystack,
                                                  sz_size_t const haystack_length,
                                                  sz_string_start_t const needle,
                                                  sz_size_t const needle_length) {
    if (haystack_length < needle_length || needle_length == 0) return NULL;
#if defined(__ARM_NEON)
    return sz_find_substring_neon(haystack, haystack_length, needle, needle_length);
#elif defined(__AVX2__)
    return sz_find_substring_avx2(haystack, haystack_length, needle, needle_length);
#else
    return sz_find_substring_swar(haystack, haystack_length, needle, needle_length);
#endif
}

/**
 *  @brief  Maps any ASCII character to itself, or the lowercase variant, if available.
 */
inline static char sz_tolower_ascii(char c) {
    static unsigned char lowered[256] = {
        0,   1,   2,   3,   4,   5,   6,   7,   8,   9,   10,  11,  12,  13,  14,  15,  //
        16,  17,  18,  19,  20,  21,  22,  23,  24,  25,  26,  27,  28,  29,  30,  31,  //
        32,  33,  34,  35,  36,  37,  38,  39,  40,  41,  42,  43,  44,  45,  46,  47,  //
        48,  49,  50,  51,  52,  53,  54,  55,  56,  57,  58,  59,  60,  61,  62,  63,  //
        64,  97,  98,  99,  100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, //
        112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 91,  92,  93,  94,  95,  //
        96,  97,  98,  99,  100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, //
        112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, //
        128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, //
        144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, //
        160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, //
        176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, //
        224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, //
        240, 241, 242, 243, 244, 245, 246, 215, 248, 249, 250, 251, 252, 253, 254, 223, //
        224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, //
        240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, //
    };
    return *(char *)&lowered[(int)c];
}

/**
 *  @brief  Maps any ASCII character to itself, or the uppercase variant, if available.
 */
inline static char sz_toupper_ascii(char c) {
    static unsigned char upped[256] = {
        0,   1,   2,   3,   4,   5,   6,   7,   8,   9,   10,  11,  12,  13,  14,  15,  //
        16,  17,  18,  19,  20,  21,  22,  23,  24,  25,  26,  27,  28,  29,  30,  31,  //
        32,  33,  34,  35,  36,  37,  38,  39,  40,  41,  42,  43,  44,  45,  46,  47,  //
        48,  49,  50,  51,  52,  53,  54,  55,  56,  57,  58,  59,  60,  61,  62,  63,  //
        64,  97,  98,  99,  100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, //
        112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 91,  92,  93,  94,  95,  //
        96,  65,  66,  67,  68,  69,  70,  71,  72,  73,  74,  75,  76,  77,  78,  79,  //
        80,  81,  82,  83,  84,  85,  86,  87,  88,  89,  90,  123, 124, 125, 126, 127, //
        128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, //
        144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, //
        160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, //
        176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, //
        224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, //
        240, 241, 242, 243, 244, 245, 246, 215, 248, 249, 250, 251, 252, 253, 254, 223, //
        224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, //
        240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, //
    };
    return *(char *)&upped[(int)c];
}

/**
 *  @brief Load a 64-bit unsigned integer from a potentially unaligned pointer.
 *
 *  @note This function uses compiler-specific attributes or keywords to
 *        ensure correct and efficient unaligned loads. It's designed to work
 *        with both MSVC and GCC/Clang.
 */
inline static sz_u64_t sz_u64_unaligned_load(void const *ptr) {
#ifdef _MSC_VER
    return *((__unaligned sz_u64_t *)ptr);
#else
    __attribute__((aligned(1))) sz_u64_t const *uptr = (sz_u64_t const *)ptr;
    return *uptr;
#endif
}

/**
 *  @brief Reverse the byte order of a 64-bit unsigned integer.
 *
 *  @note This function uses compiler-specific intrinsics to achieve the
 *        byte-reversal. It's designed to work with both MSVC and GCC/Clang.
 */
inline static sz_u64_t sz_u64_byte_reverse(sz_u64_t val) {
#ifdef _MSC_VER
    return _byteswap_uint64(val);
#else
    return __builtin_bswap64(val);
#endif
}

/**
 *  @brief  Compute the logarithm base 2 of an integer.
 *
 *  @note If n is 0, the function returns 0 to avoid undefined behavior.
 *  @note This function uses compiler-specific intrinsics or built-ins
 *        to achieve the computation. It's designed to work with GCC/Clang and MSVC.
 */
inline static sz_size_t sz_log2i(sz_size_t n) {
    if (n == 0) return 0;

#ifdef _WIN64
#ifdef _MSC_VER
    unsigned long index;
    if (_BitScanReverse64(&index, n)) return index;
    return 0; // This line might be redundant due to the initial check, but it's safer to include it.
#else
    return 63 - __builtin_clzll(n);
#endif
#elif defined(_WIN32)
#ifdef _MSC_VER
    unsigned long index;
    if (_BitScanReverse(&index, n)) return index;
    return 0; // Same note as above.
#else
    return 31 - __builtin_clz(n);
#endif
#else
// Handle non-Windows platforms. You can further differentiate between 32-bit and 64-bit if needed.
#if defined(__LP64__)
    return 63 - __builtin_clzll(n);
#else
    return 31 - __builtin_clz(n);
#endif
#endif
}

/**
 *  @brief  Char-level lexicographic comparison of two strings.
 *          Doesn't provide major performance improvements, but helps avoid the LibC dependency.
 */
inline static sz_bool_t sz_is_less_ascii(sz_string_start_t a,
                                         sz_size_t const a_length,
                                         sz_string_start_t b,
                                         sz_size_t const b_length) {

    sz_size_t min_length = (a_length < b_length) ? a_length : b_length;
    sz_string_start_t const min_end = a + min_length;
    while (a + 8 <= min_end && sz_u64_unaligned_load(a) == sz_u64_unaligned_load(b)) a += 8, b += 8;
    while (a != min_end && *a == *b) a++, b++;
    return a != min_end ? (*a < *b) : (a_length < b_length);
}

/**
 *  @brief  Char-level lexicographic comparison of two strings, insensitive to the case of ASCII symbols.
 *          Doesn't provide major performance improvements, but helps avoid the LibC dependency.
 */
inline static sz_bool_t sz_is_less_uncased_ascii(sz_string_start_t const a,
                                                 sz_size_t const a_length,
                                                 sz_string_start_t const b,
                                                 sz_size_t const b_length) {

    sz_size_t min_length = (a_length < b_length) ? a_length : b_length;
    for (sz_size_t i = 0; i < min_length; ++i) {
        char a_lower = sz_tolower_ascii(a[i]);
        char b_lower = sz_tolower_ascii(b[i]);
        if (a_lower < b_lower) return 1;
        if (a_lower > b_lower) return 0;
    }
    return a_length < b_length;
}

/**
 *  @brief  Helper, that swaps two 64-bit integers representing the order of elements in the sequence.
 */
inline static void _sz_swap_order(sz_u64_t *a, sz_u64_t *b) {
    sz_u64_t t = *a;
    *a = *b;
    *b = t;
}

struct sz_sequence_t;

typedef sz_string_start_t (*sz_sequence_member_start_t)(struct sz_sequence_t const *, sz_size_t);
typedef sz_size_t (*sz_sequence_member_length_t)(struct sz_sequence_t const *, sz_size_t);
typedef sz_bool_t (*sz_sequence_predicate_t)(struct sz_sequence_t const *, sz_size_t);
typedef sz_bool_t (*sz_sequence_comparator_t)(struct sz_sequence_t const *, sz_size_t, sz_size_t);
typedef sz_bool_t (*sz_string_is_less_t)(sz_string_start_t, sz_size_t, sz_string_start_t, sz_size_t);

typedef struct sz_sequence_t {
    sz_u64_t *order;
    sz_size_t count;
    sz_sequence_member_start_t get_start;
    sz_sequence_member_length_t get_length;
    void const *handle;
} sz_sequence_t;

/**
 *  @brief  Similar to `std::partition`, given a predicate splits the sequence into two parts.
 *          The algorithm is unstable, meaning that elements may change relative order, as long
 *          as they are in the right partition. This is the simpler algorithm for partitioning.
 */
inline static sz_size_t sz_partition(sz_sequence_t *sequence, sz_sequence_predicate_t predicate) {

    sz_size_t matches = 0;
    while (matches != sequence->count && predicate(sequence, sequence->order[matches])) ++matches;

    for (sz_size_t i = matches + 1; i < sequence->count; ++i)
        if (predicate(sequence, sequence->order[i]))
            _sz_swap_order(sequence->order + i, sequence->order + matches), ++matches;

    return matches;
}

/**
 *  @brief  Inplace `std::set_union` for two consecutive chunks forming the same continuous `sequence`.
 *
 *  @param partition The number of elements in the first sub-sequence in `sequence`.
 *  @param less Comparison function, to determine the lexicographic ordering.
 */
inline static void sz_merge(sz_sequence_t *sequence, sz_size_t partition, sz_sequence_comparator_t less) {

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

inline static void sz_sort_insertion(sz_sequence_t *sequence, sz_sequence_comparator_t less) {
    sz_u64_t *keys = sequence->order;
    sz_size_t keys_count = sequence->count;
    for (sz_size_t i = 1; i < keys_count; i++) {
        sz_u64_t i_key = keys[i];
        sz_size_t j = i;
        for (; j > 0 && less(sequence, i_key, keys[j - 1]); --j) keys[j] = keys[j - 1];
        keys[j] = i_key;
    }
}

inline static void _sz_sift_down(
    sz_sequence_t *sequence, sz_sequence_comparator_t less, sz_u64_t *order, sz_size_t start, sz_size_t end) {
    sz_size_t root = start;
    while (2 * root + 1 <= end) {
        sz_size_t child = 2 * root + 1;
        if (child + 1 <= end && less(sequence, order[child], order[child + 1])) { child++; }
        if (!less(sequence, order[root], order[child])) { return; }
        _sz_swap_order(order + root, order + child);
        root = child;
    }
}

inline static void _sz_heapify(sz_sequence_t *sequence,
                               sz_sequence_comparator_t less,
                               sz_u64_t *order,
                               sz_size_t count) {
    sz_size_t start = (count - 2) / 2;
    while (1) {
        _sz_sift_down(sequence, less, order, start, count - 1);
        if (start == 0) return;
        start--;
    }
}

inline static void _sz_heapsort(sz_sequence_t *sequence,
                                sz_sequence_comparator_t less,
                                sz_size_t first,
                                sz_size_t last) {
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

inline static void _sz_introsort(
    sz_sequence_t *sequence, sz_sequence_comparator_t less, sz_size_t first, sz_size_t last, sz_size_t depth) {

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

inline static void sz_sort_introsort(sz_sequence_t *sequence, sz_sequence_comparator_t less) {
    sz_size_t depth_limit = 2 * sz_log2i(sequence->count);
    _sz_introsort(sequence, less, 0, sequence->count, depth_limit);
}

inline static void _sz_sort_recursion( //
    sz_sequence_t *sequence,
    sz_size_t bit_idx,
    sz_size_t bit_max,
    sz_sequence_comparator_t comparator,
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

inline static sz_bool_t _sz_sort_compare_less_ascii(sz_sequence_t *sequence, sz_size_t i_key, sz_size_t j_key) {
    sz_string_start_t i_str = sequence->get_start(sequence, i_key);
    sz_size_t i_len = sequence->get_length(sequence, i_key);
    sz_string_start_t j_str = sequence->get_start(sequence, j_key);
    sz_size_t j_len = sequence->get_length(sequence, j_key);
    return sz_is_less_ascii(i_str, i_len, j_str, j_len);
}

inline static sz_bool_t _sz_sort_compare_less_uncased_ascii(sz_sequence_t *sequence, sz_size_t i_key, sz_size_t j_key) {
    sz_string_start_t i_str = sequence->get_start(sequence, i_key);
    sz_size_t i_len = sequence->get_length(sequence, i_key);
    sz_string_start_t j_str = sequence->get_start(sequence, j_key);
    sz_size_t j_len = sequence->get_length(sequence, j_key);
    return sz_is_less_uncased_ascii(i_str, i_len, j_str, j_len);
}

typedef struct sz_sort_config_t {
    sz_bool_t case_insensitive;
    sz_size_t partial_order_length;
} sz_sort_config_t;

/**
 *  @brief  Sorting algorithm, combining Radix Sort for the first 32 bits of every word
 *          and a follow-up by a more conventional sorting procedure on equally prefixed parts.
 */
inline static void sz_sort(sz_sequence_t *sequence, sz_sort_config_t const *config) {

    sz_bool_t case_insensitive = config && config->case_insensitive;
    sz_size_t partial_order_length =
        config && config->partial_order_length ? config->partial_order_length : sequence->count;

    // Export up to 4 bytes into the `sequence` bits themselves
    for (sz_size_t i = 0; i != sequence->count; ++i) {
        sz_string_start_t begin = sequence->get_start(sequence, sequence->order[i]);
        sz_size_t length = sequence->get_length(sequence, sequence->order[i]);
        length = length > 4ull ? 4ull : length;
        char *prefix = (char *)&sequence->order[i];
        for (sz_size_t j = 0; j != length; ++j) prefix[7 - j] = begin[j];
        if (case_insensitive) {
            prefix[0] = sz_tolower_ascii(prefix[0]);
            prefix[1] = sz_tolower_ascii(prefix[1]);
            prefix[2] = sz_tolower_ascii(prefix[2]);
            prefix[3] = sz_tolower_ascii(prefix[3]);
        }
    }

    sz_sequence_comparator_t comparator = (sz_sequence_comparator_t)_sz_sort_compare_less_ascii;
    if (case_insensitive) comparator = (sz_sequence_comparator_t)_sz_sort_compare_less_uncased_ascii;

    // Perform optionally-parallel radix sort on them
    _sz_sort_recursion(sequence, 0, 32, comparator, partial_order_length);
}

/**
 *  @return Amount of temporary memory (in bytes) needed to efficiently compute
 *          the Levenshtein distance between two strings of given size.
 */
inline static sz_size_t sz_levenshtein_memory_needed(sz_size_t _, sz_size_t b_length) {
    return b_length + b_length + 2;
}

/**
 *  @brief  Auxiliary function, that computes the minimum of three values.
 */
inline static levenshtein_distance_t _sz_levenshtein_minimum( //
    levenshtein_distance_t const a,
    levenshtein_distance_t const b,
    levenshtein_distance_t const c) {

    return (a < b ? (a < c ? a : c) : (b < c ? b : c));
}

/**
 *  @brief  Levenshtein String Similarity function, implemented with linear memory consumption.
 *          It accepts an upper bound on the possible error. Quadratic complexity in time, linear in space.
 */
inline static levenshtein_distance_t sz_levenshtein( //
    sz_string_start_t const a,
    sz_size_t const a_length,
    sz_string_start_t const b,
    sz_size_t const b_length,
    levenshtein_distance_t const bound,
    void *buffer) {

    // If one of the strings is empty - the edit distance is equal to the length of the other one
    if (a_length == 0) return b_length <= bound ? b_length : bound;
    if (b_length == 0) return a_length <= bound ? a_length : bound;

    // If the difference in length is beyond the `bound`, there is no need to check at all
    if (a_length > b_length) {
        if (a_length - b_length > bound) return bound + 1;
    }
    else {
        if (b_length - a_length > bound) return bound + 1;
    }

    levenshtein_distance_t *previous_distances = (levenshtein_distance_t *)buffer;
    levenshtein_distance_t *current_distances = previous_distances + b_length + 1;

    for (sz_size_t idx_b = 0; idx_b != (b_length + 1); ++idx_b) previous_distances[idx_b] = idx_b;

    for (sz_size_t idx_a = 0; idx_a != a_length; ++idx_a) {
        current_distances[0] = idx_a + 1;

        // Initialize min_distance with a value greater than bound
        levenshtein_distance_t min_distance = bound;

        for (sz_size_t idx_b = 0; idx_b != b_length; ++idx_b) {
            levenshtein_distance_t cost_deletion = previous_distances[idx_b + 1] + 1;
            levenshtein_distance_t cost_insertion = current_distances[idx_b] + 1;
            levenshtein_distance_t cost_substitution = previous_distances[idx_b] + (a[idx_a] != b[idx_b]);
            current_distances[idx_b + 1] = _sz_levenshtein_minimum(cost_deletion, cost_insertion, cost_substitution);

            // Keep track of the minimum distance seen so far in this row
            if (current_distances[idx_b + 1] < min_distance) { min_distance = current_distances[idx_b + 1]; }
        }

        // If the minimum distance in this row exceeded the bound, return early
        if (min_distance > bound) return bound;

        // Swap previous_distances and current_distances pointers
        levenshtein_distance_t *temp = previous_distances;
        previous_distances = current_distances;
        current_distances = temp;
    }

    return previous_distances[b_length] <= bound ? previous_distances[b_length] : bound;
}

inline static sz_u32_t sz_hash_crc32_swar(sz_string_start_t start, sz_size_t length) {
    /*
     * The following CRC lookup table was generated automagically using the
     * following model parameters:
     *
     * Generator Polynomial = ................. 0x1EDC6F41
     * Generator Polynomial Length = .......... 32 bits
     * Reflected Bits = ....................... TRUE
     * Table Generation Offset = .............. 32 bits
     * Number of Slices = ..................... 8 slices
     * Slice Lengths = ........................ 8 8 8 8 8 8 8 8
     */

    static sz_u32_t const table[256] = {
        0x00000000, 0xF26B8303, 0xE13B70F7, 0x1350F3F4, 0xC79A971F, 0x35F1141C, 0x26A1E7E8, 0xD4CA64EB, //
        0x8AD958CF, 0x78B2DBCC, 0x6BE22838, 0x9989AB3B, 0x4D43CFD0, 0xBF284CD3, 0xAC78BF27, 0x5E133C24, //
        0x105EC76F, 0xE235446C, 0xF165B798, 0x030E349B, 0xD7C45070, 0x25AFD373, 0x36FF2087, 0xC494A384, //
        0x9A879FA0, 0x68EC1CA3, 0x7BBCEF57, 0x89D76C54, 0x5D1D08BF, 0xAF768BBC, 0xBC267848, 0x4E4DFB4B, //
        0x20BD8EDE, 0xD2D60DDD, 0xC186FE29, 0x33ED7D2A, 0xE72719C1, 0x154C9AC2, 0x061C6936, 0xF477EA35, //
        0xAA64D611, 0x580F5512, 0x4B5FA6E6, 0xB93425E5, 0x6DFE410E, 0x9F95C20D, 0x8CC531F9, 0x7EAEB2FA, //
        0x30E349B1, 0xC288CAB2, 0xD1D83946, 0x23B3BA45, 0xF779DEAE, 0x05125DAD, 0x1642AE59, 0xE4292D5A, //
        0xBA3A117E, 0x4851927D, 0x5B016189, 0xA96AE28A, 0x7DA08661, 0x8FCB0562, 0x9C9BF696, 0x6EF07595, //
        0x417B1DBC, 0xB3109EBF, 0xA0406D4B, 0x522BEE48, 0x86E18AA3, 0x748A09A0, 0x67DAFA54, 0x95B17957, //
        0xCBA24573, 0x39C9C670, 0x2A993584, 0xD8F2B687, 0x0C38D26C, 0xFE53516F, 0xED03A29B, 0x1F682198, //
        0x5125DAD3, 0xA34E59D0, 0xB01EAA24, 0x42752927, 0x96BF4DCC, 0x64D4CECF, 0x77843D3B, 0x85EFBE38, //
        0xDBFC821C, 0x2997011F, 0x3AC7F2EB, 0xC8AC71E8, 0x1C661503, 0xEE0D9600, 0xFD5D65F4, 0x0F36E6F7, //
        0x61C69362, 0x93AD1061, 0x80FDE395, 0x72966096, 0xA65C047D, 0x5437877E, 0x4767748A, 0xB50CF789, //
        0xEB1FCBAD, 0x197448AE, 0x0A24BB5A, 0xF84F3859, 0x2C855CB2, 0xDEEEDFB1, 0xCDBE2C45, 0x3FD5AF46, //
        0x7198540D, 0x83F3D70E, 0x90A324FA, 0x62C8A7F9, 0xB602C312, 0x44694011, 0x5739B3E5, 0xA55230E6, //
        0xFB410CC2, 0x092A8FC1, 0x1A7A7C35, 0xE811FF36, 0x3CDB9BDD, 0xCEB018DE, 0xDDE0EB2A, 0x2F8B6829, //
        0x82F63B78, 0x709DB87B, 0x63CD4B8F, 0x91A6C88C, 0x456CAC67, 0xB7072F64, 0xA457DC90, 0x563C5F93, //
        0x082F63B7, 0xFA44E0B4, 0xE9141340, 0x1B7F9043, 0xCFB5F4A8, 0x3DDE77AB, 0x2E8E845F, 0xDCE5075C, //
        0x92A8FC17, 0x60C37F14, 0x73938CE0, 0x81F80FE3, 0x55326B08, 0xA759E80B, 0xB4091BFF, 0x466298FC, //
        0x1871A4D8, 0xEA1A27DB, 0xF94AD42F, 0x0B21572C, 0xDFEB33C7, 0x2D80B0C4, 0x3ED04330, 0xCCBBC033, //
        0xA24BB5A6, 0x502036A5, 0x4370C551, 0xB11B4652, 0x65D122B9, 0x97BAA1BA, 0x84EA524E, 0x7681D14D, //
        0x2892ED69, 0xDAF96E6A, 0xC9A99D9E, 0x3BC21E9D, 0xEF087A76, 0x1D63F975, 0x0E330A81, 0xFC588982, //
        0xB21572C9, 0x407EF1CA, 0x532E023E, 0xA145813D, 0x758FE5D6, 0x87E466D5, 0x94B49521, 0x66DF1622, //
        0x38CC2A06, 0xCAA7A905, 0xD9F75AF1, 0x2B9CD9F2, 0xFF56BD19, 0x0D3D3E1A, 0x1E6DCDEE, 0xEC064EED, //
        0xC38D26C4, 0x31E6A5C7, 0x22B65633, 0xD0DDD530, 0x0417B1DB, 0xF67C32D8, 0xE52CC12C, 0x1747422F, //
        0x49547E0B, 0xBB3FFD08, 0xA86F0EFC, 0x5A048DFF, 0x8ECEE914, 0x7CA56A17, 0x6FF599E3, 0x9D9E1AE0, //
        0xD3D3E1AB, 0x21B862A8, 0x32E8915C, 0xC083125F, 0x144976B4, 0xE622F5B7, 0xF5720643, 0x07198540, //
        0x590AB964, 0xAB613A67, 0xB831C993, 0x4A5A4A90, 0x9E902E7B, 0x6CFBAD78, 0x7FAB5E8C, 0x8DC0DD8F, //
        0xE330A81A, 0x115B2B19, 0x020BD8ED, 0xF0605BEE, 0x24AA3F05, 0xD6C1BC06, 0xC5914FF2, 0x37FACCF1, //
        0x69E9F0D5, 0x9B8273D6, 0x88D28022, 0x7AB90321, 0xAE7367CA, 0x5C18E4C9, 0x4F48173D, 0xBD23943E, //
        0xF36E6F75, 0x0105EC76, 0x12551F82, 0xE03E9C81, 0x34F4F86A, 0xC69F7B69, 0xD5CF889D, 0x27A40B9E, //
        0x79B737BA, 0x8BDCB4B9, 0x988C474D, 0x6AE7C44E, 0xBE2DA0A5, 0x4C4623A6, 0x5F16D052, 0xAD7D5351  //
    };

    sz_u32_t crc = 0xFFFFFFFF;
    for (sz_string_start_t const end = start + length; start != end; ++start)
        crc = (crc >> 8) ^ table[(crc ^ (sz_u32_t)*start) & 0xff];
    return crc ^ 0xFFFFFFFF;
}

#if defined(__ARM_FEATURE_CRC32)
inline static sz_u32_t sz_hash_crc32_arm(sz_string_start_t start, sz_size_t length) {
    sz_u32_t crc = 0xFFFFFFFF;
    sz_string_start_t const end = start + length;

    // Align the input to the word boundary
    while (((unsigned long)start & 7ull) && start != end) { crc = __crc32cb(crc, *start), start++; }

    // Process the body 8 bytes at a time
    while (start + 8 <= end) { crc = __crc32cd(crc, *(unsigned long long *)start), start += 8; }

    // Process the tail bytes
    if (start + 4 <= end) { crc = __crc32cw(crc, *(unsigned int *)start), start += 4; }
    if (start + 2 <= end) { crc = __crc32ch(crc, *(unsigned short *)start), start += 2; }
    if (start < end) { crc = __crc32cb(crc, *start); }
    return crc ^ 0xFFFFFFFF;
}
#endif

#if defined(__SSE4_2__)
inline static sz_u32_t sz_hash_crc32_sse(sz_string_start_t start, sz_size_t length) {
    sz_u32_t crc = 0xFFFFFFFF;
    sz_string_start_t const end = start + length;

    // Align the input to the word boundary
    while (((unsigned long)start & 7ull) && start != end) { crc = _mm_crc32_u8(crc, *start), start++; }

    // Process the body 8 bytes at a time
    while (start + 8 <= end) { crc = (sz_u32_t)_mm_crc32_u64(crc, *(unsigned long long *)start), start += 8; }

    // Process the tail bytes
    if (start + 4 <= end) { crc = _mm_crc32_u32(crc, *(unsigned int *)start), start += 4; }
    if (start + 2 <= end) { crc = _mm_crc32_u16(crc, *(unsigned short *)start), start += 2; }
    if (start < end) { crc = _mm_crc32_u8(crc, *start); }
    return crc ^ 0xFFFFFFFF;
}
#endif

/**
 *  @brief  Hashes provided string using hardware-accelerated CRC32 instructions.
 */
inline static sz_u32_t sz_hash_crc32(sz_string_start_t start, sz_size_t length) {
#if defined(__ARM_FEATURE_CRC32)
    return sz_hash_crc32_arm(start, length);
#elif defined(__SSE4_2__)
    return sz_hash_crc32_sse(start, length);
#else
    return sz_hash_crc32_swar(start, length);
#endif
}

#ifdef __cplusplus
}
#endif

#undef popcount64
#undef ctz64
#undef clz64

#endif // STRINGZILLA_H_
