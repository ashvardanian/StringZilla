#ifndef STRINGZILLA_H_
#define STRINGZILLA_H_

#include <search.h> // `qsort_s`
#include <stdlib.h> // `qsort_r`
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
#define strncasecmp _strnicmp
#define strcasecmp _stricmp
#else
#define popcount64 __builtin_popcountll
#define ctz64 __builtin_ctzll
#endif

#ifdef __cplusplus
extern "C" {
#endif

#if defined(__LP64__) || defined(_LP64) || defined(__x86_64__) || defined(_WIN64)
typedef unsigned long sz_size_t; // 64-bit on most platforms when pointers are 64-bit
#else
typedef unsigned sz_size_t; // 32-bit on most platforms when pointers are 32-bit
#endif

typedef unsigned sz_u32_t;           // Always 32 bits
typedef unsigned long long sz_u64_t; // Always 64 bits

typedef union sz_quadgram_t {
    unsigned u32;
    unsigned char u8s[4];
} sz_quadgram_t; // Always 32-bit unsigned integer, representing 8 bytes/characters

typedef union sz_octogram_t {
    unsigned long long u64;
    unsigned char u8s[8];
} sz_octogram_t; // Always 64-bit unsigned integer, representing 8 bytes/characters

inline static sz_size_t sz_divide_round_up(sz_size_t x, sz_size_t divisor) { return (x + (divisor - 1)) / divisor; }

inline static sz_size_t sz_tolower_ascii(char c) {
    static char lowered[256] = {
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
    return lowered[(int)c];
}

inline static sz_size_t sz_toupper_ascii(char c) {
    static char upped[256] = {
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
    return upped[(int)c];
}

/**
 *  @brief This is a faster alternative to `strncmp(a, b, length) == 0`.
 *  @return 1 for `true`, and 0 for `false`.
 */
inline static int sz_equal(char const *a, char const *b, sz_size_t length) {
    char const *const a_end = a + length;
    while (a != a_end && *a == *b) a++, b++;
    return a_end == a;
}

typedef struct sz_haystack_t {
    char const *start;
    sz_size_t length;
} sz_haystack_t;

typedef struct sz_needle_t {
    char const *start;
    sz_size_t length;
    sz_size_t quadgram_offset;
} sz_needle_t;

/**
 *  @brief  SWAR single-character counting procedure, jumping 8 bytes at a time.
 */
inline static sz_size_t sz_count_unigram_swar(sz_haystack_t h, char n) {

    sz_size_t result = 0;
    char const *text = h.start;
    char const *end = h.start + h.length;

    // Process the misaligned head, to void UB on unaligned 64-bit loads.
    for (; ((unsigned long)text & 7ul) && text < end; ++text) result += *text == n;

    // This code simulates hyper-scalar execution, comparing 8 characters at a time.
    sz_u64_t nnnnnnnn = n;
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

    for (; text < end; ++text) result += *text == n;
    return result;
}

/**
 *  @brief  SWAR single-character search in string, jumping 8 bytes at a time.
 */
inline static sz_size_t sz_find_unigram_swar(sz_haystack_t h, char n) {

    char const *text = h.start;
    char const *end = h.start + h.length;

    // Process the misaligned head, to void UB on unaligned 64-bit loads.
    for (; ((unsigned long)text & 7ul) && text < end; ++text)
        if (*text == n) return text - h.start;

    // This code simulates hyper-scalar execution, analyzing 8 offsets at a time.
    sz_u64_t nnnnnnnn = n;
    nnnnnnnn |= nnnnnnnn << 8;  // broadcast `n` into `nnnnnnnn`
    nnnnnnnn |= nnnnnnnn << 16; // broadcast `n` into `nnnnnnnn`
    nnnnnnnn |= nnnnnnnn << 32; // broadcast `n` into `nnnnnnnn`
    for (; text + 8 <= end; text += 8) {
        sz_u64_t text_slice = *(sz_u64_t const *)text;
        sz_u64_t match_indicators = ~(text_slice ^ nnnnnnnn);
        match_indicators &= match_indicators >> 1;
        match_indicators &= match_indicators >> 2;
        match_indicators &= match_indicators >> 4;
        match_indicators &= 0x0101010101010101;

        if (match_indicators != 0) return text - h.start + ctz64(match_indicators) / 8;
    }

    for (; text < end; ++text)
        if (*text == n) return text - h.start;
    return h.length;
}

/**
 *  @brief  SWAR character-bigram search in string, jumping 8 bytes at a time.
 */
inline static sz_size_t sz_find_bigram_swar(sz_haystack_t h, char const *n) {

    char const *text = h.start;
    char const *end = h.start + h.length;

    // Process the misaligned head, to void UB on unaligned 64-bit loads.
    for (; ((unsigned long)text & 7ul) && text + 2 <= end; ++text)
        if (text[0] == n[0] && text[1] == n[1]) return text - h.start;

    // This code simulates hyper-scalar execution, analyzing 7 offsets at a time.
    sz_u64_t nnnn = ((sz_u64_t)(n[0]) << 0) | ((sz_u64_t)(n[1]) << 8); // broadcast `n` into `nnnn`
    nnnn |= nnnn << 16;                                                // broadcast `n` into `nnnn`
    nnnn |= nnnn << 32;                                                // broadcast `n` into `nnnn`
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
            return text - h.start + ctz64(match_indicators) / 8;
        }
    }

    for (; text + 2 <= end; ++text)
        if (text[0] == n[0] && text[1] == n[1]) return text - h.start;
    return h.length;
}

/**
 *  @brief  SWAR character-trigram search in string, jumping 8 bytes at a time.
 */
inline static sz_size_t sz_find_trigram_swar(sz_haystack_t h, char const *n) {

    char const *text = h.start;
    char const *end = h.start + h.length;

    // Process the misaligned head, to void UB on unaligned 64-bit loads.
    for (; ((unsigned long)text & 7ul) && text + 3 <= end; ++text)
        if (text[0] == n[0] && text[1] == n[1] && text[2] == n[2]) return text - h.start;

    // This code simulates hyper-scalar execution, analyzing 6 offsets at a time.
    // We have two unused bytes at the end.
    sz_u64_t nn = (sz_u64_t)(n[0] << 0) | ((sz_u64_t)(n[1]) << 8) | ((sz_u64_t)(n[2]) << 16); // broadcast `n` into `nn`
    nn |= nn << 24;                                                                           // broadcast `n` into `nn`
    nn <<= 16;                                                                                // broadcast `n` into `nn`

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
        if (match_indicators != 0) return text - h.start + ctz64(match_indicators) / 8;
    }

    for (; text + 3 <= end; ++text)
        if (text[0] == n[0] && text[1] == n[1] && text[2] == n[2]) return text - h.start;
    return h.length;
}

/**
 *  @brief  SWAR character-quadgram search in string, jumping 8 bytes at a time.
 */
inline static sz_size_t sz_find_quadgram_swar(sz_haystack_t h, char const *n) {

    char const *text = h.start;
    char const *end = h.start + h.length;

    // Process the misaligned head, to void UB on unaligned 64-bit loads.
    for (; ((unsigned long)text & 7ul) && text + 4 <= end; ++text)
        if (text[0] == n[0] && text[1] == n[1] && text[2] == n[2] && text[3] == n[3]) return text - h.start;

    // This code simulates hyper-scalar execution, analyzing 4 offsets at a time.
    sz_u64_t nn = (sz_u64_t)(n[0] << 0) | ((sz_u64_t)(n[1]) << 8) | ((sz_u64_t)(n[2]) << 16) | ((sz_u64_t)(n[3]) << 24);
    nn |= nn << 32;

    //
    unsigned char lookup[16] = {0};
    lookup[0x2] = lookup[0x6] = lookup[0xA] = lookup[0xE] = 1;
    lookup[0x4] = lookup[0xC] = 2;
    lookup[0x8] = 3;

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
            return text - h.start + lookup[match_indicators];
        }
    }

    for (; text + 4 <= end; ++text)
        if (text[0] == n[0] && text[1] == n[1] && text[2] == n[2] && text[3] == n[3]) return text - h.start;
    return h.length;
}

/**
 *  @brief  Trivial substring search with scalar code. Instead of comparing characters one-by-one
 *          it compares 4-byte quadgrams first, most commonly prefixes. It's computationally cheaper.
 *          Matching performance fluctuates between 1 GB/s and 3,5 GB/s per core.
 */
inline static sz_size_t sz_find_substr_swar(sz_haystack_t h, sz_needle_t n) {

    if (h.length < n.length) return h.length;

    switch (n.length) {
    case 0: return 0;
    case 1: return sz_find_unigram_swar(h, *n.start);
    case 2: return sz_find_bigram_swar(h, n.start);
    case 3: return sz_find_trigram_swar(h, n.start);
    case 4: return sz_find_quadgram_swar(h, n.start);
    default: {
        char const *text = h.start;
        char const *const end = h.start + h.length;

        sz_quadgram_t n_quadgram, h_quadgram;
        sz_size_t const n_suffix_len = n.length - 4 - n.quadgram_offset;
        char const *n_suffix_ptr = n.start + 4 + n.quadgram_offset;
        n_quadgram.u8s[0] = n.start[n.quadgram_offset];
        n_quadgram.u8s[1] = n.start[n.quadgram_offset + 1];
        n_quadgram.u8s[2] = n.start[n.quadgram_offset + 2];
        n_quadgram.u8s[3] = n.start[n.quadgram_offset + 3];
        h_quadgram.u8s[0] = h.start[0];
        h_quadgram.u8s[1] = h.start[1];
        h_quadgram.u8s[2] = h.start[2];
        h_quadgram.u8s[3] = h.start[3];

        text += n.quadgram_offset;
        while (text + n.length <= end) {
            h_quadgram.u8s[3] = text[3];
            if (h_quadgram.u32 == n_quadgram.u32)                                       // Match quadgram.
                if (sz_equal(text + 4, n_suffix_ptr, n_suffix_len))                     // Match suffix.
                    if (sz_equal(text - n.quadgram_offset, n.start, n.quadgram_offset)) // Match prefix.
                        return text - h.start - n.quadgram_offset;

            h_quadgram.u32 >>= 8;
            ++text;
        }
        return h.length;
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
inline static sz_size_t sz_find_substr_avx2(sz_haystack_t h, sz_needle_t n) {

    // Precomputed constants
    char const *const end = h.start + h.length;
    sz_quadgram_t quadgram = 0;
    sz_quadgram_t mask = 0;
    switch (n.length) {
    case 1: memset(&mask, 0xFF, 1), memcpy(&quadgram, n.start, 1); break;
    case 2: memset(&mask, 0xFF, 2), memcpy(&quadgram, n.start, 2); break;
    case 3: memset(&mask, 0xFF, 3), memcpy(&quadgram, n.start, 3); break;
    default: memset(&mask, 0xFF, 4), memcpy(&quadgram, n.start, 4); break;
    }

    __m256i const quadgrams = _mm256_set1_epi32(quadgram.u32);
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
    char const *text = h.start;
    for (; (text + n.length + 32) <= end; text += 32) {

        // Performing many unaligned loads ends up being faster than loading once and shuffling around.
        __m256i texts0 = _mm256_and_si256(_mm256_loadu_si256((__m256i const *)(text + 0)), masks);
        int matches0 = _mm256_movemask_epi8(_mm256_cmpeq_epi32(texts0, quadgrams));
        __m256i texts1 = _mm256_and_si256(_mm256_loadu_si256((__m256i const *)(text + 1)), masks);
        int matches1 = _mm256_movemask_epi8(_mm256_cmpeq_epi32(texts1, quadgrams));
        __m256i text2 = _mm256_and_si256(_mm256_loadu_si256((__m256i const *)(text + 2)), masks);
        int matches2 = _mm256_movemask_epi8(_mm256_cmpeq_epi32(text2, quadgrams));
        __m256i texts3 = _mm256_and_si256(_mm256_loadu_si256((__m256i const *)(text + 3)), masks);
        int matches3 = _mm256_movemask_epi8(_mm256_cmpeq_epi32(texts3, quadgrams));

        if (matches0 | matches1 | matches2 | matches3) {
            for (sz_size_t i = 0; i < 32; i++) {
                if (sz_equal(text + i, n.start, n.length)) return i + (text - h.start);
            }
        }
    }

    // Don't forget the last (up to 35) characters.
    sz_haystack_t tail;
    tail.start = text;
    tail.length = end - text;
    size_t tail_match = sz_find_substr_swar(tail, n);
    return text + tail_match - h.start;
}

#endif // x86 AVX2

#if defined(__ARM_NEON)

/**
 *  @brief  Substring-search implementation, leveraging Arm Neon intrinsics and speculative
 *          execution capabilities on modern CPUs. Performing 4 unaligned vector loads per cycle
 *          was practically more efficient than loading once and shifting around, as introduces
 *          less data dependencies.
 */
inline static sz_size_t sz_find_substr_neon(sz_haystack_t h, sz_needle_t n) {

    // Precomputed constants
    char const *const end = h.start + h.length;
    sz_quadgram_t quadgram = {};
    sz_quadgram_t mask = {};
    switch (n.length) {
    case 1: mask.u8s[0] = 0xFF, quadgram.u8s[0] = n.start[0]; break;
    case 2: mask.u8s[0] = mask.u8s[1] = 0xFF, quadgram.u8s[0] = n.start[0], quadgram.u8s[1] = n.start[1]; break;
    case 3:
        mask.u8s[0] = mask.u8s[1] = mask.u8s[2] = 0xFF, quadgram.u8s[0] = n.start[0], quadgram.u8s[1] = n.start[1],
        quadgram.u8s[2] = n.start[2];
        break;
    default:
        mask.u32 = 0xFFFFFFFF, quadgram.u8s[0] = n.start[0], quadgram.u8s[1] = n.start[1], quadgram.u8s[2] = n.start[2],
        quadgram.u8s[3] = n.start[3];
        break;
    }

    uint32x4_t const quadgrams = vld1q_dup_u32(&quadgram.u32);
    uint32x4_t const masks = vld1q_dup_u32(&mask);
    uint32x4_t matches, matches0, matches1, matches2, matches3;

    char const *text = h.start;
    while (text + n.length + 16 <= end) {

        // Each of the following `matchesX` contains only 4 relevant bits - one per word.
        // Each signifies a match at the given offset.
        matches0 = vceqq_u32(vandq_u32(vld1q_u32((sz_quadgram_t const *)(text + 0)), masks), quadgrams);
        matches1 = vceqq_u32(vandq_u32(vld1q_u32((sz_quadgram_t const *)(text + 1)), masks), quadgrams);
        matches2 = vceqq_u32(vandq_u32(vld1q_u32((sz_quadgram_t const *)(text + 2)), masks), quadgrams);
        matches3 = vceqq_u32(vandq_u32(vld1q_u32((sz_quadgram_t const *)(text + 3)), masks), quadgrams);
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
            size_t first_match_offset = __builtin_ctz(matches_u16);
            if (n.length > 4) {
                if (sz_equal(text + first_match_offset + 4, n.start + 4, n.length - 4))
                    return text + first_match_offset - h.start;
                else
                    text += first_match_offset + 1;
            }
            else
                return text + first_match_offset - h.start;
        }
        else
            text += 16;
    }

    // Don't forget the last (up to 16+3=19) characters.
    sz_haystack_t tail;
    tail.start = text;
    tail.length = end - text;
    size_t tail_match = sz_find_substr_swar(tail, n);
    return text + tail_match - h.start;
}

#endif // Arm Neon

inline static sz_size_t sz_count_char(sz_haystack_t h, char n) { return sz_count_unigram_swar(h, n); }
inline static sz_size_t sz_find_unigram(sz_haystack_t h, char n) { return sz_find_unigram_swar(h, n); }

inline static sz_size_t sz_find_substr(sz_haystack_t h, sz_needle_t n) {
    if (h.length < n.length) return h.length;

#if defined(__ARM_NEON)
    return sz_find_substr_neon(h, n);
#elif defined(__AVX2__)
    return sz_find_substr_avx2(h, n);
#else
    return sz_find_substr_swar(h, n);
#endif
}

inline static void sz_swap(sz_size_t *a, sz_size_t *b) {
    sz_size_t t = *a;
    *a = *b;
    *b = t;
}

typedef char const *(*sz_sequence_get_start_t)(void const *, sz_size_t);
typedef sz_size_t (*sz_sequence_get_length_t)(void const *, sz_size_t);
typedef int (*sz_sequence_predicate_t)(void const *, sz_size_t);
typedef int (*sz_sequence_comparator_t)(void const *, sz_size_t, sz_size_t);

// Define a type for the comparison function, depending on the platform.
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__) || defined(__NT__) || defined(__APPLE__)
typedef int (*sz_qsort_comparison_func_t)(void *, void const *, void const *);
#else
typedef int (*sz_qsort_comparison_func_t)(void const *, void const *, void *);
#endif

typedef struct sz_sequence_t {
    sz_size_t *order;
    sz_size_t count;
    sz_sequence_get_start_t get_start;
    sz_sequence_get_length_t get_length;
    void const *handle;
} sz_sequence_t;

/**
 *  @brief  Similar to `std::partition`, given a predicate splits the
 *          sequence into two parts.
 */
inline static sz_size_t sz_partition(sz_sequence_t *sequence, sz_sequence_predicate_t predicate) {

    sz_size_t matches = 0;
    while (matches != sequence->count && predicate(sequence->handle, sequence->order[matches])) ++matches;

    for (sz_size_t i = matches + 1; i < sequence->count; ++i)
        if (predicate(sequence->handle, sequence->order[i]))
            sz_swap(sequence->order + i, sequence->order + matches), ++matches;

    return matches;
}

/**
 *  @brief  Inplace `std::set_union` for two consecutive chunks forming
 *          the same continuous sequence.
 */
inline static void sz_merge(sz_sequence_t *sequence, sz_size_t partition, sz_sequence_comparator_t less) {

    sz_size_t start_b = partition + 1;

    // If the direct merge is already sorted
    if (!less(sequence->handle, sequence->order[start_b], sequence->order[partition])) return;

    sz_size_t start_a = 0;
    while (start_a <= partition && start_b <= sequence->count) {

        // If element 1 is in right place
        if (!less(sequence->handle, sequence->order[start_b], sequence->order[start_a])) { start_a++; }
        else {
            sz_size_t value = sequence->order[start_b];
            sz_size_t index = start_b;

            // Shift all the elements between element 1
            // element 2, right by 1.
            while (index != start_a) {
                sequence->order[index] = sequence->order[index - 1];
                index--;
            }
            sequence->order[start_a] = value;

            // Update all the pointers
            start_a++;
            partition++;
            start_b++;
        }
    }
}

inline static void _sz_sort_recursion( //
    sz_sequence_t *sequence,
    sz_size_t bit_idx,
    sz_size_t bit_max,
    sz_qsort_comparison_func_t qsort_comparator) {

    if (!sequence->count) return;

    // Partition a range of integers according to a specific bit value
    sz_size_t split = 0;
    {
        sz_size_t mask = (1ul << 63) >> bit_idx;
        while (split != sequence->count && !(sequence->order[split] & mask)) ++split;
        for (sz_size_t i = split + 1; i < sequence->count; ++i)
            if (!(sequence->order[i] & mask)) sz_swap(sequence->order + i, sequence->order + split), ++split;
    }

    // Go down recursively
    if (bit_idx < bit_max) {
        sz_sequence_t a = *sequence;
        a.count = split;
        _sz_sort_recursion(&a, bit_idx + 1, bit_max, qsort_comparator);

        sz_sequence_t b = *sequence;
        b.order += split;
        b.count -= split;
        _sz_sort_recursion(&b, bit_idx + 1, bit_max, qsort_comparator);
    }
    // Reached the end of recursion
    else {
        // Discard the prefixes
        for (sz_size_t i = 0; i != sequence->count; ++i) { memset((char *)(&sequence->order[i]) + 4, 0, 4ul); }

        // Perform sorts on smaller chunks instead of the whole handle
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__) || defined(__NT__)
        // https://stackoverflow.com/a/39561369
        // https://learn.microsoft.com/en-us/cpp/c-runtime-library/reference/qsort-s?view=msvc-170
        qsort_s(sequence->order, split, sizeof(sz_size_t), qsort_comparator, (void *)sequence);
        qsort_s(sequence->order + split,
                sequence->count - split,
                sizeof(sz_size_t),
                qsort_comparator,
                (void *)sequence);
#elif __APPLE__
        qsort_r(sequence->order, split, sizeof(sz_size_t), (void *)sequence, qsort_comparator);
        qsort_r(sequence->order + split,
                sequence->count - split,
                sizeof(sz_size_t),
                (void *)sequence,
                qsort_comparator);
#else
        // https://linux.die.net/man/3/qsort_r
        qsort_r(sequence->order, split, sizeof(sz_size_t), qsort_comparator, (void *)sequence);
        qsort_r(sequence->order + split,
                sequence->count - split,
                sizeof(sz_size_t),
                qsort_comparator,
                (void *)sequence);
#endif
    }
}

inline static int _sz_sort_sequence_strncmp(
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__) || defined(__NT__) || __APPLE__
    void *sequence_raw, void const *a_raw, void const *b_raw
#else
    void const *a_raw, void const *b_raw, void *sequence_raw
#endif
) {
    // https://man.freebsd.org/cgi/man.cgi?query=qsort_s&sektion=3&n=1
    // https://www.man7.org/linux/man-pages/man3/strcmp.3.html
    sz_sequence_t *sequence = (sz_sequence_t *)sequence_raw;
    sz_size_t a = *(sz_size_t *)a_raw;
    sz_size_t b = *(sz_size_t *)b_raw;
    sz_size_t a_len = sequence->get_length(sequence->handle, a);
    sz_size_t b_len = sequence->get_length(sequence->handle, b);
    int res = strncmp( //
        sequence->get_start(sequence->handle, a),
        sequence->get_start(sequence->handle, b),
        a_len > b_len ? b_len : a_len);
    return res ? res : a_len - b_len;
}

inline static int _sz_sort_sequence_strncasecmp(
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__) || defined(__NT__) || __APPLE__
    void *sequence_raw, void const *a_raw, void const *b_raw
#else
    void const *a_raw, void const *b_raw, void *sequence_raw
#endif
) {
    // https://man.freebsd.org/cgi/man.cgi?query=qsort_s&sektion=3&n=1
    // https://www.man7.org/linux/man-pages/man3/strcmp.3.html
    sz_sequence_t *sequence = (sz_sequence_t *)sequence_raw;
    sz_size_t a = *(sz_size_t *)a_raw;
    sz_size_t b = *(sz_size_t *)b_raw;
    sz_size_t a_len = sequence->get_length(sequence->handle, a);
    sz_size_t b_len = sequence->get_length(sequence->handle, b);
    int res = strncasecmp( //
        sequence->get_start(sequence->handle, a),
        sequence->get_start(sequence->handle, b),
        a_len > b_len ? b_len : a_len);
    return res ? res : a_len - b_len;
}

typedef struct sz_sort_config_t {
    int case_insensitive;
} sz_sort_config_t;

/**
 *  @brief  Sorting algorithm, combining Radix Sort for the first 32 bits of every word
 *          and a follow-up Quick Sort on resulting structure.
 */
inline static void sz_sort(sz_sequence_t *sequence, sz_sort_config_t const *config) {

    int case_insensitive = config && config->case_insensitive;

    // Export up to 4 bytes into the `sequence` bits themselves
    for (sz_size_t i = 0; i != sequence->count; ++i) {
        char const *begin = sequence->get_start(sequence->handle, sequence->order[i]);
        sz_size_t length = sequence->get_length(sequence->handle, sequence->order[i]);
        length = length > 4ul ? 4ul : length;
        char *prefix = (char *)&sequence->order[i];
        for (sz_size_t j = 0; j != length; ++j) prefix[7 - j] = begin[j];
        if (case_insensitive) {
            prefix[0] = sz_tolower_ascii(prefix[0]);
            prefix[1] = sz_tolower_ascii(prefix[1]);
            prefix[2] = sz_tolower_ascii(prefix[2]);
            prefix[3] = sz_tolower_ascii(prefix[3]);
        }
    }

    sz_qsort_comparison_func_t comparator = _sz_sort_sequence_strncmp;
    if (case_insensitive) comparator = _sz_sort_sequence_strncasecmp;

    // Perform optionally-parallel radix sort on them
    _sz_sort_recursion(sequence, 0, 32, comparator);
}

typedef unsigned char levenstein_distance_t;

/**
 *  @return Amount of temporary memory (in bytes) needed to efficiently compute
 *          the Levenstein distance between two strings of given size.
 */
inline static sz_size_t sz_levenstein_memory_needed(sz_size_t _, sz_size_t b_length) { return b_length + b_length + 2; }

/**
 *  @brief  Auxiliary function, that computes the minimum of three values.
 */
inline static levenstein_distance_t _sz_levenstein_minimum( //
    levenstein_distance_t a,
    levenstein_distance_t b,
    levenstein_distance_t c) {

    return (a < b ? (a < c ? a : c) : (b < c ? b : c));
}

/**
 *  @brief  Levenshtein String Similarity function, implemented with linear memory consumption.
 *          It accepts an upper bound on the possible error. Quadratic complexity in time, linear in space.
 */
inline static levenstein_distance_t sz_levenstein( //
    char const *a,
    sz_size_t a_length,
    char const *b,
    sz_size_t b_length,
    levenstein_distance_t bound,
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

    levenstein_distance_t *previous_distances = (levenstein_distance_t *)buffer;
    levenstein_distance_t *current_distances = previous_distances + b_length + 1;

    for (sz_size_t idx_b = 0; idx_b != (b_length + 1); ++idx_b) previous_distances[idx_b] = idx_b;

    for (sz_size_t idx_a = 0; idx_a != a_length; ++idx_a) {
        current_distances[0] = idx_a + 1;

        // Initialize min_distance with a value greater than bound
        levenstein_distance_t min_distance = bound;

        for (sz_size_t idx_b = 0; idx_b != b_length; ++idx_b) {
            levenstein_distance_t cost_deletion = previous_distances[idx_b + 1] + 1;
            levenstein_distance_t cost_insertion = current_distances[idx_b] + 1;
            levenstein_distance_t cost_substitution = previous_distances[idx_b] + (a[idx_a] != b[idx_b]);
            current_distances[idx_b + 1] = _sz_levenstein_minimum(cost_deletion, cost_insertion, cost_substitution);

            // Keep track of the minimum distance seen so far in this row
            if (current_distances[idx_b + 1] < min_distance) { min_distance = current_distances[idx_b + 1]; }
        }

        // If the minimum distance in this row exceeded the bound, return early
        if (min_distance > bound) return bound;

        // Swap previous_distances and current_distances pointers
        levenstein_distance_t *temp = previous_distances;
        previous_distances = current_distances;
        current_distances = temp;
    }

    return previous_distances[b_length] <= bound ? previous_distances[b_length] : bound;
}

/**
 *  @brief  Hashes provided string using hardware-accelerated CRC32 instructions.
 */
inline static sz_u32_t sz_hash_crc32_native(char const *start, sz_size_t length) { return 0; }

inline static sz_u32_t sz_hash_crc32_neon(char const *start, sz_size_t length) { return 0; }

inline static sz_u32_t sz_hash_crc32_sse(char const *start, sz_size_t length) { return 0; }

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
