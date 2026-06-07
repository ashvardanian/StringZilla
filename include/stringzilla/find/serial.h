/**
 *  @brief Serial backend for substring & byte-set search.
 *  @file include/stringzilla/find/serial.h
 *  @author Ash Vardanian
 *  @sa include/stringzilla/find.h
 */
#ifndef STRINGZILLA_FIND_SERIAL_H_
#define STRINGZILLA_FIND_SERIAL_H_

#include "stringzilla/types.h"
#include "stringzilla/compare.h" // `sz_equal`

#ifdef __cplusplus
extern "C" {
#endif

/**
 *  @brief Chooses the offsets of the most interesting characters in a search needle.
 *
 *  Search throughput can significantly deteriorate if we are matching the wrong characters.
 *  Say the needle is "aXaYa", and we are comparing the first, second, and last character.
 *  If we use SIMD and compare many offsets at a time, comparing against "a" in every register is a waste.
 *
 *  Similarly, dealing with UTF-8 inputs, we know that the lower bits of each character code carry more information.
 *  Cyrillic alphabet, for example, falls into [0x0410, 0x042F] code range for uppercase [А, Я], and
 *  into [0x0430, 0x044F] for lowercase [а, я]. Scanning through a text written in Russian, half of the
 *  bytes will carry absolutely no value and will be equal to 0x04.
 */
SZ_INTERNAL void sz_locate_needle_anomalies_( //
    sz_cptr_t start, sz_size_t length,        //
    sz_size_t *first, sz_size_t *second, sz_size_t *third) {

    *first = 0;
    *second = length / 2;
    *third = length - 1;

    //
    int has_duplicates =                   //
        start[*first] == start[*second] || //
        start[*first] == start[*third] ||  //
        start[*second] == start[*third];

    // Loop through letters to find non-colliding variants.
    if (length > 3 && has_duplicates) {
        // Pivot the middle point right, until we find a character different from the first one.
        while (start[*second] == start[*first] && *second + 1 < *third) ++(*second);
        // Pivot the third (last) point left, until we find a different character.
        while ((start[*third] == start[*second] || start[*third] == start[*first]) && *third > (*second + 1))
            --(*third);
    }

    // TODO: Investigate alternative strategies for long needles.
    // On very long needles we have the luxury to choose!
    // Often dealing with UTF-8, we will likely benefit from shifting the first and second characters
    // further to the right, to achieve not only uniqueness within the needle, but also avoid common
    // rune prefixes of 2-, 3-, and 4-byte codes.
    if (length > 8) {
        // Pivot the first and second points right, until we find a character, that:
        // > is different from others.
        // > doesn't start with 0b'110x'xxxx - only 5 bits of relevant info.
        // > doesn't start with 0b'1110'xxxx - only 4 bits of relevant info.
        // > doesn't start with 0b'1111'0xxx - only 3 bits of relevant info.
        //
        // So we are practically searching for byte values that start with 0b0xxx'xxxx or 0b'10xx'xxxx.
        // Meaning they fall in the range [0, 127] and [128, 191], in other words any unsigned int up to 191.
        sz_u8_t const *start_u8 = (sz_u8_t const *)start;
        sz_size_t vibrant_first = *first, vibrant_second = *second, vibrant_third = *third;

        // Let's begin with the second character, as the termination criteria there is more obvious
        // and we may end up with more variants to check for the first candidate.
        while ((start_u8[vibrant_second] > 191 || start_u8[vibrant_second] == start_u8[vibrant_third]) &&
               (vibrant_second + 1 < vibrant_third))
            ++vibrant_second;

        // Now check if we've indeed found a good candidate or should revert the `vibrant_second` to `second`.
        if (start_u8[vibrant_second] < 191) { *second = vibrant_second; }
        else { vibrant_second = *second; }

        // Now check the first character.
        while ((start_u8[vibrant_first] > 191 || start_u8[vibrant_first] == start_u8[vibrant_second] ||
                start_u8[vibrant_first] == start_u8[vibrant_third]) &&
               (vibrant_first + 1 < vibrant_second))
            ++vibrant_first;

        // Now check if we've indeed found a good candidate or should revert the `vibrant_first` to `first`.
        // We don't need to shift the third one when dealing with texts as the last byte of the text is
        // also the last byte of a rune and contains the most information.
        if (start_u8[vibrant_first] < 191) { *first = vibrant_first; }
    }
}

SZ_PUBLIC sz_cptr_t sz_find_byteset_serial(sz_cptr_t text, sz_size_t length, sz_byteset_t const *set) {
    for (sz_cptr_t const end = text + length; text != end; ++text)
        if (sz_byteset_contains(set, *text)) return text;
    return SZ_NULL_CHAR;
}

SZ_PUBLIC sz_cptr_t sz_rfind_byteset_serial(sz_cptr_t text, sz_size_t length, sz_byteset_t const *set) {
#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Warray-bounds"
#endif
    sz_cptr_t const end = text;
    for (text += length; text != end;)
        if (sz_byteset_contains(set, *(text -= 1))) return text;
    return SZ_NULL_CHAR;
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif
}

/*  Find the first occurrence of a @b single-character needle in an arbitrary length haystack.
 *  This implementation uses hardware-agnostic SWAR technique, to process 8 characters at a time.
 *  Identical to `memchr(haystack, needle[0], haystack_length)`.
 */
SZ_PUBLIC sz_cptr_t sz_find_byte_serial(sz_cptr_t h_chars, sz_size_t h_length, sz_cptr_t n_chars) {

    if (!h_length) return SZ_NULL_CHAR;
    // Reinterpret as unsigned bytes so the SWAR broadcast below cannot sign-extend
    // on platforms where `char` is signed (e.g. `-fsigned-char`). See issue #306.
    sz_u8_t const *h = (sz_u8_t const *)h_chars;
    sz_u8_t const *const n = (sz_u8_t const *)n_chars;
    sz_u8_t const *const h_end = h + h_length;

#if !SZ_IS_BIG_ENDIAN_       // Use SWAR only on little-endian platforms for brevity.
#if !SZ_USE_MISALIGNED_LOADS // Process the misaligned head, to void UB on unaligned 64-bit loads.
    for (; ((sz_size_t)h & 7ull) && h < h_end; ++h)
        if (*h == *n) return (sz_cptr_t)h;
#endif

    // Broadcast the n into every byte of a 64-bit integer to use SWAR
    // techniques and process eight characters at a time.
    sz_u64_vec_t h_vec, n_vec, match_vec;
    match_vec.u64 = 0;
    n_vec.u64 = (sz_u64_t)*n * 0x0101010101010101ull;
    for (; h + 8 <= h_end; h += 8) {
        h_vec.u64 = *(sz_u64_t const *)h;
        match_vec = sz_u64_each_byte_equal_(h_vec, n_vec);
        if (match_vec.u64) return (sz_cptr_t)(h + sz_u64_ctz(match_vec.u64) / 8);
    }
#endif

    // Handle the misaligned tail.
    for (; h < h_end; ++h)
        if (*h == *n) return (sz_cptr_t)h;
    return SZ_NULL_CHAR;
}

/*  Find the last occurrence of a @b single-character needle in an arbitrary length haystack.
 *  This implementation uses hardware-agnostic SWAR technique, to process 8 characters at a time.
 *  Identical to `memrchr(haystack, needle[0], haystack_length)`.
 */
sz_cptr_t sz_rfind_byte_serial(sz_cptr_t h_chars, sz_size_t h_length, sz_cptr_t n_chars) {

    if (!h_length) return SZ_NULL_CHAR;
    // Reinterpret as unsigned bytes so the SWAR broadcast below cannot sign-extend
    // on platforms where `char` is signed (e.g. `-fsigned-char`). See issue #306.
    sz_u8_t const *const h_start = (sz_u8_t const *)h_chars;
    sz_u8_t const *const n = (sz_u8_t const *)n_chars;

    // Reposition the `h` pointer to the end, as we will be walking backwards.
    sz_u8_t const *h = h_start + h_length - 1;

#if !SZ_IS_BIG_ENDIAN_       // Use SWAR only on little-endian platforms for brevity.
#if !SZ_USE_MISALIGNED_LOADS // Process the misaligned head, to void UB on unaligned 64-bit loads.
    for (; ((sz_size_t)(h + 1) & 7ull) && h >= h_start; --h)
        if (*h == *n) return (sz_cptr_t)h;
#endif

    // Broadcast the n into every byte of a 64-bit integer to use SWAR
    // techniques and process eight characters at a time.
    sz_u64_vec_t h_vec, n_vec, match_vec;
    n_vec.u64 = (sz_u64_t)*n * 0x0101010101010101ull;
    for (; h >= h_start + 7; h -= 8) {
        h_vec.u64 = *(sz_u64_t const *)(h - 7);
        match_vec = sz_u64_each_byte_equal_(h_vec, n_vec);
        if (match_vec.u64) return (sz_cptr_t)(h - sz_u64_clz(match_vec.u64) / 8);
    }
#endif

    for (; h >= h_start; --h)
        if (*h == *n) return (sz_cptr_t)h;
    return SZ_NULL_CHAR;
}

/**
 *  @brief 2Byte-level equality comparison between two 64-bit integers.
 *  @return 64-bit integer, where every top bit in each 2byte signifies a match.
 */
SZ_INTERNAL sz_u64_vec_t sz_u64_each_2byte_equal_(sz_u64_vec_t a, sz_u64_vec_t b) {
    sz_u64_vec_t vec;
    vec.u64 = ~(a.u64 ^ b.u64);
    // The match is valid, if every bit within each 2byte is set.
    // For that take the bottom 15 bits of each 2byte, add one to them,
    // and if this sets the top bit to one, then all the 15 bits are ones as well.
    vec.u64 = ((vec.u64 & 0x7FFF7FFF7FFF7FFFull) + 0x0001000100010001ull) & ((vec.u64 & 0x8000800080008000ull));
    return vec;
}

SZ_INTERNAL sz_cptr_t sz_find_1byte_serial_(sz_cptr_t h, sz_size_t h_length, sz_cptr_t n, sz_size_t n_length) {
    sz_unused_(n_length); //? We keep this argument only for `sz_find_t` signature compatibility.
    return sz_find_byte_serial(h, h_length, n);
}

SZ_INTERNAL sz_cptr_t sz_rfind_1byte_serial_(sz_cptr_t h, sz_size_t h_length, sz_cptr_t n, sz_size_t n_length) {
    sz_unused_(n_length); //? We keep this argument only for `sz_rfind_t` signature compatibility.
    return sz_rfind_byte_serial(h, h_length, n);
}

/**
 *  @brief Find the first occurrence of a @b two-character needle in an arbitrary length haystack.
 *         This implementation uses hardware-agnostic SWAR technique, to process 8 possible offsets at a time.
 */
SZ_INTERNAL sz_cptr_t sz_find_2byte_serial_(sz_cptr_t h, sz_size_t h_length, sz_cptr_t n, sz_size_t n_length) {

    // This is an internal method, and the haystack is guaranteed to be at least 2 bytes long.
    sz_assert_(h_length >= 2 && "The haystack is too short.");
    sz_unused_(n_length); //? We keep this argument only for `sz_find_t` signature compatibility.
    sz_cptr_t const h_end = h + h_length;

    // On big-endian systems, skip SWAR and use simple serial search
#if SZ_IS_BIG_ENDIAN_
    for (; h + 2 <= h_end; ++h)
        if ((h[0] == n[0]) + (h[1] == n[1]) == 2) return h;
    return SZ_NULL_CHAR;
#endif

    // Process the misaligned head, to void UB on unaligned 64-bit loads.
#if !SZ_USE_MISALIGNED_LOADS
    for (; ((sz_size_t)h & 7ull) && h + 2 <= h_end; ++h)
        if ((h[0] == n[0]) + (h[1] == n[1]) == 2) return h;
#endif

    sz_u64_vec_t h_even_vec, h_odd_vec, n_vec, matches_even_vec, matches_odd_vec;
    n_vec.u64 = 0;
    n_vec.u8s[0] = n[0], n_vec.u8s[1] = n[1];
    n_vec.u64 *= 0x0001000100010001ull; // broadcast

    // This code simulates hyper-scalar execution, analyzing 8 offsets at a time.
    for (; h + 9 <= h_end; h += 8) {
        h_even_vec.u64 = *(sz_u64_t *)h;
        h_odd_vec.u64 = (h_even_vec.u64 >> 8) | ((sz_u64_t)h[8] << 56);
        matches_even_vec = sz_u64_each_2byte_equal_(h_even_vec, n_vec);
        matches_odd_vec = sz_u64_each_2byte_equal_(h_odd_vec, n_vec);
        matches_even_vec.u64 >>= 8;
        if (matches_even_vec.u64 + matches_odd_vec.u64) {
            sz_u64_t match_indicators = matches_even_vec.u64 | matches_odd_vec.u64;
            return h + sz_u64_ctz(match_indicators) / 8;
        }
    }

    for (; h + 2 <= h_end; ++h)
        if ((h[0] == n[0]) + (h[1] == n[1]) == 2) return h;
    return SZ_NULL_CHAR;
}

/**
 *  @brief 4Byte-level equality comparison between two 64-bit integers.
 *  @return 64-bit integer, where every top bit in each 4byte signifies a match.
 */
SZ_INTERNAL sz_u64_vec_t sz_u64_each_4byte_equal_(sz_u64_vec_t a, sz_u64_vec_t b) {
    sz_u64_vec_t vec;
    vec.u64 = ~(a.u64 ^ b.u64);
    // The match is valid, if every bit within each 4byte is set.
    // For that take the bottom 31 bits of each 4byte, add one to them,
    // and if this sets the top bit to one, then all the 31 bits are ones as well.
    vec.u64 = ((vec.u64 & 0x7FFFFFFF7FFFFFFFull) + 0x0000000100000001ull) & ((vec.u64 & 0x8000000080000000ull));
    return vec;
}

/**
 *  @brief Find the first occurrence of a @b four-character needle in an arbitrary length haystack.
 *         This implementation uses hardware-agnostic SWAR technique, to process 8 possible offsets at a time.
 */
SZ_INTERNAL sz_cptr_t sz_find_4byte_serial_(sz_cptr_t h, sz_size_t h_length, sz_cptr_t n, sz_size_t n_length) {

    // This is an internal method, and the haystack is guaranteed to be at least 4 bytes long.
    sz_assert_(h_length >= 4 && "The haystack is too short.");
    sz_unused_(n_length); //? We keep this argument only for `sz_find_t` signature compatibility.
    sz_cptr_t const h_end = h + h_length;

    // On big-endian systems, skip SWAR and use simple serial search
#if SZ_IS_BIG_ENDIAN_
    for (; h + 4 <= h_end; ++h)
        if ((h[0] == n[0]) + (h[1] == n[1]) + (h[2] == n[2]) + (h[3] == n[3]) == 4) return h;
    return SZ_NULL_CHAR;
#endif

    // Process the misaligned head, to void UB on unaligned 64-bit loads.
#if !SZ_USE_MISALIGNED_LOADS
    for (; ((sz_size_t)h & 7ull) && h + 4 <= h_end; ++h)
        if ((h[0] == n[0]) + (h[1] == n[1]) + (h[2] == n[2]) + (h[3] == n[3]) == 4) return h;
#endif

    sz_u64_vec_t h0_vec, h1_vec, h2_vec, h3_vec, n_vec, matches0_vec, matches1_vec, matches2_vec, matches3_vec;
    n_vec.u64 = 0;
    n_vec.u8s[0] = n[0], n_vec.u8s[1] = n[1], n_vec.u8s[2] = n[2], n_vec.u8s[3] = n[3];
    n_vec.u64 *= 0x0000000100000001ull; // broadcast

    // This code simulates hyper-scalar execution, analyzing 8 offsets at a time using four 64-bit words.
    // We load the subsequent four-byte word as well, taking its first bytes. Think of it as a glorified prefetch :)
    sz_u64_t h_page_current, h_page_next;
    for (; h + sizeof(sz_u64_t) + sizeof(sz_u32_t) <= h_end; h += sizeof(sz_u64_t)) {
        h_page_current = *(sz_u64_t *)h;
        h_page_next = *(sz_u32_t *)(h + 8);
        h0_vec.u64 = (h_page_current);
        h1_vec.u64 = (h_page_current >> 8) | (h_page_next << 56);
        h2_vec.u64 = (h_page_current >> 16) | (h_page_next << 48);
        h3_vec.u64 = (h_page_current >> 24) | (h_page_next << 40);
        matches0_vec = sz_u64_each_4byte_equal_(h0_vec, n_vec);
        matches1_vec = sz_u64_each_4byte_equal_(h1_vec, n_vec);
        matches2_vec = sz_u64_each_4byte_equal_(h2_vec, n_vec);
        matches3_vec = sz_u64_each_4byte_equal_(h3_vec, n_vec);

        if (matches0_vec.u64 | matches1_vec.u64 | matches2_vec.u64 | matches3_vec.u64) {
            matches0_vec.u64 >>= 24;
            matches1_vec.u64 >>= 16;
            matches2_vec.u64 >>= 8;
            sz_u64_t match_indicators = matches0_vec.u64 | matches1_vec.u64 | matches2_vec.u64 | matches3_vec.u64;
            return h + sz_u64_ctz(match_indicators) / 8;
        }
    }

    for (; h + 4 <= h_end; ++h)
        if ((h[0] == n[0]) + (h[1] == n[1]) + (h[2] == n[2]) + (h[3] == n[3]) == 4) return h;
    return SZ_NULL_CHAR;
}

/**
 *  @brief 3Byte-level equality comparison between two 64-bit integers.
 *  @return 64-bit integer, where every top bit in each 3byte signifies a match.
 */
SZ_INTERNAL sz_u64_vec_t sz_u64_each_3byte_equal_(sz_u64_vec_t a, sz_u64_vec_t b) {
    sz_u64_vec_t vec;
    vec.u64 = ~(a.u64 ^ b.u64);
    // The match is valid, if every bit within each 4byte is set.
    // For that take the bottom 31 bits of each 4byte, add one to them,
    // and if this sets the top bit to one, then all the 31 bits are ones as well.
    vec.u64 = ((vec.u64 & 0xFFFF7FFFFF7FFFFFull) + 0x0000000001000001ull) & ((vec.u64 & 0x0000800000800000ull));
    return vec;
}

/**
 *  @brief Find the first occurrence of a @b three-character needle in an arbitrary length haystack.
 *         This implementation uses hardware-agnostic SWAR technique, to process 8 possible offsets at a time.
 */
SZ_INTERNAL sz_cptr_t sz_find_3byte_serial_(sz_cptr_t h, sz_size_t h_length, sz_cptr_t n, sz_size_t n_length) {

    // This is an internal method, and the haystack is guaranteed to be at least 4 bytes long.
    sz_assert_(h_length >= 3 && "The haystack is too short.");
    sz_unused_(n_length); //? We keep this argument only for `sz_find_t` signature compatibility.
    sz_cptr_t const h_end = h + h_length;

    // On big-endian systems, skip SWAR and use simple serial search
#if SZ_IS_BIG_ENDIAN_
    for (; h + 3 <= h_end; ++h)
        if ((h[0] == n[0]) + (h[1] == n[1]) + (h[2] == n[2]) == 3) return h;
    return SZ_NULL_CHAR;
#endif

    // Process the misaligned head, to void UB on unaligned 64-bit loads.
#if !SZ_USE_MISALIGNED_LOADS
    for (; ((sz_size_t)h & 7ull) && h + 3 <= h_end; ++h)
        if ((h[0] == n[0]) + (h[1] == n[1]) + (h[2] == n[2]) == 3) return h;
#endif

    // We fetch 12
    sz_u64_vec_t h0_vec, h1_vec, h2_vec, h3_vec, h4_vec;
    sz_u64_vec_t matches0_vec, matches1_vec, matches2_vec, matches3_vec, matches4_vec;
    sz_u64_vec_t n_vec;
    n_vec.u64 = 0;
    n_vec.u8s[0] = n[0], n_vec.u8s[1] = n[1], n_vec.u8s[2] = n[2];
    n_vec.u64 *= 0x0000000001000001ull; // broadcast

    // This code simulates hyper-scalar execution, analyzing 8 offsets at a time using three 64-bit words.
    // We load the subsequent two-byte word as well.
    sz_u64_t h_page_current, h_page_next;
    for (; h + sizeof(sz_u64_t) + sizeof(sz_u16_t) <= h_end; h += sizeof(sz_u64_t)) {
        h_page_current = *(sz_u64_t *)h;
        h_page_next = *(sz_u16_t *)(h + 8);
        h0_vec.u64 = (h_page_current);
        h1_vec.u64 = (h_page_current >> 8) | (h_page_next << 56);
        h2_vec.u64 = (h_page_current >> 16) | (h_page_next << 48);
        h3_vec.u64 = (h_page_current >> 24) | (h_page_next << 40);
        h4_vec.u64 = (h_page_current >> 32) | (h_page_next << 32);
        matches0_vec = sz_u64_each_3byte_equal_(h0_vec, n_vec);
        matches1_vec = sz_u64_each_3byte_equal_(h1_vec, n_vec);
        matches2_vec = sz_u64_each_3byte_equal_(h2_vec, n_vec);
        matches3_vec = sz_u64_each_3byte_equal_(h3_vec, n_vec);
        matches4_vec = sz_u64_each_3byte_equal_(h4_vec, n_vec);

        if (matches0_vec.u64 | matches1_vec.u64 | matches2_vec.u64 | matches3_vec.u64 | matches4_vec.u64) {
            matches0_vec.u64 >>= 16;
            matches1_vec.u64 >>= 8;
            matches3_vec.u64 <<= 8;
            matches4_vec.u64 <<= 16;
            sz_u64_t match_indicators =
                matches0_vec.u64 | matches1_vec.u64 | matches2_vec.u64 | matches3_vec.u64 | matches4_vec.u64;
            return h + sz_u64_ctz(match_indicators) / 8;
        }
    }

    for (; h + 3 <= h_end; ++h)
        if ((h[0] == n[0]) + (h[1] == n[1]) + (h[2] == n[2]) == 3) return h;
    return SZ_NULL_CHAR;
}

/**
 *  @brief Boyer-Moore-Horspool algorithm for exact matching of patterns up to @b 256-bytes long.
 *         Uses the Raita heuristic to match the first two, the last, and the middle character of the pattern.
 */
SZ_INTERNAL sz_cptr_t sz_find_horspool_upto_256bytes_serial_( //
    sz_cptr_t h_chars, sz_size_t h_length,                    //
    sz_cptr_t n_chars, sz_size_t n_length) {
    sz_assert_(n_length <= 256 && "The pattern is too long.");
    // Several popular string matching algorithms are using a bad-character shift table.
    // Boyer Moore: https://www-igm.univ-mlv.fr/~lecroq/string/node14.html
    // Quick Search: https://www-igm.univ-mlv.fr/~lecroq/string/node19.html
    // Smith: https://www-igm.univ-mlv.fr/~lecroq/string/node21.html
    union {
        sz_u8_t jumps[256];
        sz_u64_vec_t vecs[64];
    } bad_shift_table;

    // Let's initialize the table using SWAR to the total length of the string.
    sz_u8_t const *h = (sz_u8_t const *)h_chars;
    sz_u8_t const *n = (sz_u8_t const *)n_chars;
    {
        sz_u64_vec_t n_length_vec;
        n_length_vec.u64 = ((sz_u8_t)(n_length - 1)) * 0x0101010101010101ull; // broadcast
        for (sz_size_t i = 0; i != 64; ++i) bad_shift_table.vecs[i].u64 = n_length_vec.u64;
        for (sz_size_t i = 0; i + 1 < n_length; ++i) bad_shift_table.jumps[n[i]] = (sz_u8_t)(n_length - i - 1);
    }

    // Another common heuristic is to match a few characters from different parts of a string.
    // Raita suggests to use the first two, the last, and the middle character of the pattern.
    sz_u32_vec_t h_vec, n_vec;

    // Pick the parts of the needle that are worth comparing.
    sz_size_t offset_first, offset_mid, offset_last;
    sz_locate_needle_anomalies_(n_chars, n_length, &offset_first, &offset_mid, &offset_last);

    // Broadcast those characters into an unsigned integer.
    n_vec.u8s[0] = n[offset_first];
    n_vec.u8s[1] = n[offset_first + 1];
    n_vec.u8s[2] = n[offset_mid];
    n_vec.u8s[3] = n[offset_last];

    // Scan through the whole haystack, skipping the last `n_length - 1` bytes.
    for (sz_size_t i = 0; i <= h_length - n_length;) {
        h_vec.u8s[0] = h[i + offset_first];
        h_vec.u8s[1] = h[i + offset_first + 1];
        h_vec.u8s[2] = h[i + offset_mid];
        h_vec.u8s[3] = h[i + offset_last];
        if (h_vec.u32 == n_vec.u32 && sz_equal_serial((sz_cptr_t)h + i, n_chars, n_length)) return (sz_cptr_t)h + i;
        i += bad_shift_table.jumps[h[i + n_length - 1]];
    }
    return SZ_NULL_CHAR;
}

/**
 *  @brief Boyer-Moore-Horspool algorithm for @b reverse-order exact matching of patterns up to @b 256-bytes long.
 *         Uses the Raita heuristic to match the first two, the last, and the middle character of the pattern.
 */
SZ_INTERNAL sz_cptr_t sz_rfind_horspool_upto_256bytes_serial_( //
    sz_cptr_t h_chars, sz_size_t h_length,                     //
    sz_cptr_t n_chars, sz_size_t n_length) {
    sz_assert_(n_length <= 256 && "The pattern is too long.");
    union {
        sz_u8_t jumps[256];
        sz_u64_vec_t vecs[64];
    } bad_shift_table;

    // Let's initialize the table using SWAR to the total length of the string.
    sz_u8_t const *h = (sz_u8_t const *)h_chars;
    sz_u8_t const *n = (sz_u8_t const *)n_chars;
    {
        sz_u64_vec_t n_length_vec;
        n_length_vec.u64 = ((sz_u8_t)(n_length - 1)) * 0x0101010101010101ull; // broadcast
        for (sz_size_t i = 0; i != 64; ++i) bad_shift_table.vecs[i].u64 = n_length_vec.u64;
        for (sz_size_t i = 0; i + 1 < n_length; ++i)
            bad_shift_table.jumps[n[n_length - i - 1]] = (sz_u8_t)(n_length - i - 1);
    }

    // Another common heuristic is to match a few characters from different parts of a string.
    // Raita suggests to use the first two, the last, and the middle character of the pattern.
    sz_u32_vec_t h_vec, n_vec;

    // Pick the parts of the needle that are worth comparing.
    sz_size_t offset_first, offset_mid, offset_last;
    sz_locate_needle_anomalies_(n_chars, n_length, &offset_first, &offset_mid, &offset_last);

    // Broadcast those characters into an unsigned integer.
    n_vec.u8s[0] = n[offset_first];
    n_vec.u8s[1] = n[offset_first + 1];
    n_vec.u8s[2] = n[offset_mid];
    n_vec.u8s[3] = n[offset_last];

    // Scan through the whole haystack, skipping the first `n_length - 1` bytes.
    for (sz_size_t j = 0; j <= h_length - n_length;) {
        sz_size_t i = h_length - n_length - j;
        h_vec.u8s[0] = h[i + offset_first];
        h_vec.u8s[1] = h[i + offset_first + 1];
        h_vec.u8s[2] = h[i + offset_mid];
        h_vec.u8s[3] = h[i + offset_last];
        if (h_vec.u32 == n_vec.u32 && sz_equal_serial((sz_cptr_t)h + i, n_chars, n_length)) return (sz_cptr_t)h + i;
        j += bad_shift_table.jumps[h[i]];
    }
    return SZ_NULL_CHAR;
}

/**
 *  @brief Exact substring search helper function, that finds the first occurrence of a prefix of the needle
 *         using a given search function, and then verifies the remaining part of the needle.
 */
SZ_INTERNAL sz_cptr_t sz_find_with_prefix_( //
    sz_cptr_t h, sz_size_t h_length, sz_cptr_t n, sz_size_t n_length, sz_find_t find_prefix, sz_size_t prefix_length) {

    sz_size_t suffix_length = n_length - prefix_length;
    while (1) {
        sz_cptr_t found = find_prefix(h, h_length, n, prefix_length);
        if (!found) return SZ_NULL_CHAR;

        // Verify the remaining part of the needle
        sz_size_t remaining = h_length - (found - h);
        if (remaining < n_length) return SZ_NULL_CHAR;
        if (sz_equal_serial(found + prefix_length, n + prefix_length, suffix_length)) return found;

        // Adjust the position.
        h = found + 1;
        h_length = remaining - 1;
    }

    // Unreachable, but helps silence compiler warnings:
    return SZ_NULL_CHAR;
}

/**
 *  @brief Exact reverse-order substring search helper function, that finds the last occurrence of a suffix of the
 *         needle using a given search function, and then verifies the remaining part of the needle.
 */
SZ_INTERNAL sz_cptr_t sz_rfind_with_suffix_(sz_cptr_t h, sz_size_t h_length, sz_cptr_t n, sz_size_t n_length,
                                            sz_find_t find_suffix, sz_size_t suffix_length) {

    sz_size_t prefix_length = n_length - suffix_length;
    while (1) {
        sz_cptr_t found = find_suffix(h, h_length, n + prefix_length, suffix_length);
        if (!found) return SZ_NULL_CHAR;

        // Verify the remaining part of the needle
        sz_size_t remaining = found - h;
        if (remaining < prefix_length) return SZ_NULL_CHAR;
        if (sz_equal_serial(found - prefix_length, n, prefix_length)) return found - prefix_length;

        // Adjust the position.
        h_length = remaining - 1;
    }

    // Unreachable, but helps silence compiler warnings:
    return SZ_NULL_CHAR;
}

SZ_INTERNAL sz_cptr_t sz_find_over_4bytes_serial_(sz_cptr_t h, sz_size_t h_length, sz_cptr_t n, sz_size_t n_length) {
    return sz_find_with_prefix_(h, h_length, n, n_length, (sz_find_t)sz_find_4byte_serial_, 4);
}

SZ_INTERNAL sz_cptr_t sz_find_horspool_over_256bytes_serial_( //
    sz_cptr_t h, sz_size_t h_length, sz_cptr_t n, sz_size_t n_length) {
    return sz_find_with_prefix_(h, h_length, n, n_length, sz_find_horspool_upto_256bytes_serial_, 256);
}

SZ_INTERNAL sz_cptr_t sz_rfind_horspool_over_256bytes_serial_( //
    sz_cptr_t h, sz_size_t h_length, sz_cptr_t n, sz_size_t n_length) {
    return sz_rfind_with_suffix_(h, h_length, n, n_length, sz_rfind_horspool_upto_256bytes_serial_, 256);
}

SZ_PUBLIC sz_cptr_t sz_find_serial(sz_cptr_t h, sz_size_t h_length, sz_cptr_t n, sz_size_t n_length) {
    // This almost never fires, but it's better to be safe than sorry.
    if (h_length < n_length || !n_length) return SZ_NULL_CHAR;

    sz_find_t backends[] = {
        // For very short strings brute-force SWAR makes sense - now optimized for both endianness!
        sz_find_1byte_serial_,
        sz_find_2byte_serial_,
        sz_find_3byte_serial_,
        sz_find_4byte_serial_,
        // To avoid constructing the skip-table, let's use the prefixed approach.
        sz_find_over_4bytes_serial_,
        // For longer needles - use skip tables.
        sz_find_horspool_upto_256bytes_serial_,
        sz_find_horspool_over_256bytes_serial_,
    };

    return backends[
        // For very short strings brute-force SWAR makes sense.
        (n_length > 1) + (n_length > 2) + (n_length > 3) +
        // To avoid constructing the skip-table, let's use the prefixed approach.
        (n_length > 4) +
        // For longer needles - use skip tables.
        (n_length > 8) + (n_length > 256)](h, h_length, n, n_length);
}

SZ_PUBLIC sz_cptr_t sz_rfind_serial(sz_cptr_t h, sz_size_t h_length, sz_cptr_t n, sz_size_t n_length) {

    // This almost never fires, but it's better to be safe than sorry.
    if (h_length < n_length || !n_length) return SZ_NULL_CHAR;

    sz_find_t backends[] = {
        // For very short strings brute-force SWAR makes sense.
        sz_rfind_1byte_serial_,
        //  TODO: implement reverse-order SWAR for 2/3/4 byte variants.
        //  TODO: sz_rfind_2byte_serial_,
        //  TODO: sz_rfind_3byte_serial_,
        //  TODO: sz_rfind_4byte_serial_,
        // To avoid constructing the skip-table, let's use the prefixed approach.
        // sz_rfind_over_4bytes_serial_,
        // For longer needles - use skip tables.
        sz_rfind_horspool_upto_256bytes_serial_,
        sz_rfind_horspool_over_256bytes_serial_,
    };

    return backends[
        // For very short strings brute-force SWAR makes sense.
        0 +
        // To avoid constructing the skip-table, let's use the prefixed approach.
        (n_length > 1) +
        // For longer needles - use skip tables.
        (n_length > 256)](h, h_length, n, n_length);
}

#ifdef __cplusplus
}
#endif

#endif // STRINGZILLA_FIND_SERIAL_H_
