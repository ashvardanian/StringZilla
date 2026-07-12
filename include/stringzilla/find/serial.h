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
 *
 *  @param start Pointer to the needle bytes.
 *  @param length Length of the needle in bytes.
 *  @param first Output offset of the first anomalous byte.
 *  @param second Output offset of the second anomalous byte.
 *  @param third Output offset of the third anomalous byte.
 */
SZ_HELPER_AUTO void sz_locate_needle_anomalies_( //
    sz_cptr_t start, sz_size_t length,           //
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

SZ_API_COMPTIME sz_cptr_t sz_find_byteset_serial(sz_cptr_t text, sz_size_t length, sz_byteset_t const *set) {
    for (sz_cptr_t const end = text + length; text != end; ++text)
        if (sz_byteset_contains(set, *text)) return text;
    return SZ_NULL_CHAR;
}

SZ_API_COMPTIME sz_cptr_t sz_rfind_byteset_serial(sz_cptr_t text, sz_size_t length, sz_byteset_t const *set) {
    sz_cptr_t const end = text;
    for (text += length; text != end;)
        if (sz_byteset_contains(set, *(text -= 1))) return text;
    return SZ_NULL_CHAR;
}

/*  Find the first occurrence of a @b single-character needle in an arbitrary length haystack.
 *  This implementation uses hardware-agnostic SWAR technique, to process 8 characters at a time.
 *  Identical to `memchr(haystack, needle[0], haystack_length)`.
 */
SZ_API_COMPTIME sz_cptr_t sz_find_byte_serial(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle) {

    if (!haystack_length) return SZ_NULL_CHAR;
    // Reinterpret as unsigned bytes so the SWAR broadcast below cannot sign-extend
    // on platforms where `char` is signed (e.g. `-fsigned-char`). See issue #306.
    sz_u8_t const *haystack_cursor = (sz_u8_t const *)haystack;
    sz_u8_t const *const needle_u8 = (sz_u8_t const *)needle;
    sz_u8_t const *const haystack_end = haystack_cursor + haystack_length;

#if !SZ_IS_BIG_ENDIAN_       // Use SWAR only on little-endian platforms for brevity.
#if !SZ_USE_MISALIGNED_LOADS // Process the misaligned head, to void UB on unaligned 64-bit loads.
    for (; ((sz_size_t)haystack_cursor & 7ull) && haystack_cursor < haystack_end; ++haystack_cursor)
        if (*haystack_cursor == *needle_u8) return (sz_cptr_t)haystack_cursor;
#endif

    // Broadcast the needle into every byte of a 64-bit integer to use SWAR
    // techniques and process eight characters at a time.
    sz_u64_vec_t haystack_vec, needle_vec, match_vec;
    match_vec.u64 = 0;
    needle_vec.u64 = (sz_u64_t)*needle_u8 * 0x0101010101010101ull;
    for (; haystack_cursor + 8 <= haystack_end; haystack_cursor += 8) {
        haystack_vec.u64 = *(sz_u64_t const *)haystack_cursor;
        match_vec = sz_u64_each_byte_equal_(haystack_vec, needle_vec);
        if (match_vec.u64) return (sz_cptr_t)(haystack_cursor + sz_u64_ctz(match_vec.u64) / 8);
    }
#endif

    // Handle the misaligned tail.
    for (; haystack_cursor < haystack_end; ++haystack_cursor)
        if (*haystack_cursor == *needle_u8) return (sz_cptr_t)haystack_cursor;
    return SZ_NULL_CHAR;
}

/*  Find the last occurrence of a @b single-character needle in an arbitrary length haystack.
 *  This implementation uses hardware-agnostic SWAR technique, to process 8 characters at a time.
 *  Identical to `memrchr(haystack, needle[0], haystack_length)`.
 */
sz_cptr_t sz_rfind_byte_serial(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle) {

    if (!haystack_length) return SZ_NULL_CHAR;
    // Reinterpret as unsigned bytes so the SWAR broadcast below cannot sign-extend
    // on platforms where `char` is signed (e.g. `-fsigned-char`). See issue #306.
    sz_u8_t const *const haystack_start = (sz_u8_t const *)haystack;
    sz_u8_t const *const needle_u8 = (sz_u8_t const *)needle;

    // Reposition the cursor to the end, as we will be walking backwards.
    sz_u8_t const *haystack_cursor = haystack_start + haystack_length - 1;

#if !SZ_IS_BIG_ENDIAN_       // Use SWAR only on little-endian platforms for brevity.
#if !SZ_USE_MISALIGNED_LOADS // Process the misaligned head, to void UB on unaligned 64-bit loads.
    for (; ((sz_size_t)(haystack_cursor + 1) & 7ull) && haystack_cursor >= haystack_start; --haystack_cursor)
        if (*haystack_cursor == *needle_u8) return (sz_cptr_t)haystack_cursor;
#endif

    // Broadcast the needle into every byte of a 64-bit integer to use SWAR
    // techniques and process eight characters at a time.
    sz_u64_vec_t haystack_vec, needle_vec, match_vec;
    needle_vec.u64 = (sz_u64_t)*needle_u8 * 0x0101010101010101ull;
    for (; haystack_cursor >= haystack_start + 7; haystack_cursor -= 8) {
        haystack_vec.u64 = *(sz_u64_t const *)(haystack_cursor - 7);
        match_vec = sz_u64_each_byte_equal_(haystack_vec, needle_vec);
        if (match_vec.u64) return (sz_cptr_t)(haystack_cursor - sz_u64_clz(match_vec.u64) / 8);
    }
#endif

    for (; haystack_cursor >= haystack_start; --haystack_cursor)
        if (*haystack_cursor == *needle_u8) return (sz_cptr_t)haystack_cursor;
    return SZ_NULL_CHAR;
}

/**
 *  @brief 2-byte-level equality comparison between two 64-bit integers.
 *  @return 64-bit integer, where every top bit in each 2-byte group signifies a match.
 */
SZ_HELPER_INLINE sz_u64_vec_t sz_u64_each_2byte_equal_(sz_u64_vec_t a, sz_u64_vec_t b) {
    sz_u64_vec_t vec;
    vec.u64 = ~(a.u64 ^ b.u64);
    // The match is valid, if every bit within each 2-byte group is set.
    // For that take the bottom 15 bits of each 2-byte group, add one to them,
    // and if this sets the top bit to one, then all the 15 bits are ones as well.
    vec.u64 = ((vec.u64 & 0x7FFF7FFF7FFF7FFFull) + 0x0001000100010001ull) & ((vec.u64 & 0x8000800080008000ull));
    return vec;
}

SZ_HELPER_NOINLINE sz_cptr_t sz_find_1byte_serial_(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle,
                                                   sz_size_t needle_length) {
    sz_unused_(needle_length); //? We keep this argument only for `sz_find_t` signature compatibility.
    return sz_find_byte_serial(haystack, haystack_length, needle);
}

SZ_HELPER_NOINLINE sz_cptr_t sz_rfind_1byte_serial_(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle,
                                                    sz_size_t needle_length) {
    sz_unused_(needle_length); //? We keep this argument only for `sz_rfind_t` signature compatibility.
    return sz_rfind_byte_serial(haystack, haystack_length, needle);
}

/**
 *  @brief Find the first occurrence of a @b two-character needle in an arbitrary length haystack.
 *         This implementation uses hardware-agnostic SWAR technique, to process 8 possible offsets at a time.
 */
SZ_HELPER_NOINLINE sz_cptr_t sz_find_2byte_serial_(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle,
                                                   sz_size_t needle_length) {

    // This is an internal method, and the haystack is guaranteed to be at least 2 bytes long.
    sz_assert_(haystack_length >= 2 && "The haystack is too short.");
    sz_unused_(needle_length); //? We keep this argument only for `sz_find_t` signature compatibility.
    sz_cptr_t const haystack_end = haystack + haystack_length;

    // On big-endian systems, skip SWAR and use simple serial search
#if SZ_IS_BIG_ENDIAN_
    for (; haystack + 2 <= haystack_end; ++haystack)
        if ((haystack[0] == needle[0]) + (haystack[1] == needle[1]) == 2) return haystack;
    return SZ_NULL_CHAR;
#endif

    // Process the misaligned head, to void UB on unaligned 64-bit loads.
#if !SZ_USE_MISALIGNED_LOADS
    for (; ((sz_size_t)haystack & 7ull) && haystack + 2 <= haystack_end; ++haystack)
        if ((haystack[0] == needle[0]) + (haystack[1] == needle[1]) == 2) return haystack;
#endif

    sz_u64_vec_t haystack_even_vec, haystack_odd_vec, needle_vec, matches_even_vec, matches_odd_vec;
    needle_vec.u64 = 0;
    needle_vec.u8s[0] = needle[0], needle_vec.u8s[1] = needle[1];
    needle_vec.u64 *= 0x0001000100010001ull; // broadcast

    // This code simulates hyper-scalar execution, analyzing 8 offsets at a time.
    for (; haystack + 9 <= haystack_end; haystack += 8) {
        haystack_even_vec.u64 = *(sz_u64_t *)haystack;
        haystack_odd_vec.u64 = (haystack_even_vec.u64 >> 8) | ((sz_u64_t)haystack[8] << 56);
        matches_even_vec = sz_u64_each_2byte_equal_(haystack_even_vec, needle_vec);
        matches_odd_vec = sz_u64_each_2byte_equal_(haystack_odd_vec, needle_vec);
        matches_even_vec.u64 >>= 8;
        if (matches_even_vec.u64 + matches_odd_vec.u64) {
            sz_u64_t match_indicators = matches_even_vec.u64 | matches_odd_vec.u64;
            return haystack + sz_u64_ctz(match_indicators) / 8;
        }
    }

    for (; haystack + 2 <= haystack_end; ++haystack)
        if ((haystack[0] == needle[0]) + (haystack[1] == needle[1]) == 2) return haystack;
    return SZ_NULL_CHAR;
}

/**
 *  @brief 4-byte-level equality comparison between two 64-bit integers.
 *  @return 64-bit integer, where every top bit in each 4-byte group signifies a match.
 */
SZ_HELPER_INLINE sz_u64_vec_t sz_u64_each_4byte_equal_(sz_u64_vec_t a, sz_u64_vec_t b) {
    sz_u64_vec_t vec;
    vec.u64 = ~(a.u64 ^ b.u64);
    // The match is valid, if every bit within each 4-byte group is set.
    // For that take the bottom 31 bits of each 4-byte group, add one to them,
    // and if this sets the top bit to one, then all the 31 bits are ones as well.
    vec.u64 = ((vec.u64 & 0x7FFFFFFF7FFFFFFFull) + 0x0000000100000001ull) & ((vec.u64 & 0x8000000080000000ull));
    return vec;
}

/**
 *  @brief Find the first occurrence of a @b four-character needle in an arbitrary length haystack.
 *         This implementation uses hardware-agnostic SWAR technique, to process 8 possible offsets at a time.
 */
SZ_HELPER_NOINLINE sz_cptr_t sz_find_4byte_serial_(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle,
                                                   sz_size_t needle_length) {

    // This is an internal method, and the haystack is guaranteed to be at least 4 bytes long.
    sz_assert_(haystack_length >= 4 && "The haystack is too short.");
    sz_unused_(needle_length); //? We keep this argument only for `sz_find_t` signature compatibility.
    sz_cptr_t const haystack_end = haystack + haystack_length;

    // On big-endian systems, skip SWAR and use simple serial search
#if SZ_IS_BIG_ENDIAN_
    for (; haystack + 4 <= haystack_end; ++haystack)
        if ((haystack[0] == needle[0]) + (haystack[1] == needle[1]) + (haystack[2] == needle[2]) +
                (haystack[3] == needle[3]) ==
            4)
            return haystack;
    return SZ_NULL_CHAR;
#endif

    // Process the misaligned head, to void UB on unaligned 64-bit loads.
#if !SZ_USE_MISALIGNED_LOADS
    for (; ((sz_size_t)haystack & 7ull) && haystack + 4 <= haystack_end; ++haystack)
        if ((haystack[0] == needle[0]) + (haystack[1] == needle[1]) + (haystack[2] == needle[2]) +
                (haystack[3] == needle[3]) ==
            4)
            return haystack;
#endif

    sz_u64_vec_t haystack0_vec, haystack1_vec, haystack2_vec, haystack3_vec, needle_vec;
    sz_u64_vec_t matches0_vec, matches1_vec, matches2_vec, matches3_vec;
    needle_vec.u64 = 0;
    needle_vec.u8s[0] = needle[0], needle_vec.u8s[1] = needle[1], needle_vec.u8s[2] = needle[2],
    needle_vec.u8s[3] = needle[3];
    needle_vec.u64 *= 0x0000000100000001ull; // broadcast

    // This code simulates hyper-scalar execution, analyzing 8 offsets at a time using four 64-bit words.
    // We load the subsequent four-byte word as well, taking its first bytes. Think of it as a glorified prefetch :)
    sz_u64_t haystack_page_current, haystack_page_next;
    for (; haystack + sizeof(sz_u64_t) + sizeof(sz_u32_t) <= haystack_end; haystack += sizeof(sz_u64_t)) {
        haystack_page_current = *(sz_u64_t *)haystack;
        haystack_page_next = *(sz_u32_t *)(haystack + 8);
        haystack0_vec.u64 = (haystack_page_current);
        haystack1_vec.u64 = (haystack_page_current >> 8) | (haystack_page_next << 56);
        haystack2_vec.u64 = (haystack_page_current >> 16) | (haystack_page_next << 48);
        haystack3_vec.u64 = (haystack_page_current >> 24) | (haystack_page_next << 40);
        matches0_vec = sz_u64_each_4byte_equal_(haystack0_vec, needle_vec);
        matches1_vec = sz_u64_each_4byte_equal_(haystack1_vec, needle_vec);
        matches2_vec = sz_u64_each_4byte_equal_(haystack2_vec, needle_vec);
        matches3_vec = sz_u64_each_4byte_equal_(haystack3_vec, needle_vec);

        if (matches0_vec.u64 | matches1_vec.u64 | matches2_vec.u64 | matches3_vec.u64) {
            matches0_vec.u64 >>= 24;
            matches1_vec.u64 >>= 16;
            matches2_vec.u64 >>= 8;
            sz_u64_t match_indicators = matches0_vec.u64 | matches1_vec.u64 | matches2_vec.u64 | matches3_vec.u64;
            return haystack + sz_u64_ctz(match_indicators) / 8;
        }
    }

    for (; haystack + 4 <= haystack_end; ++haystack)
        if ((haystack[0] == needle[0]) + (haystack[1] == needle[1]) + (haystack[2] == needle[2]) +
                (haystack[3] == needle[3]) ==
            4)
            return haystack;
    return SZ_NULL_CHAR;
}

/**
 *  @brief 3-byte-level equality comparison between two 64-bit integers.
 *  @return 64-bit integer, where every top bit in each 3-byte group signifies a match.
 */
SZ_HELPER_INLINE sz_u64_vec_t sz_u64_each_3byte_equal_(sz_u64_vec_t a, sz_u64_vec_t b) {
    sz_u64_vec_t vec;
    vec.u64 = ~(a.u64 ^ b.u64);
    // The match is valid, if every bit within each 4-byte group is set.
    // For that take the bottom 31 bits of each 4-byte group, add one to them,
    // and if this sets the top bit to one, then all the 31 bits are ones as well.
    vec.u64 = ((vec.u64 & 0xFFFF7FFFFF7FFFFFull) + 0x0000000001000001ull) & ((vec.u64 & 0x0000800000800000ull));
    return vec;
}

/**
 *  @brief Find the first occurrence of a @b three-character needle in an arbitrary length haystack.
 *         This implementation uses hardware-agnostic SWAR technique, to process 8 possible offsets at a time.
 */
SZ_HELPER_NOINLINE sz_cptr_t sz_find_3byte_serial_(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle,
                                                   sz_size_t needle_length) {

    // This is an internal method, and the haystack is guaranteed to be at least 4 bytes long.
    sz_assert_(haystack_length >= 3 && "The haystack is too short.");
    sz_unused_(needle_length); //? We keep this argument only for `sz_find_t` signature compatibility.
    sz_cptr_t const haystack_end = haystack + haystack_length;

    // On big-endian systems, skip SWAR and use simple serial search
#if SZ_IS_BIG_ENDIAN_
    for (; haystack + 3 <= haystack_end; ++haystack)
        if ((haystack[0] == needle[0]) + (haystack[1] == needle[1]) + (haystack[2] == needle[2]) == 3) return haystack;
    return SZ_NULL_CHAR;
#endif

    // Process the misaligned head, to void UB on unaligned 64-bit loads.
#if !SZ_USE_MISALIGNED_LOADS
    for (; ((sz_size_t)haystack & 7ull) && haystack + 3 <= haystack_end; ++haystack)
        if ((haystack[0] == needle[0]) + (haystack[1] == needle[1]) + (haystack[2] == needle[2]) == 3) return haystack;
#endif

    // We fetch 12
    sz_u64_vec_t haystack0_vec, haystack1_vec, haystack2_vec, haystack3_vec, haystack4_vec;
    sz_u64_vec_t matches0_vec, matches1_vec, matches2_vec, matches3_vec, matches4_vec;
    sz_u64_vec_t needle_vec;
    needle_vec.u64 = 0;
    needle_vec.u8s[0] = needle[0], needle_vec.u8s[1] = needle[1], needle_vec.u8s[2] = needle[2];
    needle_vec.u64 *= 0x0000000001000001ull; // broadcast

    // This code simulates hyper-scalar execution, analyzing 8 offsets at a time using three 64-bit words.
    // We load the subsequent two-byte word as well.
    sz_u64_t haystack_page_current, haystack_page_next;
    for (; haystack + sizeof(sz_u64_t) + sizeof(sz_u16_t) <= haystack_end; haystack += sizeof(sz_u64_t)) {
        haystack_page_current = *(sz_u64_t *)haystack;
        haystack_page_next = *(sz_u16_t *)(haystack + 8);
        haystack0_vec.u64 = (haystack_page_current);
        haystack1_vec.u64 = (haystack_page_current >> 8) | (haystack_page_next << 56);
        haystack2_vec.u64 = (haystack_page_current >> 16) | (haystack_page_next << 48);
        haystack3_vec.u64 = (haystack_page_current >> 24) | (haystack_page_next << 40);
        haystack4_vec.u64 = (haystack_page_current >> 32) | (haystack_page_next << 32);
        matches0_vec = sz_u64_each_3byte_equal_(haystack0_vec, needle_vec);
        matches1_vec = sz_u64_each_3byte_equal_(haystack1_vec, needle_vec);
        matches2_vec = sz_u64_each_3byte_equal_(haystack2_vec, needle_vec);
        matches3_vec = sz_u64_each_3byte_equal_(haystack3_vec, needle_vec);
        matches4_vec = sz_u64_each_3byte_equal_(haystack4_vec, needle_vec);

        if (matches0_vec.u64 | matches1_vec.u64 | matches2_vec.u64 | matches3_vec.u64 | matches4_vec.u64) {
            matches0_vec.u64 >>= 16;
            matches1_vec.u64 >>= 8;
            matches3_vec.u64 <<= 8;
            matches4_vec.u64 <<= 16;
            sz_u64_t match_indicators = matches0_vec.u64 | matches1_vec.u64 | matches2_vec.u64 | matches3_vec.u64 |
                                        matches4_vec.u64;
            return haystack + sz_u64_ctz(match_indicators) / 8;
        }
    }

    for (; haystack + 3 <= haystack_end; ++haystack)
        if ((haystack[0] == needle[0]) + (haystack[1] == needle[1]) + (haystack[2] == needle[2]) == 3) return haystack;
    return SZ_NULL_CHAR;
}

/**
 *  @brief Boyer-Moore-Horspool algorithm for exact matching of patterns up to @b 256-bytes long.
 *         Uses the Raita heuristic to match the first two, the last, and the middle character of the pattern.
 *
 *  @param haystack The haystack bytes.
 *  @param haystack_length Length of the haystack in bytes.
 *  @param needle The needle bytes.
 *  @param needle_length Length of the needle in bytes (must be <= 256).
 *  @return Pointer to first match, or SZ_NULL_CHAR if none.
 */
SZ_HELPER_NOINLINE sz_cptr_t sz_find_horspool_upto_256bytes_serial_( //
    sz_cptr_t haystack, sz_size_t haystack_length,                   //
    sz_cptr_t needle, sz_size_t needle_length) {
    sz_assert_(needle_length <= 256 && "The pattern is too long.");
    // Several popular string matching algorithms are using a bad-character shift table.
    // Boyer Moore: https://www-igm.univ-mlv.fr/~lecroq/string/node14.html
    // Quick Search: https://www-igm.univ-mlv.fr/~lecroq/string/node19.html
    // Smith: https://www-igm.univ-mlv.fr/~lecroq/string/node21.html
    union {
        sz_u8_t jumps[256];
        sz_u64_vec_t vecs[64];
    } bad_shift_table;

    // Let's initialize the table using SWAR to the total length of the string.
    sz_u8_t const *haystack_u8 = (sz_u8_t const *)haystack;
    sz_u8_t const *needle_u8 = (sz_u8_t const *)needle;
    {
        sz_u64_vec_t needle_length_vec;
        needle_length_vec.u64 = ((sz_u8_t)(needle_length - 1)) * 0x0101010101010101ull; // broadcast
        for (sz_size_t byte_index = 0; byte_index != 64; ++byte_index)
            bad_shift_table.vecs[byte_index].u64 = needle_length_vec.u64;
        for (sz_size_t byte_index = 0; byte_index + 1 < needle_length; ++byte_index)
            bad_shift_table.jumps[needle_u8[byte_index]] = (sz_u8_t)(needle_length - byte_index - 1);
    }

    // Another common heuristic is to match a few characters from different parts of a string.
    // Raita suggests to use the first two, the last, and the middle character of the pattern.
    sz_u32_vec_t haystack_vec, needle_vec;

    // Pick the parts of the needle that are worth comparing.
    sz_size_t offset_first, offset_mid, offset_last;
    sz_locate_needle_anomalies_(needle, needle_length, &offset_first, &offset_mid, &offset_last);

    // Broadcast those characters into an unsigned integer.
    needle_vec.u8s[0] = needle_u8[offset_first];
    needle_vec.u8s[1] = needle_u8[offset_first + 1];
    needle_vec.u8s[2] = needle_u8[offset_mid];
    needle_vec.u8s[3] = needle_u8[offset_last];

    // Scan through the whole haystack, skipping the last `needle_length - 1` bytes.
    for (sz_size_t byte_index = 0; byte_index <= haystack_length - needle_length;) {
        haystack_vec.u8s[0] = haystack_u8[byte_index + offset_first];
        haystack_vec.u8s[1] = haystack_u8[byte_index + offset_first + 1];
        haystack_vec.u8s[2] = haystack_u8[byte_index + offset_mid];
        haystack_vec.u8s[3] = haystack_u8[byte_index + offset_last];
        if (haystack_vec.u32 == needle_vec.u32 &&
            sz_equal_serial((sz_cptr_t)haystack_u8 + byte_index, needle, needle_length))
            return (sz_cptr_t)haystack_u8 + byte_index;
        byte_index += bad_shift_table.jumps[haystack_u8[byte_index + needle_length - 1]];
    }
    return SZ_NULL_CHAR;
}

/**
 *  @brief Boyer-Moore-Horspool algorithm for @b reverse-order exact matching of patterns up to @b 256-bytes long.
 *         Uses the Raita heuristic to match the first two, the last, and the middle character of the pattern.
 *
 *  @param haystack The haystack bytes.
 *  @param haystack_length Length of the haystack in bytes.
 *  @param needle The needle bytes.
 *  @param needle_length Length of the needle in bytes (must be <= 256).
 *  @return Pointer to last match, or SZ_NULL_CHAR if none.
 */
SZ_HELPER_NOINLINE sz_cptr_t sz_rfind_horspool_upto_256bytes_serial_( //
    sz_cptr_t haystack, sz_size_t haystack_length,                    //
    sz_cptr_t needle, sz_size_t needle_length) {
    sz_assert_(needle_length <= 256 && "The pattern is too long.");
    union {
        sz_u8_t jumps[256];
        sz_u64_vec_t vecs[64];
    } bad_shift_table;

    // Let's initialize the table using SWAR to the total length of the string.
    sz_u8_t const *haystack_u8 = (sz_u8_t const *)haystack;
    sz_u8_t const *needle_u8 = (sz_u8_t const *)needle;
    {
        sz_u64_vec_t needle_length_vec;
        needle_length_vec.u64 = ((sz_u8_t)(needle_length - 1)) * 0x0101010101010101ull; // broadcast
        for (sz_size_t byte_index = 0; byte_index != 64; ++byte_index)
            bad_shift_table.vecs[byte_index].u64 = needle_length_vec.u64;
        for (sz_size_t byte_index = 0; byte_index + 1 < needle_length; ++byte_index)
            bad_shift_table.jumps[needle_u8[needle_length - byte_index - 1]] = (sz_u8_t)(needle_length - byte_index -
                                                                                         1);
    }

    // Another common heuristic is to match a few characters from different parts of a string.
    // Raita suggests to use the first two, the last, and the middle character of the pattern.
    sz_u32_vec_t haystack_vec, needle_vec;

    // Pick the parts of the needle that are worth comparing.
    sz_size_t offset_first, offset_mid, offset_last;
    sz_locate_needle_anomalies_(needle, needle_length, &offset_first, &offset_mid, &offset_last);

    // Broadcast those characters into an unsigned integer.
    needle_vec.u8s[0] = needle_u8[offset_first];
    needle_vec.u8s[1] = needle_u8[offset_first + 1];
    needle_vec.u8s[2] = needle_u8[offset_mid];
    needle_vec.u8s[3] = needle_u8[offset_last];

    // Scan through the whole haystack, skipping the first `needle_length - 1` bytes.
    for (sz_size_t skip_index = 0; skip_index <= haystack_length - needle_length;) {
        sz_size_t byte_index = haystack_length - needle_length - skip_index;
        haystack_vec.u8s[0] = haystack_u8[byte_index + offset_first];
        haystack_vec.u8s[1] = haystack_u8[byte_index + offset_first + 1];
        haystack_vec.u8s[2] = haystack_u8[byte_index + offset_mid];
        haystack_vec.u8s[3] = haystack_u8[byte_index + offset_last];
        if (haystack_vec.u32 == needle_vec.u32 &&
            sz_equal_serial((sz_cptr_t)haystack_u8 + byte_index, needle, needle_length))
            return (sz_cptr_t)haystack_u8 + byte_index;
        skip_index += bad_shift_table.jumps[haystack_u8[byte_index]];
    }
    return SZ_NULL_CHAR;
}

/**
 *  @brief Exact substring search helper function, that finds the first occurrence of a prefix of the needle
 *         using a given search function, and then verifies the remaining part of the needle.
 *
 *  @param haystack Pointer to the haystack.
 *  @param haystack_length Length of the haystack in bytes.
 *  @param needle Pointer to the needle.
 *  @param needle_length Length of the needle in bytes.
 *  @param find_prefix Function used to search for the prefix.
 *  @param prefix_length Length of the prefix to search for.
 *  @return Pointer to first match, or SZ_NULL_CHAR if none.
 */
SZ_HELPER_AUTO sz_cptr_t sz_find_with_prefix_( //
    sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle, sz_size_t needle_length, sz_find_t find_prefix,
    sz_size_t prefix_length) {

    sz_size_t suffix_length = needle_length - prefix_length;
    while (1) {
        sz_cptr_t found = find_prefix(haystack, haystack_length, needle, prefix_length);
        if (!found) return SZ_NULL_CHAR;

        // Verify the remaining part of the needle
        sz_size_t remaining = haystack_length - (found - haystack);
        if (remaining < needle_length) return SZ_NULL_CHAR;
        if (sz_equal_serial(found + prefix_length, needle + prefix_length, suffix_length)) return found;

        // Adjust the position.
        haystack = found + 1;
        haystack_length = remaining - 1;
    }

    // Unreachable, but helps silence compiler warnings:
    return SZ_NULL_CHAR;
}

/**
 *  @brief Exact reverse-order substring search helper function, that finds the last occurrence of a suffix of the
 *         needle using a given search function, and then verifies the remaining part of the needle.
 *
 *  @param haystack Pointer to the haystack.
 *  @param haystack_length Length of the haystack in bytes.
 *  @param needle Pointer to the needle.
 *  @param needle_length Length of the needle in bytes.
 *  @param find_suffix Function used to search for the suffix.
 *  @param suffix_length Length of the suffix to search for.
 *  @return Pointer to last match start, or SZ_NULL_CHAR if none.
 */
SZ_HELPER_AUTO sz_cptr_t sz_rfind_with_suffix_(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle,
                                               sz_size_t needle_length, sz_find_t find_suffix,
                                               sz_size_t suffix_length) {

    sz_size_t prefix_length = needle_length - suffix_length;
    while (1) {
        sz_cptr_t found = find_suffix(haystack, haystack_length, needle + prefix_length, suffix_length);
        if (!found) return SZ_NULL_CHAR;

        // Verify the remaining part of the needle
        sz_size_t remaining = found - haystack;
        if (remaining < prefix_length) return SZ_NULL_CHAR;
        if (sz_equal_serial(found - prefix_length, needle, prefix_length)) return found - prefix_length;

        // Adjust the position.
        haystack_length = remaining - 1;
    }

    // Unreachable, but helps silence compiler warnings:
    return SZ_NULL_CHAR;
}

SZ_HELPER_NOINLINE sz_cptr_t sz_find_over_4bytes_serial_(sz_cptr_t haystack, sz_size_t haystack_length,
                                                         sz_cptr_t needle, sz_size_t needle_length) {
    return sz_find_with_prefix_(haystack, haystack_length, needle, needle_length, (sz_find_t)sz_find_4byte_serial_, 4);
}

SZ_HELPER_NOINLINE sz_cptr_t sz_find_horspool_over_256bytes_serial_( //
    sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle, sz_size_t needle_length) {
    return sz_find_with_prefix_(haystack, haystack_length, needle, needle_length,
                                sz_find_horspool_upto_256bytes_serial_, 256);
}

SZ_HELPER_NOINLINE sz_cptr_t sz_rfind_horspool_over_256bytes_serial_( //
    sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle, sz_size_t needle_length) {
    return sz_rfind_with_suffix_(haystack, haystack_length, needle, needle_length,
                                 sz_rfind_horspool_upto_256bytes_serial_, 256);
}

SZ_API_COMPTIME sz_cptr_t sz_find_serial(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle,
                                         sz_size_t needle_length) {
    // Empty needle matches at the start, like `strstr`.
    if (!needle_length) return haystack;
    if (haystack_length < needle_length) return SZ_NULL_CHAR;

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
        (needle_length > 1) + (needle_length > 2) + (needle_length > 3) +
        // To avoid constructing the skip-table, let's use the prefixed approach.
        (needle_length > 4) +
        // For longer needles - use skip tables.
        (needle_length > 8) + (needle_length > 256)](haystack, haystack_length, needle, needle_length);
}

SZ_API_COMPTIME sz_cptr_t sz_rfind_serial(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle,
                                          sz_size_t needle_length) {

    // Empty needle matches at the end.
    if (!needle_length) return haystack + haystack_length;
    if (haystack_length < needle_length) return SZ_NULL_CHAR;

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
        (needle_length > 1) +
        // For longer needles - use skip tables.
        (needle_length > 256)](haystack, haystack_length, needle, needle_length);
}

#ifdef __cplusplus
}
#endif

#endif // STRINGZILLA_FIND_SERIAL_H_
