/**
 *  @file include/stringzilla/find/serial.h
 *  @author Ash Vardanian
 *  @date August 14, 2020
 *  @brief Serial backend for substring & byte-set search.
 *
 *  @sa include/stringzilla/find.h
 */
#ifndef STRINGZILLA_FIND_SERIAL_H_
#define STRINGZILLA_FIND_SERIAL_H_

#include "stringzilla/types.h"
#include "stringzilla/compare/serial.h" // `sz_equal_serial_`

#ifdef __cplusplus
extern "C" {
#endif

/**
 *  @brief Chooses the offsets of the most interesting characters in a search needle.
 *
 *  Search throughput can significantly deteriorate if we are matching the wrong characters. Say the
 *  needle is "aXaYa", and we are comparing the first, second, and last character. If we use SIMD
 *  and compare many offsets at a time, comparing against "a" in every register is a waste.
 *
 *  Similarly, dealing with UTF-8 inputs, we know that the lower bits of each character code carry
 *  more information. Cyrillic alphabet, for example, falls into [0x0410, 0x042F] code range for
 *  uppercase [А, Я], and into [0x0430, 0x044F] for lowercase [а, я]. Scanning through a text
 *  written in Russian, half of the bytes will carry absolutely no value and will be equal to 0x04.
 *
 *  @param[in] start Pointer to the needle bytes.
 *  @param[in] length Length of the needle in bytes.
 *  @param[out] first Output offset of the first anomalous byte.
 *  @param[out] second Output offset of the second anomalous byte.
 *  @param[out] third Output offset of the third anomalous byte.
 */
STRINGZILLA_CONSTEXPR void sz_locate_needle_anomalies_( //
    sz_cptr_t start, sz_size_t length,                  //
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

/** Number of byte values present in @p set - four branchless word popcounts. */
STRINGZILLA_INLINE sz_size_t sz_byteset_population_serial_(sz_byteset_t const *set) {
    return (sz_size_t)(sz_u64_popcount(set->_u64s[0]) + sz_u64_popcount(set->_u64s[1]) +
                       sz_u64_popcount(set->_u64s[2]) + sz_u64_popcount(set->_u64s[3]));
}

/** Unpack the member byte values of @p set into @p members in ascending order; the caller has
 *  already sized the destination from @ref sz_byteset_population_serial_. Shared by the ISA
 *  back-ends whose small-set fast paths broadcast the members as a needle segment. */
STRINGZILLA_INLINE void sz_byteset_members_serial_(sz_byteset_t const *set, sz_u8_t *members) {
    sz_size_t filled = 0;
    for (sz_size_t word_index = 0; word_index != 4; ++word_index)
        for (sz_u64_t word_bits = set->_u64s[word_index]; word_bits; word_bits &= word_bits - 1)
            members[filled++] = (sz_u8_t)(word_index * 64 + sz_u64_ctz(word_bits));
}

STRINGZILLA_INLINE sz_cptr_t sz_find_byteset_serial_(sz_cptr_t haystack, sz_size_t haystack_length,
                                                     sz_byteset_t const *set) {
    for (sz_cptr_t const end = haystack + haystack_length; haystack != end; ++haystack)
        if (sz_byteset_contains(set, *haystack)) return haystack;
    return STRINGZILLA_NULL_CHAR;
}

STRINGZILLA_INLINE sz_cptr_t sz_rfind_byteset_serial_(sz_cptr_t haystack, sz_size_t haystack_length,
                                                      sz_byteset_t const *set) {
    sz_cptr_t const end = haystack;
    for (haystack += haystack_length; haystack != end;)
        if (sz_byteset_contains(set, *(haystack -= 1))) return haystack;
    return STRINGZILLA_NULL_CHAR;
}

STRINGZILLA_INLINE sz_cptr_t sz_find_byte_serial_(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle) {

    if (!haystack_length) return STRINGZILLA_NULL_CHAR;
    // Reinterpret as unsigned bytes so the SWAR broadcast below cannot sign-extend
    // on platforms where `char` is signed (e.g. `-fsigned-char`). See issue #306.
    sz_u8_t const *haystack_cursor = (sz_u8_t const *)haystack;
    sz_u8_t const *const needle_u8 = (sz_u8_t const *)needle;
    sz_u8_t const *const haystack_end = haystack_cursor + haystack_length;

#if !STRINGZILLA_ARCH_BIG_ENDIAN_       // Use SWAR only on little-endian platforms for brevity.
#if !STRINGZILLA_ALLOW_MISALIGNED_LOADS // Process the misaligned head, to void UB on unaligned 64-bit loads.
    for (; ((sz_size_t)haystack_cursor & 7ull) && haystack_cursor < haystack_end; ++haystack_cursor)
        if (*haystack_cursor == *needle_u8) return (sz_cptr_t)haystack_cursor;
#endif

    // Broadcast the needle into every byte of a 64-bit integer to use SWAR
    // techniques and process eight characters at a time.
    sz_u64_vec_t haystack_vec, needle_vec, match_vec;
    match_vec.u64 = 0;
    needle_vec.u64 = (sz_u64_t)*needle_u8 * 0x0101010101010101ull;
    for (; haystack_cursor + 8 <= haystack_end; haystack_cursor += 8) {
        haystack_vec.u64 = sz_u64_load((sz_cptr_t)haystack_cursor).u64;
        match_vec = sz_u64_each_byte_equal_(haystack_vec, needle_vec);
        if (match_vec.u64) return (sz_cptr_t)(haystack_cursor + sz_u64_ctz(match_vec.u64) / 8);
    }
#endif

    // Handle the misaligned tail.
    for (; haystack_cursor < haystack_end; ++haystack_cursor)
        if (*haystack_cursor == *needle_u8) return (sz_cptr_t)haystack_cursor;
    return STRINGZILLA_NULL_CHAR;
}

STRINGZILLA_INLINE sz_cptr_t sz_rfind_byte_serial_(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle) {

    if (!haystack_length) return STRINGZILLA_NULL_CHAR;
    // Reinterpret as unsigned bytes so the SWAR broadcast below cannot sign-extend
    // on platforms where `char` is signed (e.g. `-fsigned-char`). See issue #306.
    sz_u8_t const *const haystack_start = (sz_u8_t const *)haystack;
    sz_u8_t const *const needle_u8 = (sz_u8_t const *)needle;

    // Reposition the cursor to the end, as we will be walking backwards.
    sz_u8_t const *haystack_cursor = haystack_start + haystack_length - 1;

#if !STRINGZILLA_ARCH_BIG_ENDIAN_       // Use SWAR only on little-endian platforms for brevity.
#if !STRINGZILLA_ALLOW_MISALIGNED_LOADS // Process the misaligned head, to void UB on unaligned 64-bit loads.
    for (; ((sz_size_t)(haystack_cursor + 1) & 7ull) && haystack_cursor >= haystack_start; --haystack_cursor)
        if (*haystack_cursor == *needle_u8) return (sz_cptr_t)haystack_cursor;
#endif

    // Broadcast the needle into every byte of a 64-bit integer to use SWAR
    // techniques and process eight characters at a time.
    sz_u64_vec_t haystack_vec, needle_vec, match_vec;
    needle_vec.u64 = (sz_u64_t)*needle_u8 * 0x0101010101010101ull;
    for (; haystack_cursor >= haystack_start + 7; haystack_cursor -= 8) {
        haystack_vec.u64 = sz_u64_load((sz_cptr_t)(haystack_cursor - 7)).u64;
        match_vec = sz_u64_each_byte_equal_(haystack_vec, needle_vec);
        if (match_vec.u64) return (sz_cptr_t)(haystack_cursor - sz_u64_clz(match_vec.u64) / 8);
    }
#endif

    for (; haystack_cursor >= haystack_start; --haystack_cursor)
        if (*haystack_cursor == *needle_u8) return (sz_cptr_t)haystack_cursor;
    return STRINGZILLA_NULL_CHAR;
}

/**
 *  @brief 2-byte-level equality comparison between two 64-bit integers.
 *  @return 64-bit integer, where every top bit in each 2-byte group signifies a match.
 */
STRINGZILLA_INLINE sz_u64_vec_t sz_u64_each_2byte_equal_(sz_u64_vec_t a_vec, sz_u64_vec_t b_vec) {
    sz_u64_vec_t vec_vec;
    vec_vec.u64 = ~(a_vec.u64 ^ b_vec.u64);
    // The match is valid, if every bit within each 2-byte group is set.
    // For that take the bottom 15 bits of each 2-byte group, add one to them,
    // and if this sets the top bit to one, then all the 15 bits are ones as well.
    vec_vec.u64 = ((vec_vec.u64 & 0x7FFF7FFF7FFF7FFFull) + 0x0001000100010001ull) &
                  ((vec_vec.u64 & 0x8000800080008000ull));
    return vec_vec;
}

STRINGZILLA_OUTLINED_ sz_cptr_t sz_find_1byte_serial_(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle,
                                                      sz_size_t needle_length) {
    sz_unused_(needle_length); //? We keep this argument only to share the signature of the other backends.
    return sz_find_byte_serial_(haystack, haystack_length, needle);
}

STRINGZILLA_OUTLINED_ sz_cptr_t sz_rfind_1byte_serial_(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle,
                                                       sz_size_t needle_length) {
    sz_unused_(needle_length); //? We keep this argument only to share the signature of the other backends.
    return sz_rfind_byte_serial_(haystack, haystack_length, needle);
}

/** Find the first occurrence of a @b two-character needle in an arbitrary length haystack, using a
 *  hardware-agnostic SWAR technique to process 8 possible offsets at a time. */
STRINGZILLA_OUTLINED_ sz_cptr_t sz_find_2byte_serial_(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle,
                                                      sz_size_t needle_length) {

    // This is an internal method, and the haystack is guaranteed to be at least 2 bytes long.
    sz_assert_(haystack_length >= 2 && "The haystack is too short.");
    sz_unused_(needle_length); //? We keep this argument only to share the signature of the other backends.
    sz_cptr_t const haystack_end = haystack + haystack_length;

    // On big-endian systems, skip SWAR and use simple serial search
#if STRINGZILLA_ARCH_BIG_ENDIAN_
    for (; haystack + 2 <= haystack_end; ++haystack)
        if ((haystack[0] == needle[0]) + (haystack[1] == needle[1]) == 2) return haystack;
    return STRINGZILLA_NULL_CHAR;
#endif

    // Process the misaligned head, to void UB on unaligned 64-bit loads.
#if !STRINGZILLA_ALLOW_MISALIGNED_LOADS
    for (; ((sz_size_t)haystack & 7ull) && haystack + 2 <= haystack_end; ++haystack)
        if ((haystack[0] == needle[0]) + (haystack[1] == needle[1]) == 2) return haystack;
#endif

    sz_u64_vec_t haystack_even_vec, haystack_odd_vec, needle_vec, matches_even_vec, matches_odd_vec;
    needle_vec.u64 = 0;
    needle_vec.u8s[0] = needle[0], needle_vec.u8s[1] = needle[1];
    needle_vec.u64 *= 0x0001000100010001ull; // broadcast

    // This code simulates hyper-scalar execution, analyzing 8 offsets at a time.
    for (; haystack + 9 <= haystack_end; haystack += 8) {
        haystack_even_vec.u64 = sz_u64_load(haystack).u64;
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
    return STRINGZILLA_NULL_CHAR;
}

/**
 *  @brief 4-byte-level equality comparison between two 64-bit integers.
 *  @return 64-bit integer, where every top bit in each 4-byte group signifies a match.
 */
STRINGZILLA_INLINE sz_u64_vec_t sz_u64_each_4byte_equal_(sz_u64_vec_t a_vec, sz_u64_vec_t b_vec) {
    sz_u64_vec_t vec_vec;
    vec_vec.u64 = ~(a_vec.u64 ^ b_vec.u64);
    // The match is valid, if every bit within each 4-byte group is set.
    // For that take the bottom 31 bits of each 4-byte group, add one to them,
    // and if this sets the top bit to one, then all the 31 bits are ones as well.
    vec_vec.u64 = ((vec_vec.u64 & 0x7FFFFFFF7FFFFFFFull) + 0x0000000100000001ull) &
                  ((vec_vec.u64 & 0x8000000080000000ull));
    return vec_vec;
}

/** Find the first occurrence of a @b four-character needle in an arbitrary length haystack, using a
 *  hardware-agnostic SWAR technique to process 8 possible offsets at a time. */
STRINGZILLA_OUTLINED_ sz_cptr_t sz_find_4byte_serial_(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle,
                                                      sz_size_t needle_length) {

    // This is an internal method, and the haystack is guaranteed to be at least 4 bytes long.
    sz_assert_(haystack_length >= 4 && "The haystack is too short.");
    sz_unused_(needle_length); //? We keep this argument only to share the signature of the other backends.
    sz_cptr_t const haystack_end = haystack + haystack_length;

    // On big-endian systems, skip SWAR and use simple serial search
#if STRINGZILLA_ARCH_BIG_ENDIAN_
    for (; haystack + 4 <= haystack_end; ++haystack)
        if ((haystack[0] == needle[0]) + (haystack[1] == needle[1]) + (haystack[2] == needle[2]) +
                (haystack[3] == needle[3]) ==
            4)
            return haystack;
    return STRINGZILLA_NULL_CHAR;
#endif

    // Process the misaligned head, to void UB on unaligned 64-bit loads.
#if !STRINGZILLA_ALLOW_MISALIGNED_LOADS
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
        haystack_page_current = sz_u64_load(haystack).u64;
        haystack_page_next = sz_u32_load(haystack + 8).u32;
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
    return STRINGZILLA_NULL_CHAR;
}

/**
 *  @brief 3-byte-level equality comparison between two 64-bit integers.
 *  @return 64-bit integer, where every top bit in each 3-byte group signifies a match.
 */
STRINGZILLA_INLINE sz_u64_vec_t sz_u64_each_3byte_equal_(sz_u64_vec_t a_vec, sz_u64_vec_t b_vec) {
    sz_u64_vec_t vec_vec;
    vec_vec.u64 = ~(a_vec.u64 ^ b_vec.u64);
    // The match is valid, if every bit within each 4-byte group is set.
    // For that take the bottom 31 bits of each 4-byte group, add one to them,
    // and if this sets the top bit to one, then all the 31 bits are ones as well.
    vec_vec.u64 = ((vec_vec.u64 & 0xFFFF7FFFFF7FFFFFull) + 0x0000000001000001ull) &
                  ((vec_vec.u64 & 0x0000800000800000ull));
    return vec_vec;
}

/** Find the first occurrence of a @b three-character needle in an arbitrary length haystack, using
 *  a hardware-agnostic SWAR technique to process 8 possible offsets at a time. */
STRINGZILLA_OUTLINED_ sz_cptr_t sz_find_3byte_serial_(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle,
                                                      sz_size_t needle_length) {

    // This is an internal method, and the haystack is guaranteed to be at least 4 bytes long.
    sz_assert_(haystack_length >= 3 && "The haystack is too short.");
    sz_unused_(needle_length); //? We keep this argument only to share the signature of the other backends.
    sz_cptr_t const haystack_end = haystack + haystack_length;

    // On big-endian systems, skip SWAR and use simple serial search
#if STRINGZILLA_ARCH_BIG_ENDIAN_
    for (; haystack + 3 <= haystack_end; ++haystack)
        if ((haystack[0] == needle[0]) + (haystack[1] == needle[1]) + (haystack[2] == needle[2]) == 3) return haystack;
    return STRINGZILLA_NULL_CHAR;
#endif

    // Process the misaligned head, to void UB on unaligned 64-bit loads.
#if !STRINGZILLA_ALLOW_MISALIGNED_LOADS
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
        haystack_page_current = sz_u64_load(haystack).u64;
        haystack_page_next = sz_u16_load(haystack + 8).u16;
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
    return STRINGZILLA_NULL_CHAR;
}

/**
 *  @brief Boyer-Moore-Horspool algorithm for exact matching of patterns up to @b 256-bytes
 *      long. Uses the Raita heuristic to match the first two, the last, and the middle
 *      character of the pattern.
 *
 *  @param[in] haystack The haystack bytes.
 *  @param[in] haystack_length Length of the haystack in bytes.
 *  @param[in] needle The needle bytes.
 *  @param[in] needle_length Length of the needle in bytes (must be <= 256).
 *  @return Pointer to first match, or STRINGZILLA_NULL_CHAR if none.
 */
STRINGZILLA_OUTLINED_ sz_cptr_t sz_find_horspool_upto_256bytes_serial_( //
    sz_cptr_t haystack, sz_size_t haystack_length,                      //
    sz_cptr_t needle, sz_size_t needle_length) {
    sz_assert_(needle_length >= 2 && needle_length <= 256 &&
               "A 1-byte needle never advances, and a longer than 256-byte one overflows the shifts");
    sz_assert_(haystack_length >= needle_length && "The haystack is too short.");
    // Several popular string matching algorithms are using a bad-character shift table.
    // Boyer Moore: https://www-igm.univ-mlv.fr/~lecroq/string/node14.html
    // Quick Search: https://www-igm.univ-mlv.fr/~lecroq/string/node19.html
    // Smith: https://www-igm.univ-mlv.fr/~lecroq/string/node21.html
    union {
        sz_u8_t jumps[256];
        sz_u64_vec_t vecs_vec[64];
    } bad_shift_table;

    // Let's initialize the table using SWAR to the total length of the string.
    sz_u8_t const *haystack_u8 = (sz_u8_t const *)haystack;
    sz_u8_t const *needle_u8 = (sz_u8_t const *)needle;
    {
        sz_u64_vec_t needle_length_vec;
        needle_length_vec.u64 = ((sz_u8_t)(needle_length - 1)) * 0x0101010101010101ull; // broadcast
        for (sz_size_t byte_index = 0; byte_index != 64; ++byte_index)
            bad_shift_table.vecs_vec[byte_index].u64 = needle_length_vec.u64;
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
            sz_equal_serial_((sz_cptr_t)haystack_u8 + byte_index, needle, needle_length))
            return (sz_cptr_t)haystack_u8 + byte_index;
        byte_index += bad_shift_table.jumps[haystack_u8[byte_index + needle_length - 1]];
    }
    return STRINGZILLA_NULL_CHAR;
}

/**
 *  @brief Boyer-Moore-Horspool algorithm for @b reverse-order exact matching of patterns up to
 *      @b 256-bytes long. Uses the Raita heuristic to match the first two, the last, and the middle
 *      character of the pattern.
 *
 *  @param[in] haystack The haystack bytes.
 *  @param[in] haystack_length Length of the haystack in bytes.
 *  @param[in] needle The needle bytes.
 *  @param[in] needle_length Length of the needle in bytes (must be <= 256).
 *  @return Pointer to last match, or STRINGZILLA_NULL_CHAR if none.
 */
STRINGZILLA_OUTLINED_ sz_cptr_t sz_rfind_horspool_upto_256bytes_serial_( //
    sz_cptr_t haystack, sz_size_t haystack_length,                       //
    sz_cptr_t needle, sz_size_t needle_length) {
    sz_assert_(needle_length >= 2 && needle_length <= 256 &&
               "A 1-byte needle never advances, and a longer than 256-byte one overflows the shifts");
    sz_assert_(haystack_length >= needle_length && "The haystack is too short.");
    union {
        sz_u8_t jumps[256];
        sz_u64_vec_t vecs_vec[64];
    } bad_shift_table;

    // Let's initialize the table using SWAR to the total length of the string.
    sz_u8_t const *haystack_u8 = (sz_u8_t const *)haystack;
    sz_u8_t const *needle_u8 = (sz_u8_t const *)needle;
    {
        sz_u64_vec_t needle_length_vec;
        needle_length_vec.u64 = ((sz_u8_t)(needle_length - 1)) * 0x0101010101010101ull; // broadcast
        for (sz_size_t byte_index = 0; byte_index != 64; ++byte_index)
            bad_shift_table.vecs_vec[byte_index].u64 = needle_length_vec.u64;
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
            sz_equal_serial_((sz_cptr_t)haystack_u8 + byte_index, needle, needle_length))
            return (sz_cptr_t)haystack_u8 + byte_index;
        skip_index += bad_shift_table.jumps[haystack_u8[byte_index]];
    }
    return STRINGZILLA_NULL_CHAR;
}

/**
 *  @brief Exact substring search helper function, that finds the first occurrence of a
 *      prefix of the needle using a given search function, and then verifies the remaining
 *      part of the needle.
 *
 *  @param[in] haystack Pointer to the haystack.
 *  @param[in] haystack_length Length of the haystack in bytes.
 *  @param[in] needle Pointer to the needle.
 *  @param[in] needle_length Length of the needle in bytes.
 *  @param[in] find_prefix Function used to search for the prefix.
 *  @param[in] prefix_length Length of the prefix to search for.
 *  @return Pointer to first match, or STRINGZILLA_NULL_CHAR if none.
 */
STRINGZILLA_CONSTEXPR sz_cptr_t sz_find_with_prefix_( //
    sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle, sz_size_t needle_length,
    sz_cptr_t (*find_prefix)(sz_cptr_t, sz_size_t, sz_cptr_t, sz_size_t), sz_size_t prefix_length) {

    sz_size_t suffix_length = needle_length - prefix_length;
    while (1) {
        sz_cptr_t found = find_prefix(haystack, haystack_length, needle, prefix_length);
        if (!found) return STRINGZILLA_NULL_CHAR;

        // Verify the remaining part of the needle
        sz_size_t remaining = haystack_length - (found - haystack);
        if (remaining < needle_length) return STRINGZILLA_NULL_CHAR;
        if (sz_equal_serial_(found + prefix_length, needle + prefix_length, suffix_length)) return found;

        // Adjust the position.
        haystack = found + 1;
        haystack_length = remaining - 1;
    }

    // Unreachable, but helps silence compiler warnings:
    return STRINGZILLA_NULL_CHAR;
}

/**
 *  @brief Exact reverse-order substring search helper function, that finds the last occurrence of
 *      a suffix of the needle using a given search function, and then verifies the remaining part
 *      of the needle.
 *
 *  @param[in] haystack Pointer to the haystack.
 *  @param[in] haystack_length Length of the haystack in bytes.
 *  @param[in] needle Pointer to the needle.
 *  @param[in] needle_length Length of the needle in bytes.
 *  @param[in] find_suffix Function used to search for the suffix.
 *  @param[in] suffix_length Length of the suffix to search for.
 *  @return Pointer to last match start, or STRINGZILLA_NULL_CHAR if none.
 */
STRINGZILLA_CONSTEXPR sz_cptr_t sz_rfind_with_suffix_(
    sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle, sz_size_t needle_length,
    sz_cptr_t (*find_suffix)(sz_cptr_t, sz_size_t, sz_cptr_t, sz_size_t), sz_size_t suffix_length) {

    sz_size_t prefix_length = needle_length - suffix_length;
    while (1) {
        sz_cptr_t found = find_suffix(haystack, haystack_length, needle + prefix_length, suffix_length);
        if (!found) return STRINGZILLA_NULL_CHAR;

        // Verify the remaining part of the needle
        sz_size_t remaining = found - haystack;
        if (remaining < prefix_length) return STRINGZILLA_NULL_CHAR;
        if (sz_equal_serial_(found - prefix_length, needle, prefix_length)) return found - prefix_length;

        // Adjust the position, upholding `haystack_length >= needle_length` for the next `find_suffix`.
        if (remaining <= needle_length) return STRINGZILLA_NULL_CHAR;
        haystack_length = remaining - 1;
    }

    // Unreachable, but helps silence compiler warnings:
    return STRINGZILLA_NULL_CHAR;
}

STRINGZILLA_OUTLINED_ sz_cptr_t sz_find_over_4bytes_serial_(sz_cptr_t haystack, sz_size_t haystack_length,
                                                            sz_cptr_t needle, sz_size_t needle_length) {
    return sz_find_with_prefix_(haystack, haystack_length, needle, needle_length, sz_find_4byte_serial_, 4);
}

STRINGZILLA_OUTLINED_ sz_cptr_t sz_find_horspool_over_256bytes_serial_( //
    sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle, sz_size_t needle_length) {
    return sz_find_with_prefix_(haystack, haystack_length, needle, needle_length,
                                sz_find_horspool_upto_256bytes_serial_, 256);
}

STRINGZILLA_OUTLINED_ sz_cptr_t sz_rfind_horspool_over_256bytes_serial_( //
    sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle, sz_size_t needle_length) {
    return sz_rfind_with_suffix_(haystack, haystack_length, needle, needle_length,
                                 sz_rfind_horspool_upto_256bytes_serial_, 256);
}

STRINGZILLA_INLINE sz_cptr_t sz_find_serial_(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle,
                                             sz_size_t needle_length) {
    // Empty needle matches at the start, like `strstr`.
    if (!needle_length) return haystack;
    if (haystack_length < needle_length) return STRINGZILLA_NULL_CHAR;

    sz_cptr_t (*const backends[])(sz_cptr_t, sz_size_t, sz_cptr_t, sz_size_t) = {
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

STRINGZILLA_INLINE sz_cptr_t sz_rfind_serial_(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle,
                                              sz_size_t needle_length) {

    // Empty needle matches at the end.
    if (!needle_length) return haystack + haystack_length;
    if (haystack_length < needle_length) return STRINGZILLA_NULL_CHAR;

    sz_cptr_t (*const backends[])(sz_cptr_t, sz_size_t, sz_cptr_t, sz_size_t) = {
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

/** The 8 bytes at @p text, the first in the lowest byte whatever the platform's byte order. */
STRINGZILLA_INLINE sz_u64_t sz_find_word_serial_(sz_u8_t const *text) {
    sz_u64_t const word = sz_u64_load((sz_cptr_t)text).u64;
    return STRINGZILLA_ARCH_BIG_ENDIAN_ ? sz_u64_bytes_reverse(word) : word;
}

/** Bit @c i set where @p bytes match at `text + i + offsets[k]` for every @c k, for the first
 *  @p starts of the 64 starts from @p text, one start at a time. */
STRINGZILLA_INLINE sz_u64_t sz_find_candidates_serial_(sz_u8_t const *text, sz_size_t starts,
                                                       sz_size_t const offsets[3], sz_u8_t const bytes[3]) {
    sz_u64_t candidates = 0;
    for (sz_size_t start = 0; start != starts; ++start)
        candidates |= (sz_u64_t)((text[start + offsets[0]] == bytes[0]) & (text[start + offsets[1]] == bytes[1]) &
                                 (text[start + offsets[2]] == bytes[2]))
                      << start;
    return candidates;
}

/** Like @c sz_find_candidates_serial_ for all 64 starts, 8 at a time: a byte equal to its needle
 *  byte zeroes its lane of the three XORed words, and only a zero lane keeps its top bit clear. */
STRINGZILLA_INLINE sz_u64_t sz_find_candidates_swar_serial_(sz_u8_t const *text, sz_size_t const offsets[3],
                                                            sz_u8_t const bytes[3]) {
    sz_u64_t const lows = 0x7F7F7F7F7F7F7F7Full, ones = 0x0101010101010101ull;
    sz_u64_t candidates = 0;
    for (sz_size_t word = 0; word != 8; ++word) {
        sz_u8_t const *const starts = text + word * 8;
        sz_u64_t const differences = (sz_find_word_serial_(starts + offsets[0]) ^ bytes[0] * ones) |
                                     (sz_find_word_serial_(starts + offsets[1]) ^ bytes[1] * ones) |
                                     (sz_find_word_serial_(starts + offsets[2]) ^ bytes[2] * ones);
        sz_u64_t const zeros = ~(((differences & lows) + lows) | differences | lows);
        // The multiply packs the top bit of every byte into the top byte, the first byte lowest.
        candidates |= ((zeros >> 7) * 0x0102040810204080ull >> 56) << (word * 8);
    }
    return candidates;
}

#if STRINGZILLA_TARGET_SERIAL

STRINGZILLA_API sz_status_t sz_find_byteset_serial(sz_cptr_t haystack, sz_size_t haystack_length,
                                                   sz_byteset_t const *set, sz_cptr_t *match, sz_stream_t stream) {
    sz_assert_(stream == STRINGZILLA_NULL);
    *match = sz_find_byteset_serial_(haystack, haystack_length, set);
    return sz_success_k;
}

STRINGZILLA_API sz_status_t sz_rfind_byteset_serial(sz_cptr_t haystack, sz_size_t haystack_length,
                                                    sz_byteset_t const *set, sz_cptr_t *match, sz_stream_t stream) {
    sz_assert_(stream == STRINGZILLA_NULL);
    *match = sz_rfind_byteset_serial_(haystack, haystack_length, set);
    return sz_success_k;
}

STRINGZILLA_API sz_status_t sz_find_byte_serial(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle,
                                                sz_cptr_t *match, sz_stream_t stream) {
    sz_assert_(stream == STRINGZILLA_NULL);
    *match = sz_find_byte_serial_(haystack, haystack_length, needle);
    return sz_success_k;
}

STRINGZILLA_API sz_status_t sz_rfind_byte_serial(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle,
                                                 sz_cptr_t *match, sz_stream_t stream) {
    sz_assert_(stream == STRINGZILLA_NULL);
    *match = sz_rfind_byte_serial_(haystack, haystack_length, needle);
    return sz_success_k;
}

STRINGZILLA_API sz_status_t sz_find_serial(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle,
                                           sz_size_t needle_length, sz_cptr_t *match, sz_stream_t stream) {
    sz_assert_(stream == STRINGZILLA_NULL);
    *match = sz_find_serial_(haystack, haystack_length, needle, needle_length);
    return sz_success_k;
}

STRINGZILLA_API sz_status_t sz_find_strings_serial(sz_strings_view_t const *strings, sz_size_t const *order,
                                                   sz_size_t count, sz_cptr_t needle, sz_size_t needle_length,
                                                   sz_size_t *positions, sz_stream_t stream) {
    sz_assert_(stream == STRINGZILLA_NULL);
    for (sz_size_t row = 0; !needle_length && row != count; ++row) positions[row] = 0;
    if (!needle_length) return sz_success_k;
    // Orders and accessors search string by string, their views fetched 64 at a time.
    sz_string_view_t views[64];
    sz_bool_t const by_row = (sz_bool_t)(order || strings->layout == sz_strings_accessor_k);
    for (sz_size_t row = 0; by_row && row != count; ++row) {
        if (row % 64 == 0 && order)
            sz_strings_get_(strings, order + row, sz_min_of_two(count - row, (sz_size_t)64), views);
        if (row % 64 == 0 && !order) sz_strings_get_block_(strings, row, views);
        sz_string_view_t const view = views[row % 64];
        sz_cptr_t const match = sz_find_serial_(view.start, view.length, needle, needle_length);
        positions[row] = match ? (sz_size_t)(match - view.start) : STRINGZILLA_SIZE_MAX;
    }
    if (by_row || !count) return sz_success_k;

    sz_u8_t const *const data = (sz_u8_t const *)sz_strings_view_data_(strings);
    sz_size_t offsets[3];
    sz_locate_needle_anomalies_(needle, needle_length, &offsets[0], &offsets[1], &offsets[2]);
    sz_u8_t const bytes[3] = {(sz_u8_t)needle[offsets[0]], (sz_u8_t)needle[offsets[1]], (sz_u8_t)needle[offsets[2]]};
    sz_size_t const end = sz_strings_view_offset_(strings, count);
    sz_size_t row = 0, row_start = sz_strings_view_offset_(strings, 0), row_end = sz_strings_view_offset_(strings, 1);
    // A block of 64 starts begins at the first unsettled string, or right after the last block.
    for (sz_size_t block = row_start; row != count; block = sz_max_of_two(block + 64, row_start)) {
        sz_size_t const starts = end > block + offsets[2] ? sz_min_of_two(end - block - offsets[2], (sz_size_t)64) : 0;
        // The last block backs off to end where the payload does, dropping the starts it repeats.
        sz_size_t const back = 64 - starts;
        sz_u64_t candidates = starts && block >= back
                                  ? sz_find_candidates_swar_serial_(data + block - back, offsets, bytes) >> back
                                  : sz_find_candidates_serial_(data + block, starts, offsets, bytes);
        // Each step settles a string or drops one of its candidates, until one outruns the block.
        while (row != count && row_start < block + 64) {
            // The lanes of the string's starts in the block, none when the needle outgrows it.
            sz_size_t const first_lane = row_start > block ? row_start - block : 0;
            sz_size_t const end_lane = row_end - row_start < needle_length ? first_lane
                                                                           : row_end - needle_length + 1 - block;
            sz_u64_t const own = candidates & sz_u64_mask_until_serial_(end_lane) &
                                 ~sz_u64_mask_until_serial_(first_lane);
            if (!own && end_lane > 64) break;
            sz_size_t const lane = own ? (sz_size_t)sz_u64_ctz(own) : 64;
            candidates &= ~(own & (0 - own));
            if (own && needle_length > 3 && !sz_equal_serial_((sz_cptr_t)data + block + lane, needle, needle_length))
                continue;
            positions[row] = own ? block + lane - row_start : STRINGZILLA_SIZE_MAX;
            row_start = row_end, ++row;
            row_end = sz_strings_view_offset_(strings, sz_min_of_two(row + 1, count));
        }
    }
    return sz_success_k;
}

STRINGZILLA_API sz_status_t sz_rfind_serial(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle,
                                            sz_size_t needle_length, sz_cptr_t *match, sz_stream_t stream) {
    sz_assert_(stream == STRINGZILLA_NULL);
    *match = sz_rfind_serial_(haystack, haystack_length, needle, needle_length);
    return sz_success_k;
}

#endif // STRINGZILLA_TARGET_SERIAL

#ifdef __cplusplus
}
#endif

#endif // STRINGZILLA_FIND_SERIAL_H_
