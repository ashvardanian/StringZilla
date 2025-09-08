/**
 *  @brief  Hardware-accelerated sub-string and character-set search utilities.
 *  @file   find.h
 *  @author Ash Vardanian
 *
 *  Includes core APIs:
 *
 *  - `sz_find` and reverse-order `sz_rfind`
 *  - `sz_find_byte` and reverse-order `sz_rfind_byte`
 *  - `sz_find_byteset` and reverse-order `sz_rfind_byteset`
 *
 *  Convenience functions for character-set matching:
 *
 *  - `sz_find_byte_from` shortcut for `sz_find_byteset`
 *  - `sz_find_byte_not_from` shortcut for `sz_find_byteset` with inverted set
 *  - `sz_rfind_byte_from` shortcut for `sz_rfind_byteset`
 *  - `sz_rfind_byte_not_from` shortcut for `sz_rfind_byteset` with inverted set
 */
#ifndef STRINGZILLA_FIND_H_
#define STRINGZILLA_FIND_H_

#include "types.h"

#include "compare.h" // `sz_equal`

#ifdef __cplusplus
extern "C" {
#endif

#pragma region Core API

/**
 *  @brief  Locates first matching byte in a string. Equivalent to `memchr(haystack, *needle, h_length)` in LibC.
 *
 *  @see X86_64 implementation: https://github.com/lattera/glibc/blob/master/sysdeps/x86_64/memchr.S
 *  @see Aarch64 implementation: https://github.com/lattera/glibc/blob/master/sysdeps/aarch64/memchr.S
 *
 *  @param[in] haystack Haystack - the string to search in.
 *  @param[in] h_length Number of bytes in the haystack.
 *  @param[in] needle Needle - single-byte substring to find.
 *  @return Address of the first match. NULL if not found.
 */
SZ_DYNAMIC sz_cptr_t sz_find_byte(sz_cptr_t haystack, sz_size_t h_length, sz_cptr_t needle);

/**
 *  @brief  Locates last matching byte in a string. Equivalent to `memrchr(haystack, *needle, h_length)` in LibC.
 *
 *  @see X86_64 implementation: https://github.com/lattera/glibc/blob/master/sysdeps/x86_64/memrchr.S
 *  @see Aarch64 implementation: missing
 *
 *  @param[in] haystack Haystack - the string to search in.
 *  @param[in] h_length Number of bytes in the haystack.
 *  @param[in] needle Needle - single-byte substring to find.
 *  @return Address of the last match. NULL if not found.
 */
SZ_DYNAMIC sz_cptr_t sz_rfind_byte(sz_cptr_t haystack, sz_size_t h_length, sz_cptr_t needle);

/** @copydoc sz_find_byte */
SZ_PUBLIC sz_cptr_t sz_find_byte_serial(sz_cptr_t haystack, sz_size_t h_length, sz_cptr_t needle);
/** @copydoc sz_rfind_byte */
SZ_PUBLIC sz_cptr_t sz_rfind_byte_serial(sz_cptr_t haystack, sz_size_t h_length, sz_cptr_t needle);

#if SZ_USE_HASWELL
/** @copydoc sz_find_byte */
SZ_PUBLIC sz_cptr_t sz_find_byte_haswell(sz_cptr_t haystack, sz_size_t h_length, sz_cptr_t needle);
/** @copydoc sz_rfind_byte */
SZ_PUBLIC sz_cptr_t sz_rfind_byte_haswell(sz_cptr_t haystack, sz_size_t h_length, sz_cptr_t needle);
#endif

#if SZ_USE_SKYLAKE
/** @copydoc sz_find_byte */
SZ_PUBLIC sz_cptr_t sz_find_byte_skylake(sz_cptr_t haystack, sz_size_t h_length, sz_cptr_t needle);
/** @copydoc sz_rfind_byte */
SZ_PUBLIC sz_cptr_t sz_rfind_byte_skylake(sz_cptr_t haystack, sz_size_t h_length, sz_cptr_t needle);
#endif

#if SZ_USE_NEON
/** @copydoc sz_find_byte */
SZ_PUBLIC sz_cptr_t sz_find_byte_neon(sz_cptr_t haystack, sz_size_t h_length, sz_cptr_t needle);
/** @copydoc sz_rfind_byte */
SZ_PUBLIC sz_cptr_t sz_rfind_byte_neon(sz_cptr_t haystack, sz_size_t h_length, sz_cptr_t needle);
#endif

/**
 *  @brief  Locates first matching substring.
 *          Equivalent to `memmem(haystack, h_length, needle, n_length)` in LibC.
 *          Similar to `strstr(haystack, needle)` in LibC, but requires known length.
 *
 *  @param[in] haystack Haystack - the string to search in.
 *  @param[in] h_length Number of bytes in the haystack.
 *  @param[in] needle Needle - substring to find.
 *  @param[in] n_length Number of bytes in the needle.
 *  @return Address of the first match.
 */
SZ_DYNAMIC sz_cptr_t sz_find(sz_cptr_t haystack, sz_size_t h_length, sz_cptr_t needle, sz_size_t n_length);

/**
 *  @brief  Locates the last matching substring.
 *
 *  @param[in] haystack Haystack - the string to search in.
 *  @param[in] h_length Number of bytes in the haystack.
 *  @param[in] needle Needle - substring to find.
 *  @param[in] n_length Number of bytes in the needle.
 *  @return Address of the last match.
 */
SZ_DYNAMIC sz_cptr_t sz_rfind(sz_cptr_t haystack, sz_size_t h_length, sz_cptr_t needle, sz_size_t n_length);

/** @copydoc sz_find */
SZ_PUBLIC sz_cptr_t sz_find_serial(sz_cptr_t haystack, sz_size_t h_length, sz_cptr_t needle, sz_size_t n_length);
/** @copydoc sz_rfind */
SZ_PUBLIC sz_cptr_t sz_rfind_serial(sz_cptr_t haystack, sz_size_t h_length, sz_cptr_t needle, sz_size_t n_length);

#if SZ_USE_HASWELL
/** @copydoc sz_find */
SZ_PUBLIC sz_cptr_t sz_find_haswell(sz_cptr_t haystack, sz_size_t h_length, sz_cptr_t needle, sz_size_t n_length);
/** @copydoc sz_rfind */
SZ_PUBLIC sz_cptr_t sz_rfind_haswell(sz_cptr_t haystack, sz_size_t h_length, sz_cptr_t needle, sz_size_t n_length);
#endif

#if SZ_USE_SKYLAKE
/** @copydoc sz_find */
SZ_PUBLIC sz_cptr_t sz_find_skylake(sz_cptr_t haystack, sz_size_t h_length, sz_cptr_t needle, sz_size_t n_length);
/** @copydoc sz_rfind */
SZ_PUBLIC sz_cptr_t sz_rfind_skylake(sz_cptr_t haystack, sz_size_t h_length, sz_cptr_t needle, sz_size_t n_length);
#endif

#if SZ_USE_NEON
/** @copydoc sz_find */
SZ_PUBLIC sz_cptr_t sz_find_neon(sz_cptr_t haystack, sz_size_t h_length, sz_cptr_t needle, sz_size_t n_length);
/** @copydoc sz_rfind */
SZ_PUBLIC sz_cptr_t sz_rfind_neon(sz_cptr_t haystack, sz_size_t h_length, sz_cptr_t needle, sz_size_t n_length);
#endif

/**
 *  @brief  Finds the first character present from the @p set, present in @p text.
 *          Equivalent to `strspn(text, accepted)` and `strcspn(text, rejected)` in LibC.
 *          May have identical implementation and performance to ::sz_rfind_byteset.
 *
 *  Useful for parsing, when we want to skip a set of characters. Examples:
 *  - 6 whitespaces: " \t\n\r\v\f".
 *  - 16 digits forming a float number: "0123456789,.eE+-".
 *  - 5 HTML reserved characters: "\"'&<>", of which "<>" can be useful for parsing.
 *  - 2 JSON string special characters useful to locate the end of the string: "\"\\".
 *
 *  @param[in] text String to be scanned.
 *  @param[in] set Set of relevant characters.
 *  @return Pointer to the first matching character from @p set.
 */
SZ_DYNAMIC sz_cptr_t sz_find_byteset(sz_cptr_t text, sz_size_t length, sz_byteset_t const *set);

/**
 *  @brief  Finds the last character present from the @p set, present in @p text.
 *          Equivalent to `strspn(text, accepted)` and `strcspn(text, rejected)` in LibC.
 *          May have identical implementation and performance to ::sz_find_byteset.
 *
 *  Useful for parsing, when we want to skip a set of characters. Examples:
 *  - 6 whitespaces: " \t\n\r\v\f".
 *  - 16 digits forming a float number: "0123456789,.eE+-".
 *  - 5 HTML reserved characters: "\"'&<>", of which "<>" can be useful for parsing.
 *  - 2 JSON string special characters useful to locate the end of the string: "\"\\".
 *
 *  @param[in] text String to be scanned.
 *  @param[in] set Set of relevant characters.
 *  @return Pointer to the last matching character from @p set.
 */
SZ_DYNAMIC sz_cptr_t sz_rfind_byteset(sz_cptr_t text, sz_size_t length, sz_byteset_t const *set);

/** @copydoc sz_find_byteset */
SZ_PUBLIC sz_cptr_t sz_find_byteset_serial(sz_cptr_t text, sz_size_t length, sz_byteset_t const *set);
/** @copydoc sz_rfind_byteset */
SZ_PUBLIC sz_cptr_t sz_rfind_byteset_serial(sz_cptr_t text, sz_size_t length, sz_byteset_t const *set);

#if SZ_USE_HASWELL
/** @copydoc sz_find_byteset */
SZ_PUBLIC sz_cptr_t sz_find_byteset_haswell(sz_cptr_t haystack, sz_size_t length, sz_byteset_t const *set);
/** @copydoc sz_rfind_byteset */
SZ_PUBLIC sz_cptr_t sz_rfind_byteset_haswell(sz_cptr_t haystack, sz_size_t length, sz_byteset_t const *set);
#endif

#if SZ_USE_ICE
/** @copydoc sz_find_byteset */
SZ_PUBLIC sz_cptr_t sz_find_byteset_ice(sz_cptr_t haystack, sz_size_t length, sz_byteset_t const *set);
/** @copydoc sz_rfind_byteset */
SZ_PUBLIC sz_cptr_t sz_rfind_byteset_ice(sz_cptr_t haystack, sz_size_t length, sz_byteset_t const *set);
#endif

#if SZ_USE_NEON
/** @copydoc sz_find_byteset */
SZ_PUBLIC sz_cptr_t sz_find_byteset_neon(sz_cptr_t haystack, sz_size_t length, sz_byteset_t const *set);
/** @copydoc sz_rfind_byteset */
SZ_PUBLIC sz_cptr_t sz_rfind_byteset_neon(sz_cptr_t haystack, sz_size_t length, sz_byteset_t const *set);
#endif

#pragma endregion // Core API

#pragma region Helper Shortcuts

SZ_PUBLIC sz_cptr_t sz_find_byte_from(sz_cptr_t h, sz_size_t h_length, sz_cptr_t n, sz_size_t n_length) {
    sz_byteset_t set;
    sz_byteset_init(&set);
    for (; n_length; ++n, --n_length) sz_byteset_add(&set, *n);
    return sz_find_byteset(h, h_length, &set);
}

SZ_PUBLIC sz_cptr_t sz_find_byte_not_from(sz_cptr_t h, sz_size_t h_length, sz_cptr_t n, sz_size_t n_length) {
    sz_byteset_t set;
    sz_byteset_init(&set);
    for (; n_length; ++n, --n_length) sz_byteset_add(&set, *n);
    sz_byteset_invert(&set);
    return sz_find_byteset(h, h_length, &set);
}

SZ_PUBLIC sz_cptr_t sz_rfind_byte_from(sz_cptr_t h, sz_size_t h_length, sz_cptr_t n, sz_size_t n_length) {
    sz_byteset_t set;
    sz_byteset_init(&set);
    for (; n_length; ++n, --n_length) sz_byteset_add(&set, *n);
    return sz_rfind_byteset(h, h_length, &set);
}

SZ_PUBLIC sz_cptr_t sz_rfind_byte_not_from(sz_cptr_t h, sz_size_t h_length, sz_cptr_t n, sz_size_t n_length) {
    sz_byteset_t set;
    sz_byteset_init(&set);
    for (; n_length; ++n, --n_length) sz_byteset_add(&set, *n);
    sz_byteset_invert(&set);
    return sz_rfind_byteset(h, h_length, &set);
}

#pragma endregion // Helper Shortcuts

#pragma region Serial Implementation

/**
 *  @brief  Chooses the offsets of the most interesting characters in a search needle.
 *
 *  Search throughput can significantly deteriorate if we are matching the wrong characters.
 *  Say the needle is "aXaYa", and we are comparing the first, second, and last character.
 *  If we use SIMD and compare many offsets at a time, comparing against "a" in every register is a waste.
 *
 *  Similarly, dealing with UTF8 inputs, we know that the lower bits of each character code carry more information.
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
    // Often dealing with UTF8, we will likely benefit from shifting the first and second characters
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

        // Let's begin with the seccond character, as the termination criteria there is more obvious
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
SZ_PUBLIC sz_cptr_t sz_find_byte_serial(sz_cptr_t h, sz_size_t h_length, sz_cptr_t n) {

    if (!h_length) return SZ_NULL_CHAR;
    sz_cptr_t const h_end = h + h_length;

#if !SZ_IS_BIG_ENDIAN_       // Use SWAR only on little-endian platforms for brevity.
#if !SZ_USE_MISALIGNED_LOADS // Process the misaligned head, to void UB on unaligned 64-bit loads.
    for (; ((sz_size_t)h & 7ull) && h < h_end; ++h)
        if (*h == *n) return h;
#endif

    // Broadcast the n into every byte of a 64-bit integer to use SWAR
    // techniques and process eight characters at a time.
    sz_u64_vec_t h_vec, n_vec, match_vec;
    match_vec.u64 = 0;
    n_vec.u64 = (sz_u64_t)n[0] * 0x0101010101010101ull;
    for (; h + 8 <= h_end; h += 8) {
        h_vec.u64 = *(sz_u64_t const *)h;
        match_vec = sz_u64_each_byte_equal_(h_vec, n_vec);
        if (match_vec.u64) return h + sz_u64_ctz(match_vec.u64) / 8;
    }
#endif

    // Handle the misaligned tail.
    for (; h < h_end; ++h)
        if (*h == *n) return h;
    return SZ_NULL_CHAR;
}

/*  Find the last occurrence of a @b single-character needle in an arbitrary length haystack.
 *  This implementation uses hardware-agnostic SWAR technique, to process 8 characters at a time.
 *  Identical to `memrchr(haystack, needle[0], haystack_length)`.
 */
sz_cptr_t sz_rfind_byte_serial(sz_cptr_t h, sz_size_t h_length, sz_cptr_t n) {

    if (!h_length) return SZ_NULL_CHAR;
    sz_cptr_t const h_start = h;

    // Reposition the `h` pointer to the end, as we will be walking backwards.
    h = h + h_length - 1;

#if !SZ_IS_BIG_ENDIAN_       // Use SWAR only on little-endian platforms for brevity.
#if !SZ_USE_MISALIGNED_LOADS // Process the misaligned head, to void UB on unaligned 64-bit loads.
    for (; ((sz_size_t)(h + 1) & 7ull) && h >= h_start; --h)
        if (*h == *n) return h;
#endif

    // Broadcast the n into every byte of a 64-bit integer to use SWAR
    // techniques and process eight characters at a time.
    sz_u64_vec_t h_vec, n_vec, match_vec;
    n_vec.u64 = (sz_u64_t)n[0] * 0x0101010101010101ull;
    for (; h >= h_start + 7; h -= 8) {
        h_vec.u64 = *(sz_u64_t const *)(h - 7);
        match_vec = sz_u64_each_byte_equal_(h_vec, n_vec);
        if (match_vec.u64) return h - sz_u64_clz(match_vec.u64) / 8;
    }
#endif

    for (; h >= h_start; --h)
        if (*h == *n) return h;
    return SZ_NULL_CHAR;
}

/**
 *  @brief  2Byte-level equality comparison between two 64-bit integers.
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
 *  @brief  Find the first occurrence of a @b two-character needle in an arbitrary length haystack.
 *          This implementation uses hardware-agnostic SWAR technique, to process 8 possible offsets at a time.
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
 *  @brief  4Byte-level equality comparison between two 64-bit integers.
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
 *  @brief  Find the first occurrence of a @b four-character needle in an arbitrary length haystack.
 *          This implementation uses hardware-agnostic SWAR technique, to process 8 possible offsets at a time.
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
 *  @brief  3Byte-level equality comparison between two 64-bit integers.
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
 *  @brief  Find the first occurrence of a @b three-character needle in an arbitrary length haystack.
 *          This implementation uses hardware-agnostic SWAR technique, to process 8 possible offsets at a time.
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
 *  @brief  Boyer-Moore-Horspool algorithm for exact matching of patterns up to @b 256-bytes long.
 *          Uses the Raita heuristic to match the first two, the last, and the middle character of the pattern.
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
 *  @brief  Boyer-Moore-Horspool algorithm for @b reverse-order exact matching of patterns up to @b 256-bytes long.
 *          Uses the Raita heuristic to match the first two, the last, and the middle character of the pattern.
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
 *  @brief  Exact substring search helper function, that finds the first occurrence of a prefix of the needle
 *          using a given search function, and then verifies the remaining part of the needle.
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
 *  @brief  Exact reverse-order substring search helper function, that finds the last occurrence of a suffix of the
 *          needle using a given search function, and then verifies the remaining part of the needle.
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

#pragma endregion // Serial Implementation

/*  AVX2 implementation of the string search algorithms for Haswell processors and newer.
 *  Very minimalistic (compared to AVX-512), but still faster than the serial implementation.
 */
#pragma region Haswell Implementation
#if SZ_USE_HASWELL
#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("avx2"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("avx2")
#endif

SZ_PUBLIC sz_cptr_t sz_find_byte_haswell(sz_cptr_t h, sz_size_t h_length, sz_cptr_t n) {
    int mask;
    sz_u256_vec_t h_vec, n_vec;
    n_vec.ymm = _mm256_set1_epi8(n[0]);

    while (h_length >= 32) {
        h_vec.ymm = _mm256_lddqu_si256((__m256i const *)h);
        mask = _mm256_movemask_epi8(_mm256_cmpeq_epi8(h_vec.ymm, n_vec.ymm));
        if (mask) return h + sz_u32_ctz(mask);
        h += 32, h_length -= 32;
    }

    return sz_find_byte_serial(h, h_length, n);
}

SZ_PUBLIC sz_cptr_t sz_rfind_byte_haswell(sz_cptr_t h, sz_size_t h_length, sz_cptr_t n) {
    int mask;
    sz_u256_vec_t h_vec, n_vec;
    n_vec.ymm = _mm256_set1_epi8(n[0]);

    while (h_length >= 32) {
        h_vec.ymm = _mm256_lddqu_si256((__m256i const *)(h + h_length - 32));
        mask = _mm256_movemask_epi8(_mm256_cmpeq_epi8(h_vec.ymm, n_vec.ymm));
        if (mask) return h + h_length - 1 - sz_u32_clz(mask);
        h_length -= 32;
    }

    return sz_rfind_byte_serial(h, h_length, n);
}

SZ_PUBLIC sz_cptr_t sz_find_haswell(sz_cptr_t h, sz_size_t h_length, sz_cptr_t n, sz_size_t n_length) {

    // This almost never fires, but it's better to be safe than sorry.
    if (h_length < n_length || !n_length) return SZ_NULL_CHAR;
    if (n_length == 1) return sz_find_byte_haswell(h, h_length, n);

    // Pick the parts of the needle that are worth comparing.
    sz_size_t offset_first, offset_mid, offset_last;
    sz_locate_needle_anomalies_(n, n_length, &offset_first, &offset_mid, &offset_last);

    // Broadcast those characters into YMM registers.
    int matches;
    sz_u256_vec_t h_first_vec, h_mid_vec, h_last_vec, n_first_vec, n_mid_vec, n_last_vec;
    n_first_vec.ymm = _mm256_set1_epi8(n[offset_first]);
    n_mid_vec.ymm = _mm256_set1_epi8(n[offset_mid]);
    n_last_vec.ymm = _mm256_set1_epi8(n[offset_last]);

    // Scan through the string.
    for (; h_length >= n_length + 32; h += 32, h_length -= 32) {
        h_first_vec.ymm = _mm256_lddqu_si256((__m256i const *)(h + offset_first));
        h_mid_vec.ymm = _mm256_lddqu_si256((__m256i const *)(h + offset_mid));
        h_last_vec.ymm = _mm256_lddqu_si256((__m256i const *)(h + offset_last));
        matches = //
            _mm256_movemask_epi8(_mm256_cmpeq_epi8(h_first_vec.ymm, n_first_vec.ymm)) &
            _mm256_movemask_epi8(_mm256_cmpeq_epi8(h_mid_vec.ymm, n_mid_vec.ymm)) &
            _mm256_movemask_epi8(_mm256_cmpeq_epi8(h_last_vec.ymm, n_last_vec.ymm));
        while (matches) {
            int potential_offset = sz_u32_ctz(matches);
            if (sz_equal_haswell(h + potential_offset, n, n_length)) return h + potential_offset;
            matches &= matches - 1;
        }
    }

    return sz_find_serial(h, h_length, n, n_length);
}

SZ_PUBLIC sz_cptr_t sz_rfind_haswell(sz_cptr_t h, sz_size_t h_length, sz_cptr_t n, sz_size_t n_length) {

    // This almost never fires, but it's better to be safe than sorry.
    if (h_length < n_length || !n_length) return SZ_NULL_CHAR;
    if (n_length == 1) return sz_rfind_byte_haswell(h, h_length, n);

    // Pick the parts of the needle that are worth comparing.
    sz_size_t offset_first, offset_mid, offset_last;
    sz_locate_needle_anomalies_(n, n_length, &offset_first, &offset_mid, &offset_last);

    // Broadcast those characters into YMM registers.
    int matches;
    sz_u256_vec_t h_first_vec, h_mid_vec, h_last_vec, n_first_vec, n_mid_vec, n_last_vec;
    n_first_vec.ymm = _mm256_set1_epi8(n[offset_first]);
    n_mid_vec.ymm = _mm256_set1_epi8(n[offset_mid]);
    n_last_vec.ymm = _mm256_set1_epi8(n[offset_last]);

    // Scan through the string.
    sz_cptr_t h_reversed;
    for (; h_length >= n_length + 32; h_length -= 32) {
        h_reversed = h + h_length - n_length - 32 + 1;
        h_first_vec.ymm = _mm256_lddqu_si256((__m256i const *)(h_reversed + offset_first));
        h_mid_vec.ymm = _mm256_lddqu_si256((__m256i const *)(h_reversed + offset_mid));
        h_last_vec.ymm = _mm256_lddqu_si256((__m256i const *)(h_reversed + offset_last));
        matches = //
            _mm256_movemask_epi8(_mm256_cmpeq_epi8(h_first_vec.ymm, n_first_vec.ymm)) &
            _mm256_movemask_epi8(_mm256_cmpeq_epi8(h_mid_vec.ymm, n_mid_vec.ymm)) &
            _mm256_movemask_epi8(_mm256_cmpeq_epi8(h_last_vec.ymm, n_last_vec.ymm));
        while (matches) {
            int potential_offset = sz_u32_clz(matches);
            if (sz_equal_haswell(h + h_length - n_length - potential_offset, n, n_length))
                return h + h_length - n_length - potential_offset;
            matches &= ~(1 << (31 - potential_offset));
        }
    }

    return sz_rfind_serial(h, h_length, n, n_length);
}

SZ_PUBLIC sz_cptr_t sz_find_byteset_haswell(sz_cptr_t text, sz_size_t length, sz_byteset_t const *filter) {

    // Let's unzip even and odd elements and replicate them into both lanes of the YMM register.
    // That way when we invoke `_mm256_shuffle_epi8` we can use the same mask for both lanes.
    sz_u256_vec_t filter_even_vec, filter_odd_vec;
    for (sz_size_t i = 0; i != 16; ++i)
        filter_even_vec.u8s[i] = filter->_u8s[i * 2], filter_odd_vec.u8s[i] = filter->_u8s[i * 2 + 1];
    filter_even_vec.xmms[1] = filter_even_vec.xmms[0];
    filter_odd_vec.xmms[1] = filter_odd_vec.xmms[0];

    sz_u256_vec_t text_vec;
    sz_u256_vec_t matches_vec;
    sz_u256_vec_t lower_nibbles_vec, higher_nibbles_vec;
    sz_u256_vec_t bitset_even_vec, bitset_odd_vec;
    sz_u256_vec_t bitmask_vec, bitmask_lookup_vec;
    bitmask_lookup_vec.ymm = _mm256_set_epi8(                       //
        -128, 64, 32, 16, 8, 4, 2, 1, -128, 64, 32, 16, 8, 4, 2, 1, //
        -128, 64, 32, 16, 8, 4, 2, 1, -128, 64, 32, 16, 8, 4, 2, 1);

    while (length >= 32) {
        // The following algorithm is a transposed equivalent of the "SIMD-ized check which bytes are in a set"
        // solutions by Wojciech Muła. We populate the bitmask differently and target newer CPUs, so
        // StrinZilla uses a somewhat different approach.
        // http://0x80.pl/articles/simd-byte-lookup.html#alternative-implementation-new
        //
        //      sz_u8_t input = *(sz_u8_t const *)text;
        //      sz_u8_t lo_nibble = input & 0x0f;
        //      sz_u8_t hi_nibble = input >> 4;
        //      sz_u8_t bitset_even = filter_even_vec.u8s[hi_nibble];
        //      sz_u8_t bitset_odd = filter_odd_vec.u8s[hi_nibble];
        //      sz_u8_t bitmask = (1 << (lo_nibble & 0x7));
        //      sz_u8_t bitset = lo_nibble < 8 ? bitset_even : bitset_odd;
        //      if ((bitset & bitmask) != 0) return text;
        //      else { length--, text++; }
        //
        // The nice part about this, loading the strided data is vey easy with Arm NEON,
        // while with x86 CPUs after AVX, shuffles within 256 bits shouldn't be an issue either.
        text_vec.ymm = _mm256_lddqu_si256((__m256i const *)text);
        lower_nibbles_vec.ymm = _mm256_and_si256(text_vec.ymm, _mm256_set1_epi8(0x0f));
        bitmask_vec.ymm = _mm256_shuffle_epi8(bitmask_lookup_vec.ymm, lower_nibbles_vec.ymm);
        //
        // At this point we can validate the `bitmask_vec` contents like this:
        //
        //      for (sz_size_t i = 0; i != 32; ++i) {
        //          sz_u8_t input = *(sz_u8_t const *)(text + i);
        //          sz_u8_t lo_nibble = input & 0x0f;
        //          sz_u8_t bitmask = (1 << (lo_nibble & 0x7));
        //          sz_assert_(bitmask_vec.u8s[i] == bitmask);
        //      }
        //
        // Shift right every byte by 4 bits.
        // There is no `_mm256_srli_epi8` intrinsic, so we have to use `_mm256_srli_epi16`
        // and combine it with a mask to clear the higher bits.
        higher_nibbles_vec.ymm = _mm256_and_si256(_mm256_srli_epi16(text_vec.ymm, 4), _mm256_set1_epi8(0x0f));
        bitset_even_vec.ymm = _mm256_shuffle_epi8(filter_even_vec.ymm, higher_nibbles_vec.ymm);
        bitset_odd_vec.ymm = _mm256_shuffle_epi8(filter_odd_vec.ymm, higher_nibbles_vec.ymm);
        //
        // At this point we can validate the `bitset_even_vec` and `bitset_odd_vec` contents like this:
        //
        //      for (sz_size_t i = 0; i != 32; ++i) {
        //          sz_u8_t input = *(sz_u8_t const *)(text + i);
        //          sz_u8_t const *bitset_ptr = &filter->_u8s[0];
        //          sz_u8_t hi_nibble = input >> 4;
        //          sz_u8_t bitset_even = bitset_ptr[hi_nibble * 2];
        //          sz_u8_t bitset_odd = bitset_ptr[hi_nibble * 2 + 1];
        //          sz_assert_(bitset_even_vec.u8s[i] == bitset_even);
        //          sz_assert_(bitset_odd_vec.u8s[i] == bitset_odd);
        //      }
        //
        __m256i take_first = _mm256_cmpgt_epi8(_mm256_set1_epi8(8), lower_nibbles_vec.ymm);
        bitset_even_vec.ymm = _mm256_blendv_epi8(bitset_odd_vec.ymm, bitset_even_vec.ymm, take_first);

        // It would have been great to have an instruction that tests the bits and then broadcasts
        // the matching bit into all bits in that byte. But we don't have that, so we have to
        // `and`, `cmpeq`, `movemask`, and then invert at the end...
        matches_vec.ymm = _mm256_and_si256(bitset_even_vec.ymm, bitmask_vec.ymm);
        matches_vec.ymm = _mm256_cmpeq_epi8(matches_vec.ymm, _mm256_setzero_si256());
        int matches_mask = ~_mm256_movemask_epi8(matches_vec.ymm);
        if (matches_mask) {
            int offset = sz_u32_ctz(matches_mask);
            return text + offset;
        }
        else { text += 32, length -= 32; }
    }

    return sz_find_byteset_serial(text, length, filter);
}

SZ_PUBLIC sz_cptr_t sz_rfind_byteset_haswell(sz_cptr_t text, sz_size_t length, sz_byteset_t const *filter) {
    return sz_rfind_byteset_serial(text, length, filter);
}

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#endif            // SZ_USE_HASWELL
#pragma endregion // Haswell Implementation

/*  AVX512 implementation of the string search algorithms for Skylake and newer CPUs.
 *  Includes extensions: F, CD, ER, PF, VL, DQ, BW.
 *
 *  This is the "starting level" for the advanced algorithms using K-mask registers on x86.
 */
#pragma region Skylake Implementation
#if SZ_USE_SKYLAKE
#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("avx,avx512f,avx512vl,avx512bw,bmi,bmi2"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("avx", "avx512f", "avx512vl", "avx512bw", "bmi", "bmi2")
#endif

SZ_PUBLIC sz_cptr_t sz_find_byte_skylake(sz_cptr_t h, sz_size_t h_length, sz_cptr_t n) {
    __mmask64 mask;
    sz_u512_vec_t h_vec, n_vec;
    n_vec.zmm = _mm512_set1_epi8(n[0]);

    while (h_length >= 64) {
        h_vec.zmm = _mm512_loadu_si512(h);
        mask = _mm512_cmpeq_epi8_mask(h_vec.zmm, n_vec.zmm);
        if (mask) return h + sz_u64_ctz(mask);
        h += 64, h_length -= 64;
    }

    if (h_length) {
        mask = sz_u64_mask_until_(h_length);
        h_vec.zmm = _mm512_maskz_loadu_epi8(mask, h);
        // Reuse the same `mask` variable to find the bit that doesn't match
        mask = _mm512_mask_cmpeq_epu8_mask(mask, h_vec.zmm, n_vec.zmm);
        if (mask) return h + sz_u64_ctz(mask);
    }

    return SZ_NULL_CHAR;
}

SZ_PUBLIC sz_cptr_t sz_find_skylake(sz_cptr_t h, sz_size_t h_length, sz_cptr_t n, sz_size_t n_length) {

    // This almost never fires, but it's better to be safe than sorry.
    if (h_length < n_length || !n_length) return SZ_NULL_CHAR;
    if (n_length == 1) return sz_find_byte_skylake(h, h_length, n);

    // Pick the parts of the needle that are worth comparing.
    sz_size_t offset_first, offset_mid, offset_last;
    sz_locate_needle_anomalies_(n, n_length, &offset_first, &offset_mid, &offset_last);

    // Broadcast those characters into ZMM registers.
    __mmask64 matches;
    __mmask64 mask;
    sz_u512_vec_t h_first_vec, h_mid_vec, h_last_vec, n_first_vec, n_mid_vec, n_last_vec;
    n_first_vec.zmm = _mm512_set1_epi8(n[offset_first]);
    n_mid_vec.zmm = _mm512_set1_epi8(n[offset_mid]);
    n_last_vec.zmm = _mm512_set1_epi8(n[offset_last]);

    // Scan through the string.
    // We have several optimized versions of the algorithm for shorter strings,
    // but they all mimic the default case for unbounded length needles
    if (n_length >= 64) {
        for (; h_length >= n_length + 64; h += 64, h_length -= 64) {
            h_first_vec.zmm = _mm512_loadu_si512(h + offset_first);
            h_mid_vec.zmm = _mm512_loadu_si512(h + offset_mid);
            h_last_vec.zmm = _mm512_loadu_si512(h + offset_last);
            matches = _kand_mask64( //
                _kand_mask64(       // Intersect the masks
                    _mm512_cmpeq_epi8_mask(h_first_vec.zmm, n_first_vec.zmm),
                    _mm512_cmpeq_epi8_mask(h_mid_vec.zmm, n_mid_vec.zmm)),
                _mm512_cmpeq_epi8_mask(h_last_vec.zmm, n_last_vec.zmm));
            while (matches) {
                int potential_offset = sz_u64_ctz(matches);
                if (sz_equal_skylake(h + potential_offset, n, n_length)) return h + potential_offset;
                matches &= matches - 1;
            }

            // TODO: If the last character contains a bad byte, we can reposition the start of the next iteration.
            // This will be very helpful for very long needles.
        }
    }
    // If there are only 2 or 3 characters in the needle, we don't even need the nested loop.
    else if (n_length <= 3) {
        for (; h_length >= n_length + 64; h += 64, h_length -= 64) {
            h_first_vec.zmm = _mm512_loadu_si512(h + offset_first);
            h_mid_vec.zmm = _mm512_loadu_si512(h + offset_mid);
            h_last_vec.zmm = _mm512_loadu_si512(h + offset_last);
            matches = _kand_mask64( //
                _kand_mask64(       // Intersect the masks
                    _mm512_cmpeq_epi8_mask(h_first_vec.zmm, n_first_vec.zmm),
                    _mm512_cmpeq_epi8_mask(h_mid_vec.zmm, n_mid_vec.zmm)),
                _mm512_cmpeq_epi8_mask(h_last_vec.zmm, n_last_vec.zmm));
            if (matches) return h + sz_u64_ctz(matches);
        }
    }
    // If the needle is smaller than the size of the ZMM register, we can use masked comparisons
    // to avoid the the inner-most nested loop and compare the entire needle against a haystack
    // slice in 3 CPU cycles.
    else {
        __mmask64 n_mask = sz_u64_mask_until_(n_length);
        sz_u512_vec_t n_full_vec, h_full_vec;
        n_full_vec.zmm = _mm512_maskz_loadu_epi8(n_mask, n);
        for (; h_length >= n_length + 64; h += 64, h_length -= 64) {
            h_first_vec.zmm = _mm512_loadu_si512(h + offset_first);
            h_mid_vec.zmm = _mm512_loadu_si512(h + offset_mid);
            h_last_vec.zmm = _mm512_loadu_si512(h + offset_last);
            matches = _kand_mask64( //
                _kand_mask64(       // Intersect the masks
                    _mm512_cmpeq_epi8_mask(h_first_vec.zmm, n_first_vec.zmm),
                    _mm512_cmpeq_epi8_mask(h_mid_vec.zmm, n_mid_vec.zmm)),
                _mm512_cmpeq_epi8_mask(h_last_vec.zmm, n_last_vec.zmm));
            while (matches) {
                int potential_offset = sz_u64_ctz(matches);
                h_full_vec.zmm = _mm512_maskz_loadu_epi8(n_mask, h + potential_offset);
                if (_mm512_mask_cmpneq_epi8_mask(n_mask, h_full_vec.zmm, n_full_vec.zmm) == 0)
                    return h + potential_offset;
                matches &= matches - 1;
            }
        }
    }

    // The "tail" of the function uses masked loads to process the remaining bytes.
    {
        mask = sz_u64_mask_until_(h_length - n_length + 1);
        h_first_vec.zmm = _mm512_maskz_loadu_epi8(mask, h + offset_first);
        h_mid_vec.zmm = _mm512_maskz_loadu_epi8(mask, h + offset_mid);
        h_last_vec.zmm = _mm512_maskz_loadu_epi8(mask, h + offset_last);
        matches = _kand_mask64( //
            _kand_mask64(       // Intersect the masks
                _mm512_cmpeq_epi8_mask(h_first_vec.zmm, n_first_vec.zmm),
                _mm512_cmpeq_epi8_mask(h_mid_vec.zmm, n_mid_vec.zmm)),
            _mm512_cmpeq_epi8_mask(h_last_vec.zmm, n_last_vec.zmm));
        while (matches) {
            int potential_offset = sz_u64_ctz(matches);
            if (n_length <= 3 || sz_equal_skylake(h + potential_offset, n, n_length)) return h + potential_offset;
            matches &= matches - 1;
        }
    }
    return SZ_NULL_CHAR;
}

SZ_PUBLIC sz_cptr_t sz_rfind_byte_skylake(sz_cptr_t h, sz_size_t h_length, sz_cptr_t n) {
    __mmask64 mask;
    sz_u512_vec_t h_vec, n_vec;
    n_vec.zmm = _mm512_set1_epi8(n[0]);

    while (h_length >= 64) {
        h_vec.zmm = _mm512_loadu_si512(h + h_length - 64);
        mask = _mm512_cmpeq_epi8_mask(h_vec.zmm, n_vec.zmm);
        if (mask) return h + h_length - 1 - sz_u64_clz(mask);
        h_length -= 64;
    }

    if (h_length) {
        mask = sz_u64_mask_until_(h_length);
        h_vec.zmm = _mm512_maskz_loadu_epi8(mask, h);
        // Reuse the same `mask` variable to find the bit that doesn't match
        mask = _mm512_mask_cmpeq_epu8_mask(mask, h_vec.zmm, n_vec.zmm);
        if (mask) return h + 64 - sz_u64_clz(mask) - 1;
    }

    return SZ_NULL_CHAR;
}

SZ_PUBLIC sz_cptr_t sz_rfind_skylake(sz_cptr_t h, sz_size_t h_length, sz_cptr_t n, sz_size_t n_length) {

    // This almost never fires, but it's better to be safe than sorry.
    if (h_length < n_length || !n_length) return SZ_NULL_CHAR;
    if (n_length == 1) return sz_rfind_byte_skylake(h, h_length, n);

    // Pick the parts of the needle that are worth comparing.
    sz_size_t offset_first, offset_mid, offset_last;
    sz_locate_needle_anomalies_(n, n_length, &offset_first, &offset_mid, &offset_last);

    // Broadcast those characters into ZMM registers.
    __mmask64 mask;
    __mmask64 matches;
    sz_u512_vec_t h_first_vec, h_mid_vec, h_last_vec, n_first_vec, n_mid_vec, n_last_vec;
    n_first_vec.zmm = _mm512_set1_epi8(n[offset_first]);
    n_mid_vec.zmm = _mm512_set1_epi8(n[offset_mid]);
    n_last_vec.zmm = _mm512_set1_epi8(n[offset_last]);

    // Scan through the string.
    sz_cptr_t h_reversed;
    for (; h_length >= n_length + 64; h_length -= 64) {
        h_reversed = h + h_length - n_length - 64 + 1;
        h_first_vec.zmm = _mm512_loadu_si512(h_reversed + offset_first);
        h_mid_vec.zmm = _mm512_loadu_si512(h_reversed + offset_mid);
        h_last_vec.zmm = _mm512_loadu_si512(h_reversed + offset_last);
        matches = _kand_mask64( //
            _kand_mask64(       // Intersect the masks
                _mm512_cmpeq_epi8_mask(h_first_vec.zmm, n_first_vec.zmm),
                _mm512_cmpeq_epi8_mask(h_mid_vec.zmm, n_mid_vec.zmm)),
            _mm512_cmpeq_epi8_mask(h_last_vec.zmm, n_last_vec.zmm));
        while (matches) {
            int potential_offset = sz_u64_clz(matches);
            if (n_length <= 3 || sz_equal_skylake(h + h_length - n_length - potential_offset, n, n_length))
                return h + h_length - n_length - potential_offset;
            sz_assert_((matches & ((sz_u64_t)1 << (63 - potential_offset))) != 0 &&
                       "The bit must be set before we squash it");
            matches &= ~((sz_u64_t)1 << (63 - potential_offset));
        }
    }

    // The "tail" of the function uses masked loads to process the remaining bytes.
    {
        mask = sz_u64_mask_until_(h_length - n_length + 1);
        h_first_vec.zmm = _mm512_maskz_loadu_epi8(mask, h + offset_first);
        h_mid_vec.zmm = _mm512_maskz_loadu_epi8(mask, h + offset_mid);
        h_last_vec.zmm = _mm512_maskz_loadu_epi8(mask, h + offset_last);
        matches = _kand_mask64( //
            _kand_mask64(       // Intersect the masks
                _mm512_cmpeq_epi8_mask(h_first_vec.zmm, n_first_vec.zmm),
                _mm512_cmpeq_epi8_mask(h_mid_vec.zmm, n_mid_vec.zmm)),
            _mm512_cmpeq_epi8_mask(h_last_vec.zmm, n_last_vec.zmm));
        while (matches) {
            int potential_offset = sz_u64_clz(matches);
            if (n_length <= 3 || sz_equal_skylake(h + 64 - potential_offset - 1, n, n_length))
                return h + 64 - potential_offset - 1;
            sz_assert_((matches & ((sz_u64_t)1 << (63 - potential_offset))) != 0 &&
                       "The bit must be set before we squash it");
            matches &= ~((sz_u64_t)1 << (63 - potential_offset));
        }
    }

    return SZ_NULL_CHAR;
}

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#endif            // SZ_USE_SKYLAKE
#pragma endregion // Skylake Implementation

/*  AVX512 implementation of the string search algorithms for Ice Lake and newer CPUs.
 *  Includes extensions:
 *      - 2017 Skylake: F, CD, ER, PF, VL, DQ, BW,
 *      - 2018 CannonLake: IFMA, VBMI,
 *      - 2019 Ice Lake: VPOPCNTDQ, VNNI, VBMI2, BITALG, GFNI, VPCLMULQDQ, VAES.
 *
 *  We are going to use VBMI2 for `_mm256_maskz_compress_epi8`.
 */
#pragma region Ice Lake Implementation
#if SZ_USE_ICE
#if defined(__clang__)
#pragma clang attribute push(                                                                          \
    __attribute__((target("avx,avx512f,avx512vl,avx512bw,avx512dq,avx512vbmi,avx512vbmi2,bmi,bmi2"))), \
    apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("avx", "avx512f", "avx512vl", "avx512bw", "avx512dq", "avx512vbmi", "avx512vbmi2", "bmi", "bmi2")
#endif

SZ_PUBLIC sz_cptr_t sz_find_byteset_ice(sz_cptr_t text, sz_size_t length, sz_byteset_t const *filter) {

    // Before initializing the AVX-512 vectors, we may want to run the sequential code for the first few bytes.
    // In practice, that only hurts, even when we have matches every 5-ish bytes.
    //
    //      if (length < SZ_SWAR_THRESHOLD) return sz_find_byteset_serial(text, length, filter);
    //      sz_cptr_t early_result = sz_find_byteset_serial(text, SZ_SWAR_THRESHOLD, filter);
    //      if (early_result) return early_result;
    //      text += SZ_SWAR_THRESHOLD;
    //      length -= SZ_SWAR_THRESHOLD;
    //
    // Let's unzip even and odd elements and replicate them into both lanes of the YMM register.
    // That way when we invoke `_mm512_shuffle_epi8` we can use the same mask for both lanes.
    sz_u512_vec_t filter_even_vec, filter_odd_vec;
    __m256i filter_ymm = _mm256_lddqu_si256((__m256i const *)filter);
    // There are a few way to initialize filters without having native strided loads.
    // In the chronological order of experiments:
    // - serial code initializing 128 bytes of odd and even mask
    // - using several shuffles
    // - using `_mm512_permutexvar_epi8`
    // - using `_mm512_broadcast_i32x4(_mm256_castsi256_si128(_mm256_maskz_compress_epi8(0x55555555, filter_ymm)))`
    //   and `_mm512_broadcast_i32x4(_mm256_castsi256_si128(_mm256_maskz_compress_epi8(0xaaaaaaaa, filter_ymm)))`
    filter_even_vec.zmm = _mm512_broadcast_i32x4(_mm256_castsi256_si128( // broadcast __m128i to __m512i
        _mm256_maskz_compress_epi8(0x55555555, filter_ymm)));
    filter_odd_vec.zmm = _mm512_broadcast_i32x4(_mm256_castsi256_si128( // broadcast __m128i to __m512i
        _mm256_maskz_compress_epi8(0xaaaaaaaa, filter_ymm)));
    // After the unzipping operation, we can validate the contents of the vectors like this:
    //
    //      for (sz_size_t i = 0; i != 16; ++i) {
    //          sz_assert_(filter_even_vec.u8s[i] == filter->_u8s[i * 2]);
    //          sz_assert_(filter_odd_vec.u8s[i] == filter->_u8s[i * 2 + 1]);
    //          sz_assert_(filter_even_vec.u8s[i + 16] == filter->_u8s[i * 2]);
    //          sz_assert_(filter_odd_vec.u8s[i + 16] == filter->_u8s[i * 2 + 1]);
    //          sz_assert_(filter_even_vec.u8s[i + 32] == filter->_u8s[i * 2]);
    //          sz_assert_(filter_odd_vec.u8s[i + 32] == filter->_u8s[i * 2 + 1]);
    //          sz_assert_(filter_even_vec.u8s[i + 48] == filter->_u8s[i * 2]);
    //          sz_assert_(filter_odd_vec.u8s[i + 48] == filter->_u8s[i * 2 + 1]);
    //      }
    //
    sz_u512_vec_t text_vec;
    sz_u512_vec_t lower_nibbles_vec, higher_nibbles_vec;
    sz_u512_vec_t bitset_even_vec, bitset_odd_vec;
    sz_u512_vec_t bitmask_vec, bitmask_lookup_vec;
    bitmask_lookup_vec.zmm = _mm512_set_epi8(                       //
        -128, 64, 32, 16, 8, 4, 2, 1, -128, 64, 32, 16, 8, 4, 2, 1, //
        -128, 64, 32, 16, 8, 4, 2, 1, -128, 64, 32, 16, 8, 4, 2, 1, //
        -128, 64, 32, 16, 8, 4, 2, 1, -128, 64, 32, 16, 8, 4, 2, 1, //
        -128, 64, 32, 16, 8, 4, 2, 1, -128, 64, 32, 16, 8, 4, 2, 1);

    while (length) {
        // The following algorithm is a transposed equivalent of the "SIMDized check which bytes are in a set"
        // solutions by Wojciech Muła. We populate the bitmask differently and target newer CPUs, so
        // StrinZilla uses a somewhat different approach.
        // http://0x80.pl/articles/simd-byte-lookup.html#alternative-implementation-new
        //
        //      sz_u8_t input = *(sz_u8_t const *)text;
        //      sz_u8_t lo_nibble = input & 0x0f;
        //      sz_u8_t hi_nibble = input >> 4;
        //      sz_u8_t bitset_even = filter_even_vec.u8s[hi_nibble];
        //      sz_u8_t bitset_odd = filter_odd_vec.u8s[hi_nibble];
        //      sz_u8_t bitmask = (1 << (lo_nibble & 0x7));
        //      sz_u8_t bitset = lo_nibble < 8 ? bitset_even : bitset_odd;
        //      if ((bitset & bitmask) != 0) return text;
        //      else { length--, text++; }
        //
        // The nice part about this, loading the strided data is vey easy with Arm NEON,
        // while with x86 CPUs after AVX, shuffles within 256 bits shouldn't be an issue either.
        sz_size_t load_length = sz_min_of_two(length, 64);
        __mmask64 load_mask = sz_u64_mask_until_(load_length);
        text_vec.zmm = _mm512_maskz_loadu_epi8(load_mask, text);
        lower_nibbles_vec.zmm = _mm512_and_si512(text_vec.zmm, _mm512_set1_epi8(0x0f));
        bitmask_vec.zmm = _mm512_shuffle_epi8(bitmask_lookup_vec.zmm, lower_nibbles_vec.zmm);
        //
        // At this point we can validate the `bitmask_vec` contents like this:
        //
        //      for (sz_size_t i = 0; i != load_length; ++i) {
        //          sz_u8_t input = *(sz_u8_t const *)(text + i);
        //          sz_u8_t lo_nibble = input & 0x0f;
        //          sz_u8_t bitmask = (1 << (lo_nibble & 0x7));
        //          sz_assert_(bitmask_vec.u8s[i] == bitmask);
        //      }
        //
        // Shift right every byte by 4 bits.
        // There is no `_mm512_srli_epi8` intrinsic, so we have to use `_mm512_srli_epi16`
        // and combine it with a mask to clear the higher bits.
        higher_nibbles_vec.zmm = _mm512_and_si512(_mm512_srli_epi16(text_vec.zmm, 4), _mm512_set1_epi8(0x0f));
        bitset_even_vec.zmm = _mm512_shuffle_epi8(filter_even_vec.zmm, higher_nibbles_vec.zmm);
        bitset_odd_vec.zmm = _mm512_shuffle_epi8(filter_odd_vec.zmm, higher_nibbles_vec.zmm);
        //
        // At this point we can validate the `bitset_even_vec` and `bitset_odd_vec` contents like this:
        //
        //      for (sz_size_t i = 0; i != load_length; ++i) {
        //          sz_u8_t input = *(sz_u8_t const *)(text + i);
        //          sz_u8_t const *bitset_ptr = &filter->_u8s[0];
        //          sz_u8_t hi_nibble = input >> 4;
        //          sz_u8_t bitset_even = bitset_ptr[hi_nibble * 2];
        //          sz_u8_t bitset_odd = bitset_ptr[hi_nibble * 2 + 1];
        //          sz_assert_(bitset_even_vec.u8s[i] == bitset_even);
        //          sz_assert_(bitset_odd_vec.u8s[i] == bitset_odd);
        //      }
        //
        // TODO: Is this a good place for ternary logic?
        __mmask64 take_first = _mm512_cmplt_epi8_mask(lower_nibbles_vec.zmm, _mm512_set1_epi8(8));
        bitset_even_vec.zmm = _mm512_mask_blend_epi8(take_first, bitset_odd_vec.zmm, bitset_even_vec.zmm);
        __mmask64 matches_mask = _mm512_mask_test_epi8_mask(load_mask, bitset_even_vec.zmm, bitmask_vec.zmm);
        if (matches_mask) {
            int offset = sz_u64_ctz(matches_mask);
            return text + offset;
        }
        else { text += load_length, length -= load_length; }
    }

    return SZ_NULL_CHAR;
}

SZ_PUBLIC sz_cptr_t sz_rfind_byteset_ice(sz_cptr_t text, sz_size_t length, sz_byteset_t const *filter) {
    return sz_rfind_byteset_serial(text, length, filter);
}

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#endif            // SZ_USE_ICE
#pragma endregion // Ice Lake Implementation

/*  Implementation of the string search algorithms using the Arm NEON instruction set, available on 64-bit
 *  Arm processors. Covers billions of mobile CPUs worldwide, including Apple's A-series, and Qualcomm's Snapdragon.
 */
#pragma region NEON Implementation
#if SZ_USE_NEON
#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("arch=armv8.2-a+simd"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("arch=armv8.2-a+simd")
#endif

SZ_INTERNAL sz_u64_t sz_vreinterpretq_u8_u4_(uint8x16_t vec) {
    // Use `vshrn` to produce a bitmask, similar to `movemask` in SSE.
    // https://community.arm.com/arm-community-blogs/b/infrastructure-solutions-blog/posts/porting-x86-vector-bitmask-optimizations-to-arm-neon
    return vget_lane_u64(vreinterpret_u64_u8(vshrn_n_u16(vreinterpretq_u16_u8(vec), 4)), 0) & 0x8888888888888888ull;
}

SZ_PUBLIC sz_cptr_t sz_find_byte_neon(sz_cptr_t h, sz_size_t h_length, sz_cptr_t n) {
    sz_u64_t matches;
    sz_u128_vec_t h_vec, n_vec, matches_vec;
    n_vec.u8x16 = vld1q_dup_u8((sz_u8_t const *)n);

    while (h_length >= 16) {
        h_vec.u8x16 = vld1q_u8((sz_u8_t const *)h);
        matches_vec.u8x16 = vceqq_u8(h_vec.u8x16, n_vec.u8x16);
        // In Arm NEON we don't have a `movemask` to combine it with `ctz` and get the offset of the match.
        // But assuming the `vmaxvq` is cheap, we can use it to find the first match, by blending (bitwise selecting)
        // the vector with a relative offsets array.
        matches = sz_vreinterpretq_u8_u4_(matches_vec.u8x16);
        if (matches) return h + sz_u64_ctz(matches) / 4;

        h += 16, h_length -= 16;
    }

    return sz_find_byte_serial(h, h_length, n);
}

SZ_PUBLIC sz_cptr_t sz_rfind_byte_neon(sz_cptr_t h, sz_size_t h_length, sz_cptr_t n) {
    sz_u64_t matches;
    sz_u128_vec_t h_vec, n_vec, matches_vec;
    n_vec.u8x16 = vld1q_dup_u8((sz_u8_t const *)n);

    while (h_length >= 16) {
        h_vec.u8x16 = vld1q_u8((sz_u8_t const *)h + h_length - 16);
        matches_vec.u8x16 = vceqq_u8(h_vec.u8x16, n_vec.u8x16);
        matches = sz_vreinterpretq_u8_u4_(matches_vec.u8x16);
        if (matches) return h + h_length - 1 - sz_u64_clz(matches) / 4;
        h_length -= 16;
    }

    return sz_rfind_byte_serial(h, h_length, n);
}

SZ_PUBLIC sz_u64_t sz_find_byteset_neon_register_( //
    sz_u128_vec_t h_vec, uint8x16_t set_top_vec_u8x16, uint8x16_t set_bottom_vec_u8x16) {

    // Once we've read the characters in the haystack, we want to
    // compare them against our bitset. The serial version of that code
    // would look like: `(set_->_u8s[c >> 3] & (1u << (c & 7u))) != 0`.
    uint8x16_t byte_index_vec = vshrq_n_u8(h_vec.u8x16, 3);
    uint8x16_t byte_mask_vec = vshlq_u8(vdupq_n_u8(1), vreinterpretq_s8_u8(vandq_u8(h_vec.u8x16, vdupq_n_u8(7))));
    uint8x16_t matches_top_vec = vqtbl1q_u8(set_top_vec_u8x16, byte_index_vec);
    // The table lookup instruction in NEON replies to out-of-bound requests with zeros.
    // The values in `byte_index_vec` all fall in [0; 32). So for values under 16, substracting 16 will underflow
    // and map into interval [240, 256). Meaning that those will be populated with zeros and we can safely
    // merge `matches_top_vec` and `matches_bottom_vec` with a bitwise OR.
    uint8x16_t matches_bottom_vec = vqtbl1q_u8(set_bottom_vec_u8x16, vsubq_u8(byte_index_vec, vdupq_n_u8(16)));
    uint8x16_t matches_vec = vorrq_u8(matches_top_vec, matches_bottom_vec);
    // Istead of pure `vandq_u8`, we can immediately broadcast a match presence across each 8-bit word.
    matches_vec = vtstq_u8(matches_vec, byte_mask_vec);
    return sz_vreinterpretq_u8_u4_(matches_vec);
}

SZ_PUBLIC sz_cptr_t sz_find_neon(sz_cptr_t h, sz_size_t h_length, sz_cptr_t n, sz_size_t n_length) {

    // This almost never fires, but it's better to be safe than sorry.
    if (h_length < n_length || !n_length) return SZ_NULL_CHAR;
    if (n_length == 1) return sz_find_byte_neon(h, h_length, n);

    // Scan through the string.
    // Assuming how tiny the Arm NEON registers are, we should avoid internal branches at all costs.
    // That's why, for smaller needles, we use different loops.
    if (n_length == 2) {
        // Broadcast needle characters into SIMD registers.
        sz_u64_t matches;
        sz_u128_vec_t h_first_vec, h_last_vec, n_first_vec, n_last_vec, matches_vec;
        // Dealing with 16-bit values, we can load 2 registers at a time and compare 31 possible offsets
        // in a single loop iteration.
        n_first_vec.u8x16 = vld1q_dup_u8((sz_u8_t const *)&n[0]);
        n_last_vec.u8x16 = vld1q_dup_u8((sz_u8_t const *)&n[1]);
        for (; h_length >= 17; h += 16, h_length -= 16) {
            h_first_vec.u8x16 = vld1q_u8((sz_u8_t const *)(h + 0));
            h_last_vec.u8x16 = vld1q_u8((sz_u8_t const *)(h + 1));
            matches_vec.u8x16 =
                vandq_u8(vceqq_u8(h_first_vec.u8x16, n_first_vec.u8x16), vceqq_u8(h_last_vec.u8x16, n_last_vec.u8x16));
            matches = sz_vreinterpretq_u8_u4_(matches_vec.u8x16);
            if (matches) return h + sz_u64_ctz(matches) / 4;
        }
    }
    else if (n_length == 3) {
        // Broadcast needle characters into SIMD registers.
        sz_u64_t matches;
        sz_u128_vec_t h_first_vec, h_mid_vec, h_last_vec, n_first_vec, n_mid_vec, n_last_vec, matches_vec;
        // Comparing 24-bit values is a bumer. Being lazy, I went with the same approach
        // as when searching for string over 4 characters long. I only avoid the last comparison.
        n_first_vec.u8x16 = vld1q_dup_u8((sz_u8_t const *)&n[0]);
        n_mid_vec.u8x16 = vld1q_dup_u8((sz_u8_t const *)&n[1]);
        n_last_vec.u8x16 = vld1q_dup_u8((sz_u8_t const *)&n[2]);
        for (; h_length >= 18; h += 16, h_length -= 16) {
            h_first_vec.u8x16 = vld1q_u8((sz_u8_t const *)(h + 0));
            h_mid_vec.u8x16 = vld1q_u8((sz_u8_t const *)(h + 1));
            h_last_vec.u8x16 = vld1q_u8((sz_u8_t const *)(h + 2));
            matches_vec.u8x16 = vandq_u8(                           //
                vandq_u8(                                           //
                    vceqq_u8(h_first_vec.u8x16, n_first_vec.u8x16), //
                    vceqq_u8(h_mid_vec.u8x16, n_mid_vec.u8x16)),
                vceqq_u8(h_last_vec.u8x16, n_last_vec.u8x16));
            matches = sz_vreinterpretq_u8_u4_(matches_vec.u8x16);
            if (matches) return h + sz_u64_ctz(matches) / 4;
        }
    }
    else {
        // Pick the parts of the needle that are worth comparing.
        sz_size_t offset_first, offset_mid, offset_last;
        sz_locate_needle_anomalies_(n, n_length, &offset_first, &offset_mid, &offset_last);
        // Broadcast those characters into SIMD registers.
        sz_u64_t matches;
        sz_u128_vec_t h_first_vec, h_mid_vec, h_last_vec, n_first_vec, n_mid_vec, n_last_vec, matches_vec;
        n_first_vec.u8x16 = vld1q_dup_u8((sz_u8_t const *)&n[offset_first]);
        n_mid_vec.u8x16 = vld1q_dup_u8((sz_u8_t const *)&n[offset_mid]);
        n_last_vec.u8x16 = vld1q_dup_u8((sz_u8_t const *)&n[offset_last]);
        // Walk through the string.
        for (; h_length >= n_length + 16; h += 16, h_length -= 16) {
            h_first_vec.u8x16 = vld1q_u8((sz_u8_t const *)(h + offset_first));
            h_mid_vec.u8x16 = vld1q_u8((sz_u8_t const *)(h + offset_mid));
            h_last_vec.u8x16 = vld1q_u8((sz_u8_t const *)(h + offset_last));
            matches_vec.u8x16 = vandq_u8(                           //
                vandq_u8(                                           //
                    vceqq_u8(h_first_vec.u8x16, n_first_vec.u8x16), //
                    vceqq_u8(h_mid_vec.u8x16, n_mid_vec.u8x16)),
                vceqq_u8(h_last_vec.u8x16, n_last_vec.u8x16));
            matches = sz_vreinterpretq_u8_u4_(matches_vec.u8x16);
            while (matches) {
                int potential_offset = sz_u64_ctz(matches) / 4;
                if (sz_equal_neon(h + potential_offset, n, n_length)) return h + potential_offset;
                matches &= matches - 1;
            }
        }
    }

    return sz_find_serial(h, h_length, n, n_length);
}

SZ_PUBLIC sz_cptr_t sz_rfind_neon(sz_cptr_t h, sz_size_t h_length, sz_cptr_t n, sz_size_t n_length) {

    // This almost never fires, but it's better to be safe than sorry.
    if (h_length < n_length || !n_length) return SZ_NULL_CHAR;
    if (n_length == 1) return sz_rfind_byte_neon(h, h_length, n);

    // Pick the parts of the needle that are worth comparing.
    sz_size_t offset_first, offset_mid, offset_last;
    sz_locate_needle_anomalies_(n, n_length, &offset_first, &offset_mid, &offset_last);

    // Will contain 4 bits per character.
    sz_u64_t matches;
    sz_u128_vec_t h_first_vec, h_mid_vec, h_last_vec, n_first_vec, n_mid_vec, n_last_vec, matches_vec;
    n_first_vec.u8x16 = vld1q_dup_u8((sz_u8_t const *)&n[offset_first]);
    n_mid_vec.u8x16 = vld1q_dup_u8((sz_u8_t const *)&n[offset_mid]);
    n_last_vec.u8x16 = vld1q_dup_u8((sz_u8_t const *)&n[offset_last]);

    sz_cptr_t h_reversed;
    for (; h_length >= n_length + 16; h_length -= 16) {
        h_reversed = h + h_length - n_length - 16 + 1;
        h_first_vec.u8x16 = vld1q_u8((sz_u8_t const *)(h_reversed + offset_first));
        h_mid_vec.u8x16 = vld1q_u8((sz_u8_t const *)(h_reversed + offset_mid));
        h_last_vec.u8x16 = vld1q_u8((sz_u8_t const *)(h_reversed + offset_last));
        matches_vec.u8x16 = vandq_u8(                           //
            vandq_u8(                                           //
                vceqq_u8(h_first_vec.u8x16, n_first_vec.u8x16), //
                vceqq_u8(h_mid_vec.u8x16, n_mid_vec.u8x16)),
            vceqq_u8(h_last_vec.u8x16, n_last_vec.u8x16));
        matches = sz_vreinterpretq_u8_u4_(matches_vec.u8x16);
        while (matches) {
            int potential_offset = sz_u64_clz(matches) / 4;
            if (sz_equal_neon(h + h_length - n_length - potential_offset, n, n_length))
                return h + h_length - n_length - potential_offset;
            sz_assert_((matches & (1ull << (63 - potential_offset * 4))) != 0 &&
                       "The bit must be set before we squash it");
            matches &= ~(1ull << (63 - potential_offset * 4));
        }
    }

    return sz_rfind_serial(h, h_length, n, n_length);
}

SZ_PUBLIC sz_cptr_t sz_find_byteset_neon(sz_cptr_t h, sz_size_t h_length, sz_byteset_t const *set) {
    sz_u64_t matches;
    sz_u128_vec_t h_vec;
    uint8x16_t set_top_vec_u8x16 = vld1q_u8(&set->_u8s[0]);
    uint8x16_t set_bottom_vec_u8x16 = vld1q_u8(&set->_u8s[16]);

    for (; h_length >= 16; h += 16, h_length -= 16) {
        h_vec.u8x16 = vld1q_u8((sz_u8_t const *)(h));
        matches = sz_find_byteset_neon_register_(h_vec, set_top_vec_u8x16, set_bottom_vec_u8x16);
        if (matches) return h + sz_u64_ctz(matches) / 4;
    }

    return sz_find_byteset_serial(h, h_length, set);
}

SZ_PUBLIC sz_cptr_t sz_rfind_byteset_neon(sz_cptr_t h, sz_size_t h_length, sz_byteset_t const *set) {
    sz_u64_t matches;
    sz_u128_vec_t h_vec;
    uint8x16_t set_top_vec_u8x16 = vld1q_u8(&set->_u8s[0]);
    uint8x16_t set_bottom_vec_u8x16 = vld1q_u8(&set->_u8s[16]);

    // Check `sz_find_byteset_neon` for explanations.
    for (; h_length >= 16; h_length -= 16) {
        h_vec.u8x16 = vld1q_u8((sz_u8_t const *)(h) + h_length - 16);
        matches = sz_find_byteset_neon_register_(h_vec, set_top_vec_u8x16, set_bottom_vec_u8x16);
        if (matches) return h + h_length - 1 - sz_u64_clz(matches) / 4;
    }

    return sz_rfind_byteset_serial(h, h_length, set);
}

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#endif            // SZ_USE_NEON
#pragma endregion // NEON Implementation

/*  Implementation of the string search algorithms using the Arm SVE variable-length registers,
 *  available in Arm v9 processors, like in Apple M4+ and Graviton 3+ CPUs.
 */
#pragma region SVE Implementation
#if SZ_USE_SVE
#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("arch=armv8.2-a+sve"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("arch=armv8.2-a+sve")
#endif

SZ_PUBLIC sz_cptr_t sz_find_byte_sve(sz_cptr_t h, sz_size_t h_length, sz_cptr_t n) {
    sz_u8_t const n_scalar = *n;
    // Determine the number of bytes in an SVE vector.
    sz_size_t const vector_bytes = svcntb();
    sz_size_t progress = 0;
    do {
        svbool_t progress_mask = svwhilelt_b8((sz_u64_t)progress, (sz_u64_t)h_length);
        svuint8_t h_vec = svld1(progress_mask, (sz_u8_t const *)(h + progress));
        // Compare: generate a predicate marking lanes where h[i]!=n
        svbool_t equal_vec = svcmpeq_n_u8(progress_mask, h_vec, n_scalar);
        if (svptest_any(progress_mask, equal_vec)) {
            sz_size_t forward_offset_in_register = svcntp_b8(progress_mask, svbrkb_b_z(progress_mask, equal_vec));
            return h + progress + forward_offset_in_register;
        }
        progress += vector_bytes;
    } while (progress < h_length);
    // No match found.
    return SZ_NULL_CHAR;
}

SZ_PUBLIC sz_cptr_t sz_rfind_byte_sve(sz_cptr_t h, sz_size_t h_length, sz_cptr_t n) {
    sz_u8_t const n_scalar = *n;
    // Determine the number of bytes in an SVE vector.
    sz_size_t const vector_bytes = svcntb();
    sz_size_t progress = 0;
    do {
        svbool_t progress_mask = svwhilelt_b8((sz_u64_t)progress, (sz_u64_t)h_length);
        svbool_t backward_mask = svrev_b8(progress_mask);
        svuint8_t h_vec = svld1(backward_mask, (sz_u8_t const *)(h + h_length - progress - vector_bytes));
        // Compare: generate a predicate marking lanes where h[i]!=n
        svbool_t equal_vec = svcmpeq_n_u8(backward_mask, h_vec, n_scalar);
        if (svptest_any(backward_mask, equal_vec)) {
            sz_size_t backward_offset_in_register =
                svcntp_b8(progress_mask, svbrkb_b_z(progress_mask, svrev_b8(equal_vec)));
            return h + h_length - progress - backward_offset_in_register - 1;
        }
        progress += vector_bytes;
    } while (progress < h_length);
    // No match found.
    return SZ_NULL_CHAR;
}

SZ_PUBLIC sz_cptr_t sz_find_sve(sz_cptr_t h, sz_size_t h_length, sz_cptr_t n, sz_size_t n_length) {
    if (h_length < n_length || !n_length) return SZ_NULL_CHAR;
    if (n_length == 1) return sz_find_byte_sve(h, h_length, n);

    // Determine the number of bytes in an SVE vector.
    sz_size_t const vector_bytes = svcntb();
    sz_size_t progress = 0;

    if (n_length == 2) {
        // Broadcast needle characters.
        sz_u8_t n0 = ((sz_u8_t *)n)[0];
        sz_u8_t n1 = ((sz_u8_t *)n)[1];
        do {
            // We must avoid overrunning the haystack for the second byte.
            svbool_t pred = svwhilelt_b8((sz_u64_t)progress, (sz_u64_t)(h_length - 1));
            // Load two adjacent vectors.
            svuint8_t hay0 = svld1(pred, (sz_u8_t const *)(h + progress));
            svuint8_t hay1 = svld1(pred, (sz_u8_t const *)(h + progress + 1));
            svbool_t cmp0 = svcmpeq_n_u8(pred, hay0, n0);
            svbool_t cmp1 = svcmpeq_n_u8(pred, hay1, n1);
            svbool_t matches = svmov_b_z(cmp0, cmp1); //? Practically a bitwise AND
            if (svptest_any(pred, matches)) return h + progress + svcntp_b8(pred, svbrkb_b_z(pred, matches));
            progress += vector_bytes;
        } while (progress < (h_length - 1));
        return SZ_NULL_CHAR;
    }
    else if (n_length == 3) {
        // Broadcast needle characters.
        sz_u8_t n0 = ((sz_u8_t *)n)[0];
        sz_u8_t n1 = ((sz_u8_t *)n)[1];
        sz_u8_t n2 = ((sz_u8_t *)n)[2];
        do {
            // Prevent overrunning for the 3rd byte.
            svbool_t pred = svwhilelt_b8((sz_u64_t)progress, (sz_u64_t)(h_length - 2));
            svuint8_t hay0 = svld1(pred, (sz_u8_t const *)(h + progress));
            svuint8_t hay1 = svld1(pred, (sz_u8_t const *)(h + progress + 1));
            svuint8_t hay2 = svld1(pred, (sz_u8_t const *)(h + progress + 2));
            svbool_t cmp0 = svcmpeq_n_u8(pred, hay0, n0);
            svbool_t cmp1 = svcmpeq_n_u8(pred, hay1, n1);
            svbool_t cmp2 = svcmpeq_n_u8(pred, hay2, n2);
            svbool_t matches = svand_b_z(cmp0, cmp1, cmp2); //? Practically a 3-way AND.
            if (svptest_any(pred, matches)) return h + progress + svcntp_b8(pred, svbrkb_b_z(pred, matches));
            progress += vector_bytes;
        } while (progress < (h_length - 2));
        return SZ_NULL_CHAR;
    }
    else {
        // For longer needles we first pick "anomalies" (i.e. informative offsets)
        sz_size_t offset_first, offset_mid, offset_last;
        sz_locate_needle_anomalies_(n, n_length, &offset_first, &offset_mid, &offset_last);
        // Broadcast the selected needle bytes.
        sz_u8_t n_first = ((sz_u8_t *)n)[offset_first];
        sz_u8_t n_mid = ((sz_u8_t *)n)[offset_mid];
        sz_u8_t n_last = ((sz_u8_t *)n)[offset_last];
        do {
            // Make sure the predicate does not run off the end.
            svbool_t pred = svwhilelt_b8((sz_u64_t)progress, (sz_u64_t)(h_length - n_length + 1));
            // Load haystack bytes at the chosen offsets.
            svuint8_t hay_first = svld1(pred, (sz_u8_t const *)(h + progress + offset_first));
            svuint8_t hay_mid = svld1(pred, (sz_u8_t const *)(h + progress + offset_mid));
            svuint8_t hay_last = svld1(pred, (sz_u8_t const *)(h + progress + offset_last));
            svbool_t cmp0 = svcmpeq_n_u8(pred, hay_first, n_first);
            svbool_t cmp1 = svcmpeq_n_u8(pred, hay_mid, n_mid);
            svbool_t cmp2 = svcmpeq_n_u8(pred, hay_last, n_last);
            svbool_t matches = svand_b_z(cmp0, cmp1, cmp2); //? Practically a 3-way AND.
            // There might be multiple candidate positions, so we need to iterate over them.
            while (svptest_any(pred, matches)) {
                svbool_t pred_to_skip = svbrkb_b_z(pred, matches);
                sz_size_t forward_offset_in_register = svcntp_b8(pred, pred_to_skip);
                if (sz_equal_sve(h + progress + forward_offset_in_register, n, n_length))
                    return h + progress + forward_offset_in_register;
                // If it doesn't match - clear the first bit and continue
                svbool_t first_match = svpnext_b8(svptrue_b8(), pred_to_skip);
                sz_assert_(svcntp_b8(svptrue_b8(), first_match) == 1);
                matches = svbic_b_z(svptrue_b8(), matches, first_match);
            }
            progress += vector_bytes;
        } while (progress < h_length - (n_length - 1));
        return SZ_NULL_CHAR;
    }
}

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#endif            // SZ_USE_SVE
#pragma endregion // SVE Implementation

/*  Pick the right implementation for the string search algorithms.
 *  To override this behavior and precompile all backends - set `SZ_DYNAMIC_DISPATCH` to 1.
 */
#pragma region Compile Time Dispatching
#if !SZ_DYNAMIC_DISPATCH

#pragma region Core Functionality

SZ_DYNAMIC sz_cptr_t sz_find_byte(sz_cptr_t haystack, sz_size_t h_length, sz_cptr_t needle) {
#if SZ_USE_SKYLAKE
    return sz_find_byte_skylake(haystack, h_length, needle);
#elif SZ_USE_HASWELL
    return sz_find_byte_haswell(haystack, h_length, needle);
#elif SZ_USE_SVE
    return sz_find_byte_sve(haystack, h_length, needle);
#elif SZ_USE_NEON
    return sz_find_byte_neon(haystack, h_length, needle);
#else
    return sz_find_byte_serial(haystack, h_length, needle);
#endif
}

SZ_DYNAMIC sz_cptr_t sz_rfind_byte(sz_cptr_t haystack, sz_size_t h_length, sz_cptr_t needle) {
#if SZ_USE_SKYLAKE
    return sz_rfind_byte_skylake(haystack, h_length, needle);
#elif SZ_USE_HASWELL
    return sz_rfind_byte_haswell(haystack, h_length, needle);
#elif SZ_USE_SVE
    return sz_rfind_byte_sve(haystack, h_length, needle);
#elif SZ_USE_NEON
    return sz_rfind_byte_neon(haystack, h_length, needle);
#else
    return sz_rfind_byte_serial(haystack, h_length, needle);
#endif
}

SZ_DYNAMIC sz_cptr_t sz_find(sz_cptr_t haystack, sz_size_t h_length, sz_cptr_t needle, sz_size_t n_length) {
#if SZ_USE_SKYLAKE
    return sz_find_skylake(haystack, h_length, needle, n_length);
#elif SZ_USE_HASWELL
    return sz_find_haswell(haystack, h_length, needle, n_length);
#elif SZ_USE_SVE
    return sz_find_sve(haystack, h_length, needle, n_length);
#elif SZ_USE_NEON
    return sz_find_neon(haystack, h_length, needle, n_length);
#else
    return sz_find_serial(haystack, h_length, needle, n_length);
#endif
}

SZ_DYNAMIC sz_cptr_t sz_rfind(sz_cptr_t haystack, sz_size_t h_length, sz_cptr_t needle, sz_size_t n_length) {
#if SZ_USE_SKYLAKE
    return sz_rfind_skylake(haystack, h_length, needle, n_length);
#elif SZ_USE_HASWELL
    return sz_rfind_haswell(haystack, h_length, needle, n_length);
#elif SZ_USE_NEON
    return sz_rfind_neon(haystack, h_length, needle, n_length);
#else
    return sz_rfind_serial(haystack, h_length, needle, n_length);
#endif
}

SZ_DYNAMIC sz_cptr_t sz_find_byteset(sz_cptr_t text, sz_size_t length, sz_byteset_t const *set) {
#if SZ_USE_ICE
    return sz_find_byteset_ice(text, length, set);
#elif SZ_USE_HASWELL
    return sz_find_byteset_haswell(text, length, set);
#elif SZ_USE_NEON
    return sz_find_byteset_neon(text, length, set);
#else
    return sz_find_byteset_serial(text, length, set);
#endif
}

SZ_DYNAMIC sz_cptr_t sz_rfind_byteset(sz_cptr_t text, sz_size_t length, sz_byteset_t const *set) {
#if SZ_USE_ICE
    return sz_rfind_byteset_ice(text, length, set);
#elif SZ_USE_HASWELL
    return sz_rfind_byteset_haswell(text, length, set);
#elif SZ_USE_NEON
    return sz_rfind_byteset_neon(text, length, set);
#else
    return sz_rfind_byteset_serial(text, length, set);
#endif
}

#pragma endregion
#endif            // !SZ_DYNAMIC_DISPATCH
#pragma endregion // Compile Time Dispatching

#ifdef __cplusplus
}
#endif // __cplusplus
#endif // STRINGZILLA_FIND_H_
