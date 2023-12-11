#include <stringzilla/stringzilla.h>

/**
 *  @brief Load a 16-bit unsigned integer from a potentially unaligned pointer, can be expensive on some platforms.
 */
SZ_INTERNAL sz_u16_t sz_u16_unaligned_load(void const *ptr) {
#ifdef _MSC_VER
    return *((__unaligned sz_u16_t *)ptr);
#else
    __attribute__((aligned(1))) sz_u16_t const *uptr = (sz_u16_t const *)ptr;
    return *uptr;
#endif
}

/**
 *  @brief Load a 32-bit unsigned integer from a potentially unaligned pointer, can be expensive on some platforms.
 */
SZ_INTERNAL sz_u32_t sz_u32_unaligned_load(void const *ptr) {
#ifdef _MSC_VER
    return *((__unaligned sz_u32_t *)ptr);
#else
    __attribute__((aligned(1))) sz_u32_t const *uptr = (sz_u32_t const *)ptr;
    return *uptr;
#endif
}

/**
 *  @brief Load a 64-bit unsigned integer from a potentially unaligned pointer, can be expensive on some platforms.
 */
SZ_INTERNAL sz_u64_t sz_u64_unaligned_load(void const *ptr) {
#ifdef _MSC_VER
    return *((__unaligned sz_u64_t *)ptr);
#else
    __attribute__((aligned(1))) sz_u64_t const *uptr = (sz_u64_t const *)ptr;
    return *uptr;
#endif
}

/**
 *  @brief  Byte-level equality comparison between two strings.
 *          If unaligned loads are allowed, uses a switch-table to avoid loops on short strings.
 */
SZ_PUBLIC sz_bool_t sz_equal_serial(sz_cptr_t a, sz_cptr_t b, sz_size_t length) {
#if SZ_USE_MISALIGNED_LOADS
sz_equal_serial_cycle:
    switch (length) {
    case 0: return 1;
    case 1: return a[0] == b[0];
    case 2: return (sz_u16_unaligned_load(a) == sz_u16_unaligned_load(b));
    case 3: return (sz_u16_unaligned_load(a) == sz_u16_unaligned_load(b)) & (a[2] == b[2]);
    case 4: return (sz_u32_unaligned_load(a) == sz_u32_unaligned_load(b));
    case 5: return (sz_u32_unaligned_load(a) == sz_u32_unaligned_load(b)) & (a[4] == b[4]);
    case 6:
        return (sz_u32_unaligned_load(a) == sz_u32_unaligned_load(b)) &
               (sz_u16_unaligned_load(a + 4) == sz_u16_unaligned_load(b + 4));
    case 7:
        return (sz_u32_unaligned_load(a) == sz_u32_unaligned_load(b)) &
               (sz_u16_unaligned_load(a + 4) == sz_u16_unaligned_load(b + 4)) & (a[6] == b[6]);
    case 8: return sz_u64_unaligned_load(a) == sz_u64_unaligned_load(b);
    default:
        if (sz_u64_unaligned_load(a) != sz_u64_unaligned_load(b)) return 0;
        a += 8, b += 8, length -= 8;
        goto sz_equal_serial_cycle;
    }
#else
    sz_cptr_t const a_end = a + length;
    while (a != a_end && *a == *b) a++, b++;
    return a_end == a;
#endif
}

/**
 *  @brief  Byte-level lexicographic order comparison of two strings.
 *          If unaligned loads are allowed, uses a switch-table to avoid loops on short strings.
 */
SZ_PUBLIC sz_ordering_t sz_order_serial(sz_cptr_t a, sz_size_t a_length, sz_cptr_t b, sz_size_t b_length) {
    sz_ordering_t ordering_lookup[2] = {sz_greater_k, sz_less_k};
#if SZ_USE_MISALIGNED_LOADS
    sz_bool_t a_shorter = a_length < b_length;
    sz_size_t min_length = a_shorter ? a_length : b_length;
    sz_cptr_t min_end = a + min_length;
    for (; a + 8 <= min_end; a += 8, b += 8) {
        sz_u64_t a_vec = sz_u64_unaligned_load(a);
        sz_u64_t b_vec = sz_u64_unaligned_load(b);
        if (a_vec != b_vec) return ordering_lookup[sz_u64_byte_reverse(a_vec) < sz_u64_byte_reverse(b_vec)];
    }
#endif
    for (; a != min_end; ++a, ++b)
        if (*a != *b) return ordering_lookup[*a < *b];
    return a_length != b_length ? ordering_lookup[a_shorter] : sz_equal_k;
}

/**
 *  @brief  Byte-level lexicographic order comparison of two NULL-terminated strings.
 */
SZ_PUBLIC sz_ordering_t sz_order_terminated(sz_cptr_t a, sz_cptr_t b) {
    sz_ordering_t ordering_lookup[2] = {sz_greater_k, sz_less_k};
    for (; *a != '\0' && *b != '\0'; ++a, ++b)
        if (*a != *b) return ordering_lookup[*a < *b];

    // Handle strings of different length
    if (*a == '\0' && *b == '\0') { return sz_equal_k; } // Both strings ended, they are equal
    else if (*a == '\0') { return sz_less_k; }           // String 'a' ended first, it is smaller
    else { return sz_greater_k; }                        // String 'b' ended first, 'a' is larger
}

/**
 *  @brief  Byte-level equality comparison between two 64-bit integers.
 *  @return 64-bit integer, where every top bit in each byte signifies a match.
 */
SZ_INTERNAL sz_u64_t sz_u64_each_byte_equal(sz_u64_t a, sz_u64_t b) {
    sz_u64_t match_indicators = ~(a ^ b);
    // The match is valid, if every bit within each byte is set.
    // For that take the bottom 7 bits of each byte, add one to them,
    // and if this sets the top bit to one, then all the 7 bits are ones as well.
    match_indicators = ((match_indicators & 0x7F7F7F7F7F7F7F7Full) + 0x0101010101010101ull) &
                       ((match_indicators & 0x8080808080808080ull));
    return match_indicators;
}

/**
 *  @brief  Find the first occurrence of a @b single-character needle in an arbitrary length haystack.
 *          This implementation uses hardware-agnostic SWAR technique, to process 8 characters at a time.
 *          Identical to `memchr(haystack, needle[0], haystack_length)`.
 */
SZ_PUBLIC sz_cptr_t sz_find_byte_serial(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle) {

    sz_cptr_t text = haystack;
    sz_cptr_t const end = haystack + haystack_length;

    // Process the misaligned head, to void UB on unaligned 64-bit loads.
    for (; ((sz_size_t)text & 7ull) && text < end; ++text)
        if (*text == *needle) return text;

    // Broadcast the needle into every byte of a 64-bit integer to use SWAR
    // techniques and process eight characters at a time.
    sz_u64_parts_t needle_vec;
    needle_vec.u64 = (sz_u64_t)needle[0] * 0x0101010101010101ull;
    for (; text + 8 <= end; text += 8) {
        sz_u64_t text_slice = *(sz_u64_t const *)text;
        sz_u64_t match_indicators = sz_u64_each_byte_equal(text_slice, needle_vec.u64);
        if (match_indicators != 0) return text + sz_ctz64(match_indicators) / 8;
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
sz_cptr_t sz_rfind_byte_serial(sz_cptr_t const haystack, sz_size_t const haystack_length, sz_cptr_t const needle) {

    sz_cptr_t const end = haystack + haystack_length;
    sz_cptr_t text = end - 1;

    // Process the misaligned head, to void UB on unaligned 64-bit loads.
    for (; ((sz_size_t)text & 7ull) && text >= haystack; --text)
        if (*text == *needle) return text;

    // Broadcast the needle into every byte of a 64-bit integer to use SWAR
    // techniques and process eight characters at a time.
    sz_u64_parts_t needle_vec;
    needle_vec.u64 = (sz_u64_t)needle[0] * 0x0101010101010101ull;
    for (; text - 8 >= haystack; text -= 8) {
        sz_u64_t text_slice = *(sz_u64_t const *)text;
        sz_u64_t match_indicators = sz_u64_each_byte_equal(text_slice, needle_vec.u64);
        if (match_indicators != 0) return text - 8 + sz_clz64(match_indicators) / 8;
    }

    for (; text >= haystack; --text)
        if (*text == *needle) return text;
    return NULL;
}

/**
 *  @brief  2Byte-level equality comparison between two 64-bit integers.
 *  @return 64-bit integer, where every top bit in each 2byte signifies a match.
 */
SZ_INTERNAL sz_u64_t sz_u64_each_2byte_equal(sz_u64_t a, sz_u64_t b) {
    sz_u64_t match_indicators = ~(a ^ b);
    // The match is valid, if every bit within each 2byte is set.
    // For that take the bottom 15 bits of each 2byte, add one to them,
    // and if this sets the top bit to one, then all the 15 bits are ones as well.
    match_indicators = ((match_indicators & 0x7FFF7FFF7FFF7FFFull) + 0x0001000100010001ull) &
                       ((match_indicators & 0x8000800080008000ull));
    return match_indicators;
}

/**
 *  @brief  Find the first occurrence of a @b two-character needle in an arbitrary length haystack.
 *          This implementation uses hardware-agnostic SWAR technique, to process 8 characters at a time.
 */
sz_cptr_t sz_find_2byte_serial(sz_cptr_t const haystack, sz_size_t const haystack_length, sz_cptr_t const needle) {

    sz_cptr_t text = haystack;
    sz_cptr_t const end = haystack + haystack_length;

    // This code simulates hyper-scalar execution, analyzing 7 offsets at a time.
    sz_u64_parts_t text_vec, needle_vec, matches_odd_vec, matches_even_vec;
    needle_vec.u64 = 0;
    needle_vec.u8s[0] = needle[0];
    needle_vec.u8s[1] = needle[1];
    needle_vec.u64 *= 0x0001000100010001ull;

    for (; text + 8 <= end; text += 7) {
        text_vec.u64 = sz_u64_unaligned_load(text);
        matches_even_vec.u64 = sz_u64_each_2byte_equal(text_vec.u64, needle_vec.u64);
        matches_odd_vec.u64 = sz_u64_each_2byte_equal(text_vec.u64 >> 8, needle_vec.u64);

        if (matches_even_vec.u64 + matches_odd_vec.u64) {
            sz_u64_t match_indicators = (matches_even_vec.u64 >> 8) | (matches_odd_vec.u64);
            return text + sz_ctz64(match_indicators) / 8;
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
sz_cptr_t sz_find_3byte_serial(sz_cptr_t const haystack, sz_size_t const haystack_length, sz_cptr_t const needle) {

    sz_cptr_t text = haystack;
    sz_cptr_t end = haystack + haystack_length;

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
        if (match_indicators != 0) return text + sz_ctz64(match_indicators) / 8;
    }

    for (; text + 3 <= end; ++text)
        if (text[0] == needle[0] && text[1] == needle[1] && text[2] == needle[2]) return text;
    return NULL;
}

/**
 *  @brief  Find the first occurrence of a @b four-character needle in an arbitrary length haystack.
 *          This implementation uses hardware-agnostic SWAR technique, to process 8 characters at a time.
 */
sz_cptr_t sz_find_4byte_serial(sz_cptr_t const haystack, sz_size_t const haystack_length, sz_cptr_t const needle) {

    sz_cptr_t text = haystack;
    sz_cptr_t end = haystack + haystack_length;

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
 *  @brief  Bitap algo for exact matching of patterns under @b 64-bytes long.
 *          https://en.wikipedia.org/wiki/Bitap_algorithm
 */
sz_cptr_t sz_find_under64byte_serial(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle,
                                     sz_size_t needle_length) {

    sz_u64_t running_match = ~0ull;
    sz_u64_t pattern_mask[256];
    for (sz_size_t i = 0; i < 256; ++i) { pattern_mask[i] = ~0ull; }
    for (sz_size_t i = 0; i < needle_length; ++i) { pattern_mask[needle[i]] &= ~(1ull << i); }
    for (sz_size_t i = 0; i < haystack_length; ++i) {
        running_match = (running_match << 1) | pattern_mask[haystack[i]];
        if ((running_match & (1ull << (needle_length - 1))) == 0) { return haystack + i - needle_length + 1; }
    }

    return NULL;
}

/**
 *  @brief  Bitap algo for exact matching of patterns under @b 16-bytes long.
 *          https://en.wikipedia.org/wiki/Bitap_algorithm
 */
sz_cptr_t sz_find_under16byte_serial(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle,
                                     sz_size_t needle_length) {

    sz_u16_t running_match = 0xFFFF;
    sz_u16_t pattern_mask[256];
    for (sz_size_t i = 0; i < 256; ++i) { pattern_mask[i] = 0xFFFF; }
    for (sz_size_t i = 0; i < needle_length; ++i) { pattern_mask[needle[i]] &= ~(1u << i); }
    for (sz_size_t i = 0; i < haystack_length; ++i) {
        running_match = (running_match << 1) | pattern_mask[haystack[i]];
        if ((running_match & (1u << (needle_length - 1))) == 0) { return haystack + i - needle_length + 1; }
    }

    return NULL;
}

/**
 *  @brief  Bitap algo for exact matching of patterns under @b 8-bytes long.
 *          https://en.wikipedia.org/wiki/Bitap_algorithm
 */
sz_cptr_t sz_find_under8byte_serial(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle,
                                    sz_size_t needle_length) {

    sz_u8_t running_match = 0xFF;
    sz_u8_t pattern_mask[256];
    for (sz_size_t i = 0; i < 256; ++i) { pattern_mask[i] = 0xFF; }
    for (sz_size_t i = 0; i < needle_length; ++i) { pattern_mask[needle[i]] &= ~(1u << i); }
    for (sz_size_t i = 0; i < haystack_length; ++i) {
        running_match = (running_match << 1) | pattern_mask[haystack[i]];
        if ((running_match & (1u << (needle_length - 1))) == 0) { return haystack + i - needle_length + 1; }
    }

    return NULL;
}

SZ_PUBLIC sz_cptr_t sz_find_serial(sz_cptr_t const haystack, sz_size_t const haystack_length, sz_cptr_t const needle,
                                   sz_size_t const needle_length) {

    if (haystack_length < needle_length) return NULL;

    // For very short strings a lookup table for an optimized backend makes a lot of sense
    switch (needle_length) {
    case 0: return NULL;
    case 1: return sz_find_byte_serial(haystack, haystack_length, needle);
    case 2: return sz_find_2byte_serial(haystack, haystack_length, needle);
    case 3: return sz_find_3byte_serial(haystack, haystack_length, needle);
    case 4: return sz_find_4byte_serial(haystack, haystack_length, needle);
    case 5:
    case 6:
    case 7:
    case 8: return sz_find_under8byte_serial(haystack, haystack_length, needle, needle_length);
    }

    // For needle lengths up to 64, use the existing Bitap algorithm
    if (needle_length <= 16) return sz_find_under16byte_serial(haystack, haystack_length, needle, needle_length);
    if (needle_length <= 64) return sz_find_under64byte_serial(haystack, haystack_length, needle, needle_length);

    // For longer needles, use Bitap for the first 64 bytes and then check the rest
    sz_size_t prefix_length = 64;
    for (sz_size_t i = 0; i <= haystack_length - needle_length; ++i) {
        sz_cptr_t found = sz_find_under64byte_serial(haystack + i, haystack_length - i, needle, prefix_length);
        if (!found) return NULL;

        // Verify the remaining part of the needle
        if (sz_equal_serial(found + prefix_length, needle + prefix_length, needle_length - prefix_length)) return found;

        // Adjust the position
        i = found - haystack + prefix_length - 1;
    }

    return NULL;
}

SZ_PUBLIC sz_size_t sz_prefix_accepted_serial(sz_cptr_t text, sz_size_t length, sz_cptr_t accepted, sz_size_t count) {
    return 0;
}

SZ_PUBLIC sz_size_t sz_prefix_rejected_serial(sz_cptr_t text, sz_size_t length, sz_cptr_t rejected, sz_size_t count) {
    return 0;
}

SZ_PUBLIC sz_size_t sz_levenshtein_memory_needed(sz_size_t _, sz_size_t b_length) {
    return (b_length + b_length + 2) * sizeof(sz_size_t);
}

SZ_PUBLIC sz_size_t sz_levenshtein_serial(       //
    sz_cptr_t const a, sz_size_t const a_length, //
    sz_cptr_t const b, sz_size_t const b_length, //
    sz_cptr_t buffer, sz_size_t const bound) {

    // If one of the strings is empty - the edit distance is equal to the length of the other one
    if (a_length == 0) return b_length <= bound ? b_length : bound;
    if (b_length == 0) return a_length <= bound ? a_length : bound;

    // If the difference in length is beyond the `bound`, there is no need to check at all
    if (a_length > b_length) {
        if (a_length - b_length > bound) return bound;
    }
    else {
        if (b_length - a_length > bound) return bound;
    }

    sz_size_t *previous_distances = (sz_size_t *)buffer;
    sz_size_t *current_distances = previous_distances + b_length + 1;

    for (sz_size_t idx_b = 0; idx_b != (b_length + 1); ++idx_b) previous_distances[idx_b] = idx_b;

    for (sz_size_t idx_a = 0; idx_a != a_length; ++idx_a) {
        current_distances[0] = idx_a + 1;

        // Initialize min_distance with a value greater than bound
        sz_size_t min_distance = bound;

        for (sz_size_t idx_b = 0; idx_b != b_length; ++idx_b) {
            sz_size_t cost_deletion = previous_distances[idx_b + 1] + 1;
            sz_size_t cost_insertion = current_distances[idx_b] + 1;
            sz_size_t cost_substitution = previous_distances[idx_b] + (a[idx_a] != b[idx_b]);
            current_distances[idx_b + 1] = sz_min_of_three(cost_deletion, cost_insertion, cost_substitution);

            // Keep track of the minimum distance seen so far in this row
            if (current_distances[idx_b + 1] < min_distance) { min_distance = current_distances[idx_b + 1]; }
        }

        // If the minimum distance in this row exceeded the bound, return early
        if (min_distance >= bound) return bound;

        // Swap previous_distances and current_distances pointers
        sz_size_t *temp = previous_distances;
        previous_distances = current_distances;
        current_distances = temp;
    }

    return previous_distances[b_length] < bound ? previous_distances[b_length] : bound;
}

SZ_PUBLIC sz_size_t sz_levenshtein_weighted_serial(   //
    sz_cptr_t const a, sz_size_t const a_length,      //
    sz_cptr_t const b, sz_size_t const b_length,      //
    sz_error_cost_t gap, sz_error_cost_t const *subs, //
    sz_cptr_t buffer, sz_size_t const bound) {

    // If one of the strings is empty - the edit distance is equal to the length of the other one
    if (a_length == 0) return (b_length * gap) <= bound ? (b_length * gap) : bound;
    if (b_length == 0) return (a_length * gap) <= bound ? (a_length * gap) : bound;

    // If the difference in length is beyond the `bound`, there is no need to check at all
    if (a_length > b_length) {
        if ((a_length - b_length) * gap > bound) return bound;
    }
    else {
        if ((b_length - a_length) * gap > bound) return bound;
    }

    sz_size_t *previous_distances = (sz_size_t *)buffer;
    sz_size_t *current_distances = previous_distances + b_length + 1;

    for (sz_size_t idx_b = 0; idx_b != (b_length + 1); ++idx_b) previous_distances[idx_b] = idx_b;

    for (sz_size_t idx_a = 0; idx_a != a_length; ++idx_a) {
        current_distances[0] = idx_a + 1;

        // Initialize min_distance with a value greater than bound
        sz_size_t min_distance = bound;
        sz_error_cost_t const *a_subs = subs + a[idx_a] * 256ul;

        for (sz_size_t idx_b = 0; idx_b != b_length; ++idx_b) {
            sz_size_t cost_deletion = previous_distances[idx_b + 1] + gap;
            sz_size_t cost_insertion = current_distances[idx_b] + gap;
            sz_size_t cost_substitution = previous_distances[idx_b] + a_subs[b[idx_b]];
            current_distances[idx_b + 1] = sz_min_of_three(cost_deletion, cost_insertion, cost_substitution);

            // Keep track of the minimum distance seen so far in this row
            if (current_distances[idx_b + 1] < min_distance) { min_distance = current_distances[idx_b + 1]; }
        }

        // If the minimum distance in this row exceeded the bound, return early
        if (min_distance >= bound) return bound;

        // Swap previous_distances and current_distances pointers
        sz_size_t *temp = previous_distances;
        previous_distances = current_distances;
        current_distances = temp;
    }

    return previous_distances[b_length] < bound ? previous_distances[b_length] : bound;
}

SZ_PUBLIC sz_u32_t sz_crc32_serial(sz_cptr_t start, sz_size_t length) {
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
    for (sz_cptr_t const end = start + length; start != end; ++start)
        crc = (crc >> 8) ^ table[(crc ^ (sz_u32_t)*start) & 0xff];
    return crc ^ 0xFFFFFFFF;
}

/**
 *  @brief  Maps any ASCII character to itself, or the lowercase variant, if available.
 */
char sz_char_tolower(char c) {
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
char sz_char_toupper(char c) {
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

SZ_PUBLIC void sz_tolower_serial(sz_cptr_t text, sz_size_t length, sz_ptr_t result) {
    for (sz_cptr_t end = text + length; text != end; ++text, ++result) { *result = sz_char_tolower(*text); }
}

SZ_PUBLIC void sz_toupper_serial(sz_cptr_t text, sz_size_t length, sz_ptr_t result) {
    for (sz_cptr_t end = text + length; text != end; ++text, ++result) { *result = sz_char_toupper(*text); }
}

SZ_PUBLIC void sz_toascii_serial(sz_cptr_t text, sz_size_t length, sz_ptr_t result) {
    for (sz_cptr_t end = text + length; text != end; ++text, ++result) { *result = *text & 0x7F; }
}
