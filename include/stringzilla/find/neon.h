/**
 *  @brief NEON backend for substring & byte-set search.
 *  @file include/stringzilla/find/neon.h
 *  @author Ash Vardanian
 *  @sa include/stringzilla/find.h
 */
#ifndef STRINGZILLA_FIND_NEON_H_
#define STRINGZILLA_FIND_NEON_H_

#include "stringzilla/types.h"
#include "stringzilla/compare.h" // `sz_equal`
#include "stringzilla/find/serial.h"

#ifdef __cplusplus
extern "C" {
#endif

/*  Implementation of the string search algorithms using the Arm NEON instruction set, available on 64-bit
 *  Arm processors. Covers billions of mobile CPUs worldwide, including Apple's A-series, and Qualcomm's Snapdragon.
 */
#if SZ_USE_NEON
#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("+simd"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("+simd")
#endif

/**
 *  @brief Produce a movemask-style 64-bit value from a NEON comparison result.
 *      Each matching byte sets one bit in the result (bit spacing is 4 bits per byte).
 *
 *  @param vec A 16-byte NEON comparison vector (0xFF where matched, 0x00 otherwise).
 *  @return 64-bit mask with one set bit per matching byte (at bit positions 0, 4, 8, ..., 60).
 */
SZ_INTERNAL sz_u64_t sz_find_vreinterpretq_u8_u4_(uint8x16_t vec) {
    // Use `vshrn` to produce a bitmask, similar to `movemask` in SSE.
    // https://community.arm.com/arm-community-blogs/b/infrastructure-solutions-blog/posts/porting-x86-vector-bitmask-optimizations-to-arm-neon
    return vget_lane_u64(vreinterpret_u64_u8(vshrn_n_u16(vreinterpretq_u16_u8(vec), 4)), 0) & 0x8888888888888888ull;
}

SZ_PUBLIC sz_cptr_t sz_find_byte_neon(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle) {
    sz_u64_t matches;
    sz_u128_vec_t haystack_vec, needle_vec, matches_vec;
    needle_vec.u8x16 = vld1q_dup_u8((sz_u8_t const *)needle);

    while (haystack_length >= 16) {
        haystack_vec.u8x16 = vld1q_u8((sz_u8_t const *)haystack);
        matches_vec.u8x16 = vceqq_u8(haystack_vec.u8x16, needle_vec.u8x16);
        // In Arm NEON we don't have a `movemask` to combine it with `ctz` and get the offset of the match.
        // But assuming the `vmaxvq` is cheap, we can use it to find the first match, by blending (bitwise
        // selecting) the vector with a relative offsets array.
        matches = sz_find_vreinterpretq_u8_u4_(matches_vec.u8x16);
        if (matches) return haystack + sz_u64_ctz(matches) / 4;

        haystack += 16, haystack_length -= 16;
    }

    return sz_find_byte_serial(haystack, haystack_length, needle);
}

SZ_PUBLIC sz_cptr_t sz_rfind_byte_neon(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle) {
    sz_u64_t matches;
    sz_u128_vec_t haystack_vec, needle_vec, matches_vec;
    needle_vec.u8x16 = vld1q_dup_u8((sz_u8_t const *)needle);

    while (haystack_length >= 16) {
        haystack_vec.u8x16 = vld1q_u8((sz_u8_t const *)haystack + haystack_length - 16);
        matches_vec.u8x16 = vceqq_u8(haystack_vec.u8x16, needle_vec.u8x16);
        matches = sz_find_vreinterpretq_u8_u4_(matches_vec.u8x16);
        if (matches) return haystack + haystack_length - 1 - sz_u64_clz(matches) / 4;
        haystack_length -= 16;
    }

    return sz_rfind_byte_serial(haystack, haystack_length, needle);
}

/**
 *  @brief Compute a movemask-style presence bitmask for a 16-byte register against a byteset.
 *
 *  @param haystack_vec The 16-byte input register.
 *  @param set_top_vec_u8x16 Top half of the 32-byte byteset (bytes 0..15).
 *  @param set_bottom_vec_u8x16 Bottom half of the 32-byte byteset (bytes 16..31).
 *  @return 64-bit mask with 4-bit-spaced bits set for matching positions.
 */
SZ_PUBLIC sz_u64_t sz_find_byteset_neon_register_( //
    sz_u128_vec_t haystack_vec, uint8x16_t set_top_vec_u8x16, uint8x16_t set_bottom_vec_u8x16) {

    // Once we've read the characters in the haystack, we want to
    // compare them against our bitset. The serial version of that code
    // would look like: `(set_->_u8s[c >> 3] & (1u << (c & 7u))) != 0`.
    uint8x16_t byte_index_vec = vshrq_n_u8(haystack_vec.u8x16, 3);
    uint8x16_t byte_mask_vec = vshlq_u8(vdupq_n_u8(1),
                                        vreinterpretq_s8_u8(vandq_u8(haystack_vec.u8x16, vdupq_n_u8(7))));
    uint8x16_t matches_top_vec = vqtbl1q_u8(set_top_vec_u8x16, byte_index_vec);
    // The table lookup instruction in NEON replies to out-of-bound requests with zeros.
    // The values in `byte_index_vec` all fall in [0; 32). So for values under 16, subtracting 16 will underflow
    // and map into interval [240, 256). Meaning that those will be populated with zeros and we can safely
    // merge `matches_top_vec` and `matches_bottom_vec` with a bitwise OR.
    uint8x16_t matches_bottom_vec = vqtbl1q_u8(set_bottom_vec_u8x16, vsubq_u8(byte_index_vec, vdupq_n_u8(16)));
    uint8x16_t matches_vec = vorrq_u8(matches_top_vec, matches_bottom_vec);
    // Instead of pure `vandq_u8`, we can immediately broadcast a match presence across each 8-bit word.
    matches_vec = vtstq_u8(matches_vec, byte_mask_vec);
    return sz_find_vreinterpretq_u8_u4_(matches_vec);
}

/**
 *  @brief Branch-light substring verify, bit-identical to `sz_equal_neon`, inlined into the match loop
 *         to avoid the per-candidate call + length re-dispatch. Loops over 16-byte `vceqq_u8` chunks with
 *         a `vminvq_u8` all-match reduction and closes with one overlapping tail window.
 */
SZ_INTERNAL sz_bool_t sz_find_verify_neon_(sz_cptr_t a, sz_cptr_t b, sz_size_t length) {
    if (length < 16) return sz_equal_serial(a, b, length);

    sz_size_t offset = 0;
    do {
        uint8x16_t a_u8x16 = vld1q_u8((sz_u8_t const *)(a + offset));
        uint8x16_t b_u8x16 = vld1q_u8((sz_u8_t const *)(b + offset));
        if (vminvq_u8(vceqq_u8(a_u8x16, b_u8x16)) != 255) return sz_false_k; // Check if all bytes match.
        offset += 16;
    } while (offset + 16 <= length);

    // Final check - load the last register-long window from the end.
    uint8x16_t a_tail_u8x16 = vld1q_u8((sz_u8_t const *)(a + length - 16));
    uint8x16_t b_tail_u8x16 = vld1q_u8((sz_u8_t const *)(b + length - 16));
    if (vminvq_u8(vceqq_u8(a_tail_u8x16, b_tail_u8x16)) != 255) return sz_false_k;
    return sz_true_k;
}

SZ_PUBLIC sz_cptr_t sz_find_neon(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle,
                                 sz_size_t needle_length) {

    // This almost never fires, but it's better to be safe than sorry.
    if (haystack_length < needle_length || !needle_length) return SZ_NULL_CHAR;
    if (needle_length == 1) return sz_find_byte_neon(haystack, haystack_length, needle);

    // Scan through the string.
    // Assuming how tiny the Arm NEON registers are, we should avoid internal branches at all costs.
    // That's why, for smaller needles, we use different loops.
    if (needle_length == 2) {
        // Broadcast needle characters into SIMD registers.
        sz_u64_t matches;
        sz_u128_vec_t h_first_vec, h_last_vec, n_first_vec, n_last_vec, matches_vec;
        // Dealing with 16-bit values, we can load 2 registers at a time and compare 31 possible offsets
        // in a single loop iteration.
        n_first_vec.u8x16 = vld1q_dup_u8((sz_u8_t const *)&needle[0]);
        n_last_vec.u8x16 = vld1q_dup_u8((sz_u8_t const *)&needle[1]);
        for (; haystack_length >= 17; haystack += 16, haystack_length -= 16) {
            h_first_vec.u8x16 = vld1q_u8((sz_u8_t const *)(haystack + 0));
            h_last_vec.u8x16 = vld1q_u8((sz_u8_t const *)(haystack + 1));
            matches_vec.u8x16 = vandq_u8(vceqq_u8(h_first_vec.u8x16, n_first_vec.u8x16),
                                         vceqq_u8(h_last_vec.u8x16, n_last_vec.u8x16));
            matches = sz_find_vreinterpretq_u8_u4_(matches_vec.u8x16);
            if (matches) return haystack + sz_u64_ctz(matches) / 4;
        }
    }
    else if (needle_length == 3) {
        // Broadcast needle characters into SIMD registers.
        sz_u64_t matches;
        sz_u128_vec_t h_first_vec, h_mid_vec, h_last_vec, n_first_vec, n_mid_vec, n_last_vec, matches_vec;
        // Comparing 24-bit values is a bumer. Being lazy, I went with the same approach
        // as when searching for string over 4 characters long. I only avoid the last comparison.
        n_first_vec.u8x16 = vld1q_dup_u8((sz_u8_t const *)&needle[0]);
        n_mid_vec.u8x16 = vld1q_dup_u8((sz_u8_t const *)&needle[1]);
        n_last_vec.u8x16 = vld1q_dup_u8((sz_u8_t const *)&needle[2]);
        for (; haystack_length >= 18; haystack += 16, haystack_length -= 16) {
            h_first_vec.u8x16 = vld1q_u8((sz_u8_t const *)(haystack + 0));
            h_mid_vec.u8x16 = vld1q_u8((sz_u8_t const *)(haystack + 1));
            h_last_vec.u8x16 = vld1q_u8((sz_u8_t const *)(haystack + 2));
            matches_vec.u8x16 = vandq_u8(                           //
                vandq_u8(                                           //
                    vceqq_u8(h_first_vec.u8x16, n_first_vec.u8x16), //
                    vceqq_u8(h_mid_vec.u8x16, n_mid_vec.u8x16)),
                vceqq_u8(h_last_vec.u8x16, n_last_vec.u8x16));
            matches = sz_find_vreinterpretq_u8_u4_(matches_vec.u8x16);
            if (matches) return haystack + sz_u64_ctz(matches) / 4;
        }
    }
    else {
        // Pick the parts of the needle that are worth comparing.
        sz_size_t offset_first, offset_mid, offset_last;
        sz_locate_needle_anomalies_(needle, needle_length, &offset_first, &offset_mid, &offset_last);
        // Broadcast those characters into SIMD registers.
        sz_u64_t matches;
        sz_u128_vec_t h_first_vec, h_mid_vec, h_last_vec, n_first_vec, n_mid_vec, n_last_vec, matches_vec;
        n_first_vec.u8x16 = vld1q_dup_u8((sz_u8_t const *)&needle[offset_first]);
        n_mid_vec.u8x16 = vld1q_dup_u8((sz_u8_t const *)&needle[offset_mid]);
        n_last_vec.u8x16 = vld1q_dup_u8((sz_u8_t const *)&needle[offset_last]);
        // Walk through the string.
        for (; haystack_length >= needle_length + 16; haystack += 16, haystack_length -= 16) {
            h_first_vec.u8x16 = vld1q_u8((sz_u8_t const *)(haystack + offset_first));
            h_mid_vec.u8x16 = vld1q_u8((sz_u8_t const *)(haystack + offset_mid));
            h_last_vec.u8x16 = vld1q_u8((sz_u8_t const *)(haystack + offset_last));
            matches_vec.u8x16 = vandq_u8(                           //
                vandq_u8(                                           //
                    vceqq_u8(h_first_vec.u8x16, n_first_vec.u8x16), //
                    vceqq_u8(h_mid_vec.u8x16, n_mid_vec.u8x16)),
                vceqq_u8(h_last_vec.u8x16, n_last_vec.u8x16));
            matches = sz_find_vreinterpretq_u8_u4_(matches_vec.u8x16);
            while (matches) {
                int potential_offset = sz_u64_ctz(matches) / 4;
                if (sz_find_verify_neon_(haystack + potential_offset, needle, needle_length))
                    return haystack + potential_offset;
                matches &= matches - 1;
            }
        }
    }

    return sz_find_serial(haystack, haystack_length, needle, needle_length);
}

SZ_PUBLIC sz_cptr_t sz_rfind_neon(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle,
                                  sz_size_t needle_length) {

    // This almost never fires, but it's better to be safe than sorry.
    if (haystack_length < needle_length || !needle_length) return SZ_NULL_CHAR;
    if (needle_length == 1) return sz_rfind_byte_neon(haystack, haystack_length, needle);

    // Pick the parts of the needle that are worth comparing.
    sz_size_t offset_first, offset_mid, offset_last;
    sz_locate_needle_anomalies_(needle, needle_length, &offset_first, &offset_mid, &offset_last);

    // Will contain 4 bits per character.
    sz_u64_t matches;
    sz_u128_vec_t h_first_vec, h_mid_vec, h_last_vec, n_first_vec, n_mid_vec, n_last_vec, matches_vec;
    n_first_vec.u8x16 = vld1q_dup_u8((sz_u8_t const *)&needle[offset_first]);
    n_mid_vec.u8x16 = vld1q_dup_u8((sz_u8_t const *)&needle[offset_mid]);
    n_last_vec.u8x16 = vld1q_dup_u8((sz_u8_t const *)&needle[offset_last]);

    sz_cptr_t haystack_cursor;
    for (; haystack_length >= needle_length + 16; haystack_length -= 16) {
        haystack_cursor = haystack + haystack_length - needle_length - 16 + 1;
        h_first_vec.u8x16 = vld1q_u8((sz_u8_t const *)(haystack_cursor + offset_first));
        h_mid_vec.u8x16 = vld1q_u8((sz_u8_t const *)(haystack_cursor + offset_mid));
        h_last_vec.u8x16 = vld1q_u8((sz_u8_t const *)(haystack_cursor + offset_last));
        matches_vec.u8x16 = vandq_u8(                           //
            vandq_u8(                                           //
                vceqq_u8(h_first_vec.u8x16, n_first_vec.u8x16), //
                vceqq_u8(h_mid_vec.u8x16, n_mid_vec.u8x16)),
            vceqq_u8(h_last_vec.u8x16, n_last_vec.u8x16));
        matches = sz_find_vreinterpretq_u8_u4_(matches_vec.u8x16);
        while (matches) {
            int potential_offset = sz_u64_clz(matches) / 4;
            if (sz_find_verify_neon_(haystack + haystack_length - needle_length - potential_offset, needle, needle_length))
                return haystack + haystack_length - needle_length - potential_offset;
            sz_assert_((matches & (1ull << (63 - potential_offset * 4))) != 0 &&
                       "The bit must be set before we squash it");
            matches &= ~(1ull << (63 - potential_offset * 4));
        }
    }

    return sz_rfind_serial(haystack, haystack_length, needle, needle_length);
}

SZ_PUBLIC sz_cptr_t sz_find_byteset_neon(sz_cptr_t haystack, sz_size_t haystack_length, sz_byteset_t const *set) {
    sz_u64_t matches;
    sz_u128_vec_t haystack_vec;
    uint8x16_t set_top_vec_u8x16 = vld1q_u8(&set->_u8s[0]);
    uint8x16_t set_bottom_vec_u8x16 = vld1q_u8(&set->_u8s[16]);

    for (; haystack_length >= 16; haystack += 16, haystack_length -= 16) {
        haystack_vec.u8x16 = vld1q_u8((sz_u8_t const *)(haystack));
        matches = sz_find_byteset_neon_register_(haystack_vec, set_top_vec_u8x16, set_bottom_vec_u8x16);
        if (matches) return haystack + sz_u64_ctz(matches) / 4;
    }

    return sz_find_byteset_serial(haystack, haystack_length, set);
}

SZ_PUBLIC sz_cptr_t sz_rfind_byteset_neon(sz_cptr_t haystack, sz_size_t haystack_length, sz_byteset_t const *set) {
    sz_u64_t matches;
    sz_u128_vec_t haystack_vec;
    uint8x16_t set_top_vec_u8x16 = vld1q_u8(&set->_u8s[0]);
    uint8x16_t set_bottom_vec_u8x16 = vld1q_u8(&set->_u8s[16]);

    // Check `sz_find_byteset_neon` for explanations.
    for (; haystack_length >= 16; haystack_length -= 16) {
        haystack_vec.u8x16 = vld1q_u8((sz_u8_t const *)(haystack) + haystack_length - 16);
        matches = sz_find_byteset_neon_register_(haystack_vec, set_top_vec_u8x16, set_bottom_vec_u8x16);
        if (matches) return haystack + haystack_length - 1 - sz_u64_clz(matches) / 4;
    }

    return sz_rfind_byteset_serial(haystack, haystack_length, set);
}

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#endif // SZ_USE_NEON

#ifdef __cplusplus
}
#endif

#endif // STRINGZILLA_FIND_NEON_H_
