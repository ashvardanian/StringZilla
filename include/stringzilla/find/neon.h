/**
 *  @file include/stringzilla/find/neon.h
 *  @author Ash Vardanian
 *  @date June 21, 2023
 *  @brief NEON backend for substring & byte-set search.
 *
 *  @sa include/stringzilla/find.h
 */
#ifndef STRINGZILLA_FIND_NEON_H_
#define STRINGZILLA_FIND_NEON_H_

#include "stringzilla/types.h"
#include "stringzilla/compare/neon.h" // `sz_equal_neon_`
#include "stringzilla/find/serial.h"

#ifdef __cplusplus
extern "C" {
#endif

/*  Implementation of the string search algorithms using the Arm NEON instruction set, available on
 *  64-bit Arm processors. Covers billions of mobile CPUs worldwide, including Apple's A-series,
 *  and Qualcomm's Snapdragon. */
#if STRINGZILLA_ARCH_ARM64_
#if STRINGZILLA_ARCH_ARM64_NEON_
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
 *  @param[in] vec_u8x16 A 16-byte NEON comparison vector (0xFF where matched, 0x00 otherwise).
 *  @return 64-bit mask with one set bit per matching byte (at bit positions 0, 4, 8, ..., 60).
 */
STRINGZILLA_INLINE sz_u64_t sz_find_vreinterpretq_u8_u4_(uint8x16_t vec_u8x16) {
    // Use `vshrn` to produce a bitmask, similar to `movemask` in SSE.
    // https://community.arm.com/arm-community-blogs/b/infrastructure-solutions-blog/posts/porting-x86-vector-bitmask-optimizations-to-arm-neon
    return vget_lane_u64(vreinterpret_u64_u8(vshrn_n_u16(vreinterpretq_u16_u8(vec_u8x16), 4)), 0) &
           0x8888888888888888ull;
}

STRINGZILLA_INLINE sz_cptr_t sz_find_byte_neon_(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle) {
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
        if (matches) return haystack + sz_u64_ctz_neon_(matches) / 4;

        haystack += 16, haystack_length -= 16;
    }

    return sz_find_byte_serial_(haystack, haystack_length, needle);
}

STRINGZILLA_INLINE sz_cptr_t sz_rfind_byte_neon_(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle) {
    sz_u64_t matches;
    sz_u128_vec_t haystack_vec, needle_vec, matches_vec;
    needle_vec.u8x16 = vld1q_dup_u8((sz_u8_t const *)needle);

    while (haystack_length >= 16) {
        haystack_vec.u8x16 = vld1q_u8((sz_u8_t const *)haystack + haystack_length - 16);
        matches_vec.u8x16 = vceqq_u8(haystack_vec.u8x16, needle_vec.u8x16);
        matches = sz_find_vreinterpretq_u8_u4_(matches_vec.u8x16);
        if (matches) return haystack + haystack_length - 1 - sz_u64_clz_neon_(matches) / 4;
        haystack_length -= 16;
    }

    return sz_rfind_byte_serial_(haystack, haystack_length, needle);
}

/**
 *  @brief Compute a movemask-style presence bitmask for a 16-byte register against a byteset.
 *
 *  @param[in] haystack_vec The 16-byte input register.
 *  @param[in] set_top_vec_u8x16 Top half of the 32-byte byteset (bytes 0..15).
 *  @param[in] set_bottom_vec_u8x16 Bottom half of the 32-byte byteset (bytes 16..31).
 *  @return 64-bit mask with 4-bit-spaced bits set for matching positions.
 */
STRINGZILLA_INLINE sz_u64_t sz_find_byteset_register_neon_( //
    sz_u128_vec_t haystack_vec, uint8x16_t set_top_vec_u8x16, uint8x16_t set_bottom_vec_u8x16) {

    // Once we've read the characters in the haystack, we want to
    // compare them against our bitset. The serial version of that code
    // would look like: `(set_->_u8s[c >> 3] & (1u << (c & 7u))) != 0`.
    uint8x16_t byte_index_u8x16 = vshrq_n_u8(haystack_vec.u8x16, 3);
    uint8x16_t byte_mask_u8x16 = vshlq_u8(vdupq_n_u8(1),
                                          vreinterpretq_s8_u8(vandq_u8(haystack_vec.u8x16, vdupq_n_u8(7))));
    uint8x16_t matches_top_u8x16 = vqtbl1q_u8(set_top_vec_u8x16, byte_index_u8x16);
    // The table lookup instruction in NEON replies to out-of-bound requests with zeros.
    // The values in `byte_index_u8x16` all fall in [0; 32). So for values under 16, subtracting 16 will underflow
    // and map into interval [240, 256). Meaning that those will be populated with zeros and we can safely
    // merge `matches_top_u8x16` and `matches_bottom_u8x16` with a bitwise OR.
    uint8x16_t matches_bottom_u8x16 = vqtbl1q_u8(set_bottom_vec_u8x16, vsubq_u8(byte_index_u8x16, vdupq_n_u8(16)));
    uint8x16_t matches_u8x16 = vorrq_u8(matches_top_u8x16, matches_bottom_u8x16);
    // Instead of pure `vandq_u8`, we can immediately broadcast a match presence across each 8-bit word.
    matches_u8x16 = vtstq_u8(matches_u8x16, byte_mask_u8x16);
    return sz_find_vreinterpretq_u8_u4_(matches_u8x16);
}

STRINGZILLA_INLINE sz_cptr_t sz_find_neon_(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle,
                                           sz_size_t needle_length) {

    // Empty needle matches at the start, like `strstr`.
    if (!needle_length) return haystack;
    if (haystack_length < needle_length) return STRINGZILLA_NULL_CHAR;
    if (needle_length == 1) return sz_find_byte_neon_(haystack, haystack_length, needle);

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
            if (matches) return haystack + sz_u64_ctz_neon_(matches) / 4;
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
            if (matches) return haystack + sz_u64_ctz_neon_(matches) / 4;
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
                int potential_offset = sz_u64_ctz_neon_(matches) / 4;
                if (sz_equal_neon_(haystack + potential_offset, needle, needle_length))
                    return haystack + potential_offset;
                matches &= matches - 1;
            }
        }
    }

    return sz_find_serial_(haystack, haystack_length, needle, needle_length);
}

/** The bits of four 0x00/0xFF masks of 16 lanes, lane @c i of the 64 in bit @c i. */
STRINGZILLA_INLINE sz_u64_t sz_find_bits_neon_(uint8x16x4_t masks_u8x16x4) {
    static sz_u8_t const weights[16] = {1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128};
    uint8x16_t const weights_u8x16 = vld1q_u8(weights);
    uint8x16_t const first_u8x16 = vpaddq_u8(vandq_u8(masks_u8x16x4.val[0], weights_u8x16),
                                             vandq_u8(masks_u8x16x4.val[1], weights_u8x16));
    uint8x16_t const second_u8x16 = vpaddq_u8(vandq_u8(masks_u8x16x4.val[2], weights_u8x16),
                                              vandq_u8(masks_u8x16x4.val[3], weights_u8x16));
    uint8x16_t const quarters_u8x16 = vpaddq_u8(first_u8x16, second_u8x16);
    return vgetq_lane_u64(vreinterpretq_u64_u8(vpaddq_u8(quarters_u8x16, quarters_u8x16)), 0);
}

/** Bit @c i set where @p bytes match at `text + i + offsets[k]` for every @c k, for the 64 starts
 *  from @p text, gathering the bits only when one narrowing finds a match at all. */
STRINGZILLA_INLINE sz_u64_t sz_find_candidates_neon_(sz_u8_t const *text, sz_size_t const offsets[3],
                                                     sz_u8_t const bytes[3]) {
    uint8x16x4_t const first_u8x16x4 = vld1q_u8_x4(text + offsets[0]);
    uint8x16x4_t const second_u8x16x4 = vld1q_u8_x4(text + offsets[1]);
    uint8x16x4_t const third_u8x16x4 = vld1q_u8_x4(text + offsets[2]);
    uint8x16x4_t matches_u8x16x4;
    for (int chunk = 0; chunk != 4; ++chunk)
        matches_u8x16x4.val[chunk] = vandq_u8(vandq_u8(vceqq_u8(first_u8x16x4.val[chunk], vdupq_n_u8(bytes[0])),
                                                       vceqq_u8(second_u8x16x4.val[chunk], vdupq_n_u8(bytes[1]))),
                                              vceqq_u8(third_u8x16x4.val[chunk], vdupq_n_u8(bytes[2])));
    uint8x16_t const any_u8x16 = vorrq_u8(vorrq_u8(matches_u8x16x4.val[0], matches_u8x16x4.val[1]),
                                          vorrq_u8(matches_u8x16x4.val[2], matches_u8x16x4.val[3]));
    return sz_find_vreinterpretq_u8_u4_(any_u8x16) ? sz_find_bits_neon_(matches_u8x16x4) : 0;
}

STRINGZILLA_INLINE sz_cptr_t sz_rfind_neon_(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle,
                                            sz_size_t needle_length) {

    // Empty needle matches at the end.
    if (!needle_length) return haystack + haystack_length;
    if (haystack_length < needle_length) return STRINGZILLA_NULL_CHAR;
    if (needle_length == 1) return sz_rfind_byte_neon_(haystack, haystack_length, needle);

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
            int potential_offset = sz_u64_clz_neon_(matches) / 4;
            if (sz_equal_neon_(haystack + haystack_length - needle_length - potential_offset, needle, needle_length))
                return haystack + haystack_length - needle_length - potential_offset;
            sz_assert_((matches & (1ull << (63 - potential_offset * 4))) != 0 &&
                       "The bit must be set before we squash it");
            matches &= ~(1ull << (63 - potential_offset * 4));
        }
    }

    return sz_rfind_serial_(haystack, haystack_length, needle, needle_length);
}

STRINGZILLA_INLINE sz_cptr_t sz_find_byteset_neon_(sz_cptr_t haystack, sz_size_t haystack_length,
                                                   sz_byteset_t const *set) {
    sz_u64_t matches;
    sz_u128_vec_t haystack_vec;
    uint8x16_t set_top_vec_u8x16 = vld1q_u8(&set->_u8s[0]);
    uint8x16_t set_bottom_vec_u8x16 = vld1q_u8(&set->_u8s[16]);

    for (; haystack_length >= 16; haystack += 16, haystack_length -= 16) {
        haystack_vec.u8x16 = vld1q_u8((sz_u8_t const *)(haystack));
        matches = sz_find_byteset_register_neon_(haystack_vec, set_top_vec_u8x16, set_bottom_vec_u8x16);
        if (matches) return haystack + sz_u64_ctz_neon_(matches) / 4;
    }

    return sz_find_byteset_serial_(haystack, haystack_length, set);
}

STRINGZILLA_INLINE sz_cptr_t sz_rfind_byteset_neon_(sz_cptr_t haystack, sz_size_t haystack_length,
                                                    sz_byteset_t const *set) {
    sz_u64_t matches;
    sz_u128_vec_t haystack_vec;
    uint8x16_t set_top_vec_u8x16 = vld1q_u8(&set->_u8s[0]);
    uint8x16_t set_bottom_vec_u8x16 = vld1q_u8(&set->_u8s[16]);

    // Check `sz_find_byteset_neon` for explanations.
    for (; haystack_length >= 16; haystack_length -= 16) {
        haystack_vec.u8x16 = vld1q_u8((sz_u8_t const *)(haystack) + haystack_length - 16);
        matches = sz_find_byteset_register_neon_(haystack_vec, set_top_vec_u8x16, set_bottom_vec_u8x16);
        if (matches) return haystack + haystack_length - 1 - sz_u64_clz_neon_(matches) / 4;
    }

    return sz_rfind_byteset_serial_(haystack, haystack_length, set);
}

#if STRINGZILLA_TARGET_NEON

STRINGZILLA_API sz_status_t sz_find_byte_neon(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle,
                                              sz_cptr_t *match, sz_stream_t stream) {
    sz_assert_(stream == STRINGZILLA_NULL);
    *match = sz_find_byte_neon_(haystack, haystack_length, needle);
    return sz_success_k;
}

STRINGZILLA_API sz_status_t sz_rfind_byte_neon(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle,
                                               sz_cptr_t *match, sz_stream_t stream) {
    sz_assert_(stream == STRINGZILLA_NULL);
    *match = sz_rfind_byte_neon_(haystack, haystack_length, needle);
    return sz_success_k;
}

STRINGZILLA_API sz_status_t sz_find_neon(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle,
                                         sz_size_t needle_length, sz_cptr_t *match, sz_stream_t stream) {
    sz_assert_(stream == STRINGZILLA_NULL);
    *match = sz_find_neon_(haystack, haystack_length, needle, needle_length);
    return sz_success_k;
}

STRINGZILLA_API sz_status_t sz_find_strings_neon(sz_strings_view_t const *strings, sz_size_t const *order,
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
        sz_cptr_t const match = sz_find_neon_(view.start, view.length, needle, needle_length);
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
                                  ? sz_find_candidates_neon_(data + block - back, offsets, bytes) >> back
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
            sz_size_t const lane = own ? (sz_size_t)sz_u64_ctz_neon_(own) : 64;
            candidates &= ~(own & (0 - own));
            if (own && needle_length > 3 && !sz_equal_neon_((sz_cptr_t)data + block + lane, needle, needle_length))
                continue;
            positions[row] = own ? block + lane - row_start : STRINGZILLA_SIZE_MAX;
            row_start = row_end, ++row;
            row_end = sz_strings_view_offset_(strings, sz_min_of_two(row + 1, count));
        }
    }
    return sz_success_k;
}

STRINGZILLA_API sz_status_t sz_rfind_neon(sz_cptr_t haystack, sz_size_t haystack_length, sz_cptr_t needle,
                                          sz_size_t needle_length, sz_cptr_t *match, sz_stream_t stream) {
    sz_assert_(stream == STRINGZILLA_NULL);
    *match = sz_rfind_neon_(haystack, haystack_length, needle, needle_length);
    return sz_success_k;
}

STRINGZILLA_API sz_status_t sz_find_byteset_neon(sz_cptr_t haystack, sz_size_t haystack_length, sz_byteset_t const *set,
                                                 sz_cptr_t *match, sz_stream_t stream) {
    sz_assert_(stream == STRINGZILLA_NULL);
    *match = sz_find_byteset_neon_(haystack, haystack_length, set);
    return sz_success_k;
}

STRINGZILLA_API sz_status_t sz_rfind_byteset_neon(sz_cptr_t haystack, sz_size_t haystack_length,
                                                  sz_byteset_t const *set, sz_cptr_t *match, sz_stream_t stream) {
    sz_assert_(stream == STRINGZILLA_NULL);
    *match = sz_rfind_byteset_neon_(haystack, haystack_length, set);
    return sz_success_k;
}

#endif // STRINGZILLA_TARGET_NEON

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#endif // STRINGZILLA_ARCH_ARM64_NEON_
#endif // STRINGZILLA_ARCH_ARM64_

#ifdef __cplusplus
}
#endif

#endif // STRINGZILLA_FIND_NEON_H_
