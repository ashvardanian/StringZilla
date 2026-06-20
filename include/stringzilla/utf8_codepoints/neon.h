/**
 *  @brief NEON backend for UTF-8 codepoint mechanics.
 *  @file include/stringzilla/utf8_codepoints/neon.h
 *  @author Ash Vardanian
 */
#ifndef STRINGZILLA_UTF8_CODEPOINTS_NEON_H_
#define STRINGZILLA_UTF8_CODEPOINTS_NEON_H_

#include "stringzilla/types.h"
#include "stringzilla/utf8_codepoints/serial.h"

#ifdef __cplusplus
extern "C" {
#endif

#if SZ_USE_NEON
#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("+simd"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("+simd")
#endif

SZ_PUBLIC sz_cptr_t sz_utf8_unpack_chunk_neon(  //
    sz_cptr_t text, sz_size_t length,           //
    sz_rune_t *runes, sz_size_t runes_capacity, //
    sz_size_t *runes_unpacked) {
    return sz_utf8_unpack_chunk_serial(text, length, runes, runes_capacity, runes_unpacked);
}

SZ_INTERNAL sz_u64_t sz_utf8_vreinterpretq_u8_u4_neon_(uint8x16_t vec) {
    // Use `vshrn` to produce a bitmask, similar to `movemask` in SSE.
    // https://community.arm.com/arm-community-blogs/b/infrastructure-solutions-blog/posts/porting-x86-vector-bitmask-optimizations-to-arm-neon
    return vget_lane_u64(vreinterpret_u64_u8(vshrn_n_u16(vreinterpretq_u16_u8(vec), 4)), 0) & 0x8888888888888888ull;
}

#pragma region Multistep newline and whitespace iteration

/*  Multistep newline / whitespace iteration (NEON / AArch64): each full 16-byte tile is classified branchlessly
 *  into per-length delimiter-start masks, narrowed to a nibble bitmask (stride 4, one set bit per match), and the
 *  trusted lanes [0,13] are SIMD left-packed by the peel; the `< 16`-byte remainder goes to the serial helper. */

SZ_PUBLIC sz_size_t sz_utf8_count_neon(sz_cptr_t text, sz_size_t length) {
    sz_u128_vec_t text_vec, headers_vec, continuation_vec;
    uint8x16_t continuation_mask_u8x16 = vdupq_n_u8(0xC0);
    uint8x16_t continuation_pattern_u8x16 = vdupq_n_u8(0x80);
    sz_u8_t const *text_u8 = (sz_u8_t const *)text;
    uint64x2_t char_count_u64x2 = vdupq_n_u64(0);
    while (length >= 16) {
        text_vec.u8x16 = vld1q_u8(text_u8);
        headers_vec.u8x16 = vandq_u8(text_vec.u8x16, continuation_mask_u8x16);
        continuation_vec.u8x16 = vceqq_u8(headers_vec.u8x16, continuation_pattern_u8x16);
        // Convert 0xFF/0x00 into 1/0 and sum.
        uint8x16_t start_flags = vshrq_n_u8(vmvnq_u8(continuation_vec.u8x16), 7);
        uint16x8_t sum16 = vpaddlq_u8(start_flags);
        uint32x4_t sum32 = vpaddlq_u16(sum16);
        uint64x2_t sum64 = vpaddlq_u32(sum32);
        char_count_u64x2 = vaddq_u64(char_count_u64x2, sum64);
        text_u8 += 16;
        length -= 16;
    }

    sz_size_t char_count = vgetq_lane_u64(char_count_u64x2, 0) + vgetq_lane_u64(char_count_u64x2, 1);
    if (length) char_count += sz_utf8_count_serial((sz_cptr_t)text_u8, length);
    return char_count;
}

/**
 *  @brief  Locate the start byte of the @p n -th codepoint (0-indexed) in @p text, or `SZ_NULL_CHAR` if
 *          @p n is past the codepoint count. Byte-exact to `sz_utf8_find_nth_serial`.
 */
SZ_PUBLIC sz_cptr_t sz_utf8_find_nth_neon(sz_cptr_t text, sz_size_t length, sz_size_t n) {
    uint8x16_t continuation_mask_u8x16 = vdupq_n_u8(0xC0);
    uint8x16_t continuation_pattern_u8x16 = vdupq_n_u8(0x80);

    sz_u8_t const *text_u8 = (sz_u8_t const *)text;
    while (length >= 16) {
        uint8x16_t window = vld1q_u8(text_u8);
        uint8x16_t headers = vandq_u8(window, continuation_mask_u8x16);
        uint8x16_t starts_cmp = vmvnq_u8(vceqq_u8(headers, continuation_pattern_u8x16));
        sz_u64_t start_bits = sz_utf8_vreinterpretq_u8_u4_neon_(starts_cmp);
        sz_size_t const start_count = (sz_size_t)sz_u64_popcount(start_bits);

        if (n >= start_count) {
            n -= start_count, text_u8 += 16, length -= 16;
            continue;
        }
        return (sz_cptr_t)(text_u8 + (sz_u64_nth_set_bit(start_bits, n) >> 2));
    }

    return sz_utf8_find_nth_serial((sz_cptr_t)text_u8, length, n);
}

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif

/*  UAX-29 word boundary detection: an all-ASCII window is classified to Word_Break property values via a
 *  128-entry table, the ASCII no-break rules are evaluated branchlessly as neighbour shifts, and boundaries are
 *  emitted directly; non-ASCII windows and the edges take one serial step, staying byte-exact to `_serial`. */

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#endif // SZ_USE_NEON

#ifdef __cplusplus
}
#endif

#endif // STRINGZILLA_UTF8_CODEPOINTS_NEON_H_
