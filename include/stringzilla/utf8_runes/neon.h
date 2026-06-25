/**
 *  @brief NEON backend for UTF-8 codepoint mechanics.
 *  @file include/stringzilla/utf8_runes/neon.h
 *  @author Ash Vardanian
 */
#ifndef STRINGZILLA_UTF8_RUNES_NEON_H_
#define STRINGZILLA_UTF8_RUNES_NEON_H_

#include "stringzilla/types.h"
#include "stringzilla/utf8_runes/serial.h"

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
        uint8x16_t start_flags_u8x16 = vshrq_n_u8(vmvnq_u8(continuation_vec.u8x16), 7);
        uint16x8_t sum_u16x8 = vpaddlq_u8(start_flags_u8x16);
        uint32x4_t sum_u32x4 = vpaddlq_u16(sum_u16x8);
        uint64x2_t sum_u64x2 = vpaddlq_u32(sum_u32x4);
        char_count_u64x2 = vaddq_u64(char_count_u64x2, sum_u64x2);
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
        uint8x16_t window_u8x16 = vld1q_u8(text_u8);
        uint8x16_t headers_u8x16 = vandq_u8(window_u8x16, continuation_mask_u8x16);
        uint8x16_t starts_cmp_u8x16 = vmvnq_u8(vceqq_u8(headers_u8x16, continuation_pattern_u8x16));
        sz_u64_t start_bits = sz_utf8_vreinterpretq_u8_u4_neon_(starts_cmp_u8x16);
        sz_size_t const start_count = (sz_size_t)sz_u64_popcount(start_bits);

        if (n >= start_count) {
            n -= start_count, text_u8 += 16, length -= 16;
            continue;
        }
        return (sz_cptr_t)(text_u8 + (sz_u64_nth_set_bit(start_bits, n) >> 2));
    }

    return sz_utf8_find_nth_serial((sz_cptr_t)text_u8, length, n);
}

#pragma endregion Multistep newline and whitespace iteration

#pragma region Shared SIMD leaf substrate

/*  Family-agnostic NEON (AArch64) leaf helpers shared by every UTF-8 segmentation kernel (word / grapheme /
 *  sentence / line / delimiters). The exact twin of the Ice Lake @ref sz_utf8_rune_window_t substrate and the
 *  AVX2 @ref sz_utf8_rune_window_haswell_t backend: a 64-byte window held as four `uint8x16_t` quarters, masks
 *  reduced to one-bit-per-byte `sz_u64_t` lane masks (bit `i` <=> lane `i`), and gather-free table cascades. The
 *  portable rule algebra in `serial.h` consumes the `sz_u64_t` masks unchanged across all three backends. */

/** @brief  The decoded 64-byte window for the NEON backend. The 64 bytes live as four `uint8x16_t` quarters
 *          (`window[0]` = lanes [0, 16), ... `window[3]` = lanes [48, 64)); the per-lane byte-domain codepoint
 *          halves `high`/`low` share that shape. Masks are `sz_u64_t` (one bit per byte-lane: the NEON movemask of
 *          each quarter placed at bit positions [0,16)/[16,32)/[32,48)/[48,64)) rather than the Ice Lake
 *          `__mmask64`. Field names and semantics match @ref sz_utf8_rune_window_t so the portable rule
 *          algebra is unchanged. */
typedef struct sz_utf8_rune_window_neon_t {
    uint8x16_t window[4];       /**< Raw input bytes for lanes [16*q, 16*q+16). */
    uint8x16_t high[4];         /**< Per-lane `codepoint >> 8`. */
    uint8x16_t low[4];          /**< Per-lane `codepoint & 0xFF`. */
    sz_u64_t continuation;      /**< Bit `i` => lane `i` is a continuation byte `10xxxxxx`. */
    sz_u64_t codepoint_starts;  /**< Bit `i` => lane `i` begins a codepoint (loaded, non-continuation). */
    sz_u64_t two_byte_starts;   /**< Bit `i` => lane `i` is a 2-byte lead `110xxxxx`. */
    sz_u64_t three_byte_starts; /**< Bit `i` => lane `i` is a 3-byte lead `1110xxxx`. */
    sz_u64_t four_byte_starts;  /**< Bit `i` => lane `i` is a 4-byte lead `11110xxx`. */
    sz_size_t loaded;           /**< Number of bytes actually loaded (<= 64). */
} sz_utf8_rune_window_neon_t;

/** @brief  Per-byte logical right shift by @p shift keeping the low @p keep bits — the NEON twin of `srl8_`.
 *          `vshrq_n_u8` needs an immediate shift; the shift amounts used by the segmentation classifiers (2 for the
 *          decode 3-byte high reconstruction, 4 for the line-break 4-byte plane reconstruction) are spelled out. */
SZ_INTERNAL uint8x16_t sz_utf8_srl8_neon_(uint8x16_t value, int shift, sz_u8_t keep) {
    uint8x16_t shifted_u8x16;
    switch (shift) {
    case 2: shifted_u8x16 = vshrq_n_u8(value, 2); break;
    case 4: shifted_u8x16 = vshrq_n_u8(value, 4); break;
    default: shifted_u8x16 = value; break;
    }
    return vandq_u8(shifted_u8x16, vdupq_n_u8(keep));
}

/**
 *  @brief  Reduce one `uint8x16_t` whose lanes are 0x00/0xFF booleans into a 16-bit lane mask (bit `i` <=> lane `i`).
 *
 *  The crux of the NEON substrate. `vshrn_n_u16(.., 4)` (the existing `sz_utf8_vreinterpretq_u8_u4_neon_`) yields four
 *  bits per byte (only 16 bytes per 64-bit word), which does NOT match the `vpmovmskb` one-bit-per-byte layout the
 *  shared rule engine consumes. Instead AND each lane's 0xFF with its bit position {1,2,4,..,128} (repeated per 8-lane
 *  half), then `vaddv` each half to collapse the eight in-position bits into one byte: the low half becomes mask bits
 *  [0,8), the high half bits [8,16). No multiply, no `vshrn`; two `vaddv_u8` horizontal adds.
 */
SZ_INTERNAL sz_u64_t sz_utf8_movemask16_neon_(uint8x16_t boolean_lanes) {
    static sz_u8_t const bit_position_lanes[16] = {1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128};
    uint8x16_t const bit_position_u8x16 = vld1q_u8(bit_position_lanes);
    uint8x16_t const isolated_u8x16 = vandq_u8(boolean_lanes, bit_position_u8x16);
    sz_u64_t const low_bits = (sz_u64_t)vaddv_u8(vget_low_u8(isolated_u8x16));
    sz_u64_t const high_bits = (sz_u64_t)vaddv_u8(vget_high_u8(isolated_u8x16));
    return low_bits | (high_bits << 8);
}

/** @brief  Combine the four per-quarter NEON movemasks into one 64-bit lane mask: quarter `q` -> bits [16*q, 16*q+16).
 *          The NEON twin of @ref sz_utf8_mask_combine_haswell_ (which OR-combines two 32-bit halves). */
SZ_INTERNAL sz_u64_t sz_utf8_mask_combine_neon_( //
    uint8x16_t quarter0, uint8x16_t quarter1, uint8x16_t quarter2, uint8x16_t quarter3) {
    sz_u64_t mask = sz_utf8_movemask16_neon_(quarter0);
    mask |= sz_utf8_movemask16_neon_(quarter1) << 16;
    mask |= sz_utf8_movemask16_neon_(quarter2) << 32;
    mask |= sz_utf8_movemask16_neon_(quarter3) << 48;
    return mask;
}

/** @brief  Masked 64-byte load into four quarters; bytes [loaded, 64) read as zero (the NEON stand-in for
 *          `_mm512_maskz_loadu_epi8`). A zero-initialized vector union stages the partial tail so we never read past
 *          `text + loaded`. Mirrors @ref sz_utf8_load_window_haswell_. */
SZ_INTERNAL void sz_utf8_load_window_neon_(sz_u8_t const *text, sz_size_t loaded, uint8x16_t *out) {
    if (loaded >= 64) {
        out[0] = vld1q_u8(text + 0);
        out[1] = vld1q_u8(text + 16);
        out[2] = vld1q_u8(text + 32);
        out[3] = vld1q_u8(text + 48);
        return;
    }
    sz_u512_vec_t window_vec;
    window_vec.u8x16s[0] = window_vec.u8x16s[1] = window_vec.u8x16s[2] = window_vec.u8x16s[3] = vdupq_n_u8(0);
    for (sz_size_t i = 0; i < loaded; ++i) window_vec.u8s[i] = text[i];
    out[0] = window_vec.u8x16s[0];
    out[1] = window_vec.u8x16s[1];
    out[2] = window_vec.u8x16s[2];
    out[3] = window_vec.u8x16s[3];
}

/**
 *  @brief  Forward neighbours `next1[i] = window[i+1]`, `next2[i] = window[i+2]`, `next3[i] = window[i+3]` over all 64
 *          lanes, with the lanes past the window WRAPPING modulo 64 to match Ice Lake's `_mm512_permutexvar_epi8`
 *          (so `next1[63]==window[0]`, etc.). NEON `vextq_u8` concatenates two adjacent quarters and extracts the
 *          shifted span; the successor of the last quarter wraps to quarter 0 (byte 64 aliases byte 0). The three
 *          neighbour distances are provided because the family classifiers need up to `next3` (4-byte sequences).
 */
SZ_INTERNAL void sz_utf8_forward_neighbours_neon_( //
    uint8x16_t const *window, uint8x16_t *next1, uint8x16_t *next2, uint8x16_t *next3) {
    for (int quarter = 0; quarter < 4; ++quarter) {
        uint8x16_t const here_u8x16 = window[quarter];
        uint8x16_t const successor_u8x16 = window[(quarter + 1) & 3];
        next1[quarter] = vextq_u8(here_u8x16, successor_u8x16, 1);
        next2[quarter] = vextq_u8(here_u8x16, successor_u8x16, 2);
        next3[quarter] = vextq_u8(here_u8x16, successor_u8x16, 3);
    }
}

/** @brief  Load up to 64 bytes (masked tail) and decode every lane into byte-domain halves — the NEON twin of
 *          @ref sz_utf8_rune_decode_window_, bit-identical to it (and to `_haswell_`) on every lane. */
SZ_INTERNAL sz_utf8_rune_window_neon_t sz_utf8_rune_decode_window_neon_( //
    sz_u8_t const *text, sz_size_t available) {
    sz_utf8_rune_window_neon_t result;
    result.loaded = available < 64 ? available : 64;

    uint8x16_t window_u8x16[4];
    sz_utf8_load_window_neon_(text, result.loaded, window_u8x16);
    for (int quarter = 0; quarter < 4; ++quarter) result.window[quarter] = window_u8x16[quarter];

    uint8x16_t next1[4], next2[4], next3[4];
    sz_utf8_forward_neighbours_neon_(window_u8x16, next1, next2, next3);

    sz_u64_t const loaded_mask = sz_u64_mask_until_serial_(result.loaded);

    /* Lead-class detection over loaded lanes: (byte & mask) == pattern, AND-clamped to loaded lanes. */
    uint8x16_t continuation_bool_u8x16[4], two_byte_bool_u8x16[4], three_byte_bool_u8x16[4], four_byte_bool_u8x16[4];
    uint8x16_t const mask_continuation_u8x16 = vdupq_n_u8(0xC0), pattern_continuation_u8x16 = vdupq_n_u8(0x80);
    uint8x16_t const mask_two_u8x16 = vdupq_n_u8(0xE0), pattern_two_u8x16 = vdupq_n_u8(0xC0);
    uint8x16_t const mask_three_u8x16 = vdupq_n_u8(0xF0), pattern_three_u8x16 = vdupq_n_u8(0xE0);
    uint8x16_t const mask_four_u8x16 = vdupq_n_u8(0xF8), pattern_four_u8x16 = vdupq_n_u8(0xF0);
    for (int quarter = 0; quarter < 4; ++quarter) {
        uint8x16_t const here_u8x16 = window_u8x16[quarter];
        continuation_bool_u8x16[quarter] = vceqq_u8(vandq_u8(here_u8x16, mask_continuation_u8x16), pattern_continuation_u8x16);
        two_byte_bool_u8x16[quarter] = vceqq_u8(vandq_u8(here_u8x16, mask_two_u8x16), pattern_two_u8x16);
        three_byte_bool_u8x16[quarter] = vceqq_u8(vandq_u8(here_u8x16, mask_three_u8x16), pattern_three_u8x16);
        four_byte_bool_u8x16[quarter] = vceqq_u8(vandq_u8(here_u8x16, mask_four_u8x16), pattern_four_u8x16);
    }
    result.continuation = sz_utf8_mask_combine_neon_(continuation_bool_u8x16[0], continuation_bool_u8x16[1],
                                                     continuation_bool_u8x16[2], continuation_bool_u8x16[3]) &
                          loaded_mask;
    result.codepoint_starts = loaded_mask & ~result.continuation;
    result.two_byte_starts = sz_utf8_mask_combine_neon_(two_byte_bool_u8x16[0], two_byte_bool_u8x16[1],
                                                        two_byte_bool_u8x16[2], two_byte_bool_u8x16[3]) &
                             loaded_mask;
    result.three_byte_starts = sz_utf8_mask_combine_neon_(three_byte_bool_u8x16[0], three_byte_bool_u8x16[1],
                                                          three_byte_bool_u8x16[2], three_byte_bool_u8x16[3]) &
                               loaded_mask;
    result.four_byte_starts = sz_utf8_mask_combine_neon_(four_byte_bool_u8x16[0], four_byte_bool_u8x16[1],
                                                         four_byte_bool_u8x16[2], four_byte_bool_u8x16[3]) &
                              loaded_mask;

    uint8x16_t const low_five_bits_u8x16 = vdupq_n_u8(0x1F);
    uint8x16_t const low_six_bits_u8x16 = vdupq_n_u8(0x3F);
    uint8x16_t const low_two_bits_u8x16 = vdupq_n_u8(0x03);
    uint8x16_t const low_four_bits_u8x16 = vdupq_n_u8(0x0F);

    for (int quarter = 0; quarter < 4; ++quarter) {
        uint8x16_t const here_u8x16 = window_u8x16[quarter];
        uint8x16_t const next_byte_u8x16 = next1[quarter];
        uint8x16_t const next_byte2_u8x16 = next2[quarter];

        /* 2-byte: codepoint = ((b0 & 0x1F) << 6) | (b1 & 0x3F); high = codepoint >> 8, low = codepoint & 0xFF. */
        uint8x16_t const high_two_u8x16 = sz_utf8_srl8_neon_(vandq_u8(here_u8x16, low_five_bits_u8x16), 2, 0x07);
        uint8x16_t const low_two_u8x16 =
            vorrq_u8(vshlq_n_u8(vandq_u8(here_u8x16, low_two_bits_u8x16), 6), vandq_u8(next_byte_u8x16, low_six_bits_u8x16));

        /* 3-byte: codepoint = ((b0 & 0x0F) << 12) | ((b1 & 0x3F) << 6) | (b2 & 0x3F). */
        uint8x16_t const high_three_u8x16 =
            vorrq_u8(vshlq_n_u8(vandq_u8(here_u8x16, low_four_bits_u8x16), 4), sz_utf8_srl8_neon_(next_byte_u8x16, 2, 0x0F));
        uint8x16_t const low_three_u8x16 =
            vorrq_u8(vshlq_n_u8(vandq_u8(next_byte_u8x16, low_two_bits_u8x16), 6), vandq_u8(next_byte2_u8x16, low_six_bits_u8x16));

        /* Blend 2-byte vs 3-byte per lane: select the 3-byte value where this lane is a 3-byte lead. */
        uint8x16_t const three_select_u8x16 = three_byte_bool_u8x16[quarter];
        result.high[quarter] = vbslq_u8(three_select_u8x16, high_three_u8x16, high_two_u8x16);
        result.low[quarter] = vbslq_u8(three_select_u8x16, low_three_u8x16, low_two_u8x16);
    }
    return result;
}

/** @brief  One nibble-cascade stage with a sub-256 selector: `result[lane] = table[selector[lane]*16 + within[lane]]`.
 *          Each of @p tile_count 16-byte rows is one selector value; @p within is the addressing nibble (caller-masked
 *          to `[0,16)`). The NEON twin of the AVX2 cascade and VBMI `vpermi2b`: each resident row is loaded into a
 *          `uint8x16_t` and shuffled by @p within via `vqtbl1q_u8`, then blended in for the lanes whose @p selector
 *          picks that row. Gather-free — only `vld1q`/`vqtbl1q`/`vceqq`/`vbslq`. @p within / @p selector address one
 *          quarter; the caller iterates the four quarters. */
SZ_INTERNAL uint8x16_t sz_utf8_rune_cascade_stage_neon_( //
    sz_u8_t const *table, int tile_count, uint8x16_t selector, uint8x16_t within) {
    uint8x16_t result_u8x16 = vdupq_n_u8(0);
    for (int tile = 0; tile < tile_count; ++tile) {
        uint8x16_t const row_u8x16 = vld1q_u8(table + tile * 16);
        uint8x16_t const picked_u8x16 = vqtbl1q_u8(row_u8x16, within);
        uint8x16_t const here_u8x16 = vceqq_u8(selector, vdupq_n_u8((sz_u8_t)tile));
        result_u8x16 = vbslq_u8(here_u8x16, picked_u8x16, result_u8x16);
    }
    return result_u8x16;
}

/** @brief  256-entry byte LUT addressed by a per-lane byte index in `[0,256)`: `result[lane] = group_base[index[lane]]`.
 *          Four `vqtbl4q_u8` reads over the four resident 64-byte quads; `vqtbl4q_u8` returns zero for indices >= 64,
 *          so subtracting 64/128/192 routes each lane to exactly one quad and the four results OR together. The NEON
 *          twin of the substrate `lut256` leaf. @p index addresses one quarter. */
SZ_INTERNAL uint8x16_t sz_utf8_rune_lut256_neon_(sz_u8_t const *group_base, uint8x16_t index) {
    uint8x16x4_t const quad0_u8x16x4 = vld1q_u8_x4(group_base + 0 * 64);
    uint8x16x4_t const quad1_u8x16x4 = vld1q_u8_x4(group_base + 1 * 64);
    uint8x16x4_t const quad2_u8x16x4 = vld1q_u8_x4(group_base + 2 * 64);
    uint8x16x4_t const quad3_u8x16x4 = vld1q_u8_x4(group_base + 3 * 64);
    uint8x16_t const within0_u8x16 = index;
    uint8x16_t const within1_u8x16 = vsubq_u8(index, vdupq_n_u8(64));
    uint8x16_t const within2_u8x16 = vsubq_u8(index, vdupq_n_u8(128));
    uint8x16_t const within3_u8x16 = vsubq_u8(index, vdupq_n_u8(192));
    uint8x16_t result_u8x16 = vqtbl4q_u8(quad0_u8x16x4, within0_u8x16);
    result_u8x16 = vorrq_u8(result_u8x16, vqtbl4q_u8(quad1_u8x16x4, within1_u8x16));
    result_u8x16 = vorrq_u8(result_u8x16, vqtbl4q_u8(quad2_u8x16x4, within2_u8x16));
    result_u8x16 = vorrq_u8(result_u8x16, vqtbl4q_u8(quad3_u8x16x4, within3_u8x16));
    return result_u8x16;
}

#pragma endregion Shared SIMD leaf substrate

#pragma region Drains

/** @brief  Left-pack the set lane indices (in [0, 64), ascending) of a 64-bit @p mask into @p out[0..popcount).
 *          NEON has no `vpcompressb`/`pext`: pull the lowest set bit's index via `ctz`, clear it via `mask & (mask-1)`.
 *          This matches the AVX2 `sz_utf8_unpack_indices_haswell_` BMI2 path semantically; on AArch64
 *          `ctz` lowers to `rbit`+`clz`, and the loop trip count is the boundary popcount (sparse for real text). */
SZ_INTERNAL void sz_utf8_unpack_indices_neon_(sz_u64_t mask, sz_u8_t *out) {
    while (mask) {
        *out++ = (sz_u8_t)sz_u64_ctz(mask);
        mask &= mask - 1; // clear the lowest set bit
    }
}

/**
 *  @brief  Left-pack the set-bit positions of a 64-bit lane @p mask into @p out as a dense, ascending array of
 *          byte-offsets in [0, 64), returning the count - the NEON `vpcompressb`-free start-compaction. Replaces the
 *          scalar `ctz` walk with a 2 KB shuffle-LUT (`leftpack8`) keyed by each 8-bit sub-mask: for every 16-lane
 *          quarter the mask splits into a low and a high byte, each `vqtbl1q_u8`-shuffles the LUT row of its set-bit
 *          positions, the high half is offset by +8, and the two halves are stitched at `popcount(low8)` with one
 *          `vqtbl1q_u8` over a gap-shift index (no scalar per-lane index walk). The quarter offset `q*16` is added in
 *          vector and the dense lanes are stored at the running cursor; one loop over the four quarters.
 */
/** @brief  Expand the low 16 bits of @p submask into a `uint8x16_t` whose lane `i` is 0xFF when bit `i` is set, else 0
 *          - the inverse of @ref sz_utf8_movemask16_neon_. Broadcasts the two mask bytes across the two
 *          8-lane halves, ANDs each lane's bit-position {1,2,..,128}, and `vceqq` against it (gather-free, no scalar).*/
SZ_INTERNAL uint8x16_t sz_utf8_expand16_neon_(sz_u32_t submask) {
    static sz_u8_t const bit_position_lanes[16] = {1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128};
    uint8x16_t const bit_position_u8x16 = vld1q_u8(bit_position_lanes);
    uint8x8_t const low_byte_u8x8 = vdup_n_u8((sz_u8_t)(submask & 0xFF));
    uint8x8_t const high_byte_u8x8 = vdup_n_u8((sz_u8_t)((submask >> 8) & 0xFF));
    uint8x16_t const per_half_u8x16 = vcombine_u8(low_byte_u8x8, high_byte_u8x8);
    return vceqq_u8(vandq_u8(per_half_u8x16, bit_position_u8x16), bit_position_u8x16);
}

SZ_INTERNAL sz_size_t sz_utf8_leftpack_offsets_neon_(sz_u64_t mask, sz_u8_t *out) {
    static sz_u8_t const leftpack8[256 * 8] = {
        0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, // 0x00
        0x00, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, // 0x01
        0x01, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, // 0x02
        0x00, 0x01, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, // 0x03
        0x02, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, // 0x04
        0x00, 0x02, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, // 0x05
        0x01, 0x02, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, // 0x06
        0x00, 0x01, 0x02, 0x80, 0x80, 0x80, 0x80, 0x80, // 0x07
        0x03, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, // 0x08
        0x00, 0x03, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, // 0x09
        0x01, 0x03, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, // 0x0A
        0x00, 0x01, 0x03, 0x80, 0x80, 0x80, 0x80, 0x80, // 0x0B
        0x02, 0x03, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, // 0x0C
        0x00, 0x02, 0x03, 0x80, 0x80, 0x80, 0x80, 0x80, // 0x0D
        0x01, 0x02, 0x03, 0x80, 0x80, 0x80, 0x80, 0x80, // 0x0E
        0x00, 0x01, 0x02, 0x03, 0x80, 0x80, 0x80, 0x80, // 0x0F
        0x04, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, // 0x10
        0x00, 0x04, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, // 0x11
        0x01, 0x04, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, // 0x12
        0x00, 0x01, 0x04, 0x80, 0x80, 0x80, 0x80, 0x80, // 0x13
        0x02, 0x04, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, // 0x14
        0x00, 0x02, 0x04, 0x80, 0x80, 0x80, 0x80, 0x80, // 0x15
        0x01, 0x02, 0x04, 0x80, 0x80, 0x80, 0x80, 0x80, // 0x16
        0x00, 0x01, 0x02, 0x04, 0x80, 0x80, 0x80, 0x80, // 0x17
        0x03, 0x04, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, // 0x18
        0x00, 0x03, 0x04, 0x80, 0x80, 0x80, 0x80, 0x80, // 0x19
        0x01, 0x03, 0x04, 0x80, 0x80, 0x80, 0x80, 0x80, // 0x1A
        0x00, 0x01, 0x03, 0x04, 0x80, 0x80, 0x80, 0x80, // 0x1B
        0x02, 0x03, 0x04, 0x80, 0x80, 0x80, 0x80, 0x80, // 0x1C
        0x00, 0x02, 0x03, 0x04, 0x80, 0x80, 0x80, 0x80, // 0x1D
        0x01, 0x02, 0x03, 0x04, 0x80, 0x80, 0x80, 0x80, // 0x1E
        0x00, 0x01, 0x02, 0x03, 0x04, 0x80, 0x80, 0x80, // 0x1F
        0x05, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, // 0x20
        0x00, 0x05, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, // 0x21
        0x01, 0x05, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, // 0x22
        0x00, 0x01, 0x05, 0x80, 0x80, 0x80, 0x80, 0x80, // 0x23
        0x02, 0x05, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, // 0x24
        0x00, 0x02, 0x05, 0x80, 0x80, 0x80, 0x80, 0x80, // 0x25
        0x01, 0x02, 0x05, 0x80, 0x80, 0x80, 0x80, 0x80, // 0x26
        0x00, 0x01, 0x02, 0x05, 0x80, 0x80, 0x80, 0x80, // 0x27
        0x03, 0x05, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, // 0x28
        0x00, 0x03, 0x05, 0x80, 0x80, 0x80, 0x80, 0x80, // 0x29
        0x01, 0x03, 0x05, 0x80, 0x80, 0x80, 0x80, 0x80, // 0x2A
        0x00, 0x01, 0x03, 0x05, 0x80, 0x80, 0x80, 0x80, // 0x2B
        0x02, 0x03, 0x05, 0x80, 0x80, 0x80, 0x80, 0x80, // 0x2C
        0x00, 0x02, 0x03, 0x05, 0x80, 0x80, 0x80, 0x80, // 0x2D
        0x01, 0x02, 0x03, 0x05, 0x80, 0x80, 0x80, 0x80, // 0x2E
        0x00, 0x01, 0x02, 0x03, 0x05, 0x80, 0x80, 0x80, // 0x2F
        0x04, 0x05, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, // 0x30
        0x00, 0x04, 0x05, 0x80, 0x80, 0x80, 0x80, 0x80, // 0x31
        0x01, 0x04, 0x05, 0x80, 0x80, 0x80, 0x80, 0x80, // 0x32
        0x00, 0x01, 0x04, 0x05, 0x80, 0x80, 0x80, 0x80, // 0x33
        0x02, 0x04, 0x05, 0x80, 0x80, 0x80, 0x80, 0x80, // 0x34
        0x00, 0x02, 0x04, 0x05, 0x80, 0x80, 0x80, 0x80, // 0x35
        0x01, 0x02, 0x04, 0x05, 0x80, 0x80, 0x80, 0x80, // 0x36
        0x00, 0x01, 0x02, 0x04, 0x05, 0x80, 0x80, 0x80, // 0x37
        0x03, 0x04, 0x05, 0x80, 0x80, 0x80, 0x80, 0x80, // 0x38
        0x00, 0x03, 0x04, 0x05, 0x80, 0x80, 0x80, 0x80, // 0x39
        0x01, 0x03, 0x04, 0x05, 0x80, 0x80, 0x80, 0x80, // 0x3A
        0x00, 0x01, 0x03, 0x04, 0x05, 0x80, 0x80, 0x80, // 0x3B
        0x02, 0x03, 0x04, 0x05, 0x80, 0x80, 0x80, 0x80, // 0x3C
        0x00, 0x02, 0x03, 0x04, 0x05, 0x80, 0x80, 0x80, // 0x3D
        0x01, 0x02, 0x03, 0x04, 0x05, 0x80, 0x80, 0x80, // 0x3E
        0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x80, 0x80, // 0x3F
        0x06, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, // 0x40
        0x00, 0x06, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, // 0x41
        0x01, 0x06, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, // 0x42
        0x00, 0x01, 0x06, 0x80, 0x80, 0x80, 0x80, 0x80, // 0x43
        0x02, 0x06, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, // 0x44
        0x00, 0x02, 0x06, 0x80, 0x80, 0x80, 0x80, 0x80, // 0x45
        0x01, 0x02, 0x06, 0x80, 0x80, 0x80, 0x80, 0x80, // 0x46
        0x00, 0x01, 0x02, 0x06, 0x80, 0x80, 0x80, 0x80, // 0x47
        0x03, 0x06, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, // 0x48
        0x00, 0x03, 0x06, 0x80, 0x80, 0x80, 0x80, 0x80, // 0x49
        0x01, 0x03, 0x06, 0x80, 0x80, 0x80, 0x80, 0x80, // 0x4A
        0x00, 0x01, 0x03, 0x06, 0x80, 0x80, 0x80, 0x80, // 0x4B
        0x02, 0x03, 0x06, 0x80, 0x80, 0x80, 0x80, 0x80, // 0x4C
        0x00, 0x02, 0x03, 0x06, 0x80, 0x80, 0x80, 0x80, // 0x4D
        0x01, 0x02, 0x03, 0x06, 0x80, 0x80, 0x80, 0x80, // 0x4E
        0x00, 0x01, 0x02, 0x03, 0x06, 0x80, 0x80, 0x80, // 0x4F
        0x04, 0x06, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, // 0x50
        0x00, 0x04, 0x06, 0x80, 0x80, 0x80, 0x80, 0x80, // 0x51
        0x01, 0x04, 0x06, 0x80, 0x80, 0x80, 0x80, 0x80, // 0x52
        0x00, 0x01, 0x04, 0x06, 0x80, 0x80, 0x80, 0x80, // 0x53
        0x02, 0x04, 0x06, 0x80, 0x80, 0x80, 0x80, 0x80, // 0x54
        0x00, 0x02, 0x04, 0x06, 0x80, 0x80, 0x80, 0x80, // 0x55
        0x01, 0x02, 0x04, 0x06, 0x80, 0x80, 0x80, 0x80, // 0x56
        0x00, 0x01, 0x02, 0x04, 0x06, 0x80, 0x80, 0x80, // 0x57
        0x03, 0x04, 0x06, 0x80, 0x80, 0x80, 0x80, 0x80, // 0x58
        0x00, 0x03, 0x04, 0x06, 0x80, 0x80, 0x80, 0x80, // 0x59
        0x01, 0x03, 0x04, 0x06, 0x80, 0x80, 0x80, 0x80, // 0x5A
        0x00, 0x01, 0x03, 0x04, 0x06, 0x80, 0x80, 0x80, // 0x5B
        0x02, 0x03, 0x04, 0x06, 0x80, 0x80, 0x80, 0x80, // 0x5C
        0x00, 0x02, 0x03, 0x04, 0x06, 0x80, 0x80, 0x80, // 0x5D
        0x01, 0x02, 0x03, 0x04, 0x06, 0x80, 0x80, 0x80, // 0x5E
        0x00, 0x01, 0x02, 0x03, 0x04, 0x06, 0x80, 0x80, // 0x5F
        0x05, 0x06, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, // 0x60
        0x00, 0x05, 0x06, 0x80, 0x80, 0x80, 0x80, 0x80, // 0x61
        0x01, 0x05, 0x06, 0x80, 0x80, 0x80, 0x80, 0x80, // 0x62
        0x00, 0x01, 0x05, 0x06, 0x80, 0x80, 0x80, 0x80, // 0x63
        0x02, 0x05, 0x06, 0x80, 0x80, 0x80, 0x80, 0x80, // 0x64
        0x00, 0x02, 0x05, 0x06, 0x80, 0x80, 0x80, 0x80, // 0x65
        0x01, 0x02, 0x05, 0x06, 0x80, 0x80, 0x80, 0x80, // 0x66
        0x00, 0x01, 0x02, 0x05, 0x06, 0x80, 0x80, 0x80, // 0x67
        0x03, 0x05, 0x06, 0x80, 0x80, 0x80, 0x80, 0x80, // 0x68
        0x00, 0x03, 0x05, 0x06, 0x80, 0x80, 0x80, 0x80, // 0x69
        0x01, 0x03, 0x05, 0x06, 0x80, 0x80, 0x80, 0x80, // 0x6A
        0x00, 0x01, 0x03, 0x05, 0x06, 0x80, 0x80, 0x80, // 0x6B
        0x02, 0x03, 0x05, 0x06, 0x80, 0x80, 0x80, 0x80, // 0x6C
        0x00, 0x02, 0x03, 0x05, 0x06, 0x80, 0x80, 0x80, // 0x6D
        0x01, 0x02, 0x03, 0x05, 0x06, 0x80, 0x80, 0x80, // 0x6E
        0x00, 0x01, 0x02, 0x03, 0x05, 0x06, 0x80, 0x80, // 0x6F
        0x04, 0x05, 0x06, 0x80, 0x80, 0x80, 0x80, 0x80, // 0x70
        0x00, 0x04, 0x05, 0x06, 0x80, 0x80, 0x80, 0x80, // 0x71
        0x01, 0x04, 0x05, 0x06, 0x80, 0x80, 0x80, 0x80, // 0x72
        0x00, 0x01, 0x04, 0x05, 0x06, 0x80, 0x80, 0x80, // 0x73
        0x02, 0x04, 0x05, 0x06, 0x80, 0x80, 0x80, 0x80, // 0x74
        0x00, 0x02, 0x04, 0x05, 0x06, 0x80, 0x80, 0x80, // 0x75
        0x01, 0x02, 0x04, 0x05, 0x06, 0x80, 0x80, 0x80, // 0x76
        0x00, 0x01, 0x02, 0x04, 0x05, 0x06, 0x80, 0x80, // 0x77
        0x03, 0x04, 0x05, 0x06, 0x80, 0x80, 0x80, 0x80, // 0x78
        0x00, 0x03, 0x04, 0x05, 0x06, 0x80, 0x80, 0x80, // 0x79
        0x01, 0x03, 0x04, 0x05, 0x06, 0x80, 0x80, 0x80, // 0x7A
        0x00, 0x01, 0x03, 0x04, 0x05, 0x06, 0x80, 0x80, // 0x7B
        0x02, 0x03, 0x04, 0x05, 0x06, 0x80, 0x80, 0x80, // 0x7C
        0x00, 0x02, 0x03, 0x04, 0x05, 0x06, 0x80, 0x80, // 0x7D
        0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x80, 0x80, // 0x7E
        0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x80, // 0x7F
        0x07, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, // 0x80
        0x00, 0x07, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, // 0x81
        0x01, 0x07, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, // 0x82
        0x00, 0x01, 0x07, 0x80, 0x80, 0x80, 0x80, 0x80, // 0x83
        0x02, 0x07, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, // 0x84
        0x00, 0x02, 0x07, 0x80, 0x80, 0x80, 0x80, 0x80, // 0x85
        0x01, 0x02, 0x07, 0x80, 0x80, 0x80, 0x80, 0x80, // 0x86
        0x00, 0x01, 0x02, 0x07, 0x80, 0x80, 0x80, 0x80, // 0x87
        0x03, 0x07, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, // 0x88
        0x00, 0x03, 0x07, 0x80, 0x80, 0x80, 0x80, 0x80, // 0x89
        0x01, 0x03, 0x07, 0x80, 0x80, 0x80, 0x80, 0x80, // 0x8A
        0x00, 0x01, 0x03, 0x07, 0x80, 0x80, 0x80, 0x80, // 0x8B
        0x02, 0x03, 0x07, 0x80, 0x80, 0x80, 0x80, 0x80, // 0x8C
        0x00, 0x02, 0x03, 0x07, 0x80, 0x80, 0x80, 0x80, // 0x8D
        0x01, 0x02, 0x03, 0x07, 0x80, 0x80, 0x80, 0x80, // 0x8E
        0x00, 0x01, 0x02, 0x03, 0x07, 0x80, 0x80, 0x80, // 0x8F
        0x04, 0x07, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, // 0x90
        0x00, 0x04, 0x07, 0x80, 0x80, 0x80, 0x80, 0x80, // 0x91
        0x01, 0x04, 0x07, 0x80, 0x80, 0x80, 0x80, 0x80, // 0x92
        0x00, 0x01, 0x04, 0x07, 0x80, 0x80, 0x80, 0x80, // 0x93
        0x02, 0x04, 0x07, 0x80, 0x80, 0x80, 0x80, 0x80, // 0x94
        0x00, 0x02, 0x04, 0x07, 0x80, 0x80, 0x80, 0x80, // 0x95
        0x01, 0x02, 0x04, 0x07, 0x80, 0x80, 0x80, 0x80, // 0x96
        0x00, 0x01, 0x02, 0x04, 0x07, 0x80, 0x80, 0x80, // 0x97
        0x03, 0x04, 0x07, 0x80, 0x80, 0x80, 0x80, 0x80, // 0x98
        0x00, 0x03, 0x04, 0x07, 0x80, 0x80, 0x80, 0x80, // 0x99
        0x01, 0x03, 0x04, 0x07, 0x80, 0x80, 0x80, 0x80, // 0x9A
        0x00, 0x01, 0x03, 0x04, 0x07, 0x80, 0x80, 0x80, // 0x9B
        0x02, 0x03, 0x04, 0x07, 0x80, 0x80, 0x80, 0x80, // 0x9C
        0x00, 0x02, 0x03, 0x04, 0x07, 0x80, 0x80, 0x80, // 0x9D
        0x01, 0x02, 0x03, 0x04, 0x07, 0x80, 0x80, 0x80, // 0x9E
        0x00, 0x01, 0x02, 0x03, 0x04, 0x07, 0x80, 0x80, // 0x9F
        0x05, 0x07, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, // 0xA0
        0x00, 0x05, 0x07, 0x80, 0x80, 0x80, 0x80, 0x80, // 0xA1
        0x01, 0x05, 0x07, 0x80, 0x80, 0x80, 0x80, 0x80, // 0xA2
        0x00, 0x01, 0x05, 0x07, 0x80, 0x80, 0x80, 0x80, // 0xA3
        0x02, 0x05, 0x07, 0x80, 0x80, 0x80, 0x80, 0x80, // 0xA4
        0x00, 0x02, 0x05, 0x07, 0x80, 0x80, 0x80, 0x80, // 0xA5
        0x01, 0x02, 0x05, 0x07, 0x80, 0x80, 0x80, 0x80, // 0xA6
        0x00, 0x01, 0x02, 0x05, 0x07, 0x80, 0x80, 0x80, // 0xA7
        0x03, 0x05, 0x07, 0x80, 0x80, 0x80, 0x80, 0x80, // 0xA8
        0x00, 0x03, 0x05, 0x07, 0x80, 0x80, 0x80, 0x80, // 0xA9
        0x01, 0x03, 0x05, 0x07, 0x80, 0x80, 0x80, 0x80, // 0xAA
        0x00, 0x01, 0x03, 0x05, 0x07, 0x80, 0x80, 0x80, // 0xAB
        0x02, 0x03, 0x05, 0x07, 0x80, 0x80, 0x80, 0x80, // 0xAC
        0x00, 0x02, 0x03, 0x05, 0x07, 0x80, 0x80, 0x80, // 0xAD
        0x01, 0x02, 0x03, 0x05, 0x07, 0x80, 0x80, 0x80, // 0xAE
        0x00, 0x01, 0x02, 0x03, 0x05, 0x07, 0x80, 0x80, // 0xAF
        0x04, 0x05, 0x07, 0x80, 0x80, 0x80, 0x80, 0x80, // 0xB0
        0x00, 0x04, 0x05, 0x07, 0x80, 0x80, 0x80, 0x80, // 0xB1
        0x01, 0x04, 0x05, 0x07, 0x80, 0x80, 0x80, 0x80, // 0xB2
        0x00, 0x01, 0x04, 0x05, 0x07, 0x80, 0x80, 0x80, // 0xB3
        0x02, 0x04, 0x05, 0x07, 0x80, 0x80, 0x80, 0x80, // 0xB4
        0x00, 0x02, 0x04, 0x05, 0x07, 0x80, 0x80, 0x80, // 0xB5
        0x01, 0x02, 0x04, 0x05, 0x07, 0x80, 0x80, 0x80, // 0xB6
        0x00, 0x01, 0x02, 0x04, 0x05, 0x07, 0x80, 0x80, // 0xB7
        0x03, 0x04, 0x05, 0x07, 0x80, 0x80, 0x80, 0x80, // 0xB8
        0x00, 0x03, 0x04, 0x05, 0x07, 0x80, 0x80, 0x80, // 0xB9
        0x01, 0x03, 0x04, 0x05, 0x07, 0x80, 0x80, 0x80, // 0xBA
        0x00, 0x01, 0x03, 0x04, 0x05, 0x07, 0x80, 0x80, // 0xBB
        0x02, 0x03, 0x04, 0x05, 0x07, 0x80, 0x80, 0x80, // 0xBC
        0x00, 0x02, 0x03, 0x04, 0x05, 0x07, 0x80, 0x80, // 0xBD
        0x01, 0x02, 0x03, 0x04, 0x05, 0x07, 0x80, 0x80, // 0xBE
        0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x07, 0x80, // 0xBF
        0x06, 0x07, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, // 0xC0
        0x00, 0x06, 0x07, 0x80, 0x80, 0x80, 0x80, 0x80, // 0xC1
        0x01, 0x06, 0x07, 0x80, 0x80, 0x80, 0x80, 0x80, // 0xC2
        0x00, 0x01, 0x06, 0x07, 0x80, 0x80, 0x80, 0x80, // 0xC3
        0x02, 0x06, 0x07, 0x80, 0x80, 0x80, 0x80, 0x80, // 0xC4
        0x00, 0x02, 0x06, 0x07, 0x80, 0x80, 0x80, 0x80, // 0xC5
        0x01, 0x02, 0x06, 0x07, 0x80, 0x80, 0x80, 0x80, // 0xC6
        0x00, 0x01, 0x02, 0x06, 0x07, 0x80, 0x80, 0x80, // 0xC7
        0x03, 0x06, 0x07, 0x80, 0x80, 0x80, 0x80, 0x80, // 0xC8
        0x00, 0x03, 0x06, 0x07, 0x80, 0x80, 0x80, 0x80, // 0xC9
        0x01, 0x03, 0x06, 0x07, 0x80, 0x80, 0x80, 0x80, // 0xCA
        0x00, 0x01, 0x03, 0x06, 0x07, 0x80, 0x80, 0x80, // 0xCB
        0x02, 0x03, 0x06, 0x07, 0x80, 0x80, 0x80, 0x80, // 0xCC
        0x00, 0x02, 0x03, 0x06, 0x07, 0x80, 0x80, 0x80, // 0xCD
        0x01, 0x02, 0x03, 0x06, 0x07, 0x80, 0x80, 0x80, // 0xCE
        0x00, 0x01, 0x02, 0x03, 0x06, 0x07, 0x80, 0x80, // 0xCF
        0x04, 0x06, 0x07, 0x80, 0x80, 0x80, 0x80, 0x80, // 0xD0
        0x00, 0x04, 0x06, 0x07, 0x80, 0x80, 0x80, 0x80, // 0xD1
        0x01, 0x04, 0x06, 0x07, 0x80, 0x80, 0x80, 0x80, // 0xD2
        0x00, 0x01, 0x04, 0x06, 0x07, 0x80, 0x80, 0x80, // 0xD3
        0x02, 0x04, 0x06, 0x07, 0x80, 0x80, 0x80, 0x80, // 0xD4
        0x00, 0x02, 0x04, 0x06, 0x07, 0x80, 0x80, 0x80, // 0xD5
        0x01, 0x02, 0x04, 0x06, 0x07, 0x80, 0x80, 0x80, // 0xD6
        0x00, 0x01, 0x02, 0x04, 0x06, 0x07, 0x80, 0x80, // 0xD7
        0x03, 0x04, 0x06, 0x07, 0x80, 0x80, 0x80, 0x80, // 0xD8
        0x00, 0x03, 0x04, 0x06, 0x07, 0x80, 0x80, 0x80, // 0xD9
        0x01, 0x03, 0x04, 0x06, 0x07, 0x80, 0x80, 0x80, // 0xDA
        0x00, 0x01, 0x03, 0x04, 0x06, 0x07, 0x80, 0x80, // 0xDB
        0x02, 0x03, 0x04, 0x06, 0x07, 0x80, 0x80, 0x80, // 0xDC
        0x00, 0x02, 0x03, 0x04, 0x06, 0x07, 0x80, 0x80, // 0xDD
        0x01, 0x02, 0x03, 0x04, 0x06, 0x07, 0x80, 0x80, // 0xDE
        0x00, 0x01, 0x02, 0x03, 0x04, 0x06, 0x07, 0x80, // 0xDF
        0x05, 0x06, 0x07, 0x80, 0x80, 0x80, 0x80, 0x80, // 0xE0
        0x00, 0x05, 0x06, 0x07, 0x80, 0x80, 0x80, 0x80, // 0xE1
        0x01, 0x05, 0x06, 0x07, 0x80, 0x80, 0x80, 0x80, // 0xE2
        0x00, 0x01, 0x05, 0x06, 0x07, 0x80, 0x80, 0x80, // 0xE3
        0x02, 0x05, 0x06, 0x07, 0x80, 0x80, 0x80, 0x80, // 0xE4
        0x00, 0x02, 0x05, 0x06, 0x07, 0x80, 0x80, 0x80, // 0xE5
        0x01, 0x02, 0x05, 0x06, 0x07, 0x80, 0x80, 0x80, // 0xE6
        0x00, 0x01, 0x02, 0x05, 0x06, 0x07, 0x80, 0x80, // 0xE7
        0x03, 0x05, 0x06, 0x07, 0x80, 0x80, 0x80, 0x80, // 0xE8
        0x00, 0x03, 0x05, 0x06, 0x07, 0x80, 0x80, 0x80, // 0xE9
        0x01, 0x03, 0x05, 0x06, 0x07, 0x80, 0x80, 0x80, // 0xEA
        0x00, 0x01, 0x03, 0x05, 0x06, 0x07, 0x80, 0x80, // 0xEB
        0x02, 0x03, 0x05, 0x06, 0x07, 0x80, 0x80, 0x80, // 0xEC
        0x00, 0x02, 0x03, 0x05, 0x06, 0x07, 0x80, 0x80, // 0xED
        0x01, 0x02, 0x03, 0x05, 0x06, 0x07, 0x80, 0x80, // 0xEE
        0x00, 0x01, 0x02, 0x03, 0x05, 0x06, 0x07, 0x80, // 0xEF
        0x04, 0x05, 0x06, 0x07, 0x80, 0x80, 0x80, 0x80, // 0xF0
        0x00, 0x04, 0x05, 0x06, 0x07, 0x80, 0x80, 0x80, // 0xF1
        0x01, 0x04, 0x05, 0x06, 0x07, 0x80, 0x80, 0x80, // 0xF2
        0x00, 0x01, 0x04, 0x05, 0x06, 0x07, 0x80, 0x80, // 0xF3
        0x02, 0x04, 0x05, 0x06, 0x07, 0x80, 0x80, 0x80, // 0xF4
        0x00, 0x02, 0x04, 0x05, 0x06, 0x07, 0x80, 0x80, // 0xF5
        0x01, 0x02, 0x04, 0x05, 0x06, 0x07, 0x80, 0x80, // 0xF6
        0x00, 0x01, 0x02, 0x04, 0x05, 0x06, 0x07, 0x80, // 0xF7
        0x03, 0x04, 0x05, 0x06, 0x07, 0x80, 0x80, 0x80, // 0xF8
        0x00, 0x03, 0x04, 0x05, 0x06, 0x07, 0x80, 0x80, // 0xF9
        0x01, 0x03, 0x04, 0x05, 0x06, 0x07, 0x80, 0x80, // 0xFA
        0x00, 0x01, 0x03, 0x04, 0x05, 0x06, 0x07, 0x80, // 0xFB
        0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x80, 0x80, // 0xFC
        0x00, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x80, // 0xFD
        0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x80, // 0xFE
        0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, // 0xFF
    };
    static sz_u8_t const iota16_lanes[16] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
    uint8x16_t const iota16_u8x16 = vld1q_u8(iota16_lanes);
    uint8x16_t const eight_u8x16 = vdupq_n_u8(8);

    sz_size_t produced = 0;
    for (int quarter = 0; quarter < 4; ++quarter) {
        sz_u32_t const quarter_mask = (sz_u32_t)((mask >> (quarter * 16)) & 0xFFFFu);
        if (quarter_mask == 0) continue;
        sz_u32_t const low8 = quarter_mask & 0xFFu;
        sz_u32_t const high8 = (quarter_mask >> 8) & 0xFFu;
        int const count_low = sz_u64_popcount(low8);
        int const count = sz_u64_popcount(quarter_mask);

        // Row of ascending set-bit positions for each 8-bit half; the high half's positions are +8 (lanes 8..15),
        // valid slots filled low, unused slots = 0x80 (out of `vqtbl1q_u8` range -> read as 0, never stored).
        uint8x8_t const packed_low_u8x8 = vld1_u8(leftpack8 + low8 * 8);
        uint8x8_t const packed_high_u8x8 = vadd_u8(vld1_u8(leftpack8 + high8 * 8), vdup_n_u8(8));
        uint8x16_t const halves_u8x16 = vcombine_u8(packed_low_u8x8, packed_high_u8x8);

        // Stitch: output lane j reads source j for j < count_low, else j + (8 - count_low) (skip the low gap).
        uint8x16_t const ge_u8x16 = vcgeq_u8(iota16_u8x16, vdupq_n_u8((sz_u8_t)count_low));
        uint8x16_t const gap_u8x16 = vandq_u8(ge_u8x16, vsubq_u8(eight_u8x16, vdupq_n_u8((sz_u8_t)count_low)));
        uint8x16_t const source_u8x16 = vaddq_u8(iota16_u8x16, gap_u8x16);
        uint8x16_t const stitched_u8x16 = vqtbl1q_u8(halves_u8x16, source_u8x16);

        // Promote the quarter-relative offsets to global byte-offsets and append the `count` dense lanes.
        uint8x16_t const global_u8x16 = vaddq_u8(stitched_u8x16, vdupq_n_u8((sz_u8_t)(quarter * 16)));
        sz_u8_t lanes[16];
        vst1q_u8(lanes, global_u8x16);
        for (int lane = 0; lane < count; ++lane) out[produced + (sz_size_t)lane] = lanes[lane];
        produced += (sz_size_t)count;
    }
    return produced;
}

/** @brief  NEON forward drain — the `vpcompressb`-free twin of @ref sz_utf8_rune_drain_forward_. Emits one
 *          (start, length) per set boundary lane (ascending), honoring @p capacity and the carried previous-boundary
 *          via @p previous_io; bit-exact with the Ice Lake leaf. Indices are unpacked once, then each segment's
 *          (start, length) is computed from the carried previous boundary. */
SZ_INTERNAL sz_size_t sz_utf8_rune_drain_forward_neon_( //
    sz_u64_t boundary, sz_size_t base, sz_size_t *starts, sz_size_t *lengths, sz_size_t produced, sz_size_t capacity,
    sz_size_t *previous_io) {
    sz_size_t const boundary_count = (sz_size_t)sz_u64_popcount(boundary);
    sz_size_t previous = *previous_io;
    if (boundary_count == 0 || produced >= capacity) {
        *previous_io = previous;
        return produced;
    }

    sz_u8_t indices[64];
    sz_utf8_unpack_indices_neon_(boundary, indices);

    sz_size_t emitted = 0;
    while (emitted < boundary_count && produced < capacity) {
        sz_size_t const position = base + (sz_size_t)indices[emitted];
        starts[produced] = previous;
        lengths[produced] = position - previous;
        previous = position;
        ++produced, ++emitted;
    }
    *previous_io = previous;
    return produced;
}

/**
 *  @brief  Decode the dense set of emitted-start lanes @p emit_starts of a classified window into sequential UTF-32
 *          runes — the NEON twin of @ref sz_utf8_rune_drain_icelake_, bit-for-bit. The start byte-offsets
 *          are left-packed once via the shuffle-LUT compaction (@ref sz_utf8_leftpack_offsets_neon_); three
 *          sibling `vqtbl4q_u8` gathers pull the lead + up to three trailing bytes per codepoint over the 64-byte
 *          @p window (four quarters), and a branchless 1/2/3/4-byte width-blend assembles each value.
 *
 *  TOTAL decode: @p emit_starts also covers promoted orphan continuation bytes, and @p ill_formed marks every start
 *  lane whose maximal ill-formed subpart collapses to one U+FFFD (Unicode 17.0 §3.9 / W3C). The per-lane ill flag is
 *  gathered with the SAME packed indices (a fourth `vqtbl4q_u8`) so ill-formed output lanes are blended to U+FFFD in
 *  register. The resume cursor reads @p consumed_length (per-lane maximal-subpart byte length, in window order) at the
 *  last emitted start, so an ill-formed trailing lane never skips bytes that owe their own next-window U+FFFD.
 *
 *  @param  window           The 64 raw input bytes, as four `uint8x16_t` quarters (lanes [16*q, 16*q+16)).
 *  @param  emit_starts      Bit `i` => lane `i` emits one rune (a codepoint start or a promoted orphan continuation).
 *  @param  ill_byte         Per-lane 0xFF where the emitted start is ill-formed (one U+FFFD), 0 otherwise; four quarters.
 *  @param  consumed_length  Per-lane maximal-subpart byte length (1..4) at each start; flat 64-byte window order.
 *  @param  emit_count       Popcount of @p emit_starts (number of runes available to emit).
 *  @param  consumed_bytes   Set to the byte span the emitted runes cover (the resume cursor delta).
 *  @return Number of runes emitted (<= min(emit_count, capacity)).
 */
SZ_INTERNAL sz_size_t sz_utf8_rune_drain_neon_(                                                                 //
    uint8x16_t const *window, sz_u64_t emit_starts, uint8x16_t const *ill_byte, sz_u8_t const *consumed_length, //
    sz_size_t emit_count, sz_rune_t *runes, sz_size_t capacity, sz_size_t *consumed_bytes) {

    // Build the 64-byte tables once: `vqtbl4q_u8` addresses all four quarters as one `uint8x16x4_t`.
    uint8x16x4_t table_u8x16x4, ill_table_u8x16x4;
    table_u8x16x4.val[0] = window[0], table_u8x16x4.val[1] = window[1];
    table_u8x16x4.val[2] = window[2], table_u8x16x4.val[3] = window[3];
    ill_table_u8x16x4.val[0] = ill_byte[0], ill_table_u8x16x4.val[1] = ill_byte[1];
    ill_table_u8x16x4.val[2] = ill_byte[2], ill_table_u8x16x4.val[3] = ill_byte[3];

    // Left-pack the emitted start byte-offsets (ascending) into a dense array via the shuffle-LUT compaction.
    sz_u8_t start_offsets[64];
    sz_utf8_leftpack_offsets_neon_(emit_starts, start_offsets);

    sz_size_t const want = emit_count < capacity ? emit_count : capacity;
    sz_size_t produced = 0;
    uint8x16_t const v1_u8x16 = vdupq_n_u8(1), v2_u8x16 = vdupq_n_u8(2), v3_u8x16 = vdupq_n_u8(3);
    uint32x4_t const replacement_u32x4 = vdupq_n_u32((sz_u32_t)sz_rune_replacement_k);
    // One 16-codepoint block per outer step; one 4-lane sub-block per inner widen-and-blend.
    for (sz_size_t block_start = 0; block_start < want; block_start += 16) {
        // Load up to 16 start offsets for this block into one index vector; trailing lanes index >= 64 read as 0.
        sz_u8_t index_lanes[16];
        for (int lane = 0; lane < 16; ++lane) {
            sz_size_t const source = block_start + (sz_size_t)lane;
            index_lanes[lane] = source < want ? start_offsets[source] : (sz_u8_t)0xFF;
        }
        uint8x16_t const index_u8x16 = vld1q_u8(index_lanes);

        // Gather the lead and all three trailing bytes per codepoint in one table read each (zero past the window).
        uint8x16_t const lead_u8x16 = vqtbl4q_u8(table_u8x16x4, index_u8x16);
        uint8x16_t const byte1_u8x16 = vqtbl4q_u8(table_u8x16x4, vaddq_u8(index_u8x16, v1_u8x16));
        uint8x16_t const byte2_u8x16 = vqtbl4q_u8(table_u8x16x4, vaddq_u8(index_u8x16, v2_u8x16));
        uint8x16_t const byte3_u8x16 = vqtbl4q_u8(table_u8x16x4, vaddq_u8(index_u8x16, v3_u8x16));
        // Gather the per-emitted-lane ill-formed flag with the SAME packed indices (0xFF => U+FFFD blend).
        uint8x16_t const ill_u8x16 = vqtbl4q_u8(ill_table_u8x16x4, index_u8x16);

        // Widen all 16 byte lanes to 32-bit codepoint lanes via the two-step u8->u16->u32 ladder (four sub-blocks).
        uint16x8_t const lead_lo_u16x8 = vmovl_u8(vget_low_u8(lead_u8x16)), lead_hi_u16x8 = vmovl_u8(vget_high_u8(lead_u8x16));
        uint16x8_t const b1_lo_u16x8 = vmovl_u8(vget_low_u8(byte1_u8x16)), b1_hi_u16x8 = vmovl_u8(vget_high_u8(byte1_u8x16));
        uint16x8_t const b2_lo_u16x8 = vmovl_u8(vget_low_u8(byte2_u8x16)), b2_hi_u16x8 = vmovl_u8(vget_high_u8(byte2_u8x16));
        uint16x8_t const b3_lo_u16x8 = vmovl_u8(vget_low_u8(byte3_u8x16)), b3_hi_u16x8 = vmovl_u8(vget_high_u8(byte3_u8x16));
        uint16x8_t const ill_lo_u16x8 = vmovl_u8(vget_low_u8(ill_u8x16)), ill_hi_u16x8 = vmovl_u8(vget_high_u8(ill_u8x16));
        uint32x4_t const lead32_u32x4[4] = {vmovl_u16(vget_low_u16(lead_lo_u16x8)), vmovl_u16(vget_high_u16(lead_lo_u16x8)),
                                            vmovl_u16(vget_low_u16(lead_hi_u16x8)), vmovl_u16(vget_high_u16(lead_hi_u16x8))};
        uint32x4_t const b1_32_u32x4[4] = {vmovl_u16(vget_low_u16(b1_lo_u16x8)), vmovl_u16(vget_high_u16(b1_lo_u16x8)),
                                           vmovl_u16(vget_low_u16(b1_hi_u16x8)), vmovl_u16(vget_high_u16(b1_hi_u16x8))};
        uint32x4_t const b2_32_u32x4[4] = {vmovl_u16(vget_low_u16(b2_lo_u16x8)), vmovl_u16(vget_high_u16(b2_lo_u16x8)),
                                           vmovl_u16(vget_low_u16(b2_hi_u16x8)), vmovl_u16(vget_high_u16(b2_hi_u16x8))};
        uint32x4_t const b3_32_u32x4[4] = {vmovl_u16(vget_low_u16(b3_lo_u16x8)), vmovl_u16(vget_high_u16(b3_lo_u16x8)),
                                           vmovl_u16(vget_low_u16(b3_hi_u16x8)), vmovl_u16(vget_high_u16(b3_hi_u16x8))};
        uint32x4_t const ill32_u32x4[4] = {vmovl_u16(vget_low_u16(ill_lo_u16x8)), vmovl_u16(vget_high_u16(ill_lo_u16x8)),
                                           vmovl_u16(vget_low_u16(ill_hi_u16x8)), vmovl_u16(vget_high_u16(ill_hi_u16x8))};

        for (int sub = 0; sub < 4 && (block_start + (sz_size_t)sub * 4) < want; ++sub) {
            sz_size_t const out_base = block_start + (sz_size_t)sub * 4;
            uint32x4_t const lead_v_u32x4 = lead32_u32x4[sub], b1_v_u32x4 = b1_32_u32x4[sub];
            uint32x4_t const b2_v_u32x4 = b2_32_u32x4[sub], b3_v_u32x4 = b3_32_u32x4[sub];
            uint32x4_t const mask_0x3f_u32x4 = vdupq_n_u32(0x3F);

            /*  Compute every sequence width for all four lanes, then select by the lead-byte range. A lane whose lead
             *  is ASCII / 2-byte never satisfies the 3- or 4-byte range, so the (harmless) wider gathers are discarded
             *  by the blend — bit-identical to the gated nesting, just branchless. */
            uint32x4_t const two_byte_u32x4 =
                vorrq_u32(vshlq_n_u32(vandq_u32(lead_v_u32x4, vdupq_n_u32(0x1F)), 6), vandq_u32(b1_v_u32x4, mask_0x3f_u32x4));
            uint32x4_t const three_byte_u32x4 =
                vorrq_u32(vorrq_u32(vshlq_n_u32(vandq_u32(lead_v_u32x4, vdupq_n_u32(0x0F)), 12),
                                    vshlq_n_u32(vandq_u32(b1_v_u32x4, mask_0x3f_u32x4), 6)),
                          vandq_u32(b2_v_u32x4, mask_0x3f_u32x4));
            uint32x4_t const four_byte_u32x4 =
                vorrq_u32(vorrq_u32(vshlq_n_u32(vandq_u32(lead_v_u32x4, vdupq_n_u32(0x07)), 18),
                                    vshlq_n_u32(vandq_u32(b1_v_u32x4, mask_0x3f_u32x4), 12)),
                          vorrq_u32(vshlq_n_u32(vandq_u32(b2_v_u32x4, mask_0x3f_u32x4), 6), vandq_u32(b3_v_u32x4, mask_0x3f_u32x4)));

            uint32x4_t const is_two_u32x4 =
                vandq_u32(vcgeq_u32(lead_v_u32x4, vdupq_n_u32(0xC0)), vcltq_u32(lead_v_u32x4, vdupq_n_u32(0xE0)));
            uint32x4_t const is_three_u32x4 =
                vandq_u32(vcgeq_u32(lead_v_u32x4, vdupq_n_u32(0xE0)), vcltq_u32(lead_v_u32x4, vdupq_n_u32(0xF0)));
            uint32x4_t const is_four_u32x4 = vcgeq_u32(lead_v_u32x4, vdupq_n_u32(0xF0));

            uint32x4_t codepoints_u32x4 = vbslq_u32(is_two_u32x4, two_byte_u32x4, lead_v_u32x4);
            codepoints_u32x4 = vbslq_u32(is_three_u32x4, three_byte_u32x4, codepoints_u32x4);
            codepoints_u32x4 = vbslq_u32(is_four_u32x4, four_byte_u32x4, codepoints_u32x4);

            // Overwrite every ill-formed lane in this sub-block with U+FFFD (the garbage decode is discarded).
            uint32x4_t const ill_mask_u32x4 = vtstq_u32(ill32_u32x4[sub], ill32_u32x4[sub]);
            codepoints_u32x4 = vbslq_u32(ill_mask_u32x4, replacement_u32x4, codepoints_u32x4);

            sz_size_t lanes = want - out_base;
            if (lanes > 4) lanes = 4;
            if (lanes == 4) { vst1q_u32((sz_u32_t *)(runes + out_base), codepoints_u32x4); }
            else {
                sz_u32_t lane_values[4];
                vst1q_u32(lane_values, codepoints_u32x4);
                for (sz_size_t lane = 0; lane < lanes; ++lane) runes[out_base + lane] = (sz_rune_t)lane_values[lane];
            }
            produced += lanes;
        }
    }

    // Resume cursor delta = the last emitted start's byte offset + its maximal-subpart length (window order). For a
    // well-formed lane this equals the declared rune length; for an ill-formed lane it is the 1-3 byte subpart, so the
    // cursor never skips bytes that must become their own next-window U+FFFD.
    sz_u8_t const last_off = start_offsets[produced - 1];
    *consumed_bytes = (sz_size_t)last_off + (sz_size_t)consumed_length[last_off];
    return produced;
}

/**
 *  @brief  Decode one <=64-byte window of @p text into dense UTF-32 @p runes by the uniform "classify -> whole-window
 *          soundness gate -> compress starts -> gather -> width-blend" path, the NEON twin of @ref
 *          sz_utf8_decode_once_icelake_. Pure ASCII takes a dedicated widen lane. An ill-formed (overlong /
 *          surrogate / out-of-range / framing) or truncated-only window declines (`*runes_unpacked == 0`, cursor
 *          unchanged) and the public entry hands the remainder to the serial reference (the U+FFFD oracle).
 */
SZ_INTERNAL sz_cptr_t sz_utf8_decode_once_neon_( //
    sz_cptr_t text, sz_size_t length,            //
    sz_rune_t *runes, sz_size_t runes_capacity,  //
    sz_size_t *runes_unpacked) {

    sz_size_t const chunk = length < 64 ? length : 64;
    uint8x16_t window_u8x16[4];
    sz_utf8_load_window_neon_((sz_u8_t const *)text, chunk, window_u8x16);
    sz_u64_t const loaded_mask = sz_u64_mask_until_serial_(chunk);

    // ASCII fast lane: a full window of 1-byte runes widens directly, no classification needed. A byte is non-ASCII
    // iff its sign bit is set, i.e. as a signed byte `< 0`; one signed compare replaces AND + compare per quarter.
    int8x16_t const zero_s8x16 = vdupq_n_s8(0);
    sz_u64_t const high_bit = sz_utf8_mask_combine_neon_(vcltq_s8(vreinterpretq_s8_u8(window_u8x16[0]), zero_s8x16),
                                                         vcltq_s8(vreinterpretq_s8_u8(window_u8x16[1]), zero_s8x16),
                                                         vcltq_s8(vreinterpretq_s8_u8(window_u8x16[2]), zero_s8x16),
                                                         vcltq_s8(vreinterpretq_s8_u8(window_u8x16[3]), zero_s8x16)) &
                              loaded_mask;
    if (high_bit == 0) {
        sz_size_t const runes_to_unpack = chunk < runes_capacity ? chunk : runes_capacity;
        for (sz_size_t quarter = 0; quarter * 16 < runes_to_unpack; ++quarter) {
            uint16x8_t const lo_u16x8 = vmovl_u8(vget_low_u8(window_u8x16[quarter]));
            uint16x8_t const hi_u16x8 = vmovl_u8(vget_high_u8(window_u8x16[quarter]));
            uint32x4_t const w_u32x4[4] = {vmovl_u16(vget_low_u16(lo_u16x8)), vmovl_u16(vget_high_u16(lo_u16x8)),
                                           vmovl_u16(vget_low_u16(hi_u16x8)), vmovl_u16(vget_high_u16(hi_u16x8))};
            for (int sub = 0; sub < 4; ++sub) {
                sz_size_t const out_base = quarter * 16 + (sz_size_t)sub * 4;
                if (out_base >= runes_to_unpack) break;
                sz_size_t lanes = runes_to_unpack - out_base;
                if (lanes >= 4) { vst1q_u32((sz_u32_t *)(runes + out_base), w_u32x4[sub]); }
                else {
                    sz_u32_t lane_values[4];
                    vst1q_u32(lane_values, w_u32x4[sub]);
                    for (sz_size_t lane = 0; lane < lanes; ++lane)
                        runes[out_base + lane] = (sz_rune_t)lane_values[lane];
                }
            }
        }
        *runes_unpacked = runes_to_unpack;
        return text + runes_to_unpack;
    }

    // Single-source classification: a high-nibble LUT gives the per-lane byte length (the SAME LUT the serial
    // reference uses, so 0xF8..0xFF map to length 4 and cannot slip the gate). Derive `len == k` masks per lane.
    static sz_u8_t const length_by_high_nibble[16] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 3, 4};
    uint8x16_t const length_lut_u8x16 = vld1q_u8(length_by_high_nibble);
    sz_u64_t starts_bits = 0, continuation_bits = 0;
    sz_u64_t len_eq2 = 0, len_eq3 = 0, len_eq4 = 0;
    sz_u64_t lead_lt_c2 = 0, lead_gt_f4 = 0;
    sz_u64_t e0 = 0, ed = 0, f0 = 0, f4 = 0;
    // Branchless overrun scan: per lane the global byte index `quarter*16 + within`; a start whose declared
    // `index + length > chunk` reaches past the window. The first such start (lowest index) bounds the decodable
    // prefix. Lanes that are not overrunning starts contribute 0xFF, so `vminvq_u8` returns either the first
    // overrunning lane index or 0xFF when the quarter has none; the per-quarter scalar minima combine to the global.
    static sz_u8_t const within_iota_lanes[16] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
    uint8x16_t const within_iota_u8x16 = vld1q_u8(within_iota_lanes);
    uint8x16_t const chunk_broadcast_u8x16 = vdupq_n_u8((sz_u8_t)chunk);
    sz_size_t overrun_position = chunk; // No overrunning start -> the decodable prefix spans the whole window.
    for (int quarter = 0; quarter < 4; ++quarter) {
        uint8x16_t const here_u8x16 = window_u8x16[quarter];
        uint8x16_t const high_nibble_u8x16 = vshrq_n_u8(here_u8x16, 4);
        uint8x16_t const len_u8x16 = vqtbl1q_u8(length_lut_u8x16, high_nibble_u8x16);
        sz_u64_t const shift = (sz_u64_t)quarter * 16;
        uint8x16_t const global_iota_u8x16 = vaddq_u8(within_iota_u8x16, vdupq_n_u8((sz_u8_t)(quarter * 16)));
        // Continuation = `0x80..0xBF`, i.e. as a signed byte `< -64`; one signed compare replaces AND + compare.
        // Its complement (signed `>= -64`) is the start predicate directly, with no `vmvnq_u8`.
        int8x16_t const here_signed_s8x16 = vreinterpretq_s8_u8(here_u8x16);
        uint8x16_t const is_cont_u8x16 = vcltq_s8(here_signed_s8x16, vdupq_n_s8(-64));
        // A start lane (non-continuation, loaded) whose `index + length` exceeds the window declares an overrun.
        uint8x16_t const is_start_u8x16 = vcgeq_s8(here_signed_s8x16, vdupq_n_s8(-64));
        uint8x16_t const reaches_past_u8x16 = vcgtq_u8(vaddq_u8(global_iota_u8x16, len_u8x16), chunk_broadcast_u8x16);
        uint8x16_t const overrun_lane_u8x16 = vandq_u8(is_start_u8x16, reaches_past_u8x16);
        // Non-overrunning lanes -> 0xFF so they lose the min; overrunning lanes keep their global index.
        uint8x16_t const overrun_candidate_u8x16 = vorrq_u8(global_iota_u8x16, vmvnq_u8(overrun_lane_u8x16));
        sz_u8_t const quarter_first = vminvq_u8(overrun_candidate_u8x16);
        if ((sz_size_t)quarter_first < overrun_position) overrun_position = quarter_first;
        continuation_bits |= sz_utf8_movemask16_neon_(is_cont_u8x16) << shift;
        len_eq2 |= sz_utf8_movemask16_neon_(vceqq_u8(len_u8x16, vdupq_n_u8(2))) << shift;
        len_eq3 |= sz_utf8_movemask16_neon_(vceqq_u8(len_u8x16, vdupq_n_u8(3))) << shift;
        len_eq4 |= sz_utf8_movemask16_neon_(vceqq_u8(len_u8x16, vdupq_n_u8(4))) << shift;
        lead_lt_c2 |= sz_utf8_movemask16_neon_(vcltq_u8(here_u8x16, vdupq_n_u8(0xC2))) << shift;
        lead_gt_f4 |= sz_utf8_movemask16_neon_(vcgtq_u8(here_u8x16, vdupq_n_u8(0xF4))) << shift;
        e0 |= sz_utf8_movemask16_neon_(vceqq_u8(here_u8x16, vdupq_n_u8(0xE0))) << shift;
        ed |= sz_utf8_movemask16_neon_(vceqq_u8(here_u8x16, vdupq_n_u8(0xED))) << shift;
        f0 |= sz_utf8_movemask16_neon_(vceqq_u8(here_u8x16, vdupq_n_u8(0xF0))) << shift;
        f4 |= sz_utf8_movemask16_neon_(vceqq_u8(here_u8x16, vdupq_n_u8(0xF4))) << shift;
    }
    continuation_bits &= loaded_mask;
    starts_bits = loaded_mask & ~continuation_bits;

    // Per-lane sequence length only matters on a start lane; mask the length classes to starts.
    len_eq2 &= starts_bits;
    len_eq3 &= starts_bits;
    len_eq4 &= starts_bits;
    sz_u64_t const len_ge2 = len_eq2 | len_eq3 | len_eq4;
    sz_u64_t const len_ge3 = len_eq3 | len_eq4;
    sz_u64_t const len_ge4 = len_eq4;

    // The FIRST start whose declared sequence reaches past the window bounds the decodable prefix (a malformed
    // lead-in-lead like `E0 C0` can overrun mid-window, not only the trailing truncation). `overrun_position` was
    // computed branchlessly above (per-lane `index + length > chunk`, `vminvq_u8` first-set), no scalar ctz walk.
    sz_size_t const decodable_end = overrun_position;
    sz_u64_t const decodable_mask = sz_u64_mask_until_serial_(decodable_end);

    // Bad lead (its own 1-byte U+FFFD subpart): a length-2 lead < 0xC2 (C0/C1 overlong) or a length-4 lead > 0xF4
    // (F5..FF out of range). `lead_lt_c2`/`lead_gt_f4` were classified per lane above; restrict to the matching length.
    sz_u64_t const bad_lead_bits = (len_eq2 & lead_lt_c2) | (len_eq4 & lead_gt_f4);

    // First-continuation range violations for E0/ED/F0/F4 (overlong 3/4-byte, surrogate, > U+10FFFF). Computed
    // whenever a 3- OR 4-byte lead is present; for any other lead the mask is empty, so `b1_range_ok` keeps the lane.
    int const has_three = len_ge3 != 0;
    int const has_four = len_ge4 != 0;
    sz_u64_t overlong_or_surrogate_or_range_bits = 0;
    if (has_three || has_four) {
        // The first continuation byte of each lead = next lane's byte. Build n1 range masks shifted into start lanes.
        sz_u64_t n1_lt_a0 = 0, n1_ge_a0 = 0, n1_lt_90 = 0, n1_ge_90 = 0;
        for (int quarter = 0; quarter < 4; ++quarter) {
            sz_u64_t const shift = (sz_u64_t)quarter * 16;
            n1_lt_a0 |= sz_utf8_movemask16_neon_(vcltq_u8(window_u8x16[quarter], vdupq_n_u8(0xA0))) << shift;
            n1_ge_a0 |= sz_utf8_movemask16_neon_(vcgeq_u8(window_u8x16[quarter], vdupq_n_u8(0xA0))) << shift;
            n1_lt_90 |= sz_utf8_movemask16_neon_(vcltq_u8(window_u8x16[quarter], vdupq_n_u8(0x90))) << shift;
            n1_ge_90 |= sz_utf8_movemask16_neon_(vcgeq_u8(window_u8x16[quarter], vdupq_n_u8(0x90))) << shift;
        }
        // Shift the next-byte range masks right by one so lead lane `i` carries the test on byte `i+1`.
        overlong_or_surrogate_or_range_bits = ((e0 & (n1_lt_a0 >> 1)) | (ed & (n1_ge_a0 >> 1)) |
                                               (f0 & (n1_lt_90 >> 1)) | (f4 & (n1_ge_90 >> 1))) &
                                              starts_bits;
    }

    // Per-lane continuation availability at the declared trailing slots, evaluated at the lead lane.
    sz_u64_t const cont1 = starts_bits & (continuation_bits >> 1);
    sz_u64_t const cont2 = starts_bits & (continuation_bits >> 2);
    sz_u64_t const cont3 = starts_bits & (continuation_bits >> 3);
    sz_u64_t const b1_range_ok = starts_bits & ~overlong_or_surrogate_or_range_bits;
    // A valid first continuation: present, range-valid, and the lead is not itself bad (C0/C1, F5..FF).
    sz_u64_t const first_ok = cont1 & b1_range_ok & ~bad_lead_bits;

    // Well-formed leads (vectorized `sz_rune_decode` success), restricted to the decodable span.
    sz_u64_t const length_one_bits = starts_bits & ~len_ge2;
    sz_u64_t const wf1 = length_one_bits;
    sz_u64_t const wf2 = len_eq2 & ~bad_lead_bits & cont1;
    sz_u64_t const wf3 = len_eq3 & b1_range_ok & cont1 & cont2;
    sz_u64_t const wf4 = len_eq4 & ~bad_lead_bits & b1_range_ok & cont1 & cont2 & cont3;
    sz_u64_t const well_formed = (wf1 | wf2 | wf3 | wf4) & decodable_mask;

    // Per-lane maximal-subpart length (mirror of `sz_utf8_maximal_subpart_`): start at 1 and extend across each
    // continuation slot that a well-formed sequence would still accept. For well-formed lanes this equals the declared
    // length; for ill-formed lanes it is the 1-3 byte subpart that one U+FFFD consumes.
    sz_u64_t const step2 = len_ge2 & first_ok;
    sz_u64_t const step3 = step2 & len_ge3 & cont2;
    sz_u64_t const step4 = step3 & len_ge4 & cont3;

    // Orphan promotion: a continuation byte not covered by ANY lead's maximal-subpart span (well-formed OR the bytes an
    // ill-formed lead's single U+FFFD consumes) becomes its own 1-byte U+FFFD. The subpart spans are exactly the
    // continuation slots the `step2/3/4` adds reached, so coverage is those slots smeared by their offset.
    sz_u64_t const covered = ((step2 & decodable_mask) << 1) | ((step3 & decodable_mask) << 2) |
                             ((step4 & decodable_mask) << 3);
    sz_u64_t const orphan = continuation_bits & decodable_mask & ~covered;
    sz_u64_t const emit_starts = (starts_bits | orphan) & decodable_mask;
    sz_size_t const emit_count = (sz_size_t)sz_u64_popcount(emit_starts);
    if (emit_count == 0) { return *runes_unpacked = 0, text; } // Nothing decodable -> window-edge finalize in driver.
    sz_u64_t const ill_formed = emit_starts & ~well_formed;    // Orphans are continuations -> never well-formed.

    // Materialize the per-lane ill-formed flag (0xFF/0x00) and the per-lane maximal-subpart length (1..4) as 64-byte
    // window-order vectors so the drain gathers both with the SAME packed start indices. `expand16` inverts the
    // movemask per quarter (gather-free); the length is `1 + step2 + step3 + step4` per lane via the expanded bits.
    uint8x16_t ill_quarter[4];
    sz_u8_t consumed_length[64];
    for (int quarter = 0; quarter < 4; ++quarter) {
        sz_u32_t const shift = (sz_u32_t)quarter * 16;
        ill_quarter[quarter] = sz_utf8_expand16_neon_((sz_u32_t)((ill_formed >> shift) & 0xFFFFu));
        uint8x16_t const one_u8x16 = vdupq_n_u8(1);
        uint8x16_t length_q_u8x16 = one_u8x16;
        length_q_u8x16 = vaddq_u8(length_q_u8x16, vandq_u8(sz_utf8_expand16_neon_((sz_u32_t)((step2 >> shift) & 0xFFFFu)), one_u8x16));
        length_q_u8x16 = vaddq_u8(length_q_u8x16, vandq_u8(sz_utf8_expand16_neon_((sz_u32_t)((step3 >> shift) & 0xFFFFu)), one_u8x16));
        length_q_u8x16 = vaddq_u8(length_q_u8x16, vandq_u8(sz_utf8_expand16_neon_((sz_u32_t)((step4 >> shift) & 0xFFFFu)), one_u8x16));
        vst1q_u8(consumed_length + quarter * 16, length_q_u8x16);
    }

    sz_size_t consumed = 0;
    sz_size_t const produced = sz_utf8_rune_drain_neon_(window_u8x16, emit_starts, ill_quarter, consumed_length, emit_count,
                                                        runes, runes_capacity, &consumed);
    *runes_unpacked = produced;
    return text + consumed;
}

SZ_PUBLIC sz_cptr_t sz_utf8_decode_neon(        //
    sz_cptr_t text, sz_size_t length,           //
    sz_rune_t *runes, sz_size_t runes_capacity, //
    sz_size_t *runes_unpacked) {

    sz_cptr_t cursor = text;
    sz_cptr_t const end = text + length;
    sz_size_t runes_written = 0;
    while (runes_written < runes_capacity && cursor < end) {
        sz_size_t step_unpacked = 0;
        sz_cptr_t next = sz_utf8_decode_once_neon_(cursor, (sz_size_t)(end - cursor), runes + runes_written,
                                                   runes_capacity - runes_written, &step_unpacked);
        if (step_unpacked) {
            runes_written += step_unpacked;
            cursor = next;
            continue;
        }
        // The in-vector step decodes its whole decodable span; `step_unpacked == 0` only when the very first lead
        // declares a sequence crossing the window edge (a boundary truncation). A resumable truncation breaks and
        // awaits more bytes; a bad/overlong truncated lead at the edge finalizes to one U+FFFD over its maximal
        // ill-formed subpart - a bounded <=3-byte finalize, never a serial window re-decode.
        if (sz_utf8_incomplete_tail_(cursor, end)) break;
        runes[runes_written++] = (sz_rune_t)sz_rune_replacement_k;
        cursor += sz_utf8_maximal_subpart_(cursor, end);
    }
    *runes_unpacked = runes_written;
    return cursor;
}

#pragma endregion Drains

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#endif // SZ_USE_NEON

#ifdef __cplusplus
}
#endif

#endif // STRINGZILLA_UTF8_RUNES_NEON_H_
