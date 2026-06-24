/**
 *  @brief NEON backend for UTF-8 codepoint mechanics.
 *  @file include/stringzilla/utf8_codepoints/neon.h
 *  @author Ash Vardanian
 */
#ifndef STRINGZILLA_UTF8_CODEPOINTS_NEON_H_
#define STRINGZILLA_UTF8_CODEPOINTS_NEON_H_

#include <string.h> // `memcpy`, `memset` (partial-window staging; NEON has no byte-masked load)

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

#pragma endregion Multistep newline and whitespace iteration

#pragma region Shared SIMD leaf substrate

/*  Family-agnostic NEON (AArch64) leaf helpers shared by every UTF-8 segmentation kernel (word / grapheme /
 *  sentence / line / delimiters). The exact twin of the Ice Lake @ref sz_utf8_codepoints_window_t substrate and the
 *  AVX2 @ref sz_utf8_codepoints_window_haswell_t backend: a 64-byte window held as four `uint8x16_t` quarters, masks
 *  reduced to one-bit-per-byte `sz_u64_t` lane masks (bit `i` <=> lane `i`), and gather-free table cascades. The
 *  portable rule algebra in `serial.h` consumes the `sz_u64_t` masks unchanged across all three backends. */

/** @brief  The decoded 64-byte window for the NEON backend. The 64 bytes live as four `uint8x16_t` quarters
 *          (`window[0]` = lanes [0, 16), ... `window[3]` = lanes [48, 64)); the per-lane byte-domain codepoint
 *          halves `high`/`low` share that shape. Masks are `sz_u64_t` (one bit per byte-lane: the NEON movemask of
 *          each quarter placed at bit positions [0,16)/[16,32)/[32,48)/[48,64)) rather than the Ice Lake
 *          `__mmask64`. Field names and semantics match @ref sz_utf8_codepoints_window_t so the portable rule
 *          algebra is unchanged. */
typedef struct sz_utf8_codepoints_window_neon_t {
    uint8x16_t window[4];       /**< Raw input bytes for lanes [16*q, 16*q+16). */
    uint8x16_t high[4];         /**< Per-lane `codepoint >> 8`. */
    uint8x16_t low[4];          /**< Per-lane `codepoint & 0xFF`. */
    sz_u64_t continuation;      /**< Bit `i` => lane `i` is a continuation byte `10xxxxxx`. */
    sz_u64_t codepoint_starts;  /**< Bit `i` => lane `i` begins a codepoint (loaded, non-continuation). */
    sz_u64_t two_byte_starts;   /**< Bit `i` => lane `i` is a 2-byte lead `110xxxxx`. */
    sz_u64_t three_byte_starts; /**< Bit `i` => lane `i` is a 3-byte lead `1110xxxx`. */
    sz_u64_t four_byte_starts;  /**< Bit `i` => lane `i` is a 4-byte lead `11110xxx`. */
    sz_size_t loaded;           /**< Number of bytes actually loaded (<= 64). */
} sz_utf8_codepoints_window_neon_t;

/** @brief  Per-byte logical right shift by @p shift keeping the low @p keep bits — the NEON twin of `srl8_`.
 *          `vshrq_n_u8` needs an immediate shift; the shift amounts used by the segmentation classifiers (2 for the
 *          decode 3-byte high reconstruction, 4 for the line-break 4-byte plane reconstruction) are spelled out. */
SZ_INTERNAL uint8x16_t sz_utf8_codepoints_srl8_neon_(uint8x16_t value, int shift, sz_u8_t keep) {
    uint8x16_t shifted;
    switch (shift) {
    case 2: shifted = vshrq_n_u8(value, 2); break;
    case 4: shifted = vshrq_n_u8(value, 4); break;
    default: shifted = value; break;
    }
    return vandq_u8(shifted, vdupq_n_u8(keep));
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
SZ_INTERNAL sz_u64_t sz_utf8_codepoints_movemask16_neon_(uint8x16_t boolean_lanes) {
    static sz_u8_t const bit_position_lanes[16] = {1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128};
    uint8x16_t const bit_position = vld1q_u8(bit_position_lanes);
    uint8x16_t const isolated = vandq_u8(boolean_lanes, bit_position);
    sz_u64_t const low_bits = (sz_u64_t)vaddv_u8(vget_low_u8(isolated));
    sz_u64_t const high_bits = (sz_u64_t)vaddv_u8(vget_high_u8(isolated));
    return low_bits | (high_bits << 8);
}

/** @brief  Combine the four per-quarter NEON movemasks into one 64-bit lane mask: quarter `q` -> bits [16*q, 16*q+16).
 *          The NEON twin of @ref sz_utf8_codepoints_mask_combine_haswell_ (which OR-combines two 32-bit halves). */
SZ_INTERNAL sz_u64_t sz_utf8_codepoints_mask_combine_neon_( //
    uint8x16_t quarter0, uint8x16_t quarter1, uint8x16_t quarter2, uint8x16_t quarter3) {
    sz_u64_t mask = sz_utf8_codepoints_movemask16_neon_(quarter0);
    mask |= sz_utf8_codepoints_movemask16_neon_(quarter1) << 16;
    mask |= sz_utf8_codepoints_movemask16_neon_(quarter2) << 32;
    mask |= sz_utf8_codepoints_movemask16_neon_(quarter3) << 48;
    return mask;
}

/** @brief  Masked 64-byte load into four quarters; bytes [loaded, 64) read as zero (the NEON stand-in for
 *          `_mm512_maskz_loadu_epi8`). A small stack staging buffer covers the partial tail so we never read past
 *          `text + loaded`. Mirrors @ref sz_utf8_codepoints_load_window_haswell_. */
SZ_INTERNAL void sz_utf8_codepoints_load_window_neon_(sz_u8_t const *text, sz_size_t loaded, uint8x16_t *out) {
    if (loaded >= 64) {
        out[0] = vld1q_u8(text + 0);
        out[1] = vld1q_u8(text + 16);
        out[2] = vld1q_u8(text + 32);
        out[3] = vld1q_u8(text + 48);
        return;
    }
    sz_u8_t staging[64];
    memset(staging, 0, sizeof(staging));
    memcpy(staging, text, loaded);
    out[0] = vld1q_u8(staging + 0);
    out[1] = vld1q_u8(staging + 16);
    out[2] = vld1q_u8(staging + 32);
    out[3] = vld1q_u8(staging + 48);
}

/**
 *  @brief  Forward neighbours `next1[i] = window[i+1]`, `next2[i] = window[i+2]`, `next3[i] = window[i+3]` over all 64
 *          lanes, with the lanes past the window WRAPPING modulo 64 to match Ice Lake's `_mm512_permutexvar_epi8`
 *          (so `next1[63]==window[0]`, etc.). NEON `vextq_u8` concatenates two adjacent quarters and extracts the
 *          shifted span; the successor of the last quarter wraps to quarter 0 (byte 64 aliases byte 0). The three
 *          neighbour distances are provided because the family classifiers need up to `next3` (4-byte sequences).
 */
SZ_INTERNAL void sz_utf8_codepoints_forward_neighbours_neon_( //
    uint8x16_t const *window, uint8x16_t *next1, uint8x16_t *next2, uint8x16_t *next3) {
    for (int quarter = 0; quarter < 4; ++quarter) {
        uint8x16_t const here = window[quarter];
        uint8x16_t const successor = window[(quarter + 1) & 3];
        next1[quarter] = vextq_u8(here, successor, 1);
        next2[quarter] = vextq_u8(here, successor, 2);
        next3[quarter] = vextq_u8(here, successor, 3);
    }
}

/** @brief  Load up to 64 bytes (masked tail) and decode every lane into byte-domain halves — the NEON twin of
 *          @ref sz_utf8_codepoints_decode_window_, bit-identical to it (and to `_haswell_`) on every lane. */
SZ_INTERNAL sz_utf8_codepoints_window_neon_t sz_utf8_codepoints_decode_window_neon_( //
    sz_u8_t const *text, sz_size_t available) {
    sz_utf8_codepoints_window_neon_t result;
    result.loaded = available < 64 ? available : 64;

    uint8x16_t window[4];
    sz_utf8_codepoints_load_window_neon_(text, result.loaded, window);
    for (int quarter = 0; quarter < 4; ++quarter) result.window[quarter] = window[quarter];

    uint8x16_t next1[4], next2[4], next3[4];
    sz_utf8_codepoints_forward_neighbours_neon_(window, next1, next2, next3);

    sz_u64_t const loaded_mask = sz_utf8_codepoints_mask_until_(result.loaded);

    // Lead-class detection over loaded lanes: (byte & mask) == pattern, AND-clamped to loaded lanes.
    uint8x16_t continuation_bool[4], two_byte_bool[4], three_byte_bool[4], four_byte_bool[4];
    uint8x16_t const mask_continuation = vdupq_n_u8(0xC0), pattern_continuation = vdupq_n_u8(0x80);
    uint8x16_t const mask_two = vdupq_n_u8(0xE0), pattern_two = vdupq_n_u8(0xC0);
    uint8x16_t const mask_three = vdupq_n_u8(0xF0), pattern_three = vdupq_n_u8(0xE0);
    uint8x16_t const mask_four = vdupq_n_u8(0xF8), pattern_four = vdupq_n_u8(0xF0);
    for (int quarter = 0; quarter < 4; ++quarter) {
        uint8x16_t const here = window[quarter];
        continuation_bool[quarter] = vceqq_u8(vandq_u8(here, mask_continuation), pattern_continuation);
        two_byte_bool[quarter] = vceqq_u8(vandq_u8(here, mask_two), pattern_two);
        three_byte_bool[quarter] = vceqq_u8(vandq_u8(here, mask_three), pattern_three);
        four_byte_bool[quarter] = vceqq_u8(vandq_u8(here, mask_four), pattern_four);
    }
    result.continuation = sz_utf8_codepoints_mask_combine_neon_(continuation_bool[0], continuation_bool[1],
                                                                continuation_bool[2], continuation_bool[3]) &
                          loaded_mask;
    result.codepoint_starts = loaded_mask & ~result.continuation;
    result.two_byte_starts = sz_utf8_codepoints_mask_combine_neon_(two_byte_bool[0], two_byte_bool[1], two_byte_bool[2],
                                                                   two_byte_bool[3]) &
                             loaded_mask;
    result.three_byte_starts = sz_utf8_codepoints_mask_combine_neon_(three_byte_bool[0], three_byte_bool[1],
                                                                     three_byte_bool[2], three_byte_bool[3]) &
                               loaded_mask;
    result.four_byte_starts = sz_utf8_codepoints_mask_combine_neon_(four_byte_bool[0], four_byte_bool[1],
                                                                    four_byte_bool[2], four_byte_bool[3]) &
                              loaded_mask;

    uint8x16_t const low_five_bits = vdupq_n_u8(0x1F);
    uint8x16_t const low_six_bits = vdupq_n_u8(0x3F);
    uint8x16_t const low_two_bits = vdupq_n_u8(0x03);
    uint8x16_t const low_four_bits = vdupq_n_u8(0x0F);

    for (int quarter = 0; quarter < 4; ++quarter) {
        uint8x16_t const here = window[quarter];
        uint8x16_t const n1 = next1[quarter];
        uint8x16_t const n2 = next2[quarter];

        // 2-byte: codepoint = ((b0 & 0x1F) << 6) | (b1 & 0x3F); high = codepoint >> 8, low = codepoint & 0xFF.
        uint8x16_t const high_two = sz_utf8_codepoints_srl8_neon_(vandq_u8(here, low_five_bits), 2, 0x07);
        uint8x16_t const low_two = vorrq_u8(vshlq_n_u8(vandq_u8(here, low_two_bits), 6), vandq_u8(n1, low_six_bits));

        // 3-byte: codepoint = ((b0 & 0x0F) << 12) | ((b1 & 0x3F) << 6) | (b2 & 0x3F).
        uint8x16_t const high_three = vorrq_u8(vshlq_n_u8(vandq_u8(here, low_four_bits), 4),
                                               sz_utf8_codepoints_srl8_neon_(n1, 2, 0x0F));
        uint8x16_t const low_three = vorrq_u8(vshlq_n_u8(vandq_u8(n1, low_two_bits), 6), vandq_u8(n2, low_six_bits));

        // Blend 2-byte vs 3-byte per lane: select the 3-byte value where this lane is a 3-byte lead.
        uint8x16_t const three_select = three_byte_bool[quarter];
        result.high[quarter] = vbslq_u8(three_select, high_three, high_two);
        result.low[quarter] = vbslq_u8(three_select, low_three, low_two);
    }
    return result;
}

/** @brief  One nibble-cascade stage with a sub-256 selector: `result[lane] = table[selector[lane]*16 + within[lane]]`.
 *          Each of @p tile_count 16-byte rows is one selector value; @p within is the addressing nibble (caller-masked
 *          to `[0,16)`). The NEON twin of the AVX2 cascade and VBMI `vpermi2b`: each resident row is loaded into a
 *          `uint8x16_t` and shuffled by @p within via `vqtbl1q_u8`, then blended in for the lanes whose @p selector
 *          picks that row. Gather-free — only `vld1q`/`vqtbl1q`/`vceqq`/`vbslq`. @p within / @p selector address one
 *          quarter; the caller iterates the four quarters. */
SZ_INTERNAL uint8x16_t sz_utf8_codepoints_cascade_stage_neon_( //
    sz_u8_t const *table, int tile_count, uint8x16_t selector, uint8x16_t within) {
    uint8x16_t result = vdupq_n_u8(0);
    for (int tile = 0; tile < tile_count; ++tile) {
        uint8x16_t const row = vld1q_u8(table + tile * 16);
        uint8x16_t const picked = vqtbl1q_u8(row, within);
        uint8x16_t const here = vceqq_u8(selector, vdupq_n_u8((sz_u8_t)tile));
        result = vbslq_u8(here, picked, result);
    }
    return result;
}

/** @brief  256-entry byte LUT addressed by a per-lane byte index in `[0,256)`: `result[lane] = group_base[index[lane]]`.
 *          Four `vqtbl4q_u8` reads over the four resident 64-byte quads; `vqtbl4q_u8` returns zero for indices >= 64,
 *          so subtracting 64/128/192 routes each lane to exactly one quad and the four results OR together. The NEON
 *          twin of the substrate `lut256` leaf. @p index addresses one quarter. */
SZ_INTERNAL uint8x16_t sz_utf8_codepoints_lut256_neon_(sz_u8_t const *group_base, uint8x16_t index) {
    uint8x16x4_t const quad0 = vld1q_u8_x4(group_base + 0 * 64);
    uint8x16x4_t const quad1 = vld1q_u8_x4(group_base + 1 * 64);
    uint8x16x4_t const quad2 = vld1q_u8_x4(group_base + 2 * 64);
    uint8x16x4_t const quad3 = vld1q_u8_x4(group_base + 3 * 64);
    uint8x16_t const within0 = index;
    uint8x16_t const within1 = vsubq_u8(index, vdupq_n_u8(64));
    uint8x16_t const within2 = vsubq_u8(index, vdupq_n_u8(128));
    uint8x16_t const within3 = vsubq_u8(index, vdupq_n_u8(192));
    uint8x16_t result = vqtbl4q_u8(quad0, within0);
    result = vorrq_u8(result, vqtbl4q_u8(quad1, within1));
    result = vorrq_u8(result, vqtbl4q_u8(quad2, within2));
    result = vorrq_u8(result, vqtbl4q_u8(quad3, within3));
    return result;
}

#pragma endregion Shared SIMD leaf substrate

#pragma region Drains

/** @brief  Left-pack the set lane indices (in [0, 64), ascending) of a 64-bit @p mask into @p out[0..popcount).
 *          NEON has no `vpcompressb`/`pext`: pull the lowest set bit's index via `ctz`, clear it via `mask & (mask-1)`.
 *          This matches the AVX2 `sz_utf8_codepoints_unpack_indices_haswell_` BMI2 path semantically; on AArch64
 *          `ctz` lowers to `rbit`+`clz`, and the loop trip count is the boundary popcount (sparse for real text). */
SZ_INTERNAL void sz_utf8_codepoints_unpack_indices_neon_(sz_u64_t mask, sz_u8_t *out) {
    while (mask) {
        *out++ = (sz_u8_t)sz_u64_ctz(mask);
        mask &= mask - 1; // clear the lowest set bit
    }
}

/** @brief  NEON forward drain — the `vpcompressb`-free twin of @ref sz_utf8_codepoints_drain_forward_. Emits one
 *          (start, length) per set boundary lane (ascending), honoring @p capacity and the carried previous-boundary
 *          via @p previous_io; bit-exact with the Ice Lake leaf. Indices are unpacked once, then each segment's
 *          (start, length) is computed from the carried previous boundary. */
SZ_INTERNAL sz_size_t sz_utf8_codepoints_drain_forward_neon_( //
    sz_u64_t boundary, sz_size_t base, sz_size_t *starts, sz_size_t *lengths, sz_size_t produced, sz_size_t capacity,
    sz_size_t *previous_io) {
    sz_size_t const boundary_count = (sz_size_t)sz_u64_popcount(boundary);
    sz_size_t previous = *previous_io;
    if (boundary_count == 0 || produced >= capacity) {
        *previous_io = previous;
        return produced;
    }

    sz_u8_t indices[64];
    sz_utf8_codepoints_unpack_indices_neon_(boundary, indices);

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

/** @brief  NEON backward drain — the `vpcompressb`-free mirror of @ref sz_utf8_codepoints_drain_backward_. Emits
 *          (start, length) per set boundary lane in DESCENDING byte order, carrying the open segment end via
 *          @p end_io; bit-exact with the Ice Lake leaf. */
SZ_INTERNAL sz_size_t sz_utf8_codepoints_drain_backward_neon_( //
    sz_u64_t boundary, sz_size_t base, sz_size_t *starts, sz_size_t *lengths, sz_size_t produced, sz_size_t capacity,
    sz_size_t *end_io) {
    sz_size_t const boundary_count = (sz_size_t)sz_u64_popcount(boundary);
    sz_size_t segment_end = *end_io;
    if (boundary_count == 0 || produced >= capacity) {
        *end_io = segment_end;
        return produced;
    }

    sz_u8_t indices[64];
    sz_utf8_codepoints_unpack_indices_neon_(boundary, indices);

    sz_size_t emitted = 0;
    sz_size_t high = boundary_count; // read the ascending index array high-to-low
    while (emitted < boundary_count && produced < capacity) {
        sz_size_t const position = base + (sz_size_t)indices[high - 1];
        starts[produced] = position;
        lengths[produced] = segment_end - position;
        segment_end = position;
        ++produced, ++emitted, --high;
    }
    *end_io = segment_end;
    return produced;
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

#endif // STRINGZILLA_UTF8_CODEPOINTS_NEON_H_
