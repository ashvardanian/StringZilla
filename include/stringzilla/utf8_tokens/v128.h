/**
 *  @file include/stringzilla/utf8_tokens/v128.h
 *  @author Ash Vardanian
 *  @date June 7, 2026
 *  @brief WebAssembly SIMD128 backend for UTF-8 newline and whitespace delimiter scanning.
 */
#ifndef STRINGZILLA_UTF8_TOKENS_V128_H_
#define STRINGZILLA_UTF8_TOKENS_V128_H_

#include "stringzilla/types.h"
#include "stringzilla/utf8_tokens/serial.h"
#include "stringzilla/utf8_runes/v128.h"
#include "stringzilla/utf8_tokens/tables.h"

#ifdef __cplusplus
extern "C" {
#endif

#if STRINGZILLA_ARCH_WASM_
#if STRINGZILLA_ARCH_WASM_V128_
#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("simd128"))), apply_to = function)
#endif

STRINGZILLA_INLINE v128_t sz_utf8_rotate1_v128_(v128_t bytes_u8x16) {
    return wasm_i8x16_shuffle(bytes_u8x16, bytes_u8x16, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0);
}

STRINGZILLA_INLINE v128_t sz_utf8_rotate2_v128_(v128_t bytes_u8x16) {
    return wasm_i8x16_shuffle(bytes_u8x16, bytes_u8x16, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0, 1);
}

#pragma region Multistep newline and whitespace iteration

/**
 *  @brief Peels the tile's first @p emit_count matches with a @c wasm_i8x16_swizzle left-pack, 4
 *      lanes per block.
 *
 *  Walks the 16-lane @p start_bits mask in four ascending 4-lane sub-blocks, gathering each block's
 *  set `(position + lane, length)` pairs to the front of a 16-wide stack scratch via one swizzle
 *  from @c compact_lut, then copies the low @p emit_count entries out in ascending lane order,
 *  byte-exact, with no per-match @c ctz.
 */
STRINGZILLA_INLINE void sz_utf8_iterate_peel_v128_(                            //
    sz_u32_t start_bits, sz_u32_t two_byte_starts, sz_u32_t three_byte_starts, //
    sz_size_t emit_count, sz_size_t position,                                  //
    sz_size_t *match_offsets, sz_size_t *match_lengths) {

    // Per-file left-pack table for `wasm_i8x16_swizzle`: row `[m]` holds the 16 byte indices that gather the
    // `m`-selected 32-bit lanes (of four, each a 4-byte group) to the front of a 4-lane register.
    static sz_u8_t const compact_lut[16][16] = {
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},       {0, 1, 2, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {4, 5, 6, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},       {0, 1, 2, 3, 4, 5, 6, 7, 0, 0, 0, 0, 0, 0, 0, 0},
        {8, 9, 10, 11, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},     {0, 1, 2, 3, 8, 9, 10, 11, 0, 0, 0, 0, 0, 0, 0, 0},
        {4, 5, 6, 7, 8, 9, 10, 11, 0, 0, 0, 0, 0, 0, 0, 0},     {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 0, 0, 0, 0},
        {12, 13, 14, 15, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},   {0, 1, 2, 3, 12, 13, 14, 15, 0, 0, 0, 0, 0, 0, 0, 0},
        {4, 5, 6, 7, 12, 13, 14, 15, 0, 0, 0, 0, 0, 0, 0, 0},   {0, 1, 2, 3, 4, 5, 6, 7, 12, 13, 14, 15, 0, 0, 0, 0},
        {8, 9, 10, 11, 12, 13, 14, 15, 0, 0, 0, 0, 0, 0, 0, 0}, {0, 1, 2, 3, 8, 9, 10, 11, 12, 13, 14, 15, 0, 0, 0, 0},
        {4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0, 0, 0, 0}, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
    };

    sz_size_t scratch_offsets[16], scratch_lengths[16];
    sz_size_t filled = 0;
    for (sz_size_t sub_block = 0; sub_block < 4; ++sub_block) {
        sz_size_t const base_lane = sub_block * 4;
        sz_u32_t const submask = (start_bits >> base_lane) & 0xFu;
        if (!submask) continue;

        // Per-lane length: 1, plus 1 on a 2-byte start, plus 2 on a 3-byte start (the masks are disjoint).
        sz_u32_t const two_byte_sub = (two_byte_starts >> base_lane) & 0xFu;
        sz_u32_t const three_byte_sub = (three_byte_starts >> base_lane) & 0xFu;
        v128_t const candidate_offsets_u32x4 = wasm_u32x4_make(
            (sz_u32_t)(position + base_lane + 0), (sz_u32_t)(position + base_lane + 1),
            (sz_u32_t)(position + base_lane + 2), (sz_u32_t)(position + base_lane + 3));
        v128_t const candidate_lengths_u32x4 = wasm_u32x4_make(
            1u + ((two_byte_sub >> 0) & 1u) + 2u * ((three_byte_sub >> 0) & 1u),
            1u + ((two_byte_sub >> 1) & 1u) + 2u * ((three_byte_sub >> 1) & 1u),
            1u + ((two_byte_sub >> 2) & 1u) + 2u * ((three_byte_sub >> 2) & 1u),
            1u + ((two_byte_sub >> 3) & 1u) + 2u * ((three_byte_sub >> 3) & 1u));

        v128_t const permutation_u8x16 = wasm_v128_load(compact_lut[submask]);
        wasm_v128_store(scratch_offsets + filled, wasm_i8x16_swizzle(candidate_offsets_u32x4, permutation_u8x16));
        wasm_v128_store(scratch_lengths + filled, wasm_i8x16_swizzle(candidate_lengths_u32x4, permutation_u8x16));
        filled += sz_popcount4_lut_[submask];
    }

    for (sz_size_t emitted = 0; emitted < emit_count; ++emitted)
        match_offsets[emitted] = scratch_offsets[emitted], match_lengths[emitted] = scratch_lengths[emitted];
}

STRINGZILLA_INLINE sz_size_t sz_utf8_newlines_v128_(    //
    sz_cptr_t text, sz_size_t length,                   //
    sz_size_t *match_offsets, sz_size_t *match_lengths, //
    sz_size_t matches_capacity, sz_size_t *bytes_consumed) {

    sz_u8_t const *text_u8 = (sz_u8_t const *)text;
    sz_size_t count = 0, position = 0;

    v128_t newline_u8x16 = wasm_i8x16_splat('\n');
    v128_t vertical_tab_u8x16 = wasm_i8x16_splat('\v');
    v128_t form_feed_u8x16 = wasm_i8x16_splat('\f');
    v128_t carriage_return_u8x16 = wasm_i8x16_splat('\r');
    v128_t lead_c2_u8x16 = wasm_i8x16_splat((sz_i8_t)0xC2);
    v128_t x_85_u8x16 = wasm_i8x16_splat((sz_i8_t)0x85);
    v128_t lead_e2_u8x16 = wasm_i8x16_splat((sz_i8_t)0xE2);
    v128_t byte_80_u8x16 = wasm_i8x16_splat((sz_i8_t)0x80);
    v128_t x_a8_u8x16 = wasm_i8x16_splat((sz_i8_t)0xA8);
    v128_t x_a9_u8x16 = wasm_i8x16_splat((sz_i8_t)0xA9);

    // We trust delimiter starts only in lanes [0,13] and step by 14, so any 2-/3-byte delimiter is fully loaded.
    sz_u32_t const trusted_lanes_mask = 0x3FFFu; // lanes [0,13]

    while (position + 16 <= length && count < matches_capacity) {
        v128_t window_u8x16 = wasm_v128_load(text_u8 + position);
        v128_t window1_u8x16 = sz_utf8_rotate1_v128_(window_u8x16); // next lane
        v128_t window2_u8x16 = sz_utf8_rotate2_v128_(window_u8x16); // lane after next

        // 1-byte matches: \n \v \f \r (the contiguous control range '\n'..'\f' plus '\r').
        v128_t newline_cmp_u8x16 = wasm_i8x16_eq(window_u8x16, newline_u8x16);
        v128_t carriage_return_cmp_u8x16 = wasm_i8x16_eq(window_u8x16, carriage_return_u8x16);
        v128_t one_byte_cmp_u8x16 = wasm_v128_or(
            wasm_v128_or(newline_cmp_u8x16, wasm_i8x16_eq(window_u8x16, vertical_tab_u8x16)),
            wasm_v128_or(wasm_i8x16_eq(window_u8x16, form_feed_u8x16), carriage_return_cmp_u8x16));

        // 2-byte: CRLF (\r\n, one match) & NEL (C2 85) - computed unconditionally.
        v128_t crlf_cmp_u8x16 = wasm_v128_and(carriage_return_cmp_u8x16, wasm_i8x16_eq(window1_u8x16, newline_u8x16));
        v128_t nel_cmp_u8x16 = wasm_v128_and(wasm_i8x16_eq(window_u8x16, lead_c2_u8x16),
                                             wasm_i8x16_eq(window1_u8x16, x_85_u8x16));
        v128_t two_byte_cmp_u8x16 = wasm_v128_or(crlf_cmp_u8x16, nel_cmp_u8x16);

        // 3-byte: LS (E2 80 A8) & PS (E2 80 A9).
        v128_t lead_e280_cmp_u8x16 = wasm_v128_and(wasm_i8x16_eq(window_u8x16, lead_e2_u8x16),
                                                   wasm_i8x16_eq(window1_u8x16, byte_80_u8x16));
        v128_t three_byte_cmp_u8x16 = wasm_v128_and(
            lead_e280_cmp_u8x16,
            wasm_v128_or(wasm_i8x16_eq(window2_u8x16, x_a8_u8x16), wasm_i8x16_eq(window2_u8x16, x_a9_u8x16)));

        // CRLF's trailing LF must not also be emitted: an LF whose previous lane is a CR.
        v128_t carriage_return_previous_u8x16 = wasm_i8x16_shuffle(wasm_i8x16_splat(0), carriage_return_cmp_u8x16, //
                                                                   15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27,
                                                                   28, 29, 30);
        v128_t lf_of_crlf_cmp_u8x16 = wasm_v128_and(newline_cmp_u8x16, carriage_return_previous_u8x16);

        v128_t starts_cmp_u8x16 = wasm_v128_andnot(
            wasm_v128_or(wasm_v128_or(one_byte_cmp_u8x16, nel_cmp_u8x16), three_byte_cmp_u8x16), lf_of_crlf_cmp_u8x16);

        sz_u32_t start_bits = (sz_u32_t)wasm_i8x16_bitmask(starts_cmp_u8x16) & trusted_lanes_mask;
        sz_u32_t two_byte_bits = (sz_u32_t)wasm_i8x16_bitmask(two_byte_cmp_u8x16);
        sz_u32_t three_byte_bits = (sz_u32_t)wasm_i8x16_bitmask(three_byte_cmp_u8x16);

        // Suppress a leading LF already consumed by a CRLF that straddled the previous tile edge.
        if (position != 0 && text_u8[position - 1] == '\r')
            start_bits &= ~((sz_u32_t)wasm_i8x16_bitmask(newline_cmp_u8x16) & 1u);

        // Count the scalar `start_bits` (already needed by the peel) rather than `wasm_i8x16_popcnt` on a vector.
        sz_size_t const window_matches = (sz_size_t)sz_u32_popcount(start_bits);
        sz_size_t const emit_count = sz_min_of_two(window_matches, matches_capacity - count);
        if (emit_count)
            sz_utf8_iterate_peel_v128_(start_bits, two_byte_bits, three_byte_bits, emit_count, position,
                                       match_offsets + count, match_lengths + count);
        count += emit_count;
        if (count == matches_capacity) { // output buffer full: resume past the last emitted match
            position = match_offsets[count - 1] + match_lengths[count - 1];
            break;
        }
        position += 14;
    }

    // Skip a CRLF's trailing LF if it straddles into the serial tail (the CR was emitted as a 2-byte match).
    if (position != 0 && position < length && text_u8[position - 1] == '\r' && text_u8[position] == '\n') ++position;
    count += sz_utf8_newlines_serial_((sz_cptr_t)(text_u8 + position), length - position, position,
                                      match_offsets + count, match_lengths + count, matches_capacity - count,
                                      bytes_consumed);
    sz_assert_(sz_utf8_batch_consistent_(length, matches_capacity, count, bytes_consumed ? *bytes_consumed : length,
                                         match_offsets, match_lengths));
    return count;
}

STRINGZILLA_INLINE sz_size_t sz_utf8_whitespaces_v128_( //
    sz_cptr_t text, sz_size_t length,                   //
    sz_size_t *match_offsets, sz_size_t *match_lengths, //
    sz_size_t matches_capacity, sz_size_t *bytes_consumed) {

    sz_u8_t const *text_u8 = (sz_u8_t const *)text;
    sz_size_t count = 0, position = 0;

    v128_t tab_u8x16 = wasm_i8x16_splat('\t');
    v128_t carriage_return_u8x16 = wasm_i8x16_splat('\r');
    v128_t x_20_u8x16 = wasm_i8x16_splat(' ');
    v128_t lead_c2_u8x16 = wasm_i8x16_splat((sz_i8_t)0xC2);
    v128_t x_85_u8x16 = wasm_i8x16_splat((sz_i8_t)0x85);
    v128_t x_a0_u8x16 = wasm_i8x16_splat((sz_i8_t)0xA0);
    v128_t x_e1_u8x16 = wasm_i8x16_splat((sz_i8_t)0xE1);
    v128_t lead_e2_u8x16 = wasm_i8x16_splat((sz_i8_t)0xE2);
    v128_t x_e3_u8x16 = wasm_i8x16_splat((sz_i8_t)0xE3);
    v128_t x_9a_u8x16 = wasm_i8x16_splat((sz_i8_t)0x9A);
    v128_t byte_80_u8x16 = wasm_i8x16_splat((sz_i8_t)0x80);
    v128_t x_81_u8x16 = wasm_i8x16_splat((sz_i8_t)0x81);
    v128_t x_8a_u8x16 = wasm_i8x16_splat((sz_i8_t)0x8A);
    v128_t x_a8_u8x16 = wasm_i8x16_splat((sz_i8_t)0xA8);
    v128_t x_a9_u8x16 = wasm_i8x16_splat((sz_i8_t)0xA9);
    v128_t x_af_u8x16 = wasm_i8x16_splat((sz_i8_t)0xAF);
    v128_t x_9f_u8x16 = wasm_i8x16_splat((sz_i8_t)0x9F);

    sz_u32_t const trusted_lanes_mask = 0x3FFFu; // lanes [0,13]

    while (position + 16 <= length && count < matches_capacity) {
        v128_t window_u8x16 = wasm_v128_load(text_u8 + position);
        v128_t window1_u8x16 = sz_utf8_rotate1_v128_(window_u8x16); // next lane
        v128_t window2_u8x16 = sz_utf8_rotate2_v128_(window_u8x16); // lane after next

        // 1-byte: space, plus the contiguous range [\t, \r] == [9, 13].
        v128_t one_byte_cmp_u8x16 = wasm_v128_or(
            wasm_i8x16_eq(window_u8x16, x_20_u8x16),
            wasm_v128_and(wasm_u8x16_ge(window_u8x16, tab_u8x16), wasm_u8x16_le(window_u8x16, carriage_return_u8x16)));

        // 2-byte: C2 85 (NEL), C2 A0 (NBSP).
        v128_t lead_c2_cmp_u8x16 = wasm_i8x16_eq(window_u8x16, lead_c2_u8x16);
        v128_t two_byte_cmp_u8x16 = wasm_v128_and(
            lead_c2_cmp_u8x16,
            wasm_v128_or(wasm_i8x16_eq(window1_u8x16, x_85_u8x16), wasm_i8x16_eq(window1_u8x16, x_a0_u8x16)));

        // 3-byte: E1 9A 80 (ogham); E2 80 [80-8A]; E2 80 AF; E2 81 9F; E2 80 A8/A9; E3 80 80.
        v128_t window1_is_80_u8x16 = wasm_i8x16_eq(window1_u8x16, byte_80_u8x16);
        v128_t lead_e280_cmp_u8x16 = wasm_v128_and(wasm_i8x16_eq(window_u8x16, lead_e2_u8x16), window1_is_80_u8x16);
        v128_t ogham_cmp_u8x16 = wasm_v128_and(
            wasm_i8x16_eq(window_u8x16, x_e1_u8x16),
            wasm_v128_and(wasm_i8x16_eq(window1_u8x16, x_9a_u8x16), wasm_i8x16_eq(window2_u8x16, byte_80_u8x16)));
        v128_t range_e280_cmp_u8x16 = wasm_v128_and(
            lead_e280_cmp_u8x16,
            wasm_v128_and(wasm_u8x16_ge(window2_u8x16, byte_80_u8x16), wasm_u8x16_le(window2_u8x16, x_8a_u8x16)));
        v128_t nnbsp_cmp_u8x16 = wasm_v128_and(lead_e280_cmp_u8x16, wasm_i8x16_eq(window2_u8x16, x_af_u8x16));
        v128_t mmsp_cmp_u8x16 = wasm_v128_and(
            wasm_v128_and(wasm_i8x16_eq(window_u8x16, lead_e2_u8x16), wasm_i8x16_eq(window1_u8x16, x_81_u8x16)),
            wasm_i8x16_eq(window2_u8x16, x_9f_u8x16));
        v128_t line_cmp_u8x16 = wasm_v128_and(lead_e280_cmp_u8x16, wasm_i8x16_eq(window2_u8x16, x_a8_u8x16));
        v128_t paragraph_cmp_u8x16 = wasm_v128_and(lead_e280_cmp_u8x16, wasm_i8x16_eq(window2_u8x16, x_a9_u8x16));
        v128_t ideographic_cmp_u8x16 = wasm_v128_and(
            wasm_v128_and(wasm_i8x16_eq(window_u8x16, x_e3_u8x16), window1_is_80_u8x16),
            wasm_i8x16_eq(window2_u8x16, byte_80_u8x16));
        v128_t three_byte_cmp_u8x16 = wasm_v128_or(
            wasm_v128_or(wasm_v128_or(ogham_cmp_u8x16, range_e280_cmp_u8x16),
                         wasm_v128_or(nnbsp_cmp_u8x16, mmsp_cmp_u8x16)),
            wasm_v128_or(wasm_v128_or(line_cmp_u8x16, paragraph_cmp_u8x16), ideographic_cmp_u8x16));

        v128_t starts_cmp_u8x16 = wasm_v128_or(wasm_v128_or(one_byte_cmp_u8x16, two_byte_cmp_u8x16),
                                               three_byte_cmp_u8x16);

        sz_u32_t start_bits = (sz_u32_t)wasm_i8x16_bitmask(starts_cmp_u8x16) & trusted_lanes_mask;
        sz_u32_t two_byte_bits = (sz_u32_t)wasm_i8x16_bitmask(two_byte_cmp_u8x16);
        sz_u32_t three_byte_bits = (sz_u32_t)wasm_i8x16_bitmask(three_byte_cmp_u8x16);

        // Count the scalar `start_bits` (already needed by the peel) rather than `wasm_i8x16_popcnt` on a vector.
        sz_size_t const window_matches = (sz_size_t)sz_u32_popcount(start_bits);
        sz_size_t const emit_count = sz_min_of_two(window_matches, matches_capacity - count);
        if (emit_count)
            sz_utf8_iterate_peel_v128_(start_bits, two_byte_bits, three_byte_bits, emit_count, position,
                                       match_offsets + count, match_lengths + count);
        count += emit_count;
        if (count == matches_capacity) { // output buffer full: resume past the last emitted match
            position = match_offsets[count - 1] + match_lengths[count - 1];
            break;
        }
        position += 14;
    }

    count += sz_utf8_whitespaces_serial_((sz_cptr_t)(text_u8 + position), length - position, position,
                                         match_offsets + count, match_lengths + count, matches_capacity - count,
                                         bytes_consumed);
    sz_assert_(sz_utf8_batch_consistent_(length, matches_capacity, count, bytes_consumed ? *bytes_consumed : length,
                                         match_offsets, match_lengths));
    return count;
}

#pragma endregion Multistep newline and whitespace iteration

#pragma region Membership

/** Lookup in a 32-byte table, as two 16-byte swizzles; an index outside a half reads zero there. */
STRINGZILLA_INLINE v128_t sz_delimiter_lookup32_v128_(sz_u8_t const *table, v128_t index_u8x16) {
    v128_t const low_half_u8x16 = wasm_v128_load(table), high_half_u8x16 = wasm_v128_load(table + 16);
    return wasm_v128_or(wasm_i8x16_swizzle(low_half_u8x16, index_u8x16),
                        wasm_i8x16_swizzle(high_half_u8x16, wasm_i8x16_sub(index_u8x16, wasm_i8x16_splat(16))));
}

/** Per-lane single-bit test `(bitmap_byte >> (low & 7)) & 1`, returned as 0x00/0xFF lanes. */
STRINGZILLA_INLINE v128_t sz_delimiter_test_bit_v128_(v128_t bitmap_byte_u8x16, v128_t low_u8x16) {
    static sz_u8_t const bit_for_low3[16] = {1, 2, 4, 8, 16, 32, 64, 128, 0, 0, 0, 0, 0, 0, 0, 0};
    v128_t const bit_mask_u8x16 = wasm_i8x16_swizzle(wasm_v128_load(bit_for_low3),
                                                     wasm_v128_and(low_u8x16, wasm_i8x16_splat(0x07)));
    return wasm_i8x16_ne(wasm_v128_and(bitmap_byte_u8x16, bit_mask_u8x16), wasm_i8x16_splat(0));
}

/** BMP delimiter membership of one quarter as 0x00/0xFF lanes, one distinct high byte per round. */
STRINGZILLA_INLINE v128_t sz_delimiter_bmp_membership_v128_(v128_t window_u8x16, v128_t high_in_u8x16,
                                                            v128_t low_in_u8x16) {
    v128_t const is_ascii_u8x16 = wasm_u8x16_lt(window_u8x16, wasm_i8x16_splat((sz_i8_t)0x80));
    v128_t const high_u8x16 = wasm_v128_andnot(high_in_u8x16, is_ascii_u8x16);
    v128_t const low_u8x16 = wasm_v128_bitselect(window_u8x16, low_in_u8x16, is_ascii_u8x16);

    v128_t const high_is_zero_u8x16 = wasm_i8x16_eq(high_u8x16, wasm_i8x16_splat(0));
    v128_t const row0_byte_u8x16 = sz_delimiter_lookup32_v128_(sz_utf8_delimiter_bmp_bitmaps_,
                                                               wasm_u8x16_shr(low_u8x16, 3));
    v128_t result_u8x16 = wasm_v128_and(sz_delimiter_test_bit_v128_(row0_byte_u8x16, low_u8x16), high_is_zero_u8x16);

    v128_t const suspicious_byte_u8x16 = sz_delimiter_lookup32_v128_(sz_utf8_delimiter_bmp_suspicious_highs_,
                                                                     wasm_u8x16_shr(high_u8x16, 3));
    v128_t const is_continuation_u8x16 = wasm_i8x16_eq(wasm_v128_and(window_u8x16, wasm_i8x16_splat((sz_i8_t)0xC0)),
                                                       wasm_i8x16_splat((sz_i8_t)0x80));
    v128_t unresolved_u8x16 = wasm_v128_andnot(
        wasm_v128_andnot(sz_delimiter_test_bit_v128_(suspicious_byte_u8x16, high_u8x16), high_is_zero_u8x16),
        is_continuation_u8x16);
    sz_u8_t high_bytes[16];
    wasm_v128_store(high_bytes, high_u8x16);
    while (wasm_v128_any_true(unresolved_u8x16)) {
        sz_u8_t const shared_high = high_bytes[sz_u64_ctz(sz_utf8_movemask16_v128_(unresolved_u8x16))];
        v128_t const same_u8x16 = wasm_v128_and(unresolved_u8x16,
                                                wasm_i8x16_eq(high_u8x16, wasm_i8x16_splat((sz_i8_t)shared_high)));
        v128_t const row_byte_u8x16 = sz_delimiter_lookup32_v128_(
            sz_utf8_delimiter_bmp_bitmaps_ + (sz_size_t)sz_utf8_delimiter_bmp_block_[shared_high] * 32,
            wasm_u8x16_shr(low_u8x16, 3));
        result_u8x16 = wasm_v128_or(result_u8x16,
                                    wasm_v128_and(sz_delimiter_test_bit_v128_(row_byte_u8x16, low_u8x16), same_u8x16));
        unresolved_u8x16 = wasm_v128_andnot(unresolved_u8x16, same_u8x16);
    }
    return result_u8x16;
}

/** Astral delimiter membership for one quarter as 0x00/0xFF lanes, keyed on the
 *  `(cp >> 16, (cp >> 8) & 0xFF)` pair of each four-byte lead. */
STRINGZILLA_INLINE v128_t sz_delimiter_astral_membership_v128_(v128_t window_u8x16, v128_t next1_u8x16,
                                                               v128_t next2_u8x16, v128_t next3_u8x16) {
    v128_t const b0_u8x16 = wasm_v128_and(window_u8x16, wasm_i8x16_splat(0x07));
    v128_t const b1_u8x16 = wasm_v128_and(next1_u8x16, wasm_i8x16_splat(0x3F));
    v128_t const b2_u8x16 = wasm_v128_and(next2_u8x16, wasm_i8x16_splat(0x3F));
    v128_t const b3_u8x16 = wasm_v128_and(next3_u8x16, wasm_i8x16_splat(0x3F));

    v128_t const codepoint_high_u8x16 = wasm_v128_or(wasm_i8x16_shl(b0_u8x16, 2), wasm_u8x16_shr(b1_u8x16, 4));
    v128_t const sub_u8x16 = wasm_v128_or(wasm_i8x16_shl(b1_u8x16, 4), wasm_u8x16_shr(b2_u8x16, 2));
    v128_t const low8_u8x16 = wasm_v128_or(wasm_i8x16_shl(b2_u8x16, 6), b3_u8x16);

    v128_t result_u8x16 = wasm_i8x16_splat(0);
    v128_t remaining_u8x16 = wasm_v128_and(wasm_v128_and(wasm_u8x16_ge(window_u8x16, wasm_i8x16_splat((sz_i8_t)0xF0)),
                                                         wasm_u8x16_le(codepoint_high_u8x16, wasm_i8x16_splat(0x10))),
                                           wasm_i8x16_ne(codepoint_high_u8x16, wasm_i8x16_splat(0)));
    sz_u8_t codepoint_high_bytes[16], sub_bytes[16];
    wasm_v128_store(codepoint_high_bytes, codepoint_high_u8x16);
    wasm_v128_store(sub_bytes, sub_u8x16);
    while (wasm_v128_any_true(remaining_u8x16)) {
        int const lane = sz_u64_ctz(sz_utf8_movemask16_v128_(remaining_u8x16));
        sz_u8_t const shared_plane = codepoint_high_bytes[lane], shared_sub = sub_bytes[lane];
        sz_u8_t const *row =
            sz_utf8_delimiter_astral_bitmaps_ +
            (sz_size_t)sz_utf8_delimiter_astral_l2_[(sz_size_t)sz_utf8_delimiter_astral_l1_[shared_plane - 1] * 256 +
                                                    shared_sub] *
                32;
        v128_t const same_u8x16 = wasm_v128_and(
            remaining_u8x16, wasm_v128_and(wasm_i8x16_eq(codepoint_high_u8x16, wasm_i8x16_splat((sz_i8_t)shared_plane)),
                                           wasm_i8x16_eq(sub_u8x16, wasm_i8x16_splat((sz_i8_t)shared_sub))));
        v128_t const row_byte_u8x16 = sz_delimiter_lookup32_v128_(row, wasm_u8x16_shr(low8_u8x16, 3));
        result_u8x16 = wasm_v128_or(result_u8x16,
                                    wasm_v128_and(sz_delimiter_test_bit_v128_(row_byte_u8x16, low8_u8x16), same_u8x16));
        remaining_u8x16 = wasm_v128_andnot(remaining_u8x16, same_u8x16);
    }
    return result_u8x16;
}

/** Per-lane UTF-8 validity of codepoint-start lanes, mirroring @ref sz_rune_decode. */
STRINGZILLA_INLINE sz_u64_t sz_delimiter_valid_starts_v128_(sz_utf8_rune_window_v128_t const *decoded,
                                                            v128_t const *next1_u8x16, v128_t const *next2_u8x16,
                                                            v128_t const *next3_u8x16) {
    v128_t const continuation_mask_u8x16 = wasm_i8x16_splat((sz_i8_t)0xC0),
                 continuation_pattern_u8x16 = wasm_i8x16_splat((sz_i8_t)0x80);
    v128_t valid_bool_u8x16[4];
    for (int quarter = 0; quarter < 4; ++quarter) {
        v128_t const here_u8x16 = decoded->window_u8x16s[quarter];
        v128_t const n1_u8x16 = next1_u8x16[quarter];
        v128_t const c1_ok_u8x16 = wasm_i8x16_eq(wasm_v128_and(n1_u8x16, continuation_mask_u8x16),
                                                 continuation_pattern_u8x16);
        v128_t const c2_ok_u8x16 = wasm_i8x16_eq(wasm_v128_and(next2_u8x16[quarter], continuation_mask_u8x16),
                                                 continuation_pattern_u8x16);
        v128_t const c3_ok_u8x16 = wasm_i8x16_eq(wasm_v128_and(next3_u8x16[quarter], continuation_mask_u8x16),
                                                 continuation_pattern_u8x16);
        v128_t const ascii_u8x16 = wasm_u8x16_lt(here_u8x16, wasm_i8x16_splat((sz_i8_t)0x80));

        v128_t const is_two_u8x16 = wasm_v128_and(wasm_u8x16_ge(here_u8x16, wasm_i8x16_splat((sz_i8_t)0xC0)),
                                                  wasm_u8x16_lt(here_u8x16, wasm_i8x16_splat((sz_i8_t)0xE0)));
        v128_t const is_three_u8x16 = wasm_v128_and(wasm_u8x16_ge(here_u8x16, wasm_i8x16_splat((sz_i8_t)0xE0)),
                                                    wasm_u8x16_lt(here_u8x16, wasm_i8x16_splat((sz_i8_t)0xF0)));
        v128_t const is_four_u8x16 = wasm_u8x16_ge(here_u8x16, wasm_i8x16_splat((sz_i8_t)0xF0));

        v128_t const two_ok_u8x16 = wasm_v128_and(c1_ok_u8x16,
                                                  wasm_u8x16_ge(here_u8x16, wasm_i8x16_splat((sz_i8_t)0xC2)));

        v128_t const lead_e0_u8x16 = wasm_i8x16_eq(here_u8x16, wasm_i8x16_splat((sz_i8_t)0xE0));
        v128_t const lead_ed_u8x16 = wasm_i8x16_eq(here_u8x16, wasm_i8x16_splat((sz_i8_t)0xED));
        v128_t const n1_lt_a0_u8x16 = wasm_u8x16_lt(n1_u8x16, wasm_i8x16_splat((sz_i8_t)0xA0));
        v128_t const bad_three_u8x16 = wasm_v128_or(wasm_v128_and(lead_e0_u8x16, n1_lt_a0_u8x16),
                                                    wasm_v128_andnot(lead_ed_u8x16, n1_lt_a0_u8x16));
        v128_t const three_ok_u8x16 = wasm_v128_andnot(wasm_v128_and(c1_ok_u8x16, c2_ok_u8x16), bad_three_u8x16);

        v128_t const lead_f0_u8x16 = wasm_i8x16_eq(here_u8x16, wasm_i8x16_splat((sz_i8_t)0xF0));
        v128_t const lead_f4_u8x16 = wasm_i8x16_eq(here_u8x16, wasm_i8x16_splat((sz_i8_t)0xF4));
        v128_t const n1_lt_90_u8x16 = wasm_u8x16_lt(n1_u8x16, wasm_i8x16_splat((sz_i8_t)0x90));
        v128_t const bad_four_u8x16 = wasm_v128_or(wasm_u8x16_gt(here_u8x16, wasm_i8x16_splat((sz_i8_t)0xF4)),
                                                   wasm_v128_or(wasm_v128_and(lead_f0_u8x16, n1_lt_90_u8x16),
                                                                wasm_v128_andnot(lead_f4_u8x16, n1_lt_90_u8x16)));
        v128_t const four_ok_u8x16 = wasm_v128_andnot(
            wasm_v128_and(wasm_v128_and(c1_ok_u8x16, c2_ok_u8x16), c3_ok_u8x16), bad_four_u8x16);

        valid_bool_u8x16[quarter] = wasm_v128_or(
            wasm_v128_or(ascii_u8x16, wasm_v128_and(is_two_u8x16, two_ok_u8x16)),
            wasm_v128_or(wasm_v128_and(is_three_u8x16, three_ok_u8x16), wasm_v128_and(is_four_u8x16, four_ok_u8x16)));
    }
    sz_u64_t const valid = sz_utf8_mask_combine_v128_(valid_bool_u8x16[0], valid_bool_u8x16[1], valid_bool_u8x16[2],
                                                      valid_bool_u8x16[3]);

    // A lead whose declared span runs past `loaded` is truncated and never valid.
    // The serial path re-syncs one byte at a time.
    sz_size_t const loaded = decoded->loaded;
    sz_u64_t const truncated = (decoded->two_byte_starts & ~sz_u64_mask_until_serial_(loaded >= 1 ? loaded - 1 : 0)) |
                               (decoded->three_byte_starts & ~sz_u64_mask_until_serial_(loaded >= 2 ? loaded - 2 : 0)) |
                               (decoded->four_byte_starts & ~sz_u64_mask_until_serial_(loaded >= 3 ? loaded - 3 : 0));
    return valid & sz_u64_mask_until_serial_(loaded) & ~truncated;
}

#pragma endregion Membership

#pragma region Forward driver

STRINGZILLA_INLINE sz_size_t sz_utf8_delimiters_v128_(  //
    sz_cptr_t text, sz_size_t length,                   //
    sz_size_t *match_offsets, sz_size_t *match_lengths, //
    sz_size_t matches_capacity, sz_size_t *bytes_consumed) {
    sz_u8_t const *const text_u8 = (sz_u8_t const *)text;

    sz_size_t base = 0, count = 0;
    while (base < length && count < matches_capacity) {
        sz_utf8_rune_window_v128_t const decoded = sz_utf8_rune_decode_window_v128_(text_u8 + base, length - base);
        sz_size_t const loaded = decoded.loaded;
        sz_u64_t const loaded_mask = sz_u64_mask_until_serial_(loaded);

        sz_size_t byte_span = loaded;
        sz_u64_t hits;

        // All-ASCII window: membership is one lookup into the first half of bitmap row 0.
        int const all_ascii = decoded.codepoint_starts == loaded_mask &&
                              !(decoded.two_byte_starts | decoded.three_byte_starts | decoded.four_byte_starts);
        if (all_ascii) {
            v128_t const row0_u8x16 = wasm_v128_load(sz_utf8_delimiter_bmp_bitmaps_);
            v128_t member_bool_u8x16[4];
            for (int quarter = 0; quarter < 4; ++quarter)
                member_bool_u8x16[quarter] = sz_delimiter_test_bit_v128_(
                    wasm_i8x16_swizzle(row0_u8x16, wasm_u8x16_shr(decoded.window_u8x16s[quarter], 3)),
                    decoded.window_u8x16s[quarter]);
            hits = sz_utf8_mask_combine_v128_(member_bool_u8x16[0], member_bool_u8x16[1], member_bool_u8x16[2],
                                              member_bool_u8x16[3]) &
                   loaded_mask;
        }
        else {
            v128_t next1_u8x16[4], next2_u8x16[4], next3_u8x16[4];
            sz_utf8_forward_neighbours_v128_(decoded.window_u8x16s, next1_u8x16, next2_u8x16, next3_u8x16);

            // A multi-byte lead near the 64-byte edge whose span runs past `loaded`
            // is deferred to the next window.
            if (loaded >= 64)
                byte_span = sz_utf8_delimiter_complete_span_(decoded.two_byte_starts, decoded.three_byte_starts,
                                                             decoded.four_byte_starts, loaded);
            sz_u64_t const span_mask = sz_u64_mask_until_serial_(byte_span);

            sz_u64_t const valid_starts = sz_delimiter_valid_starts_v128_(&decoded, next1_u8x16, next2_u8x16,
                                                                          next3_u8x16) &
                                          decoded.codepoint_starts & span_mask;

            sz_u64_t const four_byte = decoded.four_byte_starts & span_mask;
            sz_u64_t member = 0;
            for (int quarter = 0; quarter < 4; ++quarter) {
                v128_t const bmp_u8x16 = sz_delimiter_bmp_membership_v128_(decoded.window_u8x16s[quarter],
                                                                           decoded.high_byte_u8x16s[quarter],
                                                                           decoded.low_byte_u8x16s[quarter]);
                member |= sz_utf8_movemask16_v128_(bmp_u8x16) << (16 * quarter);
            }
            if (four_byte) {
                sz_u64_t astral_member = 0;
                for (int quarter = 0; quarter < 4; ++quarter) {
                    v128_t const astral_u8x16 = sz_delimiter_astral_membership_v128_(
                        decoded.window_u8x16s[quarter], next1_u8x16[quarter], next2_u8x16[quarter],
                        next3_u8x16[quarter]);
                    astral_member |= sz_utf8_movemask16_v128_(astral_u8x16) << (16 * quarter);
                }
                member = (member & ~four_byte) | (astral_member & four_byte);
            }

            hits = member & valid_starts;
        }
        while (hits && count < matches_capacity) {
            sz_size_t const lane = (sz_size_t)sz_u64_ctz(hits);
            hits &= hits - 1;
            sz_size_t length_at_lane = 1;
            length_at_lane += (decoded.two_byte_starts >> lane) & 1;
            length_at_lane += ((decoded.three_byte_starts >> lane) & 1) * 2;
            length_at_lane += ((decoded.four_byte_starts >> lane) & 1) * 3;
            match_offsets[count] = base + lane, match_lengths[count] = length_at_lane, ++count;
        }
        // Output buffer full: resume past the last emitted match, not at the window edge.
        if (count == matches_capacity) {
            base = match_offsets[count - 1] + match_lengths[count - 1];
            if (bytes_consumed) *bytes_consumed = base;
            return count;
        }
        base += byte_span ? byte_span : 1;
    }

    if (bytes_consumed) *bytes_consumed = base;
    return count;
}

#pragma endregion Forward driver

#if STRINGZILLA_TARGET_V128

STRINGZILLA_API sz_status_t sz_utf8_newlines_v128(                                  //
    sz_cptr_t text, sz_size_t length,                                               //
    sz_size_t *match_offsets, sz_size_t *match_lengths, sz_size_t matches_capacity, //
    sz_size_t *matches_count, sz_size_t *bytes_consumed, sz_stream_t stream) {
    sz_assert_(stream == STRINGZILLA_NULL);
    *matches_count = sz_utf8_newlines_v128_(text, length, match_offsets, match_lengths, matches_capacity,
                                            bytes_consumed);
    return sz_success_k;
}

STRINGZILLA_API sz_status_t sz_utf8_whitespaces_v128(                               //
    sz_cptr_t text, sz_size_t length,                                               //
    sz_size_t *match_offsets, sz_size_t *match_lengths, sz_size_t matches_capacity, //
    sz_size_t *matches_count, sz_size_t *bytes_consumed, sz_stream_t stream) {
    sz_assert_(stream == STRINGZILLA_NULL);
    *matches_count = sz_utf8_whitespaces_v128_(text, length, match_offsets, match_lengths, matches_capacity,
                                               bytes_consumed);
    return sz_success_k;
}

STRINGZILLA_API sz_status_t sz_utf8_delimiters_v128(                                //
    sz_cptr_t text, sz_size_t length,                                               //
    sz_size_t *match_offsets, sz_size_t *match_lengths, sz_size_t matches_capacity, //
    sz_size_t *matches_count, sz_size_t *bytes_consumed, sz_stream_t stream) {
    sz_assert_(stream == STRINGZILLA_NULL);
    *matches_count = sz_utf8_delimiters_v128_(text, length, match_offsets, match_lengths, matches_capacity,
                                              bytes_consumed);
    sz_assert_(sz_utf8_batch_consistent_(length, matches_capacity, *matches_count,
                                         bytes_consumed ? *bytes_consumed : length, match_offsets, match_lengths));
    return sz_success_k;
}

#endif // STRINGZILLA_TARGET_V128

#if defined(__clang__)
#pragma clang attribute pop
#endif
#endif // STRINGZILLA_ARCH_WASM_V128_
#endif // STRINGZILLA_ARCH_WASM_

#ifdef __cplusplus
}
#endif

#endif // STRINGZILLA_UTF8_TOKENS_V128_H_
