/**
 *  @brief WebAssembly SIMD128 backend for UTF-8 newline and whitespace delimiter scanning.
 *  @file include/stringzilla/utf8_tokens/v128.h
 *  @author Ash Vardanian
 */
#ifndef STRINGZILLA_UTF8_TOKENS_V128_H_
#define STRINGZILLA_UTF8_TOKENS_V128_H_

#include "stringzilla/types.h"
#include "stringzilla/utf8_tokens/serial.h"
#include "stringzilla/utf8_runes/v128.h"

#ifdef __cplusplus
extern "C" {
#endif

#if SZ_USE_V128
#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("simd128"))), apply_to = function)
#endif

SZ_INTERNAL v128_t sz_utf8_rotate1_v128_(v128_t v) {
    return wasm_i8x16_shuffle(v, v, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0);
}

SZ_INTERNAL v128_t sz_utf8_rotate2_v128_(v128_t v) {
    return wasm_i8x16_shuffle(v, v, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0, 1);
}

#pragma region Multistep newline / whitespace iteration

/**
 *  @brief  Peel the tile's first @p emit_count matches with a `wasm_i8x16_swizzle` left-pack, 4 lanes per block.
 *
 *  Walks the 16-lane @p start_bits mask in four ascending 4-lane sub-blocks, gathering each block's set
 *  `(position+lane, length)` pairs to the front of a 16-wide stack scratch via one swizzle from `compact_lut`,
 *  then copies the low @p emit_count entries out - ascending lane order, byte-exact, no per-match `ctz`.
 */
SZ_INTERNAL void sz_utf8_iterate_peel_v128_(                                   //
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
    static sz_u8_t const popcount_lut[16] = {0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4};

    sz_size_t scratch_offsets[16], scratch_lengths[16];
    sz_size_t filled = 0;
    for (sz_size_t sub_block = 0; sub_block < 4; ++sub_block) {
        sz_size_t const base_lane = sub_block * 4;
        sz_u32_t const submask = (start_bits >> base_lane) & 0xFu;
        if (!submask) continue;

        // Per-lane length: 1, plus 1 on a 2-byte start, plus 2 on a 3-byte start (the masks are disjoint).
        sz_u32_t const two_byte_sub = (two_byte_starts >> base_lane) & 0xFu;
        sz_u32_t const three_byte_sub = (three_byte_starts >> base_lane) & 0xFu;
        v128_t const candidate_offsets = wasm_u32x4_make(
            (sz_u32_t)(position + base_lane + 0), (sz_u32_t)(position + base_lane + 1),
            (sz_u32_t)(position + base_lane + 2), (sz_u32_t)(position + base_lane + 3));
        v128_t const candidate_lengths = wasm_u32x4_make(
            1u + ((two_byte_sub >> 0) & 1u) + 2u * ((three_byte_sub >> 0) & 1u),
            1u + ((two_byte_sub >> 1) & 1u) + 2u * ((three_byte_sub >> 1) & 1u),
            1u + ((two_byte_sub >> 2) & 1u) + 2u * ((three_byte_sub >> 2) & 1u),
            1u + ((two_byte_sub >> 3) & 1u) + 2u * ((three_byte_sub >> 3) & 1u));

        v128_t const permutation = wasm_v128_load(compact_lut[submask]);
        wasm_v128_store(scratch_offsets + filled, wasm_i8x16_swizzle(candidate_offsets, permutation));
        wasm_v128_store(scratch_lengths + filled, wasm_i8x16_swizzle(candidate_lengths, permutation));
        filled += popcount_lut[submask];
    }

    for (sz_size_t emitted = 0; emitted < emit_count; ++emitted)
        match_offsets[emitted] = scratch_offsets[emitted], match_lengths[emitted] = scratch_lengths[emitted];
}

SZ_PUBLIC sz_size_t sz_utf8_newlines_v128(              //
    sz_cptr_t text, sz_size_t length,                   //
    sz_size_t *match_offsets, sz_size_t *match_lengths, //
    sz_size_t matches_capacity, sz_size_t *bytes_consumed) {

    sz_u8_t const *text_u8 = (sz_u8_t const *)text;
    sz_size_t count = 0, position = 0;

    v128_t newline_u8x16 = wasm_i8x16_splat('\n');
    v128_t vertical_tab_vec = wasm_i8x16_splat('\v');
    v128_t form_feed_vec = wasm_i8x16_splat('\f');
    v128_t carriage_return_vec = wasm_i8x16_splat('\r');
    v128_t lead_c2_vec = wasm_i8x16_splat((sz_i8_t)0xC2);
    v128_t x_85_vec = wasm_i8x16_splat((sz_i8_t)0x85);
    v128_t lead_e2_vec = wasm_i8x16_splat((sz_i8_t)0xE2);
    v128_t byte_80_vec = wasm_i8x16_splat((sz_i8_t)0x80);
    v128_t x_a8_vec = wasm_i8x16_splat((sz_i8_t)0xA8);
    v128_t x_a9_vec = wasm_i8x16_splat((sz_i8_t)0xA9);

    // We trust delimiter starts only in lanes [0,13] and step by 14, so any 2-/3-byte delimiter is fully loaded.
    sz_u32_t const trusted_lanes_mask = 0x3FFFu; // lanes [0,13]

    while (position + 16 <= length && count < matches_capacity) {
        v128_t window = wasm_v128_load(text_u8 + position);
        v128_t window1 = sz_utf8_rotate1_v128_(window); // next lane
        v128_t window2 = sz_utf8_rotate2_v128_(window); // lane after next

        // 1-byte matches: \n \v \f \r (the contiguous control range '\n'..'\f' plus '\r').
        v128_t newline_cmp = wasm_i8x16_eq(window, newline_u8x16);
        v128_t carriage_return_cmp = wasm_i8x16_eq(window, carriage_return_vec);
        v128_t one_byte_cmp = wasm_v128_or(wasm_v128_or(newline_cmp, wasm_i8x16_eq(window, vertical_tab_vec)),
                                           wasm_v128_or(wasm_i8x16_eq(window, form_feed_vec), carriage_return_cmp));

        // 2-byte: CRLF (\r\n, one match) & NEL (C2 85) - computed unconditionally.
        v128_t crlf_cmp = wasm_v128_and(carriage_return_cmp, wasm_i8x16_eq(window1, newline_u8x16));
        v128_t nel_cmp = wasm_v128_and(wasm_i8x16_eq(window, lead_c2_vec), wasm_i8x16_eq(window1, x_85_vec));
        v128_t two_byte_cmp = wasm_v128_or(crlf_cmp, nel_cmp);

        // 3-byte: LS (E2 80 A8) & PS (E2 80 A9).
        v128_t lead_e280_cmp = wasm_v128_and(wasm_i8x16_eq(window, lead_e2_vec), wasm_i8x16_eq(window1, byte_80_vec));
        v128_t three_byte_cmp = wasm_v128_and(
            lead_e280_cmp, wasm_v128_or(wasm_i8x16_eq(window2, x_a8_vec), wasm_i8x16_eq(window2, x_a9_vec)));

        // CRLF's trailing LF must not also be emitted: an LF whose previous lane is a CR.
        v128_t carriage_return_previous = wasm_i8x16_shuffle(wasm_i8x16_splat(0), carriage_return_cmp, //
                                                             15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
                                                             30);
        v128_t lf_of_crlf_cmp = wasm_v128_and(newline_cmp, carriage_return_previous);

        v128_t starts_cmp = wasm_v128_andnot(wasm_v128_or(wasm_v128_or(one_byte_cmp, nel_cmp), three_byte_cmp),
                                             lf_of_crlf_cmp);

        sz_u32_t start_bits = (sz_u32_t)wasm_i8x16_bitmask(starts_cmp) & trusted_lanes_mask;
        sz_u32_t two_byte_bits = (sz_u32_t)wasm_i8x16_bitmask(two_byte_cmp);
        sz_u32_t three_byte_bits = (sz_u32_t)wasm_i8x16_bitmask(three_byte_cmp);

        // Suppress a leading LF already consumed by a CRLF that straddled the previous tile edge.
        if (position != 0 && text_u8[position - 1] == '\r')
            start_bits &= ~((sz_u32_t)wasm_i8x16_bitmask(newline_cmp) & 1u);

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
    return count;
}

SZ_PUBLIC sz_size_t sz_utf8_whitespaces_v128(           //
    sz_cptr_t text, sz_size_t length,                   //
    sz_size_t *match_offsets, sz_size_t *match_lengths, //
    sz_size_t matches_capacity, sz_size_t *bytes_consumed) {

    sz_u8_t const *text_u8 = (sz_u8_t const *)text;
    sz_size_t count = 0, position = 0;

    v128_t tab_vec = wasm_i8x16_splat('\t');
    v128_t carriage_return_vec = wasm_i8x16_splat('\r');
    v128_t x_20_vec = wasm_i8x16_splat(' ');
    v128_t lead_c2_vec = wasm_i8x16_splat((sz_i8_t)0xC2);
    v128_t x_85_vec = wasm_i8x16_splat((sz_i8_t)0x85);
    v128_t x_a0_vec = wasm_i8x16_splat((sz_i8_t)0xA0);
    v128_t x_e1_vec = wasm_i8x16_splat((sz_i8_t)0xE1);
    v128_t lead_e2_vec = wasm_i8x16_splat((sz_i8_t)0xE2);
    v128_t x_e3_vec = wasm_i8x16_splat((sz_i8_t)0xE3);
    v128_t x_9a_vec = wasm_i8x16_splat((sz_i8_t)0x9A);
    v128_t byte_80_vec = wasm_i8x16_splat((sz_i8_t)0x80);
    v128_t x_81_vec = wasm_i8x16_splat((sz_i8_t)0x81);
    v128_t x_8d_vec = wasm_i8x16_splat((sz_i8_t)0x8D);
    v128_t x_a8_vec = wasm_i8x16_splat((sz_i8_t)0xA8);
    v128_t x_a9_vec = wasm_i8x16_splat((sz_i8_t)0xA9);
    v128_t x_af_vec = wasm_i8x16_splat((sz_i8_t)0xAF);
    v128_t x_9f_vec = wasm_i8x16_splat((sz_i8_t)0x9F);

    sz_u32_t const trusted_lanes_mask = 0x3FFFu; // lanes [0,13]

    while (position + 16 <= length && count < matches_capacity) {
        v128_t window = wasm_v128_load(text_u8 + position);
        v128_t window1 = sz_utf8_rotate1_v128_(window); // next lane
        v128_t window2 = sz_utf8_rotate2_v128_(window); // lane after next

        // 1-byte: space, plus the contiguous range [\t, \r] == [9, 13].
        v128_t one_byte_cmp = wasm_v128_or(
            wasm_i8x16_eq(window, x_20_vec),
            wasm_v128_and(wasm_u8x16_ge(window, tab_vec), wasm_u8x16_le(window, carriage_return_vec)));

        // 2-byte: C2 85 (NEL), C2 A0 (NBSP).
        v128_t lead_c2_cmp = wasm_i8x16_eq(window, lead_c2_vec);
        v128_t two_byte_cmp = wasm_v128_and(
            lead_c2_cmp, wasm_v128_or(wasm_i8x16_eq(window1, x_85_vec), wasm_i8x16_eq(window1, x_a0_vec)));

        // 3-byte: E1 9A 80 (ogham); E2 80 [80-8D]; E2 80 AF; E2 81 9F; E2 80 A8/A9; E3 80 80.
        v128_t window1_is_80 = wasm_i8x16_eq(window1, byte_80_vec);
        v128_t lead_e280_cmp = wasm_v128_and(wasm_i8x16_eq(window, lead_e2_vec), window1_is_80);
        v128_t ogham_cmp = wasm_v128_and(
            wasm_i8x16_eq(window, x_e1_vec),
            wasm_v128_and(wasm_i8x16_eq(window1, x_9a_vec), wasm_i8x16_eq(window2, byte_80_vec)));
        v128_t range_e280_cmp = wasm_v128_and(
            lead_e280_cmp, wasm_v128_and(wasm_u8x16_ge(window2, byte_80_vec), wasm_u8x16_le(window2, x_8d_vec)));
        v128_t nnbsp_cmp = wasm_v128_and(lead_e280_cmp, wasm_i8x16_eq(window2, x_af_vec));
        v128_t mmsp_cmp = wasm_v128_and(
            wasm_v128_and(wasm_i8x16_eq(window, lead_e2_vec), wasm_i8x16_eq(window1, x_81_vec)),
            wasm_i8x16_eq(window2, x_9f_vec));
        v128_t line_cmp = wasm_v128_and(lead_e280_cmp, wasm_i8x16_eq(window2, x_a8_vec));
        v128_t paragraph_cmp = wasm_v128_and(lead_e280_cmp, wasm_i8x16_eq(window2, x_a9_vec));
        v128_t ideographic_cmp = wasm_v128_and(wasm_v128_and(wasm_i8x16_eq(window, x_e3_vec), window1_is_80),
                                               wasm_i8x16_eq(window2, byte_80_vec));
        v128_t three_byte_cmp = wasm_v128_or(
            wasm_v128_or(wasm_v128_or(ogham_cmp, range_e280_cmp), wasm_v128_or(nnbsp_cmp, mmsp_cmp)),
            wasm_v128_or(wasm_v128_or(line_cmp, paragraph_cmp), ideographic_cmp));

        v128_t starts_cmp = wasm_v128_or(wasm_v128_or(one_byte_cmp, two_byte_cmp), three_byte_cmp);

        sz_u32_t start_bits = (sz_u32_t)wasm_i8x16_bitmask(starts_cmp) & trusted_lanes_mask;
        sz_u32_t two_byte_bits = (sz_u32_t)wasm_i8x16_bitmask(two_byte_cmp);
        sz_u32_t three_byte_bits = (sz_u32_t)wasm_i8x16_bitmask(three_byte_cmp);

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
    return count;
}

#pragma endregion // Multistep newline / whitespace iteration

#if defined(__clang__)
#pragma clang attribute pop
#endif
#endif // SZ_USE_V128

#ifdef __cplusplus
}
#endif

#endif // STRINGZILLA_UTF8_TOKENS_V128_H_
