/**
 *  @brief WebAssembly SIMD128 backend for utf8.
 *  @file include/stringzilla/utf8_iterate/v128.h
 *  @author Ash Vardanian
 *  @sa utf8.h
 */
#ifndef STRINGZILLA_UTF8_ITERATE_V128_H_
#define STRINGZILLA_UTF8_ITERATE_V128_H_

#include "stringzilla/types.h"
#include "stringzilla/utf8_iterate/serial.h"

#ifdef __cplusplus
extern "C" {
#endif

#if SZ_USE_V128

/*  WASM SIMD128 is 16 bytes wide like NEON. We use `wasm_i8x16_bitmask` (a true movemask producing
 *  one bit per byte) instead of NEON's nibble-mask trick, so each set bit maps directly to a byte
 *  offset. To peek at the following bytes within a register we rotate it with `wasm_i8x16_shuffle`,
 *  matching NEON's `vextq_u8(v, v, k)`. The trailing lanes that wrap around are zeroed by `drop`
 *  masks so they cannot create spurious matches. We advance by 14 bytes per window so a 3-byte
 *  sequence straddling the boundary is re-examined. */
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

/*  Multistep newline / whitespace iteration (WASM SIMD128).
 *
 *  Each FULL 16-byte tile is classified branchlessly into a `start_bits` byte-bitmask (every delimiter start in
 *  the trusted lanes [0,13]) plus the disjoint 2-/3-byte start bitmasks, all built with `wasm_i8x16_bitmask` (a
 *  true movemask producing one bit per byte, so each set bit maps straight to a byte offset). WASM SIMD128 has
 *  no `vpcompressb` and no masked store, so the peel left-packs with `wasm_i8x16_swizzle` (a single-register
 *  byte table lookup, the WASM analog of `vqtbl1q`): the 16-lane mask is walked in four ascending 4-lane
 *  sub-blocks - a v128 register holds exactly four 32-bit `sz_size_t` lanes, so a 4-lane block fits one swizzle
 *  - and each block compacts its `(position+lane, length)` pairs to the front of a fixed stack scratch with one
 *  permutation drawn from `compact_lut`, then the low `emit_count` entries are copied out. There is no
 *  per-match `ctz` and no data-dependent inner loop. We trust starts in lanes [0,13] and step 14 so any 2-/3-
 *  byte delimiter from a trusted lane is fully loaded; for newlines a 1-byte `t[pos-1] == '\r'` carry
 *  suppresses an LF that completes a CRLF straddling the tile edge (the only delimiter whose tail is itself a
 *  match). The caller computes the capacity cut like the other backends (`window_matches = popcount(start_bits)`,
 *  `emit_count = min(window_matches, matches_capacity - count)`), and on a full buffer resumes mid-tile past the
 *  last emitted match. */

/**
 *  @brief  Peel the tile's first `emit_count` matches with a `wasm_i8x16_swizzle` left-pack, 4 lanes per block.
 *
 *  Walks the 16-lane `start_bits` mask in four ascending 4-lane sub-blocks. Each block builds the four candidate
 *  `(position+lane, length)` `sz_size_t` pairs in one v128 apiece, gathers the set lanes to the front of a stack
 *  scratch with one `wasm_i8x16_swizzle` (driven by the 4-bit-mask `compact_lut` left-pack table), then the low
 *  `emit_count` entries are copied to the caller's output - preserving ascending lane order and the original
 *  truncation byte-for-byte, with no per-match `ctz`. The scratch is 16-wide so the trailing 4-lane spill of the
 *  last compacted block always lands inside it (at most 14 trusted matches per tile).
 */
SZ_INTERNAL void sz_utf8_iterate_peel_window_v128_(                            //
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

SZ_PUBLIC sz_size_t sz_utf8_find_newlines_v128(         //
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

        sz_size_t const window_matches = (sz_size_t)sz_u32_popcount(start_bits);
        sz_size_t const emit_count = sz_min_of_two(window_matches, matches_capacity - count);
        if (emit_count)
            sz_utf8_iterate_peel_window_v128_(start_bits, two_byte_bits, three_byte_bits, emit_count, position,
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
    count += sz_utf8_find_newlines_serial_((sz_cptr_t)(text_u8 + position), length - position, position,
                                           match_offsets + count, match_lengths + count, matches_capacity - count,
                                           bytes_consumed);
    return count;
}

SZ_PUBLIC sz_size_t sz_utf8_find_whitespaces_v128(      //
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

        sz_size_t const window_matches = (sz_size_t)sz_u32_popcount(start_bits);
        sz_size_t const emit_count = sz_min_of_two(window_matches, matches_capacity - count);
        if (emit_count)
            sz_utf8_iterate_peel_window_v128_(start_bits, two_byte_bits, three_byte_bits, emit_count, position,
                                              match_offsets + count, match_lengths + count);
        count += emit_count;
        if (count == matches_capacity) { // output buffer full: resume past the last emitted match
            position = match_offsets[count - 1] + match_lengths[count - 1];
            break;
        }
        position += 14;
    }

    count += sz_utf8_find_whitespaces_serial_((sz_cptr_t)(text_u8 + position), length - position, position,
                                              match_offsets + count, match_lengths + count, matches_capacity - count,
                                              bytes_consumed);
    return count;
}

#pragma endregion // Multistep newline / whitespace iteration

SZ_PUBLIC sz_size_t sz_utf8_count_v128(sz_cptr_t text, sz_size_t length) {
    sz_u8_t const *text_u8 = (sz_u8_t const *)text;
    // A continuation byte satisfies `(byte & 0xC0) == 0x80`, i.e. it lies in `[0x80, 0xBF]`, which as a
    // SIGNED int8 is `[-128, -65]` — exactly the values strictly less than `0xC0` (= -64). So a single
    // signed compare `byte < 0xC0` flags continuation bytes, replacing the `and` + `eq` pair.
    v128_t continuation_threshold_vec = wasm_i8x16_splat((sz_i8_t)0xC0);

    // The UTF-8 code-point count equals (#bytes - #continuation bytes). Instead of a per-block
    // `bitmask` + `popcount` horizontal reduction, we keep a per-lane counter in a vector register:
    // the compare yields 0xFF (= -1) for each continuation lane, so SUBTRACTING it adds +1 per match.
    // A u8 lane can hold up to 255 matches, so we flush into a wider u64 accumulator every 255 blocks
    // and reduce ONCE at the end.
    sz_size_t total_bytes = (length / 16) * 16;
    sz_u128_vec_t cont8_vec, cont64_vec;
    cont8_vec.v128 = wasm_u64x2_splat(0);  // 16x u8 per-lane match counters
    cont64_vec.v128 = wasm_u64x2_splat(0); // 2x u64 accumulated counters

    while (length >= 16) {
        sz_size_t blocks = length / 16;
        if (blocks > 255) blocks = 255;
        length -= blocks * 16;
        for (sz_size_t i = 0; i < blocks; ++i, text_u8 += 16) {
            v128_t text_vec = wasm_v128_load(text_u8);
            v128_t continuation_vec = wasm_i8x16_lt(text_vec, continuation_threshold_vec);
            // `continuation_vec` lane is 0xFF (-1) on a match; subtract to increment the counter.
            cont8_vec.v128 = wasm_i8x16_sub(cont8_vec.v128, continuation_vec);
        }
        // Flush: sum the 16x u8 counters into 2x u64 lanes via two pairwise widenings.
        v128_t cont16 = wasm_u16x8_extadd_pairwise_u8x16(cont8_vec.v128);
        v128_t cont32 = wasm_u32x4_extadd_pairwise_u16x8(cont16);
        cont64_vec.v128 = wasm_i64x2_add( //
            cont64_vec.v128, wasm_i64x2_add(wasm_u64x2_extend_low_u32x4(cont32), wasm_u64x2_extend_high_u32x4(cont32)));
        cont8_vec.v128 = wasm_u64x2_splat(0);
    }

    sz_size_t continuation_count = (sz_size_t)(cont64_vec.u64s[0] + cont64_vec.u64s[1]);
    sz_size_t char_count = total_bytes - continuation_count;
    if (length) char_count += sz_utf8_count_serial((sz_cptr_t)text_u8, length);
    return char_count;
}

SZ_PUBLIC sz_cptr_t sz_utf8_find_nth_v128(sz_cptr_t text, sz_size_t length, sz_size_t n) {
    // No `pdep`-style instruction on WASM; the serial scan is already optimal here.
    return sz_utf8_find_nth_serial(text, length, n);
}

/*  `sz_utf8_unpack_chunk_v128` decodes UTF-8 into UTF-32 runes. The SIMD-clean case is a pure-ASCII
 *  prefix: every byte `< 0x80` is a single-byte code point whose rune value equals the byte, so 16
 *  bytes widen straight into 16 u32 runes via two zero-extends with NO branchy state machine. We scan
 *  16-byte windows; as soon as a window contains a non-ASCII byte (high bit set in any lane) or we run
 *  out of output capacity, we stop the vector loop and hand the *remaining* bytes to
 *  `sz_utf8_unpack_chunk_serial`, which owns all multi-byte / malformed / boundary-truncation logic.
 *  Because the ASCII fast path emits exactly the runes serial would (byte value, 1 byte advance) and
 *  the tail is the serial routine verbatim, the output is byte/value-identical to the serial reference. */
SZ_PUBLIC sz_cptr_t sz_utf8_unpack_chunk_v128(  //
    sz_cptr_t text, sz_size_t length,           //
    sz_rune_t *runes, sz_size_t runes_capacity, //
    sz_size_t *runes_unpacked) {

    sz_u8_t const *text_cursor = (sz_u8_t const *)text;
    sz_size_t runes_written = 0;

    // Vectorized pure-ASCII prefix: 16 single-byte runes per window.
    while (length - (sz_size_t)(text_cursor - (sz_u8_t const *)text) >= 16 && runes_written + 16 <= runes_capacity) {
        v128_t bytes = wasm_v128_load(text_cursor);
        // Any byte >= 0x80 has its sign bit set; if present, defer the whole window to serial.
        if (wasm_i8x16_bitmask(bytes) != 0) break;
        // Widen 16x u8 → 16x u32 and store as runes (zero-extension == the ASCII code point value).
        v128_t low_u16x8 = wasm_u16x8_extend_low_u8x16(bytes);
        v128_t high_u16x8 = wasm_u16x8_extend_high_u8x16(bytes);
        wasm_v128_store(runes + runes_written + 0, wasm_u32x4_extend_low_u16x8(low_u16x8));
        wasm_v128_store(runes + runes_written + 4, wasm_u32x4_extend_high_u16x8(low_u16x8));
        wasm_v128_store(runes + runes_written + 8, wasm_u32x4_extend_low_u16x8(high_u16x8));
        wasm_v128_store(runes + runes_written + 12, wasm_u32x4_extend_high_u16x8(high_u16x8));
        runes_written += 16;
        text_cursor += 16;
    }

    // Remainder (multi-byte sequences, ragged tail, or exhausted capacity) goes through serial.
    sz_size_t bytes_consumed = (sz_size_t)(text_cursor - (sz_u8_t const *)text);
    sz_size_t tail_unpacked = 0;
    sz_cptr_t tail_end = sz_utf8_unpack_chunk_serial((sz_cptr_t)text_cursor, length - bytes_consumed,
                                                     runes + runes_written, runes_capacity - runes_written,
                                                     &tail_unpacked);
    *runes_unpacked = runes_written + tail_unpacked;
    return tail_end;
}

/*  UAX-29 word boundary detection is a stateful machine: a candidate byte offset is a boundary based
 *  on the Word_Break properties of the surrounding code points, with several rules looking back and
 *  forward across multiple runes (WB4 Extend/Format/ZWJ skipping, WB6/WB7 mid-letter, WB11/WB12
 *  numeric, WB15/WB16 Regional-Indicator parity). The dominant cost in real text, however, is scanning
 *  through the *interior* of long ASCII/Latin words and numbers — every byte there is a non-boundary,
 *  yet the serial scanner pays a full property lookup + rule cascade per byte.
 *
 *  We accelerate exactly that case. The word-interior set is exactly the ASCII bytes {A-Z, a-z, 0-9,
 *  '_' = ExtendNumLet}, which `sz_utf8_word_break_nonboundary_mask_v128_` detects directly. For a candidate at byte
 *  offset `i`, if both `byte[i-1]` and `byte[i]` are word-interior, then *every* UAX-29 rule that could
 *  fire (WB5/8/9/10/13a/13b and their context-needing siblings) resolves to "do NOT break" — verified
 *  exhaustively against the serial decision table — because ASCII contains no Extend/Format/ZWJ/RI/
 *  Hebrew/Katakana, so WB4 ignorable-skipping and RI parity never apply between two such bytes. Those
 *  positions are skipped with a single `wasm_i8x16_bitmask`. Any byte that is non-ASCII, or whose pair
 *  is not a guaranteed interior, is handed to `sz_utf8_is_word_boundary_serial` verbatim, so the result
 *  is bit-for-bit identical to the serial reference (the hard, stateful sub-rules stay serial). */

/*  Per-class 16-bit lane mask for an all-ASCII window via `wasm_i8x16_bitmask`. Each Word_Break class is a
 *  small set of ASCII byte ranges/singletons (ALetter = A-Z|a-z via the 0x20 fold, Numeric = 0-9,
 *  ExtendNumLet = '_', MidLetter = ':', MidNum = ','|';', MidQuotes = '"'|'\''|'.', plus CR and LF), so no
 *  property table is needed. */
SZ_INTERNAL sz_u64_t sz_utf8_word_break_class_bitmask_v128_(v128_t equal_vec) {
    return (sz_u64_t)((sz_u32_t)wasm_i8x16_bitmask(equal_vec) & 0xFFFFu);
}

/*  Boundary mask for the trusted lanes [2,14] of an all-ASCII 16-byte window: bit i set => a word boundary
 *  precedes lane i. Computed branchlessly via the shared portable join routine. */
SZ_INTERNAL sz_u32_t sz_utf8_word_break_boundary_mask_v128_(v128_t window) {
    v128_t lowered = wasm_v128_or(window, wasm_i8x16_splat(0x20)); // fold A-Z onto a-z
    sz_u64_t aletter_mask = sz_utf8_word_break_class_bitmask_v128_(
        wasm_v128_and(wasm_u8x16_ge(lowered, wasm_i8x16_splat(0x61)), wasm_u8x16_le(lowered, wasm_i8x16_splat(0x7A))));
    sz_u64_t numeric_mask = sz_utf8_word_break_class_bitmask_v128_(
        wasm_v128_and(wasm_u8x16_ge(window, wasm_i8x16_splat(0x30)), wasm_u8x16_le(window, wasm_i8x16_splat(0x39))));
    sz_u64_t extendnumlet_mask = sz_utf8_word_break_class_bitmask_v128_(wasm_i8x16_eq(window, wasm_i8x16_splat(0x5F)));
    sz_u64_t midletter_mask = sz_utf8_word_break_class_bitmask_v128_(wasm_i8x16_eq(window, wasm_i8x16_splat(0x3A)));
    sz_u64_t midnum_mask = sz_utf8_word_break_class_bitmask_v128_(
        wasm_v128_or(wasm_i8x16_eq(window, wasm_i8x16_splat(0x2C)), wasm_i8x16_eq(window, wasm_i8x16_splat(0x3B))));
    sz_u64_t mid_quotes_mask = sz_utf8_word_break_class_bitmask_v128_(wasm_v128_or(
        wasm_v128_or(wasm_i8x16_eq(window, wasm_i8x16_splat(0x22)), wasm_i8x16_eq(window, wasm_i8x16_splat(0x27))),
        wasm_i8x16_eq(window, wasm_i8x16_splat(0x2E))));
    sz_u64_t carriage_return_mask = sz_utf8_word_break_class_bitmask_v128_(
        wasm_i8x16_eq(window, wasm_i8x16_splat(0x0D)));
    sz_u64_t line_feed_mask = sz_utf8_word_break_class_bitmask_v128_(wasm_i8x16_eq(window, wasm_i8x16_splat(0x0A)));
    sz_u64_t join = sz_utf8_word_break_join_from_class_masks_(aletter_mask, numeric_mask, extendnumlet_mask,
                                                              midletter_mask, midnum_mask, mid_quotes_mask,
                                                              carriage_return_mask, line_feed_mask);
    return (sz_u32_t)((~join) & 0x7FFCu); // trusted lanes [2,14]
}

#pragma region Word boundary left-pack

/**
 *  @brief  Ascending `wasm_i8x16_swizzle` permutation that gathers a 4-bit sub-block's set u32 lanes to the front.
 *
 *  WASM SIMD128 has no `vpcompressb` and no masked store, so each 4-lane sub-block of the boundary mask is
 *  compacted with one byte-table lookup: row `[m]` holds the 16 byte indices that gather the `m`-selected 32-bit
 *  `sz_size_t` lanes (of four, each a 4-byte group) to the front of a register, preserving ascending lane order.
 *  The v128 analog of `sz_utf8_word_compact4_permutation_haswell_`.
 */
SZ_INTERNAL v128_t sz_utf8_word_compact4_permutation_v128_(sz_u32_t submask) {
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
    return wasm_v128_load(compact_lut[submask & 0xFu]);
}

/**
 *  @brief  Descending counterpart of `sz_utf8_word_compact4_permutation_v128_`: gathers a 4-bit sub-block's set
 *          u32 lanes to the front in HIGH-to-LOW lane order, for the reverse word scan.
 */
SZ_INTERNAL v128_t sz_utf8_word_compact4_permutation_descending_v128_(sz_u32_t submask) {
    static sz_u8_t const compact_lut[16][16] = {
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},       {0, 1, 2, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {4, 5, 6, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},       {4, 5, 6, 7, 0, 1, 2, 3, 0, 0, 0, 0, 0, 0, 0, 0},
        {8, 9, 10, 11, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},     {8, 9, 10, 11, 0, 1, 2, 3, 0, 0, 0, 0, 0, 0, 0, 0},
        {8, 9, 10, 11, 4, 5, 6, 7, 0, 0, 0, 0, 0, 0, 0, 0},     {8, 9, 10, 11, 4, 5, 6, 7, 0, 1, 2, 3, 0, 0, 0, 0},
        {12, 13, 14, 15, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},   {12, 13, 14, 15, 0, 1, 2, 3, 0, 0, 0, 0, 0, 0, 0, 0},
        {12, 13, 14, 15, 4, 5, 6, 7, 0, 0, 0, 0, 0, 0, 0, 0},   {12, 13, 14, 15, 4, 5, 6, 7, 0, 1, 2, 3, 0, 0, 0, 0},
        {12, 13, 14, 15, 8, 9, 10, 11, 0, 0, 0, 0, 0, 0, 0, 0}, {12, 13, 14, 15, 8, 9, 10, 11, 0, 1, 2, 3, 0, 0, 0, 0},
        {12, 13, 14, 15, 8, 9, 10, 11, 4, 5, 6, 7, 0, 0, 0, 0}, {12, 13, 14, 15, 8, 9, 10, 11, 4, 5, 6, 7, 0, 1, 2, 3},
    };
    return wasm_v128_load(compact_lut[submask & 0xFu]);
}

/**
 *  @brief  Shift four packed u32 boundary lanes right by one and seat @p carry in the freed lane 0.
 *
 *  Builds `[carry, b0, b1, b2]` from `boundaries = [b0, b1, b2, b3]` with one `wasm_i8x16_shuffle` - lane 0 drawn
 *  from @p carry, lanes 1..3 from the low three lanes of @p boundaries - the v128 analog of Haswell's
 *  `lane_shift_right` permute followed by the `word_start` / `word_end` blend.
 */
SZ_INTERNAL v128_t sz_utf8_word_shift_right_carry_v128_(v128_t boundaries, v128_t carry) {
    return wasm_i8x16_shuffle(carry, boundaries, 0, 1, 2, 3, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27);
}

#pragma endregion // Word boundary left-pack

SZ_PUBLIC sz_size_t sz_utf8_word_find_boundaries_v128( //
    sz_cptr_t text, sz_size_t length,                  //
    sz_size_t *word_starts, sz_size_t *word_lengths,   //
    sz_size_t words_capacity, sz_size_t *bytes_consumed) {

    sz_size_t words = 0;
    if (length == 0 || words_capacity == 0) {
        if (bytes_consumed) *bytes_consumed = 0;
        return 0;
    }
    sz_u8_t const *text_u8 = (sz_u8_t const *)text;

    sz_size_t word_start = 0; // Start of the word currently being accumulated (always a boundary).
    // Skip the first codepoint (position 0 is always a boundary, WB1).
    sz_size_t position = sz_utf8_codepoint_length_(text_u8[0]);

    // Oracle-free fast path: an all-ASCII window [position-2, position+14) resolves boundaries at positions
    // [position, position+12]; one fixed sub-block loop compacts each group and emits it as a shifted-difference,
    // carrying the open `word_start` into lane 0 and the previous boundary into lanes 1..3.
    while (position < length) {
        int ascii_window = position >= 2 && position + 14 <= length;
        v128_t window = wasm_i8x16_splat(0);
        if (ascii_window) {
            window = wasm_v128_load(text_u8 + position - 2); // lane j = byte position-2+j
            ascii_window = ((sz_u32_t)wasm_i8x16_bitmask(window) & 0xFFFFu) == 0;
        }
        if (!ascii_window) { // Non-ASCII window or near the edges: one scalar codepoint step.
            if (sz_utf8_is_word_boundary_serial(text, length, position)) {
                if (words == words_capacity) {
                    if (bytes_consumed) *bytes_consumed = word_start;
                    return words;
                }
                word_starts[words] = word_start, word_lengths[words] = position - word_start, ++words;
                word_start = position;
            }
            position += sz_utf8_codepoint_length_(text_u8[position]);
            continue;
        }

        sz_u32_t boundary = sz_utf8_word_break_boundary_mask_v128_(window); // trusted lanes [2,14]
        for (sz_size_t sub_block = 0; sub_block < 4; ++sub_block) {
            sz_u32_t const submask = (boundary >> (sub_block * 4)) & 0xFu;
            if (!submask) continue;
            sz_size_t const taken = (sz_size_t)sz_u32_popcount(submask);
            sz_size_t const stored = sz_min_of_two(taken, words_capacity - words);

            v128_t const positions = wasm_u32x4_make(
                (sz_u32_t)(position - 2 + sub_block * 4 + 0), (sz_u32_t)(position - 2 + sub_block * 4 + 1),
                (sz_u32_t)(position - 2 + sub_block * 4 + 2), (sz_u32_t)(position - 2 + sub_block * 4 + 3));
            v128_t const boundaries = wasm_i8x16_swizzle(positions, sz_utf8_word_compact4_permutation_v128_(submask));
            v128_t const starts = sz_utf8_word_shift_right_carry_v128_(boundaries,
                                                                       wasm_u32x4_splat((sz_u32_t)word_start));
            v128_t const lengths = wasm_i32x4_sub(boundaries, starts);
            sz_size_t scratch_starts[4], scratch_lengths[4];
            wasm_v128_store(scratch_starts, starts);
            wasm_v128_store(scratch_lengths, lengths);
            for (sz_size_t i = 0; i < stored; ++i)
                word_starts[words + i] = scratch_starts[i], word_lengths[words + i] = scratch_lengths[i];
            words += stored;
            if (stored) word_start = word_starts[words - 1] + word_lengths[words - 1];
            if (words == words_capacity) {
                if (bytes_consumed) *bytes_consumed = word_start;
                return words;
            }
        }
        position += 13; // Resolved [position, position+12]; next unresolved boundary is at position+13.
    }

    if (words == words_capacity) {
        if (bytes_consumed) *bytes_consumed = word_start;
        return words;
    }
    word_starts[words] = word_start;
    word_lengths[words] = length - word_start;
    ++words;
    if (bytes_consumed) *bytes_consumed = length;
    return words;
}

SZ_PUBLIC sz_size_t sz_utf8_word_rfind_boundaries_v128( //
    sz_cptr_t text, sz_size_t length,                   //
    sz_size_t *word_starts, sz_size_t *word_lengths,    //
    sz_size_t words_capacity, sz_size_t *bytes_consumed) {

    sz_size_t words = 0;
    if (length == 0 || words_capacity == 0) {
        if (bytes_consumed) *bytes_consumed = length;
        return 0;
    }
    sz_u8_t const *text_u8 = (sz_u8_t const *)text;

    sz_size_t word_end = length; // End of the word currently being accumulated (always a boundary).
    // Move back one codepoint from the end (position length is always a boundary, WB2).
    sz_size_t position = length - 1;
    while (position > 0 && (text_u8[position] & 0xC0) == 0x80) position--;

    // Oracle-free fast path: an all-ASCII window [position-14, position+2) resolves boundaries at positions
    // [position-12, position]; one fixed sub-block loop walks high-to-low, compacting each group in descending
    // lane order and emitting it as a shifted-difference (lane 0 carries the open `word_end`).
    while (position > 0) {
        sz_size_t base = position - 14; // lane j = byte base+j; trusted lanes [2,14] → [position-12, position]
        int ascii_window = position >= 14 && position + 2 <= length;
        v128_t window = wasm_i8x16_splat(0);
        if (ascii_window) {
            window = wasm_v128_load(text_u8 + base);
            ascii_window = ((sz_u32_t)wasm_i8x16_bitmask(window) & 0xFFFFu) == 0;
        }
        if (!ascii_window) { // Non-ASCII window or near the edges: one scalar codepoint step.
            if (sz_utf8_is_word_boundary_serial(text, length, position)) {
                if (words == words_capacity) {
                    if (bytes_consumed) *bytes_consumed = word_end;
                    return words;
                }
                word_starts[words] = position, word_lengths[words] = word_end - position, ++words;
                word_end = position;
            }
            position--;
            while (position > 0 && (text_u8[position] & 0xC0) == 0x80) position--;
            continue;
        }

        sz_u32_t boundary = sz_utf8_word_break_boundary_mask_v128_(window); // trusted lanes [2,14]
        for (sz_size_t sub_block = 4; sub_block-- > 0;) {                   // high-to-low for descending emission
            sz_u32_t const submask = (boundary >> (sub_block * 4)) & 0xFu;
            if (!submask) continue;
            sz_size_t const taken = (sz_size_t)sz_u32_popcount(submask);
            sz_size_t const stored = sz_min_of_two(taken, words_capacity - words);

            v128_t const positions = wasm_u32x4_make(
                (sz_u32_t)(base + sub_block * 4 + 0), (sz_u32_t)(base + sub_block * 4 + 1),
                (sz_u32_t)(base + sub_block * 4 + 2), (sz_u32_t)(base + sub_block * 4 + 3));
            v128_t const boundaries = wasm_i8x16_swizzle(positions,
                                                         sz_utf8_word_compact4_permutation_descending_v128_(submask));
            v128_t const previous = sz_utf8_word_shift_right_carry_v128_(boundaries,
                                                                         wasm_u32x4_splat((sz_u32_t)word_end));
            v128_t const lengths = wasm_i32x4_sub(previous, boundaries);
            sz_size_t scratch_starts[4], scratch_lengths[4];
            wasm_v128_store(scratch_starts, boundaries);
            wasm_v128_store(scratch_lengths, lengths);
            for (sz_size_t i = 0; i < stored; ++i)
                word_starts[words + i] = scratch_starts[i], word_lengths[words + i] = scratch_lengths[i];
            words += stored;
            if (stored) word_end = word_starts[words - 1];
            if (words == words_capacity) {
                if (bytes_consumed) *bytes_consumed = word_end;
                return words;
            }
        }
        position = base + 1; // Resolved down to position-12; next unresolved boundary is at position-13.
    }

    if (words == words_capacity) {
        if (bytes_consumed) *bytes_consumed = word_end;
        return words;
    }
    word_starts[words] = 0;
    word_lengths[words] = word_end;
    ++words;
    if (bytes_consumed) *bytes_consumed = 0;
    return words;
}

#if defined(__clang__)
#pragma clang attribute pop
#endif
#endif // SZ_USE_V128

#ifdef __cplusplus
}
#endif

#endif // STRINGZILLA_UTF8_ITERATE_V128_H_
