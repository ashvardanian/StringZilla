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

SZ_INTERNAL v128_t sz_utf8_rotate1_wasm_(v128_t v) {
    return wasm_i8x16_shuffle(v, v, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0);
}
SZ_INTERNAL v128_t sz_utf8_rotate2_wasm_(v128_t v) {
    return wasm_i8x16_shuffle(v, v, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0, 1);
}

SZ_PUBLIC sz_cptr_t sz_utf8_find_newline_v128(sz_cptr_t text, sz_size_t length, sz_size_t *matched_length) {

    sz_u128_vec_t text_vec;
    v128_t newline_u8x16 = wasm_i8x16_splat('\n');
    v128_t f_vec = wasm_i8x16_splat('\f');
    v128_t r_vec = wasm_i8x16_splat('\r');
    v128_t x_c2_vec = wasm_i8x16_splat((sz_i8_t)0xC2);
    v128_t x_85_vec = wasm_i8x16_splat((sz_i8_t)0x85);
    v128_t x_e2_vec = wasm_i8x16_splat((sz_i8_t)0xE2);
    v128_t x_80_vec = wasm_i8x16_splat((sz_i8_t)0x80);
    v128_t x_a8_vec = wasm_i8x16_splat((sz_i8_t)0xA8);
    v128_t x_a9_vec = wasm_i8x16_splat((sz_i8_t)0xA9);

    // Zero out the last lane (drop1) and the last two lanes (drop2).
    v128_t drop1_vec = wasm_i8x16_make(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0);
    v128_t drop2_vec = wasm_i8x16_make(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 0);

    while (length >= 16) {
        text_vec.v128 = wasm_v128_load(text);

        // 1-byte matches: the contiguous control range '\n'..'\f' as one range compare (instead of
        // three separate equalities), plus '\r' which is masked at the last lane for the `\r\n` rule.
        v128_t newline_range_cmp = wasm_v128_and(wasm_u8x16_ge(text_vec.v128, newline_u8x16),
                                                 wasm_u8x16_le(text_vec.v128, f_vec));
        v128_t r_cmp = wasm_v128_and(wasm_i8x16_eq(text_vec.v128, r_vec), drop1_vec); // mask out \r at pos 15
        v128_t one_byte_cmp = wasm_v128_or(newline_range_cmp, r_cmp);

        // 2- & 3-byte matches with shifted views
        v128_t text1 = sz_utf8_rotate1_wasm_(text_vec.v128);
        v128_t text2 = sz_utf8_rotate2_wasm_(text_vec.v128);
        v128_t rn_match_u8x16 = wasm_v128_and(r_cmp, wasm_i8x16_eq(text1, newline_u8x16));
        v128_t x_c285_vec = wasm_v128_and(wasm_i8x16_eq(text_vec.v128, x_c2_vec), wasm_i8x16_eq(text1, x_85_vec));
        v128_t two_byte_cmp = wasm_v128_and(wasm_v128_or(rn_match_u8x16, x_c285_vec), drop1_vec);

        v128_t x_e280_cmp = wasm_v128_and(wasm_i8x16_eq(text_vec.v128, x_e2_vec), wasm_i8x16_eq(text1, x_80_vec));
        v128_t x_e280ax_cmp = wasm_v128_and(
            x_e280_cmp, wasm_v128_or(wasm_i8x16_eq(text2, x_a8_vec), wasm_i8x16_eq(text2, x_a9_vec)));
        v128_t three_byte_cmp = wasm_v128_and(x_e280ax_cmp, drop2_vec);

        v128_t combined_vec = wasm_v128_or(one_byte_cmp, wasm_v128_or(two_byte_cmp, three_byte_cmp));
        if (wasm_v128_any_true(combined_vec)) {
            sz_u32_t one_byte_mask = (sz_u32_t)wasm_i8x16_bitmask(one_byte_cmp);
            sz_u32_t two_mask = (sz_u32_t)wasm_i8x16_bitmask(two_byte_cmp);
            sz_u32_t three_mask = (sz_u32_t)wasm_i8x16_bitmask(three_byte_cmp);
            sz_u32_t combined_mask = one_byte_mask | two_mask | three_mask;

            int bit_index = sz_u32_ctz(combined_mask);
            sz_u32_t first_match_mask = (sz_u32_t)1 << bit_index;
            sz_size_t length_value = 1;
            length_value += (first_match_mask & (two_mask | three_mask)) != 0;
            length_value += (first_match_mask & three_mask) != 0;
            *matched_length = length_value;
            return text + bit_index;
        }
        text += 14;
        length -= 14;
    }

    return sz_utf8_find_newline_serial(text, length, matched_length);
}

SZ_PUBLIC sz_cptr_t sz_utf8_find_whitespace_v128(sz_cptr_t text, sz_size_t length, sz_size_t *matched_length) {

    sz_u128_vec_t text_vec;
    v128_t t_vec = wasm_i8x16_splat('\t');
    v128_t r_vec = wasm_i8x16_splat('\r');
    v128_t x_20_vec = wasm_i8x16_splat(' ');
    v128_t x_c2_vec = wasm_i8x16_splat((sz_i8_t)0xC2);
    v128_t x_85_vec = wasm_i8x16_splat((sz_i8_t)0x85);
    v128_t x_a0_vec = wasm_i8x16_splat((sz_i8_t)0xA0);
    v128_t x_e1_vec = wasm_i8x16_splat((sz_i8_t)0xE1);
    v128_t x_e2_vec = wasm_i8x16_splat((sz_i8_t)0xE2);
    v128_t x_e3_vec = wasm_i8x16_splat((sz_i8_t)0xE3);
    v128_t x_9a_vec = wasm_i8x16_splat((sz_i8_t)0x9A);
    v128_t x_80_vec = wasm_i8x16_splat((sz_i8_t)0x80);
    v128_t x_81_vec = wasm_i8x16_splat((sz_i8_t)0x81);
    v128_t x_8d_vec = wasm_i8x16_splat((sz_i8_t)0x8D);
    v128_t x_a8_vec = wasm_i8x16_splat((sz_i8_t)0xA8);
    v128_t x_a9_vec = wasm_i8x16_splat((sz_i8_t)0xA9);
    v128_t x_af_vec = wasm_i8x16_splat((sz_i8_t)0xAF);
    v128_t x_9f_vec = wasm_i8x16_splat((sz_i8_t)0x9F);

    v128_t drop1_vec = wasm_i8x16_make(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0);
    v128_t drop2_vec = wasm_i8x16_make(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 0);

    while (length >= 16) {
        text_vec.v128 = wasm_v128_load(text);

        // 1-byte matches: ' ' or the contiguous range '\t'..'\r'.
        v128_t x_20_cmp = wasm_i8x16_eq(text_vec.v128, x_20_vec);
        v128_t range_cmp = wasm_v128_and(wasm_u8x16_ge(text_vec.v128, t_vec), wasm_u8x16_le(text_vec.v128, r_vec));
        v128_t one_byte_cmp = wasm_v128_or(x_20_cmp, range_cmp);

        // 2-byte and 3-byte prefix indicators.
        v128_t x_c2_cmp = wasm_v128_and(wasm_i8x16_eq(text_vec.v128, x_c2_vec), drop1_vec);
        v128_t x_e1_cmp = wasm_v128_and(wasm_i8x16_eq(text_vec.v128, x_e1_vec), drop2_vec);
        v128_t x_e2_cmp = wasm_v128_and(wasm_i8x16_eq(text_vec.v128, x_e2_vec), drop2_vec);
        v128_t x_e3_cmp = wasm_v128_and(wasm_i8x16_eq(text_vec.v128, x_e3_vec), drop2_vec);
        v128_t prefix_byte_cmp = wasm_v128_or(
            one_byte_cmp, wasm_v128_or(wasm_v128_or(x_c2_cmp, x_e1_cmp), wasm_v128_or(x_e2_cmp, x_e3_cmp)));

        sz_u32_t one_byte_mask = (sz_u32_t)wasm_i8x16_bitmask(one_byte_cmp);
        sz_u32_t prefix_byte_mask = (sz_u32_t)wasm_i8x16_bitmask(prefix_byte_cmp);

        if (!prefix_byte_mask) {
            text += 14;
            length -= 14;
            continue;
        }
        else if (one_byte_mask) {
            int first_one_byte_offset = sz_u32_ctz(one_byte_mask);
            int first_prefix_offset = sz_u32_ctz(prefix_byte_mask);
            if (first_one_byte_offset < first_prefix_offset) {
                *matched_length = 1;
                return text + first_one_byte_offset;
            }
        }

        // 2-byte matches.
        v128_t text1 = sz_utf8_rotate1_wasm_(text_vec.v128);
        v128_t two_vec = wasm_v128_or(wasm_v128_and(x_c2_cmp, wasm_i8x16_eq(text1, x_85_vec)),
                                      wasm_v128_and(x_c2_cmp, wasm_i8x16_eq(text1, x_a0_vec)));

        // 3-byte matches.
        v128_t text2 = sz_utf8_rotate2_wasm_(text_vec.v128);
        v128_t x_80_ge_cmp = wasm_u8x16_ge(text2, x_80_vec);
        v128_t x_8d_le_cmp = wasm_u8x16_le(text2, x_8d_vec);

        v128_t ogham_cmp = wasm_v128_and(x_e1_cmp,
                                         wasm_v128_and(wasm_i8x16_eq(text1, x_9a_vec), wasm_i8x16_eq(text2, x_80_vec)));
        v128_t range_e280_cmp = wasm_v128_and(
            x_e2_cmp, wasm_v128_and(wasm_i8x16_eq(text1, x_80_vec), wasm_v128_and(x_80_ge_cmp, x_8d_le_cmp)));
        v128_t line_cmp = wasm_v128_and(x_e2_cmp,
                                        wasm_v128_and(wasm_i8x16_eq(text1, x_80_vec), wasm_i8x16_eq(text2, x_a8_vec)));
        v128_t paragraph_cmp = wasm_v128_and(
            x_e2_cmp, wasm_v128_and(wasm_i8x16_eq(text1, x_80_vec), wasm_i8x16_eq(text2, x_a9_vec)));
        v128_t nnbsp_cmp = wasm_v128_and(x_e2_cmp,
                                         wasm_v128_and(wasm_i8x16_eq(text1, x_80_vec), wasm_i8x16_eq(text2, x_af_vec)));
        v128_t mmsp_cmp = wasm_v128_and(x_e2_cmp,
                                        wasm_v128_and(wasm_i8x16_eq(text1, x_81_vec), wasm_i8x16_eq(text2, x_9f_vec)));
        v128_t ideographic_vec = wasm_v128_and(
            x_e3_cmp, wasm_v128_and(wasm_i8x16_eq(text1, x_80_vec), wasm_i8x16_eq(text2, x_80_vec)));
        v128_t three_vec = wasm_v128_and(
            wasm_v128_or(wasm_v128_or(wasm_v128_or(ogham_cmp, range_e280_cmp), wasm_v128_or(line_cmp, paragraph_cmp)),
                         wasm_v128_or(wasm_v128_or(nnbsp_cmp, mmsp_cmp), ideographic_vec)),
            drop2_vec);

        sz_u32_t two_byte_mask = (sz_u32_t)wasm_i8x16_bitmask(two_vec);
        sz_u32_t three_byte_mask = (sz_u32_t)wasm_i8x16_bitmask(three_vec);
        sz_u32_t combined_mask = one_byte_mask | two_byte_mask | three_byte_mask;

        if (combined_mask) {
            int bit_index = sz_u32_ctz(combined_mask);
            sz_u32_t first_match_mask = (sz_u32_t)1 << bit_index;
            sz_size_t length_value = 1;
            length_value += (first_match_mask & (two_byte_mask | three_byte_mask)) != 0;
            length_value += (first_match_mask & three_byte_mask) != 0;
            *matched_length = length_value;
            return text + bit_index;
        }
        text += 14;
        length -= 14;
    }

    return sz_utf8_find_whitespace_serial(text, length, matched_length);
}

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
        // Widen 16x u8 -> 16x u32 and store as runes (zero-extension == the ASCII code point value).
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
 *  '_' = ExtendNumLet}, which `sz_wb_interior_mask_v128_` detects directly. For a candidate at byte
 *  offset `i`, if both `byte[i-1]` and `byte[i]` are word-interior, then *every* UAX-29 rule that could
 *  fire (WB5/8/9/10/13a/13b and their context-needing siblings) resolves to "do NOT break" — verified
 *  exhaustively against the serial decision table — because ASCII contains no Extend/Format/ZWJ/RI/
 *  Hebrew/Katakana, so WB4 ignorable-skipping and RI parity never apply between two such bytes. Those
 *  positions are skipped with a single `wasm_i8x16_bitmask`. Any byte that is non-ASCII, or whose pair
 *  is not a guaranteed interior, is handed to `sz_utf8_is_word_boundary_serial` verbatim, so the result
 *  is bit-for-bit identical to the serial reference (the hard, stateful sub-rules stay serial). */

/** @brief Per-byte mask (0xFF) where a byte is ASCII and word-interior {ALetter,Numeric,ExtendNumLet}. */
SZ_INTERNAL v128_t sz_wb_interior_mask_v128_(v128_t bytes) {
    // The interior set is exactly {A-Z, a-z, 0-9, '_'}, so we test it directly instead of running the
    // full Word_Break classifier and then matching three property values. ASCII letters fold to a-z
    // by OR-ing 0x20; the `is_ascii` guard rejects any high byte that the fold might alias into range.
    v128_t is_ascii = wasm_i8x16_ge(bytes, wasm_i8x16_splat(0)); // high bit clear (signed >= 0)
    v128_t lowered_vec = wasm_v128_or(bytes, wasm_i8x16_splat(0x20));
    v128_t is_alpha = wasm_v128_and(wasm_i8x16_ge(lowered_vec, wasm_i8x16_splat(0x61)),
                                    wasm_i8x16_le(lowered_vec, wasm_i8x16_splat(0x7A)));
    v128_t is_digit = wasm_v128_and(wasm_i8x16_ge(bytes, wasm_i8x16_splat(0x30)),
                                    wasm_i8x16_le(bytes, wasm_i8x16_splat(0x39)));
    v128_t is_underscore = wasm_i8x16_eq(bytes, wasm_i8x16_splat(0x5F));
    v128_t interior = wasm_v128_or(wasm_v128_or(is_alpha, is_digit), is_underscore);
    return wasm_v128_and(is_ascii, interior);
}

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
    sz_size_t position = sz_utf8_char_length_(text_u8[0]);

    while (position < length) {
        // Vectorized fast-skip over ASCII word/number interiors. A candidate at offset `i` is a
        // guaranteed non-boundary iff byte[i-1] and byte[i] are both ASCII word-interior. We load a
        // 16-byte window anchored one byte BEFORE `position`, so lane 0 = byte[position-1] and lane 1 = byte[position].
        if (position >= 1 && position + 16 <= length) {
            v128_t window = wasm_v128_load(text_u8 + position - 1);
            v128_t interior = sz_wb_interior_mask_v128_(window);
            // Pair[i] interior iff lane i and lane i-1 both interior. Shift interior right by one lane
            // (lane i gets lane i-1's value); zero the wrapped lane 0 so it never claims a pair.
            v128_t previous = wasm_i8x16_shuffle(wasm_i8x16_splat(0), interior, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
                                                 25, 26, 27, 28, 29, 30);
            v128_t pair = wasm_v128_and(interior, previous);
            // Bit i (i>=1) set => offset (position-1+i) is a guaranteed non-boundary. Candidates correspond
            // to lanes 1..15 (offsets position..position+14). A clear bit among those is the first real candidate.
            sz_u32_t pair_mask = (sz_u32_t)wasm_i8x16_bitmask(pair);
            // Consider only lanes 1..15; a non-boundary run is a prefix of set bits there.
            sz_u32_t candidate_bits = (~pair_mask) & 0xFFFEu; // clear lane 0, look at lanes 1..15
            if (candidate_bits == 0) {
                // All of offsets position..position+14 are guaranteed non-boundaries; advance past them.
                position += 15;
                continue;
            }
            int first = sz_u32_ctz(candidate_bits); // lane index 1..15
            position += (sz_size_t)(first - 1);     // move to that codepoint-aligned candidate offset
            // Fall through to the serial check at this `position`.
        }

        if (sz_utf8_is_word_boundary_serial(text, length, position)) {
            if (words == words_capacity) {
                if (bytes_consumed) *bytes_consumed = word_start;
                return words;
            }
            word_starts[words] = word_start;
            word_lengths[words] = position - word_start;
            ++words;
            word_start = position;
        }
        position += sz_utf8_char_length_(text_u8[position]);
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

    while (position > 0) {
        // Vectorized fast-skip backward over ASCII word/number interiors. Anchor a 16-byte window so
        // lane 15 = byte[position] and lane 14 = byte[position-1]: load at (position-15). Then pair[i] (i in 1..15)
        // = interior[i] && interior[i-1] marks offset (base+i) as a guaranteed non-boundary.
        if (position >= 15 && position + 1 <= length) {
            sz_size_t base = position - 15;
            v128_t window = wasm_v128_load(text_u8 + base);
            v128_t interior = sz_wb_interior_mask_v128_(window);
            v128_t previous = wasm_i8x16_shuffle(wasm_i8x16_splat(0), interior, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
                                                 25, 26, 27, 28, 29, 30);
            v128_t pair = wasm_v128_and(interior, previous);
            sz_u32_t pair_mask = (sz_u32_t)wasm_i8x16_bitmask(pair);
            // Offsets base+1 .. base+15 map to lanes 1..15. We scan downward from lane 15 (offset position).
            // A guaranteed non-boundary run is a suffix of set bits in lanes 1..15.
            sz_u32_t candidate_bits = (~pair_mask) & 0xFFFEu;
            if (candidate_bits == 0) {
                // Offsets base+1..base+15 (== position-14..position) are all guaranteed non-boundaries; jump below.
                position = base; // base == position-15 > 0 here; loop re-checks position>0
                continue;
            }
            int last = 31 - sz_u32_clz(candidate_bits); // highest candidate lane 1..15
            position = base + (sz_size_t)last;          // first real candidate at/below current position
            // Fall through to serial check at this `position`.
        }

        if (sz_utf8_is_word_boundary_serial(text, length, position)) {
            if (words == words_capacity) {
                if (bytes_consumed) *bytes_consumed = word_end;
                return words;
            }
            word_starts[words] = position;
            word_lengths[words] = word_end - position;
            ++words;
            word_end = position;
        }
        position--;
        while (position > 0 && (text_u8[position] & 0xC0) == 0x80) position--;
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
