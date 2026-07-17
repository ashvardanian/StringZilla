/**
 *  @file   include/stringzilla/utf8_wordbreaks/v128.h
 *  @author Ash Vardanian
 *  @brief  Fully-vectorized UAX-29 Word_Break segmentation for WebAssembly SIMD128 (v128). The v128 twin of the AVX2
 *          (Haswell), Ice Lake, NEON, and RVV kernels: no path scalar-walks codepoints or spills a vector to the stack
 *          to call the serial oracle.
 *
 *  Each 64-byte window lives as four `v128_t` quarters (the codepoint substrate twin of NEON's four `uint8x16_t`
 *  quarters); every per-codepoint BMP Word_Break property resolves through the shared page-compressed flat table via
 *  @ref sz_utf8_rune_flat_lookup_v128_ (the page LUT and leaf read by bounded scalar L1 walks, since
 *  `wasm_i8x16_swizzle` reaches only 16 B), and the Supplementary Plane through a `wasm_i8x16_swizzle` nibble cascade,
 *  bit-identical to `sz_rune_word_break_property` over the whole code space.
 *
 *  The classified window is lowered to the portable @ref sz_utf8_word_break_frame_t and handed to the SHARED
 *  `sz_utf8_word_break_decide_window_` rule engine, so WB1-WB16 (including the cross-window bridge shadow / RI parity /
 *  left-context carry / WB3c next1/2/3 neighbour coupling) run once in portable `sz_u64_t` bit algebra. The WB3c
 *  (Extended_Pictographic) SMP neighbour precompute to the frame's `pictographic` mask is gated by `want_pictographic`,
 *  exactly as in NEON. The boundary-mask drain left-packs its set lanes through the substrate's shuffle-LUT compaction
 *  (v128 has no `vpcompressb` / BMI2 `pext`, and the pre-commit hook bans the `ctz` / `popcount` builtins).
 */
#ifndef STRINGZILLA_UTF8_WORDBREAKS_V128_H_
#define STRINGZILLA_UTF8_WORDBREAKS_V128_H_

#include "stringzilla/types.h"
#include "stringzilla/utf8_wordbreaks/tables.h"
#include "stringzilla/utf8_wordbreaks/serial.h"
#include "stringzilla/utf8_runes/v128.h"

#ifdef __cplusplus
extern "C" {
#endif

#if SZ_USE_V128
#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("simd128"))), apply_to = function)
#endif

#pragma region UAX 29 Word Boundaries forward kernel

#pragma region In register vectorized classifier

/** @brief  Build a per-quarter byte-boolean selector (0x00/0xFF) from the 16 lane bits of @p bits at offset @p shift,
 *          the v128 twin of @ref sz_utf8_word_break_byte_mask_from_bits_neon_ confined to one quarter. */
SZ_HELPER_INLINE v128_t sz_utf8_word_break_byte_mask_from_bits_v128_(sz_u64_t bits, int shift) {
    static sz_u8_t const bit_position_lanes[16] = {1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128};
    static sz_u8_t const lane_half[16] = {0, 0, 0, 0, 0, 0, 0, 0, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF};
    sz_u8_t const low_byte = (sz_u8_t)((bits >> shift) & 0xFF);
    sz_u8_t const high_byte = (sz_u8_t)((bits >> (shift + 8)) & 0xFF);
    v128_t const per_half_u8x16 = wasm_v128_bitselect(wasm_i8x16_splat((sz_i8_t)high_byte),
                                                      wasm_i8x16_splat((sz_i8_t)low_byte), wasm_v128_load(lane_half));
    v128_t const bit_position_u8x16 = wasm_v128_load(bit_position_lanes);
    return wasm_i8x16_eq(wasm_v128_and(per_half_u8x16, bit_position_u8x16), bit_position_u8x16);
}

/** @brief  Word_Break class byte for sixteen ASCII codepoints (cp < 0x80) via the existing 128-entry property table,
 *          read by a bounded scalar L1 walk over `raw & 0x7F` (v128's `wasm_i8x16_swizzle` reaches only 16 B, so
 *          NEON's in-register `vqtbl4q_u8` read does not port). The window byte equals the codepoint on ASCII lanes.
 *          Addresses ONE quarter. */
SZ_HELPER_AUTO v128_t sz_utf8_word_break_ascii_class_v128_(v128_t bytes_u8x16) {
    sz_align_(16) sz_u8_t index_lanes[16], class_lanes[16];
    wasm_v128_store(index_lanes, wasm_v128_and(bytes_u8x16, wasm_i8x16_splat(0x7F)));
    for (int lane = 0; lane < 16; ++lane) class_lanes[lane] = sz_utf8_word_break_property_ascii_[index_lanes[lane]];
    return wasm_v128_load(class_lanes);
}

/** @brief  Word_Break class byte for sixteen BMP codepoints (per-lane high = cp>>8, low = cp&0xFF) from the flat
 *          page-compressed table via @ref sz_utf8_rune_flat_lookup_v128_, the v128 twin of
 *          @ref sz_utf8_word_break_bmp_class_neon_. Bit-exact with `sz_rune_word_break_property` over the whole BMP.
 *          Addresses ONE quarter. */
SZ_HELPER_AUTO v128_t sz_utf8_word_break_bmp_class_v128_(v128_t high_bytes_u8x16, v128_t low_bytes_u8x16) {
    return sz_utf8_rune_flat_lookup_v128_(sz_utf8_word_break_bmp_page_lut_, sz_utf8_word_break_flat_bmp_,
                                          (int)sz_utf8_word_break_flat_pages_k, high_bytes_u8x16, low_bytes_u8x16);
}

/** @brief  Word_Break class byte for sixteen ASTRAL codepoints over the 20-bit offset = cp - 0x10000 (5-nibble
 *          cascade), the v128 twin of @ref sz_utf8_word_break_astral_class_neon_. Per-lane bytes:
 *          @p plane_off_u8x16 = (offset>>16)&0xFF (low nibble meaningful), @p high_u8x16 = (offset>>8)&0xFF,
 *          @p low_u8x16 = offset&0xFF. Bit-exact with `sz_rune_word_break_property` over the Supplementary Planes.
 *          Addresses ONE quarter. */
SZ_HELPER_AUTO v128_t sz_utf8_word_break_astral_class_v128_(v128_t plane_off_u8x16, v128_t high_u8x16,
                                                            v128_t low_u8x16) {
    v128_t const low_nibble_mask_u8x16 = wasm_i8x16_splat(0x0F);
    v128_t const nibble4_u8x16 = wasm_v128_and(plane_off_u8x16, low_nibble_mask_u8x16);
    v128_t const nibble3_u8x16 = wasm_v128_and(wasm_u8x16_shr(high_u8x16, 4), low_nibble_mask_u8x16);
    v128_t const stage1_index_u8x16 = wasm_v128_or(wasm_i8x16_shl(nibble4_u8x16, 4), nibble3_u8x16);
    v128_t const page_u8x16 = sz_utf8_rune_lut256_v128_(sz_utf8_word_break_haswell_astral_stage1_, stage1_index_u8x16);
    v128_t const nibble2_u8x16 = wasm_v128_and(high_u8x16, low_nibble_mask_u8x16);
    v128_t const leaf2_u8x16 = sz_utf8_rune_cascade_stage_v128_(
        sz_utf8_word_break_haswell_astral_stage2_lo_, sz_utf8_word_break_haswell_astral_stage2_lo_count_k / 16,
        page_u8x16, nibble2_u8x16);
    v128_t const nibble1_u8x16 = wasm_v128_and(wasm_u8x16_shr(low_u8x16, 4), low_nibble_mask_u8x16);
    v128_t const leaf_lo_u8x16 = sz_utf8_rune_cascade_stage_v128_(
        sz_utf8_word_break_haswell_astral_stage3_lo_, sz_utf8_word_break_haswell_astral_stage3_lo_count_k / 16,
        leaf2_u8x16, nibble1_u8x16);
    v128_t const leaf_hi_u8x16 = sz_utf8_rune_cascade_stage_v128_(
        sz_utf8_word_break_haswell_astral_stage3_hi_, sz_utf8_word_break_haswell_astral_stage3_hi_count_k / 16,
        leaf2_u8x16, nibble1_u8x16);
    v128_t const nibble0_u8x16 = wasm_v128_and(low_u8x16, low_nibble_mask_u8x16);
    v128_t const leaf_group_u8x16 = wasm_v128_or(wasm_v128_and(wasm_u8x16_shr(leaf_lo_u8x16, 4), low_nibble_mask_u8x16),
                                                 wasm_i8x16_shl(leaf_hi_u8x16, 4));
    v128_t const leaf_low_nibble_u8x16 = wasm_v128_and(leaf_lo_u8x16, low_nibble_mask_u8x16);
    v128_t const stage4_lut_index_u8x16 = wasm_v128_or(wasm_i8x16_shl(leaf_low_nibble_u8x16, 4), nibble0_u8x16);
    v128_t result_u8x16 = wasm_i8x16_splat(0);
    for (int group = 0; group < (int)sz_utf8_word_break_haswell_astral_leaf_groups_k; ++group) {
        v128_t const value_u8x16 = sz_utf8_rune_lut256_v128_(
            sz_utf8_word_break_haswell_astral_stage4_groups_ + group * 256, stage4_lut_index_u8x16);
        v128_t const here_u8x16 = wasm_i8x16_eq(leaf_group_u8x16, wasm_i8x16_splat((sz_i8_t)group));
        result_u8x16 = wasm_v128_bitselect(value_u8x16, result_u8x16, here_u8x16);
    }
    return result_u8x16;
}

/** @brief  Per-window byte-lane classification (v128): the Word_Break class byte per lane as four `v128_t` quarters,
 *          valid only on codepoint-start lanes (the engine reads classes only at starts). The v128 twin of
 *          @ref sz_utf8_word_break_classify_window_neon_, bit-identical on every start lane. ASCII through the property
 *          table, BMP through the flat table, 4-byte leads through the astral cascade with the codepoint high/low/plane
 *          reconstructed from the forward neighbours. The BMP flat table is read over the WHOLE window and blended only
 *          at the 2-/3-byte start lanes: the NEON `bmp_compact` start-gather needs `ctz` (banned here), and the flat
 *          lookup is index-safe on any byte, so the maskless full-window read matches the RVV backend and stays
 *          bit-exact at every start lane (every other lane is a don't-care left at zero, exactly as NEON leaves it). */
SZ_HELPER_AUTO void sz_utf8_word_break_classify_window_v128_( //
    sz_utf8_rune_window_v128_t window, v128_t *classes) {
    v128_t const *raw = window.window;
    sz_u64_t const ascii_starts = window.codepoint_starts & ~window.two_byte_starts & ~window.three_byte_starts &
                                  ~window.four_byte_starts;
    sz_u64_t const bmp_starts = window.two_byte_starts | window.three_byte_starts;

    v128_t next1[4], next2[4], next3[4];
    sz_utf8_forward_neighbours_v128_(raw, next1, next2, next3);

    v128_t const low_two_bits_u8x16 = wasm_i8x16_splat(0x03), low_three_bits_u8x16 = wasm_i8x16_splat(0x07),
                 low_nibble_u8x16 = wasm_i8x16_splat(0x0F), plane_high_bits_u8x16 = wasm_i8x16_splat(0x1C),
                 low_six_bits_u8x16 = wasm_i8x16_splat(0x3F), high_two_bits_u8x16 = wasm_i8x16_splat((sz_i8_t)0xC0),
                 high_nibble_u8x16 = wasm_i8x16_splat((sz_i8_t)0xF0);

    for (int quarter = 0; quarter < 4; ++quarter) {
        v128_t const raw_quarter_u8x16 = raw[quarter];
        int const lane_base = quarter * 16;
        v128_t class_bytes_u8x16 = wasm_i8x16_splat(0);

        // ASCII lanes: read the 128-entry property table directly off the raw byte.
        v128_t const ascii_select_u8x16 = sz_utf8_word_break_byte_mask_from_bits_v128_(ascii_starts, lane_base);
        class_bytes_u8x16 = wasm_v128_bitselect(sz_utf8_word_break_ascii_class_v128_(raw_quarter_u8x16),
                                                class_bytes_u8x16, ascii_select_u8x16);

        // BMP (2-/3-byte) start lanes: the flat page-compressed table over the reconstructed (high, low) halves.
        if (bmp_starts) {
            v128_t const bmp_select_u8x16 = sz_utf8_word_break_byte_mask_from_bits_v128_(bmp_starts, lane_base);
            class_bytes_u8x16 = wasm_v128_bitselect(
                sz_utf8_word_break_bmp_class_v128_(window.high[quarter], window.low[quarter]), class_bytes_u8x16,
                bmp_select_u8x16);
        }

        // 4-byte (astral) lanes: reconstruct the codepoint from the lead + three forward neighbours, then the astral
        // cascade addressed by offset = codepoint - 0x10000 (the offset's plane nibble is `plane - 1`).
        if (window.four_byte_starts) {
            v128_t const next1_u8x16 = next1[quarter], next2_u8x16 = next2[quarter], next3_u8x16 = next3[quarter];
            v128_t const four_select_u8x16 = sz_utf8_word_break_byte_mask_from_bits_v128_(window.four_byte_starts,
                                                                                          lane_base);
            v128_t const plane_u8x16 = wasm_v128_or(
                wasm_v128_and(wasm_i8x16_shl(wasm_v128_and(raw_quarter_u8x16, low_three_bits_u8x16), 2),
                              plane_high_bits_u8x16),
                sz_utf8_srl8_v128_(next1_u8x16, 4, 0x03));
            v128_t const high_four_u8x16 = wasm_v128_or(
                wasm_v128_and(wasm_i8x16_shl(wasm_v128_and(next1_u8x16, low_nibble_u8x16), 4), high_nibble_u8x16),
                sz_utf8_srl8_v128_(next2_u8x16, 2, 0x0F));
            v128_t const low_four_u8x16 = wasm_v128_or(
                wasm_v128_and(wasm_i8x16_shl(wasm_v128_and(next2_u8x16, low_two_bits_u8x16), 6), high_two_bits_u8x16),
                wasm_v128_and(next3_u8x16, low_six_bits_u8x16));
            v128_t const plane_off_u8x16 = wasm_i8x16_sub(plane_u8x16, wasm_i8x16_splat(1));
            class_bytes_u8x16 = wasm_v128_bitselect(
                sz_utf8_word_break_astral_class_v128_(plane_off_u8x16, high_four_u8x16, low_four_u8x16),
                class_bytes_u8x16, four_select_u8x16);
        }
        classes[quarter] = class_bytes_u8x16;
    }
}

#pragma endregion In register vectorized classifier

#pragma region Mask algebra extractor

/** @brief  A 64-bit "class byte == @p value" lane mask over the four class quarters (four `wasm_i8x16_eq` -> combine). */
SZ_HELPER_INLINE sz_u64_t sz_utf8_word_break_class_mask_v128_(v128_t const *classes, sz_u8_t value) {
    v128_t const value_broadcast_u8x16 = wasm_i8x16_splat((sz_i8_t)value);
    return sz_utf8_mask_combine_v128_(
        wasm_i8x16_eq(classes[0], value_broadcast_u8x16), wasm_i8x16_eq(classes[1], value_broadcast_u8x16),
        wasm_i8x16_eq(classes[2], value_broadcast_u8x16), wasm_i8x16_eq(classes[3], value_broadcast_u8x16));
}

/** @brief  A 64-bit "raw window byte == @p value" lane mask over the four window quarters. */
SZ_HELPER_INLINE sz_u64_t sz_utf8_word_break_byte_equal_v128_(v128_t const *quarters, sz_u8_t value) {
    v128_t const value_broadcast_u8x16 = wasm_i8x16_splat((sz_i8_t)value);
    return sz_utf8_mask_combine_v128_(
        wasm_i8x16_eq(quarters[0], value_broadcast_u8x16), wasm_i8x16_eq(quarters[1], value_broadcast_u8x16),
        wasm_i8x16_eq(quarters[2], value_broadcast_u8x16), wasm_i8x16_eq(quarters[3], value_broadcast_u8x16));
}

/** @brief  A 64-bit "raw window byte >= @p bound" (unsigned) lane mask over the four window quarters (`wasm_u8x16_ge`). */
SZ_HELPER_INLINE sz_u64_t sz_utf8_word_break_byte_ge_v128_(v128_t const *quarters, sz_u8_t bound) {
    v128_t const bound_broadcast_u8x16 = wasm_i8x16_splat((sz_i8_t)bound);
    return sz_utf8_mask_combine_v128_(
        wasm_u8x16_ge(quarters[0], bound_broadcast_u8x16), wasm_u8x16_ge(quarters[1], bound_broadcast_u8x16),
        wasm_u8x16_ge(quarters[2], bound_broadcast_u8x16), wasm_u8x16_ge(quarters[3], bound_broadcast_u8x16));
}

/** @brief  Per-quarter "(high,low) 16-bit value in `[lo, hi]`" membership for one range, the v128 unsigned 16-bit
 *          window-compare building block of @ref sz_utf8_word_break_range16_mask_v128_. `wasm_u8x16_ge` / `wasm_u8x16_lt`
 *          are native, so no `max_epu8` emulation is needed. */
SZ_HELPER_INLINE v128_t sz_utf8_word_break_range16_one_v128_(v128_t high_u8x16, v128_t low_u8x16, sz_u16_t lo,
                                                             sz_u16_t hi) {
    v128_t const lo_high_u8x16 = wasm_i8x16_splat((sz_i8_t)(lo >> 8)),
                 lo_low_u8x16 = wasm_i8x16_splat((sz_i8_t)(lo & 0xFF));
    v128_t const hi_high_u8x16 = wasm_i8x16_splat((sz_i8_t)(hi >> 8)),
                 hi_low_u8x16 = wasm_i8x16_splat((sz_i8_t)(hi & 0xFF));
    v128_t const high_eq_lo_u8x16 = wasm_i8x16_eq(high_u8x16, lo_high_u8x16);
    v128_t const high_eq_hi_u8x16 = wasm_i8x16_eq(high_u8x16, hi_high_u8x16);
    // not_below_u8x16: high_u8x16 > lo_high_u8x16, or (high_u8x16 == lo_high_u8x16 and low_u8x16 >= lo_low_u8x16).
    v128_t const not_below_u8x16 = wasm_v128_or(
        wasm_v128_andnot(wasm_u8x16_ge(high_u8x16, lo_high_u8x16), high_eq_lo_u8x16),
        wasm_v128_and(high_eq_lo_u8x16, wasm_u8x16_ge(low_u8x16, lo_low_u8x16)));
    // not_above_u8x16: high_u8x16 < hi_high_u8x16, or (high_u8x16 == hi_high_u8x16 and low_u8x16 <= hi_low_u8x16).
    v128_t const not_above_u8x16 = wasm_v128_or(
        wasm_v128_andnot(wasm_u8x16_lt(high_u8x16, hi_high_u8x16), high_eq_hi_u8x16),
        wasm_v128_and(high_eq_hi_u8x16, wasm_u8x16_ge(hi_low_u8x16, low_u8x16)));
    return wasm_v128_and(not_below_u8x16, not_above_u8x16);
}

/** @brief  A 64-bit "(high,low) 16-bit value in any sorted `[lo, hi]` range" lane mask over the four window quarters,
 *          the v128 twin of @ref sz_utf8_word_break_range16_mask_neon_ (WSegSpace / Extended_Pictographic). */
SZ_HELPER_AUTO sz_u64_t sz_utf8_word_break_range16_mask_v128_( //
    v128_t const *high, v128_t const *low, sz_u16_t const *lo_table, sz_u16_t const *hi_table, int count) {
    v128_t hit[4] = {wasm_i8x16_splat(0), wasm_i8x16_splat(0), wasm_i8x16_splat(0), wasm_i8x16_splat(0)};
    for (int range = 0; range < count; ++range)
        for (int quarter = 0; quarter < 4; ++quarter)
            hit[quarter] = wasm_v128_or(
                hit[quarter],
                sz_utf8_word_break_range16_one_v128_(high[quarter], low[quarter], lo_table[range], hi_table[range]));
    return sz_utf8_mask_combine_v128_(hit[0], hit[1], hit[2], hit[3]);
}

/**
 *  @brief  Per-ISA extractor: lower one classified 64-byte window to the portable @ref sz_utf8_word_break_frame_t — the
 *          v128 twin of @ref sz_utf8_word_break_build_frame_neon_. Applies the truncated-edge U+FFFD reclassify to the
 *          class quarters, materializes every per-class lane mask + the raw-byte membership masks, the
 *          Extended_Pictographic mask (BMP + SMP range scan), and the per-lane class byte array.
 */
SZ_HELPER_INLINE sz_utf8_word_break_frame_t sz_utf8_word_break_build_frame_v128_(
    sz_utf8_rune_window_v128_t window, v128_t *classes, sz_u64_t start_bytes_all, sz_u64_t length_two,
    sz_u64_t length_three, sz_u64_t length_four, int want_pictographic) {

    sz_size_t const loaded = window.loaded;
    sz_u64_t const valid = sz_u64_mask_until_serial_(loaded);
    sz_u64_t const start_bytes = start_bytes_all & valid;
    v128_t const *raw = window.window;

    // Truncated-edge U+FFFD reclassify (force the class to Other on a lead whose declared span runs past `loaded`).
    sz_u64_t const lead_two = length_two & start_bytes;
    sz_u64_t const lead_three = length_three & start_bytes;
    sz_u64_t const lead_four = length_four & start_bytes;
    sz_u64_t const truncated_raw = ((lead_two & ~sz_u64_mask_until_serial_(loaded > 1 ? loaded - 1 : 0)) |
                                    (lead_three & ~sz_u64_mask_until_serial_(loaded > 2 ? loaded - 2 : 0)) |
                                    (lead_four & ~sz_u64_mask_until_serial_(loaded > 3 ? loaded - 3 : 0))) &
                                   valid;
    if (truncated_raw) {
        v128_t const other_u8x16 = wasm_i8x16_splat((sz_i8_t)sz_utf8_word_break_other_k);
        for (int quarter = 0; quarter < 4; ++quarter) {
            v128_t const sel_u8x16 = sz_utf8_word_break_byte_mask_from_bits_v128_(truncated_raw, quarter * 16);
            classes[quarter] = wasm_v128_bitselect(other_u8x16, classes[quarter], sel_u8x16);
        }
    }

    sz_utf8_word_break_frame_t frame;
    frame.class_aletter = sz_utf8_word_break_class_mask_v128_(classes, sz_utf8_word_break_aletter_k);
    frame.class_hebrew = sz_utf8_word_break_class_mask_v128_(classes, sz_utf8_word_break_hebrew_letter_k);
    frame.class_numeric = sz_utf8_word_break_class_mask_v128_(classes, sz_utf8_word_break_numeric_k);
    frame.class_katakana = sz_utf8_word_break_class_mask_v128_(classes, sz_utf8_word_break_katakana_k);
    frame.class_extendnumlet = sz_utf8_word_break_class_mask_v128_(classes, sz_utf8_word_break_extendnumlet_k);
    frame.class_extend = sz_utf8_word_break_class_mask_v128_(classes, sz_utf8_word_break_extend_k);
    frame.class_zwj = sz_utf8_word_break_class_mask_v128_(classes, sz_utf8_word_break_zwj_k);
    frame.class_format = sz_utf8_word_break_class_mask_v128_(classes, sz_utf8_word_break_format_k);
    frame.class_midletter = sz_utf8_word_break_class_mask_v128_(classes, sz_utf8_word_break_midletter_k);
    frame.class_midnum = sz_utf8_word_break_class_mask_v128_(classes, sz_utf8_word_break_midnum_k);
    frame.class_mid_quotes = sz_utf8_word_break_class_mask_v128_(classes, sz_utf8_word_break_mid_quotes_k);
    frame.class_cr = sz_utf8_word_break_class_mask_v128_(classes, sz_utf8_word_break_cr_k);
    frame.class_lf = sz_utf8_word_break_class_mask_v128_(classes, sz_utf8_word_break_lf_k);
    frame.class_newline = sz_utf8_word_break_class_mask_v128_(classes, sz_utf8_word_break_newline_k);
    frame.class_regional = sz_utf8_word_break_class_mask_v128_(classes, sz_utf8_word_break_regional_ind_k);

    sz_u64_t const non_ascii_lanes = sz_utf8_word_break_byte_ge_v128_(raw, 0x80) & valid;
    frame.non_ascii_lanes = non_ascii_lanes;
    frame.double_quote_byte = sz_utf8_word_break_byte_equal_v128_(raw, 0x22) & valid;
    frame.single_quote_byte = sz_utf8_word_break_byte_equal_v128_(raw, 0x27) & valid;

    // WB3d WSegSpace raw membership: the ASCII U+0020 byte compare OR the multibyte (high,low) range scan.
    sz_u64_t wseg_multibyte = 0ull;
    if (non_ascii_lanes)
        wseg_multibyte = sz_utf8_word_break_range16_mask_v128_(window.high, window.low, sz_utf8_word_break_wseg_lo_,
                                                               sz_utf8_word_break_wseg_hi_,
                                                               sz_utf8_word_break_wseg_count_k) &
                         non_ascii_lanes;
    frame.wseg = (wseg_multibyte | (sz_utf8_word_break_byte_equal_v128_(raw, 0x20) & valid));

    // WB3c Extended_Pictographic raw membership (BMP range scan on non-4-byte lanes, SMP range scan on plane-one
    // 4-byte lanes). Rare-class gated on `want_pictographic` (an in-window ZWJ or the carried `prev_ends_in_zwj`), so
    // the ~156-range scan is skipped on the common no-ZWJ window. The engine applies the final gating.
    frame.pictographic = 0ull;
    sz_u64_t const four_byte = window.four_byte_starts & valid;
    if (want_pictographic) {
        v128_t next1[4], next2[4], next3[4];
        sz_utf8_forward_neighbours_v128_(raw, next1, next2, next3);
        v128_t const low_two_bits_u8x16 = wasm_i8x16_splat(0x03), low_three_bits_u8x16 = wasm_i8x16_splat(0x07),
                     low_nibble_u8x16 = wasm_i8x16_splat(0x0F), plane_high_bits_u8x16 = wasm_i8x16_splat(0x1C),
                     low_six_bits_u8x16 = wasm_i8x16_splat(0x3F), high_two_bits_u8x16 = wasm_i8x16_splat((sz_i8_t)0xC0),
                     high_nibble_u8x16 = wasm_i8x16_splat((sz_i8_t)0xF0);
        v128_t plane_q[4], smp_high[4], smp_low[4];
        for (int quarter = 0; quarter < 4; ++quarter) {
            v128_t const raw_quarter_u8x16 = raw[quarter], next1_u8x16 = next1[quarter], next2_u8x16 = next2[quarter],
                         next3_u8x16 = next3[quarter];
            plane_q[quarter] = wasm_v128_or(
                wasm_v128_and(wasm_i8x16_shl(wasm_v128_and(raw_quarter_u8x16, low_three_bits_u8x16), 2),
                              plane_high_bits_u8x16),
                sz_utf8_srl8_v128_(next1_u8x16, 4, 0x03));
            smp_high[quarter] = wasm_v128_or(
                wasm_v128_and(wasm_i8x16_shl(wasm_v128_and(next1_u8x16, low_nibble_u8x16), 4), high_nibble_u8x16),
                sz_utf8_srl8_v128_(next2_u8x16, 2, 0x0F));
            smp_low[quarter] = wasm_v128_or(
                wasm_v128_and(wasm_i8x16_shl(wasm_v128_and(next2_u8x16, low_two_bits_u8x16), 6), high_two_bits_u8x16),
                wasm_v128_and(next3_u8x16, low_six_bits_u8x16));
        }
        v128_t const one_u8x16 = wasm_i8x16_splat(1);
        sz_u64_t const plane_one = sz_utf8_mask_combine_v128_(
            wasm_i8x16_eq(plane_q[0], one_u8x16), wasm_i8x16_eq(plane_q[1], one_u8x16),
            wasm_i8x16_eq(plane_q[2], one_u8x16), wasm_i8x16_eq(plane_q[3], one_u8x16));
        sz_u64_t const pictographic_bmp = sz_utf8_word_break_range16_mask_v128_(
            window.high, window.low, sz_utf8_word_break_pict_bmp_lo_, sz_utf8_word_break_pict_bmp_hi_,
            sz_utf8_word_break_pict_bmp_count_k);
        sz_u64_t const pictographic_smp = sz_utf8_word_break_range16_mask_v128_(
            smp_high, smp_low, sz_utf8_word_break_pict_smp_lo_, sz_utf8_word_break_pict_smp_hi_,
            sz_utf8_word_break_pict_smp_count_k);
        frame.pictographic = (pictographic_bmp & non_ascii_lanes & ~four_byte) |
                             (pictographic_smp & four_byte & plane_one);
    }

    for (int quarter = 0; quarter < 4; ++quarter) wasm_v128_store(frame.classes_byte + quarter * 16, classes[quarter]);
    return frame;
}

#pragma endregion Mask algebra extractor

#pragma region Codepoint partition

/** @brief  Resolve one window into the maximal-subpart partition — the v128 twin of
 *          @ref sz_utf8_word_break_partition_neon_: compute the per-ISA `sz_u64_t` masks and delegate to the portable
 *          @ref sz_utf8_word_break_partition_from_masks_. */
SZ_HELPER_AUTO sz_utf8_word_break_partition_t sz_utf8_word_break_partition_v128_(sz_utf8_rune_window_v128_t window,
                                                                                 sz_u64_t valid, int at_end_of_text) {
    v128_t const *raw = window.window;
    sz_u64_t const real_continuation = window.continuation & valid;
    // Declared length follows the serial high-nibble rule: 0xC/0xD → 2, 0xE → 3, 0xF → 4. The strict
    // `two`/`three_byte_starts` masks already match 0xC0-0xDF and 0xE0-0xEF; only `length_four` needs widening to fold
    // the ill-formed leads 0xF8-0xFF so they collapse to U+FFFD like serial/Haswell instead of leaking as a class.
    sz_u64_t const length_two = window.two_byte_starts & valid;
    sz_u64_t const length_three = window.three_byte_starts & valid;
    sz_u64_t const length_four = (window.four_byte_starts | sz_utf8_word_break_byte_ge_v128_(raw, 0xF8)) & valid;
    sz_u64_t const length_ge_two = length_two | length_three | length_four;
    sz_u64_t bad_second_byte = 0ull;
    if (length_ge_two) {
        v128_t next1[4], next2[4], next3[4];
        sz_utf8_forward_neighbours_v128_(raw, next1, next2, next3);
        sz_u64_t const next1_at_least_a0 = sz_utf8_word_break_byte_ge_v128_(next1, 0xA0);
        sz_u64_t const next1_at_least_90 = sz_utf8_word_break_byte_ge_v128_(next1, 0x90);
        sz_u64_t const lead_c0_c1 =
            (sz_utf8_word_break_byte_equal_v128_(raw, 0xC0) | sz_utf8_word_break_byte_equal_v128_(raw, 0xC1)) & valid;
        sz_u64_t const lead_e0 = sz_utf8_word_break_byte_equal_v128_(raw, 0xE0) & valid;
        sz_u64_t const lead_ed = sz_utf8_word_break_byte_equal_v128_(raw, 0xED) & valid;
        sz_u64_t const lead_f0 = sz_utf8_word_break_byte_equal_v128_(raw, 0xF0) & valid;
        sz_u64_t const lead_f4 = sz_utf8_word_break_byte_equal_v128_(raw, 0xF4) & valid;
        sz_u64_t const lead_f5_or_more = sz_utf8_word_break_byte_ge_v128_(raw, 0xF5) & valid;
        bad_second_byte = lead_c0_c1 | (lead_e0 & ~next1_at_least_a0) | (lead_ed & next1_at_least_a0) |
                          (lead_f0 & ~next1_at_least_90) | (lead_f4 & next1_at_least_90) | lead_f5_or_more;
    }
    return sz_utf8_word_break_partition_from_masks_(real_continuation, length_two, length_three, length_four,
                                                    bad_second_byte, valid, at_end_of_text);
}

#pragma endregion Codepoint partition

#pragma region Forward driver

/**
 *  @brief  Forward UAX-29 word segmentation over `[0, length)` (WebAssembly SIMD128): the overlap-free advancing
 *          driver, mirroring @ref sz_utf8_wordbreaks_neon over the v128 window/classify/partition/decide/drain leaves.
 *          Bit-exact with `sz_utf8_wordbreaks_serial`, `sz_utf8_wordbreaks_neon`, and `sz_utf8_wordbreaks_rvv`.
 */
SZ_API_COMPTIME sz_size_t sz_utf8_wordbreaks_v128(   //
    sz_cptr_t text, sz_size_t length,                //
    sz_size_t *word_starts, sz_size_t *word_lengths, //
    sz_size_t words_capacity, sz_size_t *bytes_consumed) {

    if (length == 0 || words_capacity == 0) {
        if (bytes_consumed) *bytes_consumed = 0;
        return 0;
    }
    sz_u8_t const *text_u8 = (sz_u8_t const *)text;

    sz_size_t words = 0;         // words written to the output
    sz_size_t word_start = 0;    // start byte of the currently open (unfinished) word
    sz_size_t bridge_anchor = 0; // byte offset of the consumed, still-unresolved Mid* (valid while bridge_open)
    sz_size_t position = 0;      // codepoint-aligned anchor of the next window (advances cleanly)

    sz_utf8_word_break_carry_t carry = sz_utf8_word_break_carry_sot_();

    while (position < length) {
        sz_utf8_rune_window_v128_t const window = sz_utf8_rune_decode_window_v128_(text_u8 + position,
                                                                                   length - position);
        sz_size_t const loaded = window.loaded;
        sz_u64_t const valid = sz_u64_mask_until_serial_(loaded);

        v128_t classes[4];
        sz_utf8_word_break_classify_window_v128_(window, classes);

        sz_utf8_word_break_partition_t const partition = sz_utf8_word_break_partition_v128_(
            window, valid, position + loaded >= length);
        sz_u64_t const start_bytes_all = partition.start_bytes;
        sz_u64_t const continuation_all = partition.continuation;
        sz_u64_t const forced_other = partition.forced_other;
        sz_u64_t const length_two = partition.length_two;
        sz_u64_t const length_three = partition.length_three;
        sz_u64_t const length_four = partition.length_four;
        if (forced_other) {
            v128_t const other_u8x16 = wasm_i8x16_splat((sz_i8_t)sz_utf8_word_break_other_k);
            for (int quarter = 0; quarter < 4; ++quarter) {
                v128_t const sel_u8x16 = sz_utf8_word_break_byte_mask_from_bits_v128_(forced_other, quarter * 16);
                classes[quarter] = wasm_v128_bitselect(other_u8x16, classes[quarter], sel_u8x16);
            }
        }

        sz_bool_t const more_text = (position + loaded < length) ? sz_true_k : sz_false_k;

        // Effective-window<64: when more text follows, the final codepoint's blind byte span may spill past the
        // 64-byte edge. Trust and classify only up to the last fully decoded codepoint (`complete_limit`). The lowest
        // straddling lead and the last lead index come from the shuffle-LUT left-pack (ascending set-bit positions),
        // since the `ctz` / `clz` builtins are banned in this file.
        sz_size_t complete_limit = loaded;
        if (more_text) {
            sz_u64_t const two = length_two & start_bytes_all;
            sz_u64_t const three = length_three & start_bytes_all;
            sz_u64_t const four = length_four & start_bytes_all;
            sz_u64_t const straddle = ((two & ~sz_u64_mask_until_serial_(loaded > 1 ? loaded - 1 : 0)) |
                                       (three & ~sz_u64_mask_until_serial_(loaded > 2 ? loaded - 2 : 0)) |
                                       (four & ~sz_u64_mask_until_serial_(loaded > 3 ? loaded - 3 : 0))) &
                                      valid;
            sz_size_t limit = straddle ? (sz_size_t)(63 - sz_u64_clz(straddle & (~straddle + 1ull))) : loaded;
            if ((text_u8[position + loaded] & 0xC0) == 0x80) {
                sz_size_t const last_lead = (sz_size_t)(63 - sz_u64_clz(start_bytes_all));
                sz_size_t const last_lead_length = sz_utf8_lead_length_(text_u8[position + last_lead]);
                if (last_lead + last_lead_length > loaded && last_lead < limit) limit = last_lead;
            }
            if (limit > 0) complete_limit = limit;
        }

        int const want_pictographic = sz_utf8_word_break_class_mask_v128_(classes, sz_utf8_word_break_zwj_k) != 0 ||
                                      carry.prev_ends_in_zwj;
        sz_utf8_word_break_frame_t const frame = sz_utf8_word_break_build_frame_v128_(
            window, classes, start_bytes_all, length_two, length_three, length_four, want_pictographic);

        sz_utf8_word_break_carry_t carry_full = carry;
        sz_utf8_word_break_window_t const win = sz_utf8_word_break_decide_window_(
            &frame, start_bytes_all, continuation_all, forced_other, length_two, length_three, length_four,
            complete_limit, &carry_full, more_text);

        sz_size_t const adv = win.resolved;
        sz_u64_t const boundary_lanes = win.breaks & sz_u64_mask_until_serial_(adv);
        if (win.deferred_break) {
            if (words == words_capacity) {
                if (bytes_consumed) *bytes_consumed = word_start;
                return words;
            }
            word_starts[words] = word_start;
            word_lengths[words] = bridge_anchor - word_start;
            ++words;
            word_start = bridge_anchor;
        }

        words = sz_utf8_rune_drain_forward_v128_(boundary_lanes, position, word_starts, word_lengths, words,
                                                 words_capacity, &word_start);
        if (words == words_capacity) {
            if (bytes_consumed) *bytes_consumed = word_start;
            return words;
        }

        if (adv > 0 && adv < complete_limit) {
            sz_utf8_word_break_carry_t carry_to_edge = carry;
            sz_utf8_word_break_decide_window_(&frame, start_bytes_all, continuation_all, forced_other, length_two,
                                              length_three, length_four, adv, &carry_to_edge, sz_true_k);
            carry = carry_to_edge;
            position += adv;
        }
        else {
            int const bridge_opened = carry_full.bridge_open && !carry.bridge_open;
            carry = carry_full;
            if (bridge_opened) bridge_anchor = position;
            position += complete_limit ? complete_limit : loaded;
        }
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

#pragma endregion Forward driver

#pragma endregion UAX 29 Word Boundaries forward kernel

#if defined(__clang__)
#pragma clang attribute pop
#endif
#endif // SZ_USE_V128

#ifdef __cplusplus
}
#endif

#endif // STRINGZILLA_UTF8_WORDBREAKS_V128_H_
