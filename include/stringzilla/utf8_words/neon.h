/**
 *  @file   include/stringzilla/utf8_words/neon.h
 *  @author Ash Vardanian
 *  @brief  Fully-vectorized UAX-29 Word_Break segmentation for NEON (AArch64). The NEON twin of the AVX2 (Haswell)
 *          and Ice Lake kernels: no path scalar-walks codepoints, spills a vector to the stack to call the serial
 *          oracle, or issues a gather.
 *
 *  Each 64-byte window lives as four `uint8x16_t` quarters (the codepoint substrate twin of haswell's two `__m256i`
 *  halves); every per-codepoint Word_Break property resolves in-register via a `vqtbl` nibble cascade (the NEON
 *  stand-in for the VBMI trie / page LUT), bit-identical to `sz_rune_word_break_property` over the whole BMP and
 *  Supplementary Plane. The classified window is lowered to the portable @ref sz_utf8_word_break_frame_t and handed
 *  to the SHARED `sz_utf8_word_break_decide_window_` rule engine, so WB1-WB16 (including the cross-window bridge
 *  shadow / RI parity / left-context carry / WB3c next1/2/3 neighbour coupling) run once in portable `sz_u64_t` bit
 *  algebra. The WB3c (Extended_Pictographic) SMP neighbour precompute to the frame's `pictographic` mask is gated by
 *  `want_pictographic`, exactly as in haswell. Dense-compaction of the boundary mask uses the substrate's sparse
 *  `ctz` index loop (NEON has no `vpcompressb` / BMI2 `pext`).
 */
#ifndef STRINGZILLA_UTF8_WORDS_NEON_H_
#define STRINGZILLA_UTF8_WORDS_NEON_H_

#include "stringzilla/types.h"
#include "stringzilla/utf8_words/tables.h"
#include "stringzilla/utf8_words/serial.h"
#include "stringzilla/utf8_codepoints/neon.h"

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

#pragma region UAX 29 Word Boundaries forward kernel

#pragma region In register vectorized classifier

/** @brief  Build a per-quarter byte-boolean selector (0x00/0xFF) from the 16 lane bits of @p bits at offset @p shift,
 *          the NEON twin of @ref sz_utf8_codepoints_byte_mask_from_bits_haswell_ confined to one quarter. */
SZ_INTERNAL uint8x16_t sz_utf8_word_break_byte_mask_from_bits_neon_(sz_u64_t bits, int shift) {
    static sz_u8_t const bit_position_lanes[16] = {1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128};
    static sz_u8_t const lane_half[16] = {0, 0, 0, 0, 0, 0, 0, 0, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF};
    sz_u8_t const low_byte = (sz_u8_t)((bits >> shift) & 0xFF);
    sz_u8_t const high_byte = (sz_u8_t)((bits >> (shift + 8)) & 0xFF);
    uint8x16_t const byte = vbslq_u8(vld1q_u8(lane_half), vdupq_n_u8(high_byte), vdupq_n_u8(low_byte));
    uint8x16_t const position = vld1q_u8(bit_position_lanes);
    return vceqq_u8(vandq_u8(byte, position), position);
}

/** @brief  Word_Break class byte for sixteen BMP codepoints (per-lane high = cp>>8, low = cp&0xFF) via a
 *          register-resident `vqtbl` nibble cascade, the NEON twin of @ref sz_utf8_word_break_bmp_class_haswell_.
 *          Gather-free; bit-exact with `sz_rune_word_break_property` over the whole BMP. Addresses ONE quarter; the
 *          caller iterates the four quarters. */
SZ_INTERNAL uint8x16_t sz_utf8_word_break_bmp_class_neon_(uint8x16_t high, uint8x16_t low) {
    uint8x16_t const low_nibble_mask = vdupq_n_u8(0x0F);
    uint8x16_t const page = sz_utf8_codepoints_lut256_neon_(sz_utf8_word_break_haswell_stage1_, high);
    uint8x16_t const low_high = vandq_u8(vshrq_n_u8(low, 4), low_nibble_mask);
    uint8x16_t const leaf_lo = sz_utf8_codepoints_cascade_stage_neon_(
        sz_utf8_word_break_haswell_stage2_lo_, sz_utf8_word_break_haswell_stage2_lo_count_k / 16, page, low_high);
    uint8x16_t const leaf_hi = sz_utf8_codepoints_cascade_stage_neon_(
        sz_utf8_word_break_haswell_stage2_hi_, sz_utf8_word_break_haswell_stage2_hi_count_k / 16, page, low_high);
    uint8x16_t const leaf_group = vorrq_u8(vandq_u8(vshrq_n_u8(leaf_lo, 4), low_nibble_mask), vshlq_n_u8(leaf_hi, 4));
    uint8x16_t const leaf_low_nibble = vandq_u8(leaf_lo, low_nibble_mask);
    uint8x16_t const low_low = vandq_u8(low, low_nibble_mask);
    uint8x16_t const lut_index = vorrq_u8(vshlq_n_u8(leaf_low_nibble, 4), low_low);
    uint8x16_t result = vdupq_n_u8(0);
    for (int group = 0; group < (int)sz_utf8_word_break_haswell_leaf_groups_k; ++group) {
        uint8x16_t const value = sz_utf8_codepoints_lut256_neon_(
            sz_utf8_word_break_haswell_stage3_groups_ + group * 256, lut_index);
        uint8x16_t const here = vceqq_u8(leaf_group, vdupq_n_u8((sz_u8_t)group));
        result = vbslq_u8(here, value, result);
    }
    return result;
}

/** @brief  Word_Break class byte for sixteen ASTRAL codepoints over the 20-bit offset = cp - 0x10000 (5-nibble
 *          cascade), the NEON twin of @ref sz_utf8_word_break_astral_class_haswell_. Per-lane bytes:
 *          @p plane_off = (offset>>16)&0xFF (low nibble meaningful), @p high = (offset>>8)&0xFF, @p low = offset&0xFF.
 *          Gather-free; bit-exact with `sz_rune_word_break_property` over the Supplementary Planes. Addresses ONE
 *          quarter; the caller iterates the four quarters. */
SZ_INTERNAL uint8x16_t sz_utf8_word_break_astral_class_neon_(uint8x16_t plane_off, uint8x16_t high, uint8x16_t low) {
    uint8x16_t const low_nibble_mask = vdupq_n_u8(0x0F);
    uint8x16_t const n4 = vandq_u8(plane_off, low_nibble_mask);
    uint8x16_t const n3 = vandq_u8(vshrq_n_u8(high, 4), low_nibble_mask);
    uint8x16_t const stage1_index = vorrq_u8(vshlq_n_u8(n4, 4), n3);
    uint8x16_t const page = sz_utf8_codepoints_lut256_neon_(sz_utf8_word_break_haswell_astral_stage1_, stage1_index);
    uint8x16_t const n2 = vandq_u8(high, low_nibble_mask);
    uint8x16_t const leaf2 = sz_utf8_codepoints_cascade_stage_neon_(
        sz_utf8_word_break_haswell_astral_stage2_lo_, sz_utf8_word_break_haswell_astral_stage2_lo_count_k / 16, page,
        n2);
    uint8x16_t const n1 = vandq_u8(vshrq_n_u8(low, 4), low_nibble_mask);
    uint8x16_t const leaf_lo = sz_utf8_codepoints_cascade_stage_neon_(
        sz_utf8_word_break_haswell_astral_stage3_lo_, sz_utf8_word_break_haswell_astral_stage3_lo_count_k / 16, leaf2,
        n1);
    uint8x16_t const leaf_hi = sz_utf8_codepoints_cascade_stage_neon_(
        sz_utf8_word_break_haswell_astral_stage3_hi_, sz_utf8_word_break_haswell_astral_stage3_hi_count_k / 16, leaf2,
        n1);
    uint8x16_t const n0 = vandq_u8(low, low_nibble_mask);
    uint8x16_t const leaf_group = vorrq_u8(vandq_u8(vshrq_n_u8(leaf_lo, 4), low_nibble_mask), vshlq_n_u8(leaf_hi, 4));
    uint8x16_t const leaf_low_nibble = vandq_u8(leaf_lo, low_nibble_mask);
    uint8x16_t const stage4_lut_index = vorrq_u8(vshlq_n_u8(leaf_low_nibble, 4), n0);
    uint8x16_t result = vdupq_n_u8(0);
    for (int group = 0; group < (int)sz_utf8_word_break_haswell_astral_leaf_groups_k; ++group) {
        uint8x16_t const value = sz_utf8_codepoints_lut256_neon_(
            sz_utf8_word_break_haswell_astral_stage4_groups_ + group * 256, stage4_lut_index);
        uint8x16_t const here = vceqq_u8(leaf_group, vdupq_n_u8((sz_u8_t)group));
        result = vbslq_u8(here, value, result);
    }
    return result;
}

/** @brief  Word_Break class byte for sixteen ASCII codepoints (cp < 0x80) via the existing 128-entry property table,
 *          read in-register by two `vqtbl4q_u8` halves (low six bits) blended on bit 6, the NEON twin of
 *          @ref sz_utf8_word_break_ascii_class_haswell_. The window byte equals the codepoint on ASCII lanes.
 *          Addresses ONE quarter. */
SZ_INTERNAL uint8x16_t sz_utf8_word_break_ascii_class_neon_(uint8x16_t bytes) {
    uint8x16_t const index_low6 = vandq_u8(bytes, vdupq_n_u8(0x3F));
    uint8x16_t const low_half = sz_utf8_codepoints_lut256_neon_(sz_utf8_word_break_property_ascii_ + 0, index_low6);
    uint8x16_t const high_half = sz_utf8_codepoints_lut256_neon_(sz_utf8_word_break_property_ascii_ + 64, index_low6);
    uint8x16_t const bit6 = vdupq_n_u8(0x40);
    uint8x16_t const high_bit = vceqq_u8(vandq_u8(bytes, bit6), bit6);
    return vbslq_u8(high_bit, high_half, low_half);
}

/** @brief  Start-compacting BMP classify: the BMP `vqtbl` nibble cascade is bit-exact but expensive (a 16-group leaf
 *          scan per quarter), and its output is consumed only on 2-/3-byte codepoint-START lanes (ASCII lanes are
 *          overwritten by the property-table blend, 4-byte lanes by the astral blend, continuation lanes are
 *          don't-cares). A 64-byte window holds at most 32 such BMP starts, so this gathers their `(high, low)` bytes
 *          into a dense buffer via the substrate `ctz` index loop (NEON has no `vpcompressb` / BMI2 `pext`), runs the
 *          cascade only over the populated quarters (one or two `uint8x16_t`) instead of all four, then scatters the
 *          dense class bytes back to their original byte lanes in @p bmp_out (zeroed elsewhere). Bit-identical to four
 *          full @ref sz_utf8_word_break_bmp_class_neon_ quarters on every BMP-start lane; every other lane is a
 *          don't-care left at zero. */
SZ_INTERNAL void sz_utf8_word_break_bmp_compact_neon_(sz_u64_t bmp_starts, uint8x16_t const *high,
                                                      uint8x16_t const *low, uint8x16_t *bmp_out) {
    sz_u8_t high_bytes[64], low_bytes[64];
    for (int quarter = 0; quarter < 4; ++quarter) {
        vst1q_u8(high_bytes + quarter * 16, high[quarter]);
        vst1q_u8(low_bytes + quarter * 16, low[quarter]);
    }

    sz_u8_t dense_high[64] = {0}, dense_low[64] = {0};
    sz_u64_t remaining = bmp_starts;
    sz_size_t dense_index = 0;
    while (remaining) {
        sz_size_t const lane = (sz_size_t)sz_u64_ctz(remaining);
        remaining &= remaining - 1;
        dense_high[dense_index] = high_bytes[lane];
        dense_low[dense_index] = low_bytes[lane];
        ++dense_index;
    }

    sz_u8_t class_bytes[64];
    int const populated_quarters = (int)((dense_index + 15) / 16);
    for (int quarter = 0; quarter < populated_quarters; ++quarter) {
        uint8x16_t const dense_class = sz_utf8_word_break_bmp_class_neon_(vld1q_u8(dense_high + quarter * 16),
                                                                          vld1q_u8(dense_low + quarter * 16));
        vst1q_u8(class_bytes + quarter * 16, dense_class);
    }

    sz_u8_t scatter[64] = {0};
    remaining = bmp_starts;
    dense_index = 0;
    while (remaining) {
        sz_size_t const lane = (sz_size_t)sz_u64_ctz(remaining);
        remaining &= remaining - 1;
        scatter[lane] = class_bytes[dense_index++];
    }
    for (int quarter = 0; quarter < 4; ++quarter) bmp_out[quarter] = vld1q_u8(scatter + quarter * 16);
}

/** @brief  Per-window byte-lane classification (NEON): the Word_Break class byte per lane as four `uint8x16_t`
 *          quarters, valid only on codepoint-start lanes (the engine reads classes only at starts). The NEON twin of
 *          @ref sz_utf8_word_break_classify_window_haswell_, bit-identical on every start lane. ASCII through the
 *          property table, BMP through the nibble cascade, 4-byte leads through the astral cascade with the codepoint
 *          high/low/plane reconstructed from the forward neighbours. */
SZ_INTERNAL void sz_utf8_word_break_classify_window_neon_( //
    sz_utf8_codepoints_window_neon_t window, uint8x16_t *classes) {
    uint8x16_t const *raw = window.window;
    sz_u64_t const ascii_starts = window.codepoint_starts & ~window.two_byte_starts & ~window.three_byte_starts &
                                  ~window.four_byte_starts;

    uint8x16_t next1[4], next2[4], next3[4];
    sz_utf8_codepoints_forward_neighbours_neon_(raw, next1, next2, next3);

    uint8x16_t const c03 = vdupq_n_u8(0x03), c07 = vdupq_n_u8(0x07), c0f = vdupq_n_u8(0x0F), c1c = vdupq_n_u8(0x1C),
                     c3f = vdupq_n_u8(0x3F), cc0 = vdupq_n_u8(0xC0), cf0 = vdupq_n_u8(0xF0);

    // BMP class via the cascade over the compacted 2-/3-byte START lanes; ASCII / 4-byte / continuation lanes are
    // don't-cares (overwritten or unread below), so the dense walk leaves them at zero.
    uint8x16_t bmp_out[4] = {vdupq_n_u8(0), vdupq_n_u8(0), vdupq_n_u8(0), vdupq_n_u8(0)};
    sz_u64_t const bmp_starts = window.two_byte_starts | window.three_byte_starts;
    if (bmp_starts) sz_utf8_word_break_bmp_compact_neon_(bmp_starts, window.high, window.low, bmp_out);

    for (int quarter = 0; quarter < 4; ++quarter) {
        uint8x16_t const raw_q = raw[quarter];
        int const lane_base = quarter * 16;

        uint8x16_t out_q = bmp_out[quarter];

        // ASCII lanes: read the 128-entry property table directly off the raw byte.
        uint8x16_t const ascii_select = sz_utf8_word_break_byte_mask_from_bits_neon_(ascii_starts, lane_base);
        out_q = vbslq_u8(ascii_select, sz_utf8_word_break_ascii_class_neon_(raw_q), out_q);

        // 4-byte (astral) lanes: reconstruct the codepoint from the lead + three forward neighbours, then the astral
        // cascade addressed by offset = codepoint - 0x10000 (the offset's plane nibble is `plane - 1`).
        if (window.four_byte_starts) {
            uint8x16_t const n1 = next1[quarter], n2 = next2[quarter], n3 = next3[quarter];
            uint8x16_t const four_select = sz_utf8_word_break_byte_mask_from_bits_neon_(window.four_byte_starts,
                                                                                        lane_base);
            uint8x16_t const plane = vorrq_u8(vandq_u8(vshlq_n_u8(vandq_u8(raw_q, c07), 2), c1c),
                                              sz_utf8_codepoints_srl8_neon_(n1, 4, 0x03));
            uint8x16_t const high_four = vorrq_u8(vandq_u8(vshlq_n_u8(vandq_u8(n1, c0f), 4), cf0),
                                                  sz_utf8_codepoints_srl8_neon_(n2, 2, 0x0F));
            uint8x16_t const low_four = vorrq_u8(vandq_u8(vshlq_n_u8(vandq_u8(n2, c03), 6), cc0), vandq_u8(n3, c3f));
            uint8x16_t const plane_off = vsubq_u8(plane, vdupq_n_u8(1));
            out_q = vbslq_u8(four_select, sz_utf8_word_break_astral_class_neon_(plane_off, high_four, low_four), out_q);
        }
        classes[quarter] = out_q;
    }
}

#pragma endregion In register vectorized classifier

#pragma region Mask algebra extractor

/** @brief  A 64-bit "class byte == @p value" lane mask over the four class quarters (four `vceqq_u8` -> mask_combine). */
SZ_INTERNAL sz_u64_t sz_utf8_word_break_class_mask_neon_(uint8x16_t const *classes, sz_u8_t value) {
    uint8x16_t const v = vdupq_n_u8(value);
    return sz_utf8_codepoints_mask_combine_neon_(vceqq_u8(classes[0], v), vceqq_u8(classes[1], v),
                                                 vceqq_u8(classes[2], v), vceqq_u8(classes[3], v));
}

/** @brief  A 64-bit "raw window byte == @p value" lane mask over the four window quarters. */
SZ_INTERNAL sz_u64_t sz_utf8_word_break_byte_equal_neon_(uint8x16_t const *quarters, sz_u8_t value) {
    uint8x16_t const v = vdupq_n_u8(value);
    return sz_utf8_codepoints_mask_combine_neon_(vceqq_u8(quarters[0], v), vceqq_u8(quarters[1], v),
                                                 vceqq_u8(quarters[2], v), vceqq_u8(quarters[3], v));
}

/** @brief  A 64-bit "raw window byte >= @p bound" (unsigned) lane mask over the four window quarters (`vcgeq_u8`). */
SZ_INTERNAL sz_u64_t sz_utf8_word_break_byte_ge_neon_(uint8x16_t const *quarters, sz_u8_t bound) {
    uint8x16_t const v = vdupq_n_u8(bound);
    return sz_utf8_codepoints_mask_combine_neon_(vcgeq_u8(quarters[0], v), vcgeq_u8(quarters[1], v),
                                                 vcgeq_u8(quarters[2], v), vcgeq_u8(quarters[3], v));
}

/** @brief  Per-quarter "(high,low) 16-bit value in `[lo, hi]`" membership for one range, the NEON unsigned 16-bit
 *          window-compare building block of @ref sz_utf8_word_break_range16_mask_neon_. `vcgeq_u8`/`vcltq_u8` are
 *          native, so no `max_epu8` emulation is needed. */
SZ_INTERNAL uint8x16_t sz_utf8_word_break_range16_one_neon_(uint8x16_t high, uint8x16_t low, sz_u16_t lo, sz_u16_t hi) {
    uint8x16_t const lo_high = vdupq_n_u8((sz_u8_t)(lo >> 8)), lo_low = vdupq_n_u8((sz_u8_t)(lo & 0xFF));
    uint8x16_t const hi_high = vdupq_n_u8((sz_u8_t)(hi >> 8)), hi_low = vdupq_n_u8((sz_u8_t)(hi & 0xFF));
    uint8x16_t const high_eq_lo = vceqq_u8(high, lo_high);
    uint8x16_t const high_eq_hi = vceqq_u8(high, hi_high);
    // not_below: high > lo_high, or (high == lo_high and low >= lo_low).
    uint8x16_t const not_below = vorrq_u8(vbicq_u8(vcgeq_u8(high, lo_high), high_eq_lo),
                                          vandq_u8(high_eq_lo, vcgeq_u8(low, lo_low)));
    // not_above: high < hi_high, or (high == hi_high and low <= hi_low).
    uint8x16_t const not_above = vorrq_u8(vbicq_u8(vcltq_u8(high, hi_high), high_eq_hi),
                                          vandq_u8(high_eq_hi, vcgeq_u8(hi_low, low)));
    return vandq_u8(not_below, not_above);
}

/** @brief  A 64-bit "(high,low) 16-bit value in any sorted `[lo, hi]` range" lane mask over the four window quarters,
 *          the NEON twin of @ref sz_utf8_word_break_range16_mask_haswell_ (WSegSpace / Extended_Pictographic). */
SZ_INTERNAL sz_u64_t sz_utf8_word_break_range16_mask_neon_( //
    uint8x16_t const *high, uint8x16_t const *low, sz_u16_t const *lo_table, sz_u16_t const *hi_table, int count) {
    uint8x16_t hit[4] = {vdupq_n_u8(0), vdupq_n_u8(0), vdupq_n_u8(0), vdupq_n_u8(0)};
    for (int range = 0; range < count; ++range)
        for (int quarter = 0; quarter < 4; ++quarter)
            hit[quarter] = vorrq_u8(hit[quarter], sz_utf8_word_break_range16_one_neon_(
                                                      high[quarter], low[quarter], lo_table[range], hi_table[range]));
    return sz_utf8_codepoints_mask_combine_neon_(hit[0], hit[1], hit[2], hit[3]);
}

/**
 *  @brief  Per-ISA extractor: lower one classified 64-byte window to the portable @ref sz_utf8_word_break_frame_t - the
 *          NEON twin of @ref sz_utf8_word_break_build_frame_haswell_. Applies the truncated-edge U+FFFD reclassify to
 *          the class quarters, materializes every per-class lane mask + the raw-byte membership masks, the
 *          Extended_Pictographic mask (BMP + SMP range scan), and the per-lane class byte array.
 */
SZ_FORCE_INLINE sz_utf8_word_break_frame_t sz_utf8_word_break_build_frame_neon_(
    sz_utf8_codepoints_window_neon_t window, uint8x16_t *classes, sz_u64_t start_bytes_all, sz_u64_t length_two,
    sz_u64_t length_three, sz_u64_t length_four, int want_pictographic) {

    sz_size_t const loaded = window.loaded;
    sz_u64_t const valid = sz_utf8_codepoints_mask_until_(loaded);
    sz_u64_t const start_bytes = start_bytes_all & valid;
    uint8x16_t const *raw = window.window;

    // Truncated-edge U+FFFD reclassify (force the class to Other on a lead whose declared span runs past `loaded`).
    sz_u64_t const lead_two = length_two & start_bytes;
    sz_u64_t const lead_three = length_three & start_bytes;
    sz_u64_t const lead_four = length_four & start_bytes;
    sz_u64_t const truncated_raw = ((lead_two & ~sz_utf8_codepoints_mask_until_(loaded > 1 ? loaded - 1 : 0)) |
                                    (lead_three & ~sz_utf8_codepoints_mask_until_(loaded > 2 ? loaded - 2 : 0)) |
                                    (lead_four & ~sz_utf8_codepoints_mask_until_(loaded > 3 ? loaded - 3 : 0))) &
                                   valid;
    if (truncated_raw) {
        uint8x16_t const other = vdupq_n_u8((sz_u8_t)sz_utf8_word_break_other_k);
        for (int quarter = 0; quarter < 4; ++quarter) {
            uint8x16_t const sel = sz_utf8_word_break_byte_mask_from_bits_neon_(truncated_raw, quarter * 16);
            classes[quarter] = vbslq_u8(sel, other, classes[quarter]);
        }
    }

    sz_utf8_word_break_frame_t frame;
    frame.class_aletter = sz_utf8_word_break_class_mask_neon_(classes, sz_utf8_word_break_aletter_k);
    frame.class_hebrew = sz_utf8_word_break_class_mask_neon_(classes, sz_utf8_word_break_hebrew_letter_k);
    frame.class_numeric = sz_utf8_word_break_class_mask_neon_(classes, sz_utf8_word_break_numeric_k);
    frame.class_katakana = sz_utf8_word_break_class_mask_neon_(classes, sz_utf8_word_break_katakana_k);
    frame.class_extendnumlet = sz_utf8_word_break_class_mask_neon_(classes, sz_utf8_word_break_extendnumlet_k);
    frame.class_extend = sz_utf8_word_break_class_mask_neon_(classes, sz_utf8_word_break_extend_k);
    frame.class_zwj = sz_utf8_word_break_class_mask_neon_(classes, sz_utf8_word_break_zwj_k);
    frame.class_format = sz_utf8_word_break_class_mask_neon_(classes, sz_utf8_word_break_format_k);
    frame.class_midletter = sz_utf8_word_break_class_mask_neon_(classes, sz_utf8_word_break_midletter_k);
    frame.class_midnum = sz_utf8_word_break_class_mask_neon_(classes, sz_utf8_word_break_midnum_k);
    frame.class_mid_quotes = sz_utf8_word_break_class_mask_neon_(classes, sz_utf8_word_break_mid_quotes_k);
    frame.class_cr = sz_utf8_word_break_class_mask_neon_(classes, sz_utf8_word_break_cr_k);
    frame.class_lf = sz_utf8_word_break_class_mask_neon_(classes, sz_utf8_word_break_lf_k);
    frame.class_newline = sz_utf8_word_break_class_mask_neon_(classes, sz_utf8_word_break_newline_k);
    frame.class_regional = sz_utf8_word_break_class_mask_neon_(classes, sz_utf8_word_break_regional_ind_k);

    sz_u64_t const non_ascii_lanes = sz_utf8_word_break_byte_ge_neon_(raw, 0x80) & valid;
    frame.non_ascii_lanes = non_ascii_lanes;
    frame.double_quote_byte = sz_utf8_word_break_byte_equal_neon_(raw, 0x22) & valid;
    frame.single_quote_byte = sz_utf8_word_break_byte_equal_neon_(raw, 0x27) & valid;

    // WB3d WSegSpace raw membership: the ASCII U+0020 byte compare OR the multibyte (high,low) range scan.
    sz_u64_t wseg_multibyte = 0ull;
    if (non_ascii_lanes)
        wseg_multibyte = sz_utf8_word_break_range16_mask_neon_(window.high, window.low, sz_utf8_word_break_wseg_lo_,
                                                               sz_utf8_word_break_wseg_hi_,
                                                               sz_utf8_word_break_wseg_count_k) &
                         non_ascii_lanes;
    frame.wseg = (wseg_multibyte | (sz_utf8_word_break_byte_equal_neon_(raw, 0x20) & valid));

    // WB3c Extended_Pictographic raw membership (BMP range scan on non-4-byte lanes, SMP range scan on plane-one
    // 4-byte lanes). Rare-class gated on `want_pictographic` (an in-window ZWJ or the carried `prev_ends_in_zwj`), so
    // the ~156-range scan is skipped on the common no-ZWJ window. The engine applies the final gating.
    frame.pictographic = 0ull;
    sz_u64_t const four_byte = window.four_byte_starts & valid;
    if (want_pictographic) {
        uint8x16_t next1[4], next2[4], next3[4];
        sz_utf8_codepoints_forward_neighbours_neon_(raw, next1, next2, next3);
        uint8x16_t const c03 = vdupq_n_u8(0x03), c07 = vdupq_n_u8(0x07), c0f = vdupq_n_u8(0x0F), c1c = vdupq_n_u8(0x1C),
                         c3f = vdupq_n_u8(0x3F), cc0 = vdupq_n_u8(0xC0), cf0 = vdupq_n_u8(0xF0);
        uint8x16_t plane_q[4], smp_high[4], smp_low[4];
        for (int quarter = 0; quarter < 4; ++quarter) {
            uint8x16_t const raw_q = raw[quarter], n1 = next1[quarter], n2 = next2[quarter], n3 = next3[quarter];
            plane_q[quarter] = vorrq_u8(vandq_u8(vshlq_n_u8(vandq_u8(raw_q, c07), 2), c1c),
                                        sz_utf8_codepoints_srl8_neon_(n1, 4, 0x03));
            smp_high[quarter] = vorrq_u8(vandq_u8(vshlq_n_u8(vandq_u8(n1, c0f), 4), cf0),
                                         sz_utf8_codepoints_srl8_neon_(n2, 2, 0x0F));
            smp_low[quarter] = vorrq_u8(vandq_u8(vshlq_n_u8(vandq_u8(n2, c03), 6), cc0), vandq_u8(n3, c3f));
        }
        uint8x16_t const one = vdupq_n_u8(1);
        sz_u64_t const plane_one = sz_utf8_codepoints_mask_combine_neon_(
            vceqq_u8(plane_q[0], one), vceqq_u8(plane_q[1], one), vceqq_u8(plane_q[2], one), vceqq_u8(plane_q[3], one));
        sz_u64_t const pictographic_bmp = sz_utf8_word_break_range16_mask_neon_(
            window.high, window.low, sz_utf8_word_break_pict_bmp_lo_, sz_utf8_word_break_pict_bmp_hi_,
            sz_utf8_word_break_pict_bmp_count_k);
        sz_u64_t const pictographic_smp = sz_utf8_word_break_range16_mask_neon_(
            smp_high, smp_low, sz_utf8_word_break_pict_smp_lo_, sz_utf8_word_break_pict_smp_hi_,
            sz_utf8_word_break_pict_smp_count_k);
        frame.pictographic = (pictographic_bmp & non_ascii_lanes & ~four_byte) |
                             (pictographic_smp & four_byte & plane_one);
    }

    for (int quarter = 0; quarter < 4; ++quarter) vst1q_u8(frame.classes_byte + quarter * 16, classes[quarter]);
    return frame;
}

#pragma endregion Mask algebra extractor

#pragma region Codepoint partition

/** @brief  Resolve one window into the maximal-subpart partition - the NEON twin of
 *          @ref sz_utf8_word_break_partition_haswell_: compute the per-ISA `sz_u64_t` masks and delegate to the
 *          portable @ref sz_utf8_word_break_partition_from_masks_. */
SZ_INTERNAL sz_utf8_word_break_partition_t sz_utf8_word_break_partition_neon_(sz_utf8_codepoints_window_neon_t window,
                                                                              sz_u64_t valid, int at_end_of_text) {
    uint8x16_t const *raw = window.window;
    sz_u64_t const real_continuation = window.continuation & valid;
    sz_u64_t const length_two = window.two_byte_starts & valid;
    sz_u64_t const length_three = window.three_byte_starts & valid;
    sz_u64_t const length_four = window.four_byte_starts & valid;
    sz_u64_t const length_ge_two = length_two | length_three | length_four;
    sz_u64_t bad_second_byte = 0ull;
    if (length_ge_two) {
        uint8x16_t next1[4], next2[4], next3[4];
        sz_utf8_codepoints_forward_neighbours_neon_(raw, next1, next2, next3);
        sz_u64_t const next1_at_least_a0 = sz_utf8_word_break_byte_ge_neon_(next1, 0xA0);
        sz_u64_t const next1_at_least_90 = sz_utf8_word_break_byte_ge_neon_(next1, 0x90);
        sz_u64_t const lead_c0_c1 =
            (sz_utf8_word_break_byte_equal_neon_(raw, 0xC0) | sz_utf8_word_break_byte_equal_neon_(raw, 0xC1)) & valid;
        sz_u64_t const lead_e0 = sz_utf8_word_break_byte_equal_neon_(raw, 0xE0) & valid;
        sz_u64_t const lead_ed = sz_utf8_word_break_byte_equal_neon_(raw, 0xED) & valid;
        sz_u64_t const lead_f0 = sz_utf8_word_break_byte_equal_neon_(raw, 0xF0) & valid;
        sz_u64_t const lead_f4 = sz_utf8_word_break_byte_equal_neon_(raw, 0xF4) & valid;
        sz_u64_t const lead_f5_or_more = sz_utf8_word_break_byte_ge_neon_(raw, 0xF5) & valid;
        bad_second_byte = lead_c0_c1 | (lead_e0 & ~next1_at_least_a0) | (lead_ed & next1_at_least_a0) |
                          (lead_f0 & ~next1_at_least_90) | (lead_f4 & next1_at_least_90) | lead_f5_or_more;
    }
    return sz_utf8_word_break_partition_from_masks_(real_continuation, length_two, length_three, length_four,
                                                    bad_second_byte, valid, at_end_of_text);
}

#pragma endregion Codepoint partition

#pragma region Forward driver

/**
 *  @brief  Forward UAX-29 word segmentation over `[0, length)` (NEON AArch64): the overlap-free advancing driver,
 *          mirroring @ref sz_utf8_words_haswell over the NEON window/classify/partition/decide/drain leaves. Bit-exact
 *          with `sz_utf8_words_serial`, `sz_utf8_words_haswell`, and `sz_utf8_words_icelake`.
 */
SZ_PUBLIC sz_size_t sz_utf8_words_neon(              //
    sz_cptr_t text, sz_size_t length,                //
    sz_size_t *word_starts, sz_size_t *word_lengths, //
    sz_size_t words_capacity, sz_size_t *bytes_consumed) {

    if (length == 0 || words_capacity == 0) {
        if (bytes_consumed) *bytes_consumed = 0;
        return 0;
    }
    sz_u8_t const *text_u8 = (sz_u8_t const *)text;

    sz_size_t words = 0;      // words written to the output
    sz_size_t word_start = 0; // start byte of the currently open (unfinished) word
    sz_size_t position = 0;   // codepoint-aligned anchor of the next window (advances cleanly)

    sz_utf8_word_break_carry_t carry = sz_utf8_word_break_carry_sot_();

    while (position < length) {
        sz_utf8_codepoints_window_neon_t const window = sz_utf8_codepoints_decode_window_neon_(text_u8 + position,
                                                                                               length - position);
        sz_size_t const loaded = window.loaded;
        sz_u64_t const valid = sz_utf8_codepoints_mask_until_(loaded);

        uint8x16_t classes[4];
        sz_utf8_word_break_classify_window_neon_(window, classes);

        sz_utf8_word_break_partition_t const partition = sz_utf8_word_break_partition_neon_(
            window, valid, position + loaded >= length);
        sz_u64_t const start_bytes_all = partition.start_bytes;
        sz_u64_t const continuation_all = partition.continuation;
        sz_u64_t const forced_other = partition.forced_other;
        sz_u64_t const length_two = partition.length_two;
        sz_u64_t const length_three = partition.length_three;
        sz_u64_t const length_four = partition.length_four;
        if (forced_other) {
            uint8x16_t const other = vdupq_n_u8((sz_u8_t)sz_utf8_word_break_other_k);
            for (int quarter = 0; quarter < 4; ++quarter) {
                uint8x16_t const sel = sz_utf8_word_break_byte_mask_from_bits_neon_(forced_other, quarter * 16);
                classes[quarter] = vbslq_u8(sel, other, classes[quarter]);
            }
        }

        sz_bool_t const more_text = (position + loaded < length) ? sz_true_k : sz_false_k;

        // Effective-window<64: when more text follows, the final codepoint's blind byte span may spill past the
        // 64-byte edge. Trust and classify only up to the last fully decoded codepoint (`complete_limit`).
        sz_size_t complete_limit = loaded;
        if (more_text) {
            sz_u64_t const two = length_two & start_bytes_all;
            sz_u64_t const three = length_three & start_bytes_all;
            sz_u64_t const four = length_four & start_bytes_all;
            sz_u64_t const straddle = ((two & ~sz_utf8_codepoints_mask_until_(loaded > 1 ? loaded - 1 : 0)) |
                                       (three & ~sz_utf8_codepoints_mask_until_(loaded > 2 ? loaded - 2 : 0)) |
                                       (four & ~sz_utf8_codepoints_mask_until_(loaded > 3 ? loaded - 3 : 0))) &
                                      valid;
            sz_size_t limit = straddle ? (sz_size_t)sz_u64_ctz(straddle) : loaded;
            if ((text_u8[position + loaded] & 0xC0) == 0x80) {
                sz_size_t const last_lead = (sz_size_t)(63 - sz_u64_clz(start_bytes_all));
                sz_size_t const last_lead_length = sz_utf8_codepoint_length_(text_u8[position + last_lead]);
                if (last_lead + last_lead_length > loaded && last_lead < limit) limit = last_lead;
            }
            if (limit > 0) complete_limit = limit;
        }

        int const want_pictographic = sz_utf8_word_break_class_mask_neon_(classes, sz_utf8_word_break_zwj_k) != 0 ||
                                      carry.prev_ends_in_zwj;
        sz_utf8_word_break_frame_t const frame = sz_utf8_word_break_build_frame_neon_(
            window, classes, start_bytes_all, length_two, length_three, length_four, want_pictographic);

        sz_utf8_word_break_carry_t carry_full = carry;
        sz_utf8_word_break_window_t const win = sz_utf8_word_break_decide_window_(
            &frame, start_bytes_all, continuation_all, forced_other, length_two, length_three, length_four,
            complete_limit, &carry_full, more_text);

        sz_size_t const adv = win.resolved;
        sz_u64_t const boundary_lanes = win.breaks & sz_utf8_codepoints_mask_until_(adv);

        words = sz_utf8_codepoints_drain_forward_neon_(boundary_lanes, position, word_starts, word_lengths, words,
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
            carry = carry_full;
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
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#endif // SZ_USE_NEON

#ifdef __cplusplus
}
#endif

#endif // STRINGZILLA_UTF8_WORDS_NEON_H_
