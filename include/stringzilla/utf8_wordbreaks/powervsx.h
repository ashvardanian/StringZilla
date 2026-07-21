/**
 *  @file   include/stringzilla/utf8_wordbreaks/powervsx.h
 *  @author Ash Vardanian
 *  @brief  Fully-vectorized UAX-29 Word_Break segmentation for POWER VSX (IBM POWER9, little-endian). The VSX twin
 *          of the AVX2 (Haswell), Ice Lake, NEON, and RVV kernels: no path scalar-walks codepoints or spills a
 *          vector to the stack to call the serial oracle.
 *
 *  Each 64-byte window lives as four `__vector unsigned char` quarters (the codepoint substrate twin of NEON's four
 *  `uint8x16_t` quarters); every per-codepoint BMP Word_Break property resolves through the shared page-compressed
 *  flat table via @ref sz_utf8_rune_flat_lookup_powervsx_ (the page LUT and the leaf both by a bounded scalar L1
 *  walk, VSX having no gather), and the Supplementary Plane through a nibble cascade whose 16-byte rows are gathered
 *  by one `vec_perm` each, bit-identical to `sz_rune_word_break_property` over the whole code space.
 *
 *  The classified window is lowered to the portable @ref sz_utf8_word_break_frame_t and handed to the SHARED
 *  `sz_utf8_word_break_decide_window_` rule engine, so WB1-WB16 (including the cross-window bridge shadow / RI
 *  parity / left-context carry / WB3c neighbour coupling) run once in portable `sz_u64_t` bit algebra. Boundary-mask
 *  compaction avoids `vpcompressb`/`pext` (and the banned scalar `ctz`/`popcount`): set lanes left-pack through the
 *  substrate `vec_perm` shuffle-LUT (@ref sz_utf8_compress_starts_powervsx_).
 *
 *  Little-endian only: the substrate movemasks and `vec_perm` indexing assume the little-endian element order (CI is
 *  ppc64le). The big-endian entry falls back to the serial reference, so no untested big-endian branch ships.
 */
#ifndef STRINGZILLA_UTF8_WORDBREAKS_POWERVSX_H_
#define STRINGZILLA_UTF8_WORDBREAKS_POWERVSX_H_

#include "stringzilla/types.h"
#include "stringzilla/utf8_wordbreaks/tables.h"
#include "stringzilla/utf8_wordbreaks/serial.h"
#include "stringzilla/utf8_runes/powervsx.h"

#ifdef __cplusplus
extern "C" {
#endif

#if SZ_USE_POWERVSX
#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("power9-vector"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("power9-vector")
#endif

#if !SZ_IS_BIG_ENDIAN_

#pragma region UAX 29 Word Boundaries forward kernel

#pragma region In register vectorized classifier

/** @brief  Build a per-quarter byte-boolean selector (0x00/0xFF) from the 16 lane bits of @p bits at offset @p shift,
 *          the VSX twin of @ref sz_utf8_word_break_byte_mask_from_bits_neon_ confined to one quarter. */
SZ_HELPER_INLINE __vector unsigned char sz_utf8_word_break_byte_mask_from_bits_powervsx_(sz_u64_t bits, int shift) {
    static unsigned char const bit_position_lanes[16] = {1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128};
    static unsigned char const lane_half[16] = {0, 0, 0, 0, 0, 0, 0, 0, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF};
    unsigned char const low_byte = (unsigned char)((bits >> shift) & 0xFF);
    unsigned char const high_byte = (unsigned char)((bits >> (shift + 8)) & 0xFF);
    __vector unsigned char const position_u8x16 = vec_xl(0, bit_position_lanes);
    __vector bool char const lane_half_u8x16 = (__vector bool char)vec_xl(0, lane_half);
    __vector unsigned char const byte_u8x16 = vec_sel(vec_splats(low_byte), vec_splats(high_byte), lane_half_u8x16);
    return (__vector unsigned char)vec_cmpeq(vec_and(byte_u8x16, position_u8x16), position_u8x16);
}

/** @brief  Word_Break class byte for sixteen BMP codepoints (per-lane high = cp>>8, low = cp&0xFF) from the flat
 *          page-compressed table via @ref sz_utf8_rune_flat_lookup_powervsx_, the VSX twin of
 *          @ref sz_utf8_word_break_bmp_class_neon_. Bit-exact with `sz_rune_word_break_property` over the whole BMP.
 *          Addresses ONE quarter; the caller iterates the four quarters. */
SZ_HELPER_AUTO __vector unsigned char sz_utf8_word_break_bmp_class_powervsx_(__vector unsigned char high_bytes_u8x16,
                                                                             __vector unsigned char low_bytes_u8x16) {
    return sz_utf8_rune_flat_lookup_powervsx_(sz_utf8_word_break_bmp_page_lut_, sz_utf8_word_break_flat_bmp_,
                                              (int)sz_utf8_word_break_flat_pages_k, high_bytes_u8x16, low_bytes_u8x16);
}

/** @brief  Word_Break class byte for sixteen ASTRAL codepoints over the 20-bit offset = cp - 0x10000 (5-nibble
 *          cascade), the VSX twin of @ref sz_utf8_word_break_astral_class_neon_. Per-lane bytes: @p plane_off =
 *          (offset>>16)&0xFF (low nibble meaningful), @p high = (offset>>8)&0xFF, @p low = offset&0xFF. Bit-exact with
 *          `sz_rune_word_break_property` over the Supplementary Planes. Addresses ONE quarter. */
SZ_HELPER_AUTO __vector unsigned char sz_utf8_word_break_astral_class_powervsx_( //
    __vector unsigned char plane_off_u8x16, __vector unsigned char high_u8x16, __vector unsigned char low_u8x16) {
    __vector unsigned char const low_nibble_mask_u8x16 = vec_splats((unsigned char)0x0F);
    __vector unsigned char const shift_four_u8x16 = vec_splats((unsigned char)4);
    __vector unsigned char const n4_u8x16 = vec_and(plane_off_u8x16, low_nibble_mask_u8x16);
    __vector unsigned char const n3_u8x16 = vec_and(vec_sr(high_u8x16, shift_four_u8x16), low_nibble_mask_u8x16);
    __vector unsigned char const stage1_index_u8x16 = vec_or(vec_sl(n4_u8x16, shift_four_u8x16), n3_u8x16);
    __vector unsigned char const page_u8x16 = sz_utf8_rune_lut256_powervsx_(sz_utf8_word_break_haswell_astral_stage1_,
                                                                            stage1_index_u8x16);
    __vector unsigned char const n2_u8x16 = vec_and(high_u8x16, low_nibble_mask_u8x16);
    __vector unsigned char const leaf2_u8x16 = sz_utf8_rune_cascade_stage_powervsx_(
        sz_utf8_word_break_haswell_astral_stage2_lo_, sz_utf8_word_break_haswell_astral_stage2_lo_count_k / 16,
        page_u8x16, n2_u8x16);
    __vector unsigned char const n1_u8x16 = vec_and(vec_sr(low_u8x16, shift_four_u8x16), low_nibble_mask_u8x16);
    __vector unsigned char const leaf_lo_u8x16 = sz_utf8_rune_cascade_stage_powervsx_(
        sz_utf8_word_break_haswell_astral_stage3_lo_, sz_utf8_word_break_haswell_astral_stage3_lo_count_k / 16,
        leaf2_u8x16, n1_u8x16);
    __vector unsigned char const leaf_hi_u8x16 = sz_utf8_rune_cascade_stage_powervsx_(
        sz_utf8_word_break_haswell_astral_stage3_hi_, sz_utf8_word_break_haswell_astral_stage3_hi_count_k / 16,
        leaf2_u8x16, n1_u8x16);
    __vector unsigned char const n0_u8x16 = vec_and(low_u8x16, low_nibble_mask_u8x16);
    __vector unsigned char const leaf_group_u8x16 = vec_or(
        vec_and(vec_sr(leaf_lo_u8x16, shift_four_u8x16), low_nibble_mask_u8x16),
        vec_sl(leaf_hi_u8x16, shift_four_u8x16));
    __vector unsigned char const leaf_low_nibble_u8x16 = vec_and(leaf_lo_u8x16, low_nibble_mask_u8x16);
    __vector unsigned char const stage4_lut_index_u8x16 = vec_or(vec_sl(leaf_low_nibble_u8x16, shift_four_u8x16),
                                                                 n0_u8x16);
    __vector unsigned char result_u8x16 = vec_splats((unsigned char)0);
    for (int group = 0; group < (int)sz_utf8_word_break_haswell_astral_leaf_groups_k; ++group) {
        __vector unsigned char const value_u8x16 = sz_utf8_rune_lut256_powervsx_(
            sz_utf8_word_break_haswell_astral_stage4_groups_ + group * 256, stage4_lut_index_u8x16);
        __vector bool char const here_u8x16 = vec_cmpeq(leaf_group_u8x16, vec_splats((unsigned char)group));
        result_u8x16 = vec_sel(result_u8x16, value_u8x16, here_u8x16);
    }
    return result_u8x16;
}

/** @brief  Word_Break class byte for sixteen ASCII codepoints (cp < 0x80) via the existing 128-entry property table,
 *          the VSX twin of @ref sz_utf8_word_break_ascii_class_neon_, read by a bounded scalar L1 walk (the window
 *          byte equals the codepoint on ASCII lanes). Addresses ONE quarter. */
SZ_HELPER_AUTO __vector unsigned char sz_utf8_word_break_ascii_class_powervsx_(__vector unsigned char bytes_u8x16) {
    sz_u128_vec_t bytes_vec, result_vec;
    bytes_vec.vsx_u8 = bytes_u8x16;
    for (int lane = 0; lane < 16; ++lane)
        result_vec.u8s[lane] = sz_utf8_word_break_property_ascii_[bytes_vec.u8s[lane] & 0x7F];
    return result_vec.vsx_u8;
}

/** @brief  Start-compacting BMP classify, the VSX twin of @ref sz_utf8_word_break_bmp_compact_neon_: the BMP class is
 *          consumed only on 2-/3-byte codepoint-START lanes, so this gathers their `(high, low)` bytes into a dense
 *          buffer (left-packed through the substrate `vec_perm` shuffle-LUT, no scalar `ctz`), runs the flat lookup
 *          only over the populated quarters, then scatters the dense class bytes back to their original byte lanes in
 *          @p bmp_out (zeroed elsewhere). Bit-identical to four full @ref sz_utf8_word_break_bmp_class_powervsx_
 *          quarters on every BMP-start lane. */
SZ_HELPER_AUTO void sz_utf8_word_break_bmp_compact_powervsx_(sz_u64_t bmp_starts, __vector unsigned char const *high,
                                                             __vector unsigned char const *low,
                                                             __vector unsigned char *bmp_out) {
    sz_u512_vec_t high_bytes_vec, low_bytes_vec;
    for (int quarter = 0; quarter < 4; ++quarter) {
        vec_xst(high[quarter], 0, high_bytes_vec.u8s + quarter * 16);
        vec_xst(low[quarter], 0, low_bytes_vec.u8s + quarter * 16);
    }

    sz_u8_t indices[64];
    sz_size_t dense_count = 0;
    for (int reg = 0; reg < 4; ++reg)
        dense_count = sz_utf8_compress_starts_powervsx_((sz_u32_t)((bmp_starts >> (reg * 16)) & 0xFFFFu), reg * 16,
                                                        indices, dense_count);

    sz_u8_t dense_high[64], dense_low[64];
    for (int lane = 0; lane < 64; ++lane) dense_high[lane] = 0, dense_low[lane] = 0;
    for (sz_size_t dense = 0; dense < dense_count; ++dense) {
        dense_high[dense] = high_bytes_vec.u8s[indices[dense]];
        dense_low[dense] = low_bytes_vec.u8s[indices[dense]];
    }

    sz_u8_t class_bytes[64];
    int const populated_quarters = (int)((dense_count + 15) / 16);
    for (int quarter = 0; quarter < populated_quarters; ++quarter) {
        __vector unsigned char const dense_class_u8x16 = sz_utf8_word_break_bmp_class_powervsx_(
            vec_xl(0, dense_high + quarter * 16), vec_xl(0, dense_low + quarter * 16));
        vec_xst(dense_class_u8x16, 0, class_bytes + quarter * 16);
    }

    sz_u8_t scatter[64];
    for (int lane = 0; lane < 64; ++lane) scatter[lane] = 0;
    for (sz_size_t dense = 0; dense < dense_count; ++dense) scatter[indices[dense]] = class_bytes[dense];
    for (int quarter = 0; quarter < 4; ++quarter) bmp_out[quarter] = vec_xl(0, scatter + quarter * 16);
}

/** @brief  Per-window byte-lane classification (VSX): the Word_Break class byte per lane as four `__vector unsigned
 *          char` quarters, valid only on codepoint-start lanes (the engine reads classes only at starts). The VSX
 *          twin of @ref sz_utf8_word_break_classify_window_neon_, bit-identical on every start lane. ASCII through
 *          the property table, BMP through the flat lookup, 4-byte leads through the astral cascade with the
 *          codepoint high/low/plane reconstructed from the forward neighbours. */
SZ_HELPER_AUTO void sz_utf8_word_break_classify_window_powervsx_( //
    sz_utf8_rune_window_powervsx_t window, __vector unsigned char *classes) {
    __vector unsigned char const *raw = window.window;
    sz_u64_t const ascii_starts = window.codepoint_starts & ~window.two_byte_starts & ~window.three_byte_starts &
                                  ~window.four_byte_starts;

    __vector unsigned char next1_u8x16[4], next2_u8x16[4], next3_u8x16[4];
    sz_utf8_forward_neighbours_powervsx_(raw, next1_u8x16, next2_u8x16, next3_u8x16);

    __vector unsigned char const low_two_bits_u8x16 = vec_splats((unsigned char)0x03),
                                 low_three_bits_u8x16 = vec_splats((unsigned char)0x07),
                                 low_nibble_u8x16 = vec_splats((unsigned char)0x0F),
                                 plane_high_bits_u8x16 = vec_splats((unsigned char)0x1C),
                                 low_six_bits_u8x16 = vec_splats((unsigned char)0x3F),
                                 high_two_bits_u8x16 = vec_splats((unsigned char)0xC0),
                                 high_nibble_u8x16 = vec_splats((unsigned char)0xF0);
    __vector unsigned char const shift_two_u8x16 = vec_splats((unsigned char)2),
                                 shift_four_u8x16 = vec_splats((unsigned char)4),
                                 shift_six_u8x16 = vec_splats((unsigned char)6);

    // BMP class via the flat lookup over the compacted 2-/3-byte START lanes; ASCII / 4-byte / continuation lanes are
    // don't-cares (overwritten or unread below), so the dense walk leaves them at zero.
    __vector unsigned char bmp_out_u8x16[4] = {vec_splats((unsigned char)0), vec_splats((unsigned char)0),
                                               vec_splats((unsigned char)0), vec_splats((unsigned char)0)};
    sz_u64_t const bmp_starts = window.two_byte_starts | window.three_byte_starts;
    if (bmp_starts) sz_utf8_word_break_bmp_compact_powervsx_(bmp_starts, window.high, window.low, bmp_out_u8x16);

    for (int quarter = 0; quarter < 4; ++quarter) {
        __vector unsigned char const raw_q_u8x16 = raw[quarter];
        int const lane_base = quarter * 16;

        __vector unsigned char out_q_u8x16 = bmp_out_u8x16[quarter];

        // ASCII lanes: read the 128-entry property table directly off the raw byte.
        __vector unsigned char const ascii_select_u8x16 = sz_utf8_word_break_byte_mask_from_bits_powervsx_(ascii_starts,
                                                                                                           lane_base);
        out_q_u8x16 = vec_sel(out_q_u8x16, sz_utf8_word_break_ascii_class_powervsx_(raw_q_u8x16),
                              (__vector bool char)ascii_select_u8x16);

        // 4-byte (astral) lanes: reconstruct the codepoint from the lead + three forward neighbours, then the astral
        // cascade addressed by offset = codepoint - 0x10000 (the offset's plane nibble is `plane - 1`).
        if (window.four_byte_starts) {
            __vector unsigned char const n1_u8x16 = next1_u8x16[quarter], n2_u8x16 = next2_u8x16[quarter],
                                         n3_u8x16 = next3_u8x16[quarter];
            __vector unsigned char const four_select_u8x16 = sz_utf8_word_break_byte_mask_from_bits_powervsx_(
                window.four_byte_starts, lane_base);
            __vector unsigned char const plane_u8x16 = vec_or(
                vec_and(vec_sl(vec_and(raw_q_u8x16, low_three_bits_u8x16), shift_two_u8x16), plane_high_bits_u8x16),
                sz_utf8_srl8_powervsx_(n1_u8x16, 4, 0x03));
            __vector unsigned char const high_four_u8x16 = vec_or(
                vec_and(vec_sl(vec_and(n1_u8x16, low_nibble_u8x16), shift_four_u8x16), high_nibble_u8x16),
                sz_utf8_srl8_powervsx_(n2_u8x16, 2, 0x0F));
            __vector unsigned char const low_four_u8x16 = vec_or(
                vec_and(vec_sl(vec_and(n2_u8x16, low_two_bits_u8x16), shift_six_u8x16), high_two_bits_u8x16),
                vec_and(n3_u8x16, low_six_bits_u8x16));
            __vector unsigned char const plane_off_u8x16 = vec_sub(plane_u8x16, vec_splats((unsigned char)1));
            out_q_u8x16 = vec_sel(
                out_q_u8x16,
                sz_utf8_word_break_astral_class_powervsx_(plane_off_u8x16, high_four_u8x16, low_four_u8x16),
                (__vector bool char)four_select_u8x16);
        }
        classes[quarter] = out_q_u8x16;
    }
}

#pragma endregion In register vectorized classifier

#pragma region Mask algebra extractor

/** @brief  A 64-bit "class byte == @p value" lane mask over the four class quarters (four `vec_cmpeq` -> combine). */
SZ_HELPER_INLINE sz_u64_t sz_utf8_word_break_class_mask_powervsx_(__vector unsigned char const *classes,
                                                                  sz_u8_t value) {
    __vector unsigned char const value_u8x16 = vec_splats(value);
    __vector unsigned char equal_u8x16[4];
    for (int quarter = 0; quarter < 4; ++quarter)
        equal_u8x16[quarter] = (__vector unsigned char)vec_cmpeq(classes[quarter], value_u8x16);
    return sz_utf8_mask_combine_powervsx_(equal_u8x16);
}

/** @brief  A 64-bit "raw window byte == @p value" lane mask over the four window quarters. */
SZ_HELPER_INLINE sz_u64_t sz_utf8_word_break_byte_equal_powervsx_(__vector unsigned char const *quarters,
                                                                  sz_u8_t value) {
    __vector unsigned char const value_u8x16 = vec_splats(value);
    __vector unsigned char equal_u8x16[4];
    for (int quarter = 0; quarter < 4; ++quarter)
        equal_u8x16[quarter] = (__vector unsigned char)vec_cmpeq(quarters[quarter], value_u8x16);
    return sz_utf8_mask_combine_powervsx_(equal_u8x16);
}

/** @brief  A 64-bit "raw window byte >= @p bound" (unsigned) lane mask over the four window quarters (`vec_cmpge`). */
SZ_HELPER_INLINE sz_u64_t sz_utf8_word_break_byte_ge_powervsx_(__vector unsigned char const *quarters, sz_u8_t bound) {
    __vector unsigned char const bound_u8x16 = vec_splats(bound);
    __vector unsigned char ge_u8x16[4];
    for (int quarter = 0; quarter < 4; ++quarter)
        ge_u8x16[quarter] = (__vector unsigned char)vec_cmpge(quarters[quarter], bound_u8x16);
    return sz_utf8_mask_combine_powervsx_(ge_u8x16);
}

/** @brief  Per-quarter "(high,low) 16-bit value in `[lo, hi]`" membership for one range, the VSX unsigned 16-bit
 *          window-compare building block of @ref sz_utf8_word_break_range16_mask_powervsx_. `vec_cmpge`/`vec_cmplt`
 *          are native, so no `max_epu8` emulation is needed. */
SZ_HELPER_INLINE __vector unsigned char sz_utf8_word_break_range16_one_powervsx_(__vector unsigned char high_u8x16,
                                                                                 __vector unsigned char low_u8x16,
                                                                                 sz_u16_t lo, sz_u16_t hi) {
    __vector unsigned char const lo_high_u8x16 = vec_splats((unsigned char)(lo >> 8)),
                                 lo_low_u8x16 = vec_splats((unsigned char)(lo & 0xFF));
    __vector unsigned char const hi_high_u8x16 = vec_splats((unsigned char)(hi >> 8)),
                                 hi_low_u8x16 = vec_splats((unsigned char)(hi & 0xFF));
    __vector unsigned char const high_eq_lo_u8x16 = (__vector unsigned char)vec_cmpeq(high_u8x16, lo_high_u8x16);
    __vector unsigned char const high_eq_hi_u8x16 = (__vector unsigned char)vec_cmpeq(high_u8x16, hi_high_u8x16);
    // not_below: high > lo_high, or (high == lo_high and low >= lo_low).
    __vector unsigned char const not_below_u8x16 = vec_or(
        vec_andc((__vector unsigned char)vec_cmpge(high_u8x16, lo_high_u8x16), high_eq_lo_u8x16),
        vec_and(high_eq_lo_u8x16, (__vector unsigned char)vec_cmpge(low_u8x16, lo_low_u8x16)));
    // not_above: high < hi_high, or (high == hi_high and low <= hi_low).
    __vector unsigned char const not_above_u8x16 = vec_or(
        vec_andc((__vector unsigned char)vec_cmplt(high_u8x16, hi_high_u8x16), high_eq_hi_u8x16),
        vec_and(high_eq_hi_u8x16, (__vector unsigned char)vec_cmpge(hi_low_u8x16, low_u8x16)));
    return vec_and(not_below_u8x16, not_above_u8x16);
}

/** @brief  A 64-bit "(high,low) 16-bit value in any sorted `[lo, hi]` range" lane mask over the four window quarters,
 *          the VSX twin of @ref sz_utf8_word_break_range16_mask_neon_ (WSegSpace / Extended_Pictographic). */
SZ_HELPER_AUTO sz_u64_t sz_utf8_word_break_range16_mask_powervsx_( //
    __vector unsigned char const *high, __vector unsigned char const *low, sz_u16_t const *lo_table,
    sz_u16_t const *hi_table, int count) {
    __vector unsigned char hit_u8x16[4] = {vec_splats((unsigned char)0), vec_splats((unsigned char)0),
                                           vec_splats((unsigned char)0), vec_splats((unsigned char)0)};
    for (int range = 0; range < count; ++range)
        for (int quarter = 0; quarter < 4; ++quarter)
            hit_u8x16[quarter] = vec_or(hit_u8x16[quarter],
                                        sz_utf8_word_break_range16_one_powervsx_(high[quarter], low[quarter],
                                                                                 lo_table[range], hi_table[range]));
    return sz_utf8_mask_combine_powervsx_(hit_u8x16);
}

/**
 *  @brief  Per-ISA extractor: lower one classified 64-byte window to the portable @ref sz_utf8_word_break_frame_t, the
 *          VSX twin of @ref sz_utf8_word_break_build_frame_neon_. Applies the truncated-edge U+FFFD reclassify to the
 *          class quarters, materializes every per-class lane mask + the raw-byte membership masks, the
 *          Extended_Pictographic mask (BMP + SMP range scan), and the per-lane class byte array.
 */
SZ_HELPER_AUTO sz_utf8_word_break_frame_t sz_utf8_word_break_build_frame_powervsx_(
    sz_utf8_rune_window_powervsx_t window, __vector unsigned char *classes, sz_u64_t start_bytes_all,
    sz_u64_t length_two, sz_u64_t length_three, sz_u64_t length_four, int want_pictographic) {

    sz_size_t const loaded = window.loaded;
    sz_u64_t const valid = sz_u64_mask_until_serial_(loaded);
    sz_u64_t const start_bytes = start_bytes_all & valid;
    __vector unsigned char const *raw = window.window;

    // Truncated-edge U+FFFD reclassify (force the class to Other on a lead whose declared span runs past `loaded`).
    sz_u64_t const lead_two = length_two & start_bytes;
    sz_u64_t const lead_three = length_three & start_bytes;
    sz_u64_t const lead_four = length_four & start_bytes;
    sz_u64_t const truncated_raw = ((lead_two & ~sz_u64_mask_until_serial_(loaded > 1 ? loaded - 1 : 0)) |
                                    (lead_three & ~sz_u64_mask_until_serial_(loaded > 2 ? loaded - 2 : 0)) |
                                    (lead_four & ~sz_u64_mask_until_serial_(loaded > 3 ? loaded - 3 : 0))) &
                                   valid;
    if (truncated_raw) {
        __vector unsigned char const other_u8x16 = vec_splats((unsigned char)sz_utf8_word_break_other_k);
        for (int quarter = 0; quarter < 4; ++quarter) {
            __vector unsigned char const select_u8x16 = sz_utf8_word_break_byte_mask_from_bits_powervsx_(truncated_raw,
                                                                                                         quarter * 16);
            classes[quarter] = vec_sel(classes[quarter], other_u8x16, (__vector bool char)select_u8x16);
        }
    }

    sz_utf8_word_break_frame_t frame;
    frame.class_aletter = sz_utf8_word_break_class_mask_powervsx_(classes, sz_utf8_word_break_aletter_k);
    frame.class_hebrew = sz_utf8_word_break_class_mask_powervsx_(classes, sz_utf8_word_break_hebrew_letter_k);
    frame.class_numeric = sz_utf8_word_break_class_mask_powervsx_(classes, sz_utf8_word_break_numeric_k);
    frame.class_katakana = sz_utf8_word_break_class_mask_powervsx_(classes, sz_utf8_word_break_katakana_k);
    frame.class_extendnumlet = sz_utf8_word_break_class_mask_powervsx_(classes, sz_utf8_word_break_extendnumlet_k);
    frame.class_extend = sz_utf8_word_break_class_mask_powervsx_(classes, sz_utf8_word_break_extend_k);
    frame.class_zwj = sz_utf8_word_break_class_mask_powervsx_(classes, sz_utf8_word_break_zwj_k);
    frame.class_format = sz_utf8_word_break_class_mask_powervsx_(classes, sz_utf8_word_break_format_k);
    frame.class_midletter = sz_utf8_word_break_class_mask_powervsx_(classes, sz_utf8_word_break_midletter_k);
    frame.class_midnum = sz_utf8_word_break_class_mask_powervsx_(classes, sz_utf8_word_break_midnum_k);
    frame.class_mid_quotes = sz_utf8_word_break_class_mask_powervsx_(classes, sz_utf8_word_break_mid_quotes_k);
    frame.class_cr = sz_utf8_word_break_class_mask_powervsx_(classes, sz_utf8_word_break_cr_k);
    frame.class_lf = sz_utf8_word_break_class_mask_powervsx_(classes, sz_utf8_word_break_lf_k);
    frame.class_newline = sz_utf8_word_break_class_mask_powervsx_(classes, sz_utf8_word_break_newline_k);
    frame.class_regional = sz_utf8_word_break_class_mask_powervsx_(classes, sz_utf8_word_break_regional_ind_k);

    sz_u64_t const non_ascii_lanes = sz_utf8_word_break_byte_ge_powervsx_(raw, 0x80) & valid;
    frame.non_ascii_lanes = non_ascii_lanes;
    frame.double_quote_byte = sz_utf8_word_break_byte_equal_powervsx_(raw, 0x22) & valid;
    frame.single_quote_byte = sz_utf8_word_break_byte_equal_powervsx_(raw, 0x27) & valid;

    // WB3d WSegSpace raw membership: the ASCII U+0020 byte compare OR the multibyte (high,low) range scan.
    sz_u64_t wseg_multibyte = 0ull;
    if (non_ascii_lanes)
        wseg_multibyte = sz_utf8_word_break_range16_mask_powervsx_(window.high, window.low, sz_utf8_word_break_wseg_lo_,
                                                                   sz_utf8_word_break_wseg_hi_,
                                                                   sz_utf8_word_break_wseg_count_k) &
                         non_ascii_lanes;
    frame.wseg = (wseg_multibyte | (sz_utf8_word_break_byte_equal_powervsx_(raw, 0x20) & valid));

    // WB3c Extended_Pictographic raw membership (BMP range scan on non-4-byte lanes, SMP range scan on plane-one
    // 4-byte lanes). Rare-class gated on `want_pictographic` (an in-window ZWJ or the carried `prev_ends_in_zwj`), so
    // the multi-range scan is skipped on the common no-ZWJ window. The engine applies the final gating.
    frame.pictographic = 0ull;
    sz_u64_t const four_byte = window.four_byte_starts & valid;
    if (want_pictographic) {
        __vector unsigned char next1_u8x16[4], next2_u8x16[4], next3_u8x16[4];
        sz_utf8_forward_neighbours_powervsx_(raw, next1_u8x16, next2_u8x16, next3_u8x16);
        __vector unsigned char const low_two_bits_u8x16 = vec_splats((unsigned char)0x03),
                                     low_three_bits_u8x16 = vec_splats((unsigned char)0x07),
                                     low_nibble_u8x16 = vec_splats((unsigned char)0x0F),
                                     plane_high_bits_u8x16 = vec_splats((unsigned char)0x1C),
                                     low_six_bits_u8x16 = vec_splats((unsigned char)0x3F),
                                     high_two_bits_u8x16 = vec_splats((unsigned char)0xC0),
                                     high_nibble_u8x16 = vec_splats((unsigned char)0xF0);
        __vector unsigned char const shift_two_u8x16 = vec_splats((unsigned char)2),
                                     shift_four_u8x16 = vec_splats((unsigned char)4),
                                     shift_six_u8x16 = vec_splats((unsigned char)6);
        __vector unsigned char plane_q_u8x16[4], smp_high_u8x16[4], smp_low_u8x16[4];
        for (int quarter = 0; quarter < 4; ++quarter) {
            __vector unsigned char const raw_q_u8x16 = raw[quarter], n1_u8x16 = next1_u8x16[quarter],
                                         n2_u8x16 = next2_u8x16[quarter], n3_u8x16 = next3_u8x16[quarter];
            plane_q_u8x16[quarter] = vec_or(
                vec_and(vec_sl(vec_and(raw_q_u8x16, low_three_bits_u8x16), shift_two_u8x16), plane_high_bits_u8x16),
                sz_utf8_srl8_powervsx_(n1_u8x16, 4, 0x03));
            smp_high_u8x16[quarter] = vec_or(
                vec_and(vec_sl(vec_and(n1_u8x16, low_nibble_u8x16), shift_four_u8x16), high_nibble_u8x16),
                sz_utf8_srl8_powervsx_(n2_u8x16, 2, 0x0F));
            smp_low_u8x16[quarter] = vec_or(
                vec_and(vec_sl(vec_and(n2_u8x16, low_two_bits_u8x16), shift_six_u8x16), high_two_bits_u8x16),
                vec_and(n3_u8x16, low_six_bits_u8x16));
        }
        __vector unsigned char const one_u8x16 = vec_splats((unsigned char)1);
        __vector unsigned char plane_one_bool_u8x16[4];
        for (int quarter = 0; quarter < 4; ++quarter)
            plane_one_bool_u8x16[quarter] = (__vector unsigned char)vec_cmpeq(plane_q_u8x16[quarter], one_u8x16);
        sz_u64_t const plane_one = sz_utf8_mask_combine_powervsx_(plane_one_bool_u8x16);
        sz_u64_t const pictographic_bmp = sz_utf8_word_break_range16_mask_powervsx_(
            window.high, window.low, sz_utf8_word_break_pict_bmp_lo_, sz_utf8_word_break_pict_bmp_hi_,
            sz_utf8_word_break_pict_bmp_count_k);
        sz_u64_t const pictographic_smp = sz_utf8_word_break_range16_mask_powervsx_(
            smp_high_u8x16, smp_low_u8x16, sz_utf8_word_break_pict_smp_lo_, sz_utf8_word_break_pict_smp_hi_,
            sz_utf8_word_break_pict_smp_count_k);
        frame.pictographic = (pictographic_bmp & non_ascii_lanes & ~four_byte) |
                             (pictographic_smp & four_byte & plane_one);
    }

    for (int quarter = 0; quarter < 4; ++quarter) vec_xst(classes[quarter], 0, frame.classes_byte + quarter * 16);
    return frame;
}

#pragma endregion Mask algebra extractor

#pragma region Codepoint partition

/** @brief  Resolve one window into the maximal-subpart partition, the VSX twin of
 *          @ref sz_utf8_word_break_partition_neon_: compute the per-ISA `sz_u64_t` masks and delegate to the portable
 *          @ref sz_utf8_word_break_partition_from_masks_. */
SZ_HELPER_AUTO sz_utf8_word_break_partition_t sz_utf8_word_break_partition_powervsx_( //
    sz_utf8_rune_window_powervsx_t window, sz_u64_t valid, int at_end_of_text) {
    __vector unsigned char const *raw = window.window;
    sz_u64_t const real_continuation = window.continuation & valid;
    // Declared length follows the serial high-nibble rule: 0xC/0xD -> 2, 0xE -> 3, 0xF -> 4. The strict
    // `two`/`three_byte_starts` masks already match 0xC0-0xDF and 0xE0-0xEF; only `length_four` needs widening to fold
    // the ill-formed leads 0xF8-0xFF so they collapse to U+FFFD like serial/Haswell instead of leaking as a class.
    sz_u64_t const length_two = window.two_byte_starts & valid;
    sz_u64_t const length_three = window.three_byte_starts & valid;
    sz_u64_t const length_four = (window.four_byte_starts | sz_utf8_word_break_byte_ge_powervsx_(raw, 0xF8)) & valid;
    sz_u64_t const length_ge_two = length_two | length_three | length_four;
    sz_u64_t bad_second_byte = 0ull;
    if (length_ge_two) {
        __vector unsigned char next1_u8x16[4], next2_u8x16[4], next3_u8x16[4];
        sz_utf8_forward_neighbours_powervsx_(raw, next1_u8x16, next2_u8x16, next3_u8x16);
        sz_u64_t const next1_at_least_a0 = sz_utf8_word_break_byte_ge_powervsx_(next1_u8x16, 0xA0);
        sz_u64_t const next1_at_least_90 = sz_utf8_word_break_byte_ge_powervsx_(next1_u8x16, 0x90);
        sz_u64_t const lead_c0_c1 = (sz_utf8_word_break_byte_equal_powervsx_(raw, 0xC0) |
                                     sz_utf8_word_break_byte_equal_powervsx_(raw, 0xC1)) &
                                    valid;
        sz_u64_t const lead_e0 = sz_utf8_word_break_byte_equal_powervsx_(raw, 0xE0) & valid;
        sz_u64_t const lead_ed = sz_utf8_word_break_byte_equal_powervsx_(raw, 0xED) & valid;
        sz_u64_t const lead_f0 = sz_utf8_word_break_byte_equal_powervsx_(raw, 0xF0) & valid;
        sz_u64_t const lead_f4 = sz_utf8_word_break_byte_equal_powervsx_(raw, 0xF4) & valid;
        sz_u64_t const lead_f5_or_more = sz_utf8_word_break_byte_ge_powervsx_(raw, 0xF5) & valid;
        bad_second_byte = lead_c0_c1 | (lead_e0 & ~next1_at_least_a0) | (lead_ed & next1_at_least_a0) |
                          (lead_f0 & ~next1_at_least_90) | (lead_f4 & next1_at_least_90) | lead_f5_or_more;
    }
    return sz_utf8_word_break_partition_from_masks_(real_continuation, length_two, length_three, length_four,
                                                    bad_second_byte, valid, at_end_of_text);
}

#pragma endregion Codepoint partition

#pragma region Forward driver

/**
 *  @brief  Forward UAX-29 word segmentation over `[0, length)` (POWER VSX, little-endian): the overlap-free advancing
 *          driver, mirroring @ref sz_utf8_wordbreaks_neon over the VSX window/classify/partition/decide/drain leaves.
 *          Bit-exact with `sz_utf8_wordbreaks_serial` and every other windowed backend.
 */
SZ_API_COMPTIME sz_size_t sz_utf8_wordbreaks_powervsx( //
    sz_cptr_t text, sz_size_t length,                  //
    sz_size_t *word_starts, sz_size_t *word_lengths,   //
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
        sz_utf8_rune_window_powervsx_t const window = sz_utf8_rune_decode_window_powervsx_(text_u8 + position,
                                                                                           length - position);
        sz_size_t const loaded = window.loaded;
        sz_u64_t const valid = sz_u64_mask_until_serial_(loaded);

        __vector unsigned char classes[4];
        sz_utf8_word_break_classify_window_powervsx_(window, classes);

        sz_utf8_word_break_partition_t const partition = sz_utf8_word_break_partition_powervsx_(
            window, valid, position + loaded >= length);
        sz_u64_t const start_bytes_all = partition.start_bytes;
        sz_u64_t const continuation_all = partition.continuation;
        sz_u64_t const forced_other = partition.forced_other;
        sz_u64_t const length_two = partition.length_two;
        sz_u64_t const length_three = partition.length_three;
        sz_u64_t const length_four = partition.length_four;
        if (forced_other) {
            __vector unsigned char const other_u8x16 = vec_splats((unsigned char)sz_utf8_word_break_other_k);
            for (int quarter = 0; quarter < 4; ++quarter) {
                __vector unsigned char const select_u8x16 = sz_utf8_word_break_byte_mask_from_bits_powervsx_(
                    forced_other, quarter * 16);
                classes[quarter] = vec_sel(classes[quarter], other_u8x16, (__vector bool char)select_u8x16);
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
            sz_u64_t const straddle = ((two & ~sz_u64_mask_until_serial_(loaded > 1 ? loaded - 1 : 0)) |
                                       (three & ~sz_u64_mask_until_serial_(loaded > 2 ? loaded - 2 : 0)) |
                                       (four & ~sz_u64_mask_until_serial_(loaded > 3 ? loaded - 3 : 0))) &
                                      valid;
            // First straddling lead: isolate the lowest set bit and read its index via `clz` (POWER has no `ctz`
            // primitive and the pre-commit hook bans the scalar `ctz`; `63 - clz(low_bit)` is the equivalent).
            sz_size_t limit = straddle ? (sz_size_t)(63 - sz_u64_clz(straddle & (~straddle + 1))) : loaded;
            if ((text_u8[position + loaded] & 0xC0) == 0x80) {
                sz_size_t const last_lead = (sz_size_t)(63 - sz_u64_clz(start_bytes_all));
                sz_size_t const last_lead_length = sz_utf8_lead_length_(text_u8[position + last_lead]);
                if (last_lead + last_lead_length > loaded && last_lead < limit) limit = last_lead;
            }
            if (limit > 0) complete_limit = limit;
        }

        int const want_pictographic = sz_utf8_word_break_class_mask_powervsx_(classes, sz_utf8_word_break_zwj_k) != 0 ||
                                      carry.prev_ends_in_zwj;
        sz_utf8_word_break_frame_t const frame = sz_utf8_word_break_build_frame_powervsx_(
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

        words = sz_utf8_rune_drain_forward_powervsx_(boundary_lanes, position, word_starts, word_lengths, words,
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

#else // SZ_IS_BIG_ENDIAN_

/*  The vectorized substrate (movemasks, `vec_perm` neighbours) is validated only for the little-endian element order
 *  shipped on ppc64le CI; the big-endian entry defers to the serial reference rather than ship an untested branch. */
SZ_API_COMPTIME sz_size_t sz_utf8_wordbreaks_powervsx( //
    sz_cptr_t text, sz_size_t length,                  //
    sz_size_t *word_starts, sz_size_t *word_lengths,   //
    sz_size_t words_capacity, sz_size_t *bytes_consumed) {
    return sz_utf8_wordbreaks_serial(text, length, word_starts, word_lengths, words_capacity, bytes_consumed);
}

#endif // !SZ_IS_BIG_ENDIAN_

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#endif // SZ_USE_POWERVSX

#ifdef __cplusplus
}
#endif

#endif // STRINGZILLA_UTF8_WORDBREAKS_POWERVSX_H_
