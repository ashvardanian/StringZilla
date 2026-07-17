/**
 *  @file   include/stringzilla/utf8_wordbreaks/lasx.h
 *  @author Ash Vardanian
 *  @brief  Fully-vectorized UAX-29 Word_Break segmentation for LoongArch LASX. The LASX twin of the AVX2 (Haswell),
 *          Ice Lake, NEON, and RVV kernels: no path scalar-walks codepoints or spills a vector to the stack to call
 *          the serial oracle.
 *
 *  LASX is 256-bit, so each 64-byte window lives as two `__m256i` halves (`*_low_u8x32` = lanes [0,32), `*_high_u8x32` = lanes
 *  [32,64)) - the Haswell shape, NOT the NEON quarters. Every per-codepoint BMP Word_Break property resolves through
 *  the shared page-compressed flat table via @ref sz_utf8_rune_flat_lookup_lasx_ (the page LUT and leaf by a bounded
 *  scalar L1 walk, since LASX has no gather), and the Supplementary Plane through a mixed `xvshuf.b` / scalar-walk
 *  nibble cascade, bit-identical to `sz_rune_word_break_property` over the whole code space.
 *
 *  The classified window is lowered to the portable @ref sz_utf8_word_break_frame_t and handed to the SHARED
 *  `sz_utf8_word_break_decide_window_` rule engine, so WB1-WB16 (including the cross-window bridge shadow / RI parity /
 *  left-context carry / WB3c next1/2/3 neighbour coupling) run once in portable `sz_u64_t` bit algebra. Malformed
 *  input is handled in-vector: ill-formed leads and strays become `forced_other` in the portable partition resolver,
 *  truncated-edge leads are reclassified in the frame builder - no well-formedness prepass, no serial routing. The
 *  file carries no clang target pragma; it relies on the `-mlasx` compile flag.
 */
#ifndef STRINGZILLA_UTF8_WORDBREAKS_LASX_H_
#define STRINGZILLA_UTF8_WORDBREAKS_LASX_H_

#include "stringzilla/types.h"
#include "stringzilla/utf8_wordbreaks/tables.h"
#include "stringzilla/utf8_wordbreaks/serial.h"
#include "stringzilla/utf8_runes/lasx.h"

#ifdef __cplusplus
extern "C" {
#endif

#if SZ_USE_LASX

#pragma region UAX 29 Word Boundaries forward kernel

#pragma region In register vectorized classifier

/** @brief  Word_Break class byte for thirty-two BMP codepoints (per-lane high = cp>>8, low = cp&0xFF) from the flat
 *          page-compressed table via @ref sz_utf8_rune_flat_lookup_lasx_, the LASX twin of
 *          @ref sz_utf8_word_break_bmp_class_haswell_. Bit-exact with `sz_rune_word_break_property` over the BMP. */
SZ_HELPER_AUTO __m256i sz_utf8_word_break_bmp_class_lasx_(__m256i high_bytes_u8x32, __m256i low_bytes_u8x32) {
    return sz_utf8_rune_flat_lookup_lasx_(sz_utf8_word_break_bmp_page_lut_, sz_utf8_word_break_flat_bmp_,
                                          (int)sz_utf8_word_break_flat_pages_k, high_bytes_u8x32, low_bytes_u8x32);
}

/** @brief  Word_Break class byte for thirty-two ASTRAL codepoints over the 20-bit offset = cp - 0x10000 (5-nibble
 *          cascade), the LASX twin of @ref sz_utf8_word_break_astral_class_haswell_. The 256-entry stage-1 and
 *          stage-4 tables resolve by a bounded scalar walk; the 16-byte stage-2/stage-3 rows by `xvshuf.b` over a
 *          double-broadcast row. Bit-exact with `sz_rune_word_break_property` over the Supplementary Planes. */
SZ_HELPER_AUTO __m256i sz_utf8_word_break_astral_class_lasx_(__m256i plane_offset_u8x32, __m256i high_byte_u8x32,
                                                             __m256i low_byte_u8x32) {
    __m256i const low_nibble_mask_u8x32 = __lasx_xvreplgr2vr_b(0x0F);
    __m256i const nibble_4_u8x32 = __lasx_xvand_v(plane_offset_u8x32, low_nibble_mask_u8x32);
    __m256i const nibble_3_u8x32 = sz_utf8_high_nibble_lasx_(high_byte_u8x32);
    __m256i const stage1_index_u8x32 = __lasx_xvor_v(__lasx_xvslli_h(nibble_4_u8x32, 4), nibble_3_u8x32);
    __m256i const page_u8x32 = sz_utf8_rune_lut256_scalar_lasx_(sz_utf8_word_break_haswell_astral_stage1_,
                                                                stage1_index_u8x32);
    __m256i const nibble_2_u8x32 = __lasx_xvand_v(high_byte_u8x32, low_nibble_mask_u8x32);
    __m256i const stage2_leaf_u8x32 = sz_utf8_rune_cascade_stage_lasx_(
        sz_utf8_word_break_haswell_astral_stage2_lo_, sz_utf8_word_break_haswell_astral_stage2_lo_count_k / 16,
        page_u8x32, nibble_2_u8x32);
    __m256i const nibble_1_u8x32 = sz_utf8_high_nibble_lasx_(low_byte_u8x32);
    __m256i const stage3_low_leaf_u8x32 = sz_utf8_rune_cascade_stage_lasx_(
        sz_utf8_word_break_haswell_astral_stage3_lo_, sz_utf8_word_break_haswell_astral_stage3_lo_count_k / 16,
        stage2_leaf_u8x32, nibble_1_u8x32);
    __m256i const stage3_high_leaf_u8x32 = sz_utf8_rune_cascade_stage_lasx_(
        sz_utf8_word_break_haswell_astral_stage3_hi_, sz_utf8_word_break_haswell_astral_stage3_hi_count_k / 16,
        stage2_leaf_u8x32, nibble_1_u8x32);
    __m256i const nibble_0_u8x32 = __lasx_xvand_v(low_byte_u8x32, low_nibble_mask_u8x32);
    __m256i const leaf_group_u8x32 = __lasx_xvor_v(
        __lasx_xvand_v(sz_utf8_high_nibble_lasx_(stage3_low_leaf_u8x32), low_nibble_mask_u8x32),
        __lasx_xvslli_h(stage3_high_leaf_u8x32, 4));
    __m256i const stage4_lut_index_u8x32 = __lasx_xvor_v(
        __lasx_xvslli_h(__lasx_xvand_v(stage3_low_leaf_u8x32, low_nibble_mask_u8x32), 4), nibble_0_u8x32);

    // Stage 4: a grouped 256-entry scalar walk. `result[lane] = stage4_groups[group*256 + index]` when the leaf group
    // is in range, else 0 - mirroring the AVX2 blend loop that leaves out-of-range groups zeroed.
    sz_u256_vec_t leaf_group_vec, stage4_index_vec, result_vec;
    leaf_group_vec.lasx = leaf_group_u8x32;
    stage4_index_vec.lasx = stage4_lut_index_u8x32;
    for (int lane = 0; lane < 32; ++lane) {
        sz_u32_t const group = leaf_group_vec.u8s[lane];
        result_vec.u8s[lane] =
            group < (sz_u32_t)sz_utf8_word_break_haswell_astral_leaf_groups_k
                ? sz_utf8_word_break_haswell_astral_stage4_groups_[(sz_size_t)group * 256 + stage4_index_vec.u8s[lane]]
                : 0;
    }
    return result_vec.lasx;
}

/** @brief  Word_Break class byte for thirty-two ASCII codepoints (cp < 0x80) via the existing 128-entry property
 *          table, read by a bounded scalar walk (the window byte equals the codepoint on ASCII lanes). The LASX twin
 *          of @ref sz_utf8_word_break_ascii_class_haswell_. */
SZ_HELPER_AUTO __m256i sz_utf8_word_break_ascii_class_lasx_(__m256i bytes_u8x32) {
    sz_u256_vec_t byte_vec, result_vec;
    byte_vec.lasx = bytes_u8x32;
    for (int lane = 0; lane < 32; ++lane)
        result_vec.u8s[lane] = sz_utf8_word_break_property_ascii_[byte_vec.u8s[lane] & 0x7F];
    return result_vec.lasx;
}

/** @brief  Start-compacting BMP classify: the flat-table lookup is consumed only on 2-/3-byte codepoint-START lanes
 *          (ASCII lanes are overwritten by the property blend, 4-byte lanes by the astral blend, continuation lanes are
 *          don't-cares). A 64-byte window holds at most 64 such starts (adjacent malformed leads), so this packs their
 *          `(high, low)` bytes into dense buffers via a bit-test loop (no `ctz`), runs the flat lookup only over the
 *          populated 32-lane half/halves, then scatters the dense class bytes back. Bit-identical to a full
 *          @ref sz_utf8_word_break_bmp_class_lasx_ over both halves on every BMP-start lane; every other lane is a
 *          don't-care left at its incoming value. */
SZ_HELPER_AUTO void sz_utf8_word_break_bmp_compact_lasx_( //
    sz_u64_t bmp_starts, __m256i high_byte_low_u8x32, __m256i high_byte_high_u8x32, __m256i low_byte_low_u8x32,
    __m256i low_byte_high_u8x32, __m256i *out_low_u8x32, __m256i *out_high_u8x32) {
    sz_u8_t high_bytes[64], low_bytes[64];
    __lasx_xvst(high_byte_low_u8x32, high_bytes + 0, 0);
    __lasx_xvst(high_byte_high_u8x32, high_bytes + 32, 0);
    __lasx_xvst(low_byte_low_u8x32, low_bytes + 0, 0);
    __lasx_xvst(low_byte_high_u8x32, low_bytes + 32, 0);

    sz_u8_t dense_high[64] = {0}, dense_low[64] = {0};
    sz_size_t dense_index = 0;
    for (int lane = 0; lane < 64; ++lane)
        if ((bmp_starts >> lane) & 1ull) {
            dense_high[dense_index] = high_bytes[lane];
            dense_low[dense_index] = low_bytes[lane];
            ++dense_index;
        }

    sz_u8_t class_bytes[64];
    sz_size_t const populated_halves = (dense_index + 31) / 32;
    for (sz_size_t half = 0; half < populated_halves; ++half) {
        __m256i const dense_class_u8x32 = sz_utf8_word_break_bmp_class_lasx_(__lasx_xvld(dense_high + half * 32, 0),
                                                                             __lasx_xvld(dense_low + half * 32, 0));
        __lasx_xvst(dense_class_u8x32, class_bytes + half * 32, 0);
    }

    sz_u8_t scatter_low[32], scatter_high[32];
    __lasx_xvst(*out_low_u8x32, scatter_low, 0);
    __lasx_xvst(*out_high_u8x32, scatter_high, 0);
    dense_index = 0;
    for (int lane = 0; lane < 64; ++lane)
        if ((bmp_starts >> lane) & 1ull) {
            sz_u8_t const value = class_bytes[dense_index++];
            if (lane < 32) scatter_low[lane] = value;
            else scatter_high[lane - 32] = value;
        }
    *out_low_u8x32 = __lasx_xvld(scatter_low, 0);
    *out_high_u8x32 = __lasx_xvld(scatter_high, 0);
}

/** @brief  Per-window byte-lane classification (LASX): the Word_Break class byte per lane as two `__m256i` halves,
 *          valid only on codepoint-start lanes (the engine reads classes only at starts). The LASX twin of
 *          @ref sz_utf8_word_break_classify_window_haswell_, bit-identical on every start lane. ASCII through the
 *          property table, BMP through the compacted flat lookup, 4-byte leads through the astral cascade with the
 *          codepoint high/low/plane reconstructed from the forward neighbours. */
SZ_HELPER_AUTO void sz_utf8_word_break_classify_window_lasx_( //
    sz_utf8_rune_window_lasx_t window, __m256i *classes_low_u8x32, __m256i *classes_high_u8x32) {
    __m256i const raw_low_u8x32 = window.window_low_u8x32, raw_high_u8x32 = window.window_high_u8x32;
    sz_u64_t const ascii_starts = window.codepoint_starts & ~window.two_byte_starts & ~window.three_byte_starts &
                                  ~window.four_byte_starts;
    __m256i const four_select_low_u8x32 = sz_utf8_byte_mask_from_bits_lasx_((sz_u32_t)window.four_byte_starts);
    __m256i const four_select_high_u8x32 = sz_utf8_byte_mask_from_bits_lasx_((sz_u32_t)(window.four_byte_starts >> 32));
    __m256i const ascii_select_low_u8x32 = sz_utf8_byte_mask_from_bits_lasx_((sz_u32_t)ascii_starts);
    __m256i const ascii_select_high_u8x32 = sz_utf8_byte_mask_from_bits_lasx_((sz_u32_t)(ascii_starts >> 32));

    // BMP class over the compacted 2-/3-byte START lanes; ASCII / 4-byte / continuation lanes are don't-cares.
    __m256i out_low_u8x32 = __lasx_xvreplgr2vr_b(0);
    __m256i out_high_u8x32 = __lasx_xvreplgr2vr_b(0);
    sz_u64_t const bmp_starts = window.two_byte_starts | window.three_byte_starts;
    if (bmp_starts)
        sz_utf8_word_break_bmp_compact_lasx_(bmp_starts, window.high_byte_low_u8x32, window.high_byte_high_u8x32,
                                             window.low_byte_low_u8x32, window.low_byte_high_u8x32, &out_low_u8x32,
                                             &out_high_u8x32);

    // ASCII lanes: read the 128-entry property table directly off the raw byte.
    out_low_u8x32 = __lasx_xvbitsel_v(out_low_u8x32, sz_utf8_word_break_ascii_class_lasx_(raw_low_u8x32),
                                      ascii_select_low_u8x32);
    out_high_u8x32 = __lasx_xvbitsel_v(out_high_u8x32, sz_utf8_word_break_ascii_class_lasx_(raw_high_u8x32),
                                       ascii_select_high_u8x32);

    // 4-byte (astral) lanes: reconstruct the codepoint from the lead + three forward neighbours, then the astral
    // cascade addressed by offset = codepoint - 0x10000 (the offset's plane nibble is `plane - 1`).
    if (window.four_byte_starts) {
        __m256i next1_low_u8x32, next1_high_u8x32, next2_low_u8x32, next2_high_u8x32, next3_low_u8x32, next3_high_u8x32;
        sz_utf8_forward_neighbours_lasx_(raw_low_u8x32, raw_high_u8x32, &next1_low_u8x32, &next1_high_u8x32,
                                         &next2_low_u8x32, &next2_high_u8x32, &next3_low_u8x32, &next3_high_u8x32);
        __m256i const low_three_bits_u8x32 = __lasx_xvreplgr2vr_b(0x07), plane_bits_u8x32 = __lasx_xvreplgr2vr_b(0x1C),
                      low_nibble_u8x32 = __lasx_xvreplgr2vr_b(0x0F),
                      high_nibble_u8x32 = __lasx_xvreplgr2vr_b((char)0xF0),
                      low_two_bits_u8x32 = __lasx_xvreplgr2vr_b(0x03),
                      high_two_bits_u8x32 = __lasx_xvreplgr2vr_b((char)0xC0),
                      low_six_bits_u8x32 = __lasx_xvreplgr2vr_b(0x3F), one_u8x32 = __lasx_xvreplgr2vr_b(1);
        __m256i const plane_low_u8x32 = __lasx_xvor_v(
            __lasx_xvand_v(__lasx_xvslli_h(__lasx_xvand_v(raw_low_u8x32, low_three_bits_u8x32), 2), plane_bits_u8x32),
            sz_utf8_srl8_lasx_(next1_low_u8x32, 4, 0x03));
        __m256i const plane_high_u8x32 = __lasx_xvor_v(
            __lasx_xvand_v(__lasx_xvslli_h(__lasx_xvand_v(raw_high_u8x32, low_three_bits_u8x32), 2), plane_bits_u8x32),
            sz_utf8_srl8_lasx_(next1_high_u8x32, 4, 0x03));
        __m256i const high_byte_low_u8x32 = __lasx_xvor_v(
            __lasx_xvand_v(__lasx_xvslli_h(__lasx_xvand_v(next1_low_u8x32, low_nibble_u8x32), 4), high_nibble_u8x32),
            sz_utf8_srl8_lasx_(next2_low_u8x32, 2, 0x0F));
        __m256i const high_byte_high_u8x32 = __lasx_xvor_v(
            __lasx_xvand_v(__lasx_xvslli_h(__lasx_xvand_v(next1_high_u8x32, low_nibble_u8x32), 4), high_nibble_u8x32),
            sz_utf8_srl8_lasx_(next2_high_u8x32, 2, 0x0F));
        __m256i const low_byte_low_u8x32 = __lasx_xvor_v(
            __lasx_xvand_v(__lasx_xvslli_h(__lasx_xvand_v(next2_low_u8x32, low_two_bits_u8x32), 6),
                           high_two_bits_u8x32),
            __lasx_xvand_v(next3_low_u8x32, low_six_bits_u8x32));
        __m256i const low_byte_high_u8x32 = __lasx_xvor_v(
            __lasx_xvand_v(__lasx_xvslli_h(__lasx_xvand_v(next2_high_u8x32, low_two_bits_u8x32), 6),
                           high_two_bits_u8x32),
            __lasx_xvand_v(next3_high_u8x32, low_six_bits_u8x32));
        __m256i const plane_offset_low_u8x32 = __lasx_xvsub_b(plane_low_u8x32, one_u8x32);
        __m256i const plane_offset_high_u8x32 = __lasx_xvsub_b(plane_high_u8x32, one_u8x32);
        out_low_u8x32 = __lasx_xvbitsel_v(
            out_low_u8x32,
            sz_utf8_word_break_astral_class_lasx_(plane_offset_low_u8x32, high_byte_low_u8x32, low_byte_low_u8x32),
            four_select_low_u8x32);
        out_high_u8x32 = __lasx_xvbitsel_v(
            out_high_u8x32,
            sz_utf8_word_break_astral_class_lasx_(plane_offset_high_u8x32, high_byte_high_u8x32, low_byte_high_u8x32),
            four_select_high_u8x32);
    }
    *classes_low_u8x32 = out_low_u8x32;
    *classes_high_u8x32 = out_high_u8x32;
}

#pragma endregion In register vectorized classifier

#pragma region Mask algebra extractor

/** @brief  A 64-bit "class byte == @p value" lane mask over both class halves (two `xvseq.b` -> mask_combine). */
SZ_HELPER_INLINE sz_u64_t sz_utf8_word_break_class_mask_lasx_(__m256i classes_low_u8x32, __m256i classes_high_u8x32,
                                                              sz_u8_t value) {
    __m256i const value_broadcast_u8x32 = __lasx_xvreplgr2vr_b((char)value);
    return sz_utf8_mask_combine_lasx_(__lasx_xvseq_b(classes_low_u8x32, value_broadcast_u8x32),
                                      __lasx_xvseq_b(classes_high_u8x32, value_broadcast_u8x32));
}

/** @brief  A 64-bit "raw window byte == @p value" lane mask over both window halves. */
SZ_HELPER_INLINE sz_u64_t sz_utf8_word_break_byte_equal_lasx_(__m256i low_half_u8x32, __m256i high_half_u8x32,
                                                              sz_u8_t value) {
    __m256i const value_broadcast_u8x32 = __lasx_xvreplgr2vr_b((char)value);
    return sz_utf8_mask_combine_lasx_(__lasx_xvseq_b(low_half_u8x32, value_broadcast_u8x32),
                                      __lasx_xvseq_b(high_half_u8x32, value_broadcast_u8x32));
}

/** @brief  A 64-bit "raw window byte >= @p bound" (unsigned) lane mask over both window halves (native `xvsle.bu`). */
SZ_HELPER_INLINE sz_u64_t sz_utf8_word_break_byte_ge_lasx_(__m256i low_half_u8x32, __m256i high_half_u8x32,
                                                           sz_u8_t bound) {
    __m256i const bound_broadcast_u8x32 = __lasx_xvreplgr2vr_b((char)bound);
    return sz_utf8_mask_combine_lasx_(__lasx_xvsle_bu(bound_broadcast_u8x32, low_half_u8x32),
                                      __lasx_xvsle_bu(bound_broadcast_u8x32, high_half_u8x32));
}

/** @brief  Per-half "(high_byte, low_byte) 16-bit value in `[floor, ceiling]`" membership for one range (native
 *          unsigned compares), the LASX building block of @ref sz_utf8_word_break_range16_mask_lasx_. */
SZ_HELPER_INLINE __m256i sz_utf8_word_break_range16_one_lasx_(__m256i high_byte_u8x32, __m256i low_byte_u8x32,
                                                              sz_u16_t floor, sz_u16_t ceiling) {
    __m256i const floor_high_byte_u8x32 = __lasx_xvreplgr2vr_b((char)(floor >> 8)),
                  floor_low_byte_u8x32 = __lasx_xvreplgr2vr_b((char)(floor & 0xFF));
    __m256i const ceiling_high_byte_u8x32 = __lasx_xvreplgr2vr_b((char)(ceiling >> 8)),
                  ceiling_low_byte_u8x32 = __lasx_xvreplgr2vr_b((char)(ceiling & 0xFF));
    __m256i const high_byte_equals_floor_u8x32 = __lasx_xvseq_b(high_byte_u8x32, floor_high_byte_u8x32);
    __m256i const high_byte_equals_ceiling_u8x32 = __lasx_xvseq_b(high_byte_u8x32, ceiling_high_byte_u8x32);
    // not_below_u8x32: high_byte_u8x32 > floor_high_byte_u8x32, or (high_byte_u8x32 == floor_high_byte_u8x32 and
    // low_byte_u8x32 >= floor_low_byte_u8x32).
    __m256i const not_below_u8x32 = __lasx_xvor_v(
        __lasx_xvandn_v(high_byte_equals_floor_u8x32, __lasx_xvsle_bu(floor_high_byte_u8x32, high_byte_u8x32)),
        __lasx_xvand_v(high_byte_equals_floor_u8x32, __lasx_xvsle_bu(floor_low_byte_u8x32, low_byte_u8x32)));
    // not_above_u8x32: high_byte_u8x32 < ceiling_high_byte_u8x32, or (high_byte_u8x32 == ceiling_high_byte_u8x32 and
    // low_byte_u8x32 <= ceiling_low_byte_u8x32).
    __m256i const not_above_u8x32 = __lasx_xvor_v(
        __lasx_xvandn_v(high_byte_equals_ceiling_u8x32, __lasx_xvslt_bu(high_byte_u8x32, ceiling_high_byte_u8x32)),
        __lasx_xvand_v(high_byte_equals_ceiling_u8x32, __lasx_xvsle_bu(low_byte_u8x32, ceiling_low_byte_u8x32)));
    return __lasx_xvand_v(not_below_u8x32, not_above_u8x32);
}

/** @brief  A 64-bit "(high_byte, low_byte) 16-bit value in any sorted `[floor, ceiling]` range" lane mask over both
 *          window halves, the LASX twin of @ref sz_utf8_word_break_range16_mask_haswell_ (WSegSpace /
 *          Extended_Pictographic). */
SZ_HELPER_AUTO sz_u64_t sz_utf8_word_break_range16_mask_lasx_( //
    __m256i high_byte_low_u8x32, __m256i high_byte_high_u8x32, __m256i low_byte_low_u8x32, __m256i low_byte_high_u8x32,
    sz_u16_t const *floor_table, sz_u16_t const *ceiling_table, int count) {
    __m256i hit_low_u8x32 = __lasx_xvreplgr2vr_b(0), hit_high_u8x32 = __lasx_xvreplgr2vr_b(0);
    for (int range = 0; range < count; ++range) {
        hit_low_u8x32 = __lasx_xvor_v(hit_low_u8x32,
                                      sz_utf8_word_break_range16_one_lasx_(high_byte_low_u8x32, low_byte_low_u8x32,
                                                                           floor_table[range], ceiling_table[range]));
        hit_high_u8x32 = __lasx_xvor_v(hit_high_u8x32,
                                       sz_utf8_word_break_range16_one_lasx_(high_byte_high_u8x32, low_byte_high_u8x32,
                                                                            floor_table[range], ceiling_table[range]));
    }
    return sz_utf8_mask_combine_lasx_(hit_low_u8x32, hit_high_u8x32);
}

/**
 *  @brief  Per-ISA extractor: lower one classified 64-byte window to the portable @ref sz_utf8_word_break_frame_t - the
 *          LASX twin of @ref sz_utf8_word_break_build_frame_haswell_. Applies the truncated-edge U+FFFD reclassify to
 *          the class halves, materializes every per-class lane mask + the raw-byte membership masks, the
 *          Extended_Pictographic mask (BMP + SMP range scan), and the per-lane class byte array.
 */
SZ_HELPER_AUTO sz_utf8_word_break_frame_t sz_utf8_word_break_build_frame_lasx_(
    sz_utf8_rune_window_lasx_t window, __m256i classes_low_u8x32, __m256i classes_high_u8x32, sz_u64_t start_bytes_all,
    sz_u64_t length_two, sz_u64_t length_three, sz_u64_t length_four, int want_pictographic) {

    sz_size_t const loaded = window.loaded;
    sz_u64_t const valid = sz_u64_mask_until_serial_(loaded);
    sz_u64_t const start_bytes = start_bytes_all & valid;
    __m256i const raw_low_u8x32 = window.window_low_u8x32, raw_high_u8x32 = window.window_high_u8x32;

    // Truncated-edge U+FFFD reclassify (force the class to Other on a lead whose declared span runs past `loaded`).
    sz_u64_t const lead_two = length_two & start_bytes;
    sz_u64_t const lead_three = length_three & start_bytes;
    sz_u64_t const lead_four = length_four & start_bytes;
    sz_u64_t const truncated_raw = ((lead_two & ~sz_u64_mask_until_serial_(loaded > 1 ? loaded - 1 : 0)) |
                                    (lead_three & ~sz_u64_mask_until_serial_(loaded > 2 ? loaded - 2 : 0)) |
                                    (lead_four & ~sz_u64_mask_until_serial_(loaded > 3 ? loaded - 3 : 0))) &
                                   valid;
    if (truncated_raw) {
        __m256i const other_class_u8x32 = __lasx_xvreplgr2vr_b((char)sz_utf8_word_break_other_k);
        __m256i const sel_low_u8x32 = sz_utf8_byte_mask_from_bits_lasx_((sz_u32_t)truncated_raw);
        __m256i const sel_high_u8x32 = sz_utf8_byte_mask_from_bits_lasx_((sz_u32_t)(truncated_raw >> 32));
        classes_low_u8x32 = __lasx_xvbitsel_v(classes_low_u8x32, other_class_u8x32, sel_low_u8x32);
        classes_high_u8x32 = __lasx_xvbitsel_v(classes_high_u8x32, other_class_u8x32, sel_high_u8x32);
    }

    sz_utf8_word_break_frame_t frame;
    frame.class_aletter = sz_utf8_word_break_class_mask_lasx_(classes_low_u8x32, classes_high_u8x32,
                                                              sz_utf8_word_break_aletter_k);
    frame.class_hebrew = sz_utf8_word_break_class_mask_lasx_(classes_low_u8x32, classes_high_u8x32,
                                                             sz_utf8_word_break_hebrew_letter_k);
    frame.class_numeric = sz_utf8_word_break_class_mask_lasx_(classes_low_u8x32, classes_high_u8x32,
                                                              sz_utf8_word_break_numeric_k);
    frame.class_katakana = sz_utf8_word_break_class_mask_lasx_(classes_low_u8x32, classes_high_u8x32,
                                                               sz_utf8_word_break_katakana_k);
    frame.class_extendnumlet = sz_utf8_word_break_class_mask_lasx_(classes_low_u8x32, classes_high_u8x32,
                                                                   sz_utf8_word_break_extendnumlet_k);
    frame.class_extend = sz_utf8_word_break_class_mask_lasx_(classes_low_u8x32, classes_high_u8x32,
                                                             sz_utf8_word_break_extend_k);
    frame.class_zwj = sz_utf8_word_break_class_mask_lasx_(classes_low_u8x32, classes_high_u8x32,
                                                          sz_utf8_word_break_zwj_k);
    frame.class_format = sz_utf8_word_break_class_mask_lasx_(classes_low_u8x32, classes_high_u8x32,
                                                             sz_utf8_word_break_format_k);
    frame.class_midletter = sz_utf8_word_break_class_mask_lasx_(classes_low_u8x32, classes_high_u8x32,
                                                                sz_utf8_word_break_midletter_k);
    frame.class_midnum = sz_utf8_word_break_class_mask_lasx_(classes_low_u8x32, classes_high_u8x32,
                                                             sz_utf8_word_break_midnum_k);
    frame.class_mid_quotes = sz_utf8_word_break_class_mask_lasx_(classes_low_u8x32, classes_high_u8x32,
                                                                 sz_utf8_word_break_mid_quotes_k);
    frame.class_cr = sz_utf8_word_break_class_mask_lasx_(classes_low_u8x32, classes_high_u8x32,
                                                         sz_utf8_word_break_cr_k);
    frame.class_lf = sz_utf8_word_break_class_mask_lasx_(classes_low_u8x32, classes_high_u8x32,
                                                         sz_utf8_word_break_lf_k);
    frame.class_newline = sz_utf8_word_break_class_mask_lasx_(classes_low_u8x32, classes_high_u8x32,
                                                              sz_utf8_word_break_newline_k);
    frame.class_regional = sz_utf8_word_break_class_mask_lasx_(classes_low_u8x32, classes_high_u8x32,
                                                               sz_utf8_word_break_regional_ind_k);

    sz_u64_t const non_ascii_lanes = sz_utf8_mask_combine_lasx_(raw_low_u8x32, raw_high_u8x32) & valid;
    frame.non_ascii_lanes = non_ascii_lanes;
    frame.double_quote_byte = sz_utf8_word_break_byte_equal_lasx_(raw_low_u8x32, raw_high_u8x32, 0x22) & valid;
    frame.single_quote_byte = sz_utf8_word_break_byte_equal_lasx_(raw_low_u8x32, raw_high_u8x32, 0x27) & valid;

    // WB3d WSegSpace raw membership: the ASCII U+0020 byte compare OR the multibyte (high,low) range scan.
    sz_u64_t wseg_multibyte = 0ull;
    if (non_ascii_lanes)
        wseg_multibyte = sz_utf8_word_break_range16_mask_lasx_(window.high_byte_low_u8x32, window.high_byte_high_u8x32,
                                                               window.low_byte_low_u8x32, window.low_byte_high_u8x32,
                                                               sz_utf8_word_break_wseg_lo_, sz_utf8_word_break_wseg_hi_,
                                                               sz_utf8_word_break_wseg_count_k) &
                         non_ascii_lanes;
    frame.wseg = (wseg_multibyte | (sz_utf8_word_break_byte_equal_lasx_(raw_low_u8x32, raw_high_u8x32, 0x20) & valid));

    // WB3c Extended_Pictographic raw membership (BMP range scan on non-4-byte lanes, SMP range scan on plane-one
    // 4-byte lanes). Rare-class gated on `want_pictographic`, so the ~156-range scan is skipped on the common window.
    frame.pictographic = 0ull;
    sz_u64_t const four_byte = window.four_byte_starts & valid;
    if (want_pictographic) {
        __m256i next1_low_u8x32, next1_high_u8x32, next2_low_u8x32, next2_high_u8x32, next3_low_u8x32, next3_high_u8x32;
        sz_utf8_forward_neighbours_lasx_(raw_low_u8x32, raw_high_u8x32, &next1_low_u8x32, &next1_high_u8x32,
                                         &next2_low_u8x32, &next2_high_u8x32, &next3_low_u8x32, &next3_high_u8x32);
        __m256i const low_three_bits_u8x32 = __lasx_xvreplgr2vr_b(0x07), plane_bits_u8x32 = __lasx_xvreplgr2vr_b(0x1C),
                      low_nibble_u8x32 = __lasx_xvreplgr2vr_b(0x0F),
                      high_nibble_u8x32 = __lasx_xvreplgr2vr_b((char)0xF0),
                      low_two_bits_u8x32 = __lasx_xvreplgr2vr_b(0x03),
                      high_two_bits_u8x32 = __lasx_xvreplgr2vr_b((char)0xC0),
                      low_six_bits_u8x32 = __lasx_xvreplgr2vr_b(0x3F), one_u8x32 = __lasx_xvreplgr2vr_b(1);
        __m256i const plane_low_u8x32 = __lasx_xvor_v(
            __lasx_xvand_v(__lasx_xvslli_h(__lasx_xvand_v(raw_low_u8x32, low_three_bits_u8x32), 2), plane_bits_u8x32),
            sz_utf8_srl8_lasx_(next1_low_u8x32, 4, 0x03));
        __m256i const plane_high_u8x32 = __lasx_xvor_v(
            __lasx_xvand_v(__lasx_xvslli_h(__lasx_xvand_v(raw_high_u8x32, low_three_bits_u8x32), 2), plane_bits_u8x32),
            sz_utf8_srl8_lasx_(next1_high_u8x32, 4, 0x03));
        sz_u64_t const plane_one = sz_utf8_mask_combine_lasx_(__lasx_xvseq_b(plane_low_u8x32, one_u8x32),
                                                              __lasx_xvseq_b(plane_high_u8x32, one_u8x32));
        __m256i const smp_high_byte_low_u8x32 = __lasx_xvor_v(
            __lasx_xvand_v(__lasx_xvslli_h(__lasx_xvand_v(next1_low_u8x32, low_nibble_u8x32), 4), high_nibble_u8x32),
            sz_utf8_srl8_lasx_(next2_low_u8x32, 2, 0x0F));
        __m256i const smp_high_byte_high_u8x32 = __lasx_xvor_v(
            __lasx_xvand_v(__lasx_xvslli_h(__lasx_xvand_v(next1_high_u8x32, low_nibble_u8x32), 4), high_nibble_u8x32),
            sz_utf8_srl8_lasx_(next2_high_u8x32, 2, 0x0F));
        __m256i const smp_low_byte_low_u8x32 = __lasx_xvor_v(
            __lasx_xvand_v(__lasx_xvslli_h(__lasx_xvand_v(next2_low_u8x32, low_two_bits_u8x32), 6),
                           high_two_bits_u8x32),
            __lasx_xvand_v(next3_low_u8x32, low_six_bits_u8x32));
        __m256i const smp_low_byte_high_u8x32 = __lasx_xvor_v(
            __lasx_xvand_v(__lasx_xvslli_h(__lasx_xvand_v(next2_high_u8x32, low_two_bits_u8x32), 6),
                           high_two_bits_u8x32),
            __lasx_xvand_v(next3_high_u8x32, low_six_bits_u8x32));
        sz_u64_t const pictographic_bmp = sz_utf8_word_break_range16_mask_lasx_(
            window.high_byte_low_u8x32, window.high_byte_high_u8x32, window.low_byte_low_u8x32,
            window.low_byte_high_u8x32, sz_utf8_word_break_pict_bmp_lo_, sz_utf8_word_break_pict_bmp_hi_,
            sz_utf8_word_break_pict_bmp_count_k);
        sz_u64_t const pictographic_smp = sz_utf8_word_break_range16_mask_lasx_(
            smp_high_byte_low_u8x32, smp_high_byte_high_u8x32, smp_low_byte_low_u8x32, smp_low_byte_high_u8x32,
            sz_utf8_word_break_pict_smp_lo_, sz_utf8_word_break_pict_smp_hi_, sz_utf8_word_break_pict_smp_count_k);
        frame.pictographic = (pictographic_bmp & non_ascii_lanes & ~four_byte) |
                             (pictographic_smp & four_byte & plane_one);
    }

    __lasx_xvst(classes_low_u8x32, frame.classes_byte + 0, 0);
    __lasx_xvst(classes_high_u8x32, frame.classes_byte + 32, 0);
    return frame;
}

#pragma endregion Mask algebra extractor

#pragma region Codepoint partition

/** @brief  Resolve one window into the maximal-subpart partition - the LASX twin of
 *          @ref sz_utf8_word_break_partition_haswell_: compute the per-ISA `sz_u64_t` masks and delegate to the
 *          portable @ref sz_utf8_word_break_partition_from_masks_. */
SZ_HELPER_AUTO sz_utf8_word_break_partition_t sz_utf8_word_break_partition_lasx_(sz_utf8_rune_window_lasx_t window,
                                                                                 sz_u64_t valid, int at_end_of_text) {
    __m256i const raw_low_u8x32 = window.window_low_u8x32, raw_high_u8x32 = window.window_high_u8x32;
    sz_u64_t const real_continuation = window.continuation & valid;
    __m256i const high_nibble_low_u8x32 = sz_utf8_srl8_lasx_(raw_low_u8x32, 4, 0x0F);
    __m256i const high_nibble_high_u8x32 = sz_utf8_srl8_lasx_(raw_high_u8x32, 4, 0x0F);
    __m256i const two_byte_lead_first_nibble_u8x32 = __lasx_xvreplgr2vr_b(0x0C),
                  two_byte_lead_second_nibble_u8x32 = __lasx_xvreplgr2vr_b(0x0D),
                  three_byte_lead_nibble_u8x32 = __lasx_xvreplgr2vr_b(0x0E),
                  four_byte_lead_nibble_u8x32 = __lasx_xvreplgr2vr_b(0x0F);
    sz_u64_t const length_two =
        sz_utf8_mask_combine_lasx_(
            __lasx_xvor_v(__lasx_xvseq_b(high_nibble_low_u8x32, two_byte_lead_first_nibble_u8x32),
                          __lasx_xvseq_b(high_nibble_low_u8x32, two_byte_lead_second_nibble_u8x32)),
            __lasx_xvor_v(__lasx_xvseq_b(high_nibble_high_u8x32, two_byte_lead_first_nibble_u8x32),
                          __lasx_xvseq_b(high_nibble_high_u8x32, two_byte_lead_second_nibble_u8x32))) &
        valid;
    sz_u64_t const length_three = sz_utf8_mask_combine_lasx_(
                                      __lasx_xvseq_b(high_nibble_low_u8x32, three_byte_lead_nibble_u8x32),
                                      __lasx_xvseq_b(high_nibble_high_u8x32, three_byte_lead_nibble_u8x32)) &
                                  valid;
    sz_u64_t const length_four = sz_utf8_mask_combine_lasx_(
                                     __lasx_xvseq_b(high_nibble_low_u8x32, four_byte_lead_nibble_u8x32),
                                     __lasx_xvseq_b(high_nibble_high_u8x32, four_byte_lead_nibble_u8x32)) &
                                 valid;
    sz_u64_t const length_ge_two = length_two | length_three | length_four;
    sz_u64_t bad_second_byte = 0ull;
    if (length_ge_two) {
        __m256i next1_low_u8x32, next1_high_u8x32, next2_low_u8x32, next2_high_u8x32, next3_low_u8x32, next3_high_u8x32;
        sz_utf8_forward_neighbours_lasx_(raw_low_u8x32, raw_high_u8x32, &next1_low_u8x32, &next1_high_u8x32,
                                         &next2_low_u8x32, &next2_high_u8x32, &next3_low_u8x32, &next3_high_u8x32);
        (void)next2_low_u8x32, (void)next2_high_u8x32, (void)next3_low_u8x32,
            (void)next3_high_u8x32; // only next1 gates the second byte
        sz_u64_t const next1_at_least_a0 = sz_utf8_word_break_byte_ge_lasx_(next1_low_u8x32, next1_high_u8x32, 0xA0);
        sz_u64_t const next1_at_least_90 = sz_utf8_word_break_byte_ge_lasx_(next1_low_u8x32, next1_high_u8x32, 0x90);
        sz_u64_t const lead_c0_c1 = (sz_utf8_word_break_byte_equal_lasx_(raw_low_u8x32, raw_high_u8x32, 0xC0) |
                                     sz_utf8_word_break_byte_equal_lasx_(raw_low_u8x32, raw_high_u8x32, 0xC1)) &
                                    valid;
        sz_u64_t const lead_e0 = sz_utf8_word_break_byte_equal_lasx_(raw_low_u8x32, raw_high_u8x32, 0xE0) & valid;
        sz_u64_t const lead_ed = sz_utf8_word_break_byte_equal_lasx_(raw_low_u8x32, raw_high_u8x32, 0xED) & valid;
        sz_u64_t const lead_f0 = sz_utf8_word_break_byte_equal_lasx_(raw_low_u8x32, raw_high_u8x32, 0xF0) & valid;
        sz_u64_t const lead_f4 = sz_utf8_word_break_byte_equal_lasx_(raw_low_u8x32, raw_high_u8x32, 0xF4) & valid;
        sz_u64_t const lead_f5_or_more = sz_utf8_word_break_byte_ge_lasx_(raw_low_u8x32, raw_high_u8x32, 0xF5) & valid;
        bad_second_byte = lead_c0_c1 | (lead_e0 & ~next1_at_least_a0) | (lead_ed & next1_at_least_a0) |
                          (lead_f0 & ~next1_at_least_90) | (lead_f4 & next1_at_least_90) | lead_f5_or_more;
    }
    return sz_utf8_word_break_partition_from_masks_(real_continuation, length_two, length_three, length_four,
                                                    bad_second_byte, valid, at_end_of_text);
}

#pragma endregion Codepoint partition

#pragma region Forward driver

/**
 *  @brief  Forward UAX-29 word segmentation over `[0, length)` (LoongArch LASX): the overlap-free advancing driver,
 *          mirroring @ref sz_utf8_wordbreaks_haswell over the LASX window/classify/partition/decide/drain leaves.
 *          Bit-exact with `sz_utf8_wordbreaks_serial` and every other windowed backend.
 */
SZ_API_COMPTIME sz_size_t sz_utf8_wordbreaks_lasx(   //
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
        sz_utf8_rune_window_lasx_t const window = sz_utf8_rune_decode_window_lasx_(text_u8 + position,
                                                                                   length - position);
        sz_size_t const loaded = window.loaded;
        sz_u64_t const valid = sz_u64_mask_until_serial_(loaded);

        __m256i classes_low_u8x32, classes_high_u8x32;
        sz_utf8_word_break_classify_window_lasx_(window, &classes_low_u8x32, &classes_high_u8x32);

        sz_utf8_word_break_partition_t const partition = sz_utf8_word_break_partition_lasx_(
            window, valid, position + loaded >= length);
        sz_u64_t const start_bytes_all = partition.start_bytes;
        sz_u64_t const continuation_all = partition.continuation;
        sz_u64_t const forced_other = partition.forced_other;
        sz_u64_t const length_two = partition.length_two;
        sz_u64_t const length_three = partition.length_three;
        sz_u64_t const length_four = partition.length_four;
        if (forced_other) {
            __m256i const other_class_u8x32 = __lasx_xvreplgr2vr_b((char)sz_utf8_word_break_other_k);
            __m256i const sel_low_u8x32 = sz_utf8_byte_mask_from_bits_lasx_((sz_u32_t)forced_other);
            __m256i const sel_high_u8x32 = sz_utf8_byte_mask_from_bits_lasx_((sz_u32_t)(forced_other >> 32));
            classes_low_u8x32 = __lasx_xvbitsel_v(classes_low_u8x32, other_class_u8x32, sel_low_u8x32);
            classes_high_u8x32 = __lasx_xvbitsel_v(classes_high_u8x32, other_class_u8x32, sel_high_u8x32);
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
            // First straddling lead's index without a scalar `ctz`: `x & -x` isolates the lowest set bit, and
            // `63 - clz(...)` turns that into its lane index.
            sz_size_t limit = straddle ? (sz_size_t)(63 - sz_u64_clz(straddle & (~straddle + 1ull))) : loaded;
            if ((text_u8[position + loaded] & 0xC0) == 0x80) {
                sz_size_t const last_lead = (sz_size_t)(63 - sz_u64_clz(start_bytes_all));
                sz_size_t const last_lead_length = sz_utf8_lead_length_(text_u8[position + last_lead]);
                if (last_lead + last_lead_length > loaded && last_lead < limit) limit = last_lead;
            }
            if (limit > 0) complete_limit = limit;
        }

        int const want_pictographic = sz_utf8_word_break_class_mask_lasx_(classes_low_u8x32, classes_high_u8x32,
                                                                          sz_utf8_word_break_zwj_k) != 0 ||
                                      carry.prev_ends_in_zwj;
        sz_utf8_word_break_frame_t const frame = sz_utf8_word_break_build_frame_lasx_(
            window, classes_low_u8x32, classes_high_u8x32, start_bytes_all, length_two, length_three, length_four,
            want_pictographic);

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

        words = sz_utf8_rune_drain_forward_lasx_(boundary_lanes, position, word_starts, word_lengths, words,
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

#endif // SZ_USE_LASX

#ifdef __cplusplus
}
#endif

#endif // STRINGZILLA_UTF8_WORDBREAKS_LASX_H_
