/**
 *  @file include/stringzilla/utf8_words/haswell.h
 *  @author Ash Vardanian
 *  @brief  Fully-vectorized UAX-29 Word_Break segmentation for AVX2 (Haswell). The AVX2 twin of the Ice Lake kernel:
 *          no path scalar-walks codepoints, spills a YMM to the stack to call the serial oracle, or issues a gather.
 *
 *  Each 64-byte window lives as two `__m256i` halves; every per-codepoint Word_Break property resolves in-register via
 *  a `vpshufb` nibble cascade (the AVX2 stand-in for the VBMI trie / page LUT - AVX2 has no `vpermb`/`vpcompressb`),
 *  bit-identical to `sz_rune_word_break_property` over the whole BMP and Supplementary Plane. The classified window is
 *  lowered to the portable @ref sz_utf8_word_break_frame_t and handed to the shared `sz_utf8_word_break_decide_window_`
 *  rule engine, so WB1-WB16 (including the cross-window bridge shadow / RI parity / left-context carry) run once in
 *  portable `sz_u64_t` bit algebra. Dense-compaction of the boundary mask uses the substrate's BMI2 index loop.
 */
#ifndef STRINGZILLA_UTF8_WORDS_HASWELL_H_
#define STRINGZILLA_UTF8_WORDS_HASWELL_H_

#include "stringzilla/types.h"
#include "stringzilla/utf8_words/tables.h"
#include "stringzilla/utf8_words/serial.h"
#include "stringzilla/utf8_codepoints/haswell.h"

#ifdef __cplusplus
extern "C" {
#endif

#if SZ_USE_HASWELL
#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("avx2,bmi,bmi2,popcnt"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("avx2", "bmi", "bmi2", "popcnt")
#endif

#pragma region UAX 29 Word Boundaries forward kernel

#pragma region In register vectorized classifier

/** @brief  Word_Break class byte for thirty-two BMP codepoints (per-lane high = cp>>8, low = cp&0xFF) via a
 *          register-resident `vpshufb` nibble cascade, the AVX2 twin of the Ice Lake BMP classifier. Gather-free;
 *          bit-exact with `sz_rune_word_break_property` over the whole BMP. */
SZ_INTERNAL __m256i sz_utf8_word_break_bmp_class_haswell_(__m256i high, __m256i low) {
    __m256i const low_nibble_mask = _mm256_set1_epi8(0x0F);
    __m256i const page = sz_utf8_codepoints_lut256_haswell_(sz_utf8_word_break_haswell_stage1_, high);
    __m256i const low_high = _mm256_and_si256(_mm256_srli_epi16(low, 4), low_nibble_mask);
    __m256i const leaf_lo = sz_utf8_codepoints_cascade_stage_haswell_(
        sz_utf8_word_break_haswell_stage2_lo_, sz_utf8_word_break_haswell_stage2_lo_count_k / 16, page, low_high);
    __m256i const leaf_hi = sz_utf8_codepoints_cascade_stage_haswell_(
        sz_utf8_word_break_haswell_stage2_hi_, sz_utf8_word_break_haswell_stage2_hi_count_k / 16, page, low_high);
    __m256i const leaf_group = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(leaf_lo, 4), low_nibble_mask),
                                               _mm256_slli_epi16(leaf_hi, 4));
    __m256i const leaf_low_nibble = _mm256_and_si256(leaf_lo, low_nibble_mask);
    __m256i const low_low = _mm256_and_si256(low, low_nibble_mask);
    __m256i const lut_index = _mm256_or_si256(_mm256_slli_epi16(leaf_low_nibble, 4), low_low);
    __m256i result = _mm256_setzero_si256();
    for (int group = 0; group < (int)sz_utf8_word_break_haswell_leaf_groups_k; ++group) {
        __m256i const value = sz_utf8_codepoints_lut256_haswell_(
            sz_utf8_word_break_haswell_stage3_groups_ + group * 256, lut_index);
        __m256i const here = _mm256_cmpeq_epi8(leaf_group, _mm256_set1_epi8((char)group));
        result = _mm256_blendv_epi8(result, value, here);
    }
    return result;
}

/** @brief  Word_Break class byte for thirty-two ASTRAL codepoints over the 20-bit offset = cp - 0x10000 (5-nibble
 *          cascade), the AVX2 twin of `sz_utf8_word_break_classify_astral16_icelake_`. Per-lane bytes:
 *          @p plane_off = (offset>>16)&0xFF (low nibble meaningful), @p high = (offset>>8)&0xFF, @p low = offset&0xFF.
 *          Gather-free; bit-exact with `sz_rune_word_break_property` over the Supplementary Planes. */
SZ_INTERNAL __m256i sz_utf8_word_break_astral_class_haswell_(__m256i plane_off, __m256i high, __m256i low) {
    __m256i const low_nibble_mask = _mm256_set1_epi8(0x0F);
    __m256i const n4 = _mm256_and_si256(plane_off, low_nibble_mask);
    __m256i const n3 = _mm256_and_si256(_mm256_srli_epi16(high, 4), low_nibble_mask);
    __m256i const stage1_index = _mm256_or_si256(_mm256_slli_epi16(n4, 4), n3);
    __m256i const page = sz_utf8_codepoints_lut256_haswell_(sz_utf8_word_break_haswell_astral_stage1_, stage1_index);
    __m256i const n2 = _mm256_and_si256(high, low_nibble_mask);
    __m256i const leaf2 = sz_utf8_codepoints_cascade_stage_haswell_(
        sz_utf8_word_break_haswell_astral_stage2_lo_, sz_utf8_word_break_haswell_astral_stage2_lo_count_k / 16, page,
        n2);
    __m256i const n1 = _mm256_and_si256(_mm256_srli_epi16(low, 4), low_nibble_mask);
    __m256i const leaf_lo = sz_utf8_codepoints_cascade_stage_haswell_(
        sz_utf8_word_break_haswell_astral_stage3_lo_, sz_utf8_word_break_haswell_astral_stage3_lo_count_k / 16, leaf2,
        n1);
    __m256i const leaf_hi = sz_utf8_codepoints_cascade_stage_haswell_(
        sz_utf8_word_break_haswell_astral_stage3_hi_, sz_utf8_word_break_haswell_astral_stage3_hi_count_k / 16, leaf2,
        n1);
    __m256i const n0 = _mm256_and_si256(low, low_nibble_mask);
    __m256i const leaf_group = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(leaf_lo, 4), low_nibble_mask),
                                               _mm256_slli_epi16(leaf_hi, 4));
    __m256i const leaf_low_nibble = _mm256_and_si256(leaf_lo, low_nibble_mask);
    __m256i const stage4_lut_index = _mm256_or_si256(_mm256_slli_epi16(leaf_low_nibble, 4), n0);
    __m256i result = _mm256_setzero_si256();
    for (int group = 0; group < (int)sz_utf8_word_break_haswell_astral_leaf_groups_k; ++group) {
        __m256i const value = sz_utf8_codepoints_lut256_haswell_(
            sz_utf8_word_break_haswell_astral_stage4_groups_ + group * 256, stage4_lut_index);
        __m256i const here = _mm256_cmpeq_epi8(leaf_group, _mm256_set1_epi8((char)group));
        result = _mm256_blendv_epi8(result, value, here);
    }
    return result;
}

/** @brief  Word_Break class byte for thirty-two ASCII codepoints (cp < 0x80) via the existing 128-entry property
 *          table, read in-register by two `lut256` halves (low six bits) blended on bit 6, the AVX2 twin of the
 *          icelake ASCII permute. The window byte equals the codepoint on ASCII lanes. */
SZ_INTERNAL __m256i sz_utf8_word_break_ascii_class_haswell_(__m256i bytes) {
    __m256i const index_low6 = _mm256_and_si256(bytes, _mm256_set1_epi8(0x3F));
    __m256i const low_half = sz_utf8_codepoints_lut256_haswell_(sz_utf8_word_break_property_ascii_ + 0, index_low6);
    __m256i const high_half = sz_utf8_codepoints_lut256_haswell_(sz_utf8_word_break_property_ascii_ + 64, index_low6);
    __m256i const high_bit = _mm256_cmpeq_epi8(_mm256_and_si256(bytes, _mm256_set1_epi8(0x40)), _mm256_set1_epi8(0x40));
    return _mm256_blendv_epi8(low_half, high_half, high_bit);
}

/** @brief  Per-window byte-lane classification (AVX2): the Word_Break class byte per lane as two `__m256i` halves,
 *          valid only on codepoint-start lanes (the engine reads classes only at starts). The Haswell twin of
 *          @ref sz_utf8_word_break_classify_window_icelake_, bit-identical on every start lane. ASCII through the
 *          property table, BMP through the nibble cascade, 4-byte leads through the astral cascade with the codepoint
 *          high/low/plane reconstructed from the forward neighbours. */
SZ_INTERNAL void sz_utf8_word_break_classify_window_haswell_( //
    sz_utf8_codepoints_window_haswell_t window, __m256i *classes_lo, __m256i *classes_hi) {
    __m256i const raw_lo = window.window_lo, raw_hi = window.window_hi;
    sz_u64_t const ascii_starts = window.codepoint_starts & ~window.two_byte_starts & ~window.three_byte_starts &
                                  ~window.four_byte_starts;
    __m256i const four_select_lo = sz_utf8_codepoints_byte_mask_from_bits_haswell_((sz_u32_t)window.four_byte_starts);
    __m256i const four_select_hi = sz_utf8_codepoints_byte_mask_from_bits_haswell_(
        (sz_u32_t)(window.four_byte_starts >> 32));
    __m256i const ascii_select_lo = sz_utf8_codepoints_byte_mask_from_bits_haswell_((sz_u32_t)ascii_starts);
    __m256i const ascii_select_hi = sz_utf8_codepoints_byte_mask_from_bits_haswell_((sz_u32_t)(ascii_starts >> 32));

    // BMP class via the cascade over the decoder's reconstructed (high, low). An ASCII lane has high == 0 and
    // low == the raw byte; a 4-byte lane's BMP class is discarded by the astral blend below.
    __m256i out_lo = sz_utf8_word_break_bmp_class_haswell_(window.high_lo, window.low_lo);
    __m256i out_hi = sz_utf8_word_break_bmp_class_haswell_(window.high_hi, window.low_hi);

    // ASCII lanes: read the 128-entry property table directly off the raw byte.
    out_lo = _mm256_blendv_epi8(out_lo, sz_utf8_word_break_ascii_class_haswell_(raw_lo), ascii_select_lo);
    out_hi = _mm256_blendv_epi8(out_hi, sz_utf8_word_break_ascii_class_haswell_(raw_hi), ascii_select_hi);

    // 4-byte (astral) lanes: reconstruct the codepoint from the lead + three forward neighbours, then the astral
    // cascade addressed by offset = codepoint - 0x10000 (the offset's plane nibble is `plane - 1`).
    if (window.four_byte_starts) {
        __m256i next1_lo, next1_hi, next2_lo, next2_hi, next3_lo, next3_hi;
        sz_utf8_codepoints_forward_neighbours_haswell_(raw_lo, raw_hi, &next1_lo, &next1_hi, &next2_lo, &next2_hi);
        __m256i const low_successor = _mm256_permute2x128_si256(raw_lo, raw_hi, 0x21);
        next3_lo = _mm256_alignr_epi8(low_successor, raw_lo, 3);
        __m256i const high_successor = _mm256_permute2x128_si256(raw_hi, raw_lo, 0x21);
        next3_hi = _mm256_alignr_epi8(high_successor, raw_hi, 3);
        __m256i const plane_lo = _mm256_or_si256(
            _mm256_and_si256(_mm256_slli_epi16(_mm256_and_si256(raw_lo, _mm256_set1_epi8(0x07)), 2),
                             _mm256_set1_epi8(0x1C)),
            sz_utf8_codepoints_srl8_haswell_(next1_lo, 4, 0x03));
        __m256i const plane_hi = _mm256_or_si256(
            _mm256_and_si256(_mm256_slli_epi16(_mm256_and_si256(raw_hi, _mm256_set1_epi8(0x07)), 2),
                             _mm256_set1_epi8(0x1C)),
            sz_utf8_codepoints_srl8_haswell_(next1_hi, 4, 0x03));
        __m256i const high_lo = _mm256_or_si256(
            _mm256_and_si256(_mm256_slli_epi16(_mm256_and_si256(next1_lo, _mm256_set1_epi8(0x0F)), 4),
                             _mm256_set1_epi8((char)0xF0)),
            sz_utf8_codepoints_srl8_haswell_(next2_lo, 2, 0x0F));
        __m256i const high_hi = _mm256_or_si256(
            _mm256_and_si256(_mm256_slli_epi16(_mm256_and_si256(next1_hi, _mm256_set1_epi8(0x0F)), 4),
                             _mm256_set1_epi8((char)0xF0)),
            sz_utf8_codepoints_srl8_haswell_(next2_hi, 2, 0x0F));
        __m256i const low_lo = _mm256_or_si256(
            _mm256_and_si256(_mm256_slli_epi16(_mm256_and_si256(next2_lo, _mm256_set1_epi8(0x03)), 6),
                             _mm256_set1_epi8((char)0xC0)),
            _mm256_and_si256(next3_lo, _mm256_set1_epi8(0x3F)));
        __m256i const low_hi = _mm256_or_si256(
            _mm256_and_si256(_mm256_slli_epi16(_mm256_and_si256(next2_hi, _mm256_set1_epi8(0x03)), 6),
                             _mm256_set1_epi8((char)0xC0)),
            _mm256_and_si256(next3_hi, _mm256_set1_epi8(0x3F)));
        __m256i const plane_off_lo = _mm256_sub_epi8(plane_lo, _mm256_set1_epi8(1));
        __m256i const plane_off_hi = _mm256_sub_epi8(plane_hi, _mm256_set1_epi8(1));
        out_lo = _mm256_blendv_epi8(out_lo, sz_utf8_word_break_astral_class_haswell_(plane_off_lo, high_lo, low_lo),
                                    four_select_lo);
        out_hi = _mm256_blendv_epi8(out_hi, sz_utf8_word_break_astral_class_haswell_(plane_off_hi, high_hi, low_hi),
                                    four_select_hi);
    }
    *classes_lo = out_lo;
    *classes_hi = out_hi;
}

#pragma endregion In register vectorized classifier

#pragma region Mask algebra extractor

/** @brief  A 64-bit "class byte == @p value" lane mask over both class halves (two `vpcmpeqb` -> mask_combine). */
SZ_INTERNAL sz_u64_t sz_utf8_word_break_class_mask_haswell_(__m256i classes_lo, __m256i classes_hi, sz_u8_t value) {
    __m256i const v = _mm256_set1_epi8((char)value);
    return sz_utf8_codepoints_mask_combine_haswell_(_mm256_cmpeq_epi8(classes_lo, v), _mm256_cmpeq_epi8(classes_hi, v));
}

/** @brief  A 64-bit "raw window byte == @p value" lane mask over both window halves. */
SZ_INTERNAL sz_u64_t sz_utf8_word_break_byte_equal_haswell_(__m256i low_half, __m256i high_half, sz_u8_t value) {
    __m256i const v = _mm256_set1_epi8((char)value);
    return sz_utf8_codepoints_mask_combine_haswell_(_mm256_cmpeq_epi8(low_half, v), _mm256_cmpeq_epi8(high_half, v));
}

/** @brief  Per-half unsigned `value >= bound` mask (AVX2 has no unsigned compare): `max_epu8(value,bound)==value`. */
SZ_INTERNAL __m256i sz_utf8_word_break_cmpge_epu8_haswell_(__m256i value, __m256i bound) {
    return _mm256_cmpeq_epi8(_mm256_max_epu8(value, bound), value);
}

/** @brief  Per-half "(high,low) 16-bit value in `[lo, hi]`" membership for one range (the AVX2 unsigned 16-bit
 *          window-compare building block of @ref sz_utf8_word_break_range16_mask_haswell_). */
SZ_INTERNAL __m256i sz_utf8_word_break_range16_one_haswell_(__m256i high, __m256i low, sz_u16_t lo, sz_u16_t hi) {
    __m256i const lo_high = _mm256_set1_epi8((char)(lo >> 8)), lo_low = _mm256_set1_epi8((char)(lo & 0xFF));
    __m256i const hi_high = _mm256_set1_epi8((char)(hi >> 8)), hi_low = _mm256_set1_epi8((char)(hi & 0xFF));
    __m256i const ones = _mm256_set1_epi8((char)0xFF);
    __m256i const eq_lo = _mm256_cmpeq_epi8(high, lo_high);
    __m256i const not_below = _mm256_or_si256(
        _mm256_andnot_si256(eq_lo, sz_utf8_word_break_cmpge_epu8_haswell_(high, lo_high)),
        _mm256_and_si256(eq_lo, sz_utf8_word_break_cmpge_epu8_haswell_(low, lo_low)));
    __m256i const high_below_hi = _mm256_andnot_si256(sz_utf8_word_break_cmpge_epu8_haswell_(high, hi_high), ones);
    __m256i const eq_hi = _mm256_cmpeq_epi8(high, hi_high);
    __m256i const not_above = _mm256_or_si256(
        high_below_hi, _mm256_and_si256(eq_hi, sz_utf8_word_break_cmpge_epu8_haswell_(hi_low, low)));
    return _mm256_and_si256(not_below, not_above);
}

/** @brief  A 64-bit "(high,low) 16-bit value in any sorted `[lo, hi]` range" lane mask over both window halves, the
 *          AVX2 twin of @ref sz_utf8_word_break_range16_mask_icelake_ (WSegSpace / Extended_Pictographic). */
SZ_INTERNAL sz_u64_t sz_utf8_word_break_range16_mask_haswell_( //
    __m256i high_lo, __m256i high_hi, __m256i low_lo, __m256i low_hi, sz_u16_t const *lo_table,
    sz_u16_t const *hi_table, int count) {
    __m256i hit_lo = _mm256_setzero_si256(), hit_hi = _mm256_setzero_si256();
    for (int range = 0; range < count; ++range) {
        hit_lo = _mm256_or_si256(
            hit_lo, sz_utf8_word_break_range16_one_haswell_(high_lo, low_lo, lo_table[range], hi_table[range]));
        hit_hi = _mm256_or_si256(
            hit_hi, sz_utf8_word_break_range16_one_haswell_(high_hi, low_hi, lo_table[range], hi_table[range]));
    }
    return sz_utf8_codepoints_mask_combine_haswell_(hit_lo, hit_hi);
}

/**
 *  @brief  Per-ISA extractor: lower one classified 64-byte window to the portable @ref sz_utf8_word_break_frame_t - the
 *          AVX2 twin of @ref sz_utf8_word_break_build_frame_icelake_. Applies the truncated-edge U+FFFD reclassify to
 *          the class halves, materializes every per-class lane mask + the raw-byte membership masks, the
 *          Extended_Pictographic mask (BMP + SMP range scan), and the per-lane class byte array.
 */
SZ_FORCE_INLINE sz_utf8_word_break_frame_t sz_utf8_word_break_build_frame_haswell_(
    sz_utf8_codepoints_window_haswell_t window, __m256i classes_lo, __m256i classes_hi, sz_u64_t start_bytes_all,
    sz_u64_t length_two, sz_u64_t length_three, sz_u64_t length_four, int want_pictographic) {

    sz_size_t const loaded = window.loaded;
    sz_u64_t const valid = sz_utf8_codepoints_mask_until_(loaded);
    sz_u64_t const start_bytes = start_bytes_all & valid;
    __m256i const raw_lo = window.window_lo, raw_hi = window.window_hi;

    // Truncated-edge U+FFFD reclassify (force the class to Other on a lead whose declared span runs past `loaded`).
    sz_u64_t const lead_two = length_two & start_bytes;
    sz_u64_t const lead_three = length_three & start_bytes;
    sz_u64_t const lead_four = length_four & start_bytes;
    sz_u64_t const truncated_raw = ((lead_two & ~sz_utf8_codepoints_mask_until_(loaded > 1 ? loaded - 1 : 0)) |
                                    (lead_three & ~sz_utf8_codepoints_mask_until_(loaded > 2 ? loaded - 2 : 0)) |
                                    (lead_four & ~sz_utf8_codepoints_mask_until_(loaded > 3 ? loaded - 3 : 0))) &
                                   valid;
    if (truncated_raw) {
        __m256i const other = _mm256_set1_epi8((char)sz_utf8_word_break_other_k);
        __m256i const sel_lo = sz_utf8_codepoints_byte_mask_from_bits_haswell_((sz_u32_t)truncated_raw);
        __m256i const sel_hi = sz_utf8_codepoints_byte_mask_from_bits_haswell_((sz_u32_t)(truncated_raw >> 32));
        classes_lo = _mm256_blendv_epi8(classes_lo, other, sel_lo);
        classes_hi = _mm256_blendv_epi8(classes_hi, other, sel_hi);
    }

    sz_utf8_word_break_frame_t frame;
    frame.class_aletter = sz_utf8_word_break_class_mask_haswell_(classes_lo, classes_hi, sz_utf8_word_break_aletter_k);
    frame.class_hebrew = sz_utf8_word_break_class_mask_haswell_(classes_lo, classes_hi,
                                                                sz_utf8_word_break_hebrew_letter_k);
    frame.class_numeric = sz_utf8_word_break_class_mask_haswell_(classes_lo, classes_hi, sz_utf8_word_break_numeric_k);
    frame.class_katakana = sz_utf8_word_break_class_mask_haswell_(classes_lo, classes_hi,
                                                                  sz_utf8_word_break_katakana_k);
    frame.class_extendnumlet = sz_utf8_word_break_class_mask_haswell_(classes_lo, classes_hi,
                                                                      sz_utf8_word_break_extendnumlet_k);
    frame.class_extend = sz_utf8_word_break_class_mask_haswell_(classes_lo, classes_hi, sz_utf8_word_break_extend_k);
    frame.class_zwj = sz_utf8_word_break_class_mask_haswell_(classes_lo, classes_hi, sz_utf8_word_break_zwj_k);
    frame.class_format = sz_utf8_word_break_class_mask_haswell_(classes_lo, classes_hi, sz_utf8_word_break_format_k);
    frame.class_midletter = sz_utf8_word_break_class_mask_haswell_(classes_lo, classes_hi,
                                                                   sz_utf8_word_break_midletter_k);
    frame.class_midnum = sz_utf8_word_break_class_mask_haswell_(classes_lo, classes_hi, sz_utf8_word_break_midnum_k);
    frame.class_mid_quotes = sz_utf8_word_break_class_mask_haswell_(classes_lo, classes_hi,
                                                                    sz_utf8_word_break_mid_quotes_k);
    frame.class_cr = sz_utf8_word_break_class_mask_haswell_(classes_lo, classes_hi, sz_utf8_word_break_cr_k);
    frame.class_lf = sz_utf8_word_break_class_mask_haswell_(classes_lo, classes_hi, sz_utf8_word_break_lf_k);
    frame.class_newline = sz_utf8_word_break_class_mask_haswell_(classes_lo, classes_hi, sz_utf8_word_break_newline_k);
    frame.class_regional = sz_utf8_word_break_class_mask_haswell_(classes_lo, classes_hi,
                                                                  sz_utf8_word_break_regional_ind_k);

    sz_u64_t const non_ascii_lanes = sz_utf8_codepoints_mask_combine_haswell_(raw_lo, raw_hi) & valid;
    frame.non_ascii_lanes = non_ascii_lanes;
    frame.double_quote_byte = sz_utf8_word_break_byte_equal_haswell_(raw_lo, raw_hi, 0x22) & valid;
    frame.single_quote_byte = sz_utf8_word_break_byte_equal_haswell_(raw_lo, raw_hi, 0x27) & valid;

    // WB3d WSegSpace raw membership: the ASCII U+0020 byte compare OR the multibyte (high,low) range scan.
    sz_u64_t wseg_multibyte = 0ull;
    if (non_ascii_lanes)
        wseg_multibyte = sz_utf8_word_break_range16_mask_haswell_(
                             window.high_lo, window.high_hi, window.low_lo, window.low_hi, sz_utf8_word_break_wseg_lo_,
                             sz_utf8_word_break_wseg_hi_, sz_utf8_word_break_wseg_count_k) &
                         non_ascii_lanes;
    frame.wseg = (wseg_multibyte | (sz_utf8_word_break_byte_equal_haswell_(raw_lo, raw_hi, 0x20) & valid));

    // WB3c Extended_Pictographic raw membership (BMP range scan on non-4-byte lanes, SMP range scan on plane-one
    // 4-byte lanes). Rare-class gated on `want_pictographic` (an in-window ZWJ or the carried `prev_ends_in_zwj`), so
    // the ~156-range scan is skipped on the common no-ZWJ window. The engine applies the final gating.
    frame.pictographic = 0ull;
    sz_u64_t const four_byte = window.four_byte_starts & valid;
    if (want_pictographic) {
        __m256i next1_lo, next1_hi, next2_lo, next2_hi, next3_lo, next3_hi;
        sz_utf8_codepoints_forward_neighbours_haswell_(raw_lo, raw_hi, &next1_lo, &next1_hi, &next2_lo, &next2_hi);
        __m256i const low_successor = _mm256_permute2x128_si256(raw_lo, raw_hi, 0x21);
        next3_lo = _mm256_alignr_epi8(low_successor, raw_lo, 3);
        __m256i const high_successor = _mm256_permute2x128_si256(raw_hi, raw_lo, 0x21);
        next3_hi = _mm256_alignr_epi8(high_successor, raw_hi, 3);
        __m256i const plane_lo = _mm256_or_si256(
            _mm256_and_si256(_mm256_slli_epi16(_mm256_and_si256(raw_lo, _mm256_set1_epi8(0x07)), 2),
                             _mm256_set1_epi8(0x1C)),
            sz_utf8_codepoints_srl8_haswell_(next1_lo, 4, 0x03));
        __m256i const plane_hi = _mm256_or_si256(
            _mm256_and_si256(_mm256_slli_epi16(_mm256_and_si256(raw_hi, _mm256_set1_epi8(0x07)), 2),
                             _mm256_set1_epi8(0x1C)),
            sz_utf8_codepoints_srl8_haswell_(next1_hi, 4, 0x03));
        __m256i const one = _mm256_set1_epi8(1);
        sz_u64_t const plane_one = sz_utf8_codepoints_mask_combine_haswell_(_mm256_cmpeq_epi8(plane_lo, one),
                                                                            _mm256_cmpeq_epi8(plane_hi, one));
        __m256i const smp_high_lo = _mm256_or_si256(
            _mm256_and_si256(_mm256_slli_epi16(_mm256_and_si256(next1_lo, _mm256_set1_epi8(0x0F)), 4),
                             _mm256_set1_epi8((char)0xF0)),
            sz_utf8_codepoints_srl8_haswell_(next2_lo, 2, 0x0F));
        __m256i const smp_high_hi = _mm256_or_si256(
            _mm256_and_si256(_mm256_slli_epi16(_mm256_and_si256(next1_hi, _mm256_set1_epi8(0x0F)), 4),
                             _mm256_set1_epi8((char)0xF0)),
            sz_utf8_codepoints_srl8_haswell_(next2_hi, 2, 0x0F));
        __m256i const smp_low_lo = _mm256_or_si256(
            _mm256_and_si256(_mm256_slli_epi16(_mm256_and_si256(next2_lo, _mm256_set1_epi8(0x03)), 6),
                             _mm256_set1_epi8((char)0xC0)),
            _mm256_and_si256(next3_lo, _mm256_set1_epi8(0x3F)));
        __m256i const smp_low_hi = _mm256_or_si256(
            _mm256_and_si256(_mm256_slli_epi16(_mm256_and_si256(next2_hi, _mm256_set1_epi8(0x03)), 6),
                             _mm256_set1_epi8((char)0xC0)),
            _mm256_and_si256(next3_hi, _mm256_set1_epi8(0x3F)));
        sz_u64_t const pictographic_bmp = sz_utf8_word_break_range16_mask_haswell_(
            window.high_lo, window.high_hi, window.low_lo, window.low_hi, sz_utf8_word_break_pict_bmp_lo_,
            sz_utf8_word_break_pict_bmp_hi_, sz_utf8_word_break_pict_bmp_count_k);
        sz_u64_t const pictographic_smp = sz_utf8_word_break_range16_mask_haswell_(
            smp_high_lo, smp_high_hi, smp_low_lo, smp_low_hi, sz_utf8_word_break_pict_smp_lo_,
            sz_utf8_word_break_pict_smp_hi_, sz_utf8_word_break_pict_smp_count_k);
        frame.pictographic = (pictographic_bmp & non_ascii_lanes & ~four_byte) |
                             (pictographic_smp & four_byte & plane_one);
    }

    _mm256_storeu_si256((__m256i *)(frame.classes_byte + 0), classes_lo);
    _mm256_storeu_si256((__m256i *)(frame.classes_byte + 32), classes_hi);
    return frame;
}

#pragma endregion Mask algebra extractor

#pragma region Codepoint partition

/** @brief  Resolve one window into the maximal-subpart partition - the AVX2 twin of
 *          @ref sz_utf8_word_break_partition_icelake_: compute the per-ISA `sz_u64_t` masks and delegate to the
 *          portable @ref sz_utf8_word_break_partition_from_masks_. */
SZ_INTERNAL sz_utf8_word_break_partition_t sz_utf8_word_break_partition_haswell_(
    sz_utf8_codepoints_window_haswell_t window, sz_u64_t valid, int at_end_of_text) {
    __m256i const raw_lo = window.window_lo, raw_hi = window.window_hi;
    sz_u64_t const real_continuation = window.continuation & valid;
    __m256i const high_nibble_lo = sz_utf8_codepoints_srl8_haswell_(raw_lo, 4, 0x0F);
    __m256i const high_nibble_hi = sz_utf8_codepoints_srl8_haswell_(raw_hi, 4, 0x0F);
    sz_u64_t const length_two = (sz_utf8_codepoints_mask_combine_haswell_(
                                    _mm256_or_si256(_mm256_cmpeq_epi8(high_nibble_lo, _mm256_set1_epi8(0x0C)),
                                                    _mm256_cmpeq_epi8(high_nibble_lo, _mm256_set1_epi8(0x0D))),
                                    _mm256_or_si256(_mm256_cmpeq_epi8(high_nibble_hi, _mm256_set1_epi8(0x0C)),
                                                    _mm256_cmpeq_epi8(high_nibble_hi, _mm256_set1_epi8(0x0D))))) &
                                valid;
    sz_u64_t const length_three = sz_utf8_codepoints_mask_combine_haswell_(
                                      _mm256_cmpeq_epi8(high_nibble_lo, _mm256_set1_epi8(0x0E)),
                                      _mm256_cmpeq_epi8(high_nibble_hi, _mm256_set1_epi8(0x0E))) &
                                  valid;
    sz_u64_t const length_four = sz_utf8_codepoints_mask_combine_haswell_(
                                     _mm256_cmpeq_epi8(high_nibble_lo, _mm256_set1_epi8(0x0F)),
                                     _mm256_cmpeq_epi8(high_nibble_hi, _mm256_set1_epi8(0x0F))) &
                                 valid;
    sz_u64_t const length_ge_two = length_two | length_three | length_four;
    sz_u64_t bad_second_byte = 0ull;
    if (length_ge_two) {
        __m256i next1_lo, next1_hi, next2_lo, next2_hi;
        sz_utf8_codepoints_forward_neighbours_haswell_(raw_lo, raw_hi, &next1_lo, &next1_hi, &next2_lo, &next2_hi);
        __m256i const a0 = _mm256_set1_epi8((char)0xA0), b90 = _mm256_set1_epi8((char)0x90);
        sz_u64_t const next1_at_least_a0 = sz_utf8_codepoints_mask_combine_haswell_(
            sz_utf8_word_break_cmpge_epu8_haswell_(next1_lo, a0), sz_utf8_word_break_cmpge_epu8_haswell_(next1_hi, a0));
        sz_u64_t const next1_at_least_90 = sz_utf8_codepoints_mask_combine_haswell_(
            sz_utf8_word_break_cmpge_epu8_haswell_(next1_lo, b90),
            sz_utf8_word_break_cmpge_epu8_haswell_(next1_hi, b90));
        sz_u64_t const lead_c0_c1 = (sz_utf8_word_break_byte_equal_haswell_(raw_lo, raw_hi, 0xC0) |
                                     sz_utf8_word_break_byte_equal_haswell_(raw_lo, raw_hi, 0xC1)) &
                                    valid;
        sz_u64_t const lead_e0 = sz_utf8_word_break_byte_equal_haswell_(raw_lo, raw_hi, 0xE0) & valid;
        sz_u64_t const lead_ed = sz_utf8_word_break_byte_equal_haswell_(raw_lo, raw_hi, 0xED) & valid;
        sz_u64_t const lead_f0 = sz_utf8_word_break_byte_equal_haswell_(raw_lo, raw_hi, 0xF0) & valid;
        sz_u64_t const lead_f4 = sz_utf8_word_break_byte_equal_haswell_(raw_lo, raw_hi, 0xF4) & valid;
        __m256i const f5 = _mm256_set1_epi8((char)0xF5);
        sz_u64_t const lead_f5_or_more = sz_utf8_codepoints_mask_combine_haswell_(
                                             sz_utf8_word_break_cmpge_epu8_haswell_(raw_lo, f5),
                                             sz_utf8_word_break_cmpge_epu8_haswell_(raw_hi, f5)) &
                                         valid;
        bad_second_byte = lead_c0_c1 | (lead_e0 & ~next1_at_least_a0) | (lead_ed & next1_at_least_a0) |
                          (lead_f0 & ~next1_at_least_90) | (lead_f4 & next1_at_least_90) | lead_f5_or_more;
    }
    return sz_utf8_word_break_partition_from_masks_(real_continuation, length_two, length_three, length_four,
                                                    bad_second_byte, valid, at_end_of_text);
}

#pragma endregion Codepoint partition

#pragma region Forward driver

/**
 *  @brief  Forward UAX-29 word segmentation over `[0, length)` (Haswell AVX2): the overlap-free advancing driver,
 *          mirroring @ref sz_utf8_words_icelake over the AVX2 window/classify/partition/decide/drain leaves. Bit-exact
 *          with `sz_utf8_words_serial` and `sz_utf8_words_icelake`.
 */
SZ_PUBLIC sz_size_t sz_utf8_words_haswell(           //
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
        sz_utf8_codepoints_window_haswell_t const window = sz_utf8_codepoints_decode_window_haswell_(text_u8 + position,
                                                                                                     length - position);
        sz_size_t const loaded = window.loaded;
        sz_u64_t const valid = sz_utf8_codepoints_mask_until_(loaded);

        __m256i classes_lo, classes_hi;
        sz_utf8_word_break_classify_window_haswell_(window, &classes_lo, &classes_hi);

        sz_utf8_word_break_partition_t const partition = sz_utf8_word_break_partition_haswell_(
            window, valid, position + loaded >= length);
        sz_u64_t const start_bytes_all = partition.start_bytes;
        sz_u64_t const continuation_all = partition.continuation;
        sz_u64_t const forced_other = partition.forced_other;
        sz_u64_t const length_two = partition.length_two;
        sz_u64_t const length_three = partition.length_three;
        sz_u64_t const length_four = partition.length_four;
        if (forced_other) {
            __m256i const other = _mm256_set1_epi8((char)sz_utf8_word_break_other_k);
            __m256i const sel_lo = sz_utf8_codepoints_byte_mask_from_bits_haswell_((sz_u32_t)forced_other);
            __m256i const sel_hi = sz_utf8_codepoints_byte_mask_from_bits_haswell_((sz_u32_t)(forced_other >> 32));
            classes_lo = _mm256_blendv_epi8(classes_lo, other, sel_lo);
            classes_hi = _mm256_blendv_epi8(classes_hi, other, sel_hi);
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

        int const want_pictographic = sz_utf8_word_break_class_mask_haswell_(classes_lo, classes_hi,
                                                                             sz_utf8_word_break_zwj_k) != 0 ||
                                      carry.prev_ends_in_zwj;
        sz_utf8_word_break_frame_t const frame = sz_utf8_word_break_build_frame_haswell_(
            window, classes_lo, classes_hi, start_bytes_all, length_two, length_three, length_four, want_pictographic);

        sz_utf8_word_break_carry_t carry_full = carry;
        sz_utf8_word_break_window_t const win = sz_utf8_word_break_decide_window_(
            &frame, start_bytes_all, continuation_all, forced_other, length_two, length_three, length_four,
            complete_limit, &carry_full, more_text);

        sz_size_t const adv = win.resolved;
        sz_u64_t const boundary_lanes = win.breaks & sz_utf8_codepoints_mask_until_(adv);

        words = sz_utf8_codepoints_drain_forward_haswell_(boundary_lanes, position, word_starts, word_lengths, words,
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
#endif // SZ_USE_HASWELL

#ifdef __cplusplus
}
#endif

#endif // STRINGZILLA_UTF8_WORDS_HASWELL_H_
