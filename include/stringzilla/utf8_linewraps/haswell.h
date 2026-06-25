/**
 *  @brief Haswell (AVX2) backend for UAX-14 line break boundaries.
 *  @file include/stringzilla/utf8_linewraps/haswell.h
 *  @author Ash Vardanian
 */
#ifndef STRINGZILLA_UTF8_LINEWRAPS_HASWELL_H_
#define STRINGZILLA_UTF8_LINEWRAPS_HASWELL_H_

#include "stringzilla/types.h"
#include "stringzilla/utf8_linewraps/tables.h"
#include "stringzilla/utf8_linewraps/serial.h"
#include "stringzilla/utf8_runes/haswell.h"

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

#pragma region UAX 14 Line Boundaries forward kernel

#pragma region In register vectorized classifier

/*  The AVX2 twin of the Ice Lake classifier: a contiguous run of codepoints resolves to per-codepoint
 *  (class, side, dotted) with ZERO per-lane scalar loop, ZERO `vpgather`, and NO serial deferral. Each 64-byte
 *  window lives as two `__m256i` halves (`_lo` = lanes [0,32), `_hi` = lanes [32,64)); every per-lane class
 *  compare is `_mm256_cmpeq_epi8` on both halves OR-combined to a `sz_u64_t` via `mask_combine_haswell_`. The
 *  BMP palette index comes from the `sz_line_break_bmp_index_haswell_` nibble cascade (the AVX2 twin of the
 *  VBMI full-BMP trie); the astral path uses `sz_line_break_classify_astral_haswell_`. The 62-entry palette
 *  descriptor is unpacked to the LB1-resolved class byte and the engine side byte by `lut256_haswell_` reads of
 *  the precomputed palette tables, bit-identical to the icelake descriptor unpack. */

/** @brief Palette index for thirty-two BMP codepoints (per-lane high = cp>>8, low = cp&0xFF) via a register-resident
 *         3-stage `vpshufb` nibble cascade, the AVX2 twin of the VBMI full-BMP trie (`sz_line_break_bmp_full_index_icelake_`).
 *         Gather-free; bit-exact with `sz_rune_line_break_property` over the whole BMP. */
SZ_INTERNAL __m256i sz_line_break_bmp_index_haswell_(__m256i high, __m256i low) {
    __m256i const low_nibble_mask = _mm256_set1_epi8(0x0F);
    __m256i const high_high = _mm256_and_si256(_mm256_srli_epi16(high, 4), low_nibble_mask);
    __m256i const high_low = _mm256_and_si256(high, low_nibble_mask);
    __m256i const page = sz_utf8_rune_cascade_stage_haswell_(
        sz_utf8_line_break_haswell_stage1_, sz_utf8_line_break_haswell_stage1_count_k / 16, high_high, high_low);
    __m256i const low_high = _mm256_and_si256(_mm256_srli_epi16(low, 4), low_nibble_mask);
    __m256i const leaf_lo = sz_utf8_rune_cascade_stage_haswell_(
        sz_utf8_line_break_haswell_stage2_lo_, sz_utf8_line_break_haswell_stage2_lo_count_k / 16, page, low_high);
    __m256i const leaf_hi = sz_utf8_rune_cascade_stage_haswell_(
        sz_utf8_line_break_haswell_stage2_hi_, sz_utf8_line_break_haswell_stage2_hi_count_k / 16, page, low_high);
    __m256i const leaf_group = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(leaf_lo, 4), low_nibble_mask),
                                               _mm256_slli_epi16(leaf_hi, 4));
    __m256i const leaf_low_nibble = _mm256_and_si256(leaf_lo, low_nibble_mask);
    __m256i const low_low = _mm256_and_si256(low, low_nibble_mask);
    __m256i const lut_index = _mm256_or_si256(_mm256_slli_epi16(leaf_low_nibble, 4), low_low);
    __m256i result = _mm256_setzero_si256();
    for (int group = 0; group < (int)sz_utf8_line_break_haswell_leaf_groups_k; ++group) {
        __m256i const value = sz_utf8_rune_lut256_haswell_(sz_utf8_line_break_haswell_stage3_groups_ + group * 256,
                                                           lut_index);
        __m256i const here = _mm256_cmpeq_epi8(leaf_group, _mm256_set1_epi8((char)group));
        result = _mm256_blendv_epi8(result, value, here);
    }
    return result;
}

/** @brief Palette index for thirty-two ASTRAL codepoints over the 20-bit offset = cp - 0x10000 (5-nibble cascade),
 *         the AVX2 twin of `sz_line_break_classify_astral16_icelake_`. Per-lane bytes: @p plane = (offset>>16)&0xFF
 *         (low nibble meaningful), @p high = (offset>>8)&0xFF, @p low = offset&0xFF. Gather-free; bit-exact. */
SZ_INTERNAL __m256i sz_line_break_classify_astral_haswell_(__m256i plane, __m256i high, __m256i low) {
    __m256i const low_nibble_mask = _mm256_set1_epi8(0x0F);
    __m256i const n4 = _mm256_and_si256(plane, low_nibble_mask);
    __m256i const n3 = _mm256_and_si256(_mm256_srli_epi16(high, 4), low_nibble_mask);
    __m256i const stage1_index = _mm256_or_si256(_mm256_slli_epi16(n4, 4), n3);
    __m256i const page = sz_utf8_rune_lut256_haswell_(sz_utf8_line_break_haswell_astral_stage1_, stage1_index);
    __m256i const n2 = _mm256_and_si256(high, low_nibble_mask);
    __m256i const leaf2_lo = sz_utf8_rune_cascade_stage_haswell_(
        sz_utf8_line_break_haswell_astral_stage2_lo_, sz_utf8_line_break_haswell_astral_stage2_lo_count_k / 16, page,
        n2);
    __m256i const n1 = _mm256_and_si256(_mm256_srli_epi16(low, 4), low_nibble_mask);
    __m256i const leaf_lo = sz_utf8_rune_cascade_stage_haswell_(
        sz_utf8_line_break_haswell_astral_stage3_lo_, sz_utf8_line_break_haswell_astral_stage3_lo_count_k / 16,
        leaf2_lo, n1);
    __m256i const leaf_hi = sz_utf8_rune_cascade_stage_haswell_(
        sz_utf8_line_break_haswell_astral_stage3_hi_, sz_utf8_line_break_haswell_astral_stage3_hi_count_k / 16,
        leaf2_lo, n1);
    __m256i const n0 = _mm256_and_si256(low, low_nibble_mask);
    __m256i const leaf_group = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(leaf_lo, 4), low_nibble_mask),
                                               _mm256_slli_epi16(leaf_hi, 4));
    __m256i const leaf_low_nibble = _mm256_and_si256(leaf_lo, low_nibble_mask);
    __m256i const stage4_lut_index = _mm256_or_si256(_mm256_slli_epi16(leaf_low_nibble, 4), n0);
    __m256i result = _mm256_setzero_si256();
    for (int group = 0; group < (int)sz_utf8_line_break_haswell_astral_leaf_groups_k; ++group) {
        __m256i const value = sz_utf8_rune_lut256_haswell_(
            sz_utf8_line_break_haswell_astral_stage4_groups_ + group * 256, stage4_lut_index);
        __m256i const here = _mm256_cmpeq_epi8(leaf_group, _mm256_set1_epi8((char)group));
        result = _mm256_blendv_epi8(result, value, here);
    }
    return result;
}

/** @brief Per-half unsigned `value >= bound` mask (AVX2 has no unsigned compare): `max_epu8(value,bound)==value`. */
SZ_INTERNAL __m256i sz_line_break_cmpge_epu8_haswell_(__m256i value, __m256i bound) {
    return _mm256_cmpeq_epi8(_mm256_max_epu8(value, bound), value);
}

/** @brief Per-half unsigned `value < bound` mask: the complement of `value >= bound` (`min_epu8` form keeps it
 *         branch- and bias-free): `min_epu8(value,bound)==value && value!=bound` is awkward, so use the >= negation
 *         via `min_epu8(value, bound-1)==value` is fragile at bound==0; instead AND-NOT the >= mask against ones. */
SZ_INTERNAL __m256i sz_line_break_cmplt_epu8_haswell_(__m256i value, __m256i bound) {
    return _mm256_andnot_si256(sz_line_break_cmpge_epu8_haswell_(value, bound), _mm256_set1_epi8((char)0xFF));
}

/** @brief A 64-bit "(byte & mask) == pattern" lane mask over both window halves (two `vpand`+`vpcmpeqb`). */
SZ_INTERNAL sz_u64_t sz_line_break_byte_match_haswell_(__m256i low_half, __m256i high_half, sz_u8_t mask,
                                                       sz_u8_t pattern) {
    __m256i const m = _mm256_set1_epi8((char)mask), p = _mm256_set1_epi8((char)pattern);
    return sz_utf8_mask_combine_haswell_(_mm256_cmpeq_epi8(_mm256_and_si256(low_half, m), p),
                                         _mm256_cmpeq_epi8(_mm256_and_si256(high_half, m), p));
}

/** @brief A 64-bit "byte == value" lane mask over both window halves. */
SZ_INTERNAL sz_u64_t sz_line_break_byte_equal_haswell_(__m256i low_half, __m256i high_half, sz_u8_t value) {
    __m256i const v = _mm256_set1_epi8((char)value);
    return sz_utf8_mask_combine_haswell_(_mm256_cmpeq_epi8(low_half, v), _mm256_cmpeq_epi8(high_half, v));
}

/** @brief A 64-bit "byte >= bound" (unsigned) lane mask over both window halves. */
SZ_INTERNAL sz_u64_t sz_line_break_byte_ge_haswell_(__m256i low_half, __m256i high_half, sz_u8_t bound) {
    __m256i const bound_vec = _mm256_set1_epi8((char)bound);
    return sz_utf8_mask_combine_haswell_(sz_line_break_cmpge_epu8_haswell_(low_half, bound_vec),
                                         sz_line_break_cmpge_epu8_haswell_(high_half, bound_vec));
}

/** @brief A 64-bit "byte < bound" (unsigned) lane mask over both window halves. */
SZ_INTERNAL sz_u64_t sz_line_break_byte_lt_haswell_(__m256i low_half, __m256i high_half, sz_u8_t bound) {
    __m256i const bound_vec = _mm256_set1_epi8((char)bound);
    return sz_utf8_mask_combine_haswell_(sz_line_break_cmplt_epu8_haswell_(low_half, bound_vec),
                                         sz_line_break_cmplt_epu8_haswell_(high_half, bound_vec));
}

/** @brief The class/side byte held at byte-lane @p lane, extracted in-register (no scalar window store). */
SZ_INTERNAL sz_u8_t sz_line_break_byte_at_haswell_(__m256i lanes_lo, __m256i lanes_hi, sz_size_t lane) {
    __m256i const source = lane < 32 ? lanes_lo : lanes_hi;
    sz_size_t const within = lane < 32 ? lane : lane - 32;
    sz_u8_t bytes[32];
    _mm256_storeu_si256((__m256i *)bytes, source);
    return bytes[within];
}

/** @brief Per-window byte-lane classification (AVX2): class/side per lane as two `__m256i` halves plus the
 *         effective-start and U+FFFD masks. The Haswell twin of @ref sz_line_break_classified_t. */
typedef struct sz_line_break_classified_haswell_t {
    __m256i classes_lo;   /**< Per-byte-lane Line_Break class, lanes [0,32) (valid only on `starts` lanes). */
    __m256i classes_hi;   /**< Per-byte-lane Line_Break class, lanes [32,64). */
    __m256i side_lo;      /**< Per-byte-lane engine side byte, lanes [0,32). */
    __m256i side_hi;      /**< Per-byte-lane engine side byte, lanes [32,64). */
    sz_u64_t dotted;      /**< Bit i set => lane i is DottedCircle U+25CC. */
    sz_u64_t starts;      /**< Effective codepoint starts: valid leads (at their lane) + 1-byte U+FFFD units. */
    sz_u64_t replacement; /**< Effective-start lanes that are ill-formed (decoded as U+FFFD, class AL). */
    sz_u64_t non_start;   /**< Bytes that are NOT effective starts (consumed continuations) within `loaded`. */
    sz_size_t loaded;     /**< Bytes loaded into this window (<= 64). */
} sz_line_break_classified_haswell_t;

/** @brief Compute the third forward neighbour `next3[i] = window[i+3]` over all 64 lanes with mod-64 wrap, the
 *         AVX2 twin of icelake's `_mm512_permutexvar_epi8(lane_identity+3)`. Same idiom as the substrate
 *         `forward_neighbours_haswell_` (permute2x128 to bring the successor 128-bit block in, alignr by 3). */
SZ_INTERNAL void sz_line_break_next3_haswell_(__m256i window_lo, __m256i window_hi, __m256i *next3_lo,
                                              __m256i *next3_hi) {
    __m256i const low_successor = _mm256_permute2x128_si256(window_lo, window_hi, 0x21);
    *next3_lo = _mm256_alignr_epi8(low_successor, window_lo, 3);
    __m256i const high_successor = _mm256_permute2x128_si256(window_hi, window_lo, 0x21);
    *next3_hi = _mm256_alignr_epi8(high_successor, window_hi, 3);
}

/** @brief Resolve the per-lane palette index (one half) to class / side / dotted bytes through the precomputed
 *         62-entry palette tables. `lut256_haswell_` reads each 64-entry table by `index` (index < 64 so only the
 *         low four rows are selected); bit-identical to the icelake `vpermb` palette permute. */
SZ_INTERNAL void sz_line_break_palette_unpack_haswell_(__m256i index, __m256i *classes, __m256i *side,
                                                       __m256i *dotted) {
    *classes = sz_utf8_rune_lut256_haswell_(sz_utf8_line_break_palette_class_, index);
    *side = sz_utf8_rune_lut256_haswell_(sz_utf8_line_break_palette_side_, index);
    *dotted = sz_utf8_rune_lut256_haswell_(sz_utf8_line_break_palette_dotted_, index);
}

/**
 *  @brief  Classify a decoded 64-byte window onto byte-start lanes, fully in-register and zero-scalar — the AVX2
 *          twin of @ref sz_line_break_classify_window_icelake_, bit-identical on every lane. Reproduces the serial
 *          "consume-1 U+FFFD" malformed policy: an invalid lead / short or stray continuation / overlong /
 *          surrogate / out-of-range lead each become one single-byte U+FFFD unit (class AL).
 */
SZ_INTERNAL sz_line_break_classified_haswell_t sz_line_break_classify_window_haswell_(
    sz_utf8_rune_window_haswell_t window) {
    sz_u64_t const loaded_mask = sz_u64_mask_until_serial_(window.loaded);
    sz_u64_t const continuation = window.continuation & loaded_mask;
    sz_u64_t const two_byte = window.two_byte_starts;
    sz_u64_t const three_byte = window.three_byte_starts;
    sz_u64_t const four_byte = window.four_byte_starts;
    __m256i const raw_lo = window.window_lo, raw_hi = window.window_hi;

    //  Forward neighbours (mod-64 wrap, matching icelake's `_mm512_permutexvar_epi8`).
    __m256i next1_lo, next1_hi, next2_lo, next2_hi, next3_lo, next3_hi;
    sz_utf8_forward_neighbours_haswell_(raw_lo, raw_hi, &next1_lo, &next1_hi, &next2_lo, &next2_hi);
    sz_line_break_next3_haswell_(raw_lo, raw_hi, &next3_lo, &next3_hi);

    sz_u64_t const next1_continuation = continuation >> 1, next2_continuation = continuation >> 2,
                   next3_continuation = continuation >> 3;

    //  Ill-formed-lead gate (LEVER B), bit-identical to icelake: only C0/C1, E0, ED, F0, F4/>=F5 leads can be
    //  overlong / surrogate / out-of-range; detect their presence with raw-only tests and, when absent, take the
    //  cheap "lead + enough continuations" validity path.
    sz_u64_t const lead_c0_c1 = sz_line_break_byte_ge_haswell_(raw_lo, raw_hi, 0xC0) &
                                sz_line_break_byte_lt_haswell_(raw_lo, raw_hi, 0xC2);
    sz_u64_t const lead_e0 = sz_line_break_byte_equal_haswell_(raw_lo, raw_hi, 0xE0);
    sz_u64_t const lead_ed = sz_line_break_byte_equal_haswell_(raw_lo, raw_hi, 0xED);
    sz_u64_t const lead_f0 = sz_line_break_byte_equal_haswell_(raw_lo, raw_hi, 0xF0);
    sz_u64_t const lead_f4_or_above = sz_line_break_byte_ge_haswell_(raw_lo, raw_hi, 0xF4);
    sz_u64_t const danger_leads = (lead_c0_c1 | lead_e0 | lead_ed | lead_f0 | lead_f4_or_above) & loaded_mask;

    sz_u64_t valid2, valid3, valid4;
    if (!danger_leads) {
        valid2 = two_byte & next1_continuation;
        valid3 = three_byte & next1_continuation & next2_continuation;
        valid4 = four_byte & next1_continuation & next2_continuation & next3_continuation;
    }
    else {
        //  Exact prior algebra: per-lane value-bit predicates for overlong / surrogate / out-of-range detection.
        sz_u64_t const lead_not_overlong2 = ~sz_line_break_byte_match_haswell_(raw_lo, raw_hi, 0x1E,
                                                                               0x00); // (raw & 0x1E) != 0
        sz_u64_t const b0_e0 = lead_e0, b0_ed = lead_ed, b0_f0 = lead_f0;
        sz_u64_t const b0_f4 = sz_line_break_byte_equal_haswell_(raw_lo, raw_hi, 0xF4);
        sz_u64_t const b0_above_f4 = sz_line_break_byte_ge_haswell_(raw_lo, raw_hi, 0xF5);
        sz_u64_t const b1_lt_a0 = sz_line_break_byte_lt_haswell_(next1_lo, next1_hi, 0xA0);
        sz_u64_t const b1_ge_a0 = sz_line_break_byte_ge_haswell_(next1_lo, next1_hi, 0xA0);
        sz_u64_t const b1_lt_90 = sz_line_break_byte_lt_haswell_(next1_lo, next1_hi, 0x90);
        sz_u64_t const b1_ge_90 = sz_line_break_byte_ge_haswell_(next1_lo, next1_hi, 0x90);

        sz_u64_t const overlong3 = three_byte & b0_e0 & b1_lt_a0;
        sz_u64_t const surrogate3 = three_byte & b0_ed & b1_ge_a0;
        sz_u64_t const overlong4 = four_byte & b0_f0 & b1_lt_90;
        sz_u64_t const above4 = four_byte & ((b0_f4 & b1_ge_90) | b0_above_f4);
        valid2 = two_byte & next1_continuation & lead_not_overlong2;
        valid3 = three_byte & next1_continuation & next2_continuation & ~overlong3 & ~surrogate3;
        valid4 = four_byte & next1_continuation & next2_continuation & next3_continuation & ~overlong4 & ~above4;
    }
    sz_u64_t const true_ascii = sz_line_break_byte_lt_haswell_(raw_lo, raw_hi, 0x80) & loaded_mask & ~continuation;

    //  Rebuild high/low per lead length so cp = (plane<<16)|(high<<8)|low is exact (decode_window_ uses the
    //  2/3-byte formula only). Replacement lanes are overridden downstream, so their garbage never reaches the
    //  palette. low4/high4 mirror the icelake 4-byte reconstruction.
    __m256i const low_four_lo = _mm256_or_si256(
        _mm256_and_si256(_mm256_slli_epi16(_mm256_and_si256(next2_lo, _mm256_set1_epi8(0x03)), 6),
                         _mm256_set1_epi8((char)0xC0)),
        _mm256_and_si256(next3_lo, _mm256_set1_epi8(0x3F)));
    __m256i const low_four_hi = _mm256_or_si256(
        _mm256_and_si256(_mm256_slli_epi16(_mm256_and_si256(next2_hi, _mm256_set1_epi8(0x03)), 6),
                         _mm256_set1_epi8((char)0xC0)),
        _mm256_and_si256(next3_hi, _mm256_set1_epi8(0x3F)));
    __m256i const high_four_lo = _mm256_or_si256(
        _mm256_and_si256(_mm256_slli_epi16(_mm256_and_si256(next1_lo, _mm256_set1_epi8(0x0F)), 4),
                         _mm256_set1_epi8((char)0xF0)),
        sz_utf8_srl8_haswell_(next2_lo, 2, 0x0F));
    __m256i const high_four_hi = _mm256_or_si256(
        _mm256_and_si256(_mm256_slli_epi16(_mm256_and_si256(next1_hi, _mm256_set1_epi8(0x0F)), 4),
                         _mm256_set1_epi8((char)0xF0)),
        sz_utf8_srl8_haswell_(next2_hi, 2, 0x0F));

    __m256i const ascii_select_lo = sz_utf8_byte_mask_from_bits_haswell_((sz_u32_t)true_ascii);
    __m256i const ascii_select_hi = sz_utf8_byte_mask_from_bits_haswell_((sz_u32_t)(true_ascii >> 32));
    __m256i const four_select_lo = sz_utf8_byte_mask_from_bits_haswell_((sz_u32_t)four_byte);
    __m256i const four_select_hi = sz_utf8_byte_mask_from_bits_haswell_((sz_u32_t)(four_byte >> 32));

    __m256i low_lo = _mm256_blendv_epi8(window.low_lo, raw_lo, ascii_select_lo);
    __m256i low_hi = _mm256_blendv_epi8(window.low_hi, raw_hi, ascii_select_hi);
    low_lo = _mm256_blendv_epi8(low_lo, low_four_lo, four_select_lo);
    low_hi = _mm256_blendv_epi8(low_hi, low_four_hi, four_select_hi);
    //  high is zeroed on ASCII lanes (cp == raw byte, high == 0), then 4-byte high reconstructed.
    __m256i high_lo = _mm256_andnot_si256(ascii_select_lo, window.high_lo);
    __m256i high_hi = _mm256_andnot_si256(ascii_select_hi, window.high_hi);
    high_lo = _mm256_blendv_epi8(high_lo, high_four_lo, four_select_lo);
    high_hi = _mm256_blendv_epi8(high_hi, high_four_hi, four_select_hi);

    sz_u64_t const valid_start = true_ascii | valid2 | valid3 | valid4;
    sz_u64_t const consumed = (((valid2 | valid3 | valid4) << 1) | ((valid3 | valid4) << 2) | (valid4 << 3)) &
                              continuation & loaded_mask;
    sz_u64_t const starts = loaded_mask & ~consumed;
    sz_u64_t const replacement = starts & ~valid_start;

    //  4-byte plane bits (bits 16..20 of the codepoint); zero on every non-4-byte lane.
    __m256i const plane_lo = _mm256_or_si256(
        _mm256_and_si256(_mm256_slli_epi16(_mm256_and_si256(raw_lo, _mm256_set1_epi8(0x07)), 2),
                         _mm256_set1_epi8(0x1C)),
        sz_utf8_srl8_haswell_(next1_lo, 4, 0x03));
    __m256i const plane_hi = _mm256_or_si256(
        _mm256_and_si256(_mm256_slli_epi16(_mm256_and_si256(raw_hi, _mm256_set1_epi8(0x07)), 2),
                         _mm256_set1_epi8(0x1C)),
        sz_utf8_srl8_haswell_(next1_hi, 4, 0x03));
    __m256i const plane_masked_lo = _mm256_and_si256(four_select_lo, plane_lo);
    __m256i const plane_masked_hi = _mm256_and_si256(four_select_hi, plane_hi);

    //  Palette index per byte-lane, in one pass. BMP (cp < 0x10000) through the validated full-BMP cascade; astral
    //  (cp >= 0x10000) blended in; replacement lanes forced to U+FFFD's index.
    sz_u64_t const is_astral = four_byte & loaded_mask;
    __m256i index_lo = sz_line_break_bmp_index_haswell_(high_lo, low_lo);
    __m256i index_hi = sz_line_break_bmp_index_haswell_(high_hi, low_hi);
    if (is_astral) {
        //  The astral cascade is addressed by offset = codepoint - 0x10000; the codepoint's plane byte is
        //  `(cp>>16)` (>=1 for astral), so the offset plane nibble is `plane - 1`. The low 16 bits are unchanged
        //  by subtracting 0x10000, so `high`/`low` feed the cascade directly.
        __m256i const plane_off_lo = _mm256_sub_epi8(plane_masked_lo, _mm256_set1_epi8(1));
        __m256i const plane_off_hi = _mm256_sub_epi8(plane_masked_hi, _mm256_set1_epi8(1));
        __m256i const astral_lo = sz_line_break_classify_astral_haswell_(plane_off_lo, high_lo, low_lo);
        __m256i const astral_hi = sz_line_break_classify_astral_haswell_(plane_off_hi, high_hi, low_hi);
        index_lo = _mm256_blendv_epi8(index_lo, astral_lo, four_select_lo);
        index_hi = _mm256_blendv_epi8(index_hi, astral_hi, four_select_hi);
    }
    if (replacement) {
        __m256i const fffd = sz_line_break_bmp_index_haswell_(_mm256_set1_epi8((char)0xFF),
                                                              _mm256_set1_epi8((char)0xFD));
        __m256i const rep_lo = sz_utf8_byte_mask_from_bits_haswell_((sz_u32_t)replacement);
        __m256i const rep_hi = sz_utf8_byte_mask_from_bits_haswell_((sz_u32_t)(replacement >> 32));
        index_lo = _mm256_blendv_epi8(index_lo, fffd, rep_lo);
        index_hi = _mm256_blendv_epi8(index_hi, fffd, rep_hi);
    }

    sz_line_break_classified_haswell_t result;
    __m256i dotted_lo, dotted_hi;
    sz_line_break_palette_unpack_haswell_(index_lo, &result.classes_lo, &result.side_lo, &dotted_lo);
    sz_line_break_palette_unpack_haswell_(index_hi, &result.classes_hi, &result.side_hi, &dotted_hi);
    sz_u64_t const dotted = sz_utf8_mask_combine_haswell_(_mm256_cmpgt_epi8(dotted_lo, _mm256_setzero_si256()),
                                                          _mm256_cmpgt_epi8(dotted_hi, _mm256_setzero_si256()));
    result.dotted = dotted & starts;
    result.starts = starts;
    result.replacement = replacement;
    result.non_start = loaded_mask & ~starts;
    result.loaded = window.loaded;
    return result;
}

#pragma endregion In register vectorized classifier

#pragma region Mask algebra rule engine

/** @brief Build a 64-bit "lane class == @p cls" mask over both class halves (two `vpcmpeqb` -> mask_combine). */
SZ_INTERNAL sz_u64_t sz_line_break_class_mask_haswell_(__m256i classes_lo, __m256i classes_hi, sz_u8_t cls) {
    return sz_line_break_byte_equal_haswell_(classes_lo, classes_hi, cls);
}

/** @brief Build a 64-bit "lane (side & @p bit) != 0" mask over both side halves. */
SZ_INTERNAL sz_u64_t sz_line_break_side_mask_haswell_(__m256i side_lo, __m256i side_hi, sz_u8_t bit) {
    __m256i const m = _mm256_set1_epi8((char)bit);
    __m256i const masked_lo = _mm256_and_si256(side_lo, m), masked_hi = _mm256_and_si256(side_hi, m);
    return sz_utf8_mask_combine_haswell_(
        _mm256_cmpgt_epi8(masked_lo, _mm256_setzero_si256()) | _mm256_cmpeq_epi8(masked_lo, m),
        _mm256_cmpgt_epi8(masked_hi, _mm256_setzero_si256()) | _mm256_cmpeq_epi8(masked_hi, m));
}

/** @brief Byte-lane gate/base derivation (LB9/LB10) — the AVX2 twin of @ref sz_line_break_byte_frame_icelake_. */
typedef struct sz_line_break_byte_frame_haswell_t {
    __m256i classes_lo; /**< Class per lane [0,32) with lone marks reclassified to AL (LB10). */
    __m256i classes_hi; /**< Class per lane [32,64). */
    sz_u64_t base;      /**< Cluster-base lanes (every effective start except an attached CM/ZWJ). */
    sz_u64_t gate;      /**< Transparent lanes for neighbour fills: continuations + attached-mark starts. */
    sz_u64_t attached;  /**< Attached CM/ZWJ start lanes (LB9). */
    sz_u64_t lone_mark; /**< LB10 lone marks reclassified to AL; their side bits must be cleared. */
} sz_line_break_byte_frame_haswell_t;

SZ_INTERNAL sz_line_break_byte_frame_haswell_t sz_line_break_byte_frame_haswell_(
    sz_line_break_classified_haswell_t classified) {
    sz_u64_t const starts = classified.starts, non_start = classified.non_start;
    __m256i const classes_lo = classified.classes_lo, classes_hi = classified.classes_hi;
    sz_u64_t const mark_start = (sz_line_break_class_mask_haswell_(classes_lo, classes_hi, sz_line_break_cm_k) |
                                 sz_line_break_class_mask_haswell_(classes_lo, classes_hi, sz_line_break_zwj_k)) &
                                starts;
    sz_u64_t const excluded = (sz_line_break_class_mask_haswell_(classes_lo, classes_hi, sz_line_break_bk_k) |
                               sz_line_break_class_mask_haswell_(classes_lo, classes_hi, sz_line_break_cr_k) |
                               sz_line_break_class_mask_haswell_(classes_lo, classes_hi, sz_line_break_lf_k) |
                               sz_line_break_class_mask_haswell_(classes_lo, classes_hi, sz_line_break_nl_k) |
                               sz_line_break_class_mask_haswell_(classes_lo, classes_hi, sz_line_break_sp_k) |
                               sz_line_break_class_mask_haswell_(classes_lo, classes_hi, sz_line_break_zw_k)) &
                              starts;
    sz_u64_t const good_base = starts & ~excluded & ~mark_start;
    sz_u64_t const mark_bytes = sz_u64_fill_right_(mark_start, non_start) | mark_start;
    sz_u64_t const flood = sz_u64_fill_right_(good_base, non_start | mark_bytes);
    sz_u64_t const attached = flood & mark_start;
    sz_u64_t const lone_mark = mark_start & ~attached;

    sz_line_break_byte_frame_haswell_t frame;
    //  Reclassify lone marks to AL in both halves (LB10).
    __m256i const al = _mm256_set1_epi8((char)sz_line_break_al_k);
    __m256i const lone_select_lo = sz_utf8_byte_mask_from_bits_haswell_((sz_u32_t)lone_mark);
    __m256i const lone_select_hi = sz_utf8_byte_mask_from_bits_haswell_((sz_u32_t)(lone_mark >> 32));
    frame.classes_lo = _mm256_blendv_epi8(classes_lo, al, lone_select_lo);
    frame.classes_hi = _mm256_blendv_epi8(classes_hi, al, lone_select_hi);
    frame.base = starts & ~attached;
    frame.gate = non_start | attached;
    frame.attached = attached;
    frame.lone_mark = lone_mark;
    return frame;
}

/**
 *  @brief  Per-ISA extractor: lower one classified 64-byte window to the portable @ref sz_line_break_frame_t — the
 *          AVX2 twin of @ref sz_line_break_build_frame_icelake_. Builds the byte-level cluster frame (LB9/LB10),
 *          materializes per-class membership after the LB10 reclassify, the raw ZWJ + five side-bit masks, and the
 *          per-lane class/side bytes (two `vpstoreu` halves each).
 */
SZ_FORCE_INLINE sz_line_break_frame_t sz_line_break_build_frame_haswell_(sz_line_break_classified_haswell_t classified,
                                                                         sz_u8_t *effective_class_byte_out,
                                                                         sz_u8_t *side_byte_out) {
    sz_line_break_byte_frame_haswell_t const byte_frame = sz_line_break_byte_frame_haswell_(classified);
    __m256i const classes_lo = byte_frame.classes_lo, classes_hi = byte_frame.classes_hi;
    //  LB10 reclassify carries the side bits with it: zero the side byte on lone-mark lanes (serial zeros the
    //  descriptor). `andnot(lone_select, side)` clears those lanes.
    __m256i const lone_select_lo = sz_utf8_byte_mask_from_bits_haswell_((sz_u32_t)byte_frame.lone_mark);
    __m256i const lone_select_hi = sz_utf8_byte_mask_from_bits_haswell_((sz_u32_t)(byte_frame.lone_mark >> 32));
    __m256i const side_lo = _mm256_andnot_si256(lone_select_lo, classified.side_lo);
    __m256i const side_hi = _mm256_andnot_si256(lone_select_hi, classified.side_hi);

    sz_line_break_frame_t frame;
    frame.base = byte_frame.base;
    frame.gate = byte_frame.gate;
    frame.attached = byte_frame.attached;
    frame.lone_mark = byte_frame.lone_mark;
    frame.non_start = classified.non_start;
    frame.dotted = classified.dotted;
    frame.starts = classified.starts;
    frame.replacement = classified.replacement;
    //  Full unroll is load-bearing (mirrors icelake): SROA promotes the per-class array to registers and drops the
    //  `vpcmpeqb` pairs for classes the engine never reads; a runtime index keeps it stack-resident (~7% slower).
#if defined(__clang__)
#pragma clang loop unroll(full)
#elif defined(__GNUC__)
#pragma GCC unroll 64
#endif
    for (sz_size_t cls = 0; cls < sz_line_break_class_count_k; ++cls)
        frame.effective_class[cls] = sz_line_break_class_mask_haswell_(classes_lo, classes_hi, (sz_u8_t)cls);
    frame.raw_zwj = sz_line_break_class_mask_haswell_(classified.classes_lo, classified.classes_hi,
                                                      sz_line_break_zwj_k);
    frame.side_pi = sz_line_break_side_mask_haswell_(side_lo, side_hi, sz_line_break_side_pi_k);
    frame.side_pf = sz_line_break_side_mask_haswell_(side_lo, side_hi, sz_line_break_side_pf_k);
    frame.side_eaw = sz_line_break_side_mask_haswell_(side_lo, side_hi, sz_line_break_side_eaw_k);
    frame.side_cn = sz_line_break_side_mask_haswell_(side_lo, side_hi, sz_line_break_side_cn_k);
    frame.side_ext = sz_line_break_side_mask_haswell_(side_lo, side_hi, sz_line_break_side_ext_k);
    _mm256_storeu_si256((__m256i *)(effective_class_byte_out + 0), classes_lo);
    _mm256_storeu_si256((__m256i *)(effective_class_byte_out + 32), classes_hi);
    _mm256_storeu_si256((__m256i *)(side_byte_out + 0), side_lo);
    _mm256_storeu_si256((__m256i *)(side_byte_out + 32), side_hi);
    return frame;
}

/**
 *  @brief  Byte-level UAX-14 rule engine, Haswell entry: extract the portable frame in-register, then delegate every
 *          LB1-LB31 decision to the portable @ref sz_line_break_decide_window_.
 */
SZ_FORCE_INLINE sz_line_break_window_t
sz_line_break_decide_window_haswell_(sz_line_break_classified_haswell_t classified, sz_line_break_carry_t carry,
                                     sz_line_break_carry_t *carry_out, sz_size_t complete_limit, sz_bool_t more_text) {
    sz_u8_t effective_class_byte[64], side_byte[64];
    sz_line_break_frame_t const frame = sz_line_break_build_frame_haswell_(classified, effective_class_byte, side_byte);
    return sz_line_break_decide_window_(&frame, effective_class_byte, side_byte, carry, carry_out, complete_limit,
                                        more_text);
}

#pragma endregion Mask algebra rule engine

#pragma region Forward driver

/**
 *  @brief  Largest byte prefix of the window whose codepoints are all fully loaded — the AVX2 twin of
 *          @ref sz_line_break_complete_limit_ over the Haswell window struct. Never below 1.
 */
SZ_INTERNAL sz_size_t sz_line_break_complete_limit_haswell_(sz_utf8_rune_window_haswell_t window, sz_bool_t more_text) {
    sz_size_t const loaded = window.loaded;
    if (!more_text) return loaded;
    sz_u64_t const valid = sz_u64_mask_until_serial_(loaded);
    sz_u64_t const starts = window.codepoint_starts & valid;
    sz_u64_t const two = window.two_byte_starts & starts;
    sz_u64_t const three = window.three_byte_starts & starts;
    sz_u64_t const four = window.four_byte_starts & starts;
    sz_u64_t const straddle = ((two & ~sz_u64_mask_until_serial_(loaded > 1 ? loaded - 1 : 0)) |
                               (three & ~sz_u64_mask_until_serial_(loaded > 2 ? loaded - 2 : 0)) |
                               (four & ~sz_u64_mask_until_serial_(loaded > 3 ? loaded - 3 : 0))) &
                              valid;
    sz_size_t const limit = straddle ? (sz_size_t)sz_u64_ctz(straddle) : loaded;
    return limit > 0 ? limit : loaded;
}

/**
 *  @brief  Byte-level zero-scalar forward UAX-14 kernel (Haswell AVX2): the overlap-free advancing driver, mirroring
 *          @ref sz_utf8_linewraps_icelake_bytes_ over the AVX2 window/classify/drain leaves.
 */
SZ_PUBLIC sz_size_t sz_utf8_linewraps_haswell_bytes_( //
    sz_cptr_t text, sz_size_t length,                 //
    sz_size_t *starts, sz_size_t *lengths,            //
    sz_size_t capacity, sz_size_t *bytes_consumed) {

    if (length == 0 || capacity == 0) {
        if (bytes_consumed) *bytes_consumed = 0;
        return 0;
    }
    sz_u8_t const *bytes = (sz_u8_t const *)text;
    sz_size_t produced = 0;
    sz_size_t line_start = 0;
    sz_size_t position = 0;
    sz_line_break_carry_t carry = sz_line_break_carry_sot_();

    while (position < length) {
        sz_utf8_rune_window_haswell_t const window = sz_utf8_rune_decode_window_haswell_(bytes + position,
                                                                                         length - position);
        sz_bool_t const more_text = (sz_bool_t)(position + window.loaded < length);
        sz_size_t const complete_limit = sz_line_break_complete_limit_haswell_(window, more_text);
        sz_line_break_classified_haswell_t const classified = sz_line_break_classify_window_haswell_(window);

        sz_line_break_carry_t carry_next = carry;
        sz_line_break_window_t const win = sz_line_break_decide_window_haswell_(classified, carry, &carry_next,
                                                                                complete_limit, more_text);
        sz_u64_t const commit = win.breaks & sz_u64_mask_until_serial_(win.resolved);

        produced = sz_utf8_rune_drain_forward_haswell_(commit, position, starts, lengths, produced, capacity,
                                                       &line_start);
        if (produced >= capacity) {
            if (bytes_consumed) *bytes_consumed = line_start;
            return produced;
        }

        sz_size_t const advance = win.resolved ? win.resolved : complete_limit;
        carry = carry_next;
        position += advance ? advance : window.loaded;
    }

    if (produced < capacity) starts[produced] = line_start, lengths[produced] = length - line_start, ++produced;
    if (bytes_consumed) *bytes_consumed = length;
    return produced;
}

/**
 *  @brief  Forward UAX-14 line-break-opportunity kernel (Haswell AVX2). Bit-exact with `sz_utf8_linewraps_serial`
 *          and `sz_utf8_linewraps_icelake`.
 */
SZ_PUBLIC sz_size_t sz_utf8_linewraps_haswell( //
    sz_cptr_t text, sz_size_t length,          //
    sz_size_t *starts, sz_size_t *lengths,     //
    sz_size_t capacity, sz_size_t *bytes_consumed) {
    return sz_utf8_linewraps_haswell_bytes_(text, length, starts, lengths, capacity, bytes_consumed);
}

#pragma endregion Forward driver

#pragma endregion UAX 14 Line Boundaries forward kernel
#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#endif // SZ_USE_HASWELL

#ifdef __cplusplus
}
#endif

#endif // STRINGZILLA_UTF8_LINEWRAPS_HASWELL_H_
