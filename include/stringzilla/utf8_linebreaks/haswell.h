/**
 *  @brief Haswell (AVX2) backend for UAX-14 line break boundaries.
 *  @file include/stringzilla/utf8_linebreaks/haswell.h
 *  @author Ash Vardanian
 */
#ifndef STRINGZILLA_UTF8_LINEBREAKS_HASWELL_H_
#define STRINGZILLA_UTF8_LINEBREAKS_HASWELL_H_

#include "stringzilla/types.h"
#include "stringzilla/utf8_linebreaks/tables.h"
#include "stringzilla/utf8_linebreaks/serial.h"
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
 *  (class, side, dotted) with ZERO per-lane scalar loop and NO serial deferral. Each 64-byte window lives as two
 *  `__m256i` halves (`_lo` = lanes [0,32), `_hi` = lanes [32,64)); every per-lane class compare is
 *  `_mm256_cmpeq_epi8` on both halves OR-combined to a `sz_u64_t` via `mask_combine_haswell_`. The BMP palette index
 *  is ONE indexed lookup per codepoint into a page-compressed flat table via `sz_line_break_bmp_index_haswell_`
 *  (`bmp_page_lut_[cp >> 8]` picks one of 67 pages, then `flat_bmp_[page * 256 + low]`, fetched by `vpgatherdd`); the
 *  astral path uses the `sz_line_break_classify_astral_haswell_` cascade. The palette descriptor is unpacked to
 *  the LB1-resolved class byte and the engine side byte, bit-identical to the icelake descriptor unpack.
 *
 *  The flat table is chosen for port pressure, not instruction count: cross-lane shuffles are port-5-only, so any
 *  dependent shuffle cascade saturates that single port, while `vpgatherdd` issues on the load ports and leaves the
 *  shuffle port to the decode. Gathers pay off on multi-KB tables in general - see less_slow.cpp v0.3.0 "Gather and
 *  Scatter": https://github.com/ashvardanian/less_slow.cpp/releases/tag/v0.3.0 */

/** @brief Flat-palette index for thirty-two BMP codepoints (per-lane high = cp>>8, low = cp&0xFF): the `bmp_page_lut_` page
 *         LUT selects one of the 67 distinct 256-byte pages, then `flat_bmp_[page * 256 + low]` is fetched by four
 *         `vpgatherdd`. The leaf byte indexes `sz_utf8_line_break_flat_palette_`, NOT the 62-entry cascade palette.
 *         Bit-exact with `sz_rune_line_break_property` over the whole BMP. */
SZ_HELPER_AUTO __m256i sz_line_break_bmp_index_haswell_(__m256i high, __m256i low) {
    return sz_utf8_rune_flat_lookup_haswell_(sz_utf8_line_break_bmp_page_lut_, sz_utf8_line_break_flat_bmp_, high, low);
}

/** @brief All-ones lane mask where the single bit @p bit is set in @p bytes (@p bit must have exactly one bit set). */
SZ_HELPER_INLINE __m256i sz_line_break_bit_mask_haswell_(__m256i bytes_u8x32, sz_u8_t bit) {
    __m256i const bit_u8x32 = _mm256_set1_epi8((char)bit);
    return _mm256_cmpeq_epi8(_mm256_and_si256(bytes_u8x32, bit_u8x32), bit_u8x32);
}

/** @brief Split thirty-two flat-palette indices into the low and high byte of their 16-bit Line_Break descriptors.
 *         Four `vpgatherdd` (scale 2) read `flat_palette_[index]` eight lanes at a time -- the AVX2 stand-in for
 *         icelake's `vpermi2w` over the two palette tiles, AVX2 having no cross-lane word permute -- and the shared
 *         `pack4_u32_to_u8_haswell_` narrows each descriptor byte back into lane order. Every index originates in the
 *         flat leaf, so it is always < 56 and the scale-2 gather stays inside the 64-word padded palette. */
SZ_HELPER_AUTO void sz_line_break_flat_palette_descriptors_haswell_(__m256i palette_indices_u8x32,
                                                                    __m256i *descriptor_low_bytes_u8x32,
                                                                    __m256i *descriptor_high_bytes_u8x32) {
    __m128i const indices_low_u8x16 = _mm256_castsi256_si128(palette_indices_u8x32);
    __m128i const indices_high_u8x16 = _mm256_extracti128_si256(palette_indices_u8x32, 1);
    __m128i const index_quarters_u8x16[4] = {indices_low_u8x16, _mm_srli_si128(indices_low_u8x16, 8),
                                             indices_high_u8x16, _mm_srli_si128(indices_high_u8x16, 8)};
    __m256i low_byte_quarters_u32x8[4], high_byte_quarters_u32x8[4];
    for (int quarter = 0; quarter < 4; ++quarter) {
        __m256i const descriptors_u32x8 = _mm256_i32gather_epi32(
            (int const *)sz_utf8_line_break_flat_palette_, _mm256_cvtepu8_epi32(index_quarters_u8x16[quarter]), 2);
        low_byte_quarters_u32x8[quarter] = _mm256_and_si256(descriptors_u32x8, _mm256_set1_epi32(0xFF));
        high_byte_quarters_u32x8[quarter] = _mm256_and_si256(_mm256_srli_epi32(descriptors_u32x8, 8),
                                                             _mm256_set1_epi32(0xFF));
    }
    *descriptor_low_bytes_u8x32 = sz_utf8_rune_pack4_u32_to_u8_haswell_(
        low_byte_quarters_u32x8[0], low_byte_quarters_u32x8[1], low_byte_quarters_u32x8[2], low_byte_quarters_u32x8[3]);
    *descriptor_high_bytes_u8x32 = sz_utf8_rune_pack4_u32_to_u8_haswell_(
        high_byte_quarters_u32x8[0], high_byte_quarters_u32x8[1], high_byte_quarters_u32x8[2],
        high_byte_quarters_u32x8[3]);
}

/** @brief Expand thirty-two flat-palette indices to the LB1-resolved class byte, the engine side byte and the
 *         DottedCircle lane mask (all-ones where set), the AVX2 twin of
 *         @ref sz_line_break_flat_palette_unpack_icelake_. Every descriptor field the engine reads lives below bit 14
 *         -- class in bits 0-5, Pi/Pf in 6/7, EAW/Cn|Ext in 8/9, SA-is-mark in 12, DottedCircle in 13 -- so the whole
 *         unpack stays in the BYTE domain over the descriptor's low and high byte, with no 16-bit lane widening.
 *         Applies the serial resolution aliasing (SA → AL/CM, AI/SG/XX → AL, CJ → NS); RI/ZWJ side bits come from the
 *         RAW class, the mark side bit from the resolved class. */
SZ_HELPER_AUTO void sz_line_break_flat_palette_unpack_haswell_(__m256i palette_indices_u8x32,
                                                               __m256i *classes_u8x32_out, __m256i *side_u8x32_out,
                                                               __m256i *dotted_select_u8x32_out) {
    __m256i descriptor_low_bytes_u8x32, descriptor_high_bytes_u8x32;
    sz_line_break_flat_palette_descriptors_haswell_(palette_indices_u8x32, &descriptor_low_bytes_u8x32,
                                                    &descriptor_high_bytes_u8x32);

    __m256i const raw_classes_u8x32 = _mm256_and_si256(descriptor_low_bytes_u8x32, _mm256_set1_epi8(0x3F));
    __m256i classes_u8x32 = raw_classes_u8x32;
    __m256i const is_sa_u8x32 = _mm256_cmpeq_epi8(classes_u8x32, _mm256_set1_epi8((char)sz_line_break_sa_k));
    __m256i const sa_is_mark_u8x32 = sz_line_break_bit_mask_haswell_(descriptor_high_bytes_u8x32, 1 << 4);
    classes_u8x32 = _mm256_blendv_epi8(classes_u8x32, _mm256_set1_epi8((char)sz_line_break_al_k), is_sa_u8x32);
    classes_u8x32 = _mm256_blendv_epi8(classes_u8x32, _mm256_set1_epi8((char)sz_line_break_cm_k),
                                       _mm256_and_si256(is_sa_u8x32, sa_is_mark_u8x32));
    __m256i const is_alias_u8x32 = _mm256_or_si256(
        _mm256_or_si256(_mm256_cmpeq_epi8(classes_u8x32, _mm256_set1_epi8((char)sz_line_break_ai_k)),
                        _mm256_cmpeq_epi8(classes_u8x32, _mm256_set1_epi8((char)sz_line_break_sg_k))),
        _mm256_cmpeq_epi8(classes_u8x32, _mm256_set1_epi8((char)sz_line_break_xx_k)));
    classes_u8x32 = _mm256_blendv_epi8(classes_u8x32, _mm256_set1_epi8((char)sz_line_break_al_k), is_alias_u8x32);
    __m256i const is_cj_u8x32 = _mm256_cmpeq_epi8(classes_u8x32, _mm256_set1_epi8((char)sz_line_break_cj_k));
    classes_u8x32 = _mm256_blendv_epi8(classes_u8x32, _mm256_set1_epi8((char)sz_line_break_ns_k), is_cj_u8x32);

    __m256i side_u8x32 = _mm256_setzero_si256();
    side_u8x32 = _mm256_or_si256(side_u8x32,
                                 _mm256_and_si256(sz_line_break_bit_mask_haswell_(descriptor_low_bytes_u8x32, 1 << 6),
                                                  _mm256_set1_epi8((char)sz_line_break_side_pi_k)));
    side_u8x32 = _mm256_or_si256(side_u8x32,
                                 _mm256_and_si256(sz_line_break_bit_mask_haswell_(descriptor_low_bytes_u8x32, 1 << 7),
                                                  _mm256_set1_epi8((char)sz_line_break_side_pf_k)));
    side_u8x32 = _mm256_or_si256(side_u8x32,
                                 _mm256_and_si256(sz_line_break_bit_mask_haswell_(descriptor_high_bytes_u8x32, 1 << 0),
                                                  _mm256_set1_epi8((char)sz_line_break_side_eaw_k)));
    side_u8x32 = _mm256_or_si256(
        side_u8x32, _mm256_and_si256(sz_line_break_bit_mask_haswell_(descriptor_high_bytes_u8x32, 1 << 1),
                                     _mm256_set1_epi8((char)(sz_line_break_side_cn_k | sz_line_break_side_ext_k))));
    side_u8x32 = _mm256_or_si256(
        side_u8x32, _mm256_and_si256(_mm256_cmpeq_epi8(raw_classes_u8x32, _mm256_set1_epi8((char)sz_line_break_ri_k)),
                                     _mm256_set1_epi8((char)sz_line_break_side_ri_k)));
    side_u8x32 = _mm256_or_si256(
        side_u8x32, _mm256_and_si256(_mm256_cmpeq_epi8(raw_classes_u8x32, _mm256_set1_epi8((char)sz_line_break_zwj_k)),
                                     _mm256_set1_epi8((char)sz_line_break_side_zwj_k)));
    __m256i const class_is_mark_u8x32 = _mm256_or_si256(
        _mm256_cmpeq_epi8(classes_u8x32, _mm256_set1_epi8((char)sz_line_break_cm_k)),
        _mm256_cmpeq_epi8(classes_u8x32, _mm256_set1_epi8((char)sz_line_break_zwj_k)));
    side_u8x32 = _mm256_or_si256(
        side_u8x32, _mm256_and_si256(class_is_mark_u8x32, _mm256_set1_epi8((char)sz_line_break_side_mark_k)));

    *classes_u8x32_out = classes_u8x32;
    *side_u8x32_out = side_u8x32;
    *dotted_select_u8x32_out = sz_line_break_bit_mask_haswell_(descriptor_high_bytes_u8x32, 1 << 5);
}

/** @brief Palette index for thirty-two ASTRAL codepoints over the 20-bit offset = cp - 0x10000 (5-nibble cascade),
 *         the AVX2 twin of `sz_line_break_classify_astral16_icelake_`. Per-lane bytes: @p plane = (offset>>16)&0xFF
 *         (low nibble meaningful), @p high = (offset>>8)&0xFF, @p low = offset&0xFF. Bit-exact. */
SZ_HELPER_AUTO __m256i sz_line_break_classify_astral_haswell_(__m256i plane, __m256i high, __m256i low) {
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
SZ_HELPER_INLINE __m256i sz_line_break_cmpge_epu8_haswell_(__m256i value, __m256i bound) {
    return _mm256_cmpeq_epi8(_mm256_max_epu8(value, bound), value);
}

/** @brief Per-half unsigned `value < bound` mask: the complement of `value >= bound` (`min_epu8` form keeps it
 *         branch- and bias-free): `min_epu8(value,bound)==value && value!=bound` is awkward, so use the >= negation
 *         via `min_epu8(value, bound-1)==value` is fragile at bound==0; instead AND-NOT the >= mask against ones. */
SZ_HELPER_INLINE __m256i sz_line_break_cmplt_epu8_haswell_(__m256i value, __m256i bound) {
    return _mm256_andnot_si256(sz_line_break_cmpge_epu8_haswell_(value, bound), _mm256_set1_epi8((char)0xFF));
}

/** @brief A 64-bit "(byte & mask) == pattern" lane mask over both window halves (two `vpand`+`vpcmpeqb`). */
SZ_HELPER_INLINE sz_u64_t sz_line_break_byte_match_haswell_(__m256i low_half, __m256i high_half, sz_u8_t mask,
                                                            sz_u8_t pattern) {
    __m256i const m = _mm256_set1_epi8((char)mask), p = _mm256_set1_epi8((char)pattern);
    return sz_utf8_mask_combine_haswell_(_mm256_cmpeq_epi8(_mm256_and_si256(low_half, m), p),
                                         _mm256_cmpeq_epi8(_mm256_and_si256(high_half, m), p));
}

/** @brief A 64-bit "byte == value" lane mask over both window halves. */
SZ_HELPER_INLINE sz_u64_t sz_line_break_byte_equal_haswell_(__m256i low_half, __m256i high_half, sz_u8_t value) {
    __m256i const v = _mm256_set1_epi8((char)value);
    return sz_utf8_mask_combine_haswell_(_mm256_cmpeq_epi8(low_half, v), _mm256_cmpeq_epi8(high_half, v));
}

/** @brief A 64-bit "byte >= bound" (unsigned) lane mask over both window halves. */
SZ_HELPER_INLINE sz_u64_t sz_line_break_byte_ge_haswell_(__m256i low_half, __m256i high_half, sz_u8_t bound) {
    __m256i const bound_vec = _mm256_set1_epi8((char)bound);
    return sz_utf8_mask_combine_haswell_(sz_line_break_cmpge_epu8_haswell_(low_half, bound_vec),
                                         sz_line_break_cmpge_epu8_haswell_(high_half, bound_vec));
}

/** @brief A 64-bit "byte < bound" (unsigned) lane mask over both window halves. */
SZ_HELPER_INLINE sz_u64_t sz_line_break_byte_lt_haswell_(__m256i low_half, __m256i high_half, sz_u8_t bound) {
    __m256i const bound_vec = _mm256_set1_epi8((char)bound);
    return sz_utf8_mask_combine_haswell_(sz_line_break_cmplt_epu8_haswell_(low_half, bound_vec),
                                         sz_line_break_cmplt_epu8_haswell_(high_half, bound_vec));
}

/** @brief The class/side byte held at byte-lane @p lane, extracted in-register (no scalar window store). */
SZ_HELPER_INLINE sz_u8_t sz_line_break_byte_at_haswell_(__m256i lanes_lo, __m256i lanes_hi, sz_size_t lane) {
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
SZ_HELPER_INLINE void sz_line_break_next3_haswell_(__m256i window_lo, __m256i window_hi, __m256i *next3_lo,
                                                   __m256i *next3_hi) {
    __m256i const low_successor = _mm256_permute2x128_si256(window_lo, window_hi, 0x21);
    *next3_lo = _mm256_alignr_epi8(low_successor, window_lo, 3);
    __m256i const high_successor = _mm256_permute2x128_si256(window_hi, window_lo, 0x21);
    *next3_hi = _mm256_alignr_epi8(high_successor, window_hi, 3);
}

/** @brief Resolve the per-lane palette index (one half) to class / side / dotted bytes through the precomputed
 *         62-entry palette tables. `lut256_haswell_` reads each 64-entry table by `index` (index < 64 so only the
 *         low four rows are selected); bit-identical to the icelake `vpermb` palette permute. */
SZ_HELPER_INLINE void sz_line_break_palette_unpack_haswell_(__m256i index, __m256i *classes, __m256i *side,
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
SZ_HELPER_AUTO sz_line_break_classified_haswell_t sz_line_break_classify_window_haswell_(
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

    //  Flat-palette index per byte-lane, in one pass. BMP (cp < 0x10000) through the page-compressed flat leaf;
    //  replacement lanes forced to U+FFFD's index (U+FFFD is itself BMP, so it shares this index space). Astral lanes
    //  cannot join here: the astral cascade still speaks the 62-entry palette, a DIFFERENT index space, so they are
    //  blended after the expansion, on the resolved class/side bytes. Only VALID 4-byte starts join that late blend:
    //  an invalid 4-byte lead is a replacement lane whose U+FFFD resolution must survive it, so `valid4`, not
    //  `four_byte`, gates the blend.
    sz_u64_t const is_astral = valid4 & loaded_mask;
    __m256i palette_indices_low_u8x32 = sz_line_break_bmp_index_haswell_(high_lo, low_lo);
    __m256i palette_indices_high_u8x32 = sz_line_break_bmp_index_haswell_(high_hi, low_hi);
    if (replacement) {
        __m256i const replacement_indices_u8x32 = sz_line_break_bmp_index_haswell_(_mm256_set1_epi8((char)0xFF),
                                                                                   _mm256_set1_epi8((char)0xFD));
        __m256i const replacement_select_low_u8x32 = sz_utf8_byte_mask_from_bits_haswell_((sz_u32_t)replacement);
        __m256i const replacement_select_high_u8x32 = sz_utf8_byte_mask_from_bits_haswell_(
            (sz_u32_t)(replacement >> 32));
        palette_indices_low_u8x32 = _mm256_blendv_epi8(palette_indices_low_u8x32, replacement_indices_u8x32,
                                                       replacement_select_low_u8x32);
        palette_indices_high_u8x32 = _mm256_blendv_epi8(palette_indices_high_u8x32, replacement_indices_u8x32,
                                                        replacement_select_high_u8x32);
    }

    sz_line_break_classified_haswell_t result;
    __m256i dotted_select_low_u8x32, dotted_select_high_u8x32;
    sz_line_break_flat_palette_unpack_haswell_(palette_indices_low_u8x32, &result.classes_lo, &result.side_lo,
                                               &dotted_select_low_u8x32);
    sz_line_break_flat_palette_unpack_haswell_(palette_indices_high_u8x32, &result.classes_hi, &result.side_hi,
                                               &dotted_select_high_u8x32);
    if (is_astral) {
        //  The astral cascade is addressed by offset = codepoint - 0x10000; the codepoint's plane byte is
        //  `(cp>>16)` (>=1 for astral), so the offset plane nibble is `plane - 1`. The low 16 bits are unchanged
        //  by subtracting 0x10000, so `high`/`low` feed the cascade directly. Its 62-entry palette's byte tables carry
        //  the very same LB1 resolution the flat descriptor unpack applies (verified entry by entry), so blending the
        //  RESOLVED bytes matches blending indices in a single shared space, which the two palettes do not form.
        __m256i const astral_select_lo = sz_utf8_byte_mask_from_bits_haswell_((sz_u32_t)is_astral);
        __m256i const astral_select_hi = sz_utf8_byte_mask_from_bits_haswell_((sz_u32_t)(is_astral >> 32));
        __m256i const plane_off_lo = _mm256_sub_epi8(plane_masked_lo, _mm256_set1_epi8(1));
        __m256i const plane_off_hi = _mm256_sub_epi8(plane_masked_hi, _mm256_set1_epi8(1));
        __m256i const astral_indices_low_u8x32 = sz_line_break_classify_astral_haswell_(plane_off_lo, high_lo, low_lo);
        __m256i const astral_indices_high_u8x32 = sz_line_break_classify_astral_haswell_(plane_off_hi, high_hi, low_hi);
        __m256i astral_classes_u8x32, astral_side_u8x32, astral_dotted_bytes_u8x32;
        sz_line_break_palette_unpack_haswell_(astral_indices_low_u8x32, &astral_classes_u8x32, &astral_side_u8x32,
                                              &astral_dotted_bytes_u8x32);
        result.classes_lo = _mm256_blendv_epi8(result.classes_lo, astral_classes_u8x32, astral_select_lo);
        result.side_lo = _mm256_blendv_epi8(result.side_lo, astral_side_u8x32, astral_select_lo);
        dotted_select_low_u8x32 = _mm256_blendv_epi8(
            dotted_select_low_u8x32, _mm256_cmpgt_epi8(astral_dotted_bytes_u8x32, _mm256_setzero_si256()),
            astral_select_lo);
        sz_line_break_palette_unpack_haswell_(astral_indices_high_u8x32, &astral_classes_u8x32, &astral_side_u8x32,
                                              &astral_dotted_bytes_u8x32);
        result.classes_hi = _mm256_blendv_epi8(result.classes_hi, astral_classes_u8x32, astral_select_hi);
        result.side_hi = _mm256_blendv_epi8(result.side_hi, astral_side_u8x32, astral_select_hi);
        dotted_select_high_u8x32 = _mm256_blendv_epi8(
            dotted_select_high_u8x32, _mm256_cmpgt_epi8(astral_dotted_bytes_u8x32, _mm256_setzero_si256()),
            astral_select_hi);
    }
    sz_u64_t const dotted = sz_utf8_mask_combine_haswell_(dotted_select_low_u8x32, dotted_select_high_u8x32);
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
SZ_HELPER_INLINE sz_u64_t sz_line_break_class_mask_haswell_(__m256i classes_lo, __m256i classes_hi, sz_u8_t cls) {
    return sz_line_break_byte_equal_haswell_(classes_lo, classes_hi, cls);
}

/** @brief Build a 64-bit "lane (side & @p bit) != 0" mask over both side halves. */
SZ_HELPER_INLINE sz_u64_t sz_line_break_side_mask_haswell_(__m256i side_lo, __m256i side_hi, sz_u8_t bit) {
    __m256i const m = _mm256_set1_epi8((char)bit);
    __m256i const masked_lo = _mm256_and_si256(side_lo, m), masked_hi = _mm256_and_si256(side_hi, m);
    return sz_utf8_mask_combine_haswell_(
        _mm256_or_si256(_mm256_cmpgt_epi8(masked_lo, _mm256_setzero_si256()), _mm256_cmpeq_epi8(masked_lo, m)),
        _mm256_or_si256(_mm256_cmpgt_epi8(masked_hi, _mm256_setzero_si256()), _mm256_cmpeq_epi8(masked_hi, m)));
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

SZ_HELPER_INLINE sz_line_break_byte_frame_haswell_t sz_line_break_byte_frame_haswell_(
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
SZ_HELPER_INLINE sz_line_break_frame_t sz_line_break_build_frame_haswell_(sz_line_break_classified_haswell_t classified,
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
SZ_HELPER_INLINE sz_line_break_window_t sz_line_break_decide_window_haswell_(
    sz_line_break_classified_haswell_t classified, sz_line_break_carry_t carry, sz_line_break_carry_t *carry_out,
    sz_size_t complete_limit, sz_bool_t more_text) {
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
SZ_HELPER_AUTO sz_size_t sz_line_break_complete_limit_haswell_(sz_utf8_rune_window_haswell_t window,
                                                               sz_bool_t more_text) {
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
 *          @ref sz_utf8_linebreaks_icelake_bytes_ over the AVX2 window/classify/drain leaves.
 */
SZ_API_COMPTIME sz_size_t sz_utf8_linebreaks_haswell_bytes_( //
    sz_cptr_t text, sz_size_t length,                        //
    sz_size_t *starts, sz_size_t *lengths,                   //
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
 *  @brief  Forward UAX-14 line-break-opportunity kernel (Haswell AVX2). Bit-exact with `sz_utf8_linebreaks_serial`
 *          and `sz_utf8_linebreaks_icelake`.
 */
SZ_API_COMPTIME sz_size_t sz_utf8_linebreaks_haswell( //
    sz_cptr_t text, sz_size_t length,                 //
    sz_size_t *starts, sz_size_t *lengths,            //
    sz_size_t capacity, sz_size_t *bytes_consumed) {
    return sz_utf8_linebreaks_haswell_bytes_(text, length, starts, lengths, capacity, bytes_consumed);
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

#endif // STRINGZILLA_UTF8_LINEBREAKS_HASWELL_H_
