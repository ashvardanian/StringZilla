/**
 *  @brief Ice Lake backend for UAX-14 line break boundaries.
 *  @file include/stringzilla/utf8_linebreaks/icelake.h
 *  @author Ash Vardanian
 */
#ifndef STRINGZILLA_UTF8_LINEBREAKS_ICELAKE_H_
#define STRINGZILLA_UTF8_LINEBREAKS_ICELAKE_H_

#include "stringzilla/types.h"
#include "stringzilla/utf8_linebreaks/tables.h"
#include "stringzilla/utf8_linebreaks/serial.h"
#include "stringzilla/utf8_runes/icelake.h"

#ifdef __cplusplus
extern "C" {
#endif

#if SZ_USE_ICELAKE
#if defined(__clang__) && SZ_CLANG_HAS_EVEX512_
#pragma clang attribute push(                                                                                         \
    __attribute__((target("avx,avx512f,avx512vl,avx512bw,avx512dq,avx512vbmi,avx512vbmi2,bmi,bmi2,evex512,popcnt"))), \
    apply_to = function)
#elif defined(__clang__)
#pragma clang attribute push(                                                                                 \
    __attribute__((target("avx,avx512f,avx512vl,avx512bw,avx512dq,avx512vbmi,avx512vbmi2,bmi,bmi2,popcnt"))), \
    apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("avx", "avx512f", "avx512vl", "avx512bw", "avx512dq", "avx512vbmi", "avx512vbmi2", "bmi", "bmi2", \
                   "popcnt")
#endif

#pragma region UAX 14 Line Boundaries forward kernel

#pragma region In register vectorized classifier

/*  The classifier resolves a contiguous run of codepoints to per-codepoint (class, side, dotted) with ZERO
 *  per-lane scalar loop and NO serial `sz_rune_line_break_property` deferral. It processes sixteen 32-bit codepoint
 *  lanes per pass: the whole BMP is mapped to a flat-palette index by ONE indexed lookup - `bmp_page_lut_[cp >> 8]`
 *  selects one of 67 distinct 256-byte pages, then `flat_bmp_[page * 256 + (cp & 0xFF)]`, fetched by `vpgatherdd` -
 *  while the astral planes still resolve through the 8/4/4/4 trie in the 62-entry cascade palette's index space. The
 *  descriptor is then unpacked to the LB1-resolved class byte and the engine side byte by in-register compares and
 *  masked moves.
 *
 *  Design note: the flat table is chosen for port pressure rather than instruction count. `vpermb`/`vpermi2b`
 *  cross-lane shuffles are port-5-only, so any dependent shuffle cascade saturates that single port, while
 *  `vpgatherdd` issues on the load ports and leaves the shuffle port to the decode. Gathers pay off on multi-KB
 *  tables in general - see less_slow.cpp v0.3.0 "Gather and Scatter":
 *  https://github.com/ashvardanian/less_slow.cpp/releases/tag/v0.3.0
 *
 *  ! The line leaf holds a `flat_palette_` index, NOT the 62-entry cascade palette index, because the Line_Break
 *  ! descriptor is 16-bit and the leaf must stay one byte per codepoint. The two index spaces do not mix. */

/** @brief Palette index for sixteen astral (>=0x10000) codepoints via a register-resident 8/4/4/4 trie over
 *         offset = codepoint - 0x10000 (s0 -> s1 -> s2 -> leaf). Re-init-free: every tile is read straight from
 *         aligned .rodata through the substrate permute256_/lut_cascade_ helpers. Bit-exact with
 *         `sz_rune_line_break_property` over the astral planes. */
SZ_HELPER_AUTO __m512i sz_line_break_classify_astral16_icelake_(__m512i codepoints) {
    __m512i const offset = _mm512_sub_epi32(codepoints, _mm512_set1_epi32(0x10000));
    __m512i const stage1 = sz_utf8_rune_permute256_icelake_(
        sz_utf8_line_break_astral_s0_, _mm512_and_si512(_mm512_srli_epi32(offset, 12), _mm512_set1_epi32(0xFF)));
    __m512i const stage2_index = _mm512_add_epi32(
        _mm512_slli_epi32(stage1, 4), _mm512_and_si512(_mm512_srli_epi32(offset, 8), _mm512_set1_epi32(0xF)));
    __m512i const stage2 = sz_utf8_rune_lut_cascade_icelake_(sz_utf8_line_break_astral_s1_,
                                                             (int)sz_utf8_line_break_astral_s1_tiles_k, stage2_index);
    __m512i const leaf_index = _mm512_add_epi32(_mm512_slli_epi32(stage2, 4),
                                                _mm512_and_si512(_mm512_srli_epi32(offset, 4), _mm512_set1_epi32(0xF)));
    __m512i const leaf = sz_utf8_rune_lut_cascade_icelake_(sz_utf8_line_break_astral_s2_,
                                                           (int)sz_utf8_line_break_astral_s2_tiles_k, leaf_index);
    __m512i const class_index = _mm512_add_epi32(_mm512_slli_epi32(leaf, 4),
                                                 _mm512_and_si512(offset, _mm512_set1_epi32(0xF)));
    return sz_utf8_rune_lut_cascade_icelake_(sz_utf8_line_break_astral_leaf_,
                                             (int)sz_utf8_line_break_astral_leaf_tiles_k, class_index);
}

/** @brief Flat-palette index for sixteen BMP byte-lanes, read straight from the page-compressed flat leaf: one
 *         `vpermb` over `bmp_page_lut_` picks the 256-byte page, one `vpgatherdd` fetches
 *         `flat_bmp_[page * 256 + (cp & 0xFF)]`. The leaf byte is an index into
 *         `sz_utf8_line_break_flat_palette_`, NOT the 62-entry cascade palette. Only the low byte of each u32 lane
 *         is the index; the caller truncates with `vpmovdb`. */
SZ_HELPER_AUTO __m512i sz_line_break_bmp_flat_index16_icelake_(__m512i codepoints_u32x16) {
    return sz_utf8_rune_flat_lookup_icelake_(sz_utf8_line_break_bmp_page_lut_, sz_utf8_line_break_flat_bmp_,
                                             codepoints_u32x16);
}

/** @brief All-64-lane flat-palette index for `cp < 0x10000` (`cp = (high << 8) | low`): four sixteen-lane
 *         `vpgatherdd` leaves resolve the whole window on the load ports, leaving the cross-lane shuffle port to
 *         the decode. Lanes whose codepoint is >= 0x10000 are
 *         undefined (the caller blends the astral path over them). The sixteen-lane groups are unrolled because
 *         `vextracti32x4` / `vinserti32x4` take an immediate lane selector. */
SZ_HELPER_AUTO __m512i sz_line_break_bmp_flat_index_icelake_(__m512i high_bytes_u8x64, __m512i low_bytes_u8x64) {
    __m512i palette_indices_u8x64 = _mm512_setzero_si512();
    __m512i high_u32x16, low_u32x16, codepoints_u32x16, group_indices_u32x16;
#define SZ_LINE_BREAK_FLAT_GROUP_ICELAKE_(group)                                            \
    high_u32x16 = _mm512_cvtepu8_epi32(_mm512_extracti32x4_epi32(high_bytes_u8x64, group)); \
    low_u32x16 = _mm512_cvtepu8_epi32(_mm512_extracti32x4_epi32(low_bytes_u8x64, group));   \
    codepoints_u32x16 = _mm512_or_si512(_mm512_slli_epi32(high_u32x16, 8), low_u32x16);     \
    group_indices_u32x16 = sz_line_break_bmp_flat_index16_icelake_(codepoints_u32x16);      \
    palette_indices_u8x64 = _mm512_inserti32x4(palette_indices_u8x64, _mm512_cvtepi32_epi8(group_indices_u32x16), group)
    SZ_LINE_BREAK_FLAT_GROUP_ICELAKE_(0);
    SZ_LINE_BREAK_FLAT_GROUP_ICELAKE_(1);
    SZ_LINE_BREAK_FLAT_GROUP_ICELAKE_(2);
    SZ_LINE_BREAK_FLAT_GROUP_ICELAKE_(3);
#undef SZ_LINE_BREAK_FLAT_GROUP_ICELAKE_
    return palette_indices_u8x64;
}

/** @brief Unpack thirty-two 16-bit palette descriptors (one half of the 64-byte block) to the LB1-resolved class and
 *         the engine side byte, both as 16-bit lanes, plus the per-lane DottedCircle predicate. Applies the serial
 *         resolution aliasing (SA → AL/CM, AI/SG/XX → AL, CJ → NS); Pi/Pf/EAW/Cn|Ext side bits come from descriptor
 *         bits 6/7/8/9; RI/ZWJ side from the raw class; CM|ZWJ -> mark side bit; DottedCircle from bit 13. */
SZ_HELPER_INLINE void sz_line_break_descriptor_unpack_half_icelake_(__m512i descriptors, __m512i *classes_out,
                                                                    __m512i *side_out, __mmask32 *dotted_out) {
    __m512i classes = _mm512_and_si512(descriptors, _mm512_set1_epi16(0x3F));
    __mmask32 const is_sa = _mm512_cmpeq_epi16_mask(classes, _mm512_set1_epi16(sz_line_break_sa_k));
    __mmask32 const sa_is_mark = _mm512_test_epi16_mask(descriptors, _mm512_set1_epi16(1 << 12));
    classes = _mm512_mask_mov_epi16(classes, is_sa, _mm512_set1_epi16(sz_line_break_al_k));
    classes = _mm512_mask_mov_epi16(classes, is_sa & sa_is_mark, _mm512_set1_epi16(sz_line_break_cm_k));
    __mmask32 const is_alias = _mm512_cmpeq_epi16_mask(classes, _mm512_set1_epi16(sz_line_break_ai_k)) |
                               _mm512_cmpeq_epi16_mask(classes, _mm512_set1_epi16(sz_line_break_sg_k)) |
                               _mm512_cmpeq_epi16_mask(classes, _mm512_set1_epi16(sz_line_break_xx_k));
    classes = _mm512_mask_mov_epi16(classes, is_alias, _mm512_set1_epi16(sz_line_break_al_k));
    __mmask32 const is_cj = _mm512_cmpeq_epi16_mask(classes, _mm512_set1_epi16(sz_line_break_cj_k));
    classes = _mm512_mask_mov_epi16(classes, is_cj, _mm512_set1_epi16(sz_line_break_ns_k));

    __m512i side = _mm512_setzero_si512();
    side = _mm512_or_si512(side, _mm512_maskz_mov_epi16(_mm512_test_epi16_mask(descriptors, _mm512_set1_epi16(1 << 6)),
                                                        _mm512_set1_epi16(sz_line_break_side_pi_k)));
    side = _mm512_or_si512(side, _mm512_maskz_mov_epi16(_mm512_test_epi16_mask(descriptors, _mm512_set1_epi16(1 << 7)),
                                                        _mm512_set1_epi16(sz_line_break_side_pf_k)));
    side = _mm512_or_si512(side, _mm512_maskz_mov_epi16(_mm512_test_epi16_mask(descriptors, _mm512_set1_epi16(1 << 8)),
                                                        _mm512_set1_epi16(sz_line_break_side_eaw_k)));
    side = _mm512_or_si512(
        side, _mm512_maskz_mov_epi16(_mm512_test_epi16_mask(descriptors, _mm512_set1_epi16(1 << 9)),
                                     _mm512_set1_epi16(sz_line_break_side_cn_k | sz_line_break_side_ext_k)));
    __m512i const raw_class = _mm512_and_si512(descriptors, _mm512_set1_epi16(0x3F));
    side = _mm512_or_si512(
        side, _mm512_maskz_mov_epi16(_mm512_cmpeq_epi16_mask(raw_class, _mm512_set1_epi16(sz_line_break_ri_k)),
                                     _mm512_set1_epi16(sz_line_break_side_ri_k)));
    side = _mm512_or_si512(
        side, _mm512_maskz_mov_epi16(_mm512_cmpeq_epi16_mask(raw_class, _mm512_set1_epi16(sz_line_break_zwj_k)),
                                     _mm512_set1_epi16(sz_line_break_side_zwj_k)));
    __mmask32 const class_is_mark = _mm512_cmpeq_epi16_mask(classes, _mm512_set1_epi16(sz_line_break_cm_k)) |
                                    _mm512_cmpeq_epi16_mask(classes, _mm512_set1_epi16(sz_line_break_zwj_k));
    side = _mm512_or_si512(side, _mm512_maskz_mov_epi16(class_is_mark, _mm512_set1_epi16(sz_line_break_side_mark_k)));

    *classes_out = classes;
    *side_out = side;
    *dotted_out = _mm512_test_epi16_mask(descriptors, _mm512_set1_epi16(1 << 13));
}

/** @brief Expand sixty-four flat-palette indices to the LB1-resolved class byte, the engine side byte and the
 *         DottedCircle lane mask. The 56-entry `flat_palette_` of 16-bit descriptors is padded to 64 words = two
 *         aligned ZMM tiles, so ONE `vpermi2w` per 32-lane half resolves every index (the permute consumes the low
 *         six index bits, exactly the padded palette's span); the descriptor halves then feed the shared
 *         @ref sz_line_break_descriptor_unpack_half_icelake_ and narrow back to bytes with `vpmovwb`. Bit-identical
 *         to the cascade's `palette_class_` / `_side_` / `_dotted_` byte-table permutes, which carry the same
 *         resolution baked in. */
SZ_HELPER_AUTO void sz_line_break_flat_palette_unpack_icelake_(__m512i palette_indices_u8x64,
                                                               __m512i *classes_u8x64_out, __m512i *side_u8x64_out,
                                                               sz_u64_t *dotted_out) {
    __m512i const palette_low_tile_u16x32 = _mm512_load_si512((void const *)sz_utf8_line_break_flat_palette_);
    __m512i const palette_high_tile_u16x32 = _mm512_load_si512((void const *)(sz_utf8_line_break_flat_palette_ + 32));
    __m512i const indices_low_u16x32 = _mm512_cvtepu8_epi16(_mm512_castsi512_si256(palette_indices_u8x64));
    __m512i const indices_high_u16x32 = _mm512_cvtepu8_epi16(_mm512_extracti64x4_epi64(palette_indices_u8x64, 1));
    __m512i const descriptors_low_u16x32 = _mm512_permutex2var_epi16(palette_low_tile_u16x32, indices_low_u16x32,
                                                                     palette_high_tile_u16x32);
    __m512i const descriptors_high_u16x32 = _mm512_permutex2var_epi16(palette_low_tile_u16x32, indices_high_u16x32,
                                                                      palette_high_tile_u16x32);

    __m512i classes_low_u16x32, classes_high_u16x32, side_low_u16x32, side_high_u16x32;
    __mmask32 dotted_low_mask32, dotted_high_mask32;
    sz_line_break_descriptor_unpack_half_icelake_(descriptors_low_u16x32, &classes_low_u16x32, &side_low_u16x32,
                                                  &dotted_low_mask32);
    sz_line_break_descriptor_unpack_half_icelake_(descriptors_high_u16x32, &classes_high_u16x32, &side_high_u16x32,
                                                  &dotted_high_mask32);
    *classes_u8x64_out = _mm512_inserti64x4(_mm512_castsi256_si512(_mm512_cvtepi16_epi8(classes_low_u16x32)),
                                            _mm512_cvtepi16_epi8(classes_high_u16x32), 1);
    *side_u8x64_out = _mm512_inserti64x4(_mm512_castsi256_si512(_mm512_cvtepi16_epi8(side_low_u16x32)),
                                         _mm512_cvtepi16_epi8(side_high_u16x32), 1);
    *dotted_out = (sz_u64_t)_cvtmask32_u32(dotted_low_mask32) | ((sz_u64_t)_cvtmask32_u32(dotted_high_mask32) << 32);
}

/** @brief Per-window byte-lane classification: class/side per lane, plus the effective-start and U+FFFD masks. */
typedef struct sz_line_break_classified_t {
    __m512i classes;      /**< Per-byte-lane Line_Break class (valid only on `starts` lanes). */
    __m512i side;         /**< Per-byte-lane engine side byte. */
    sz_u64_t dotted;      /**< Bit i set => lane i is DottedCircle U+25CC. */
    sz_u64_t starts;      /**< Effective codepoint starts: valid leads (at their lane) + 1-byte U+FFFD units. */
    sz_u64_t replacement; /**< Effective-start lanes that are ill-formed (decoded as U+FFFD, class AL). */
    sz_u64_t non_start;   /**< Bytes that are NOT effective starts (consumed continuations) within `loaded`. */
    sz_size_t loaded;     /**< Bytes loaded into this window (<= 64). */
} sz_line_break_classified_t;

/**
 *  @brief  Classify a decoded 64-byte window onto byte-start lanes, fully in-register and zero-scalar. Reproduces
 *          the serial `sz_utf8_next_rune_` "consume-1 U+FFFD" policy: an invalid lead, a short/stray continuation, an
 *          overlong / surrogate / out-of-range lead each become a single-byte U+FFFD unit (class AL), so serial and
 *          icelake agree on malformed input. Valid leads classify by decoded VALUE (page / trie / big / astral),
 *          matching the serial resolution precedence. The BMP trie uses the shared substrate `trie_walk_icelake_`.
 */
SZ_HELPER_AUTO sz_line_break_classified_t sz_line_break_classify_window_icelake_(sz_utf8_rune_window_t window,
                                                                                 __m512i lane_identity) {
    sz_u64_t const loaded_mask = sz_u64_mask_until_(window.loaded);
    sz_u64_t const continuation = _cvtmask64_u64(window.continuation) & loaded_mask;
    sz_u64_t const two_byte = _cvtmask64_u64(window.two_byte_starts);
    sz_u64_t const three_byte = _cvtmask64_u64(window.three_byte_starts);
    sz_u64_t const four_byte = _cvtmask64_u64(window.four_byte_starts);
    __m512i const raw = window.window;
    __m512i const next1 = _mm512_permutexvar_epi8(_mm512_add_epi8(lane_identity, _mm512_set1_epi8(1)), raw);

    //  "next k lanes are continuations" tests, shifted into each lead's lane.
    sz_u64_t const next1_continuation = continuation >> 1, next2_continuation = continuation >> 2,
                   next3_continuation = continuation >> 3;

    //  Ill-formed-lead gate (LEVER B): the overlong / surrogate / out-of-range value checks only fire on the lead
    //  bytes C0/C1 (overlong-2), E0/ED (overlong-3 / surrogate-3) and F0/F4/>=F5 (overlong-4 / above-range-4). Real
    //  text never carries those, so detect their presence with ONE `raw`-only test and, when absent, take the cheap
    //  "lead + enough continuations" validity path. The gated branch runs the exact same overlong/surrogate/above
    //  algebra as before, so the result is bit-identical regardless of which side fires.
    sz_u64_t const lead_c0_c1 = _cvtmask64_u64(
        _kand_mask64(_mm512_cmp_epu8_mask(raw, _mm512_set1_epi8((char)0xC0), _MM_CMPINT_NLT),
                     _mm512_cmp_epu8_mask(raw, _mm512_set1_epi8((char)0xC1), _MM_CMPINT_LE)));
    sz_u64_t const lead_e0 = _cvtmask64_u64(_mm512_cmpeq_epi8_mask(raw, _mm512_set1_epi8((char)0xE0)));
    sz_u64_t const lead_ed = _cvtmask64_u64(_mm512_cmpeq_epi8_mask(raw, _mm512_set1_epi8((char)0xED)));
    sz_u64_t const lead_f0 = _cvtmask64_u64(_mm512_cmpeq_epi8_mask(raw, _mm512_set1_epi8((char)0xF0)));
    sz_u64_t const lead_f4_or_above = _cvtmask64_u64(
        _mm512_cmp_epu8_mask(raw, _mm512_set1_epi8((char)0xF4), _MM_CMPINT_NLT));
    sz_u64_t const danger_leads = (lead_c0_c1 | lead_e0 | lead_ed | lead_f0 | lead_f4_or_above) & loaded_mask;

    sz_u64_t valid2, valid3, valid4;
    if (!danger_leads) {
        //  Common case: no overlong / surrogate / out-of-range leads -> validity is purely "lead + continuations".
        valid2 = two_byte & next1_continuation;
        valid3 = three_byte & next1_continuation & next2_continuation;
        valid4 = four_byte & next1_continuation & next2_continuation & next3_continuation;
    }
    else {
        //  Per-lane value-bit predicates for overlong / surrogate / out-of-range detection (exact prior algebra).
        sz_u64_t const lead_not_overlong2 = _cvtmask64_u64(_mm512_test_epi8_mask(raw, _mm512_set1_epi8(0x1E)));
        sz_u64_t const b0_e0 = lead_e0;
        sz_u64_t const b0_ed = lead_ed;
        sz_u64_t const b0_f0 = lead_f0;
        sz_u64_t const b0_f4 = _cvtmask64_u64(_mm512_cmpeq_epi8_mask(raw, _mm512_set1_epi8((char)0xF4)));
        sz_u64_t const b0_above_f4 = _cvtmask64_u64(
            _mm512_cmp_epu8_mask(raw, _mm512_set1_epi8((char)0xF4), _MM_CMPINT_NLE));
        sz_u64_t const b1_lt_a0 = _cvtmask64_u64(_mm512_cmplt_epu8_mask(next1, _mm512_set1_epi8((char)0xA0)));
        sz_u64_t const b1_ge_a0 = _cvtmask64_u64(
            _mm512_cmp_epu8_mask(next1, _mm512_set1_epi8((char)0xA0), _MM_CMPINT_NLT));
        sz_u64_t const b1_lt_90 = _cvtmask64_u64(_mm512_cmplt_epu8_mask(next1, _mm512_set1_epi8((char)0x90)));
        sz_u64_t const b1_ge_90 = _cvtmask64_u64(
            _mm512_cmp_epu8_mask(next1, _mm512_set1_epi8((char)0x90), _MM_CMPINT_NLT));

        sz_u64_t const overlong3 = three_byte & b0_e0 & b1_lt_a0;
        sz_u64_t const surrogate3 = three_byte & b0_ed & b1_ge_a0;
        sz_u64_t const overlong4 = four_byte & b0_f0 & b1_lt_90;
        sz_u64_t const above4 = four_byte & ((b0_f4 & b1_ge_90) | b0_above_f4);
        valid2 = two_byte & next1_continuation & lead_not_overlong2;
        valid3 = three_byte & next1_continuation & next2_continuation & ~overlong3 & ~surrogate3;
        valid4 = four_byte & next1_continuation & next2_continuation & next3_continuation & ~overlong4 & ~above4;
    }
    sz_u64_t const true_ascii = _cvtmask64_u64(_mm512_cmplt_epu8_mask(raw, _mm512_set1_epi8((char)0x80))) &
                                loaded_mask & ~continuation;

    //  `decode_window_` reconstructs high/low with the 2/3-byte formula only: correct for 2/3-byte leads, garbage on
    //  ASCII lanes (codepoint IS the raw byte) and on 4-byte leads (true low 16 bits come from b1..b3). Rebuild both
    //  halves per lead length so cp = (plane<<16)|(high<<8)|low is exact. (Replacement lanes are overridden to
    //  U+FFFD downstream, so their garbage never reaches the palette.)
    __m512i const next2 = _mm512_permutexvar_epi8(_mm512_add_epi8(lane_identity, _mm512_set1_epi8(2)), raw);
    __m512i const next3 = _mm512_permutexvar_epi8(_mm512_add_epi8(lane_identity, _mm512_set1_epi8(3)), raw);
    __m512i const low_four = _mm512_or_si512(
        _mm512_and_si512(_mm512_slli_epi16(_mm512_and_si512(next2, _mm512_set1_epi8(0x03)), 6),
                         _mm512_set1_epi8((char)0xC0)),
        _mm512_and_si512(next3, _mm512_set1_epi8(0x3F)));
    __m512i const high_four = _mm512_or_si512(
        _mm512_and_si512(_mm512_slli_epi16(_mm512_and_si512(next1, _mm512_set1_epi8(0x0F)), 4),
                         _mm512_set1_epi8((char)0xF0)),
        sz_utf8_srl8_icelake_(next2, 2, 0x0F));
    __m512i low_fixed = _mm512_mask_mov_epi8(window.low, _cvtu64_mask64(true_ascii), raw);
    low_fixed = _mm512_mask_mov_epi8(low_fixed, _cvtu64_mask64(four_byte), low_four);
    __m512i high_fixed = _mm512_maskz_mov_epi8(_cvtu64_mask64(~true_ascii), window.high);
    high_fixed = _mm512_mask_mov_epi8(high_fixed, _cvtu64_mask64(four_byte), high_four);

    sz_u64_t const valid_start = true_ascii | valid2 | valid3 | valid4;
    //  Continuation bytes consumed by a valid multibyte lead's body (so they are NOT independent units).
    sz_u64_t const consumed = (((valid2 | valid3 | valid4) << 1) | ((valid3 | valid4) << 2) | (valid4 << 3)) &
                              continuation & loaded_mask;
    sz_u64_t const starts = loaded_mask & ~consumed;
    sz_u64_t const replacement = starts & ~valid_start;

    //  4-byte plane bits (bits 16..20 of the codepoint); zero on every non-4-byte lane so cp = (high<<8)|low there.
    __m512i const plane_all = _mm512_or_si512(
        _mm512_and_si512(_mm512_slli_epi16(_mm512_and_si512(raw, _mm512_set1_epi8(0x07)), 2), _mm512_set1_epi8(0x1C)),
        sz_utf8_srl8_icelake_(next1, 4, 0x03));
    __m512i const previous_cluster_lane = _mm512_maskz_mov_epi8(_cvtu64_mask64(four_byte), plane_all);

    //  Build the 64-lane flat-palette-index byte vector in ONE pass: the whole BMP (cp < 0x10000) resolves through
    //  the page-compressed flat leaf, four `vpgatherdd` on the load ports replacing the three-level trie walk and its
    //  page-LUT fast paths. Replacement lanes are forced to U+FFFD's index (U+FFFD is itself BMP, so it shares this
    //  index space) to match the serial U+FFFD policy. Astral lanes cannot join here: the astral trie still speaks the
    //  62-entry cascade palette, a DIFFERENT index space, so they are blended after the expansion, on class/side bytes.
    //  Only VALID 4-byte starts join that late blend: an invalid 4-byte lead is a replacement lane whose U+FFFD
    //  resolution must survive it, so `valid4`, not `four_byte`, gates the blend.
    sz_u64_t const is_astral = valid4 & loaded_mask;
    __m512i palette_indices_u8x64 = sz_line_break_bmp_flat_index_icelake_(high_fixed, low_fixed);
    if (replacement) {
        __m512i const replacement_indices_u8x64 = sz_line_break_bmp_flat_index_icelake_(_mm512_set1_epi8((char)0xFF),
                                                                                        _mm512_set1_epi8((char)0xFD));
        palette_indices_u8x64 = _mm512_mask_mov_epi8(palette_indices_u8x64, _cvtu64_mask64(replacement),
                                                     replacement_indices_u8x64);
    }

    //  Expand the flat-palette indices to the LB1-resolved class byte, the engine side byte and the DottedCircle mask.
    __m512i classes_u8x64, side_u8x64;
    sz_u64_t dotted;
    sz_line_break_flat_palette_unpack_icelake_(palette_indices_u8x64, &classes_u8x64, &side_u8x64, &dotted);

    if (is_astral) {
        //  Astral is rare: reconstruct the full 32-bit codepoint per sixteen-lane group and resolve through the
        //  shared astral trie, then blend the resulting indices back into the byte vector. No big-range scan. The
        //  groups are unrolled because `vextracti32x4`/`vinserti32x4` take an immediate lane selector.
        __m512i astral_index = _mm512_setzero_si512();
        __m512i high32, low32, plane32, codepoints, group_index;
        high32 = _mm512_cvtepu8_epi32(_mm512_extracti32x4_epi32(high_fixed, 0));
        low32 = _mm512_cvtepu8_epi32(_mm512_extracti32x4_epi32(low_fixed, 0));
        plane32 = _mm512_cvtepu8_epi32(_mm512_extracti32x4_epi32(previous_cluster_lane, 0));
        codepoints = _mm512_or_si512(_mm512_or_si512(_mm512_slli_epi32(plane32, 16), _mm512_slli_epi32(high32, 8)),
                                     low32);
        group_index = sz_line_break_classify_astral16_icelake_(codepoints);
        astral_index = _mm512_inserti32x4(astral_index, _mm512_cvtepi32_epi8(group_index), 0);
        high32 = _mm512_cvtepu8_epi32(_mm512_extracti32x4_epi32(high_fixed, 1));
        low32 = _mm512_cvtepu8_epi32(_mm512_extracti32x4_epi32(low_fixed, 1));
        plane32 = _mm512_cvtepu8_epi32(_mm512_extracti32x4_epi32(previous_cluster_lane, 1));
        codepoints = _mm512_or_si512(_mm512_or_si512(_mm512_slli_epi32(plane32, 16), _mm512_slli_epi32(high32, 8)),
                                     low32);
        group_index = sz_line_break_classify_astral16_icelake_(codepoints);
        astral_index = _mm512_inserti32x4(astral_index, _mm512_cvtepi32_epi8(group_index), 1);
        high32 = _mm512_cvtepu8_epi32(_mm512_extracti32x4_epi32(high_fixed, 2));
        low32 = _mm512_cvtepu8_epi32(_mm512_extracti32x4_epi32(low_fixed, 2));
        plane32 = _mm512_cvtepu8_epi32(_mm512_extracti32x4_epi32(previous_cluster_lane, 2));
        codepoints = _mm512_or_si512(_mm512_or_si512(_mm512_slli_epi32(plane32, 16), _mm512_slli_epi32(high32, 8)),
                                     low32);
        group_index = sz_line_break_classify_astral16_icelake_(codepoints);
        astral_index = _mm512_inserti32x4(astral_index, _mm512_cvtepi32_epi8(group_index), 2);
        high32 = _mm512_cvtepu8_epi32(_mm512_extracti32x4_epi32(high_fixed, 3));
        low32 = _mm512_cvtepu8_epi32(_mm512_extracti32x4_epi32(low_fixed, 3));
        plane32 = _mm512_cvtepu8_epi32(_mm512_extracti32x4_epi32(previous_cluster_lane, 3));
        codepoints = _mm512_or_si512(_mm512_or_si512(_mm512_slli_epi32(plane32, 16), _mm512_slli_epi32(high32, 8)),
                                     low32);
        group_index = sz_line_break_classify_astral16_icelake_(codepoints);
        astral_index = _mm512_inserti32x4(astral_index, _mm512_cvtepi32_epi8(group_index), 3);

        //  The astral index addresses the 62-entry cascade palette, whose `palette_class_` / `_side_` / `_dotted_`
        //  byte tables carry the very same LB1 resolution the flat descriptor unpack applies (verified entry by
        //  entry), so blending the RESOLVED bytes over the astral lanes matches blending indices in a single
        //  shared space, which the two palettes do not form.
        __mmask64 const astral_mask = _cvtu64_mask64(is_astral);
        classes_u8x64 = _mm512_mask_permutexvar_epi8(
            classes_u8x64, astral_mask, astral_index,
            _mm512_load_si512((void const *)sz_utf8_line_break_palette_class_));
        side_u8x64 = _mm512_mask_permutexvar_epi8(side_u8x64, astral_mask, astral_index,
                                                  _mm512_load_si512((void const *)sz_utf8_line_break_palette_side_));
        __m512i const astral_dotted_bytes_u8x64 = _mm512_permutexvar_epi8(
            astral_index, _mm512_load_si512((void const *)sz_utf8_line_break_palette_dotted_));
        sz_u64_t const astral_dotted = _cvtmask64_u64(
            _mm512_test_epi8_mask(astral_dotted_bytes_u8x64, astral_dotted_bytes_u8x64));
        dotted = (dotted & ~is_astral) | (astral_dotted & is_astral);
    }

    sz_line_break_classified_t result;
    result.classes = classes_u8x64;
    result.side = side_u8x64;
    result.dotted = dotted & starts;
    result.starts = starts;
    result.replacement = replacement;
    result.non_start = loaded_mask & ~starts;
    result.loaded = window.loaded;
    return result;
}

#pragma endregion In register vectorized classifier

#pragma region Mask algebra rule engine

/** @brief Build a 64-bit "lane class == @p cls" mask with one `vpcmpeqb` -> kmask. */
SZ_HELPER_INLINE sz_u64_t sz_line_break_class_mask_icelake_(__m512i classes, sz_u8_t cls) {
    return _cvtmask64_u64(_mm512_cmpeq_epi8_mask(classes, _mm512_set1_epi8((char)cls)));
}

/** @brief The class/side byte held at lane @p lane, extracted in-register by a single byte permute (no scalar loop). */
SZ_HELPER_INLINE sz_u8_t sz_line_break_byte_at_icelake_(__m512i lanes, sz_size_t lane) {
    __m512i const broadcast = _mm512_permutexvar_epi8(_mm512_set1_epi8((char)lane), lanes);
    return (sz_u8_t)_mm_cvtsi128_si32(_mm512_castsi512_si128(broadcast));
}

/** @brief Build a 64-bit "lane (side & @p bit) != 0" mask with one `vptestmb` -> kmask. */
SZ_HELPER_INLINE sz_u64_t sz_line_break_side_mask_icelake_(__m512i side_lo, sz_u8_t bit) {
    __m512i const masked = _mm512_and_si512(side_lo, _mm512_set1_epi8((char)bit));
    return _cvtmask64_u64(_mm512_test_epi8_mask(masked, masked));
}

/** @brief Build a 64-bit "lane class is in the inclusive byte range [@p lo, @p hi]" mask with two `vpcmpub` -> kmask.
 *         Used as a cheap combined presence test that gates the rarely-fired script blocks (Hangul, Brahmic) without
 *         extracting each individual per-class mask on the common Latin/CJK path. */
SZ_HELPER_INLINE sz_u64_t sz_line_break_class_range_mask_icelake_(__m512i classes, sz_u8_t lo, sz_u8_t hi) {
    __mmask64 const ge = _mm512_cmp_epu8_mask(classes, _mm512_set1_epi8((char)lo), _MM_CMPINT_NLT);
    __mmask64 const le = _mm512_cmp_epu8_mask(classes, _mm512_set1_epi8((char)hi), _MM_CMPINT_LE);
    return _cvtmask64_u64(_kand_mask64(ge, le));
}

/** @brief Byte-lane gate/base derivation for the byte-level rule engine: identifies cluster bases, the transparent
 *         gate (continuations + attached combining marks), and reclassifies lone marks (LB10) to AL in @p classes. */
typedef struct sz_line_break_byte_frame_t {
    __m512i classes;   /**< Class per lane with lone marks reclassified to AL (LB10). */
    sz_u64_t base;     /**< Cluster-base lanes (every effective start except an attached CM/ZWJ). */
    sz_u64_t gate;     /**< Transparent lanes for neighbour fills: continuations + attached-mark starts. */
    sz_u64_t attached; /**< Attached CM/ZWJ start lanes (LB9). */
    sz_u64_t
        lone_mark; /**< LB10 lone marks reclassified to AL; their side bits must be cleared (serial zeros the descriptor). */
} sz_line_break_byte_frame_t;

SZ_HELPER_INLINE sz_line_break_byte_frame_t sz_line_break_byte_frame_icelake_(sz_line_break_classified_t classified) {
    sz_u64_t const starts = classified.starts, non_start = classified.non_start;
    sz_u64_t const mark_start = (sz_line_break_class_mask_icelake_(classified.classes, sz_line_break_cm_k) |
                                 sz_line_break_class_mask_icelake_(classified.classes, sz_line_break_zwj_k)) &
                                starts;
    sz_u64_t const excluded = (sz_line_break_class_mask_icelake_(classified.classes, sz_line_break_bk_k) |
                               sz_line_break_class_mask_icelake_(classified.classes, sz_line_break_cr_k) |
                               sz_line_break_class_mask_icelake_(classified.classes, sz_line_break_lf_k) |
                               sz_line_break_class_mask_icelake_(classified.classes, sz_line_break_nl_k) |
                               sz_line_break_class_mask_icelake_(classified.classes, sz_line_break_sp_k) |
                               sz_line_break_class_mask_icelake_(classified.classes, sz_line_break_zw_k)) &
                              starts;
    sz_u64_t const good_base = starts & ~excluded & ~mark_start;
    //  A mark attaches (LB9) when reachable from a good base across only continuations and other marks. Flood each
    //  good base rightward over (continuations | mark starts); the mark starts it reaches are the attached marks.
    sz_u64_t const mark_bytes = sz_u64_fill_right_(mark_start, non_start) | mark_start;
    sz_u64_t const flood = sz_u64_fill_right_(good_base, non_start | mark_bytes);
    sz_u64_t const attached = flood & mark_start;
    sz_u64_t const lone_mark = mark_start & ~attached; // LB10: a mark with no attachable base acts as AL

    sz_line_break_byte_frame_t frame;
    frame.classes = _mm512_mask_mov_epi8(classified.classes, _cvtu64_mask64(lone_mark),
                                         _mm512_set1_epi8((char)sz_line_break_al_k));
    frame.base = starts & ~attached;
    frame.gate = non_start | attached;
    frame.attached = attached;
    frame.lone_mark = lone_mark;
    return frame;
}

/**
 *  @brief  Per-ISA extractor: lower one classified 64-byte window to the portable @ref sz_line_break_frame_t. Builds
 *          the byte-level cluster frame (LB9/LB10), then materializes per-class membership (after the LB10 lone-mark
 *          -> AL reclassify), the raw ZWJ mask, the five side-bit masks, and the per-lane class/side bytes for the
 *          carry-out reads. All `__m512i` -> `sz_u64_t` work lives here; the rule engine that consumes the frame is
 *          fully portable.
 */
SZ_HELPER_INLINE sz_line_break_frame_t sz_line_break_build_frame_icelake_(sz_line_break_classified_t classified,
                                                                          sz_u8_t *effective_class_byte_out,
                                                                          sz_u8_t *side_byte_out) {
    sz_line_break_byte_frame_t const byte_frame = sz_line_break_byte_frame_icelake_(classified);
    __m512i const classes = byte_frame.classes;
    //  LB10 reclassifies a lone CM/ZWJ to AL; its descriptor side bits (EAW/Pi/Pf/...) must go with it, else LB19/LB15
    //  see a phantom East-Asian / quote cluster. Mirrors the serial path zeroing `codepoint_descriptors` on LB10.
    __m512i const side = _mm512_maskz_mov_epi8(_cvtu64_mask64(~byte_frame.lone_mark), classified.side);

    sz_line_break_frame_t frame;
    frame.base = byte_frame.base;
    frame.gate = byte_frame.gate;
    frame.attached = byte_frame.attached;
    frame.lone_mark = byte_frame.lone_mark;
    frame.non_start = classified.non_start;
    frame.dotted = classified.dotted;
    frame.starts = classified.starts;
    frame.replacement = classified.replacement;
    //  Full unroll is load-bearing: with a runtime loop index the compiler keeps `effective_class` stack-resident, so
    //  every per-class read in the rule engine becomes a load (~7% slower). Unrolled, SROA promotes the array to
    //  registers and drops the `vpcmpeqb` for classes the engine never reads.
#if defined(__clang__)
#pragma clang loop unroll(full)
#elif defined(__GNUC__)
#pragma GCC unroll 64
#endif
    for (sz_size_t cls = 0; cls < sz_line_break_class_count_k; ++cls)
        frame.effective_class[cls] = sz_line_break_class_mask_icelake_(classes, (sz_u8_t)cls);
    frame.raw_zwj = sz_line_break_class_mask_icelake_(classified.classes, sz_line_break_zwj_k);
    frame.side_pi = sz_line_break_side_mask_icelake_(side, sz_line_break_side_pi_k);
    frame.side_pf = sz_line_break_side_mask_icelake_(side, sz_line_break_side_pf_k);
    frame.side_eaw = sz_line_break_side_mask_icelake_(side, sz_line_break_side_eaw_k);
    frame.side_cn = sz_line_break_side_mask_icelake_(side, sz_line_break_side_cn_k);
    frame.side_ext = sz_line_break_side_mask_icelake_(side, sz_line_break_side_ext_k);
    _mm512_storeu_si512((void *)effective_class_byte_out, classes);
    _mm512_storeu_si512((void *)side_byte_out, side);
    return frame;
}

/**
 *  @brief  Byte-level UAX-14 rule engine, Ice Lake entry: extract the portable frame in-register, then delegate every
 *          LB1-LB31 decision to the portable @ref sz_line_break_decide_window_. Kept as the driver's call target so the
 *          forward driver is unchanged.
 */
SZ_HELPER_INLINE sz_line_break_window_t sz_line_break_decide_window_icelake_(sz_line_break_classified_t classified,
                                                                             sz_line_break_carry_t carry,
                                                                             sz_line_break_carry_t *carry_out,
                                                                             sz_size_t complete_limit,
                                                                             sz_bool_t more_text) {
    sz_u8_t effective_class_byte[64], side_byte[64];
    sz_line_break_frame_t const frame = sz_line_break_build_frame_icelake_(classified, effective_class_byte, side_byte);
    return sz_line_break_decide_window_(&frame, effective_class_byte, side_byte, carry, carry_out, complete_limit,
                                        more_text);
}

#pragma endregion Mask algebra rule engine

#pragma region Forward driver

/**
 *  @brief  Largest byte prefix of the window whose codepoints are all fully loaded (no multi-byte lead straddles the
 *          64-byte edge). Mirrors the word kernel's complete-limit: a declared-length lead whose span exceeds `loaded`
 *          ends the trusted region just before it; with no more text the whole window is complete. Never below 1.
 */
SZ_HELPER_AUTO sz_size_t sz_line_break_complete_limit_(sz_utf8_rune_window_t window, sz_bool_t more_text) {
    sz_size_t const loaded = window.loaded;
    if (!more_text) return loaded;
    sz_u64_t const valid = sz_u64_mask_until_(loaded);
    sz_u64_t const starts = _cvtmask64_u64(window.codepoint_starts) & valid;
    sz_u64_t const two = _cvtmask64_u64(window.two_byte_starts) & starts;
    sz_u64_t const three = _cvtmask64_u64(window.three_byte_starts) & starts;
    sz_u64_t const four = _cvtmask64_u64(window.four_byte_starts) & starts;
    //  A multi-byte lead straddles when its declared span runs past `loaded`: a 2-byte lead in the top 1 lane, a
    //  3-byte lead in the top 2 lanes, a 4-byte lead in the top 3 lanes. Trust ends at the lowest such lead.
    sz_u64_t const straddle = ((two & ~sz_u64_mask_until_(loaded > 1 ? loaded - 1 : 0)) |
                               (three & ~sz_u64_mask_until_(loaded > 2 ? loaded - 2 : 0)) |
                               (four & ~sz_u64_mask_until_(loaded > 3 ? loaded - 3 : 0))) &
                              valid;
    sz_size_t const limit = straddle ? (sz_size_t)sz_u64_ctz(straddle) : loaded;
    return limit > 0 ? limit : loaded; // `limit == 0` keeps the whole window for guaranteed progress
}

/**
 *  @brief  Byte-level zero-scalar forward UAX-14 kernel: an overlap-free advancing driver. Each iteration decodes one
 *          64-byte window at the codepoint-aligned `position`, classifies it in-register, decides the break-before mask
 *          over the complete-codepoint region with a carry-aware rule engine, drains the trusted band below the trust
 *          horizon, and advances by `win.resolved` with the small register carry threaded forward. There is NO byte
 *          re-read (no `start_at_or_before_`, no left-context back-walk): the carry alone supplies lane-0 left context.
 *          `bytes_consumed` is always a confirmed break (`line_start`), so resume is bit-identical and capacity-free.
 */
SZ_API_COMPTIME sz_size_t sz_utf8_linebreaks_icelake_bytes_( //
    sz_cptr_t text, sz_size_t length,                        //
    sz_size_t *starts, sz_size_t *lengths,                   //
    sz_size_t capacity, sz_size_t *bytes_consumed) {

    if (length == 0 || capacity == 0) {
        if (bytes_consumed) *bytes_consumed = 0;
        return 0;
    }
    sz_u8_t const *bytes = (sz_u8_t const *)text;
    __m512i const lane_identity = sz_utf8_lane_identity_icelake_();
    sz_size_t produced = 0;
    sz_size_t line_start = 0;                                 // open line's first byte: a confirmed break (or 0)
    sz_size_t position = 0;                                   // codepoint-aligned anchor of the next window
    sz_line_break_carry_t carry = sz_line_break_carry_sot_(); // LB2: start-of-text has no left context

    while (position < length) {
        sz_utf8_rune_window_t const window = sz_utf8_rune_decode_window_icelake_(bytes + position, length - position,
                                                                                 lane_identity);
        sz_bool_t const more_text = (sz_bool_t)(position + window.loaded < length);
        sz_size_t const complete_limit = sz_line_break_complete_limit_(window, more_text);
        sz_line_break_classified_t const classified = sz_line_break_classify_window_icelake_(window, lane_identity);

        sz_line_break_carry_t carry_next = carry;
        sz_line_break_window_t const win = sz_line_break_decide_window_icelake_(classified, carry, &carry_next,
                                                                                complete_limit, more_text);
        sz_u64_t const commit = win.breaks & sz_u64_mask_until_(win.resolved);

        produced = sz_utf8_rune_drain_forward_(commit, position, lane_identity, starts, lengths, produced, capacity,
                                               &line_start);
        if (produced >= capacity) {
            if (bytes_consumed) *bytes_consumed = line_start;
            return produced;
        }

        //  Advance by the trust horizon when it bit before the complete edge, else the whole complete span (guaranteed
        //  progress). `carry_next` is already anchored at exactly this byte by `decide_window_` -- one decision per
        //  window, no second pass.
        sz_size_t const advance = win.resolved ? win.resolved : complete_limit;
        carry = carry_next;
        position += advance ? advance : window.loaded;
    }

    //  The trailing (still-open) line `[line_start, length)` finalizes the output (end of text is a break).
    if (produced < capacity) starts[produced] = line_start, lengths[produced] = length - line_start, ++produced;
    if (bytes_consumed) *bytes_consumed = length;
    return produced;
}

/**
 *  @brief  Forward UAX-14 line-break-opportunity kernel (Ice Lake AVX-512).
 *
 *  Bit-exact with `sz_utf8_linebreaks_serial`. Emits every UAX-14 break opportunity (no per-segment
 *  mandatory flag). The classifier and rule engine are fully vectorized (no per-lane scalar loop, no
 *  serial-oracle deferral); the 64-codepoint block engine threads cross-block state through a register carry
 *  (no halo back-scan), so throughput is flat in run length. Emits at most @p capacity segments; sets
 *  *@p bytes_consumed to the resume.
 */
SZ_API_COMPTIME sz_size_t sz_utf8_linebreaks_icelake( //
    sz_cptr_t text, sz_size_t length,                 //
    sz_size_t *starts, sz_size_t *lengths,            //
    sz_size_t capacity, sz_size_t *bytes_consumed) {

    return sz_utf8_linebreaks_icelake_bytes_(text, length, starts, lengths, capacity, bytes_consumed);
}

#pragma endregion Forward driver

#pragma endregion UAX 14 Line Boundaries forward kernel
#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#endif // SZ_USE_ICELAKE

#ifdef __cplusplus
}
#endif

#endif // STRINGZILLA_UTF8_LINEBREAKS_ICELAKE_H_
