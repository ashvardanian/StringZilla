/**
 *  @brief NEON (AArch64) backend for UAX-14 line break boundaries.
 *  @file include/stringzilla/utf8_linebreaks/neon.h
 *  @author Ash Vardanian
 */
#ifndef STRINGZILLA_UTF8_LINEBREAKS_NEON_H_
#define STRINGZILLA_UTF8_LINEBREAKS_NEON_H_

#include "stringzilla/types.h"
#include "stringzilla/utf8_linebreaks/tables.h"
#include "stringzilla/utf8_linebreaks/serial.h"
#include "stringzilla/utf8_runes/neon.h"

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

#pragma region UAX 14 Line Boundaries forward kernel

#pragma region In register vectorized classifier

/*  The NEON twin of the Ice Lake / AVX2 classifier: a contiguous run of codepoints resolves to per-codepoint
 *  (class, side, dotted) with ZERO per-lane scalar loop, ZERO `vpgather`, and NO serial deferral. Each 64-byte
 *  window lives as four `uint8x16_t` quarters (`window[0]` = lanes [0,16), ... `window[3]` = lanes [48,64)) instead
 *  of haswell's two halves; every per-lane class compare is `vceqq_u8` per quarter, the four boolean quarters
 *  OR-collapsed to a `sz_u64_t` via `mask_combine_neon_`. The BMP palette index comes from the
 *  `sz_line_break_bmp_index_neon_` nibble cascade (the NEON twin of the VBMI full-BMP trie); the astral path uses
 *  `sz_line_break_classify_astral_neon_`. The 62-entry palette descriptor is unpacked to the LB1-resolved class byte
 *  and the engine side byte by `lut256_neon_` reads of the (ISA-agnostic) palette tables, bit-identical to the
 *  icelake/haswell descriptor unpack. */

/** @brief Expand a 16-bit lane mask into a `uint8x16_t` select vector (byte `i` = 0xFF when bit `i` is set), the
 *         NEON twin of @ref sz_utf8_byte_mask_from_bits_haswell_ (which expands 32 bits to a `__m256i`).
 *         Inverse of `movemask16_neon_`: route the right mask byte to each lane, isolate its bit, then `vceqq`. */
SZ_HELPER_INLINE uint8x16_t sz_line_break_byte_mask_from_bits_neon_(sz_u64_t bits) {
    static sz_u8_t const byte_router_lanes[16] = {0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1};
    static sz_u8_t const bit_select_lanes[16] = {1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128};
    uint8x16_t const broadcast = vreinterpretq_u8_u16(vdupq_n_u16((sz_u16_t)bits));
    uint8x16_t const byte_router = vld1q_u8(byte_router_lanes);
    uint8x16_t const spread = vqtbl1q_u8(broadcast, byte_router);
    uint8x16_t const bit_select = vld1q_u8(bit_select_lanes);
    uint8x16_t const isolated = vandq_u8(spread, bit_select);
    return vceqq_u8(isolated, bit_select);
}

/** @brief Palette index for sixteen BMP codepoints (per-lane high = cp>>8, low = cp&0xFF) via a register-resident
 *         3-stage `vqtbl` nibble cascade, the NEON twin of @ref sz_line_break_bmp_index_haswell_ (and the VBMI
 *         full-BMP trie). Gather-free; bit-exact with `sz_rune_line_break_property` over the whole BMP. Operates on
 *         one quarter; the caller iterates the four quarters. */
SZ_HELPER_AUTO uint8x16_t sz_line_break_bmp_index_neon_(uint8x16_t high, uint8x16_t low) {
    uint8x16_t const low_nibble_mask = vdupq_n_u8(0x0F);
    uint8x16_t const high_high = vandq_u8(vshrq_n_u8(high, 4), low_nibble_mask);
    uint8x16_t const high_low = vandq_u8(high, low_nibble_mask);
    uint8x16_t const page = sz_utf8_rune_cascade_stage_neon_(
        sz_utf8_line_break_haswell_stage1_, sz_utf8_line_break_haswell_stage1_count_k / 16, high_high, high_low);
    uint8x16_t const low_high = vandq_u8(vshrq_n_u8(low, 4), low_nibble_mask);
    uint8x16_t const leaf_lo = sz_utf8_rune_cascade_stage_neon_(
        sz_utf8_line_break_haswell_stage2_lo_, sz_utf8_line_break_haswell_stage2_lo_count_k / 16, page, low_high);
    uint8x16_t const leaf_hi = sz_utf8_rune_cascade_stage_neon_(
        sz_utf8_line_break_haswell_stage2_hi_, sz_utf8_line_break_haswell_stage2_hi_count_k / 16, page, low_high);
    uint8x16_t const leaf_group = vorrq_u8(vandq_u8(vshrq_n_u8(leaf_lo, 4), low_nibble_mask), vshlq_n_u8(leaf_hi, 4));
    uint8x16_t const leaf_low_nibble = vandq_u8(leaf_lo, low_nibble_mask);
    uint8x16_t const low_low = vandq_u8(low, low_nibble_mask);
    uint8x16_t const lut_index = vorrq_u8(vshlq_n_u8(leaf_low_nibble, 4), low_low);
    uint8x16_t result = vdupq_n_u8(0);
    for (int group = 0; group < (int)sz_utf8_line_break_haswell_leaf_groups_k; ++group) {
        uint8x16_t const value = sz_utf8_rune_lut256_neon_(sz_utf8_line_break_haswell_stage3_groups_ + group * 256,
                                                           lut_index);
        uint8x16_t const here = vceqq_u8(leaf_group, vdupq_n_u8((sz_u8_t)group));
        result = vbslq_u8(here, value, result);
    }
    return result;
}

/** @brief Palette index for sixteen ASTRAL codepoints over the 20-bit offset = cp - 0x10000 (5-nibble cascade),
 *         the NEON twin of @ref sz_line_break_classify_astral_haswell_. Per-lane bytes: @p plane = (offset>>16)&0xFF
 *         (low nibble meaningful), @p high = (offset>>8)&0xFF, @p low = offset&0xFF. Gather-free; bit-exact. */
SZ_HELPER_AUTO uint8x16_t sz_line_break_classify_astral_neon_(uint8x16_t plane, uint8x16_t high, uint8x16_t low) {
    uint8x16_t const low_nibble_mask = vdupq_n_u8(0x0F);
    uint8x16_t const n4 = vandq_u8(plane, low_nibble_mask);
    uint8x16_t const n3 = vandq_u8(vshrq_n_u8(high, 4), low_nibble_mask);
    uint8x16_t const stage1_index = vorrq_u8(vshlq_n_u8(n4, 4), n3);
    uint8x16_t const page = sz_utf8_rune_lut256_neon_(sz_utf8_line_break_haswell_astral_stage1_, stage1_index);
    uint8x16_t const n2 = vandq_u8(high, low_nibble_mask);
    uint8x16_t const leaf2_lo = sz_utf8_rune_cascade_stage_neon_(
        sz_utf8_line_break_haswell_astral_stage2_lo_, sz_utf8_line_break_haswell_astral_stage2_lo_count_k / 16, page,
        n2);
    uint8x16_t const n1 = vandq_u8(vshrq_n_u8(low, 4), low_nibble_mask);
    uint8x16_t const leaf_lo = sz_utf8_rune_cascade_stage_neon_(
        sz_utf8_line_break_haswell_astral_stage3_lo_, sz_utf8_line_break_haswell_astral_stage3_lo_count_k / 16,
        leaf2_lo, n1);
    uint8x16_t const leaf_hi = sz_utf8_rune_cascade_stage_neon_(
        sz_utf8_line_break_haswell_astral_stage3_hi_, sz_utf8_line_break_haswell_astral_stage3_hi_count_k / 16,
        leaf2_lo, n1);
    uint8x16_t const n0 = vandq_u8(low, low_nibble_mask);
    uint8x16_t const leaf_group = vorrq_u8(vandq_u8(vshrq_n_u8(leaf_lo, 4), low_nibble_mask), vshlq_n_u8(leaf_hi, 4));
    uint8x16_t const leaf_low_nibble = vandq_u8(leaf_lo, low_nibble_mask);
    uint8x16_t const stage4_lut_index = vorrq_u8(vshlq_n_u8(leaf_low_nibble, 4), n0);
    uint8x16_t result = vdupq_n_u8(0);
    for (int group = 0; group < (int)sz_utf8_line_break_haswell_astral_leaf_groups_k; ++group) {
        uint8x16_t const value = sz_utf8_rune_lut256_neon_(
            sz_utf8_line_break_haswell_astral_stage4_groups_ + group * 256, stage4_lut_index);
        uint8x16_t const here = vceqq_u8(leaf_group, vdupq_n_u8((sz_u8_t)group));
        result = vbslq_u8(here, value, result);
    }
    return result;
}

/** @brief A 64-bit "(byte & mask) == pattern" lane mask over the four window quarters. */
SZ_HELPER_INLINE sz_u64_t sz_line_break_byte_match_neon_(uint8x16_t const *quarters, sz_u8_t mask, sz_u8_t pattern) {
    uint8x16_t const m = vdupq_n_u8(mask), p = vdupq_n_u8(pattern);
    return sz_utf8_mask_combine_neon_(vceqq_u8(vandq_u8(quarters[0], m), p), vceqq_u8(vandq_u8(quarters[1], m), p),
                                      vceqq_u8(vandq_u8(quarters[2], m), p), vceqq_u8(vandq_u8(quarters[3], m), p));
}

/** @brief A 64-bit "byte == value" lane mask over the four window quarters. */
SZ_HELPER_INLINE sz_u64_t sz_line_break_byte_equal_neon_(uint8x16_t const *quarters, sz_u8_t value) {
    uint8x16_t const v = vdupq_n_u8(value);
    return sz_utf8_mask_combine_neon_(vceqq_u8(quarters[0], v), vceqq_u8(quarters[1], v), vceqq_u8(quarters[2], v),
                                      vceqq_u8(quarters[3], v));
}

/** @brief A 64-bit "byte >= bound" (unsigned) lane mask over the four window quarters (`vcgeq_u8` is native). */
SZ_HELPER_INLINE sz_u64_t sz_line_break_byte_ge_neon_(uint8x16_t const *quarters, sz_u8_t bound) {
    uint8x16_t const bound_vec = vdupq_n_u8(bound);
    return sz_utf8_mask_combine_neon_(vcgeq_u8(quarters[0], bound_vec), vcgeq_u8(quarters[1], bound_vec),
                                      vcgeq_u8(quarters[2], bound_vec), vcgeq_u8(quarters[3], bound_vec));
}

/** @brief A 64-bit "byte < bound" (unsigned) lane mask over the four window quarters (`vcltq_u8` is native). */
SZ_HELPER_INLINE sz_u64_t sz_line_break_byte_lt_neon_(uint8x16_t const *quarters, sz_u8_t bound) {
    uint8x16_t const bound_vec = vdupq_n_u8(bound);
    return sz_utf8_mask_combine_neon_(vcltq_u8(quarters[0], bound_vec), vcltq_u8(quarters[1], bound_vec),
                                      vcltq_u8(quarters[2], bound_vec), vcltq_u8(quarters[3], bound_vec));
}

/** @brief Per-window byte-lane classification (NEON): class/side per lane as four `uint8x16_t` quarters plus the
 *         effective-start and U+FFFD masks. The NEON twin of @ref sz_line_break_classified_haswell_t. */
typedef struct sz_line_break_classified_neon_t {
    uint8x16_t classes[4]; /**< Per-byte-lane Line_Break class (valid only on `starts` lanes), per quarter. */
    uint8x16_t side[4];    /**< Per-byte-lane engine side byte, per quarter. */
    sz_u64_t dotted;       /**< Bit i set => lane i is DottedCircle U+25CC. */
    sz_u64_t starts;       /**< Effective codepoint starts: valid leads (at their lane) + 1-byte U+FFFD units. */
    sz_u64_t replacement;  /**< Effective-start lanes that are ill-formed (decoded as U+FFFD, class AL). */
    sz_u64_t non_start;    /**< Bytes that are NOT effective starts (consumed continuations) within `loaded`. */
    sz_size_t loaded;      /**< Bytes loaded into this window (<= 64). */
} sz_line_break_classified_neon_t;

/** @brief Resolve the per-lane palette index (one quarter) to class / side / dotted bytes through the precomputed
 *         62-entry palette tables. `lut256_neon_` reads each 64-entry table by `index` (index < 64); bit-identical to
 *         the icelake `vpermb` palette permute and the haswell `lut256` reads. */
SZ_HELPER_INLINE void sz_line_break_palette_unpack_neon_(uint8x16_t index, uint8x16_t *classes, uint8x16_t *side,
                                                         uint8x16_t *dotted) {
    *classes = sz_utf8_rune_lut256_neon_(sz_utf8_line_break_palette_class_, index);
    *side = sz_utf8_rune_lut256_neon_(sz_utf8_line_break_palette_side_, index);
    *dotted = sz_utf8_rune_lut256_neon_(sz_utf8_line_break_palette_dotted_, index);
}

/**
 *  @brief  Classify a decoded 64-byte window onto byte-start lanes, fully in-register and zero-scalar — the NEON
 *          twin of @ref sz_line_break_classify_window_haswell_, bit-identical on every lane. Reproduces the serial
 *          "consume-1 U+FFFD" malformed policy: an invalid lead / short or stray continuation / overlong /
 *          surrogate / out-of-range lead each become one single-byte U+FFFD unit (class AL).
 */
SZ_HELPER_AUTO sz_line_break_classified_neon_t sz_line_break_classify_window_neon_(sz_utf8_rune_window_neon_t window) {
    sz_u64_t const loaded_mask = sz_u64_mask_until_serial_(window.loaded);
    sz_u64_t const continuation = window.continuation & loaded_mask;
    sz_u64_t const two_byte = window.two_byte_starts;
    sz_u64_t const three_byte = window.three_byte_starts;
    sz_u64_t const four_byte = window.four_byte_starts;
    uint8x16_t const *raw = window.window;

    //  Forward neighbours (mod-64 wrap, matching icelake's `_mm512_permutexvar_epi8`).
    uint8x16_t next1[4], next2[4], next3[4];
    sz_utf8_forward_neighbours_neon_(raw, next1, next2, next3);

    sz_u64_t const next1_continuation = continuation >> 1, next2_continuation = continuation >> 2,
                   next3_continuation = continuation >> 3;

    //  Ill-formed-lead gate (LEVER B), bit-identical to icelake/haswell: only C0/C1, E0, ED, F0, F4/>=F5 leads can
    //  be overlong / surrogate / out-of-range; detect their presence with raw-only tests and, when absent, take the
    //  cheap "lead + enough continuations" validity path.
    sz_u64_t const lead_c0_c1 = sz_line_break_byte_ge_neon_(raw, 0xC0) & sz_line_break_byte_lt_neon_(raw, 0xC2);
    sz_u64_t const lead_e0 = sz_line_break_byte_equal_neon_(raw, 0xE0);
    sz_u64_t const lead_ed = sz_line_break_byte_equal_neon_(raw, 0xED);
    sz_u64_t const lead_f0 = sz_line_break_byte_equal_neon_(raw, 0xF0);
    sz_u64_t const lead_f4_or_above = sz_line_break_byte_ge_neon_(raw, 0xF4);
    sz_u64_t const danger_leads = (lead_c0_c1 | lead_e0 | lead_ed | lead_f0 | lead_f4_or_above) & loaded_mask;

    sz_u64_t valid2, valid3, valid4;
    if (!danger_leads) {
        valid2 = two_byte & next1_continuation;
        valid3 = three_byte & next1_continuation & next2_continuation;
        valid4 = four_byte & next1_continuation & next2_continuation & next3_continuation;
    }
    else {
        //  Exact prior algebra: per-lane value-bit predicates for overlong / surrogate / out-of-range detection.
        sz_u64_t const lead_not_overlong2 = ~sz_line_break_byte_match_neon_(raw, 0x1E, 0x00); // (raw & 0x1E) != 0
        sz_u64_t const b0_e0 = lead_e0, b0_ed = lead_ed, b0_f0 = lead_f0;
        sz_u64_t const b0_f4 = sz_line_break_byte_equal_neon_(raw, 0xF4);
        sz_u64_t const b0_above_f4 = sz_line_break_byte_ge_neon_(raw, 0xF5);
        sz_u64_t const b1_lt_a0 = sz_line_break_byte_lt_neon_(next1, 0xA0);
        sz_u64_t const b1_ge_a0 = sz_line_break_byte_ge_neon_(next1, 0xA0);
        sz_u64_t const b1_lt_90 = sz_line_break_byte_lt_neon_(next1, 0x90);
        sz_u64_t const b1_ge_90 = sz_line_break_byte_ge_neon_(next1, 0x90);

        sz_u64_t const overlong3 = three_byte & b0_e0 & b1_lt_a0;
        sz_u64_t const surrogate3 = three_byte & b0_ed & b1_ge_a0;
        sz_u64_t const overlong4 = four_byte & b0_f0 & b1_lt_90;
        sz_u64_t const above4 = four_byte & ((b0_f4 & b1_ge_90) | b0_above_f4);
        valid2 = two_byte & next1_continuation & lead_not_overlong2;
        valid3 = three_byte & next1_continuation & next2_continuation & ~overlong3 & ~surrogate3;
        valid4 = four_byte & next1_continuation & next2_continuation & next3_continuation & ~overlong4 & ~above4;
    }
    sz_u64_t const true_ascii = sz_line_break_byte_lt_neon_(raw, 0x80) & loaded_mask & ~continuation;

    //  Rebuild high/low per lead length so cp = (plane<<16)|(high<<8)|low is exact (decode_window_ uses the
    //  2/3-byte formula only). Replacement lanes are overridden downstream, so their garbage never reaches the
    //  palette. The 4-byte high/low mirror the icelake/haswell 4-byte reconstruction, per quarter.
    sz_u64_t const is_astral = four_byte & loaded_mask;

    sz_u64_t const valid_start = true_ascii | valid2 | valid3 | valid4;
    sz_u64_t const consumed = (((valid2 | valid3 | valid4) << 1) | ((valid3 | valid4) << 2) | (valid4 << 3)) &
                              continuation & loaded_mask;
    sz_u64_t const starts = loaded_mask & ~consumed;
    sz_u64_t const replacement = starts & ~valid_start;

    sz_line_break_classified_neon_t result;
    uint8x16_t dotted_q[4];
    uint8x16_t const fffd_index = sz_line_break_bmp_index_neon_(vdupq_n_u8(0xFF), vdupq_n_u8(0xFD));

    uint8x16_t const low_two_bits = vdupq_n_u8(0x03);
    uint8x16_t const low_three_bits = vdupq_n_u8(0x07);
    uint8x16_t const low_four_bits = vdupq_n_u8(0x0F);
    uint8x16_t const low_six_bits = vdupq_n_u8(0x3F);

    for (int quarter = 0; quarter < 4; ++quarter) {
        uint8x16_t const raw_q = raw[quarter];
        uint8x16_t const n1 = next1[quarter];
        uint8x16_t const n2 = next2[quarter];
        uint8x16_t const n3 = next3[quarter];
        int const lane_base = quarter * 16;

        //  Select masks for this quarter (low 16 bits of the per-quarter slice of each global mask).
        uint8x16_t const ascii_select = sz_line_break_byte_mask_from_bits_neon_(true_ascii >> lane_base);
        uint8x16_t const four_select = sz_line_break_byte_mask_from_bits_neon_(four_byte >> lane_base);

        //  4-byte low/high reconstruction.
        uint8x16_t const low_four = vorrq_u8(vandq_u8(vshlq_n_u8(vandq_u8(n2, low_two_bits), 6), vdupq_n_u8(0xC0)),
                                             vandq_u8(n3, low_six_bits));
        uint8x16_t const high_four = vorrq_u8(vandq_u8(vshlq_n_u8(vandq_u8(n1, low_four_bits), 4), vdupq_n_u8(0xF0)),
                                              sz_utf8_srl8_neon_(n2, 2, 0x0F));

        //  Blend ASCII (cp == raw byte, high == 0) then 4-byte reconstruction over the decode-window halves.
        uint8x16_t low_q = vbslq_u8(ascii_select, raw_q, window.low[quarter]);
        low_q = vbslq_u8(four_select, low_four, low_q);
        uint8x16_t high_q = vbicq_u8(window.high[quarter], ascii_select); // zero high on ASCII lanes
        high_q = vbslq_u8(four_select, high_four, high_q);

        //  4-byte plane bits (bits 16..20 of the codepoint); zero on every non-4-byte lane.
        uint8x16_t const plane = vorrq_u8(vandq_u8(vshlq_n_u8(vandq_u8(raw_q, low_three_bits), 2), vdupq_n_u8(0x1C)),
                                          sz_utf8_srl8_neon_(n1, 4, 0x03));
        uint8x16_t const plane_masked = vandq_u8(four_select, plane);

        //  Palette index per byte-lane. BMP through the validated full-BMP cascade; astral blended in; replacement
        //  lanes forced to U+FFFD's index.
        uint8x16_t index_q = sz_line_break_bmp_index_neon_(high_q, low_q);
        if (is_astral) {
            //  The astral cascade is addressed by offset = codepoint - 0x10000; the offset plane nibble is
            //  `plane - 1`. The low 16 bits are unchanged by subtracting 0x10000, so `high`/`low` feed directly.
            uint8x16_t const plane_off = vsubq_u8(plane_masked, vdupq_n_u8(1));
            uint8x16_t const astral = sz_line_break_classify_astral_neon_(plane_off, high_q, low_q);
            index_q = vbslq_u8(four_select, astral, index_q);
        }
        if (replacement) {
            uint8x16_t const rep_select = sz_line_break_byte_mask_from_bits_neon_(replacement >> lane_base);
            index_q = vbslq_u8(rep_select, fffd_index, index_q);
        }

        sz_line_break_palette_unpack_neon_(index_q, &result.classes[quarter], &result.side[quarter],
                                           &dotted_q[quarter]);
    }

    sz_u64_t const dotted = sz_utf8_mask_combine_neon_(
        vtstq_u8(dotted_q[0], dotted_q[0]), vtstq_u8(dotted_q[1], dotted_q[1]), vtstq_u8(dotted_q[2], dotted_q[2]),
        vtstq_u8(dotted_q[3], dotted_q[3]));
    result.dotted = dotted & starts;
    result.starts = starts;
    result.replacement = replacement;
    result.non_start = loaded_mask & ~starts;
    result.loaded = window.loaded;
    return result;
}

#pragma endregion In register vectorized classifier

#pragma region Mask algebra rule engine

/** @brief Build a 64-bit "lane class == @p cls" mask over the four class quarters (four `vceqq_u8` -> mask_combine). */
SZ_HELPER_INLINE sz_u64_t sz_line_break_class_mask_neon_(uint8x16_t const *classes, sz_u8_t cls) {
    return sz_line_break_byte_equal_neon_(classes, cls);
}

/** @brief Build a 64-bit "lane (side & @p bit) != 0" mask over the four side quarters (`vtstq_u8` is native). */
SZ_HELPER_INLINE sz_u64_t sz_line_break_side_mask_neon_(uint8x16_t const *side, sz_u8_t bit) {
    uint8x16_t const m = vdupq_n_u8(bit);
    return sz_utf8_mask_combine_neon_(vtstq_u8(side[0], m), vtstq_u8(side[1], m), vtstq_u8(side[2], m),
                                      vtstq_u8(side[3], m));
}

/** @brief Byte-lane gate/base derivation (LB9/LB10) — the NEON twin of @ref sz_line_break_byte_frame_haswell_t. */
typedef struct sz_line_break_byte_frame_neon_t {
    uint8x16_t classes[4]; /**< Class per lane with lone marks reclassified to AL (LB10), per quarter. */
    sz_u64_t base;         /**< Cluster-base lanes (every effective start except an attached CM/ZWJ). */
    sz_u64_t gate;         /**< Transparent lanes for neighbour fills: continuations + attached-mark starts. */
    sz_u64_t attached;     /**< Attached CM/ZWJ start lanes (LB9). */
    sz_u64_t lone_mark;    /**< LB10 lone marks reclassified to AL; their side bits must be cleared. */
} sz_line_break_byte_frame_neon_t;

SZ_HELPER_INLINE sz_line_break_byte_frame_neon_t sz_line_break_byte_frame_neon_(
    sz_line_break_classified_neon_t classified) {
    sz_u64_t const starts = classified.starts, non_start = classified.non_start;
    uint8x16_t const *classes = classified.classes;
    sz_u64_t const mark_start = (sz_line_break_class_mask_neon_(classes, sz_line_break_cm_k) |
                                 sz_line_break_class_mask_neon_(classes, sz_line_break_zwj_k)) &
                                starts;
    sz_u64_t const excluded = (sz_line_break_class_mask_neon_(classes, sz_line_break_bk_k) |
                               sz_line_break_class_mask_neon_(classes, sz_line_break_cr_k) |
                               sz_line_break_class_mask_neon_(classes, sz_line_break_lf_k) |
                               sz_line_break_class_mask_neon_(classes, sz_line_break_nl_k) |
                               sz_line_break_class_mask_neon_(classes, sz_line_break_sp_k) |
                               sz_line_break_class_mask_neon_(classes, sz_line_break_zw_k)) &
                              starts;
    sz_u64_t const good_base = starts & ~excluded & ~mark_start;
    sz_u64_t const mark_bytes = sz_u64_fill_right_(mark_start, non_start) | mark_start;
    sz_u64_t const flood = sz_u64_fill_right_(good_base, non_start | mark_bytes);
    sz_u64_t const attached = flood & mark_start;
    sz_u64_t const lone_mark = mark_start & ~attached;

    sz_line_break_byte_frame_neon_t frame;
    //  Reclassify lone marks to AL in every quarter (LB10).
    uint8x16_t const al = vdupq_n_u8(sz_line_break_al_k);
    for (int quarter = 0; quarter < 4; ++quarter) {
        uint8x16_t const lone_select = sz_line_break_byte_mask_from_bits_neon_(lone_mark >> (quarter * 16));
        frame.classes[quarter] = vbslq_u8(lone_select, al, classes[quarter]);
    }
    frame.base = starts & ~attached;
    frame.gate = non_start | attached;
    frame.attached = attached;
    frame.lone_mark = lone_mark;
    return frame;
}

/**
 *  @brief  Per-ISA extractor: lower one classified 64-byte window to the portable @ref sz_line_break_frame_t — the
 *          NEON twin of @ref sz_line_break_build_frame_haswell_. Builds the byte-level cluster frame (LB9/LB10),
 *          materializes per-class membership after the LB10 reclassify, the raw ZWJ + five side-bit masks, and the
 *          per-lane class/side bytes (four `vst1q_u8` quarters each).
 */
SZ_HELPER_INLINE sz_line_break_frame_t sz_line_break_build_frame_neon_(sz_line_break_classified_neon_t classified,
                                                                       sz_u8_t *effective_class_byte_out,
                                                                       sz_u8_t *side_byte_out) {
    sz_line_break_byte_frame_neon_t const byte_frame = sz_line_break_byte_frame_neon_(classified);
    uint8x16_t const *classes = byte_frame.classes;
    //  LB10 reclassify carries the side bits with it: zero the side byte on lone-mark lanes (serial zeros the
    //  descriptor). `vbicq_u8(side, lone_select)` clears those lanes.
    uint8x16_t side[4];
    for (int quarter = 0; quarter < 4; ++quarter) {
        uint8x16_t const lone_select = sz_line_break_byte_mask_from_bits_neon_(byte_frame.lone_mark >> (quarter * 16));
        side[quarter] = vbicq_u8(classified.side[quarter], lone_select);
    }

    sz_line_break_frame_t frame;
    frame.base = byte_frame.base;
    frame.gate = byte_frame.gate;
    frame.attached = byte_frame.attached;
    frame.lone_mark = byte_frame.lone_mark;
    frame.non_start = classified.non_start;
    frame.dotted = classified.dotted;
    frame.starts = classified.starts;
    frame.replacement = classified.replacement;
    //  Full unroll is load-bearing (mirrors icelake/haswell): SROA promotes the per-class array to registers and
    //  drops the `vceqq_u8` quads for classes the engine never reads; a runtime index keeps it stack-resident.
#if defined(__clang__)
#pragma clang loop unroll(full)
#elif defined(__GNUC__)
#pragma GCC unroll 64
#endif
    for (sz_size_t cls = 0; cls < sz_line_break_class_count_k; ++cls)
        frame.effective_class[cls] = sz_line_break_class_mask_neon_(classes, (sz_u8_t)cls);
    frame.raw_zwj = sz_line_break_class_mask_neon_(classified.classes, sz_line_break_zwj_k);
    frame.side_pi = sz_line_break_side_mask_neon_(side, sz_line_break_side_pi_k);
    frame.side_pf = sz_line_break_side_mask_neon_(side, sz_line_break_side_pf_k);
    frame.side_eaw = sz_line_break_side_mask_neon_(side, sz_line_break_side_eaw_k);
    frame.side_cn = sz_line_break_side_mask_neon_(side, sz_line_break_side_cn_k);
    frame.side_ext = sz_line_break_side_mask_neon_(side, sz_line_break_side_ext_k);
    for (int quarter = 0; quarter < 4; ++quarter) {
        vst1q_u8(effective_class_byte_out + quarter * 16, classes[quarter]);
        vst1q_u8(side_byte_out + quarter * 16, side[quarter]);
    }
    return frame;
}

/**
 *  @brief  Byte-level UAX-14 rule engine, NEON entry: extract the portable frame in-register, then delegate every
 *          LB1-LB31 decision to the portable @ref sz_line_break_decide_window_.
 */
SZ_HELPER_INLINE sz_line_break_window_t sz_line_break_decide_window_neon_(sz_line_break_classified_neon_t classified,
                                                                          sz_line_break_carry_t carry,
                                                                          sz_line_break_carry_t *carry_out,
                                                                          sz_size_t complete_limit,
                                                                          sz_bool_t more_text) {
    sz_u8_t effective_class_byte[64], side_byte[64];
    sz_line_break_frame_t const frame = sz_line_break_build_frame_neon_(classified, effective_class_byte, side_byte);
    return sz_line_break_decide_window_(&frame, effective_class_byte, side_byte, carry, carry_out, complete_limit,
                                        more_text);
}

#pragma endregion Mask algebra rule engine

#pragma region Forward driver

/**
 *  @brief  Largest byte prefix of the window whose codepoints are all fully loaded — the NEON twin of
 *          @ref sz_line_break_complete_limit_haswell_ over the NEON window struct. Never below 1.
 */
SZ_HELPER_AUTO sz_size_t sz_line_break_complete_limit_neon_(sz_utf8_rune_window_neon_t window, sz_bool_t more_text) {
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
 *  @brief  Byte-level zero-scalar forward UAX-14 kernel (NEON AArch64): the overlap-free advancing driver, mirroring
 *          @ref sz_utf8_linebreaks_haswell_bytes_ over the NEON window/classify/drain leaves.
 */
SZ_API_COMPTIME sz_size_t sz_utf8_linebreaks_neon_bytes_( //
    sz_cptr_t text, sz_size_t length,                     //
    sz_size_t *starts, sz_size_t *lengths,                //
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
        sz_utf8_rune_window_neon_t const window = sz_utf8_rune_decode_window_neon_(bytes + position, length - position);
        sz_bool_t const more_text = (sz_bool_t)(position + window.loaded < length);
        sz_size_t const complete_limit = sz_line_break_complete_limit_neon_(window, more_text);
        sz_line_break_classified_neon_t const classified = sz_line_break_classify_window_neon_(window);

        sz_line_break_carry_t carry_next = carry;
        sz_line_break_window_t const win = sz_line_break_decide_window_neon_(classified, carry, &carry_next,
                                                                             complete_limit, more_text);
        sz_u64_t const commit = win.breaks & sz_u64_mask_until_serial_(win.resolved);

        produced = sz_utf8_rune_drain_forward_neon_(commit, position, starts, lengths, produced, capacity, &line_start);
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
 *  @brief  Forward UAX-14 line-break-opportunity kernel (NEON AArch64). Bit-exact with `sz_utf8_linebreaks_serial`,
 *          `sz_utf8_linebreaks_haswell`, and `sz_utf8_linebreaks_icelake`.
 */
SZ_API_COMPTIME sz_size_t sz_utf8_linebreaks_neon( //
    sz_cptr_t text, sz_size_t length,              //
    sz_size_t *starts, sz_size_t *lengths,         //
    sz_size_t capacity, sz_size_t *bytes_consumed) {
    return sz_utf8_linebreaks_neon_bytes_(text, length, starts, lengths, capacity, bytes_consumed);
}

#pragma endregion Forward driver

#pragma endregion UAX 14 Line Boundaries forward kernel
#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#endif // SZ_USE_NEON

#ifdef __cplusplus
}
#endif

#endif // STRINGZILLA_UTF8_LINEBREAKS_NEON_H_
