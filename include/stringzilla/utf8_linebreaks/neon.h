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
 *  (class, side, dotted) with ZERO per-lane scalar loop and NO serial deferral.
 *
 *  Each 64-byte window lives as four `uint8x16_t` quarters (`window[0]` = lanes [0,16), ... `window[3]` = lanes
 *  [48,64)) instead of haswell's two halves; every per-lane class compare is `vceqq_u8` per quarter, the four boolean
 *  quarters OR-collapsed to a `sz_u64_t` via `mask_combine_neon_`. The BMP flat-palette index comes from the
 *  page-compressed flat table via @ref sz_utf8_rune_flat_lookup_neon_ and expands to the LB1-resolved class / side /
 *  dotted bytes by `sz_line_break_flat_palette_unpack_neon_`. The astral path still walks the
 *  `sz_line_break_classify_astral_neon_` cascade and its 62-entry palette, a DIFFERENT index space, so its RESOLVED
 *  bytes blend over the valid 4-byte lanes after the expansion, bit-identical to the icelake/haswell blend. */

/** @brief Expand a 16-bit lane mask into a `uint8x16_t` select vector (byte `i` = 0xFF when bit `i` is set), the
 *         NEON twin of @ref sz_utf8_byte_mask_from_bits_haswell_ (which expands 32 bits to a `__m256i`).
 *         Inverse of `movemask16_neon_`: route the right mask byte to each lane, isolate its bit, then `vceqq`. */
SZ_HELPER_INLINE uint8x16_t sz_line_break_byte_mask_from_bits_neon_(sz_u64_t bits) {
    static sz_u8_t const byte_router_lanes[16] = {0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1};
    static sz_u8_t const bit_select_lanes[16] = {1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128};
    uint8x16_t const broadcast_u8x16 = vreinterpretq_u8_u16(vdupq_n_u16((sz_u16_t)bits));
    uint8x16_t const byte_router_u8x16 = vld1q_u8(byte_router_lanes);
    uint8x16_t const spread_u8x16 = vqtbl1q_u8(broadcast_u8x16, byte_router_u8x16);
    uint8x16_t const bit_select_u8x16 = vld1q_u8(bit_select_lanes);
    uint8x16_t const isolated_u8x16 = vandq_u8(spread_u8x16, bit_select_u8x16);
    return vceqq_u8(isolated_u8x16, bit_select_u8x16);
}

/** @brief Flat-palette index for sixteen BMP codepoints (per-lane high = cp>>8, low = cp&0xFF) from the
 *         page-compressed flat leaf via @ref sz_utf8_rune_flat_lookup_neon_, the NEON twin of
 *         @ref sz_line_break_bmp_index_haswell_. Bit-exact with `sz_rune_line_break_property` over the whole BMP
 *         once `flat_palette_` expands the index. Operates on one quarter; the caller iterates the four. */
SZ_HELPER_AUTO uint8x16_t sz_line_break_bmp_index_neon_(uint8x16_t high_bytes_u8x16, uint8x16_t low_bytes_u8x16) {
    return sz_utf8_rune_flat_lookup_neon_(sz_utf8_line_break_bmp_page_lut_, sz_utf8_line_break_flat_bmp_,
                                          (int)sz_utf8_line_break_flat_pages_k, high_bytes_u8x16, low_bytes_u8x16);
}

/** @brief Palette index for sixteen ASTRAL codepoints over the 20-bit offset = cp - 0x10000 (5-nibble cascade),
 *         the NEON twin of @ref sz_line_break_classify_astral_haswell_. Per-lane bytes: @p plane_u8x16 =
 *         (offset>>16)&0xFF (low nibble meaningful), @p high_u8x16 = (offset>>8)&0xFF, @p low_u8x16 = offset&0xFF.
 *         Bit-exact. */
SZ_HELPER_AUTO uint8x16_t sz_line_break_classify_astral_neon_(uint8x16_t plane_u8x16, uint8x16_t high_u8x16,
                                                              uint8x16_t low_u8x16) {
    uint8x16_t const low_nibble_mask_u8x16 = vdupq_n_u8(0x0F);
    uint8x16_t const n4_u8x16 = vandq_u8(plane_u8x16, low_nibble_mask_u8x16);
    uint8x16_t const n3_u8x16 = vandq_u8(vshrq_n_u8(high_u8x16, 4), low_nibble_mask_u8x16);
    uint8x16_t const stage1_index_u8x16 = vorrq_u8(vshlq_n_u8(n4_u8x16, 4), n3_u8x16);
    uint8x16_t const page_u8x16 = sz_utf8_rune_lut256_neon_(sz_utf8_line_break_haswell_astral_stage1_,
                                                            stage1_index_u8x16);
    uint8x16_t const n2_u8x16 = vandq_u8(high_u8x16, low_nibble_mask_u8x16);
    uint8x16_t const leaf2_lo_u8x16 = sz_utf8_rune_cascade_stage_neon_(
        sz_utf8_line_break_haswell_astral_stage2_lo_, sz_utf8_line_break_haswell_astral_stage2_lo_count_k / 16,
        page_u8x16, n2_u8x16);
    uint8x16_t const n1_u8x16 = vandq_u8(vshrq_n_u8(low_u8x16, 4), low_nibble_mask_u8x16);
    uint8x16_t const leaf_lo_u8x16 = sz_utf8_rune_cascade_stage_neon_(
        sz_utf8_line_break_haswell_astral_stage3_lo_, sz_utf8_line_break_haswell_astral_stage3_lo_count_k / 16,
        leaf2_lo_u8x16, n1_u8x16);
    uint8x16_t const leaf_hi_u8x16 = sz_utf8_rune_cascade_stage_neon_(
        sz_utf8_line_break_haswell_astral_stage3_hi_, sz_utf8_line_break_haswell_astral_stage3_hi_count_k / 16,
        leaf2_lo_u8x16, n1_u8x16);
    uint8x16_t const n0_u8x16 = vandq_u8(low_u8x16, low_nibble_mask_u8x16);
    uint8x16_t const leaf_group_u8x16 = vorrq_u8(vandq_u8(vshrq_n_u8(leaf_lo_u8x16, 4), low_nibble_mask_u8x16),
                                                 vshlq_n_u8(leaf_hi_u8x16, 4));
    uint8x16_t const leaf_low_nibble_u8x16 = vandq_u8(leaf_lo_u8x16, low_nibble_mask_u8x16);
    uint8x16_t const stage4_lut_index_u8x16 = vorrq_u8(vshlq_n_u8(leaf_low_nibble_u8x16, 4), n0_u8x16);
    uint8x16_t result_u8x16 = vdupq_n_u8(0);
    for (int group = 0; group < (int)sz_utf8_line_break_haswell_astral_leaf_groups_k; ++group) {
        uint8x16_t const value_u8x16 = sz_utf8_rune_lut256_neon_(
            sz_utf8_line_break_haswell_astral_stage4_groups_ + group * 256, stage4_lut_index_u8x16);
        uint8x16_t const here_u8x16 = vceqq_u8(leaf_group_u8x16, vdupq_n_u8((sz_u8_t)group));
        result_u8x16 = vbslq_u8(here_u8x16, value_u8x16, result_u8x16);
    }
    return result_u8x16;
}

/** @brief Split sixteen flat-palette indices into the low and high byte of their 16-bit Line_Break descriptors.
 *         Four `vld2q_u8` deinterleave the 64-word padded palette into two resident 64-byte tables (descriptor low
 *         bytes and high bytes) and one `vqtbl4q_u8` each resolves all sixteen lanes -- the NEON stand-in for the
 *         AVX2 scale-2 `vpgatherdd` over the same palette. Every index originates in the flat leaf, so it is always
 *         < 56 and lands inside the four resident quads, never in the `vqtbl4q` zero-fill. */
SZ_HELPER_INLINE void sz_line_break_flat_palette_descriptors_neon_(uint8x16_t palette_indices_u8x16,
                                                                   uint8x16_t *descriptor_low_bytes_u8x16,
                                                                   uint8x16_t *descriptor_high_bytes_u8x16) {
    sz_u8_t const *palette_bytes = (sz_u8_t const *)sz_utf8_line_break_flat_palette_;
    uint8x16x2_t const words_first_u8x16x2 = vld2q_u8(palette_bytes + 0);
    uint8x16x2_t const words_second_u8x16x2 = vld2q_u8(palette_bytes + 32);
    uint8x16x2_t const words_third_u8x16x2 = vld2q_u8(palette_bytes + 64);
    uint8x16x2_t const words_fourth_u8x16x2 = vld2q_u8(palette_bytes + 96);
    uint8x16x4_t low_table_u8x16x4, high_table_u8x16x4;
    low_table_u8x16x4.val[0] = words_first_u8x16x2.val[0], low_table_u8x16x4.val[1] = words_second_u8x16x2.val[0];
    low_table_u8x16x4.val[2] = words_third_u8x16x2.val[0], low_table_u8x16x4.val[3] = words_fourth_u8x16x2.val[0];
    high_table_u8x16x4.val[0] = words_first_u8x16x2.val[1], high_table_u8x16x4.val[1] = words_second_u8x16x2.val[1];
    high_table_u8x16x4.val[2] = words_third_u8x16x2.val[1], high_table_u8x16x4.val[3] = words_fourth_u8x16x2.val[1];
    *descriptor_low_bytes_u8x16 = vqtbl4q_u8(low_table_u8x16x4, palette_indices_u8x16);
    *descriptor_high_bytes_u8x16 = vqtbl4q_u8(high_table_u8x16x4, palette_indices_u8x16);
}

/** @brief Expand sixteen flat-palette indices to the LB1-resolved class byte, the engine side byte and the
 *         DottedCircle select (0xFF where set), the NEON twin of @ref sz_line_break_flat_palette_unpack_haswell_.
 *         Every descriptor field the engine reads lives below bit 14 -- class in bits 0-5, Pi/Pf in 6/7, EAW/Cn|Ext
 *         in 8/9, SA-is-mark in 12, DottedCircle in 13 -- so the whole unpack stays in the byte domain. Applies the
 *         serial resolution aliasing (SA → AL/CM, AI/SG/XX → AL, CJ → NS); RI/ZWJ side bits come from the RAW
 *         class, the mark side bit from the resolved class. */
SZ_HELPER_AUTO void sz_line_break_flat_palette_unpack_neon_(uint8x16_t palette_indices_u8x16,
                                                            uint8x16_t *classes_u8x16_out, uint8x16_t *side_u8x16_out,
                                                            uint8x16_t *dotted_select_u8x16_out) {
    uint8x16_t descriptor_low_bytes_u8x16, descriptor_high_bytes_u8x16;
    sz_line_break_flat_palette_descriptors_neon_(palette_indices_u8x16, &descriptor_low_bytes_u8x16,
                                                 &descriptor_high_bytes_u8x16);

    uint8x16_t const raw_classes_u8x16 = vandq_u8(descriptor_low_bytes_u8x16, vdupq_n_u8(0x3F));
    uint8x16_t const is_sa_u8x16 = vceqq_u8(raw_classes_u8x16, vdupq_n_u8((sz_u8_t)sz_line_break_sa_k));
    uint8x16_t const sa_is_mark_u8x16 = vtstq_u8(descriptor_high_bytes_u8x16, vdupq_n_u8(1 << 4));
    uint8x16_t classes_u8x16 = vbslq_u8(is_sa_u8x16, vdupq_n_u8((sz_u8_t)sz_line_break_al_k), raw_classes_u8x16);
    classes_u8x16 = vbslq_u8(vandq_u8(is_sa_u8x16, sa_is_mark_u8x16), vdupq_n_u8((sz_u8_t)sz_line_break_cm_k),
                             classes_u8x16);
    uint8x16_t const is_alias_u8x16 = vorrq_u8(
        vorrq_u8(vceqq_u8(classes_u8x16, vdupq_n_u8((sz_u8_t)sz_line_break_ai_k)),
                 vceqq_u8(classes_u8x16, vdupq_n_u8((sz_u8_t)sz_line_break_sg_k))),
        vceqq_u8(classes_u8x16, vdupq_n_u8((sz_u8_t)sz_line_break_xx_k)));
    classes_u8x16 = vbslq_u8(is_alias_u8x16, vdupq_n_u8((sz_u8_t)sz_line_break_al_k), classes_u8x16);
    uint8x16_t const is_cj_u8x16 = vceqq_u8(classes_u8x16, vdupq_n_u8((sz_u8_t)sz_line_break_cj_k));
    classes_u8x16 = vbslq_u8(is_cj_u8x16, vdupq_n_u8((sz_u8_t)sz_line_break_ns_k), classes_u8x16);

    uint8x16_t side_u8x16 = vandq_u8(vtstq_u8(descriptor_low_bytes_u8x16, vdupq_n_u8(1 << 6)),
                                     vdupq_n_u8((sz_u8_t)sz_line_break_side_pi_k));
    side_u8x16 = vorrq_u8(side_u8x16, vandq_u8(vtstq_u8(descriptor_low_bytes_u8x16, vdupq_n_u8(1 << 7)),
                                               vdupq_n_u8((sz_u8_t)sz_line_break_side_pf_k)));
    side_u8x16 = vorrq_u8(side_u8x16, vandq_u8(vtstq_u8(descriptor_high_bytes_u8x16, vdupq_n_u8(1 << 0)),
                                               vdupq_n_u8((sz_u8_t)sz_line_break_side_eaw_k)));
    side_u8x16 = vorrq_u8(side_u8x16,
                          vandq_u8(vtstq_u8(descriptor_high_bytes_u8x16, vdupq_n_u8(1 << 1)),
                                   vdupq_n_u8((sz_u8_t)(sz_line_break_side_cn_k | sz_line_break_side_ext_k))));
    side_u8x16 = vorrq_u8(side_u8x16, vandq_u8(vceqq_u8(raw_classes_u8x16, vdupq_n_u8((sz_u8_t)sz_line_break_ri_k)),
                                               vdupq_n_u8((sz_u8_t)sz_line_break_side_ri_k)));
    side_u8x16 = vorrq_u8(side_u8x16, vandq_u8(vceqq_u8(raw_classes_u8x16, vdupq_n_u8((sz_u8_t)sz_line_break_zwj_k)),
                                               vdupq_n_u8((sz_u8_t)sz_line_break_side_zwj_k)));
    uint8x16_t const class_is_mark_u8x16 = vorrq_u8(vceqq_u8(classes_u8x16, vdupq_n_u8((sz_u8_t)sz_line_break_cm_k)),
                                                    vceqq_u8(classes_u8x16, vdupq_n_u8((sz_u8_t)sz_line_break_zwj_k)));
    side_u8x16 = vorrq_u8(side_u8x16, vandq_u8(class_is_mark_u8x16, vdupq_n_u8((sz_u8_t)sz_line_break_side_mark_k)));

    *classes_u8x16_out = classes_u8x16;
    *side_u8x16_out = side_u8x16;
    *dotted_select_u8x16_out = vtstq_u8(descriptor_high_bytes_u8x16, vdupq_n_u8(1 << 5));
}

/** @brief A 64-bit "(byte & mask) == pattern" lane mask over the four window quarters. */
SZ_HELPER_INLINE sz_u64_t sz_line_break_byte_match_neon_(uint8x16_t const *quarters_u8x16, sz_u8_t mask,
                                                         sz_u8_t pattern) {
    uint8x16_t const m_u8x16 = vdupq_n_u8(mask), p_u8x16 = vdupq_n_u8(pattern);
    return sz_utf8_mask_combine_neon_(vceqq_u8(vandq_u8(quarters_u8x16[0], m_u8x16), p_u8x16),
                                      vceqq_u8(vandq_u8(quarters_u8x16[1], m_u8x16), p_u8x16),
                                      vceqq_u8(vandq_u8(quarters_u8x16[2], m_u8x16), p_u8x16),
                                      vceqq_u8(vandq_u8(quarters_u8x16[3], m_u8x16), p_u8x16));
}

/** @brief A 64-bit "byte == value" lane mask over the four window quarters. */
SZ_HELPER_INLINE sz_u64_t sz_line_break_byte_equal_neon_(uint8x16_t const *quarters_u8x16, sz_u8_t value) {
    uint8x16_t const v_u8x16 = vdupq_n_u8(value);
    return sz_utf8_mask_combine_neon_(vceqq_u8(quarters_u8x16[0], v_u8x16), vceqq_u8(quarters_u8x16[1], v_u8x16),
                                      vceqq_u8(quarters_u8x16[2], v_u8x16), vceqq_u8(quarters_u8x16[3], v_u8x16));
}

/** @brief A 64-bit "byte >= bound" (unsigned) lane mask over the four window quarters (`vcgeq_u8` is native). */
SZ_HELPER_INLINE sz_u64_t sz_line_break_byte_ge_neon_(uint8x16_t const *quarters_u8x16, sz_u8_t bound) {
    uint8x16_t const bound_u8x16 = vdupq_n_u8(bound);
    return sz_utf8_mask_combine_neon_(
        vcgeq_u8(quarters_u8x16[0], bound_u8x16), vcgeq_u8(quarters_u8x16[1], bound_u8x16),
        vcgeq_u8(quarters_u8x16[2], bound_u8x16), vcgeq_u8(quarters_u8x16[3], bound_u8x16));
}

/** @brief A 64-bit "byte < bound" (unsigned) lane mask over the four window quarters (`vcltq_u8` is native). */
SZ_HELPER_INLINE sz_u64_t sz_line_break_byte_lt_neon_(uint8x16_t const *quarters_u8x16, sz_u8_t bound) {
    uint8x16_t const bound_u8x16 = vdupq_n_u8(bound);
    return sz_utf8_mask_combine_neon_(
        vcltq_u8(quarters_u8x16[0], bound_u8x16), vcltq_u8(quarters_u8x16[1], bound_u8x16),
        vcltq_u8(quarters_u8x16[2], bound_u8x16), vcltq_u8(quarters_u8x16[3], bound_u8x16));
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

/** @brief Resolve the per-lane 62-entry-palette index (one quarter) to class / side / dotted bytes through the
 *         precomputed palette tables; `lut256_neon_` reads each 256-byte-padded table by `index`. Serves only the
 *         astral cascade now -- the BMP path expands flat-palette descriptors via
 *         @ref sz_line_break_flat_palette_unpack_neon_ instead, a different index space. */
SZ_HELPER_INLINE void sz_line_break_palette_unpack_neon_(uint8x16_t index_u8x16, uint8x16_t *classes_u8x16,
                                                         uint8x16_t *side_u8x16, uint8x16_t *dotted_u8x16) {
    *classes_u8x16 = sz_utf8_rune_lut256_neon_(sz_utf8_line_break_palette_class_, index_u8x16);
    *side_u8x16 = sz_utf8_rune_lut256_neon_(sz_utf8_line_break_palette_side_, index_u8x16);
    *dotted_u8x16 = sz_utf8_rune_lut256_neon_(sz_utf8_line_break_palette_dotted_, index_u8x16);
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
    uint8x16_t const *raw_u8x16 = window.window;

    //  Forward neighbours (mod-64 wrap, matching icelake's `_mm512_permutexvar_epi8`).
    uint8x16_t next1_u8x16[4], next2_u8x16[4], next3_u8x16[4];
    sz_utf8_forward_neighbours_neon_(raw_u8x16, next1_u8x16, next2_u8x16, next3_u8x16);

    sz_u64_t const next1_continuation = continuation >> 1, next2_continuation = continuation >> 2,
                   next3_continuation = continuation >> 3;

    //  Ill-formed-lead gate (LEVER B), bit-identical to icelake/haswell: only C0/C1, E0, ED, F0, F4/>=F5 leads can
    //  be overlong / surrogate / out-of-range; detect their presence with raw-only tests and, when absent, take the
    //  cheap "lead + enough continuations" validity path.
    sz_u64_t const lead_c0_c1 = sz_line_break_byte_ge_neon_(raw_u8x16, 0xC0) &
                                sz_line_break_byte_lt_neon_(raw_u8x16, 0xC2);
    sz_u64_t const lead_e0 = sz_line_break_byte_equal_neon_(raw_u8x16, 0xE0);
    sz_u64_t const lead_ed = sz_line_break_byte_equal_neon_(raw_u8x16, 0xED);
    sz_u64_t const lead_f0 = sz_line_break_byte_equal_neon_(raw_u8x16, 0xF0);
    sz_u64_t const lead_f4_or_above = sz_line_break_byte_ge_neon_(raw_u8x16, 0xF4);
    sz_u64_t const danger_leads = (lead_c0_c1 | lead_e0 | lead_ed | lead_f0 | lead_f4_or_above) & loaded_mask;

    sz_u64_t valid2, valid3, valid4;
    if (!danger_leads) {
        valid2 = two_byte & next1_continuation;
        valid3 = three_byte & next1_continuation & next2_continuation;
        valid4 = four_byte & next1_continuation & next2_continuation & next3_continuation;
    }
    else {
        //  Exact prior algebra: per-lane value-bit predicates for overlong / surrogate / out-of-range detection.
        sz_u64_t const lead_not_overlong2 = ~sz_line_break_byte_match_neon_(raw_u8x16, 0x1E, 0x00); // (raw & 0x1E) != 0
        sz_u64_t const b0_e0 = lead_e0, b0_ed = lead_ed, b0_f0 = lead_f0;
        sz_u64_t const b0_f4 = sz_line_break_byte_equal_neon_(raw_u8x16, 0xF4);
        sz_u64_t const b0_above_f4 = sz_line_break_byte_ge_neon_(raw_u8x16, 0xF5);
        sz_u64_t const b1_lt_a0 = sz_line_break_byte_lt_neon_(next1_u8x16, 0xA0);
        sz_u64_t const b1_ge_a0 = sz_line_break_byte_ge_neon_(next1_u8x16, 0xA0);
        sz_u64_t const b1_lt_90 = sz_line_break_byte_lt_neon_(next1_u8x16, 0x90);
        sz_u64_t const b1_ge_90 = sz_line_break_byte_ge_neon_(next1_u8x16, 0x90);

        sz_u64_t const overlong3 = three_byte & b0_e0 & b1_lt_a0;
        sz_u64_t const surrogate3 = three_byte & b0_ed & b1_ge_a0;
        sz_u64_t const overlong4 = four_byte & b0_f0 & b1_lt_90;
        sz_u64_t const above4 = four_byte & ((b0_f4 & b1_ge_90) | b0_above_f4);
        valid2 = two_byte & next1_continuation & lead_not_overlong2;
        valid3 = three_byte & next1_continuation & next2_continuation & ~overlong3 & ~surrogate3;
        valid4 = four_byte & next1_continuation & next2_continuation & next3_continuation & ~overlong4 & ~above4;
    }
    sz_u64_t const true_ascii = sz_line_break_byte_lt_neon_(raw_u8x16, 0x80) & loaded_mask & ~continuation;

    //  Rebuild high/low per lead length so cp = (plane<<16)|(high<<8)|low is exact (decode_window_ uses the
    //  2/3-byte formula only). Replacement lanes are overridden downstream, so their garbage never reaches the
    //  palette. The 4-byte high/low mirror the icelake/haswell 4-byte reconstruction, per quarter. Only VALID 4-byte
    //  starts take the astral blend: an invalid 4-byte lead is a replacement lane whose U+FFFD resolution must
    //  survive it, so `valid4`, not `four_byte`, gates it.
    sz_u64_t const is_astral = valid4 & loaded_mask;

    sz_u64_t const valid_start = true_ascii | valid2 | valid3 | valid4;
    sz_u64_t const consumed = (((valid2 | valid3 | valid4) << 1) | ((valid3 | valid4) << 2) | (valid4 << 3)) &
                              continuation & loaded_mask;
    sz_u64_t const starts = loaded_mask & ~consumed;
    sz_u64_t const replacement = starts & ~valid_start;

    sz_line_break_classified_neon_t result;
    uint8x16_t dotted_q_u8x16[4];
    uint8x16_t fffd_index_u8x16 = vdupq_n_u8(0);
    if (replacement) fffd_index_u8x16 = sz_line_break_bmp_index_neon_(vdupq_n_u8(0xFF), vdupq_n_u8(0xFD));

    uint8x16_t const low_two_bits_u8x16 = vdupq_n_u8(0x03);
    uint8x16_t const low_three_bits_u8x16 = vdupq_n_u8(0x07);
    uint8x16_t const low_four_bits_u8x16 = vdupq_n_u8(0x0F);
    uint8x16_t const low_six_bits_u8x16 = vdupq_n_u8(0x3F);

    for (int quarter = 0; quarter < 4; ++quarter) {
        uint8x16_t const raw_q_u8x16 = raw_u8x16[quarter];
        uint8x16_t const n1_u8x16 = next1_u8x16[quarter];
        uint8x16_t const n2_u8x16 = next2_u8x16[quarter];
        uint8x16_t const n3_u8x16 = next3_u8x16[quarter];
        int const lane_base = quarter * 16;

        //  Select masks for this quarter (low 16 bits of the per-quarter slice of each global mask).
        uint8x16_t const ascii_select_u8x16 = sz_line_break_byte_mask_from_bits_neon_(true_ascii >> lane_base);
        uint8x16_t const four_select_u8x16 = sz_line_break_byte_mask_from_bits_neon_(four_byte >> lane_base);

        //  4-byte low/high reconstruction.
        uint8x16_t const low_four_u8x16 = vorrq_u8(
            vandq_u8(vshlq_n_u8(vandq_u8(n2_u8x16, low_two_bits_u8x16), 6), vdupq_n_u8(0xC0)),
            vandq_u8(n3_u8x16, low_six_bits_u8x16));
        uint8x16_t const high_four_u8x16 = vorrq_u8(
            vandq_u8(vshlq_n_u8(vandq_u8(n1_u8x16, low_four_bits_u8x16), 4), vdupq_n_u8(0xF0)),
            sz_utf8_srl8_neon_(n2_u8x16, 2, 0x0F));

        //  Blend ASCII (cp == raw byte, high == 0) then 4-byte reconstruction over the decode-window halves.
        uint8x16_t low_q_u8x16 = vbslq_u8(ascii_select_u8x16, raw_q_u8x16, window.low[quarter]);
        low_q_u8x16 = vbslq_u8(four_select_u8x16, low_four_u8x16, low_q_u8x16);
        uint8x16_t high_q_u8x16 = vbicq_u8(window.high[quarter], ascii_select_u8x16); // zero high on ASCII lanes
        high_q_u8x16 = vbslq_u8(four_select_u8x16, high_four_u8x16, high_q_u8x16);

        //  4-byte plane bits (bits 16..20 of the codepoint); zero on every non-4-byte lane.
        uint8x16_t const plane_u8x16 = vorrq_u8(
            vandq_u8(vshlq_n_u8(vandq_u8(raw_q_u8x16, low_three_bits_u8x16), 2), vdupq_n_u8(0x1C)),
            sz_utf8_srl8_neon_(n1_u8x16, 4, 0x03));
        uint8x16_t const plane_masked_u8x16 = vandq_u8(four_select_u8x16, plane_u8x16);

        //  Flat-palette index per byte-lane: BMP through the page-compressed flat leaf; replacement lanes forced to
        //  U+FFFD's index (U+FFFD is itself BMP, so it shares this index space). Astral lanes cannot join here: the
        //  astral cascade still speaks the 62-entry palette, a DIFFERENT index space, so their RESOLVED bytes blend
        //  after the expansion, over the valid 4-byte lanes only.
        uint8x16_t bmp_index_u8x16 = sz_line_break_bmp_index_neon_(high_q_u8x16, low_q_u8x16);
        if (replacement) {
            uint8x16_t const rep_select_u8x16 = sz_line_break_byte_mask_from_bits_neon_(replacement >> lane_base);
            bmp_index_u8x16 = vbslq_u8(rep_select_u8x16, fffd_index_u8x16, bmp_index_u8x16);
        }
        sz_line_break_flat_palette_unpack_neon_(bmp_index_u8x16, &result.classes[quarter], &result.side[quarter],
                                                &dotted_q_u8x16[quarter]);
        if (is_astral) {
            //  The astral cascade is addressed by offset = codepoint - 0x10000; the offset plane nibble is
            //  `plane - 1`. The low 16 bits are unchanged by subtracting 0x10000, so `high`/`low` feed directly.
            //  Its 62-entry palette's byte tables carry the very same LB1 resolution the flat descriptor unpack
            //  applies, so blending the RESOLVED bytes is bit-identical to blending indices in one shared space.
            uint8x16_t const astral_select_u8x16 = sz_line_break_byte_mask_from_bits_neon_(is_astral >> lane_base);
            uint8x16_t const plane_off_u8x16 = vsubq_u8(plane_masked_u8x16, vdupq_n_u8(1));
            uint8x16_t const astral_index_u8x16 = sz_line_break_classify_astral_neon_(plane_off_u8x16, high_q_u8x16,
                                                                                      low_q_u8x16);
            uint8x16_t astral_classes_u8x16, astral_side_u8x16, astral_dotted_bytes_u8x16;
            sz_line_break_palette_unpack_neon_(astral_index_u8x16, &astral_classes_u8x16, &astral_side_u8x16,
                                               &astral_dotted_bytes_u8x16);
            result.classes[quarter] = vbslq_u8(astral_select_u8x16, astral_classes_u8x16, result.classes[quarter]);
            result.side[quarter] = vbslq_u8(astral_select_u8x16, astral_side_u8x16, result.side[quarter]);
            dotted_q_u8x16[quarter] = vbslq_u8(astral_select_u8x16,
                                               vtstq_u8(astral_dotted_bytes_u8x16, astral_dotted_bytes_u8x16),
                                               dotted_q_u8x16[quarter]);
        }
    }

    sz_u64_t const dotted = sz_utf8_mask_combine_neon_(
        vtstq_u8(dotted_q_u8x16[0], dotted_q_u8x16[0]), vtstq_u8(dotted_q_u8x16[1], dotted_q_u8x16[1]),
        vtstq_u8(dotted_q_u8x16[2], dotted_q_u8x16[2]), vtstq_u8(dotted_q_u8x16[3], dotted_q_u8x16[3]));
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
SZ_HELPER_INLINE sz_u64_t sz_line_break_class_mask_neon_(uint8x16_t const *classes_u8x16, sz_u8_t cls) {
    return sz_line_break_byte_equal_neon_(classes_u8x16, cls);
}

/** @brief Build a 64-bit "lane (side & @p bit) != 0" mask over the four side quarters (`vtstq_u8` is native). */
SZ_HELPER_INLINE sz_u64_t sz_line_break_side_mask_neon_(uint8x16_t const *side_u8x16, sz_u8_t bit) {
    uint8x16_t const m_u8x16 = vdupq_n_u8(bit);
    return sz_utf8_mask_combine_neon_(vtstq_u8(side_u8x16[0], m_u8x16), vtstq_u8(side_u8x16[1], m_u8x16),
                                      vtstq_u8(side_u8x16[2], m_u8x16), vtstq_u8(side_u8x16[3], m_u8x16));
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
    uint8x16_t const *classes_u8x16 = classified.classes;
    sz_u64_t const mark_start = (sz_line_break_class_mask_neon_(classes_u8x16, sz_line_break_cm_k) |
                                 sz_line_break_class_mask_neon_(classes_u8x16, sz_line_break_zwj_k)) &
                                starts;
    sz_u64_t const excluded = (sz_line_break_class_mask_neon_(classes_u8x16, sz_line_break_bk_k) |
                               sz_line_break_class_mask_neon_(classes_u8x16, sz_line_break_cr_k) |
                               sz_line_break_class_mask_neon_(classes_u8x16, sz_line_break_lf_k) |
                               sz_line_break_class_mask_neon_(classes_u8x16, sz_line_break_nl_k) |
                               sz_line_break_class_mask_neon_(classes_u8x16, sz_line_break_sp_k) |
                               sz_line_break_class_mask_neon_(classes_u8x16, sz_line_break_zw_k)) &
                              starts;
    sz_u64_t const good_base = starts & ~excluded & ~mark_start;
    sz_u64_t const mark_bytes = sz_u64_fill_right_(mark_start, non_start) | mark_start;
    sz_u64_t const flood = sz_u64_fill_right_(good_base, non_start | mark_bytes);
    sz_u64_t const attached = flood & mark_start;
    sz_u64_t const lone_mark = mark_start & ~attached;

    sz_line_break_byte_frame_neon_t frame;
    //  Reclassify lone marks to AL in every quarter (LB10).
    uint8x16_t const al_u8x16 = vdupq_n_u8(sz_line_break_al_k);
    for (int quarter = 0; quarter < 4; ++quarter) {
        uint8x16_t const lone_select_u8x16 = sz_line_break_byte_mask_from_bits_neon_(lone_mark >> (quarter * 16));
        frame.classes[quarter] = vbslq_u8(lone_select_u8x16, al_u8x16, classes_u8x16[quarter]);
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
    uint8x16_t const *classes_u8x16 = byte_frame.classes;
    //  LB10 reclassify carries the side bits with it: zero the side byte on lone-mark lanes (serial zeros the
    //  descriptor). `vbicq_u8(side, lone_select)` clears those lanes.
    uint8x16_t side_u8x16[4];
    for (int quarter = 0; quarter < 4; ++quarter) {
        uint8x16_t const lone_select_u8x16 = sz_line_break_byte_mask_from_bits_neon_(byte_frame.lone_mark >>
                                                                                     (quarter * 16));
        side_u8x16[quarter] = vbicq_u8(classified.side[quarter], lone_select_u8x16);
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
        frame.effective_class[cls] = sz_line_break_class_mask_neon_(classes_u8x16, (sz_u8_t)cls);
    frame.raw_zwj = sz_line_break_class_mask_neon_(classified.classes, sz_line_break_zwj_k);
    frame.side_pi = sz_line_break_side_mask_neon_(side_u8x16, sz_line_break_side_pi_k);
    frame.side_pf = sz_line_break_side_mask_neon_(side_u8x16, sz_line_break_side_pf_k);
    frame.side_eaw = sz_line_break_side_mask_neon_(side_u8x16, sz_line_break_side_eaw_k);
    frame.side_cn = sz_line_break_side_mask_neon_(side_u8x16, sz_line_break_side_cn_k);
    frame.side_ext = sz_line_break_side_mask_neon_(side_u8x16, sz_line_break_side_ext_k);
    for (int quarter = 0; quarter < 4; ++quarter) {
        vst1q_u8(effective_class_byte_out + quarter * 16, classes_u8x16[quarter]);
        vst1q_u8(side_byte_out + quarter * 16, side_u8x16[quarter]);
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
