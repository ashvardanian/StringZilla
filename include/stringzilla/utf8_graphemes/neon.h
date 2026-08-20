/**
 *  @file   include/stringzilla/utf8_graphemes/neon.h
 *  @author Ash Vardanian
 *  @brief  NEON (AArch64) backend for UAX-29 extended grapheme cluster boundaries, fully vectorized end-to-end.
 *
 *  The NEON twin of the Haswell (AVX2) grapheme kernel. Each 64-byte window lives as four `uint8x16_t` quarters; every
 *  lane is decoded through the shared codepoint substrate (`sz_utf8_rune_decode_window_neon_`), classified into one
 *  packed Grapheme_Cluster_Break descriptor per lane through the shared page-compressed flat table (the page LUT
 *  in-register via `vqtbl4q`, the leaf by a bounded scalar L1 walk — NEON has no gather) plus a register-resident
 *  5-nibble astral cascade, compacted to a codepoint-dense descriptor byte buffer, and resolved by the SHARED portable
 *  boundary engine (`sz_grapheme_window_boundaries_`). No `_pext_u64` / `_pdep_u64` (NEON lacks them — replaced by
 *  scalar sparse bit shuffles), no per-lane scalar rule loop, no serial deferral. The tables and rule engine are
 *  reused verbatim.
 */
#ifndef STRINGZILLA_UTF8_GRAPHEMES_NEON_H_
#define STRINGZILLA_UTF8_GRAPHEMES_NEON_H_

#include "stringzilla/types.h"
#include "stringzilla/utf8_graphemes/tables.h"
#include "stringzilla/utf8_graphemes/serial.h"
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

#pragma region Branchless bit gather and scatter

/** @brief  Drop-in branchless `_pext_u64` (builds a one-shot route). Prefer `sz_grapheme_bit_gather_` when one
 *          selector serves several values, as inside `sz_grapheme_build_masks_neon_`. Bit-exact with BMI2. */
SZ_HELPER_INLINE sz_u64_t sz_grapheme_pext_neon_(sz_u64_t value, sz_u64_t selector) {
    sz_grapheme_bit_route_t const route = sz_grapheme_bit_route_build_(selector);
    return sz_grapheme_bit_gather_(value, &route);
}

/** @brief  Drop-in branchless `_pdep_u64`. Bit-exact with BMI2. */
SZ_HELPER_INLINE sz_u64_t sz_grapheme_pdep_neon_(sz_u64_t value, sz_u64_t selector) {
    sz_grapheme_bit_route_t const route = sz_grapheme_bit_route_build_(selector);
    return sz_grapheme_bit_scatter_(value, &route);
}

#pragma endregion Branchless bit gather and scatter

#pragma region Grapheme_Cluster_Break classifier

/** @brief  Packed descriptor byte for sixteen BMP codepoints (per-lane high = cp>>8, low = cp&0xFF): the `bmp_page_lut_`
 *          page LUT selects one of the 54 distinct 256-byte pages, then `flat_bmp_` is read per lane. The leaf
 *          carries the descriptor directly (the serial `id_to_desc` permute is folded in). Bit-exact with
 *          `sz_rune_grapheme_break_property` over the whole BMP (Hangul included, no separate formula). */
SZ_HELPER_INLINE uint8x16_t sz_grapheme_bmp_descriptor_neon_(uint8x16_t high_bytes_u8x16, uint8x16_t low_bytes_u8x16) {
    return sz_utf8_rune_flat_lookup_neon_(sz_utf8_grapheme_break_bmp_page_lut_, sz_utf8_grapheme_break_flat_bmp_,
                                          (int)sz_utf8_grapheme_break_flat_pages_k, high_bytes_u8x16, low_bytes_u8x16);
}

/** @brief  Packed descriptor byte for sixteen ASTRAL codepoints over offset = cp - 0x10000 (5-nibble cascade), the
 *          NEON twin of the AVX2 astral trie. Per-lane bytes: @p plane_u8x16 = (offset>>16)&0xFF (low nibble
 *          meaningful), @p high_u8x16 = (offset>>8)&0xFF, @p low_u8x16 = offset&0xFF. Gather-free; bit-exact.
 *          Addresses ONE quarter. */
SZ_HELPER_INLINE uint8x16_t sz_grapheme_astral_descriptor_neon_(uint8x16_t plane_u8x16, uint8x16_t high_u8x16,
                                                                uint8x16_t low_u8x16) {
    uint8x16_t const low_nibble_mask_u8x16 = vdupq_n_u8(0x0F);
    uint8x16_t const n4_u8x16 = vandq_u8(plane_u8x16, low_nibble_mask_u8x16);
    uint8x16_t const n3_u8x16 = vandq_u8(vshrq_n_u8(high_u8x16, 4), low_nibble_mask_u8x16);
    uint8x16_t const stage1_index_u8x16 = vorrq_u8(vshlq_n_u8(n4_u8x16, 4), n3_u8x16);
    uint8x16_t const page_u8x16 = sz_utf8_rune_lut256_neon_(sz_utf8_grapheme_break_haswell_astral_stage1_,
                                                            stage1_index_u8x16);
    uint8x16_t const n2_u8x16 = vandq_u8(high_u8x16, low_nibble_mask_u8x16);
    // leaf2 ids fit in a byte (n_leaf2 <= 255), so only the lo plane carries information; the stage3 cascade selects
    // on `leaf2` directly with a tile_count covering every leaf2 id (the hi plane is all-zero and unused).
    uint8x16_t const leaf2_u8x16 = sz_utf8_rune_cascade_stage_neon_(
        sz_utf8_grapheme_break_haswell_astral_stage2_lo_, sz_utf8_grapheme_break_haswell_astral_stage2_lo_count_k / 16,
        page_u8x16, n2_u8x16);
    uint8x16_t const n1_u8x16 = vandq_u8(vshrq_n_u8(low_u8x16, 4), low_nibble_mask_u8x16);
    uint8x16_t const leaf_lo_u8x16 = sz_utf8_rune_cascade_stage_neon_(
        sz_utf8_grapheme_break_haswell_astral_stage3_lo_, sz_utf8_grapheme_break_haswell_astral_stage3_lo_count_k / 16,
        leaf2_u8x16, n1_u8x16);
    uint8x16_t const leaf_hi_u8x16 = sz_utf8_rune_cascade_stage_neon_(
        sz_utf8_grapheme_break_haswell_astral_stage3_hi_, sz_utf8_grapheme_break_haswell_astral_stage3_hi_count_k / 16,
        leaf2_u8x16, n1_u8x16);
    uint8x16_t const n0_u8x16 = vandq_u8(low_u8x16, low_nibble_mask_u8x16);
    uint8x16_t const leaf_group_u8x16 = vorrq_u8(vandq_u8(vshrq_n_u8(leaf_lo_u8x16, 4), low_nibble_mask_u8x16),
                                                 vshlq_n_u8(leaf_hi_u8x16, 4));
    uint8x16_t const leaf_low_nibble_u8x16 = vandq_u8(leaf_lo_u8x16, low_nibble_mask_u8x16);
    uint8x16_t const stage4_lut_index_u8x16 = vorrq_u8(vshlq_n_u8(leaf_low_nibble_u8x16, 4), n0_u8x16);
    // Same gather collapse as the BMP path: descriptor[lane] = astral_stage4_groups[leaf_group*256 + index].
    sz_u8_t leaf_group_lanes[16], stage4_index_lanes[16], descriptor_lanes[16];
    vst1q_u8(leaf_group_lanes, leaf_group_u8x16);
    vst1q_u8(stage4_index_lanes, stage4_lut_index_u8x16);
    for (int lane_index = 0; lane_index < 16; ++lane_index) {
        int const group = leaf_group_lanes[lane_index];
        int const safe_group = group < (int)sz_utf8_grapheme_break_haswell_astral_leaf_groups_k ? group : 0;
        descriptor_lanes[lane_index] =
            sz_utf8_grapheme_break_haswell_astral_stage4_groups_[safe_group * 256 + stage4_index_lanes[lane_index]];
    }
    uint8x16_t const in_range_u8x16 = vcltq_u8(
        leaf_group_u8x16, vdupq_n_u8((sz_u8_t)sz_utf8_grapheme_break_haswell_astral_leaf_groups_k));
    return vandq_u8(vld1q_u8(descriptor_lanes), in_range_u8x16);
}

/** @brief  Per-quarter unsigned `value >= bound` boolean mask (0x00/0xFF lanes), the NEON `vcgeq_u8` twin of the AVX2
 *          `max_epu8(value,bound)==value` idiom. */
SZ_HELPER_INLINE uint8x16_t sz_grapheme_cmpge_epu8_neon_(uint8x16_t value_u8x16, uint8x16_t bound_u8x16) {
    return vcgeq_u8(value_u8x16, bound_u8x16);
}

/** @brief  64-bit unsigned `low <= cp <= high` mask over reconstructed BMP codepoints carried in @p high_byte /
 *          @p low_byte quarters. cp = (high<<8)|low, so the inclusive 16-bit range test is: high in (lo_hi,hi_hi)
 *          unconditionally, or on the boundary high bytes the low byte within bound. Four quarters, branchless. */
SZ_HELPER_INLINE sz_u64_t sz_grapheme_cp_in_range_neon_(uint8x16_t const *high_u8x16, uint8x16_t const *low_u8x16,
                                                        sz_u16_t lo, sz_u16_t hi) {
    sz_u8_t const lo_h = (sz_u8_t)(lo >> 8), lo_l = (sz_u8_t)(lo & 0xFF);
    sz_u8_t const hi_h = (sz_u8_t)(hi >> 8), hi_l = (sz_u8_t)(hi & 0xFF);
    uint8x16_t const lo_h_u8x16 = vdupq_n_u8(lo_h), lo_l_u8x16 = vdupq_n_u8(lo_l);
    uint8x16_t const hi_h_u8x16 = vdupq_n_u8(hi_h), hi_l_u8x16 = vdupq_n_u8(hi_l);
    uint8x16_t in_range_u8x16[4];
    for (int quarter = 0; quarter < 4; ++quarter) {
        uint8x16_t const high_eq_lo_u8x16 = vceqq_u8(high_u8x16[quarter], lo_h_u8x16);
        uint8x16_t const high_eq_hi_u8x16 = vceqq_u8(high_u8x16[quarter], hi_h_u8x16);
        // ge_low: high>lo_h, or (high==lo_h and low>=lo_l). le_high: high<hi_h, or (high==hi_h and low<=hi_l).
        uint8x16_t const ge_low_u8x16 = vorrq_u8(
            vbicq_u8(sz_grapheme_cmpge_epu8_neon_(high_u8x16[quarter], lo_h_u8x16), high_eq_lo_u8x16),
            vandq_u8(high_eq_lo_u8x16, sz_grapheme_cmpge_epu8_neon_(low_u8x16[quarter], lo_l_u8x16)));
        uint8x16_t const le_high_u8x16 = vorrq_u8(
            vbicq_u8(sz_grapheme_cmpge_epu8_neon_(hi_h_u8x16, high_u8x16[quarter]), high_eq_hi_u8x16),
            vandq_u8(high_eq_hi_u8x16, sz_grapheme_cmpge_epu8_neon_(hi_l_u8x16, low_u8x16[quarter])));
        in_range_u8x16[quarter] = vandq_u8(ge_low_u8x16, le_high_u8x16);
    }
    return sz_utf8_mask_combine_neon_(in_range_u8x16[0], in_range_u8x16[1], in_range_u8x16[2], in_range_u8x16[3]);
}

/** @brief  Lanes whose BMP codepoint resolves uniformly to GCB=Other via the CJK / Kana arithmetic ranges (the NEON
 *          twin of `sz_grapheme_cjk_other_haswell_`): `[0x3000,0xA66E] | [0xD7FC,0xFB1D]` minus the interior Extend /
 *          enclosed exceptions. Such lanes need no cold cascade (their descriptor is 0). */
SZ_HELPER_INLINE sz_u64_t sz_grapheme_cjk_other_neon_(uint8x16_t const *high_u8x16, uint8x16_t const *low_u8x16) {
    sz_u64_t const run_a = sz_grapheme_cp_in_range_neon_(high_u8x16, low_u8x16, 0x3000, 0xA66E);
    sz_u64_t const run_b = sz_grapheme_cp_in_range_neon_(high_u8x16, low_u8x16, 0xD7FC, 0xFB1D);
    sz_u64_t const exc_a = sz_grapheme_cp_in_range_neon_(high_u8x16, low_u8x16, 0x302A, 0x3030);
    sz_u64_t const exc_b = sz_grapheme_cp_in_range_neon_(high_u8x16, low_u8x16, 0x3099, 0x309A);
    sz_u64_t const exc_c = sz_grapheme_cp_in_range_neon_(high_u8x16, low_u8x16, 0x303D, 0x303D);
    sz_u64_t const exc_d = sz_grapheme_cp_in_range_neon_(high_u8x16, low_u8x16, 0x3297, 0x3297);
    sz_u64_t const exc_e = sz_grapheme_cp_in_range_neon_(high_u8x16, low_u8x16, 0x3299, 0x3299);
    return (run_a | run_b) & ~(exc_a | exc_b | exc_c | exc_d | exc_e);
}

/** @brief  Build a per-quarter byte-boolean selector (0x00/0xFF) from the 16 lane bits of @p bits at offset @p shift,
 *          the NEON twin of `sz_utf8_byte_mask_from_bits_haswell_` confined to one quarter. */
SZ_HELPER_INLINE uint8x16_t sz_grapheme_byte_mask_from_bits_neon_(sz_u64_t bits, int shift) {
    static sz_u8_t const bit_position_lanes[16] = {1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128};
    static sz_u8_t const lane_half[16] = {0, 0, 0, 0, 0, 0, 0, 0, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF};
    sz_u8_t const low_byte = (sz_u8_t)((bits >> shift) & 0xFF);
    sz_u8_t const high_byte = (sz_u8_t)((bits >> (shift + 8)) & 0xFF);
    uint8x16_t const byte_u8x16 = vbslq_u8(vld1q_u8(lane_half), vdupq_n_u8(high_byte), vdupq_n_u8(low_byte));
    uint8x16_t const position_u8x16 = vld1q_u8(bit_position_lanes);
    return vceqq_u8(vandq_u8(byte_u8x16, position_u8x16), position_u8x16);
}

/** @brief  Per-window per-lane descriptors as four `uint8x16_t` quarters, plus the codepoint-start lane mask and
 *          geometry. The NEON twin of the AVX2 `sz_grapheme_classified_haswell_t` outputs. */
typedef struct sz_grapheme_classified_neon_t {
    uint8x16_t descriptors_u8x16s[4]; /**< Packed descriptor per byte-lane (valid only on start lanes). */
    sz_u64_t start_lanes;             /**< Codepoint-start lanes within the effective span (trimmed to `byte_span`). */
    sz_size_t codepoint_count;        /**< Number of codepoint starts resolved (<= 64). */
    sz_size_t byte_span;              /**< Bytes consumed by the resolved codepoints (offset of the next start). */
} sz_grapheme_classified_neon_t;

/**
 *  @brief  Decode a 64-byte window, classify every lane to a packed descriptor gather-free, and emit the per-lane
 *          descriptor quarters with the trimmed codepoint-start geometry. Mirrors the haswell decode/classify path
 *          (value-based blind reconstruction so malformed input agrees byte-for-byte) without table gather.
 */
SZ_HELPER_INLINE sz_grapheme_classified_neon_t sz_grapheme_classify_window_neon_( //
    sz_u8_t const *text, sz_size_t length, sz_size_t base) {

    sz_utf8_rune_window_neon_t const decoded = sz_utf8_rune_decode_window_neon_(text + base, length - base);
    sz_size_t const loaded = decoded.loaded;
    sz_u64_t const loaded_mask = sz_u64_mask_until_serial_(loaded);
    sz_u64_t start_lanes = decoded.codepoint_starts & loaded_mask;

    // Effective-window trim: a multi-byte lead near the 64-byte edge whose span runs past `loaded` would decode
    // against a wrapped neighbour; defer it to the next window.
    sz_size_t byte_span = loaded;
    if (loaded >= 64) {
        sz_u64_t const two = decoded.two_byte_starts & loaded_mask;
        sz_u64_t const three = decoded.three_byte_starts & loaded_mask;
        sz_u64_t const four = decoded.four_byte_starts & loaded_mask;
        sz_u64_t const overrun = (two & ~sz_u64_mask_until_serial_(loaded - 1)) |
                                 (three & ~sz_u64_mask_until_serial_(loaded - 2)) |
                                 (four & ~sz_u64_mask_until_serial_(loaded - 3));
        byte_span = overrun ? (sz_size_t)sz_u64_ctz_neon_(overrun) : loaded;
        start_lanes &= sz_u64_mask_until_serial_(byte_span);
    }

    int const quarters_used = (int)((loaded + 15) >> 4); // classify work proportional to the loaded span
    uint8x16_t raw_u8x16[4];
    for (int quarter = 0; quarter < 4; ++quarter) raw_u8x16[quarter] = decoded.window_u8x16s[quarter];
    uint8x16_t next1_u8x16[4], next2_u8x16[4], next3_u8x16[4];
    sz_utf8_forward_neighbours_neon_(raw_u8x16, next1_u8x16, next2_u8x16, next3_u8x16);

    // Zero each `next k` lane whose source byte index reaches `loaded` (short final window only), matching the haswell
    // maskz neighbour fetch / serial blind decode that pads out-of-window continuation bytes with 0. On a FULL window
    // (`loaded == 64`) `loaded_mask` is all-ones so this is inert and the only touched lanes (truncated edge leads) are
    // already deferred by the effective-window trim — so skip the neighbour mask builds entirely on the hot path.
    if (loaded < 64) {
        for (int quarter = 0; quarter < 4; ++quarter) {
            next1_u8x16[quarter] = vandq_u8(next1_u8x16[quarter],
                                            sz_grapheme_byte_mask_from_bits_neon_(loaded_mask >> 1, quarter * 16));
            next2_u8x16[quarter] = vandq_u8(next2_u8x16[quarter],
                                            sz_grapheme_byte_mask_from_bits_neon_(loaded_mask >> 2, quarter * 16));
            next3_u8x16[quarter] = vandq_u8(next3_u8x16[quarter],
                                            sz_grapheme_byte_mask_from_bits_neon_(loaded_mask >> 3, quarter * 16));
        }
    }

    // Reconstruct per-lane high/low (BMP) and plane/mid/low (astral) BLINDLY from the raw lead + zeroed neighbours,
    // mirroring `sz_grapheme_classify_window_haswell_` byte-for-byte. ASCII lanes take identity (high=0, low=raw).
    uint8x16_t const c03_u8x16 = vdupq_n_u8(0x03), c07_u8x16 = vdupq_n_u8(0x07), c0f_u8x16 = vdupq_n_u8(0x0F),
                     c1f_u8x16 = vdupq_n_u8(0x1F), c3f_u8x16 = vdupq_n_u8(0x3F), c80_u8x16 = vdupq_n_u8(0x80);
    sz_u64_t const three_byte = decoded.three_byte_starts;
    sz_u64_t const four_byte = decoded.four_byte_starts;

    uint8x16_t high_u8x16[4], low_u8x16[4], plane_u8x16[4], mid_u8x16[4], alo_u8x16[4], ascii_desc_u8x16[4],
        desc_u8x16[4];
    uint8x16_t ascii_bool_u8x16[4];
    for (int quarter = quarters_used; quarter < 4; ++quarter)
        desc_u8x16[quarter] = ascii_bool_u8x16[quarter] = plane_u8x16[quarter] = vdupq_n_u8(0);
    for (int quarter = 0; quarter < quarters_used; ++quarter) {
        uint8x16_t const here_u8x16 = raw_u8x16[quarter];
        uint8x16_t const n1_u8x16 = next1_u8x16[quarter], n2_u8x16 = next2_u8x16[quarter],
                         n3_u8x16 = next3_u8x16[quarter];

        // 2-byte: high = ((lead & 0x1F) >> 2) & 0x07; low = ((lead & 0x03) << 6) | (next1 & 0x3F).
        uint8x16_t const two_high_u8x16 = sz_utf8_srl8_neon_(vandq_u8(here_u8x16, c1f_u8x16), 2, 0x07);
        uint8x16_t const two_low_u8x16 = vorrq_u8(vshlq_n_u8(vandq_u8(here_u8x16, c03_u8x16), 6),
                                                  vandq_u8(n1_u8x16, c3f_u8x16));
        // 3-byte: high = ((lead & 0x0F) << 4) | ((next1 >> 2) & 0x0F); low = ((next1 & 0x03) << 6) | (next2 & 0x3F).
        uint8x16_t const three_high_u8x16 = vorrq_u8(vshlq_n_u8(vandq_u8(here_u8x16, c0f_u8x16), 4),
                                                     sz_utf8_srl8_neon_(n1_u8x16, 2, 0x0F));
        uint8x16_t const three_low_u8x16 = vorrq_u8(vshlq_n_u8(vandq_u8(n1_u8x16, c03_u8x16), 6),
                                                    vandq_u8(n2_u8x16, c3f_u8x16));

        uint8x16_t const three_select_u8x16 = sz_grapheme_byte_mask_from_bits_neon_(three_byte, quarter * 16);
        uint8x16_t const ascii_select_u8x16 = vceqq_u8(vandq_u8(here_u8x16, c80_u8x16), vdupq_n_u8(0));
        ascii_bool_u8x16[quarter] = ascii_select_u8x16;
        uint8x16_t const non_ascii_high_u8x16 = vbslq_u8(three_select_u8x16, three_high_u8x16, two_high_u8x16);
        uint8x16_t const non_ascii_low_u8x16 = vbslq_u8(three_select_u8x16, three_low_u8x16, two_low_u8x16);
        // high = ascii ? 0 : non_ascii_high; low = ascii ? raw : non_ascii_low.
        high_u8x16[quarter] = vbicq_u8(non_ascii_high_u8x16, ascii_select_u8x16);
        low_u8x16[quarter] = vbslq_u8(ascii_select_u8x16, here_u8x16, non_ascii_low_u8x16);

        // Astral plane/mid/low (offset domain): plane = ((b0&7)<<2)|((next1>>4)&3);
        // mid = ((next1&F)<<4)|((next2>>2)&F); lo = ((next2&3)<<6)|(next3&3F).
        plane_u8x16[quarter] = vorrq_u8(vshlq_n_u8(vandq_u8(here_u8x16, c07_u8x16), 2),
                                        vandq_u8(vshrq_n_u8(n1_u8x16, 4), c03_u8x16));
        mid_u8x16[quarter] = vorrq_u8(vshlq_n_u8(vandq_u8(n1_u8x16, c0f_u8x16), 4),
                                      sz_utf8_srl8_neon_(n2_u8x16, 2, 0x0F));
        alo_u8x16[quarter] = vorrq_u8(vshlq_n_u8(vandq_u8(n2_u8x16, c03_u8x16), 6), vandq_u8(n3_u8x16, c3f_u8x16));

        // A 4-byte lead's blind codepoint is `(plane<<16)|(mid<<8)|alo`, NOT the 2-byte fold seated in high/low above.
        // Override its BMP high/low to mid/alo so a plane-0 (overlong) 4-byte lead resolves on the BMP path exactly as
        // serial/haswell do — those carry the full 21-bit value and split BMP-vs-astral on `cp < 0x10000`, so a plane-0
        // 4-byte lane (e.g. `F0 87 ..`) is a BMP codepoint, not astral. (The astral overwrite below is then gated on a
        // non-zero in-range plane, mirroring haswell's `is_astral`.)
        uint8x16_t const four_sel_blend_u8x16 = sz_grapheme_byte_mask_from_bits_neon_(four_byte, quarter * 16);
        high_u8x16[quarter] = vbslq_u8(four_sel_blend_u8x16, mid_u8x16[quarter], high_u8x16[quarter]);
        low_u8x16[quarter] = vbslq_u8(four_sel_blend_u8x16, alo_u8x16[quarter], low_u8x16[quarter]);

        // ASCII descriptor (cp < 0x80) via a single 256-LUT over the raw byte — the cheap gated fast path so a
        // pure-ASCII window never pays the full BMP nibble cascade. Mirrors haswell's `ascii_desc` LUT.
        ascii_desc_u8x16[quarter] = sz_utf8_rune_lut256_neon_(sz_utf8_grapheme_break_haswell_ascii_desc_, here_u8x16);
        desc_u8x16[quarter] = ascii_desc_u8x16[quarter];
    }

    sz_u64_t const ascii = sz_utf8_mask_combine_neon_(ascii_bool_u8x16[0], ascii_bool_u8x16[1], ascii_bool_u8x16[2],
                                                      ascii_bool_u8x16[3]);

    // Any non-ASCII lead present? gate the cold full-BMP cascade behind it. The CJK / Kana arithmetic ranges resolve
    // to GCB=Other (descriptor 0), so a pure-CJK or pure-ASCII window skips the cascade entirely (haswell `cjk_other`).
    sz_u64_t const non_ascii = loaded_mask & ~ascii;
    sz_u64_t cold = non_ascii;
    if (non_ascii) {
        sz_u64_t const cjk_other = sz_grapheme_cjk_other_neon_(high_u8x16, low_u8x16) & non_ascii;
        cold = non_ascii & ~cjk_other;
        // Force the CJK-other lanes to descriptor 0 (their ASCII-LUT value on a non-ASCII byte is junk; clear it).
        for (int quarter = 0; quarter < quarters_used; ++quarter)
            desc_u8x16[quarter] = vbicq_u8(desc_u8x16[quarter],
                                           sz_grapheme_byte_mask_from_bits_neon_(cjk_other, quarter * 16));
    }
    if (cold) {
        // A cold (non-ASCII, non-CJK-other) lane is present: resolve EVERY lane through the full-BMP cascade,
        // overwriting the ASCII fast path. ASCII and CJK-other lanes resolve identically (the cascade reproduces 0
        // for the carved CJK ranges), so the blend below is exact.
        for (int quarter = 0; quarter < quarters_used; ++quarter) {
            uint8x16_t const bmp_u8x16 = sz_grapheme_bmp_descriptor_neon_(high_u8x16[quarter], low_u8x16[quarter]);
            desc_u8x16[quarter] = vbslq_u8(ascii_bool_u8x16[quarter], ascii_desc_u8x16[quarter], bmp_u8x16);
        }
    }
    // A 4-byte lead splits three ways on its blind plane = bits[16..20] of cp, matching haswell/serial value dispatch
    // (`is_bmp = cp < 0x10000`, `is_astral = 0x10000 ≤ cp < 0x110000`, else Other):
    //   plane == 0      → BMP codepoint (cp = (mid<<8)|alo); already resolved through the BMP path above, since the
    //                      high/low override seated `mid`/`alo` for these lanes (e.g. overlong `F0 87 ..`).
    //   plane in [1,16] → genuine astral (cp in 0x10000..0x10FFFF); overwrite with the 4-byte trie descriptor.
    //   plane ≥ 17      → cp ≥ 0x110000 (e.g. overlong `F4 90 80 80`); neither BMP nor astral, force Other (0).
    // `plane[q]` carries the 5-bit plane per 4-byte-lead lane (junk on other lanes, gated out by `four_byte`).
    uint8x16_t const plane_one_u8x16 = vdupq_n_u8(1), plane_sixteen_u8x16 = vdupq_n_u8(0x10);
    uint8x16_t plane_nonzero_q_u8x16[4], plane_le16_q_u8x16[4];
    for (int quarter = 0; quarter < 4; ++quarter) {
        plane_nonzero_q_u8x16[quarter] = vcgeq_u8(plane_u8x16[quarter], plane_one_u8x16);
        plane_le16_q_u8x16[quarter] = sz_grapheme_cmpge_epu8_neon_(plane_sixteen_u8x16, plane_u8x16[quarter]);
    }
    sz_u64_t const plane_nonzero = sz_utf8_mask_combine_neon_(plane_nonzero_q_u8x16[0], plane_nonzero_q_u8x16[1],
                                                              plane_nonzero_q_u8x16[2], plane_nonzero_q_u8x16[3]);
    sz_u64_t const plane_le_16 = sz_utf8_mask_combine_neon_(plane_le16_q_u8x16[0], plane_le16_q_u8x16[1],
                                                            plane_le16_q_u8x16[2], plane_le16_q_u8x16[3]);
    sz_u64_t const is_astral = four_byte & plane_nonzero & plane_le_16 & loaded_mask;
    sz_u64_t const is_overrange = four_byte & plane_nonzero & ~plane_le_16 & loaded_mask;
    if (is_astral) {
        // offset = cp - 0x10000; high16/low16 unchanged, plane nibble = cp_plane - 1.
        for (int quarter = 0; quarter < quarters_used; ++quarter) {
            uint8x16_t const four_sel_u8x16 = sz_grapheme_byte_mask_from_bits_neon_(is_astral, quarter * 16);
            uint8x16_t const plane_off_u8x16 = vsubq_u8(vandq_u8(four_sel_u8x16, plane_u8x16[quarter]), vdupq_n_u8(1));
            uint8x16_t const astral_u8x16 = sz_grapheme_astral_descriptor_neon_(plane_off_u8x16, mid_u8x16[quarter],
                                                                                alo_u8x16[quarter]);
            desc_u8x16[quarter] = vbslq_u8(four_sel_u8x16, astral_u8x16, desc_u8x16[quarter]);
        }
    }
    if (is_overrange) {
        // cp ≥ 0x110000: clear to Other (0); the BMP-path value seated on these lanes is meaningless.
        for (int quarter = 0; quarter < quarters_used; ++quarter)
            desc_u8x16[quarter] = vbicq_u8(desc_u8x16[quarter],
                                           sz_grapheme_byte_mask_from_bits_neon_(is_overrange, quarter * 16));
    }

    // `0xF8..0xFF` begin no valid sequence and match no lead-length mask; force their start lanes to the Other
    // descriptor (0) so they classify neighbour-independently, matching serial/haswell (U+FFFD → Other).
    uint8x16_t invalid_bool_u8x16[4];
    for (int quarter = quarters_used; quarter < 4; ++quarter) invalid_bool_u8x16[quarter] = vdupq_n_u8(0);
    for (int quarter = 0; quarter < quarters_used; ++quarter)
        invalid_bool_u8x16[quarter] = vcgeq_u8(raw_u8x16[quarter], vdupq_n_u8(0xF8));
    sz_u64_t const invalid_lead = sz_utf8_mask_combine_neon_(invalid_bool_u8x16[0], invalid_bool_u8x16[1],
                                                             invalid_bool_u8x16[2], invalid_bool_u8x16[3]);
    for (int quarter = 0; quarter < quarters_used; ++quarter)
        desc_u8x16[quarter] = vbicq_u8(desc_u8x16[quarter],
                                       sz_grapheme_byte_mask_from_bits_neon_(invalid_lead, quarter * 16));

    sz_grapheme_classified_neon_t result;
    for (int quarter = 0; quarter < 4; ++quarter) result.descriptors_u8x16s[quarter] = desc_u8x16[quarter];
    result.start_lanes = start_lanes;
    result.codepoint_count = (sz_size_t)sz_u64_popcount_neon_(start_lanes);
    result.byte_span = byte_span;
    return result;
}

#pragma endregion Grapheme_Cluster_Break classifier

#pragma region Boundary algebra extractor

/**
 *  @brief  Build the codepoint-dense per-class window masks from the per-lane descriptor quarters at the codepoint-start
 *          lanes — the NEON twin of `sz_grapheme_build_masks_haswell_` feeding the SAME portable engine. Each per-class
 *          membership mask is built in the BYTE-lane domain (`vceqq_u8` quarters -> `mask_combine`), then compacted to
 *          the codepoint-dense domain by a single software-`pext` over the start lanes (the BMI2-free analogue of the
 *          haswell `_pext_u64`). No scalar per-lane loop, no table gather, no rule control flow.
 */
SZ_HELPER_INLINE sz_grapheme_window_masks_t sz_grapheme_build_masks_neon_(sz_grapheme_classified_neon_t classified,
                                                                          sz_u64_t valid) {
    sz_u64_t const starts = classified.start_lanes;
    // Every class compaction below gathers the byte-lane mask down to the codepoint-dense domain over the SAME
    // `starts` selector, so build the bit-route once and reuse it for all 18 gathers (the branchless PEXT).
    sz_grapheme_bit_route_t const start_route = sz_grapheme_bit_route_build_(starts);
    uint8x16_t const nibble_mask_u8x16 = vdupq_n_u8(0x0F);
    uint8x16_t class_q_u8x16[4], desc_q_u8x16[4];
    for (int quarter = 0; quarter < 4; ++quarter) {
        desc_q_u8x16[quarter] = classified.descriptors_u8x16s[quarter];
        class_q_u8x16[quarter] = vandq_u8(desc_q_u8x16[quarter], nibble_mask_u8x16);
    }

    sz_grapheme_window_masks_t masks;
    for (int class_index = 0; class_index < 14; ++class_index) {
        uint8x16_t const class_id_u8x16 = vdupq_n_u8((sz_u8_t)class_index);
        sz_u64_t const byte_mask = sz_utf8_mask_combine_neon_(
            vceqq_u8(class_q_u8x16[0], class_id_u8x16), vceqq_u8(class_q_u8x16[1], class_id_u8x16),
            vceqq_u8(class_q_u8x16[2], class_id_u8x16), vceqq_u8(class_q_u8x16[3], class_id_u8x16));
        masks.class_bit[class_index] = sz_grapheme_bit_gather_(byte_mask, &start_route) & valid;
    }
    uint8x16_t const ext_bit_u8x16 = vdupq_n_u8(0x40);
    uint8x16_t ext_q_u8x16[4];
    for (int quarter = 0; quarter < 4; ++quarter)
        ext_q_u8x16[quarter] = vceqq_u8(vandq_u8(desc_q_u8x16[quarter], ext_bit_u8x16), ext_bit_u8x16);
    sz_u64_t const ext_byte = sz_utf8_mask_combine_neon_(ext_q_u8x16[0], ext_q_u8x16[1], ext_q_u8x16[2],
                                                         ext_q_u8x16[3]);
    masks.extended_pictographic = sz_grapheme_bit_gather_(ext_byte, &start_route) & valid;

    uint8x16_t const incb_consonant_u8x16 = vdupq_n_u8((sz_u8_t)sz_grapheme_incb_consonant_k);
    uint8x16_t const incb_extend_u8x16 = vdupq_n_u8((sz_u8_t)sz_grapheme_incb_extend_k);
    uint8x16_t const incb_linker_u8x16 = vdupq_n_u8((sz_u8_t)sz_grapheme_incb_linker_k);
    uint8x16_t consonant_q_u8x16[4], extend_q_u8x16[4], linker_q_u8x16[4];
    for (int quarter = 0; quarter < 4; ++quarter) {
        uint8x16_t const incb_u8x16 = vandq_u8(vshrq_n_u8(desc_q_u8x16[quarter], 4), vdupq_n_u8(0x03));
        consonant_q_u8x16[quarter] = vceqq_u8(incb_u8x16, incb_consonant_u8x16);
        extend_q_u8x16[quarter] = vceqq_u8(incb_u8x16, incb_extend_u8x16);
        linker_q_u8x16[quarter] = vceqq_u8(incb_u8x16, incb_linker_u8x16);
    }
    masks.indic_consonant = sz_grapheme_bit_gather_(
                                sz_utf8_mask_combine_neon_(consonant_q_u8x16[0], consonant_q_u8x16[1],
                                                           consonant_q_u8x16[2], consonant_q_u8x16[3]),
                                &start_route) &
                            valid;
    masks.indic_extend = sz_grapheme_bit_gather_(sz_utf8_mask_combine_neon_(extend_q_u8x16[0], extend_q_u8x16[1],
                                                                            extend_q_u8x16[2], extend_q_u8x16[3]),
                                                 &start_route) &
                         valid;
    masks.indic_linker = sz_grapheme_bit_gather_(sz_utf8_mask_combine_neon_(linker_q_u8x16[0], linker_q_u8x16[1],
                                                                            linker_q_u8x16[2], linker_q_u8x16[3]),
                                                 &start_route) &
                         valid;
    return masks;
}

#pragma endregion Boundary algebra extractor

#pragma region Grapheme forward driver

SZ_API_COMPTIME sz_size_t sz_utf8_graphemes_neon(          //
    sz_cptr_t text, sz_size_t length,                      //
    sz_size_t *cluster_starts, sz_size_t *cluster_lengths, //
    sz_size_t clusters_capacity, sz_size_t *bytes_consumed) {

    sz_size_t clusters = 0;
    if (length == 0 || clusters_capacity == 0) {
        if (bytes_consumed) *bytes_consumed = 0;
        return 0;
    }
    sz_u8_t const *text_u8 = (sz_u8_t const *)text;
    sz_grapheme_carry_t carry = sz_grapheme_carry_empty_();
    sz_size_t cluster_start = 0;
    sz_size_t base = 0;
    while (base < length) {
        sz_grapheme_classified_neon_t const window = sz_grapheme_classify_window_neon_(text_u8, length, base);
        if (window.codepoint_count == 0) break; // defensive: cannot happen for valid input

        int const codepoint_count = (int)window.codepoint_count;
        sz_u64_t const valid = (codepoint_count >= 64) ? ~0ull : ((1ull << codepoint_count) - 1);
        sz_grapheme_window_masks_t const masks = sz_grapheme_build_masks_neon_(window, valid);
        sz_u64_t const dense_boundary = sz_grapheme_window_boundaries_(&masks, codepoint_count, valid, &carry);
        // Scatter dense boundary bits back to codepoint-start byte lanes (software `pdep` deposits the j-th dense bit
        // into the j-th set start lane), recovering the byte-domain boundary mask the drain consumes.
        sz_u64_t boundary = sz_grapheme_pdep_neon_(dense_boundary, window.start_lanes);

        // GB1 anchor at byte 0 of the first window is the open cluster's own start, not a new break: clear it.
        if (base == 0) boundary &= ~1ull;
        clusters = sz_utf8_rune_drain_forward_neon_(boundary, base, cluster_starts, cluster_lengths, clusters,
                                                    clusters_capacity, &cluster_start);
        if (clusters == clusters_capacity) {
            if (bytes_consumed) *bytes_consumed = cluster_start;
            return clusters;
        }
        base += window.byte_span;
    }

    cluster_starts[clusters] = cluster_start;
    cluster_lengths[clusters] = length - cluster_start;
    ++clusters;
    if (bytes_consumed) *bytes_consumed = length;
    return clusters;
}

#pragma endregion Grapheme forward driver

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#endif // SZ_USE_NEON

#ifdef __cplusplus
}
#endif

#endif // STRINGZILLA_UTF8_GRAPHEMES_NEON_H_
