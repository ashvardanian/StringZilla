/**
 *  @brief NEON (AArch64) backend for UAX-29 sentence boundaries.
 *  @file include/stringzilla/utf8_sentences/neon.h
 *  @author Ash Vardanian
 */
#ifndef STRINGZILLA_UTF8_SENTENCES_NEON_H_
#define STRINGZILLA_UTF8_SENTENCES_NEON_H_

#include "stringzilla/types.h"
#include "stringzilla/utf8_sentences/tables.h"
#include "stringzilla/utf8_sentences/serial.h"
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

#pragma region UAX 29 Sentence Boundaries forward kernel

#pragma region Scalar bit shuffles

/** @brief  Software `_pext_u64`: gather the bits of @p value selected by @p selector, packed to the low end (bit `j`
 *          of the result = the `j`-th set bit of @p value within @p selector). NEON has no `pext`; the sparse loop
 *          trips once per set @p selector bit (codepoint-dense compaction over the start lanes). Bit-exact with BMI2. */
SZ_HELPER_AUTO sz_u64_t sz_sentence_break_pext_neon_(sz_u64_t value, sz_u64_t selector) {
    sz_u64_t result = 0;
    sz_u64_t out_bit = 1;
    while (selector) {
        sz_u64_t const low = selector & (~selector + 1); // lowest set bit of `selector`
        if (value & low) result |= out_bit;
        out_bit <<= 1;
        selector &= selector - 1;
    }
    return result;
}

/** @brief  Software `_pdep_u64`: scatter the low bits of @p value into the positions set in @p selector (the `j`-th
 *          set bit of @p selector receives bit `j` of @p value). NEON has no `pdep`; the sparse loop trips once per
 *          set @p selector bit (the dense-boundary scatter back onto codepoint-start lanes). Bit-exact with BMI2. */
SZ_HELPER_AUTO sz_u64_t sz_sentence_break_pdep_neon_(sz_u64_t value, sz_u64_t selector) {
    sz_u64_t result = 0;
    while (selector) {
        sz_u64_t const low = selector & (~selector + 1); // lowest set bit of `selector`
        if (value & 1) result |= low;
        value >>= 1;
        selector &= selector - 1;
    }
    return result;
}

#pragma endregion Scalar bit shuffles

#pragma region In register vectorized classifier

/*  The NEON twin of the Haswell / Ice Lake Sentence_Break classifier: a contiguous run of codepoints resolves to
 *  per-codepoint Sentence_Break class bytes with ZERO per-lane scalar loop and NO serial deferral.
 *
 *  Each 64-byte window lives as four `uint8x16_t` quarters (`window[0]` = lanes [0,16), ... `window[3]` = lanes
 *  [48,64)) instead of haswell's two halves; every per-lane class compare is `vceqq_u8` per quarter, the four boolean
 *  quarters OR-collapsed to a `sz_u64_t` via `mask_combine_neon_`. The per-lane (high, low) codepoint byte pair reads
 *  the shared page-compressed flat table via @ref sz_utf8_rune_flat_lookup_neon_ over the whole BMP, and a 5-nibble
 *  `vqtbl` astral cascade for 4-byte leads. Both emit the Sentence_Break class byte directly, bit-identical with
 *  `sz_rune_sentence_break_property` over the entire code space. */

/** @brief  Expand a 16-bit lane mask into a `uint8x16_t` select vector (byte `i` = 0xFF when bit `i` is set), the NEON
 *          twin of @ref sz_utf8_byte_mask_from_bits_haswell_ confined to one quarter. */
SZ_HELPER_INLINE uint8x16_t sz_sentence_break_byte_mask_from_bits_neon_(sz_u64_t bits) {
    static sz_u8_t const byte_router_lanes[16] = {0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1};
    static sz_u8_t const bit_select_lanes[16] = {1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128};
    uint8x16_t const broadcast = vreinterpretq_u8_u16(vdupq_n_u16((sz_u16_t)bits));
    uint8x16_t const byte_router = vld1q_u8(byte_router_lanes);
    uint8x16_t const spread = vqtbl1q_u8(broadcast, byte_router);
    uint8x16_t const bit_select = vld1q_u8(bit_select_lanes);
    uint8x16_t const isolated = vandq_u8(spread, bit_select);
    return vceqq_u8(isolated, bit_select);
}

/** @brief  Reconstruct the BMP (2-/3-byte) codepoint high/low bytes for one 16-lane quarter from EDGE-MASKED forward
 *          neighbours — the NEON twin of @ref sz_utf8_sentence_break_bmp_highlow_haswell_. The shared rune-window decode
 *          built `window.{high,low}` from its own mod-64-wrapping neighbours, so a 2-/3-byte lead straddling the loaded
 *          edge (the truncated trailing lead at end-of-input, where `complete_limit` does not trim because `!more_text`)
 *          would read a wrapped byte as its missing continuation. Recomputing from the driver's `keep`-masked neighbours
 *          makes those missing continuations read as zero, exactly like serial's blind decode (`text[start+k]` past the
 *          input reads 0). Lanes that are neither a 2- nor a 3-byte lead keep `low = raw`, `high = 0`; the classifier
 *          re-seats raw / 4-byte lanes anyway. */
SZ_HELPER_AUTO void sz_utf8_sentence_break_bmp_highlow_neon_( //
    uint8x16_t raw, uint8x16_t next1, uint8x16_t next2, sz_u64_t two_bits, sz_u64_t three_bits, uint8x16_t *out_high,
    uint8x16_t *out_low) {
    uint8x16_t const low_two_bits = vdupq_n_u8(0x03);
    uint8x16_t const low_four_bits = vdupq_n_u8(0x0F);
    uint8x16_t const low_five_bits = vdupq_n_u8(0x1F);
    uint8x16_t const low_six_bits = vdupq_n_u8(0x3F);
    //  2-byte: high = (b0 & 0x1F) >> 2, low = ((b0 & 0x03) << 6) | (next1 & 0x3F).
    uint8x16_t const two_high = sz_utf8_srl8_neon_(vandq_u8(raw, low_five_bits), 2, 0x07);
    uint8x16_t const two_low = vorrq_u8(vshlq_n_u8(vandq_u8(raw, low_two_bits), 6), vandq_u8(next1, low_six_bits));
    //  3-byte: high = ((b0 & 0x0F) << 4) | ((next1 >> 2) & 0x0F), low = ((next1 & 0x03) << 6) | (next2 & 0x3F).
    uint8x16_t const three_high = vorrq_u8(vshlq_n_u8(vandq_u8(raw, low_four_bits), 4),
                                           sz_utf8_srl8_neon_(next1, 2, 0x0F));
    uint8x16_t const three_low = vorrq_u8(vshlq_n_u8(vandq_u8(next1, low_two_bits), 6), vandq_u8(next2, low_six_bits));
    uint8x16_t const two_select = sz_sentence_break_byte_mask_from_bits_neon_(two_bits);
    uint8x16_t const three_select = sz_sentence_break_byte_mask_from_bits_neon_(three_bits);
    uint8x16_t high = vdupq_n_u8(0);
    uint8x16_t low = raw;
    high = vbslq_u8(two_select, two_high, high);
    low = vbslq_u8(two_select, two_low, low);
    high = vbslq_u8(three_select, three_high, high);
    low = vbslq_u8(three_select, three_low, low);
    *out_high = high, *out_low = low;
}

/** @brief  Sentence_Break class byte for sixteen BMP codepoints (per-lane high = cp>>8, low = cp&0xFF) from the flat
 *          page-compressed table via @ref sz_utf8_rune_flat_lookup_neon_, the NEON twin of
 *          @ref sz_utf8_sentence_break_bmp_class_haswell_. Bit-exact with `sz_rune_sentence_break_property` over the
 *          whole BMP. Operates on one quarter; the caller iterates the four quarters. */
SZ_HELPER_AUTO uint8x16_t sz_utf8_sentence_break_bmp_class_neon_(uint8x16_t high_bytes_u8x16,
                                                                 uint8x16_t low_bytes_u8x16) {
    return sz_utf8_rune_flat_lookup_neon_(sz_utf8_sentence_break_bmp_page_lut_, sz_utf8_sentence_break_flat_bmp_,
                                          (int)sz_utf8_sentence_break_flat_pages_k, high_bytes_u8x16, low_bytes_u8x16);
}

/** @brief  Sentence_Break class byte for sixteen ASTRAL codepoints over the 20-bit offset = cp - 0x10000 (5-nibble
 *          cascade), the NEON twin of @ref sz_utf8_sentence_break_astral_class_haswell_. Per-lane bytes:
 *          @p plane = (offset>>16)&0xFF (low nibble meaningful), @p high = (offset>>8)&0xFF, @p low = offset&0xFF.
 *          Bit-exact with `sz_rune_sentence_break_property` over all astral. Operates on one quarter. */
SZ_HELPER_AUTO uint8x16_t sz_utf8_sentence_break_astral_class_neon_(uint8x16_t plane, uint8x16_t high, uint8x16_t low) {
    uint8x16_t const low_nibble_mask = vdupq_n_u8(0x0F);
    uint8x16_t const n4 = vandq_u8(plane, low_nibble_mask);
    uint8x16_t const n3 = vandq_u8(vshrq_n_u8(high, 4), low_nibble_mask);
    uint8x16_t const stage1_index = vorrq_u8(vshlq_n_u8(n4, 4), n3);
    uint8x16_t const page = sz_utf8_rune_lut256_neon_(sz_utf8_sentence_break_haswell_astral_stage1_, stage1_index);
    uint8x16_t const n2 = vandq_u8(high, low_nibble_mask);
    uint8x16_t const leaf2_lo = sz_utf8_rune_cascade_stage_neon_(
        sz_utf8_sentence_break_haswell_astral_stage2_lo_, sz_utf8_sentence_break_haswell_astral_stage2_lo_count_k / 16,
        page, n2);
    uint8x16_t const n1 = vandq_u8(vshrq_n_u8(low, 4), low_nibble_mask);
    uint8x16_t const leaf_lo = sz_utf8_rune_cascade_stage_neon_(
        sz_utf8_sentence_break_haswell_astral_stage3_lo_, sz_utf8_sentence_break_haswell_astral_stage3_lo_count_k / 16,
        leaf2_lo, n1);
    uint8x16_t const leaf_hi = sz_utf8_rune_cascade_stage_neon_(
        sz_utf8_sentence_break_haswell_astral_stage3_hi_, sz_utf8_sentence_break_haswell_astral_stage3_hi_count_k / 16,
        leaf2_lo, n1);
    uint8x16_t const n0 = vandq_u8(low, low_nibble_mask);
    uint8x16_t const leaf_group = vorrq_u8(vandq_u8(vshrq_n_u8(leaf_lo, 4), low_nibble_mask), vshlq_n_u8(leaf_hi, 4));
    uint8x16_t const leaf_low_nibble = vandq_u8(leaf_lo, low_nibble_mask);
    uint8x16_t const stage4_lut_index = vorrq_u8(vshlq_n_u8(leaf_low_nibble, 4), n0);
    uint8x16_t result = vdupq_n_u8(0);
    for (int group = 0; group < (int)sz_utf8_sentence_break_haswell_astral_leaf_groups_k; ++group) {
        uint8x16_t const value = sz_utf8_rune_lut256_neon_(
            sz_utf8_sentence_break_haswell_astral_stage4_groups_ + group * 256, stage4_lut_index);
        uint8x16_t const here = vceqq_u8(leaf_group, vdupq_n_u8((sz_u8_t)group));
        result = vbslq_u8(here, value, result);
    }
    return result;
}

/** @brief  Per-byte-lane Sentence_Break class for ONE decoded window quarter, fully in-register and zero-scalar - the
 *          NEON twin of @ref sz_utf8_sentence_break_classify_half_haswell_. The decoded window only carries the
 *          2-/3-byte (high, low) reconstruction; this leaf rebuilds the ASCII (`low = raw`, `high = 0`) and 4-byte
 *          (`high`/`low` from the four-byte formula) codepoint bytes before the cascade, exactly as the haswell driver
 *          reconstructs them. BMP lanes go through the BMP cascade; 4-byte lanes are routed by reconstructed plane
 *          through the astral cascade. The class on non-codepoint-start lanes is irrelevant (the dense compaction only
 *          reads start lanes), so those lanes are never selected. */
SZ_HELPER_AUTO uint8x16_t sz_utf8_sentence_break_classify_quarter_neon_( //
    uint8x16_t window_high, uint8x16_t window_low, uint8x16_t raw, uint8x16_t next1, uint8x16_t next2, uint8x16_t next3,
    sz_u64_t four_byte_bits) {
    uint8x16_t const low_two_bits = vdupq_n_u8(0x03);
    uint8x16_t const low_four_bits = vdupq_n_u8(0x0F);
    uint8x16_t const low_six_bits = vdupq_n_u8(0x3F);

    //  Raw-byte reconstruction (codepoint == raw byte: low = raw, high = 0) for every lane that is NOT a 2-/3-/4-byte
    //  lead — ASCII (`raw < 0x80`), continuation bytes `0x80..0xBF` (a lone continuation forced to a start at position 0
    //  decodes to its raw value, e.g. `0x85` → U+0085 NEL), and the non-lead bytes `0xF8..0xFF`. The decode window
    //  pre-folds the 2-byte arithmetic into high/low on ALL lanes, so these must be overwritten. The 2-/3-/4-byte leads
    //  are exactly `0xC0..0xF7`, so raw treatment is the complement `raw < 0xC0 || raw ≥ 0xF8`, matching serial's blind
    //  `rune = lead` decode and the haswell / icelake classifiers (which seat `low = raw`, `high = 0` on every such
    //  lane); 4-byte leads `0xF0..0xF7` are re-seated by the `four_select` blend below.
    uint8x16_t const raw_select = vorrq_u8(vcltq_u8(raw, vdupq_n_u8(0xC0)), vcgeq_u8(raw, vdupq_n_u8(0xF8)));
    uint8x16_t low = vbslq_u8(raw_select, raw, window_low);
    uint8x16_t high = vbicq_u8(window_high, raw_select); // zero high on raw-byte lanes

    if (four_byte_bits) {
        //  4-byte: cp = ((b0&7)<<18)|((b1&0x3F)<<12)|((b2&0x3F)<<6)|(b3&0x3F). high/low mirror haswell's four_high /
        //  four_low; the astral cascade is addressed by the offset plane nibble `plane - 1` (cp - 0x10000).
        uint8x16_t const four_low = vorrq_u8(vandq_u8(vshlq_n_u8(vandq_u8(next2, low_two_bits), 6), vdupq_n_u8(0xC0)),
                                             vandq_u8(next3, low_six_bits));
        uint8x16_t const four_high = vorrq_u8(vandq_u8(vshlq_n_u8(vandq_u8(next1, low_four_bits), 4), vdupq_n_u8(0xF0)),
                                              sz_utf8_srl8_neon_(next2, 2, 0x0F));
        uint8x16_t const four_select = sz_sentence_break_byte_mask_from_bits_neon_(four_byte_bits);
        low = vbslq_u8(four_select, four_low, low);
        high = vbslq_u8(four_select, four_high, high);

        //  Split the 4-byte lanes on their blind plane (cp bits[16..20]) by VALUE, matching serial / haswell / icelake:
        //    plane == 0      → BMP codepoint (cp = (four_high<<8)|four_low); resolved by the BMP cascade (e.g. the
        //                       overlong `F0 80 87 AB` → U+01EB Lower lands here, NOT on the astral path).
        //    plane in [1,16]  → genuine astral (cp in 0x10000..0x10FFFF); routed to the astral cascade.
        //    plane ≥ 17       → cp ≥ 0x110000 (e.g. `F4 A0 ..`, `F5 ..`); neither BMP nor astral, class Other (0).
        //  The astral cascade only consumes the low nibble of `plane - 1`, so a plane ≥ 17 lane would alias a valid
        //  offset and MUST be excluded, not just left to the cascade.
        uint8x16_t const plane = vorrq_u8(vandq_u8(vshlq_n_u8(vandq_u8(raw, vdupq_n_u8(0x07)), 2), vdupq_n_u8(0x1C)),
                                          sz_utf8_srl8_neon_(next1, 4, 0x03));
        uint8x16_t const plane_nonzero = vandq_u8(four_select, vcgtq_u8(plane, vdupq_n_u8(0)));
        uint8x16_t const plane_le_16 = vcleq_u8(plane, vdupq_n_u8(0x10)); // plane <= 16
        uint8x16_t const is_astral = vandq_u8(plane_nonzero, plane_le_16);
        uint8x16_t const is_overrange = vbicq_u8(plane_nonzero, plane_le_16); // plane != 0 && plane > 16
        uint8x16_t const plane_off = vsubq_u8(vandq_u8(is_astral, plane), vdupq_n_u8(1));
        uint8x16_t const bmp = sz_utf8_sentence_break_bmp_class_neon_(high, low);
        uint8x16_t const astral = sz_utf8_sentence_break_astral_class_neon_(plane_off, high, low);
        uint8x16_t const classed = vbslq_u8(is_astral, astral, bmp); // BMP for plane 0, astral for [1,16]
        return vbicq_u8(classed, is_overrange);                      // force plane >= 17 lanes to Other (0)
    }
    return sz_utf8_sentence_break_bmp_class_neon_(high, low);
}

#pragma endregion In register vectorized classifier

#pragma region Dense compaction and scatter

/** @brief  Build the per-class membership frame from the dense class byte stream with NEON compares: each class is one
 *          `vceqq_u8` per quarter OR-combined to a u64, the NEON twin of @ref sz_utf8_sentence_break_frame_haswell_ (no
 *          scalar pass). The dense stream is at most 64 lanes, held as four `uint8x16_t` quarters. */
SZ_HELPER_AUTO sz_utf8_sentence_break_frame_t sz_utf8_sentence_break_frame_neon_(sz_u8_t const *dense_classes,
                                                                                 sz_u64_t valid) {
    uint8x16_t dense[4];
    dense[0] = vld1q_u8(dense_classes + 0);
    dense[1] = vld1q_u8(dense_classes + 16);
    dense[2] = vld1q_u8(dense_classes + 32);
    dense[3] = vld1q_u8(dense_classes + 48);
    sz_utf8_sentence_break_frame_t frame;
    for (int cls = 0; cls < 15; ++cls) {
        uint8x16_t const value = vdupq_n_u8((sz_u8_t)cls);
        frame.by_class[cls] = sz_utf8_mask_combine_neon_(vceqq_u8(dense[0], value), vceqq_u8(dense[1], value),
                                                         vceqq_u8(dense[2], value), vceqq_u8(dense[3], value)) &
                              valid;
    }
    return frame;
}

/** @brief  Run the portable rule engine over a dense class stream, building the frame with NEON compares first. */
SZ_HELPER_INLINE sz_utf8_sentence_break_window_t sz_utf8_sentence_break_decide_dense_neon_( //
    sz_u8_t const *dense_classes, sz_size_t count, sz_utf8_sentence_break_carry_t *carry, sz_bool_t more_text) {
    sz_u64_t const valid = (count >= 64) ? ~0ull : ((1ull << count) - 1);
    sz_utf8_sentence_break_frame_t const frame = sz_utf8_sentence_break_frame_neon_(dense_classes, valid);
    return sz_utf8_sentence_break_decide_block_(&frame, dense_classes, count, carry, more_text);
}

/** @brief  Largest byte prefix of the window whose codepoints are all fully loaded — the NEON twin of
 *          @ref sz_utf8_sentence_break_complete_limit_haswell_ over the NEON window struct. Never below 1 when the
 *          window is non-empty. */
SZ_HELPER_AUTO sz_size_t sz_utf8_sentence_break_complete_limit_neon_(sz_utf8_rune_window_neon_t window,
                                                                     sz_u8_t const *bytes_after, sz_bool_t more_text) {
    sz_size_t const loaded = window.loaded;
    if (!more_text) return loaded;
    sz_u64_t const valid = sz_u64_mask_until_serial_(loaded);
    sz_u64_t const start_bytes = window.codepoint_starts & valid;
    if (!start_bytes) return loaded;
    sz_u64_t const two = window.two_byte_starts & valid;
    sz_u64_t const three = window.three_byte_starts & valid;
    sz_u64_t const four = window.four_byte_starts & valid;
    sz_u64_t const straddle = ((two & ~sz_u64_mask_until_serial_(loaded > 1 ? loaded - 1 : 0)) |
                               (three & ~sz_u64_mask_until_serial_(loaded > 2 ? loaded - 2 : 0)) |
                               (four & ~sz_u64_mask_until_serial_(loaded > 3 ? loaded - 3 : 0))) &
                              valid;
    sz_size_t limit = straddle ? (sz_size_t)sz_u64_ctz(straddle) : loaded;
    if ((bytes_after[0] & 0xC0) == 0x80) {
        sz_size_t const last_lead = (sz_size_t)(63 - sz_u64_clz(start_bytes));
        if (last_lead < limit) limit = last_lead;
    }
    return limit > 0 ? limit : loaded;
}

#pragma endregion Dense compaction and scatter

#pragma region Forward driver

/**
 *  @brief  Forward UAX-29 sentence segmentation kernel (NEON AArch64). Bit-exact with `sz_utf8_sentences_serial`,
 *          `sz_utf8_sentences_haswell`, and `sz_utf8_sentences_icelake`: a NEON window/classify/dense-compaction
 *          front-end feeds the shared portable rule engine @ref sz_utf8_sentence_break_decide_block_, whose dense
 *          breaks are scattered back to byte lanes.
 */
SZ_API_COMPTIME sz_size_t sz_utf8_sentences_neon(            //
    sz_cptr_t text, sz_size_t length,                        //
    sz_size_t *sentence_starts, sz_size_t *sentence_lengths, //
    sz_size_t sentences_capacity, sz_size_t *bytes_consumed) {

    sz_size_t sentences = 0;
    if (length == 0 || sentences_capacity == 0) {
        if (bytes_consumed) *bytes_consumed = 0;
        return 0;
    }

    sz_u8_t const *text_u8 = (sz_u8_t const *)text;
    sz_size_t sentence_start = 0;
    sz_size_t position = 0;

    sz_utf8_sentence_break_carry_t carry;
    carry.have_prev = 0;
    carry.prev_raw = carry.prev_eff = carry.prev_prev_eff = (sz_u8_t)sz_sentence_break_other_k;
    carry.in_shadow = carry.shadow_aterm = carry.shadow_saw_sp = 0;
    carry.sb8_pending = 0;

    sz_size_t sb8_pending_position = 0;
    int sb8_pending_active = 0;

    while (position < length) {
        sz_utf8_rune_window_neon_t const window = sz_utf8_rune_decode_window_neon_(text_u8 + position,
                                                                                   length - position);
        sz_size_t const loaded = window.loaded;
        sz_u64_t const lead_continuation = (position == 0) ? 1ull : 0ull;
        sz_u64_t const start_bytes = window.codepoint_starts | lead_continuation;
        sz_bool_t const more_text = (position + loaded < length) ? sz_true_k : sz_false_k;

        //  Forward neighbours for the 2-/3-/4-byte codepoint reconstruction inside the classifier. The NEON neighbour
        //  shuffles wrap mod-64, so for a multi-byte lead whose continuations straddle the loaded edge (the truncated
        //  trailing lead at end-of-input, where `complete_limit` does not trim because `!more_text`) `next1`/`next2`/
        //  `next3` would otherwise alias bytes from the window start. Zero the lanes at or past `loaded - k`, matching
        //  haswell's `keep1`/`keep2`/`keep3` and icelake's `maskz_permutexvar`, so the missing continuations read as
        //  zero exactly like serial's blind decode (`text[start+k]` past the input reads 0).
        uint8x16_t next1[4], next2[4], next3[4];
        sz_utf8_forward_neighbours_neon_(window.window, next1, next2, next3);
        sz_u64_t const keep1 = sz_u64_mask_until_serial_(loaded >= 1 ? loaded - 1 : 0);
        sz_u64_t const keep2 = sz_u64_mask_until_serial_(loaded >= 2 ? loaded - 2 : 0);
        sz_u64_t const keep3 = sz_u64_mask_until_serial_(loaded >= 3 ? loaded - 3 : 0);
        for (int quarter = 0; quarter < 4; ++quarter) {
            int const lane_base = quarter * 16;
            next1[quarter] = vandq_u8(next1[quarter], sz_sentence_break_byte_mask_from_bits_neon_(keep1 >> lane_base));
            next2[quarter] = vandq_u8(next2[quarter], sz_sentence_break_byte_mask_from_bits_neon_(keep2 >> lane_base));
            next3[quarter] = vandq_u8(next3[quarter], sz_sentence_break_byte_mask_from_bits_neon_(keep3 >> lane_base));
        }

        //  Classify every byte-lane to a Sentence_Break class byte, one quarter at a time, and store to a 64-byte
        //  scratch. The classifier re-seats the raw-byte (ASCII / continuation / `≥ 0xF8`) and 4-byte codepoints from
        //  the raw window bytes itself, so no per-quarter ASCII mask needs to be threaded in. The BMP (2-/3-byte)
        //  high/low are re-derived from the edge-masked neighbours (the decode window built them from its own wrapping
        //  neighbours); the per-quarter mask slices come from the low 16 bits of each quarter's offset.
        sz_u8_t class_bytes[64];
        for (int quarter = 0; quarter < 4; ++quarter) {
            int const lane_base = quarter * 16;
            uint8x16_t high_q, low_q;
            sz_utf8_sentence_break_bmp_highlow_neon_(window.window[quarter], next1[quarter], next2[quarter],
                                                     window.two_byte_starts >> lane_base,
                                                     window.three_byte_starts >> lane_base, &high_q, &low_q);
            uint8x16_t const classes_q = sz_utf8_sentence_break_classify_quarter_neon_(
                high_q, low_q, window.window[quarter], next1[quarter], next2[quarter], next3[quarter],
                window.four_byte_starts >> lane_base);
            vst1q_u8(class_bytes + lane_base, classes_q);
        }

        sz_size_t const complete_limit = sz_utf8_sentence_break_complete_limit_neon_(
            window, text_u8 + position + loaded, more_text);

        //  Dense compaction: pack the start-lane classes into a dense `0..count-1` array via software-`pext` lane
        //  shuffles (NEON has no `vpcompressb`).
        sz_u64_t const complete_mask = sz_u64_mask_until_serial_(complete_limit);
        sz_u64_t const dense_start_lanes = start_bytes & complete_mask;
        sz_size_t const dense_count = (sz_size_t)sz_u64_popcount(dense_start_lanes);
        sz_u8_t dense_classes[64];
        {
            sz_u64_t remaining = dense_start_lanes;
            sz_size_t dense_index = 0;
            while (remaining) {
                sz_size_t const lane = (sz_size_t)sz_u64_ctz(remaining);
                remaining &= remaining - 1; // clear the lowest set bit
                dense_classes[dense_index++] = class_bytes[lane];
            }
        }

        sz_utf8_sentence_break_carry_t carry_full = carry;
        sz_utf8_sentence_break_window_t const win = sz_utf8_sentence_break_decide_dense_neon_(
            dense_classes, dense_count, &carry_full, more_text);

        //  Resolve a previously deferred SB8 boundary before any of this window's boundaries.
        if (sb8_pending_active && win.sb8_resolution != 0) {
            if (win.sb8_resolution == 1) {
                if (sentences == sentences_capacity) {
                    if (bytes_consumed) *bytes_consumed = sentence_start;
                    return sentences;
                }
                sentence_starts[sentences] = sentence_start;
                sentence_lengths[sentences] = sb8_pending_position - sentence_start;
                ++sentences;
                sentence_start = sb8_pending_position;
            }
            sb8_pending_active = 0;
        }

        //  Scatter the trusted dense breaks back to byte lanes (software-`pdep` into the dense start lanes).
        sz_size_t const dense_adv = win.resolved;
        sz_u64_t const dense_breaks = win.breaks & sz_u64_mask_until_serial_(dense_adv);
        sz_u64_t boundary_lanes = sz_sentence_break_pdep_neon_(dense_breaks, dense_start_lanes);
        if (!carry.have_prev) boundary_lanes &= ~1ull;

        sz_size_t byte_adv;
        if (dense_adv >= dense_count) { byte_adv = complete_limit ? complete_limit : loaded; }
        else {
            sz_u64_t const upto = sz_sentence_break_pdep_neon_((1ull << dense_adv), dense_start_lanes);
            byte_adv = (sz_size_t)sz_u64_ctz(upto);
        }

        sentences = sz_utf8_rune_drain_forward_neon_(boundary_lanes, position, sentence_starts, sentence_lengths,
                                                     sentences, sentences_capacity, &sentence_start);
        if (sentences == sentences_capacity) {
            if (bytes_consumed) *bytes_consumed = sentence_start;
            return sentences;
        }

        if (dense_adv >= dense_count) {
            carry = carry_full;
            position += byte_adv;
        }
        else if (dense_adv > 0) {
            sz_utf8_sentence_break_carry_t carry_to_edge = carry;
            sz_utf8_sentence_break_decide_dense_neon_(dense_classes, dense_adv, &carry_to_edge, sz_true_k);
            carry = carry_to_edge;
            position += byte_adv;
        }
        else {
            if (!sb8_pending_active) {
                sb8_pending_active = 1;
                sb8_pending_position = position;
            }
            carry = carry_full;
            carry.sb8_pending = 1;
            position += complete_limit ? complete_limit : loaded;
        }
    }

    if (sb8_pending_active) {
        if (sentences == sentences_capacity) {
            if (bytes_consumed) *bytes_consumed = sentence_start;
            return sentences;
        }
        sentence_starts[sentences] = sentence_start;
        sentence_lengths[sentences] = sb8_pending_position - sentence_start;
        ++sentences;
        sentence_start = sb8_pending_position;
    }

    if (sentences == sentences_capacity) {
        if (bytes_consumed) *bytes_consumed = sentence_start;
        return sentences;
    }
    sentence_starts[sentences] = sentence_start;
    sentence_lengths[sentences] = length - sentence_start;
    ++sentences;
    if (bytes_consumed) *bytes_consumed = length;
    return sentences;
}

#pragma endregion Forward driver

#pragma endregion UAX 29 Sentence Boundaries forward kernel
#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#endif // SZ_USE_NEON

#ifdef __cplusplus
}
#endif

#endif // STRINGZILLA_UTF8_SENTENCES_NEON_H_
