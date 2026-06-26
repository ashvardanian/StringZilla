/**
 *  @brief Haswell (AVX2) backend for UAX-29 sentence boundaries.
 *  @file include/stringzilla/utf8_sentences/haswell.h
 *  @author Ash Vardanian
 */
#ifndef STRINGZILLA_UTF8_SENTENCES_HASWELL_H_
#define STRINGZILLA_UTF8_SENTENCES_HASWELL_H_

#include "stringzilla/types.h"
#include "stringzilla/utf8_sentences/tables.h"
#include "stringzilla/utf8_sentences/serial.h"
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

#pragma region UAX 29 Sentence Boundaries forward kernel

#pragma region In register vectorized classifier

/*  The AVX2 twin of the Ice Lake Sentence_Break classifier: a contiguous run of codepoints resolves to per-codepoint
 *  Sentence_Break class bytes with ZERO per-lane scalar loop, ZERO `vpgather`, and NO serial deferral. Each 64-byte
 *  window lives as two `__m256i` halves; the per-lane codepoint (high, low) byte pair from the decoded window feeds a
 *  register-resident 3-stage `vpshufb` nibble cascade over the whole BMP (the AVX2 twin of the icelake page-LUT + trie
 *  + OLetter-range mix), and a 4-stage astral cascade for 4-byte leads. Both emit the Sentence_Break class byte
 *  directly, bit-identical with `sz_rune_sentence_break_property` over the entire code space. */

/** @brief  Sentence_Break class byte for thirty-two BMP codepoints (per-lane high = cp>>8, low = cp&0xFF) via a
 *          register-resident 3-stage `vpshufb` nibble cascade. Gather-free; bit-exact with
 *          `sz_rune_sentence_break_property` over the whole BMP. */
SZ_INTERNAL __m256i sz_utf8_sentence_break_bmp_class_haswell_(__m256i high, __m256i low) {
    __m256i const low_nibble_mask = _mm256_set1_epi8(0x0F);
    __m256i const high_high = _mm256_and_si256(_mm256_srli_epi16(high, 4), low_nibble_mask);
    __m256i const high_low = _mm256_and_si256(high, low_nibble_mask);
    __m256i const page = sz_utf8_rune_cascade_stage_haswell_(sz_utf8_sentence_break_haswell_stage1_,
                                                             sz_utf8_sentence_break_haswell_stage1_count_k / 16,
                                                             high_high, high_low);
    __m256i const low_high = _mm256_and_si256(_mm256_srli_epi16(low, 4), low_nibble_mask);
    __m256i const leaf_lo = sz_utf8_rune_cascade_stage_haswell_(sz_utf8_sentence_break_haswell_stage2_lo_,
                                                                sz_utf8_sentence_break_haswell_stage2_lo_count_k / 16,
                                                                page, low_high);
    __m256i const leaf_hi = sz_utf8_rune_cascade_stage_haswell_(sz_utf8_sentence_break_haswell_stage2_hi_,
                                                                sz_utf8_sentence_break_haswell_stage2_hi_count_k / 16,
                                                                page, low_high);
    __m256i const leaf_group = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(leaf_lo, 4), low_nibble_mask),
                                               _mm256_slli_epi16(leaf_hi, 4));
    __m256i const leaf_low_nibble = _mm256_and_si256(leaf_lo, low_nibble_mask);
    __m256i const low_low = _mm256_and_si256(low, low_nibble_mask);
    __m256i const lut_index = _mm256_or_si256(_mm256_slli_epi16(leaf_low_nibble, 4), low_low);
    __m256i result = _mm256_setzero_si256();
    for (int group = 0; group < (int)sz_utf8_sentence_break_haswell_leaf_groups_k; ++group) {
        __m256i const value = sz_utf8_rune_lut256_haswell_(sz_utf8_sentence_break_haswell_stage3_groups_ + group * 256,
                                                           lut_index);
        __m256i const here = _mm256_cmpeq_epi8(leaf_group, _mm256_set1_epi8((char)group));
        result = _mm256_blendv_epi8(result, value, here);
    }
    return result;
}

/** @brief  Sentence_Break class byte for thirty-two ASTRAL codepoints over the 20-bit offset = cp - 0x10000 (5-nibble
 *          cascade). Per-lane bytes: @p plane = (offset>>16)&0xFF (low nibble meaningful), @p high = (offset>>8)&0xFF,
 *          @p low = offset&0xFF. Gather-free; bit-exact with `sz_rune_sentence_break_property` over all astral. */
SZ_INTERNAL __m256i sz_utf8_sentence_break_astral_class_haswell_(__m256i plane, __m256i high, __m256i low) {
    __m256i const low_nibble_mask = _mm256_set1_epi8(0x0F);
    __m256i const n4 = _mm256_and_si256(plane, low_nibble_mask);
    __m256i const n3 = _mm256_and_si256(_mm256_srli_epi16(high, 4), low_nibble_mask);
    __m256i const stage1_index = _mm256_or_si256(_mm256_slli_epi16(n4, 4), n3);
    __m256i const page = sz_utf8_rune_lut256_haswell_(sz_utf8_sentence_break_haswell_astral_stage1_, stage1_index);
    __m256i const n2 = _mm256_and_si256(high, low_nibble_mask);
    __m256i const leaf2_lo = sz_utf8_rune_cascade_stage_haswell_(
        sz_utf8_sentence_break_haswell_astral_stage2_lo_, sz_utf8_sentence_break_haswell_astral_stage2_lo_count_k / 16,
        page, n2);
    __m256i const n1 = _mm256_and_si256(_mm256_srli_epi16(low, 4), low_nibble_mask);
    __m256i const leaf_lo = sz_utf8_rune_cascade_stage_haswell_(
        sz_utf8_sentence_break_haswell_astral_stage3_lo_, sz_utf8_sentence_break_haswell_astral_stage3_lo_count_k / 16,
        leaf2_lo, n1);
    __m256i const leaf_hi = sz_utf8_rune_cascade_stage_haswell_(
        sz_utf8_sentence_break_haswell_astral_stage3_hi_, sz_utf8_sentence_break_haswell_astral_stage3_hi_count_k / 16,
        leaf2_lo, n1);
    __m256i const n0 = _mm256_and_si256(low, low_nibble_mask);
    __m256i const leaf_group = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(leaf_lo, 4), low_nibble_mask),
                                               _mm256_slli_epi16(leaf_hi, 4));
    __m256i const leaf_low_nibble = _mm256_and_si256(leaf_lo, low_nibble_mask);
    __m256i const stage4_lut_index = _mm256_or_si256(_mm256_slli_epi16(leaf_low_nibble, 4), n0);
    __m256i result = _mm256_setzero_si256();
    for (int group = 0; group < (int)sz_utf8_sentence_break_haswell_astral_leaf_groups_k; ++group) {
        __m256i const value = sz_utf8_rune_lut256_haswell_(
            sz_utf8_sentence_break_haswell_astral_stage4_groups_ + group * 256, stage4_lut_index);
        __m256i const here = _mm256_cmpeq_epi8(leaf_group, _mm256_set1_epi8((char)group));
        result = _mm256_blendv_epi8(result, value, here);
    }
    return result;
}

/** @brief  Per-byte-lane Sentence_Break class for one decoded window half, fully in-register and zero-scalar - the
 *          AVX2 twin of @ref sz_utf8_sentence_break_classify_window_icelake_. The decoded window only carries the
 *          2-/3-byte (high, low) reconstruction; this leaf rebuilds the ASCII (`low = raw`, `high = 0`) and 4-byte
 *          (`high`/`low` from the four-byte formula) codepoint bytes before the cascade, exactly as the icelake driver
 *          reconstructs them. BMP lanes go through the BMP cascade; 4-byte lanes are routed by reconstructed plane
 *          through the astral cascade. The class on non-codepoint-start lanes is irrelevant (the dense compaction only
 *          reads start lanes), so those lanes are never selected. */
SZ_INTERNAL __m256i sz_utf8_sentence_break_classify_half_haswell_( //
    __m256i window_high, __m256i window_low, __m256i raw, __m256i next1, __m256i next2, __m256i next3,
    sz_u32_t four_byte_bits) {
    __m256i const low_two_bits = _mm256_set1_epi8(0x03);
    __m256i const low_four_bits = _mm256_set1_epi8(0x0F);
    __m256i const low_six_bits = _mm256_set1_epi8(0x3F);

    //  Raw-byte reconstruction (codepoint == raw byte: low = raw, high = 0) for every lane that is NOT a 2-/3-/4-byte
    //  lead: ASCII (`raw < 0x80`), continuation bytes `0x80..0xBF` (a lone continuation forced to a start at position 0
    //  decodes to its raw value, e.g. `0x85` -> U+0085 NEL), and the non-lead bytes `0xF8..0xFF`. The decode window
    //  pre-folds the 2-byte arithmetic into high/low on ALL lanes, so these must be overwritten — serial's blind
    //  decode gives `rune = lead` for every such byte, matching icelake whose driver seats `low = raw`, `high = 0`
    //  on every non-2/3/4-byte lane. The 2-/3-/4-byte leads are exactly `0xC0..0xF7`, so raw treatment is the
    //  complement `raw < 0xC0 || raw >= 0xF8`.
    __m256i const below_c0 = _mm256_cmpeq_epi8(_mm256_min_epu8(raw, _mm256_set1_epi8((char)0xC0)), raw);
    __m256i const at_least_f8 = _mm256_cmpeq_epi8(_mm256_max_epu8(raw, _mm256_set1_epi8((char)0xF8)), raw);
    __m256i const raw_select = _mm256_andnot_si256(_mm256_cmpeq_epi8(raw, _mm256_set1_epi8((char)0xC0)),
                                                   _mm256_or_si256(below_c0, at_least_f8));
    __m256i low = _mm256_blendv_epi8(window_low, raw, raw_select);
    __m256i high = _mm256_andnot_si256(raw_select, window_high);

    if (four_byte_bits) {
        //  4-byte lead: reconstruct the blind codepoint's low-16 bits cp = (mid<<8)|alo with mid = ((b1&0xF)<<4) |
        //  ((b2>>2)&0xF), alo = ((b2&0x3)<<6) | (b3&0x3F) — the cp's high/low bytes, NOT the offset domain (so a
        //  4-byte lead whose blind plane is 0, e.g. the overlong `F0 80 8D A9` -> U+0369, lands on the BMP path with
        //  the right value). Mirrors icelake's `four_high`/`four_low` exactly.
        __m256i const four_low = _mm256_or_si256(
            _mm256_slli_epi16(_mm256_and_si256(next2, low_two_bits), 6), _mm256_and_si256(next3, low_six_bits));
        __m256i const four_high = _mm256_or_si256(
            _mm256_slli_epi16(_mm256_and_si256(next1, low_four_bits), 4), sz_utf8_srl8_haswell_(next2, 2, 0x0F));
        __m256i const four_select = sz_utf8_byte_mask_from_bits_haswell_(four_byte_bits);
        low = _mm256_blendv_epi8(low, four_low, four_select);
        high = _mm256_blendv_epi8(high, four_high, four_select);

        //  Split the 4-byte lanes on their blind plane (cp bits[16..20]) by VALUE, matching serial/icelake:
        //    plane == 0      -> BMP codepoint (cp = (mid<<8)|alo); resolved by the BMP cascade above.
        //    plane in [1,16]  -> genuine astral (cp in 0x10000..0x10FFFF); routed to the astral cascade.
        //    plane >= 17      -> cp >= 0x110000 (e.g. `F4 A0 ..`, `F5 ..`); neither BMP nor astral, class Other (0).
        //  The astral cascade is addressed by the OFFSET plane nibble `plane - 1` (cp - 0x10000) and only its low
        //  nibble is consumed, so a plane >= 17 lane would alias a valid offset — it MUST be excluded, not just
        //  left to the cascade. `four_high`/`four_low` already carry the offset's low 16 bits (== cp's low 16 bits).
        __m256i const plane = _mm256_or_si256(
            _mm256_slli_epi16(_mm256_and_si256(raw, _mm256_set1_epi8(0x07)), 2), sz_utf8_srl8_haswell_(next1, 4, 0x03));
        __m256i const plane_nonzero = _mm256_andnot_si256(_mm256_cmpeq_epi8(plane, _mm256_setzero_si256()), four_select);
        //  plane <= 16 via the unsigned `max(16, plane) == 16` test (AVX2 has no unsigned byte compare).
        __m256i const plane_le_16 = _mm256_cmpeq_epi8(_mm256_max_epu8(plane, _mm256_set1_epi8(0x10)),
                                                      _mm256_set1_epi8(0x10));
        __m256i const is_astral_lanes = _mm256_and_si256(plane_nonzero, plane_le_16);
        __m256i const is_overrange_lanes = _mm256_andnot_si256(plane_le_16, plane_nonzero);
        __m256i const plane_off = _mm256_sub_epi8(_mm256_and_si256(is_astral_lanes, plane), _mm256_set1_epi8(1));
        __m256i const bmp = sz_utf8_sentence_break_bmp_class_haswell_(high, low);
        __m256i const astral = sz_utf8_sentence_break_astral_class_haswell_(plane_off, high, low);
        //  BMP for plane 0, astral for plane in [1,16], then force plane >= 17 lanes to Other (0).
        __m256i const classed = _mm256_blendv_epi8(bmp, astral, is_astral_lanes);
        return _mm256_andnot_si256(is_overrange_lanes, classed);
    }
    return sz_utf8_sentence_break_bmp_class_haswell_(high, low);
}

/** @brief  Reconstruct the BMP (2-/3-byte) codepoint high/low bytes for one 32-lane half from edge-masked forward
 *          neighbours, matching the rune-window decode's arithmetic but with `next1`/`next2` already zeroed past the
 *          loaded edge. Lanes that are neither a 2- nor a 3-byte lead keep `low = raw`, `high = 0` (the classifier
 *          re-seats raw / 4-byte lanes anyway). Used by the sentence driver so a truncated trailing multi-byte lead
 *          reads its missing continuations as zero, exactly like serial / icelake (no mod-64 wrap aliasing). */
SZ_INTERNAL void sz_utf8_sentence_break_bmp_highlow_haswell_( //
    __m256i raw, __m256i next1, __m256i next2, sz_u32_t two_byte_bits, sz_u32_t three_byte_bits, __m256i *out_high,
    __m256i *out_low) {
    __m256i const low_two_bits = _mm256_set1_epi8(0x03);
    __m256i const low_four_bits = _mm256_set1_epi8(0x0F);
    __m256i const low_five_bits = _mm256_set1_epi8(0x1F);
    __m256i const low_six_bits = _mm256_set1_epi8(0x3F);
    //  2-byte: high = (b0 & 0x1F) >> 2, low = ((b0 & 0x03) << 6) | (next1 & 0x3F).
    __m256i const two_high = sz_utf8_srl8_haswell_(_mm256_and_si256(raw, low_five_bits), 2, 0x07);
    __m256i const two_low = _mm256_or_si256(_mm256_slli_epi16(_mm256_and_si256(raw, low_two_bits), 6),
                                            _mm256_and_si256(next1, low_six_bits));
    //  3-byte: high = ((b0 & 0x0F) << 4) | ((next1 >> 2) & 0x0F), low = ((next1 & 0x03) << 6) | (next2 & 0x3F).
    __m256i const three_high = _mm256_or_si256(_mm256_slli_epi16(_mm256_and_si256(raw, low_four_bits), 4),
                                               sz_utf8_srl8_haswell_(next1, 2, 0x0F));
    __m256i const three_low = _mm256_or_si256(_mm256_slli_epi16(_mm256_and_si256(next1, low_two_bits), 6),
                                              _mm256_and_si256(next2, low_six_bits));
    __m256i const two_select = sz_utf8_byte_mask_from_bits_haswell_(two_byte_bits);
    __m256i const three_select = sz_utf8_byte_mask_from_bits_haswell_(three_byte_bits);
    __m256i high = _mm256_setzero_si256();
    __m256i low = raw;
    high = _mm256_blendv_epi8(high, two_high, two_select);
    low = _mm256_blendv_epi8(low, two_low, two_select);
    high = _mm256_blendv_epi8(high, three_high, three_select);
    low = _mm256_blendv_epi8(low, three_low, three_select);
    *out_high = high, *out_low = low;
}

/** @brief  Third forward neighbour `next3[i] = window[i+3]` over all 64 lanes with mod-64 wrap, the AVX2 twin of
 *          icelake's `_mm512_permutexvar_epi8(lane_identity+3)`. Same idiom as the substrate `forward_neighbours_`. */
SZ_INTERNAL void sz_utf8_sentence_break_next3_haswell_(__m256i window_lo, __m256i window_hi, __m256i *next3_lo,
                                                       __m256i *next3_hi) {
    __m256i const low_successor = _mm256_permute2x128_si256(window_lo, window_hi, 0x21);
    *next3_lo = _mm256_alignr_epi8(low_successor, window_lo, 3);
    __m256i const high_successor = _mm256_permute2x128_si256(window_hi, window_lo, 0x21);
    *next3_hi = _mm256_alignr_epi8(high_successor, window_hi, 3);
}

#pragma endregion In register vectorized classifier

#pragma region Dense compaction and scatter

/** @brief  Build the per-class membership frame from the dense class byte stream with AVX2 compares: each class is one
 *          `vpcmpeqb` per 32-lane half OR-combined to a u64, the AVX2 twin of the icelake fifteen-`vpcmpeqb` build (no
 *          scalar pass). The dense stream is at most 64 lanes, held as two `__m256i`. */
SZ_INTERNAL sz_utf8_sentence_break_frame_t sz_utf8_sentence_break_frame_haswell_(sz_u8_t const *dense_classes,
                                                                                 sz_u64_t valid) {
    __m256i const dense_lo = _mm256_loadu_si256((__m256i const *)(dense_classes + 0));
    __m256i const dense_hi = _mm256_loadu_si256((__m256i const *)(dense_classes + 32));
    sz_utf8_sentence_break_frame_t frame;
    for (int cls = 0; cls < 15; ++cls) {
        __m256i const value = _mm256_set1_epi8((char)cls);
        frame.by_class[cls] = sz_utf8_mask_combine_haswell_(_mm256_cmpeq_epi8(dense_lo, value),
                                                            _mm256_cmpeq_epi8(dense_hi, value)) &
                              valid;
    }
    return frame;
}

/** @brief  Run the portable rule engine over a dense class stream, building the frame with AVX2 compares first. */
SZ_FORCE_INLINE sz_utf8_sentence_break_window_t sz_utf8_sentence_break_decide_dense_haswell_( //
    sz_u8_t const *dense_classes, sz_size_t count, sz_utf8_sentence_break_carry_t *carry, sz_bool_t more_text) {
    sz_u64_t const valid = (count >= 64) ? ~0ull : ((1ull << count) - 1);
    sz_utf8_sentence_break_frame_t const frame = sz_utf8_sentence_break_frame_haswell_(dense_classes, valid);
    return sz_utf8_sentence_break_decide_block_(&frame, dense_classes, count, carry, more_text);
}

/** @brief  Largest byte prefix of the window whose codepoints are all fully loaded — the AVX2 twin of the icelake
 *          driver's effective-window<64 trim. Never below 1 when the window is non-empty. */
SZ_INTERNAL sz_size_t sz_utf8_sentence_break_complete_limit_haswell_(sz_utf8_rune_window_haswell_t window,
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
 *  @brief  Forward UAX-29 sentence segmentation kernel (Haswell AVX2). Bit-exact with `sz_utf8_sentences_serial` and
 *          `sz_utf8_sentences_icelake`: an AVX2 window/classify/dense-compaction front-end feeds the shared portable
 *          rule engine @ref sz_utf8_sentence_break_decide_block_, whose dense breaks are scattered back to byte lanes.
 */
SZ_PUBLIC sz_size_t sz_utf8_sentences_haswell(               //
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
        sz_utf8_rune_window_haswell_t const window = sz_utf8_rune_decode_window_haswell_(text_u8 + position,
                                                                                         length - position);
        sz_size_t const loaded = window.loaded;
        sz_u64_t const lead_continuation = (position == 0) ? 1ull : 0ull;
        sz_u64_t const start_bytes = window.codepoint_starts | lead_continuation;
        sz_bool_t const more_text = (position + loaded < length) ? sz_true_k : sz_false_k;

        //  Forward neighbours for the ASCII / 4-byte codepoint reconstruction inside the classifier. The AVX2
        //  neighbour gathers wrap mod-64, so for a multi-byte lead whose continuations straddle the loaded edge
        //  (the truncated trailing lead at end-of-input, where `complete_limit` does not trim because `!more_text`)
        //  `next2`/`next3` would otherwise alias bytes from the window start. Zero the lanes at or past `loaded - k`,
        //  matching icelake's `maskz_permutexvar` `keep1`/`keep2`/`keep3`, so the missing continuations read as zero
        //  exactly like serial's blind decode (`text[start+k]` past the input reads 0).
        __m256i next1_lo, next1_hi, next2_lo, next2_hi, next3_lo, next3_hi;
        sz_utf8_forward_neighbours_haswell_(window.window_lo, window.window_hi, &next1_lo, &next1_hi, &next2_lo,
                                            &next2_hi);
        sz_utf8_sentence_break_next3_haswell_(window.window_lo, window.window_hi, &next3_lo, &next3_hi);
        sz_u64_t const keep1 = sz_u64_mask_until_serial_(loaded >= 1 ? loaded - 1 : 0);
        sz_u64_t const keep2 = sz_u64_mask_until_serial_(loaded >= 2 ? loaded - 2 : 0);
        sz_u64_t const keep3 = sz_u64_mask_until_serial_(loaded >= 3 ? loaded - 3 : 0);
        next1_lo = _mm256_and_si256(next1_lo, sz_utf8_byte_mask_from_bits_haswell_((sz_u32_t)keep1));
        next1_hi = _mm256_and_si256(next1_hi, sz_utf8_byte_mask_from_bits_haswell_((sz_u32_t)(keep1 >> 32)));
        next2_lo = _mm256_and_si256(next2_lo, sz_utf8_byte_mask_from_bits_haswell_((sz_u32_t)keep2));
        next2_hi = _mm256_and_si256(next2_hi, sz_utf8_byte_mask_from_bits_haswell_((sz_u32_t)(keep2 >> 32)));
        next3_lo = _mm256_and_si256(next3_lo, sz_utf8_byte_mask_from_bits_haswell_((sz_u32_t)keep3));
        next3_hi = _mm256_and_si256(next3_hi, sz_utf8_byte_mask_from_bits_haswell_((sz_u32_t)(keep3 >> 32)));

        //  Re-derive the BMP (2-/3-byte) high/low from the edge-masked neighbours. The shared rune-window decode
        //  built `window.{high,low}_{lo,hi}` from its own mod-64-wrapping neighbours, so a 3-byte lead straddling
        //  the loaded edge would read a wrapped byte as its missing continuation (icelake recomputes high/low from
        //  its `keep*`-masked neighbours for exactly this reason; here we patch the decoded pair to match).
        __m256i high_lo, high_hi, low_lo, low_hi;
        sz_utf8_sentence_break_bmp_highlow_haswell_(window.window_lo, next1_lo, next2_lo,
                                                    (sz_u32_t)window.two_byte_starts, (sz_u32_t)window.three_byte_starts,
                                                    &high_lo, &low_lo);
        sz_utf8_sentence_break_bmp_highlow_haswell_(
            window.window_hi, next1_hi, next2_hi, (sz_u32_t)(window.two_byte_starts >> 32),
            (sz_u32_t)(window.three_byte_starts >> 32), &high_hi, &low_hi);

        //  The classifier reconstructs the raw-byte (ASCII / continuation / `>= 0xF8`) and 4-byte codepoints from
        //  the raw window bytes itself, so no per-half ASCII mask needs to be threaded in.
        __m256i const classes_lo = sz_utf8_sentence_break_classify_half_haswell_(
            high_lo, low_lo, window.window_lo, next1_lo, next2_lo, next3_lo, (sz_u32_t)window.four_byte_starts);
        __m256i const classes_hi = sz_utf8_sentence_break_classify_half_haswell_(
            high_hi, low_hi, window.window_hi, next1_hi, next2_hi, next3_hi, (sz_u32_t)(window.four_byte_starts >> 32));

        sz_size_t const complete_limit = sz_utf8_sentence_break_complete_limit_haswell_(
            window, text_u8 + position + loaded, more_text);

        //  Dense compaction: store the per-byte-lane class bytes once, then gather the start-lane classes into a dense
        //  `0..count-1` array via BMI2 lane-index unpacking (no `vpgather`, no `vpcompressb`).
        sz_u8_t class_bytes[64];
        _mm256_storeu_si256((__m256i *)(class_bytes + 0), classes_lo);
        _mm256_storeu_si256((__m256i *)(class_bytes + 32), classes_hi);

        sz_u64_t const complete_mask = sz_u64_mask_until_serial_(complete_limit);
        sz_u64_t const dense_start_lanes = start_bytes & complete_mask;
        sz_size_t const dense_count = (sz_size_t)_mm_popcnt_u64(dense_start_lanes);
        sz_u8_t dense_classes[64];
        {
            sz_u64_t remaining = dense_start_lanes;
            sz_size_t dense_index = 0;
            while (remaining) {
                sz_size_t const lane = (sz_size_t)sz_u64_ctz(remaining);
                remaining = _blsr_u64(remaining);
                dense_classes[dense_index++] = class_bytes[lane];
            }
        }

        sz_utf8_sentence_break_carry_t carry_full = carry;
        sz_utf8_sentence_break_window_t const win = sz_utf8_sentence_break_decide_dense_haswell_(
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

        //  Scatter the trusted dense breaks back to byte lanes (`_pdep_u64` into the dense start lanes).
        sz_size_t const dense_adv = win.resolved;
        sz_u64_t const dense_breaks = win.breaks & sz_u64_mask_until_serial_(dense_adv);
        sz_u64_t boundary_lanes = _pdep_u64(dense_breaks, dense_start_lanes);
        if (!carry.have_prev) boundary_lanes &= ~1ull;

        sz_size_t byte_adv;
        if (dense_adv >= dense_count) { byte_adv = complete_limit ? complete_limit : loaded; }
        else {
            sz_u64_t const upto = _pdep_u64((1ull << dense_adv), dense_start_lanes);
            byte_adv = (sz_size_t)sz_u64_ctz(upto);
        }

        sentences = sz_utf8_rune_drain_forward_haswell_(boundary_lanes, position, sentence_starts, sentence_lengths,
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
            sz_utf8_sentence_break_decide_dense_haswell_(dense_classes, dense_adv, &carry_to_edge, sz_true_k);
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
#endif // SZ_USE_HASWELL

#ifdef __cplusplus
}
#endif

#endif // STRINGZILLA_UTF8_SENTENCES_HASWELL_H_
