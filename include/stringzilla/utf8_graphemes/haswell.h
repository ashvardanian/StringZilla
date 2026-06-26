/**
 *  @file   include/stringzilla/utf8_graphemes/haswell.h
 *  @author Ash Vardanian
 *  @brief  Haswell (AVX2) backend for UAX-29 extended grapheme cluster boundaries, fully vectorized end-to-end.
 *
 *  The AVX2 twin of the Ice Lake grapheme kernel. Each 64-byte window lives as two `__m256i` halves; every lane is
 *  decoded through the shared codepoint substrate (`sz_utf8_rune_decode_window_haswell_`), classified gather-free
 *  into one packed Grapheme_Cluster_Break descriptor per lane by a register-resident `vpshufb` nibble cascade (the AVX2
 *  twin of the VBMI BMP trie + 4-stage astral trie), compacted to a codepoint-dense descriptor byte buffer, and
 *  resolved by the SHARED portable boundary engine (`sz_grapheme_window_boundaries_`). No `vpgatherdd`, no `vpermb` /
 *  `vpcompressb` (AVX2 lacks them), no per-lane scalar rule loop, no serial deferral.
 */
#ifndef STRINGZILLA_UTF8_GRAPHEMES_HASWELL_H_
#define STRINGZILLA_UTF8_GRAPHEMES_HASWELL_H_

#include "stringzilla/types.h"
#include "stringzilla/utf8_graphemes/tables.h"
#include "stringzilla/utf8_graphemes/serial.h"
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

#pragma region Gather free Grapheme_Cluster_Break classifier

/** @brief  Packed descriptor byte for thirty-two BMP codepoints (per-lane high = cp>>8, low = cp&0xFF) via a
 *          register-resident 3-stage `vpshufb` nibble cascade, the AVX2 twin of the VBMI full-BMP trie. The cascade
 *          emits the descriptor directly (the serial `id_to_desc` permute is folded into the leaf tables). Gather-free;
 *          bit-exact with `sz_rune_grapheme_break_property` over the whole BMP (Hangul included, no separate formula). */
SZ_INTERNAL __m256i sz_grapheme_bmp_descriptor_haswell_(__m256i high, __m256i low) {
    __m256i const low_nibble_mask = _mm256_set1_epi8(0x0F);
    __m256i const page = sz_utf8_rune_lut256_haswell_(sz_utf8_grapheme_break_haswell_stage1_, high);
    __m256i const low_high = _mm256_and_si256(_mm256_srli_epi16(low, 4), low_nibble_mask);
    __m256i const leaf_lo = sz_utf8_rune_cascade_stage_haswell_(sz_utf8_grapheme_break_haswell_stage2_lo_,
                                                                sz_utf8_grapheme_break_haswell_stage2_lo_count_k / 16,
                                                                page, low_high);
    __m256i const leaf_hi = sz_utf8_rune_cascade_stage_haswell_(sz_utf8_grapheme_break_haswell_stage2_hi_,
                                                                sz_utf8_grapheme_break_haswell_stage2_hi_count_k / 16,
                                                                page, low_high);
    __m256i const leaf_group = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(leaf_lo, 4), low_nibble_mask),
                                               _mm256_slli_epi16(leaf_hi, 4));
    __m256i const leaf_low_nibble = _mm256_and_si256(leaf_lo, low_nibble_mask);
    __m256i const low_low = _mm256_and_si256(low, low_nibble_mask);
    __m256i const lut_index = _mm256_or_si256(_mm256_slli_epi16(leaf_low_nibble, 4), low_low);
    __m256i result = _mm256_setzero_si256();
    for (int group = 0; group < (int)sz_utf8_grapheme_break_haswell_leaf_groups_k; ++group) {
        __m256i const value = sz_utf8_rune_lut256_haswell_(sz_utf8_grapheme_break_haswell_stage3_groups_ + group * 256,
                                                           lut_index);
        __m256i const here = _mm256_cmpeq_epi8(leaf_group, _mm256_set1_epi8((char)group));
        result = _mm256_blendv_epi8(result, value, here);
    }
    return result;
}

/** @brief  Packed descriptor byte for thirty-two ASTRAL codepoints over offset = cp - 0x10000 (5-nibble cascade), the
 *          AVX2 twin of the icelake astral trie. Per-lane bytes: @p plane = (offset>>16)&0xFF (low nibble meaningful),
 *          @p high = (offset>>8)&0xFF, @p low = offset&0xFF. Gather-free; bit-exact. */
SZ_INTERNAL __m256i sz_grapheme_astral_descriptor_haswell_(__m256i plane, __m256i high, __m256i low) {
    __m256i const low_nibble_mask = _mm256_set1_epi8(0x0F);
    __m256i const n4 = _mm256_and_si256(plane, low_nibble_mask);
    __m256i const n3 = _mm256_and_si256(_mm256_srli_epi16(high, 4), low_nibble_mask);
    __m256i const stage1_index = _mm256_or_si256(_mm256_slli_epi16(n4, 4), n3);
    __m256i const page = sz_utf8_rune_lut256_haswell_(sz_utf8_grapheme_break_haswell_astral_stage1_, stage1_index);
    __m256i const n2 = _mm256_and_si256(high, low_nibble_mask);
    // leaf2 ids fit in a byte (n_leaf2 <= 255), so only the lo plane carries information; the stage3 cascade selects
    // on `leaf2` directly with a tile_count covering every leaf2 id (the hi plane is all-zero and unused).
    __m256i const leaf2 = sz_utf8_rune_cascade_stage_haswell_(
        sz_utf8_grapheme_break_haswell_astral_stage2_lo_, sz_utf8_grapheme_break_haswell_astral_stage2_lo_count_k / 16,
        page, n2);
    __m256i const n1 = _mm256_and_si256(_mm256_srli_epi16(low, 4), low_nibble_mask);
    __m256i const leaf_lo = sz_utf8_rune_cascade_stage_haswell_(
        sz_utf8_grapheme_break_haswell_astral_stage3_lo_, sz_utf8_grapheme_break_haswell_astral_stage3_lo_count_k / 16,
        leaf2, n1);
    __m256i const leaf_hi = sz_utf8_rune_cascade_stage_haswell_(
        sz_utf8_grapheme_break_haswell_astral_stage3_hi_, sz_utf8_grapheme_break_haswell_astral_stage3_hi_count_k / 16,
        leaf2, n1);
    __m256i const n0 = _mm256_and_si256(low, low_nibble_mask);
    __m256i const leaf_group = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(leaf_lo, 4), low_nibble_mask),
                                               _mm256_slli_epi16(leaf_hi, 4));
    __m256i const leaf_low_nibble = _mm256_and_si256(leaf_lo, low_nibble_mask);
    __m256i const stage4_lut_index = _mm256_or_si256(_mm256_slli_epi16(leaf_low_nibble, 4), n0);
    __m256i result = _mm256_setzero_si256();
    for (int group = 0; group < (int)sz_utf8_grapheme_break_haswell_astral_leaf_groups_k; ++group) {
        __m256i const value = sz_utf8_rune_lut256_haswell_(
            sz_utf8_grapheme_break_haswell_astral_stage4_groups_ + group * 256, stage4_lut_index);
        __m256i const here = _mm256_cmpeq_epi8(leaf_group, _mm256_set1_epi8((char)group));
        result = _mm256_blendv_epi8(result, value, here);
    }
    return result;
}

/** @brief  Per-half unsigned `value >= bound` mask (AVX2 lacks unsigned byte compare): `max_epu8(value,bound)==value`. */
SZ_INTERNAL __m256i sz_grapheme_cmpge_epu8_haswell_(__m256i value, __m256i bound) {
    return _mm256_cmpeq_epi8(_mm256_max_epu8(value, bound), value);
}

/** @brief  64-bit unsigned `low <= cp <= high` mask over reconstructed BMP codepoints carried in @p high_byte /
 *          @p low_byte halves. cp = (high<<8)|low, so the inclusive 16-bit range test is: high in (lo_hi,hi_hi)
 *          unconditionally, or on the boundary high bytes the low byte within bound. Two halves, branchless. */
SZ_INTERNAL sz_u64_t sz_grapheme_cp_in_range_haswell_(__m256i high_lo, __m256i low_lo, __m256i high_hi, __m256i low_hi,
                                                      sz_u16_t lo, sz_u16_t hi) {
    sz_u8_t const lo_h = (sz_u8_t)(lo >> 8), lo_l = (sz_u8_t)(lo & 0xFF);
    sz_u8_t const hi_h = (sz_u8_t)(hi >> 8), hi_l = (sz_u8_t)(hi & 0xFF);
    __m256i const lo_h_v = _mm256_set1_epi8((char)lo_h), lo_l_v = _mm256_set1_epi8((char)lo_l);
    __m256i const hi_h_v = _mm256_set1_epi8((char)hi_h), hi_l_v = _mm256_set1_epi8((char)hi_l);
    // ge_low: high>lo_h, or (high==lo_h and low>=lo_l). le_high: high<hi_h, or (high==hi_h and low<=hi_l).
    __m256i const ge_low_lo = _mm256_or_si256(
        _mm256_andnot_si256(_mm256_cmpeq_epi8(high_lo, lo_h_v), sz_grapheme_cmpge_epu8_haswell_(high_lo, lo_h_v)),
        _mm256_and_si256(_mm256_cmpeq_epi8(high_lo, lo_h_v), sz_grapheme_cmpge_epu8_haswell_(low_lo, lo_l_v)));
    __m256i const ge_low_hi = _mm256_or_si256(
        _mm256_andnot_si256(_mm256_cmpeq_epi8(high_hi, lo_h_v), sz_grapheme_cmpge_epu8_haswell_(high_hi, lo_h_v)),
        _mm256_and_si256(_mm256_cmpeq_epi8(high_hi, lo_h_v), sz_grapheme_cmpge_epu8_haswell_(low_hi, lo_l_v)));
    __m256i const le_high_lo = _mm256_or_si256(
        _mm256_andnot_si256(_mm256_cmpeq_epi8(high_lo, hi_h_v), sz_grapheme_cmpge_epu8_haswell_(hi_h_v, high_lo)),
        _mm256_and_si256(_mm256_cmpeq_epi8(high_lo, hi_h_v), sz_grapheme_cmpge_epu8_haswell_(hi_l_v, low_lo)));
    __m256i const le_high_hi = _mm256_or_si256(
        _mm256_andnot_si256(_mm256_cmpeq_epi8(high_hi, hi_h_v), sz_grapheme_cmpge_epu8_haswell_(hi_h_v, high_hi)),
        _mm256_and_si256(_mm256_cmpeq_epi8(high_hi, hi_h_v), sz_grapheme_cmpge_epu8_haswell_(hi_l_v, low_hi)));
    return sz_utf8_mask_combine_haswell_(_mm256_and_si256(ge_low_lo, le_high_lo),
                                         _mm256_and_si256(ge_low_hi, le_high_hi));
}

/** @brief  Lanes whose BMP codepoint resolves uniformly to GCB=Other via the CJK / Kana arithmetic ranges (the AVX2
 *          twin of `sz_grapheme_cjk_other_icelake_`): `[0x3000,0xA66E] | [0xD7FC,0xFB1D]` minus the interior Extend /
 *          enclosed exceptions. Such lanes need no cold cascade (their descriptor is 0). */
SZ_INTERNAL sz_u64_t sz_grapheme_cjk_other_haswell_(__m256i high_lo, __m256i low_lo, __m256i high_hi, __m256i low_hi) {
    sz_u64_t const run_a = sz_grapheme_cp_in_range_haswell_(high_lo, low_lo, high_hi, low_hi, 0x3000, 0xA66E);
    sz_u64_t const run_b = sz_grapheme_cp_in_range_haswell_(high_lo, low_lo, high_hi, low_hi, 0xD7FC, 0xFB1D);
    sz_u64_t const exc_a = sz_grapheme_cp_in_range_haswell_(high_lo, low_lo, high_hi, low_hi, 0x302A, 0x3030);
    sz_u64_t const exc_b = sz_grapheme_cp_in_range_haswell_(high_lo, low_lo, high_hi, low_hi, 0x3099, 0x309A);
    sz_u64_t const exc_c = sz_grapheme_cp_in_range_haswell_(high_lo, low_lo, high_hi, low_hi, 0x303D, 0x303D);
    sz_u64_t const exc_d = sz_grapheme_cp_in_range_haswell_(high_lo, low_lo, high_hi, low_hi, 0x3297, 0x3297);
    sz_u64_t const exc_e = sz_grapheme_cp_in_range_haswell_(high_lo, low_lo, high_hi, low_hi, 0x3299, 0x3299);
    return (run_a | run_b) & ~(exc_a | exc_b | exc_c | exc_d | exc_e);
}

/** @brief  Compute `next3[i] = window[i+3]` over all 64 lanes (mod-64 wrap), the AVX2 twin of icelake's
 *          `_mm512_permutexvar_epi8(lane_identity+3)` (same idiom as the substrate forward-neighbours). */
SZ_INTERNAL void sz_grapheme_next3_haswell_(__m256i window_lo, __m256i window_hi, __m256i *next3_lo,
                                            __m256i *next3_hi) {
    __m256i const low_successor = _mm256_permute2x128_si256(window_lo, window_hi, 0x21);
    *next3_lo = _mm256_alignr_epi8(low_successor, window_lo, 3);
    __m256i const high_successor = _mm256_permute2x128_si256(window_hi, window_lo, 0x21);
    *next3_hi = _mm256_alignr_epi8(high_successor, window_hi, 3);
}

/** @brief  Per-window per-lane descriptors as two `__m256i` halves, plus the codepoint-start lane mask and geometry.
 *          The AVX2 twin of the icelake `sz_grapheme_classify_window_full_icelake_` outputs. */
typedef struct sz_grapheme_classified_haswell_t {
    __m256i descriptors_lo;    /**< Packed descriptor per byte-lane, lanes [0,32) (valid only on start lanes). */
    __m256i descriptors_hi;    /**< Packed descriptor per byte-lane, lanes [32,64). */
    sz_u64_t start_lanes;      /**< Codepoint-start lanes within the effective span (trimmed to `byte_span`). */
    sz_size_t codepoint_count; /**< Number of codepoint starts resolved (<= 64). */
    sz_size_t byte_span;       /**< Bytes consumed by the resolved codepoints (offset of the next start). */
} sz_grapheme_classified_haswell_t;

/**
 *  @brief  Decode a 64-byte window, classify every lane to a packed descriptor gather-free, and emit the per-lane
 *          descriptor halves with the trimmed codepoint-start geometry. Mirrors the icelake decode/classify path
 *          (value-based blind reconstruction so malformed input agrees byte-for-byte) without `vpermb`/`vpgather`.
 */
SZ_INTERNAL sz_grapheme_classified_haswell_t sz_grapheme_classify_window_haswell_( //
    sz_u8_t const *text, sz_size_t length, sz_size_t base) {

    sz_utf8_rune_window_haswell_t const decoded = sz_utf8_rune_decode_window_haswell_(text + base, length - base);
    sz_size_t const loaded = decoded.loaded;
    sz_u64_t const loaded_mask = sz_u64_mask_until_serial_(loaded);
    sz_u64_t start_lanes = decoded.codepoint_starts;

    // Effective-window trim: a multi-byte lead near the 64-byte edge whose span runs past `loaded` would decode against
    // a wrapped neighbour; defer it to the next window. Branchless union of overrunning starts, take the lowest.
    sz_size_t byte_span = loaded;
    if (loaded >= 64) {
        sz_u64_t const two = decoded.two_byte_starts;
        sz_u64_t const three = decoded.three_byte_starts;
        sz_u64_t const four = decoded.four_byte_starts;
        sz_u64_t const overrun = (two & ~sz_u64_mask_until_serial_(loaded - 1)) |
                                 (three & ~sz_u64_mask_until_serial_(loaded - 2)) |
                                 (four & ~sz_u64_mask_until_serial_(loaded - 3));
        byte_span = overrun ? (sz_size_t)sz_u64_ctz(overrun) : loaded;
        start_lanes &= sz_u64_mask_until_serial_(byte_span);
    }

    __m256i const raw_lo = decoded.window_lo, raw_hi = decoded.window_hi;
    __m256i next1_lo, next1_hi, next2_lo, next2_hi, next3_lo, next3_hi;
    sz_utf8_forward_neighbours_haswell_(raw_lo, raw_hi, &next1_lo, &next1_hi, &next2_lo, &next2_hi);
    sz_grapheme_next3_haswell_(raw_lo, raw_hi, &next3_lo, &next3_hi);

    // Zero each `next k` lane whose source byte index reaches `loaded` (short final window only), matching the icelake
    // maskz neighbour fetch / serial blind decode that pads out-of-window continuation bytes with 0. On a FULL window
    // (`loaded == 64`) `loaded_mask` is all-ones so this is inert and the only touched lanes (truncated edge leads) are
    // already deferred by the effective-window trim — so skip the six `vpshufb` mask builds entirely on the hot path.
    __m256i next1_lo_p = next1_lo, next1_hi_p = next1_hi, next2_lo_p = next2_lo, next2_hi_p = next2_hi,
            next3_lo_p = next3_lo, next3_hi_p = next3_hi;
    if (loaded < 64) {
        next1_lo_p = _mm256_and_si256(next1_lo, sz_utf8_byte_mask_from_bits_haswell_((sz_u32_t)(loaded_mask >> 1)));
        next1_hi_p = _mm256_and_si256(next1_hi,
                                      sz_utf8_byte_mask_from_bits_haswell_((sz_u32_t)((loaded_mask >> 1) >> 32)));
        next2_lo_p = _mm256_and_si256(next2_lo, sz_utf8_byte_mask_from_bits_haswell_((sz_u32_t)(loaded_mask >> 2)));
        next2_hi_p = _mm256_and_si256(next2_hi,
                                      sz_utf8_byte_mask_from_bits_haswell_((sz_u32_t)((loaded_mask >> 2) >> 32)));
        next3_lo_p = _mm256_and_si256(next3_lo, sz_utf8_byte_mask_from_bits_haswell_((sz_u32_t)(loaded_mask >> 3)));
        next3_hi_p = _mm256_and_si256(next3_hi,
                                      sz_utf8_byte_mask_from_bits_haswell_((sz_u32_t)((loaded_mask >> 3) >> 32)));
    }

    // Reconstruct per-lane high/low (BMP) and plane/mid/low (astral) BLINDLY from the raw lead + zeroed neighbours,
    // mirroring `sz_grapheme_classify_window_icelake_` byte-for-byte. ASCII lanes take identity (high=0, low=raw).
    __m256i const c03 = _mm256_set1_epi8(0x03), c07 = _mm256_set1_epi8(0x07), c0f = _mm256_set1_epi8(0x0F),
                  c1f = _mm256_set1_epi8(0x1F), c3f = _mm256_set1_epi8(0x3F);
    // 2-byte: high = ((lead & 0x1F) >> 2) & 0x07; low = ((lead & 0x03) << 6) | (next1 & 0x3F).
    __m256i const two_high_lo = sz_utf8_srl8_haswell_(_mm256_and_si256(raw_lo, c1f), 2, 0x07);
    __m256i const two_high_hi = sz_utf8_srl8_haswell_(_mm256_and_si256(raw_hi, c1f), 2, 0x07);
    __m256i const two_low_lo = _mm256_or_si256(_mm256_slli_epi16(_mm256_and_si256(raw_lo, c03), 6),
                                               _mm256_and_si256(next1_lo_p, c3f));
    __m256i const two_low_hi = _mm256_or_si256(_mm256_slli_epi16(_mm256_and_si256(raw_hi, c03), 6),
                                               _mm256_and_si256(next1_hi_p, c3f));
    // 3-byte: high = ((lead & 0x0F) << 4) | ((next1 >> 2) & 0x0F); low = ((next1 & 0x03) << 6) | (next2 & 0x3F).
    __m256i const three_high_lo = _mm256_or_si256(_mm256_slli_epi16(_mm256_and_si256(raw_lo, c0f), 4),
                                                  sz_utf8_srl8_haswell_(next1_lo_p, 2, 0x0F));
    __m256i const three_high_hi = _mm256_or_si256(_mm256_slli_epi16(_mm256_and_si256(raw_hi, c0f), 4),
                                                  sz_utf8_srl8_haswell_(next1_hi_p, 2, 0x0F));
    __m256i const three_low_lo = _mm256_or_si256(_mm256_slli_epi16(_mm256_and_si256(next1_lo_p, c03), 6),
                                                 _mm256_and_si256(next2_lo_p, c3f));
    __m256i const three_low_hi = _mm256_or_si256(_mm256_slli_epi16(_mm256_and_si256(next1_hi_p, c03), 6),
                                                 _mm256_and_si256(next2_hi_p, c3f));

    sz_u64_t const ascii = sz_utf8_mask_combine_haswell_(
        _mm256_cmpeq_epi8(_mm256_and_si256(raw_lo, _mm256_set1_epi8((char)0x80)), _mm256_setzero_si256()),
        _mm256_cmpeq_epi8(_mm256_and_si256(raw_hi, _mm256_set1_epi8((char)0x80)), _mm256_setzero_si256()));
    sz_u64_t const three_byte = decoded.three_byte_starts;
    sz_u64_t const four_byte = decoded.four_byte_starts;

    __m256i const three_sel_lo = sz_utf8_byte_mask_from_bits_haswell_((sz_u32_t)three_byte);
    __m256i const three_sel_hi = sz_utf8_byte_mask_from_bits_haswell_((sz_u32_t)(three_byte >> 32));
    __m256i const ascii_sel_lo = sz_utf8_byte_mask_from_bits_haswell_((sz_u32_t)ascii);
    __m256i const ascii_sel_hi = sz_utf8_byte_mask_from_bits_haswell_((sz_u32_t)(ascii >> 32));

    __m256i non_ascii_high_lo = _mm256_blendv_epi8(two_high_lo, three_high_lo, three_sel_lo);
    __m256i non_ascii_high_hi = _mm256_blendv_epi8(two_high_hi, three_high_hi, three_sel_hi);
    __m256i non_ascii_low_lo = _mm256_blendv_epi8(two_low_lo, three_low_lo, three_sel_lo);
    __m256i non_ascii_low_hi = _mm256_blendv_epi8(two_low_hi, three_low_hi, three_sel_hi);
    // high = ascii ? 0 : non_ascii_high; low = ascii ? raw : non_ascii_low.
    __m256i high_lo = _mm256_andnot_si256(ascii_sel_lo, non_ascii_high_lo);
    __m256i high_hi = _mm256_andnot_si256(ascii_sel_hi, non_ascii_high_hi);
    __m256i low_lo = _mm256_blendv_epi8(non_ascii_low_lo, raw_lo, ascii_sel_lo);
    __m256i low_hi = _mm256_blendv_epi8(non_ascii_low_hi, raw_hi, ascii_sel_hi);

    // Astral plane/mid/low (offset domain reconstruction): plane = ((b0&7)<<2)|((next1>>4)&3);
    // mid = ((next1&F)<<4)|((next2>>2)&F); lo = ((next2&3)<<6)|(next3&3F).
    __m256i const plane_lo = _mm256_or_si256(_mm256_slli_epi16(_mm256_and_si256(raw_lo, c07), 2),
                                             sz_utf8_srl8_haswell_(next1_lo_p, 4, 0x03));
    __m256i const plane_hi = _mm256_or_si256(_mm256_slli_epi16(_mm256_and_si256(raw_hi, c07), 2),
                                             sz_utf8_srl8_haswell_(next1_hi_p, 4, 0x03));
    __m256i const mid_lo = _mm256_or_si256(_mm256_slli_epi16(_mm256_and_si256(next1_lo_p, c0f), 4),
                                           sz_utf8_srl8_haswell_(next2_lo_p, 2, 0x0F));
    __m256i const mid_hi = _mm256_or_si256(_mm256_slli_epi16(_mm256_and_si256(next1_hi_p, c0f), 4),
                                           sz_utf8_srl8_haswell_(next2_hi_p, 2, 0x0F));
    __m256i const alo_lo = _mm256_or_si256(_mm256_slli_epi16(_mm256_and_si256(next2_lo_p, c03), 6),
                                           _mm256_and_si256(next3_lo_p, c3f));
    __m256i const alo_hi = _mm256_or_si256(_mm256_slli_epi16(_mm256_and_si256(next2_hi_p, c03), 6),
                                           _mm256_and_si256(next3_hi_p, c3f));

    // A 4-byte lead's blind codepoint is `(plane << 16) | (mid << 8) | alo` (NOT the 2-byte fold the
    // non-ASCII high/low computed above). Override its BMP high/low halves to `mid`/`alo` so an overlong
    // 4-byte lead whose blind plane is 0 (e.g. `F0 80 8D A9` -> U+0369) resolves on the BMP path exactly as
    // serial/icelake do — those backends carry the full 21-bit value and split BMP-vs-astral on `cp < 0x10000`,
    // so a plane-0 4-byte lane is a BMP codepoint, not an astral one. (The astral overwrite below is then
    // gated on a non-zero in-range plane, mirroring icelake's `is_astral`.)
    {
        __m256i const four_sel_lo = sz_utf8_byte_mask_from_bits_haswell_((sz_u32_t)decoded.four_byte_starts);
        __m256i const four_sel_hi = sz_utf8_byte_mask_from_bits_haswell_((sz_u32_t)(decoded.four_byte_starts >> 32));
        high_lo = _mm256_blendv_epi8(high_lo, mid_lo, four_sel_lo);
        high_hi = _mm256_blendv_epi8(high_hi, mid_hi, four_sel_hi);
        low_lo = _mm256_blendv_epi8(low_lo, alo_lo, four_sel_lo);
        low_hi = _mm256_blendv_epi8(low_hi, alo_hi, four_sel_hi);
    }

    // ASCII descriptor (cp < 0x80) via a single 256-LUT over the raw byte — the cheap gated fast path so a pure-ASCII
    // window (the common English case) never pays the full BMP nibble cascade. Mirrors icelake's `ascii_desc` vpermb.
    __m256i const ascii_desc_lo = sz_utf8_rune_lut256_haswell_(sz_utf8_grapheme_break_haswell_ascii_desc_, raw_lo);
    __m256i const ascii_desc_hi = sz_utf8_rune_lut256_haswell_(sz_utf8_grapheme_break_haswell_ascii_desc_, raw_hi);
    // Any non-ASCII lead present? gate the cold full-BMP cascade behind it. The CJK / Kana arithmetic ranges resolve
    // to GCB=Other (descriptor 0, the ASCII fast path's value on those lanes is overwritten to 0), so a pure-CJK or
    // pure-ASCII window skips the 13-group cascade entirely (mirrors icelake's `cjk_other` cold-gate).
    sz_u64_t const non_ascii = loaded_mask & ~ascii;
    __m256i desc_lo = ascii_desc_lo, desc_hi = ascii_desc_hi;
    sz_u64_t cold = non_ascii;
    if (non_ascii) {
        sz_u64_t const cjk_other = sz_grapheme_cjk_other_haswell_(high_lo, low_lo, high_hi, low_hi) & non_ascii;
        cold = non_ascii & ~cjk_other;
        // Force the CJK-other lanes to descriptor 0 (their ASCII-LUT value on a non-ASCII byte is junk; clear it).
        __m256i const cjk_sel_lo = sz_utf8_byte_mask_from_bits_haswell_((sz_u32_t)cjk_other);
        __m256i const cjk_sel_hi = sz_utf8_byte_mask_from_bits_haswell_((sz_u32_t)(cjk_other >> 32));
        desc_lo = _mm256_andnot_si256(cjk_sel_lo, desc_lo);
        desc_hi = _mm256_andnot_si256(cjk_sel_hi, desc_hi);
    }
    if (cold) {
        // A cold (non-ASCII, non-CJK-other) lane is present: resolve EVERY lane through the full-BMP cascade
        // (Hangul / 2-byte / 3-byte), overwriting the ASCII fast path. ASCII and CJK-other lanes resolve
        // identically (the cascade reproduces 0 for the carved CJK ranges), so the blend below is exact.
        __m256i const bmp_lo = sz_grapheme_bmp_descriptor_haswell_(high_lo, low_lo);
        __m256i const bmp_hi = sz_grapheme_bmp_descriptor_haswell_(high_hi, low_hi);
        __m256i const ascii_sel2_lo = sz_utf8_byte_mask_from_bits_haswell_((sz_u32_t)ascii);
        __m256i const ascii_sel2_hi = sz_utf8_byte_mask_from_bits_haswell_((sz_u32_t)(ascii >> 32));
        desc_lo = _mm256_blendv_epi8(bmp_lo, ascii_desc_lo, ascii_sel2_lo);
        desc_hi = _mm256_blendv_epi8(bmp_hi, ascii_desc_hi, ascii_sel2_hi);
    }
    // A 4-byte lead splits three ways on its blind plane = bits[16..20] of cp, matching icelake's value dispatch
    // (`is_bmp = cp < 0x10000`, `is_astral = 0x10000 <= cp < 0x110000`, else Other):
    //   plane == 0      -> BMP codepoint (cp = (mid<<8)|alo); already resolved through the BMP path above, since the
    //                      high/low override seated `mid`/`alo` for these lanes (e.g. overlong `F0 80 8D A9` -> U+0369).
    //   plane in [1,16]  -> genuine astral (cp in 0x10000..0x10FFFF); overwrite with the 4-byte trie descriptor.
    //   plane >= 17      -> cp >= 0x110000 (e.g. overlong `F4 90 80 80`); neither BMP nor astral, force Other (0).
    // `plane_lo/hi` carry the 5-bit plane per 4-byte-lead lane (junk on other lanes, gated out by `four_byte`).
    sz_u64_t const plane_nonzero = sz_utf8_mask_combine_haswell_(
        _mm256_cmpgt_epi8(plane_lo, _mm256_setzero_si256()), _mm256_cmpgt_epi8(plane_hi, _mm256_setzero_si256()));
    sz_u64_t const plane_le_16 = sz_utf8_mask_combine_haswell_(
        sz_grapheme_cmpge_epu8_haswell_(_mm256_set1_epi8(0x10), plane_lo),
        sz_grapheme_cmpge_epu8_haswell_(_mm256_set1_epi8(0x10), plane_hi));
    sz_u64_t const is_astral = four_byte & plane_nonzero & plane_le_16 & loaded_mask;
    sz_u64_t const is_overrange = four_byte & plane_nonzero & ~plane_le_16 & loaded_mask;
    if (is_astral) {
        // offset = cp - 0x10000; high16/low16 unchanged, plane nibble = cp_plane - 1. The reconstructed `plane`
        // holds bits [16..20] of cp; subtract 1 to get the offset plane nibble (matching the linewraps astral path).
        __m256i const astral_sel_lo = sz_utf8_byte_mask_from_bits_haswell_((sz_u32_t)is_astral);
        __m256i const astral_sel_hi = sz_utf8_byte_mask_from_bits_haswell_((sz_u32_t)(is_astral >> 32));
        __m256i const plane_off_lo = _mm256_sub_epi8(_mm256_and_si256(astral_sel_lo, plane_lo), _mm256_set1_epi8(1));
        __m256i const plane_off_hi = _mm256_sub_epi8(_mm256_and_si256(astral_sel_hi, plane_hi), _mm256_set1_epi8(1));
        __m256i const astral_lo = sz_grapheme_astral_descriptor_haswell_(plane_off_lo, mid_lo, alo_lo);
        __m256i const astral_hi = sz_grapheme_astral_descriptor_haswell_(plane_off_hi, mid_hi, alo_hi);
        desc_lo = _mm256_blendv_epi8(desc_lo, astral_lo, astral_sel_lo);
        desc_hi = _mm256_blendv_epi8(desc_hi, astral_hi, astral_sel_hi);
    }
    if (is_overrange) {
        // cp >= 0x110000: clear to Other (0), the BMP-path value seated on these lanes is meaningless.
        desc_lo = _mm256_andnot_si256(sz_utf8_byte_mask_from_bits_haswell_((sz_u32_t)is_overrange), desc_lo);
        desc_hi = _mm256_andnot_si256(sz_utf8_byte_mask_from_bits_haswell_((sz_u32_t)(is_overrange >> 32)), desc_hi);
    }

    // `0xF8..0xFF` begin no valid sequence and match no lead-length mask; force their start lanes to the Other
    // descriptor (0) so they classify neighbour-independently, matching serial/icelake (§6.2 U+FFFD -> Other).
    sz_u64_t const invalid_lead = sz_utf8_mask_combine_haswell_(
        _mm256_cmpeq_epi8(_mm256_max_epu8(raw_lo, _mm256_set1_epi8((char)0xF8)), raw_lo),
        _mm256_cmpeq_epi8(_mm256_max_epu8(raw_hi, _mm256_set1_epi8((char)0xF8)), raw_hi));
    __m256i const valid_sel_lo = sz_utf8_byte_mask_from_bits_haswell_((sz_u32_t)~invalid_lead);
    __m256i const valid_sel_hi = sz_utf8_byte_mask_from_bits_haswell_((sz_u32_t)(~invalid_lead >> 32));
    desc_lo = _mm256_and_si256(desc_lo, valid_sel_lo);
    desc_hi = _mm256_and_si256(desc_hi, valid_sel_hi);

    sz_grapheme_classified_haswell_t result;
    result.descriptors_lo = desc_lo;
    result.descriptors_hi = desc_hi;
    result.start_lanes = start_lanes;
    result.codepoint_count = (sz_size_t)sz_u64_popcount(start_lanes);
    result.byte_span = byte_span;
    return result;
}

#pragma endregion Gather free Grapheme_Cluster_Break classifier

#pragma region Boundary algebra extractor

/**
 *  @brief  Build the codepoint-dense per-class window masks from the per-lane descriptor halves at the codepoint-start
 *          lanes — the AVX2 twin of `sz_grapheme_build_masks_icelake_` feeding the SAME portable engine. Each per-class
 *          membership mask is built in the BYTE-lane domain (`vpcmpeqb` halves -> `mask_combine`), then compacted to
 *          the codepoint-dense domain by a single `_pext_u64` over the start lanes (the `vpcompressb`-free analogue of
 *          icelake's per-class `& valid` after the dense compress). No scalar loop, no `vpgather`, no rule control flow.
 */
SZ_INTERNAL sz_grapheme_window_masks_t sz_grapheme_build_masks_haswell_(sz_grapheme_classified_haswell_t classified,
                                                                        sz_u64_t valid) {
    sz_u64_t const starts = classified.start_lanes;
    __m256i const desc_lo = classified.descriptors_lo, desc_hi = classified.descriptors_hi;
    __m256i const class_lo = _mm256_and_si256(desc_lo, _mm256_set1_epi8(0x0F));
    __m256i const class_hi = _mm256_and_si256(desc_hi, _mm256_set1_epi8(0x0F));

    sz_grapheme_window_masks_t masks;
    for (int class_index = 0; class_index < 14; ++class_index) {
        __m256i const v = _mm256_set1_epi8((char)class_index);
        sz_u64_t const byte_mask = sz_utf8_mask_combine_haswell_(_mm256_cmpeq_epi8(class_lo, v),
                                                                 _mm256_cmpeq_epi8(class_hi, v));
        masks.class_bit[class_index] = _pext_u64(byte_mask, starts) & valid;
    }
    __m256i const ext_bit = _mm256_set1_epi8(0x40);
    sz_u64_t const ext_byte = sz_utf8_mask_combine_haswell_(
        _mm256_cmpeq_epi8(_mm256_and_si256(desc_lo, ext_bit), ext_bit),
        _mm256_cmpeq_epi8(_mm256_and_si256(desc_hi, ext_bit), ext_bit));
    masks.extended_pictographic = _pext_u64(ext_byte, starts) & valid;
    __m256i const incb_lo = _mm256_and_si256(sz_utf8_srl8_haswell_(desc_lo, 4, 0x0F), _mm256_set1_epi8(0x03));
    __m256i const incb_hi = _mm256_and_si256(sz_utf8_srl8_haswell_(desc_hi, 4, 0x0F), _mm256_set1_epi8(0x03));
    __m256i const incb_consonant = _mm256_set1_epi8((char)sz_grapheme_incb_consonant_k);
    __m256i const incb_extend = _mm256_set1_epi8((char)sz_grapheme_incb_extend_k);
    __m256i const incb_linker = _mm256_set1_epi8((char)sz_grapheme_incb_linker_k);
    masks.indic_consonant = _pext_u64(sz_utf8_mask_combine_haswell_(_mm256_cmpeq_epi8(incb_lo, incb_consonant),
                                                                    _mm256_cmpeq_epi8(incb_hi, incb_consonant)),
                                      starts) &
                            valid;
    masks.indic_extend = _pext_u64(sz_utf8_mask_combine_haswell_(_mm256_cmpeq_epi8(incb_lo, incb_extend),
                                                                 _mm256_cmpeq_epi8(incb_hi, incb_extend)),
                                   starts) &
                         valid;
    masks.indic_linker = _pext_u64(sz_utf8_mask_combine_haswell_(_mm256_cmpeq_epi8(incb_lo, incb_linker),
                                                                 _mm256_cmpeq_epi8(incb_hi, incb_linker)),
                                   starts) &
                         valid;
    return masks;
}

#pragma endregion Boundary algebra extractor

#pragma region Grapheme forward driver

SZ_PUBLIC sz_size_t sz_utf8_graphemes_haswell(             //
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
        sz_grapheme_classified_haswell_t const window = sz_grapheme_classify_window_haswell_(text_u8, length, base);
        if (window.codepoint_count == 0) break; // defensive: cannot happen for valid input

        int const codepoint_count = (int)window.codepoint_count;
        sz_u64_t const valid = (codepoint_count >= 64) ? ~0ull : ((1ull << codepoint_count) - 1);
        sz_grapheme_window_masks_t const masks = sz_grapheme_build_masks_haswell_(window, valid);
        sz_u64_t const dense_boundary = sz_grapheme_window_boundaries_(&masks, codepoint_count, valid, &carry);
        // Scatter dense boundary bits back to codepoint-start byte lanes (`_pdep_u64` deposits the j-th dense bit into
        // the j-th set start lane), recovering the byte-domain boundary mask the drain consumes.
        sz_u64_t boundary = _pdep_u64(dense_boundary, window.start_lanes);

        // GB1 anchor at byte 0 of the first window is the open cluster's own start, not a new break: clear it.
        if (base == 0) boundary &= ~1ull;
        clusters = sz_utf8_rune_drain_forward_haswell_(boundary, base, cluster_starts, cluster_lengths, clusters,
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
#endif // SZ_USE_HASWELL

#ifdef __cplusplus
}
#endif

#endif // STRINGZILLA_UTF8_GRAPHEMES_HASWELL_H_
