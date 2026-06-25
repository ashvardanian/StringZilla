/**
 *  @file   include/stringzilla/utf8_graphemes/icelake.h
 *  @author Ash Vardanian
 *  @brief  Ice Lake (AVX-512) backend for UAX-29 extended grapheme cluster boundaries, fully vectorized end-to-end.
 *
 *  Every 64-byte window is decoded for all 64 lanes through the shared codepoint substrate
 *  (`sz_utf8_rune_decode_window_`), classified gather-free into one packed Grapheme_Cluster_Break descriptor
 *  per lane, compacted to a codepoint-dense descriptor vector with a single `vpcompressb`, and resolved by the
 *  branchless rule algebra. The classifier routes each codepoint through one rare-class-gated path - an ASCII
 *  `vpermb`, a 2-byte page LUT, the cold three-stage BMP trie, or a 4-stage astral trie - so a pure-ASCII or 2-byte
 *  window never pays the cold cascade and emoji resolve without a linear range scan.
 *
 *  Design note: the per-family classifier tables + 4-stage astral trie are the deliberate, measurement-justified
 *  shape (CONTRIBUTING-KERNELS.md §6.9). The astral trie measured ~24x over the sorted-range scan and the gated fast
 *  paths ~2-9x over a single substrate cascade. The tables are stored `sz_align_(64)` + zero-padded to a multiple of
 *  64 in `tables.h` and every tile is read straight from `.rodata` via the substrate `permute256_`/`lut_cascade_`
 *  helpers — re-init-free (§6.13): there is no per-call `luts` struct or `_init_` load, so calling the kernel on a
 *  short input is cheap. The astral is a constant-latency walk (no per-range `for`), dispatch is value-based (§6.1)
 *  and ill-formed input is forced to U+FFFD in-register (§6.2). Satisfies the intent of defects #1/#2.
 *
 *  The three unbounded runs (GB12/13 Regional_Indicator parity, GB11 Extended_Pictographic-ZWJ chain, GB9c Indic
 *  conjunct) are threaded across windows by a small register carry, so no rule is re-walked scalar-wise and no
 *  boundary is deferred to the serial oracle. There is no per-lane scalar loop, no spill-to-stack-then-reload of the
 *  classifier, and no `vpgather` on any path.
 */
#ifndef STRINGZILLA_UTF8_GRAPHEMES_ICELAKE_H_
#define STRINGZILLA_UTF8_GRAPHEMES_ICELAKE_H_

#include "stringzilla/types.h"
#include "stringzilla/utf8_graphemes/tables.h"
#include "stringzilla/utf8_graphemes/serial.h"
#include "stringzilla/utf8_runes/icelake.h"

#ifdef __cplusplus
extern "C" {
#endif

#if SZ_USE_ICELAKE
#if defined(__clang__)
#pragma clang attribute push(                                                                          \
    __attribute__((target("avx,avx512f,avx512vl,avx512bw,avx512dq,avx512vbmi,avx512vbmi2,bmi,bmi2"))), \
    apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("avx", "avx512f", "avx512vl", "avx512bw", "avx512dq", "avx512vbmi", "avx512vbmi2", "bmi", "bmi2")
#endif

#pragma region Gather free Grapheme_Cluster_Break classifier

enum {
    sz_grapheme_break_mid_tiles_k = sizeof(sz_utf8_grapheme_break_stage_mid_) / 64, /**< ZMM tiles of `stage_mid`. */
    sz_grapheme_break_sub_packed_tiles_k = sizeof(sz_utf8_grapheme_break_stage_sub_packed_) /
        64, /**< ZMM tiles of 4-bit-packed `stage_sub`. */
    sz_grapheme_break_astral_stage1_tiles_k = sizeof(sz_utf8_grapheme_break_astral_s1_) / 64,
    sz_grapheme_break_astral_stage2_tiles_k = sizeof(sz_utf8_grapheme_break_astral_s2_) / 64,
    sz_grapheme_break_astral_leaf_tiles_k = sizeof(sz_utf8_grapheme_break_astral_leaf_) / 64,
};

/** @brief  Descriptor of a codepoint < 0x80 by a `vpermb` over the two aligned `ascii_desc` `.rodata` tiles. */
SZ_INTERNAL __m512i sz_grapheme_ascii_descriptor_icelake_(__m512i codepoints) {
    __m512i const low_six = _mm512_and_si512(codepoints, _mm512_set1_epi32(0x3F));
    __mmask16 const high_half = _mm512_test_epi32_mask(codepoints, _mm512_set1_epi32(0x40));
    __m512i const tile_low = _mm512_load_si512((void const *)(sz_utf8_grapheme_break_ascii_desc_ + 0));
    __m512i const tile_high = _mm512_load_si512((void const *)(sz_utf8_grapheme_break_ascii_desc_ + 64));
    __m512i const low = _mm512_and_si512(_mm512_permutexvar_epi8(low_six, tile_low), _mm512_set1_epi32(0xFF));
    __m512i const high = _mm512_and_si512(_mm512_permutexvar_epi8(low_six, tile_high), _mm512_set1_epi32(0xFF));
    return _mm512_mask_blend_epi32(high_half, low, high);
}

/** @brief  Descriptor of a codepoint < 0x800 via the aligned `page_0800` LUT (a substrate `vpermi2b` cascade). */
SZ_INTERNAL __m512i sz_grapheme_small_page_icelake_(__m512i codepoints) {
    return sz_utf8_rune_lut_cascade_icelake_(sz_utf8_grapheme_break_page_0800_, 32,
                                             _mm512_and_si512(codepoints, _mm512_set1_epi32(0x7FF)));
}

/** @brief  Descriptor of a cold BMP codepoint (0x800..0xFFFF) by the three-stage trie: `stage_hi` `vpermb`, then the
 *          `stage_mid` `vpermi2b` cascade and the 4-bit-packed `stage_sub` nibble cascade (half the tiles), then the
 *          18-byte `id_to_desc` `vpermb`. All tables read straight from aligned `.rodata` — no per-call load. */
SZ_INTERNAL __m512i sz_grapheme_classify_cascade_icelake_(__m512i codepoints) {
    __m512i const high_byte = _mm512_and_si512(_mm512_srli_epi32(codepoints, 8), _mm512_set1_epi32(0xFF));
    __m512i const mid = sz_utf8_rune_permute256_icelake_(sz_utf8_grapheme_break_stage_hi_, high_byte);
    __m512i const mid_index = _mm512_add_epi32(
        _mm512_slli_epi32(mid, 4), _mm512_and_si512(_mm512_srli_epi32(codepoints, 4), _mm512_set1_epi32(0xF)));
    __m512i const sub = sz_utf8_rune_lut_cascade_icelake_(sz_utf8_grapheme_break_stage_mid_,
                                                          sz_grapheme_break_mid_tiles_k, mid_index);
    __m512i const sub_index = _mm512_add_epi32(_mm512_slli_epi32(sub, 4),
                                               _mm512_and_si512(codepoints, _mm512_set1_epi32(0xF)));
    // `stage_sub` outputs a 4-bit descriptor index, so it is stored two cells per byte: the nibble cascade walks half
    // the tiles (25 vs 50), halving this stage's `vpermi2b` port-5 cost — the residual-cascade hot spot for Indic/CJK.
    __m512i const descriptor_index = sz_utf8_rune_lut_cascade_nibble_icelake_(
        sz_utf8_grapheme_break_stage_sub_packed_, sz_grapheme_break_sub_packed_tiles_k, sub_index);
    __m512i const id_to_desc = _mm512_load_si512((void const *)sz_utf8_grapheme_break_id_to_desc_);
    return _mm512_and_si512(_mm512_permutexvar_epi8(descriptor_index, id_to_desc), _mm512_set1_epi32(0xFF));
}

/** @brief  Descriptor of an astral codepoint (>= 0x10000) via the 4-stage trie over offset = codepoint - 0x10000
 *          (an 8/4/4/4 split), every tile read straight from aligned `.rodata`. Byte-identical to the serial
 *          sorted-range scan, replacing the per-window linear fold. */
SZ_INTERNAL __m512i sz_grapheme_classify_astral16_icelake_(__m512i codepoints) {
    __m512i const offset = _mm512_sub_epi32(codepoints, _mm512_set1_epi32(0x10000));
    __m512i const stage1 = sz_utf8_rune_permute256_icelake_(
        sz_utf8_grapheme_break_astral_s0_, _mm512_and_si512(_mm512_srli_epi32(offset, 12), _mm512_set1_epi32(0xFF)));
    __m512i const stage2_index = _mm512_add_epi32(
        _mm512_slli_epi32(stage1, 4), _mm512_and_si512(_mm512_srli_epi32(offset, 8), _mm512_set1_epi32(0xF)));
    __m512i const stage2 = sz_utf8_rune_lut_cascade_icelake_(sz_utf8_grapheme_break_astral_s1_,
                                                             sz_grapheme_break_astral_stage1_tiles_k, stage2_index);
    __m512i const leaf_index = _mm512_add_epi32(_mm512_slli_epi32(stage2, 4),
                                                _mm512_and_si512(_mm512_srli_epi32(offset, 4), _mm512_set1_epi32(0xF)));
    __m512i const leaf = sz_utf8_rune_lut_cascade_icelake_(sz_utf8_grapheme_break_astral_s2_,
                                                           sz_grapheme_break_astral_stage2_tiles_k, leaf_index);
    __m512i const class_index = _mm512_add_epi32(_mm512_slli_epi32(leaf, 4),
                                                 _mm512_and_si512(offset, _mm512_set1_epi32(0xF)));
    return sz_utf8_rune_lut_cascade_icelake_(sz_utf8_grapheme_break_astral_leaf_, sz_grapheme_break_astral_leaf_tiles_k,
                                             class_index);
}

/**
 *  @brief  Mask of lanes whose BMP codepoint resolves uniformly to GCB=Other via the CJK / Kana arithmetic ranges
 *          (so they need no cold cascade): `[0x3000,0xA66E] | [0xD7FC,0xFB1D]` minus the tight interior exceptions
 *          (Extend voicing marks `302A-3030` / `3099-309A`, `303D`, enclosed `3297` / `3299`). Mirrors the word
 *          kernel's `cjk_combined` carve. Six `vpcmp`; called only when a cold lane is present.
 */
SZ_INTERNAL __mmask16 sz_grapheme_cjk_other_icelake_(__m512i codepoints) {
    __mmask16 const run_a = _kand_mask16(_mm512_cmpge_epu32_mask(codepoints, _mm512_set1_epi32(0x3000)),
                                         _mm512_cmple_epu32_mask(codepoints, _mm512_set1_epi32(0xA66E)));
    __mmask16 const run_b = _kand_mask16(_mm512_cmpge_epu32_mask(codepoints, _mm512_set1_epi32(0xD7FC)),
                                         _mm512_cmple_epu32_mask(codepoints, _mm512_set1_epi32(0xFB1D)));
    __mmask16 const exception_run = _kor_mask16(
        _kand_mask16(_mm512_cmpge_epu32_mask(codepoints, _mm512_set1_epi32(0x302A)),
                     _mm512_cmple_epu32_mask(codepoints, _mm512_set1_epi32(0x3030))),
        _kand_mask16(_mm512_cmpge_epu32_mask(codepoints, _mm512_set1_epi32(0x3099)),
                     _mm512_cmple_epu32_mask(codepoints, _mm512_set1_epi32(0x309A))));
    __mmask16 const exception_point = _kor_mask16(
        _mm512_cmpeq_epi32_mask(codepoints, _mm512_set1_epi32(0x303D)),
        _kor_mask16(_mm512_cmpeq_epi32_mask(codepoints, _mm512_set1_epi32(0x3297)),
                    _mm512_cmpeq_epi32_mask(codepoints, _mm512_set1_epi32(0x3299))));
    return _kandn_mask16(_kor_mask16(exception_run, exception_point), _kor_mask16(run_a, run_b));
}

/**
 *  @brief  Classify 16 codepoints (one u32 ZMM) into 16 packed Grapheme_Cluster_Break descriptors (one per low byte).
 *
 *  Each lane takes exactly one rare-class-gated path so a pure-ASCII / 2-byte / astral chunk never pays the cold
 *  cascade: Hangul U+AC00..U+D7A3 by the LV/LVT residue formula (no table); codepoint < 0x80 by the `ascii_desc`
 *  `vpermb`; codepoint < 0x800 by the `page_0800` LUT; the cold 0x800..0xFFFF residue by the three-stage BMP trie;
 *  and codepoint >= 0x10000 by the 4-stage astral trie. No `vpgatherdd`, no linear range scan, no scalar loop.
 */
SZ_INTERNAL __m512i sz_grapheme_classify16_icelake_(__m512i codepoints) {
    __m512i const hangul_base = _mm512_set1_epi32(0xAC00);
    __mmask16 const is_hangul = _kand_mask16(_mm512_cmpge_epu32_mask(codepoints, hangul_base),
                                             _mm512_cmple_epu32_mask(codepoints, _mm512_set1_epi32(0xD7A3)));
    __m512i const hangul_relative = _mm512_sub_epi32(codepoints, hangul_base);
    // relative < 11172, so a mul-hi reciprocal for `/ 28` is exact (2342 = ceil(2^16 / 28)).
    __m512i const quotient = _mm512_srli_epi32(_mm512_mullo_epi32(hangul_relative, _mm512_set1_epi32(2342)), 16);
    __mmask16 const is_lv = _mm512_cmpeq_epi32_mask(
        _mm512_sub_epi32(hangul_relative, _mm512_mullo_epi32(quotient, _mm512_set1_epi32(28))), _mm512_setzero_si512());
    __m512i const hangul_descriptor = _mm512_mask_blend_epi32(is_lv,
                                                              _mm512_set1_epi32((int)sz_grapheme_break_hangul_lvt_k),
                                                              _mm512_set1_epi32((int)sz_grapheme_break_hangul_lv_k));

    __mmask16 const is_ascii = _mm512_cmplt_epu32_mask(codepoints, _mm512_set1_epi32(0x80));
    __mmask16 const below_0800 = _mm512_cmplt_epu32_mask(codepoints, _mm512_set1_epi32(0x800));
    __mmask16 const is_bmp = _mm512_cmplt_epu32_mask(codepoints, _mm512_set1_epi32(0x10000));
    __mmask16 const is_small = _kandn_mask16(is_ascii, below_0800);                         // 0x80..0x7FF
    __mmask16 const cold_raw = _kandn_mask16(is_hangul, _kandn_mask16(below_0800, is_bmp)); // 0x800..0xFFFF, non-Hangul
    // CJK / Kana arithmetic fast-path, gated behind any-cold-lane so ASCII / Hangul windows pay nothing: the carved
    // lanes resolve to GCB=Other (descriptor 0, the zero-init value), so a pure-CJK window skips the 50-tile cascade.
    // Byte-identical -- the cascade would produce the same Other for these lanes whether or not residual cold remains.
    __mmask16 is_cold = cold_raw;
    if (cold_raw) is_cold = _kandn_mask16(sz_grapheme_cjk_other_icelake_(codepoints), cold_raw);
    // `cp >= 0x110000` (e.g. the overlong `F4 90 80 80`) stays Other, matching serial.
    __mmask16 const is_astral = _kand_mask16(_kandn_mask16(is_bmp, (__mmask16)0xFFFF),
                                             _mm512_cmplt_epu32_mask(codepoints, _mm512_set1_epi32((int)0x110000)));

    __mmask16 const bmp_non_hangul = _kandn_mask16(is_hangul, is_bmp);
    __m512i descriptor = _mm512_setzero_si512();
    if (is_cold) {
        // A cold 3-byte lane is present: the cascade resolves EVERY BMP lane (ASCII and 2-byte included), so a mixed
        // CJK/ASCII window pays one cascade instead of the cascade plus the ASCII and page fast paths.
        descriptor = _mm512_mask_mov_epi32(descriptor, bmp_non_hangul,
                                           sz_grapheme_classify_cascade_icelake_(codepoints));
    }
    else {
        if (is_ascii)
            descriptor = _mm512_mask_mov_epi32(descriptor, is_ascii, sz_grapheme_ascii_descriptor_icelake_(codepoints));
        if (is_small)
            descriptor = _mm512_mask_mov_epi32(descriptor, is_small, sz_grapheme_small_page_icelake_(codepoints));
    }
    if (is_astral)
        descriptor = _mm512_mask_mov_epi32(descriptor, is_astral, sz_grapheme_classify_astral16_icelake_(codepoints));
    return _mm512_mask_blend_epi32(is_hangul, descriptor, hangul_descriptor);
}

/**
 *  @brief  Widen one 16-lane quarter to full 21-bit codepoints and classify it to packed descriptors. The astral
 *          (4-byte) lanes in the quarter take the reconstructed plane/mid/low codepoint; all others take the BMP
 *          `(high << 8) | low`. Returns 16 descriptors in the low byte of each 32-bit lane.
 */
SZ_INTERNAL __m512i sz_grapheme_classify_quarter_icelake_( //
    __m128i high_slice, __m128i low_slice, __m128i plane_slice, __m128i mid_slice, __m128i lo_slice,
    __mmask16 astral_quarter) {
    __m512i const codepoint_bmp = _mm512_or_si512(_mm512_slli_epi32(_mm512_cvtepu8_epi32(high_slice), 8),
                                                  _mm512_cvtepu8_epi32(low_slice));
    __m512i const codepoint_astral = _mm512_or_si512(
        _mm512_or_si512(_mm512_slli_epi32(_mm512_cvtepu8_epi32(plane_slice), 16),
                        _mm512_slli_epi32(_mm512_cvtepu8_epi32(mid_slice), 8)),
        _mm512_cvtepu8_epi32(lo_slice));
    __m512i const codepoint = _mm512_mask_blend_epi32(astral_quarter, codepoint_bmp, codepoint_astral);
    return sz_grapheme_classify16_icelake_(codepoint);
}

/**
 *  @brief  Classify all 64 lanes of a decoded window into one packed Grapheme_Cluster_Break descriptor per lane.
 *
 *  The substrate decode already holds `high = codepoint >> 8` / `low = codepoint & 0xFF` for the BMP path and the raw bytes for
 *  astral reconstruction. Each 16-lane quarter is widened in-register to a full 21-bit codepoint (the astral lanes
 *  reassemble plane/mid/low from the four UTF-8 bytes) and classified; the four descriptor quarters are written back
 *  as one byte per lane. No scalar per-lane loop, no `vpgather`, no spill round-trip.
 */
SZ_INTERNAL __m512i sz_grapheme_classify_window_icelake_( //
    sz_utf8_rune_window_t const *decoded, __m512i next1, __m512i next2, __m512i next3) {
    // Astral (4-byte) lead reconstruction: plane = ((b0 & 7) << 2) | ((b1 >> 4) & 3); mid = ((b1 & F) << 4) |
    // ((b2 >> 2) & F); low = ((b2 & 3) << 6) | (b3 & 3F); codepoint = (plane << 16) | (mid << 8) | low.
    __m512i const window = decoded->window;
    // The BMP `high`/`low` are reconstructed HERE from the raw lead and the (already edge-zeroed) neighbour bytes
    // `next1`/`next2`, not read from `decoded->high`/`decoded->low`. The substrate's neighbour fetch is an in-register
    // rotate that wraps the window head into a truncated trailing lead's missing continuation byte; recomputing from
    // the zeroed neighbours pads out-of-window bytes with 0, matching the serial blind decode byte-for-byte (§6.1) so a
    // 2-/3-byte lead at the loaded edge classifies neighbour-independently. ASCII (1-byte) lanes take the identity
    // codepoint `(0, raw byte)`; the 2-/3-byte formulas mirror `sz_grapheme_break_property_at_`.
    __mmask64 const ascii = _mm512_cmplt_epu8_mask(window, _mm512_set1_epi8((char)0x80));
    __mmask64 const three_byte = decoded->three_byte_starts;
    // 2-byte lead `110xxxxx`: high = ((lead & 0x1F) >> 2) & 0x07; low = ((lead & 0x03) << 6) | (next1 & 0x3F).
    __m512i const two_high = sz_utf8_srl8_icelake_(_mm512_and_si512(window, _mm512_set1_epi8(0x1F)), 2, 0x07);
    __m512i const two_low = _mm512_or_si512(_mm512_slli_epi16(_mm512_and_si512(window, _mm512_set1_epi8(0x03)), 6),
                                            _mm512_and_si512(next1, _mm512_set1_epi8(0x3F)));
    // 3-byte lead `1110xxxx`: high = ((lead & 0x0F) << 4) | ((next1 >> 2) & 0x0F); low = ((next1 & 0x03) << 6) |
    // (next2 & 0x3F).
    __m512i const three_high = _mm512_or_si512(_mm512_slli_epi16(_mm512_and_si512(window, _mm512_set1_epi8(0x0F)), 4),
                                               sz_utf8_srl8_icelake_(next1, 2, 0x0F));
    __m512i const three_low = _mm512_or_si512(_mm512_slli_epi16(_mm512_and_si512(next1, _mm512_set1_epi8(0x03)), 6),
                                              _mm512_and_si512(next2, _mm512_set1_epi8(0x3F)));
    __m512i const non_ascii_high = _mm512_mask_blend_epi8(three_byte, two_high, three_high);
    __m512i const non_ascii_low = _mm512_mask_blend_epi8(three_byte, two_low, three_low);
    __m512i const high = _mm512_maskz_mov_epi8(_knot_mask64(ascii), non_ascii_high);
    __m512i const low = _mm512_mask_blend_epi8(ascii, non_ascii_low, window);
    __m512i const plane = _mm512_or_si512(_mm512_slli_epi16(_mm512_and_si512(window, _mm512_set1_epi8(0x07)), 2),
                                          sz_utf8_srl8_icelake_(next1, 4, 0x03));
    __m512i const mid_byte = _mm512_or_si512(_mm512_slli_epi16(_mm512_and_si512(next1, _mm512_set1_epi8(0x0F)), 4),
                                             sz_utf8_srl8_icelake_(next2, 2, 0x0F));
    __m512i const lo_byte = _mm512_or_si512(_mm512_slli_epi16(_mm512_and_si512(next2, _mm512_set1_epi8(0x03)), 6),
                                            _mm512_and_si512(next3, _mm512_set1_epi8(0x3F)));
    sz_u64_t const four_byte = _cvtmask64_u64(decoded->four_byte_starts);

    __m512i const quarter0 = sz_grapheme_classify_quarter_icelake_(
        _mm512_extracti32x4_epi32(high, 0), _mm512_extracti32x4_epi32(low, 0), _mm512_extracti32x4_epi32(plane, 0),
        _mm512_extracti32x4_epi32(mid_byte, 0), _mm512_extracti32x4_epi32(lo_byte, 0), (__mmask16)(four_byte & 0xFFFF));
    __m512i const quarter1 = sz_grapheme_classify_quarter_icelake_(
        _mm512_extracti32x4_epi32(high, 1), _mm512_extracti32x4_epi32(low, 1), _mm512_extracti32x4_epi32(plane, 1),
        _mm512_extracti32x4_epi32(mid_byte, 1), _mm512_extracti32x4_epi32(lo_byte, 1),
        (__mmask16)((four_byte >> 16) & 0xFFFF));
    __m512i const quarter2 = sz_grapheme_classify_quarter_icelake_(
        _mm512_extracti32x4_epi32(high, 2), _mm512_extracti32x4_epi32(low, 2), _mm512_extracti32x4_epi32(plane, 2),
        _mm512_extracti32x4_epi32(mid_byte, 2), _mm512_extracti32x4_epi32(lo_byte, 2),
        (__mmask16)((four_byte >> 32) & 0xFFFF));
    __m512i const quarter3 = sz_grapheme_classify_quarter_icelake_(
        _mm512_extracti32x4_epi32(high, 3), _mm512_extracti32x4_epi32(low, 3), _mm512_extracti32x4_epi32(plane, 3),
        _mm512_extracti32x4_epi32(mid_byte, 3), _mm512_extracti32x4_epi32(lo_byte, 3),
        (__mmask16)((four_byte >> 48) & 0xFFFF));

    // Re-pack the four 16-lane descriptor quarters (each in the low byte of its 32-bit lanes) into one byte/lane.
    __m512i const packed01 = _mm512_packus_epi32(_mm512_and_si512(quarter0, _mm512_set1_epi32(0xFF)),
                                                 _mm512_and_si512(quarter1, _mm512_set1_epi32(0xFF)));
    __m512i const packed23 = _mm512_packus_epi32(_mm512_and_si512(quarter2, _mm512_set1_epi32(0xFF)),
                                                 _mm512_and_si512(quarter3, _mm512_set1_epi32(0xFF)));
    __m512i const packed = _mm512_packus_epi16(packed01, packed23);
    // `vpackus` interleaves the four 128-bit lanes as {q0a,q1a,q2a,q3a, q0b,...}; restore ascending lane order.
    __m512i const restore = _mm512_setr_epi32(0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15);
    __m512i const descriptors = _mm512_permutexvar_epi32(restore, packed);
    // `0xF8..0xFF` begin no valid UTF-8 sequence and match no lead-length mask, so the 2-byte fold above read a
    // wrapped neighbour and classified them by a junk value (neighbour-dependent at the 64-byte window edge). Force
    // those start lanes to the Other descriptor (§6.2 U+FFFD substitution) so they classify neighbour-independently,
    // identically to the serial backend. One masked compare, no scalar walk.
    __mmask64 const invalid_lead = _mm512_cmpge_epu8_mask(window, _mm512_set1_epi8((char)0xF8));
    return _mm512_maskz_mov_epi8(_knot_mask64(invalid_lead), descriptors);
}

#pragma endregion Gather free Grapheme_Cluster_Break classifier

#pragma region Grapheme boundary algebra

/*  The `sz_grapheme_carry_t`, `sz_grapheme_window_masks_t`, `sz_grapheme_carry_empty_`, `sz_grapheme_stateful_joins_`,
 *  `sz_grapheme_previous_` and the portable `sz_grapheme_window_boundaries_` engine live in `serial.h`; only the
 *  `__m512i`->`sz_u64_t` mask extractor and the thin wrapper that feeds the portable engine are ISA-specific here. */

/**
 *  @brief  Reduce the codepoint-dense packed descriptor bytes (in a ZMM) into per-class membership masks for the
 *          rule algebra. One `vpcmpeqb`/`kmov` per grapheme class plus the InCB / Extended_Pictographic bit tests.
 */
SZ_INTERNAL sz_grapheme_window_masks_t sz_grapheme_build_masks_icelake_(__m512i descriptors, sz_u64_t valid) {
    sz_grapheme_window_masks_t masks;
    __m512i const class_field = _mm512_and_si512(descriptors, _mm512_set1_epi8(0x0F));
    for (int class_index = 0; class_index < 14; ++class_index)
        masks.class_bit[class_index] =
            _cvtmask64_u64(_mm512_cmpeq_epi8_mask(class_field, _mm512_set1_epi8((char)class_index))) & valid;
    masks.extended_pictographic = _cvtmask64_u64(_mm512_test_epi8_mask(descriptors, _mm512_set1_epi8(0x40))) & valid;
    __m512i const incb_field = _mm512_and_si512(_mm512_srli_epi16(descriptors, 4), _mm512_set1_epi8(0x03));
    masks.indic_consonant = _cvtmask64_u64(_mm512_cmpeq_epi8_mask(
                                incb_field, _mm512_set1_epi8((char)sz_grapheme_incb_consonant_k))) &
                            valid;
    masks.indic_extend =
        _cvtmask64_u64(_mm512_cmpeq_epi8_mask(incb_field, _mm512_set1_epi8((char)sz_grapheme_incb_extend_k))) & valid;
    masks.indic_linker =
        _cvtmask64_u64(_mm512_cmpeq_epi8_mask(incb_field, _mm512_set1_epi8((char)sz_grapheme_incb_linker_k))) & valid;
    return masks;
}

/**
 *  @brief  Boundary bitmask for one codepoint-dense window. Bit i set → a grapheme break sits before dense codepoint
 *          i. Builds the per-class masks in-register (the only `__m512i`->`sz_u64_t` contact), then delegates every
 *          GB1-GB13 decision to the shared portable @ref sz_grapheme_window_boundaries_ engine.
 */
SZ_INTERNAL sz_u64_t sz_grapheme_window_boundaries_icelake_(__m512i descriptors, int codepoint_count,
                                                            sz_grapheme_carry_t *carry) {
    sz_u64_t const valid = (codepoint_count >= 64) ? ~0ull : ((1ull << codepoint_count) - 1);
    sz_grapheme_window_masks_t const window = sz_grapheme_build_masks_icelake_(descriptors, valid);
    return sz_grapheme_window_boundaries_(&window, codepoint_count, valid, carry);
}

#pragma endregion Grapheme boundary algebra

#pragma region Window classification driver

/**
 *  @brief  One byte window resolved to codepoint-dense boundary bits plus the byte geometry to map them back.
 *
 *  `boundary` bit j set → a grapheme break sits before the j-th resolved codepoint start in this window.
 *  `start_offsets[j]` is that codepoint's byte offset from the window base; `codepoint_count` is how many full
 *  codepoint starts the window resolved; `byte_span` is the offset of the next start (window advance).
 */
typedef struct sz_grapheme_window_t {
    sz_u64_t boundary;         /**< Byte-domain: bit i => a grapheme break sits before byte lane i. */
    sz_size_t codepoint_count; /**< Number of codepoint starts resolved in this window (<= 64). */
    sz_size_t byte_span;       /**< Bytes consumed by the resolved codepoints (offset of the next start). */
} sz_grapheme_window_t;

/**
 *  @brief  Decode a 64-byte byte-window for all 64 lanes, classify them gather-free into packed descriptors,
 *          compact the descriptors at codepoint starts to a dense vector, and run the boundary algebra. The
 *          trailing partial codepoint (whose continuation bytes fall outside the window) is left for the next
 *          window so cross-window runs stay exact. Pure register dataflow: one decode, one classify, one compress.
 */
SZ_INTERNAL sz_grapheme_window_t sz_grapheme_classify_window_full_icelake_( //
    sz_u8_t const *text, sz_size_t length, sz_size_t base, __m512i lane_identity, sz_grapheme_carry_t *carry) {

    sz_utf8_rune_window_t const decoded = sz_utf8_rune_decode_window_icelake_(text + base, length - base,
                                                                              lane_identity);
    sz_size_t const loaded = decoded.loaded;
    sz_u64_t start_lanes = _cvtmask64_u64(decoded.codepoint_starts);

    // Effective-window trim: on a FULL window a multi-byte lead near the 64-byte edge may declare a sequence that runs
    // past `loaded`, so its decode would read a wrapped (lane 0) neighbour for the missing continuation byte and
    // misclassify. A lead overruns iff its declared span reaches past `loaded`: a 2-byte start at lane >= loaded-1, a
    // 3-byte at lane >= loaded-2, a 4-byte at lane >= loaded-3 (a 1-byte start never overruns). Trim to the LOWEST
    // overrunning lane, fully branchless: union the overrunning starts and take the lowest set bit. A truncated lead
    // can sit below a shorter trailing start that fits, so the union (not a high-to-low first hit) is what is needed.
    sz_size_t byte_span = loaded;
    if (loaded >= 64) {
        sz_u64_t const two = _cvtmask64_u64(decoded.two_byte_starts);
        sz_u64_t const three = _cvtmask64_u64(decoded.three_byte_starts);
        sz_u64_t const four = _cvtmask64_u64(decoded.four_byte_starts);
        sz_u64_t const overrun = (two & ~sz_u64_mask_until_(loaded - 1)) | (three & ~sz_u64_mask_until_(loaded - 2)) |
                                 (four & ~sz_u64_mask_until_(loaded - 3));
        byte_span = overrun ? (sz_size_t)sz_u64_ctz(overrun) : loaded;
        start_lanes &= sz_u64_mask_until_(byte_span);
    }

    // The neighbour fetches `next{1,2,3}` are an in-register byte rotate of `decoded.window`, so a lane near the loaded
    // end reads a WRAPPED lane-0 byte for any continuation that falls at or beyond `loaded`. On a short final window
    // (`loaded < 64`) those continuation bytes are genuinely absent (the input ended mid-sequence), so a truncated
    // trailing lead would otherwise decode against the wrapped window head and classify neighbour-dependently. Zero
    // every `next k` lane whose source byte index `i + k` reaches `loaded`, matching the serial blind decode that pads
    // out-of-bounds continuation bytes with 0 (§6.1). For a FULL window (`loaded == 64`) the only lanes this touches
    // are the top one or two truncated leads, which the effective-window trim already deferred to the next window where
    // their real continuation bytes live; zeroing them here is inert. No scalar walk, one masked move per neighbour.
    sz_u64_t const in_window = _cvtmask64_u64(sz_u64_mask_until_(loaded));
    __mmask64 const next1_present = _cvtu64_mask64(in_window >> 1);
    __mmask64 const next2_present = _cvtu64_mask64(in_window >> 2);
    __mmask64 const next3_present = _cvtu64_mask64(in_window >> 3);
    __m512i const next1 = _mm512_maskz_permutexvar_epi8(
        next1_present, _mm512_add_epi8(lane_identity, _mm512_set1_epi8(1)), decoded.window);
    __m512i const next2 = _mm512_maskz_permutexvar_epi8(
        next2_present, _mm512_add_epi8(lane_identity, _mm512_set1_epi8(2)), decoded.window);
    __m512i const next3 = _mm512_maskz_permutexvar_epi8(
        next3_present, _mm512_add_epi8(lane_identity, _mm512_set1_epi8(3)), decoded.window);
    __m512i const descriptors_per_lane = sz_grapheme_classify_window_icelake_(&decoded, next1, next2, next3);

    // Compact the per-lane descriptors at codepoint-start lanes into the dense codepoint domain (one `vpcompressb`)
    // so the rule algebra runs on adjacent codepoints with no continuation-byte gaps.
    __mmask64 const start_mask = _cvtu64_mask64(start_lanes);
    sz_grapheme_window_t result;
    result.codepoint_count = (sz_size_t)_mm_popcnt_u64(start_lanes);
    result.byte_span = byte_span;
    __m512i const dense_descriptors = _mm512_maskz_compress_epi8(start_mask, descriptors_per_lane);
    sz_u64_t const dense_boundary = sz_grapheme_window_boundaries_icelake_(dense_descriptors,
                                                                           (int)result.codepoint_count, carry);
    // Scatter the dense boundary bits back to their codepoint-start byte lanes: `_pdep_u64` deposits the j-th dense
    // bit into the j-th set bit of `start_lanes`, recovering the byte-domain boundary mask the drains consume.
    result.boundary = _pdep_u64(dense_boundary, start_lanes);
    return result;
}

#pragma endregion Window classification driver

#pragma region Grapheme forward driver

SZ_PUBLIC sz_size_t sz_utf8_graphemes_icelake(             //
    sz_cptr_t text, sz_size_t length,                      //
    sz_size_t *cluster_starts, sz_size_t *cluster_lengths, //
    sz_size_t clusters_capacity, sz_size_t *bytes_consumed) {

    sz_size_t clusters = 0;
    if (length == 0 || clusters_capacity == 0) {
        if (bytes_consumed) *bytes_consumed = 0;
        return 0;
    }
    sz_u8_t const *text_u8 = (sz_u8_t const *)text;
    __m512i const lane_identity = sz_utf8_lane_identity_icelake_();

    sz_grapheme_carry_t carry = sz_grapheme_carry_empty_();
    sz_size_t cluster_start = 0;
    sz_size_t base = 0;

    while (base < length) {
        sz_grapheme_window_t const window = sz_grapheme_classify_window_full_icelake_(text_u8, length, base,
                                                                                      lane_identity, &carry);
        if (window.codepoint_count == 0) break; // defensive: no resolvable start (cannot happen for valid input)

        // The GB1 anchor at byte 0 of the first window is the open cluster's own start, not a new break: clear it so
        // the substrate drain (which has no zero-length-cluster guard) never emits an empty leading cluster.
        sz_u64_t boundary = base == 0 ? (window.boundary & ~1ull) : window.boundary;
        clusters = sz_utf8_rune_drain_forward_(boundary, base, lane_identity, cluster_starts, cluster_lengths, clusters,
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
#endif // SZ_USE_ICELAKE

#ifdef __cplusplus
}
#endif

#endif // STRINGZILLA_UTF8_GRAPHEMES_ICELAKE_H_
