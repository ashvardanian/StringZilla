/**
 *  @brief Ice Lake backend for UAX-29 grapheme cluster boundaries.
 *  @file include/stringzilla/utf8_graphemes/icelake.h
 *  @author Ash Vardanian
 */
#ifndef STRINGZILLA_UTF8_GRAPHEMES_ICELAKE_H_
#define STRINGZILLA_UTF8_GRAPHEMES_ICELAKE_H_

#include "stringzilla/types.h"
#include "stringzilla/utf8_graphemes/tables.h"
#include "stringzilla/utf8_graphemes/serial.h"
#include "stringzilla/utf8_codepoints/icelake.h"

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

/*  UAX-29 Extended Grapheme Cluster boundaries (Ice Lake / AVX-512), fully vectorized end-to-end.
 *
 *  Every 64-byte window is decoded for all 64 lanes through the shared codepoint substrate
 *  (`sz_utf8_codepoints_decode_window_`), classified gather-free into one packed Grapheme_Cluster_Break
 *  descriptor per lane, compacted to a codepoint-dense descriptor vector with a single `vpcompressb`, and
 *  resolved by the branchless rule algebra. The three unbounded runs (GB12/13 Regional_Indicator parity,
 *  GB11 Extended_Pictographic-ZWJ chain, GB9c Indic conjunct) are threaded across windows by a small
 *  register carry, so no rule is ever re-walked scalar-wise and no boundary is deferred to the serial
 *  oracle. There is no per-lane scalar loop, no spill-to-stack-then-reload of the classifier, and no
 *  `vpgather` on any path: the descriptor trie is an in-register `vpermi2b` cascade and the astral range
 *  scan is a uniform broadcast-compare fold over all lanes. */

#pragma region Gather-free Grapheme_Cluster_Break classifier

enum {
    sz_grapheme_break_mid_tiles_k = (752 + 63) / 64,  /**< 64-byte ZMM tiles spanning the flattened `stage_mid`. */
    sz_grapheme_break_sub_tiles_k = (3152 + 63) / 64, /**< 64-byte ZMM tiles spanning the flattened `stage_sub`. */
};

/** @brief  Register-resident images of the in-tree gather-free Grapheme_Cluster_Break LUTs, built once per call. */
typedef struct sz_grapheme_luts_t {
    __m512i stage_hi[4];                            /**< 256 B `sz_utf8_grapheme_break_stage_hi_` split into 4 ZMM. */
    __m512i stage_mid[sz_grapheme_break_mid_tiles_k]; /**< Flattened `stage_mid[mid*16 + nibble]`. */
    __m512i stage_sub[sz_grapheme_break_sub_tiles_k]; /**< Flattened `stage_sub[sub*16 + nibble]`. */
    __m512i id_to_desc;                            /**< 18 descriptor bytes for the final `vpermb`. */
} sz_grapheme_luts_t;

/** @brief  Load the in-tree Grapheme_Cluster_Break tables into ZMM tiles (zero-padding the partial tail tiles). */
SZ_INTERNAL void sz_grapheme_luts_init_(sz_grapheme_luts_t *luts) {
    for (int quarter = 0; quarter < 4; ++quarter)
        luts->stage_hi[quarter] = _mm512_loadu_si512(sz_utf8_grapheme_break_stage_hi_ + quarter * 64);

    sz_align_(64) sz_u8_t flat_mid[sz_grapheme_break_mid_tiles_k * 64];
    for (int byte = 0; byte < sz_grapheme_break_mid_tiles_k * 64; ++byte)
        flat_mid[byte] = byte < 752 ? sz_utf8_grapheme_break_stage_mid_[byte] : (sz_u8_t)0;
    for (int tile = 0; tile < sz_grapheme_break_mid_tiles_k; ++tile)
        luts->stage_mid[tile] = _mm512_load_si512(flat_mid + tile * 64);

    sz_align_(64) sz_u8_t flat_sub[sz_grapheme_break_sub_tiles_k * 64];
    for (int byte = 0; byte < sz_grapheme_break_sub_tiles_k * 64; ++byte)
        flat_sub[byte] = byte < 3152 ? sz_utf8_grapheme_break_stage_sub_[byte] : (sz_u8_t)0;
    for (int tile = 0; tile < sz_grapheme_break_sub_tiles_k; ++tile)
        luts->stage_sub[tile] = _mm512_load_si512(flat_sub + tile * 64);

    sz_align_(64) sz_u8_t descriptors[64];
    for (int byte = 0; byte < 64; ++byte) descriptors[byte] = byte < 18 ? sz_utf8_grapheme_break_id_to_desc_[byte] : 0;
    luts->id_to_desc = _mm512_load_si512(descriptors);
}

/**
 *  @brief  Gather-free indexed read of a register-resident byte LUT held in @p tile_count tiles of 64 bytes, with a
 *          per-32-bit-lane byte index in [0, tile_count*64). A `vpermi2b` cascade: each ZMM pair covers 128 byte
 *          slots addressed by the low 7 bits; the high bits select the pair via masked blends. No `vpgatherdd`.
 */
SZ_INTERNAL __m512i sz_grapheme_permi2b_lut_(__m512i const *tiles, int tile_count, __m512i index_dwords) {
    __m512i const within = _mm512_and_si512(index_dwords, _mm512_set1_epi32(0x7F));
    __m512i const selector = _mm512_srli_epi32(index_dwords, 7);
    __m512i result = _mm512_setzero_si512();
    int const pairs = (tile_count + 1) / 2;
    for (int pair = 0; pair < pairs; ++pair) {
        __m512i const low_tile = tiles[pair * 2];
        __m512i const high_tile = (pair * 2 + 1 < tile_count) ? tiles[pair * 2 + 1] : _mm512_setzero_si512();
        __m512i const picked = _mm512_permutex2var_epi8(low_tile, within, high_tile);
        __mmask16 const here = _mm512_cmpeq_epi32_mask(selector, _mm512_set1_epi32(pair));
        result = _mm512_mask_blend_epi32(here, result, _mm512_and_si512(picked, _mm512_set1_epi32(0xFF)));
    }
    return result;
}

/**
 *  @brief  Classify 16 codepoints (one u32 ZMM) into 16 packed Grapheme_Cluster_Break descriptors (one per low byte).
 *
 *  Hangul U+AC00..U+D7A3 resolves by the LV/LVT residue formula (zero table). Every other BMP codepoint walks the
 *  three-stage trie: `stage_hi` (256 B) by `vpermb` over 4 resident ZMM, then `stage_mid` (752 B) and `stage_sub`
 *  (3152 B) by `vpermi2b` cascades, then the 18-byte `id_to_desc` `vpermb`. Astral codepoints route to the uniform
 *  range scan only when present in the block. No `vpgatherdd` anywhere on this path.
 */
SZ_INTERNAL __m512i sz_grapheme_classify16_(__m512i codepoints, sz_grapheme_luts_t const *luts) {
    __m512i const hangul_base = _mm512_set1_epi32(0xAC00);
    __mmask16 const is_hangul = _kand_mask16(_mm512_cmpge_epu32_mask(codepoints, hangul_base),
                                             _mm512_cmple_epu32_mask(codepoints, _mm512_set1_epi32(0xD7A3)));
    __m512i const hangul_relative = _mm512_sub_epi32(codepoints, hangul_base);
    /* relative < 11172, so a mul-hi reciprocal for `/ 28` is exact (2342 = ceil(2^16 / 28)). */
    __m512i const quotient = _mm512_srli_epi32(_mm512_mullo_epi32(hangul_relative, _mm512_set1_epi32(2342)), 16);
    __m512i const remainder = _mm512_sub_epi32(hangul_relative, _mm512_mullo_epi32(quotient, _mm512_set1_epi32(28)));
    __mmask16 const is_lv = _mm512_cmpeq_epi32_mask(remainder, _mm512_setzero_si512());
    __m512i const hangul_descriptor = _mm512_mask_blend_epi32(
        is_lv, _mm512_set1_epi32((int)sz_grapheme_break_hangul_lvt_k),
        _mm512_set1_epi32((int)sz_grapheme_break_hangul_lv_k));

    __mmask16 const is_bmp = _mm512_cmplt_epu32_mask(codepoints, _mm512_set1_epi32(0x10000));

    /* stage_hi[cp >> 8]: a 256-byte `vpermb` over 4 resident ZMM, addressed per 32-bit lane by the high byte. */
    __m512i const high_byte = _mm512_and_si512(_mm512_srli_epi32(codepoints, 8), _mm512_set1_epi32(0xFF));
    __m512i const high_low_six = _mm512_and_si512(high_byte, _mm512_set1_epi32(0x3F));
    __m512i const high_top_two = _mm512_and_si512(high_byte, _mm512_set1_epi32(0xC0));
    __m512i const permuted0 = _mm512_permutexvar_epi8(high_low_six, luts->stage_hi[0]);
    __m512i const permuted1 = _mm512_permutexvar_epi8(high_low_six, luts->stage_hi[1]);
    __m512i const permuted2 = _mm512_permutexvar_epi8(high_low_six, luts->stage_hi[2]);
    __m512i const permuted3 = _mm512_permutexvar_epi8(high_low_six, luts->stage_hi[3]);
    __mmask16 const quadrant1 = _mm512_cmpeq_epi32_mask(high_top_two, _mm512_set1_epi32(0x40));
    __mmask16 const quadrant2 = _mm512_cmpeq_epi32_mask(high_top_two, _mm512_set1_epi32(0x80));
    __mmask16 const quadrant3 = _mm512_cmpeq_epi32_mask(high_top_two, _mm512_set1_epi32(0xC0));
    __m512i mid = _mm512_and_si512(permuted0, _mm512_set1_epi32(0xFF));
    mid = _mm512_mask_blend_epi32(quadrant1, mid, _mm512_and_si512(permuted1, _mm512_set1_epi32(0xFF)));
    mid = _mm512_mask_blend_epi32(quadrant2, mid, _mm512_and_si512(permuted2, _mm512_set1_epi32(0xFF)));
    mid = _mm512_mask_blend_epi32(quadrant3, mid, _mm512_and_si512(permuted3, _mm512_set1_epi32(0xFF)));

    /* stage_mid[mid][(cp >> 4) & 0xF]: flattened index mid*16 + nibble, via the `vpermi2b` cascade. */
    __m512i const mid_nibble = _mm512_and_si512(_mm512_srli_epi32(codepoints, 4), _mm512_set1_epi32(0xF));
    __m512i const mid_index = _mm512_add_epi32(_mm512_slli_epi32(mid, 4), mid_nibble);
    __m512i const sub = sz_grapheme_permi2b_lut_(luts->stage_mid, sz_grapheme_break_mid_tiles_k, mid_index);

    /* stage_sub[sub][cp & 0xF]: flattened index sub*16 + nibble, via the `vpermi2b` cascade. */
    __m512i const sub_nibble = _mm512_and_si512(codepoints, _mm512_set1_epi32(0xF));
    __m512i const sub_index = _mm512_add_epi32(_mm512_slli_epi32(sub, 4), sub_nibble);
    __m512i const descriptor_index = sz_grapheme_permi2b_lut_(luts->stage_sub, sz_grapheme_break_sub_tiles_k, sub_index);

    __m512i const bmp_descriptor = _mm512_and_si512(_mm512_permutexvar_epi8(descriptor_index, luts->id_to_desc),
                                                    _mm512_set1_epi32(0xFF));

    /* Astral (cp >= 0x10000): a uniform 16-lane scan over the sorted range table. Each iteration broadcasts one
     * range's [lo, hi] and folds matching lanes; the descriptor index closes out through the resident `id_to_desc`
     * `vpermb`. Entered only when the block actually carries an astral lane (a whole-vector skip, never per-lane). */
    __mmask16 const is_astral = _kandn_mask16(is_bmp,
                                              _mm512_cmplt_epu32_mask(codepoints, _mm512_set1_epi32((int)0x110000)));
    __m512i astral_descriptor = bmp_descriptor;
    if (is_astral) {
        __m512i identifier = _mm512_setzero_si512();
        __mmask16 found = 0;
        for (sz_size_t range = 0; range < sz_utf8_grapheme_break_astral_count_k; ++range) {
            __mmask16 const at_least =
                _mm512_cmpge_epu32_mask(codepoints, _mm512_set1_epi32((int)sz_utf8_grapheme_break_astral_lo_[range]));
            __mmask16 const at_most =
                _mm512_cmple_epu32_mask(codepoints, _mm512_set1_epi32((int)sz_utf8_grapheme_break_astral_hi_[range]));
            __mmask16 const hit = _kandn_mask16(found, _kand_mask16(at_least, at_most));
            identifier = _mm512_mask_mov_epi32(identifier, hit,
                                               _mm512_set1_epi32(sz_utf8_grapheme_break_astral_id_[range]));
            found = _kor_mask16(found, hit);
        }
        __m512i const descriptor = _mm512_and_si512(_mm512_permutexvar_epi8(identifier, luts->id_to_desc),
                                                    _mm512_set1_epi32(0xFF));
        astral_descriptor = _mm512_maskz_mov_epi32(found, descriptor);
    }

    __m512i const bmp_or_astral = _mm512_mask_blend_epi32(is_bmp, astral_descriptor, bmp_descriptor);
    return _mm512_mask_blend_epi32(is_hangul, bmp_or_astral, hangul_descriptor);
}

/**
 *  @brief  Classify all 64 lanes of a decoded window into one packed Grapheme_Cluster_Break descriptor per lane.
 *
 *  The substrate decode already holds `high = cp >> 8` / `low = cp & 0xFF` for the BMP path and the raw bytes for
 *  astral reconstruction. Each 16-lane quarter is widened in-register to a full 21-bit codepoint (the astral quarters
 *  reassemble plane/mid/lo from the four UTF-8 bytes), classified through @ref sz_grapheme_classify16_, and the four
 *  descriptor quarters are written back as one byte per lane. No scalar per-lane loop, no `vpgather`, no spill.
 */
/**
 *  @brief  Widen one 16-lane quarter to full 21-bit codepoints and classify it to packed descriptors. The astral
 *          (4-byte) lanes in the quarter take the reconstructed plane/mid/lo codepoint; all others take the BMP
 *          `(high << 8) | low`. Returns 16 descriptors in the low byte of each 32-bit lane.
 */
SZ_INTERNAL __m512i sz_grapheme_classify_quarter_( //
    __m128i high_slice, __m128i low_slice, __m128i plane_slice, __m128i mid_slice, __m128i lo_slice,
    __mmask16 astral_quarter, sz_grapheme_luts_t const *luts) {
    __m512i const codepoint_bmp = _mm512_or_si512(_mm512_slli_epi32(_mm512_cvtepu8_epi32(high_slice), 8),
                                                  _mm512_cvtepu8_epi32(low_slice));
    __m512i const codepoint_astral = _mm512_or_si512(
        _mm512_or_si512(_mm512_slli_epi32(_mm512_cvtepu8_epi32(plane_slice), 16),
                        _mm512_slli_epi32(_mm512_cvtepu8_epi32(mid_slice), 8)),
        _mm512_cvtepu8_epi32(lo_slice));
    __m512i const codepoint = _mm512_mask_blend_epi32(astral_quarter, codepoint_bmp, codepoint_astral);
    return sz_grapheme_classify16_(codepoint, luts);
}

/**
 *  @brief  Classify all 64 lanes of a decoded window into one packed Grapheme_Cluster_Break descriptor per lane.
 *
 *  The substrate decode already holds `high = cp >> 8` / `low = cp & 0xFF` for the BMP path and the raw bytes for
 *  astral reconstruction. Each 16-lane quarter is widened in-register to a full 21-bit codepoint (the astral lanes
 *  reassemble plane/mid/lo from the four UTF-8 bytes) and classified; the four descriptor quarters are written back
 *  as one byte per lane. No scalar per-lane loop, no `vpgather`, no spill round-trip.
 */
SZ_INTERNAL __m512i sz_grapheme_classify_window_( //
    sz_utf8_codepoints_window_t const *decoded, __m512i next1, __m512i next2, __m512i next3,
    sz_grapheme_luts_t const *luts) {
    /* Astral (4-byte) lead reconstruction: plane = ((b0 & 7) << 2) | ((b1 >> 4) & 3); mid = ((b1 & F) << 4) |
     * ((b2 >> 2) & F); lo = ((b2 & 3) << 6) | (b3 & 3F); cp = (plane << 16) | (mid << 8) | lo. */
    __m512i const window = decoded->window;
    /* The substrate `high`/`low` are exact for 2-/3-byte BMP leads but undefined on 1-byte (ASCII) lanes (no
     * lead-length mask selected them). Force ASCII lanes to the identity codepoint `(0, raw byte)` so control
     * characters (CR/LF/Control) classify correctly. */
    __mmask64 const ascii = _mm512_cmplt_epu8_mask(window, _mm512_set1_epi8((char)0x80));
    __m512i const high = _mm512_maskz_mov_epi8(_knot_mask64(ascii), decoded->high);
    __m512i const low = _mm512_mask_blend_epi8(ascii, decoded->low, window);
    __m512i const plane = _mm512_or_si512(_mm512_slli_epi16(_mm512_and_si512(window, _mm512_set1_epi8(0x07)), 2),
                                          sz_utf8_codepoints_srl8_(next1, 4, 0x03));
    __m512i const mid_byte = _mm512_or_si512(_mm512_slli_epi16(_mm512_and_si512(next1, _mm512_set1_epi8(0x0F)), 4),
                                             sz_utf8_codepoints_srl8_(next2, 2, 0x0F));
    __m512i const lo_byte = _mm512_or_si512(_mm512_slli_epi16(_mm512_and_si512(next2, _mm512_set1_epi8(0x03)), 6),
                                            _mm512_and_si512(next3, _mm512_set1_epi8(0x3F)));
    sz_u64_t const four_byte = _cvtmask64_u64(decoded->four_byte_starts);

    __m512i const quarter0 =
        sz_grapheme_classify_quarter_(_mm512_extracti32x4_epi32(high, 0), _mm512_extracti32x4_epi32(low, 0),
                                      _mm512_extracti32x4_epi32(plane, 0), _mm512_extracti32x4_epi32(mid_byte, 0),
                                      _mm512_extracti32x4_epi32(lo_byte, 0), (__mmask16)(four_byte & 0xFFFF), luts);
    __m512i const quarter1 =
        sz_grapheme_classify_quarter_(_mm512_extracti32x4_epi32(high, 1), _mm512_extracti32x4_epi32(low, 1),
                                      _mm512_extracti32x4_epi32(plane, 1), _mm512_extracti32x4_epi32(mid_byte, 1),
                                      _mm512_extracti32x4_epi32(lo_byte, 1), (__mmask16)((four_byte >> 16) & 0xFFFF),
                                      luts);
    __m512i const quarter2 =
        sz_grapheme_classify_quarter_(_mm512_extracti32x4_epi32(high, 2), _mm512_extracti32x4_epi32(low, 2),
                                      _mm512_extracti32x4_epi32(plane, 2), _mm512_extracti32x4_epi32(mid_byte, 2),
                                      _mm512_extracti32x4_epi32(lo_byte, 2), (__mmask16)((four_byte >> 32) & 0xFFFF),
                                      luts);
    __m512i const quarter3 =
        sz_grapheme_classify_quarter_(_mm512_extracti32x4_epi32(high, 3), _mm512_extracti32x4_epi32(low, 3),
                                      _mm512_extracti32x4_epi32(plane, 3), _mm512_extracti32x4_epi32(mid_byte, 3),
                                      _mm512_extracti32x4_epi32(lo_byte, 3), (__mmask16)((four_byte >> 48) & 0xFFFF),
                                      luts);

    /* Re-pack the four 16-lane descriptor quarters (each in the low byte of its 32-bit lanes) into one byte/lane. */
    __m512i const packed01 = _mm512_packus_epi32(_mm512_and_si512(quarter0, _mm512_set1_epi32(0xFF)),
                                                 _mm512_and_si512(quarter1, _mm512_set1_epi32(0xFF)));
    __m512i const packed23 = _mm512_packus_epi32(_mm512_and_si512(quarter2, _mm512_set1_epi32(0xFF)),
                                                 _mm512_and_si512(quarter3, _mm512_set1_epi32(0xFF)));
    __m512i const packed = _mm512_packus_epi16(packed01, packed23);
    /* `vpackus` interleaves the four 128-bit lanes as {q0a,q1a,q2a,q3a, q0b,...}; restore ascending lane order. */
    __m512i const restore = _mm512_setr_epi32(0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15);
    return _mm512_permutexvar_epi32(restore, packed);
}

#pragma endregion Gather-free Grapheme_Cluster_Break classifier

#pragma region Grapheme boundary algebra

/**
 *  @brief  Cross-window left-context carry: the situation just AFTER the previous window's last codepoint, so this
 *          window's codepoint 0 sees an exact (i-1) context for all of GB3..GB13.
 */
typedef struct sz_grapheme_carry_t {
    int has_previous;                /**< Stream has emitted at least one codepoint already. */
    sz_u64_t previous_class_bit[14]; /**< Bit 0 set => the previous codepoint had that gcb class. */
    int regional_indicator_run_odd;  /**< RI run ending at the previous codepoint has odd inclusive length. */
    int extended_pictographic_run;   /**< Previous codepoint tails an (ExtPict Extend*) chain. */
    int zero_width_joiner_connector; /**< Previous codepoint is a ZWJ closing an (ExtPict Extend*) run. */
    int indic_conjunct_open;         /**< An InCB run rooted at a Consonant is open at the previous cp. */
    int indic_conjunct_seen_linker;  /**< ... and a Linker has appeared in that open run. */
} sz_grapheme_carry_t;

/** @brief  A zero-initialized cross-window carry (stream start: no previous codepoint, no open runs). */
SZ_INTERNAL sz_grapheme_carry_t sz_grapheme_carry_empty_(void) {
    sz_grapheme_carry_t carry;
    carry.has_previous = 0;
    for (int class_index = 0; class_index < 14; ++class_index) carry.previous_class_bit[class_index] = 0;
    carry.regional_indicator_run_odd = 0;
    carry.extended_pictographic_run = 0;
    carry.zero_width_joiner_connector = 0;
    carry.indic_conjunct_open = 0;
    carry.indic_conjunct_seen_linker = 0;
    return carry;
}

/** @brief  Per-window class membership masks derived once from the codepoint-dense packed descriptor bytes. */
typedef struct sz_grapheme_window_masks_t {
    sz_u64_t class_bit[14]; /**< `class_bit[c]` bit i set => dense codepoint i is grapheme class c. */
    sz_u64_t extended_pictographic;
    sz_u64_t indic_consonant, indic_extend, indic_linker;
} sz_grapheme_window_masks_t;

/**
 *  @brief  Reduce the codepoint-dense packed descriptor bytes (in a ZMM) into per-class membership masks for the
 *          rule algebra. One `vpcmpeqb`/`kmov` per grapheme class plus the InCB / Extended_Pictographic bit tests.
 */
SZ_INTERNAL sz_grapheme_window_masks_t sz_grapheme_build_masks_(__m512i descriptors, sz_u64_t valid) {
    sz_grapheme_window_masks_t masks;
    __m512i const class_field = _mm512_and_si512(descriptors, _mm512_set1_epi8(0x0F));
    for (int class_index = 0; class_index < 14; ++class_index)
        masks.class_bit[class_index] =
            _cvtmask64_u64(_mm512_cmpeq_epi8_mask(class_field, _mm512_set1_epi8((char)class_index))) & valid;
    masks.extended_pictographic =
        _cvtmask64_u64(_mm512_test_epi8_mask(descriptors, _mm512_set1_epi8(0x40))) & valid;
    __m512i const incb_field = _mm512_and_si512(_mm512_srli_epi16(descriptors, 4), _mm512_set1_epi8(0x03));
    masks.indic_consonant =
        _cvtmask64_u64(_mm512_cmpeq_epi8_mask(incb_field, _mm512_set1_epi8((char)sz_grapheme_incb_consonant_k))) & valid;
    masks.indic_extend =
        _cvtmask64_u64(_mm512_cmpeq_epi8_mask(incb_field, _mm512_set1_epi8((char)sz_grapheme_incb_extend_k))) & valid;
    masks.indic_linker =
        _cvtmask64_u64(_mm512_cmpeq_epi8_mask(incb_field, _mm512_set1_epi8((char)sz_grapheme_incb_linker_k))) & valid;
    return masks;
}

/**
 *  @brief  The three unbounded run-carries (GB12/13 RI parity, GB11 ExtPict-ZWJ, GB9c InCB), computed once from this
 *          window's masks and the inbound carry. Returns their join-mask contribution and updates @p next for the
 *          following window. Bit i of the result set => a no-break suppression sits before dense codepoint i.
 */
SZ_INTERNAL sz_u64_t sz_grapheme_stateful_joins_(sz_grapheme_window_masks_t const *window, int codepoint_count,
                                                 sz_grapheme_carry_t const *previous, sz_grapheme_carry_t *next) {
    sz_u64_t const regional = window->class_bit[sz_grapheme_break_regional_indicator_k];
    sz_u64_t const extend = window->class_bit[sz_grapheme_break_extend_k];
    sz_u64_t const zero_width_joiner = window->class_bit[sz_grapheme_break_zwj_k];
    int const last = codepoint_count - 1;
    sz_u64_t const last_bit = (codepoint_count > 0) ? (1ull << last) : 0;
    sz_u64_t join = 0;

    /* GB12/13: no-break before a Regional_Indicator that is an EVEN member of its maximal run. A segmented XOR
     * parity scan over the run computes membership; the run is seeded from the inbound carry's RI parity. */
    {
        sz_u64_t const ri = regional;
        sz_u64_t const previous_ri =
            (ri << 1) | (previous->previous_class_bit[sz_grapheme_break_regional_indicator_k] & 1ull);
        sz_u64_t const starts = ri & ~previous_ri;
        sz_u64_t parity = ri;
        if ((ri & 1ull) && !(starts & 1ull)) parity ^= (previous->regional_indicator_run_odd ? 1ull : 0ull);
        sz_u64_t reach = ri & ~starts;
        for (int step = 1; step < 64; step <<= 1) {
            parity ^= (parity << step) & reach;
            reach &= (reach << step);
        }
        join |= ri & previous_ri & ~parity; /* parity[i]==0 => even member => pairs with i-1 => no break */
        next->regional_indicator_run_odd = (ri & last_bit) ? ((parity & last_bit) != 0) : 0;
    }

    /* GB11: Extended_Pictographic Extend* ZWJ x Extended_Pictographic, via a right-smear of the ExtPict run through
     * Extend, a ZWJ whose left lane sits inside that run, and a no-break before the following ExtPict. */
    {
        sz_u64_t run = window->extended_pictographic;
        if (previous->extended_pictographic_run && (extend & 1ull)) run |= 1ull;
        run = sz_utf8_codepoints_fill_right_(run, extend);
        sz_u64_t const previous_run = (run << 1) | (previous->extended_pictographic_run ? 1ull : 0ull);
        sz_u64_t const connector = zero_width_joiner & previous_run;
        sz_u64_t const previous_connector = (connector << 1) | (previous->zero_width_joiner_connector ? 1ull : 0ull);
        join |= previous_connector & window->extended_pictographic;
        next->extended_pictographic_run = (run & last_bit) != 0;
        next->zero_width_joiner_connector = (connector & last_bit) != 0;
    }

    /* GB9c: Consonant [Extend|Linker]* Linker [Extend|Linker]* x Consonant. The open run and the seen-linker run
     * both flood rightward through the continuation lanes (Extend|Linker), seeded from the inbound carry. */
    {
        sz_u64_t const continuation = window->indic_extend | window->indic_linker;
        sz_u64_t open = window->indic_consonant;
        if (previous->indic_conjunct_open && (continuation & 1ull)) open |= 1ull;
        open = sz_utf8_codepoints_fill_right_(open, continuation);
        sz_u64_t seen = window->indic_linker & open;
        if (previous->indic_conjunct_open && previous->indic_conjunct_seen_linker && (continuation & 1ull)) seen |= 1ull;
        seen = sz_utf8_codepoints_fill_right_(seen, continuation & open);
        sz_u64_t previous_open_seen = ((open & seen) << 1);
        if (previous->indic_conjunct_open && previous->indic_conjunct_seen_linker) previous_open_seen |= 1ull;
        join |= previous_open_seen & window->indic_consonant;
        int const last_in_run = ((window->indic_consonant | continuation) & last_bit) != 0 && (open & last_bit) != 0;
        next->indic_conjunct_open = last_in_run;
        next->indic_conjunct_seen_linker = last_in_run && ((seen & last_bit) != 0);
    }
    return join;
}

/**
 *  @brief  Per-class shift of @p current into the (i-1) position, seeding lane 0 from the inbound carry's last class.
 */
SZ_INTERNAL sz_u64_t sz_grapheme_previous_(sz_u64_t current, sz_grapheme_carry_t const *previous, int class_index) {
    return (current << 1) | (previous->previous_class_bit[class_index] & 1ull);
}

/**
 *  @brief  Boundary bitmask for one codepoint-dense window. Bit i set => a grapheme break sits BEFORE dense
 *          codepoint i. The carry-update is produced exactly once here and threaded out through @p carry.
 */
SZ_INTERNAL sz_u64_t sz_grapheme_window_boundaries_(__m512i descriptors, int codepoint_count,
                                                    sz_grapheme_carry_t *carry) {
    sz_u64_t const valid = (codepoint_count >= 64) ? ~0ull : ((1ull << codepoint_count) - 1);
    sz_grapheme_window_masks_t const window = sz_grapheme_build_masks_(descriptors, valid);
    sz_grapheme_carry_t const previous = *carry;

    sz_u64_t const carriage_return = window.class_bit[sz_grapheme_break_cr_k];
    sz_u64_t const line_feed = window.class_bit[sz_grapheme_break_lf_k];
    sz_u64_t const control = window.class_bit[sz_grapheme_break_control_k];
    sz_u64_t const extend = window.class_bit[sz_grapheme_break_extend_k];
    sz_u64_t const zero_width_joiner = window.class_bit[sz_grapheme_break_zwj_k];
    sz_u64_t const prepend = window.class_bit[sz_grapheme_break_prepend_k];
    sz_u64_t const spacing_mark = window.class_bit[sz_grapheme_break_spacingmark_k];
    sz_u64_t const hangul_l = window.class_bit[sz_grapheme_break_hangul_l_k];
    sz_u64_t const hangul_v = window.class_bit[sz_grapheme_break_hangul_v_k];
    sz_u64_t const hangul_t = window.class_bit[sz_grapheme_break_hangul_t_k];
    sz_u64_t const hangul_lv = window.class_bit[sz_grapheme_break_hangul_lv_k];
    sz_u64_t const hangul_lvt = window.class_bit[sz_grapheme_break_hangul_lvt_k];

    sz_u64_t const previous_cr = sz_grapheme_previous_(carriage_return, &previous, sz_grapheme_break_cr_k);
    sz_u64_t const previous_lf = sz_grapheme_previous_(line_feed, &previous, sz_grapheme_break_lf_k);
    sz_u64_t const previous_control = sz_grapheme_previous_(control, &previous, sz_grapheme_break_control_k);
    sz_u64_t const previous_prepend = sz_grapheme_previous_(prepend, &previous, sz_grapheme_break_prepend_k);
    sz_u64_t const previous_l = sz_grapheme_previous_(hangul_l, &previous, sz_grapheme_break_hangul_l_k);
    sz_u64_t const previous_v = sz_grapheme_previous_(hangul_v, &previous, sz_grapheme_break_hangul_v_k);
    sz_u64_t const previous_t = sz_grapheme_previous_(hangul_t, &previous, sz_grapheme_break_hangul_t_k);
    sz_u64_t const previous_lv = sz_grapheme_previous_(hangul_lv, &previous, sz_grapheme_break_hangul_lv_k);
    sz_u64_t const previous_lvt = sz_grapheme_previous_(hangul_lvt, &previous, sz_grapheme_break_hangul_lvt_k);

    sz_u64_t join = 0;
    join |= previous_cr & line_feed;                                    /* GB3  CR x LF */
    join |= previous_l & (hangul_l | hangul_v | hangul_lv | hangul_lvt); /* GB6 */
    join |= (previous_lv | previous_v) & (hangul_v | hangul_t);         /* GB7 */
    join |= (previous_lvt | previous_t) & hangul_t;                     /* GB8 */
    join |= (extend | zero_width_joiner);                              /* GB9  x (Extend|ZWJ) */
    join |= spacing_mark;                                              /* GB9a x SpacingMark */
    join |= previous_prepend;                                          /* GB9b Prepend x */

    join |= sz_grapheme_stateful_joins_(&window, codepoint_count, &previous, carry); /* GB9c, GB11, GB12/13 */

    /* GB4/GB5 force a break around Control/CR/LF (except CR x LF kept by GB3). */
    sz_u64_t const force = ((previous_control | previous_cr | previous_lf) | (control | carriage_return | line_feed)) &
                           ~(previous_cr & line_feed);

    sz_u64_t boundary = (~join | force) & valid;
    if (!previous.has_previous) boundary |= (valid & 1ull); /* GB1 at stream start */

    if (codepoint_count > 0) {
        sz_u64_t const last_bit = 1ull << (codepoint_count - 1);
        for (int class_index = 0; class_index < 14; ++class_index)
            carry->previous_class_bit[class_index] = (window.class_bit[class_index] & last_bit) ? 1ull : 0ull;
        carry->has_previous = 1;
    }
    return boundary;
}

#pragma endregion Grapheme boundary algebra

#pragma region Window classification driver

/**
 *  @brief  One byte window resolved to codepoint-dense boundary bits plus the byte geometry to map them back.
 *
 *  `boundary` bit j set => a grapheme break sits before the j-th resolved codepoint START in this window.
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
SZ_INTERNAL sz_grapheme_window_t sz_grapheme_classify_window_full_( //
    sz_u8_t const *text, sz_size_t length, sz_size_t base, __m512i lane_identity, sz_grapheme_luts_t const *luts,
    sz_grapheme_carry_t *carry) {

    sz_utf8_codepoints_window_t const decoded =
        sz_utf8_codepoints_decode_window_(text + base, length - base, lane_identity);
    sz_size_t const loaded = decoded.loaded;
    sz_u64_t start_lanes = _cvtmask64_u64(decoded.codepoint_starts);

    /* Effective-window trim: on a FULL window the trailing codepoint may straddle the 64-byte edge. Only the top
     * one or two starts can be truncated, so re-mask off any start whose multi-byte sequence runs past `loaded`.
     * The per-lane sequence length is read from the substrate's lead-length masks (no scalar classify). */
    sz_size_t byte_span = loaded;
    if (loaded >= 64) {
        sz_u64_t const four = _cvtmask64_u64(decoded.four_byte_starts);
        sz_u64_t const three = _cvtmask64_u64(decoded.three_byte_starts);
        sz_u64_t const two = _cvtmask64_u64(decoded.two_byte_starts);
        sz_u64_t probe = start_lanes;
        for (int attempt = 0; attempt < 4 && probe; ++attempt) {
            int const lane = 63 - sz_u64_clz(probe);
            int const sequence = ((four >> lane) & 1) ? 4 : ((three >> lane) & 1) ? 3 : ((two >> lane) & 1) ? 2 : 1;
            if ((sz_size_t)lane + (sz_size_t)sequence > loaded) {
                byte_span = (sz_size_t)lane;
                probe &= sz_u64_mask_until_((sz_size_t)lane);
            }
            else
                break;
        }
        start_lanes &= sz_u64_mask_until_(byte_span);
    }

    __m512i const next1 = _mm512_permutexvar_epi8(_mm512_add_epi8(lane_identity, _mm512_set1_epi8(1)), decoded.window);
    __m512i const next2 = _mm512_permutexvar_epi8(_mm512_add_epi8(lane_identity, _mm512_set1_epi8(2)), decoded.window);
    __m512i const next3 = _mm512_permutexvar_epi8(_mm512_add_epi8(lane_identity, _mm512_set1_epi8(3)), decoded.window);
    __m512i const descriptors_per_lane = sz_grapheme_classify_window_(&decoded, next1, next2, next3, luts);

    /* Compact the per-lane descriptors at codepoint-start lanes into the dense codepoint domain (one `vpcompressb`)
     * so the rule algebra runs on adjacent codepoints with no continuation-byte gaps. */
    __mmask64 const start_mask = _cvtu64_mask64(start_lanes);
    sz_grapheme_window_t result;
    result.codepoint_count = (sz_size_t)_mm_popcnt_u64(start_lanes);
    result.byte_span = byte_span;
    __m512i const dense_descriptors = _mm512_maskz_compress_epi8(start_mask, descriptors_per_lane);
    sz_u64_t const dense_boundary =
        sz_grapheme_window_boundaries_(dense_descriptors, (int)result.codepoint_count, carry);
    /* Scatter the dense boundary bits back to their codepoint-start byte lanes: `_pdep_u64` deposits the j-th dense
     * bit into the j-th set bit of `start_lanes`, recovering the byte-domain boundary mask the drains consume. */
    result.boundary = _pdep_u64(dense_boundary, start_lanes);
    return result;
}

#pragma endregion Window classification driver

#pragma region Grapheme forward driver

SZ_PUBLIC sz_size_t sz_utf8_graphemes_icelake( //
    sz_cptr_t text, sz_size_t length,                         //
    sz_size_t *cluster_starts, sz_size_t *cluster_lengths,    //
    sz_size_t clusters_capacity, sz_size_t *bytes_consumed) {

    sz_size_t clusters = 0;
    if (length == 0 || clusters_capacity == 0) {
        if (bytes_consumed) *bytes_consumed = 0;
        return 0;
    }
    sz_u8_t const *text_u8 = (sz_u8_t const *)text;
    sz_grapheme_luts_t luts;
    sz_grapheme_luts_init_(&luts);
    __m512i const lane_identity = sz_utf8_codepoints_lane_identity_icelake_();

    sz_grapheme_carry_t carry = sz_grapheme_carry_empty_();
    sz_size_t cluster_start = 0;
    sz_size_t base = 0;

    while (base < length) {
        sz_grapheme_window_t const window =
            sz_grapheme_classify_window_full_(text_u8, length, base, lane_identity, &luts, &carry);
        if (window.codepoint_count == 0) break; /* defensive: no resolvable start (cannot happen for valid input) */

        /* The GB1 anchor at byte 0 of the first window is the open cluster's own start, not a new break: clear it so
         * the substrate drain (which has no zero-length-cluster guard) never emits an empty leading cluster. */
        sz_u64_t boundary = base == 0 ? (window.boundary & ~1ull) : window.boundary;
        clusters = sz_utf8_codepoints_drain_forward_(boundary, base, lane_identity, cluster_starts, cluster_lengths,
                                                     clusters, clusters_capacity, &cluster_start);
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
