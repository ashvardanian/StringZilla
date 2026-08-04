/**
 *  @brief Ice Lake backend for UTF-8 newline and whitespace delimiter scanning.
 *  @file include/stringzilla/utf8_tokens/icelake.h
 *  @author Ash Vardanian
 */
#ifndef STRINGZILLA_UTF8_TOKENS_ICELAKE_H_
#define STRINGZILLA_UTF8_TOKENS_ICELAKE_H_

#include "stringzilla/types.h"
#include "stringzilla/utf8_tokens/serial.h"
#include "stringzilla/utf8_runes/icelake.h" // shared lane-identity + peel substrate
#include "stringzilla/utf8_tokens/tables.h"

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

/*  Multistep newline / whitespace iteration (Ice Lake / AVX-512).
 *
 *  Each 64-byte window is classified branchlessly into a `starts` mask plus a per-lane byte-length vector,
 *  then `vpcompressb` peels the matching lanes and lengths. Starts are trusted in lanes [0,61] (step 62) so
 *  any 2-/3-byte delimiter is fully loaded; a `t[pos-1] == '\r'` carry suppresses an LF closing an edge CRLF. */
SZ_API_COMPTIME sz_size_t sz_utf8_newlines_icelake(     //
    sz_cptr_t text, sz_size_t length,                   //
    sz_size_t *match_offsets, sz_size_t *match_lengths, //
    sz_size_t matches_capacity, sz_size_t *bytes_consumed) {

    sz_u8_t const *text_u8 = (sz_u8_t const *)text;
    sz_size_t count = 0, position = 0;
    __m512i newline_u8x64 = _mm512_set1_epi8('\n'), vertical_tab_u8x64 = _mm512_set1_epi8('\v'),
            form_feed_u8x64 = _mm512_set1_epi8('\f'), carriage_return_u8x64 = _mm512_set1_epi8('\r'),
            lead_c2_u8x64 = _mm512_set1_epi8('\xC2'), x_85_u8x64 = _mm512_set1_epi8('\x85'),
            lead_e2_u8x64 = _mm512_set1_epi8('\xE2'), byte_80_u8x64 = _mm512_set1_epi8('\x80'),
            x_a8_u8x64 = _mm512_set1_epi8('\xA8'), x_a9_u8x64 = _mm512_set1_epi8('\xA9');
    __m512i const lane_identity_u8x64 = sz_utf8_lane_identity_icelake_();

    while (position < length && count < matches_capacity) {
        sz_size_t const valid_lanes = length - position;
        // Plain load + trust [0,61] on a full window (the hot path); a masked load clips the final partial window
        // to its valid lanes (zero-filled past `length`, so a truncated multi-byte delimiter cannot match).
        __m512i window_u8x64;
        sz_u64_t trusted_bits;
        if (valid_lanes >= 64) {
            window_u8x64 = _mm512_loadu_epi8(text_u8 + position), trusted_bits = (1ull << 62) - 1;
        }
        else {
            __mmask64 const load_mask_m64 = sz_u64_mask_until_(valid_lanes);
            window_u8x64 = _mm512_maskz_loadu_epi8(load_mask_m64, text_u8 + position),
            trusted_bits = _cvtmask64_u64(load_mask_m64);
        }
        __mmask64 newline_mask_m64 = _mm512_cmpeq_epi8_mask(window_u8x64, newline_u8x64);
        __mmask64 carriage_return_mask_m64 = _mm512_cmpeq_epi8_mask(window_u8x64, carriage_return_u8x64);
        __mmask64 one_byte_mask_m64 = _kor_mask64(
            _kor_mask64(newline_mask_m64, _mm512_cmpeq_epi8_mask(window_u8x64, vertical_tab_u8x64)),
            _kor_mask64(_mm512_cmpeq_epi8_mask(window_u8x64, form_feed_u8x64), carriage_return_mask_m64));
        // 2-byte NEL (C2 85); 3-byte LS/PS (E2 80 A8/A9) - computed unconditionally (cheaper than a data-dependent gate).
        __mmask64 nel_mask_m64 = _kand_mask64(_mm512_cmpeq_epi8_mask(window_u8x64, lead_c2_u8x64),
                                              _kshiftri_mask64(_mm512_cmpeq_epi8_mask(window_u8x64, x_85_u8x64), 1));
        __mmask64 lead_e280_mask_m64 = _kand_mask64(
            _mm512_cmpeq_epi8_mask(window_u8x64, lead_e2_u8x64),
            _kshiftri_mask64(_mm512_cmpeq_epi8_mask(window_u8x64, byte_80_u8x64), 1));
        __mmask64 line_para_mask_m64 = _kand_mask64(
            lead_e280_mask_m64, _kshiftri_mask64(_kor_mask64(_mm512_cmpeq_epi8_mask(window_u8x64, x_a8_u8x64),
                                                             _mm512_cmpeq_epi8_mask(window_u8x64, x_a9_u8x64)),
                                                 2));
        // CRLF: a CR whose next lane is LF is a single 2-byte match; its trailing LF must not also be emitted.
        __mmask64 crlf_mask_m64 = _kand_mask64(carriage_return_mask_m64, _kshiftri_mask64(newline_mask_m64, 1));
        __mmask64 lf_of_crlf_mask_m64 = _kand_mask64(newline_mask_m64, _kshiftli_mask64(carriage_return_mask_m64, 1));

        __mmask64 two_byte_starts_m64 = _kor_mask64(crlf_mask_m64, nel_mask_m64);
        __mmask64 three_byte_starts_m64 = line_para_mask_m64;
        __mmask64 starts_m64 = _kandn_mask64(
            lf_of_crlf_mask_m64, _kor_mask64(_kor_mask64(one_byte_mask_m64, nel_mask_m64), line_para_mask_m64));
        sz_u64_t start_bits = _cvtmask64_u64(starts_m64) & trusted_bits;
        // Suppress a leading LF already consumed by a CRLF that straddled the previous window edge.
        if (position != 0 && text_u8[position - 1] == '\r') start_bits &= ~(_cvtmask64_u64(newline_mask_m64) & 1ull);

        sz_size_t const window_matches = (sz_size_t)_mm_popcnt_u64(start_bits);
        sz_size_t const emit = sz_min_of_two(window_matches, matches_capacity - count);
        if (emit)
            sz_utf8_rune_peel_icelake_(start_bits, two_byte_starts_m64, three_byte_starts_m64, emit, position,
                                       lane_identity_u8x64, match_offsets + count, match_lengths + count);
        count += emit;
        if (count == matches_capacity) { // output buffer full: resume past the last emitted match
            position = match_offsets[count - 1] + match_lengths[count - 1];
            break;
        }
        position += valid_lanes >= 64 ? 62 : valid_lanes;
    }

    // If the loop stopped on the LF of an already-emitted CRLF, resume past it (the CR carried the length-2 match).
    if (position != 0 && position < length && text_u8[position - 1] == '\r' && text_u8[position] == '\n') ++position;
    if (bytes_consumed) *bytes_consumed = position;
    return count;
}

SZ_API_COMPTIME sz_size_t sz_utf8_whitespaces_icelake(  //
    sz_cptr_t text, sz_size_t length,                   //
    sz_size_t *match_offsets, sz_size_t *match_lengths, //
    sz_size_t matches_capacity, sz_size_t *bytes_consumed) {

    sz_u8_t const *text_u8 = (sz_u8_t const *)text;
    sz_size_t count = 0, position = 0;
    __m512i tab_u8x64 = _mm512_set1_epi8('\t'), carriage_return_u8x64 = _mm512_set1_epi8('\r'),
            x_20_u8x64 = _mm512_set1_epi8(' '), lead_c2_u8x64 = _mm512_set1_epi8('\xC2'),
            x_85_u8x64 = _mm512_set1_epi8('\x85'), x_a0_u8x64 = _mm512_set1_epi8('\xA0'),
            x_e1_u8x64 = _mm512_set1_epi8('\xE1'), lead_e2_u8x64 = _mm512_set1_epi8('\xE2'),
            x_e3_u8x64 = _mm512_set1_epi8('\xE3'), x_9a_u8x64 = _mm512_set1_epi8('\x9A'),
            byte_80_u8x64 = _mm512_set1_epi8('\x80'), x_81_u8x64 = _mm512_set1_epi8('\x81'),
            x_8a_u8x64 = _mm512_set1_epi8('\x8A'), x_a8_u8x64 = _mm512_set1_epi8('\xA8'),
            x_a9_u8x64 = _mm512_set1_epi8('\xA9'), x_af_u8x64 = _mm512_set1_epi8('\xAF'),
            x_9f_u8x64 = _mm512_set1_epi8('\x9F');
    __m512i const lane_identity_u8x64 = sz_utf8_lane_identity_icelake_();

    while (position < length && count < matches_capacity) {
        sz_size_t const valid_lanes = length - position;
        __m512i window_u8x64;
        sz_u64_t trusted_bits;
        if (valid_lanes >= 64) {
            window_u8x64 = _mm512_loadu_epi8(text_u8 + position), trusted_bits = (1ull << 62) - 1;
        }
        else {
            __mmask64 const load_mask_m64 = sz_u64_mask_until_(valid_lanes);
            window_u8x64 = _mm512_maskz_loadu_epi8(load_mask_m64, text_u8 + position),
            trusted_bits = _cvtmask64_u64(load_mask_m64);
        }
        // 1-byte: space, plus the contiguous range [\t, \r] == [9, 13].
        __mmask64 one_byte_mask_m64 = _kor_mask64(
            _mm512_cmpeq_epi8_mask(window_u8x64, x_20_u8x64),
            _kand_mask64(_mm512_cmpge_epu8_mask(window_u8x64, tab_u8x64),
                         _mm512_cmple_epu8_mask(window_u8x64, carriage_return_u8x64)));

        // 2-byte: C2 85 (NEL), C2 A0 (NBSP).
        __mmask64 lead_c2_mask_m64 = _mm512_cmpeq_epi8_mask(window_u8x64, lead_c2_u8x64);
        __mmask64 two_byte_starts_m64 = _kand_mask64(
            lead_c2_mask_m64, _kor_mask64(_kshiftri_mask64(_mm512_cmpeq_epi8_mask(window_u8x64, x_85_u8x64), 1),
                                          _kshiftri_mask64(_mm512_cmpeq_epi8_mask(window_u8x64, x_a0_u8x64), 1)));

        // 3-byte: E1 9A 80 (ogham); E2 80 [80-8A]; E2 80 AF; E2 81 9F; E2 80 A8/A9; E3 80 80.
        __mmask64 byte_80_mask_m64 = _mm512_cmpeq_epi8_mask(window_u8x64, byte_80_u8x64);
        __mmask64 lead_e2_mask_m64 = _mm512_cmpeq_epi8_mask(window_u8x64, lead_e2_u8x64);
        __mmask64 lead_e280_mask_m64 = _kand_mask64(lead_e2_mask_m64, _kshiftri_mask64(byte_80_mask_m64, 1));
        __mmask64 ogham_mask_m64 = _kand_mask64(
            _mm512_cmpeq_epi8_mask(window_u8x64, x_e1_u8x64),
            _kand_mask64(_kshiftri_mask64(_mm512_cmpeq_epi8_mask(window_u8x64, x_9a_u8x64), 1),
                         _kshiftri_mask64(byte_80_mask_m64, 2)));
        __mmask64 range_e280_mask_m64 = _kand_mask64(
            lead_e280_mask_m64, _kand_mask64(_kshiftri_mask64(_mm512_cmpge_epu8_mask(window_u8x64, byte_80_u8x64), 2),
                                             _kshiftri_mask64(_mm512_cmple_epu8_mask(window_u8x64, x_8a_u8x64), 2)));
        __mmask64 nnbsp_mask_m64 = _kand_mask64(lead_e280_mask_m64,
                                                _kshiftri_mask64(_mm512_cmpeq_epi8_mask(window_u8x64, x_af_u8x64), 2));
        __mmask64 mmsp_mask_m64 = _kand_mask64(
            _kand_mask64(lead_e2_mask_m64, _kshiftri_mask64(_mm512_cmpeq_epi8_mask(window_u8x64, x_81_u8x64), 1)),
            _kshiftri_mask64(_mm512_cmpeq_epi8_mask(window_u8x64, x_9f_u8x64), 2));
        __mmask64 line_mask_m64 = _kand_mask64(lead_e280_mask_m64,
                                               _kshiftri_mask64(_mm512_cmpeq_epi8_mask(window_u8x64, x_a8_u8x64), 2));
        __mmask64 para_mask_m64 = _kand_mask64(lead_e280_mask_m64,
                                               _kshiftri_mask64(_mm512_cmpeq_epi8_mask(window_u8x64, x_a9_u8x64), 2));
        __mmask64 ideographic_mask_m64 = _kand_mask64(
            _kand_mask64(_mm512_cmpeq_epi8_mask(window_u8x64, x_e3_u8x64), _kshiftri_mask64(byte_80_mask_m64, 1)),
            _kshiftri_mask64(byte_80_mask_m64, 2));
        __mmask64 three_byte_starts_m64 = _kor_mask64(
            _kor_mask64(_kor_mask64(ogham_mask_m64, range_e280_mask_m64), _kor_mask64(nnbsp_mask_m64, mmsp_mask_m64)),
            _kor_mask64(_kor_mask64(line_mask_m64, para_mask_m64), ideographic_mask_m64));

        __mmask64 starts_m64 = _kor_mask64(_kor_mask64(one_byte_mask_m64, two_byte_starts_m64), three_byte_starts_m64);
        sz_u64_t start_bits = _cvtmask64_u64(starts_m64) & trusted_bits;

        sz_size_t const window_matches = (sz_size_t)_mm_popcnt_u64(start_bits);
        sz_size_t const emit = sz_min_of_two(window_matches, matches_capacity - count);
        if (emit)
            sz_utf8_rune_peel_icelake_(start_bits, two_byte_starts_m64, three_byte_starts_m64, emit, position,
                                       lane_identity_u8x64, match_offsets + count, match_lengths + count);
        count += emit;
        if (count == matches_capacity) {
            position = match_offsets[count - 1] + match_lengths[count - 1];
            break;
        }
        position += valid_lanes >= 64 ? 62 : valid_lanes;
    }

    if (bytes_consumed) *bytes_consumed = position;
    return count;
}

#pragma region Membership

/** @brief  Per-lane single-bit test `(bitmap_byte_u8x64 >> (low_u8x64 & 7)) & 1` over all 64 lanes, as a mask. */
SZ_HELPER_INLINE __mmask64 sz_delimiter_test_bit_icelake_(__m512i bitmap_byte_u8x64, __m512i low_u8x64) {
    __m512i const bit_table_u8x64 = _mm512_broadcast_i32x4(_mm_setr_epi8( //
        1, 2, 4, 8, 16, 32, 64, (char)128, 0, 0, 0, 0, 0, 0, 0, 0));
    __m512i const bit_mask_u8x64 = _mm512_permutexvar_epi8(_mm512_and_si512(low_u8x64, _mm512_set1_epi8(0x07)),
                                                           bit_table_u8x64);
    return _mm512_test_epi8_mask(bitmap_byte_u8x64, bit_mask_u8x64);
}

/**
 *  @brief  BMP (codepoint < 0x10000) delimiter membership for every lane, resolved in-register.
 *
 *  `high` (cp >> 8, in [0,256)) selects a 32-byte bitmap row id through the aligned 256-entry `bmp_block` table via
 *  `vpermb` (`sz_utf8_rune_permute256_icelake_`); the bitmap byte at `row_id*32 + (low >> 3)` is read through the
 *  substrate page network; the bit `(low & 7)` is tested. ASCII lanes (high == 0) fall through naturally — block 0
 *  encodes the ASCII delimiters.
 */
SZ_HELPER_INLINE __m512i sz_delimiter_pack_chunks_epi8_icelake_(__m512i chunk0_u32x16, __m512i chunk1_u32x16,
                                                                __m512i chunk2_u32x16, __m512i chunk3_u32x16) {
    // Each chunk holds 16 byte-domain results in its low 16 32-bit lanes; place them in byte order [0,64).
    __m512i result_u8x64 = _mm512_castsi128_si512(_mm512_cvtepi32_epi8(chunk0_u32x16));
    result_u8x64 = _mm512_mask_expand_epi8(result_u8x64, _cvtu64_mask64((sz_u64_t)0xFFFFull << 16),
                                           _mm512_castsi128_si512(_mm512_cvtepi32_epi8(chunk1_u32x16)));
    result_u8x64 = _mm512_mask_expand_epi8(result_u8x64, _cvtu64_mask64((sz_u64_t)0xFFFFull << 32),
                                           _mm512_castsi128_si512(_mm512_cvtepi32_epi8(chunk2_u32x16)));
    result_u8x64 = _mm512_mask_expand_epi8(result_u8x64, _cvtu64_mask64((sz_u64_t)0xFFFFull << 48),
                                           _mm512_castsi128_si512(_mm512_cvtepi32_epi8(chunk3_u32x16)));
    return result_u8x64;
}

SZ_HELPER_AUTO __mmask64 sz_delimiter_bmp_membership_icelake_(__m512i window_u8x64, __m512i high_in_u8x64,
                                                              __m512i low_in_u8x64) {
    // The decode window only reconstructs `high`/`low` for 2-/3-byte leads; ASCII lanes (top bit clear) carry their
    // codepoint in the raw byte itself, so override them with (high=0, low=byte) before addressing the BMP tables.
    __mmask64 const ascii_m64 = ~_mm512_movepi8_mask(window_u8x64);
    __m512i const high_u8x64 = _mm512_maskz_mov_epi8(~ascii_m64, high_in_u8x64);
    __m512i const low_u8x64 = _mm512_mask_mov_epi8(low_in_u8x64, ascii_m64, window_u8x64);

    // Four 16-lane chunks of `high` index the aligned 256-entry block table; recombine byte-ordered.
    __m512i const high0_u32x16 = _mm512_cvtepu8_epi32(_mm512_castsi512_si128(high_u8x64));
    __m512i const high1_u32x16 = _mm512_cvtepu8_epi32(_mm512_extracti32x4_epi32(high_u8x64, 1));
    __m512i const high2_u32x16 = _mm512_cvtepu8_epi32(_mm512_extracti32x4_epi32(high_u8x64, 2));
    __m512i const high3_u32x16 = _mm512_cvtepu8_epi32(_mm512_extracti32x4_epi32(high_u8x64, 3));
    __m512i const block_id_u8x64 = sz_delimiter_pack_chunks_epi8_icelake_(
        sz_utf8_rune_permute256_icelake_(sz_utf8_delimiter_bmp_block_, high0_u32x16),
        sz_utf8_rune_permute256_icelake_(sz_utf8_delimiter_bmp_block_, high1_u32x16),
        sz_utf8_rune_permute256_icelake_(sz_utf8_delimiter_bmp_block_, high2_u32x16),
        sz_utf8_rune_permute256_icelake_(sz_utf8_delimiter_bmp_block_, high3_u32x16));

    // bit_index = block_id*32 + (low>>3), as 16-bit indices in [0, 1664); read through the page network.
    __m512i const nibble_u8x64 = _mm512_srli_epi16(_mm512_and_si512(low_u8x64, _mm512_set1_epi8((char)0xF8)), 3);
    __m512i const idx_lo_u16x32 = _mm512_add_epi16(
        _mm512_slli_epi16(
            _mm512_and_si512(_mm512_unpacklo_epi8(block_id_u8x64, _mm512_setzero_si512()), _mm512_set1_epi16(0x00FF)),
            5),
        _mm512_and_si512(_mm512_unpacklo_epi8(nibble_u8x64, _mm512_setzero_si512()), _mm512_set1_epi16(0x001F)));
    __m512i const idx_hi_u16x32 = _mm512_add_epi16(
        _mm512_slli_epi16(
            _mm512_and_si512(_mm512_unpackhi_epi8(block_id_u8x64, _mm512_setzero_si512()), _mm512_set1_epi16(0x00FF)),
            5),
        _mm512_and_si512(_mm512_unpackhi_epi8(nibble_u8x64, _mm512_setzero_si512()), _mm512_set1_epi16(0x001F)));
    __m512i const byte_lo_u16x32 = sz_utf8_rune_gather_byte_(sz_utf8_delimiter_bmp_bitmaps_,
                                                             sz_utf8_delimiter_bmp_bitmaps_count_k * 32, idx_lo_u16x32);
    __m512i const byte_hi_u16x32 = sz_utf8_rune_gather_byte_(sz_utf8_delimiter_bmp_bitmaps_,
                                                             sz_utf8_delimiter_bmp_bitmaps_count_k * 32, idx_hi_u16x32);
    __m512i const bitmap_byte_u8x64 = _mm512_packus_epi16(byte_lo_u16x32, byte_hi_u16x32);
    return sz_delimiter_test_bit_icelake_(bitmap_byte_u8x64, low_u8x64);
}

/**
 *  @brief  Astral (codepoint >= 0x10000) delimiter membership for the four-byte lanes, resolved in-register.
 *
 *  Reconstructs the full 21-bit codepoint per lane from the raw lead/continuation bytes, then walks the small astral
 *  network: `super = offset>>16` selects an L1 group, `group*256 + ((offset>>8)&0xFF)` selects a bitmap row id, and the
 *  bit `(offset & 7)` is tested. Resolved over all 64 lanes; the caller blends the result onto the four-byte lanes.
 */
SZ_HELPER_AUTO __mmask64 sz_delimiter_astral_membership_icelake_(__m512i window_u8x64, __m512i next1_u8x64,
                                                                 __m512i next2_u8x64, __m512i next3_u8x64) {
    __m512i const byte0_u8x64 = _mm512_and_si512(window_u8x64, _mm512_set1_epi8(0x07));
    __m512i const byte1_u8x64 = _mm512_and_si512(next1_u8x64, _mm512_set1_epi8(0x3F));
    __m512i const byte2_u8x64 = _mm512_and_si512(next2_u8x64, _mm512_set1_epi8(0x3F));
    __m512i const byte3_u8x64 = _mm512_and_si512(next3_u8x64, _mm512_set1_epi8(0x3F));
    __m512i const lane_identity_u8x64 = sz_utf8_lane_identity_icelake_();
    __m512i row_bytes_u8x64 = _mm512_setzero_si512();
    __m512i low_bytes_u8x64 = _mm512_setzero_si512();
    for (int chunk = 0; chunk < 4; ++chunk) {
        __m512i const select_u8x64 = _mm512_add_epi8(lane_identity_u8x64, _mm512_set1_epi8((char)(chunk * 16)));
        __m512i const lead_u32x16 = _mm512_cvtepu8_epi32(
            _mm512_castsi512_si128(_mm512_permutexvar_epi8(select_u8x64, byte0_u8x64)));
        __m512i const c1_u32x16 = _mm512_cvtepu8_epi32(
            _mm512_castsi512_si128(_mm512_permutexvar_epi8(select_u8x64, byte1_u8x64)));
        __m512i const c2_u32x16 = _mm512_cvtepu8_epi32(
            _mm512_castsi512_si128(_mm512_permutexvar_epi8(select_u8x64, byte2_u8x64)));
        __m512i const c3_u32x16 = _mm512_cvtepu8_epi32(
            _mm512_castsi512_si128(_mm512_permutexvar_epi8(select_u8x64, byte3_u8x64)));
        __m512i const codepoint_u32x16 = _mm512_or_si512(
            _mm512_or_si512(_mm512_slli_epi32(lead_u32x16, 18), _mm512_slli_epi32(c1_u32x16, 12)),
            _mm512_or_si512(_mm512_slli_epi32(c2_u32x16, 6), c3_u32x16));
        __m512i const offset_u32x16 = _mm512_sub_epi32(codepoint_u32x16, _mm512_set1_epi32(0x10000));
        __m512i const super_u32x16 = _mm512_and_si512(_mm512_srli_epi32(offset_u32x16, 16), _mm512_set1_epi32(0x0F));
        __m512i const group_u32x16 = sz_utf8_rune_lut_cascade_icelake_(sz_utf8_delimiter_astral_l1_, (16 + 63) / 64,
                                                                       super_u32x16);
        __m512i const sub_u32x16 = _mm512_and_si512(_mm512_srli_epi32(offset_u32x16, 8), _mm512_set1_epi32(0xFF));
        __m512i const l2_index_u32x16 = _mm512_add_epi32(_mm512_slli_epi32(group_u32x16, 8), sub_u32x16);
        __m512i const row_u32x16 = sz_utf8_rune_lut_cascade_icelake_(
            sz_utf8_delimiter_astral_l2_, (sz_utf8_delimiter_astral_groups_count_k * 256 + 63) / 64, l2_index_u32x16);
        __m512i const low8_u32x16 = _mm512_and_si512(offset_u32x16, _mm512_set1_epi32(0xFF));
        __m128i const row_packed_u8x16 = _mm512_cvtepi32_epi8(row_u32x16);
        __m128i const low_packed_u8x16 = _mm512_cvtepi32_epi8(low8_u32x16);
        __mmask64 const here_m64 = _cvtu64_mask64((sz_u64_t)0xFFFFull << (chunk * 16));
        row_bytes_u8x64 = _mm512_mask_mov_epi8(
            row_bytes_u8x64, here_m64, _mm512_maskz_expand_epi8(here_m64, _mm512_castsi128_si512(row_packed_u8x16)));
        low_bytes_u8x64 = _mm512_mask_mov_epi8(
            low_bytes_u8x64, here_m64, _mm512_maskz_expand_epi8(here_m64, _mm512_castsi128_si512(low_packed_u8x16)));
    }
    __m512i const nibble_u8x64 = _mm512_srli_epi16(_mm512_and_si512(low_bytes_u8x64, _mm512_set1_epi8((char)0xF8)), 3);
    __m512i const idx_lo_u16x32 = _mm512_add_epi16(
        _mm512_slli_epi16(
            _mm512_and_si512(_mm512_unpacklo_epi8(row_bytes_u8x64, _mm512_setzero_si512()), _mm512_set1_epi16(0x00FF)),
            5),
        _mm512_and_si512(_mm512_unpacklo_epi8(nibble_u8x64, _mm512_setzero_si512()), _mm512_set1_epi16(0x001F)));
    __m512i const idx_hi_u16x32 = _mm512_add_epi16(
        _mm512_slli_epi16(
            _mm512_and_si512(_mm512_unpackhi_epi8(row_bytes_u8x64, _mm512_setzero_si512()), _mm512_set1_epi16(0x00FF)),
            5),
        _mm512_and_si512(_mm512_unpackhi_epi8(nibble_u8x64, _mm512_setzero_si512()), _mm512_set1_epi16(0x001F)));
    __m512i const byte_lo_u16x32 = sz_utf8_rune_gather_byte_(
        sz_utf8_delimiter_astral_bitmaps_, sz_utf8_delimiter_astral_bitmaps_count_k * 32, idx_lo_u16x32);
    __m512i const byte_hi_u16x32 = sz_utf8_rune_gather_byte_(
        sz_utf8_delimiter_astral_bitmaps_, sz_utf8_delimiter_astral_bitmaps_count_k * 32, idx_hi_u16x32);
    __m512i const bitmap_byte_u8x64 = _mm512_packus_epi16(byte_lo_u16x32, byte_hi_u16x32);
    return sz_delimiter_test_bit_icelake_(bitmap_byte_u8x64, low_bytes_u8x64);
}

/**
 *  @brief  Per-lane UTF-8 validity for codepoint-start lanes, mirroring `sz_rune_decode` exactly: a 2/3/4-byte lead is
 *          valid only when its continuation bytes are present (within the loaded span) and well-formed, and it is not
 *          overlong, a surrogate, or beyond U+10FFFF. Invalid leads are never reported (serial advances one byte and
 *          re-syncs, which never matches the cleared lane).
 */
SZ_HELPER_AUTO __mmask64 sz_delimiter_valid_starts_icelake_( //
    __m512i window_u8x64, __m512i next1_u8x64, __m512i next2_u8x64, __m512i next3_u8x64,
    sz_utf8_rune_window_t const *decoded) {
    __mmask64 const loaded_m64 = sz_u64_clamp_mask_until_(decoded->loaded);
    __m512i const continuation_pattern_u8x64 = _mm512_set1_epi8((char)0x80);
    __m512i const continuation_mask_u8x64 = _mm512_set1_epi8((char)0xC0);
    __mmask64 const c1_ok_m64 = _mm512_cmpeq_epi8_mask(_mm512_and_si512(next1_u8x64, continuation_mask_u8x64),
                                                       continuation_pattern_u8x64);
    __mmask64 const c2_ok_m64 = _mm512_cmpeq_epi8_mask(_mm512_and_si512(next2_u8x64, continuation_mask_u8x64),
                                                       continuation_pattern_u8x64);
    __mmask64 const c3_ok_m64 = _mm512_cmpeq_epi8_mask(_mm512_and_si512(next3_u8x64, continuation_mask_u8x64),
                                                       continuation_pattern_u8x64);

    __mmask64 const ascii_m64 = loaded_m64 & ~_mm512_movepi8_mask(window_u8x64);
    __mmask64 const two_m64 = decoded->two_byte_starts;
    __mmask64 const three_m64 = decoded->three_byte_starts;
    __mmask64 const four_m64 = decoded->four_byte_starts;

    // Spans must lie within the loaded window (so we never validate against a wrapped neighbour or read past loaded).
    __mmask64 const span2_m64 = sz_u64_clamp_mask_until_(decoded->loaded >= 1 ? decoded->loaded - 1 : 0);
    __mmask64 const span3_m64 = sz_u64_clamp_mask_until_(decoded->loaded >= 2 ? decoded->loaded - 2 : 0);
    __mmask64 const span4_m64 = sz_u64_clamp_mask_until_(decoded->loaded >= 3 ? decoded->loaded - 3 : 0);

    // 2-byte: lead >= 0xC2 (reject C0/C1 overlong); continuation ok.
    __mmask64 const lead_ge_c2_m64 = _mm512_cmp_epu8_mask(window_u8x64, _mm512_set1_epi8((char)0xC2), _MM_CMPINT_NLT);
    __mmask64 const two_ok_m64 = two_m64 & span2_m64 & c1_ok_m64 & lead_ge_c2_m64;

    // 3-byte: c1,c2 ok; not overlong (E0 with next1 < 0xA0); not surrogate (ED with next1 >= 0xA0).
    __mmask64 const lead_e0_m64 = _mm512_cmpeq_epi8_mask(window_u8x64, _mm512_set1_epi8((char)0xE0));
    __mmask64 const lead_ed_m64 = _mm512_cmpeq_epi8_mask(window_u8x64, _mm512_set1_epi8((char)0xED));
    __mmask64 const n1_lt_a0_m64 = _mm512_cmp_epu8_mask(next1_u8x64, _mm512_set1_epi8((char)0xA0), _MM_CMPINT_LT);
    __mmask64 const three_ok_m64 = three_m64 & span3_m64 & c1_ok_m64 & c2_ok_m64 & ~(lead_e0_m64 & n1_lt_a0_m64) &
                                   ~(lead_ed_m64 & ~n1_lt_a0_m64);

    // 4-byte: lead <= 0xF4; c1,c2,c3 ok; not overlong (F0 with next1 < 0x90); not > U+10FFFF (F4 with next1 >= 0x90).
    __mmask64 const lead_le_f4_m64 = _mm512_cmp_epu8_mask(window_u8x64, _mm512_set1_epi8((char)0xF4), _MM_CMPINT_LE);
    __mmask64 const lead_f0_m64 = _mm512_cmpeq_epi8_mask(window_u8x64, _mm512_set1_epi8((char)0xF0));
    __mmask64 const lead_f4_m64 = _mm512_cmpeq_epi8_mask(window_u8x64, _mm512_set1_epi8((char)0xF4));
    __mmask64 const n1_lt_90_m64 = _mm512_cmp_epu8_mask(next1_u8x64, _mm512_set1_epi8((char)0x90), _MM_CMPINT_LT);
    __mmask64 const four_ok_m64 = four_m64 & span4_m64 & c1_ok_m64 & c2_ok_m64 & c3_ok_m64 & lead_le_f4_m64 &
                                  ~(lead_f0_m64 & n1_lt_90_m64) & ~(lead_f4_m64 & ~n1_lt_90_m64);

    return (ascii_m64 | two_ok_m64 | three_ok_m64 | four_ok_m64);
}

#pragma endregion Membership

#pragma region Forward driver

/** @copydoc sz_utf8_delimiters */
SZ_API_COMPTIME sz_size_t sz_utf8_delimiters_icelake(   //
    sz_cptr_t text, sz_size_t length,                   //
    sz_size_t *match_offsets, sz_size_t *match_lengths, //
    sz_size_t matches_capacity, sz_size_t *bytes_consumed) {
    sz_u8_t const *const text_u8 = (sz_u8_t const *)text;
    __m512i const lane_identity_u8x64 = sz_utf8_lane_identity_icelake_();

    sz_size_t base = 0, count = 0;
    while (base < length && count < matches_capacity) {
        sz_utf8_rune_window_t const decoded = sz_utf8_rune_decode_window_icelake_(text_u8 + base, length - base,
                                                                                  lane_identity_u8x64);
        sz_size_t const loaded = decoded.loaded;
        __m512i const window_u8x64 = decoded.window;
        __m512i const next1_u8x64 = _mm512_permutexvar_epi8(_mm512_add_epi8(lane_identity_u8x64, _mm512_set1_epi8(1)),
                                                            window_u8x64);
        __m512i const next2_u8x64 = _mm512_permutexvar_epi8(_mm512_add_epi8(lane_identity_u8x64, _mm512_set1_epi8(2)),
                                                            window_u8x64);
        __m512i const next3_u8x64 = _mm512_permutexvar_epi8(_mm512_add_epi8(lane_identity_u8x64, _mm512_set1_epi8(3)),
                                                            window_u8x64);

        // Effective-window trim: a multi-byte lead near the 64-byte edge whose span runs past `loaded` would decode
        // against a wrapped neighbour; defer it to the next window.
        sz_size_t byte_span = loaded;
        if (loaded >= 64) {
            sz_u64_t const overrun =
                (_cvtmask64_u64(decoded.two_byte_starts) & ~sz_u64_mask_until_serial_(loaded - 1)) |
                (_cvtmask64_u64(decoded.three_byte_starts) & ~sz_u64_mask_until_serial_(loaded - 2)) |
                (_cvtmask64_u64(decoded.four_byte_starts) & ~sz_u64_mask_until_serial_(loaded - 3));
            byte_span = overrun ? (sz_size_t)_tzcnt_u64(overrun) : loaded;
        }
        __mmask64 const span_mask_m64 = _cvtu64_mask64(sz_u64_mask_until_serial_(byte_span));

        __mmask64 const valid_starts_m64 = sz_delimiter_valid_starts_icelake_(window_u8x64, next1_u8x64, next2_u8x64,
                                                                              next3_u8x64, &decoded) &
                                           decoded.codepoint_starts & span_mask_m64;

        // BMP membership for every lane; astral membership blended onto the four-byte lanes (gated on presence).
        __mmask64 member_m64 = sz_delimiter_bmp_membership_icelake_(window_u8x64, decoded.high, decoded.low);
        __mmask64 const four_byte_m64 = decoded.four_byte_starts & span_mask_m64;
        if (four_byte_m64) {
            __mmask64 const astral_member_m64 = sz_delimiter_astral_membership_icelake_(window_u8x64, next1_u8x64,
                                                                                        next2_u8x64, next3_u8x64);
            member_m64 = (member_m64 & ~four_byte_m64) | (astral_member_m64 & four_byte_m64);
        }

        sz_u64_t hits = _cvtmask64_u64(member_m64 & valid_starts_m64);
        sz_u64_t const two_byte_u64 = _cvtmask64_u64(decoded.two_byte_starts);
        sz_u64_t const three_byte_u64 = _cvtmask64_u64(decoded.three_byte_starts);
        sz_u64_t const four_byte_u64 = _cvtmask64_u64(decoded.four_byte_starts);
        while (hits && count < matches_capacity) {
            sz_size_t const lane = (sz_size_t)_tzcnt_u64(hits);
            hits &= hits - 1;
            sz_size_t length_at_lane = 1;
            length_at_lane += (two_byte_u64 >> lane) & 1;
            length_at_lane += ((three_byte_u64 >> lane) & 1) * 2;
            length_at_lane += ((four_byte_u64 >> lane) & 1) * 3;
            match_offsets[count] = base + lane, match_lengths[count] = length_at_lane, ++count;
        }
        // Output buffer full: resume past the last emitted match, never at the window edge. The window's last hit
        // can fill the capacity exactly, and the undelimited bytes between that match and the edge still belong to
        // the caller's next segment, whose start is derived from `bytes_consumed`.
        if (count == matches_capacity) {
            base = match_offsets[count - 1] + match_lengths[count - 1];
            if (bytes_consumed) *bytes_consumed = base;
            return count;
        }
        base += byte_span ? byte_span : 1;
    }

    if (bytes_consumed) *bytes_consumed = base;
    return count;
}

#pragma endregion Forward driver

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#endif // SZ_USE_ICELAKE

#ifdef __cplusplus
}
#endif

#endif // STRINGZILLA_UTF8_TOKENS_ICELAKE_H_
