/**
 *  @brief Ice Lake (AVX-512 VBMI) backend for UTF-8 delimiter scanning.
 *  @file include/stringzilla/utf8_delimiters/icelake.h
 *  @author Ash Vardanian
 *
 *  Fully-vectorized, gather-free twin of @ref sz_find_delimiters_utf8_serial. Each 64-byte window is decoded once
 *  through the shared codepoint substrate (`sz_utf8_rune_decode_window_`), and EVERY codepoint-start lane's
 *  delimiter membership is resolved IN-REGISTER over the decoded `high`/`low` halves: the BMP block id via a `vpermb`
 *  256-entry lookup, the 32-byte bitmap byte via the substrate `vpermi2b` page network, and the per-lane bit test by a
 *  `vpermb` bit-mask. The 4-byte (astral) lanes reconstruct the full 21-bit codepoint and walk the small astral L1/L2
 *  network the same gather-free way; the whole astral block is gated on a 4-byte lead being present. Per-lane validity
 *  (overlong / surrogate / out-of-range / bad-continuation, mirroring `sz_rune_decode`) is computed as a mask and
 *  intersected with membership, so a malformed lead is never reported and the result is bit-identical to the serial
 *  oracle for the returned pointer and matched length. There is NO scalar per-codepoint membership and NO ASCII fast
 *  path with a scalar tail; no `vpgather` / `vpscatter` is ever issued.
 */
#ifndef STRINGZILLA_UTF8_DELIMITERS_ICELAKE_H_
#define STRINGZILLA_UTF8_DELIMITERS_ICELAKE_H_

#include "stringzilla/types.h"

#include "stringzilla/utf8_runes/icelake.h"
#include "stringzilla/utf8_delimiters/serial.h"
#include "stringzilla/utf8_delimiters/tables.h"

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

#pragma region Gather free membership

/** @brief  Per-lane single-bit test `(bitmap_byte >> (low & 7)) & 1` over all 64 lanes, returned as a mask. */
SZ_INTERNAL __mmask64 sz_delimiter_test_bit_icelake_(__m512i bitmap_byte, __m512i low) {
    __m512i const bit_table = _mm512_broadcast_i32x4(_mm_setr_epi8( //
        1, 2, 4, 8, 16, 32, 64, (char)128, 0, 0, 0, 0, 0, 0, 0, 0));
    __m512i const bit_mask = _mm512_permutexvar_epi8(_mm512_and_si512(low, _mm512_set1_epi8(0x07)), bit_table);
    return _mm512_test_epi8_mask(bitmap_byte, bit_mask);
}

/**
 *  @brief  BMP (codepoint < 0x10000) delimiter membership for every lane, gather-free.
 *
 *  `high` (cp >> 8, in [0,256)) selects a 32-byte bitmap row id through the aligned 256-entry `bmp_block` table via
 *  `vpermb` (`sz_utf8_rune_permute256_icelake_`); the bitmap byte at `row_id*32 + (low >> 3)` is read through the
 *  substrate page network; the bit `(low & 7)` is tested. ASCII lanes (high == 0) fall through naturally — block 0
 *  encodes the ASCII delimiters. No `vpgather`.
 */
SZ_INTERNAL __m512i sz_delimiter_pack_chunks_epi8_icelake_(__m512i chunk0, __m512i chunk1, __m512i chunk2,
                                                           __m512i chunk3) {
    // Each chunk holds 16 byte-domain results in its low 16 32-bit lanes; place them in byte order [0,64).
    __m512i result = _mm512_castsi128_si512(_mm512_cvtepi32_epi8(chunk0));
    result = _mm512_mask_expand_epi8(result, _cvtu64_mask64((sz_u64_t)0xFFFFull << 16),
                                     _mm512_castsi128_si512(_mm512_cvtepi32_epi8(chunk1)));
    result = _mm512_mask_expand_epi8(result, _cvtu64_mask64((sz_u64_t)0xFFFFull << 32),
                                     _mm512_castsi128_si512(_mm512_cvtepi32_epi8(chunk2)));
    result = _mm512_mask_expand_epi8(result, _cvtu64_mask64((sz_u64_t)0xFFFFull << 48),
                                     _mm512_castsi128_si512(_mm512_cvtepi32_epi8(chunk3)));
    return result;
}

SZ_INTERNAL __mmask64 sz_delimiter_bmp_membership_icelake_(__m512i window, __m512i high_in, __m512i low_in) {
    // The decode window only reconstructs `high`/`low` for 2-/3-byte leads; ASCII lanes (top bit clear) carry their
    // codepoint in the raw byte itself, so override them with (high=0, low=byte) before addressing the BMP tables.
    __mmask64 const ascii = ~_mm512_movepi8_mask(window);
    __m512i const high = _mm512_maskz_mov_epi8(~ascii, high_in);
    __m512i const low = _mm512_mask_mov_epi8(low_in, ascii, window);

    // Four 16-lane chunks of `high` index the aligned 256-entry block table; recombine byte-ordered.
    __m512i const high0 = _mm512_cvtepu8_epi32(_mm512_castsi512_si128(high));
    __m512i const high1 = _mm512_cvtepu8_epi32(_mm512_extracti32x4_epi32(high, 1));
    __m512i const high2 = _mm512_cvtepu8_epi32(_mm512_extracti32x4_epi32(high, 2));
    __m512i const high3 = _mm512_cvtepu8_epi32(_mm512_extracti32x4_epi32(high, 3));
    __m512i const block_id = sz_delimiter_pack_chunks_epi8_icelake_(
        sz_utf8_rune_permute256_icelake_(sz_utf8_delimiter_bmp_block_, high0),
        sz_utf8_rune_permute256_icelake_(sz_utf8_delimiter_bmp_block_, high1),
        sz_utf8_rune_permute256_icelake_(sz_utf8_delimiter_bmp_block_, high2),
        sz_utf8_rune_permute256_icelake_(sz_utf8_delimiter_bmp_block_, high3));

    // bit_index = block_id*32 + (low>>3), as 16-bit indices in [0, 1664); read through the page network.
    __m512i const nibble = _mm512_srli_epi16(_mm512_and_si512(low, _mm512_set1_epi8((char)0xF8)), 3);
    __m512i const idx_lo = _mm512_add_epi16(
        _mm512_slli_epi16(
            _mm512_and_si512(_mm512_unpacklo_epi8(block_id, _mm512_setzero_si512()), _mm512_set1_epi16(0x00FF)), 5),
        _mm512_and_si512(_mm512_unpacklo_epi8(nibble, _mm512_setzero_si512()), _mm512_set1_epi16(0x001F)));
    __m512i const idx_hi = _mm512_add_epi16(
        _mm512_slli_epi16(
            _mm512_and_si512(_mm512_unpackhi_epi8(block_id, _mm512_setzero_si512()), _mm512_set1_epi16(0x00FF)), 5),
        _mm512_and_si512(_mm512_unpackhi_epi8(nibble, _mm512_setzero_si512()), _mm512_set1_epi16(0x001F)));
    __m512i const byte_lo = sz_utf8_rune_gather_byte_(sz_utf8_delimiter_bmp_bitmaps_,
                                                      sz_utf8_delimiter_bmp_bitmaps_count_k * 32, idx_lo);
    __m512i const byte_hi = sz_utf8_rune_gather_byte_(sz_utf8_delimiter_bmp_bitmaps_,
                                                      sz_utf8_delimiter_bmp_bitmaps_count_k * 32, idx_hi);
    __m512i const bitmap_byte = _mm512_packus_epi16(byte_lo, byte_hi);
    return sz_delimiter_test_bit_icelake_(bitmap_byte, low);
}

/**
 *  @brief  Astral (codepoint >= 0x10000) delimiter membership for the four-byte lanes, gather-free.
 *
 *  Reconstructs the full 21-bit codepoint per lane from the raw lead/continuation bytes, then walks the small astral
 *  network: `super = offset>>16` selects an L1 group, `group*256 + ((offset>>8)&0xFF)` selects a bitmap row id, and the
 *  bit `(offset & 7)` is tested. Resolved over all 64 lanes; the caller blends the result onto the four-byte lanes.
 */
SZ_INTERNAL __mmask64 sz_delimiter_astral_membership_icelake_(__m512i window, __m512i next1, __m512i next2,
                                                              __m512i next3) {
    __m512i const byte0 = _mm512_and_si512(window, _mm512_set1_epi8(0x07));
    __m512i const byte1 = _mm512_and_si512(next1, _mm512_set1_epi8(0x3F));
    __m512i const byte2 = _mm512_and_si512(next2, _mm512_set1_epi8(0x3F));
    __m512i const byte3 = _mm512_and_si512(next3, _mm512_set1_epi8(0x3F));
    __m512i const lane_identity = sz_utf8_lane_identity_icelake_();
    __m512i row_bytes = _mm512_setzero_si512();
    __m512i low_bytes = _mm512_setzero_si512();
    for (int chunk = 0; chunk < 4; ++chunk) {
        __m512i const select = _mm512_add_epi8(lane_identity, _mm512_set1_epi8((char)(chunk * 16)));
        __m512i const lead = _mm512_cvtepu8_epi32(_mm512_castsi512_si128(_mm512_permutexvar_epi8(select, byte0)));
        __m512i const c1 = _mm512_cvtepu8_epi32(_mm512_castsi512_si128(_mm512_permutexvar_epi8(select, byte1)));
        __m512i const c2 = _mm512_cvtepu8_epi32(_mm512_castsi512_si128(_mm512_permutexvar_epi8(select, byte2)));
        __m512i const c3 = _mm512_cvtepu8_epi32(_mm512_castsi512_si128(_mm512_permutexvar_epi8(select, byte3)));
        __m512i const codepoint = _mm512_or_si512(
            _mm512_or_si512(_mm512_slli_epi32(lead, 18), _mm512_slli_epi32(c1, 12)),
            _mm512_or_si512(_mm512_slli_epi32(c2, 6), c3));
        __m512i const offset = _mm512_sub_epi32(codepoint, _mm512_set1_epi32(0x10000));
        __m512i const super = _mm512_and_si512(_mm512_srli_epi32(offset, 16), _mm512_set1_epi32(0x0F));
        __m512i const group = sz_utf8_rune_lut_cascade_icelake_(sz_utf8_delimiter_astral_l1_, (16 + 63) / 64, super);
        __m512i const sub = _mm512_and_si512(_mm512_srli_epi32(offset, 8), _mm512_set1_epi32(0xFF));
        __m512i const l2_index = _mm512_add_epi32(_mm512_slli_epi32(group, 8), sub);
        __m512i const row = sz_utf8_rune_lut_cascade_icelake_(
            sz_utf8_delimiter_astral_l2_, (sz_utf8_delimiter_astral_groups_count_k * 256 + 63) / 64, l2_index);
        __m512i const low8 = _mm512_and_si512(offset, _mm512_set1_epi32(0xFF));
        __m128i const row_packed = _mm512_cvtepi32_epi8(row);
        __m128i const low_packed = _mm512_cvtepi32_epi8(low8);
        __mmask64 const here = _cvtu64_mask64((sz_u64_t)0xFFFFull << (chunk * 16));
        row_bytes = _mm512_mask_mov_epi8(row_bytes, here,
                                         _mm512_maskz_expand_epi8(here, _mm512_castsi128_si512(row_packed)));
        low_bytes = _mm512_mask_mov_epi8(low_bytes, here,
                                         _mm512_maskz_expand_epi8(here, _mm512_castsi128_si512(low_packed)));
    }
    __m512i const nibble = _mm512_srli_epi16(_mm512_and_si512(low_bytes, _mm512_set1_epi8((char)0xF8)), 3);
    __m512i const idx_lo = _mm512_add_epi16(
        _mm512_slli_epi16(
            _mm512_and_si512(_mm512_unpacklo_epi8(row_bytes, _mm512_setzero_si512()), _mm512_set1_epi16(0x00FF)), 5),
        _mm512_and_si512(_mm512_unpacklo_epi8(nibble, _mm512_setzero_si512()), _mm512_set1_epi16(0x001F)));
    __m512i const idx_hi = _mm512_add_epi16(
        _mm512_slli_epi16(
            _mm512_and_si512(_mm512_unpackhi_epi8(row_bytes, _mm512_setzero_si512()), _mm512_set1_epi16(0x00FF)), 5),
        _mm512_and_si512(_mm512_unpackhi_epi8(nibble, _mm512_setzero_si512()), _mm512_set1_epi16(0x001F)));
    __m512i const byte_lo = sz_utf8_rune_gather_byte_(sz_utf8_delimiter_astral_bitmaps_,
                                                      sz_utf8_delimiter_astral_bitmaps_count_k * 32, idx_lo);
    __m512i const byte_hi = sz_utf8_rune_gather_byte_(sz_utf8_delimiter_astral_bitmaps_,
                                                      sz_utf8_delimiter_astral_bitmaps_count_k * 32, idx_hi);
    __m512i const bitmap_byte = _mm512_packus_epi16(byte_lo, byte_hi);
    return sz_delimiter_test_bit_icelake_(bitmap_byte, low_bytes);
}

/**
 *  @brief  Per-lane UTF-8 validity for codepoint-start lanes, mirroring `sz_rune_decode` exactly: a 2/3/4-byte lead is
 *          valid only when its continuation bytes are present (within the loaded span) and well-formed, and it is not
 *          overlong, a surrogate, or beyond U+10FFFF. Invalid leads are never reported (serial advances one byte and
 *          re-syncs, which never matches the cleared lane).
 */
SZ_INTERNAL __mmask64 sz_delimiter_valid_starts_icelake_( //
    __m512i window, __m512i next1, __m512i next2, __m512i next3, sz_utf8_rune_window_t const *decoded) {
    __mmask64 const loaded = sz_u64_clamp_mask_until_(decoded->loaded);
    __m512i const continuation_pattern = _mm512_set1_epi8((char)0x80);
    __m512i const continuation_mask = _mm512_set1_epi8((char)0xC0);
    __mmask64 const c1_ok = _mm512_cmpeq_epi8_mask(_mm512_and_si512(next1, continuation_mask), continuation_pattern);
    __mmask64 const c2_ok = _mm512_cmpeq_epi8_mask(_mm512_and_si512(next2, continuation_mask), continuation_pattern);
    __mmask64 const c3_ok = _mm512_cmpeq_epi8_mask(_mm512_and_si512(next3, continuation_mask), continuation_pattern);

    __mmask64 const ascii = loaded & ~_mm512_movepi8_mask(window);
    __mmask64 const two = decoded->two_byte_starts;
    __mmask64 const three = decoded->three_byte_starts;
    __mmask64 const four = decoded->four_byte_starts;

    // Spans must lie within the loaded window (so we never validate against a wrapped neighbour or read past loaded).
    __mmask64 const span2 = sz_u64_clamp_mask_until_(decoded->loaded >= 1 ? decoded->loaded - 1 : 0);
    __mmask64 const span3 = sz_u64_clamp_mask_until_(decoded->loaded >= 2 ? decoded->loaded - 2 : 0);
    __mmask64 const span4 = sz_u64_clamp_mask_until_(decoded->loaded >= 3 ? decoded->loaded - 3 : 0);

    // 2-byte: lead >= 0xC2 (reject C0/C1 overlong); continuation ok.
    __mmask64 const lead_ge_c2 = _mm512_cmp_epu8_mask(window, _mm512_set1_epi8((char)0xC2), _MM_CMPINT_NLT);
    __mmask64 const two_ok = two & span2 & c1_ok & lead_ge_c2;

    // 3-byte: c1,c2 ok; not overlong (E0 with next1 < 0xA0); not surrogate (ED with next1 >= 0xA0).
    __mmask64 const lead_e0 = _mm512_cmpeq_epi8_mask(window, _mm512_set1_epi8((char)0xE0));
    __mmask64 const lead_ed = _mm512_cmpeq_epi8_mask(window, _mm512_set1_epi8((char)0xED));
    __mmask64 const n1_lt_a0 = _mm512_cmp_epu8_mask(next1, _mm512_set1_epi8((char)0xA0), _MM_CMPINT_LT);
    __mmask64 const three_ok = three & span3 & c1_ok & c2_ok & ~(lead_e0 & n1_lt_a0) & ~(lead_ed & ~n1_lt_a0);

    // 4-byte: lead <= 0xF4; c1,c2,c3 ok; not overlong (F0 with next1 < 0x90); not > U+10FFFF (F4 with next1 >= 0x90).
    __mmask64 const lead_le_f4 = _mm512_cmp_epu8_mask(window, _mm512_set1_epi8((char)0xF4), _MM_CMPINT_LE);
    __mmask64 const lead_f0 = _mm512_cmpeq_epi8_mask(window, _mm512_set1_epi8((char)0xF0));
    __mmask64 const lead_f4 = _mm512_cmpeq_epi8_mask(window, _mm512_set1_epi8((char)0xF4));
    __mmask64 const n1_lt_90 = _mm512_cmp_epu8_mask(next1, _mm512_set1_epi8((char)0x90), _MM_CMPINT_LT);
    __mmask64 const four_ok = four & span4 & c1_ok & c2_ok & c3_ok & lead_le_f4 & ~(lead_f0 & n1_lt_90) &
                              ~(lead_f4 & ~n1_lt_90);

    return (ascii | two_ok | three_ok | four_ok);
}

#pragma endregion Gather free membership

#pragma region Forward driver

/** @copydoc sz_find_delimiter_utf8 */
SZ_PUBLIC sz_cptr_t sz_find_delimiters_utf8_icelake(sz_cptr_t text, sz_size_t length, sz_size_t *matched_length) {
    sz_u8_t const *const text_u8 = (sz_u8_t const *)text;
    __m512i const lane_identity = sz_utf8_lane_identity_icelake_();

    sz_size_t base = 0;
    while (base < length) {
        sz_utf8_rune_window_t const decoded = sz_utf8_rune_decode_window_icelake_(text_u8 + base, length - base,
                                                                                  lane_identity);
        sz_size_t const loaded = decoded.loaded;
        __m512i const window = decoded.window;
        __m512i const next1 = _mm512_permutexvar_epi8(_mm512_add_epi8(lane_identity, _mm512_set1_epi8(1)), window);
        __m512i const next2 = _mm512_permutexvar_epi8(_mm512_add_epi8(lane_identity, _mm512_set1_epi8(2)), window);
        __m512i const next3 = _mm512_permutexvar_epi8(_mm512_add_epi8(lane_identity, _mm512_set1_epi8(3)), window);

        // Effective-window trim: a multi-byte lead near the 64-byte edge whose span runs past `loaded` would decode
        // against a wrapped neighbour; defer it to the next window.
        sz_size_t byte_span = loaded;
        if (loaded >= 64) {
            sz_u64_t const overrun =
                (_cvtmask64_u64(decoded.two_byte_starts) & ~sz_u64_mask_until_serial_(loaded - 1)) |
                (_cvtmask64_u64(decoded.three_byte_starts) & ~sz_u64_mask_until_serial_(loaded - 2)) |
                (_cvtmask64_u64(decoded.four_byte_starts) & ~sz_u64_mask_until_serial_(loaded - 3));
            byte_span = overrun ? (sz_size_t)sz_u64_ctz(overrun) : loaded;
        }
        __mmask64 const span_mask = _cvtu64_mask64(sz_u64_mask_until_serial_(byte_span));

        __mmask64 const valid_starts = sz_delimiter_valid_starts_icelake_(window, next1, next2, next3, &decoded) &
                                       decoded.codepoint_starts & span_mask;

        // BMP membership for every lane; astral membership blended onto the four-byte lanes (gated on presence).
        __mmask64 member = sz_delimiter_bmp_membership_icelake_(window, decoded.high, decoded.low);
        __mmask64 const four_byte = decoded.four_byte_starts & span_mask;
        if (four_byte) {
            __mmask64 const astral_member = sz_delimiter_astral_membership_icelake_(window, next1, next2, next3);
            member = (member & ~four_byte) | (astral_member & four_byte);
        }

        sz_u64_t const hits = _cvtmask64_u64(member & valid_starts);
        if (hits) {
            sz_size_t const lane = (sz_size_t)sz_u64_ctz(hits);
            if (matched_length) {
                sz_size_t length_at_lane = 1;
                length_at_lane += (_cvtmask64_u64(decoded.two_byte_starts) >> lane) & 1;
                length_at_lane += ((_cvtmask64_u64(decoded.three_byte_starts) >> lane) & 1) * 2;
                length_at_lane += ((_cvtmask64_u64(decoded.four_byte_starts) >> lane) & 1) * 3;
                *matched_length = length_at_lane;
            }
            return (sz_cptr_t)(text_u8 + base + lane);
        }
        base += byte_span ? byte_span : 1;
    }

    if (matched_length) *matched_length = 0;
    return SZ_NULL_CHAR;
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

#endif // STRINGZILLA_UTF8_DELIMITERS_ICELAKE_H_
