/**
 *  @brief Haswell (AVX2) backend for UTF-8 newline and whitespace delimiter scanning.
 *  @file include/stringzilla/utf8_tokens/haswell.h
 *  @author Ash Vardanian
 */
#ifndef STRINGZILLA_UTF8_TOKENS_HASWELL_H_
#define STRINGZILLA_UTF8_TOKENS_HASWELL_H_

#include "stringzilla/types.h"
#include "stringzilla/utf8_tokens/serial.h"
#include "stringzilla/utf8_runes/haswell.h"
#include "stringzilla/utf8_tokens/tables.h"

#ifdef __cplusplus
extern "C" {
#endif

#if SZ_USE_HASWELL
#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("avx2,bmi,bmi2,popcnt"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("avx2,bmi,bmi2,popcnt")
#endif

/**
 *  @brief Unsigned byte greater-than-or-equal comparison for AVX2 via the `max(a, b) == a` identity.
 */
SZ_HELPER_INLINE __m256i sz_mm256_cmpge_epu8_haswell_(__m256i a, __m256i b) {
    return _mm256_cmpeq_epi8(_mm256_max_epu8(a, b), a);
}

#pragma region Multistep newline / whitespace iteration

/*  Multistep newline / whitespace iteration (Haswell / AVX2). Each 32-byte window is classified into a
 *  `start_bits` mask plus per-length start masks, then the peel left-packs matches with `vpermd`. Starts in
 *  lanes [0,29] are trusted and the cursor steps 30, so any 2-/3-byte delimiter is fully loaded. */

/**
 *  @brief  Peel the window's first `emit_count` matches with a `vpermd` left-pack, 4 lanes per sub-block.
 *          Each sub-block gathers its set lanes to the front and masked-stores them at the advancing cursor.
 */
SZ_HELPER_AUTO void sz_utf8_iterate_peel_haswell_(                             //
    sz_u32_t start_bits, sz_u32_t two_byte_starts, sz_u32_t three_byte_starts, //
    sz_size_t emit_count, sz_size_t position,                                  //
    sz_size_t *match_offsets, sz_size_t *match_lengths) {

    // Per-file copy of the sorting backend's left-pack table: row `[m]` holds the 8 dword indices that gather the
    // `m`-selected u64 lanes (of 4, each a dword pair) to the front for `_mm256_permutevar8x32_epi32`.
    static sz_u32_t const compact_lut[16][8] = {
        {0, 0, 0, 0, 0, 0, 0, 0}, {0, 1, 0, 0, 0, 0, 0, 0}, {2, 3, 0, 0, 0, 0, 0, 0}, {0, 1, 2, 3, 0, 0, 0, 0},
        {4, 5, 0, 0, 0, 0, 0, 0}, {0, 1, 4, 5, 0, 0, 0, 0}, {2, 3, 4, 5, 0, 0, 0, 0}, {0, 1, 2, 3, 4, 5, 0, 0},
        {6, 7, 0, 0, 0, 0, 0, 0}, {0, 1, 6, 7, 0, 0, 0, 0}, {2, 3, 6, 7, 0, 0, 0, 0}, {0, 1, 2, 3, 6, 7, 0, 0},
        {4, 5, 6, 7, 0, 0, 0, 0}, {0, 1, 4, 5, 6, 7, 0, 0}, {2, 3, 4, 5, 6, 7, 0, 0}, {0, 1, 2, 3, 4, 5, 6, 7},
    };

    __m256i const lane_ramp = _mm256_setr_epi64x(0, 1, 2, 3);
    sz_size_t emitted = 0;
    for (sz_size_t sub_block = 0; sub_block < 8 && emitted < emit_count; ++sub_block) {
        sz_u32_t const submask = (start_bits >> (sub_block * 4)) & 0xFu;
        if (!submask) continue;

        sz_size_t const base_lane = sub_block * 4;
        // Length per lane: 1, plus 1 on a 2-byte start, plus 2 on a 3-byte start (the masks are disjoint).
        sz_u32_t const two_byte_sub = (two_byte_starts >> base_lane) & 0xFu;
        sz_u32_t const three_byte_sub = (three_byte_starts >> base_lane) & 0xFu;
        __m256i const two_byte_add = _mm256_setr_epi64x((two_byte_sub >> 0) & 1u, (two_byte_sub >> 1) & 1u,
                                                        (two_byte_sub >> 2) & 1u, (two_byte_sub >> 3) & 1u);
        __m256i const three_byte_add = _mm256_setr_epi64x((three_byte_sub >> 0) & 1u, (three_byte_sub >> 1) & 1u,
                                                          (three_byte_sub >> 2) & 1u, (three_byte_sub >> 3) & 1u);
        __m256i const offsets_u64x4 = _mm256_add_epi64(_mm256_set1_epi64x((long long)(position + base_lane)),
                                                       lane_ramp);
        __m256i const lengths_u64x4 = _mm256_add_epi64(
            _mm256_set1_epi64x(1), _mm256_add_epi64(two_byte_add, _mm256_add_epi64(three_byte_add, three_byte_add)));

        __m256i const permutation = _mm256_loadu_si256((__m256i const *)compact_lut[submask]);
        __m256i const packed_offsets = _mm256_permutevar8x32_epi32(offsets_u64x4, permutation);
        __m256i const packed_lengths = _mm256_permutevar8x32_epi32(lengths_u64x4, permutation);

        sz_size_t const taken = sz_min_of_two((sz_size_t)_mm_popcnt_u32(submask), emit_count - emitted);
        __m256i const store_mask = sz_mm256_store_mask_epi64_(taken);
        _mm256_maskstore_epi64((long long *)(match_offsets + emitted), store_mask, packed_offsets);
        _mm256_maskstore_epi64((long long *)(match_lengths + emitted), store_mask, packed_lengths);
        emitted += taken;
    }
}

SZ_API_COMPTIME sz_size_t sz_utf8_newlines_haswell(     //
    sz_cptr_t text, sz_size_t length,                   //
    sz_size_t *match_offsets, sz_size_t *match_lengths, //
    sz_size_t matches_capacity, sz_size_t *bytes_consumed) {

    sz_u8_t const *text_u8 = (sz_u8_t const *)text;
    sz_size_t count = 0, position = 0;

    __m256i newline_u8x32 = _mm256_set1_epi8('\n'), vertical_tab_u8x32 = _mm256_set1_epi8('\v'),
            form_feed_u8x32 = _mm256_set1_epi8('\f'), carriage_return_u8x32 = _mm256_set1_epi8('\r'),
            lead_c2_u8x32 = _mm256_set1_epi8('\xC2'), x_85_u8x32 = _mm256_set1_epi8('\x85'),
            lead_e2_u8x32 = _mm256_set1_epi8('\xE2'), byte_80_u8x32 = _mm256_set1_epi8('\x80'),
            x_a8_u8x32 = _mm256_set1_epi8('\xA8'), x_a9_u8x32 = _mm256_set1_epi8('\xA9');

    while (position + 32 <= length && count < matches_capacity) {
        __m256i window = _mm256_loadu_si256((__m256i const *)(text_u8 + position));

        // 1-byte starts: \n, \v, \f, \r.
        sz_u32_t newline_mask = (sz_u32_t)_mm256_movemask_epi8(_mm256_cmpeq_epi8(window, newline_u8x32));
        sz_u32_t carriage_return_mask = (sz_u32_t)_mm256_movemask_epi8(
            _mm256_cmpeq_epi8(window, carriage_return_u8x32));
        sz_u32_t one_byte_mask = newline_mask |
                                 (sz_u32_t)_mm256_movemask_epi8(_mm256_cmpeq_epi8(window, vertical_tab_u8x32)) |
                                 (sz_u32_t)_mm256_movemask_epi8(_mm256_cmpeq_epi8(window, form_feed_u8x32)) |
                                 carriage_return_mask;

        // 2-byte NEL (C2 85); 3-byte LS/PS (E2 80 A8/A9) - computed unconditionally (cheaper than a data-dependent gate).
        sz_u32_t lead_c2_mask = (sz_u32_t)_mm256_movemask_epi8(_mm256_cmpeq_epi8(window, lead_c2_u8x32));
        sz_u32_t x_85_mask = (sz_u32_t)_mm256_movemask_epi8(_mm256_cmpeq_epi8(window, x_85_u8x32));
        sz_u32_t lead_e2_mask = (sz_u32_t)_mm256_movemask_epi8(_mm256_cmpeq_epi8(window, lead_e2_u8x32));
        sz_u32_t byte_80_mask = (sz_u32_t)_mm256_movemask_epi8(_mm256_cmpeq_epi8(window, byte_80_u8x32));
        sz_u32_t x_a8_mask = (sz_u32_t)_mm256_movemask_epi8(_mm256_cmpeq_epi8(window, x_a8_u8x32));
        sz_u32_t x_a9_mask = (sz_u32_t)_mm256_movemask_epi8(_mm256_cmpeq_epi8(window, x_a9_u8x32));

        sz_u32_t nel_mask = lead_c2_mask & (x_85_mask >> 1);
        sz_u32_t lead_e280_mask = lead_e2_mask & (byte_80_mask >> 1);
        sz_u32_t line_para_mask = lead_e280_mask & ((x_a8_mask | x_a9_mask) >> 2);
        // CRLF: a CR whose next lane is LF is a single 2-byte match; its trailing LF must not also be emitted.
        sz_u32_t crlf_mask = carriage_return_mask & (newline_mask >> 1);
        sz_u32_t lf_of_crlf_mask = newline_mask & (carriage_return_mask << 1);

        sz_u32_t two_byte_starts = crlf_mask | nel_mask;
        sz_u32_t three_byte_starts = line_para_mask;
        sz_u32_t starts = (one_byte_mask | nel_mask | line_para_mask) & ~lf_of_crlf_mask;
        sz_u32_t start_bits = starts & ((1u << 30) - 1); // trust lanes [0,29]; step 30
        // Suppress a leading LF already consumed by a CRLF that straddled the previous window edge.
        if (position != 0 && text_u8[position - 1] == '\r') start_bits &= ~(newline_mask & 1u);

        sz_size_t const window_matches = (sz_size_t)_mm_popcnt_u32(start_bits);
        sz_size_t const emit_count = sz_min_of_two(window_matches, matches_capacity - count);
        if (emit_count)
            sz_utf8_iterate_peel_haswell_(start_bits, two_byte_starts, three_byte_starts, emit_count, position,
                                          match_offsets + count, match_lengths + count);
        count += emit_count;
        if (count == matches_capacity) { // output buffer full: resume past the last emitted match
            position = match_offsets[count - 1] + match_lengths[count - 1];
            break;
        }
        position += 30;
    }

    // Skip a CRLF's trailing LF if it straddles into the serial tail (the CR was emitted as a 2-byte match).
    if (position != 0 && position < length && text_u8[position - 1] == '\r' && text_u8[position] == '\n') ++position;
    count += sz_utf8_newlines_serial_((sz_cptr_t)(text_u8 + position), length - position, position,
                                      match_offsets + count, match_lengths + count, matches_capacity - count,
                                      bytes_consumed);
    return count;
}

SZ_API_COMPTIME sz_size_t sz_utf8_whitespaces_haswell(  //
    sz_cptr_t text, sz_size_t length,                   //
    sz_size_t *match_offsets, sz_size_t *match_lengths, //
    sz_size_t matches_capacity, sz_size_t *bytes_consumed) {

    sz_u8_t const *text_u8 = (sz_u8_t const *)text;
    sz_size_t count = 0, position = 0;

    __m256i x_20_u8x32 = _mm256_set1_epi8(' '), lead_c2_u8x32 = _mm256_set1_epi8('\xC2'),
            x_85_u8x32 = _mm256_set1_epi8('\x85'), x_a0_u8x32 = _mm256_set1_epi8('\xA0'),
            x_e1_u8x32 = _mm256_set1_epi8('\xE1'), lead_e2_u8x32 = _mm256_set1_epi8('\xE2'),
            x_e3_u8x32 = _mm256_set1_epi8('\xE3'), x_9a_u8x32 = _mm256_set1_epi8('\x9A'),
            byte_80_u8x32 = _mm256_set1_epi8('\x80'), x_81_u8x32 = _mm256_set1_epi8('\x81'),
            x_8a_u8x32 = _mm256_set1_epi8('\x8A'), x_a8_u8x32 = _mm256_set1_epi8('\xA8'),
            x_a9_u8x32 = _mm256_set1_epi8('\xA9'), x_af_u8x32 = _mm256_set1_epi8('\xAF'),
            x_9f_u8x32 = _mm256_set1_epi8('\x9F');

    while (position + 32 <= length && count < matches_capacity) {
        __m256i window = _mm256_loadu_si256((__m256i const *)(text_u8 + position));

        // 1-byte: space, plus the contiguous range [\t, \r] == [9, 13]. AVX2 has no unsigned compare, so reuse the
        // signed-bound idiom from `find/haswell.h` (these bytes are all < 0x80, so signed is exact).
        __m256i x_20_cmp = _mm256_cmpeq_epi8(window, x_20_u8x32);
        __m256i tab_lower_bound_cmp = _mm256_cmpgt_epi8(window, _mm256_set1_epi8((char)0x08)); // >= '\t' (0x09)
        __m256i carriage_return_upper_bound_cmp = _mm256_cmpgt_epi8(_mm256_set1_epi8((char)0x0E), window); // <= '\r'
        __m256i one_byte_cmp = _mm256_or_si256(x_20_cmp,
                                               _mm256_and_si256(tab_lower_bound_cmp, carriage_return_upper_bound_cmp));
        sz_u32_t one_byte_mask = (sz_u32_t)_mm256_movemask_epi8(one_byte_cmp);

        // 2-byte: C2 85 (NEL), C2 A0 (NBSP).
        sz_u32_t lead_c2_mask = (sz_u32_t)_mm256_movemask_epi8(_mm256_cmpeq_epi8(window, lead_c2_u8x32));
        sz_u32_t x_85_mask = (sz_u32_t)_mm256_movemask_epi8(_mm256_cmpeq_epi8(window, x_85_u8x32));
        sz_u32_t x_a0_mask = (sz_u32_t)_mm256_movemask_epi8(_mm256_cmpeq_epi8(window, x_a0_u8x32));
        sz_u32_t two_byte_starts = lead_c2_mask & ((x_85_mask | x_a0_mask) >> 1);

        // 3-byte: E1 9A 80 (ogham); E2 80 [80-8A]; E2 80 AF; E2 81 9F; E2 80 A8/A9; E3 80 80.
        sz_u32_t x_e1_mask = (sz_u32_t)_mm256_movemask_epi8(_mm256_cmpeq_epi8(window, x_e1_u8x32));
        sz_u32_t lead_e2_mask = (sz_u32_t)_mm256_movemask_epi8(_mm256_cmpeq_epi8(window, lead_e2_u8x32));
        sz_u32_t x_e3_mask = (sz_u32_t)_mm256_movemask_epi8(_mm256_cmpeq_epi8(window, x_e3_u8x32));
        sz_u32_t x_9a_mask = (sz_u32_t)_mm256_movemask_epi8(_mm256_cmpeq_epi8(window, x_9a_u8x32));
        sz_u32_t byte_80_mask = (sz_u32_t)_mm256_movemask_epi8(_mm256_cmpeq_epi8(window, byte_80_u8x32));
        sz_u32_t x_81_mask = (sz_u32_t)_mm256_movemask_epi8(_mm256_cmpeq_epi8(window, x_81_u8x32));
        __m256i x_80_ge_cmp = sz_mm256_cmpge_epu8_haswell_(window, byte_80_u8x32); // >= 0x80
        __m256i x_8a_le_cmp = sz_mm256_cmpge_epu8_haswell_(x_8a_u8x32, window);    // <= 0x8A
        sz_u32_t x_8a_range_mask = (sz_u32_t)_mm256_movemask_epi8(_mm256_and_si256(x_80_ge_cmp, x_8a_le_cmp));
        sz_u32_t x_a8_mask = (sz_u32_t)_mm256_movemask_epi8(_mm256_cmpeq_epi8(window, x_a8_u8x32));
        sz_u32_t x_a9_mask = (sz_u32_t)_mm256_movemask_epi8(_mm256_cmpeq_epi8(window, x_a9_u8x32));
        sz_u32_t x_af_mask = (sz_u32_t)_mm256_movemask_epi8(_mm256_cmpeq_epi8(window, x_af_u8x32));
        sz_u32_t x_9f_mask = (sz_u32_t)_mm256_movemask_epi8(_mm256_cmpeq_epi8(window, x_9f_u8x32));

        sz_u32_t ogham_mask = x_e1_mask & (x_9a_mask >> 1) & (byte_80_mask >> 2);               // E1 9A 80
        sz_u32_t range_e280_mask = lead_e2_mask & (byte_80_mask >> 1) & (x_8a_range_mask >> 2); // E2 80 [80-8A]
        sz_u32_t nnbsp_mask = lead_e2_mask & (byte_80_mask >> 1) & (x_af_mask >> 2);            // E2 80 AF
        sz_u32_t mmsp_mask = lead_e2_mask & (x_81_mask >> 1) & (x_9f_mask >> 2);                // E2 81 9F
        sz_u32_t line_mask = lead_e2_mask & (byte_80_mask >> 1) & (x_a8_mask >> 2);             // E2 80 A8
        sz_u32_t para_mask = lead_e2_mask & (byte_80_mask >> 1) & (x_a9_mask >> 2);             // E2 80 A9
        sz_u32_t ideographic_mask = x_e3_mask & (byte_80_mask >> 1) & (byte_80_mask >> 2);      // E3 80 80
        sz_u32_t three_byte_starts = ogham_mask | range_e280_mask | nnbsp_mask | mmsp_mask | line_mask | para_mask |
                                     ideographic_mask;

        sz_u32_t starts = one_byte_mask | two_byte_starts | three_byte_starts;
        sz_u32_t start_bits = starts & ((1u << 30) - 1); // trust lanes [0,29]; step 30

        sz_size_t const window_matches = (sz_size_t)_mm_popcnt_u32(start_bits);
        sz_size_t const emit_count = sz_min_of_two(window_matches, matches_capacity - count);
        if (emit_count)
            sz_utf8_iterate_peel_haswell_(start_bits, two_byte_starts, three_byte_starts, emit_count, position,
                                          match_offsets + count, match_lengths + count);
        count += emit_count;
        if (count == matches_capacity) {
            position = match_offsets[count - 1] + match_lengths[count - 1];
            break;
        }
        position += 30;
    }

    count += sz_utf8_whitespaces_serial_((sz_cptr_t)(text_u8 + position), length - position, position,
                                         match_offsets + count, match_lengths + count, matches_capacity - count,
                                         bytes_consumed);
    return count;
}

#pragma endregion // Multistep newline / whitespace iteration

#pragma region Gather free membership

/** @brief  Per-half unsigned `value >= bound` mask (AVX2 has no unsigned compare): `max_epu8(value,bound)==value`. */
SZ_HELPER_INLINE __m256i sz_delimiter_cmpge_epu8_haswell_(__m256i value, __m256i bound) {
    return _mm256_cmpeq_epi8(_mm256_max_epu8(value, bound), value);
}

/** @brief  Per-half "third forward neighbour" `next3[i] = window[i+3]`, wrapping modulo 64 to mirror the substrate
 *          neighbour helper (which only emits next1/next2). Needed for the 4-byte astral codepoint reconstruction. */
SZ_HELPER_INLINE void sz_delimiter_forward_neighbour3_haswell_( //
    __m256i window_lo, __m256i window_hi, __m256i *next3_lo, __m256i *next3_hi) {
    __m256i const low_successor = _mm256_permute2x128_si256(window_lo, window_hi, 0x21);
    *next3_lo = _mm256_alignr_epi8(low_successor, window_lo, 3);
    __m256i const high_successor = _mm256_permute2x128_si256(window_hi, window_lo, 0x21);
    *next3_hi = _mm256_alignr_epi8(high_successor, window_hi, 3);
}

/** @brief  Per-lane single-bit test `(bitmap_byte >> (low & 7)) & 1` for one 32-lane half, returned as a 0xFF/0x00
 *          byte mask. The bit mask `1 << (low & 7)` is built by a `vpshufb` over the resident power-of-two table. */
SZ_HELPER_INLINE __m256i sz_delimiter_test_bit_haswell_(__m256i bitmap_byte, __m256i low) {
    __m256i const bit_table = _mm256_setr_epi8(                    //
        1, 2, 4, 8, 16, 32, 64, (char)128, 0, 0, 0, 0, 0, 0, 0, 0, //
        1, 2, 4, 8, 16, 32, 64, (char)128, 0, 0, 0, 0, 0, 0, 0, 0);
    __m256i const bit_mask = _mm256_shuffle_epi8(bit_table, _mm256_and_si256(low, _mm256_set1_epi8(0x07)));
    __m256i const isolated = _mm256_and_si256(bitmap_byte, bit_mask);
    return _mm256_cmpeq_epi8(isolated, bit_mask);
}

/** @brief  Read the bitmap byte `columns[(low>>3)*64 + block_id]` for one 32-lane half, gather-free: 32 candidate
 *          column reads (each a 64-entry `cascade_stage` over `block_id`) blended by which column `(low >> 3)` selects.
 *          The transposed `..._columns_` layout (column c holds `bitmaps[id*32+c]`) makes each column lut256-addressable
 *          for `block_id < 64` without a page network. */
SZ_HELPER_AUTO __m256i sz_delimiter_bitmap_byte_haswell_(sz_u8_t const *columns, __m256i block_id, __m256i low) {
    __m256i const selector = _mm256_and_si256(_mm256_srli_epi16(block_id, 4),
                                              _mm256_set1_epi8(0x0F));         // block_id>>4 (0..3)
    __m256i const within = _mm256_and_si256(block_id, _mm256_set1_epi8(0x0F)); // block_id&15
    __m256i const column_index = _mm256_and_si256(_mm256_srli_epi16(low, 3),
                                                  _mm256_set1_epi8(0x1F)); // (low>>3) in [0,32)
    __m256i result = _mm256_setzero_si256();
    for (int column = 0; column < 32; ++column) {
        __m256i const candidate = sz_utf8_rune_cascade_stage_haswell_(columns + column * 64, 4, selector, within);
        __m256i const here = _mm256_cmpeq_epi8(column_index, _mm256_set1_epi8((char)column));
        result = _mm256_blendv_epi8(result, candidate, here);
    }
    return result;
}

/** @brief  BMP (codepoint < 0x10000) delimiter membership for one 32-lane half, as a 0xFF/0x00 byte mask. ASCII lanes
 *          (top bit clear) carry their codepoint in the raw byte, so are overridden to (high=0, low=byte). */
SZ_HELPER_AUTO __m256i sz_delimiter_bmp_membership_haswell_(__m256i window, __m256i high_in, __m256i low_in) {
    __m256i const ascii = _mm256_cmpeq_epi8(_mm256_and_si256(window, _mm256_set1_epi8((char)0x80)),
                                            _mm256_setzero_si256());
    __m256i const high = _mm256_andnot_si256(ascii, high_in);
    __m256i const low = _mm256_blendv_epi8(low_in, window, ascii);
    __m256i const block_id = sz_utf8_rune_lut256_haswell_(sz_utf8_delimiter_bmp_block_, high);
    __m256i const bitmap_byte = sz_delimiter_bitmap_byte_haswell_(sz_utf8_delimiter_bmp_bitmaps_columns_, block_id,
                                                                  low);
    return sz_delimiter_test_bit_haswell_(bitmap_byte, low);
}

/** @brief  Astral (codepoint >= 0x10000) delimiter membership for one 32-lane half, as a 0xFF/0x00 byte mask. The full
 *          21-bit codepoint is reconstructed in byte-domain from the raw lead/continuation bytes; the small L1/L2 network
 *          and bitmap are then walked exactly as for the BMP path. Only meaningful on 4-byte lead lanes (caller blends). */
SZ_HELPER_AUTO __m256i sz_delimiter_astral_membership_haswell_( //
    __m256i window, __m256i next1, __m256i next2, __m256i next3) {
    __m256i const b0 = _mm256_and_si256(window, _mm256_set1_epi8(0x07)); // lead bits  cp[20:18]
    __m256i const b1 = _mm256_and_si256(next1, _mm256_set1_epi8(0x3F));  // cp[17:12]
    __m256i const b2 = _mm256_and_si256(next2, _mm256_set1_epi8(0x3F));  // cp[11:6]
    __m256i const b3 = _mm256_and_si256(next3, _mm256_set1_epi8(0x3F));  // cp[5:0]

    // super = (cp >> 16) - 1 = (((b0 << 2) | (b1 >> 4)) - 1). cp[20:16] = b0(3 bits) << 2 | b1[5:4].
    __m256i const cp_hi5 = _mm256_or_si256(_mm256_slli_epi16(b0, 2),
                                           _mm256_and_si256(_mm256_srli_epi16(b1, 4), _mm256_set1_epi8(0x03)));
    __m256i const super = _mm256_and_si256(_mm256_sub_epi8(cp_hi5, _mm256_set1_epi8(1)), _mm256_set1_epi8(0x0F));

    // sub = (offset >> 8) & 0xFF where offset = cp - 0x10000. (cp >> 8) = (b1[3:0] << 4) | (b2 >> 2 ... wait b2 spans
    // cp[11:6], so cp[15:8] = (b1[3:0] << 4) | (b2[5:2]). offset>>8 differs from cp>>8 by the -0x100 borrow folded into
    // super; (offset >> 8) & 0xFF == (cp >> 8) & 0xFF since the subtraction only affects bit 16+. So sub = (cp>>8)&0xFF.
    __m256i const sub = _mm256_or_si256(_mm256_slli_epi16(_mm256_and_si256(b1, _mm256_set1_epi8(0x0F)), 4),
                                        _mm256_and_si256(_mm256_srli_epi16(b2, 2), _mm256_set1_epi8(0x0F)));

    // low8 = offset & 0xFF = cp & 0xFF = (b2[1:0] << 6) | b3.
    __m256i const low8 = _mm256_or_si256(_mm256_slli_epi16(_mm256_and_si256(b2, _mm256_set1_epi8(0x03)), 6), b3);

    // group = astral_l1[super]; row id = astral_l2[group*256 + sub]; bitmap byte from astral columns.
    __m256i const group = sz_utf8_rune_lut256_haswell_(sz_utf8_delimiter_astral_l1_, super);
    // l2 index = group*256 + sub. group < 2, so the index high byte is `group` and the low byte is `sub`; a 512-entry
    // table is two 256-entry pages selected by `group`.
    __m256i const page0 = sz_utf8_rune_lut256_haswell_(sz_utf8_delimiter_astral_l2_ + 0, sub);
    __m256i const page1 = sz_utf8_rune_lut256_haswell_(sz_utf8_delimiter_astral_l2_ + 256, sub);
    __m256i const pick_page1 = _mm256_cmpeq_epi8(group, _mm256_set1_epi8(1));
    __m256i const row = _mm256_blendv_epi8(page0, page1, pick_page1);
    __m256i const bitmap_byte = sz_delimiter_bitmap_byte_haswell_(sz_utf8_delimiter_astral_bitmaps_columns_, row, low8);
    return sz_delimiter_test_bit_haswell_(bitmap_byte, low8);
}

/** @brief  Per-lane UTF-8 validity for codepoint-start lanes, mirroring `sz_rune_decode` exactly: a 2/3/4-byte lead is
 *          valid only when its continuation bytes are present (within the loaded span) and well-formed, and it is not
 *          overlong, a surrogate, or beyond U+10FFFF. Returned as a `sz_u64_t` lane mask. */
SZ_HELPER_AUTO sz_u64_t sz_delimiter_valid_starts_haswell_( //
    sz_utf8_rune_window_haswell_t const *decoded, __m256i next1_lo, __m256i next1_hi, __m256i next2_lo,
    __m256i next2_hi, __m256i next3_lo, __m256i next3_hi) {
    sz_size_t const loaded = decoded->loaded;
    sz_u64_t const loaded_mask = loaded >= 64 ? ~(sz_u64_t)0 : (((sz_u64_t)1 << loaded) - 1);

    __m256i const continuation_mask = _mm256_set1_epi8((char)0xC0);
    __m256i const continuation_pattern = _mm256_set1_epi8((char)0x80);
    sz_u64_t const c1_ok = sz_utf8_mask_combine_haswell_(
        _mm256_cmpeq_epi8(_mm256_and_si256(next1_lo, continuation_mask), continuation_pattern),
        _mm256_cmpeq_epi8(_mm256_and_si256(next1_hi, continuation_mask), continuation_pattern));
    sz_u64_t const c2_ok = sz_utf8_mask_combine_haswell_(
        _mm256_cmpeq_epi8(_mm256_and_si256(next2_lo, continuation_mask), continuation_pattern),
        _mm256_cmpeq_epi8(_mm256_and_si256(next2_hi, continuation_mask), continuation_pattern));
    sz_u64_t const c3_ok = sz_utf8_mask_combine_haswell_(
        _mm256_cmpeq_epi8(_mm256_and_si256(next3_lo, continuation_mask), continuation_pattern),
        _mm256_cmpeq_epi8(_mm256_and_si256(next3_hi, continuation_mask), continuation_pattern));

    // 2-byte: lead >= 0xC2 (reject C0/C1 overlong).
    sz_u64_t const lead_ge_c2 = sz_utf8_mask_combine_haswell_(
        sz_delimiter_cmpge_epu8_haswell_(decoded->window_lo, _mm256_set1_epi8((char)0xC2)),
        sz_delimiter_cmpge_epu8_haswell_(decoded->window_hi, _mm256_set1_epi8((char)0xC2)));

    // 3-byte: not overlong (E0 with next1 < 0xA0), not surrogate (ED with next1 >= 0xA0).
    sz_u64_t const lead_e0 = sz_utf8_mask_combine_haswell_(
        _mm256_cmpeq_epi8(decoded->window_lo, _mm256_set1_epi8((char)0xE0)),
        _mm256_cmpeq_epi8(decoded->window_hi, _mm256_set1_epi8((char)0xE0)));
    sz_u64_t const lead_ed = sz_utf8_mask_combine_haswell_(
        _mm256_cmpeq_epi8(decoded->window_lo, _mm256_set1_epi8((char)0xED)),
        _mm256_cmpeq_epi8(decoded->window_hi, _mm256_set1_epi8((char)0xED)));
    __m256i const n1_lo_ge_a0 = sz_delimiter_cmpge_epu8_haswell_(next1_lo, _mm256_set1_epi8((char)0xA0));
    __m256i const n1_hi_ge_a0 = sz_delimiter_cmpge_epu8_haswell_(next1_hi, _mm256_set1_epi8((char)0xA0));
    sz_u64_t const n1_ge_a0 = sz_utf8_mask_combine_haswell_(n1_lo_ge_a0, n1_hi_ge_a0);
    sz_u64_t const n1_lt_a0 = ~n1_ge_a0;

    // 4-byte: lead <= 0xF4, not overlong (F0 with next1 < 0x90), not > U+10FFFF (F4 with next1 >= 0x90).
    sz_u64_t const lead_le_f4 = sz_utf8_mask_combine_haswell_(
        _mm256_andnot_si256(sz_delimiter_cmpge_epu8_haswell_(decoded->window_lo, _mm256_set1_epi8((char)0xF5)),
                            _mm256_set1_epi8((char)0xFF)),
        _mm256_andnot_si256(sz_delimiter_cmpge_epu8_haswell_(decoded->window_hi, _mm256_set1_epi8((char)0xF5)),
                            _mm256_set1_epi8((char)0xFF)));
    sz_u64_t const lead_f0 = sz_utf8_mask_combine_haswell_(
        _mm256_cmpeq_epi8(decoded->window_lo, _mm256_set1_epi8((char)0xF0)),
        _mm256_cmpeq_epi8(decoded->window_hi, _mm256_set1_epi8((char)0xF0)));
    sz_u64_t const lead_f4 = sz_utf8_mask_combine_haswell_(
        _mm256_cmpeq_epi8(decoded->window_lo, _mm256_set1_epi8((char)0xF4)),
        _mm256_cmpeq_epi8(decoded->window_hi, _mm256_set1_epi8((char)0xF4)));
    sz_u64_t const n1_ge_90 = sz_utf8_mask_combine_haswell_(
        sz_delimiter_cmpge_epu8_haswell_(next1_lo, _mm256_set1_epi8((char)0x90)),
        sz_delimiter_cmpge_epu8_haswell_(next1_hi, _mm256_set1_epi8((char)0x90)));
    sz_u64_t const n1_lt_90 = ~n1_ge_90;

    sz_u64_t const ascii = loaded_mask &
                           ~sz_utf8_mask_combine_haswell_(
                               _mm256_cmpeq_epi8(_mm256_and_si256(decoded->window_lo, _mm256_set1_epi8((char)0x80)),
                                                 _mm256_set1_epi8((char)0x80)),
                               _mm256_cmpeq_epi8(_mm256_and_si256(decoded->window_hi, _mm256_set1_epi8((char)0x80)),
                                                 _mm256_set1_epi8((char)0x80)));

    // Spans must lie within the loaded window (so we never validate against a wrapped neighbour or read past loaded).
    sz_u64_t const span2 = loaded >= 1 ? sz_u64_mask_until_serial_(loaded - 1) : 0;
    sz_u64_t const span3 = loaded >= 2 ? sz_u64_mask_until_serial_(loaded - 2) : 0;
    sz_u64_t const span4 = loaded >= 3 ? sz_u64_mask_until_serial_(loaded - 3) : 0;

    sz_u64_t const two_ok = decoded->two_byte_starts & span2 & c1_ok & lead_ge_c2;
    sz_u64_t const three_ok = decoded->three_byte_starts & span3 & c1_ok & c2_ok & ~(lead_e0 & n1_lt_a0) &
                              ~(lead_ed & n1_ge_a0);
    sz_u64_t const four_ok = decoded->four_byte_starts & span4 & c1_ok & c2_ok & c3_ok & lead_le_f4 &
                             ~(lead_f0 & n1_lt_90) & ~(lead_f4 & n1_ge_90);
    return (ascii | two_ok | three_ok | four_ok) & loaded_mask;
}

#pragma endregion Gather free membership

#pragma region Enumerate delimiters

/** @copydoc sz_utf8_delimiters */
SZ_API_COMPTIME sz_size_t sz_utf8_delimiters_haswell(   //
    sz_cptr_t text, sz_size_t length,                   //
    sz_size_t *match_offsets, sz_size_t *match_lengths, //
    sz_size_t matches_capacity, sz_size_t *bytes_consumed) {
    sz_u8_t const *text_u8 = (sz_u8_t const *)text;
    sz_size_t base = 0, count = 0;

    while (base < length && count < matches_capacity) {
        sz_utf8_rune_window_haswell_t const decoded = sz_utf8_rune_decode_window_haswell_(text_u8 + base,
                                                                                          length - base);
        sz_size_t const loaded = decoded.loaded;

        __m256i next1_lo, next1_hi, next2_lo, next2_hi;
        sz_utf8_forward_neighbours_haswell_(decoded.window_lo, decoded.window_hi, &next1_lo, &next1_hi, &next2_lo,
                                            &next2_hi);
        __m256i next3_lo, next3_hi;
        sz_delimiter_forward_neighbour3_haswell_(decoded.window_lo, decoded.window_hi, &next3_lo, &next3_hi);

        // Effective-window trim: a multi-byte lead near the 64-byte edge whose span runs past `loaded` would decode
        // against a wrapped neighbour; defer it to the next window. The lowest overrunning start bounds the span.
        sz_size_t byte_span = loaded;
        if (loaded >= 64) {
            sz_u64_t const overrun = (decoded.two_byte_starts & ~sz_u64_mask_until_serial_(loaded - 1)) |
                                     (decoded.three_byte_starts & ~sz_u64_mask_until_serial_(loaded - 2)) |
                                     (decoded.four_byte_starts & ~sz_u64_mask_until_serial_(loaded - 3));
            byte_span = overrun ? (sz_size_t)sz_u64_ctz(overrun) : loaded;
        }
        sz_u64_t const span_mask = sz_u64_mask_until_serial_(byte_span);

        sz_u64_t const valid_starts = sz_delimiter_valid_starts_haswell_(&decoded, next1_lo, next1_hi, next2_lo,
                                                                         next2_hi, next3_lo, next3_hi) &
                                      decoded.codepoint_starts & span_mask;

        // BMP membership for every lane; astral membership blended onto the four-byte lanes (gated on presence).
        sz_u64_t member = sz_utf8_mask_combine_haswell_(
            sz_delimiter_bmp_membership_haswell_(decoded.window_lo, decoded.high_lo, decoded.low_lo),
            sz_delimiter_bmp_membership_haswell_(decoded.window_hi, decoded.high_hi, decoded.low_hi));
        sz_u64_t const four_byte = decoded.four_byte_starts & span_mask;
        if (four_byte) {
            sz_u64_t const astral_member = sz_utf8_mask_combine_haswell_(
                sz_delimiter_astral_membership_haswell_(decoded.window_lo, next1_lo, next2_lo, next3_lo),
                sz_delimiter_astral_membership_haswell_(decoded.window_hi, next1_hi, next2_hi, next3_hi));
            member = (member & ~four_byte) | (astral_member & four_byte);
        }

        sz_u64_t hits = member & valid_starts;
        while (hits && count < matches_capacity) {
            sz_size_t const lane = (sz_size_t)sz_u64_ctz(hits);
            hits &= hits - 1;
            sz_size_t length_at_lane = 1;
            length_at_lane += (decoded.two_byte_starts >> lane) & 1;
            length_at_lane += ((decoded.three_byte_starts >> lane) & 1) * 2;
            length_at_lane += ((decoded.four_byte_starts >> lane) & 1) * 3;
            match_offsets[count] = base + lane, match_lengths[count] = length_at_lane, ++count;
        }
        if (count == matches_capacity && hits) { // output full mid-window: resume past the last emitted match
            base = match_offsets[count - 1] + match_lengths[count - 1];
            if (bytes_consumed) *bytes_consumed = base;
            return count;
        }
        base += byte_span ? byte_span : 1;
    }

    if (bytes_consumed) *bytes_consumed = base;
    return count;
}

#pragma endregion Enumerate delimiters

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#endif // SZ_USE_HASWELL

#ifdef __cplusplus
}
#endif

#endif // STRINGZILLA_UTF8_TOKENS_HASWELL_H_
