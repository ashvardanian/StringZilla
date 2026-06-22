/**
 *  @brief Haswell (AVX2) backend for UTF-8 newline and whitespace delimiter scanning.
 *  @file include/stringzilla/utf8_tokens/haswell.h
 *  @author Ash Vardanian
 */
#ifndef STRINGZILLA_UTF8_TOKENS_HASWELL_H_
#define STRINGZILLA_UTF8_TOKENS_HASWELL_H_

#include "stringzilla/types.h"
#include "stringzilla/utf8_tokens/serial.h"
#include "stringzilla/utf8_codepoints/haswell.h"

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
SZ_INTERNAL __m256i sz_mm256_cmpge_epu8_haswell_(__m256i a, __m256i b) {
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
SZ_INTERNAL void sz_utf8_iterate_peel_haswell_(                                //
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

SZ_PUBLIC sz_size_t sz_utf8_newlines_haswell(           //
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

SZ_PUBLIC sz_size_t sz_utf8_whitespaces_haswell(        //
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
            x_8d_u8x32 = _mm256_set1_epi8('\x8D'), x_a8_u8x32 = _mm256_set1_epi8('\xA8'),
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

        // 3-byte: E1 9A 80 (ogham); E2 80 [80-8D]; E2 80 AF; E2 81 9F; E2 80 A8/A9; E3 80 80.
        sz_u32_t x_e1_mask = (sz_u32_t)_mm256_movemask_epi8(_mm256_cmpeq_epi8(window, x_e1_u8x32));
        sz_u32_t lead_e2_mask = (sz_u32_t)_mm256_movemask_epi8(_mm256_cmpeq_epi8(window, lead_e2_u8x32));
        sz_u32_t x_e3_mask = (sz_u32_t)_mm256_movemask_epi8(_mm256_cmpeq_epi8(window, x_e3_u8x32));
        sz_u32_t x_9a_mask = (sz_u32_t)_mm256_movemask_epi8(_mm256_cmpeq_epi8(window, x_9a_u8x32));
        sz_u32_t byte_80_mask = (sz_u32_t)_mm256_movemask_epi8(_mm256_cmpeq_epi8(window, byte_80_u8x32));
        sz_u32_t x_81_mask = (sz_u32_t)_mm256_movemask_epi8(_mm256_cmpeq_epi8(window, x_81_u8x32));
        __m256i x_80_ge_cmp = sz_mm256_cmpge_epu8_haswell_(window, byte_80_u8x32); // >= 0x80
        __m256i x_8d_le_cmp = sz_mm256_cmpge_epu8_haswell_(x_8d_u8x32, window);    // <= 0x8D
        sz_u32_t x_8d_range_mask = (sz_u32_t)_mm256_movemask_epi8(_mm256_and_si256(x_80_ge_cmp, x_8d_le_cmp));
        sz_u32_t x_a8_mask = (sz_u32_t)_mm256_movemask_epi8(_mm256_cmpeq_epi8(window, x_a8_u8x32));
        sz_u32_t x_a9_mask = (sz_u32_t)_mm256_movemask_epi8(_mm256_cmpeq_epi8(window, x_a9_u8x32));
        sz_u32_t x_af_mask = (sz_u32_t)_mm256_movemask_epi8(_mm256_cmpeq_epi8(window, x_af_u8x32));
        sz_u32_t x_9f_mask = (sz_u32_t)_mm256_movemask_epi8(_mm256_cmpeq_epi8(window, x_9f_u8x32));

        sz_u32_t ogham_mask = x_e1_mask & (x_9a_mask >> 1) & (byte_80_mask >> 2);               // E1 9A 80
        sz_u32_t range_e280_mask = lead_e2_mask & (byte_80_mask >> 1) & (x_8d_range_mask >> 2); // E2 80 [80-8D]
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
