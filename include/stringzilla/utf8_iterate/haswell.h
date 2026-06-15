/**
 *  @brief Haswell (AVX2) backend for UTF-8 traversal.
 *  @file include/stringzilla/utf8_iterate/haswell.h
 *  @author Ash Vardanian
 *  @sa utf8.h
 */
#ifndef STRINGZILLA_UTF8_ITERATE_HASWELL_H_
#define STRINGZILLA_UTF8_ITERATE_HASWELL_H_

#include "stringzilla/types.h"
#include "stringzilla/utf8_iterate/serial.h"

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
 *  @brief Unsigned byte greater-than-or-equal comparison for AVX2.
 *
 *  AVX2 lacks unsigned comparison intrinsics like `_mm256_cmpge_epu8`.
 *  This uses the identity: a >= b  ⟺  max(a, b) == a
 *  Since `_mm256_max_epu8` treats bytes as unsigned, this gives correct results.
 */
SZ_INTERNAL __m256i sz_mm256_cmpge_epu8_haswell_(__m256i a, __m256i b) {
    return _mm256_cmpeq_epi8(_mm256_max_epu8(a, b), a);
}

#pragma region Multistep newline / whitespace iteration

/*  Multistep newline / whitespace iteration (Haswell / AVX2).
 *
 *  Each 32-byte window is classified branchlessly into a `start_bits` mask (every delimiter start) plus the
 *  per-length 2-byte / 3-byte start masks. AVX2 has no `vpcompressb`, so the peel left-packs with the same
 *  `vpermd` table the sorting backend uses (`sz_sort_haswell_compact4_`): the 32-lane mask is processed in
 *  eight 4-lane sub-blocks, each compacting its `(position+lane, length)` pairs to the front with one
 *  `_mm256_permutevar8x32_epi32` and masked-storing the survivors - no per-match `ctz`, no data-dependent inner
 *  loop. We trust starts in lanes [0,29] and step 30 so any 2-/3-byte delimiter is fully loaded; a 1-byte
 *  `t[pos-1] == '\r'` carry suppresses an LF that completes a CRLF straddling the window edge. */

/** @brief  Mask for `_mm256_maskstore_epi64` selecting the low `count` (0..4) of four 64-bit lanes. */
SZ_INTERNAL __m256i sz_mm256_store_mask_epi64_(sz_size_t count) {
    return _mm256_cmpgt_epi64(_mm256_set1_epi64x((long long)count), _mm256_setr_epi64x(0, 1, 2, 3));
}

/**
 *  @brief  Peel the window's first `emit_count` matches with a `vpermd` left-pack, 4 lanes per sub-block.
 *
 *  Walks the 32-lane `start_bits` mask in eight ascending 4-lane sub-blocks. Each sub-block builds the four
 *  candidate `(position+lane, length)` `u64` pairs, gathers the set lanes to the front with one
 *  `_mm256_permutevar8x32_epi32` (driven by the same left-pack table as `sz_sort_haswell_compact4_`), and
 *  masked-stores `min(popcount, remaining)` of them at the advancing cursor - preserving ascending lane order
 *  and the original `emit_count` truncation byte-for-byte, with no per-match `ctz`.
 */
SZ_INTERNAL void sz_utf8_iterate_peel_window_haswell_(                         //
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
        __m256i const two_byte_add = _mm256_setr_epi64x(
            (two_byte_sub >> 0) & 1u, (two_byte_sub >> 1) & 1u, (two_byte_sub >> 2) & 1u, (two_byte_sub >> 3) & 1u);
        __m256i const three_byte_add = _mm256_setr_epi64x(
            (three_byte_sub >> 0) & 1u, (three_byte_sub >> 1) & 1u, (three_byte_sub >> 2) & 1u,
            (three_byte_sub >> 3) & 1u);
        __m256i const offsets_u64x4 = _mm256_add_epi64(_mm256_set1_epi64x((long long)(position + base_lane)),
                                                       lane_ramp);
        __m256i const lengths_u64x4 = _mm256_add_epi64(
            _mm256_set1_epi64x(1), _mm256_add_epi64(two_byte_add, _mm256_add_epi64(three_byte_add, three_byte_add)));

        __m256i const permutation = _mm256_loadu_si256((__m256i const *)compact_lut[submask]);
        __m256i const packed_offsets = _mm256_permutevar8x32_epi32(offsets_u64x4, permutation);
        __m256i const packed_lengths = _mm256_permutevar8x32_epi32(lengths_u64x4, permutation);

        sz_size_t const taken = sz_min_of_two((sz_size_t)sz_u32_popcount(submask), emit_count - emitted);
        __m256i const store_mask = sz_mm256_store_mask_epi64_(taken);
        _mm256_maskstore_epi64((long long *)(match_offsets + emitted), store_mask, packed_offsets);
        _mm256_maskstore_epi64((long long *)(match_lengths + emitted), store_mask, packed_lengths);
        emitted += taken;
    }
}

SZ_PUBLIC sz_size_t sz_utf8_find_newlines_haswell(      //
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

        sz_size_t const window_matches = (sz_size_t)sz_u32_popcount(start_bits);
        sz_size_t const emit_count = sz_min_of_two(window_matches, matches_capacity - count);
        if (emit_count)
            sz_utf8_iterate_peel_window_haswell_(start_bits, two_byte_starts, three_byte_starts, emit_count, position,
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
    count += sz_utf8_find_newlines_serial_((sz_cptr_t)(text_u8 + position), length - position, position,
                                           match_offsets + count, match_lengths + count, matches_capacity - count,
                                           bytes_consumed);
    return count;
}

SZ_PUBLIC sz_size_t sz_utf8_find_whitespaces_haswell(   //
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

        sz_size_t const window_matches = (sz_size_t)sz_u32_popcount(start_bits);
        sz_size_t const emit_count = sz_min_of_two(window_matches, matches_capacity - count);
        if (emit_count)
            sz_utf8_iterate_peel_window_haswell_(start_bits, two_byte_starts, three_byte_starts, emit_count, position,
                                                 match_offsets + count, match_lengths + count);
        count += emit_count;
        if (count == matches_capacity) {
            position = match_offsets[count - 1] + match_lengths[count - 1];
            break;
        }
        position += 30;
    }

    count += sz_utf8_find_whitespaces_serial_((sz_cptr_t)(text_u8 + position), length - position, position,
                                              match_offsets + count, match_lengths + count, matches_capacity - count,
                                              bytes_consumed);
    return count;
}

#pragma endregion // Multistep newline / whitespace iteration

SZ_PUBLIC sz_size_t sz_utf8_count_haswell(sz_cptr_t text, sz_size_t length) {
    sz_u256_vec_t continuation_mask_vec, continuation_pattern_vec;
    continuation_mask_vec.ymm = _mm256_set1_epi8((char)0xC0);
    continuation_pattern_vec.ymm = _mm256_set1_epi8((char)0x80);

    sz_u8_t const *text_u8 = (sz_u8_t const *)text;
    sz_size_t char_count = 0;

    // Process 32 bytes at a time
    sz_u256_vec_t text_vec, headers_vec;
    while (length >= 32) {
        text_vec.ymm = _mm256_loadu_si256((__m256i const *)text_u8);

        // Apply mask (byte & 0xC0) to extract top 2 bits of each byte
        headers_vec.ymm = _mm256_and_si256(text_vec.ymm, continuation_mask_vec.ymm);

        // Compare with 0x80 (0b10000000) to find continuation bytes
        sz_u32_t start_byte_mask = ~(sz_u32_t)_mm256_movemask_epi8(
            _mm256_cmpeq_epi8(headers_vec.ymm, continuation_pattern_vec.ymm));

        char_count += _mm_popcnt_u32(start_byte_mask);
        text_u8 += 32;
        length -= 32;
    }

    // Process remaining bytes with serial
    char_count += sz_utf8_count_serial((sz_cptr_t)text_u8, length);
    return char_count;
}

SZ_PUBLIC sz_cptr_t sz_utf8_find_nth_haswell(sz_cptr_t text, sz_size_t length, sz_size_t n) {
    // The logic of this function is similar to `sz_utf8_count_haswell`, but uses PDEP
    // instruction in the inner loop to locate Nth character start byte efficiently
    // without one more loop.
    sz_u256_vec_t continuation_mask_vec, continuation_pattern_vec;
    continuation_mask_vec.ymm = _mm256_set1_epi8((char)0xC0);
    continuation_pattern_vec.ymm = _mm256_set1_epi8((char)0x80);

    sz_u8_t const *text_u8 = (sz_u8_t const *)text;

    // Process 32 bytes at a time
    sz_u256_vec_t text_vec, headers_vec;
    while (length >= 32) {
        text_vec.ymm = _mm256_loadu_si256((__m256i const *)text_u8);

        // Apply mask (byte & 0xC0) to extract top 2 bits of each byte
        headers_vec.ymm = _mm256_and_si256(text_vec.ymm, continuation_mask_vec.ymm);

        // Compare with 0x80 (0b10000000) to find continuation bytes
        sz_u32_t start_byte_mask = ~(sz_u32_t)_mm256_movemask_epi8(
            _mm256_cmpeq_epi8(headers_vec.ymm, continuation_pattern_vec.ymm));
        sz_size_t start_byte_count = _mm_popcnt_u32(start_byte_mask);

        // Check if we've reached the terminal part of our search
        if (n < start_byte_count) {
            // PDEP directly gives us the nth set bit position
            // Example: _pdep_u32(0b10, 0b00010101) = 0b00000100
            sz_u32_t deposited_bits = _pdep_u32((sz_u32_t)1 << n, start_byte_mask);
            int byte_offset = sz_u32_ctz(deposited_bits);
            return (sz_cptr_t)(text_u8 + byte_offset);
        }
        // Jump to the next block
        else {
            n -= start_byte_count;
            text_u8 += 32;
            length -= 32;
        }
    }

    // Process remaining bytes with serial
    return sz_utf8_find_nth_serial((sz_cptr_t)text_u8, length, n);
}

/*  UAX-29 word boundary detection (vectorized).
 *
 *  The serial reference walks the text codepoint-by-codepoint and, at each candidate position,
 *  evaluates `sz_utf8_is_word_boundary_serial`. The dominant runtime cost for typical (ASCII / Latin)
 *  text is the per-codepoint Word_Break property classification.
 *
 *  The Haswell backend keeps the same outer walk but accelerates it in two ways while remaining
 *  byte-exact versus the serial reference:
 *
 *   1) Vectorized property classification. A 32-byte window is classified with `_mm256_shuffle_epi8`.
 *      ASCII bytes (<0x80) are mapped to their Word_Break property through an 8-row nibble LUT (one
 *      `_mm256_shuffle_epi8` per high-nibble row, selected by an equality mask). The full ASCII
 *      property table is an 8x16 grid, so a single shuffle is insufficient; eight row-shuffles
 *      reproduce it exactly. Continuation bytes (0x80-0xBF) and non-ASCII lead bytes are tagged with
 *      a sentinel so the scalar fallback handles them. The result is cached into a small stack window
 *      of per-byte properties.
 *
 *   2) Vectorized local boundary decision. For positions whose previous and following effective
 *      properties are both "plain" (Other/CR/LF/Newline/ALetter/Hebrew/Numeric/Katakana/ExtendNumLet)
 *      the boundary is fully determined by the immediate pair via rules WB3/3a/3b/5/8/9/10/13/13a/13b
 *      (encoded in `sz_utf8_word_break_pair_decision_`). These are resolved without any look-around.
 *
 *  Every position that could be governed by a stateful rule -- WB4 (Extend/Format/ZWJ skipping),
 *  WB6/WB7 (MidLetter look-around), WB7a, WB11/WB12 (MidNum look-around), WB15/WB16 (Regional
 *  Indicator parity), the emoji-ZWJ rule WB3c, or any non-ASCII / window-edge position where local
 *  context is insufficient -- defers to `sz_utf8_is_word_boundary_serial`, which re-reads the full
 *  `text`/`length` and is therefore always exact. Sub-rules left to serial: WB3c, WB4, WB6, WB7,
 *  WB7a, WB11, WB12, WB15, WB16 (and all non-BMP / SMP property cases).
 */

/* Per-byte Word_Break property of ASCII codepoints, laid out as eight 16-entry rows indexed by the
 * low nibble; row R serves high nibble R (covering bytes 0x00-0x7F). */
SZ_INTERNAL __m256i sz_utf8_word_break_classify_ascii_haswell_(__m256i bytes) {
    // Eight rows of the ASCII Word_Break property table (high nibble → 16 low-nibble entries).
    __m256i row0 = _mm256_setr_epi8(                    //
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 3, 3, 1, 0, 0, //
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 3, 3, 1, 0, 0);
    __m256i row1 = _mm256_setzero_si256();
    __m256i row2 = _mm256_setr_epi8(                        //
        0, 0, 15, 0, 0, 0, 0, 15, 0, 0, 0, 0, 14, 0, 15, 0, //
        0, 0, 15, 0, 0, 0, 0, 15, 0, 0, 0, 0, 14, 0, 15, 0);
    __m256i row3 = _mm256_setr_epi8(                                //
        10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 13, 14, 0, 0, 0, 0, //
        10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 13, 14, 0, 0, 0, 0);
    __m256i row4 = _mm256_setr_epi8(                    //
        0, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, //
        0, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8);
    __m256i row5 = _mm256_setr_epi8(                     //
        8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 0, 0, 0, 0, 12, //
        8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 0, 0, 0, 0, 12);
    __m256i row6 = _mm256_setr_epi8(                    //
        0, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, //
        0, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8);
    __m256i row7 = _mm256_setr_epi8(                    //
        8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 0, 0, 0, 0, 0, //
        8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 0, 0, 0, 0, 0);

    __m256i lo_nibble = _mm256_and_si256(bytes, _mm256_set1_epi8(0x0F));
    __m256i hi_nibble = _mm256_and_si256(_mm256_srli_epi16(bytes, 4), _mm256_set1_epi8(0x0F));

    __m256i result = _mm256_setzero_si256();
    __m256i shuf0 = _mm256_shuffle_epi8(row0, lo_nibble);
    __m256i shuf1 = _mm256_shuffle_epi8(row1, lo_nibble);
    __m256i shuf2 = _mm256_shuffle_epi8(row2, lo_nibble);
    __m256i shuf3 = _mm256_shuffle_epi8(row3, lo_nibble);
    __m256i shuf4 = _mm256_shuffle_epi8(row4, lo_nibble);
    __m256i shuf5 = _mm256_shuffle_epi8(row5, lo_nibble);
    __m256i shuf6 = _mm256_shuffle_epi8(row6, lo_nibble);
    __m256i shuf7 = _mm256_shuffle_epi8(row7, lo_nibble);

    result = _mm256_blendv_epi8(result, shuf0, _mm256_cmpeq_epi8(hi_nibble, _mm256_set1_epi8(0)));
    result = _mm256_blendv_epi8(result, shuf1, _mm256_cmpeq_epi8(hi_nibble, _mm256_set1_epi8(1)));
    result = _mm256_blendv_epi8(result, shuf2, _mm256_cmpeq_epi8(hi_nibble, _mm256_set1_epi8(2)));
    result = _mm256_blendv_epi8(result, shuf3, _mm256_cmpeq_epi8(hi_nibble, _mm256_set1_epi8(3)));
    result = _mm256_blendv_epi8(result, shuf4, _mm256_cmpeq_epi8(hi_nibble, _mm256_set1_epi8(4)));
    result = _mm256_blendv_epi8(result, shuf5, _mm256_cmpeq_epi8(hi_nibble, _mm256_set1_epi8(5)));
    result = _mm256_blendv_epi8(result, shuf6, _mm256_cmpeq_epi8(hi_nibble, _mm256_set1_epi8(6)));
    result = _mm256_blendv_epi8(result, shuf7, _mm256_cmpeq_epi8(hi_nibble, _mm256_set1_epi8(7)));
    return result;
}

/* Per-class lane mask: bit i set where lane i has Word_Break property `value`. */
SZ_INTERNAL sz_u32_t sz_utf8_word_break_class_mask_haswell_(__m256i classes, int value) {
    return (sz_u32_t)_mm256_movemask_epi8(_mm256_cmpeq_epi8(classes, _mm256_set1_epi8((char)value)));
}

/* 32-bit "joined" (guaranteed non-boundary) mask for an all-ASCII 32-byte window: bit i set => the boundary
 * before lane i is suppressed by a UAX-29 no-break rule. ASCII has no Extend/Format/ZWJ/Regional_Indicator/
 * Hebrew/Katakana, so WB4 and WB15/16 never apply and WB6/7/11/12 reduce to neighbour bit-shifts (`<< 1` =
 * previous lane, `>> 1` = next, `<< 2` = two back). Exact for lanes whose i-2 and i+1 neighbours are in-window. */
SZ_INTERNAL sz_u32_t sz_utf8_word_break_join_mask_ascii_haswell_(__m256i classes) {
    // Reduce each class to a 32-lane bitmask and defer the rule logic to the shared portable routine; the
    // caller restricts the result to its trusted lane window, so the wider u64 math is harmless.
    return (sz_u32_t)sz_utf8_word_break_join_from_class_masks_(
        sz_utf8_word_break_class_mask_haswell_(classes, sz_tr29_word_break_aletter_k),
        sz_utf8_word_break_class_mask_haswell_(classes, sz_tr29_word_break_numeric_k),
        sz_utf8_word_break_class_mask_haswell_(classes, sz_tr29_word_break_extendnumlet_k),
        sz_utf8_word_break_class_mask_haswell_(classes, sz_tr29_word_break_midletter_k),
        sz_utf8_word_break_class_mask_haswell_(classes, sz_tr29_word_break_midnum_k),
        sz_utf8_word_break_class_mask_haswell_(classes, sz_tr29_word_break_mid_quotes_k),
        sz_utf8_word_break_class_mask_haswell_(classes, sz_tr29_word_break_cr_k),
        sz_utf8_word_break_class_mask_haswell_(classes, sz_tr29_word_break_lf_k));
}

/* Per-file copy of the sorting backend's left-pack table for compacting u64 lanes: row `[m]` holds the 8 dword
 * indices that gather the `m`-selected u64 lanes (of 4, each a dword pair) to the front for
 * `_mm256_permutevar8x32_epi32`. Identical to the table in `sz_utf8_iterate_peel_window_haswell_`. */
SZ_INTERNAL __m256i sz_utf8_word_compact4_permutation_haswell_(sz_u32_t submask) {
    static sz_u32_t const compact_lut[16][8] = {
        {0, 0, 0, 0, 0, 0, 0, 0}, {0, 1, 0, 0, 0, 0, 0, 0}, {2, 3, 0, 0, 0, 0, 0, 0}, {0, 1, 2, 3, 0, 0, 0, 0},
        {4, 5, 0, 0, 0, 0, 0, 0}, {0, 1, 4, 5, 0, 0, 0, 0}, {2, 3, 4, 5, 0, 0, 0, 0}, {0, 1, 2, 3, 4, 5, 0, 0},
        {6, 7, 0, 0, 0, 0, 0, 0}, {0, 1, 6, 7, 0, 0, 0, 0}, {2, 3, 6, 7, 0, 0, 0, 0}, {0, 1, 2, 3, 6, 7, 0, 0},
        {4, 5, 6, 7, 0, 0, 0, 0}, {0, 1, 4, 5, 6, 7, 0, 0}, {2, 3, 4, 5, 6, 7, 0, 0}, {0, 1, 2, 3, 4, 5, 6, 7},
    };
    return _mm256_loadu_si256((__m256i const *)compact_lut[submask & 0xFu]);
}

/** @brief  Descending-order counterpart of `sz_utf8_word_compact4_permutation_haswell_`: row `[m]` gathers the
 *          `m`-selected u64 lanes (of 4) to the front in HIGH-to-LOW lane order, for the reverse word scan. */
SZ_INTERNAL __m256i sz_utf8_word_compact4_permutation_descending_haswell_(sz_u32_t submask) {
    static sz_u32_t const compact_lut[16][8] = {
        {0, 0, 0, 0, 0, 0, 0, 0}, {0, 1, 0, 0, 0, 0, 0, 0}, {2, 3, 0, 0, 0, 0, 0, 0}, {2, 3, 0, 1, 0, 0, 0, 0},
        {4, 5, 0, 0, 0, 0, 0, 0}, {4, 5, 0, 1, 0, 0, 0, 0}, {4, 5, 2, 3, 0, 0, 0, 0}, {4, 5, 2, 3, 0, 1, 0, 0},
        {6, 7, 0, 0, 0, 0, 0, 0}, {6, 7, 0, 1, 0, 0, 0, 0}, {6, 7, 2, 3, 0, 0, 0, 0}, {6, 7, 2, 3, 0, 1, 0, 0},
        {6, 7, 4, 5, 0, 0, 0, 0}, {6, 7, 4, 5, 0, 1, 0, 0}, {6, 7, 4, 5, 2, 3, 0, 0}, {6, 7, 4, 5, 2, 3, 0, 1},
    };
    return _mm256_loadu_si256((__m256i const *)compact_lut[submask & 0xFu]);
}

SZ_PUBLIC sz_size_t sz_utf8_word_find_boundaries_haswell( //
    sz_cptr_t text, sz_size_t length,                     //
    sz_size_t *word_starts, sz_size_t *word_lengths,      //
    sz_size_t words_capacity, sz_size_t *bytes_consumed) {

    sz_size_t words = 0;
    if (length == 0 || words_capacity == 0) {
        if (bytes_consumed) *bytes_consumed = 0;
        return 0;
    }

    sz_u8_t const *text_u8 = (sz_u8_t const *)text;
    sz_size_t word_start = 0; // Start of the word currently being accumulated (always a boundary).
    // Skip first codepoint (position 0 is always a boundary), matching the serial reference.
    sz_size_t position = sz_utf8_codepoint_length_(text_u8[0]);

    // Oracle-free fast path: an all-ASCII window [position-2, position+30) resolves boundaries at positions
    // [position, position+28]; one fixed sub-block loop compacts each group and emits it as a shifted-difference,
    // carrying the open `word_start` into lane 0 and the previous boundary into lanes 1..3.
    __m256i const lane_ramp = _mm256_setr_epi64x(0, 1, 2, 3);
    __m256i const lane_shift_right = _mm256_setr_epi32(0, 0, 0, 1, 2, 3, 4, 5);
    while (position < length) {
        int ascii_window = position >= 2 && position + 30 <= length;
        __m256i window = _mm256_setzero_si256();
        if (ascii_window) {
            window = _mm256_loadu_si256((__m256i const *)(text_u8 + position - 2)); // lane j = byte position-2+j
            ascii_window = _mm256_movemask_epi8(window) == 0;
        }
        if (!ascii_window) { // Non-ASCII window or near the edges: one scalar codepoint step.
            if (sz_utf8_is_word_boundary_serial(text, length, position)) {
                if (words == words_capacity) {
                    if (bytes_consumed) *bytes_consumed = word_start;
                    return words;
                }
                word_starts[words] = word_start, word_lengths[words] = position - word_start, ++words;
                word_start = position;
            }
            position += sz_utf8_codepoint_length_(text_u8[position]);
            continue;
        }

        sz_u32_t boundary =
            (~sz_utf8_word_break_join_mask_ascii_haswell_(sz_utf8_word_break_classify_ascii_haswell_(window))) &
            0x7FFFFFFCu; // trusted lanes [2,30]
        for (sz_size_t sub_block = 0; sub_block < 8; ++sub_block) {
            sz_u32_t const submask = (boundary >> (sub_block * 4)) & 0xFu;
            if (!submask) continue;
            sz_size_t const taken = (sz_size_t)sz_u32_popcount(submask);
            sz_size_t const stored = sz_min_of_two(taken, words_capacity - words);

            __m256i const positions =
                _mm256_add_epi64(_mm256_set1_epi64x((long long)(position - 2 + sub_block * 4)), lane_ramp);
            __m256i const boundaries =
                _mm256_permutevar8x32_epi32(positions, sz_utf8_word_compact4_permutation_haswell_(submask));
            __m256i const starts = _mm256_blend_epi32(_mm256_permutevar8x32_epi32(boundaries, lane_shift_right),
                                                      _mm256_set1_epi64x((long long)word_start), 0x03);
            __m256i const store_mask = sz_mm256_store_mask_epi64_(stored);
            _mm256_maskstore_epi64((long long *)(word_starts + words), store_mask, starts);
            _mm256_maskstore_epi64((long long *)(word_lengths + words), store_mask,
                                   _mm256_sub_epi64(boundaries, starts));
            words += stored;
            if (stored) word_start = word_starts[words - 1] + word_lengths[words - 1];
            if (words == words_capacity) {
                if (bytes_consumed) *bytes_consumed = word_start;
                return words;
            }
        }
        position += 29; // Resolved [position, position+28]; next unresolved boundary is at position+29.
    }

    if (words == words_capacity) {
        if (bytes_consumed) *bytes_consumed = word_start;
        return words;
    }
    word_starts[words] = word_start;
    word_lengths[words] = length - word_start;
    ++words;
    if (bytes_consumed) *bytes_consumed = length;
    return words;
}

SZ_PUBLIC sz_size_t sz_utf8_word_rfind_boundaries_haswell( //
    sz_cptr_t text, sz_size_t length,                      //
    sz_size_t *word_starts, sz_size_t *word_lengths,       //
    sz_size_t words_capacity, sz_size_t *bytes_consumed) {

    sz_size_t words = 0;
    if (length == 0 || words_capacity == 0) {
        if (bytes_consumed) *bytes_consumed = length;
        return 0;
    }

    sz_u8_t const *text_u8 = (sz_u8_t const *)text;
    sz_size_t word_end = length; // End of the word currently being accumulated (always a boundary).
    sz_size_t position = length - 1;
    while (position > 0 && (text_u8[position] & 0xC0) == 0x80) position--;

    // Oracle-free fast path: an all-ASCII window [position-30, position+2) resolves boundaries at positions
    // [position-28, position]; one fixed sub-block loop walks high-to-low, compacting each group in descending
    // lane order and emitting it as a shifted-difference (lane 0 carries the open `word_end`).
    __m256i const lane_ramp = _mm256_setr_epi64x(0, 1, 2, 3);
    __m256i const lane_shift_right = _mm256_setr_epi32(0, 0, 0, 1, 2, 3, 4, 5);
    while (position > 0) {
        sz_size_t base = position - 30; // lane j = byte base+j; trusted lanes [2,30] → [position-28, position]
        int ascii_window = position >= 30 && position + 2 <= length;
        __m256i window = _mm256_setzero_si256();
        if (ascii_window) {
            window = _mm256_loadu_si256((__m256i const *)(text_u8 + base));
            ascii_window = _mm256_movemask_epi8(window) == 0;
        }
        if (!ascii_window) { // Non-ASCII window or near the edges: one scalar codepoint step.
            if (sz_utf8_is_word_boundary_serial(text, length, position)) {
                if (words == words_capacity) {
                    if (bytes_consumed) *bytes_consumed = word_end;
                    return words;
                }
                word_starts[words] = position, word_lengths[words] = word_end - position, ++words;
                word_end = position;
            }
            position--;
            while (position > 0 && (text_u8[position] & 0xC0) == 0x80) position--;
            continue;
        }

        sz_u32_t boundary =
            (~sz_utf8_word_break_join_mask_ascii_haswell_(sz_utf8_word_break_classify_ascii_haswell_(window))) &
            0x7FFFFFFCu; // trusted lanes [2,30]
        for (sz_size_t sub_block = 8; sub_block-- > 0;) { // high-to-low for descending emission
            sz_u32_t const submask = (boundary >> (sub_block * 4)) & 0xFu;
            if (!submask) continue;
            sz_size_t const taken = (sz_size_t)sz_u32_popcount(submask);
            sz_size_t const stored = sz_min_of_two(taken, words_capacity - words);

            __m256i const positions =
                _mm256_add_epi64(_mm256_set1_epi64x((long long)(base + sub_block * 4)), lane_ramp);
            __m256i const boundaries = _mm256_permutevar8x32_epi32(
                positions, sz_utf8_word_compact4_permutation_descending_haswell_(submask));
            __m256i const previous = _mm256_blend_epi32(_mm256_permutevar8x32_epi32(boundaries, lane_shift_right),
                                                        _mm256_set1_epi64x((long long)word_end), 0x03);
            __m256i const store_mask = sz_mm256_store_mask_epi64_(stored);
            _mm256_maskstore_epi64((long long *)(word_starts + words), store_mask, boundaries);
            _mm256_maskstore_epi64((long long *)(word_lengths + words), store_mask,
                                   _mm256_sub_epi64(previous, boundaries));
            words += stored;
            if (stored) word_end = word_starts[words - 1];
            if (words == words_capacity) {
                if (bytes_consumed) *bytes_consumed = word_end;
                return words;
            }
        }
        position = base + 1; // Resolved down to position-28; next unresolved boundary is at position-29.
    }

    if (words == words_capacity) {
        if (bytes_consumed) *bytes_consumed = word_end;
        return words;
    }
    word_starts[words] = 0;
    word_lengths[words] = word_end;
    ++words;
    if (bytes_consumed) *bytes_consumed = 0;
    return words;
}

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#endif // SZ_USE_HASWELL

#ifdef __cplusplus
}
#endif

#endif // STRINGZILLA_UTF8_ITERATE_HASWELL_H_
