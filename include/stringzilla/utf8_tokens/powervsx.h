/**
 *  @brief POWER VSX backend for UTF-8 newline and whitespace delimiter scanning.
 *  @file include/stringzilla/utf8_tokens/powervsx.h
 *  @author Ash Vardanian
 */
#ifndef STRINGZILLA_UTF8_TOKENS_POWERVSX_H_
#define STRINGZILLA_UTF8_TOKENS_POWERVSX_H_

#include "stringzilla/types.h"
#include "stringzilla/utf8_tokens/serial.h"
#include "stringzilla/utf8_runes/powervsx.h"

#ifdef __cplusplus
extern "C" {
#endif

#if SZ_USE_POWERVSX
#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("power9-vector"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("power9-vector")
#endif

/**
 *  @brief Peel the window's first `emit_count` matches by SIMD left-pack (no `ctz`, no per-match branch).
 *         Each ascending 2-lane sub-block gathers its set lanes' `(position+lane, length)` `u64` pairs with one
 *         `vec_perm` and full-stores to an 18-wide scratch (absorbing the last sub-block's 2-lane spill, since
 *         VSX has no masked store); the low `emit_count` entries copy out in ascending lane order, byte-exact.
 */
SZ_HELPER_AUTO void sz_utf8_iterate_peel_powervsx_(                            //
    sz_u32_t start_bits, sz_u32_t two_byte_starts, sz_u32_t three_byte_starts, //
    sz_size_t emit_count, sz_size_t position,                                  //
    sz_size_t *match_offsets, sz_size_t *match_lengths) {

    // Byte-permutation rows for the four 2-bit sub-block masks (lane 0 = bytes [0,8), lane 1 = bytes [8,16)):
    // row `[m]` gathers the `m`-selected `u64` lanes to the front of a `vector unsigned long long` via `vec_perm`.
    // Rows in mask order: none (unused identity), lane 0 only, lane 1 only, both lanes.
    static unsigned char const compact2_lut[4][16] = {
        {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
        {0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7},
        {8, 9, 10, 11, 12, 13, 14, 15, 8, 9, 10, 11, 12, 13, 14, 15},
        {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
    };
    static sz_size_t const popcount2_lut[4] = {0, 1, 1, 2};

    sz_size_t scratch_offsets[18], scratch_lengths[18];
    sz_size_t filled = 0;
    for (sz_size_t sub_block = 0; sub_block < 8; ++sub_block) {
        sz_size_t const base_lane = sub_block * 2;
        sz_u32_t const submask = (start_bits >> base_lane) & 0x3u;
        if (!submask) continue;

        // Per-lane length: 1, plus 1 on a 2-byte start, plus 2 on a 3-byte start (the masks are disjoint).
        sz_u32_t const two_byte_sub = (two_byte_starts >> base_lane) & 0x3u;
        sz_u32_t const three_byte_sub = (three_byte_starts >> base_lane) & 0x3u;
        __vector unsigned long long const candidate_offsets = {(unsigned long long)(position + base_lane),
                                                               (unsigned long long)(position + base_lane + 1)};
        __vector unsigned long long const candidate_lengths = {
            1u + (two_byte_sub & 1u) + 2u * (three_byte_sub & 1u),
            1u + ((two_byte_sub >> 1) & 1u) + 2u * ((three_byte_sub >> 1) & 1u)};

        __vector unsigned char const permutation = vec_xl(0, (unsigned char const *)compact2_lut[submask]);
        __vector unsigned long long const packed_offsets = (__vector unsigned long long)vec_perm(
            (__vector unsigned char)candidate_offsets, (__vector unsigned char)candidate_offsets, permutation);
        __vector unsigned long long const packed_lengths = (__vector unsigned long long)vec_perm(
            (__vector unsigned char)candidate_lengths, (__vector unsigned char)candidate_lengths, permutation);
        vec_xst(packed_offsets, 0, (unsigned long long *)(scratch_offsets + filled));
        vec_xst(packed_lengths, 0, (unsigned long long *)(scratch_lengths + filled));
        filled += popcount2_lut[submask];
    }

    for (sz_size_t emitted = 0; emitted < emit_count; ++emitted)
        match_offsets[emitted] = scratch_offsets[emitted], match_lengths[emitted] = scratch_lengths[emitted];
}

SZ_API_COMPTIME sz_size_t sz_utf8_newlines_powervsx(    //
    sz_cptr_t text, sz_size_t length,                   //
    sz_size_t *match_offsets, sz_size_t *match_lengths, //
    sz_size_t matches_capacity, sz_size_t *bytes_consumed) {

    sz_u8_t const *text_u8 = (sz_u8_t const *)text;
    sz_size_t count = 0, position = 0;

    __vector unsigned char const newline_vec = vec_splats((unsigned char)'\n');
    __vector unsigned char const vertical_tab_vec = vec_splats((unsigned char)'\v');
    __vector unsigned char const form_feed_vec = vec_splats((unsigned char)'\f');
    __vector unsigned char const carriage_return_vec = vec_splats((unsigned char)'\r');
    __vector unsigned char const lead_c2_vec = vec_splats((unsigned char)0xC2);
    __vector unsigned char const x_85_vec = vec_splats((unsigned char)0x85);
    __vector unsigned char const lead_e2_vec = vec_splats((unsigned char)0xE2);
    __vector unsigned char const byte_80_vec = vec_splats((unsigned char)0x80);
    __vector unsigned char const x_a8_vec = vec_splats((unsigned char)0xA8);
    __vector unsigned char const x_a9_vec = vec_splats((unsigned char)0xA9);

    while (position + 16 <= length && count < matches_capacity) {
        __vector unsigned char window = vec_xl(0, text_u8 + position);
        sz_u32_t newline_bits = sz_utf8_iterate_movemask_powervsx_(
            (__vector unsigned char)vec_cmpeq(window, newline_vec));
        sz_u32_t carriage_return_bits = sz_utf8_iterate_movemask_powervsx_(
            (__vector unsigned char)vec_cmpeq(window, carriage_return_vec));
        sz_u32_t one_byte_bits =
            newline_bits | carriage_return_bits |
            sz_utf8_iterate_movemask_powervsx_((__vector unsigned char)vec_cmpeq(window, vertical_tab_vec)) |
            sz_utf8_iterate_movemask_powervsx_((__vector unsigned char)vec_cmpeq(window, form_feed_vec));

        // 2-byte NEL (C2 85); 3-byte LS/PS (E2 80 A8/A9) - bit `i+1` is the next lane, so suffixes shift right.
        sz_u32_t lead_c2_bits = sz_utf8_iterate_movemask_powervsx_(
            (__vector unsigned char)vec_cmpeq(window, lead_c2_vec));
        sz_u32_t x_85_bits = sz_utf8_iterate_movemask_powervsx_((__vector unsigned char)vec_cmpeq(window, x_85_vec));
        sz_u32_t nel_bits = lead_c2_bits & (x_85_bits >> 1);
        sz_u32_t lead_e2_bits = sz_utf8_iterate_movemask_powervsx_(
            (__vector unsigned char)vec_cmpeq(window, lead_e2_vec));
        sz_u32_t byte_80_bits = sz_utf8_iterate_movemask_powervsx_(
            (__vector unsigned char)vec_cmpeq(window, byte_80_vec));
        sz_u32_t lead_e280_bits = lead_e2_bits & (byte_80_bits >> 1);
        sz_u32_t x_a8_bits = sz_utf8_iterate_movemask_powervsx_((__vector unsigned char)vec_cmpeq(window, x_a8_vec));
        sz_u32_t x_a9_bits = sz_utf8_iterate_movemask_powervsx_((__vector unsigned char)vec_cmpeq(window, x_a9_vec));
        sz_u32_t line_para_bits = lead_e280_bits & ((x_a8_bits | x_a9_bits) >> 2);

        // CRLF: a CR whose next lane is LF is a single 2-byte match; its trailing LF must not also be emitted.
        sz_u32_t crlf_bits = carriage_return_bits & (newline_bits >> 1);
        sz_u32_t lf_of_crlf_bits = newline_bits & (carriage_return_bits << 1);

        sz_u32_t two_byte_starts = crlf_bits | nel_bits;
        sz_u32_t three_byte_starts = line_para_bits;
        sz_u32_t start_bits = (one_byte_bits | nel_bits | line_para_bits) & ~lf_of_crlf_bits;
        start_bits &= (sz_u32_t)0x3FFF; // trust lanes [0,13]; step 14

        // Suppress a leading LF already consumed by a CRLF that straddled the previous window edge.
        if (position != 0 && text_u8[position - 1] == '\r') start_bits &= ~(newline_bits & (sz_u32_t)1);

        sz_size_t const window_matches = (sz_size_t)sz_u32_popcount(start_bits);
        sz_size_t const emit_count = sz_min_of_two(window_matches, matches_capacity - count);
        if (emit_count)
            sz_utf8_iterate_peel_powervsx_(start_bits, two_byte_starts, three_byte_starts, emit_count, position,
                                           match_offsets + count, match_lengths + count);
        count += emit_count;
        if (count == matches_capacity) { // output buffer full: resume past the last emitted match.
            position = match_offsets[count - 1] + match_lengths[count - 1];
            break;
        }
        position += 14;
    }

    // Skip a CRLF's trailing LF if it straddles into the serial tail (the CR was emitted as a 2-byte match).
    if (position != 0 && position < length && text_u8[position - 1] == '\r' && text_u8[position] == '\n') ++position;
    count += sz_utf8_newlines_serial_((sz_cptr_t)(text_u8 + position), length - position, position,
                                      match_offsets + count, match_lengths + count, matches_capacity - count,
                                      bytes_consumed);
    return count;
}

SZ_API_COMPTIME sz_size_t sz_utf8_whitespaces_powervsx( //
    sz_cptr_t text, sz_size_t length,                   //
    sz_size_t *match_offsets, sz_size_t *match_lengths, //
    sz_size_t matches_capacity, sz_size_t *bytes_consumed) {

    sz_u8_t const *text_u8 = (sz_u8_t const *)text;
    sz_size_t count = 0, position = 0;

    __vector unsigned char const tab_vec = vec_splats((unsigned char)'\t');
    __vector unsigned char const carriage_return_vec = vec_splats((unsigned char)'\r');
    __vector unsigned char const x_20_vec = vec_splats((unsigned char)' ');
    __vector unsigned char const lead_c2_vec = vec_splats((unsigned char)0xC2);
    __vector unsigned char const x_85_vec = vec_splats((unsigned char)0x85);
    __vector unsigned char const x_a0_vec = vec_splats((unsigned char)0xA0);
    __vector unsigned char const x_e1_vec = vec_splats((unsigned char)0xE1);
    __vector unsigned char const lead_e2_vec = vec_splats((unsigned char)0xE2);
    __vector unsigned char const x_e3_vec = vec_splats((unsigned char)0xE3);
    __vector unsigned char const x_9a_vec = vec_splats((unsigned char)0x9A);
    __vector unsigned char const byte_80_vec = vec_splats((unsigned char)0x80);
    __vector unsigned char const x_81_vec = vec_splats((unsigned char)0x81);
    __vector unsigned char const x_8a_vec = vec_splats((unsigned char)0x8A);
    __vector unsigned char const x_a8_vec = vec_splats((unsigned char)0xA8);
    __vector unsigned char const x_a9_vec = vec_splats((unsigned char)0xA9);
    __vector unsigned char const x_af_vec = vec_splats((unsigned char)0xAF);
    __vector unsigned char const x_9f_vec = vec_splats((unsigned char)0x9F);

    while (position + 16 <= length && count < matches_capacity) {
        __vector unsigned char window = vec_xl(0, text_u8 + position);
        // 1-byte: space, plus the contiguous range [\t, \r] == [9, 13].
        sz_u32_t one_byte_bits =
            sz_utf8_iterate_movemask_powervsx_((__vector unsigned char)vec_cmpeq(window, x_20_vec)) |
            sz_utf8_iterate_movemask_powervsx_(vec_and((__vector unsigned char)vec_cmpge(window, tab_vec),
                                                       (__vector unsigned char)vec_cmple(window, carriage_return_vec)));

        // 2-byte: C2 85 (NEL), C2 A0 (NBSP).
        sz_u32_t lead_c2_bits = sz_utf8_iterate_movemask_powervsx_(
            (__vector unsigned char)vec_cmpeq(window, lead_c2_vec));
        sz_u32_t x_85_bits = sz_utf8_iterate_movemask_powervsx_((__vector unsigned char)vec_cmpeq(window, x_85_vec));
        sz_u32_t x_a0_bits = sz_utf8_iterate_movemask_powervsx_((__vector unsigned char)vec_cmpeq(window, x_a0_vec));
        sz_u32_t two_byte_starts = lead_c2_bits & ((x_85_bits >> 1) | (x_a0_bits >> 1));

        // 3-byte: E1 9A 80 (ogham); E2 80 [80-8A]; E2 80 AF; E2 81 9F; E2 80 A8/A9; E3 80 80.
        sz_u32_t byte_80_bits = sz_utf8_iterate_movemask_powervsx_(
            (__vector unsigned char)vec_cmpeq(window, byte_80_vec));
        sz_u32_t lead_e2_bits = sz_utf8_iterate_movemask_powervsx_(
            (__vector unsigned char)vec_cmpeq(window, lead_e2_vec));
        sz_u32_t lead_e280_bits = lead_e2_bits & (byte_80_bits >> 1);
        sz_u32_t x_e1_bits = sz_utf8_iterate_movemask_powervsx_((__vector unsigned char)vec_cmpeq(window, x_e1_vec));
        sz_u32_t x_9a_bits = sz_utf8_iterate_movemask_powervsx_((__vector unsigned char)vec_cmpeq(window, x_9a_vec));
        sz_u32_t ogham_bits = x_e1_bits & (x_9a_bits >> 1) & (byte_80_bits >> 2);
        sz_u32_t x_80_ge_bits = sz_utf8_iterate_movemask_powervsx_(
            (__vector unsigned char)vec_cmpge(window, byte_80_vec));
        sz_u32_t x_8a_le_bits = sz_utf8_iterate_movemask_powervsx_((__vector unsigned char)vec_cmple(window, x_8a_vec));
        sz_u32_t range_e280_bits = lead_e280_bits & (x_80_ge_bits >> 2) & (x_8a_le_bits >> 2);
        sz_u32_t x_af_bits = sz_utf8_iterate_movemask_powervsx_((__vector unsigned char)vec_cmpeq(window, x_af_vec));
        sz_u32_t nnbsp_bits = lead_e280_bits & (x_af_bits >> 2);
        sz_u32_t x_81_bits = sz_utf8_iterate_movemask_powervsx_((__vector unsigned char)vec_cmpeq(window, x_81_vec));
        sz_u32_t x_9f_bits = sz_utf8_iterate_movemask_powervsx_((__vector unsigned char)vec_cmpeq(window, x_9f_vec));
        sz_u32_t mmsp_bits = lead_e2_bits & (x_81_bits >> 1) & (x_9f_bits >> 2);
        sz_u32_t x_a8_bits = sz_utf8_iterate_movemask_powervsx_((__vector unsigned char)vec_cmpeq(window, x_a8_vec));
        sz_u32_t x_a9_bits = sz_utf8_iterate_movemask_powervsx_((__vector unsigned char)vec_cmpeq(window, x_a9_vec));
        sz_u32_t line_bits = lead_e280_bits & (x_a8_bits >> 2);
        sz_u32_t para_bits = lead_e280_bits & (x_a9_bits >> 2);
        sz_u32_t x_e3_bits = sz_utf8_iterate_movemask_powervsx_((__vector unsigned char)vec_cmpeq(window, x_e3_vec));
        sz_u32_t ideographic_bits = x_e3_bits & (byte_80_bits >> 1) & (byte_80_bits >> 2);
        sz_u32_t three_byte_starts = ogham_bits | range_e280_bits | nnbsp_bits | mmsp_bits | line_bits | para_bits |
                                     ideographic_bits;

        sz_u32_t start_bits = (one_byte_bits | two_byte_starts | three_byte_starts) & (sz_u32_t)0x3FFF;

        sz_size_t const window_matches = (sz_size_t)sz_u32_popcount(start_bits);
        sz_size_t const emit_count = sz_min_of_two(window_matches, matches_capacity - count);
        if (emit_count)
            sz_utf8_iterate_peel_powervsx_(start_bits, two_byte_starts, three_byte_starts, emit_count, position,
                                           match_offsets + count, match_lengths + count);
        count += emit_count;
        if (count == matches_capacity) { // output buffer full: resume past the last emitted match.
            position = match_offsets[count - 1] + match_lengths[count - 1];
            break;
        }
        position += 14;
    }

    count += sz_utf8_whitespaces_serial_((sz_cptr_t)(text_u8 + position), length - position, position,
                                         match_offsets + count, match_lengths + count, matches_capacity - count,
                                         bytes_consumed);
    return count;
}

#pragma endregion Multistep newline / whitespace iteration

/**
 *  @brief UAX-29 word boundary detection using IBM Power VSX (forward & reverse). Stateful sub-rules stay in
 *         the serial reference; all-ASCII windows resolve their trusted lanes in-vector and emit the proven
 *         boundaries, deferring every uncertain position to `_serial` so the output stays byte-exact.
 */

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#endif // SZ_USE_POWERVSX

#ifdef __cplusplus
}
#endif

#endif // STRINGZILLA_UTF8_TOKENS_POWERVSX_H_
