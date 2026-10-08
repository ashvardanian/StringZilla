/**
 *  @file include/stringzilla/utf8_tokens/powervsx.h
 *  @author Ash Vardanian
 *  @date June 7, 2026
 *  @brief POWER VSX backend for UTF-8 newline and whitespace delimiter scanning.
 */
#ifndef STRINGZILLA_UTF8_TOKENS_POWERVSX_H_
#define STRINGZILLA_UTF8_TOKENS_POWERVSX_H_

#include "stringzilla/types.h"
#include "stringzilla/utf8_tokens/serial.h"
#include "stringzilla/utf8_runes/powervsx.h"

#ifdef __cplusplus
extern "C" {
#endif

#if STRINGZILLA_ARCH_PPC64_
#if STRINGZILLA_TARGET_POWERVSX
#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("power9-vector"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("power9-vector")
#endif

/**
 *  @brief Peels the window's first @p emit_count matches by a branchless, ctz-free SIMD left-pack.
 *
 *  Each ascending 2-lane sub-block gathers its set lanes' `(position + lane, length)` @c u64 pairs
 *  with one @c vec_perm and full-stores to an 18-wide scratch, absorbing the last sub-block's
 *  2-lane spill since VSX has no masked store; the low @p emit_count entries copy out in ascending
 *  lane order, byte-exact.
 */
STRINGZILLA_INLINE void sz_utf8_iterate_peel_powervsx_(                        //
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
        __vector unsigned long long const candidate_offsets_u64x2 = {(unsigned long long)(position + base_lane),
                                                                     (unsigned long long)(position + base_lane + 1)};
        __vector unsigned long long const candidate_lengths_u64x2 = {
            1u + (two_byte_sub & 1u) + 2u * (three_byte_sub & 1u),
            1u + ((two_byte_sub >> 1) & 1u) + 2u * ((three_byte_sub >> 1) & 1u)};

        __vector unsigned char const permutation_u8x16 = vec_xl(0, (unsigned char const *)compact2_lut[submask]);
        __vector unsigned long long const packed_offsets_u64x2 = (__vector unsigned long long)vec_perm(
            (__vector unsigned char)candidate_offsets_u64x2, (__vector unsigned char)candidate_offsets_u64x2,
            permutation_u8x16);
        __vector unsigned long long const packed_lengths_u64x2 = (__vector unsigned long long)vec_perm(
            (__vector unsigned char)candidate_lengths_u64x2, (__vector unsigned char)candidate_lengths_u64x2,
            permutation_u8x16);
        vec_xst(packed_offsets_u64x2, 0, (unsigned long long *)(scratch_offsets + filled));
        vec_xst(packed_lengths_u64x2, 0, (unsigned long long *)(scratch_lengths + filled));
        filled += popcount2_lut[submask];
    }

    for (sz_size_t emitted = 0; emitted < emit_count; ++emitted)
        match_offsets[emitted] = scratch_offsets[emitted], match_lengths[emitted] = scratch_lengths[emitted];
}

STRINGZILLA_INLINE sz_size_t sz_utf8_newlines_powervsx_( //
    sz_cptr_t text, sz_size_t length,                    //
    sz_size_t *match_offsets, sz_size_t *match_lengths,  //
    sz_size_t matches_capacity, sz_size_t *bytes_consumed) {

    sz_u8_t const *text_u8 = (sz_u8_t const *)text;
    sz_size_t count = 0, position = 0;

    __vector unsigned char const newline_u8x16 = vec_splats((unsigned char)'\n');
    __vector unsigned char const vertical_tab_u8x16 = vec_splats((unsigned char)'\v');
    __vector unsigned char const form_feed_u8x16 = vec_splats((unsigned char)'\f');
    __vector unsigned char const carriage_return_u8x16 = vec_splats((unsigned char)'\r');
    __vector unsigned char const lead_c2_u8x16 = vec_splats((unsigned char)0xC2);
    __vector unsigned char const x_85_u8x16 = vec_splats((unsigned char)0x85);
    __vector unsigned char const lead_e2_u8x16 = vec_splats((unsigned char)0xE2);
    __vector unsigned char const byte_80_u8x16 = vec_splats((unsigned char)0x80);
    __vector unsigned char const x_a8_u8x16 = vec_splats((unsigned char)0xA8);
    __vector unsigned char const x_a9_u8x16 = vec_splats((unsigned char)0xA9);

    while (position + 16 <= length && count < matches_capacity) {
        __vector unsigned char window_u8x16 = vec_xl(0, text_u8 + position);
        sz_u32_t newline_bits = sz_utf8_iterate_movemask_powervsx_(
            (__vector unsigned char)vec_cmpeq(window_u8x16, newline_u8x16));
        sz_u32_t carriage_return_bits = sz_utf8_iterate_movemask_powervsx_(
            (__vector unsigned char)vec_cmpeq(window_u8x16, carriage_return_u8x16));
        sz_u32_t one_byte_bits =
            newline_bits | carriage_return_bits |
            sz_utf8_iterate_movemask_powervsx_((__vector unsigned char)vec_cmpeq(window_u8x16, vertical_tab_u8x16)) |
            sz_utf8_iterate_movemask_powervsx_((__vector unsigned char)vec_cmpeq(window_u8x16, form_feed_u8x16));

        // 2-byte NEL (C2 85); 3-byte LS/PS (E2 80 A8/A9) - bit `i+1` is the next lane, so suffixes shift right.
        sz_u32_t lead_c2_bits = sz_utf8_iterate_movemask_powervsx_(
            (__vector unsigned char)vec_cmpeq(window_u8x16, lead_c2_u8x16));
        sz_u32_t x_85_bits = sz_utf8_iterate_movemask_powervsx_(
            (__vector unsigned char)vec_cmpeq(window_u8x16, x_85_u8x16));
        sz_u32_t nel_bits = lead_c2_bits & (x_85_bits >> 1);
        sz_u32_t lead_e2_bits = sz_utf8_iterate_movemask_powervsx_(
            (__vector unsigned char)vec_cmpeq(window_u8x16, lead_e2_u8x16));
        sz_u32_t byte_80_bits = sz_utf8_iterate_movemask_powervsx_(
            (__vector unsigned char)vec_cmpeq(window_u8x16, byte_80_u8x16));
        sz_u32_t lead_e280_bits = lead_e2_bits & (byte_80_bits >> 1);
        sz_u32_t x_a8_bits = sz_utf8_iterate_movemask_powervsx_(
            (__vector unsigned char)vec_cmpeq(window_u8x16, x_a8_u8x16));
        sz_u32_t x_a9_bits = sz_utf8_iterate_movemask_powervsx_(
            (__vector unsigned char)vec_cmpeq(window_u8x16, x_a9_u8x16));
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
    sz_assert_(sz_utf8_batch_consistent_(length, matches_capacity, count, bytes_consumed ? *bytes_consumed : length,
                                         match_offsets, match_lengths));
    return count;
}

STRINGZILLA_API sz_status_t sz_utf8_newlines_powervsx(                              //
    sz_cptr_t text, sz_size_t length,                                               //
    sz_size_t *match_offsets, sz_size_t *match_lengths, sz_size_t matches_capacity, //
    sz_size_t *matches_count, sz_size_t *bytes_consumed, sz_stream_t stream) {
    sz_assert_(stream == STRINGZILLA_NULL);
    *matches_count = sz_utf8_newlines_powervsx_(text, length, match_offsets, match_lengths, matches_capacity,
                                                bytes_consumed);
    return sz_success_k;
}

STRINGZILLA_INLINE sz_size_t sz_utf8_whitespaces_powervsx_( //
    sz_cptr_t text, sz_size_t length,                       //
    sz_size_t *match_offsets, sz_size_t *match_lengths,     //
    sz_size_t matches_capacity, sz_size_t *bytes_consumed) {

    sz_u8_t const *text_u8 = (sz_u8_t const *)text;
    sz_size_t count = 0, position = 0;

    __vector unsigned char const tab_u8x16 = vec_splats((unsigned char)'\t');
    __vector unsigned char const carriage_return_u8x16 = vec_splats((unsigned char)'\r');
    __vector unsigned char const x_20_u8x16 = vec_splats((unsigned char)' ');
    __vector unsigned char const lead_c2_u8x16 = vec_splats((unsigned char)0xC2);
    __vector unsigned char const x_85_u8x16 = vec_splats((unsigned char)0x85);
    __vector unsigned char const x_a0_u8x16 = vec_splats((unsigned char)0xA0);
    __vector unsigned char const x_e1_u8x16 = vec_splats((unsigned char)0xE1);
    __vector unsigned char const lead_e2_u8x16 = vec_splats((unsigned char)0xE2);
    __vector unsigned char const x_e3_u8x16 = vec_splats((unsigned char)0xE3);
    __vector unsigned char const x_9a_u8x16 = vec_splats((unsigned char)0x9A);
    __vector unsigned char const byte_80_u8x16 = vec_splats((unsigned char)0x80);
    __vector unsigned char const x_81_u8x16 = vec_splats((unsigned char)0x81);
    __vector unsigned char const x_8a_u8x16 = vec_splats((unsigned char)0x8A);
    __vector unsigned char const x_a8_u8x16 = vec_splats((unsigned char)0xA8);
    __vector unsigned char const x_a9_u8x16 = vec_splats((unsigned char)0xA9);
    __vector unsigned char const x_af_u8x16 = vec_splats((unsigned char)0xAF);
    __vector unsigned char const x_9f_u8x16 = vec_splats((unsigned char)0x9F);

    while (position + 16 <= length && count < matches_capacity) {
        __vector unsigned char window_u8x16 = vec_xl(0, text_u8 + position);
        // 1-byte: space, plus the contiguous range [\t, \r] == [9, 13].
        sz_u32_t one_byte_bits = sz_utf8_iterate_movemask_powervsx_(
                                     (__vector unsigned char)vec_cmpeq(window_u8x16, x_20_u8x16)) |
                                 sz_utf8_iterate_movemask_powervsx_(
                                     vec_and((__vector unsigned char)vec_cmpge(window_u8x16, tab_u8x16),
                                             (__vector unsigned char)vec_cmple(window_u8x16, carriage_return_u8x16)));

        // 2-byte: C2 85 (NEL), C2 A0 (NBSP).
        sz_u32_t lead_c2_bits = sz_utf8_iterate_movemask_powervsx_(
            (__vector unsigned char)vec_cmpeq(window_u8x16, lead_c2_u8x16));
        sz_u32_t x_85_bits = sz_utf8_iterate_movemask_powervsx_(
            (__vector unsigned char)vec_cmpeq(window_u8x16, x_85_u8x16));
        sz_u32_t x_a0_bits = sz_utf8_iterate_movemask_powervsx_(
            (__vector unsigned char)vec_cmpeq(window_u8x16, x_a0_u8x16));
        sz_u32_t two_byte_starts = lead_c2_bits & ((x_85_bits >> 1) | (x_a0_bits >> 1));

        // 3-byte: E1 9A 80 (ogham); E2 80 [80-8A]; E2 80 AF; E2 81 9F; E2 80 A8/A9; E3 80 80.
        sz_u32_t byte_80_bits = sz_utf8_iterate_movemask_powervsx_(
            (__vector unsigned char)vec_cmpeq(window_u8x16, byte_80_u8x16));
        sz_u32_t lead_e2_bits = sz_utf8_iterate_movemask_powervsx_(
            (__vector unsigned char)vec_cmpeq(window_u8x16, lead_e2_u8x16));
        sz_u32_t lead_e280_bits = lead_e2_bits & (byte_80_bits >> 1);
        sz_u32_t x_e1_bits = sz_utf8_iterate_movemask_powervsx_(
            (__vector unsigned char)vec_cmpeq(window_u8x16, x_e1_u8x16));
        sz_u32_t x_9a_bits = sz_utf8_iterate_movemask_powervsx_(
            (__vector unsigned char)vec_cmpeq(window_u8x16, x_9a_u8x16));
        sz_u32_t ogham_bits = x_e1_bits & (x_9a_bits >> 1) & (byte_80_bits >> 2);
        sz_u32_t x_80_ge_bits = sz_utf8_iterate_movemask_powervsx_(
            (__vector unsigned char)vec_cmpge(window_u8x16, byte_80_u8x16));
        sz_u32_t x_8a_le_bits = sz_utf8_iterate_movemask_powervsx_(
            (__vector unsigned char)vec_cmple(window_u8x16, x_8a_u8x16));
        sz_u32_t range_e280_bits = lead_e280_bits & (x_80_ge_bits >> 2) & (x_8a_le_bits >> 2);
        sz_u32_t x_af_bits = sz_utf8_iterate_movemask_powervsx_(
            (__vector unsigned char)vec_cmpeq(window_u8x16, x_af_u8x16));
        sz_u32_t nnbsp_bits = lead_e280_bits & (x_af_bits >> 2);
        sz_u32_t x_81_bits = sz_utf8_iterate_movemask_powervsx_(
            (__vector unsigned char)vec_cmpeq(window_u8x16, x_81_u8x16));
        sz_u32_t x_9f_bits = sz_utf8_iterate_movemask_powervsx_(
            (__vector unsigned char)vec_cmpeq(window_u8x16, x_9f_u8x16));
        sz_u32_t mmsp_bits = lead_e2_bits & (x_81_bits >> 1) & (x_9f_bits >> 2);
        sz_u32_t x_a8_bits = sz_utf8_iterate_movemask_powervsx_(
            (__vector unsigned char)vec_cmpeq(window_u8x16, x_a8_u8x16));
        sz_u32_t x_a9_bits = sz_utf8_iterate_movemask_powervsx_(
            (__vector unsigned char)vec_cmpeq(window_u8x16, x_a9_u8x16));
        sz_u32_t line_bits = lead_e280_bits & (x_a8_bits >> 2);
        sz_u32_t para_bits = lead_e280_bits & (x_a9_bits >> 2);
        sz_u32_t x_e3_bits = sz_utf8_iterate_movemask_powervsx_(
            (__vector unsigned char)vec_cmpeq(window_u8x16, x_e3_u8x16));
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
    sz_assert_(sz_utf8_batch_consistent_(length, matches_capacity, count, bytes_consumed ? *bytes_consumed : length,
                                         match_offsets, match_lengths));
    return count;
}

STRINGZILLA_API sz_status_t sz_utf8_whitespaces_powervsx(                           //
    sz_cptr_t text, sz_size_t length,                                               //
    sz_size_t *match_offsets, sz_size_t *match_lengths, sz_size_t matches_capacity, //
    sz_size_t *matches_count, sz_size_t *bytes_consumed, sz_stream_t stream) {
    sz_assert_(stream == STRINGZILLA_NULL);
    *matches_count = sz_utf8_whitespaces_powervsx_(text, length, match_offsets, match_lengths, matches_capacity,
                                                   bytes_consumed);
    return sz_success_k;
}

#pragma endregion Multistep newline and whitespace iteration

#if !STRINGZILLA_ARCH_BIG_ENDIAN_

#pragma region Membership

/** Largest byte across the 16 lanes, which is zero iff every lane is zero. */
STRINGZILLA_INLINE sz_u8_t sz_utf8_max_reduce_powervsx_(__vector unsigned char value_u8x16) {
    value_u8x16 = vec_max(value_u8x16, vec_sld(value_u8x16, value_u8x16, 8));
    value_u8x16 = vec_max(value_u8x16, vec_sld(value_u8x16, value_u8x16, 4));
    value_u8x16 = vec_max(value_u8x16, vec_sld(value_u8x16, value_u8x16, 2));
    value_u8x16 = vec_max(value_u8x16, vec_sld(value_u8x16, value_u8x16, 1));
    return vec_extract(value_u8x16, 0);
}

/** Per-lane single-bit test `(bitmap_byte >> (low & 7)) & 1`, returned as 0x00/0xFF lanes. */
STRINGZILLA_INLINE __vector unsigned char sz_delimiter_test_bit_powervsx_(__vector unsigned char bitmap_byte_u8x16,
                                                                          __vector unsigned char low_u8x16) {
    static sz_u8_t const bit_for_low3[16] = {1, 2, 4, 8, 16, 32, 64, 128, 0, 0, 0, 0, 0, 0, 0, 0};
    __vector unsigned char const bit_table_u8x16 = vec_xl(0, bit_for_low3);
    __vector unsigned char const bit_mask_u8x16 = vec_perm(bit_table_u8x16, bit_table_u8x16,
                                                           vec_and(low_u8x16, vec_splats((unsigned char)0x07)));
    __vector unsigned char const is_clear_u8x16 = (__vector unsigned char)vec_cmpeq(
        vec_and(bitmap_byte_u8x16, bit_mask_u8x16), vec_splats((unsigned char)0));
    return vec_nor(is_clear_u8x16, is_clear_u8x16);
}

/** Looks up `row[index]` per lane in a 32-byte bitmap row with one @c vec_perm. */
STRINGZILLA_INLINE __vector unsigned char sz_delimiter_row_lookup_powervsx_(sz_u8_t const *row,
                                                                            __vector unsigned char index_u8x16) {
    return vec_perm(vec_xl(0, row), vec_xl(0, row + 16), index_u8x16);
}

/**
 *  @brief BMP (codepoint < 0x10000) delimiter membership for one quarter, as 0x00/0xFF lanes.
 *
 *  The POWER twin of @c sz_delimiter_bmp_membership_neon_. ASCII lanes read row 0, blocks without
 *  delimiters drop out in a pre-filter, and the survivors resolve one distinct high byte at a time.
 */
STRINGZILLA_INLINE __vector unsigned char sz_delimiter_bmp_membership_powervsx_(__vector unsigned char window_u8x16,
                                                                                __vector unsigned char high_in_u8x16,
                                                                                __vector unsigned char low_in_u8x16) {
    __vector unsigned char const is_ascii_u8x16 = (__vector unsigned char)vec_cmplt(window_u8x16,
                                                                                    vec_splats((unsigned char)0x80));
    __vector unsigned char const high_u8x16 = vec_andc(high_in_u8x16, is_ascii_u8x16);
    __vector unsigned char const low_u8x16 = vec_sel(low_in_u8x16, window_u8x16, is_ascii_u8x16);
    __vector unsigned char const high_is_zero_u8x16 = (__vector unsigned char)vec_cmpeq(high_u8x16,
                                                                                        vec_splats((unsigned char)0));

    __vector unsigned char const row0_byte_u8x16 = sz_delimiter_row_lookup_powervsx_(
        sz_utf8_delimiter_bmp_bitmaps_, sz_utf8_srl8_powervsx_(low_u8x16, 3, 0x1F));
    __vector unsigned char result_u8x16 = vec_and(sz_delimiter_test_bit_powervsx_(row0_byte_u8x16, low_u8x16),
                                                  high_is_zero_u8x16);

    __vector unsigned char const suspicious_byte_u8x16 = sz_delimiter_row_lookup_powervsx_(
        sz_utf8_delimiter_bmp_suspicious_highs_, sz_utf8_srl8_powervsx_(high_u8x16, 3, 0x1F));
    __vector unsigned char const is_continuation_u8x16 = (__vector unsigned char)vec_cmpeq(
        vec_and(window_u8x16, vec_splats((unsigned char)0xC0)), vec_splats((unsigned char)0x80));
    __vector unsigned char unresolved_u8x16 = vec_andc(
        vec_andc(sz_delimiter_test_bit_powervsx_(suspicious_byte_u8x16, high_u8x16), high_is_zero_u8x16),
        is_continuation_u8x16);
    for (;;) {
        sz_u8_t const shared_high = sz_utf8_max_reduce_powervsx_(vec_and(high_u8x16, unresolved_u8x16));
        if (!shared_high) break;
        sz_u8_t const *row = sz_utf8_delimiter_bmp_bitmaps_ + (sz_size_t)sz_utf8_delimiter_bmp_block_[shared_high] * 32;
        __vector unsigned char const same_u8x16 = vec_and(
            unresolved_u8x16, (__vector unsigned char)vec_cmpeq(high_u8x16, vec_splats(shared_high)));
        __vector unsigned char const row_byte_u8x16 = sz_delimiter_row_lookup_powervsx_(
            row, sz_utf8_srl8_powervsx_(low_u8x16, 3, 0x1F));
        result_u8x16 = vec_or(result_u8x16,
                              vec_and(sz_delimiter_test_bit_powervsx_(row_byte_u8x16, low_u8x16), same_u8x16));
        unresolved_u8x16 = vec_andc(unresolved_u8x16, same_u8x16);
    }
    return result_u8x16;
}

/**
 *  @brief Astral (codepoint >= 0x10000) delimiter membership for one quarter, as 0x00/0xFF lanes.
 *
 *  The POWER twin of @c sz_delimiter_astral_membership_neon_. The plane and sub byte of each
 *  lane select a 32-byte bitmap row, resolved one distinct pair at a time.
 */
STRINGZILLA_INLINE __vector unsigned char sz_delimiter_astral_membership_powervsx_(__vector unsigned char window_u8x16,
                                                                                   __vector unsigned char next1_u8x16,
                                                                                   __vector unsigned char next2_u8x16,
                                                                                   __vector unsigned char next3_u8x16) {
    __vector unsigned char const b0_u8x16 = vec_and(window_u8x16, vec_splats((unsigned char)0x07));
    __vector unsigned char const b1_u8x16 = vec_and(next1_u8x16, vec_splats((unsigned char)0x3F));
    __vector unsigned char const b2_u8x16 = vec_and(next2_u8x16, vec_splats((unsigned char)0x3F));
    __vector unsigned char const b3_u8x16 = vec_and(next3_u8x16, vec_splats((unsigned char)0x3F));

    __vector unsigned char const codepoint_high_u8x16 = vec_or(vec_sl(b0_u8x16, vec_splats((unsigned char)2)),
                                                               vec_sr(b1_u8x16, vec_splats((unsigned char)4)));
    __vector unsigned char const sub_u8x16 = vec_or(vec_sl(b1_u8x16, vec_splats((unsigned char)4)),
                                                    vec_sr(b2_u8x16, vec_splats((unsigned char)2)));
    __vector unsigned char const low8_u8x16 = vec_or(vec_sl(b2_u8x16, vec_splats((unsigned char)6)), b3_u8x16);

    __vector unsigned char result_u8x16 = vec_splats((unsigned char)0);
    __vector unsigned char remaining_u8x16 = vec_and(
        (__vector unsigned char)vec_cmpge(window_u8x16, vec_splats((unsigned char)0xF0)),
        (__vector unsigned char)vec_cmple(codepoint_high_u8x16, vec_splats((unsigned char)0x10)));
    for (;;) {
        sz_u8_t const shared_plane = sz_utf8_max_reduce_powervsx_(vec_and(codepoint_high_u8x16, remaining_u8x16));
        if (!shared_plane) break;
        __vector unsigned char const plane_same_u8x16 = vec_and(
            remaining_u8x16, (__vector unsigned char)vec_cmpeq(codepoint_high_u8x16, vec_splats(shared_plane)));
        sz_u8_t const shared_sub = sz_utf8_max_reduce_powervsx_(vec_and(sub_u8x16, plane_same_u8x16));
        sz_u8_t const *row =
            sz_utf8_delimiter_astral_bitmaps_ +
            (sz_size_t)sz_utf8_delimiter_astral_l2_[(sz_size_t)sz_utf8_delimiter_astral_l1_[shared_plane - 1] * 256 +
                                                    shared_sub] *
                32;
        __vector unsigned char const same_u8x16 = vec_and(
            plane_same_u8x16, (__vector unsigned char)vec_cmpeq(sub_u8x16, vec_splats(shared_sub)));
        __vector unsigned char const row_byte_u8x16 = sz_delimiter_row_lookup_powervsx_(
            row, sz_utf8_srl8_powervsx_(low8_u8x16, 3, 0x1F));
        result_u8x16 = vec_or(result_u8x16,
                              vec_and(sz_delimiter_test_bit_powervsx_(row_byte_u8x16, low8_u8x16), same_u8x16));
        remaining_u8x16 = vec_andc(remaining_u8x16, same_u8x16);
    }
    return result_u8x16;
}

/** Per-lane UTF-8 validity of codepoint starts, as @ref sz_rune_decode judges it. */
STRINGZILLA_INLINE sz_u64_t sz_delimiter_valid_starts_powervsx_(sz_utf8_rune_window_powervsx_t const *decoded,
                                                                __vector unsigned char const *next1_u8x16,
                                                                __vector unsigned char const *next2_u8x16,
                                                                __vector unsigned char const *next3_u8x16) {
    __vector unsigned char const continuation_mask_u8x16 = vec_splats((unsigned char)0xC0);
    __vector unsigned char const continuation_pattern_u8x16 = vec_splats((unsigned char)0x80);
    __vector unsigned char valid_bool_u8x16[4];
    for (int quarter = 0; quarter < 4; ++quarter) {
        __vector unsigned char const here_u8x16 = decoded->window_u8x16s[quarter];
        __vector unsigned char const n1_u8x16 = next1_u8x16[quarter];
        __vector unsigned char const c1_ok_u8x16 = (__vector unsigned char)vec_cmpeq(
            vec_and(n1_u8x16, continuation_mask_u8x16), continuation_pattern_u8x16);
        __vector unsigned char const c2_ok_u8x16 = (__vector unsigned char)vec_cmpeq(
            vec_and(next2_u8x16[quarter], continuation_mask_u8x16), continuation_pattern_u8x16);
        __vector unsigned char const c3_ok_u8x16 = (__vector unsigned char)vec_cmpeq(
            vec_and(next3_u8x16[quarter], continuation_mask_u8x16), continuation_pattern_u8x16);
        __vector unsigned char const ascii_u8x16 = (__vector unsigned char)vec_cmplt(here_u8x16,
                                                                                     vec_splats((unsigned char)0x80));

        __vector unsigned char const is_two_u8x16 = vec_and(
            (__vector unsigned char)vec_cmpge(here_u8x16, vec_splats((unsigned char)0xC0)),
            (__vector unsigned char)vec_cmplt(here_u8x16, vec_splats((unsigned char)0xE0)));
        __vector unsigned char const is_three_u8x16 = vec_and(
            (__vector unsigned char)vec_cmpge(here_u8x16, vec_splats((unsigned char)0xE0)),
            (__vector unsigned char)vec_cmplt(here_u8x16, vec_splats((unsigned char)0xF0)));
        __vector unsigned char const is_four_u8x16 = (__vector unsigned char)vec_cmpge(here_u8x16,
                                                                                       vec_splats((unsigned char)0xF0));

        __vector unsigned char const two_ok_u8x16 = vec_and(
            c1_ok_u8x16, (__vector unsigned char)vec_cmpge(here_u8x16, vec_splats((unsigned char)0xC2)));

        __vector unsigned char const lead_e0_u8x16 = (__vector unsigned char)vec_cmpeq(here_u8x16,
                                                                                       vec_splats((unsigned char)0xE0));
        __vector unsigned char const lead_ed_u8x16 = (__vector unsigned char)vec_cmpeq(here_u8x16,
                                                                                       vec_splats((unsigned char)0xED));
        __vector unsigned char const n1_lt_a0_u8x16 = (__vector unsigned char)vec_cmplt(
            n1_u8x16, vec_splats((unsigned char)0xA0));
        __vector unsigned char const bad_three_u8x16 = vec_or(vec_and(lead_e0_u8x16, n1_lt_a0_u8x16),
                                                              vec_andc(lead_ed_u8x16, n1_lt_a0_u8x16));
        __vector unsigned char const three_ok_u8x16 = vec_andc(vec_and(c1_ok_u8x16, c2_ok_u8x16), bad_three_u8x16);

        __vector unsigned char const lead_f0_u8x16 = (__vector unsigned char)vec_cmpeq(here_u8x16,
                                                                                       vec_splats((unsigned char)0xF0));
        __vector unsigned char const lead_f4_u8x16 = (__vector unsigned char)vec_cmpeq(here_u8x16,
                                                                                       vec_splats((unsigned char)0xF4));
        __vector unsigned char const n1_lt_90_u8x16 = (__vector unsigned char)vec_cmplt(
            n1_u8x16, vec_splats((unsigned char)0x90));
        __vector unsigned char const above_f4_u8x16 = (__vector unsigned char)vec_cmpgt(
            here_u8x16, vec_splats((unsigned char)0xF4));
        __vector unsigned char const bad_four_u8x16 = vec_or(
            above_f4_u8x16, vec_or(vec_and(lead_f0_u8x16, n1_lt_90_u8x16), vec_andc(lead_f4_u8x16, n1_lt_90_u8x16)));
        __vector unsigned char const four_ok_u8x16 = vec_andc(vec_and(vec_and(c1_ok_u8x16, c2_ok_u8x16), c3_ok_u8x16),
                                                              bad_four_u8x16);

        valid_bool_u8x16[quarter] = vec_or(
            vec_or(ascii_u8x16, vec_and(is_two_u8x16, two_ok_u8x16)),
            vec_or(vec_and(is_three_u8x16, three_ok_u8x16), vec_and(is_four_u8x16, four_ok_u8x16)));
    }
    sz_u64_t const valid = sz_utf8_mask_combine_powervsx_(valid_bool_u8x16);

    // A lead whose declared span runs past `loaded` is truncated and never valid.
    // The serial path re-syncs one byte at a time.
    sz_size_t const loaded = decoded->loaded;
    sz_u64_t const truncated = (decoded->two_byte_starts & ~sz_u64_mask_until_serial_(loaded >= 1 ? loaded - 1 : 0)) |
                               (decoded->three_byte_starts & ~sz_u64_mask_until_serial_(loaded >= 2 ? loaded - 2 : 0)) |
                               (decoded->four_byte_starts & ~sz_u64_mask_until_serial_(loaded >= 3 ? loaded - 3 : 0));
    return valid & sz_u64_mask_until_serial_(loaded) & ~truncated;
}

#pragma endregion Membership

#pragma region Forward driver

STRINGZILLA_INLINE sz_size_t sz_utf8_delimiters_powervsx_( //
    sz_cptr_t text, sz_size_t length,                      //
    sz_size_t *match_offsets, sz_size_t *match_lengths,    //
    sz_size_t matches_capacity, sz_size_t *bytes_consumed) {
    sz_u8_t const *const text_u8 = (sz_u8_t const *)text;

    sz_size_t base = 0, count = 0;
    while (base < length && count < matches_capacity) {
        sz_utf8_rune_window_powervsx_t const decoded = sz_utf8_rune_decode_window_powervsx_(text_u8 + base,
                                                                                            length - base);
        sz_size_t const loaded = decoded.loaded;
        sz_u64_t const loaded_mask = sz_u64_mask_until_serial_(loaded);

        sz_size_t byte_span = loaded;
        sz_u64_t hits;

        // All-ASCII window: every loaded lane is a valid one-byte start,
        // and membership is one lookup over the first half of bitmap row 0.
        int const all_ascii = decoded.codepoint_starts == loaded_mask &&
                              !(decoded.two_byte_starts | decoded.three_byte_starts | decoded.four_byte_starts);
        if (all_ascii) {
            __vector unsigned char const row0_u8x16 = vec_xl(0, sz_utf8_delimiter_bmp_bitmaps_);
            __vector unsigned char member_bool_u8x16[4];
            for (int quarter = 0; quarter < 4; ++quarter) {
                // @c vec_perm wraps its index modulo 32, so stray non-ASCII bytes are masked out.
                __vector unsigned char const is_ascii_u8x16 = (__vector unsigned char)vec_cmplt(
                    decoded.window_u8x16s[quarter], vec_splats((unsigned char)0x80));
                member_bool_u8x16[quarter] = vec_and(
                    sz_delimiter_test_bit_powervsx_(
                        vec_perm(row0_u8x16, row0_u8x16,
                                 sz_utf8_srl8_powervsx_(decoded.window_u8x16s[quarter], 3, 0x1F)),
                        decoded.window_u8x16s[quarter]),
                    is_ascii_u8x16);
            }
            hits = sz_utf8_mask_combine_powervsx_(member_bool_u8x16) & loaded_mask;
        }
        else {
            __vector unsigned char next1_u8x16[4], next2_u8x16[4], next3_u8x16[4];
            sz_utf8_forward_neighbours_powervsx_(decoded.window_u8x16s, next1_u8x16, next2_u8x16, next3_u8x16);

            // A multi-byte lead near the 64-byte edge whose span runs past `loaded`
            // would decode against a wrapped neighbour; defer it to the next window.
            if (loaded >= 64)
                byte_span = sz_utf8_delimiter_complete_span_(decoded.two_byte_starts, decoded.three_byte_starts,
                                                             decoded.four_byte_starts, loaded);
            sz_u64_t const span_mask = sz_u64_mask_until_serial_(byte_span);

            sz_u64_t const valid_starts = sz_delimiter_valid_starts_powervsx_(&decoded, next1_u8x16, next2_u8x16,
                                                                              next3_u8x16) &
                                          decoded.codepoint_starts & span_mask;

            sz_u64_t const four_byte = decoded.four_byte_starts & span_mask;
            sz_u64_t member = 0;
            for (int quarter = 0; quarter < 4; ++quarter) {
                __vector unsigned char const bmp_u8x16 = sz_delimiter_bmp_membership_powervsx_(
                    decoded.window_u8x16s[quarter], decoded.high_byte_u8x16s[quarter],
                    decoded.low_byte_u8x16s[quarter]);
                member |= (sz_u64_t)sz_utf8_movemask16_powervsx_(bmp_u8x16) << (16 * quarter);
            }
            if (four_byte) {
                sz_u64_t astral_member = 0;
                for (int quarter = 0; quarter < 4; ++quarter) {
                    __vector unsigned char const astral_u8x16 = sz_delimiter_astral_membership_powervsx_(
                        decoded.window_u8x16s[quarter], next1_u8x16[quarter], next2_u8x16[quarter],
                        next3_u8x16[quarter]);
                    astral_member |= (sz_u64_t)sz_utf8_movemask16_powervsx_(astral_u8x16) << (16 * quarter);
                }
                member = (member & ~four_byte) | (astral_member & four_byte);
            }

            hits = member & valid_starts;
        }
        while (hits && count < matches_capacity) {
            sz_size_t const lane = (sz_size_t)(63 - sz_u64_clz(hits & (~hits + 1)));
            hits &= hits - 1;
            sz_size_t length_at_lane = 1;
            length_at_lane += (decoded.two_byte_starts >> lane) & 1;
            length_at_lane += ((decoded.three_byte_starts >> lane) & 1) * 2;
            length_at_lane += ((decoded.four_byte_starts >> lane) & 1) * 3;
            match_offsets[count] = base + lane, match_lengths[count] = length_at_lane, ++count;
        }
        // Output buffer full: resume past the last emitted match, never at the window edge.
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

STRINGZILLA_API sz_status_t sz_utf8_delimiters_powervsx(                            //
    sz_cptr_t text, sz_size_t length,                                               //
    sz_size_t *match_offsets, sz_size_t *match_lengths, sz_size_t matches_capacity, //
    sz_size_t *matches_count, sz_size_t *bytes_consumed, sz_stream_t stream) {
    sz_assert_(stream == STRINGZILLA_NULL);
    *matches_count = sz_utf8_delimiters_powervsx_(text, length, match_offsets, match_lengths, matches_capacity,
                                                  bytes_consumed);
    sz_assert_(sz_utf8_batch_consistent_(length, matches_capacity, *matches_count,
                                         bytes_consumed ? *bytes_consumed : length, match_offsets, match_lengths));
    return sz_success_k;
}

#else // STRINGZILLA_ARCH_BIG_ENDIAN_

STRINGZILLA_API sz_status_t sz_utf8_delimiters_powervsx(                            //
    sz_cptr_t text, sz_size_t length,                                               //
    sz_size_t *match_offsets, sz_size_t *match_lengths, sz_size_t matches_capacity, //
    sz_size_t *matches_count, sz_size_t *bytes_consumed, sz_stream_t stream) {
    return sz_utf8_delimiters_serial(text, length, match_offsets, match_lengths, matches_capacity, matches_count,
                                     bytes_consumed, stream);
}

#endif // !STRINGZILLA_ARCH_BIG_ENDIAN_

/*  UAX-29 word boundary detection using IBM Power VSX, both forward and in reverse. Stateful
 *  sub-rules stay in the serial reference; all-ASCII windows resolve their trusted lanes in-vector
 *  and emit the proven boundaries, deferring every uncertain position to @c _serial so that the
 *  output stays byte-exact. */

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#endif // STRINGZILLA_TARGET_POWERVSX
#endif // STRINGZILLA_ARCH_PPC64_

#ifdef __cplusplus
}
#endif

#endif // STRINGZILLA_UTF8_TOKENS_POWERVSX_H_
