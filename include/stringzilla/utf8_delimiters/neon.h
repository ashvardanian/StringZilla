/**
 *  @brief NEON backend for UTF-8 newline and whitespace delimiter scanning.
 *  @file include/stringzilla/utf8_delimiters/neon.h
 *  @author Ash Vardanian
 */
#ifndef STRINGZILLA_UTF8_DELIMITERS_NEON_H_
#define STRINGZILLA_UTF8_DELIMITERS_NEON_H_

#include "stringzilla/types.h"
#include "stringzilla/utf8_delimiters/serial.h"
#include "stringzilla/utf8_codepoints/neon.h"

#ifdef __cplusplus
extern "C" {
#endif

#if SZ_USE_NEON
#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("+simd"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("+simd")
#endif

/**
 *  @brief Left-packs the @p submask -selected lanes of one 4-lane sub-block to the @p out_offsets / @p out_lengths
 *         cursors in lane order via a `vqtbl2q_u8` table; returns how many lanes were written.
 */
SZ_INTERNAL sz_size_t sz_utf8_iterate_compact4_neon_(                       //
    uint8x16x2_t const offsets_u8x16x2, uint8x16x2_t const lengths_u8x16x2, //
    sz_u32_t const submask, sz_size_t *const out_offsets, sz_size_t *const out_lengths) {

    static sz_u8_t const compact_lut[16][32] = {
        {0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7},
        {0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7},
        {8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7},
        {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7},
        {16, 17, 18, 19, 20, 21, 22, 23, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7},
        {0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7},
        {8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7},
        {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 0, 1, 2, 3, 4, 5, 6, 7},
        {24, 25, 26, 27, 28, 29, 30, 31, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7},
        {0, 1, 2, 3, 4, 5, 6, 7, 24, 25, 26, 27, 28, 29, 30, 31, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7},
        {8, 9, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29, 30, 31, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7},
        {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29, 30, 31, 0, 1, 2, 3, 4, 5, 6, 7},
        {16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
         0,  1,  2,  3,  4,  5,  6,  7,  0,  1,  2,  3,  4,  5,  6,  7},
        {0,  1,  2,  3,  4,  5,  6,  7,  16, 17, 18, 19, 20, 21, 22, 23,
         24, 25, 26, 27, 28, 29, 30, 31, 0,  1,  2,  3,  4,  5,  6,  7},
        {8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
         24, 25, 26, 27, 28, 29, 30, 31, 0,  1,  2,  3,  4,  5,  6,  7},
        {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15,
         16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31},
    };
    static sz_u8_t const popcount_lut[16] = {0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4};

    uint8x16_t const indices_low_u8x16 = vld1q_u8(compact_lut[submask]);
    uint8x16_t const indices_high_u8x16 = vld1q_u8(compact_lut[submask] + 16);
    vst1q_u64((sz_u64_t *)(out_offsets), vreinterpretq_u64_u8(vqtbl2q_u8(offsets_u8x16x2, indices_low_u8x16)));
    vst1q_u64((sz_u64_t *)(out_offsets + 2), vreinterpretq_u64_u8(vqtbl2q_u8(offsets_u8x16x2, indices_high_u8x16)));
    vst1q_u64((sz_u64_t *)(out_lengths), vreinterpretq_u64_u8(vqtbl2q_u8(lengths_u8x16x2, indices_low_u8x16)));
    vst1q_u64((sz_u64_t *)(out_lengths + 2), vreinterpretq_u64_u8(vqtbl2q_u8(lengths_u8x16x2, indices_high_u8x16)));
    return popcount_lut[submask];
}

/**
 *  @brief  Peel the tile's first @p emit_count matches by SIMD left-pack over four 4-lane sub-blocks into a
 *          fixed-width stack scratch, then copy the surviving prefix to the caller (no `ctz`, no per-match branch).
 */
SZ_INTERNAL void sz_utf8_iterate_peel_neon_(         //
    sz_u64_t start_bits, uint8x16_t length_per_lane, //
    sz_size_t emit_count, sz_size_t position,        //
    sz_size_t *match_offsets, sz_size_t *match_lengths) {

    static sz_u64_t const lane_offset_low[2] = {0, 1};
    static sz_u64_t const lane_offset_high[2] = {2, 3};
    static sz_u8_t const sub_block_gather[16] = {0, 1, 2, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    uint64x2_t const lane_offset_low_u64x2 = vld1q_u64(lane_offset_low);
    uint64x2_t const lane_offset_high_u64x2 = vld1q_u64(lane_offset_high);
    uint8x16_t const sub_block_gather_u8x16 = vld1q_u8(sub_block_gather);

    sz_size_t scratch_offsets[20], scratch_lengths[20];
    sz_size_t filled = 0;
    for (sz_size_t base_lane = 0; base_lane < 16; base_lane += 4) {
        sz_u32_t const submask = (sz_u32_t)(((start_bits >> (base_lane * 4 + 3)) & 1) |
                                            (((start_bits >> (base_lane * 4 + 7)) & 1) << 1) |
                                            (((start_bits >> (base_lane * 4 + 11)) & 1) << 2) |
                                            (((start_bits >> (base_lane * 4 + 15)) & 1) << 3));
        if (!submask) continue;

        uint64x2_t const base_broadcast = vdupq_n_u64((sz_u64_t)position + base_lane);
        uint8x16_t const lane_lengths_u8x16 = vqtbl1q_u8(
            length_per_lane, vaddq_u8(sub_block_gather_u8x16, vdupq_n_u8((sz_u8_t)base_lane)));
        uint16x8_t const lengths_u16x8 = vmovl_u8(vget_low_u8(lane_lengths_u8x16));
        uint32x4_t const lengths_u32x4 = vmovl_u16(vget_low_u16(lengths_u16x8));

        uint8x16x2_t offsets_u8x16x2, lengths_u8x16x2;
        offsets_u8x16x2.val[0] = vreinterpretq_u8_u64(vaddq_u64(base_broadcast, lane_offset_low_u64x2));
        offsets_u8x16x2.val[1] = vreinterpretq_u8_u64(vaddq_u64(base_broadcast, lane_offset_high_u64x2));
        lengths_u8x16x2.val[0] = vreinterpretq_u8_u64(vmovl_u32(vget_low_u32(lengths_u32x4)));
        lengths_u8x16x2.val[1] = vreinterpretq_u8_u64(vmovl_u32(vget_high_u32(lengths_u32x4)));

        filled += sz_utf8_iterate_compact4_neon_(offsets_u8x16x2, lengths_u8x16x2, submask, scratch_offsets + filled,
                                                 scratch_lengths + filled);
    }

    for (sz_size_t emitted = 0; emitted < emit_count; ++emitted)
        match_offsets[emitted] = scratch_offsets[emitted], match_lengths[emitted] = scratch_lengths[emitted];
}

SZ_PUBLIC sz_size_t sz_utf8_find_newlines_neon(         //
    sz_cptr_t text, sz_size_t length,                   //
    sz_size_t *match_offsets, sz_size_t *match_lengths, //
    sz_size_t matches_capacity, sz_size_t *bytes_consumed) {

    sz_u8_t const *text_u8 = (sz_u8_t const *)text;
    sz_size_t count = 0, position = 0;

    uint8x16_t newline_u8x16 = vdupq_n_u8('\n');
    uint8x16_t vertical_tab_u8x16 = vdupq_n_u8('\v');
    uint8x16_t form_feed_u8x16 = vdupq_n_u8('\f');
    uint8x16_t carriage_return_u8x16 = vdupq_n_u8('\r');
    uint8x16_t lead_c2_u8x16 = vdupq_n_u8(0xC2);
    uint8x16_t x_85_u8x16 = vdupq_n_u8(0x85);
    uint8x16_t lead_e2_u8x16 = vdupq_n_u8(0xE2);
    uint8x16_t byte_80_u8x16 = vdupq_n_u8(0x80);
    uint8x16_t x_a8_u8x16 = vdupq_n_u8(0xA8);
    uint8x16_t x_a9_u8x16 = vdupq_n_u8(0xA9);

    // We trust delimiter starts only in lanes [0,13] and step by 14, so any 2-/3-byte delimiter is fully loaded.
    static sz_u8_t const trusted_lanes[16] = {0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
                                              0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x00, 0x00};
    uint8x16_t trusted_lanes_u8x16 = vld1q_u8(trusted_lanes);

    while (position + 16 <= length && count < matches_capacity) {
        uint8x16_t window = vld1q_u8(text_u8 + position);
        uint8x16_t window1 = vextq_u8(window, window, 1); // next lane
        uint8x16_t window2 = vextq_u8(window, window, 2); // lane after next

        // 1-byte matches: \n \v \f \r.
        uint8x16_t newline_cmp = vceqq_u8(window, newline_u8x16);
        uint8x16_t carriage_return_cmp = vceqq_u8(window, carriage_return_u8x16);
        uint8x16_t one_byte_cmp = vorrq_u8(vorrq_u8(newline_cmp, vceqq_u8(window, vertical_tab_u8x16)),
                                           vorrq_u8(vceqq_u8(window, form_feed_u8x16), carriage_return_cmp));

        // 2-byte: CRLF (\r\n, one match) & NEL (C2 85) - computed unconditionally.
        uint8x16_t crlf_cmp = vandq_u8(carriage_return_cmp, vceqq_u8(window1, newline_u8x16));
        uint8x16_t nel_cmp = vandq_u8(vceqq_u8(window, lead_c2_u8x16), vceqq_u8(window1, x_85_u8x16));
        uint8x16_t two_byte_cmp = vorrq_u8(crlf_cmp, nel_cmp);

        // 3-byte: LS (E2 80 A8) & PS (E2 80 A9).
        uint8x16_t lead_e280_cmp = vandq_u8(vceqq_u8(window, lead_e2_u8x16), vceqq_u8(window1, byte_80_u8x16));
        uint8x16_t three_byte_cmp = vandq_u8(lead_e280_cmp,
                                             vorrq_u8(vceqq_u8(window2, x_a8_u8x16), vceqq_u8(window2, x_a9_u8x16)));

        // CRLF's trailing LF must not also be emitted: an LF whose previous lane is a CR.
        uint8x16_t carriage_return_previous = vextq_u8(vdupq_n_u8(0), carriage_return_cmp, 15);
        uint8x16_t lf_of_crlf_cmp = vandq_u8(newline_cmp, carriage_return_previous);

        uint8x16_t starts_cmp = vandq_u8(
            vbicq_u8(vorrq_u8(vorrq_u8(one_byte_cmp, nel_cmp), three_byte_cmp), lf_of_crlf_cmp), trusted_lanes_u8x16);

        sz_u64_t start_bits = sz_utf8_vreinterpretq_u8_u4_neon_(starts_cmp);

        // Per-lane byte length: 1, plus 1 on a 2-byte start, plus 2 on a 3-byte start (the masks are disjoint).
        uint8x16_t length_per_lane = vaddq_u8(
            vdupq_n_u8(1), vaddq_u8(vshrq_n_u8(two_byte_cmp, 7), vshlq_n_u8(vshrq_n_u8(three_byte_cmp, 7), 1)));

        // Suppress a leading LF already consumed by a CRLF that straddled the previous tile edge.
        if (position != 0 && text_u8[position - 1] == '\r')
            start_bits &= ~(sz_utf8_vreinterpretq_u8_u4_neon_(newline_cmp) & 0xFull);

        // One set bit per match (nibble stride 4), so popcount is the true count of trusted set lanes.
        sz_size_t const window_matches = (sz_size_t)sz_u64_popcount(start_bits);
        sz_size_t const emit_count = sz_min_of_two(window_matches, matches_capacity - count);
        if (emit_count)
            sz_utf8_iterate_peel_neon_(start_bits, length_per_lane, emit_count, position, match_offsets + count,
                                       match_lengths + count);
        count += emit_count;
        if (count == matches_capacity) { // output buffer full: resume past the last emitted match
            position = match_offsets[count - 1] + match_lengths[count - 1];
            break;
        }
        position += 14;
    }

    // Skip a CRLF's trailing LF if it straddles into the serial tail (the CR was emitted as a 2-byte match).
    if (position != 0 && position < length && text_u8[position - 1] == '\r' && text_u8[position] == '\n') ++position;
    count += sz_utf8_find_newlines_serial_((sz_cptr_t)(text_u8 + position), length - position, position,
                                           match_offsets + count, match_lengths + count, matches_capacity - count,
                                           bytes_consumed);
    return count;
}

SZ_PUBLIC sz_size_t sz_utf8_find_whitespaces_neon(      //
    sz_cptr_t text, sz_size_t length,                   //
    sz_size_t *match_offsets, sz_size_t *match_lengths, //
    sz_size_t matches_capacity, sz_size_t *bytes_consumed) {

    sz_u8_t const *text_u8 = (sz_u8_t const *)text;
    sz_size_t count = 0, position = 0;

    uint8x16_t tab_u8x16 = vdupq_n_u8('\t');
    uint8x16_t carriage_return_u8x16 = vdupq_n_u8('\r');
    uint8x16_t x_20_u8x16 = vdupq_n_u8(' ');
    uint8x16_t lead_c2_u8x16 = vdupq_n_u8(0xC2);
    uint8x16_t x_85_u8x16 = vdupq_n_u8(0x85);
    uint8x16_t x_a0_u8x16 = vdupq_n_u8(0xA0);
    uint8x16_t x_e1_u8x16 = vdupq_n_u8(0xE1);
    uint8x16_t lead_e2_u8x16 = vdupq_n_u8(0xE2);
    uint8x16_t x_e3_u8x16 = vdupq_n_u8(0xE3);
    uint8x16_t x_9a_u8x16 = vdupq_n_u8(0x9A);
    uint8x16_t byte_80_u8x16 = vdupq_n_u8(0x80);
    uint8x16_t x_81_u8x16 = vdupq_n_u8(0x81);
    uint8x16_t x_8d_u8x16 = vdupq_n_u8(0x8D);
    uint8x16_t x_a8_u8x16 = vdupq_n_u8(0xA8);
    uint8x16_t x_a9_u8x16 = vdupq_n_u8(0xA9);
    uint8x16_t x_af_u8x16 = vdupq_n_u8(0xAF);
    uint8x16_t x_9f_u8x16 = vdupq_n_u8(0x9F);

    static sz_u8_t const trusted_lanes[16] = {0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
                                              0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x00, 0x00};
    uint8x16_t trusted_lanes_u8x16 = vld1q_u8(trusted_lanes);

    while (position + 16 <= length && count < matches_capacity) {
        uint8x16_t window = vld1q_u8(text_u8 + position);
        uint8x16_t window1 = vextq_u8(window, window, 1); // next lane
        uint8x16_t window2 = vextq_u8(window, window, 2); // lane after next

        // 1-byte: space, plus the contiguous range [\t, \r] == [9, 13].
        uint8x16_t one_byte_cmp = vorrq_u8(
            vceqq_u8(window, x_20_u8x16),
            vandq_u8(vcgeq_u8(window, tab_u8x16), vcleq_u8(window, carriage_return_u8x16)));

        // 2-byte: C2 85 (NEL), C2 A0 (NBSP).
        uint8x16_t lead_c2_cmp = vceqq_u8(window, lead_c2_u8x16);
        uint8x16_t two_byte_cmp = vandq_u8(lead_c2_cmp,
                                           vorrq_u8(vceqq_u8(window1, x_85_u8x16), vceqq_u8(window1, x_a0_u8x16)));

        // 3-byte: E1 9A 80 (ogham); E2 80 [80-8D]; E2 80 AF; E2 81 9F; E2 80 A8/A9; E3 80 80.
        uint8x16_t window1_is_80 = vceqq_u8(window1, byte_80_u8x16);
        uint8x16_t lead_e280_cmp = vandq_u8(vceqq_u8(window, lead_e2_u8x16), window1_is_80);
        uint8x16_t ogham_cmp = vandq_u8(vceqq_u8(window, x_e1_u8x16),
                                        vandq_u8(vceqq_u8(window1, x_9a_u8x16), vceqq_u8(window2, byte_80_u8x16)));
        uint8x16_t range_e280_cmp = vandq_u8(lead_e280_cmp,
                                             vandq_u8(vcgeq_u8(window2, byte_80_u8x16), vcleq_u8(window2, x_8d_u8x16)));
        uint8x16_t nnbsp_cmp = vandq_u8(lead_e280_cmp, vceqq_u8(window2, x_af_u8x16));
        uint8x16_t mmsp_cmp = vandq_u8(vandq_u8(vceqq_u8(window, lead_e2_u8x16), vceqq_u8(window1, x_81_u8x16)),
                                       vceqq_u8(window2, x_9f_u8x16));
        uint8x16_t line_cmp = vandq_u8(lead_e280_cmp, vceqq_u8(window2, x_a8_u8x16));
        uint8x16_t paragraph_cmp = vandq_u8(lead_e280_cmp, vceqq_u8(window2, x_a9_u8x16));
        uint8x16_t ideographic_cmp = vandq_u8(vandq_u8(vceqq_u8(window, x_e3_u8x16), window1_is_80),
                                              vceqq_u8(window2, byte_80_u8x16));
        uint8x16_t three_byte_cmp = vorrq_u8(
            vorrq_u8(vorrq_u8(ogham_cmp, range_e280_cmp), vorrq_u8(nnbsp_cmp, mmsp_cmp)),
            vorrq_u8(vorrq_u8(line_cmp, paragraph_cmp), ideographic_cmp));

        uint8x16_t starts_cmp = vandq_u8(vorrq_u8(vorrq_u8(one_byte_cmp, two_byte_cmp), three_byte_cmp),
                                         trusted_lanes_u8x16);

        sz_u64_t start_bits = sz_utf8_vreinterpretq_u8_u4_neon_(starts_cmp);

        // Per-lane byte length: 1, plus 1 on a 2-byte start, plus 2 on a 3-byte start (the masks are disjoint).
        uint8x16_t length_per_lane = vaddq_u8(
            vdupq_n_u8(1), vaddq_u8(vshrq_n_u8(two_byte_cmp, 7), vshlq_n_u8(vshrq_n_u8(three_byte_cmp, 7), 1)));

        // One set bit per match (nibble stride 4), so popcount is the true count of trusted set lanes.
        sz_size_t const window_matches = (sz_size_t)sz_u64_popcount(start_bits);
        sz_size_t const emit_count = sz_min_of_two(window_matches, matches_capacity - count);
        if (emit_count)
            sz_utf8_iterate_peel_neon_(start_bits, length_per_lane, emit_count, position, match_offsets + count,
                                       match_lengths + count);
        count += emit_count;
        if (count == matches_capacity) { // output buffer full: resume past the last emitted match
            position = match_offsets[count - 1] + match_lengths[count - 1];
            break;
        }
        position += 14;
    }

    count += sz_utf8_find_whitespaces_serial_((sz_cptr_t)(text_u8 + position), length - position, position,
                                              match_offsets + count, match_lengths + count, matches_capacity - count,
                                              bytes_consumed);
    return count;
}

#pragma endregion // Multistep newline and whitespace iteration

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#endif // SZ_USE_NEON

#ifdef __cplusplus
}
#endif

#endif // STRINGZILLA_UTF8_DELIMITERS_NEON_H_
