/**
 *  @brief NEON backend for UTF-8 newline and whitespace delimiter scanning.
 *  @file include/stringzilla/utf8_tokens/neon.h
 *  @author Ash Vardanian
 */
#ifndef STRINGZILLA_UTF8_TOKENS_NEON_H_
#define STRINGZILLA_UTF8_TOKENS_NEON_H_

#include "stringzilla/types.h"
#include "stringzilla/utf8_tokens/serial.h"
#include "stringzilla/utf8_runes/neon.h"
#include "stringzilla/utf8_tokens/tables.h"

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

#pragma region Multistep newline and whitespace iteration

/**
 *  @brief Left-packs the @p submask -selected lanes of one 4-lane sub-block to the @p out_offsets / @p out_lengths
 *         cursors in lane order via a `vqtbl2q_u8` table; returns how many lanes were written.
 */
SZ_HELPER_INLINE sz_size_t sz_utf8_iterate_compact4_neon_(                  //
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
SZ_HELPER_AUTO void sz_utf8_iterate_peel_neon_(      //
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

SZ_API_COMPTIME sz_size_t sz_utf8_newlines_neon(        //
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
    count += sz_utf8_newlines_serial_((sz_cptr_t)(text_u8 + position), length - position, position,
                                      match_offsets + count, match_lengths + count, matches_capacity - count,
                                      bytes_consumed);
    return count;
}

SZ_API_COMPTIME sz_size_t sz_utf8_whitespaces_neon(     //
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
    uint8x16_t x_8a_u8x16 = vdupq_n_u8(0x8A);
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

        // 3-byte: E1 9A 80 (ogham); E2 80 [80-8A]; E2 80 AF; E2 81 9F; E2 80 A8/A9; E3 80 80.
        uint8x16_t window1_is_80 = vceqq_u8(window1, byte_80_u8x16);
        uint8x16_t lead_e280_cmp = vandq_u8(vceqq_u8(window, lead_e2_u8x16), window1_is_80);
        uint8x16_t ogham_cmp = vandq_u8(vceqq_u8(window, x_e1_u8x16),
                                        vandq_u8(vceqq_u8(window1, x_9a_u8x16), vceqq_u8(window2, byte_80_u8x16)));
        uint8x16_t range_e280_cmp = vandq_u8(lead_e280_cmp,
                                             vandq_u8(vcgeq_u8(window2, byte_80_u8x16), vcleq_u8(window2, x_8a_u8x16)));
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

    count += sz_utf8_whitespaces_serial_((sz_cptr_t)(text_u8 + position), length - position, position,
                                         match_offsets + count, match_lengths + count, matches_capacity - count,
                                         bytes_consumed);
    return count;
}

#pragma endregion Multistep newline and whitespace iteration

#pragma region Membership

/** @brief  Per-lane single-bit test `(bitmap_byte >> (low & 7)) & 1` for one quarter, returned as 0x00/0xFF lanes. */
SZ_HELPER_INLINE uint8x16_t sz_delimiter_test_bit_neon_(uint8x16_t bitmap_byte, uint8x16_t low) {
    static sz_u8_t const bit_for_low3[16] = {1, 2, 4, 8, 16, 32, 64, 128, 0, 0, 0, 0, 0, 0, 0, 0};
    uint8x16_t const bit_table = vld1q_u8(bit_for_low3);
    uint8x16_t const bit_mask = vqtbl1q_u8(bit_table, vandq_u8(low, vdupq_n_u8(0x07)));
    return vtstq_u8(bitmap_byte, bit_mask);
}

/**
 *  @brief  Read one bitmap byte per lane from the row-major bitmap rows: `bitmap_byte = bitmaps[block_id*32 +
 *          (low >> 3)]`. NEON has no gather and the rows span up to ~1.7 KB, so the fused 11-bit indices take a
 *          bounded scalar L1 walk - the same leaf pattern as @ref sz_utf8_rune_flat_lookup_neon_. @p block_id and
 *          @p low address one quarter; every block id indexes a real row, so the read is always in-bounds.
 */
SZ_HELPER_AUTO uint8x16_t sz_delimiter_bitmap_byte_neon_(sz_u8_t const *bitmaps, uint8x16_t block_id, uint8x16_t low) {
    uint8x16_t const column = vshrq_n_u8(low, 3); // (low >> 3) in [0, 32): the byte within the row
    uint16x8_t const bitmap_indices_low_u16x8 = vorrq_u16(vshll_n_u8(vget_low_u8(block_id), 5),
                                                          vmovl_u8(vget_low_u8(column)));
    uint16x8_t const bitmap_indices_high_u16x8 = vorrq_u16(vshll_n_u8(vget_high_u8(block_id), 5),
                                                           vmovl_u8(vget_high_u8(column)));
    sz_align_(16) sz_u16_t bitmap_indices[16];
    sz_align_(16) sz_u8_t bitmap_bytes[16];
    vst1q_u16(bitmap_indices, bitmap_indices_low_u16x8);
    vst1q_u16(bitmap_indices + 8, bitmap_indices_high_u16x8);
    for (int lane = 0; lane < 16; ++lane) bitmap_bytes[lane] = bitmaps[bitmap_indices[lane]];
    return vld1q_u8(bitmap_bytes);
}

/**
 *  @brief  BMP (codepoint < 0x10000) delimiter membership for one quarter, in-register; returns 0x00/0xFF lanes.
 *
 *  The decode window only reconstructs `high`/`low` for 2-/3-byte leads; ASCII lanes (top bit clear) carry their
 *  codepoint in the raw byte itself, so override them with (high=0, low=byte) before addressing the BMP tables.
 *  `high` (cp >> 8, in [0,256)) selects a 32-byte bitmap row id through the aligned 256-entry `bmp_block` table via
 *  `vqtbl4q_u8`; the bitmap byte at `row_id*32 + (low >> 3)` is read from the row-major rows; the bit `(low & 7)` is
 *  tested.
 */
SZ_HELPER_AUTO uint8x16_t sz_delimiter_bmp_membership_neon_(uint8x16_t window, uint8x16_t high_in, uint8x16_t low_in) {
    uint8x16_t const is_ascii = vcltq_u8(window, vdupq_n_u8(0x80));
    uint8x16_t const high = vbicq_u8(high_in, is_ascii);       // high = is_ascii ? 0 : high_in
    uint8x16_t const low = vbslq_u8(is_ascii, window, low_in); // low  = is_ascii ? byte : low_in
    uint8x16_t const block_id = sz_utf8_rune_lut256_neon_(sz_utf8_delimiter_bmp_block_, high);
    uint8x16_t const bitmap_byte = sz_delimiter_bitmap_byte_neon_(sz_utf8_delimiter_bmp_bitmaps_, block_id, low);
    return sz_delimiter_test_bit_neon_(bitmap_byte, low);
}

/**
 *  @brief  Astral (codepoint >= 0x10000) delimiter membership for one quarter, in-register; returns 0x00/0xFF lanes.
 *
 *  Reconstructs the byte-domain components of `offset = cp - 0x10000` directly per lane (no 32-bit-lane widening):
 *  with `cp = b0<<18 | b1<<12 | b2<<6 | b3` (b0 = lead & 7, b1..b3 = continuation & 0x3F) and `offset = cp - 0x10000`,
 *  the parts are `super = (cp >> 16) - 1` (no borrow: subtracting exactly 1<<16), `sub = (cp >> 8) & 0xFF`,
 *  `low8 = cp & 0xFF`. The `super` (0..15) selects an L1 group; `group*256 + sub` selects a bitmap row id (group < 2,
 *  so the two 256-entry halves of the L2 table are read and blended by the group bit); the bit `(low8 & 7)` is tested.
 */
SZ_HELPER_AUTO uint8x16_t sz_delimiter_astral_membership_neon_(uint8x16_t window, uint8x16_t next1, uint8x16_t next2,
                                                               uint8x16_t next3) {
    uint8x16_t const b0 = vandq_u8(window, vdupq_n_u8(0x07));
    uint8x16_t const b1 = vandq_u8(next1, vdupq_n_u8(0x3F));
    uint8x16_t const b2 = vandq_u8(next2, vdupq_n_u8(0x3F));
    uint8x16_t const b3 = vandq_u8(next3, vdupq_n_u8(0x3F));

    // cp >> 16 = (b0 << 2) | (b1 >> 4); super = (cp >> 16) - 1.
    uint8x16_t const cp_hi = vorrq_u8(vshlq_n_u8(b0, 2), vshrq_n_u8(b1, 4));
    uint8x16_t const super = vsubq_u8(cp_hi, vdupq_n_u8(1));
    // (cp >> 8) & 0xFF = (b1 << 4) | (b2 >> 2).
    uint8x16_t const sub = vorrq_u8(vshlq_n_u8(b1, 4), vshrq_n_u8(b2, 2));
    // cp & 0xFF = (b2 << 6) | b3.
    uint8x16_t const low8 = vorrq_u8(vshlq_n_u8(b2, 6), b3);

    // group = astral_l1[super]; super in [0, 16), so a single 16-entry cascade row addresses it.
    uint8x16_t const group = sz_utf8_rune_cascade_stage_neon_(sz_utf8_delimiter_astral_l1_, 1, vdupq_n_u8(0), super);

    // row_id = astral_l2[group*256 + sub]: read both 256-entry halves and blend by the group bit (group < 2).
    uint8x16_t const row_group0 = sz_utf8_rune_lut256_neon_(sz_utf8_delimiter_astral_l2_ + 0, sub);
    uint8x16_t const row_group1 = sz_utf8_rune_lut256_neon_(sz_utf8_delimiter_astral_l2_ + 256, sub);
    uint8x16_t const group_is_one = vceqq_u8(group, vdupq_n_u8(1));
    uint8x16_t const row_id = vbslq_u8(group_is_one, row_group1, row_group0);

    uint8x16_t const bitmap_byte = sz_delimiter_bitmap_byte_neon_(sz_utf8_delimiter_astral_bitmaps_, row_id, low8);
    return sz_delimiter_test_bit_neon_(bitmap_byte, low8);
}

/**
 *  @brief  Per-lane UTF-8 validity for codepoint-start lanes, mirroring `sz_rune_decode` exactly: a 2/3/4-byte lead is
 *          valid only when its continuation bytes are well-formed and it is not overlong, a surrogate, or beyond
 *          U+10FFFF. Returned as one 64-bit lane mask. Span-clamping (so a lead near the loaded edge whose
 *          continuation bytes wrap is rejected) is applied by the caller via `byte_span`; here the substrate masks are
 *          already loaded-clamped. An invalid lead is never reported (serial advances one byte and re-syncs).
 */
SZ_HELPER_AUTO sz_u64_t sz_delimiter_valid_starts_neon_(sz_utf8_rune_window_neon_t const *decoded,
                                                        uint8x16_t const *next1, uint8x16_t const *next2,
                                                        uint8x16_t const *next3) {
    uint8x16_t const continuation_mask = vdupq_n_u8(0xC0), continuation_pattern = vdupq_n_u8(0x80);
    uint8x16_t valid_bool[4];
    for (int quarter = 0; quarter < 4; ++quarter) {
        uint8x16_t const here = decoded->window[quarter];
        uint8x16_t const n1 = next1[quarter];
        uint8x16_t const c1_ok = vceqq_u8(vandq_u8(n1, continuation_mask), continuation_pattern);
        uint8x16_t const c2_ok = vceqq_u8(vandq_u8(next2[quarter], continuation_mask), continuation_pattern);
        uint8x16_t const c3_ok = vceqq_u8(vandq_u8(next3[quarter], continuation_mask), continuation_pattern);
        uint8x16_t const ascii = vcltq_u8(here, vdupq_n_u8(0x80));

        // Lead-byte length classes; the caller's `codepoint_starts` AND keeps only claimed start lanes,
        // so continuation and mid-sequence lanes are don't-cares here.
        uint8x16_t const is_two = vandq_u8(vcgeq_u8(here, vdupq_n_u8(0xC0)), vcltq_u8(here, vdupq_n_u8(0xE0)));
        uint8x16_t const is_three = vandq_u8(vcgeq_u8(here, vdupq_n_u8(0xE0)), vcltq_u8(here, vdupq_n_u8(0xF0)));
        uint8x16_t const is_four = vcgeq_u8(here, vdupq_n_u8(0xF0));

        // 2-byte: lead >= 0xC2 (reject C0/C1 overlong).
        uint8x16_t const two_ok = vandq_u8(c1_ok, vcgeq_u8(here, vdupq_n_u8(0xC2)));

        // 3-byte: not overlong (E0 with next1 < 0xA0); not surrogate (ED with next1 >= 0xA0).
        uint8x16_t const lead_e0 = vceqq_u8(here, vdupq_n_u8(0xE0));
        uint8x16_t const lead_ed = vceqq_u8(here, vdupq_n_u8(0xED));
        uint8x16_t const n1_lt_a0 = vcltq_u8(n1, vdupq_n_u8(0xA0));
        uint8x16_t const bad_three = vorrq_u8(vandq_u8(lead_e0, n1_lt_a0), vandq_u8(lead_ed, vmvnq_u8(n1_lt_a0)));
        uint8x16_t const three_ok = vbicq_u8(vandq_u8(c1_ok, c2_ok), bad_three);

        // 4-byte: lead <= 0xF4; not overlong (F0 with next1 < 0x90); not > U+10FFFF (F4 with next1 >= 0x90).
        uint8x16_t const lead_f0 = vceqq_u8(here, vdupq_n_u8(0xF0));
        uint8x16_t const lead_f4 = vceqq_u8(here, vdupq_n_u8(0xF4));
        uint8x16_t const n1_lt_90 = vcltq_u8(n1, vdupq_n_u8(0x90));
        uint8x16_t const bad_four = vorrq_u8(
            vmvnq_u8(vcleq_u8(here, vdupq_n_u8(0xF4))),
            vorrq_u8(vandq_u8(lead_f0, n1_lt_90), vandq_u8(lead_f4, vmvnq_u8(n1_lt_90))));
        uint8x16_t const four_ok = vbicq_u8(vandq_u8(vandq_u8(c1_ok, c2_ok), c3_ok), bad_four);

        valid_bool[quarter] = vorrq_u8(vorrq_u8(ascii, vandq_u8(is_two, two_ok)),
                                       vorrq_u8(vandq_u8(is_three, three_ok), vandq_u8(is_four, four_ok)));
    }
    sz_u64_t const valid = sz_utf8_mask_combine_neon_(valid_bool[0], valid_bool[1], valid_bool[2], valid_bool[3]);

    // A lead whose declared span runs past `loaded` is truncated and never valid (serial re-syncs one byte at a time).
    sz_size_t const loaded = decoded->loaded;
    sz_u64_t const truncated = (decoded->two_byte_starts & ~sz_u64_mask_until_serial_(loaded >= 1 ? loaded - 1 : 0)) |
                               (decoded->three_byte_starts & ~sz_u64_mask_until_serial_(loaded >= 2 ? loaded - 2 : 0)) |
                               (decoded->four_byte_starts & ~sz_u64_mask_until_serial_(loaded >= 3 ? loaded - 3 : 0));
    return valid & sz_u64_mask_until_serial_(loaded) & ~truncated;
}

#pragma endregion Membership

#pragma region Forward driver

/** @copydoc sz_utf8_delimiters */
SZ_API_COMPTIME sz_size_t sz_utf8_delimiters_neon(      //
    sz_cptr_t text, sz_size_t length,                   //
    sz_size_t *match_offsets, sz_size_t *match_lengths, //
    sz_size_t matches_capacity, sz_size_t *bytes_consumed) {
    sz_u8_t const *const text_u8 = (sz_u8_t const *)text;

    sz_size_t base = 0, count = 0;
    while (base < length && count < matches_capacity) {
        sz_utf8_rune_window_neon_t const decoded = sz_utf8_rune_decode_window_neon_(text_u8 + base, length - base);
        sz_size_t const loaded = decoded.loaded;
        sz_u64_t const loaded_mask = sz_u64_mask_until_serial_(loaded);

        sz_size_t byte_span = loaded;
        sz_u64_t hits;

        // All-ASCII window: every loaded lane is a valid one-byte start, and membership is one 16-byte TBL over
        // the first half of bitmap row 0 (`bmp_block_[0] == 0`, and ASCII columns stop at (0x7F >> 3) == 15).
        int const all_ascii = decoded.codepoint_starts == loaded_mask &&
                              !(decoded.two_byte_starts | decoded.three_byte_starts | decoded.four_byte_starts);
        if (all_ascii) {
            uint8x16_t const row0_u8x16 = vld1q_u8(sz_utf8_delimiter_bmp_bitmaps_);
            uint8x16_t member_bool[4];
            for (int quarter = 0; quarter < 4; ++quarter)
                member_bool[quarter] = sz_delimiter_test_bit_neon_(
                    vqtbl1q_u8(row0_u8x16, vshrq_n_u8(decoded.window[quarter], 3)), decoded.window[quarter]);
            hits = sz_utf8_mask_combine_neon_(member_bool[0], member_bool[1], member_bool[2], member_bool[3]) &
                   loaded_mask;
        }
        else {
            uint8x16_t next1[4], next2[4], next3[4];
            sz_utf8_forward_neighbours_neon_(decoded.window, next1, next2, next3);

            // Effective-window trim: a multi-byte lead near the 64-byte edge whose span runs past `loaded` would
            // decode against a wrapped neighbour; defer it to the next window.
            if (loaded >= 64)
                byte_span = sz_utf8_delimiter_complete_span_(decoded.two_byte_starts, decoded.three_byte_starts,
                                                             decoded.four_byte_starts, loaded);
            sz_u64_t const span_mask = sz_u64_mask_until_serial_(byte_span);

            sz_u64_t const valid_starts = sz_delimiter_valid_starts_neon_(&decoded, next1, next2, next3) &
                                          decoded.codepoint_starts & span_mask;

            // BMP membership for every lane; astral membership blended onto the four-byte lanes (gated on presence).
            sz_u64_t const four_byte = decoded.four_byte_starts & span_mask;
            sz_u64_t member = 0;
            for (int quarter = 0; quarter < 4; ++quarter) {
                uint8x16_t const bmp = sz_delimiter_bmp_membership_neon_(decoded.window[quarter], decoded.high[quarter],
                                                                         decoded.low[quarter]);
                member |= sz_utf8_movemask16_neon_(bmp) << (16 * quarter);
            }
            // Blend astral over the four-byte lanes; the per-lane BMP/astral decision stays exact on the bit masks.
            if (four_byte) {
                sz_u64_t astral_member = 0;
                for (int quarter = 0; quarter < 4; ++quarter) {
                    uint8x16_t const astral = sz_delimiter_astral_membership_neon_(
                        decoded.window[quarter], next1[quarter], next2[quarter], next3[quarter]);
                    astral_member |= sz_utf8_movemask16_neon_(astral) << (16 * quarter);
                }
                member = (member & ~four_byte) | (astral_member & four_byte);
            }

            hits = member & valid_starts;
        }
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

#pragma endregion Forward driver

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#endif // SZ_USE_NEON

#ifdef __cplusplus
}
#endif

#endif // STRINGZILLA_UTF8_TOKENS_NEON_H_
