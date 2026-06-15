/**
 *  @brief NEON backend for UTF-8 traversal.
 *  @file include/stringzilla/utf8_iterate/neon.h
 *  @author Ash Vardanian
 *  @sa utf8.h
 */
#ifndef STRINGZILLA_UTF8_ITERATE_NEON_H_
#define STRINGZILLA_UTF8_ITERATE_NEON_H_

#include "stringzilla/types.h"
#include "stringzilla/utf8_iterate/serial.h"

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

SZ_PUBLIC sz_cptr_t sz_utf8_unpack_chunk_neon(  //
    sz_cptr_t text, sz_size_t length,           //
    sz_rune_t *runes, sz_size_t runes_capacity, //
    sz_size_t *runes_unpacked) {
    return sz_utf8_unpack_chunk_serial(text, length, runes, runes_capacity, runes_unpacked);
}

SZ_INTERNAL sz_u64_t sz_utf8_vreinterpretq_u8_u4_neon_(uint8x16_t vec) {
    // Use `vshrn` to produce a bitmask, similar to `movemask` in SSE.
    // https://community.arm.com/arm-community-blogs/b/infrastructure-solutions-blog/posts/porting-x86-vector-bitmask-optimizations-to-arm-neon
    return vget_lane_u64(vreinterpret_u64_u8(vshrn_n_u16(vreinterpretq_u16_u8(vec), 4)), 0) & 0x8888888888888888ull;
}

#pragma region Multistep newline and whitespace iteration

/*  Multistep newline / whitespace iteration (NEON / AArch64).
 *
 *  Each FULL 16-byte tile is classified branchlessly into per-length delimiter-start masks (all multi-byte
 *  masks are computed unconditionally - no data-dependent `if(lead)` gate). NEON has no `movemask` and no
 *  masked load, so we only ever scan full tiles (`position + 16 <= length`) and hand the `< 16`-byte remainder
 *  to the scalar multistep helper in `serial.h`. Each per-lane compare vector is narrowed to a 64-bit nibble
 *  bitmask via the shared `vshrn`-based `sz_utf8_vreinterpretq_u8_u4_neon_` helper (bit `4*j+3` marks lane `j`,
 *  so the bit-per-lane stride is 4 and `sz_u64_popcount(start_bits)` already counts whole lanes, one set bit per
 *  match). We trust delimiter starts only in lanes [0,13] and step the cursor by 14 so any <=3-byte delimiter
 *  from a trusted lane is fully loaded. NEON has no `vpcompressb`, so the peel left-packs the 16 lanes in four
 *  4-lane sub-blocks with a `vqtbl2q_u8` table (the same `compact_lut`/`popcount_lut` the NEON string-sort uses),
 *  building the absolute `(position+lane, length)` u64 pairs from a caller-supplied per-lane `length_per_lane`
 *  byte vector (`1 + is_two_byte + 2*is_three_byte`, mirroring Ice Lake) - no `ctz` index-find and no group/else
 *  branch. The caller computes the capacity cut like Ice Lake (`window_matches = popcount(start_bits)`,
 *  `emit_count = min(window_matches, capacity - count)`, mid-window resume past the last emitted match). A
 *  `t[pos-1] == '\r'` carry suppresses the leading LF that completes a CRLF straddling the tile edge (newlines
 *  only). */

/**
 *  @brief Left-packs the @p submask -selected lanes of one 4-lane sub-block (offsets @p offsets_u8x16x2 and
 *         lengths @p lengths_u8x16x2, each four u64 = 32 bytes) to the @p out_offsets / @p out_lengths cursors,
 *         preserving lane order; returns how many lanes were written. Mirrors `sz_sort_neon_compact4_`: both
 *         half-vectors are stored unconditionally (the caller advances by the surviving count, so the unwritten
 *         tail lands in the scratch slack), so the body stays branchless.
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
 *  @brief  Peel the tile's first `emit_count` matches by SIMD left-pack (no `ctz`, no per-match branch).
 *
 *  Walks the 16 lanes in four 4-lane sub-blocks: each derives its dense 4-bit submask from the nibble-stride
 *  `start_bits`, builds the four candidate `(position + lane, length)` u64 pairs (the lengths gathered from
 *  `length_per_lane` with `vqtbl1q_u8`, the offsets from `position + lane`), and `vqtbl2q_u8`-compacts them into a
 *  fixed-width stack scratch with full-width stores. The low `emit_count` entries are then copied to the caller's
 *  output. The scratch is 20-wide so the trailing 4-lane spill of the last compacted sub-block always lands inside
 *  it (at most 14 trusted matches per tile).
 */
SZ_INTERNAL void sz_utf8_iterate_peel_tile_neon_(    //
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
            sz_utf8_iterate_peel_tile_neon_(start_bits, length_per_lane, emit_count, position, match_offsets + count,
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
            sz_utf8_iterate_peel_tile_neon_(start_bits, length_per_lane, emit_count, position, match_offsets + count,
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

SZ_PUBLIC sz_size_t sz_utf8_count_neon(sz_cptr_t text, sz_size_t length) {
    sz_u128_vec_t text_vec, headers_vec, continuation_vec;
    uint8x16_t continuation_mask_u8x16 = vdupq_n_u8(0xC0);
    uint8x16_t continuation_pattern_u8x16 = vdupq_n_u8(0x80);
    sz_u8_t const *text_u8 = (sz_u8_t const *)text;
    uint64x2_t char_count_u64x2 = vdupq_n_u64(0);
    while (length >= 16) {
        text_vec.u8x16 = vld1q_u8(text_u8);
        headers_vec.u8x16 = vandq_u8(text_vec.u8x16, continuation_mask_u8x16);
        continuation_vec.u8x16 = vceqq_u8(headers_vec.u8x16, continuation_pattern_u8x16);
        // Convert 0xFF/0x00 into 1/0 and sum.
        uint8x16_t start_flags = vshrq_n_u8(vmvnq_u8(continuation_vec.u8x16), 7);
        uint16x8_t sum16 = vpaddlq_u8(start_flags);
        uint32x4_t sum32 = vpaddlq_u16(sum16);
        uint64x2_t sum64 = vpaddlq_u32(sum32);
        char_count_u64x2 = vaddq_u64(char_count_u64x2, sum64);
        text_u8 += 16;
        length -= 16;
    }

    sz_size_t char_count = vgetq_lane_u64(char_count_u64x2, 0) + vgetq_lane_u64(char_count_u64x2, 1);
    if (length) char_count += sz_utf8_count_serial((sz_cptr_t)text_u8, length);
    return char_count;
}

SZ_PUBLIC sz_cptr_t sz_utf8_find_nth_neon(sz_cptr_t text, sz_size_t length, sz_size_t n) {
    // TODO: Implement a NEON-accelerated version of sz_utf8_find_nth in absence of PDEP instruction.
    return sz_utf8_find_nth_serial(text, length, n);
}

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif

/*  UAX-29 word boundary detection.
 *
 *  An all-ASCII window is classified to Word_Break property values via a 128-entry table lookup, the full set
 *  of ASCII no-break rules is evaluated branchlessly as neighbour bit-shifts (see
 *  `sz_utf8_word_break_boundary_mask_neon_`), and the resulting boundary positions are emitted directly - no
 *  per-candidate serial oracle. ASCII has no Extend/Format/ZWJ/Regional_Indicator/Hebrew/Katakana, so the
 *  stateful WB4 and WB15/16 rules never apply inside such a window; non-ASCII windows and the leading/trailing
 *  edges take a single step of the serial reference, keeping the result byte-for-byte identical to `_serial`. */

/*  Classify an all-ASCII 16-byte window into Word_Break property values via the 128-entry table. The property
 *  table is not separable into two nibble bitsets (ALetter spans non-rectangular byte ranges), so this is a
 *  full byte lookup: two `vqtbl4q_u8` cover entries 0..63 and 64..127 (each returns zero for out-of-range
 *  indices), combined with an OR. Valid only when every byte is ASCII (< 0x80). */
SZ_INTERNAL uint8x16_t sz_utf8_word_break_classify_property_neon_(uint8x16_t bytes) {
    uint8x16x4_t low_table = vld1q_u8_x4(sz_utf8_word_break_property_ascii_);         // entries 0..63
    uint8x16x4_t high_table = vld1q_u8_x4(sz_utf8_word_break_property_ascii_ + 64);   // entries 64..127
    uint8x16_t from_low = vqtbl4q_u8(low_table, bytes);                               // zero where byte >= 64
    uint8x16_t from_high = vqtbl4q_u8(high_table, veorq_u8(bytes, vdupq_n_u8(0x40))); // zero where byte < 64
    return vorrq_u8(from_low, from_high);
}

/*  Given an all-ASCII 16-byte window, return a nibble-mask (bit 4*j+3 set) of word boundaries at lanes [2,14] -
 *  the lanes whose i-2 and i+1 neighbours are both in-window, so every UAX-29 rule that can apply to ASCII is
 *  resolved without the serial oracle. ASCII contains no Extend/Format/ZWJ/Regional_Indicator/Hebrew/Katakana,
 *  so WB4 and WB15/16 never apply and WB6/7/11/12 reduce to neighbour lane shifts. */
SZ_INTERNAL sz_u64_t sz_utf8_word_break_boundary_mask_neon_(uint8x16_t window_bytes) {
    uint8x16_t classes = sz_utf8_word_break_classify_property_neon_(window_bytes);
    uint8x16_t aletter = vceqq_u8(classes, vdupq_n_u8(sz_tr29_word_break_aletter_k));
    uint8x16_t numeric = vceqq_u8(classes, vdupq_n_u8(sz_tr29_word_break_numeric_k));
    uint8x16_t extendnumlet = vceqq_u8(classes, vdupq_n_u8(sz_tr29_word_break_extendnumlet_k));
    uint8x16_t midletter = vceqq_u8(classes, vdupq_n_u8(sz_tr29_word_break_midletter_k));
    uint8x16_t midnum = vceqq_u8(classes, vdupq_n_u8(sz_tr29_word_break_midnum_k));
    uint8x16_t mid_quotes = vceqq_u8(classes, vdupq_n_u8(sz_tr29_word_break_mid_quotes_k));
    uint8x16_t carriage_return = vceqq_u8(classes, vdupq_n_u8(sz_tr29_word_break_cr_k));
    uint8x16_t line_feed = vceqq_u8(classes, vdupq_n_u8(sz_tr29_word_break_lf_k));
    uint8x16_t mid_letter_or_quotes = vorrq_u8(midletter, mid_quotes);
    uint8x16_t mid_num_or_quotes = vorrq_u8(midnum, mid_quotes);
    uint8x16_t zero = vdupq_n_u8(0);

    // Neighbour group vectors: `previous` brings lane (j-1) to lane j, `before_previous` lane (j-2), `next` (j+1).
    uint8x16_t aletter_previous = vextq_u8(zero, aletter, 15);
    uint8x16_t aletter_before_previous = vextq_u8(zero, aletter, 14);
    uint8x16_t aletter_next = vextq_u8(aletter, zero, 1);
    uint8x16_t numeric_previous = vextq_u8(zero, numeric, 15);
    uint8x16_t numeric_before_previous = vextq_u8(zero, numeric, 14);
    uint8x16_t numeric_next = vextq_u8(numeric, zero, 1);
    uint8x16_t extendnumlet_previous = vextq_u8(zero, extendnumlet, 15);
    uint8x16_t mid_letter_or_quotes_previous = vextq_u8(zero, mid_letter_or_quotes, 15);
    uint8x16_t mid_num_or_quotes_previous = vextq_u8(zero, mid_num_or_quotes, 15);
    uint8x16_t carriage_return_previous = vextq_u8(zero, carriage_return, 15);

    uint8x16_t join = vandq_u8(carriage_return_previous, line_feed);                                 // WB3  CR x LF
    join = vorrq_u8(join, vandq_u8(aletter_previous, aletter));                                      // WB5
    join = vorrq_u8(join, vandq_u8(vandq_u8(aletter_previous, mid_letter_or_quotes), aletter_next)); // WB6
    join = vorrq_u8(join, vandq_u8(vandq_u8(aletter_before_previous, mid_letter_or_quotes_previous), aletter)); // WB7
    join = vorrq_u8(join, vandq_u8(numeric_previous, numeric));                                                 // WB8
    join = vorrq_u8(join, vandq_u8(aletter_previous, numeric));                                                 // WB9
    join = vorrq_u8(join, vandq_u8(numeric_previous, aletter));                                                 // WB10
    join = vorrq_u8(join, vandq_u8(vandq_u8(numeric_previous, mid_num_or_quotes), numeric_next));               // WB11
    join = vorrq_u8(join, vandq_u8(vandq_u8(numeric_before_previous, mid_num_or_quotes_previous), numeric));    // WB12
    uint8x16_t aletter_numeric_or_extendnumlet_previous = vorrq_u8(vorrq_u8(aletter_previous, numeric_previous),
                                                                   extendnumlet_previous);
    join = vorrq_u8(join, vandq_u8(aletter_numeric_or_extendnumlet_previous, extendnumlet)); // WB13a
    join = vorrq_u8(join, vandq_u8(extendnumlet_previous, vorrq_u8(aletter, numeric)));      // WB13b

    uint8x16_t boundary = vmvnq_u8(join);
    // Trust only lanes [2,14]; lanes 0,1 lack a left neighbour and lane 15 lacks a right neighbour.
    static sz_u8_t const trusted_lanes[16] = {0,    0,    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
                                              0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0};
    boundary = vandq_u8(boundary, vld1q_u8(trusted_lanes));
    return sz_utf8_vreinterpretq_u8_u4_neon_(boundary) & 0x8888888888888888ull;
}

/**
 *  @brief  Left-pack the @p submask -selected lanes of one 4-lane sub-block of @p boundaries (four absolute u64
 *          positions held as `low` = lanes 0,1 and `high` = lanes 2,3) to the front in ASCENDING lane order.
 *
 *  Per-file byte-table left-pack mirroring `sz_utf8_iterate_compact4_neon_`: `compact_lut[m]` gathers the
 *  `m`-selected lanes (each an 8-byte u64) to the front via `vqtbl2q_u8`. The returned `uint8x16x2_t` holds the
 *  compacted lanes 0,1 in `val[0]` and lanes 2,3 in `val[1]`.
 */
SZ_INTERNAL uint8x16x2_t sz_utf8_word_compact4_positions_neon_(uint64x2_t low, uint64x2_t high, sz_u32_t submask) {
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
    uint8x16x2_t lanes_u8x16x2;
    lanes_u8x16x2.val[0] = vreinterpretq_u8_u64(low);
    lanes_u8x16x2.val[1] = vreinterpretq_u8_u64(high);
    uint8x16x2_t compacted_u8x16x2;
    compacted_u8x16x2.val[0] = vqtbl2q_u8(lanes_u8x16x2, vld1q_u8(compact_lut[submask]));
    compacted_u8x16x2.val[1] = vqtbl2q_u8(lanes_u8x16x2, vld1q_u8(compact_lut[submask] + 16));
    return compacted_u8x16x2;
}

/**
 *  @brief  Descending-order counterpart of `sz_utf8_word_compact4_positions_neon_`: gathers the @p submask
 *          -selected lanes to the front in HIGH-to-LOW lane order, for the reverse word scan.
 */
SZ_INTERNAL uint8x16x2_t sz_utf8_word_compact4_positions_descending_neon_(uint64x2_t low, uint64x2_t high,
                                                                          sz_u32_t submask) {
    static sz_u8_t const compact_lut[16][32] = {
        {0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7},
        {0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7},
        {8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7},
        {8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7},
        {16, 17, 18, 19, 20, 21, 22, 23, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7},
        {16, 17, 18, 19, 20, 21, 22, 23, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7},
        {16, 17, 18, 19, 20, 21, 22, 23, 8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7},
        {16, 17, 18, 19, 20, 21, 22, 23, 8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7},
        {24, 25, 26, 27, 28, 29, 30, 31, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7},
        {24, 25, 26, 27, 28, 29, 30, 31, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7},
        {24, 25, 26, 27, 28, 29, 30, 31, 8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7},
        {24, 25, 26, 27, 28, 29, 30, 31, 8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7},
        {24, 25, 26, 27, 28, 29, 30, 31, 16, 17, 18, 19, 20, 21, 22, 23,
         0,  1,  2,  3,  4,  5,  6,  7,  0,  1,  2,  3,  4,  5,  6,  7},
        {24, 25, 26, 27, 28, 29, 30, 31, 16, 17, 18, 19, 20, 21, 22, 23,
         0,  1,  2,  3,  4,  5,  6,  7,  0,  1,  2,  3,  4,  5,  6,  7},
        {24, 25, 26, 27, 28, 29, 30, 31, 16, 17, 18, 19, 20, 21, 22, 23,
         8,  9,  10, 11, 12, 13, 14, 15, 0,  1,  2,  3,  4,  5,  6,  7},
        {24, 25, 26, 27, 28, 29, 30, 31, 16, 17, 18, 19, 20, 21, 22, 23,
         8,  9,  10, 11, 12, 13, 14, 15, 0,  1,  2,  3,  4,  5,  6,  7},
    };
    uint8x16x2_t lanes_u8x16x2;
    lanes_u8x16x2.val[0] = vreinterpretq_u8_u64(low);
    lanes_u8x16x2.val[1] = vreinterpretq_u8_u64(high);
    uint8x16x2_t compacted_u8x16x2;
    compacted_u8x16x2.val[0] = vqtbl2q_u8(lanes_u8x16x2, vld1q_u8(compact_lut[submask]));
    compacted_u8x16x2.val[1] = vqtbl2q_u8(lanes_u8x16x2, vld1q_u8(compact_lut[submask] + 16));
    return compacted_u8x16x2;
}

/** @brief  Bounded store of the low @p stored (0..4) u64 lanes of @p low (lanes 0,1) and @p high (lanes 2,3) at
 *          @p destination, via a fixed 4-wide stack scratch and a copy of the surviving prefix. */
SZ_INTERNAL void sz_utf8_word_store_lanes_neon_(uint64x2_t low, uint64x2_t high, sz_size_t stored,
                                                sz_size_t *destination) {
    sz_size_t scratch[4];
    vst1q_u64((sz_u64_t *)scratch, low);
    vst1q_u64((sz_u64_t *)scratch + 2, high);
    for (sz_size_t lane = 0; lane < stored; ++lane) destination[lane] = scratch[lane];
}

SZ_PUBLIC sz_size_t sz_utf8_word_find_boundaries_neon( //
    sz_cptr_t text, sz_size_t length,                  //
    sz_size_t *word_starts, sz_size_t *word_lengths,   //
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

    // Oracle-free fast path: an all-ASCII window [position-2, position+14) resolves boundaries at positions
    // [position, position+12]; one fixed sub-block loop compacts each group and emits it as a shifted-difference,
    // carrying the open `word_start` into lane 0 and the previous boundary into lanes 1..3.
    static sz_u64_t const lane_offset_low[2] = {0, 1};
    static sz_u64_t const lane_offset_high[2] = {2, 3};
    uint64x2_t const lane_offset_low_u64x2 = vld1q_u64(lane_offset_low);
    uint64x2_t const lane_offset_high_u64x2 = vld1q_u64(lane_offset_high);
    while (position < length) {
        int ascii_window = position >= 2 && position + 14 <= length;
        uint8x16_t window = vdupq_n_u8(0);
        if (ascii_window) {
            window = vld1q_u8(text_u8 + position - 2); // lane j = byte position-2+j
            ascii_window = vmaxvq_u8(window) < 0x80;
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

        sz_u64_t boundary = sz_utf8_word_break_boundary_mask_neon_(window);
        for (sz_size_t sub_block = 0; sub_block < 4; ++sub_block) {
            sz_size_t const base_lane = sub_block * 4;
            sz_u32_t const submask = (sz_u32_t)(((boundary >> (base_lane * 4 + 3)) & 1) |
                                                (((boundary >> (base_lane * 4 + 7)) & 1) << 1) |
                                                (((boundary >> (base_lane * 4 + 11)) & 1) << 2) |
                                                (((boundary >> (base_lane * 4 + 15)) & 1) << 3));
            if (!submask) continue;
            sz_size_t const taken = (sz_size_t)sz_u32_popcount(submask);
            sz_size_t const stored = sz_min_of_two(taken, words_capacity - words);

            uint64x2_t const base_broadcast = vdupq_n_u64((sz_u64_t)position - 2 + base_lane);
            uint64x2_t const positions_low = vaddq_u64(base_broadcast, lane_offset_low_u64x2);
            uint64x2_t const positions_high = vaddq_u64(base_broadcast, lane_offset_high_u64x2);
            uint8x16x2_t const boundaries = sz_utf8_word_compact4_positions_neon_(positions_low, positions_high,
                                                                                  submask);
            uint64x2_t const boundaries_low = vreinterpretq_u64_u8(boundaries.val[0]);
            uint64x2_t const boundaries_high = vreinterpretq_u64_u8(boundaries.val[1]);
            uint64x2_t const word_start_u64x2 = vdupq_n_u64((sz_u64_t)word_start);
            uint64x2_t const starts_low = vextq_u64(word_start_u64x2, boundaries_low, 1);
            uint64x2_t const starts_high = vextq_u64(boundaries_low, boundaries_high, 1);

            sz_utf8_word_store_lanes_neon_(starts_low, starts_high, stored, word_starts + words);
            sz_utf8_word_store_lanes_neon_(vsubq_u64(boundaries_low, starts_low),
                                           vsubq_u64(boundaries_high, starts_high), stored, word_lengths + words);
            words += stored;
            if (stored) word_start = word_starts[words - 1] + word_lengths[words - 1];
            if (words == words_capacity) {
                if (bytes_consumed) *bytes_consumed = word_start;
                return words;
            }
        }
        position += 13; // Resolved [position, position+12]; next unresolved boundary is at position+13.
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

SZ_PUBLIC sz_size_t sz_utf8_word_rfind_boundaries_neon( //
    sz_cptr_t text, sz_size_t length,                   //
    sz_size_t *word_starts, sz_size_t *word_lengths,    //
    sz_size_t words_capacity, sz_size_t *bytes_consumed) {

    sz_size_t words = 0;
    if (length == 0 || words_capacity == 0) {
        if (bytes_consumed) *bytes_consumed = length;
        return 0;
    }

    sz_u8_t const *text_u8 = (sz_u8_t const *)text;
    sz_size_t word_end = length; // End of the word currently being accumulated (always a boundary).
    sz_size_t position = length - 1;
    while (position > 0 && ((sz_u8_t)text[position] & 0xC0) == 0x80) position--;

    // Oracle-free fast path: an all-ASCII window [position-14, position+2) resolves boundaries at positions
    // [position-12, position]; one fixed sub-block loop walks high-to-low, compacting each group in descending
    // lane order and emitting it as a shifted-difference (lane 0 carries the open `word_end`).
    static sz_u64_t const lane_offset_low[2] = {0, 1};
    static sz_u64_t const lane_offset_high[2] = {2, 3};
    uint64x2_t const lane_offset_low_u64x2 = vld1q_u64(lane_offset_low);
    uint64x2_t const lane_offset_high_u64x2 = vld1q_u64(lane_offset_high);
    while (position > 0) {
        int ascii_window = position >= 14 && position + 2 <= length;
        uint8x16_t window = vdupq_n_u8(0);
        if (ascii_window) {
            window = vld1q_u8(text_u8 + position - 14); // lane j = byte position-14+j; lane 14 = position
            ascii_window = vmaxvq_u8(window) < 0x80;
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
            while (position > 0 && ((sz_u8_t)text[position] & 0xC0) == 0x80) position--;
            continue;
        }

        sz_u64_t boundary = sz_utf8_word_break_boundary_mask_neon_(window);
        for (sz_size_t sub_block = 4; sub_block-- > 0;) { // high-to-low for descending emission
            sz_size_t const base_lane = sub_block * 4;
            sz_u32_t const submask = (sz_u32_t)(((boundary >> (base_lane * 4 + 3)) & 1) |
                                                (((boundary >> (base_lane * 4 + 7)) & 1) << 1) |
                                                (((boundary >> (base_lane * 4 + 11)) & 1) << 2) |
                                                (((boundary >> (base_lane * 4 + 15)) & 1) << 3));
            if (!submask) continue;
            sz_size_t const taken = (sz_size_t)sz_u32_popcount(submask);
            sz_size_t const stored = sz_min_of_two(taken, words_capacity - words);

            uint64x2_t const base_broadcast = vdupq_n_u64((sz_u64_t)position - 14 + base_lane);
            uint64x2_t const positions_low = vaddq_u64(base_broadcast, lane_offset_low_u64x2);
            uint64x2_t const positions_high = vaddq_u64(base_broadcast, lane_offset_high_u64x2);
            uint8x16x2_t const boundaries = sz_utf8_word_compact4_positions_descending_neon_(positions_low,
                                                                                             positions_high, submask);
            uint64x2_t const boundaries_low = vreinterpretq_u64_u8(boundaries.val[0]);
            uint64x2_t const boundaries_high = vreinterpretq_u64_u8(boundaries.val[1]);
            uint64x2_t const word_end_u64x2 = vdupq_n_u64((sz_u64_t)word_end);
            uint64x2_t const previous_low = vextq_u64(word_end_u64x2, boundaries_low, 1);
            uint64x2_t const previous_high = vextq_u64(boundaries_low, boundaries_high, 1);

            sz_utf8_word_store_lanes_neon_(boundaries_low, boundaries_high, stored, word_starts + words);
            sz_utf8_word_store_lanes_neon_(vsubq_u64(previous_low, boundaries_low),
                                           vsubq_u64(previous_high, boundaries_high), stored, word_lengths + words);
            words += stored;
            if (stored) word_end = word_starts[words - 1];
            if (words == words_capacity) {
                if (bytes_consumed) *bytes_consumed = word_end;
                return words;
            }
        }
        position -= 13; // Resolved [position-12, position]; next unresolved boundary is at position-13.
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
#endif // SZ_USE_NEON

#ifdef __cplusplus
}
#endif

#endif // STRINGZILLA_UTF8_ITERATE_NEON_H_
