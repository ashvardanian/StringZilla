/**
 *  @brief LoongArch LASX backend for UTF-8 newline and whitespace delimiter scanning.
 *  @file include/stringzilla/utf8_tokens/lasx.h
 *  @author Ash Vardanian
 */
#ifndef STRINGZILLA_UTF8_TOKENS_LASX_H_
#define STRINGZILLA_UTF8_TOKENS_LASX_H_

#include "stringzilla/types.h"
#include "stringzilla/utf8_tokens/serial.h"
#include "stringzilla/utf8_runes/lasx.h"

#ifdef __cplusplus
extern "C" {
#endif

#if SZ_USE_LASX
/** @brief  Peel the tile's first `emit_count` matches with a `__lasx_xvperm_w` left-pack, 4 lanes per sub-block.
 *  Each sub-block gathers its set `(position+lane, length)` pairs to the front (same dword-index table as
 *  `sz_utf8_iterate_peel_haswell_`) and element-stores `min(popcount, remaining)` at the advancing cursor. */
SZ_HELPER_AUTO void sz_utf8_iterate_peel_lasx_(                                //
    sz_u32_t start_bits, sz_u32_t two_byte_starts, sz_u32_t three_byte_starts, //
    sz_size_t emit_count, sz_size_t position,                                  //
    sz_size_t *match_offsets, sz_size_t *match_lengths) {

    // Per-file copy of the AVX2 backend's left-pack table: row `[m]` holds the 8 dword indices that gather the
    // `m`-selected u64 lanes (of 4, each a dword pair) to the front for `__lasx_xvperm_w`.
    static sz_u32_t const compact_lut[16][8] = {
        {0, 0, 0, 0, 0, 0, 0, 0}, {0, 1, 0, 0, 0, 0, 0, 0}, {2, 3, 0, 0, 0, 0, 0, 0}, {0, 1, 2, 3, 0, 0, 0, 0},
        {4, 5, 0, 0, 0, 0, 0, 0}, {0, 1, 4, 5, 0, 0, 0, 0}, {2, 3, 4, 5, 0, 0, 0, 0}, {0, 1, 2, 3, 4, 5, 0, 0},
        {6, 7, 0, 0, 0, 0, 0, 0}, {0, 1, 6, 7, 0, 0, 0, 0}, {2, 3, 6, 7, 0, 0, 0, 0}, {0, 1, 2, 3, 6, 7, 0, 0},
        {4, 5, 6, 7, 0, 0, 0, 0}, {0, 1, 4, 5, 6, 7, 0, 0}, {2, 3, 4, 5, 6, 7, 0, 0}, {0, 1, 2, 3, 4, 5, 6, 7},
    };
    static sz_u64_t const lane_ramp[4] = {0, 1, 2, 3};

    __m256i const lane_ramp_u64x4 = __lasx_xvld(lane_ramp, 0);
    sz_size_t emitted = 0;
    for (sz_size_t sub_block = 0; sub_block < 8 && emitted < emit_count; ++sub_block) {
        sz_u32_t const submask = (start_bits >> (sub_block * 4)) & 0xFu;
        if (!submask) continue;

        sz_size_t const base_lane = sub_block * 4;
        // Per-lane length: 1, plus 1 on a 2-byte start, plus 2 on a 3-byte start (the masks are disjoint). Pack
        // the four sub-block lengths into one byte word and widen to four `u64` lanes with one `vext2xv_du_bu`.
        sz_u32_t const two_byte_sub = (two_byte_starts >> base_lane) & 0xFu;
        sz_u32_t const three_byte_sub = (three_byte_starts >> base_lane) & 0xFu;
        sz_u32_t packed_lengths = 0;
        for (sz_size_t lane_in_block = 0; lane_in_block < 4; ++lane_in_block) {
            sz_u32_t const match_length = 1u + ((two_byte_sub >> lane_in_block) & 1u) +
                                          2u * ((three_byte_sub >> lane_in_block) & 1u);
            packed_lengths |= match_length << (lane_in_block * 8);
        }
        __m256i const lengths_byte_vec = __lasx_xvinsgr2vr_w(__lasx_xvreplgr2vr_b(0), (int)packed_lengths, 0);
        __m256i const offsets_u64x4 = __lasx_xvadd_d(__lasx_xvreplgr2vr_d((long long)(position + base_lane)),
                                                     lane_ramp_u64x4);
        __m256i const lengths_u64x4 = __lasx_vext2xv_du_bu(lengths_byte_vec);

        __m256i const permutation = __lasx_xvld(compact_lut[submask], 0);
        __m256i const packed_offsets = __lasx_xvperm_w(offsets_u64x4, permutation);
        __m256i const packed_lengths_u64x4 = __lasx_xvperm_w(lengths_u64x4, permutation);

        sz_size_t const taken = sz_min_of_two((sz_size_t)sz_u32_popcount(submask), emit_count - emitted);
        sz_utf8_iterate_store_group_lasx_(packed_offsets, taken, match_offsets + emitted);
        sz_utf8_iterate_store_group_lasx_(packed_lengths_u64x4, taken, match_lengths + emitted);
        emitted += taken;
    }
}

SZ_API_COMPTIME sz_size_t sz_utf8_newlines_lasx(        //
    sz_cptr_t text, sz_size_t length,                   //
    sz_size_t *match_offsets, sz_size_t *match_lengths, //
    sz_size_t matches_capacity, sz_size_t *bytes_consumed) {

    sz_u8_t const *text_u8 = (sz_u8_t const *)text;
    sz_size_t count = 0, position = 0;

    __m256i newline_vec = __lasx_xvreplgr2vr_b('\n'), vertical_tab_vec = __lasx_xvreplgr2vr_b('\v'),
            form_feed_vec = __lasx_xvreplgr2vr_b('\f'), carriage_return_vec = __lasx_xvreplgr2vr_b('\r'),
            lead_c2_vec = __lasx_xvreplgr2vr_b((char)0xC2), x_85_vec = __lasx_xvreplgr2vr_b((char)0x85),
            lead_e2_vec = __lasx_xvreplgr2vr_b((char)0xE2), byte_80_vec = __lasx_xvreplgr2vr_b((char)0x80),
            x_a8_vec = __lasx_xvreplgr2vr_b((char)0xA8), x_a9_vec = __lasx_xvreplgr2vr_b((char)0xA9);

    // Trust delimiter STARTS only in lanes [0,29] and step by 30, so any <=3-byte delimiter from a trusted
    // lane is fully loaded; the peel honours `matches_capacity` and may cut mid-tile.
    while (position + 32 <= length && count < matches_capacity) {
        __m256i window = __lasx_xvld(text_u8 + position, 0);

        // 1-byte newline indicators & matches.
        sz_u32_t newline_mask = sz_xvmovemask_b_utf8_lasx_(__lasx_xvseq_b(window, newline_vec));
        sz_u32_t carriage_return_mask = sz_xvmovemask_b_utf8_lasx_(__lasx_xvseq_b(window, carriage_return_vec));
        sz_u32_t one_byte_mask = newline_mask | sz_xvmovemask_b_utf8_lasx_(__lasx_xvseq_b(window, vertical_tab_vec)) |
                                 sz_xvmovemask_b_utf8_lasx_(__lasx_xvseq_b(window, form_feed_vec)) |
                                 carriage_return_mask;

        // 2-byte NEL (C2 85); 3-byte LS/PS (E2 80 A8/A9) - computed unconditionally.
        sz_u32_t lead_c2_mask = sz_xvmovemask_b_utf8_lasx_(__lasx_xvseq_b(window, lead_c2_vec));
        sz_u32_t x_85_mask = sz_xvmovemask_b_utf8_lasx_(__lasx_xvseq_b(window, x_85_vec));
        sz_u32_t lead_e2_mask = sz_xvmovemask_b_utf8_lasx_(__lasx_xvseq_b(window, lead_e2_vec));
        sz_u32_t byte_80_mask = sz_xvmovemask_b_utf8_lasx_(__lasx_xvseq_b(window, byte_80_vec));
        sz_u32_t x_a8_mask = sz_xvmovemask_b_utf8_lasx_(__lasx_xvseq_b(window, x_a8_vec));
        sz_u32_t x_a9_mask = sz_xvmovemask_b_utf8_lasx_(__lasx_xvseq_b(window, x_a9_vec));

        sz_u32_t nel_mask = lead_c2_mask & (x_85_mask >> 1);                       // C2 85
        sz_u32_t lead_e280_mask = lead_e2_mask & (byte_80_mask >> 1);              // E2 80
        sz_u32_t line_para_mask = lead_e280_mask & ((x_a8_mask | x_a9_mask) >> 2); // E2 80 A8/A9

        // CRLF: a CR whose next lane is LF is a single 2-byte match; its trailing LF must not also emit.
        sz_u32_t crlf_mask = carriage_return_mask & (newline_mask >> 1);
        sz_u32_t lf_of_crlf_mask = newline_mask & (carriage_return_mask << 1);

        sz_u32_t two_byte_starts = crlf_mask | nel_mask;
        sz_u32_t three_byte_starts = line_para_mask;
        sz_u32_t start_bits = (one_byte_mask | nel_mask | line_para_mask) & ~lf_of_crlf_mask;
        start_bits &= 0x3FFFFFFFu; // Trust lanes [0,29]; step 30.

        // Suppress a leading LF already consumed by a CRLF that straddled the previous tile edge.
        if (position != 0 && text_u8[position - 1] == '\r') start_bits &= ~(newline_mask & 1u);

        sz_size_t const window_matches = (sz_size_t)sz_u32_popcount(start_bits);
        sz_size_t const emit_count = sz_min_of_two(window_matches, matches_capacity - count);
        if (emit_count)
            sz_utf8_iterate_peel_lasx_(start_bits, two_byte_starts, three_byte_starts, emit_count, position,
                                       match_offsets + count, match_lengths + count);
        count += emit_count;
        if (count == matches_capacity) { // output buffer full: resume past the last emitted match.
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

SZ_API_COMPTIME sz_size_t sz_utf8_whitespaces_lasx(     //
    sz_cptr_t text, sz_size_t length,                   //
    sz_size_t *match_offsets, sz_size_t *match_lengths, //
    sz_size_t matches_capacity, sz_size_t *bytes_consumed) {

    sz_u8_t const *text_u8 = (sz_u8_t const *)text;
    sz_size_t count = 0, position = 0;

    __m256i x_20_vec = __lasx_xvreplgr2vr_b(' '), x_08_vec = __lasx_xvreplgr2vr_b((char)0x08),
            x_0e_vec = __lasx_xvreplgr2vr_b((char)0x0E), lead_c2_vec = __lasx_xvreplgr2vr_b((char)0xC2),
            x_85_vec = __lasx_xvreplgr2vr_b((char)0x85), x_a0_vec = __lasx_xvreplgr2vr_b((char)0xA0),
            x_e1_vec = __lasx_xvreplgr2vr_b((char)0xE1), lead_e2_vec = __lasx_xvreplgr2vr_b((char)0xE2),
            x_e3_vec = __lasx_xvreplgr2vr_b((char)0xE3), x_9a_vec = __lasx_xvreplgr2vr_b((char)0x9A),
            byte_80_vec = __lasx_xvreplgr2vr_b((char)0x80), x_81_vec = __lasx_xvreplgr2vr_b((char)0x81),
            x_8d_vec = __lasx_xvreplgr2vr_b((char)0x8D), x_a8_vec = __lasx_xvreplgr2vr_b((char)0xA8),
            x_a9_vec = __lasx_xvreplgr2vr_b((char)0xA9), x_af_vec = __lasx_xvreplgr2vr_b((char)0xAF),
            x_9f_vec = __lasx_xvreplgr2vr_b((char)0x9F);

    while (position + 32 <= length && count < matches_capacity) {
        __m256i window = __lasx_xvld(text_u8 + position, 0);

        // 1-byte: space, plus the contiguous range [\t, \r] == [9, 13] via signed band 0x08 < b < 0x0E.
        __m256i tab_lower_bound = __lasx_xvslt_b(x_08_vec, window);
        __m256i carriage_return_upper_bound = __lasx_xvslt_b(window, x_0e_vec);
        __m256i one_byte_cmp = __lasx_xvor_v(__lasx_xvseq_b(window, x_20_vec),
                                             __lasx_xvand_v(tab_lower_bound, carriage_return_upper_bound));
        sz_u32_t one_byte_mask = sz_xvmovemask_b_utf8_lasx_(one_byte_cmp);

        // 2-byte: C2 85 (NEL), C2 A0 (NBSP).
        sz_u32_t lead_c2_mask = sz_xvmovemask_b_utf8_lasx_(__lasx_xvseq_b(window, lead_c2_vec));
        sz_u32_t x_85_mask = sz_xvmovemask_b_utf8_lasx_(__lasx_xvseq_b(window, x_85_vec));
        sz_u32_t x_a0_mask = sz_xvmovemask_b_utf8_lasx_(__lasx_xvseq_b(window, x_a0_vec));
        sz_u32_t two_byte_starts = lead_c2_mask & ((x_85_mask | x_a0_mask) >> 1);

        // 3-byte: E1 9A 80 (ogham); E2 80 [80-8D]; E2 80 AF; E2 81 9F; E2 80 A8/A9; E3 80 80.
        sz_u32_t x_e1_mask = sz_xvmovemask_b_utf8_lasx_(__lasx_xvseq_b(window, x_e1_vec));
        sz_u32_t lead_e2_mask = sz_xvmovemask_b_utf8_lasx_(__lasx_xvseq_b(window, lead_e2_vec));
        sz_u32_t x_e3_mask = sz_xvmovemask_b_utf8_lasx_(__lasx_xvseq_b(window, x_e3_vec));
        sz_u32_t x_9a_mask = sz_xvmovemask_b_utf8_lasx_(__lasx_xvseq_b(window, x_9a_vec));
        sz_u32_t byte_80_mask = sz_xvmovemask_b_utf8_lasx_(__lasx_xvseq_b(window, byte_80_vec));
        sz_u32_t x_81_mask = sz_xvmovemask_b_utf8_lasx_(__lasx_xvseq_b(window, x_81_vec));
        sz_u32_t x_a8_mask = sz_xvmovemask_b_utf8_lasx_(__lasx_xvseq_b(window, x_a8_vec));
        sz_u32_t x_a9_mask = sz_xvmovemask_b_utf8_lasx_(__lasx_xvseq_b(window, x_a9_vec));
        sz_u32_t x_af_mask = sz_xvmovemask_b_utf8_lasx_(__lasx_xvseq_b(window, x_af_vec));
        sz_u32_t x_9f_mask = sz_xvmovemask_b_utf8_lasx_(__lasx_xvseq_b(window, x_9f_vec));
        // [0x80, 0x8D] range: unsigned `b >= 0x80` AND `b <= 0x8D`.
        __m256i x_80_ge_cmp = __lasx_xvsle_bu(byte_80_vec, window);
        __m256i x_8d_le_cmp = __lasx_xvsle_bu(window, x_8d_vec);
        sz_u32_t x_8d_range_mask = sz_xvmovemask_b_utf8_lasx_(__lasx_xvand_v(x_80_ge_cmp, x_8d_le_cmp));

        sz_u32_t lead_e280_mask = lead_e2_mask & (byte_80_mask >> 1);                      // E2 80
        sz_u32_t ogham_mask = x_e1_mask & (x_9a_mask >> 1) & (byte_80_mask >> 2);          // E1 9A 80
        sz_u32_t range_e280_mask = lead_e280_mask & (x_8d_range_mask >> 2);                // E2 80 [80-8D]
        sz_u32_t nnbsp_mask = lead_e280_mask & (x_af_mask >> 2);                           // E2 80 AF
        sz_u32_t mmsp_mask = lead_e2_mask & (x_81_mask >> 1) & (x_9f_mask >> 2);           // E2 81 9F
        sz_u32_t line_mask = lead_e280_mask & (x_a8_mask >> 2);                            // E2 80 A8
        sz_u32_t para_mask = lead_e280_mask & (x_a9_mask >> 2);                            // E2 80 A9
        sz_u32_t ideographic_mask = x_e3_mask & (byte_80_mask >> 1) & (byte_80_mask >> 2); // E3 80 80
        sz_u32_t three_byte_starts = ogham_mask | range_e280_mask | nnbsp_mask | mmsp_mask | line_mask | para_mask |
                                     ideographic_mask;

        sz_u32_t start_bits = (one_byte_mask | two_byte_starts | three_byte_starts) & 0x3FFFFFFFu; // lanes [0,29]

        sz_size_t const window_matches = (sz_size_t)sz_u32_popcount(start_bits);
        sz_size_t const emit_count = sz_min_of_two(window_matches, matches_capacity - count);
        if (emit_count)
            sz_utf8_iterate_peel_lasx_(start_bits, two_byte_starts, three_byte_starts, emit_count, position,
                                       match_offsets + count, match_lengths + count);
        count += emit_count;
        if (count == matches_capacity) { // output buffer full: resume past the last emitted match.
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

#pragma endregion Multistep Newline &Whitespace Iteration
#endif // SZ_USE_LASX

#ifdef __cplusplus
}
#endif

#endif // STRINGZILLA_UTF8_TOKENS_LASX_H_
