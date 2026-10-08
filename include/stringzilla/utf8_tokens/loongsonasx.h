/**
 *  @file include/stringzilla/utf8_tokens/loongsonasx.h
 *  @author Ash Vardanian
 *  @date June 7, 2026
 *  @brief LoongArch LASX backend for UTF-8 newline and whitespace delimiter scanning.
 */
#ifndef STRINGZILLA_UTF8_TOKENS_LOONGSONASX_H_
#define STRINGZILLA_UTF8_TOKENS_LOONGSONASX_H_

#include "stringzilla/types.h"
#include "stringzilla/utf8_tokens/serial.h"
#include "stringzilla/utf8_runes/loongsonasx.h"

#ifdef __cplusplus
extern "C" {
#endif

#if STRINGZILLA_ARCH_LOONGARCH64_
#if STRINGZILLA_TARGET_LOONGSONASX
#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("lasx"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("lasx")
#endif

/** Peels the tile's first @p emit_count matches with a @c __lasx_xvperm_w left-pack, 4 lanes per
 *  sub-block. Each sub-block gathers its set `(position + lane, length)` pairs to the front, with
 *  the same dword-index table as @ref sz_utf8_iterate_peel_haswell_, and element-stores
 *  `min(popcount, remaining)` at the advancing cursor. */
STRINGZILLA_INLINE void sz_utf8_iterate_peel_loongsonasx_(                     //
    sz_u32_t start_bits, sz_u32_t two_byte_starts, sz_u32_t three_byte_starts, //
    sz_size_t emit_count, sz_size_t position,                                  //
    sz_size_t *match_offsets, sz_size_t *match_lengths) {

    static sz_u64_t const lane_ramp[4] = {0, 1, 2, 3};

    __m256i const lane_ramp_u64x4 = __lasx_xvld(lane_ramp, 0);
    sz_size_t emitted = 0;
    for (sz_size_t sub_block = 0; sub_block < 8 && emitted < emit_count; ++sub_block) {
        sz_u32_t const submask = (start_bits >> (sub_block * 4)) & 0xFu;
        if (!submask) continue;

        sz_size_t const base_lane = sub_block * 4;
        // Per-lane length: 1, plus 1 on a 2-byte start, plus 2 on a 3-byte start (the masks are
        // disjoint). Pack the four sub-block lengths into one byte word and widen to four `u64`
        // lanes with one `vext2xv_du_bu`.
        sz_u32_t const two_byte_sub = (two_byte_starts >> base_lane) & 0xFu;
        sz_u32_t const three_byte_sub = (three_byte_starts >> base_lane) & 0xFu;
        sz_u32_t packed_lengths = 0;
        for (sz_size_t lane_in_block = 0; lane_in_block < 4; ++lane_in_block) {
            sz_u32_t const match_length = 1u + ((two_byte_sub >> lane_in_block) & 1u) +
                                          2u * ((three_byte_sub >> lane_in_block) & 1u);
            packed_lengths |= match_length << (lane_in_block * 8);
        }
        __m256i const lengths_u8x32 = __lasx_xvinsgr2vr_w(__lasx_xvreplgr2vr_b(0), (int)packed_lengths, 0);
        __m256i const offsets_u64x4 = __lasx_xvadd_d(__lasx_xvreplgr2vr_d((long long)(position + base_lane)),
                                                     lane_ramp_u64x4);
        __m256i const lengths_u64x4 = __lasx_vext2xv_du_bu(lengths_u8x32);

        __m256i const permutation_u32x8 = __lasx_xvld(sz_compact4_dword_indices_[submask], 0);
        __m256i const packed_offsets_u64x4 = __lasx_xvperm_w(offsets_u64x4, permutation_u32x8);
        __m256i const packed_lengths_u64x4 = __lasx_xvperm_w(lengths_u64x4, permutation_u32x8);

        sz_size_t const taken = sz_min_of_two((sz_size_t)sz_u32_popcount(submask), emit_count - emitted);
        sz_utf8_iterate_store_group_loongsonasx_(packed_offsets_u64x4, taken, match_offsets + emitted);
        sz_utf8_iterate_store_group_loongsonasx_(packed_lengths_u64x4, taken, match_lengths + emitted);
        emitted += taken;
    }
}

STRINGZILLA_INLINE sz_size_t sz_utf8_newlines_loongsonasx_( //
    sz_cptr_t text, sz_size_t length,                       //
    sz_size_t *match_offsets, sz_size_t *match_lengths,     //
    sz_size_t matches_capacity, sz_size_t *bytes_consumed) {

    sz_u8_t const *text_u8 = (sz_u8_t const *)text;
    sz_size_t count = 0, position = 0;

    __m256i newline_u8x32 = __lasx_xvreplgr2vr_b('\n'), vertical_tab_u8x32 = __lasx_xvreplgr2vr_b('\v'),
            form_feed_u8x32 = __lasx_xvreplgr2vr_b('\f'), carriage_return_u8x32 = __lasx_xvreplgr2vr_b('\r'),
            lead_c2_u8x32 = __lasx_xvreplgr2vr_b((char)0xC2), x_85_u8x32 = __lasx_xvreplgr2vr_b((char)0x85),
            lead_e2_u8x32 = __lasx_xvreplgr2vr_b((char)0xE2), byte_80_u8x32 = __lasx_xvreplgr2vr_b((char)0x80),
            x_a8_u8x32 = __lasx_xvreplgr2vr_b((char)0xA8), x_a9_u8x32 = __lasx_xvreplgr2vr_b((char)0xA9);

    // Only lanes [0,29] may start a delimiter and the step is 30, so each delimiter of up to 3
    // bytes loads in full, and the peel stops mid-tile once `matches_capacity` is reached.
    while (position + 32 <= length && count < matches_capacity) {
        __m256i window_u8x32 = __lasx_xvld(text_u8 + position, 0);

        // 1-byte newline indicators & matches.
        sz_u32_t newline_mask = sz_xvmovemask_b_utf8_loongsonasx_(__lasx_xvseq_b(window_u8x32, newline_u8x32));
        sz_u32_t carriage_return_mask = sz_xvmovemask_b_utf8_loongsonasx_(
            __lasx_xvseq_b(window_u8x32, carriage_return_u8x32));
        sz_u32_t one_byte_mask = newline_mask |
                                 sz_xvmovemask_b_utf8_loongsonasx_(__lasx_xvseq_b(window_u8x32, vertical_tab_u8x32)) |
                                 sz_xvmovemask_b_utf8_loongsonasx_(__lasx_xvseq_b(window_u8x32, form_feed_u8x32)) |
                                 carriage_return_mask;

        // 2-byte NEL (C2 85); 3-byte LS/PS (E2 80 A8/A9) - computed unconditionally.
        sz_u32_t lead_c2_mask = sz_xvmovemask_b_utf8_loongsonasx_(__lasx_xvseq_b(window_u8x32, lead_c2_u8x32));
        sz_u32_t x_85_mask = sz_xvmovemask_b_utf8_loongsonasx_(__lasx_xvseq_b(window_u8x32, x_85_u8x32));
        sz_u32_t lead_e2_mask = sz_xvmovemask_b_utf8_loongsonasx_(__lasx_xvseq_b(window_u8x32, lead_e2_u8x32));
        sz_u32_t byte_80_mask = sz_xvmovemask_b_utf8_loongsonasx_(__lasx_xvseq_b(window_u8x32, byte_80_u8x32));
        sz_u32_t x_a8_mask = sz_xvmovemask_b_utf8_loongsonasx_(__lasx_xvseq_b(window_u8x32, x_a8_u8x32));
        sz_u32_t x_a9_mask = sz_xvmovemask_b_utf8_loongsonasx_(__lasx_xvseq_b(window_u8x32, x_a9_u8x32));

        sz_u32_t nel_mask = lead_c2_mask & (x_85_mask >> 1);                       // C2 85
        sz_u32_t lead_e280_mask = lead_e2_mask & (byte_80_mask >> 1);              // E2 80
        sz_u32_t line_para_mask = lead_e280_mask & ((x_a8_mask | x_a9_mask) >> 2); // E2 80 A8/A9

        // CRLF: a CR before an LF is one 2-byte match, so that LF must not emit on its own.
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
            sz_utf8_iterate_peel_loongsonasx_(start_bits, two_byte_starts, three_byte_starts, emit_count, position,
                                              match_offsets + count, match_lengths + count);
        count += emit_count;
        if (count == matches_capacity) { // output buffer full: resume past the last emitted match.
            position = match_offsets[count - 1] + match_lengths[count - 1];
            break;
        }
        position += 30;
    }

    // Skip the trailing LF of a CRLF straddling into the serial tail, as its CR was a 2-byte match.
    if (position != 0 && position < length && text_u8[position - 1] == '\r' && text_u8[position] == '\n') ++position;
    count += sz_utf8_newlines_serial_((sz_cptr_t)(text_u8 + position), length - position, position,
                                      match_offsets + count, match_lengths + count, matches_capacity - count,
                                      bytes_consumed);
    sz_assert_(sz_utf8_batch_consistent_(length, matches_capacity, count, bytes_consumed ? *bytes_consumed : length,
                                         match_offsets, match_lengths));
    return count;
}

STRINGZILLA_API sz_status_t sz_utf8_newlines_loongsonasx(                           //
    sz_cptr_t text, sz_size_t length,                                               //
    sz_size_t *match_offsets, sz_size_t *match_lengths, sz_size_t matches_capacity, //
    sz_size_t *matches_count, sz_size_t *bytes_consumed, sz_stream_t stream) {
    sz_assert_(stream == STRINGZILLA_NULL);
    *matches_count = sz_utf8_newlines_loongsonasx_(text, length, match_offsets, match_lengths, matches_capacity,
                                                   bytes_consumed);
    return sz_success_k;
}

STRINGZILLA_INLINE sz_size_t sz_utf8_whitespaces_loongsonasx_( //
    sz_cptr_t text, sz_size_t length,                          //
    sz_size_t *match_offsets, sz_size_t *match_lengths,        //
    sz_size_t matches_capacity, sz_size_t *bytes_consumed) {

    sz_u8_t const *text_u8 = (sz_u8_t const *)text;
    sz_size_t count = 0, position = 0;

    __m256i x_20_u8x32 = __lasx_xvreplgr2vr_b(' '), x_08_u8x32 = __lasx_xvreplgr2vr_b((char)0x08),
            x_0e_u8x32 = __lasx_xvreplgr2vr_b((char)0x0E), lead_c2_u8x32 = __lasx_xvreplgr2vr_b((char)0xC2),
            x_85_u8x32 = __lasx_xvreplgr2vr_b((char)0x85), x_a0_u8x32 = __lasx_xvreplgr2vr_b((char)0xA0),
            x_e1_u8x32 = __lasx_xvreplgr2vr_b((char)0xE1), lead_e2_u8x32 = __lasx_xvreplgr2vr_b((char)0xE2),
            x_e3_u8x32 = __lasx_xvreplgr2vr_b((char)0xE3), x_9a_u8x32 = __lasx_xvreplgr2vr_b((char)0x9A),
            byte_80_u8x32 = __lasx_xvreplgr2vr_b((char)0x80), x_81_u8x32 = __lasx_xvreplgr2vr_b((char)0x81),
            x_8a_u8x32 = __lasx_xvreplgr2vr_b((char)0x8A), x_a8_u8x32 = __lasx_xvreplgr2vr_b((char)0xA8),
            x_a9_u8x32 = __lasx_xvreplgr2vr_b((char)0xA9), x_af_u8x32 = __lasx_xvreplgr2vr_b((char)0xAF),
            x_9f_u8x32 = __lasx_xvreplgr2vr_b((char)0x9F);

    while (position + 32 <= length && count < matches_capacity) {
        __m256i window_u8x32 = __lasx_xvld(text_u8 + position, 0);

        // 1-byte: space and the range [\t, \r] = [9, 13], taken as the signed band 0x08 < b < 0x0E.
        __m256i tab_lower_bound_u8x32 = __lasx_xvslt_b(x_08_u8x32, window_u8x32);
        __m256i carriage_return_upper_bound_u8x32 = __lasx_xvslt_b(window_u8x32, x_0e_u8x32);
        __m256i one_byte_cmp_u8x32 = __lasx_xvor_v(
            __lasx_xvseq_b(window_u8x32, x_20_u8x32),
            __lasx_xvand_v(tab_lower_bound_u8x32, carriage_return_upper_bound_u8x32));
        sz_u32_t one_byte_mask = sz_xvmovemask_b_utf8_loongsonasx_(one_byte_cmp_u8x32);

        // 2-byte: C2 85 (NEL), C2 A0 (NBSP).
        sz_u32_t lead_c2_mask = sz_xvmovemask_b_utf8_loongsonasx_(__lasx_xvseq_b(window_u8x32, lead_c2_u8x32));
        sz_u32_t x_85_mask = sz_xvmovemask_b_utf8_loongsonasx_(__lasx_xvseq_b(window_u8x32, x_85_u8x32));
        sz_u32_t x_a0_mask = sz_xvmovemask_b_utf8_loongsonasx_(__lasx_xvseq_b(window_u8x32, x_a0_u8x32));
        sz_u32_t two_byte_starts = lead_c2_mask & ((x_85_mask | x_a0_mask) >> 1);

        // 3-byte: E1 9A 80 (ogham); E2 80 [80-8A]; E2 80 AF; E2 81 9F; E2 80 A8/A9; E3 80 80.
        sz_u32_t x_e1_mask = sz_xvmovemask_b_utf8_loongsonasx_(__lasx_xvseq_b(window_u8x32, x_e1_u8x32));
        sz_u32_t lead_e2_mask = sz_xvmovemask_b_utf8_loongsonasx_(__lasx_xvseq_b(window_u8x32, lead_e2_u8x32));
        sz_u32_t x_e3_mask = sz_xvmovemask_b_utf8_loongsonasx_(__lasx_xvseq_b(window_u8x32, x_e3_u8x32));
        sz_u32_t x_9a_mask = sz_xvmovemask_b_utf8_loongsonasx_(__lasx_xvseq_b(window_u8x32, x_9a_u8x32));
        sz_u32_t byte_80_mask = sz_xvmovemask_b_utf8_loongsonasx_(__lasx_xvseq_b(window_u8x32, byte_80_u8x32));
        sz_u32_t x_81_mask = sz_xvmovemask_b_utf8_loongsonasx_(__lasx_xvseq_b(window_u8x32, x_81_u8x32));
        sz_u32_t x_a8_mask = sz_xvmovemask_b_utf8_loongsonasx_(__lasx_xvseq_b(window_u8x32, x_a8_u8x32));
        sz_u32_t x_a9_mask = sz_xvmovemask_b_utf8_loongsonasx_(__lasx_xvseq_b(window_u8x32, x_a9_u8x32));
        sz_u32_t x_af_mask = sz_xvmovemask_b_utf8_loongsonasx_(__lasx_xvseq_b(window_u8x32, x_af_u8x32));
        sz_u32_t x_9f_mask = sz_xvmovemask_b_utf8_loongsonasx_(__lasx_xvseq_b(window_u8x32, x_9f_u8x32));
        // [0x80, 0x8A] range: unsigned `b >= 0x80` and `b <= 0x8A`.
        __m256i x_80_ge_cmp_u8x32 = __lasx_xvsle_bu(byte_80_u8x32, window_u8x32);
        __m256i x_8a_le_cmp_u8x32 = __lasx_xvsle_bu(window_u8x32, x_8a_u8x32);
        sz_u32_t x_8a_range_mask = sz_xvmovemask_b_utf8_loongsonasx_(
            __lasx_xvand_v(x_80_ge_cmp_u8x32, x_8a_le_cmp_u8x32));

        sz_u32_t lead_e280_mask = lead_e2_mask & (byte_80_mask >> 1);                      // E2 80
        sz_u32_t ogham_mask = x_e1_mask & (x_9a_mask >> 1) & (byte_80_mask >> 2);          // E1 9A 80
        sz_u32_t range_e280_mask = lead_e280_mask & (x_8a_range_mask >> 2);                // E2 80 [80-8A]
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
            sz_utf8_iterate_peel_loongsonasx_(start_bits, two_byte_starts, three_byte_starts, emit_count, position,
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
    sz_assert_(sz_utf8_batch_consistent_(length, matches_capacity, count, bytes_consumed ? *bytes_consumed : length,
                                         match_offsets, match_lengths));
    return count;
}

STRINGZILLA_API sz_status_t sz_utf8_whitespaces_loongsonasx(                        //
    sz_cptr_t text, sz_size_t length,                                               //
    sz_size_t *match_offsets, sz_size_t *match_lengths, sz_size_t matches_capacity, //
    sz_size_t *matches_count, sz_size_t *bytes_consumed, sz_stream_t stream) {
    sz_assert_(stream == STRINGZILLA_NULL);
    *matches_count = sz_utf8_whitespaces_loongsonasx_(text, length, match_offsets, match_lengths, matches_capacity,
                                                      bytes_consumed);
    return sz_success_k;
}

#pragma endregion Multistep newline and whitespace iteration

#pragma region Membership

/** Per-lane lookup into a 32-byte table at @p table indexed below 32, as NEON's @c vqtbl2q_u8:
 *  @c xvshuf.b indexes within each 128-bit lane, so both table halves go to both lanes. */
STRINGZILLA_INLINE __m256i sz_delimiter_lookup32_loongsonasx_(sz_u8_t const *table, __m256i index_u8x32) {
    __m256i const table_u8x32 = __lasx_xvld(table, 0);
    __m256i const low_half_u8x32 = __lasx_xvpermi_q(table_u8x32, table_u8x32, 0x00);
    __m256i const high_half_u8x32 = __lasx_xvpermi_q(table_u8x32, table_u8x32, 0x11);
    return __lasx_xvshuf_b(high_half_u8x32, low_half_u8x32, index_u8x32);
}

/** Per-lane single-bit test `(bitmap_byte >> (low & 7)) & 1`, as 0x00/0xFF lanes. */
STRINGZILLA_INLINE __m256i sz_delimiter_test_bit_loongsonasx_(__m256i bitmap_byte_u8x32, __m256i low_u8x32) {
    static sz_u8_t const bit_for_low3[32] = {1, 2, 4, 8, 16, 32, 64, 128, 0, 0, 0, 0, 0, 0, 0, 0,
                                             1, 2, 4, 8, 16, 32, 64, 128, 0, 0, 0, 0, 0, 0, 0, 0};
    __m256i const bit_table_u8x32 = __lasx_xvld(bit_for_low3, 0);
    __m256i const bit_mask_u8x32 = __lasx_xvshuf_b(bit_table_u8x32, bit_table_u8x32,
                                                   __lasx_xvand_v(low_u8x32, __lasx_xvreplgr2vr_b(0x07)));
    __m256i const is_clear_u8x32 = __lasx_xvseq_b(__lasx_xvand_v(bitmap_byte_u8x32, bit_mask_u8x32),
                                                  __lasx_xvreplgr2vr_b(0));
    return __lasx_xvnor_v(is_clear_u8x32, is_clear_u8x32);
}

/** BMP (codepoint < 0x10000) delimiter membership for one 32-byte half, as 0x00/0xFF lanes.
 *  Lanes below U+0100 read bitmap row 0; the rest survive a pre-filter over @p high_in_u8x32
 *  and resolve one distinct high byte at a time, each round settling every lane carrying it. */
STRINGZILLA_INLINE __m256i sz_delimiter_bmp_membership_loongsonasx_(__m256i window_u8x32, __m256i high_in_u8x32,
                                                                    __m256i low_in_u8x32) {
    __m256i const is_ascii_u8x32 = __lasx_xvslt_bu(window_u8x32, __lasx_xvreplgr2vr_b((char)0x80));
    __m256i const high_u8x32 = __lasx_xvandn_v(is_ascii_u8x32, high_in_u8x32);
    __m256i const low_u8x32 = __lasx_xvbitsel_v(low_in_u8x32, window_u8x32, is_ascii_u8x32);

    __m256i const high_is_zero_u8x32 = __lasx_xvseq_b(high_u8x32, __lasx_xvreplgr2vr_b(0));
    __m256i const row0_byte_u8x32 = sz_delimiter_lookup32_loongsonasx_(sz_utf8_delimiter_bmp_bitmaps_,
                                                                       __lasx_xvsrli_b(low_u8x32, 3));
    __m256i result_u8x32 = __lasx_xvand_v(sz_delimiter_test_bit_loongsonasx_(row0_byte_u8x32, low_u8x32),
                                          high_is_zero_u8x32);

    __m256i const suspicious_byte_u8x32 = sz_delimiter_lookup32_loongsonasx_(sz_utf8_delimiter_bmp_suspicious_highs_,
                                                                             __lasx_xvsrli_b(high_u8x32, 3));
    __m256i const is_continuation_u8x32 = __lasx_xvseq_b(__lasx_xvand_v(window_u8x32, __lasx_xvreplgr2vr_b((char)0xC0)),
                                                         __lasx_xvreplgr2vr_b((char)0x80));
    __m256i unresolved_u8x32 = __lasx_xvandn_v(__lasx_xvor_v(high_is_zero_u8x32, is_continuation_u8x32),
                                               sz_delimiter_test_bit_loongsonasx_(suspicious_byte_u8x32, high_u8x32));
    sz_u256_vec_t high_vec;
    high_vec.lasx = high_u8x32;
    for (sz_u32_t unresolved_bits = sz_xvmovemask_b_utf8_loongsonasx_(unresolved_u8x32); unresolved_bits;
         unresolved_bits = sz_xvmovemask_b_utf8_loongsonasx_(unresolved_u8x32)) {
        sz_u8_t const shared_high = high_vec.u8s[sz_u32_ctz(unresolved_bits)];
        __m256i const same_u8x32 = __lasx_xvand_v(unresolved_u8x32,
                                                  __lasx_xvseq_b(high_u8x32, __lasx_xvreplgr2vr_b((char)shared_high)));
        __m256i const row_byte_u8x32 = sz_delimiter_lookup32_loongsonasx_(
            sz_utf8_delimiter_bmp_bitmaps_ + (sz_size_t)sz_utf8_delimiter_bmp_block_[shared_high] * 32,
            __lasx_xvsrli_b(low_u8x32, 3));
        result_u8x32 = __lasx_xvor_v(
            result_u8x32, __lasx_xvand_v(sz_delimiter_test_bit_loongsonasx_(row_byte_u8x32, low_u8x32), same_u8x32));
        unresolved_u8x32 = __lasx_xvandn_v(same_u8x32, unresolved_u8x32);
    }
    return result_u8x32;
}

/** Astral (codepoint >= 0x10000) delimiter membership for one 32-byte half, as 0x00/0xFF lanes,
 *  over the byte-domain parts of @c offset=cp-0x10000, as the NEON twin does: plane selects an L1
 *  group, group and sub byte select a bitmap row, and @c low8&7 is the bit. */
STRINGZILLA_INLINE __m256i sz_delimiter_astral_membership_loongsonasx_(__m256i window_u8x32, __m256i next1_u8x32,
                                                                       __m256i next2_u8x32, __m256i next3_u8x32) {
    __m256i const b0_u8x32 = __lasx_xvand_v(window_u8x32, __lasx_xvreplgr2vr_b(0x07));
    __m256i const b1_u8x32 = __lasx_xvand_v(next1_u8x32, __lasx_xvreplgr2vr_b(0x3F));
    __m256i const b2_u8x32 = __lasx_xvand_v(next2_u8x32, __lasx_xvreplgr2vr_b(0x3F));
    __m256i const b3_u8x32 = __lasx_xvand_v(next3_u8x32, __lasx_xvreplgr2vr_b(0x3F));

    __m256i const codepoint_high_u8x32 = __lasx_xvor_v(__lasx_xvslli_b(b0_u8x32, 2), __lasx_xvsrli_b(b1_u8x32, 4));
    __m256i const sub_u8x32 = __lasx_xvor_v(__lasx_xvslli_b(b1_u8x32, 4), __lasx_xvsrli_b(b2_u8x32, 2));
    __m256i const low8_u8x32 = __lasx_xvor_v(__lasx_xvslli_b(b2_u8x32, 6), b3_u8x32);

    // Only planes 1..16 are addressable; a lane with a zero or over 0x10 high part is an invalid
    // lead that the caller's validity mask rejects.
    __m256i const codepoint_high_is_zero_u8x32 = __lasx_xvseq_b(codepoint_high_u8x32, __lasx_xvreplgr2vr_b(0));
    __m256i remaining_u8x32 = __lasx_xvandn_v(
        codepoint_high_is_zero_u8x32,
        __lasx_xvand_v(__lasx_xvsle_bu(__lasx_xvreplgr2vr_b((char)0xF0), window_u8x32),
                       __lasx_xvsle_bu(codepoint_high_u8x32, __lasx_xvreplgr2vr_b(0x10))));
    __m256i result_u8x32 = __lasx_xvreplgr2vr_b(0);
    sz_u256_vec_t codepoint_high_vec, sub_vec;
    codepoint_high_vec.lasx = codepoint_high_u8x32;
    sub_vec.lasx = sub_u8x32;
    for (sz_u32_t remaining_bits = sz_xvmovemask_b_utf8_loongsonasx_(remaining_u8x32); remaining_bits;
         remaining_bits = sz_xvmovemask_b_utf8_loongsonasx_(remaining_u8x32)) {
        sz_u32_t const first_lane = sz_u32_ctz(remaining_bits);
        sz_u8_t const shared_plane = codepoint_high_vec.u8s[first_lane], shared_sub = sub_vec.u8s[first_lane];
        sz_u8_t const *row =
            sz_utf8_delimiter_astral_bitmaps_ +
            (sz_size_t)sz_utf8_delimiter_astral_l2_[(sz_size_t)sz_utf8_delimiter_astral_l1_[shared_plane - 1] * 256 +
                                                    shared_sub] *
                32;
        __m256i const same_u8x32 = __lasx_xvand_v(
            remaining_u8x32,
            __lasx_xvand_v(__lasx_xvseq_b(codepoint_high_u8x32, __lasx_xvreplgr2vr_b((char)shared_plane)),
                           __lasx_xvseq_b(sub_u8x32, __lasx_xvreplgr2vr_b((char)shared_sub))));
        __m256i const row_byte_u8x32 = sz_delimiter_lookup32_loongsonasx_(row, __lasx_xvsrli_b(low8_u8x32, 3));
        result_u8x32 = __lasx_xvor_v(
            result_u8x32, __lasx_xvand_v(sz_delimiter_test_bit_loongsonasx_(row_byte_u8x32, low8_u8x32), same_u8x32));
        remaining_u8x32 = __lasx_xvandn_v(same_u8x32, remaining_u8x32);
    }
    return result_u8x32;
}

/** Per-lane UTF-8 validity for the 64 lanes of a decoded window, mirroring @ref sz_rune_decode: a
 *  2/3/4-byte lead is valid only with well-formed continuations and no overlong, surrogate or
 *  beyond-U+10FFFF form. A lead whose span runs past the loaded bytes is never valid. */
STRINGZILLA_INLINE sz_u64_t sz_delimiter_valid_starts_loongsonasx_(sz_utf8_rune_window_loongsonasx_t const *decoded,
                                                                   __m256i const *next1_u8x32,
                                                                   __m256i const *next2_u8x32,
                                                                   __m256i const *next3_u8x32) {
    __m256i const continuation_mask_u8x32 = __lasx_xvreplgr2vr_b((char)0xC0);
    __m256i const continuation_pattern_u8x32 = __lasx_xvreplgr2vr_b((char)0x80);
    __m256i const lead_c0_u8x32 = __lasx_xvreplgr2vr_b((char)0xC0), lead_e0_u8x32 = __lasx_xvreplgr2vr_b((char)0xE0),
                  lead_f0_u8x32 = __lasx_xvreplgr2vr_b((char)0xF0);
    __m256i valid_bool_u8x32[2];
    for (int half = 0; half < 2; ++half) {
        __m256i const here_u8x32 = half ? decoded->window_high_u8x32 : decoded->window_low_u8x32;
        __m256i const n1_u8x32 = next1_u8x32[half];
        __m256i const c1_ok_u8x32 = __lasx_xvseq_b(__lasx_xvand_v(n1_u8x32, continuation_mask_u8x32),
                                                   continuation_pattern_u8x32);
        __m256i const c2_ok_u8x32 = __lasx_xvseq_b(__lasx_xvand_v(next2_u8x32[half], continuation_mask_u8x32),
                                                   continuation_pattern_u8x32);
        __m256i const c3_ok_u8x32 = __lasx_xvseq_b(__lasx_xvand_v(next3_u8x32[half], continuation_mask_u8x32),
                                                   continuation_pattern_u8x32);
        __m256i const ascii_u8x32 = __lasx_xvslt_bu(here_u8x32, continuation_pattern_u8x32);

        __m256i const is_two_u8x32 = __lasx_xvand_v(__lasx_xvsle_bu(lead_c0_u8x32, here_u8x32),
                                                    __lasx_xvslt_bu(here_u8x32, lead_e0_u8x32));
        __m256i const is_three_u8x32 = __lasx_xvand_v(__lasx_xvsle_bu(lead_e0_u8x32, here_u8x32),
                                                      __lasx_xvslt_bu(here_u8x32, lead_f0_u8x32));
        __m256i const is_four_u8x32 = __lasx_xvsle_bu(lead_f0_u8x32, here_u8x32);

        __m256i const two_ok_u8x32 = __lasx_xvand_v(c1_ok_u8x32,
                                                    __lasx_xvsle_bu(__lasx_xvreplgr2vr_b((char)0xC2), here_u8x32));

        __m256i const lead_e0_match_u8x32 = __lasx_xvseq_b(here_u8x32, lead_e0_u8x32);
        __m256i const lead_ed_u8x32 = __lasx_xvseq_b(here_u8x32, __lasx_xvreplgr2vr_b((char)0xED));
        __m256i const n1_lt_a0_u8x32 = __lasx_xvslt_bu(n1_u8x32, __lasx_xvreplgr2vr_b((char)0xA0));
        __m256i const bad_three_u8x32 = __lasx_xvor_v(__lasx_xvand_v(lead_e0_match_u8x32, n1_lt_a0_u8x32),
                                                      __lasx_xvandn_v(n1_lt_a0_u8x32, lead_ed_u8x32));
        __m256i const three_ok_u8x32 = __lasx_xvandn_v(bad_three_u8x32, __lasx_xvand_v(c1_ok_u8x32, c2_ok_u8x32));

        __m256i const lead_f0_match_u8x32 = __lasx_xvseq_b(here_u8x32, lead_f0_u8x32);
        __m256i const lead_f4_u8x32 = __lasx_xvseq_b(here_u8x32, __lasx_xvreplgr2vr_b((char)0xF4));
        __m256i const n1_lt_90_u8x32 = __lasx_xvslt_bu(n1_u8x32, __lasx_xvreplgr2vr_b((char)0x90));
        __m256i const bad_four_u8x32 = __lasx_xvor_v(__lasx_xvslt_bu(__lasx_xvreplgr2vr_b((char)0xF4), here_u8x32),
                                                     __lasx_xvor_v(__lasx_xvand_v(lead_f0_match_u8x32, n1_lt_90_u8x32),
                                                                   __lasx_xvandn_v(n1_lt_90_u8x32, lead_f4_u8x32)));
        __m256i const four_ok_u8x32 = __lasx_xvandn_v(
            bad_four_u8x32, __lasx_xvand_v(__lasx_xvand_v(c1_ok_u8x32, c2_ok_u8x32), c3_ok_u8x32));

        valid_bool_u8x32[half] = __lasx_xvor_v(__lasx_xvor_v(ascii_u8x32, __lasx_xvand_v(is_two_u8x32, two_ok_u8x32)),
                                               __lasx_xvor_v(__lasx_xvand_v(is_three_u8x32, three_ok_u8x32),
                                                             __lasx_xvand_v(is_four_u8x32, four_ok_u8x32)));
    }
    sz_u64_t const valid = sz_utf8_mask_combine_loongsonasx_(valid_bool_u8x32[0], valid_bool_u8x32[1]);

    sz_size_t const loaded = decoded->loaded;
    sz_u64_t const truncated = (decoded->two_byte_starts & ~sz_u64_mask_until_serial_(loaded >= 1 ? loaded - 1 : 0)) |
                               (decoded->three_byte_starts & ~sz_u64_mask_until_serial_(loaded >= 2 ? loaded - 2 : 0)) |
                               (decoded->four_byte_starts & ~sz_u64_mask_until_serial_(loaded >= 3 ? loaded - 3 : 0));
    return valid & sz_u64_mask_until_serial_(loaded) & ~truncated;
}

#pragma endregion Membership

#pragma region Forward driver

STRINGZILLA_INLINE sz_size_t sz_utf8_delimiters_loongsonasx_( //
    sz_cptr_t text, sz_size_t length,                         //
    sz_size_t *match_offsets, sz_size_t *match_lengths,       //
    sz_size_t matches_capacity, sz_size_t *bytes_consumed) {
    sz_u8_t const *const text_u8 = (sz_u8_t const *)text;

    sz_size_t base = 0, count = 0;
    while (base < length && count < matches_capacity) {
        sz_utf8_rune_window_loongsonasx_t const decoded = sz_utf8_rune_decode_window_loongsonasx_(text_u8 + base,
                                                                                                  length - base);
        sz_size_t const loaded = decoded.loaded;
        sz_u64_t const loaded_mask = sz_u64_mask_until_serial_(loaded);

        sz_size_t byte_span = loaded;
        sz_u64_t hits;

        // All-ASCII window: one lookup over the first half of bitmap row 0 decides every lane.
        int const all_ascii = decoded.codepoint_starts == loaded_mask &&
                              !(decoded.two_byte_starts | decoded.three_byte_starts | decoded.four_byte_starts);
        if (all_ascii) {
            __m256i const low_member_u8x32 = sz_delimiter_test_bit_loongsonasx_(
                sz_delimiter_lookup32_loongsonasx_(sz_utf8_delimiter_bmp_bitmaps_,
                                                   __lasx_xvsrli_b(decoded.window_low_u8x32, 3)),
                decoded.window_low_u8x32);
            __m256i const high_member_u8x32 = sz_delimiter_test_bit_loongsonasx_(
                sz_delimiter_lookup32_loongsonasx_(sz_utf8_delimiter_bmp_bitmaps_,
                                                   __lasx_xvsrli_b(decoded.window_high_u8x32, 3)),
                decoded.window_high_u8x32);
            hits = sz_utf8_mask_combine_loongsonasx_(low_member_u8x32, high_member_u8x32) & loaded_mask;
        }
        else {
            __m256i next1_u8x32[2], next2_u8x32[2], next3_u8x32[2];
            sz_utf8_forward_neighbours_loongsonasx_(decoded.window_low_u8x32, decoded.window_high_u8x32,
                                                    &next1_u8x32[0], &next1_u8x32[1], &next2_u8x32[0], &next2_u8x32[1],
                                                    &next3_u8x32[0], &next3_u8x32[1]);

            // A multi-byte lead near the 64-byte edge whose span runs past `loaded`
            // defers to the next window.
            if (loaded >= 64)
                byte_span = sz_utf8_delimiter_complete_span_(decoded.two_byte_starts, decoded.three_byte_starts,
                                                             decoded.four_byte_starts, loaded);
            sz_u64_t const span_mask = sz_u64_mask_until_serial_(byte_span);

            sz_u64_t const valid_starts = sz_delimiter_valid_starts_loongsonasx_(&decoded, next1_u8x32, next2_u8x32,
                                                                                 next3_u8x32) &
                                          decoded.codepoint_starts & span_mask;

            sz_u64_t const four_byte = decoded.four_byte_starts & span_mask;
            sz_u64_t member = sz_utf8_mask_combine_loongsonasx_(
                sz_delimiter_bmp_membership_loongsonasx_(decoded.window_low_u8x32, decoded.high_byte_low_u8x32,
                                                         decoded.low_byte_low_u8x32),
                sz_delimiter_bmp_membership_loongsonasx_(decoded.window_high_u8x32, decoded.high_byte_high_u8x32,
                                                         decoded.low_byte_high_u8x32));
            if (four_byte) {
                sz_u64_t const astral_member = sz_utf8_mask_combine_loongsonasx_(
                    sz_delimiter_astral_membership_loongsonasx_(decoded.window_low_u8x32, next1_u8x32[0],
                                                                next2_u8x32[0], next3_u8x32[0]),
                    sz_delimiter_astral_membership_loongsonasx_(decoded.window_high_u8x32, next1_u8x32[1],
                                                                next2_u8x32[1], next3_u8x32[1]));
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

STRINGZILLA_API sz_status_t sz_utf8_delimiters_loongsonasx(                         //
    sz_cptr_t text, sz_size_t length,                                               //
    sz_size_t *match_offsets, sz_size_t *match_lengths, sz_size_t matches_capacity, //
    sz_size_t *matches_count, sz_size_t *bytes_consumed, sz_stream_t stream) {
    sz_assert_(stream == STRINGZILLA_NULL);
    *matches_count = sz_utf8_delimiters_loongsonasx_(text, length, match_offsets, match_lengths, matches_capacity,
                                                     bytes_consumed);
    sz_assert_(sz_utf8_batch_consistent_(length, matches_capacity, *matches_count,
                                         bytes_consumed ? *bytes_consumed : length, match_offsets, match_lengths));
    return sz_success_k;
}

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#endif // STRINGZILLA_TARGET_LOONGSONASX
#endif // STRINGZILLA_ARCH_LOONGARCH64_

#ifdef __cplusplus
}
#endif

#endif // STRINGZILLA_UTF8_TOKENS_LOONGSONASX_H_
