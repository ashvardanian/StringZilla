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

SZ_PUBLIC sz_cptr_t sz_utf8_find_newline_neon(sz_cptr_t text, sz_size_t length, sz_size_t *matched_length) {

    sz_u128_vec_t text_vec;
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

    uint8x16_t drop1_u8x16 = vsetq_lane_u8(0x00, vdupq_n_u8(0xFF), 15);
    uint8x16_t drop2_u8x16 = vsetq_lane_u8(0x00, drop1_u8x16, 14);

    while (length >= 16) {
        text_vec.u8x16 = vld1q_u8((sz_u8_t const *)text);

        // 1-byte matches
        uint8x16_t newline_cmp = vceqq_u8(text_vec.u8x16, newline_u8x16);
        uint8x16_t vertical_tab_cmp = vceqq_u8(text_vec.u8x16, vertical_tab_u8x16);
        uint8x16_t form_feed_cmp = vceqq_u8(text_vec.u8x16, form_feed_u8x16);
        uint8x16_t carriage_return_cmp = vandq_u8(vceqq_u8(text_vec.u8x16, carriage_return_u8x16),
                                                  drop1_u8x16); // Mask out \r at position 15
        uint8x16_t one_byte_cmp = vorrq_u8(vorrq_u8(newline_cmp, vertical_tab_cmp),
                                           vorrq_u8(form_feed_cmp, carriage_return_cmp));

        // 2- & 3-byte matches with shifted views
        uint8x16_t text1 = vextq_u8(text_vec.u8x16, text_vec.u8x16, 1);
        uint8x16_t text2 = vextq_u8(text_vec.u8x16, text_vec.u8x16, 2);
        uint8x16_t rn_match_u8x16 = vandq_u8(carriage_return_cmp, vceqq_u8(text1, newline_u8x16));
        uint8x16_t x_c285_u8x16 = vandq_u8(vceqq_u8(text_vec.u8x16, lead_c2_u8x16), vceqq_u8(text1, x_85_u8x16));
        uint8x16_t two_byte_cmp = vandq_u8(vorrq_u8(rn_match_u8x16, x_c285_u8x16),
                                           drop1_u8x16); // Ignore last split match

        uint8x16_t x_e280_cmp = vandq_u8(vceqq_u8(text_vec.u8x16, lead_e2_u8x16), vceqq_u8(text1, byte_80_u8x16));
        uint8x16_t x_e280ax_cmp = vandq_u8(x_e280_cmp,
                                           vorrq_u8(vceqq_u8(text2, x_a8_u8x16), vceqq_u8(text2, x_a9_u8x16)));
        uint8x16_t three_byte_cmp = vandq_u8(x_e280ax_cmp, drop2_u8x16); // Ignore last two split matches

        // Quick presence check
        uint8x16_t combined_vec = vorrq_u8(one_byte_cmp, vorrq_u8(two_byte_cmp, three_byte_cmp));
        if (vmaxvq_u8(combined_vec)) {

            // Late mask extraction only when a match exists
            sz_u64_t one_byte_mask = sz_utf8_vreinterpretq_u8_u4_neon_(one_byte_cmp);
            sz_u64_t two_mask = sz_utf8_vreinterpretq_u8_u4_neon_(two_byte_cmp);
            sz_u64_t three_mask = sz_utf8_vreinterpretq_u8_u4_neon_(three_byte_cmp);
            sz_u64_t combined_mask = one_byte_mask | two_mask | three_mask;

            int bit_index = sz_u64_ctz(combined_mask);
            sz_u64_t first_match_mask = (sz_u64_t)1 << bit_index;
            sz_size_t length_value = 1;
            length_value += (first_match_mask & (two_mask | three_mask)) != 0;
            length_value += (first_match_mask & three_mask) != 0;
            *matched_length = length_value;
            return text + (bit_index / 4);
        }
        text += 14;
        length -= 14;
    }

    return sz_utf8_find_newline_serial(text, length, matched_length);
}

SZ_PUBLIC sz_cptr_t sz_utf8_find_whitespace_neon(sz_cptr_t text, sz_size_t length, sz_size_t *matched_length) {

    sz_u128_vec_t text_vec;
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

    uint8x16_t drop1_u8x16 = vsetq_lane_u8(0x00, vdupq_n_u8(0xFF), 15);
    uint8x16_t drop2_u8x16 = vsetq_lane_u8(0x00, drop1_u8x16, 14);

    while (length >= 16) {
        text_vec.u8x16 = vld1q_u8((sz_u8_t const *)text);

        // 1-byte matches
        uint8x16_t x_20_cmp = vceqq_u8(text_vec.u8x16, x_20_u8x16);
        uint8x16_t range_cmp = vandq_u8(vcgeq_u8(text_vec.u8x16, tab_u8x16),
                                        vcleq_u8(text_vec.u8x16, carriage_return_u8x16));
        uint8x16_t one_byte_cmp = vorrq_u8(x_20_cmp, range_cmp);

        // 2-byte and 3-byte prefix indicators
        uint8x16_t lead_c2_cmp = vandq_u8(vceqq_u8(text_vec.u8x16, lead_c2_u8x16), drop1_u8x16);
        uint8x16_t x_e1_cmp = vandq_u8(vceqq_u8(text_vec.u8x16, x_e1_u8x16), drop2_u8x16);
        uint8x16_t lead_e2_cmp = vandq_u8(vceqq_u8(text_vec.u8x16, lead_e2_u8x16), drop2_u8x16);
        uint8x16_t x_e3_cmp = vandq_u8(vceqq_u8(text_vec.u8x16, x_e3_u8x16), drop2_u8x16);
        uint8x16_t prefix_byte_cmp = vorrq_u8(
            one_byte_cmp, vorrq_u8(vorrq_u8(lead_c2_cmp, x_e1_cmp), vorrq_u8(lead_e2_cmp, x_e3_cmp)));

        sz_u64_t one_byte_mask = sz_utf8_vreinterpretq_u8_u4_neon_(one_byte_cmp);
        sz_u64_t prefix_byte_mask = sz_utf8_vreinterpretq_u8_u4_neon_(prefix_byte_cmp);

        // Check for fast path - no whitespaces in this chunk
        if (!prefix_byte_mask) {
            text += 14;
            length -= 14;
            continue;
        }
        // Another simple common case - we have spotted a one-byte match before any prefix
        else if (one_byte_mask) {
            int first_one_byte_offset = sz_u64_ctz(one_byte_mask);
            int first_prefix_offset = sz_u64_ctz(prefix_byte_mask);
            if (first_one_byte_offset < first_prefix_offset) {
                *matched_length = 1;
                return text + (first_one_byte_offset / 4);
            }
        }

        // 2-byte matches
        uint8x16_t text1 = vextq_u8(text_vec.u8x16, text_vec.u8x16, 1);
        uint8x16_t two_u8x16 = vorrq_u8(vandq_u8(lead_c2_cmp, vceqq_u8(text1, x_85_u8x16)),
                                        vandq_u8(lead_c2_cmp, vceqq_u8(text1, x_a0_u8x16)));

        // 3-byte matches
        uint8x16_t text2 = vextq_u8(text_vec.u8x16, text_vec.u8x16, 2);
        uint8x16_t x_80_ge_cmp = vcgeq_u8(text2, byte_80_u8x16);
        uint8x16_t x_8d_le_cmp = vcleq_u8(text2, x_8d_u8x16);

        uint8x16_t ogham_cmp = vandq_u8(x_e1_cmp,
                                        vandq_u8(vceqq_u8(text1, x_9a_u8x16), vceqq_u8(text2, byte_80_u8x16)));
        uint8x16_t range_e280_cmp = vandq_u8(
            lead_e2_cmp, vandq_u8(vceqq_u8(text1, byte_80_u8x16), vandq_u8(x_80_ge_cmp, x_8d_le_cmp)));
        uint8x16_t line_cmp = vandq_u8(lead_e2_cmp,
                                       vandq_u8(vceqq_u8(text1, byte_80_u8x16), vceqq_u8(text2, x_a8_u8x16)));
        uint8x16_t paragraph_cmp = vandq_u8(lead_e2_cmp,
                                            vandq_u8(vceqq_u8(text1, byte_80_u8x16), vceqq_u8(text2, x_a9_u8x16)));
        uint8x16_t nnbsp_cmp = vandq_u8(lead_e2_cmp,
                                        vandq_u8(vceqq_u8(text1, byte_80_u8x16), vceqq_u8(text2, x_af_u8x16)));
        uint8x16_t mmsp_cmp = vandq_u8(lead_e2_cmp, vandq_u8(vceqq_u8(text1, x_81_u8x16), vceqq_u8(text2, x_9f_u8x16)));
        uint8x16_t ideographic_u8x16 = vandq_u8(
            x_e3_cmp, vandq_u8(vceqq_u8(text1, byte_80_u8x16), vceqq_u8(text2, byte_80_u8x16)));
        uint8x16_t three_u8x16 = vandq_u8(
            vorrq_u8(vorrq_u8(vorrq_u8(ogham_cmp, range_e280_cmp), vorrq_u8(line_cmp, paragraph_cmp)),
                     vorrq_u8(vorrq_u8(nnbsp_cmp, mmsp_cmp), ideographic_u8x16)),
            drop2_u8x16);

        sz_u64_t two_byte_mask = sz_utf8_vreinterpretq_u8_u4_neon_(two_u8x16);
        sz_u64_t three_byte_mask = sz_utf8_vreinterpretq_u8_u4_neon_(three_u8x16);
        sz_u64_t combined_mask = one_byte_mask | two_byte_mask | three_byte_mask;

        if (combined_mask) {
            int bit_index = sz_u64_ctz(combined_mask);
            sz_u64_t first_match_mask = (sz_u64_t)1 << bit_index;
            sz_size_t length_value = 1;
            length_value += (first_match_mask & (two_byte_mask | three_byte_mask)) != 0;
            length_value += (first_match_mask & three_byte_mask) != 0;
            *matched_length = length_value;
            return text + (bit_index / 4);
        }
        text += 14;
        length -= 14;
    }

    return sz_utf8_find_whitespace_serial(text, length, matched_length);
}

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
    sz_size_t position = sz_utf8_codepoint_length_(text_u8[0]);
    while (position < length) {
        // Oracle-free fast path: a window [position-2, position+14) gives lanes [2,14] full +/-2 context, so an
        // all-ASCII window resolves boundaries at positions [position, position+12] directly from the mask.
        if (position >= 2 && position + 14 <= length) {
            uint8x16_t window = vld1q_u8(text_u8 + position - 2);
            if (vmaxvq_u8(window) < 0x80) {
                sz_u64_t boundary_mask = sz_utf8_word_break_boundary_mask_neon_(window);
                while (boundary_mask) {
                    sz_size_t lane = (sz_size_t)(sz_u64_ctz(boundary_mask) >> 2);
                    sz_size_t boundary_position = position - 2 + lane;
                    if (words == words_capacity) {
                        if (bytes_consumed) *bytes_consumed = word_start;
                        return words;
                    }
                    word_starts[words] = word_start;
                    word_lengths[words] = boundary_position - word_start;
                    ++words;
                    word_start = boundary_position;
                    boundary_mask &= boundary_mask - 1;
                }
                position += 13; // Resolved [position, position+12]; next unresolved boundary is at position+13.
                continue;
            }
        }
        // Serial step for non-ASCII windows and the leading/trailing edges.
        if (sz_utf8_is_word_boundary_serial(text, length, position)) {
            if (words == words_capacity) {
                if (bytes_consumed) *bytes_consumed = word_start;
                return words;
            }
            word_starts[words] = word_start;
            word_lengths[words] = position - word_start;
            ++words;
            word_start = position;
        }
        position += sz_utf8_codepoint_length_(text_u8[position]);
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
    while (position > 0) {
        // Oracle-free fast path: a window [position-14, position+2) gives lanes [2,14] full +/-2 context,
        // resolving boundaries at positions [position-12, position]; emit them high-to-low.
        if (position >= 14 && position + 2 <= length) {
            uint8x16_t window = vld1q_u8(text_u8 + position - 14); // lane j = byte position-14+j; lane 14 = position
            if (vmaxvq_u8(window) < 0x80) {
                sz_u64_t boundary_mask = sz_utf8_word_break_boundary_mask_neon_(window);
                while (boundary_mask) {
                    sz_size_t bit = (sz_size_t)(63 - sz_u64_clz(boundary_mask)); // highest set lane first
                    sz_size_t boundary_position = position - 14 + (bit >> 2);
                    if (words == words_capacity) {
                        if (bytes_consumed) *bytes_consumed = word_end;
                        return words;
                    }
                    word_starts[words] = boundary_position;
                    word_lengths[words] = word_end - boundary_position;
                    ++words;
                    word_end = boundary_position;
                    boundary_mask &= ~((sz_u64_t)1 << bit);
                }
                position -= 13; // Resolved [position-12, position]; next unresolved boundary is at position-13.
                continue;
            }
        }
        if (sz_utf8_is_word_boundary_serial(text, length, position)) {
            if (words == words_capacity) {
                if (bytes_consumed) *bytes_consumed = word_end;
                return words;
            }
            word_starts[words] = position;
            word_lengths[words] = word_end - position;
            ++words;
            word_end = position;
        }
        position--;
        while (position > 0 && ((sz_u8_t)text[position] & 0xC0) == 0x80) position--;
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
