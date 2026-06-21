/**
 *  @brief NEON backend for UAX-29 word boundaries.
 *  @file include/stringzilla/utf8_words/neon.h
 *  @author Ash Vardanian
 */
#ifndef STRINGZILLA_UTF8_WORDS_NEON_H_
#define STRINGZILLA_UTF8_WORDS_NEON_H_

#include "stringzilla/types.h"
#include "stringzilla/utf8_words/tables.h"
#include "stringzilla/utf8_words/serial.h"
#include "stringzilla/utf8_codepoints/neon.h"

#ifdef __cplusplus
extern "C" {
#endif

#if SZ_USE_NEON
/*  Classify an all-ASCII 16-byte window into Word_Break property values via the 128-entry table: two `vqtbl4q_u8`
 *  cover entries 0..63 and 64..127 (zero outside range), combined with an OR. Valid only when every byte is < 0x80. */
SZ_INTERNAL uint8x16_t sz_utf8_word_break_classify_property_neon_(uint8x16_t bytes) {
    uint8x16x4_t low_table = vld1q_u8_x4(sz_utf8_word_break_property_ascii_);         // entries 0..63
    uint8x16x4_t high_table = vld1q_u8_x4(sz_utf8_word_break_property_ascii_ + 64);   // entries 64..127
    uint8x16_t from_low = vqtbl4q_u8(low_table, bytes);                               // zero where byte >= 64
    uint8x16_t from_high = vqtbl4q_u8(high_table, veorq_u8(bytes, vdupq_n_u8(0x40))); // zero where byte < 64
    return vorrq_u8(from_low, from_high);
}

/*  Given an all-ASCII 16-byte window, return a nibble-mask (bit 4*j+3 set) of word boundaries at trusted lanes
 *  [2,14] - the lanes whose i-2 and i+1 neighbours are in-window - resolving every UAX-29 rule applicable to ASCII. */
SZ_INTERNAL sz_u64_t sz_utf8_word_break_boundary_mask_neon_(uint8x16_t window_bytes) {
    uint8x16_t classes = sz_utf8_word_break_classify_property_neon_(window_bytes);
    uint8x16_t aletter = vceqq_u8(classes, vdupq_n_u8(sz_utf8_word_break_aletter_k));
    uint8x16_t numeric = vceqq_u8(classes, vdupq_n_u8(sz_utf8_word_break_numeric_k));
    uint8x16_t extendnumlet = vceqq_u8(classes, vdupq_n_u8(sz_utf8_word_break_extendnumlet_k));
    uint8x16_t midletter = vceqq_u8(classes, vdupq_n_u8(sz_utf8_word_break_midletter_k));
    uint8x16_t midnum = vceqq_u8(classes, vdupq_n_u8(sz_utf8_word_break_midnum_k));
    uint8x16_t mid_quotes = vceqq_u8(classes, vdupq_n_u8(sz_utf8_word_break_mid_quotes_k));
    uint8x16_t carriage_return = vceqq_u8(classes, vdupq_n_u8(sz_utf8_word_break_cr_k));
    uint8x16_t line_feed = vceqq_u8(classes, vdupq_n_u8(sz_utf8_word_break_lf_k));
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
 *  @brief  Left-pack the @p submask -selected lanes of one 4-lane sub-block (positions as @p low = lanes 0,1 and
 *          @p high = lanes 2,3) to the front in ASCENDING lane order via a `vqtbl2q_u8` table.
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

SZ_PUBLIC sz_size_t sz_utf8_words_neon(              //
    sz_cptr_t text, sz_size_t length,                //
    sz_size_t *word_starts, sz_size_t *word_lengths, //
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

#endif // SZ_USE_NEON

#ifdef __cplusplus
}
#endif

#endif // STRINGZILLA_UTF8_WORDS_NEON_H_
