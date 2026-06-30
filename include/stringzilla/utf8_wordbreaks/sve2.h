/**
 *  @brief SVE2 backend for UAX-29 word boundaries.
 *  @file include/stringzilla/utf8_wordbreaks/sve2.h
 *  @author Ash Vardanian
 */
#ifndef STRINGZILLA_UTF8_WORDS_SVE2_H_
#define STRINGZILLA_UTF8_WORDS_SVE2_H_

#include "stringzilla/types.h"
#include "stringzilla/utf8_wordbreaks/tables.h"
#include "stringzilla/utf8_wordbreaks/serial.h"
#include "stringzilla/utf8_runes/sve2.h"

#ifdef __cplusplus
extern "C" {
#endif

#if SZ_USE_SVE2
#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("+sve+sve2"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("+sve+sve2")
#endif

/** @brief  Classify an all-ASCII window to `sz_utf8_word_break_t` properties via eight 16-entry `svtbl_u8` rows. */
SZ_HELPER_INLINE svuint8_t sz_utf8_word_break_classify_ascii_sve2_(svbool_t pg, svuint8_t bytes) {
    // Eight rows of the ASCII Word_Break property table (high nibble → 16 low-nibble entries); a single `svtbl_u8`
    // reads 16 entries from lanes [0,16), so each row is broadcast with `svdupq_n_u8` and indexed by the low nibble.
    svuint8_t const row0 = svdupq_n_u8(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0x02, 0x03, 0x03, 0x01, 0, 0);
    svuint8_t const row1 = svdup_n_u8(0);
    svuint8_t const row2 = svdupq_n_u8(0, 0, 0x0F, 0, 0, 0, 0, 0x0F, 0, 0, 0, 0, 0x0E, 0, 0x0F, 0);
    svuint8_t const row3 = svdupq_n_u8(0x0A, 0x0A, 0x0A, 0x0A, 0x0A, 0x0A, 0x0A, 0x0A, 0x0A, 0x0A, 0x0D, 0x0E, 0, 0, 0,
                                       0);
    svuint8_t const row4 = svdupq_n_u8(0, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08,
                                       0x08, 0x08);
    svuint8_t const row5 = svdupq_n_u8(0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0, 0, 0, 0,
                                       0x0C);
    svuint8_t const row6 = svdupq_n_u8(0, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08,
                                       0x08, 0x08);
    svuint8_t const row7 = svdupq_n_u8(0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0, 0, 0, 0, 0);

    svuint8_t const low_nibble = svand_n_u8_x(pg, bytes, 0x0F);
    svuint8_t const high_nibble = svlsr_n_u8_x(pg, bytes, 4);
    svuint8_t classes = svtbl_u8(row0, low_nibble);
    classes = svsel_u8(svcmpeq_n_u8(pg, high_nibble, 1), svtbl_u8(row1, low_nibble), classes);
    classes = svsel_u8(svcmpeq_n_u8(pg, high_nibble, 2), svtbl_u8(row2, low_nibble), classes);
    classes = svsel_u8(svcmpeq_n_u8(pg, high_nibble, 3), svtbl_u8(row3, low_nibble), classes);
    classes = svsel_u8(svcmpeq_n_u8(pg, high_nibble, 4), svtbl_u8(row4, low_nibble), classes);
    classes = svsel_u8(svcmpeq_n_u8(pg, high_nibble, 5), svtbl_u8(row5, low_nibble), classes);
    classes = svsel_u8(svcmpeq_n_u8(pg, high_nibble, 6), svtbl_u8(row6, low_nibble), classes);
    classes = svsel_u8(svcmpeq_n_u8(pg, high_nibble, 7), svtbl_u8(row7, low_nibble), classes);
    return classes;
}

/** @brief  Vector replay of `sz_utf8_word_break_join_from_class_masks_`: lane `i` is 1 where the boundary before
 *          it is suppressed by a UAX-29 no-break rule. `<< k` becomes a `k`-lane up-shift (`svinsr`), `>> 1` a
 *          one-lane down-shift (`svext`). Intentional divergence: a scalable SVE register exceeds 64 bits at
 *          VLEN >= 1024, so there is no `sz_u64_t` movemask to shift — the cascade runs on per-lane 0/1 vectors. */
SZ_HELPER_AUTO svuint8_t sz_utf8_word_break_join_lanes_sve2_(svbool_t pg, svuint8_t classes) {
    svuint8_t const aletter = svdup_u8_z(svcmpeq_n_u8(pg, classes, sz_utf8_word_break_aletter_k), 1);
    svuint8_t const numeric = svdup_u8_z(svcmpeq_n_u8(pg, classes, sz_utf8_word_break_numeric_k), 1);
    svuint8_t const extendnumlet = svdup_u8_z(svcmpeq_n_u8(pg, classes, sz_utf8_word_break_extendnumlet_k), 1);
    svuint8_t const midletter = svdup_u8_z(svcmpeq_n_u8(pg, classes, sz_utf8_word_break_midletter_k), 1);
    svuint8_t const midnum = svdup_u8_z(svcmpeq_n_u8(pg, classes, sz_utf8_word_break_midnum_k), 1);
    svuint8_t const mid_quotes = svdup_u8_z(svcmpeq_n_u8(pg, classes, sz_utf8_word_break_mid_quotes_k), 1);
    svuint8_t const carriage_return = svdup_u8_z(svcmpeq_n_u8(pg, classes, sz_utf8_word_break_cr_k), 1);
    svuint8_t const line_feed = svdup_u8_z(svcmpeq_n_u8(pg, classes, sz_utf8_word_break_lf_k), 1);
    svuint8_t const zeros = svdup_n_u8(0);

    svuint8_t const mid_letter_or_quotes = svorr_u8_x(pg, midletter, mid_quotes);
    svuint8_t const mid_num_or_quotes = svorr_u8_x(pg, midnum, mid_quotes);
    svuint8_t const aletter_or_numeric = svorr_u8_x(pg, aletter, numeric);

    // `prev_1` lane i holds lane i-1 (bitmask `<< 1`); `prev_2` holds i-2 (`<< 2`); `next_1` holds i+1 (`>> 1`).
    svuint8_t const aletter_prev_1 = svinsr_n_u8(aletter, 0);
    svuint8_t const aletter_prev_2 = svinsr_n_u8(aletter_prev_1, 0);
    svuint8_t const aletter_next_1 = svext_u8(aletter, zeros, 1);
    svuint8_t const numeric_prev_1 = svinsr_n_u8(numeric, 0);
    svuint8_t const numeric_prev_2 = svinsr_n_u8(numeric_prev_1, 0);
    svuint8_t const numeric_next_1 = svext_u8(numeric, zeros, 1);
    svuint8_t const mid_letter_or_quotes_prev_1 = svinsr_n_u8(mid_letter_or_quotes, 0);
    svuint8_t const mid_num_or_quotes_prev_1 = svinsr_n_u8(mid_num_or_quotes, 0);
    svuint8_t const extendnumlet_prev_1 = svinsr_n_u8(extendnumlet, 0);
    svuint8_t const aletter_or_numeric_or_extendnumlet_prev_1 = svinsr_n_u8(
        svorr_u8_x(pg, aletter_or_numeric, extendnumlet), 0);
    svuint8_t const carriage_return_prev_1 = svinsr_n_u8(carriage_return, 0);

    svuint8_t join = svand_u8_x(pg, carriage_return_prev_1, line_feed);   // WB3
    join = svorr_u8_x(pg, join, svand_u8_x(pg, aletter_prev_1, aletter)); // WB5
    join = svorr_u8_x(pg, join,
                      svand_u8_x(pg, svand_u8_x(pg, aletter_prev_1, mid_letter_or_quotes), aletter_next_1)); // WB6
    join = svorr_u8_x(pg, join,
                      svand_u8_x(pg, svand_u8_x(pg, aletter_prev_2, mid_letter_or_quotes_prev_1), aletter)); // WB7
    join = svorr_u8_x(pg, join, svand_u8_x(pg, numeric_prev_1, numeric));                                    // WB8
    join = svorr_u8_x(pg, join, svand_u8_x(pg, aletter_prev_1, numeric));                                    // WB9
    join = svorr_u8_x(pg, join, svand_u8_x(pg, numeric_prev_1, aletter));                                    // WB10
    join = svorr_u8_x(pg, join,
                      svand_u8_x(pg, svand_u8_x(pg, numeric_prev_1, mid_num_or_quotes), numeric_next_1)); // WB11
    join = svorr_u8_x(pg, join,
                      svand_u8_x(pg, svand_u8_x(pg, numeric_prev_2, mid_num_or_quotes_prev_1), numeric)); // WB12
    join = svorr_u8_x(pg, join, svand_u8_x(pg, aletter_or_numeric_or_extendnumlet_prev_1, extendnumlet)); // WB13a
    join = svorr_u8_x(pg, join, svand_u8_x(pg, extendnumlet_prev_1, aletter_or_numeric));                 // WB13b
    return join;
}

/** @brief  Compact the absolute byte positions of the set lanes of `boundary` (over lanes `[0, usable)`) into
 *          `out`, walking `svcntw()`-lane sub-blocks with `svcompact_u32`. Returns the number of positions. */
SZ_HELPER_AUTO sz_size_t sz_utf8_word_compact_boundaries_sve2_(svbool_t boundary, sz_size_t window_base,
                                                               sz_size_t usable, sz_u32_t *out) {
    svuint8_t const boundary_flags = svdup_u8_z(boundary, 1);
    svuint8_t const lane_iota = svindex_u8(0, 1);
    sz_size_t const words = svcntw();
    sz_size_t produced = 0;
    for (sz_size_t base = 0; base < usable; base += words) {
        svbool_t const pw = svwhilelt_b32_u64((sz_u64_t)base, (sz_u64_t)usable);
        svuint8_t const gather_indices = svadd_n_u8_x(svptrue_b8(), lane_iota, (sz_u8_t)base);
        svuint32_t const flags_w = svunpklo_u32(svunpklo_u16(svtbl_u8(boundary_flags, gather_indices)));
        svbool_t const sub_boundary = svcmpne_n_u32(pw, flags_w, 0);
        if (!svptest_any(pw, sub_boundary)) continue;
        svuint32_t const positions = svindex_u32((sz_u32_t)(window_base + base), 1);
        svuint32_t const packed = svcompact_u32(sub_boundary, positions);
        sz_size_t const here = svcntp_b32(pw, sub_boundary);
        svst1_u32(svwhilelt_b32_u64(0, (sz_u64_t)here), out + produced, packed);
        produced += here;
    }
    return produced;
}

SZ_API_COMPTIME sz_size_t sz_utf8_wordbreaks_sve2(   //
    sz_cptr_t text, sz_size_t length,                //
    sz_size_t *word_starts, sz_size_t *word_lengths, //
    sz_size_t words_capacity, sz_size_t *bytes_consumed) {

    sz_size_t words = 0;
    if (length == 0 || words_capacity == 0) {
        if (bytes_consumed) *bytes_consumed = 0;
        return 0;
    }

    sz_size_t const window_bytes = svcntb();
    if (window_bytes < 16) // Too narrow to host the [2, W-2] trusted window: defer wholesale to serial.
        return sz_utf8_wordbreaks_serial(text, length, word_starts, word_lengths, words_capacity, bytes_consumed);

    sz_u8_t const *text_u8 = (sz_u8_t const *)text;
    sz_size_t word_start = 0; // Start of the word currently being accumulated (always a boundary).
    // Position 0 is always a boundary; the first reportable boundary is after the first codepoint.
    sz_size_t position = sz_utf8_lead_length_(text_u8[0]);
    // Sized to the 2048-bit SVE architectural maximum: one boundary per window byte, and `svcntb() <= 256`.
    sz_u32_t boundaries[256];

    // Oracle-free fast path: an all-ASCII window `[position-2, position-2+window_bytes)` resolves boundaries at
    // positions `[position, position+window_bytes-4]`, carrying the open `word_start` into the first emitted word.
    while (position < length) {
        sz_size_t const base = position - 2; // lane j = byte base+j; trusted lanes [2, window_bytes-2]
        sz_bool_t ascii_window = (sz_bool_t)(position >= 2 && position - 2 + window_bytes <= length);
        svbool_t const window_pg = svptrue_b8();
        svuint8_t window = svdup_n_u8(0);
        if (ascii_window) {
            window = svld1_u8(window_pg, text_u8 + base);
            ascii_window = (sz_bool_t)!svptest_any(window_pg, svcmpge_n_u8(window_pg, window, 0x80));
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
            position += sz_utf8_lead_length_(text_u8[position]);
            continue;
        }

        svuint8_t const classes = sz_utf8_word_break_classify_ascii_sve2_(window_pg, window);
        svuint8_t const join = sz_utf8_word_break_join_lanes_sve2_(window_pg, classes);
        // Boundaries are the complement of `join` over the trusted lanes [2, window_bytes-2].
        svbool_t const trusted = svand_b_z(window_pg, svcmpge_n_u8(window_pg, svindex_u8(0, 1), 2),
                                           svcmple_n_u8(window_pg, svindex_u8(0, 1), (sz_u8_t)(window_bytes - 2)));
        svbool_t const boundary = svand_b_z(window_pg, trusted, svcmpeq_n_u8(window_pg, join, 0));

        sz_size_t const found = sz_utf8_word_compact_boundaries_sve2_(boundary, base, window_bytes, boundaries);
        sz_size_t const emit = sz_min_of_two(found, words_capacity - words);
        for (sz_size_t i = 0; i < emit; ++i) {
            sz_size_t const next_boundary = boundaries[i];
            word_starts[words] = word_start, word_lengths[words] = next_boundary - word_start, ++words;
            word_start = next_boundary;
        }
        if (words == words_capacity) {
            if (bytes_consumed) *bytes_consumed = word_start;
            return words;
        }
        // Resolved [position, position+window_bytes-4]; if all boundaries fit, the next unresolved boundary is at
        // position+window_bytes-3, otherwise resume from the open word start to re-classify the dropped tail.
        position = emit < found ? word_start : position + window_bytes - 3;
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

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#endif // SZ_USE_SVE2

#ifdef __cplusplus
}
#endif

#endif // STRINGZILLA_UTF8_WORDS_SVE2_H_
