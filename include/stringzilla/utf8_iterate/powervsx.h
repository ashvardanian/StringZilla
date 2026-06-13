/**
 *  @brief IBM Power VSX backend for utf8.
 *  @file include/stringzilla/utf8_iterate/powervsx.h
 *  @author Ash Vardanian
 *  @sa utf8.h
 */
#ifndef STRINGZILLA_UTF8_ITERATE_POWERVSX_H_
#define STRINGZILLA_UTF8_ITERATE_POWERVSX_H_

#include "stringzilla/types.h"
#include "stringzilla/utf8_iterate/serial.h"

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

SZ_PUBLIC sz_size_t sz_utf8_count_powervsx(sz_cptr_t text, sz_size_t length) {
    sz_u8_t const *text_u8 = (sz_u8_t const *)text;
    sz_size_t char_count = 0;

    __vector unsigned char const mask_c0_vec = vec_splats((unsigned char)0xC0);
    __vector unsigned char const pat_80_vec = vec_splats((unsigned char)0x80);
    __vector unsigned char const ones_vec = vec_splats((unsigned char)1);

    // `vec_msum(starts, ones, acc)` is the in-lane accumulator: each `starts` byte is 0 or 1, so each
    // step adds at most 4 to any 32-bit lane. A lane can absorb `UINT_MAX / 4 ≈ 1.07B` iterations
    // before overflowing. We process the input in blocks of that many 16-byte windows so the only
    // horizontal reduction happens once per block — out of the hot inner loop — and the inner loop
    // carries no per-iteration counter or branch. Results stay identical to `sz_utf8_count_serial`.
    sz_size_t const block_windows = (sz_size_t)1000000000; // 1B * 4 < UINT_MAX.
    while (length >= 16) {
        sz_size_t windows = length / 16;
        if (windows > block_windows) windows = block_windows;
        __vector unsigned int accumulator_vec = vec_splats((unsigned int)0);
        for (sz_size_t window_index = 0; window_index < windows; ++window_index, text_u8 += 16) {
            __vector unsigned char bytes_vec = vec_xl(0, text_u8);
            __vector unsigned char headers_vec = vec_and(bytes_vec, mask_c0_vec);
            // Continuation bytes -> 0xFF, starts -> 0x00; invert so each start contributes 0x01.
            __vector unsigned char is_continuation_vec = (__vector unsigned char)vec_cmpeq(headers_vec, pat_80_vec);
            __vector unsigned char starts_vec = vec_andc(ones_vec, is_continuation_vec);
            accumulator_vec = vec_msum(starts_vec, ones_vec,
                                       accumulator_vec); // Sum the start flags into the 32-bit lanes.
        }
        char_count += (sz_size_t)accumulator_vec[0] + accumulator_vec[1] + accumulator_vec[2] + accumulator_vec[3];
        length -= windows * 16;
    }

    if (length) char_count += sz_utf8_count_serial((sz_cptr_t)text_u8, length);
    return char_count;
}

SZ_PUBLIC sz_cptr_t sz_utf8_find_nth_powervsx(sz_cptr_t text, sz_size_t length, sz_size_t n) {
    //! Extracting the n-th code-point start requires a per-bit population scan; not worth
    //! vectorizing without a PDEP-equivalent — defer to serial, mirroring the NEON backend.
    return sz_utf8_find_nth_serial(text, length, n);
}

SZ_PUBLIC sz_cptr_t sz_utf8_unpack_chunk_powervsx( //
    sz_cptr_t text, sz_size_t length,              //
    sz_rune_t *runes, sz_size_t runes_capacity,    //
    sz_size_t *runes_unpacked) {
#if !SZ_IS_BIG_ENDIAN_
    sz_u8_t const *text_cursor = (sz_u8_t const *)text;
    sz_size_t runes_written = 0;

    // `vec_perm` selectors that zero-extend bytes [0..3], [4..7], [8..11], [12..15] of a 16-byte
    // vector into four `u32` lanes each. Index `0x10` selects a byte from the (zero) second operand,
    // placing 0x00 in the upper three bytes of every little-endian 32-bit lane.
    __vector unsigned char const selector0_vec = {0x00, 0x10, 0x10, 0x10, 0x01, 0x10, 0x10, 0x10,
                                                  0x02, 0x10, 0x10, 0x10, 0x03, 0x10, 0x10, 0x10};
    __vector unsigned char const selector1_vec = {0x04, 0x10, 0x10, 0x10, 0x05, 0x10, 0x10, 0x10,
                                                  0x06, 0x10, 0x10, 0x10, 0x07, 0x10, 0x10, 0x10};
    __vector unsigned char const selector2_vec = {0x08, 0x10, 0x10, 0x10, 0x09, 0x10, 0x10, 0x10,
                                                  0x0a, 0x10, 0x10, 0x10, 0x0b, 0x10, 0x10, 0x10};
    __vector unsigned char const selector3_vec = {0x0c, 0x10, 0x10, 0x10, 0x0d, 0x10, 0x10, 0x10,
                                                  0x0e, 0x10, 0x10, 0x10, 0x0f, 0x10, 0x10, 0x10};
    __vector unsigned char const zero_vec = vec_splats((unsigned char)0);

    while (length >= 16 && runes_written + 16 <= runes_capacity) {
        __vector unsigned char bytes_vec = vec_xl(0, text_cursor);
        // A window is pure ASCII when no byte has its high bit set.
        if (vec_any_ge(bytes_vec, vec_splats((unsigned char)0x80))) break;
        vec_xst((__vector unsigned int)vec_perm(bytes_vec, zero_vec, selector0_vec), 0,
                (unsigned int *)(runes + runes_written + 0));
        vec_xst((__vector unsigned int)vec_perm(bytes_vec, zero_vec, selector1_vec), 0,
                (unsigned int *)(runes + runes_written + 4));
        vec_xst((__vector unsigned int)vec_perm(bytes_vec, zero_vec, selector2_vec), 0,
                (unsigned int *)(runes + runes_written + 8));
        vec_xst((__vector unsigned int)vec_perm(bytes_vec, zero_vec, selector3_vec), 0,
                (unsigned int *)(runes + runes_written + 12));
        runes_written += 16;
        text_cursor += 16;
        length -= 16;
    }

    // Finish the remainder (non-ASCII or sub-window) with the serial decoder so the output and the
    // returned cursor stay bit-exact with `sz_utf8_unpack_chunk_serial`.
    sz_size_t tail_unpacked = 0;
    sz_cptr_t cursor = sz_utf8_unpack_chunk_serial((sz_cptr_t)text_cursor, length, runes + runes_written,
                                                   runes_capacity - runes_written, &tail_unpacked);
    *runes_unpacked = runes_written + tail_unpacked;
    return cursor;
#else
    return sz_utf8_unpack_chunk_serial(text, length, runes, runes_capacity, runes_unpacked);
#endif
}

SZ_PUBLIC sz_cptr_t sz_utf8_find_newline_powervsx(sz_cptr_t text, sz_size_t length, sz_size_t *matched_length) {
    __vector unsigned char const newline_vec = vec_splats((unsigned char)'\n');
    __vector unsigned char const vtab_vec = vec_splats((unsigned char)'\v');
    __vector unsigned char const formfeed_vec = vec_splats((unsigned char)'\f');
    __vector unsigned char const carriage_return_vec = vec_splats((unsigned char)'\r');
    __vector unsigned char const lead_c2_vec = vec_splats((unsigned char)0xC2);
    __vector unsigned char const lead_e2_vec = vec_splats((unsigned char)0xE2);

    while (length >= 16) {
        __vector unsigned char bytes_vec = vec_xl(0, (unsigned char const *)text);
        __vector unsigned char candidate_vec = vec_or(
            vec_or(vec_or((__vector unsigned char)vec_cmpeq(bytes_vec, newline_vec),
                          (__vector unsigned char)vec_cmpeq(bytes_vec, vtab_vec)),
                   vec_or((__vector unsigned char)vec_cmpeq(bytes_vec, formfeed_vec),
                          (__vector unsigned char)vec_cmpeq(bytes_vec, carriage_return_vec))),
            vec_or((__vector unsigned char)vec_cmpeq(bytes_vec, lead_c2_vec),
                   (__vector unsigned char)vec_cmpeq(bytes_vec, lead_e2_vec)));
        if (vec_any_ne(candidate_vec, vec_splats((unsigned char)0)))
            return sz_utf8_find_newline_serial(text, length, matched_length);
        text += 16, length -= 16;
    }

    return sz_utf8_find_newline_serial(text, length, matched_length);
}

SZ_PUBLIC sz_cptr_t sz_utf8_find_whitespace_powervsx(sz_cptr_t text, sz_size_t length, sz_size_t *matched_length) {
    __vector unsigned char const space_vec = vec_splats((unsigned char)' ');
    __vector unsigned char const tab_vec = vec_splats((unsigned char)'\t');
    __vector unsigned char const carriage_return_vec = vec_splats((unsigned char)'\r');
    __vector unsigned char const lead_c2_vec = vec_splats((unsigned char)0xC2);
    __vector unsigned char const lead_e1_vec = vec_splats((unsigned char)0xE1);
    __vector unsigned char const lead_e2_vec = vec_splats((unsigned char)0xE2);
    __vector unsigned char const lead_e3_vec = vec_splats((unsigned char)0xE3);

    while (length >= 16) {
        __vector unsigned char bytes_vec = vec_xl(0, (unsigned char const *)text);
        // 1-byte ASCII whitespace: ' ' or any of [\t, \r] (0x09..0x0D).
        __vector unsigned char ascii_range_vec = vec_and(
            (__vector unsigned char)vec_cmpge(bytes_vec, tab_vec),
            (__vector unsigned char)vec_cmple(bytes_vec, carriage_return_vec));
        __vector unsigned char candidate_vec = vec_or(
            vec_or(vec_or((__vector unsigned char)vec_cmpeq(bytes_vec, space_vec), ascii_range_vec),
                   vec_or((__vector unsigned char)vec_cmpeq(bytes_vec, lead_c2_vec),
                          (__vector unsigned char)vec_cmpeq(bytes_vec, lead_e1_vec))),
            vec_or((__vector unsigned char)vec_cmpeq(bytes_vec, lead_e2_vec),
                   (__vector unsigned char)vec_cmpeq(bytes_vec, lead_e3_vec)));
        if (vec_any_ne(candidate_vec, vec_splats((unsigned char)0)))
            return sz_utf8_find_whitespace_serial(text, length, matched_length);
        text += 16, length -= 16;
    }

    return sz_utf8_find_whitespace_serial(text, length, matched_length);
}

/**
 *  @brief UAX-29 word boundary detection using IBM Power VSX (forward & reverse).
 *
 *  The full UAX-29 segmenter is a stateful walk whose decision at each candidate position depends on the
 *  Word_Break properties of the surrounding runes, including multi-code-point look-around (WB6/WB7 mid-letter,
 *  WB11/WB12 numeric, WB15/WB16 Regional_Indicator parity) and WB4 Extend/Format/ZWJ skipping. We keep those
 *  stateful sub-rules in the serial reference, but accelerate the dominant common case with VSX: long runs of
 *  ASCII letters (`[A-Za-z]`) or ASCII digits (`[0-9]`).
 *
 *  Interior positions of a maximal ASCII-letter run are governed @b unconditionally by WB5 (AHLetter ×
 *  AHLetter ⇒ no break, no look-around needed), and interior positions of an ASCII-digit run by WB8 (Numeric ×
 *  Numeric ⇒ no break). Such positions can never be a boundary, so we skip them at vector speed. The first
 *  position whose byte class differs from its predecessor (or is not letter/digit) is the earliest position
 *  that could be a boundary; from there the serial reference re-tests that exact position. Because we only ever
 *  skip @b proven non-boundaries, the result is byte-for-byte identical to `_serial` (same returned pointer and
 *  `boundary_width`).
 *
 *  Per-byte classification (0 = ASCII letter, 1 = ASCII digit, 2 = other ASCII, 3 = non-ASCII) is computed
 *  in-register with VSX range compares and `vec_sel`. We store the 16-byte class vector and fold it against
 *  the previous byte's class to form a "safe interior" bitmask, then scan that mask with a trailing/leading
 *  zero count to locate the first/last candidate boundary.
 */

SZ_INTERNAL sz_u8_t sz_utf8_wb_class1_(sz_u8_t b) {
    if (b >= 0x80) return 3;
    if (b >= '0' && b <= '9') return 1;
    if ((b >= 'A' && b <= 'Z') || (b >= 'a' && b <= 'z')) return 0;
    return 2;
}

// 16-bit mask: bit i set when position base+i is a proven non-boundary (safe interior of a letter/digit run).
SZ_INTERNAL sz_u64_t sz_utf8_wb_safe_mask_powervsx_(sz_cptr_t text, sz_size_t base) {
    sz_u8_t const *text_u8 = (sz_u8_t const *)text;
    __vector unsigned char bytes_vec = vec_xl(0, text_u8 + base);
    __vector unsigned char const digit_low_vec = vec_splats((unsigned char)'0');
    __vector unsigned char const digit_high_vec = vec_splats((unsigned char)'9');
    __vector unsigned char const upper_low_vec = vec_splats((unsigned char)'A');
    __vector unsigned char const upper_high_vec = vec_splats((unsigned char)'Z');
    __vector unsigned char const lower_low_vec = vec_splats((unsigned char)'a');
    __vector unsigned char const lower_high_vec = vec_splats((unsigned char)'z');
    __vector unsigned char const high_bit_vec = vec_splats((unsigned char)0x80);

    __vector unsigned char is_high_vec = (__vector unsigned char)vec_cmpge(bytes_vec, high_bit_vec);
    __vector unsigned char is_digit_vec = vec_and((__vector unsigned char)vec_cmpge(bytes_vec, digit_low_vec),
                                                  (__vector unsigned char)vec_cmple(bytes_vec, digit_high_vec));
    __vector unsigned char is_letter_vec = vec_or(
        vec_and((__vector unsigned char)vec_cmpge(bytes_vec, upper_low_vec),
                (__vector unsigned char)vec_cmple(bytes_vec, upper_high_vec)),
        vec_and((__vector unsigned char)vec_cmpge(bytes_vec, lower_low_vec),
                (__vector unsigned char)vec_cmple(bytes_vec, lower_high_vec)));
    // class = 2 default; 1 if digit; 0 if letter; 3 if non-ASCII.
    __vector unsigned char class_vec = vec_splats((unsigned char)2);
    class_vec = vec_sel(class_vec, vec_splats((unsigned char)1), is_digit_vec);
    class_vec = vec_sel(class_vec, vec_splats((unsigned char)0), is_letter_vec);
    class_vec = vec_sel(class_vec, vec_splats((unsigned char)3), is_high_vec);

    unsigned char classes[16];
    vec_xst(class_vec, 0, classes);
    sz_u8_t previous_class = sz_utf8_wb_class1_(text_u8[base - 1]);
    sz_u64_t mask = 0;
    for (sz_size_t lane_index = 0; lane_index < 16; ++lane_index) {
        sz_u8_t predecessor_class = (lane_index == 0) ? previous_class : classes[lane_index - 1];
        if (classes[lane_index] == predecessor_class && classes[lane_index] <= 1) mask |= (sz_u64_t)1 << lane_index;
    }
    return mask;
}

SZ_PUBLIC sz_size_t sz_utf8_word_find_boundaries_powervsx( //
    sz_cptr_t text, sz_size_t length,                      //
    sz_size_t *word_starts, sz_size_t *word_lengths,       //
    sz_size_t words_capacity, sz_size_t *bytes_consumed) {

    sz_size_t words = 0;
    if (length == 0 || words_capacity == 0) {
        if (bytes_consumed) *bytes_consumed = 0;
        return 0;
    }
    sz_u8_t first_byte = (sz_u8_t)text[0];
    sz_size_t word_start = 0; // Start of the word currently being accumulated (always a boundary).
    sz_size_t position = (sz_size_t)(1 + (first_byte >= 0xC0) + (first_byte >= 0xE0) + (first_byte >= 0xF0));
    while (position < length) {
        if (position > 0 && position + 16 <= length) {
            sz_u64_t safe = sz_utf8_wb_safe_mask_powervsx_(text, position);
            sz_u64_t not_safe = (~safe) & 0xFFFFull;
            if (not_safe == 0) {
                position += 16;
                continue;
            }
            sz_size_t skipped = position + (sz_size_t)sz_u64_ctz(not_safe);
            if (skipped > position) {
                position = skipped;
                if (position >= length) break;
            }
        }
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
        sz_u8_t byte_at_position = (sz_u8_t)text[position];
        position += (sz_size_t)(1 + (byte_at_position >= 0xC0) + (byte_at_position >= 0xE0) +
                                (byte_at_position >= 0xF0));
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

SZ_PUBLIC sz_size_t sz_utf8_word_rfind_boundaries_powervsx( //
    sz_cptr_t text, sz_size_t length,                       //
    sz_size_t *word_starts, sz_size_t *word_lengths,        //
    sz_size_t words_capacity, sz_size_t *bytes_consumed) {

    sz_size_t words = 0;
    if (length == 0 || words_capacity == 0) {
        if (bytes_consumed) *bytes_consumed = length;
        return 0;
    }
    sz_size_t word_end = length; // End of the word currently being accumulated (always a boundary).
    sz_size_t position = length - 1;
    while (position > 0 && ((sz_u8_t)text[position] & 0xC0) == 0x80) position--;
    while (position > 0) {
        if (position >= 16) {
            // Window [position-15, position+1): top lane is `position` itself.
            sz_size_t base = position - 15;
            sz_u64_t safe = sz_utf8_wb_safe_mask_powervsx_(text, base);
            sz_u64_t not_safe = (~safe) & 0xFFFFull;
            if (not_safe == 0) { position = base; }
            else {
                int highest_unsafe_lane = 15 - sz_u64_clz(not_safe << 48); // highest unsafe lane within [0,16)
                position = base + (sz_size_t)highest_unsafe_lane;
            }
            if (position == 0) break;
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

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#endif // SZ_USE_POWERVSX

#ifdef __cplusplus
}
#endif

#endif // STRINGZILLA_UTF8_ITERATE_POWERVSX_H_
