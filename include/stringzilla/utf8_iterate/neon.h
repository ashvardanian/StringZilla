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

SZ_INTERNAL sz_u64_t sz_utf8_vreinterpretq_u8_u4_(uint8x16_t vec) {
    // Use `vshrn` to produce a bitmask, similar to `movemask` in SSE.
    // https://community.arm.com/arm-community-blogs/b/infrastructure-solutions-blog/posts/porting-x86-vector-bitmask-optimizations-to-arm-neon
    return vget_lane_u64(vreinterpret_u64_u8(vshrn_n_u16(vreinterpretq_u16_u8(vec), 4)), 0) & 0x8888888888888888ull;
}

SZ_PUBLIC sz_cptr_t sz_utf8_find_newline_neon(sz_cptr_t text, sz_size_t length, sz_size_t *matched_length) {

    sz_u128_vec_t text_vec;
    uint8x16_t newline_u8x16 = vdupq_n_u8('\n');
    uint8x16_t v_u8x16 = vdupq_n_u8('\v');
    uint8x16_t f_u8x16 = vdupq_n_u8('\f');
    uint8x16_t r_u8x16 = vdupq_n_u8('\r');
    uint8x16_t x_c2_u8x16 = vdupq_n_u8(0xC2);
    uint8x16_t x_85_u8x16 = vdupq_n_u8(0x85);
    uint8x16_t x_e2_u8x16 = vdupq_n_u8(0xE2);
    uint8x16_t x_80_u8x16 = vdupq_n_u8(0x80);
    uint8x16_t x_a8_u8x16 = vdupq_n_u8(0xA8);
    uint8x16_t x_a9_u8x16 = vdupq_n_u8(0xA9);

    uint8x16_t drop1_u8x16 = vsetq_lane_u8(0x00, vdupq_n_u8(0xFF), 15);
    uint8x16_t drop2_u8x16 = vsetq_lane_u8(0x00, drop1_u8x16, 14);

    while (length >= 16) {
        text_vec.u8x16 = vld1q_u8((sz_u8_t const *)text);

        // 1-byte matches
        uint8x16_t newline_cmp = vceqq_u8(text_vec.u8x16, newline_u8x16);
        uint8x16_t v_cmp = vceqq_u8(text_vec.u8x16, v_u8x16);
        uint8x16_t f_cmp = vceqq_u8(text_vec.u8x16, f_u8x16);
        uint8x16_t r_cmp = vandq_u8(vceqq_u8(text_vec.u8x16, r_u8x16), drop1_u8x16); // Mask out \r at position 15
        uint8x16_t one_byte_cmp = vorrq_u8(vorrq_u8(newline_cmp, v_cmp), vorrq_u8(f_cmp, r_cmp));

        // 2- & 3-byte matches with shifted views
        uint8x16_t text1 = vextq_u8(text_vec.u8x16, text_vec.u8x16, 1);
        uint8x16_t text2 = vextq_u8(text_vec.u8x16, text_vec.u8x16, 2);
        uint8x16_t rn_match_u8x16 = vandq_u8(r_cmp, vceqq_u8(text1, newline_u8x16));
        uint8x16_t x_c285_u8x16 = vandq_u8(vceqq_u8(text_vec.u8x16, x_c2_u8x16), vceqq_u8(text1, x_85_u8x16));
        uint8x16_t two_byte_cmp = vandq_u8(vorrq_u8(rn_match_u8x16, x_c285_u8x16),
                                           drop1_u8x16); // Ignore last split match

        uint8x16_t x_e280_cmp = vandq_u8(vceqq_u8(text_vec.u8x16, x_e2_u8x16), vceqq_u8(text1, x_80_u8x16));
        uint8x16_t x_e280ax_cmp = vandq_u8(x_e280_cmp,
                                           vorrq_u8(vceqq_u8(text2, x_a8_u8x16), vceqq_u8(text2, x_a9_u8x16)));
        uint8x16_t three_byte_cmp = vandq_u8(x_e280ax_cmp, drop2_u8x16); // Ignore last two split matches

        // Quick presence check
        uint8x16_t combined_vec = vorrq_u8(one_byte_cmp, vorrq_u8(two_byte_cmp, three_byte_cmp));
        if (vmaxvq_u8(combined_vec)) {

            // Late mask extraction only when a match exists
            sz_u64_t one_byte_mask = sz_utf8_vreinterpretq_u8_u4_(one_byte_cmp);
            sz_u64_t two_mask = sz_utf8_vreinterpretq_u8_u4_(two_byte_cmp);
            sz_u64_t three_mask = sz_utf8_vreinterpretq_u8_u4_(three_byte_cmp);
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
    uint8x16_t t_u8x16 = vdupq_n_u8('\t');
    uint8x16_t r_u8x16 = vdupq_n_u8('\r');
    uint8x16_t x_20_u8x16 = vdupq_n_u8(' ');
    uint8x16_t x_c2_u8x16 = vdupq_n_u8(0xC2);
    uint8x16_t x_85_u8x16 = vdupq_n_u8(0x85);
    uint8x16_t x_a0_u8x16 = vdupq_n_u8(0xA0);
    uint8x16_t x_e1_u8x16 = vdupq_n_u8(0xE1);
    uint8x16_t x_e2_u8x16 = vdupq_n_u8(0xE2);
    uint8x16_t x_e3_u8x16 = vdupq_n_u8(0xE3);
    uint8x16_t x_9a_u8x16 = vdupq_n_u8(0x9A);
    uint8x16_t x_80_u8x16 = vdupq_n_u8(0x80);
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
        uint8x16_t range_cmp = vandq_u8(vcgeq_u8(text_vec.u8x16, t_u8x16), vcleq_u8(text_vec.u8x16, r_u8x16));
        uint8x16_t one_byte_cmp = vorrq_u8(x_20_cmp, range_cmp);

        // 2-byte and 3-byte prefix indicators
        uint8x16_t x_c2_cmp = vandq_u8(vceqq_u8(text_vec.u8x16, x_c2_u8x16), drop1_u8x16);
        uint8x16_t x_e1_cmp = vandq_u8(vceqq_u8(text_vec.u8x16, x_e1_u8x16), drop2_u8x16);
        uint8x16_t x_e2_cmp = vandq_u8(vceqq_u8(text_vec.u8x16, x_e2_u8x16), drop2_u8x16);
        uint8x16_t x_e3_cmp = vandq_u8(vceqq_u8(text_vec.u8x16, x_e3_u8x16), drop2_u8x16);
        uint8x16_t prefix_byte_cmp = vorrq_u8(one_byte_cmp,
                                              vorrq_u8(vorrq_u8(x_c2_cmp, x_e1_cmp), vorrq_u8(x_e2_cmp, x_e3_cmp)));

        sz_u64_t one_byte_mask = sz_utf8_vreinterpretq_u8_u4_(one_byte_cmp);
        sz_u64_t prefix_byte_mask = sz_utf8_vreinterpretq_u8_u4_(prefix_byte_cmp);

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
        uint8x16_t two_u8x16 = vorrq_u8(vandq_u8(x_c2_cmp, vceqq_u8(text1, x_85_u8x16)),
                                        vandq_u8(x_c2_cmp, vceqq_u8(text1, x_a0_u8x16)));

        // 3-byte matches
        uint8x16_t text2 = vextq_u8(text_vec.u8x16, text_vec.u8x16, 2);
        uint8x16_t x_80_ge_cmp = vcgeq_u8(text2, x_80_u8x16);
        uint8x16_t x_8d_le_cmp = vcleq_u8(text2, x_8d_u8x16);

        uint8x16_t ogham_cmp = vandq_u8(x_e1_cmp, vandq_u8(vceqq_u8(text1, x_9a_u8x16), vceqq_u8(text2, x_80_u8x16)));
        uint8x16_t range_e280_cmp = vandq_u8(x_e2_cmp,
                                             vandq_u8(vceqq_u8(text1, x_80_u8x16), vandq_u8(x_80_ge_cmp, x_8d_le_cmp)));
        uint8x16_t line_cmp = vandq_u8(x_e2_cmp, vandq_u8(vceqq_u8(text1, x_80_u8x16), vceqq_u8(text2, x_a8_u8x16)));
        uint8x16_t paragraph_cmp = vandq_u8(x_e2_cmp,
                                            vandq_u8(vceqq_u8(text1, x_80_u8x16), vceqq_u8(text2, x_a9_u8x16)));
        uint8x16_t nnbsp_cmp = vandq_u8(x_e2_cmp, vandq_u8(vceqq_u8(text1, x_80_u8x16), vceqq_u8(text2, x_af_u8x16)));
        uint8x16_t mmsp_cmp = vandq_u8(x_e2_cmp, vandq_u8(vceqq_u8(text1, x_81_u8x16), vceqq_u8(text2, x_9f_u8x16)));
        uint8x16_t ideographic_u8x16 = vandq_u8(x_e3_cmp,
                                                vandq_u8(vceqq_u8(text1, x_80_u8x16), vceqq_u8(text2, x_80_u8x16)));
        uint8x16_t three_u8x16 = vandq_u8(
            vorrq_u8(vorrq_u8(vorrq_u8(ogham_cmp, range_e280_cmp), vorrq_u8(line_cmp, paragraph_cmp)),
                     vorrq_u8(vorrq_u8(nnbsp_cmp, mmsp_cmp), ideographic_u8x16)),
            drop2_u8x16);

        sz_u64_t two_byte_mask = sz_utf8_vreinterpretq_u8_u4_(two_u8x16);
        sz_u64_t three_byte_mask = sz_utf8_vreinterpretq_u8_u4_(three_u8x16);
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
 *  The full UAX-29 segmenter is a stateful walk whose decision at each candidate position depends on the
 *  Word_Break properties of the surrounding runes, including look-around (WB6/WB7 mid-letter, WB11/WB12
 *  numeric, WB15/WB16 Regional Indicator parity) and WB4 Extend/Format/ZWJ skipping. We keep those stateful
 *  sub-rules in the serial reference, but accelerate the dominant common case: long runs of ASCII letters
 *  (`[A-Za-z]`) or ASCII digits (`[0-9]`).
 *
 *  Inside a maximal run of consecutive ASCII letters, every interior position is governed @b unconditionally
 *  by WB5 (AHLetter × AHLetter ⇒ no break): both neighbours are ALetter, neither is CR/LF/Newline/ZWJ/
 *  Extend/Format, and WB5 needs no look-around. Likewise interior positions of an ASCII-digit run are covered
 *  unconditionally by WB8 (Numeric × Numeric ⇒ no break). So no interior position of such a run can be a word
 *  boundary, and we may skip it. The first byte that is @b not the same word-class as its predecessor (or the
 *  first non-ASCII byte) is the earliest position that could possibly be a boundary; from there we hand off to
 *  the serial reference, which re-tests that exact position. Skipping only proven non-boundaries keeps the
 *  result byte-for-byte identical to `_serial` (same returned pointer and `boundary_width`).
 *
 *  `vqtbl1q_u8` classifies a 16-byte window into a per-byte class (0 = letter, 1 = digit, 2 = other/ASCII-
 *  punct, 3 = non-ASCII) via the low nibble for ASCII and a high-bit test for the rest. We then look for the
 *  first lane where the class differs from the previous lane (a class transition) or is not letter/digit. */

SZ_INTERNAL uint8x16_t sz_utf8_wb_classify_neon_(uint8x16_t v) {
    // Classify each byte: 0 = ASCII letter [A-Za-z], 1 = ASCII digit [0-9], 2 = other ASCII, 3 = non-ASCII.
    // Non-ASCII (high bit set) -> class 3.
    uint8x16_t is_high = vcgeq_u8(v, vdupq_n_u8(0x80));
    uint8x16_t is_digit = vandq_u8(vcgeq_u8(v, vdupq_n_u8('0')), vcleq_u8(v, vdupq_n_u8('9')));
    uint8x16_t is_upper = vandq_u8(vcgeq_u8(v, vdupq_n_u8('A')), vcleq_u8(v, vdupq_n_u8('Z')));
    uint8x16_t is_lower = vandq_u8(vcgeq_u8(v, vdupq_n_u8('a')), vcleq_u8(v, vdupq_n_u8('z')));
    uint8x16_t is_letter = vorrq_u8(is_upper, is_lower);
    // Start from class 2 (other ASCII), override.
    uint8x16_t cls = vdupq_n_u8(2);
    cls = vbslq_u8(is_digit, vdupq_n_u8(1), cls);
    cls = vbslq_u8(is_letter, vdupq_n_u8(0), cls);
    cls = vbslq_u8(is_high, vdupq_n_u8(3), cls);
    return cls;
}

/*  Advance `position` over bytes that are provably interior to an ASCII letter/digit run (and hence not word
 *  boundaries), starting from a position that is known to be a run interior or run start. Returns the first
 *  position that may be a boundary. */
SZ_INTERNAL sz_size_t sz_utf8_wb_skip_run_neon_(sz_cptr_t text, sz_size_t length, sz_size_t position) {
    sz_u8_t const *text_u8 = (sz_u8_t const *)text;
    while (position + 16 <= length) {
        uint8x16_t v = vld1q_u8(text_u8 + position);
        uint8x16_t cls = sz_utf8_wb_classify_neon_(v);
        // Previous-lane class via a 1-byte shift bringing in the byte at position-1.
        uint8x16_t previous = vextq_u8(sz_utf8_wb_classify_neon_(vdupq_n_u8(text_u8[position - 1])), cls, 15);
        // A position is a "safe interior" iff its class == previous class AND class is letter(0) or digit(1).
        uint8x16_t same = vceqq_u8(cls, previous);
        uint8x16_t wordy = vcleq_u8(cls, vdupq_n_u8(1)); // class 0 or 1
        uint8x16_t safe = vandq_u8(same, wordy);
        // Find first lane that is NOT safe.
        sz_u64_t safe_mask = sz_utf8_vreinterpretq_u8_u4_(safe);
        sz_u64_t not_safe = (~safe_mask) & 0x8888888888888888ull;
        if (not_safe) return position + (sz_u64_ctz(not_safe) / 4);
        position += 16;
    }
    return position;
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

    sz_size_t word_start = 0; // Start of the word currently being accumulated (always a boundary).
    sz_size_t position = sz_utf8_char_length_((sz_u8_t)text[0]);
    while (position < length) {
        // Try to skip a vectorized run of proven non-boundaries first - no boundary lives inside such a run.
        if (position > 0 && position + 16 <= length) {
            sz_size_t skipped = sz_utf8_wb_skip_run_neon_(text, length, position);
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
        position += sz_utf8_char_length_((sz_u8_t)text[position]);
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

/*  Reverse counterpart of `sz_utf8_wb_skip_run_neon_`: given a `position` known to sit at a run interior
 *  or end, walk backward over bytes that are provably non-boundaries (same ASCII letter/digit class as their
 *  predecessor). Returns the smallest position whose non-boundary status is still proven; the caller tests it
 *  and anything below it serially. */
SZ_INTERNAL sz_size_t sz_utf8_wb_rskip_run_neon_(sz_cptr_t text, sz_size_t position) {
    // Window [base, base+16) == [position-15, position+1), so the top lane is `position` itself. Position `j`
    // is a proven non-boundary when its class equals the class of byte j-1 and both are letter/digit. The
    // highest unsafe lane `high_bit_index` is the lowest position we must still test serially; everything in
    // (high_bit_index, position] is proven safe, so we jump straight to it. If all lanes are safe we jump to
    // `base` and continue from
    // there.
    sz_u8_t const *text_u8 = (sz_u8_t const *)text;
    while (position >= 16) {
        sz_size_t base = position - 15;
        uint8x16_t v = vld1q_u8(text_u8 + base);
        uint8x16_t cls = sz_utf8_wb_classify_neon_(v);
        uint8x16_t previous = vextq_u8(sz_utf8_wb_classify_neon_(vdupq_n_u8(text_u8[base - 1])), cls, 15);
        uint8x16_t same = vceqq_u8(cls, previous);
        uint8x16_t wordy = vcleq_u8(cls, vdupq_n_u8(1));
        uint8x16_t safe = vandq_u8(same, wordy);
        sz_u64_t safe_mask = sz_utf8_vreinterpretq_u8_u4_(safe);
        sz_u64_t not_safe = (~safe_mask) & 0x8888888888888888ull;
        if (not_safe) {
            int high_bit_index = 63 - sz_u64_clz(not_safe); // bit index of highest unsafe lane
            return base + (high_bit_index / 4);
        }
        position = base;
    }
    return position;
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

    sz_size_t word_end = length; // End of the word currently being accumulated (always a boundary).
    sz_size_t position = length - 1;
    while (position > 0 && ((sz_u8_t)text[position] & 0xC0) == 0x80) position--;
    while (position > 0) {
        if (position >= 16) {
            sz_size_t skipped = sz_utf8_wb_rskip_run_neon_(text, position);
            if (skipped < position) {
                position = skipped;
                if (position == 0) break;
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
