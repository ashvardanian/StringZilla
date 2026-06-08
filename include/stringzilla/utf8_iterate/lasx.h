/**
 *  @brief LoongArch LASX (256-bit) backend for utf8.
 *  @file include/stringzilla/utf8_iterate/lasx.h
 *  @author Ash Vardanian
 *  @sa utf8.h
 */
#ifndef STRINGZILLA_UTF8_ITERATE_LASX_H_
#define STRINGZILLA_UTF8_ITERATE_LASX_H_

#include "stringzilla/types.h"
#include "stringzilla/utf8_iterate/serial.h"

#ifdef __cplusplus
extern "C" {
#endif

#if SZ_USE_LASX

/*  UTF-8 -> UTF-32 transcoding for one chunk.
 *
 *  The serial reference walks the input one codepoint at a time via `sz_rune_parse`, where each step's byte
 *  width (1..4) depends on the just-decoded leading byte. That data-dependent stride, combined with the
 *  output-capacity bound and the "stop on an incomplete trailing sequence" rule, has no clean value-identical
 *  256-bit LASX form (the same reason the NEON backend delegates). We therefore forward to the serial
 *  reference, keeping the public LASX symbol present and byte-for-byte correct. */
SZ_PUBLIC sz_cptr_t sz_utf8_unpack_chunk_lasx(  //
    sz_cptr_t text, sz_size_t length,           //
    sz_rune_t *runes, sz_size_t runes_capacity, //
    sz_size_t *runes_unpacked) {
    return sz_utf8_unpack_chunk_serial(text, length, runes, runes_capacity, runes_unpacked);
}

/*  See `compare/lasx.h`: `__lasx_xvmskltz_b` packs each byte's sign bit into a per-128-bit-lane 16-bit
 *  mask (word 0 = low lane, word 4 = high lane), recombined to match AVX2's `_mm256_movemask_epi8`. */
SZ_INTERNAL sz_u32_t sz_xvmovemask_b_utf8_lasx_(__m256i sign_extended) {
    __m256i collected = __lasx_xvmskltz_b(sign_extended);
    sz_u32_t low = (sz_u32_t)__lasx_xvpickve2gr_wu(collected, 0);
    sz_u32_t high = (sz_u32_t)__lasx_xvpickve2gr_wu(collected, 4);
    return (low & 0xFFFFu) | ((high & 0xFFFFu) << 16);
}

SZ_PUBLIC sz_cptr_t sz_utf8_find_newline_lasx(sz_cptr_t text, sz_size_t length, sz_size_t *matched_length) {

    // We need to check if the ASCII chars in [10,13] ('\n', '\v', '\f', '\r') are present.
    // '\r' needs special handling to differentiate between "\r" and "\r\n".
    __m256i n_vec = __lasx_xvreplgr2vr_b('\n');
    __m256i v_vec = __lasx_xvreplgr2vr_b('\v');
    __m256i f_vec = __lasx_xvreplgr2vr_b('\f');
    __m256i r_vec = __lasx_xvreplgr2vr_b('\r');

    // 2-byte newline 0xC285 (NEL), 3-byte 0xE280A8 (LS) and 0xE280A9 (PS).
    __m256i x_c2_vec = __lasx_xvreplgr2vr_b((char)0xC2);
    __m256i x_85_vec = __lasx_xvreplgr2vr_b((char)0x85);
    __m256i x_e2_vec = __lasx_xvreplgr2vr_b((char)0xE2);
    __m256i x_80_vec = __lasx_xvreplgr2vr_b((char)0x80);
    __m256i x_a8_vec = __lasx_xvreplgr2vr_b((char)0xA8);
    __m256i x_a9_vec = __lasx_xvreplgr2vr_b((char)0xA9);

    while (length >= 32) {
        __m256i text_vec = __lasx_xvld(text, 0);

        sz_u32_t n_mask = sz_xvmovemask_b_utf8_lasx_(__lasx_xvseq_b(text_vec, n_vec));
        sz_u32_t v_mask = sz_xvmovemask_b_utf8_lasx_(__lasx_xvseq_b(text_vec, v_vec));
        sz_u32_t f_mask = sz_xvmovemask_b_utf8_lasx_(__lasx_xvseq_b(text_vec, f_vec));
        sz_u32_t r_mask = sz_xvmovemask_b_utf8_lasx_(__lasx_xvseq_b(text_vec, r_vec)) & 0x7FFFFFFF; // Ignore last byte
        sz_u32_t one_byte_mask = n_mask | v_mask | f_mask | r_mask;

        sz_u32_t x_c2_mask = sz_xvmovemask_b_utf8_lasx_(__lasx_xvseq_b(text_vec, x_c2_vec));
        sz_u32_t x_85_mask = sz_xvmovemask_b_utf8_lasx_(__lasx_xvseq_b(text_vec, x_85_vec));
        sz_u32_t x_e2_mask = sz_xvmovemask_b_utf8_lasx_(__lasx_xvseq_b(text_vec, x_e2_vec));
        sz_u32_t x_80_mask = sz_xvmovemask_b_utf8_lasx_(__lasx_xvseq_b(text_vec, x_80_vec));
        sz_u32_t x_a8_mask = sz_xvmovemask_b_utf8_lasx_(__lasx_xvseq_b(text_vec, x_a8_vec));
        sz_u32_t x_a9_mask = sz_xvmovemask_b_utf8_lasx_(__lasx_xvseq_b(text_vec, x_a9_vec));

        sz_u32_t rn_mask = r_mask & (n_mask >> 1);
        sz_u32_t x_c285_mask = x_c2_mask & (x_85_mask >> 1);
        sz_u32_t two_byte_mask = rn_mask | x_c285_mask;

        sz_u32_t x_e280_mask = x_e2_mask & (x_80_mask >> 1);
        sz_u32_t x_e280a8_mask = x_e280_mask & (x_a8_mask >> 2);
        sz_u32_t x_e280a9_mask = x_e280_mask & (x_a9_mask >> 2);
        sz_u32_t three_byte_mask = x_e280a8_mask | x_e280a9_mask;

        sz_u32_t combined_mask = one_byte_mask | two_byte_mask | three_byte_mask;
        if (combined_mask) {
            int first_offset = sz_u32_ctz(combined_mask);
            sz_u32_t first_match_mask = (sz_u32_t)(1) << first_offset;
            sz_size_t length_value = 1;
            length_value += (first_match_mask & (two_byte_mask | three_byte_mask)) != 0;
            length_value += (first_match_mask & (three_byte_mask)) != 0;
            *matched_length = length_value;
            return text + first_offset;
        }
        else { text += 30, length -= 30; }
    }

    return sz_utf8_find_newline_serial(text, length, matched_length);
}

SZ_PUBLIC sz_cptr_t sz_utf8_find_whitespace_lasx(sz_cptr_t text, sz_size_t length, sz_size_t *matched_length) {

    __m256i x_20_vec = __lasx_xvreplgr2vr_b(' ');

    __m256i x_c2_vec = __lasx_xvreplgr2vr_b((char)0xC2);
    __m256i x_85_vec = __lasx_xvreplgr2vr_b((char)0x85);
    __m256i x_a0_vec = __lasx_xvreplgr2vr_b((char)0xA0);

    __m256i x_e1_vec = __lasx_xvreplgr2vr_b((char)0xE1);
    __m256i x_e2_vec = __lasx_xvreplgr2vr_b((char)0xE2);
    __m256i x_e3_vec = __lasx_xvreplgr2vr_b((char)0xE3);
    __m256i x_9a_vec = __lasx_xvreplgr2vr_b((char)0x9A);
    __m256i x_80_vec = __lasx_xvreplgr2vr_b((char)0x80);
    __m256i x_81_vec = __lasx_xvreplgr2vr_b((char)0x81);
    __m256i x_a8_vec = __lasx_xvreplgr2vr_b((char)0xA8);
    __m256i x_a9_vec = __lasx_xvreplgr2vr_b((char)0xA9);
    __m256i x_af_vec = __lasx_xvreplgr2vr_b((char)0xAF);
    __m256i x_9f_vec = __lasx_xvreplgr2vr_b((char)0x9F);

    while (length >= 32) {
        __m256i text_vec = __lasx_xvld(text, 0);

        // 1-byte indicators & matches. Range [9,13] covers \t, \n, \v, \f, \r.
        __m256i x_20_cmp = __lasx_xvseq_b(text_vec, x_20_vec);
        // `cmpgt_epi8(text, 0x08)` -> signed `0x08 < text`; `cmpgt_epi8(0x0E, text)` -> signed `text < 0x0E`.
        __m256i t_cmp = __lasx_xvslt_b(__lasx_xvreplgr2vr_b((char)0x08), text_vec);
        __m256i r_cmp = __lasx_xvslt_b(text_vec, __lasx_xvreplgr2vr_b((char)0x0E));
        __m256i tr_range = __lasx_xvand_v(t_cmp, r_cmp);
        __m256i one_byte_cmp = __lasx_xvor_v(x_20_cmp, tr_range);
        sz_u32_t one_byte_mask = sz_xvmovemask_b_utf8_lasx_(one_byte_cmp);

        __m256i x_c2_cmp = __lasx_xvseq_b(text_vec, x_c2_vec);
        __m256i x_e1_cmp = __lasx_xvseq_b(text_vec, x_e1_vec);
        __m256i x_e2_cmp = __lasx_xvseq_b(text_vec, x_e2_vec);
        __m256i x_e3_cmp = __lasx_xvseq_b(text_vec, x_e3_vec);

        sz_u32_t x_c2_mask = sz_xvmovemask_b_utf8_lasx_(x_c2_cmp) & 0x7FFFFFFF;
        sz_u32_t x_e1_mask = sz_xvmovemask_b_utf8_lasx_(x_e1_cmp) & 0x3FFFFFFF;
        sz_u32_t x_e2_mask = sz_xvmovemask_b_utf8_lasx_(x_e2_cmp) & 0x3FFFFFFF;
        sz_u32_t x_e3_mask = sz_xvmovemask_b_utf8_lasx_(x_e3_cmp) & 0x3FFFFFFF;
        sz_u32_t prefix_byte_mask = x_c2_mask | x_e1_mask | x_e2_mask | x_e3_mask;

        // Fast path: a one-byte match before any multi-byte prefix.
        if (one_byte_mask) {
            if (prefix_byte_mask) {
                int first_one_byte_offset = sz_u32_ctz(one_byte_mask);
                int first_prefix_offset = sz_u32_ctz(prefix_byte_mask);
                if (first_one_byte_offset < first_prefix_offset) {
                    *matched_length = 1;
                    return text + first_one_byte_offset;
                }
            }
            else {
                int first_one_byte_offset = sz_u32_ctz(one_byte_mask);
                *matched_length = 1;
                return text + first_one_byte_offset;
            }
        }

        // 2-byte suffixes.
        sz_u32_t x_85_mask = sz_xvmovemask_b_utf8_lasx_(__lasx_xvseq_b(text_vec, x_85_vec));
        sz_u32_t x_a0_mask = sz_xvmovemask_b_utf8_lasx_(__lasx_xvseq_b(text_vec, x_a0_vec));
        sz_u32_t x_c285_mask = x_c2_mask & (x_85_mask >> 1); // U+0085 NEL
        sz_u32_t x_c2a0_mask = x_c2_mask & (x_a0_mask >> 1); // U+00A0 NBSP
        sz_u32_t two_byte_mask = x_c285_mask | x_c2a0_mask;

        // 3-byte suffixes.
        sz_u32_t x_9a_mask = sz_xvmovemask_b_utf8_lasx_(__lasx_xvseq_b(text_vec, x_9a_vec));
        sz_u32_t x_80_mask = sz_xvmovemask_b_utf8_lasx_(__lasx_xvseq_b(text_vec, x_80_vec));
        sz_u32_t x_81_mask = sz_xvmovemask_b_utf8_lasx_(__lasx_xvseq_b(text_vec, x_81_vec));
        // [0x80, 0x8D] range: unsigned `text >= 0x80` AND `text <= 0x8D`.
        __m256i x_80_ge_cmp = __lasx_xvsle_bu(x_80_vec, text_vec);
        __m256i x_8d_le_cmp = __lasx_xvsle_bu(text_vec, __lasx_xvreplgr2vr_b((char)0x8D));
        __m256i x_8d_range = __lasx_xvand_v(x_80_ge_cmp, x_8d_le_cmp);
        sz_u32_t x_8d_range_mask = sz_xvmovemask_b_utf8_lasx_(x_8d_range);
        sz_u32_t x_a8_mask = sz_xvmovemask_b_utf8_lasx_(__lasx_xvseq_b(text_vec, x_a8_vec));
        sz_u32_t x_a9_mask = sz_xvmovemask_b_utf8_lasx_(__lasx_xvseq_b(text_vec, x_a9_vec));
        sz_u32_t x_af_mask = sz_xvmovemask_b_utf8_lasx_(__lasx_xvseq_b(text_vec, x_af_vec));
        sz_u32_t x_9f_mask = sz_xvmovemask_b_utf8_lasx_(__lasx_xvseq_b(text_vec, x_9f_vec));

        sz_u32_t ogham_mask = x_e1_mask & (x_9a_mask >> 1) & (x_80_mask >> 2);            // E1 9A 80
        sz_u32_t range_e280_mask = x_e2_mask & (x_80_mask >> 1) & (x_8d_range_mask >> 2); // E2 80 [80-8D]
        sz_u32_t line_mask = x_e2_mask & (x_80_mask >> 1) & (x_a8_mask >> 2);             // E2 80 A8
        sz_u32_t paragraph_mask = x_e2_mask & (x_80_mask >> 1) & (x_a9_mask >> 2);        // E2 80 A9
        sz_u32_t nnbsp_mask = x_e2_mask & (x_80_mask >> 1) & (x_af_mask >> 2);            // E2 80 AF
        sz_u32_t mmsp_mask = x_e2_mask & (x_81_mask >> 1) & (x_9f_mask >> 2);             // E2 81 9F
        sz_u32_t ideographic_mask = x_e3_mask & (x_80_mask >> 1) & (x_80_mask >> 2);      // E3 80 80
        sz_u32_t three_byte_mask =
            ogham_mask | range_e280_mask | nnbsp_mask | mmsp_mask | line_mask | paragraph_mask | ideographic_mask;

        sz_u32_t combined_mask = one_byte_mask | two_byte_mask | three_byte_mask;
        if (combined_mask) {
            int first_offset = sz_u32_ctz(combined_mask);
            sz_u32_t first_match_mask = (sz_u32_t)(1) << first_offset;
            sz_size_t length_value = 1;
            length_value += (first_match_mask & (two_byte_mask | three_byte_mask)) != 0;
            length_value += (first_match_mask & (three_byte_mask)) != 0;
            *matched_length = length_value;
            return text + first_offset;
        }
        else { text += 30, length -= 30; }
    }

    return sz_utf8_find_whitespace_serial(text, length, matched_length);
}

SZ_PUBLIC sz_size_t sz_utf8_count_lasx(sz_cptr_t text, sz_size_t length) {
    __m256i continuation_mask_vec = __lasx_xvreplgr2vr_b((char)0xC0);
    __m256i continuation_pattern_vec = __lasx_xvreplgr2vr_b((char)0x80);
    __m256i const one_vec = __lasx_xvreplgr2vr_b(1);
    __m256i const zero_vec = __lasx_xvreplgr2vr_b(0);

    sz_u8_t const *text_u8 = (sz_u8_t const *)text;
    sz_size_t char_count = 0;

    // Keep a per-lane vector accumulator across the loop and reduce ONCE at the end, instead of doing a
    // movemask + scalar `popcount` per block. A start byte is any byte where `(b & 0xC0) != 0x80`; we map
    // that to a 0/1 byte and fold adjacent pairs into a 16x u16 accumulator via `xvhaddw_hu_bu`. Each u16
    // lane grows by at most 2 per block, so 65535/2 = 32767 blocks fit before overflow; we flush the u16
    // lanes into a 4x u64 accumulator every 32000 blocks to stay safe (rare for realistic inputs).
    __m256i sums_d = zero_vec;
    __m256i sums_h = zero_vec;
    sz_size_t blocks_since_flush = 0;
    while (length >= 32) {
        __m256i text_vec = __lasx_xvld(text_u8, 0);
        // Extract top 2 bits of each byte; continuation bytes equal 0x80. `is_start` is 0xFF for starts.
        __m256i headers = __lasx_xvand_v(text_vec, continuation_mask_vec);
        __m256i is_cont = __lasx_xvseq_b(headers, continuation_pattern_vec); // 0xFF where continuation
        // `andn` style: 1 where start, 0 where continuation. (~is_cont) & 1.
        __m256i is_start = __lasx_xvandn_v(is_cont, one_vec); // (~is_cont) & one
        sums_h = __lasx_xvadd_h(sums_h, __lasx_xvhaddw_hu_bu(is_start, is_start));
        if (++blocks_since_flush == 32000) {
            __m256i widened_w = __lasx_xvhaddw_wu_hu(sums_h, sums_h);
            sums_d = __lasx_xvadd_d(sums_d, __lasx_xvhaddw_du_wu(widened_w, widened_w));
            sums_h = zero_vec;
            blocks_since_flush = 0;
        }
        text_u8 += 32, length -= 32;
    }
    {
        __m256i widened_w = __lasx_xvhaddw_wu_hu(sums_h, sums_h);
        sums_d = __lasx_xvadd_d(sums_d, __lasx_xvhaddw_du_wu(widened_w, widened_w));
    }
    char_count += (sz_size_t)__lasx_xvpickve2gr_du(sums_d, 0) + (sz_size_t)__lasx_xvpickve2gr_du(sums_d, 1) +
                  (sz_size_t)__lasx_xvpickve2gr_du(sums_d, 2) + (sz_size_t)__lasx_xvpickve2gr_du(sums_d, 3);

    char_count += sz_utf8_count_serial((sz_cptr_t)text_u8, length);
    return char_count;
}

SZ_PUBLIC sz_cptr_t sz_utf8_find_nth_lasx(sz_cptr_t text, sz_size_t length, sz_size_t n) {
    // Same block logic as `sz_utf8_count_lasx`, but locating the Nth start byte.
    // LASX has no `PDEP`, so we iterate the start-byte bitmask with `ctz` to land on the Nth set bit.
    __m256i continuation_mask_vec = __lasx_xvreplgr2vr_b((char)0xC0);
    __m256i continuation_pattern_vec = __lasx_xvreplgr2vr_b((char)0x80);

    sz_u8_t const *text_u8 = (sz_u8_t const *)text;

    sz_u256_vec_t text_vec, headers_vec;
    while (length >= 32) {
        text_vec.lasx = __lasx_xvld(text_u8, 0);
        headers_vec.lasx = __lasx_xvand_v(text_vec.lasx, continuation_mask_vec);
        sz_u32_t start_byte_mask =
            ~sz_xvmovemask_b_utf8_lasx_(__lasx_xvseq_b(headers_vec.lasx, continuation_pattern_vec));
        sz_size_t start_byte_count = (sz_size_t)__builtin_popcount(start_byte_mask);

        if (n < start_byte_count) {
            // Drop the lowest `n` set bits, then the next set bit is our target.
            for (sz_size_t skipped = 0; skipped < n; ++skipped) start_byte_mask &= start_byte_mask - 1;
            int byte_offset = sz_u32_ctz(start_byte_mask);
            return (sz_cptr_t)(text_u8 + byte_offset);
        }
        else { n -= start_byte_count, text_u8 += 32, length -= 32; }
    }

    return sz_utf8_find_nth_serial((sz_cptr_t)text_u8, length, n);
}

/*  UAX-29 word boundary detection (forward & reverse).
 *
 *  The full UAX-29 segmenter is a stateful walk whose decision at each candidate position depends on the
 *  Word_Break properties of the surrounding runes, including multi-codepoint look-around (WB6/WB7 mid-letter,
 *  WB11/WB12 numeric, WB15/WB16 Regional_Indicator parity) and WB4 Extend/Format/ZWJ skipping. We keep those
 *  stateful sub-rules in the serial reference, but accelerate the dominant common case with 256-bit LASX:
 *  long runs of ASCII letters (`[A-Za-z]`) or ASCII digits (`[0-9]`).
 *
 *  Interior positions of a maximal ASCII-letter run are covered @b unconditionally by WB5 (AHLetter ×
 *  AHLetter ⇒ no break, no look-around), and interior positions of an ASCII-digit run by WB8 (Numeric ×
 *  Numeric ⇒ no break). Such positions can never be a boundary, so we skip them; the first position whose byte
 *  class differs from its predecessor (or is non-letter/digit) is the earliest possible boundary, handed off
 *  to the serial reference which re-tests it. Skipping only proven non-boundaries keeps the result
 *  byte-for-byte identical to `_serial` (same returned pointer and `boundary_width`).
 *
 *  Classification uses plain LASX byte compares: class 0 = ASCII letter, 1 = ASCII digit, 2 = other ASCII,
 *  3 = non-ASCII. We then build a per-byte "safe" mask (class == predecessor's class AND class ∈ {0,1}) and
 *  scan it via `sz_xvmovemask_b_utf8_lasx_` for the first/last position that is not safe. */

SZ_INTERNAL __m256i sz_utf8_wb_classify_lasx_(__m256i bytes_vec) {
    __m256i is_high = __lasx_xvsle_bu(__lasx_xvreplgr2vr_b((char)0x80), bytes_vec); // byte >= 0x80
    __m256i ge_zero = __lasx_xvsle_bu(__lasx_xvreplgr2vr_b('0'), bytes_vec);
    __m256i le_nine = __lasx_xvsle_bu(bytes_vec, __lasx_xvreplgr2vr_b('9'));
    __m256i is_digit = __lasx_xvand_v(ge_zero, le_nine);
    __m256i ge_upper_a = __lasx_xvsle_bu(__lasx_xvreplgr2vr_b('A'), bytes_vec);
    __m256i le_upper_z = __lasx_xvsle_bu(bytes_vec, __lasx_xvreplgr2vr_b('Z'));
    __m256i ge_lower_a = __lasx_xvsle_bu(__lasx_xvreplgr2vr_b('a'), bytes_vec);
    __m256i le_lower_z = __lasx_xvsle_bu(bytes_vec, __lasx_xvreplgr2vr_b('z'));
    __m256i is_letter = __lasx_xvor_v(__lasx_xvand_v(ge_upper_a, le_upper_z), __lasx_xvand_v(ge_lower_a, le_lower_z));
    // Start with class 2; override with 1 for digit, 0 for letter, 3 for non-ASCII.
    __m256i classes_vec = __lasx_xvreplgr2vr_b(2);
    classes_vec = __lasx_xvbitsel_v(classes_vec, __lasx_xvreplgr2vr_b(1), is_digit);
    classes_vec = __lasx_xvbitsel_v(classes_vec, __lasx_xvreplgr2vr_b(0), is_letter);
    classes_vec = __lasx_xvbitsel_v(classes_vec, __lasx_xvreplgr2vr_b(3), is_high);
    return classes_vec;
}

// 32-bit mask: bit i set when position base+i is a proven non-boundary (safe interior of a letter/digit run).
SZ_INTERNAL sz_u32_t sz_utf8_wb_safe_mask_lasx_(sz_cptr_t text, sz_size_t base) {
    sz_u8_t const *text_bytes = (sz_u8_t const *)text;
    // Position base+i is a proven non-boundary when its class equals the class of byte base+i-1 and both are
    // letter(0)/digit(1). Rather than store the class window and walk it scalar-side, we classify a SECOND
    // window loaded one byte earlier: its lane i holds the class of byte base+i-1 — exactly the per-lane
    // predecessor, lane 0 included. Callers guarantee base >= 1 and base+32 <= length, so both loads are
    // in-bounds (see `sz_utf8_word_find_boundary_lasx` / `_rfind_`).
    __m256i classes_vec = sz_utf8_wb_classify_lasx_(__lasx_xvld(text_bytes + base, 0));
    __m256i predecessor_vec = sz_utf8_wb_classify_lasx_(__lasx_xvld(text_bytes + base - 1, 0));
    __m256i same_class_vec = __lasx_xvseq_b(classes_vec, predecessor_vec);          // class[i] == class[i-1]
    __m256i word_class_vec = __lasx_xvsle_bu(classes_vec, __lasx_xvreplgr2vr_b(1)); // class[i] in {letter, digit}
    return sz_xvmovemask_b_utf8_lasx_(__lasx_xvand_v(same_class_vec, word_class_vec));
}

SZ_PUBLIC sz_cptr_t sz_utf8_word_find_boundary_lasx(sz_cptr_t text, sz_size_t length, sz_size_t *boundary_width) {
    if (length == 0) {
        if (boundary_width) *boundary_width = 0;
        return text;
    }
    sz_size_t pos =
        (sz_size_t)(1 + ((sz_u8_t)text[0] >= 0xC0) + ((sz_u8_t)text[0] >= 0xE0) + ((sz_u8_t)text[0] >= 0xF0));
    while (pos < length) {
        if (pos > 0 && pos + 32 <= length) {
            sz_u32_t safe = sz_utf8_wb_safe_mask_lasx_(text, pos);
            sz_u32_t not_safe = ~safe;
            if (not_safe == 0) {
                pos += 32;
                continue;
            }
            sz_size_t skipped = pos + (sz_size_t)sz_u32_ctz(not_safe);
            if (skipped > pos) {
                pos = skipped;
                if (pos >= length) break;
            }
        }
        if (sz_utf8_is_word_boundary_serial(text, length, pos)) {
            if (boundary_width) *boundary_width = pos;
            return text + pos;
        }
        pos +=
            (sz_size_t)(1 + ((sz_u8_t)text[pos] >= 0xC0) + ((sz_u8_t)text[pos] >= 0xE0) + ((sz_u8_t)text[pos] >= 0xF0));
    }
    if (boundary_width) *boundary_width = length;
    return text + length;
}

SZ_PUBLIC sz_cptr_t sz_utf8_word_rfind_boundary_lasx(sz_cptr_t text, sz_size_t length, sz_size_t *boundary_width) {
    if (length == 0) {
        if (boundary_width) *boundary_width = 0;
        return text;
    }
    sz_size_t pos = length - 1;
    while (pos > 0 && ((sz_u8_t)text[pos] & 0xC0) == 0x80) pos--;
    while (pos > 0) {
        if (pos >= 32) {
            // Window [pos-31, pos+1): top lane is `pos` itself.
            sz_size_t base = pos - 31;
            sz_u32_t safe = sz_utf8_wb_safe_mask_lasx_(text, base);
            sz_u32_t not_safe = ~safe;
            if (not_safe == 0) { pos = base; }
            else {
                int hi = 31 - sz_u32_clz(not_safe); // highest unsafe lane
                pos = base + (sz_size_t)hi;
            }
            if (pos == 0) break;
        }
        if (sz_utf8_is_word_boundary_serial(text, length, pos)) {
            if (boundary_width) *boundary_width = length - pos;
            return text + pos;
        }
        pos--;
        while (pos > 0 && ((sz_u8_t)text[pos] & 0xC0) == 0x80) pos--;
    }
    if (boundary_width) *boundary_width = length;
    return text;
}

#endif // SZ_USE_LASX

#ifdef __cplusplus
}
#endif

#endif // STRINGZILLA_UTF8_ITERATE_LASX_H_
