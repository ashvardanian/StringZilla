/**
 *  @brief SVE2 backend for UTF-8 traversal.
 *  @file include/stringzilla/utf8_iterate/sve2.h
 *  @author Ash Vardanian
 *  @sa utf8.h
 */
#ifndef STRINGZILLA_UTF8_ITERATE_SVE2_H_
#define STRINGZILLA_UTF8_ITERATE_SVE2_H_

#include "stringzilla/types.h"
#include "stringzilla/utf8_iterate/serial.h"

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

SZ_PUBLIC sz_size_t sz_utf8_count_sve2(sz_cptr_t text, sz_size_t length) {
    sz_u8_t const *text_u8 = (sz_u8_t const *)text;
    sz_size_t const step = svcntb();
    sz_size_t char_count = 0;

    // Count bytes that are NOT continuation bytes: (byte & 0xC0) != 0x80
    for (sz_size_t offset = 0; offset < length; offset += step) {
        svbool_t pg = svwhilelt_b8((sz_u64_t)offset, (sz_u64_t)length);
        svuint8_t text_vec = svld1_u8(pg, text_u8 + offset);
        svbool_t is_start = svcmpne_n_u8(pg, svand_n_u8_x(pg, text_vec, 0xC0), 0x80);
        char_count += svcntp_b8(pg, is_start);
    }
    return char_count;
}

SZ_PUBLIC sz_cptr_t sz_utf8_find_nth_sve2(sz_cptr_t text, sz_size_t length, sz_size_t n) {
    sz_u8_t const *text_u8 = (sz_u8_t const *)text;
    sz_size_t const step = svcntb();

    // Find character start bytes: (byte & 0xC0) != 0x80
    for (sz_size_t offset = 0; offset < length; offset += step) {
        svbool_t pg = svwhilelt_b8((sz_u64_t)offset, (sz_u64_t)length);
        svuint8_t text_vec = svld1_u8(pg, text_u8 + offset);
        svbool_t is_start = svcmpne_n_u8(pg, svand_n_u8_x(pg, text_vec, 0xC0), 0x80);
        sz_size_t start_count = svcntp_b8(pg, is_start);

        // When we find the chunk containing the Nth character, let serial handle extraction.
        // There is no `svcompact_u8` in SVE2 (only 32/64-bit variants), and no direct instruction
        // to find the position of the Nth set bit in a predicate.
        if (n < start_count) return sz_utf8_find_nth_serial((sz_cptr_t)(text_u8 + offset), length - offset, n);
        n -= start_count;
    }

    return SZ_NULL_CHAR;
}

SZ_PUBLIC sz_cptr_t sz_utf8_find_newline_sve2(sz_cptr_t text, sz_size_t length, sz_size_t *matched_length) {
    sz_u8_t const *text_u8 = (sz_u8_t const *)text;
    sz_size_t const step = svcntb();

    // Early return for short inputs
    if (length < step) return sz_utf8_find_newline_serial(text, length, matched_length);

    // SVE2 kernels are a bit different from both NEON and Ice Lake due to presence of
    // a few very convenient and cheap instructions. Most importantly, we have `svmatch` instruction
    // that can match against a set of bytes in one go, similar to many invocations of `vceqq` in NEON
    // with subsequent mask combination.
    svuint8_t prefix_byte_set = svdupq_n_u8('\n', '\v', '\f', '\r', 0xC2, 0xE2, '\n', '\n', '\n', '\n', '\n', '\n',
                                            '\n', '\n', '\n', '\n');
    svuint8_t one_byte_set = svdupq_n_u8('\n', '\v', '\f', '\r', '\n', '\n', '\n', '\n', '\n', '\n', '\n', '\n', '\n',
                                         '\n', '\n', '\n');
    svuint8_t zeros = svdup_n_u8(0);

    // We load full `step` bytes but only match on first `step - 2` positions.
    // This allows using svext for shifted views without extra loads.
    sz_size_t const usable_step = step - 2;
    sz_size_t offset = 0;
    while (offset + step <= length) {
        svbool_t pg = svwhilelt_b8_u64(0, usable_step);             // First step-2 lanes active
        svuint8_t text0 = svld1_u8(svptrue_b8(), text_u8 + offset); // Load full step bytes

        // Fast rejection: any potential first byte?
        if (!svptest_any(pg, svmatch_u8(pg, text0, prefix_byte_set))) {
            offset += usable_step;
            continue;
        }

        // Shifted views via svext - zeros fill unused lanes at end, but pg masks them out
        svuint8_t text1 = svext_u8(text0, zeros, 1);
        svuint8_t text2 = svext_u8(text0, zeros, 2);

        // 1-byte matches
        svbool_t one_byte_mask = svmatch_u8(pg, text0, one_byte_set);

        // 2-byte matches
        svbool_t rn_mask = svand_b_z(pg, svcmpeq_n_u8(pg, text0, '\r'), svcmpeq_n_u8(pg, text1, '\n'));
        svbool_t x_c285_mask = svand_b_z(pg, svcmpeq_n_u8(pg, text0, 0xC2), svcmpeq_n_u8(pg, text1, 0x85));
        svbool_t two_byte_mask = svorr_b_z(pg, rn_mask, x_c285_mask);

        // 3-byte matches
        svbool_t x_e280_mask = svand_b_z(pg, svcmpeq_n_u8(pg, text0, 0xE2), svcmpeq_n_u8(pg, text1, 0x80));
        svbool_t three_byte_mask = svand_b_z(
            pg, x_e280_mask, svorr_b_z(pg, svcmpeq_n_u8(pg, text2, 0xA8), svcmpeq_n_u8(pg, text2, 0xA9)));

        // Technically, we may want to exclude "\r" that is part of "\r\n" from one-byte matches,
        // but we don't really need it here - it won't affect the exstimates.
        //
        //      one_byte_mask = svbic_b_z(pg, one_byte_mask, rn_mask);
        svbool_t combined_mask = svorr_b_z(pg, one_byte_mask, svorr_b_z(pg, two_byte_mask, three_byte_mask));
        if (svptest_any(pg, combined_mask)) {
            sz_size_t byte_offset = svcntp_b8(pg, svbrkb_b_z(pg, combined_mask));
            svbool_t at_pos = svcmpeq_n_u8(svptrue_b8(), svindex_u8(0, 1), (sz_u8_t)byte_offset);
            sz_size_t has_two_byte = svptest_any(at_pos, two_byte_mask);
            sz_size_t has_three_byte = svptest_any(at_pos, three_byte_mask);
            sz_size_t length_value = 1;
            length_value += has_two_byte | has_three_byte;
            length_value += has_three_byte;
            *matched_length = length_value;
            return (sz_cptr_t)(text_u8 + offset + byte_offset);
        }
        offset += usable_step;
    }

    // Handle remaining bytes with serial fallback
    return sz_utf8_find_newline_serial(text + offset, length - offset, matched_length);
}

SZ_PUBLIC sz_cptr_t sz_utf8_find_whitespace_sve2(sz_cptr_t text, sz_size_t length, sz_size_t *matched_length) {
    sz_u8_t const *text_u8 = (sz_u8_t const *)text;
    sz_size_t const step = svcntb();

    // Early return for short inputs
    if (length < step) return sz_utf8_find_whitespace_serial(text, length, matched_length);

    // Character sets for MATCH (DUPQ replicates 128-bit pattern, no stack/loads)
    svuint8_t any_byte_set = svdupq_n_u8(' ', '\t', '\n', '\v', '\f', '\r', 0xC2, 0xE1, 0xE2, 0xE3, ' ', ' ', ' ', ' ',
                                         ' ', ' ');
    svuint8_t one_byte_set = svdupq_n_u8(' ', '\t', '\n', '\v', '\f', '\r', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ',
                                         ' ');
    // Valid third bytes for E2 80 XX: U+2000-U+200D (0x80-0x8D), U+2028 (0xA8), U+2029 (0xA9)
    svuint8_t e280_third_bytes = svdupq_n_u8(0x80, 0x81, 0x82, 0x83, 0x84, 0x85, 0x86, 0x87, 0x88, 0x89, 0x8A, 0x8B,
                                             0x8C, 0x8D, 0xA8, 0xA9);
    svuint8_t zeros = svdup_n_u8(0);

    // We load full `step` bytes but only match on first `step - 2` positions.
    // This allows using svext for shifted views without extra loads.
    sz_size_t const usable_step = step - 2;
    sz_size_t offset = 0;
    while (offset + step <= length) {
        svbool_t pg = svwhilelt_b8_u64(0, usable_step);             // First step-2 lanes active
        svuint8_t text0 = svld1_u8(svptrue_b8(), text_u8 + offset); // Load full step bytes

        // Fast rejection: skip if no whitespace-related bytes at all
        if (!svptest_any(pg, svmatch_u8(pg, text0, any_byte_set))) {
            offset += usable_step;
            continue;
        }

        // 1-byte whitespace: space, tab, newlines
        svbool_t one_byte_mask = svmatch_u8(pg, text0, one_byte_set);

        // Shifted views via svext - zeros fill unused lanes at end, but pg masks them out
        svuint8_t text1 = svext_u8(text0, zeros, 1);
        svuint8_t text2 = svext_u8(text0, zeros, 2);

        // 2-byte: C2 + {85, A0} (NEL, NBSP)
        svbool_t x_c2_mask = svcmpeq_n_u8(pg, text0, 0xC2);
        svbool_t x_85_mask = svcmpeq_n_u8(pg, text1, 0x85);
        svbool_t x_a0_mask = svcmpeq_n_u8(pg, text1, 0xA0);
        svbool_t two_byte_mask = svand_b_z(pg, x_c2_mask, svorr_b_z(pg, x_85_mask, x_a0_mask));

        // 3-byte: E1 9A 80 (Ogham Space Mark)
        svbool_t ogham_mask = svand_b_z(pg, svand_b_z(pg, svcmpeq_n_u8(pg, text0, 0xE1), svcmpeq_n_u8(pg, text1, 0x9A)),
                                        svcmpeq_n_u8(pg, text2, 0x80));

        // 3-byte: E2 80 XX - various Unicode spaces (U+2000-U+200D, U+2028, U+2029, U+202F)
        svbool_t x_e2_mask = svcmpeq_n_u8(pg, text0, 0xE2);
        svbool_t x_e280_mask = svand_b_z(pg, x_e2_mask, svcmpeq_n_u8(pg, text1, 0x80));
        svbool_t x_e280xx_mask = svand_b_z(pg, x_e280_mask, svmatch_u8(pg, text2, e280_third_bytes));
        // U+202F: E2 80 AF (NARROW NO-BREAK SPACE) - doesn't fit in the 16-byte set
        svbool_t nnbsp_mask = svand_b_z(pg, x_e280_mask, svcmpeq_n_u8(pg, text2, 0xAF));
        // U+205F: E2 81 9F (MEDIUM MATHEMATICAL SPACE)
        svbool_t mmsp_mask = svand_b_z(pg, svand_b_z(pg, x_e2_mask, svcmpeq_n_u8(pg, text1, 0x81)),
                                       svcmpeq_n_u8(pg, text2, 0x9F));

        // 3-byte: E3 80 80 (IDEOGRAPHIC SPACE)
        svbool_t ideographic_mask = svand_b_z(
            pg, svand_b_z(pg, svcmpeq_n_u8(pg, text0, 0xE3), svcmpeq_n_u8(pg, text1, 0x80)),
            svcmpeq_n_u8(pg, text2, 0x80));

        svbool_t three_byte_mask = svorr_b_z(pg, svorr_b_z(pg, ogham_mask, svorr_b_z(pg, x_e280xx_mask, nnbsp_mask)),
                                             svorr_b_z(pg, mmsp_mask, ideographic_mask));
        svbool_t combined_mask = svorr_b_z(pg, one_byte_mask, svorr_b_z(pg, two_byte_mask, three_byte_mask));

        if (svptest_any(pg, combined_mask)) {
            sz_size_t byte_offset = svcntp_b8(pg, svbrkb_b_z(pg, combined_mask));
            svbool_t at_pos = svcmpeq_n_u8(svptrue_b8(), svindex_u8(0, 1), (sz_u8_t)byte_offset);
            sz_size_t has_two_byte = svptest_any(at_pos, two_byte_mask);
            sz_size_t has_three_byte = svptest_any(at_pos, three_byte_mask);
            sz_size_t length_value = 1;
            length_value += has_two_byte | has_three_byte;
            length_value += has_three_byte;
            *matched_length = length_value;
            return (sz_cptr_t)(text_u8 + offset + byte_offset);
        }
        offset += usable_step;
    }

    // Handle remaining bytes with serial fallback
    return sz_utf8_find_whitespace_serial(text + offset, length - offset, matched_length);
}

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif

/*  UAX-29 word boundary detection is an inherently sequential state machine whose decision at each
 *  candidate position depends on the Word_Break properties of the surrounding runes (WB3 CR/LF, WB4
 *  Extend/Format/ZWJ skipping, WB6/WB7 mid-letter look-around, WB11/WB12 numeric, WB15/WB16 Regional
 *  Indicator pairing). There is no byte-exact slice a vector pass can accelerate cheaply, so the
 *  sve2 symbols delegate to their `_serial` references for value-identical output. */
SZ_PUBLIC sz_size_t sz_utf8_word_find_boundaries_sve2(sz_cptr_t text, sz_size_t length, sz_size_t *word_starts,
                                                      sz_size_t *word_lengths, sz_size_t words_capacity,
                                                      sz_size_t *bytes_consumed) {
    return sz_utf8_word_find_boundaries_serial(text, length, word_starts, word_lengths, words_capacity, bytes_consumed);
}

SZ_PUBLIC sz_size_t sz_utf8_word_rfind_boundaries_sve2(sz_cptr_t text, sz_size_t length, sz_size_t *word_starts,
                                                       sz_size_t *word_lengths, sz_size_t words_capacity,
                                                       sz_size_t *bytes_consumed) {
    return sz_utf8_word_rfind_boundaries_serial(text, length, word_starts, word_lengths, words_capacity,
                                                bytes_consumed);
}
#endif // SZ_USE_SVE2

#ifdef __cplusplus
}
#endif

#endif // STRINGZILLA_UTF8_ITERATE_SVE2_H_
