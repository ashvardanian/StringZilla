/**
 *  @brief Ice Lake (AVX-512 VBMI+VAES) backend for UTF-8 traversal.
 *  @file include/stringzilla/utf8_iterate/icelake.h
 *  @author Ash Vardanian
 *  @sa utf8.h
 */
#ifndef STRINGZILLA_UTF8_ITERATE_ICELAKE_H_
#define STRINGZILLA_UTF8_ITERATE_ICELAKE_H_

#include "stringzilla/types.h"
#include "stringzilla/utf8_iterate/serial.h"

#ifdef __cplusplus
extern "C" {
#endif

#if SZ_USE_ICELAKE
#if defined(__clang__)
#pragma clang attribute push(                                                                          \
    __attribute__((target("avx,avx512f,avx512vl,avx512bw,avx512dq,avx512vbmi,avx512vbmi2,bmi,bmi2"))), \
    apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("avx", "avx512f", "avx512vl", "avx512bw", "avx512dq", "avx512vbmi", "avx512vbmi2", "bmi", "bmi2")
#endif

SZ_PUBLIC sz_cptr_t sz_utf8_find_newline_icelake(sz_cptr_t text, sz_size_t length, sz_size_t *matched_length) {

    // We need to check if the ASCII chars in [10,13] (same as '\n', '\v', '\f', '\r') are present.
    // The last one - '\r' - needs special handling to differentiate between "\r" and "\r\n".
    sz_u512_vec_t newline_vec, v_vec, f_vec, r_vec;
    newline_vec.zmm = _mm512_set1_epi8('\n');
    v_vec.zmm = _mm512_set1_epi8('\v');
    f_vec.zmm = _mm512_set1_epi8('\f');
    r_vec.zmm = _mm512_set1_epi8('\r');

    // We also need to match the 2-byte newline character 0xC285 (NEL),
    // as well as the 3-byte characters 0xE280A8 (PS) and 0xE280A9 (LS).
    sz_u512_vec_t x_c2_vec, x_85_vec, x_e2_vec, x_80_vec, x_a8_vec, x_a9_vec;
    x_c2_vec.zmm = _mm512_set1_epi8('\xC2');
    x_85_vec.zmm = _mm512_set1_epi8('\x85');
    x_e2_vec.zmm = _mm512_set1_epi8('\xE2');
    x_80_vec.zmm = _mm512_set1_epi8('\x80');
    x_a8_vec.zmm = _mm512_set1_epi8('\xA8');
    x_a9_vec.zmm = _mm512_set1_epi8('\xA9');

    // We check 64 bytes of data at once, but only step forward by 62 bytes for split-register matches.
    sz_u512_vec_t text_vec;
    while (length >= 64) {
        text_vec.zmm = _mm512_loadu_epi8(text);

        // 1-byte indicators & matches
        __mmask64 newline_mask = _mm512_cmpeq_epi8_mask(text_vec.zmm, newline_vec.zmm);
        __mmask64 v_mask = _mm512_cmpeq_epi8_mask(text_vec.zmm, v_vec.zmm);
        __mmask64 f_mask = _mm512_cmpeq_epi8_mask(text_vec.zmm, f_vec.zmm);
        __mmask64 r_mask = _mm512_mask_cmpeq_epi8_mask(0x7FFFFFFFFFFFFFFF, text_vec.zmm, r_vec.zmm); // Ignore last
        sz_u64_t one_byte_mask = _cvtmask64_u64(
            _kor_mask64(_kor_mask64(newline_mask, v_mask), _kor_mask64(f_mask, r_mask)));

        // 2-byte indicators
        __mmask64 x_c2_mask = _mm512_cmpeq_epi8_mask(text_vec.zmm, x_c2_vec.zmm);
        __mmask64 x_85_mask = _mm512_cmpeq_epi8_mask(text_vec.zmm, x_85_vec.zmm);
        __mmask64 x_e2_mask = _mm512_cmpeq_epi8_mask(text_vec.zmm, x_e2_vec.zmm);
        __mmask64 x_80_mask = _mm512_cmpeq_epi8_mask(text_vec.zmm, x_80_vec.zmm);
        __mmask64 x_a8_mask = _mm512_cmpeq_epi8_mask(text_vec.zmm, x_a8_vec.zmm);
        __mmask64 x_a9_mask = _mm512_cmpeq_epi8_mask(text_vec.zmm, x_a9_vec.zmm);

        // 2-byte matches
        __mmask64 rn_match_mask = _kand_mask64(r_mask, _kshiftri_mask64(newline_mask, 1));
        __mmask64 x_c285_mask = _kand_mask64(x_c2_mask, _kshiftri_mask64(x_85_mask, 1));
        sz_u64_t two_byte_mask = _cvtmask64_u64(_kor_mask64(rn_match_mask, x_c285_mask));

        // 3-byte matches
        __mmask64 x_e280_mask = _kand_mask64(x_e2_mask, _kshiftri_mask64(x_80_mask, 1));
        __mmask64 x_e280a8_mask = _kand_mask64(x_e280_mask, _kshiftri_mask64(x_a8_mask, 2));
        __mmask64 x_e280a9_mask = _kand_mask64(x_e280_mask, _kshiftri_mask64(x_a9_mask, 2));
        sz_u64_t three_byte_mask = _cvtmask64_u64(_kor_mask64(x_e280a8_mask, x_e280a9_mask));

        // Find the earliest match regardless of length
        sz_u64_t combined_mask = one_byte_mask | two_byte_mask | three_byte_mask;
        if (combined_mask) {
            int first_offset = sz_u64_ctz(combined_mask);
            sz_u64_t first_match_mask = (sz_u64_t)(1) << first_offset;

            // We don't want to produce too much divergent control flow,
            // but need to achieve a behavior similar to this:
            //
            //  if (first_match_mask & three_byte_mask) { *matched_length = 3; }
            //  else if (first_match_mask & two_byte_mask) { *matched_length = 2; }
            //  else { *matched_length = 1; }
            sz_size_t length_value = 1;
            length_value += (first_match_mask & (two_byte_mask | three_byte_mask)) != 0;
            length_value += (first_match_mask & (three_byte_mask)) != 0;
            *matched_length = length_value;
            return text + first_offset;
        }
        else {
            text += 62;
            length -= 62;
        }
    }

    return sz_utf8_find_newline_serial(text, length, matched_length);
}

SZ_PUBLIC sz_cptr_t sz_utf8_find_whitespace_icelake(sz_cptr_t text, sz_size_t length, sz_size_t *matched_length) {

    // We need to check if the ASCII chars in [9,13] (same as '\t', '\n', '\v', '\f', '\r') are present.
    // There is also the canonical space ' ' (0x20).
    sz_u512_vec_t t_vec, r_vec, x_20_vec;
    t_vec.zmm = _mm512_set1_epi8('\t');
    r_vec.zmm = _mm512_set1_epi8('\r');
    x_20_vec.zmm = _mm512_set1_epi8(' ');

    // We also need to match the 2-byte characters 0xC285 (NEL) and 0xC2A0 (NBSP),
    sz_u512_vec_t x_c2_vec, x_85_vec, x_a0_vec;
    x_c2_vec.zmm = _mm512_set1_epi8('\xC2');
    x_85_vec.zmm = _mm512_set1_epi8('\x85');
    x_a0_vec.zmm = _mm512_set1_epi8('\xA0');

    // We also need to match 3-byte ogham space mark 0xE19A80 (OGHAM SPACE MARK),
    // a range of 3-byte characters from 0xE28080 to 0xE2808D (various spaces),
    // U+202F (0xE280AF), U+205F (0xE2819F),
    // U+2028 (0xE280A8) LINE SEPARATOR, U+2029 (0xE280A9) PARAGRAPH SEPARATOR,
    // and the 3-byte ideographic space 0xE38080 (IDEOGRAPHIC SPACE).
    sz_u512_vec_t x_e1_vec, x_e2_vec, x_e3_vec,           // ? possible first byte values
        x_9a_vec, x_80_vec, x_81_vec,                     // ? possible second byte values
        x_8d_vec, x_a8_vec, x_a9_vec, x_af_vec, x_9f_vec; // ? third byte values for ranges and specific matches
    x_e1_vec.zmm = _mm512_set1_epi8('\xE1');
    x_e2_vec.zmm = _mm512_set1_epi8('\xE2');
    x_e3_vec.zmm = _mm512_set1_epi8('\xE3');
    x_9a_vec.zmm = _mm512_set1_epi8('\x9A');
    x_80_vec.zmm = _mm512_set1_epi8('\x80');
    x_81_vec.zmm = _mm512_set1_epi8('\x81');
    x_8d_vec.zmm = _mm512_set1_epi8('\x8D');
    x_a8_vec.zmm = _mm512_set1_epi8('\xA8');
    x_a9_vec.zmm = _mm512_set1_epi8('\xA9');
    x_af_vec.zmm = _mm512_set1_epi8('\xAF');
    x_9f_vec.zmm = _mm512_set1_epi8('\x9F');

    // We check 64 bytes of data at once, but only step forward by 62 bytes for split-register matches.
    sz_u512_vec_t text_vec;
    while (length >= 64) {
        text_vec.zmm = _mm512_loadu_epi8(text);

        // 1-byte indicators & matches
        // Range [9,13] covers \t, \n, \v, \f, \r
        __mmask64 x_20_mask = _mm512_cmpeq_epi8_mask(text_vec.zmm, x_20_vec.zmm);
        __mmask64 t_mask = _mm512_cmpge_epu8_mask(text_vec.zmm, t_vec.zmm);
        __mmask64 r_mask = _mm512_cmple_epu8_mask(text_vec.zmm, r_vec.zmm);
        sz_u64_t one_byte_mask = _cvtmask64_u64(_kor_mask64(x_20_mask, _kand_mask64(t_mask, r_mask)));

        // Instead of immediately checking for 2-byte and 3-byte matches with a ridiculous number of masks and
        // comparisons, let's define a "fast path" for following cases:
        // - no whitespaces are found in the range
        // - a one-byte match comes before any possible prefix byte of a multi-byte match
        __mmask64 x_c2_mask = _mm512_mask_cmpeq_epi8_mask(0x7FFFFFFFFFFFFFFF, text_vec.zmm, x_c2_vec.zmm);
        __mmask64 x_e1_mask = _mm512_mask_cmpeq_epi8_mask(0x3FFFFFFFFFFFFFFF, text_vec.zmm, x_e1_vec.zmm);
        __mmask64 x_e2_mask = _mm512_mask_cmpeq_epi8_mask(0x3FFFFFFFFFFFFFFF, text_vec.zmm, x_e2_vec.zmm);
        __mmask64 x_e3_mask = _mm512_mask_cmpeq_epi8_mask(0x3FFFFFFFFFFFFFFF, text_vec.zmm, x_e3_vec.zmm);

        // Check if we matched the "fast path"
        if (one_byte_mask) {
            sz_u64_t prefix_byte_mask = _cvtmask64_u64(
                _kor_mask64(_kor_mask64(x_c2_mask, x_e1_mask), _kor_mask64(x_e2_mask, x_e3_mask)));
            if (prefix_byte_mask) {
                int first_one_byte_offset = sz_u64_ctz(one_byte_mask);
                int first_prefix_offset = sz_u64_ctz(prefix_byte_mask);
                if (first_one_byte_offset < first_prefix_offset) {
                    *matched_length = 1;
                    return text + first_one_byte_offset;
                }
            }
            else {
                int first_one_byte_offset = sz_u64_ctz(one_byte_mask);
                *matched_length = 1;
                return text + first_one_byte_offset;
            }
        }

        // 2-byte indicators suffixes & matches
        __mmask64 x_85_mask = _mm512_cmpeq_epi8_mask(text_vec.zmm, x_85_vec.zmm);
        __mmask64 x_a0_mask = _mm512_cmpeq_epi8_mask(text_vec.zmm, x_a0_vec.zmm);
        sz_u64_t two_byte_mask = _cvtmask64_u64(             //
            _kand_mask64(x_c2_mask,                          //
                         _kor_mask64(                        //
                             _kshiftri_mask64(x_85_mask, 1), // U+0085 NEL
                             _kshiftri_mask64(x_a0_mask, 1)  // U+00A0 NBSP
                             )));

        // 3-byte indicators suffixes
        __mmask64 x_9a_mask = _mm512_cmpeq_epi8_mask(text_vec.zmm, x_9a_vec.zmm);
        __mmask64 x_80_mask = _mm512_cmpeq_epi8_mask(text_vec.zmm, x_80_vec.zmm);
        __mmask64 x_81_mask = _mm512_cmpeq_epi8_mask(text_vec.zmm, x_81_vec.zmm);
        __mmask64 x_80_ge_mask = _mm512_cmpge_epu8_mask(text_vec.zmm, x_80_vec.zmm);
        __mmask64 x_8d_le_mask = _mm512_cmple_epu8_mask(text_vec.zmm, x_8d_vec.zmm);
        __mmask64 x_a8_mask = _mm512_cmpeq_epi8_mask(text_vec.zmm, x_a8_vec.zmm);
        __mmask64 x_a9_mask = _mm512_cmpeq_epi8_mask(text_vec.zmm, x_a9_vec.zmm);
        __mmask64 x_af_mask = _mm512_cmpeq_epi8_mask(text_vec.zmm, x_af_vec.zmm);
        __mmask64 x_9f_mask = _mm512_cmpeq_epi8_mask(text_vec.zmm, x_9f_vec.zmm);

        // 3-byte matches
        __mmask64 ogham_mask = _kand_mask64(
            x_e1_mask, _kand_mask64(_kshiftri_mask64(x_9a_mask, 1), _kshiftri_mask64(x_80_mask, 2)));
        // U+2000 to U+200D: E2 80 [80-8D]
        __mmask64 range_e280_mask = _kand_mask64(
            x_e2_mask, _kand_mask64(_kshiftri_mask64(x_80_mask, 1), _kand_mask64(_kshiftri_mask64(x_80_ge_mask, 2),
                                                                                 _kshiftri_mask64(x_8d_le_mask, 2))));
        // U+202F: E2 80 AF (NARROW NO-BREAK SPACE)
        __mmask64 nnbsp_mask = _kand_mask64(
            x_e2_mask, _kand_mask64(_kshiftri_mask64(x_80_mask, 1), _kshiftri_mask64(x_af_mask, 2)));
        // U+205F: E2 81 9F (MEDIUM MATHEMATICAL SPACE)
        __mmask64 mmsp_mask = _kand_mask64(
            x_e2_mask, _kand_mask64(_kshiftri_mask64(x_81_mask, 1), _kshiftri_mask64(x_9f_mask, 2)));
        // U+2028: E2 80 A8 (LINE SEPARATOR)
        __mmask64 line_mask = _kand_mask64(
            x_e2_mask, _kand_mask64(_kshiftri_mask64(x_80_mask, 1), _kshiftri_mask64(x_a8_mask, 2)));
        // U+2029: E2 80 A9 (PARAGRAPH SEPARATOR)
        __mmask64 paragraph_mask = _kand_mask64(
            x_e2_mask, _kand_mask64(_kshiftri_mask64(x_80_mask, 1), _kshiftri_mask64(x_a9_mask, 2)));
        __mmask64 ideographic_mask = _kand_mask64(
            x_e3_mask, _kand_mask64(_kshiftri_mask64(x_80_mask, 1), _kshiftri_mask64(x_80_mask, 2)));
        sz_u64_t three_byte_mask = _cvtmask64_u64(_kor_mask64(
            _kor_mask64(_kor_mask64(_kor_mask64(ogham_mask, range_e280_mask), _kor_mask64(nnbsp_mask, mmsp_mask)),
                        _kor_mask64(line_mask, paragraph_mask)),
            ideographic_mask));

        // Find the earliest match regardless of length
        sz_u64_t combined_mask = one_byte_mask | two_byte_mask | three_byte_mask;
        if (combined_mask) {
            int first_offset = sz_u64_ctz(combined_mask);
            sz_u64_t first_match_mask = (sz_u64_t)(1) << first_offset;

            // We don't want to produce too much divergent control flow,
            // but need to achieve a behavior similar to this:
            //
            //  if (first_match_mask & three_byte_mask) { *matched_length = 3; }
            //  else if (first_match_mask & two_byte_mask) { *matched_length = 2; }
            //  else { *matched_length = 1; }
            sz_size_t length_value = 1;
            length_value += (first_match_mask & (two_byte_mask | three_byte_mask)) != 0;
            length_value += (first_match_mask & (three_byte_mask)) != 0;
            *matched_length = length_value;
            return text + first_offset;
        }
        else {
            text += 62;
            length -= 62;
        }
    }

    return sz_utf8_find_whitespace_serial(text, length, matched_length);
}

SZ_PUBLIC sz_size_t sz_utf8_count_icelake(sz_cptr_t text, sz_size_t length) {
    // UTF-8 character counting strategy:
    // Count every byte that is NOT a continuation byte (i.e., character start bytes).
    //
    // UTF-8 byte patterns:
    //   ASCII:        0xxxxxxx (0x00-0x7F)  - single byte character
    //   Start 2-byte: 110xxxxx (0xC0-0xDF)  - first byte of 2-byte sequence
    //   Start 3-byte: 1110xxxx (0xE0-0xEF)  - first byte of 3-byte sequence
    //   Start 4-byte: 11110xxx (0xF0-0xF7)  - first byte of 4-byte sequence
    //   Continuation: 10xxxxxx (0x80-0xBF)  - continuation byte (NOT a character start)
    //
    // To detect continuation bytes: (byte & 0xC0) == 0x80
    //   0xC0 = 11000000  - masks the top 2 bits
    //   0x80 = 10000000  - pattern for continuation bytes after masking

    sz_u512_vec_t continuation_mask_vec, continuation_pattern_vec;
    continuation_mask_vec.zmm = _mm512_set1_epi8((char)0xC0);    // 0xC0 = 0b11000000 - mask top 2 bits
    continuation_pattern_vec.zmm = _mm512_set1_epi8((char)0x80); // 0x80 = 0b10000000 - continuation pattern

    sz_u8_t const *text_u8 = (sz_u8_t const *)text;
    sz_size_t char_count = 0;

    // Process 64 bytes at a time
    sz_u512_vec_t text_vec, headers_vec;
    while (length >= 64) {
        text_vec.zmm = _mm512_loadu_epi8(text_u8);

        // Apply mask (byte & 0xC0) to extract top 2 bits of each byte
        headers_vec.zmm = _mm512_and_si512(text_vec.zmm, continuation_mask_vec.zmm);

        // Compare with 0x80 (0b10000000) to find continuation bytes
        sz_u64_t start_byte_mask = _cvtmask64_u64(
            _mm512_cmpneq_epi8_mask(headers_vec.zmm, continuation_pattern_vec.zmm));

        // Count non-continuation bytes (i.e., character starts)
        char_count += _mm_popcnt_u64(start_byte_mask);
        text_u8 += 64;
        length -= 64;
    }

    // Process remaining bytes with a masked variant
    if (length) {
        __mmask64 load_mask = sz_u64_mask_until_(length);
        text_vec.zmm = _mm512_maskz_loadu_epi8(load_mask, text_u8);
        headers_vec.zmm = _mm512_and_si512(text_vec.zmm, continuation_mask_vec.zmm);
        __mmask64 start_byte_mask = _mm512_mask_cmpneq_epi8_mask(load_mask, headers_vec.zmm,
                                                                 continuation_pattern_vec.zmm);
        char_count += _mm_popcnt_u64(_cvtmask64_u64(start_byte_mask));
    }
    return char_count;
}

SZ_PUBLIC sz_cptr_t sz_utf8_find_nth_icelake(sz_cptr_t text, sz_size_t length, sz_size_t n) {

    // The logic of this function is similar to `sz_utf8_count_icelake`, but uses PDEP & PEXT
    // instructions in the inner loop to locate Nth character start byte efficiently
    // without one more loop.
    sz_u512_vec_t continuation_mask_vec, continuation_pattern_vec;
    continuation_mask_vec.zmm = _mm512_set1_epi8((char)0xC0);
    continuation_pattern_vec.zmm = _mm512_set1_epi8((char)0x80);

    sz_u8_t const *text_u8 = (sz_u8_t const *)text;

    // Process 64 bytes at a time
    sz_u512_vec_t text_vec, headers_vec;
    while (length >= 64) {
        text_vec.zmm = _mm512_loadu_epi8(text_u8);

        // Apply mask (byte & 0xC0) to extract top 2 bits of each byte
        headers_vec.zmm = _mm512_and_si512(text_vec.zmm, continuation_mask_vec.zmm);

        // Compare with 0x80 (0b10000000) to find continuation bytes
        sz_u64_t start_byte_mask = _cvtmask64_u64(
            _mm512_cmpneq_epi8_mask(headers_vec.zmm, continuation_pattern_vec.zmm));
        sz_size_t start_byte_count = _mm_popcnt_u64(start_byte_mask);

        // Check if we've reached the terminal part of our search
        if (n < start_byte_count) {
            // PDEP directly gives us the nth set bit position
            // Example: _pdep_u64(0b10, 0b0001010100) = 0b0000010000
            sz_u64_t deposited_bits = _pdep_u64((sz_u64_t)1 << n, start_byte_mask);
            int byte_offset = sz_u64_ctz(deposited_bits);
            return (sz_cptr_t)(text_u8 + byte_offset);
        }
        // Jump to the next block
        else {
            n -= start_byte_count;
            text_u8 += 64;
            length -= 64;
        }
    }

    // Process remaining bytes with serial
    return sz_utf8_find_nth_serial((sz_cptr_t)text_u8, length, n);
}

SZ_PUBLIC sz_cptr_t sz_utf8_unpack_chunk_icelake( //
    sz_cptr_t text, sz_size_t length,             //
    sz_rune_t *runes, sz_size_t runes_capacity,   //
    sz_size_t *runes_unpacked) {

    // Filter out obsolte calls
    if (!runes_capacity || !length) return text;

    // Process up to the minimum of: available bytes, (output capacity * 4), or optimal chunk size (64)
    sz_size_t chunk_size = sz_min_of_three(length, runes_capacity * 4, 64);
    sz_u512_vec_t text_vec, runes_vec;
    __mmask64 load_mask = sz_u64_mask_until_(chunk_size);
    text_vec.zmm = _mm512_maskz_loadu_epi8(load_mask, (sz_u8_t const *)text);
    __mmask64 is_non_ascii = _mm512_movepi8_mask(text_vec.zmm);

    // Check if its our lucky day and we have an entire register worth of ASCII text,
    // that we will output into runes directly. English is responsible for roughly 60% of the text
    // on the Internet, so this will often be our primary execution path.
    if (is_non_ascii == 0) {
        // For ASCII, 1 byte = 1 rune, so limit to runes_capacity
        sz_size_t runes_to_unpack = sz_min_of_two(chunk_size, runes_capacity);
        _mm512_mask_storeu_epi32(runes, sz_u16_clamp_mask_until_(runes_to_unpack),
                                 _mm512_cvtepu8_epi32(_mm512_castsi512_si128(text_vec.zmm)));
        if (runes_to_unpack > 16)
            _mm512_mask_storeu_epi32(runes + 16, sz_u16_clamp_mask_until_(runes_to_unpack - 16),
                                     _mm512_cvtepu8_epi32(_mm512_extracti32x4_epi32(text_vec.zmm, 1)));
        if (runes_to_unpack > 32)
            _mm512_mask_storeu_epi32(runes + 32, sz_u16_clamp_mask_until_(runes_to_unpack - 32),
                                     _mm512_cvtepu8_epi32(_mm512_extracti32x4_epi32(text_vec.zmm, 2)));
        if (runes_to_unpack > 48)
            _mm512_mask_storeu_epi32(runes + 48, sz_u16_clamp_mask_until_(runes_to_unpack - 48),
                                     _mm512_cvtepu8_epi32(_mm512_extracti32x4_epi32(text_vec.zmm, 3)));
        *runes_unpacked = runes_to_unpack;
        return text + runes_to_unpack;
    }

    // Russian, Spanish, German, and French are the 2nd, 3rd, 4th, and 5th most common languages on the Internet,
    // and all of them are composed of a mixture of 2-byte and 1-byte UTF-8 characters. When dealing with such text
    // we plan the algorithm with respect to the number of decoded entries we can fit in a single output register.
    // We don't need to validate the UTF-8 encoding, just classify the inputs to locate the first 3- or 4-byte
    // character in the input:
    // - ASCII: bit 7 = 0, i.e., 0xxxxxxx (0x00-0x7F)
    // - 2-byte lead: bits 7-5 = 110, i.e., 110xxxxx (0xC0-0xDF)
    // - Continuation: bits 7-6 = 10, i.e., 10xxxxxx (0x80-0xBF)
    __mmask64 is_ascii = ~is_non_ascii & load_mask;
    __mmask64 is_two_byte_start = _mm512_mask_cmpeq_epi8_mask(
        load_mask, _mm512_and_si512(text_vec.zmm, _mm512_set1_epi8((char)0xE0)), _mm512_set1_epi8((char)0xC0));
    __mmask64 is_continuation = _mm512_mask_cmpeq_epi8_mask(
        load_mask, _mm512_and_si512(text_vec.zmm, _mm512_set1_epi8((char)0xC0)), _mm512_set1_epi8((char)0x80));

    // Find longest prefix containing only ASCII and complete 2-byte sequences - let's call it the "Mixed 12" case
    __mmask64 is_expected_continuation = is_two_byte_start << 1;
    __mmask64 is_valid_mixed12 = is_ascii | is_two_byte_start | (is_continuation & is_expected_continuation);
    sz_size_t mixed12_prefix_length = sz_u64_ctz(~is_valid_mixed12 | ~load_mask);
    mixed12_prefix_length -= mixed12_prefix_length && ((is_two_byte_start >> (mixed12_prefix_length - 1)) & 1);

    if (mixed12_prefix_length >= 2) {
        __mmask64 prefix_mask = sz_u64_mask_until_(mixed12_prefix_length);
        __mmask64 is_char_start = (is_ascii | is_two_byte_start) & prefix_mask;
        sz_size_t num_runes = (sz_size_t)sz_u64_popcount(is_char_start);
        sz_size_t runes_to_unpack = sz_min_of_three(num_runes, runes_capacity, 16);

        // Compress character start positions into sequential indices, then gather bytes
        sz_u512_vec_t char_indices;
        char_indices.zmm = _mm512_set_epi8(                                 //
            63, 62, 61, 60, 59, 58, 57, 56, 55, 54, 53, 52, 51, 50, 49, 48, //
            47, 46, 45, 44, 43, 42, 41, 40, 39, 38, 37, 36, 35, 34, 33, 32, //
            31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, //
            15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
        char_indices.zmm = _mm512_maskz_compress_epi8(is_char_start, char_indices.zmm);

        sz_u512_vec_t first_bytes, second_bytes;
        first_bytes.zmm = _mm512_permutexvar_epi8(char_indices.zmm, text_vec.zmm);
        second_bytes.zmm = _mm512_permutexvar_epi8(_mm512_add_epi8(char_indices.zmm, _mm512_set1_epi8(1)),
                                                   text_vec.zmm);

        // Expand to 32-bit and decode 2-byte sequences: ((first & 0x1F) << 6) | (second & 0x3F)
        __m512i first_bytes_wide = _mm512_cvtepu8_epi32(_mm512_castsi512_si128(first_bytes.zmm));
        __m512i second_bytes_wide = _mm512_cvtepu8_epi32(_mm512_castsi512_si128(second_bytes.zmm));
        __mmask16 is_two_byte_char = (__mmask16)_pext_u64(is_two_byte_start, is_char_start);
        __m512i decoded_two_byte = _mm512_or_si512(
            _mm512_slli_epi32(_mm512_and_si512(first_bytes_wide, _mm512_set1_epi32(0x1F)), 6),
            _mm512_and_si512(second_bytes_wide, _mm512_set1_epi32(0x3F)));

        // Blend: ASCII positions keep byte value, 2-byte positions get decoded rune
        runes_vec.zmm = _mm512_mask_blend_epi32(is_two_byte_char, first_bytes_wide, decoded_two_byte);
        _mm512_mask_storeu_epi32(runes, sz_u16_mask_until_(runes_to_unpack), runes_vec.zmm);

        // Bytes consumed: one per ASCII, two per 2-byte sequence
        sz_size_t two_byte_count = (sz_size_t)sz_u64_popcount(is_two_byte_char & sz_u16_mask_until_(runes_to_unpack));
        *runes_unpacked = runes_to_unpack;
        return text + runes_to_unpack + two_byte_count;
    }

    // Check for the number of 3-byte characters - in this case we can't easily cast to 16-bit integers
    // and check for equality, but we can pre-define the masks and values we expect at each byte position.
    // For 3-byte UTF-8 sequences, we check if bytes match the pattern: 1110xxxx 10xxxxxx 10xxxxxx
    // We need to check every 3rd byte starting from position 0.
    sz_u512_vec_t three_byte_mask_vec, three_byte_pattern_vec;
    three_byte_mask_vec.zmm = _mm512_set1_epi32(0x00C0C0F0);    // Mask: [F0, C0, C0, 00] per 4-byte slot
    three_byte_pattern_vec.zmm = _mm512_set1_epi32(0x008080E0); // Pattern: [E0, 80, 80, 00] per 4-byte slot

    // Create permutation indices to gather 3-byte sequences into 4-byte slots
    // Input:  [b0 b1 b2]    [b3 b4 b5]    [b6 b7 b8]    ... (up to 16 triplets from 48 bytes)
    // Output: [b0 b1 b2 XX] [b3 b4 b5 XX] [b6 b7 b8 XX] ... (16 slots, 4th byte zeroed)
    sz_u512_vec_t permute_indices;
    permute_indices.zmm = _mm512_setr_epi32(
        // Triplets 0-3:  [0,1,2,_] [3,4,5,_] [6,7,8,_] [9,10,11,_]
        0x40020100, 0x40050403, 0x40080706, 0x400B0A09,
        // Triplets 4-7:  [12,13,14,_] [15,16,17,_] [18,19,20,_] [21,22,23,_]
        0x400E0D0C, 0x40111010, 0x40141312, 0x40171615,
        // Triplets 8-11: [24,25,26,_] [27,28,29,_] [30,31,32,_] [33,34,35,_]
        0x401A1918, 0x401D1C1B, 0x40201F1E, 0x40232221,
        // Triplets 12-15: [36,37,38,_] [39,40,41,_] [42,43,44,_] [45,46,47,_]
        0x40262524, 0x40292827, 0x402C2B2A, 0x402F2E2D);

    // Permute to gather triplets into slots
    sz_u512_vec_t gathered_triplets;
    gathered_triplets.zmm = _mm512_permutexvar_epi8(permute_indices.zmm, text_vec.zmm);

    // Check if gathered bytes match 3-byte UTF-8 pattern
    sz_u512_vec_t masked_triplets;
    masked_triplets.zmm = _mm512_and_si512(gathered_triplets.zmm, three_byte_mask_vec.zmm);
    __mmask16 three_byte_match_mask = _mm512_cmpeq_epi32_mask(masked_triplets.zmm, three_byte_pattern_vec.zmm);
    sz_size_t three_byte_prefix_length = sz_u64_ctz(~three_byte_match_mask);

    if (three_byte_prefix_length) {
        // Unpack up to 16 three-byte characters (48 bytes of input).
        sz_size_t runes_to_place = sz_min_of_three(three_byte_prefix_length, 16, runes_capacity);
        // Decode: ((b0 & 0x0F) << 12) | ((b1 & 0x3F) << 6) | (b2 & 0x3F)
        // gathered_triplets has: [b0, b1, b2, XX] in each 32-bit slot (little-endian: 0xXXb2b1b0)
        // Extract: b0 from bits 7-0, b1 from bits 15-8, b2 from bits 23-16
        runes_vec.zmm = _mm512_or_si512(
            _mm512_or_si512(
                // (b0 & 0x0F) << 12
                _mm512_slli_epi32(_mm512_and_si512(gathered_triplets.zmm, _mm512_set1_epi32(0x0FU)), 12),
                // (b1 & 0x3F) << 6
                _mm512_slli_epi32(
                    _mm512_and_si512(_mm512_srli_epi32(gathered_triplets.zmm, 8), _mm512_set1_epi32(0x3FU)), 6)),
            _mm512_and_si512(_mm512_srli_epi32(gathered_triplets.zmm, 16), _mm512_set1_epi32(0x3FU))); // (b2 & 0x3F)
        _mm512_mask_storeu_epi32(runes, sz_u16_mask_until_(runes_to_place), runes_vec.zmm);
        *runes_unpacked = runes_to_place;
        return text + runes_to_place * 3;
    }

    // Check for the number of 4-byte characters
    // For 4-byte UTF-8 sequences: 11110xxx 10xxxxxx 10xxxxxx 10xxxxxx
    // With a homogeneous 4-byte prefix, we have perfect 4-byte alignment (up to 16 sequences in 64 bytes)
    sz_u512_vec_t four_byte_mask_vec, four_byte_pattern_vec;
    four_byte_mask_vec.zmm = _mm512_set1_epi32((int)0xC0C0C0F8);    // Mask: [F8, C0, C0, C0] per 4-byte slot
    four_byte_pattern_vec.zmm = _mm512_set1_epi32((int)0x808080F0); // Pattern: [F0, 80, 80, 80] per 4-byte slot

    // Mask and check for 4-byte pattern in each 32-bit slot
    sz_u512_vec_t masked_quads;
    masked_quads.zmm = _mm512_and_si512(text_vec.zmm, four_byte_mask_vec.zmm);
    __mmask16 four_byte_match_mask = _mm512_cmpeq_epi32_mask(masked_quads.zmm, four_byte_pattern_vec.zmm);
    sz_size_t four_byte_prefix_length = sz_u64_ctz(~four_byte_match_mask);

    if (four_byte_prefix_length) {
        // Unpack up to 16 four-byte characters (64 bytes of input).
        sz_size_t runes_to_place = sz_min_of_three(four_byte_prefix_length, 16, runes_capacity);
        // Decode: ((b0 & 0x07) << 18) | ((b1 & 0x3F) << 12) | ((b2 & 0x3F) << 6) | (b3 & 0x3F)
        runes_vec.zmm = _mm512_or_si512(
            _mm512_or_si512(
                // (b0 & 0x07) << 18
                _mm512_slli_epi32(_mm512_and_si512(text_vec.zmm, _mm512_set1_epi32(0x07U)), 18),
                // (b1 & 0x3F) << 12
                _mm512_slli_epi32(_mm512_and_si512(_mm512_srli_epi32(text_vec.zmm, 8), _mm512_set1_epi32(0x3FU)), 12)),
            _mm512_or_si512(
                // (b2 & 0x3F) << 6
                _mm512_slli_epi32(_mm512_and_si512(_mm512_srli_epi32(text_vec.zmm, 16), _mm512_set1_epi32(0x3FU)), 6),
                // (b3 & 0x3F)
                _mm512_and_si512(_mm512_srli_epi32(text_vec.zmm, 24), _mm512_set1_epi32(0x3FU))));
        _mm512_mask_storeu_epi32(runes, sz_u16_mask_until_(runes_to_place), runes_vec.zmm);
        *runes_unpacked = runes_to_place;
        return text + runes_to_place * 4;
    }

    // Fallback to serial for mixed/malformed content
    return sz_utf8_unpack_chunk_serial(text, length, runes, runes_capacity, runes_unpacked);
}

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif

#if defined(__clang__)
#pragma clang attribute push(                                                                          \
    __attribute__((target("avx,avx512f,avx512vl,avx512bw,avx512dq,avx512vbmi,avx512vbmi2,bmi,bmi2"))), \
    apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("avx", "avx512f", "avx512vl", "avx512bw", "avx512dq", "avx512vbmi", "avx512vbmi2", "bmi", "bmi2")
#endif

/*  UAX-29 word boundary detection (Ice Lake / AVX-512).
 *
 *  The UAX-29 word-break algorithm is a stateful machine: the decision at a candidate position can
 *  depend on Extend/Format/ZWJ runs to skip (WB4), one-step look-ahead/look-back across ignorables
 *  (WB6/7, WB11/12), and Regional-Indicator parity (WB15/16). A naive full SIMD reimplementation of
 *  all those stateful sub-rules is fragile, so instead we vectorize the *common* and *safe* part:
 *
 *  1. Vectorized property classification. For a 64-byte chunk that is pure ASCII (the dominant case
 *     for English text and source code, where every byte is its own codepoint), we look up each
 *     byte's `sz_tr29_word_break_t` class through a single `_mm512_permutexvar_epi8` over the
 *     128-entry `sz_utf8_word_break_property_ascii_` table (each rune < 0x80 maps directly).
 *
 *  2. Local "definitely-not-a-boundary" mask. From the class vector and its shift-by-one neighbor we
 *     compute, with masked compares, the set of adjacencies that the *unconditional* local rules
 *     (WB5, WB8, WB9, WB10, WB13a, WB13b) join with no possible stateful override. Because those
 *     rules `return sz_false_k` regardless of context, any position whose (prev, cur) pair matches
 *     one of them is provably NOT a boundary and can be skipped in bulk. Every other position is a
 *     *candidate* (a safe superset of the true boundaries).
 *
 *  3. Serial fixup. Each candidate position is confirmed by the serial oracle
 *     `sz_utf8_is_word_boundary_serial`, which owns all the stateful sub-rules (WB3/3a/3b/3c, WB4,
 *     WB6/7, WB7a, WB11/12, WB13, WB15/16). The result is therefore value-identical to serial while
 *     the scan over long ASCII word/number runs is driven by vector compares instead of per-byte
 *     decoding. Non-ASCII chunks fall back to the serial scan for that region.
 *
 *  This keeps `find`/`rfind` byte-for-byte and width-for-width equal to `_serial` for every input,
 *  with the ASCII/Latin common case advanced 64 bytes at a time. The stateful sub-rules listed in
 *  (3) intentionally stay serial — only classification and candidate filtering are vectorized. */

/** @brief  AVX-512 classification of an all-ASCII 64-byte vector to WB properties via table lookup. */
SZ_INTERNAL __m512i sz_utf8_word_break_classify_ascii_icelake_(__m512i ascii_bytes) {
    // The 128-entry `sz_utf8_word_break_property_ascii_` table fits in two ZMM registers; a 7-bit index (ASCII) selects
    // via `_mm512_permutexvar_epi8` from a 64-lane table, so we need a two-table blend on bit 6.
    __m512i low_table = _mm512_loadu_epi8(sz_utf8_word_break_property_ascii_);       // entries 0..63
    __m512i high_table = _mm512_loadu_epi8(sz_utf8_word_break_property_ascii_ + 64); // entries 64..127
    __mmask64 high_half = _mm512_test_epi8_mask(ascii_bytes, _mm512_set1_epi8(0x40));
    __m512i low_result = _mm512_permutexvar_epi8(ascii_bytes, low_table);
    __m512i high_result = _mm512_permutexvar_epi8(ascii_bytes, high_table);
    return _mm512_mask_blend_epi8(high_half, low_result, high_result);
}

/** @brief  Mask of class==v lanes. */
SZ_INTERNAL __mmask64 sz_utf8_word_break_class_mask_(__m512i classes, int v) {
    return _mm512_cmpeq_epi8_mask(classes, _mm512_set1_epi8((char)v));
}

/**
 *  @brief Compute, for an all-ASCII chunk, the per-position "joined" (guaranteed non-boundary) mask.
 *
 *  Bit `i` is set when the boundary before lane `i` is suppressed by a UAX-29 no-break rule. ASCII contains no
 *  Extend/Format/ZWJ/Regional_Indicator/Hebrew/Katakana, so WB4 and WB15/16 never apply and the look-around
 *  rules WB6/7/11/12 reduce to neighbour bit-shifts (`<< 1` = previous lane, `>> 1` = next, `<< 2` = two back).
 *  The result is exact for lanes whose i-2 and i+1 neighbours are in-window, so the caller needs no oracle.
 */
SZ_INTERNAL sz_u64_t sz_utf8_word_break_join_mask_ascii_(__m512i classes) {
    // ASCII has no Hebrew, so AHLetter == ALetter; reduce each class to a 64-lane bitmask and defer the rule
    // logic to the shared portable routine.
    return sz_utf8_word_break_join_from_class_masks_(
        _cvtmask64_u64(sz_utf8_word_break_class_mask_(classes, sz_tr29_word_break_aletter_k)),
        _cvtmask64_u64(sz_utf8_word_break_class_mask_(classes, sz_tr29_word_break_numeric_k)),
        _cvtmask64_u64(sz_utf8_word_break_class_mask_(classes, sz_tr29_word_break_extendnumlet_k)),
        _cvtmask64_u64(sz_utf8_word_break_class_mask_(classes, sz_tr29_word_break_midletter_k)),
        _cvtmask64_u64(sz_utf8_word_break_class_mask_(classes, sz_tr29_word_break_midnum_k)),
        _cvtmask64_u64(sz_utf8_word_break_class_mask_(classes, sz_tr29_word_break_mid_quotes_k)),
        _cvtmask64_u64(sz_utf8_word_break_class_mask_(classes, sz_tr29_word_break_cr_k)),
        _cvtmask64_u64(sz_utf8_word_break_class_mask_(classes, sz_tr29_word_break_lf_k)));
}

SZ_PUBLIC sz_size_t sz_utf8_word_find_boundaries_icelake( //
    sz_cptr_t text, sz_size_t length,                     //
    sz_size_t *word_starts, sz_size_t *word_lengths,      //
    sz_size_t words_capacity, sz_size_t *bytes_consumed) {

    sz_size_t words = 0;
    if (length == 0 || words_capacity == 0) {
        if (bytes_consumed) *bytes_consumed = 0;
        return 0;
    }

    sz_u8_t const *text_u8 = (sz_u8_t const *)text;
    sz_size_t word_start = 0; // Start of the word currently being accumulated (always a boundary).
    // Position 0 is always a boundary; the first reportable boundary is after the first codepoint.
    sz_size_t position = sz_utf8_codepoint_length_(text_u8[0]);

    // Lane identity {0,1,...,63}: compressing it by the boundary mask yields the boundary lane offsets.
    // `_mm512_set_epi8` takes lane 63 first, so the arguments descend (GCC has no `_mm512_setr_epi8`).
    __m512i const lane_identity = _mm512_set_epi8(                      //
        63, 62, 61, 60, 59, 58, 57, 56, 55, 54, 53, 52, 51, 50, 49, 48, //
        47, 46, 45, 44, 43, 42, 41, 40, 39, 38, 37, 36, 35, 34, 33, 32, //
        31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, //
        15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);

    // Oracle-free fast path: a window [position-2, position+62) gives lanes [2,62] full +/-2 context, so an
    // all-ASCII window resolves boundaries at positions [position, position+60] directly from the mask.
    while (position < length) {
        if (position >= 2 && position + 62 <= length) {
            __m512i window = _mm512_loadu_epi8(text_u8 + position - 2); // lane j = byte position-2+j
            if (_mm512_movepi8_mask(window) == 0) {
                __m512i classes = sz_utf8_word_break_classify_ascii_icelake_(window);
                sz_u64_t join = sz_utf8_word_break_join_mask_ascii_(classes);
                sz_u64_t boundary = (~join) & 0x7FFFFFFFFFFFFFFCull; // trusted lanes [2,62]
                sz_u8_t boundary_lanes[64];
                _mm512_mask_compressstoreu_epi8(boundary_lanes, boundary, lane_identity);
                sz_size_t boundary_count = (sz_size_t)_mm_popcnt_u64(boundary);
                for (sz_size_t i = 0; i < boundary_count; ++i) {
                    sz_size_t boundary_position = position - 2 + (sz_size_t)boundary_lanes[i];
                    if (words == words_capacity) {
                        if (bytes_consumed) *bytes_consumed = word_start;
                        return words;
                    }
                    word_starts[words] = word_start;
                    word_lengths[words] = boundary_position - word_start;
                    ++words;
                    word_start = boundary_position;
                }
                position += 61; // Resolved [position, position+60]; next unresolved boundary is at position+61.
                continue;
            }
        }
        // Scalar step (non-ASCII window, or near the leading/trailing edges).
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

SZ_PUBLIC sz_size_t sz_utf8_word_rfind_boundaries_icelake( //
    sz_cptr_t text, sz_size_t length,                      //
    sz_size_t *word_starts, sz_size_t *word_lengths,       //
    sz_size_t words_capacity, sz_size_t *bytes_consumed) {

    sz_size_t words = 0;
    if (length == 0 || words_capacity == 0) {
        if (bytes_consumed) *bytes_consumed = length;
        return 0;
    }

    sz_u8_t const *text_u8 = (sz_u8_t const *)text;
    sz_size_t word_end = length; // End of the word currently being accumulated (always a boundary).
    // Position `length` is always a boundary; step back one codepoint to the first reportable position.
    sz_size_t position = length - 1;
    while (position > 0 && (text_u8[position] & 0xC0) == 0x80) position--;

    // `_mm512_set_epi8` takes lane 63 first, so the arguments descend (GCC has no `_mm512_setr_epi8`).
    __m512i const lane_identity = _mm512_set_epi8(63, 62, 61, 60, 59, 58, 57, 56, 55, 54, 53, 52, 51, 50, 49, 48, 47, 46,
                                                  45, 44, 43, 42, 41, 40, 39, 38, 37, 36, 35, 34, 33, 32, 31, 30, 29, 28,
                                                  27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10,
                                                  9, 8, 7, 6, 5, 4, 3, 2, 1, 0);

    while (position > 0) {
        // Oracle-free fast path: a window [position-62, position+2) gives lanes [2,62] full +/-2 context,
        // resolving boundaries at positions [position-60, position]; emit them high-to-low.
        if (position >= 62 && position + 2 <= length) {
            sz_size_t base = position - 62; // lane j = byte base+j; trusted lanes [2,62] → [position-60, position]
            __m512i window = _mm512_loadu_epi8(text_u8 + base);
            if (_mm512_movepi8_mask(window) == 0) {
                __m512i classes = sz_utf8_word_break_classify_ascii_icelake_(window);
                sz_u64_t join = sz_utf8_word_break_join_mask_ascii_(classes);
                sz_u64_t boundary = (~join) & 0x7FFFFFFFFFFFFFFCull; // trusted lanes [2,62]
                sz_u8_t boundary_lanes[64];
                _mm512_mask_compressstoreu_epi8(boundary_lanes, boundary, lane_identity);
                sz_size_t boundary_count = (sz_size_t)_mm_popcnt_u64(boundary);
                for (sz_size_t i = boundary_count; i-- > 0;) { // descending position (highest lane first)
                    sz_size_t boundary_position = base + (sz_size_t)boundary_lanes[i];
                    if (words == words_capacity) {
                        if (bytes_consumed) *bytes_consumed = word_end;
                        return words;
                    }
                    word_starts[words] = boundary_position;
                    word_lengths[words] = word_end - boundary_position;
                    ++words;
                    word_end = boundary_position;
                }
                position = base + 1; // Resolved down to position-60; next unresolved boundary is at position-61.
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
        while (position > 0 && (text_u8[position] & 0xC0) == 0x80) position--;
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
#endif // SZ_USE_ICELAKE

#ifdef __cplusplus
}
#endif

#endif // STRINGZILLA_UTF8_ITERATE_ICELAKE_H_
