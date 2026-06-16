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

/*  Multistep newline / whitespace iteration (Ice Lake / AVX-512).
 *
 *  Each 64-byte window is classified branchlessly into a `starts` mask plus a per-lane byte-length vector,
 *  then `vpcompressb` peels the matching lanes and lengths. Starts are trusted in lanes [0,61] (step 62) so
 *  any 2-/3-byte delimiter is fully loaded; a `t[pos-1] == '\r'` carry suppresses an LF closing an edge CRLF. */

SZ_INTERNAL __m512i sz_utf8_iterate_lane_identity_icelake_(void) {
    return _mm512_set_epi8(                                             //
        63, 62, 61, 60, 59, 58, 57, 56, 55, 54, 53, 52, 51, 50, 49, 48, //
        47, 46, 45, 44, 43, 42, 41, 40, 39, 38, 37, 36, 35, 34, 33, 32, //
        31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, //
        15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
}

/**
 *  @brief Emit a window's `emit` matches in-register via `vpcompressb` and `ceil(emit/8)` masked widen-stores.
 *  @note `_mm512_alignr_epi64` shifts the compressed registers down between waves, so `emit` may exceed 8.
 */
SZ_INTERNAL void sz_utf8_iterate_peel_icelake_(                                  //
    sz_u64_t start_bits, __mmask64 two_byte_starts, __mmask64 three_byte_starts, //
    sz_size_t emit, sz_size_t position, __m512i lane_identity,                   //
    sz_size_t *match_offsets, sz_size_t *match_lengths) {
    __mmask64 const compress_mask = _cvtu64_mask64(start_bits);

    // Per-lane byte length: 1, plus 1 on a 2-byte start, plus 2 on a 3-byte start (the masks are disjoint).
    __m512i length_per_lane = _mm512_set1_epi8(1);
    length_per_lane = _mm512_mask_add_epi8(length_per_lane, two_byte_starts, length_per_lane, _mm512_set1_epi8(1));
    length_per_lane = _mm512_mask_add_epi8(length_per_lane, three_byte_starts, length_per_lane, _mm512_set1_epi8(2));

    __m512i compressed_offsets = _mm512_maskz_compress_epi8(compress_mask, lane_identity);
    __m512i compressed_lengths = _mm512_maskz_compress_epi8(compress_mask, length_per_lane);
    __m512i const position_broadcast = _mm512_set1_epi64((long long)position);

    __m512i const zero = _mm512_setzero_si512();
    for (sz_size_t emitted = 0; emitted < emit; emitted += 8) {
        __mmask8 const store_mask = sz_u8_clamp_mask_until_(emit - emitted);
        _mm512_mask_storeu_epi64(
            (void *)(match_offsets + emitted), store_mask,
            _mm512_add_epi64(_mm512_cvtepu8_epi64(_mm512_castsi512_si128(compressed_offsets)), position_broadcast));
        _mm512_mask_storeu_epi64((void *)(match_lengths + emitted), store_mask,
                                 _mm512_cvtepu8_epi64(_mm512_castsi512_si128(compressed_lengths)));
        compressed_offsets = _mm512_alignr_epi64(zero, compressed_offsets, 1);
        compressed_lengths = _mm512_alignr_epi64(zero, compressed_lengths, 1);
    }
}

SZ_PUBLIC sz_size_t sz_utf8_find_newlines_icelake(      //
    sz_cptr_t text, sz_size_t length,                   //
    sz_size_t *match_offsets, sz_size_t *match_lengths, //
    sz_size_t matches_capacity, sz_size_t *bytes_consumed) {

    sz_u8_t const *text_u8 = (sz_u8_t const *)text;
    sz_size_t count = 0, position = 0;
    __m512i newline_vec = _mm512_set1_epi8('\n'), vertical_tab_vec = _mm512_set1_epi8('\v'),
            form_feed_vec = _mm512_set1_epi8('\f'), carriage_return_vec = _mm512_set1_epi8('\r'),
            lead_c2_vec = _mm512_set1_epi8('\xC2'), x_85_vec = _mm512_set1_epi8('\x85'),
            lead_e2_vec = _mm512_set1_epi8('\xE2'), byte_80_vec = _mm512_set1_epi8('\x80'),
            x_a8_vec = _mm512_set1_epi8('\xA8'), x_a9_vec = _mm512_set1_epi8('\xA9');
    __m512i const lane_identity = sz_utf8_iterate_lane_identity_icelake_();

    while (position < length && count < matches_capacity) {
        sz_size_t const valid_lanes = length - position;
        // Plain load + trust [0,61] on a full window (the hot path); a masked load clips the final partial window
        // to its valid lanes (zero-filled past `length`, so a truncated multi-byte delimiter cannot match).
        __m512i window;
        sz_u64_t trusted_bits;
        if (valid_lanes >= 64) { window = _mm512_loadu_epi8(text_u8 + position), trusted_bits = (1ull << 62) - 1; }
        else {
            __mmask64 const load_mask = sz_u64_mask_until_(valid_lanes);
            window = _mm512_maskz_loadu_epi8(load_mask, text_u8 + position), trusted_bits = _cvtmask64_u64(load_mask);
        }
        __mmask64 newline_mask = _mm512_cmpeq_epi8_mask(window, newline_vec);
        __mmask64 carriage_return_mask = _mm512_cmpeq_epi8_mask(window, carriage_return_vec);
        __mmask64 one_byte_mask = _kor_mask64(
            _kor_mask64(newline_mask, _mm512_cmpeq_epi8_mask(window, vertical_tab_vec)),
            _kor_mask64(_mm512_cmpeq_epi8_mask(window, form_feed_vec), carriage_return_mask));
        // 2-byte NEL (C2 85); 3-byte LS/PS (E2 80 A8/A9) - computed unconditionally (cheaper than a data-dependent gate).
        __mmask64 nel_mask = _kand_mask64(_mm512_cmpeq_epi8_mask(window, lead_c2_vec),
                                          _kshiftri_mask64(_mm512_cmpeq_epi8_mask(window, x_85_vec), 1));
        __mmask64 lead_e280_mask = _kand_mask64(_mm512_cmpeq_epi8_mask(window, lead_e2_vec),
                                                _kshiftri_mask64(_mm512_cmpeq_epi8_mask(window, byte_80_vec), 1));
        __mmask64 line_para_mask = _kand_mask64(
            lead_e280_mask,
            _kshiftri_mask64(
                _kor_mask64(_mm512_cmpeq_epi8_mask(window, x_a8_vec), _mm512_cmpeq_epi8_mask(window, x_a9_vec)), 2));
        // CRLF: a CR whose next lane is LF is a single 2-byte match; its trailing LF must not also be emitted.
        __mmask64 crlf_mask = _kand_mask64(carriage_return_mask, _kshiftri_mask64(newline_mask, 1));
        __mmask64 lf_of_crlf_mask = _kand_mask64(newline_mask, _kshiftli_mask64(carriage_return_mask, 1));

        __mmask64 two_byte_starts = _kor_mask64(crlf_mask, nel_mask);
        __mmask64 three_byte_starts = line_para_mask;
        __mmask64 starts = _kandn_mask64(lf_of_crlf_mask,
                                         _kor_mask64(_kor_mask64(one_byte_mask, nel_mask), line_para_mask));
        sz_u64_t start_bits = _cvtmask64_u64(starts) & trusted_bits;
        // Suppress a leading LF already consumed by a CRLF that straddled the previous window edge.
        if (position != 0 && text_u8[position - 1] == '\r') start_bits &= ~(_cvtmask64_u64(newline_mask) & 1ull);

        sz_size_t const window_matches = (sz_size_t)_mm_popcnt_u64(start_bits);
        sz_size_t const emit = sz_min_of_two(window_matches, matches_capacity - count);
        if (emit)
            sz_utf8_iterate_peel_icelake_(start_bits, two_byte_starts, three_byte_starts, emit, position, lane_identity,
                                          match_offsets + count, match_lengths + count);
        count += emit;
        if (count == matches_capacity) { // output buffer full: resume past the last emitted match
            position = match_offsets[count - 1] + match_lengths[count - 1];
            break;
        }
        position += valid_lanes >= 64 ? 62 : valid_lanes;
    }

    // If the loop stopped on the LF of an already-emitted CRLF, resume past it (the CR carried the length-2 match).
    if (position != 0 && position < length && text_u8[position - 1] == '\r' && text_u8[position] == '\n') ++position;
    if (bytes_consumed) *bytes_consumed = position;
    return count;
}

SZ_PUBLIC sz_size_t sz_utf8_find_whitespaces_icelake(   //
    sz_cptr_t text, sz_size_t length,                   //
    sz_size_t *match_offsets, sz_size_t *match_lengths, //
    sz_size_t matches_capacity, sz_size_t *bytes_consumed) {

    sz_u8_t const *text_u8 = (sz_u8_t const *)text;
    sz_size_t count = 0, position = 0;
    __m512i tab_vec = _mm512_set1_epi8('\t'), carriage_return_vec = _mm512_set1_epi8('\r'),
            x_20_vec = _mm512_set1_epi8(' '), lead_c2_vec = _mm512_set1_epi8('\xC2'),
            x_85_vec = _mm512_set1_epi8('\x85'), x_a0_vec = _mm512_set1_epi8('\xA0'),
            x_e1_vec = _mm512_set1_epi8('\xE1'), lead_e2_vec = _mm512_set1_epi8('\xE2'),
            x_e3_vec = _mm512_set1_epi8('\xE3'), x_9a_vec = _mm512_set1_epi8('\x9A'),
            byte_80_vec = _mm512_set1_epi8('\x80'), x_81_vec = _mm512_set1_epi8('\x81'),
            x_8d_vec = _mm512_set1_epi8('\x8D'), x_a8_vec = _mm512_set1_epi8('\xA8'),
            x_a9_vec = _mm512_set1_epi8('\xA9'), x_af_vec = _mm512_set1_epi8('\xAF'),
            x_9f_vec = _mm512_set1_epi8('\x9F');
    __m512i const lane_identity = sz_utf8_iterate_lane_identity_icelake_();

    while (position < length && count < matches_capacity) {
        sz_size_t const valid_lanes = length - position;
        __m512i window;
        sz_u64_t trusted_bits;
        if (valid_lanes >= 64) { window = _mm512_loadu_epi8(text_u8 + position), trusted_bits = (1ull << 62) - 1; }
        else {
            __mmask64 const load_mask = sz_u64_mask_until_(valid_lanes);
            window = _mm512_maskz_loadu_epi8(load_mask, text_u8 + position), trusted_bits = _cvtmask64_u64(load_mask);
        }
        // 1-byte: space, plus the contiguous range [\t, \r] == [9, 13].
        __mmask64 one_byte_mask = _kor_mask64(
            _mm512_cmpeq_epi8_mask(window, x_20_vec),
            _kand_mask64(_mm512_cmpge_epu8_mask(window, tab_vec), _mm512_cmple_epu8_mask(window, carriage_return_vec)));

        // 2-byte: C2 85 (NEL), C2 A0 (NBSP).
        __mmask64 lead_c2_mask = _mm512_cmpeq_epi8_mask(window, lead_c2_vec);
        __mmask64 two_byte_starts = _kand_mask64(
            lead_c2_mask, _kor_mask64(_kshiftri_mask64(_mm512_cmpeq_epi8_mask(window, x_85_vec), 1),
                                      _kshiftri_mask64(_mm512_cmpeq_epi8_mask(window, x_a0_vec), 1)));

        // 3-byte: E1 9A 80 (ogham); E2 80 [80-8D]; E2 80 AF; E2 81 9F; E2 80 A8/A9; E3 80 80.
        __mmask64 byte_80_mask = _mm512_cmpeq_epi8_mask(window, byte_80_vec);
        __mmask64 lead_e2_mask = _mm512_cmpeq_epi8_mask(window, lead_e2_vec);
        __mmask64 lead_e280_mask = _kand_mask64(lead_e2_mask, _kshiftri_mask64(byte_80_mask, 1));
        __mmask64 ogham_mask = _kand_mask64(_mm512_cmpeq_epi8_mask(window, x_e1_vec),
                                            _kand_mask64(_kshiftri_mask64(_mm512_cmpeq_epi8_mask(window, x_9a_vec), 1),
                                                         _kshiftri_mask64(byte_80_mask, 2)));
        __mmask64 range_e280_mask = _kand_mask64(
            lead_e280_mask, _kand_mask64(_kshiftri_mask64(_mm512_cmpge_epu8_mask(window, byte_80_vec), 2),
                                         _kshiftri_mask64(_mm512_cmple_epu8_mask(window, x_8d_vec), 2)));
        __mmask64 nnbsp_mask = _kand_mask64(lead_e280_mask,
                                            _kshiftri_mask64(_mm512_cmpeq_epi8_mask(window, x_af_vec), 2));
        __mmask64 mmsp_mask = _kand_mask64(
            _kand_mask64(lead_e2_mask, _kshiftri_mask64(_mm512_cmpeq_epi8_mask(window, x_81_vec), 1)),
            _kshiftri_mask64(_mm512_cmpeq_epi8_mask(window, x_9f_vec), 2));
        __mmask64 line_mask = _kand_mask64(lead_e280_mask,
                                           _kshiftri_mask64(_mm512_cmpeq_epi8_mask(window, x_a8_vec), 2));
        __mmask64 para_mask = _kand_mask64(lead_e280_mask,
                                           _kshiftri_mask64(_mm512_cmpeq_epi8_mask(window, x_a9_vec), 2));
        __mmask64 ideographic_mask = _kand_mask64(
            _kand_mask64(_mm512_cmpeq_epi8_mask(window, x_e3_vec), _kshiftri_mask64(byte_80_mask, 1)),
            _kshiftri_mask64(byte_80_mask, 2));
        __mmask64 three_byte_starts = _kor_mask64(
            _kor_mask64(_kor_mask64(ogham_mask, range_e280_mask), _kor_mask64(nnbsp_mask, mmsp_mask)),
            _kor_mask64(_kor_mask64(line_mask, para_mask), ideographic_mask));

        __mmask64 starts = _kor_mask64(_kor_mask64(one_byte_mask, two_byte_starts), three_byte_starts);
        sz_u64_t start_bits = _cvtmask64_u64(starts) & trusted_bits;

        sz_size_t const window_matches = (sz_size_t)_mm_popcnt_u64(start_bits);
        sz_size_t const emit = sz_min_of_two(window_matches, matches_capacity - count);
        if (emit)
            sz_utf8_iterate_peel_icelake_(start_bits, two_byte_starts, three_byte_starts, emit, position, lane_identity,
                                          match_offsets + count, match_lengths + count);
        count += emit;
        if (count == matches_capacity) {
            position = match_offsets[count - 1] + match_lengths[count - 1];
            break;
        }
        position += valid_lanes >= 64 ? 62 : valid_lanes;
    }

    if (bytes_consumed) *bytes_consumed = position;
    return count;
}

SZ_PUBLIC sz_size_t sz_utf8_count_icelake(sz_cptr_t text, sz_size_t length) {
    // Count every byte that is NOT a continuation byte: `(byte & 0xC0) != 0x80` selects character starts.
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

        // If the Nth start byte is past this window, skip the whole block.
        if (n >= start_byte_count) {
            n -= start_byte_count;
            text_u8 += 64;
            length -= 64;
            continue;
        }

        // PDEP isolates the Nth set bit in one step: `_pdep_u64(0b10, 0b0001010100) == 0b0000010000`.
        sz_u64_t deposited_bits = _pdep_u64((sz_u64_t)1 << n, start_byte_mask);
        int byte_offset = (int)_tzcnt_u64(deposited_bits);
        return (sz_cptr_t)(text_u8 + byte_offset);
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
    sz_size_t mixed12_prefix_length = (sz_size_t)_tzcnt_u64(~is_valid_mixed12 | ~load_mask);
    mixed12_prefix_length -= mixed12_prefix_length && ((is_two_byte_start >> (mixed12_prefix_length - 1)) & 1);

    if (mixed12_prefix_length >= 2) {
        __mmask64 prefix_mask = sz_u64_mask_until_(mixed12_prefix_length);
        __mmask64 is_char_start = (is_ascii | is_two_byte_start) & prefix_mask;
        sz_size_t num_runes = (sz_size_t)_mm_popcnt_u64(_cvtmask64_u64(is_char_start));
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
        sz_size_t two_byte_count =
            (sz_size_t)_mm_popcnt_u64(_cvtmask16_u32(_kand_mask16(is_two_byte_char, sz_u16_mask_until_(runes_to_unpack))));
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
    sz_size_t three_byte_prefix_length = (sz_size_t)_tzcnt_u64((sz_u64_t)(~three_byte_match_mask));

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
    sz_size_t four_byte_prefix_length = (sz_size_t)_tzcnt_u64((sz_u64_t)(~four_byte_match_mask));

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

/*  UAX-29 word boundary detection (Ice Lake / AVX-512).
 *
 *  Only the common, safe part is vectorized: an all-ASCII 64-byte chunk is classified by table lookup, then
 *  the unconditional local no-break rules mark a superset of the true boundaries. Non-ASCII chunks and edges
 *  fall back to `sz_utf8_is_word_boundary_serial`, which owns the stateful sub-rules, keeping output identical
 *  to serial for every input while ASCII runs advance 64 bytes at a time. */

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
 *  @brief Per-position "joined" (guaranteed non-boundary) mask for an all-ASCII chunk: bit `i` set when a
 *         UAX-29 no-break rule suppresses the boundary before lane `i`. Exact for in-window i-2 and i+1 neighbours.
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

    // Oracle-free fast path: an all-ASCII window [position-2, position+62) resolves boundaries at positions
    // [position, position+60]; `vpcompressb` packs their lane offsets and a shifted-difference emits the words.
    while (position < length) {
        int ascii_window = position >= 2 && position + 62 <= length;
        __m512i window = _mm512_setzero_si512();
        if (ascii_window) {
            window = _mm512_loadu_epi8(text_u8 + position - 2); // lane j = byte position-2+j
            ascii_window = _mm512_movepi8_mask(window) == 0;
        }
        if (!ascii_window) { // Non-ASCII window or near the edges: one scalar codepoint step.
            // `position == word_start` only after a re-classify resume: the open word's start is not a word end.
            if (position != word_start && sz_utf8_is_word_boundary_serial(text, length, position)) {
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

        sz_u64_t boundary = (~sz_utf8_word_break_join_mask_ascii_(sz_utf8_word_break_classify_ascii_icelake_(window))) &
                            0x7FFFFFFFFFFFFFFCull; // trusted lanes [2,62]
        // After a re-classify resume the open word's own start lands in lane 2; it is a word start, not a word end.
        if (position == word_start) boundary &= ~(1ull << 2);
        sz_size_t boundary_count = (sz_size_t)_mm_popcnt_u64(boundary);
        sz_size_t emit = sz_min_of_three(boundary_count, words_capacity - words, 8);

        __m512i compressed_lanes = _mm512_maskz_compress_epi8(boundary, lane_identity);
        __m512i boundary_positions = _mm512_add_epi64(_mm512_cvtepu8_epi64(_mm512_castsi512_si128(compressed_lanes)),
                                                      _mm512_set1_epi64((long long)position - 2));
        __m512i starts = _mm512_alignr_epi64(boundary_positions, _mm512_set1_epi64((long long)word_start), 7);
        __mmask8 store_mask = sz_u8_clamp_mask_until_(emit);
        _mm512_mask_storeu_epi64((void *)(word_starts + words), store_mask, starts);
        _mm512_mask_storeu_epi64((void *)(word_lengths + words), store_mask,
                                 _mm512_sub_epi64(boundary_positions, starts));
        words += emit;
        if (emit) word_start = word_starts[words - 1] + word_lengths[words - 1];
        if (words == words_capacity) {
            if (bytes_consumed) *bytes_consumed = word_start;
            return words;
        }
        position = emit < boundary_count ? word_start : position + 61; // re-classify past the 8th, else full step
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
    // The descending identity holds `63 - lane`, so pairing it with the bit-reversed boundary mask lets
    // `vpcompressb` pack the boundary lane offsets high-to-low (the emission order this kernel needs).
    __m512i const lane_identity_descending = _mm512_set_epi8(
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,           //
        16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, //
        32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, //
        48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63);

    // Oracle-free fast path: an all-ASCII window [position-62, position+2) resolves boundaries at positions
    // [position-60, position]; the descending identity + bit-reversed mask pack them high-to-low for emission.
    while (position > 0) {
        sz_size_t base = position - 62; // lane j = byte base+j; trusted lanes [2,62] → [position-60, position]
        int ascii_window = position >= 62 && position + 2 <= length;
        __m512i window = _mm512_setzero_si512();
        if (ascii_window) {
            window = _mm512_loadu_epi8(text_u8 + base);
            ascii_window = _mm512_movepi8_mask(window) == 0;
        }
        if (!ascii_window) { // Non-ASCII window or near the edges: one scalar codepoint step.
            // `position == word_end` only after a re-classify resume: the open word's end is not a new break.
            if (position != word_end && sz_utf8_is_word_boundary_serial(text, length, position)) {
                if (words == words_capacity) {
                    if (bytes_consumed) *bytes_consumed = word_end;
                    return words;
                }
                word_starts[words] = position, word_lengths[words] = word_end - position, ++words;
                word_end = position;
            }
            position--;
            while (position > 0 && (text_u8[position] & 0xC0) == 0x80) position--;
            continue;
        }

        sz_u64_t boundary = (~sz_utf8_word_break_join_mask_ascii_(sz_utf8_word_break_classify_ascii_icelake_(window))) &
                            0x7FFFFFFFFFFFFFFCull; // trusted lanes [2,62]
        // After a re-classify resume the open word's own end lands in lane 62; it is a word end, not a new break.
        if (position == word_end) boundary &= ~(1ull << 62);
        sz_size_t boundary_count = (sz_size_t)_mm_popcnt_u64(boundary);
        sz_size_t emit = sz_min_of_three(boundary_count, words_capacity - words, 8);

        __m512i compressed_lanes = _mm512_maskz_compress_epi8(sz_u64_bits_reverse(boundary), lane_identity_descending);
        __m512i boundary_positions = _mm512_add_epi64(_mm512_cvtepu8_epi64(_mm512_castsi512_si128(compressed_lanes)),
                                                      _mm512_set1_epi64((long long)base));
        __m512i previous = _mm512_alignr_epi64(boundary_positions, _mm512_set1_epi64((long long)word_end), 7);
        __mmask8 store_mask = sz_u8_clamp_mask_until_(emit);
        _mm512_mask_storeu_epi64((void *)(word_starts + words), store_mask, boundary_positions);
        _mm512_mask_storeu_epi64((void *)(word_lengths + words), store_mask,
                                 _mm512_sub_epi64(previous, boundary_positions));
        words += emit;
        if (emit) word_end = word_starts[words - 1];
        if (words == words_capacity) {
            if (bytes_consumed) *bytes_consumed = word_end;
            return words;
        }
        position = emit < boundary_count ? word_end : base + 1; // re-classify past the 8th, else full step
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
