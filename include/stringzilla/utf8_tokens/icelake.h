/**
 *  @brief Ice Lake backend for UTF-8 newline and whitespace delimiter scanning.
 *  @file include/stringzilla/utf8_tokens/icelake.h
 *  @author Ash Vardanian
 */
#ifndef STRINGZILLA_UTF8_TOKENS_ICELAKE_H_
#define STRINGZILLA_UTF8_TOKENS_ICELAKE_H_

#include "stringzilla/types.h"
#include "stringzilla/utf8_tokens/serial.h"
#include "stringzilla/utf8_runes/icelake.h" // shared lane-identity + peel substrate

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
SZ_PUBLIC sz_size_t sz_utf8_newlines_icelake(           //
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
    __m512i const lane_identity = sz_utf8_lane_identity_icelake_();

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
            sz_utf8_rune_peel_icelake_(start_bits, two_byte_starts, three_byte_starts, emit, position, lane_identity,
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

SZ_PUBLIC sz_size_t sz_utf8_whitespaces_icelake(        //
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
    __m512i const lane_identity = sz_utf8_lane_identity_icelake_();

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
            sz_utf8_rune_peel_icelake_(start_bits, two_byte_starts, three_byte_starts, emit, position, lane_identity,
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

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#endif // SZ_USE_ICELAKE

#ifdef __cplusplus
}
#endif

#endif // STRINGZILLA_UTF8_TOKENS_ICELAKE_H_
