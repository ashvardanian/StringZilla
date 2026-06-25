/**
 *  @brief Haswell (AVX2) backend for UTF-8 delimiter (punctuation/symbol/separator/whitespace) scanning.
 *  @file include/stringzilla/utf8_delimiters/haswell.h
 *  @author Ash Vardanian
 *
 *  Fully-vectorized, gather-free twin of @ref sz_find_delimiters_utf8_serial and AVX2 mirror of the Ice Lake backend.
 *  Each 64-byte window is decoded once through the shared codepoint substrate (`sz_utf8_rune_decode_window_haswell_`),
 *  and EVERY codepoint-start lane's delimiter membership is resolved IN-REGISTER over the decoded `high`/`low` halves:
 *  the BMP block id via a 256-entry `vpshufb` cascade (`sz_utf8_rune_lut256_haswell_`), the 32-byte bitmap byte via
 *  a transposed-column select over the `..._columns_` tables (AVX2 has no `vpermi2b` page network), and the per-lane bit
 *  test by a `vpshufb` bit-mask. The 4-byte (astral) lanes reconstruct the full 21-bit codepoint in byte-domain and walk
 *  the small astral L1/L2 network the same gather-free way; the whole astral block is gated on a 4-byte lead being present.
 *  Per-lane validity (overlong / surrogate / out-of-range / bad-continuation, mirroring `sz_rune_decode`) is computed as a
 *  mask and intersected with membership, so a malformed lead is never reported and the result is bit-identical to the
 *  serial oracle for the returned pointer and matched length. There is NO scalar per-codepoint membership and NO ASCII
 *  fast path with a scalar tail; no `vpgather` is ever issued.
 */
#ifndef STRINGZILLA_UTF8_DELIMITERS_HASWELL_H_
#define STRINGZILLA_UTF8_DELIMITERS_HASWELL_H_

#include "stringzilla/types.h"

#include "stringzilla/utf8_runes/haswell.h"
#include "stringzilla/utf8_delimiters/serial.h"
#include "stringzilla/utf8_delimiters/tables.h"

#ifdef __cplusplus
extern "C" {
#endif

#if SZ_USE_HASWELL
#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("avx2,bmi,bmi2,popcnt"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("avx2,bmi,bmi2,popcnt")
#endif

#pragma region Gather free membership

/** @brief  Per-half unsigned `value >= bound` mask (AVX2 has no unsigned compare): `max_epu8(value,bound)==value`. */
SZ_INTERNAL __m256i sz_delimiter_cmpge_epu8_haswell_(__m256i value, __m256i bound) {
    return _mm256_cmpeq_epi8(_mm256_max_epu8(value, bound), value);
}

/** @brief  Per-half "third forward neighbour" `next3[i] = window[i+3]`, wrapping modulo 64 to mirror the substrate
 *          neighbour helper (which only emits next1/next2). Needed for the 4-byte astral codepoint reconstruction. */
SZ_INTERNAL void sz_delimiter_forward_neighbour3_haswell_( //
    __m256i window_lo, __m256i window_hi, __m256i *next3_lo, __m256i *next3_hi) {
    __m256i const low_successor = _mm256_permute2x128_si256(window_lo, window_hi, 0x21);
    *next3_lo = _mm256_alignr_epi8(low_successor, window_lo, 3);
    __m256i const high_successor = _mm256_permute2x128_si256(window_hi, window_lo, 0x21);
    *next3_hi = _mm256_alignr_epi8(high_successor, window_hi, 3);
}

/** @brief  Per-lane single-bit test `(bitmap_byte >> (low & 7)) & 1` for one 32-lane half, returned as a 0xFF/0x00
 *          byte mask. The bit mask `1 << (low & 7)` is built by a `vpshufb` over the resident power-of-two table. */
SZ_INTERNAL __m256i sz_delimiter_test_bit_haswell_(__m256i bitmap_byte, __m256i low) {
    __m256i const bit_table = _mm256_setr_epi8(                    //
        1, 2, 4, 8, 16, 32, 64, (char)128, 0, 0, 0, 0, 0, 0, 0, 0, //
        1, 2, 4, 8, 16, 32, 64, (char)128, 0, 0, 0, 0, 0, 0, 0, 0);
    __m256i const bit_mask = _mm256_shuffle_epi8(bit_table, _mm256_and_si256(low, _mm256_set1_epi8(0x07)));
    __m256i const isolated = _mm256_and_si256(bitmap_byte, bit_mask);
    return _mm256_cmpeq_epi8(isolated, bit_mask);
}

/** @brief  Read the bitmap byte `columns[(low>>3)*64 + block_id]` for one 32-lane half, gather-free: 32 candidate
 *          column reads (each a 64-entry `cascade_stage` over `block_id`) blended by which column `(low >> 3)` selects.
 *          The transposed `..._columns_` layout (column c holds `bitmaps[id*32+c]`) makes each column lut256-addressable
 *          for `block_id < 64` without a page network. */
SZ_INTERNAL __m256i sz_delimiter_bitmap_byte_haswell_(sz_u8_t const *columns, __m256i block_id, __m256i low) {
    __m256i const selector = _mm256_and_si256(_mm256_srli_epi16(block_id, 4),
                                              _mm256_set1_epi8(0x0F));         // block_id>>4 (0..3)
    __m256i const within = _mm256_and_si256(block_id, _mm256_set1_epi8(0x0F)); // block_id&15
    __m256i const column_index = _mm256_and_si256(_mm256_srli_epi16(low, 3),
                                                  _mm256_set1_epi8(0x1F)); // (low>>3) in [0,32)
    __m256i result = _mm256_setzero_si256();
    for (int column = 0; column < 32; ++column) {
        __m256i const candidate = sz_utf8_rune_cascade_stage_haswell_(columns + column * 64, 4, selector, within);
        __m256i const here = _mm256_cmpeq_epi8(column_index, _mm256_set1_epi8((char)column));
        result = _mm256_blendv_epi8(result, candidate, here);
    }
    return result;
}

/** @brief  BMP (codepoint < 0x10000) delimiter membership for one 32-lane half, as a 0xFF/0x00 byte mask. ASCII lanes
 *          (top bit clear) carry their codepoint in the raw byte, so are overridden to (high=0, low=byte). */
SZ_INTERNAL __m256i sz_delimiter_bmp_membership_haswell_(__m256i window, __m256i high_in, __m256i low_in) {
    __m256i const ascii = _mm256_cmpeq_epi8(_mm256_and_si256(window, _mm256_set1_epi8((char)0x80)),
                                            _mm256_setzero_si256());
    __m256i const high = _mm256_andnot_si256(ascii, high_in);
    __m256i const low = _mm256_blendv_epi8(low_in, window, ascii);
    __m256i const block_id = sz_utf8_rune_lut256_haswell_(sz_utf8_delimiter_bmp_block_, high);
    __m256i const bitmap_byte = sz_delimiter_bitmap_byte_haswell_(sz_utf8_delimiter_bmp_bitmaps_columns_, block_id,
                                                                  low);
    return sz_delimiter_test_bit_haswell_(bitmap_byte, low);
}

/** @brief  Astral (codepoint >= 0x10000) delimiter membership for one 32-lane half, as a 0xFF/0x00 byte mask. The full
 *          21-bit codepoint is reconstructed in byte-domain from the raw lead/continuation bytes; the small L1/L2 network
 *          and bitmap are then walked exactly as for the BMP path. Only meaningful on 4-byte lead lanes (caller blends). */
SZ_INTERNAL __m256i sz_delimiter_astral_membership_haswell_( //
    __m256i window, __m256i next1, __m256i next2, __m256i next3) {
    __m256i const b0 = _mm256_and_si256(window, _mm256_set1_epi8(0x07)); // lead bits  cp[20:18]
    __m256i const b1 = _mm256_and_si256(next1, _mm256_set1_epi8(0x3F));  // cp[17:12]
    __m256i const b2 = _mm256_and_si256(next2, _mm256_set1_epi8(0x3F));  // cp[11:6]
    __m256i const b3 = _mm256_and_si256(next3, _mm256_set1_epi8(0x3F));  // cp[5:0]

    // super = (cp >> 16) - 1 = (((b0 << 2) | (b1 >> 4)) - 1). cp[20:16] = b0(3 bits) << 2 | b1[5:4].
    __m256i const cp_hi5 = _mm256_or_si256(_mm256_slli_epi16(b0, 2),
                                           _mm256_and_si256(_mm256_srli_epi16(b1, 4), _mm256_set1_epi8(0x03)));
    __m256i const super = _mm256_and_si256(_mm256_sub_epi8(cp_hi5, _mm256_set1_epi8(1)), _mm256_set1_epi8(0x0F));

    // sub = (offset >> 8) & 0xFF where offset = cp - 0x10000. (cp >> 8) = (b1[3:0] << 4) | (b2 >> 2 ... wait b2 spans
    // cp[11:6], so cp[15:8] = (b1[3:0] << 4) | (b2[5:2]). offset>>8 differs from cp>>8 by the -0x100 borrow folded into
    // super; (offset >> 8) & 0xFF == (cp >> 8) & 0xFF since the subtraction only affects bit 16+. So sub = (cp>>8)&0xFF.
    __m256i const sub = _mm256_or_si256(_mm256_slli_epi16(_mm256_and_si256(b1, _mm256_set1_epi8(0x0F)), 4),
                                        _mm256_and_si256(_mm256_srli_epi16(b2, 2), _mm256_set1_epi8(0x0F)));

    // low8 = offset & 0xFF = cp & 0xFF = (b2[1:0] << 6) | b3.
    __m256i const low8 = _mm256_or_si256(_mm256_slli_epi16(_mm256_and_si256(b2, _mm256_set1_epi8(0x03)), 6), b3);

    // group = astral_l1[super]; row id = astral_l2[group*256 + sub]; bitmap byte from astral columns.
    __m256i const group = sz_utf8_rune_lut256_haswell_(sz_utf8_delimiter_astral_l1_, super);
    // l2 index = group*256 + sub. group < 2, so the index high byte is `group` and the low byte is `sub`; a 512-entry
    // table is two 256-entry pages selected by `group`.
    __m256i const page0 = sz_utf8_rune_lut256_haswell_(sz_utf8_delimiter_astral_l2_ + 0, sub);
    __m256i const page1 = sz_utf8_rune_lut256_haswell_(sz_utf8_delimiter_astral_l2_ + 256, sub);
    __m256i const pick_page1 = _mm256_cmpeq_epi8(group, _mm256_set1_epi8(1));
    __m256i const row = _mm256_blendv_epi8(page0, page1, pick_page1);
    __m256i const bitmap_byte = sz_delimiter_bitmap_byte_haswell_(sz_utf8_delimiter_astral_bitmaps_columns_, row, low8);
    return sz_delimiter_test_bit_haswell_(bitmap_byte, low8);
}

/** @brief  Per-lane UTF-8 validity for codepoint-start lanes, mirroring `sz_rune_decode` exactly: a 2/3/4-byte lead is
 *          valid only when its continuation bytes are present (within the loaded span) and well-formed, and it is not
 *          overlong, a surrogate, or beyond U+10FFFF. Returned as a `sz_u64_t` lane mask. */
SZ_INTERNAL sz_u64_t sz_delimiter_valid_starts_haswell_( //
    sz_utf8_rune_window_haswell_t const *decoded, __m256i next1_lo, __m256i next1_hi, __m256i next2_lo,
    __m256i next2_hi, __m256i next3_lo, __m256i next3_hi) {
    sz_size_t const loaded = decoded->loaded;
    sz_u64_t const loaded_mask = loaded >= 64 ? ~(sz_u64_t)0 : (((sz_u64_t)1 << loaded) - 1);

    __m256i const continuation_mask = _mm256_set1_epi8((char)0xC0);
    __m256i const continuation_pattern = _mm256_set1_epi8((char)0x80);
    sz_u64_t const c1_ok = sz_utf8_mask_combine_haswell_(
        _mm256_cmpeq_epi8(_mm256_and_si256(next1_lo, continuation_mask), continuation_pattern),
        _mm256_cmpeq_epi8(_mm256_and_si256(next1_hi, continuation_mask), continuation_pattern));
    sz_u64_t const c2_ok = sz_utf8_mask_combine_haswell_(
        _mm256_cmpeq_epi8(_mm256_and_si256(next2_lo, continuation_mask), continuation_pattern),
        _mm256_cmpeq_epi8(_mm256_and_si256(next2_hi, continuation_mask), continuation_pattern));
    sz_u64_t const c3_ok = sz_utf8_mask_combine_haswell_(
        _mm256_cmpeq_epi8(_mm256_and_si256(next3_lo, continuation_mask), continuation_pattern),
        _mm256_cmpeq_epi8(_mm256_and_si256(next3_hi, continuation_mask), continuation_pattern));

    // 2-byte: lead >= 0xC2 (reject C0/C1 overlong).
    sz_u64_t const lead_ge_c2 = sz_utf8_mask_combine_haswell_(
        sz_delimiter_cmpge_epu8_haswell_(decoded->window_lo, _mm256_set1_epi8((char)0xC2)),
        sz_delimiter_cmpge_epu8_haswell_(decoded->window_hi, _mm256_set1_epi8((char)0xC2)));

    // 3-byte: not overlong (E0 with next1 < 0xA0), not surrogate (ED with next1 >= 0xA0).
    sz_u64_t const lead_e0 = sz_utf8_mask_combine_haswell_(
        _mm256_cmpeq_epi8(decoded->window_lo, _mm256_set1_epi8((char)0xE0)),
        _mm256_cmpeq_epi8(decoded->window_hi, _mm256_set1_epi8((char)0xE0)));
    sz_u64_t const lead_ed = sz_utf8_mask_combine_haswell_(
        _mm256_cmpeq_epi8(decoded->window_lo, _mm256_set1_epi8((char)0xED)),
        _mm256_cmpeq_epi8(decoded->window_hi, _mm256_set1_epi8((char)0xED)));
    __m256i const n1_lo_ge_a0 = sz_delimiter_cmpge_epu8_haswell_(next1_lo, _mm256_set1_epi8((char)0xA0));
    __m256i const n1_hi_ge_a0 = sz_delimiter_cmpge_epu8_haswell_(next1_hi, _mm256_set1_epi8((char)0xA0));
    sz_u64_t const n1_ge_a0 = sz_utf8_mask_combine_haswell_(n1_lo_ge_a0, n1_hi_ge_a0);
    sz_u64_t const n1_lt_a0 = ~n1_ge_a0;

    // 4-byte: lead <= 0xF4, not overlong (F0 with next1 < 0x90), not > U+10FFFF (F4 with next1 >= 0x90).
    sz_u64_t const lead_le_f4 = sz_utf8_mask_combine_haswell_(
        _mm256_andnot_si256(sz_delimiter_cmpge_epu8_haswell_(decoded->window_lo, _mm256_set1_epi8((char)0xF5)),
                            _mm256_set1_epi8((char)0xFF)),
        _mm256_andnot_si256(sz_delimiter_cmpge_epu8_haswell_(decoded->window_hi, _mm256_set1_epi8((char)0xF5)),
                            _mm256_set1_epi8((char)0xFF)));
    sz_u64_t const lead_f0 = sz_utf8_mask_combine_haswell_(
        _mm256_cmpeq_epi8(decoded->window_lo, _mm256_set1_epi8((char)0xF0)),
        _mm256_cmpeq_epi8(decoded->window_hi, _mm256_set1_epi8((char)0xF0)));
    sz_u64_t const lead_f4 = sz_utf8_mask_combine_haswell_(
        _mm256_cmpeq_epi8(decoded->window_lo, _mm256_set1_epi8((char)0xF4)),
        _mm256_cmpeq_epi8(decoded->window_hi, _mm256_set1_epi8((char)0xF4)));
    sz_u64_t const n1_ge_90 = sz_utf8_mask_combine_haswell_(
        sz_delimiter_cmpge_epu8_haswell_(next1_lo, _mm256_set1_epi8((char)0x90)),
        sz_delimiter_cmpge_epu8_haswell_(next1_hi, _mm256_set1_epi8((char)0x90)));
    sz_u64_t const n1_lt_90 = ~n1_ge_90;

    sz_u64_t const ascii = loaded_mask &
                           ~sz_utf8_mask_combine_haswell_(
                               _mm256_cmpeq_epi8(_mm256_and_si256(decoded->window_lo, _mm256_set1_epi8((char)0x80)),
                                                 _mm256_set1_epi8((char)0x80)),
                               _mm256_cmpeq_epi8(_mm256_and_si256(decoded->window_hi, _mm256_set1_epi8((char)0x80)),
                                                 _mm256_set1_epi8((char)0x80)));

    // Spans must lie within the loaded window (so we never validate against a wrapped neighbour or read past loaded).
    sz_u64_t const span2 = loaded >= 1 ? sz_u64_mask_until_serial_(loaded - 1) : 0;
    sz_u64_t const span3 = loaded >= 2 ? sz_u64_mask_until_serial_(loaded - 2) : 0;
    sz_u64_t const span4 = loaded >= 3 ? sz_u64_mask_until_serial_(loaded - 3) : 0;

    sz_u64_t const two_ok = decoded->two_byte_starts & span2 & c1_ok & lead_ge_c2;
    sz_u64_t const three_ok = decoded->three_byte_starts & span3 & c1_ok & c2_ok & ~(lead_e0 & n1_lt_a0) &
                              ~(lead_ed & n1_ge_a0);
    sz_u64_t const four_ok = decoded->four_byte_starts & span4 & c1_ok & c2_ok & c3_ok & lead_le_f4 &
                             ~(lead_f0 & n1_lt_90) & ~(lead_f4 & n1_ge_90);
    return (ascii | two_ok | three_ok | four_ok) & loaded_mask;
}

#pragma endregion Gather free membership

#pragma region Find first delimiter

/** @copydoc sz_find_delimiter_utf8 */
SZ_PUBLIC sz_cptr_t sz_find_delimiters_utf8_haswell(sz_cptr_t text, sz_size_t length, sz_size_t *matched_length) {
    sz_u8_t const *text_u8 = (sz_u8_t const *)text;
    sz_size_t base = 0;

    while (base < length) {
        sz_utf8_rune_window_haswell_t const decoded = sz_utf8_rune_decode_window_haswell_(text_u8 + base,
                                                                                          length - base);
        sz_size_t const loaded = decoded.loaded;

        __m256i next1_lo, next1_hi, next2_lo, next2_hi;
        sz_utf8_forward_neighbours_haswell_(decoded.window_lo, decoded.window_hi, &next1_lo, &next1_hi, &next2_lo,
                                            &next2_hi);
        __m256i next3_lo, next3_hi;
        sz_delimiter_forward_neighbour3_haswell_(decoded.window_lo, decoded.window_hi, &next3_lo, &next3_hi);

        // Effective-window trim: a multi-byte lead near the 64-byte edge whose span runs past `loaded` would decode
        // against a wrapped neighbour; defer it to the next window. The lowest overrunning start bounds the span.
        sz_size_t byte_span = loaded;
        if (loaded >= 64) {
            sz_u64_t const overrun = (decoded.two_byte_starts & ~sz_u64_mask_until_serial_(loaded - 1)) |
                                     (decoded.three_byte_starts & ~sz_u64_mask_until_serial_(loaded - 2)) |
                                     (decoded.four_byte_starts & ~sz_u64_mask_until_serial_(loaded - 3));
            byte_span = overrun ? (sz_size_t)sz_u64_ctz(overrun) : loaded;
        }
        sz_u64_t const span_mask = sz_u64_mask_until_serial_(byte_span);

        sz_u64_t const valid_starts = sz_delimiter_valid_starts_haswell_(&decoded, next1_lo, next1_hi, next2_lo,
                                                                         next2_hi, next3_lo, next3_hi) &
                                      decoded.codepoint_starts & span_mask;

        // BMP membership for every lane; astral membership blended onto the four-byte lanes (gated on presence).
        sz_u64_t member = sz_utf8_mask_combine_haswell_(
            sz_delimiter_bmp_membership_haswell_(decoded.window_lo, decoded.high_lo, decoded.low_lo),
            sz_delimiter_bmp_membership_haswell_(decoded.window_hi, decoded.high_hi, decoded.low_hi));
        sz_u64_t const four_byte = decoded.four_byte_starts & span_mask;
        if (four_byte) {
            sz_u64_t const astral_member = sz_utf8_mask_combine_haswell_(
                sz_delimiter_astral_membership_haswell_(decoded.window_lo, next1_lo, next2_lo, next3_lo),
                sz_delimiter_astral_membership_haswell_(decoded.window_hi, next1_hi, next2_hi, next3_hi));
            member = (member & ~four_byte) | (astral_member & four_byte);
        }

        sz_u64_t const hits = member & valid_starts;
        if (hits) {
            sz_size_t const lane = (sz_size_t)sz_u64_ctz(hits);
            if (matched_length) {
                sz_size_t length_at_lane = 1;
                length_at_lane += (decoded.two_byte_starts >> lane) & 1;
                length_at_lane += ((decoded.three_byte_starts >> lane) & 1) * 2;
                length_at_lane += ((decoded.four_byte_starts >> lane) & 1) * 3;
                *matched_length = length_at_lane;
            }
            return (sz_cptr_t)(text_u8 + base + lane);
        }
        base += byte_span ? byte_span : 1;
    }

    if (matched_length) *matched_length = 0;
    return SZ_NULL_CHAR;
}

#pragma endregion Find first delimiter

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#endif // SZ_USE_HASWELL

#ifdef __cplusplus
}
#endif

#endif // STRINGZILLA_UTF8_DELIMITERS_HASWELL_H_
