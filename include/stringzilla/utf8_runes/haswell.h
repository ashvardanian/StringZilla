/**
 *  @brief Haswell (AVX2) backend for UTF-8 codepoint mechanics.
 *  @file include/stringzilla/utf8_runes/haswell.h
 *  @author Ash Vardanian
 */
#ifndef STRINGZILLA_UTF8_RUNES_HASWELL_H_
#define STRINGZILLA_UTF8_RUNES_HASWELL_H_

#include "stringzilla/types.h"
#include "stringzilla/utf8_runes/serial.h"

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

/** @brief  Mask for `_mm256_maskstore_epi64` selecting the low `count` (0..4) of four 64-bit lanes. */
SZ_HELPER_INLINE __m256i sz_mm256_store_mask_epi64_(sz_size_t count) {
    return _mm256_cmpgt_epi64(_mm256_set1_epi64x((long long)count), _mm256_setr_epi64x(0, 1, 2, 3));
}

SZ_API_COMPTIME sz_size_t sz_utf8_count_haswell(sz_cptr_t text, sz_size_t length) {
    // Continuation bytes are `0x80..0xBF` = signed `-128..-65`, so a signed `vpcmpgtb` against `-65` selects
    // character starts directly in one op, replacing the `AND(0xC0)`, `cmpeq(0x80)`, and mask-negate trio.
    sz_u256_vec_t start_threshold_vec;
    start_threshold_vec.ymm = _mm256_set1_epi8((char)-65);

    sz_u8_t const *text_u8 = (sz_u8_t const *)text;
    sz_size_t char_count = 0;

    // Process 32 bytes at a time
    sz_u256_vec_t text_vec;
    while (length >= 32) {
        text_vec.ymm = _mm256_loadu_si256((__m256i const *)text_u8);
        sz_u32_t start_byte_mask = (sz_u32_t)_mm256_movemask_epi8(
            _mm256_cmpgt_epi8(text_vec.ymm, start_threshold_vec.ymm));
        char_count += _mm_popcnt_u32(start_byte_mask);
        text_u8 += 32;
        length -= 32;
    }

    // Process remaining bytes with serial
    char_count += sz_utf8_count_serial((sz_cptr_t)text_u8, length);
    return char_count;
}

SZ_API_COMPTIME sz_cptr_t sz_utf8_seek_haswell(sz_cptr_t text, sz_size_t length, sz_size_t n) {
    // The logic of this function is similar to `sz_utf8_count_haswell`, but uses PDEP
    // instruction in the inner loop to locate Nth character start byte efficiently
    // without one more loop.
    sz_u256_vec_t start_threshold_vec;
    start_threshold_vec.ymm = _mm256_set1_epi8((char)-65); // signed boundary: starts are bytes > -65

    sz_u8_t const *text_u8 = (sz_u8_t const *)text;

    // Process 32 bytes at a time
    sz_u256_vec_t text_vec;
    while (length >= 32) {
        text_vec.ymm = _mm256_loadu_si256((__m256i const *)text_u8);
        sz_u32_t start_byte_mask = (sz_u32_t)_mm256_movemask_epi8(
            _mm256_cmpgt_epi8(text_vec.ymm, start_threshold_vec.ymm));
        sz_size_t start_byte_count = _mm_popcnt_u32(start_byte_mask);

        // The Nth start byte is not in this window: advance to the next block.
        if (n >= start_byte_count) {
            n -= start_byte_count;
            text_u8 += 32;
            length -= 32;
            continue;
        }

        // `_pdep_u32` isolates the Nth set bit; `_tzcnt_u32` then gives its lane.
        sz_u32_t deposited_bits = _pdep_u32((sz_u32_t)1 << n, start_byte_mask);
        int byte_offset = (int)_tzcnt_u32(deposited_bits);
        return (sz_cptr_t)(text_u8 + byte_offset);
    }

    // Process remaining bytes with serial
    return sz_utf8_seek_serial((sz_cptr_t)text_u8, length, n);
}

#pragma region Shared SIMD leaf substrate

/** @brief  The decoded 64-byte window for the AVX2 backend. The 64 bytes live as two `__m256i` halves (`*_lo` =
 *          lanes [0, 32), `*_hi` = lanes [32, 64)); the per-lane byte-domain codepoint halves `high`/`low` share that
 *          shape. Masks are `sz_u64_t` (`vpmovmskb` per half, OR-combined) rather than the Ice Lake `__mmask64`. Field
 *          names and semantics match @ref sz_utf8_rune_window_t so the portable rule algebra is unchanged. */
typedef struct sz_utf8_rune_window_haswell_t {
    __m256i window_lo;          /**< Raw input bytes for lanes [0, 32). */
    __m256i window_hi;          /**< Raw input bytes for lanes [32, 64). */
    __m256i high_lo;            /**< Per-lane `codepoint >> 8` for lanes [0, 32). */
    __m256i high_hi;            /**< Per-lane `codepoint >> 8` for lanes [32, 64). */
    __m256i low_lo;             /**< Per-lane `codepoint & 0xFF` for lanes [0, 32). */
    __m256i low_hi;             /**< Per-lane `codepoint & 0xFF` for lanes [32, 64). */
    sz_u64_t continuation;      /**< Bit `i` => lane `i` is a continuation byte `10xxxxxx`. */
    sz_u64_t codepoint_starts;  /**< Bit `i` => lane `i` begins a codepoint (loaded, non-continuation). */
    sz_u64_t two_byte_starts;   /**< Bit `i` => lane `i` is a 2-byte lead `110xxxxx`. */
    sz_u64_t three_byte_starts; /**< Bit `i` => lane `i` is a 3-byte lead `1110xxxx`. */
    sz_u64_t four_byte_starts;  /**< Bit `i` => lane `i` is a 4-byte lead `11110xxx`. */
    sz_size_t loaded;           /**< Number of bytes actually loaded (<= 64). */
} sz_utf8_rune_window_haswell_t;

/** @brief  Per-byte logical right shift by @p shift keeping the low @p keep bits — the AVX2 twin of `srl8_`. */
SZ_HELPER_INLINE __m256i sz_utf8_srl8_haswell_(__m256i value, int shift, sz_u8_t keep) {
    return _mm256_and_si256(_mm256_srli_epi16(value, shift), _mm256_set1_epi8((char)keep));
}

/** @brief  Combine two per-half `vpmovmskb` results into one 64-bit lane mask. */
SZ_HELPER_INLINE sz_u64_t sz_utf8_mask_combine_haswell_(__m256i low_half, __m256i high_half) {
    sz_u64_t const low_bits = (sz_u32_t)_mm256_movemask_epi8(low_half);
    sz_u64_t const high_bits = (sz_u32_t)_mm256_movemask_epi8(high_half);
    return low_bits | (high_bits << 32);
}

/** @brief  Masked 64-byte load into two halves; bytes [loaded, 64) read as zero (the AVX2 stand-in for
 *          `_mm512_maskz_loadu_epi8`). A small stack staging union covers the partial tail so we never read past
 *          `text + loaded`. */
SZ_HELPER_AUTO void sz_utf8_load_window_haswell_( //
    sz_u8_t const *text, sz_size_t loaded, __m256i *out_low, __m256i *out_high) {
    if (loaded >= 64) {
        *out_low = _mm256_loadu_si256((__m256i const *)(text + 0));
        *out_high = _mm256_loadu_si256((__m256i const *)(text + 32));
        return;
    }
    sz_u512_vec_t staging;
    staging.ymms[0] = _mm256_setzero_si256();
    staging.ymms[1] = _mm256_setzero_si256();
    for (sz_size_t i = 0; i < loaded; ++i) staging.u8s[i] = text[i];
    *out_low = staging.ymms[0];
    *out_high = staging.ymms[1];
}

/** @brief  Forward neighbours `next1[i] = window[i+1]`, `next2[i] = window[i+2]` over all 64 lanes, with the lanes
 *          past the window WRAPPING modulo 64 to match Ice Lake's `_mm512_permutexvar_epi8` (so `next1[63]==window[0]`,
 *          `next2[62]==window[0]`, `next2[63]==window[1]`). AVX2 has no 32-byte byte permute, so each 128-bit lane is
 *          fed its following bytes via `_mm256_permute2x128_si256` (to rotate in the successor 128-bit block, the
 *          window head wrapping in after `window_hi`) then `_mm256_alignr_epi8` to shift across the lane boundary. */
SZ_HELPER_INLINE void sz_utf8_forward_neighbours_haswell_( //
    __m256i window_low_u8x32, __m256i window_high_u8x32,   //
    __m256i *next_byte_1_low_u8x32, __m256i *next_byte_1_high_u8x32, __m256i *next_byte_2_low_u8x32,
    __m256i *next_byte_2_high_u8x32) {
    // The 128-bit block following each 128-bit lane of window_low: its high lane (bytes 16..31) is followed by
    // window_high's low lane (bytes 32..47), assembled by permute2x128 selector 0x21.
    __m256i const successor_low_u8x32 = _mm256_permute2x128_si256(window_low_u8x32, window_high_u8x32, 0x21);
    *next_byte_1_low_u8x32 = _mm256_alignr_epi8(successor_low_u8x32, window_low_u8x32, 1);
    *next_byte_2_low_u8x32 = _mm256_alignr_epi8(successor_low_u8x32, window_low_u8x32, 2);
    // The block following window_high's high lane (bytes 48..63) wraps to the window head (bytes 0..15 of
    // window_low), so window_high's successor is window_low (byte 64 aliases byte 0).
    __m256i const successor_high_u8x32 = _mm256_permute2x128_si256(window_high_u8x32, window_low_u8x32, 0x21);
    *next_byte_1_high_u8x32 = _mm256_alignr_epi8(successor_high_u8x32, window_high_u8x32, 1);
    *next_byte_2_high_u8x32 = _mm256_alignr_epi8(successor_high_u8x32, window_high_u8x32, 2);
}

/** @brief  Expand a 32-bit lane mask into a 32-byte select vector (byte `i` = 0xFF when bit `i` is set) to drive
 *          `_mm256_blendv_epi8` in place of Ice Lake's `_mm512_mask_blend_epi8`. In-register: broadcast the mask,
 *          route the right mask byte to each output byte via `vpshufb`, isolate the per-lane bit, then `cmpeq`. */
SZ_HELPER_INLINE __m256i sz_utf8_byte_mask_from_bits_haswell_(sz_u32_t bits) {
    __m256i const mask_broadcast_u32x8 = _mm256_set1_epi32((int)bits);
    __m256i const byte_router_shuffle_u8x32 = _mm256_setr_epi8( //
        0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,         //
        2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3);
    __m256i const mask_spread_u8x32 = _mm256_shuffle_epi8(mask_broadcast_u32x8, byte_router_shuffle_u8x32);
    __m256i const bit_select_isolation_u8x32 = _mm256_setr_epi8(              //
        1, 2, 4, 8, 16, 32, 64, (char)128, 1, 2, 4, 8, 16, 32, 64, (char)128, //
        1, 2, 4, 8, 16, 32, 64, (char)128, 1, 2, 4, 8, 16, 32, 64, (char)128);
    __m256i const bit_isolated_u8x32 = _mm256_and_si256(mask_spread_u8x32, bit_select_isolation_u8x32);
    return _mm256_cmpeq_epi8(bit_isolated_u8x32, bit_select_isolation_u8x32);
}

/** @brief  Load up to 64 bytes (masked tail) and decode every lane into byte-domain halves — the AVX2 twin of
 *          @ref sz_utf8_rune_decode_window_, bit-identical to it on every lane. */
SZ_HELPER_AUTO sz_utf8_rune_window_haswell_t sz_utf8_rune_decode_window_haswell_( //
    sz_u8_t const *text, sz_size_t available) {
    sz_utf8_rune_window_haswell_t result;
    result.loaded = available < 64 ? available : 64;

    __m256i window_bytes_low_u8x32, window_bytes_high_u8x32;
    sz_utf8_load_window_haswell_(text, result.loaded, &window_bytes_low_u8x32, &window_bytes_high_u8x32);
    result.window_lo = window_bytes_low_u8x32, result.window_hi = window_bytes_high_u8x32;

    __m256i next_byte_1_low_u8x32, next_byte_1_high_u8x32, next_byte_2_low_u8x32, next_byte_2_high_u8x32;
    sz_utf8_forward_neighbours_haswell_(window_bytes_low_u8x32, window_bytes_high_u8x32, &next_byte_1_low_u8x32,
                                        &next_byte_1_high_u8x32, &next_byte_2_low_u8x32, &next_byte_2_high_u8x32);

    sz_u64_t const loaded_mask = result.loaded >= 64 ? ~(sz_u64_t)0 : (((sz_u64_t)1 << result.loaded) - 1);

    // Lead-class detection over loaded lanes: (byte & mask) == pattern, AND-clamped to loaded lanes.
    __m256i const continuation_mask_u8x32 = _mm256_set1_epi8((char)0xC0);
    __m256i const is_continuation_low_u8x32 = _mm256_cmpeq_epi8(
        _mm256_and_si256(window_bytes_low_u8x32, continuation_mask_u8x32), _mm256_set1_epi8((char)0x80));
    __m256i const is_continuation_high_u8x32 = _mm256_cmpeq_epi8(
        _mm256_and_si256(window_bytes_high_u8x32, continuation_mask_u8x32), _mm256_set1_epi8((char)0x80));
    result.continuation = sz_utf8_mask_combine_haswell_(is_continuation_low_u8x32, is_continuation_high_u8x32) &
                          loaded_mask;
    result.codepoint_starts = loaded_mask & ~result.continuation;

    __m256i const two_byte_lead_mask_u8x32 = _mm256_set1_epi8((char)0xE0);
    result.two_byte_starts = sz_utf8_mask_combine_haswell_(
                                 _mm256_cmpeq_epi8(_mm256_and_si256(window_bytes_low_u8x32, two_byte_lead_mask_u8x32),
                                                   _mm256_set1_epi8((char)0xC0)),
                                 _mm256_cmpeq_epi8(_mm256_and_si256(window_bytes_high_u8x32, two_byte_lead_mask_u8x32),
                                                   _mm256_set1_epi8((char)0xC0))) &
                             loaded_mask;
    __m256i const three_byte_lead_mask_u8x32 = _mm256_set1_epi8((char)0xF0);
    result.three_byte_starts =
        sz_utf8_mask_combine_haswell_(
            _mm256_cmpeq_epi8(_mm256_and_si256(window_bytes_low_u8x32, three_byte_lead_mask_u8x32),
                              _mm256_set1_epi8((char)0xE0)),
            _mm256_cmpeq_epi8(_mm256_and_si256(window_bytes_high_u8x32, three_byte_lead_mask_u8x32),
                              _mm256_set1_epi8((char)0xE0))) &
        loaded_mask;
    __m256i const four_byte_lead_mask_u8x32 = _mm256_set1_epi8((char)0xF8);
    result.four_byte_starts = sz_utf8_mask_combine_haswell_(
                                  _mm256_cmpeq_epi8(_mm256_and_si256(window_bytes_low_u8x32, four_byte_lead_mask_u8x32),
                                                    _mm256_set1_epi8((char)0xF0)),
                                  _mm256_cmpeq_epi8(
                                      _mm256_and_si256(window_bytes_high_u8x32, four_byte_lead_mask_u8x32),
                                      _mm256_set1_epi8((char)0xF0))) &
                              loaded_mask;

    __m256i const mask_five_low_bits_u8x32 = _mm256_set1_epi8(0x1F);
    __m256i const mask_six_low_bits_u8x32 = _mm256_set1_epi8(0x3F);
    __m256i const mask_two_low_bits_u8x32 = _mm256_set1_epi8(0x03);
    __m256i const mask_four_low_bits_u8x32 = _mm256_set1_epi8(0x0F);

    // 2-byte: codepoint = ((b0 & 0x1F) << 6) | (b1 & 0x3F); high = codepoint >> 8, low = codepoint & 0xFF.
    __m256i const high_two_byte_low_u8x32 = sz_utf8_srl8_haswell_(
        _mm256_and_si256(window_bytes_low_u8x32, mask_five_low_bits_u8x32), 2, 0x07);
    __m256i const high_two_byte_high_u8x32 = sz_utf8_srl8_haswell_(
        _mm256_and_si256(window_bytes_high_u8x32, mask_five_low_bits_u8x32), 2, 0x07);
    __m256i const low_two_byte_low_u8x32 = _mm256_or_si256(
        _mm256_slli_epi16(_mm256_and_si256(window_bytes_low_u8x32, mask_two_low_bits_u8x32), 6),
        _mm256_and_si256(next_byte_1_low_u8x32, mask_six_low_bits_u8x32));
    __m256i const low_two_byte_high_u8x32 = _mm256_or_si256(
        _mm256_slli_epi16(_mm256_and_si256(window_bytes_high_u8x32, mask_two_low_bits_u8x32), 6),
        _mm256_and_si256(next_byte_1_high_u8x32, mask_six_low_bits_u8x32));

    // 3-byte: codepoint = ((b0 & 0x0F) << 12) | ((b1 & 0x3F) << 6) | (b2 & 0x3F).
    __m256i const high_three_byte_low_u8x32 = _mm256_or_si256(
        _mm256_slli_epi16(_mm256_and_si256(window_bytes_low_u8x32, mask_four_low_bits_u8x32), 4),
        sz_utf8_srl8_haswell_(next_byte_1_low_u8x32, 2, 0x0F));
    __m256i const high_three_byte_high_u8x32 = _mm256_or_si256(
        _mm256_slli_epi16(_mm256_and_si256(window_bytes_high_u8x32, mask_four_low_bits_u8x32), 4),
        sz_utf8_srl8_haswell_(next_byte_1_high_u8x32, 2, 0x0F));
    __m256i const low_three_byte_low_u8x32 = _mm256_or_si256(
        _mm256_slli_epi16(_mm256_and_si256(next_byte_1_low_u8x32, mask_two_low_bits_u8x32), 6),
        _mm256_and_si256(next_byte_2_low_u8x32, mask_six_low_bits_u8x32));
    __m256i const low_three_byte_high_u8x32 = _mm256_or_si256(
        _mm256_slli_epi16(_mm256_and_si256(next_byte_1_high_u8x32, mask_two_low_bits_u8x32), 6),
        _mm256_and_si256(next_byte_2_high_u8x32, mask_six_low_bits_u8x32));

    // Blend 2-byte vs 3-byte per lane (AVX2 `blendv` selects on the byte MSB, so expand the 3-byte lane mask).
    __m256i const is_three_byte_low_select_u8x32 = sz_utf8_byte_mask_from_bits_haswell_(
        (sz_u32_t)(result.three_byte_starts & 0xFFFFFFFFu));
    __m256i const is_three_byte_high_select_u8x32 = sz_utf8_byte_mask_from_bits_haswell_(
        (sz_u32_t)((result.three_byte_starts >> 32) & 0xFFFFFFFFu));
    result.high_lo = _mm256_blendv_epi8(high_two_byte_low_u8x32, high_three_byte_low_u8x32,
                                        is_three_byte_low_select_u8x32);
    result.high_hi = _mm256_blendv_epi8(high_two_byte_high_u8x32, high_three_byte_high_u8x32,
                                        is_three_byte_high_select_u8x32);
    result.low_lo = _mm256_blendv_epi8(low_two_byte_low_u8x32, low_three_byte_low_u8x32,
                                       is_three_byte_low_select_u8x32);
    result.low_hi = _mm256_blendv_epi8(low_two_byte_high_u8x32, low_three_byte_high_u8x32,
                                       is_three_byte_high_select_u8x32);
    return result;
}

/** @brief  One nibble-cascade stage with a sub-256 selector: `result[lane] = table[selector[lane]*16 + within[lane]]`.
 *          Each of @p tile_count 16-byte rows is one selector value; @p within is the addressing nibble (caller-masked
 *          to `[0,16)`). The AVX2 stand-in for VBMI `vpermi2b`: there is no 32-byte byte permute, so each resident row
 *          is broadcast and shuffled by @p within, then blended in for the lanes whose @p selector picks that row.
 *          In-register — only `vbroadcasti128`/`vpshufb`/`vpcmpeqb`/`vpblendvb`.
 *
 *  ! Cost scales with @p tile_count, not with the window: every resident row is shuffled and blended on the shuffle
 *  ! port. The BMP classifiers use @ref sz_utf8_rune_flat_lookup_haswell_ instead for that reason. */
SZ_HELPER_AUTO __m256i sz_utf8_rune_cascade_stage_haswell_( //
    sz_u8_t const *table, int tile_count, __m256i selector, __m256i within) {
    __m256i result_u8x32 = _mm256_setzero_si256();
    for (int tile = 0; tile < tile_count; ++tile) {
        __m256i const lut_row_broadcast_u8x32 = _mm256_broadcastsi128_si256(
            _mm_loadu_si128((__m128i const *)(table + tile * 16)));
        __m256i const lut_row_picked_u8x32 = _mm256_shuffle_epi8(lut_row_broadcast_u8x32, within);
        __m256i const selector_match_u8x32 = _mm256_cmpeq_epi8(selector, _mm256_set1_epi8((char)tile));
        result_u8x32 = _mm256_blendv_epi8(result_u8x32, lut_row_picked_u8x32, selector_match_u8x32);
    }
    return result_u8x32;
}

/** @brief  256-entry byte LUT addressed by a per-lane byte index in `[0,256)`: `result[lane] = group_base[index[lane]]`.
 *          Two-stage `vpshufb` over 16 resident rows — the low nibble shuffles within a row, the high nibble selects
 *          the row. The AVX2 twin of the substrate `lut256` leaf. */
SZ_HELPER_INLINE __m256i sz_utf8_rune_lut256_haswell_(sz_u8_t const *group_base, __m256i index) {
    __m256i const index_within_low_u8x32 = _mm256_and_si256(index, _mm256_set1_epi8(0x0F));
    __m256i const index_selector_high_u8x32 = _mm256_and_si256(_mm256_srli_epi16(index, 4), _mm256_set1_epi8(0x0F));
    return sz_utf8_rune_cascade_stage_haswell_(group_base, 16, index_selector_high_u8x32, index_within_low_u8x32);
}

/** @brief  Narrow four `u32x8` vectors, each carrying one class byte in the low byte of every dword, into a single
 *          `u8x32` in ascending lane order. The companion of the flat-leaf `vpgatherdd`, which resolves eight lanes
 *          per gather and so needs four gathers per 32-lane window. `packus` saturates per 128-bit half, leaving the
 *          dwords interleaved, so a final `vpermd` restores the order. */
SZ_HELPER_INLINE __m256i sz_utf8_rune_pack4_u32_to_u8_haswell_( //
    __m256i first_u32x8, __m256i second_u32x8, __m256i third_u32x8, __m256i fourth_u32x8) {
    __m256i const first_second_u16x16 = _mm256_packus_epi32(first_u32x8, second_u32x8);
    __m256i const third_fourth_u16x16 = _mm256_packus_epi32(third_u32x8, fourth_u32x8);
    __m256i const interleaved_u8x32 = _mm256_packus_epi16(first_second_u16x16, third_fourth_u16x16);
    return _mm256_permutevar8x32_epi32(interleaved_u8x32, _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7));
}

/** @brief  Class byte per lane from a page-compressed flat table: `page_lut[high]` selects a 256-byte page, then
 *          `flat[page * 256 + low]` is fetched with four `vpgatherdd`. Unlike the nibble cascade this scales with the
 *          window, not the table: the cascade scanned every 16-byte tile of every stage on the shuffle port, while the
 *          gather issues on the load ports and leaves the shuffle port to the decode. @p flat must extend four bytes
 *          past its last index, since the dword gather over-reads three. */
SZ_HELPER_AUTO __m256i sz_utf8_rune_flat_lookup_haswell_( //
    sz_u8_t const *page_lut, sz_u8_t const *flat, __m256i high_bytes_u8x32, __m256i low_bytes_u8x32) {
    __m256i const page_indices_u8x32 = sz_utf8_rune_lut256_haswell_(page_lut, high_bytes_u8x32);
    __m128i const page_indices_low_u8x16 = _mm256_castsi256_si128(page_indices_u8x32);
    __m128i const page_indices_high_u8x16 = _mm256_extracti128_si256(page_indices_u8x32, 1);
    __m128i const low_bytes_low_u8x16 = _mm256_castsi256_si128(low_bytes_u8x32);
    __m128i const low_bytes_high_u8x16 = _mm256_extracti128_si256(low_bytes_u8x32, 1);
    __m128i const page_quarters_u8x16[4] = {page_indices_low_u8x16, _mm_srli_si128(page_indices_low_u8x16, 8),
                                            page_indices_high_u8x16, _mm_srli_si128(page_indices_high_u8x16, 8)};
    __m128i const low_quarters_u8x16[4] = {low_bytes_low_u8x16, _mm_srli_si128(low_bytes_low_u8x16, 8),
                                           low_bytes_high_u8x16, _mm_srli_si128(low_bytes_high_u8x16, 8)};
    __m256i gathered_u32x8[4];
    for (int quarter = 0; quarter < 4; ++quarter) {
        __m256i const flat_indices_u32x8 = _mm256_add_epi32(
            _mm256_slli_epi32(_mm256_cvtepu8_epi32(page_quarters_u8x16[quarter]), 8),
            _mm256_cvtepu8_epi32(low_quarters_u8x16[quarter]));
        gathered_u32x8[quarter] = _mm256_and_si256(_mm256_i32gather_epi32((int const *)flat, flat_indices_u32x8, 1),
                                                   _mm256_set1_epi32(0xFF));
    }
    return sz_utf8_rune_pack4_u32_to_u8_haswell_(gathered_u32x8[0], gathered_u32x8[1], gathered_u32x8[2],
                                                 gathered_u32x8[3]);
}

#pragma endregion Shared SIMD leaf substrate

#pragma region Drains

/** @brief  Left-pack the set lane indices (in [0, 64), ascending) of a 64-bit @p mask into @p out[0..popcount).
 *          BMI2 path: `tzcnt` pulls the lowest set bit's index, `blsr` clears it. On Intel Haswell `tzcnt`/`blsr`
 *          are single-uop and beat a `vpshufb` left-pack LUT; the LUT only wins where `pext`/`blsr` is microcoded
 *          (AMD pre-Zen3). */
SZ_HELPER_AUTO void sz_utf8_unpack_indices_haswell_(sz_u64_t mask, sz_u8_t *out) {
    while (mask) {
        *out++ = (sz_u8_t)sz_u64_ctz(mask);
        mask = _blsr_u64(mask); // clear the lowest set bit
    }
}

/** @brief  AVX2 forward drain — the `vpcompressb`-free twin of @ref sz_utf8_rune_drain_forward_. Emits one
 *          (start, length) per set boundary lane (ascending), honoring @p capacity and the carried previous-boundary
 *          via @p previous_io; bit-exact with the Ice Lake leaf. The set lanes are index-unpacked once (BMI2), then
 *          streamed in waves of four u64 positions (`vpmovzxbq` widen + `base`, segment starts via `vpermq` shift +
 *          `vpblendd` carry-seat, lengths via `vpsubq`), with a scalar tail for the final partial wave (no AVX2
 *          masked store). */
SZ_HELPER_AUTO sz_size_t sz_utf8_rune_drain_forward_haswell_( //
    sz_u64_t boundary, sz_size_t base, sz_size_t *starts, sz_size_t *lengths, sz_size_t produced, sz_size_t capacity,
    sz_size_t *previous_io) {
    sz_size_t const boundary_count = (sz_size_t)sz_u64_popcount(boundary);
    sz_u64_t previous = (sz_u64_t)*previous_io;
    if (boundary_count == 0 || produced >= capacity) {
        *previous_io = (sz_size_t)previous;
        return produced;
    }

    sz_u512_vec_t indices;
    sz_utf8_unpack_indices_haswell_(boundary, indices.u8s);

    __m256i const base_address_u64x4 = _mm256_set1_epi64x((long long)base);
    sz_size_t emitted = 0;
    while (emitted < boundary_count && produced < capacity) {
        sz_size_t const wave = sz_min_of_three(boundary_count - emitted, capacity - produced, 4);
        // Widen four lane indices to u64 positions: position[k] = base + indices[emitted + k].
        __m128i const indices_bytes_u8x16 = _mm_loadl_epi64((__m128i const *)(indices.u8s + emitted));
        __m256i const positions_u64x4 = _mm256_add_epi64(_mm256_cvtepu8_epi64(indices_bytes_u8x16), base_address_u64x4);
        // segment_starts = [previous, p0, p1, p2]: shift positions up one u64, seat the carry in lane 0.
        __m256i const previous_start_u64x4 = _mm256_set1_epi64x((long long)previous);
        __m256i const positions_shifted_u64x4 = _mm256_permute4x64_epi64(positions_u64x4, _MM_SHUFFLE(2, 1, 0, 0));
        __m256i const segment_starts_u64x4 = _mm256_blend_epi32(positions_shifted_u64x4, previous_start_u64x4, 0x03);
        __m256i const segment_lengths_u64x4 = _mm256_sub_epi64(positions_u64x4, segment_starts_u64x4);
        if (produced + 4 <= capacity && wave == 4) {
            _mm256_storeu_si256((__m256i *)(starts + produced), segment_starts_u64x4);
            _mm256_storeu_si256((__m256i *)(lengths + produced), segment_lengths_u64x4);
        }
        else {
            sz_size_t tail_starts[4], tail_lengths[4];
            _mm256_storeu_si256((__m256i *)tail_starts, segment_starts_u64x4);
            _mm256_storeu_si256((__m256i *)tail_lengths, segment_lengths_u64x4);
            for (sz_size_t lane = 0; lane < wave; ++lane)
                starts[produced + lane] = tail_starts[lane], lengths[produced + lane] = tail_lengths[lane];
        }
        produced += wave, emitted += wave;
        previous = (sz_u64_t)(starts[produced - 1] + lengths[produced - 1]);
    }
    *previous_io = (sz_size_t)previous;
    return produced;
}

#pragma endregion Drains

#pragma region Total vectorized decode

/** @brief  Low @p count bits set within a 32-bit lane mask (0 for `count==0`, all-ones for `count>=32`). The 32-bit
 *          twin of @ref sz_u64_mask_until_serial_, used to bound a sub-window of an AVX2 32-byte window. */
SZ_HELPER_INLINE sz_u32_t sz_u32_mask_until_serial_(sz_size_t count) {
    return count >= 32 ? 0xFFFFFFFFu : (((sz_u32_t)1 << count) - 1u);
}

/**
 *  @brief  Gather one byte per 32-bit lane from a 32-byte window addressed by the 8 per-dword byte @p offsets (each in
 *          `[0, 32)`), entirely in-register (NO `vpgatherdd`). The window's two 16-byte halves are each broadcast into
 *          both 128-bit lanes (@p window_dup_lo / @p window_dup_hi); a `vpshufb` per half routes `window[offset & 15]`
 *          into the low byte of every dword, and the high offset bit blends the two halves. Lanes pointing past lane 31
 *          read the wrapped low half, but the caller only consults lanes whose offset is a real emitted start.
 */
SZ_HELPER_INLINE __m256i sz_utf8_rune_gather8_window_haswell_( //
    __m256i window_dup_lo, __m256i window_dup_hi, __m256i offsets) {
    __m256i const offset_within_u32x8 = _mm256_and_si256(offsets, _mm256_set1_epi32(0x0F));
    __m256i const shuffle_control_u8x32 = _mm256_or_si256(offset_within_u32x8, _mm256_set1_epi32((int)0x80808000u));
    __m256i const window_low_picked_u8x32 = _mm256_shuffle_epi8(window_dup_lo, shuffle_control_u8x32);
    __m256i const window_high_picked_u8x32 = _mm256_shuffle_epi8(window_dup_hi, shuffle_control_u8x32);
    __m256i const offset_high_bit_select_u32x8 = _mm256_cmpeq_epi32(_mm256_and_si256(offsets, _mm256_set1_epi32(0x10)),
                                                                    _mm256_set1_epi32(0x10));
    return _mm256_blendv_epi8(window_low_picked_u8x32, window_high_picked_u8x32, offset_high_bit_select_u32x8);
}

/**
 *  @brief  Left-pack the set-bit positions of a 32-bit lane @p mask into @p out as a dense, ascending array of
 *          byte-offsets in [0, 32), returning the count - the AVX2 `vpcompressb`-free start-compaction shared with the
 *          NEON / LASX / PowerVSX backends. Replaces a scalar `ctz` walk with a 2 KB shuffle-LUT (`leftpack8`) keyed by
 *          each 8-bit sub-mask: for every 16-lane half the mask splits into a low and a high byte, each `vpshufb` over
 *          the LUT row of its set-bit positions, the high half offset by +8, the two stitched at `popcount(low8)` via a
 *          gap-shift `vpshufb` (no scalar per-lane index walk). The half offset `h*16` is added in vector; one loop over
 *          the two halves.
 */
SZ_HELPER_AUTO sz_size_t sz_utf8_leftpack_offsets_haswell_(sz_u32_t mask, sz_u8_t *out) {
    static sz_u8_t const leftpack8[256 * 8] = {
        0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, // 0x00
        0x00, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, // 0x01
        0x01, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, // 0x02
        0x00, 0x01, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, // 0x03
        0x02, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, // 0x04
        0x00, 0x02, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, // 0x05
        0x01, 0x02, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, // 0x06
        0x00, 0x01, 0x02, 0x80, 0x80, 0x80, 0x80, 0x80, // 0x07
        0x03, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, // 0x08
        0x00, 0x03, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, // 0x09
        0x01, 0x03, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, // 0x0A
        0x00, 0x01, 0x03, 0x80, 0x80, 0x80, 0x80, 0x80, // 0x0B
        0x02, 0x03, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, // 0x0C
        0x00, 0x02, 0x03, 0x80, 0x80, 0x80, 0x80, 0x80, // 0x0D
        0x01, 0x02, 0x03, 0x80, 0x80, 0x80, 0x80, 0x80, // 0x0E
        0x00, 0x01, 0x02, 0x03, 0x80, 0x80, 0x80, 0x80, // 0x0F
        0x04, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, // 0x10
        0x00, 0x04, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, // 0x11
        0x01, 0x04, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, // 0x12
        0x00, 0x01, 0x04, 0x80, 0x80, 0x80, 0x80, 0x80, // 0x13
        0x02, 0x04, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, // 0x14
        0x00, 0x02, 0x04, 0x80, 0x80, 0x80, 0x80, 0x80, // 0x15
        0x01, 0x02, 0x04, 0x80, 0x80, 0x80, 0x80, 0x80, // 0x16
        0x00, 0x01, 0x02, 0x04, 0x80, 0x80, 0x80, 0x80, // 0x17
        0x03, 0x04, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, // 0x18
        0x00, 0x03, 0x04, 0x80, 0x80, 0x80, 0x80, 0x80, // 0x19
        0x01, 0x03, 0x04, 0x80, 0x80, 0x80, 0x80, 0x80, // 0x1A
        0x00, 0x01, 0x03, 0x04, 0x80, 0x80, 0x80, 0x80, // 0x1B
        0x02, 0x03, 0x04, 0x80, 0x80, 0x80, 0x80, 0x80, // 0x1C
        0x00, 0x02, 0x03, 0x04, 0x80, 0x80, 0x80, 0x80, // 0x1D
        0x01, 0x02, 0x03, 0x04, 0x80, 0x80, 0x80, 0x80, // 0x1E
        0x00, 0x01, 0x02, 0x03, 0x04, 0x80, 0x80, 0x80, // 0x1F
        0x05, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, // 0x20
        0x00, 0x05, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, // 0x21
        0x01, 0x05, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, // 0x22
        0x00, 0x01, 0x05, 0x80, 0x80, 0x80, 0x80, 0x80, // 0x23
        0x02, 0x05, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, // 0x24
        0x00, 0x02, 0x05, 0x80, 0x80, 0x80, 0x80, 0x80, // 0x25
        0x01, 0x02, 0x05, 0x80, 0x80, 0x80, 0x80, 0x80, // 0x26
        0x00, 0x01, 0x02, 0x05, 0x80, 0x80, 0x80, 0x80, // 0x27
        0x03, 0x05, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, // 0x28
        0x00, 0x03, 0x05, 0x80, 0x80, 0x80, 0x80, 0x80, // 0x29
        0x01, 0x03, 0x05, 0x80, 0x80, 0x80, 0x80, 0x80, // 0x2A
        0x00, 0x01, 0x03, 0x05, 0x80, 0x80, 0x80, 0x80, // 0x2B
        0x02, 0x03, 0x05, 0x80, 0x80, 0x80, 0x80, 0x80, // 0x2C
        0x00, 0x02, 0x03, 0x05, 0x80, 0x80, 0x80, 0x80, // 0x2D
        0x01, 0x02, 0x03, 0x05, 0x80, 0x80, 0x80, 0x80, // 0x2E
        0x00, 0x01, 0x02, 0x03, 0x05, 0x80, 0x80, 0x80, // 0x2F
        0x04, 0x05, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, // 0x30
        0x00, 0x04, 0x05, 0x80, 0x80, 0x80, 0x80, 0x80, // 0x31
        0x01, 0x04, 0x05, 0x80, 0x80, 0x80, 0x80, 0x80, // 0x32
        0x00, 0x01, 0x04, 0x05, 0x80, 0x80, 0x80, 0x80, // 0x33
        0x02, 0x04, 0x05, 0x80, 0x80, 0x80, 0x80, 0x80, // 0x34
        0x00, 0x02, 0x04, 0x05, 0x80, 0x80, 0x80, 0x80, // 0x35
        0x01, 0x02, 0x04, 0x05, 0x80, 0x80, 0x80, 0x80, // 0x36
        0x00, 0x01, 0x02, 0x04, 0x05, 0x80, 0x80, 0x80, // 0x37
        0x03, 0x04, 0x05, 0x80, 0x80, 0x80, 0x80, 0x80, // 0x38
        0x00, 0x03, 0x04, 0x05, 0x80, 0x80, 0x80, 0x80, // 0x39
        0x01, 0x03, 0x04, 0x05, 0x80, 0x80, 0x80, 0x80, // 0x3A
        0x00, 0x01, 0x03, 0x04, 0x05, 0x80, 0x80, 0x80, // 0x3B
        0x02, 0x03, 0x04, 0x05, 0x80, 0x80, 0x80, 0x80, // 0x3C
        0x00, 0x02, 0x03, 0x04, 0x05, 0x80, 0x80, 0x80, // 0x3D
        0x01, 0x02, 0x03, 0x04, 0x05, 0x80, 0x80, 0x80, // 0x3E
        0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x80, 0x80, // 0x3F
        0x06, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, // 0x40
        0x00, 0x06, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, // 0x41
        0x01, 0x06, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, // 0x42
        0x00, 0x01, 0x06, 0x80, 0x80, 0x80, 0x80, 0x80, // 0x43
        0x02, 0x06, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, // 0x44
        0x00, 0x02, 0x06, 0x80, 0x80, 0x80, 0x80, 0x80, // 0x45
        0x01, 0x02, 0x06, 0x80, 0x80, 0x80, 0x80, 0x80, // 0x46
        0x00, 0x01, 0x02, 0x06, 0x80, 0x80, 0x80, 0x80, // 0x47
        0x03, 0x06, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, // 0x48
        0x00, 0x03, 0x06, 0x80, 0x80, 0x80, 0x80, 0x80, // 0x49
        0x01, 0x03, 0x06, 0x80, 0x80, 0x80, 0x80, 0x80, // 0x4A
        0x00, 0x01, 0x03, 0x06, 0x80, 0x80, 0x80, 0x80, // 0x4B
        0x02, 0x03, 0x06, 0x80, 0x80, 0x80, 0x80, 0x80, // 0x4C
        0x00, 0x02, 0x03, 0x06, 0x80, 0x80, 0x80, 0x80, // 0x4D
        0x01, 0x02, 0x03, 0x06, 0x80, 0x80, 0x80, 0x80, // 0x4E
        0x00, 0x01, 0x02, 0x03, 0x06, 0x80, 0x80, 0x80, // 0x4F
        0x04, 0x06, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, // 0x50
        0x00, 0x04, 0x06, 0x80, 0x80, 0x80, 0x80, 0x80, // 0x51
        0x01, 0x04, 0x06, 0x80, 0x80, 0x80, 0x80, 0x80, // 0x52
        0x00, 0x01, 0x04, 0x06, 0x80, 0x80, 0x80, 0x80, // 0x53
        0x02, 0x04, 0x06, 0x80, 0x80, 0x80, 0x80, 0x80, // 0x54
        0x00, 0x02, 0x04, 0x06, 0x80, 0x80, 0x80, 0x80, // 0x55
        0x01, 0x02, 0x04, 0x06, 0x80, 0x80, 0x80, 0x80, // 0x56
        0x00, 0x01, 0x02, 0x04, 0x06, 0x80, 0x80, 0x80, // 0x57
        0x03, 0x04, 0x06, 0x80, 0x80, 0x80, 0x80, 0x80, // 0x58
        0x00, 0x03, 0x04, 0x06, 0x80, 0x80, 0x80, 0x80, // 0x59
        0x01, 0x03, 0x04, 0x06, 0x80, 0x80, 0x80, 0x80, // 0x5A
        0x00, 0x01, 0x03, 0x04, 0x06, 0x80, 0x80, 0x80, // 0x5B
        0x02, 0x03, 0x04, 0x06, 0x80, 0x80, 0x80, 0x80, // 0x5C
        0x00, 0x02, 0x03, 0x04, 0x06, 0x80, 0x80, 0x80, // 0x5D
        0x01, 0x02, 0x03, 0x04, 0x06, 0x80, 0x80, 0x80, // 0x5E
        0x00, 0x01, 0x02, 0x03, 0x04, 0x06, 0x80, 0x80, // 0x5F
        0x05, 0x06, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, // 0x60
        0x00, 0x05, 0x06, 0x80, 0x80, 0x80, 0x80, 0x80, // 0x61
        0x01, 0x05, 0x06, 0x80, 0x80, 0x80, 0x80, 0x80, // 0x62
        0x00, 0x01, 0x05, 0x06, 0x80, 0x80, 0x80, 0x80, // 0x63
        0x02, 0x05, 0x06, 0x80, 0x80, 0x80, 0x80, 0x80, // 0x64
        0x00, 0x02, 0x05, 0x06, 0x80, 0x80, 0x80, 0x80, // 0x65
        0x01, 0x02, 0x05, 0x06, 0x80, 0x80, 0x80, 0x80, // 0x66
        0x00, 0x01, 0x02, 0x05, 0x06, 0x80, 0x80, 0x80, // 0x67
        0x03, 0x05, 0x06, 0x80, 0x80, 0x80, 0x80, 0x80, // 0x68
        0x00, 0x03, 0x05, 0x06, 0x80, 0x80, 0x80, 0x80, // 0x69
        0x01, 0x03, 0x05, 0x06, 0x80, 0x80, 0x80, 0x80, // 0x6A
        0x00, 0x01, 0x03, 0x05, 0x06, 0x80, 0x80, 0x80, // 0x6B
        0x02, 0x03, 0x05, 0x06, 0x80, 0x80, 0x80, 0x80, // 0x6C
        0x00, 0x02, 0x03, 0x05, 0x06, 0x80, 0x80, 0x80, // 0x6D
        0x01, 0x02, 0x03, 0x05, 0x06, 0x80, 0x80, 0x80, // 0x6E
        0x00, 0x01, 0x02, 0x03, 0x05, 0x06, 0x80, 0x80, // 0x6F
        0x04, 0x05, 0x06, 0x80, 0x80, 0x80, 0x80, 0x80, // 0x70
        0x00, 0x04, 0x05, 0x06, 0x80, 0x80, 0x80, 0x80, // 0x71
        0x01, 0x04, 0x05, 0x06, 0x80, 0x80, 0x80, 0x80, // 0x72
        0x00, 0x01, 0x04, 0x05, 0x06, 0x80, 0x80, 0x80, // 0x73
        0x02, 0x04, 0x05, 0x06, 0x80, 0x80, 0x80, 0x80, // 0x74
        0x00, 0x02, 0x04, 0x05, 0x06, 0x80, 0x80, 0x80, // 0x75
        0x01, 0x02, 0x04, 0x05, 0x06, 0x80, 0x80, 0x80, // 0x76
        0x00, 0x01, 0x02, 0x04, 0x05, 0x06, 0x80, 0x80, // 0x77
        0x03, 0x04, 0x05, 0x06, 0x80, 0x80, 0x80, 0x80, // 0x78
        0x00, 0x03, 0x04, 0x05, 0x06, 0x80, 0x80, 0x80, // 0x79
        0x01, 0x03, 0x04, 0x05, 0x06, 0x80, 0x80, 0x80, // 0x7A
        0x00, 0x01, 0x03, 0x04, 0x05, 0x06, 0x80, 0x80, // 0x7B
        0x02, 0x03, 0x04, 0x05, 0x06, 0x80, 0x80, 0x80, // 0x7C
        0x00, 0x02, 0x03, 0x04, 0x05, 0x06, 0x80, 0x80, // 0x7D
        0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x80, 0x80, // 0x7E
        0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x80, // 0x7F
        0x07, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, // 0x80
        0x00, 0x07, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, // 0x81
        0x01, 0x07, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, // 0x82
        0x00, 0x01, 0x07, 0x80, 0x80, 0x80, 0x80, 0x80, // 0x83
        0x02, 0x07, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, // 0x84
        0x00, 0x02, 0x07, 0x80, 0x80, 0x80, 0x80, 0x80, // 0x85
        0x01, 0x02, 0x07, 0x80, 0x80, 0x80, 0x80, 0x80, // 0x86
        0x00, 0x01, 0x02, 0x07, 0x80, 0x80, 0x80, 0x80, // 0x87
        0x03, 0x07, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, // 0x88
        0x00, 0x03, 0x07, 0x80, 0x80, 0x80, 0x80, 0x80, // 0x89
        0x01, 0x03, 0x07, 0x80, 0x80, 0x80, 0x80, 0x80, // 0x8A
        0x00, 0x01, 0x03, 0x07, 0x80, 0x80, 0x80, 0x80, // 0x8B
        0x02, 0x03, 0x07, 0x80, 0x80, 0x80, 0x80, 0x80, // 0x8C
        0x00, 0x02, 0x03, 0x07, 0x80, 0x80, 0x80, 0x80, // 0x8D
        0x01, 0x02, 0x03, 0x07, 0x80, 0x80, 0x80, 0x80, // 0x8E
        0x00, 0x01, 0x02, 0x03, 0x07, 0x80, 0x80, 0x80, // 0x8F
        0x04, 0x07, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, // 0x90
        0x00, 0x04, 0x07, 0x80, 0x80, 0x80, 0x80, 0x80, // 0x91
        0x01, 0x04, 0x07, 0x80, 0x80, 0x80, 0x80, 0x80, // 0x92
        0x00, 0x01, 0x04, 0x07, 0x80, 0x80, 0x80, 0x80, // 0x93
        0x02, 0x04, 0x07, 0x80, 0x80, 0x80, 0x80, 0x80, // 0x94
        0x00, 0x02, 0x04, 0x07, 0x80, 0x80, 0x80, 0x80, // 0x95
        0x01, 0x02, 0x04, 0x07, 0x80, 0x80, 0x80, 0x80, // 0x96
        0x00, 0x01, 0x02, 0x04, 0x07, 0x80, 0x80, 0x80, // 0x97
        0x03, 0x04, 0x07, 0x80, 0x80, 0x80, 0x80, 0x80, // 0x98
        0x00, 0x03, 0x04, 0x07, 0x80, 0x80, 0x80, 0x80, // 0x99
        0x01, 0x03, 0x04, 0x07, 0x80, 0x80, 0x80, 0x80, // 0x9A
        0x00, 0x01, 0x03, 0x04, 0x07, 0x80, 0x80, 0x80, // 0x9B
        0x02, 0x03, 0x04, 0x07, 0x80, 0x80, 0x80, 0x80, // 0x9C
        0x00, 0x02, 0x03, 0x04, 0x07, 0x80, 0x80, 0x80, // 0x9D
        0x01, 0x02, 0x03, 0x04, 0x07, 0x80, 0x80, 0x80, // 0x9E
        0x00, 0x01, 0x02, 0x03, 0x04, 0x07, 0x80, 0x80, // 0x9F
        0x05, 0x07, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, // 0xA0
        0x00, 0x05, 0x07, 0x80, 0x80, 0x80, 0x80, 0x80, // 0xA1
        0x01, 0x05, 0x07, 0x80, 0x80, 0x80, 0x80, 0x80, // 0xA2
        0x00, 0x01, 0x05, 0x07, 0x80, 0x80, 0x80, 0x80, // 0xA3
        0x02, 0x05, 0x07, 0x80, 0x80, 0x80, 0x80, 0x80, // 0xA4
        0x00, 0x02, 0x05, 0x07, 0x80, 0x80, 0x80, 0x80, // 0xA5
        0x01, 0x02, 0x05, 0x07, 0x80, 0x80, 0x80, 0x80, // 0xA6
        0x00, 0x01, 0x02, 0x05, 0x07, 0x80, 0x80, 0x80, // 0xA7
        0x03, 0x05, 0x07, 0x80, 0x80, 0x80, 0x80, 0x80, // 0xA8
        0x00, 0x03, 0x05, 0x07, 0x80, 0x80, 0x80, 0x80, // 0xA9
        0x01, 0x03, 0x05, 0x07, 0x80, 0x80, 0x80, 0x80, // 0xAA
        0x00, 0x01, 0x03, 0x05, 0x07, 0x80, 0x80, 0x80, // 0xAB
        0x02, 0x03, 0x05, 0x07, 0x80, 0x80, 0x80, 0x80, // 0xAC
        0x00, 0x02, 0x03, 0x05, 0x07, 0x80, 0x80, 0x80, // 0xAD
        0x01, 0x02, 0x03, 0x05, 0x07, 0x80, 0x80, 0x80, // 0xAE
        0x00, 0x01, 0x02, 0x03, 0x05, 0x07, 0x80, 0x80, // 0xAF
        0x04, 0x05, 0x07, 0x80, 0x80, 0x80, 0x80, 0x80, // 0xB0
        0x00, 0x04, 0x05, 0x07, 0x80, 0x80, 0x80, 0x80, // 0xB1
        0x01, 0x04, 0x05, 0x07, 0x80, 0x80, 0x80, 0x80, // 0xB2
        0x00, 0x01, 0x04, 0x05, 0x07, 0x80, 0x80, 0x80, // 0xB3
        0x02, 0x04, 0x05, 0x07, 0x80, 0x80, 0x80, 0x80, // 0xB4
        0x00, 0x02, 0x04, 0x05, 0x07, 0x80, 0x80, 0x80, // 0xB5
        0x01, 0x02, 0x04, 0x05, 0x07, 0x80, 0x80, 0x80, // 0xB6
        0x00, 0x01, 0x02, 0x04, 0x05, 0x07, 0x80, 0x80, // 0xB7
        0x03, 0x04, 0x05, 0x07, 0x80, 0x80, 0x80, 0x80, // 0xB8
        0x00, 0x03, 0x04, 0x05, 0x07, 0x80, 0x80, 0x80, // 0xB9
        0x01, 0x03, 0x04, 0x05, 0x07, 0x80, 0x80, 0x80, // 0xBA
        0x00, 0x01, 0x03, 0x04, 0x05, 0x07, 0x80, 0x80, // 0xBB
        0x02, 0x03, 0x04, 0x05, 0x07, 0x80, 0x80, 0x80, // 0xBC
        0x00, 0x02, 0x03, 0x04, 0x05, 0x07, 0x80, 0x80, // 0xBD
        0x01, 0x02, 0x03, 0x04, 0x05, 0x07, 0x80, 0x80, // 0xBE
        0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x07, 0x80, // 0xBF
        0x06, 0x07, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, // 0xC0
        0x00, 0x06, 0x07, 0x80, 0x80, 0x80, 0x80, 0x80, // 0xC1
        0x01, 0x06, 0x07, 0x80, 0x80, 0x80, 0x80, 0x80, // 0xC2
        0x00, 0x01, 0x06, 0x07, 0x80, 0x80, 0x80, 0x80, // 0xC3
        0x02, 0x06, 0x07, 0x80, 0x80, 0x80, 0x80, 0x80, // 0xC4
        0x00, 0x02, 0x06, 0x07, 0x80, 0x80, 0x80, 0x80, // 0xC5
        0x01, 0x02, 0x06, 0x07, 0x80, 0x80, 0x80, 0x80, // 0xC6
        0x00, 0x01, 0x02, 0x06, 0x07, 0x80, 0x80, 0x80, // 0xC7
        0x03, 0x06, 0x07, 0x80, 0x80, 0x80, 0x80, 0x80, // 0xC8
        0x00, 0x03, 0x06, 0x07, 0x80, 0x80, 0x80, 0x80, // 0xC9
        0x01, 0x03, 0x06, 0x07, 0x80, 0x80, 0x80, 0x80, // 0xCA
        0x00, 0x01, 0x03, 0x06, 0x07, 0x80, 0x80, 0x80, // 0xCB
        0x02, 0x03, 0x06, 0x07, 0x80, 0x80, 0x80, 0x80, // 0xCC
        0x00, 0x02, 0x03, 0x06, 0x07, 0x80, 0x80, 0x80, // 0xCD
        0x01, 0x02, 0x03, 0x06, 0x07, 0x80, 0x80, 0x80, // 0xCE
        0x00, 0x01, 0x02, 0x03, 0x06, 0x07, 0x80, 0x80, // 0xCF
        0x04, 0x06, 0x07, 0x80, 0x80, 0x80, 0x80, 0x80, // 0xD0
        0x00, 0x04, 0x06, 0x07, 0x80, 0x80, 0x80, 0x80, // 0xD1
        0x01, 0x04, 0x06, 0x07, 0x80, 0x80, 0x80, 0x80, // 0xD2
        0x00, 0x01, 0x04, 0x06, 0x07, 0x80, 0x80, 0x80, // 0xD3
        0x02, 0x04, 0x06, 0x07, 0x80, 0x80, 0x80, 0x80, // 0xD4
        0x00, 0x02, 0x04, 0x06, 0x07, 0x80, 0x80, 0x80, // 0xD5
        0x01, 0x02, 0x04, 0x06, 0x07, 0x80, 0x80, 0x80, // 0xD6
        0x00, 0x01, 0x02, 0x04, 0x06, 0x07, 0x80, 0x80, // 0xD7
        0x03, 0x04, 0x06, 0x07, 0x80, 0x80, 0x80, 0x80, // 0xD8
        0x00, 0x03, 0x04, 0x06, 0x07, 0x80, 0x80, 0x80, // 0xD9
        0x01, 0x03, 0x04, 0x06, 0x07, 0x80, 0x80, 0x80, // 0xDA
        0x00, 0x01, 0x03, 0x04, 0x06, 0x07, 0x80, 0x80, // 0xDB
        0x02, 0x03, 0x04, 0x06, 0x07, 0x80, 0x80, 0x80, // 0xDC
        0x00, 0x02, 0x03, 0x04, 0x06, 0x07, 0x80, 0x80, // 0xDD
        0x01, 0x02, 0x03, 0x04, 0x06, 0x07, 0x80, 0x80, // 0xDE
        0x00, 0x01, 0x02, 0x03, 0x04, 0x06, 0x07, 0x80, // 0xDF
        0x05, 0x06, 0x07, 0x80, 0x80, 0x80, 0x80, 0x80, // 0xE0
        0x00, 0x05, 0x06, 0x07, 0x80, 0x80, 0x80, 0x80, // 0xE1
        0x01, 0x05, 0x06, 0x07, 0x80, 0x80, 0x80, 0x80, // 0xE2
        0x00, 0x01, 0x05, 0x06, 0x07, 0x80, 0x80, 0x80, // 0xE3
        0x02, 0x05, 0x06, 0x07, 0x80, 0x80, 0x80, 0x80, // 0xE4
        0x00, 0x02, 0x05, 0x06, 0x07, 0x80, 0x80, 0x80, // 0xE5
        0x01, 0x02, 0x05, 0x06, 0x07, 0x80, 0x80, 0x80, // 0xE6
        0x00, 0x01, 0x02, 0x05, 0x06, 0x07, 0x80, 0x80, // 0xE7
        0x03, 0x05, 0x06, 0x07, 0x80, 0x80, 0x80, 0x80, // 0xE8
        0x00, 0x03, 0x05, 0x06, 0x07, 0x80, 0x80, 0x80, // 0xE9
        0x01, 0x03, 0x05, 0x06, 0x07, 0x80, 0x80, 0x80, // 0xEA
        0x00, 0x01, 0x03, 0x05, 0x06, 0x07, 0x80, 0x80, // 0xEB
        0x02, 0x03, 0x05, 0x06, 0x07, 0x80, 0x80, 0x80, // 0xEC
        0x00, 0x02, 0x03, 0x05, 0x06, 0x07, 0x80, 0x80, // 0xED
        0x01, 0x02, 0x03, 0x05, 0x06, 0x07, 0x80, 0x80, // 0xEE
        0x00, 0x01, 0x02, 0x03, 0x05, 0x06, 0x07, 0x80, // 0xEF
        0x04, 0x05, 0x06, 0x07, 0x80, 0x80, 0x80, 0x80, // 0xF0
        0x00, 0x04, 0x05, 0x06, 0x07, 0x80, 0x80, 0x80, // 0xF1
        0x01, 0x04, 0x05, 0x06, 0x07, 0x80, 0x80, 0x80, // 0xF2
        0x00, 0x01, 0x04, 0x05, 0x06, 0x07, 0x80, 0x80, // 0xF3
        0x02, 0x04, 0x05, 0x06, 0x07, 0x80, 0x80, 0x80, // 0xF4
        0x00, 0x02, 0x04, 0x05, 0x06, 0x07, 0x80, 0x80, // 0xF5
        0x01, 0x02, 0x04, 0x05, 0x06, 0x07, 0x80, 0x80, // 0xF6
        0x00, 0x01, 0x02, 0x04, 0x05, 0x06, 0x07, 0x80, // 0xF7
        0x03, 0x04, 0x05, 0x06, 0x07, 0x80, 0x80, 0x80, // 0xF8
        0x00, 0x03, 0x04, 0x05, 0x06, 0x07, 0x80, 0x80, // 0xF9
        0x01, 0x03, 0x04, 0x05, 0x06, 0x07, 0x80, 0x80, // 0xFA
        0x00, 0x01, 0x03, 0x04, 0x05, 0x06, 0x07, 0x80, // 0xFB
        0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x80, 0x80, // 0xFC
        0x00, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x80, // 0xFD
        0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x80, // 0xFE
        0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, // 0xFF
    };
    __m128i const lane_iota_u8x16 = _mm_setr_epi8(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
    __m128i const constant_eight_u8x16 = _mm_set1_epi8(8);

    sz_size_t produced = 0;
    for (int half = 0; half < 2; ++half) {
        sz_u32_t const half_mask = (mask >> (half * 16)) & 0xFFFFu;
        if (half_mask == 0) continue;
        sz_u32_t const low8 = half_mask & 0xFFu;
        sz_u32_t const high8 = (half_mask >> 8) & 0xFFu;
        int const count_low = sz_u64_popcount(low8);
        int const count = sz_u64_popcount(half_mask);

        // Ascending set-bit positions for each 8-bit byte; the high byte's positions are +8 (lanes 8..15). Unused slots
        // are 0x80 (high bit set → `vpshufb` reads 0, never stored).
        __m128i const packed_offsets_low_u8x16 = _mm_loadl_epi64((__m128i const *)(leftpack8 + low8 * 8));
        __m128i const packed_offsets_high_raw_u8x16 = _mm_loadl_epi64((__m128i const *)(leftpack8 + high8 * 8));
        __m128i const packed_offsets_high_adjusted_u8x16 = _mm_add_epi8(packed_offsets_high_raw_u8x16,
                                                                        _mm_set1_epi8(8));
        __m128i const packed_halves_u8x16 = _mm_unpacklo_epi64(packed_offsets_low_u8x16,
                                                               packed_offsets_high_adjusted_u8x16);

        // Stitch: output lane j reads source j for j < count_low, else j + (8 - count_low) (skip the low gap).
        __m128i const greater_equal_count_u8x16 = _mm_cmpgt_epi8(lane_iota_u8x16, _mm_set1_epi8((char)(count_low - 1)));
        __m128i const gap_adjustment_u8x16 = _mm_and_si128(
            greater_equal_count_u8x16, _mm_sub_epi8(constant_eight_u8x16, _mm_set1_epi8((char)count_low)));
        __m128i const stitch_source_u8x16 = _mm_add_epi8(lane_iota_u8x16, gap_adjustment_u8x16);
        __m128i const stitched_offsets_u8x16 = _mm_shuffle_epi8(packed_halves_u8x16, stitch_source_u8x16);

        // Promote the half-relative offsets to global byte-offsets and append the `count` dense lanes.
        __m128i const global_offsets_u8x16 = _mm_add_epi8(stitched_offsets_u8x16, _mm_set1_epi8((char)(half * 16)));
        sz_u128_vec_t lanes;
        _mm_storeu_si128((__m128i *)lanes.u8s, global_offsets_u8x16);
        for (int lane = 0; lane < count; ++lane) out[produced + (sz_size_t)lane] = lanes.u8s[lane];
        produced += (sz_size_t)count;
    }
    return produced;
}

/**
 *  @brief  Decode the dense emitted-start lanes of one classified AVX2 window into sequential UTF-32 runes, the
 *          `vpcompressb`-free twin of @ref sz_utf8_rune_drain_icelake_. The set lanes of @p emit_starts are left-packed
 *          once into a dense ascending byte-offset array (BMI2 `tzcnt`/`blsr`, @ref sz_utf8_unpack_indices_haswell_),
 *          and each block of up to 8 starts loads its offsets to dwords, gathers the lead and trailing bytes from the
 *          in-register window, width-blends 1/2/3/4-byte lanes branchlessly, and — only when @p has_ill — overwrites
 *          ill-formed lanes with U+FFFD (their per-emitted-lane flag gathered with the SAME packed offsets out of
 *          @p ill_formed_lanes).
 *  @return Number of runes emitted; sets @p last_off_out to the last emitted start's window byte-offset (the caller
 *          turns it into the resume cursor by adding that lane's maximal-subpart length).
 */
SZ_HELPER_AUTO sz_size_t sz_utf8_rune_drain_haswell_(               //
    __m256i window, __m256i ill_formed_lanes, sz_u32_t emit_starts, //
    int has_three, int has_four, int has_ill,                       //
    sz_size_t emit_count, sz_rune_t *runes, sz_size_t capacity, sz_u8_t *last_off_out) {

    // Left-pack the emitted-start byte-offsets (ascending) into a dense array. On Intel Haswell single-uop
    // `tzcnt`/`blsr` (@ref sz_utf8_unpack_indices_haswell_) beat the `vpshufb` stitch LUT; the LUT only wins where
    // `blsr` is microcoded.
    sz_u512_vec_t offsets;
    sz_utf8_unpack_indices_haswell_(emit_starts, offsets.u8s);

    __m256i const window_low_broadcast_u8x32 = _mm256_broadcastsi128_si256(_mm256_castsi256_si128(window));
    __m256i const window_high_broadcast_u8x32 = _mm256_broadcastsi128_si256(_mm256_extracti128_si256(window, 1));

    sz_size_t const want = emit_count < capacity ? emit_count : capacity;
    sz_size_t produced = 0;
    for (sz_size_t block_start = 0; block_start < want; block_start += 8) {
        __m128i const offsets_bytes_u8x16 = _mm_loadl_epi64((__m128i const *)(offsets.u8s + block_start));
        __m256i const offsets_u32x8 = _mm256_cvtepu8_epi32(offsets_bytes_u8x16);
        __m256i const lead_byte_u32x8 = sz_utf8_rune_gather8_window_haswell_(
            window_low_broadcast_u8x32, window_high_broadcast_u8x32, offsets_u32x8);
        __m256i const second_byte_u32x8 = sz_utf8_rune_gather8_window_haswell_(
            window_low_broadcast_u8x32, window_high_broadcast_u8x32,
            _mm256_add_epi32(offsets_u32x8, _mm256_set1_epi32(1)));

        // Width-blend 1/2/3/4-byte lead lanes. The lead-byte-range predicates are cheap (from the lead byte alone),
        // so they stay unconditional; the 2-byte form needs only the second byte, also already gathered. The wider
        // trailing bytes are gathered CONDITIONALLY via two sibling `if`s so a CJK (3-byte-only) window never pays
        // for the 4th-byte gather + 4-byte assembly, and ASCII/2-byte windows skip both.
        __m256i const is_greater_equal_0xc0_u32x8 = _mm256_cmpgt_epi32(lead_byte_u32x8, _mm256_set1_epi32(0xC0 - 1));
        __m256i const is_greater_equal_0xe0_u32x8 = _mm256_cmpgt_epi32(lead_byte_u32x8, _mm256_set1_epi32(0xE0 - 1));
        __m256i const is_greater_equal_0xf0_u32x8 = _mm256_cmpgt_epi32(lead_byte_u32x8, _mm256_set1_epi32(0xF0 - 1));
        __m256i const is_two_byte_u32x8 = _mm256_andnot_si256(is_greater_equal_0xe0_u32x8, is_greater_equal_0xc0_u32x8);
        __m256i const is_three_byte_u32x8 = _mm256_andnot_si256(is_greater_equal_0xf0_u32x8,
                                                                is_greater_equal_0xe0_u32x8);
        __m256i const two_byte_codepoints_u32x8 = _mm256_or_si256(
            _mm256_slli_epi32(_mm256_and_si256(lead_byte_u32x8, _mm256_set1_epi32(0x1F)), 6),
            _mm256_and_si256(second_byte_u32x8, _mm256_set1_epi32(0x3F)));
        // ASCII / orphan-byte default kept by the lead byte; 2-byte blends in next (lead + second only).
        __m256i codepoints_u32x8 = _mm256_blendv_epi8(lead_byte_u32x8, two_byte_codepoints_u32x8, is_two_byte_u32x8);

        __m256i third_byte_u32x8 = _mm256_setzero_si256(); // hoisted so the 4-byte sibling reuses it.
        if (has_three) {
            third_byte_u32x8 = sz_utf8_rune_gather8_window_haswell_(
                window_low_broadcast_u8x32, window_high_broadcast_u8x32,
                _mm256_add_epi32(offsets_u32x8, _mm256_set1_epi32(2)));
            __m256i const three_byte_codepoints_u32x8 = _mm256_or_si256(
                _mm256_or_si256(_mm256_slli_epi32(_mm256_and_si256(lead_byte_u32x8, _mm256_set1_epi32(0x0F)), 12),
                                _mm256_slli_epi32(_mm256_and_si256(second_byte_u32x8, _mm256_set1_epi32(0x3F)), 6)),
                _mm256_and_si256(third_byte_u32x8, _mm256_set1_epi32(0x3F)));
            codepoints_u32x8 = _mm256_blendv_epi8(codepoints_u32x8, three_byte_codepoints_u32x8, is_three_byte_u32x8);
        }
        if (has_four) { // SIBLING, not nested; has_four ⟹ has_three so third_byte_u32x8 is already set.
            __m256i const fourth_byte_u32x8 = sz_utf8_rune_gather8_window_haswell_(
                window_low_broadcast_u8x32, window_high_broadcast_u8x32,
                _mm256_add_epi32(offsets_u32x8, _mm256_set1_epi32(3)));
            __m256i const four_byte_codepoints_u32x8 = _mm256_or_si256(
                _mm256_or_si256(_mm256_slli_epi32(_mm256_and_si256(lead_byte_u32x8, _mm256_set1_epi32(0x07)), 18),
                                _mm256_slli_epi32(_mm256_and_si256(second_byte_u32x8, _mm256_set1_epi32(0x3F)), 12)),
                _mm256_or_si256(_mm256_slli_epi32(_mm256_and_si256(third_byte_u32x8, _mm256_set1_epi32(0x3F)), 6),
                                _mm256_and_si256(fourth_byte_u32x8, _mm256_set1_epi32(0x3F))));
            codepoints_u32x8 = _mm256_blendv_epi8(codepoints_u32x8, four_byte_codepoints_u32x8,
                                                  is_greater_equal_0xf0_u32x8);
        }

        // Ill-formed lanes collapse to U+FFFD; well-formed-only windows (the overwhelming common case) skip the gather.
        if (has_ill) {
            __m256i const ill_formed_low_broadcast_u8x32 = _mm256_broadcastsi128_si256(
                _mm256_castsi256_si128(ill_formed_lanes));
            __m256i const ill_formed_high_broadcast_u8x32 = _mm256_broadcastsi128_si256(
                _mm256_extracti128_si256(ill_formed_lanes, 1));
            __m256i const ill_formed_flags_u32x8 = sz_utf8_rune_gather8_window_haswell_(
                ill_formed_low_broadcast_u8x32, ill_formed_high_broadcast_u8x32, offsets_u32x8);
            __m256i const ill_formed_detected_u32x8 = _mm256_cmpgt_epi32(ill_formed_flags_u32x8,
                                                                         _mm256_setzero_si256());
            codepoints_u32x8 = _mm256_blendv_epi8(codepoints_u32x8, _mm256_set1_epi32((int)sz_rune_replacement_k),
                                                  ill_formed_detected_u32x8);
        }

        sz_size_t lanes = want - produced;
        if (lanes > 8) lanes = 8;
        if (lanes == 8) { _mm256_storeu_si256((__m256i *)(runes + produced), codepoints_u32x8); }
        else {
            sz_rune_t tail[8];
            _mm256_storeu_si256((__m256i *)tail, codepoints_u32x8);
            for (sz_size_t lane = 0; lane < lanes; ++lane) runes[produced + lane] = tail[lane];
        }
        produced += lanes;
    }

    *last_off_out = offsets.u8s[produced - 1];
    return produced;
}

/**
 *  @brief  Decode one 32-byte window of @p text into dense UTF-32 @p runes by the uniform "classify → per-lane
 *          well-formed + orphan promotion → compact emitted starts → in-register gather → width-blend → blend
 *          U+FFFD" path - the AVX2 twin of @ref sz_utf8_decode_once_icelake_, bit-exact with the serial
 *          reference (one U+FFFD per maximal ill-formed subpart, Unicode 17.0 §3.9 / W3C). Pure-ASCII windows take
 *          a `vpmovzxbd` widen lane. The step declines (`*runes_unpacked == 0`, cursor unchanged) ONLY when the
 *          first lead's declared sequence crosses the window edge (a boundary truncation), which the public entry
 *          finalizes without a serial re-decode. The decode is TOTAL: no decline-to-serial.
 */
SZ_HELPER_AUTO sz_cptr_t sz_utf8_decode_once_haswell_( //
    sz_cptr_t text, sz_size_t length,                  //
    sz_rune_t *runes, sz_size_t runes_capacity,        //
    sz_size_t *runes_unpacked) {

    sz_size_t const chunk = length < 32 ? length : 32;
    sz_u32_t const load_mask = sz_u32_mask_until_serial_(chunk);

    __m256i window_u8x32; // The 32-byte window; lanes [chunk, 32) read as zero (the AVX2 masked-tail stand-in).
    if (chunk >= 32) { window_u8x32 = _mm256_loadu_si256((__m256i const *)text); }
    else {
        sz_u256_vec_t staging;
        staging.ymm = _mm256_setzero_si256();
        for (sz_size_t i = 0; i < chunk; ++i) staging.u8s[i] = (sz_u8_t)text[i];
        window_u8x32 = staging.ymm;
    }

    // ASCII fast lane: a whole window of 1-byte runes widens directly with `vpmovzxbd`, no classification needed.
    if (((sz_u32_t)_mm256_movemask_epi8(window_u8x32) & load_mask) == 0) {
        sz_size_t const runes_to_unpack = chunk < runes_capacity ? chunk : runes_capacity;
        __m128i const window_low_half_u8x16 = _mm256_castsi256_si128(window_u8x32);
        __m128i const window_high_half_u8x16 = _mm256_extracti128_si256(window_u8x32, 1);
        sz_rune_t ascii_runes[32];
        _mm256_storeu_si256((__m256i *)(ascii_runes + 0), _mm256_cvtepu8_epi32(window_low_half_u8x16));
        _mm256_storeu_si256((__m256i *)(ascii_runes + 8),
                            _mm256_cvtepu8_epi32(_mm_srli_si128(window_low_half_u8x16, 8)));
        _mm256_storeu_si256((__m256i *)(ascii_runes + 16), _mm256_cvtepu8_epi32(window_high_half_u8x16));
        _mm256_storeu_si256((__m256i *)(ascii_runes + 24),
                            _mm256_cvtepu8_epi32(_mm_srli_si128(window_high_half_u8x16, 8)));
        for (sz_size_t i = 0; i < runes_to_unpack; ++i) runes[i] = ascii_runes[i];
        *runes_unpacked = runes_to_unpack;
        return text + runes_to_unpack;
    }

    // Single-source classification: a high-nibble LUT gives the per-lane byte length, and both the validity gate and
    // the width-blend derive from it - so a lead and its length can never disagree (the class-of-malformed bug).
    __m256i const length_lut_u8x32 = _mm256_broadcastsi128_si256(
        _mm_setr_epi8(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 3, 4));
    sz_u32_t const starts_bits =
        (sz_u32_t)_mm256_movemask_epi8(_mm256_cmpgt_epi8(window_u8x32, _mm256_set1_epi8((char)-65))) & load_mask;
    sz_u32_t const continuation_bits = load_mask & ~starts_bits;
    __m256i const byte_high_nibble_u8x32 = _mm256_and_si256(_mm256_srli_epi16(window_u8x32, 4), _mm256_set1_epi8(0x0F));
    __m256i const sequence_lengths_u8x32 = _mm256_shuffle_epi8(length_lut_u8x32, byte_high_nibble_u8x32);
    sz_u32_t const len_ge_two =
        (sz_u32_t)_mm256_movemask_epi8(_mm256_cmpgt_epi8(sequence_lengths_u8x32, _mm256_set1_epi8(1))) & starts_bits;
    sz_u32_t const len_ge_three =
        (sz_u32_t)_mm256_movemask_epi8(_mm256_cmpgt_epi8(sequence_lengths_u8x32, _mm256_set1_epi8(2))) & starts_bits;
    sz_u32_t const len_ge_four =
        (sz_u32_t)_mm256_movemask_epi8(_mm256_cmpgt_epi8(sequence_lengths_u8x32, _mm256_set1_epi8(3))) & starts_bits;

    // Any start whose declared sequence reaches past the window is deferred: the FIRST overrunning start bounds the
    // decodable prefix (a malformed lead-in-lead can overrun before the trailing truncation), and its bytes resume.
    __m256i const lane_iota_u8x32 = _mm256_setr_epi8(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, //
                                                     16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31);
    __m256i const sequence_end_u8x32 = _mm256_add_epi8(lane_iota_u8x32, sequence_lengths_u8x32);
    // Unsigned "> chunk": chunk <= 32, lengths <= 4, sequence_end <= 35 fits a byte; compare via the signed-bias trick.
    __m256i const signed_bias_u8x32 = _mm256_set1_epi8((char)0x80);
    __m256i const overrun_detected_u8x32 = _mm256_cmpgt_epi8(
        _mm256_add_epi8(sequence_end_u8x32, signed_bias_u8x32),
        _mm256_add_epi8(_mm256_set1_epi8((char)chunk), signed_bias_u8x32));
    sz_u32_t const overruns_bits = (sz_u32_t)_mm256_movemask_epi8(overrun_detected_u8x32) & starts_bits;
    sz_size_t const decodable_end = overruns_bits ? (sz_size_t)sz_u64_ctz(overruns_bits) : chunk;
    sz_u32_t const decodable_mask = sz_u32_mask_until_serial_(decodable_end);

    // Bad lead: 0xC0/0xC1 (overlong 2-byte by the LUT) or 0xF5..0xFF (out of range, length-4 by the LUT).
    __m256i const is_length_two_u8x32 = _mm256_cmpeq_epi8(sequence_lengths_u8x32, _mm256_set1_epi8(2));
    __m256i const is_length_four_u8x32 = _mm256_cmpeq_epi8(sequence_lengths_u8x32, _mm256_set1_epi8(4));
    __m256i const is_less_than_0xc2_u8x32 = _mm256_cmpgt_epi8(
        _mm256_add_epi8(_mm256_set1_epi8((char)0xC2), signed_bias_u8x32),
        _mm256_add_epi8(window_u8x32, signed_bias_u8x32));
    __m256i const is_greater_than_0xf4_u8x32 = _mm256_cmpgt_epi8(
        _mm256_add_epi8(window_u8x32, signed_bias_u8x32),
        _mm256_add_epi8(_mm256_set1_epi8((char)0xF4), signed_bias_u8x32));
    sz_u32_t const bad_lead_bits = (sz_u32_t)_mm256_movemask_epi8(_mm256_or_si256(
                                       _mm256_and_si256(is_length_two_u8x32, is_less_than_0xc2_u8x32),
                                       _mm256_and_si256(is_length_four_u8x32, is_greater_than_0xf4_u8x32))) &
                                   starts_bits;

    // First-continuation range violations for E0/ED/F0/F4 (overlong 3/4-byte, surrogate, > U+10FFFF). Such leads are
    // rare in real CJK/emoji text (which lives in E1..EC / F1..F3), so a single cmpeq-any gate on their presence skips
    // the neighbour build and range compares for the common window.
    sz_u32_t overlong_or_surrogate_or_range_bits = 0;
    __m256i const is_lead_0xe0_u8x32 = _mm256_cmpeq_epi8(window_u8x32, _mm256_set1_epi8((char)0xE0));
    __m256i const is_lead_0xed_u8x32 = _mm256_cmpeq_epi8(window_u8x32, _mm256_set1_epi8((char)0xED));
    __m256i const is_lead_0xf0_u8x32 = _mm256_cmpeq_epi8(window_u8x32, _mm256_set1_epi8((char)0xF0));
    __m256i const is_lead_0xf4_u8x32 = _mm256_cmpeq_epi8(window_u8x32, _mm256_set1_epi8((char)0xF4));
    __m256i const is_special_lead_u8x32 = _mm256_or_si256(_mm256_or_si256(is_lead_0xe0_u8x32, is_lead_0xed_u8x32),
                                                          _mm256_or_si256(is_lead_0xf0_u8x32, is_lead_0xf4_u8x32));
    if (_mm256_movemask_epi8(is_special_lead_u8x32)) {
        __m256i const window_successor_u8x32 = _mm256_permute2x128_si256(window_u8x32, window_u8x32, 0x01);
        __m256i const next_byte_u8x32 = _mm256_alignr_epi8(window_successor_u8x32, window_u8x32, 1);
        __m256i const next_byte_biased_u8x32 = _mm256_add_epi8(next_byte_u8x32, signed_bias_u8x32);
        __m256i const is_next_byte_lt_0xa0_u8x32 = _mm256_cmpgt_epi8(
            _mm256_add_epi8(_mm256_set1_epi8((char)0xA0), signed_bias_u8x32), next_byte_biased_u8x32);
        __m256i const is_next_byte_ge_0xa0_u8x32 = _mm256_cmpgt_epi8(
            next_byte_biased_u8x32, _mm256_add_epi8(_mm256_set1_epi8((char)0x9F), signed_bias_u8x32));
        __m256i const is_next_byte_lt_0x90_u8x32 = _mm256_cmpgt_epi8(
            _mm256_add_epi8(_mm256_set1_epi8((char)0x90), signed_bias_u8x32), next_byte_biased_u8x32);
        __m256i const is_next_byte_ge_0x90_u8x32 = _mm256_cmpgt_epi8(
            next_byte_biased_u8x32, _mm256_add_epi8(_mm256_set1_epi8((char)0x8F), signed_bias_u8x32));
        __m256i const range_violation_u8x32 = _mm256_or_si256(
            _mm256_or_si256(_mm256_and_si256(is_lead_0xe0_u8x32, is_next_byte_lt_0xa0_u8x32),
                            _mm256_and_si256(is_lead_0xed_u8x32, is_next_byte_ge_0xa0_u8x32)),
            _mm256_or_si256(_mm256_and_si256(is_lead_0xf0_u8x32, is_next_byte_lt_0x90_u8x32),
                            _mm256_and_si256(is_lead_0xf4_u8x32, is_next_byte_ge_0x90_u8x32)));
        overlong_or_surrogate_or_range_bits = (sz_u32_t)_mm256_movemask_epi8(range_violation_u8x32) & starts_bits;
    }

    // Per-lane continuation availability at the declared trailing slots, evaluated at the lead lane.
    sz_u32_t const cont1 = starts_bits & (continuation_bits >> 1);
    sz_u32_t const cont2 = starts_bits & (continuation_bits >> 2);
    sz_u32_t const cont3 = starts_bits & (continuation_bits >> 3);
    sz_u32_t const b1_range_ok = starts_bits & ~overlong_or_surrogate_or_range_bits;
    sz_u32_t const first_ok = cont1 & b1_range_ok & ~bad_lead_bits;

    sz_u32_t const length_one_bits = starts_bits & ~len_ge_two;
    sz_u32_t const length_two_bits = len_ge_two & ~len_ge_three;
    sz_u32_t const length_three_bits = len_ge_three & ~len_ge_four;
    sz_u32_t const length_four_bits = len_ge_four;

    // Well-formed leads (vectorized `sz_rune_decode` success), restricted to the decodable span.
    sz_u32_t const wf1 = length_one_bits;
    sz_u32_t const wf2 = length_two_bits & ~bad_lead_bits & cont1;
    sz_u32_t const wf3 = length_three_bits & b1_range_ok & cont1 & cont2;
    sz_u32_t const wf4 = length_four_bits & ~bad_lead_bits & b1_range_ok & cont1 & cont2 & cont3;
    sz_u32_t const well_formed = (wf1 | wf2 | wf3 | wf4) & decodable_mask;

    // Per-lane maximal-subpart length (mirror of `sz_utf8_maximal_subpart_`): start at 1 and extend across each
    // continuation slot a well-formed sequence would still accept.
    sz_u32_t const step2 = len_ge_two & first_ok;
    sz_u32_t const step3 = step2 & len_ge_three & cont2;
    sz_u32_t const step4 = step3 & len_ge_four & cont3;

    // Orphan promotion: a continuation byte covered by NO lead's maximal-subpart span becomes its own 1-byte U+FFFD.
    sz_u32_t const covered = ((step2 & decodable_mask) << 1) | ((step3 & decodable_mask) << 2) |
                             ((step4 & decodable_mask) << 3);
    sz_u32_t const orphan = continuation_bits & decodable_mask & ~covered;
    sz_u32_t const emit_starts = (starts_bits | orphan) & decodable_mask;
    sz_size_t const emit_count = (sz_size_t)sz_u64_popcount(emit_starts);
    if (emit_count == 0) { return *runes_unpacked = 0, text; } // Nothing decodable → window-edge finalize in driver.
    sz_u32_t const ill_formed = emit_starts & ~well_formed;

    // Collapse ill-formed lanes to U+FFFD only when the window holds any; well-formed windows skip the mask build.
    int const has_ill = ill_formed != 0;
    __m256i const ill_formed_lanes_u8x32 = has_ill ? sz_utf8_byte_mask_from_bits_haswell_(ill_formed)
                                                   : _mm256_setzero_si256();

    // Gate the wide-byte gathers in the drain: a window with no 3-byte lead skips the 3rd-byte gather, and a window
    // with no 4-byte lead skips the 4th. `has_four ⟹ has_three` (a 4-byte lead is also `len_ge_three`).
    int const has_three = (len_ge_three | len_ge_four) != 0;
    int const has_four = len_ge_four != 0;

    sz_u8_t last_off = 0;
    sz_size_t const produced = sz_utf8_rune_drain_haswell_(window_u8x32, ill_formed_lanes_u8x32, emit_starts, has_three,
                                                           has_four, has_ill, emit_count, runes, runes_capacity,
                                                           &last_off);

    // Resume cursor delta = last emitted start's offset + its maximal-subpart length (1 + the step slots it reached),
    // read scalar-ly from the bitmasks at that one lane - no per-lane length vector materialize/store/reload.
    sz_size_t const last_length = (sz_size_t)1 + ((step2 >> last_off) & 1u) + ((step3 >> last_off) & 1u) +
                                  ((step4 >> last_off) & 1u);
    *runes_unpacked = produced;
    return text + (sz_size_t)last_off + last_length;
}

/**
 *  @brief  Decode UTF-8 into dense UTF-32 @p runes, TOTAL in-vector over 32-byte AVX2 windows (NO decline-to-serial).
 *          Each window classifies all lanes, promotes orphan continuation bytes, compacts the emitted starts, gathers
 *          their bytes in-register, and width-blends; ill-formed leads/orphans collapse to one U+FFFD per maximal
 *          ill-formed subpart (Unicode 17.0 §3.9 / W3C), bit-exact with @ref sz_utf8_decode_serial. A pure-ASCII
 *          window widens with `vpmovzxbd`. Gather-free on the hot path: `vbroadcasti128` + `vpshufb` window reads and
 *          BMI2 `pext`/`tzcnt`/`blsr` compaction, never a `vpgatherdd`.
 */
SZ_API_COMPTIME sz_cptr_t sz_utf8_decode_haswell( //
    sz_cptr_t text, sz_size_t length,             //
    sz_rune_t *runes, sz_size_t runes_capacity,   //
    sz_size_t *runes_unpacked) {

    sz_cptr_t cursor = text;
    sz_cptr_t const end = text + length;
    sz_size_t runes_written = 0;
    while (runes_written < runes_capacity && cursor < end) {
        sz_size_t step_unpacked = 0;
        sz_cptr_t next = sz_utf8_decode_once_haswell_(cursor, (sz_size_t)(end - cursor), runes + runes_written,
                                                      runes_capacity - runes_written, &step_unpacked);
        if (step_unpacked) {
            runes_written += step_unpacked;
            cursor = next;
            continue;
        }
        // `step_unpacked == 0` only when the very first lead declares a sequence crossing the window edge. A resumable
        // truncation breaks and awaits more bytes; a bad/overlong truncated lead at the edge finalizes to one U+FFFD
        // over its maximal ill-formed subpart - a bounded <=3-byte finalize, never a serial window re-decode.
        if (sz_utf8_incomplete_tail_(cursor, end)) break;
        runes[runes_written++] = (sz_rune_t)sz_rune_replacement_k;
        cursor += sz_utf8_maximal_subpart_(cursor, end);
    }
    *runes_unpacked = runes_written;
    return cursor;
}

#pragma endregion Total vectorized decode

/*  UAX-29 word boundary detection (vectorized). The same outer walk as the serial reference, but all-ASCII
 *  windows resolve their boundaries with `_mm256_shuffle_epi8` Word_Break classification plus a local pair
 *  decision. Any stateful rule (WB4/6/7/7a/11/12/15/16/3c), non-ASCII, or window-edge position defers to
 *  `sz_utf8_is_word_boundary_serial`, keeping the output byte-exact versus serial. */

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#endif // SZ_USE_HASWELL

#ifdef __cplusplus
}
#endif

#endif // STRINGZILLA_UTF8_RUNES_HASWELL_H_
