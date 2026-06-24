/**
 *  @brief Haswell (AVX2) backend for UTF-8 codepoint mechanics.
 *  @file include/stringzilla/utf8_codepoints/haswell.h
 *  @author Ash Vardanian
 */
#ifndef STRINGZILLA_UTF8_CODEPOINTS_HASWELL_H_
#define STRINGZILLA_UTF8_CODEPOINTS_HASWELL_H_

#include <string.h> // `memcpy`, `memset` (AVX2 partial-window staging; no byte-masked load)

#include "stringzilla/types.h"
#include "stringzilla/utf8_codepoints/serial.h"

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
SZ_INTERNAL __m256i sz_mm256_store_mask_epi64_(sz_size_t count) {
    return _mm256_cmpgt_epi64(_mm256_set1_epi64x((long long)count), _mm256_setr_epi64x(0, 1, 2, 3));
}

SZ_PUBLIC sz_size_t sz_utf8_count_haswell(sz_cptr_t text, sz_size_t length) {
    sz_u256_vec_t continuation_mask_vec, continuation_pattern_vec;
    continuation_mask_vec.ymm = _mm256_set1_epi8((char)0xC0);
    continuation_pattern_vec.ymm = _mm256_set1_epi8((char)0x80);

    sz_u8_t const *text_u8 = (sz_u8_t const *)text;
    sz_size_t char_count = 0;

    // Process 32 bytes at a time
    sz_u256_vec_t text_vec, headers_vec;
    while (length >= 32) {
        text_vec.ymm = _mm256_loadu_si256((__m256i const *)text_u8);

        // Apply mask (byte & 0xC0) to extract top 2 bits of each byte
        headers_vec.ymm = _mm256_and_si256(text_vec.ymm, continuation_mask_vec.ymm);

        // Compare with 0x80 (0b10000000) to find continuation bytes
        sz_u32_t start_byte_mask = ~(sz_u32_t)_mm256_movemask_epi8(
            _mm256_cmpeq_epi8(headers_vec.ymm, continuation_pattern_vec.ymm));

        char_count += _mm_popcnt_u32(start_byte_mask);
        text_u8 += 32;
        length -= 32;
    }

    // Process remaining bytes with serial
    char_count += sz_utf8_count_serial((sz_cptr_t)text_u8, length);
    return char_count;
}

SZ_PUBLIC sz_cptr_t sz_utf8_find_nth_haswell(sz_cptr_t text, sz_size_t length, sz_size_t n) {
    // The logic of this function is similar to `sz_utf8_count_haswell`, but uses PDEP
    // instruction in the inner loop to locate Nth character start byte efficiently
    // without one more loop.
    sz_u256_vec_t continuation_mask_vec, continuation_pattern_vec;
    continuation_mask_vec.ymm = _mm256_set1_epi8((char)0xC0);
    continuation_pattern_vec.ymm = _mm256_set1_epi8((char)0x80);

    sz_u8_t const *text_u8 = (sz_u8_t const *)text;

    // Process 32 bytes at a time
    sz_u256_vec_t text_vec, headers_vec;
    while (length >= 32) {
        text_vec.ymm = _mm256_loadu_si256((__m256i const *)text_u8);

        // Apply mask (byte & 0xC0) to extract top 2 bits of each byte
        headers_vec.ymm = _mm256_and_si256(text_vec.ymm, continuation_mask_vec.ymm);

        // Compare with 0x80 (0b10000000) to find continuation bytes
        sz_u32_t start_byte_mask = ~(sz_u32_t)_mm256_movemask_epi8(
            _mm256_cmpeq_epi8(headers_vec.ymm, continuation_pattern_vec.ymm));
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
    return sz_utf8_find_nth_serial((sz_cptr_t)text_u8, length, n);
}

#pragma region Shared SIMD leaf substrate

/** @brief  The decoded 64-byte window for the AVX2 backend. The 64 bytes live as two `__m256i` halves (`*_lo` =
 *          lanes [0, 32), `*_hi` = lanes [32, 64)); the per-lane byte-domain codepoint halves `high`/`low` share that
 *          shape. Masks are `sz_u64_t` (`vpmovmskb` per half, OR-combined) rather than the Ice Lake `__mmask64`. Field
 *          names and semantics match @ref sz_utf8_codepoints_window_t so the portable rule algebra is unchanged. */
typedef struct sz_utf8_codepoints_window_haswell_t {
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
} sz_utf8_codepoints_window_haswell_t;

/** @brief  Per-byte logical right shift by @p shift keeping the low @p keep bits — the AVX2 twin of `srl8_`. */
SZ_INTERNAL __m256i sz_utf8_codepoints_srl8_haswell_(__m256i value, int shift, sz_u8_t keep) {
    return _mm256_and_si256(_mm256_srli_epi16(value, shift), _mm256_set1_epi8((char)keep));
}

/** @brief  Combine two per-half `vpmovmskb` results into one 64-bit lane mask. */
SZ_INTERNAL sz_u64_t sz_utf8_codepoints_mask_combine_haswell_(__m256i low_half, __m256i high_half) {
    sz_u64_t const low_bits = (sz_u32_t)_mm256_movemask_epi8(low_half);
    sz_u64_t const high_bits = (sz_u32_t)_mm256_movemask_epi8(high_half);
    return low_bits | (high_bits << 32);
}

/** @brief  Masked 64-byte load into two halves; bytes [loaded, 64) read as zero (the AVX2 stand-in for
 *          `_mm512_maskz_loadu_epi8`). A small stack staging buffer covers the partial tail so we never read past
 *          `text + loaded`. */
SZ_INTERNAL void sz_utf8_codepoints_load_window_haswell_( //
    sz_u8_t const *text, sz_size_t loaded, __m256i *out_low, __m256i *out_high) {
    if (loaded >= 64) {
        *out_low = _mm256_loadu_si256((__m256i const *)(text + 0));
        *out_high = _mm256_loadu_si256((__m256i const *)(text + 32));
        return;
    }
    sz_u8_t staging[64];
    memset(staging, 0, sizeof(staging));
    memcpy(staging, text, loaded);
    *out_low = _mm256_loadu_si256((__m256i const *)(staging + 0));
    *out_high = _mm256_loadu_si256((__m256i const *)(staging + 32));
}

/** @brief  Forward neighbours `next1[i] = window[i+1]`, `next2[i] = window[i+2]` over all 64 lanes, with the lanes
 *          past the window WRAPPING modulo 64 to match Ice Lake's `_mm512_permutexvar_epi8` (so `next1[63]==window[0]`,
 *          `next2[62]==window[0]`, `next2[63]==window[1]`). AVX2 has no 32-byte byte permute, so each 128-bit lane is
 *          fed its following bytes via `_mm256_permute2x128_si256` (to rotate in the successor 128-bit block, the
 *          window head wrapping in after `window_hi`) then `_mm256_alignr_epi8` to shift across the lane boundary. */
SZ_INTERNAL void sz_utf8_codepoints_forward_neighbours_haswell_( //
    __m256i window_lo, __m256i window_hi,                        //
    __m256i *next1_lo, __m256i *next1_hi, __m256i *next2_lo, __m256i *next2_hi) {
    // The 128-bit block following each 128-bit lane of window_lo: its high lane (bytes 16..31) is followed by
    // window_hi's low lane (bytes 32..47), assembled by permute2x128 selector 0x21.
    __m256i const low_successor = _mm256_permute2x128_si256(window_lo, window_hi, 0x21);
    *next1_lo = _mm256_alignr_epi8(low_successor, window_lo, 1);
    *next2_lo = _mm256_alignr_epi8(low_successor, window_lo, 2);
    // The block following window_hi's high lane (bytes 48..63) wraps to the window head (bytes 0..15 of window_lo),
    // so window_hi's successor is window_lo (byte 64 aliases byte 0).
    __m256i const high_successor = _mm256_permute2x128_si256(window_hi, window_lo, 0x21);
    *next1_hi = _mm256_alignr_epi8(high_successor, window_hi, 1);
    *next2_hi = _mm256_alignr_epi8(high_successor, window_hi, 2);
}

/** @brief  Expand a 32-bit lane mask into a 32-byte select vector (byte `i` = 0xFF when bit `i` is set) to drive
 *          `_mm256_blendv_epi8` in place of Ice Lake's `_mm512_mask_blend_epi8`. Gather-free: broadcast the mask,
 *          route the right mask byte to each output byte via `vpshufb`, isolate the per-lane bit, then `cmpeq`. */
SZ_INTERNAL __m256i sz_utf8_codepoints_byte_mask_from_bits_haswell_(sz_u32_t bits) {
    __m256i const broadcast = _mm256_set1_epi32((int)bits);
    __m256i const byte_router = _mm256_setr_epi8(       //
        0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, //
        2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3);
    __m256i const spread = _mm256_shuffle_epi8(broadcast, byte_router);
    __m256i const bit_select = _mm256_setr_epi8(                              //
        1, 2, 4, 8, 16, 32, 64, (char)128, 1, 2, 4, 8, 16, 32, 64, (char)128, //
        1, 2, 4, 8, 16, 32, 64, (char)128, 1, 2, 4, 8, 16, 32, 64, (char)128);
    __m256i const isolated = _mm256_and_si256(spread, bit_select);
    return _mm256_cmpeq_epi8(isolated, bit_select);
}

/** @brief  Load up to 64 bytes (masked tail) and decode every lane into byte-domain halves — the AVX2 twin of
 *          @ref sz_utf8_codepoints_decode_window_, bit-identical to it on every lane. */
SZ_INTERNAL sz_utf8_codepoints_window_haswell_t sz_utf8_codepoints_decode_window_haswell_( //
    sz_u8_t const *text, sz_size_t available) {
    sz_utf8_codepoints_window_haswell_t result;
    result.loaded = available < 64 ? available : 64;

    __m256i window_lo, window_hi;
    sz_utf8_codepoints_load_window_haswell_(text, result.loaded, &window_lo, &window_hi);
    result.window_lo = window_lo, result.window_hi = window_hi;

    __m256i next1_lo, next1_hi, next2_lo, next2_hi;
    sz_utf8_codepoints_forward_neighbours_haswell_(window_lo, window_hi, &next1_lo, &next1_hi, &next2_lo, &next2_hi);

    sz_u64_t const loaded_mask = result.loaded >= 64 ? ~(sz_u64_t)0 : (((sz_u64_t)1 << result.loaded) - 1);

    // Lead-class detection over loaded lanes: (byte & mask) == pattern, AND-clamped to loaded lanes.
    __m256i const lead_mask_continuation = _mm256_set1_epi8((char)0xC0);
    __m256i const continuation_lo = _mm256_cmpeq_epi8(_mm256_and_si256(window_lo, lead_mask_continuation),
                                                      _mm256_set1_epi8((char)0x80));
    __m256i const continuation_hi = _mm256_cmpeq_epi8(_mm256_and_si256(window_hi, lead_mask_continuation),
                                                      _mm256_set1_epi8((char)0x80));
    result.continuation = sz_utf8_codepoints_mask_combine_haswell_(continuation_lo, continuation_hi) & loaded_mask;
    result.codepoint_starts = loaded_mask & ~result.continuation;

    __m256i const lead_mask_two = _mm256_set1_epi8((char)0xE0);
    result.two_byte_starts =
        sz_utf8_codepoints_mask_combine_haswell_(
            _mm256_cmpeq_epi8(_mm256_and_si256(window_lo, lead_mask_two), _mm256_set1_epi8((char)0xC0)),
            _mm256_cmpeq_epi8(_mm256_and_si256(window_hi, lead_mask_two), _mm256_set1_epi8((char)0xC0))) &
        loaded_mask;
    __m256i const lead_mask_three = _mm256_set1_epi8((char)0xF0);
    result.three_byte_starts =
        sz_utf8_codepoints_mask_combine_haswell_(
            _mm256_cmpeq_epi8(_mm256_and_si256(window_lo, lead_mask_three), _mm256_set1_epi8((char)0xE0)),
            _mm256_cmpeq_epi8(_mm256_and_si256(window_hi, lead_mask_three), _mm256_set1_epi8((char)0xE0))) &
        loaded_mask;
    __m256i const lead_mask_four = _mm256_set1_epi8((char)0xF8);
    result.four_byte_starts =
        sz_utf8_codepoints_mask_combine_haswell_(
            _mm256_cmpeq_epi8(_mm256_and_si256(window_lo, lead_mask_four), _mm256_set1_epi8((char)0xF0)),
            _mm256_cmpeq_epi8(_mm256_and_si256(window_hi, lead_mask_four), _mm256_set1_epi8((char)0xF0))) &
        loaded_mask;

    __m256i const low_five_bits = _mm256_set1_epi8(0x1F);
    __m256i const low_six_bits = _mm256_set1_epi8(0x3F);
    __m256i const low_two_bits = _mm256_set1_epi8(0x03);
    __m256i const low_four_bits = _mm256_set1_epi8(0x0F);

    // 2-byte: codepoint = ((b0 & 0x1F) << 6) | (b1 & 0x3F); high = codepoint >> 8, low = codepoint & 0xFF.
    __m256i const high_two_lo = sz_utf8_codepoints_srl8_haswell_(_mm256_and_si256(window_lo, low_five_bits), 2, 0x07);
    __m256i const high_two_hi = sz_utf8_codepoints_srl8_haswell_(_mm256_and_si256(window_hi, low_five_bits), 2, 0x07);
    __m256i const low_two_lo = _mm256_or_si256(_mm256_slli_epi16(_mm256_and_si256(window_lo, low_two_bits), 6),
                                               _mm256_and_si256(next1_lo, low_six_bits));
    __m256i const low_two_hi = _mm256_or_si256(_mm256_slli_epi16(_mm256_and_si256(window_hi, low_two_bits), 6),
                                               _mm256_and_si256(next1_hi, low_six_bits));

    // 3-byte: codepoint = ((b0 & 0x0F) << 12) | ((b1 & 0x3F) << 6) | (b2 & 0x3F).
    __m256i const high_three_lo = _mm256_or_si256(_mm256_slli_epi16(_mm256_and_si256(window_lo, low_four_bits), 4),
                                                  sz_utf8_codepoints_srl8_haswell_(next1_lo, 2, 0x0F));
    __m256i const high_three_hi = _mm256_or_si256(_mm256_slli_epi16(_mm256_and_si256(window_hi, low_four_bits), 4),
                                                  sz_utf8_codepoints_srl8_haswell_(next1_hi, 2, 0x0F));
    __m256i const low_three_lo = _mm256_or_si256(_mm256_slli_epi16(_mm256_and_si256(next1_lo, low_two_bits), 6),
                                                 _mm256_and_si256(next2_lo, low_six_bits));
    __m256i const low_three_hi = _mm256_or_si256(_mm256_slli_epi16(_mm256_and_si256(next1_hi, low_two_bits), 6),
                                                 _mm256_and_si256(next2_hi, low_six_bits));

    // Blend 2-byte vs 3-byte per lane (AVX2 `blendv` selects on the byte MSB, so expand the 3-byte lane mask).
    __m256i const three_select_lo = sz_utf8_codepoints_byte_mask_from_bits_haswell_(
        (sz_u32_t)(result.three_byte_starts & 0xFFFFFFFFu));
    __m256i const three_select_hi = sz_utf8_codepoints_byte_mask_from_bits_haswell_(
        (sz_u32_t)((result.three_byte_starts >> 32) & 0xFFFFFFFFu));
    result.high_lo = _mm256_blendv_epi8(high_two_lo, high_three_lo, three_select_lo);
    result.high_hi = _mm256_blendv_epi8(high_two_hi, high_three_hi, three_select_hi);
    result.low_lo = _mm256_blendv_epi8(low_two_lo, low_three_lo, three_select_lo);
    result.low_hi = _mm256_blendv_epi8(low_two_hi, low_three_hi, three_select_hi);
    return result;
}

/** @brief  One nibble-cascade stage with a sub-256 selector: `result[lane] = table[selector[lane]*16 + within[lane]]`.
 *          Each of @p tile_count 16-byte rows is one selector value; @p within is the addressing nibble (caller-masked
 *          to `[0,16)`). The AVX2 stand-in for VBMI `vpermi2b`: there is no 32-byte byte permute, so each resident row
 *          is broadcast and shuffled by @p within, then blended in for the lanes whose @p selector picks that row.
 *          Gather-free — only `vbroadcasti128`/`vpshufb`/`vpcmpeqb`/`vpblendvb`. */
SZ_INTERNAL __m256i sz_utf8_codepoints_cascade_stage_haswell_( //
    sz_u8_t const *table, int tile_count, __m256i selector, __m256i within) {
    __m256i result = _mm256_setzero_si256();
    for (int tile = 0; tile < tile_count; ++tile) {
        __m256i const row = _mm256_broadcastsi128_si256(_mm_loadu_si128((__m128i const *)(table + tile * 16)));
        __m256i const picked = _mm256_shuffle_epi8(row, within);
        __m256i const here = _mm256_cmpeq_epi8(selector, _mm256_set1_epi8((char)tile));
        result = _mm256_blendv_epi8(result, picked, here);
    }
    return result;
}

/** @brief  256-entry byte LUT addressed by a per-lane byte index in `[0,256)`: `result[lane] = group_base[index[lane]]`.
 *          Two-stage `vpshufb` over 16 resident rows — the low nibble shuffles within a row, the high nibble selects
 *          the row. The AVX2 twin of the substrate `lut256` leaf. */
SZ_INTERNAL __m256i sz_utf8_codepoints_lut256_haswell_(sz_u8_t const *group_base, __m256i index) {
    __m256i const within = _mm256_and_si256(index, _mm256_set1_epi8(0x0F));
    __m256i const selector = _mm256_and_si256(_mm256_srli_epi16(index, 4), _mm256_set1_epi8(0x0F));
    return sz_utf8_codepoints_cascade_stage_haswell_(group_base, 16, selector, within);
}

#pragma endregion Shared SIMD leaf substrate

#pragma region Drains

/** @brief  Left-pack the set lane indices (in [0, 64), ascending) of a 64-bit @p mask into @p out[0..popcount).
 *          BMI2 path: `tzcnt` pulls the lowest set bit's index, `blsr` clears it. On Intel Haswell `tzcnt`/`blsr`
 *          are single-uop and beat a `vpshufb` left-pack LUT (measured ~0.91x vs ~0.76x of Ice Lake `vpcompressb`);
 *          the LUT only wins where `pext`/`blsr` is microcoded (AMD pre-Zen3), refining CONTRIBUTING-KERNELS §6.6. */
SZ_INTERNAL void sz_utf8_codepoints_unpack_indices_haswell_(sz_u64_t mask, sz_u8_t *out) {
    while (mask) {
        *out++ = (sz_u8_t)sz_u64_ctz(mask);
        mask = _blsr_u64(mask); // clear the lowest set bit
    }
}

/** @brief  AVX2 forward drain — the `vpcompressb`-free twin of @ref sz_utf8_codepoints_drain_forward_. Emits one
 *          (start, length) per set boundary lane (ascending), honoring @p capacity and the carried previous-boundary
 *          via @p previous_io; bit-exact with the Ice Lake leaf. The set lanes are index-unpacked once (BMI2), then
 *          streamed in waves of four u64 positions (`vpmovzxbq` widen + `base`, segment starts via `vpermq` shift +
 *          `vpblendd` carry-seat, lengths via `vpsubq`), with a scalar tail for the final partial wave (no AVX2
 *          masked store). */
SZ_INTERNAL sz_size_t sz_utf8_codepoints_drain_forward_haswell_( //
    sz_u64_t boundary, sz_size_t base, sz_size_t *starts, sz_size_t *lengths, sz_size_t produced, sz_size_t capacity,
    sz_size_t *previous_io) {
    sz_size_t const boundary_count = (sz_size_t)sz_u64_popcount(boundary);
    sz_u64_t previous = (sz_u64_t)*previous_io;
    if (boundary_count == 0 || produced >= capacity) {
        *previous_io = (sz_size_t)previous;
        return produced;
    }

    sz_u8_t indices[64];
    sz_utf8_codepoints_unpack_indices_haswell_(boundary, indices);

    __m256i const base_quad = _mm256_set1_epi64x((long long)base);
    sz_size_t emitted = 0;
    while (emitted < boundary_count && produced < capacity) {
        sz_size_t const wave = sz_min_of_three(boundary_count - emitted, capacity - produced, 4);
        // Widen four lane indices to u64 positions: position[k] = base + indices[emitted + k].
        __m128i const index_bytes = _mm_loadl_epi64((__m128i const *)(indices + emitted));
        __m256i const positions = _mm256_add_epi64(_mm256_cvtepu8_epi64(index_bytes), base_quad);
        // segment_starts = [previous, p0, p1, p2]: shift positions up one u64, seat the carry in lane 0.
        __m256i const carry = _mm256_set1_epi64x((long long)previous);
        __m256i const shifted = _mm256_permute4x64_epi64(positions, _MM_SHUFFLE(2, 1, 0, 0));
        __m256i const segment_starts = _mm256_blend_epi32(shifted, carry, 0x03);
        __m256i const segment_lengths = _mm256_sub_epi64(positions, segment_starts);
        if (produced + 4 <= capacity && wave == 4) {
            _mm256_storeu_si256((__m256i *)(starts + produced), segment_starts);
            _mm256_storeu_si256((__m256i *)(lengths + produced), segment_lengths);
        }
        else {
            sz_size_t tail_starts[4], tail_lengths[4];
            _mm256_storeu_si256((__m256i *)tail_starts, segment_starts);
            _mm256_storeu_si256((__m256i *)tail_lengths, segment_lengths);
            for (sz_size_t lane = 0; lane < wave; ++lane)
                starts[produced + lane] = tail_starts[lane], lengths[produced + lane] = tail_lengths[lane];
        }
        produced += wave, emitted += wave;
        previous = (sz_u64_t)(starts[produced - 1] + lengths[produced - 1]);
    }
    *previous_io = (sz_size_t)previous;
    return produced;
}

/** @brief  AVX2 backward drain — the `vpcompressb`-free mirror of @ref sz_utf8_codepoints_drain_backward_. Emits
 *          (start, length) per set boundary lane in DESCENDING byte order, carrying the open segment end via
 *          @p end_io; bit-exact with the Ice Lake leaf. */
SZ_INTERNAL sz_size_t sz_utf8_codepoints_drain_backward_haswell_( //
    sz_u64_t boundary, sz_size_t base, sz_size_t *starts, sz_size_t *lengths, sz_size_t produced, sz_size_t capacity,
    sz_size_t *end_io) {
    sz_size_t const boundary_count = (sz_size_t)sz_u64_popcount(boundary);
    sz_u64_t segment_end = (sz_u64_t)*end_io;
    if (boundary_count == 0 || produced >= capacity) {
        *end_io = (sz_size_t)segment_end;
        return produced;
    }

    // Unpack ascending indices, then read them high-to-low (equivalent to icelake's reversed compress, no bit-reverse).
    sz_u8_t indices[64];
    sz_utf8_codepoints_unpack_indices_haswell_(boundary, indices);

    __m256i const base_quad = _mm256_set1_epi64x((long long)base);
    sz_size_t emitted = 0;
    sz_size_t high = boundary_count; // exclusive upper bound into the ascending index array
    while (emitted < boundary_count && produced < capacity) {
        sz_size_t const wave = sz_min_of_three(boundary_count - emitted, capacity - produced, 4);
        sz_u8_t descending[4];
        for (sz_size_t lane = 0; lane < wave; ++lane) descending[lane] = indices[high - 1 - lane];
        __m128i const index_bytes = _mm_loadl_epi64((__m128i const *)descending);
        __m256i const positions = _mm256_add_epi64(_mm256_cvtepu8_epi64(index_bytes), base_quad);
        // ends = [segment_end, p0, p1, p2]; start = position, length = end - position.
        __m256i const carry = _mm256_set1_epi64x((long long)segment_end);
        __m256i const shifted = _mm256_permute4x64_epi64(positions, _MM_SHUFFLE(2, 1, 0, 0));
        __m256i const ends = _mm256_blend_epi32(shifted, carry, 0x03);
        __m256i const segment_lengths = _mm256_sub_epi64(ends, positions);
        if (produced + 4 <= capacity && wave == 4) {
            _mm256_storeu_si256((__m256i *)(starts + produced), positions);
            _mm256_storeu_si256((__m256i *)(lengths + produced), segment_lengths);
        }
        else {
            sz_size_t tail_starts[4], tail_lengths[4];
            _mm256_storeu_si256((__m256i *)tail_starts, positions);
            _mm256_storeu_si256((__m256i *)tail_lengths, segment_lengths);
            for (sz_size_t lane = 0; lane < wave; ++lane)
                starts[produced + lane] = tail_starts[lane], lengths[produced + lane] = tail_lengths[lane];
        }
        produced += wave, emitted += wave, high -= wave;
        segment_end = (sz_u64_t)starts[produced - 1];
    }
    *end_io = (sz_size_t)segment_end;
    return produced;
}

#pragma endregion Drains

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

#endif // STRINGZILLA_UTF8_CODEPOINTS_HASWELL_H_
