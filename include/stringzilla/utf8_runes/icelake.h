/**
 *  @brief Ice Lake (AVX-512 VBMI+VAES) backend for UTF-8 codepoint mechanics and shared SIMD substrate.
 *  @file include/stringzilla/utf8_runes/icelake.h
 *  @author Ash Vardanian
 */
#ifndef STRINGZILLA_UTF8_RUNES_ICELAKE_H_
#define STRINGZILLA_UTF8_RUNES_ICELAKE_H_

#include "stringzilla/types.h"
#include "stringzilla/utf8_runes/serial.h"

#ifdef __cplusplus
extern "C" {
#endif

#if SZ_USE_ICELAKE
#if defined(__clang__) && SZ_CLANG_HAS_EVEX512_
#pragma clang attribute push(                                                                                         \
    __attribute__((target("avx,avx512f,avx512vl,avx512bw,avx512dq,avx512vbmi,avx512vbmi2,bmi,bmi2,evex512,popcnt"))), \
    apply_to = function)
#elif defined(__clang__)
#pragma clang attribute push(                                                                                 \
    __attribute__((target("avx,avx512f,avx512vl,avx512bw,avx512dq,avx512vbmi,avx512vbmi2,bmi,bmi2,popcnt"))), \
    apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("avx", "avx512f", "avx512vl", "avx512bw", "avx512dq", "avx512vbmi", "avx512vbmi2", "bmi", "bmi2", \
                   "popcnt")
#endif

#pragma region Shared SIMD leaf substrate

/*  Family-agnostic AVX-512 leaf helpers shared by every UTF-8 segmentation kernel (word / grapheme /
 *  sentence / line / delimiters). Each family `#include`s this header and calls these substrate helpers
 *  primitives instead of carrying its own copy. The substrate bakes in zero property semantics: classifier
 *  tables flow in as bare pointers, boundary algebra flows in as bare masks, so the same code resolves any
 *  break property. Every routine treats all 64 lanes uniformly with no scalar per-lane loop, no gather, and
 *  no spill-to-stack-then-reload round-trip. */

/** @brief  Byte lane identity (lane `i` holds the value `i`: {0,1,...,63}) for `vpcompressb`-based drains and
 *          permute waves. The `_mm512_set_epi8` arguments read 63..0 because they fill highest lane first. */
SZ_HELPER_INLINE __m512i sz_utf8_lane_identity_icelake_(void) {
    return _mm512_set_epi8(                                             //
        63, 62, 61, 60, 59, 58, 57, 56, 55, 54, 53, 52, 51, 50, 49, 48, //
        47, 46, 45, 44, 43, 42, 41, 40, 39, 38, 37, 36, 35, 34, 33, 32, //
        31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, //
        15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
}

/** @brief  Mask of codepoint-start bytes in a loaded window: every lane that is not a continuation byte `0x80..0xBF`.
 *          Continuation bytes are signed `-128..-65`, so a single signed `vpcmpgtb` against `-65` selects starts -
 *          the one-op form shared by count, find-nth, and the unpack classifier (was written four different ways). */
SZ_HELPER_INLINE __mmask64 sz_utf8_rune_start_mask_icelake_(__m512i window, __mmask64 load_mask) {
    return _mm512_mask_cmpgt_epi8_mask(load_mask, window, _mm512_set1_epi8((char)-65));
}

/**
 *  @brief  Emit a window's @p emit delimiter matches in-register via `vpcompressb` and `ceil(emit/8)` masked
 *          widen-stores. `_mm512_alignr_epi64` shifts the compressed registers down between waves, so @p emit may
 *          exceed 8. Per-lane byte length is 1, plus 1 on a 2-byte start, plus 2 on a 3-byte start (disjoint masks).
 */
SZ_HELPER_AUTO void sz_utf8_rune_peel_icelake_(                                  //
    sz_u64_t start_bits, __mmask64 two_byte_starts, __mmask64 three_byte_starts, //
    sz_size_t emit, sz_size_t position, __m512i lane_identity,                   //
    sz_size_t *match_offsets, sz_size_t *match_lengths) {
    __mmask64 const compress_mask = _cvtu64_mask64(start_bits);

    __m512i length_per_lane_u8x64 = _mm512_set1_epi8(1);
    length_per_lane_u8x64 = _mm512_mask_add_epi8(length_per_lane_u8x64, two_byte_starts, length_per_lane_u8x64,
                                                 _mm512_set1_epi8(1));
    length_per_lane_u8x64 = _mm512_mask_add_epi8(length_per_lane_u8x64, three_byte_starts, length_per_lane_u8x64,
                                                 _mm512_set1_epi8(2));

    __m512i compressed_offsets_u8x64 = _mm512_maskz_compress_epi8(compress_mask, lane_identity);
    __m512i compressed_lengths_u8x64 = _mm512_maskz_compress_epi8(compress_mask, length_per_lane_u8x64);
    __m512i const position_broadcast_u64x8 = _mm512_set1_epi64((long long)position);

    __m512i const zero_u64x8 = _mm512_setzero_si512();
    for (sz_size_t emitted = 0; emitted < emit; emitted += 8) {
        __mmask8 const store_mask = sz_u8_clamp_mask_until_(emit - emitted);
        _mm512_mask_storeu_epi64(
            (void *)(match_offsets + emitted), store_mask,
            _mm512_add_epi64(_mm512_cvtepu8_epi64(_mm512_castsi512_si128(compressed_offsets_u8x64)),
                             position_broadcast_u64x8));
        _mm512_mask_storeu_epi64((void *)(match_lengths + emitted), store_mask,
                                 _mm512_cvtepu8_epi64(_mm512_castsi512_si128(compressed_lengths_u8x64)));
        compressed_offsets_u8x64 = _mm512_alignr_epi64(zero_u64x8, compressed_offsets_u8x64, 1);
        compressed_lengths_u8x64 = _mm512_alignr_epi64(zero_u64x8, compressed_lengths_u8x64, 1);
    }
}

#pragma region Decode window

/** @brief  Per-byte logical right shift by @p shift, retaining only the low @p keep bits of every lane. */
SZ_HELPER_INLINE __m512i sz_utf8_srl8_icelake_(__m512i value, int shift, sz_u8_t keep) {
    return _mm512_and_si512(_mm512_srli_epi16(value, shift), _mm512_set1_epi8((char)keep));
}

/**
 *  @brief  Per-lane codepoint reconstruction of a 64-byte window, family-agnostic and gather-free.
 *
 *  Loads up to 64 bytes (masked at the buffer tail) and, treating every lane as a potential codepoint start,
 *  reassembles each codepoint's value via the lead/2nd/3rd-byte arithmetic trick into byte-domain halves
 *  `high = codepoint >> 8` and `low = codepoint & 0xFF`. No codepoint is ever materialized to memory and no
 *  gather is issued; downstream classifiers read the halves directly and address tables from them. The 2-byte
 *  and 3-byte BMP cases are blended in-register; ASCII lanes keep `high == 0`, and 4-byte (SMP) leads leave
 *  `high`/`low` holding the low 16 bits, which the caller resolves through arithmetic ranges.
 */
typedef struct sz_utf8_rune_window_t {
    __m512i window;              /**< The raw 64 input bytes (continuation bytes included). */
    __m512i high;                /**< Per-lane `codepoint >> 8` for the codepoint that starts at this lane. */
    __m512i low;                 /**< Per-lane `codepoint & 0xFF` for the codepoint that starts at this lane. */
    __mmask64 continuation;      /**< Bit `i` set => lane `i` is a UTF-8 continuation byte `10xxxxxx`. */
    __mmask64 codepoint_starts;  /**< Bit `i` set => lane `i` begins a codepoint (loaded, non-continuation). */
    __mmask64 two_byte_starts;   /**< Bit `i` set => lane `i` is a 2-byte lead `110xxxxx`. */
    __mmask64 three_byte_starts; /**< Bit `i` set => lane `i` is a 3-byte lead `1110xxxx`. */
    __mmask64 four_byte_starts;  /**< Bit `i` set => lane `i` is a 4-byte lead `11110xxx`. */
    sz_size_t loaded;            /**< Number of bytes actually loaded (<= 64) into lanes [0, loaded). */
} sz_utf8_rune_window_t;

/** @brief  Load up to 64 bytes from @p text (masked tail) and decode every lane into byte-domain halves. */
SZ_HELPER_AUTO sz_utf8_rune_window_t sz_utf8_rune_decode_window_icelake_( //
    sz_u8_t const *text, sz_size_t available, __m512i lane_identity) {
    sz_utf8_rune_window_t result;
    result.loaded = available < 64 ? available : 64;
    __mmask64 const load_mask = sz_u64_clamp_mask_until_(result.loaded);
    __m512i const window_u8x64 = _mm512_maskz_loadu_epi8(load_mask, text);
    result.window = window_u8x64;

    // The three forward neighbours of each lane, gathered via in-register permutes (never `vpgather`).
    __m512i const next1_u8x64 = _mm512_permutexvar_epi8(_mm512_add_epi8(lane_identity, _mm512_set1_epi8(1)),
                                                        window_u8x64);
    __m512i const next2_u8x64 = _mm512_permutexvar_epi8(_mm512_add_epi8(lane_identity, _mm512_set1_epi8(2)),
                                                        window_u8x64);

    __mmask64 const loaded_lanes = load_mask;
    __mmask64 const is_continuation = _mm512_mask_cmpeq_epi8_mask(
        loaded_lanes, _mm512_and_si512(window_u8x64, _mm512_set1_epi8((char)0xC0)), _mm512_set1_epi8((char)0x80));
    result.continuation = is_continuation;
    result.codepoint_starts = loaded_lanes & ~is_continuation;
    result.two_byte_starts = _mm512_mask_cmpeq_epi8_mask(
        loaded_lanes, _mm512_and_si512(window_u8x64, _mm512_set1_epi8((char)0xE0)), _mm512_set1_epi8((char)0xC0));
    result.three_byte_starts = _mm512_mask_cmpeq_epi8_mask(
        loaded_lanes, _mm512_and_si512(window_u8x64, _mm512_set1_epi8((char)0xF0)), _mm512_set1_epi8((char)0xE0));
    result.four_byte_starts = _mm512_mask_cmpeq_epi8_mask(
        loaded_lanes, _mm512_and_si512(window_u8x64, _mm512_set1_epi8((char)0xF8)), _mm512_set1_epi8((char)0xF0));

    // 2-byte: cp = ((lead & 0x1F) << 6) | (next1 & 0x3F); high = cp >> 8, low = cp & 0xFF.
    __m512i const lead_five_bits_u8x64 = _mm512_and_si512(window_u8x64, _mm512_set1_epi8(0x1F));
    __m512i const next1_six_bits_u8x64 = _mm512_and_si512(next1_u8x64, _mm512_set1_epi8(0x3F));
    __m512i const high_two_byte_u8x64 = sz_utf8_srl8_icelake_(lead_five_bits_u8x64, 2, 0x07);
    __m512i const low_two_byte_u16x32 = _mm512_or_si512(
        _mm512_slli_epi16(_mm512_and_si512(window_u8x64, _mm512_set1_epi8(0x03)), 6), next1_six_bits_u8x64);
    // 3-byte: cp = ((lead & 0x0F) << 12) | ((next1 & 0x3F) << 6) | (next2 & 0x3F).
    __m512i const lead_four_bits_u8x64 = _mm512_and_si512(window_u8x64, _mm512_set1_epi8(0x0F));
    __m512i const high_three_byte_u16x32 = _mm512_or_si512(_mm512_slli_epi16(lead_four_bits_u8x64, 4),
                                                           sz_utf8_srl8_icelake_(next1_u8x64, 2, 0x0F));
    __m512i const low_three_byte_u16x32 = _mm512_or_si512(
        _mm512_slli_epi16(_mm512_and_si512(next1_u8x64, _mm512_set1_epi8(0x03)), 6),
        _mm512_and_si512(next2_u8x64, _mm512_set1_epi8(0x3F)));

    result.high = _mm512_mask_blend_epi8(result.three_byte_starts, high_two_byte_u8x64, high_three_byte_u16x32);
    result.low = _mm512_mask_blend_epi8(result.three_byte_starts, low_two_byte_u16x32, low_three_byte_u16x32);
    return result;
}

#pragma endregion Decode window

#pragma region In register two stage trie

/**
 *  @brief  Gather one byte per 16-bit lane from a register-resident byte table @p table of @p count entries,
 *          addressed by the 16-bit @p indices, using a `vpermi2b` page network (NO `vpgather`). Each 128-byte
 *          page is two ZMM tiles selected by `vpermi2b` on the low 7 index bits, with the page chosen by the
 *          high index bits via masked moves. The final partial page is `maskz`-loaded so an unpadded @p table
 *          is never over-read. Out-of-range lanes (none in valid trie use) read as zero.
 */
SZ_HELPER_AUTO __m512i sz_utf8_rune_gather_byte_(sz_u8_t const *table, int count, __m512i indices) {
    __m512i const within_u16x32 = _mm512_and_si512(indices, _mm512_set1_epi16(0x7F));
    __m512i const page_u16x32 = _mm512_srli_epi16(indices, 7);
    int const page_count = (count + 127) / 128;
    // Visit only the distinct 128-byte pages present across the lanes (real text targets ~1-3 of up to ~20), not the
    // whole table: the present-page set is a `1 << page` OR-reduction; an absent page is bit-exact to skip (its
    // `mask_mov` selects no lane).
    __m512i const single_bit_u32x16 = _mm512_set1_epi32(1);
    __m512i const page_low_half_u32x16 = _mm512_cvtepu16_epi32(_mm512_castsi512_si256(page_u16x32));
    __m512i const page_high_half_u32x16 = _mm512_cvtepu16_epi32(_mm512_extracti64x4_epi64(page_u16x32, 1));
    sz_u32_t present_pages = (sz_u32_t)_mm512_reduce_or_epi32(
        _mm512_or_si512(_mm512_sllv_epi32(single_bit_u32x16, page_low_half_u32x16),
                        _mm512_sllv_epi32(single_bit_u32x16, page_high_half_u32x16)));
    present_pages &= page_count >= 32 ? 0xFFFFFFFFu : (((sz_u32_t)1 << page_count) - 1u);
    __m512i result_u16x32 = _mm512_setzero_si512();
    while (present_pages) {
        int const page_index = (int)sz_u64_ctz((sz_u64_t)present_pages);
        present_pages &= present_pages - 1;
        int const tile_base = page_index * 128;
        int const tile_remaining = count - tile_base;
        __m512i const tile_lo_u8x64 = _mm512_maskz_loadu_epi8(sz_u64_clamp_mask_until_((sz_size_t)tile_remaining),
                                                              table + tile_base);
        __m512i const tile_hi_u8x64 = tile_remaining > 64
                                          ? _mm512_maskz_loadu_epi8(
                                                sz_u64_clamp_mask_until_((sz_size_t)(tile_remaining - 64)),
                                                table + tile_base + 64)
                                          : _mm512_setzero_si512();
        __m512i const permuted_u8x64 = _mm512_permutex2var_epi8(tile_lo_u8x64, within_u16x32, tile_hi_u8x64);
        __mmask32 const hit = _mm512_cmpeq_epi16_mask(page_u16x32, _mm512_set1_epi16((short)page_index));
        result_u16x32 = _mm512_mask_mov_epi16(result_u16x32, hit,
                                              _mm512_and_si512(permuted_u8x64, _mm512_set1_epi16(0x00FF)));
    }
    return result_u16x32;
}

/**
 *  @brief  Gather one 16-bit word per lane from a register-resident word table @p table of @p count entries,
 *          addressed by the 16-bit @p indices, using a `vpermi2w` page network (NO `vpgather`). Each 64-word
 *          page is two ZMM tiles selected by `vpermi2w` on the low 6 index bits, page chosen by the high bits.
 *          The final partial page is `maskz`-loaded so an unpadded @p table is never over-read.
 */
SZ_HELPER_AUTO __m512i sz_utf8_rune_gather_word_(sz_u16_t const *table, int count, __m512i indices) {
    __m512i const within_u16x32 = _mm512_and_si512(indices, _mm512_set1_epi16(0x3F));
    __m512i const page_u16x32 = _mm512_srli_epi16(indices, 6);
    int const page_count = (count + 63) / 64;
    // Data-dependent page loop (see `sz_utf8_rune_gather_byte_`): visit only the distinct 64-word pages
    // present across the lanes; bit-exact since an absent page contributes no `hit`.
    __m512i const single_bit_u32x16 = _mm512_set1_epi32(1);
    __m512i const page_low_half_u32x16 = _mm512_cvtepu16_epi32(_mm512_castsi512_si256(page_u16x32));
    __m512i const page_high_half_u32x16 = _mm512_cvtepu16_epi32(_mm512_extracti64x4_epi64(page_u16x32, 1));
    sz_u32_t present_pages = (sz_u32_t)_mm512_reduce_or_epi32(
        _mm512_or_si512(_mm512_sllv_epi32(single_bit_u32x16, page_low_half_u32x16),
                        _mm512_sllv_epi32(single_bit_u32x16, page_high_half_u32x16)));
    present_pages &= page_count >= 32 ? 0xFFFFFFFFu : (((sz_u32_t)1 << page_count) - 1u);
    __m512i result_u16x32 = _mm512_setzero_si512();
    while (present_pages) {
        int const page_index = (int)sz_u64_ctz((sz_u64_t)present_pages);
        present_pages &= present_pages - 1;
        int const tile_base = page_index * 64;
        int const tile_remaining = count - tile_base;
        __m512i const tile_lo_u16x32 = _mm512_maskz_loadu_epi16(sz_u32_clamp_mask_until_((sz_size_t)tile_remaining),
                                                                table + tile_base);
        __m512i const tile_hi_u16x32 = tile_remaining > 32
                                           ? _mm512_maskz_loadu_epi16(
                                                 sz_u32_clamp_mask_until_((sz_size_t)(tile_remaining - 32)),
                                                 table + tile_base + 32)
                                           : _mm512_setzero_si512();
        __m512i const permuted_u16x32 = _mm512_permutex2var_epi16(tile_lo_u16x32, within_u16x32, tile_hi_u16x32);
        __mmask32 const hit = _mm512_cmpeq_epi16_mask(page_u16x32, _mm512_set1_epi16((short)page_index));
        result_u16x32 = _mm512_mask_mov_epi16(result_u16x32, hit, permuted_u16x32);
    }
    return result_u16x32;
}

/**
 *  @brief  In-register two-stage trie classify of BMP codepoints `0x800..0xFFFF`, replacing the scalar
 *          `while (remaining) { ctz; classify_codepoint }` cold loop (anti-pattern #6/#7). Resolves all 64
 *          lanes uniformly via `vpermi2w` over the L1/L2 word tiles and `vpermi2b` over the leaf byte tiles
 *          held in registers; no `vpgather`, no spill. The trie shape is the canonical L1 (super-block) →
 *          L2 (block) → leaf layout:
 *
 *              offset = codepoint - 0x800
 *              block  = offset / block             ; within        = offset % block
 *              super  = block  / superblock        ; super_offset  = block  % superblock
 *              level1 = l1[super]
 *              leaf   = l2[level1 * superblock + super_offset]
 *              class  = leaf_bytes[leaf * block + within]
 *
 *  @param  high        Per-lane `codepoint >> 8` (from @ref sz_utf8_rune_decode_window_).
 *  @param  low         Per-lane `codepoint & 0xFF`.
 *  @param  l1          Stage-1 super-block table (`sz_u8_t`), @p l1_count entries.
 *  @param  l2          Stage-2 block table (`sz_u16_t` leaf indices), @p l2_count entries.
 *  @param  leaf        Leaf class bytes (`sz_u8_t`), @p leaf_count entries.
 *  @param  block       Codepoints per leaf block (power of two).
 *  @param  superblock  Blocks per super-block (power of two).
 *  @param  l1_count    Number of L1 entries (bounds the resident tiles; only @p leaf_count is read here).
 *  @param  l2_count    Number of L2 entries (bounds the resident word tiles).
 *  @param  leaf_count  Number of leaf class bytes (bounds the resident byte tiles).
 *  @return Per-lane class byte; lanes whose codepoint is outside `0x800..0xFFFF` are undefined (caller masks).
 *  @note   @p block and @p superblock are powers of two in the canonical layout; division/modulo lower to a
 *          uniform shift / mask. @p l1_count is accepted for symmetry; the L1 gather is bounded by the page loop.
 */
SZ_HELPER_AUTO __m512i sz_utf8_rune_trie_walk_icelake_( //
    __m512i high, __m512i low,                          //
    sz_u8_t const *l1, sz_u16_t const *l2, sz_u8_t const *leaf, int block, int superblock, int l1_count, int l2_count,
    int leaf_count) {
    __m128i const block_log2_u32x4 = _mm_cvtsi32_si128(sz_u64_ctz((sz_u64_t)block));
    __m128i const super_log2_u32x4 = _mm_cvtsi32_si128(sz_u64_ctz((sz_u64_t)superblock));
    __m512i const within_mask_u16x32 = _mm512_set1_epi16((short)(block - 1));
    __m512i const super_off_mask_u16x32 = _mm512_set1_epi16((short)(superblock - 1));
    __m512i const super_v16_u16x32 = _mm512_set1_epi16((short)superblock);
    __m512i const block_v16_u16x32 = _mm512_set1_epi16((short)block);

    // Reconstruct cp = (high << 8) | low into two 16-bit halves so the trie math stays exact to 0xFFFF.
    __m512i const zero_u16x32 = _mm512_setzero_si512();
    __m512i const offset_lo_u16x32 = _mm512_sub_epi16(
        _mm512_or_si512(_mm512_slli_epi16(_mm512_unpacklo_epi8(high, zero_u16x32), 8),
                        _mm512_unpacklo_epi8(low, zero_u16x32)),
        _mm512_set1_epi16(0x800));
    __m512i const offset_hi_u16x32 = _mm512_sub_epi16(
        _mm512_or_si512(_mm512_slli_epi16(_mm512_unpackhi_epi8(high, zero_u16x32), 8),
                        _mm512_unpackhi_epi8(low, zero_u16x32)),
        _mm512_set1_epi16(0x800));

    __m512i const within_lo_u16x32 = _mm512_and_si512(offset_lo_u16x32, within_mask_u16x32);
    __m512i const within_hi_u16x32 = _mm512_and_si512(offset_hi_u16x32, within_mask_u16x32);
    __m512i const block_idx_lo_u16x32 = _mm512_srl_epi16(offset_lo_u16x32, block_log2_u32x4);
    __m512i const block_idx_hi_u16x32 = _mm512_srl_epi16(offset_hi_u16x32, block_log2_u32x4);
    __m512i const super_off_lo_u16x32 = _mm512_and_si512(block_idx_lo_u16x32, super_off_mask_u16x32);
    __m512i const super_off_hi_u16x32 = _mm512_and_si512(block_idx_hi_u16x32, super_off_mask_u16x32);
    __m512i const super_lo_u16x32 = _mm512_srl_epi16(block_idx_lo_u16x32, super_log2_u32x4);
    __m512i const super_hi_u16x32 = _mm512_srl_epi16(block_idx_hi_u16x32, super_log2_u32x4);

    // Stage 1: level1 = l1[super].
    __m512i const level1_lo_u8x64 = sz_utf8_rune_gather_byte_(l1, l1_count, super_lo_u16x32);
    __m512i const level1_hi_u8x64 = sz_utf8_rune_gather_byte_(l1, l1_count, super_hi_u16x32);

    // Stage 2: leaf = l2[level1 * superblock + super_offset].
    __m512i const l2_index_lo_u16x32 = _mm512_add_epi16(_mm512_mullo_epi16(level1_lo_u8x64, super_v16_u16x32),
                                                        super_off_lo_u16x32);
    __m512i const l2_index_hi_u16x32 = _mm512_add_epi16(_mm512_mullo_epi16(level1_hi_u8x64, super_v16_u16x32),
                                                        super_off_hi_u16x32);
    __m512i const leaf_idx_lo_u16x32 = sz_utf8_rune_gather_word_(l2, l2_count, l2_index_lo_u16x32);
    __m512i const leaf_idx_hi_u16x32 = sz_utf8_rune_gather_word_(l2, l2_count, l2_index_hi_u16x32);

    // Leaf: class = leaf_bytes[leaf * block + within].
    __m512i const leaf_byte_lo_u16x32 = _mm512_add_epi16(_mm512_mullo_epi16(leaf_idx_lo_u16x32, block_v16_u16x32),
                                                         within_lo_u16x32);
    __m512i const leaf_byte_hi_u16x32 = _mm512_add_epi16(_mm512_mullo_epi16(leaf_idx_hi_u16x32, block_v16_u16x32),
                                                         within_hi_u16x32);
    __m512i const class_lo_u8x64 = sz_utf8_rune_gather_byte_(leaf, leaf_count, leaf_byte_lo_u16x32);
    __m512i const class_hi_u8x64 = sz_utf8_rune_gather_byte_(leaf, leaf_count, leaf_byte_hi_u16x32);

    // Re-pack the two 16-bit class halves (each already in [0,255]) into the original byte lane order.
    return _mm512_packus_epi16(class_lo_u8x64, class_hi_u8x64);
}

/**
 *  @brief  256-entry byte LUT read over a 64-byte-aligned 256-byte @p table, per 32-bit lane @p index in [0,256). A
 *          `vpermb` over the four resident quads; the high two bits of the index select the quad via masked blends.
 *          Tiles load directly from `.rodata` (no per-call materialization), so the family classifiers stay
 *          re-init-free. @p table must be `sz_align_(64)` and exactly 256 bytes.
 */
SZ_HELPER_AUTO __m512i sz_utf8_rune_permute256_icelake_(sz_u8_t const *table, __m512i index) {
    __m512i const quad0_u8x64 = _mm512_load_si512((void const *)(table + 0 * 64));
    __m512i const quad1_u8x64 = _mm512_load_si512((void const *)(table + 1 * 64));
    __m512i const quad2_u8x64 = _mm512_load_si512((void const *)(table + 2 * 64));
    __m512i const quad3_u8x64 = _mm512_load_si512((void const *)(table + 3 * 64));
    __m512i const low_six_u32x16 = _mm512_and_si512(index, _mm512_set1_epi32(0x3F));
    __m512i const top_two_u32x16 = _mm512_and_si512(index, _mm512_set1_epi32(0xC0));
    __m512i const permuted0_u32x16 = _mm512_and_si512(_mm512_permutexvar_epi8(low_six_u32x16, quad0_u8x64),
                                                      _mm512_set1_epi32(0xFF));
    __m512i const permuted1_u32x16 = _mm512_and_si512(_mm512_permutexvar_epi8(low_six_u32x16, quad1_u8x64),
                                                      _mm512_set1_epi32(0xFF));
    __m512i const permuted2_u32x16 = _mm512_and_si512(_mm512_permutexvar_epi8(low_six_u32x16, quad2_u8x64),
                                                      _mm512_set1_epi32(0xFF));
    __m512i const permuted3_u32x16 = _mm512_and_si512(_mm512_permutexvar_epi8(low_six_u32x16, quad3_u8x64),
                                                      _mm512_set1_epi32(0xFF));
    __m512i result_u32x16 = permuted0_u32x16;
    result_u32x16 = _mm512_mask_blend_epi32(_mm512_cmpeq_epi32_mask(top_two_u32x16, _mm512_set1_epi32(0x40)),
                                            result_u32x16, permuted1_u32x16);
    result_u32x16 = _mm512_mask_blend_epi32(_mm512_cmpeq_epi32_mask(top_two_u32x16, _mm512_set1_epi32(0x80)),
                                            result_u32x16, permuted2_u32x16);
    result_u32x16 = _mm512_mask_blend_epi32(_mm512_cmpeq_epi32_mask(top_two_u32x16, _mm512_set1_epi32(0xC0)),
                                            result_u32x16, permuted3_u32x16);
    return result_u32x16;
}

/**
 *  @brief  Gather-free indexed read of a 64-byte-aligned byte LUT spanning @p tile_count tiles of 64 bytes, with a
 *          per-32-bit-lane byte @p index_dwords in [0, tile_count*64). A `vpermi2b` cascade: each ZMM pair covers 128
 *          byte slots addressed by the low 7 bits; the high bits select the pair via masked blends. Tiles load
 *          directly from the aligned `.rodata` @p table — no `luts` struct, no per-call init. No `vpgatherdd`.
 *          @p table must be `sz_align_(64)` and zero-padded to `tile_count * 64` bytes.
 */
SZ_HELPER_AUTO __m512i sz_utf8_rune_lut_cascade_icelake_(sz_u8_t const *table, int tile_count, __m512i index_dwords) {
    __m512i const within_u32x16 = _mm512_and_si512(index_dwords, _mm512_set1_epi32(0x7F));
    __m512i const selector_u32x16 = _mm512_srli_epi32(index_dwords, 7);
    __m512i result_u32x16 = _mm512_setzero_si512();
    int const pairs = (tile_count + 1) / 2;
    for (int pair = 0; pair < pairs; ++pair) {
        __m512i const low_tile_u8x64 = _mm512_load_si512((void const *)(table + (pair * 2) * 64));
        __m512i const high_tile_u8x64 = (pair * 2 + 1 < tile_count)
                                            ? _mm512_load_si512((void const *)(table + (pair * 2 + 1) * 64))
                                            : _mm512_setzero_si512();
        __m512i const picked_u8x64 = _mm512_permutex2var_epi8(low_tile_u8x64, within_u32x16, high_tile_u8x64);
        __mmask16 const here = _mm512_cmpeq_epi32_mask(selector_u32x16, _mm512_set1_epi32(pair));
        result_u32x16 = _mm512_mask_blend_epi32(here, result_u32x16,
                                                _mm512_and_si512(picked_u8x64, _mm512_set1_epi32(0xFF)));
    }
    return result_u32x16;
}

/**
 *  @brief  Indexed read of a 4-bit-per-cell LUT: @p packed holds two output nibbles per byte (cell `i` is the low
 *          nibble of `packed[i/2]` for even `i`, the high nibble for odd `i`). Halves the table and so HALVES the
 *          `vpermi2b` cascade depth vs a byte-per-cell layout, for tables whose outputs fit in 4 bits (e.g. the
 *          grapheme `stage_sub` descriptor index, the word `astral_leaf` class). @p tile_count counts the packed
 *          tiles; @p index_dwords is the unpacked cell index per 32-bit lane. Reads straight from aligned `.rodata`.
 */
SZ_HELPER_AUTO __m512i sz_utf8_rune_lut_cascade_nibble_icelake_(sz_u8_t const *packed, int tile_count,
                                                                __m512i index_dwords) {
    __m512i const byte_index_u32x16 = _mm512_srli_epi32(index_dwords, 1);
    __m512i const packed_byte_u8x64 = sz_utf8_rune_lut_cascade_icelake_(packed, tile_count, byte_index_u32x16);
    __mmask16 const odd_cell = _mm512_test_epi32_mask(index_dwords, _mm512_set1_epi32(1));
    __m512i const low_nibble_u32x16 = _mm512_and_si512(packed_byte_u8x64, _mm512_set1_epi32(0x0F));
    __m512i const high_nibble_u32x16 = _mm512_and_si512(_mm512_srli_epi32(packed_byte_u8x64, 4),
                                                        _mm512_set1_epi32(0x0F));
    return _mm512_mask_blend_epi32(odd_cell, low_nibble_u32x16, high_nibble_u32x16);
}

#pragma endregion In register two stage trie

#pragma region Drains

/**
 *  @brief  Emit boundary-lane offsets within an effective window via `vpcompressb`, honoring @p capacity and a
 *          carried previous-boundary position. Effective-window aware: only lanes in `[effective_lo, effective_hi]`
 *          (those with full in-register context) are trusted; the caller advances by `effective_step < 64` so
 *          edge lanes serve as context only and are never re-walked scalar-wise.
 *
 *  The `boundary` mask is pre-masked by the caller to the trusted band. Each set lane `i` opens a segment whose
 *  start is the previous boundary position and whose length reaches to `base + i`. Output is widened to 64-bit
 *  `starts[]` / `lengths[]` in waves of eight, carrying the open segment across waves and windows via @p previous_io.
 */
SZ_HELPER_AUTO sz_size_t sz_utf8_rune_drain_forward_( //
    sz_u64_t boundary, sz_size_t base, __m512i lane_identity, sz_size_t *starts, sz_size_t *lengths, sz_size_t produced,
    sz_size_t capacity, sz_size_t *previous_io) {
    __m512i const wave_shift_u8x64 = _mm512_add_epi8(lane_identity, _mm512_set1_epi8(8));
    sz_size_t const boundary_count = (sz_size_t)_mm_popcnt_u64(boundary);
    __m512i packed_u8x64 = _mm512_maskz_compress_epi8(_cvtu64_mask64(boundary), lane_identity);
    sz_u64_t previous = (sz_u64_t)*previous_io;
    sz_size_t emitted = 0;
    while (emitted < boundary_count && produced < capacity) {
        sz_size_t const wave = sz_min_of_three(boundary_count - emitted, capacity - produced, 8);
        __m512i const positions_u64x8 = _mm512_add_epi64(_mm512_cvtepu8_epi64(_mm512_castsi512_si128(packed_u8x64)),
                                                         _mm512_set1_epi64((long long)base));
        __m512i const segment_starts_u64x8 = _mm512_alignr_epi64(positions_u64x8,
                                                                 _mm512_set1_epi64((long long)previous), 7);
        __mmask8 const store_mask = sz_u8_clamp_mask_until_(wave);
        _mm512_mask_storeu_epi64((void *)(starts + produced), store_mask, segment_starts_u64x8);
        _mm512_mask_storeu_epi64((void *)(lengths + produced), store_mask,
                                 _mm512_sub_epi64(positions_u64x8, segment_starts_u64x8));
        produced += wave, emitted += wave;
        previous = (sz_u64_t)(starts[produced - 1] + lengths[produced - 1]);
        packed_u8x64 = _mm512_permutexvar_epi8(wave_shift_u8x64, packed_u8x64);
    }
    *previous_io = (sz_size_t)previous;
    return produced;
}

/** @brief  Bring the @p block-th group of 16 bytes of @p value down to the low 128 bits (for `vpmovzxbd` widening),
 *          selecting the group with a runtime `vpermb` (the block index need not be a compile-time immediate). */
SZ_HELPER_INLINE __m128i sz_utf8_rune_pick16_icelake_(__m512i value, __m512i lane_identity, int block) {
    return _mm512_castsi512_si128(
        _mm512_permutexvar_epi8(_mm512_add_epi8(lane_identity, _mm512_set1_epi8((char)(block * 16))), value));
}

/**
 *  @brief  Decode the dense set of emitted-start lanes @p emit_starts of a classified window into sequential UTF-32
 *          runes, the rune-valued sibling of @ref sz_utf8_rune_drain_forward_. `vpcompressb` packs the start
 *          byte-offsets, three sibling permutes gather the lead + up to three trailing bytes per codepoint, and a
 *          width-blend (1/2/3/4-byte) assembles each value branchlessly in 16-lane blocks. A narrower lead always
 *          blends in a value that is already inert for its width, so all four widths run unconditionally.
 *
 *  TOTAL decode: @p emit_starts also covers promoted orphan continuation bytes, and @p ill_formed marks every start
 *  lane whose maximal ill-formed subpart must collapse to a single U+FFFD (Unicode 17.0 §3.9 / W3C). After the
 *  width-blend assembles the (for ill-formed lanes, garbage) values, those lanes are overwritten with U+FFFD per
 *  16-lane block. The resume cursor is read from @p consumed_length (the per-lane maximal-subpart length, in window
 *  order) at the last emitted lane, so an ill-formed trailing lane never skips bytes that owe their own next U+FFFD.
 *  @return Number of runes emitted; sets @p consumed_bytes to the byte span they cover (the resume cursor delta).
 */
SZ_HELPER_AUTO sz_size_t sz_utf8_rune_drain_icelake_( //
    __m512i window, sz_u64_t emit_starts, sz_u64_t ill_formed, __m512i consumed_length, __m512i lane_identity,
    int has_three, int has_four, sz_size_t emit_count, sz_rune_t *runes, sz_size_t capacity,
    sz_size_t *consumed_bytes) {

    __mmask64 const compress_mask = _cvtu64_mask64(emit_starts);
    __m512i const start_offsets_u8x64 = _mm512_maskz_compress_epi8(compress_mask, lane_identity);
    __m512i const consumed_compressed_u8x64 = _mm512_maskz_compress_epi8(compress_mask, consumed_length);
    __m512i const lead_bytes_u8x64 = _mm512_permutexvar_epi8(start_offsets_u8x64, window);
    __m512i const second_bytes_u8x64 = _mm512_permutexvar_epi8(
        _mm512_add_epi8(start_offsets_u8x64, _mm512_set1_epi8(1)), window);
    // The 3rd and 4th trailing bytes are gathered only when a 3/4-byte lead is present in this window (their
    // forms are otherwise dead, and a narrower lead never satisfies the `is_three`/`is_four` blend predicate).
    // `third_bytes_u8x64` is hoisted so the 4-byte sibling reuses it (has_four implies has_three).
    __m512i third_bytes_u8x64 = _mm512_setzero_si512();
    __m512i fourth_bytes_u8x64 = _mm512_setzero_si512();
    if (has_three)
        third_bytes_u8x64 = _mm512_permutexvar_epi8(_mm512_add_epi8(start_offsets_u8x64, _mm512_set1_epi8(2)), window);
    if (has_four)
        fourth_bytes_u8x64 = _mm512_permutexvar_epi8(_mm512_add_epi8(start_offsets_u8x64, _mm512_set1_epi8(3)), window);

    // Per-lane (window-order) ill-formed selector, broadcast into a dword mask we can compact in lockstep with the
    // start offsets: lane holds 0xFF where the start is ill-formed, 0 otherwise. After the same `vpcompressb`, lane
    // `j` of `ill_compressed` is the ill-formed flag of the `j`-th emitted start.
    __m512i const ill_byte_u8x64 = _mm512_maskz_mov_epi8(_cvtu64_mask64(ill_formed), _mm512_set1_epi8((char)0xFF));
    __m512i const ill_compressed_u8x64 = _mm512_maskz_compress_epi8(compress_mask, ill_byte_u8x64);

    sz_size_t const want = emit_count < capacity ? emit_count : capacity;
    sz_size_t produced = 0;
    for (sz_size_t block_start = 0; block_start < want; block_start += 16) {
        int const block = (int)(block_start >> 4);
        __m512i const b0_u32x16 = _mm512_cvtepu8_epi32(
            sz_utf8_rune_pick16_icelake_(lead_bytes_u8x64, lane_identity, block));
        __m512i const b1_u32x16 = _mm512_cvtepu8_epi32(
            sz_utf8_rune_pick16_icelake_(second_bytes_u8x64, lane_identity, block));

        // Width-blend by the lead-byte range: each wider path only fires for its own lead, and a value blended into
        // a narrower lead is already inert, so the cases compose and stay bit-identical. The 2-byte form is always
        // assembled; the 3/4-byte forms (and their trailing-byte widens) run only when such a lead is present in the
        // window, so a 3-byte window never gathers/assembles the 4th byte (the regression that flattening added).
        __mmask16 const is_two = _kandn_mask16(_mm512_cmpge_epu32_mask(b0_u32x16, _mm512_set1_epi32(0xE0)),
                                               _mm512_cmpge_epu32_mask(b0_u32x16, _mm512_set1_epi32(0xC0)));
        __m512i const two_byte_u32x16 = _mm512_or_si512(
            _mm512_slli_epi32(_mm512_and_si512(b0_u32x16, _mm512_set1_epi32(0x1F)), 6),
            _mm512_and_si512(b1_u32x16, _mm512_set1_epi32(0x3F)));
        __m512i codepoints_u32x16 = _mm512_mask_blend_epi32(is_two, b0_u32x16, two_byte_u32x16);
        if (has_three) {
            __m512i const b2_u32x16 = _mm512_cvtepu8_epi32(
                sz_utf8_rune_pick16_icelake_(third_bytes_u8x64, lane_identity, block));
            __mmask16 const is_three = _kandn_mask16(_mm512_cmpge_epu32_mask(b0_u32x16, _mm512_set1_epi32(0xF0)),
                                                     _mm512_cmpge_epu32_mask(b0_u32x16, _mm512_set1_epi32(0xE0)));
            __m512i const three_byte_u32x16 = _mm512_or_si512(
                _mm512_or_si512(_mm512_slli_epi32(_mm512_and_si512(b0_u32x16, _mm512_set1_epi32(0x0F)), 12),
                                _mm512_slli_epi32(_mm512_and_si512(b1_u32x16, _mm512_set1_epi32(0x3F)), 6)),
                _mm512_and_si512(b2_u32x16, _mm512_set1_epi32(0x3F)));
            codepoints_u32x16 = _mm512_mask_blend_epi32(is_three, codepoints_u32x16, three_byte_u32x16);
        }
        if (has_four) {
            __m512i const b2_u32x16 = _mm512_cvtepu8_epi32(
                sz_utf8_rune_pick16_icelake_(third_bytes_u8x64, lane_identity, block));
            __m512i const b3_u32x16 = _mm512_cvtepu8_epi32(
                sz_utf8_rune_pick16_icelake_(fourth_bytes_u8x64, lane_identity, block));
            __mmask16 const is_four = _mm512_cmpge_epu32_mask(b0_u32x16, _mm512_set1_epi32(0xF0));
            __m512i const four_byte_u32x16 = _mm512_or_si512(
                _mm512_or_si512(_mm512_slli_epi32(_mm512_and_si512(b0_u32x16, _mm512_set1_epi32(0x07)), 18),
                                _mm512_slli_epi32(_mm512_and_si512(b1_u32x16, _mm512_set1_epi32(0x3F)), 12)),
                _mm512_or_si512(_mm512_slli_epi32(_mm512_and_si512(b2_u32x16, _mm512_set1_epi32(0x3F)), 6),
                                _mm512_and_si512(b3_u32x16, _mm512_set1_epi32(0x3F))));
            codepoints_u32x16 = _mm512_mask_blend_epi32(is_four, codepoints_u32x16, four_byte_u32x16);
        }

        // Overwrite every ill-formed lane in this block with U+FFFD (the garbage decode is discarded).
        __m512i const ill_block_u32x16 = _mm512_cvtepu8_epi32(
            sz_utf8_rune_pick16_icelake_(ill_compressed_u8x64, lane_identity, block));
        __mmask16 const ill_mask = _mm512_test_epi32_mask(ill_block_u32x16, ill_block_u32x16);
        codepoints_u32x16 = _mm512_mask_mov_epi32(codepoints_u32x16, ill_mask,
                                                  _mm512_set1_epi32((int)sz_rune_replacement_k));
        sz_size_t lanes = want - produced;
        if (lanes > 16) lanes = 16;
        _mm512_mask_storeu_epi32(runes + produced, sz_u16_mask_until_(lanes), codepoints_u32x16);
        produced += lanes;
    }

    // Resume cursor delta = end of the last emitted start = its byte offset + its maximal-subpart length. For a
    // well-formed lane that subpart length equals the declared rune length; for an ill-formed lane it is the 1-3 byte
    // subpart, so the cursor never skips bytes that must become their own next-window U+FFFD.
    sz_u8_t const start_off = (sz_u8_t)(_mm_cvtsi128_si32(_mm512_castsi512_si128(_mm512_permutexvar_epi8(
                                            _mm512_set1_epi8((char)(produced - 1)), start_offsets_u8x64))) &
                                        0xFF);
    sz_u8_t const last_length = (sz_u8_t)(_mm_cvtsi128_si32(_mm512_castsi512_si128(_mm512_permutexvar_epi8(
                                              _mm512_set1_epi8((char)(produced - 1)), consumed_compressed_u8x64))) &
                                          0xFF);
    *consumed_bytes = (sz_size_t)start_off + (sz_size_t)last_length;
    return produced;
}

#pragma endregion Drains

#pragma endregion Shared SIMD leaf substrate

SZ_API_COMPTIME sz_size_t sz_utf8_count_icelake(sz_cptr_t text, sz_size_t length) {
    // Count every byte that begins a codepoint (non-continuation) via the shared one-op start-byte test.
    sz_u8_t const *text_u8 = (sz_u8_t const *)text;
    sz_size_t char_count = 0;

    // Process 64 bytes at a time
    sz_u512_vec_t text_vec;
    while (length >= 64) {
        text_vec.zmm = _mm512_loadu_epi8(text_u8);
        __mmask64 start_byte_mask = sz_utf8_rune_start_mask_icelake_(text_vec.zmm, ~(__mmask64)0);
        char_count += _mm_popcnt_u64(_cvtmask64_u64(start_byte_mask));
        text_u8 += 64;
        length -= 64;
    }

    // Process remaining bytes with a masked variant - lanes outside `load_mask` stay clear in the result.
    if (length) {
        __mmask64 load_mask = sz_u64_mask_until_(length);
        text_vec.zmm = _mm512_maskz_loadu_epi8(load_mask, text_u8);
        __mmask64 start_byte_mask = sz_utf8_rune_start_mask_icelake_(text_vec.zmm, load_mask);
        char_count += _mm_popcnt_u64(_cvtmask64_u64(start_byte_mask));
    }
    return char_count;
}

SZ_API_COMPTIME sz_cptr_t sz_utf8_seek_icelake(sz_cptr_t text, sz_size_t length, sz_size_t n) {

    // The logic of this function is similar to `sz_utf8_count_icelake`, but uses PDEP to locate the Nth start byte
    // within a window in one step. The start-byte test is the same shared one-op `vpcmpgtb(-65)` form.
    sz_u8_t const *text_u8 = (sz_u8_t const *)text;

    // Process 64 bytes at a time
    sz_u512_vec_t text_vec;
    while (length >= 64) {
        text_vec.zmm = _mm512_loadu_epi8(text_u8);
        sz_u64_t start_byte_mask = _cvtmask64_u64(sz_utf8_rune_start_mask_icelake_(text_vec.zmm, ~(__mmask64)0));
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
    return sz_utf8_seek_serial((sz_cptr_t)text_u8, length, n);
}

/**
 *  @brief  Decode one window of @p text into dense UTF-32 @p runes by the uniform "classify → per-lane well-formed +
 *          orphan promotion → compress emitted starts → gather → width-blend → blend U+FFFD" path, emitting at
 *          most @p runes_capacity runes and returning the resume cursor. Pure ASCII takes a dedicated `vpmovzxbd`
 *          widen lane. The decode is TOTAL: clean and dirty bytes are handled in-vector, one U+FFFD per maximal
 *          ill-formed subpart (Unicode 17.0 §3.9 / W3C), bit-exact with @ref sz_utf8_decode_serial. The step
 *          declines (`*runes_unpacked == 0`, cursor unchanged) ONLY when the first lead's declared sequence crosses
 *          the window edge (a boundary truncation), which the public entry finalizes without a serial re-decode.
 */
SZ_HELPER_AUTO sz_cptr_t sz_utf8_decode_once_icelake_( //
    sz_cptr_t text, sz_size_t length,                  //
    sz_rune_t *runes, sz_size_t runes_capacity,        //
    sz_size_t *runes_unpacked) {

    __m512i const lane_identity_u8x64 = sz_utf8_lane_identity_icelake_();
    sz_size_t const chunk = length < 64 ? length : 64;
    __mmask64 const load_mask = sz_u64_mask_until_(chunk);
    __m512i const window_u8x64 = _mm512_maskz_loadu_epi8(load_mask, (sz_u8_t const *)text);

    // ASCII fast lane: an entire window of 1-byte runes widens directly with `vpmovzxbd`, no classification needed.
    if ((_mm512_movepi8_mask(window_u8x64) & load_mask) == 0) {
        sz_size_t const runes_to_unpack = chunk < runes_capacity ? chunk : runes_capacity;
        _mm512_mask_storeu_epi32(runes, sz_u16_clamp_mask_until_(runes_to_unpack),
                                 _mm512_cvtepu8_epi32(_mm512_castsi512_si128(window_u8x64)));
        if (runes_to_unpack > 16)
            _mm512_mask_storeu_epi32(runes + 16, sz_u16_clamp_mask_until_(runes_to_unpack - 16),
                                     _mm512_cvtepu8_epi32(_mm512_extracti32x4_epi32(window_u8x64, 1)));
        if (runes_to_unpack > 32)
            _mm512_mask_storeu_epi32(runes + 32, sz_u16_clamp_mask_until_(runes_to_unpack - 32),
                                     _mm512_cvtepu8_epi32(_mm512_extracti32x4_epi32(window_u8x64, 2)));
        if (runes_to_unpack > 48)
            _mm512_mask_storeu_epi32(runes + 48, sz_u16_clamp_mask_until_(runes_to_unpack - 48),
                                     _mm512_cvtepu8_epi32(_mm512_extracti32x4_epi32(window_u8x64, 3)));
        *runes_unpacked = runes_to_unpack;
        return text + runes_to_unpack;
    }

    // Single-source classification: a high-nibble LUT gives the per-lane byte length, and both the validity gate and
    // the width-blend derive from it, so a lead and its length can never disagree (the class of malformed-gate bug).
    __m512i const length_lut_u8x64 = _mm512_broadcast_i32x4(
        _mm_setr_epi8(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 3, 4));
    __mmask64 const starts = sz_utf8_rune_start_mask_icelake_(window_u8x64, load_mask);
    __mmask64 const continuations = _kandn_mask64(starts, load_mask);
    __m512i const high_nibble_u8x64 = _mm512_and_si512(_mm512_srli_epi16(window_u8x64, 4), _mm512_set1_epi8(0x0F));
    __m512i const lengths_u8x64 = _mm512_shuffle_epi8(length_lut_u8x64, high_nibble_u8x64);
    sz_u64_t const starts_bits = _cvtmask64_u64(starts);
    // Any start whose declared sequence would reach past the window is deferred: well-formed text has only the
    // trailing one (a resumable truncation), but a malformed lead-in-lead (e.g. `E0 C0`) can overrun earlier - the
    // FIRST overrunning start bounds the decodable prefix, and its bytes resume in the next window or via serial.
    __m512i const sequence_end_u8x64 = _mm512_add_epi8(lane_identity_u8x64, lengths_u8x64);
    __mmask64 const overruns = _kand_mask64(_mm512_cmpgt_epu8_mask(sequence_end_u8x64, _mm512_set1_epi8((char)chunk)),
                                            starts);
    sz_u64_t const overruns_bits = _cvtmask64_u64(overruns);
    sz_size_t const decodable_end = overruns_bits ? (sz_size_t)_tzcnt_u64(overruns_bits) : chunk;
    sz_u64_t const decodable_mask = _cvtmask64_u64(sz_u64_mask_until_(decodable_end));

    // Per-lane validity, classifying every lane uniformly (no per-lane loop, no decline). The window is decoded TOTAL:
    // well-formed leads decode to their value, ill-formed leads (bad lead, broken continuation chain, overlong /
    // surrogate / out-of-range first continuation) and orphan continuation bytes each collapse to one U+FFFD over the
    // maximal ill-formed subpart (Unicode 17.0 §3.9 / W3C), bit-exact with the serial reference.
    __mmask64 const length_ge_two = _kand_mask64(_mm512_cmpge_epu8_mask(lengths_u8x64, _mm512_set1_epi8(2)), starts);
    __mmask64 const length_ge_three = _kand_mask64(_mm512_cmpge_epu8_mask(lengths_u8x64, _mm512_set1_epi8(3)), starts);
    __mmask64 const length_ge_four = _kand_mask64(_mm512_cmpge_epu8_mask(lengths_u8x64, _mm512_set1_epi8(4)), starts);
    sz_u64_t const length_one_bits = starts_bits & ~_cvtmask64_u64(length_ge_two);
    sz_u64_t const length_two_bits = _cvtmask64_u64(length_ge_two) & ~_cvtmask64_u64(length_ge_three);
    sz_u64_t const length_three_bits = _cvtmask64_u64(length_ge_three) & ~_cvtmask64_u64(length_ge_four);
    sz_u64_t const length_four_bits = _cvtmask64_u64(length_ge_four);
    sz_u64_t const continuation_bits = _cvtmask64_u64(continuations);

    // Bad lead: 0xC0/0xC1 (overlong 2-byte by the LUT) or 0xF5..0xFF (out of range, length-4 by the LUT).
    __mmask64 const bad_lead = _kor_mask64(
        _kand_mask64(_mm512_cmpeq_epi8_mask(lengths_u8x64, _mm512_set1_epi8(2)),
                     _mm512_cmplt_epu8_mask(window_u8x64, _mm512_set1_epi8((char)0xC2))),
        _kand_mask64(_mm512_cmpeq_epi8_mask(lengths_u8x64, _mm512_set1_epi8(4)),
                     _mm512_cmpgt_epu8_mask(window_u8x64, _mm512_set1_epi8((char)0xF4))));
    sz_u64_t const bad_lead_bits = _cvtmask64_u64(bad_lead);
    // First-continuation range violations for E0/ED/F0/F4 (overlong 3/4-byte, surrogate, > U+10FFFF). Computed
    // whenever a 3/4-byte lead is present; for any other lead this mask is empty, so `b1_ok` keeps the lane.
    int const has_three = _cvtmask64_u64(length_ge_three) != 0;
    int const has_four = _cvtmask64_u64(length_ge_four) != 0;
    sz_u64_t overlong_or_surrogate_or_range_bits = 0;
    if (has_three || has_four) {
        __m512i const next_byte_u8x64 = _mm512_permutexvar_epi8(
            _mm512_add_epi8(lane_identity_u8x64, _mm512_set1_epi8(1)), window_u8x64);
        __mmask64 const overlong_or_surrogate_or_range = _kor_mask64(
            _kor_mask64(_kand_mask64(_mm512_cmpeq_epi8_mask(window_u8x64, _mm512_set1_epi8((char)0xE0)),
                                     _mm512_cmplt_epu8_mask(next_byte_u8x64, _mm512_set1_epi8((char)0xA0))),
                        _kand_mask64(_mm512_cmpeq_epi8_mask(window_u8x64, _mm512_set1_epi8((char)0xED)),
                                     _mm512_cmpge_epu8_mask(next_byte_u8x64, _mm512_set1_epi8((char)0xA0)))),
            _kor_mask64(_kand_mask64(_mm512_cmpeq_epi8_mask(window_u8x64, _mm512_set1_epi8((char)0xF0)),
                                     _mm512_cmplt_epu8_mask(next_byte_u8x64, _mm512_set1_epi8((char)0x90))),
                        _kand_mask64(_mm512_cmpeq_epi8_mask(window_u8x64, _mm512_set1_epi8((char)0xF4)),
                                     _mm512_cmpge_epu8_mask(next_byte_u8x64, _mm512_set1_epi8((char)0x90)))));
        overlong_or_surrogate_or_range_bits = _cvtmask64_u64(overlong_or_surrogate_or_range) & starts_bits;
    }

    // Per-lane continuation availability at the declared trailing slots, evaluated at the lead lane.
    sz_u64_t const cont1 = starts_bits & (continuation_bits >> 1);
    sz_u64_t const cont2 = starts_bits & (continuation_bits >> 2);
    sz_u64_t const cont3 = starts_bits & (continuation_bits >> 3);
    sz_u64_t const b1_range_ok = starts_bits & ~overlong_or_surrogate_or_range_bits;
    // A valid first continuation: present, range-valid, and the lead is not itself bad (C0/C1, F5..FF).
    sz_u64_t const first_ok = cont1 & b1_range_ok & ~bad_lead_bits;

    // Well-formed leads (vectorized `sz_rune_decode` success), restricted to the decodable span.
    sz_u64_t const wf1 = length_one_bits;
    sz_u64_t const wf2 = length_two_bits & ~bad_lead_bits & cont1;
    sz_u64_t const wf3 = length_three_bits & b1_range_ok & cont1 & cont2;
    sz_u64_t const wf4 = length_four_bits & ~bad_lead_bits & b1_range_ok & cont1 & cont2 & cont3;
    sz_u64_t const well_formed = (wf1 | wf2 | wf3 | wf4) & decodable_mask;

    // Per-lane maximal-subpart length (mirror of `sz_utf8_maximal_subpart_`): start at 1 and extend across each
    // continuation slot that a well-formed sequence would still accept. For well-formed lanes this equals the
    // declared length; for ill-formed lanes it is the 1-3 byte subpart that one U+FFFD consumes.
    sz_u64_t const step2 = _cvtmask64_u64(length_ge_two) & first_ok;
    sz_u64_t const step3 = step2 & _cvtmask64_u64(length_ge_three) & cont2;
    sz_u64_t const step4 = step3 & _cvtmask64_u64(length_ge_four) & cont3;

    // Orphan promotion: a continuation byte not covered by ANY lead's maximal-subpart span (well-formed OR the bytes
    // an ill-formed lead's single U+FFFD consumes) becomes its own 1-byte U+FFFD. The subpart spans are exactly the
    // continuation slots the `step2/3/4` adds reached, so coverage is those slots smeared by their offset.
    sz_u64_t const covered = ((step2 & decodable_mask) << 1) | ((step3 & decodable_mask) << 2) |
                             ((step4 & decodable_mask) << 3);
    sz_u64_t const orphan = continuation_bits & decodable_mask & ~covered;
    sz_u64_t const emit_starts = (starts_bits | orphan) & decodable_mask;
    sz_size_t const emit_count = (sz_size_t)_mm_popcnt_u64(emit_starts);
    if (emit_count == 0) { return *runes_unpacked = 0, text; } // Nothing decodable → window-edge finalize in driver.
    sz_u64_t const ill_formed = emit_starts & ~well_formed;    // Orphans are continuations → never well-formed.
    __m512i consumed_length_u8x64 = _mm512_set1_epi8(1);
    __m512i const one_u8x64 = _mm512_set1_epi8(1);
    consumed_length_u8x64 = _mm512_mask_add_epi8(consumed_length_u8x64, _cvtu64_mask64(step2), consumed_length_u8x64,
                                                 one_u8x64);
    consumed_length_u8x64 = _mm512_mask_add_epi8(consumed_length_u8x64, _cvtu64_mask64(step3), consumed_length_u8x64,
                                                 one_u8x64);
    consumed_length_u8x64 = _mm512_mask_add_epi8(consumed_length_u8x64, _cvtu64_mask64(step4), consumed_length_u8x64,
                                                 one_u8x64);

    // Drain-side wide-byte gating: a 3/4-byte trailing gather is needed only when such a lead appears in this window.
    // `length_ge_three` already subsumes `length_ge_four`, so `(length_ge_three | length_ge_four) != 0` is just
    // `length_ge_three != 0`. Distinct from the validity-gate `has_three`/`has_four` above (those guard the
    // first-continuation range-check) — these only steer the drain's conditional trailing-byte assembly.
    int const drain_has_three = (_cvtmask64_u64(length_ge_three) | _cvtmask64_u64(length_ge_four)) != 0;
    int const drain_has_four = _cvtmask64_u64(length_ge_four) != 0;
    sz_size_t consumed = 0;
    sz_size_t const produced = sz_utf8_rune_drain_icelake_(window_u8x64, emit_starts, ill_formed, consumed_length_u8x64,
                                                           lane_identity_u8x64, drain_has_three, drain_has_four,
                                                           emit_count, runes, runes_capacity, &consumed);
    *runes_unpacked = produced;
    return text + consumed;
}

SZ_API_COMPTIME sz_cptr_t sz_utf8_decode_icelake( //
    sz_cptr_t text, sz_size_t length,             //
    sz_rune_t *runes, sz_size_t runes_capacity,   //
    sz_size_t *runes_unpacked) {

    sz_cptr_t cursor = text;
    sz_cptr_t const end = text + length;
    sz_size_t runes_written = 0;
    while (runes_written < runes_capacity && cursor < end) {
        sz_size_t step_unpacked = 0;
        sz_cptr_t next = sz_utf8_decode_once_icelake_(cursor, (sz_size_t)(end - cursor), runes + runes_written,
                                                      runes_capacity - runes_written, &step_unpacked);
        if (step_unpacked) {
            runes_written += step_unpacked;
            cursor = next;
            continue;
        }
        // The in-vector step decodes its whole decodable span; `step_unpacked == 0` only when the very first lead
        // declares a sequence crossing the window edge (a boundary truncation). A resumable truncation breaks and
        // awaits more bytes; a bad/overlong truncated lead at the edge finalizes to one U+FFFD over its maximal
        // ill-formed subpart, a bounded <=3-byte finalize, never a serial window re-decode.
        if (sz_utf8_incomplete_tail_(cursor, end)) break;
        runes[runes_written++] = (sz_rune_t)sz_rune_replacement_k;
        cursor += sz_utf8_maximal_subpart_(cursor, end);
    }
    *runes_unpacked = runes_written;
    return cursor;
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

#endif // STRINGZILLA_UTF8_RUNES_ICELAKE_H_
