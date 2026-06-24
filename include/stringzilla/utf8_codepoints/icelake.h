/**
 *  @brief Ice Lake (AVX-512 VBMI+VAES) backend for UTF-8 codepoint mechanics and shared SIMD substrate.
 *  @file include/stringzilla/utf8_codepoints/icelake.h
 *  @author Ash Vardanian
 */
#ifndef STRINGZILLA_UTF8_CODEPOINTS_ICELAKE_H_
#define STRINGZILLA_UTF8_CODEPOINTS_ICELAKE_H_

#include "stringzilla/types.h"
#include "stringzilla/utf8_codepoints/serial.h"

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

#pragma region Shared SIMD leaf substrate

/*  Family-agnostic AVX-512 leaf helpers shared by every UTF-8 segmentation kernel (word / grapheme /
 *  sentence / line / delimiters). Each family `#include`s this header and calls these `sz_utf8_codepoints_*`
 *  primitives instead of carrying its own copy. The substrate bakes in zero property semantics: classifier
 *  tables flow in as bare pointers, boundary algebra flows in as bare masks, so the same code resolves any
 *  break property. Every routine treats all 64 lanes uniformly with no scalar per-lane loop, no gather, and
 *  no spill-to-stack-then-reload round-trip. */

/** @brief  Descending byte lane identity {63,62,...,1,0} for `vpcompressb`-based drains and permute waves. */
SZ_INTERNAL __m512i sz_utf8_codepoints_lane_identity_icelake_(void) {
    return _mm512_set_epi8(                                             //
        63, 62, 61, 60, 59, 58, 57, 56, 55, 54, 53, 52, 51, 50, 49, 48, //
        47, 46, 45, 44, 43, 42, 41, 40, 39, 38, 37, 36, 35, 34, 33, 32, //
        31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, //
        15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
}

/**
 *  @brief  Emit a window's @p emit delimiter matches in-register via `vpcompressb` and `ceil(emit/8)` masked
 *          widen-stores. `_mm512_alignr_epi64` shifts the compressed registers down between waves, so @p emit may
 *          exceed 8. Per-lane byte length is 1, plus 1 on a 2-byte start, plus 2 on a 3-byte start (disjoint masks).
 */
SZ_INTERNAL void sz_utf8_codepoints_peel_icelake_(                               //
    sz_u64_t start_bits, __mmask64 two_byte_starts, __mmask64 three_byte_starts, //
    sz_size_t emit, sz_size_t position, __m512i lane_identity,                   //
    sz_size_t *match_offsets, sz_size_t *match_lengths) {
    __mmask64 const compress_mask = _cvtu64_mask64(start_bits);

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

#pragma region Decode window

/** @brief  Per-byte logical right shift by @p shift, retaining only the low @p keep bits of every lane. */
SZ_INTERNAL __m512i sz_utf8_codepoints_srl8_(__m512i value, int shift, sz_u8_t keep) {
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
typedef struct sz_utf8_codepoints_window_t {
    __m512i window;              /**< The raw 64 input bytes (continuation bytes included). */
    __m512i high;                /**< Per-lane `codepoint >> 8` for the codepoint that starts at this lane. */
    __m512i low;                 /**< Per-lane `codepoint & 0xFF` for the codepoint that starts at this lane. */
    __mmask64 continuation;      /**< Bit `i` set => lane `i` is a UTF-8 continuation byte `10xxxxxx`. */
    __mmask64 codepoint_starts;  /**< Bit `i` set => lane `i` begins a codepoint (loaded, non-continuation). */
    __mmask64 two_byte_starts;   /**< Bit `i` set => lane `i` is a 2-byte lead `110xxxxx`. */
    __mmask64 three_byte_starts; /**< Bit `i` set => lane `i` is a 3-byte lead `1110xxxx`. */
    __mmask64 four_byte_starts;  /**< Bit `i` set => lane `i` is a 4-byte lead `11110xxx`. */
    sz_size_t loaded;            /**< Number of bytes actually loaded (<= 64) into lanes [0, loaded). */
} sz_utf8_codepoints_window_t;

/** @brief  Load up to 64 bytes from @p text (masked tail) and decode every lane into byte-domain halves. */
SZ_INTERNAL sz_utf8_codepoints_window_t sz_utf8_codepoints_decode_window_( //
    sz_u8_t const *text, sz_size_t available, __m512i lane_identity) {
    sz_utf8_codepoints_window_t result;
    result.loaded = available < 64 ? available : 64;
    __mmask64 const load_mask = sz_u64_clamp_mask_until_(result.loaded);
    __m512i const window = _mm512_maskz_loadu_epi8(load_mask, text);
    result.window = window;

    // The three forward neighbours of each lane, gathered via in-register permutes (never `vpgather`).
    __m512i const next1 = _mm512_permutexvar_epi8(_mm512_add_epi8(lane_identity, _mm512_set1_epi8(1)), window);
    __m512i const next2 = _mm512_permutexvar_epi8(_mm512_add_epi8(lane_identity, _mm512_set1_epi8(2)), window);

    __mmask64 const loaded_lanes = load_mask;
    __mmask64 const is_continuation = _mm512_mask_cmpeq_epi8_mask(
        loaded_lanes, _mm512_and_si512(window, _mm512_set1_epi8((char)0xC0)), _mm512_set1_epi8((char)0x80));
    result.continuation = is_continuation;
    result.codepoint_starts = loaded_lanes & ~is_continuation;
    result.two_byte_starts = _mm512_mask_cmpeq_epi8_mask(
        loaded_lanes, _mm512_and_si512(window, _mm512_set1_epi8((char)0xE0)), _mm512_set1_epi8((char)0xC0));
    result.three_byte_starts = _mm512_mask_cmpeq_epi8_mask(
        loaded_lanes, _mm512_and_si512(window, _mm512_set1_epi8((char)0xF0)), _mm512_set1_epi8((char)0xE0));
    result.four_byte_starts = _mm512_mask_cmpeq_epi8_mask(
        loaded_lanes, _mm512_and_si512(window, _mm512_set1_epi8((char)0xF8)), _mm512_set1_epi8((char)0xF0));

    // 2-byte: cp = ((b0 & 0x1F) << 6) | (b1 & 0x3F); high = cp >> 8, low = cp & 0xFF.
    __m512i const lead_five_bits = _mm512_and_si512(window, _mm512_set1_epi8(0x1F));
    __m512i const next1_six_bits = _mm512_and_si512(next1, _mm512_set1_epi8(0x3F));
    __m512i const high_two_byte = sz_utf8_codepoints_srl8_(lead_five_bits, 2, 0x07);
    __m512i const low_two_byte = _mm512_or_si512(_mm512_slli_epi16(_mm512_and_si512(window, _mm512_set1_epi8(0x03)), 6),
                                                 next1_six_bits);
    // 3-byte: cp = ((b0 & 0x0F) << 12) | ((b1 & 0x3F) << 6) | (b2 & 0x3F).
    __m512i const lead_four_bits = _mm512_and_si512(window, _mm512_set1_epi8(0x0F));
    __m512i const high_three_byte = _mm512_or_si512(_mm512_slli_epi16(lead_four_bits, 4),
                                                    sz_utf8_codepoints_srl8_(next1, 2, 0x0F));
    __m512i const low_three_byte = _mm512_or_si512(
        _mm512_slli_epi16(_mm512_and_si512(next1, _mm512_set1_epi8(0x03)), 6),
        _mm512_and_si512(next2, _mm512_set1_epi8(0x3F)));

    result.high = _mm512_mask_blend_epi8(result.three_byte_starts, high_two_byte, high_three_byte);
    result.low = _mm512_mask_blend_epi8(result.three_byte_starts, low_two_byte, low_three_byte);
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
SZ_INTERNAL __m512i sz_utf8_codepoints_gather_byte_(sz_u8_t const *table, int count, __m512i indices) {
    __m512i const within = _mm512_and_si512(indices, _mm512_set1_epi16(0x7F));
    __m512i const page = _mm512_srli_epi16(indices, 7);
    int const page_count = (count + 127) / 128;
    //  Visit only the distinct 128-byte pages present across the lanes (real text targets ~1-3 of up to ~20), not the
    //  whole table: the present-page set is a `1 << page` OR-reduction; an absent page is bit-exact to skip (its
    //  `mask_mov` selects no lane).
    __m512i const single_bit = _mm512_set1_epi32(1);
    __m512i const page_low_half = _mm512_cvtepu16_epi32(_mm512_castsi512_si256(page));
    __m512i const page_high_half = _mm512_cvtepu16_epi32(_mm512_extracti64x4_epi64(page, 1));
    sz_u32_t present_pages = (sz_u32_t)_mm512_reduce_or_epi32(
        _mm512_or_si512(_mm512_sllv_epi32(single_bit, page_low_half), _mm512_sllv_epi32(single_bit, page_high_half)));
    present_pages &= page_count >= 32 ? 0xFFFFFFFFu : (((sz_u32_t)1 << page_count) - 1u);
    __m512i result = _mm512_setzero_si512();
    while (present_pages) {
        int const page_index = (int)sz_u64_ctz((sz_u64_t)present_pages);
        present_pages &= present_pages - 1;
        int const tile_base = page_index * 128;
        int const tile_remaining = count - tile_base;
        __m512i const tile_lo = _mm512_maskz_loadu_epi8(sz_u64_clamp_mask_until_((sz_size_t)tile_remaining),
                                                        table + tile_base);
        __m512i const tile_hi = tile_remaining > 64 ? _mm512_maskz_loadu_epi8(
                                                          sz_u64_clamp_mask_until_((sz_size_t)(tile_remaining - 64)),
                                                          table + tile_base + 64)
                                                    : _mm512_setzero_si512();
        __m512i const permuted = _mm512_permutex2var_epi8(tile_lo, within, tile_hi);
        __mmask32 const hit = _mm512_cmpeq_epi16_mask(page, _mm512_set1_epi16((short)page_index));
        result = _mm512_mask_mov_epi16(result, hit, _mm512_and_si512(permuted, _mm512_set1_epi16(0x00FF)));
    }
    return result;
}

/**
 *  @brief  Gather one 16-bit word per lane from a register-resident word table @p table of @p count entries,
 *          addressed by the 16-bit @p indices, using a `vpermi2w` page network (NO `vpgather`). Each 64-word
 *          page is two ZMM tiles selected by `vpermi2w` on the low 6 index bits, page chosen by the high bits.
 *          The final partial page is `maskz`-loaded so an unpadded @p table is never over-read.
 */
SZ_INTERNAL __m512i sz_utf8_codepoints_gather_word_(sz_u16_t const *table, int count, __m512i indices) {
    __m512i const within = _mm512_and_si512(indices, _mm512_set1_epi16(0x3F));
    __m512i const page = _mm512_srli_epi16(indices, 6);
    int const page_count = (count + 63) / 64;
    //  Data-dependent page loop (see `sz_utf8_codepoints_gather_byte_`): visit only the distinct 64-word pages
    //  present across the lanes; bit-exact since an absent page contributes no `hit`.
    __m512i const single_bit = _mm512_set1_epi32(1);
    __m512i const page_low_half = _mm512_cvtepu16_epi32(_mm512_castsi512_si256(page));
    __m512i const page_high_half = _mm512_cvtepu16_epi32(_mm512_extracti64x4_epi64(page, 1));
    sz_u32_t present_pages = (sz_u32_t)_mm512_reduce_or_epi32(
        _mm512_or_si512(_mm512_sllv_epi32(single_bit, page_low_half), _mm512_sllv_epi32(single_bit, page_high_half)));
    present_pages &= page_count >= 32 ? 0xFFFFFFFFu : (((sz_u32_t)1 << page_count) - 1u);
    __m512i result = _mm512_setzero_si512();
    while (present_pages) {
        int const page_index = (int)sz_u64_ctz((sz_u64_t)present_pages);
        present_pages &= present_pages - 1;
        int const tile_base = page_index * 64;
        int const tile_remaining = count - tile_base;
        __m512i const tile_lo = _mm512_maskz_loadu_epi16(sz_u32_clamp_mask_until_((sz_size_t)tile_remaining),
                                                         table + tile_base);
        __m512i const tile_hi = tile_remaining > 32 ? _mm512_maskz_loadu_epi16(
                                                          sz_u32_clamp_mask_until_((sz_size_t)(tile_remaining - 32)),
                                                          table + tile_base + 32)
                                                    : _mm512_setzero_si512();
        __m512i const permuted = _mm512_permutex2var_epi16(tile_lo, within, tile_hi);
        __mmask32 const hit = _mm512_cmpeq_epi16_mask(page, _mm512_set1_epi16((short)page_index));
        result = _mm512_mask_mov_epi16(result, hit, permuted);
    }
    return result;
}

/**
 *  @brief  In-register two-stage trie classify of BMP codepoints `0x800..0xFFFF`, replacing the scalar
 *          `while (remaining) { ctz; classify_codepoint }` cold loop (anti-pattern #6/#7). Resolves all 64
 *          lanes uniformly via `vpermi2w` over the L1/L2 word tiles and `vpermi2b` over the leaf byte tiles
 *          held in registers; no `vpgather`, no spill. The trie shape is the canonical L1 (super-block) ->
 *          L2 (block) -> leaf layout:
 *
 *              offset = codepoint - 0x800
 *              block  = offset / block             ; within        = offset % block
 *              super  = block  / superblock        ; super_offset  = block  % superblock
 *              level1 = l1[super]
 *              leaf   = l2[level1 * superblock + super_offset]
 *              class  = leaf_bytes[leaf * block + within]
 *
 *  @param  high        Per-lane `codepoint >> 8` (from @ref sz_utf8_codepoints_decode_window_).
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
SZ_INTERNAL __m512i sz_utf8_codepoints_trie_walk_icelake_( //
    __m512i high, __m512i low,                             //
    sz_u8_t const *l1, sz_u16_t const *l2, sz_u8_t const *leaf, int block, int superblock, int l1_count, int l2_count,
    int leaf_count) {
    __m128i const block_log2 = _mm_cvtsi32_si128(sz_u64_ctz((sz_u64_t)block));
    __m128i const super_log2 = _mm_cvtsi32_si128(sz_u64_ctz((sz_u64_t)superblock));
    __m512i const within_mask = _mm512_set1_epi16((short)(block - 1));
    __m512i const super_off_mask = _mm512_set1_epi16((short)(superblock - 1));
    __m512i const super_v16 = _mm512_set1_epi16((short)superblock);
    __m512i const block_v16 = _mm512_set1_epi16((short)block);

    // Reconstruct cp = (high << 8) | low into two 16-bit halves so the trie math stays exact to 0xFFFF.
    __m512i const zero = _mm512_setzero_si512();
    __m512i const offset_lo = _mm512_sub_epi16(
        _mm512_or_si512(_mm512_slli_epi16(_mm512_unpacklo_epi8(high, zero), 8), _mm512_unpacklo_epi8(low, zero)),
        _mm512_set1_epi16(0x800));
    __m512i const offset_hi = _mm512_sub_epi16(
        _mm512_or_si512(_mm512_slli_epi16(_mm512_unpackhi_epi8(high, zero), 8), _mm512_unpackhi_epi8(low, zero)),
        _mm512_set1_epi16(0x800));

    __m512i const within_lo = _mm512_and_si512(offset_lo, within_mask);
    __m512i const within_hi = _mm512_and_si512(offset_hi, within_mask);
    __m512i const block_idx_lo = _mm512_srl_epi16(offset_lo, block_log2);
    __m512i const block_idx_hi = _mm512_srl_epi16(offset_hi, block_log2);
    __m512i const super_off_lo = _mm512_and_si512(block_idx_lo, super_off_mask);
    __m512i const super_off_hi = _mm512_and_si512(block_idx_hi, super_off_mask);
    __m512i const super_lo = _mm512_srl_epi16(block_idx_lo, super_log2);
    __m512i const super_hi = _mm512_srl_epi16(block_idx_hi, super_log2);

    // Stage 1: level1 = l1[super].
    __m512i const level1_lo = sz_utf8_codepoints_gather_byte_(l1, l1_count, super_lo);
    __m512i const level1_hi = sz_utf8_codepoints_gather_byte_(l1, l1_count, super_hi);

    // Stage 2: leaf = l2[level1 * superblock + super_offset].
    __m512i const l2_index_lo = _mm512_add_epi16(_mm512_mullo_epi16(level1_lo, super_v16), super_off_lo);
    __m512i const l2_index_hi = _mm512_add_epi16(_mm512_mullo_epi16(level1_hi, super_v16), super_off_hi);
    __m512i const leaf_idx_lo = sz_utf8_codepoints_gather_word_(l2, l2_count, l2_index_lo);
    __m512i const leaf_idx_hi = sz_utf8_codepoints_gather_word_(l2, l2_count, l2_index_hi);

    // Leaf: class = leaf_bytes[leaf * block + within].
    __m512i const leaf_byte_lo = _mm512_add_epi16(_mm512_mullo_epi16(leaf_idx_lo, block_v16), within_lo);
    __m512i const leaf_byte_hi = _mm512_add_epi16(_mm512_mullo_epi16(leaf_idx_hi, block_v16), within_hi);
    __m512i const class_lo = sz_utf8_codepoints_gather_byte_(leaf, leaf_count, leaf_byte_lo);
    __m512i const class_hi = sz_utf8_codepoints_gather_byte_(leaf, leaf_count, leaf_byte_hi);

    // Re-pack the two 16-bit class halves (each already in [0,255]) into the original byte lane order.
    return _mm512_packus_epi16(class_lo, class_hi);
}

/**
 *  @brief  256-entry byte LUT read over a 64-byte-aligned 256-byte @p table, per 32-bit lane @p index in [0,256). A
 *          `vpermb` over the four resident quads; the high two bits of the index select the quad via masked blends.
 *          Tiles load directly from `.rodata` (no per-call materialization), so the family classifiers stay
 *          re-init-free. @p table must be `sz_align_(64)` and exactly 256 bytes.
 */
SZ_INTERNAL __m512i sz_utf8_codepoints_permute256_icelake_(sz_u8_t const *table, __m512i index) {
    __m512i const quad0 = _mm512_load_si512((void const *)(table + 0 * 64));
    __m512i const quad1 = _mm512_load_si512((void const *)(table + 1 * 64));
    __m512i const quad2 = _mm512_load_si512((void const *)(table + 2 * 64));
    __m512i const quad3 = _mm512_load_si512((void const *)(table + 3 * 64));
    __m512i const low_six = _mm512_and_si512(index, _mm512_set1_epi32(0x3F));
    __m512i const top_two = _mm512_and_si512(index, _mm512_set1_epi32(0xC0));
    __m512i const permuted0 = _mm512_and_si512(_mm512_permutexvar_epi8(low_six, quad0), _mm512_set1_epi32(0xFF));
    __m512i const permuted1 = _mm512_and_si512(_mm512_permutexvar_epi8(low_six, quad1), _mm512_set1_epi32(0xFF));
    __m512i const permuted2 = _mm512_and_si512(_mm512_permutexvar_epi8(low_six, quad2), _mm512_set1_epi32(0xFF));
    __m512i const permuted3 = _mm512_and_si512(_mm512_permutexvar_epi8(low_six, quad3), _mm512_set1_epi32(0xFF));
    __m512i result = permuted0;
    result = _mm512_mask_blend_epi32(_mm512_cmpeq_epi32_mask(top_two, _mm512_set1_epi32(0x40)), result, permuted1);
    result = _mm512_mask_blend_epi32(_mm512_cmpeq_epi32_mask(top_two, _mm512_set1_epi32(0x80)), result, permuted2);
    result = _mm512_mask_blend_epi32(_mm512_cmpeq_epi32_mask(top_two, _mm512_set1_epi32(0xC0)), result, permuted3);
    return result;
}

/**
 *  @brief  Gather-free indexed read of a 64-byte-aligned byte LUT spanning @p tile_count tiles of 64 bytes, with a
 *          per-32-bit-lane byte @p index_dwords in [0, tile_count*64). A `vpermi2b` cascade: each ZMM pair covers 128
 *          byte slots addressed by the low 7 bits; the high bits select the pair via masked blends. Tiles load
 *          directly from the aligned `.rodata` @p table — no `luts` struct, no per-call init. No `vpgatherdd`.
 *          @p table must be `sz_align_(64)` and zero-padded to `tile_count * 64` bytes.
 */
SZ_INTERNAL __m512i sz_utf8_codepoints_lut_cascade_icelake_(sz_u8_t const *table, int tile_count,
                                                            __m512i index_dwords) {
    __m512i const within = _mm512_and_si512(index_dwords, _mm512_set1_epi32(0x7F));
    __m512i const selector = _mm512_srli_epi32(index_dwords, 7);
    __m512i result = _mm512_setzero_si512();
    int const pairs = (tile_count + 1) / 2;
    for (int pair = 0; pair < pairs; ++pair) {
        __m512i const low_tile = _mm512_load_si512((void const *)(table + (pair * 2) * 64));
        __m512i const high_tile = (pair * 2 + 1 < tile_count)
                                      ? _mm512_load_si512((void const *)(table + (pair * 2 + 1) * 64))
                                      : _mm512_setzero_si512();
        __m512i const picked = _mm512_permutex2var_epi8(low_tile, within, high_tile);
        __mmask16 const here = _mm512_cmpeq_epi32_mask(selector, _mm512_set1_epi32(pair));
        result = _mm512_mask_blend_epi32(here, result, _mm512_and_si512(picked, _mm512_set1_epi32(0xFF)));
    }
    return result;
}

/**
 *  @brief  Indexed read of a 4-bit-per-cell LUT: @p packed holds two output nibbles per byte (cell `i` is the low
 *          nibble of `packed[i/2]` for even `i`, the high nibble for odd `i`). Halves the table and so HALVES the
 *          `vpermi2b` cascade depth vs a byte-per-cell layout, for tables whose outputs fit in 4 bits (e.g. the
 *          grapheme `stage_sub` descriptor index, the word `astral_leaf` class). @p tile_count counts the packed
 *          tiles; @p index_dwords is the unpacked cell index per 32-bit lane. Reads straight from aligned `.rodata`.
 */
SZ_INTERNAL __m512i sz_utf8_codepoints_lut_cascade_nibble_icelake_(sz_u8_t const *packed, int tile_count,
                                                                   __m512i index_dwords) {
    __m512i const byte_index = _mm512_srli_epi32(index_dwords, 1);
    __m512i const packed_byte = sz_utf8_codepoints_lut_cascade_icelake_(packed, tile_count, byte_index);
    __mmask16 const odd_cell = _mm512_test_epi32_mask(index_dwords, _mm512_set1_epi32(1));
    __m512i const low_nibble = _mm512_and_si512(packed_byte, _mm512_set1_epi32(0x0F));
    __m512i const high_nibble = _mm512_and_si512(_mm512_srli_epi32(packed_byte, 4), _mm512_set1_epi32(0x0F));
    return _mm512_mask_blend_epi32(odd_cell, low_nibble, high_nibble);
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
SZ_INTERNAL sz_size_t sz_utf8_codepoints_drain_forward_( //
    sz_u64_t boundary, sz_size_t base, __m512i lane_identity, sz_size_t *starts, sz_size_t *lengths, sz_size_t produced,
    sz_size_t capacity, sz_size_t *previous_io) {
    __m512i const wave_shift = _mm512_add_epi8(lane_identity, _mm512_set1_epi8(8));
    sz_size_t const boundary_count = (sz_size_t)_mm_popcnt_u64(boundary);
    __m512i packed = _mm512_maskz_compress_epi8(_cvtu64_mask64(boundary), lane_identity);
    sz_u64_t previous = (sz_u64_t)*previous_io;
    sz_size_t emitted = 0;
    while (emitted < boundary_count && produced < capacity) {
        sz_size_t const wave = sz_min_of_three(boundary_count - emitted, capacity - produced, 8);
        __m512i const positions = _mm512_add_epi64(_mm512_cvtepu8_epi64(_mm512_castsi512_si128(packed)),
                                                   _mm512_set1_epi64((long long)base));
        __m512i const segment_starts = _mm512_alignr_epi64(positions, _mm512_set1_epi64((long long)previous), 7);
        __mmask8 const store_mask = sz_u8_clamp_mask_until_(wave);
        _mm512_mask_storeu_epi64((void *)(starts + produced), store_mask, segment_starts);
        _mm512_mask_storeu_epi64((void *)(lengths + produced), store_mask, _mm512_sub_epi64(positions, segment_starts));
        produced += wave, emitted += wave;
        previous = (sz_u64_t)(starts[produced - 1] + lengths[produced - 1]);
        packed = _mm512_permutexvar_epi8(wave_shift, packed);
    }
    *previous_io = (sz_size_t)previous;
    return produced;
}

/**
 *  @brief  Emit boundary-lane offsets in DESCENDING byte order via `vpcompressb`, carrying the open segment's end
 *          position across waves and windows. Mirror of @ref sz_utf8_codepoints_drain_forward_ for reverse drivers.
 *  @param  lane_identity_descending  The {0..63} -> {63..0} reversed lane identity, reused across calls.
 */
SZ_INTERNAL sz_size_t sz_utf8_codepoints_drain_backward_( //
    sz_u64_t boundary, sz_size_t base, __m512i lane_identity, __m512i lane_identity_descending, sz_size_t *starts,
    sz_size_t *lengths, sz_size_t produced, sz_size_t capacity, sz_size_t *end_io) {
    sz_size_t const boundary_count = (sz_size_t)_mm_popcnt_u64(boundary);
    __m512i packed = _mm512_maskz_compress_epi8(sz_u64_bits_reverse(boundary), lane_identity_descending);
    sz_u64_t segment_end = (sz_u64_t)*end_io;
    sz_size_t emitted = 0;
    __m512i const wave_shift = _mm512_add_epi8(lane_identity, _mm512_set1_epi8(8));
    while (emitted < boundary_count && produced < capacity) {
        sz_size_t const wave = sz_min_of_three(boundary_count - emitted, capacity - produced, 8);
        __m512i const positions = _mm512_add_epi64(_mm512_cvtepu8_epi64(_mm512_castsi512_si128(packed)),
                                                   _mm512_set1_epi64((long long)base));
        __m512i const ends = _mm512_alignr_epi64(positions, _mm512_set1_epi64((long long)segment_end), 7);
        __mmask8 const store_mask = sz_u8_clamp_mask_until_(wave);
        _mm512_mask_storeu_epi64((void *)(starts + produced), store_mask, positions);
        _mm512_mask_storeu_epi64((void *)(lengths + produced), store_mask, _mm512_sub_epi64(ends, positions));
        produced += wave, emitted += wave;
        segment_end = (sz_u64_t)starts[produced - 1];
        packed = _mm512_permutexvar_epi8(wave_shift, packed);
    }
    *end_io = (sz_size_t)segment_end;
    return produced;
}

#pragma endregion Drains

#pragma endregion Shared SIMD leaf substrate

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
        sz_size_t two_byte_count = (sz_size_t)_mm_popcnt_u64(
            _cvtmask16_u32(_kand_mask16(is_two_byte_char, sz_u16_mask_until_(runes_to_unpack))));
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
#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#endif // SZ_USE_ICELAKE

#ifdef __cplusplus
}
#endif

#endif // STRINGZILLA_UTF8_CODEPOINTS_ICELAKE_H_
