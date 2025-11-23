/**
 *  @brief  Hardware-accelerated UTF-8 text processing utilities.
 *  @file   utf8.h
 *  @author Ash Vardanian

 */
#ifndef STRINGZILLA_UTF8_H_
#define STRINGZILLA_UTF8_H_

#include "types.h"

#ifdef __cplusplus
extern "C" {
#endif

SZ_PUBLIC sz_cptr_t sz_utf8_unpack_chunk_ice(   //
    sz_cptr_t text, sz_size_t length,           //
    sz_rune_t *runes, sz_size_t runes_capacity, //
    sz_size_t *runes_unpacked) {

    // Process up to the minimum of: available bytes, output capacity * 4, or optimal chunk size (64)
    sz_size_t chunk_size = sz_min_of_three(length, runes_capacity * 4, 64);
    sz_u512_vec_t text_vec, runes_vec;
    __mmask64 load_mask = sz_u64_mask_until_(chunk_size);
    text_vec.zmm = _mm512_maskz_loadu_epi8(load_mask, (sz_u8_t const *)text);

    // Check, how many of the next characters are single byte (ASCII) codepoints
    // ASCII bytes have bit 7 clear (0x00-0x7F), non-ASCII have bit 7 set (0x80-0xFF)
    __mmask64 non_ascii_mask = _mm512_movepi8_mask(text_vec.zmm);
    // Find first non-ASCII byte or end of loaded data
    sz_size_t ascii_prefix_length = sz_u64_ctz(non_ascii_mask | ~load_mask);

    if (ascii_prefix_length) {
        // Unpack the last 16 bytes of text into the next 16 runes.
        // Even if we have more than 16 ASCII characters, we don't want to overcomplicate control flow here.
        sz_size_t runes_to_place = sz_min_of_three(ascii_prefix_length, 16, runes_capacity);
        runes_vec.zmm = _mm512_cvtepu8_epi32(_mm512_castsi512_si128(text_vec.zmm));
        _mm512_mask_storeu_epi32(runes, sz_u16_mask_until_(runes_to_place), runes_vec.zmm);
        *runes_unpacked = runes_to_place;
        return text + runes_to_place;
    }

    // Check for the number of 2-byte characters
    // 2-byte UTF-8: [lead, cont] where lead=110xxxxx (0xC0-0xDF), cont=10xxxxxx (0x80-0xBF)
    // In 16-bit little-endian: 0xCCLL where LL=lead, CC=cont
    // Mask: 0xC0E0 (cont & 0xC0, lead & 0xE0), Pattern: 0x80C0 (cont=0x80, lead=0xC0)
    __mmask32 non_two_byte_mask =
        _mm512_cmpneq_epi16_mask(_mm512_and_si512(text_vec.zmm, _mm512_set1_epi16(0xC0E0)), _mm512_set1_epi16(0x80C0));
    sz_size_t two_byte_prefix_length = sz_u64_ctz(non_two_byte_mask);
    if (two_byte_prefix_length) {
        // Unpack the last 32 bytes of text into the next 32 runes.
        // Even if we have more than 32 two-byte characters, we don't want to overcomplicate control flow here.
        sz_size_t runes_to_place = sz_min_of_three(two_byte_prefix_length, 32, runes_capacity);
        runes_vec.zmm = _mm512_cvtepu16_epi32(_mm512_castsi512_si256(text_vec.zmm));
        // Decode 2-byte UTF-8: ((lead & 0x1F) << 6) | (cont & 0x3F)
        // After cvtepu16_epi32: value = 0x0000CCLL where LL=lead (bits 7-0), CC=cont (bits 15-8)
        runes_vec.zmm = _mm512_or_si512(                                                      //
            _mm512_slli_epi32(_mm512_and_si512(runes_vec.zmm, _mm512_set1_epi32(0x1FU)), 6),  // (lead & 0x1F) << 6
            _mm512_and_si512(_mm512_srli_epi32(runes_vec.zmm, 8), _mm512_set1_epi32(0x3FU))); // (cont & 0x3F)
        _mm512_mask_storeu_epi32(runes, sz_u32_mask_until_(runes_to_place), runes_vec.zmm);
        *runes_unpacked = runes_to_place;
        return text + runes_to_place * 2;
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
    four_byte_mask_vec.zmm = _mm512_set1_epi32(0xC0C0C0F8);    // Mask: [F8, C0, C0, C0] per 4-byte slot
    four_byte_pattern_vec.zmm = _mm512_set1_epi32(0x808080F0); // Pattern: [F0, 80, 80, 80] per 4-byte slot

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

    // Seems like broken unicoode?
    *runes_unpacked = 0;
    return text;
}

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#endif            // SZ_USE_ICE
#pragma endregion // Ice Lake Implementation

#pragma region Haswell Implementation
#if SZ_USE_HASWELL
#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("avx2,bmi,bmi2,popcnt"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("avx2,bmi,bmi2,popcnt")
#endif

SZ_PUBLIC sz_cptr_t sz_utf8_unpack_chunk_haswell( //
    sz_cptr_t text, sz_size_t length,             //
    sz_rune_t *runes, sz_size_t runes_capacity,   //
    sz_size_t *runes_unpacked) {
    // Fallback to serial implementation for now
    // A future optimization could use AVX2 for decoding
    return sz_utf8_unpack_chunk_serial(text, length, runes, runes_capacity, runes_unpacked);
}

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#endif            // SZ_USE_HASWELL
#pragma endregion // Haswell Implementation

#pragma region NEON Implementation
#if SZ_USE_NEON
#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("+simd"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("+simd")
#endif

SZ_PUBLIC sz_cptr_t sz_utf8_unpack_chunk_neon(  //
    sz_cptr_t text, sz_size_t length,           //
    sz_rune_t *runes, sz_size_t runes_capacity, //
    sz_size_t *runes_unpacked) {
    // TODO: Implement a fast NEON version once we come up with an AVX-512 design.
    return sz_utf8_unpack_chunk_serial(text, length, runes, runes_capacity, runes_unpacked);
}

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#endif // SZ_USE_NEON

#pragma endregion // NEON Implementation

#ifdef __cplusplus
}
#endif

#endif // STRINGZILLA_UTF8_H_
