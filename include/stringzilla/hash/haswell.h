/**
 *  @brief Haswell (AVX2) backend for string hashing and checksums.
 *  @file include/stringzilla/hash/haswell.h
 *  @author Ash Vardanian
 *  @sa include/stringzilla/hash.h
 */
#ifndef STRINGZILLA_HASH_HASWELL_H_
#define STRINGZILLA_HASH_HASWELL_H_

#include "stringzilla/types.h"
#include "stringzilla/compare.h" // `sz_equal`
#include "stringzilla/hash/serial.h"

#ifdef __cplusplus
extern "C" {
#endif

#if SZ_USE_HASWELL
#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("avx2"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("avx2")
#endif

SZ_API_COMPTIME sz_u64_t sz_bytesum_haswell(sz_cptr_t text, sz_size_t length) {
    // The naive implementation of this function is very simple.
    // It assumes the CPU is great at handling unaligned "loads".
    //
    // A typical AWS Skylake instance can have 32 KB x 2 blocks of L1 data cache per core,
    // 1 MB x 2 blocks of L2 cache per core, and one shared L3 cache buffer.
    // For now, let's avoid the cases beyond the L2 size.
    int is_huge = length > 1ull * 1024ull * 1024ull;

    // When the buffer is small, there isn't much to innovate.
    if (length <= 32) { return sz_bytesum_serial(text, length); }
    else if (!is_huge) {
        sz_u256_vec_t text_vec, sums_vec;
        sums_vec.ymm = _mm256_setzero_si256();
        for (; length >= 32; text += 32, length -= 32) {
            text_vec.ymm = _mm256_lddqu_si256((__m256i const *)text);
            sums_vec.ymm = _mm256_add_epi64(sums_vec.ymm, _mm256_sad_epu8(text_vec.ymm, _mm256_setzero_si256()));
        }
        // We can also avoid the final serial loop by fetching 32 bytes from end, in reverse direction,
        // and shifting the data within the register to zero-out the duplicate bytes.

        // Accumulating 256 bits is harder, as we need to extract the 128-bit sums first.
        __m128i low_u64x2 = _mm256_castsi256_si128(sums_vec.ymm);
        __m128i high_u64x2 = _mm256_extracti128_si256(sums_vec.ymm, 1);
        __m128i sums_u64x2 = _mm_add_epi64(low_u64x2, high_u64x2);
        sz_u64_t low = (sz_u64_t)_mm_cvtsi128_si64(sums_u64x2);
        sz_u64_t high = (sz_u64_t)_mm_extract_epi64(sums_u64x2, 1);
        sz_u64_t result = low + high;
        if (length) result += sz_bytesum_serial(text, length);
        return result;
    }
    // For gigantic buffers, exceeding typical L1 cache sizes, there are other tricks we can use.
    // Most notably, we can avoid populating the cache with the entire buffer, and instead traverse it in 2 directions.
    else {
        sz_size_t head_length = (32 - ((sz_size_t)text % 32)) % 32; // 31 or less.
        sz_size_t tail_length = (sz_size_t)(text + length) % 32;    // 31 or less.
        sz_size_t body_length = length - head_length - tail_length; // Multiple of 32.
        sz_u64_t result = 0;

        // Handle the tail before we start updating the `text` pointer. Read as `sz_u8_t`: a plain `char` is
        // signed on x86, so a byte >= 0x80 would sign-extend to a negative `int` and wrongly subtract from the
        // running sum (the SIMD body already accumulates unsigned via `_mm256_sad_epu8`).
        while (tail_length) result += (sz_u8_t)text[length - (tail_length--)];
        // Handle the head
        while (head_length--) result += (sz_u8_t)*text++;

        // This branch is reached only when `is_huge` (the `!is_huge` body above handled the rest), so the
        // buffer is traversed in two directions at once - touching both ends keeps neither cache-resident and
        // avoids evicting the whole buffer through L1.
        sz_u256_vec_t text_vec, sums_vec;
        sz_u256_vec_t text_reversed_vec, sums_reversed_vec;
        sums_vec.ymm = _mm256_setzero_si256();
        sums_reversed_vec.ymm = _mm256_setzero_si256();
        for (; body_length >= 64; text += 32, body_length -= 64) {
            text_vec.ymm = _mm256_stream_load_si256((__m256i *)(text));
            sums_vec.ymm = _mm256_add_epi64(sums_vec.ymm, _mm256_sad_epu8(text_vec.ymm, _mm256_setzero_si256()));
            text_reversed_vec.ymm = _mm256_stream_load_si256((__m256i *)(text + body_length - 32));
            sums_reversed_vec.ymm = _mm256_add_epi64(sums_reversed_vec.ymm,
                                                     _mm256_sad_epu8(text_reversed_vec.ymm, _mm256_setzero_si256()));
        }
        if (body_length >= 32) {
            sz_assert_(body_length == 32);
            text_vec.ymm = _mm256_stream_load_si256((__m256i *)(text));
            sums_vec.ymm = _mm256_add_epi64(sums_vec.ymm, _mm256_sad_epu8(text_vec.ymm, _mm256_setzero_si256()));
            text += 32;
        }
        sums_vec.ymm = _mm256_add_epi64(sums_vec.ymm, sums_reversed_vec.ymm);

        // Accumulating 256 bits is harder, as we need to extract the 128-bit sums first.
        __m128i low_u64x2 = _mm256_castsi256_si128(sums_vec.ymm);
        __m128i high_u64x2 = _mm256_extracti128_si256(sums_vec.ymm, 1);
        __m128i sums_u64x2 = _mm_add_epi64(low_u64x2, high_u64x2);
        sz_u64_t low = (sz_u64_t)_mm_cvtsi128_si64(sums_u64x2);
        sz_u64_t high = (sz_u64_t)_mm_extract_epi64(sums_u64x2, 1);
        result += low + high;
        return result;
    }
}

/*  Without `vpternlogd` every bitwise primitive expands, and without `vprord` every rotation becomes a shift
 *  pair and an or. Both are cheaper than the instruction counts suggest: logic issues on all three vector
 *  ports on this class of core, so the round mix stays throughput-bound rather than shift-bound. */

/** @brief Evaluates `(state_e & state_f) ^ (~state_e & state_g)` across 8 lanes. */
SZ_HELPER_INLINE __m256i sz_sha256_choice_haswell_(__m256i state_e_u32x8, __m256i state_f_u32x8,
                                                   __m256i state_g_u32x8) {
    return _mm256_xor_si256(state_g_u32x8,
                            _mm256_and_si256(state_e_u32x8, _mm256_xor_si256(state_f_u32x8, state_g_u32x8)));
}

/** @brief Evaluates `(state_a & state_b) ^ (state_a & state_c) ^ (state_b & state_c)` across 8 lanes. */
SZ_HELPER_INLINE __m256i sz_sha256_majority_haswell_(__m256i state_a_u32x8, __m256i state_b_u32x8,
                                                     __m256i state_c_u32x8) {
    return _mm256_xor_si256(_mm256_and_si256(_mm256_xor_si256(state_a_u32x8, state_b_u32x8), state_c_u32x8),
                            _mm256_and_si256(state_a_u32x8, state_b_u32x8));
}

/** @brief Evaluates `ror(state_a_u32x8, 2) ^ ror(state_a_u32x8, 13) ^ ror(state_a_u32x8, 22)` across 8 lanes. */
SZ_HELPER_INLINE __m256i sz_sha256_big_sigma0_haswell_(__m256i state_a_u32x8) {
    __m256i const rotated_by_2_u32x8 = _mm256_or_si256(_mm256_srli_epi32(state_a_u32x8, 2),
                                                       _mm256_slli_epi32(state_a_u32x8, 30));
    __m256i const rotated_by_13_u32x8 = _mm256_or_si256(_mm256_srli_epi32(state_a_u32x8, 13),
                                                        _mm256_slli_epi32(state_a_u32x8, 19));
    __m256i const rotated_by_22_u32x8 = _mm256_or_si256(_mm256_srli_epi32(state_a_u32x8, 22),
                                                        _mm256_slli_epi32(state_a_u32x8, 10));
    return _mm256_xor_si256(_mm256_xor_si256(rotated_by_2_u32x8, rotated_by_13_u32x8), rotated_by_22_u32x8);
}

/** @brief Evaluates `ror(state_e_u32x8, 6) ^ ror(state_e_u32x8, 11) ^ ror(state_e_u32x8, 25)` across 8 lanes. */
SZ_HELPER_INLINE __m256i sz_sha256_big_sigma1_haswell_(__m256i state_e_u32x8) {
    __m256i const rotated_by_6_u32x8 = _mm256_or_si256(_mm256_srli_epi32(state_e_u32x8, 6),
                                                       _mm256_slli_epi32(state_e_u32x8, 26));
    __m256i const rotated_by_11_u32x8 = _mm256_or_si256(_mm256_srli_epi32(state_e_u32x8, 11),
                                                        _mm256_slli_epi32(state_e_u32x8, 21));
    __m256i const rotated_by_25_u32x8 = _mm256_or_si256(_mm256_srli_epi32(state_e_u32x8, 25),
                                                        _mm256_slli_epi32(state_e_u32x8, 7));
    return _mm256_xor_si256(_mm256_xor_si256(rotated_by_6_u32x8, rotated_by_11_u32x8), rotated_by_25_u32x8);
}

/** @brief Evaluates `ror(word, 7) ^ ror(word, 18) ^ (word >> 3)` across 8 lanes. */
SZ_HELPER_INLINE __m256i sz_sha256_small_sigma0_haswell_(__m256i message_word_u32x8) {
    __m256i const rotated_by_7_u32x8 = _mm256_or_si256(_mm256_srli_epi32(message_word_u32x8, 7),
                                                       _mm256_slli_epi32(message_word_u32x8, 25));
    __m256i const rotated_by_18_u32x8 = _mm256_or_si256(_mm256_srli_epi32(message_word_u32x8, 18),
                                                        _mm256_slli_epi32(message_word_u32x8, 14));
    return _mm256_xor_si256(_mm256_xor_si256(rotated_by_7_u32x8, rotated_by_18_u32x8),
                            _mm256_srli_epi32(message_word_u32x8, 3));
}

/** @brief Evaluates `ror(word, 17) ^ ror(word, 19) ^ (word >> 10)` across 8 lanes. */
SZ_HELPER_INLINE __m256i sz_sha256_small_sigma1_haswell_(__m256i message_word_u32x8) {
    __m256i const rotated_by_17_u32x8 = _mm256_or_si256(_mm256_srli_epi32(message_word_u32x8, 17),
                                                        _mm256_slli_epi32(message_word_u32x8, 15));
    __m256i const rotated_by_19_u32x8 = _mm256_or_si256(_mm256_srli_epi32(message_word_u32x8, 19),
                                                        _mm256_slli_epi32(message_word_u32x8, 13));
    return _mm256_xor_si256(_mm256_xor_si256(rotated_by_17_u32x8, rotated_by_19_u32x8),
                            _mm256_srli_epi32(message_word_u32x8, 10));
}

/**
 *  @brief Transposes eight 32-byte halves into word-major big-endian order.
 *  @param lanes_u32x8 The eight per-lane halves, byte-swapped on entry.
 *  @param words_u32x8 Receives eight registers, where `words_u32x8[index]` holds that word from all 8 lanes.
 *
 *  Two interleave stages gather 32-bit and then 64-bit neighbours inside each 128-bit half, and a final
 *  cross-half permute completes the exchange. Transposition is its own inverse, so this also carries the
 *  hash state the other way, from word-major registers back to one 32-byte lane each.
 */
SZ_HELPER_INLINE void sz_sha256_transpose_8x8_haswell_(__m256i const lanes_u32x8[8], __m256i words_u32x8[8]) {
    __m256i const paired0_u32x8 = _mm256_unpacklo_epi32(lanes_u32x8[0], lanes_u32x8[1]);
    __m256i const paired1_u32x8 = _mm256_unpackhi_epi32(lanes_u32x8[0], lanes_u32x8[1]);
    __m256i const paired2_u32x8 = _mm256_unpacklo_epi32(lanes_u32x8[2], lanes_u32x8[3]);
    __m256i const paired3_u32x8 = _mm256_unpackhi_epi32(lanes_u32x8[2], lanes_u32x8[3]);
    __m256i const paired4_u32x8 = _mm256_unpacklo_epi32(lanes_u32x8[4], lanes_u32x8[5]);
    __m256i const paired5_u32x8 = _mm256_unpackhi_epi32(lanes_u32x8[4], lanes_u32x8[5]);
    __m256i const paired6_u32x8 = _mm256_unpacklo_epi32(lanes_u32x8[6], lanes_u32x8[7]);
    __m256i const paired7_u32x8 = _mm256_unpackhi_epi32(lanes_u32x8[6], lanes_u32x8[7]);

    __m256i const quadded0_u32x8 = _mm256_unpacklo_epi64(paired0_u32x8, paired2_u32x8);
    __m256i const quadded1_u32x8 = _mm256_unpackhi_epi64(paired0_u32x8, paired2_u32x8);
    __m256i const quadded2_u32x8 = _mm256_unpacklo_epi64(paired1_u32x8, paired3_u32x8);
    __m256i const quadded3_u32x8 = _mm256_unpackhi_epi64(paired1_u32x8, paired3_u32x8);
    __m256i const quadded4_u32x8 = _mm256_unpacklo_epi64(paired4_u32x8, paired6_u32x8);
    __m256i const quadded5_u32x8 = _mm256_unpackhi_epi64(paired4_u32x8, paired6_u32x8);
    __m256i const quadded6_u32x8 = _mm256_unpacklo_epi64(paired5_u32x8, paired7_u32x8);
    __m256i const quadded7_u32x8 = _mm256_unpackhi_epi64(paired5_u32x8, paired7_u32x8);

    words_u32x8[0] = _mm256_permute2x128_si256(quadded0_u32x8, quadded4_u32x8, 0x20);
    words_u32x8[4] = _mm256_permute2x128_si256(quadded0_u32x8, quadded4_u32x8, 0x31);
    words_u32x8[1] = _mm256_permute2x128_si256(quadded1_u32x8, quadded5_u32x8, 0x20);
    words_u32x8[5] = _mm256_permute2x128_si256(quadded1_u32x8, quadded5_u32x8, 0x31);
    words_u32x8[2] = _mm256_permute2x128_si256(quadded2_u32x8, quadded6_u32x8, 0x20);
    words_u32x8[6] = _mm256_permute2x128_si256(quadded2_u32x8, quadded6_u32x8, 0x31);
    words_u32x8[3] = _mm256_permute2x128_si256(quadded3_u32x8, quadded7_u32x8, 0x20);
    words_u32x8[7] = _mm256_permute2x128_si256(quadded3_u32x8, quadded7_u32x8, 0x31);
}

/**
 *  @brief Extends the rolling message window by one word across 8 lanes.
 *  @param oldest_word_u32x8 The word being replaced, sixteen rounds behind.
 *  @param next_word_u32x8 The word one position ahead, feeding the low sigma.
 *  @param ninth_word_u32x8 The word nine positions ahead.
 *  @param fourteenth_word_u32x8 The word fourteen positions ahead, feeding the high sigma.
 */
SZ_HELPER_INLINE __m256i sz_sha256_extend_haswell_(__m256i oldest_word_u32x8, __m256i next_word_u32x8,
                                                   __m256i ninth_word_u32x8, __m256i fourteenth_word_u32x8) {
    return _mm256_add_epi32(_mm256_add_epi32(oldest_word_u32x8, sz_sha256_small_sigma0_haswell_(next_word_u32x8)),
                            _mm256_add_epi32(ninth_word_u32x8, sz_sha256_small_sigma1_haswell_(fourteenth_word_u32x8)));
}

/**
 *  @brief Applies one SHA256 round to 8 lanes, writing the two working values that change.
 *
 *  The eight working variables rotate by one position per round, which the callers express by passing the
 *  same locals in a different order rather than by moving data. Only `state_d` and `state_h` are written:
 *  `state_d` becomes the next round's `state_e`, and `state_h` is dead on entry so it receives the next
 *  round's `state_a`.
 */
SZ_HELPER_INLINE void sz_sha256_round_haswell_(                                                  //
    __m256i state_a_u32x8, __m256i state_b_u32x8, __m256i state_c_u32x8, __m256i *state_d_u32x8, //
    __m256i state_e_u32x8, __m256i state_f_u32x8, __m256i state_g_u32x8, __m256i *state_h_u32x8, //
    __m256i message_word_u32x8, sz_u32_t round_constant) {
    __m256i const temporary_first_u32x8 = _mm256_add_epi32( //
        _mm256_add_epi32(_mm256_add_epi32(*state_h_u32x8, sz_sha256_big_sigma1_haswell_(state_e_u32x8)),
                         _mm256_add_epi32(sz_sha256_choice_haswell_(state_e_u32x8, state_f_u32x8, state_g_u32x8),
                                          message_word_u32x8)),
        _mm256_set1_epi32((int)round_constant));
    __m256i const temporary_second_u32x8 = _mm256_add_epi32(
        sz_sha256_big_sigma0_haswell_(state_a_u32x8),
        sz_sha256_majority_haswell_(state_a_u32x8, state_b_u32x8, state_c_u32x8));
    *state_d_u32x8 = _mm256_add_epi32(*state_d_u32x8, temporary_first_u32x8);
    *state_h_u32x8 = _mm256_add_epi32(temporary_first_u32x8, temporary_second_u32x8);
}

/**
 *  @brief Transposes and compresses one 64-byte block for each of 8 lanes into word-major hash registers.
 *  @param hashes Eight registers of word-major hash state, updated in place.
 *  @param lane_blocks Pointers to 8 message blocks, one per lane.
 *  @param active_u32x8 All-ones per lane whose block counts; the rest keep their hash bit-for-bit.
 *
 *  Masking rides on the closing accumulation rather than on a saved copy and a blend. Compression ends in
 *  `hash += working`, so clearing the addend for a lane is all it needs to sit out, which costs one `vpand`
 *  per word and one live mask register at a point where the state and the message window are already
 *  competing for sixteen. AVX2 has no write mask, hence the gated addend rather than a masked add. The
 *  rounds still run for an inactive lane and produce a value nobody reads, so its block pointer only has to
 *  be readable - `sz_sha256_state_t::block` is always 64 valid bytes and serves that purpose.
 *
 *  A 64-byte block is two 256-bit registers per lane, so the sixteen message words come from two independent
 *  eight-by-eight transposes. The window lives in a local rather than a parameter so it is not forced to
 *  memory across a call boundary.
 */
SZ_HELPER_INLINE void sz_sha256_compress_haswell_(__m256i hashes_u32x8[8], sz_u8_t const *const *lane_blocks,
                                                  __m256i active_u32x8) {
    sz_u32_t const *round_constants = (sz_u32_t const *)sz_x86_hide_pointer_origin_(sz_sha256_round_constants_());

    __m256i const byte_swap_u8x32 = _mm256_setr_epi8(3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12, //
                                                     3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12);
    __m256i low_lanes_u32x8[8], high_lanes_u32x8[8], schedule_u32x8[16];
    for (sz_size_t lane_index = 0; lane_index != 8; ++lane_index) {
        low_lanes_u32x8[lane_index] = _mm256_shuffle_epi8(
            _mm256_loadu_si256((__m256i const *)&lane_blocks[lane_index][0]), byte_swap_u8x32);
        high_lanes_u32x8[lane_index] = _mm256_shuffle_epi8(
            _mm256_loadu_si256((__m256i const *)&lane_blocks[lane_index][32]), byte_swap_u8x32);
    }
    sz_sha256_transpose_8x8_haswell_(low_lanes_u32x8, &schedule_u32x8[0]);
    sz_sha256_transpose_8x8_haswell_(high_lanes_u32x8, &schedule_u32x8[8]);

    __m256i state_a_u32x8 = hashes_u32x8[0], state_b_u32x8 = hashes_u32x8[1];
    __m256i state_c_u32x8 = hashes_u32x8[2], state_d_u32x8 = hashes_u32x8[3];
    __m256i state_e_u32x8 = hashes_u32x8[4], state_f_u32x8 = hashes_u32x8[5];
    __m256i state_g_u32x8 = hashes_u32x8[6], state_h_u32x8 = hashes_u32x8[7];

    sz_sha256_round_haswell_(state_a_u32x8, state_b_u32x8, state_c_u32x8, &state_d_u32x8, state_e_u32x8, state_f_u32x8,
                             state_g_u32x8, &state_h_u32x8, schedule_u32x8[0], round_constants[0]);
    sz_sha256_round_haswell_(state_h_u32x8, state_a_u32x8, state_b_u32x8, &state_c_u32x8, state_d_u32x8, state_e_u32x8,
                             state_f_u32x8, &state_g_u32x8, schedule_u32x8[1], round_constants[1]);
    sz_sha256_round_haswell_(state_g_u32x8, state_h_u32x8, state_a_u32x8, &state_b_u32x8, state_c_u32x8, state_d_u32x8,
                             state_e_u32x8, &state_f_u32x8, schedule_u32x8[2], round_constants[2]);
    sz_sha256_round_haswell_(state_f_u32x8, state_g_u32x8, state_h_u32x8, &state_a_u32x8, state_b_u32x8, state_c_u32x8,
                             state_d_u32x8, &state_e_u32x8, schedule_u32x8[3], round_constants[3]);
    sz_sha256_round_haswell_(state_e_u32x8, state_f_u32x8, state_g_u32x8, &state_h_u32x8, state_a_u32x8, state_b_u32x8,
                             state_c_u32x8, &state_d_u32x8, schedule_u32x8[4], round_constants[4]);
    sz_sha256_round_haswell_(state_d_u32x8, state_e_u32x8, state_f_u32x8, &state_g_u32x8, state_h_u32x8, state_a_u32x8,
                             state_b_u32x8, &state_c_u32x8, schedule_u32x8[5], round_constants[5]);
    sz_sha256_round_haswell_(state_c_u32x8, state_d_u32x8, state_e_u32x8, &state_f_u32x8, state_g_u32x8, state_h_u32x8,
                             state_a_u32x8, &state_b_u32x8, schedule_u32x8[6], round_constants[6]);
    sz_sha256_round_haswell_(state_b_u32x8, state_c_u32x8, state_d_u32x8, &state_e_u32x8, state_f_u32x8, state_g_u32x8,
                             state_h_u32x8, &state_a_u32x8, schedule_u32x8[7], round_constants[7]);
    sz_sha256_round_haswell_(state_a_u32x8, state_b_u32x8, state_c_u32x8, &state_d_u32x8, state_e_u32x8, state_f_u32x8,
                             state_g_u32x8, &state_h_u32x8, schedule_u32x8[8], round_constants[8]);
    sz_sha256_round_haswell_(state_h_u32x8, state_a_u32x8, state_b_u32x8, &state_c_u32x8, state_d_u32x8, state_e_u32x8,
                             state_f_u32x8, &state_g_u32x8, schedule_u32x8[9], round_constants[9]);
    sz_sha256_round_haswell_(state_g_u32x8, state_h_u32x8, state_a_u32x8, &state_b_u32x8, state_c_u32x8, state_d_u32x8,
                             state_e_u32x8, &state_f_u32x8, schedule_u32x8[10], round_constants[10]);
    sz_sha256_round_haswell_(state_f_u32x8, state_g_u32x8, state_h_u32x8, &state_a_u32x8, state_b_u32x8, state_c_u32x8,
                             state_d_u32x8, &state_e_u32x8, schedule_u32x8[11], round_constants[11]);
    sz_sha256_round_haswell_(state_e_u32x8, state_f_u32x8, state_g_u32x8, &state_h_u32x8, state_a_u32x8, state_b_u32x8,
                             state_c_u32x8, &state_d_u32x8, schedule_u32x8[12], round_constants[12]);
    sz_sha256_round_haswell_(state_d_u32x8, state_e_u32x8, state_f_u32x8, &state_g_u32x8, state_h_u32x8, state_a_u32x8,
                             state_b_u32x8, &state_c_u32x8, schedule_u32x8[13], round_constants[13]);
    sz_sha256_round_haswell_(state_c_u32x8, state_d_u32x8, state_e_u32x8, &state_f_u32x8, state_g_u32x8, state_h_u32x8,
                             state_a_u32x8, &state_b_u32x8, schedule_u32x8[14], round_constants[14]);
    sz_sha256_round_haswell_(state_b_u32x8, state_c_u32x8, state_d_u32x8, &state_e_u32x8, state_f_u32x8, state_g_u32x8,
                             state_h_u32x8, &state_a_u32x8, schedule_u32x8[15], round_constants[15]);

    for (sz_size_t turn_index = 1; turn_index != 4; ++turn_index) {
        sz_u32_t const *turn_constants = round_constants + turn_index * 16;
        schedule_u32x8[0] = sz_sha256_extend_haswell_(schedule_u32x8[0], schedule_u32x8[1], schedule_u32x8[9],
                                                      schedule_u32x8[14]);
        sz_sha256_round_haswell_(state_a_u32x8, state_b_u32x8, state_c_u32x8, &state_d_u32x8, state_e_u32x8,
                                 state_f_u32x8, state_g_u32x8, &state_h_u32x8, schedule_u32x8[0], turn_constants[0]);
        schedule_u32x8[1] = sz_sha256_extend_haswell_(schedule_u32x8[1], schedule_u32x8[2], schedule_u32x8[10],
                                                      schedule_u32x8[15]);
        sz_sha256_round_haswell_(state_h_u32x8, state_a_u32x8, state_b_u32x8, &state_c_u32x8, state_d_u32x8,
                                 state_e_u32x8, state_f_u32x8, &state_g_u32x8, schedule_u32x8[1], turn_constants[1]);
        schedule_u32x8[2] = sz_sha256_extend_haswell_(schedule_u32x8[2], schedule_u32x8[3], schedule_u32x8[11],
                                                      schedule_u32x8[0]);
        sz_sha256_round_haswell_(state_g_u32x8, state_h_u32x8, state_a_u32x8, &state_b_u32x8, state_c_u32x8,
                                 state_d_u32x8, state_e_u32x8, &state_f_u32x8, schedule_u32x8[2], turn_constants[2]);
        schedule_u32x8[3] = sz_sha256_extend_haswell_(schedule_u32x8[3], schedule_u32x8[4], schedule_u32x8[12],
                                                      schedule_u32x8[1]);
        sz_sha256_round_haswell_(state_f_u32x8, state_g_u32x8, state_h_u32x8, &state_a_u32x8, state_b_u32x8,
                                 state_c_u32x8, state_d_u32x8, &state_e_u32x8, schedule_u32x8[3], turn_constants[3]);
        schedule_u32x8[4] = sz_sha256_extend_haswell_(schedule_u32x8[4], schedule_u32x8[5], schedule_u32x8[13],
                                                      schedule_u32x8[2]);
        sz_sha256_round_haswell_(state_e_u32x8, state_f_u32x8, state_g_u32x8, &state_h_u32x8, state_a_u32x8,
                                 state_b_u32x8, state_c_u32x8, &state_d_u32x8, schedule_u32x8[4], turn_constants[4]);
        schedule_u32x8[5] = sz_sha256_extend_haswell_(schedule_u32x8[5], schedule_u32x8[6], schedule_u32x8[14],
                                                      schedule_u32x8[3]);
        sz_sha256_round_haswell_(state_d_u32x8, state_e_u32x8, state_f_u32x8, &state_g_u32x8, state_h_u32x8,
                                 state_a_u32x8, state_b_u32x8, &state_c_u32x8, schedule_u32x8[5], turn_constants[5]);
        schedule_u32x8[6] = sz_sha256_extend_haswell_(schedule_u32x8[6], schedule_u32x8[7], schedule_u32x8[15],
                                                      schedule_u32x8[4]);
        sz_sha256_round_haswell_(state_c_u32x8, state_d_u32x8, state_e_u32x8, &state_f_u32x8, state_g_u32x8,
                                 state_h_u32x8, state_a_u32x8, &state_b_u32x8, schedule_u32x8[6], turn_constants[6]);
        schedule_u32x8[7] = sz_sha256_extend_haswell_(schedule_u32x8[7], schedule_u32x8[8], schedule_u32x8[0],
                                                      schedule_u32x8[5]);
        sz_sha256_round_haswell_(state_b_u32x8, state_c_u32x8, state_d_u32x8, &state_e_u32x8, state_f_u32x8,
                                 state_g_u32x8, state_h_u32x8, &state_a_u32x8, schedule_u32x8[7], turn_constants[7]);
        schedule_u32x8[8] = sz_sha256_extend_haswell_(schedule_u32x8[8], schedule_u32x8[9], schedule_u32x8[1],
                                                      schedule_u32x8[6]);
        sz_sha256_round_haswell_(state_a_u32x8, state_b_u32x8, state_c_u32x8, &state_d_u32x8, state_e_u32x8,
                                 state_f_u32x8, state_g_u32x8, &state_h_u32x8, schedule_u32x8[8], turn_constants[8]);
        schedule_u32x8[9] = sz_sha256_extend_haswell_(schedule_u32x8[9], schedule_u32x8[10], schedule_u32x8[2],
                                                      schedule_u32x8[7]);
        sz_sha256_round_haswell_(state_h_u32x8, state_a_u32x8, state_b_u32x8, &state_c_u32x8, state_d_u32x8,
                                 state_e_u32x8, state_f_u32x8, &state_g_u32x8, schedule_u32x8[9], turn_constants[9]);
        schedule_u32x8[10] = sz_sha256_extend_haswell_(schedule_u32x8[10], schedule_u32x8[11], schedule_u32x8[3],
                                                       schedule_u32x8[8]);
        sz_sha256_round_haswell_(state_g_u32x8, state_h_u32x8, state_a_u32x8, &state_b_u32x8, state_c_u32x8,
                                 state_d_u32x8, state_e_u32x8, &state_f_u32x8, schedule_u32x8[10], turn_constants[10]);
        schedule_u32x8[11] = sz_sha256_extend_haswell_(schedule_u32x8[11], schedule_u32x8[12], schedule_u32x8[4],
                                                       schedule_u32x8[9]);
        sz_sha256_round_haswell_(state_f_u32x8, state_g_u32x8, state_h_u32x8, &state_a_u32x8, state_b_u32x8,
                                 state_c_u32x8, state_d_u32x8, &state_e_u32x8, schedule_u32x8[11], turn_constants[11]);
        schedule_u32x8[12] = sz_sha256_extend_haswell_(schedule_u32x8[12], schedule_u32x8[13], schedule_u32x8[5],
                                                       schedule_u32x8[10]);
        sz_sha256_round_haswell_(state_e_u32x8, state_f_u32x8, state_g_u32x8, &state_h_u32x8, state_a_u32x8,
                                 state_b_u32x8, state_c_u32x8, &state_d_u32x8, schedule_u32x8[12], turn_constants[12]);
        schedule_u32x8[13] = sz_sha256_extend_haswell_(schedule_u32x8[13], schedule_u32x8[14], schedule_u32x8[6],
                                                       schedule_u32x8[11]);
        sz_sha256_round_haswell_(state_d_u32x8, state_e_u32x8, state_f_u32x8, &state_g_u32x8, state_h_u32x8,
                                 state_a_u32x8, state_b_u32x8, &state_c_u32x8, schedule_u32x8[13], turn_constants[13]);
        schedule_u32x8[14] = sz_sha256_extend_haswell_(schedule_u32x8[14], schedule_u32x8[15], schedule_u32x8[7],
                                                       schedule_u32x8[12]);
        sz_sha256_round_haswell_(state_c_u32x8, state_d_u32x8, state_e_u32x8, &state_f_u32x8, state_g_u32x8,
                                 state_h_u32x8, state_a_u32x8, &state_b_u32x8, schedule_u32x8[14], turn_constants[14]);
        schedule_u32x8[15] = sz_sha256_extend_haswell_(schedule_u32x8[15], schedule_u32x8[0], schedule_u32x8[8],
                                                       schedule_u32x8[13]);
        sz_sha256_round_haswell_(state_b_u32x8, state_c_u32x8, state_d_u32x8, &state_e_u32x8, state_f_u32x8,
                                 state_g_u32x8, state_h_u32x8, &state_a_u32x8, schedule_u32x8[15], turn_constants[15]);
    }

    hashes_u32x8[0] = _mm256_add_epi32(hashes_u32x8[0], _mm256_and_si256(active_u32x8, state_a_u32x8));
    hashes_u32x8[1] = _mm256_add_epi32(hashes_u32x8[1], _mm256_and_si256(active_u32x8, state_b_u32x8));
    hashes_u32x8[2] = _mm256_add_epi32(hashes_u32x8[2], _mm256_and_si256(active_u32x8, state_c_u32x8));
    hashes_u32x8[3] = _mm256_add_epi32(hashes_u32x8[3], _mm256_and_si256(active_u32x8, state_d_u32x8));
    hashes_u32x8[4] = _mm256_add_epi32(hashes_u32x8[4], _mm256_and_si256(active_u32x8, state_e_u32x8));
    hashes_u32x8[5] = _mm256_add_epi32(hashes_u32x8[5], _mm256_and_si256(active_u32x8, state_f_u32x8));
    hashes_u32x8[6] = _mm256_add_epi32(hashes_u32x8[6], _mm256_and_si256(active_u32x8, state_g_u32x8));
    hashes_u32x8[7] = _mm256_add_epi32(hashes_u32x8[7], _mm256_and_si256(active_u32x8, state_h_u32x8));
}

/**
 *  @brief Compresses each lane's own run of whole blocks, plus an optional buffered block ahead of them.
 *  @param states The lane states, whose hash words are gathered on entry and scattered on exit.
 *  @param active_lanes_count Lanes owning a state, 1 to 8; the rest borrow lane zero and are discarded.
 *  @param buffered_bitmask Lanes whose `block` the caller has just filled to 64 bytes, compressed first.
 *  @param cursors Per-lane read positions, advanced past every block consumed.
 *  @param blocks_per_lane Whole 64-byte blocks each lane owns; the loop runs as far as the largest.
 *
 *  Lanes retire independently. `blocks_per_lane` rides in a register as a countdown, and two masks come off
 *  it each turn: `counts > 0` says whose accumulation lands, `counts > 1` says whose cursor steps. Splitting
 *  the two is what parks a retiring lane on its @b last full block instead of on a short tail, so every read
 *  stays in bounds with no per-block pointer select. A lane owning no blocks at all reads its own `block`
 *  buffer, which is always 64 valid bytes.
 *
 *  The gather and scatter go through one 32-byte temporary rather than a 256-byte array, since neither is
 *  live across the loop between them.
 *
 */
SZ_HELPER_INLINE void sz_sha256_multistate_blocks_haswell_(sz_sha256_state_t *states, sz_size_t active_lanes_count,
                                                           sz_u32_t buffered_bitmask, sz_u8_t const **cursors,
                                                           sz_size_t const *blocks_per_lane) {
    __m256i hashes_u32x8[8];
    sz_u256_vec_t counts_vec, buffered_vec;
    sz_u8_t const *sources[8];
    sz_size_t largest_blocks_count = 0;

    // The topped-up block and the fallback for a lane with nothing whole left are the same buffer, so one
    // source array serves both phases and the head costs no extra stack. An absent lane, or one with
    // nothing whole left, keeps reading that buffer and never advances, so the transpose stays in bounds
    // without a bounds test in the loop.
    for (sz_size_t lane_index = 0; lane_index != 8; ++lane_index) {
        sz_size_t const source_lane = lane_index < active_lanes_count ? lane_index : 0;
        sz_size_t const lane_blocks = lane_index < active_lanes_count ? blocks_per_lane[lane_index] : 0;
        counts_vec.u32s[lane_index] = (sz_u32_t)lane_blocks;
        buffered_vec.u32s[lane_index] = ((buffered_bitmask >> lane_index) & 1u) ? 0xFFFFFFFFu : 0u;
        sources[lane_index] = states[source_lane].block;
        if (lane_blocks > largest_blocks_count) largest_blocks_count = lane_blocks;
    }

    // Token-sized messages compress nothing here - their whole hash is the padding block the digest path
    // emits - so leaving before the gather keeps them off the word-major round trip entirely.
    if (buffered_bitmask == 0 && largest_blocks_count == 0) return;

    // A lane's eight hash words are exactly one 256-bit register, so the transpose that carries the message
    // schedule carries the state too, and AVX2's missing scatter never comes up.
    __m256i lanes_u32x8[8];
    for (sz_size_t lane_index = 0; lane_index != 8; ++lane_index) {
        sz_size_t const source_lane = lane_index < active_lanes_count ? lane_index : 0;
        lanes_u32x8[lane_index] = _mm256_loadu_si256((__m256i const *)states[source_lane].hash);
    }
    sz_sha256_transpose_8x8_haswell_(lanes_u32x8, hashes_u32x8);

    if (buffered_bitmask) sz_sha256_compress_haswell_(hashes_u32x8, sources, buffered_vec.ymm);
    for (sz_size_t lane_index = 0; lane_index != 8; ++lane_index)
        if (counts_vec.u32s[lane_index] != 0) sources[lane_index] = cursors[lane_index];

    __m256i counts_u32x8 = counts_vec.ymm;
    __m256i const ones_u32x8 = _mm256_set1_epi32(1);
    for (sz_size_t block_index = 0; block_index != largest_blocks_count; ++block_index) {
        __m256i const active_u32x8 = _mm256_cmpgt_epi32(counts_u32x8, _mm256_setzero_si256());
        __m256i const advance_u32x8 = _mm256_cmpgt_epi32(counts_u32x8, ones_u32x8);
        sz_u32_t const advance_bitmask = (sz_u32_t)_mm256_movemask_ps(_mm256_castsi256_ps(advance_u32x8));
        sz_sha256_compress_haswell_(hashes_u32x8, sources, active_u32x8);
        for (sz_size_t lane_index = 0; lane_index != 8; ++lane_index)
            sources[lane_index] += ((advance_bitmask >> lane_index) & 1u) * SZ_SHA256_BLOCK_LENGTH;
        counts_u32x8 = _mm256_add_epi32(counts_u32x8, active_u32x8);
    }

    sz_sha256_transpose_8x8_haswell_(hashes_u32x8, lanes_u32x8);
    for (sz_size_t lane_index = 0; lane_index != active_lanes_count; ++lane_index)
        _mm256_storeu_si256((__m256i *)states[lane_index].hash, lanes_u32x8[lane_index]);
    for (sz_size_t lane_index = 0; lane_index != active_lanes_count; ++lane_index)
        cursors[lane_index] += blocks_per_lane[lane_index] * SZ_SHA256_BLOCK_LENGTH;
}

SZ_API_COMPTIME void sz_sha256_multistate_update_haswell(sz_sha256_state_t *states, sz_sequence_t const *texts) {
    sz_size_t const lanes_count = texts->count;

    for (sz_size_t first_lane_index = 0; first_lane_index < lanes_count; first_lane_index += 8) {
        sz_size_t const lanes_left = lanes_count - first_lane_index;
        sz_size_t const active_lanes_count = lanes_left < 8 ? lanes_left : 8;
        sz_u8_t const *cursors[8];
        sz_size_t remaining[8], blocks_per_lane[8];
        sz_u32_t buffered_bitmask = 0;

        // Top up any buffered partial block into the state's own 64-byte buffer, which then serves as that
        // lane's first block source. Every lane's whole chunk is charged to `total_length` here, once, so
        // the head, body and tail below only move bytes.
        for (sz_size_t lane_index = 0; lane_index != active_lanes_count; ++lane_index) {
            sz_sha256_state_t *const state = &states[first_lane_index + lane_index];
            cursors[lane_index] = (sz_u8_t const *)texts->get_start(texts->handle, first_lane_index + lane_index);
            remaining[lane_index] = texts->get_length(texts->handle, first_lane_index + lane_index);

            // The countdown rides in 32-bit lanes AVX2 compares as signed, so a chunk longer than that many
            // blocks goes through the single-state kernel over the very same state and sits the group out.
            if (remaining[lane_index] / SZ_SHA256_BLOCK_LENGTH > 0x7FFFFFFFull) {
                sz_sha256_state_update_serial(state, (sz_cptr_t)cursors[lane_index], remaining[lane_index]);
                remaining[lane_index] = 0, blocks_per_lane[lane_index] = 0;
                continue;
            }

            state->total_length += remaining[lane_index];
            if (state->block_length != 0) {
                sz_size_t const missing = SZ_SHA256_BLOCK_LENGTH - state->block_length;
                if (remaining[lane_index] >= missing) {
                    for (sz_size_t byte_index = 0; byte_index != missing; ++byte_index)
                        state->block[state->block_length + byte_index] = cursors[lane_index][byte_index];
                    buffered_bitmask |= (sz_u32_t)1 << lane_index;
                    state->block_length = 0;
                    cursors[lane_index] += missing, remaining[lane_index] -= missing;
                }
            }
            blocks_per_lane[lane_index] = remaining[lane_index] / SZ_SHA256_BLOCK_LENGTH;
        }

        sz_sha256_multistate_blocks_haswell_(&states[first_lane_index], active_lanes_count, buffered_bitmask, cursors,
                                             blocks_per_lane);

        // Whatever is left cannot fill a block, so it only ever buffers.
        for (sz_size_t lane_index = 0; lane_index != active_lanes_count; ++lane_index) {
            sz_sha256_state_t *const state = &states[first_lane_index + lane_index];
            sz_size_t const tail_length = remaining[lane_index] % SZ_SHA256_BLOCK_LENGTH;
            for (sz_size_t byte_index = 0; byte_index != tail_length; ++byte_index)
                state->block[state->block_length + byte_index] = cursors[lane_index][byte_index];
            state->block_length += tail_length;
        }
    }
}

/**
 *  @brief Finalizes 8 already-gathered lanes into their digests.
 *  @param states The 8 lane states, left untouched.
 *  @param active_lanes_count Lanes to emit, 1 to 8; the rest ride along and are discarded.
 *  @param digests Receives `active_lanes_count` 32-byte big-endian digests.
 *
 *  Same two-pass shape as the Skylake path, except the lane select is a blend vector rather than a k-mask,
 *  since AVX2 has no mask registers.
 */
SZ_HELPER_INLINE void sz_sha256_multistate_digest_lanes_haswell_(sz_sha256_state_t const *states,
                                                                 sz_size_t active_lanes_count, sz_u8_t *digests) {
    // A SHA256 block is 64 bytes whatever the vector width, so the staged blocks are 512-bit unions even
    // though the lanes themselves are 256-bit wide.
    sz_u512_vec_t staged_vec[8];
    sz_u256_vec_t overflow_vec;
    sz_u8_t const *staged_blocks[8];
    __m256i hashes_u32x8[8], lanes_u32x8[8];
    sz_bool_t any_overflow = sz_false_k;

    for (sz_size_t lane_index = 0; lane_index != 8; ++lane_index) {
        sz_size_t const source_lane = lane_index < active_lanes_count ? lane_index : 0;
        staged_blocks[lane_index] = staged_vec[lane_index].u8s;
        lanes_u32x8[lane_index] = _mm256_loadu_si256((__m256i const *)states[source_lane].hash);
        overflow_vec.u32s[lane_index] = 0;

        // The terminator lands in a carrier block whenever it would crowd out the trailing bit length.
        if (states[source_lane].block_length + 1 > SZ_SHA256_BLOCK_LENGTH - 8)
            overflow_vec.u32s[lane_index] = 0xFFFFFFFFu, any_overflow = sz_true_k;
    }
    sz_sha256_transpose_8x8_haswell_(lanes_u32x8, hashes_u32x8);

    // One staging buffer serves both blocks: the carrier compression is gated to the lanes that need it and
    // has consumed the buffer before the final block overwrites it.
    if (any_overflow == sz_true_k) {
        for (sz_size_t lane_index = 0; lane_index != 8; ++lane_index) {
            sz_size_t const source_lane = lane_index < active_lanes_count ? lane_index : 0;
            sz_size_t const buffered = states[source_lane].block_length;
            for (sz_size_t byte_index = 0; byte_index != SZ_SHA256_BLOCK_LENGTH; ++byte_index)
                staged_vec[lane_index].u8s[byte_index] = byte_index < buffered ? states[source_lane].block[byte_index]
                                                                               : (sz_u8_t)0;
            staged_vec[lane_index].u8s[buffered] = 0x80;
        }
        sz_sha256_compress_haswell_(hashes_u32x8, staged_blocks, overflow_vec.ymm);
    }

    for (sz_size_t lane_index = 0; lane_index != 8; ++lane_index) {
        sz_size_t const source_lane = lane_index < active_lanes_count ? lane_index : 0;
        sz_size_t const buffered = states[source_lane].block_length;
        // An overflowing lane has already spent its buffered bytes on the carrier block, so its final block
        // carries nothing but the trailing length.
        int const carried = overflow_vec.u32s[lane_index] != 0;
        sz_size_t const kept = carried ? 0 : buffered;
        for (sz_size_t byte_index = 0; byte_index != SZ_SHA256_BLOCK_LENGTH; ++byte_index)
            staged_vec[lane_index].u8s[byte_index] = byte_index < kept ? states[source_lane].block[byte_index]
                                                                       : (sz_u8_t)0;
        if (!carried) staged_vec[lane_index].u8s[buffered] = 0x80;
        staged_vec[lane_index].u64s[7] = sz_u64_bytes_reverse(states[source_lane].total_length * 8);
    }
    sz_sha256_compress_haswell_(hashes_u32x8, staged_blocks, _mm256_set1_epi32(-1));

    // Big-endian output is a byte reverse inside each word, and the same transpose that gathered the state
    // returns each lane's digest as one 32-byte register.
    __m256i const byte_swap_u8x32 = _mm256_setr_epi8(         //
        3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12, //
        3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12);
    for (sz_size_t word_index = 0; word_index != 8; ++word_index)
        hashes_u32x8[word_index] = _mm256_shuffle_epi8(hashes_u32x8[word_index], byte_swap_u8x32);
    sz_sha256_transpose_8x8_haswell_(hashes_u32x8, lanes_u32x8);
    for (sz_size_t lane_index = 0; lane_index != active_lanes_count; ++lane_index)
        _mm256_storeu_si256((__m256i *)&digests[lane_index * SZ_SHA256_DIGEST_LENGTH], lanes_u32x8[lane_index]);
}

SZ_API_COMPTIME void sz_sha256_multistate_digest_haswell(sz_sha256_state_t const *states, sz_size_t states_count,
                                                         sz_u8_t *digests) {
    sz_size_t first_lane_index = 0;
    for (; first_lane_index < states_count; first_lane_index += 8) {
        sz_size_t const remaining = states_count - first_lane_index;
        sz_size_t const active_lanes_count = remaining < 8 ? remaining : 8;
        sz_sha256_multistate_digest_lanes_haswell_(&states[first_lane_index], active_lanes_count,
                                                   &digests[first_lane_index * SZ_SHA256_DIGEST_LENGTH]);
    }
}

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#endif // SZ_USE_HASWELL

#ifdef __cplusplus
}
#endif

#endif // STRINGZILLA_HASH_HASWELL_H_
