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
        __m128i low_xmm = _mm256_castsi256_si128(sums_vec.ymm);
        __m128i high_xmm = _mm256_extracti128_si256(sums_vec.ymm, 1);
        __m128i sums_xmm = _mm_add_epi64(low_xmm, high_xmm);
        sz_u64_t low = (sz_u64_t)_mm_cvtsi128_si64(sums_xmm);
        sz_u64_t high = (sz_u64_t)_mm_extract_epi64(sums_xmm, 1);
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
        __m128i low_xmm = _mm256_castsi256_si128(sums_vec.ymm);
        __m128i high_xmm = _mm256_extracti128_si256(sums_vec.ymm, 1);
        __m128i sums_xmm = _mm_add_epi64(low_xmm, high_xmm);
        sz_u64_t low = (sz_u64_t)_mm_cvtsi128_si64(sums_xmm);
        sz_u64_t high = (sz_u64_t)_mm_extract_epi64(sums_xmm, 1);
        result += low + high;
        return result;
    }
}

/*  Without `vpternlogd` every bitwise primitive expands, and without `vprord` every rotation becomes a shift
 *  pair and an or. Both are cheaper than the instruction counts suggest: logic issues on all three vector
 *  ports on this class of core, so the round mix stays throughput-bound rather than shift-bound. */

/** @brief Evaluates `(state_e & state_f) ^ (~state_e & state_g)` across 8 lanes. */
SZ_HELPER_INLINE __m256i sz_sha256_choice_haswell_(__m256i state_e_ymm, __m256i state_f_ymm, __m256i state_g_ymm) {
    return _mm256_xor_si256(state_g_ymm, _mm256_and_si256(state_e_ymm, _mm256_xor_si256(state_f_ymm, state_g_ymm)));
}

/** @brief Evaluates `(state_a & state_b) ^ (state_a & state_c) ^ (state_b & state_c)` across 8 lanes. */
SZ_HELPER_INLINE __m256i sz_sha256_majority_haswell_(__m256i state_a_ymm, __m256i state_b_ymm, __m256i state_c_ymm) {
    return _mm256_xor_si256(_mm256_and_si256(_mm256_xor_si256(state_a_ymm, state_b_ymm), state_c_ymm),
                            _mm256_and_si256(state_a_ymm, state_b_ymm));
}

/** @brief Evaluates `ror(state_a_ymm, 2) ^ ror(state_a_ymm, 13) ^ ror(state_a_ymm, 22)` across 8 lanes. */
SZ_HELPER_INLINE __m256i sz_sha256_big_sigma0_haswell_(__m256i state_a_ymm) {
    __m256i const rotated_by_2_ymm = _mm256_or_si256(_mm256_srli_epi32(state_a_ymm, 2),
                                                     _mm256_slli_epi32(state_a_ymm, 30));
    __m256i const rotated_by_13_ymm = _mm256_or_si256(_mm256_srli_epi32(state_a_ymm, 13),
                                                      _mm256_slli_epi32(state_a_ymm, 19));
    __m256i const rotated_by_22_ymm = _mm256_or_si256(_mm256_srli_epi32(state_a_ymm, 22),
                                                      _mm256_slli_epi32(state_a_ymm, 10));
    return _mm256_xor_si256(_mm256_xor_si256(rotated_by_2_ymm, rotated_by_13_ymm), rotated_by_22_ymm);
}

/** @brief Evaluates `ror(state_e_ymm, 6) ^ ror(state_e_ymm, 11) ^ ror(state_e_ymm, 25)` across 8 lanes. */
SZ_HELPER_INLINE __m256i sz_sha256_big_sigma1_haswell_(__m256i state_e_ymm) {
    __m256i const rotated_by_6_ymm = _mm256_or_si256(_mm256_srli_epi32(state_e_ymm, 6),
                                                     _mm256_slli_epi32(state_e_ymm, 26));
    __m256i const rotated_by_11_ymm = _mm256_or_si256(_mm256_srli_epi32(state_e_ymm, 11),
                                                      _mm256_slli_epi32(state_e_ymm, 21));
    __m256i const rotated_by_25_ymm = _mm256_or_si256(_mm256_srli_epi32(state_e_ymm, 25),
                                                      _mm256_slli_epi32(state_e_ymm, 7));
    return _mm256_xor_si256(_mm256_xor_si256(rotated_by_6_ymm, rotated_by_11_ymm), rotated_by_25_ymm);
}

/** @brief Evaluates `ror(word, 7) ^ ror(word, 18) ^ (word >> 3)` across 8 lanes. */
SZ_HELPER_INLINE __m256i sz_sha256_small_sigma0_haswell_(__m256i message_word_ymm) {
    __m256i const rotated_by_7_ymm = _mm256_or_si256(_mm256_srli_epi32(message_word_ymm, 7),
                                                     _mm256_slli_epi32(message_word_ymm, 25));
    __m256i const rotated_by_18_ymm = _mm256_or_si256(_mm256_srli_epi32(message_word_ymm, 18),
                                                      _mm256_slli_epi32(message_word_ymm, 14));
    return _mm256_xor_si256(_mm256_xor_si256(rotated_by_7_ymm, rotated_by_18_ymm),
                            _mm256_srli_epi32(message_word_ymm, 3));
}

/** @brief Evaluates `ror(word, 17) ^ ror(word, 19) ^ (word >> 10)` across 8 lanes. */
SZ_HELPER_INLINE __m256i sz_sha256_small_sigma1_haswell_(__m256i message_word_ymm) {
    __m256i const rotated_by_17_ymm = _mm256_or_si256(_mm256_srli_epi32(message_word_ymm, 17),
                                                      _mm256_slli_epi32(message_word_ymm, 15));
    __m256i const rotated_by_19_ymm = _mm256_or_si256(_mm256_srli_epi32(message_word_ymm, 19),
                                                      _mm256_slli_epi32(message_word_ymm, 13));
    return _mm256_xor_si256(_mm256_xor_si256(rotated_by_17_ymm, rotated_by_19_ymm),
                            _mm256_srli_epi32(message_word_ymm, 10));
}

/**
 *  @brief Transposes eight 32-byte halves into word-major big-endian order.
 *  @param lanes_ymm The eight per-lane halves, byte-swapped on entry.
 *  @param words_ymm Receives eight registers, where `words_ymm[index]` holds that word from all 8 lanes.
 *
 *  Two interleave stages gather 32-bit and then 64-bit neighbours inside each 128-bit half, and a final
 *  cross-half permute completes the exchange.
 */
SZ_HELPER_INLINE void sz_sha256_transpose_8x8_haswell_(__m256i const lanes_ymm[8], __m256i words_ymm[8]) {
    __m256i first_stage_ymm[8], second_stage_ymm[8];
    for (sz_size_t butterfly_index = 0; butterfly_index != 4; ++butterfly_index) {
        first_stage_ymm[butterfly_index * 2 + 0] = _mm256_unpacklo_epi32(lanes_ymm[butterfly_index * 2 + 0],
                                                                    lanes_ymm[butterfly_index * 2 + 1]);
        first_stage_ymm[butterfly_index * 2 + 1] = _mm256_unpackhi_epi32(lanes_ymm[butterfly_index * 2 + 0],
                                                                    lanes_ymm[butterfly_index * 2 + 1]);
    }
    second_stage_ymm[0] = _mm256_unpacklo_epi64(first_stage_ymm[0], first_stage_ymm[2]);
    second_stage_ymm[1] = _mm256_unpackhi_epi64(first_stage_ymm[0], first_stage_ymm[2]);
    second_stage_ymm[2] = _mm256_unpacklo_epi64(first_stage_ymm[1], first_stage_ymm[3]);
    second_stage_ymm[3] = _mm256_unpackhi_epi64(first_stage_ymm[1], first_stage_ymm[3]);
    second_stage_ymm[4] = _mm256_unpacklo_epi64(first_stage_ymm[4], first_stage_ymm[6]);
    second_stage_ymm[5] = _mm256_unpackhi_epi64(first_stage_ymm[4], first_stage_ymm[6]);
    second_stage_ymm[6] = _mm256_unpacklo_epi64(first_stage_ymm[5], first_stage_ymm[7]);
    second_stage_ymm[7] = _mm256_unpackhi_epi64(first_stage_ymm[5], first_stage_ymm[7]);
    for (sz_size_t offset = 0; offset != 4; ++offset) {
        words_ymm[offset] = _mm256_permute2x128_si256(second_stage_ymm[offset], second_stage_ymm[offset + 4], 0x20);
        words_ymm[offset + 4] = _mm256_permute2x128_si256(second_stage_ymm[offset], second_stage_ymm[offset + 4], 0x31);
    }
}

/**
 *  @brief Extends the rolling message window by one word across 8 lanes.
 *  @param oldest_word_ymm The word being replaced, sixteen rounds behind.
 *  @param next_word_ymm The word one position ahead, feeding the low sigma.
 *  @param ninth_word_ymm The word nine positions ahead.
 *  @param fourteenth_word_ymm The word fourteen positions ahead, feeding the high sigma.
 */
SZ_HELPER_INLINE __m256i sz_sha256_extend_haswell_(__m256i oldest_word_ymm, __m256i next_word_ymm,
                                                   __m256i ninth_word_ymm, __m256i fourteenth_word_ymm) {
    return _mm256_add_epi32(_mm256_add_epi32(oldest_word_ymm, sz_sha256_small_sigma0_haswell_(next_word_ymm)),
                            _mm256_add_epi32(ninth_word_ymm, sz_sha256_small_sigma1_haswell_(fourteenth_word_ymm)));
}

/**
 *  @brief Applies one SHA256 round to 8 lanes, writing the two working values that change.
 *
 *  The eight working variables rotate by one position per round, which the callers express by passing the
 *  same locals in a different order rather than by moving data. Only `state_d` and `state_h` are written:
 *  `state_d` becomes the next round's `state_e`, and `state_h` is dead on entry so it receives the next
 *  round's `state_a`.
 */
SZ_HELPER_INLINE void sz_sha256_round_haswell_(                                          //
    __m256i state_a_ymm, __m256i state_b_ymm, __m256i state_c_ymm, __m256i *state_d_ymm, //
    __m256i state_e_ymm, __m256i state_f_ymm, __m256i state_g_ymm, __m256i *state_h_ymm, //
    __m256i message_word_ymm, sz_u32_t round_constant) {
    __m256i const temporary_first_ymm = _mm256_add_epi32( //
        _mm256_add_epi32(
            _mm256_add_epi32(*state_h_ymm, sz_sha256_big_sigma1_haswell_(state_e_ymm)),
            _mm256_add_epi32(sz_sha256_choice_haswell_(state_e_ymm, state_f_ymm, state_g_ymm), message_word_ymm)),
        _mm256_set1_epi32((int)round_constant));
    __m256i const temporary_second_ymm = _mm256_add_epi32(
        sz_sha256_big_sigma0_haswell_(state_a_ymm), sz_sha256_majority_haswell_(state_a_ymm, state_b_ymm, state_c_ymm));
    *state_d_ymm = _mm256_add_epi32(*state_d_ymm, temporary_first_ymm);
    *state_h_ymm = _mm256_add_epi32(temporary_first_ymm, temporary_second_ymm);
}

/**
 *  @brief Transposes and compresses one 64-byte block for each of 8 lanes into word-major hash registers.
 *  @param hashes Eight registers of word-major hash state, updated in place.
 *  @param lane_blocks Pointers to 8 message blocks, one per lane.
 *
 *  A 64-byte block is two 256-bit registers per lane, so the sixteen message words come from two independent
 *  eight-by-eight transposes. The window lives in a local rather than a parameter so it is not forced to
 *  memory across a call boundary.
 */
SZ_HELPER_INLINE void sz_sha256_compress_haswell_(__m256i hashes_ymm[8], sz_u8_t const *const *lane_blocks) {
    sz_u32_t const *round_constants = (sz_u32_t const *)sz_x86_hide_pointer_origin_(sz_sha256_round_constants_());

    __m256i const byte_swap_ymm = _mm256_setr_epi8(3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12, //
                                                   3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12);
    __m256i low_lanes_ymm[8], high_lanes_ymm[8], schedule_ymm[16];
    for (sz_size_t lane_index = 0; lane_index != 8; ++lane_index) {
        low_lanes_ymm[lane_index] = _mm256_shuffle_epi8(
            _mm256_loadu_si256((__m256i const *)&lane_blocks[lane_index][0]), byte_swap_ymm);
        high_lanes_ymm[lane_index] = _mm256_shuffle_epi8(
            _mm256_loadu_si256((__m256i const *)&lane_blocks[lane_index][32]), byte_swap_ymm);
    }
    sz_sha256_transpose_8x8_haswell_(low_lanes_ymm, &schedule_ymm[0]);
    sz_sha256_transpose_8x8_haswell_(high_lanes_ymm, &schedule_ymm[8]);

    __m256i state_a_ymm = hashes_ymm[0], state_b_ymm = hashes_ymm[1];
    __m256i state_c_ymm = hashes_ymm[2], state_d_ymm = hashes_ymm[3];
    __m256i state_e_ymm = hashes_ymm[4], state_f_ymm = hashes_ymm[5];
    __m256i state_g_ymm = hashes_ymm[6], state_h_ymm = hashes_ymm[7];

    sz_sha256_round_haswell_(state_a_ymm, state_b_ymm, state_c_ymm, &state_d_ymm, state_e_ymm, state_f_ymm, state_g_ymm,
                             &state_h_ymm, schedule_ymm[0], round_constants[0]);
    sz_sha256_round_haswell_(state_h_ymm, state_a_ymm, state_b_ymm, &state_c_ymm, state_d_ymm, state_e_ymm, state_f_ymm,
                             &state_g_ymm, schedule_ymm[1], round_constants[1]);
    sz_sha256_round_haswell_(state_g_ymm, state_h_ymm, state_a_ymm, &state_b_ymm, state_c_ymm, state_d_ymm, state_e_ymm,
                             &state_f_ymm, schedule_ymm[2], round_constants[2]);
    sz_sha256_round_haswell_(state_f_ymm, state_g_ymm, state_h_ymm, &state_a_ymm, state_b_ymm, state_c_ymm, state_d_ymm,
                             &state_e_ymm, schedule_ymm[3], round_constants[3]);
    sz_sha256_round_haswell_(state_e_ymm, state_f_ymm, state_g_ymm, &state_h_ymm, state_a_ymm, state_b_ymm, state_c_ymm,
                             &state_d_ymm, schedule_ymm[4], round_constants[4]);
    sz_sha256_round_haswell_(state_d_ymm, state_e_ymm, state_f_ymm, &state_g_ymm, state_h_ymm, state_a_ymm, state_b_ymm,
                             &state_c_ymm, schedule_ymm[5], round_constants[5]);
    sz_sha256_round_haswell_(state_c_ymm, state_d_ymm, state_e_ymm, &state_f_ymm, state_g_ymm, state_h_ymm, state_a_ymm,
                             &state_b_ymm, schedule_ymm[6], round_constants[6]);
    sz_sha256_round_haswell_(state_b_ymm, state_c_ymm, state_d_ymm, &state_e_ymm, state_f_ymm, state_g_ymm, state_h_ymm,
                             &state_a_ymm, schedule_ymm[7], round_constants[7]);
    sz_sha256_round_haswell_(state_a_ymm, state_b_ymm, state_c_ymm, &state_d_ymm, state_e_ymm, state_f_ymm, state_g_ymm,
                             &state_h_ymm, schedule_ymm[8], round_constants[8]);
    sz_sha256_round_haswell_(state_h_ymm, state_a_ymm, state_b_ymm, &state_c_ymm, state_d_ymm, state_e_ymm, state_f_ymm,
                             &state_g_ymm, schedule_ymm[9], round_constants[9]);
    sz_sha256_round_haswell_(state_g_ymm, state_h_ymm, state_a_ymm, &state_b_ymm, state_c_ymm, state_d_ymm, state_e_ymm,
                             &state_f_ymm, schedule_ymm[10], round_constants[10]);
    sz_sha256_round_haswell_(state_f_ymm, state_g_ymm, state_h_ymm, &state_a_ymm, state_b_ymm, state_c_ymm, state_d_ymm,
                             &state_e_ymm, schedule_ymm[11], round_constants[11]);
    sz_sha256_round_haswell_(state_e_ymm, state_f_ymm, state_g_ymm, &state_h_ymm, state_a_ymm, state_b_ymm, state_c_ymm,
                             &state_d_ymm, schedule_ymm[12], round_constants[12]);
    sz_sha256_round_haswell_(state_d_ymm, state_e_ymm, state_f_ymm, &state_g_ymm, state_h_ymm, state_a_ymm, state_b_ymm,
                             &state_c_ymm, schedule_ymm[13], round_constants[13]);
    sz_sha256_round_haswell_(state_c_ymm, state_d_ymm, state_e_ymm, &state_f_ymm, state_g_ymm, state_h_ymm, state_a_ymm,
                             &state_b_ymm, schedule_ymm[14], round_constants[14]);
    sz_sha256_round_haswell_(state_b_ymm, state_c_ymm, state_d_ymm, &state_e_ymm, state_f_ymm, state_g_ymm, state_h_ymm,
                             &state_a_ymm, schedule_ymm[15], round_constants[15]);

    for (sz_size_t turn_index = 1; turn_index != 4; ++turn_index) {
        sz_u32_t const *turn_constants = round_constants + turn_index * 16;
        schedule_ymm[0] = sz_sha256_extend_haswell_(schedule_ymm[0], schedule_ymm[1], schedule_ymm[9],
                                                    schedule_ymm[14]);
        sz_sha256_round_haswell_(state_a_ymm, state_b_ymm, state_c_ymm, &state_d_ymm, state_e_ymm, state_f_ymm,
                                 state_g_ymm, &state_h_ymm, schedule_ymm[0], turn_constants[0]);
        schedule_ymm[1] = sz_sha256_extend_haswell_(schedule_ymm[1], schedule_ymm[2], schedule_ymm[10],
                                                    schedule_ymm[15]);
        sz_sha256_round_haswell_(state_h_ymm, state_a_ymm, state_b_ymm, &state_c_ymm, state_d_ymm, state_e_ymm,
                                 state_f_ymm, &state_g_ymm, schedule_ymm[1], turn_constants[1]);
        schedule_ymm[2] = sz_sha256_extend_haswell_(schedule_ymm[2], schedule_ymm[3], schedule_ymm[11],
                                                    schedule_ymm[0]);
        sz_sha256_round_haswell_(state_g_ymm, state_h_ymm, state_a_ymm, &state_b_ymm, state_c_ymm, state_d_ymm,
                                 state_e_ymm, &state_f_ymm, schedule_ymm[2], turn_constants[2]);
        schedule_ymm[3] = sz_sha256_extend_haswell_(schedule_ymm[3], schedule_ymm[4], schedule_ymm[12],
                                                    schedule_ymm[1]);
        sz_sha256_round_haswell_(state_f_ymm, state_g_ymm, state_h_ymm, &state_a_ymm, state_b_ymm, state_c_ymm,
                                 state_d_ymm, &state_e_ymm, schedule_ymm[3], turn_constants[3]);
        schedule_ymm[4] = sz_sha256_extend_haswell_(schedule_ymm[4], schedule_ymm[5], schedule_ymm[13],
                                                    schedule_ymm[2]);
        sz_sha256_round_haswell_(state_e_ymm, state_f_ymm, state_g_ymm, &state_h_ymm, state_a_ymm, state_b_ymm,
                                 state_c_ymm, &state_d_ymm, schedule_ymm[4], turn_constants[4]);
        schedule_ymm[5] = sz_sha256_extend_haswell_(schedule_ymm[5], schedule_ymm[6], schedule_ymm[14],
                                                    schedule_ymm[3]);
        sz_sha256_round_haswell_(state_d_ymm, state_e_ymm, state_f_ymm, &state_g_ymm, state_h_ymm, state_a_ymm,
                                 state_b_ymm, &state_c_ymm, schedule_ymm[5], turn_constants[5]);
        schedule_ymm[6] = sz_sha256_extend_haswell_(schedule_ymm[6], schedule_ymm[7], schedule_ymm[15],
                                                    schedule_ymm[4]);
        sz_sha256_round_haswell_(state_c_ymm, state_d_ymm, state_e_ymm, &state_f_ymm, state_g_ymm, state_h_ymm,
                                 state_a_ymm, &state_b_ymm, schedule_ymm[6], turn_constants[6]);
        schedule_ymm[7] = sz_sha256_extend_haswell_(schedule_ymm[7], schedule_ymm[8], schedule_ymm[0], schedule_ymm[5]);
        sz_sha256_round_haswell_(state_b_ymm, state_c_ymm, state_d_ymm, &state_e_ymm, state_f_ymm, state_g_ymm,
                                 state_h_ymm, &state_a_ymm, schedule_ymm[7], turn_constants[7]);
        schedule_ymm[8] = sz_sha256_extend_haswell_(schedule_ymm[8], schedule_ymm[9], schedule_ymm[1], schedule_ymm[6]);
        sz_sha256_round_haswell_(state_a_ymm, state_b_ymm, state_c_ymm, &state_d_ymm, state_e_ymm, state_f_ymm,
                                 state_g_ymm, &state_h_ymm, schedule_ymm[8], turn_constants[8]);
        schedule_ymm[9] = sz_sha256_extend_haswell_(schedule_ymm[9], schedule_ymm[10], schedule_ymm[2],
                                                    schedule_ymm[7]);
        sz_sha256_round_haswell_(state_h_ymm, state_a_ymm, state_b_ymm, &state_c_ymm, state_d_ymm, state_e_ymm,
                                 state_f_ymm, &state_g_ymm, schedule_ymm[9], turn_constants[9]);
        schedule_ymm[10] = sz_sha256_extend_haswell_(schedule_ymm[10], schedule_ymm[11], schedule_ymm[3],
                                                     schedule_ymm[8]);
        sz_sha256_round_haswell_(state_g_ymm, state_h_ymm, state_a_ymm, &state_b_ymm, state_c_ymm, state_d_ymm,
                                 state_e_ymm, &state_f_ymm, schedule_ymm[10], turn_constants[10]);
        schedule_ymm[11] = sz_sha256_extend_haswell_(schedule_ymm[11], schedule_ymm[12], schedule_ymm[4],
                                                     schedule_ymm[9]);
        sz_sha256_round_haswell_(state_f_ymm, state_g_ymm, state_h_ymm, &state_a_ymm, state_b_ymm, state_c_ymm,
                                 state_d_ymm, &state_e_ymm, schedule_ymm[11], turn_constants[11]);
        schedule_ymm[12] = sz_sha256_extend_haswell_(schedule_ymm[12], schedule_ymm[13], schedule_ymm[5],
                                                     schedule_ymm[10]);
        sz_sha256_round_haswell_(state_e_ymm, state_f_ymm, state_g_ymm, &state_h_ymm, state_a_ymm, state_b_ymm,
                                 state_c_ymm, &state_d_ymm, schedule_ymm[12], turn_constants[12]);
        schedule_ymm[13] = sz_sha256_extend_haswell_(schedule_ymm[13], schedule_ymm[14], schedule_ymm[6],
                                                     schedule_ymm[11]);
        sz_sha256_round_haswell_(state_d_ymm, state_e_ymm, state_f_ymm, &state_g_ymm, state_h_ymm, state_a_ymm,
                                 state_b_ymm, &state_c_ymm, schedule_ymm[13], turn_constants[13]);
        schedule_ymm[14] = sz_sha256_extend_haswell_(schedule_ymm[14], schedule_ymm[15], schedule_ymm[7],
                                                     schedule_ymm[12]);
        sz_sha256_round_haswell_(state_c_ymm, state_d_ymm, state_e_ymm, &state_f_ymm, state_g_ymm, state_h_ymm,
                                 state_a_ymm, &state_b_ymm, schedule_ymm[14], turn_constants[14]);
        schedule_ymm[15] = sz_sha256_extend_haswell_(schedule_ymm[15], schedule_ymm[0], schedule_ymm[8],
                                                     schedule_ymm[13]);
        sz_sha256_round_haswell_(state_b_ymm, state_c_ymm, state_d_ymm, &state_e_ymm, state_f_ymm, state_g_ymm,
                                 state_h_ymm, &state_a_ymm, schedule_ymm[15], turn_constants[15]);
    }

    hashes_ymm[0] = _mm256_add_epi32(hashes_ymm[0], state_a_ymm);
    hashes_ymm[1] = _mm256_add_epi32(hashes_ymm[1], state_b_ymm);
    hashes_ymm[2] = _mm256_add_epi32(hashes_ymm[2], state_c_ymm);
    hashes_ymm[3] = _mm256_add_epi32(hashes_ymm[3], state_d_ymm);
    hashes_ymm[4] = _mm256_add_epi32(hashes_ymm[4], state_e_ymm);
    hashes_ymm[5] = _mm256_add_epi32(hashes_ymm[5], state_f_ymm);
    hashes_ymm[6] = _mm256_add_epi32(hashes_ymm[6], state_g_ymm);
    hashes_ymm[7] = _mm256_add_epi32(hashes_ymm[7], state_h_ymm);
}

/**
 *  @brief Compresses @p blocks_count whole blocks for 8 already block-aligned lanes.
 *  @param states The 8 lane states, whose hash words are gathered on entry and scattered on exit.
 *  @param cursors Per-lane read positions, advanced past everything consumed.
 *  @param blocks_count Whole 64-byte blocks available in every one of the 8 lanes.
 */
SZ_HELPER_AUTO void sz_sha256_multistate_blocks_haswell_(sz_sha256_state_t *states, sz_u8_t const **cursors,
                                                         sz_size_t blocks_count) {
    sz_u256_vec_t interleaved_vec[8];
    __m256i hashes_ymm[8];
    for (sz_size_t word_index = 0; word_index != 8; ++word_index)
        for (sz_size_t lane_index = 0; lane_index != 8; ++lane_index)
            interleaved_vec[word_index].u32s[lane_index] = states[lane_index].hash[word_index];
    for (sz_size_t word_index = 0; word_index != 8; ++word_index)
        hashes_ymm[word_index] = interleaved_vec[word_index].ymm;

    for (sz_size_t block_index = 0; block_index != blocks_count; ++block_index) {
        sz_sha256_compress_haswell_(hashes_ymm, cursors);
        for (sz_size_t lane_index = 0; lane_index != 8; ++lane_index) cursors[lane_index] += 64;
    }

    for (sz_size_t word_index = 0; word_index != 8; ++word_index)
        interleaved_vec[word_index].ymm = hashes_ymm[word_index];
    for (sz_size_t lane_index = 0; lane_index != 8; ++lane_index) {
        for (sz_size_t word_index = 0; word_index != 8; ++word_index)
            states[lane_index].hash[word_index] = interleaved_vec[word_index].u32s[lane_index];
        states[lane_index].total_length += blocks_count * 64;
    }
}

SZ_API_COMPTIME void sz_sha256_multistate_update_haswell(sz_sha256_state_t *states, sz_sequence_t const *texts) {
    sz_size_t const lanes_count = texts->count;
    sz_size_t first_lane_index = 0;

    for (; first_lane_index + 8 <= lanes_count; first_lane_index += 8) {
        sz_u8_t const *cursors[8];
        sz_size_t remaining[8];
        sz_size_t shared_blocks = (sz_size_t)-1;

        /*  Top up any buffered partial block first, so every lane that still holds a whole block is aligned.
         *  A lane too short to fill its block keeps fewer than 64 bytes, which drives the shared count to
         *  zero on its own and keeps the wide path off. */
        for (sz_size_t lane_index = 0; lane_index != 8; ++lane_index) {
            sz_sha256_state_t *const state = &states[first_lane_index + lane_index];
            cursors[lane_index] = (sz_u8_t const *)texts->get_start(texts->handle, first_lane_index + lane_index);
            remaining[lane_index] = texts->get_length(texts->handle, first_lane_index + lane_index);
            if (state->block_length != 0) {
                sz_size_t const missing = 64 - state->block_length;
                if (remaining[lane_index] >= missing) {
                    sz_sha256_state_update_serial(state, (sz_cptr_t)cursors[lane_index], missing);
                    cursors[lane_index] += missing, remaining[lane_index] -= missing;
                }
            }
            if (remaining[lane_index] / 64 < shared_blocks) shared_blocks = remaining[lane_index] / 64;
        }

        if (shared_blocks != 0) {
            sz_sha256_multistate_blocks_haswell_(&states[first_lane_index], cursors, shared_blocks);
            for (sz_size_t lane_index = 0; lane_index != 8; ++lane_index) remaining[lane_index] -= shared_blocks * 64;
        }

        for (sz_size_t lane_index = 0; lane_index != 8; ++lane_index)
            if (remaining[lane_index] != 0)
                sz_sha256_state_update_serial(&states[first_lane_index + lane_index], (sz_cptr_t)cursors[lane_index],
                                              remaining[lane_index]);
    }

    for (; first_lane_index != lanes_count; ++first_lane_index)
        sz_sha256_state_update_serial(&states[first_lane_index], texts->get_start(texts->handle, first_lane_index),
                                      texts->get_length(texts->handle, first_lane_index));
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
SZ_HELPER_AUTO void sz_sha256_multistate_digest_lanes_haswell_(sz_sha256_state_t const *states, sz_size_t active_lanes_count,
                                                               sz_u8_t *digests) {
    /*  A SHA256 block is 64 bytes whatever the vector width, so the staged blocks are 512-bit unions even
     *  though the lanes themselves are 256-bit wide. */
    sz_u512_vec_t carrier_vec[8], final_vec[8];
    sz_u256_vec_t interleaved_vec[8], overflow_vec;
    sz_u8_t const *carrier_blocks[8], *final_blocks[8];
    sz_bool_t any_overflow = sz_false_k;

    for (sz_size_t lane_index = 0; lane_index != 8; ++lane_index) {
        sz_size_t const source_lane = lane_index < active_lanes_count ? lane_index : 0;
        sz_sha256_state_t const *const state = &states[source_lane];
        sz_size_t const buffered = state->block_length;
        sz_u64_t const bit_length = state->total_length * 8;
        for (sz_size_t byte_index = 0; byte_index != 64; ++byte_index)
            carrier_vec[lane_index].u8s[byte_index] = 0, final_vec[lane_index].u8s[byte_index] = 0;
        overflow_vec.u32s[lane_index] = 0;

        /*  The terminator lands in a carrier block whenever it would crowd out the trailing bit length. */
        if (buffered + 1 > 56) {
            overflow_vec.u32s[lane_index] = 0xFFFFFFFFu, any_overflow = sz_true_k;
            for (sz_size_t byte_index = 0; byte_index != buffered; ++byte_index)
                carrier_vec[lane_index].u8s[byte_index] = state->block[byte_index];
            carrier_vec[lane_index].u8s[buffered] = 0x80;
        }
        else {
            for (sz_size_t byte_index = 0; byte_index != buffered; ++byte_index)
                final_vec[lane_index].u8s[byte_index] = state->block[byte_index];
            final_vec[lane_index].u8s[buffered] = 0x80;
        }
        for (sz_size_t byte_index = 0; byte_index != 8; ++byte_index)
            final_vec[lane_index].u8s[56 + byte_index] = (sz_u8_t)(bit_length >> (56 - byte_index * 8));

        carrier_blocks[lane_index] = carrier_vec[lane_index].u8s;
        final_blocks[lane_index] = final_vec[lane_index].u8s;
        for (sz_size_t word_index = 0; word_index != 8; ++word_index)
            interleaved_vec[word_index].u32s[lane_index] = state->hash[word_index];
    }

    __m256i hashes_ymm[8];
    for (sz_size_t word_index = 0; word_index != 8; ++word_index)
        hashes_ymm[word_index] = interleaved_vec[word_index].ymm;

    if (any_overflow == sz_true_k) {
        __m256i unchanged_ymm[8];
        for (sz_size_t word_index = 0; word_index != 8; ++word_index)
            unchanged_ymm[word_index] = hashes_ymm[word_index];
        sz_sha256_compress_haswell_(hashes_ymm, carrier_blocks);
        for (sz_size_t word_index = 0; word_index != 8; ++word_index)
            hashes_ymm[word_index] = _mm256_blendv_epi8(unchanged_ymm[word_index], hashes_ymm[word_index],
                                                        overflow_vec.ymm);
    }
    sz_sha256_compress_haswell_(hashes_ymm, final_blocks);

    for (sz_size_t word_index = 0; word_index != 8; ++word_index)
        interleaved_vec[word_index].ymm = hashes_ymm[word_index];
    for (sz_size_t lane_index = 0; lane_index != active_lanes_count; ++lane_index)
        for (sz_size_t word_index = 0; word_index != 8; ++word_index) {
            sz_u32_t const word = interleaved_vec[word_index].u32s[lane_index];
            sz_u8_t *const target = &digests[lane_index * 32 + word_index * 4];
            target[0] = (sz_u8_t)(word >> 24), target[1] = (sz_u8_t)(word >> 16);
            target[2] = (sz_u8_t)(word >> 8), target[3] = (sz_u8_t)(word >> 0);
        }
}

SZ_API_COMPTIME void sz_sha256_multistate_digest_haswell(sz_sha256_state_t const *states, sz_size_t states_count,
                                                         sz_u8_t *digests) {
    sz_size_t first_lane_index = 0;
    for (; first_lane_index < states_count; first_lane_index += 8) {
        sz_size_t const remaining = states_count - first_lane_index;
        sz_size_t const active_lanes_count = remaining < 8 ? remaining : 8;
        sz_sha256_multistate_digest_lanes_haswell_(&states[first_lane_index], active_lanes_count, &digests[first_lane_index * 32]);
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
