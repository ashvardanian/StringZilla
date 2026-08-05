/**
 *  @brief Skylake (AVX-512) backend for string hashing and checksums.
 *  @file include/stringzilla/hash/skylake.h
 *  @author Ash Vardanian
 *  @sa include/stringzilla/hash.h
 */
#ifndef STRINGZILLA_HASH_SKYLAKE_H_
#define STRINGZILLA_HASH_SKYLAKE_H_

#include "stringzilla/types.h"
#include "stringzilla/compare.h" // `sz_equal`
#include "stringzilla/hash/serial.h"

#ifdef __cplusplus
extern "C" {
#endif

#if SZ_USE_SKYLAKE
#if defined(__clang__) && SZ_CLANG_HAS_EVEX512_
#pragma clang attribute push(__attribute__((target("avx,avx512f,avx512vl,avx512bw,bmi,bmi2,aes,evex512"))), \
                             apply_to = function)
#elif defined(__clang__)
#pragma clang attribute push(__attribute__((target("avx,avx512f,avx512vl,avx512bw,bmi,bmi2,aes"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("avx", "avx512f", "avx512vl", "avx512bw", "bmi", "bmi2", "aes")
#endif

SZ_API_COMPTIME sz_u64_t sz_bytesum_skylake(sz_cptr_t text, sz_size_t length) {
    // The naive implementation of this function is very simple.
    // It assumes the CPU is great at handling unaligned "loads".
    //
    // A typical AWS Sapphire Rapids instance can have 48 KB x 2 blocks of L1 data cache per core,
    // 2 MB x 2 blocks of L2 cache per core, and one shared 60 MB buffer of L3 cache.
    // With two strings, we may consider the overall workload huge, if each exceeds 1 MB in length.
    int const is_huge = length >= 1ull * 1024ull * 1024ull;
    sz_u512_vec_t text_vec, sums_vec;

    // When the buffer is small, there isn't much to innovate.
    // Separately handling even smaller payloads doesn't increase performance even on synthetic benchmarks.
    if (length <= 16) {
        __mmask16 mask_m16 = sz_u16_mask_until_(length);
        text_vec.xmms[0] = _mm_maskz_loadu_epi8(mask_m16, text);
        sums_vec.xmms[0] = _mm_sad_epu8(text_vec.xmms[0], _mm_setzero_si128());
        sz_u64_t low = (sz_u64_t)_mm_cvtsi128_si64(sums_vec.xmms[0]);
        sz_u64_t high = (sz_u64_t)_mm_extract_epi64(sums_vec.xmms[0], 1);
        return low + high;
    }
    else if (length <= 32) {
        __mmask32 mask_m32 = sz_u32_mask_until_(length);
        text_vec.ymms[0] = _mm256_maskz_loadu_epi8(mask_m32, text);
        sums_vec.ymms[0] = _mm256_sad_epu8(text_vec.ymms[0], _mm256_setzero_si256());
        // Accumulating 256 bits is harder, as we need to extract the 128-bit sums first.
        __m128i low_u64x2 = _mm256_castsi256_si128(sums_vec.ymms[0]);
        __m128i high_u64x2 = _mm256_extracti128_si256(sums_vec.ymms[0], 1);
        __m128i sums_u64x2 = _mm_add_epi64(low_u64x2, high_u64x2);
        sz_u64_t low = (sz_u64_t)_mm_cvtsi128_si64(sums_u64x2);
        sz_u64_t high = (sz_u64_t)_mm_extract_epi64(sums_u64x2, 1);
        return low + high;
    }
    else if (length <= 64) {
        __mmask64 mask_m64 = sz_u64_mask_until_(length);
        text_vec.zmm = _mm512_maskz_loadu_epi8(mask_m64, text);
        sums_vec.zmm = _mm512_sad_epu8(text_vec.zmm, _mm512_setzero_si512());
        return _mm512_reduce_add_epi64(sums_vec.zmm);
    }
    // For large buffers, fitting into L1 cache sizes, there are other tricks we can use.
    //
    // 1. Moving in both directions to maximize the throughput, when fetching from multiple
    //    memory pages. Also helps with cache set-associativity issues, as we won't always
    //    be fetching the same buckets in the lookup table.
    //
    // Bidirectional traversal generally adds about 10% to such algorithms.
    else if (!is_huge) {
        sz_size_t head_length = (64 - ((sz_size_t)text % 64)) % 64; // 63 or less.
        sz_size_t tail_length = (sz_size_t)(text + length) % 64;    // 63 or less.
        sz_size_t body_length = length - head_length - tail_length; // Multiple of 64.
        sz_assert_(body_length % 64 == 0 && head_length < 64 && tail_length < 64);
        __mmask64 head_mask_m64 = sz_u64_mask_until_(head_length);
        __mmask64 tail_mask_m64 = sz_u64_mask_until_(tail_length);

        text_vec.zmm = _mm512_maskz_loadu_epi8(head_mask_m64, text);
        sums_vec.zmm = _mm512_sad_epu8(text_vec.zmm, _mm512_setzero_si512());
        for (text += head_length; body_length >= 64; text += 64, body_length -= 64) {
            text_vec.zmm = _mm512_load_si512((__m512i const *)text);
            sums_vec.zmm = _mm512_add_epi64(sums_vec.zmm, _mm512_sad_epu8(text_vec.zmm, _mm512_setzero_si512()));
        }
        text_vec.zmm = _mm512_maskz_loadu_epi8(tail_mask_m64, text);
        sums_vec.zmm = _mm512_add_epi64(sums_vec.zmm, _mm512_sad_epu8(text_vec.zmm, _mm512_setzero_si512()));
        return _mm512_reduce_add_epi64(sums_vec.zmm);
    }
    // For gigantic buffers, exceeding typical L1 cache sizes, there are other tricks we can use.
    //
    // 1. Using non-temporal loads to avoid polluting the cache.
    // 2. Prefetching the next cache line, to avoid stalling the CPU. This generally useless
    //    for predictable patterns, so disregard this advice.
    //
    // Bidirectional traversal generally adds about 10% to such algorithms.
    else {
        sz_u512_vec_t text_reversed_vec, sums_reversed_vec;
        sz_size_t head_length = (64 - ((sz_size_t)text % 64)) % 64;
        sz_size_t tail_length = (sz_size_t)(text + length) % 64;
        sz_size_t body_length = length - head_length - tail_length;
        __mmask64 head_mask_m64 = sz_u64_mask_until_(head_length);
        __mmask64 tail_mask_m64 = sz_u64_mask_until_(tail_length);

        text_vec.zmm = _mm512_maskz_loadu_epi8(head_mask_m64, text);
        sums_vec.zmm = _mm512_sad_epu8(text_vec.zmm, _mm512_setzero_si512());
        text_reversed_vec.zmm = _mm512_maskz_loadu_epi8(tail_mask_m64, text + head_length + body_length);
        sums_reversed_vec.zmm = _mm512_sad_epu8(text_reversed_vec.zmm, _mm512_setzero_si512());

        // Now in the main loop, we can use non-temporal loads, performing the operation in both directions.
        for (text += head_length; body_length >= 128; text += 64, body_length -= 128) {
            text_vec.zmm = _mm512_stream_load_si512((__m512i *)(text));
            sums_vec.zmm = _mm512_add_epi64(sums_vec.zmm, _mm512_sad_epu8(text_vec.zmm, _mm512_setzero_si512()));
            text_reversed_vec.zmm = _mm512_stream_load_si512((__m512i *)(text + body_length - 64));
            sums_reversed_vec.zmm = _mm512_add_epi64(sums_reversed_vec.zmm,
                                                     _mm512_sad_epu8(text_reversed_vec.zmm, _mm512_setzero_si512()));
        }
        if (body_length >= 64) {
            text_vec.zmm = _mm512_stream_load_si512((__m512i *)(text));
            sums_vec.zmm = _mm512_add_epi64(sums_vec.zmm, _mm512_sad_epu8(text_vec.zmm, _mm512_setzero_si512()));
        }

        return _mm512_reduce_add_epi64(_mm512_add_epi64(sums_vec.zmm, sums_reversed_vec.zmm));
    }
}

SZ_API_COMPTIME void sz_hash_state_init_skylake(sz_hash_state_t *state, sz_u64_t seed) {
    // The key is made from the seed and half of it will be mixed with the length in the end
    __m512i seed_u64x8 = _mm512_set1_epi64(seed);
    // ! In this kernel, assuming it may be called on arbitrarily misaligned `state`,
    // ! we must use `_mm_storeu_si128` stores to update the state.
    _mm_storeu_si128((__m128i *)state->key, _mm512_castsi512_si128(seed_u64x8));

    // XOR the user-supplied keys with the two "pi" constants
    sz_u64_t const *pi = sz_hash_pi_constants_();
    __m512i const pi0_u64x8 = _mm512_load_epi64((__m512i const *)(pi));
    __m512i const pi1_u64x8 = _mm512_load_epi64((__m512i const *)(pi + 8));
    _mm512_storeu_si512((__m512i *)state->aes, _mm512_xor_si512(seed_u64x8, pi0_u64x8));
    _mm512_storeu_si512((__m512i *)state->sum, _mm512_xor_si512(seed_u64x8, pi1_u64x8));

    // The inputs are zeroed out at the beginning
    _mm512_storeu_si512((__m512i *)state->ins, _mm512_setzero_si512());
    state->ins_length = 0;
}

SZ_API_COMPTIME SZ_NO_STACK_PROTECTOR sz_u64_t sz_hash_skylake(sz_cptr_t start, sz_size_t length, sz_u64_t seed) {

    if (length <= 16) {
        // Initialize the AES block with a given seed
        sz_align_(16) sz_hash_state_aligned_for_short_t state;
        sz_hash_state_short_init_westmere_aligned_(&state, seed);

        // Load the data and update the state
        sz_u128_vec_t data_vec;
        data_vec.xmm = _mm_maskz_loadu_epi8(sz_u16_mask_until_(length), start);

        // Shuffle with the same mask
        __m128i const order_u8x16 = _mm_load_si128((__m128i const *)sz_hash_u8x16x4_shuffle_());
        sz_hash_state_short_update_westmere_aligned_(&state, data_vec.xmm, order_u8x16);
        return sz_hash_state_short_finalize_westmere_aligned_(&state, length);
    }
    else if (length <= 32) {
        // Initialize the AES block with a given seed
        sz_align_(16) sz_hash_state_aligned_for_short_t state;
        sz_hash_state_short_init_westmere_aligned_(&state, seed);

        // Load the data and update the state
        sz_u128_vec_t data0_vec, data1_vec;
        data0_vec.xmm = _mm_lddqu_si128((__m128i const *)(start));
        data1_vec.xmm = _mm_maskz_loadu_epi8(sz_u16_mask_until_(length - 16), start + 16);

        // Shuffle with the same mask
        __m128i const order_u8x16 = _mm_load_si128((__m128i const *)sz_hash_u8x16x4_shuffle_());
        sz_hash_state_short_update_westmere_aligned_(&state, data0_vec.xmm, order_u8x16);
        sz_hash_state_short_update_westmere_aligned_(&state, data1_vec.xmm, order_u8x16);
        return sz_hash_state_short_finalize_westmere_aligned_(&state, length);
    }
    else if (length <= 48) {
        // Initialize the AES block with a given seed
        sz_align_(16) sz_hash_state_aligned_for_short_t state;
        sz_hash_state_short_init_westmere_aligned_(&state, seed);

        // Load the data and update the state
        sz_u128_vec_t data0_vec, data1_vec, data2_vec;
        data0_vec.xmm = _mm_lddqu_si128((__m128i const *)(start));
        data1_vec.xmm = _mm_lddqu_si128((__m128i const *)(start + 16));
        data2_vec.xmm = _mm_maskz_loadu_epi8(sz_u16_mask_until_(length - 32), start + 32);

        // Shuffle with the same mask
        __m128i const order_u8x16 = _mm_load_si128((__m128i const *)sz_hash_u8x16x4_shuffle_());
        sz_hash_state_short_update_westmere_aligned_(&state, data0_vec.xmm, order_u8x16);
        sz_hash_state_short_update_westmere_aligned_(&state, data1_vec.xmm, order_u8x16);
        sz_hash_state_short_update_westmere_aligned_(&state, data2_vec.xmm, order_u8x16);
        return sz_hash_state_short_finalize_westmere_aligned_(&state, length);
    }
    else if (length <= 64) {
        // Initialize the AES block with a given seed
        sz_align_(16) sz_hash_state_aligned_for_short_t state;
        sz_hash_state_short_init_westmere_aligned_(&state, seed);

        // Load the data and update the state
        sz_u128_vec_t data0_vec, data1_vec, data2_vec, data3_vec;
        data0_vec.xmm = _mm_lddqu_si128((__m128i const *)(start));
        data1_vec.xmm = _mm_lddqu_si128((__m128i const *)(start + 16));
        data2_vec.xmm = _mm_lddqu_si128((__m128i const *)(start + 32));
        data3_vec.xmm = _mm_maskz_loadu_epi8(sz_u16_mask_until_(length - 48), start + 48);

        // Shuffle with the same mask
        __m128i const order_u8x16 = _mm_load_si128((__m128i const *)sz_hash_u8x16x4_shuffle_());
        sz_hash_state_short_update_westmere_aligned_(&state, data0_vec.xmm, order_u8x16);
        sz_hash_state_short_update_westmere_aligned_(&state, data1_vec.xmm, order_u8x16);
        sz_hash_state_short_update_westmere_aligned_(&state, data2_vec.xmm, order_u8x16);
        sz_hash_state_short_update_westmere_aligned_(&state, data3_vec.xmm, order_u8x16);
        return sz_hash_state_short_finalize_westmere_aligned_(&state, length);
    }
    // Skylake has no VAES, so its four-lane AES-NI absorb has no throughput edge over Westmere; in 512-bit form it
    // is in fact slower (per-lane `vextracti128` contends on the shuffle port). Skylake's win is the masked-load
    // short path above, so inputs over one block defer to the full-clock pure-SSE Westmere kernel.
    return sz_hash_westmere(start, length, seed);
}

SZ_API_COMPTIME void sz_hash_state_update_skylake(sz_hash_state_t *state_ptr, sz_cptr_t text, sz_size_t length) {
    // Skylake has AVX-512BW but neither VBMI (no `vpermb` byte slide) nor VAES, so the absorb stays the four-lane
    // AES-NI Westmere kernel. What Westmere lacks is a masked load: it merges incoming bytes one at a time into
    // `ins.u8s[...]`, then reads `ins` back as wide lanes - a ~12-cycle store-forwarding stall on every cross-call
    // merge, the dominant cost for short streamed tokens. AVX-512 removes it without VBMI: a single fault-suppressed
    // masked load from `text - buffered` lands the incoming bytes directly at their buffer offset. The masked-off
    // low `buffered` lanes alias the bytes before `text` and are never accessed; the set lanes
    // [buffered, buffered + to_copy) read exactly [text, text + to_copy). One `vpblendmb` drops them into `ins`;
    // bit-identical digest to the per-byte copy.
    sz_hash_state_aligned_t state = sz_hash_state_load_westmere_(state_ptr);
    sz_size_t buffered = state.ins_length % 64;
    if (buffered == 0 && state.ins_length) buffered = 64;
    while (length) {
        if (buffered == 64) { // the deferred block is now interior - absorb it (4x AES-NI) and re-zero the buffer
            sz_hash_state_update_westmere_(&state);
            state.ins.zmm = _mm512_setzero_si512();
            buffered = 0;
        }
        sz_size_t const to_copy = sz_min_of_two(length, (sz_size_t)64 - buffered);
        __mmask64 const place_mask_m64 = _cvtu64_mask64(sz_u64_mask_until_(to_copy) << buffered);
        __m512i const incoming_u8x64 = _mm512_maskz_loadu_epi8(place_mask_m64, (void const *)(text - buffered));
        state.ins.zmm = _mm512_mask_blend_epi8(place_mask_m64, state.ins.zmm, incoming_u8x64);
        buffered += to_copy, text += to_copy, length -= to_copy, state.ins_length += to_copy;
    }
    sz_hash_state_store_westmere_(state_ptr, &state);
}

SZ_API_COMPTIME sz_u64_t sz_hash_state_digest_skylake(sz_hash_state_t const *state) {
    // ? We don't know a better way to fold the state on Ice Lake, than to use the Haswell implementation.
    return sz_hash_state_digest_westmere(state);
}

SZ_API_COMPTIME void sz_fill_random_skylake(sz_ptr_t text, sz_size_t length, sz_u64_t nonce) {
    sz_fill_random_westmere(text, length, nonce);
}

/*  `vpternlogd` collapses each bitwise primitive of SHA256 to a single instruction: `choice` is the select
 *  truth table, `majority` is the majority truth table, and the three-way exclusive or that both sigma
 *  families need is the parity truth table. */

/** @brief Evaluates `(state_e & state_f) ^ (~state_e & state_g)` across 16 lanes. */
SZ_HELPER_INLINE __m512i sz_sha256_choice_skylake_(__m512i state_e_u32x16, __m512i state_f_u32x16, __m512i state_g_u32x16) {
    return _mm512_ternarylogic_epi32(state_e_u32x16, state_f_u32x16, state_g_u32x16, 0xCA);
}

/** @brief Evaluates `(state_a & state_b) ^ (state_a & state_c) ^ (state_b & state_c)` across 16 lanes. */
SZ_HELPER_INLINE __m512i sz_sha256_majority_skylake_(__m512i state_a_u32x16, __m512i state_b_u32x16, __m512i state_c_u32x16) {
    return _mm512_ternarylogic_epi32(state_a_u32x16, state_b_u32x16, state_c_u32x16, 0xE8);
}

/** @brief Evaluates `ror(state_a, 2) ^ ror(state_a, 13) ^ ror(state_a, 22)` across 16 lanes. */
SZ_HELPER_INLINE __m512i sz_sha256_big_sigma0_skylake_(__m512i state_a_u32x16) {
    return _mm512_ternarylogic_epi32(_mm512_ror_epi32(state_a_u32x16, 2), _mm512_ror_epi32(state_a_u32x16, 13),
                                     _mm512_ror_epi32(state_a_u32x16, 22), 0x96);
}

/** @brief Evaluates `ror(state_e, 6) ^ ror(state_e, 11) ^ ror(state_e, 25)` across 16 lanes. */
SZ_HELPER_INLINE __m512i sz_sha256_big_sigma1_skylake_(__m512i state_e_u32x16) {
    return _mm512_ternarylogic_epi32(_mm512_ror_epi32(state_e_u32x16, 6), _mm512_ror_epi32(state_e_u32x16, 11),
                                     _mm512_ror_epi32(state_e_u32x16, 25), 0x96);
}

/** @brief Evaluates `ror(word, 7) ^ ror(word, 18) ^ (word >> 3)` across 16 lanes. */
SZ_HELPER_INLINE __m512i sz_sha256_small_sigma0_skylake_(__m512i message_word_u32x16) {
    return _mm512_ternarylogic_epi32(_mm512_ror_epi32(message_word_u32x16, 7), _mm512_ror_epi32(message_word_u32x16, 18),
                                     _mm512_srli_epi32(message_word_u32x16, 3), 0x96);
}

/** @brief Evaluates `ror(word, 17) ^ ror(word, 19) ^ (word >> 10)` across 16 lanes. */
SZ_HELPER_INLINE __m512i sz_sha256_small_sigma1_skylake_(__m512i message_word_u32x16) {
    return _mm512_ternarylogic_epi32(_mm512_ror_epi32(message_word_u32x16, 17), _mm512_ror_epi32(message_word_u32x16, 19),
                                     _mm512_srli_epi32(message_word_u32x16, 10), 0x96);
}

/**
 *  @brief Exchanges eight 16-lane registers with sixteen 8-word lanes, in either direction.
 *  @param words Eight registers, rewritten in place.
 *
 *  Going in, register `pair_index` holds lane `pair_index` in its low half and lane `pair_index + 8` in its
 *  high half; coming out, register `word_index` holds that word from all 16 lanes. The butterfly is its own
 *  inverse, so the same sequence serves the gather and the scatter and there is only one thing to get right.
 *
 *  Built from `vpunpck` and `vpermt2d` rather than `vpgatherdd` and `vpscatterdd`: a 512-bit scatter is 36
 *  uops on Ice Lake and Sapphire Rapids, the shuffle form measures level with it there, and no cost model
 *  available for AMD parts describes their gather and scatter at all - so the form built from ordinary
 *  shuffles is the one whose cost is knowable everywhere.
 */
SZ_HELPER_INLINE void sz_sha256_transpose_8x16_skylake_(__m512i words_u32x16[8]) {
    __m512i const low_halves_u32x16 = _mm512_setr_epi32(0, 1, 2, 3, 16, 17, 18, 19, 8, 9, 10, 11, 24, 25, 26, 27);
    __m512i const high_halves_u32x16 = _mm512_setr_epi32(4, 5, 6, 7, 20, 21, 22, 23, 12, 13, 14, 15, 28, 29, 30, 31);
    // Pair 32-bit neighbours inside every 128-bit sub-lane.
    __m512i const paired0_u32x16 = _mm512_unpacklo_epi32(words_u32x16[0], words_u32x16[1]);
    __m512i const paired1_u32x16 = _mm512_unpackhi_epi32(words_u32x16[0], words_u32x16[1]);
    __m512i const paired2_u32x16 = _mm512_unpacklo_epi32(words_u32x16[2], words_u32x16[3]);
    __m512i const paired3_u32x16 = _mm512_unpackhi_epi32(words_u32x16[2], words_u32x16[3]);
    __m512i const paired4_u32x16 = _mm512_unpacklo_epi32(words_u32x16[4], words_u32x16[5]);
    __m512i const paired5_u32x16 = _mm512_unpackhi_epi32(words_u32x16[4], words_u32x16[5]);
    __m512i const paired6_u32x16 = _mm512_unpacklo_epi32(words_u32x16[6], words_u32x16[7]);
    __m512i const paired7_u32x16 = _mm512_unpackhi_epi32(words_u32x16[6], words_u32x16[7]);

    // Pair those into 64-bit neighbours, still inside each 128-bit sub-lane.
    __m512i const quadded0_u32x16 = _mm512_unpacklo_epi64(paired0_u32x16, paired2_u32x16);
    __m512i const quadded1_u32x16 = _mm512_unpackhi_epi64(paired0_u32x16, paired2_u32x16);
    __m512i const quadded2_u32x16 = _mm512_unpacklo_epi64(paired1_u32x16, paired3_u32x16);
    __m512i const quadded3_u32x16 = _mm512_unpackhi_epi64(paired1_u32x16, paired3_u32x16);
    __m512i const quadded4_u32x16 = _mm512_unpacklo_epi64(paired4_u32x16, paired6_u32x16);
    __m512i const quadded5_u32x16 = _mm512_unpackhi_epi64(paired4_u32x16, paired6_u32x16);
    __m512i const quadded6_u32x16 = _mm512_unpacklo_epi64(paired5_u32x16, paired7_u32x16);
    __m512i const quadded7_u32x16 = _mm512_unpackhi_epi64(paired5_u32x16, paired7_u32x16);

    // Exchange whole quarters across the register, which is the only stage that crosses 128-bit boundaries.
    words_u32x16[0] = _mm512_permutex2var_epi32(quadded0_u32x16, low_halves_u32x16, quadded4_u32x16);
    words_u32x16[4] = _mm512_permutex2var_epi32(quadded0_u32x16, high_halves_u32x16, quadded4_u32x16);
    words_u32x16[1] = _mm512_permutex2var_epi32(quadded1_u32x16, low_halves_u32x16, quadded5_u32x16);
    words_u32x16[5] = _mm512_permutex2var_epi32(quadded1_u32x16, high_halves_u32x16, quadded5_u32x16);
    words_u32x16[2] = _mm512_permutex2var_epi32(quadded2_u32x16, low_halves_u32x16, quadded6_u32x16);
    words_u32x16[6] = _mm512_permutex2var_epi32(quadded2_u32x16, high_halves_u32x16, quadded6_u32x16);
    words_u32x16[3] = _mm512_permutex2var_epi32(quadded3_u32x16, low_halves_u32x16, quadded7_u32x16);
    words_u32x16[7] = _mm512_permutex2var_epi32(quadded3_u32x16, high_halves_u32x16, quadded7_u32x16);
}

/**
 *  @brief Transposes one 64-byte block from each of 16 lanes into word-major big-endian order.
 *  @param lane_blocks Pointers to 16 message blocks, one per lane.
 *  @param schedule Receives 16 registers, where `schedule[word_index]` holds that word from all 16 lanes.
 *
 *  Two interleave stages gather 32-bit and then 64-bit neighbours inside each 128-bit sub-lane, two sub-lane
 *  exchange stages then move whole quarters across the register. The big-endian byte swap folds into the
 *  initial load, so it costs nothing beyond the shuffle that would happen anyway.
 */
SZ_HELPER_INLINE void sz_sha256_transpose_16x16_skylake_(sz_u8_t const *const *lane_blocks, __m512i schedule_u32x16[16]) {
    __m512i const byte_swap_u8x64 = _mm512_set_epi8(                      //
        60, 61, 62, 63, 56, 57, 58, 59, 52, 53, 54, 55, 48, 49, 50, 51, //
        44, 45, 46, 47, 40, 41, 42, 43, 36, 37, 38, 39, 32, 33, 34, 35, //
        28, 29, 30, 31, 24, 25, 26, 27, 20, 21, 22, 23, 16, 17, 18, 19, //
        12, 13, 14, 15, 8, 9, 10, 11, 4, 5, 6, 7, 0, 1, 2, 3);

    __m512i lanes_u32x16[16], first_stage_u32x16[16], second_stage_u32x16[16];
    for (sz_size_t lane_index = 0; lane_index != 16; ++lane_index)
        lanes_u32x16[lane_index] = _mm512_shuffle_epi8(_mm512_loadu_si512(lane_blocks[lane_index]), byte_swap_u8x64);

    for (sz_size_t butterfly_index = 0; butterfly_index != 8; ++butterfly_index) {
        first_stage_u32x16[butterfly_index * 2 + 0] = _mm512_unpacklo_epi32(lanes_u32x16[butterfly_index * 2 + 0],
                                                                         lanes_u32x16[butterfly_index * 2 + 1]);
        first_stage_u32x16[butterfly_index * 2 + 1] = _mm512_unpackhi_epi32(lanes_u32x16[butterfly_index * 2 + 0],
                                                                         lanes_u32x16[butterfly_index * 2 + 1]);
    }
    for (sz_size_t butterfly_index = 0; butterfly_index != 4; ++butterfly_index) {
        sz_size_t const butterfly_base = butterfly_index * 4;
        second_stage_u32x16[butterfly_base + 0] = _mm512_unpacklo_epi64(first_stage_u32x16[butterfly_base + 0],
                                                                     first_stage_u32x16[butterfly_base + 2]);
        second_stage_u32x16[butterfly_base + 1] = _mm512_unpackhi_epi64(first_stage_u32x16[butterfly_base + 0],
                                                                     first_stage_u32x16[butterfly_base + 2]);
        second_stage_u32x16[butterfly_base + 2] = _mm512_unpacklo_epi64(first_stage_u32x16[butterfly_base + 1],
                                                                     first_stage_u32x16[butterfly_base + 3]);
        second_stage_u32x16[butterfly_base + 3] = _mm512_unpackhi_epi64(first_stage_u32x16[butterfly_base + 1],
                                                                     first_stage_u32x16[butterfly_base + 3]);
    }
    for (sz_size_t butterfly_index = 0; butterfly_index != 2; ++butterfly_index) {
        sz_size_t const butterfly_base = butterfly_index * 8;
        for (sz_size_t offset = 0; offset != 4; ++offset) {
            first_stage_u32x16[butterfly_base + offset] = _mm512_shuffle_i32x4(
                second_stage_u32x16[butterfly_base + offset], second_stage_u32x16[butterfly_base + offset + 4], 0x88);
            first_stage_u32x16[butterfly_base + offset + 4] = _mm512_shuffle_i32x4(
                second_stage_u32x16[butterfly_base + offset], second_stage_u32x16[butterfly_base + offset + 4], 0xDD);
        }
    }
    for (sz_size_t offset = 0; offset != 8; ++offset) {
        schedule_u32x16[offset] = _mm512_shuffle_i32x4(first_stage_u32x16[offset], first_stage_u32x16[offset + 8], 0x88);
        schedule_u32x16[offset + 8] = _mm512_shuffle_i32x4(first_stage_u32x16[offset], first_stage_u32x16[offset + 8], 0xDD);
    }
}

/**
 *  @brief Extends the rolling message window by one word across 16 lanes.
 *  @param oldest_word_u32x16 The word being replaced, sixteen rounds behind.
 *  @param next_word_u32x16 The word one position ahead, feeding the low sigma.
 *  @param ninth_word_u32x16 The word nine positions ahead.
 *  @param fourteenth_word_u32x16 The word fourteen positions ahead, feeding the high sigma.
 */
SZ_HELPER_INLINE __m512i sz_sha256_extend_skylake_(__m512i oldest_word_u32x16, __m512i next_word_u32x16,
                                                   __m512i ninth_word_u32x16, __m512i fourteenth_word_u32x16) {
    return _mm512_add_epi32(_mm512_add_epi32(oldest_word_u32x16, sz_sha256_small_sigma0_skylake_(next_word_u32x16)),
                            _mm512_add_epi32(ninth_word_u32x16, sz_sha256_small_sigma1_skylake_(fourteenth_word_u32x16)));
}

/**
 *  @brief Applies one SHA256 round to 16 lanes, writing the two working values that change.
 *
 *  The eight working variables rotate by one position per round, which the callers express by passing the
 *  same locals in a different order rather than by moving data. Only `state_d` and `state_h` are written:
 *  `state_d` becomes the next round's `state_e`, and `state_h` is dead on entry so it receives the next
 *  round's `state_a`.
 */
SZ_HELPER_INLINE void sz_sha256_round_skylake_(                                          //
    __m512i state_a_u32x16, __m512i state_b_u32x16, __m512i state_c_u32x16, __m512i *state_d_u32x16, //
    __m512i state_e_u32x16, __m512i state_f_u32x16, __m512i state_g_u32x16, __m512i *state_h_u32x16, //
    __m512i message_word_u32x16, sz_u32_t round_constant) {
    __m512i const temporary_first_u32x16 = _mm512_add_epi32( //
        _mm512_add_epi32(
            _mm512_add_epi32(*state_h_u32x16, sz_sha256_big_sigma1_skylake_(state_e_u32x16)),
            _mm512_add_epi32(sz_sha256_choice_skylake_(state_e_u32x16, state_f_u32x16, state_g_u32x16), message_word_u32x16)),
        _mm512_set1_epi32((int)round_constant));
    __m512i const temporary_second_u32x16 = _mm512_add_epi32(
        sz_sha256_big_sigma0_skylake_(state_a_u32x16), sz_sha256_majority_skylake_(state_a_u32x16, state_b_u32x16, state_c_u32x16));
    *state_d_u32x16 = _mm512_add_epi32(*state_d_u32x16, temporary_first_u32x16);
    *state_h_u32x16 = _mm512_add_epi32(temporary_first_u32x16, temporary_second_u32x16);
}

/**
 *  @brief Transposes and compresses one 64-byte block for each of 16 lanes into word-major hash registers.
 *  @param hashes Eight registers of word-major hash state, updated in place.
 *  @param lane_blocks Pointers to 16 message blocks, one per lane.
 *  @param active_m16 Lanes whose block counts; the rest keep their hash bit-for-bit.
 *
 *  Masking rides on the closing accumulation rather than on a saved copy and a blend. Compression ends in
 *  `hash += working`, so withholding that one add per word is all a lane needs to sit out, which costs no
 *  extra register at a point where eight state and sixteen schedule registers are already live. The rounds
 *  still run for an inactive lane and produce a value nobody reads, so its block pointer only has to be
 *  readable - `sz_sha256_state_t::block` is always 64 valid bytes and serves that purpose.
 *
 *  The message window is a local rather than a parameter on purpose: crossing a function boundary forces the
 *  compiler to treat it as memory that may alias, which spills the whole window and the round constants to
 *  the stack. Fused like this it stays in registers.
 *
 *  Written as one straight-line turn of sixteen rounds, then three more turns of the same shape with the
 *  window extension folded in. Sixteen rounds return the eight working variables to their original names and
 *  advance the window exactly one full turn, so every index below is a compile-time constant.
 */
SZ_HELPER_INLINE void sz_sha256_compress_skylake_(__m512i hashes_u32x16[8], sz_u8_t const *const *lane_blocks,
                                                  __mmask16 active_m16) {
    sz_u32_t const *round_constants = (sz_u32_t const *)sz_x86_hide_pointer_origin_(sz_sha256_round_constants_());
    __m512i schedule_u32x16[16];
    sz_sha256_transpose_16x16_skylake_(lane_blocks, schedule_u32x16);

    __m512i state_a_u32x16 = hashes_u32x16[0], state_b_u32x16 = hashes_u32x16[1];
    __m512i state_c_u32x16 = hashes_u32x16[2], state_d_u32x16 = hashes_u32x16[3];
    __m512i state_e_u32x16 = hashes_u32x16[4], state_f_u32x16 = hashes_u32x16[5];
    __m512i state_g_u32x16 = hashes_u32x16[6], state_h_u32x16 = hashes_u32x16[7];

    sz_sha256_round_skylake_(state_a_u32x16, state_b_u32x16, state_c_u32x16, &state_d_u32x16, state_e_u32x16, state_f_u32x16, state_g_u32x16,
                             &state_h_u32x16, schedule_u32x16[0], round_constants[0]);
    sz_sha256_round_skylake_(state_h_u32x16, state_a_u32x16, state_b_u32x16, &state_c_u32x16, state_d_u32x16, state_e_u32x16, state_f_u32x16,
                             &state_g_u32x16, schedule_u32x16[1], round_constants[1]);
    sz_sha256_round_skylake_(state_g_u32x16, state_h_u32x16, state_a_u32x16, &state_b_u32x16, state_c_u32x16, state_d_u32x16, state_e_u32x16,
                             &state_f_u32x16, schedule_u32x16[2], round_constants[2]);
    sz_sha256_round_skylake_(state_f_u32x16, state_g_u32x16, state_h_u32x16, &state_a_u32x16, state_b_u32x16, state_c_u32x16, state_d_u32x16,
                             &state_e_u32x16, schedule_u32x16[3], round_constants[3]);
    sz_sha256_round_skylake_(state_e_u32x16, state_f_u32x16, state_g_u32x16, &state_h_u32x16, state_a_u32x16, state_b_u32x16, state_c_u32x16,
                             &state_d_u32x16, schedule_u32x16[4], round_constants[4]);
    sz_sha256_round_skylake_(state_d_u32x16, state_e_u32x16, state_f_u32x16, &state_g_u32x16, state_h_u32x16, state_a_u32x16, state_b_u32x16,
                             &state_c_u32x16, schedule_u32x16[5], round_constants[5]);
    sz_sha256_round_skylake_(state_c_u32x16, state_d_u32x16, state_e_u32x16, &state_f_u32x16, state_g_u32x16, state_h_u32x16, state_a_u32x16,
                             &state_b_u32x16, schedule_u32x16[6], round_constants[6]);
    sz_sha256_round_skylake_(state_b_u32x16, state_c_u32x16, state_d_u32x16, &state_e_u32x16, state_f_u32x16, state_g_u32x16, state_h_u32x16,
                             &state_a_u32x16, schedule_u32x16[7], round_constants[7]);
    sz_sha256_round_skylake_(state_a_u32x16, state_b_u32x16, state_c_u32x16, &state_d_u32x16, state_e_u32x16, state_f_u32x16, state_g_u32x16,
                             &state_h_u32x16, schedule_u32x16[8], round_constants[8]);
    sz_sha256_round_skylake_(state_h_u32x16, state_a_u32x16, state_b_u32x16, &state_c_u32x16, state_d_u32x16, state_e_u32x16, state_f_u32x16,
                             &state_g_u32x16, schedule_u32x16[9], round_constants[9]);
    sz_sha256_round_skylake_(state_g_u32x16, state_h_u32x16, state_a_u32x16, &state_b_u32x16, state_c_u32x16, state_d_u32x16, state_e_u32x16,
                             &state_f_u32x16, schedule_u32x16[10], round_constants[10]);
    sz_sha256_round_skylake_(state_f_u32x16, state_g_u32x16, state_h_u32x16, &state_a_u32x16, state_b_u32x16, state_c_u32x16, state_d_u32x16,
                             &state_e_u32x16, schedule_u32x16[11], round_constants[11]);
    sz_sha256_round_skylake_(state_e_u32x16, state_f_u32x16, state_g_u32x16, &state_h_u32x16, state_a_u32x16, state_b_u32x16, state_c_u32x16,
                             &state_d_u32x16, schedule_u32x16[12], round_constants[12]);
    sz_sha256_round_skylake_(state_d_u32x16, state_e_u32x16, state_f_u32x16, &state_g_u32x16, state_h_u32x16, state_a_u32x16, state_b_u32x16,
                             &state_c_u32x16, schedule_u32x16[13], round_constants[13]);
    sz_sha256_round_skylake_(state_c_u32x16, state_d_u32x16, state_e_u32x16, &state_f_u32x16, state_g_u32x16, state_h_u32x16, state_a_u32x16,
                             &state_b_u32x16, schedule_u32x16[14], round_constants[14]);
    sz_sha256_round_skylake_(state_b_u32x16, state_c_u32x16, state_d_u32x16, &state_e_u32x16, state_f_u32x16, state_g_u32x16, state_h_u32x16,
                             &state_a_u32x16, schedule_u32x16[15], round_constants[15]);

    for (sz_size_t turn_index = 1; turn_index != 4; ++turn_index) {
        sz_u32_t const *turn_constants = round_constants + turn_index * 16;
        schedule_u32x16[0] = sz_sha256_extend_skylake_(schedule_u32x16[0], schedule_u32x16[1], schedule_u32x16[9],
                                                    schedule_u32x16[14]);
        sz_sha256_round_skylake_(state_a_u32x16, state_b_u32x16, state_c_u32x16, &state_d_u32x16, state_e_u32x16, state_f_u32x16,
                                 state_g_u32x16, &state_h_u32x16, schedule_u32x16[0], turn_constants[0]);
        schedule_u32x16[1] = sz_sha256_extend_skylake_(schedule_u32x16[1], schedule_u32x16[2], schedule_u32x16[10],
                                                    schedule_u32x16[15]);
        sz_sha256_round_skylake_(state_h_u32x16, state_a_u32x16, state_b_u32x16, &state_c_u32x16, state_d_u32x16, state_e_u32x16,
                                 state_f_u32x16, &state_g_u32x16, schedule_u32x16[1], turn_constants[1]);
        schedule_u32x16[2] = sz_sha256_extend_skylake_(schedule_u32x16[2], schedule_u32x16[3], schedule_u32x16[11],
                                                    schedule_u32x16[0]);
        sz_sha256_round_skylake_(state_g_u32x16, state_h_u32x16, state_a_u32x16, &state_b_u32x16, state_c_u32x16, state_d_u32x16,
                                 state_e_u32x16, &state_f_u32x16, schedule_u32x16[2], turn_constants[2]);
        schedule_u32x16[3] = sz_sha256_extend_skylake_(schedule_u32x16[3], schedule_u32x16[4], schedule_u32x16[12],
                                                    schedule_u32x16[1]);
        sz_sha256_round_skylake_(state_f_u32x16, state_g_u32x16, state_h_u32x16, &state_a_u32x16, state_b_u32x16, state_c_u32x16,
                                 state_d_u32x16, &state_e_u32x16, schedule_u32x16[3], turn_constants[3]);
        schedule_u32x16[4] = sz_sha256_extend_skylake_(schedule_u32x16[4], schedule_u32x16[5], schedule_u32x16[13],
                                                    schedule_u32x16[2]);
        sz_sha256_round_skylake_(state_e_u32x16, state_f_u32x16, state_g_u32x16, &state_h_u32x16, state_a_u32x16, state_b_u32x16,
                                 state_c_u32x16, &state_d_u32x16, schedule_u32x16[4], turn_constants[4]);
        schedule_u32x16[5] = sz_sha256_extend_skylake_(schedule_u32x16[5], schedule_u32x16[6], schedule_u32x16[14],
                                                    schedule_u32x16[3]);
        sz_sha256_round_skylake_(state_d_u32x16, state_e_u32x16, state_f_u32x16, &state_g_u32x16, state_h_u32x16, state_a_u32x16,
                                 state_b_u32x16, &state_c_u32x16, schedule_u32x16[5], turn_constants[5]);
        schedule_u32x16[6] = sz_sha256_extend_skylake_(schedule_u32x16[6], schedule_u32x16[7], schedule_u32x16[15],
                                                    schedule_u32x16[4]);
        sz_sha256_round_skylake_(state_c_u32x16, state_d_u32x16, state_e_u32x16, &state_f_u32x16, state_g_u32x16, state_h_u32x16,
                                 state_a_u32x16, &state_b_u32x16, schedule_u32x16[6], turn_constants[6]);
        schedule_u32x16[7] = sz_sha256_extend_skylake_(schedule_u32x16[7], schedule_u32x16[8], schedule_u32x16[0], schedule_u32x16[5]);
        sz_sha256_round_skylake_(state_b_u32x16, state_c_u32x16, state_d_u32x16, &state_e_u32x16, state_f_u32x16, state_g_u32x16,
                                 state_h_u32x16, &state_a_u32x16, schedule_u32x16[7], turn_constants[7]);
        schedule_u32x16[8] = sz_sha256_extend_skylake_(schedule_u32x16[8], schedule_u32x16[9], schedule_u32x16[1], schedule_u32x16[6]);
        sz_sha256_round_skylake_(state_a_u32x16, state_b_u32x16, state_c_u32x16, &state_d_u32x16, state_e_u32x16, state_f_u32x16,
                                 state_g_u32x16, &state_h_u32x16, schedule_u32x16[8], turn_constants[8]);
        schedule_u32x16[9] = sz_sha256_extend_skylake_(schedule_u32x16[9], schedule_u32x16[10], schedule_u32x16[2],
                                                    schedule_u32x16[7]);
        sz_sha256_round_skylake_(state_h_u32x16, state_a_u32x16, state_b_u32x16, &state_c_u32x16, state_d_u32x16, state_e_u32x16,
                                 state_f_u32x16, &state_g_u32x16, schedule_u32x16[9], turn_constants[9]);
        schedule_u32x16[10] = sz_sha256_extend_skylake_(schedule_u32x16[10], schedule_u32x16[11], schedule_u32x16[3],
                                                     schedule_u32x16[8]);
        sz_sha256_round_skylake_(state_g_u32x16, state_h_u32x16, state_a_u32x16, &state_b_u32x16, state_c_u32x16, state_d_u32x16,
                                 state_e_u32x16, &state_f_u32x16, schedule_u32x16[10], turn_constants[10]);
        schedule_u32x16[11] = sz_sha256_extend_skylake_(schedule_u32x16[11], schedule_u32x16[12], schedule_u32x16[4],
                                                     schedule_u32x16[9]);
        sz_sha256_round_skylake_(state_f_u32x16, state_g_u32x16, state_h_u32x16, &state_a_u32x16, state_b_u32x16, state_c_u32x16,
                                 state_d_u32x16, &state_e_u32x16, schedule_u32x16[11], turn_constants[11]);
        schedule_u32x16[12] = sz_sha256_extend_skylake_(schedule_u32x16[12], schedule_u32x16[13], schedule_u32x16[5],
                                                     schedule_u32x16[10]);
        sz_sha256_round_skylake_(state_e_u32x16, state_f_u32x16, state_g_u32x16, &state_h_u32x16, state_a_u32x16, state_b_u32x16,
                                 state_c_u32x16, &state_d_u32x16, schedule_u32x16[12], turn_constants[12]);
        schedule_u32x16[13] = sz_sha256_extend_skylake_(schedule_u32x16[13], schedule_u32x16[14], schedule_u32x16[6],
                                                     schedule_u32x16[11]);
        sz_sha256_round_skylake_(state_d_u32x16, state_e_u32x16, state_f_u32x16, &state_g_u32x16, state_h_u32x16, state_a_u32x16,
                                 state_b_u32x16, &state_c_u32x16, schedule_u32x16[13], turn_constants[13]);
        schedule_u32x16[14] = sz_sha256_extend_skylake_(schedule_u32x16[14], schedule_u32x16[15], schedule_u32x16[7],
                                                     schedule_u32x16[12]);
        sz_sha256_round_skylake_(state_c_u32x16, state_d_u32x16, state_e_u32x16, &state_f_u32x16, state_g_u32x16, state_h_u32x16,
                                 state_a_u32x16, &state_b_u32x16, schedule_u32x16[14], turn_constants[14]);
        schedule_u32x16[15] = sz_sha256_extend_skylake_(schedule_u32x16[15], schedule_u32x16[0], schedule_u32x16[8],
                                                     schedule_u32x16[13]);
        sz_sha256_round_skylake_(state_b_u32x16, state_c_u32x16, state_d_u32x16, &state_e_u32x16, state_f_u32x16, state_g_u32x16,
                                 state_h_u32x16, &state_a_u32x16, schedule_u32x16[15], turn_constants[15]);
    }

    hashes_u32x16[0] = _mm512_mask_add_epi32(hashes_u32x16[0], active_m16, hashes_u32x16[0], state_a_u32x16);
    hashes_u32x16[1] = _mm512_mask_add_epi32(hashes_u32x16[1], active_m16, hashes_u32x16[1], state_b_u32x16);
    hashes_u32x16[2] = _mm512_mask_add_epi32(hashes_u32x16[2], active_m16, hashes_u32x16[2], state_c_u32x16);
    hashes_u32x16[3] = _mm512_mask_add_epi32(hashes_u32x16[3], active_m16, hashes_u32x16[3], state_d_u32x16);
    hashes_u32x16[4] = _mm512_mask_add_epi32(hashes_u32x16[4], active_m16, hashes_u32x16[4], state_e_u32x16);
    hashes_u32x16[5] = _mm512_mask_add_epi32(hashes_u32x16[5], active_m16, hashes_u32x16[5], state_f_u32x16);
    hashes_u32x16[6] = _mm512_mask_add_epi32(hashes_u32x16[6], active_m16, hashes_u32x16[6], state_g_u32x16);
    hashes_u32x16[7] = _mm512_mask_add_epi32(hashes_u32x16[7], active_m16, hashes_u32x16[7], state_h_u32x16);
}

/**
 *  @brief Compresses each lane's own run of whole blocks, plus an optional buffered block ahead of them.
 *  @param states The lane states, whose hash words are gathered on entry and scattered on exit.
 *  @param active_lanes_count Lanes owning a state, 1 to 16; the rest borrow lane zero and are discarded.
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
 *  Kept apart from the lane bookkeeping so the block loop gets the whole register file to itself. The hash
 *  words move by real gather and scatter rather than by 128 scalar loads through a stack union, whose
 *  store-to-load forwarding was the largest fixed cost of a call and the one short messages cannot amortize.
 */
SZ_HELPER_AUTO void sz_sha256_multistate_blocks_skylake_(sz_sha256_state_t *states, sz_size_t active_lanes_count,
                                                         sz_u32_t buffered_bitmask, sz_u8_t const **cursors,
                                                         sz_size_t const *blocks_per_lane) {
    __m512i hashes_u32x16[8];
    sz_u512_vec_t counts_vec;
    sz_u8_t const *sources[16];
    sz_size_t largest_blocks_count = 0;


    // The topped-up block and the fallback for a lane with nothing whole left are the same buffer, so one
    // source array serves both phases and the head costs no extra stack. An absent lane, or one with
    // nothing whole left, keeps reading that buffer and never advances, so the transpose stays in bounds
    // without a bounds test in the loop.
    for (sz_size_t lane_index = 0; lane_index != 16; ++lane_index) {
        sz_size_t const source_lane = lane_index < active_lanes_count ? lane_index : 0;
        sz_size_t const lane_blocks = lane_index < active_lanes_count ? blocks_per_lane[lane_index] : 0;
        counts_vec.u32s[lane_index] = (sz_u32_t)lane_blocks;
        sources[lane_index] = states[source_lane].block;
        if (lane_blocks > largest_blocks_count) largest_blocks_count = lane_blocks;
    }

    // Token-sized messages compress nothing here - their whole hash is the padding block the digest path
    // emits - so leaving before the gather keeps them off the word-major round trip entirely.
    if (buffered_bitmask == 0 && largest_blocks_count == 0) return;

    // A lane's eight hash words are 32 contiguous bytes, so one 256-bit load and an insert pack lanes
    // `pair_index` and `pair_index + 8` into a register, and the butterfly does the rest.
    for (sz_size_t pair_index = 0; pair_index != 8; ++pair_index) {
        sz_size_t const low_lane = pair_index < active_lanes_count ? pair_index : 0;
        sz_size_t const high_lane = pair_index + 8 < active_lanes_count ? pair_index + 8 : 0;
        hashes_u32x16[pair_index] = _mm512_inserti64x4(
            _mm512_castsi256_si512(_mm256_loadu_si256((__m256i const *)states[low_lane].hash)),
            _mm256_loadu_si256((__m256i const *)states[high_lane].hash), 1);
    }
    sz_sha256_transpose_8x16_skylake_(hashes_u32x16);

    if (buffered_bitmask) sz_sha256_compress_skylake_(hashes_u32x16, sources, (__mmask16)buffered_bitmask);
    for (sz_size_t lane_index = 0; lane_index != 16; ++lane_index)
        if (counts_vec.u32s[lane_index] != 0) sources[lane_index] = cursors[lane_index];

    __m512i counts_u32x16 = counts_vec.zmm;
    __m512i const ones_u32x16 = _mm512_set1_epi32(1);
    for (sz_size_t block_index = 0; block_index != largest_blocks_count; ++block_index) {
        __mmask16 const active_m16 = _mm512_cmpgt_epu32_mask(counts_u32x16, _mm512_setzero_si512());
        __mmask16 const advance_m16 = _mm512_cmpgt_epu32_mask(counts_u32x16, ones_u32x16);
        sz_sha256_compress_skylake_(hashes_u32x16, sources, active_m16);
        for (sz_size_t lane_index = 0; lane_index != 16; ++lane_index)
            sources[lane_index] += ((advance_m16 >> lane_index) & 1u) * SZ_SHA256_BLOCK_LENGTH;
        counts_u32x16 = _mm512_mask_sub_epi32(counts_u32x16, active_m16, counts_u32x16, ones_u32x16);
    }

    sz_sha256_transpose_8x16_skylake_(hashes_u32x16);
    for (sz_size_t pair_index = 0; pair_index != 8; ++pair_index) {
        if (pair_index < active_lanes_count)
            _mm256_storeu_si256((__m256i *)states[pair_index].hash,
                                _mm512_castsi512_si256(hashes_u32x16[pair_index]));
        if (pair_index + 8 < active_lanes_count)
            _mm256_storeu_si256((__m256i *)states[pair_index + 8].hash,
                                _mm512_extracti64x4_epi64(hashes_u32x16[pair_index], 1));
    }
    for (sz_size_t lane_index = 0; lane_index != active_lanes_count; ++lane_index)
        cursors[lane_index] += blocks_per_lane[lane_index] * SZ_SHA256_BLOCK_LENGTH;
}

SZ_API_COMPTIME void sz_sha256_multistate_update_skylake(sz_sha256_state_t *states, sz_sequence_t const *texts) {
    sz_size_t const lanes_count = texts->count;

    for (sz_size_t first_lane_index = 0; first_lane_index < lanes_count; first_lane_index += 16) {
        sz_size_t const lanes_left = lanes_count - first_lane_index;
        sz_size_t const active_lanes_count = lanes_left < 16 ? lanes_left : 16;
        sz_u8_t const *cursors[16];
        sz_size_t remaining[16], blocks_per_lane[16];
        sz_u32_t buffered_bitmask = 0;

        // Top up any buffered partial block into the state's own 64-byte buffer, which then serves as that
        // lane's first block source. Every lane's whole chunk is charged to `total_length` here, once, so
        // the head, body and tail below only move bytes.
        for (sz_size_t lane_index = 0; lane_index != active_lanes_count; ++lane_index) {
            sz_sha256_state_t *const state = &states[first_lane_index + lane_index];
            cursors[lane_index] = (sz_u8_t const *)texts->get_start(texts->handle, first_lane_index + lane_index);
            remaining[lane_index] = texts->get_length(texts->handle, first_lane_index + lane_index);

            // The countdown rides in 32-bit lanes, so a chunk longer than that many blocks goes through the
            // single-state kernel over the very same state and the lane sits the group out.
            if (remaining[lane_index] / SZ_SHA256_BLOCK_LENGTH > 0xFFFFFFFFull) {
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

        sz_sha256_multistate_blocks_skylake_(&states[first_lane_index], active_lanes_count, buffered_bitmask, cursors,
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
 *  @brief Finalizes 16 already-gathered lanes into their digests.
 *  @param states The 16 lane states, left untouched.
 *  @param active_lanes_count Lanes to emit, 1 to 16; the rest ride along and are discarded.
 *  @param digests Receives `active_lanes_count` 32-byte big-endian digests.
 *
 *  Padding is per lane but compression is not. Both candidate blocks are built for every lane, and a k-mask
 *  decides which lanes actually consumed the carrier block, so a lane that needed only one block keeps its
 *  state bit-for-bit. Inactive lanes borrow lane zero's hash so the gather stays in bounds.
 */
SZ_HELPER_AUTO void sz_sha256_multistate_digest_lanes_skylake_(sz_sha256_state_t const *states,
                                                               sz_size_t active_lanes_count, sz_u8_t *digests) {
    sz_u512_vec_t staged_vec[16];
    sz_u8_t const *staged_blocks[16];
    __m512i hashes_u32x16[8];
    __mmask16 overflow_m16 = 0;

    for (sz_size_t lane_index = 0; lane_index != 16; ++lane_index) {
        sz_size_t const source_lane = lane_index < active_lanes_count ? lane_index : 0;
        staged_blocks[lane_index] = staged_vec[lane_index].u8s;
        if (states[source_lane].block_length + 1 > SZ_SHA256_BLOCK_LENGTH - 8)
            overflow_m16 |= (__mmask16)((sz_u32_t)1 << lane_index);
    }

    // A lane's eight hash words are 32 contiguous bytes, so one 256-bit load and an insert pack lanes
    // `pair_index` and `pair_index + 8` into a register, and the butterfly does the rest.
    for (sz_size_t pair_index = 0; pair_index != 8; ++pair_index) {
        sz_size_t const low_lane = pair_index < active_lanes_count ? pair_index : 0;
        sz_size_t const high_lane = pair_index + 8 < active_lanes_count ? pair_index + 8 : 0;
        hashes_u32x16[pair_index] = _mm512_inserti64x4(
            _mm512_castsi256_si512(_mm256_loadu_si256((__m256i const *)states[low_lane].hash)),
            _mm256_loadu_si256((__m256i const *)states[high_lane].hash), 1);
    }
    sz_sha256_transpose_8x16_skylake_(hashes_u32x16);

    // The terminator lands in a carrier block whenever it would crowd out the trailing bit length. One
    // staging buffer serves both blocks: the carrier compression is masked to the lanes that need it and
    // has consumed the buffer before the final block overwrites it.
    if (overflow_m16) {
        for (sz_size_t lane_index = 0; lane_index != 16; ++lane_index) {
            sz_size_t const source_lane = lane_index < active_lanes_count ? lane_index : 0;
            sz_size_t const buffered = states[source_lane].block_length;
            __mmask64 const buffered_m64 = sz_u64_clamp_mask_until_(buffered);
            staged_vec[lane_index].zmm =
                _mm512_maskz_loadu_epi8(buffered_m64, (void const *)states[source_lane].block);
            staged_vec[lane_index].u8s[buffered] = 0x80;
        }
        sz_sha256_compress_skylake_(hashes_u32x16, staged_blocks, overflow_m16);
    }

    for (sz_size_t lane_index = 0; lane_index != 16; ++lane_index) {
        sz_size_t const source_lane = lane_index < active_lanes_count ? lane_index : 0;
        sz_size_t const buffered = states[source_lane].block_length;
        sz_u64_t const bit_length = states[source_lane].total_length * 8;
        // An overflowing lane has already spent its buffered bytes on the carrier block, so its final
        // block carries nothing but the trailing length.
        int const carried = (overflow_m16 >> lane_index) & 1;
        __mmask64 const buffered_m64 = carried ? (__mmask64)0 : sz_u64_clamp_mask_until_(buffered);
        staged_vec[lane_index].zmm = _mm512_maskz_loadu_epi8(buffered_m64, (void const *)states[source_lane].block);
        if (!carried) staged_vec[lane_index].u8s[buffered] = 0x80;
        staged_vec[lane_index].u64s[7] = sz_u64_bytes_reverse(bit_length);
    }
    sz_sha256_compress_skylake_(hashes_u32x16, staged_blocks, (__mmask16)0xFFFF);

    // Big-endian output is a byte reverse inside each word, and the same butterfly that gathered the state
    // scatters it back to one 32-byte digest per lane.
    __m512i const byte_swap_u8x64 = _mm512_set_epi8(                     //
        60, 61, 62, 63, 56, 57, 58, 59, 52, 53, 54, 55, 48, 49, 50, 51,  //
        44, 45, 46, 47, 40, 41, 42, 43, 36, 37, 38, 39, 32, 33, 34, 35,  //
        28, 29, 30, 31, 24, 25, 26, 27, 20, 21, 22, 23, 16, 17, 18, 19,  //
        12, 13, 14, 15, 8, 9, 10, 11, 4, 5, 6, 7, 0, 1, 2, 3);
    for (sz_size_t word_index = 0; word_index != 8; ++word_index)
        hashes_u32x16[word_index] = _mm512_shuffle_epi8(hashes_u32x16[word_index], byte_swap_u8x64);
    sz_sha256_transpose_8x16_skylake_(hashes_u32x16);

    // The butterfly leaves lanes paired as `pair_index` and `pair_index + 8`, but digests land adjacent, so
    // one more sub-lane exchange brings neighbours together and every lane leaves in a full-width store.
    __m512i const adjacent0_u32x16 = _mm512_shuffle_i64x2(hashes_u32x16[0], hashes_u32x16[1], 0x44);
    __m512i const adjacent1_u32x16 = _mm512_shuffle_i64x2(hashes_u32x16[2], hashes_u32x16[3], 0x44);
    __m512i const adjacent2_u32x16 = _mm512_shuffle_i64x2(hashes_u32x16[4], hashes_u32x16[5], 0x44);
    __m512i const adjacent3_u32x16 = _mm512_shuffle_i64x2(hashes_u32x16[6], hashes_u32x16[7], 0x44);
    __m512i const adjacent4_u32x16 = _mm512_shuffle_i64x2(hashes_u32x16[0], hashes_u32x16[1], 0xEE);
    __m512i const adjacent5_u32x16 = _mm512_shuffle_i64x2(hashes_u32x16[2], hashes_u32x16[3], 0xEE);
    __m512i const adjacent6_u32x16 = _mm512_shuffle_i64x2(hashes_u32x16[4], hashes_u32x16[5], 0xEE);
    __m512i const adjacent7_u32x16 = _mm512_shuffle_i64x2(hashes_u32x16[6], hashes_u32x16[7], 0xEE);
    hashes_u32x16[0] = adjacent0_u32x16, hashes_u32x16[1] = adjacent1_u32x16;
    hashes_u32x16[2] = adjacent2_u32x16, hashes_u32x16[3] = adjacent3_u32x16;
    hashes_u32x16[4] = adjacent4_u32x16, hashes_u32x16[5] = adjacent5_u32x16;
    hashes_u32x16[6] = adjacent6_u32x16, hashes_u32x16[7] = adjacent7_u32x16;

    for (sz_size_t pair_index = 0; pair_index != 8; ++pair_index) {
        __mmask8 const first_m8 = pair_index * 2 + 0 < active_lanes_count ? (__mmask8)0x0F : (__mmask8)0;
        __mmask8 const second_m8 = pair_index * 2 + 1 < active_lanes_count ? (__mmask8)0xF0 : (__mmask8)0;
        _mm512_mask_storeu_epi64(&digests[pair_index * 2 * SZ_SHA256_DIGEST_LENGTH], first_m8 | second_m8,
                                 hashes_u32x16[pair_index]);
    }
}

SZ_API_COMPTIME void sz_sha256_multistate_digest_skylake(sz_sha256_state_t const *states, sz_size_t states_count,
                                                         sz_u8_t *digests) {
    sz_size_t first_lane_index = 0;
    for (; first_lane_index < states_count; first_lane_index += 16) {
        sz_size_t const remaining = states_count - first_lane_index;
        sz_size_t const active_lanes_count = remaining < 16 ? remaining : 16;
        sz_sha256_multistate_digest_lanes_skylake_(&states[first_lane_index], active_lanes_count,
                                                   &digests[first_lane_index * SZ_SHA256_DIGEST_LENGTH]);
    }
}

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
#endif // SZ_USE_SKYLAKE

#ifdef __cplusplus
}
#endif

#endif // STRINGZILLA_HASH_SKYLAKE_H_
